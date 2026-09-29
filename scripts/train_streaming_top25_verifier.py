"""Train/evaluate a strictly streaming top-25 local vehicle verifier.

The model sees one current query and candidates from a static gallery.  No
query-query feature, cluster, score, or prediction is constructed.  Training
uses train identities only; validation follows the official mAP@10 protocol.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.probe_gallery_groups_official import _groups
from scripts.probe_multi_query_aggregation import (
    CONFIRM_SEEDS,
    TUNE_SEEDS,
    _camera_representative_protocol,
)
from scripts.probe_streaming_official import _aggregate, _official, _protocols
from scripts.train_contextual_multiquery_reranker import DEVICE
from scripts.train_cross_image_transformer import CrossImageTransformer, _load
from src.reranking import database_side_augmentation


TOP = 25
TRAIN_SEEDS = (1709, 2711, 3907)
QUERY_BATCH = 8


def _normalize(values):
    values = values.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _make_train_protocol(frame, embedding, seed):
    qi, gi = _camera_representative_protocol(frame, seed)
    q_vehicle = frame.vehicle_id.to_numpy()[qi]
    q_camera = frame.camera_id.to_numpy()[qi]
    g_vehicle = frame.vehicle_id.to_numpy()[gi]
    g_camera = frame.camera_id.to_numpy()[gi]
    score = embedding[qi] @ embedding[gi].T
    positives, negatives = [], []
    for row in range(len(qi)):
        positive = np.flatnonzero(
            (g_vehicle == q_vehicle[row]) & (g_camera != q_camera[row])
        )
        junk = (g_vehicle == q_vehicle[row]) & (g_camera == q_camera[row])
        order_score = score[row].copy()
        order_score[junk | (g_vehicle == q_vehicle[row])] = -np.inf
        negative = np.argsort(-order_score, kind="stable")[:80]
        positives.append(positive.astype(np.int64))
        negatives.append(negative.astype(np.int64))
    usable = np.asarray([index for index, value in enumerate(positives) if len(value)])
    return {
        "qi": qi,
        "gi": gi,
        "positives": positives,
        "negatives": negatives,
        "usable": usable,
    }


def _train_candidates(protocol, rows, rng):
    candidates, labels = [], []
    for row in rows:
        positive = protocol["positives"][int(row)]
        # Keep every cross-camera positive up to a generous cap so that the
        # loss rewards placing all images of the vehicle before hard negatives.
        if len(positive) > 6:
            positive = rng.choice(positive, 6, replace=False)
        negative_count = TOP - len(positive)
        negative = protocol["negatives"][int(row)][:max(negative_count * 2, negative_count)]
        negative = rng.choice(negative, negative_count, replace=False)
        candidate = np.concatenate((positive, negative))
        label = np.concatenate((np.ones(len(positive), dtype=bool),
                                np.zeros(len(negative), dtype=bool)))
        permutation = rng.permutation(TOP)
        candidates.append(candidate[permutation])
        labels.append(label[permutation])
    return np.stack(candidates), np.stack(labels)


def _pair_batch(conv, osnet, embedding, query, gallery):
    similarity = np.sum(embedding[query] * embedding[gallery], axis=1).astype(np.float32)
    return (
        torch.from_numpy(conv[query]).to(DEVICE, non_blocking=True),
        torch.from_numpy(osnet[query]).to(DEVICE, non_blocking=True),
        torch.from_numpy(conv[gallery]).to(DEVICE, non_blocking=True),
        torch.from_numpy(osnet[gallery]).to(DEVICE, non_blocking=True),
        torch.from_numpy(similarity).to(DEVICE, non_blocking=True),
    )


def _multi_positive_loss(score, label):
    scaled = score / 0.10
    positive_lse = torch.logsumexp(scaled.masked_fill(~label, -torch.inf), dim=1)
    listwise = (torch.logsumexp(scaled, dim=1) - positive_lse).mean()
    pair_terms = []
    for row in range(len(score)):
        positive = score[row, label[row]]
        negative = score[row, ~label[row]]
        pair_terms.append(F.softplus((negative[:, None] - positive[None, :] + 0.03) / 0.08).mean())
    return listwise + 0.40 * torch.stack(pair_terms).mean()


def _train_epoch(model, optimizer, protocols, conv, osnet, embedding, seed):
    model.train()
    rng = np.random.default_rng(seed)
    losses = []
    for protocol in protocols:
        rows = protocol["usable"].copy()
        rng.shuffle(rows)
        # Three protocols already expose different representatives.  Capping
        # each keeps an epoch short without collapsing hard-negative diversity.
        rows = rows[:min(2600, len(rows))]
        for start in range(0, len(rows), QUERY_BATCH):
            current = rows[start:start + QUERY_BATCH]
            candidate, label = _train_candidates(protocol, current, rng)
            query = np.repeat(protocol["qi"][current], TOP)
            gallery = protocol["gi"][candidate.reshape(-1)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=DEVICE.type,
                enabled=DEVICE.type == "cuda",
                dtype=torch.float16,
            ):
                score = model(*_pair_batch(conv, osnet, embedding, query, gallery))
                score = score.reshape(len(current), TOP)
                loss = _multi_positive_loss(
                    score, torch.from_numpy(label).to(DEVICE)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def _pair_scores(model, conv, osnet, embedding, query, gallery, batch=512):
    model.eval()
    result = []
    for start in range(0, len(query), batch):
        q = query[start:start + batch]
        g = gallery[start:start + batch]
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            value = model(*_pair_batch(conv, osnet, embedding, q, g))
        result.append(value.float().cpu().numpy())
    return np.concatenate(result)


def _prepared(protocol, embedding):
    query = embedding[protocol["qi"]]
    raw_gallery = embedding[protocol["gi"]]
    gallery = database_side_augmentation(raw_gallery, top_k=5, alpha=2.0)
    base = query @ gallery.T
    candidates = np.argsort(-base, axis=1, kind="stable")[:, :TOP]
    similarity = gallery @ gallery.T
    groups = _groups(similarity, neighbor_k=1, threshold=0.50, complete_link=True)
    return {
        "base": base,
        "candidates": candidates,
        "groups": groups,
        "qi": protocol["qi"],
        "gi": protocol["gi"],
    }


def _local_for_protocol(model, protocol, prepared, conv, osnet, embedding):
    candidates = prepared["candidates"]
    query = np.repeat(prepared["qi"], TOP)
    gallery = prepared["gi"][candidates.reshape(-1)]
    return _pair_scores(model, conv, osnet, embedding, query, gallery).reshape(len(candidates), TOP)


def _order(prepared, local, verifier_weight, group_weight):
    score = prepared["base"].copy()
    candidates = prepared["candidates"]
    base_local = np.take_along_axis(score, candidates, axis=1)
    # The verifier contains the original cosine plus a bounded learned
    # residual.  Blend only that residual to preserve calibrated base geometry.
    reranked = base_local + verifier_weight * (local - base_local)
    np.put_along_axis(score, candidates, reranked, axis=1)
    for group in prepared["groups"]:
        family = score[:, group].max(axis=1, keepdims=True)
        score[:, group] = (1.0 - group_weight) * score[:, group] + group_weight * family
    return np.argsort(-score, axis=1, kind="stable")


def _evaluate(protocols, prepared, local, verifier_weight, group_weight):
    rows = []
    for protocol, data, value in zip(protocols, prepared, local, strict=True):
        order = _order(data, value, verifier_weight, group_weight)
        rows.append(_official(order, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    torch.manual_seed(92021)
    np.random.seed(92021)
    train = pd.read_csv("splits/train.csv")
    val = pd.read_csv("splits/val.csv")

    train_embedding = _normalize(_load(
        "outputs/expert_fusion/cache/osnet_smoothap_train.npz", train, "embeddings"
    ))
    val_embedding = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
        allow_pickle=False,
    ))
    train_conv = _load(
        "outputs/expert_fusion/cache/convnext_train_parts_3x3.npz", train, "parts"
    )
    train_osnet = _load(
        "outputs/expert_fusion/cache/osnet_smoothap_train_parts4.npz", train, "parts"
    )
    val_conv = _load(
        "outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", val, "parts"
    )
    val_osnet = _load(
        "outputs/expert_fusion/cache/osnet_smoothap_val_parts4.npz", val, "parts"
    )

    train_protocols = [
        _make_train_protocol(train, train_embedding, seed) for seed in TRAIN_SEEDS
    ]
    tune_protocols = _protocols(val, TUNE_SEEDS, {"base": val_embedding})
    confirm_protocols = _protocols(val, CONFIRM_SEEDS, {"base": val_embedding})
    tune_prepared = [_prepared(protocol, val_embedding) for protocol in tune_protocols]
    confirm_prepared = [_prepared(protocol, val_embedding) for protocol in confirm_protocols]

    model = CrossImageTransformer(dim=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-3)
    history, snapshots = [], []
    for epoch in range(1, 6):
        loss = _train_epoch(
            model, optimizer, train_protocols, train_conv, train_osnet,
            train_embedding, 92021 + epoch,
        )
        local = [_local_for_protocol(
            model, tune_protocols[0], tune_prepared[0], val_conv, val_osnet,
            val_embedding,
        )]
        metrics = _evaluate(
            tune_protocols[:1], tune_prepared[:1], local,
            verifier_weight=0.50, group_weight=0.25,
        )
        row = {"epoch": epoch, "loss": loss, **metrics}
        history.append(row)
        snapshots.append(copy.deepcopy(model.state_dict()))
        print(json.dumps(row), flush=True)

    selected_epoch = max(
        range(len(history)),
        key=lambda index: (history[index]["mAP@10"], history[index]["Rank-1"]),
    )
    model.load_state_dict(snapshots[selected_epoch])
    tune_local = [
        _local_for_protocol(model, p, d, val_conv, val_osnet, val_embedding)
        for p, d in zip(tune_protocols, tune_prepared, strict=True)
    ]
    confirm_local = [
        _local_for_protocol(model, p, d, val_conv, val_osnet, val_embedding)
        for p, d in zip(confirm_protocols, confirm_prepared, strict=True)
    ]

    grid = []
    for verifier_weight in (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        for group_weight in (0.0, 0.10, 0.25, 0.40):
            metrics = _evaluate(
                tune_protocols, tune_prepared, tune_local,
                verifier_weight, group_weight,
            )
            grid.append({
                "verifier_weight": verifier_weight,
                "group_weight": group_weight,
                **metrics,
            })
    selected = max(
        grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"])
    )
    confirmation = _evaluate(
        confirm_protocols, confirm_prepared, confirm_local,
        selected["verifier_weight"], selected["group_weight"],
    )
    baseline = _evaluate(
        confirm_protocols, confirm_prepared, confirm_local, 0.0, 0.25,
    )
    raw = _evaluate(
        confirm_protocols, confirm_prepared, confirm_local, 0.0, 0.0,
    )
    report = {
        "design": (
            "strict streaming top-25 multi-positive cross-image verifier; "
            "one query plus static gallery only"
        ),
        "top_k": TOP,
        "history": history,
        "selected_epoch": selected_epoch + 1,
        "selected_tune": selected,
        "raw_dba_confirmation": raw,
        "gallery_group_baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta_vs_gallery_group": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "top_grid": sorted(
            grid,
            key=lambda row: (row["mAP@10"], row["Rank-1"]),
            reverse=True,
        )[:10],
    }
    output = Path("outputs/expert_fusion/streaming_top25_verifier.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {"model_state": model.state_dict(), "report": report},
        "weights/streaming_top25_verifier.pt",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
