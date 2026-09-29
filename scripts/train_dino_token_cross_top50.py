"""Train a strict-streaming DINO token cross-matcher for the first-stage top-50.

The network is deliberately small: the expensive DINO backbone remains frozen
and only cached 5x5 patch tokens are used.  A query is compared with each
candidate through bidirectional soft token alignment.  Training identities are
disjoint from validation identities; validation hyper-parameters are selected
on TUNE_SEEDS and reported once on CONFIRM_SEEDS.

No query-query information, camera labels, test labels or row-order features
are used.  At inference the model sees one query and its static gallery only.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.probe_multi_query_aggregation import (
    CONFIRM_SEEDS,
    TUNE_SEEDS,
    _camera_representative_protocol,
)
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_dino_patch_matcher import _fuse, _load
from scripts.train_streaming_top25_verifier import _normalize
from src.reranking import database_side_augmentation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP = 50
TRAIN_SEEDS = (1709, 2711, 3907)
QUERY_BATCH = 6
TOKEN_GRID = int(os.environ.get("DINO_TOKEN_GRID", "5"))
TOKEN_COUNT = TOKEN_GRID * TOKEN_GRID
LOSS_VARIANT = os.environ.get("DINO_TOKEN_LOSS", "listwise").lower()
OSNET_TRAIN_PATH = os.environ.get(
    "OSNET_TRAIN_PATH",
    "outputs/expert_fusion/cache/osnet_smoothap_train.npz",
)
OSNET_VAL_PATH = os.environ.get(
    "OSNET_ENSEMBLE_PATH",
    "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
)
EXPERIMENT_SUFFIX = os.environ.get("TOKEN_EXPERIMENT_SUFFIX", "").strip()


class DinoTokenCrossMatcher(nn.Module):
    """Lightweight learned cross-attention over frozen DINO patch tokens."""

    def __init__(self, input_dim: int = 768, dim: int = 48,
                 token_count: int = 25):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.position = nn.Parameter(torch.zeros(1, 1, token_count, dim))
        nn.init.trunc_normal_(self.position, std=0.01)
        feature_dim = 8 * dim + 9
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Start exactly at the first-stage retrieval score.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, query, gallery, base):
        # query: [B, 25, 768], gallery: [B, K, 25, 768]
        q = self.projection(query)[:, None]
        g = self.projection(gallery)
        q = F.normalize(q + self.position, dim=-1)
        g = F.normalize(g + self.position, dim=-1)
        correlation = torch.einsum("bqtd,bksd->bkts", q, g)
        q_aligned = torch.einsum(
            "bkts,bksd->bktd", F.softmax(10.0 * correlation, dim=3), g
        )
        g_aligned = torch.einsum(
            "bkst,bqtd->bksd", F.softmax(10.0 * correlation.transpose(2, 3), dim=3), q
        )
        q_expanded = q.expand(-1, gallery.shape[1], -1, -1)

        q_abs, q_mul = torch.abs(q_expanded - q_aligned), q_expanded * q_aligned
        g_abs, g_mul = torch.abs(g - g_aligned), g * g_aligned
        visual = torch.cat(
            (
                q_abs.mean(2), q_abs.amax(2), q_mul.mean(2), q_mul.amax(2),
                g_abs.mean(2), g_abs.amax(2), g_mul.mean(2), g_mul.amax(2),
            ),
            dim=2,
        )
        flat = correlation.flatten(2)
        row_best = correlation.amax(3)
        col_best = correlation.amax(2)
        diagonal = correlation.diagonal(dim1=2, dim2=3)
        scalar = torch.stack(
            (
                base,
                flat.mean(2),
                flat.amax(2),
                flat.topk(16, dim=2).values.mean(2),
                row_best.mean(2),
                row_best.amin(2),
                col_best.mean(2),
                col_best.amin(2),
                diagonal.mean(2),
            ),
            dim=2,
        )
        residual = 0.30 * torch.tanh(self.head(torch.cat((visual, scalar), dim=2)).squeeze(2))
        return base + residual


def _base_protocol(frame, embedding, seed):
    qi, gi = _camera_representative_protocol(frame, seed)
    q_vehicle = frame.vehicle_id.to_numpy()[qi]
    q_camera = frame.camera_id.to_numpy()[qi]
    g_vehicle = frame.vehicle_id.to_numpy()[gi]
    g_camera = frame.camera_id.to_numpy()[gi]
    gallery = database_side_augmentation(embedding[gi], top_k=5, alpha=2.0)
    score = embedding[qi] @ gallery.T
    candidates = np.argsort(-score, axis=1, kind="stable")[:, :TOP]
    positives, usable = [], []
    for row in range(len(qi)):
        positive = np.flatnonzero(
            (g_vehicle == q_vehicle[row]) & (g_camera != q_camera[row])
        ).astype(np.int64)
        positives.append(positive)
        if len(positive):
            usable.append(row)
    return {
        "qi": qi,
        "gi": gi,
        "score": score.astype(np.float32),
        "candidates": candidates,
        "positives": positives,
        "usable": np.asarray(usable, dtype=np.int64),
    }


def _training_candidates(protocol, rows, rng):
    candidate = protocol["candidates"][rows].copy()
    for out_row, protocol_row in enumerate(rows):
        positive = protocol["positives"][int(protocol_row)]
        present = np.isin(positive, candidate[out_row])
        missing = positive[~present]
        # Expose a small number of missed hard positives without turning the
        # episode into an artificial all-positive shortlist.
        if len(missing):
            inject = rng.choice(missing, min(2, len(missing)), replace=False)
            labels = np.isin(candidate[out_row], positive)
            replace = np.flatnonzero(~labels)[-len(inject):]
            candidate[out_row, replace] = inject
    label = np.stack([
        np.isin(value, protocol["positives"][int(row)])
        for row, value in zip(rows, candidate, strict=True)
    ])
    return candidate, label


def _batch(tokens, embedding, protocol, rows, candidate):
    query_index = protocol["qi"][rows]
    gallery_index = protocol["gi"][candidate]
    base = np.take_along_axis(protocol["score"][rows], candidate, axis=1)
    return (
        torch.from_numpy(tokens[query_index]).to(DEVICE, non_blocking=True),
        torch.from_numpy(tokens[gallery_index]).to(DEVICE, non_blocking=True),
        torch.from_numpy(base).to(DEVICE, non_blocking=True),
    )


def _loss(score, label):
    label = label.bool()
    scaled = score / 0.09
    positive_lse = torch.logsumexp(scaled.masked_fill(~label, -torch.inf), dim=1)
    listwise = (torch.logsumexp(scaled, dim=1) - positive_lse).mean()
    pairwise = []
    smooth_aps = []
    top1_terms = []
    for row in range(len(score)):
        positive, negative = score[row, label[row]], score[row, ~label[row]]
        hard_negative = negative.topk(min(12, len(negative))).values
        pairwise.append(
            F.softplus(
                (hard_negative[:, None] - positive[None, :] + 0.025) / 0.07
            ).mean()
        )
        # Directly optimize the ordering that the official AP@10 rewards:
        # every positive should precede the hard negatives, not merely one
        # representative positive as in positive log-sum-exp.
        differences = (score[row][None, :] - positive[:, None]) / 0.06
        rank_all = 0.5 + torch.sigmoid(differences).sum(1)
        positive_differences = (positive[None, :] - positive[:, None]) / 0.06
        rank_positive = 0.5 + torch.sigmoid(positive_differences).sum(1)
        smooth_aps.append((rank_positive / rank_all).mean())
        top1_terms.append(F.softplus(
            (hard_negative[0] - positive.max() + 0.02) / 0.06
        ))
    binary = F.binary_cross_entropy_with_logits(
        (score - score.mean(1, keepdim=True)) / 0.08,
        label.float(),
        pos_weight=torch.tensor(5.0, device=score.device),
    )
    pairwise_loss = torch.stack(pairwise).mean()
    if LOSS_VARIANT == "smoothap":
        return (
            0.25 * listwise
            + 1.25 * (1.0 - torch.stack(smooth_aps).mean())
            + 0.45 * pairwise_loss
            + 0.20 * torch.stack(top1_terms).mean()
            + 0.05 * binary
        )
    return listwise + 0.35 * pairwise_loss + 0.10 * binary


def _train_epoch(model, optimizer, protocols, tokens, embedding, seed):
    model.train()
    rng = np.random.default_rng(seed)
    losses = []
    for protocol in protocols:
        rows = protocol["usable"].copy()
        rng.shuffle(rows)
        rows = rows[:min(1400, len(rows))]
        for start in range(0, len(rows), QUERY_BATCH):
            current = rows[start:start + QUERY_BATCH]
            candidate, label = _training_candidates(protocol, current, rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=DEVICE.type,
                enabled=DEVICE.type == "cuda",
                dtype=torch.float16,
            ):
                score = model(*_batch(tokens, embedding, protocol, current, candidate))
                loss = _loss(score, torch.from_numpy(label).to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def _predict(model, protocol, tokens, embedding, query_batch=12):
    model.eval()
    result = []
    all_rows = np.arange(len(protocol["qi"]))
    for start in range(0, len(all_rows), query_batch):
        rows = all_rows[start:start + query_batch]
        candidate = protocol["candidates"][rows]
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            value = model(*_batch(tokens, embedding, protocol, rows, candidate))
        result.append(value.float().cpu().numpy())
    return np.concatenate(result)


def _evaluate(official_protocols, base_protocols, predictions, weight):
    rows = []
    for official, data, prediction in zip(
        official_protocols, base_protocols, predictions, strict=True
    ):
        score = data["score"].copy()
        candidate = data["candidates"]
        base = np.take_along_axis(score, candidate, axis=1)
        # Prediction is a calibrated residual around base; keep the original
        # first-stage geometry when validation chooses a conservative weight.
        reranked = base + weight * (prediction - base)
        np.put_along_axis(score, candidate, reranked, axis=1)
        rows.append(_official_from_scores(score, official["q"], official["g"]))
    return _aggregate(rows)


def _load_features(frame, split):
    if split == "train":
        osnet = _normalize(_load(
            OSNET_TRAIN_PATH, frame, "embeddings"
        ))
        dino = _normalize(_load(
            "outputs/expert_fusion/cache/dinov2_vehicle_cls_train.npz", frame, "embeddings"
        ))
        tokens = _load(
            f"outputs/expert_fusion/cache/dinov2_vehicle_patches_train"
            f"{'_10x10' if TOKEN_GRID == 10 else ''}.npz", frame, "tokens"
        )
    else:
        osnet = _normalize(np.load(
            OSNET_VAL_PATH,
            allow_pickle=False,
        ))
        dino = _normalize(_load(
            "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz", frame, "cls"
        ))
        tokens = _load(
            f"outputs/expert_fusion/cache/dinov2_vehicle_patches_val"
            f"{'_10x10' if TOKEN_GRID == 10 else ''}.npz", frame, "tokens"
        )
    return _fuse(osnet, dino), tokens


def main():
    torch.manual_seed(92311)
    np.random.seed(92311)
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_embedding, train_tokens = _load_features(train, "train")
    val_embedding, val_tokens = _load_features(val, "val")

    train_protocols = [
        _base_protocol(train, train_embedding, seed) for seed in TRAIN_SEEDS
    ]
    tune_official = _protocols(val, TUNE_SEEDS, {"base": val_embedding})
    confirm_official = _protocols(val, CONFIRM_SEEDS, {"base": val_embedding})
    tune_protocols = [
        _base_protocol(val, val_embedding, seed) for seed in TUNE_SEEDS
    ]
    confirm_protocols = [
        _base_protocol(val, val_embedding, seed) for seed in CONFIRM_SEEDS
    ]

    model = DinoTokenCrossMatcher(token_count=TOKEN_COUNT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=4e-3)
    history, snapshots = [], []
    for epoch in range(1, 5):
        loss = _train_epoch(
            model, optimizer, train_protocols, train_tokens, train_embedding,
            92311 + epoch,
        )
        prediction = [_predict(model, tune_protocols[0], val_tokens, val_embedding)]
        metrics = _evaluate(tune_official[:1], tune_protocols[:1], prediction, 0.50)
        row = {"epoch": epoch, "loss": loss, **metrics}
        history.append(row)
        snapshots.append(copy.deepcopy(model.state_dict()))
        print(json.dumps(row), flush=True)

    selected_epoch = max(
        range(len(history)),
        key=lambda index: (history[index]["mAP@10"], history[index]["Rank-1"]),
    )
    model.load_state_dict(snapshots[selected_epoch])
    tune_prediction = [
        _predict(model, value, val_tokens, val_embedding) for value in tune_protocols
    ]
    confirm_prediction = [
        _predict(model, value, val_tokens, val_embedding) for value in confirm_protocols
    ]
    grid = [
        {"token_weight": weight, **_evaluate(
            tune_official, tune_protocols, tune_prediction, weight
        )}
        for weight in (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0)
    ]
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    confirmation = _evaluate(
        confirm_official, confirm_protocols, confirm_prediction, selected["token_weight"]
    )
    baseline = _evaluate(confirm_official, confirm_protocols, confirm_prediction, 0.0)
    report = {
        "design": "strict-streaming top-50 bidirectional DINO token cross-matcher; train IDs only",
        "top_k": TOP,
        "history": history,
        "selected_epoch": selected_epoch + 1,
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "grid": grid,
    }
    Path("outputs/retrieval_v2").mkdir(parents=True, exist_ok=True)
    suffix = "_10x10" if TOKEN_GRID == 10 else ""
    if LOSS_VARIANT == "smoothap":
        suffix += "_smoothap"
    if EXPERIMENT_SUFFIX:
        suffix += (
            EXPERIMENT_SUFFIX
            if EXPERIMENT_SUFFIX.startswith("_")
            else f"_{EXPERIMENT_SUFFIX}"
        )
    Path(f"outputs/retrieval_v2/dino_token_cross_top50{suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {"model_state": model.state_dict(), "report": report},
        f"weights/dino_token_cross_top50{suffix}.pt",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
