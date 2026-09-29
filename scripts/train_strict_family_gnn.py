"""Train a strict query-conditioned gallery-family graph reranker.

Nodes are the top-50 candidates of one current query.  Edges are constructed
only from the static gallery (global similarities and a frozen token matcher).
The GNN learns to promote coherent same-vehicle families and suppress isolated
look-alike candidates.  No other query, camera feature, row order or validation
identity is used for training.
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
from scripts.probe_token_gallery_linker import _gallery_protocol
from scripts.train_dino_patch_matcher import _load
from scripts.train_dino_token_cross_top50 import (
    DEVICE,
    TOP,
    DinoTokenCrossMatcher,
    _fuse,
    _normalize,
    _predict as _predict_token,
)
from src.reranking import database_side_augmentation


TRAIN_SEEDS = (1709, 2711, 3907)
QUERY_BATCH = 16
LOSS_VARIANT = os.environ.get("FAMILY_GNN_VARIANT", "listwise").lower()
OSNET_TRAIN_PATH = os.environ.get(
    "OSNET_TRAIN_PATH",
    "outputs/expert_fusion/cache/osnet_smoothap_train.npz",
)
TOKEN_MODEL_PATH = os.environ.get(
    "DINO_TOKEN_MODEL", "weights/dino_token_cross_top50.pt"
)
EXPERIMENT_SUFFIX = os.environ.get("FAMILY_EXPERIMENT_SUFFIX", "").strip()


class FamilyGraphReranker(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 80, layers: int = 3):
        super().__init__()
        self.node = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden),
        )
        self.edge = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.updates = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4 * hidden),
                nn.Linear(4 * hidden, 2 * hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(2 * hidden, hidden),
            )
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + feature_dim),
            nn.Linear(hidden + feature_dim, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, relation):
        # relation channels: fused, OSNet, symmetric token score, token-valid.
        batch, count = features.shape[:2]
        h = self.node(features)
        fused = relation[..., 0]
        eye = torch.eye(count, dtype=torch.bool, device=features.device)[None]
        search = fused.masked_fill(eye, -torch.inf)
        neighbors = search.topk(min(8, count - 1), dim=2).indices
        mask = torch.zeros_like(fused, dtype=torch.bool)
        mask.scatter_(2, neighbors, True)
        mask |= relation[..., 3] > 0.5
        mask |= eye
        edge = self.edge(relation).squeeze(3).masked_fill(~mask, -torch.inf)
        attention = F.softmax(edge, dim=2)
        for update, norm in zip(self.updates, self.norms, strict=True):
            message = torch.bmm(attention, h)
            delta = update(torch.cat((h, message, torch.abs(h - message), h * message), dim=2))
            h = norm(h + delta)
        residual = 0.35 * torch.tanh(
            self.head(torch.cat((h, features), dim=2)).squeeze(2)
        )
        return features[..., 0] + residual


def _load_features(frame, split):
    if split == "train":
        osnet = _normalize(_load(
            OSNET_TRAIN_PATH, frame, "embeddings"
        ))
        dino = _normalize(_load(
            "outputs/expert_fusion/cache/dinov2_vehicle_cls_train.npz", frame, "embeddings"
        ))
        tokens = _load(
            "outputs/expert_fusion/cache/dinov2_vehicle_patches_train.npz", frame, "tokens"
        )
    else:
        osnet = _normalize(np.load(
            os.environ.get(
                "OSNET_ENSEMBLE_PATH",
                "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
            ),
            allow_pickle=False,
        ))
        dino = _normalize(_load(
            os.environ.get(
                "DINO_PARTS_VAL_PATH",
                "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz",
            ),
            frame,
            "cls",
        ))
        tokens = _load(
            "outputs/expert_fusion/cache/dinov2_vehicle_patches_val.npz", frame, "tokens"
        )
    return _fuse(osnet, dino), osnet, dino, tokens


def _symmetric_token_matrix(model, indices, tokens, fused):
    data = _gallery_protocol(indices, fused)
    prediction = _predict_token(model, data, tokens, fused)
    n = len(indices)
    directed = np.full((n, n), -np.inf, dtype=np.float32)
    np.put_along_axis(directed, data["candidates"], prediction, axis=1)
    mutual = np.isfinite(directed) & np.isfinite(directed.T)
    score = np.where(mutual, 0.5 * (directed + directed.T), -1.0).astype(np.float32)
    np.fill_diagonal(score, 1.0)
    base_order = np.argsort(-data["score"], axis=1, kind="stable")
    base_rank = np.empty_like(base_order, dtype=np.int16)
    np.put_along_axis(
        base_rank,
        base_order,
        np.broadcast_to(np.arange(n, dtype=np.int16), base_order.shape),
        axis=1,
    )
    return score, mutual.astype(np.float32), base_rank


def _gather_relation(candidate, fused_similarity, osnet_similarity,
                     token_similarity, token_valid, batch=128):
    output = []
    for start in range(0, len(candidate), batch):
        index = candidate[start:start + batch]
        left, right = index[:, :, None], index[:, None, :]
        relation = np.stack(
            (
                fused_similarity[left, right],
                osnet_similarity[left, right],
                token_similarity[left, right],
                token_valid[left, right],
            ),
            axis=3,
        )
        output.append(relation.astype(np.float16))
    return np.concatenate(output)


@torch.inference_mode()
def _episode(frame, fused, osnet, dino, tokens, token_model, seed):
    qi, gi = _camera_representative_protocol(frame, seed)
    gallery = database_side_augmentation(fused[gi], top_k=5, alpha=2.0)
    score = (fused[qi] @ gallery.T).astype(np.float32)
    candidate = np.argsort(-score, axis=1, kind="stable")[:, :TOP]
    token_data = {"qi": qi, "gi": gi, "score": score, "candidates": candidate}
    token = _predict_token(token_model, token_data, tokens, fused)

    rows = np.arange(len(qi))[:, None]
    raw_fused = fused[qi] @ fused[gi].T
    raw_osnet = osnet[qi] @ osnet[gi].T
    raw_dino = dino[qi] @ dino[gi].T
    base = score[rows, candidate]
    raw = np.stack(
        (
            base,
            token,
            raw_fused[rows, candidate],
            raw_osnet[rows, candidate],
            raw_dino[rows, candidate],
        ),
        axis=2,
    ).astype(np.float32)
    z = (raw - raw.mean(1, keepdims=True)) / (raw.std(1, keepdims=True) + 1e-6)
    rank = np.broadcast_to(
        np.linspace(0.0, 1.0, TOP, dtype=np.float32)[None, :, None],
        (len(qi), TOP, 1),
    )
    gap = raw[..., :2] - raw[:, :1, :2]
    features = np.concatenate((raw, z, rank, gap), axis=2).astype(np.float32)

    fused_g, osnet_g = fused[gi], osnet[gi]
    fused_similarity = fused_g @ fused_g.T
    osnet_similarity = osnet_g @ osnet_g.T
    token_similarity, token_valid, gallery_rank = _symmetric_token_matrix(
        token_model, gi, tokens, fused
    )
    relation = _gather_relation(
        candidate, fused_similarity, osnet_similarity, token_similarity, token_valid
    )

    q_vehicle = frame.vehicle_id.to_numpy()[qi]
    q_camera = frame.camera_id.to_numpy()[qi]
    g_vehicle = frame.vehicle_id.to_numpy()[gi]
    g_camera = frame.camera_id.to_numpy()[gi]
    label = g_vehicle[candidate] == q_vehicle[:, None]
    junk = label & (g_camera[candidate] == q_camera[:, None])
    label &= ~junk
    valid = ~junk
    usable = np.flatnonzero(label.any(1)).astype(np.int64)
    return {
        "qi": qi,
        "gi": gi,
        "score": score,
        "candidate": candidate,
        "features": features,
        "relation": relation,
        "label": label,
        "valid": valid,
        "usable": usable,
        "gallery_token_similarity": token_similarity,
        "gallery_token_valid": token_valid,
        "gallery_rank": gallery_rank,
    }


def _batch(episode, rows):
    return (
        torch.from_numpy(episode["features"][rows]).to(DEVICE, non_blocking=True),
        torch.from_numpy(episode["relation"][rows].astype(np.float32)).to(
            DEVICE, non_blocking=True
        ),
    )


def _loss(score, label, valid):
    label, valid = label.bool(), valid.bool()
    scaled = score / 0.08
    all_lse = torch.logsumexp(scaled.masked_fill(~valid, -torch.inf), dim=1)
    positive_lse = torch.logsumexp(scaled.masked_fill(~label, -torch.inf), dim=1)
    listwise = (all_lse - positive_lse).mean()
    pairwise = []
    smooth_aps = []
    top1_terms = []
    for row in range(len(score)):
        positive = score[row, label[row]]
        negative = score[row, valid[row] & ~label[row]]
        hard_negative = negative.topk(min(10, len(negative))).values
        pairwise.append(F.softplus(
            (hard_negative[:, None] - positive[None, :] + 0.025) / 0.07
        ).mean())
        # Differentiable AP: every positive is rewarded for preceding every
        # hard negative, rather than allowing one easy positive to explain the
        # whole query through positive log-sum-exp.
        values = score[row, valid[row]]
        differences = (values[None, :] - positive[:, None]) / 0.06
        rank_all = 0.5 + torch.sigmoid(differences).sum(1)
        positive_differences = (positive[None, :] - positive[:, None]) / 0.06
        rank_positive = 0.5 + torch.sigmoid(positive_differences).sum(1)
        smooth_aps.append((rank_positive / rank_all).mean())
        top1_terms.append(F.softplus(
            (hard_negative[0] - positive.max() + 0.02) / 0.06
        ))
    centered = (score - score.masked_fill(~valid, 0.0).sum(1, keepdim=True)
                / valid.sum(1, keepdim=True).clamp_min(1)) / 0.08
    binary = F.binary_cross_entropy_with_logits(
        centered[valid], label.float()[valid],
        pos_weight=torch.tensor(5.0, device=score.device),
    )
    pairwise_loss = torch.stack(pairwise).mean()
    if LOSS_VARIANT == "smoothap":
        smooth_ap_loss = 1.0 - torch.stack(smooth_aps).mean()
        top1_loss = torch.stack(top1_terms).mean()
        return (
            0.25 * listwise
            + 1.25 * smooth_ap_loss
            + 0.45 * pairwise_loss
            + 0.20 * top1_loss
            + 0.05 * binary
        )
    return listwise + 0.45 * pairwise_loss + 0.08 * binary


def _train_epoch(model, optimizer, episodes, seed):
    model.train()
    rng = np.random.default_rng(seed)
    losses = []
    for episode in episodes:
        rows = episode["usable"].copy()
        rng.shuffle(rows)
        rows = rows[:min(1800, len(rows))]
        for start in range(0, len(rows), QUERY_BATCH):
            current = rows[start:start + QUERY_BATCH]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=DEVICE.type,
                enabled=DEVICE.type == "cuda",
                dtype=torch.float16,
            ):
                score = model(*_batch(episode, current))
                loss = _loss(
                    score,
                    torch.from_numpy(episode["label"][current]).to(DEVICE),
                    torch.from_numpy(episode["valid"][current]).to(DEVICE),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def _predict(model, episode, batch=32):
    model.eval()
    output = []
    for start in range(0, len(episode["qi"]), batch):
        rows = np.arange(start, min(start + batch, len(episode["qi"])))
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            value = model(*_batch(episode, rows))
        output.append(value.float().cpu().numpy())
    return np.concatenate(output)


def _evaluate(protocols, episodes, predictions, weight):
    rows = []
    for protocol, episode, prediction in zip(protocols, episodes, predictions, strict=True):
        score = episode["score"].copy()
        candidate = episode["candidate"]
        base = episode["features"][..., 0]
        reranked = base + weight * (prediction - base)
        np.put_along_axis(score, candidate, reranked, axis=1)
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    torch.manual_seed(92621)
    np.random.seed(92621)
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_features = _load_features(train, "train")
    val_features = _load_features(val, "val")

    token_state = torch.load(
        TOKEN_MODEL_PATH, map_location=DEVICE, weights_only=False
    )
    token_model = DinoTokenCrossMatcher().to(DEVICE)
    token_model.load_state_dict(token_state["model_state"])
    token_model.eval()

    print("building train family episodes", flush=True)
    train_episodes = [
        _episode(train, *train_features, token_model, seed) for seed in TRAIN_SEEDS
    ]
    print("building validation family episodes", flush=True)
    tune_official = _protocols(val, TUNE_SEEDS, {"base": val_features[0]})
    confirm_official = _protocols(val, CONFIRM_SEEDS, {"base": val_features[0]})
    val_episodes = [
        _episode(val, *val_features, token_model, seed)
        for seed in TUNE_SEEDS + CONFIRM_SEEDS
    ]
    tune_episodes = val_episodes[:len(TUNE_SEEDS)]
    confirm_episodes = val_episodes[len(TUNE_SEEDS):]

    feature_dim = train_episodes[0]["features"].shape[2]
    model = FamilyGraphReranker(feature_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=4e-3)
    history, snapshots = [], []
    for epoch in range(1, 9):
        loss = _train_epoch(model, optimizer, train_episodes, 92621 + epoch)
        prediction = [_predict(model, tune_episodes[0])]
        metrics = _evaluate(tune_official[:1], tune_episodes[:1], prediction, 0.50)
        row = {"epoch": epoch, "loss": loss, **metrics}
        history.append(row)
        snapshots.append(copy.deepcopy(model.state_dict()))
        print(json.dumps(row), flush=True)

    selected_epoch = max(
        range(len(history)),
        key=lambda index: (history[index]["mAP@10"], history[index]["Rank-1"]),
    )
    model.load_state_dict(snapshots[selected_epoch])
    tune_prediction = [_predict(model, episode) for episode in tune_episodes]
    confirm_prediction = [_predict(model, episode) for episode in confirm_episodes]
    grid = [
        {"family_weight": weight, **_evaluate(
            tune_official, tune_episodes, tune_prediction, weight
        )}
        for weight in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
                       0.40, 0.50, 0.70, 1.0)
    ]
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    confirmation = _evaluate(
        confirm_official, confirm_episodes, confirm_prediction,
        selected["family_weight"],
    )
    baseline = _evaluate(confirm_official, confirm_episodes, confirm_prediction, 0.0)
    report = {
        "design": "train-only top-50 query-conditioned static-gallery family GNN",
        "loss_variant": LOSS_VARIANT,
        "feature_dim": feature_dim,
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
    suffix = "_smoothap" if LOSS_VARIANT == "smoothap" else ""
    if EXPERIMENT_SUFFIX:
        suffix += (
            EXPERIMENT_SUFFIX
            if EXPERIMENT_SUFFIX.startswith("_")
            else f"_{EXPERIMENT_SUFFIX}"
        )
    Path(f"outputs/retrieval_v2/family_gnn{suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {"model_state": model.state_dict(), "feature_dim": feature_dim, "report": report},
        f"weights/strict_family_gnn{suffix}.pt",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
