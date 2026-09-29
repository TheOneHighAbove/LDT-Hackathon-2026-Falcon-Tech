"""Train a true local-token cross-encoder on train-ID hard lookalikes."""

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
from scripts.probe_streaming_official import _aggregate, _official
from scripts.train_dino_patch_matcher import _fuse, _load
from scripts.train_streaming_top25_verifier import _normalize
from src.reranking import database_side_augmentation


DEVICE = torch.device("cuda")
TOP = 25
GRID = 7
EPOCHS = int(os.environ.get("CLIP_DEFORMABLE_EPOCHS", "5"))
QUERY_BATCH = 3
TRAIN_SEEDS = (1709, 2711, 3907)
TRAIN_TOKENS = Path(os.environ.get(
    "CLIP_TRAIN_TOKENS",
    "outputs/expert_fusion/cache/clip_vehicle_stage2_train_tokens_7x7.npy",
))
VAL_TOKENS_14 = Path(os.environ.get(
    "CLIP_VAL_TOKENS", "outputs/expert_fusion/cache/clip_vehicle_stage2_val_tokens.npy"
))
VAL_TOKENS_7 = Path(os.environ.get(
    "CLIP_VAL_TOKENS_7",
    "outputs/expert_fusion/cache/clip_vehicle_stage2_val_tokens_7x7.npy",
))
STACK = Path(os.environ.get(
    "TOP25_CACHE",
    "outputs/retrieval_v2/lbs_lambdarank_top25_val.npz",
))
OUTPUT = Path(os.environ.get(
    "CLIP_DEFORMABLE_REPORT",
    "outputs/retrieval_v2/clip_deformable_cross_encoder_lbs_lambdarank.json",
))
WEIGHT = Path(os.environ.get(
    "CLIP_DEFORMABLE_WEIGHT",
    "weights/clip_deformable_cross_encoder_lbs_lambdarank.pt",
))
TRAIN_RETRIEVAL = Path(os.environ.get(
    "CLIP_TRAIN_RETRIEVAL",
    "outputs/expert_fusion/osnet_loss_branch_ensemble_train_embeddings.npy",
))
INIT_WEIGHT = os.environ.get("CLIP_DEFORMABLE_INIT", "").strip()


class CLIPDeformableCrossEncoder(nn.Module):
    def __init__(self, input_dim=768, dim=64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128, bias=False),
            nn.GELU(),
            nn.Linear(128, dim, bias=False),
        )
        self.position = nn.Parameter(torch.zeros(1, 1, GRID * GRID, dim))
        nn.init.trunc_normal_(self.position, std=0.01)
        index = torch.arange(GRID * GRID)
        row, column = index // GRID, index % GRID
        direct = (
            (row[:, None] - row[None, :]).float().square()
            + (column[:, None] - column[None, :]).float().square()
        ) / (GRID * GRID)
        flip = (
            (row[:, None] - row[None, :]).float().square()
            + ((GRID - 1 - column[:, None]) - column[None, :]).float().square()
        ) / (GRID * GRID)
        self.register_buffer("direct_distance", direct)
        self.register_buffer("flip_distance", flip)
        self.geometry_strength = nn.Parameter(torch.tensor(3.0))
        feature_dim = 12 * dim + 17
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _aligned_features(q, g, attention):
        aligned = torch.einsum("bkts,bksd->bktd", attention, g)
        q_expanded = q.expand(-1, g.shape[1], -1, -1)
        difference = torch.abs(q_expanded - aligned)
        product = q_expanded * aligned
        return torch.cat((
            difference.mean(2), difference.amax(2),
            product.mean(2), product.amax(2),
        ), dim=2)

    def forward(self, query, gallery, base):
        q = F.normalize(self.projection(query)[:, None] + self.position, dim=3)
        g = F.normalize(self.projection(gallery) + self.position, dim=3)
        correlation = torch.einsum("bqtd,bksd->bkts", q, g)
        strength = F.softplus(self.geometry_strength)
        free_attention = F.softmax(10.0 * correlation, dim=3)
        direct_attention = F.softmax(
            10.0 * correlation - strength * self.direct_distance[None, None], dim=3
        )
        flip_attention = F.softmax(
            10.0 * correlation - strength * self.flip_distance[None, None], dim=3
        )
        visual = torch.cat((
            self._aligned_features(q, g, free_attention),
            self._aligned_features(q, g, direct_attention),
            self._aligned_features(q, g, flip_attention),
        ), dim=2)
        flat = correlation.flatten(2)
        row_best = correlation.amax(3)
        column_best = correlation.amax(2)
        index = torch.arange(GRID * GRID, device=query.device)
        flip_index = index.reshape(GRID, GRID).flip(1).flatten()
        direct = correlation[:, :, index, index]
        flipped = correlation[:, :, index, flip_index]
        scalar = torch.stack((
            base,
            flat.mean(2), flat.std(2), flat.amax(2),
            flat.topk(16, dim=2).values.mean(2),
            row_best.mean(2), row_best.std(2), row_best.amin(2), row_best.amax(2),
            column_best.mean(2), column_best.std(2),
            direct.mean(2), direct.amax(2),
            flipped.mean(2), flipped.amax(2),
            (free_attention * correlation).sum(3).mean(2),
            torch.maximum(
                (direct_attention * correlation).sum(3).mean(2),
                (flip_attention * correlation).sum(3).mean(2),
            ),
        ), dim=2)
        residual = 0.45 * torch.tanh(
            self.head(torch.cat((visual, scalar), dim=2)).squeeze(2)
        )
        return base + residual


def _compact_val_tokens(frame):
    if VAL_TOKENS_7.is_file():
        return np.load(VAL_TOKENS_7, mmap_mode="r", allow_pickle=False)
    source = np.load(VAL_TOKENS_14, mmap_mode="r", allow_pickle=False)
    output = np.lib.format.open_memmap(
        VAL_TOKENS_7, mode="w+", dtype=np.float16,
        shape=(len(frame), GRID * GRID, source.shape[2]),
    )
    for start in range(0, len(frame), 64):
        value = torch.from_numpy(
            np.asarray(source[start:start + 64]).astype(np.float32)
        ).to(DEVICE)
        value = F.avg_pool2d(
            value.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2), 2
        ).permute(0, 2, 3, 1).flatten(1, 2)
        output[start:start + len(value)] = F.normalize(value, dim=2).half().cpu().numpy()
    output.flush()
    return np.load(VAL_TOKENS_7, mmap_mode="r", allow_pickle=False)


def _train_protocol(frame, embedding, seed):
    qi, gi = _camera_representative_protocol(frame, seed)
    gallery = database_side_augmentation(embedding[gi], top_k=5, alpha=2.0)
    score = (embedding[qi] @ gallery.T).astype(np.float32)
    candidate = np.argsort(-score, axis=1, kind="stable")[:, :TOP]
    q_vehicle = frame.vehicle_id.to_numpy()[qi]
    q_camera = frame.camera_id.to_numpy()[qi]
    g_vehicle = frame.vehicle_id.to_numpy()[gi]
    g_camera = frame.camera_id.to_numpy()[gi]
    positives = [
        np.flatnonzero((g_vehicle == vehicle) & (g_camera != camera))
        for vehicle, camera in zip(q_vehicle, q_camera, strict=True)
    ]
    return {
        "qi": qi, "gi": gi, "score": score, "candidate": candidate,
        "q_vehicle": q_vehicle, "q_camera": q_camera,
        "g_vehicle": g_vehicle, "g_camera": g_camera,
        "positives": positives,
        "usable": np.asarray([i for i, value in enumerate(positives) if len(value)]),
    }


def _training_candidates(protocol, rows, rng):
    candidate = protocol["candidate"][rows].copy()
    for output_row, protocol_row in enumerate(rows):
        positive = protocol["positives"][int(protocol_row)]
        missing = positive[~np.isin(positive, candidate[output_row])]
        if len(missing):
            inject = rng.choice(missing, min(3, len(missing)), replace=False)
            same = np.isin(candidate[output_row], positive)
            replace = np.flatnonzero(~same)[-len(inject):]
            candidate[output_row, replace] = inject
    q_vehicle = protocol["q_vehicle"][rows]
    q_camera = protocol["q_camera"][rows]
    g_vehicle = protocol["g_vehicle"][candidate]
    g_camera = protocol["g_camera"][candidate]
    same = g_vehicle == q_vehicle[:, None]
    label = same & (g_camera != q_camera[:, None])
    valid = ~(same & (g_camera == q_camera[:, None]))
    n_positive = np.asarray([len(protocol["positives"][int(row)]) for row in rows])
    return candidate, label, valid, n_positive


def _batch(tokens, protocol, rows, candidate):
    query_index = protocol["qi"][rows]
    gallery_index = protocol["gi"][candidate]
    base = np.take_along_axis(protocol["score"][rows], candidate, axis=1)
    return (
        torch.from_numpy(np.asarray(tokens[query_index]).astype(np.float32)).to(
            DEVICE, non_blocking=True
        ),
        torch.from_numpy(np.asarray(tokens[gallery_index]).astype(np.float32)).to(
            DEVICE, non_blocking=True
        ),
        torch.from_numpy(base).to(DEVICE, non_blocking=True),
    )


def _loss(score, label, valid, n_positive):
    differences = (score[:, None, :] - score[:, :, None]) / 0.055
    soft_precedes = torch.sigmoid(differences)
    rank_all = 0.5 + (soft_precedes * valid[:, None, :]).sum(2)
    rank_positive = 0.5 + (soft_precedes * label[:, None, :]).sum(2)
    cutoff = torch.sigmoid((10.5 - rank_all) / 0.70)
    smooth_ap = (
        (rank_positive / rank_all.clamp_min(1e-6)) * cutoff * label
    ).sum(1) / n_positive.float().clamp(max=10.0)
    negative = valid & ~label
    hard_negative = score.masked_fill(~negative, -torch.inf).topk(10, dim=1).values
    pair = F.softplus(
        (hard_negative[:, :, None] - score[:, None, :] + 0.025) / 0.065
    )
    pair = (pair * label[:, None, :]).sum((1, 2)) / (
        10.0 * label.sum(1).clamp_min(1)
    )
    positive_max = score.masked_fill(~label, -torch.inf).amax(1)
    top1 = F.softplus(
        (hard_negative[:, 0] - positive_max + 0.03) / 0.055
    )
    return 1.40 * (1.0 - smooth_ap).mean() + 0.35 * pair.mean() + 0.30 * top1.mean()


def _train_epoch(model, optimizer, protocols, tokens, seed):
    model.train()
    rng = np.random.default_rng(seed)
    losses = []
    for protocol in protocols:
        rows = protocol["usable"].copy()
        rng.shuffle(rows)
        rows = rows[:min(1800, len(rows))]
        for start in range(0, len(rows), QUERY_BATCH):
            current = rows[start:start + QUERY_BATCH]
            candidate, label, valid, n_positive = _training_candidates(
                protocol, current, rng
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                score = model(*_batch(tokens, protocol, current, candidate))
                loss = _loss(
                    score,
                    torch.from_numpy(label).to(DEVICE),
                    torch.from_numpy(valid).to(DEVICE),
                    torch.from_numpy(n_positive).to(DEVICE),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _validation_protocols(frame, seeds):
    output = []
    with np.load(STACK, allow_pickle=False) as archive:
        for seed in seeds:
            prefix = f"s{seed}_"
            qi = archive[prefix + "qi"].astype(np.int64)
            gi = archive[prefix + "gi"].astype(np.int64)
            output.append({
                "q": frame.iloc[qi], "g": frame.iloc[gi],
                "qi": qi, "gi": gi,
                "candidate": archive[prefix + "candidate"].astype(np.int64),
                "base": archive[prefix + "base"].astype(np.float32),
            })
    return output


@torch.inference_mode()
def _predict(model, protocol, tokens, batch=6):
    model.eval()
    output = []
    rows = np.arange(len(protocol["qi"]))
    proxy = {
        "qi": protocol["qi"], "gi": protocol["gi"],
        "score": np.zeros((len(rows), len(protocol["gi"])), dtype=np.float32),
    }
    np.put_along_axis(proxy["score"], protocol["candidate"], protocol["base"], axis=1)
    for start in range(0, len(rows), batch):
        current = rows[start:start + batch]
        candidate = protocol["candidate"][current]
        with torch.autocast("cuda", dtype=torch.float16):
            value = model(*_batch(tokens, proxy, current, candidate))
        output.append(value.float().cpu().numpy())
    return np.concatenate(output)


def _z(values):
    return (values - values.mean(1, keepdims=True)) / (
        values.std(1, keepdims=True) + 1e-6
    )


def _evaluate(protocols, predictions, weight, rerank_k):
    rows = []
    for protocol, prediction in zip(protocols, predictions, strict=True):
        mixed = _z(protocol["base"]) + weight * (
            _z(prediction) - _z(protocol["base"])
        )
        local = np.argsort(-mixed[:, :rerank_k], axis=1, kind="stable")
        if rerank_k < TOP:
            local = np.concatenate((
                local,
                np.broadcast_to(
                    np.arange(rerank_k, TOP), (len(local), TOP - rerank_k)
                ),
            ), axis=1)
        order = np.take_along_axis(protocol["candidate"], local, axis=1)
        rows.append(_official(order, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    torch.manual_seed(260950)
    np.random.seed(260950)
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_tokens = np.load(TRAIN_TOKENS, mmap_mode="r", allow_pickle=False)
    val_tokens = _compact_val_tokens(val)
    if TRAIN_RETRIEVAL.is_file():
        train_embedding = _normalize(np.load(TRAIN_RETRIEVAL, allow_pickle=False))
    else:
        train_osnet = _normalize(_load(
            "outputs/expert_fusion/cache/osnet_smoothap_train.npz",
            train,
            "embeddings",
        ))
        train_dino = _normalize(_load(
            "outputs/expert_fusion/cache/dinov2_vehicle_cls_train.npz",
            train,
            "embeddings",
        ))
        train_embedding = _fuse(train_osnet, train_dino)
    train_protocols = [
        _train_protocol(train, train_embedding, seed) for seed in TRAIN_SEEDS
    ]
    tune = _validation_protocols(val, TUNE_SEEDS)

    model = CLIPDeformableCrossEncoder().to(DEVICE)
    if INIT_WEIGHT:
        initial = torch.load(INIT_WEIGHT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(os.environ.get("CLIP_DEFORMABLE_LR", "2e-4")),
        weight_decay=4e-3,
    )
    history, states, tune_predictions = [], [], []
    for epoch in range(1, EPOCHS + 1):
        loss = _train_epoch(model, optimizer, train_protocols, train_tokens, 260950 + epoch)
        prediction = [_predict(model, protocol, val_tokens) for protocol in tune]
        grid = [
            {"weight": weight, "rerank_k": rerank_k,
             **_evaluate(tune, prediction, weight, rerank_k)}
            for rerank_k in (10, 15, 25)
            for weight in (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0)
        ]
        selected = max(
            grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"])
        )
        row = {"epoch": epoch, "loss": loss, **selected}
        history.append(row)
        states.append(copy.deepcopy(model.state_dict()))
        tune_predictions.append(prediction)
        print(json.dumps(row), flush=True)
    selected_epoch = max(
        range(EPOCHS),
        key=lambda index: (
            history[index]["mAP@10"], history[index]["Rank-1"], history[index]["Rank-5"]
        ),
    )
    selected = history[selected_epoch]
    model.load_state_dict(states[selected_epoch])
    confirm = _validation_protocols(val, CONFIRM_SEEDS)
    confirm_prediction = [_predict(model, protocol, val_tokens) for protocol in confirm]
    baseline = _evaluate(confirm, confirm_prediction, 0.0, selected["rerank_k"])
    confirmation = _evaluate(
        confirm, confirm_prediction, selected["weight"], selected["rerank_k"]
    )
    report = {
        "design": (
            "train-ID target-adapted CLIP 7x7 learned free/direct/flip deformable "
            "cross-encoder; SmoothAP@10 hard-lookalike training"
        ),
        "train_retrieval": str(TRAIN_RETRIEVAL),
        "init_weight": INIT_WEIGHT or None,
        "history": history,
        "selected_epoch": selected_epoch + 1,
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({
        "model_state": states[selected_epoch],
        "report": report,
    }, WEIGHT)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
