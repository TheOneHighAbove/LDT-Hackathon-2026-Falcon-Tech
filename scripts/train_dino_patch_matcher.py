"""Learn a strict-streaming hard-negative matcher from adapted DINO patches.

The matcher never sees validation identities.  It compares a current query
with each first-stage candidate independently and therefore remains valid for
an unseen static gallery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_streaming_top25_verifier import (
    TOP, TRAIN_SEEDS, _make_train_protocol, _normalize, _prepared, _train_candidates,
)


DEVICE = torch.device("cuda")
CONTEXTUAL = os.environ.get("DINO_PATCH_CONTEXTUAL", "0").lower() in {"1", "true", "yes"}
PATCH_GRID = int(os.environ.get("DINO_PATCH_GRID", "5"))
OSNET_TRAIN_PATH = os.environ.get(
    "OSNET_TRAIN_PATH",
    "outputs/expert_fusion/cache/osnet_smoothap_train.npz",
)
OSNET_VAL_PATH = os.environ.get(
    "OSNET_ENSEMBLE_PATH",
    "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
)
EXPERIMENT_SUFFIX = os.environ.get("PATCH_EXPERIMENT_SUFFIX", "").strip()
TRAIN_TOKEN_PATH = os.environ.get("DINO_PATCH_TRAIN_TOKENS", "").strip()
VAL_TOKEN_PATH = os.environ.get("DINO_PATCH_VAL_TOKENS", "").strip()
MAX_TRAIN_QUERIES = int(os.environ.get("DINO_PATCH_MAX_TRAIN_QUERIES", "3000"))


def _load(path, frame, key):
    with np.load(path, allow_pickle=False) as archive:
        expected = frame.image_id.astype(str).to_numpy(dtype=np.str_)
        if not np.array_equal(archive["image_ids"], expected):
            raise RuntimeError(f"unaligned cache: {path}")
        return archive[key].astype(np.float32)


def _fuse(osnet, dino):
    return _normalize(np.concatenate((np.sqrt(0.75) * osnet, np.sqrt(0.25) * dino), axis=1))


@torch.inference_mode()
def _patch_features(tokens, fused, osnet, dino, query, gallery, batch=1024,
                    grid_size=None):
    grid_size = PATCH_GRID if grid_size is None else grid_size
    output = []
    for start in range(0, len(query), batch):
        qidx, gidx = query[start:start + batch], gallery[start:start + batch]
        q = torch.from_numpy(tokens[qidx]).to(DEVICE, non_blocking=True)
        g = torch.from_numpy(tokens[gidx]).to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            correlation = torch.bmm(q, g.transpose(1, 2)).float()
        # Appearance-invariant evidence.
        top = correlation.flatten(1).topk(32, dim=1).values
        # Keep a bounded set of the strongest invariant matches at high
        # resolution; the aligned blocks below retain exact part locations.
        best_count = min(32, grid_size * grid_size)
        qbest = correlation.amax(2).topk(best_count, dim=1).values
        gbest = correlation.amax(1).topk(best_count, dim=1).values
        # Spatially aligned evidence, evaluated in both horizontal directions.
        aligned = correlation.diagonal(dim1=1, dim2=2)
        token_count = grid_size * grid_size
        flip_index = torch.arange(token_count, device=DEVICE).reshape(
            grid_size, grid_size
        ).flip(1).flatten()
        flipped = correlation[:, torch.arange(token_count, device=DEVICE), flip_index]
        # Same-height matches tolerate viewpoint changes but retain vehicle-part layout.
        grid = correlation.reshape(
            -1, grid_size, grid_size, grid_size, grid_size
        )
        same_row = torch.stack([
            grid[:, row, :, row, :].amax(2) for row in range(grid_size)
        ], 1).flatten(1)
        visual = torch.cat((top, qbest, gbest, aligned, flipped, same_row), dim=1).cpu().numpy()
        scalar = np.stack((
            np.sum(fused[qidx] * fused[gidx], axis=1),
            np.sum(osnet[qidx] * osnet[gidx], axis=1),
            np.sum(dino[qidx] * dino[gidx], axis=1),
        ), axis=1).astype(np.float32)
        output.append(np.concatenate((scalar, visual), axis=1))
    return np.concatenate(output)


def _contextualize(features):
    """Add shortlist-relative evidence without looking at any other query."""
    values = features.reshape(-1, TOP, features.shape[1])
    blocks = (
        values[:, :, :3],
        values[:, :, 3:35].mean(2, keepdims=True),
        values[:, :, 3:35].max(2, keepdims=True),
        values[:, :, 35:60].mean(2, keepdims=True),
        values[:, :, 60:85].mean(2, keepdims=True),
        values[:, :, 85:110].mean(2, keepdims=True),
        values[:, :, 85:110].max(2, keepdims=True),
        values[:, :, 110:135].mean(2, keepdims=True),
        values[:, :, 110:135].max(2, keepdims=True),
        values[:, :, 135:160].mean(2, keepdims=True),
        values[:, :, 135:160].max(2, keepdims=True),
    )
    signal = np.concatenate(blocks, axis=2)
    z = (signal - signal.mean(1, keepdims=True)) / (signal.std(1, keepdims=True) + 1e-6)
    gap = signal - signal[:, :1]
    rank = np.broadcast_to(
        np.linspace(0.0, 1.0, TOP, dtype=np.float32)[None, :, None],
        (len(values), TOP, 1),
    )
    return np.concatenate((values, z, gap, rank), axis=2).reshape(len(features), -1)


def _training_arrays(frame, fused, osnet, dino, tokens):
    rng = np.random.default_rng(94231)
    features, labels = [], []
    for seed in TRAIN_SEEDS:
        protocol = _make_train_protocol(frame, fused, seed)
        rows = protocol["usable"].copy()
        rng.shuffle(rows)
        rows = rows[:min(MAX_TRAIN_QUERIES, len(rows))]
        candidates, target = _train_candidates(protocol, rows, rng)
        if CONTEXTUAL:
            raw = np.einsum(
                "qd,qkd->qk", fused[protocol["qi"][rows]], fused[protocol["gi"]][candidates]
            )
            order = np.argsort(-raw, axis=1, kind="stable")
            candidates = np.take_along_axis(candidates, order, axis=1)
            target = np.take_along_axis(target, order, axis=1)
        query = np.repeat(protocol["qi"][rows], TOP)
        gallery = protocol["gi"][candidates.reshape(-1)]
        value = _patch_features(tokens, fused, osnet, dino, query, gallery)
        features.append(_contextualize(value) if CONTEXTUAL else value)
        labels.append(target.reshape(-1))
        print(f"train protocol {seed}: {len(query)} pairs", flush=True)
    return np.concatenate(features), np.concatenate(labels)


def _predict(model, protocol, prepared, tokens, fused, osnet, dino):
    candidate = prepared["candidates"]
    query = np.repeat(prepared["qi"], TOP)
    gallery = prepared["gi"][candidate.reshape(-1)]
    features = _patch_features(tokens, fused, osnet, dino, query, gallery)
    if CONTEXTUAL:
        features = _contextualize(features)
    probability = model.predict_proba(features)[:, 1]
    return probability.reshape(len(candidate), TOP)


def _evaluate(protocols, prepared, prediction, weight):
    rows = []
    for protocol, data, local in zip(protocols, prepared, prediction, strict=True):
        score = data["base"].copy()
        candidate = data["candidates"]
        base = np.take_along_axis(score, candidate, axis=1)
        base_z = (base - base.mean(1, keepdims=True)) / (base.std(1, keepdims=True) + 1e-6)
        local_z = (local - local.mean(1, keepdims=True)) / (local.std(1, keepdims=True) + 1e-6)
        reranked = (1.0 - weight) * base_z + weight * local_z
        # Only the relative order inside the original top-25 changes.
        order = np.argsort(-reranked, axis=1, kind="stable")
        ordered = np.take_along_axis(candidate, order, axis=1)
        ranked = np.argsort(-score, axis=1, kind="stable")
        ranked[:, :TOP] = ordered
        artificial = np.empty_like(score)
        artificial[np.arange(len(score))[:, None], ranked] = np.arange(score.shape[1], 0, -1)
        rows.append(_official_from_scores(artificial, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_osnet = _normalize(_load(
        OSNET_TRAIN_PATH, train, "embeddings"
    ))
    val_osnet = _normalize(np.load(
        OSNET_VAL_PATH, allow_pickle=False
    ))
    train_dino = _normalize(_load(
        "outputs/expert_fusion/cache/dinov2_vehicle_cls_train.npz", train, "embeddings"
    ))
    val_dino = _normalize(_load(
        "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz", val, "cls"
    ))
    default_train_tokens = (
        f"outputs/expert_fusion/cache/dinov2_vehicle_patches_train"
        f"{'_10x10' if PATCH_GRID == 10 else ''}.npz"
    )
    default_val_tokens = (
        f"outputs/expert_fusion/cache/dinov2_vehicle_patches_val"
        f"{'_10x10' if PATCH_GRID == 10 else ''}.npz"
    )
    train_tokens = _load(TRAIN_TOKEN_PATH or default_train_tokens, train, "tokens")
    val_tokens = _load(VAL_TOKEN_PATH or default_val_tokens, val, "tokens")
    train_fused, val_fused = _fuse(train_osnet, train_dino), _fuse(val_osnet, val_dino)

    x, y = _training_arrays(train, train_fused, train_osnet, train_dino, train_tokens)
    model = HistGradientBoostingClassifier(
        learning_rate=0.055 if CONTEXTUAL else 0.07,
        max_iter=360 if CONTEXTUAL else 240, max_leaf_nodes=31,
        min_samples_leaf=35, l2_regularization=2.0 if CONTEXTUAL else 1.5,
        class_weight="balanced",
        early_stopping=True, validation_fraction=0.12, random_state=94231,
    )
    model.fit(x, y)
    print(json.dumps({"train_pairs": len(y), "positive_rate": float(y.mean()),
                      "iterations": int(model.n_iter_)}), flush=True)

    tune = _protocols(val, TUNE_SEEDS, {"base": val_fused})
    confirm = _protocols(val, CONFIRM_SEEDS, {"base": val_fused})
    tune_data = [_prepared(p, val_fused) for p in tune]
    confirm_data = [_prepared(p, val_fused) for p in confirm]
    tune_prediction = [_predict(model, p, d, val_tokens, val_fused, val_osnet, val_dino)
                       for p, d in zip(tune, tune_data, strict=True)]
    confirm_prediction = [_predict(model, p, d, val_tokens, val_fused, val_osnet, val_dino)
                          for p, d in zip(confirm, confirm_data, strict=True)]
    grid = [{"patch_weight": weight, **_evaluate(tune, tune_data, tune_prediction, weight)}
            for weight in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)]
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    confirmation = _evaluate(
        confirm, confirm_data, confirm_prediction, selected["patch_weight"]
    )
    baseline = _evaluate(confirm, confirm_data, confirm_prediction, 0.0)
    report = {
        "design": f"train-only adapted-DINO {PATCH_GRID}x{PATCH_GRID} spatial hard-negative matcher; strict streaming",
        "train_pairs": len(y), "positive_rate": float(y.mean()),
        "selected_tune": selected, "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {key: confirmation[key] - baseline[key]
                  for key in ("mAP@10", "Rank-1", "Rank-5")},
        "grid": grid,
    }
    suffix = "_contextual" if CONTEXTUAL else ""
    if PATCH_GRID == 10:
        suffix += "_10x10"
    if EXPERIMENT_SUFFIX:
        suffix += (
            EXPERIMENT_SUFFIX
            if EXPERIMENT_SUFFIX.startswith("_")
            else f"_{EXPERIMENT_SUFFIX}"
        )
    Path(f"outputs/expert_fusion/dino_patch_matcher{suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(model, f"weights/dino_patch_matcher{suffix}.joblib")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
