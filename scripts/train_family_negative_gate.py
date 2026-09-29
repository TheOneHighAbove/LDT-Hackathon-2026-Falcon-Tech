"""Train an explicit train-only gate for suppressing top-50 look-alike cars."""

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
from scripts.train_dino_token_cross_top50 import DEVICE, DinoTokenCrossMatcher
from scripts.train_strict_family_gnn import (
    TRAIN_SEEDS,
    FamilyGraphReranker,
    _episode,
    _load_features,
    _predict as _predict_family,
)


def _context_features(episode, family_score):
    relation = episode["relation"].astype(np.float32)
    count = relation.shape[1]
    diagonal = np.eye(count, dtype=bool)[None]
    fused = np.where(diagonal, -np.inf, relation[..., 0])
    osnet = np.where(diagonal, -np.inf, relation[..., 1])
    token_valid = (relation[..., 3] > 0.5) & ~diagonal
    token_relation = np.where(token_valid, relation[..., 2], -np.inf)
    token_query = episode["features"][..., 1]

    top_index = np.argpartition(-fused, 4, axis=2)[..., :4]
    family_matrix = np.broadcast_to(family_score[:, None, :], fused.shape)
    token_query_matrix = np.broadcast_to(token_query[:, None, :], fused.shape)
    top_family = np.take_along_axis(family_matrix, top_index, axis=2)
    top_query = np.take_along_axis(token_query_matrix, top_index, axis=2)
    top_fused = np.take_along_axis(fused, top_index, axis=2)
    top_osnet = np.take_along_axis(osnet, top_index, axis=2)

    token_family_support = np.where(token_valid, family_matrix, -1e4).max(2)
    token_query_support = np.where(token_valid, token_query_matrix, -1e4).max(2)
    no_token = ~token_valid.any(2)
    token_family_support[no_token] = family_score[no_token]
    token_query_support[no_token] = token_query[no_token]
    token_max = token_relation.max(2)
    token_max[~np.isfinite(token_max)] = -1.0

    contextual = np.stack(
        (
            family_score,
            family_score - episode["features"][..., 0],
            top_family.max(2),
            top_family.mean(2),
            top_query.max(2),
            top_query.mean(2),
            token_family_support,
            token_query_support,
            token_max,
            token_valid.sum(2).astype(np.float32) / 8.0,
            top_fused.max(2),
            top_fused.mean(2),
            top_osnet.max(2),
            top_osnet.mean(2),
        ),
        axis=2,
    ).astype(np.float32)
    raw = np.concatenate((episode["features"], contextual), axis=2)
    z = (contextual - contextual.mean(1, keepdims=True)) / (
        contextual.std(1, keepdims=True) + 1e-6
    )
    return np.concatenate((raw, z), axis=2).astype(np.float32)


def _training_arrays(episodes, family_predictions):
    x, y = [], []
    for episode, prediction in zip(episodes, family_predictions, strict=True):
        features = _context_features(episode, prediction)
        rows = episode["usable"]
        valid = episode["valid"][rows]
        x.append(features[rows][valid])
        y.append(episode["label"][rows][valid])
    return np.concatenate(x), np.concatenate(y).astype(np.uint8)


def _predict(model, episode, family_prediction):
    features = _context_features(episode, family_prediction)
    flat = features.reshape(-1, features.shape[2])
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(flat)[:, 1]
    else:
        # Ranking models (for example LGBMRanker) emit an uncalibrated score.
        # Downstream code standardizes/ranks this value inside each current
        # query, so no probability calibration is required.
        values = model.predict(flat)
    return values.reshape(features.shape[:2])


def _evaluate(protocols, episodes, probability, weight):
    rows = []
    for protocol, episode, local in zip(protocols, episodes, probability, strict=True):
        score = episode["score"].copy()
        candidate = episode["candidate"]
        base = episode["features"][..., 0]
        base_z = (base - base.mean(1, keepdims=True)) / (base.std(1, keepdims=True) + 1e-6)
        local_z = (local - local.mean(1, keepdims=True)) / (local.std(1, keepdims=True) + 1e-6)
        combined = (1.0 - weight) * base_z + weight * local_z
        mean, std = base.mean(1, keepdims=True), base.std(1, keepdims=True) + 1e-6
        np.put_along_axis(score, candidate, mean + std * combined, axis=1)
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_features = _load_features(train, "train")
    val_features = _load_features(val, "val")
    token_state = torch.load(
        os.environ.get("DINO_TOKEN_MODEL", "weights/dino_token_cross_top50.pt"),
        map_location=DEVICE,
        weights_only=False,
    )
    token_model = DinoTokenCrossMatcher().to(DEVICE)
    token_model.load_state_dict(token_state["model_state"])
    token_model.eval()
    family_state = torch.load(
        os.environ.get(
            "FAMILY_GNN_MODEL", "weights/strict_family_gnn_smoothap.pt"
        ),
        map_location=DEVICE,
        weights_only=False,
    )
    family_model = FamilyGraphReranker(family_state["feature_dim"]).to(DEVICE)
    family_model.load_state_dict(family_state["model_state"])
    family_model.eval()

    print("building train episodes for negative gate", flush=True)
    train_episodes = [
        _episode(train, *train_features, token_model, seed) for seed in TRAIN_SEEDS
    ]
    train_family = [_predict_family(family_model, value) for value in train_episodes]
    x, y = _training_arrays(train_episodes, train_family)
    positives = max(int(y.sum()), 1)
    sample_weight = np.where(y > 0, min((len(y) - positives) / positives, 6.0), 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=280,
        max_leaf_nodes=23,
        min_samples_leaf=80,
        l2_regularization=6.0,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=92731,
    )
    model.fit(x, y, sample_weight=sample_weight)
    print(json.dumps({
        "train_pairs": len(y), "positives": positives,
        "feature_dim": x.shape[1], "iterations": int(model.n_iter_),
    }), flush=True)

    print("building validation episodes for negative gate", flush=True)
    tune_official = _protocols(val, TUNE_SEEDS, {"base": val_features[0]})
    confirm_official = _protocols(val, CONFIRM_SEEDS, {"base": val_features[0]})
    episodes = [
        _episode(val, *val_features, token_model, seed)
        for seed in TUNE_SEEDS + CONFIRM_SEEDS
    ]
    family = [_predict_family(family_model, value) for value in episodes]
    probability = [
        _predict(model, episode, prediction)
        for episode, prediction in zip(episodes, family, strict=True)
    ]
    tune_episodes, confirm_episodes = episodes[:len(TUNE_SEEDS)], episodes[len(TUNE_SEEDS):]
    tune_probability = probability[:len(TUNE_SEEDS)]
    confirm_probability = probability[len(TUNE_SEEDS):]
    grid = [
        {"gate_weight": weight, **_evaluate(
            tune_official, tune_episodes, tune_probability, weight
        )}
        for weight in (0.0, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20,
                       0.30, 0.40, 0.50, 0.70, 1.0)
    ]
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    confirmation = _evaluate(
        confirm_official, confirm_episodes, confirm_probability, selected["gate_weight"]
    )
    baseline = _evaluate(confirm_official, confirm_episodes, confirm_probability, 0.0)
    report = {
        "design": "train-only explicit look-alike negative gate with static-gallery support",
        "train_pairs": len(y),
        "train_positives": positives,
        "feature_dim": x.shape[1],
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "grid": grid,
    }
    suffix = os.environ.get("GATE_EXPERIMENT_SUFFIX", "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    Path(f"outputs/retrieval_v2/family_negative_gate{suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(model, f"weights/family_negative_gate{suffix}.joblib")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
