"""Train a query-group LambdaMART reranker for the strict top-50 shortlist.

Unlike the binary negative gate, this model optimizes ordering inside each
query group with a LambdaRank objective and MAP cutoffs.  All features are
computed from one current query and its static gallery.  Training uses target
train identities only; tune and confirmation validation protocols remain
strictly separated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

# Build episodes with the loss-branch OSNet base while retaining the older,
# independently trained token/family experts used by the strongest mixed stack.
os.environ.setdefault(
    "OSNET_TRAIN_PATH",
    "outputs/expert_fusion/cache/osnet_loss_branch_ensemble_train.npz",
)
os.environ.setdefault(
    "OSNET_ENSEMBLE_PATH",
    "outputs/expert_fusion/osnet_loss_branch_ensemble_val_embeddings.npy",
)

from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_dino_token_cross_top50 import DEVICE, DinoTokenCrossMatcher
from scripts.train_family_negative_gate import _context_features
from scripts.probe_train_fitted_verifier import _colors
from scripts.train_strict_family_gnn import (
    TRAIN_SEEDS,
    FamilyGraphReranker,
    _episode,
    _load_features,
    _predict as _predict_family,
)


OUTPUT = Path(os.environ.get(
    "FAMILY_LAMBDARANK_MODEL", "weights/family_lambdarank_lbs.joblib"
))
REPORT = Path(os.environ.get(
    "FAMILY_LAMBDARANK_REPORT",
    "outputs/retrieval_v2/family_lambdarank_lbs.json",
))
USE_APPEARANCE = os.environ.get(
    "FAMILY_LAMBDARANK_APPEARANCE", "0"
).lower() in {"1", "true", "yes"}


def _cosine(left, right, axis=-1):
    numerator = np.sum(left * right, axis=axis)
    denominator = np.linalg.norm(left, axis=axis) * np.linalg.norm(
        right, axis=axis
    )
    return numerator / np.maximum(denominator, 1e-6)


def _appearance_features(frame, episode, colors):
    """Pairwise regional color and bbox-aspect evidence for each shortlist."""

    query = episode["qi"]
    gallery = episode["gi"][episode["candidate"]]
    query_color = colors[query].reshape(-1, 5, 32)[:, None]
    gallery_color = colors[gallery].reshape(*gallery.shape, 5, 32)
    blocks = []
    for start, stop in ((0, 32), (0, 16), (16, 24), (24, 32)):
        blocks.append(_cosine(
            query_color[..., start:stop],
            gallery_color[..., start:stop],
        ))
    region = np.concatenate(blocks, axis=2)

    aspect = np.log(np.maximum(
        frame.w.to_numpy(dtype=np.float32)
        / np.maximum(frame.h.to_numpy(dtype=np.float32), 1e-6),
        1e-6,
    ))
    aspect_gap = np.abs(aspect[query, None] - aspect[gallery])[..., None]
    aspect_product = (aspect[query, None] * aspect[gallery])[..., None]
    return np.concatenate((
        region,
        region.mean(2, keepdims=True),
        region.min(2, keepdims=True),
        region.max(2, keepdims=True),
        aspect_gap,
        aspect_product,
    ), axis=2).astype(np.float32)


def _ranker_features(frame, episode, family_prediction, colors=None):
    features = _context_features(episode, family_prediction)
    if colors is not None:
        features = np.concatenate((
            features, _appearance_features(frame, episode, colors)
        ), axis=2)
    return features


def _arrays(frame, episode, family_prediction, colors=None):
    features = _ranker_features(
        frame, episode, family_prediction, colors
    )
    x, y, groups = [], [], []
    for row in episode["usable"]:
        valid = episode["valid"][row]
        values = features[row, valid]
        labels = episode["label"][row, valid].astype(np.uint8)
        if not labels.any() or labels.all():
            continue
        x.append(values)
        y.append(labels)
        groups.append(len(labels))
    return (
        np.concatenate(x).astype(np.float32, copy=False),
        np.concatenate(y),
        np.asarray(groups, dtype=np.int32),
    )


def _combine(items):
    return (
        np.concatenate([item[0] for item in items]),
        np.concatenate([item[1] for item in items]),
        np.concatenate([item[2] for item in items]),
    )


def _parameters(n_estimators: int):
    return {
        "objective": "lambdarank",
        "metric": "map",
        "eval_at": [1, 5, 10],
        "label_gain": [0, 1],
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 5.0,
        "max_bin": 127,
        "random_state": 73621,
        "n_jobs": -1,
        "verbosity": -1,
    }


def _predict(model, episode, family_prediction, frame=None, colors=None):
    if frame is None and colors is not None:
        raise ValueError("frame is required when appearance colors are supplied")
    features = (
        _context_features(episode, family_prediction)
        if frame is None
        else _ranker_features(frame, episode, family_prediction, colors)
    )
    values = model.booster_.predict(
        features.reshape(-1, features.shape[2]),
        num_iteration=model.best_iteration_ if model.best_iteration_ else -1,
    )
    return values.reshape(features.shape[:2]).astype(np.float32)


def _evaluate(protocols, episodes, predictions, weight):
    rows = []
    for protocol, episode, local in zip(
        protocols, episodes, predictions, strict=True
    ):
        score = episode["score"].copy()
        candidate = episode["candidate"]
        base = episode["features"][..., 0]
        base_z = (base - base.mean(1, keepdims=True)) / (
            base.std(1, keepdims=True) + 1e-6
        )
        local_z = (local - local.mean(1, keepdims=True)) / (
            local.std(1, keepdims=True) + 1e-6
        )
        combined = (1.0 - weight) * base_z + weight * local_z
        mean = base.mean(1, keepdims=True)
        std = base.std(1, keepdims=True) + 1e-6
        np.put_along_axis(score, candidate, mean + std * combined, axis=1)
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main() -> None:
    np.random.seed(73621)
    torch.manual_seed(73621)
    train = pd.read_csv("splits/train.csv")
    val = pd.read_csv("splits/val.csv")
    train_features = _load_features(train, "train")
    val_features = _load_features(val, "val")
    train_colors = _colors(train, "train") if USE_APPEARANCE else None
    val_colors = _colors(val, "val") if USE_APPEARANCE else None

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

    print("building train-only LambdaRank episodes", flush=True)
    train_episodes = [
        _episode(train, *train_features, token_model, seed) for seed in TRAIN_SEEDS
    ]
    train_family = [
        _predict_family(family_model, episode) for episode in train_episodes
    ]
    arrays = [
        _arrays(train, episode, prediction, train_colors)
        for episode, prediction in zip(
            train_episodes, train_family, strict=True
        )
    ]
    fit_x, fit_y, fit_group = _combine(arrays[:2])
    eval_x, eval_y, eval_group = arrays[2]
    selector = lgb.LGBMRanker(**_parameters(900))
    selector.fit(
        fit_x,
        fit_y,
        group=fit_group,
        eval_set=[(eval_x, eval_y)],
        eval_group=[eval_group],
        callbacks=[lgb.early_stopping(70), lgb.log_evaluation(25)],
    )
    best_iteration = int(selector.best_iteration_ or selector.n_estimators)
    all_x, all_y, all_group = _combine(arrays)
    model = lgb.LGBMRanker(**_parameters(best_iteration))
    model.fit(all_x, all_y, group=all_group)
    print(json.dumps({
        "train_pairs": len(all_y),
        "train_queries": len(all_group),
        "positive_rate": float(all_y.mean()),
        "best_iteration": best_iteration,
        "feature_dim": all_x.shape[1],
    }), flush=True)

    print("building separated tune/confirmation episodes", flush=True)
    all_seeds = TUNE_SEEDS + CONFIRM_SEEDS
    val_episodes = [
        _episode(val, *val_features, token_model, seed) for seed in all_seeds
    ]
    val_family = [
        _predict_family(family_model, episode) for episode in val_episodes
    ]
    predictions = [
        _predict(model, episode, family, val, val_colors)
        for episode, family in zip(val_episodes, val_family, strict=True)
    ]
    split = len(TUNE_SEEDS)
    tune_episodes, confirmation_episodes = (
        val_episodes[:split], val_episodes[split:]
    )
    tune_predictions, confirmation_predictions = (
        predictions[:split], predictions[split:]
    )
    tune_protocols = _protocols(val, TUNE_SEEDS, {"base": val_features[0]})
    confirmation_protocols = _protocols(
        val, CONFIRM_SEEDS, {"base": val_features[0]}
    )
    grid = [
        {"ranker_weight": weight, **_evaluate(
            tune_protocols, tune_episodes, tune_predictions, weight
        )}
        for weight in (
            0.0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20,
            0.30, 0.40, 0.50, 0.65, 0.80, 1.0,
        )
    ]
    selected = max(
        grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"])
    )
    confirmation = _evaluate(
        confirmation_protocols,
        confirmation_episodes,
        confirmation_predictions,
        selected["ranker_weight"],
    )
    baseline = _evaluate(
        confirmation_protocols,
        confirmation_episodes,
        confirmation_predictions,
        0.0,
    )
    report = {
        "design": (
            "train-only query-group LambdaMART on strict current-query + "
            "static-gallery top-50 features"
        ),
        "compliance": {
            "validation_identity_training": False,
            "camera_at_inference": False,
            "other_queries_at_inference": False,
            "test_pseudo_labels": False,
        },
        "train_pairs": len(all_y),
        "train_queries": len(all_group),
        "positive_rate": float(all_y.mean()),
        "feature_dim": all_x.shape[1],
        "appearance_features": USE_APPEARANCE,
        "best_iteration": best_iteration,
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {
            metric: confirmation[metric] - baseline[metric]
            for metric in ("mAP@10", "Rank-1", "Rank-5")
        },
        "grid": grid,
        "feature_importance": sorted(
            (
                {"feature": int(index), "gain": float(gain)}
                for index, gain in enumerate(
                    model.booster_.feature_importance(importance_type="gain")
                )
            ),
            key=lambda item: item["gain"],
            reverse=True,
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
