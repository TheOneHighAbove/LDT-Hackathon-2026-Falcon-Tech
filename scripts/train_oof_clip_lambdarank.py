"""Identity-OOF LambdaRank over the exact LBS+LambdaRank final top-25."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official


CACHE = Path(os.environ.get(
    "CLIP_DEFORMABLE_CACHE",
    "outputs/retrieval_v2/clip_deformable_lbs_lambdarank_top25_val.npz",
))
NEURAL = Path(os.environ.get(
    "CLIP_DEFORMABLE_PREDICTIONS",
    "outputs/retrieval_v2/clip_deformable_lbs_lambdarank_predictions.npz",
))
OUTPUT = Path(os.environ.get(
    "OOF_LAMBDARANK_OUTPUT",
    "outputs/retrieval_v2/oof_clip_lambdarank_lbs_lambdarank.json",
))
WEIGHT = Path(os.environ.get(
    "OOF_LAMBDARANK_WEIGHT",
    "weights/oof_clip_lambdarank_lbs_lambdarank.joblib",
))
FOLDS = 5
TOP = 25


def _z(value: np.ndarray) -> np.ndarray:
    return (value - value.mean(1, keepdims=True)) / (
        value.std(1, keepdims=True) + 1e-6
    )


def _load(frame: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    with (
        np.load(CACHE, allow_pickle=False) as cache,
        np.load(NEURAL, allow_pickle=False) as neural,
    ):
        for seed in TUNE_SEEDS + CONFIRM_SEEDS:
            prefix = f"s{seed}_"
            qi = cache[prefix + "qi"].astype(np.int64)
            gi = cache[prefix + "gi"].astype(np.int64)
            base = cache[prefix + "base"].astype(np.float32)
            neural_score = neural[prefix + "score"].astype(np.float32)
            extra = np.stack(
                (
                    neural_score,
                    _z(neural_score),
                    neural_score - base,
                    _z(neural_score) - _z(base),
                ),
                axis=2,
            )
            feature = np.concatenate(
                (cache[prefix + "feature"].astype(np.float32), extra),
                axis=2,
            )
            result[seed] = {
                "q": frame.iloc[qi],
                "g": frame.iloc[gi],
                "query_vehicle": frame.vehicle_id.to_numpy()[qi],
                "feature": feature,
                "label": cache[prefix + "label"].astype(bool),
                "valid": cache[prefix + "valid"].astype(bool),
                "candidate": cache[prefix + "candidate"].astype(np.int64),
                "base": base,
            }
    return result


def _training_arrays(protocols, held):
    features, labels, groups = [], [], []
    for protocol in protocols:
        rows = np.flatnonzero(
            ~np.isin(protocol["query_vehicle"], held)
            & protocol["label"].any(1)
        )
        for row in rows:
            valid = protocol["valid"][row]
            features.append(protocol["feature"][row, valid])
            labels.append(protocol["label"][row, valid].astype(np.int8))
            groups.append(int(valid.sum()))
    return np.concatenate(features), np.concatenate(labels), np.asarray(groups)


def _predict(model, protocol):
    shape = protocol["feature"].shape[:2]
    flat = protocol["feature"].reshape(-1, protocol["feature"].shape[2])
    return model.predict(flat).reshape(shape)


def _ranking(protocol, prediction, weight, rerank_k):
    mixed = _z(protocol["base"]) + weight * _z(prediction)
    local = np.argsort(-mixed[:, :rerank_k], axis=1, kind="stable")
    if rerank_k < TOP:
        local = np.concatenate(
            (
                local,
                np.broadcast_to(
                    np.arange(rerank_k, TOP),
                    (len(local), TOP - rerank_k),
                ),
            ),
            axis=1,
        )
    return np.take_along_axis(protocol["candidate"], local, axis=1)


def _evaluate(protocols, predictions, weight, rerank_k):
    return _aggregate([
        _official(
            _ranking(protocol, prediction, weight, rerank_k),
            protocol["q"],
            protocol["g"],
        )
        for protocol, prediction in zip(protocols, predictions, strict=True)
    ])


def _baseline(protocols):
    return _aggregate([
        _official(protocol["candidate"], protocol["q"], protocol["g"])
        for protocol in protocols
    ])


def main() -> None:
    frame = pd.read_csv("splits/val.csv")
    data = _load(frame)
    tune = [data[seed] for seed in TUNE_SEEDS]
    confirm = [data[seed] for seed in CONFIRM_SEEDS]
    identities = np.unique(frame.vehicle_id.to_numpy())
    rng = np.random.default_rng(20260925)
    rng.shuffle(identities)
    folds = np.array_split(identities, FOLDS)
    tune_prediction = [np.zeros_like(value["base"]) for value in tune]
    confirm_prediction = [np.zeros_like(value["base"]) for value in confirm]
    models, training = [], []
    for fold_index, held in enumerate(folds):
        x, y, group = _training_arrays(tune, held)
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="map",
            n_estimators=500,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=45,
            max_bin=127,
            colsample_bytree=0.72,
            reg_lambda=6.0,
            reg_alpha=0.15,
            verbosity=-1,
            random_state=260950 + fold_index,
        )
        model.fit(x, y, group=group)
        models.append(model)
        record = {
            "fold": fold_index + 1,
            "held_identities": len(held),
            "queries": len(group),
            "pairs": len(y),
            "positives": int(y.sum()),
        }
        training.append(record)
        print(json.dumps(record), flush=True)
        for protocols, destinations in (
            (tune, tune_prediction),
            (confirm, confirm_prediction),
        ):
            for protocol, destination in zip(protocols, destinations, strict=True):
                mask = np.isin(protocol["query_vehicle"], held)
                destination[mask] = _predict(model, protocol)[mask]

    baseline_tune = _baseline(tune)
    grid = [
        {
            "weight": weight,
            "rerank_k": rerank_k,
            **_evaluate(tune, tune_prediction, weight, rerank_k),
        }
        for rerank_k in (10, 15, 25)
        for weight in (
            0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
            0.20, 0.30, 0.40, 0.50, 0.70, 1.0,
        )
    ]
    selected = max(
        grid,
        key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]),
    )
    baseline_confirmation = _baseline(confirm)
    confirmation = _evaluate(
        confirm,
        confirm_prediction,
        selected["weight"],
        selected["rerank_k"],
    )
    report = {
        "design": (
            "5-fold identity-OOF query-group LambdaRank over exact LBS+appearance-"
            "LambdaRank top-25 with frozen CLIP geometry and neural cross-score"
        ),
        "feature_dim": tune[0]["feature"].shape[2],
        "training": training,
        "baseline_tune": baseline_tune,
        "selected_tune": selected,
        "baseline_confirmation": baseline_confirmation,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline_confirmation[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "top_tune": sorted(
            grid,
            key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]),
            reverse=True,
        )[:20],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    joblib.dump(
        {
            "models": models,
            "fold_identities": folds,
            "tune_oof_prediction": [value.astype(np.float16) for value in tune_prediction],
            "confirm_oof_prediction": [value.astype(np.float16) for value in confirm_prediction],
            "report": report,
        },
        WEIGHT,
        compress=3,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
