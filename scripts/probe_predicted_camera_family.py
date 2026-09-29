"""End-to-end camera-aware family recovery with camera IDs inferred from pixels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_camera_transition_prior import _camera_log_likelihood
from scripts.probe_camera_aware_family import _evaluate
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_query_family_clustering import (
    _aggregate,
    _cluster_stats,
    _components,
    _normalize,
    _prepare,
)


def _load_probability(path, frame, camera_index):
    with np.load(path, allow_pickle=False) as archive:
        ids = frame.image_id.astype(str).to_numpy(dtype=np.str_)
        if not np.array_equal(ids, archive["image_ids"]):
            raise RuntimeError("camera prediction cache is not aligned")
        raw = archive["probability"].astype(np.float64)
        classes = archive["classes"]
    probability = np.zeros((len(frame), len(camera_index)), dtype=np.float64)
    for source, camera in enumerate(classes):
        probability[:, camera_index[camera]] = raw[:, source]
    return probability


def _sharpen(probability, power):
    if power == np.inf:
        result = np.zeros_like(probability)
        result[np.arange(len(result)), probability.argmax(axis=1)] = 1.0
        return result
    result = probability ** power
    return result / result.sum(axis=1, keepdims=True).clip(min=1e-12)


def _augment(protocol, probability, prior, pair_alpha, gallery_alpha):
    result = dict(protocol)
    q_probability = probability[protocol["qi"]]
    g_probability = probability[protocol["gi"]]
    q_camera_score = q_probability @ prior @ q_probability.T
    g_camera_score = q_probability @ prior @ g_probability.T
    result["qsim"] = protocol["qsim"] + pair_alpha * q_camera_score
    initial = protocol["raw_query"] @ protocol["gallery"].T + gallery_alpha * g_camera_score
    result["gallery_top10"] = np.argsort(-initial, axis=1, kind="mergesort")[:, :10]
    result["camera_score"] = g_camera_score
    return result


def main():
    train = pd.read_csv("splits/train.csv")
    val = pd.read_csv("splits/val.csv")
    prior, camera_index = _camera_log_likelihood(train, 0.25)
    geometry_probability = _load_probability(
        "outputs/expert_fusion/cache/camera_predictions_val.npz", val, camera_index
    )
    scene_probability = _load_probability(
        "outputs/expert_fusion/cache/scene_camera_mlp_predictions_val.npz", val, camera_index
    )
    embeddings = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
        allow_pickle=False,
    ).astype(np.float64))
    raw_tune = {seed: _prepare(embeddings, val, seed) for seed in TUNE_SEEDS}
    raw_confirmation = {seed: _prepare(embeddings, val, seed) for seed in CONFIRM_SEEDS}
    configs = []
    for scene_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        raw_probability = (
            scene_weight * scene_probability
            + (1.0 - scene_weight) * geometry_probability
        )
        raw_probability /= raw_probability.sum(axis=1, keepdims=True).clip(min=1e-12)
        for power in (1.0, 2.0, 4.0, 8.0):
            probability = _sharpen(raw_probability, power)
            pair_alpha, gallery_alpha, threshold = 0.05, 0.02, 0.55
            tune = {
                seed: _augment(protocol, probability, prior, pair_alpha, gallery_alpha)
                for seed, protocol in raw_tune.items()
            }
            clustered = {
                seed: _components(
                    protocol, neighbor_k=3, threshold=threshold,
                    overlap_min=2, complete_link=False,
                )
                for seed, protocol in tune.items()
            }
            rows = [
                _evaluate(tune[seed], clustered[seed], 1.0, gallery_alpha)
                for seed in TUNE_SEEDS
            ]
            stats = [
                _cluster_stats(clustered[seed], tune[seed]["q"])
                for seed in TUNE_SEEDS
            ]
            configs.append({
                "scene_weight": scene_weight,
                "power": power,
                "pair_alpha": pair_alpha,
                "gallery_alpha": gallery_alpha,
                "threshold": threshold,
                "tune": _aggregate(rows),
                "pair_precision": float(np.mean([x["pair_precision"] for x in stats])),
                "pair_recall": float(np.mean([x["pair_recall"] for x in stats])),
            })
        print(f"finished scene_weight={scene_weight}", flush=True)
    selected = max(configs, key=lambda row: (row["tune"]["mAP"], row["tune"]["AP10"]))
    raw_probability = (
        selected["scene_weight"] * scene_probability
        + (1.0 - selected["scene_weight"]) * geometry_probability
    )
    raw_probability /= raw_probability.sum(axis=1, keepdims=True).clip(min=1e-12)
    probability = _sharpen(raw_probability, selected["power"])
    confirmation = {
        seed: _augment(
            protocol, probability, prior,
            selected["pair_alpha"], selected["gallery_alpha"],
        )
        for seed, protocol in raw_confirmation.items()
    }
    clustered = {
        seed: _components(
            protocol, neighbor_k=3, threshold=selected["threshold"],
            overlap_min=2, complete_link=False,
        )
        for seed, protocol in confirmation.items()
    }
    confirmed = _aggregate([
        _evaluate(
            confirmation[seed], clustered[seed], 1.0, selected["gallery_alpha"]
        )
        for seed in CONFIRM_SEEDS
    ])
    baseline = _aggregate([
        _evaluate(
            protocol, [[index] for index in range(len(protocol["q"]))], 1.0, 0.0
        )
        for protocol in raw_confirmation.values()
    ])
    stats = [
        _cluster_stats(clustered[seed], confirmation[seed]["q"])
        for seed in CONFIRM_SEEDS
    ]
    result = {
        "selected": selected,
        "confirmation": confirmed,
        "baseline_confirmation": baseline,
        "delta": {key: confirmed[key] - baseline[key] for key in baseline},
        "confirmation_pair_precision": float(np.mean([x["pair_precision"] for x in stats])),
        "confirmation_pair_recall": float(np.mean([x["pair_recall"] for x in stats])),
        "top_configs": sorted(configs, key=lambda row: row["tune"]["mAP"], reverse=True)[:20],
    }
    Path("outputs/expert_fusion/predicted_camera_family.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
