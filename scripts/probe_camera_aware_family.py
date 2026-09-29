"""Combine camera-transition evidence with latent query-family recovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_camera_transition_prior import _camera_log_likelihood
from scripts.probe_family_reranking import _metrics
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_query_family_clustering import (
    _aggregate,
    _cluster_stats,
    _components,
    _normalize,
    _prepare,
)
from src.reranking import query_expansion


def _camera_augmented(protocol, prior, camera_index, pair_alpha, gallery_alpha):
    result = dict(protocol)
    q_camera = np.asarray([camera_index[value] for value in protocol["q"].camera_id])
    g_camera = np.asarray([camera_index[value] for value in protocol["g"].camera_id])
    q_prior = prior[q_camera[:, None], q_camera[None, :]]
    g_prior = prior[q_camera[:, None], g_camera[None, :]]
    result["qsim"] = protocol["qsim"] + pair_alpha * q_prior
    gallery_score = protocol["raw_query"] @ protocol["gallery"].T + gallery_alpha * g_prior
    result["gallery_top10"] = np.argsort(
        -gallery_score, axis=1, kind="mergesort"
    )[:, :10]
    result["camera_score"] = g_prior
    return result


def _evaluate(protocol, components, self_weight, gallery_alpha):
    raw = protocol["raw_query"]
    augmented = raw.copy()
    for component in components:
        if len(component) <= 1:
            continue
        index = np.asarray(component, dtype=np.int64)
        prototype = _normalize(raw[index].mean(axis=0, keepdims=True))[0]
        augmented[index] = _normalize(self_weight * raw[index] + prototype)
    query = query_expansion(augmented, protocol["gallery"], top_k=2, alpha=1.0)
    score = query @ protocol["gallery"].T
    if gallery_alpha:
        score = score + gallery_alpha * protocol["camera_score"]
    order = np.argsort(-score, axis=1, kind="mergesort")
    return _metrics(order, protocol["q"], protocol["g"])


def main() -> None:
    train_frame = pd.read_csv("splits/train.csv")
    val_frame = pd.read_csv("splits/val.csv")
    embeddings = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
        allow_pickle=False,
    ).astype(np.float64))
    raw_tune = {seed: _prepare(embeddings, val_frame, seed) for seed in TUNE_SEEDS}
    raw_confirmation = {
        seed: _prepare(embeddings, val_frame, seed) for seed in CONFIRM_SEEDS
    }
    prior, camera_index = _camera_log_likelihood(train_frame, 0.25)
    gallery_alpha = 0.02
    configs = []
    for pair_alpha in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
        tune = {
            seed: _camera_augmented(
                protocol, prior, camera_index, pair_alpha, gallery_alpha
            )
            for seed, protocol in raw_tune.items()
        }
        # Search locally around the already-confirmed visual graph optimum.
        # A broad grid repeats the expensive full ranking thousands of times
        # without testing a distinct hypothesis.
        for neighbor_k in (2, 3):
            for threshold in (0.55, 0.60):
                for overlap_min in (1, 2):
                    for complete_link in (False,):
                        clustered = {
                            seed: _components(
                                protocol, neighbor_k=neighbor_k, threshold=threshold,
                                overlap_min=overlap_min, complete_link=complete_link,
                            )
                            for seed, protocol in tune.items()
                        }
                        stats = [
                            _cluster_stats(clustered[seed], tune[seed]["q"])
                            for seed in TUNE_SEEDS
                        ]
                        for self_weight in (1.0, 2.0):
                            rows = [
                                _evaluate(tune[seed], clustered[seed], self_weight, gallery_alpha)
                                for seed in TUNE_SEEDS
                            ]
                            configs.append({
                                "pair_alpha": pair_alpha,
                                "gallery_alpha": gallery_alpha,
                                "neighbor_k": neighbor_k,
                                "threshold": threshold,
                                "overlap_min": overlap_min,
                                "complete_link": complete_link,
                                "self_weight": self_weight,
                                "tune": _aggregate(rows),
                                "pair_precision": float(np.mean([x["pair_precision"] for x in stats])),
                                "pair_recall": float(np.mean([x["pair_recall"] for x in stats])),
                                "grouped_fraction": float(np.mean([x["grouped_fraction"] for x in stats])),
                            })
        print(f"finished pair_alpha={pair_alpha}", flush=True)
    selected = max(configs, key=lambda row: (row["tune"]["mAP"], row["tune"]["AP10"]))
    confirmation = {
        seed: _camera_augmented(
            protocol, prior, camera_index, selected["pair_alpha"], gallery_alpha
        )
        for seed, protocol in raw_confirmation.items()
    }
    component_keys = ("neighbor_k", "threshold", "overlap_min", "complete_link")
    clustered = {
        seed: _components(
            protocol, **{key: selected[key] for key in component_keys}
        )
        for seed, protocol in confirmation.items()
    }
    confirmed = _aggregate([
        _evaluate(confirmation[seed], clustered[seed], selected["self_weight"], gallery_alpha)
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
        "confirmation_grouped_fraction": float(np.mean([x["grouped_fraction"] for x in stats])),
        "top_configs": sorted(configs, key=lambda row: row["tune"]["mAP"], reverse=True)[:20],
    }
    Path("outputs/expert_fusion/camera_aware_family.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
