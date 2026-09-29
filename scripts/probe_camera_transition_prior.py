"""Evaluate train-only camera-transition likelihood ratios for vehicle retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_family_reranking import _metrics
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.train_query_family_clustering import _aggregate, _aligned, _prepare
from src.reranking import query_expansion


def _camera_log_likelihood(frame: pd.DataFrame, smoothing: float) -> tuple[np.ndarray, dict]:
    camera_values = np.sort(frame.camera_id.unique())
    camera_index = {camera: index for index, camera in enumerate(camera_values)}
    identities = frame.vehicle_id.unique()
    incidence = np.zeros((len(identities), len(camera_values)), dtype=np.float64)
    grouped = frame.groupby("vehicle_id").camera_id.unique()
    for row, vehicle_id in enumerate(identities):
        incidence[row, [camera_index[camera] for camera in grouped[vehicle_id]]] = 1.0
    pair_count = incidence.T @ incidence
    marginal = incidence.sum(axis=0)
    # PMI is the likelihood ratio against a random gallery camera.  This
    # avoids merely rewarding popular cameras and transfers across protocols.
    numerator = (pair_count + smoothing) * (len(identities) + smoothing)
    denominator = (marginal[:, None] + smoothing) * (marginal[None, :] + smoothing)
    log_likelihood = np.log(numerator / denominator).clip(-4.0, 4.0)
    return log_likelihood, camera_index


def _evaluate(protocol, prior, camera_index, alpha):
    query = query_expansion(protocol["raw_query"], protocol["gallery"], top_k=2, alpha=1.0)
    visual = query @ protocol["gallery"].T
    q_camera = np.asarray([camera_index[value] for value in protocol["q"].camera_id])
    g_camera = np.asarray([camera_index[value] for value in protocol["g"].camera_id])
    camera_score = prior[q_camera[:, None], g_camera[None, :]]
    order = np.argsort(-(visual + alpha * camera_score), axis=1, kind="mergesort")
    return _metrics(order, protocol["q"], protocol["g"])


def main() -> None:
    train_frame = pd.read_csv("splits/train.csv")
    val_frame = pd.read_csv("splits/val.csv")
    conv = _aligned("outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz", val_frame)
    osnet = _aligned("outputs/expert_fusion/cache/osnet_target_finetuned_val_hard_smoothap.npz", val_frame)
    tune = {seed: _prepare(val_frame, conv, osnet, seed) for seed in TUNE_SEEDS}
    confirmation = {seed: _prepare(val_frame, conv, osnet, seed) for seed in CONFIRM_SEEDS}
    configs = []
    for smoothing in (0.25, 1.0, 4.0, 16.0):
        prior, camera_index = _camera_log_likelihood(train_frame, smoothing)
        for alpha in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
            rows = [_evaluate(protocol, prior, camera_index, alpha) for protocol in tune.values()]
            configs.append({
                "smoothing": smoothing,
                "alpha": alpha,
                "tune": _aggregate(rows),
            })
    selected = max(configs, key=lambda row: (row["tune"]["mAP"], row["tune"]["AP10"]))
    prior, camera_index = _camera_log_likelihood(train_frame, selected["smoothing"])
    confirmed = _aggregate([
        _evaluate(protocol, prior, camera_index, selected["alpha"])
        for protocol in confirmation.values()
    ])
    baseline = _aggregate([
        _evaluate(protocol, prior, camera_index, 0.0)
        for protocol in confirmation.values()
    ])
    result = {
        "selected": selected,
        "confirmation": confirmed,
        "baseline_confirmation": baseline,
        "delta": {key: confirmed[key] - baseline[key] for key in baseline},
        "top_configs": sorted(configs, key=lambda row: row["tune"]["mAP"], reverse=True)[:15],
    }
    Path("outputs/expert_fusion/camera_transition_prior.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
