"""Probe transductive query-side aggregation on a test-shaped protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.expert_fusion import weighted_concatenate
from src.reranking import database_side_augmentation, query_expansion
from scripts.probe_family_reranking import _metrics


TUNE_SEEDS = (1337, 2027, 3407, 4517, 7919)
CONFIRM_SEEDS = (101, 211, 307, 401, 503)


def _camera_representative_protocol(frame: pd.DataFrame, seed: int):
    """Put one random image per identity-camera pair in gallery."""

    rng = np.random.default_rng(seed)
    gallery = []
    for _, group in frame.groupby(["vehicle_id", "camera_id"], sort=False):
        indices = group.index.to_numpy()
        gallery.append(int(indices[int(rng.integers(len(indices)))]))
    gallery_indices = np.asarray(gallery, dtype=np.int64)
    is_query = np.ones(len(frame), dtype=bool)
    is_query[gallery_indices] = False
    query_indices = np.flatnonzero(is_query)
    return query_indices, gallery_indices


def _protocol(embeddings: np.ndarray, frame: pd.DataFrame, seed: int):
    qi, gi = _camera_representative_protocol(frame, seed)
    raw_query, raw_gallery = embeddings[qi], embeddings[gi]
    gallery = database_side_augmentation(raw_gallery, top_k=5, alpha=2.0)
    baseline_query = query_expansion(raw_query, gallery, top_k=2, alpha=1.0)
    return raw_query, gallery, baseline_query, frame.iloc[qi], frame.iloc[gi]


def _evaluate(protocols, config):
    rows = []
    for seed, (raw_query, gallery, baseline_query, q, g) in protocols.items():
        if config["stage"] == "baseline":
            query = baseline_query
        elif config["stage"] == "before_qe":
            query = database_side_augmentation(
                raw_query,
                top_k=config["neighbor_k"],
                alpha=config["alpha"],
                self_weight=config["self_weight"],
            )
            query = query_expansion(query, gallery, top_k=2, alpha=1.0)
        elif config["stage"] == "after_qe":
            query = database_side_augmentation(
                baseline_query,
                top_k=config["neighbor_k"],
                alpha=config["alpha"],
                self_weight=config["self_weight"],
            )
        else:
            raise ValueError(config["stage"])
        scores = query @ gallery.T
        order = np.argsort(-scores, axis=1, kind="mergesort")
        rows.append({"seed": seed, **_metrics(order, q, g)})
    keys = ("mAP", "AP10", "positive_recall_at_10", "mean_positive_images_at_10", "rank1", "rank5")
    return {
        **config,
        **{key: float(np.mean([row[key] for row in rows])) for key in keys},
        "per_seed": rows,
    }


def main() -> None:
    frame = pd.read_csv("splits/val.csv")
    with np.load("outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz", allow_pickle=False) as archive:
        conv = archive["embeddings"]
    with np.load("outputs/expert_fusion/cache/osnet_openvino_val_4aaad3e5db648618.npz", allow_pickle=False) as archive:
        osnet = archive["embeddings"]
    embeddings = weighted_concatenate(conv, osnet, primary_weight=0.30).astype(np.float64)
    tune_protocols = {seed: _protocol(embeddings, frame, seed) for seed in TUNE_SEEDS}
    confirm_protocols = {seed: _protocol(embeddings, frame, seed) for seed in CONFIRM_SEEDS}

    configs = [{"stage": "baseline"}]
    for stage in ("before_qe", "after_qe"):
        for neighbor_k in (1, 2, 3, 5):
            for alpha in (1.0, 2.0):
                for self_weight in (1.0, 2.0, 4.0):
                    configs.append(
                        {
                            "stage": stage,
                            "neighbor_k": neighbor_k,
                            "alpha": alpha,
                            "self_weight": self_weight,
                        }
                    )
    tune = [_evaluate(tune_protocols, config) for config in configs]
    selected_ap = max(tune, key=lambda row: (row["AP10"], row["mAP"]))
    selected_recall = max(
        tune,
        key=lambda row: (
            row["positive_recall_at_10"],
            row["mean_positive_images_at_10"],
            row["AP10"],
        ),
    )
    config_keys = ("stage", "neighbor_k", "alpha", "self_weight")
    confirm_ap = _evaluate(
        confirm_protocols,
        {key: selected_ap[key] for key in config_keys if key in selected_ap},
    )
    confirm_recall = _evaluate(
        confirm_protocols,
        {key: selected_recall[key] for key in config_keys if key in selected_recall},
    )
    baseline = _evaluate(confirm_protocols, {"stage": "baseline"})
    metric_keys = ("mAP", "AP10", "positive_recall_at_10", "mean_positive_images_at_10", "rank1", "rank5")
    result = {
        "protocol": "one random gallery image per identity-camera; all remaining images are queries",
        "shape": {
            "queries": len(next(iter(tune_protocols.values()))[3]),
            "gallery": len(next(iter(tune_protocols.values()))[4]),
        },
        "selected_for_AP10": selected_ap,
        "selected_for_positive_recall_at_10": selected_recall,
        "confirmation_AP10": confirm_ap,
        "confirmation_positive_recall_at_10": confirm_recall,
        "baseline": baseline,
        "delta_AP10": {key: confirm_ap[key] - baseline[key] for key in metric_keys},
        "delta_positive_recall_at_10": {key: confirm_recall[key] - baseline[key] for key in metric_keys},
        "tune_grid": tune,
    }
    Path("outputs/expert_fusion/multi_query_aggregation_probe.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in result if key != "tune_grid"}, indent=2))


if __name__ == "__main__":
    main()
