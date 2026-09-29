"""Lock fusion weights on tune seeds and report target-OSNet confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_pair_verifier import CONFIRM_SEEDS, _normalize
from src.postprocess_eval import build_identity_stratified_protocol
from src.reranking import database_side_augmentation, query_expansion


def _fuse(conv, frozen, tuned, weights):
    arrays = []
    for weight, values in zip(weights, (conv, frozen, tuned), strict=True):
        if weight > 0:
            arrays.append(np.sqrt(weight) * values)
    return _normalize(np.concatenate(arrays, axis=1))


def _evaluate(frame, embeddings, seeds):
    rows = []
    for seed in seeds:
        protocol = build_identity_stratified_protocol(frame, seed=seed)
        qi, gi = protocol.query_indices, protocol.gallery_indices
        gallery_embeddings = database_side_augmentation(embeddings[gi], top_k=5, alpha=2.0)
        query_embeddings = query_expansion(embeddings[qi], gallery_embeddings, top_k=2, alpha=1.0)
        score = query_embeddings @ gallery_embeddings.T
        query, gallery = frame.iloc[qi], frame.iloc[gi]
        gp, gc = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
        aps, ap10s, rank1s, rank5s, recalls = [], [], [], [], []
        for index, row in enumerate(query.itertuples(index=False)):
            valid = ~((gp == row.vehicle_id) & (gc == row.camera_id))
            order = np.flatnonzero(valid)[np.argsort(-score[index, valid], kind="mergesort")]
            positions = np.flatnonzero(gp[order] == row.vehicle_id) + 1
            precision = np.arange(1, len(positions) + 1) / positions
            in10 = positions <= 10
            aps.append(float(precision.mean()))
            ap10s.append(float(precision[in10].sum() / len(positions)))
            recalls.append(float(in10.sum() / len(positions)))
            rank1s.append(float(positions[0] == 1))
            rank5s.append(float(positions[0] <= 5))
        rows.append({
            "seed": seed,
            "mAP": float(np.mean(aps)),
            "AP10": float(np.mean(ap10s)),
            "positive_recall10": float(np.mean(recalls)),
            "rank1": float(np.mean(rank1s)),
            "rank5": float(np.mean(rank5s)),
        })
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("mAP", "AP10", "positive_recall10", "rank1", "rank5")
    }, rows


def main():
    frame = pd.read_csv("splits/val.csv")
    conv = _normalize(np.load(
        "outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz", allow_pickle=False
    )["embeddings"])
    frozen = _normalize(np.load(
        "outputs/expert_fusion/cache/osnet_openvino_val_4aaad3e5db648618.npz", allow_pickle=False
    )["embeddings"])
    tuned = _normalize(np.load(
        "outputs/expert_fusion/cache/osnet_target_finetuned_val.npz", allow_pickle=False
    )["embeddings"])
    tune_seeds, confirmation_seeds = CONFIRM_SEEDS[:3], CONFIRM_SEEDS[3:]
    grid = []
    for conv_weight in (0.0, 0.10, 0.20, 0.30):
        for frozen_weight in (0.0, 0.10, 0.20, 0.30):
            tuned_weight = 1.0 - conv_weight - frozen_weight
            if tuned_weight <= 0:
                continue
            weights = (conv_weight, frozen_weight, tuned_weight)
            metrics, _ = _evaluate(frame, _fuse(conv, frozen, tuned, weights), tune_seeds)
            grid.append({"weights_conv_frozen_tuned": weights, "tune": metrics})
            print(weights, metrics["mAP"], flush=True)
    selected = max(grid, key=lambda row: row["tune"]["mAP"])
    weights = tuple(selected["weights_conv_frozen_tuned"])
    confirmation, rows = _evaluate(
        frame, _fuse(conv, frozen, tuned, weights), confirmation_seeds
    )
    baseline, _ = _evaluate(
        frame, _fuse(conv, frozen, tuned, (0.30, 0.70, 0.0)), confirmation_seeds
    )
    target_fixed, _ = _evaluate(
        frame, _fuse(conv, frozen, tuned, (0.30, 0.0, 0.70)), confirmation_seeds
    )
    result = {
        "selection": "maximize mAP on seeds 101,211,307",
        "selected": selected,
        "confirmation": confirmation,
        "baseline_confirmation": baseline,
        "fixed_30_70_target_confirmation": target_fixed,
        "delta_vs_baseline": {key: confirmation[key] - baseline[key] for key in confirmation},
        "confirmation_rows": rows,
        "top_grid": sorted(grid, key=lambda row: row["tune"]["mAP"], reverse=True)[:8],
    }
    path = Path("outputs/expert_fusion/osnet_target_fusion.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
