"""Optimize top-10 family completeness with gallery-neighbour propagation.

The probe is label-free at inference.  It keeps the global top-50 candidate
set, then propagates a query score between visually neighbouring gallery
images so that several views of one likely vehicle can rise together.
Configuration selection uses truncated AP@10 on tune seeds and is evaluated
once on the locked confirmation seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.expert_fusion import weighted_concatenate
from src.postprocess_eval import build_identity_stratified_protocol
from src.reranking import database_side_augmentation, query_expansion


TUNE_SEEDS = (1337, 2027, 3407, 4517, 7919)
CONFIRM_SEEDS = (101, 211, 307, 401, 503)
POOL_SIZE = 50
_NEIGHBOR_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _normalized_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.linspace(1.0, 0.0, len(values), endpoint=True)
    return result


def _gallery_neighbors(gallery_similarity: np.ndarray, count: int) -> np.ndarray:
    ranking = gallery_similarity.copy()
    np.fill_diagonal(ranking, -np.inf)
    return np.argsort(-ranking, axis=1, kind="mergesort")[:, :count]


def _rerank(
    base: np.ndarray,
    gallery_similarity: np.ndarray,
    *,
    method: str,
    neighbor_k: int,
    blend: float,
    preserve_top1: bool,
) -> np.ndarray:
    """Return a full order while preserving the baseline top-50 membership."""

    baseline_order = np.argsort(-base, axis=1, kind="mergesort")
    output = baseline_order.copy()
    cache_key = (id(gallery_similarity), neighbor_k)
    cached = _NEIGHBOR_CACHE.get(cache_key)
    if cached is None:
        neighbors = _gallery_neighbors(gallery_similarity, neighbor_k)
        # Reciprocal edges are a conservative pseudo-family relation.
        reciprocal = np.zeros_like(gallery_similarity, dtype=bool)
        rows = np.arange(len(neighbors))[:, None]
        reciprocal[rows, neighbors] = True
        reciprocal &= reciprocal.T
        _NEIGHBOR_CACHE[cache_key] = (neighbors, reciprocal)
    else:
        neighbors, reciprocal = cached

    for query_index in range(len(base)):
        pool = baseline_order[query_index, :POOL_SIZE]
        base_local = base[query_index, pool]
        propagated = np.empty(len(pool), dtype=np.float64)
        for local_index, candidate in enumerate(pool):
            candidate_neighbors = neighbors[candidate]
            if method.startswith("reciprocal"):
                candidate_neighbors = candidate_neighbors[
                    reciprocal[candidate, candidate_neighbors]
                ]
            if not len(candidate_neighbors):
                propagated[local_index] = base[query_index, candidate]
                continue
            neighbor_scores = base[query_index, candidate_neighbors]
            affinities = np.clip(
                gallery_similarity[candidate, candidate_neighbors], 0.0, 1.0
            )
            if method.endswith("max"):
                propagated[local_index] = float(np.max(neighbor_scores * affinities))
            elif method.endswith("mean"):
                weights = affinities**2
                propagated[local_index] = float(
                    np.sum(neighbor_scores * weights) / max(np.sum(weights), 1e-12)
                )
            else:
                raise ValueError(f"unknown method: {method}")

        combined = (
            (1.0 - blend) * _normalized_rank(base_local)
            + blend * _normalized_rank(propagated)
        )
        local_order = np.argsort(-combined, kind="mergesort")
        if preserve_top1:
            root_position = int(np.flatnonzero(local_order == 0)[0])
            local_order = np.concatenate(
                (np.array([0]), np.delete(local_order, root_position))
            )
        output[query_index, :POOL_SIZE] = pool[local_order]
    return output


def _metrics(order: np.ndarray, q: pd.DataFrame, g: pd.DataFrame) -> dict[str, float]:
    gp, gc = g.vehicle_id.to_numpy(), g.camera_id.to_numpy()
    aps, truncated, positive_recall, hit_count, r1, r5 = [], [], [], [], [], []
    for index, row in enumerate(q.itertuples(index=False)):
        valid_order = order[index][
            ~((gp[order[index]] == row.vehicle_id) & (gc[order[index]] == row.camera_id))
        ]
        relevant = gp[valid_order] == row.vehicle_id
        positions = np.flatnonzero(relevant) + 1
        precision = np.arange(1, len(positions) + 1, dtype=np.float64) / positions
        aps.append(float(np.mean(precision)))
        top_positions = positions[positions <= 10]
        top_precision = (
            np.arange(1, len(top_positions) + 1, dtype=np.float64) / top_positions
        )
        truncated.append(float(np.sum(top_precision) / len(positions)))
        positive_recall.append(float(len(top_positions) / len(positions)))
        hit_count.append(float(len(top_positions)))
        r1.append(float(positions[0] == 1))
        r5.append(float(positions[0] <= 5))
    return {
        "mAP": float(np.mean(aps)),
        "AP10": float(np.mean(truncated)),
        "positive_recall_at_10": float(np.mean(positive_recall)),
        "mean_positive_images_at_10": float(np.mean(hit_count)),
        "rank1": float(np.mean(r1)),
        "rank5": float(np.mean(r5)),
    }


def _protocol(embeddings: np.ndarray, frame: pd.DataFrame, seed: int):
    protocol = build_identity_stratified_protocol(frame, seed=seed)
    qi, gi = protocol.query_indices, protocol.gallery_indices
    gallery = database_side_augmentation(embeddings[gi], top_k=5, alpha=2.0)
    query = query_expansion(embeddings[qi], gallery, top_k=2, alpha=1.0)
    base = query @ gallery.T
    gallery_similarity = np.clip(gallery @ gallery.T, -1.0, 1.0)
    return base, gallery_similarity, frame.iloc[qi], frame.iloc[gi]


def _evaluate(protocols, config):
    rows = []
    for seed, (base, gallery_similarity, q, g) in protocols.items():
        if config["method"] == "baseline":
            order = np.argsort(-base, axis=1, kind="mergesort")
        else:
            order = _rerank(base, gallery_similarity, **config)
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

    configs = [{"method": "baseline"}]
    for method in ("neighbor_max", "neighbor_mean", "reciprocal_max", "reciprocal_mean"):
        for neighbor_k in (3, 5, 10, 20):
            for blend in (0.10, 0.20, 0.30, 0.40):
                for preserve_top1 in (True, False):
                    configs.append(
                        {
                            "method": method,
                            "neighbor_k": neighbor_k,
                            "blend": blend,
                            "preserve_top1": preserve_top1,
                        }
                    )
    tune = [_evaluate(tune_protocols, config) for config in configs]
    # AP10 is primary; full mAP and Rank1 are deterministic tie-breaks.
    selected = max(tune, key=lambda row: (row["AP10"], row["mAP"], row["rank1"]))
    selected_config = {
        key: selected[key]
        for key in ("method", "neighbor_k", "blend", "preserve_top1")
    }
    confirmation = _evaluate(confirm_protocols, selected_config)
    baseline = _evaluate(confirm_protocols, {"method": "baseline"})
    metric_keys = ("mAP", "AP10", "positive_recall_at_10", "mean_positive_images_at_10", "rank1", "rank5")
    result = {
        "selection_metric": "truncated AP@10",
        "selected_on_tune": selected,
        "confirmation": confirmation,
        "baseline": baseline,
        "delta": {key: confirmation[key] - baseline[key] for key in metric_keys},
        "tune_grid": tune,
    }
    output = Path("outputs/expert_fusion/family_reranking_probe.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected_on_tune", "confirmation", "baseline", "delta")}, indent=2))


if __name__ == "__main__":
    main()
