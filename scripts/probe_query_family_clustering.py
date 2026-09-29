"""Recover latent query families and rank gallery from family prototypes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_family_reranking import _metrics
from scripts.probe_multi_query_aggregation import (
    CONFIRM_SEEDS,
    TUNE_SEEDS,
    _camera_representative_protocol,
)
from src.reranking import database_side_augmentation, query_expansion


def _normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _prepare(embeddings, frame, seed):
    qi, gi = _camera_representative_protocol(frame, seed)
    raw_query, raw_gallery = embeddings[qi], embeddings[gi]
    gallery = database_side_augmentation(raw_gallery, top_k=5, alpha=2.0)
    qsim = np.clip(raw_query @ raw_query.T, -1.0, 1.0)
    search = qsim.copy()
    np.fill_diagonal(search, -np.inf)
    neighbors = np.argsort(-search, axis=1, kind="mergesort")[:, :8]
    gallery_score = raw_query @ gallery.T
    gallery_top10 = np.argsort(-gallery_score, axis=1, kind="mergesort")[:, :10]
    return {
        "qi": qi,
        "gi": gi,
        "raw_query": raw_query,
        "raw_gallery": raw_gallery,
        "gallery": gallery,
        "qsim": qsim,
        "neighbors": neighbors,
        "gallery_top10": gallery_top10,
        "q": frame.iloc[qi].reset_index(drop=True),
        "g": frame.iloc[gi].reset_index(drop=True),
    }


def _components(protocol, *, neighbor_k, threshold, overlap_min, complete_link):
    similarity = protocol["qsim"]
    neighbors = protocol["neighbors"][:, :neighbor_k]
    count = len(similarity)
    directed = np.zeros((count, count), dtype=bool)
    rows = np.arange(count)[:, None]
    directed[rows, neighbors] = True
    mutual = directed & directed.T
    first, second = np.triu_indices(count, 1)
    valid = mutual[first, second] & (similarity[first, second] >= threshold)
    first, second = first[valid], second[valid]
    if overlap_min:
        top = protocol["gallery_top10"]
        overlap = np.asarray([
            len(np.intersect1d(top[a], top[b], assume_unique=False))
            for a, b in zip(first, second, strict=True)
        ])
        keep = overlap >= overlap_min
        first, second = first[keep], second[keep]
    edge_score = similarity[first, second]
    order = np.argsort(-edge_score, kind="mergesort")

    parent = np.arange(count)
    groups = {index: [index] for index in range(count)}

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    for edge_index in order:
        left, right = root(int(first[edge_index])), root(int(second[edge_index]))
        if left == right:
            continue
        merged = groups[left] + groups[right]
        if len(merged) > 8:
            continue
        if complete_link:
            cross = similarity[np.ix_(groups[left], groups[right])]
            if float(cross.min()) < threshold:
                continue
        parent[right] = left
        groups[left] = merged
        del groups[right]
    return list(groups.values())


def _cluster_stats(components, query_meta):
    labels = query_meta.vehicle_id.to_numpy()
    predicted_pairs = correct_pairs = 0
    grouped = 0
    for component in components:
        size = len(component)
        if size > 1:
            grouped += size
        predicted_pairs += size * (size - 1) // 2
        values = labels[component]
        _, counts = np.unique(values, return_counts=True)
        correct_pairs += int(np.sum(counts * (counts - 1) // 2))
    _, all_counts = np.unique(labels, return_counts=True)
    true_pairs = int(np.sum(all_counts * (all_counts - 1) // 2))
    return {
        "pair_precision": correct_pairs / max(predicted_pairs, 1),
        "pair_recall": correct_pairs / max(true_pairs, 1),
        "grouped_fraction": grouped / len(labels),
        "clusters": len(components),
        "max_cluster": max(map(len, components)),
    }


def _evaluate(protocol, components, self_weight):
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
    order = np.argsort(-score, axis=1, kind="mergesort")
    return _metrics(order, protocol["q"], protocol["g"])


def _aggregate(rows):
    keys = ("mAP", "AP10", "positive_recall_at_10", "rank1", "rank5")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    frame = pd.read_csv("splits/val.csv")
    embeddings = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
        allow_pickle=False,
    ).astype(np.float64))
    tune = {seed: _prepare(embeddings, frame, seed) for seed in TUNE_SEEDS}
    confirmation = {seed: _prepare(embeddings, frame, seed) for seed in CONFIRM_SEEDS}

    configs = []
    for neighbor_k in (2, 3, 5):
        for threshold in (0.55, 0.65, 0.75):
            for overlap_min in (0, 1, 2):
                for complete_link in (False, True):
                    clustered = {
                        seed: _components(
                            protocol,
                            neighbor_k=neighbor_k,
                            threshold=threshold,
                            overlap_min=overlap_min,
                            complete_link=complete_link,
                        )
                        for seed, protocol in tune.items()
                    }
                    for self_weight in (0.0, 1.0, 2.0, 4.0):
                        rows = [
                            _evaluate(tune[seed], clustered[seed], self_weight)
                            for seed in TUNE_SEEDS
                        ]
                        stats = [
                            _cluster_stats(clustered[seed], tune[seed]["q"])
                            for seed in TUNE_SEEDS
                        ]
                        configs.append({
                            "neighbor_k": neighbor_k,
                            "threshold": threshold,
                            "overlap_min": overlap_min,
                            "complete_link": complete_link,
                            "self_weight": self_weight,
                            "tune": _aggregate(rows),
                            "tune_pair_precision": float(np.mean([x["pair_precision"] for x in stats])),
                            "tune_pair_recall": float(np.mean([x["pair_recall"] for x in stats])),
                            "tune_grouped_fraction": float(np.mean([x["grouped_fraction"] for x in stats])),
                        })
    selected = max(configs, key=lambda row: (row["tune"]["mAP"], row["tune"]["AP10"]))
    cluster_keys = ("neighbor_k", "threshold", "overlap_min", "complete_link")
    confirm_components = {
        seed: _components(
            protocol, **{key: selected[key] for key in cluster_keys}
        )
        for seed, protocol in confirmation.items()
    }
    confirmation_rows = [
        _evaluate(confirmation[seed], confirm_components[seed], selected["self_weight"])
        for seed in CONFIRM_SEEDS
    ]
    baseline_rows = []
    for protocol in confirmation.values():
        query = query_expansion(
            protocol["raw_query"], protocol["gallery"], top_k=2, alpha=1.0
        )
        baseline_rows.append(_metrics(
            np.argsort(-(query @ protocol["gallery"].T), axis=1, kind="mergesort"),
            protocol["q"], protocol["g"],
        ))
    confirmation_stats = [
        _cluster_stats(confirm_components[seed], confirmation[seed]["q"])
        for seed in CONFIRM_SEEDS
    ]
    baseline = _aggregate(baseline_rows)
    confirmed = _aggregate(confirmation_rows)
    result = {
        "selected": selected,
        "confirmation": confirmed,
        "baseline_confirmation": baseline,
        "delta": {key: confirmed[key] - baseline[key] for key in baseline},
        "confirmation_pair_precision": float(np.mean([x["pair_precision"] for x in confirmation_stats])),
        "confirmation_pair_recall": float(np.mean([x["pair_recall"] for x in confirmation_stats])),
        "confirmation_grouped_fraction": float(np.mean([x["grouped_fraction"] for x in confirmation_stats])),
        "top_configs": sorted(configs, key=lambda row: row["tune"]["mAP"], reverse=True)[:20],
    }
    Path("outputs/expert_fusion/query_family_clustering.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
