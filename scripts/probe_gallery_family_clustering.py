"""Recover gallery vehicle families and propagate their strongest evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_camera_transition_prior import _camera_log_likelihood
from scripts.probe_family_reranking import _metrics
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_predicted_camera_family import _load_probability
from scripts.probe_query_family_clustering import _aggregate, _normalize, _prepare
from src.reranking import query_expansion


def _components(protocol, probability, prior, config):
    gallery = protocol[config["embedding"]]
    gp = probability[protocol["gi"]]
    similarity = gallery @ gallery.T + config["pair_alpha"] * (gp @ prior @ gp.T)
    search = similarity.copy()
    np.fill_diagonal(search, -np.inf)
    neighbors = np.argsort(-search, axis=1, kind="mergesort")[:, :config["neighbor_k"]]
    directed = np.zeros_like(search, dtype=bool)
    directed[np.arange(len(search))[:, None], neighbors] = True
    valid = directed & directed.T & (similarity >= config["threshold"])
    first, second = np.where(np.triu(valid, 1))
    order = np.argsort(-similarity[first, second], kind="mergesort")
    camera = gp.argmax(axis=1)
    parent = np.arange(len(gallery))
    groups = {index: [index] for index in range(len(gallery))}

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    for edge in order:
        left, right = root(int(first[edge])), root(int(second[edge]))
        if left == right or len(groups[left]) + len(groups[right]) > 8:
            continue
        merged = groups[left] + groups[right]
        # Gallery contains at most one representative per identity-camera.
        if len(np.unique(camera[merged])) != len(merged):
            continue
        if config["complete_link"]:
            cross = similarity[np.ix_(groups[left], groups[right])]
            if float(cross.min()) < config["threshold"]:
                continue
        parent[right] = left
        groups[left] = merged
        del groups[right]
    return list(groups.values())


def _stats(protocol, components):
    labels = protocol["g"].vehicle_id.to_numpy()
    correct = predicted = grouped = 0
    for component in components:
        size = len(component)
        grouped += size if size > 1 else 0
        predicted += size * (size - 1) // 2
        _, counts = np.unique(labels[component], return_counts=True)
        correct += int(np.sum(counts * (counts - 1) // 2))
    _, counts = np.unique(labels, return_counts=True)
    total = int(np.sum(counts * (counts - 1) // 2))
    return correct / max(predicted, 1), correct / max(total, 1), grouped / len(labels)


def _evaluate(protocol, components, probability, prior, gallery_alpha, propagation):
    qp, gp = probability[protocol["qi"]], probability[protocol["gi"]]
    query = query_expansion(protocol["raw_query"], protocol["gallery"], top_k=2, alpha=1.0)
    score = query @ protocol["gallery"].T + gallery_alpha * (qp @ prior @ gp.T)
    if propagation:
        for component in components:
            if len(component) <= 1:
                continue
            index = np.asarray(component)
            family = score[:, index].max(axis=1, keepdims=True)
            score[:, index] = (1.0 - propagation) * score[:, index] + propagation * family
    order = np.argsort(-score, axis=1, kind="mergesort")
    return _metrics(order, protocol["q"], protocol["g"])


def main():
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    prior, camera_index = _camera_log_likelihood(train, 0.25)
    probability = _load_probability(
        "outputs/expert_fusion/cache/scene_camera_mlp_predictions_val.npz",
        val, camera_index,
    )
    embeddings = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
        allow_pickle=False,
    ).astype(np.float64))
    tune = {seed: _prepare(embeddings, val, seed) for seed in TUNE_SEEDS}
    confirmation = {seed: _prepare(embeddings, val, seed) for seed in CONFIRM_SEEDS}
    configs = []
    for embedding in ("raw_gallery", "gallery"):
        for pair_alpha in (0.03, 0.05, 0.075):
            for neighbor_k in (1, 2):
                for threshold in (0.55, 0.60, 0.65):
                    for complete_link in (False, True):
                        config = {
                            "embedding": embedding,
                            "pair_alpha": pair_alpha,
                            "neighbor_k": neighbor_k,
                            "threshold": threshold,
                            "complete_link": complete_link,
                        }
                        clustered = {
                            seed: _components(protocol, probability, prior, config)
                            for seed, protocol in tune.items()
                        }
                        stats = [_stats(tune[seed], clustered[seed]) for seed in TUNE_SEEDS]
                        for propagation in (0.25, 0.5, 0.75, 1.0):
                            rows = [
                                _evaluate(
                                    tune[seed], clustered[seed], probability, prior,
                                    0.02, propagation,
                                )
                                for seed in TUNE_SEEDS
                            ]
                            configs.append({
                                **config,
                                "gallery_alpha": 0.02,
                                "propagation": propagation,
                                "tune": _aggregate(rows),
                                "pair_precision": float(np.mean([x[0] for x in stats])),
                                "pair_recall": float(np.mean([x[1] for x in stats])),
                                "grouped_fraction": float(np.mean([x[2] for x in stats])),
                            })
        print(f"finished embedding={embedding}", flush=True)
    selected = max(configs, key=lambda row: (row["tune"]["mAP"], row["tune"]["AP10"]))
    component_keys = ("embedding", "pair_alpha", "neighbor_k", "threshold", "complete_link")
    clustered = {
        seed: _components(
            protocol, probability, prior,
            {key: selected[key] for key in component_keys},
        )
        for seed, protocol in confirmation.items()
    }
    confirmed = _aggregate([
        _evaluate(
            confirmation[seed], clustered[seed], probability, prior,
            selected["gallery_alpha"], selected["propagation"],
        )
        for seed in CONFIRM_SEEDS
    ])
    baseline = _aggregate([
        _evaluate(protocol, [], probability, prior, 0.0, 0.0)
        for protocol in confirmation.values()
    ])
    stats = [_stats(confirmation[seed], clustered[seed]) for seed in CONFIRM_SEEDS]
    result = {
        "selected": selected,
        "confirmation": confirmed,
        "baseline_confirmation": baseline,
        "delta": {key: confirmed[key] - baseline[key] for key in baseline},
        "confirmation_pair_precision": float(np.mean([x[0] for x in stats])),
        "confirmation_pair_recall": float(np.mean([x[1] for x in stats])),
        "confirmation_grouped_fraction": float(np.mean([x[2] for x in stats])),
        "top_configs": sorted(configs, key=lambda row: row["tune"]["mAP"], reverse=True)[:20],
    }
    Path("outputs/expert_fusion/gallery_family_clustering.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
