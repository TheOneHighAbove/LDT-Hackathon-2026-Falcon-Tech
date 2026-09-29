"""Official streaming mAP@10 for gallery-only vehicle-family propagation."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official, _protocols


_GROUP_CACHE = {}


def _groups(similarity, neighbor_k, threshold, complete_link):
    search = similarity.copy()
    np.fill_diagonal(search, -np.inf)
    neighbors = np.argsort(-search, axis=1, kind="stable")[:, :neighbor_k]
    directed = np.zeros_like(search, dtype=bool)
    directed[np.arange(len(search))[:, None], neighbors] = True
    valid = directed & directed.T & (similarity >= threshold)
    left, right = np.where(np.triu(valid, 1))
    edge_order = np.argsort(-similarity[left, right], kind="stable")
    parent = np.arange(len(similarity))
    components = {index: [index] for index in range(len(similarity))}

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    for edge in edge_order:
        first, second = root(int(left[edge])), root(int(right[edge]))
        if first == second or len(components[first]) + len(components[second]) > 8:
            continue
        if complete_link:
            cross = similarity[np.ix_(components[first], components[second])]
            if float(cross.min()) < threshold:
                continue
        parent[second] = first
        components[first].extend(components.pop(second))
    return [np.asarray(value) for value in components.values() if len(value) > 1]


def _order(protocol, embedding, config):
    query = embedding[protocol["qi"]]
    gallery = embedding[protocol["gi"]]
    key = (id(protocol), config["neighbor_k"], config["threshold"], config["complete_link"])
    groups = _GROUP_CACHE.get(key)
    if groups is None:
        similarity = gallery @ gallery.T
        groups = _groups(similarity, config["neighbor_k"], config["threshold"],
                         config["complete_link"])
        _GROUP_CACHE[key] = groups
    score = query @ gallery.T
    for group in groups:
        family = score[:, group].max(axis=1, keepdims=True)
        score[:, group] = ((1.0 - config["propagation"]) * score[:, group]
                           + config["propagation"] * family)
    return np.argsort(-score, axis=1, kind="stable"), groups


def _evaluate(protocols, embedding, config):
    rows, stats = [], []
    for protocol in protocols:
        order, groups = _order(protocol, embedding, config)
        rows.append(_official(order, protocol["q"], protocol["g"]))
        labels = protocol["g"].vehicle_id.to_numpy()
        predicted = correct = 0
        for group in groups:
            predicted += len(group) * (len(group) - 1) // 2
            _, counts = np.unique(labels[group], return_counts=True)
            correct += int(np.sum(counts * (counts - 1) // 2))
        stats.append((correct / max(predicted, 1), sum(map(len, groups)) / len(labels)))
    return {**_aggregate(rows), "group_pair_precision": float(np.mean([x[0] for x in stats])),
            "grouped_fraction": float(np.mean([x[1] for x in stats]))}


def main():
    frame = pd.read_csv("splits/val.csv")
    embedding = np.load("outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy")
    embedding = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    protocols = [_protocols(frame, seeds, {"base": embedding})
                 for seeds in (TUNE_SEEDS, CONFIRM_SEEDS)]
    baseline = {"neighbor_k": 1, "threshold": 2.0, "complete_link": True,
                "propagation": 0.0}
    baseline_tune = _evaluate(protocols[0], embedding, baseline)
    grid = []
    for neighbor_k, threshold, complete_link, propagation in itertools.product(
            (1, 2, 3), (0.50, 0.55, 0.60, 0.65, 0.70), (False, True),
            (0.10, 0.25, 0.50, 0.75, 1.0)):
        config = {"neighbor_k": neighbor_k, "threshold": threshold,
                  "complete_link": complete_link, "propagation": propagation}
        grid.append({**config, **_evaluate(protocols[0], embedding, config)})
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    config = {key: selected[key] for key in
              ("neighbor_k", "threshold", "complete_link", "propagation")}
    confirmation = _evaluate(protocols[1], embedding, config)
    baseline_confirmation = _evaluate(protocols[1], embedding, baseline)
    report = {"design": "current query plus static gallery groups; no other queries or camera features",
              "baseline_tune": baseline_tune, "selected_tune": selected,
              "baseline_confirmation": baseline_confirmation,
              "confirmation": confirmation,
              "delta": {key: confirmation[key] - baseline_confirmation[key]
                        for key in ("mAP@10", "Rank-1", "Rank-5")}}
    Path("outputs/expert_fusion/gallery_groups_official.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
