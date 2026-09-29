"""Rebuild a rule-compliant streaming baseline with the official mAP@10."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_multi_query_aggregation import (
    CONFIRM_SEEDS,
    TUNE_SEEDS,
    _camera_representative_protocol,
)


def _normalize(values):
    values = values.astype(np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return _normalize(archive["embeddings"])


def _rank_rows(values):
    order = np.argsort(-values, axis=1, kind="stable")
    rank = np.empty_like(values, dtype=np.float32)
    score = np.linspace(1.0, 0.0, values.shape[1], dtype=np.float32)
    np.put_along_axis(rank, order, np.broadcast_to(score, order.shape), axis=1)
    return rank


def _official(order, q, g):
    gp, gc = g.vehicle_id.to_numpy(), g.camera_id.to_numpy()
    aps, rank1, rank5 = [], [], []
    for ranking, query in zip(order, q.itertuples(index=False), strict=True):
        positive = (gp == query.vehicle_id) & (gc != query.camera_id)
        n_positive = int(positive.sum())
        if n_positive == 0:
            continue
        junk = (gp == query.vehicle_id) & (gc == query.camera_id)
        clean = ranking[~junk[ranking]][:10]
        relevant = positive[clean]
        cumulative = np.cumsum(relevant)
        precision = cumulative / np.arange(1, len(clean) + 1)
        aps.append(float((precision * relevant).sum() / min(n_positive, 10)))
        rank1.append(bool(relevant[:1].any()))
        rank5.append(bool(relevant[:5].any()))
    return {"mAP@10": float(np.mean(aps)), "Rank-1": float(np.mean(rank1)),
            "Rank-5": float(np.mean(rank5)), "n_scored": len(aps)}


def _official_from_scores(scores, q, g):
    # The protocol has at most one junk gallery row per query.  Keeping the
    # best 16 is therefore sufficient to recover the official valid top-10.
    count = min(16, scores.shape[1])
    candidate = np.argpartition(-scores, count - 1, axis=1)[:, :count]
    candidate_score = np.take_along_axis(scores, candidate, axis=1)
    local_order = np.argsort(-candidate_score, axis=1, kind="stable")
    return _official(np.take_along_axis(candidate, local_order, axis=1), q, g)


def _aggregate(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ("mAP@10", "Rank-1", "Rank-5", "n_scored")}


def _protocols(frame, seeds, embeddings):
    output = []
    for seed in seeds:
        query_indices, gallery_indices = _camera_representative_protocol(frame, seed)
        scores = {name: value[query_indices] @ value[gallery_indices].T
                  for name, value in embeddings.items()}
        output.append({"q": frame.iloc[query_indices], "g": frame.iloc[gallery_indices],
                       "qi": query_indices, "gi": gallery_indices,
                       "scores": scores,
                       "ranks": {name: _rank_rows(value) for name, value in scores.items()}})
    return output


def _evaluate(protocols, weights, propagation=None):
    rows = []
    for protocol in protocols:
        total = sum(weight * protocol["ranks"][name] for name, weight in weights.items())
        if propagation is not None:
            total = _propagate(total, protocol, weights, **propagation)
        rows.append(_official_from_scores(total, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def _propagate(score, protocol, weights, neighbor_k, blend, method):
    if "gallery_neighbors" not in protocol:
        gallery_score = sum(weight * protocol["gallery_similarity"][name]
                            for name, weight in weights.items())
        np.fill_diagonal(gallery_score, -np.inf)
        protocol["gallery_neighbors"] = np.argpartition(
            -gallery_score, 4, axis=1)[:, :5]
        local_score = np.take_along_axis(
            gallery_score, protocol["gallery_neighbors"], axis=1)
        local_order = np.argsort(-local_score, axis=1, kind="stable")
        protocol["gallery_neighbors"] = np.take_along_axis(
            protocol["gallery_neighbors"], local_order, axis=1)
    neighbors = protocol["gallery_neighbors"][:, :neighbor_k]
    neighbor_scores = score[:, neighbors]
    support = neighbor_scores.max(2) if method == "max" else neighbor_scores.mean(2)
    return score + blend * _rank_rows(support)


def _attach_gallery_similarity(protocols, embeddings):
    for protocol in protocols:
        protocol["gallery_similarity"] = {
            name: value[protocol["gi"]] @ value[protocol["gi"]].T
            for name, value in embeddings.items()
        }


def _oracle(protocol, weights, k=10):
    total = sum(weight * protocol["ranks"][name] for name, weight in weights.items())
    order = np.argsort(-total, axis=1, kind="stable")
    gp, gc = protocol["g"].vehicle_id.to_numpy(), protocol["g"].camera_id.to_numpy()
    output = order.copy()
    for index, query in enumerate(protocol["q"].itertuples(index=False)):
        junk = (gp == query.vehicle_id) & (gc == query.camera_id)
        valid = order[index][~junk[order[index]]]
        head, tail = valid[:k], valid[k:]
        positive = gp == query.vehicle_id
        output[index] = np.concatenate((head[positive[head]], head[~positive[head]], tail,
                                        order[index][junk[order[index]]]))[:len(order[index])]
    return _official(output, protocol["q"], protocol["g"])


def main():
    frame = pd.read_csv("splits/val.csv")
    embeddings = {
        "smooth_ensemble": _normalize(np.load(
            "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy")),
        "hard_smoothap": _load_npz(
            "outputs/expert_fusion/cache/osnet_target_finetuned_val_hard_smoothap.npz"),
        "hard_metric2": _load_npz(
            "outputs/expert_fusion/cache/osnet_target_finetuned_val_hard_metric_stage2.npz"),
        "convnext": _load_npz(
            "outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz"),
        "dinov2_small": _load_npz("outputs/expert_fusion/cache/dinov2_val.npz"),
        "dinov2_b": _load_npz("outputs/expert_fusion/cache/ames_dinov2_val.npz"),
    }
    tune = _protocols(frame, TUNE_SEEDS, embeddings)
    confirm = _protocols(frame, CONFIRM_SEEDS, embeddings)
    _attach_gallery_similarity(tune + confirm, embeddings)

    singles = {name: _evaluate(tune, {name: 1.0}) for name in embeddings}
    print(json.dumps({"single_models_tune": singles}, indent=2), flush=True)
    base_name = max(singles, key=lambda name: singles[name]["mAP@10"])
    # Greedy forward rank fusion: it tests the same meaningful weights without
    # repeatedly sorting every source matrix for a 5^4 brute-force grid.
    weights = {base_name: 1.0}
    grid = [{"weights": dict(weights), **_evaluate(tune, weights)}]
    for name in ("hard_smoothap", "convnext", "dinov2_small", "dinov2_b"):
        candidates = []
        for value in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5):
            trial = dict(weights)
            trial[name] = trial.get(name, 0.0) + value
            metrics = _evaluate(tune, trial)
            candidates.append({"weights": trial, **metrics})
        best = max(candidates, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
        weights = best["weights"]
        grid.extend(candidates)
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    confirmation = _evaluate(confirm, selected["weights"])
    print(json.dumps({"selected_fusion_tune": selected,
                      "fusion_confirmation": confirmation}, indent=2), flush=True)

    propagation_grid = []
    for neighbor_k, blend, method in itertools.product((1, 2, 3, 5), (0.05, 0.1, 0.2, 0.35), ("max", "mean")):
        config = {"neighbor_k": neighbor_k, "blend": blend, "method": method}
        metrics = _evaluate(tune, selected["weights"], config)
        propagation_grid.append({**config, **metrics})
    selected_propagation = max(propagation_grid,
                               key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    prop_config = {key: selected_propagation[key] for key in ("neighbor_k", "blend", "method")}
    propagation_confirmation = _evaluate(confirm, selected["weights"], prop_config)
    oracle = {str(k): _aggregate([_oracle(protocol, selected["weights"], k)
                                  for protocol in confirm])
              for k in (10, 25, 50)}
    report = {
        "protocol": "streaming: one current query plus static gallery; official evaluate.py mAP@10",
        "single_models_tune": singles,
        "selected_fusion_tune": selected,
        "fusion_confirmation": confirmation,
        "selected_gallery_propagation_tune": selected_propagation,
        "gallery_propagation_confirmation": propagation_confirmation,
        "official_top10_oracle_confirmation": oracle,
    }
    Path("outputs/expert_fusion/streaming_official.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
