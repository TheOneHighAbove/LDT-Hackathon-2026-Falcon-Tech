"""Train an unseen-ID query-pair model and use it for family aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.probe_family_reranking import _metrics
from scripts.probe_multi_query_aggregation import (
    CONFIRM_SEEDS,
    TUNE_SEEDS,
    _camera_representative_protocol,
)
from scripts.probe_pair_verifier import _normalize
from src.expert_fusion import weighted_concatenate
from src.reranking import database_side_augmentation, query_expansion


TRAIN_SEEDS = (9109, 11213)
PAIR_K = 12


def _aligned(path, frame):
    with np.load(path, allow_pickle=False) as archive:
        ids = frame.image_id.astype(str).to_numpy(dtype=np.str_)
        if not np.array_equal(ids, archive["image_ids"]):
            raise RuntimeError(f"unaligned cache: {path}")
        return _normalize(archive["embeddings"].astype(np.float32))


def _candidate_pairs(similarity, k=PAIR_K):
    search = similarity.copy()
    np.fill_diagonal(search, -np.inf)
    neighbors = np.argsort(-search, axis=1, kind="mergesort")[:, :k]
    rank = np.full((len(search), len(search)), k + 1, dtype=np.int16)
    rank[np.arange(len(search))[:, None], neighbors] = np.arange(1, k + 1)
    first = np.repeat(np.arange(len(search)), k)
    second = neighbors.reshape(-1)
    low, high = np.minimum(first, second), np.maximum(first, second)
    encoded = low.astype(np.int64) * len(search) + high
    unique = np.unique(encoded)
    return unique // len(search), unique % len(search), rank


def _pair_features(first, second, rank, fused_sim, conv_sim, osnet_sim, qg, frame):
    top = np.argsort(-qg, axis=1, kind="mergesort")[:, :20]
    qg_norm = _normalize(qg)
    max_score = qg.max(axis=1)
    top2 = np.partition(qg, -2, axis=1)[:, -2:]
    margin = top2.max(axis=1) - top2.min(axis=1)
    features = []
    for left, right in zip(first, second, strict=True):
        overlaps = [
            len(np.intersect1d(top[left, :size], top[right, :size])) / size
            for size in (3, 5, 10, 20)
        ]
        shared_support = float(np.max(np.minimum(qg[left], qg[right])))
        features.append([
            fused_sim[left, right],
            conv_sim[left, right],
            osnet_sim[left, right],
            abs(conv_sim[left, right] - osnet_sim[left, right]),
            float(top[left, 0] == top[right, 0]),
            *overlaps,
            float(qg_norm[left] @ qg_norm[right]),
            shared_support,
            abs(max_score[left] - max_score[right]),
            min(margin[left], margin[right]),
            min(rank[left, right], rank[right, left]) / (PAIR_K + 1),
            max(rank[left, right], rank[right, left]) / (PAIR_K + 1),
            abs(np.log(frame.iloc[left].w / frame.iloc[left].h) - np.log(frame.iloc[right].w / frame.iloc[right].h)),
            abs(np.log(frame.iloc[left].w * frame.iloc[left].h) - np.log(frame.iloc[right].w * frame.iloc[right].h)),
        ])
    return np.asarray(features, dtype=np.float32)


def _prepare(frame, conv, osnet, seed):
    fused = weighted_concatenate(conv, osnet, primary_weight=0.25).astype(np.float64)
    qi, gi = _camera_representative_protocol(frame, seed)
    q, g = frame.iloc[qi].reset_index(drop=True), frame.iloc[gi].reset_index(drop=True)
    raw_query, raw_gallery = fused[qi], fused[gi]
    gallery = database_side_augmentation(raw_gallery, top_k=5, alpha=2.0)
    fused_sim = np.clip(raw_query @ raw_query.T, -1.0, 1.0)
    conv_sim = np.clip(conv[qi] @ conv[qi].T, -1.0, 1.0)
    osnet_sim = np.clip(osnet[qi] @ osnet[qi].T, -1.0, 1.0)
    qg = raw_query @ gallery.T
    first, second, rank = _candidate_pairs(fused_sim)
    features = _pair_features(
        first, second, rank, fused_sim, conv_sim, osnet_sim, qg, q
    )
    labels = (q.vehicle_id.to_numpy()[first] == q.vehicle_id.to_numpy()[second]).astype(np.uint8)
    return {
        "qi": qi,
        "gi": gi,
        "raw_query": raw_query,
        "gallery": gallery,
        "q": q,
        "g": g,
        "first": first,
        "second": second,
        "rank": rank,
        "features": features,
        "labels": labels,
        "fused_sim": fused_sim,
    }


def _clusters(protocol, probability, threshold, mutual_only, complete_link):
    count = len(protocol["q"])
    parent = np.arange(count)
    groups = {index: [index] for index in range(count)}

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    order = np.argsort(-probability, kind="mergesort")
    for edge in order:
        if probability[edge] < threshold:
            break
        a, b = int(protocol["first"][edge]), int(protocol["second"][edge])
        if mutual_only and max(protocol["rank"][a, b], protocol["rank"][b, a]) > PAIR_K:
            continue
        left, right = root(a), root(b)
        if left == right or len(groups[left]) + len(groups[right]) > 8:
            continue
        if complete_link:
            cross = protocol["fused_sim"][np.ix_(groups[left], groups[right])]
            if float(cross.min()) < 0.45:
                continue
        parent[right] = left
        groups[left] += groups[right]
        del groups[right]
    return list(groups.values())


def _evaluate(protocol, components, self_weight):
    raw = protocol["raw_query"]
    augmented = raw.copy()
    for component in components:
        if len(component) <= 1:
            continue
        index = np.asarray(component)
        prototype = _normalize(raw[index].mean(axis=0, keepdims=True))[0]
        augmented[index] = _normalize(self_weight * raw[index] + prototype)
    query = query_expansion(augmented, protocol["gallery"], top_k=2, alpha=1.0)
    order = np.argsort(-(query @ protocol["gallery"].T), axis=1, kind="mergesort")
    return _metrics(order, protocol["q"], protocol["g"])


def _stats(protocol, components):
    labels = protocol["q"].vehicle_id.to_numpy()
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


def _aggregate(rows):
    keys = ("mAP", "AP10", "positive_recall_at_10", "rank1", "rank5")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    train_frame = pd.read_csv("splits/train.csv")
    train_conv = _aligned("outputs/expert_fusion/cache/convnext_train_verifier.npz", train_frame)
    train_osnet = _aligned("outputs/expert_fusion/cache/osnet_smoothap_train.npz", train_frame)
    train_protocols = [
        _prepare(train_frame, train_conv, train_osnet, seed) for seed in TRAIN_SEEDS
    ]
    x = np.concatenate([protocol["features"] for protocol in train_protocols])
    y = np.concatenate([protocol["labels"] for protocol in train_protocols])
    positives = max(int(y.sum()), 1)
    sample_weight = np.where(y > 0, (len(y) - positives) / positives, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=7301,
    )
    model.fit(x, y, sample_weight=sample_weight)
    print(f"trained pairs={len(y)} positives={int(y.sum())}", flush=True)

    val_frame = pd.read_csv("splits/val.csv")
    val_conv = _aligned("outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz", val_frame)
    val_osnet = _aligned("outputs/expert_fusion/cache/osnet_target_finetuned_val_hard_smoothap.npz", val_frame)
    tune = {seed: _prepare(val_frame, val_conv, val_osnet, seed) for seed in TUNE_SEEDS}
    confirmation = {seed: _prepare(val_frame, val_conv, val_osnet, seed) for seed in CONFIRM_SEEDS}
    tune_probability = {
        seed: model.predict_proba(protocol["features"])[:, 1]
        for seed, protocol in tune.items()
    }
    configs = []
    for threshold in (0.50, 0.65, 0.80, 0.90, 0.96):
        for mutual_only in (False, True):
            for complete_link in (False, True):
                components = {
                    seed: _clusters(tune[seed], tune_probability[seed], threshold, mutual_only, complete_link)
                    for seed in TUNE_SEEDS
                }
                for self_weight in (0.0, 1.0, 2.0, 4.0):
                    rows = [_evaluate(tune[seed], components[seed], self_weight) for seed in TUNE_SEEDS]
                    stats = [_stats(tune[seed], components[seed]) for seed in TUNE_SEEDS]
                    configs.append({
                        "threshold": threshold,
                        "mutual_only": mutual_only,
                        "complete_link": complete_link,
                        "self_weight": self_weight,
                        "tune": _aggregate(rows),
                        "pair_precision": float(np.mean([item[0] for item in stats])),
                        "pair_recall": float(np.mean([item[1] for item in stats])),
                        "grouped_fraction": float(np.mean([item[2] for item in stats])),
                    })
    selected = max(configs, key=lambda row: (row["tune"]["mAP"], row["tune"]["AP10"]))
    confirm_probability = {
        seed: model.predict_proba(protocol["features"])[:, 1]
        for seed, protocol in confirmation.items()
    }
    cluster_keys = ("threshold", "mutual_only", "complete_link")
    components = {
        seed: _clusters(
            confirmation[seed], confirm_probability[seed],
            **{key: selected[key] for key in cluster_keys},
        )
        for seed in CONFIRM_SEEDS
    }
    rows = [
        _evaluate(confirmation[seed], components[seed], selected["self_weight"])
        for seed in CONFIRM_SEEDS
    ]
    baseline_rows = [
        _evaluate(protocol, [[index] for index in range(len(protocol["q"]))], 1.0)
        for protocol in confirmation.values()
    ]
    stats = [_stats(confirmation[seed], components[seed]) for seed in CONFIRM_SEEDS]
    confirmed, baseline = _aggregate(rows), _aggregate(baseline_rows)
    result = {
        "train_pairs": len(y),
        "train_positive_pairs": int(y.sum()),
        "selected": selected,
        "confirmation": confirmed,
        "baseline_confirmation": baseline,
        "delta": {key: confirmed[key] - baseline[key] for key in baseline},
        "confirmation_pair_precision": float(np.mean([item[0] for item in stats])),
        "confirmation_pair_recall": float(np.mean([item[1] for item in stats])),
        "confirmation_grouped_fraction": float(np.mean([item[2] for item in stats])),
        "top_configs": sorted(configs, key=lambda row: row["tune"]["mAP"], reverse=True)[:20],
    }
    Path("outputs/expert_fusion/query_family_learned.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(model, "weights/query_family_hgb.joblib")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
