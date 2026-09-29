"""Train a strict static-gallery linker from modern target-domain features.

The model is supervised only by train identities.  At validation/inference it
uses the static gallery alone: no other queries, camera IDs, row order, or test
labels.  It combines current OSNet/DINO similarities, reciprocal ranks,
neighbourhood overlap, and a symmetric local-patch verification score.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.probe_multi_query_aggregation import (
    CONFIRM_SEEDS,
    TUNE_SEEDS,
    _camera_representative_protocol,
)
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_dino_patch_matcher import _load, _patch_features
from scripts.train_streaming_top25_verifier import _normalize
from src.reranking import database_side_augmentation


TRAIN_SEEDS = (1709, 2711, 3907)
PAIR_K = 16
OSNET_TRAIN_PATH = os.environ.get(
    "OSNET_TRAIN_PATH",
    "outputs/expert_fusion/cache/osnet_smoothap_train.npz",
)
OSNET_VAL_PATH = os.environ.get(
    "OSNET_ENSEMBLE_PATH",
    "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
)


def _rank_matrix(similarity):
    order = np.argsort(-similarity, axis=1, kind="stable")
    rank = np.empty_like(order, dtype=np.int16)
    np.put_along_axis(
        rank,
        order,
        np.broadcast_to(np.arange(len(order), dtype=np.int16), order.shape),
        axis=1,
    )
    return order, rank


def _candidate_pairs(similarities):
    n = len(next(iter(similarities.values())))
    encoded, orders, ranks = [], {}, {}
    for name, similarity in similarities.items():
        search = similarity.copy()
        np.fill_diagonal(search, -np.inf)
        order, rank = _rank_matrix(search)
        orders[name], ranks[name] = order, rank
        first = np.repeat(np.arange(n, dtype=np.int64), PAIR_K)
        second = order[:, :PAIR_K].reshape(-1).astype(np.int64)
        low, high = np.minimum(first, second), np.maximum(first, second)
        encoded.append(low * n + high)
    unique = np.unique(np.concatenate(encoded))
    return unique // n, unique % n, orders, ranks


def _overlap(first, second, order):
    output = np.empty((len(first), 4), dtype=np.float32)
    for row, (left, right) in enumerate(zip(first, second, strict=True)):
        output[row] = [
            len(np.intersect1d(order[left, :size], order[right, :size], assume_unique=True))
            / size
            for size in (3, 5, 10, 20)
        ]
    return output


def _build(frame, indices, fused, osnet, dino, tokens, patch_model):
    local = frame.iloc[indices].reset_index(drop=True)
    fused_g, osnet_g, dino_g = fused[indices], osnet[indices], dino[indices]
    dba = database_side_augmentation(fused_g, top_k=5, alpha=2.0)
    similarities = {
        "dba": dba @ dba.T,
        "fused": fused_g @ fused_g.T,
        "osnet": osnet_g @ osnet_g.T,
        "dino": dino_g @ dino_g.T,
    }
    first, second, orders, ranks = _candidate_pairs(similarities)
    blocks = []
    n = len(indices)
    for name, similarity in similarities.items():
        rank = ranks[name]
        blocks.extend((
            similarity[first, second, None],
            (np.minimum(rank[first, second], rank[second, first]) / n)[:, None],
            (np.maximum(rank[first, second], rank[second, first]) / n)[:, None],
        ))
    for name in ("dba", "fused", "osnet", "dino"):
        blocks.append(_overlap(first, second, orders[name]))

    global_first, global_second = indices[first], indices[second]
    forward = _patch_features(
        tokens, fused, osnet, dino, global_first, global_second, batch=2048,
        grid_size=5,
    )
    reverse = _patch_features(
        tokens, fused, osnet, dino, global_second, global_first, batch=2048,
        grid_size=5,
    )
    forward_probability = patch_model.predict_proba(forward)[:, 1]
    reverse_probability = patch_model.predict_proba(reverse)[:, 1]
    local_probability = np.stack((
        np.minimum(forward_probability, reverse_probability),
        np.maximum(forward_probability, reverse_probability),
        (forward_probability + reverse_probability) / 2.0,
        np.abs(forward_probability - reverse_probability),
    ), axis=1).astype(np.float32)
    blocks.append(local_probability)

    aspect = np.log(
        np.maximum(local.w.to_numpy(), 1) / np.maximum(local.h.to_numpy(), 1)
    )
    area = np.log(np.maximum(local.w.to_numpy() * local.h.to_numpy(), 1))
    blocks.append(np.stack((
        np.abs(aspect[first] - aspect[second]),
        np.abs(area[first] - area[second]),
    ), axis=1).astype(np.float32))
    labels = (
        local.vehicle_id.to_numpy()[first] == local.vehicle_id.to_numpy()[second]
    ).astype(np.uint8)
    return {
        "indices": indices,
        "first": first,
        "second": second,
        "features": np.concatenate(blocks, axis=1).astype(np.float32),
        "labels": labels,
        "rank": ranks["dba"],
        "base": dba,
    }


def _affinity(data, probability):
    n = len(data["indices"])
    output = np.full((n, n), -np.inf, dtype=np.float32)
    output[data["first"], data["second"]] = probability
    output[data["second"], data["first"]] = probability
    np.fill_diagonal(output, -np.inf)
    return output


def _neighborhoods(affinity, rank, neighbor_k, reciprocal_rank, threshold):
    order = np.argsort(-affinity, axis=1, kind="stable")
    result = []
    for anchor, candidates in enumerate(order[:, :neighbor_k]):
        valid = [
            int(value) for value in candidates
            if affinity[anchor, value] >= threshold
            and max(rank[anchor, value], rank[value, anchor]) < reciprocal_rank
        ]
        result.append(np.asarray([anchor, *valid], dtype=np.int64))
    return result


def _evaluate(protocols, galleries, probabilities, fused, config):
    rows, precisions, recalls, linked = [], [], [], []
    for protocol, data, probability in zip(protocols, galleries, probabilities, strict=True):
        gallery = fused[protocol["gi"]]
        gallery_dba = database_side_augmentation(gallery, top_k=5, alpha=2.0)
        score = fused[protocol["qi"]] @ gallery_dba.T
        affinity = _affinity(data, probability)
        neighborhoods = _neighborhoods(
            affinity, data["rank"], config["neighbor_k"],
            config["reciprocal_rank"], config["threshold"],
        )
        original = score.copy()
        for target, neighborhood in enumerate(neighborhoods):
            if len(neighborhood) > 1:
                support = original[:, neighborhood].max(axis=1)
                score[:, target] = (
                    (1.0 - config["propagation"]) * original[:, target]
                    + config["propagation"] * support
                )
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))

        selected = []
        for anchor, neighborhood in enumerate(neighborhoods):
            selected.extend((anchor, int(value)) for value in neighborhood[1:])
        if selected:
            first = np.asarray([x[0] for x in selected])
            second = np.asarray([x[1] for x in selected])
            labels = data["labels"]
            lookup = {
                (int(a), int(b)): int(y)
                for a, b, y in zip(data["first"], data["second"], labels, strict=True)
            }
            correct = [lookup.get((min(a, b), max(a, b)), 0) for a, b in selected]
            precisions.append(float(np.mean(correct)))
        else:
            precisions.append(0.0)
        all_positive = max(int(data["labels"].sum()), 1)
        recalls.append(float(sum(correct) / all_positive) if selected else 0.0)
        linked.append(float(np.mean([len(x) > 1 for x in neighborhoods])))
    return {
        **_aggregate(rows),
        "edge_precision": float(np.mean(precisions)),
        "candidate_positive_recall": float(np.mean(recalls)),
        "linked_fraction": float(np.mean(linked)),
    }


def _features(frame, split):
    if split == "train":
        osnet = _normalize(
            np.load(OSNET_TRAIN_PATH, allow_pickle=False).astype(np.float32)
            if str(OSNET_TRAIN_PATH).lower().endswith(".npy")
            else _load(OSNET_TRAIN_PATH, frame, "embeddings")
        )
        dino = _normalize(_load(
            "outputs/expert_fusion/cache/dinov2_vehicle_cls_train.npz", frame, "embeddings"
        ))
        tokens = _load(
            "outputs/expert_fusion/cache/dinov2_vehicle_patches_train.npz", frame, "tokens"
        )
    else:
        osnet = _normalize(np.load(
            OSNET_VAL_PATH,
            allow_pickle=False,
        ))
        dino = _normalize(_load(
            os.environ.get(
                "DINO_PARTS_VAL_PATH",
                "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz",
            ),
            frame,
            "cls",
        ))
        tokens = _load(
            "outputs/expert_fusion/cache/dinov2_vehicle_patches_val.npz", frame, "tokens"
        )
    fused = _normalize(np.concatenate((
        np.sqrt(0.75) * osnet, np.sqrt(0.25) * dino
    ), axis=1))
    return fused, osnet, dino, tokens


def main():
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_features, val_features = _features(train, "train"), _features(val, "val")
    patch_model = joblib.load(os.environ.get(
        "DINO_PATCH_MODEL", "weights/dino_patch_matcher.joblib"
    ))
    train_data = []
    for seed in TRAIN_SEEDS:
        _, gallery = _camera_representative_protocol(train, seed)
        train_data.append(_build(train, gallery, *train_features, patch_model))
        print(f"built train gallery {seed}: {len(train_data[-1]['labels'])} pairs", flush=True)
    x = np.concatenate([value["features"] for value in train_data])
    y = np.concatenate([value["labels"] for value in train_data])
    positive = max(int(y.sum()), 1)
    sample_weight = np.where(y > 0, min((len(y) - positive) / positive, 4.0), 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=60,
        l2_regularization=5.0,
        random_state=92731,
    )
    model.fit(x, y, sample_weight=sample_weight)
    print(json.dumps({
        "pairs": len(y), "positives": int(y.sum()), "features": x.shape[1],
        "iterations": int(model.n_iter_),
    }), flush=True)

    tune = _protocols(val, TUNE_SEEDS, {"base": val_features[0]})
    confirm = _protocols(val, CONFIRM_SEEDS, {"base": val_features[0]})
    gallery_data = [
        _build(val, protocol["gi"], *val_features, patch_model)
        for protocol in tune + confirm
    ]
    predictions = [model.predict_proba(value["features"])[:, 1] for value in gallery_data]
    tune_data, confirm_data = gallery_data[:len(tune)], gallery_data[len(tune):]
    tune_prediction = predictions[:len(tune)]
    confirm_prediction = predictions[len(tune):]
    grid = []
    for neighbor_k in (1, 2, 3):
        for reciprocal_rank in (2, 4, 8, 16):
            for threshold in (0.30, 0.50, 0.70, 0.80, 0.90, 0.95):
                for propagation in (0.10, 0.20, 0.30, 0.50):
                    config = {
                        "neighbor_k": neighbor_k,
                        "reciprocal_rank": reciprocal_rank,
                        "threshold": threshold,
                        "propagation": propagation,
                    }
                    grid.append({**config, **_evaluate(
                        tune, tune_data, tune_prediction, val_features[0], config
                    )})
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    config = {key: selected[key] for key in (
        "neighbor_k", "reciprocal_rank", "threshold", "propagation"
    )}
    confirmation = _evaluate(
        confirm, confirm_data, confirm_prediction, val_features[0], config
    )
    baseline_config = {
        "neighbor_k": 1, "reciprocal_rank": 1, "threshold": 2.0,
        "propagation": 0.0,
    }
    baseline = _evaluate(
        confirm, confirm_data, confirm_prediction, val_features[0], baseline_config
    )
    report = {
        "design": "train-only modern static-gallery linker; no query-query/camera/order",
        "train_pairs": len(y),
        "train_positives": int(y.sum()),
        "feature_count": int(x.shape[1]),
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {key: confirmation[key] - baseline[key]
                  for key in ("mAP@10", "Rank-1", "Rank-5")},
        "top_tune": sorted(
            grid, key=lambda row: (row["mAP@10"], row["Rank-1"]), reverse=True
        )[:30],
    }
    suffix = os.environ.get("LINKER_EXPERIMENT_SUFFIX", "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    Path(f"outputs/retrieval_v2/modern_gallery_linker{suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(model, f"weights/modern_gallery_linker{suffix}.joblib")
    # Cache validation affinities only for strict ablation/integration.  The
    # production path rebuilds the same gallery-only features on its gallery.
    np.savez_compressed(
        f"outputs/retrieval_v2/modern_gallery_linker_val{suffix}.npz",
        **{
            f"affinity_{seed}": _affinity(data, probability)
            for seed, data, probability in zip(
                TUNE_SEEDS + CONFIRM_SEEDS, gallery_data, predictions, strict=True
            )
        },
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
