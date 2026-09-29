"""Identity-disjoint OOF probe of a lightweight top-50 pair verifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.data import crop_vehicle
from src.postprocess_eval import build_identity_stratified_protocol
from src.reranking import database_side_augmentation, query_expansion


TUNE_SEEDS = (1337, 2027, 3407, 4517, 7919)
CONFIRM_SEEDS = (101, 211, 307, 401, 503)
TOP_K = 50
FEATURE_NAMES = (
    "processed_fused",
    "raw_fused",
    "convnext",
    "osnet",
    "expert_disagreement",
    "aspect_distance",
    "area_distance",
    "color_similarity",
    "local_aligned",
    "local_chamfer",
    "view_probability_overlap",
    "expected_circular_view_distance",
    "base_gap_from_top1",
    "normalized_rank",
)


def _normalize(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def _color_descriptors(frame: pd.DataFrame) -> np.ndarray:
    cache = Path("outputs/expert_fusion/cache/val_color_histograms.npz")
    ids = frame.image_id.astype(str).to_numpy(dtype=np.str_)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as archive:
            if np.array_equal(ids, archive["image_ids"]):
                return archive["descriptors"].astype(np.float32)
    descriptors = []
    for row in frame.itertuples(index=False):
        path = Path("dataset/images") / f"{row.image_id}.jpg"
        with Image.open(path) as image:
            crop = crop_vehicle(image, (row.x, row.y, row.w, row.h), padding=0.0)
            array = np.asarray(crop.resize((64, 64)).convert("HSV"), dtype=np.uint8)
        regions = (
            array,
            array[:32, :32], array[:32, 32:],
            array[32:, :32], array[32:, 32:],
        )
        values = []
        for region in regions:
            for channel, bins in ((0, 16), (1, 8), (2, 8)):
                hist = np.histogram(region[..., channel], bins=bins, range=(0, 256))[0]
                hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
                values.append(np.sqrt(hist))
        descriptor = np.concatenate(values)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
        descriptors.append(descriptor)
    result = np.stack(descriptors).astype(np.float16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, image_ids=ids, descriptors=result)
    return result.astype(np.float32)


def _load_parts(frame: pd.DataFrame) -> np.ndarray:
    path = Path("outputs/expert_fusion/cache/convnext_val_parts_3x3.npz")
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(
            archive["image_ids"], frame.image_id.astype(str).to_numpy(dtype=np.str_)
        ):
            raise RuntimeError("part cache does not match validation rows")
        return archive["parts"].astype(np.float32)


def _local(q_parts: np.ndarray, g_parts: np.ndarray, candidates: np.ndarray):
    selected = g_parts[candidates]
    similarity = np.einsum("qid,qkjd->qkij", q_parts, selected, optimize=True)
    aligned = np.diagonal(similarity, axis1=2, axis2=3).mean(axis=2)
    chamfer = 0.5 * (
        similarity.max(axis=3).mean(axis=2) + similarity.max(axis=2).mean(axis=2)
    )
    return aligned, chamfer


def _protocol(
    seed: int,
    frame: pd.DataFrame,
    fused: np.ndarray,
    conv: np.ndarray,
    osnet: np.ndarray,
    colors: np.ndarray,
    parts: np.ndarray,
    view_probabilities: np.ndarray,
):
    protocol = build_identity_stratified_protocol(frame, seed=seed)
    qi, gi = protocol.query_indices, protocol.gallery_indices
    gallery = database_side_augmentation(fused[gi], top_k=5, alpha=2.0)
    query = query_expansion(fused[qi], gallery, top_k=2, alpha=1.0)
    base = query @ gallery.T
    raw_fused = fused[qi] @ fused[gi].T
    conv_score = conv[qi] @ conv[gi].T
    osnet_score = osnet[qi] @ osnet[gi].T
    q, g = frame.iloc[qi].reset_index(drop=True), frame.iloc[gi].reset_index(drop=True)
    gp, gc = g.vehicle_id.to_numpy(), g.camera_id.to_numpy()
    candidates = np.empty((len(q), TOP_K), dtype=np.int64)
    for index, row in enumerate(q.itertuples(index=False)):
        valid = ~((gp == row.vehicle_id) & (gc == row.camera_id))
        candidates[index] = np.flatnonzero(valid)[
            np.argsort(-base[index, valid], kind="mergesort")[:TOP_K]
        ]
    row_index = np.arange(len(q))[:, None]
    aligned, chamfer = _local(parts[qi], parts[gi], candidates)
    q_view = view_probabilities[qi]
    g_view = view_probabilities[gi][candidates]
    view_overlap = np.einsum("qb,qkb->qk", q_view, g_view, optimize=True)
    bins = q_view.shape[1]
    positions = np.arange(bins)
    circular = np.abs(positions[:, None] - positions[None, :])
    circular = np.minimum(circular, bins - circular).astype(np.float32)
    circular /= max(float(circular.max()), 1.0)
    view_distance = np.einsum(
        "qa,ab,qkb->qk", q_view, circular, g_view, optimize=True
    )
    aspect = np.abs(
        np.log((q.w / q.h).to_numpy())[:, None]
        - np.log((g.w / g.h).to_numpy()[candidates])
    )
    area = np.abs(
        np.log((q.w * q.h).to_numpy())[:, None]
        - np.log((g.w * g.h).to_numpy()[candidates])
    )
    color = np.einsum(
        "qd,qkd->qk", colors[qi], colors[gi][candidates], optimize=True
    )
    selected_base = base[row_index, candidates]
    selected_conv = conv_score[row_index, candidates]
    selected_osnet = osnet_score[row_index, candidates]
    features = np.stack(
        (
            selected_base,
            raw_fused[row_index, candidates],
            selected_conv,
            selected_osnet,
            np.abs(selected_conv - selected_osnet),
            aspect,
            area,
            color,
            aligned,
            chamfer,
            view_overlap,
            view_distance,
            selected_base - selected_base[:, :1],
            np.broadcast_to(np.arange(TOP_K) / (TOP_K - 1), selected_base.shape),
        ),
        axis=2,
    ).astype(np.float32)
    labels = (gp[candidates] == q.vehicle_id.to_numpy()[:, None]).astype(np.uint8)
    return {
        "seed": seed,
        "base": base,
        "query": q,
        "gallery": g,
        "candidates": candidates,
        "features": features,
        "labels": labels,
    }


def _metrics_from_order(protocol, probabilities: np.ndarray | None, blend: float):
    q, g = protocol["query"], protocol["gallery"]
    gp, gc = g.vehicle_id.to_numpy(), g.camera_id.to_numpy()
    base = protocol["base"]
    candidates = protocol["candidates"]
    aps, rank1, rank5 = [], [], []
    for index, row in enumerate(q.itertuples(index=False)):
        valid = ~((gp == row.vehicle_id) & (gc == row.camera_id))
        base_order = np.flatnonzero(valid)[np.argsort(-base[index, valid], kind="mergesort")]
        if probabilities is not None:
            local_base = base[index, candidates[index]]
            local_base = (local_base - local_base.mean()) / (local_base.std() + 1e-6)
            probability = probabilities[index]
            probability = (probability - probability.mean()) / (probability.std() + 1e-6)
            local_order = np.argsort(
                -(blend * probability + (1.0 - blend) * local_base), kind="mergesort"
            )
            base_order[:TOP_K] = candidates[index, local_order]
        positions = np.flatnonzero(gp[base_order] == row.vehicle_id) + 1
        aps.append(float(np.mean(np.arange(1, len(positions) + 1) / positions)))
        rank1.append(float(positions[0] == 1))
        rank5.append(float(positions[0] <= 5))
    return {"mAP": float(np.mean(aps)), "rank1": float(np.mean(rank1)), "rank5": float(np.mean(rank5))}


def _fit_predict(tune, confirmation, feature_indices: tuple[int, ...]):
    groups = tune[0]["query"].vehicle_id.to_numpy()
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))
    tune_predictions = {p["seed"]: np.zeros_like(p["labels"], dtype=np.float32) for p in tune}
    confirm_predictions = {p["seed"]: np.zeros_like(p["labels"], dtype=np.float32) for p in confirmation}
    for train_query, held_query in splits:
        x_train = np.concatenate(
            [p["features"][train_query][:, :, feature_indices].reshape(-1, len(feature_indices)) for p in tune]
        )
        y_train = np.concatenate([p["labels"][train_query].reshape(-1) for p in tune])
        positives = max(int(y_train.sum()), 1)
        sample_weight = np.where(y_train == 1, (len(y_train) - positives) / positives, 1.0)
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        for protocol in tune:
            x = protocol["features"][held_query][:, :, feature_indices].reshape(-1, len(feature_indices))
            tune_predictions[protocol["seed"]][held_query] = model.predict_proba(x)[:, 1].reshape(-1, TOP_K)
        for protocol in confirmation:
            x = protocol["features"][held_query][:, :, feature_indices].reshape(-1, len(feature_indices))
            confirm_predictions[protocol["seed"]][held_query] = model.predict_proba(x)[:, 1].reshape(-1, TOP_K)
    return tune_predictions, confirm_predictions


def _evaluate(protocols, predictions, blend: float):
    rows = []
    for protocol in protocols:
        row = _metrics_from_order(protocol, predictions.get(protocol["seed"]), blend)
        rows.append({"seed": protocol["seed"], **row})
    return {
        "mAP": float(np.mean([r["mAP"] for r in rows])),
        "rank1": float(np.mean([r["rank1"] for r in rows])),
        "rank5": float(np.mean([r["rank5"] for r in rows])),
        "per_seed": rows,
    }


def main() -> None:
    frame = pd.read_csv("splits/val.csv")
    with np.load("outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz", allow_pickle=False) as archive:
        conv = _normalize(archive["embeddings"])
    with np.load("outputs/expert_fusion/cache/osnet_openvino_val_4aaad3e5db648618.npz", allow_pickle=False) as archive:
        osnet = _normalize(archive["embeddings"])
    # The input-geometry probe selected 30/70 on tune seeds and confirmed it
    # before this verifier is fitted.
    fused = _normalize(
        np.concatenate((np.sqrt(0.30) * conv, np.sqrt(0.70) * osnet), axis=1)
    )
    colors, parts = _color_descriptors(frame), _load_parts(frame)
    with np.load("outputs/expert_fusion/cache/val_vehiclex_view_probabilities.npz", allow_pickle=False) as archive:
        if not np.array_equal(
            archive["image_ids"], frame.image_id.astype(str).to_numpy(dtype=np.str_)
        ):
            raise RuntimeError("viewpoint cache does not match validation rows")
        view_probabilities = archive["probabilities"].astype(np.float32)
    tune = [_protocol(seed, frame, fused, conv, osnet, colors, parts, view_probabilities) for seed in TUNE_SEEDS]
    confirmation = [_protocol(seed, frame, fused, conv, osnet, colors, parts, view_probabilities) for seed in CONFIRM_SEEDS]

    variants = {
        "expert_scores": (0, 1, 2, 3, 4, 12, 13),
        "all_pair_cues_no_view": tuple(
            index for index in range(len(FEATURE_NAMES)) if index not in (10, 11)
        ),
        "all_pair_cues": tuple(range(len(FEATURE_NAMES))),
    }
    candidates = []
    prediction_cache = {}
    for name, indices in variants.items():
        tune_pred, confirm_pred = _fit_predict(tune, confirmation, indices)
        prediction_cache[name] = (tune_pred, confirm_pred)
        for blend in (0.25, 0.5, 0.75, 1.0):
            metrics = _evaluate(tune, tune_pred, blend)
            candidates.append({"variant": name, "blend": blend, **metrics})
    selected = max(candidates, key=lambda row: row["mAP"])
    confirm_pred = prediction_cache[selected["variant"]][1]
    confirmed = _evaluate(confirmation, confirm_pred, selected["blend"])
    baseline = _evaluate(confirmation, {}, 0.0)
    result = {
        "feature_names": FEATURE_NAMES,
        "validation_design": "5-fold identity-disjoint OOF training on tune protocols; one locked confirmation",
        "selected_on_tune": selected,
        "confirmation": confirmed,
        "baseline": baseline,
        "delta": {key: confirmed[key] - baseline[key] for key in ("mAP", "rank1", "rank5")},
        "candidates": candidates,
    }
    output = Path("outputs/expert_fusion/pair_verifier_probe.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected_on_tune", "confirmation", "baseline", "delta")}, indent=2))


if __name__ == "__main__":
    main()
