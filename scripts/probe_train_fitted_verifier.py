"""Fit the selected pair verifier on train IDs and test on unseen val IDs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data import crop_vehicle
from src.engine import load_inference_checkpoint
from src.expert_fusion import (
    _osnet_embeddings,
    _osnet_session,
    _torch_embeddings,
)
from scripts.probe_local_verifier import _extract_parts
from scripts.probe_pair_verifier import (
    CONFIRM_SEEDS,
    FEATURE_NAMES,
    TOP_K,
    _metrics_from_order,
    _normalize,
    _protocol,
)


TRAIN_SEEDS = (9109, 11213)
SELECTED_FEATURES = tuple(
    index for index in range(len(FEATURE_NAMES)) if index not in (10, 11)
)


def _aligned_cache(path: Path, frame: pd.DataFrame, key: str = "embeddings"):
    if not path.is_file():
        return None
    ids = frame.image_id.astype(str).to_numpy(dtype=np.str_)
    with np.load(path, allow_pickle=False) as archive:
        if np.array_equal(archive["image_ids"], ids):
            return archive[key].astype(np.float32)
    return None


def _train_global_embeddings(frame: pd.DataFrame):
    conv_path = Path("outputs/expert_fusion/cache/convnext_train_verifier.npz")
    osnet_path = Path("outputs/expert_fusion/cache/osnet_train_verifier.npz")
    conv = _aligned_cache(conv_path, frame)
    osnet = _aligned_cache(osnet_path, frame)
    ids = frame.image_id.astype(str).to_numpy(dtype=np.str_)
    model = checkpoint = None
    if conv is None:
        model, checkpoint = load_inference_checkpoint(
            Path("weights/best.pt"), device=torch.device("cuda")
        )
        model = model.to(memory_format=torch.channels_last)
        conv = _torch_embeddings(
            model,
            frame,
            images_dir=Path("dataset/images"),
            preprocessing=checkpoint["preprocessing"],
            batch_size=64,
            workers=4,
            device=torch.device("cuda"),
            channels_last=True,
        )
        conv = _normalize(conv)
        np.savez(conv_path, image_ids=ids, embeddings=conv.astype(np.float16))
    if osnet is None:
        if checkpoint is None:
            _, checkpoint = load_inference_checkpoint(
                Path("weights/best.pt"), device=torch.device("cpu")
            )
        session = _osnet_session(Path("weights/osnet_ain_x1_0_vehicle_reid.onnx"))
        osnet = _osnet_embeddings(
            session,
            frame,
            images_dir=Path("dataset/images"),
            bbox_padding=float(checkpoint["preprocessing"]["bbox_padding"]),
            batch_size=64,
            workers=4,
        )
        osnet = _normalize(osnet)
        np.savez(osnet_path, image_ids=ids, embeddings=osnet.astype(np.float16))
    return _normalize(conv), _normalize(osnet)


def _train_parts(frame: pd.DataFrame):
    path = Path("outputs/expert_fusion/cache/convnext_train_parts_3x3.npz")
    cached = _aligned_cache(path, frame, key="parts")
    if cached is not None:
        return cached
    parts = _extract_parts(frame)
    np.savez_compressed(
        path,
        image_ids=frame.image_id.astype(str).to_numpy(dtype=np.str_),
        parts=parts.astype(np.float16),
    )
    return parts.astype(np.float32)


def _colors(frame: pd.DataFrame, split: str):
    path = Path(f"outputs/expert_fusion/cache/{split}_verifier_colors.npz")
    cached = _aligned_cache(path, frame, key="descriptors")
    if cached is not None:
        return cached
    descriptors = []
    for row in frame.itertuples(index=False):
        with Image.open(Path("dataset/images") / f"{row.image_id}.jpg") as image:
            crop = crop_vehicle(image, (row.x, row.y, row.w, row.h), padding=0.0)
            array = np.asarray(crop.resize((64, 64)).convert("HSV"), dtype=np.uint8)
        regions = (array, array[:32, :32], array[:32, 32:], array[32:, :32], array[32:, 32:])
        values = []
        for region in regions:
            for channel, bins in ((0, 16), (1, 8), (2, 8)):
                hist = np.histogram(region[..., channel], bins=bins, range=(0, 256))[0].astype(np.float32)
                hist /= max(float(hist.sum()), 1.0)
                values.append(np.sqrt(hist))
        descriptor = np.concatenate(values)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
        descriptors.append(descriptor)
    result = np.stack(descriptors).astype(np.float32)
    np.savez_compressed(
        path,
        image_ids=frame.image_id.astype(str).to_numpy(dtype=np.str_),
        descriptors=result.astype(np.float16),
    )
    return result


def _fused(conv, osnet):
    return _normalize(
        np.concatenate((np.sqrt(0.30) * conv, np.sqrt(0.70) * osnet), axis=1)
    )


def main() -> None:
    train_frame = pd.read_csv("splits/train.csv")
    val_frame = pd.read_csv("splits/val.csv")
    train_conv, train_osnet = _train_global_embeddings(train_frame)
    train_parts = _train_parts(train_frame)
    train_colors = _colors(train_frame, "train")
    uniform_train_view = np.full((len(train_frame), 8), 1.0 / 8.0, dtype=np.float32)
    train_protocols = [
        _protocol(
            seed,
            train_frame,
            _fused(train_conv, train_osnet),
            train_conv,
            train_osnet,
            train_colors,
            train_parts,
            uniform_train_view,
        )
        for seed in TRAIN_SEEDS
    ]
    features = np.concatenate(
        [p["features"][:, :, SELECTED_FEATURES].reshape(-1, len(SELECTED_FEATURES)) for p in train_protocols]
    )
    labels = np.concatenate([p["labels"].reshape(-1) for p in train_protocols])
    positives = max(int(labels.sum()), 1)
    weights = np.where(labels == 1, (len(labels) - positives) / positives, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(features, labels, sample_weight=weights)

    with np.load("outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz", allow_pickle=False) as archive:
        val_conv = _normalize(archive["embeddings"])
    with np.load("outputs/expert_fusion/cache/osnet_openvino_val_4aaad3e5db648618.npz", allow_pickle=False) as archive:
        val_osnet = _normalize(archive["embeddings"])
    with np.load("outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", allow_pickle=False) as archive:
        val_parts = archive["parts"].astype(np.float32)
    val_colors = _colors(val_frame, "val")
    uniform_val_view = np.full((len(val_frame), 8), 1.0 / 8.0, dtype=np.float32)
    val_protocols = [
        _protocol(
            seed,
            val_frame,
            _fused(val_conv, val_osnet),
            val_conv,
            val_osnet,
            val_colors,
            val_parts,
            uniform_val_view,
        )
        for seed in CONFIRM_SEEDS
    ]
    rows, baseline_rows = [], []
    for protocol in val_protocols:
        x = protocol["features"][:, :, SELECTED_FEATURES].reshape(-1, len(SELECTED_FEATURES))
        probabilities = model.predict_proba(x)[:, 1].reshape(-1, TOP_K)
        rows.append({"seed": protocol["seed"], **_metrics_from_order(protocol, probabilities, 0.75)})
        baseline_rows.append({"seed": protocol["seed"], **_metrics_from_order(protocol, None, 0.0)})
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("mAP", "rank1", "rank5")
    }
    baseline = {
        key: float(np.mean([row[key] for row in baseline_rows]))
        for key in ("mAP", "rank1", "rank5")
    }
    result = {
        "design": "Verifier fitted only on train identities; all val identities unseen.",
        "train_seeds": TRAIN_SEEDS,
        "feature_names": [FEATURE_NAMES[index] for index in SELECTED_FEATURES],
        "blend": 0.75,
        "num_train_pairs": int(len(labels)),
        "num_positive_pairs": int(labels.sum()),
        "confirmation": aggregate,
        "baseline": baseline,
        "delta": {key: aggregate[key] - baseline[key] for key in aggregate},
        "per_seed": rows,
    }
    Path("outputs/expert_fusion/train_fitted_verifier.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
