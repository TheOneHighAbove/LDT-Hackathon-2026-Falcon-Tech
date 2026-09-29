"""Probe a frozen ConvNeXt part matcher as a top-K pairwise verifier.

The global model is not retrained.  We retain a 3x3 grid of normalized
high-level descriptors and compare candidate pairs both positionally and with
bidirectional soft part alignment.  Hyperparameters are selected on the five
tune seeds and evaluated once on the locked confirmation seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data import VehicleDataset, build_eval_transform
from src.engine import load_inference_checkpoint
from src.postprocess_eval import build_identity_stratified_protocol
from src.reranking import database_side_augmentation, query_expansion


TUNE_SEEDS = (1337, 2027, 3407, 4517, 7919)
CONFIRM_SEEDS = (101, 211, 307, 401, 503)
CACHE = Path("outputs/expert_fusion/cache/convnext_val_parts_3x3.npz")


@torch.inference_mode()
def _extract_parts(frame: pd.DataFrame) -> np.ndarray:
    model, checkpoint = load_inference_checkpoint(
        Path("weights/best.pt"), device=torch.device("cuda")
    )
    model.eval().to(memory_format=torch.channels_last)
    prep = checkpoint["preprocessing"]
    dataset = VehicleDataset(
        frame,
        Path("dataset/images"),
        transform=build_eval_transform(
            int(prep["input_size"]),
            resize_mode=str(prep.get("resize_mode", "direct")),
        ),
        bbox_padding=float(prep["bbox_padding"]),
        require_camera_id=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    chunks: list[np.ndarray] = []
    observed: list[str] = []
    feature_dim = int(model.projection.in_features)
    for batch in loader:
        images = batch["image"].to(
            "cuda", non_blocking=True, memory_format=torch.channels_last
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            spatial = model.backbone.forward_features(images)
            spatial = model._to_spatial_feature_map(
                spatial, feature_dim, mode="local verifier"
            )
            pooled = F.adaptive_avg_pool2d(spatial, (3, 3)).flatten(2).transpose(1, 2)
            pooled = F.normalize(pooled.float(), dim=2)
        chunks.append(pooled.cpu().numpy().astype(np.float16))
        observed.extend(str(value) for value in batch["image_id"])
    expected = frame.image_id.astype(str).tolist()
    if observed != expected:
        raise RuntimeError("part extractor changed annotation order")
    return np.concatenate(chunks)


def _parts(frame: pd.DataFrame) -> np.ndarray:
    expected = frame.image_id.astype(str).to_numpy(dtype=np.str_)
    if CACHE.is_file():
        with np.load(CACHE, allow_pickle=False) as archive:
            if np.array_equal(archive["image_ids"], expected):
                return archive["parts"].astype(np.float32)
    parts = _extract_parts(frame)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, image_ids=expected, parts=parts)
    return parts.astype(np.float32)


def _base_protocol(embeddings: np.ndarray, frame: pd.DataFrame, seed: int):
    protocol = build_identity_stratified_protocol(frame, seed=seed)
    q_idx, g_idx = protocol.query_indices, protocol.gallery_indices
    gallery = database_side_augmentation(embeddings[g_idx], top_k=5, alpha=2.0)
    query = query_expansion(embeddings[q_idx], gallery, top_k=2, alpha=1.0)
    return query @ gallery.T, q_idx, g_idx


def _local_scores(
    q_parts: np.ndarray, g_parts: np.ndarray, candidate_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    candidates = g_parts[candidate_indices]
    # [query, candidate, query_part, gallery_part]
    similarity = np.einsum("qid,qkjd->qkij", q_parts, candidates, optimize=True)
    aligned = np.diagonal(similarity, axis1=2, axis2=3).mean(axis=2)
    chamfer = 0.5 * (
        similarity.max(axis=3).mean(axis=2) + similarity.max(axis=2).mean(axis=2)
    )
    return aligned, chamfer


def _build_protocol_cache(
    embeddings: np.ndarray,
    parts: np.ndarray,
    frame: pd.DataFrame,
    seeds: tuple[int, ...],
    top_k: int = 50,
):
    result = {}
    for seed in seeds:
        base, q_idx, g_idx = _base_protocol(embeddings, frame, seed)
        q = frame.iloc[q_idx].reset_index(drop=True)
        g = frame.iloc[g_idx].reset_index(drop=True)
        gp, gc = g.vehicle_id.to_numpy(), g.camera_id.to_numpy()
        candidates = np.empty((len(q), top_k), dtype=np.int64)
        for index, row in enumerate(q.itertuples(index=False)):
            valid = ~((gp == row.vehicle_id) & (gc == row.camera_id))
            order = np.flatnonzero(valid)[np.argsort(-base[index, valid], kind="mergesort")]
            candidates[index] = order[:top_k]
        aligned, chamfer = _local_scores(parts[q_idx], parts[g_idx], candidates)
        result[seed] = (base, q, g, candidates, aligned, chamfer)
    return result


def _metrics(scores: np.ndarray, q: pd.DataFrame, g: pd.DataFrame):
    gp, gc = g.vehicle_id.to_numpy(), g.camera_id.to_numpy()
    aps, r1, r5 = [], [], []
    for index, row in enumerate(q.itertuples(index=False)):
        valid = ~((gp == row.vehicle_id) & (gc == row.camera_id))
        order = np.flatnonzero(valid)[np.argsort(-scores[index, valid], kind="mergesort")]
        positions = np.flatnonzero(gp[order] == row.vehicle_id) + 1
        aps.append(float(np.mean(np.arange(1, len(positions) + 1) / positions)))
        r1.append(float(positions[0] == 1))
        r5.append(float(positions[0] <= 5))
    return float(np.mean(aps)), float(np.mean(r1)), float(np.mean(r5))


def _evaluate(protocols, method: str, weight: float):
    rows = []
    local_index = 4 if method == "aligned" else 5
    for seed, payload in protocols.items():
        base, q, g, candidates = payload[:4]
        local = payload[local_index]
        adjusted = base.copy()
        # Candidate-wise standardization makes one global weight meaningful
        # for easy and hard queries while changing only the top-50 shortlist.
        normalized = (local - local.mean(axis=1, keepdims=True)) / (
            local.std(axis=1, keepdims=True) + 1e-6
        )
        rows_idx = np.arange(len(q))[:, None]
        adjusted[rows_idx, candidates] += weight * normalized
        m_ap, rank1, rank5 = _metrics(adjusted, q, g)
        rows.append({"seed": seed, "mAP": m_ap, "rank1": rank1, "rank5": rank5})
    return {
        "method": method,
        "weight": weight,
        "mAP": float(np.mean([row["mAP"] for row in rows])),
        "rank1": float(np.mean([row["rank1"] for row in rows])),
        "rank5": float(np.mean([row["rank5"] for row in rows])),
        "per_seed": rows,
    }


def main() -> None:
    frame = pd.read_csv("splits/val.csv")
    embeddings = np.load("outputs/expert_fusion/val_embeddings.npy").astype(np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    parts = _parts(frame)
    tune = _build_protocol_cache(embeddings, parts, frame, TUNE_SEEDS)
    confirm = _build_protocol_cache(embeddings, parts, frame, CONFIRM_SEEDS)
    weights = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
    grid = [_evaluate(tune, method, weight) for method in ("aligned", "chamfer") for weight in weights]
    selected = max(grid, key=lambda row: row["mAP"])
    confirmation = _evaluate(confirm, selected["method"], selected["weight"])
    baseline = _evaluate(confirm, "aligned", 0.0)
    result = {
        "selected_on_tune": {key: selected[key] for key in ("method", "weight", "mAP", "rank1", "rank5")},
        "confirmation": confirmation,
        "baseline": baseline,
        "delta": {key: confirmation[key] - baseline[key] for key in ("mAP", "rank1", "rank5")},
        "grid": grid,
    }
    output = Path("outputs/expert_fusion/local_verifier_probe.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected_on_tune", "confirmation", "baseline", "delta")}, indent=2))


if __name__ == "__main__":
    main()
