"""Cross-camera closed/open-set validation protocols and score calibration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .metrics import calibrate_open_set, evaluate_retrieval, split_known_unknown_ids


def build_open_set_protocol(
    annotations: pd.DataFrame,
    *,
    unknown_fraction: float = 0.25,
    seed: int = 1337,
) -> dict[str, Any]:
    """Build a deterministic artificial no-match validation protocol.

    Known identities contribute one gallery image and queries from every other
    camera.  Unknown identities are removed from gallery in full and all their
    images become no-match queries.  This prevents temporal near-duplicates
    from one camera from making refusal calibration artificially easy.
    """

    required = {"image_id", "vehicle_id", "camera_id"}
    missing = sorted(required.difference(annotations.columns))
    if missing:
        raise ValueError(f"open-set protocol is missing columns: {missing}")
    frame = annotations.reset_index(drop=True)
    known_ids, unknown_ids = split_known_unknown_ids(
        frame["vehicle_id"].tolist(),
        unknown_fraction=unknown_fraction,
        seed=seed,
    )
    known_set, unknown_set = set(known_ids), set(unknown_ids)
    rng = np.random.default_rng(seed)

    gallery_indices: list[int] = []
    known_query_indices: list[int] = []
    for pid in known_ids:
        identity_indices = np.flatnonzero(frame["vehicle_id"].to_numpy() == pid)
        cameras = frame.iloc[identity_indices]["camera_id"].to_numpy()
        unique_cameras = np.unique(cameras)
        if len(unique_cameras) < 2:
            raise ValueError(f"vehicle_id={pid!r} has no cross-camera positive")
        gallery_camera = unique_cameras[int(rng.integers(0, len(unique_cameras)))]
        camera_indices = identity_indices[cameras == gallery_camera]
        gallery_index = int(camera_indices[int(rng.integers(0, len(camera_indices)))])
        gallery_indices.append(gallery_index)
        known_query_indices.extend(
            int(index)
            for index, camera in zip(identity_indices, cameras, strict=True)
            if camera != gallery_camera
        )

    unknown_query_indices = [
        int(index)
        for index, pid in enumerate(frame["vehicle_id"].tolist())
        if pid in unknown_set
    ]
    query_indices = np.asarray(known_query_indices + unknown_query_indices, dtype=np.int64)
    gallery_indices_array = np.asarray(gallery_indices, dtype=np.int64)
    if len(query_indices) == 0 or len(gallery_indices_array) == 0:
        raise RuntimeError("open-set protocol unexpectedly produced an empty partition")
    return {
        "query_indices": query_indices,
        "gallery_indices": gallery_indices_array,
        "known_pids": known_set,
        "unknown_pids": unknown_set,
        "num_known_identities": len(known_set),
        "num_unknown_identities": len(unknown_set),
        "num_known_queries": len(known_query_indices),
        "num_unknown_queries": len(unknown_query_indices),
        "seed": int(seed),
        "unknown_fraction": float(unknown_fraction),
    }


def fit_platt_calibration(
    scores: np.ndarray,
    correct_candidates: np.ndarray,
    *,
    fallback_threshold: float,
    fallback_temperature: float = 0.08,
) -> dict[str, float | str]:
    """Fit a one-dimensional logistic map from cosine to confidence."""

    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(correct_candidates, dtype=bool)
    if scores.ndim != 1 or targets.shape != scores.shape:
        raise ValueError("scores and correct_candidates must be aligned vectors")
    if np.unique(targets).size < 2:
        slope = 1.0 / max(float(fallback_temperature), 1e-6)
        return {
            "type": "temperature",
            "slope": slope,
            "intercept": -slope * float(fallback_threshold),
        }
    try:
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(
            C=100.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        )
        estimator.fit(scores.reshape(-1, 1), targets.astype(np.uint8))
        slope = float(estimator.coef_[0, 0])
        intercept = float(estimator.intercept_[0])
        if not math.isfinite(slope + intercept) or slope <= 0:
            raise ValueError("non-monotonic Platt calibration")
        return {"type": "platt", "slope": slope, "intercept": intercept}
    except (ImportError, ValueError):
        slope = 1.0 / max(float(fallback_temperature), 1e-6)
        return {
            "type": "temperature",
            "slope": slope,
            "intercept": -slope * float(fallback_threshold),
        }


def calibrated_confidence(
    similarity: np.ndarray | float,
    calibration: dict[str, Any] | None,
) -> np.ndarray:
    values = np.asarray(similarity, dtype=np.float64)
    calibration = calibration or {}
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    logits = np.clip(slope * values + intercept, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def evaluate_validation(
    embeddings: np.ndarray,
    annotations: pd.DataFrame,
    *,
    unknown_fraction: float = 0.25,
    seed: int = 1337,
    fallback_temperature: float = 0.08,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate retrieval plus an identity-disjoint artificial open set."""

    frame = annotations.reset_index(drop=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(frame) != len(embeddings):
        raise ValueError("validation annotations and embeddings are not aligned")
    retrieval = evaluate_retrieval(
        embeddings,
        embeddings,
        frame["vehicle_id"].to_numpy(),
        frame["vehicle_id"].to_numpy(),
        query_camera_ids=frame["camera_id"].to_numpy(),
        gallery_camera_ids=frame["camera_id"].to_numpy(),
        query_image_ids=frame["image_id"].to_numpy(),
        gallery_image_ids=frame["image_id"].to_numpy(),
        same_source=True,
    )
    protocol = build_open_set_protocol(
        frame, unknown_fraction=unknown_fraction, seed=seed
    )
    query_indices = protocol["query_indices"]
    gallery_indices = protocol["gallery_indices"]
    query = frame.iloc[query_indices]
    gallery = frame.iloc[gallery_indices]
    open_result = calibrate_open_set(
        embeddings[query_indices],
        embeddings[gallery_indices],
        query["vehicle_id"].to_numpy(),
        gallery["vehicle_id"].to_numpy(),
        known_pids=protocol["known_pids"],
        unknown_pids=protocol["unknown_pids"],
        query_camera_ids=query["camera_id"].to_numpy(),
        gallery_camera_ids=gallery["camera_id"].to_numpy(),
        query_image_ids=query["image_id"].to_numpy(),
        gallery_image_ids=gallery["image_id"].to_numpy(),
    )
    calibration = fit_platt_calibration(
        open_result["scores"],
        open_result["top1_correct"],
        fallback_threshold=float(open_result["threshold"]),
        fallback_temperature=fallback_temperature,
    )
    confidence_threshold = float(
        calibrated_confidence(float(open_result["threshold"]), calibration)
    )
    open_scalars = {
        key: value
        for key, value in open_result.items()
        if not isinstance(value, np.ndarray)
    }
    protocol_scalars = {
        key: value
        for key, value in protocol.items()
        if key not in {"query_indices", "gallery_indices", "known_pids", "unknown_pids"}
    }
    report = {
        "retrieval": retrieval,
        "open_set": open_scalars,
        "confidence_calibration": calibration,
        "confidence_threshold": confidence_threshold,
        "protocol": protocol_scalars,
    }
    details = {
        "scores": np.asarray(open_result["scores"], dtype=np.float32),
        "is_known": np.asarray(open_result["is_known"], dtype=bool),
        "top1_correct": np.asarray(open_result["top1_correct"], dtype=bool),
        "query_indices": query_indices,
        "gallery_indices": gallery_indices,
    }
    return report, details


__all__ = [
    "build_open_set_protocol",
    "calibrated_confidence",
    "evaluate_validation",
    "fit_platt_calibration",
]
