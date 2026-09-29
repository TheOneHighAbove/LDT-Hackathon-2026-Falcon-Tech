"""Robust CPU-only open-set calibration for gallery-conditioned DBA + QE.

The protocol repeats whole-identity known/unknown splits.  For every known
identity it reserves one query and one mandatory gallery image from another
camera, then fills the gallery with other known images up to the requested
size.  Every image of an unknown identity is a query and unknown identities
are absent from gallery.  Query and gallery image IDs are always disjoint.

The mandatory deployment output is a raw cosine threshold after gallery DBA
(``k=3, alpha=2`` by default) and query expansion conditioned on that augmented
gallery (``k=2, alpha=1``).  An optional multivariate correctness model is only
an OOF diagnostic and never replaces the reported raw threshold.  When locked
confirmation seeds are supplied, feature-family selection, probability
thresholding, and fitting all use tune seeds only; confirmation is evaluated
once with frozen parameters.

Limitations:
    * Validation embeddings and CSV rows must already be aligned.
    * Every validation identity must occur in at least two cameras.
    * The protocol models a fixed gallery size but not an unknown production
      identity distribution or temporal camera priors.
    * Reusing identities across repeated seeds reduces variance estimation;
      seeds measure protocol sensitivity, not independent dataset uncertainty.

Example::

    python -m src.robust_open_set \
        --embeddings outputs/val_embeddings.npy \
        --annotations splits/val.csv \
        --output outputs/robust_open_set_calibration.json \
        --seeds 1337 2027 3407 4517 7919 --fit-oof-features
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import evaluate_open_set, search_rejection_threshold
from .reranking import (
    database_side_augmentation,
    l2_normalize_embeddings,
    query_expansion,
)
from .score_calibration import (
    ADAPTIVE_REFUSAL_FEATURE_NAMES,
    ADAPTIVE_REFUSAL_FEATURE_VARIANT,
    CANDIDATE_FEATURE_NAMES,
    evaluate_group_oof,
    extract_candidate_features,
    fit_candidate_correctness_model,
)
from .validation import calibrated_confidence, fit_platt_calibration


DEFAULT_SEEDS: tuple[int, ...] = (1337, 2027, 3407, 4517, 7919)
REPORT_SCHEMA_VERSION = 1


_ADAPTIVE_FEATURE_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("top1", ("top1_similarity",)),
    (
        "top1_margins",
        ("top1_similarity", "top1_top2_margin", "top1_to_top5_gap"),
    ),
    (
        ADAPTIVE_REFUSAL_FEATURE_VARIANT,
        ADAPTIVE_REFUSAL_FEATURE_NAMES,
    ),
    (
        "top1_margins_local_density_mutual_nn",
        (
            "top1_similarity",
            "top1_top2_margin",
            "top1_to_top5_gap",
            "top_candidate_gallery_density",
            "top_candidate_gallery_isolation",
            "top_candidate_mutual_nn",
        ),
    ),
    ("all_features", CANDIDATE_FEATURE_NAMES),
)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(annotations, pd.DataFrame):
        raise ValueError("annotations must be a pandas DataFrame")
    required = {"image_id", "vehicle_id", "camera_id"}
    missing = sorted(required.difference(annotations.columns))
    if missing:
        raise ValueError(f"annotations are missing columns: {missing}")
    frame = annotations.reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError("annotations must not be empty")
    if frame[list(required)].isna().any().any():
        raise ValueError("annotations contain missing IDs")
    if frame["image_id"].duplicated().any():
        raise ValueError("image_id must be unique")
    camera_counts = frame.groupby("vehicle_id", sort=False)["camera_id"].nunique()
    invalid = camera_counts[camera_counts < 2]
    if len(invalid):
        examples = [_python_scalar(value) for value in invalid.index[:5].tolist()]
        raise ValueError(
            "every vehicle_id must have images from at least two cameras; "
            f"invalid IDs include {examples!r}"
        )
    if frame["vehicle_id"].nunique() < 2:
        raise ValueError("at least two vehicle identities are required")
    return frame


@dataclass(frozen=True)
class GalleryConditionedProtocol:
    """One deterministic identity-disjoint open-set split."""

    seed: int
    query_indices: np.ndarray
    gallery_indices: np.ndarray
    query_is_known: np.ndarray
    known_vehicle_ids: tuple[Any, ...]
    unknown_vehicle_ids: tuple[Any, ...]
    target_gallery_size: int

    def __post_init__(self) -> None:
        query = np.asarray(self.query_indices, dtype=np.int64).copy()
        gallery = np.asarray(self.gallery_indices, dtype=np.int64).copy()
        known_mask = np.asarray(self.query_is_known, dtype=bool).copy()
        if query.ndim != 1 or gallery.ndim != 1 or known_mask.ndim != 1:
            raise ValueError("protocol indices and known mask must be vectors")
        if len(query) == 0 or len(gallery) == 0:
            raise ValueError("protocol query and gallery must not be empty")
        if len(known_mask) != len(query):
            raise ValueError("query_is_known must align with query_indices")
        if len(gallery) != int(self.target_gallery_size):
            raise ValueError("gallery length does not match target_gallery_size")
        if len(np.unique(query)) != len(query) or len(np.unique(gallery)) != len(gallery):
            raise ValueError("protocol contains duplicate row indices")
        if np.intersect1d(query, gallery).size:
            raise ValueError("protocol query and gallery rows overlap")
        query.setflags(write=False)
        gallery.setflags(write=False)
        known_mask.setflags(write=False)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "query_indices", query)
        object.__setattr__(self, "gallery_indices", gallery)
        object.__setattr__(self, "query_is_known", known_mask)
        object.__setattr__(self, "target_gallery_size", int(self.target_gallery_size))
        object.__setattr__(
            self,
            "known_vehicle_ids",
            tuple(_python_scalar(value) for value in self.known_vehicle_ids),
        )
        object.__setattr__(
            self,
            "unknown_vehicle_ids",
            tuple(_python_scalar(value) for value in self.unknown_vehicle_ids),
        )


def build_gallery_conditioned_protocol(
    annotations: pd.DataFrame,
    *,
    seed: int,
    unknown_fraction: float = 0.25,
    target_gallery_size: int | None = None,
) -> GalleryConditionedProtocol:
    """Construct a reproducible no-leakage open-set protocol.

    Exactly one query is sampled per identity.  A known query receives a
    mandatory gallery mate from a different camera; an unknown identity is
    excluded from the gallery entirely.  Remaining gallery slots are sampled
    without replacement from the other known images.  Consequently,
    ``unknown_fraction`` controls the query distribution directly instead of
    being amplified by identities that contain many images.
    """

    frame = _validate_annotations(annotations)
    try:
        unknown_fraction = float(unknown_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown_fraction must be strictly between 0 and 1") from exc
    if not np.isfinite(unknown_fraction) or not 0.0 < unknown_fraction < 1.0:
        raise ValueError("unknown_fraction must be strictly between 0 and 1")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")

    identities = [_python_scalar(value) for value in frame["vehicle_id"].unique()]
    unknown_count = min(
        len(identities) - 1,
        max(1, round(len(identities) * unknown_fraction)),
    )
    rng = np.random.default_rng(int(seed))
    unknown_positions = set(
        int(value)
        for value in rng.choice(len(identities), size=unknown_count, replace=False)
    )
    known_ids = tuple(
        identity
        for index, identity in enumerate(identities)
        if index not in unknown_positions
    )
    unknown_ids = tuple(
        identity
        for index, identity in enumerate(identities)
        if index in unknown_positions
    )
    known_set = set(known_ids)
    unknown_set = set(unknown_ids)
    pids = frame["vehicle_id"].to_numpy()
    cameras = frame["camera_id"].to_numpy()

    known_queries: list[int] = []
    mandatory_gallery: list[int] = []
    for vehicle_id in known_ids:
        identity_indices = np.flatnonzero(pids == vehicle_id)
        query_index = int(identity_indices[int(rng.integers(len(identity_indices)))])
        cross_camera = identity_indices[cameras[identity_indices] != cameras[query_index]]
        if not len(cross_camera):  # Guard even though validation checked cameras.
            raise RuntimeError(f"vehicle_id={vehicle_id!r} has no cross-camera positive")
        gallery_index = int(cross_camera[int(rng.integers(len(cross_camera)))])
        known_queries.append(query_index)
        mandatory_gallery.append(gallery_index)

    unknown_queries: list[int] = []
    for vehicle_id in unknown_ids:
        identity_indices = np.flatnonzero(pids == vehicle_id)
        query_index = int(
            identity_indices[int(rng.integers(len(identity_indices)))]
        )
        unknown_queries.append(query_index)

    query_set = set(known_queries)
    eligible_gallery = [
        index
        for index, vehicle_id in enumerate(pids.tolist())
        if vehicle_id in known_set and index not in query_set
    ]
    available = len(eligible_gallery)
    if target_gallery_size is None:
        target = min(750, available)
    else:
        target = _positive_int(target_gallery_size, "target_gallery_size")
    if target < len(mandatory_gallery):
        raise ValueError(
            "target_gallery_size is too small to include a cross-camera "
            "positive for every known identity"
        )
    if target > available:
        raise ValueError(
            f"target_gallery_size={target} exceeds {available} eligible known images"
        )

    mandatory_set = set(mandatory_gallery)
    extra_candidates = np.asarray(
        [index for index in eligible_gallery if index not in mandatory_set],
        dtype=np.int64,
    )
    if len(extra_candidates):
        extra_candidates = rng.permutation(extra_candidates)
    extra_count = target - len(mandatory_gallery)
    gallery_indices = np.asarray(
        mandatory_gallery + extra_candidates[:extra_count].tolist(),
        dtype=np.int64,
    )
    gallery_indices = rng.permutation(gallery_indices)

    query_indices = np.asarray(known_queries + unknown_queries, dtype=np.int64)
    query_indices = rng.permutation(query_indices)
    query_is_known = np.asarray(
        [pids[index] in known_set for index in query_indices], dtype=bool
    )
    protocol = GalleryConditionedProtocol(
        seed=int(seed),
        query_indices=query_indices,
        gallery_indices=gallery_indices,
        query_is_known=query_is_known,
        known_vehicle_ids=known_ids,
        unknown_vehicle_ids=unknown_ids,
        target_gallery_size=target,
    )
    _protocol_summary(protocol, frame)  # Enforce all invariants before return.
    return protocol


def _id_digest(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(_python_scalar(value)).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _protocol_summary(
    protocol: GalleryConditionedProtocol,
    annotations: pd.DataFrame,
) -> dict[str, Any]:
    frame = annotations.reset_index(drop=True)
    query = frame.iloc[protocol.query_indices]
    gallery = frame.iloc[protocol.gallery_indices]
    known_set = set(protocol.known_vehicle_ids)
    unknown_set = set(protocol.unknown_vehicle_ids)
    query_images = set(query["image_id"].tolist())
    gallery_images = set(gallery["image_id"].tolist())
    gallery_pids = set(gallery["vehicle_id"].tolist())
    if query_images.intersection(gallery_images):
        raise RuntimeError("query/gallery image leakage detected")
    if unknown_set.intersection(gallery_pids):
        raise RuntimeError("unknown identity leaked into gallery")
    unknown_query = query.loc[~protocol.query_is_known]
    if set(unknown_query["vehicle_id"].tolist()) != unknown_set:
        raise RuntimeError("unknown query reservation does not cover every unknown identity")
    if bool(unknown_query["vehicle_id"].duplicated().any()):
        raise RuntimeError("unknown query reservation contains duplicate identities")

    cross_camera_matches = 0
    for row in query.loc[protocol.query_is_known].itertuples(index=False):
        match = (gallery["vehicle_id"] == row.vehicle_id) & (
            gallery["camera_id"] != row.camera_id
        )
        if not bool(match.any()):
            raise RuntimeError("known query has no cross-camera gallery positive")
        cross_camera_matches += 1
    if set(query.loc[protocol.query_is_known, "vehicle_id"].tolist()) != known_set:
        raise RuntimeError("known query reservation does not cover every known identity")

    return {
        "seed": protocol.seed,
        "num_known_identities": len(known_set),
        "num_unknown_identities": len(unknown_set),
        "num_queries": len(query),
        "num_known_queries": int(protocol.query_is_known.sum()),
        "num_unknown_queries": int((~protocol.query_is_known).sum()),
        "unknown_query_fraction": float((~protocol.query_is_known).mean()),
        "num_gallery": len(gallery),
        "known_queries_with_cross_camera_positive": cross_camera_matches,
        "query_gallery_image_overlap": 0,
        "unknown_gallery_identity_overlap": 0,
        "query_image_ids_sha256": _id_digest(query["image_id"].tolist()),
        "gallery_image_ids_sha256": _id_digest(gallery["image_id"].tolist()),
        "known_vehicle_ids_sha256": _id_digest(protocol.known_vehicle_ids),
        "unknown_vehicle_ids_sha256": _id_digest(protocol.unknown_vehicle_ids),
    }


@dataclass(frozen=True)
class _SeedEvaluation:
    seed: int
    scores: np.ndarray
    is_known: np.ndarray
    top1_correct: np.ndarray
    query_groups: np.ndarray
    protocol_summary: dict[str, Any]
    optimum: dict[str, float | int]
    features: np.ndarray | None


def _evaluate_seed(
    embeddings: np.ndarray,
    annotations: pd.DataFrame,
    protocol: GalleryConditionedProtocol,
    *,
    dba_top_k: int,
    dba_alpha: float,
    qe_top_k: int,
    qe_alpha: float,
    include_features: bool,
    feature_density_k: int,
) -> _SeedEvaluation:
    query_indices = protocol.query_indices
    gallery_indices = protocol.gallery_indices
    # Mirror deployment exactly: network descriptors, the stored DBA gallery,
    # and the gallery-conditioned QE query are all persisted/searched as
    # float32.  The reranking helpers calculate internally in float64, hence
    # the explicit casts at the same boundaries as release inference.
    query = np.ascontiguousarray(embeddings[query_indices], dtype=np.float32)
    gallery = np.ascontiguousarray(embeddings[gallery_indices], dtype=np.float32)
    augmented_gallery = database_side_augmentation(
        gallery, top_k=dba_top_k, alpha=dba_alpha
    ).astype(np.float32, copy=False)
    expanded_query = query_expansion(
        query,
        augmented_gallery,
        top_k=qe_top_k,
        alpha=qe_alpha,
    ).astype(np.float32, copy=False)
    similarities = np.clip(expanded_query @ augmented_gallery.T, -1.0, 1.0)
    top_indices = np.argmax(similarities, axis=1)
    scores = similarities[np.arange(len(similarities)), top_indices]
    pids = annotations["vehicle_id"].to_numpy()
    query_pids = pids[query_indices]
    gallery_pids = pids[gallery_indices]
    correct = np.asarray(query_pids == gallery_pids[top_indices], dtype=bool)
    if np.any(correct & ~protocol.query_is_known):
        raise RuntimeError("unknown query unexpectedly has a correct gallery candidate")
    optimum = search_rejection_threshold(
        scores,
        protocol.query_is_known,
        top1_correct=correct,
    )
    features = None
    if include_features:
        gallery_similarity = np.clip(
            augmented_gallery @ augmented_gallery.T, -1.0, 1.0
        )
        features = extract_candidate_features(
            similarities,
            gallery_similarity,
            density_k=feature_density_k,
        )
    return _SeedEvaluation(
        seed=protocol.seed,
        scores=np.asarray(scores, dtype=np.float64),
        is_known=protocol.query_is_known.copy(),
        top1_correct=correct,
        query_groups=np.asarray(query_pids, dtype=object),
        protocol_summary=_protocol_summary(protocol, annotations),
        optimum={key: _python_scalar(value) for key, value in optimum.items()},
        features=features,
    )


def select_pooled_threshold(
    score_batches: Sequence[Any],
    known_batches: Sequence[Any],
    correct_batches: Sequence[Any],
) -> dict[str, float | int]:
    """Select one candidate-level micro-F1 threshold across all seed rows."""

    if not score_batches or not known_batches or not correct_batches:
        raise ValueError("pooled threshold requires at least one seed batch")
    if not (
        len(score_batches) == len(known_batches) == len(correct_batches)
    ):
        raise ValueError("pooled threshold batch lists must have equal lengths")
    scores: list[np.ndarray] = []
    known: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    for index, (score_batch, known_batch, correct_batch) in enumerate(
        zip(score_batches, known_batches, correct_batches, strict=True)
    ):
        score_array = np.asarray(score_batch, dtype=np.float64)
        known_array = np.asarray(known_batch)
        correct_array = np.asarray(correct_batch)
        if score_array.ndim != 1 or score_array.size == 0:
            raise ValueError(f"score batch {index} must be a non-empty vector")
        if known_array.shape != score_array.shape or correct_array.shape != score_array.shape:
            raise ValueError(f"pooled batch {index} arrays are not aligned")
        scores.append(score_array)
        known.append(known_array)
        correct.append(correct_array)
    result = search_rejection_threshold(
        np.concatenate(scores),
        np.concatenate(known),
        top1_correct=np.concatenate(correct),
    )
    return {key: _python_scalar(value) for key, value in result.items()}


def _mean_std(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    return {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "std": float(np.std([float(row[key]) for row in rows], ddof=0)),
        }
        for key in keys
    }


def _feature_columns(names: Sequence[str]) -> np.ndarray:
    positions = {name: index for index, name in enumerate(CANDIDATE_FEATURE_NAMES)}
    try:
        return np.asarray([positions[name] for name in names], dtype=np.int64)
    except KeyError as exc:  # Internal programming error, not user input.
        raise RuntimeError(f"unknown adaptive refusal feature: {exc.args[0]}") from exc


def _candidate_oof_metrics(
    probabilities: np.ndarray,
    is_known: np.ndarray,
    top1_correct: np.ndarray,
) -> dict[str, float | int]:
    """Apply the task's candidate-level F1 definition to OOF probabilities.

    A plain binary F1 over ``top1_correct`` is optimistic because a known query
    whose top-1 is wrong must still count as a missed match (FN), even when the
    wrong candidate is correctly rejected.  Reusing the central metric helper
    keeps scalar and adaptive refusal directly comparable.
    """

    result = search_rejection_threshold(
        probabilities,
        is_known,
        top1_correct=top1_correct,
    )
    return {key: _python_scalar(value) for key, value in result.items()}


def select_constrained_candidate_threshold(
    scores: Any,
    is_known: Any,
    top1_correct: Any,
    *,
    minimum_tnr: float,
) -> dict[str, float | int]:
    """Maximize candidate-level F1 subject to a tune-only TNR floor.

    Threshold ties follow the main refusal policy: prefer larger TNR and then
    the more conservative threshold.  Reject-all is always a feasible point
    for a valid ``minimum_tnr`` in ``[0, 1]``.
    """

    values = np.asarray(scores, dtype=np.float64)
    known = np.asarray(is_known)
    correct = np.asarray(top1_correct)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")
    if known.shape != values.shape or not np.isin(known, [0, 1, False, True]).all():
        raise ValueError("is_known must be a binary vector aligned with scores")
    if correct.shape != values.shape or not np.isin(
        correct, [0, 1, False, True]
    ).all():
        raise ValueError("top1_correct must be a binary vector aligned with scores")
    known = known.astype(bool, copy=False)
    correct = correct.astype(bool, copy=False)
    if known.all() or (~known).all():
        raise ValueError("open-set threshold selection requires known and unknown rows")
    if np.any(correct & ~known):
        raise ValueError("an unknown query cannot have a correct top-1 candidate")
    try:
        minimum_tnr = float(minimum_tnr)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_tnr must be within [0, 1]") from exc
    if not np.isfinite(minimum_tnr) or not 0.0 <= minimum_tnr <= 1.0:
        raise ValueError("minimum_tnr must be within [0, 1]")

    order = np.argsort(-values, kind="mergesort")
    sorted_scores = values[order]
    sorted_known = known[order]
    sorted_correct = correct[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    cumulative_tp = np.cumsum(sorted_correct, dtype=np.int64)
    cumulative_accepted = np.arange(1, len(values) + 1, dtype=np.int64)
    cumulative_unknown = np.cumsum(~sorted_known, dtype=np.int64)
    num_known = int(known.sum())
    num_unknown = int((~known).sum())

    # Start with reject-all.  It is needed for minimum_tnr=1 and also makes
    # the function total even for pathological score distributions.
    reject_all_threshold = float(np.nextafter(sorted_scores[0], np.inf))
    best_key: tuple[float, float, float] = (0.0, 1.0, reject_all_threshold)
    best_threshold = reject_all_threshold
    tolerance = 1e-15
    for end in group_ends.tolist():
        tp = int(cumulative_tp[end])
        accepted = int(cumulative_accepted[end])
        fp = accepted - tp
        fn = num_known - tp
        unknown_fp = int(cumulative_unknown[end])
        tnr = (num_unknown - unknown_fp) / num_unknown
        if tnr + tolerance < minimum_tnr:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        threshold = float(sorted_scores[end])
        key = (float(f1), float(tnr), threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold

    result = evaluate_open_set(
        values,
        known,
        threshold=best_threshold,
        top1_correct=correct,
    )
    if float(result["tnr"]) + tolerance < minimum_tnr:
        raise RuntimeError("selected threshold violates the requested TNR floor")
    result["minimum_tnr"] = minimum_tnr
    return {key: _python_scalar(value) for key, value in result.items()}


def select_weighted_candidate_threshold(
    scores: Any,
    is_known: Any,
    top1_correct: Any,
    *,
    f1_weight: float = 0.7,
    tnr_weight: float = 0.3,
) -> dict[str, float | int]:
    """Maximize the hackathon's official weighted candidate score.

    The organizers define the candidate block as ``0.7 * F1 + 0.3 * TNR``.
    Threshold ties prefer the larger F1, then larger TNR, and finally the more
    conservative threshold.  Selection uses tune-only predictions exactly like
    the existing F1 and constrained-TNR operating points.
    """

    values = np.asarray(scores, dtype=np.float64)
    known = np.asarray(is_known)
    correct = np.asarray(top1_correct)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")
    if known.shape != values.shape or not np.isin(known, [0, 1, False, True]).all():
        raise ValueError("is_known must be a binary vector aligned with scores")
    if correct.shape != values.shape or not np.isin(
        correct, [0, 1, False, True]
    ).all():
        raise ValueError("top1_correct must be a binary vector aligned with scores")
    known = known.astype(bool, copy=False)
    correct = correct.astype(bool, copy=False)
    if known.all() or (~known).all():
        raise ValueError("open-set threshold selection requires known and unknown rows")
    if np.any(correct & ~known):
        raise ValueError("an unknown query cannot have a correct top-1 candidate")
    try:
        f1_weight = float(f1_weight)
        tnr_weight = float(tnr_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate score weights must be finite and non-negative") from exc
    if (
        not np.isfinite(f1_weight)
        or not np.isfinite(tnr_weight)
        or f1_weight < 0.0
        or tnr_weight < 0.0
        or not np.isclose(f1_weight + tnr_weight, 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("candidate score weights must be non-negative and sum to 1")

    order = np.argsort(-values, kind="mergesort")
    sorted_scores = values[order]
    sorted_known = known[order]
    sorted_correct = correct[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    cumulative_tp = np.cumsum(sorted_correct, dtype=np.int64)
    cumulative_unknown = np.cumsum(~sorted_known, dtype=np.int64)
    num_known = int(known.sum())
    num_unknown = int((~known).sum())

    reject_all_threshold = float(np.nextafter(sorted_scores[0], np.inf))
    best_key = (tnr_weight, 0.0, 1.0, reject_all_threshold)
    best_threshold = reject_all_threshold
    for end in group_ends.tolist():
        tp = int(cumulative_tp[end])
        accepted = end + 1
        fp = accepted - tp
        fn = num_known - tp
        unknown_fp = int(cumulative_unknown[end])
        tnr = (num_unknown - unknown_fp) / num_unknown
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        official_score = f1_weight * f1 + tnr_weight * tnr
        threshold = float(sorted_scores[end])
        key = (float(official_score), float(f1), float(tnr), threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold

    result = evaluate_open_set(
        values,
        known,
        threshold=best_threshold,
        top1_correct=correct,
    )
    result["candidate_score"] = (
        f1_weight * float(result["f1"]) + tnr_weight * float(result["tnr"])
    )
    result["f1_weight"] = f1_weight
    result["tnr_weight"] = tnr_weight
    return {key: _python_scalar(value) for key, value in result.items()}


def _fit_adaptive_feature_candidates(
    features: np.ndarray,
    is_known: np.ndarray,
    top1_correct: np.ndarray,
    groups: np.ndarray,
    *,
    oof_splits: int,
    feature_variant: str | None = None,
) -> tuple[dict[str, Any], Any, float, np.ndarray]:
    """Tune feature family and threshold using group-disjoint OOF predictions."""

    candidates: dict[str, Any] = {}
    best_key: tuple[float, float, float, str] | None = None
    best_name: str | None = None
    best_oof: dict[str, Any] | None = None
    variants = _ADAPTIVE_FEATURE_VARIANTS
    if feature_variant is not None:
        variants = tuple(
            item for item in variants if item[0] == feature_variant
        )
        if not variants:
            raise ValueError(f"unknown adaptive feature variant: {feature_variant!r}")
    for name, feature_names in variants:
        columns = _feature_columns(feature_names)
        oof = evaluate_group_oof(
            features[:, columns],
            top1_correct,
            groups,
            n_splits=oof_splits,
            feature_names=feature_names,
        )
        candidate_metrics = _candidate_oof_metrics(
            oof["probabilities"], is_known, top1_correct
        )
        candidates[name] = {
            "feature_names": list(feature_names),
            "binary_correctness_metrics": {
                key: _python_scalar(value)
                for key, value in oof["metrics"].items()
            },
            "candidate_open_set_metrics": candidate_metrics,
            "folds": oof["folds"],
        }
        # All selection signals are tune OOF only.  Prefer the task metric,
        # then TNR, then PR-AUC, and finally the stable variant name.
        selection_key = (
            float(candidate_metrics["f1"]),
            float(candidate_metrics["tnr"]),
            float(candidate_metrics["pr_auc"]),
            name,
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_name = name
            best_oof = oof

    assert best_name is not None and best_oof is not None
    selected_names = tuple(candidates[best_name]["feature_names"])
    selected_columns = _feature_columns(selected_names)
    model = fit_candidate_correctness_model(
        features[:, selected_columns],
        top1_correct,
        feature_names=selected_names,
    )
    threshold = float(
        candidates[best_name]["candidate_open_set_metrics"]["threshold"]
    )
    return (
        {
            "selection_metric": "candidate_open_set_f1",
            "selection_tie_break": ["tnr", "pr_auc", "variant_name"],
            "selected_variant": best_name,
            "selected_feature_names": list(selected_names),
            "selected_probability_threshold": threshold,
            "candidates": candidates,
        },
        model,
        threshold,
        np.asarray(best_oof["probabilities"], dtype=np.float64),
    )


def _fixed_probability_evaluation(
    evaluations: Sequence[_SeedEvaluation],
    *,
    model: Any,
    feature_names: Sequence[str],
    probability_threshold: float,
    raw_threshold: float,
) -> dict[str, Any]:
    """Evaluate frozen scalar/adaptive policies on untouched seed protocols."""

    columns = _feature_columns(feature_names)
    scalar_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    probability_batches: list[np.ndarray] = []
    for evaluation in evaluations:
        assert evaluation.features is not None
        probabilities = model.predict_proba(evaluation.features[:, columns])[:, 1]
        probability_batches.append(probabilities)
        scalar = evaluate_open_set(
            evaluation.scores,
            evaluation.is_known,
            threshold=raw_threshold,
            top1_correct=evaluation.top1_correct,
        )
        adaptive = evaluate_open_set(
            probabilities,
            evaluation.is_known,
            threshold=probability_threshold,
            top1_correct=evaluation.top1_correct,
        )
        scalar_rows.append({key: _python_scalar(value) for key, value in scalar.items()})
        adaptive_rows.append(
            {key: _python_scalar(value) for key, value in adaptive.items()}
        )

    known = np.concatenate([evaluation.is_known for evaluation in evaluations])
    correct = np.concatenate(
        [evaluation.top1_correct for evaluation in evaluations]
    )
    scores = np.concatenate([evaluation.scores for evaluation in evaluations])
    probabilities = np.concatenate(probability_batches)
    pooled_scalar = evaluate_open_set(
        scores,
        known,
        threshold=raw_threshold,
        top1_correct=correct,
    )
    pooled_adaptive = evaluate_open_set(
        probabilities,
        known,
        threshold=probability_threshold,
        top1_correct=correct,
    )
    aggregate_keys = ("f1", "tnr", "pr_auc", "precision", "recall")
    return {
        "num_samples": int(len(known)),
        "scalar_raw_cosine": {
            "threshold": float(raw_threshold),
            "pooled_metrics": {
                key: _python_scalar(value) for key, value in pooled_scalar.items()
            },
            "seed_mean_std": _mean_std(scalar_rows, aggregate_keys),
        },
        "adaptive_probability": {
            "threshold": float(probability_threshold),
            "pooled_metrics": {
                key: _python_scalar(value) for key, value in pooled_adaptive.items()
            },
            "seed_mean_std": _mean_std(adaptive_rows, aggregate_keys),
        },
        "per_seed": [
            {
                "seed": evaluation.seed,
                "protocol": evaluation.protocol_summary,
                "scalar_raw_cosine": scalar,
                "adaptive_probability": adaptive,
            }
            for evaluation, scalar, adaptive in zip(
                evaluations, scalar_rows, adaptive_rows, strict=True
            )
        ],
    }


def calibrate_robust_open_set(
    embeddings: Any,
    annotations: pd.DataFrame,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    unknown_fraction: float = 0.25,
    target_gallery_size: int | None = None,
    dba_top_k: int = 3,
    dba_alpha: float = 2.0,
    qe_top_k: int = 2,
    qe_alpha: float = 1.0,
    fit_oof_features: bool = False,
    oof_splits: int = 5,
    feature_density_k: int = 5,
    adaptive_confirmation_seeds: Sequence[int] | None = None,
    adaptive_feature_variant: str | None = None,
) -> dict[str, Any]:
    """Run repeated protocols and return a fully JSON-serializable report."""

    frame = _validate_annotations(annotations)
    normalized = l2_normalize_embeddings(embeddings, name="embeddings")
    if len(normalized) != len(frame):
        raise ValueError("embeddings and annotation rows are not aligned")
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    clean_seeds: list[int] = []
    for seed in seeds:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise ValueError("seeds must contain only integers")
        clean_seeds.append(int(seed))
    if len(set(clean_seeds)) != len(clean_seeds):
        raise ValueError("seeds must be unique")
    clean_confirmation_seeds: list[int] = []
    if adaptive_confirmation_seeds is not None:
        if not fit_oof_features:
            raise ValueError(
                "adaptive_confirmation_seeds requires fit_oof_features=True"
            )
        for seed in adaptive_confirmation_seeds:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(
                seed, (int, np.integer)
            ):
                raise ValueError(
                    "adaptive_confirmation_seeds must contain only integers"
                )
            clean_confirmation_seeds.append(int(seed))
        if not clean_confirmation_seeds:
            raise ValueError("adaptive_confirmation_seeds must not be empty")
        if len(set(clean_confirmation_seeds)) != len(clean_confirmation_seeds):
            raise ValueError("adaptive_confirmation_seeds must be unique")
        overlap = sorted(set(clean_seeds).intersection(clean_confirmation_seeds))
        if overlap:
            raise ValueError(
                "adaptive confirmation seeds must be disjoint from tune seeds; "
                f"overlap={overlap}"
            )

    dba_top_k = _positive_int(dba_top_k, "dba_top_k")
    qe_top_k = _positive_int(qe_top_k, "qe_top_k")
    feature_density_k = _positive_int(feature_density_k, "feature_density_k")
    evaluations: list[_SeedEvaluation] = []
    for seed in clean_seeds:
        protocol = build_gallery_conditioned_protocol(
            frame,
            seed=seed,
            unknown_fraction=unknown_fraction,
            target_gallery_size=target_gallery_size,
        )
        evaluations.append(
            _evaluate_seed(
                normalized,
                frame,
                protocol,
                dba_top_k=dba_top_k,
                dba_alpha=dba_alpha,
                qe_top_k=qe_top_k,
                qe_alpha=qe_alpha,
                include_features=bool(fit_oof_features),
                feature_density_k=feature_density_k,
            )
        )

    pooled = select_pooled_threshold(
        [evaluation.scores for evaluation in evaluations],
        [evaluation.is_known for evaluation in evaluations],
        [evaluation.top1_correct for evaluation in evaluations],
    )
    raw_threshold = float(pooled["threshold"])
    pooled_scores = np.concatenate(
        [evaluation.scores for evaluation in evaluations]
    )
    pooled_correct = np.concatenate(
        [evaluation.top1_correct for evaluation in evaluations]
    )
    confidence_calibration = fit_platt_calibration(
        pooled_scores,
        pooled_correct,
        fallback_threshold=raw_threshold,
    )
    confidence_threshold = float(
        calibrated_confidence(raw_threshold, confidence_calibration)
    )
    fixed_rows: list[dict[str, float | int]] = []
    seed_rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        fixed = evaluate_open_set(
            evaluation.scores,
            evaluation.is_known,
            threshold=raw_threshold,
            top1_correct=evaluation.top1_correct,
        )
        fixed = {key: _python_scalar(value) for key, value in fixed.items()}
        fixed_rows.append(fixed)
        seed_rows.append(
            {
                "seed": evaluation.seed,
                "protocol": evaluation.protocol_summary,
                "seed_optimum": evaluation.optimum,
                "fixed_pooled_threshold": fixed,
            }
        )

    aggregate_keys = ("f1", "tnr", "pr_auc", "precision", "recall")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "method": "gallery_conditioned_dba_qe_raw_cosine",
        "configuration": {
            "seeds": clean_seeds,
            "unknown_fraction": float(unknown_fraction),
            "target_gallery_size": target_gallery_size,
            "default_target_rule": "min(750, eligible_known_images)",
            "dba": {"top_k": dba_top_k, "alpha": float(dba_alpha)},
            "query_expansion": {"top_k": qe_top_k, "alpha": float(qe_alpha)},
            "known_queries_per_identity": 1,
            "feature_density_k": feature_density_k,
            "oof_splits": int(oof_splits),
        },
        "raw_cosine_threshold": raw_threshold,
        "threshold_domain": "cosine(expanded_query, dba_gallery)",
        "confidence_calibration": confidence_calibration,
        "confidence_threshold": confidence_threshold,
        "pooled_micro_metrics": pooled,
        "fixed_threshold_mean_std": _mean_std(fixed_rows, aggregate_keys),
        "seed_optimum_mean_std": _mean_std(
            [evaluation.optimum for evaluation in evaluations],
            ("threshold",) + aggregate_keys,
        ),
        "seeds": seed_rows,
        "protocol_guarantees": {
            "identity_disjoint_known_unknown": True,
            "unknown_identities_absent_from_gallery": True,
            "all_unknown_images_are_queries": True,
            "query_gallery_image_disjoint": True,
            "every_known_query_has_cross_camera_positive": True,
            "gallery_fill_uses_known_images_only": True,
        },
        "candidate_f1_definition": (
            "accepted correct top1=TP; accepted wrong top1=FP; a known query "
            "without an accepted correct top1=FN; TNR uses unknown queries only"
        ),
        "limitations": [
            "Validation CSV rows must be aligned exactly with embeddings.npy rows.",
            "Every identity must have at least two observed cameras.",
            "Repeated seeds reuse identities and measure protocol sensitivity, not independent data uncertainty.",
            "A fixed gallery size does not reproduce unknown production identity frequencies or temporal priors.",
            "DBA and QE are gallery-conditioned, so changing gallery contents requires recalibration.",
        ],
    }

    if fit_oof_features:
        feature_batches = [evaluation.features for evaluation in evaluations]
        assert all(features is not None for features in feature_batches)
        features = np.concatenate(feature_batches, axis=0)  # type: ignore[arg-type]
        labels = np.concatenate(
            [evaluation.top1_correct for evaluation in evaluations]
        )
        known = np.concatenate([evaluation.is_known for evaluation in evaluations])
        groups = np.concatenate([evaluation.query_groups for evaluation in evaluations])
        selection, final_model, probability_threshold, selected_oof_probabilities = (
            _fit_adaptive_feature_candidates(
                features,
                known,
                labels,
                groups,
                oof_splits=oof_splits,
                feature_variant=adaptive_feature_variant,
            )
        )
        tune_operating_points: dict[str, dict[str, Any]] = {
            "official_candidate_score": {
                "selection_rule": "maximize_0.7_f1_plus_0.3_tnr",
                "metrics": select_weighted_candidate_threshold(
                    selected_oof_probabilities,
                    known,
                    labels,
                    f1_weight=0.7,
                    tnr_weight=0.3,
                ),
            },
            "max_candidate_f1": {
                "selection_rule": "maximize_candidate_f1",
                "metrics": selection["candidates"][selection["selected_variant"]][
                    "candidate_open_set_metrics"
                ],
            },
            "tnr_at_least_0_97": {
                "selection_rule": "maximize_candidate_f1_subject_to_minimum_tnr",
                "metrics": select_constrained_candidate_threshold(
                    selected_oof_probabilities,
                    known,
                    labels,
                    minimum_tnr=0.97,
                ),
            },
        }
        scalar_tune_tnr = float(pooled["tnr"])
        if not np.isclose(scalar_tune_tnr, 0.97, rtol=0.0, atol=1e-12):
            tune_operating_points["tnr_at_least_scalar_tune"] = {
                "selection_rule": (
                    "maximize_candidate_f1_subject_to_minimum_tnr"
                ),
                "reference": "scalar_raw_cosine_tune_tnr",
                "reference_tnr": scalar_tune_tnr,
                "metrics": select_constrained_candidate_threshold(
                    selected_oof_probabilities,
                    known,
                    labels,
                    minimum_tnr=scalar_tune_tnr,
                ),
            }
        selection["tune_operating_points"] = tune_operating_points
        selected = selection["candidates"][selection["selected_variant"]]
        report["optional_feature_correctness_model"] = {
            "diagnostic_only": True,
            "does_not_replace_raw_cosine_threshold": True,
            "feature_names": selection["selected_feature_names"],
            # Kept for compatibility: these are ordinary binary metrics for
            # predicting correctness, not the task's candidate-level F1.
            "oof_metrics": selected["binary_correctness_metrics"],
            "candidate_open_set_oof_metrics": selected[
                "candidate_open_set_metrics"
            ],
            "folds": selected["folds"],
            "tune_only_feature_search": selection,
            "model": final_model.to_dict(),
        }
        if clean_confirmation_seeds:
            confirmation_evaluations: list[_SeedEvaluation] = []
            for seed in clean_confirmation_seeds:
                protocol = build_gallery_conditioned_protocol(
                    frame,
                    seed=seed,
                    unknown_fraction=unknown_fraction,
                    target_gallery_size=target_gallery_size,
                )
                confirmation_evaluations.append(
                    _evaluate_seed(
                        normalized,
                        frame,
                        protocol,
                        dba_top_k=dba_top_k,
                        dba_alpha=dba_alpha,
                        qe_top_k=qe_top_k,
                        qe_alpha=qe_alpha,
                        include_features=True,
                        feature_density_k=feature_density_k,
                    )
                )
            confirmation = _fixed_probability_evaluation(
                confirmation_evaluations,
                model=final_model,
                feature_names=selection["selected_feature_names"],
                probability_threshold=probability_threshold,
                raw_threshold=raw_threshold,
            )
            adaptive_operating_points: dict[str, Any] = {}
            for name, tune_point in tune_operating_points.items():
                point_threshold = float(tune_point["metrics"]["threshold"])
                point_evaluation = _fixed_probability_evaluation(
                    confirmation_evaluations,
                    model=final_model,
                    feature_names=selection["selected_feature_names"],
                    probability_threshold=point_threshold,
                    raw_threshold=raw_threshold,
                )
                adaptive_operating_points[name] = {
                    "selection_rule": tune_point["selection_rule"],
                    "reference": tune_point.get("reference"),
                    "reference_tnr": tune_point.get("reference_tnr"),
                    "tune_metrics": tune_point["metrics"],
                    "confirmation_metrics": point_evaluation[
                        "adaptive_probability"
                    ],
                    "per_seed": [
                        {
                            "seed": row["seed"],
                            "metrics": row["adaptive_probability"],
                        }
                        for row in point_evaluation["per_seed"]
                    ],
                }
            confirmation["adaptive_operating_points"] = (
                adaptive_operating_points
            )
            confirmation.update(
                {
                    "seeds": clean_confirmation_seeds,
                    "locked_before_evaluation": True,
                    "used_for_feature_selection": False,
                    "used_for_model_fit": False,
                    "used_for_threshold_fit": False,
                    "thresholds_derived_from_tune_oof_only": True,
                }
            )
            report["optional_feature_correctness_model"][
                "locked_confirmation"
            ] = confirmation

    # Fail here rather than emit a JSON file containing NaN or NumPy scalars.
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate_from_files(
    embeddings_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load aligned artifacts, calibrate, and atomically write the JSON report."""

    embeddings_path = Path(embeddings_path)
    annotations_path = Path(annotations_path)
    output_path = Path(output_path)
    embeddings = np.load(embeddings_path, allow_pickle=False)
    annotations = pd.read_csv(annotations_path)
    report = calibrate_robust_open_set(embeddings, annotations, **kwargs)
    report["inputs"] = {
        "embeddings": str(embeddings_path.resolve()),
        "embeddings_sha256": _sha256_file(embeddings_path),
        "annotations": str(annotations_path.resolve()),
        "annotations_sha256": _sha256_file(annotations_path),
        "num_rows": len(annotations),
        "embedding_dim": int(embeddings.shape[1]),
    }
    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                f"Calibration checkpoint does not exist: {resolved_checkpoint}"
            )
        report["inputs"].update(
            {
                "checkpoint": str(resolved_checkpoint),
                "checkpoint_sha256": _sha256_file(resolved_checkpoint),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(output_path)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", type=Path, default=Path("outputs/val_embeddings.npy")
    )
    parser.add_argument("--annotations", type=Path, default=Path("splits/val.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robust_open_set_calibration.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint that produced the aligned embeddings; records a SHA-256 provenance binding.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--unknown-fraction", type=float, default=0.25)
    parser.add_argument(
        "--target-gallery-size",
        type=int,
        default=None,
        help="Defaults independently per seed to min(750, eligible known images).",
    )
    parser.add_argument("--dba-top-k", type=int, default=3)
    parser.add_argument("--dba-alpha", type=float, default=2.0)
    parser.add_argument("--qe-top-k", type=int, default=2)
    parser.add_argument("--qe-alpha", type=float, default=1.0)
    parser.add_argument("--fit-oof-features", action="store_true")
    parser.add_argument("--oof-splits", type=int, default=5)
    parser.add_argument("--feature-density-k", type=int, default=5)
    parser.add_argument(
        "--adaptive-feature-variant",
        choices=[name for name, _ in _ADAPTIVE_FEATURE_VARIANTS],
        default=None,
        help="Optionally freeze the adaptive feature schema before OOF fitting.",
    )
    parser.add_argument(
        "--adaptive-confirmation-seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Locked seeds evaluated only after tune-seed OOF feature-family, "
            "model and probability-threshold fitting. Requires "
            "--fit-oof-features."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = calibrate_from_files(
        args.embeddings,
        args.annotations,
        args.output,
        checkpoint_path=args.checkpoint,
        seeds=args.seeds,
        unknown_fraction=args.unknown_fraction,
        target_gallery_size=args.target_gallery_size,
        dba_top_k=args.dba_top_k,
        dba_alpha=args.dba_alpha,
        qe_top_k=args.qe_top_k,
        qe_alpha=args.qe_alpha,
        fit_oof_features=args.fit_oof_features,
        oof_splits=args.oof_splits,
        feature_density_k=args.feature_density_k,
        adaptive_confirmation_seeds=args.adaptive_confirmation_seeds,
        adaptive_feature_variant=args.adaptive_feature_variant,
    )
    aggregate = report["fixed_threshold_mean_std"]
    print(
        "raw_threshold={:.8f} F1={:.4f}±{:.4f} TNR={:.4f}±{:.4f} "
        "PR-AUC={:.4f}±{:.4f}".format(
            report["raw_cosine_threshold"],
            aggregate["f1"]["mean"],
            aggregate["f1"]["std"],
            aggregate["tnr"]["mean"],
            aggregate["tnr"]["std"],
            aggregate["pr_auc"]["mean"],
            aggregate["pr_auc"]["std"],
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_SEEDS",
    "GalleryConditionedProtocol",
    "build_gallery_conditioned_protocol",
    "calibrate_from_files",
    "calibrate_robust_open_set",
    "select_constrained_candidate_threshold",
    "select_weighted_candidate_threshold",
    "select_pooled_threshold",
]
