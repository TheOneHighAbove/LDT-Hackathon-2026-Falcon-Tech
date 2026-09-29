"""Cross-camera retrieval and open-set rejection metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _embeddings(value: Any, name: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, D] matrix")
    try:
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms <= np.finfo(np.float64).eps).any():
        raise ValueError(f"{name} contains a zero-norm embedding")
    return array / norms


def _metadata(value: Any, name: str, expected_length: int) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    array = _as_numpy(value)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(f"{name} must be one-dimensional with length {expected_length}")
    for item in array.tolist():
        if item is None:
            raise ValueError(f"{name} contains a missing value")
        try:
            unequal_to_self = item != item
            is_missing = bool(unequal_to_self)
        except (TypeError, ValueError):
            # Some custom identifiers do not define a scalar inequality; they
            # remain usable as long as NumPy can compare them below.
            is_missing = False
        if is_missing:
            raise ValueError(f"{name} contains a missing value")
    return array


def cosine_similarity_matrix(
    query_embeddings: Any, gallery_embeddings: Any
) -> np.ndarray:
    """Compute a finite cosine-similarity matrix after safe L2 normalization."""

    query = _embeddings(query_embeddings, "query_embeddings")
    gallery = _embeddings(gallery_embeddings, "gallery_embeddings")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions do not match")
    # Clip only tiny floating-point excursions outside cosine's valid range.
    return np.clip(query @ gallery.T, -1.0, 1.0)


def _prepare_protocol(
    query_embeddings: Any,
    gallery_embeddings: Any,
    query_pids: Any,
    gallery_pids: Any,
    query_camera_ids: Any | None,
    gallery_camera_ids: Any | None,
    query_image_ids: Any | None,
    gallery_image_ids: Any | None,
    same_source: bool | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    bool,
]:
    inferred_same_source = query_embeddings is gallery_embeddings
    scores = cosine_similarity_matrix(query_embeddings, gallery_embeddings)
    query_pid_array = _metadata(query_pids, "query_pids", scores.shape[0])
    gallery_pid_array = _metadata(gallery_pids, "gallery_pids", scores.shape[1])

    if (query_camera_ids is None) != (gallery_camera_ids is None):
        raise ValueError("query and gallery camera IDs must either both be set or both omitted")
    query_cameras = (
        _metadata(query_camera_ids, "query_camera_ids", scores.shape[0])
        if query_camera_ids is not None
        else None
    )
    gallery_cameras = (
        _metadata(gallery_camera_ids, "gallery_camera_ids", scores.shape[1])
        if gallery_camera_ids is not None
        else None
    )

    if (query_image_ids is None) != (gallery_image_ids is None):
        raise ValueError("query and gallery image IDs must either both be set or both omitted")
    query_images = (
        _metadata(query_image_ids, "query_image_ids", scores.shape[0])
        if query_image_ids is not None
        else None
    )
    gallery_images = (
        _metadata(gallery_image_ids, "gallery_image_ids", scores.shape[1])
        if gallery_image_ids is not None
        else None
    )
    if same_source is None:
        same_source = inferred_same_source
    same_source = bool(same_source)
    if same_source and query_images is None and scores.shape[0] != scores.shape[1]:
        raise ValueError(
            "same_source without image IDs requires aligned query/gallery lengths"
        )
    return (
        scores,
        query_pid_array,
        gallery_pid_array,
        query_cameras,
        gallery_cameras,
        query_images,
        gallery_images,
        same_source,
    )


def _valid_gallery_mask(
    query_index: int,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_cameras: np.ndarray | None,
    gallery_cameras: np.ndarray | None,
    query_images: np.ndarray | None,
    gallery_images: np.ndarray | None,
    same_source: bool,
) -> np.ndarray:
    valid = np.ones(len(gallery_pids), dtype=bool)
    if query_images is not None and gallery_images is not None:
        # Image IDs make self-removal correct even if query/gallery row order differs.
        valid &= gallery_images != query_images[query_index]
    elif same_source:
        valid[query_index] = False
    if query_cameras is not None and gallery_cameras is not None:
        same_pid_same_camera = (gallery_pids == query_pids[query_index]) & (
            gallery_cameras == query_cameras[query_index]
        )
        valid &= ~same_pid_same_camera
    return valid


def evaluate_retrieval(
    query_embeddings: Any,
    gallery_embeddings: Any,
    query_pids: Any,
    gallery_pids: Any,
    *,
    query_camera_ids: Any | None = None,
    gallery_camera_ids: Any | None = None,
    query_image_ids: Any | None = None,
    gallery_image_ids: Any | None = None,
    same_source: bool | None = None,
) -> dict[str, float | int]:
    """Evaluate standard ReID mAP/CMC/mINP with junk samples removed.

    For every query, its exact image and every gallery image with the same
    identity *and* camera are excluded.  Queries without a remaining positive
    are ignored in metric means and reported through ``num_ignored_queries``.
    Stable sorting makes tied similarities reproducible.
    """

    (
        scores,
        query_pid_array,
        gallery_pid_array,
        query_cameras,
        gallery_cameras,
        query_images,
        gallery_images,
        same_source,
    ) = _prepare_protocol(
        query_embeddings,
        gallery_embeddings,
        query_pids,
        gallery_pids,
        query_camera_ids,
        gallery_camera_ids,
        query_image_ids,
        gallery_image_ids,
        same_source,
    )

    average_precisions: list[float] = []
    inverse_negative_penalties: list[float] = []
    rank1_hits: list[float] = []
    rank5_hits: list[float] = []
    for query_index in range(scores.shape[0]):
        valid = _valid_gallery_mask(
            query_index,
            query_pid_array,
            gallery_pid_array,
            query_cameras,
            gallery_cameras,
            query_images,
            gallery_images,
            same_source,
        )
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            continue
        order = np.argsort(-scores[query_index, valid_indices], kind="mergesort")
        ranked_indices = valid_indices[order]
        matches = gallery_pid_array[ranked_indices] == query_pid_array[query_index]
        positive_ranks = np.flatnonzero(matches)
        if not len(positive_ranks):
            continue

        one_based_ranks = positive_ranks + 1
        precision_at_positives = np.arange(1, len(positive_ranks) + 1) / one_based_ranks
        average_precisions.append(float(precision_at_positives.mean()))
        inverse_negative_penalties.append(
            float(len(positive_ranks) / one_based_ranks[-1])
        )
        rank1_hits.append(float(positive_ranks[0] < 1))
        rank5_hits.append(float(positive_ranks[0] < 5))

    valid_queries = len(average_precisions)
    total_queries = scores.shape[0]
    if valid_queries == 0:
        return {
            "mAP": 0.0,
            "rank1": 0.0,
            "rank5": 0.0,
            "mINP": 0.0,
            "num_queries": int(total_queries),
            "num_valid_queries": 0,
            "num_ignored_queries": int(total_queries),
        }
    return {
        "mAP": float(np.mean(average_precisions)),
        "rank1": float(np.mean(rank1_hits)),
        "rank5": float(np.mean(rank5_hits)),
        "mINP": float(np.mean(inverse_negative_penalties)),
        "num_queries": int(total_queries),
        "num_valid_queries": int(valid_queries),
        "num_ignored_queries": int(total_queries - valid_queries),
    }


# Common name used in training loops.
compute_retrieval_metrics = evaluate_retrieval


def _id_set(values: Iterable[Any] | None, name: str) -> set[Any] | None:
    if values is None:
        return None
    result = set(values)
    if not result:
        raise ValueError(f"{name} must not be empty when provided")
    return result


def derive_known_mask(
    query_pids: Sequence[Any],
    gallery_pids: Sequence[Any],
    *,
    known_pids: Iterable[Any] | None = None,
    unknown_pids: Iterable[Any] | None = None,
) -> np.ndarray:
    """Label open-set queries from explicit known/unknown identity sets.

    With neither set supplied, an identity is known iff it appears in gallery.
    With both supplied, every query identity must be explicitly covered.
    """

    query = _metadata(query_pids, "query_pids", len(query_pids))
    gallery = _metadata(gallery_pids, "gallery_pids", len(gallery_pids))
    known = _id_set(known_pids, "known_pids")
    unknown = _id_set(unknown_pids, "unknown_pids")
    if known is not None and unknown is not None:
        overlap = known.intersection(unknown)
        if overlap:
            raise ValueError(f"known_pids and unknown_pids overlap: {list(overlap)[:5]!r}")
        uncovered = set(query.tolist()).difference(known.union(unknown))
        if uncovered:
            raise ValueError(f"query identities are not assigned: {list(uncovered)[:5]!r}")
    if known is None:
        known = set(gallery.tolist())
        if unknown is not None:
            known.difference_update(unknown)
    mask = np.asarray([pid in known for pid in query.tolist()], dtype=bool)
    if unknown is not None:
        mask &= np.asarray([pid not in unknown for pid in query.tolist()], dtype=bool)
    return mask


def _top1_and_match_availability(
    protocol: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        bool,
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (
        scores,
        query_pid_array,
        gallery_pid_array,
        query_cameras,
        gallery_cameras,
        query_images,
        gallery_images,
        same_source,
    ) = protocol
    top1 = np.full(scores.shape[0], -1.0, dtype=np.float64)
    has_positive = np.zeros(scores.shape[0], dtype=bool)
    top1_correct = np.zeros(scores.shape[0], dtype=bool)
    for query_index in range(scores.shape[0]):
        valid = _valid_gallery_mask(
            query_index,
            query_pid_array,
            gallery_pid_array,
            query_cameras,
            gallery_cameras,
            query_images,
            gallery_images,
            same_source,
        )
        if valid.any():
            valid_indices = np.flatnonzero(valid)
            best_index = valid_indices[
                int(np.argmax(scores[query_index, valid_indices]))
            ]
            top1[query_index] = float(scores[query_index, best_index])
            has_positive[query_index] = bool(
                np.any(valid & (gallery_pid_array == query_pid_array[query_index]))
            )
            top1_correct[query_index] = bool(
                gallery_pid_array[best_index] == query_pid_array[query_index]
            )
    return top1, has_positive, top1_correct


def open_set_top1_scores(
    query_embeddings: Any,
    gallery_embeddings: Any,
    query_pids: Any,
    gallery_pids: Any,
    *,
    query_camera_ids: Any | None = None,
    gallery_camera_ids: Any | None = None,
    query_image_ids: Any | None = None,
    gallery_image_ids: Any | None = None,
    same_source: bool | None = None,
) -> np.ndarray:
    """Return each query's maximum valid gallery cosine similarity."""

    protocol = _prepare_protocol(
        query_embeddings,
        gallery_embeddings,
        query_pids,
        gallery_pids,
        query_camera_ids,
        gallery_camera_ids,
        query_image_ids,
        gallery_image_ids,
        same_source,
    )
    top1, _, _ = _top1_and_match_availability(protocol)
    return top1


def _binary_metrics(
    scores: np.ndarray,
    is_known: np.ndarray,
    threshold: float,
    top1_correct: np.ndarray | None = None,
) -> dict[str, float | int]:
    # Candidate-level F1 must penalize a confident wrong retrieval even when
    # the query identity exists somewhere else in the gallery.  Such a query
    # contributes both a false candidate (FP) and a missed true match (FN).
    if top1_correct is None:
        top1_correct = is_known
    correct_candidate = is_known & top1_correct
    accepted = scores >= threshold
    tp = int(np.sum(accepted & correct_candidate))
    fp = int(np.sum(accepted & ~correct_candidate))
    fn = int(np.sum(is_known & ~(accepted & correct_candidate)))
    tn = int(np.sum(~accepted & ~is_known))
    unknown_fp = int(np.sum(accepted & ~is_known))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    # TNR is defined only over queries whose identity is absent from gallery.
    # Candidate-level FP additionally includes wrong top-1 results for known
    # queries, so it must not be reused in the TNR denominator.
    tnr = tn / (tn + unknown_fp) if tn + unknown_fp else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tnr": float(tnr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unknown_fp": unknown_fp,
        "num_wrong_top1": int(np.sum(is_known & ~top1_correct)),
    }


def _validate_open_set_inputs(
    scores: Any, is_known: Any
) -> tuple[np.ndarray, np.ndarray]:
    score_array = _as_numpy(scores)
    label_array = _as_numpy(is_known)
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if label_array.ndim != 1 or len(label_array) != len(score_array):
        raise ValueError("is_known must be one-dimensional and match scores")
    score_array = score_array.astype(np.float64, copy=False)
    if not np.isfinite(score_array).all():
        raise ValueError("scores contain NaN or infinite values")
    if not np.isin(label_array, [False, True, 0, 1]).all():
        raise ValueError("is_known must contain only boolean/0/1 labels")
    label_array = label_array.astype(bool, copy=False)
    if label_array.all() or (~label_array).all():
        raise ValueError("open-set evaluation requires known and unknown queries")
    return score_array, label_array


def _pr_auc(scores: np.ndarray, is_known: np.ndarray) -> float:
    # Average precision is the standard step-wise area under the PR curve and
    # handles tied scores as threshold groups rather than arbitrary row order.
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(is_known.astype(np.uint8), scores))
    except ImportError:  # Lightweight deterministic fallback for minimal images.
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = is_known[order]
        group_ends = np.flatnonzero(
            np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        )
        cumulative_tp = np.cumsum(sorted_labels)
        precisions = cumulative_tp[group_ends] / (group_ends + 1)
        recalls = cumulative_tp[group_ends] / int(np.sum(is_known))
        recall_increments = np.diff(np.r_[0.0, recalls])
        return float(np.sum(recall_increments * precisions))


def search_rejection_threshold(
    scores: Any,
    is_known: Any,
    *,
    top1_correct: Any | None = None,
) -> dict[str, float | int]:
    """Find the deterministic F1-optimal accept threshold.

    Predictions use ``score >= threshold``.  Ties in F1 are resolved by larger
    TNR and then the larger (more conservative) threshold.
    """

    score_array, label_array = _validate_open_set_inputs(scores, is_known)
    if top1_correct is None:
        correct_array = label_array.copy()
    else:
        correct_array = _as_numpy(top1_correct)
        if correct_array.ndim != 1 or len(correct_array) != len(score_array):
            raise ValueError("top1_correct must be one-dimensional and match scores")
        if not np.isin(correct_array, [False, True, 0, 1]).all():
            raise ValueError("top1_correct must contain only boolean/0/1 labels")
        correct_array = correct_array.astype(bool, copy=False)
        if np.any(correct_array & ~label_array):
            raise ValueError("an unknown query cannot have a correct top-1 candidate")
    unique_scores = np.unique(score_array)
    reject_all = np.nextafter(unique_scores[-1], np.inf)
    thresholds = np.r_[reject_all, unique_scores[::-1]]
    best: dict[str, float | int] | None = None
    best_key: tuple[float, float, float] | None = None
    for threshold in thresholds:
        metrics = _binary_metrics(
            score_array, label_array, float(threshold), correct_array
        )
        key = (
            float(metrics["f1"]),
            float(metrics["tnr"]),
            float(metrics["threshold"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = metrics
    assert best is not None
    best["pr_auc"] = _pr_auc(score_array, correct_array)
    best["num_known"] = int(np.sum(label_array))
    best["num_unknown"] = int(np.sum(~label_array))
    best["num_top1_correct"] = int(np.sum(correct_array))
    return best


def evaluate_open_set(
    scores: Any,
    is_known: Any,
    *,
    threshold: float | None = None,
    top1_correct: Any | None = None,
) -> dict[str, float | int]:
    """Evaluate known-query acceptance and unknown-query rejection."""

    score_array, label_array = _validate_open_set_inputs(scores, is_known)
    if top1_correct is None:
        correct_array = label_array.copy()
    else:
        correct_array = _as_numpy(top1_correct).astype(bool, copy=False)
        if correct_array.ndim != 1 or len(correct_array) != len(score_array):
            raise ValueError("top1_correct must be one-dimensional and match scores")
        if np.any(correct_array & ~label_array):
            raise ValueError("an unknown query cannot have a correct top-1 candidate")
    if threshold is None:
        return search_rejection_threshold(
            score_array, label_array, top1_correct=correct_array
        )
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    result = _binary_metrics(
        score_array, label_array, float(threshold), correct_array
    )
    result["pr_auc"] = _pr_auc(score_array, correct_array)
    result["num_known"] = int(np.sum(label_array))
    result["num_unknown"] = int(np.sum(~label_array))
    result["num_top1_correct"] = int(np.sum(correct_array))
    return result


def calibrate_open_set(
    query_embeddings: Any,
    gallery_embeddings: Any,
    query_pids: Any,
    gallery_pids: Any,
    *,
    known_pids: Iterable[Any] | None = None,
    unknown_pids: Iterable[Any] | None = None,
    query_camera_ids: Any | None = None,
    gallery_camera_ids: Any | None = None,
    query_image_ids: Any | None = None,
    gallery_image_ids: Any | None = None,
    same_source: bool | None = None,
) -> dict[str, Any]:
    """Calibrate rejection from valid top-1 scores and known/unknown IDs."""

    protocol = _prepare_protocol(
        query_embeddings,
        gallery_embeddings,
        query_pids,
        gallery_pids,
        query_camera_ids,
        gallery_camera_ids,
        query_image_ids,
        gallery_image_ids,
        same_source,
    )
    query_pid_array = protocol[1]
    gallery_pid_array = protocol[2]
    known_mask = derive_known_mask(
        query_pid_array,
        gallery_pid_array,
        known_pids=known_pids,
        unknown_pids=unknown_pids,
    )
    scores, has_positive, top1_correct = _top1_and_match_availability(protocol)
    invalid_known = np.flatnonzero(known_mask & ~has_positive)
    if len(invalid_known):
        raise ValueError(
            "known query has no valid cross-camera gallery match; query indices "
            f"include {invalid_known[:5].tolist()}"
        )
    invalid_unknown = np.flatnonzero(~known_mask & has_positive)
    if len(invalid_unknown):
        raise ValueError(
            "unknown query has a valid gallery match; query indices include "
            f"{invalid_unknown[:5].tolist()}"
        )
    result: dict[str, Any] = search_rejection_threshold(
        scores, known_mask, top1_correct=top1_correct
    )
    result["scores"] = scores
    result["is_known"] = known_mask
    result["has_gallery_match"] = has_positive
    result["top1_correct"] = top1_correct
    return result


def split_known_unknown_ids(
    vehicle_ids: Sequence[Any],
    *,
    unknown_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Any], list[Any]]:
    """Deterministically reserve whole identities as artificial unknowns."""

    identities = list(dict.fromkeys(vehicle_ids))
    if len(identities) < 2:
        raise ValueError("at least two identities are required")
    if not 0.0 < float(unknown_fraction) < 1.0:
        raise ValueError("unknown_fraction must be strictly between 0 and 1")
    count = min(len(identities) - 1, max(1, round(len(identities) * unknown_fraction)))
    rng = np.random.default_rng(int(seed))
    positions = set(
        int(value)
        for value in rng.choice(len(identities), size=count, replace=False)
    )
    known = [pid for index, pid in enumerate(identities) if index not in positions]
    unknown = [pid for index, pid in enumerate(identities) if index in positions]
    return known, unknown


__all__ = [
    "calibrate_open_set",
    "compute_retrieval_metrics",
    "cosine_similarity_matrix",
    "derive_known_mask",
    "evaluate_open_set",
    "evaluate_retrieval",
    "open_set_top1_scores",
    "search_rejection_threshold",
    "split_known_unknown_ids",
]
