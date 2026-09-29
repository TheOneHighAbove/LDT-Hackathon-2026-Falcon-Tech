"""Candidate-correctness features and group-aware logistic calibration.

The existing scalar refusal score answers only "how similar is the best
candidate?".  This module adds rank-margin, query-normalized and gallery
neighbourhood evidence while keeping deployment inference NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CANDIDATE_FEATURE_NAMES: tuple[str, ...] = (
    "top1_similarity",
    "top1_top2_margin",
    "top1_to_top5_gap",
    "query_top1_zscore",
    "top_candidate_gallery_density",
    "top_candidate_gallery_isolation",
    "top_candidate_mutual_nn",
)

# The production adaptive-refusal feature family is deliberately frozen.  The
# robust calibration experiment evaluates several candidate families, but the
# deployed model must never silently change column order or feature semantics.
ADAPTIVE_REFUSAL_FEATURE_VARIANT = "top1_margins_local_density"
ADAPTIVE_REFUSAL_FEATURE_NAMES: tuple[str, ...] = (
    "top1_similarity",
    "top1_top2_margin",
    "top1_to_top5_gap",
    "top_candidate_gallery_density",
    "top_candidate_gallery_isolation",
)
FINAL_RANKER_REFUSAL_FEATURE_NAMES: tuple[str, ...] = (
    "final_top1_score",
    "final_top1_top2_margin",
    "final_top1_to_top5_gap",
    "selected_candidate_cosine",
    "selected_candidate_cosine_margin",
    "selected_candidate_cosine_top5_gap",
    "top_candidate_gallery_density",
    "top_candidate_gallery_isolation",
)
ADAPTIVE_REFUSAL_DENSITY_K = 5

_MODEL_TYPE = "logistic_candidate_correctness"
_MODEL_VERSION = 1
_COSINE_TOLERANCE = 1e-6


def _finite_matrix(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if (
        array.min() < -1.0 - _COSINE_TOLERANCE
        or array.max() > 1.0 + _COSINE_TOLERANCE
    ):
        raise ValueError(f"{name} must contain cosine similarities in [-1, 1]")
    return np.clip(array, -1.0, 1.0)


def _gallery_neighbourhood_statistics(
    gallery_similarity: np.ndarray,
    *,
    density_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    gallery_size = gallery_similarity.shape[0]
    if gallery_similarity.shape[1] != gallery_size:
        raise ValueError("gallery_gallery_similarity must be square")
    if gallery_size < 2:
        raise ValueError("at least two gallery candidates are required")
    if not np.allclose(
        gallery_similarity,
        gallery_similarity.T,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("gallery_gallery_similarity must be symmetric")
    if isinstance(density_k, bool) or not isinstance(density_k, (int, np.integer)):
        raise ValueError("density_k must be a positive integer")
    if int(density_k) < 1:
        raise ValueError("density_k must be a positive integer")

    # Never let self-similarity dominate neighbourhood statistics.  The
    # diagonal need not be exactly one, which also supports pre-masked input.
    neighbours = gallery_similarity.copy()
    np.fill_diagonal(neighbours, -np.inf)
    k = min(int(density_k), gallery_size - 1)
    nearest = np.partition(neighbours, gallery_size - k, axis=1)[:, -k:]
    density = nearest.mean(axis=1)
    nearest_one = nearest.max(axis=1)
    isolation = 1.0 - nearest_one
    return density, isolation


def extract_candidate_features(
    query_gallery_similarity: Any,
    gallery_gallery_similarity: Any,
    *,
    density_k: int = 5,
) -> np.ndarray:
    """Extract correctness features for each query's top-1 candidate.

    ``query_gallery_similarity`` has shape ``[num_queries, num_gallery]`` and
    ``gallery_gallery_similarity`` has shape ``[num_gallery, num_gallery]``.
    The returned columns follow :data:`CANDIDATE_FEATURE_NAMES`:

    1. top-1 cosine ``s1``;
    2. ``s1 - s2``;
    3. ``s1 - mean(s2..s5)`` (all available ranks when gallery has <5 rows);
    4. per-query z-score of ``s1`` against the complete gallery;
    5. mean similarity of the top candidate to its ``density_k`` neighbours;
    6. its isolation, ``1 - max(non-self gallery cosine)``;
    7. a deployment-safe mutual-nearest-neighbour indicator: one when the
       query is at least as similar to its top candidate as that candidate's
       closest *other* gallery item.

    Stable ``argmax`` semantics select the first gallery row when scores tie.
    """

    scores = _finite_matrix(
        query_gallery_similarity, "query_gallery_similarity"
    )
    gallery = _finite_matrix(
        gallery_gallery_similarity, "gallery_gallery_similarity"
    )
    if scores.shape[1] != gallery.shape[0]:
        raise ValueError(
            "query/gallery width must match gallery_gallery_similarity size"
        )
    if scores.shape[1] < 2:
        raise ValueError("at least two gallery candidates are required")

    density, isolation = _gallery_neighbourhood_statistics(
        gallery, density_k=density_k
    )
    top_count = min(5, scores.shape[1])
    top_values = np.partition(
        scores, scores.shape[1] - top_count, axis=1
    )[:, -top_count:]
    top_values.sort(axis=1)
    top_values = top_values[:, ::-1]

    top1 = top_values[:, 0]
    margin = top1 - top_values[:, 1]
    top5_gap = top1 - top_values[:, 1:].mean(axis=1)
    query_mean = scores.mean(axis=1)
    query_std = scores.std(axis=1)
    safe_std = np.where(query_std > np.finfo(np.float64).eps, query_std, 1.0)
    zscore = (top1 - query_mean) / safe_std
    # A constant query row has no relative evidence despite the safe divisor.
    zscore = np.where(query_std > np.finfo(np.float64).eps, zscore, 0.0)

    top_indices = np.argmax(scores, axis=1)
    candidate_nearest_gallery = 1.0 - isolation[top_indices]
    mutual_nn = (top1 >= candidate_nearest_gallery).astype(np.float64)
    features = np.column_stack(
        (
            top1,
            margin,
            top5_gap,
            zscore,
            density[top_indices],
            isolation[top_indices],
            mutual_nn,
        )
    )
    return features.astype(np.float64, copy=False)


def extract_adaptive_refusal_features(
    query_gallery_similarity: Any,
    gallery_gallery_similarity: Any,
    *,
    density_k: int = ADAPTIVE_REFUSAL_DENSITY_K,
) -> np.ndarray:
    """Return the frozen production feature family in its canonical order.

    This is a thin selection over :func:`extract_candidate_features`, the same
    extractor used by ``src.robust_open_set``.  Keeping a single numerical
    implementation prevents offline calibration and deployment from drifting.
    ``density_k`` remains an argument for explicit parity tests; production
    checkpoint validation requires the frozen value ``5``.
    """

    all_features = extract_candidate_features(
        query_gallery_similarity,
        gallery_gallery_similarity,
        density_k=density_k,
    )
    positions = {name: index for index, name in enumerate(CANDIDATE_FEATURE_NAMES)}
    columns = [positions[name] for name in ADAPTIVE_REFUSAL_FEATURE_NAMES]
    return all_features[:, columns]


def extract_selected_candidate_refusal_features(
    query_gallery_similarity: Any,
    gallery_gallery_similarity: Any,
    selected_indices: Any,
    *,
    density_k: int = ADAPTIVE_REFUSAL_DENSITY_K,
) -> np.ndarray:
    """Describe the candidate actually emitted by the final ranker.

    The original adaptive extractor implicitly described the cosine top-1.
    A downstream reranker can select a different gallery row, so deployment
    must bind confidence and correctness calibration to that final candidate.
    When ``selected_indices`` is the cosine top-1, this function is exactly
    equivalent to :func:`extract_adaptive_refusal_features`.
    """

    scores = _finite_matrix(
        query_gallery_similarity, "query_gallery_similarity"
    )
    gallery = _finite_matrix(
        gallery_gallery_similarity, "gallery_gallery_similarity"
    )
    if scores.shape[1] != gallery.shape[0]:
        raise ValueError(
            "query/gallery width must match gallery_gallery_similarity size"
        )
    if scores.shape[1] < 5:
        raise ValueError("at least five gallery candidates are required")
    selected = np.asarray(selected_indices)
    if selected.ndim != 1 or len(selected) != len(scores):
        raise ValueError("selected_indices must align with query rows")
    if not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("selected_indices must contain integers")
    selected = selected.astype(np.int64, copy=False)
    if np.any(selected < 0) or np.any(selected >= scores.shape[1]):
        raise ValueError("selected_indices contains an out-of-range index")

    rows = np.arange(len(scores))
    selected_score = scores[rows, selected]
    alternatives = scores.copy()
    alternatives[rows, selected] = -np.inf
    strongest = np.partition(
        alternatives, alternatives.shape[1] - 4, axis=1
    )[:, -4:]
    strongest.sort(axis=1)
    strongest = strongest[:, ::-1]
    density, isolation = _gallery_neighbourhood_statistics(
        gallery, density_k=density_k
    )
    return np.column_stack(
        (
            selected_score,
            selected_score - strongest[:, 0],
            selected_score - strongest.mean(axis=1),
            density[selected],
            isolation[selected],
        )
    ).astype(np.float64, copy=False)


def extract_final_ranker_refusal_features(
    query_gallery_similarity: Any,
    gallery_gallery_similarity: Any,
    final_scores: Any,
    final_top1: Any,
    *,
    density_k: int = ADAPTIVE_REFUSAL_DENSITY_K,
) -> np.ndarray:
    """Bind correctness evidence to the candidate selected by final ranking."""

    selected = extract_selected_candidate_refusal_features(
        query_gallery_similarity,
        gallery_gallery_similarity,
        final_top1,
        density_k=density_k,
    )
    scores = np.asarray(final_scores, dtype=np.float64)
    if scores.shape != np.asarray(query_gallery_similarity).shape:
        raise ValueError(
            "final_scores must match query_gallery_similarity shape"
        )
    if not np.isfinite(scores).all():
        raise ValueError("final_scores contains NaN or infinite values")
    top_indices = np.asarray(final_top1, dtype=np.int64)
    if not np.array_equal(np.argmax(scores, axis=1), top_indices):
        raise ValueError("final_top1 does not match final_scores")
    top_values = np.partition(scores, scores.shape[1] - 5, axis=1)[
        :, -5:
    ]
    top_values.sort(axis=1)
    top_values = top_values[:, ::-1]
    final = np.column_stack(
        (
            top_values[:, 0],
            top_values[:, 0] - top_values[:, 1],
            top_values[:, 0] - top_values[:, 1:].mean(axis=1),
        )
    )
    return np.column_stack(
        (final, selected[:, :3], selected[:, 3:])
    ).astype(np.float64, copy=False)


def _feature_matrix(value: Any, expected_features: int | None = None) -> np.ndarray:
    try:
        features = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be numeric") from exc
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must be a non-empty [N, F] matrix")
    if expected_features is not None and features.shape[1] != expected_features:
        raise ValueError(
            f"features must have {expected_features} columns, got {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values")
    return features


def _binary_labels(value: Any, expected_length: int) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 1 or len(labels) != expected_length:
        raise ValueError("labels must be one-dimensional and match features")
    if not np.isin(labels, [False, True, 0, 1]).all():
        raise ValueError("labels must contain only boolean/0/1 values")
    labels = labels.astype(np.uint8, copy=False)
    if np.unique(labels).size != 2:
        raise ValueError("logistic calibration requires both target classes")
    return labels


def _feature_names(names: Sequence[str] | None, count: int) -> tuple[str, ...]:
    if names is None:
        if count == len(CANDIDATE_FEATURE_NAMES):
            return CANDIDATE_FEATURE_NAMES
        return tuple(f"feature_{index}" for index in range(count))
    result = tuple(str(name) for name in names)
    if len(result) != count:
        raise ValueError("feature_names length must match the feature count")
    if any(not name for name in result) or len(set(result)) != len(result):
        raise ValueError("feature_names must be non-empty and unique")
    return result


@dataclass(frozen=True)
class CandidateCorrectnessModel:
    """Serializable standardized logistic model with NumPy-only inference."""

    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.coefficients, dtype=np.float64)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        names = tuple(self.feature_names)
        if coefficient.ndim != 1 or coefficient.size == 0:
            raise ValueError("coefficients must be a non-empty vector")
        if mean.shape != coefficient.shape or scale.shape != coefficient.shape:
            raise ValueError(
                "coefficients, feature_mean and feature_scale must have equal shapes"
            )
        if len(names) != len(coefficient):
            raise ValueError("feature_names length must match coefficients")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        if not (
            np.isfinite(coefficient).all()
            and np.isfinite(mean).all()
            and np.isfinite(scale).all()
            and np.isfinite(float(self.intercept))
        ):
            raise ValueError("model parameters must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("feature_scale must be strictly positive")

        coefficient = coefficient.copy()
        mean = mean.copy()
        scale = scale.copy()
        coefficient.setflags(write=False)
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficient)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "intercept", float(self.intercept))

    def decision_function(self, features: Any) -> np.ndarray:
        values = _feature_matrix(features, len(self.coefficients))
        standardized = (values - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.intercept

    def predict_proba(self, features: Any) -> np.ndarray:
        """Return sklearn-compatible ``[P(incorrect), P(correct)]`` columns."""

        logits = self.decision_function(features)
        positive = np.empty_like(logits)
        mask = logits >= 0.0
        positive[mask] = 1.0 / (1.0 + np.exp(-logits[mask]))
        exp_logits = np.exp(logits[~mask])
        positive[~mask] = exp_logits / (1.0 + exp_logits)
        return np.column_stack((1.0 - positive, positive))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": _MODEL_TYPE,
            "version": _MODEL_VERSION,
            "feature_names": list(self.feature_names),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateCorrectnessModel":
        if not isinstance(payload, Mapping):
            raise ValueError("model payload must be a mapping")
        if payload.get("model_type") != _MODEL_TYPE:
            raise ValueError("unsupported candidate correctness model type")
        if payload.get("version") != _MODEL_VERSION:
            raise ValueError("unsupported candidate correctness model version")
        required = {
            "feature_names",
            "coefficients",
            "intercept",
            "feature_mean",
            "feature_scale",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"model payload is missing fields: {missing}")
        try:
            return cls(
                coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
                intercept=float(payload["intercept"]),
                feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
                feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
                feature_names=tuple(payload["feature_names"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid candidate correctness model: {exc}") from exc

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    @classmethod
    def load_json(cls, path: str | Path) -> "CandidateCorrectnessModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def fit_candidate_correctness_model(
    features: Any,
    labels: Any,
    *,
    feature_names: Sequence[str] | None = None,
    regularization_c: float = 1.0,
    class_weight: str | Mapping[int, float] | None = "balanced",
    random_state: int = 42,
    max_iter: int = 2000,
) -> CandidateCorrectnessModel:
    """Fit logistic correctness calibration and detach it from sklearn."""

    values = _feature_matrix(features)
    targets = _binary_labels(labels, len(values))
    names = _feature_names(feature_names, values.shape[1])
    if not np.isfinite(float(regularization_c)) or float(regularization_c) <= 0.0:
        raise ValueError("regularization_c must be finite and positive")
    if isinstance(max_iter, bool) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")

    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    standardized = (values - mean) / scale
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # Training dependency; deployment stays NumPy-only.
        raise ImportError("scikit-learn is required to fit score calibration") from exc

    estimator = LogisticRegression(
        C=float(regularization_c),
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    estimator.fit(standardized, targets)
    return CandidateCorrectnessModel(
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(estimator.intercept_[0]),
        feature_mean=mean,
        feature_scale=scale,
        feature_names=names,
    )


def _group_codes(groups: Any, expected_length: int) -> tuple[np.ndarray, int]:
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError("groups must be one-dimensional and match features")
    codes = np.empty(expected_length, dtype=np.int64)
    mapping: dict[Any, int] = {}
    for index, value in enumerate(values.tolist()):
        if value is None:
            raise ValueError("groups contain a missing value")
        try:
            if bool(value != value):
                raise ValueError("groups contain a missing value")
            code = mapping.setdefault(value, len(mapping))
        except TypeError as exc:
            raise ValueError("group identifiers must be hashable scalars") from exc
        codes[index] = code
    return codes, len(mapping)


def _optimal_binary_f1(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, float | int]:
    unique = np.unique(probabilities)
    thresholds = np.r_[np.nextafter(unique[-1], np.inf), unique[::-1]]
    best: dict[str, float | int] | None = None
    best_key: tuple[float, float, float] | None = None
    positives = labels.astype(bool)
    for threshold in thresholds:
        accepted = probabilities >= threshold
        tp = int(np.sum(accepted & positives))
        fp = int(np.sum(accepted & ~positives))
        fn = int(np.sum(~accepted & positives))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        key = (f1, precision, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
    assert best is not None
    return best


def evaluate_group_oof(
    features: Any,
    labels: Any,
    groups: Any,
    *,
    n_splits: int = 5,
    feature_names: Sequence[str] | None = None,
    regularization_c: float = 1.0,
    class_weight: str | Mapping[int, float] | None = "balanced",
    random_state: int = 42,
) -> dict[str, Any]:
    """Generate identity-grouped OOF probabilities and aggregate diagnostics.

    Every group is assigned wholly to one validation fold.  Standardization
    and logistic fitting happen inside each training fold, preventing feature
    statistics from leaking from its validation groups.
    """

    values = _feature_matrix(features)
    targets = _binary_labels(labels, len(values))
    names = _feature_names(feature_names, values.shape[1])
    group_codes, group_count = _group_codes(groups, len(values))
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise ValueError("n_splits must be an integer")
    n_splits = int(n_splits)
    if n_splits < 2 or n_splits > group_count:
        raise ValueError("n_splits must be between 2 and the number of groups")

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise ImportError("scikit-learn is required for group-aware OOF") from exc

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )
    probabilities = np.full(len(values), np.nan, dtype=np.float64)
    fold_assignments = np.full(len(values), -1, dtype=np.int64)
    folds: list[dict[str, int]] = []
    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(values, targets, group_codes)
    ):
        if np.unique(targets[train_indices]).size != 2:
            raise ValueError(
                f"OOF fold {fold} training partition does not contain both classes"
            )
        model = fit_candidate_correctness_model(
            values[train_indices],
            targets[train_indices],
            feature_names=names,
            regularization_c=regularization_c,
            class_weight=class_weight,
            random_state=random_state + fold,
        )
        probabilities[valid_indices] = model.predict_proba(values[valid_indices])[:, 1]
        fold_assignments[valid_indices] = fold
        folds.append(
            {
                "fold": fold,
                "num_train": int(len(train_indices)),
                "num_validation": int(len(valid_indices)),
                "num_train_groups": int(np.unique(group_codes[train_indices]).size),
                "num_validation_groups": int(
                    np.unique(group_codes[valid_indices]).size
                ),
                "num_validation_positive": int(targets[valid_indices].sum()),
            }
        )

    if not np.isfinite(probabilities).all() or np.any(fold_assignments < 0):
        raise RuntimeError("group-aware OOF did not predict every input row exactly once")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    log_loss = -np.mean(
        targets * np.log(clipped) + (1 - targets) * np.log(1.0 - clipped)
    )
    metrics: dict[str, float | int] = {
        "average_precision": float(average_precision_score(targets, probabilities)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "log_loss": float(log_loss),
        "num_samples": int(len(values)),
        "num_groups": int(group_count),
        "num_positive": int(targets.sum()),
        "num_negative": int(len(targets) - targets.sum()),
    }
    metrics.update(_optimal_binary_f1(probabilities, targets))
    return {
        "probabilities": probabilities,
        "fold_assignments": fold_assignments,
        "metrics": metrics,
        "folds": folds,
        "feature_names": names,
    }


__all__ = [
    "ADAPTIVE_REFUSAL_DENSITY_K",
    "ADAPTIVE_REFUSAL_FEATURE_NAMES",
    "ADAPTIVE_REFUSAL_FEATURE_VARIANT",
    "CANDIDATE_FEATURE_NAMES",
    "FINAL_RANKER_REFUSAL_FEATURE_NAMES",
    "CandidateCorrectnessModel",
    "evaluate_group_oof",
    "extract_adaptive_refusal_features",
    "extract_final_ranker_refusal_features",
    "extract_selected_candidate_refusal_features",
    "extract_candidate_features",
    "fit_candidate_correctness_model",
]
