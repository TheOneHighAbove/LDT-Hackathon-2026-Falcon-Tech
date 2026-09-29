"""Deterministic NumPy post-processing for image-retrieval embeddings.

The module implements three common, training-free ReID techniques:

* average query expansion (AQE);
* database-side augmentation (DBA);
* k-reciprocal re-ranking from Zhong et al., CVPR 2017.

Every public function accepts arbitrary finite, non-zero embeddings and performs
safe L2 normalization internally.  Stable sorting is used deliberately: when
two candidates have exactly the same score, their original row order is the
tie-breaker.  No SciPy, scikit-learn or FAISS dependency is required.

Reference:
    Zhun Zhong, Liang Zheng, Donglin Cao, Shaozi Li,
    "Re-ranking Person Re-identification with k-reciprocal Encoding",
    CVPR 2017, https://arxiv.org/abs/1701.08398
"""

from __future__ import annotations

from typing import Literal

import numpy as np


OutputKind = Literal["distance", "similarity"]


def l2_normalize_embeddings(
    embeddings: object,
    *,
    name: str = "embeddings",
) -> np.ndarray:
    """Validate and row-wise L2-normalize an embedding matrix.

    Args:
        embeddings: A non-empty numeric matrix with shape ``[N, D]``.
        name: Name included in validation errors.

    Returns:
        A finite ``float64`` matrix whose rows have unit L2 norm.  The input is
        never modified.

    Raises:
        ValueError: If the input is not a non-empty 2-D numeric matrix, contains
            non-finite values, or has a zero-norm row.
    """

    array = np.asarray(embeddings)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, D] matrix")
    try:
        array = array.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms <= np.finfo(np.float64).eps).any():
        raise ValueError(f"{name} contains a zero-norm embedding")
    normalized = array / norms
    if not np.isfinite(normalized).all():  # Defensive guard for extreme values.
        raise ValueError(f"{name} cannot be safely L2-normalized")
    return normalized


def _validate_pair(
    query_embeddings: object,
    gallery_embeddings: object,
) -> tuple[np.ndarray, np.ndarray]:
    query = l2_normalize_embeddings(query_embeddings, name="query_embeddings")
    gallery = l2_normalize_embeddings(
        gallery_embeddings, name="gallery_embeddings"
    )
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions do not match")
    return query, gallery


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _non_negative_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _neighbor_weights(similarities: np.ndarray, alpha: float) -> np.ndarray:
    """Return non-negative AQE weights without fractional powers of negatives."""

    if alpha == 0.0:
        return np.ones_like(similarities, dtype=np.float64)
    return np.power(np.clip(similarities, 0.0, 1.0), alpha)


def _normalize_with_fallback(
    augmented: np.ndarray,
    original: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Normalize augmented rows, retaining originals after exact cancellation."""

    norms = np.linalg.norm(augmented, axis=1)
    degenerate = norms <= np.finfo(np.float64).eps
    if degenerate.any():
        augmented = augmented.copy()
        augmented[degenerate] = original[degenerate]
    return l2_normalize_embeddings(augmented, name=name)


def query_expansion(
    query_embeddings: object,
    gallery_embeddings: object,
    *,
    top_k: int = 3,
    alpha: float = 1.0,
    query_weight: float = 1.0,
) -> np.ndarray:
    """Expand every query with its nearest gallery embeddings.

    The output for query ``q`` is the L2-normalized weighted sum

    ``query_weight * q + sum(max(cos(q, g_i), 0) ** alpha * g_i)``.

    ``alpha=0`` gives uniform neighbour weights.  Negative-similarity
    neighbours receive zero weight for positive ``alpha``.  Exact score ties
    preserve gallery row order.

    Args:
        query_embeddings: Query matrix ``[num_queries, embedding_dim]``.
        gallery_embeddings: Gallery matrix ``[num_gallery, embedding_dim]``.
        top_k: Number of nearest gallery rows used for each query.
        alpha: Non-negative similarity-weight exponent.
        query_weight: Positive weight of the original query.

    Returns:
        L2-normalized expanded queries with the same shape as the query input.
    """

    query, gallery = _validate_pair(query_embeddings, gallery_embeddings)
    top_k = _positive_int(top_k, "top_k")
    if top_k > len(gallery):
        raise ValueError("top_k cannot exceed the number of gallery embeddings")
    alpha = _non_negative_float(alpha, "alpha")
    query_weight = _non_negative_float(query_weight, "query_weight")
    if query_weight == 0.0:
        raise ValueError("query_weight must be greater than zero")

    similarities = np.clip(query @ gallery.T, -1.0, 1.0)
    order = np.argsort(-similarities, axis=1, kind="mergesort")[:, :top_k]
    selected_scores = np.take_along_axis(similarities, order, axis=1)
    weights = _neighbor_weights(selected_scores, alpha)
    neighbors = gallery[order]
    expanded = query_weight * query + np.einsum(
        "qk,qkd->qd", weights, neighbors, optimize=True
    )
    return _normalize_with_fallback(
        expanded, query, name="expanded_queries"
    )


def database_side_augmentation(
    gallery_embeddings: object,
    *,
    top_k: int = 3,
    alpha: float = 1.0,
    self_weight: float = 1.0,
) -> np.ndarray:
    """Augment each gallery row with its nearest *other* gallery rows.

    The operation mirrors :func:`query_expansion`, but excludes the row itself
    from its neighbour list and keeps it separately with ``self_weight``.
    Exact ties preserve original gallery order.

    Args:
        gallery_embeddings: Gallery matrix ``[num_gallery, embedding_dim]``.
        top_k: Number of neighbours, excluding the row itself.
        alpha: Non-negative similarity-weight exponent; zero means uniform.
        self_weight: Positive weight of the original gallery embedding.

    Returns:
        L2-normalized augmented gallery embeddings of the same shape.
    """

    gallery = l2_normalize_embeddings(
        gallery_embeddings, name="gallery_embeddings"
    )
    top_k = _positive_int(top_k, "top_k")
    if top_k >= len(gallery):
        raise ValueError(
            "top_k must be smaller than the number of gallery embeddings"
        )
    alpha = _non_negative_float(alpha, "alpha")
    self_weight = _non_negative_float(self_weight, "self_weight")
    if self_weight == 0.0:
        raise ValueError("self_weight must be greater than zero")

    similarities = np.clip(gallery @ gallery.T, -1.0, 1.0)
    # Infinity excludes self regardless of otherwise tied similarities.
    ranking_scores = similarities.copy()
    np.fill_diagonal(ranking_scores, -np.inf)
    order = np.argsort(-ranking_scores, axis=1, kind="mergesort")[:, :top_k]
    selected_scores = np.take_along_axis(similarities, order, axis=1)
    weights = _neighbor_weights(selected_scores, alpha)
    neighbors = gallery[order]
    augmented = self_weight * gallery + np.einsum(
        "gk,gkd->gd", weights, neighbors, optimize=True
    )
    return _normalize_with_fallback(
        augmented, gallery, name="augmented_gallery"
    )


def _squared_euclidean_matrix(normalized: np.ndarray) -> np.ndarray:
    # For unit vectors ||a-b||^2 = 2 - 2*cos(a,b).  The clipped formulation is
    # faster and more numerically stable than materialising pairwise deltas.
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return np.clip(2.0 - 2.0 * cosine, 0.0, 4.0)


def _k_reciprocal_distance(
    query: np.ndarray,
    gallery: np.ndarray,
    *,
    k1: int,
    k2: int,
    lambda_value: float,
) -> np.ndarray:
    features = np.concatenate((query, gallery), axis=0)
    query_count = len(query)
    all_count = len(features)

    original_distance = _squared_euclidean_matrix(features)
    # This is the row-wise equivalent of the transpose-after-column-scaling in
    # the authors' reference implementation.  Fully identical sets have a zero
    # maximum; keeping their all-zero distance rows is the natural limit.
    row_maximum = original_distance.max(axis=1, keepdims=True)
    original_distance = np.divide(
        original_distance,
        row_maximum,
        out=np.zeros_like(original_distance),
        where=row_maximum > np.finfo(np.float64).eps,
    )
    initial_rank = np.argsort(
        original_distance, axis=1, kind="mergesort"
    )

    affinity = np.zeros((all_count, all_count), dtype=np.float64)
    half_k = int(np.around(k1 / 2.0))
    for row_index in range(all_count):
        forward = initial_rank[row_index, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal_positions = np.flatnonzero(
            np.any(backward == row_index, axis=1)
        )
        reciprocal = forward[reciprocal_positions]
        expanded = reciprocal.tolist()

        for candidate in reciprocal.tolist():
            candidate_forward = initial_rank[candidate, : half_k + 1]
            candidate_backward = initial_rank[candidate_forward, : half_k + 1]
            candidate_positions = np.flatnonzero(
                np.any(candidate_backward == candidate, axis=1)
            )
            candidate_reciprocal = candidate_forward[candidate_positions]
            overlap = np.intersect1d(
                candidate_reciprocal, reciprocal, assume_unique=False
            ).size
            if overlap > (2.0 / 3.0) * len(candidate_reciprocal):
                expanded.extend(candidate_reciprocal.tolist())

        # np.unique also gives a canonical deterministic column order.
        expanded_indices = np.unique(np.asarray(expanded, dtype=np.int64))
        weights = np.exp(-original_distance[row_index, expanded_indices])
        weight_sum = weights.sum()
        if weight_sum <= np.finfo(np.float64).eps:  # exp(-d) should prevent it.
            affinity[row_index, row_index] = 1.0
        else:
            affinity[row_index, expanded_indices] = weights / weight_sum

    if k2 > 1:
        expanded_affinity = np.empty_like(affinity)
        for row_index in range(all_count):
            expanded_affinity[row_index] = affinity[
                initial_rank[row_index, :k2]
            ].mean(axis=0)
        affinity = expanded_affinity

    inverted_index = [
        np.flatnonzero(affinity[:, column_index] > 0.0)
        for column_index in range(all_count)
    ]
    jaccard_distance = np.ones((query_count, all_count), dtype=np.float64)
    for query_index in range(query_count):
        nonzero_columns = np.flatnonzero(affinity[query_index] > 0.0)
        if not len(nonzero_columns):
            continue
        related_rows = np.unique(
            np.concatenate([inverted_index[column] for column in nonzero_columns])
        )
        intersections = np.zeros(len(related_rows), dtype=np.float64)
        for column in nonzero_columns.tolist():
            present = affinity[related_rows, column] > 0.0
            if present.any():
                intersections[present] += np.minimum(
                    affinity[query_index, column],
                    affinity[related_rows[present], column],
                )
        unions = 2.0 - intersections
        jaccard_distance[query_index, related_rows] = 1.0 - np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions > np.finfo(np.float64).eps,
        )

    combined = (
        (1.0 - lambda_value) * jaccard_distance
        + lambda_value * original_distance[:query_count]
    )
    return np.clip(combined[:, query_count:], 0.0, 1.0)


def k_reciprocal_rerank(
    query_embeddings: object,
    gallery_embeddings: object,
    *,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    output: OutputKind = "distance",
) -> np.ndarray:
    """Compute k-reciprocal re-ranked query-to-gallery values.

    This follows Zhong et al.'s k-reciprocal encoding, local query expansion,
    Jaccard distance and original-distance interpolation.  The complete
    query+gallery set participates in neighbourhood construction.

    Args:
        query_embeddings: Query matrix ``[num_queries, embedding_dim]``.
        gallery_embeddings: Gallery matrix ``[num_gallery, embedding_dim]``.
        k1: Reciprocal-neighbour size. Must be smaller than the total number of
            query and gallery rows.
        k2: Local-query-expansion size in ``[1, k1]``.  Set to one to disable
            the expansion step.
        lambda_value: Weight of normalized original distance in ``[0, 1]``;
            the Jaccard weight is ``1 - lambda_value``.
        output: Return ``"distance"`` (smaller is better) or ``"similarity"``
            (larger is better).  Similarity is exactly ``1 - distance``.

    Returns:
        A finite matrix ``[num_queries, num_gallery]`` in the closed interval
        ``[0, 1]``.

    Notes:
        The algorithm materializes square ``(Q + G) x (Q + G)`` matrices and
        is therefore intended for offline evaluation or moderate galleries.
        Use an ANN index for very large production galleries.
    """

    query, gallery = _validate_pair(query_embeddings, gallery_embeddings)
    k1 = _positive_int(k1, "k1")
    k2 = _positive_int(k2, "k2")
    all_count = len(query) + len(gallery)
    if k1 >= all_count:
        raise ValueError(
            "k1 must be smaller than the total number of query and gallery embeddings"
        )
    if k2 > k1:
        raise ValueError("k2 must be less than or equal to k1")
    try:
        lambda_value = float(lambda_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("lambda_value must be a finite number in [0, 1]") from exc
    if not np.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_value must be a finite number in [0, 1]")
    if output not in ("distance", "similarity"):
        raise ValueError("output must be 'distance' or 'similarity'")

    distances = _k_reciprocal_distance(
        query,
        gallery,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    if output == "similarity":
        return 1.0 - distances
    return distances


__all__ = [
    "database_side_augmentation",
    "k_reciprocal_rerank",
    "l2_normalize_embeddings",
    "query_expansion",
]
