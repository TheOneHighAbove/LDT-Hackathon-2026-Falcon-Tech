from __future__ import annotations

import numpy as np
import pytest

from src.reranking import (
    database_side_augmentation,
    k_reciprocal_rerank,
    l2_normalize_embeddings,
    query_expansion,
)


def test_l2_normalize_validates_input_and_does_not_mutate() -> None:
    original = np.asarray([[3.0, 4.0], [0.0, -2.0]])
    before = original.copy()
    normalized = l2_normalize_embeddings(original)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
    np.testing.assert_array_equal(original, before)

    for invalid in (
        np.asarray([]),
        np.asarray([1.0, 2.0]),
        np.asarray([[0.0, 0.0]]),
        np.asarray([[np.nan, 1.0]]),
        np.asarray([[np.inf, 1.0]]),
    ):
        with pytest.raises(ValueError):
            l2_normalize_embeddings(invalid)


def test_query_expansion_uses_nearest_rows_and_returns_unit_vectors() -> None:
    queries = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    gallery = np.asarray([[1.0, 1.0], [-1.0, 0.0], [0.0, 1.0]])
    expanded = query_expansion(queries, gallery, top_k=1, alpha=0.0)

    expected = np.asarray(
        [
            [1.0, 0.0] + gallery[0] / np.linalg.norm(gallery[0]),
            [0.0, 1.0] + gallery[2] / np.linalg.norm(gallery[2]),
        ]
    )
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(expanded, expected)
    np.testing.assert_allclose(np.linalg.norm(expanded, axis=1), 1.0)


def test_query_expansion_ties_are_resolved_by_gallery_order() -> None:
    query = np.asarray([[1.0, 0.0]])
    # Both rows have cosine 0.8. Stable sorting must select row zero.
    gallery = np.asarray([[0.8, 0.6], [0.8, -0.6]])
    first = query_expansion(query, gallery, top_k=1, alpha=0.0)
    second = query_expansion(query, gallery, top_k=1, alpha=0.0)
    np.testing.assert_array_equal(first, second)
    assert first[0, 1] > 0.0


def test_augmentation_falls_back_after_exact_vector_cancellation() -> None:
    opposite = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    expanded = query_expansion(
        opposite[:1], opposite[1:], top_k=1, alpha=0.0
    )
    augmented = database_side_augmentation(opposite, top_k=1, alpha=0.0)
    np.testing.assert_array_equal(expanded, opposite[:1])
    np.testing.assert_array_equal(augmented, opposite)


def test_database_augmentation_excludes_self_and_is_deterministic() -> None:
    gallery = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    first = database_side_augmentation(gallery, top_k=1, alpha=0.0)
    second = database_side_augmentation(gallery, top_k=1, alpha=0.0)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    # Row zero selects row one rather than selecting itself.
    expected_zero = gallery[0] + gallery[1] / np.linalg.norm(gallery[1])
    expected_zero /= np.linalg.norm(expected_zero)
    np.testing.assert_allclose(first[0], expected_zero)


def test_k_reciprocal_has_expected_shape_range_and_matching_top1() -> None:
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    gallery = np.asarray(
        [[0.99, 0.01], [-1.0, 0.0], [0.01, 0.99], [0.0, -1.0]]
    )
    distances = k_reciprocal_rerank(
        queries, gallery, k1=2, k2=1, lambda_value=0.3
    )
    assert distances.shape == (2, 4)
    assert np.isfinite(distances).all()
    assert ((0.0 <= distances) & (distances <= 1.0)).all()
    np.testing.assert_array_equal(np.argmin(distances, axis=1), [0, 2])


def test_k_reciprocal_similarity_is_exact_distance_complement() -> None:
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    gallery = np.asarray([[1.0, 0.0], [0.2, 0.8], [-1.0, 0.0]])
    kwargs = {"k1": 2, "k2": 2, "lambda_value": 0.4}
    distance = k_reciprocal_rerank(
        queries, gallery, output="distance", **kwargs
    )
    similarity = k_reciprocal_rerank(
        queries, gallery, output="similarity", **kwargs
    )
    np.testing.assert_allclose(similarity, 1.0 - distance, rtol=0.0, atol=0.0)


def test_k_reciprocal_is_finite_for_fully_tied_embeddings() -> None:
    queries = np.ones((2, 3))
    gallery = np.ones((3, 3))
    first = k_reciprocal_rerank(queries, gallery, k1=2, k2=1)
    second = k_reciprocal_rerank(queries, gallery, k1=2, k2=1)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: query_expansion([[1.0, 0.0]], [[1.0, 0.0]], top_k=2),
            "top_k",
        ),
        (
            lambda: database_side_augmentation(
                [[1.0, 0.0], [0.0, 1.0]], top_k=2
            ),
            "top_k",
        ),
        (
            lambda: k_reciprocal_rerank(
                [[1.0, 0.0]], [[1.0, 0.0]], k1=2, k2=1
            ),
            "k1",
        ),
        (
            lambda: k_reciprocal_rerank(
                [[1.0, 0.0]], [[1.0, 0.0]], k1=1, k2=2
            ),
            "k2",
        ),
        (
            lambda: k_reciprocal_rerank(
                [[1.0, 0.0]], [[1.0, 0.0]], k1=1, k2=1, lambda_value=1.1
            ),
            "lambda_value",
        ),
        (
            lambda: k_reciprocal_rerank(
                [[1.0, 0.0]], [[1.0, 0.0]], k1=1, k2=1, output="scores"
            ),
            "output",
        ),
        (
            lambda: query_expansion([[1.0, 0.0]], [[1.0, 0.0, 0.0]]),
            "dimensions",
        ),
    ],
)
def test_parameter_validation(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
