from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    calibrate_open_set,
    derive_known_mask,
    evaluate_open_set,
    evaluate_retrieval,
    search_rejection_threshold,
    split_known_unknown_ids,
)


def test_retrieval_metrics_match_hand_calculation_and_skip_no_match_query() -> None:
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    gallery = np.asarray(
        [
            [1.0, 0.0],  # pid 1, cross-camera positive at rank 1
            [0.8, 0.6],  # wrong identity at rank 2
            [0.6, 0.8],  # pid 1, cross-camera positive at rank 3
            [0.99, 0.01],  # pid 1, same-camera junk (must be removed)
        ]
    )
    result = evaluate_retrieval(
        queries,
        gallery,
        query_pids=[1, 3],
        gallery_pids=[1, 2, 1, 1],
        query_camera_ids=[0, 0],
        gallery_camera_ids=[1, 1, 2, 0],
    )
    assert result["mAP"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert result["mINP"] == pytest.approx(2.0 / 3.0)
    assert result["rank1"] == 1.0
    assert result["rank5"] == 1.0
    assert result["num_valid_queries"] == 1
    assert result["num_ignored_queries"] == 1


def test_same_source_excludes_self_and_same_pid_camera() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    result = evaluate_retrieval(
        embeddings,
        embeddings,
        query_pids=[1, 1, 2, 2],
        gallery_pids=[1, 1, 2, 2],
        query_camera_ids=[0, 1, 0, 1],
        gallery_camera_ids=[0, 1, 0, 1],
    )
    assert result["num_valid_queries"] == 4
    assert result["mAP"] == 1.0
    assert result["rank1"] == 1.0
    assert result["mINP"] == 1.0


def test_exact_image_ids_exclude_self_for_copied_arrays() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    image_ids = ["a", "b", "c", "d"]
    result = evaluate_retrieval(
        embeddings.copy(),
        embeddings.copy(),
        query_pids=[1, 1, 2, 2],
        gallery_pids=[1, 1, 2, 2],
        query_image_ids=image_ids,
        gallery_image_ids=image_ids,
    )
    assert result["mAP"] == 1.0
    assert result["num_valid_queries"] == 4


def test_f1_threshold_tnr_and_pr_auc_are_deterministic() -> None:
    scores = np.asarray([0.95, 0.80, 0.70, 0.10])
    known = np.asarray([1, 1, 0, 0])
    first = search_rejection_threshold(scores, known)
    second = search_rejection_threshold(scores, known)
    assert first == second
    assert first["threshold"] == pytest.approx(0.80)
    assert first["f1"] == 1.0
    assert first["tnr"] == 1.0
    assert first["pr_auc"] == 1.0

    fixed = evaluate_open_set(scores, known, threshold=0.90)
    assert fixed["f1"] == pytest.approx(2.0 / 3.0)
    assert fixed["tnr"] == 1.0


def test_candidate_f1_penalizes_wrong_top1_for_a_known_query() -> None:
    scores = np.asarray([0.95, 0.90, 0.10])
    known = np.asarray([1, 1, 0])
    top1_correct = np.asarray([1, 0, 0])
    result = evaluate_open_set(
        scores,
        known,
        threshold=0.80,
        top1_correct=top1_correct,
    )
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["f1"] == pytest.approx(0.5)
    assert result["tnr"] == 1.0
    assert result["unknown_fp"] == 0


def test_open_set_calibration_uses_explicit_known_and_unknown_ids() -> None:
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    gallery = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = calibrate_open_set(
        queries,
        gallery,
        query_pids=[1, 2, 9],
        gallery_pids=[1, 2],
        known_pids=[1, 2],
        unknown_pids=[9],
    )
    assert result["f1"] == 1.0
    assert result["tnr"] == 1.0
    assert result["num_known"] == 2
    assert result["num_unknown"] == 1
    assert result["is_known"].tolist() == [True, True, False]


def test_known_unknown_helpers_validate_and_are_reproducible() -> None:
    with pytest.raises(ValueError, match="not assigned"):
        derive_known_mask(
            [1, 2, 3], [1], known_pids=[1], unknown_pids=[2]
        )
    first = split_known_unknown_ids([1, 1, 2, 3, 4, 5], seed=42)
    second = split_known_unknown_ids([1, 1, 2, 3, 4, 5], seed=42)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]).union(first[1]) == {1, 2, 3, 4, 5}
