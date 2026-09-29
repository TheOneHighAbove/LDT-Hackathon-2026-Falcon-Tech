from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.robust_open_set import (
    build_gallery_conditioned_protocol,
    calibrate_robust_open_set,
    select_constrained_candidate_threshold,
    select_pooled_threshold,
    select_weighted_candidate_threshold,
)


def _synthetic_validation(num_identities: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    dimensions = num_identities + 2
    for vehicle_id in range(num_identities):
        for image_number, camera_id in enumerate((0, 0, 1, 1)):
            rows.append(
                {
                    "image_id": f"vehicle-{vehicle_id}-image-{image_number}",
                    "vehicle_id": vehicle_id,
                    "camera_id": camera_id,
                }
            )
            vector = np.zeros(dimensions, dtype=np.float64)
            vector[vehicle_id] = 1.0
            vector[-2 + camera_id] = 0.03
            vector /= np.linalg.norm(vector)
            embeddings.append(vector)
    return pd.DataFrame(rows), np.asarray(embeddings)


def test_protocol_is_deterministic_and_has_no_leakage() -> None:
    frame, _ = _synthetic_validation()
    first = build_gallery_conditioned_protocol(
        frame,
        seed=123,
        unknown_fraction=0.2,
        target_gallery_size=16,
    )
    second = build_gallery_conditioned_protocol(
        frame,
        seed=123,
        unknown_fraction=0.2,
        target_gallery_size=16,
    )
    np.testing.assert_array_equal(first.query_indices, second.query_indices)
    np.testing.assert_array_equal(first.gallery_indices, second.gallery_indices)
    np.testing.assert_array_equal(first.query_is_known, second.query_is_known)
    assert first.known_vehicle_ids == second.known_vehicle_ids
    assert first.unknown_vehicle_ids == second.unknown_vehicle_ids

    query = frame.iloc[first.query_indices]
    gallery = frame.iloc[first.gallery_indices]
    unknown = set(first.unknown_vehicle_ids)
    assert set(query.image_id).isdisjoint(gallery.image_id)
    assert unknown.isdisjoint(gallery.vehicle_id)
    unknown_query = query.loc[~first.query_is_known]
    assert set(unknown_query.vehicle_id) == unknown
    assert unknown_query.vehicle_id.is_unique
    assert len(unknown_query) == len(unknown)
    assert (~first.query_is_known).mean() == pytest.approx(0.2)
    for row in query.loc[first.query_is_known].itertuples():
        assert (
            (gallery.vehicle_id == row.vehicle_id)
            & (gallery.camera_id != row.camera_id)
        ).any()


def test_protocol_reaches_requested_and_default_target_size() -> None:
    frame, _ = _synthetic_validation()
    exact = build_gallery_conditioned_protocol(
        frame, seed=7, unknown_fraction=0.2, target_gallery_size=17
    )
    assert len(exact.gallery_indices) == 17

    # Eight known IDs leave 8 * (4 - 1 query) = 24 eligible gallery rows.
    default = build_gallery_conditioned_protocol(
        frame, seed=7, unknown_fraction=0.2
    )
    assert len(default.gallery_indices) == 24
    with pytest.raises(ValueError, match="too small"):
        build_gallery_conditioned_protocol(
            frame, seed=7, unknown_fraction=0.2, target_gallery_size=7
        )
    with pytest.raises(ValueError, match="exceeds"):
        build_gallery_conditioned_protocol(
            frame, seed=7, unknown_fraction=0.2, target_gallery_size=25
        )


def test_pooled_threshold_maximizes_candidate_micro_f1() -> None:
    pooled = select_pooled_threshold(
        [np.asarray([0.9, 0.8, 0.2]), np.asarray([0.7, 0.6])],
        [np.asarray([1, 1, 0]), np.asarray([1, 0])],
        [np.asarray([1, 0, 0]), np.asarray([1, 0])],
    )
    assert pooled["threshold"] == pytest.approx(0.7)
    assert pooled["f1"] == pytest.approx(2.0 / 3.0)
    assert pooled["tnr"] == 1.0
    assert pooled["tp"] == 2
    assert pooled["fp"] == 1
    assert pooled["fn"] == 1


def test_constrained_threshold_respects_tnr_floor() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    known = np.asarray([1, 0, 1, 0])
    correct = np.asarray([1, 0, 1, 0])
    strict = select_constrained_candidate_threshold(
        scores, known, correct, minimum_tnr=1.0
    )
    assert strict["threshold"] == pytest.approx(0.9)
    assert strict["tnr"] == 1.0
    assert strict["f1"] == pytest.approx(2.0 / 3.0)

    relaxed = select_constrained_candidate_threshold(
        scores, known, correct, minimum_tnr=0.5
    )
    assert relaxed["threshold"] == pytest.approx(0.7)
    assert relaxed["tnr"] == 0.5
    assert relaxed["f1"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="within"):
        select_constrained_candidate_threshold(
            scores, known, correct, minimum_tnr=1.01
        )


def test_weighted_threshold_maximizes_official_candidate_score() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    known = np.asarray([1, 0, 1, 0])
    correct = np.asarray([1, 0, 1, 0])
    selected = select_weighted_candidate_threshold(scores, known, correct)
    # Pure F1 prefers 0.7, but the official 0.7*F1 + 0.3*TNR objective
    # correctly prefers the more conservative 0.9 operating point.
    assert selected["threshold"] == pytest.approx(0.9)
    assert selected["f1"] == pytest.approx(2.0 / 3.0)
    assert selected["tnr"] == pytest.approx(1.0)
    assert selected["candidate_score"] == pytest.approx(23.0 / 30.0)

    with pytest.raises(ValueError, match="sum to 1"):
        select_weighted_candidate_threshold(
            scores, known, correct, f1_weight=0.8, tnr_weight=0.3
        )


def test_complete_report_is_deterministic_json_and_uses_fixed_threshold() -> None:
    frame, embeddings = _synthetic_validation()
    kwargs = {
        "seeds": [11, 12, 13],
        "unknown_fraction": 0.2,
        "target_gallery_size": 16,
    }
    first = calibrate_robust_open_set(embeddings, frame, **kwargs)
    second = calibrate_robust_open_set(embeddings, frame, **kwargs)
    assert first == second
    json.dumps(first, allow_nan=False)
    assert first["raw_cosine_threshold"] == first["pooled_micro_metrics"]["threshold"]
    assert set(first["confidence_calibration"]) >= {"type", "slope", "intercept"}
    assert 0.0 <= first["confidence_threshold"] <= 1.0
    assert len(first["seeds"]) == 3
    for seed in first["seeds"]:
        assert seed["protocol"]["num_gallery"] == 16
        assert seed["protocol"]["query_gallery_image_overlap"] == 0
        assert seed["protocol"]["unknown_gallery_identity_overlap"] == 0
        assert (
            seed["fixed_pooled_threshold"]["threshold"]
            == first["raw_cosine_threshold"]
        )
    assert set(first["fixed_threshold_mean_std"]) >= {"f1", "tnr", "pr_auc"}


def test_optional_oof_feature_model_is_diagnostic_only() -> None:
    frame, embeddings = _synthetic_validation(num_identities=12)
    report = calibrate_robust_open_set(
        embeddings,
        frame,
        seeds=[21, 22, 23, 24],
        unknown_fraction=0.25,
        target_gallery_size=18,
        fit_oof_features=True,
        oof_splits=3,
    )
    diagnostic = report["optional_feature_correctness_model"]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["does_not_replace_raw_cosine_threshold"] is True
    assert diagnostic["model"]["model_type"] == "logistic_candidate_correctness"
    assert np.isfinite(diagnostic["oof_metrics"]["average_precision"])
    assert np.isfinite(diagnostic["candidate_open_set_oof_metrics"]["f1"])
    assert diagnostic["tune_only_feature_search"]["selected_variant"]
    json.dumps(report, allow_nan=False)


def test_optional_oof_feature_variant_can_be_frozen() -> None:
    frame, embeddings = _synthetic_validation(num_identities=12)
    report = calibrate_robust_open_set(
        embeddings,
        frame,
        seeds=[31, 32, 33, 34],
        unknown_fraction=0.25,
        target_gallery_size=18,
        fit_oof_features=True,
        oof_splits=3,
        adaptive_feature_variant="top1_margins_local_density",
    )
    search = report["optional_feature_correctness_model"][
        "tune_only_feature_search"
    ]
    assert search["selected_variant"] == "top1_margins_local_density"
    assert set(search["candidates"]) == {"top1_margins_local_density"}


def test_adaptive_confirmation_is_locked_and_compares_same_protocols() -> None:
    frame, embeddings = _synthetic_validation(num_identities=16)
    report = calibrate_robust_open_set(
        embeddings,
        frame,
        seeds=[21, 22, 23, 24],
        unknown_fraction=0.25,
        target_gallery_size=24,
        fit_oof_features=True,
        oof_splits=3,
        adaptive_confirmation_seeds=[101, 103],
    )
    diagnostic = report["optional_feature_correctness_model"]
    confirmation = diagnostic["locked_confirmation"]
    assert confirmation["seeds"] == [101, 103]
    assert confirmation["locked_before_evaluation"] is True
    assert confirmation["used_for_feature_selection"] is False
    assert confirmation["used_for_model_fit"] is False
    assert confirmation["used_for_threshold_fit"] is False
    assert confirmation["thresholds_derived_from_tune_oof_only"] is True
    assert confirmation["num_samples"] > 0
    assert set(confirmation) >= {
        "scalar_raw_cosine",
        "adaptive_probability",
        "per_seed",
    }
    assert len(confirmation["per_seed"]) == 2
    points = confirmation["adaptive_operating_points"]
    assert set(points) >= {"max_candidate_f1", "tnr_at_least_0_97"}
    assert points["tnr_at_least_0_97"]["tune_metrics"]["tnr"] >= 0.97
    json.dumps(report, allow_nan=False)


def test_adaptive_confirmation_requires_disjoint_tune_seeds() -> None:
    frame, embeddings = _synthetic_validation(num_identities=12)
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_robust_open_set(
            embeddings,
            frame,
            seeds=[21, 22, 23],
            fit_oof_features=True,
            oof_splits=3,
            adaptive_confirmation_seeds=[23, 101],
        )
    with pytest.raises(ValueError, match="requires"):
        calibrate_robust_open_set(
            embeddings,
            frame,
            seeds=[21, 22, 23],
            adaptive_confirmation_seeds=[101],
        )


def test_protocol_input_validation() -> None:
    frame, embeddings = _synthetic_validation()
    bad = frame.copy()
    bad.loc[bad.vehicle_id == 0, "camera_id"] = 0
    with pytest.raises(ValueError, match="at least two cameras"):
        build_gallery_conditioned_protocol(bad, seed=1)
    with pytest.raises(ValueError, match="aligned"):
        calibrate_robust_open_set(embeddings[:-1], frame, seeds=[1])
    with pytest.raises(ValueError, match="unique"):
        calibrate_robust_open_set(embeddings, frame, seeds=[1, 1])
