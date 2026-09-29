from __future__ import annotations

import json

import numpy as np
import pytest

from src.score_calibration import (
    CANDIDATE_FEATURE_NAMES,
    CandidateCorrectnessModel,
    evaluate_group_oof,
    extract_candidate_features,
    fit_candidate_correctness_model,
)


def _gallery_similarity() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.8, 0.2, 0.1, 0.0],
            [0.8, 1.0, 0.4, 0.3, 0.2],
            [0.2, 0.4, 1.0, 0.6, 0.5],
            [0.1, 0.3, 0.6, 1.0, 0.7],
            [0.0, 0.2, 0.5, 0.7, 1.0],
        ],
        dtype=np.float64,
    )


def test_extract_candidate_features_matches_definitions() -> None:
    scores = np.asarray(
        [
            [0.9, 0.7, 0.1, -0.1, 0.3],
            [0.2, 0.4, 0.8, 0.5, 0.6],
        ]
    )
    features = extract_candidate_features(
        scores, _gallery_similarity(), density_k=2
    )

    assert features.shape == (2, len(CANDIDATE_FEATURE_NAMES))
    assert features[0, 0] == pytest.approx(0.9)
    assert features[0, 1] == pytest.approx(0.2)
    assert features[0, 2] == pytest.approx(0.9 - np.mean([0.7, 0.3, 0.1, -0.1]))
    assert features[0, 3] == pytest.approx(
        (0.9 - scores[0].mean()) / scores[0].std()
    )
    # Candidate zero has nearest gallery similarities 0.8 and 0.2.
    assert features[0, 4] == pytest.approx(0.5)
    assert features[0, 5] == pytest.approx(0.2)
    assert features[0, 6] == 1.0
    # Candidate two has nearest gallery similarities 0.6 and 0.5.
    assert features[1, 4] == pytest.approx(0.55)
    assert features[1, 5] == pytest.approx(0.4)
    assert features[1, 6] == 1.0


def test_feature_extraction_handles_small_and_constant_gallery_rows() -> None:
    scores = np.asarray([[0.4, 0.4], [0.2, 0.2]])
    gallery = np.asarray([[1.0, 0.25], [0.25, 1.0]])
    features = extract_candidate_features(scores, gallery, density_k=10)
    assert np.isfinite(features).all()
    assert features[:, 1].tolist() == [0.0, 0.0]
    assert features[:, 2].tolist() == [0.0, 0.0]
    assert features[:, 3].tolist() == [0.0, 0.0]
    assert features[:, 4].tolist() == [0.25, 0.25]
    assert features[:, 5].tolist() == [0.75, 0.75]
    assert features[:, 6].tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("scores", "gallery", "match"),
    [
        (np.ones(3), np.eye(3), "two-dimensional"),
        (np.asarray([[0.0, np.nan]]), np.eye(2), "NaN"),
        (np.asarray([[0.0, 1.1]]), np.eye(2), "cosine"),
        (np.asarray([[0.0, 0.1, 0.2]]), np.eye(2), "width"),
        (np.asarray([[0.0, 0.1]]), np.ones((2, 3)), "square"),
        (
            np.asarray([[0.0, 0.1]]),
            np.asarray([[1.0, 0.3], [0.1, 1.0]]),
            "symmetric",
        ),
    ],
)
def test_feature_extraction_rejects_invalid_matrices(
    scores: np.ndarray, gallery: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        extract_candidate_features(scores, gallery)


def test_logistic_model_round_trip_and_numpy_probabilities(tmp_path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(240, len(CANDIDATE_FEATURE_NAMES)))
    target_logit = 2.5 * features[:, 0] + features[:, 1] - 0.5 * features[:, 4]
    labels = (target_logit > np.median(target_logit)).astype(np.uint8)
    model = fit_candidate_correctness_model(features, labels)

    probabilities = model.predict_proba(features)
    assert probabilities.shape == (len(features), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean((probabilities[:, 1] >= 0.5) == labels) > 0.95

    restored = CandidateCorrectnessModel.from_dict(model.to_dict())
    assert np.array_equal(restored.predict_proba(features), probabilities)
    path = tmp_path / "candidate_calibration.json"
    model.save_json(path)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    loaded = CandidateCorrectnessModel.load_json(path)
    assert np.array_equal(loaded.predict_proba(features), probabilities)


def test_model_and_fit_validate_inputs() -> None:
    features = np.arange(21, dtype=np.float64).reshape(3, 7)
    with pytest.raises(ValueError, match="both target classes"):
        fit_candidate_correctness_model(features, [1, 1, 1])
    with pytest.raises(ValueError, match="match features"):
        fit_candidate_correctness_model(features, [0, 1])

    model = CandidateCorrectnessModel(
        coefficients=np.ones(7),
        intercept=0.0,
        feature_mean=np.zeros(7),
        feature_scale=np.ones(7),
    )
    with pytest.raises(ValueError, match="7 columns"):
        model.predict_proba(np.ones((2, 6)))
    payload = model.to_dict()
    payload["version"] = 999
    with pytest.raises(ValueError, match="version"):
        CandidateCorrectnessModel.from_dict(payload)


def test_group_oof_is_deterministic_group_disjoint_and_predictive() -> None:
    rng = np.random.default_rng(19)
    group_count = 30
    rows_per_group = 6
    groups = np.repeat(np.arange(group_count), rows_per_group)
    labels = np.tile([0, 0, 0, 1, 1, 1], group_count)
    features = rng.normal(
        scale=0.25, size=(len(groups), len(CANDIDATE_FEATURE_NAMES))
    )
    features[:, 0] += labels * 2.0 - 1.0
    features[:, 1] += labels * 0.8

    first = evaluate_group_oof(
        features, labels, groups, n_splits=5, random_state=123
    )
    second = evaluate_group_oof(
        features, labels, groups, n_splits=5, random_state=123
    )
    assert np.array_equal(first["fold_assignments"], second["fold_assignments"])
    assert np.array_equal(first["probabilities"], second["probabilities"])
    assert first["metrics"]["average_precision"] > 0.99
    assert first["metrics"]["roc_auc"] > 0.99
    assert first["metrics"]["f1"] > 0.95
    assert len(first["folds"]) == 5
    for group in np.unique(groups):
        assert np.unique(first["fold_assignments"][groups == group]).size == 1


def test_group_oof_rejects_impossible_protocols() -> None:
    features = np.arange(56, dtype=np.float64).reshape(
        8, len(CANDIDATE_FEATURE_NAMES)
    )
    labels = np.asarray([0, 1] * 4)
    with pytest.raises(ValueError, match="number of groups"):
        evaluate_group_oof(features, labels, np.zeros(8), n_splits=2)
    with pytest.raises(ValueError, match="missing"):
        evaluate_group_oof(
            features, labels, [0, 0, 1, 1, 2, 2, None, None], n_splits=2
        )
