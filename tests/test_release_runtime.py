from __future__ import annotations

from pathlib import Path

import numpy as np

from src.release_io import write_submission
from src.release_reranking import (
    augment_static_gallery,
    prepare_retrieval,
    refusal_features,
)
from src.reranking import database_side_augmentation
from src.score_calibration import (
    FINAL_RANKER_REFUSAL_FEATURE_NAMES,
    extract_adaptive_refusal_features,
    extract_selected_candidate_refusal_features,
)


def test_release_gallery_augmentation_is_numerically_locked() -> None:
    values = np.random.default_rng(4103).normal(size=(31, 13)).astype(np.float32)
    expected = database_side_augmentation(values, top_k=5, alpha=2.0)
    observed = augment_static_gallery(values, top_k=5, alpha=2.0)
    np.testing.assert_array_equal(observed, expected)


def test_selected_refusal_features_match_legacy_for_cosine_top1() -> None:
    rng = np.random.default_rng(917)
    query = rng.normal(size=(7, 12))
    gallery = rng.normal(size=(15, 12))
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    similarities = query @ gallery.T
    gallery_similarity = gallery @ gallery.T
    expected = extract_adaptive_refusal_features(similarities, gallery_similarity)
    observed = extract_selected_candidate_refusal_features(
        similarities,
        gallery_similarity,
        np.argmax(similarities, axis=1),
    )
    np.testing.assert_array_equal(observed, expected)


def test_release_refusal_is_independent_of_other_queries() -> None:
    rng = np.random.default_rng(260927)
    query = rng.normal(size=(3, 16)).astype(np.float32)
    gallery = rng.normal(size=(12, 16)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    final_scores = rng.normal(size=(3, len(gallery))).astype(np.float32)
    final_top1 = np.argmax(final_scores, axis=1)
    config = {
        "dba_top_k": 3,
        "dba_alpha": 2.0,
        "gallery_refinement_top_k": 2,
        "gallery_refinement_alpha": 1.0,
        "density_k": 5,
        "model": {"feature_names": FINAL_RANKER_REFUSAL_FEATURE_NAMES},
    }

    together = refusal_features(
        np.concatenate((query, gallery)),
        len(query),
        final_top1,
        final_scores,
        config,
    )
    for index in range(len(query)):
        alone = refusal_features(
            np.concatenate((query[index : index + 1], gallery)),
            1,
            final_top1[index : index + 1],
            final_scores[index : index + 1],
            config,
        )
        np.testing.assert_allclose(together[index], alone[0], rtol=0.0, atol=1e-7)


def test_release_retrieval_is_independent_of_other_queries() -> None:
    rng = np.random.default_rng(260928)

    def normalized(rows: int, columns: int) -> np.ndarray:
        values = rng.normal(size=(rows, columns)).astype(np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    query_count, gallery_count = 3, 31
    total = query_count + gallery_count
    fused = normalized(total, 20)
    osnet = normalized(total, 14)
    parts = {"mean": normalized(total, 11), "h2": normalized(total, 9)}
    protocol = {
        "qi": np.arange(query_count),
        "gi": np.arange(query_count, total),
    }
    together = prepare_retrieval(protocol, fused, osnet, parts)

    for index in range(query_count):
        selected = np.concatenate(([index], np.arange(query_count, total)))
        single_protocol = {
            "qi": np.asarray([0]),
            "gi": np.arange(1, gallery_count + 1),
        }
        alone = prepare_retrieval(
            single_protocol,
            fused[selected],
            osnet[selected],
            {name: values[selected] for name, values in parts.items()},
        )
        np.testing.assert_allclose(
            together["base"][index], alone["base"][0], rtol=2e-6, atol=1e-7
        )
        np.testing.assert_array_equal(
            together["candidates"][index], alone["candidates"][0]
        )
        np.testing.assert_allclose(
            together["raw_fused"][index],
            alone["raw_fused"][0],
            rtol=2e-6,
            atol=1e-7,
        )
        for name in parts:
            np.testing.assert_allclose(
                together["part_scores"][name][index],
                alone["part_scores"][name][0],
                rtol=2e-6,
                atol=1e-7,
            )


def test_release_submission_is_headerless(tmp_path: Path) -> None:
    query = np.asarray(["q1", "q2"])
    gallery = np.asarray([f"g{index}" for index in range(10)])
    order = np.stack((np.arange(10), np.arange(9, -1, -1)))
    path = tmp_path / "submission.csv"
    write_submission(query, gallery, order, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == "q1," + ",".join(f"g{index}" for index in range(10))
    assert lines[1] == "q2," + ",".join(f"g{index}" for index in range(9, -1, -1))
