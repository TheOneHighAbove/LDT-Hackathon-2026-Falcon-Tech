from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.postprocess_eval import (
    build_identity_stratified_protocol,
    build_postprocess_grid,
    evaluate_from_files,
    evaluate_postprocessing,
)
from src.utils import sha256_file


def _synthetic_validation(
    num_identities: int = 6,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    dimensions = num_identities + 3
    for vehicle_id in range(num_identities):
        for image_number, camera_id in enumerate((0, 0, 1, 2)):
            rows.append(
                {
                    "image_id": f"vehicle-{vehicle_id}-image-{image_number}",
                    "vehicle_id": vehicle_id,
                    "camera_id": camera_id,
                }
            )
            vector = np.zeros(dimensions, dtype=np.float32)
            vector[vehicle_id] = 1.0
            vector[num_identities + camera_id] = 0.05
            vector /= np.linalg.norm(vector)
            embeddings.append(vector)
    return pd.DataFrame(rows), np.asarray(embeddings, dtype=np.float32)


def test_identity_stratified_protocol_is_deterministic_and_leakage_safe() -> None:
    frame, _ = _synthetic_validation()
    first = build_identity_stratified_protocol(frame, seed=73)
    second = build_identity_stratified_protocol(frame, seed=73)
    np.testing.assert_array_equal(first.query_indices, second.query_indices)
    np.testing.assert_array_equal(first.gallery_indices, second.gallery_indices)

    query = frame.iloc[first.query_indices]
    gallery = frame.iloc[first.gallery_indices]
    assert len(query) == frame.vehicle_id.nunique()
    assert query.vehicle_id.nunique() == len(query)
    assert set(query.image_id).isdisjoint(gallery.image_id)
    assert set(query.index).union(gallery.index) == set(frame.index)
    for row in query.itertuples(index=False):
        assert (
            (gallery.vehicle_id == row.vehicle_id)
            & (gallery.camera_id != row.camera_id)
        ).any()


def test_grid_is_canonical_de_duplicated_and_always_contains_raw() -> None:
    grid = build_postprocess_grid(
        dba_top_k=[0, 0, 1],
        dba_alpha=[1.0, 2.0],
        qe_top_k=[0, 1, 1],
        qe_alpha=[0.5, 1.0],
    )
    method_ids = [spec.method_id for spec in grid]
    assert method_ids[0] == "raw"
    assert len(method_ids) == len(set(method_ids)) == 9
    assert "dba_k1_a2__qe_k1_a0.5" in method_ids

    raw_only = build_postprocess_grid(
        dba_top_k=[0],
        dba_alpha=[1.0, 9.0],
        qe_top_k=[0],
        qe_alpha=[0.1, 8.0],
    )
    assert [spec.method_id for spec in raw_only] == ["raw"]


def test_repeated_tune_confirmation_report_is_deterministic_and_complete() -> None:
    frame, embeddings = _synthetic_validation()
    kwargs = {
        "tune_seeds": [11, 13],
        "confirmation_seeds": [17, 19],
        "dba_top_k": [0, 1],
        "dba_alpha": [1.0],
        "qe_top_k": [0, 1],
        "qe_alpha": [1.0],
    }
    first = evaluate_postprocessing(embeddings, frame, **kwargs)
    second = evaluate_postprocessing(embeddings, frame, **kwargs)
    assert first == second
    json.dumps(first, allow_nan=False)

    assert first["tune"]["selected_method_id"] == "raw"
    assert first["selected_method"]["method_id"] == "raw"
    assert len(first["tune"]["candidates"]) == 4
    assert len(first["tune"]["protocols"]) == 2
    assert len(first["confirmation"]["protocols"]) == 2
    assert first["confirmation"]["selected"]["aggregate"]["mAP"]["mean"] == 1.0
    for protocol in (
        first["tune"]["protocols"] + first["confirmation"]["protocols"]
    ):
        assert protocol["query_gallery_image_overlap"] == 0
        assert protocol["queries_with_cross_camera_positive"] == 6
        assert len(protocol["protocol_sha256"]) == 64


def test_file_report_binds_provenance_and_has_stable_bytes(tmp_path: Path) -> None:
    frame, embeddings = _synthetic_validation()
    embeddings_path = tmp_path / "embeddings.npy"
    annotations_path = tmp_path / "val.csv"
    checkpoint_path = tmp_path / "best.pt"
    output_path = tmp_path / "report.json"
    np.save(embeddings_path, embeddings)
    frame.to_csv(annotations_path, index=False)
    checkpoint_path.write_bytes(b"deterministic-checkpoint-placeholder")

    kwargs = {
        "checkpoint_path": checkpoint_path,
        "tune_seeds": [23],
        "confirmation_seeds": [29],
        "dba_top_k": [0],
        "dba_alpha": [1.0],
        "qe_top_k": [0],
        "qe_alpha": [1.0],
    }
    first = evaluate_from_files(
        embeddings_path, annotations_path, output_path, **kwargs
    )
    first_bytes = output_path.read_bytes()
    second = evaluate_from_files(
        embeddings_path, annotations_path, output_path, **kwargs
    )
    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["provenance"]["embeddings"]["sha256"] == sha256_file(
        embeddings_path
    )
    assert first["provenance"]["annotations"]["sha256"] == sha256_file(
        annotations_path
    )
    assert first["provenance"]["checkpoint"]["sha256"] == sha256_file(
        checkpoint_path
    )
    assert len(first["provenance"]["provenance_sha256"]) == 64
    assert len(first["report_payload_sha256"]) == 64


def test_protocol_and_seed_validation_fail_closed() -> None:
    frame, embeddings = _synthetic_validation()
    bad = frame.copy()
    bad.loc[bad.vehicle_id == 0, "camera_id"] = 0
    with pytest.raises(ValueError, match="at least two cameras"):
        build_identity_stratified_protocol(bad, seed=1)

    with pytest.raises(ValueError, match="must be disjoint"):
        evaluate_postprocessing(
            embeddings,
            frame,
            tune_seeds=[1, 2],
            confirmation_seeds=[2, 3],
            dba_top_k=[0],
            dba_alpha=[1.0],
            qe_top_k=[0],
            qe_alpha=[1.0],
        )
    with pytest.raises(ValueError, match="row-aligned"):
        evaluate_postprocessing(
            embeddings[:-1],
            frame,
            tune_seeds=[1],
            confirmation_seeds=[2],
        )
