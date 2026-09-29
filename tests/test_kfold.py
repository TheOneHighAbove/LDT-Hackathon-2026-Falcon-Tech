from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

from src.kfold import (
    create_and_save_kfold,
    identity_disjoint_kfold,
    load_kfold_partition,
    resolve_kfold_partition,
    verify_kfold_manifest,
)
from src.train import _parse_args as parse_train_args
from src.train import _check_resume_split, _resolve_training_split
from src.validate import _check_checkpoint_split, _parse_args as parse_validate_args
from src.validate import _resolve_validation_split


def _annotations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    identity_cameras = {
        "a": (0, 1, 2),
        "b": (0, 1),
        "c": (1, 2, 2),
        "d": (0, 2),
        "e": (0, 1, 1),
        "f": (1, 2),
        "single-image": (0,),
        "single-camera": (4, 4),
    }
    for vehicle_id, cameras in identity_cameras.items():
        for occurrence, camera_id in enumerate(cameras):
            rows.append(
                {
                    "image_id": f"{vehicle_id}-{occurrence}",
                    "vehicle_id": vehicle_id,
                    "camera_id": camera_id,
                    "x": occurrence,
                }
            )
    return pd.DataFrame(rows)


def test_kfold_is_deterministic_disjoint_and_covers_each_eligible_id_once() -> None:
    frame = _annotations()
    folds = identity_disjoint_kfold(frame, n_splits=3, seed=17)
    repeated = identity_disjoint_kfold(frame, n_splits=3, seed=17)

    val_seen: list[str] = []
    for (train, val), (train_again, val_again) in zip(folds, repeated, strict=True):
        pd.testing.assert_frame_equal(train, train_again)
        pd.testing.assert_frame_equal(val, val_again)
        assert set(train.vehicle_id).isdisjoint(val.vehicle_id)
        assert (val.groupby("vehicle_id").size() >= 2).all()
        assert (val.groupby("vehicle_id").camera_id.nunique() >= 2).all()
        assert set(train.image_id).union(val.image_id) == set(frame.image_id)
        val_seen.extend(val.vehicle_id.unique().tolist())

    assert set(val_seen) == {"a", "b", "c", "d", "e", "f"}
    assert len(val_seen) == len(set(val_seen))
    for train, val in folds:
        assert "single-image" in set(train.vehicle_id)
        assert "single-camera" in set(train.vehicle_id)
        assert "single-image" not in set(val.vehicle_id)
        assert "single-camera" not in set(val.vehicle_id)


def test_kfold_rejects_insufficient_eligible_identities() -> None:
    with pytest.raises(ValueError, match="validation-eligible"):
        identity_disjoint_kfold(_annotations(), n_splits=7)


def test_kfold_persists_id_lists_hash_manifest_and_loads(tmp_path) -> None:
    frame = _annotations()
    folds, manifest_path = create_and_save_kfold(
        frame, tmp_path / "kfold", n_splits=3, seed=91
    )
    manifest = verify_kfold_manifest(manifest_path)

    assert manifest["protocol"] == "identity_disjoint_kfold"
    assert manifest["n_splits"] == 3
    assert manifest["source"]["train_only_identities"] == 2
    assert len(manifest["source"]["canonical_csv_sha256"]) == 64
    for entry in manifest["folds"]:
        assert set(entry["artifacts"]) == {
            "train_csv",
            "val_csv",
            "train_image_ids",
            "val_image_ids",
            "train_vehicle_ids",
            "val_vehicle_ids",
        }
        for artifact in entry["artifacts"].values():
            assert len(artifact["sha256"]) == 64

    loaded_train, loaded_val = load_kfold_partition(manifest_path, 1)
    pd.testing.assert_frame_equal(loaded_train, folds[1][0])
    pd.testing.assert_frame_equal(loaded_val, folds[1][1])
    fold_entry = manifest["folds"][1]
    val_ids_path = manifest_path.parent / fold_entry["artifacts"][
        "val_vehicle_ids"
    ]["path"]
    assert val_ids_path.read_text(encoding="utf-8").splitlines() == list(
        folds[1][1].vehicle_id.unique()
    )


def test_manifest_verifier_detects_tampering(tmp_path) -> None:
    _, manifest_path = create_and_save_kfold(
        _annotations(), tmp_path / "kfold", n_splits=3
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = manifest_path.parent / manifest["folds"][0]["artifacts"][
        "train_csv"
    ]["path"]
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_kfold_manifest(manifest_path)


def test_resolve_partition_returns_auditable_absolute_provenance(tmp_path) -> None:
    folds, manifest_path = create_and_save_kfold(
        _annotations(), tmp_path / "kfold", n_splits=3, seed=23
    )
    train, val, provenance = resolve_kfold_partition(manifest_path, 2)

    pd.testing.assert_frame_equal(train, folds[2][0])
    pd.testing.assert_frame_equal(val, folds[2][1])
    assert provenance["protocol"] == "identity_disjoint_kfold"
    assert provenance["fold_index"] == 2
    assert provenance["seed"] == 23
    assert len(provenance["manifest"]["sha256"]) == 64
    assert len(provenance["train_csv"]["sha256"]) == 64
    assert len(provenance["val_csv"]["sha256"]) == 64
    assert str(manifest_path.resolve()) == provenance["manifest"]["path"]


def test_train_and_validate_resolve_the_same_verified_fold(tmp_path) -> None:
    folds, manifest_path = create_and_save_kfold(
        _annotations(), tmp_path / "kfold", n_splits=3, seed=5
    )
    args = argparse.Namespace(
        kfold_manifest=manifest_path,
        fold_index=1,
        recreate_split=False,
    )
    config: dict[str, object] = {}

    train, train_val, train_provenance = _resolve_training_split(config, args)
    validation_val, validation_provenance = _resolve_validation_split(config, args)

    pd.testing.assert_frame_equal(train, folds[1][0])
    pd.testing.assert_frame_equal(train_val, folds[1][1])
    pd.testing.assert_frame_equal(validation_val, folds[1][1])
    assert train_provenance == validation_provenance
    _check_checkpoint_split(
        {"source": {"data_split": train_provenance}}, validation_provenance
    )


def test_split_selection_rejects_partial_or_mismatched_kfold_arguments(tmp_path) -> None:
    _, manifest_path = create_and_save_kfold(
        _annotations(), tmp_path / "kfold", n_splits=3
    )
    config: dict[str, object] = {}
    with pytest.raises(ValueError, match="must be provided together"):
        _resolve_training_split(
            config,
            argparse.Namespace(
                kfold_manifest=manifest_path,
                fold_index=None,
                recreate_split=False,
            ),
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_training_split(
            config,
            argparse.Namespace(
                kfold_manifest=manifest_path,
                fold_index=0,
                recreate_split=True,
            ),
        )

    _, _, fold_zero = resolve_kfold_partition(manifest_path, 0)
    _, _, fold_one = resolve_kfold_partition(manifest_path, 1)
    with pytest.raises(ValueError, match="does not match"):
        _check_checkpoint_split({"source": {"data_split": fold_zero}}, fold_one)
    with pytest.raises(ValueError, match="different data split"):
        _check_resume_split({"data_split": fold_zero}, fold_one)


def test_train_and_validate_cli_expose_kfold_selection() -> None:
    train_args = parse_train_args(
        ["--kfold-manifest", "folds/kfold/kfold_manifest.json", "--fold-index", "2"]
    )
    validation_args = parse_validate_args(
        ["--kfold-manifest", "folds/kfold/kfold_manifest.json", "--fold-index", "2"]
    )
    assert train_args.fold_index == validation_args.fold_index == 2
    assert train_args.kfold_manifest == validation_args.kfold_manifest
