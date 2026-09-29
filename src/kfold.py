"""Reproducible identity-disjoint K-fold generation and artifact verification.

Each validation identity occurs in exactly one fold and is guaranteed to have
enough observations and cameras for cross-camera retrieval.  Identities that
do not meet the validation constraints are retained in every training split.

Examples
--------
Create three folds::

    python -m src.kfold create --input dataset/train.csv
        --output-dir splits/kfold --n-splits 3

Verify every persisted artifact against the manifest::

    python -m src.kfold verify --manifest splits/kfold/kfold_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .split import DEFAULT_SEED
from .utils import save_json, sha256_file


MANIFEST_NAME = "kfold_manifest.json"
SCHEMA_VERSION = 1


def _load_annotations(
    annotations: str | Path | pd.DataFrame,
    *,
    vehicle_id_column: str,
    camera_id_column: str,
) -> pd.DataFrame:
    if isinstance(annotations, (str, Path)):
        frame = pd.read_csv(annotations)
    elif isinstance(annotations, pd.DataFrame):
        frame = annotations.copy(deep=True)
    else:
        raise TypeError("annotations must be a CSV path or pandas DataFrame")

    required = {"image_id", vehicle_id_column, camera_id_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"annotation columns are missing: {sorted(missing)}")
    if frame.empty:
        raise ValueError("annotations are empty")
    for column in required:
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing values")
    if frame["image_id"].astype(str).duplicated().any():
        raise ValueError("image_id must be unique before creating folds")
    return frame.reset_index(drop=True)


def _positive_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return parsed


def _eligible_identities(
    frame: pd.DataFrame,
    *,
    min_val_images: int,
    min_val_cameras: int,
    vehicle_id_column: str,
    camera_id_column: str,
) -> tuple[list[Any], list[Any]]:
    identities = list(pd.unique(frame[vehicle_id_column]))
    image_counts = frame.groupby(vehicle_id_column, sort=False).size().to_dict()
    camera_counts = (
        frame.groupby(vehicle_id_column, sort=False)[camera_id_column]
        .nunique(dropna=False)
        .to_dict()
    )
    eligible = [
        identity
        for identity in identities
        if image_counts[identity] >= min_val_images
        and camera_counts[identity] >= min_val_cameras
    ]
    eligible_set = set(eligible)
    ineligible = [identity for identity in identities if identity not in eligible_set]
    return eligible, ineligible


def identity_disjoint_kfold(
    annotations: str | Path | pd.DataFrame,
    *,
    n_splits: int = 3,
    seed: int = DEFAULT_SEED,
    min_val_images: int = 2,
    min_val_cameras: int = 2,
    vehicle_id_column: str = "vehicle_id",
    camera_id_column: str = "camera_id",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return deterministic train/validation folds split by whole identity.

    Eligible identities are shuffled with ``seed`` and partitioned as evenly
    as possible by identity count.  Every eligible identity is validation in
    exactly one fold.  An identity that does not satisfy ``min_val_images`` or
    ``min_val_cameras`` is never used for validation and remains available for
    training in every fold.
    """

    n_splits = _positive_integer(n_splits, "n_splits", minimum=2)
    min_val_images = _positive_integer(
        min_val_images, "min_val_images", minimum=2
    )
    min_val_cameras = _positive_integer(
        min_val_cameras, "min_val_cameras", minimum=2
    )
    frame = _load_annotations(
        annotations,
        vehicle_id_column=vehicle_id_column,
        camera_id_column=camera_id_column,
    )
    identities = list(pd.unique(frame[vehicle_id_column]))
    if len(identities) < 2:
        raise ValueError("at least two vehicle identities are required for K-fold")

    eligible, _ = _eligible_identities(
        frame,
        min_val_images=min_val_images,
        min_val_cameras=min_val_cameras,
        vehicle_id_column=vehicle_id_column,
        camera_id_column=camera_id_column,
    )
    if len(eligible) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} requires at least {n_splits} validation-eligible "
            f"identities, found {len(eligible)}"
        )

    rng = np.random.default_rng(int(seed))
    shuffled = [eligible[int(position)] for position in rng.permutation(len(eligible))]
    fold_identities = [
        list(chunk) for chunk in np.array_split(np.asarray(shuffled, dtype=object), n_splits)
    ]

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    validation_seen: set[Any] = set()
    for fold_index, val_identities in enumerate(fold_identities):
        val_identity_set = set(val_identities)
        if not val_identity_set:
            raise RuntimeError(f"fold {fold_index} unexpectedly has no validation IDs")
        if validation_seen.intersection(val_identity_set):
            raise RuntimeError("a validation identity was assigned to multiple folds")
        validation_seen.update(val_identity_set)

        val_mask = frame[vehicle_id_column].isin(val_identity_set)
        train = frame.loc[~val_mask].copy().reset_index(drop=True)
        val = frame.loc[val_mask].copy().reset_index(drop=True)
        if train.empty or val.empty:
            raise RuntimeError(f"fold {fold_index} unexpectedly has an empty partition")
        if set(train[vehicle_id_column]).intersection(val[vehicle_id_column]):
            raise RuntimeError(f"identity leakage detected in fold {fold_index}")
        if (val.groupby(vehicle_id_column).size() < min_val_images).any():
            raise RuntimeError(f"fold {fold_index} contains an ID with too few images")
        val_camera_counts = val.groupby(vehicle_id_column)[camera_id_column].nunique()
        if (val_camera_counts < min_val_cameras).any():
            raise RuntimeError(f"fold {fold_index} contains an ID with too few cameras")
        folds.append((train, val))

    if validation_seen != set(eligible):
        raise RuntimeError("eligible validation identities were not partitioned exactly once")
    return folds


def _ordered_unique_lines(values: pd.Series) -> str:
    return "".join(f"{value}\n" for value in pd.unique(values))


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _artifact_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def save_kfold(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: str | Path,
    *,
    source_annotations: str | Path | pd.DataFrame,
    seed: int = DEFAULT_SEED,
    min_val_images: int = 2,
    min_val_cameras: int = 2,
    vehicle_id_column: str = "vehicle_id",
    camera_id_column: str = "camera_id",
) -> Path:
    """Persist folds and return the path to their hash-bearing manifest."""

    if len(folds) < 2:
        raise ValueError("at least two folds are required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = _load_annotations(
        source_annotations,
        vehicle_id_column=vehicle_id_column,
        camera_id_column=camera_id_column,
    )
    eligible, ineligible = _eligible_identities(
        source,
        min_val_images=min_val_images,
        min_val_cameras=min_val_cameras,
        vehicle_id_column=vehicle_id_column,
        camera_id_column=camera_id_column,
    )
    source_image_ids = set(source["image_id"].astype(str))
    source_identity_ids = set(source[vehicle_id_column])

    manifest_folds: list[dict[str, Any]] = []
    all_validation_ids: set[Any] = set()
    for fold_index, (train, val) in enumerate(folds):
        for partition_name, partition in (("train", train), ("val", val)):
            required = {"image_id", vehicle_id_column, camera_id_column}
            missing = required.difference(partition.columns)
            if missing:
                raise ValueError(
                    f"fold {fold_index} {partition_name} is missing columns: "
                    f"{sorted(missing)}"
                )
            if partition.empty:
                raise ValueError(f"fold {fold_index} {partition_name} is empty")
        train_ids = set(train[vehicle_id_column])
        val_ids = set(val[vehicle_id_column])
        if train_ids.intersection(val_ids):
            raise ValueError(f"fold {fold_index} has train/validation identity overlap")
        if not val_ids.issubset(set(eligible)):
            raise ValueError(f"fold {fold_index} contains a validation-ineligible ID")
        if train_ids.union(val_ids) != source_identity_ids:
            raise ValueError(f"fold {fold_index} does not cover every source identity")
        train_image_ids = set(train["image_id"].astype(str))
        val_image_ids = set(val["image_id"].astype(str))
        if train["image_id"].astype(str).duplicated().any() or val[
            "image_id"
        ].astype(str).duplicated().any():
            raise ValueError(f"fold {fold_index} contains duplicate image_id values")
        if train_image_ids.intersection(val_image_ids):
            raise ValueError(f"fold {fold_index} has train/validation image overlap")
        if train_image_ids.union(val_image_ids) != source_image_ids:
            raise ValueError(f"fold {fold_index} does not cover every source image")
        if len(train) + len(val) != len(source):
            raise ValueError(f"fold {fold_index} row count does not match the source")
        if (val.groupby(vehicle_id_column).size() < min_val_images).any():
            raise ValueError(f"fold {fold_index} violates min_val_images")
        if (
            val.groupby(vehicle_id_column)[camera_id_column].nunique()
            < min_val_cameras
        ).any():
            raise ValueError(f"fold {fold_index} violates min_val_cameras")
        if all_validation_ids.intersection(val_ids):
            raise ValueError("a validation identity occurs in more than one fold")
        all_validation_ids.update(val_ids)

        fold_dir = root / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "train_csv": fold_dir / "train.csv",
            "val_csv": fold_dir / "val.csv",
            "train_image_ids": fold_dir / "train_image_ids.txt",
            "val_image_ids": fold_dir / "val_image_ids.txt",
            "train_vehicle_ids": fold_dir / "train_vehicle_ids.txt",
            "val_vehicle_ids": fold_dir / "val_vehicle_ids.txt",
        }
        train.to_csv(paths["train_csv"], index=False, lineterminator="\n")
        val.to_csv(paths["val_csv"], index=False, lineterminator="\n")
        paths["train_image_ids"].write_text(
            _ordered_unique_lines(train["image_id"]), encoding="utf-8", newline="\n"
        )
        paths["val_image_ids"].write_text(
            _ordered_unique_lines(val["image_id"]), encoding="utf-8", newline="\n"
        )
        paths["train_vehicle_ids"].write_text(
            _ordered_unique_lines(train[vehicle_id_column]),
            encoding="utf-8",
            newline="\n",
        )
        paths["val_vehicle_ids"].write_text(
            _ordered_unique_lines(val[vehicle_id_column]),
            encoding="utf-8",
            newline="\n",
        )

        manifest_folds.append(
            {
                "index": fold_index,
                "directory": fold_dir.relative_to(root).as_posix(),
                "train_images": int(len(train)),
                "val_images": int(len(val)),
                "train_identities": int(train[vehicle_id_column].nunique()),
                "val_identities": int(val[vehicle_id_column].nunique()),
                "artifacts": {
                    name: _artifact_entry(path, root) for name, path in paths.items()
                },
            }
        )

    if all_validation_ids != set(eligible):
        raise ValueError(
            "validation identities across folds do not exactly cover eligible IDs"
        )

    manifest: Mapping[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "identity_disjoint_kfold",
        "seed": int(seed),
        "n_splits": len(folds),
        "constraints": {
            "min_val_images": int(min_val_images),
            "min_val_cameras": int(min_val_cameras),
        },
        "columns": {
            "image_id": "image_id",
            "vehicle_id": vehicle_id_column,
            "camera_id": camera_id_column,
        },
        "source": {
            "rows": int(len(source)),
            "identities": int(source[vehicle_id_column].nunique()),
            "eligible_validation_identities": len(eligible),
            "train_only_identities": len(ineligible),
            "canonical_csv_sha256": _canonical_frame_sha256(source),
        },
        "coverage": {
            "eligible_identities_appear_in_validation_exactly_once": True,
            "train_only_identities_never_appear_in_validation": True,
        },
        "folds": manifest_folds,
    }
    manifest_path = root / MANIFEST_NAME
    save_json(manifest, manifest_path)
    return manifest_path


def create_and_save_kfold(
    annotations: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    n_splits: int = 3,
    seed: int = DEFAULT_SEED,
    min_val_images: int = 2,
    min_val_cameras: int = 2,
    vehicle_id_column: str = "vehicle_id",
    camera_id_column: str = "camera_id",
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame]], Path]:
    """Create, persist, and describe an identity-disjoint K-fold protocol."""

    source = _load_annotations(
        annotations,
        vehicle_id_column=vehicle_id_column,
        camera_id_column=camera_id_column,
    )
    folds = identity_disjoint_kfold(
        source,
        n_splits=n_splits,
        seed=seed,
        min_val_images=min_val_images,
        min_val_cameras=min_val_cameras,
        vehicle_id_column=vehicle_id_column,
        camera_id_column=camera_id_column,
    )
    manifest_path = save_kfold(
        folds,
        output_dir,
        source_annotations=source,
        seed=seed,
        min_val_images=min_val_images,
        min_val_cameras=min_val_cameras,
        vehicle_id_column=vehicle_id_column,
        camera_id_column=camera_id_column,
    )
    return folds, manifest_path


def _safe_artifact_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe artifact path in K-fold manifest: {relative_path!r}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"artifact path escapes K-fold root: {relative_path!r}")
    return resolved


def verify_kfold_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify manifest schema, artifact hashes, and fold identity invariants."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported K-fold manifest schema_version")
    if manifest.get("protocol") != "identity_disjoint_kfold":
        raise ValueError("manifest does not describe identity_disjoint_kfold")
    fold_entries = manifest.get("folds")
    if not isinstance(fold_entries, list) or len(fold_entries) < 2:
        raise ValueError("manifest must contain at least two folds")
    if manifest.get("n_splits") != len(fold_entries):
        raise ValueError("manifest n_splits does not match its fold entries")

    root = path.parent
    vehicle_column = manifest["columns"]["vehicle_id"]
    camera_column = manifest["columns"]["camera_id"]
    min_images = int(manifest["constraints"]["min_val_images"])
    min_cameras = int(manifest["constraints"]["min_val_cameras"])
    validation_seen: set[Any] = set()
    reference_images: set[str] | None = None

    for expected_index, fold_entry in enumerate(fold_entries):
        if fold_entry.get("index") != expected_index:
            raise ValueError("fold indices must be consecutive and ordered")
        artifacts = fold_entry.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"fold {expected_index} has no artifact mapping")
        required_artifacts = {
            "train_csv",
            "val_csv",
            "train_image_ids",
            "val_image_ids",
            "train_vehicle_ids",
            "val_vehicle_ids",
        }
        if required_artifacts.difference(artifacts):
            raise ValueError(f"fold {expected_index} has missing artifact entries")
        resolved: dict[str, Path] = {}
        for name, artifact in artifacts.items():
            artifact_path = _safe_artifact_path(root, artifact["path"])
            if not artifact_path.is_file():
                raise ValueError(f"missing K-fold artifact: {artifact_path}")
            if artifact_path.stat().st_size != int(artifact["size_bytes"]):
                raise ValueError(f"size mismatch for K-fold artifact: {artifact_path}")
            if sha256_file(artifact_path) != artifact["sha256"]:
                raise ValueError(f"SHA-256 mismatch for K-fold artifact: {artifact_path}")
            resolved[name] = artifact_path

        train = pd.read_csv(resolved["train_csv"])
        val = pd.read_csv(resolved["val_csv"])
        required_columns = {"image_id", vehicle_column, camera_column}
        if required_columns.difference(train.columns) or required_columns.difference(
            val.columns
        ):
            raise ValueError(f"fold {expected_index} CSV columns do not match manifest")
        train_ids = set(train[vehicle_column])
        val_ids = set(val[vehicle_column])
        if train_ids.intersection(val_ids):
            raise ValueError(f"identity leakage detected in fold {expected_index}")
        if validation_seen.intersection(val_ids):
            raise ValueError("a validation identity occurs in multiple folds")
        validation_seen.update(val_ids)
        if (val.groupby(vehicle_column).size() < min_images).any():
            raise ValueError(f"fold {expected_index} violates min_val_images")
        if (val.groupby(vehicle_column)[camera_column].nunique() < min_cameras).any():
            raise ValueError(f"fold {expected_index} violates min_val_cameras")

        expected_lists = {
            "train_image_ids": [str(value) for value in pd.unique(train["image_id"])],
            "val_image_ids": [str(value) for value in pd.unique(val["image_id"])],
            "train_vehicle_ids": [
                str(value) for value in pd.unique(train[vehicle_column])
            ],
            "val_vehicle_ids": [str(value) for value in pd.unique(val[vehicle_column])],
        }
        for artifact_name, expected_values in expected_lists.items():
            actual_values = resolved[artifact_name].read_text(
                encoding="utf-8"
            ).splitlines()
            if actual_values != expected_values:
                raise ValueError(
                    f"fold {expected_index} {artifact_name} does not match its CSV"
                )

        all_images = set(train["image_id"].astype(str)).union(
            val["image_id"].astype(str)
        )
        if set(train["image_id"].astype(str)).intersection(
            val["image_id"].astype(str)
        ):
            raise ValueError(f"image leakage detected in fold {expected_index}")
        if reference_images is None:
            reference_images = all_images
        elif all_images != reference_images:
            raise ValueError("folds do not cover the same source image set")

        expected_counts = {
            "train_images": len(train),
            "val_images": len(val),
            "train_identities": train[vehicle_column].nunique(),
            "val_identities": val[vehicle_column].nunique(),
        }
        for name, expected in expected_counts.items():
            if int(fold_entry.get(name, -1)) != int(expected):
                raise ValueError(f"fold {expected_index} has inconsistent {name}")

    if len(validation_seen) != int(
        manifest["source"]["eligible_validation_identities"]
    ):
        raise ValueError("validation identity coverage does not match manifest")
    if reference_images is None or len(reference_images) != int(
        manifest["source"]["rows"]
    ):
        raise ValueError("source row coverage does not match manifest")
    return manifest


def load_kfold_partition(
    manifest_path: str | Path,
    fold_index: int,
    *,
    verify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one persisted fold, optionally verifying the complete protocol."""

    path = Path(manifest_path)
    manifest = (
        verify_kfold_manifest(path)
        if verify
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if isinstance(fold_index, bool) or not isinstance(fold_index, (int, np.integer)):
        raise ValueError("fold_index must be an integer")
    fold_index = int(fold_index)
    folds = manifest["folds"]
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index must be between 0 and {len(folds) - 1}")
    artifacts = folds[fold_index]["artifacts"]
    root = path.parent
    train_path = _safe_artifact_path(root, artifacts["train_csv"]["path"])
    val_path = _safe_artifact_path(root, artifacts["val_csv"]["path"])
    return pd.read_csv(train_path), pd.read_csv(val_path)


def resolve_kfold_partition(
    manifest_path: str | Path,
    fold_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Verify and load a fold together with checkpoint-ready provenance."""

    path = Path(manifest_path).resolve()
    manifest = verify_kfold_manifest(path)
    if isinstance(fold_index, bool) or not isinstance(fold_index, (int, np.integer)):
        raise ValueError("fold_index must be an integer")
    fold_index = int(fold_index)
    if fold_index < 0 or fold_index >= len(manifest["folds"]):
        raise ValueError(
            f"fold_index must be between 0 and {len(manifest['folds']) - 1}"
        )
    fold = manifest["folds"][fold_index]
    train_artifact = fold["artifacts"]["train_csv"]
    val_artifact = fold["artifacts"]["val_csv"]
    train_path = _safe_artifact_path(path.parent, train_artifact["path"])
    val_path = _safe_artifact_path(path.parent, val_artifact["path"])
    provenance = {
        "protocol": "identity_disjoint_kfold",
        "manifest": {
            "path": str(path),
            "sha256": sha256_file(path),
        },
        "fold_index": fold_index,
        "n_splits": int(manifest["n_splits"]),
        "seed": int(manifest["seed"]),
        "constraints": dict(manifest["constraints"]),
        "train_csv": {
            "path": str(train_path),
            "sha256": train_artifact["sha256"],
            "size_bytes": int(train_artifact["size_bytes"]),
        },
        "val_csv": {
            "path": str(val_path),
            "sha256": val_artifact["sha256"],
            "size_bytes": int(val_artifact["size_bytes"]),
        },
    }
    return pd.read_csv(train_path), pd.read_csv(val_path), provenance


def split_provenance_fingerprint(
    provenance: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Return the path-independent identity of a persisted data split."""

    protocol = provenance.get("protocol")
    if protocol == "identity_disjoint_kfold":
        manifest = provenance.get("manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("K-fold provenance has no manifest metadata")
        digest = manifest.get("sha256")
        fold_index = provenance.get("fold_index")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("K-fold provenance has an invalid manifest SHA-256")
        if isinstance(fold_index, bool) or not isinstance(
            fold_index, (int, np.integer)
        ):
            raise ValueError("K-fold provenance has an invalid fold_index")
        return protocol, digest, int(fold_index)
    if protocol == "identity_disjoint_holdout":
        train_csv = provenance.get("train_csv")
        val_csv = provenance.get("val_csv")
        if not isinstance(train_csv, Mapping) or not isinstance(val_csv, Mapping):
            raise ValueError("holdout provenance has incomplete CSV metadata")
        train_digest = train_csv.get("sha256")
        val_digest = val_csv.get("sha256")
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (train_digest, val_digest)
        ):
            raise ValueError("holdout provenance has an invalid CSV SHA-256")
        return protocol, train_digest, val_digest
    raise ValueError(f"unsupported split provenance protocol: {protocol!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create and persist K folds")
    create.add_argument("--input", required=True, type=Path, help="training CSV")
    create.add_argument("--output-dir", required=True, type=Path)
    create.add_argument("--n-splits", type=int, default=3)
    create.add_argument("--seed", type=int, default=DEFAULT_SEED)
    create.add_argument("--min-val-images", type=int, default=2)
    create.add_argument("--min-val-cameras", type=int, default=2)
    create.add_argument("--vehicle-id-column", default="vehicle_id")
    create.add_argument("--camera-id-column", default="camera_id")

    verify = commands.add_parser("verify", help="verify hashes and split invariants")
    verify.add_argument("--manifest", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "create":
        folds, manifest_path = create_and_save_kfold(
            args.input,
            args.output_dir,
            n_splits=args.n_splits,
            seed=args.seed,
            min_val_images=args.min_val_images,
            min_val_cameras=args.min_val_cameras,
            vehicle_id_column=args.vehicle_id_column,
            camera_id_column=args.camera_id_column,
        )
        print(
            f"saved {len(folds)} identity-disjoint folds and manifest to "
            f"{manifest_path}"
        )
    else:
        manifest = verify_kfold_manifest(args.manifest)
        print(
            f"verified {manifest['n_splits']} folds under "
            f"{Path(args.manifest).parent}"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "create_and_save_kfold",
    "identity_disjoint_kfold",
    "load_kfold_partition",
    "resolve_kfold_partition",
    "save_kfold",
    "split_provenance_fingerprint",
    "verify_kfold_manifest",
]
