"""Fail-closed verifier for score-optimized release files and outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import sha256_file

LIMIT_BYTES = 2_000_000_000


def verify(
    output_dir: Path, weights_dir: Path, query_csv: Path, gallery_csv: Path
) -> dict:
    manifest_path = output_dir / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    query = pd.read_csv(query_csv)
    gallery = pd.read_csv(gallery_csv)
    submission_columns = ["query_id", *[f"gallery_id_{i}" for i in range(1, 11)]]
    submission = pd.read_csv(
        output_dir / "submission.csv",
        header=None,
        names=submission_columns,
        dtype=str,
        keep_default_na=False,
    )
    candidates = pd.read_csv(output_dir / "candidates.csv")
    embeddings = np.load(output_dir / "embeddings.npy", allow_pickle=False)

    if int(manifest.get("query_count", -1)) != len(query) or int(
        manifest.get("gallery_count", -1)
    ) != len(gallery):
        raise ValueError("manifest query/gallery counts do not match input CSVs")
    for flag in ("uses_other_queries", "uses_camera_at_inference", "uses_csv_order"):
        if manifest.get(flag) is not False:
            raise ValueError(f"manifest release-safety flag is not false: {flag}")

    if list(submission.columns) != submission_columns:
        raise ValueError("submission.csv schema mismatch")
    expected_query_ids = query.image_id.astype(str).tolist()
    if (
        len(submission) != len(query)
        or submission.query_id.astype(str).duplicated().any()
        or submission.query_id.astype(str).tolist() != expected_query_ids
    ):
        raise ValueError("submission.csv must contain exactly one row per query")
    gallery_ids = set(gallery.image_id.astype(str))
    if not submission.iloc[:, 1:].astype(str).isin(gallery_ids).all().all():
        raise ValueError("submission.csv contains an ID outside the gallery")
    if submission.iloc[:, 1:].nunique(axis=1).ne(10).any():
        raise ValueError("submission.csv top-10 IDs must be unique per query")
    if list(candidates.columns) != ["query_id", "gallery_id", "confidence"]:
        raise ValueError("candidates.csv schema mismatch")
    if candidates.query_id.astype(str).duplicated().any():
        raise ValueError("candidates.csv contains duplicate accepted queries")
    if not set(candidates.query_id.astype(str)).issubset(
        set(query.image_id.astype(str))
    ):
        raise ValueError("candidates.csv contains an unknown query")
    if not candidates.gallery_id.astype(str).isin(gallery_ids).all():
        raise ValueError("candidates.csv contains an ID outside the gallery")
    expected_top1 = submission.set_index(submission.query_id.astype(str))[
        "gallery_id_1"
    ].astype(str)
    observed_top1 = candidates.set_index(candidates.query_id.astype(str))[
        "gallery_id"
    ].astype(str)
    if not observed_top1.equals(expected_top1.loc[observed_top1.index]):
        raise ValueError("accepted candidate is not the submission top-1")
    if not candidates.confidence.between(0.0, 1.0).all():
        raise ValueError("candidate confidence must be within [0, 1]")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(query) + len(gallery)
        or embeddings.shape[1] < 1
        or embeddings.dtype != np.float32
    ):
        raise ValueError("embeddings.npy shape or dtype mismatch")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings.npy contains non-finite values")

    observed_weight_bytes = 0
    for name, expected in manifest["weights"].items():
        candidates_paths = (weights_dir / name, weights_dir / "release" / name)
        path = next((value for value in candidates_paths if value.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"release weight is missing: {name}")
        if (
            path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"release weight digest mismatch: {name}")
        observed_weight_bytes += path.stat().st_size
    if observed_weight_bytes != manifest["total_weight_bytes"]:
        raise ValueError("release weight byte total mismatch")
    if observed_weight_bytes > LIMIT_BYTES:
        raise ValueError("release weights exceed the 2 GB hard limit")

    required_artifacts = {
        "submission.csv",
        "candidates.csv",
        "embeddings.npy",
        "gallery_index.npz",
    }
    if set(manifest.get("artifacts", {})) != required_artifacts:
        raise ValueError("manifest artifact allowlist mismatch")
    for name, expected in manifest["artifacts"].items():
        path = output_dir / name
        if (
            path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"output artifact digest mismatch: {name}")
    return {
        "status": "ok",
        "queries": len(query),
        "gallery": len(gallery),
        "accepted": len(candidates),
        "refused": len(query) - len(candidates),
        "weight_bytes": observed_weight_bytes,
        "weight_limit_bytes": LIMIT_BYTES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/score_optimized")
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--query-csv", type=Path, default=Path("dataset/test_query.csv")
    )
    parser.add_argument(
        "--gallery-csv", type=Path, default=Path("dataset/test_gallery.csv")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.output_dir, args.weights_dir, args.query_csv, args.gallery_csv),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
