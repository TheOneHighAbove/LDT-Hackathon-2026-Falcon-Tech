from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from prototype.desktop_app import (
    default_dataset_directory,
    default_output_directory,
)
from prototype.server import ExplorerData


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    dataset.mkdir()
    output.mkdir()
    images = dataset / "images"
    images.mkdir()
    header = ["image_id", "x", "y", "w", "h"]
    _write_csv(dataset / "test_query.csv", [header, ["q1", "0", "0", "10", "10"]])
    gallery_rows = [header] + [
        [f"g{index}", "0", "0", "10", "10"] for index in range(10)
    ]
    _write_csv(dataset / "test_gallery.csv", gallery_rows)
    _write_csv(
        output / "submission.csv",
        [["q1", *[f"g{index}" for index in range(10)]]],
    )
    _write_csv(
        output / "candidates.csv",
        [["query_id", "gallery_id", "confidence"], ["q1", "g0", "0.9"]],
    )
    (output / "inference_manifest.json").write_text(
        json.dumps(
            {
                "profile": "speed",
                "refusal": {"accepted": 1, "refused": 0},
            }
        ),
        encoding="utf-8",
    )
    for image_id in ["q1", *[f"g{index}" for index in range(10)]]:
        Image.new("RGB", (12, 12), (40, 80, 120)).save(images / f"{image_id}.jpg")
    return dataset, output


def test_explorer_loads_verified_release_artifacts(tmp_path: Path) -> None:
    dataset, output = _fixture(tmp_path)
    explorer = ExplorerData.load(dataset, output)

    assert explorer.health() == {
        "status": "ok",
        "profile": "speed",
        "queries": 1,
        "accepted": 1,
        "refused": 0,
        "artifact_mode": "frozen_batch_results",
    }
    assert explorer.result("q1")["top10"][0]["gallery_id"] == "g0"
    assert explorer.cropped_jpeg("q1").startswith(b"\xff\xd8")


def test_explorer_rejects_candidate_that_disagrees_with_top1(tmp_path: Path) -> None:
    dataset, output = _fixture(tmp_path)
    _write_csv(
        output / "candidates.csv",
        [["query_id", "gallery_id", "confidence"], ["q1", "g1", "0.9"]],
    )

    with pytest.raises(ValueError, match="does not match submission top-1"):
        ExplorerData.load(dataset, output)


def test_desktop_defaults_honor_external_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "external-dataset"
    output = tmp_path / "external-results"
    monkeypatch.setenv("DATASET_DIR", str(dataset))
    monkeypatch.setenv("OUTPUT_DIR", str(output))

    assert default_dataset_directory() == dataset
    assert default_output_directory() == output
