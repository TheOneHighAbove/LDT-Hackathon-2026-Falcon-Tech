from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from src.aggregate_validation import (
    aggregate_validation_files,
    aggregate_validation_reports,
    main,
)


def _report(offset: float = 0.0) -> dict:
    return {
        "retrieval": {
            "mAP": 0.40 + offset,
            "rank1": 0.50 + offset,
            "rank5": 0.70 + offset,
            "mINP": 0.30 + offset,
            "num_queries": 100,
            "num_valid_queries": 100,
            "num_ignored_queries": 0,
        },
        "open_set": {
            "f1": 0.60 + offset,
            "precision": 0.61 + offset,
            "recall": 0.59 + offset,
            "tnr": 0.80 + offset,
            "pr_auc": 0.75 + offset,
        },
        "protocol": {
            "num_known_identities": 20,
            "num_unknown_identities": 5,
            "num_known_queries": 80,
            "num_unknown_queries": 20,
            "seed": 1337,
            "unknown_fraction": 0.25,
        },
        "num_validation_embeddings": 100,
        "data_split": {
            "protocol": "identity_disjoint_holdout",
            "seed": 42,
            "requested_val_fraction": 0.2,
            "train_csv": {"sha256": "train-hash"},
            "val_csv": {"sha256": "val-hash"},
        },
        "performance": {
            "batch_size": 1.0,
            "tta_horizontal_flip": True,
            "device": "cuda",
        },
    }


def test_aggregate_reports_has_population_stats_and_reference_deltas() -> None:
    first = _report(0.0)
    second = _report(0.02)
    # The historical provenance seed may follow the optimizer seed.  Matching
    # CSV hashes, not this metadata value, establish the paired split.
    second["data_split"]["seed"] = 2027

    result = aggregate_validation_reports(
        [first, second], labels=["seed42", "seed2027"]
    )

    assert result["num_runs"] == 2
    assert result["reference_label"] == "seed42"
    assert result["paired_fields"]["data_split"]["val_csv"]["sha256"] == "val-hash"
    assert result["aggregate"]["retrieval"]["mAP"]["mean"] == pytest.approx(0.41)
    assert result["aggregate"]["retrieval"]["mAP"]["std"] == pytest.approx(0.01)
    assert result["runs"][1]["delta_vs_reference"]["retrieval"]["mAP"] == pytest.approx(0.02)


def test_aggregate_rejects_unpaired_split_or_protocol() -> None:
    mismatched_split = _report(0.01)
    mismatched_split["data_split"]["val_csv"]["sha256"] = "different"
    with pytest.raises(ValueError, match=r"data_split\.val_csv\.sha256"):
        aggregate_validation_reports([_report(), mismatched_split])

    mismatched_protocol = _report(0.01)
    mismatched_protocol["retrieval"]["num_queries"] = 99
    with pytest.raises(ValueError, match=r"retrieval\.num_queries"):
        aggregate_validation_reports([_report(), mismatched_protocol])


def test_aggregate_rejects_mixed_schema_and_nonfinite_metrics() -> None:
    missing_optional = _report(0.01)
    missing_optional.pop("performance")
    with pytest.raises(ValueError, match="mix schemas"):
        aggregate_validation_reports([_report(), missing_optional])

    nonfinite = _report(0.01)
    nonfinite["open_set"]["f1"] = math.nan
    with pytest.raises(ValueError, match="must be finite"):
        aggregate_validation_reports([_report(), nonfinite])


def test_file_loader_and_cli_print_json_without_mutating_inputs(tmp_path, capsys) -> None:
    first_path = tmp_path / "seed42.json"
    second_path = tmp_path / "seed2027.json"
    first_text = json.dumps(_report())
    second_text = json.dumps(_report(0.01))
    first_path.write_text(first_text, encoding="utf-8")
    second_path.write_text(second_text, encoding="utf-8")

    loaded = aggregate_validation_files(
        [first_path, second_path], labels=["seed42", "seed2027"]
    )
    assert loaded["runs"][0]["source"] == str(first_path.resolve())

    assert main(
        [
            str(first_path),
            str(second_path),
            "--labels",
            "seed42",
            "seed2027",
            "--indent",
            "0",
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["num_runs"] == 2
    assert first_path.read_text(encoding="utf-8") == first_text
    assert second_path.read_text(encoding="utf-8") == second_text


def test_aggregate_validates_cardinality_labels_and_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        aggregate_validation_reports([_report()])
    with pytest.raises(ValueError, match="equal length"):
        aggregate_validation_reports([_report(), _report()], labels=["one"])
    with pytest.raises(ValueError, match="unique"):
        aggregate_validation_reports(
            [_report(), _report()], labels=["same", "same"]
        )

    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    with pytest.raises(ValueError, match="paths must be unique"):
        aggregate_validation_files([path, path])
