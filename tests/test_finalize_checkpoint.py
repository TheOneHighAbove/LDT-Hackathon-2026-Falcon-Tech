from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import pytest

from src.engine import load_inference_checkpoint, save_inference_checkpoint
from src.finalize_checkpoint import _portable_metadata, finalize_checkpoint
from src.model import VehicleReIDModel


def _report(checkpoint_sha256: str = "a" * 64) -> dict:
    confidence_threshold = 1.0 / (1.0 + math.exp(-(8.0 * 0.81 - 5.0)))
    return {
        "schema_version": 1,
        "method": "gallery_conditioned_dba_qe_raw_cosine",
        "raw_cosine_threshold": 0.81,
        "threshold_domain": "cosine(expanded_query, dba_gallery)",
        "confidence_calibration": {
            "type": "platt",
            "slope": 8.0,
            "intercept": -5.0,
        },
        "confidence_threshold": confidence_threshold,
        "configuration": {
            "dba": {"top_k": 3, "alpha": 2.0},
            "query_expansion": {"top_k": 2, "alpha": 1.0},
        },
        "pooled_micro_metrics": {"f1": 0.7},
        "fixed_threshold_mean_std": {"f1": {"mean": 0.7, "std": 0.01}},
        "inputs": {
            "embeddings_sha256": "abc",
            "checkpoint_sha256": checkpoint_sha256,
        },
    }


def test_finalize_checkpoint_replaces_only_calibration_metadata(tmp_path):
    model = VehicleReIDModel("resnet18", 8, pretrained=False)
    source = tmp_path / "source.pt"
    report_path = tmp_path / "calibration.json"
    output = tmp_path / "final.pt"
    save_inference_checkpoint(
        source,
        model,
        input_size=64,
        bbox_padding=0.05,
        refusal_threshold=0.2,
        metrics={"retrieval": {"mAP": 0.4}},
        source={"epoch": 3},
    )
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(_report(source_digest)), encoding="utf-8")

    result = finalize_checkpoint(source, report_path, output)
    _, checkpoint = load_inference_checkpoint(output)

    assert result["threshold"] == pytest.approx(0.81)
    assert checkpoint["refusal"]["similarity_threshold"] == pytest.approx(0.81)
    assert checkpoint["refusal"]["calibration"]["slope"] == 8.0
    assert checkpoint["metrics"]["retrieval"]["mAP"] == 0.4
    assert checkpoint["metrics"]["robust_open_set"]["raw_cosine_threshold"] == 0.81
    assert checkpoint["source"]["epoch"] == 3
    assert "robust_open_set_calibration_sha256" in checkpoint["source"]
    assert checkpoint["search"]["postprocess"]["enabled"] is True
    assert checkpoint["search"]["postprocess"]["qe_top_k"] == 2


def test_finalize_checkpoint_rejects_wrong_report_method(tmp_path):
    report = _report()
    report["method"] = "raw_cosine"
    report_path = tmp_path / "bad.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported calibration method"):
        finalize_checkpoint(tmp_path / "missing.pt", report_path, tmp_path / "out.pt")


def test_portable_metadata_removes_absolute_workstation_paths():
    local = str((Path.cwd() / "configs" / "base.yaml").resolve())
    payload = {
        "config": local,
        "foreign": r"C:\Users\private-user\secret\report.json",
        "score_domain": "cosine(expanded_query, dba_gallery)",
        "nested": [local],
    }

    portable = _portable_metadata(payload)

    assert portable["config"] == "configs/base.yaml"
    assert portable["foreign"] == "report.json"
    assert portable["score_domain"] == payload["score_domain"]
    assert portable["nested"] == ["configs/base.yaml"]
