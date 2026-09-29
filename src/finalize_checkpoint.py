"""Attach a robust gallery-conditioned refusal calibration to a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PureWindowsPath
from collections.abc import Mapping
from typing import Any

from .config import normalize_postprocess_config
from .engine import (
    DBA_QE_SCORE_DOMAIN,
    load_inference_checkpoint,
    save_inference_checkpoint,
)
from .utils import sha256_file
from .validation import calibrated_confidence
from .refusal import build_adaptive_refusal_metadata
from .score_calibration import (
    ADAPTIVE_REFUSAL_DENSITY_K,
    ADAPTIVE_REFUSAL_FEATURE_NAMES,
    ADAPTIVE_REFUSAL_FEATURE_VARIANT,
    CandidateCorrectnessModel,
)


EXPECTED_METHOD = "gallery_conditioned_dba_qe_raw_cosine"
EXPECTED_REPORT_SCHEMA_VERSION = 1


def _portable_path(path: Path) -> str:
    """Serialize repository artifacts without embedding a workstation path."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        # The SHA-256 below is authoritative; for an external input retain only
        # its filename instead of leaking user/home directory components.
        return resolved.name


def _portable_metadata(value: Any) -> Any:
    """Recursively remove workstation-specific absolute paths.

    Checkpoint provenance is published with the release. Hashes are the
    authority for referenced artifacts, so absolute host paths add no useful
    reproducibility and can disclose a username. Repository-local paths are
    made relative; external paths retain only their basename. Non-path text is
    left unchanged.
    """

    if isinstance(value, Mapping):
        return {str(key): _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_portable_metadata(item) for item in value)
    if not isinstance(value, str):
        return value

    windows_path = PureWindowsPath(value)
    native_path = Path(value).expanduser()
    if windows_path.is_absolute():
        try:
            return native_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except (OSError, ValueError):
            return windows_path.name
    if native_path.is_absolute():
        try:
            return native_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except (OSError, ValueError):
            return native_path.name
    return value


def _report_postprocess(report: Mapping[str, Any]) -> dict[str, bool | int | float]:
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("calibration report configuration must be a mapping")
    dba = configuration.get("dba")
    qe = configuration.get("query_expansion")
    if not isinstance(dba, Mapping) or not isinstance(qe, Mapping):
        raise ValueError(
            "calibration report must contain configuration.dba and "
            "configuration.query_expansion mappings"
        )
    try:
        raw = {
            "enabled": True,
            "dba_top_k": dba["top_k"],
            "dba_alpha": dba["alpha"],
            "qe_top_k": qe["top_k"],
            "qe_alpha": qe["alpha"],
        }
    except KeyError as exc:
        raise ValueError(f"calibration report postprocess field is missing: {exc}") from exc
    return normalize_postprocess_config(raw)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _adaptive_report_payload(
    report: Mapping[str, Any],
    *,
    operating_point: str,
    postprocess: Mapping[str, Any],
    report_sha256: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a locked adaptive experiment and build deploy metadata."""

    if not isinstance(operating_point, str) or not operating_point.strip():
        raise ValueError("adaptive operating point must be a non-empty name")
    if operating_point != operating_point.strip():
        raise ValueError("adaptive operating point must not contain outer whitespace")

    configuration = _mapping(report.get("configuration"), "report configuration")
    if configuration.get("feature_density_k") != ADAPTIVE_REFUSAL_DENSITY_K:
        raise ValueError(
            "adaptive report feature_density_k must equal the production value "
            f"{ADAPTIVE_REFUSAL_DENSITY_K}"
        )
    adaptive = _mapping(
        report.get("optional_feature_correctness_model"),
        "optional_feature_correctness_model",
    )
    if adaptive.get("diagnostic_only") is not True:
        raise ValueError("adaptive report must identify the model as diagnostic_only")
    if adaptive.get("does_not_replace_raw_cosine_threshold") is not True:
        raise ValueError("adaptive report scalar/adaptive separation is missing")
    if tuple(adaptive.get("feature_names", ())) != ADAPTIVE_REFUSAL_FEATURE_NAMES:
        raise ValueError("adaptive report feature_names do not match production schema")

    feature_search = _mapping(
        adaptive.get("tune_only_feature_search"),
        "adaptive tune_only_feature_search",
    )
    if feature_search.get("selected_variant") != ADAPTIVE_REFUSAL_FEATURE_VARIANT:
        raise ValueError(
            "adaptive report selected_variant is not top1_margins_local_density"
        )
    if (
        tuple(feature_search.get("selected_feature_names", ()))
        != ADAPTIVE_REFUSAL_FEATURE_NAMES
    ):
        raise ValueError("adaptive selected_feature_names do not match production schema")
    tune_points = _mapping(
        feature_search.get("tune_operating_points"),
        "adaptive tune_operating_points",
    )
    if operating_point not in tune_points:
        raise ValueError(f"unknown adaptive operating point: {operating_point!r}")
    tune_point = _mapping(
        tune_points[operating_point],
        f"adaptive tune operating point {operating_point!r}",
    )
    tune_metrics = _mapping(
        tune_point.get("metrics"),
        f"adaptive tune operating point {operating_point!r}.metrics",
    )
    try:
        probability_threshold = float(tune_metrics["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("adaptive tune threshold is missing or non-numeric") from exc
    if not math.isfinite(probability_threshold) or not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("adaptive tune threshold must be within [0, 1]")

    locked = _mapping(
        adaptive.get("locked_confirmation"), "adaptive locked_confirmation"
    )
    required_guarantees = (
        "locked_before_evaluation",
        "thresholds_derived_from_tune_oof_only",
    )
    if any(locked.get(name) is not True for name in required_guarantees):
        raise ValueError("adaptive report does not contain a locked confirmation")
    forbidden_usage = (
        "used_for_feature_selection",
        "used_for_model_fit",
        "used_for_threshold_fit",
    )
    if any(locked.get(name) is not False for name in forbidden_usage):
        raise ValueError("adaptive confirmation contaminated model/threshold selection")

    tune_seeds = configuration.get("seeds")
    confirmation_seeds = locked.get("seeds")
    if not isinstance(tune_seeds, list) or not isinstance(confirmation_seeds, list):
        raise ValueError("adaptive tune and confirmation seeds must be lists")
    if not tune_seeds or not confirmation_seeds:
        raise ValueError("adaptive tune and confirmation seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in tune_seeds + confirmation_seeds):
        raise ValueError("adaptive tune and confirmation seeds must be integers")
    if len(set(tune_seeds)) != len(tune_seeds) or len(set(confirmation_seeds)) != len(confirmation_seeds):
        raise ValueError("adaptive tune and confirmation seeds must be unique")
    if set(tune_seeds).intersection(confirmation_seeds):
        raise ValueError("adaptive tune and confirmation seeds must be disjoint")

    confirmation_points = _mapping(
        locked.get("adaptive_operating_points"),
        "adaptive locked_confirmation.adaptive_operating_points",
    )
    if operating_point not in confirmation_points:
        raise ValueError(
            f"locked confirmation is missing operating point {operating_point!r}"
        )
    confirmation_point = _mapping(
        confirmation_points[operating_point],
        f"adaptive confirmation operating point {operating_point!r}",
    )
    confirmation_tune_metrics = _mapping(
        confirmation_point.get("tune_metrics"),
        "adaptive confirmation tune_metrics",
    )
    confirmation_metrics = _mapping(
        confirmation_point.get("confirmation_metrics"),
        "adaptive confirmation_metrics",
    )
    pooled_confirmation = _mapping(
        confirmation_metrics.get("pooled_metrics"),
        "adaptive confirmation pooled_metrics",
    )
    for name, value in (
        ("confirmation tune", confirmation_tune_metrics.get("threshold")),
        ("confirmation pooled", pooled_confirmation.get("threshold")),
        ("confirmation wrapper", confirmation_metrics.get("threshold")),
    ):
        try:
            candidate = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"adaptive {name} threshold is invalid") from exc
        if not math.isclose(
            candidate, probability_threshold, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"adaptive {name} threshold does not match tune OOF threshold"
            )

    model_payload = _mapping(adaptive.get("model"), "adaptive model")
    model = CandidateCorrectnessModel.from_dict(model_payload)
    if model.feature_names != ADAPTIVE_REFUSAL_FEATURE_NAMES:
        raise ValueError("adaptive model feature_names do not match production schema")

    metadata = build_adaptive_refusal_metadata(
        model,
        probability_threshold=probability_threshold,
        operating_point=operating_point,
        score_domain=str(report["threshold_domain"]),
        postprocess=postprocess,
        calibration_report_sha256=report_sha256,
        calibration_report_schema_version=int(report["schema_version"]),
        calibration_report_method=str(report["method"]),
        calibrated_checkpoint_sha256=checkpoint_sha256,
    )
    metrics = {
        "operating_point": operating_point,
        "probability_threshold": probability_threshold,
        "feature_variant": ADAPTIVE_REFUSAL_FEATURE_VARIANT,
        "feature_names": list(ADAPTIVE_REFUSAL_FEATURE_NAMES),
        "density_k": ADAPTIVE_REFUSAL_DENSITY_K,
        "tune_metrics": dict(tune_metrics),
        "confirmation_metrics": dict(confirmation_metrics),
        "confirmation_seeds": list(confirmation_seeds),
    }
    return metadata, metrics


def finalize_checkpoint(
    checkpoint_path: str | Path,
    calibration_report_path: str | Path,
    output_path: str | Path,
    *,
    adaptive_operating_point: str | None = None,
) -> dict[str, Any]:
    """Write an offline checkpoint with threshold/confidence from a robust report."""

    checkpoint_path = Path(checkpoint_path)
    report_path = Path(calibration_report_path)
    output_path = Path(output_path)
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema_version") != EXPECTED_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported calibration schema: {report.get('schema_version')!r}"
        )
    if report.get("method") != EXPECTED_METHOD:
        raise ValueError(f"unsupported calibration method: {report.get('method')!r}")
    if report.get("threshold_domain") != DBA_QE_SCORE_DOMAIN:
        raise ValueError(
            "calibration threshold_domain does not match the DBA/QE search domain"
        )
    postprocess = _report_postprocess(report)
    threshold = float(report["raw_cosine_threshold"])
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("raw_cosine_threshold must be finite and within [-1, 1]")
    calibration = report.get("confidence_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("calibration report is missing confidence_calibration")
    if calibration.get("type") != "platt":
        raise ValueError("confidence_calibration.type must be 'platt'")
    for key in ("slope", "intercept"):
        value = float(calibration[key])
        if not math.isfinite(value):
            raise ValueError(f"confidence_calibration.{key} must be finite")

    confidence_threshold = float(report["confidence_threshold"])
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be finite and within [0, 1]")
    expected_confidence = float(calibrated_confidence(threshold, calibration))
    if not math.isclose(
        confidence_threshold, expected_confidence, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "confidence_threshold does not equal calibrated raw_cosine_threshold"
        )

    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("calibration report inputs must be a mapping")
    checkpoint_digest = inputs.get("checkpoint_sha256")
    actual_checkpoint_digest = sha256_file(checkpoint_path)
    if checkpoint_digest != actual_checkpoint_digest:
        raise ValueError(
            "calibration report checkpoint_sha256 does not match the source checkpoint"
        )

    adaptive_metadata: dict[str, Any] | None = None
    adaptive_metrics: dict[str, Any] | None = None
    if adaptive_operating_point is not None:
        adaptive_metadata, adaptive_metrics = _adaptive_report_payload(
            report,
            operating_point=adaptive_operating_point,
            postprocess=postprocess,
            report_sha256=sha256_file(report_path),
            checkpoint_sha256=actual_checkpoint_digest,
        )

    model, checkpoint = load_inference_checkpoint(checkpoint_path, device="cpu")
    preprocessing = dict(checkpoint["preprocessing"])
    source = _portable_metadata(dict(checkpoint.get("source", {})))
    source.update(
        {
            "robust_open_set_calibration": _portable_path(report_path),
            "robust_open_set_calibration_sha256": sha256_file(report_path),
            "refusal_score_domain": report.get("threshold_domain"),
        }
    )
    metrics = _portable_metadata(dict(checkpoint.get("metrics", {})))
    metrics["robust_open_set"] = {
        "method": report["method"],
        "configuration": report.get("configuration", {}),
        "raw_cosine_threshold": threshold,
        "confidence_threshold": confidence_threshold,
        "pooled_micro_metrics": report.get("pooled_micro_metrics", {}),
        "fixed_threshold_mean_std": report.get("fixed_threshold_mean_std", {}),
        "inputs": _portable_metadata(report.get("inputs", {})),
    }
    if adaptive_metrics is not None:
        metrics["adaptive_open_set"] = adaptive_metrics
        source.update(
            {
                "adaptive_refusal_operating_point": adaptive_operating_point,
                "adaptive_refusal_report_sha256": sha256_file(report_path),
            }
        )
    save_inference_checkpoint(
        output_path,
        model,
        input_size=preprocessing["input_size"],
        bbox_padding=float(preprocessing["bbox_padding"]),
        resize_mode=str(preprocessing.get("resize_mode", "direct")),
        refusal_threshold=threshold,
        tta_horizontal_flip=bool(preprocessing.get("tta_horizontal_flip", False)),
        postprocess=postprocess,
        calibration=calibration,
        adaptive_refusal=adaptive_metadata,
        metrics=metrics,
        source=source,
    )
    result = {
        "output": str(output_path.resolve()),
        "threshold": threshold,
        "postprocess": postprocess,
        "calibration": calibration,
    }
    if adaptive_metadata is not None:
        result["adaptive_refusal"] = adaptive_metadata
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--adaptive-operating-point",
        default=None,
        help=(
            "Promote the named locked adaptive operating point from the robust "
            "report (for example: tnr_at_least_scalar_tune)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = finalize_checkpoint(
        args.checkpoint,
        args.calibration_report,
        args.output_checkpoint,
        adaptive_operating_point=args.adaptive_operating_point,
    )
    suffix = ""
    if "adaptive_refusal" in result:
        adaptive = result["adaptive_refusal"]
        suffix = (
            f" adaptive={adaptive['operating_point']} "
            f"probability_threshold={adaptive['probability_threshold']:.8f}"
        )
    print(
        f"saved={result['output']} threshold={result['threshold']:.8f} "
        f"calibration={result['calibration'].get('type', 'unknown')}{suffix}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_METHOD",
    "EXPECTED_REPORT_SCHEMA_VERSION",
    "finalize_checkpoint",
]
