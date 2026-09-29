"""Aggregate directly comparable validation reports across training seeds.

The command is deliberately read-only: it reads ``validation_metrics.json``
files and prints one JSON report to stdout.  Besides mean/std, it verifies the
paired evaluation contract (same split hashes, query counts, open-set protocol,
and TTA settings) before reporting deltas against a reference run.

Example::

    python -m src.aggregate_validation \
        outputs/experiments/p24k2/final_validation/validation_metrics.json \
        outputs/experiments/p24k2_seed2027/final_validation/validation_metrics.json \
        outputs/experiments/p24k2_seed3407/final_validation/validation_metrics.json \
        --labels seed42 seed2027 seed3407
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


METRIC_PATHS: tuple[str, ...] = (
    "retrieval.mAP",
    "retrieval.rank1",
    "retrieval.rank5",
    "retrieval.mINP",
    "open_set.f1",
    "open_set.precision",
    "open_set.recall",
    "open_set.tnr",
    "open_set.pr_auc",
)

REQUIRED_PAIRED_PATHS: tuple[str, ...] = (
    "retrieval.num_queries",
    "retrieval.num_valid_queries",
    "retrieval.num_ignored_queries",
    "protocol.num_known_identities",
    "protocol.num_unknown_identities",
    "protocol.num_known_queries",
    "protocol.num_unknown_queries",
    "protocol.seed",
    "protocol.unknown_fraction",
)

# Old epoch-selection reports omit these fields, whereas final validation
# reports contain them.  If a field occurs in any input it must occur in all.
# data_split.seed is intentionally not paired: historically it mirrored the
# optimizer seed even when the already-materialized CSV split stayed fixed.
# The two CSV SHA-256 values are the authoritative split identity.
OPTIONAL_PAIRED_PATHS: tuple[str, ...] = (
    "num_validation_embeddings",
    "data_split.protocol",
    "data_split.requested_val_fraction",
    "data_split.train_csv.sha256",
    "data_split.val_csv.sha256",
    "performance.batch_size",
    "performance.tta_horizontal_flip",
    "performance.device",
)

_MISSING = object()
_NO_DEFAULT = object()


def _field(report: Mapping[str, Any], path: str, default: Any = _NO_DEFAULT) -> Any:
    value: Any = report
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            if default is not _NO_DEFAULT:
                return default
            raise ValueError(f"validation report is missing required field '{path}'")
        value = value[part]
    return value


def _finite_metric(report: Mapping[str, Any], path: str) -> float:
    value = _field(report, path)
    if isinstance(value, bool):
        raise ValueError(f"validation metric '{path}' must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"validation metric '{path}' must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"validation metric '{path}' must be finite")
    return number


def _nested(items: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path, value in items.items():
        cursor = result
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def _same_value(left: Any, right: Any) -> bool:
    # bool and int compare equal in Python; protocol schemas should not accept
    # that accidental equivalence.
    return type(left) is type(right) and left == right


def _paired_fields(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for path in REQUIRED_PAIRED_PATHS:
        values = [_field(report, path) for report in reports]
        if any(not _same_value(values[0], value) for value in values[1:]):
            raise ValueError(
                f"validation reports are not paired: field '{path}' differs: {values}"
            )
        paired[path] = values[0]

    for path in OPTIONAL_PAIRED_PATHS:
        values = [_field(report, path, _MISSING) for report in reports]
        present = [value is not _MISSING for value in values]
        if any(present) and not all(present):
            raise ValueError(
                f"validation reports mix schemas: optional paired field '{path}' "
                "is missing from some inputs"
            )
        if not any(present):
            continue
        if any(not _same_value(values[0], value) for value in values[1:]):
            raise ValueError(
                f"validation reports are not paired: field '{path}' differs: {values}"
            )
        paired[path] = values[0]
    return _nested(paired)


def aggregate_validation_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    reference_index: int = 0,
) -> dict[str, Any]:
    """Validate and aggregate reports produced by the same eval protocol."""

    if len(reports) < 2:
        raise ValueError("at least two validation reports are required")
    count = len(reports)
    if labels is None:
        clean_labels = [f"run_{index}" for index in range(count)]
    else:
        clean_labels = [str(label) for label in labels]
        if len(clean_labels) != count:
            raise ValueError("labels and validation reports must have equal length")
        if any(not label.strip() for label in clean_labels):
            raise ValueError("labels must be non-empty")
        if len(set(clean_labels)) != count:
            raise ValueError("labels must be unique")
    if sources is not None and len(sources) != count:
        raise ValueError("sources and validation reports must have equal length")
    if isinstance(reference_index, bool) or not 0 <= reference_index < count:
        raise ValueError("reference_index is outside the validation report list")

    paired = _paired_fields(reports)
    metric_rows = [
        {path: _finite_metric(report, path) for path in METRIC_PATHS}
        for report in reports
    ]
    reference = metric_rows[reference_index]

    aggregates: dict[str, Any] = {}
    for path in METRIC_PATHS:
        values = [row[path] for row in metric_rows]
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        aggregates[path] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
        }

    run_rows: list[dict[str, Any]] = []
    for index, (label, metrics) in enumerate(zip(clean_labels, metric_rows, strict=True)):
        row: dict[str, Any] = {
            "label": label,
            "metrics": _nested(metrics),
            "delta_vs_reference": _nested(
                {path: metrics[path] - reference[path] for path in METRIC_PATHS}
            ),
        }
        if sources is not None:
            row["source"] = sources[index]
        run_rows.append(row)

    return {
        "schema_version": 1,
        "num_runs": count,
        "reference_label": clean_labels[reference_index],
        "std_definition": "population standard deviation (ddof=0)",
        "paired_fields": paired,
        "aggregate": _nested(aggregates),
        "runs": run_rows,
    }


def aggregate_validation_files(
    paths: Sequence[str | Path],
    *,
    labels: Sequence[str] | None = None,
    reference_index: int = 0,
) -> dict[str, Any]:
    """Load validation JSON files without modifying them and aggregate them."""

    if len(paths) < 2:
        raise ValueError("at least two validation report paths are required")
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("validation report paths must be unique")
    reports: list[Mapping[str, Any]] = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"validation report does not exist: {path}")
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        if not isinstance(report, Mapping):
            raise ValueError(f"validation report root must be a mapping: {path}")
        reports.append(report)
    return aggregate_validation_reports(
        reports,
        labels=labels,
        sources=[str(path) for path in resolved],
        reference_index=reference_index,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate paired validation JSON reports across training seeds."
    )
    parser.add_argument("reports", nargs="+", help="validation_metrics.json files")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional unique labels in the same order as the report paths.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Zero-based run used for per-run metric deltas (default: 0).",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = aggregate_validation_files(
        args.reports,
        labels=args.labels,
        reference_index=args.reference_index,
    )
    print(json.dumps(report, indent=args.indent, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())


__all__ = [
    "METRIC_PATHS",
    "OPTIONAL_PAIRED_PATHS",
    "REQUIRED_PAIRED_PATHS",
    "aggregate_validation_files",
    "aggregate_validation_reports",
    "build_parser",
    "main",
]
