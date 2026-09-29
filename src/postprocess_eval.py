"""Leakage-safe repeated evaluation of retrieval post-processing.

The input embeddings must be row-aligned with a validation annotation CSV.
For every seed, exactly one image per vehicle identity is sampled as a query;
all remaining images form the gallery.  The sampled query is guaranteed to
have a cross-camera positive in that gallery.  DBA/QE configurations are
selected on ``tune_seeds`` and the selected configuration is then evaluated,
without re-selection, on disjoint ``confirmation_seeds``.

Example::

    python -m src.postprocess_eval \
        --embeddings outputs/experiments/soft_polish/val_embeddings.npy \
        --annotations splits/val.csv \
        --checkpoint weights/experiments/soft_polish/best.pt \
        --output outputs/experiments/soft_polish/postprocess_eval.json \
        --dba-top-k 0 3 5 --dba-alpha 2 \
        --qe-top-k 0 2 --qe-alpha 1 1.5

``0`` disables the corresponding DBA or QE stage.  Disabled stages are
canonicalized, so their alpha grid does not create duplicate methods.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import evaluate_retrieval
from .reranking import database_side_augmentation, query_expansion
from .utils import configure_logging, save_json, sha256_file


SCHEMA_VERSION = 1
DEFAULT_TUNE_SEEDS: tuple[int, ...] = (1337, 2027, 3407, 4517, 7919)
DEFAULT_CONFIRMATION_SEEDS: tuple[int, ...] = (101, 211, 307, 401, 503)
METRIC_NAMES: tuple[str, ...] = ("mAP", "rank1", "rank5", "mINP")


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _ordered_values_sha256(values: Iterable[Any]) -> str:
    """Hash an ordered heterogeneous ID sequence without delimiter ambiguity."""

    digest = hashlib.sha256()
    for value in values:
        scalar = _python_scalar(value)
        tagged = {
            "type": type(scalar).__name__,
            "value": str(scalar),
        }
        encoded = _canonical_json(tagged).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_seed_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} must contain at least one seed")
    cleaned: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must contain integers")
        cleaned.append(int(value))
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{name} must not contain duplicate seeds")
    return tuple(cleaned)


def _validate_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(annotations, pd.DataFrame):
        raise ValueError("annotations must be a pandas DataFrame")
    required = ("image_id", "vehicle_id", "camera_id")
    missing = sorted(set(required).difference(annotations.columns))
    if missing:
        raise ValueError(f"annotations are missing columns: {missing}")
    frame = annotations.reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError("annotations must not be empty")
    if frame.loc[:, required].isna().any().any():
        raise ValueError("annotations contain missing IDs")
    if frame["image_id"].duplicated().any():
        raise ValueError("image_id must be unique")
    camera_counts = frame.groupby("vehicle_id", sort=False)["camera_id"].nunique()
    invalid = camera_counts[camera_counts < 2]
    if len(invalid):
        examples = [_python_scalar(value) for value in invalid.index[:5].tolist()]
        raise ValueError(
            "every vehicle_id must have images from at least two cameras so each "
            f"query has a cross-camera gallery positive; invalid IDs: {examples!r}"
        )
    if frame["vehicle_id"].nunique() < 2:
        raise ValueError("at least two vehicle identities are required")
    return frame


@dataclass(frozen=True)
class RetrievalProtocol:
    """One closed-set, one-query-per-identity retrieval protocol."""

    seed: int
    query_indices: np.ndarray
    gallery_indices: np.ndarray

    def __post_init__(self) -> None:
        query = np.asarray(self.query_indices, dtype=np.int64).copy()
        gallery = np.asarray(self.gallery_indices, dtype=np.int64).copy()
        if query.ndim != 1 or gallery.ndim != 1:
            raise ValueError("protocol indices must be one-dimensional")
        if not len(query) or not len(gallery):
            raise ValueError("query and gallery must both be non-empty")
        if len(np.unique(query)) != len(query):
            raise ValueError("query indices contain duplicates")
        if len(np.unique(gallery)) != len(gallery):
            raise ValueError("gallery indices contain duplicates")
        if np.intersect1d(query, gallery).size:
            raise ValueError("query and gallery indices overlap")
        query.setflags(write=False)
        gallery.setflags(write=False)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "query_indices", query)
        object.__setattr__(self, "gallery_indices", gallery)


def build_identity_stratified_protocol(
    annotations: pd.DataFrame,
    *,
    seed: int,
) -> RetrievalProtocol:
    """Sample one query per identity and retain every other row as gallery."""

    frame = _validate_annotations(annotations)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")

    pids = frame["vehicle_id"].to_numpy()
    identities = frame["vehicle_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(int(seed))
    query_indices: list[int] = []
    for vehicle_id in identities:
        identity_indices = np.flatnonzero(pids == vehicle_id)
        query_indices.append(
            int(identity_indices[int(rng.integers(0, len(identity_indices)))])
        )

    query = np.asarray(query_indices, dtype=np.int64)
    is_gallery = np.ones(len(frame), dtype=bool)
    is_gallery[query] = False
    gallery = np.flatnonzero(is_gallery)
    protocol = RetrievalProtocol(
        seed=int(seed), query_indices=query, gallery_indices=gallery
    )
    _protocol_summary(protocol, frame)  # Validate every guarantee immediately.
    return protocol


def _protocol_summary(
    protocol: RetrievalProtocol,
    annotations: pd.DataFrame,
) -> dict[str, Any]:
    frame = annotations.reset_index(drop=True)
    query = frame.iloc[protocol.query_indices]
    gallery = frame.iloc[protocol.gallery_indices]
    query_images = query["image_id"].tolist()
    gallery_images = gallery["image_id"].tolist()
    if set(query_images).intersection(gallery_images):
        raise RuntimeError("query/gallery image leakage detected")
    if len(query) != frame["vehicle_id"].nunique():
        raise RuntimeError("protocol does not contain exactly one query per identity")
    if query["vehicle_id"].duplicated().any():
        raise RuntimeError("an identity was sampled as query more than once")

    cross_camera_positive_count = 0
    for row in query.itertuples(index=False):
        positive = (gallery["vehicle_id"] == row.vehicle_id) & (
            gallery["camera_id"] != row.camera_id
        )
        if not bool(positive.any()):
            raise RuntimeError(
                f"query image_id={row.image_id!r} has no cross-camera gallery positive"
            )
        cross_camera_positive_count += 1

    identity_order = query["vehicle_id"].tolist()
    hash_payload = {
        "protocol": "one_query_per_identity_all_remainder_gallery_v1",
        "seed": protocol.seed,
        "query_image_ids_sha256": _ordered_values_sha256(query_images),
        "gallery_image_ids_sha256": _ordered_values_sha256(gallery_images),
        "query_vehicle_ids_sha256": _ordered_values_sha256(identity_order),
    }
    return {
        "seed": protocol.seed,
        "num_identities": int(frame["vehicle_id"].nunique()),
        "num_queries": int(len(query)),
        "num_gallery": int(len(gallery)),
        "queries_with_cross_camera_positive": cross_camera_positive_count,
        "query_gallery_image_overlap": 0,
        **{key: value for key, value in hash_payload.items() if key.endswith("sha256")},
        "protocol_sha256": _payload_sha256(hash_payload),
    }


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


@dataclass(frozen=True)
class PostprocessSpec:
    """Canonical DBA/QE configuration; ``top_k=0`` disables a stage."""

    dba_top_k: int = 0
    dba_alpha: float | None = None
    qe_top_k: int = 0
    qe_alpha: float | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.dba_top_k, "dba_top_k"),
            (self.qe_top_k, "qe_top_k"),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{name} must be a non-negative integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "dba_top_k", int(self.dba_top_k))
        object.__setattr__(self, "qe_top_k", int(self.qe_top_k))

        for top_k, alpha, name in (
            (self.dba_top_k, self.dba_alpha, "dba_alpha"),
            (self.qe_top_k, self.qe_alpha, "qe_alpha"),
        ):
            if top_k == 0:
                object.__setattr__(self, name, None)
                continue
            if alpha is None:
                raise ValueError(f"{name} is required when its stage is enabled")
            cleaned = float(alpha)
            if not math.isfinite(cleaned) or cleaned < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, cleaned)

    @property
    def method_id(self) -> str:
        if self.dba_top_k == 0 and self.qe_top_k == 0:
            return "raw"
        dba = (
            "dba_off"
            if self.dba_top_k == 0
            else f"dba_k{self.dba_top_k}_a{_format_number(self.dba_alpha or 0.0)}"
        )
        qe = (
            "qe_off"
            if self.qe_top_k == 0
            else f"qe_k{self.qe_top_k}_a{_format_number(self.qe_alpha or 0.0)}"
        )
        return f"{dba}__{qe}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "dba": {
                "enabled": self.dba_top_k > 0,
                "top_k": self.dba_top_k,
                "alpha": self.dba_alpha,
            },
            "query_expansion": {
                "enabled": self.qe_top_k > 0,
                "top_k": self.qe_top_k,
                "alpha": self.qe_alpha,
            },
            "score_domain": (
                "cosine(processed_query, processed_gallery)"
                if self.dba_top_k or self.qe_top_k
                else "cosine(raw_query, raw_gallery)"
            ),
        }

    def complexity_key(self) -> tuple[int, int, int, int, float, float, str]:
        return (
            int(self.dba_top_k > 0) + int(self.qe_top_k > 0),
            self.dba_top_k + self.qe_top_k,
            self.dba_top_k,
            self.qe_top_k,
            float(self.dba_alpha or 0.0),
            float(self.qe_alpha or 0.0),
            self.method_id,
        )


def build_postprocess_grid(
    *,
    dba_top_k: Sequence[int],
    dba_alpha: Sequence[float],
    qe_top_k: Sequence[int],
    qe_alpha: Sequence[float],
) -> tuple[PostprocessSpec, ...]:
    """Build a de-duplicated Cartesian grid that always contains raw cosine."""

    def clean_top_k(values: Sequence[int], name: str) -> tuple[int, ...]:
        if not values:
            raise ValueError(f"{name} must not be empty")
        cleaned: list[int] = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{name} must contain non-negative integers")
            if int(value) < 0:
                raise ValueError(f"{name} must contain non-negative integers")
            if int(value) not in cleaned:
                cleaned.append(int(value))
        return tuple(cleaned)

    def clean_alpha(values: Sequence[float], name: str) -> tuple[float, ...]:
        if not values:
            raise ValueError(f"{name} must not be empty")
        cleaned: list[float] = []
        for value in values:
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must contain finite non-negative values")
            if number not in cleaned:
                cleaned.append(number)
        return tuple(cleaned)

    dba_k_values = clean_top_k(dba_top_k, "dba_top_k")
    qe_k_values = clean_top_k(qe_top_k, "qe_top_k")
    dba_alpha_values = clean_alpha(dba_alpha, "dba_alpha")
    qe_alpha_values = clean_alpha(qe_alpha, "qe_alpha")

    # Raw is mandatory even if callers only supplied enabled stage sizes.
    specs: dict[str, PostprocessSpec] = {"raw": PostprocessSpec()}
    dba_choices = [
        PostprocessSpec(dba_top_k=k, dba_alpha=alpha)
        for k in dba_k_values
        for alpha in ((None,) if k == 0 else dba_alpha_values)
    ]
    qe_choices = [
        PostprocessSpec(qe_top_k=k, qe_alpha=alpha)
        for k in qe_k_values
        for alpha in ((None,) if k == 0 else qe_alpha_values)
    ]
    for dba in dba_choices:
        for qe in qe_choices:
            spec = PostprocessSpec(
                dba_top_k=dba.dba_top_k,
                dba_alpha=dba.dba_alpha,
                qe_top_k=qe.qe_top_k,
                qe_alpha=qe.qe_alpha,
            )
            specs[spec.method_id] = spec
    raw = specs.pop("raw")
    ordered = sorted(specs.values(), key=lambda spec: spec.complexity_key())
    return (raw, *ordered)


def _evaluate_protocol_grid(
    embeddings: np.ndarray,
    annotations: pd.DataFrame,
    protocol: RetrievalProtocol,
    specs: Sequence[PostprocessSpec],
) -> dict[str, dict[str, float | int]]:
    query_indices = protocol.query_indices
    gallery_indices = protocol.gallery_indices
    query = np.ascontiguousarray(embeddings[query_indices], dtype=np.float32)
    gallery = np.ascontiguousarray(embeddings[gallery_indices], dtype=np.float32)
    if not np.isfinite(query).all() or not np.isfinite(gallery).all():
        raise ValueError("embeddings contain NaN or infinite values")

    frame = annotations.reset_index(drop=True)
    query_rows = frame.iloc[query_indices]
    gallery_rows = frame.iloc[gallery_indices]
    gallery_cache: dict[tuple[int, float | None], np.ndarray] = {
        (0, None): gallery
    }
    query_cache: dict[
        tuple[int, float | None, int, float | None], np.ndarray
    ] = {}
    results: dict[str, dict[str, float | int]] = {}

    for spec in specs:
        gallery_key = (spec.dba_top_k, spec.dba_alpha)
        processed_gallery = gallery_cache.get(gallery_key)
        if processed_gallery is None:
            if spec.dba_top_k >= len(gallery):
                raise ValueError(
                    f"{spec.method_id}: DBA top_k must be smaller than gallery size "
                    f"{len(gallery)}"
                )
            processed_gallery = database_side_augmentation(
                gallery,
                top_k=spec.dba_top_k,
                alpha=float(spec.dba_alpha),
            ).astype(np.float32, copy=False)
            gallery_cache[gallery_key] = processed_gallery

        query_key = (*gallery_key, spec.qe_top_k, spec.qe_alpha)
        processed_query = query_cache.get(query_key)
        if processed_query is None:
            if spec.qe_top_k == 0:
                processed_query = query
            else:
                if spec.qe_top_k > len(processed_gallery):
                    raise ValueError(
                        f"{spec.method_id}: QE top_k cannot exceed gallery size "
                        f"{len(processed_gallery)}"
                    )
                processed_query = query_expansion(
                    query,
                    processed_gallery,
                    top_k=spec.qe_top_k,
                    alpha=float(spec.qe_alpha),
                ).astype(np.float32, copy=False)
            query_cache[query_key] = processed_query

        metrics = evaluate_retrieval(
            processed_query,
            processed_gallery,
            query_rows["vehicle_id"].to_numpy(),
            gallery_rows["vehicle_id"].to_numpy(),
            query_camera_ids=query_rows["camera_id"].to_numpy(),
            gallery_camera_ids=gallery_rows["camera_id"].to_numpy(),
            query_image_ids=query_rows["image_id"].to_numpy(),
            gallery_image_ids=gallery_rows["image_id"].to_numpy(),
            same_source=False,
        )
        if int(metrics["num_ignored_queries"]) != 0:
            raise RuntimeError(
                f"{spec.method_id}: a protocol query had no cross-camera positive"
            )
        results[spec.method_id] = {
            key: _python_scalar(value) for key, value in metrics.items()
        }
    return results


def _aggregate_metrics(
    seed_rows: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    if not seed_rows:
        raise ValueError("cannot aggregate an empty metric sequence")
    result: dict[str, Any] = {"num_seeds": len(seed_rows)}
    for metric in METRIC_NAMES:
        values = np.asarray([float(row[metric]) for row in seed_rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    result["num_queries_per_seed"] = int(seed_rows[0]["num_queries"])
    result["num_valid_queries_per_seed"] = int(
        seed_rows[0]["num_valid_queries"]
    )
    return result


def _paired_delta(
    candidate_rows: Sequence[Mapping[str, float | int]],
    raw_rows: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    if len(candidate_rows) != len(raw_rows) or not candidate_rows:
        raise ValueError("paired metric sequences must have the same non-zero length")
    result: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        deltas = np.asarray(
            [
                float(candidate[metric]) - float(raw[metric])
                for candidate, raw in zip(candidate_rows, raw_rows, strict=True)
            ],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(deltas.mean()),
            "std": float(deltas.std(ddof=0)),
            "min": float(deltas.min()),
            "max": float(deltas.max()),
            "positive_seed_count": int(np.sum(deltas > 0.0)),
            "negative_seed_count": int(np.sum(deltas < 0.0)),
            "tie_seed_count": int(np.sum(deltas == 0.0)),
        }
    return result


def _method_report(
    spec: PostprocessSpec,
    seeds: Sequence[int],
    rows: Sequence[Mapping[str, float | int]],
    raw_rows: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    if len(seeds) != len(rows):
        raise ValueError("seeds and metric rows must align")
    return {
        "configuration": spec.to_dict(),
        "aggregate": _aggregate_metrics(rows),
        "paired_delta_vs_raw": _paired_delta(rows, raw_rows),
        "per_seed": [
            {"seed": int(seed), "metrics": dict(metrics)}
            for seed, metrics in zip(seeds, rows, strict=True)
        ],
    }


def _select_spec(
    specs: Sequence[PostprocessSpec],
    aggregate_by_method: Mapping[str, Mapping[str, Any]],
    *,
    selection_metric: str,
) -> PostprocessSpec:
    if selection_metric not in METRIC_NAMES:
        raise ValueError(f"selection_metric must be one of {METRIC_NAMES!r}")

    # Primary metric first, then the other official metrics, then the cheaper
    # method.  This rule is explicit so exact ties never depend on dict order.
    metric_order = (selection_metric,) + tuple(
        metric for metric in METRIC_NAMES if metric != selection_metric
    )

    def key(spec: PostprocessSpec) -> tuple[Any, ...]:
        aggregate = aggregate_by_method[spec.method_id]
        quality = tuple(-float(aggregate[metric]["mean"]) for metric in metric_order)
        return (*quality, *spec.complexity_key())

    return min(specs, key=key)


def evaluate_postprocessing(
    embeddings: np.ndarray,
    annotations: pd.DataFrame,
    *,
    tune_seeds: Sequence[int] = DEFAULT_TUNE_SEEDS,
    confirmation_seeds: Sequence[int] = DEFAULT_CONFIRMATION_SEEDS,
    dba_top_k: Sequence[int] = (0, 3, 5),
    dba_alpha: Sequence[float] = (2.0,),
    qe_top_k: Sequence[int] = (0, 2),
    qe_alpha: Sequence[float] = (1.0, 1.5),
    selection_metric: str = "mAP",
) -> dict[str, Any]:
    """Tune a DBA/QE grid and confirm the winner on untouched seed protocols."""

    frame = _validate_annotations(annotations)
    array = np.asarray(embeddings)
    if array.ndim != 2 or array.shape[0] != len(frame) or array.shape[1] == 0:
        raise ValueError(
            "embeddings must be a non-empty [N, D] matrix row-aligned with annotations"
        )
    try:
        array = np.ascontiguousarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("embeddings must be numeric") from exc
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain NaN or infinite values")
    if np.any(np.linalg.norm(array, axis=1) <= np.finfo(np.float32).eps):
        raise ValueError("embeddings contain a zero-norm row")

    tune = _validate_seed_sequence(tune_seeds, "tune_seeds")
    confirmation = _validate_seed_sequence(
        confirmation_seeds, "confirmation_seeds"
    )
    overlap = sorted(set(tune).intersection(confirmation))
    if overlap:
        raise ValueError(
            "tune_seeds and confirmation_seeds must be disjoint; overlap: "
            f"{overlap!r}"
        )
    if selection_metric not in METRIC_NAMES:
        raise ValueError(f"selection_metric must be one of {METRIC_NAMES!r}")
    specs = build_postprocess_grid(
        dba_top_k=dba_top_k,
        dba_alpha=dba_alpha,
        qe_top_k=qe_top_k,
        qe_alpha=qe_alpha,
    )
    raw_spec = specs[0]
    assert raw_spec.method_id == "raw"

    tune_protocols = [
        build_identity_stratified_protocol(frame, seed=seed) for seed in tune
    ]
    tune_protocol_summaries = [
        _protocol_summary(protocol, frame) for protocol in tune_protocols
    ]
    tune_rows_by_method: dict[str, list[dict[str, float | int]]] = {
        spec.method_id: [] for spec in specs
    }
    for protocol in tune_protocols:
        seed_results = _evaluate_protocol_grid(array, frame, protocol, specs)
        for method_id, metrics in seed_results.items():
            tune_rows_by_method[method_id].append(metrics)

    aggregate_by_method = {
        method_id: _aggregate_metrics(rows)
        for method_id, rows in tune_rows_by_method.items()
    }
    selected = _select_spec(
        specs,
        aggregate_by_method,
        selection_metric=selection_metric,
    )
    tune_candidates = [
        _method_report(
            spec,
            tune,
            tune_rows_by_method[spec.method_id],
            tune_rows_by_method["raw"],
        )
        for spec in specs
    ]

    confirmation_specs = (
        (raw_spec,) if selected.method_id == "raw" else (raw_spec, selected)
    )
    confirmation_protocols = [
        build_identity_stratified_protocol(frame, seed=seed)
        for seed in confirmation
    ]
    confirmation_protocol_summaries = [
        _protocol_summary(protocol, frame) for protocol in confirmation_protocols
    ]
    confirmation_rows: dict[str, list[dict[str, float | int]]] = {
        spec.method_id: [] for spec in confirmation_specs
    }
    for protocol in confirmation_protocols:
        seed_results = _evaluate_protocol_grid(
            array, frame, protocol, confirmation_specs
        )
        for method_id, metrics in seed_results.items():
            confirmation_rows[method_id].append(metrics)
    raw_confirmation = confirmation_rows["raw"]
    selected_confirmation = confirmation_rows[selected.method_id]

    grid_payload = [spec.to_dict() for spec in specs]
    protocol_definition = {
        "name": "one_query_per_identity_all_remainder_gallery_v1",
        "query_sampling": "one uniformly sampled image per vehicle_id",
        "gallery": "all annotation rows not selected as queries",
        "junk_rule": "same-vehicle same-camera gallery rows are excluded",
        "positive_rule": "every query must retain a different-camera same-ID match",
        "selection": (
            f"maximize tune mean {selection_metric}; tie-break by remaining official "
            "metrics, then lower post-processing complexity"
        ),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": "closed_set_vehicle_reid_postprocess_selection",
        "protocol": protocol_definition,
        "configuration": {
            "tune_seeds": list(tune),
            "confirmation_seeds": list(confirmation),
            "selection_metric": selection_metric,
            "num_grid_methods": len(specs),
            "grid": grid_payload,
            "grid_sha256": _payload_sha256(grid_payload),
        },
        "tune": {
            "protocols": tune_protocol_summaries,
            "protocol_set_sha256": _payload_sha256(tune_protocol_summaries),
            "candidates": tune_candidates,
            "selected_method_id": selected.method_id,
        },
        "confirmation": {
            "protocols": confirmation_protocol_summaries,
            "protocol_set_sha256": _payload_sha256(
                confirmation_protocol_summaries
            ),
            "raw": _method_report(
                raw_spec, confirmation, raw_confirmation, raw_confirmation
            ),
            "selected": _method_report(
                selected,
                confirmation,
                selected_confirmation,
                raw_confirmation,
            ),
        },
        "selected_method": selected.to_dict(),
        "protocol_guarantees": {
            "tune_confirmation_seed_disjoint": True,
            "one_query_per_identity": True,
            "query_gallery_image_disjoint": True,
            "every_query_has_cross_camera_positive": True,
            "all_non_query_images_are_in_gallery": True,
            "selection_never_uses_confirmation_metrics": True,
        },
        "limitations": [
            "The CSV row order must match the embeddings row order exactly.",
            "Repeated seeds reuse identities and quantify query-choice sensitivity, not independent-data uncertainty.",
            "DBA and QE are gallery-conditioned; a changed gallery requires a new evaluation.",
            "This closed-set protocol evaluates ranking only and does not calibrate open-set refusal.",
        ],
    }
    # Reject non-standard JSON values before returning to callers.
    _canonical_json(report)
    return report


def evaluate_from_files(
    embeddings_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load aligned files, evaluate, bind provenance, and atomically save JSON."""

    embeddings_path = Path(embeddings_path).expanduser().resolve()
    annotations_path = Path(annotations_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"embeddings file does not exist: {embeddings_path}")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"annotations file does not exist: {annotations_path}")
    embeddings = np.load(embeddings_path, allow_pickle=False)
    annotations = pd.read_csv(annotations_path)
    report = evaluate_postprocessing(embeddings, annotations, **kwargs)

    immutable_inputs: dict[str, Any] = {
        "embeddings_sha256": sha256_file(embeddings_path),
        "annotations_sha256": sha256_file(annotations_path),
        "annotation_image_ids_sha256": _ordered_values_sha256(
            annotations["image_id"].tolist()
        )
        if "image_id" in annotations
        else None,
        "num_rows": int(len(annotations)),
        "embedding_shape": [int(value) for value in embeddings.shape],
        "embedding_dtype": str(embeddings.dtype),
    }
    provenance: dict[str, Any] = {
        "embeddings": {
            "path": str(embeddings_path),
            "sha256": immutable_inputs["embeddings_sha256"],
            "size_bytes": int(embeddings_path.stat().st_size),
        },
        "annotations": {
            "path": str(annotations_path),
            "sha256": immutable_inputs["annotations_sha256"],
            "size_bytes": int(annotations_path.stat().st_size),
        },
        "alignment_contract": "CSV row i corresponds to embeddings row i",
        "annotation_image_ids_sha256": immutable_inputs[
            "annotation_image_ids_sha256"
        ],
        "num_rows": immutable_inputs["num_rows"],
        "embedding_shape": immutable_inputs["embedding_shape"],
        "embedding_dtype": immutable_inputs["embedding_dtype"],
    }
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        checkpoint_sha = sha256_file(checkpoint)
        immutable_inputs["checkpoint_sha256"] = checkpoint_sha
        provenance["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "size_bytes": int(checkpoint.stat().st_size),
        }
    provenance["provenance_sha256"] = _payload_sha256(immutable_inputs)
    report["provenance"] = provenance
    report["report_payload_sha256"] = _payload_sha256(report)

    # Canonicalize nested dict insertion order before using the shared atomic
    # JSON writer.  This makes output bytes deterministic for identical paths
    # and inputs while retaining the project's established write semantics.
    canonical_report = json.loads(_canonical_json(report))
    save_json(canonical_report, output_path)
    return canonical_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("outputs/experiments/soft_polish/val_embeddings.npy"),
    )
    parser.add_argument("--annotations", type=Path, default=Path("splits/val.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/postprocess_eval.json")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose SHA-256 is bound into provenance.",
    )
    parser.add_argument(
        "--tune-seeds", nargs="+", type=int, default=list(DEFAULT_TUNE_SEEDS)
    )
    parser.add_argument(
        "--confirmation-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_CONFIRMATION_SEEDS),
    )
    parser.add_argument("--dba-top-k", nargs="+", type=int, default=[0, 3, 5])
    parser.add_argument("--dba-alpha", nargs="+", type=float, default=[2.0])
    parser.add_argument("--qe-top-k", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--qe-alpha", nargs="+", type=float, default=[1.0, 1.5])
    parser.add_argument(
        "--selection-metric", choices=METRIC_NAMES, default="mAP"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    configure_logging(args.verbose)
    report = evaluate_from_files(
        args.embeddings,
        args.annotations,
        args.output,
        checkpoint_path=args.checkpoint,
        tune_seeds=args.tune_seeds,
        confirmation_seeds=args.confirmation_seeds,
        dba_top_k=args.dba_top_k,
        dba_alpha=args.dba_alpha,
        qe_top_k=args.qe_top_k,
        qe_alpha=args.qe_alpha,
        selection_metric=args.selection_metric,
    )
    raw = report["confirmation"]["raw"]["aggregate"]
    selected = report["confirmation"]["selected"]["aggregate"]
    print(
        "selected={} | confirmation mAP={:.6f} Rank-1={:.6f} "
        "Rank-5={:.6f} mINP={:.6f} | raw mAP={:.6f}".format(
            report["selected_method"]["method_id"],
            selected["mAP"]["mean"],
            selected["rank1"]["mean"],
            selected["rank5"]["mean"],
            selected["mINP"]["mean"],
            raw["mAP"]["mean"],
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CONFIRMATION_SEEDS",
    "DEFAULT_TUNE_SEEDS",
    "PostprocessSpec",
    "RetrievalProtocol",
    "build_identity_stratified_protocol",
    "build_postprocess_grid",
    "evaluate_from_files",
    "evaluate_postprocessing",
]
