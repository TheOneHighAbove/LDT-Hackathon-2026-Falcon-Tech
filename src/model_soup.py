"""Create a reproducible model soup from compatible inference checkpoints.

Only floating/complex entries of ``model_state`` are averaged.  Integer and
boolean state entries (most notably BatchNorm ``num_batches_tracked``) are
bookkeeping buffers rather than values with a meaningful arithmetic mean, so
they are copied from an explicitly recorded anchor checkpoint.  Floating
BatchNorm running means/variances *are* averaged: they are part of the eval
function represented by every member and use the same convex coefficients as
the trainable parameters.

A soup changes the embedding function and therefore invalidates every input
refusal threshold, confidence calibration, and retrieval metric.  The output
keeps those values only inside provenance metadata and deliberately marks the
active refusal contract as requiring recalibration.  Run validation and the
robust open-set calibration/finalization pipeline before deployment.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import normalize_input_size, normalize_resize_mode
from .engine import (
    CHECKPOINT_SCHEMA_VERSION,
    RAW_COSINE_SCORE_DOMAIN,
    checkpoint_postprocess,
    search_contract,
)
from .utils import atomic_torch_save, sha256_file


MODEL_SOUP_SCHEMA_VERSION = 1
REFUSAL_RECALIBRATION_THRESHOLD = 1.0
_HEX_DIGITS = frozenset("0123456789abcdef")
_REQUIRED_PREPROCESSING_FIELDS = frozenset(
    {"input_size", "bbox_padding", "tta_horizontal_flip", "mean", "std"}
)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _finite_float(value: Any, name: str) -> float:
    if not _is_number(value):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like metadata without bool/int or list/tuple coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"checkpoint {name} keys must be strings")
    return value


def _validate_architecture(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    architecture = _require_mapping(checkpoint.get("model"), "model")
    backbone = architecture.get("backbone_name")
    dimension = architecture.get("embedding_dim")
    if not isinstance(backbone, str) or not backbone:
        raise ValueError("checkpoint model.backbone_name must be non-empty")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("checkpoint model.embedding_dim must be a positive integer")
    if architecture.get("pretrained", False) is not False:
        raise ValueError("inference checkpoint model.pretrained must be false")
    return deepcopy(dict(architecture))


def _validate_preprocessing(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    preprocessing = _require_mapping(
        checkpoint.get("preprocessing"), "preprocessing"
    )
    missing = sorted(_REQUIRED_PREPROCESSING_FIELDS.difference(preprocessing))
    if missing:
        raise ValueError(f"checkpoint preprocessing is missing fields: {missing}")
    normalize_input_size(preprocessing["input_size"])
    normalize_resize_mode(preprocessing.get("resize_mode", "direct"))
    padding = _finite_float(preprocessing["bbox_padding"], "bbox_padding")
    if padding < 0.0:
        raise ValueError("checkpoint bbox_padding must be non-negative")
    if not isinstance(preprocessing["tta_horizontal_flip"], bool):
        raise TypeError("checkpoint tta_horizontal_flip must be boolean")
    for name in ("mean", "std"):
        values = preprocessing[name]
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"checkpoint preprocessing.{name} must have 3 values")
        parsed = [_finite_float(item, f"preprocessing.{name}") for item in values]
        if name == "std" and any(item <= 0.0 for item in parsed):
            raise ValueError("checkpoint preprocessing.std must be positive")
    return deepcopy(dict(preprocessing))


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_refusal_provenance(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    refusal = _require_mapping(checkpoint.get("refusal"), "refusal")
    expected_keys = {"similarity_threshold", "calibration"}
    if set(refusal) != expected_keys:
        raise ValueError(
            "checkpoint refusal must contain exactly similarity_threshold and "
            "calibration"
        )
    threshold = _finite_float(
        refusal["similarity_threshold"], "refusal.similarity_threshold"
    )
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("refusal.similarity_threshold must be within [-1, 1]")

    calibration = _require_mapping(refusal["calibration"], "refusal.calibration")
    if calibration:
        if set(calibration) != {"type", "slope", "intercept"}:
            raise ValueError(
                "non-empty refusal.calibration must be a complete Platt mapping"
            )
        if calibration.get("type") != "platt":
            raise ValueError("refusal.calibration.type must be 'platt'")
        slope = _finite_float(calibration["slope"], "refusal.calibration.slope")
        intercept = _finite_float(
            calibration["intercept"], "refusal.calibration.intercept"
        )
        if slope <= 0.0:
            raise ValueError("refusal.calibration.slope must be positive")
        normalized_calibration: dict[str, Any] = {
            "type": "platt",
            "slope": slope,
            "intercept": intercept,
        }
        calibration_kind = "platt"
    else:
        normalized_calibration = {}
        calibration_kind = "none"

    # Validate the authoritative search contract even when no post-processing
    # is enabled.  For legacy checkpoints, a missing search block means raw
    # query/gallery cosine similarity.
    postprocess = checkpoint_postprocess(checkpoint)
    score_domain = search_contract(postprocess)["score_domain"]
    source = _require_mapping(checkpoint.get("source", {}), "source")
    explicit_domain = source.get("refusal_score_domain")
    if explicit_domain is not None and explicit_domain != score_domain:
        raise ValueError(
            "checkpoint refusal_score_domain does not match its search contract"
        )

    report_path = source.get("robust_open_set_calibration")
    report_digest = source.get("robust_open_set_calibration_sha256")
    if (report_path is None) != (report_digest is None):
        raise ValueError(
            "robust refusal provenance requires both report path and SHA-256"
        )
    if report_path is not None:
        if not isinstance(report_path, str) or not report_path:
            raise ValueError("robust_open_set_calibration must be a non-empty path")
        report_digest = _validate_sha256(
            report_digest, "robust_open_set_calibration_sha256"
        )
        provenance_kind = "robust_report"
    elif calibration_kind == "platt":
        provenance_kind = "embedded_unlinked"
    else:
        provenance_kind = "uncalibrated"

    return {
        "similarity_threshold": threshold,
        "calibration": normalized_calibration,
        "score_domain": score_domain,
        "score_domain_explicit": explicit_domain is not None,
        "provenance_kind": provenance_kind,
        "report_path": report_path,
        "report_sha256": report_digest,
    }


def _canonical_search_contract(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve equivalent raw-cosine contracts to one explicit representation.

    Legacy checkpoints omit ``search`` entirely.  An explicit contract with
    ``postprocess.enabled=false`` has exactly the same ranking and score
    domain; its dormant DBA/QE values cannot affect inference.  Both cases are
    therefore serialized with the default disabled contract.  Enabled
    contracts retain every normalized parameter for strict comparison.
    """

    postprocess = checkpoint_postprocess(checkpoint)
    if postprocess is None or not bool(postprocess["enabled"]):
        return search_contract(None)
    return search_contract(postprocess)


def _validate_checkpoint(
    checkpoint: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mapping[str, Tensor],
    dict[str, Any],
]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema: {checkpoint.get('schema_version')!r}"
        )
    architecture = _validate_architecture(checkpoint)
    preprocessing = _validate_preprocessing(checkpoint)
    state = _require_mapping(checkpoint.get("model_state"), "model_state")
    if not state:
        raise ValueError("checkpoint model_state must not be empty")
    for name, value in state.items():
        if not isinstance(value, Tensor):
            raise ValueError(f"model_state[{name!r}] must be a tensor")
    refusal = _validate_refusal_provenance(checkpoint)
    metrics = checkpoint.get("metrics", {})
    _require_mapping(metrics, "metrics")
    return architecture, preprocessing, state, refusal


def _validate_state_tensor(tensor: Tensor, name: str) -> None:
    if tensor.layout != torch.strided:
        raise ValueError(f"model_state[{name!r}] must use strided layout")
    if tensor.is_quantized:
        raise ValueError(f"model_state[{name!r}] cannot be quantized")
    if tensor.is_floating_point() or tensor.is_complex():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"model_state[{name!r}] contains NaN or infinity")


def _normalized_weights(count: int, weights: Sequence[float] | None) -> list[float]:
    if count < 2:
        raise ValueError("model soup requires at least two checkpoints")
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError("the number of weights must equal the number of checkpoints")
    parsed = [_finite_float(value, "soup weight") for value in weights]
    if any(value <= 0.0 for value in parsed):
        raise ValueError("all soup weights must be strictly positive")
    total = math.fsum(parsed)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("sum of soup weights must be finite and positive")
    return [value / total for value in parsed]


def _accumulation_dtype(tensor: Tensor) -> torch.dtype:
    if tensor.is_complex():
        return torch.complex128
    return torch.float64


def _initialize_accumulator(
    state: Mapping[str, Tensor], weight: float
) -> tuple[dict[str, Tensor], dict[str, tuple[torch.Size, torch.dtype]], list[str]]:
    accumulator: dict[str, Tensor] = {}
    specifications: dict[str, tuple[torch.Size, torch.dtype]] = {}
    preserved: list[str] = []
    for name, tensor in state.items():
        value = tensor.detach().cpu()
        _validate_state_tensor(value, name)
        specifications[name] = (value.shape, value.dtype)
        if value.is_floating_point() or value.is_complex():
            accumulator[name] = value.to(_accumulation_dtype(value)).mul(weight)
        else:
            accumulator[name] = value.clone()
            preserved.append(name)
    return accumulator, specifications, preserved


def _accumulate_state(
    accumulator: dict[str, Tensor],
    specifications: Mapping[str, tuple[torch.Size, torch.dtype]],
    state: Mapping[str, Tensor],
    weight: float,
    *,
    member_index: int,
    non_floating_differences: set[str],
) -> None:
    expected_keys = set(specifications)
    actual_keys = set(state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        unexpected = sorted(actual_keys.difference(expected_keys))
        raise ValueError(
            f"checkpoint {member_index} model_state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in specifications:
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise ValueError(
                f"checkpoint {member_index} model_state[{name!r}] is not a tensor"
            )
        value = tensor.detach().cpu()
        _validate_state_tensor(value, name)
        expected_shape, expected_dtype = specifications[name]
        if value.shape != expected_shape:
            raise ValueError(
                f"checkpoint {member_index} model_state[{name!r}] shape differs: "
                f"{tuple(value.shape)} != {tuple(expected_shape)}"
            )
        if value.dtype != expected_dtype:
            raise ValueError(
                f"checkpoint {member_index} model_state[{name!r}] dtype differs: "
                f"{value.dtype} != {expected_dtype}"
            )
        target = accumulator[name]
        if value.is_floating_point() or value.is_complex():
            target.add_(value.to(target.dtype), alpha=weight)
        elif not torch.equal(target, value):
            non_floating_differences.add(name)


def _finalize_state(
    accumulator: Mapping[str, Tensor],
    specifications: Mapping[str, tuple[torch.Size, torch.dtype]],
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, tensor in accumulator.items():
        _, dtype = specifications[name]
        if tensor.is_floating_point() or tensor.is_complex():
            value = tensor.to(dtype)
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"averaged model_state[{name!r}] is non-finite")
            result[name] = value
        else:
            result[name] = tensor.clone()
    return result


def _compatibility_signature(refusal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "score_domain": refusal["score_domain"],
        "calibration_kind": refusal["calibration"].get("type", "none"),
        "provenance_kind": refusal["provenance_kind"],
    }


def create_model_soup(
    checkpoint_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    weights: Sequence[float] | None = None,
    anchor_index: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Average compatible inference checkpoints and atomically save the soup.

    Parameters are averaged with normalized positive convex weights.  When
    ``weights`` is omitted, uniform averaging is used.  The anchor supplies
    integer/bool state buffers and all shared inference metadata.
    """

    paths = [Path(path).expanduser() for path in checkpoint_paths]
    normalized_weights = _normalized_weights(len(paths), weights)
    if isinstance(anchor_index, bool) or not isinstance(anchor_index, int):
        raise TypeError("anchor_index must be an integer")
    if not 0 <= anchor_index < len(paths):
        raise ValueError("anchor_index is outside the checkpoint list")

    destination = Path(output_path).expanduser()
    resolved_destination = destination.resolve()
    resolved_inputs: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
        if resolved == resolved_destination:
            raise ValueError("output checkpoint must not overwrite an input checkpoint")
        resolved_inputs.append(resolved)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output checkpoint already exists (use --overwrite): {destination}"
        )

    digests = [sha256_file(path) for path in resolved_inputs]
    if len(set(digests)) != len(digests):
        raise ValueError("model soup inputs must have distinct checkpoint contents")

    anchor_path = resolved_inputs[anchor_index]
    anchor_checkpoint = torch.load(
        anchor_path, map_location="cpu", weights_only=True
    )
    architecture, preprocessing, anchor_state, anchor_refusal = _validate_checkpoint(
        anchor_checkpoint
    )
    # Missing legacy search metadata and explicit disabled post-processing are
    # semantically identical raw-cosine contracts.  Enabled DBA/QE contracts
    # preserve every normalized setting and remain strict.
    anchor_search = _canonical_search_contract(anchor_checkpoint)
    anchor_refusal_signature = _compatibility_signature(anchor_refusal)

    accumulator, specifications, preserved = _initialize_accumulator(
        anchor_state, normalized_weights[anchor_index]
    )
    non_floating_differences: set[str] = set()
    refusal_provenance: list[dict[str, Any] | None] = [None] * len(paths)
    refusal_provenance[anchor_index] = anchor_refusal
    del anchor_state
    del anchor_checkpoint

    for index, path in enumerate(resolved_inputs):
        if index == anchor_index:
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        candidate_architecture, candidate_preprocessing, state, refusal = (
            _validate_checkpoint(checkpoint)
        )
        if not _strict_equal(candidate_architecture, architecture):
            raise ValueError(f"checkpoint {index} model architecture differs from anchor")
        if not _strict_equal(candidate_preprocessing, preprocessing):
            raise ValueError(f"checkpoint {index} preprocessing differs from anchor")
        if not _strict_equal(_canonical_search_contract(checkpoint), anchor_search):
            raise ValueError(f"checkpoint {index} search contract differs from anchor")
        if _compatibility_signature(refusal) != anchor_refusal_signature:
            raise ValueError(
                f"checkpoint {index} refusal provenance differs from anchor"
            )
        refusal_provenance[index] = refusal
        _accumulate_state(
            accumulator,
            specifications,
            state,
            normalized_weights[index],
            member_index=index,
            non_floating_differences=non_floating_differences,
        )
        del state
        del checkpoint

    soup_state = _finalize_state(accumulator, specifications)
    members = [
        {
            "index": index,
            "path": str(paths[index]),
            "sha256": digests[index],
            "weight": normalized_weights[index],
            "refusal_provenance": refusal_provenance[index],
        }
        for index in range(len(paths))
    ]
    source = {
        "model_soup": {
            "schema_version": MODEL_SOUP_SCHEMA_VERSION,
            "method": "uniform" if weights is None else "weighted",
            "anchor_index": anchor_index,
            "members": members,
            "floating_state_policy": "convex_mean_accumulated_in_float64",
            "bn_running_statistics_policy": "convex_mean",
            "non_floating_state_policy": "copied_from_anchor",
            "anchor_preserved_state_keys": sorted(preserved),
            "non_floating_member_differences": sorted(non_floating_differences),
            "search_policy": "canonical_explicit_contract",
            "refusal_policy": "invalidated_requires_recalibration",
            "metrics_policy": "invalidated_not_inherited",
        }
    }
    output: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": architecture,
        "model_state": soup_state,
        "preprocessing": preprocessing,
        "refusal": {
            # A finite conservative sentinel keeps the checkpoint schema
            # inspectable while the status prevents interpreting it as a
            # calibrated deployment threshold.
            "similarity_threshold": REFUSAL_RECALIBRATION_THRESHOLD,
            "calibration": {},
            "status": "requires_recalibration",
            "reason": "model_state_changed_by_model_soup",
        },
        "metrics": {},
        "source": source,
    }
    output["search"] = anchor_search

    # Re-check immediately before the atomic replace to avoid an accidental
    # overwrite if another process created the target during the computation.
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output checkpoint already exists (use --overwrite): {destination}"
        )
    atomic_torch_save(output, destination)
    return {
        "output": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "method": source["model_soup"]["method"],
        "weights": normalized_weights,
        "anchor_index": anchor_index,
        "members": len(paths),
        "floating_tensors_averaged": sum(
            tensor.is_floating_point() or tensor.is_complex()
            for tensor in soup_state.values()
        ),
        "non_floating_tensors_from_anchor": len(preserved),
        "requires_refusal_recalibration": True,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="two or more compatible inference checkpoints",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="positive per-checkpoint weights; omit for a uniform soup",
    )
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=0,
        help="member supplying integer/bool buffers (default: first)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing output (never an input)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = create_model_soup(
        args.checkpoints,
        args.output,
        weights=args.weights,
        anchor_index=args.anchor_index,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "MODEL_SOUP_SCHEMA_VERSION",
    "REFUSAL_RECALIBRATION_THRESHOLD",
    "create_model_soup",
    "main",
]
