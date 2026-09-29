"""Configuration loading and path resolution helpers."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import yaml


POSTPROCESS_DEFAULTS: dict[str, bool | int | float] = {
    "enabled": False,
    "dba_top_k": 3,
    "dba_alpha": 2.0,
    "qe_top_k": 2,
    "qe_alpha": 1.0,
}

XBM_DEFAULTS: dict[str, bool | int] = {
    "enabled": False,
    "capacity": 2048,
}

MULTI_SIMILARITY_DEFAULTS: dict[str, float] = {
    "alpha": 2.0,
    "beta": 50.0,
    "base": 0.5,
    "epsilon": 0.1,
}

HARD_IDENTITY_MINING_DEFAULTS: dict[str, bool | int | float] = {
    "enabled": False,
    "hard_fraction": 0.5,
    "neighbors_per_identity": 32,
    "warmup_epochs": 1,
    "refresh_interval": 1,
    "embedding_batch_size": 64,
    "tta_horizontal_flip": False,
}


def normalize_hard_identity_mining_config(
    value: Any | None,
) -> dict[str, bool | int | float]:
    """Validate deterministic offline hard-identity mining settings."""

    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError("training.hard_identity_mining must be a mapping")
    unknown = sorted(set(raw).difference(HARD_IDENTITY_MINING_DEFAULTS))
    if unknown:
        raise ValueError(
            "Unknown training.hard_identity_mining fields: " + ", ".join(unknown)
        )

    enabled = raw.get("enabled", HARD_IDENTITY_MINING_DEFAULTS["enabled"])
    if not isinstance(enabled, bool):
        raise TypeError("training.hard_identity_mining.enabled must be boolean")
    tta = raw.get(
        "tta_horizontal_flip",
        HARD_IDENTITY_MINING_DEFAULTS["tta_horizontal_flip"],
    )
    if not isinstance(tta, bool):
        raise TypeError(
            "training.hard_identity_mining.tta_horizontal_flip must be boolean"
        )

    fraction_value = raw.get(
        "hard_fraction", HARD_IDENTITY_MINING_DEFAULTS["hard_fraction"]
    )
    if isinstance(fraction_value, bool):
        raise TypeError(
            "training.hard_identity_mining.hard_fraction must be numeric"
        )
    try:
        hard_fraction = float(fraction_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.hard_identity_mining.hard_fraction must be finite and in (0, 1)"
        ) from exc
    if not math.isfinite(hard_fraction) or not 0.0 < hard_fraction < 1.0:
        raise ValueError(
            "training.hard_identity_mining.hard_fraction must be finite and in (0, 1)"
        )

    integers: dict[str, int] = {}
    for name, minimum in (
        ("neighbors_per_identity", 1),
        ("warmup_epochs", 0),
        ("refresh_interval", 1),
        ("embedding_batch_size", 1),
    ):
        item = raw.get(name, HARD_IDENTITY_MINING_DEFAULTS[name])
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(
                f"training.hard_identity_mining.{name} must be a {qualifier} integer"
            )
        integers[name] = int(item)

    return {
        "enabled": enabled,
        "hard_fraction": hard_fraction,
        **integers,
        "tta_horizontal_flip": tta,
    }


def normalize_multi_similarity_config(value: Any | None) -> dict[str, float]:
    """Validate Multi-Similarity loss hyperparameters strictly."""

    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError("training.multi_similarity must be a mapping")
    unknown = sorted(set(raw).difference(MULTI_SIMILARITY_DEFAULTS))
    if unknown:
        raise ValueError(
            "Unknown training.multi_similarity fields: " + ", ".join(unknown)
        )

    normalized: dict[str, float] = {}
    for name, default in MULTI_SIMILARITY_DEFAULTS.items():
        item = raw.get(name, default)
        if isinstance(item, bool):
            raise TypeError(f"training.multi_similarity.{name} must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"training.multi_similarity.{name} must be finite"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(
                f"training.multi_similarity.{name} must be finite"
            )
        normalized[name] = number

    if normalized["alpha"] <= 0.0:
        raise ValueError("training.multi_similarity.alpha must be positive")
    if normalized["beta"] <= 0.0:
        raise ValueError("training.multi_similarity.beta must be positive")
    if not -1.0 <= normalized["base"] <= 1.0:
        raise ValueError("training.multi_similarity.base must be in [-1, 1]")
    if not 0.0 <= normalized["epsilon"] <= 2.0:
        raise ValueError("training.multi_similarity.epsilon must be in [0, 2]")
    return normalized


def normalize_cross_batch_memory_config(
    value: Any | None,
) -> dict[str, bool | int]:
    """Validate the optional detached cross-batch metric-learning queue.

    The queue is disabled when the section is absent, preserving the historical
    batch-hard objective exactly.  Unknown keys are rejected deliberately: a
    misspelled capacity must not silently turn an expensive experiment into the
    baseline.
    """

    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError("training.cross_batch_memory must be a mapping")
    unknown = sorted(set(raw).difference(XBM_DEFAULTS))
    if unknown:
        raise ValueError(
            "Unknown training.cross_batch_memory fields: " + ", ".join(unknown)
        )

    enabled = raw.get("enabled", XBM_DEFAULTS["enabled"])
    if not isinstance(enabled, bool):
        raise TypeError("training.cross_batch_memory.enabled must be boolean")
    capacity = raw.get("capacity", XBM_DEFAULTS["capacity"])
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError(
            "training.cross_batch_memory.capacity must be a positive integer"
        )
    return {"enabled": enabled, "capacity": int(capacity)}


def normalize_postprocess_config(value: Any | None) -> dict[str, bool | int | float]:
    """Validate optional inference DBA/QE settings without mutating the input.

    Missing settings deliberately resolve to a disabled configuration so old
    YAML files and indexes retain the baseline raw-cosine behaviour.
    """

    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError("inference.postprocess must be a mapping")
    unknown = sorted(set(raw).difference(POSTPROCESS_DEFAULTS))
    if unknown:
        raise ValueError(
            "Unknown inference.postprocess fields: " + ", ".join(unknown)
        )

    enabled = raw.get("enabled", POSTPROCESS_DEFAULTS["enabled"])
    if not isinstance(enabled, bool):
        raise TypeError("inference.postprocess.enabled must be boolean")

    normalized: dict[str, bool | int | float] = {"enabled": enabled}
    for name in ("dba_top_k", "qe_top_k"):
        item = raw.get(name, POSTPROCESS_DEFAULTS[name])
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(
                f"inference.postprocess.{name} must be a positive integer"
            )
        normalized[name] = int(item)
    for name in ("dba_alpha", "qe_alpha"):
        try:
            item = float(raw.get(name, POSTPROCESS_DEFAULTS[name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"inference.postprocess.{name} must be finite and non-negative"
            ) from exc
        if not math.isfinite(item) or item < 0.0:
            raise ValueError(
                f"inference.postprocess.{name} must be finite and non-negative"
            )
        normalized[name] = item
    return normalized


def normalize_input_size(value: Any, *, minimum: int = 32) -> int | tuple[int, int]:
    """Validate and normalize ``input_size`` as an int or ``(height, width)``."""

    if isinstance(value, bool):
        raise TypeError("model.input_size must be an integer or [height, width]")
    if isinstance(value, int):
        dimensions = (value, value)
        normalized: int | tuple[int, int] = value
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("model.input_size must contain [height, width]")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise TypeError("model.input_size dimensions must be integers")
        dimensions = (int(value[0]), int(value[1]))
        normalized = dimensions
    else:
        raise TypeError("model.input_size must be an integer or [height, width]")
    if any(dimension < minimum for dimension in dimensions):
        raise ValueError(
            f"model.input_size dimensions must each be at least {minimum}"
        )
    return normalized


def normalize_resize_mode(value: Any) -> str:
    """Validate the preprocessing geometry selector."""

    mode = str(value).lower()
    if mode not in {"direct", "letterbox"}:
        raise ValueError("data.resize_mode must be either 'direct' or 'letterbox'")
    return mode


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and attach its source/repository paths."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("The configuration root must be a mapping")

    config = deepcopy(loaded)
    config["_config_path"] = str(config_path)
    # configs/*.yaml is the documented layout.  Keeping the root explicit makes
    # every data/output path independent of the caller's current directory.
    config["_repo_root"] = str(config_path.parent.parent)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"seed", "paths", "model", "data", "training", "inference"}
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    model = config["model"]
    if int(model.get("embedding_dim", 0)) <= 0:
        raise ValueError("model.embedding_dim must be positive")
    normalize_input_size(model.get("input_size", 0))
    pooling = str(model.get("pooling", "avg")).lower()
    if pooling not in {"avg", "gem", "global_local"}:
        raise ValueError(
            "model.pooling must be one of 'avg', 'gem', or 'global_local'"
        )
    for name, default in (("gem_p", 3.0), ("gem_eps", 1e-6)):
        value = float(model.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"model.{name} must be finite and positive")
    local_heads = model.get("local_heads", 4)
    if (
        isinstance(local_heads, bool)
        or not isinstance(local_heads, int)
        or local_heads <= 0
    ):
        raise ValueError("model.local_heads must be a positive integer")
    local_temperature = model.get("local_temperature", 1.0)
    if isinstance(local_temperature, bool) or not isinstance(
        local_temperature, (int, float)
    ):
        raise TypeError("model.local_temperature must be a finite positive number")
    local_temperature = float(local_temperature)
    if not math.isfinite(local_temperature) or local_temperature <= 0:
        raise ValueError("model.local_temperature must be finite and positive")
    normalize_resize_mode(config["data"].get("resize_mode", "direct"))

    training = config["training"]
    if "lr_local_head" in training:
        lr_local_head = training["lr_local_head"]
        if isinstance(lr_local_head, bool) or not isinstance(
            lr_local_head, (int, float)
        ):
            raise TypeError(
                "training.lr_local_head must be a finite positive number"
            )
        lr_local_head = float(lr_local_head)
        if not math.isfinite(lr_local_head) or lr_local_head <= 0.0:
            raise ValueError(
                "training.lr_local_head must be finite and positive"
            )
        if pooling != "global_local":
            raise ValueError(
                "training.lr_local_head is only valid when "
                "model.pooling='global_local'"
            )
    identities = int(training.get("identities_per_batch", 0))
    instances = int(training.get("instances_per_identity", 0))
    if identities < 2 or instances < 2:
        raise ValueError(
            "PK sampling requires at least two identities and two instances per identity"
        )

    metric_loss = str(training.get("metric_loss", "triplet")).lower()
    if metric_loss not in {"triplet", "multi_similarity"}:
        raise ValueError(
            "training.metric_loss must be 'triplet' or 'multi_similarity'"
        )
    normalize_multi_similarity_config(training.get("multi_similarity"))

    subcenters = training.get("arcface_subcenters", 1)
    if isinstance(subcenters, bool) or not isinstance(subcenters, int) or subcenters < 1:
        raise ValueError("training.arcface_subcenters must be a positive integer")
    margin_mode = str(training.get("triplet_margin_mode", "fixed")).lower()
    if margin_mode not in {"fixed", "soft"}:
        raise ValueError("training.triplet_margin_mode must be 'fixed' or 'soft'")
    positive_mining = str(training.get("triplet_positive_mining", "all")).lower()
    if positive_mining not in {
        "all",
        "cross_camera_preferred",
        "cross_camera_only",
    }:
        raise ValueError(
            "training.triplet_positive_mining must be 'all', "
            "'cross_camera_preferred', or 'cross_camera_only'"
        )
    xbm = normalize_cross_batch_memory_config(
        training.get("cross_batch_memory")
    )
    if xbm["enabled"]:
        try:
            triplet_weight = float(training.get("triplet_weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("training.triplet_weight must be finite") from exc
        if not math.isfinite(triplet_weight) or triplet_weight <= 0.0:
            raise ValueError(
                "training.cross_batch_memory requires a positive "
                "training.triplet_weight"
            )

    hard_mining = normalize_hard_identity_mining_config(
        training.get("hard_identity_mining")
    )
    if hard_mining["enabled"]:
        if identities < 3:
            raise ValueError(
                "training.hard_identity_mining requires at least three "
                "identities_per_batch so every batch retains random IDs"
            )
        if xbm["enabled"]:
            raise ValueError(
                "training.hard_identity_mining and training.cross_batch_memory "
                "cannot both be enabled; hard mining must use fresh current-batch "
                "features only"
            )

    normalize_postprocess_config(config["inference"].get("postprocess"))


def repo_root(config: dict[str, Any]) -> Path:
    return Path(config["_repo_root"])


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve configuration paths relative to the repository root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root(config) / path).resolve()


def configured_path(config: dict[str, Any], name: str) -> Path:
    try:
        value = config["paths"][name]
    except KeyError as exc:
        raise KeyError(f"Unknown configured path: paths.{name}") from exc
    return resolve_path(config, value)


__all__ = [
    "configured_path",
    "HARD_IDENTITY_MINING_DEFAULTS",
    "load_config",
    "normalize_cross_batch_memory_config",
    "normalize_hard_identity_mining_config",
    "normalize_input_size",
    "normalize_multi_similarity_config",
    "normalize_postprocess_config",
    "normalize_resize_mode",
    "POSTPROCESS_DEFAULTS",
    "XBM_DEFAULTS",
    "MULTI_SIMILARITY_DEFAULTS",
    "repo_root",
    "resolve_path",
    "validate_config",
]
