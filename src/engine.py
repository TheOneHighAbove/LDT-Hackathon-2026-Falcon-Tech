"""Model construction, checkpoint I/O, and memory-safe embedding extraction."""

from __future__ import annotations

import time
from copy import deepcopy
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import (
    normalize_input_size,
    normalize_postprocess_config,
    normalize_resize_mode,
)
from .model import VehicleReIDModel
from .utils import atomic_torch_save, cpu_state_dict


CHECKPOINT_SCHEMA_VERSION = 1
RAW_COSINE_SCORE_DOMAIN = "cosine(raw_query,raw_gallery)"
DBA_QE_SCORE_DOMAIN = "cosine(expanded_query, dba_gallery)"


def search_contract(
    postprocess: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the portable ranking/score-domain contract stored in checkpoints."""

    normalized = normalize_postprocess_config(postprocess)
    return {
        "postprocess": normalized,
        "score_domain": (
            DBA_QE_SCORE_DOMAIN
            if bool(normalized["enabled"])
            else RAW_COSINE_SCORE_DOMAIN
        ),
    }


def checkpoint_postprocess(
    checkpoint: Mapping[str, Any],
) -> dict[str, bool | int | float] | None:
    """Validate and return an authoritative checkpoint search contract.

    Legacy checkpoints have no ``search`` field and return ``None``.  Once the
    field is present its post-processing and score domain are inseparable from
    the calibrated refusal threshold.
    """

    raw_search = checkpoint.get("search")
    if raw_search is None:
        return None
    if not isinstance(raw_search, Mapping):
        raise ValueError("checkpoint search metadata must be a mapping")
    unknown = sorted(set(raw_search).difference({"postprocess", "score_domain"}))
    if unknown:
        raise ValueError(f"checkpoint search metadata has unknown fields: {unknown}")
    if "postprocess" not in raw_search or "score_domain" not in raw_search:
        raise ValueError("checkpoint search metadata is incomplete")
    normalized = normalize_postprocess_config(raw_search["postprocess"])
    expected_domain = search_contract(normalized)["score_domain"]
    if raw_search["score_domain"] != expected_domain:
        raise ValueError(
            "checkpoint search score_domain does not match its postprocess contract"
        )
    return normalized


def resolve_checkpoint_postprocess(
    checkpoint: Mapping[str, Any],
    configured: Mapping[str, Any] | None,
) -> dict[str, bool | int | float]:
    """Resolve search settings, rejecting YAML/checkpoint domain mismatches."""

    normalized_config = normalize_postprocess_config(configured)
    authoritative = checkpoint_postprocess(checkpoint)
    if authoritative is None:
        return normalized_config
    if authoritative != normalized_config:
        raise ValueError(
            "inference.postprocess does not match the authoritative checkpoint "
            f"search contract: config={normalized_config}, checkpoint={authoritative}"
        )
    return authoritative


def create_model_from_config(
    config: Mapping[str, Any], *, pretrained: bool | None = None
) -> VehicleReIDModel:
    model_config = config["model"] if "model" in config else config
    use_pretrained = bool(model_config.get("pretrained", False))
    if pretrained is not None:
        use_pretrained = pretrained
    return VehicleReIDModel(
        backbone_name=str(model_config["backbone"]),
        embedding_dim=int(model_config["embedding_dim"]),
        pretrained=use_pretrained,
        pooling=str(model_config.get("pooling", "avg")),
        gem_p=float(model_config.get("gem_p", 3.0)),
        gem_eps=float(model_config.get("gem_eps", 1e-6)),
        local_heads=int(model_config.get("local_heads", 4)),
        local_temperature=float(model_config.get("local_temperature", 1.0)),
    )


def build_inference_checkpoint(
    model: VehicleReIDModel,
    *,
    input_size: int | list[int] | tuple[int, int],
    bbox_padding: float,
    refusal_threshold: float,
    resize_mode: str = "direct",
    tta_horizontal_flip: bool = False,
    postprocess: Mapping[str, Any] | None = None,
    calibration: Mapping[str, float] | None = None,
    adaptive_refusal: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_size = normalize_input_size(input_size)
    serialized_size = (
        int(normalized_size)
        if isinstance(normalized_size, int)
        else list(normalized_size)
    )
    refusal: dict[str, Any] = {
        "similarity_threshold": float(refusal_threshold),
        "calibration": dict(calibration or {}),
    }
    # Keep legacy checkpoint serialization structurally identical when no
    # adaptive policy is requested.  The payload is validated by its dedicated
    # deployment loader before use and copied here to avoid caller mutation.
    if adaptive_refusal is not None:
        refusal["adaptive"] = deepcopy(dict(adaptive_refusal))
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.get_config(),
        "model_state": cpu_state_dict(model),
        "preprocessing": {
            "input_size": serialized_size,
            "bbox_padding": float(bbox_padding),
            "resize_mode": normalize_resize_mode(resize_mode),
            "tta_horizontal_flip": bool(tta_horizontal_flip),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "refusal": refusal,
        "search": search_contract(postprocess),
        "metrics": dict(metrics or {}),
        "source": dict(source or {}),
    }
    if adaptive_refusal is not None:
        from .refusal import checkpoint_adaptive_refusal

        checkpoint_adaptive_refusal(checkpoint)
    return checkpoint


def save_inference_checkpoint(
    path: str | Path,
    model: VehicleReIDModel,
    **metadata: Any,
) -> None:
    atomic_torch_save(build_inference_checkpoint(model, **metadata), path)


def load_inference_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[VehicleReIDModel, dict[str, Any]]:
    """Restore a model without invoking any pretrained-weight download."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a mapping")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema: {checkpoint.get('schema_version')!r}"
        )
    # Validate optional portable refusal metadata before constructing the
    # network.  Legacy checkpoints have no adaptive payload and return fast.
    from .refusal import checkpoint_adaptive_refusal

    checkpoint_adaptive_refusal(checkpoint)
    architecture = dict(checkpoint["model"])
    architecture["pretrained"] = False
    model = VehicleReIDModel(**architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _metadata_values(batch: Mapping[str, Any], key: str, batch_size: int) -> list[Any]:
    if key not in batch:
        return [None] * batch_size
    value = batch[key]
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * batch_size


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    amp: bool = True,
    tta_horizontal_flip: bool = False,
    channels_last: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract normalized float32 embeddings while preserving loader order."""

    model.eval()
    chunks: list[np.ndarray] = []
    metadata: dict[str, list[Any]] = {
        "image_id": [],
        "vehicle_id": [],
        "camera_id": [],
    }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        with _autocast_context(device, amp):
            embeddings = model(images)
            if tta_horizontal_flip:
                flipped = model(torch.flip(images, dims=(-1,)))
                embeddings = F.normalize(
                    embeddings.float() + flipped.float(), p=2, dim=1, eps=1e-12
                )
        embeddings = F.normalize(embeddings.float(), p=2, dim=1, eps=1e-12)
        chunks.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
        batch_size = int(images.shape[0])
        for key in metadata:
            metadata[key].extend(_metadata_values(batch, key, batch_size))

    if not chunks:
        embedding_dim = int(getattr(model, "embedding_dim", 0))
        matrix = np.empty((0, embedding_dim), dtype=np.float32)
    else:
        matrix = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    packed = {key: np.asarray(values) for key, values in metadata.items()}
    return matrix, packed


@torch.inference_mode()
def benchmark_embedding_model(
    model: nn.Module,
    sample: Tensor,
    *,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
    amp: bool = True,
    tta_horizontal_flip: bool = False,
) -> dict[str, float | bool]:
    """Measure synchronized batch=1 latency; intended as a local reference only."""

    if sample.shape[0] != 1:
        raise ValueError("Latency benchmark requires batch=1")
    model.eval()
    sample = sample.to(device)
    def descriptor() -> Tensor:
        with _autocast_context(device, amp):
            embedding = model(sample)
            if tta_horizontal_flip:
                flipped = model(torch.flip(sample, dims=(-1,)))
                embedding = F.normalize(
                    embedding.float() + flipped.float(), p=2, dim=1, eps=1e-12
                )
        return embedding

    for _ in range(warmup):
        descriptor()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        descriptor()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    latency_ms = elapsed * 1000.0 / iterations
    return {
        "batch_size": 1.0,
        "latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms,
        "tta_horizontal_flip": bool(tta_horizontal_flip),
    }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DBA_QE_SCORE_DOMAIN",
    "RAW_COSINE_SCORE_DOMAIN",
    "benchmark_embedding_model",
    "build_inference_checkpoint",
    "checkpoint_postprocess",
    "create_model_from_config",
    "extract_embeddings",
    "load_inference_checkpoint",
    "resolve_checkpoint_postprocess",
    "save_inference_checkpoint",
    "search_contract",
]
