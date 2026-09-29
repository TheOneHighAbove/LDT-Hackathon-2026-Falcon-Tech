"""Export the four release backbones as self-contained offline TorchScript files."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.benchmark_backbone_components import _clip, _convnext_tiny, _dinov2
from scripts.probe_osnet_loss_branch_fusion import _model as load_osnet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path("weights/release")


class ConvNeXtRelease(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.projection = model.projection
        self.bnneck = model.bnneck

    def forward(self, image):
        spatial = self.backbone.forward_features(image)
        pooled = self.backbone.forward_head(spatial)
        embedding = F.normalize(
            self.bnneck(self.projection(pooled)).float(), dim=1
        )
        parts = F.adaptive_avg_pool2d(spatial, (3, 3)).flatten(2).transpose(1, 2)
        return embedding, F.normalize(parts.float(), dim=2)


class OSNetRelease(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        spatial = self.model.featuremaps(image)
        pooled = self.model.global_avgpool(spatial).flatten(1)
        identity = F.normalize(self.model.fc[0](pooled).float(), dim=1)
        metric = F.normalize(self.model.fc[1](pooled).float(), dim=1)
        parts = F.adaptive_avg_pool2d(spatial, (4, 4)).flatten(2).transpose(1, 2)
        return identity, metric, F.normalize(parts.float(), dim=2)


class DINOv2Release(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        value = self.model.forward_features(image)
        return value["x_norm_clstoken"], value["x_norm_patchtokens"]


class CLIPRelease(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        value = self.model.forward_intermediates(
            image,
            indices=1,
            normalize_intermediates=True,
            intermediates_only=False,
            output_fmt="NLC",
            output_extra_tokens=False,
        )
        return value["image_features"], value["image_intermediates"][0]


def _export(name: str, model: nn.Module, example: torch.Tensor) -> dict:
    destination = OUTPUT / f"{name}.ts"
    if destination.is_file() and destination.stat().st_size > 1_000_000:
        loaded = torch.jit.load(str(destination), map_location=DEVICE).eval()
        with torch.inference_mode(), torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            observed = loaded(example)
        return {
            "path": destination.as_posix(),
            "bytes": destination.stat().st_size,
            "outputs": [list(value.shape) for value in observed],
        }
    model = model.to(DEVICE).eval()
    with torch.inference_mode(), torch.autocast(
        device_type=DEVICE.type,
        enabled=DEVICE.type == "cuda",
        dtype=torch.float16,
    ):
        reference = model(example)
        traced = torch.jit.trace(model, example, strict=False, check_trace=False)
        observed = traced(example)
    for expected, actual in zip(reference, observed, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    traced = torch.jit.freeze(traced.eval())
    torch.jit.save(traced, destination)
    return {
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "outputs": [list(value.shape) for value in observed],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"device": str(DEVICE), "artifacts": {}}

    conv, *_ = _convnext_tiny()
    report["artifacts"]["convnext"] = _export(
        "convnext_global_parts", ConvNeXtRelease(conv),
        torch.randn(1, 3, 256, 256, device=DEVICE),
    )
    del conv
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    osnet = load_osnet(Path("weights/osnet_target_loss_branches.pt"))
    report["artifacts"]["osnet"] = _export(
        "osnet_loss_branch_global_parts", OSNetRelease(osnet),
        torch.randn(1, 3, 208, 208, device=DEVICE),
    )
    del osnet
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    dino, *_ = _dinov2()
    report["artifacts"]["dinov2"] = _export(
        "dinov2_vehicle_cls_tokens", DINOv2Release(dino),
        torch.randn(1, 3, 280, 280, device=DEVICE),
    )
    del dino
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    clip, *_ = _clip()
    clip.set_grad_checkpointing(False)
    report["artifacts"]["clip"] = _export(
        "clip_vehicle_global_tokens", CLIPRelease(clip),
        torch.randn(1, 3, 224, 224, device=DEVICE),
    )
    report["total_bytes"] = sum(
        value["bytes"] for value in report["artifacts"].values()
    )
    (OUTPUT / "backbone_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
