"""One-command strict test inference for the score-optimized release stack.

The image extractor uses three offline TorchScript modules in the speed profile
and adds CLIP in the quality profile. Every module runs once per crop; DINOv2
and optional CLIP expose global and local tokens from that same pass. All
downstream operations use one current query and the static gallery only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import decode_jpeg, read_file
from torchvision.ops import roi_align
from torchvision.transforms import InterpolationMode

from src.release_features import (
    clip_geometry_features,
    color_descriptor,
    dino_part_pool,
)
from src.release_io import write_submission
from src.release_models import (
    ClipDeformableCrossEncoder,
    CrossImageTransformer,
    DinoTokenCrossMatcher,
    FamilyGraphReranker,
    predict_clip_cross,
    predict_dino_token,
    predict_family_graph,
)
from src.release_reranking import (
    PATCH_TOP,
    TOKEN_TOP,
    augment_static_gallery,
    gather_relation,
    local_for_protocol,
    modern_affinity,
    modern_linker_build,
    patch_gallery_affinity,
    predict_family_ranker,
    predict_patch,
    prepare_retrieval,
    refusal_features,
    score_gallery,
    symmetric_token_matrix,
    zscore_rows,
)
from src.data import IMAGENET_MEAN, IMAGENET_STD, crop_vehicle, padded_bbox
from src.score_calibration import CandidateCorrectnessModel
from src.utils import save_json, sha256_file


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP = PATCH_TOP
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

cv2.setNumThreads(1)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _weighted_concat(arrays, weights):
    return _normalize(np.concatenate([
        math.sqrt(float(weight)) * _normalize(value)
        for value, weight in zip(arrays, weights, strict=True)
        if weight > 0
    ], axis=1))


class SharedCropDataset(Dataset):
    """Decode an original JPEG once and create the four model inputs."""

    def __init__(
        self, frame: pd.DataFrame, images_dir: Path, dino_size: int = 280,
        include_clip: bool = True,
    ):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        def imagenet(size):
            return transforms.Compose([
                transforms.Resize(
                    (size, size), interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        self.conv = imagenet(256)
        self.osnet = imagenet(208)
        self.dino = imagenet(dino_size)
        self.clip = None
        if include_clip:
            self.clip = transforms.Compose([
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.images_dir / f"{row.image_id}.jpg") as source:
            image = source.convert("RGB")
            tight = crop_vehicle(
                image, (row.x, row.y, row.w, row.h), padding=0.0
            )
            padded = crop_vehicle(
                image, (row.x, row.y, row.w, row.h), padding=0.05
            )
        result = {
            "conv": self.conv(padded),
            "osnet": self.osnet(padded),
            "dino": self.dino(tight),
            "color": color_descriptor(tight),
            "image_id": str(row.image_id),
        }
        if self.clip is not None:
            result["clip"] = self.clip(padded)
        return result


class FastDecodeSharedCropDataset(SharedCropDataset):
    """Exact Pillow resize path with OpenCV's faster JPEG decoder."""

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        decoded = cv2.imread(
            str(self.images_dir / f"{row.image_id}.jpg"), cv2.IMREAD_COLOR
        )
        if decoded is None:
            raise FileNotFoundError(self.images_dir / f"{row.image_id}.jpg")
        image = Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        tight = crop_vehicle(image, (row.x, row.y, row.w, row.h), padding=0.0)
        padded = crop_vehicle(image, (row.x, row.y, row.w, row.h), padding=0.05)
        result = {
            "conv": self.conv(padded),
            "osnet": self.osnet(padded),
            "dino": self.dino(tight),
            "color": color_descriptor(tight),
            "image_id": str(row.image_id),
        }
        if self.clip is not None:
            result["clip"] = self.clip(padded)
        return result


class FastSharedCropDataset(Dataset):
    """OpenCV equivalent of the four direct-resize evaluation transforms."""

    def __init__(
        self, frame: pd.DataFrame, images_dir: Path, dino_size: int = 280,
        include_clip: bool = True,
    ):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.imagenet_mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        self.imagenet_std = np.asarray(IMAGENET_STD, dtype=np.float32)
        self.clip_mean = np.asarray(CLIP_MEAN, dtype=np.float32)
        self.clip_std = np.asarray(CLIP_STD, dtype=np.float32)
        self.dino_size = dino_size
        self.include_clip = include_clip

    def __len__(self):
        return len(self.frame)

    @staticmethod
    def _tensor(image, size, mean, std):
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        normalized = (rgb - mean) / std
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = cv2.imread(
            str(self.images_dir / f"{row.image_id}.jpg"), cv2.IMREAD_COLOR
        )
        if image is None:
            raise FileNotFoundError(self.images_dir / f"{row.image_id}.jpg")
        height, width = image.shape[:2]
        bbox = (row.x, row.y, row.w, row.h)
        left, top, right, bottom = padded_bbox((width, height), bbox, 0.05)
        padded = image[top:bottom, left:right]
        left, top, right, bottom = padded_bbox((width, height), bbox, 0.0)
        tight = image[top:bottom, left:right]
        result = {
            "conv": self._tensor(
                padded, 256, self.imagenet_mean, self.imagenet_std
            ),
            "osnet": self._tensor(
                padded, 208, self.imagenet_mean, self.imagenet_std
            ),
            "dino": self._tensor(
                tight, self.dino_size, self.imagenet_mean, self.imagenet_std
            ),
            "color": color_descriptor(
                Image.fromarray(cv2.cvtColor(tight, cv2.COLOR_BGR2RGB))
            ),
            "image_id": str(row.image_id),
        }
        if self.include_clip:
            result["clip"] = self._tensor(
                padded, 224, self.clip_mean, self.clip_std
            )
        return result


class SharedBaseCropDataset(Dataset):
    """Decode once and defer branch resize/normalization to batched CUDA ops."""

    def __init__(
        self, frame: pd.DataFrame, images_dir: Path, dino_size: int = 280,
        include_clip: bool = True,
    ):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.padded = transforms.Compose([
            transforms.Resize(
                (256, 256), interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ])
        self.tight = transforms.Compose([
            transforms.Resize(
                (dino_size, dino_size), interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ])
        self.include_clip = include_clip

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        decoded = cv2.imread(
            str(self.images_dir / f"{row.image_id}.jpg"), cv2.IMREAD_COLOR
        )
        if decoded is None:
            raise FileNotFoundError(self.images_dir / f"{row.image_id}.jpg")
        image = Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        tight = crop_vehicle(image, (row.x, row.y, row.w, row.h), padding=0.0)
        padded = crop_vehicle(image, (row.x, row.y, row.w, row.h), padding=0.05)
        result = {
            "padded": self.padded(padded),
            "tight": self.tight(tight),
            "color": color_descriptor(tight),
            "image_id": str(row.image_id),
        }
        return result


class PillowSharedBaseCropDataset(SharedBaseCropDataset):
    """Pillow decode with two shared base crops; CUDA derives branch sizes."""

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.images_dir / f"{row.image_id}.jpg") as source:
            image = source.convert("RGB")
            tight = crop_vehicle(
                image, (row.x, row.y, row.w, row.h), padding=0.0
            )
            padded = crop_vehicle(
                image, (row.x, row.y, row.w, row.h), padding=0.05
            )
        return {
            "padded": self.padded(padded),
            "tight": self.tight(tight),
            "color": color_descriptor(tight),
            "image_id": str(row.image_id),
        }


class EncodedVehicleDataset(Dataset):
    """Read compressed bytes only; CUDA performs JPEG decode and resizing."""

    def __init__(self, frame: pd.DataFrame, images_dir: Path):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self.images_dir / f"{row.image_id}.jpg"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            tight = crop_vehicle(
                image, (row.x, row.y, row.w, row.h), padding=0.0
            )
        return {
            "encoded": read_file(str(image_path)),
            "bbox": torch.tensor((row.x, row.y, row.w, row.h), dtype=torch.float64),
            "color": color_descriptor(tight),
            "image_id": str(row.image_id),
        }


def _encoded_collate(items):
    return {
        "encoded": [item["encoded"] for item in items],
        "bbox": torch.stack([item["bbox"] for item in items]),
        "color": torch.from_numpy(
            np.stack([item["color"] for item in items])
        ),
        "image_id": [item["image_id"] for item in items],
    }


def _cuda_jpeg_inputs(
    batch, imagenet_mean, imagenet_std, clip_mean, clip_std, dino_size=280
):
    images = decode_jpeg(batch["encoded"], device=DEVICE)
    conv, osnet, dino, clip = [], [], [], []
    for image, bbox_tensor in zip(images, batch["bbox"], strict=True):
        _, height, width = image.shape
        bbox = bbox_tensor.tolist()
        left, top, right, bottom = padded_bbox((width, height), bbox, 0.05)
        padded = image[:, top:bottom, left:right].float().div_(255.0)[None]
        left, top, right, bottom = padded_bbox((width, height), bbox, 0.0)
        tight = image[:, top:bottom, left:right].float().div_(255.0)[None]
        conv.append(F.interpolate(
            padded, (256, 256), mode="bicubic", align_corners=False,
            antialias=True,
        )[0])
        osnet.append(F.interpolate(
            padded, (208, 208), mode="bicubic", align_corners=False,
            antialias=True,
        )[0])
        clip.append(F.interpolate(
            padded, (224, 224), mode="bicubic", align_corners=False,
            antialias=True,
        )[0])
        dino.append(F.interpolate(
            tight, (dino_size, dino_size), mode="bicubic", align_corners=False,
            antialias=True,
        )[0])
    values = {
        "conv": torch.stack(conv),
        "osnet": torch.stack(osnet),
        "dino": torch.stack(dino),
        "clip": torch.stack(clip),
    }
    values["conv"] = (values["conv"] - imagenet_mean) / imagenet_std
    values["osnet"] = (values["osnet"] - imagenet_mean) / imagenet_std
    values["dino"] = (values["dino"] - imagenet_mean) / imagenet_std
    values["clip"] = (values["clip"] - clip_mean) / clip_std
    return values


def _cuda_roi_inputs(
    batch, imagenet_mean, imagenet_std, clip_mean, clip_std, dino_size=280
):
    decoded = decode_jpeg(batch["encoded"], device=DEVICE)
    images = torch.stack(decoded).float().div_(255.0)
    padded_boxes, tight_boxes = [], []
    _, _, height, width = images.shape
    for bbox_tensor in batch["bbox"]:
        bbox = bbox_tensor.tolist()
        padded_boxes.append(torch.tensor(
            [padded_bbox((width, height), bbox, 0.05)],
            device=DEVICE, dtype=torch.float32,
        ))
        tight_boxes.append(torch.tensor(
            [padded_bbox((width, height), bbox, 0.0)],
            device=DEVICE, dtype=torch.float32,
        ))
    def aligned(boxes, size):
        return roi_align(
            images, boxes, output_size=(size, size), spatial_scale=1.0,
            sampling_ratio=2, aligned=False,
        )
    values = {
        "conv": aligned(padded_boxes, 256),
        "osnet": aligned(padded_boxes, 208),
        "dino": aligned(tight_boxes, dino_size),
        "clip": aligned(padded_boxes, 224),
    }
    values["conv"] = (values["conv"] - imagenet_mean) / imagenet_std
    values["osnet"] = (values["osnet"] - imagenet_mean) / imagenet_std
    values["dino"] = (values["dino"] - imagenet_mean) / imagenet_std
    values["clip"] = (values["clip"] - clip_mean) / clip_std
    return values


def _pool_tokens(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    side = int(round(tokens.shape[1] ** 0.5))
    spatial = tokens.reshape(len(tokens), side, side, tokens.shape[2]).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(spatial.float(), (grid, grid))
    return F.normalize(pooled.flatten(2).transpose(1, 2), dim=2)


def _backbone_streams(include_clip: bool = True) -> dict[str, torch.cuda.Stream] | None:
    """Create one persistent stream per independent backbone on CUDA."""
    if DEVICE.type != "cuda":
        return None
    names = ("conv", "osnet", "dino", "clip") if include_clip else ("conv", "osnet", "dino")
    return {name: torch.cuda.Stream() for name in names}


def _run_backbones(models, inputs, streams=None):
    """Run independent backbones concurrently without changing their outputs."""
    autocast = lambda: torch.autocast(
        device_type=DEVICE.type,
        enabled=DEVICE.type == "cuda",
        dtype=torch.float16,
    )
    if streams is None:
        with autocast():
            conv = models["conv"](inputs["conv"])
            osnet = models["osnet"](inputs["osnet"])
            dino = models["dino"](inputs["dino"])
            clip = models["clip"](inputs["clip"]) if "clip" in models else None
        return conv, osnet, dino, clip

    producer = torch.cuda.current_stream()
    outputs = {}
    for name, stream in streams.items():
        stream.wait_stream(producer)
        with torch.cuda.stream(stream), autocast():
            outputs[name] = models[name](inputs[name])
    for stream in streams.values():
        producer.wait_stream(stream)
    return outputs["conv"], outputs["osnet"], outputs["dino"], outputs.get("clip")


@torch.inference_mode()
def extract_features(
    frame: pd.DataFrame,
    images_dir: Path,
    weights_dir: Path,
    batch_size: int,
    workers: int,
    preprocessing: str = "pillow",
    dino_size: int = 280,
    include_clip: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    if preprocessing not in {
        "cuda_jpeg", "cuda_roi", "gpu_shared", "pillow_shared",
        "pillow", "fast_decode", "opencv"
    }:
        raise ValueError(f"unknown preprocessing mode: {preprocessing}")
    models = {
        "conv": torch.jit.load(
            str(weights_dir / "convnext_global_parts.ts"), map_location=DEVICE
        ).eval(),
        "osnet": torch.jit.load(
            str(weights_dir / "osnet_loss_branch_global_parts.ts"), map_location=DEVICE
        ).eval(),
        "dino": torch.jit.load(
            str(weights_dir / "dinov2_vehicle_cls_tokens.ts"), map_location=DEVICE
        ).eval(),
    }
    if include_clip:
        models["clip"] = torch.jit.load(
            str(weights_dir / "clip_vehicle_global_tokens.ts"), map_location=DEVICE
        ).eval()
    streams = _backbone_streams(include_clip)
    # Trigger TorchScript graph specialization and CUDA kernel setup before
    # the steady-state extraction timer.  The official service benchmark is
    # warm; process/model startup is reported separately by the CLI wall time.
    warm_shapes = {
        "conv": (3, 256, 256),
        "osnet": (3, 208, 208),
        "dino": (3, dino_size, dino_size),
        "clip": (3, 224, 224),
    }
    warm_batch = min(batch_size, 32)
    with torch.autocast(
        device_type=DEVICE.type,
        enabled=DEVICE.type == "cuda",
        dtype=torch.float16,
    ):
        for name, model in models.items():
            model(torch.zeros(
                (warm_batch, *warm_shapes[name]), device=DEVICE,
                dtype=torch.float32,
            ))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dataset_type = {
        "gpu_shared": SharedBaseCropDataset,
        "pillow_shared": PillowSharedBaseCropDataset,
        "cuda_jpeg": EncodedVehicleDataset,
        "cuda_roi": EncodedVehicleDataset,
        "pillow": SharedCropDataset,
        "fast_decode": FastDecodeSharedCropDataset,
        "opencv": FastSharedCropDataset,
    }[preprocessing]
    dataset = (
        dataset_type(frame, images_dir)
        if preprocessing in {"cuda_jpeg", "cuda_roi"}
        else dataset_type(
            frame, images_dir, dino_size=dino_size, include_clip=include_clip
        )
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=DEVICE.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=(
            _encoded_collate
            if preprocessing in {"cuda_jpeg", "cuda_roi"}
            else None
        ),
    )
    output_names = [
        "conv", "conv_parts", "os_identity", "os_metric", "os_parts",
        "dino", "dino_mean", "dino_h2", "dino_h4", "dino_tokens5",
        "dino_tokens10",
    ]
    if include_clip:
        output_names.extend(("clip", "clip_tokens7"))
    output = {name: [] for name in output_names}
    output["colors"] = []
    observed = []
    imagenet_mean = torch.tensor(
        IMAGENET_MEAN, device=DEVICE, dtype=torch.float32
    )[None, :, None, None]
    imagenet_std = torch.tensor(
        IMAGENET_STD, device=DEVICE, dtype=torch.float32
    )[None, :, None, None]
    clip_mean = torch.tensor(
        CLIP_MEAN, device=DEVICE, dtype=torch.float32
    )[None, :, None, None]
    clip_std = torch.tensor(
        CLIP_STD, device=DEVICE, dtype=torch.float32
    )[None, :, None, None]
    started = time.perf_counter()
    for batch in loader:
        if preprocessing == "cuda_jpeg":
            inputs = _cuda_jpeg_inputs(
                batch, imagenet_mean, imagenet_std, clip_mean, clip_std,
                dino_size=dino_size,
            )
        elif preprocessing == "cuda_roi":
            inputs = _cuda_roi_inputs(
                batch, imagenet_mean, imagenet_std, clip_mean, clip_std,
                dino_size=dino_size,
            )
        elif preprocessing in {"gpu_shared", "pillow_shared"}:
            padded = batch["padded"].to(DEVICE, non_blocking=True)
            tight = batch["tight"].to(DEVICE, non_blocking=True)
            osnet_image = F.interpolate(
                padded, (208, 208), mode="bicubic", align_corners=False,
                antialias=True,
            )
            inputs = {
                "conv": (padded - imagenet_mean) / imagenet_std,
                "osnet": (osnet_image - imagenet_mean) / imagenet_std,
                "dino": (tight - imagenet_mean) / imagenet_std,
            }
            if include_clip:
                clip_image = F.interpolate(
                    padded, (224, 224), mode="bicubic", align_corners=False,
                    antialias=True,
                )
                inputs["clip"] = (clip_image - clip_mean) / clip_std
        else:
            inputs = {
                name: batch[name].to(DEVICE, non_blocking=True)
                for name in models
            }
        conv_output, osnet_output, dino_output, clip_output = _run_backbones(
            models, inputs, streams
        )
        conv, conv_parts = conv_output
        identity, metric, os_parts = osnet_output
        dino, dino_tokens = dino_output
        if include_clip:
            clip, clip_tokens = clip_output
        # Schedule every normalization/pooling kernel before the first host
        # transfer.  Calling ``.cpu()`` between these operations serialized
        # CUDA repeatedly and cut measured end-to-end throughput in half.
        gpu_values = {
            "conv": conv.float(),
            "conv_parts": conv_parts.half(),
            "os_identity": identity.float(),
            "os_metric": metric.float(),
            "os_parts": os_parts.half(),
            "dino": F.normalize(dino.float(), dim=1),
            "dino_mean": F.normalize(dino_tokens.mean(1).float(), dim=1),
            "dino_h2": dino_part_pool(dino_tokens, 2),
            "dino_h4": dino_part_pool(dino_tokens, 4),
            "dino_tokens5": _pool_tokens(dino_tokens, 5).half(),
            "dino_tokens10": _pool_tokens(dino_tokens, 10).half(),
        }
        if include_clip:
            gpu_values["clip"] = F.normalize(clip.float(), dim=1)
            gpu_values["clip_tokens7"] = _pool_tokens(clip_tokens, 7).half()
        for name, values in gpu_values.items():
            output[name].append(values.cpu().numpy())
        color = batch["color"]
        output["colors"].append(
            color.numpy() if torch.is_tensor(color) else np.asarray(color)
        )
        observed.extend(str(value) for value in batch["image_id"])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    expected = frame.image_id.astype(str).tolist()
    if observed != expected:
        raise RuntimeError("feature extraction changed CSV row order")
    arrays = {name: np.concatenate(values) for name, values in output.items()}
    return arrays, {
        "images": len(frame), "seconds": seconds,
        "milliseconds_per_image": 1000.0 * seconds / len(frame),
        "images_per_second": len(frame) / seconds,
        "batch_size": batch_size,
        "preprocessing": preprocessing,
        "dino_input_size": dino_size,
        "includes_clip": include_clip,
    }


def _episode(frame, protocol, fused, osnet, dino, tokens, token_model):
    qi, gi = protocol["qi"], protocol["gi"]
    gallery = augment_static_gallery(fused[gi], top_k=5, alpha=2.0)
    score = (fused[qi] @ gallery.T).astype(np.float32)
    candidate = np.argsort(-score, axis=1, kind="stable")[:, :TOKEN_TOP]
    token_data = {"qi": qi, "gi": gi, "score": score, "candidates": candidate}
    token = predict_dino_token(token_model, token_data, tokens)
    rows = np.arange(len(qi))[:, None]
    raw_fused = fused[qi] @ fused[gi].T
    raw_osnet = osnet[qi] @ osnet[gi].T
    raw_dino = dino[qi] @ dino[gi].T
    base = score[rows, candidate]
    raw = np.stack((
        base, token, raw_fused[rows, candidate],
        raw_osnet[rows, candidate], raw_dino[rows, candidate],
    ), axis=2).astype(np.float32)
    z = (raw - raw.mean(1, keepdims=True)) / (raw.std(1, keepdims=True) + 1e-6)
    rank = np.broadcast_to(
        np.linspace(0.0, 1.0, TOKEN_TOP, dtype=np.float32)[None, :, None],
        (len(qi), TOKEN_TOP, 1),
    )
    features = np.concatenate((raw, z, rank, raw[..., :2] - raw[:, :1, :2]), axis=2)
    fused_g, osnet_g = fused[gi], osnet[gi]
    token_similarity, token_valid, gallery_rank = symmetric_token_matrix(
        token_model, gi, tokens, fused
    )
    relation = gather_relation(
        candidate, fused_g @ fused_g.T, osnet_g @ osnet_g.T,
        token_similarity, token_valid,
    )
    return {
        "qi": qi, "gi": gi, "score": score, "candidate": candidate,
        "features": features.astype(np.float32), "relation": relation,
        "gallery_token_similarity": token_similarity,
        "gallery_token_valid": token_valid,
        "gallery_rank": gallery_rank,
    }


def _load_models(weights: Path, include_clip: bool = True):
    token_state = torch.load(weights / "dino_token_cross_top50.pt", map_location=DEVICE, weights_only=False)
    token = DinoTokenCrossMatcher().to(DEVICE).eval()
    token.load_state_dict(token_state["model_state"])
    family_state = torch.load(weights / "strict_family_gnn_smoothap.pt", map_location=DEVICE, weights_only=False)
    family = FamilyGraphReranker(family_state["feature_dim"]).to(DEVICE).eval()
    family.load_state_dict(family_state["model_state"])
    verifier_state = torch.load(weights / "dino_top25_verifier.pt", map_location=DEVICE, weights_only=False)
    verifier = CrossImageTransformer(dim=96).to(DEVICE).eval()
    verifier.load_state_dict(verifier_state["model_state"])
    clip = None
    if include_clip:
        clip_state = torch.load(weights / "clip_deformable_cross_encoder_no_tta.pt", map_location=DEVICE, weights_only=False)
        clip = ClipDeformableCrossEncoder().to(DEVICE).eval()
        clip.load_state_dict(clip_state["model_state"])
    return token, family, verifier, clip


def rank_gallery(
    frame, features, weights: Path, release: dict, include_clip: bool = True,
    colors: np.ndarray | None = None,
):
    count_query = int((frame["split"] == "query").sum())
    qi = np.arange(count_query, dtype=np.int64)
    gi = np.arange(count_query, len(frame), dtype=np.int64)
    protocol = {"qi": qi, "gi": gi}

    specialized = _weighted_concat(
        (features["os_identity"], features["os_metric"]), (0.75, 0.25)
    )
    osnet = _weighted_concat((features["conv"], specialized), (0.20, 0.80))
    dino = _normalize(features["dino"])
    fused = _weighted_concat((osnet, dino), (0.75, 0.25))
    parts = {
        name: _normalize(features[f"dino_{name}"])
        for name in ("mean", "h2", "h4")
    }
    parts["metric_dino"] = parts["mean"]

    token_model, family_model, verifier, clip_model = _load_models(
        weights, include_clip=include_clip
    )
    episode = _episode(
        frame, protocol, fused, osnet, dino,
        features["dino_tokens5"], token_model,
    )
    family_value = predict_family_graph(family_model, episode)
    gate_model = joblib.load(weights / "family_lambdarank_appearance_lbs.joblib")
    if colors is None:
        raise ValueError("precomputed color descriptors are required")
    gate_value = predict_family_ranker(
        gate_model, frame, episode, family_value, colors
    )

    prepared = prepare_retrieval(protocol, fused, osnet, parts)
    local = local_for_protocol(
        verifier, protocol, prepared, features["conv_parts"],
        features["os_parts"], osnet,
    )
    patch_model = joblib.load(weights / "dino_patch_matcher.joblib")
    highres_model = joblib.load(weights / "dino_patch_matcher_10x10_lbs.joblib")
    patch = predict_patch(
        patch_model, protocol, prepared, features["dino_tokens5"],
        fused, osnet, dino,
    )
    highres = predict_patch(
        highres_model, protocol, prepared, features["dino_tokens10"],
        fused, osnet, dino, grid_size=10,
    )
    prepared["gallery_sources"]["patch"] = patch_gallery_affinity(
        patch_model, protocol, features["dino_tokens5"], fused, osnet, dino
    )
    modern_model = joblib.load(weights / "modern_gallery_linker_lossbranch.joblib")
    modern_data = modern_linker_build(
        frame, gi, fused, osnet, dino, features["dino_tokens5"], patch_model
    )
    prepared["gallery_sources"]["modern"] = modern_affinity(
        modern_data, modern_model.predict_proba(modern_data["features"])[:, 1]
    )

    previous = release["core_reranker"]
    linker = release["token_gallery_linker"]
    modern_config = release["modern_gallery_linker"]
    score = score_gallery(
        prepared, local, patch, episode, family_value, gate_value, highres,
        previous, linker, query_token_weight=0.10, family_weight=0.15,
        gate_weight=1.0, highres_weight=0.15,
        gate_reject={"window": 10, "rank_gap": 10, "penalty": 0.2},
        modern_linker=modern_config, metric_dino_weight=0.0,
    )

    if not include_clip:
        return np.argsort(-score, axis=1, kind="stable"), score, fused

    candidate = np.argsort(-score, axis=1, kind="stable")[:, :TOP]
    rows = np.arange(len(qi))[:, None]
    base = score[rows, candidate].astype(np.float32)
    clip_protocol = {
        "qi": qi, "gi": gi, "candidate": candidate, "base": base,
    }
    neural = predict_clip_cross(
        clip_model, clip_protocol, features["clip_tokens7"]
    ).astype(np.float32)
    query_pair = np.repeat(qi, TOP)
    gallery_pair = gi[candidate.reshape(-1)]
    geometric = clip_geometry_features(
        features["clip_tokens7"], query_pair, gallery_pair
    ).reshape(len(qi), TOP, -1)
    clip_similarity = np.sum(
        features["clip"][qi, None] * features["clip"][gi[candidate]], axis=2
    ).astype(np.float32)
    rank = np.broadcast_to(
        np.linspace(0.0, 1.0, TOP, dtype=np.float32)[None, :, None],
        (len(qi), TOP, 1),
    )
    raw = np.concatenate((base[..., None], clip_similarity[..., None], geometric), axis=2)
    clip_feature = np.concatenate(
        (raw, zscore_rows(raw), raw - raw[:, :1], rank), axis=2
    )
    extra = np.stack(
        (
            neural,
            zscore_rows(neural),
            neural - base,
            zscore_rows(neural) - zscore_rows(base),
        ),
        axis=2,
    )
    ranker_feature = np.concatenate((clip_feature, extra), axis=2).astype(np.float32)
    ranker_payload = joblib.load(
        weights / "oof_clip_lambdarank_score_optimized_no_dino_tta.joblib"
    )
    flat = ranker_feature.reshape(-1, ranker_feature.shape[2])
    ranker = np.mean([
        model.predict(flat).reshape(len(qi), TOP)
        for model in ranker_payload["models"]
    ], axis=0).astype(np.float32)
    clip_config = release["clip_top25"]
    clip_score = (
        zscore_rows(base)
        + float(clip_config["neural_weight"])
        * (zscore_rows(neural) - zscore_rows(base))
        + float(clip_config["ranker_weight"]) * zscore_rows(ranker)
    )
    rerank_k = int(clip_config["rerank_k"])
    head = np.argsort(-clip_score[:, :rerank_k], axis=1, kind="stable")
    local_order = np.concatenate((
        head,
        np.broadcast_to(
            np.arange(rerank_k, TOP), (len(qi), TOP - rerank_k)
        ),
    ), axis=1)

    # Tune-locked family-frequency consensus; only the current top-5 is
    # permuted, so the Rank-5 set is invariant.
    affinity = prepared["gallery_sources"]["patch"]
    relation = affinity[candidate[:, :, None], candidate[:, None, :]]
    relation = 0.5 * (relation + relation.transpose(0, 2, 1))
    diagonal = np.arange(TOP)
    relation[:, diagonal, diagonal] = -np.inf
    membership = np.zeros_like(base, dtype=np.float32)
    np.put_along_axis(membership, local_order[:, :1], 1.0, axis=1)
    consensus_config = release["family_consensus"]
    threshold = float(consensus_config["threshold"])
    edges = np.clip((relation - threshold) / (1.0 - threshold), 0.0, 1.0)
    bonus = zscore_rows(
        membership + np.einsum("qij,qj->qi", edges, membership)
    )
    consensus = clip_score + float(consensus_config["weight"]) * bonus
    prefix = int(consensus_config["permuted_prefix"])
    first5 = local_order[:, :prefix]
    first5 = np.take_along_axis(
        first5,
        np.argsort(
            -np.take_along_axis(consensus, first5, axis=1),
            axis=1, kind="stable",
        ),
        axis=1,
    )
    local_order = np.concatenate((first5, local_order[:, prefix:]), axis=1)
    top25 = np.take_along_axis(candidate, local_order, axis=1)
    full_order = np.argsort(-score, axis=1, kind="stable")
    full_order[:, :TOP] = top25
    return full_order, score, fused


def _metadata(query: pd.DataFrame, gallery: pd.DataFrame) -> pd.DataFrame:
    q, g = query.copy(), gallery.copy()
    q["split"], g["split"] = "query", "gallery"
    combined = pd.concat((q, g), ignore_index=True)
    # Reranker helpers never use these labels at inference, but keep a valid
    # schema so accidental label access cannot match a query to the gallery.
    combined["vehicle_id"] = np.arange(len(combined), dtype=np.int64)
    combined["camera_id"] = -1
    return combined


def calibrated_refusal(
    fused: np.ndarray,
    count_query: int,
    final_top1: np.ndarray,
    final_scores: np.ndarray,
    release: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the tune-frozen adaptive policy in its calibrated score domain."""

    refusal = release["refusal"]
    threshold = float(refusal["probability_threshold"])
    model = CandidateCorrectnessModel.from_dict(refusal["model"])

    features = refusal_features(
        fused, count_query, final_top1, final_scores, refusal
    )
    probability = model.predict_proba(features)[:, 1]
    accepted = probability >= threshold
    confirmation = refusal["locked_confirmation"]
    metadata = {
        "mode": "adaptive_logistic_top1_correctness",
        "probability_threshold": threshold,
        "accepted": int(accepted.sum()),
        "refused": int((~accepted).sum()),
        "validation_f1": float(confirmation["f1"]),
        "validation_tnr": float(confirmation["tnr"]),
        "validation_official_score": float(confirmation["official_score"]),
    }
    return accepted, probability.astype(np.float32), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--query-csv", type=Path, default=Path("dataset/test_query.csv"))
    parser.add_argument("--gallery-csv", type=Path, default=Path("dataset/test_gallery.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/score_optimized"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--release-config", type=Path,
        default=Path("configs/score_optimized.json"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--dino-size", type=int, choices=(224, 280), default=None,
        help="override the release-config DINOv2 crop size",
    )
    parser.add_argument(
        "--preprocessing",
        choices=(
            "cuda_jpeg", "cuda_roi", "gpu_shared", "pillow",
            "pillow_shared", "fast_decode", "opencv",
        ),
        default=None,
        help="override the release-config crop/resize backend",
    )
    args = parser.parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("score-optimized release requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query, gallery = pd.read_csv(args.query_csv), pd.read_csv(args.gallery_csv)
    release = json.loads(args.release_config.read_text(encoding="utf-8"))
    profile = release.get("profile", "quality")
    if profile not in {"quality", "speed"}:
        raise ValueError(f"unknown release profile: {profile}")
    runtime = release.get("runtime", {})
    preprocessing = args.preprocessing or runtime.get("preprocessing", "pillow")
    dino_size = args.dino_size or int(runtime.get("dino_input_size", 280))
    include_clip = profile == "quality"
    frame = _metadata(query, gallery)
    arrays, extraction = extract_features(
        frame, args.images_dir, args.weights_dir / "release",
        args.batch_size, args.workers, preprocessing=preprocessing,
        dino_size=dino_size, include_clip=include_clip,
    )
    ranking_started = time.perf_counter()
    order, score, fused = rank_gallery(
        frame,
        arrays,
        args.weights_dir,
        release,
        include_clip=include_clip,
        colors=arrays["colors"],
    )
    ranking_seconds = time.perf_counter() - ranking_started
    query_ids = query.image_id.astype(str).to_numpy(dtype=np.str_)
    gallery_ids = gallery.image_id.astype(str).to_numpy(dtype=np.str_)
    write_submission(
        query_ids, gallery_ids, order[:, :10], args.output_dir / "submission.csv"
    )
    np.save(args.output_dir / "embeddings.npy", fused, allow_pickle=False)

    accepted, confidence, refusal = calibrated_refusal(
        fused, len(query), order[:, 0], score, release
    )
    candidates = pd.DataFrame({
        "query_id": query_ids[accepted],
        "gallery_id": gallery_ids[order[accepted, 0]],
        "confidence": np.round(confidence[accepted], 8),
    })
    candidates.to_csv(args.output_dir / "candidates.csv", index=False, lineterminator="\n")
    index_path = args.output_dir / "gallery_index.npz"
    np.savez_compressed(
        index_path,
        image_ids=gallery_ids,
        embeddings=fused[len(query):].astype(np.float32, copy=False),
        release_config_sha256=np.asarray(sha256_file(args.release_config)),
    )
    reranker_names = (
        "dino_token_cross_top50.pt",
        "strict_family_gnn_smoothap.pt",
        "dino_top25_verifier.pt",
        "family_lambdarank_appearance_lbs.joblib",
        "dino_patch_matcher.joblib",
        "dino_patch_matcher_10x10_lbs.joblib",
        "modern_gallery_linker_lossbranch.joblib",
    )
    if include_clip:
        reranker_names += (
            "clip_deformable_cross_encoder_no_tta.pt",
            "oof_clip_lambdarank_score_optimized_no_dino_tta.joblib",
        )
    backbone_names = (
        "convnext_global_parts.ts",
        "osnet_loss_branch_global_parts.ts",
        "dinov2_vehicle_cls_tokens.ts",
    ) + (("clip_vehicle_global_tokens.ts",) if include_clip else ())
    weight_files = [args.weights_dir / "release" / name for name in backbone_names] + [
        args.weights_dir / name for name in reranker_names
    ]
    total_weight_bytes = sum(path.stat().st_size for path in weight_files)
    if total_weight_bytes > 2_000_000_000:
        raise RuntimeError("release inference weights exceed the 2 GB hard limit")
    manifest = {
        "method": f"{len(backbone_names)}-backbone single-pass strict gallery-aware reranker",
        "profile": profile,
        "validation": release["validation"],
        "uses_other_queries": False,
        "uses_camera_at_inference": False,
        "uses_csv_order": False,
        "query_count": len(query), "gallery_count": len(gallery),
        "extraction": extraction, "ranking_seconds": ranking_seconds,
        "weights": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in weight_files
        },
        "total_weight_bytes": total_weight_bytes,
        "release_config_sha256": sha256_file(args.release_config),
        "refusal": refusal,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                args.output_dir / "submission.csv",
                args.output_dir / "embeddings.npy",
                args.output_dir / "candidates.csv",
                index_path,
            )
        },
    }
    save_json(manifest, args.output_dir / "inference_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
