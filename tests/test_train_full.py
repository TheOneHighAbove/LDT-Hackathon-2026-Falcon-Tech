from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F

from src.engine import build_inference_checkpoint
from src.train_full import _parse_args, run_full_data_training


class TinyReIDModel(nn.Module):
    """Small model exposing the same training contract as VehicleReIDModel."""

    embedding_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.pool = nn.Identity()
        self.projection = nn.Linear(4, self.embedding_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(self.embedding_dim)

    def forward(self, images, *, return_dict=False):
        features = self.projection(self.backbone(images))
        bn_features = self.bnneck(features)
        embeddings = F.normalize(bn_features.float(), dim=1)
        if return_dict:
            return {
                "embeddings": embeddings,
                "features": features,
                "bn_features": bn_features,
            }
        return embeddings

    def get_config(self):
        return {
            "backbone_name": "tiny-test-model",
            "embedding_dim": self.embedding_dim,
            "pretrained": False,
            "backbone_kwargs": {},
        }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    images = root / "dataset" / "images"
    weights = root / "weights"
    configs = root / "configs"
    images.mkdir(parents=True)
    weights.mkdir()
    configs.mkdir()

    rows = []
    for index, (vehicle_id, camera_id) in enumerate(
        [(10, 1), (10, 2), (20, 1), (20, 2)]
    ):
        image_id = f"car-{index}"
        Image.new(
            "RGB", (40, 24), color=(30 + index * 20, 80, 140)
        ).save(images / f"{image_id}.jpg")
        rows.append(
            {
                "image_id": image_id,
                "vehicle_id": vehicle_id,
                "camera_id": camera_id,
                "x": 0,
                "y": 0,
                "w": 40,
                "h": 24,
            }
        )
    pd.DataFrame(rows).to_csv(root / "dataset" / "train.csv", index=False)

    config = {
        "seed": 7,
        "paths": {
            "images_dir": "dataset/images",
            "train_csv": "dataset/train.csv",
            "weights_dir": "weights",
            "outputs_dir": "outputs",
        },
        "model": {
            "backbone": "tiny-test-model",
            "embedding_dim": 4,
            "input_size": 32,
            "channels_last": False,
        },
        "data": {
            # Geometry here intentionally disagrees with the selected artifact:
            # full-data continuation must inherit inference preprocessing.
            "bbox_padding": 0.25,
            "resize_mode": "letterbox",
            "num_workers": 0,
            "pin_memory": False,
            "train_crop_scale": [0.9, 1.0],
            "train_crop_ratio": [0.9, 1.1],
            "horizontal_flip_probability": 0.0,
            "random_erasing_probability": 0.0,
        },
        "training": {
            "epochs": 5,
            "identities_per_batch": 2,
            "instances_per_identity": 2,
            "lr_backbone": 0.001,
            "lr_head": 0.002,
            "lr_classifier": 0.003,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "min_lr_ratio": 0.1,
            "amp": False,
            "gradient_clip_norm": 5.0,
            "arcface_scale": 8.0,
            "arcface_margin": 0.2,
            "arcface_subcenters": 2,
            "label_smoothing": 0.0,
            "triplet_margin": 0.3,
            "triplet_margin_mode": "soft",
            "triplet_positive_mining": "cross_camera_preferred",
            "classification_weight": 1.0,
            "triplet_weight": 1.0,
        },
        "inference": {"tta_horizontal_flip": True},
    }
    config_path = configs / "test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    torch.manual_seed(123)
    initial_model = TinyReIDModel()
    checkpoint = build_inference_checkpoint(
        initial_model,
        input_size=32,
        bbox_padding=0.0,
        refusal_threshold=0.27,
        tta_horizontal_flip=True,
        calibration={"slope": 2.5, "intercept": -0.4},
        metrics={"retrieval": {"mAP": 0.42}, "open_set": {"f1": 0.31}},
        source={"epoch": 3, "experiment": "honest-split"},
    )
    checkpoint["preprocessing"]["contract_extension"] = {
        "preserve_unknown_fields": True
    }
    # These must be ignored: only model_state initializes the full-data run.
    checkpoint["optimizer_state"] = {"invalid": "old optimizer"}
    checkpoint["criterion_state"] = {"invalid": "old classifier"}
    init_path = weights / "init.pt"
    torch.save(checkpoint, init_path)
    return config_path, init_path


def test_full_data_smoke_copies_honest_metadata_and_saves_no_train_state(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, init_path = _write_fixture(tmp_path)
    monkeypatch.setattr(
        "src.train_full.create_model_from_config",
        lambda config, pretrained=False: TinyReIDModel(),
    )
    args = argparse.Namespace(
        config=config_path,
        init_checkpoint=init_path,
        device="cpu",
        epochs=1,
        max_train_batches=1,
    )

    summary = run_full_data_training(args)
    destination = tmp_path / "weights" / "full_data.pt"
    output = torch.load(destination, map_location="cpu", weights_only=True)

    assert summary["rows"] == 4
    assert summary["vehicle_ids"] == 2
    assert summary["epochs"] == 1
    assert len(summary["history"]) == 1
    assert output["refusal"] == {
        "similarity_threshold": 0.27,
        "calibration": {"slope": 2.5, "intercept": -0.4},
    }
    assert output["metrics"] == {
        "retrieval": {"mAP": 0.42},
        "open_set": {"f1": 0.31},
    }
    assert output["preprocessing"]["input_size"] == 32
    assert output["preprocessing"]["resize_mode"] == "direct"
    assert output["preprocessing"]["tta_horizontal_flip"] is True
    assert output["preprocessing"]["bbox_padding"] == 0.0
    assert output["preprocessing"]["contract_extension"] == {
        "preserve_unknown_fields": True
    }
    assert "optimizer_state" not in output
    assert "criterion_state" not in output

    source = output["source"]
    assert source["full_data_finetune"] is True
    assert source["full_data_rows"] == 4
    assert source["full_data_vehicle_ids"] == 2
    assert source["full_data_cameras"] == 2
    assert source["full_data_epochs"] == 1
    assert source["strict_model_state_initialization"] is True
    assert source["metrics_evaluated_after_full_data_finetune"] is False
    assert "pre-full-data" in source["metrics_provenance"]
    assert source["parent_source"]["experiment"] == "honest-split"
    assert source["config_snapshot"]["training"]["lr_classifier"] == 0.003
    assert sorted(path.name for path in (tmp_path / "weights").iterdir()) == [
        "full_data.pt",
        "init.pt",
    ]


def test_strict_initialization_rejects_architecture_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, init_path = _write_fixture(tmp_path)
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
    checkpoint["model_state"].pop("projection.weight")
    torch.save(checkpoint, init_path)
    monkeypatch.setattr(
        "src.train_full.create_model_from_config",
        lambda config, pretrained=False: TinyReIDModel(),
    )
    args = argparse.Namespace(
        config=config_path,
        init_checkpoint=init_path,
        device="cpu",
        epochs=1,
        max_train_batches=1,
    )
    with pytest.raises(RuntimeError, match="strictly compatible"):
        run_full_data_training(args)
    assert not (tmp_path / "weights" / "full_data.pt").exists()


def test_cli_exposes_reproducible_full_data_controls() -> None:
    args = _parse_args(
        [
            "--config",
            "configs/experiment.yaml",
            "--init-checkpoint",
            "weights/fold.pt",
            "--device",
            "cpu",
            "--epochs",
            "7",
        ]
    )
    assert args.config == Path("configs/experiment.yaml")
    assert args.init_checkpoint == Path("weights/fold.pt")
    assert args.device == "cpu"
    assert args.epochs == 7
