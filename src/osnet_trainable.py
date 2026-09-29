"""Trainable PyTorch reconstruction of the bundled vehicle OSNet-AIN.

The architecture follows the MIT-licensed torchreid OSNet-AIN implementation.
The bundled Open Model Zoo ONNX initializers are loaded by name, so no second
pretrained checkpoint or network access is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *, instance=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if instance
            else nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, *, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.layers = nn.Sequential(*(LightConv3x3(channels) for _ in range(depth)))

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 16, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // 16, channels, 1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        gate = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * gate


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, instance_inside=False):
        super().__init__()
        mid = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2 = nn.ModuleList(LightConvStream(mid, depth) for depth in range(1, 5))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels, bn=not instance_inside)
        self.downsample = (
            Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        )
        if instance_inside:
            # Capitalization intentionally matches the original/ONNX state keys.
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        stem = self.conv1(x)
        mixed = sum(self.gate(stream(stem)) for stream in self.conv2)
        mixed = self.conv3(mixed)
        if hasattr(self, "IN"):
            mixed = self.IN(mixed)
        return F.relu(mixed + identity)


class VehicleOSNetAIN(nn.Module):
    """Exact differentiable counterpart of ``vehicle-reid-0001``."""

    def __init__(self):
        super().__init__()
        self.input_IN = nn.InstanceNorm2d(3, affine=True)
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3, instance=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            OSBlock(64, 256, instance_inside=True),
            OSBlock(256, 256, instance_inside=True),
        )
        self.pool2 = nn.Sequential(Conv1x1(256, 256), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(
            OSBlock(256, 384, instance_inside=False),
            OSBlock(384, 384, instance_inside=True),
        )
        self.pool3 = nn.Sequential(Conv1x1(384, 384), nn.AvgPool2d(2, stride=2))
        self.conv4 = nn.Sequential(
            OSBlock(384, 512, instance_inside=True),
            OSBlock(512, 512, instance_inside=False),
        )
        self.conv5 = Conv1x1(512, 512)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.ModuleList(
            (
                nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256)),
                nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256)),
            )
        )

    def featuremaps(self, x):
        x = self.input_IN(x)
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        return self.conv5(self.conv4(x))

    def forward_branches(self, x):
        """Return the two native OSNet descriptors before concatenation.

        The bundled vehicle model already contains two independent 256-D
        projection heads.  Exposing them separately lets training assign
        complementary objectives to the heads (identity classification versus
        cross-camera metric learning) while preserving the exact legacy
        ``forward`` output and checkpoint layout.
        """

        pooled = self.global_avgpool(self.featuremaps(x)).flatten(1)
        return tuple(head(pooled) for head in self.fc)

    def forward(self, x):
        return torch.cat(self.forward_branches(x), dim=1)


def load_vehicle_osnet_onnx(path: str | Path) -> VehicleOSNetAIN:
    model = VehicleOSNetAIN()
    graph = onnx.load(str(path))
    initializers = {
        item.name: torch.from_numpy(np.array(numpy_helper.to_array(item), copy=True))
        for item in graph.graph.initializer
    }
    state = model.state_dict()
    missing = sorted(set(state) - set(initializers))
    extra = sorted(set(initializers) - set(state))
    if missing or extra:
        raise RuntimeError(
            f"ONNX/PyTorch OSNet state mismatch; missing={missing}, extra={extra}"
        )
    model.load_state_dict({key: initializers[key] for key in state}, strict=True)
    return model
