from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.losses import ReIDLoss
from src.train import train_one_epoch


class _TinyMetricModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)

    def forward(self, images, *, return_dict=False):
        features = self.projection(images)
        output = {
            "features": features,
            "embeddings": F.normalize(features.float(), dim=1),
        }
        return output if return_dict else output["embeddings"]


def test_train_one_epoch_resets_cross_batch_memory_at_epoch_boundary():
    model = _TinyMetricModel()
    criterion = ReIDLoss(3, 3, cross_batch_memory_capacity=16)
    # Simulate stale state left by a prior epoch.  The training entry point must
    # remove it before processing the new epoch.
    criterion(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([2]),
        camera_ids=torch.tensor([9]),
    )
    assert criterion.cross_batch_memory_count == 1

    batch = {
        "image": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ]
        ),
        "label": torch.tensor([0, 0, 1, 1]),
        "camera_id": torch.tensor([0, 1, 0, 1]),
    }
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(criterion.parameters()), lr=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = train_one_epoch(
        model,
        criterion,
        [batch],
        optimizer,
        scaler,
        device=torch.device("cpu"),
        amp=False,
        channels_last=False,
        gradient_clip_norm=0.0,
    )

    assert metrics["loss"] > 0.0
    # Exactly the current batch remains: the stale singleton was reset.
    assert criterion.cross_batch_memory_count == 4
