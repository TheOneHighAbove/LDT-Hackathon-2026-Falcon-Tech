"""Pure feature helpers used by the score-optimized release runtime.

This module intentionally has no dependency on training or experiment scripts.
The functions preserve the numerical operations used to produce the locked
release metrics while keeping deployment imports small and auditable.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_GRID = 7
PATCH_GRID = 5


def color_descriptor(crop: Image.Image) -> np.ndarray:
    """Return the locked five-region HSV histogram for one tight vehicle crop."""

    array = np.asarray(crop.resize((64, 64)).convert("HSV"), dtype=np.uint8)
    regions = (
        array,
        array[:32, :32],
        array[:32, 32:],
        array[32:, :32],
        array[32:, 32:],
    )
    values: list[np.ndarray] = []
    for region in regions:
        for channel, bins in ((0, 16), (1, 8), (2, 8)):
            # The bins are uniform over the complete uint8 range, therefore
            # bincount is exactly equivalent to np.histogram here and avoids
            # repeatedly constructing generic histogram edges in the hot path.
            indices = (
                region[..., channel].astype(np.int32, copy=False) * bins // 256
            )
            histogram = np.bincount(
                indices.reshape(-1), minlength=bins
            ).astype(np.float32)
            histogram /= max(float(histogram.sum()), 1.0)
            values.append(np.sqrt(histogram))
    descriptor = np.concatenate(values)
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    return descriptor.astype(np.float32, copy=False)


def dino_part_pool(tokens: torch.Tensor, bands: int) -> torch.Tensor:
    """Pool square DINO tokens into horizontal part descriptors."""

    side = int(round(tokens.shape[1] ** 0.5))
    spatial = tokens.reshape(len(tokens), side, side, tokens.shape[2])
    chunks = torch.tensor_split(spatial, bands, dim=1)
    values = [
        F.normalize(chunk.mean(dim=(1, 2)).float(), dim=1)
        for chunk in chunks
    ]
    return F.normalize(torch.cat(values, dim=1), dim=1)


def patch_features(
    tokens: np.ndarray,
    fused: np.ndarray,
    osnet: np.ndarray,
    dino: np.ndarray,
    query: np.ndarray,
    gallery: np.ndarray,
    batch: int = 1024,
    grid_size: int | None = None,
) -> np.ndarray:
    """Build the locked DINO local-matching features for query/gallery pairs."""

    grid_size = PATCH_GRID if grid_size is None else grid_size
    output = []
    for start in range(0, len(query), batch):
        query_index = query[start : start + batch]
        gallery_index = gallery[start : start + batch]
        query_tokens = torch.from_numpy(tokens[query_index]).to(
            DEVICE, non_blocking=True
        )
        gallery_tokens = torch.from_numpy(tokens[gallery_index]).to(
            DEVICE, non_blocking=True
        )
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            correlation = torch.bmm(
                query_tokens, gallery_tokens.transpose(1, 2)
            ).float()
        top = correlation.flatten(1).topk(32, dim=1).values
        best_count = min(32, grid_size * grid_size)
        query_best = correlation.amax(2).topk(best_count, dim=1).values
        gallery_best = correlation.amax(1).topk(best_count, dim=1).values
        aligned = correlation.diagonal(dim1=1, dim2=2)
        token_count = grid_size * grid_size
        flip_index = (
            torch.arange(token_count, device=DEVICE)
            .reshape(grid_size, grid_size)
            .flip(1)
            .flatten()
        )
        flipped = correlation[
            :, torch.arange(token_count, device=DEVICE), flip_index
        ]
        grid = correlation.reshape(
            -1, grid_size, grid_size, grid_size, grid_size
        )
        same_row = torch.stack(
            [grid[:, row, :, row, :].amax(2) for row in range(grid_size)], 1
        ).flatten(1)
        visual = torch.cat(
            (top, query_best, gallery_best, aligned, flipped, same_row), dim=1
        ).cpu().numpy()
        scalar = np.stack(
            (
                np.sum(fused[query_index] * fused[gallery_index], axis=1),
                np.sum(osnet[query_index] * osnet[gallery_index], axis=1),
                np.sum(dino[query_index] * dino[gallery_index], axis=1),
            ),
            axis=1,
        ).astype(np.float32)
        output.append(np.concatenate((scalar, visual), axis=1))
    return np.concatenate(output)


def _clip_summary(values: torch.Tensor) -> torch.Tensor:
    count = values.shape[1]
    top4 = values.topk(min(4, count), dim=1).values.mean(1)
    top16 = values.topk(min(16, count), dim=1).values.mean(1)
    bottom4 = (-values).topk(min(4, count), dim=1).values.neg().mean(1)
    return torch.stack(
        (
            values.mean(1),
            values.std(1),
            values.amin(1),
            values.amax(1),
            top4,
            top16,
            bottom4,
        ),
        dim=1,
    )


def _clip_masks() -> dict[str, torch.Tensor]:
    index = torch.arange(CLIP_GRID * CLIP_GRID, device=DEVICE)
    row, column = index // CLIP_GRID, index % CLIP_GRID
    query_row, gallery_row = row[:, None], row[None, :]
    query_column, gallery_column = column[:, None], column[None, :]
    flipped_query_column = CLIP_GRID - 1 - query_column
    return {
        "vertical0": query_row == gallery_row,
        "vertical1": torch.abs(query_row - gallery_row) <= 1,
        "local1": torch.maximum(
            torch.abs(query_row - gallery_row),
            torch.abs(query_column - gallery_column),
        )
        <= 1,
        "local2": torch.maximum(
            torch.abs(query_row - gallery_row),
            torch.abs(query_column - gallery_column),
        )
        <= 2,
        "flip_local1": torch.maximum(
            torch.abs(query_row - gallery_row),
            torch.abs(flipped_query_column - gallery_column),
        )
        <= 1,
        "flip_local2": torch.maximum(
            torch.abs(query_row - gallery_row),
            torch.abs(flipped_query_column - gallery_column),
        )
        <= 2,
        "quadrant": (
            (query_row // 4 == gallery_row // 4)
            & (query_column // 4 == gallery_column // 4)
        ),
        "flip_quadrant": (
            (query_row // 4 == gallery_row // 4)
            & (flipped_query_column // 4 == gallery_column // 4)
        ),
    }


def clip_geometry_features(
    tokens: np.ndarray,
    query: np.ndarray,
    gallery: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Build the optional quality-profile CLIP geometry features."""

    masks = _clip_masks()
    index = torch.arange(CLIP_GRID * CLIP_GRID, device=DEVICE)
    row, column = index // CLIP_GRID, index % CLIP_GRID
    output = []
    for start in range(0, len(query), batch_size):
        query_index = query[start : start + batch_size]
        gallery_index = gallery[start : start + batch_size]
        query_tokens = torch.from_numpy(np.asarray(tokens[query_index])).to(
            DEVICE, non_blocking=True
        )
        gallery_tokens = torch.from_numpy(np.asarray(tokens[gallery_index])).to(
            DEVICE, non_blocking=True
        )
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            if query_tokens.shape[1] == CLIP_GRID * CLIP_GRID:
                pass
            elif query_tokens.shape[1] == 14 * 14:
                query_tokens = F.avg_pool2d(
                    query_tokens.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2),
                    2,
                ).permute(0, 2, 3, 1).flatten(1, 2)
                gallery_tokens = F.avg_pool2d(
                    gallery_tokens.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2),
                    2,
                ).permute(0, 2, 3, 1).flatten(1, 2)
            else:
                raise ValueError(
                    "expected 49 or 196 CLIP tokens, got "
                    f"{query_tokens.shape[1]}"
                )
            query_tokens = F.normalize(query_tokens.float(), dim=2)
            gallery_tokens = F.normalize(gallery_tokens.float(), dim=2)
            correlation = torch.bmm(
                query_tokens, gallery_tokens.transpose(1, 2)
            )

        flat = correlation.flatten(1)
        values = [
            _clip_summary(flat),
            _clip_summary(correlation.amax(2)),
            _clip_summary(correlation.amax(1)),
            _clip_summary(correlation.diagonal(dim1=1, dim2=2)),
            _clip_summary(
                correlation[
                    :, index, index.reshape(CLIP_GRID, CLIP_GRID).flip(1).flatten()
                ]
            ),
        ]
        for mask in masks.values():
            matched = correlation.masked_fill(~mask[None], -torch.inf).amax(2)
            values.append(_clip_summary(matched))

        query_to_gallery = correlation.argmax(2)
        gallery_to_query = correlation.argmax(1)
        back = torch.gather(gallery_to_query, 1, query_to_gallery)
        mutual = back == index[None]
        query_best = correlation.amax(2)
        mutual_count = mutual.float().mean(1)
        mutual_mean = (query_best * mutual).sum(1) / mutual.sum(1).clamp_min(1)
        matched_row = row[query_to_gallery]
        matched_column = column[query_to_gallery]
        delta_row = (matched_row - row[None]).float()
        delta_column = (matched_column - column[None]).float()
        flip_column = (
            matched_column + column[None] - (CLIP_GRID - 1)
        ).float()
        values.append(
            torch.stack(
                (
                    mutual_count,
                    mutual_mean,
                    delta_row.abs().mean(1),
                    delta_row.std(1),
                    delta_column.abs().mean(1),
                    delta_column.std(1),
                    flip_column.abs().mean(1),
                    flip_column.std(1),
                    torch.minimum(delta_column.std(1), flip_column.std(1)),
                ),
                dim=1,
            )
        )
        output.append(
            torch.cat(values, dim=1).cpu().numpy().astype(np.float32)
        )
    return np.concatenate(output)


__all__ = [
    "clip_geometry_features",
    "color_descriptor",
    "dino_part_pool",
    "patch_features",
]
