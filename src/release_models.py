"""Neural modules and inference adapters for the frozen release weights."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_GRID = 7


class ClipDeformableCrossEncoder(nn.Module):
    """Pairwise CLIP-token verifier used by the optional quality profile."""

    def __init__(self, input_dim: int = 768, dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128, bias=False),
            nn.GELU(),
            nn.Linear(128, dim, bias=False),
        )
        self.position = nn.Parameter(torch.zeros(1, 1, CLIP_GRID * CLIP_GRID, dim))
        nn.init.trunc_normal_(self.position, std=0.01)
        index = torch.arange(CLIP_GRID * CLIP_GRID)
        row, column = index // CLIP_GRID, index % CLIP_GRID
        direct = (
            (row[:, None] - row[None, :]).float().square()
            + (column[:, None] - column[None, :]).float().square()
        ) / (CLIP_GRID * CLIP_GRID)
        flip = (
            (row[:, None] - row[None, :]).float().square()
            + (
                (CLIP_GRID - 1 - column[:, None]) - column[None, :]
            ).float().square()
        ) / (CLIP_GRID * CLIP_GRID)
        self.register_buffer("direct_distance", direct)
        self.register_buffer("flip_distance", flip)
        self.geometry_strength = nn.Parameter(torch.tensor(3.0))
        feature_dim = 12 * dim + 17
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _aligned_features(
        query: torch.Tensor,
        gallery: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        aligned = torch.einsum("bkts,bksd->bktd", attention, gallery)
        expanded_query = query.expand(-1, gallery.shape[1], -1, -1)
        difference = torch.abs(expanded_query - aligned)
        product = expanded_query * aligned
        return torch.cat(
            (
                difference.mean(2),
                difference.amax(2),
                product.mean(2),
                product.amax(2),
            ),
            dim=2,
        )

    def forward(
        self,
        query: torch.Tensor,
        gallery: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        query_projected = F.normalize(
            self.projection(query)[:, None] + self.position, dim=3
        )
        gallery_projected = F.normalize(
            self.projection(gallery) + self.position, dim=3
        )
        correlation = torch.einsum(
            "bqtd,bksd->bkts", query_projected, gallery_projected
        )
        strength = F.softplus(self.geometry_strength)
        free_attention = F.softmax(10.0 * correlation, dim=3)
        direct_attention = F.softmax(
            10.0 * correlation
            - strength * self.direct_distance[None, None],
            dim=3,
        )
        flip_attention = F.softmax(
            10.0 * correlation - strength * self.flip_distance[None, None],
            dim=3,
        )
        visual = torch.cat(
            (
                self._aligned_features(
                    query_projected, gallery_projected, free_attention
                ),
                self._aligned_features(
                    query_projected, gallery_projected, direct_attention
                ),
                self._aligned_features(
                    query_projected, gallery_projected, flip_attention
                ),
            ),
            dim=2,
        )
        flat = correlation.flatten(2)
        row_best = correlation.amax(3)
        column_best = correlation.amax(2)
        index = torch.arange(CLIP_GRID * CLIP_GRID, device=query.device)
        flip_index = index.reshape(CLIP_GRID, CLIP_GRID).flip(1).flatten()
        direct = correlation[:, :, index, index]
        flipped = correlation[:, :, index, flip_index]
        scalar = torch.stack(
            (
                base,
                flat.mean(2),
                flat.std(2),
                flat.amax(2),
                flat.topk(16, dim=2).values.mean(2),
                row_best.mean(2),
                row_best.std(2),
                row_best.amin(2),
                row_best.amax(2),
                column_best.mean(2),
                column_best.std(2),
                direct.mean(2),
                direct.amax(2),
                flipped.mean(2),
                flipped.amax(2),
                (free_attention * correlation).sum(3).mean(2),
                torch.maximum(
                    (direct_attention * correlation).sum(3).mean(2),
                    (flip_attention * correlation).sum(3).mean(2),
                ),
            ),
            dim=2,
        )
        residual = 0.45 * torch.tanh(
            self.head(torch.cat((visual, scalar), dim=2)).squeeze(2)
        )
        return base + residual


class CrossImageTransformer(nn.Module):
    """Local ConvNeXt/OSNet cross-image verifier."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, dim, bias=False)
        )
        self.osnet = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, dim, bias=False)
        )
        self.position = nn.Parameter(torch.randn(25, dim) * 0.02)
        self.source = nn.Parameter(torch.randn(2, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim,
            4,
            192,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, 1, enable_nested_tensor=False
        )
        self.cross = nn.MultiheadAttention(
            dim, 4, dropout=0.10, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 4 + 22),
            nn.Linear(dim * 4 + 22, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _tokens(
        self, conv: torch.Tensor, osnet: torch.Tensor
    ) -> torch.Tensor:
        values = torch.cat(
            (
                self.conv(conv) + self.source[0],
                self.osnet(osnet) + self.source[1],
            ),
            dim=1,
        )
        return self.encoder(values + self.position)

    def forward(
        self,
        query_conv: torch.Tensor,
        query_osnet: torch.Tensor,
        gallery_conv: torch.Tensor,
        gallery_osnet: torch.Tensor,
        global_similarity: torch.Tensor,
    ) -> torch.Tensor:
        query = self._tokens(query_conv, query_osnet)
        gallery = self._tokens(gallery_conv, gallery_osnet)
        query_cross, _ = self.cross(
            query, gallery, gallery, need_weights=False
        )
        gallery_cross, _ = self.cross(
            gallery, query, query, need_weights=False
        )
        query_cross = self.cross_norm(query + query_cross)
        gallery_cross = self.cross_norm(gallery + gallery_cross)

        query_normalized = F.normalize(query, dim=2)
        gallery_normalized = F.normalize(gallery, dim=2)
        correlation = torch.einsum(
            "bqd,bgd->bqg", query_normalized, gallery_normalized
        )
        top = correlation.flatten(1).topk(16, dim=1).values
        query_best = correlation.amax(2)
        gallery_best = correlation.amax(1)
        correlation_stats = torch.cat(
            (
                top,
                query_best.mean(1, keepdim=True),
                query_best.std(1, keepdim=True),
                query_best.amax(1, keepdim=True),
                gallery_best.mean(1, keepdim=True),
                gallery_best.std(1, keepdim=True),
                gallery_best.amax(1, keepdim=True),
            ),
            dim=1,
        )
        query_match = torch.einsum(
            "bqg,bgd->bqd", F.softmax(10 * correlation, dim=2), gallery
        )
        gallery_match = torch.einsum(
            "bgq,bqd->bgd",
            F.softmax(10 * correlation.transpose(1, 2), dim=2),
            query,
        )
        local = torch.cat(
            (
                torch.abs(query_cross - query_match).mean(1),
                (query_cross * query_match).mean(1),
                torch.abs(gallery_cross - gallery_match).mean(1),
                (gallery_cross * gallery_match).mean(1),
            ),
            dim=1,
        )
        learned = self.head(
            torch.cat((local, correlation_stats), dim=1)
        ).squeeze(1)
        return global_similarity + 0.35 * torch.tanh(learned)


class DinoTokenCrossMatcher(nn.Module):
    """Lightweight learned cross-attention over frozen DINO patch tokens."""

    def __init__(
        self, input_dim: int = 768, dim: int = 48, token_count: int = 25
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.position = nn.Parameter(torch.zeros(1, 1, token_count, dim))
        nn.init.trunc_normal_(self.position, std=0.01)
        feature_dim = 8 * dim + 9
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        gallery: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        query_projected = self.projection(query)[:, None]
        gallery_projected = self.projection(gallery)
        query_projected = F.normalize(
            query_projected + self.position, dim=-1
        )
        gallery_projected = F.normalize(
            gallery_projected + self.position, dim=-1
        )
        correlation = torch.einsum(
            "bqtd,bksd->bkts", query_projected, gallery_projected
        )
        query_aligned = torch.einsum(
            "bkts,bksd->bktd",
            F.softmax(10.0 * correlation, dim=3),
            gallery_projected,
        )
        gallery_aligned = torch.einsum(
            "bkst,bqtd->bksd",
            F.softmax(10.0 * correlation.transpose(2, 3), dim=3),
            query_projected,
        )
        expanded_query = query_projected.expand(
            -1, gallery.shape[1], -1, -1
        )
        query_abs = torch.abs(expanded_query - query_aligned)
        query_product = expanded_query * query_aligned
        gallery_abs = torch.abs(gallery_projected - gallery_aligned)
        gallery_product = gallery_projected * gallery_aligned
        visual = torch.cat(
            (
                query_abs.mean(2),
                query_abs.amax(2),
                query_product.mean(2),
                query_product.amax(2),
                gallery_abs.mean(2),
                gallery_abs.amax(2),
                gallery_product.mean(2),
                gallery_product.amax(2),
            ),
            dim=2,
        )
        flat = correlation.flatten(2)
        row_best = correlation.amax(3)
        column_best = correlation.amax(2)
        diagonal = correlation.diagonal(dim1=2, dim2=3)
        scalar = torch.stack(
            (
                base,
                flat.mean(2),
                flat.amax(2),
                flat.topk(16, dim=2).values.mean(2),
                row_best.mean(2),
                row_best.amin(2),
                column_best.mean(2),
                column_best.amin(2),
                diagonal.mean(2),
            ),
            dim=2,
        )
        residual = 0.30 * torch.tanh(
            self.head(torch.cat((visual, scalar), dim=2)).squeeze(2)
        )
        return base + residual


class FamilyGraphReranker(nn.Module):
    """Static-gallery graph reranker for one independent query shortlist."""

    def __init__(self, feature_dim: int, hidden: int = 80, layers: int = 3):
        super().__init__()
        self.node = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden),
        )
        self.edge = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(4 * hidden),
                    nn.Linear(4 * hidden, 2 * hidden),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(2 * hidden, hidden),
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + feature_dim),
            nn.Linear(hidden + feature_dim, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, features: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        _, count = features.shape[:2]
        hidden = self.node(features)
        fused = relation[..., 0]
        eye = torch.eye(
            count, dtype=torch.bool, device=features.device
        )[None]
        search = fused.masked_fill(eye, -torch.inf)
        neighbors = search.topk(min(8, count - 1), dim=2).indices
        mask = torch.zeros_like(fused, dtype=torch.bool)
        mask.scatter_(2, neighbors, True)
        mask |= relation[..., 3] > 0.5
        mask |= eye
        edge = self.edge(relation).squeeze(3).masked_fill(~mask, -torch.inf)
        attention = F.softmax(edge, dim=2)
        for update, norm in zip(self.updates, self.norms, strict=True):
            message = torch.bmm(attention, hidden)
            delta = update(
                torch.cat(
                    (
                        hidden,
                        message,
                        torch.abs(hidden - message),
                        hidden * message,
                    ),
                    dim=2,
                )
            )
            hidden = norm(hidden + delta)
        residual = 0.35 * torch.tanh(
            self.head(torch.cat((hidden, features), dim=2)).squeeze(2)
        )
        return features[..., 0] + residual


@torch.inference_mode()
def predict_clip_cross(
    model: ClipDeformableCrossEncoder,
    protocol: dict,
    tokens: np.ndarray,
    batch: int = 6,
) -> np.ndarray:
    model.eval()
    output = []
    rows = np.arange(len(protocol["qi"]))
    proxy = {
        "qi": protocol["qi"],
        "gi": protocol["gi"],
        "score": np.zeros(
            (len(rows), len(protocol["gi"])), dtype=np.float32
        ),
    }
    np.put_along_axis(
        proxy["score"], protocol["candidate"], protocol["base"], axis=1
    )
    for start in range(0, len(rows), batch):
        current = rows[start : start + batch]
        candidate = protocol["candidate"][current]
        query_index = proxy["qi"][current]
        gallery_index = proxy["gi"][candidate]
        base = np.take_along_axis(
            proxy["score"][current], candidate, axis=1
        )
        inputs = (
            torch.from_numpy(
                np.asarray(tokens[query_index]).astype(np.float32)
            ).to(DEVICE, non_blocking=True),
            torch.from_numpy(
                np.asarray(tokens[gallery_index]).astype(np.float32)
            ).to(DEVICE, non_blocking=True),
            torch.from_numpy(base).to(DEVICE, non_blocking=True),
        )
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            value = model(*inputs)
        output.append(value.float().cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def predict_dino_token(
    model: DinoTokenCrossMatcher,
    protocol: dict,
    tokens: np.ndarray,
    query_batch: int = 12,
) -> np.ndarray:
    model.eval()
    result = []
    all_rows = np.arange(len(protocol["qi"]))
    for start in range(0, len(all_rows), query_batch):
        rows = all_rows[start : start + query_batch]
        candidate = protocol["candidates"][rows]
        query_index = protocol["qi"][rows]
        gallery_index = protocol["gi"][candidate]
        base = np.take_along_axis(
            protocol["score"][rows], candidate, axis=1
        )
        inputs = (
            torch.from_numpy(tokens[query_index]).to(
                DEVICE, non_blocking=True
            ),
            torch.from_numpy(tokens[gallery_index]).to(
                DEVICE, non_blocking=True
            ),
            torch.from_numpy(base).to(DEVICE, non_blocking=True),
        )
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            value = model(*inputs)
        result.append(value.float().cpu().numpy())
    return np.concatenate(result)


@torch.inference_mode()
def predict_cross_image(
    model: CrossImageTransformer,
    conv: np.ndarray,
    osnet: np.ndarray,
    embedding: np.ndarray,
    query: np.ndarray,
    gallery: np.ndarray,
    batch: int = 512,
) -> np.ndarray:
    model.eval()
    result = []
    for start in range(0, len(query), batch):
        query_index = query[start : start + batch]
        gallery_index = gallery[start : start + batch]
        similarity = np.sum(
            embedding[query_index] * embedding[gallery_index], axis=1
        ).astype(np.float32)
        inputs = (
            torch.from_numpy(conv[query_index]).to(
                DEVICE, non_blocking=True
            ),
            torch.from_numpy(osnet[query_index]).to(
                DEVICE, non_blocking=True
            ),
            torch.from_numpy(conv[gallery_index]).to(
                DEVICE, non_blocking=True
            ),
            torch.from_numpy(osnet[gallery_index]).to(
                DEVICE, non_blocking=True
            ),
            torch.from_numpy(similarity).to(DEVICE, non_blocking=True),
        )
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            value = model(*inputs)
        result.append(value.float().cpu().numpy())
    return np.concatenate(result)


@torch.inference_mode()
def predict_family_graph(
    model: FamilyGraphReranker, episode: dict, batch: int = 32
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(episode["qi"]), batch):
        rows = np.arange(start, min(start + batch, len(episode["qi"])))
        inputs = (
            torch.from_numpy(episode["features"][rows]).to(
                DEVICE, non_blocking=True
            ),
            torch.from_numpy(
                episode["relation"][rows].astype(np.float32)
            ).to(DEVICE, non_blocking=True),
        )
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            value = model(*inputs)
        output.append(value.float().cpu().numpy())
    return np.concatenate(output)


__all__ = [
    "ClipDeformableCrossEncoder",
    "CrossImageTransformer",
    "DinoTokenCrossMatcher",
    "FamilyGraphReranker",
    "predict_clip_cross",
    "predict_cross_image",
    "predict_dino_token",
    "predict_family_graph",
]
