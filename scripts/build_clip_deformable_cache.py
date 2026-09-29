"""Build geometric CLIP-token evidence for the resulting final top-25."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.train_dino_patch_matcher import _load
from scripts.train_streaming_top25_verifier import _normalize


DEVICE = torch.device("cuda")
STACK = Path(os.environ.get(
    "TOP25_CACHE",
    "outputs/retrieval_v2/lbs_lambdarank_top25_val.npz",
))
TOKENS = Path(os.environ.get(
    "CLIP_VAL_TOKENS", "outputs/expert_fusion/cache/clip_vehicle_stage2_val_tokens.npy"
))
TOKEN_IDS = Path(os.environ.get(
    "CLIP_VAL_TOKEN_IDS",
    "outputs/expert_fusion/cache/clip_vehicle_stage2_val_token_ids.npy",
))
CLIP = Path(os.environ.get(
    "CLIP_METRIC_CACHE",
    "outputs/expert_fusion/cache/clip_vitb16_vehicle_metric_val_stage2.npz",
))
OUTPUT = Path(os.environ.get(
    "CLIP_DEFORMABLE_CACHE",
    "outputs/retrieval_v2/clip_deformable_lbs_lambdarank_top25_val.npz",
))
GRID = 7
TOP = 25


def _summary(values):
    count = values.shape[1]
    top4 = values.topk(min(4, count), dim=1).values.mean(1)
    top16 = values.topk(min(16, count), dim=1).values.mean(1)
    bottom4 = (-values).topk(min(4, count), dim=1).values.neg().mean(1)
    return torch.stack((
        values.mean(1), values.std(1), values.amin(1), values.amax(1),
        top4, top16, bottom4,
    ), dim=1)


def _masks():
    index = torch.arange(GRID * GRID, device=DEVICE)
    row, column = index // GRID, index % GRID
    qr, gr = row[:, None], row[None, :]
    qc, gc = column[:, None], column[None, :]
    flip_qc = GRID - 1 - qc
    return {
        "vertical0": (qr == gr),
        "vertical1": (torch.abs(qr - gr) <= 1),
        "local1": (torch.maximum(torch.abs(qr - gr), torch.abs(qc - gc)) <= 1),
        "local2": (torch.maximum(torch.abs(qr - gr), torch.abs(qc - gc)) <= 2),
        "flip_local1": (
            torch.maximum(torch.abs(qr - gr), torch.abs(flip_qc - gc)) <= 1
        ),
        "flip_local2": (
            torch.maximum(torch.abs(qr - gr), torch.abs(flip_qc - gc)) <= 2
        ),
        "quadrant": ((qr // 4 == gr // 4) & (qc // 4 == gc // 4)),
        "flip_quadrant": (
            (qr // 4 == gr // 4) & (flip_qc // 4 == gc // 4)
        ),
    }


@torch.inference_mode()
def _features(tokens, query, gallery, batch_size=256):
    masks = _masks()
    index = torch.arange(GRID * GRID, device=DEVICE)
    row, column = index // GRID, index % GRID
    output = []
    for start in range(0, len(query), batch_size):
        qi, gi = query[start:start + batch_size], gallery[start:start + batch_size]
        q = torch.from_numpy(np.asarray(tokens[qi])).to(DEVICE, non_blocking=True)
        g = torch.from_numpy(np.asarray(tokens[gi])).to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if q.shape[1] == GRID * GRID:
                # Production extraction already pools the single CLIP pass to
                # 7x7 so the cache is four times smaller.
                pass
            elif q.shape[1] == 14 * 14:
                q = F.avg_pool2d(
                    q.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2), 2
                ).permute(0, 2, 3, 1).flatten(1, 2)
                g = F.avg_pool2d(
                    g.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2), 2
                ).permute(0, 2, 3, 1).flatten(1, 2)
            else:
                raise ValueError(
                    f"expected 49 or 196 CLIP tokens, got {q.shape[1]}"
                )
            q, g = F.normalize(q.float(), dim=2), F.normalize(g.float(), dim=2)
            correlation = torch.bmm(q, g.transpose(1, 2))

        flat = correlation.flatten(1)
        values = [
            _summary(flat),
            _summary(correlation.amax(2)),
            _summary(correlation.amax(1)),
            _summary(correlation.diagonal(dim1=1, dim2=2)),
            _summary(correlation[:, index, index.reshape(GRID, GRID).flip(1).flatten()]),
        ]
        for mask in masks.values():
            matched = correlation.masked_fill(~mask[None], -torch.inf).amax(2)
            values.append(_summary(matched))

        q_to_g = correlation.argmax(2)
        g_to_q = correlation.argmax(1)
        back = torch.gather(g_to_q, 1, q_to_g)
        mutual = back == index[None]
        q_best = correlation.amax(2)
        mutual_count = mutual.float().mean(1)
        mutual_mean = (q_best * mutual).sum(1) / mutual.sum(1).clamp_min(1)

        matched_row = row[q_to_g]
        matched_column = column[q_to_g]
        delta_row = (matched_row - row[None]).float()
        delta_column = (matched_column - column[None]).float()
        flip_column = (matched_column + column[None] - (GRID - 1)).float()
        geometry = torch.stack((
            mutual_count,
            mutual_mean,
            delta_row.abs().mean(1),
            delta_row.std(1),
            delta_column.abs().mean(1),
            delta_column.std(1),
            flip_column.abs().mean(1),
            flip_column.std(1),
            torch.minimum(delta_column.std(1), flip_column.std(1)),
        ), dim=1)
        values.append(geometry)
        output.append(torch.cat(values, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def _z(values):
    return (values - values.mean(1, keepdims=True)) / (
        values.std(1, keepdims=True) + 1e-6
    )


def main():
    if OUTPUT.is_file():
        print(f"cache already exists: {OUTPUT}", flush=True)
        return
    frame = pd.read_csv("splits/val.csv")
    ids = np.load(TOKEN_IDS, allow_pickle=False)
    expected = frame.image_id.astype(str).to_numpy(dtype=np.str_)
    if not np.array_equal(ids, expected):
        raise RuntimeError("unaligned CLIP token cache")
    tokens = np.load(TOKENS, mmap_mode="r", allow_pickle=False)
    clip = _normalize(_load(CLIP, frame, "embeddings"))
    arrays = {}
    with np.load(STACK, allow_pickle=False) as stack:
        for position, seed in enumerate(TUNE_SEEDS + CONFIRM_SEEDS, start=1):
            print(f"deformable CLIP cache {position}/10 seed={seed}", flush=True)
            prefix = f"s{seed}_"
            qi = stack[prefix + "qi"].astype(np.int64)
            gi = stack[prefix + "gi"].astype(np.int64)
            candidate = stack[prefix + "candidate"].astype(np.int64)
            query = np.repeat(qi, TOP)
            gallery = gi[candidate.reshape(-1)]
            geometric = _features(tokens, query, gallery).reshape(len(qi), TOP, -1)
            clip_similarity = np.sum(
                clip[qi, None] * clip[gi[candidate]], axis=2
            ).astype(np.float32)
            base = stack[prefix + "base"].astype(np.float32)
            rank = np.broadcast_to(
                np.linspace(0.0, 1.0, TOP, dtype=np.float32)[None, :, None],
                (len(qi), TOP, 1),
            )
            raw = np.concatenate((
                base[..., None],
                clip_similarity[..., None],
                geometric,
            ), axis=2)
            feature = np.concatenate((
                raw,
                _z(raw),
                raw - raw[:, :1],
                rank,
            ), axis=2).astype(np.float16)
            for name, value in (
                ("feature", feature),
                ("label", stack[prefix + "label"]),
                ("valid", stack[prefix + "valid"]),
                ("candidate", candidate.astype(np.int16)),
                ("base", base),
                ("qi", qi.astype(np.int16)),
                ("gi", gi.astype(np.int16)),
            ):
                arrays[prefix + name] = value
    arrays["seeds"] = np.asarray(TUNE_SEEDS + CONFIRM_SEEDS, dtype=np.int32)
    arrays["feature_dim"] = np.asarray(arrays[f"s{TUNE_SEEDS[0]}_feature"].shape[2])
    np.savez(OUTPUT, **arrays)
    print({
        "cache": str(OUTPUT),
        "feature_dim": int(arrays["feature_dim"]),
        "bytes": OUTPUT.stat().st_size,
    }, flush=True)


if __name__ == "__main__":
    main()
