"""Retrain the streaming local verifier on the adapted DINO+OSNet shortlist."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official, _protocols
from scripts.train_contextual_multiquery_reranker import DEVICE
from scripts.train_cross_image_transformer import CrossImageTransformer, _load
from scripts.train_streaming_top25_verifier import (
    TRAIN_SEEDS,
    _local_for_protocol,
    _make_train_protocol,
    _normalize,
    _prepared,
    _train_epoch,
)


def _fuse(osnet, dino):
    return _normalize(np.concatenate((np.sqrt(0.75) * osnet, np.sqrt(0.25) * dino), axis=1))


def _evaluate(protocols, prepared, local, weight):
    rows = []
    for protocol, data, prediction in zip(protocols, prepared, local, strict=True):
        score = data["base"].copy()
        candidate = data["candidates"]
        base = np.take_along_axis(score, candidate, axis=1)
        np.put_along_axis(score, candidate, base + weight * (prediction - base), axis=1)
        rows.append(_official(np.argsort(-score, axis=1, kind="stable"), protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    torch.manual_seed(93217)
    np.random.seed(93217)
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_osnet = _normalize(_load(
        "outputs/expert_fusion/cache/osnet_smoothap_train.npz", train, "embeddings"
    ))
    val_osnet = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy", allow_pickle=False
    ))
    train_dino = _normalize(_load(
        "outputs/expert_fusion/cache/dinov2_vehicle_cls_train.npz", train, "embeddings"
    ))
    with np.load("outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz", allow_pickle=False) as archive:
        val_dino = _normalize(archive["cls"])
    train_embedding, val_embedding = _fuse(train_osnet, train_dino), _fuse(val_osnet, val_dino)
    train_conv = _load("outputs/expert_fusion/cache/convnext_train_parts_3x3.npz", train, "parts")
    train_parts = _load("outputs/expert_fusion/cache/osnet_smoothap_train_parts4.npz", train, "parts")
    val_conv = _load("outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", val, "parts")
    val_parts = _load("outputs/expert_fusion/cache/osnet_smoothap_val_parts4.npz", val, "parts")

    train_protocols = [_make_train_protocol(train, train_embedding, seed) for seed in TRAIN_SEEDS]
    tune = _protocols(val, TUNE_SEEDS, {"base": val_embedding})
    confirm = _protocols(val, CONFIRM_SEEDS, {"base": val_embedding})
    tune_data = [_prepared(p, val_embedding) for p in tune]
    confirm_data = [_prepared(p, val_embedding) for p in confirm]

    model = CrossImageTransformer(dim=96).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=4e-3)
    history, snapshots = [], []
    for epoch in range(1, 6):
        loss = _train_epoch(
            model, optimizer, train_protocols, train_conv, train_parts,
            train_embedding, 93217 + epoch,
        )
        local = [_local_for_protocol(
            model, tune[0], tune_data[0], val_conv, val_parts, val_embedding
        )]
        metrics = _evaluate(tune[:1], tune_data[:1], local, 0.25)
        row = {"epoch": epoch, "loss": loss, **metrics}
        history.append(row)
        snapshots.append(copy.deepcopy(model.state_dict()))
        print(json.dumps(row), flush=True)

    selected_epoch = max(
        range(len(history)),
        key=lambda index: (history[index]["mAP@10"], history[index]["Rank-1"]),
    )
    model.load_state_dict(snapshots[selected_epoch])
    tune_local = [
        _local_for_protocol(model, p, d, val_conv, val_parts, val_embedding)
        for p, d in zip(tune, tune_data, strict=True)
    ]
    confirm_local = [
        _local_for_protocol(model, p, d, val_conv, val_parts, val_embedding)
        for p, d in zip(confirm, confirm_data, strict=True)
    ]
    grid = [
        {"verifier_weight": weight, **_evaluate(tune, tune_data, tune_local, weight)}
        for weight in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
    ]
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    confirmation = _evaluate(
        confirm, confirm_data, confirm_local, selected["verifier_weight"]
    )
    baseline = _evaluate(confirm, confirm_data, confirm_local, 0.0)
    report = {
        "design": "DINO+OSNet shortlist-specific streaming local verifier",
        "history": history,
        "selected_epoch": selected_epoch + 1,
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {key: confirmation[key] - baseline[key] for key in ("mAP@10", "Rank-1", "Rank-5")},
        "grid": grid,
    }
    Path("outputs/expert_fusion/dino_top25_verifier.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    torch.save({"model_state": model.state_dict(), "report": report}, "weights/dino_top25_verifier.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
