"""Test-shaped viewpoint-conditioned contextual reranking of a frozen top-25."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.probe_camera_transition_prior import _camera_log_likelihood
from scripts.probe_family_reranking import _metrics
from scripts.probe_gallery_family_clustering import _components as _gallery_components
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_predicted_camera_family import _augment, _load_probability
from scripts.probe_query_family_clustering import _components as _query_components, _normalize, _prepare
from src.reranking import query_expansion


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP = 25
QUERY_CONFIG = dict(neighbor_k=3, threshold=0.55, overlap_min=2, complete_link=False)
GALLERY_CONFIG = dict(embedding="raw_gallery", pair_alpha=0.05, neighbor_k=2,
                      threshold=0.55, complete_link=True)


class ContextualRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 96), nn.GELU())
        layer = nn.TransformerEncoderLayer(96, 4, 256, dropout=0.15, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1))
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)

    def forward(self, values):
        return self.output(self.context(self.input(values))).squeeze(2)


def _joint_score(protocol, probability, prior):
    query_groups = _query_components(protocol, **QUERY_CONFIG)
    gallery_groups = _gallery_components(protocol, probability, prior, GALLERY_CONFIG)
    raw = protocol["raw_query"]
    augmented = raw.copy()
    for group in query_groups:
        if len(group) > 1:
            index = np.asarray(group)
            prototype = _normalize(raw[index].mean(axis=0, keepdims=True))[0]
            augmented[index] = _normalize(raw[index] + prototype)
    query = query_expansion(augmented, protocol["gallery"], top_k=2, alpha=1.0)
    score = query @ protocol["gallery"].T + 0.02 * protocol["camera_score"]
    for group in gallery_groups:
        if len(group) > 1:
            index = np.asarray(group)
            family = score[:, index].max(axis=1, keepdims=True)
            score[:, index] = 0.35 * score[:, index] + 0.65 * family
    return score, query_groups, gallery_groups


def _arrays(protocol, probability, prior, view_probability, mode="full"):
    score, query_groups, gallery_groups = _joint_score(protocol, probability, prior)
    order = np.argsort(-score, axis=1, kind="stable")
    candidates = order[:, :TOP]
    query_to_group = np.arange(len(protocol["q"]))
    for group in query_groups:
        for index in group: query_to_group[index] = group[0]
    group_lookup = {group[0]: np.asarray(group, dtype=np.int64) for group in query_groups}
    gallery_size = np.ones(len(protocol["g"]), dtype=np.float32)
    for group in gallery_groups:
        gallery_size[group] = len(group)
    q_views = view_probability[protocol["qi"]]
    g_views = view_probability[protocol["gi"]]
    raw_pair = protocol["raw_query"] @ protocol["raw_gallery"].T
    features, labels, validity = [], [], []
    gp = protocol["g"].vehicle_id.to_numpy()
    gc = protocol["g"].camera_id.to_numpy()
    for qi, row in enumerate(protocol["q"].itertuples(index=False)):
        cand = candidates[qi]
        qgroup = group_lookup[query_to_group[qi]]
        appearance = protocol["raw_query"][qgroup] @ protocol["raw_gallery"][cand].T
        view = q_views[qgroup] @ g_views[cand].T
        attention = np.exp(5.0 * (view - view.max(axis=0, keepdims=True)))
        attention /= attention.sum(axis=0, keepdims=True).clip(min=1e-12)
        conditioned = np.sum(attention * appearance, axis=0)
        affinity = protocol["raw_gallery"][cand] @ protocol["raw_gallery"][cand].T
        selected = score[qi, cand]
        selected_z = (selected - selected.mean()) / (selected.std() + 1e-6)
        scalar = np.stack((
            selected_z,
            raw_pair[qi, cand],
            protocol["camera_score"][qi, cand],
            appearance.max(axis=0), appearance.mean(axis=0), appearance.min(axis=0),
            appearance.std(axis=0), conditioned,
            view.max(axis=0), view.mean(axis=0),
            np.log1p(gallery_size[cand]),
            np.full(TOP, np.log1p(len(qgroup)), dtype=np.float64),
            np.arange(TOP, dtype=np.float64) / (TOP - 1),
        ), axis=1)
        features.append(np.concatenate((scalar, affinity), axis=1).astype(np.float32))
        same_identity = gp[cand] == row.vehicle_id
        junk = same_identity & (gc[cand] == row.camera_id)
        labels.append(same_identity & ~junk)
        validity.append(~junk)
    return {"features": np.stack(features), "labels": np.stack(labels),
            "valid": np.stack(validity), "candidates": candidates,
            "score": score, "query_groups": query_groups, "gallery_groups": gallery_groups}


def _loss(logits, labels, candidate_valid=None):
    if candidate_valid is None:
        candidate_valid = torch.ones_like(labels, dtype=torch.bool)
    candidate_valid = candidate_valid.bool()
    labels = labels.bool() & candidate_valid
    valid_rows = labels.any(1)
    target = labels.float() / labels.sum(1, keepdim=True).clamp_min(1)
    scaled = (logits / 0.25).masked_fill(~candidate_valid, -torch.inf)
    log_probability = scaled - torch.logsumexp(scaled, dim=1, keepdim=True)
    log_probability = log_probability.masked_fill(~candidate_valid, 0.0)
    listwise = -(target[valid_rows] * log_probability[valid_rows]).sum(1).mean()
    bce = F.binary_cross_entropy_with_logits(
        logits[candidate_valid], labels.float()[candidate_valid],
        pos_weight=torch.tensor(8.0, device=logits.device),
    )
    return listwise + 0.30 * bce


def _train(protocols, arrays, held_ids, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = ContextualRanker(arrays[0]["features"].shape[2]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=3e-3)
    best, best_loss = None, np.inf
    for epoch in range(1, 13):
        model.train(); losses = []
        for protocol, data in zip(protocols, arrays, strict=True):
            keep = ~protocol["q"].vehicle_id.isin(held_ids).to_numpy()
            indices = np.flatnonzero(keep & data["labels"].any(1)); rng.shuffle(indices)
            for start in range(0, len(indices), 48):
                index = indices[start:start + 48]
                x = torch.from_numpy(data["features"][index]).to(DEVICE)
                y = torch.from_numpy(data["labels"][index]).to(DEVICE)
                valid = torch.from_numpy(data["valid"][index]).to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda", dtype=torch.float16):
                    loss = _loss(model(x), y, valid)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
                losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss, best = mean_loss, copy.deepcopy(model.state_dict())
        if epoch in (1, 4, 8, 12): print(f"seed={seed} epoch={epoch} loss={mean_loss:.4f}", flush=True)
    return best


@torch.inference_mode()
def _predict(model, features):
    model.eval(); chunks = []
    for start in range(0, len(features), 96):
        x = torch.from_numpy(features[start:start + 96]).to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda", dtype=torch.float16):
            chunks.append(model(x).float().cpu().numpy())
    return np.concatenate(chunks)


def _rerank(data, prediction, blend):
    output = np.argsort(-data["score"], axis=1, kind="stable")
    for query in range(len(output)):
        base = data["score"][query, data["candidates"][query]]
        base = (base - base.mean()) / (base.std() + 1e-6)
        learned = prediction[query]
        learned = (learned - learned.mean()) / (learned.std() + 1e-6)
        local = np.argsort(-((1 - blend) * base + blend * learned), kind="stable")
        output[query, :TOP] = data["candidates"][query, local]
    return output


def _aggregate(rows):
    keys = ("mAP", "AP10", "positive_recall_at_10", "rank1", "rank5")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main():
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    prior, camera_index = _camera_log_likelihood(train, 0.25)
    camera_probability = _load_probability("outputs/expert_fusion/cache/scene_camera_mlp_predictions_val.npz", val, camera_index)
    with np.load("outputs/expert_fusion/cache/val_vehiclex_view_probabilities.npz", allow_pickle=False) as archive:
        view_probability = archive["probabilities"].astype(np.float64)
    embeddings = _normalize(np.load("outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy", allow_pickle=False).astype(np.float64))
    tune_protocols = [_augment(_prepare(embeddings, val, seed), camera_probability, prior, 0.05, 0.02) for seed in TUNE_SEEDS]
    confirm_protocols = [_augment(_prepare(embeddings, val, seed), camera_probability, prior, 0.05, 0.02) for seed in CONFIRM_SEEDS]
    tune_arrays = [_arrays(p, camera_probability, prior, view_probability) for p in tune_protocols]
    confirm_arrays = [_arrays(p, camera_probability, prior, view_probability) for p in confirm_protocols]
    identities = np.unique(val.vehicle_id); np.random.default_rng(88201).shuffle(identities)
    folds = np.array_split(identities, 5)
    tune_predictions = [np.zeros_like(a["labels"], np.float32) for a in tune_arrays]
    confirm_predictions = [np.zeros_like(a["labels"], np.float32) for a in confirm_arrays]
    states = []
    for fold, ids in enumerate(folds):
        held = set(ids.tolist()); print(f"fold={fold + 1}/5 ids={len(held)}", flush=True)
        state = _train(tune_protocols, tune_arrays, held, 71000 + fold); states.append(state)
        model = ContextualRanker(tune_arrays[0]["features"].shape[2]).to(DEVICE); model.load_state_dict(state)
        for protocols, arrays, destinations in ((tune_protocols, tune_arrays, tune_predictions),
                                                 (confirm_protocols, confirm_arrays, confirm_predictions)):
            for protocol, data, destination in zip(protocols, arrays, destinations, strict=True):
                mask = protocol["q"].vehicle_id.isin(held).to_numpy()
                destination[mask] = _predict(model, data["features"])[mask]
        del model; torch.cuda.empty_cache()
    grid = []
    for blend in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        rows = [_metrics(_rerank(a, pred, blend), p["q"], p["g"]) for p, a, pred in zip(tune_protocols, tune_arrays, tune_predictions, strict=True)]
        grid.append({"blend": blend, **_aggregate(rows)})
    selected = max(grid, key=lambda row: (row["rank1"], row["mAP"], row["rank5"]))
    rows = [_metrics(_rerank(a, pred, selected["blend"]), p["q"], p["g"]) for p, a, pred in zip(confirm_protocols, confirm_arrays, confirm_predictions, strict=True)]
    baseline_rows = [_metrics(np.argsort(-a["score"], axis=1, kind="stable"), p["q"], p["g"]) for p, a in zip(confirm_protocols, confirm_arrays, strict=True)]
    confirmation, baseline = _aggregate(rows), _aggregate(baseline_rows)
    report = {"design": "test-shaped OOF viewpoint-conditioned CSA/MTC top-25", "selected": selected,
              "grid": grid, "confirmation": confirmation, "baseline": baseline,
              "delta": {key: confirmation[key] - baseline[key] for key in confirmation}}
    Path("outputs/expert_fusion/contextual_multiquery_reranker.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"states": states, "fold_identities": folds, "report": report}, "weights/contextual_multiquery_reranker.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
