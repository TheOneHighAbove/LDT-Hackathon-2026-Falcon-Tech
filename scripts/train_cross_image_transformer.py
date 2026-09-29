"""Train an ETR/R2Former-style local cross-image verifier on train identities."""

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
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_predicted_camera_family import _augment, _load_probability
from scripts.probe_query_family_clustering import _normalize, _prepare
from scripts.train_contextual_multiquery_reranker import DEVICE, _aggregate, _joint_score


TOP = 15
CANDIDATES = 10


def _load(path, frame, key):
    with np.load(path, allow_pickle=False) as archive:
        expected = frame.image_id.astype(str).to_numpy(dtype=np.str_)
        if not np.array_equal(archive["image_ids"], expected):
            raise RuntimeError(f"unaligned cache: {path}")
        return archive[key].astype(np.float32)


def _fused(conv, osnet):
    return _normalize(np.concatenate((np.sqrt(0.25) * conv, np.sqrt(0.75) * osnet), axis=1))


class CrossImageTransformer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim, bias=False))
        self.osnet = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, dim, bias=False))
        self.position = nn.Parameter(torch.randn(25, dim) * 0.02)
        self.source = nn.Parameter(torch.randn(2, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, 4, 192, dropout=0.10, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(dim, 4, dropout=0.10, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 4 + 22),
            nn.Linear(dim * 4 + 22, 192), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(192, 64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _tokens(self, conv, osnet):
        values = torch.cat((
            self.conv(conv) + self.source[0],
            self.osnet(osnet) + self.source[1],
        ), dim=1)
        return self.encoder(values + self.position)

    def forward(self, q_conv, q_osnet, g_conv, g_osnet, global_similarity):
        query, gallery = self._tokens(q_conv, q_osnet), self._tokens(g_conv, g_osnet)
        q_cross, _ = self.cross(query, gallery, gallery, need_weights=False)
        g_cross, _ = self.cross(gallery, query, query, need_weights=False)
        q_cross = self.cross_norm(query + q_cross)
        g_cross = self.cross_norm(gallery + g_cross)

        qn, gn = F.normalize(query, dim=2), F.normalize(gallery, dim=2)
        correlation = torch.einsum("bqd,bgd->bqg", qn, gn)
        top = correlation.flatten(1).topk(16, dim=1).values
        q_best, g_best = correlation.amax(2), correlation.amax(1)
        correlation_stats = torch.cat((
            top,
            q_best.mean(1, keepdim=True), q_best.std(1, keepdim=True), q_best.amax(1, keepdim=True),
            g_best.mean(1, keepdim=True), g_best.std(1, keepdim=True), g_best.amax(1, keepdim=True),
        ), dim=1)

        q_match = torch.einsum("bqg,bgd->bqd", F.softmax(10 * correlation, dim=2), gallery)
        g_match = torch.einsum("bgq,bqd->bgd", F.softmax(10 * correlation.transpose(1, 2), dim=2), query)
        local = torch.cat((
            torch.abs(q_cross - q_match).mean(1), (q_cross * q_match).mean(1),
            torch.abs(g_cross - g_match).mean(1), (g_cross * g_match).mean(1),
        ), dim=1)
        learned = self.head(torch.cat((local, correlation_stats), dim=1)).squeeze(1)
        return global_similarity + 0.35 * torch.tanh(learned)


@torch.inference_mode()
def _hard_negatives(embeddings, labels, number=80):
    values = torch.from_numpy(embeddings).to(DEVICE)
    label = torch.from_numpy(labels).to(DEVICE)
    output = []
    for start in range(0, len(values), 256):
        similarity = values[start:start + 256] @ values.T
        invalid = label[start:start + 256, None].eq(label[None])
        similarity.masked_fill_(invalid, -1e4)
        output.append(similarity.topk(number, dim=1).indices.cpu().numpy())
    return np.concatenate(output)


def _positive_lookup(frame):
    labels = frame.vehicle_id.to_numpy()
    cameras = frame.camera_id.to_numpy()
    by_id = {}
    for index, identity in enumerate(labels):
        by_id.setdefault(identity, []).append(index)
    return [
        np.asarray([j for j in by_id[identity] if cameras[j] != cameras[i]], dtype=np.int64)
        for i, identity in enumerate(labels)
    ]


def _pair_batch(conv_parts, osnet_parts, query, gallery, similarity):
    return (
        torch.from_numpy(conv_parts[query]).to(DEVICE),
        torch.from_numpy(osnet_parts[query]).to(DEVICE),
        torch.from_numpy(conv_parts[gallery]).to(DEVICE),
        torch.from_numpy(osnet_parts[gallery]).to(DEVICE),
        torch.from_numpy(similarity.astype(np.float32)).to(DEVICE),
    )


def _train_epoch(model, optimizer, fused, conv_parts, osnet_parts, positives, negatives, seed):
    model.train()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(fused))
    losses = []
    for start in range(0, len(order), 20):
        query = order[start:start + 20]
        if not len(query):
            continue
        candidate_rows, targets = [], []
        for qi in query:
            positive = int(rng.choice(positives[qi]))
            pool = negatives[qi, :40]
            chosen = rng.choice(pool, CANDIDATES - 1, replace=False)
            candidate = np.concatenate(([positive], chosen))
            scores = fused[qi] @ fused[candidate].T
            ranking = np.argsort(-scores, kind="stable")
            candidate_rows.append(candidate[ranking])
            targets.append(int(np.flatnonzero(ranking == 0)[0]))
        candidates = np.stack(candidate_rows)
        qflat = np.repeat(query, CANDIDATES)
        gflat = candidates.reshape(-1)
        global_similarity = np.sum(fused[qflat] * fused[gflat], axis=1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda", dtype=torch.float16):
            score = model(*_pair_batch(conv_parts, osnet_parts, qflat, gflat, global_similarity))
            score = score.reshape(len(query), CANDIDATES)
            target = torch.tensor(targets, device=DEVICE)
            ce = F.cross_entropy(score / 0.10, target)
            positive_score = score.gather(1, target[:, None]).squeeze(1)
            mask = F.one_hot(target, CANDIDATES).bool()
            negative_score = score.masked_fill(mask, -1e4).amax(1)
            loss = ce + 0.25 * F.softplus((negative_score - positive_score) / 0.08).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def _score_pairs(model, conv_parts, osnet_parts, fused, q_index, g_index, batch=256):
    model.eval()
    result = []
    for start in range(0, len(q_index), batch):
        q = q_index[start:start + batch]
        g = g_index[start:start + batch]
        global_similarity = np.sum(fused[q] * fused[g], axis=1)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda", dtype=torch.float16):
            result.append(model(*_pair_batch(conv_parts, osnet_parts, q, g, global_similarity)).float().cpu().numpy())
    return np.concatenate(result)


def _cross_scores(model, protocol, data, fused, conv_parts, osnet_parts, aggregation):
    query_lookup = {}
    for group in data["query_groups"]:
        for member in group:
            query_lookup[member] = np.asarray(group, dtype=np.int64)
    q_pairs, g_pairs, owners = [], [], []
    for qi, candidates in enumerate(data["candidates"][:, :TOP]):
        group = query_lookup[qi]
        for rank, candidate in enumerate(candidates):
            q_pairs.extend(protocol["qi"][group].tolist())
            g_pairs.extend([int(protocol["gi"][candidate])] * len(group))
            owners.extend([(qi, rank)] * len(group))
    values = _score_pairs(
        model, conv_parts, osnet_parts, fused,
        np.asarray(q_pairs, dtype=np.int64), np.asarray(g_pairs, dtype=np.int64),
    )
    buckets = [[[] for _ in range(TOP)] for _ in range(len(protocol["q"]))]
    for owner, value in zip(owners, values, strict=True):
        buckets[owner[0]][owner[1]].append(float(value))
    if aggregation == "max":
        return np.asarray([[max(cell) for cell in row] for row in buckets], dtype=np.float32)
    return np.asarray([[np.mean(cell) for cell in row] for row in buckets], dtype=np.float32)


def _rerank(data, local_score, blend):
    output = np.argsort(-data["score"], axis=1, kind="stable")
    candidates = data["candidates"][:, :TOP]
    base = np.take_along_axis(data["score"], candidates, axis=1)
    base = (base - base.mean(1, keepdims=True)) / (base.std(1, keepdims=True) + 1e-6)
    local = (local_score - local_score.mean(1, keepdims=True)) / (local_score.std(1, keepdims=True) + 1e-6)
    order = np.argsort(-((1 - blend) * base + blend * local), axis=1, kind="stable")
    output[:, :TOP] = np.take_along_axis(candidates, order, axis=1)
    return output


def _protocol_arrays(frame, embeddings, probability, prior, seeds):
    protocols = [_augment(_prepare(embeddings, frame, seed), probability, prior, 0.05, 0.02) for seed in seeds]
    result = []
    for protocol in protocols:
        score, query_groups, gallery_groups = _joint_score(protocol, probability, prior)
        result.append({
            "score": score,
            "candidates": np.argsort(-score, axis=1, kind="stable")[:, :TOP],
            "query_groups": query_groups,
            "gallery_groups": gallery_groups,
        })
    return protocols, result


def main():
    torch.manual_seed(81001)
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_conv = _normalize(_load("outputs/expert_fusion/cache/convnext_train_verifier.npz", train, "embeddings"))
    train_osnet = _normalize(_load("outputs/expert_fusion/cache/osnet_smoothap_train.npz", train, "embeddings"))
    train_fused = _fused(train_conv, train_osnet).astype(np.float32)
    train_conv_parts = _load("outputs/expert_fusion/cache/convnext_train_parts_3x3.npz", train, "parts")
    train_osnet_parts = _load("outputs/expert_fusion/cache/osnet_smoothap_train_parts4.npz", train, "parts")
    positives = _positive_lookup(train)
    negatives = _hard_negatives(train_fused, train.vehicle_id.to_numpy())

    val_embeddings = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy", allow_pickle=False
    ).astype(np.float32))
    val_conv_parts = _load("outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", val, "parts")
    val_osnet_parts = _load("outputs/expert_fusion/cache/osnet_smoothap_val_parts4.npz", val, "parts")
    prior, camera_index = _camera_log_likelihood(train, 0.25)
    probability = _load_probability(
        "outputs/expert_fusion/cache/scene_camera_mlp_predictions_val.npz", val, camera_index
    )
    tune_protocols, tune_data = _protocol_arrays(val, val_embeddings, probability, prior, TUNE_SEEDS)
    confirm_protocols, confirm_data = _protocol_arrays(val, val_embeddings, probability, prior, CONFIRM_SEEDS)

    model = CrossImageTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-3)
    best_state, history, best_rank1 = None, [], -1.0
    # A single tune seed is used only for epoch selection; all confirmation seeds remain untouched.
    for epoch in range(1, 7):
        loss = _train_epoch(
            model, optimizer, train_fused, train_conv_parts, train_osnet_parts,
            positives, negatives, 81001 + epoch,
        )
        local = _cross_scores(
            model, tune_protocols[0], tune_data[0], val_embeddings,
            val_conv_parts, val_osnet_parts, "max",
        )
        metrics = _metrics(_rerank(tune_data[0], local, 0.50), tune_protocols[0]["q"], tune_protocols[0]["g"])
        row = {"epoch": epoch, "loss": loss, **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["rank1"] > best_rank1:
            best_rank1, best_state = metrics["rank1"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    scores = {}
    for aggregation in ("max", "mean"):
        scores[("tune", aggregation)] = [
            _cross_scores(model, p, d, val_embeddings, val_conv_parts, val_osnet_parts, aggregation)
            for p, d in zip(tune_protocols, tune_data, strict=True)
        ]
        scores[("confirm", aggregation)] = [
            _cross_scores(model, p, d, val_embeddings, val_conv_parts, val_osnet_parts, aggregation)
            for p, d in zip(confirm_protocols, confirm_data, strict=True)
        ]

    grid = []
    for aggregation in ("max", "mean"):
        for blend in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
            rows = [
                _metrics(_rerank(d, s, blend), p["q"], p["g"])
                for p, d, s in zip(tune_protocols, tune_data, scores[("tune", aggregation)], strict=True)
            ]
            grid.append({"aggregation": aggregation, "blend": blend, **_aggregate(rows)})
    selected = max(grid, key=lambda row: (row["rank1"], row["mAP"], row["rank5"]))
    confirmation = _aggregate([
        _metrics(_rerank(d, s, selected["blend"]), p["q"], p["g"])
        for p, d, s in zip(
            confirm_protocols, confirm_data, scores[("confirm", selected["aggregation"])], strict=True
        )
    ])
    baseline = _aggregate([
        _metrics(np.argsort(-d["score"], axis=1, kind="stable"), p["q"], p["g"])
        for p, d in zip(confirm_protocols, confirm_data, strict=True)
    ])
    report = {
        "design": "train-ID ETR/R2Former-style bidirectional local cross-attention",
        "identity_disjoint": True,
        "history": history,
        "selected": selected,
        "confirmation": confirmation,
        "baseline": baseline,
        "delta": {key: confirmation[key] - baseline[key] for key in confirmation},
        "grid": grid,
    }
    Path("outputs/expert_fusion/cross_image_transformer.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    torch.save({"model_state": best_state, "report": report}, "weights/cross_image_transformer.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
