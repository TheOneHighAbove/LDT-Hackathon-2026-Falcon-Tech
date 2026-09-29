"""Add the train-only family GNN residual to the fixed strict final stack."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.probe_dino_verifier_streaming import (
    _patch_gallery_affinity,
    _predict_patch_plain,
    _prepare,
    _scores,
)
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.probe_token_cross_final import _propagate_patch_knn
from scripts.probe_token_gallery_linker import _propagate, _token_neighborhoods
from scripts.train_cross_image_transformer import CrossImageTransformer, _load
from scripts.train_dino_patch_matcher import _load as _load_aligned
from scripts.train_dino_token_cross_top50 import DEVICE, DinoTokenCrossMatcher
from scripts.train_family_negative_gate import _predict as _predict_gate
from scripts.train_family_lambdarank import _predict as _predict_ranker
from scripts.probe_train_fitted_verifier import _colors
from scripts.train_streaming_top25_verifier import _local_for_protocol
from scripts.train_strict_family_gnn import (
    FamilyGraphReranker,
    _episode,
    _load_features,
    _predict as _predict_family,
)


def _score_one(item, local_value, patch_value, episode, family_value,
               gate_value, highres_value, previous, linker,
               query_token_weight, family_weight, gate_weight=0.0,
               highres_weight=0.0, hard_family=None, gate_reject=None,
               modern_linker=None, metric_dino_weight=0.0,
               ranker_value=None, ranker_weight=0.0):
    """Return the full-gallery score after every selected streaming stage.

    Keeping this as a pure helper lets downstream OOF specialists consume the
    *resulting* shortlist instead of silently training against an earlier
    retrieval candidate set.
    """
    score = _scores(
        item, local_value, patch_value,
        previous["verifier_weight"], previous["part_source"],
        previous["part_weight"], previous["patch_weight"],
    )
    if metric_dino_weight:
        score += metric_dino_weight * (
            item["part_scores"]["metric_dino"] - item["raw_fused"]
        )
    if highres_weight:
        patch_candidate = item["candidates"]
        patch_current = np.take_along_axis(score, patch_candidate, axis=1)
        patch_z = (highres_value - highres_value.mean(1, keepdims=True)) / (
            highres_value.std(1, keepdims=True) + 1e-6
        )
        patch_current += (
            highres_weight
            * (patch_current.std(1, keepdims=True) + 1e-6)
            * patch_z
        )
        np.put_along_axis(score, patch_candidate, patch_current, axis=1)
    candidate = episode["candidate"]
    base = episode["features"][..., 0]
    token = episode["features"][..., 1]
    current = np.take_along_axis(score, candidate, axis=1)
    reranked = (
        current
        + query_token_weight * (token - base)
        + family_weight * (family_value - base)
    )
    if gate_weight:
        gate_z = (gate_value - gate_value.mean(1, keepdims=True)) / (
            gate_value.std(1, keepdims=True) + 1e-6
        )
        reranked += gate_weight * (reranked.std(1, keepdims=True) + 1e-6) * gate_z
    if ranker_weight:
        ranker_z = (ranker_value - ranker_value.mean(1, keepdims=True)) / (
            ranker_value.std(1, keepdims=True) + 1e-6
        )
        reranked += (
            ranker_weight
            * (reranked.std(1, keepdims=True) + 1e-6)
            * ranker_z
        )
    if gate_reject is not None:
        current_order = np.argsort(-reranked, axis=1, kind="stable")
        current_rank = np.empty_like(current_order)
        np.put_along_axis(
            current_rank, current_order,
            np.broadcast_to(np.arange(reranked.shape[1]), current_order.shape), axis=1,
        )
        gate_order = np.argsort(-gate_value, axis=1, kind="stable")
        gate_rank = np.empty_like(gate_order)
        np.put_along_axis(
            gate_rank, gate_order,
            np.broadcast_to(np.arange(reranked.shape[1]), gate_order.shape), axis=1,
        )
        rejected = (
            (current_rank < gate_reject["window"])
            & ((gate_rank - current_rank) >= gate_reject["rank_gap"])
        )
        reranked -= (
            gate_reject["penalty"]
            * (reranked.std(1, keepdims=True) + 1e-6)
            * rejected
        )
    np.put_along_axis(score, candidate, reranked, axis=1)
    if linker["keep_patch_linker"]:
        score = _propagate_patch_knn(score, item, previous)
    affinity = np.where(
        episode["gallery_token_valid"] > 0.5,
        episode["gallery_token_similarity"],
        -np.inf,
    )
    np.fill_diagonal(affinity, -np.inf)
    neighborhoods = _token_neighborhoods(
        affinity, episode["gallery_rank"], linker
    )
    score = _propagate(score, neighborhoods, linker["propagation"])
    if modern_linker is not None:
        modern_affinity = item["gallery_sources"]["modern"]
        modern_neighborhoods = _token_neighborhoods(
            modern_affinity, episode["gallery_rank"], modern_linker
        )
        score = _propagate(
            score, modern_neighborhoods, modern_linker["propagation"]
        )
    if hard_family is not None:
        score = _hard_family_scores(score, episode, gate_value, affinity, hard_family)
    return score


def _evaluate(protocols, data, local, patch, episodes, family_prediction,
              gate_probability, highres_patch, previous, linker,
              query_token_weight, family_weight, gate_weight=0.0,
              highres_weight=0.0, hard_family=None, gate_reject=None,
              modern_linker=None, metric_dino_weight=0.0,
              ranker_probability=None, ranker_weight=0.0):
    if ranker_probability is None:
        ranker_probability = [np.zeros_like(value) for value in gate_probability]
    rows = []
    for (protocol, item, local_value, patch_value, episode, family_value,
         gate_value, highres_value, ranker_value) in zip(
            protocols, data, local, patch, episodes, family_prediction,
            gate_probability, highres_patch, ranker_probability, strict=True,
        ):
        score = _score_one(
            item, local_value, patch_value, episode, family_value, gate_value,
            highres_value, previous, linker, query_token_weight, family_weight,
            gate_weight, highres_weight, hard_family, gate_reject,
            modern_linker, metric_dino_weight, ranker_value, ranker_weight,
        )
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def _hard_family_scores(score, episode, gate, affinity, config):
    """Place trusted static-gallery relatives directly behind strong anchors."""
    neighborhoods = _token_neighborhoods(affinity, episode["gallery_rank"], config)
    base_order = np.argsort(-score, axis=1, kind="stable")
    output = np.empty_like(score)
    descending = np.arange(score.shape[1], 0, -1, dtype=np.float32)
    for row, order in enumerate(base_order):
        candidate = episode["candidate"][row]
        candidate_position = np.full(score.shape[1], -1, dtype=np.int16)
        candidate_position[candidate] = np.arange(len(candidate), dtype=np.int16)
        used = np.zeros(score.shape[1], dtype=bool)
        ranked = []
        for original_rank, anchor in enumerate(order):
            anchor = int(anchor)
            if used[anchor]:
                continue
            ranked.append(anchor)
            used[anchor] = True
            anchor_position = int(candidate_position[anchor])
            if original_rank >= config["seed_k"] or anchor_position < 0:
                continue
            anchor_gate = float(gate[row, anchor_position])
            for relative in neighborhoods[anchor][1:]:
                relative = int(relative)
                relative_position = int(candidate_position[relative])
                if (
                    used[relative]
                    or relative_position < 0
                    or relative_position >= config["partner_max_rank"]
                    or gate[row, relative_position] < anchor_gate - config["gate_margin"]
                ):
                    continue
                ranked.append(relative)
                used[relative] = True
        ranked.extend(int(index) for index in order if not used[index])
        output[row, np.asarray(ranked, dtype=np.int64)] = descending
    return output


def main():
    frame = pd.read_csv("splits/val.csv")
    fused, osnet, dino, tokens = _load_features(frame, "val")
    with np.load(
        "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz",
        allow_pickle=False,
    ) as archive:
        parts = {name: archive[name].astype(np.float32) for name in ("mean", "h2", "h4")}
    parts = {
        name: value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
        for name, value in parts.items()
    }
    metric_dino = _load_aligned(
        "outputs/expert_fusion/cache/dinov2_vehicle_metric_part_cont_val.npz",
        frame,
        "embeddings",
    )
    parts["metric_dino"] = metric_dino / np.maximum(
        np.linalg.norm(metric_dino, axis=1, keepdims=True), 1e-12
    )
    conv_parts = _load(
        "outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", frame, "parts"
    )
    osnet_parts = _load(
        "outputs/expert_fusion/cache/osnet_smoothap_val_parts4.npz", frame, "parts"
    )

    token_state = torch.load(
        os.environ.get("DINO_TOKEN_MODEL", "weights/dino_token_cross_top50.pt"),
        map_location=DEVICE,
        weights_only=False,
    )
    token_model = DinoTokenCrossMatcher().to(DEVICE)
    token_model.load_state_dict(token_state["model_state"])
    token_model.eval()
    family_variant = os.environ.get("FAMILY_GNN_VARIANT", "listwise").lower()
    family_suffix = "_smoothap" if family_variant == "smoothap" else ""
    family_state = torch.load(
        os.environ.get(
            "FAMILY_GNN_MODEL", f"weights/strict_family_gnn{family_suffix}.pt"
        ),
        map_location=DEVICE,
        weights_only=False,
    )
    family_model = FamilyGraphReranker(family_state["feature_dim"]).to(DEVICE)
    family_model.load_state_dict(family_state["model_state"])
    family_model.eval()
    use_gate = os.environ.get("USE_NEGATIVE_GATE", "0").lower() in {"1", "true", "yes"}
    gate_model = joblib.load(
        os.environ.get("FAMILY_GATE_MODEL", "weights/family_negative_gate.joblib")
    ) if use_gate else None
    verifier_state = torch.load(
        "weights/dino_top25_verifier.pt", map_location=DEVICE, weights_only=False
    )
    verifier = CrossImageTransformer(dim=96).to(DEVICE)
    verifier.load_state_dict(verifier_state["model_state"])
    verifier.eval()
    patch_model = joblib.load(
        os.environ.get("DINO_PATCH_MODEL", "weights/dino_patch_matcher.joblib")
    )
    use_highres = os.environ.get("USE_HIGHRES_PATCH", "0").lower() in {
        "1", "true", "yes"
    }
    if use_highres:
        highres_model = joblib.load(os.environ.get(
            "DINO_HIGHRES_MODEL", "weights/dino_patch_matcher_10x10.joblib"
        ))
        highres_tokens = _load_aligned(
            "outputs/expert_fusion/cache/dinov2_vehicle_patches_val_10x10.npz",
            frame,
            "tokens",
        )

    tune = _protocols(frame, TUNE_SEEDS, {"base": fused})
    confirm = _protocols(frame, CONFIRM_SEEDS, {"base": fused})
    episodes = [
        _episode(frame, fused, osnet, dino, tokens, token_model, seed)
        for seed in TUNE_SEEDS + CONFIRM_SEEDS
    ]
    tune_episodes, confirm_episodes = episodes[:len(tune)], episodes[len(tune):]
    tune_family = [_predict_family(family_model, value) for value in tune_episodes]
    confirm_family = [_predict_family(family_model, value) for value in confirm_episodes]
    if use_gate:
        gate_colors = (
            _colors(frame, "val")
            if getattr(gate_model, "n_features_in_", 41) > 41
            else None
        )
        tune_gate = [
            (
                _predict_ranker(
                    gate_model, episode, prediction, frame, gate_colors
                )
                if gate_colors is not None
                else _predict_gate(gate_model, episode, prediction)
            )
            for episode, prediction in zip(tune_episodes, tune_family, strict=True)
        ]
        confirm_gate = [
            (
                _predict_ranker(
                    gate_model, episode, prediction, frame, gate_colors
                )
                if gate_colors is not None
                else _predict_gate(gate_model, episode, prediction)
            )
            for episode, prediction in zip(confirm_episodes, confirm_family, strict=True)
        ]
    else:
        tune_gate = [np.zeros_like(value) for value in tune_family]
        confirm_gate = [np.zeros_like(value) for value in confirm_family]
    use_ranker = os.environ.get("USE_FAMILY_RANKER", "0").lower() in {
        "1", "true", "yes"
    }
    if use_ranker:
        ranker_model = joblib.load(os.environ.get(
            "FAMILY_RANKER_MODEL", "weights/family_lambdarank_lbs.joblib"
        ))
        ranker_colors = (
            _colors(frame, "val")
            if getattr(ranker_model, "n_features_in_", 41) > 41
            else None
        )
        tune_ranker = [
            _predict_ranker(
                ranker_model, episode, prediction, frame, ranker_colors
            )
            for episode, prediction in zip(tune_episodes, tune_family, strict=True)
        ]
        confirm_ranker = [
            _predict_ranker(
                ranker_model, episode, prediction, frame, ranker_colors
            )
            for episode, prediction in zip(
                confirm_episodes, confirm_family, strict=True
            )
        ]
    else:
        tune_ranker = [np.zeros_like(value) for value in tune_family]
        confirm_ranker = [np.zeros_like(value) for value in confirm_family]

    tune_data = [_prepare(value, fused, osnet, parts) for value in tune]
    confirm_data = [_prepare(value, fused, osnet, parts) for value in confirm]
    tune_local = [
        _local_for_protocol(verifier, p, d, conv_parts, osnet_parts, osnet)
        for p, d in zip(tune, tune_data, strict=True)
    ]
    confirm_local = [
        _local_for_protocol(verifier, p, d, conv_parts, osnet_parts, osnet)
        for p, d in zip(confirm, confirm_data, strict=True)
    ]
    tune_patch = [
        _predict_patch_plain(patch_model, p, d, tokens, fused, osnet, dino)
        for p, d in zip(tune, tune_data, strict=True)
    ]
    confirm_patch = [
        _predict_patch_plain(patch_model, p, d, tokens, fused, osnet, dino)
        for p, d in zip(confirm, confirm_data, strict=True)
    ]
    if use_highres:
        tune_highres = [
            _predict_patch_plain(
                highres_model, p, d, highres_tokens, fused, osnet, dino,
                grid_size=10,
            )
            for p, d in zip(tune, tune_data, strict=True)
        ]
        confirm_highres = [
            _predict_patch_plain(
                highres_model, p, d, highres_tokens, fused, osnet, dino,
                grid_size=10,
            )
            for p, d in zip(confirm, confirm_data, strict=True)
        ]
    else:
        tune_highres = [np.zeros_like(value) for value in tune_patch]
        confirm_highres = [np.zeros_like(value) for value in confirm_patch]
    for protocol, data in zip(tune + confirm, tune_data + confirm_data, strict=True):
        data["gallery_sources"]["patch"] = _patch_gallery_affinity(
            patch_model, protocol, tokens, fused, osnet, dino
        )
    use_modern_linker = os.environ.get("USE_MODERN_LINKER", "0").lower() in {
        "1", "true", "yes"
    }
    if use_modern_linker:
        with np.load(
            os.environ.get(
                "MODERN_LINKER_CACHE",
                "outputs/retrieval_v2/modern_gallery_linker_val.npz",
            ),
            allow_pickle=False,
        ) as archive:
            for seed, data in zip(
                TUNE_SEEDS + CONFIRM_SEEDS,
                tune_data + confirm_data,
                strict=True,
            ):
                data["gallery_sources"]["modern"] = archive[f"affinity_{seed}"].copy()

    previous_report = json.loads(Path(
        "outputs/expert_fusion/dino_verifier_streaming.json"
    ).read_text(encoding="utf-8"))
    previous_keys = (
        "verifier_weight", "part_source", "part_weight", "patch_weight",
        "group_source", "neighbor_k", "threshold", "propagation",
    )
    previous = {
        key: previous_report["selected_tune"][key] for key in previous_keys
    }
    linker_report = json.loads(Path(
        "outputs/retrieval_v2/token_gallery_linker.json"
    ).read_text(encoding="utf-8"))
    linker_keys = (
        "keep_patch_linker", "neighbor_k", "reciprocal_rank", "threshold", "propagation",
    )
    linker = {key: linker_report["selected_tune"][key] for key in linker_keys}

    grid = []
    if use_gate:
        first_stage = []
        for gate_weight in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20,
                            0.30, 0.40, 0.50, 0.70, 1.0):
            metrics = _evaluate(
                tune, tune_data, tune_local, tune_patch, tune_episodes, tune_family,
                tune_gate, tune_highres, previous, linker, 0.15, 0.15, gate_weight,
            )
            first_stage.append({
                "query_token_weight": 0.15, "family_weight": 0.15,
                "gate_weight": gate_weight, **metrics,
            })
        best_gate = max(first_stage, key=lambda row: (row["mAP@10"], row["Rank-1"]))
        grid.extend(first_stage)
        for query_weight in (0.10, 0.15, 0.20):
            for family_weight in (0.10, 0.15, 0.20):
                metrics = _evaluate(
                    tune, tune_data, tune_local, tune_patch, tune_episodes, tune_family,
                    tune_gate, tune_highres, previous, linker, query_weight, family_weight,
                    best_gate["gate_weight"],
                )
                grid.append({
                    "query_token_weight": query_weight,
                    "family_weight": family_weight,
                    "gate_weight": best_gate["gate_weight"],
                    **metrics,
                })
    else:
        for query_weight in (0.15, 0.20, 0.25, 0.30):
            for family_weight in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20,
                                  0.30, 0.40, 0.50):
                metrics = _evaluate(
                    tune, tune_data, tune_local, tune_patch, tune_episodes, tune_family,
                    tune_gate, tune_highres, previous, linker, query_weight, family_weight,
                )
                grid.append({
                    "query_token_weight": query_weight,
                    "family_weight": family_weight,
                    "gate_weight": 0.0,
                    **metrics,
                })
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    selected.setdefault("highres_weight", 0.0)
    if use_highres:
        for highres_weight in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30):
            metrics = _evaluate(
                tune, tune_data, tune_local, tune_patch, tune_episodes, tune_family,
                tune_gate, tune_highres, previous, linker,
                selected["query_token_weight"], selected["family_weight"],
                selected["gate_weight"], highres_weight,
            )
            grid.append({
                "query_token_weight": selected["query_token_weight"],
                "family_weight": selected["family_weight"],
                "gate_weight": selected["gate_weight"],
                "highres_weight": highres_weight,
                **metrics,
            })
        selected = max(
            grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"])
        )
        selected.setdefault("highres_weight", 0.0)
    use_selective_gate = (
        use_gate
        and os.environ.get("SELECTIVE_GATE", "0").lower() in {"1", "true", "yes"}
    )
    reject_grid = []
    selected_reject = None
    if use_selective_gate:
        for window in (5, 10, 15):
            for rank_gap in (5, 10, 15, 20):
                for penalty in (0.025, 0.05, 0.10, 0.20, 0.40):
                    reject = {
                        "window": window,
                        "rank_gap": rank_gap,
                        "penalty": penalty,
                    }
                    metrics = _evaluate(
                        tune, tune_data, tune_local, tune_patch,
                        tune_episodes, tune_family, tune_gate, tune_highres,
                        previous, linker, selected["query_token_weight"],
                        selected["family_weight"], selected["gate_weight"],
                        selected["highres_weight"], gate_reject=reject,
                    )
                    reject_grid.append({**reject, **metrics})
        best_reject = max(
            reject_grid,
            key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]),
        )
        selected_reject = {
            key: best_reject[key] for key in ("window", "rank_gap", "penalty")
        }
    modern_grid = []
    selected_modern = None
    if use_modern_linker:
        for neighbor_k in (1, 2):
            for reciprocal_rank in (4, 8, 16):
                for threshold in (0.30, 0.50, 0.70, 0.80, 0.90):
                    for propagation in (0.05, 0.10, 0.20, 0.30, 0.50):
                        config = {
                            "neighbor_k": neighbor_k,
                            "reciprocal_rank": reciprocal_rank,
                            "threshold": threshold,
                            "propagation": propagation,
                        }
                        metrics = _evaluate(
                            tune, tune_data, tune_local, tune_patch,
                            tune_episodes, tune_family, tune_gate, tune_highres,
                            previous, linker, selected["query_token_weight"],
                            selected["family_weight"], selected["gate_weight"],
                            selected["highres_weight"],
                            gate_reject=selected_reject, modern_linker=config,
                        )
                        modern_grid.append({**config, **metrics})
        best_modern = max(
            modern_grid,
            key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]),
        )
        selected_modern = {
            key: best_modern[key]
            for key in ("neighbor_k", "reciprocal_rank", "threshold", "propagation")
        }
    use_metric_dino = os.environ.get("USE_METRIC_DINO", "0").lower() in {
        "1", "true", "yes"
    }
    metric_dino_grid = []
    selected_metric_dino_weight = 0.0
    if use_metric_dino:
        for metric_weight in (0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30):
            metrics = _evaluate(
                tune, tune_data, tune_local, tune_patch,
                tune_episodes, tune_family, tune_gate, tune_highres,
                previous, linker, selected["query_token_weight"],
                selected["family_weight"], selected["gate_weight"],
                selected["highres_weight"], gate_reject=selected_reject,
                modern_linker=selected_modern, metric_dino_weight=metric_weight,
            )
            metric_dino_grid.append({"metric_dino_weight": metric_weight, **metrics})
        selected_metric = max(
            metric_dino_grid,
            key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]),
        )
        selected_metric_dino_weight = selected_metric["metric_dino_weight"]
    ranker_grid = []
    selected_ranker_weight = 0.0
    if use_ranker:
        for ranker_weight in (
            0.025, 0.05, 0.075, 0.10, 0.15, 0.20,
            0.30, 0.40, 0.50, 0.70, 1.0,
        ):
            metrics = _evaluate(
                tune, tune_data, tune_local, tune_patch,
                tune_episodes, tune_family, tune_gate, tune_highres,
                previous, linker, selected["query_token_weight"],
                selected["family_weight"], selected["gate_weight"],
                selected["highres_weight"], gate_reject=selected_reject,
                modern_linker=selected_modern,
                metric_dino_weight=selected_metric_dino_weight,
                ranker_probability=tune_ranker,
                ranker_weight=ranker_weight,
            )
            ranker_grid.append({"ranker_weight": ranker_weight, **metrics})
        selected_ranker = max(
            ranker_grid,
            key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]),
        )
        selected_ranker_weight = selected_ranker["ranker_weight"]
    use_hard_family = os.environ.get("HARD_FAMILY", "0").lower() in {"1", "true", "yes"}
    hard_grid = []
    selected_hard = None
    if use_hard_family:
        for reciprocal_rank in (1, 2):
            for seed_k in (1, 3, 5):
                for partner_max_rank in (25, 50):
                    for gate_margin in (0.15, 0.30, 1.0):
                        config = {
                            "neighbor_k": 1,
                            "reciprocal_rank": reciprocal_rank,
                            "threshold": -np.inf,
                            "seed_k": seed_k,
                            "partner_max_rank": partner_max_rank,
                            "gate_margin": gate_margin,
                        }
                        metrics = _evaluate(
                            tune, tune_data, tune_local, tune_patch,
                            tune_episodes, tune_family, tune_gate, tune_highres,
                            previous, linker,
                            selected["query_token_weight"], selected["family_weight"],
                            selected["gate_weight"], selected["highres_weight"], config,
                        )
                        hard_grid.append({**config, **metrics})
        selected_hard = max(
            hard_grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"])
        )
        selected_hard = {
            key: selected_hard[key]
            for key in (
                "neighbor_k", "reciprocal_rank", "threshold", "seed_k",
                "partner_max_rank", "gate_margin",
            )
        }
    confirmation = _evaluate(
        confirm, confirm_data, confirm_local, confirm_patch,
        confirm_episodes, confirm_family, confirm_gate, confirm_highres, previous, linker,
        selected["query_token_weight"], selected["family_weight"],
        selected["gate_weight"], selected["highres_weight"], selected_hard,
        selected_reject, selected_modern, selected_metric_dino_weight,
        ranker_probability=confirm_ranker,
        ranker_weight=selected_ranker_weight,
    )
    baseline = _evaluate(
        confirm, confirm_data, confirm_local, confirm_patch,
        confirm_episodes, confirm_family, confirm_gate, confirm_highres, previous, linker,
        0.25, 0.0, 0.0, 0.0,
    )
    report = {
        "design": "train-only family GNN residual added to the strict token+gallery stack",
        "family_variant": family_variant,
        "negative_gate": use_gate,
        "highres_patch": use_highres,
        "selective_gate": use_selective_gate,
        "modern_linker": use_modern_linker,
        "metric_dino": use_metric_dino,
        "family_ranker": use_ranker,
        "hard_family": use_hard_family,
        "selected_tune": selected,
        "selected_hard_family": selected_hard,
        "selected_gate_reject": selected_reject,
        "selected_modern_linker": selected_modern,
        "selected_metric_dino_weight": selected_metric_dino_weight,
        "selected_ranker_weight": selected_ranker_weight,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "top_tune": sorted(
            grid, key=lambda row: (row["mAP@10"], row["Rank-1"]), reverse=True
        )[:25],
        "top_hard_family_tune": sorted(
            hard_grid, key=lambda row: (row["mAP@10"], row["Rank-1"]), reverse=True
        )[:20],
        "top_gate_reject_tune": sorted(
            reject_grid, key=lambda row: (row["mAP@10"], row["Rank-1"]), reverse=True
        )[:20],
        "top_modern_linker_tune": sorted(
            modern_grid, key=lambda row: (row["mAP@10"], row["Rank-1"]), reverse=True
        )[:20],
        "metric_dino_tune": metric_dino_grid,
        "ranker_tune": ranker_grid,
    }
    gate_suffix = "_gate" if use_gate else ""
    hard_suffix = "_hard" if use_hard_family else ""
    highres_suffix = "_highres" if use_highres else ""
    reject_suffix = "_reject" if use_selective_gate else ""
    modern_suffix = "_modern" if use_modern_linker else ""
    metric_suffix = "_metricdino" if use_metric_dino else ""
    ranker_suffix = "_ranker" if use_ranker else ""
    experiment_suffix = os.environ.get("STRICT_REPORT_SUFFIX", "")
    if experiment_suffix and not experiment_suffix.startswith("_"):
        experiment_suffix = f"_{experiment_suffix}"
    Path(f"outputs/retrieval_v2/family_gnn_final{family_suffix}{gate_suffix}{highres_suffix}{reject_suffix}{modern_suffix}{metric_suffix}{ranker_suffix}{hard_suffix}{experiment_suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
