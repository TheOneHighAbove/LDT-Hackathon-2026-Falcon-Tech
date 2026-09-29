"""Materialize the selected full-stack top-25 for an OOF specialist.

The cache is intentionally built from the score *after* every selected
streaming component (local verifier, patch verifier, family GNN, negative
gate, and static-gallery linkers).  This avoids the historical mistake of
training a specialist on the initial retrieval shortlist while evaluating it
on a different, downstream ranking.
"""

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
)
from scripts.probe_family_gnn_final import _score_one
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_train_fitted_verifier import _colors
from scripts.probe_streaming_official import _protocols
from scripts.train_cross_image_transformer import CrossImageTransformer, _load
from scripts.train_dino_patch_matcher import _load as _load_aligned
from scripts.train_dino_token_cross_top50 import DEVICE, DinoTokenCrossMatcher
from scripts.train_family_negative_gate import _predict as _predict_gate
from scripts.train_family_lambdarank import _predict as _predict_ranker
from scripts.train_streaming_top25_verifier import _local_for_protocol
from scripts.train_strict_family_gnn import (
    FamilyGraphReranker,
    _episode,
    _load_features,
    _predict as _predict_family,
)


TOP = 25
CACHE = Path(os.environ.get(
    "TOP25_CACHE",
    "outputs/retrieval_v2/lbs_lambdarank_top25_val.npz",
))
REPORT = Path(
    os.environ.get(
        "TOP25_SOURCE_REPORT",
        "outputs/retrieval_v2/"
        "family_gnn_final_smoothap_gate_highres_reject_modern_metricdino_"
        "lbs_lambdarank_appearance.json",
    )
)
METRIC_DINO_WEIGHT_OVERRIDE = os.environ.get(
    "METRIC_DINO_WEIGHT_OVERRIDE", ""
).strip()


def _normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _take(matrix: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.take_along_axis(matrix, candidate, axis=1).astype(np.float32)


def _map_candidate(values: np.ndarray, source: np.ndarray,
                   target: np.ndarray, gallery_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Map per-source-candidate values onto an arbitrary downstream shortlist."""
    rows = np.arange(len(source))[:, None]
    lookup = np.full((len(source), gallery_count), -1, dtype=np.int16)
    lookup[rows, source] = np.arange(source.shape[1], dtype=np.int16)[None]
    position = lookup[rows, target]
    valid = position >= 0
    safe = np.maximum(position, 0)
    if values.ndim == 2:
        mapped = values[rows, safe]
        mapped = np.where(valid, mapped, 0.0)
    else:
        mapped = values[rows, safe]
        mapped = np.where(valid[..., None], mapped, 0.0)
    return mapped.astype(np.float32), valid.astype(np.float32)


def _submatrix(matrix: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return matrix[candidate[:, :, None], candidate[:, None, :]].astype(np.float32)


def _z(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(1, keepdims=True)) / (
        values.std(1, keepdims=True) + 1e-6
    )


def _build_nodes(frame, protocol, data, episode, score, family, gate,
                 local, patch, highres, fused, osnet, dino):
    candidate = np.argsort(-score, axis=1, kind="stable")[:, :TOP]
    rows = np.arange(len(candidate))[:, None]

    raw_osnet = osnet[protocol["qi"]] @ osnet[protocol["gi"]].T
    raw_dino = dino[protocol["qi"]] @ dino[protocol["gi"]].T
    source_names = [
        "final", "dba_fused", "raw_fused", "raw_osnet", "raw_dino",
        "part_mean", "part_h2", "part_h4", "metric_dino",
    ]
    full_sources = [
        score,
        data["base"],
        data["raw_fused"],
        raw_osnet,
        raw_dino,
        data["part_scores"]["mean"],
        data["part_scores"]["h2"],
        data["part_scores"]["h4"],
        data["part_scores"]["metric_dino"],
    ]
    raw = np.stack([_take(value, candidate) for value in full_sources], axis=2)
    raw_z = _z(raw)

    rank = np.broadcast_to(
        np.linspace(0.0, 1.0, TOP, dtype=np.float32)[None, :, None],
        (len(candidate), TOP, 1),
    )
    top_gap = raw[..., :1] - raw[:, :1, :1]
    previous = np.concatenate((raw[:, :1, :1], raw[:, :-1, :1]), axis=1)
    previous_gap = raw[..., :1] - previous

    episode_value, in_top50 = _map_candidate(
        episode["features"], episode["candidate"], candidate, score.shape[1]
    )
    pair_source = data["candidates"]
    pair_values = np.stack((local, patch, highres), axis=2)
    mapped_pair, in_top25 = _map_candidate(
        pair_values, pair_source, candidate, score.shape[1]
    )
    family_gate = np.stack((family, gate), axis=2)
    mapped_family, _ = _map_candidate(
        family_gate, episode["candidate"], candidate, score.shape[1]
    )
    mapped = np.concatenate((mapped_pair, mapped_family), axis=2)
    mapped_z = _z(mapped)
    mapped_z *= in_top25[..., None]

    relation_names = (
        "gallery_fused", "gallery_osnet", "gallery_token",
        "gallery_token_valid", "gallery_modern", "gallery_patch",
        "gallery_raw_dino",
    )
    token = np.where(
        episode["gallery_token_valid"] > 0.5,
        episode["gallery_token_similarity"],
        -1.0,
    )
    gallery_dino = dino[protocol["gi"]] @ dino[protocol["gi"]].T
    relations = np.stack(
        tuple(_submatrix(value, candidate) for value in (
            data["gallery_sources"]["fused"],
            data["gallery_sources"]["osnet"],
            token,
            episode["gallery_token_valid"],
            data["gallery_sources"]["modern"],
            data["gallery_sources"]["patch"],
            gallery_dino,
        )),
        axis=3,
    ).astype(np.float32)
    diagonal = np.arange(TOP)
    relations[:, diagonal, diagonal, :] = 0.0
    relations = np.nan_to_num(relations, nan=0.0, posinf=1.0, neginf=-1.0)

    support_source = relations.copy()
    support_source[:, diagonal, diagonal, :] = -1e4
    strongest = np.max(support_source, axis=2)
    top3 = np.partition(support_source, -3, axis=2)[:, :, -3:, :].mean(2)

    node = np.concatenate(
        (
            raw_z[..., :1],
            raw,
            raw_z[..., 1:],
            rank,
            top_gap,
            previous_gap,
            episode_value,
            mapped,
            mapped_z,
            in_top50[..., None],
            in_top25[..., None],
            strongest,
            top3,
        ),
        axis=2,
    ).astype(np.float32)
    node_names = (
        ["final_z"]
        + source_names
        + [f"{name}_z" for name in source_names[1:]]
        + ["rank", "gap_from_top", "gap_from_previous"]
        + [f"episode_{index}" for index in range(episode_value.shape[2])]
        + ["local", "patch", "highres", "family", "gate"]
        + ["local_z", "patch_z", "highres_z", "family_z", "gate_z"]
        + ["in_initial_top50", "in_initial_top25"]
        + [f"{name}_max" for name in relation_names]
        + [f"{name}_top3" for name in relation_names]
    )

    q_vehicle = frame.vehicle_id.to_numpy()[protocol["qi"]]
    q_camera = frame.camera_id.to_numpy()[protocol["qi"]]
    g_vehicle = frame.vehicle_id.to_numpy()[protocol["gi"]]
    g_camera = frame.camera_id.to_numpy()[protocol["gi"]]
    label = g_vehicle[candidate] == q_vehicle[:, None]
    junk = label & (g_camera[candidate] == q_camera[:, None])
    label &= ~junk
    valid = ~junk
    total_positive = np.asarray([
        np.sum((g_vehicle == vehicle) & (g_camera != camera))
        for vehicle, camera in zip(q_vehicle, q_camera, strict=True)
    ], dtype=np.int16)
    return {
        "candidate": candidate.astype(np.int16),
        "base": raw[..., 0].astype(np.float32),
        "node": node.astype(np.float16),
        "relation": relations.astype(np.float16),
        "label": label,
        "valid": valid,
        "n_positive": total_positive,
        "feature_names": node_names,
        "relation_names": relation_names,
    }


def main():
    if CACHE.exists():
        print(f"cache already exists: {CACHE}", flush=True)
        return
    os.environ.setdefault(
        "OSNET_ENSEMBLE_PATH",
        "outputs/expert_fusion/osnet_loss_branch_ensemble_val_embeddings.npy",
    )
    frame = pd.read_csv("splits/val.csv")
    fused, osnet, dino, tokens = _load_features(frame, "val")
    with np.load(
        os.environ.get(
            "DINO_PARTS_VAL_PATH",
            "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz",
        ),
        allow_pickle=False,
    ) as archive:
        parts = {name: _normalized(archive[name]) for name in ("mean", "h2", "h4")}
    if METRIC_DINO_WEIGHT_OVERRIDE and float(METRIC_DINO_WEIGHT_OVERRIDE) == 0.0:
        # Keep downstream feature construction well-defined without loading a
        # second DINOv2 state.  Its residual weight is forced to zero, so this
        # alias cannot affect the ranking score.
        parts["metric_dino"] = parts["mean"]
    else:
        metric_dino = _load_aligned(
            "outputs/expert_fusion/cache/dinov2_vehicle_metric_part_cont_val.npz",
            frame, "embeddings",
        )
        parts["metric_dino"] = _normalized(metric_dino)
    conv_parts = _load(
        "outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", frame, "parts"
    )
    osnet_parts = _load(
        os.environ.get(
            "OSNET_PARTS_PATH",
            "outputs/expert_fusion/cache/osnet_smoothap_val_parts4.npz",
        ),
        frame,
        "parts",
    )
    highres_tokens = _load_aligned(
        "outputs/expert_fusion/cache/dinov2_vehicle_patches_val_10x10.npz",
        frame, "tokens",
    )

    token_state = torch.load(
        "weights/dino_token_cross_top50.pt", map_location=DEVICE, weights_only=False
    )
    token_model = DinoTokenCrossMatcher().to(DEVICE)
    token_model.load_state_dict(token_state["model_state"])
    token_model.eval()
    family_state = torch.load(
        "weights/strict_family_gnn_smoothap.pt",
        map_location=DEVICE, weights_only=False,
    )
    family_model = FamilyGraphReranker(family_state["feature_dim"]).to(DEVICE)
    family_model.load_state_dict(family_state["model_state"])
    family_model.eval()
    gate_model = joblib.load(os.environ.get(
        "FAMILY_GATE_MODEL",
        "weights/family_lambdarank_appearance_lbs.joblib",
    ))
    gate_colors = (
        _colors(frame, "val")
        if getattr(gate_model, "n_features_in_", 41) > 41
        else None
    )
    verifier_state = torch.load(
        "weights/dino_top25_verifier.pt", map_location=DEVICE, weights_only=False
    )
    verifier = CrossImageTransformer(dim=96).to(DEVICE)
    verifier.load_state_dict(verifier_state["model_state"])
    verifier.eval()
    patch_model = joblib.load("weights/dino_patch_matcher.joblib")
    highres_model = joblib.load(os.environ.get(
        "DINO_HIGHRES_MODEL",
        "weights/dino_patch_matcher_10x10_lbs.joblib",
    ))

    final_report = json.loads(REPORT.read_text(encoding="utf-8"))
    selected = final_report["selected_tune"]
    selected_reject = final_report["selected_gate_reject"]
    selected_modern = final_report["selected_modern_linker"]
    modern_report_path = os.environ.get("MODERN_LINKER_REPORT", "").strip()
    if modern_report_path:
        modern_report = json.loads(
            Path(modern_report_path).read_text(encoding="utf-8")
        )
        selected_modern = {
            key: modern_report["selected_tune"][key]
            for key in (
                "neighbor_k", "reciprocal_rank", "threshold", "propagation"
            )
        }
    selected_metric_dino_weight = final_report.get(
        "selected_metric_dino_weight", 0.0
    )
    if METRIC_DINO_WEIGHT_OVERRIDE:
        selected_metric_dino_weight = float(METRIC_DINO_WEIGHT_OVERRIDE)
    previous_report = json.loads(Path(
        "outputs/expert_fusion/dino_verifier_streaming.json"
    ).read_text(encoding="utf-8"))
    previous_keys = (
        "verifier_weight", "part_source", "part_weight", "patch_weight",
        "group_source", "neighbor_k", "threshold", "propagation",
    )
    previous = {key: previous_report["selected_tune"][key] for key in previous_keys}
    linker_report = json.loads(Path(
        "outputs/retrieval_v2/token_gallery_linker.json"
    ).read_text(encoding="utf-8"))
    linker_keys = (
        "keep_patch_linker", "neighbor_k", "reciprocal_rank", "threshold",
        "propagation",
    )
    linker = {key: linker_report["selected_tune"][key] for key in linker_keys}

    arrays: dict[str, np.ndarray] = {}
    feature_names = relation_names = None
    seeds = TUNE_SEEDS + CONFIRM_SEEDS
    with np.load(
        os.environ.get(
            "MODERN_LINKER_CACHE",
            "outputs/retrieval_v2/modern_gallery_linker_val.npz",
        ),
        allow_pickle=False,
    ) as modern_archive:
        for position, seed in enumerate(seeds, start=1):
            print(f"building full-stack specialist cache {position}/{len(seeds)} seed={seed}", flush=True)
            protocol = _protocols(frame, (seed,), {"base": fused})[0]
            episode = _episode(frame, fused, osnet, dino, tokens, token_model, seed)
            family = _predict_family(family_model, episode)
            gate = (
                _predict_ranker(gate_model, episode, family, frame, gate_colors)
                if gate_colors is not None
                else _predict_gate(gate_model, episode, family)
            )
            data = _prepare(protocol, fused, osnet, parts)
            local = _local_for_protocol(
                verifier, protocol, data, conv_parts, osnet_parts, osnet
            )
            patch = _predict_patch_plain(
                patch_model, protocol, data, tokens, fused, osnet, dino
            )
            highres = _predict_patch_plain(
                highres_model, protocol, data, highres_tokens, fused, osnet, dino,
                grid_size=10,
            )
            data["gallery_sources"]["patch"] = _patch_gallery_affinity(
                patch_model, protocol, tokens, fused, osnet, dino
            )
            data["gallery_sources"]["modern"] = modern_archive[
                f"affinity_{seed}"
            ].copy()
            score = _score_one(
                data, local, patch, episode, family, gate, highres,
                previous, linker,
                selected["query_token_weight"], selected["family_weight"],
                selected["gate_weight"], selected["highres_weight"],
                gate_reject=selected_reject, modern_linker=selected_modern,
                metric_dino_weight=selected_metric_dino_weight,
            )
            built = _build_nodes(
                frame, protocol, data, episode, score, family, gate,
                local, patch, highres, fused, osnet, dino,
            )
            prefix = f"s{seed}_"
            for name in (
                "candidate", "base", "node", "relation", "label", "valid",
                "n_positive",
            ):
                arrays[prefix + name] = built[name]
            arrays[prefix + "qi"] = protocol["qi"].astype(np.int16)
            arrays[prefix + "gi"] = protocol["gi"].astype(np.int16)
            feature_names = built["feature_names"]
            relation_names = built["relation_names"]
            del protocol, episode, data, local, patch, highres, family, gate, score

    arrays["seeds"] = np.asarray(seeds, dtype=np.int32)
    arrays["feature_names"] = np.asarray(feature_names)
    arrays["relation_names"] = np.asarray(relation_names)
    arrays["source_report"] = np.asarray(REPORT.as_posix())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **arrays)
    print(json.dumps({
        "cache": str(CACHE),
        "seeds": list(seeds),
        "feature_dim": len(feature_names),
        "relation_dim": len(relation_names),
        "bytes": CACHE.stat().st_size,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
