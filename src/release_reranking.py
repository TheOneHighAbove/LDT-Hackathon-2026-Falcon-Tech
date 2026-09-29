"""Pure inference-time reranking for the frozen score-optimized release.

The helpers in this module are deliberately independent of training and probe
scripts.  Each query is processed independently; the only shared context is
the fixed gallery supplied for the current evaluation run.
"""

from __future__ import annotations

import numpy as np

from src.release_features import patch_features
from src.release_models import (
    predict_cross_image,
    predict_dino_token,
)
from src.score_calibration import (
    ADAPTIVE_REFUSAL_FEATURE_NAMES,
    FINAL_RANKER_REFUSAL_FEATURE_NAMES,
    extract_adaptive_refusal_features,
    extract_final_ranker_refusal_features,
)


PATCH_TOP = 25
TOKEN_TOP = 50
PAIR_K = 16


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12
    )


def augment_static_gallery(
    embeddings: np.ndarray,
    *,
    top_k: int,
    alpha: float,
    self_weight: float = 1.0,
) -> np.ndarray:
    """Deterministically smooth each gallery row from gallery neighbours."""

    gallery = np.asarray(embeddings).astype(np.float64, copy=True)
    if gallery.ndim != 2 or not len(gallery) or not gallery.shape[1]:
        raise ValueError("gallery must be a non-empty matrix")
    if not np.isfinite(gallery).all():
        raise ValueError("gallery contains non-finite values")
    norms = np.linalg.norm(gallery, axis=1, keepdims=True)
    if (norms <= np.finfo(np.float64).eps).any():
        raise ValueError("gallery contains a zero-norm row")
    gallery /= norms
    if top_k < 1 or top_k >= len(gallery):
        raise ValueError("top_k must be positive and smaller than gallery")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not np.isfinite(self_weight) or self_weight <= 0.0:
        raise ValueError("self_weight must be finite and positive")
    similarities = np.clip(gallery @ gallery.T, -1.0, 1.0)
    ranking = similarities.copy()
    np.fill_diagonal(ranking, -np.inf)
    order = np.argsort(-ranking, axis=1, kind="mergesort")[:, :top_k]
    selected = np.take_along_axis(similarities, order, axis=1)
    weights = (
        np.ones_like(selected, dtype=np.float64)
        if alpha == 0.0
        else np.power(np.clip(selected, 0.0, 1.0), alpha)
    )
    augmented = self_weight * gallery + np.einsum(
        "gk,gkd->gd", weights, gallery[order], optimize=True
    )
    augmented_norm = np.linalg.norm(augmented, axis=1)
    degenerate = augmented_norm <= np.finfo(np.float64).eps
    if degenerate.any():
        augmented = augmented.copy()
        augmented[degenerate] = gallery[degenerate]
    return augmented / np.linalg.norm(augmented, axis=1, keepdims=True)


def zscore_rows(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(1, keepdims=True)) / (
        values.std(1, keepdims=True) + 1e-6
    )


def gallery_conditioned_refinement(
    query: np.ndarray,
    gallery: np.ndarray,
    *,
    top_k: int = 2,
    alpha: float = 1.0,
) -> np.ndarray:
    """Refine each query independently from its nearest static-gallery rows."""

    similarities = np.clip(query @ gallery.T, -1.0, 1.0)
    order = np.argsort(-similarities, axis=1, kind="mergesort")[:, :top_k]
    selected = np.take_along_axis(similarities, order, axis=1)
    weights = np.power(np.clip(selected, 0.0, 1.0), alpha)
    refined = query + np.einsum(
        "qk,qkd->qd", weights, gallery[order], optimize=True
    )
    return normalize(refined).astype(np.float32, copy=False)


def refusal_features(
    fused: np.ndarray,
    count_query: int,
    final_top1: np.ndarray,
    final_scores: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Build correctness features for the final emitted top-1 candidate."""

    query = fused[:count_query].astype(np.float32, copy=False)
    gallery = fused[count_query:].astype(np.float32, copy=False)
    augmented = augment_static_gallery(
        gallery,
        top_k=int(config["dba_top_k"]),
        alpha=float(config["dba_alpha"]),
    ).astype(np.float32, copy=False)
    refined = gallery_conditioned_refinement(
        query,
        augmented,
        top_k=int(config["gallery_refinement_top_k"]),
        alpha=float(config["gallery_refinement_alpha"]),
    )
    similarities = np.clip(refined @ augmented.T, -1.0, 1.0)
    gallery_similarity = np.clip(augmented @ augmented.T, -1.0, 1.0)
    feature_names = tuple(config["model"]["feature_names"])
    if feature_names == FINAL_RANKER_REFUSAL_FEATURE_NAMES:
        return extract_final_ranker_refusal_features(
            similarities,
            gallery_similarity,
            final_scores,
            final_top1,
            density_k=int(config["density_k"]),
        )
    if feature_names == ADAPTIVE_REFUSAL_FEATURE_NAMES:
        return extract_adaptive_refusal_features(
            similarities,
            gallery_similarity,
            density_k=int(config["density_k"]),
        )
    raise ValueError("unsupported refusal feature schema")


def prepare_retrieval(
    protocol: dict,
    fused: np.ndarray,
    osnet: np.ndarray,
    parts: dict[str, np.ndarray],
) -> dict:
    query = fused[protocol["qi"]]
    fused_gallery = augment_static_gallery(
        fused[protocol["gi"]], top_k=5, alpha=2.0
    )
    base = query @ fused_gallery.T
    candidates = np.argsort(-base, axis=1, kind="stable")[:, :PATCH_TOP]
    raw_fused = query @ fused[protocol["gi"]].T
    part_scores = {}
    for name, embedding in parts.items():
        gallery = augment_static_gallery(
            embedding[protocol["gi"]], top_k=5, alpha=2.0
        )
        part_scores[name] = embedding[protocol["qi"]] @ gallery.T
    osnet_gallery = augment_static_gallery(
        osnet[protocol["gi"]], top_k=5, alpha=2.0
    )
    return {
        "base": base,
        "candidates": candidates,
        "qi": protocol["qi"],
        "gi": protocol["gi"],
        "raw_fused": raw_fused,
        "part_scores": part_scores,
        "gallery_sources": {
            "fused": fused_gallery @ fused_gallery.T,
            "osnet": osnet_gallery @ osnet_gallery.T,
        },
    }


def patch_gallery_affinity(
    model,
    protocol: dict,
    tokens: np.ndarray,
    fused: np.ndarray,
    osnet: np.ndarray,
    dino: np.ndarray,
    *,
    neighbors: int = 16,
    grid_size: int | None = None,
) -> np.ndarray:
    indices = protocol["gi"]
    similarity = fused[indices] @ fused[indices].T
    np.fill_diagonal(similarity, -np.inf)
    candidate = np.argpartition(
        -similarity, neighbors - 1, axis=1
    )[:, :neighbors]
    query = np.repeat(indices, neighbors)
    gallery = indices[candidate.reshape(-1)]
    features = patch_features(
        tokens,
        fused,
        osnet,
        dino,
        query,
        gallery,
        grid_size=grid_size,
    )
    probability = model.predict_proba(features)[:, 1].reshape(
        len(indices), neighbors
    )
    affinity = np.zeros_like(similarity, dtype=np.float32)
    np.put_along_axis(affinity, candidate, probability, axis=1)
    affinity = np.minimum(affinity, affinity.T)
    np.fill_diagonal(affinity, -np.inf)
    return affinity


def predict_patch(
    model,
    protocol: dict,
    prepared: dict,
    tokens: np.ndarray,
    fused: np.ndarray,
    osnet: np.ndarray,
    dino: np.ndarray,
    *,
    grid_size: int | None = None,
) -> np.ndarray:
    candidate = prepared["candidates"]
    query = np.repeat(prepared["qi"], PATCH_TOP)
    gallery = prepared["gi"][candidate.reshape(-1)]
    features = patch_features(
        tokens,
        fused,
        osnet,
        dino,
        query,
        gallery,
        grid_size=grid_size,
    )
    return model.predict_proba(features)[:, 1].reshape(
        len(candidate), PATCH_TOP
    )


def base_reranked_scores(
    data: dict,
    local: np.ndarray,
    patch: np.ndarray,
    verifier_weight: float,
    part_source: str = "none",
    part_weight: float = 0.0,
    patch_weight: float = 0.0,
) -> np.ndarray:
    score = data["base"].copy()
    candidates = data["candidates"]
    base = np.take_along_axis(score, candidates, axis=1)
    raw = np.take_along_axis(data["raw_fused"], candidates, axis=1)
    reranked = base + verifier_weight * (local - raw)
    if part_source != "none" and part_weight:
        part = np.take_along_axis(
            data["part_scores"][part_source], candidates, axis=1
        )
        reranked += part_weight * (part - raw)
    if patch_weight:
        mean = reranked.mean(1, keepdims=True)
        std = reranked.std(1, keepdims=True) + 1e-6
        base_z = (reranked - mean) / std
        patch_z = (patch - patch.mean(1, keepdims=True)) / (
            patch.std(1, keepdims=True) + 1e-6
        )
        reranked = mean + std * (
            (1.0 - patch_weight) * base_z + patch_weight * patch_z
        )
    np.put_along_axis(score, candidates, reranked, axis=1)
    return score


def token_neighborhoods(
    affinity: np.ndarray,
    base_rank: np.ndarray,
    config: dict,
) -> list[np.ndarray]:
    order = np.argsort(-affinity, axis=1, kind="stable")
    neighborhoods = []
    for index, row in enumerate(order[:, : config["neighbor_k"]]):
        keep = []
        for other in row:
            if not np.isfinite(affinity[index, other]):
                continue
            if affinity[index, other] < config["threshold"]:
                continue
            if (
                max(base_rank[index, other], base_rank[other, index])
                >= config["reciprocal_rank"]
            ):
                continue
            keep.append(int(other))
        neighborhoods.append(
            np.asarray([index, *keep], dtype=np.int64)
        )
    return neighborhoods


def propagate(
    score: np.ndarray,
    neighborhoods: list[np.ndarray],
    amount: float,
) -> np.ndarray:
    original = score.copy()
    width = max(len(value) for value in neighborhoods)
    index = np.arange(len(neighborhoods), dtype=np.int64)[:, None]
    padded = np.broadcast_to(
        index, (len(neighborhoods), width)
    ).copy()
    for row, value in enumerate(neighborhoods):
        padded[row, : len(value)] = value
    support = original[:, padded].max(axis=2)
    return (1.0 - amount) * original + amount * support


def propagate_patch_knn(
    score: np.ndarray, data: dict, config: dict
) -> np.ndarray:
    affinity = data["gallery_sources"]["patch"]
    order = np.argsort(-affinity, axis=1, kind="stable")
    neighborhoods = []
    for index, row in enumerate(order[:, : config["neighbor_k"]]):
        valid = row[affinity[index, row] >= config["threshold"]]
        neighborhoods.append(np.concatenate(([index], valid)))
    original = score.copy()
    for index, neighborhood in enumerate(neighborhoods):
        support = original[:, neighborhood].max(axis=1)
        score[:, index] = (
            (1.0 - config["propagation"]) * original[:, index]
            + config["propagation"] * support
        )
    return score


def score_gallery(
    item: dict,
    local_value: np.ndarray,
    patch_value: np.ndarray,
    episode: dict,
    family_value: np.ndarray,
    gate_value: np.ndarray,
    highres_value: np.ndarray,
    previous: dict,
    linker: dict,
    *,
    query_token_weight: float,
    family_weight: float,
    gate_weight: float = 0.0,
    highres_weight: float = 0.0,
    gate_reject: dict | None = None,
    modern_linker: dict | None = None,
    metric_dino_weight: float = 0.0,
) -> np.ndarray:
    score = base_reranked_scores(
        item,
        local_value,
        patch_value,
        previous["verifier_weight"],
        previous["part_source"],
        previous["part_weight"],
        previous["patch_weight"],
    )
    if metric_dino_weight:
        score += metric_dino_weight * (
            item["part_scores"]["metric_dino"] - item["raw_fused"]
        )
    if highres_weight:
        patch_candidate = item["candidates"]
        patch_current = np.take_along_axis(
            score, patch_candidate, axis=1
        )
        patch_z = zscore_rows(highres_value)
        patch_current += (
            highres_weight
            * (patch_current.std(1, keepdims=True) + 1e-6)
            * patch_z
        )
        np.put_along_axis(
            score, patch_candidate, patch_current, axis=1
        )

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
        reranked += (
            gate_weight
            * (reranked.std(1, keepdims=True) + 1e-6)
            * zscore_rows(gate_value)
        )
    if gate_reject is not None:
        current_order = np.argsort(-reranked, axis=1, kind="stable")
        current_rank = np.empty_like(current_order)
        np.put_along_axis(
            current_rank,
            current_order,
            np.broadcast_to(
                np.arange(reranked.shape[1]), current_order.shape
            ),
            axis=1,
        )
        gate_order = np.argsort(-gate_value, axis=1, kind="stable")
        gate_rank = np.empty_like(gate_order)
        np.put_along_axis(
            gate_rank,
            gate_order,
            np.broadcast_to(
                np.arange(reranked.shape[1]), gate_order.shape
            ),
            axis=1,
        )
        rejected = (
            (current_rank < gate_reject["window"])
            & (
                (gate_rank - current_rank)
                >= gate_reject["rank_gap"]
            )
        )
        reranked -= (
            gate_reject["penalty"]
            * (reranked.std(1, keepdims=True) + 1e-6)
            * rejected
        )
    np.put_along_axis(score, candidate, reranked, axis=1)

    if linker["keep_patch_linker"]:
        score = propagate_patch_knn(score, item, previous)
    affinity = np.where(
        episode["gallery_token_valid"] > 0.5,
        episode["gallery_token_similarity"],
        -np.inf,
    )
    np.fill_diagonal(affinity, -np.inf)
    neighborhoods = token_neighborhoods(
        affinity, episode["gallery_rank"], linker
    )
    score = propagate(score, neighborhoods, linker["propagation"])
    if modern_linker is not None:
        modern_neighborhoods = token_neighborhoods(
            item["gallery_sources"]["modern"],
            episode["gallery_rank"],
            modern_linker,
        )
        score = propagate(
            score,
            modern_neighborhoods,
            modern_linker["propagation"],
        )
    return score


def context_features(
    episode: dict, family_score: np.ndarray
) -> np.ndarray:
    relation = episode["relation"].astype(np.float32)
    count = relation.shape[1]
    diagonal = np.eye(count, dtype=bool)[None]
    fused = np.where(diagonal, -np.inf, relation[..., 0])
    osnet = np.where(diagonal, -np.inf, relation[..., 1])
    token_valid = (relation[..., 3] > 0.5) & ~diagonal
    token_relation = np.where(
        token_valid, relation[..., 2], -np.inf
    )
    token_query = episode["features"][..., 1]

    top_index = np.argpartition(-fused, 4, axis=2)[..., :4]
    family_matrix = np.broadcast_to(
        family_score[:, None, :], fused.shape
    )
    token_query_matrix = np.broadcast_to(
        token_query[:, None, :], fused.shape
    )
    top_family = np.take_along_axis(
        family_matrix, top_index, axis=2
    )
    top_query = np.take_along_axis(
        token_query_matrix, top_index, axis=2
    )
    top_fused = np.take_along_axis(fused, top_index, axis=2)
    top_osnet = np.take_along_axis(osnet, top_index, axis=2)

    token_family_support = np.where(
        token_valid, family_matrix, -1e4
    ).max(2)
    token_query_support = np.where(
        token_valid, token_query_matrix, -1e4
    ).max(2)
    no_token = ~token_valid.any(2)
    token_family_support[no_token] = family_score[no_token]
    token_query_support[no_token] = token_query[no_token]
    token_max = token_relation.max(2)
    token_max[~np.isfinite(token_max)] = -1.0

    contextual = np.stack(
        (
            family_score,
            family_score - episode["features"][..., 0],
            top_family.max(2),
            top_family.mean(2),
            top_query.max(2),
            top_query.mean(2),
            token_family_support,
            token_query_support,
            token_max,
            token_valid.sum(2).astype(np.float32) / 8.0,
            top_fused.max(2),
            top_fused.mean(2),
            top_osnet.max(2),
            top_osnet.mean(2),
        ),
        axis=2,
    ).astype(np.float32)
    raw = np.concatenate((episode["features"], contextual), axis=2)
    z = (contextual - contextual.mean(1, keepdims=True)) / (
        contextual.std(1, keepdims=True) + 1e-6
    )
    return np.concatenate((raw, z), axis=2).astype(np.float32)


def _cosine(
    left: np.ndarray, right: np.ndarray, axis: int = -1
) -> np.ndarray:
    numerator = np.sum(left * right, axis=axis)
    denominator = np.linalg.norm(left, axis=axis) * np.linalg.norm(
        right, axis=axis
    )
    return numerator / np.maximum(denominator, 1e-6)


def appearance_features(
    frame, episode: dict, colors: np.ndarray
) -> np.ndarray:
    query = episode["qi"]
    gallery = episode["gi"][episode["candidate"]]
    query_color = colors[query].reshape(-1, 5, 32)[:, None]
    gallery_color = colors[gallery].reshape(*gallery.shape, 5, 32)
    blocks = []
    for start, stop in ((0, 32), (0, 16), (16, 24), (24, 32)):
        blocks.append(
            _cosine(
                query_color[..., start:stop],
                gallery_color[..., start:stop],
            )
        )
    region = np.concatenate(blocks, axis=2)
    aspect = np.log(
        np.maximum(
            frame.w.to_numpy(dtype=np.float32)
            / np.maximum(
                frame.h.to_numpy(dtype=np.float32), 1e-6
            ),
            1e-6,
        )
    )
    aspect_gap = np.abs(
        aspect[query, None] - aspect[gallery]
    )[..., None]
    aspect_product = (
        aspect[query, None] * aspect[gallery]
    )[..., None]
    return np.concatenate(
        (
            region,
            region.mean(2, keepdims=True),
            region.min(2, keepdims=True),
            region.max(2, keepdims=True),
            aspect_gap,
            aspect_product,
        ),
        axis=2,
    ).astype(np.float32)


def predict_family_ranker(
    model,
    frame,
    episode: dict,
    family_prediction: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    features = np.concatenate(
        (
            context_features(episode, family_prediction),
            appearance_features(frame, episode, colors),
        ),
        axis=2,
    )
    values = model.booster_.predict(
        features.reshape(-1, features.shape[2]),
        num_iteration=(
            model.best_iteration_ if model.best_iteration_ else -1
        ),
    )
    return values.reshape(features.shape[:2]).astype(np.float32)


def _rank_matrix(
    similarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-similarity, axis=1, kind="stable")
    rank = np.empty_like(order, dtype=np.int16)
    np.put_along_axis(
        rank,
        order,
        np.broadcast_to(
            np.arange(len(order), dtype=np.int16), order.shape
        ),
        axis=1,
    )
    return order, rank


def _candidate_pairs(similarities: dict[str, np.ndarray]):
    count = len(next(iter(similarities.values())))
    encoded, orders, ranks = [], {}, {}
    for name, similarity in similarities.items():
        search = similarity.copy()
        np.fill_diagonal(search, -np.inf)
        order, rank = _rank_matrix(search)
        orders[name], ranks[name] = order, rank
        first = np.repeat(
            np.arange(count, dtype=np.int64), PAIR_K
        )
        second = order[:, :PAIR_K].reshape(-1).astype(np.int64)
        low, high = np.minimum(first, second), np.maximum(first, second)
        encoded.append(low * count + high)
    unique = np.unique(np.concatenate(encoded))
    return unique // count, unique % count, orders, ranks


def _overlap(
    first: np.ndarray, second: np.ndarray, order: np.ndarray
) -> np.ndarray:
    output = np.empty((len(first), 4), dtype=np.float32)
    for row, (left, right) in enumerate(
        zip(first, second, strict=True)
    ):
        output[row] = [
            len(
                np.intersect1d(
                    order[left, :size],
                    order[right, :size],
                    assume_unique=True,
                )
            )
            / size
            for size in (3, 5, 10, 20)
        ]
    return output


def modern_linker_build(
    frame,
    indices: np.ndarray,
    fused: np.ndarray,
    osnet: np.ndarray,
    dino: np.ndarray,
    tokens: np.ndarray,
    patch_model,
) -> dict:
    local = frame.iloc[indices].reset_index(drop=True)
    fused_gallery = fused[indices]
    osnet_gallery = osnet[indices]
    dino_gallery = dino[indices]
    dba = augment_static_gallery(
        fused_gallery, top_k=5, alpha=2.0
    )
    similarities = {
        "dba": dba @ dba.T,
        "fused": fused_gallery @ fused_gallery.T,
        "osnet": osnet_gallery @ osnet_gallery.T,
        "dino": dino_gallery @ dino_gallery.T,
    }
    first, second, orders, ranks = _candidate_pairs(similarities)
    blocks = []
    count = len(indices)
    for similarity in similarities.values():
        _, rank = _rank_matrix(
            np.where(np.eye(count, dtype=bool), -np.inf, similarity)
        )
        blocks.extend(
            (
                similarity[first, second, None],
                (
                    np.minimum(rank[first, second], rank[second, first])
                    / count
                )[:, None],
                (
                    np.maximum(rank[first, second], rank[second, first])
                    / count
                )[:, None],
            )
        )
    for name in ("dba", "fused", "osnet", "dino"):
        blocks.append(_overlap(first, second, orders[name]))

    global_first, global_second = indices[first], indices[second]
    forward = patch_features(
        tokens,
        fused,
        osnet,
        dino,
        global_first,
        global_second,
        batch=2048,
        grid_size=5,
    )
    reverse = patch_features(
        tokens,
        fused,
        osnet,
        dino,
        global_second,
        global_first,
        batch=2048,
        grid_size=5,
    )
    forward_probability = patch_model.predict_proba(forward)[:, 1]
    reverse_probability = patch_model.predict_proba(reverse)[:, 1]
    blocks.append(
        np.stack(
            (
                np.minimum(forward_probability, reverse_probability),
                np.maximum(forward_probability, reverse_probability),
                (forward_probability + reverse_probability) / 2.0,
                np.abs(forward_probability - reverse_probability),
            ),
            axis=1,
        ).astype(np.float32)
    )
    aspect = np.log(
        np.maximum(local.w.to_numpy(), 1)
        / np.maximum(local.h.to_numpy(), 1)
    )
    area = np.log(
        np.maximum(local.w.to_numpy() * local.h.to_numpy(), 1)
    )
    blocks.append(
        np.stack(
            (
                np.abs(aspect[first] - aspect[second]),
                np.abs(area[first] - area[second]),
            ),
            axis=1,
        ).astype(np.float32)
    )
    return {
        "indices": indices,
        "first": first,
        "second": second,
        "features": np.concatenate(blocks, axis=1).astype(np.float32),
        "rank": ranks["dba"],
        "base": dba,
    }


def modern_affinity(data: dict, probability: np.ndarray) -> np.ndarray:
    count = len(data["indices"])
    output = np.full((count, count), -np.inf, dtype=np.float32)
    output[data["first"], data["second"]] = probability
    output[data["second"], data["first"]] = probability
    np.fill_diagonal(output, -np.inf)
    return output


def local_for_protocol(
    model,
    protocol: dict,
    prepared: dict,
    conv: np.ndarray,
    osnet: np.ndarray,
    embedding: np.ndarray,
) -> np.ndarray:
    candidates = prepared["candidates"]
    query = np.repeat(prepared["qi"], PATCH_TOP)
    gallery = prepared["gi"][candidates.reshape(-1)]
    return predict_cross_image(
        model, conv, osnet, embedding, query, gallery
    ).reshape(len(candidates), PATCH_TOP)


def _gallery_protocol(
    indices: np.ndarray, fused: np.ndarray
) -> dict:
    raw = fused[indices]
    gallery = augment_static_gallery(raw, top_k=5, alpha=2.0)
    score = raw @ gallery.T
    np.fill_diagonal(score, -np.inf)
    candidates = np.argsort(-score, axis=1, kind="stable")[
        :, :TOKEN_TOP
    ]
    return {
        "qi": indices,
        "gi": indices,
        "score": score.astype(np.float32),
        "candidates": candidates,
    }


def symmetric_token_matrix(
    model,
    indices: np.ndarray,
    tokens: np.ndarray,
    fused: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _gallery_protocol(indices, fused)
    prediction = predict_dino_token(model, data, tokens)
    count = len(indices)
    directed = np.full((count, count), -np.inf, dtype=np.float32)
    np.put_along_axis(
        directed, data["candidates"], prediction, axis=1
    )
    mutual = np.isfinite(directed) & np.isfinite(directed.T)
    score = np.where(
        mutual, 0.5 * (directed + directed.T), -1.0
    ).astype(np.float32)
    np.fill_diagonal(score, 1.0)
    base_order = np.argsort(-data["score"], axis=1, kind="stable")
    base_rank = np.empty_like(base_order, dtype=np.int16)
    np.put_along_axis(
        base_rank,
        base_order,
        np.broadcast_to(
            np.arange(count, dtype=np.int16), base_order.shape
        ),
        axis=1,
    )
    return score, mutual.astype(np.float32), base_rank


def gather_relation(
    candidate: np.ndarray,
    fused_similarity: np.ndarray,
    osnet_similarity: np.ndarray,
    token_similarity: np.ndarray,
    token_valid: np.ndarray,
    *,
    batch: int = 128,
) -> np.ndarray:
    output = []
    for start in range(0, len(candidate), batch):
        index = candidate[start : start + batch]
        left, right = index[:, :, None], index[:, None, :]
        relation = np.stack(
            (
                fused_similarity[left, right],
                osnet_similarity[left, right],
                token_similarity[left, right],
                token_valid[left, right],
            ),
            axis=3,
        )
        output.append(relation.astype(np.float16))
    return np.concatenate(output)


__all__ = [
    "PATCH_TOP",
    "TOKEN_TOP",
    "augment_static_gallery",
    "base_reranked_scores",
    "gallery_conditioned_refinement",
    "gather_relation",
    "local_for_protocol",
    "modern_affinity",
    "modern_linker_build",
    "patch_gallery_affinity",
    "predict_family_ranker",
    "predict_patch",
    "prepare_retrieval",
    "refusal_features",
    "score_gallery",
    "symmetric_token_matrix",
    "zscore_rows",
]
