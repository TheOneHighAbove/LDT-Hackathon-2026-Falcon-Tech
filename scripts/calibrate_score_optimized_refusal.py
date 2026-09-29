"""Calibrate refusal against the top-1 emitted by the frozen release ranker.

This is an offline maintenance command, not part of the submission runtime.
Tune seeds select the probability threshold and fit the logistic model; the
disjoint confirmation seeds are evaluated once with that frozen policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.infer_score_optimized import extract_features, rank_gallery
from src.metrics import evaluate_open_set
from src.release_reranking import refusal_features
from src.robust_open_set import (
    build_gallery_conditioned_protocol,
    select_constrained_candidate_threshold,
)
from src.score_calibration import (
    CandidateCorrectnessModel,
    FINAL_RANKER_REFUSAL_FEATURE_NAMES,
    evaluate_group_oof,
    fit_candidate_correctness_model,
)


TUNE_SEEDS = (1337, 2027, 3407, 4517, 7919)
CONFIRM_SEEDS = (21001, 22003, 23003, 24001, 25013)
TARGET_UNKNOWN_FRACTION = 0.20
TARGET_GALLERY_SIZE = 750
MINIMUM_TNR = 0.94


def _python_values(values: dict) -> dict:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in values.items()
    }


def _protocol_result(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    weights_dir: Path,
    release: dict,
    seed: int,
) -> dict:
    protocol = build_gallery_conditioned_protocol(
        frame,
        seed=seed,
        unknown_fraction=TARGET_UNKNOWN_FRACTION,
        target_gallery_size=TARGET_GALLERY_SIZE,
    )
    query_indices = protocol.query_indices
    gallery_indices = protocol.gallery_indices
    indices = np.concatenate((query_indices, gallery_indices))
    episode = frame.iloc[indices].reset_index(drop=True).copy()
    episode["split"] = ["query"] * len(query_indices) + [
        "gallery"
    ] * len(gallery_indices)
    episode_arrays = {
        name: values[indices] for name, values in arrays.items()
    }
    order, score, fused = rank_gallery(
        episode,
        episode_arrays,
        weights_dir,
        release,
        include_clip=False,
        colors=episode_arrays["colors"],
    )
    final_top1 = order[:, 0]
    pids = frame.vehicle_id.to_numpy()
    correct = (
        pids[query_indices] == pids[gallery_indices[final_top1]]
    )
    if np.any(correct & ~protocol.query_is_known):
        raise RuntimeError("unknown query has a correct gallery candidate")
    return {
        "seed": seed,
        "features": refusal_features(
            fused,
            len(query_indices),
            final_top1,
            score,
            release["refusal"],
        ),
        "known": protocol.query_is_known,
        "correct": correct,
        "groups": pids[query_indices],
        "queries": len(query_indices),
        "known_queries": int(protocol.query_is_known.sum()),
        "unknown_queries": int((~protocol.query_is_known).sum()),
        "unknown_query_fraction": float((~protocol.query_is_known).mean()),
        "correct_top1": int(correct.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=Path("configs/score_optimized_speed.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/score_optimized_speed/refusal_calibration.json"
        ),
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("splits/val.csv")
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("dataset/images")
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("weights")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    release = json.loads(args.release_config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.annotations).reset_index(drop=True)
    arrays, extraction = extract_features(
        frame,
        args.images_dir,
        args.weights_dir / "release",
        args.batch_size,
        args.workers,
        preprocessing=release["runtime"]["preprocessing"],
        dino_size=int(release["runtime"]["dino_input_size"]),
        include_clip=False,
    )
    tune = [
        _protocol_result(
            frame, arrays, args.weights_dir, release, seed
        )
        for seed in TUNE_SEEDS
    ]
    features = np.concatenate([row["features"] for row in tune])
    known = np.concatenate([row["known"] for row in tune])
    correct = np.concatenate([row["correct"] for row in tune])
    groups = np.concatenate([row["groups"] for row in tune])
    oof = evaluate_group_oof(
        features,
        correct,
        groups,
        n_splits=5,
        feature_names=FINAL_RANKER_REFUSAL_FEATURE_NAMES,
    )
    operating_point = select_constrained_candidate_threshold(
        oof["probabilities"],
        known,
        correct,
        minimum_tnr=MINIMUM_TNR,
    )
    model = fit_candidate_correctness_model(
        features,
        correct,
        feature_names=FINAL_RANKER_REFUSAL_FEATURE_NAMES,
    )

    confirmation = [
        _protocol_result(
            frame, arrays, args.weights_dir, release, seed
        )
        for seed in CONFIRM_SEEDS
    ]
    confirmation_features = np.concatenate(
        [row["features"] for row in confirmation]
    )
    confirmation_known = np.concatenate(
        [row["known"] for row in confirmation]
    )
    confirmation_correct = np.concatenate(
        [row["correct"] for row in confirmation]
    )
    probabilities = model.predict_proba(confirmation_features)[:, 1]
    metrics = evaluate_open_set(
        probabilities,
        confirmation_known,
        threshold=float(operating_point["threshold"]),
        top1_correct=confirmation_correct,
    )
    metrics = _python_values(metrics)
    metrics["official_score"] = (
        0.7 * float(metrics["f1"]) + 0.3 * float(metrics["tnr"])
    )
    previous_model = CandidateCorrectnessModel.from_dict(
        release["refusal"]["model"]
    )
    previous_probabilities = previous_model.predict_proba(
        confirmation_features
    )[:, 1]
    previous_metrics = evaluate_open_set(
        previous_probabilities,
        confirmation_known,
        threshold=float(release["refusal"]["probability_threshold"]),
        top1_correct=confirmation_correct,
    )
    previous_metrics = _python_values(previous_metrics)
    previous_metrics["official_score"] = (
        0.7 * float(previous_metrics["f1"])
        + 0.3 * float(previous_metrics["tnr"])
    )
    result = {
        "schema_version": 2,
        "description": (
            "Final-ranker top-1 correctness calibration with one query per "
            "identity and approximately 20% unknown queries; tune and "
            "confirmation seeds are disjoint."
        ),
        "target": "final_submission_top1_correctness",
        "protocol": {
            "target_unknown_query_fraction": TARGET_UNKNOWN_FRACTION,
            "target_gallery_size": TARGET_GALLERY_SIZE,
            "minimum_tnr": MINIMUM_TNR,
            "query_sampling": "one_query_per_identity",
        },
        "feature_domain": (
            "gallery-conditioned cosine evidence for the final top-1"
        ),
        "extraction": extraction,
        "tune_seeds": list(TUNE_SEEDS),
        "confirmation_seeds": list(CONFIRM_SEEDS),
        "tune_protocols": [
            {key: value for key, value in row.items() if key != "features"
             and key not in {"known", "correct", "groups"}}
            for row in tune
        ],
        "tune_oof": {
            "metrics": _python_values(oof["metrics"]),
            "operating_point": _python_values(operating_point),
            "folds": oof["folds"],
        },
        "model": model.to_dict(),
        "confirmation_protocols": [
            {key: value for key, value in row.items() if key != "features"
             and key not in {"known", "correct", "groups"}}
            for row in confirmation
        ],
        "confirmation": metrics,
        "previous_release_declared_confirmation": release["refusal"].get(
            "locked_confirmation"
        ),
        "previous_release_confirmation": previous_metrics,
        "confirmation_delta_vs_previous_release": {
            key: float(metrics[key]) - float(previous_metrics[key])
            for key in ("f1", "tnr", "official_score")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
