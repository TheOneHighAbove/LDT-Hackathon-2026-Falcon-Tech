"""Portable open-set refusal policies used by offline and API inference.

Legacy checkpoints use a scalar cosine threshold and are intentionally left
untouched.  A finalized checkpoint may additionally carry the frozen adaptive
top-1 correctness model described here.  Its metadata repeats and validates
the ranking score domain and post-processing contract so a calibration cannot
be accidentally reused with another gallery representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .config import normalize_postprocess_config
from .score_calibration import (
    ADAPTIVE_REFUSAL_DENSITY_K,
    ADAPTIVE_REFUSAL_FEATURE_NAMES,
    ADAPTIVE_REFUSAL_FEATURE_VARIANT,
    CandidateCorrectnessModel,
    extract_adaptive_refusal_features,
)


ADAPTIVE_REFUSAL_SCHEMA_VERSION = 1
ADAPTIVE_REFUSAL_TYPE = "adaptive_logistic_top1_correctness"
ADAPTIVE_FEATURE_EXTRACTOR_VERSION = 1

_METADATA_KEYS = {
    "schema_version",
    "type",
    "probability_threshold",
    "operating_point",
    "score_domain",
    "postprocess",
    "feature_extractor",
    "model",
    "provenance",
}
_EXTRACTOR_KEYS = {"name", "version", "density_k", "feature_names"}
_MODEL_KEYS = {
    "model_type",
    "version",
    "feature_names",
    "coefficients",
    "intercept",
    "feature_mean",
    "feature_scale",
}
_PROVENANCE_KEYS = {
    "calibration_report_sha256",
    "calibration_report_schema_version",
    "calibration_report_method",
    "calibrated_checkpoint_sha256",
    "threshold_source",
    "locked_confirmation",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"{name} schema mismatch: {', '.join(details)}")


def _finite_probability(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a probability within [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a probability within [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a probability within [0, 1]")
    return result


def _non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _sha256(value: Any, name: str) -> str:
    digest = _non_empty_text(value, name)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class AdaptiveRefusalPolicy:
    """Frozen NumPy-only top-1 acceptance policy."""

    model: CandidateCorrectnessModel
    probability_threshold: float
    operating_point: str
    score_domain: str
    postprocess: dict[str, bool | int | float]
    provenance: dict[str, Any]

    def probabilities(
        self,
        query_gallery_similarity: Any,
        gallery_gallery_similarity: Any,
    ) -> np.ndarray:
        features = extract_adaptive_refusal_features(
            query_gallery_similarity,
            gallery_gallery_similarity,
            density_k=ADAPTIVE_REFUSAL_DENSITY_K,
        )
        return self.model.predict_proba(features)[:, 1]

    def decisions(
        self,
        query_gallery_similarity: Any,
        gallery_gallery_similarity: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.probabilities(
            query_gallery_similarity, gallery_gallery_similarity
        )
        return probabilities >= self.probability_threshold, probabilities

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_REFUSAL_SCHEMA_VERSION,
            "type": ADAPTIVE_REFUSAL_TYPE,
            "probability_threshold": self.probability_threshold,
            "operating_point": self.operating_point,
            "score_domain": self.score_domain,
            "postprocess": dict(self.postprocess),
            "feature_extractor": {
                "name": ADAPTIVE_REFUSAL_FEATURE_VARIANT,
                "version": ADAPTIVE_FEATURE_EXTRACTOR_VERSION,
                "density_k": ADAPTIVE_REFUSAL_DENSITY_K,
                "feature_names": list(ADAPTIVE_REFUSAL_FEATURE_NAMES),
            },
            "model": self.model.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_metadata(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_score_domain: str,
        expected_postprocess: Mapping[str, Any],
    ) -> "AdaptiveRefusalPolicy":
        if not isinstance(payload, Mapping):
            raise ValueError("adaptive refusal metadata must be a mapping")
        _exact_keys(payload, _METADATA_KEYS, "adaptive refusal metadata")
        if payload["schema_version"] != ADAPTIVE_REFUSAL_SCHEMA_VERSION:
            raise ValueError("unsupported adaptive refusal schema version")
        if payload["type"] != ADAPTIVE_REFUSAL_TYPE:
            raise ValueError("unsupported adaptive refusal type")

        operating_point = _non_empty_text(
            payload["operating_point"], "adaptive refusal operating_point"
        )
        score_domain = _non_empty_text(
            payload["score_domain"], "adaptive refusal score_domain"
        )
        if score_domain != expected_score_domain:
            raise ValueError(
                "adaptive refusal score_domain does not match checkpoint search contract"
            )
        try:
            postprocess = normalize_postprocess_config(payload["postprocess"])
            authoritative_postprocess = normalize_postprocess_config(
                expected_postprocess
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid adaptive refusal postprocess: {exc}") from exc
        if postprocess != authoritative_postprocess:
            raise ValueError(
                "adaptive refusal postprocess does not match checkpoint search contract"
            )

        extractor = payload["feature_extractor"]
        if not isinstance(extractor, Mapping):
            raise ValueError("adaptive refusal feature_extractor must be a mapping")
        _exact_keys(extractor, _EXTRACTOR_KEYS, "adaptive feature_extractor")
        if extractor["name"] != ADAPTIVE_REFUSAL_FEATURE_VARIANT:
            raise ValueError("unsupported adaptive refusal feature extractor")
        if extractor["version"] != ADAPTIVE_FEATURE_EXTRACTOR_VERSION:
            raise ValueError("unsupported adaptive feature extractor version")
        if extractor["density_k"] != ADAPTIVE_REFUSAL_DENSITY_K:
            raise ValueError(
                f"adaptive feature extractor density_k must be {ADAPTIVE_REFUSAL_DENSITY_K}"
            )
        try:
            extractor_names = tuple(extractor["feature_names"])
        except TypeError as exc:
            raise ValueError("adaptive feature_names must be a sequence") from exc
        if extractor_names != ADAPTIVE_REFUSAL_FEATURE_NAMES:
            raise ValueError("adaptive feature_names do not match the frozen schema")

        model_payload = payload["model"]
        if not isinstance(model_payload, Mapping):
            raise ValueError("adaptive refusal model must be a mapping")
        _exact_keys(model_payload, _MODEL_KEYS, "adaptive refusal model")
        model = CandidateCorrectnessModel.from_dict(model_payload)
        if model.feature_names != ADAPTIVE_REFUSAL_FEATURE_NAMES:
            raise ValueError("adaptive model feature_names do not match extractor")

        provenance = payload["provenance"]
        if not isinstance(provenance, Mapping):
            raise ValueError("adaptive refusal provenance must be a mapping")
        _exact_keys(provenance, _PROVENANCE_KEYS, "adaptive refusal provenance")
        _sha256(
            provenance["calibration_report_sha256"],
            "adaptive provenance calibration_report_sha256",
        )
        _sha256(
            provenance["calibrated_checkpoint_sha256"],
            "adaptive provenance calibrated_checkpoint_sha256",
        )
        if provenance["calibration_report_schema_version"] != 1:
            raise ValueError("unsupported adaptive calibration report schema version")
        _non_empty_text(
            provenance["calibration_report_method"],
            "adaptive provenance calibration_report_method",
        )
        if provenance["threshold_source"] != "tune_oof_operating_point":
            raise ValueError("adaptive threshold must originate from tune OOF")
        if provenance["locked_confirmation"] is not True:
            raise ValueError("adaptive promotion requires locked confirmation")

        return cls(
            model=model,
            probability_threshold=_finite_probability(
                payload["probability_threshold"],
                "adaptive refusal probability_threshold",
            ),
            operating_point=operating_point,
            score_domain=score_domain,
            postprocess=postprocess,
            provenance=dict(provenance),
        )


def build_adaptive_refusal_metadata(
    model: CandidateCorrectnessModel,
    *,
    probability_threshold: float,
    operating_point: str,
    score_domain: str,
    postprocess: Mapping[str, Any],
    calibration_report_sha256: str,
    calibration_report_schema_version: int,
    calibration_report_method: str,
    calibrated_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build and self-validate a portable checkpoint payload."""

    policy = AdaptiveRefusalPolicy(
        model=model,
        probability_threshold=_finite_probability(
            probability_threshold, "adaptive refusal probability_threshold"
        ),
        operating_point=_non_empty_text(
            operating_point, "adaptive refusal operating_point"
        ),
        score_domain=_non_empty_text(score_domain, "adaptive refusal score_domain"),
        postprocess=normalize_postprocess_config(postprocess),
        provenance={
            "calibration_report_sha256": _sha256(
                calibration_report_sha256,
                "adaptive provenance calibration_report_sha256",
            ),
            "calibration_report_schema_version": calibration_report_schema_version,
            "calibration_report_method": _non_empty_text(
                calibration_report_method,
                "adaptive provenance calibration_report_method",
            ),
            "calibrated_checkpoint_sha256": _sha256(
                calibrated_checkpoint_sha256,
                "adaptive provenance calibrated_checkpoint_sha256",
            ),
            "threshold_source": "tune_oof_operating_point",
            "locked_confirmation": True,
        },
    )
    metadata = policy.to_metadata()
    # Exercise the same strict loader used in deployment before persistence.
    AdaptiveRefusalPolicy.from_metadata(
        metadata,
        expected_score_domain=score_domain,
        expected_postprocess=postprocess,
    )
    return metadata


def checkpoint_adaptive_refusal(
    checkpoint: Mapping[str, Any],
) -> AdaptiveRefusalPolicy | None:
    """Load an optional adaptive policy and bind it to checkpoint search."""

    refusal = checkpoint.get("refusal", {})
    if not isinstance(refusal, Mapping):
        raise ValueError("checkpoint refusal metadata must be a mapping")
    payload = refusal.get("adaptive")
    if payload is None:
        return None

    # Local import avoids a module cycle: engine checkpoint builders accept
    # opaque adaptive metadata but do not need to understand its internals.
    from .engine import checkpoint_postprocess, search_contract

    postprocess = checkpoint_postprocess(checkpoint)
    if postprocess is None:
        raise ValueError(
            "adaptive refusal requires an authoritative checkpoint search contract"
        )
    score_domain = str(search_contract(postprocess)["score_domain"])
    return AdaptiveRefusalPolicy.from_metadata(
        payload,
        expected_score_domain=score_domain,
        expected_postprocess=postprocess,
    )


__all__ = [
    "ADAPTIVE_FEATURE_EXTRACTOR_VERSION",
    "ADAPTIVE_REFUSAL_SCHEMA_VERSION",
    "ADAPTIVE_REFUSAL_TYPE",
    "AdaptiveRefusalPolicy",
    "build_adaptive_refusal_metadata",
    "checkpoint_adaptive_refusal",
]
