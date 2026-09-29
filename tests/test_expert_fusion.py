from __future__ import annotations

import numpy as np
import pytest

from src.expert_fusion import weighted_concatenate


def test_weighted_concatenation_has_exact_weighted_cosine() -> None:
    primary = np.asarray([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    expert = np.asarray([[0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    fused = weighted_concatenate(primary, expert, primary_weight=0.25)
    expected = 0.25 * (primary @ primary.T) + 0.75 * (expert @ expert.T)
    np.testing.assert_allclose(fused @ fused.T, expected, atol=1e-6)


def test_weighted_concatenation_rejects_invalid_weight() -> None:
    values = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="primary_weight"):
        weighted_concatenate(values, values, primary_weight=1.1)
