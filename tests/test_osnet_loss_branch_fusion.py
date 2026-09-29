import numpy as np

from scripts.probe_osnet_loss_branch_fusion import _weighted_concat


def test_weighted_concat_represents_weighted_cosine_sum():
    rng = np.random.default_rng(7)
    first = rng.normal(size=(5, 3)).astype(np.float32)
    second = rng.normal(size=(5, 4)).astype(np.float32)
    fused = _weighted_concat((first, second), (0.25, 0.75))
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    expected = 0.25 * (first @ first.T) + 0.75 * (second @ second.T)
    np.testing.assert_allclose(fused @ fused.T, expected, atol=2e-6)
