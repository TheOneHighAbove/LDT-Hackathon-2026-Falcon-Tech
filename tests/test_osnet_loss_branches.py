import torch

from scripts.train_osnet_loss_branches import _balanced_embedding
from src.osnet_trainable import VehicleOSNetAIN


def test_osnet_forward_branches_preserve_legacy_forward():
    model = VehicleOSNetAIN().eval()
    image = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        branches = model.forward_branches(image)
        combined = model(image)
    assert len(branches) == 2
    assert branches[0].shape == branches[1].shape == (2, 256)
    torch.testing.assert_close(combined, torch.cat(branches, dim=1))


def test_balanced_embedding_gives_equal_branch_energy():
    first = torch.randn(4, 256) * 20.0
    second = torch.randn(4, 256) * 0.01
    embedding = _balanced_embedding((first, second))
    assert embedding.shape == (4, 512)
    torch.testing.assert_close(
        embedding.norm(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6
    )
    expected = torch.full((4,), 2 ** -0.5)
    torch.testing.assert_close(
        embedding[:, :256].norm(dim=1), expected, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        embedding[:, 256:].norm(dim=1), expected, atol=1e-6, rtol=1e-6
    )
