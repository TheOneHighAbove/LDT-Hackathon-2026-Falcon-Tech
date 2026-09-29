import torch

from scripts.train_osnet_target import _group_topk_loss


def test_group_topk_loss_rewards_all_positives_above_boundary():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    cameras = torch.tensor([0, 1, 2, 0, 1, 2])
    compact = torch.tensor([
        [1.0, 0.0], [0.99, 0.01], [0.98, -0.01],
        [0.0, 1.0], [0.01, 0.99], [-0.01, 0.98],
    ])
    scattered = compact.clone()
    scattered[2] = torch.tensor([0.1, 0.9])
    assert _group_topk_loss(compact, labels, cameras, shortlist_k=2) < _group_topk_loss(
        scattered, labels, cameras, shortlist_k=2
    )


def test_group_topk_loss_is_permutation_invariant():
    torch.manual_seed(17)
    embeddings = torch.randn(12, 8)
    labels = torch.arange(4).repeat_interleave(3)
    cameras = torch.arange(3).repeat(4)
    permutation = torch.randperm(12)
    expected = _group_topk_loss(embeddings, labels, cameras, shortlist_k=4)
    observed = _group_topk_loss(
        embeddings[permutation], labels[permutation], cameras[permutation], shortlist_k=4
    )
    torch.testing.assert_close(expected, observed)
