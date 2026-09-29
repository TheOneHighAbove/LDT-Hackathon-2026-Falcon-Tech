import torch

from scripts.train_contextual_multiquery_reranker import _loss


def test_contextual_loss_ignores_same_camera_junk_candidate():
    labels = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, False, True]])
    first = _loss(torch.tensor([[0.4, -100.0, 0.2]]), labels, valid)
    second = _loss(torch.tensor([[0.4, 100.0, 0.2]]), labels, valid)
    torch.testing.assert_close(first, second)
