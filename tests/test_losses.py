import pytest


torch = pytest.importorskip("torch", reason="loss tests require PyTorch")

from src.losses import (
    ArcMarginProduct,
    BatchHardTripletLoss,
    CrossBatchMemory,
    MultiSimilarityLoss,
    ReIDLoss,
)


def test_arc_margin_shape_margin_and_gradients():
    head = ArcMarginProduct(2, 2, scale=1.0, margin=0.5)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))

    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    plain_logits = head(embeddings)
    margin_logits = head(embeddings, labels)

    assert margin_logits.shape == (2, 2)
    assert margin_logits.dtype == torch.float32
    assert torch.all(margin_logits.diagonal() < plain_logits.diagonal())
    # Non-target logits must be untouched by the angular margin.
    assert torch.allclose(margin_logits.flip(1).diagonal(), plain_logits.flip(1).diagonal())

    margin_logits.sum().backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert head.weight.grad is not None


def test_arcface_one_subcenter_preserves_legacy_shape_and_computation():
    head = ArcMarginProduct(
        3,
        2,
        scale=7.0,
        margin=0.3,
        num_subcenters=1,
    )
    legacy_weight = torch.tensor([[1.0, 2.0, 0.5], [-0.5, 1.0, 2.0]])
    # This is also the exact shape stored by checkpoints created before
    # sub-center support existed.
    head.load_state_dict({"weight": legacy_weight})
    assert head.weight.shape == (2, 3)

    embeddings = torch.tensor([[0.2, 0.5, 1.0], [1.0, -0.3, 0.1]])
    expected = torch.nn.functional.linear(
        torch.nn.functional.normalize(embeddings.float(), dim=1),
        torch.nn.functional.normalize(legacy_weight.float(), dim=1),
    ).clamp(-1.0 + 1e-7, 1.0 - 1e-7) * 7.0
    assert torch.equal(head(embeddings), expected)


def test_arcface_subcenters_select_max_before_angular_margin():
    head = ArcMarginProduct(
        2,
        2,
        scale=1.0,
        margin=0.4,
        num_subcenters=2,
    )
    with torch.no_grad():
        # Class-major layout: two centers for class 0, then two for class 1.
        head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        )
    embeddings = torch.tensor([[0.8, 0.2]], requires_grad=True)
    labels = torch.tensor([0])

    plain = head(embeddings)
    with_margin = head(embeddings, labels)
    normalized = torch.nn.functional.normalize(embeddings.detach(), dim=1)
    assert plain.shape == (1, 2)
    assert plain[0, 0].item() == pytest.approx(normalized[0, 0].item())
    assert plain[0, 1].item() == pytest.approx(-normalized[0, 1].item())
    assert with_margin[0, 0] < plain[0, 0]
    assert torch.equal(with_margin[:, 1], plain[:, 1])

    plain.sum().backward()
    gradients = head.weight.grad.reshape(2, 2, 2)
    assert torch.count_nonzero(gradients[0, 0]).item() > 0
    assert torch.count_nonzero(gradients[0, 1]).item() == 0
    assert torch.count_nonzero(gradients[1, 0]).item() == 0
    assert torch.count_nonzero(gradients[1, 1]).item() > 0


def test_arcface_subcenters_have_expected_shapes_and_gradients():
    head = ArcMarginProduct(5, 4, num_subcenters=3)
    embeddings = torch.randn(7, 5, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    logits = head(embeddings, labels)
    assert head.weight.shape == (12, 5)
    assert logits.shape == (7, 4)
    logits.square().mean().backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_batch_hard_triplet_is_finite_and_differentiable(metric):
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.0, 1.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss = BatchHardTripletLoss(margin=0.3, metric=metric)(embeddings, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0, 1, 2]),  # no positive for any anchor
        torch.tensor([4, 4, 4]),  # no negative for any anchor
    ],
)
def test_batch_hard_triplet_returns_differentiable_zero_without_valid_anchor(labels):
    embeddings = torch.randn(3, 5, requires_grad=True)
    loss = BatchHardTripletLoss()(embeddings, labels)

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.count_nonzero(embeddings.grad).item() == 0


def test_default_triplet_retains_fixed_margin_formula():
    embeddings = torch.tensor([[0.0], [2.0], [1.0], [4.0]])
    labels = torch.tensor([0, 0, 1, 2])
    criterion = BatchHardTripletLoss(
        margin=0.5,
        metric="euclidean",
        normalize_embeddings=False,
    )

    # Only the first two anchors have positives.  Their hard-negative
    # distances are 1 and 1, so both losses are relu(2 - 1 + 0.5).
    assert criterion(embeddings, labels).item() == pytest.approx(1.5)


def test_soft_margin_keeps_gradient_after_fixed_hinge_is_satisfied():
    fixed_embeddings = torch.tensor(
        [[0.0], [0.1], [10.0], [10.1]], requires_grad=True
    )
    soft_embeddings = fixed_embeddings.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1])
    options = {"metric": "euclidean", "normalize_embeddings": False}

    fixed = BatchHardTripletLoss(margin=0.3, **options)(fixed_embeddings, labels)
    soft = BatchHardTripletLoss(margin_mode="soft", **options)(
        soft_embeddings, labels
    )
    assert fixed.item() == pytest.approx(0.0)
    assert soft.item() > 0.0
    soft.backward()
    assert soft_embeddings.grad is not None
    assert torch.isfinite(soft_embeddings.grad).all()
    assert torch.count_nonzero(soft_embeddings.grad).item() > 0


def test_cross_camera_preferred_mines_cross_camera_and_falls_back_per_anchor():
    embeddings = torch.tensor([[0.0], [10.0], [1.0], [4.0]])
    labels = torch.tensor([0, 0, 0, 1])
    cameras = torch.tensor([0, 0, 1, 2])
    options = {
        "margin": 0.0,
        "metric": "euclidean",
        "normalize_embeddings": False,
    }
    all_positives = BatchHardTripletLoss(**options)(embeddings, labels)
    cross_camera = BatchHardTripletLoss(
        positive_mining="cross_camera_preferred", **options
    )(embeddings, labels, cameras)

    # Cross-camera restriction changes the first anchor's positive from
    # distance 10 to 1 while the other anchors still have valid positives.
    assert all_positives.item() == pytest.approx((6.0 + 4.0 + 6.0) / 3.0)
    assert cross_camera.item() == pytest.approx((0.0 + 3.0 + 6.0) / 3.0)

    # With no cross-camera positives, preferred mode must equal legacy mining.
    fallback_embeddings = torch.tensor([[0.0], [2.0], [4.0]])
    fallback_labels = torch.tensor([0, 0, 1])
    fallback_cameras = torch.tensor([7, 7, 8])
    legacy = BatchHardTripletLoss(**options)(fallback_embeddings, fallback_labels)
    preferred = BatchHardTripletLoss(
        positive_mining="cross_camera_preferred", **options
    )(fallback_embeddings, fallback_labels, fallback_cameras)
    assert torch.equal(legacy, preferred)


def test_cross_camera_only_ignores_anchors_without_cross_camera_positive():
    embeddings = torch.tensor([[0.0], [2.0], [4.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    cameras = torch.tensor([7, 7, 8])
    loss = BatchHardTripletLoss(
        metric="euclidean",
        normalize_embeddings=False,
        positive_mining="cross_camera_only",
    )(embeddings, labels, cameras)
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.count_nonzero(embeddings.grad).item() == 0


def test_reid_loss_accepts_model_output_and_returns_components():
    raw = torch.randn(4, 8, requires_grad=True)
    normalized = torch.nn.functional.normalize(raw, dim=1)
    output = {"embeddings": normalized, "features": raw}
    labels = torch.tensor([0, 0, 1, 1])
    criterion = ReIDLoss(
        embedding_dim=8,
        num_classes=3,
        arcface_scale=16.0,
        arcface_margin=0.3,
        triplet_metric="cosine",
        label_smoothing=0.1,
        classification_weight=1.0,
        triplet_weight=0.5,
    )

    details = criterion(output, labels, return_details=True)
    assert set(details) == {
        "loss",
        "classification_loss",
        "arcface_loss",
        "metric_loss",
        "triplet_loss",
        "logits",
    }
    assert details["logits"].shape == (4, 3)
    assert torch.allclose(details["arcface_loss"], details["classification_loss"])
    assert torch.equal(details["metric_loss"], details["triplet_loss"])
    assert torch.allclose(
        details["loss"],
        details["classification_loss"] + 0.5 * details["triplet_loss"],
    )

    details["loss"].backward()
    assert raw.grad is not None
    assert criterion.arcface.weight.grad is not None


def test_reid_loss_tensor_api_returns_scalar():
    embeddings = torch.randn(4, 6, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    criterion = ReIDLoss(6, 2)
    loss = criterion(embeddings, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_reid_loss_forwards_optional_camera_ids_to_triplet():
    embeddings = torch.randn(4, 6, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])
    criterion = ReIDLoss(
        6,
        2,
        triplet_margin_mode="soft",
        triplet_positive_mining="cross_camera_preferred",
    )
    details = criterion(
        embeddings,
        labels,
        camera_ids=cameras,
        return_details=True,
    )
    assert torch.isfinite(details["loss"])
    assert details["triplet_loss"].item() > 0.0


def test_reid_loss_configures_arcface_subcenters():
    criterion = ReIDLoss(6, 3, arcface_subcenters=2)
    embeddings = torch.randn(6, 6, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = criterion(embeddings, labels)
    assert criterion.arcface.num_subcenters == 2
    assert criterion.arcface.weight.shape == (6, 6)
    loss.backward()
    assert criterion.arcface.weight.grad is not None


def test_cross_batch_memory_is_bounded_fifo_and_detached():
    memory = CrossBatchMemory(capacity=3)
    first = torch.tensor([[1.0, 0.0], [2.0, 0.0]], requires_grad=True)
    memory.enqueue(first, torch.tensor([10, 20]), torch.tensor([1, 2]))
    memory.enqueue(
        torch.tensor([[3.0, 0.0], [4.0, 0.0]], requires_grad=True),
        torch.tensor([30, 40]),
        torch.tensor([3, 4]),
    )

    embeddings, labels, cameras = memory.snapshot()
    assert memory.count == 3
    assert embeddings.tolist() == [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    assert labels.tolist() == [20, 30, 40]
    assert cameras is not None and cameras.tolist() == [2, 3, 4]
    assert not embeddings.requires_grad
    assert not any("_embeddings" in key for key in memory.state_dict())

    memory.reset()
    assert memory.count == 0
    assert memory.snapshot()[0].shape == (0, 0)


def test_reid_loss_cross_batch_memory_adds_hard_candidates_and_resets():
    with_memory = ReIDLoss(
        2,
        3,
        triplet_margin=0.3,
        triplet_metric="cosine",
        label_smoothing=0.0,
        cross_batch_memory_capacity=8,
    )
    without_memory = ReIDLoss(
        2,
        3,
        triplet_margin=0.3,
        triplet_metric="cosine",
        label_smoothing=0.0,
    )
    # This singleton has no valid in-batch triplet, but becomes a very hard
    # negative for identity 0 in the following batch.
    with_memory(torch.tensor([[1.0, 0.0]]), torch.tensor([2]))
    assert with_memory.cross_batch_memory_count == 1

    current = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    xbm_details = with_memory(current, labels, return_details=True)
    baseline_details = without_memory(
        current.detach().clone().requires_grad_(True),
        labels,
        return_details=True,
    )
    assert xbm_details["triplet_loss"] > baseline_details["triplet_loss"]
    xbm_details["loss"].backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert with_memory.cross_batch_memory_count == 5

    # The queue is transient training state, not checkpoint state.
    assert not any(
        "cross_batch_memory._" in key for key in with_memory.state_dict()
    )
    with_memory.reset_cross_batch_memory()
    assert with_memory.cross_batch_memory_count == 0


def test_cross_batch_memory_is_ignored_and_not_mutated_in_eval_mode():
    criterion = ReIDLoss(3, 3, cross_batch_memory_capacity=4)
    embeddings = torch.randn(4, 3)
    labels = torch.tensor([0, 0, 1, 1])
    criterion.train()
    criterion(embeddings, labels)
    assert criterion.cross_batch_memory_count == 4

    criterion.eval()
    eval_with_queue = criterion(
        embeddings, labels, return_details=True
    )["triplet_loss"]
    assert criterion.cross_batch_memory_count == 4
    criterion.reset_cross_batch_memory()
    eval_without_queue = criterion(
        embeddings, labels, return_details=True
    )["triplet_loss"]
    assert torch.equal(eval_with_queue, eval_without_queue)
    assert criterion.cross_batch_memory_count == 0


def test_cross_batch_memory_supports_cross_camera_positive_mining():
    criterion = ReIDLoss(
        2,
        3,
        triplet_margin_mode="soft",
        triplet_positive_mining="cross_camera_only",
        cross_batch_memory_capacity=4,
    )
    criterion(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 2]),
        camera_ids=torch.tensor([2, 4]),
    )
    details = criterion(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        torch.tensor([0, 1]),
        camera_ids=torch.tensor([1, 1]),
        return_details=True,
    )
    assert torch.isfinite(details["triplet_loss"])
    assert details["triplet_loss"].item() > 0.0


def test_multi_similarity_matches_direct_formula_and_has_gradients():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [0.6, 0.8],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    alpha, beta, base = 2.0, 10.0, 0.5
    criterion = MultiSimilarityLoss(
        alpha=alpha,
        beta=beta,
        base=base,
        # Include every positive and negative pair so the expected value below
        # tests the loss formula independently from the mining boundary.
        epsilon=2.0,
    )
    actual = criterion(embeddings, labels)

    normalized = torch.nn.functional.normalize(embeddings.detach(), dim=1)
    similarities = normalized @ normalized.T
    expected_terms = []
    for anchor in range(len(labels)):
        positive = torch.tensor(
            [
                similarities[anchor, candidate]
                for candidate in range(len(labels))
                if candidate != anchor and labels[candidate] == labels[anchor]
            ]
        )
        negative = torch.tensor(
            [
                similarities[anchor, candidate]
                for candidate in range(len(labels))
                if labels[candidate] != labels[anchor]
            ]
        )
        expected_terms.append(
            torch.log1p(torch.exp(-alpha * (positive - base)).sum()) / alpha
            + torch.log1p(torch.exp(beta * (negative - base)).sum()) / beta
        )
    expected = torch.stack(expected_terms).mean()
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)

    actual.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_multi_similarity_pair_mining_can_return_differentiable_zero():
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        requires_grad=True,
    )
    loss = MultiSimilarityLoss(epsilon=0.1)(
        embeddings, torch.tensor([0, 0, 1, 1])
    )
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.count_nonzero(embeddings.grad).item() == 0


def test_multi_similarity_supports_cross_camera_xbm_candidates():
    criterion = ReIDLoss(
        2,
        3,
        metric_loss="multi_similarity",
        multi_similarity_epsilon=2.0,
        triplet_positive_mining="cross_camera_only",
        cross_batch_memory_capacity=8,
    )
    criterion(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 2]),
        camera_ids=torch.tensor([2, 4]),
    )
    details = criterion(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True),
        torch.tensor([0, 1]),
        camera_ids=torch.tensor([1, 1]),
        return_details=True,
    )
    assert criterion.metric_loss_type == "multi_similarity"
    assert isinstance(criterion.metric_objective, MultiSimilarityLoss)
    assert torch.isfinite(details["metric_loss"])
    assert details["metric_loss"].item() > 0.0
    assert criterion.cross_batch_memory_count == 4
    details["loss"].backward()


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("alpha", 0.0, "alpha"),
        ("beta", float("inf"), "beta"),
        ("base", 1.1, "base"),
        ("epsilon", -0.1, "epsilon"),
        ("positive_mining", "same_camera", "positive_mining"),
    ],
)
def test_invalid_multi_similarity_parameters_are_rejected(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        MultiSimilarityLoss(**{keyword: value})


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="num_subcenters"):
        ArcMarginProduct(4, 2, num_subcenters=0)
    with pytest.raises(ValueError, match="num_subcenters"):
        ArcMarginProduct(4, 2, num_subcenters=True)
    with pytest.raises(ValueError, match="metric"):
        BatchHardTripletLoss(metric="manhattan")
    with pytest.raises(ValueError, match="margin_mode"):
        BatchHardTripletLoss(margin_mode="smoothish")
    with pytest.raises(ValueError, match="positive_mining"):
        BatchHardTripletLoss(positive_mining="sometimes")
    with pytest.raises(ValueError, match="camera_ids are required"):
        BatchHardTripletLoss(positive_mining="cross_camera_preferred")(
            torch.randn(4, 3), torch.tensor([0, 0, 1, 1])
        )
    with pytest.raises(ValueError, match="camera_ids must have shape"):
        BatchHardTripletLoss()(
            torch.randn(4, 3),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 2]),
        )
    with pytest.raises(ValueError, match="label_smoothing"):
        ReIDLoss(8, 3, label_smoothing=1.0)
    with pytest.raises(ValueError, match="cross_batch_memory_capacity"):
        ReIDLoss(8, 3, cross_batch_memory_capacity=-1)
    with pytest.raises(ValueError, match="positive triplet_weight"):
        ReIDLoss(
            8,
            3,
            classification_weight=1.0,
            triplet_weight=0.0,
            cross_batch_memory_capacity=8,
        )
    with pytest.raises(ValueError, match="metric_loss"):
        ReIDLoss(8, 3, metric_loss="circle")
    with pytest.raises(ValueError, match="outside"):
        ArcMarginProduct(4, 2)(torch.randn(1, 4), torch.tensor([2]))
