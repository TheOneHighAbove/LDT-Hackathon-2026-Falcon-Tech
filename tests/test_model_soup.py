from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.engine import RAW_COSINE_SCORE_DOMAIN, search_contract
from src.model_soup import create_model_soup, main


def _checkpoint(
    value: float,
    *,
    counter: int,
    flag: bool = False,
) -> dict:
    return {
        "schema_version": 1,
        "model": {
            "backbone_name": "synthetic_backbone",
            "embedding_dim": 2,
            "pretrained": False,
            "backbone_kwargs": {},
        },
        "model_state": {
            "projection.weight": torch.tensor(
                [[value, value + 1.0], [value + 2.0, value + 3.0]],
                dtype=torch.float32,
            ),
            "bnneck.running_mean": torch.tensor(
                [value, value + 2.0], dtype=torch.float32
            ),
            "bnneck.running_var": torch.tensor(
                [value + 4.0, value + 6.0], dtype=torch.float32
            ),
            "bnneck.num_batches_tracked": torch.tensor(counter, dtype=torch.int64),
            "synthetic.enabled": torch.tensor(flag, dtype=torch.bool),
        },
        "preprocessing": {
            "input_size": 64,
            "bbox_padding": 0.05,
            "resize_mode": "direct",
            "tta_horizontal_flip": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "search": search_contract(None),
        "refusal": {
            "similarity_threshold": 0.2 + value / 100.0,
            "calibration": {
                "type": "platt",
                "slope": 5.0 + value,
                "intercept": -2.0,
            },
        },
        "metrics": {"retrieval": {"mAP": value / 10.0}},
        "source": {
            "epoch": int(value),
            "refusal_score_domain": RAW_COSINE_SCORE_DOMAIN,
        },
    }


def _save(path: Path, checkpoint: dict) -> Path:
    torch.save(checkpoint, path)
    return path


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def test_uniform_soup_averages_float_state_and_keeps_anchor_integer_buffers(
    tmp_path: Path,
) -> None:
    first = _save(tmp_path / "first.pt", _checkpoint(1.0, counter=7, flag=False))
    second = _save(tmp_path / "second.pt", _checkpoint(3.0, counter=99, flag=True))
    output = tmp_path / "soup.pt"

    summary = create_model_soup([first, second], output)
    soup = _load(output)
    state = soup["model_state"]

    assert torch.equal(
        state["projection.weight"],
        torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
    )
    assert torch.equal(
        state["bnneck.running_mean"], torch.tensor([2.0, 4.0])
    )
    assert state["bnneck.num_batches_tracked"].item() == 7
    assert state["synthetic.enabled"].item() is False
    assert soup["metrics"] == {}
    assert soup["refusal"] == {
        "similarity_threshold": 1.0,
        "calibration": {},
        "status": "requires_recalibration",
        "reason": "model_state_changed_by_model_soup",
    }
    provenance = soup["source"]["model_soup"]
    assert provenance["method"] == "uniform"
    assert provenance["bn_running_statistics_policy"] == "convex_mean"
    assert provenance["non_floating_state_policy"] == "copied_from_anchor"
    assert provenance["non_floating_member_differences"] == [
        "bnneck.num_batches_tracked",
        "synthetic.enabled",
    ]
    assert summary["requires_refusal_recalibration"] is True
    assert summary["weights"] == pytest.approx([0.5, 0.5])
    assert not list(tmp_path.glob(".soup.pt.*.tmp"))


def test_weighted_soup_normalizes_weights_and_uses_selected_anchor(
    tmp_path: Path,
) -> None:
    first = _save(tmp_path / "first.pt", _checkpoint(1.0, counter=7))
    second = _save(tmp_path / "second.pt", _checkpoint(3.0, counter=99))
    output = tmp_path / "weighted.pt"

    summary = create_model_soup(
        [first, second], output, weights=[1.0, 3.0], anchor_index=1
    )
    soup = _load(output)

    assert torch.equal(
        soup["model_state"]["projection.weight"],
        torch.tensor([[2.5, 3.5], [4.5, 5.5]], dtype=torch.float32),
    )
    assert soup["model_state"]["bnneck.num_batches_tracked"].item() == 99
    assert summary["method"] == "weighted"
    assert summary["weights"] == pytest.approx([0.25, 0.75])
    assert soup["source"]["model_soup"]["anchor_index"] == 1


def test_legacy_and_explicit_disabled_search_are_semantically_compatible(
    tmp_path: Path,
) -> None:
    legacy_checkpoint = _checkpoint(1.0, counter=1)
    legacy_checkpoint.pop("search")
    explicit_checkpoint = _checkpoint(2.0, counter=2)
    # Dormant parameters are intentionally different: disabled DBA/QE cannot
    # change either embeddings or the raw-cosine score domain.
    explicit_checkpoint["search"] = search_contract(
        {
            "enabled": False,
            "dba_top_k": 17,
            "dba_alpha": 7.0,
            "qe_top_k": 9,
            "qe_alpha": 4.0,
        }
    )
    first = _save(tmp_path / "legacy.pt", legacy_checkpoint)
    second = _save(tmp_path / "explicit.pt", explicit_checkpoint)
    output = tmp_path / "soup.pt"

    create_model_soup([first, second], output)
    soup = _load(output)

    assert soup["search"] == search_contract(None)
    assert (
        soup["source"]["model_soup"]["search_policy"]
        == "canonical_explicit_contract"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda checkpoint: checkpoint["model"].update({"embedding_dim": 3}),
            "model architecture differs",
        ),
        (
            lambda checkpoint: checkpoint["preprocessing"].update(
                {"input_size": 96}
            ),
            "preprocessing differs",
        ),
        (
            lambda checkpoint: checkpoint.update(
                {
                    "search": search_contract(
                        {
                            "enabled": True,
                            "dba_top_k": 3,
                            "dba_alpha": 2.0,
                            "qe_top_k": 2,
                            "qe_alpha": 1.0,
                        }
                    )
                }
            )
            or checkpoint["source"].update(
                {"refusal_score_domain": "cosine(expanded_query, dba_gallery)"}
            ),
            "search contract differs",
        ),
        (
            lambda checkpoint: checkpoint["source"].update(
                {
                    "robust_open_set_calibration": "report.json",
                    "robust_open_set_calibration_sha256": "a" * 64,
                }
            ),
            "refusal provenance differs",
        ),
    ],
)
def test_soup_rejects_incompatible_inference_contracts(
    tmp_path: Path, mutation, message: str
) -> None:
    first_checkpoint = _checkpoint(1.0, counter=1)
    second_checkpoint = _checkpoint(2.0, counter=2)
    mutation(second_checkpoint)
    first = _save(tmp_path / "first.pt", first_checkpoint)
    second = _save(tmp_path / "second.pt", second_checkpoint)

    with pytest.raises(ValueError, match=message):
        create_model_soup([first, second], tmp_path / "out.pt")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda state: state.pop("projection.weight"),
            "model_state keys differ",
        ),
        (
            lambda state: state.update(
                {"projection.weight": torch.ones(3, 2, dtype=torch.float32)}
            ),
            "shape differs",
        ),
        (
            lambda state: state.update(
                {"projection.weight": torch.ones(2, 2, dtype=torch.float64)}
            ),
            "dtype differs",
        ),
        (
            lambda state: state.update(
                {
                    "projection.weight": torch.tensor(
                        [[float("nan"), 1.0], [2.0, 3.0]]
                    )
                }
            ),
            "contains NaN or infinity",
        ),
    ],
)
def test_soup_rejects_incompatible_or_nonfinite_model_state(
    tmp_path: Path, mutation, message: str
) -> None:
    first_checkpoint = _checkpoint(1.0, counter=1)
    second_checkpoint = _checkpoint(2.0, counter=2)
    mutation(second_checkpoint["model_state"])
    first = _save(tmp_path / "first.pt", first_checkpoint)
    second = _save(tmp_path / "second.pt", second_checkpoint)

    with pytest.raises(ValueError, match=message):
        create_model_soup([first, second], tmp_path / "out.pt")


def test_soup_rejects_mismatched_or_incomplete_refusal_provenance(
    tmp_path: Path,
) -> None:
    first_checkpoint = _checkpoint(1.0, counter=1)
    second_checkpoint = _checkpoint(2.0, counter=2)
    second_checkpoint["source"]["refusal_score_domain"] = (
        "cosine(expanded_query, dba_gallery)"
    )
    first = _save(tmp_path / "first.pt", first_checkpoint)
    second = _save(tmp_path / "second.pt", second_checkpoint)

    with pytest.raises(ValueError, match="does not match its search contract"):
        create_model_soup([first, second], tmp_path / "out.pt")

    second_checkpoint = _checkpoint(2.0, counter=2)
    second_checkpoint["source"]["robust_open_set_calibration"] = "report.json"
    _save(second, second_checkpoint)
    with pytest.raises(ValueError, match="requires both report path and SHA-256"):
        create_model_soup([first, second], tmp_path / "out.pt")


def test_soup_protects_inputs_and_existing_outputs(tmp_path: Path) -> None:
    first = _save(tmp_path / "first.pt", _checkpoint(1.0, counter=1))
    second = _save(tmp_path / "second.pt", _checkpoint(2.0, counter=2))
    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"do not replace")

    with pytest.raises(ValueError, match="must not overwrite an input"):
        create_model_soup([first, second], first, overwrite=True)
    with pytest.raises(FileExistsError, match="already exists"):
        create_model_soup([first, second], existing)
    assert existing.read_bytes() == b"do not replace"


def test_soup_failure_does_not_modify_existing_output(tmp_path: Path) -> None:
    first = _save(tmp_path / "first.pt", _checkpoint(1.0, counter=1))
    bad_checkpoint = deepcopy(_checkpoint(2.0, counter=2))
    bad_checkpoint["model_state"]["projection.weight"] = torch.full(
        (2, 2), float("inf")
    )
    second = _save(tmp_path / "bad.pt", bad_checkpoint)
    output = tmp_path / "out.pt"
    original = b"existing checkpoint bytes"
    output.write_bytes(original)

    with pytest.raises(ValueError, match="contains NaN or infinity"):
        create_model_soup([first, second], output, overwrite=True)
    assert output.read_bytes() == original


def test_cli_supports_uniform_and_weighted_modes(tmp_path: Path, capsys) -> None:
    first = _save(tmp_path / "first.pt", _checkpoint(1.0, counter=1))
    second = _save(tmp_path / "second.pt", _checkpoint(2.0, counter=2))
    output = tmp_path / "cli.pt"

    main(
        [
            "--checkpoints",
            str(first),
            str(second),
            "--output",
            str(output),
            "--weights",
            "2",
            "1",
        ]
    )
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["method"] == "weighted"
    assert payload["weights"] == pytest.approx([2 / 3, 1 / 3])
    assert output.is_file()


def test_soup_rejects_duplicate_members_and_invalid_weights(tmp_path: Path) -> None:
    first = _save(tmp_path / "first.pt", _checkpoint(1.0, counter=1))
    duplicate = tmp_path / "duplicate.pt"
    duplicate.write_bytes(first.read_bytes())

    with pytest.raises(ValueError, match="distinct checkpoint contents"):
        create_model_soup([first, duplicate], tmp_path / "out.pt")
    with pytest.raises(ValueError, match="strictly positive"):
        create_model_soup(
            [first, _save(tmp_path / "second.pt", _checkpoint(2.0, counter=2))],
            tmp_path / "out.pt",
            weights=[1.0, 0.0],
        )
