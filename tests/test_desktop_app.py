from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from prototype.desktop_app import (
    docker_commands,
    docker_daemon_ready,
    safe_run_directory,
    validate_dataset_directory,
)


def test_dataset_validation_reports_missing_contract(tmp_path: Path) -> None:
    assert validate_dataset_directory(tmp_path) == [
        f"Нет файла {tmp_path / 'test_query.csv'}",
        f"Нет файла {tmp_path / 'test_gallery.csv'}",
        f"Нет каталога {tmp_path / 'images'}",
    ]


def test_safe_run_directory_never_overwrites_bundle(tmp_path: Path) -> None:
    selected = tmp_path / "score_optimized_speed"
    selected.mkdir()
    assert safe_run_directory(selected) == selected.resolve()

    (selected / "submission.csv").write_text("result", encoding="utf-8")
    moment = datetime(2026, 9, 25, 14, 30, 0, tzinfo=timezone.utc)
    expected = tmp_path / "runs" / "run_20260925_143000"
    assert safe_run_directory(selected, moment) == expected

    expected.mkdir(parents=True)
    assert safe_run_directory(selected, moment) == Path(f"{expected}_1")


def test_docker_command_writes_only_to_mounted_output() -> None:
    commands = docker_commands(rebuild=True)
    assert commands[0] == ["docker", "compose", "build", "reid-api"]
    run = commands[1]
    assert run[:7] == [
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "reid-api",
        "python",
    ]
    assert run[run.index("--output-dir") + 1] == "/app/outputs"


def test_docker_preflight_handles_missing_cli(monkeypatch) -> None:
    monkeypatch.setattr("prototype.desktop_app.shutil.which", lambda _name: None)
    assert docker_daemon_ready() is False
