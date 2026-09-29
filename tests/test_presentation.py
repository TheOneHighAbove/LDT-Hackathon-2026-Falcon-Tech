from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "presentation" / "Falcon_Tech_Vehicle_ReID.pptx"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _deck_text() -> str:
    fragments: list[str] = []
    with ZipFile(DECK) as archive:
        slide_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for name in slide_names:
            root = ElementTree.fromstring(archive.read(name))
            fragments.extend(
                node.text or "" for node in root.iter(f"{{{DRAWING_NS}}}t")
            )
    return "\n".join(fragments)


def test_final_deck_has_no_internal_todos_or_placeholders() -> None:
    text = _deck_text()
    forbidden = (
        "[ВПИСАТЬ",
        "Что осталось перед сдачей",
        "нужен финальный замер",
        "Заполнить / подтвердить",
    )
    assert all(fragment not in text for fragment in forbidden)


def test_final_deck_contains_delivery_identity_and_entrypoint() -> None:
    text = _deck_text()
    assert "TryToBeatMyCodex" in text
    assert "Роман Захаров · Александр Пирогов" in text
    assert "github.com/TheOneHighAbove/LDT-Hackathon-2026-Falcon-Tech" in text
    assert "START_GUI.cmd" in text
