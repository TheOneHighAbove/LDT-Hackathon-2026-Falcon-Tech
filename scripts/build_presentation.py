"""Legacy internal deck generator.

The submission deck is ``presentation/Falcon_Tech_Vehicle_ReID.pptx``.  This
script predates the final official-template rebuild and must not be used during
final packaging because it produces the older simplified design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]

BG = RGBColor(18, 18, 21)
PANEL = RGBColor(31, 31, 36)
PANEL_2 = RGBColor(40, 39, 46)
WHITE = RGBColor(244, 244, 246)
MUTED = RGBColor(184, 184, 195)
PINK = RGBColor(255, 0, 83)
PURPLE = RGBColor(138, 131, 209)
GREEN = RGBColor(68, 201, 143)
YELLOW = RGBColor(255, 196, 72)
FONT = "Montserrat"
TEAM_NAME = "TryToBeatMyCodex"
TEAM_MEMBERS = "Захаров Роман · Пирогов Александр"
TEAM_CONTACT = "roman.zakharov.d@gmail.com · @TheFool"
REPOSITORY_URL = "github.com/TheOneHighAbove/LDT-Hackathon-2026-Falcon-Tech"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _find_template(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    templates = [
        path
        for path in ROOT.glob("*.pptx")
        if "Falcon_Tech_Vehicle_ReID" not in path.name
    ]
    if len(templates) != 1:
        raise RuntimeError(
            "Expected exactly one organizer PPTX template in the repository root; "
            f"found {len(templates)}"
        )
    return templates[0]


def _remove_template_slides(prs: Presentation) -> None:
    for slide_id in list(prs.slides._sldIdLst):
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)


def _set_fill(shape, color: RGBColor, transparency: int = 0) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.fill.transparency = transparency
    shape.line.fill.background()


def _box(slide, x, y, w, h, color=PANEL, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, x, y, w, h)
    _set_fill(shape, color)
    return shape


def _text(
    slide,
    text: str,
    x,
    y,
    w,
    h,
    *,
    size: float = 18,
    color: RGBColor = WHITE,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin: float = 0.04,
):
    shape = slide.shapes.add_textbox(x, y, w, h)
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def _bullets(slide, items: list[str], x, y, w, h, *, size=18, color=WHITE):
    shape = slide.shapes.add_textbox(x, y, w, h)
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.05)
    frame.margin_right = Inches(0.04)
    frame.margin_top = Inches(0.03)
    frame.margin_bottom = Inches(0.02)
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = f"•  {item}"
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(10)
    return shape


def _base_slide(prs: Presentation, number: int, title: str, kicker: str):
    slide = prs.slides.add_slide(prs.slide_layouts[22])
    background = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
    )
    _set_fill(background, BG)
    _text(
        slide,
        kicker.upper(),
        Inches(0.62),
        Inches(0.28),
        Inches(6.2),
        Inches(0.25),
        size=9,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        title,
        Inches(0.62),
        Inches(0.58),
        Inches(11.9),
        Inches(0.58),
        size=28,
        bold=True,
    )
    _text(
        slide,
        "ЛЦТ 2026 · Фалькон Тех",
        Inches(0.62),
        Inches(7.13),
        Inches(4.0),
        Inches(0.2),
        size=8.5,
        color=MUTED,
    )
    _text(
        slide,
        f"{number:02d}",
        Inches(12.05),
        Inches(7.08),
        Inches(0.6),
        Inches(0.25),
        size=9,
        color=MUTED,
        align=PP_ALIGN.RIGHT,
    )
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.62), Inches(1.23), Inches(1.05), Inches(0.045)
    )
    _set_fill(accent, PINK)
    return slide


def _metric_card(
    slide, x, y, w, h, value: str, label: str, *, accent=PURPLE, note: str | None = None
):
    _box(slide, x, y, w, h)
    _text(
        slide,
        value,
        x + Inches(0.18),
        y + Inches(0.16),
        w - Inches(0.36),
        Inches(0.45),
        size=25,
        color=accent,
        bold=True,
    )
    _text(
        slide,
        label,
        x + Inches(0.18),
        y + Inches(0.68),
        w - Inches(0.36),
        Inches(0.42),
        size=12,
        color=WHITE,
        bold=True,
    )
    if note:
        _text(
            slide,
            note,
            x + Inches(0.18),
            y + h - Inches(0.42),
            w - Inches(0.36),
            Inches(0.28),
            size=8.5,
            color=MUTED,
        )


def _step(slide, x, y, w, h, index: str, title: str, body: str, *, accent=PURPLE):
    _box(slide, x, y, w, h)
    _text(
        slide,
        index,
        x + Inches(0.16),
        y + Inches(0.14),
        Inches(0.38),
        Inches(0.32),
        size=12,
        color=accent,
        bold=True,
    )
    _text(
        slide,
        title,
        x + Inches(0.58),
        y + Inches(0.12),
        w - Inches(0.74),
        Inches(0.35),
        size=15,
        bold=True,
    )
    _text(
        slide,
        body,
        x + Inches(0.18),
        y + Inches(0.58),
        w - Inches(0.36),
        h - Inches(0.72),
        size=10.5,
        color=MUTED,
    )


def _add_picture_fit(slide, image_path: Path, x, y, w, h):
    with Image.open(image_path) as image:
        image_ratio = image.width / image.height
    box_ratio = w / h
    picture = slide.shapes.add_picture(str(image_path), x, y, width=w, height=h)
    if image_ratio > box_ratio:
        crop = (1.0 - box_ratio / image_ratio) / 2.0
        picture.crop_left = crop
        picture.crop_right = crop
    else:
        crop = (1.0 - image_ratio / box_ratio) / 2.0
        picture.crop_top = crop
        picture.crop_bottom = crop
    return picture


def build(template: Path, output: Path) -> None:
    validation = _read_json(ROOT / "outputs/score_optimized_speed/validation.json")
    benchmark = _read_json(ROOT / "outputs/score_optimized_speed/benchmark.json")
    refusal = _read_json(
        ROOT / "outputs/score_optimized_speed/refusal_calibration.json"
    )
    errors = _read_json(ROOT / "outputs/score_optimized_speed/error_analysis.json")
    manifest = _read_json(
        ROOT / "outputs/score_optimized_speed/inference_manifest.json"
    )

    metrics = validation["confirmation"]
    open_metrics = refusal["confirmation"]
    previous_metrics = refusal["previous_release_confirmation"]
    aggregate = errors["aggregate"]
    speed = benchmark["best_throughput"]

    prs = Presentation(str(template))
    _remove_template_slides(prs)

    # 1. Title.
    slide = prs.slides.add_slide(prs.slide_layouts[22])
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
    )
    _set_fill(bg, BG)
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.20), prs.slide_height
    )
    _set_fill(accent, PINK)
    _text(
        slide,
        "ЛЦТ 2026 · ФАЛЬКОН ТЕХ",
        Inches(0.72),
        Inches(0.62),
        Inches(5.5),
        Inches(0.35),
        size=11,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "Vehicle ReID",
        Inches(0.72),
        Inches(1.52),
        Inches(7.8),
        Inches(0.75),
        size=44,
        bold=True,
    )
    _text(
        slide,
        "Быстрый offline-поиск автомобиля\nв статической галерее",
        Inches(0.76),
        Inches(2.40),
        Inches(7.3),
        Inches(1.10),
        size=24,
        color=MUTED,
    )
    _metric_card(
        slide,
        Inches(8.35),
        Inches(1.45),
        Inches(2.0),
        Inches(1.55),
        f"{metrics['mAP@10']:.3f}",
        "mAP@10",
        accent=PINK,
        note="5 locked seeds",
    )
    _metric_card(
        slide,
        Inches(10.55),
        Inches(1.45),
        Inches(2.0),
        Inches(1.55),
        f"{speed['images_per_second']:.1f}",
        "FPS локально",
        accent=GREEN,
        note="batch 32",
    )
    _box(slide, Inches(0.72), Inches(5.20), Inches(11.85), Inches(1.02), PANEL)
    _text(
        slide,
        "Финальный профиль: speed",
        Inches(1.02),
        Inches(5.43),
        Inches(4.2),
        Inches(0.32),
        size=17,
        bold=True,
    )
    _text(
        slide,
        f"Команда: {TEAM_NAME}    ·    {TEAM_MEMBERS}",
        Inches(4.55),
        Inches(5.46),
        Inches(7.6),
        Inches(0.35),
        size=11.5,
        color=WHITE,
        align=PP_ALIGN.RIGHT,
    )
    _text(
        slide,
        "01",
        Inches(12.0),
        Inches(7.08),
        Inches(0.6),
        Inches(0.25),
        size=9,
        color=MUTED,
        align=PP_ALIGN.RIGHT,
    )

    # 2. Product flow.
    slide = _base_slide(prs, 2, "Что получает пользователь", "сценарий")
    x0, y0 = Inches(0.68), Inches(1.62)
    step_w, gap = Inches(2.75), Inches(0.37)
    steps = [
        ("01", "Query", "Один crop автомобиля и BBox из входного CSV."),
        ("02", "Независимый embedding", "Три backbone, один проход на изображение."),
        ("03", "Top-10", "Ранжирование по static gallery без других query."),
        ("04", "Ответ или отказ", "Top-1 confidence и fail-closed candidates.csv."),
    ]
    for index, (num, title, body) in enumerate(steps):
        x = x0 + index * (step_w + gap)
        _step(
            slide,
            x,
            y0,
            step_w,
            Inches(2.15),
            num,
            title,
            body,
            accent=PINK if index == 3 else PURPLE,
        )
        if index < 3:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.CHEVRON,
                x + step_w + Inches(0.08),
                y0 + Inches(0.84),
                Inches(0.22),
                Inches(0.42),
            )
            _set_fill(arrow, PURPLE)
    _box(slide, Inches(0.68), Inches(4.30), Inches(12.0), Inches(1.55), PANEL_2)
    _text(
        slide,
        "Ключевая гарантия",
        Inches(0.98),
        Inches(4.58),
        Inches(2.6),
        Inches(0.34),
        size=15,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "Результат для query не зависит от наличия, порядка или признаков других query.",
        Inches(3.35),
        Inches(4.52),
        Inches(8.8),
        Inches(0.52),
        size=20,
        bold=True,
    )
    _text(
        slide,
        "Это одновременно соответствует правилам и делает batch-инференс воспроизводимым.",
        Inches(3.36),
        Inches(5.13),
        Inches(8.7),
        Inches(0.34),
        size=11,
        color=MUTED,
    )

    # 3. Architecture.
    slide = _base_slide(prs, 3, "Архитектура speed-профиля", "решение")
    _step(
        slide,
        Inches(0.68),
        Inches(1.58),
        Inches(3.35),
        Inches(1.48),
        "A",
        "OSNet",
        "Identity + metric + part descriptors",
        accent=PINK,
    )
    _step(
        slide,
        Inches(0.68),
        Inches(3.27),
        Inches(3.35),
        Inches(1.48),
        "B",
        "DINOv2 ViT-B/14",
        "CLS + regional + patch tokens · 224 px",
        accent=PURPLE,
    )
    _step(
        slide,
        Inches(0.68),
        Inches(4.96),
        Inches(3.35),
        Inches(1.48),
        "C",
        "ConvNeXt-Tiny",
        "Global + local descriptors",
        accent=GREEN,
    )
    stages = [
        ("1", "Cosine retrieval", "общий candidate pool"),
        ("2", "Top-50 / top-25", "малые reranker-модели"),
        ("3", "Gallery graph", "DINO patch + family GNN"),
        ("4", "LambdaRank", "финальный top-10"),
        ("5", "Open-set", "confidence → ответ/отказ"),
    ]
    for index, (num, title, body) in enumerate(stages):
        y = Inches(1.58 + index * 1.01)
        _step(
            slide,
            Inches(4.55),
            y,
            Inches(7.9),
            Inches(0.78),
            num,
            title,
            body,
            accent=PINK if index == 4 else PURPLE,
        )
    _text(
        slide,
        "Все связи строятся только внутри static gallery.",
        Inches(4.70),
        Inches(6.62),
        Inches(7.4),
        Inches(0.30),
        size=11,
        color=MUTED,
        align=PP_ALIGN.RIGHT,
    )

    # 4. Compliance.
    slide = _base_slide(prs, 4, "Соответствие ограничениям хакатона", "compliance")
    cards = [
        ("QUERY", "Независимы", "Нет query-to-query агрегации", GREEN),
        ("CAMERA", "Не используется", "camera_id только в offline validation", GREEN),
        ("ПОРЯДОК CSV", "Не используется", "ID и BBox читаются явно", GREEN),
        ("GALLERY", "Только static", "Граф и индексы без test-query", GREEN),
        ("INTERNET", "Не требуется", "Docker полностью offline", GREEN),
        ("ВЕСА", "244.6 MB", "Ниже лимита 2 GB", GREEN),
    ]
    for i, (tag, value, note, color) in enumerate(cards):
        col, row = i % 3, i // 3
        x, y = Inches(0.68 + col * 4.05), Inches(1.62 + row * 2.18)
        _box(slide, x, y, Inches(3.75), Inches(1.78))
        _text(
            slide,
            tag,
            x + Inches(0.20),
            y + Inches(0.18),
            Inches(3.3),
            Inches(0.25),
            size=9,
            color=PINK,
            bold=True,
        )
        _text(
            slide,
            value,
            x + Inches(0.20),
            y + Inches(0.55),
            Inches(3.3),
            Inches(0.45),
            size=20,
            color=color,
            bold=True,
        )
        _text(
            slide,
            note,
            x + Inches(0.20),
            y + Inches(1.19),
            Inches(3.3),
            Inches(0.31),
            size=10,
            color=MUTED,
        )
    _text(
        slide,
        "Схема, хеши, top-1 consistency и лимит веса проверяются fail-closed verifier’ом.",
        Inches(0.75),
        Inches(6.20),
        Inches(11.7),
        Inches(0.34),
        size=13,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )

    # 5. Retrieval quality.
    slide = _base_slide(prs, 5, "Качество retrieval", "locked validation")
    _metric_card(
        slide,
        Inches(0.68),
        Inches(1.62),
        Inches(2.75),
        Inches(1.78),
        f"{metrics['mAP@10']:.4f}",
        "mAP@10",
        accent=PINK,
    )
    _metric_card(
        slide,
        Inches(3.62),
        Inches(1.62),
        Inches(2.75),
        Inches(1.78),
        f"{metrics['Rank-1']:.4f}",
        "Rank-1",
        accent=PURPLE,
    )
    _metric_card(
        slide,
        Inches(6.56),
        Inches(1.62),
        Inches(2.75),
        Inches(1.78),
        f"{metrics['Rank-5']:.4f}",
        "Rank-5",
        accent=PURPLE,
    )
    _metric_card(
        slide,
        Inches(9.50),
        Inches(1.62),
        Inches(2.75),
        Inches(1.78),
        f"{aggregate['rank10']:.4f}",
        "Rank-10",
        accent=GREEN,
    )
    _box(slide, Inches(0.68), Inches(3.80), Inches(12.0), Inches(2.10), PANEL)
    _text(
        slide,
        "Как измерено",
        Inches(0.98),
        Inches(4.08),
        Inches(2.3),
        Inches(0.35),
        size=15,
        color=PINK,
        bold=True,
    )
    _bullets(
        slide,
        [
            "5 фиксированных seed: 101 / 211 / 307 / 401 / 503",
            "1 134 оцениваемых query на seed · итог — среднее по пяти эпизодам",
            "Speed уступает quality около 0.005 mAP, но проходит локальную границу throughput",
        ],
        Inches(3.10),
        Inches(4.04),
        Inches(8.8),
        Inches(1.45),
        size=15,
    )
    _text(
        slide,
        "Метрики получены точным release-пайплайном внутри Docker.",
        Inches(3.12),
        Inches(5.52),
        Inches(8.6),
        Inches(0.28),
        size=9.5,
        color=MUTED,
    )

    # 6. Open set.
    slide = _base_slide(prs, 6, "Open-set: калибровка под 20% unknown", "отказ")
    _metric_card(
        slide,
        Inches(0.68),
        Inches(1.62),
        Inches(2.75),
        Inches(1.75),
        "20.13%",
        "unknown query",
        accent=PINK,
        note="62 из 308 identity",
    )
    _metric_card(
        slide,
        Inches(3.62),
        Inches(1.62),
        Inches(2.75),
        Inches(1.75),
        f"{open_metrics['f1']:.4f}",
        "F1",
        accent=PURPLE,
    )
    _metric_card(
        slide,
        Inches(6.56),
        Inches(1.62),
        Inches(2.75),
        Inches(1.75),
        f"{open_metrics['tnr']:.4f}",
        "TNR",
        accent=GREEN,
    )
    _metric_card(
        slide,
        Inches(9.50),
        Inches(1.62),
        Inches(2.75),
        Inches(1.75),
        f"{open_metrics['official_score']:.4f}",
        "0.7 F1 + 0.3 TNR",
        accent=PINK,
    )
    _box(slide, Inches(0.68), Inches(3.78), Inches(5.62), Inches(2.15), PANEL)
    _text(
        slide,
        "Порог",
        Inches(0.98),
        Inches(4.07),
        Inches(1.5),
        Inches(0.30),
        size=13,
        color=MUTED,
        bold=True,
    )
    _text(
        slide,
        f"{manifest['refusal']['probability_threshold']:.6f}",
        Inches(0.98),
        Inches(4.45),
        Inches(3.0),
        Inches(0.50),
        size=28,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "Лучший candidate F1 при TNR ≥ 0.94",
        Inches(0.98),
        Inches(5.15),
        Inches(4.7),
        Inches(0.35),
        size=11,
        color=WHITE,
    )
    _box(slide, Inches(6.57), Inches(3.78), Inches(6.11), Inches(2.15), PANEL_2)
    delta = open_metrics["official_score"] - previous_metrics["official_score"]
    _text(
        slide,
        "Контроль переобучения",
        Inches(6.88),
        Inches(4.07),
        Inches(4.8),
        Inches(0.33),
        size=15,
        color=GREEN,
        bold=True,
    )
    _text(
        slide,
        f"+{delta:.4f}",
        Inches(6.88),
        Inches(4.48),
        Inches(2.0),
        Inches(0.50),
        size=28,
        color=GREEN,
        bold=True,
    )
    _text(
        slide,
        "к прошлой policy на тех же confirmation seeds",
        Inches(8.75),
        Inches(4.55),
        Inches(3.2),
        Inches(0.58),
        size=11,
        color=WHITE,
    )
    _text(
        slide,
        "Tune и confirmation identity/seed разделены; один query на identity.",
        Inches(6.90),
        Inches(5.34),
        Inches(5.2),
        Inches(0.35),
        size=9.5,
        color=MUTED,
    )

    # 7. Performance.
    slide = _base_slide(prs, 7, "Производительность и поставка", "engineering")
    _metric_card(
        slide,
        Inches(0.68),
        Inches(1.62),
        Inches(3.55),
        Inches(1.75),
        f"{benchmark['latency']['median_ms']:.2f} ms",
        "batch-1 median",
        accent=YELLOW,
        note=benchmark["gpu"],
    )
    _metric_card(
        slide,
        Inches(4.45),
        Inches(1.62),
        Inches(3.55),
        Inches(1.75),
        f"{speed['images_per_second']:.2f} FPS",
        "best throughput",
        accent=GREEN,
        note=f"batch {speed['batch_size']}",
    )
    _metric_card(
        slide,
        Inches(8.22),
        Inches(1.62),
        Inches(3.55),
        Inches(1.75),
        f"{manifest['total_weight_bytes'] / 1_000_000:.1f} MB",
        "все release-веса",
        accent=PURPLE,
        note="allowlist + SHA-256",
    )
    _box(slide, Inches(0.68), Inches(3.78), Inches(12.0), Inches(2.20), PANEL)
    _bullets(
        slide,
        [
            "50 warmup + 300 синхронизированных batch-1 запусков; JPEG I/O, crop и preprocess включены",
            "Throughput измерен по ≥10 секунд для batch 1 / 8 / 16 / 32",
            "Docker: pinned digest, offline, network none, read-only root, dataset read-only",
        ],
        Inches(0.98),
        Inches(4.03),
        Inches(7.55),
        Inches(1.62),
        size=13.5,
    )
    _box(slide, Inches(8.88), Inches(4.10), Inches(3.25), Inches(1.28), PANEL_2)
    _text(
        slide,
        "ЛОКАЛЬНЫЙ СТЕНД",
        Inches(9.12),
        Inches(4.30),
        Inches(2.6),
        Inches(0.26),
        size=9,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "RTX 3070 Ti Laptop",
        Inches(9.12),
        Inches(4.69),
        Inches(2.7),
        Inches(0.38),
        size=15,
        color=WHITE,
        bold=True,
    )
    _text(
        slide,
        "Метрики привязаны к указанному GPU; benchmark воспроизводится внутри release-контейнера.",
        Inches(8.90),
        Inches(5.50),
        Inches(3.20),
        Inches(0.54),
        size=8.5,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )

    # 8. Error analysis.
    slide = _base_slide(prs, 8, "Где решение ошибается", "error analysis")
    _box(slide, Inches(0.68), Inches(1.58), Inches(7.15), Inches(4.95), PANEL)
    _add_picture_fit(
        slide,
        ROOT / "docs/error_analysis_montage.jpg",
        Inches(0.82),
        Inches(1.72),
        Inches(6.87),
        Inches(4.67),
    )
    _metric_card(
        slide,
        Inches(8.12),
        Inches(1.58),
        Inches(4.18),
        Inches(1.22),
        str(aggregate["rescued_at_ranks_2_to_5"]),
        "ошибок top-1 спасены в rank 2–5",
        accent=PURPLE,
    )
    _metric_card(
        slide,
        Inches(8.12),
        Inches(3.00),
        Inches(4.18),
        Inches(1.22),
        str(aggregate["rescued_at_ranks_6_to_10"]),
        "правильный ответ на rank 6–10",
        accent=YELLOW,
    )
    _metric_card(
        slide,
        Inches(8.12),
        Inches(4.42),
        Inches(4.18),
        Inches(1.22),
        str(aggregate["no_positive_in_top10"]),
        "нет positive в top-10",
        accent=PINK,
    )
    _text(
        slide,
        "Типовые причины: смена ракурса, occlusion/crop, малый тёмный объект, визуально близкие кузова/цвета, смена ливреи.",
        Inches(8.15),
        Inches(5.92),
        Inches(4.05),
        Inches(0.72),
        size=9.5,
        color=MUTED,
    )

    # 9. Scaling.
    slide = _base_slide(
        prs, 9, "Путь к галерее 1 000 000 изображений", "масштабирование"
    )
    scale_steps = [
        ("01", "Exact → ANN", "IVF-PQ / HNSW для candidate retrieval"),
        (
            "02",
            "Двухуровневые признаки",
            "компактный global для всех, local только top-K",
        ),
        ("03", "Rerank top-K", "текущие verifier и graph без полного N²"),
        (
            "04",
            "Инкрементальные обновления",
            "versioned index + атомарная смена snapshot",
        ),
    ]
    for i, (num, title, body) in enumerate(scale_steps):
        x = Inches(0.68 + i * 3.02)
        _step(
            slide,
            x,
            Inches(1.62),
            Inches(2.70),
            Inches(2.20),
            num,
            title,
            body,
            accent=PINK if i == 0 else PURPLE,
        )
    _box(slide, Inches(0.68), Inches(4.30), Inches(12.0), Inches(1.65), PANEL)
    _text(
        slide,
        "Память global-векторов",
        Inches(0.98),
        Inches(4.58),
        Inches(2.65),
        Inches(0.32),
        size=14,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "6.68 GiB",
        Inches(3.75),
        Inches(4.50),
        Inches(1.75),
        Inches(0.45),
        size=22,
        color=WHITE,
        bold=True,
    )
    _text(
        slide,
        "float32",
        Inches(3.78),
        Inches(5.03),
        Inches(1.55),
        Inches(0.24),
        size=9,
        color=MUTED,
    )
    _text(
        slide,
        "3.34 GiB",
        Inches(5.75),
        Inches(4.50),
        Inches(1.75),
        Inches(0.45),
        size=22,
        color=WHITE,
        bold=True,
    )
    _text(
        slide,
        "float16",
        Inches(5.78),
        Inches(5.03),
        Inches(1.55),
        Inches(0.24),
        size=9,
        color=MUTED,
    )
    _text(
        slide,
        "64–128 MB",
        Inches(7.75),
        Inches(4.50),
        Inches(2.10),
        Inches(0.45),
        size=22,
        color=GREEN,
        bold=True,
    )
    _text(
        slide,
        "PQ-коды",
        Inches(7.78),
        Inches(5.03),
        Inches(1.55),
        Inches(0.24),
        size=9,
        color=MUTED,
    )
    _text(
        slide,
        "Цель: ≥95% recall candidate pool перед неизменным точным rerank.",
        Inches(9.78),
        Inches(4.56),
        Inches(2.35),
        Inches(0.74),
        size=11,
        color=WHITE,
    )

    # 10. Desktop prototype.
    slide = _base_slide(prs, 10, "Рабочее desktop-приложение", "демо")
    _box(slide, Inches(0.68), Inches(1.58), Inches(8.05), Inches(4.95), PANEL)
    _box(slide, Inches(0.92), Inches(1.83), Inches(7.57), Inches(0.70), PANEL_2)
    _text(
        slide,
        "FT",
        Inches(1.10),
        Inches(2.02),
        Inches(0.42),
        Inches(0.27),
        size=13,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "Falcon ReID",
        Inches(1.58),
        Inches(1.98),
        Inches(2.1),
        Inches(0.30),
        size=16,
        bold=True,
    )
    _text(
        slide,
        "●  SPEED READY",
        Inches(6.56),
        Inches(2.00),
        Inches(1.55),
        Inches(0.25),
        size=8,
        color=GREEN,
        bold=True,
        align=PP_ALIGN.RIGHT,
    )
    _text(
        slide,
        "ЗАПУСК И ПРОВЕРКА",
        Inches(1.10),
        Inches(2.72),
        Inches(2.2),
        Inches(0.24),
        size=8,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "ПРОСМОТР РЕЗУЛЬТАТОВ",
        Inches(3.22),
        Inches(2.72),
        Inches(2.4),
        Inches(0.24),
        size=8,
        color=MUTED,
        bold=True,
    )
    _box(slide, Inches(1.08), Inches(3.12), Inches(4.72), Inches(1.23), PANEL_2)
    _text(
        slide,
        "Готов к offline-запуску",
        Inches(1.32),
        Inches(3.37),
        Inches(3.6),
        Inches(0.32),
        size=17,
        bold=True,
    )
    _text(
        slide,
        "Выбор данных · Docker inference · verifier",
        Inches(1.32),
        Inches(3.83),
        Inches(3.8),
        Inches(0.25),
        size=9,
        color=MUTED,
    )
    for index, (value, label, color) in enumerate(
        [
            ("0.8361", "mAP@10", PINK),
            ("105.48", "FPS", GREEN),
            ("244.6", "MB", PURPLE),
        ]
    ):
        x = Inches(1.08 + index * 1.62)
        _box(slide, x, Inches(4.62), Inches(1.42), Inches(1.14), PANEL_2)
        _text(
            slide,
            value,
            x + Inches(0.15),
            Inches(4.82),
            Inches(1.10),
            Inches(0.31),
            size=16,
            color=color,
            bold=True,
        )
        _text(
            slide,
            label,
            x + Inches(0.15),
            Inches(5.25),
            Inches(1.05),
            Inches(0.20),
            size=8,
            color=MUTED,
            bold=True,
        )
    _box(slide, Inches(6.04), Inches(3.12), Inches(2.15), Inches(2.64), PANEL_2)
    _text(
        slide,
        "TOP-10",
        Inches(6.27),
        Inches(3.36),
        Inches(1.5),
        Inches(0.25),
        size=10,
        color=PINK,
        bold=True,
    )
    for index in range(6):
        x = Inches(6.28 + (index % 3) * 0.58)
        y = Inches(3.87 + (index // 3) * 0.72)
        _box(
            slide, x, y, Inches(0.45), Inches(0.52), RGBColor(48, 54, 68), radius=False
        )
        _text(
            slide,
            f"#{index + 1}",
            x + Inches(0.04),
            y + Inches(0.15),
            Inches(0.36),
            Inches(0.16),
            size=6.5,
            color=PINK if index == 0 else MUTED,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
    _text(
        slide,
        "Двойной щелчок по START_GUI.cmd",
        Inches(1.09),
        Inches(6.02),
        Inches(6.9),
        Inches(0.28),
        size=10,
        color=GREEN,
        bold=True,
    )

    _box(slide, Inches(9.02), Inches(1.58), Inches(3.66), Inches(4.95), PANEL_2)
    _text(
        slide,
        "БЕЗ ТЕРМИНАЛА",
        Inches(9.36),
        Inches(1.96),
        Inches(2.8),
        Inches(0.28),
        size=9,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "Один интерфейс\nдля всей проверки",
        Inches(9.36),
        Inches(2.40),
        Inches(2.85),
        Inches(0.82),
        size=22,
        bold=True,
    )
    _bullets(
        slide,
        [
            "Безопасный новый прогон",
            "Проверка release-bundle",
            "Query + top-10 + confidence",
            "975 ответов / 135 отказов",
        ],
        Inches(9.36),
        Inches(3.55),
        Inches(2.78),
        Inches(1.88),
        size=11.5,
    )
    _text(
        slide,
        "Реальные submission.csv и candidates.csv",
        Inches(9.38),
        Inches(5.75),
        Inches(2.72),
        Inches(0.48),
        size=9,
        color=MUTED,
    )

    # 11. Ready for delivery.
    slide = _base_slide(prs, 11, "Решение готово к проверке", "финальная поставка")
    _box(slide, Inches(0.68), Inches(1.58), Inches(7.38), Inches(4.95), PANEL)
    _text(
        slide,
        "В составе поставки",
        Inches(1.02),
        Inches(1.95),
        Inches(3.1),
        Inches(0.35),
        size=18,
        color=GREEN,
        bold=True,
    )
    _bullets(
        slide,
        [
            "Финальный speed runtime и воспроизводимый Docker",
            "Locked retrieval + 20%-open-set validation",
            "Fail-closed verifier, manifest и SHA-256",
            "Error analysis и план масштабирования до 1 млн",
            "Desktop-приложение для запуска и просмотра",
        ],
        Inches(1.02),
        Inches(2.52),
        Inches(6.45),
        Inches(2.75),
        size=14.5,
    )
    _box(slide, Inches(8.38), Inches(1.58), Inches(4.30), Inches(4.95), PANEL_2)
    _text(
        slide,
        "Быстрый старт",
        Inches(8.74),
        Inches(1.95),
        Inches(3.4),
        Inches(0.35),
        size=17,
        color=PINK,
        bold=True,
    )
    _text(
        slide,
        "START_GUI.cmd",
        Inches(8.74),
        Inches(2.52),
        Inches(3.35),
        Inches(0.42),
        size=21,
        color=WHITE,
        bold=True,
    )
    _text(
        slide,
        "Запуск · проверка · просмотр top-10",
        Inches(8.75),
        Inches(3.05),
        Inches(3.25),
        Inches(0.35),
        size=10,
        color=MUTED,
    )
    _text(
        slide,
        REPOSITORY_URL,
        Inches(8.75),
        Inches(3.76),
        Inches(3.25),
        Inches(0.52),
        size=11,
        color=GREEN,
        bold=True,
    )
    _text(
        slide,
        TEAM_NAME,
        Inches(8.75),
        Inches(4.55),
        Inches(3.25),
        Inches(0.30),
        size=14,
        color=WHITE,
        bold=True,
    )
    _text(
        slide,
        TEAM_CONTACT,
        Inches(8.75),
        Inches(5.02),
        Inches(3.25),
        Inches(0.55),
        size=9.5,
        color=MUTED,
    )
    _text(
        slide,
        "Спасибо!",
        Inches(0.70),
        Inches(6.72),
        Inches(6.0),
        Inches(0.38),
        size=20,
        bold=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final hackathon deck from the official template."
    )
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "presentation/Falcon_Tech_Vehicle_ReID.pptx",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    template_path = _find_template(args.template)
    build(template_path, args.output.resolve())
    print(args.output.resolve())
