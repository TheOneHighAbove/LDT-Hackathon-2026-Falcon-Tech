"""Desktop control panel for the frozen Falcon ReID speed release."""

from __future__ import annotations

import ctypes
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    BooleanVar,
    StringVar,
    Tk,
    X,
    filedialog,
    messagebox,
    ttk,
)

from PIL import Image, ImageTk

from prototype.server import ExplorerData

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "score_optimized_speed"
REQUIRED_DATASET_FILES = ("test_query.csv", "test_gallery.csv", "images")
REQUIRED_RESULT_FILES = (
    "submission.csv",
    "candidates.csv",
    "embeddings.npy",
    "gallery_index.npz",
    "inference_manifest.json",
)

BG = "#0b0e14"
PANEL = "#131822"
PANEL_ALT = "#1b2230"
BORDER = "#2a3446"
TEXT = "#f6f7fb"
MUTED = "#98a4b7"
PINK = "#ff2d6f"
PURPLE = "#8b7cff"
GREEN = "#35d39a"
YELLOW = "#f9c74f"
RED = "#ff6b7a"
BLUE = "#58a6ff"


def default_dataset_directory() -> Path:
    """Return the explicit external dataset path or the repository placeholder."""

    configured = os.environ.get("DATASET_DIR", "").strip()
    return Path(configured).expanduser() if configured else ROOT / "dataset"


def default_output_directory() -> Path:
    """Return the explicit result path or the final speed-profile directory."""

    configured = os.environ.get("OUTPUT_DIR", "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else ROOT / "outputs" / PROFILE
    )


def validate_dataset_directory(path: Path) -> list[str]:
    """Return human-readable problems with a selected dataset directory."""
    problems: list[str] = []
    if not path.is_dir():
        return [f"Каталог датасета не найден: {path}"]
    for name in REQUIRED_DATASET_FILES:
        item = path / name
        if name == "images":
            if not item.is_dir():
                problems.append(f"Нет каталога {item}")
        elif not item.is_file():
            problems.append(f"Нет файла {item}")
    return problems


def validate_result_directory(path: Path) -> list[str]:
    """Return missing files for a release result directory."""
    if not path.is_dir():
        return [f"Каталог результатов не найден: {path}"]
    return [
        f"Нет файла {path / name}"
        for name in REQUIRED_RESULT_FILES
        if not (path / name).is_file()
    ]


def safe_run_directory(selected: Path, now: datetime | None = None) -> Path:
    """Never overwrite an existing release bundle; choose a timestamped sibling."""
    selected = selected.resolve()
    if not selected.exists() or not any(selected.iterdir()):
        return selected
    stamp = (now or datetime.now(timezone.utc).astimezone()).strftime("%Y%m%d_%H%M%S")
    candidate = selected.parent / "runs" / f"run_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = selected.parent / "runs" / f"run_{stamp}_{suffix}"
        suffix += 1
    return candidate


def docker_commands(rebuild: bool) -> list[list[str]]:
    """Build the exact commands used by the GUI without shell interpolation."""
    commands: list[list[str]] = []
    if rebuild:
        commands.append(["docker", "compose", "build", "reid-api"])
    commands.append(
        [
            "docker",
            "compose",
            "run",
            "--rm",
            "--no-deps",
            "reid-api",
            "python",
            "-m",
            "scripts.infer_score_optimized",
            "--release-config",
            "/app/configs/score_optimized_speed.json",
            "--weights-dir",
            "/app/weights",
            "--images-dir",
            "/app/dataset/images",
            "--query-csv",
            "/app/dataset/test_query.csv",
            "--gallery-csv",
            "/app/dataset/test_gallery.csv",
            "--output-dir",
            "/app/outputs",
        ]
    )
    return commands


def docker_daemon_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _python_cli() -> str:
    candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.is_file():
        return str(candidate)
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        console = executable.with_name("python.exe")
        if console.is_file():
            return str(console)
    return str(executable)


def _settings_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    return base / "FalconReID" / "desktop.json"


def _open_path(path: Path) -> None:
    path = path.resolve()
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _enable_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def _normalize_tk_scaling(root: Tk) -> None:
    """Keep the dense dashboard usable on high-DPI Windows displays."""
    current = float(root.tk.call("tk", "scaling"))
    if current > 1.65:
        root.tk.call("tk", "scaling", 1.65)


def _blend(color: str, target: str = "#ffffff", amount: float = 0.12) -> str:
    """Return a small RGB blend used for native-looking button hover states."""
    source_rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    target_rgb = tuple(int(target[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(source + (destination - source) * amount)
        for source, destination in zip(source_rgb, target_rgb, strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def _create_app_icon(root: Tk) -> tk.PhotoImage:
    """Build a small code-native icon so the release needs no extra asset file."""
    icon = tk.PhotoImage(master=root, width=32, height=32)
    icon.put(BG, to=(0, 0, 32, 32))
    icon.put(PINK, to=(2, 2, 30, 30))
    icon.put("#ffffff", to=(7, 7, 11, 25))
    icon.put("#ffffff", to=(11, 7, 20, 11))
    icon.put("#ffffff", to=(11, 14, 18, 18))
    icon.put("#ffffff", to=(19, 7, 27, 11))
    icon.put("#ffffff", to=(21, 11, 25, 25))
    return icon


class FalconDesktopApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Falcon ReID · Панель управления")
        self.root.geometry("1480x920")
        self.root.minsize(1120, 700)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.app_icon = _create_app_icon(root)
        self.root.iconphoto(True, self.app_icon)

        self.events: queue.Queue[tuple] = queue.Queue()
        self.busy = False
        self.explorer: ExplorerData | None = None
        self.visible_queries: list[str] = []
        self.photo_refs: list[ImageTk.PhotoImage] = []

        settings = self._load_settings()
        self.dataset_var = StringVar(
            value=settings.get("dataset", str(default_dataset_directory()))
        )
        self.output_var = StringVar(
            value=settings.get("output", str(default_output_directory()))
        )
        self.rebuild_var = BooleanVar(value=False)
        self.query_var = StringVar()
        self.filter_var = StringVar(value="Все запросы")
        self.status_var = StringVar(value="Готово")
        self.position_var = StringVar(value="0 / 0")
        self.decision_var = StringVar(value="Результаты ещё не загружены")
        self.decision_detail_var = StringVar(value="Выберите готовый bundle")
        self.confidence_summary_var = StringVar(value="confidence —")
        self.acceptance_summary_var = StringVar(value="bundle ещё не загружен")
        self._acceptance_counts = (0, 0)
        self._confidence_state: tuple[float | None, float, bool] = (
            None,
            0.0,
            False,
        )

        self._configure_style()
        self._build_header()
        self._build_tabs()
        self._build_footer()
        self.dataset_var.trace_add("write", self._schedule_path_refresh)
        self.output_var.trace_add("write", self._schedule_path_refresh)
        self.root.bind("<Left>", lambda _event: self._move_query(-1))
        self.root.bind("<Right>", lambda _event: self._move_query(1))
        self.root.bind("<F5>", lambda _event: self.load_results())
        self.root.after(100, self._poll_events)
        self.root.after_idle(self._refresh_path_states)
        self.root.after(180, lambda: self.load_results(silent=True))

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Alt.TFrame", background=PANEL_ALT)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure(
            "Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10)
        )
        style.configure(
            "Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9)
        )
        style.configure(
            "PanelMuted.TLabel",
            background=PANEL,
            foreground=MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Title.TLabel",
            background=BG,
            foreground=TEXT,
            font=("Segoe UI Semibold", 23),
        )
        style.configure(
            "Eyebrow.TLabel",
            background=BG,
            foreground=PINK,
            font=("Segoe UI Semibold", 8),
        )
        style.configure(
            "Section.TLabel",
            background=PANEL,
            foreground=TEXT,
            font=("Segoe UI Semibold", 14),
        )
        style.configure(
            "Metric.TLabel",
            background=PANEL_ALT,
            foreground=TEXT,
            font=("Segoe UI Semibold", 20),
        )
        style.configure(
            "MetricName.TLabel",
            background=PANEL_ALT,
            foreground=MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "TEntry",
            fieldbackground=PANEL_ALT,
            foreground=TEXT,
            insertcolor=TEXT,
            bordercolor="#343a46",
        )
        style.configure(
            "TCombobox",
            fieldbackground=PANEL_ALT,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor="#343a46",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL_ALT)],
            foreground=[("readonly", TEXT)],
        )
        style.configure(
            "TCheckbutton", background=PANEL, foreground=TEXT, font=("Segoe UI", 10)
        )
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=PANEL,
            foreground=MUTED,
            padding=(24, 11),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", PANEL_ALT)],
            foreground=[("selected", TEXT)],
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=PANEL_ALT,
            background=PINK,
            bordercolor=PANEL_ALT,
            lightcolor=PINK,
            darkcolor=PINK,
        )

    def _build_header(self) -> None:
        tk.Frame(self.root, bg=PINK, height=3).pack(fill=X)
        header = ttk.Frame(self.root, padding=(28, 17, 28, 9))
        header.pack(fill=X)
        actions = ttk.Frame(header)
        actions.pack(side=RIGHT, pady=(6, 0))
        tk.Label(
            actions,
            text="●  SPEED READY",
            bg="#12372d",
            fg=GREEN,
            padx=12,
            pady=8,
            font=("Segoe UI Semibold", 8),
        ).pack(side=LEFT, padx=(0, 8))
        self._button(
            actions, "README", lambda: _open_path(ROOT / "README.md"), PANEL_ALT
        ).pack(side=LEFT, padx=5)
        self._button(
            actions,
            "Презентация",
            lambda: _open_path(ROOT / "presentation" / "Falcon_Tech_Vehicle_ReID.pptx"),
            PURPLE,
        ).pack(side=LEFT, padx=5)
        left = ttk.Frame(header)
        left.pack(side=LEFT, fill=X, expand=True)
        logo = tk.Label(
            left,
            text="FT",
            bg=PINK,
            fg="white",
            width=3,
            height=1,
            font=("Segoe UI Black", 15),
        )
        logo.pack(side=LEFT, padx=(0, 14), pady=(1, 0))
        brand = ttk.Frame(left)
        brand.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(brand, text="VEHICLE SEARCH CONSOLE", style="Eyebrow.TLabel").pack(
            anchor="w"
        )
        ttk.Label(brand, text="Falcon ReID", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            brand,
            text="Финальный speed-профиль · запуск, проверка и просмотр результатов",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(1, 0))

    def _build_tabs(self) -> None:
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill=BOTH, expand=True, padx=24, pady=(8, 12))
        self.run_tab = ttk.Frame(self.tabs, padding=4)
        self.results_tab = ttk.Frame(self.tabs, padding=4)
        self.log_tab = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(self.run_tab, text="  Запуск и проверка  ")
        self.tabs.add(self.results_tab, text="  Просмотр результатов  ")
        self.tabs.add(self.log_tab, text="  Журнал  ")
        self._build_run_tab()
        self._build_results_tab()
        self._build_log_tab()

    def _panel(self, parent, padding=(22, 18)):
        return ttk.Frame(parent, style="Panel.TFrame", padding=padding)

    def _button(self, parent, text: str, command, color=PINK, *, width=None):
        hover = _blend(color)
        border = color if color != PANEL_ALT else BORDER
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=TEXT,
            activebackground=color,
            activeforeground=TEXT,
            disabledforeground="#777c86",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=16,
            pady=9,
            cursor="hand2",
            font=("Segoe UI Semibold", 10),
            width=width,
        )
        button.bind(
            "<Enter>",
            lambda _event: (
                button.configure(bg=hover)
                if str(button.cget("state")) != "disabled"
                else None
            ),
        )
        button.bind("<Leave>", lambda _event: button.configure(bg=color))
        return button

    def _build_run_tab(self) -> None:
        hero = tk.Frame(
            self.run_tab,
            bg="#171526",
            highlightbackground="#332953",
            highlightthickness=1,
            padx=20,
            pady=14,
        )
        hero.pack(fill=X, pady=(6, 12))
        hero_right = tk.Frame(hero, bg="#171526")
        hero_right.pack(side=RIGHT)
        for text, color in (
            ("0.8361  mAP@10", PINK),
            ("105.48  FPS", GREEN),
            ("244.6  MB", PURPLE),
        ):
            tk.Label(
                hero_right,
                text=text,
                bg=PANEL_ALT,
                fg=color,
                padx=12,
                pady=8,
                font=("Segoe UI Semibold", 9),
            ).pack(side=LEFT, padx=(8, 0))
        hero_copy = tk.Frame(hero, bg="#171526")
        hero_copy.pack(side=LEFT, fill=X, expand=True)
        tk.Label(
            hero_copy,
            text="РЕЛИЗНЫЙ ПРОФИЛЬ",
            bg="#171526",
            fg=PINK,
            font=("Segoe UI Semibold", 8),
        ).pack(anchor="w")
        tk.Label(
            hero_copy,
            text="Готов к воспроизводимому offline-запуску",
            bg="#171526",
            fg=TEXT,
            font=("Segoe UI Semibold", 16),
            justify="left",
        ).pack(anchor="w", pady=(3, 1))
        tk.Label(
            hero_copy,
            text="Docker запускает тот же speed runtime, которым получены итоговые метрики.",
            bg="#171526",
            fg=MUTED,
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w")

        setup = ttk.Frame(self.run_tab)
        setup.pack(fill=X, pady=(0, 12))
        paths = self._panel(setup, padding=(18, 14))
        paths.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 6))
        ttk.Label(paths, text="Пути", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 13)
        )
        path_states = tk.Frame(paths, bg=PANEL)
        path_states.grid(row=0, column=1, columnspan=2, sticky="e", pady=(0, 13))
        self.dataset_state_badge = tk.Label(
            path_states,
            text="ДАННЫЕ …",
            bg=PANEL_ALT,
            fg=MUTED,
            padx=9,
            pady=4,
            font=("Segoe UI Semibold", 7),
        )
        self.dataset_state_badge.pack(side=LEFT, padx=(0, 6))
        self.output_state_badge = tk.Label(
            path_states,
            text="BUNDLE …",
            bg=PANEL_ALT,
            fg=MUTED,
            padx=9,
            pady=4,
            font=("Segoe UI Semibold", 7),
        )
        self.output_state_badge.pack(side=LEFT)
        paths.columnconfigure(1, weight=1)
        ttk.Label(paths, text="Датасет", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 14), pady=7
        )
        ttk.Entry(paths, textvariable=self.dataset_var).grid(
            row=1, column=1, sticky="ew", pady=7
        )
        self._button(paths, "Выбрать…", self._browse_dataset, PANEL_ALT).grid(
            row=1, column=2, padx=(12, 0), pady=7
        )
        ttk.Label(paths, text="Результаты", style="Panel.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 14), pady=7
        )
        ttk.Entry(paths, textvariable=self.output_var).grid(
            row=2, column=1, sticky="ew", pady=7
        )
        self._button(paths, "Выбрать…", self._browse_output, PANEL_ALT).grid(
            row=2, column=2, padx=(12, 0), pady=7
        )
        ttk.Label(
            paths,
            text="Существующий bundle никогда не перезаписывается: новый прогон автоматически попадёт в outputs/runs/.",
            style="PanelMuted.TLabel",
            wraplength=520,
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(3, 0))

        actions = self._panel(setup, padding=(18, 14))
        actions.pack(side=RIGHT, fill=BOTH, expand=True, padx=(6, 0))
        top = ttk.Frame(actions, style="Panel.TFrame")
        top.pack(fill=X)
        ttk.Label(top, text="Действия", style="Section.TLabel").pack(side=LEFT)
        ttk.Checkbutton(
            top, text="Пересобрать Docker-образ", variable=self.rebuild_var
        ).pack(side=RIGHT)
        row = ttk.Frame(actions, style="Panel.TFrame")
        row.pack(fill=X, pady=(16, 10))
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        self.run_button = self._button(
            row, "Запустить inference", self.run_inference, PINK
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        self.verify_button = self._button(
            row, "Проверить bundle", self.verify_results, PURPLE
        )
        self.verify_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=(0, 6))
        self.load_button = self._button(
            row, "Показать результаты", self.load_results, GREEN
        )
        self.load_button.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(6, 0))
        self._button(row, "Открыть папку", self.open_output, PANEL_ALT).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0)
        )
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(fill=X, pady=(4, 0))

        metrics = ttk.Frame(self.run_tab)
        metrics.pack(fill=X, pady=(0, 12))
        for column in range(4):
            metrics.columnconfigure(column, weight=1)
        self.metric_values: dict[str, ttk.Label] = {}
        for column, (key, title, hint, accent) in enumerate(
            (
                ("map", "mAP@10", "locked validation", PINK),
                ("queries", "QUERY", "в текущем bundle", BLUE),
                ("accepted", "ПРИНЯТО / ОТКАЗ", "open-set policy", GREEN),
                ("weights", "ВЕСА", "проверены по SHA-256", PURPLE),
            )
        ):
            card = tk.Frame(
                metrics,
                bg=PANEL_ALT,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            card.grid(
                row=0,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 6, 0 if column == 3 else 6),
            )
            tk.Frame(card, bg=accent, height=3).pack(fill=X)
            body = tk.Frame(card, bg=PANEL_ALT, padx=17, pady=11)
            body.pack(fill=X)
            value = tk.Label(
                body,
                text="—",
                bg=PANEL_ALT,
                fg=accent,
                font=("Segoe UI Semibold", 20),
            )
            value.pack(anchor="w")
            tk.Label(
                body,
                text=title,
                bg=PANEL_ALT,
                fg=TEXT,
                font=("Segoe UI Semibold", 8),
            ).pack(anchor="w", pady=(2, 0))
            tk.Label(
                body,
                text=hint,
                bg=PANEL_ALT,
                fg=MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(1, 0))
            self.metric_values[key] = value

        delivery = tk.Frame(self.run_tab, bg=PANEL, padx=18, pady=10)
        delivery.pack(fill=X)
        acceptance = tk.Frame(delivery, bg=PANEL)
        acceptance.pack(side=RIGHT, padx=(24, 0))
        acceptance_heading = tk.Frame(acceptance, bg=PANEL)
        acceptance_heading.pack(fill=X, pady=(0, 5))
        tk.Label(
            acceptance_heading,
            text="OPEN-SET",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(side=LEFT)
        tk.Label(
            acceptance_heading,
            textvariable=self.acceptance_summary_var,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 8),
        ).pack(side=RIGHT, padx=(20, 0))
        self.acceptance_canvas = tk.Canvas(
            acceptance,
            width=280,
            height=8,
            bg="#252d3b",
            highlightthickness=0,
            bd=0,
        )
        self.acceptance_canvas.pack(fill=X)
        self.acceptance_canvas.bind(
            "<Configure>", lambda _event: self._draw_acceptance_bar()
        )
        tk.Frame(delivery, bg=BORDER, width=1).pack(side=RIGHT, fill="y", padx=(18, 0))
        trust = tk.Frame(delivery, bg=PANEL)
        trust.pack(side=LEFT, fill=X, expand=True)
        tk.Label(
            trust,
            text="КОНТУР ПОСТАВКИ",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(side=LEFT, padx=(0, 18))
        for text in (
            "✓  OFFLINE",
            "✓  STATIC GALLERY",
            "✓  QUERY-INDEPENDENT",
            "✓  FAIL-CLOSED VERIFY",
        ):
            tk.Label(
                trust,
                text=text,
                bg=PANEL,
                fg=GREEN,
                font=("Segoe UI Semibold", 8),
            ).pack(side=LEFT, padx=(0, 18))

    def _build_log_tab(self) -> None:
        log_panel = self._panel(self.log_tab, padding=(18, 14))
        log_panel.pack(fill=BOTH, expand=True, pady=(6, 0))
        title = ttk.Frame(log_panel, style="Panel.TFrame")
        title.pack(fill=X, pady=(0, 8))
        ttk.Label(title, text="Журнал выполнения", style="Section.TLabel").pack(
            side=LEFT
        )
        self._button(title, "Очистить", self._clear_log, PANEL_ALT).pack(side=RIGHT)
        self.log = tk.Text(
            log_panel,
            height=10,
            bg="#0d0f13",
            fg="#d8dce5",
            insertbackground=TEXT,
            selectbackground="#3d4250",
            relief="flat",
            font=("Cascadia Mono", 9),
            padx=12,
            pady=10,
            wrap="word",
        )
        self.log.pack(fill=BOTH, expand=True)
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", END)
        self.log.configure(state="disabled")

    def _build_results_tab(self) -> None:
        heading = ttk.Frame(self.results_tab)
        heading.pack(fill=X, pady=(7, 10))
        ttk.Label(
            heading,
            text="RESULT EXPLORER",
            style="Eyebrow.TLabel",
        ).pack(anchor="w")
        title_row = ttk.Frame(heading)
        title_row.pack(fill=X, pady=(2, 0))
        ttk.Label(
            title_row,
            text="Поиск автомобиля в статической галерее",
            style="Title.TLabel",
        ).pack(side=LEFT)
        tk.Label(
            title_row,
            text="реальные release-артефакты",
            bg="#12372d",
            fg=GREEN,
            padx=11,
            pady=6,
            font=("Segoe UI Semibold", 8),
        ).pack(side=RIGHT)

        controls = self._panel(self.results_tab, padding=(16, 12))
        controls.pack(fill=X, pady=(0, 12))
        ttk.Label(controls, text="Запрос", style="PanelMuted.TLabel").pack(
            side=LEFT, padx=(0, 9)
        )
        self.prev_button = self._button(
            controls, "‹", lambda: self._move_query(-1), PANEL_ALT, width=3
        )
        self.prev_button.pack(side=LEFT, padx=(0, 6))
        self.next_button = self._button(
            controls, "›", lambda: self._move_query(1), PANEL_ALT, width=3
        )
        self.next_button.pack(side=LEFT, padx=(0, 14))
        self.query_combo = ttk.Combobox(
            controls, textvariable=self.query_var, state="readonly", width=42
        )
        self.query_combo.pack(side=LEFT, padx=(0, 12))
        self.query_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._render_query()
        )
        self.filter_combo = ttk.Combobox(
            controls,
            textvariable=self.filter_var,
            values=("Все запросы", "Только принятые", "Только отказы"),
            state="readonly",
            width=20,
        )
        self.filter_combo.pack(side=LEFT)
        self.filter_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._apply_filter()
        )
        tk.Label(
            controls,
            textvariable=self.position_var,
            bg=PANEL,
            fg=MUTED,
            font=("Cascadia Mono", 9),
            padx=12,
        ).pack(side=LEFT)
        self._button(controls, "Обновить", self.load_results, GREEN).pack(side=RIGHT)

        content = ttk.Frame(self.results_tab)
        content.pack(fill=BOTH, expand=True)
        self.query_panel = tk.Frame(
            content,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=16,
            pady=16,
        )
        self.query_panel.pack(side=LEFT, fill="y", padx=(0, 12))
        self.query_panel.configure(width=324)
        self.query_panel.pack_propagate(False)
        tk.Label(
            self.query_panel,
            text="QUERY CROP",
            bg=PANEL,
            fg=PINK,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w")
        tk.Label(
            self.query_panel,
            text="Входной автомобиль",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 15),
        ).pack(anchor="w", pady=(3, 0))
        tk.Label(
            self.query_panel,
            text="BBox применяется до извлечения признаков",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(1, 0))
        image_frame = tk.Frame(
            self.query_panel,
            bg="#090c12",
            highlightbackground="#30394a",
            highlightthickness=1,
        )
        image_frame.pack(fill=X, pady=(14, 11))
        self.query_image = tk.Label(image_frame, bg="#090c12", bd=0)
        self.query_image.pack(fill=X, padx=5, pady=5)
        self.query_id_label = tk.Label(
            self.query_panel,
            text="—",
            bg=PANEL,
            fg=MUTED,
            font=("Cascadia Mono", 8),
            wraplength=282,
            justify="left",
        )
        self.query_id_label.pack(anchor="w")
        self.decision_card = tk.Frame(
            self.query_panel,
            bg=PANEL_ALT,
            padx=12,
            pady=8,
        )
        self.decision_card.pack(fill=X, pady=(12, 0))
        decision_heading = tk.Frame(self.decision_card, bg=PANEL_ALT)
        decision_heading.pack(fill=X)
        self.decision_label = tk.Label(
            decision_heading,
            textvariable=self.decision_var,
            bg=PANEL_ALT,
            fg=MUTED,
            font=("Segoe UI Semibold", 11),
            wraplength=270,
            justify="left",
        )
        self.decision_label.pack(side=LEFT)
        self.decision_tag = tk.Label(
            decision_heading,
            text="OPEN-SET",
            bg="#242c3a",
            fg=MUTED,
            padx=6,
            pady=2,
            font=("Segoe UI Semibold", 6),
        )
        self.decision_tag.pack(side=RIGHT)
        self.decision_detail_label = tk.Label(
            self.decision_card,
            textvariable=self.decision_detail_var,
            bg=PANEL_ALT,
            fg=MUTED,
            font=("Cascadia Mono", 7),
            wraplength=270,
            justify="left",
        )
        self.decision_detail_label.pack(anchor="w", pady=(3, 6))
        self.confidence_canvas = tk.Canvas(
            self.decision_card,
            height=7,
            bg="#2b3443",
            highlightthickness=0,
            bd=0,
        )
        self.confidence_canvas.pack(fill=X)
        self.confidence_canvas.bind(
            "<Configure>", lambda _event: self._draw_confidence_meter()
        )
        self.confidence_label = tk.Label(
            self.decision_card,
            textvariable=self.confidence_summary_var,
            bg=PANEL_ALT,
            fg=MUTED,
            font=("Segoe UI", 7),
        )
        self.confidence_label.pack(anchor="w", pady=(4, 0))

        gallery_panel = tk.Frame(
            content,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=16,
            pady=16,
        )
        gallery_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        gallery_title = tk.Frame(gallery_panel, bg=PANEL)
        gallery_title.pack(fill=X, pady=(0, 12))
        tk.Label(
            gallery_title,
            text="TOP-10 STATIC GALLERY",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 14),
        ).pack(side=LEFT)
        tk.Label(
            gallery_title,
            text="query-independent ranking",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).pack(side=RIGHT)
        self.gallery_grid = ttk.Frame(gallery_panel, style="Panel.TFrame")
        self.gallery_grid.pack(fill=BOTH, expand=True)
        for column in range(5):
            self.gallery_grid.columnconfigure(column, weight=1)
        for row in range(2):
            self.gallery_grid.rowconfigure(row, weight=1)

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.root, padding=(28, 0, 28, 14))
        footer.pack(fill=X)
        self.status_dot = ttk.Label(footer, text="●", foreground=GREEN)
        self.status_dot.pack(side=LEFT)
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").pack(
            side=LEFT, padx=(7, 0)
        )
        ttk.Label(
            footer,
            text="SPEED PROFILE  ·  OFFLINE  ·  STATIC GALLERY  ·  F5 ОБНОВИТЬ",
            style="Muted.TLabel",
        ).pack(side=RIGHT)

    def _schedule_path_refresh(self, *_args: str) -> None:
        self.root.after_idle(self._refresh_path_states)

    def _refresh_path_states(self) -> None:
        try:
            dataset = Path(self.dataset_var.get()).expanduser().resolve()
            output = Path(self.output_var.get()).expanduser().resolve()
        except (OSError, RuntimeError):
            self.dataset_state_badge.configure(
                text="ДАННЫЕ  ×", bg="#482124", fg="#ffadb2"
            )
            self.output_state_badge.configure(
                text="BUNDLE  ×", bg="#482124", fg="#ffadb2"
            )
            return

        if validate_dataset_directory(dataset):
            self.dataset_state_badge.configure(
                text="ДАННЫЕ  ×", bg="#482124", fg="#ffadb2"
            )
        else:
            self.dataset_state_badge.configure(
                text="ДАННЫЕ  ✓", bg="#153b2a", fg="#7ee2ad"
            )

        output_problems = validate_result_directory(output)
        if not output_problems:
            self.output_state_badge.configure(
                text="BUNDLE  ✓", bg="#153b2a", fg="#7ee2ad"
            )
        elif not output.exists() or (
            output.is_dir()
            and not any((output / name).exists() for name in REQUIRED_RESULT_FILES)
        ):
            self.output_state_badge.configure(
                text="НОВЫЙ BUNDLE", bg="#172b42", fg="#8bc4ff"
            )
        else:
            self.output_state_badge.configure(
                text="BUNDLE  !", bg="#49391b", fg="#ffe08a"
            )

    def _update_acceptance_bar(self, accepted: int, refused: int) -> None:
        self._acceptance_counts = (accepted, refused)
        total = accepted + refused
        if total:
            accepted_percent = accepted / total * 100
            self.acceptance_summary_var.set(
                f"{accepted} принято · {refused} отказов · {accepted_percent:.1f}%"
            )
        else:
            self.acceptance_summary_var.set("нет решений")
        self.root.after_idle(self._draw_acceptance_bar)

    def _draw_acceptance_bar(self) -> None:
        canvas = self.acceptance_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        accepted, refused = self._acceptance_counts
        total = accepted + refused
        if total <= 0:
            canvas.create_rectangle(0, 0, width, height, fill="#252d3b", width=0)
            return
        accepted_width = round(width * accepted / total)
        canvas.create_rectangle(0, 0, accepted_width, height, fill=GREEN, width=0)
        canvas.create_rectangle(accepted_width, 0, width, height, fill=RED, width=0)

    def _update_confidence_meter(
        self, confidence: float | None, threshold: float, accepted: bool
    ) -> None:
        self._confidence_state = (confidence, threshold, accepted)
        self.root.after_idle(self._draw_confidence_meter)

    def _draw_confidence_meter(self) -> None:
        canvas = self.confidence_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        confidence, threshold, accepted = self._confidence_state
        canvas.create_rectangle(0, 0, width, height, fill="#2b3443", width=0)
        if accepted and confidence is not None:
            confidence_width = round(width * min(max(confidence, 0.0), 1.0))
            canvas.create_rectangle(0, 0, confidence_width, height, fill=GREEN, width=0)
        marker = round(width * min(max(threshold, 0.0), 1.0))
        canvas.create_line(marker, 0, marker, height, fill="#ffffff", width=2)

    def _browse_dataset(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=self.dataset_var.get(), title="Выберите каталог датасета"
        )
        if selected:
            self.dataset_var.set(selected)
            self._save_settings()

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=self.output_var.get(), title="Выберите каталог результатов"
        )
        if selected:
            self.output_var.set(selected)
            self._save_settings()

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(END, text.rstrip() + "\n")
        self.log.see(END)
        self.log.configure(state="disabled")

    def _set_busy(self, value: bool, label: str = "Готово") -> None:
        self.busy = value
        state = "disabled" if value else "normal"
        for button in (self.run_button, self.verify_button, self.load_button):
            button.configure(state=state)
        if value:
            self.progress.start(12)
            self.status_dot.configure(foreground=YELLOW)
        else:
            self.progress.stop()
            self.status_dot.configure(foreground=GREEN)
        self.status_var.set(label)

    def _run_processes(
        self, commands: list[list[str]], env: dict[str, str], success_callback=None
    ) -> None:
        if self.busy:
            return
        self._set_busy(True, "Выполняется…")
        self._append_log("\n" + "═" * 72)
        self.tabs.select(self.log_tab)

        def worker() -> None:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                for command in commands:
                    self.events.put(("line", "> " + subprocess.list2cmdline(command)))
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        creationflags=flags,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        self.events.put(("line", line.rstrip()))
                    return_code = process.wait()
                    if return_code:
                        self.events.put(("done", return_code, None))
                        return
                self.events.put(("done", 0, success_callback))
            except Exception as exc:  # noqa: BLE001 - surface UI boundary errors.
                self.events.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "line":
                    self._append_log(event[1])
                elif event[0] == "done":
                    return_code, callback = event[1], event[2]
                    if return_code == 0:
                        self._append_log("✓ Операция завершена успешно")
                        self._set_busy(False, "Готово")
                        if callback:
                            callback()
                    else:
                        self._append_log(f"✗ Процесс завершился с кодом {return_code}")
                        self._set_busy(False, "Ошибка выполнения")
                        messagebox.showerror(
                            "Falcon ReID",
                            f"Процесс завершился с кодом {return_code}. Подробности — в журнале.",
                        )
                elif event[0] == "error":
                    self._append_log("✗ " + event[1])
                    self._set_busy(False, "Ошибка выполнения")
                    messagebox.showerror("Falcon ReID", event[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def run_inference(self) -> None:
        dataset = Path(self.dataset_var.get()).expanduser().resolve()
        problems = validate_dataset_directory(dataset)
        if problems:
            messagebox.showerror("Некорректный датасет", "\n".join(problems))
            return
        if not docker_daemon_ready():
            messagebox.showerror(
                "Docker недоступен",
                "Установите или запустите Docker Desktop, дождитесь готовности engine и повторите запуск.",
            )
            return
        selected = Path(self.output_var.get()).expanduser().resolve()
        target = safe_run_directory(selected)
        if target != selected:
            messagebox.showinfo(
                "Безопасный новый прогон",
                f"Текущие результаты не будут перезаписаны.\nНовый прогон сохранится в:\n{target}",
            )
        target.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["DATASET_DIR"] = str(dataset)
        env["OUTPUTS_DIR"] = str(target)

        def completed() -> None:
            self.output_var.set(str(target))
            self._save_settings()
            self.load_results()

        self._append_log(f"Датасет: {dataset}")
        self._append_log(f"Результаты: {target}")
        self._run_processes(docker_commands(self.rebuild_var.get()), env, completed)

    def verify_results(self) -> None:
        output = Path(self.output_var.get()).expanduser().resolve()
        problems = validate_result_directory(output)
        if problems:
            messagebox.showerror("Нет готовых результатов", "\n".join(problems))
            return
        command = [
            _python_cli(),
            "-m",
            "src.verify_score_optimized",
            "--output-dir",
            str(output),
        ]
        self._run_processes(
            [command], os.environ.copy(), lambda: self.load_results(silent=True)
        )

    def load_results(self, silent: bool = False) -> None:
        dataset = Path(self.dataset_var.get()).expanduser().resolve()
        output = Path(self.output_var.get()).expanduser().resolve()
        problems = [
            *validate_dataset_directory(dataset),
            *validate_result_directory(output),
        ]
        if problems:
            if not silent:
                messagebox.showerror(
                    "Не удалось открыть результаты", "\n".join(problems)
                )
            return
        try:
            self.explorer = ExplorerData.load(dataset, output)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not silent:
                messagebox.showerror("Некорректные артефакты", str(exc))
            self._append_log(f"✗ Не удалось загрузить результаты: {exc}")
            return
        self._save_settings()
        health = self.explorer.health()
        validation = self.explorer.manifest.get("validation", {})
        weight_bytes = int(self.explorer.manifest.get("total_weight_bytes", 0))
        self.metric_values["map"].configure(
            text=f"{float(validation.get('mAP@10', 0)):.4f}"
        )
        self.metric_values["queries"].configure(text=str(health["queries"]))
        self.metric_values["accepted"].configure(
            text=f"{health['accepted']} / {health['refused']}"
        )
        self.metric_values["weights"].configure(
            text=f"{weight_bytes / 1_000_000:.1f} MB"
        )
        self._update_acceptance_bar(int(health["accepted"]), int(health["refused"]))
        self._refresh_path_states()
        self._append_log(
            f"✓ Загружен {output} · {health['queries']} query · {health['accepted']} принято · {health['refused']} отказов"
        )
        self._apply_filter()
        if not silent:
            self.tabs.select(self.results_tab)

    def _apply_filter(self) -> None:
        if self.explorer is None:
            return
        mode = self.filter_var.get()
        if mode == "Только принятые":
            self.visible_queries = [
                query
                for query in self.explorer.query_ids
                if query in self.explorer.accepted
            ]
        elif mode == "Только отказы":
            self.visible_queries = [
                query
                for query in self.explorer.query_ids
                if query not in self.explorer.accepted
            ]
        else:
            self.visible_queries = list(self.explorer.query_ids)
        self.query_combo.configure(values=self.visible_queries)
        if self.visible_queries:
            current = self.query_var.get()
            self.query_var.set(
                current if current in self.visible_queries else self.visible_queries[0]
            )
            self._render_query()
        else:
            self.position_var.set("0 / 0")

    def _move_query(self, delta: int) -> None:
        if not self.visible_queries:
            return
        try:
            index = self.visible_queries.index(self.query_var.get())
        except ValueError:
            index = 0
        self.query_var.set(
            self.visible_queries[(index + delta) % len(self.visible_queries)]
        )
        self._render_query()

    def _photo(self, image_id: str, size: tuple[int, int]) -> ImageTk.PhotoImage:
        assert self.explorer is not None
        with Image.open(io.BytesIO(self.explorer.cropped_jpeg(image_id))) as source:
            image = source.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (13, 15, 19))
        canvas.paste(
            image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        )
        return ImageTk.PhotoImage(canvas)

    def _render_query(self) -> None:
        if self.explorer is None or not self.query_var.get():
            return
        query_id = self.query_var.get()
        result = self.explorer.result(query_id)
        try:
            position = self.visible_queries.index(query_id) + 1
        except ValueError:
            position = 0
        self.position_var.set(f"{position} / {len(self.visible_queries)}")
        self.photo_refs = []
        query_photo = self._photo(query_id, (280, 170))
        self.photo_refs.append(query_photo)
        self.query_image.configure(image=query_photo)
        self.query_id_label.configure(text=f"ID  {query_id}")
        refusal = self.explorer.manifest.get("refusal", {})
        threshold = float(refusal.get("probability_threshold", 0.0))
        if result["accepted"]:
            confidence = float(result["confidence"])
            self.decision_var.set("ПРИНЯТ")
            candidate_id = str(result["candidate_id"])
            short_candidate = f"{candidate_id[:12]}…{candidate_id[-6:]}"
            self.decision_detail_var.set(f"top-1  {short_candidate}")
            self.confidence_summary_var.set(
                f"confidence {confidence:.4f}  ·  порог {threshold:.4f}"
            )
            decision_bg = "#153b2a"
            decision_fg = "#7ee2ad"
            self.decision_tag.configure(text="ACCEPT", bg="#1d563c", fg="#9af0c4")
            self._update_confidence_meter(confidence, threshold, True)
        else:
            self.decision_var.set("ОТКАЗ")
            self.decision_detail_var.set("top-1 не экспортирован в candidates.csv")
            self.confidence_summary_var.set(
                f"score не экспортирован  ·  порог {threshold:.4f}"
            )
            decision_bg = "#482124"
            decision_fg = "#ffadb2"
            self.decision_tag.configure(text="REFUSE", bg="#6b2d33", fg="#ffc3c7")
            self._update_confidence_meter(None, threshold, False)
        self.decision_card.configure(bg=decision_bg)
        self.decision_label.configure(bg=decision_bg, fg=decision_fg)
        self.decision_detail_label.configure(bg=decision_bg, fg=decision_fg)
        self.confidence_label.configure(bg=decision_bg, fg=decision_fg)
        for child in self.decision_card.winfo_children():
            if child is not self.confidence_canvas:
                child.configure(bg=decision_bg)

        for child in self.gallery_grid.winfo_children():
            child.destroy()
        for item in result["top10"]:
            rank = int(item["rank"])
            row, column = divmod(rank - 1, 5)
            border = PINK if rank == 1 else BORDER
            card = tk.Frame(
                self.gallery_grid,
                bg=PANEL_ALT,
                padx=8,
                pady=8,
                highlightbackground=border,
                highlightthickness=1,
            )
            card.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)
            photo = self._photo(item["gallery_id"], (148, 103))
            self.photo_refs.append(photo)
            tk.Label(card, image=photo, bg="#090c12", bd=0).pack(fill=X)
            color = PINK if rank == 1 else PURPLE
            meta = tk.Frame(card, bg=PANEL_ALT)
            meta.pack(fill=X, pady=(7, 0))
            tk.Label(
                meta,
                text=f"#{rank}",
                bg=PANEL_ALT,
                fg=color,
                font=("Segoe UI Semibold", 10),
            ).pack(side=LEFT)
            if rank == 1:
                tk.Label(
                    meta,
                    text="TOP MATCH",
                    bg="#3a1730",
                    fg=PINK,
                    padx=5,
                    pady=2,
                    font=("Segoe UI Semibold", 6),
                ).pack(side=RIGHT)
            tk.Label(
                card,
                text=item["gallery_id"],
                bg=PANEL_ALT,
                fg=MUTED,
                font=("Cascadia Mono", 7),
                wraplength=148,
                justify="left",
            ).pack(anchor="w")

    def open_output(self) -> None:
        output = Path(self.output_var.get()).expanduser()
        if output.exists():
            _open_path(output)
        else:
            messagebox.showerror("Falcon ReID", f"Каталог не найден: {output}")

    def _load_settings(self) -> dict[str, str]:
        path = _settings_path()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_settings(self) -> None:
        path = _settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "dataset": self.dataset_var.get(),
                        "output": self.output_var.get(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _close(self) -> None:
        if self.busy:
            messagebox.showwarning(
                "Операция выполняется",
                "Дождитесь завершения Docker или проверки, чтобы не оставить незавершённый контейнер.",
            )
            return
        self._save_settings()
        self.root.destroy()


def main() -> None:
    _enable_dpi_awareness()
    root = Tk()
    _normalize_tk_scaling(root)
    FalconDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
