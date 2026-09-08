"""GUI tool for safely cropping experiment videos.

The tool reads original videos in-place and writes derived cropped videos to a
`crop_videos` folder. It never moves or deletes source videos.
"""
from __future__ import annotations

import json
import math
import queue
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
ROTATION_OPTIONS = (0, 45, 90, 135, 180, -135, -90, -45)
PREVIEW_MAX_WIDTH = 320
PREVIEW_MAX_HEIGHT = 220
DEFAULT_CROP_COUNT = 4


@dataclass
class CropSpec:
    """One crop definition for one source video."""

    name: str
    video_key: str
    rect: tuple[int, int, int, int] | None = None  # x, y, width, height in source-frame pixels
    rotation: int = 0

    def is_complete(self) -> bool:
        return bool(self.name.strip()) and bool(self.video_key) and self.rect is not None


def sanitize_label(label: str, fallback: str = "crop") -> str:
    """Return a Windows-friendly filename token."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("._-")
    return cleaned or fallback


def validate_rotation(angle: int) -> int:
    """Validate that angle is one of the GUI-supported post-crop rotations."""
    angle = int(angle)
    if angle not in ROTATION_OPTIONS:
        raise ValueError(f"Unsupported rotation {angle}; choose one of {ROTATION_OPTIONS}")
    return angle


def rotation_token(angle: int) -> str:
    angle = validate_rotation(angle)
    return f"m{abs(angle)}" if angle < 0 else str(angle)


def default_output_dir(pair_dir: Path) -> Path:
    """Default safe destination for derived videos."""
    return pair_dir / "crop_videos"


def make_output_filename(experiment: str, pair: str, video_key: str, crop_name: str, rotation: int) -> str:
    """Build a clear output filename that includes provenance."""
    parts = [
        sanitize_label(experiment, "experiment"),
        sanitize_label(pair, "pair"),
        sanitize_label(video_key, "video"),
        sanitize_label(crop_name, "crop"),
        f"rot{rotation_token(rotation)}",
    ]
    return "_".join(parts) + ".mp4"


def video_files_in_pair(pair_dir: Path) -> list[Path]:
    files = [p for p in pair_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    preferred_order = {"side_camera": 0, "top_camera": 1}
    return sorted(files, key=lambda p: (preferred_order.get(p.stem, 99), p.name.lower()))


def build_video_map(video_paths: list[Path]) -> dict[str, Path]:
    """Build unique human-readable keys for all videos in a chosen folder."""
    video_map: dict[str, Path] = {}
    for path in video_paths:
        key = path.stem
        if key in video_map:
            suffix = path.suffix.lower().lstrip(".")
            key = f"{path.stem}_{suffix}"
        counter = 2
        base_key = key
        while key in video_map:
            key = f"{base_key}_{counter}"
            counter += 1
        video_map[key] = path
    return video_map


def find_experiment_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def find_pair_dirs(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.exists():
        return []
    dirs = [p for p in experiment_dir.iterdir() if p.is_dir() and video_files_in_pair(p)]
    return sorted(dirs, key=lambda p: p.name.lower())


def guess_default_data_root(start_dir: Path) -> Path:
    """Choose a helpful initial folder while still allowing any pasted path."""
    candidates = [
        start_dir / "Data",
        start_dir.parent / "Servomotors_thimble" / "Data",
        start_dir.parent / "Data",
        start_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return start_dir


def clamp_rect(rect: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Clamp x/y/w/h to valid frame boundaries."""
    x, y, w, h = rect
    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, min(frame_w - 1, int(round(x))))
    y1 = max(0, min(frame_h - 1, int(round(y))))
    x2 = max(0, min(frame_w, int(round(x + w))))
    y2 = max(0, min(frame_h, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop rectangle {rect} for frame {frame_w}x{frame_h}")
    return x1, y1, x2 - x1, y2 - y1


def crop_frame(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = clamp_rect(rect, frame.shape)
    return frame[y : y + h, x : x + w].copy()


def rotate_frame_bound(frame: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image around its center without clipping the crop."""
    angle = validate_rotation(angle)
    if angle == 0:
        return frame.copy()

    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_width = int(math.ceil((height * sin_a) + (width * cos_a)))
    new_height = int(math.ceil((height * cos_a) + (width * sin_a)))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(
        frame,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def export_crop_video(
    video_path: Path,
    rect: tuple[int, int, int, int],
    rotation: int,
    output_path: Path,
    progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Create one cropped video. Source video is opened read-only by OpenCV."""
    rotation = validate_rotation(rotation)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer: cv2.VideoWriter | None = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        written = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = crop_frame(frame, rect)
            rotated = rotate_frame_bound(cropped, rotation)
            out_h, out_w = rotated.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output_path}")
            writer.write(rotated)
            written += 1
            if progress and (written == 1 or written % 25 == 0):
                progress(written, total_frames)

        if written == 0:
            raise RuntimeError(f"No frames were read from: {video_path}")
        if progress:
            progress(written, total_frames)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


class VideoPreview:
    """Canvas wrapper that displays one video frame and supports rectangle drawing."""

    def __init__(self, parent: tk.Widget, app: "CropVideoApp", title: str) -> None:
        self.app = app
        self.video_key = ""
        self.frame_bgr: np.ndarray | None = None
        self.display_scale = 1.0
        self.display_width = PREVIEW_MAX_WIDTH
        self.display_height = PREVIEW_MAX_HEIGHT
        self.drag_start: tuple[int, int] | None = None
        self.photo: ImageTk.PhotoImage | None = None

        self.container = ttk.LabelFrame(parent, text=title)
        self.title_var = tk.StringVar(value="No video loaded")
        ttk.Label(self.container, textvariable=self.title_var).pack(anchor="w")
        self.canvas = tk.Canvas(
            self.container,
            width=PREVIEW_MAX_WIDTH,
            height=PREVIEW_MAX_HEIGHT,
            bg="#202020",
            cursor="crosshair",
            highlightthickness=1,
            highlightbackground="#606060",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def set_frame(self, video_key: str, frame_bgr: np.ndarray | None) -> None:
        self.video_key = video_key
        self.frame_bgr = frame_bgr
        self.title_var.set(video_key if frame_bgr is not None else "No video loaded")
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.frame_bgr is None:
            self.canvas.create_text(
                PREVIEW_MAX_WIDTH // 2,
                PREVIEW_MAX_HEIGHT // 2,
                text="Load a pair to preview video",
                fill="white",
            )
            return

        rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        self.display_scale = min(PREVIEW_MAX_WIDTH / width, PREVIEW_MAX_HEIGHT / height, 1.0)
        self.display_width = max(1, int(round(width * self.display_scale)))
        self.display_height = max(1, int(round(height * self.display_scale)))
        image = Image.fromarray(rgb).resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.draw_existing_rectangles()

    def draw_existing_rectangles(self) -> None:
        for idx, spec in enumerate(self.app.get_crop_specs_from_widgets(include_incomplete=True)):
            if spec.video_key != self.video_key or spec.rect is None:
                continue
            x, y, w, h = spec.rect
            x1, y1 = self.source_to_display(x, y)
            x2, y2 = self.source_to_display(x + w, y + h)
            color = "#ff3b30" if idx == self.app.active_crop_index.get() else "#00e676"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            label = spec.name.strip() or f"Crop {idx + 1}"
            self.canvas.create_text(x1 + 4, y1 + 4, anchor=tk.NW, text=label, fill=color)

    def source_to_display(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x * self.display_scale)), int(round(y * self.display_scale))

    def display_to_source(self, x: float, y: float) -> tuple[int, int]:
        if self.frame_bgr is None:
            return 0, 0
        frame_h, frame_w = self.frame_bgr.shape[:2]
        src_x = int(round(max(0, min(self.display_width, x)) / self.display_scale))
        src_y = int(round(max(0, min(self.display_height, y)) / self.display_scale))
        return max(0, min(frame_w, src_x)), max(0, min(frame_h, src_y))

    def on_press(self, event: tk.Event) -> None:
        if self.frame_bgr is None or not self.video_key:
            return
        self.drag_start = (event.x, event.y)
        self.app.set_active_crop_video(self.video_key)

    def on_drag(self, event: tk.Event) -> None:
        if self.drag_start is None or self.frame_bgr is None:
            return
        self.redraw()
        x0, y0 = self.drag_start
        x1 = max(0, min(self.display_width, event.x))
        y1 = max(0, min(self.display_height, event.y))
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ffcc00", width=2, dash=(4, 2))

    def on_release(self, event: tk.Event) -> None:
        if self.drag_start is None or self.frame_bgr is None:
            return
        x0, y0 = self.drag_start
        self.drag_start = None
        x1 = max(0, min(self.display_width, event.x))
        y1 = max(0, min(self.display_height, event.y))
        src_x0, src_y0 = self.display_to_source(min(x0, x1), min(y0, y1))
        src_x1, src_y1 = self.display_to_source(max(x0, x1), max(y0, y1))
        rect = (src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0)
        try:
            clamp_rect(rect, self.frame_bgr.shape)
        except ValueError as exc:
            messagebox.showwarning("Invalid crop", str(exc))
            self.redraw()
            return
        self.app.set_active_crop_rect(self.video_key, rect)
        self.redraw()


class CropVideoApp(tk.Tk):
    """Main GUI application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Experiment Video Cropper - safe derived outputs only")
        self.geometry("1280x820")
        self.minsize(900, 650)

        self.workspace_root = Path.cwd()
        self.current_experiment_dir: Path | None = None
        self.current_pair_dir: Path | None = None
        self.current_videos: dict[str, Path] = {}
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.data_root_var = tk.StringVar(value=str(guess_default_data_root(self.workspace_root)))
        self.output_dir_var = tk.StringVar(value="")
        self.frame_percent_var = tk.IntVar(value=0)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.active_crop_index = tk.IntVar(value=0)

        self.crop_name_vars: list[tk.StringVar] = []
        self.crop_video_vars: list[tk.StringVar] = []
        self.crop_rotation_vars: list[tk.StringVar] = []
        self.crop_rect_vars: list[tk.StringVar] = []
        self.crop_rects: list[tuple[int, int, int, int] | None] = []
        self.crop_video_combos: list[ttk.Combobox] = []
        self.crop_rows: list[ttk.Frame] = []
        self.previews: dict[str, VideoPreview] = {}

        self._build_ui()
        self.after(100, self.ask_for_initial_data_root)

    def ask_for_initial_data_root(self) -> None:
        """Ask which folder contains the videos before the user starts working."""
        dialog = tk.Toplevel(self)
        dialog.title("Choose video folder")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(True, False)

        ttk.Label(
            dialog,
            text=(
                "Paste or browse to the folder that contains the videos you want to crop.\n"
                "This can be a pair folder with mp4 files, or a parent folder with pair folders."
            ),
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=12, pady=(12, 6))

        path_var = tk.StringVar(value=self.data_root_var.get())
        row = ttk.Frame(dialog)
        row.pack(fill=tk.X, padx=12, pady=6)
        entry = ttk.Entry(row, textvariable=path_var, width=95)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.focus_set()
        entry.select_range(0, tk.END)

        def browse() -> None:
            chosen = filedialog.askdirectory(title="Select folder containing videos")
            if chosen:
                path_var.set(chosen)

        ttk.Button(row, text="Browse", command=browse).pack(side=tk.LEFT, padx=(6, 0))

        button_row = ttk.Frame(dialog)
        button_row.pack(fill=tk.X, padx=12, pady=(6, 12))

        def use_folder() -> None:
            candidate = Path(path_var.get().strip().strip('"'))
            if not candidate.exists() or not candidate.is_dir():
                messagebox.showwarning("Folder not found", f"Please choose an existing folder:\n{candidate}")
                return
            self.data_root_var.set(str(candidate))
            dialog.destroy()
            self.refresh_experiments()

        def cancel_dialog() -> None:
            dialog.destroy()
            self.refresh_experiments()

        ttk.Button(button_row, text="Use this folder", command=use_folder).pack(side=tk.RIGHT)
        ttk.Button(button_row, text="Cancel", command=cancel_dialog).pack(side=tk.RIGHT, padx=6)
        dialog.bind("<Return>", lambda _event: use_folder())
        dialog.bind("<Escape>", lambda _event: cancel_dialog())
        self.wait_window(dialog)

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(top, text="Video folder:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.data_root_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(top, text="Browse", command=self.browse_data_root).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Load folder", command=self.refresh_experiments).pack(side=tk.LEFT, padx=2)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        browser = ttk.Frame(main, width=260)
        main.add(browser, weight=0)
        ttk.Label(browser, text="Parent folders / experiments").pack(anchor="w")
        self.experiment_list = tk.Listbox(browser, height=10, exportselection=False)
        self.experiment_list.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.experiment_list.bind("<<ListboxSelect>>", lambda _event: self.on_experiment_selected())
        ttk.Label(browser, text="Video folders / pairs").pack(anchor="w")
        self.pair_list = tk.Listbox(browser, height=14, exportselection=False)
        self.pair_list.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.pair_list.bind("<Double-Button-1>", lambda _event: self.load_selected_pair())
        ttk.Button(browser, text="Load selected video folder", command=self.load_selected_pair).pack(fill=tk.X)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        preview_controls = ttk.Frame(right)
        preview_controls.pack(fill=tk.X)
        ttk.Label(preview_controls, text="Preview frame position (%):").pack(side=tk.LEFT)
        ttk.Scale(
            preview_controls,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.frame_percent_var,
            command=lambda _value: self.load_preview_frames(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(preview_controls, text="Reload preview", command=self.load_preview_frames).pack(side=tk.LEFT)

        preview_area = ttk.LabelFrame(right, text="Video previews - one panel is created for every video in the folder")
        preview_area.pack(fill=tk.BOTH, expand=True, pady=(4, 2))
        self.preview_canvas = tk.Canvas(preview_area, highlightthickness=0)
        preview_scroll_y = ttk.Scrollbar(preview_area, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        preview_scroll_x = ttk.Scrollbar(preview_area, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=preview_scroll_y.set, xscrollcommand=preview_scroll_x.set)
        preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        preview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview_grid = ttk.Frame(self.preview_canvas)
        self.preview_canvas_window = self.preview_canvas.create_window((0, 0), window=self.preview_grid, anchor=tk.NW)
        self.preview_grid.bind(
            "<Configure>",
            lambda _event: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")),
        )

        crops = ttk.LabelFrame(right, text="Crop definitions - select a row, then draw on either video")
        crops.pack(fill=tk.X, pady=6)
        header = ttk.Frame(crops)
        header.pack(fill=tk.X)
        for text, width in [("Use", 5), ("Name", 24), ("Video", 18), ("Rotation after crop", 18), ("Rectangle", 28)]:
            ttk.Label(header, text=text, width=width).pack(side=tk.LEFT, padx=2)

        self.crop_rows_frame = ttk.Frame(crops)
        self.crop_rows_frame.pack(fill=tk.X)
        for _idx in range(DEFAULT_CROP_COUNT):
            self.add_crop_row()
        crop_buttons = ttk.Frame(crops)
        crop_buttons.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(crop_buttons, text="Add another crop row", command=self.add_crop_row).pack(side=tk.LEFT, padx=2)

        output = ttk.LabelFrame(right, text="Output")
        output.pack(fill=tk.X, pady=4)
        ttk.Label(output, text="Folder:").pack(side=tk.LEFT)
        ttk.Entry(output, textvariable=self.output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(output, text="Browse", command=self.browse_output_dir).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(output, text="Overwrite existing files", variable=self.overwrite_var).pack(side=tk.LEFT, padx=6)

        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=4)
        self.export_button = ttk.Button(actions, text="Export filled crops", command=self.export_filled_crops)
        self.export_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Save crop settings", command=self.save_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Load crop settings", command=self.load_settings).pack(side=tk.LEFT, padx=2)

        log_frame = ttk.LabelFrame(right, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=False, pady=4)
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log("Ready. Choose a folder with videos; original videos stay in place.")

    def add_crop_row(self) -> None:
        idx = len(self.crop_name_vars)
        row = ttk.Frame(self.crop_rows_frame)
        row.pack(fill=tk.X, pady=1)
        self.crop_rows.append(row)

        ttk.Radiobutton(row, variable=self.active_crop_index, value=idx).pack(side=tk.LEFT, padx=8)
        name_var = tk.StringVar(value=f"crop_{idx + 1}")
        video_var = tk.StringVar(value="")
        rot_var = tk.StringVar(value="0")
        rect_var = tk.StringVar(value="not selected")
        self.crop_name_vars.append(name_var)
        self.crop_video_vars.append(video_var)
        self.crop_rotation_vars.append(rot_var)
        self.crop_rect_vars.append(rect_var)
        self.crop_rects.append(None)

        ttk.Entry(row, textvariable=name_var, width=26).pack(side=tk.LEFT, padx=2)
        combo_video = ttk.Combobox(
            row,
            textvariable=video_var,
            values=list(self.current_videos.keys()),
            width=18,
            state="readonly",
        )
        combo_video.pack(side=tk.LEFT, padx=2)
        self.crop_video_combos.append(combo_video)
        ttk.Combobox(
            row,
            textvariable=rot_var,
            values=[str(x) for x in ROTATION_OPTIONS],
            width=18,
            state="readonly",
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, textvariable=rect_var, width=30).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Clear", command=lambda i=idx: self.clear_crop(i)).pack(side=tk.LEFT, padx=2)

    def ensure_crop_rows(self, minimum_rows: int) -> None:
        while len(self.crop_name_vars) < minimum_rows:
            self.add_crop_row()

    def redraw_previews(self) -> None:
        for preview in self.previews.values():
            preview.redraw()

    def log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def browse_data_root(self) -> None:
        chosen = filedialog.askdirectory(title="Select folder with videos")
        if chosen:
            self.data_root_var.set(chosen)
            self.refresh_experiments()

    def browse_output_dir(self) -> None:
        initial = self.output_dir_var.get() or str(self.workspace_root)
        chosen = filedialog.askdirectory(title="Select output folder", initialdir=initial)
        if chosen:
            self.output_dir_var.set(chosen)

    def refresh_experiments(self) -> None:
        self.experiment_list.delete(0, tk.END)
        self.pair_list.delete(0, tk.END)
        data_root = Path(self.data_root_var.get().strip().strip('"'))
        if not data_root.exists() or not data_root.is_dir():
            self.log(f"Folder not found: {data_root}")
            messagebox.showwarning("Folder not found", f"Please choose an existing folder:\n{data_root}")
            return

        direct_videos = video_files_in_pair(data_root)
        if direct_videos:
            self.load_video_folder(data_root)
            return

        pair_dirs = find_pair_dirs(data_root)
        if pair_dirs:
            self.current_experiment_dir = data_root
            for pair in pair_dirs:
                self.pair_list.insert(tk.END, pair.name)
            self.log(f"Found {len(pair_dirs)} video folder(s) under {data_root}")
            if len(pair_dirs) == 1:
                self.pair_list.selection_set(0)
                self.load_selected_pair()
            return

        experiments = [p for p in find_experiment_dirs(data_root) if find_pair_dirs(p) or video_files_in_pair(p)]
        for exp in experiments:
            self.experiment_list.insert(tk.END, exp.name)
        self.log(
            f"Found {len(experiments)} parent folder(s) under {data_root}. "
            "If you want one pair, choose the folder that directly contains the mp4 files."
        )

    def load_video_folder(self, video_dir: Path) -> None:
        videos = video_files_in_pair(video_dir)
        if not videos:
            messagebox.showwarning("No videos", f"No video files found directly in:\n{video_dir}")
            return

        self.current_pair_dir = video_dir
        self.current_experiment_dir = video_dir.parent
        self.current_videos = build_video_map(videos)
        video_keys = list(self.current_videos.keys())
        self.ensure_crop_rows(max(DEFAULT_CROP_COUNT, len(video_keys)))

        for idx, var in enumerate(self.crop_video_vars):
            combo = self.crop_video_combos[idx]
            combo.configure(values=video_keys)
            if not var.get() or var.get() not in video_keys:
                var.set(video_keys[min(idx, len(video_keys) - 1)])

        self.output_dir_var.set(str(default_output_dir(video_dir)))
        self.clear_rectangles_only()
        self.rebuild_preview_grid(video_keys)
        self.load_preview_frames()
        self.log(f"Loaded {len(videos)} video(s) from {video_dir}")
        self.log(f"Default output: {self.output_dir_var.get()}")

    def selected_experiment_path(self) -> Path | None:
        selection = self.experiment_list.curselection()
        if not selection:
            return None
        return Path(self.data_root_var.get()) / self.experiment_list.get(selection[0])

    def on_experiment_selected(self) -> None:
        self.pair_list.delete(0, tk.END)
        exp = self.selected_experiment_path()
        self.current_experiment_dir = exp
        if exp is None:
            return
        if video_files_in_pair(exp):
            self.load_video_folder(exp)
            return
        pairs = find_pair_dirs(exp)
        for pair in pairs:
            self.pair_list.insert(tk.END, pair.name)
        self.log(f"Selected {exp.name}: found {len(pairs)} video folder(s)")

    def load_selected_pair(self) -> None:
        if self.current_experiment_dir is None:
            self.on_experiment_selected()
        if self.current_experiment_dir is None:
            messagebox.showinfo("No video folder", "Please select a folder with videos first.")
            return
        selection = self.pair_list.curselection()
        if not selection:
            messagebox.showinfo("No video folder", "Please select a video folder/pair first.")
            return
        pair_dir = self.current_experiment_dir / self.pair_list.get(selection[0])
        self.load_video_folder(pair_dir)

    def clear_rectangles_only(self) -> None:
        self.crop_rects = [None] * len(self.crop_rect_vars)
        for rect_var in self.crop_rect_vars:
            rect_var.set("not selected")
        self.redraw_previews()

    def clear_crop(self, idx: int) -> None:
        self.crop_rects[idx] = None
        self.crop_rect_vars[idx].set("not selected")
        self.redraw_previews()

    def rebuild_preview_grid(self, video_keys: list[str]) -> None:
        for preview in self.previews.values():
            preview.container.destroy()
        self.previews = {}
        columns = min(max(len(video_keys), 1), 4)
        for idx, video_key in enumerate(video_keys):
            preview = VideoPreview(self.preview_grid, self, video_key)
            row = idx // columns
            col = idx % columns
            preview.container.grid(row=row, column=col, sticky="nsew", padx=6, pady=4)
            self.preview_grid.columnconfigure(col, weight=1)
            self.previews[video_key] = preview

    def load_preview_frames(self) -> None:
        if not self.current_videos:
            return
        percent = self.frame_percent_var.get() / 100.0
        for video_key, video_path in self.current_videos.items():
            preview = self.previews.get(video_key)
            if preview is None:
                continue
            frame = self.read_frame_at_percent(video_path, percent)
            preview.set_frame(video_key, frame)

    @staticmethod
    def read_frame_at_percent(video_path: Path, percent: float) -> np.ndarray | None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 1:
                target = int(round(max(0.0, min(1.0, percent)) * (total - 1)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def set_active_crop_video(self, video_key: str) -> None:
        self.crop_video_vars[self.active_crop_index.get()].set(video_key)

    def set_active_crop_rect(self, video_key: str, rect: tuple[int, int, int, int]) -> None:
        idx = self.active_crop_index.get()
        self.crop_video_vars[idx].set(video_key)
        self.crop_rects[idx] = rect
        self.crop_rect_vars[idx].set(f"x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]}")
        self.redraw_previews()

    def get_crop_specs_from_widgets(self, include_incomplete: bool = False) -> list[CropSpec]:
        specs: list[CropSpec] = []
        for idx in range(len(self.crop_name_vars)):
            rotation_text = self.crop_rotation_vars[idx].get() or "0"
            try:
                rotation = validate_rotation(int(rotation_text))
            except ValueError:
                rotation = 0
            spec = CropSpec(
                name=self.crop_name_vars[idx].get(),
                video_key=self.crop_video_vars[idx].get(),
                rect=self.crop_rects[idx],
                rotation=rotation,
            )
            if include_incomplete or spec.is_complete():
                specs.append(spec)
        return specs

    def validate_export_specs(self) -> list[CropSpec]:
        if self.current_pair_dir is None or self.current_experiment_dir is None:
            raise ValueError("Load an experiment pair before exporting.")
        specs = self.get_crop_specs_from_widgets(include_incomplete=False)
        if not specs:
            raise ValueError("No complete crop rows. Draw at least one rectangle and give it a name.")
        output_dir = Path(self.output_dir_var.get())
        if not output_dir.name:
            raise ValueError("Choose an output folder.")
        if output_dir.resolve() in {p.resolve() for p in self.current_videos.values()}:
            raise ValueError("Output folder cannot be a video file path.")
        filenames = [
            make_output_filename(
                self.current_experiment_dir.name,
                self.current_pair_dir.name,
                spec.video_key,
                spec.name,
                spec.rotation,
            )
            for spec in specs
        ]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Two crop rows would create the same filename. Use unique crop names.")
        for spec in specs:
            if spec.video_key not in self.current_videos:
                raise ValueError(f"Unknown video selection: {spec.video_key}")
            validate_rotation(spec.rotation)
        return specs

    def export_filled_crops(self) -> None:
        try:
            specs = self.validate_export_specs()
        except ValueError as exc:
            messagebox.showwarning("Cannot export", str(exc))
            return

        if self.current_pair_dir is None or self.current_experiment_dir is None:
            messagebox.showwarning("Cannot export", "Load an experiment pair before exporting.")
            return
        output_dir = Path(self.output_dir_var.get())
        jobs = []
        existing = []
        for spec in specs:
            output_path = output_dir / make_output_filename(
                self.current_experiment_dir.name,
                self.current_pair_dir.name,
                spec.video_key,
                spec.name,
                spec.rotation,
            )
            if output_path.exists():
                existing.append(output_path.name)
            jobs.append((spec, self.current_videos[spec.video_key], output_path))

        if existing and not self.overwrite_var.get():
            proceed = messagebox.askyesno(
                "Existing outputs",
                "These files already exist:\n\n"
                + "\n".join(existing[:10])
                + ("\n..." if len(existing) > 10 else "")
                + "\n\nOverwrite them?",
            )
            if not proceed:
                self.log("Export cancelled before writing outputs.")
                return

        self.export_button.configure(state=tk.DISABLED)
        self.log(f"Starting export of {len(jobs)} crop video(s) to {output_dir}")
        worker = threading.Thread(target=self._export_worker, args=(jobs,), daemon=True)
        worker.start()
        self.after(100, self.poll_worker_queue)

    def _export_worker(self, jobs: Iterable[tuple[CropSpec, Path, Path]]) -> None:
        try:
            for spec, video_path, output_path in jobs:
                self.worker_queue.put(("log", f"Exporting {spec.name} from {video_path.name} -> {output_path.name}"))

                def progress(done: int, total: int, crop_name: str = spec.name) -> None:
                    if total:
                        self.worker_queue.put(("log", f"  {crop_name}: {done}/{total} frames"))
                    else:
                        self.worker_queue.put(("log", f"  {crop_name}: {done} frames"))

                if spec.rect is None:
                    raise RuntimeError(f"Missing rectangle for crop: {spec.name}")
                export_crop_video(video_path, spec.rect, spec.rotation, output_path, progress=progress)
            self.worker_queue.put(("done", "Export complete."))
        except Exception as exc:  # surfaced to GUI log/messagebox on main thread
            self.worker_queue.put(("error", str(exc)))

    def poll_worker_queue(self) -> None:
        keep_polling = True
        while True:
            try:
                kind, payload = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log(str(payload))
            elif kind == "done":
                self.log(str(payload))
                messagebox.showinfo("Done", str(payload))
                self.export_button.configure(state=tk.NORMAL)
                keep_polling = False
            elif kind == "error":
                self.log("ERROR: " + str(payload))
                messagebox.showerror("Export failed", str(payload))
                self.export_button.configure(state=tk.NORMAL)
                keep_polling = False
        if keep_polling:
            self.after(100, self.poll_worker_queue)

    def settings_payload(self) -> dict[str, object]:
        return {
            "experiment": self.current_experiment_dir.name if self.current_experiment_dir else "",
            "pair": self.current_pair_dir.name if self.current_pair_dir else "",
            "crops": [asdict(spec) for spec in self.get_crop_specs_from_widgets(include_incomplete=True)],
        }

    def save_settings(self) -> None:
        if self.current_pair_dir is None:
            messagebox.showinfo("No pair", "Load a pair before saving settings.")
            return
        default_path = Path(self.output_dir_var.get() or default_output_dir(self.current_pair_dir)) / "crop_settings.json"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        chosen = filedialog.asksaveasfilename(
            title="Save crop settings",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not chosen:
            return
        Path(chosen).write_text(json.dumps(self.settings_payload(), indent=2), encoding="utf-8")
        self.log(f"Saved settings: {chosen}")

    def load_settings(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Load crop settings",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not chosen:
            return
        payload = json.loads(Path(chosen).read_text(encoding="utf-8"))
        crops = payload.get("crops", [])
        if not isinstance(crops, list):
            messagebox.showwarning("Invalid settings", "Settings file does not contain a crop list.")
            return
        self.ensure_crop_rows(len(crops))
        for idx in range(len(self.crop_name_vars)):
            if idx >= len(crops) or not isinstance(crops[idx], dict):
                continue
            item = crops[idx]
            self.crop_name_vars[idx].set(str(item.get("name", f"crop_{idx + 1}")))
            video_key = str(item.get("video_key", ""))
            if video_key:
                self.crop_video_vars[idx].set(video_key)
            rotation = int(item.get("rotation", 0))
            self.crop_rotation_vars[idx].set(str(validate_rotation(rotation)))
            rect = item.get("rect")
            if isinstance(rect, list) and len(rect) == 4:
                rect_tuple = tuple(int(v) for v in rect)
                self.crop_rects[idx] = rect_tuple  # type: ignore[assignment]
                self.crop_rect_vars[idx].set(
                    f"x={rect_tuple[0]}, y={rect_tuple[1]}, w={rect_tuple[2]}, h={rect_tuple[3]}"
                )
        self.redraw_previews()
        self.log(f"Loaded settings: {chosen}")


def main() -> None:
    app = CropVideoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
