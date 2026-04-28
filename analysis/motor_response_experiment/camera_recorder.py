"""Background camera recorder.

Continuously grabs frames from a `cv2.VideoCapture` on its own thread and
writes them to disk as an MP4 video. The runner can request the most recent
frame at any time (used to capture the "settled" frame after each motor
command).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    index: int = 1
    width: Optional[int] = None
    height: Optional[int] = None
    fps: float = 30.0
    fourcc: str = "mp4v"
    backend: int = cv2.CAP_DSHOW   # DirectShow is much faster on Windows


class CameraRecorder:
    """Thread-based camera capture with optional MP4 recording."""

    def __init__(
        self,
        config: CameraConfig,
        video_path: Optional[Path] = None,
        log_print=print,
    ):
        self._cfg = config
        self._video_path = Path(video_path) if video_path is not None else None
        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frame_count: int = 0
        self._actual_fps: float = config.fps
        self._log = log_print

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        cap = cv2.VideoCapture(self._cfg.index, self._cfg.backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._cfg.index)  # fallback to default backend
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self._cfg.index}")

        if self._cfg.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        if self._cfg.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        if self._cfg.fps:
            cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        # Warm up the camera and grab one real frame.
        warmup_deadline = time.time() + 2.0
        first_frame: Optional[np.ndarray] = None
        while time.time() < warmup_deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                first_frame = frame
                break
            time.sleep(0.05)
        if first_frame is None:
            cap.release()
            raise RuntimeError("Camera opened but did not deliver any frames")

        self._cap = cap
        self._latest_frame = first_frame
        self._latest_ts = time.time()
        h, w = first_frame.shape[:2]

        if self._video_path is not None:
            self._video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self._cfg.fourcc)
            fps = float(self._cfg.fps if self._cfg.fps else 30.0)
            self._writer = cv2.VideoWriter(str(self._video_path), fourcc, fps, (w, h))
            if not self._writer.isOpened():
                self._log(f"[camera] WARNING: VideoWriter failed to open {self._video_path}")
                self._writer = None

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera-recorder", daemon=True)
        self._thread.start()
        self._log(
            f"[camera] started: idx={self._cfg.index}  size={w}x{h}  "
            f"fps={self._cfg.fps}  recording={self._video_path is not None}"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._frame_count > 0:
            self._log(
                f"[camera] stopped: captured {self._frame_count} frames at "
                f"{self._actual_fps:.1f} fps"
            )

    def __enter__(self) -> "CameraRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        assert self._cap is not None
        ema_dt = 1.0 / max(1.0, self._cfg.fps)
        last_t = time.time()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            now = time.time()
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = now
                self._frame_count += 1
            if self._writer is not None:
                self._writer.write(frame)

            dt = now - last_t
            last_t = now
            if dt > 0:
                ema_dt = 0.9 * ema_dt + 0.1 * dt
                self._actual_fps = 1.0 / ema_dt if ema_dt > 0 else 0.0

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            return (None if self._latest_frame is None else self._latest_frame.copy(), self._latest_ts)

    def wait_for_frame_after(self, ts_after: float, timeout_s: float = 3.0) -> Optional[np.ndarray]:
        """Wait until ``latest_ts > ts_after``, then return a copy of that frame.

        Used after settling so we do not analyze a buffered frame captured *before*
        the movement finished (fixes target/angle mismatches).
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if (
                    self._latest_frame is not None
                    and self._latest_ts > ts_after
                ):
                    return self._latest_frame.copy()
            time.sleep(0.003)
        return None

    def wait_for_fresh(self, min_age_s: float = 0.0, timeout_s: float = 1.0) -> Optional[np.ndarray]:
        """Wait until at least one frame newer than `now - min_age_s` is available.

        Returns ``None`` on timeout.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._latest_frame is not None and (time.time() - self._latest_ts) <= min_age_s:
                    return self._latest_frame.copy()
            time.sleep(0.005)
        return None

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def frame_count(self) -> int:
        return self._frame_count
