"""Render an annotated QC clip so a human can *see* whether the detectors are
tracking correctly - the cheapest, most honest validation of the vision stack.

For each frame it draws:
  * the three region boxes,
  * the detected object centroid (with its colour label) inside VISION,
  * the bar-fill value,
  * the tactor point (yellow/green) inside TACTOR,
  * each spool's measured angle line inside MOTORS.

Output is a short ``qc_overlay.mp4`` plus a single ``qc_frame.png`` for reports.
Run it after calibration to confirm the thresholds/ROIs are good before
committing to a full extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from side_camera_separator import Regions, annotate, detect_regions_robust, representative_frame
from region_signals import (
    detect_bar_fill, detect_object, detect_tactor, init_spools, measure_groove,
)


def _draw_frame(frame, regions: Regions, spools, colors: Optional[dict]):
    out = annotate(frame, regions)
    v, t = regions.vision, regions.tactor

    obj = detect_object(v.crop(frame), colors=colors)
    if np.isfinite(obj["obj_x"]):
        px, py = int(v.x0 + obj["obj_x"]), int(v.y0 + obj["obj_y"])
        cv2.circle(out, (px, py), 7, (0, 165, 255), 2)
        cv2.putText(out, str(obj["obj_color"]), (px + 8, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    bar = detect_bar_fill(v.crop(frame), colors=colors)
    cv2.putText(out, f"bar_fill={bar:.2f}", (v.x0 + 4, v.y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (144, 238, 144), 1, cv2.LINE_AA)

    tac = detect_tactor(t.crop(frame), colors=colors)
    if tac["tactor_found"]:
        cv2.circle(out, (int(t.x0 + tac["tactor_x"]), int(t.y0 + tac["tactor_y"])),
                   5, (0, 255, 255), -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for roi in spools:
        ang, con, direction = measure_groove(gray, roi.cx, roi.cy, roi.radius)
        cv2.circle(out, (roi.cx, roi.cy), int(roi.radius * 0.72), (0, 255, 0), 1)
        if direction is not None:
            dx, dy = direction
            L = roi.radius * 0.9
            cv2.line(out, (int(roi.cx - dx * L), int(roi.cy - dy * L)),
                     (int(roi.cx + dx * L), int(roi.cy + dy * L)), (0, 0, 255), 2)
    return out


def render(video_path: str | Path, out_dir: str | Path,
           regions: Optional[Regions] = None, spools=None, colors: Optional[dict] = None,
           start_frac: float = 0.4, seconds: float = 6.0) -> dict:
    """Write qc_overlay.mp4 + qc_frame.png for a short window of the video."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if regions is None:
        regions = detect_regions_robust(video_path)
    if spools is None:
        spools = init_spools(representative_frame(video_path, 0.5), regions.motors)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(n * start_frac)
    count = int(fps * seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("could not read video for QC overlay")
    h, w = frame.shape[:2]
    vw = cv2.VideoWriter(str(out_dir / "qc_overlay.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    png = out_dir / "qc_frame.png"
    for i in range(count):
        if i > 0:
            ok, frame = cap.read()
            if not ok:
                break
        ann = _draw_frame(frame, regions, spools, colors)
        vw.write(ann)
        if i == count // 2:
            cv2.imwrite(str(png), ann)
    cap.release()
    vw.release()
    return {"video": str(out_dir / "qc_overlay.mp4"), "frame": str(png)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render an annotated QC overlay clip.")
    ap.add_argument("video")
    ap.add_argument("--out", default="qc")
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()
    print(render(a.video, a.out, seconds=a.seconds))
