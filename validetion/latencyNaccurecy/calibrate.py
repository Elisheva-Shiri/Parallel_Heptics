"""Interactive, once-per-session calibration for the side-camera pipeline.

Motivation
----------
Auto-detection (largest-dark-blob monitor, green-blob tactor, Hough spools) is
convenient but brittle across rigs/lighting and gives only ~20 px spools. This
module lets the operator spend ~30 s up front on **one good frame** to lock in:

  * the three region boxes (VISION / MOTORS / TACTOR) - confirm the auto guess
    or draw them by hand;
  * the HSV colour ranges for the tactor green, its yellow centre, and the
    orange/blue on-screen object - sampled by *clicking* those things, so the
    thresholds match the actual camera/lighting instead of hard-coded guesses;
  * the individual spool circles (clicked) when auto-Hough finds too few.

Everything is saved to a single ``calibration.json`` that `build_signals.py`
reuses for every pair, so the manual step happens once per session.

The GUI helpers need a display; the numeric helpers (`hsv_range_from_samples`,
`Calibration` (de)serialisation, `apply_to_regions`) are pure and unit-tested.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from side_camera_separator import (
    Box, Regions, annotate, detect_regions_robust, representative_frame,
)


# --------------------------------------------------------------------------- #
# Colour sampling (pure, testable)
# --------------------------------------------------------------------------- #

def hsv_range_from_samples(
    bgr_pixels: np.ndarray,
    h_pad: int = 10, s_pad: int = 60, v_pad: int = 60,
    s_floor: int = 40, v_floor: int = 40,
) -> tuple[list[int], list[int]]:
    """Robust HSV ``(lo, hi)`` from a set of sampled BGR pixels.

    Uses the 5th/95th percentiles (ignoring outliers from anti-aliased edges)
    plus a pad, so the range covers the object's natural variation under the
    actual lighting. Hue wrap-around near red is handled by widening, not
    splitting (callers needing red can post-split). Saturation/Value lower
    bounds are floored so we never accept washed-out greys.
    """
    px = np.asarray(bgr_pixels, dtype=np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    lo_p = np.percentile(hsv, 5, axis=0)
    hi_p = np.percentile(hsv, 95, axis=0)
    lo = [
        int(max(0, lo_p[0] - h_pad)),
        int(max(s_floor, lo_p[1] - s_pad)),
        int(max(v_floor, lo_p[2] - v_pad)),
    ]
    hi = [
        int(min(179, hi_p[0] + h_pad)),
        int(min(255, hi_p[1] + s_pad)),
        int(min(255, hi_p[2] + v_pad)),
    ]
    return lo, hi


# --------------------------------------------------------------------------- #
# Calibration container (pure, testable)
# --------------------------------------------------------------------------- #

@dataclass
class Calibration:
    """Everything the pipeline needs that benefits from a human in the loop."""

    frame_w: int
    frame_h: int
    regions: dict                       # {"vision":[x0,y0,x1,y1], ...}
    colors: dict = field(default_factory=dict)   # {"green":[lo,hi], "yellow":..., "orange":..., "blue":...}
    spools: list = field(default_factory=list)   # [[cx,cy,r], ...] in full-frame coords
    notes: str = ""

    def to_regions(self) -> Regions:
        r = self.regions
        return Regions(self.frame_w, self.frame_h,
                       Box.from_list(r["vision"]), Box.from_list(r["motors"]),
                       Box.from_list(r["tactor"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Interactive helpers (need a display)
# --------------------------------------------------------------------------- #
# Colour each region/object is drawn/labelled in (BGR), matching annotate().
LABEL_WHITE = (255, 255, 255)
COL_VISION = (0, 165, 255)   # orange
COL_MOTORS = (255, 80, 0)    # blue
COL_TACTOR = (0, 255, 0)     # green


def _put_label(img: np.ndarray, lines: list, color, org=(8, 4)) -> None:
    """Draw multi-line text with a black outline so it is readable on any
    background, in ``color`` (BGR). Each entry of ``lines`` can be a string
    (uses ``color``) or a ``(text, col)`` tuple to colour that line itself."""
    x, y = org
    for i, item in enumerate(lines):
        text, col = item if isinstance(item, tuple) else (item, color)
        yy = y + 20 + i * 22
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)


def _sample_clicks(frame_bgr: np.ndarray, win: str, instructions: list,
                   color=(0, 0, 255), patch: int = 3) -> np.ndarray:
    """Left-click several points on the target; return sampled BGR pixels.

    ``instructions`` are short on-image lines telling what/how to mark, drawn in
    ``color`` (the object's own colour). Enter=done, Esc=skip."""
    pts: list[np.ndarray] = []
    base = frame_bgr.copy()
    _put_label(base, instructions, color)
    disp = base.copy()

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            y0, y1 = max(0, y - patch), min(frame_bgr.shape[0], y + patch + 1)
            x0, x1 = max(0, x - patch), min(frame_bgr.shape[1], x + patch + 1)
            pts.append(frame_bgr[y0:y1, x0:x1].reshape(-1, 3))
            cv2.circle(disp, (x, y), 4, color, -1)
            cv2.putText(disp, str(len(pts)), (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow(win, disp)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, disp)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10, 27):
            break
    cv2.destroyWindow(win)
    return np.vstack(pts) if pts else np.empty((0, 3), np.uint8)


_REGION_COL = {"vision": COL_VISION, "motors": COL_MOTORS, "tactor": COL_TACTOR}


def pick_box(frame_bgr: np.ndarray, name: str, boxes: dict, win: str) -> Box:
    """Drag a new rectangle for region ``name``, drawn in that region's colour.

    Unlike cv2.selectROI (single fixed colour), this keeps the *other* boxes
    visible in their own colours and draws the box being edited in ``name``'s
    colour. Drag with the left mouse button; Enter=confirm, r=redo, Esc=cancel."""
    col = _REGION_COL[name]
    state = {"p0": None, "p1": None, "drawing": False}

    def render():
        img = frame_bgr.copy()
        for other, b in boxes.items():           # the other boxes, in their colours
            if other != name:
                cv2.rectangle(img, (b.x0, b.y0), (b.x1, b.y1), _REGION_COL[other], 1)
        if state["p0"] and state["p1"]:
            cv2.rectangle(img, state["p0"], state["p1"], col, 2)
        _put_label(img, [(f"Draw the {name.upper()} box (drag a rectangle)", col),
                         ("Enter = confirm   r = redo   Esc = cancel", col)], col)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["p0"], state["p1"], state["drawing"] = (x, y), (x, y), True
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["p1"] = (x, y); render()
        elif event == cv2.EVENT_LBUTTONUP:
            state["p1"], state["drawing"] = (x, y), False; render()

    cv2.setMouseCallback(win, on_mouse)
    render()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):           # Enter
            if state["p0"] and state["p1"]:
                (x0, y0), (x1, y1) = state["p0"], state["p1"]
                return Box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if k in (ord("r"), ord("R")):
            state["p0"] = state["p1"] = None; render()
        if k == 27:                 # Esc -> keep existing
            return boxes[name]


def confirm_or_edit_regions(frame_bgr: np.ndarray, regions: Regions) -> Regions:
    """Show the (auto) regions and let the operator accept or redraw any of them.

    Keys: Enter=accept all, v/m/t=redraw that region, Esc=accept as-is.
    """
    boxes = regions.boxes()
    win = "Step 1/3: regions"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        disp = annotate(frame_bgr, regions)
        _put_label(disp, [
            ("STEP 1/3  Confirm the 3 region boxes", LABEL_WHITE),
            ("Enter = accept all", LABEL_WHITE),
            ("v = redraw VISION (monitor)", COL_VISION),
            ("m = redraw MOTORS (spools)", COL_MOTORS),
            ("t = redraw TACTOR (finger)", COL_TACTOR),
            ("Esc = keep as-is", LABEL_WHITE),
        ], LABEL_WHITE)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10, 27):
            break
        if k in (ord("v"), ord("m"), ord("t")):
            name = {ord("v"): "vision", ord("m"): "motors", ord("t"): "tactor"}[k]
            boxes[name] = pick_box(frame_bgr, name, boxes, win)
            regions = Regions(regions.frame_w, regions.frame_h,
                              boxes["vision"], boxes["motors"], boxes["tactor"])
            cv2.setMouseCallback(win, lambda *a: None)  # drop the box-drag callback
    cv2.destroyAllWindows()
    return regions


def pick_spools(frame_bgr: np.ndarray, motors: Box, n_max: int = 3, zoom: int = 9) -> list:
    """Click the centre of each motor disc on a zoomed view of the motors region.

    The discs are tiny in the full frame, so we zoom the motors ROI. Click each
    disc centre (top-to-bottom doesn't matter, they're sorted), press Enter when
    done, ``z`` to undo the last click, Esc to skip. Radius is inferred from the
    spacing between clicks. Returns ``[[cx, cy, r], ...]`` in full-frame coords."""
    crop = motors.crop(frame_bgr)
    if crop.size == 0:
        return []
    big = cv2.resize(crop, (crop.shape[1] * zoom, crop.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
    clicks: list[tuple[int, int]] = []

    win = "Step 3/3: motor discs"
    disc_col = (0, 0, 255)  # red marks on the white/grey discs

    def redraw():
        img = big.copy()
        for i, (x, y) in enumerate(clicks, 1):
            cv2.circle(img, (x, y), 7, disc_col, 2)
            cv2.putText(img, str(i), (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, disc_col, 2)
        _put_label(img, [
            (f"STEP 3/3  Click each DISC centre  ({len(clicks)}/{n_max})", disc_col),
            ("z = undo last   Enter = done   Esc = skip", disc_col),
        ], disc_col)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < n_max:
            clicks.append((x, y))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10, 27):
            break
        if k in (ord("z"), ord("Z")) and clicks:
            clicks.pop()
            redraw()
    cv2.destroyWindow(win)
    if not clicks:
        return []
    # full-frame centres
    centres = [(motors.x0 + x / zoom, motors.y0 + y / zoom) for (x, y) in clicks]
    centres.sort(key=lambda p: p[1])
    ys = [c[1] for c in centres]
    spacing = float(np.median(np.diff(ys))) if len(ys) > 1 else float(min(motors.width, motors.height) * 0.4)
    r = int(max(5, min(spacing * 0.45, motors.width * 0.45)))
    return [[int(round(cx)), int(round(cy)), r] for (cx, cy) in centres]


def calibrate_session(video_path: str | Path, out_path: str | Path,
                      at_fraction: float = 0.5) -> Calibration:
    """Full interactive calibration on one representative frame -> calibration.json."""
    frame = representative_frame(video_path, at_fraction)
    h, w = frame.shape[:2]
    regions = confirm_or_edit_regions(frame, detect_regions_robust(video_path))

    # (object name, window title, BGR colour of the text, on-image instructions)
    colors: dict = {}
    prompts = [
        ("green", "Step 2/3: GREEN tactor body", (0, 200, 0),
         ["STEP 2/3  GREEN tactor body", "click 3+ spots on the green tactor",
          "Enter = next   Esc = skip"]),
        ("yellow", "Step 2/3: YELLOW tactor centre", (0, 215, 215),
         ["STEP 2/3  YELLOW tactor centre", "click 3+ spots on the yellow dot",
          "Enter = next   Esc = skip"]),
        ("orange", "Step 2/3: ORANGE object", (0, 165, 255),
         ["STEP 2/3  ORANGE object on the screen", "click 3+ spots on the orange square",
          "Enter = next   Esc = skip"]),
        ("blue", "Step 2/3: BLUE object", (255, 130, 30),
         ["STEP 2/3  BLUE object on the screen", "click 3+ spots (Esc if the object is orange)",
          "Enter = next   Esc = skip"]),
    ]
    for key, win, col, lines in prompts:
        samples = _sample_clicks(frame, win, lines, color=col)
        if samples.shape[0] >= 3:
            colors[key] = list(hsv_range_from_samples(samples))

    spools = pick_spools(frame, regions.motors)

    cal = Calibration(frame_w=w, frame_h=h, regions=regions.to_dict(), colors=colors,
                      spools=spools,
                      notes=f"calibrated from {Path(video_path).name} @ frac {at_fraction}")
    cal.save(out_path)
    print(f"saved calibration ({len(spools)} discs, {len(colors)} colours) -> {out_path}")
    return cal


def auto_calibrate(video_path: str | Path, out_path: str | Path) -> Calibration:
    """No-click calibration: auto-detect regions + validate spool discs.

    Writes a calibration.json without any GUI - a good starting point that the
    pipeline uses immediately. Refine later with the interactive
    ``calibrate_session`` (click) if a disc still looks wrong (e.g. a top disc
    near the box edge)."""
    from region_signals import validated_spools
    regions = detect_regions_robust(video_path)
    spools = validated_spools(video_path, regions.motors)
    frame = representative_frame(video_path, 0.5)
    cal = Calibration(frame_w=frame.shape[1], frame_h=frame.shape[0],
                      regions=regions.to_dict(), colors={},
                      spools=[[s.cx, s.cy, s.radius] for s in spools],
                      notes=f"auto-calibrated from {Path(video_path).name}")
    cal.save(out_path)
    print(f"auto-calibration: {len(spools)} discs validated -> {out_path}")
    return cal


def default_calibration_path(video_path: str | Path) -> Path:
    """Where to save by default: the run folder's own ``calibration.json``.

    A pair video lives at ``<run>/pair_xxx/side_camera.mp4``, and the pipeline
    reads ``<run>/calibration.json``. So we save there by default - unique per
    run, no path to type. Falls back to the video's own folder if the layout
    differs."""
    v = Path(video_path).resolve()
    run_dir = v.parent.parent if v.parent.name.lower().startswith("pair") else v.parent
    return run_dir / "calibration.json"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Side-camera calibration (interactive or --auto).")
    ap.add_argument("video")
    ap.add_argument("--out", default=None,
                    help="output path (default: the run folder's calibration.json)")
    ap.add_argument("--frac", type=float, default=0.5, help="which frame (0..1) to calibrate on")
    ap.add_argument("--auto", action="store_true", help="no-click: auto-detect + validate discs")
    a = ap.parse_args()
    out = Path(a.out) if a.out else default_calibration_path(a.video)
    if a.auto:
        auto_calibrate(a.video, out)
    else:
        calibrate_session(a.video, out, a.frac)
