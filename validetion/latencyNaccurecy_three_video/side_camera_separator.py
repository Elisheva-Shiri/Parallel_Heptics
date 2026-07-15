"""Separate a *single* physical ``side_camera.mp4`` into its three functional
regions and (optionally) export each as its own cropped video.

WHY THIS EXISTS
---------------
In the latency/accuracy rig the side camera is framed so that one image
contains three different things at once:

    +-------------------------------+--------------------+
    |  VISION  (the experiment      |                    |
    |  monitor: black background,   |     empty white    |
    |  orange/blue object square,   |     desk           |
    |  green "progress" bar)        |                    |
    |                               |                    |
    +----------------+--------------+--------------------+
    |  MOTORS        |  TACTOR (green body, yellow       |
    |  (white box,   |  centre, sitting on the finger)   |
    |  ~3 spools     |                                   |
    |  each w/ a     |                                   |
    |  black line)   |                                   |
    +----------------+-----------------------------------+

So "separating the video" means finding three rectangular regions of interest
(ROIs) in the frame and treating each as an independent stream.  Because the
camera, the monitor and the motor box are physically fixed during a session,
the ROIs are (to good approximation) constant for the whole video, so we detect
them *once* on a representative frame and reuse them.

ALGORITHM CHOICES (and why)
---------------------------
* **VISION / monitor** - the monitor is by far the largest *dark* object in the
  scene.  We therefore threshold for dark pixels, morphologically clean the
  mask, and take the largest connected component's bounding box.  This is robust
  to the object/bar drawn *inside* the monitor (they are small relative to the
  black screen) and needs no hand-tuned coordinates, so it transfers to new
  videos with a differently placed monitor.

* **TACTOR** - the tactor is a strongly *saturated green* blob.  The monitor
  also renders green (progress bar / green cues), so we search for green
  **only outside the monitor ROI**.  The largest green blob there is the
  physical tactor.  We pad its bounding box so the ROI still contains the tactor
  across its full range of motion.

* **MOTORS / spools** - the spool box is a bright (white) structure in the
  bottom-left, to the left of the tactor and below the monitor.  Pure
  brightness is ambiguous (the whole desk is white), so we locate the box by the
  *circular spools* inside it using a Hough-circle search restricted to the
  bottom-left quadrant, then take the bounding box of the found circles (with
  padding).  If no circles are found we fall back to a sensible default ROI
  (bottom-left, below the monitor, left of the tactor).

Every automatic step can be overridden: pass an explicit ``regions.json`` (or a
``Regions`` object) and detection is skipped entirely.  A manual picker is also
provided for brand-new rigs.

Typical use::

    # detect + save overlay + dump regions.json (no video re-encode)
    python side_camera_separator.py path/to/side_camera.mp4

    # also export the three cropped mp4s
    python side_camera_separator.py path/to/side_camera.mp4 --export

    # reuse a previously saved layout on a whole new video
    python side_camera_separator.py new_video.mp4 --regions regions.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Region container
# --------------------------------------------------------------------------- #

# Region names are stable identifiers used by the rest of the pipeline.
VISION = "vision"
MOTORS = "motors"
TACTOR = "tactor"
REGION_NAMES = (VISION, MOTORS, TACTOR)


@dataclass
class Box:
    """An axis-aligned ROI in pixel coordinates (inclusive x0,y0; exclusive x1,y1)."""

    x0: int
    y0: int
    x1: int
    y1: int

    def clip(self, w: int, h: int) -> "Box":
        return Box(
            max(0, min(self.x0, w - 1)),
            max(0, min(self.y0, h - 1)),
            max(1, min(self.x1, w)),
            max(1, min(self.y1, h)),
        )

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y0:self.y1, self.x0:self.x1]

    def as_list(self) -> list[int]:
        return [int(self.x0), int(self.y0), int(self.x1), int(self.y1)]

    @classmethod
    def from_list(cls, v) -> "Box":
        return cls(int(v[0]), int(v[1]), int(v[2]), int(v[3]))


@dataclass
class Regions:
    """The three ROIs plus the frame size they were computed for."""

    frame_w: int
    frame_h: int
    vision: Box
    motors: Box
    tactor: Box

    def boxes(self) -> dict[str, Box]:
        return {VISION: self.vision, MOTORS: self.motors, TACTOR: self.tactor}

    # -- serialization ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "vision": self.vision.as_list(),
            "motors": self.motors.as_list(),
            "tactor": self.tactor.as_list(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Regions":
        return cls(
            frame_w=int(d["frame_w"]),
            frame_h=int(d["frame_h"]),
            vision=Box.from_list(d["vision"]),
            motors=Box.from_list(d["motors"]),
            tactor=Box.from_list(d["tactor"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Regions":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# HSV ranges (shared with vision/color/utils.py conventions)
# --------------------------------------------------------------------------- #
# Green of the physical tactor: a vivid plastic green.  Slightly wider than the
# on-screen green so it survives camera white-balance.
GREEN_LO = np.array([35, 60, 40], dtype=np.uint8)
GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)


def _largest_contour(mask: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_a = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= min_area and a > best_a:
            best, best_a = c, a
    return best


# --------------------------------------------------------------------------- #
# Automatic detection
# --------------------------------------------------------------------------- #

def detect_vision_roi(frame_bgr: np.ndarray) -> Optional[Box]:
    """Largest dark region = the experiment monitor (black background)."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # The screen is much darker than the lit white desk; an absolute threshold
    # around 70 separates them reliably without depending on the frame mean.
    _, dark = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    c = _largest_contour(dark, min_area=0.05 * w * h)
    if c is None:
        return None
    x, y, ww, hh = cv2.boundingRect(c)
    return Box(x, y, x + ww, y + hh).clip(w, h)


def detect_tactor_roi(
    frame_bgr: np.ndarray,
    exclude: Optional[Box] = None,
    travel_pad: int = 35,
) -> Optional[Box]:
    """Largest saturated-green blob outside the monitor = the physical tactor."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    if exclude is not None:
        # Blank out the monitor so its on-screen green never wins.
        mask[exclude.y0:exclude.y1, exclude.x0:exclude.x1] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    c = _largest_contour(mask, min_area=80)
    if c is None:
        return None
    x, y, ww, hh = cv2.boundingRect(c)
    # Pad so the ROI keeps the tactor as it moves with the finger.
    return Box(
        x - travel_pad, y - travel_pad, x + ww + travel_pad, y + hh + travel_pad
    ).clip(w, h)


def detect_motors_roi(
    frame_bgr: np.ndarray,
    vision: Optional[Box] = None,
    tactor: Optional[Box] = None,
) -> Optional[Box]:
    """Find the spool box from its cluster of circular spools.

    Framing-robust: instead of assuming the box is bottom-left, we look for
    circles anywhere in the lower band of the frame *except* the monitor and the
    tactor (which are masked out), then keep the densest cluster of circles -
    that compact group of 2-3 discs is the spool box wherever it sits.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    fill = int(gray.mean())
    work = gray.copy()
    work[:int(0.45 * h), :] = fill           # spools live in the lower part
    for b in (vision, tactor):               # never look inside these
        if b is not None:
            work[b.y0:b.y1, b.x0:b.x1] = fill

    circles = cv2.HoughCircles(
        work, cv2.HOUGH_GRADIENT, dp=1.2, minDist=14,
        param1=120, param2=20,
        minRadius=max(5, int(0.012 * w)), maxRadius=max(9, int(0.06 * w)),
    )
    if circles is None or not len(circles):
        return None
    cc = np.round(circles[0, :]).astype(int)
    # Keep the densest cluster: seed = circle with most neighbours within ~55 px,
    # then take every circle near that seed. Rejects stray circles (table edge...).
    if len(cc) > 1:
        cen = cc[:, :2].astype(float)
        d = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
        near = d < 55
        seed = int(near.sum(axis=1).argmax())
        keep = cc[near[seed]]
    else:
        keep = cc
    pad = 12
    x0 = int((keep[:, 0] - keep[:, 2]).min()) - pad
    y0 = int((keep[:, 1] - keep[:, 2]).min()) - pad
    x1 = int((keep[:, 0] + keep[:, 2]).max()) + pad
    y1 = int((keep[:, 1] + keep[:, 2]).max()) + pad
    return Box(x0, y0, x1, y1).clip(w, h)


def representative_frame(video_path: str | Path, at_fraction: float = 0.5) -> np.ndarray:
    """Grab one frame from the middle of the video (interaction is underway)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, min(n - 1, n * at_fraction))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")
    return frame


def detect_regions_robust(video_path: str | Path, n_samples: int = 5) -> Regions:
    """Detect ROIs by *median* over several frames - resilient to one bad frame.

    The monitor and motor box never move, so we take the median ROI corners
    across sampled frames.  The tactor moves, so we take the union (bounding box
    of all sampled tactor detections) to cover its travel.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fracs = np.linspace(0.2, 0.8, n_samples)
    visions, motors, tactors = [], [], []
    for f in fracs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
        ok, frame = cap.read()
        if not ok:
            continue
        v = detect_vision_roi(frame)
        if v is not None:
            visions.append(v.as_list())
        t = detect_tactor_roi(frame, exclude=v)
        if t is not None:
            tactors.append(t.as_list())
    cap.release()

    if visions:
        vision = Box.from_list(np.median(np.array(visions), axis=0).astype(int))
    else:
        vision = Box(0, 0, int(0.60 * w), int(0.63 * h))
    if tactors:
        arr = np.array(tactors)
        tactor = Box(arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max())
    else:
        tactor = Box(int(0.32 * w), int(0.78 * h), int(0.52 * w), h)

    # Motors: detect once on the middle frame relative to the (stable) vision/tactor.
    mid = representative_frame(video_path, 0.5)
    m = detect_motors_roi(mid, vision=vision, tactor=tactor)
    motors = m if m is not None else Box(0, int(0.85 * h), int(0.25 * w), h)
    return Regions(frame_w=w, frame_h=h, vision=vision, motors=motors, tactor=tactor)


# --------------------------------------------------------------------------- #
# Visualization + export
# --------------------------------------------------------------------------- #

_COLORS = {VISION: (0, 165, 255), MOTORS: (255, 80, 0), TACTOR: (0, 255, 0)}


def annotate(frame_bgr: np.ndarray, regions: Regions) -> np.ndarray:
    out = frame_bgr.copy()
    for name, box in regions.boxes().items():
        col = _COLORS[name]
        cv2.rectangle(out, (box.x0, box.y0), (box.x1, box.y1), col, 2)
        cv2.putText(out, name, (box.x0 + 4, max(16, box.y0 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return out


def export_region_videos(
    video_path: str | Path, regions: Regions, out_dir: str | Path
) -> dict[str, Path]:
    """Write one cropped mp4 per region.  Returns {region_name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers, paths = {}, {}
    for name, box in regions.boxes().items():
        p = out_dir / f"region_{name}.mp4"
        writers[name] = cv2.VideoWriter(str(p), fourcc, fps, (box.width, box.height))
        paths[name] = p
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for name, box in regions.boxes().items():
            writers[name].write(box.crop(frame))
    cap.release()
    for wr in writers.values():
        wr.release()
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Separate side_camera.mp4 into 3 regions.")
    ap.add_argument("video", help="path to side_camera.mp4")
    ap.add_argument("--regions", help="reuse an existing regions.json (skip detection)")
    ap.add_argument("--out", help="output dir (default: alongside the video)")
    ap.add_argument("--export", action="store_true", help="also write cropped region mp4s")
    args = ap.parse_args(argv)

    video = Path(args.video)
    out_dir = Path(args.out) if args.out else video.parent / "separated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.regions:
        regions = Regions.load(args.regions)
        print(f"Loaded regions from {args.regions}")
    else:
        regions = detect_regions_robust(video)
        print("Detected regions:", json.dumps(regions.to_dict()))

    regions.save(out_dir / "regions.json")
    overlay = annotate(representative_frame(video, 0.5), regions)
    cv2.imwrite(str(out_dir / "regions_overlay.png"), overlay)
    print(f"Wrote {out_dir/'regions.json'} and {out_dir/'regions_overlay.png'}")

    if args.export:
        paths = export_region_videos(video, regions, out_dir)
        print("Exported:", {k: str(v) for k, v in paths.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
