"""Extract per-frame signals from the three side-camera regions.

For every video frame we turn each region into a small number of scalar
time-series that the latency/accuracy notebook can correlate against
``tracking.csv`` and ``motor_commands.txt``:

VISION  (the monitor)
    * object centroid (orange OR blue square on black)  -> obj_x, obj_y (panel px)
    * which object colour was seen                      -> obj_color
    * green progress-bar fill                           -> bar_fill (0..1)

TACTOR  (green body, yellow centre, on the finger)
    * yellow-centre position (preferred) or green-body  -> tactor_x, tactor_y
    * whether it was detected                           -> tactor_found

MOTORS  (white box, up to 3 spools, each with a black line)
    * per-spool black-line angle (deg)                  -> spool0_angle, ...

ALGORITHM CHOICES
-----------------
* **Object on the monitor** - the screen background is black, so a coloured
  square is trivially segmented in HSV.  We test the orange and the blue range
  (the two object colours used by frontend_pygame.py: FIRST_COLOR=orange,
  SECOND_COLOR=blue) and keep the larger blob.  Centroid via image moments.
* **Progress bar** - the bar is the dominant *green* region in the lower part of
  the monitor.  We measure green pixels inside a horizontal band near the bottom
  of the panel and normalise by the band capacity to get a 0..1 fill.  Restricting
  to that band stops the (occasionally green-ish) object from contaminating it.
* **Tactor** - vivid green plastic with a yellow dot at the contact point.  The
  yellow dot is the most reliable single point (the green body is large and its
  centroid wanders as it rotates), so we prefer the yellow centroid and fall
  back to the green-body centroid when the dot is occluded.
* **Spool angle** - reuse SpoolAngleDetector from the existing
  motor_response_analizer_servo: dark-pixel PCA inside each spool circle.  This
  is the same, already-validated method used for the dedicated motor rig, so the
  video-derived "motor movement" is measured consistently with the rest of the
  codebase.

All detectors return NaN (not a guess) when their target is absent, and emit a
confidence/found flag so the notebook can filter low-quality frames.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from side_camera_separator import Box, Regions, detect_regions_robust

# Reuse the validated spool-angle detector from the motor-response rig.
_VA_DIR = Path(__file__).resolve().parent.parent / "motor_response_analizer_servo"
if str(_VA_DIR) not in sys.path:
    sys.path.insert(0, str(_VA_DIR))
from vision_angle import SpoolAngleDetector, SpoolROI  # noqa: E402

try:
    from scipy.ndimage import map_coordinates
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Spool groove angle by PROJECTION (Radon-style), not thresholded-pixel PCA
# --------------------------------------------------------------------------- #
GROOVE_INSET = 0.72       # sample diameters inside this fraction of the disc radius


def _sample_line(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY:
        return map_coordinates(gray, [ys, xs], order=1, mode="nearest")
    yi = np.clip(np.round(ys).astype(int), 0, gray.shape[0] - 1)
    xi = np.clip(np.round(xs).astype(int), 0, gray.shape[1] - 1)
    return gray[yi, xi]


def measure_groove(gray: np.ndarray, cx: float, cy: float, r: float,
                   inset: float = GROOVE_INSET, n_ang: int = 180,
                   min_contrast: float = 1.5):
    """Groove angle on a white disc = the DARKEST diameter through its centre.

    For each orientation we integrate intensity along the diameter through
    (cx, cy); the groove is the orientation with the lowest mean (a dark line).
    Using raw intensity (no threshold) is robust to the disc's shading and to the
    surrounding dark navy gap, which is what made thresholded-pixel PCA fail on
    these small low-contrast discs. Returns ``(angle_deg, contrast, direction)``
    where ``angle_deg`` is wrapped to (-90, 90], ``contrast`` is how much darker
    the groove is than a typical diameter (quality), and ``direction`` is the
    unit vector along the line (for drawing); angle is NaN if contrast is weak."""
    R = r * inset
    if R < 3:
        return np.nan, 0.0, None
    ts = np.linspace(-R, R, max(9, int(2 * R) + 1))
    cos = np.cos(np.deg2rad(np.arange(n_ang)))
    sin = np.sin(np.deg2rad(np.arange(n_ang)))
    prof = np.empty(n_ang, dtype=np.float64)
    for i in range(n_ang):
        prof[i] = _sample_line(gray, cx + ts * cos[i], cy + ts * sin[i]).mean()
    k = int(np.argmin(prof))
    a0, a1, a2 = prof[(k - 1) % n_ang], prof[k], prof[(k + 1) % n_ang]
    denom = a0 - 2 * a1 + a2
    delta = float(np.clip(0.5 * (a0 - a2) / denom, -0.5, 0.5)) if denom != 0 else 0.0
    ang = k + delta
    contrast = float(np.median(prof) - a1)
    if not np.isfinite(contrast) or contrast < min_contrast:
        return np.nan, contrast, None
    th = np.deg2rad(ang)
    wrapped = ((ang + 90.0) % 180.0) - 90.0
    return wrapped, contrast, (float(np.cos(th)), float(np.sin(th)))


# --- HSV ranges (OpenCV H in 0..179) --------------------------------------- #
# Object colours as drawn by frontend_pygame.py, widened for camera capture.
ORANGE_LO = np.array([8, 90, 90], dtype=np.uint8)
ORANGE_HI = np.array([24, 255, 255], dtype=np.uint8)
BLUE_LO = np.array([100, 80, 60], dtype=np.uint8)
BLUE_HI = np.array([135, 255, 255], dtype=np.uint8)
# Green / yellow of the tactor and the on-screen progress bar.
GREEN_LO = np.array([35, 60, 40], dtype=np.uint8)
GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)
YELLOW_LO = np.array([18, 90, 90], dtype=np.uint8)
YELLOW_HI = np.array([34, 255, 255], dtype=np.uint8)

# Default colour ranges. A calibration.json (see calibrate.py) can override any
# of these per session so the thresholds match the real camera/lighting.
DEFAULT_COLORS = {
    "orange": (ORANGE_LO, ORANGE_HI),
    "blue": (BLUE_LO, BLUE_HI),
    "green": (GREEN_LO, GREEN_HI),
    "yellow": (YELLOW_LO, YELLOW_HI),
}


def _resolve_colors(colors: Optional[dict]) -> dict:
    """Merge user/calibration colour overrides onto the defaults.

    Accepts values as (lo, hi) sequences (e.g. from calibration.json) and
    returns uint8 numpy arrays the detectors can pass to cv2.inRange.
    """
    out = dict(DEFAULT_COLORS)
    for key, val in (colors or {}).items():
        if val is None:
            continue
        lo, hi = val
        out[key] = (np.asarray(lo, dtype=np.uint8), np.asarray(hi, dtype=np.uint8))
    return out


def _centroid(mask: np.ndarray, min_area: int) -> Optional[tuple[float, float, int]]:
    """Largest blob centroid (x, y, area) in mask coords, or None."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_a = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= min_area and a > best_a:
            best, best_a = c, a
    if best is None:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"], int(best_a)


# --------------------------------------------------------------------------- #
# Per-region detectors
# --------------------------------------------------------------------------- #

def _compact_blob(mask: np.ndarray, w: int, h: int, min_area: int = 12, border: int = 8):
    """Largest *compact* blob (cx, cy, area), rejecting elongated/wide/edge regions.

    The object is a small square moving inside the black screen; UI strips
    (taskbar, progress bar) are wide and thin, and desktop/title-bar bleed sits
    in the panel corners. We therefore reject blobs that are elongated, span most
    of the panel width, or touch the panel border, so none can masquerade as the
    object.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_a = None, 0.0
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if a < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 4.0 or bw > 0.6 * w:        # elongated or screen-spanning -> not the object
            continue
        if bx <= border or by <= border or bx + bw >= w - border or by + bh >= h - border:
            continue                            # touches a panel edge -> desktop/UI bleed, not the object
        if a > best_a:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            best, best_a = (M["m10"] / M["m00"], M["m01"] / M["m00"], int(a)), a
    return best


def detect_object(vision_crop: np.ndarray, colors: Optional[dict] = None) -> dict:
    """Object square centroid on the black monitor (orange or blue)."""
    c = _resolve_colors(colors)
    hsv = cv2.cvtColor(vision_crop, cv2.COLOR_BGR2HSV)
    h, w = vision_crop.shape[:2]
    out = {"obj_x": np.nan, "obj_y": np.nan, "obj_color": None, "obj_area": 0}
    best = None
    for color in ("orange", "blue"):
        lo, hi = c[color]
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        res = _compact_blob(mask, w, h, min_area=12)
        if res is not None and (best is None or res[2] > best[3]):
            best = (res[0], res[1], color, res[2])
    if best is not None:
        out.update(obj_x=best[0], obj_y=best[1], obj_color=best[2], obj_area=best[3])
    return out


def detect_bar_fill(vision_crop: np.ndarray, band: tuple[float, float] = (0.72, 0.95),
                    colors: Optional[dict] = None) -> float:
    """Green progress-bar fill (0..1), measured in a horizontal band near the bottom.

    Returns the green-filled width fraction of the band's green-bearing rows.
    Normalisation to a true 0..1 is done later against the per-trial maximum.
    """
    green_lo, green_hi = _resolve_colors(colors)["green"]
    h, w = vision_crop.shape[:2]
    y0, y1 = int(band[0] * h), int(band[1] * h)
    strip = vision_crop[y0:y1, :]
    if strip.size == 0:
        return np.nan
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lo, green_hi)
    cols = mask.any(axis=0)            # which columns contain any green
    if not cols.any():
        return 0.0
    # Filled width = span from first to last green column (the bar is contiguous).
    xs = np.nonzero(cols)[0]
    filled = xs.max() - xs.min() + 1
    return float(filled) / float(w)


def detect_tactor(tactor_crop: np.ndarray, colors: Optional[dict] = None) -> dict:
    """Track the tactor using colour-difference channels (not HSV).

    On camera, HSV thresholds for the small yellow dot are unreliable. Two linear
    channels separate the parts cleanly and robustly to lighting:
      * green body  = 2G - R - B  (the vivid green plastic stands out from skin/desk)
      * yellow dot  = (R+G)/2 - B  (the yellow contact marker is a bright blob)
    We find the green body first, then take the brightest *yellow-channel* blob
    inside its (padded) bounding box - so a stray yellow pixel elsewhere can't win,
    and we fall back to the green-body centre when the dot is occluded. ``colors``
    is accepted for API compatibility but not needed here."""
    h, w = tactor_crop.shape[:2]
    b, g, r = cv2.split(tactor_crop.astype(np.int16))
    green = np.clip(2 * g - r - b, 0, 255).astype(np.uint8)
    yellow = np.clip((r + g) // 2 - b, 0, 255).astype(np.uint8)
    out = {"tactor_x": np.nan, "tactor_y": np.nan, "tactor_found": False, "tactor_src": None}

    gthr = max(30, int(green.mean() + green.std()))
    gmask = (green > gthr).astype(np.uint8)
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(gmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body = max(cnts, key=cv2.contourArea) if cnts else None
    if body is None or cv2.contourArea(body) < 60:
        return out  # no tactor in view

    bx, by, bw, bh = cv2.boundingRect(body)
    pad = max(4, bw // 5)
    gate = np.zeros((h, w), np.uint8)
    gate[max(0, by - pad):min(h, by + bh + pad), max(0, bx - pad):min(w, bx + bw + pad)] = 255

    # Brightest yellow-channel blob inside the green body = the contact marker.
    yv = yellow.copy()
    yv[gate == 0] = 0
    ythr = max(40, int(yv.max() * 0.6))
    yres = _centroid((yv >= ythr).astype(np.uint8) * 255, min_area=3)
    if yres is not None:
        out.update(tactor_x=yres[0], tactor_y=yres[1], tactor_found=True, tactor_src="yellow")
        return out

    mom = cv2.moments(body)
    out.update(tactor_x=mom["m10"] / mom["m00"], tactor_y=mom["m01"] / mom["m00"],
               tactor_found=True, tactor_src="green")
    return out


def validated_spools(video_path, motors: Box, max_spools: int = 3,
                     min_rate: float = 0.7, n_check: int = 25, search: int = 3) -> list[SpoolROI]:
    """Auto-propose disc ROIs and KEEP ONLY those that really carry a line.

    Starts from the stacked-disc guess, then for each candidate nudges the centre
    within +/-``search`` px to maximise how often a line is measured over
    ``n_check`` frames, and keeps the disc only if that best rate exceeds
    ``min_rate``. This rejects phantom discs (wall/edge) and self-corrects small
    placement errors, so it is a reliable no-click calibration where the imagery
    allows it (and the user can still override by clicking)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.5))
    ok, mid = cap.read()
    if not ok:
        cap.release()
        return []
    guesses = init_spools(mid, motors, max_spools=max_spools)
    frames = []
    for f in np.linspace(0.06, 0.94, n_check):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        return guesses

    grays = [cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) for fr in frames]
    mid_gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)

    def interior_brightness(roi: SpoolROI) -> float:
        msk = np.zeros(mid_gray.shape, np.uint8)
        cv2.circle(msk, (roi.cx, roi.cy), int(roi.radius * 0.7), 255, -1)
        px = mid_gray[msk > 0]
        return float(px.mean()) if px.size else 0.0

    kept: list[SpoolROI] = []
    for g in guesses:
        # Candidate grid; only positions sitting on a bright disc (not the dark
        # box edge / navy), then pick the centre with the strongest groove
        # *contrast* over time - a real disc has a clear dark groove every frame.
        grid = [SpoolROI(g.cx + dx, g.cy + dy, g.radius)
                for dx in range(-search, search + 1) for dy in range(-search, search + 1)]
        bright = {id(r): interior_brightness(r) for r in grid}
        bmax = max(bright.values()) if bright else 0.0
        best, best_score = g, -1.0
        for roi in grid:
            if bmax > 0 and bright[id(roi)] < 0.9 * bmax:
                continue  # off the disc -> skip
            cons = [measure_groove(gf, roi.cx, roi.cy, roi.radius)[1] for gf in grays]
            score = float(np.median(cons))
            if score > best_score:
                best, best_score = roi, score
        if best_score >= 2.5:   # median groove contrast (intensity units) of a real disc
            kept.append(best)
    return kept


def init_spools(frame_bgr: np.ndarray, motors: Box, max_spools: int = 3) -> list[SpoolROI]:
    """Locate the ``max_spools`` spools as a vertical stack of equal discs.

    The three motors are bright white discs stacked vertically with dark gaps
    between them, each carrying a dark groove-line. Generic HoughCircles gives
    inconsistent radii (and stray huge circles), so we instead exploit that known
    geometry: find the bright column (the spool x-centre and width), find the
    vertical extent of the bright stack, then place ``max_spools`` equal,
    evenly-spaced discs down that column. This yields consistent, non-overlapping
    ROIs that the line detector can measure reliably.
    """
    crop = motors.crop(frame_bgr)
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    ch, cw = gray.shape

    # 1) spool x-column = the band with the most *structure*. The discs+grooves+
    #    gaps make those columns high-variance, while a bright but flat wall in
    #    the ROI is low-variance -> variance localises the spools, brightness
    #    alone does not.
    col_std = gray.std(axis=0)
    peak = int(np.argmax(col_std))
    strong = col_std > 0.5 * float(col_std.max())
    x_lo = peak
    while x_lo > 0 and strong[x_lo - 1]:
        x_lo -= 1
    x_hi = peak
    while x_hi < cw - 1 and strong[x_hi + 1]:
        x_hi += 1
    cx = (x_lo + x_hi) // 2
    half_w = max(4, (x_hi - x_lo) // 2)

    # 2) vertical extent of the bright stack within that column
    band = gray[:, max(0, cx - half_w):min(cw, cx + half_w + 1)]
    row = band.mean(axis=1)
    thr_r = 0.5 * (float(row.max()) + float(row.min()))
    bright_r = np.where(row > thr_r)[0]
    y_lo, y_hi = (int(bright_r.min()), int(bright_r.max())) if bright_r.size else (0, ch - 1)
    stack_h = max(max_spools, y_hi - y_lo)

    # 3) place equal, non-overlapping disc candidates
    r = int(max(5, min(half_w * 0.95, stack_h / (2 * max_spools) * 0.95)))
    cand = []
    for k in range(max_spools):
        cy = y_lo + int((k + 0.5) * stack_h / max_spools)
        cand.append((cx, cy))

    # 4) validate: a real spool disc has high *interior* variance (white disc +
    #    dark groove), measured inside a circular mask so the box edge / corners
    #    don't leak in. A candidate on flat wall scores ~5x lower and is dropped,
    #    so when fewer than max_spools are actually in frame we keep only the real
    #    ones (e.g. some sessions show 2 spools, not 3).
    struct = []
    for (ccx, ccy) in cand:
        mask = np.zeros((ch, cw), np.uint8)
        cv2.circle(mask, (ccx, ccy), int(r * 0.85), 255, -1)
        px = gray[mask > 0]
        struct.append(float(px.std()) if px.size else 0.0)
    smax = max(struct) if struct else 0.0
    rois: list[SpoolROI] = []
    for (ccx, ccy), s in zip(cand, struct):
        if smax > 0 and s >= 0.4 * smax and s >= 15.0:
            rois.append(SpoolROI(cx=ccx + motors.x0, cy=ccy + motors.y0, radius=r))
    return rois


# --------------------------------------------------------------------------- #
# Full-video extraction
# --------------------------------------------------------------------------- #

@dataclass
class ExtractionResult:
    df: pd.DataFrame
    regions: Regions
    spools: list[SpoolROI]


def extract_signals(
    video_path: str | Path,
    regions: Optional[Regions] = None,
    tracking_csv: Optional[str | Path] = None,
    stride: int = 1,
    max_frames: Optional[int] = None,
    spools: Optional[list[SpoolROI]] = None,
    colors: Optional[dict] = None,
) -> ExtractionResult:
    """Run all three detectors over the video and return a tidy DataFrame.

    ``stride`` subsamples frames (e.g. 2 = every other frame) for speed on the
    very long pair_001 video.  ``tracking_csv`` attaches the absolute timestamp
    of each frame (frame i <-> tracking row i, 1:1 in this rig).  Pass
    ``regions``/``spools`` from a single detection to reuse the *same* layout
    across every pair of one session (the rig does not move between pairs).
    ``colors`` (from a calibration.json) overrides the default HSV ranges.
    """
    video_path = Path(video_path)
    if regions is None:
        regions = detect_regions_robust(video_path)

    track_ts = None
    if tracking_csv is not None:
        tdf = pd.read_csv(tracking_csv)
        track_ts = pd.to_datetime(tdf["timestamp"]).to_numpy()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Locate spools on a mid-video frame unless the caller supplied them.
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if spools is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.5))
        ok, mid = cap.read()
        spools = init_spools(mid, regions.motors) if ok else []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            rec = {"frame": idx, "time_s": idx / fps}
            if track_ts is not None and idx < len(track_ts):
                rec["timestamp"] = track_ts[idx]
            rec.update(detect_object(regions.vision.crop(frame), colors=colors))
            rec["bar_fill"] = detect_bar_fill(regions.vision.crop(frame), colors=colors)
            rec.update(detect_tactor(regions.tactor.crop(frame), colors=colors))
            # Groove angle per spool via projection on the grayscale frame.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for si, roi in enumerate(spools):
                ang, con, _ = measure_groove(gray, roi.cx, roi.cy, roi.radius)
                rec[f"spool{si}_angle"] = ang
                rec[f"spool{si}_conf"] = con
            rows.append(rec)
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break
    cap.release()

    df = pd.DataFrame(rows)
    # Continuously unwrap each spool's wrapped (-90,90] angle.
    for si in range(len(spools)):
        col = f"spool{si}_angle"
        if col in df.columns:
            df[f"{col}_unwrapped"] = SpoolAngleDetector.unwrap_series(df[col].tolist())
    return ExtractionResult(df=df, regions=regions, spools=spools)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--tracking")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out", default="signals.csv")
    a = ap.parse_args()
    res = extract_signals(a.video, tracking_csv=a.tracking, stride=a.stride, max_frames=a.max_frames)
    res.df.to_csv(a.out, index=False)
    print(f"frames={len(res.df)} spools={len(res.spools)}")
    print("detection rates:")
    print("  object:", float(res.df['obj_x'].notna().mean()))
    print("  tactor:", float(res.df['tactor_found'].mean()))
    for si in range(len(res.spools)):
        print(f"  spool{si}:", float(res.df[f'spool{si}_angle'].notna().mean()))
    print(f"wrote {a.out}")
