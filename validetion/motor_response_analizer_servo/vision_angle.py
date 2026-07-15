"""Detect the angle of a black line on a white motor spool.

Geometry assumed in the camera frame:
    (white background)
        +---------------------------------+
        |                                 |
        |       (darker grey spot)        |   <- the "darker spot" the user described
        |     +---------------------+     |
        |     |   white spool       |     |
        |     |   /                 |     |
        |     |  / black line       |     |   <- this is what we measure
        |     |                     |     |
        |     +---------------------+     |
        |                                 |
        +---------------------------------+

The pipeline is:

1. **Locate the spool ROI** (a circle).
   * `auto`: threshold for the *darker* region (grey spot), find its largest
     contour, then look for the brightest big circle inside it -> that is the
     white spool.
   * `manual`: the user clicks the spool centre and drags to set its radius.
   * `both`: try `auto` first, fall back to `manual` if detection fails.

2. **Find the black line** inside the spool ROI by thresholding for dark
   pixels (the spool is white, the line is black).

3. **Estimate the line orientation** with PCA (`cv2.PCACompute2`) on the
   coordinates of the dark pixels.  PCA gives us the principal direction
   even when the line passes through the spool centre (in which case
   `cv2.fitLine` would also work but PCA is simpler and gives an explicit
   variance ratio we can use as a quality metric).

4. **Convert orientation -> angle** in degrees, defined so a horizontal line
   = 0 deg, +90 = vertical.  The angle is wrapped to (-90, 90] (an undirected
   line has a 180-deg ambiguity).  Per-frame we compute an *unwrapped* angle
   relative to a calibration zero set on the first capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SpoolROI:
    """Circular region-of-interest covering the white motor spool."""

    cx: int
    cy: int
    radius: int

    def mask(self, frame_shape: tuple[int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(m, (self.cx, self.cy), self.radius, 255, -1)
        return m

    def crop_box(self, frame_shape: tuple[int, int], pad: int = 0) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x0 = max(0, self.cx - self.radius - pad)
        y0 = max(0, self.cy - self.radius - pad)
        x1 = min(w, self.cx + self.radius + pad)
        y1 = min(h, self.cy + self.radius + pad)
        return x0, y0, x1, y1


@dataclass
class AngleMeasurement:
    """Result of running the angle detector on a single frame."""

    angle_deg: Optional[float]            # wrapped angle in (-90, 90] (None on failure)
    confidence: float                     # PCA variance ratio (1st / total), 0..1
    line_pixel_count: int
    centroid: Optional[tuple[float, float]]
    direction: Optional[tuple[float, float]]   # unit vector along the line


# ---------------------------------------------------------------------------
# Spool ROI detection
# ---------------------------------------------------------------------------

def detect_dark_spot(frame_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Return the bounding box (x0, y0, x1, y1) of the darker (grey) spot.

    Logic: the background is "different shades of white" and the dark spot is
    noticeably darker (but lighter than the black line). We threshold the
    grayscale frame at a value below the global mean and pick the largest
    connected component that is not too small and not the full frame.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    h, w = blur.shape
    total = h * w

    # Threshold for "darker than typical background". Use a permissive value:
    # background is mostly bright (~200+), the grey spot is meaningfully darker.
    thresh_value = max(60, int(blur.mean() * 0.85))
    _, mask = cv2.threshold(blur, thresh_value, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < total * 0.005 or area > total * 0.9:
            continue
        if area > best_area:
            best_area = area
            best = c
    if best is None:
        return None
    x, y, ww, hh = cv2.boundingRect(best)
    pad = max(5, int(0.05 * min(ww, hh)))
    return (
        max(0, x - pad),
        max(0, y - pad),
        min(w, x + ww + pad),
        min(h, y + hh + pad),
    )


def detect_white_spool(
    frame_bgr: np.ndarray,
    search_box: Optional[tuple[int, int, int, int]] = None,
) -> Optional[SpoolROI]:
    """Find the white spool: a bright circular region inside the dark spot.

    Returns the spool's centre and radius in *frame* coordinates (not the
    cropped search box).
    """
    h, w = frame_bgr.shape[:2]
    if search_box is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = search_box

    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    crop_h, crop_w = gray.shape
    min_radius = max(15, int(0.10 * min(crop_h, crop_w)))
    max_radius = max(min_radius + 5, int(0.55 * min(crop_h, crop_w)))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(crop_h, crop_w),  # we only want the single best circle
        param1=120,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None or len(circles) == 0:
        return None

    circles = np.round(circles[0, :]).astype(int)
    # Pick the brightest circle (the spool is bright white).
    best = None
    best_score = -np.inf
    for (cx, cy, r) in circles:
        if cx - r < 0 or cy - r < 0 or cx + r >= crop_w or cy + r >= crop_h:
            continue
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), int(r * 0.85), 255, -1)
        mean_brightness = float(cv2.mean(gray, mask)[0])
        score = mean_brightness + 0.1 * r  # prefer bright AND large
        if score > best_score:
            best_score = score
            best = (cx, cy, r)
    if best is None:
        return None
    cx, cy, r = best
    return SpoolROI(cx=cx + x0, cy=cy + y0, radius=int(r))


def manual_pick_spool(frame_bgr: np.ndarray, window_name: str = "Pick spool") -> SpoolROI:
    """Let the user click the spool centre and drag to set the radius.

    Click & release: centre = click point, radius = distance to release point.
    Press ``r`` to redo, ``Enter`` / ``Space`` to accept, ``Esc`` to cancel.
    """
    state = {"centre": None, "radius": None, "drawing": False, "preview": frame_bgr.copy()}

    def redraw():
        img = frame_bgr.copy()
        if state["centre"] is not None and state["radius"] is not None and state["radius"] > 0:
            cv2.circle(img, state["centre"], state["radius"], (0, 255, 0), 2)
            cv2.circle(img, state["centre"], 3, (0, 0, 255), -1)
        cv2.putText(
            img, "Click centre, drag to radius. r=redo, Enter=ok, Esc=cancel",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.imshow(window_name, img)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["centre"] = (x, y)
            state["radius"] = 0
            state["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            cx, cy = state["centre"]
            state["radius"] = int(math.hypot(x - cx, y - cy))
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            cx, cy = state["centre"]
            state["radius"] = int(math.hypot(x - cx, y - cy))
            state["drawing"] = False
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Spool ROI selection cancelled by user")
        if key in (13, 10, 32):  # Enter / Space
            if state["centre"] is not None and state["radius"] is not None and state["radius"] > 5:
                cv2.destroyWindow(window_name)
                cx, cy = state["centre"]
                return SpoolROI(cx=cx, cy=cy, radius=int(state["radius"]))
        if key in (ord("r"), ord("R")):
            state["centre"] = None
            state["radius"] = None
            state["drawing"] = False
            redraw()


def find_spool_roi(
    frame_bgr: np.ndarray,
    mode: str = "both",
) -> SpoolROI:
    """Top-level dispatcher that returns a :class:`SpoolROI`.

    ``mode``:
        * ``"auto"``       – auto-detect, raise on failure.
        * ``"manual"``     – always ask the user to click.
        * ``"both"``       – try auto first, fall back to manual.
    """
    if mode in ("auto", "both"):
        box = detect_dark_spot(frame_bgr)
        roi = detect_white_spool(frame_bgr, search_box=box)
        if roi is not None:
            return roi
        if roi is None and mode == "auto":
            raise RuntimeError("Auto-detection of the spool failed")
    return manual_pick_spool(frame_bgr)


# ---------------------------------------------------------------------------
# Black-line angle measurement
# ---------------------------------------------------------------------------

class SpoolAngleDetector:
    """Measure the orientation of the black line on a white spool.

    The detector works in *cropped* spool space (radius ~ R) and computes the
    line angle by PCA of the dark pixels.  Because a line has a 180-deg
    ambiguity, the returned angle is wrapped to ``(-90, 90]``.  An *unwrapped*
    angle (continuous across the +/-90 deg discontinuity) is provided by
    :meth:`unwrap_series`.
    """

    def __init__(
        self,
        roi: SpoolROI,
        line_threshold: Optional[int] = None,   # None -> auto from spool brightness
        radial_inset: float = 0.92,             # ignore the outer rim
        min_pixels: int = 30,
        morph_kernel: int = 3,
        background_drop: float = 0.55,          # line is < background * this fraction
        min_contrast: int = 25,                 # AND at least this much darker (gray)
    ):
        self.roi = roi
        self.line_threshold = line_threshold
        self.radial_inset = radial_inset
        self.min_pixels = min_pixels
        self.morph_kernel = morph_kernel
        self.background_drop = background_drop
        self.min_contrast = min_contrast

    # ------------------------------------------------------------------
    # Frame-level measurement
    # ------------------------------------------------------------------

    def measure(self, frame_bgr: np.ndarray) -> AngleMeasurement:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy, r = self.roi.cx, self.roi.cy, self.roi.radius

        # Restrict to the spool interior.
        spool_mask = np.zeros_like(gray)
        cv2.circle(spool_mask, (cx, cy), int(r * self.radial_inset), 255, -1)

        # Threshold for the black line. The spool is ~white and the line takes
        # only a small fraction of the pixels, so the 75th-percentile of the
        # spool interior is a robust estimate of the bright background.  Any
        # pixel that is meaningfully darker than that is treated as line.
        if self.line_threshold is None:
            spool_pixels = gray[spool_mask > 0]
            if spool_pixels.size == 0:
                return AngleMeasurement(None, 0.0, 0, None, None)
            background = float(np.percentile(spool_pixels, 75))
            thr = min(background * self.background_drop, background - self.min_contrast)
        else:
            thr = float(self.line_threshold)
        line_mask = ((gray < thr).astype(np.uint8) * 255) & spool_mask

        if self.morph_kernel > 0:
            k = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, k)

        ys, xs = np.nonzero(line_mask)
        if xs.size < self.min_pixels:
            return AngleMeasurement(None, 0.0, int(xs.size), None, None)

        # PCA on (x, y) coordinates with origin at the spool centre.
        pts = np.stack([xs.astype(np.float32) - cx, ys.astype(np.float32) - cy], axis=1)
        mean = pts.mean(axis=0)
        centred = pts - mean
        cov = np.cov(centred, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
        # Largest-eigenvalue eigenvector is the line direction.
        direction = eigvecs[:, -1]
        dx, dy = float(direction[0]), float(direction[1])

        # Image y-axis points downward; we want angle measured in the standard
        # math frame so a horizontal line is 0 deg and +ccw is positive.
        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)
        # Wrap to (-90, 90]  (line is undirected).
        while angle_deg <= -90.0:
            angle_deg += 180.0
        while angle_deg > 90.0:
            angle_deg -= 180.0

        total_var = float(eigvals.sum())
        confidence = float(eigvals[-1] / total_var) if total_var > 0 else 0.0

        return AngleMeasurement(
            angle_deg=angle_deg,
            confidence=confidence,
            line_pixel_count=int(xs.size),
            centroid=(float(mean[0] + cx), float(mean[1] + cy)),
            direction=(dx, dy),
        )

    # ------------------------------------------------------------------
    # Series helpers
    # ------------------------------------------------------------------

    @staticmethod
    def unwrap_series(angles_deg: list[Optional[float]]) -> list[Optional[float]]:
        """Continuously unwrap a list of wrapped angles in (-90, 90].

        ``None`` entries are passed through unchanged. The first non-None entry
        is treated as the reference; subsequent jumps larger than 90 deg are
        compensated by adding/subtracting 180 deg (the line ambiguity).
        """
        out: list[Optional[float]] = []
        offset = 0.0
        prev: Optional[float] = None
        for a in angles_deg:
            if a is None:
                out.append(None)
                continue
            adjusted = a + offset
            if prev is not None:
                while adjusted - prev > 90.0:
                    adjusted -= 180.0
                    offset -= 180.0
                while prev - adjusted > 90.0:
                    adjusted += 180.0
                    offset += 180.0
            out.append(adjusted)
            prev = adjusted
        return out

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def annotate(self, frame_bgr: np.ndarray, m: AngleMeasurement) -> np.ndarray:
        out = frame_bgr.copy()
        cx, cy, r = self.roi.cx, self.roi.cy, self.roi.radius
        cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
        if m.angle_deg is not None and m.direction is not None and m.centroid is not None:
            dx, dy = m.direction
            ex, ey = m.centroid
            length = float(r) * 0.95
            x0 = int(ex - dx * length)
            y0 = int(ey - dy * length)
            x1 = int(ex + dx * length)
            y1 = int(ey + dy * length)
            cv2.line(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(
                out,
                f"angle={m.angle_deg:+.2f} deg  conf={m.confidence:.2f}",
                (max(10, cx - r), max(20, cy - r - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
            )
        return out
