"""Three-video validation pipeline.

This is the new, additive validation entry point for the requested data
architecture.  It intentionally lives in ``latencyNaccurecy_three_video`` so the
old ``latencyNaccurecy`` implementation remains untouched.

Streams per pair:
* top_camera.mp4       -> real hand movement, re-tracked offline with MediaPipe
* virtual_object.mp4   -> virtual object / virtual finger response
* side_camera.mp4      -> tactor yellow marker and motor spool angles
* tracking.csv         -> realtime/digital-twin log captured during experiment

All spatial summary errors are reported in millimetres when a scale can be
automatically estimated.  Motor-command comparison is explicitly marked as
INFERRED because motor command logs are unavailable in the new example data.
"""

from __future__ import annotations

import argparse
import enum
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import latency_analysis as LA
from consts import EDGE_THRESHOLD, STIFFNESS_MAX, TOP_HEIGHT, TOP_WIDTH
from haptic_mapping import map_object_displacement_to_tactor
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum
from motor_controller import HandOrientation, MotorController, MotorSetId, MovementStrategy
try:
    from region_signals import measure_groove, init_spools
    from side_camera_separator import Box
except Exception:  # keep the top/virtual paths usable even if old helpers change
    measure_groove = None
    init_spools = None
    Box = None

FINGER_LANDMARK = {
    "thumb": mp.solutions.hands.HandLandmark.THUMB_TIP,
    "index": mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    "middle": mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
    "ring": mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
    "pinky": mp.solutions.hands.HandLandmark.PINKY_TIP,
}

KNOWN_RED_DOT_DIST_MM = np.array([0.0, 10.0, 30.0, 60.0, 110.0, 170.0])
MARKER_DOT_DIAMETER_MM = 5.0  # user: red/yellow dots are 0.5 cm
YELLOW_DOT_DIAMETER_MM = MARKER_DOT_DIAMETER_MM


@dataclass
class PairResult:
    pair: str
    finger: str
    tracking_movement_start_s: float = np.nan
    top_movement_start_s: float = np.nan
    virtual_movement_start_s: float = np.nan
    side_movement_start_s: float = np.nan
    motor_movement_start_s: float = np.nan
    virtual_alignment_source: str = ""
    side_alignment_source: str = ""
    top_detection_rate: float = np.nan
    top_hand_r2: float = np.nan
    top_hand_rmse_mm: float = np.nan
    top_hand_latency_ms: float = np.nan
    top_hand_latency_corr: float = np.nan
    top_px_per_mm: float = np.nan
    virtual_object_detection_rate: float = np.nan
    virtual_object_r2: float = np.nan
    virtual_object_rmse_mm: float = np.nan
    virtual_object_latency_ms: float = np.nan
    virtual_object_latency_corr: float = np.nan
    finger_to_virtual_object_latency_ms: float = np.nan
    finger_to_virtual_object_latency_corr: float = np.nan
    virtual_finger_detection_rate: float = np.nan
    tactor_detection_rate: float = np.nan
    tactor_r2: float = np.nan
    tactor_rmse_mm: float = np.nan
    tactor_latency_ms: float = np.nan
    tactor_latency_corr: float = np.nan
    side_px_per_mm: float = np.nan
    motor_spool_detection_rate: float = np.nan
    backend_command_to_motor_latency_ms: float = np.nan
    backend_command_to_motor_latency_corr: float = np.nan
    backend_tactor_target_to_tactor_latency_ms: float = np.nan
    backend_tactor_target_to_tactor_latency_corr: float = np.nan
    backend_tactor_target_to_tactor_r2: float = np.nan
    backend_tactor_target_to_tactor_rmse_mm: float = np.nan
    inferred_command_to_motor_latency_ms: float = np.nan
    inferred_command_to_motor_latency_corr: float = np.nan
    motor_to_tactor_latency_ms: float = np.nan
    motor_to_tactor_latency_corr: float = np.nan
    inferred_command: bool = True
    motor_command_source: str = "reconstructed_from_tracking_via_backend_motor_controller"
    warnings: str = ""


@dataclass
class SpoolROI:
    cx: float
    cy: float
    r: float


def video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opened": False, "fps": np.nan, "frames": 0, "width": 0, "height": 0}
    meta = {
        "opened": True,
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return meta


def discover_pairs(session: Path, ignore_pair_ge: int = 5) -> list[Path]:
    pairs = []
    for p in sorted(session.glob("pair_*")):
        if not p.is_dir():
            continue
        try:
            n = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if n >= ignore_pair_ge:
            continue
        pairs.append(p)
    return pairs


def read_tracking(pair: Path) -> pd.DataFrame:
    df = pd.read_csv(pair / "tracking.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    t0 = df["timestamp"].iloc[0]
    df["t"] = (df["timestamp"] - t0).dt.total_seconds()
    return df


def parse_motor_set_id(session_or_pair: Path) -> MotorSetId:
    """Infer backend motor-set id from names like ``...motor_set_0_144``."""
    text = " ".join([session_or_pair.name, session_or_pair.parent.name])
    m = re.search(r"motor_set_(\d+)", text)
    if not m:
        return MotorSetId.MOTORS_0_2
    try:
        return MotorSetId(int(m.group(1)))
    except Exception:
        return MotorSetId.MOTORS_0_2


def reconstruct_backend_motor_commands(tracking: pd.DataFrame, pair: Path, out_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """Replay ``tracking.csv`` through the same backend motor-command path.

    This is not a measured firmware log.  It is a deterministic reconstruction
    of what ``backend.py`` would command from the recorded virtual object state:
    object displacement from screen center -> haptic tactor target -> IK
    ``MotorController.calculate_motor_movements``.  It lets us compare the
    intended command signal to observed motor black-line angles and yellow
    tactor motion even when UDP/firmware motor logs were not recorded.
    """
    warnings: list[str] = []
    if tracking.empty:
        return pd.DataFrame(), ["backend motor command reconstruction skipped: empty tracking.csv"]

    motor_set = parse_motor_set_id(pair)
    controller = MotorController(
        movement_strategy=MovementStrategy.IK,
        top_width=TOP_WIDTH,
        top_height=TOP_HEIGHT,
        edge_threshold=EDGE_THRESHOLD,
        move_factor=7,
        hand_orientation=HandOrientation.NOT_MIRRORED,
    )
    center_x = TOP_WIDTH / 2.0
    center_y = TOP_HEIGHT / 2.0
    rows: list[dict] = []
    motors_are_displaced = False
    previous_obj: tuple[float, float] | None = None

    for _, row in tracking.iterrows():
        t = float(row.get("t", np.nan))
        ts = row.get("timestamp")
        interacting = bool(row.get("interacting", False))
        stiffness_raw = float(row.get("stiffness", STIFFNESS_MAX))
        stiffness_norm = min(stiffness_raw, STIFFNESS_MAX) / STIFFNESS_MAX
        obj_x = float(row.get("object_x", center_x)) - center_x
        obj_y = float(row.get("object_y", center_y)) - center_y
        target_x, target_y = map_object_displacement_to_tactor(obj_x=obj_x, obj_y=obj_y, oppose_motion=True)

        cmd = {
            "time_s": t,
            "timestamp": ts,
            "interacting": interacting,
            "stiffness": stiffness_raw,
            "stiffness_norm": stiffness_norm,
            "object_dx": obj_x,
            "object_dy": obj_y,
            "backend_tactor_target_x": target_x,
            "backend_tactor_target_y": target_y,
            "motor_set_base_index": motor_set.base_index,
            "backend_message": "",
            "backend_command_sent": False,
            "backend_reset_command": False,
        }
        for i in range(3):
            cmd[f"backend_motor{i}_pos"] = 0

        try:
            if not interacting:
                if motors_are_displaced:
                    motors = controller.zero_motor_positions(motor_set)
                    cmd["backend_reset_command"] = True
                else:
                    motors = []
            else:
                # Backend skips repeated object positions at the motor loop.
                # For analysis we still write held target positions every row;
                # backend_command_sent tells whether a new firmware message
                # would have been emitted for this sample.
                obj_key = (obj_x, obj_y)
                motors = controller.calculate_motor_movements(
                    motor_set_id=motor_set,
                    stiffness_value=stiffness_norm,
                    obj_x=target_x,
                    obj_y=target_y,
                    motors_enabled=True,
                    reset_to_origin=False,
                )
                if previous_obj == obj_key:
                    cmd["backend_command_sent"] = False
                else:
                    cmd["backend_command_sent"] = bool(motors)
                previous_obj = obj_key

            if motors:
                cmd["backend_message"] = controller.build_message(motors)
                for motor in motors:
                    local = motor.index - motor_set.base_index
                    if 0 <= local < 3:
                        cmd[f"backend_motor{local}_pos"] = motor.pos
                motors_are_displaced = interacting and any(m.pos != 0 for m in motors)
            elif not interacting:
                previous_obj = None
                motors_are_displaced = False
        except Exception as e:
            warnings.append(f"backend motor command reconstruction failed at t={t:.3f}s: {e}")

        vals = np.array([cmd[f"backend_motor{i}_pos"] for i in range(3)], dtype=float)
        cmd["backend_motor_command_magnitude"] = float(np.linalg.norm(vals))
        cmd["backend_motor_command_mean"] = float(np.mean(vals))
        rows.append(cmd)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df, list(dict.fromkeys(warnings))


def frame_timestamp(tracking: pd.DataFrame, frame_idx: int, fps: float):
    if frame_idx < len(tracking):
        return tracking["timestamp"].iloc[frame_idx], float(tracking["t"].iloc[frame_idx])
    t = frame_idx / fps
    return tracking["timestamp"].iloc[0] + pd.Timedelta(seconds=t), t


def write_clip(src: Path, dst: Path, start_frame: int, frame_count: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
    written = 0
    while written < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        written += 1
    out.release()
    cap.release()


def split_screen_recording(session: Path, screen_recording: Path | None, pairs: list[Path], rebuild: bool) -> list[str]:
    """Create pair_###/virtual_object.mp4 clips.

    The preferred signal for pair boundaries is the existing per-pair recording
    duration; the comparison screen/end-screen time is treated as the gap between
    those durations.  This is automatic and deterministic, and the run_info file
    records that the segmentation is duration-aligned rather than manually cut.
    """
    warnings = []
    if screen_recording is None:
        warnings.append("No screen recording supplied; virtual_object.mp4 clips were not generated.")
        return warnings
    if not screen_recording.exists():
        warnings.append(f"Screen recording not found: {screen_recording}")
        return warnings
    meta = video_meta(screen_recording)
    if not meta["opened"]:
        warnings.append(f"Could not open screen recording: {screen_recording}")
        return warnings

    durations = []
    for pair in pairs:
        trk = read_tracking(pair)
        n = len(trk)
        top = video_meta(pair / "top_camera.mp4")
        if top["opened"] and top["frames"] > 0:
            n = min(n, top["frames"])
        durations.append(int(n))

    total_needed = sum(durations)
    extra = max(0, meta["frames"] - total_needed)
    gap = int(round(extra / (len(pairs) + 1))) if pairs else 0
    cursor = gap
    for pair, n in zip(pairs, durations):
        dst = pair / "virtual_object.mp4"
        if dst.exists() and not rebuild:
            cursor += n + gap
            continue
        write_clip(screen_recording, dst, cursor, n)
        cursor += n + gap
    warnings.append(
        "virtual_object.mp4 clips generated automatically by duration alignment; "
        "comparison/end screens are treated as inter-pair gaps. Inspect QC plots if exact screen-cut timing matters."
    )
    return warnings


def detect_red_centers(frame: np.ndarray) -> list[tuple[float, float, float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(frame)
    chroma = r.astype(np.int16) - np.maximum(g, b).astype(np.int16)
    mask1 = cv2.inRange(hsv, (0, 35, 35), (14, 255, 255))
    mask2 = cv2.inRange(hsv, (165, 35, 35), (179, 255, 255))
    score_mask = ((chroma > 18) & (r > 55)).astype("uint8") * 255
    mask = mask1 | mask2 | score_mask
    h, w = mask.shape
    # Calibration dots are printed on the board; reject border/reflection areas
    # that previously dominated the detector.
    keep = np.zeros_like(mask)
    keep[int(h * 0.03) : int(h * 0.88), int(w * 0.05) : int(w * 0.95)] = 255
    mask = cv2.bitwise_and(mask, keep)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 3 or area > 180:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if 1.0 <= r <= 10.0:
            centers.append((float(x), float(y), float(r)))
    return centers


def detect_small_calibration_blobs(frame: np.ndarray) -> np.ndarray:
    """Detect small printed calibration dots on the top-camera board.

    The red dots are sometimes desaturated by the camera, so hue-only detection
    can miss them.  This fallback uses a small-blob detector on grayscale and
    rejects border/large components; it is used only for scale estimation.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 2
    params.maxArea = 90
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False
    params.minThreshold = 70
    params.maxThreshold = 245
    params.thresholdStep = 5
    kps = cv2.SimpleBlobDetector_create(params).detect(gray)
    h, w = gray.shape
    pts = []
    for k in kps:
        x, y = k.pt
        if 8 <= x <= w - 8 and 8 <= y <= h - 8:
            pts.append((x, y))
    return np.asarray(pts, float)


def top_px_per_mm_from_calibration_dot_field(frames: list[np.ndarray]) -> float:
    """Estimate px/mm from the first-frame calibration-dot field.

    The top-camera board contains calibration dots with known spacing out to
    +/-17 cm from the centre.  In the real recordings, the hand often occludes
    the central red dots, so hue-only red matching is not a reliable primary
    calibration signal.  This routine uses the visible small calibration dots in
    the first frames and the known 340 mm full span as the automatic primary
    scale when exact labelled red-dot matching is under-constrained.
    """
    pts = []
    for frame in frames:
        arr = detect_small_calibration_blobs(frame)
        if arr.size:
            pts.append(arr)
    if not pts:
        return np.nan
    pts = np.vstack(pts)
    if len(pts) < 8:
        return np.nan
    xlo, xhi = np.nanpercentile(pts[:, 0], [3, 97])
    ylo, yhi = np.nanpercentile(pts[:, 1], [3, 97])
    span = max(float(xhi - xlo), float(yhi - ylo))
    px_per_mm = span / (2.0 * float(KNOWN_RED_DOT_DIST_MM[-1]))
    return px_per_mm if np.isfinite(px_per_mm) and 0.5 <= px_per_mm <= 30.0 else np.nan


def estimate_top_px_per_mm(video: Path, max_frames: int = 90) -> tuple[float, list[str]]:
    warnings = []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return np.nan, ["top scale: cannot open video"]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_centers = []
    sampled_frames = []
    for i in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, int(fps // 5)) != 0:
            continue
        sampled_frames.append(frame)
        all_centers.extend(detect_red_centers(frame))
    cap.release()
    field_scale = top_px_per_mm_from_calibration_dot_field(sampled_frames)
    if len(all_centers) < 3:
        if np.isfinite(field_scale):
            return field_scale, ["top scale: calibrated from first-frame calibration-dot field using known 34 cm span"]
        return np.nan, ["top scale: calibration dots not visible enough in first frames; mm errors may be NaN"]
    radii = np.array([r for _, _, r in all_centers], float)
    diameter_px_per_mm = float(np.nanmedian(2.0 * radii) / MARKER_DOT_DIAMETER_MM) if radii.size else np.nan
    pts = np.array([(x, y) for x, y, _ in all_centers], float)
    # Merge repeated detections by coarse rounding.
    rounded = np.round(pts / 4.0) * 4.0
    uniq = []
    for u in np.unique(rounded, axis=0):
        d = np.linalg.norm(rounded - u, axis=1)
        group = pts[d < 6.0]
        if len(group) >= 1:
            uniq.append(group.mean(axis=0))
    pts = np.array(uniq, float)
    if len(pts) < 3:
        if np.isfinite(field_scale):
            return field_scale, ["top scale: calibrated from first-frame calibration-dot field using known 34 cm span"]
        return np.nan, ["top scale: calibration dot detections were unstable; mm errors may be NaN"]
    center = pts[np.argmin(np.linalg.norm(pts - np.median(pts, axis=0), axis=1))]
    pix = np.sort(np.linalg.norm(pts - center, axis=1))
    pix = pix[pix > 2.0]
    m = min(len(pix), len(KNOWN_RED_DOT_DIST_MM) - 1)
    if m < 2:
        if np.isfinite(diameter_px_per_mm) and 0.5 <= diameter_px_per_mm <= 30.0:
            return diameter_px_per_mm, ["top scale: used known 5mm red-dot diameter because spacing match was sparse"]
        if np.isfinite(field_scale):
            return field_scale, ["top scale: calibrated from first-frame calibration-dot field using known 34 cm span"]
        return np.nan, ["top scale: not enough non-center red dots for px/mm"]
    mm = KNOWN_RED_DOT_DIST_MM[1 : m + 1]
    # Least-squares through origin: pixel distance = px_per_mm * mm.
    px_per_mm = float(np.dot(mm, pix[:m]) / np.dot(mm, mm))
    # A normal webcam view of this board should be on the order of 1-10 px/mm.
    # Values far below that usually mean the detector latched onto unrelated
    # reddish pixels (for example wrist shadows or UI artifacts).  It is safer
    # to report NaN + warning than publish inflated "mm" errors.
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        return np.nan, ["top scale: invalid px/mm estimate"]
    if not (0.5 <= px_per_mm <= 30.0):
        if np.isfinite(field_scale):
            return field_scale, [
                "top scale: calibrated from first-frame calibration-dot field using known 34 cm span; "
                "labelled red-dot spacing was under-constrained by occlusion/colour contrast"
            ]
        if np.isfinite(diameter_px_per_mm) and 0.5 <= diameter_px_per_mm <= 30.0:
            return diameter_px_per_mm, [
                "top scale: used known 5mm red-dot diameter because labelled spacing was under-constrained"
            ]
        return np.nan, ["top scale: labelled red-dot spacing under-constrained and field scale unavailable; mm errors set to NaN"]
    return px_per_mm, warnings


def affine_metrics_mm(
    t_truth: np.ndarray,
    truth_x: np.ndarray,
    truth_y: np.ndarray,
    t_meas: np.ndarray,
    meas_x: np.ndarray,
    meas_y: np.ndarray,
    fs: float = 30.0,
    meas_px_per_mm: float | None = None,
) -> tuple[float, float, int, float]:
    """Fit truth->measured, score residual in measured pixels/mm."""
    t0 = max(np.nanmin(t_truth), np.nanmin(t_meas))
    t1 = min(np.nanmax(t_truth), np.nanmax(t_meas))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.nan, np.nan, 0, np.nan
    grid = np.arange(t0, t1, 1.0 / fs)
    _, tx = LA.resample_uniform(t_truth, truth_x, fs, grid)
    _, ty = LA.resample_uniform(t_truth, truth_y, fs, grid)
    _, mx = LA.resample_uniform(t_meas, meas_x, fs, grid)
    _, my = LA.resample_uniform(t_meas, meas_y, fs, grid)
    n = min(len(tx), len(ty), len(mx), len(my))
    if n < 6:
        return np.nan, np.nan, n, np.nan
    tx, ty, mx, my = tx[:n], ty[:n], mx[:n], my[:n]
    good = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(mx) & np.isfinite(my)
    tx, ty, mx, my = tx[good], ty[good], mx[good], my[good]
    n = len(tx)
    if n < 6:
        return np.nan, np.nan, n, np.nan
    A = np.vstack([tx, ty, np.ones(n)]).T
    target = np.vstack([mx, my]).T
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    pred = A @ coef
    res = target - pred
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((target - target.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse_px = float(np.sqrt(ss_res / n))
    rmse_mm = rmse_px / meas_px_per_mm if meas_px_per_mm and np.isfinite(meas_px_per_mm) and meas_px_per_mm > 0 else np.nan
    # linear scale of truth-units -> measured-pixels, useful to convert virtual residuals later
    linear = coef[:2, :].T
    try:
        scale = float(np.mean(np.linalg.svd(linear, compute_uv=False)))
    except Exception:
        scale = np.nan
    return r2, rmse_mm, n, scale


def motion_signal(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.hypot(x - np.nanmedian(x), y - np.nanmedian(y))


def _gate_boundary_lag(lag_ms: float, max_lag_s: float = 2.0) -> float:
    """Reject lags pinned to the search boundary; they are under-constrained."""
    if np.isfinite(lag_ms) and abs(lag_ms) >= 0.95 * max_lag_s * 1000.0:
        return np.nan
    return lag_ms


def latency_ms(t_truth, truth_x, truth_y, t_meas, meas_x, meas_y, fs=30.0, max_lag_s: float = 2.0):
    a = motion_signal(truth_x, truth_y)
    b = motion_signal(meas_x, meas_y)
    res = LA.estimate_lag(np.asarray(t_truth, float), a, np.asarray(t_meas, float), b, fs=fs, max_lag_s=max_lag_s, use_speed=True)
    lag, corr = LA.gated_lag(res, min_corr=0.20)
    lag = _gate_boundary_lag(lag, max_lag_s=max_lag_s)
    return lag, corr


def latency_1d_ms(t_a, a, t_b, b, fs=30.0, max_lag_s: float = 2.0):
    res = LA.estimate_lag(np.asarray(t_a, float), np.asarray(a, float), np.asarray(t_b, float), np.asarray(b, float), fs=fs, max_lag_s=max_lag_s, use_speed=True)
    lag, corr = LA.gated_lag(res, min_corr=0.20)
    lag = _gate_boundary_lag(lag, max_lag_s=max_lag_s)
    return lag, corr


def _mad(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.nanmedian(np.abs(a - np.nanmedian(a))))


def _sustained_first_time(t: np.ndarray, active: np.ndarray, min_duration_s: float = 0.25) -> float:
    if t.size == 0 or active.size == 0:
        return np.nan
    dt = float(np.nanmedian(np.diff(t))) if t.size > 1 else min_duration_s
    need = max(1, int(np.ceil(min_duration_s / max(dt, 1e-6))))
    run = 0
    first = 0
    for i, ok in enumerate(active):
        if ok:
            if run == 0:
                first = i
            run += 1
            if run >= need:
                return float(t[first])
        else:
            run = 0
    return np.nan


def movement_onset_1d(t, values) -> float:
    """Estimate the first sustained movement time from a one-dimensional trace."""
    t = np.asarray(t, float)
    y = np.asarray(values, float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if t.size < 4:
        return np.nan
    order = np.argsort(t)
    t, y = t[order], y[order]
    unique = np.r_[True, np.diff(t) > 1e-9]
    t, y = t[unique], y[unique]
    if t.size < 4:
        return np.nan
    duration = float(t[-1] - t[0])
    baseline_end = t[0] + min(2.0, max(0.5, 0.20 * duration))
    base_mask = t <= baseline_end
    if base_mask.sum() < 3:
        base_mask = np.zeros_like(t, dtype=bool)
        base_mask[: max(1, min(5, t.size // 3))] = True

    # Primary detector: sustained speed above baseline noise.
    dy = np.gradient(y)
    dt = np.gradient(t)
    speed = np.abs(dy) / np.maximum(np.abs(dt), 1e-6)
    if speed.size >= 5:
        k = min(5, speed.size if speed.size % 2 == 1 else speed.size - 1)
        if k >= 3:
            speed = np.convolve(speed, np.ones(k) / k, mode="same")
    base = speed[base_mask]
    spread = float(np.nanpercentile(speed, 95) - np.nanpercentile(speed, 5))
    threshold = float(np.nanmedian(base) + max(4.0 * (_mad(base) or 0.0), 0.12 * spread, 1e-9))
    onset = _sustained_first_time(t, speed > threshold)
    if np.isfinite(onset):
        return onset

    # Fallback detector: sustained displacement from the initial rest position.
    y0 = float(np.nanmedian(y[base_mask]))
    disp = np.abs(y - y0)
    amp = float(np.nanpercentile(disp, 95) - np.nanpercentile(disp, 5))
    if amp <= 1e-9:
        return np.nan
    return _sustained_first_time(t, disp > max(0.15 * amp, 3.0 * (_mad(disp[base_mask]) or 0.0), 1e-9))


def movement_onset_2d(df: pd.DataFrame, time_col: str, x_col: str, y_col: str) -> float:
    if df.empty or not {time_col, x_col, y_col}.issubset(df.columns):
        return np.nan
    t = df[time_col].to_numpy(float)
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)
    good = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    if good.sum() < 4:
        return np.nan
    # Use scalar displacement around the rest position, which is robust to axis
    # flips and different camera coordinate systems.
    tg = t[good]
    xg, yg = x[good], y[good]
    order = np.argsort(tg)
    tg, xg, yg = tg[order], xg[order], yg[order]
    duration = float(tg[-1] - tg[0])
    baseline_end = tg[0] + min(2.0, max(0.5, 0.20 * duration))
    base = tg <= baseline_end
    if base.sum() < 3:
        base = np.zeros_like(tg, dtype=bool)
        base[: max(1, min(5, tg.size // 3))] = True
    x0 = float(np.nanmedian(xg[base]))
    y0 = float(np.nanmedian(yg[base]))
    return movement_onset_1d(tg, np.hypot(xg - x0, yg - y0))


def _finite_or_zero(value: float, warnings: list[str], label: str) -> float:
    if np.isfinite(value):
        return float(value)
    warnings.append(f"{label}: movement start could not be detected; using first frame as sync zero")
    return 0.0


def apply_movement_start_alignment(
    tracking: pd.DataFrame,
    top: pd.DataFrame,
    virt: pd.DataFrame,
    side: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """Add synchronized time columns using automatic per-stream movement starts.

    Raw ``time_s`` columns are preserved.  ``sync_time_s`` is zero at the
    detected movement-start mark for each video stream.  For virtual video the
    preferred mark is the white virtual-finger dot, so object delay is preserved.
    For side video the preferred mark is motor angle movement, so motor->tactor
    delay is preserved; if motor angles are unavailable we fall back to the
    yellow tactor dot and warn because this removes absolute tactor onset delay.
    """
    warnings: list[str] = []
    tracking = tracking.copy()
    top = top.copy()
    virt = virt.copy()
    side = side.copy()

    tracking_start = _finite_or_zero(
        movement_onset_2d(tracking, "t", "active_finger_x", "active_finger_y"),
        warnings,
        "tracking.csv active finger",
    )
    top_start = _finite_or_zero(
        movement_onset_2d(top, "time_s", "mp_active_x", "mp_active_y"),
        warnings,
        "top_camera MediaPipe hand",
    )

    virt_source = "virtual_finger"
    virt_start = movement_onset_2d(virt, "time_s", "virtual_finger_x", "virtual_finger_y")
    if not np.isfinite(virt_start):
        virt_source = "virtual_object"
        virt_start = movement_onset_2d(virt, "time_s", "obj_x", "obj_y")
    virt_start = _finite_or_zero(virt_start, warnings, f"virtual_object.mp4 {virt_source}")

    side_source = "motor_angle"
    mt_raw, ma_raw = mean_motor_angle_signal(side, time_col="time_s")
    motor_start = movement_onset_1d(mt_raw, ma_raw)
    side_start = motor_start
    if not np.isfinite(side_start):
        side_source = "yellow_tactor"
        side_start = movement_onset_2d(side, "time_s", "tactor_x", "tactor_y")
        if np.isfinite(side_start):
            warnings.append("side_camera sync used yellow tactor start because motor angle start was unavailable; absolute motor/tactor onset latency is less reliable")
    side_start = _finite_or_zero(side_start, warnings, f"side_camera {side_source}")

    tracking["sync_time_s"] = tracking["t"].astype(float) - tracking_start
    top["sync_time_s"] = top["time_s"].astype(float) - top_start if not top.empty else []
    virt["sync_time_s"] = virt["time_s"].astype(float) - virt_start if not virt.empty else []
    side["sync_time_s"] = side["time_s"].astype(float) - side_start if not side.empty else []
    for df, start in [(tracking, tracking_start), (top, top_start), (virt, virt_start), (side, side_start)]:
        if not df.empty:
            df["movement_start_s"] = start

    meta = {
        "tracking_movement_start_s": tracking_start,
        "top_movement_start_s": top_start,
        "virtual_movement_start_s": virt_start,
        "side_movement_start_s": side_start,
        "motor_movement_start_s": motor_start,
        "virtual_alignment_source": virt_source,
        "side_alignment_source": side_source,
    }
    return tracking, top, virt, side, meta, warnings


def signal_time(df: pd.DataFrame, raw_col: str = "time_s") -> pd.Series:
    if "sync_time_s" in df.columns:
        return df["sync_time_s"]
    if raw_col in df.columns:
        return df[raw_col]
    return pd.Series(dtype=float)


def command_time(command: pd.DataFrame) -> pd.Series:
    return signal_time(command)


def backend_command_signal(command: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if command.empty or "backend_motor_command_magnitude" not in command:
        return np.array([]), np.array([])
    t = command_time(command).to_numpy(float)
    sig = command["backend_motor_command_magnitude"].to_numpy(float)
    good = np.isfinite(t) & np.isfinite(sig)
    return t[good], sig[good]


def tracking_time(tracking: pd.DataFrame) -> pd.Series:
    if "sync_time_s" in tracking.columns:
        return tracking["sync_time_s"]
    return tracking["t"]


def mean_motor_angle_signal(side: pd.DataFrame, time_col: str = "time_s") -> tuple[np.ndarray, np.ndarray]:
    """Return time and average unwrapped motor-angle signal from all detected spools."""
    if side.empty:
        return np.array([]), np.array([])
    if time_col not in side.columns:
        time_col = "time_s" if "time_s" in side.columns else ""
    if not time_col:
        return np.array([]), np.array([])
    cols = [c for c in side.columns if c.endswith("_angle_unwrapped")]
    if not cols:
        cols = [c for c in side.columns if c.endswith("_angle")]
    if not cols:
        return np.array([]), np.array([])
    arr = side[cols].astype(float).to_numpy()
    valid_rate = np.isfinite(arr).mean(axis=0)
    keep = valid_rate >= 0.20
    if not keep.any():
        return np.array([]), np.array([])
    arr = arr[:, keep]
    sig = np.nanmean(arr, axis=1)
    good = np.isfinite(sig) & np.isfinite(side[time_col].to_numpy(float))
    return side[time_col].to_numpy(float)[good], sig[good]


def draw_label(frame: np.ndarray, text: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def nearest_signal_row(df: pd.DataFrame, frame_idx: int) -> pd.Series | None:
    if df.empty or "frame" not in df:
        return None
    frames = df["frame"].to_numpy()
    pos = int(np.argmin(np.abs(frames - frame_idx)))
    if abs(int(frames[pos]) - frame_idx) > max(1, np.nanmedian(np.diff(frames)) if len(frames) > 1 else 1):
        return None
    return df.iloc[pos]


def render_detected_video(video_path: Path, signals: pd.DataFrame, out_path: Path, kind: str) -> str:
    """Render a full-length annotated video into the dedicated results folder."""
    meta = video_meta(video_path)
    if not meta["opened"]:
        return f"detected video skipped: cannot open {video_path.name}"
    cap = cv2.VideoCapture(str(video_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, meta["fps"] or 30.0, (meta["width"], meta["height"]))
    start_s = np.nan
    if "movement_start_s" in signals.columns and signals["movement_start_s"].notna().any():
        start_s = float(signals["movement_start_s"].dropna().iloc[0])
    start_frame = int(round(start_s * (meta["fps"] or 30.0))) if np.isfinite(start_s) else None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = nearest_signal_row(signals, frame_idx)
        draw_label(frame, f"{kind} detection | frame {frame_idx}", 24, (255, 255, 0))
        if np.isfinite(start_s):
            sync_t = frame_idx / (meta["fps"] or 30.0) - start_s
            draw_label(frame, f"sync time {sync_t:+.2f}s | movement start {start_s:.2f}s", meta["height"] - 16, (255, 255, 255))
        if start_frame is not None and abs(frame_idx - start_frame) <= int(0.5 * (meta["fps"] or 30.0)):
            cv2.rectangle(frame, (0, 0), (meta["width"] - 1, meta["height"] - 1), (0, 255, 255), 4)
            draw_label(frame, "MOVEMENT START / SYNC MARK", 72, (0, 255, 255))
        if row is not None:
            if kind == "top":
                if bool(row.get("detected", False)):
                    ax, ay = row.get("mp_active_x", np.nan), row.get("mp_active_y", np.nan)
                    tx, ty = row.get("mp_thumb_x", np.nan), row.get("mp_thumb_y", np.nan)
                    if np.isfinite(ax) and np.isfinite(ay):
                        cv2.circle(frame, (int(ax), int(ay)), 7, (0, 255, 255), 2)
                        draw_label(frame, "active finger", 48, (0, 255, 255))
                    if np.isfinite(tx) and np.isfinite(ty):
                        cv2.circle(frame, (int(tx), int(ty)), 7, (255, 0, 255), 2)
                else:
                    draw_label(frame, "MediaPipe hand not detected", 48, (0, 0, 255))
            elif kind == "virtual":
                ox, oy = row.get("obj_x", np.nan), row.get("obj_y", np.nan)
                fx, fy = row.get("virtual_finger_x", np.nan), row.get("virtual_finger_y", np.nan)
                if np.isfinite(ox) and np.isfinite(oy):
                    cv2.circle(frame, (int(ox), int(oy)), 8, (0, 165, 255), 2)
                    draw_label(frame, "virtual object", 48, (0, 165, 255))
                if np.isfinite(fx) and np.isfinite(fy):
                    cv2.circle(frame, (int(fx), int(fy)), 5, (255, 255, 255), 2)
            elif kind == "side":
                if {"thimble_x", "thimble_y", "thimble_w", "thimble_h"}.issubset(row.index):
                    x, y, w, h = [row.get(c, np.nan) for c in ["thimble_x", "thimble_y", "thimble_w", "thimble_h"]]
                    if all(np.isfinite(v) for v in [x, y, w, h]):
                        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 180, 0), 2)
                        draw_label(frame, "green thimble ROI", 48, (0, 220, 0))
                tx, ty = row.get("tactor_x", np.nan), row.get("tactor_y", np.nan)
                if np.isfinite(tx) and np.isfinite(ty):
                    cv2.circle(frame, (int(tx), int(ty)), 7, (0, 255, 255), 2)
                    draw_label(frame, "yellow tactor dot", 70, (0, 255, 255))
                if {"motor_slot_x", "motor_slot_y", "motor_slot_w", "motor_slot_h"}.issubset(row.index):
                    x, y, w, h = [row.get(c, np.nan) for c in ["motor_slot_x", "motor_slot_y", "motor_slot_w", "motor_slot_h"]]
                    if all(np.isfinite(v) for v in [x, y, w, h]):
                        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 0), 2)
                for i in range(3):
                    ang = row.get(f"spool{i}_angle", np.nan)
                    cx = row.get(f"spool{i}_cx", np.nan)
                    cy = row.get(f"spool{i}_cy", np.nan)
                    r = row.get(f"spool{i}_r", np.nan)
                    dx = row.get(f"spool{i}_dx", np.nan)
                    dy = row.get(f"spool{i}_dy", np.nan)
                    if np.isfinite(ang):
                        draw_label(frame, f"spool{i} angle={ang:.1f} deg", 72 + 22 * i, (255, 255, 0))
                    if all(np.isfinite(v) for v in [cx, cy, r]):
                        cv2.circle(frame, (int(cx), int(cy)), int(r), (255, 255, 0), 1)
                    if all(np.isfinite(v) for v in [cx, cy, r, dx, dy]):
                        p1 = (int(cx - dx * r), int(cy - dy * r))
                        p2 = (int(cx + dx * r), int(cy + dy * r))
                        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        writer.write(frame)
        frame_idx += 1
    writer.release()
    cap.release()
    return f"detected video saved: {out_path}"


def extract_top_mediapipe(pair: Path, stride: int, out_csv: Path) -> tuple[pd.DataFrame, list[str], float]:
    tracking = read_tracking(pair)
    video = pair / "top_camera.mp4"
    meta = video_meta(video)
    if not meta["opened"]:
        return pd.DataFrame(), [f"missing top_camera.mp4 for {pair.name}"], np.nan
    px_per_mm, warnings = estimate_top_px_per_mm(video)
    finger = str(tracking["finger"].dropna().iloc[0]).lower() if "finger" in tracking and tracking["finger"].notna().any() else "index"
    landmark = FINGER_LANDMARK.get(finger, FINGER_LANDMARK["index"])
    cap = cv2.VideoCapture(str(video))
    # Offline validation should re-detect each sampled frame. In tracking mode
    # MediaPipe can temporarily lose the flat/edge hand and then the rendered
    # overlay looks very wrong. Static-image mode is slower but much more
    # reliable for post-record analysis.
    try:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.20,
            min_tracking_confidence=0.10,
        )
    except TypeError:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.20,
            min_tracking_confidence=0.10,
        )
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        ts, t = frame_timestamp(tracking, frame_idx, meta["fps"])
        row = {"frame": frame_idx, "time_s": t, "timestamp": ts, "detected": False,
               "mp_thumb_x": np.nan, "mp_thumb_y": np.nan, "mp_active_x": np.nan, "mp_active_y": np.nan}
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            thumb = lm[mp.solutions.hands.HandLandmark.THUMB_TIP]
            active = lm[landmark]
            row.update({
                "detected": True,
                # Offline post-record video coordinates: keep camera pixel
                # coordinates unmirrored so detection overlays line up with the
                # visible hand. Any realtime coordinate-system differences are
                # handled by affine metrics later.
                "mp_thumb_x": thumb.x * meta["width"],
                "mp_thumb_y": thumb.y * meta["height"],
                "mp_active_x": active.x * meta["width"],
                "mp_active_y": active.y * meta["height"],
            })
        rows.append(row)
        frame_idx += 1
    hands.close()
    cap.release()
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df, warnings, px_per_mm


def detect_virtual(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # orange and blue square on virtual display
    orange = cv2.inRange(hsv, (5, 70, 50), (30, 255, 255))
    blue = cv2.inRange(hsv, (90, 50, 40), (135, 255, 255))
    color_mask = cv2.morphologyEx(orange | blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obj = (np.nan, np.nan, 0.0)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 20:
            m = cv2.moments(c)
            if m["m00"]:
                obj = (m["m10"] / m["m00"], m["m01"] / m["m00"], area)
    white = cv2.inRange(hsv, (0, 0, 180), (179, 70, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot = (np.nan, np.nan, 0.0)
    small = []
    h, w = frame.shape[:2]
    for c in cnts:
        area = cv2.contourArea(c)
        if 3 <= area <= 400:
            x, y, ww, hh = cv2.boundingRect(c)
            if 0 < x < w - 1 and 0 < y < h - 1:
                small.append(c)
    if small:
        c = max(small, key=cv2.contourArea)
        m = cv2.moments(c)
        if m["m00"]:
            dot = (m["m10"] / m["m00"], m["m01"] / m["m00"], cv2.contourArea(c))
    return obj, dot


def extract_virtual(pair: Path, stride: int, out_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    tracking = read_tracking(pair)
    video = pair / "virtual_object.mp4"
    warnings = []
    meta = video_meta(video)
    if not meta["opened"]:
        return pd.DataFrame(), [f"missing virtual_object.mp4 for {pair.name}"]
    cap = cv2.VideoCapture(str(video))
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        ts, t = frame_timestamp(tracking, frame_idx, meta["fps"])
        (ox, oy, oa), (fx, fy, fa) = detect_virtual(frame)
        rows.append({"frame": frame_idx, "time_s": t, "timestamp": ts,
                     "obj_x": ox, "obj_y": oy, "obj_area": oa,
                     "virtual_finger_x": fx, "virtual_finger_y": fy, "virtual_finger_area": fa})
        frame_idx += 1
    cap.release()
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df, warnings


def detect_green_thimble_roi(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the green thimble bounding box, constrained to the left/finger area."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    # Saturated green plastic thimble. Restrict to left 65% to avoid motor box
    # and cable/specular artifacts.
    mask = cv2.inRange(hsv, (35, 35, 20), (95, 255, 230))
    mask[:, int(w * 0.65) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        if 20 <= ww <= 220 and 20 <= hh <= 220:
            candidates.append((area, x, y, ww, hh))
    if not candidates:
        return None
    _, x, y, ww, hh = max(candidates, key=lambda z: z[0])
    pad = 18
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + ww + pad)
    y1 = min(h, y + hh + pad)
    return x0, y0, x1 - x0, y1 - y0


def detect_yellow_tactor(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = detect_green_thimble_roi(frame)
    mask = cv2.inRange(hsv, (18, 70, 85), (45, 255, 255))
    if roi is not None:
        x, y, w, h = roi
        keep = np.zeros(mask.shape, dtype=np.uint8)
        keep[y : y + h, x : x + w] = 255
        mask = cv2.bitwise_and(mask, keep)
    else:
        # Still keep the detector away from motor-window highlights if green is
        # temporarily missed.
        mask[:, int(frame.shape[1] * 0.65) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 8 <= area <= 800:
            (x, y), r = cv2.minEnclosingCircle(c)
            if 2.0 <= r <= 18.0:
                candidates.append((area, x, y, r))
    if not candidates:
        return np.nan, np.nan, np.nan
    area, x, y, r = max(candidates, key=lambda z: z[0])
    return float(x), float(y), float(2 * r)


def find_motor_slot(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the vertical dark motor/spool window on the right-side white box."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x0 = int(w * 0.60)
    roi = gray[:, x0:]
    mask = (roi < 105).astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        gx = x + x0
        area = cv2.contourArea(c)
        aspect = hh / max(ww, 1)
        if area >= 350 and 12 <= ww <= 90 and 55 <= hh <= 220 and aspect >= 1.8:
            # Prefer the actual tall spool aperture: far right, vertical, and
            # not a wide shadow blob.
            boxes.append((area, gx, y, ww, hh, aspect))
    if not boxes:
        return None
    area, x, y, ww, hh, _ = max(boxes, key=lambda b: (b[1], b[5], b[0]))
    return int(x), int(y), int(ww), int(hh)


def detect_spools_in_motor_slot(frame: np.ndarray) -> tuple[list[SpoolROI], tuple[int, int, int, int] | None]:
    """Find the dark motor slot and the three white spools inside it.

    The new side-camera view has a white motor box with a dark vertical window
    on the right.  Inside that window are three white spools carrying black
    angle lines.  This detector finds the slot first, then groups bright rows
    inside the slot into exactly three spool ROIs.  It is more appropriate for
    the new data than the old full-frame stacked-disc heuristic.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    slot = find_motor_slot(frame)
    if slot is None:
        return [], None
    x, y, ww, hh = slot
    crop_bgr = frame[y : y + hh, x : x + ww]
    crop = gray[y : y + hh, x : x + ww]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, (0, 0, 115), (179, 95, 255)) > 0
    row = bright.sum(axis=1).astype(float)
    smoothed = np.convolve(row, np.ones(5) / 5, mode="same")
    thresh = max(3.0, float(np.percentile(smoothed, 70) * 0.5))
    active = smoothed > thresh
    groups = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        if (not is_active or i == len(active) - 1) and start is not None:
            end = i if not is_active else i + 1
            if end - start >= 5:
                groups.append((start, end))
            start = None
    candidates = []
    for a, b in groups:
        ys, xs = np.where(bright[a:b, :])
        if len(xs) < 10:
            continue
        cx = x + float(np.median(xs))
        cy = y + a + float(np.median(ys))
        r = float(max((b - a) / 2, (xs.max() - xs.min() + 1) / 2, 6.0))
        candidates.append(SpoolROI(cx=cx, cy=cy, r=r))
    candidates = sorted(candidates, key=lambda s: s.cy)
    if len(candidates) >= 3:
        # Use exactly the three row groups in the motor slot. Their centers stay
        # fixed for the whole video; only the black groove line rotates.
        candidates = candidates[:3]
    return candidates, (x, y, ww, hh)


def extract_side(pair: Path, stride: int, out_csv: Path) -> tuple[pd.DataFrame, list[str], float]:
    tracking = read_tracking(pair)
    video = pair / "side_camera.mp4"
    warnings = []
    meta = video_meta(video)
    if not meta["opened"]:
        return pd.DataFrame(), [f"missing side_camera.mp4 for {pair.name}"], np.nan
    cap = cv2.VideoCapture(str(video))
    rows = []
    spools = []
    motor_slot = None
    ok, first = cap.read()
    if ok:
        spools, motor_slot = detect_spools_in_motor_slot(first)
        if len(spools) < 3 and init_spools is not None and Box is not None:
            try:
                spools = init_spools(first, Box(0, 0, meta["width"], meta["height"]), max_spools=3)
                warnings.append("motor spools: used legacy full-frame detector because motor-slot detector found fewer than 3 spools")
            except Exception as e:
                warnings.append(f"motor spool auto-detection failed: {e}")
        if len(spools) < 3:
            warnings.append(f"motor spools: automatic detector found {len(spools)} of 3 spools")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    diameters = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        ts, t = frame_timestamp(tracking, frame_idx, meta["fps"])
        thimble_roi = detect_green_thimble_roi(frame)
        tx, ty, diam = detect_yellow_tactor(frame)
        if np.isfinite(diam):
            diameters.append(diam)
        row = {"frame": frame_idx, "time_s": t, "timestamp": ts,
               "tactor_x": tx, "tactor_y": ty, "tactor_dot_diameter_px": diam}
        if thimble_roi is not None:
            row.update({"thimble_x": thimble_roi[0], "thimble_y": thimble_roi[1], "thimble_w": thimble_roi[2], "thimble_h": thimble_roi[3]})
        if motor_slot:
            row.update({"motor_slot_x": motor_slot[0], "motor_slot_y": motor_slot[1], "motor_slot_w": motor_slot[2], "motor_slot_h": motor_slot[3]})
        if measure_groove is not None and spools:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i, s in enumerate(spools[:3]):
                # SpoolROI coordinates are relative to the supplied Box; full-frame box => already global-ish.
                try:
                    ang, conf, direction = measure_groove(gray, s.cx, s.cy, s.r)
                except Exception:
                    ang, conf, direction = np.nan, np.nan, (np.nan, np.nan)
                row[f"spool{i}_cx"] = s.cx
                row[f"spool{i}_cy"] = s.cy
                row[f"spool{i}_r"] = s.r
                row[f"spool{i}_angle"] = ang
                row[f"spool{i}_conf"] = conf
                row[f"spool{i}_dx"] = direction[0] if direction is not None else np.nan
                row[f"spool{i}_dy"] = direction[1] if direction is not None else np.nan
        rows.append(row)
        frame_idx += 1
    cap.release()
    df = pd.DataFrame(rows)
    angle_cols = [c for c in df.columns if c.endswith("_angle")]
    for c in angle_cols:
        df[c + "_unwrapped"] = np.rad2deg(np.unwrap(np.deg2rad(df[c].astype(float).interpolate(limit_direction="both"))))
    px_per_mm = float(np.nanmedian(diameters) / YELLOW_DOT_DIAMETER_MM) if diameters else np.nan
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        warnings.append("side scale: could not estimate yellow-dot px/mm; tactor mm errors may be NaN")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df, warnings, px_per_mm


def plot_pair(pair: str, tracking: pd.DataFrame, top: pd.DataFrame, virt: pd.DataFrame, side: pd.DataFrame, command: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    tt = tracking_time(tracking)
    if not top.empty:
        axes[0].plot(tt, motion_signal(tracking["active_finger_x"], tracking["active_finger_y"]), label="realtime tracking")
        axes[0].plot(signal_time(top), motion_signal(top["mp_active_x"], top["mp_active_y"]), label="offline MediaPipe")
        axes[0].set_title(f"{pair}: top camera hand motion")
        axes[0].legend()
    if not virt.empty:
        axes[1].plot(tt, motion_signal(tracking["object_x"], tracking["object_y"]), label="tracking object")
        axes[1].plot(signal_time(virt), motion_signal(virt["obj_x"], virt["obj_y"]), label="video object")
        axes[1].set_title("virtual object motion")
        axes[1].legend()
    if not side.empty:
        ct, cs = backend_command_signal(command)
        if len(ct):
            axes[2].plot(ct, _norm(cs), label="backend-equivalent motor command")
        else:
            axes[2].plot(tt, motion_signal(tracking["active_finger_x"], tracking["active_finger_y"]), label="fallback finger motion")
        axes[2].plot(signal_time(side), motion_signal(side["tactor_x"], side["tactor_y"]), label="yellow tactor")
        mt, ma = mean_motor_angle_signal(side, time_col="sync_time_s")
        if len(mt):
            axes[2].plot(mt, _norm(ma), label="motor black-line angle")
        axes[2].set_title("tactor vs reconstructed backend command (not recorded firmware log)")
        axes[2].legend()
    for ax in axes:
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlabel("synchronized time (s); movement start = 0")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"{pair}_three_video_qc.png", dpi=140)
    plt.close(fig)


def add_window_latency(rows: list[dict], pair: str, chain: str, t0: float, t1: float, lag: float, corr: float) -> None:
    rows.append({
        "pair": pair,
        "chain": chain,
        "window_start_s": t0,
        "window_end_s": t1,
        "latency_ms": lag,
        "corr": corr,
    })


def windowed_latency_metrics(pair: str, tracking: pd.DataFrame, top: pd.DataFrame, virt: pd.DataFrame, side: pd.DataFrame, command: pd.DataFrame, fs: float, out_csv: Path, fig_dir: Path) -> pd.DataFrame:
    """Latency over time in sliding windows for the three requested chains."""
    rows: list[dict] = []
    if tracking.empty:
        return pd.DataFrame()
    trk_t = tracking_time(tracking)
    top_t = signal_time(top) if not top.empty else pd.Series(dtype=float)
    virt_t = signal_time(virt) if not virt.empty else pd.Series(dtype=float)
    side_t = signal_time(side) if not side.empty else pd.Series(dtype=float)
    cmd_t = command_time(command) if not command.empty else pd.Series(dtype=float)
    t_min, t_max = float(trk_t.min()), float(trk_t.max())
    win = 15.0
    step = 5.0
    if t_max - t_min < win:
        win = max(3.0, t_max - t_min)
        step = win
    starts = np.arange(t_min, max(t_min + 1e-6, t_max - win + 1e-6), step)
    if starts.size == 0:
        starts = np.array([t_min])

    for start in starts:
        end = min(start + win, t_max)
        tm = (trk_t >= start) & (trk_t <= end)
        if tm.sum() < 4:
            continue
        if not top.empty:
            m = (top_t >= start) & (top_t <= end) & top["mp_active_x"].notna() & top["mp_active_y"].notna()
            if m.sum() >= 4:
                lag, corr = latency_ms(trk_t.loc[tm], tracking.loc[tm, "active_finger_x"], tracking.loc[tm, "active_finger_y"], top_t.loc[m], top.loc[m, "mp_active_x"], top.loc[m, "mp_active_y"], fs=fs)
                add_window_latency(rows, pair, "finger_realtime_to_finger_offline_top", start, end, lag, corr)
        if not virt.empty:
            m = (virt_t >= start) & (virt_t <= end) & virt["obj_x"].notna() & virt["obj_y"].notna()
            if m.sum() >= 4:
                lag, corr = latency_ms(trk_t.loc[tm], tracking.loc[tm, "active_finger_x"], tracking.loc[tm, "active_finger_y"], virt_t.loc[m], virt.loc[m, "obj_x"], virt.loc[m, "obj_y"], fs=fs)
                add_window_latency(rows, pair, "finger_to_virtual_object", start, end, lag, corr)
        if not side.empty:
            sm = (side_t >= start) & (side_t <= end)
            tg = sm & side["tactor_x"].notna() & side["tactor_y"].notna()
            cm = (cmd_t >= start) & (cmd_t <= end) if not command.empty else pd.Series(False, index=command.index)
            if tg.sum() >= 4:
                if not command.empty and cm.sum() >= 4:
                    lag, corr = latency_ms(cmd_t.loc[cm], command.loc[cm, "backend_tactor_target_x"], command.loc[cm, "backend_tactor_target_y"], side_t.loc[tg], side.loc[tg, "tactor_x"], side.loc[tg, "tactor_y"], fs=fs)
                    add_window_latency(rows, pair, "backend_tactor_target_to_tactor", start, end, lag, corr)
                else:
                    lag, corr = latency_ms(trk_t.loc[tm], tracking.loc[tm, "active_finger_x"], tracking.loc[tm, "active_finger_y"], side_t.loc[tg], side.loc[tg, "tactor_x"], side.loc[tg, "tactor_y"], fs=fs)
                    add_window_latency(rows, pair, "fallback_finger_to_tactor", start, end, lag, corr)
            mt, ma = mean_motor_angle_signal(side.loc[sm].copy(), time_col="sync_time_s")
            if len(mt) >= 4:
                if not command.empty and cm.sum() >= 4:
                    lag, corr = latency_1d_ms(cmd_t.loc[cm], command.loc[cm, "backend_motor_command_magnitude"], mt, ma, fs=fs)
                    add_window_latency(rows, pair, "backend_motor_command_to_motor_angle", start, end, lag, corr)
                else:
                    inferred = motion_signal(tracking.loc[tm, "active_finger_x"], tracking.loc[tm, "active_finger_y"])
                    lag, corr = latency_1d_ms(trk_t.loc[tm], inferred, mt, ma, fs=fs)
                    add_window_latency(rows, pair, "fallback_finger_to_motor_angle", start, end, lag, corr)
                if tg.sum() >= 4:
                    tactor_motion = motion_signal(side.loc[tg, "tactor_x"], side.loc[tg, "tactor_y"])
                    lag, corr = latency_1d_ms(mt, ma, side_t.loc[tg], tactor_motion, fs=fs)
                    add_window_latency(rows, pair, "motor_angle_to_tactor", start, end, lag, corr)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df.empty:
        for chain, sub in df.groupby("chain"):
            mid = (sub["window_start_s"] + sub["window_end_s"]) / 2
            ax.plot(mid, sub["latency_ms"], marker="o", label=chain)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No reliable windowed latency estimates at this stride", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("synchronized time in pair (s); movement start = 0")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"{pair}: latency over time")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{pair}_latency_over_time.png", dpi=140)
    plt.close(fig)
    return df


def _norm(a) -> np.ndarray:
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)


def _ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return summary.sort_values("pair").reset_index(drop=True)


def fig_summary_table(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        "pair", "finger", "top_detection_rate", "top_hand_rmse_mm", "top_hand_latency_ms",
        "virtual_object_rmse_mm", "finger_to_virtual_object_latency_ms",
        "tactor_rmse_mm", "tactor_latency_ms", "motor_to_tactor_latency_ms",
        "top_movement_start_s", "virtual_movement_start_s", "side_movement_start_s",
    ]
    view = _ordered_summary(summary)[[c for c in cols if c in summary.columns]].copy()
    for c in view.select_dtypes("number").columns:
        view[c] = view[c].round(2)
    fig, ax = plt.subplots(figsize=(min(24, 1.35 * max(1, len(view.columns))), 1.0 + 0.45 * max(1, len(view))))
    ax.axis("off")
    if view.empty:
        ax.text(0.5, 0.5, "No pair summary rows", ha="center", va="center")
    else:
        tbl = ax.table(cellText=view.astype(str).values, colLabels=[c.replace("_", "\n") for c in view.columns], loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.55)
    ax.set_title("Three-video latency & accuracy summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_detection_accuracy(summary: pd.DataFrame, path: Path) -> None:
    s = _ordered_summary(summary)
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = [
        ("top_detection_rate", "top hand"),
        ("virtual_object_detection_rate", "virtual object"),
        ("virtual_finger_detection_rate", "virtual finger"),
        ("tactor_detection_rate", "tactor dot"),
        ("motor_spool_detection_rate", "motor spools"),
    ]
    width = 0.16
    for i, (col, label) in enumerate(cols):
        if col in s:
            ax.bar(x + (i - 2) * width, s[col].astype(float), width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(s["pair"] if "pair" in s else [])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection rate")
    ax.set_title("Detection rates per pair")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_latencies(summary: pd.DataFrame, path: Path) -> None:
    s = _ordered_summary(summary)
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cols = [
        ("top_hand_latency_ms", "realtime finger -> offline top"),
        ("finger_to_virtual_object_latency_ms", "finger -> virtual object"),
        ("backend_tactor_target_to_tactor_latency_ms", "backend tactor target -> tactor"),
        ("backend_command_to_motor_latency_ms", "backend motor command -> motor"),
        ("motor_to_tactor_latency_ms", "motor -> tactor"),
    ]
    width = 0.16
    for i, (col, label) in enumerate(cols):
        if col not in s:
            continue
        vals = s[col].astype(float).to_numpy()
        ax.bar(x + (i - 2) * width, np.nan_to_num(vals), width, label=label)
        for xi, v in zip(x + (i - 2) * width, vals):
            if not np.isfinite(v):
                ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=7, rotation=90, color="grey")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(s["pair"] if "pair" in s else [])
    ax.set_ylabel("latency (ms)")
    ax.set_title("Movement-start synchronized latencies per pair")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_alignment_onsets(summary: pd.DataFrame, path: Path) -> None:
    s = _ordered_summary(summary)
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = [
        ("tracking_movement_start_s", "tracking.csv"),
        ("top_movement_start_s", "top camera"),
        ("virtual_movement_start_s", "virtual video"),
        ("side_movement_start_s", "side camera"),
        ("motor_movement_start_s", "motor angle"),
    ]
    width = 0.16
    for i, (col, label) in enumerate(cols):
        if col in s:
            ax.bar(x + (i - 2) * width, s[col].astype(float), width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(s["pair"] if "pair" in s else [])
    ax.set_ylabel("raw movement-start time in each file (s)")
    ax.set_title("Automatic movement-start marks used for synchronization")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_object_tracking(session: Path, summary: pd.DataFrame, out_root: Path, path: Path) -> None:
    s = _ordered_summary(summary)
    fig, axes = plt.subplots(max(1, len(s)), 1, figsize=(10, max(2.4, 2.0 * max(1, len(s)))), squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], s.iterrows()):
        pair_name = row["pair"]
        pair_dir = session / pair_name
        virt_csv = out_root / "signals" / f"virtual_{pair_name}.csv"
        if not pair_dir.exists() or not virt_csv.exists():
            ax.text(0.5, 0.5, f"{pair_name}: missing data", transform=ax.transAxes, ha="center")
            continue
        trk = read_tracking(pair_dir)
        virt = pd.read_csv(virt_csv)
        trk_start = float(row.get("tracking_movement_start_s", 0.0) or 0.0)
        trk["sync_time_s"] = trk["t"] - trk_start
        ax.plot(tracking_time(trk), _norm(trk["object_x"]), lw=1, label="tracking object_x")
        if "sync_time_s" in virt:
            ax.plot(virt["sync_time_s"], _norm(virt["obj_x"]), lw=1, alpha=0.85, label="video object_x")
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylabel(pair_name)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("synchronized time (s); movement start = 0")
    fig.suptitle("Logged vs video virtual object tracking", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_motor_command_response(session: Path, summary: pd.DataFrame, out_root: Path, path: Path) -> None:
    s = _ordered_summary(summary)
    fig, axes = plt.subplots(max(1, len(s)), 1, figsize=(10, max(2.4, 2.1 * max(1, len(s)))), squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], s.iterrows()):
        pair_name = row["pair"]
        side_csv = out_root / "signals" / f"side_{pair_name}.csv"
        command_csv = out_root / "signals" / f"motor_command_{pair_name}.csv"
        if not side_csv.exists() or not command_csv.exists():
            ax.text(0.5, 0.5, f"{pair_name}: missing side/command data", transform=ax.transAxes, ha="center")
            continue
        side = pd.read_csv(side_csv)
        command = pd.read_csv(command_csv)
        st = signal_time(side)
        ct, cs = backend_command_signal(command)
        if len(ct):
            ax.plot(ct, _norm(cs), lw=1, label="backend-equivalent motor command")
        mt, ma = mean_motor_angle_signal(side, time_col="sync_time_s")
        if len(mt):
            ax.plot(mt, _norm(ma), lw=1, label="motor black-line angle")
        if {"tactor_x", "tactor_y"}.issubset(side.columns):
            ax.plot(st, _norm(motion_signal(side["tactor_x"], side["tactor_y"])), lw=1, label="yellow tactor")
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylabel(pair_name)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("synchronized time (s); movement start = 0")
    fig.suptitle("Backend-equivalent command -> motor angle -> tactor response", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_algorithm_vision_vs_hand_accuracy(summary: pd.DataFrame, path: Path) -> None:
    s = _ordered_summary(summary)
    x = np.arange(len(s))
    fig, ax1 = plt.subplots(figsize=(9, 4))
    if "top_hand_r2" in s:
        ax1.bar(x - 0.15, s["top_hand_r2"].astype(float), 0.3, label="top hand R²", color="#0072B2")
    if "virtual_object_r2" in s:
        ax1.bar(x + 0.15, s["virtual_object_r2"].astype(float), 0.3, label="virtual object R²", color="#009E73")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("R²")
    ax2 = ax1.twinx()
    if "top_hand_rmse_mm" in s:
        ax2.plot(x, s["top_hand_rmse_mm"].astype(float), "o-", color="#D55E00", label="top RMSE mm")
    ax2.set_ylabel("RMSE (mm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(s["pair"] if "pair" in s else [])
    ax1.set_title("Algorithm vision vs hand accuracy")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_bottom_line(summary: pd.DataFrame, path: Path) -> None:
    s = _ordered_summary(summary)
    metrics = [
        ("top_hand_rmse_mm", "top hand RMSE (mm)"),
        ("finger_to_virtual_object_latency_ms", "finger -> virtual latency (ms)"),
        ("backend_tactor_target_to_tactor_rmse_mm", "backend tactor target -> observed tactor RMSE (mm)"),
        ("backend_command_to_motor_latency_ms", "backend command -> motor angle latency (ms)"),
        ("motor_to_tactor_latency_ms", "motor -> tactor latency (ms)"),
    ]
    lines = ["Bottom-line synchronized validation summary", ""]
    for col, label in metrics:
        if col in s:
            vals = s[col].astype(float).to_numpy()
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lines.append(f"{label}: median {np.nanmedian(finite):.2f}, mean {np.nanmean(finite):.2f}")
            else:
                lines.append(f"{label}: n/a")
    lines.append("")
    lines.append("Motor command columns are inferred because motor command logs are missing.")
    lines.append("Latencies use movement-start synchronized time; raw start offsets are in alignment_onsets.png.")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0.03, 0.95, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_session_figures(session: Path, summary: pd.DataFrame, out_root: Path) -> list[str]:
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    figure_jobs = [
        ("summary_table.png", lambda p: fig_summary_table(summary, p)),
        ("detection_accuracy.png", lambda p: fig_detection_accuracy(summary, p)),
        ("latencies.png", lambda p: fig_latencies(summary, p)),
        ("alignment_onsets.png", lambda p: fig_alignment_onsets(summary, p)),
        ("object_tracking.png", lambda p: fig_object_tracking(session, summary, out_root, p)),
        ("motor_command_response.png", lambda p: fig_motor_command_response(session, summary, out_root, p)),
        ("algorithm_vision_vs_hand_accuracy.png", lambda p: fig_algorithm_vision_vs_hand_accuracy(summary, p)),
        ("bottom_line_latency_accuracy_summary.png", lambda p: fig_bottom_line(summary, p)),
    ]
    for name, writer in figure_jobs:
        writer(fig_dir / name)
        written.append(name)
    return written


def analyze_pair(pair: Path, out_root: Path, stride: int, save_detected_videos: bool = True) -> PairResult:
    tracking = read_tracking(pair)
    finger = str(tracking["finger"].dropna().iloc[0]) if "finger" in tracking and tracking["finger"].notna().any() else "unknown"
    warnings = []
    signals_dir = out_root / "signals"
    top, w, top_pxmm = extract_top_mediapipe(pair, stride, signals_dir / f"top_{pair.name}.csv")
    warnings += w
    virt, w = extract_virtual(pair, stride, signals_dir / f"virtual_{pair.name}.csv")
    warnings += w
    side, w, side_pxmm = extract_side(pair, stride, signals_dir / f"side_{pair.name}.csv")
    warnings += w
    command, w = reconstruct_backend_motor_commands(tracking, pair, signals_dir / f"motor_command_{pair.name}.csv")
    warnings += w

    tracking, top, virt, side, sync_meta, w = apply_movement_start_alignment(tracking, top, virt, side)
    warnings += w
    if not command.empty:
        command["sync_time_s"] = command["time_s"].astype(float) - sync_meta["tracking_movement_start_s"]
        command["movement_start_s"] = sync_meta["tracking_movement_start_s"]
        command.to_csv(signals_dir / f"motor_command_{pair.name}.csv", index=False)
    # Overwrite the per-stream CSVs after alignment so every downstream graph
    # and manual inspection has both raw time_s and synchronized sync_time_s.
    if not top.empty:
        top.to_csv(signals_dir / f"top_{pair.name}.csv", index=False)
    if not virt.empty:
        virt.to_csv(signals_dir / f"virtual_{pair.name}.csv", index=False)
    if not side.empty:
        side.to_csv(signals_dir / f"side_{pair.name}.csv", index=False)

    if save_detected_videos:
        video_dir = out_root / "detected_videos"
        if not top.empty:
            msg = render_detected_video(pair / "top_camera.mp4", top, video_dir / f"{pair.name}_top_detected.mp4", "top")
            if "skipped" in msg:
                warnings.append(msg)
        if not virt.empty:
            msg = render_detected_video(pair / "virtual_object.mp4", virt, video_dir / f"{pair.name}_virtual_object_detected.mp4", "virtual")
            if "skipped" in msg:
                warnings.append(msg)
        if not side.empty:
            msg = render_detected_video(pair / "side_camera.mp4", side, video_dir / f"{pair.name}_side_detected.mp4", "side")
            if "skipped" in msg:
                warnings.append(msg)

    res = PairResult(pair=pair.name, finger=finger, top_px_per_mm=top_pxmm, side_px_per_mm=side_pxmm, **sync_meta)
    t = tracking_time(tracking).to_numpy()
    fs = max(1.0, 30.0 / max(1, stride))

    if not top.empty:
        good = top["detected"].astype(bool) & top["mp_active_x"].notna() & top["mp_active_y"].notna()
        res.top_detection_rate = float(good.mean()) if len(top) else np.nan
        if good.sum() >= 6:
            r2, rmse, _, tracking_to_top_scale = affine_metrics_mm(
                t, tracking["active_finger_x"], tracking["active_finger_y"],
                signal_time(top).loc[good], top.loc[good, "mp_active_x"], top.loc[good, "mp_active_y"],
                fs=fs, meas_px_per_mm=top_pxmm,
            )
            res.top_hand_r2, res.top_hand_rmse_mm = r2, rmse
            lag, corr = latency_ms(t, tracking["active_finger_x"], tracking["active_finger_y"],
                                   signal_time(top).loc[good], top.loc[good, "mp_active_x"], top.loc[good, "mp_active_y"], fs=fs)
            res.top_hand_latency_ms, res.top_hand_latency_corr = lag, corr
        else:
            warnings.append("top MediaPipe detection too sparse for metrics")

    tracking_units_per_mm = np.nan
    if np.isfinite(res.top_px_per_mm) and not top.empty and top["detected"].sum() >= 6:
        _, _, _, scale = affine_metrics_mm(
            t, tracking["active_finger_x"], tracking["active_finger_y"],
            signal_time(top).loc[top["detected"].astype(bool)], top.loc[top["detected"].astype(bool), "mp_active_x"], top.loc[top["detected"].astype(bool), "mp_active_y"],
            fs=fs, meas_px_per_mm=res.top_px_per_mm,
        )
        if np.isfinite(scale) and scale > 0:
            tracking_units_per_mm = res.top_px_per_mm / scale

    if not virt.empty:
        good = virt["obj_x"].notna() & virt["obj_y"].notna()
        res.virtual_object_detection_rate = float(good.mean()) if len(virt) else np.nan
        res.virtual_finger_detection_rate = float((virt["virtual_finger_x"].notna() & virt["virtual_finger_y"].notna()).mean()) if len(virt) else np.nan
        if good.sum() >= 6:
            acc = LA.detection_accuracy_2d(t, tracking["object_x"], tracking["object_y"],
                                           signal_time(virt).loc[good].to_numpy(), virt.loc[good, "obj_x"].to_numpy(), virt.loc[good, "obj_y"].to_numpy(), fs=fs)
            res.virtual_object_r2 = acc.r2_affine
            res.virtual_object_rmse_mm = np.nan
            # Convert residual in tracking space to mm when top-derived tracking scale exists.
            if np.isfinite(tracking_units_per_mm) and tracking_units_per_mm > 0:
                # LA rmse_norm is normalized, so recompute absolute tracking residual via affine measured->truth.
                vt = signal_time(virt).loc[good]
                grid = np.arange(max(np.nanmin(t), np.nanmin(vt)), min(np.nanmax(t), np.nanmax(vt)), 1/fs)
                _, tx = LA.resample_uniform(t, tracking["object_x"], fs, grid)
                _, ty = LA.resample_uniform(t, tracking["object_y"], fs, grid)
                _, mx = LA.resample_uniform(vt, virt.loc[good, "obj_x"], fs, grid)
                _, my = LA.resample_uniform(vt, virt.loc[good, "obj_y"], fs, grid)
                n = min(len(tx), len(ty), len(mx), len(my))
                A = np.vstack([mx[:n], my[:n], np.ones(n)]).T
                target = np.vstack([tx[:n], ty[:n]]).T
                coef, *_ = np.linalg.lstsq(A, target, rcond=None)
                pred = A @ coef
                rmse_tracking = float(np.sqrt(np.mean((target - pred) ** 2)))
                res.virtual_object_rmse_mm = rmse_tracking / tracking_units_per_mm
            else:
                warnings.append("virtual object mm scale inferred from top calibration unavailable; virtual_object_rmse_mm set to NaN")
            lag, corr = latency_ms(t, tracking["object_x"], tracking["object_y"], signal_time(virt).loc[good], virt.loc[good, "obj_x"], virt.loc[good, "obj_y"], fs=fs)
            res.virtual_object_latency_ms, res.virtual_object_latency_corr = lag, corr
            lag, corr = latency_ms(t, tracking["active_finger_x"], tracking["active_finger_y"], signal_time(virt).loc[good], virt.loc[good, "obj_x"], virt.loc[good, "obj_y"], fs=fs)
            res.finger_to_virtual_object_latency_ms, res.finger_to_virtual_object_latency_corr = lag, corr
        else:
            warnings.append("virtual object detection too sparse for metrics")

    if not side.empty:
        good = side["tactor_x"].notna() & side["tactor_y"].notna()
        res.tactor_detection_rate = float(good.mean()) if len(side) else np.nan
        if good.sum() >= 6:
            r2, rmse, _, _ = affine_metrics_mm(t, tracking["active_finger_x"], tracking["active_finger_y"],
                                               signal_time(side).loc[good], side.loc[good, "tactor_x"], side.loc[good, "tactor_y"],
                                               fs=fs, meas_px_per_mm=side_pxmm)
            res.tactor_r2, res.tactor_rmse_mm = r2, rmse
            lag, corr = latency_ms(t, tracking["active_finger_x"], tracking["active_finger_y"], signal_time(side).loc[good], side.loc[good, "tactor_x"], side.loc[good, "tactor_y"], fs=fs)
            res.tactor_latency_ms, res.tactor_latency_corr = lag, corr
        else:
            warnings.append("tactor detection too sparse for metrics")
        spool_cols = [c for c in side.columns if c.endswith("_conf")]
        if spool_cols:
            vals = []
            for c in spool_cols:
                angle_col = c[:-5] + "_angle"
                if angle_col in side.columns:
                    angle = side[angle_col].astype(float)
                    vals.append(float(np.isfinite(angle).mean()))
                else:
                    vals.append(0.0)
            res.motor_spool_detection_rate = float(np.nanmean(vals)) if vals else np.nan
            mt, ma = mean_motor_angle_signal(side, time_col="sync_time_s")
            if len(mt) >= 6:
                cmd_t, cmd_sig = backend_command_signal(command)
                if len(cmd_t) >= 6:
                    lag, corr = latency_1d_ms(cmd_t, cmd_sig, mt, ma, fs=fs)
                    res.backend_command_to_motor_latency_ms, res.backend_command_to_motor_latency_corr = lag, corr
                    # Backward-compatible column name, but now it means backend-reconstructed command.
                    res.inferred_command_to_motor_latency_ms, res.inferred_command_to_motor_latency_corr = lag, corr
                else:
                    inferred = motion_signal(tracking["active_finger_x"], tracking["active_finger_y"])
                    lag, corr = latency_1d_ms(t, inferred, mt, ma, fs=fs)
                    res.inferred_command_to_motor_latency_ms, res.inferred_command_to_motor_latency_corr = lag, corr
                    warnings.append("backend motor command signal unavailable; used fallback finger-motion signal for command-to-motor latency")
                if good.sum() >= 6:
                    tactor_motion = motion_signal(side.loc[good, "tactor_x"], side.loc[good, "tactor_y"])
                    lag, corr = latency_1d_ms(mt, ma, signal_time(side).loc[good], tactor_motion, fs=fs)
                    res.motor_to_tactor_latency_ms, res.motor_to_tactor_latency_corr = lag, corr
            else:
                warnings.append("motor black-line angle signal too sparse for command/motor/tactor chain metrics")
        else:
            warnings.append("motor spool angles not available from automatic detector")

        if not command.empty and good.sum() >= 6:
            cgood = command["backend_tactor_target_x"].notna() & command["backend_tactor_target_y"].notna()
            if cgood.sum() >= 6:
                lag, corr = latency_ms(
                    command_time(command).loc[cgood],
                    command.loc[cgood, "backend_tactor_target_x"],
                    command.loc[cgood, "backend_tactor_target_y"],
                    signal_time(side).loc[good],
                    side.loc[good, "tactor_x"],
                    side.loc[good, "tactor_y"],
                    fs=fs,
                )
                res.backend_tactor_target_to_tactor_latency_ms = lag
                res.backend_tactor_target_to_tactor_latency_corr = corr
                r2, rmse, _, _ = affine_metrics_mm(
                    command_time(command).loc[cgood],
                    command.loc[cgood, "backend_tactor_target_x"],
                    command.loc[cgood, "backend_tactor_target_y"],
                    signal_time(side).loc[good],
                    side.loc[good, "tactor_x"],
                    side.loc[good, "tactor_y"],
                    fs=fs,
                    meas_px_per_mm=side_pxmm,
                )
                res.backend_tactor_target_to_tactor_r2 = r2
                res.backend_tactor_target_to_tactor_rmse_mm = rmse

    res.warnings = " | ".join(dict.fromkeys(warnings))
    plot_pair(pair.name, tracking, top, virt, side, command, out_root / "figures")
    windowed_latency_metrics(pair.name, tracking, top, virt, side, command, fs=fs, out_csv=out_root / "csv" / f"{pair.name}_latency_over_time.csv", fig_dir=out_root / "figures")
    return res


def analyze_session(session: Path, results_root: Path, screen_recording: Path | None, stride: int, rebuild: bool, save_detected_videos: bool = True) -> Path:
    session = session.resolve()
    out = results_root / session.name
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "signals").mkdir(parents=True, exist_ok=True)
    if save_detected_videos:
        (out / "detected_videos").mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(session)
    warnings = split_screen_recording(session, screen_recording, pairs, rebuild=rebuild)
    rows = []
    for pair in pairs:
        rows.append(asdict(analyze_pair(pair, out, stride, save_detected_videos=save_detected_videos)))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "csv" / "three_video_pair_summary.csv", index=False)
    written_figures = render_session_figures(session, summary, out) if not summary.empty else []
    with (out / "run_info.txt").open("w", encoding="utf-8") as f:
        f.write("Three-video validation analysis\n")
        f.write(f"session: {session}\n")
        f.write(f"stride: {stride}\n")
        f.write("pair filter: pair_005 and above ignored always\n")
        f.write(f"detected videos: {'saved in detected_videos/' if save_detected_videos else 'disabled'}\n")
        f.write("timing: raw time_s is preserved; metrics/plots use sync_time_s after automatic movement-start alignment.\n")
        f.write("sync marks: top uses MediaPipe hand start; virtual uses white virtual-finger start when available; side uses motor-angle start when available.\n")
        f.write(f"figures: {', '.join(written_figures)}\n")
        f.write("IMPORTANT: motor command logs are RECONSTRUCTED from tracking.csv through the same backend.py mapping path; they are not measured firmware/UDP logs.\n")
        f.write("backend command path: object displacement -> map_object_displacement_to_tactor -> MotorController(IK, move_factor=7) -> motor positions.\n")
        for w in warnings:
            f.write(f"WARNING: {w}\n")
        if not summary.empty:
            for w in summary["warnings"].dropna().astype(str):
                if w:
                    f.write(f"WARNING: {w}\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze three-video validation data with mm/ms metrics.")
    ap.add_argument("--session", required=True, type=Path, help="Session directory containing pair_### folders")
    ap.add_argument("--screen-recording", type=Path, default=None, help="Full-session virtual-object screen recording to split")
    ap.add_argument("--results-root", type=Path, default=HERE / "Results", help="Where to save analysis outputs")
    ap.add_argument("--stride", type=int, default=3, help="Frame stride for offline analysis")
    ap.add_argument("--rebuild", action="store_true", help="Regenerate virtual_object.mp4 clips even if present")
    ap.add_argument("--no-detected-videos", action="store_true", help="Skip full-length annotated detection videos")
    args = ap.parse_args(argv)
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    out = analyze_session(args.session, args.results_root, args.screen_recording, args.stride, args.rebuild, save_detected_videos=not args.no_detected_videos)
    print(f"Saved three-video analysis to {out}")
    print(f"Summary: {out / 'csv' / 'three_video_pair_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
