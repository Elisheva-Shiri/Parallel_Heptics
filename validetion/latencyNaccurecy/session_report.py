"""Orchestration layer that turns the cached per-pair signals + tracking.csv +
motor_commands.txt into the per-finger latency/accuracy summary table.

It glues together:
  * region_signals  - the video-derived signals (cached by build_signals.py)
  * tracking.csv    - the simulation ground truth (object & finger positions)
  * motor_commands  - the ESP32 bridge log (commanded vs applied motor positions)
  * latency_analysis- the cross-correlation / accuracy primitives

The latency "chain" we report, and what each link physically means:

  hand --(control loop)--> object(sim) --(render+capture)--> object(video)
   |                                                              |
   |                                                              v
   +----------------(haptic actuation)----------------------> tactor(video)
                                                                  ^
                              motor command --(firmware)--> motor/spool angle

Because the supplied motor log does not overlap the videos, the motor<->video
links are reported as the firmware command->ack latency (always computable) and
the per-pair overlap flag is recorded so the gap is explicit, never faked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import latency_analysis as LA
from motor_commands import MotorCommandLog, empty_log, find_motor_log, parse_motor_commands


def _motion_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance of each sample from the series median (a frame-invariant 1D
    motion signal for a 2D trajectory that may move in any of 8 directions)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.hypot(x - np.nanmedian(x), y - np.nanmedian(y))


# Orientation of each side-camera region relative to the finger-tracking (top
# camera) frame, which is DEFINED as 0 deg (provided by the experimenter):
#   vision (the monitor) ......... 180 deg
#   tactor (with the finger on) .. 135 deg ccw   (at rest, no finger: 90 deg)
# Rotating a region's detected (x, y) by -orientation brings its motion into the
# finger frame, so the DIRECTION of movement can be compared with the hand (not
# just the timing). Adjust here if the rig geometry changes.
ORIENTATION_DEG = {"finger": 0.0, "vision": 180.0, "tactor": 135.0, "tactor_rest": 90.0}


def _to_finger_frame(x, y, orient_deg: float):
    """Rotate (x, y) by -orient_deg about its own median -> centred finger-frame."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc, yc = x - np.nanmedian(x), y - np.nanmedian(y)
    th = np.radians(-orient_deg)
    cs, sn = np.cos(th), np.sin(th)
    return xc * cs - yc * sn, xc * sn + yc * cs


def _direction_corr(t_h, hx, hy, t_o, ox, oy, orient_deg: float, fs: float = 15.0) -> float:
    """Signed agreement between hand and a region's movement DIRECTION.

    Both are brought into the finger frame; we take the hand's principal motion
    axis and project each onto it, then correlate. +1 = moves the same way as the
    hand, -1 = opposite, ~0 = unrelated direction. Complements the magnitude
    latency (which is direction-blind)."""
    hxr, hyr = _to_finger_frame(hx, hy, ORIENTATION_DEG["finger"])
    oxr, oyr = _to_finger_frame(ox, oy, orient_deg)
    t0 = max(np.nanmin(t_h), np.nanmin(t_o))
    t1 = min(np.nanmax(t_h), np.nanmax(t_o))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.nan
    grid = np.arange(t0, t1, 1.0 / fs)
    _, hxg = LA.resample_uniform(t_h, hxr, fs, grid)
    _, hyg = LA.resample_uniform(t_h, hyr, fs, grid)
    _, oxg = LA.resample_uniform(t_o, oxr, fs, grid)
    _, oyg = LA.resample_uniform(t_o, oyr, fs, grid)
    n = min(len(hxg), len(hyg), len(oxg), len(oyg))
    if n < 4:
        return np.nan
    H = np.vstack([hxg[:n], hyg[:n]]).T
    cov = np.cov(H, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return np.nan
    _, evec = np.linalg.eigh(cov)
    u = evec[:, -1]                       # hand principal motion axis
    h1 = H @ u
    o1 = np.vstack([oxg[:n], oyg[:n]]).T @ u
    if h1.std() == 0 or o1.std() == 0:
        return np.nan
    return float(np.corrcoef(h1, o1)[0, 1])


def load_pair(out_dir: Path, session_dir: Path, pair: str) -> dict:
    """Load cached signals + tracking for one pair, aligned in absolute time."""
    sig = pd.read_csv(out_dir / f"signals_{pair}.csv")
    if "timestamp" in sig:
        sig["timestamp"] = pd.to_datetime(sig["timestamp"])
        sig["t"] = LA.to_seconds(sig["timestamp"])
    else:
        sig["t"] = sig["time_s"]
    trk = pd.read_csv(session_dir / pair / "tracking.csv")
    trk["timestamp"] = pd.to_datetime(trk["timestamp"])
    # Shared zero so signal-time and tracking-time are on the same axis.
    t0 = min(sig["timestamp"].min(), trk["timestamp"].min())
    sig["t"] = (sig["timestamp"] - t0).dt.total_seconds()
    trk["t"] = (trk["timestamp"] - t0).dt.total_seconds()
    return {"signals": sig, "tracking": trk, "t0": t0}


def _hand_to_motor_latency(trk: pd.DataFrame, sig: pd.DataFrame, fs: float = 15.0,
                           min_corr: float = 0.25):
    """Per-trial finger->motor latency, averaged over the spools.

    Over the *interaction window* (where the finger is tracked and touching), the
    finger pushes the object repeatedly and each spool turns in response. We
    cross-correlate the finger's motion speed against each spool's angular speed
    (which averages the lag over all the push cycles), keep the spools whose
    correlation is reliable, and return the mean lag. Restricting to the
    interaction window removes the startup period (finger not yet tracked) that
    would otherwise corrupt a single-onset estimate. Returns
    ``(latency_ms, mean_corr, n_spools_used)``."""
    spool_cols = [c for c in sig.columns if c.endswith("_angle_unwrapped")]
    if not spool_cols:
        return np.nan, np.nan, 0
    inter = trk["interacting"].astype(str).isin(["True", "1", "1.0"])
    moved = (trk["active_finger_x"] != 0) | (trk["active_finger_y"] != 0)
    mask = inter & moved
    if int(mask.sum()) < 50:
        return np.nan, np.nan, 0
    tlo, thi = float(trk["t"][mask].min()), float(trk["t"][mask].max())
    tm = (trk["t"] >= tlo) & (trk["t"] <= thi)
    sm = (sig["t"] >= tlo) & (sig["t"] <= thi)
    hx, hy = trk["active_finger_x"], trk["active_finger_y"]
    hand = np.hypot(hx - hx[mask].median(), hy - hy[mask].median())
    ht, hv = trk["t"][tm].to_numpy(), hand[tm].to_numpy()
    s_t = sig["t"][sm].to_numpy()
    lags, corrs = [], []
    for c in spool_cols:
        r = LA.estimate_lag(ht, hv, s_t, sig[c][sm].to_numpy(), fs=fs, max_lag_s=1.0)
        if np.isfinite(r.peak_corr) and r.peak_corr > min_corr and abs(r.lag_ms) < 1000:
            lags.append(r.lag_ms)
            corrs.append(r.peak_corr)
    if not lags:
        return np.nan, np.nan, 0
    return float(np.mean(lags)), float(np.mean(corrs)), len(lags)


def per_pair_metrics(
    pair: str, finger: str, sig: pd.DataFrame, trk: pd.DataFrame,
    log: MotorCommandLog, fs: float = 15.0,
) -> dict:
    """Compute detection accuracy + the latency chain for a single pair."""
    spool_cols = [c for c in sig.columns if c.endswith("_angle_unwrapped")]

    # --- ground-truth (tracking) motion signals --------------------------- #
    trk_t = trk["t"].to_numpy()
    obj_truth = _motion_2d(trk["object_x"], trk["object_y"])
    hand = _motion_2d(trk["active_finger_x"], trk["active_finger_y"])

    # --- video motion signals --------------------------------------------- #
    sig_t = sig["t"].to_numpy()
    obj_vid = _motion_2d(sig["obj_x"], sig["obj_y"])
    tac = _motion_2d(sig["tactor_x"], sig["tactor_y"])
    spool = sig[spool_cols[0]].to_numpy() if spool_cols else np.full(len(sig), np.nan)

    m: dict = {"pair": pair, "finger": finger}

    # 1) VISION detection accuracy: video object vs tracking object ---------- #
    # 2-D affine fit because the camera views the monitor rotated+mirrored.
    m["vision_detection_rate"] = float(sig["obj_x"].notna().mean())
    acc = LA.detection_accuracy_2d(
        trk_t, trk["object_x"].to_numpy(), trk["object_y"].to_numpy(),
        sig_t, sig["obj_x"].to_numpy(), sig["obj_y"].to_numpy(), fs=fs)
    m["vision_accuracy_R2"] = acc.r2_affine
    m["vision_accuracy_r"] = acc.pearson_r
    # display latency: sim object -> on-screen object (render+capture)
    m["display_latency_ms"], m["display_corr"] = LA.gated_lag(
        LA.estimate_lag(trk_t, obj_truth, sig_t, obj_vid, fs=fs, max_lag_s=1.0))

    # 2) hand -> vision (object on screen) --------------------------------- #
    m["hand_to_vision_latency_ms"], m["hand_to_vision_corr"] = LA.gated_lag(
        LA.estimate_lag(trk_t, hand, sig_t, obj_vid, fs=fs, max_lag_s=2.0))
    # direction agreement in the finger frame (vision rotated 180 deg)
    m["hand_to_vision_dir_corr"] = _direction_corr(
        trk_t, trk["active_finger_x"].to_numpy(), trk["active_finger_y"].to_numpy(),
        sig_t, sig["obj_x"].to_numpy(), sig["obj_y"].to_numpy(),
        ORIENTATION_DEG["vision"], fs=fs)

    # 3) TACTOR movement: detection + hand -> tactor latency --------------- #
    m["tactor_detection_rate"] = float(sig["tactor_found"].mean())
    m["hand_to_tactor_latency_ms"], m["hand_to_tactor_corr"] = LA.gated_lag(
        LA.estimate_lag(trk_t, hand, sig_t, tac, fs=fs, max_lag_s=2.0))
    # direction agreement in the finger frame (tactor-with-finger rotated 135 deg)
    m["hand_to_tactor_dir_corr"] = _direction_corr(
        trk_t, trk["active_finger_x"].to_numpy(), trk["active_finger_y"].to_numpy(),
        sig_t, sig["tactor_x"].to_numpy(), sig["tactor_y"].to_numpy(),
        ORIENTATION_DEG["tactor"], fs=fs)

    # 4) vision -> motor (object on screen -> spool angle), same video clock - #
    if spool_cols and np.isfinite(spool).sum() > 4:
        m["vision_to_motor_latency_ms"], m["vision_to_motor_corr"] = LA.gated_lag(
            LA.estimate_lag(sig_t, obj_vid, sig_t, spool, fs=fs, max_lag_s=1.5))
        m["motor_spool_detection_rate"] = float(np.isfinite(sig[spool_cols[0]]).mean())
    else:
        m["vision_to_motor_latency_ms"] = np.nan
        m["vision_to_motor_corr"] = np.nan
        m["motor_spool_detection_rate"] = 0.0

    # 4b) HAND -> MOTOR onset latency (per trial): over the interaction window,
    # the lag from finger motion to each spool's motion, averaged over the 3
    # motors. This is the physical actuation latency the cross-correlation above
    # could not pin down on the noisy on-screen-object signal.
    (m["hand_to_motor_latency_ms"], m["hand_to_motor_corr"],
     m["hand_to_motor_n_spools"]) = _hand_to_motor_latency(trk, sig, fs=fs)

    # 5) motor command coverage for THIS pair ------------------------------ #
    pt0, pt1 = trk["timestamp"].min(), trk["timestamp"].max()
    m["motor_log_overlap"] = LA.overlaps(pt0, pt1, log.t_start, log.t_end)

    # 6) true motor latency: commanded position -> video spool angle. Only
    # computable when the log overlaps the video (it does not in this dataset,
    # so this is NaN); auto-computes once a synchronised recording is supplied.
    ack = log.acknowledged()
    if m["motor_log_overlap"] and not ack.empty and spool_cols and "timestamp" in sig:
        epoch = min(sig["timestamp"].min(), ack["timestamp"].min())
        cmd_t = (ack["timestamp"] - epoch).dt.total_seconds().to_numpy()
        vid_t = (sig["timestamp"] - epoch).dt.total_seconds().to_numpy()
        cv = LA.command_vs_angle_latency(cmd_t, ack[log.motor_columns[0]].to_numpy(),
                                         vid_t, spool, fs=fs)
        m["motor_cmd_to_video_latency_ms"] = cv.lag_ms
    else:
        m["motor_cmd_to_video_latency_ms"] = np.nan
    return m


def build_summary(session_dir: str | Path, out_dir: str | Path | None = None,
                  fs: float = 15.0) -> dict:
    """Full pipeline: load every pair, compute metrics, assemble the table.

    Returns {'summary': DataFrame, 'cmd_ack': DataFrame, 'log': MotorCommandLog,
             'pairs': {pair: loaded dict}} so the notebook can also plot details.
    """
    session_dir = Path(session_dir)
    out_dir = Path(out_dir) if out_dir else session_dir / "analysis_output"
    meta = pd.read_csv(out_dir / "sessions_meta.csv")
    log_path = find_motor_log(session_dir)
    log = parse_motor_commands(log_path) if log_path is not None else empty_log()
    cmd_ack = LA.command_to_ack_latency(log.commanded(), log.acknowledged(), log.motor_columns)
    session_cmd_ack_ms = float(cmd_ack["latency_ms"].median()) if not cmd_ack.empty else np.nan

    rows, pairs = [], {}
    for _, r in meta.iterrows():
        pair = r["pair"]
        loaded = load_pair(out_dir, session_dir, pair)
        pairs[pair] = loaded
        mm = per_pair_metrics(pair, r["finger"], loaded["signals"], loaded["tracking"], log, fs=fs)
        # Motor latency: the firmware command->ack round-trip is the only motor
        # latency computable here (the log never overlaps the video). It is a
        # session-wide property, so every finger gets the same session median.
        mm["motor_cmd_to_ack_latency_ms"] = session_cmd_ack_ms
        rows.append(mm)
    summary = pd.DataFrame(rows)
    return {"summary": summary, "cmd_ack": cmd_ack, "log": log, "pairs": pairs}


# Columns grouped the way the user asked: per finger, the accuracy & latency of
# (a) detection/vision, (b) tactor movement, (c) motors.
SUMMARY_VIEW = [
    "finger", "pair",
    # detection (vision)
    "vision_detection_rate", "vision_accuracy_R2",
    "hand_to_vision_latency_ms", "hand_to_vision_dir_corr",
    # tactor
    "tactor_detection_rate", "hand_to_tactor_corr",
    "hand_to_tactor_latency_ms", "hand_to_tactor_dir_corr",
    # motors
    "motor_spool_detection_rate",
    "hand_to_motor_latency_ms", "hand_to_motor_corr", "hand_to_motor_n_spools",
    "motor_cmd_to_ack_latency_ms", "motor_log_overlap",
]
