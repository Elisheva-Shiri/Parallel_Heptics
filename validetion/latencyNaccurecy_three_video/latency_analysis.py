"""Latency & accuracy primitives for comparing the side-camera signals against
``tracking.csv`` and ``motor_commands.txt``.

The three things we measure:

1. **Detection accuracy** - how well a video-derived signal reproduces a
   "ground truth" signal (e.g. the on-screen object position vs the object
   position the simulation logged in tracking.csv).  Quantified by Pearson r
   and by R^2 / RMSE after a best-fit affine map (the two live in different
   pixel frames, so we fit scale+offset before scoring the residual).

2. **Latency between two time-series** - estimated by the lag that maximises
   their normalised cross-correlation.  We correlate *speed* (|d/dt|) rather
   than raw position, because onset of motion is a sharper, frame-rate-robust
   alignment feature than absolute position and is invariant to the different
   coordinate frames/baselines of the two signals.

3. **Command->ack latency** - directly from the bridge log: for each commanded
   position (UDP_IN) the time until the matching ESP32 acknowledgement
   (SERIAL_IN).  This is the only motor latency that is computable when the log
   does not temporally overlap the video.

WHY CROSS-CORRELATION OF SPEED
------------------------------
Both signals are sampled on (possibly different) clocks.  We resample both onto
a common uniform grid, z-score them, then slide one against the other.  The
peak-correlation lag is the latency; its sign tells direction (positive lag =>
the second signal follows the first).  Using speed makes the estimate robust to
slow drift and to the fact that position offsets differ between frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Resampling / signal prep
# --------------------------------------------------------------------------- #

def to_seconds(ts: pd.Series | np.ndarray) -> np.ndarray:
    """Convert a datetime series to seconds-from-start (float)."""
    t = pd.to_datetime(pd.Series(ts)).to_numpy()
    t0 = t[0]
    return (t - t0) / np.timedelta64(1, "s")


def resample_uniform(
    t: np.ndarray, y: np.ndarray, fs: float, t_grid: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate (t, y) onto a uniform grid at sample rate ``fs`` (Hz).

    NaNs in ``y`` are dropped before interpolation; if fewer than 2 valid
    samples remain, returns an empty grid.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if t.size < 2:
        return np.array([]), np.array([])
    if t_grid is None:
        t_grid = np.arange(t.min(), t.max(), 1.0 / fs)
    yg = np.interp(t_grid, t, y)
    return t_grid, yg


def speed(y: np.ndarray) -> np.ndarray:
    """Absolute first difference (motion magnitude); same length as input."""
    d = np.abs(np.diff(y, prepend=y[:1]))
    return d


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


# --------------------------------------------------------------------------- #
# Cross-correlation latency
# --------------------------------------------------------------------------- #

@dataclass
class LagResult:
    lag_s: float            # +ve => signal b follows signal a
    peak_corr: float        # normalised correlation at the peak (-1..1)
    n: int                  # samples used
    fs: float               # grid sample rate

    @property
    def lag_ms(self) -> float:
        return self.lag_s * 1000.0


def estimate_lag(
    ta: np.ndarray, a: np.ndarray,
    tb: np.ndarray, b: np.ndarray,
    fs: float = 30.0,
    max_lag_s: float = 2.0,
    use_speed: bool = True,
) -> LagResult:
    """Estimate the lag at which ``b`` best matches ``a`` (b follows a if +).

    Both series are resampled to a shared uniform grid (intersection of their
    time spans), optionally converted to speed, z-scored, then cross-correlated.
    """
    # Shared time span on a common grid.
    t0 = max(np.nanmin(ta), np.nanmin(tb))
    t1 = min(np.nanmax(ta), np.nanmax(tb))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 - t0 < 2.0 / fs:
        return LagResult(np.nan, np.nan, 0, fs)
    grid = np.arange(t0, t1, 1.0 / fs)
    _, ag = resample_uniform(ta, a, fs, grid)
    _, bg = resample_uniform(tb, b, fs, grid)
    if ag.size < 4 or bg.size < 4:
        return LagResult(np.nan, np.nan, int(min(ag.size, bg.size)), fs)
    if use_speed:
        ag, bg = speed(ag), speed(bg)
    ag, bg = _zscore(ag), _zscore(bg)

    max_lag = int(round(max_lag_s * fs))
    max_lag = min(max_lag, len(ag) - 2)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.empty(len(lags), float)
    n = len(ag)
    for i, L in enumerate(lags):
        if L >= 0:
            x, y = ag[: n - L], bg[L:]
        else:
            x, y = ag[-L:], bg[: n + L]
        corrs[i] = np.corrcoef(x, y)[0, 1] if x.size > 2 else np.nan
    if np.all(np.isnan(corrs)):
        return LagResult(np.nan, np.nan, n, fs)
    k = int(np.nanargmax(corrs))
    # Sub-sample peak via parabolic interpolation of the 3 points around the
    # peak, so the lag is not quantised to the 1/fs grid step.
    delta = 0.0
    if 0 < k < len(corrs) - 1:
        cm, c0, cp = corrs[k - 1], corrs[k], corrs[k + 1]
        denom = cm - 2 * c0 + cp
        if np.isfinite(denom) and denom != 0:
            delta = float(np.clip(0.5 * (cm - cp) / denom, -0.5, 0.5))
    return LagResult(lag_s=(lags[k] + delta) / fs, peak_corr=float(corrs[k]), n=n, fs=fs)


# --------------------------------------------------------------------------- #
# Detection accuracy (video signal vs logged ground truth)
# --------------------------------------------------------------------------- #

@dataclass
class AccuracyResult:
    pearson_r: float        # raw correlation of the two series (on shared grid)
    r2_affine: float        # R^2 after best-fit scale+offset
    rmse_norm: float        # RMSE of residual / std(truth), after affine fit
    n: int


# --------------------------------------------------------------------------- #
# Command -> ack latency (from the bridge log)
# --------------------------------------------------------------------------- #

def command_to_ack_latency(
    commanded: pd.DataFrame, acknowledged: pd.DataFrame, motor_cols: list[str],
    tolerance_s: float = 0.5,
) -> pd.DataFrame:
    """Match each acknowledgement to the most recent commanded position with the
    same motor targets and return the per-event latency (ms).

    For each SERIAL_IN row we look back for the last UDP_IN row whose (m0,m1,m2)
    equals the acknowledged values, within ``tolerance_s``.  Latency = ack_time -
    command_time.  This isolates the firmware/serial round-trip.
    """
    if commanded.empty or acknowledged.empty:
        return pd.DataFrame(columns=["ack_time", "latency_ms", *motor_cols])
    cmd = commanded.sort_values("timestamp").reset_index(drop=True)
    cmd_t = cmd["timestamp"].to_numpy()
    rows = []
    for _, ack in acknowledged.iterrows():
        at = ack["timestamp"]
        vals = tuple(ack[c] for c in motor_cols)
        # candidates: commands at or before the ack, within tolerance
        lo = at - pd.Timedelta(seconds=tolerance_s)
        m = (cmd_t <= np.datetime64(at)) & (cmd_t >= np.datetime64(lo))
        cand = cmd[m]
        match = cand[np.all([cand[c] == v for c, v in zip(motor_cols, vals)], axis=0)] if not cand.empty else cand
        if not match.empty:
            ct = match["timestamp"].iloc[-1]
            rows.append({
                "ack_time": at,
                "latency_ms": (at - ct).total_seconds() * 1000.0,
                **{c: ack[c] for c in motor_cols},
            })
    return pd.DataFrame(rows)


def detection_accuracy_2d(
    t_truth: np.ndarray, truth_x: np.ndarray, truth_y: np.ndarray,
    t_meas: np.ndarray, meas_x: np.ndarray, meas_y: np.ndarray,
    fs: float = 30.0,
) -> AccuracyResult:
    """2-D version that fits a full affine map (meas_x, meas_y) -> (truth_x, truth_y).

    The side camera views the monitor *rotated and mirrored*, so a per-axis
    comparison understates detection quality.  A 2-D affine (6 params: rotation,
    scale, shear, offset) removes that nuisance transform and scores the residual
    over both coordinates jointly.  ``r2_affine`` is the joint R^2; ``pearson_r``
    is the correlation of the matched motion magnitudes.
    """
    t0 = max(np.nanmin(t_truth), np.nanmin(t_meas))
    t1 = min(np.nanmax(t_truth), np.nanmax(t_meas))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return AccuracyResult(np.nan, np.nan, np.nan, 0)
    grid = np.arange(t0, t1, 1.0 / fs)
    _, tx = resample_uniform(t_truth, truth_x, fs, grid)
    _, ty = resample_uniform(t_truth, truth_y, fs, grid)
    _, mx = resample_uniform(t_meas, meas_x, fs, grid)
    _, my = resample_uniform(t_meas, meas_y, fs, grid)
    n = min(len(tx), len(ty), len(mx), len(my))
    if n < 6:
        return AccuracyResult(np.nan, np.nan, np.nan, n)
    tx, ty, mx, my = tx[:n], ty[:n], mx[:n], my[:n]
    A = np.vstack([mx, my, np.ones(n)]).T           # [n,3]
    target = np.vstack([tx, ty]).T                  # [n,2]
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / n))
    truth_mag = np.hypot(tx - tx.mean(), ty - ty.mean())
    meas_mag = np.hypot(mx - mx.mean(), my - my.mean())
    r = float(np.corrcoef(truth_mag, meas_mag)[0, 1])
    norm = float(np.sqrt(ss_tot / n))
    return AccuracyResult(pearson_r=r, r2_affine=r2,
                          rmse_norm=rmse / norm if norm > 0 else np.nan, n=n)


def gated_lag(res: "LagResult", min_corr: float = 0.25) -> tuple[float, float]:
    """Return (lag_ms, corr) but NaN the lag if the peak correlation is too weak
    to trust - a low correlation means the cross-correlation peak is noise, not a
    real delay (e.g. a barely-moving signal)."""
    if not np.isfinite(res.peak_corr) or res.peak_corr < min_corr:
        return np.nan, res.peak_corr
    return res.lag_ms, res.peak_corr


def overlaps(t_start_a, t_end_a, t_start_b, t_end_b) -> bool:
    """True if two [start,end] datetime intervals overlap at all."""
    if any(x is None for x in (t_start_a, t_end_a, t_start_b, t_end_b)):
        return False
    return (t_start_a <= t_end_b) and (t_start_b <= t_end_a)


def command_vs_angle_latency(
    cmd_times: np.ndarray, cmd_pos: np.ndarray,
    vid_times: np.ndarray, vid_angle: np.ndarray,
    fs: float = 30.0, max_lag_s: float = 1.0, min_corr: float = 0.3,
) -> LagResult:
    """Latency from a commanded motor position to the spool angle seen on video.

    This is the *true* motor latency the project wants. It is only meaningful
    when the command log and the video overlap in absolute time (start the
    bridge logging before recording!). ``cmd_times``/``vid_times`` are seconds on
    a *shared* clock. Returns a gated LagResult (NaN lag if correlation is weak
    or the windows do not overlap), so it stays honest until good data exists.
    """
    if cmd_times.size < 2 or vid_times.size < 2:
        return LagResult(np.nan, np.nan, 0, fs)
    if not overlaps(np.nanmin(cmd_times), np.nanmax(cmd_times),
                    np.nanmin(vid_times), np.nanmax(vid_times)):
        return LagResult(np.nan, np.nan, 0, fs)
    res = estimate_lag(cmd_times, cmd_pos, vid_times, vid_angle,
                       fs=fs, max_lag_s=max_lag_s, use_speed=True)
    lag_ms, corr = gated_lag(res, min_corr=min_corr)
    return LagResult(lag_s=lag_ms / 1000.0 if np.isfinite(lag_ms) else np.nan,
                     peak_corr=corr, n=res.n, fs=fs)
