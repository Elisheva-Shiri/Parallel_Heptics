"""Center-to-side probing analysis for tracking-derived kinematics.

This analysis counts how many times each participant moves from the center area
to a side area during each stiffness segment. It also provides trial-level
perception-action direction success analyses. Outputs are written under
``analysis/success_factors/results/probing`` by default.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import math
from pathlib import Path
import sys
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
for _support_dir in (_ANALYSIS_DIR / "psychophysics",):
    _support_dir_str = str(_support_dir)
    if _support_dir.exists() and _support_dir_str not in sys.path:
        sys.path.insert(0, _support_dir_str)

try:
    from analysis.group_comparisons import (
        add_experiment_group_columns,
        expand_analysis_scopes,
        compute_analysis_scope_tables,
        compute_group_comparison_tables,
        compute_setup_factor_tables,
        ANALYSIS_SCOPE_COLUMN,
        ANALYSIS_SCOPE_VALUE_COLUMN,
        EXPERIMENT_GROUP_COLUMN,
        EXPERIMENT_GROUP_ORDER,
        normalize_experiment_group,
    )
    from analysis.scope_plots import save_scope_summary_plots
    from analysis.Kinematics import kinematics_analysis as ka
    from analysis.psychophysics import twoafc_psychophysics as pp
except ModuleNotFoundError:  # pragma: no cover - supports running from analysis subfolders
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from analysis.group_comparisons import (
        add_experiment_group_columns,
        expand_analysis_scopes,
        compute_analysis_scope_tables,
        compute_group_comparison_tables,
        compute_setup_factor_tables,
        ANALYSIS_SCOPE_COLUMN,
        ANALYSIS_SCOPE_VALUE_COLUMN,
        EXPERIMENT_GROUP_COLUMN,
        EXPERIMENT_GROUP_ORDER,
        normalize_experiment_group,
    )
    from analysis.scope_plots import save_scope_summary_plots
    from analysis.Kinematics import kinematics_analysis as ka
    from analysis.psychophysics import twoafc_psychophysics as pp

try:  # SciPy is available in the analysis environment but remains optional.
    from scipy import stats as scipy_stats
except ModuleNotFoundError:  # pragma: no cover - p-values become unavailable without SciPy.
    scipy_stats = None

DEFAULT_CENTER_RADIUS_PX = 25.0
DEFAULT_SIDE_RADIUS_PX = 80.0
DEFAULT_MIN_PROBE_DURATION_S = 0.05
DIRECTION_LABELS_8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


TRIAL_KEYS = ["subject_id", "trial_index_raw", "stiffness_segment_id"]
SAMPLE_COLUMNS = [
    "subject_id",
    "subject_group",
    EXPERIMENT_GROUP_COLUMN,
    "trial_index_raw",
    "pair_number",
    "finger_condition",
    "comparison_value",
    "standard_value",
    "signed_stiffness_delta",
    "correct_response",
    "answer_code",
    "stiffness_value",
    "stiffness_segment_id",
    "stiffness_order_in_trial",
    "time_s",
    "trial_time_fraction",
    "stiffness_time_s",
    "stiffness_time_fraction",
    "x_centered_px",
    "y_centered_px",
    "r_center_px",
    "position_angle_deg",
    "speed_px_s",
    "acceleration_px_s2",
    "jerk_px_s3",
    "radial_velocity_px_s",
    "interacting_bool",
]


def save_csv(df: pd.DataFrame, output_root: Path, name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / name
    df.to_csv(path, index=False)
    return path


def compute_order_effects(clean_trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order/fatigue response-drift diagnostics now owned by probing analysis."""
    return pp.compute_order_effects(clean_trials)


def save_order_effects_outputs(
    output_root: Path,
    clean_trials: pd.DataFrame,
    *,
    fig_dpi: int = 160,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    """Save order-effect CSVs and figure in the probing analysis tree."""
    summary, binned = compute_order_effects(clean_trials)
    save_csv(summary, output_root, "order_effects_summary.csv")
    save_csv(binned, output_root, "order_effects_binned.csv")
    fig_path = pp._save_order_effects_plot(
        binned,
        out_path=output_root / "figures" / "order_effects" / "order_effects.png",
        title="Order effects: response drift over trial order",
        fig_dpi=fig_dpi,
    )
    return summary, binned, fig_path


def _sem(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def _mean_ci95_lower(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(x.mean() - 1.96 * _sem(x)) if len(x) > 1 else float(x.mean())


def _mean_ci95_upper(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(x.mean() + 1.96 * _sem(x)) if len(x) > 1 else float(x.mean())


def _wilson_ci95_bounds(s: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan, np.nan
    n = float(len(x))
    phat = float(x.mean())
    z = 1.96
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _wilson_ci95_lower(s: pd.Series) -> float:
    return _wilson_ci95_bounds(s)[0]


def _wilson_ci95_upper(s: pd.Series) -> float:
    return _wilson_ci95_bounds(s)[1]


def _add_log_backtransform_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reverse-transformed means/CIs for log1p probing columns."""
    out = df.copy()
    for raw_col in ["probe_count", "probe_rate_per_s", "center_visit_count", "path_length_px"]:
        log_mean = f"mean_log1p_{raw_col}"
        if log_mean in out:
            out[f"geomean_{raw_col}_from_log1p"] = np.expm1(out[log_mean])
        log_lo = f"log1p_{raw_col}_ci95_lower"
        log_hi = f"log1p_{raw_col}_ci95_upper"
        if log_lo in out and log_hi in out:
            out[f"{raw_col}_log_ci95_lower_backtransformed"] = np.expm1(out[log_lo])
            out[f"{raw_col}_log_ci95_upper_backtransformed"] = np.expm1(out[log_hi])
    return out


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    mask = xx.notna() & yy.notna()
    if mask.sum() < 3 or xx[mask].nunique() < 2 or yy[mask].nunique() < 2:
        return np.nan
    return float(np.corrcoef(xx[mask], yy[mask])[0, 1])


def _direction_label(angle_deg: float) -> str:
    if not np.isfinite(angle_deg):
        return "unknown"
    # 0 deg is +x/right. Convert to 8 compass sectors centered on E.
    idx = int(np.floor(((angle_deg + 22.5) % 360.0) / 45.0))
    return DIRECTION_LABELS_8[idx % len(DIRECTION_LABELS_8)]


def _mode_or_unknown(s: pd.Series) -> str:
    vals = s.dropna().astype(str)
    if vals.empty:
        return "unknown"
    return str(vals.value_counts().index[0])


def _numeric_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _nanmean_or_nan(values: list[Any]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan


def _nanmedian_or_nan(values: list[Any]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan


def _probe_path_metrics(
    segment: pd.DataFrame,
    *,
    start_time_s: float,
    end_time_s: float,
    side_time_s: float,
) -> dict[str, float]:
    """Return full center->side->center path, jerk, and straightness metrics."""
    if segment.empty or not (np.isfinite(start_time_s) and np.isfinite(end_time_s)):
        return {
            "probe_path_length_px": np.nan,
            "probe_ideal_out_back_distance_px": np.nan,
            "probe_straightness_score": np.nan,
            "probe_mean_jerk_px_s3": np.nan,
            "probe_median_jerk_px_s3": np.nan,
            "probe_max_jerk_px_s3": np.nan,
        }
    cols = ["time_s", "x_centered_px", "y_centered_px", "r_center_px", "jerk_px_s3"]
    available = [c for c in cols if c in segment.columns]
    d = segment.loc[
        (pd.to_numeric(segment["time_s"], errors="coerce") >= start_time_s)
        & (pd.to_numeric(segment["time_s"], errors="coerce") <= end_time_s),
        available,
    ].copy()
    if "time_s" in d:
        d = d.sort_values("time_s")

    path_length = np.nan
    ideal_distance = np.nan
    straightness = np.nan
    if {"x_centered_px", "y_centered_px"}.issubset(d.columns) and len(d) >= 2:
        x = pd.to_numeric(d["x_centered_px"], errors="coerce")
        y = pd.to_numeric(d["y_centered_px"], errors="coerce")
        valid = x.notna() & y.notna()
        xy = pd.DataFrame({"x": x[valid], "y": y[valid]})
        if len(xy) >= 2:
            step = np.hypot(xy["x"].diff(), xy["y"].diff())
            path_length = float(step.iloc[1:].sum())
            side_idx = (
                pd.to_numeric(d.loc[valid, "time_s"], errors="coerce") - side_time_s
            ).abs().idxmin()
            start_xy = xy.iloc[0].to_numpy(dtype=float)
            end_xy = xy.iloc[-1].to_numpy(dtype=float)
            side_xy = np.array(
                [
                    _numeric_or_nan(d.loc[side_idx, "x_centered_px"]),
                    _numeric_or_nan(d.loc[side_idx, "y_centered_px"]),
                ],
                dtype=float,
            )
            if np.isfinite(side_xy).all():
                ideal_distance = float(
                    np.hypot(*(side_xy - start_xy)) + np.hypot(*(end_xy - side_xy))
                )
    elif "r_center_px" in d.columns and len(d) >= 2:
        radius = pd.to_numeric(d["r_center_px"], errors="coerce")
        if radius.notna().any():
            ideal_distance = float(2.0 * radius.max())

    if np.isfinite(path_length) and path_length > 0 and np.isfinite(ideal_distance):
        straightness = float(np.clip(ideal_distance / path_length, 0.0, 1.0))

    jerk = (
        pd.to_numeric(d["jerk_px_s3"], errors="coerce").dropna()
        if "jerk_px_s3" in d.columns
        else pd.Series(dtype=float)
    )
    return {
        "probe_path_length_px": path_length,
        "probe_ideal_out_back_distance_px": ideal_distance,
        "probe_straightness_score": straightness,
        "probe_mean_jerk_px_s3": float(jerk.mean()) if len(jerk) else np.nan,
        "probe_median_jerk_px_s3": float(jerk.median()) if len(jerk) else np.nan,
        "probe_max_jerk_px_s3": float(jerk.max()) if len(jerk) else np.nan,
    }


def load_kinematic_inputs(kinematics_results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the sample-level and segment-level kinematic tables."""
    samples_path = kinematics_results / "kinematic_samples.csv"
    samples_parquet_path = kinematics_results / "kinematic_samples.parquet"
    trials_path = kinematics_results / "trial_kinematic_summary.csv"
    if not samples_path.exists() and not samples_parquet_path.exists():
        raise FileNotFoundError(
            f"Missing sample table: expected {samples_path} or {samples_parquet_path}"
        )
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing segment summary table: {trials_path}")

    if samples_path.exists():
        header = pd.read_csv(samples_path, nrows=0)
        usecols = [c for c in SAMPLE_COLUMNS if c in header.columns]
        samples = pd.read_csv(samples_path, usecols=usecols)
    else:
        try:
            import pyarrow.parquet as pq  # type: ignore

            available_columns = set(pq.ParquetFile(samples_parquet_path).schema.names)
        except Exception:  # pragma: no cover - fallback for non-pyarrow engines
            available_columns = set(pd.read_parquet(samples_parquet_path, engine="auto").head(0).columns)
        usecols = [c for c in SAMPLE_COLUMNS if c in available_columns]
        samples = pd.read_parquet(samples_parquet_path, columns=usecols, engine="auto")
    trials = pd.read_csv(trials_path)
    return samples, trials


def detect_probe_events(
    segment: pd.DataFrame,
    *,
    center_radius_px: float = DEFAULT_CENTER_RADIUS_PX,
    side_radius_px: float = DEFAULT_SIDE_RADIUS_PX,
    min_probe_duration_s: float = DEFAULT_MIN_PROBE_DURATION_S,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect center-to-side excursions within one stiffness segment.

    A probe is counted when radius first enters the center zone
    (``r <= center_radius_px``), then crosses the side zone
    (``r >= side_radius_px``). A second probe is counted only after the cursor
    returns to center. This hysteresis avoids counting jitter on the side
    threshold as multiple probes.
    """
    d = segment.sort_values("time_s").copy()
    d["r_center_px"] = pd.to_numeric(d["r_center_px"], errors="coerce")
    d["time_s"] = pd.to_numeric(d["time_s"], errors="coerce")
    d["stiffness_time_s"] = pd.to_numeric(d.get("stiffness_time_s", d["time_s"]), errors="coerce")
    d["position_angle_deg"] = pd.to_numeric(d.get("position_angle_deg", np.nan), errors="coerce")
    d = d[d["r_center_px"].notna() & d["time_s"].notna()]

    events: list[dict[str, Any]] = []
    center_visits = 0
    was_center = False
    ready_from_center = False
    pending_center: dict[str, Any] | None = None
    active: dict[str, Any] | None = None

    for _, row in d.iterrows():
        r = float(row["r_center_px"])
        t = float(row["time_s"])
        st = float(row["stiffness_time_s"]) if pd.notna(row.get("stiffness_time_s")) else t
        angle = float(row["position_angle_deg"]) if pd.notna(row.get("position_angle_deg")) else np.nan
        speed = float(row["speed_px_s"]) if pd.notna(row.get("speed_px_s")) else np.nan
        x = _numeric_or_nan(row.get("x_centered_px"))
        y = _numeric_or_nan(row.get("y_centered_px"))

        in_center = r <= center_radius_px
        if in_center and not was_center:
            center_visits += 1
            ready_from_center = True
            pending_center = {
                "center_start_time_s": t,
                "center_start_stiffness_time_s": st,
                "center_start_x_centered_px": x,
                "center_start_y_centered_px": y,
                "center_start_radius_px": r,
            }
        was_center = in_center

        if active is None:
            if ready_from_center and r >= side_radius_px:
                start = pending_center or {
                    "center_start_time_s": t,
                    "center_start_stiffness_time_s": st,
                    "center_start_x_centered_px": x,
                    "center_start_y_centered_px": y,
                    "center_start_radius_px": r,
                }
                active = {
                    **start,
                    "probe_start_time_s": start["center_start_time_s"],
                    "probe_start_stiffness_time_s": start["center_start_stiffness_time_s"],
                    "side_cross_time_s": t,
                    "side_cross_stiffness_time_s": st,
                    "side_cross_radius_px": r,
                    "side_cross_x_centered_px": x,
                    "side_cross_y_centered_px": y,
                    "side_cross_angle_deg": angle,
                    "side_cross_direction": _direction_label(angle),
                    "peak_radius_px": r,
                    "peak_time_s": t,
                    "peak_stiffness_time_s": st,
                    "peak_x_centered_px": x,
                    "peak_y_centered_px": y,
                    "peak_angle_deg": angle,
                    "peak_direction": _direction_label(angle),
                    "max_speed_px_s": speed,
                }
                ready_from_center = False
            continue

        if r > active["peak_radius_px"]:
            active["peak_radius_px"] = r
            active["peak_time_s"] = t
            active["peak_stiffness_time_s"] = st
            active["peak_x_centered_px"] = x
            active["peak_y_centered_px"] = y
            active["peak_angle_deg"] = angle
            active["peak_direction"] = _direction_label(angle)
        if np.isfinite(speed):
            previous_speed = active.get("max_speed_px_s", np.nan)
            active["max_speed_px_s"] = float(np.nanmax([previous_speed, speed]))

        if in_center:
            active["probe_end_time_s"] = t
            active["probe_end_stiffness_time_s"] = st
            active["probe_end_x_centered_px"] = x
            active["probe_end_y_centered_px"] = y
            active["probe_end_radius_px"] = r
            duration = active["probe_end_time_s"] - active["probe_start_time_s"]
            if duration >= min_probe_duration_s:
                active["probe_duration_s"] = duration
                active.update(
                    _probe_path_metrics(
                        d,
                        start_time_s=active["probe_start_time_s"],
                        end_time_s=active["probe_end_time_s"],
                        side_time_s=active["peak_time_s"],
                    )
                )
                events.append(active)
            active = None
            ready_from_center = True
            pending_center = {
                "center_start_time_s": t,
                "center_start_stiffness_time_s": st,
                "center_start_x_centered_px": x,
                "center_start_y_centered_px": y,
                "center_start_radius_px": r,
            }

    if active is not None and not d.empty:
        last = d.iloc[-1]
        active["probe_end_time_s"] = float(last["time_s"])
        active["probe_end_stiffness_time_s"] = float(last["stiffness_time_s"]) if pd.notna(last.get("stiffness_time_s")) else float(last["time_s"])
        active["probe_end_x_centered_px"] = _numeric_or_nan(last.get("x_centered_px"))
        active["probe_end_y_centered_px"] = _numeric_or_nan(last.get("y_centered_px"))
        active["probe_end_radius_px"] = _numeric_or_nan(last.get("r_center_px"))
        duration = active["probe_end_time_s"] - active["probe_start_time_s"]
        if duration >= min_probe_duration_s:
            active["probe_duration_s"] = duration
            active.update(
                _probe_path_metrics(
                    d,
                    start_time_s=active["probe_start_time_s"],
                    end_time_s=active["probe_end_time_s"],
                    side_time_s=active["peak_time_s"],
                )
            )
            events.append(active)

    duration_s = float(d["stiffness_time_s"].max() - d["stiffness_time_s"].min()) if len(d) > 1 else np.nan
    if len(d) > 1:
        dt_next = d["stiffness_time_s"].shift(-1) - d["stiffness_time_s"]
        valid_dt = pd.to_numeric(dt_next, errors="coerce").clip(lower=0)
        center_dwell_s = float(valid_dt.where(d["r_center_px"] <= center_radius_px, 0).sum())
        side_dwell_s = float(valid_dt.where(d["r_center_px"] >= side_radius_px, 0).sum())
        exploration_band_dwell_s = float(
            valid_dt.where((d["r_center_px"] > center_radius_px) & (d["r_center_px"] < side_radius_px), 0).sum()
        )
    else:
        center_dwell_s = np.nan
        side_dwell_s = np.nan
        exploration_band_dwell_s = np.nan
    first_probe_latency_s = (
        float(events[0]["side_cross_stiffness_time_s"] - d["stiffness_time_s"].min())
        if events and d["stiffness_time_s"].notna().any()
        else np.nan
    )
    summary = {
        "n_samples_for_probing": int(len(d)),
        "center_radius_px": center_radius_px,
        "side_radius_px": side_radius_px,
        "center_visit_count": int(center_visits),
        "probe_count": int(len(events)),
        "probe_rate_per_s": float(len(events) / duration_s) if np.isfinite(duration_s) and duration_s > 0 else np.nan,
        "first_probe_latency_s": first_probe_latency_s,
        "center_dwell_s": center_dwell_s,
        "side_dwell_s": side_dwell_s,
        "exploration_band_dwell_s": exploration_band_dwell_s,
        "center_dwell_fraction": float(center_dwell_s / duration_s) if np.isfinite(center_dwell_s) and np.isfinite(duration_s) and duration_s > 0 else np.nan,
        "side_dwell_fraction": float(side_dwell_s / duration_s) if np.isfinite(side_dwell_s) and np.isfinite(duration_s) and duration_s > 0 else np.nan,
        "exploration_band_dwell_fraction": float(exploration_band_dwell_s / duration_s) if np.isfinite(exploration_band_dwell_s) and np.isfinite(duration_s) and duration_s > 0 else np.nan,
        "mean_probe_duration_s": float(np.nanmean([e.get("probe_duration_s", np.nan) for e in events])) if events else np.nan,
        "mean_probe_peak_radius_px": float(np.nanmean([e.get("peak_radius_px", np.nan) for e in events])) if events else np.nan,
        "max_probe_peak_radius_px": float(np.nanmax([e.get("peak_radius_px", np.nan) for e in events])) if events else np.nan,
        "mean_probe_max_speed_px_s": float(np.nanmean([e.get("max_speed_px_s", np.nan) for e in events])) if events else np.nan,
        "mean_probe_jerk_px_s3": _nanmean_or_nan([e.get("probe_mean_jerk_px_s3", np.nan) for e in events]) if events else np.nan,
        "median_probe_jerk_px_s3": _nanmedian_or_nan([e.get("probe_median_jerk_px_s3", np.nan) for e in events]) if events else np.nan,
        "mean_probe_straightness_score": _nanmean_or_nan([e.get("probe_straightness_score", np.nan) for e in events]) if events else np.nan,
        "median_probe_straightness_score": _nanmedian_or_nan([e.get("probe_straightness_score", np.nan) for e in events]) if events else np.nan,
        "unique_probe_directions": int(pd.Series([e.get("peak_direction") for e in events]).dropna().nunique()) if events else 0,
        "dominant_probe_direction": _mode_or_unknown(pd.Series([e.get("peak_direction") for e in events])),
    }
    return events, summary


def compute_probing_metrics(
    samples: pd.DataFrame,
    trial_summary: pd.DataFrame,
    *,
    center_radius_px: float = DEFAULT_CENTER_RADIUS_PX,
    side_radius_px: float = DEFAULT_SIDE_RADIUS_PX,
    min_probe_duration_s: float = DEFAULT_MIN_PROBE_DURATION_S,
) -> dict[str, pd.DataFrame]:
    """Compute probe event and segment summary tables."""
    samples = samples.copy()
    samples = samples[pd.to_numeric(samples.get("stiffness_value"), errors="coerce") > 0]
    samples["stiffness_value"] = pd.to_numeric(samples["stiffness_value"], errors="coerce")
    samples["trial_index_raw"] = pd.to_numeric(samples["trial_index_raw"], errors="coerce")
    samples["stiffness_segment_id"] = pd.to_numeric(samples["stiffness_segment_id"], errors="coerce")
    samples = samples.dropna(subset=TRIAL_KEYS)
    samples = samples.sort_values(TRIAL_KEYS + ["time_s"])

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    meta_cols = [
        "subject_id",
        "subject_group",
        EXPERIMENT_GROUP_COLUMN,
        "trial_index_raw",
        "pair_number",
        "finger_condition",
        "comparison_value",
        "standard_value",
        "signed_stiffness_delta",
        "correct_response",
        "answer_code",
        "stiffness_value",
        "stiffness_segment_id",
        "stiffness_order_in_trial",
    ]

    for keys, seg in samples.groupby(TRIAL_KEYS, sort=False, dropna=False):
        meta = {c: seg[c].dropna().iloc[0] if c in seg and seg[c].notna().any() else np.nan for c in meta_cols}
        events, probing_summary = detect_probe_events(
            seg,
            center_radius_px=center_radius_px,
            side_radius_px=side_radius_px,
            min_probe_duration_s=min_probe_duration_s,
        )
        summary_rows.append({**meta, **probing_summary})
        for idx, event in enumerate(events, start=1):
            event_rows.append({**meta, "probe_index": idx, **event})

    probing_trial_summary = pd.DataFrame(summary_rows)
    probing_event_log = pd.DataFrame(event_rows)

    if not trial_summary.empty and not probing_trial_summary.empty:
        keep = [
            *TRIAL_KEYS,
            "duration_s",
            "pair_duration_s",
            "n_tracking_samples",
            "mean_r_center_px",
            "max_r_center_px",
            "path_length_px",
            "mean_speed_px_s",
            "mean_acceleration_px_s2",
        ]
        keep = [c for c in keep if c in trial_summary.columns]
        probing_trial_summary = probing_trial_summary.merge(
            trial_summary[keep].drop_duplicates(TRIAL_KEYS),
            on=TRIAL_KEYS,
            how="left",
        )

    if not probing_trial_summary.empty:
        for col in ["probe_count", "probe_rate_per_s", "center_visit_count", "path_length_px"]:
            if col in probing_trial_summary:
                vals = pd.to_numeric(probing_trial_summary[col], errors="coerce")
                probing_trial_summary[f"log1p_{col}"] = np.where(vals >= 0, np.log1p(vals), np.nan)

    summaries = summarize_probing(probing_trial_summary, probing_event_log)
    return {
        "probing_event_log": probing_event_log,
        "probing_trial_summary": probing_trial_summary,
        **summaries,
    }


def _merge_probe_event_metrics(
    base: pd.DataFrame,
    events: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Merge count-from-segments with jerk/straightness-from-events summaries."""
    count_cols = [c for c in group_cols if c in base.columns]
    if not count_cols:
        return pd.DataFrame()
    segment_summary = (
        base.groupby(count_cols, dropna=False)
        .agg(
            n_segments=("trial_index_raw", "count"),
            total_probe_count=("probe_count", "sum"),
            mean_probe_count_per_segment=("probe_count", "mean"),
            median_probe_count_per_segment=("probe_count", "median"),
            n_subjects=("subject_id", "nunique") if "subject_id" in base.columns else ("trial_index_raw", "count"),
        )
        .reset_index()
    )
    event_cols = [c for c in group_cols if c in events.columns]
    if not events.empty and event_cols == count_cols:
        event_summary = (
            events.groupby(event_cols, dropna=False)
            .agg(
                n_probe_events=("probe_index", "count"),
                mean_probe_jerk_px_s3=("probe_mean_jerk_px_s3", "mean"),
                median_probe_jerk_px_s3=("probe_median_jerk_px_s3", "median"),
                mean_probe_straightness_score=("probe_straightness_score", "mean"),
                median_probe_straightness_score=("probe_straightness_score", "median"),
                mean_probe_path_length_px=("probe_path_length_px", "mean"),
                median_probe_path_length_px=("probe_path_length_px", "median"),
                mean_probe_duration_s=("probe_duration_s", "mean"),
                median_probe_duration_s=("probe_duration_s", "median"),
            )
            .reset_index()
        )
        segment_summary = segment_summary.merge(event_summary, on=count_cols, how="left")
    else:
        for col in [
            "n_probe_events",
            "mean_probe_jerk_px_s3",
            "median_probe_jerk_px_s3",
            "mean_probe_straightness_score",
            "median_probe_straightness_score",
            "mean_probe_path_length_px",
            "median_probe_path_length_px",
            "mean_probe_duration_s",
            "median_probe_duration_s",
        ]:
            segment_summary[col] = np.nan
    segment_summary["n_probe_events"] = segment_summary["n_probe_events"].fillna(0).astype(int)
    segment_summary["probe_events_minus_segment_count_check"] = (
        pd.to_numeric(segment_summary["n_probe_events"], errors="coerce")
        - pd.to_numeric(segment_summary["total_probe_count"], errors="coerce")
    )
    return segment_summary


def _compute_probe_metric_summary_tables(
    probing_trial_summary: pd.DataFrame,
    probing_event_log: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Requested probe count, jerk, and straightness summaries by subject/group."""
    if probing_trial_summary.empty:
        empty = pd.DataFrame()
        return {
            "probing_subject_finger_stiffness_probe_metrics_summary": empty,
            "probing_group_finger_stiffness_probe_metrics_summary": empty,
            "probing_analysis_scope_finger_stiffness_probe_metrics_summary": empty,
        }
    pts = add_experiment_group_columns(probing_trial_summary.copy())
    events = add_experiment_group_columns(probing_event_log.copy()) if not probing_event_log.empty else pd.DataFrame()
    subject = _merge_probe_event_metrics(
        pts,
        events,
        ["subject_id", "subject_group", EXPERIMENT_GROUP_COLUMN, "finger_condition", "stiffness_value"],
    )
    group = _merge_probe_event_metrics(
        pts,
        events,
        [EXPERIMENT_GROUP_COLUMN, "finger_condition", "stiffness_value"],
    )
    scoped_pts = expand_analysis_scopes(pts)
    scoped_events = expand_analysis_scopes(events) if not events.empty else pd.DataFrame()
    scoped = _merge_probe_event_metrics(
        scoped_pts,
        scoped_events,
        [ANALYSIS_SCOPE_COLUMN, ANALYSIS_SCOPE_VALUE_COLUMN, "finger_condition", "stiffness_value"],
    )
    return {
        "probing_subject_finger_stiffness_probe_metrics_summary": subject,
        "probing_group_finger_stiffness_probe_metrics_summary": group,
        "probing_analysis_scope_finger_stiffness_probe_metrics_summary": scoped,
    }


def summarize_probing(probing_trial_summary: pd.DataFrame, probing_event_log: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if probing_trial_summary.empty:
        return {
            "probing_subject_finger_stiffness_summary": pd.DataFrame(),
            "probing_stiffness_summary": pd.DataFrame(),
            "probing_finger_stiffness_summary": pd.DataFrame(),
            "probing_comparison_summary": pd.DataFrame(),
            "probing_success_summary": pd.DataFrame(),
            "probing_success_by_stiffness": pd.DataFrame(),
            "probing_subject_finger_correlations": pd.DataFrame(),
            "probing_direction_summary": pd.DataFrame(),
            "probing_group_metric_summary": pd.DataFrame(),
            "probing_group_condition_metric_summary": pd.DataFrame(),
            "probing_within_group_condition_comparisons": pd.DataFrame(),
            "probing_between_group_metric_comparisons": pd.DataFrame(),
            "probing_analysis_scope_metric_summary": pd.DataFrame(),
            "probing_analysis_scope_condition_metric_summary": pd.DataFrame(),
            "probing_within_analysis_scope_condition_comparisons": pd.DataFrame(),
            "probing_between_analysis_scope_metric_comparisons": pd.DataFrame(),
            "probing_success_one_way_anova": pd.DataFrame(),
            "probing_success_anova_factor_summary": pd.DataFrame(),
            "probing_success_anova_pairwise": pd.DataFrame(),
            "probing_subject_finger_stiffness_probe_metrics_summary": pd.DataFrame(),
            "probing_group_finger_stiffness_probe_metrics_summary": pd.DataFrame(),
            "probing_analysis_scope_finger_stiffness_probe_metrics_summary": pd.DataFrame(),
        }
    pts = add_experiment_group_columns(probing_trial_summary.copy())
    for col in [
        "first_probe_latency_s",
        "center_dwell_fraction",
        "side_dwell_fraction",
        "exploration_band_dwell_fraction",
    ]:
        if col not in pts.columns:
            pts[col] = np.nan
    pts["correct_response"] = pd.to_numeric(pts["correct_response"], errors="coerce")
    for col in ["probe_count", "probe_rate_per_s", "path_length_px"]:
        log_col = f"log1p_{col}"
        if col in pts.columns and log_col not in pts.columns:
            vals = pd.to_numeric(pts[col], errors="coerce")
            pts[log_col] = np.where(vals >= 0, np.log1p(vals), np.nan)

    agg_kwargs = dict(
        n_data_points=("trial_index_raw", "count"),
        n_trials=("trial_index_raw", "count"),
        success_rate=("correct_response", "mean"),
        success_rate_ci95_lower=("correct_response", _wilson_ci95_lower),
        success_rate_ci95_upper=("correct_response", _wilson_ci95_upper),
        mean_probe_count=("probe_count", "mean"),
        sem_probe_count=("probe_count", _sem),
        probe_count_ci95_lower=("probe_count", _mean_ci95_lower),
        probe_count_ci95_upper=("probe_count", _mean_ci95_upper),
        median_probe_count=("probe_count", "median"),
        mean_probe_rate_per_s=("probe_rate_per_s", "mean"),
        sem_probe_rate_per_s=("probe_rate_per_s", _sem),
        probe_rate_per_s_ci95_lower=("probe_rate_per_s", _mean_ci95_lower),
        probe_rate_per_s_ci95_upper=("probe_rate_per_s", _mean_ci95_upper),
        median_probe_rate_per_s=("probe_rate_per_s", "median"),
        mean_center_visit_count=("center_visit_count", "mean"),
        median_center_visit_count=("center_visit_count", "median"),
        mean_first_probe_latency_s=("first_probe_latency_s", "mean"),
        median_first_probe_latency_s=("first_probe_latency_s", "median"),
        mean_center_dwell_fraction=("center_dwell_fraction", "mean"),
        mean_side_dwell_fraction=("side_dwell_fraction", "mean"),
        mean_exploration_band_dwell_fraction=("exploration_band_dwell_fraction", "mean"),
        mean_unique_probe_directions=("unique_probe_directions", "mean"),
        mean_probe_peak_radius_px=("mean_probe_peak_radius_px", "mean"),
        median_probe_peak_radius_px=("mean_probe_peak_radius_px", "median"),
        mean_probe_duration_s=("mean_probe_duration_s", "mean"),
        median_probe_duration_s=("mean_probe_duration_s", "median"),
        mean_probe_jerk_px_s3=("mean_probe_jerk_px_s3", "mean"),
        median_probe_jerk_px_s3=("median_probe_jerk_px_s3", "median"),
        mean_probe_straightness_score=("mean_probe_straightness_score", "mean"),
        median_probe_straightness_score=("median_probe_straightness_score", "median"),
        mean_path_length_px=("path_length_px", "mean"),
        median_path_length_px=("path_length_px", "median"),
        mean_speed_px_s=("mean_speed_px_s", "mean"),
        median_speed_px_s=("mean_speed_px_s", "median"),
        mean_log1p_probe_count=("log1p_probe_count", "mean"),
        log1p_probe_count_ci95_lower=("log1p_probe_count", _mean_ci95_lower),
        log1p_probe_count_ci95_upper=("log1p_probe_count", _mean_ci95_upper),
        median_log1p_probe_count=("log1p_probe_count", "median"),
        mean_log1p_probe_rate_per_s=("log1p_probe_rate_per_s", "mean"),
        log1p_probe_rate_per_s_ci95_lower=("log1p_probe_rate_per_s", _mean_ci95_lower),
        log1p_probe_rate_per_s_ci95_upper=("log1p_probe_rate_per_s", _mean_ci95_upper),
        median_log1p_probe_rate_per_s=("log1p_probe_rate_per_s", "median"),
        mean_log1p_path_length_px=("log1p_path_length_px", "mean"),
        log1p_path_length_px_ci95_lower=("log1p_path_length_px", _mean_ci95_lower),
        log1p_path_length_px_ci95_upper=("log1p_path_length_px", _mean_ci95_upper),
    )

    subject_finger_stiffness = pts.groupby(["subject_id", "subject_group", EXPERIMENT_GROUP_COLUMN, "finger_condition", "stiffness_value"], dropna=False).agg(**agg_kwargs).reset_index()
    stiffness_summary = pts.groupby(["stiffness_value"], dropna=False).agg(
        n_subjects=("subject_id", "nunique"),
        **agg_kwargs,
    ).reset_index()
    finger_stiffness_summary = pts.groupby(["finger_condition", "stiffness_value"], dropna=False).agg(
        n_subjects=("subject_id", "nunique"),
        **agg_kwargs,
    ).reset_index()
    comparison_summary = pts.groupby(["comparison_value", "stiffness_value"], dropna=False).agg(
        n_subjects=("subject_id", "nunique"),
        **agg_kwargs,
    ).reset_index()
    success_summary = pts.groupby(["correct_response"], dropna=False).agg(
        n_subjects=("subject_id", "nunique"),
        **{k: v for k, v in agg_kwargs.items() if k != "success_rate"},
    ).reset_index()
    success_by_stiffness = pts.groupby(["stiffness_value", "correct_response"], dropna=False).agg(
        n_subjects=("subject_id", "nunique"),
        **{k: v for k, v in agg_kwargs.items() if k != "success_rate"},
    ).reset_index()

    corr_rows: list[dict[str, Any]] = []
    for keys, g in pts.groupby(["subject_id", "subject_group", EXPERIMENT_GROUP_COLUMN, "finger_condition"], dropna=False):
        corr_rows.append(
            {
                "subject_id": keys[0],
                "subject_group": keys[1],
                EXPERIMENT_GROUP_COLUMN: keys[2],
                "finger_condition": keys[3],
                "n_trials": int(len(g)),
                "probe_count_success_corr": _pearson(g["probe_count"], g["correct_response"]),
                "probe_rate_success_corr": _pearson(g["probe_rate_per_s"], g["correct_response"]),
                "probe_count_stiffness_corr": _pearson(g["probe_count"], g["stiffness_value"]),
                "success_rate": float(g["correct_response"].mean()) if g["correct_response"].notna().any() else np.nan,
                "mean_probe_count": float(g["probe_count"].mean()),
            }
        )
    correlations = pd.DataFrame(corr_rows)

    if probing_event_log.empty:
        direction_summary = pd.DataFrame()
    else:
        direction_summary = probing_event_log.groupby(["stiffness_value", "peak_direction"], dropna=False).agg(
            n_probe_events=("probe_index", "count"),
            n_subjects=("subject_id", "nunique"),
            mean_peak_radius_px=("peak_radius_px", "mean"),
            mean_probe_duration_s=("probe_duration_s", "mean"),
        ).reset_index()

    group_comparisons = compute_experiment_group_comparisons(pts)
    anova_tables = compute_success_one_way_anova_tables(pts)
    requested_probe_summaries = _compute_probe_metric_summary_tables(pts, probing_event_log)

    return {
        "probing_subject_finger_stiffness_summary": _add_log_backtransform_columns(subject_finger_stiffness),
        "probing_stiffness_summary": _add_log_backtransform_columns(stiffness_summary),
        "probing_finger_stiffness_summary": _add_log_backtransform_columns(finger_stiffness_summary),
        "probing_comparison_summary": _add_log_backtransform_columns(comparison_summary),
        "probing_success_summary": _add_log_backtransform_columns(success_summary),
        "probing_success_by_stiffness": _add_log_backtransform_columns(success_by_stiffness),
        "probing_subject_finger_correlations": correlations,
        "probing_direction_summary": direction_summary,
        **requested_probe_summaries,
        **group_comparisons,
        **anova_tables,
    }


# ---------------------------------------------------------------------------
# Experiment-group comparison section (N_E, L_E, L_P)

PROBING_GROUP_METRICS = [
    "correct_response",
    "probe_count",
    "probe_rate_per_s",
    "center_visit_count",
    "first_probe_latency_s",
    "center_dwell_fraction",
    "side_dwell_fraction",
    "exploration_band_dwell_fraction",
    "unique_probe_directions",
    "mean_probe_peak_radius_px",
    "mean_probe_duration_s",
    "mean_probe_jerk_px_s3",
    "median_probe_jerk_px_s3",
    "mean_probe_straightness_score",
    "median_probe_straightness_score",
    "path_length_px",
    "mean_speed_px_s",
]


ANOVA_FACTOR_SPECS = [
    ("amount_of_probing", "probe_count_bin"),
    ("stiffness_value", "stiffness_value"),
    ("finger", "finger_condition"),
]
ANOVA_ALPHA = 0.05


def _add_probe_count_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable probe-amount categories with enough replication for ANOVA."""
    out = df.copy()
    counts = pd.to_numeric(out.get("probe_count"), errors="coerce")
    labels = pd.Series(np.nan, index=out.index, dtype="object")
    labels[counts == 0] = "0"
    labels[counts == 1] = "1"
    labels[counts == 2] = "2"
    labels[counts == 3] = "3"
    labels[counts >= 4] = "4+"
    out["probe_count_bin"] = labels
    return out


def _clean_anova_input(df: pd.DataFrame, factor_col: str, outcome_col: str = "correct_response") -> pd.DataFrame:
    if df.empty or factor_col not in df.columns or outcome_col not in df.columns:
        return pd.DataFrame(columns=[factor_col, outcome_col])
    out = df[[factor_col, outcome_col]].copy()
    out[outcome_col] = pd.to_numeric(out[outcome_col], errors="coerce")
    out = out[out[factor_col].notna() & out[outcome_col].notna()].copy()
    out[factor_col] = out[factor_col].astype(str)
    return out


def _one_way_anova_row(
    df: pd.DataFrame,
    *,
    factor_name: str,
    factor_col: str,
    outcome_col: str = "correct_response",
) -> dict[str, Any]:
    """Return a dependency-light one-factor ANOVA row for success by one factor."""
    clean = _clean_anova_input(df, factor_col, outcome_col)
    groups = [
        pd.to_numeric(g[outcome_col], errors="coerce").dropna().to_numpy(dtype=float)
        for _, g in clean.groupby(factor_col, dropna=False)
    ]
    groups = [g for g in groups if len(g) > 0]
    n_observations = int(sum(len(g) for g in groups))
    n_factor_levels = int(len(groups))
    base = {
        "factor": factor_name,
        "factor_column": factor_col,
        "outcome": outcome_col,
        "n_observations": n_observations,
        "n_factor_levels": n_factor_levels,
        "df_between": np.nan,
        "df_within": np.nan,
        "ss_between": np.nan,
        "ss_within": np.nan,
        "f_statistic": np.nan,
        "p_value": np.nan,
        "eta_squared": np.nan,
        "omega_squared": np.nan,
        f"significant_alpha_{str(ANOVA_ALPHA).replace('.', '_')}": False,
        "status": "insufficient_factor_levels",
    }
    if n_factor_levels < 2:
        return base
    if any(len(g) < 2 for g in groups) or n_observations <= n_factor_levels:
        return {**base, "status": "insufficient_within_level_replication"}

    all_values = np.concatenate(groups)
    grand_mean = float(np.mean(all_values))
    ss_between = float(sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups))
    ss_within = float(sum(np.sum((g - float(np.mean(g))) ** 2) for g in groups))
    ss_total = ss_between + ss_within
    df_between = n_factor_levels - 1
    df_within = n_observations - n_factor_levels
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    if np.isfinite(ms_within) and ms_within > 0:
        f_stat = float(ms_between / ms_within)
    elif ss_between > 0 and ss_within == 0:
        f_stat = np.inf
    else:
        f_stat = np.nan
    p_value = float(scipy_stats.f.sf(f_stat, df_between, df_within)) if scipy_stats is not None and np.isfinite(f_stat) else np.nan
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else np.nan
    omega_num = ss_between - df_between * ms_within if np.isfinite(ms_within) else np.nan
    omega_den = ss_total + ms_within if np.isfinite(ms_within) else np.nan
    omega_squared = float(omega_num / omega_den) if np.isfinite(omega_den) and omega_den > 0 else np.nan
    significant = bool(np.isfinite(p_value) and p_value < ANOVA_ALPHA)
    return {
        **base,
        "df_between": int(df_between),
        "df_within": int(df_within),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "f_statistic": f_stat,
        "p_value": p_value,
        "eta_squared": eta_squared,
        "omega_squared": omega_squared,
        f"significant_alpha_{str(ANOVA_ALPHA).replace('.', '_')}": significant,
        "status": "ok" if scipy_stats is not None else "ok_no_scipy_p_value",
    }


def _pairwise_success_rows(
    df: pd.DataFrame,
    *,
    factor_name: str,
    factor_col: str,
    outcome_col: str = "correct_response",
) -> list[dict[str, Any]]:
    clean = _clean_anova_input(df, factor_col, outcome_col)
    if clean.empty:
        return []
    levels = sorted(clean[factor_col].dropna().unique(), key=lambda x: str(x))
    pairs = list(combinations(levels, 2))
    rows: list[dict[str, Any]] = []
    for level_a, level_b in pairs:
        a = clean.loc[clean[factor_col] == level_a, outcome_col]
        b = clean.loc[clean[factor_col] == level_b, outcome_col]
        av = pd.to_numeric(a, errors="coerce").dropna()
        bv = pd.to_numeric(b, errors="coerce").dropna()
        p_value = np.nan
        av_var = av.var(ddof=1) if len(av) >= 2 else np.nan
        bv_var = bv.var(ddof=1) if len(bv) >= 2 else np.nan
        if scipy_stats is not None and len(av) >= 2 and len(bv) >= 2 and av_var > 0 and bv_var > 0:
            p_value = float(scipy_stats.ttest_ind(av, bv, equal_var=False, nan_policy="omit").pvalue)
        pooled_d = _pooled_cohens_d_for_values(av, bv)
        p_value_bonferroni = float(min(1.0, p_value * len(pairs))) if np.isfinite(p_value) else np.nan
        rows.append(
            {
                "factor": factor_name,
                "factor_column": factor_col,
                "outcome": outcome_col,
                "level_a": level_a,
                "level_b": level_b,
                "comparison": f"{level_b} - {level_a}",
                "n_a": int(len(av)),
                "n_b": int(len(bv)),
                "mean_success_a": float(av.mean()) if len(av) else np.nan,
                "mean_success_b": float(bv.mean()) if len(bv) else np.nan,
                "mean_difference_b_minus_a": float(bv.mean() - av.mean()) if len(av) and len(bv) else np.nan,
                "cohens_d_b_minus_a": pooled_d,
                "p_value": p_value,
                "p_value_bonferroni": p_value_bonferroni,
                f"significant_bonferroni_alpha_{str(ANOVA_ALPHA).replace('.', '_')}": bool(
                    np.isfinite(p_value_bonferroni) and p_value_bonferroni < ANOVA_ALPHA
                ),
            }
        )
    return rows


def _pooled_cohens_d_for_values(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce").dropna()
    bb = pd.to_numeric(b, errors="coerce").dropna()
    if len(aa) < 2 or len(bb) < 2:
        return np.nan
    pooled_var = ((len(aa) - 1) * aa.var(ddof=1) + (len(bb) - 1) * bb.var(ddof=1)) / (len(aa) + len(bb) - 2)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return np.nan
    return float((bb.mean() - aa.mean()) / math.sqrt(pooled_var))


def _factor_summary_rows(
    df: pd.DataFrame,
    *,
    factor_name: str,
    factor_col: str,
    outcome_col: str = "correct_response",
) -> list[dict[str, Any]]:
    if df.empty or factor_col not in df.columns:
        return []
    clean = df.copy()
    clean[outcome_col] = pd.to_numeric(clean.get(outcome_col), errors="coerce")
    clean = clean[clean[factor_col].notna() & clean[outcome_col].notna()]
    rows: list[dict[str, Any]] = []
    for level, g in clean.groupby(factor_col, dropna=False):
        success = pd.to_numeric(g[outcome_col], errors="coerce").dropna()
        lo, hi = _wilson_ci95_bounds(success)
        rows.append(
            {
                "factor": factor_name,
                "factor_column": factor_col,
                "factor_level": level,
                "n_observations": int(len(success)),
                "mean_success_rate": float(success.mean()) if len(success) else np.nan,
                "success_rate_ci95_lower": lo,
                "success_rate_ci95_upper": hi,
                "mean_probe_count": float(pd.to_numeric(g.get("probe_count"), errors="coerce").mean()) if "probe_count" in g else np.nan,
                "mean_stiffness_value": float(pd.to_numeric(g.get("stiffness_value"), errors="coerce").mean()) if "stiffness_value" in g else np.nan,
                "n_subjects": int(g["subject_id"].nunique()) if "subject_id" in g else np.nan,
            }
        )
    return rows


def compute_success_one_way_anova_tables(probing_trial_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute one-way ANOVA of success by probe amount, stiffness, and finger.

    The requested views are represented explicitly:
    - ``analysis_scope=all`` for all participants.
    - ``analysis_scope=experiment_group`` for groups such as ``N_E`` and ``L_E``.
    - ``analysis_scope=participant`` for per-person tests.
    - ``observation_level=trial`` for segment/trial rows.
    - ``observation_level=participant_mean`` for group/all tests on participant-level
      means, reducing trial-level pseudo-replication.
    """
    empty = {
        "probing_success_one_way_anova": pd.DataFrame(),
        "probing_success_anova_factor_summary": pd.DataFrame(),
        "probing_success_anova_pairwise": pd.DataFrame(),
    }
    if probing_trial_summary.empty:
        return empty

    prepared = _add_probe_count_bins(add_experiment_group_columns(probing_trial_summary.copy()))
    prepared["correct_response"] = pd.to_numeric(prepared["correct_response"], errors="coerce")
    expanded = expand_analysis_scopes(prepared)
    if expanded.empty:
        return empty

    anova_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    scope_cols = [ANALYSIS_SCOPE_COLUMN, ANALYSIS_SCOPE_VALUE_COLUMN]

    for scope_keys, scoped in expanded.groupby(scope_cols, dropna=False):
        scope_keys = scope_keys if isinstance(scope_keys, tuple) else (scope_keys,)
        scope_meta = dict(zip(scope_cols, scope_keys))
        trial_df = scoped.copy()
        datasets: list[tuple[str, pd.DataFrame]] = [("trial", trial_df)]

        for factor_name, factor_col in ANOVA_FACTOR_SPECS:
            if factor_col not in scoped.columns:
                continue
            agg_map: dict[str, tuple[str, str]] = {"correct_response": ("correct_response", "mean")}
            if "probe_count" in scoped.columns and factor_col != "probe_count":
                agg_map["probe_count"] = ("probe_count", "mean")
            if "stiffness_value" in scoped.columns and factor_col != "stiffness_value":
                agg_map["stiffness_value"] = ("stiffness_value", "mean")
            participant_mean = (
                scoped.dropna(subset=["subject_id", factor_col])
                .groupby(["subject_id", factor_col], dropna=False)
                .agg(**agg_map)
                .reset_index()
            )
            datasets.append((f"participant_mean__{factor_name}", participant_mean))

        for observation_level, data in datasets:
            if observation_level.startswith("participant_mean__"):
                allowed_factor = observation_level.split("__", 1)[1]
                observation_label = "participant_mean"
            else:
                allowed_factor = None
                observation_label = observation_level
            for factor_name, factor_col in ANOVA_FACTOR_SPECS:
                if allowed_factor is not None and factor_name != allowed_factor:
                    continue
                if factor_col not in data.columns:
                    continue
                meta = {**scope_meta, "observation_level": observation_label}
                anova_rows.append({**meta, **_one_way_anova_row(data, factor_name=factor_name, factor_col=factor_col)})
                summary_rows.extend({**meta, **row} for row in _factor_summary_rows(data, factor_name=factor_name, factor_col=factor_col))
                pairwise_rows.extend({**meta, **row} for row in _pairwise_success_rows(data, factor_name=factor_name, factor_col=factor_col))

    return {
        "probing_success_one_way_anova": pd.DataFrame(anova_rows),
        "probing_success_anova_factor_summary": pd.DataFrame(summary_rows),
        "probing_success_anova_pairwise": pd.DataFrame(pairwise_rows),
    }


def compute_experiment_group_comparisons(probing_trial_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Restrict the experiment-group comparison to the main experiment groups
    # (N_E, L_E) and drop protocol/pilot groups such as L_P, which otherwise leak
    # into every probing_group_metric_summary_* figure. (The general group
    # normaliser keeps L_P, so we filter the input rows here by subject group.)
    pts = probing_trial_summary
    if not pts.empty and "subject_id" in pts.columns:
        group = pts["subject_id"].map(normalize_experiment_group)
        pts = pts[group.isin(EXPERIMENT_GROUP_ORDER)].copy()
    exact_tables = {
        f"probing_{name}": table
        for name, table in compute_group_comparison_tables(
            pts,
            metric_columns=PROBING_GROUP_METRICS,
            condition_cols=["finger_condition", "stiffness_value"],
        ).items()
    }
    scope_tables = {
        f"probing_{name}": table
        for name, table in compute_analysis_scope_tables(
            pts,
            metric_columns=PROBING_GROUP_METRICS,
            condition_cols=["finger_condition", "stiffness_value"],
        ).items()
    }
    setup_tables = {
        f"probing_{name}": table
        for name, table in compute_setup_factor_tables(
            pts,
            metric_columns=PROBING_GROUP_METRICS,
            condition_cols=["finger_condition", "stiffness_value"],
        ).items()
    }
    return {**exact_tables, **scope_tables, **setup_tables}


def save_figures(output_root: Path, tables: dict[str, pd.DataFrame], fig_dpi: int = 160) -> list[Path]:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    stiffness = tables.get("probing_stiffness_summary", pd.DataFrame()).sort_values("stiffness_value")
    trials = tables.get("probing_trial_summary", pd.DataFrame())
    if not stiffness.empty:
        for y_col, y_label, filename in [
            ("mean_probe_count", "Mean center-to-side probes per stiffness segment", "probe_count_by_stiffness.png"),
            ("mean_probe_rate_per_s", "Mean probe rate (probes/s)", "probe_rate_by_stiffness.png"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            raw_col = "probe_count" if y_col == "mean_probe_count" else "probe_rate_per_s"
            if not trials.empty and raw_col in trials:
                raw = trials[["stiffness_value", raw_col]].dropna().copy()
                if not raw.empty:
                    jitter = np.random.default_rng(7).normal(0, 0.45, len(raw))
                    ax.scatter(raw["stiffness_value"] + jitter, raw[raw_col], s=8, alpha=0.08, color="0.25", label="trial data points")
            lower_col = raw_col + "_ci95_lower"
            upper_col = raw_col + "_ci95_upper"
            if lower_col in stiffness and upper_col in stiffness:
                yerr = np.vstack([
                    stiffness[y_col] - stiffness[lower_col],
                    stiffness[upper_col] - stiffness[y_col],
                ])
            else:
                err_col = "sem_probe_count" if y_col == "mean_probe_count" else "sem_probe_rate_per_s"
                yerr = stiffness.get(err_col)
            ax.errorbar(stiffness["stiffness_value"], stiffness[y_col], yerr=yerr, marker="o", capsize=3, linewidth=2, label="mean ± 95% CI")
            ax.set_xlabel("Stiffness value")
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label} by stiffness")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out = fig_dir / filename
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

        fig, ax = plt.subplots(figsize=(7, 5))
        sf = tables.get("probing_subject_finger_stiffness_summary", pd.DataFrame())
        if not sf.empty:
            ax.scatter(sf["mean_probe_count"], sf["success_rate"], s=18, alpha=0.25, color="0.35", label="subject/finger/stiffness data points")
        ax.scatter(stiffness["mean_probe_count"], stiffness["success_rate"], s=70)
        for row in stiffness.itertuples():
            ax.annotate(str(getattr(row, "stiffness_value")), (getattr(row, "mean_probe_count"), getattr(row, "success_rate")), fontsize=8)
        ax.set_xlabel("Mean probe count")
        ax.set_ylabel("Success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Success rate vs probing by stiffness")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = fig_dir / "success_vs_probe_count_by_stiffness.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

    finger = tables.get("probing_finger_stiffness_summary", pd.DataFrame())
    if not finger.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        subject_finger = tables.get("probing_subject_finger_stiffness_summary", pd.DataFrame())
        for finger_name, g in finger.groupby("finger_condition", dropna=False):
            g = g.sort_values("stiffness_value")
            ax.plot(g["stiffness_value"], g["mean_probe_count"], marker="o", label=str(finger_name))
            if not subject_finger.empty:
                raw = subject_finger[subject_finger["finger_condition"].astype(str) == str(finger_name)]
                ax.scatter(raw["stiffness_value"], raw["mean_probe_count"], s=12, alpha=0.18)
        ax.set_xlabel("Stiffness value")
        ax.set_ylabel("Mean probe count")
        ax.set_title("Within-finger probing by stiffness")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = fig_dir / "finger_probe_count_by_stiffness.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

    requested_subject = tables.get("probing_subject_finger_stiffness_probe_metrics_summary", pd.DataFrame())
    requested_group = tables.get("probing_group_finger_stiffness_probe_metrics_summary", pd.DataFrame())
    if not requested_group.empty:
        metric_plots = [
            (
                "mean_probe_count_per_segment",
                "Mean probes per stiffness segment",
                "requested_probe_count_by_finger_stiffness_group.png",
            ),
            (
                "median_probe_jerk_px_s3",
                "Median probe jerk (px/s³)",
                "requested_probe_jerk_by_finger_stiffness_group.png",
            ),
            (
                "median_probe_straightness_score",
                "Median straightness score (0 messy, 1 straight)",
                "requested_probe_straightness_by_finger_stiffness_group.png",
            ),
        ]
        for metric, ylabel, filename in metric_plots:
            if metric not in requested_group:
                continue
            plot_df = requested_group.dropna(subset=["finger_condition", "stiffness_value", metric]).copy()
            if plot_df.empty:
                continue
            fingers = list(plot_df["finger_condition"].dropna().astype(str).unique())
            ncols = min(3, max(1, len(fingers)))
            nrows = int(math.ceil(len(fingers) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False, sharey=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, finger_name in zip(axes.ravel(), fingers):
                ax.set_visible(True)
                g_finger = plot_df[plot_df["finger_condition"].astype(str) == finger_name]
                if not requested_subject.empty and metric in requested_subject:
                    raw = requested_subject[
                        requested_subject["finger_condition"].astype(str) == finger_name
                    ].dropna(subset=["stiffness_value", metric])
                    if not raw.empty:
                        jitter = np.random.default_rng(19).normal(0, 0.35, len(raw))
                        ax.scatter(
                            pd.to_numeric(raw["stiffness_value"], errors="coerce") + jitter,
                            pd.to_numeric(raw[metric], errors="coerce"),
                            s=14,
                            alpha=0.18,
                            color="0.25",
                            label="subject summaries",
                        )
                for group_name, g in g_finger.groupby(EXPERIMENT_GROUP_COLUMN, dropna=False):
                    g = g.sort_values("stiffness_value")
                    ax.plot(
                        pd.to_numeric(g["stiffness_value"], errors="coerce"),
                        pd.to_numeric(g[metric], errors="coerce"),
                        marker="o",
                        linewidth=2,
                        label=str(group_name),
                    )
                if "straightness" in metric:
                    ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"Finger: {finger_name}")
                ax.set_xlabel("Stiffness value")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            fig.suptitle(ylabel + " by finger/stiffness/group", y=1.02)
            fig.tight_layout()
            out = fig_dir / filename
            fig.savefig(out, dpi=fig_dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(out)

    success_by_stiffness = tables.get("probing_success_by_stiffness", pd.DataFrame())
    if not success_by_stiffness.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        if not trials.empty:
            raw = trials[["stiffness_value", "correct_response", "probe_count"]].dropna()
            for success_value, g_raw in raw.groupby("correct_response"):
                offset = -0.35 if success_value == 0 else 0.35
                ax.scatter(g_raw["stiffness_value"] + offset, g_raw["probe_count"], s=8, alpha=0.06, color="0.25")
        for success_value, g in success_by_stiffness.groupby("correct_response", dropna=False):
            g = g.sort_values("stiffness_value")
            label = "success" if success_value == 1 else "failure" if success_value == 0 else "missing"
            ax.plot(g["stiffness_value"], g["mean_probe_count"], marker="o", label=label)
        ax.set_xlabel("Stiffness value")
        ax.set_ylabel("Mean probe count")
        ax.set_title("Probe count by success/failure and stiffness")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = fig_dir / "success_failure_probe_count_by_stiffness.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

    anova = tables.get("probing_success_one_way_anova", pd.DataFrame())
    anova_summary = tables.get("probing_success_anova_factor_summary", pd.DataFrame())
    if not anova.empty:
        plot_anova = anova[
            (anova["observation_level"] == "trial")
            & (anova[ANALYSIS_SCOPE_COLUMN].isin(["all", "experiment_group"]))
            & (anova["status"].astype(str).str.startswith("ok"))
        ].copy()
        if not plot_anova.empty:
            plot_anova["scope_label"] = np.where(
                plot_anova[ANALYSIS_SCOPE_COLUMN] == "all",
                "all",
                plot_anova[ANALYSIS_SCOPE_VALUE_COLUMN].astype(str),
            )
            plot_anova["neg_log10_p"] = -np.log10(pd.to_numeric(plot_anova["p_value"], errors="coerce").clip(lower=1e-300))
            pivot = plot_anova.pivot_table(index="factor", columns="scope_label", values="neg_log10_p", aggfunc="first")
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(9, 5))
                pivot.plot(kind="bar", ax=ax)
                ax.axhline(-math.log10(ANOVA_ALPHA), color="red", linestyle="--", linewidth=1, label=f"p={ANOVA_ALPHA}")
                ax.set_xlabel("One-way ANOVA factor")
                ax.set_ylabel("-log10(p-value)")
                ax.set_title("Success-rate one-way ANOVA significance by scope")
                ax.legend(fontsize=8)
                ax.grid(axis="y", alpha=0.25)
                fig.tight_layout()
                out = fig_dir / "success_anova_pvalues_by_scope.png"
                fig.savefig(out, dpi=fig_dpi)
                plt.close(fig)
                paths.append(out)

    if not anova_summary.empty:
        summary_plot = anova_summary[
            (anova_summary["observation_level"] == "trial")
            & (anova_summary[ANALYSIS_SCOPE_COLUMN].isin(["all", "experiment_group"]))
        ].copy()
        if not summary_plot.empty:
            summary_plot["scope_label"] = np.where(
                summary_plot[ANALYSIS_SCOPE_COLUMN] == "all",
                "all",
                summary_plot[ANALYSIS_SCOPE_VALUE_COLUMN].astype(str),
            )
            fdat = summary_plot[summary_plot["factor"] == "amount_of_probing"].copy()
            if not fdat.empty:
                order = ["0", "1", "2", "3", "4+"]
                fdat["factor_level"] = pd.Categorical(fdat["factor_level"].astype(str), categories=order, ordered=True)
                pivot = fdat.pivot_table(index="factor_level", columns="scope_label", values="mean_success_rate", aggfunc="first", observed=False)
                pivot = pivot.reindex([x for x in order if x in set(fdat["factor_level"].astype(str))])
                if not pivot.empty:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    pivot.plot(kind="bar", ax=ax)
                    ax.set_xlabel("Probe count bin")
                    ax.set_ylabel("Success rate")
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_title("Success rate by amount of probing (ANOVA factor summary)")
                    ax.legend(fontsize=8)
                    ax.grid(axis="y", alpha=0.25)
                    fig.tight_layout()
                    out = fig_dir / "success_by_probe_count_anova.png"
                    fig.savefig(out, dpi=fig_dpi)
                    plt.close(fig)
                    paths.append(out)

            for factor_name, factor_col, filename, xlabel in [
                ("stiffness_value", "stiffness_value", "success_by_stiffness_anova.png", "Stiffness value"),
            ]:
                fdat = summary_plot[summary_plot["factor"] == factor_name].copy()
                if fdat.empty:
                    continue
                fdat["factor_level_numeric"] = pd.to_numeric(fdat["factor_level"], errors="coerce")
                fdat = fdat[fdat["factor_level_numeric"].notna()].sort_values(["scope_label", "factor_level_numeric"])
                if fdat.empty:
                    continue
                fig, ax = plt.subplots(figsize=(10, 5))
                for scope_label, g in fdat.groupby("scope_label", dropna=False):
                    ax.plot(g["factor_level_numeric"], g["mean_success_rate"], marker="o", label=str(scope_label))
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Success rate")
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"Success rate by {xlabel.lower()} (ANOVA factor summary)")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.25)
                fig.tight_layout()
                out = fig_dir / filename
                fig.savefig(out, dpi=fig_dpi)
                plt.close(fig)
                paths.append(out)

            fdat = summary_plot[summary_plot["factor"] == "finger"].copy()
            if not fdat.empty:
                pivot = fdat.pivot_table(index="factor_level", columns="scope_label", values="mean_success_rate", aggfunc="first")
                if not pivot.empty:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    pivot.plot(kind="bar", ax=ax)
                    ax.set_xlabel("Finger")
                    ax.set_ylabel("Success rate")
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_title("Success rate by finger (ANOVA factor summary)")
                    ax.legend(fontsize=8)
                    ax.grid(axis="y", alpha=0.25)
                    fig.tight_layout()
                    out = fig_dir / "success_by_finger_anova.png"
                    fig.savefig(out, dpi=fig_dpi)
                    plt.close(fig)
                    paths.append(out)

    subject_finger = tables.get("probing_subject_finger_stiffness_summary", pd.DataFrame())
    if not subject_finger.empty:
        heat = subject_finger.copy()
        heat["subject_finger"] = heat["subject_id"].astype(str) + "_" + heat["finger_condition"].astype(str)
        pivot = heat.pivot_table(index="subject_finger", columns="stiffness_value", values="mean_probe_count", aggfunc="mean")
        if not pivot.empty:
            fig_height = min(18, max(5, 0.22 * len(pivot)))
            fig, ax = plt.subplots(figsize=(10, fig_height))
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=6)
            ax.set_xlabel("Stiffness value")
            ax.set_ylabel("Subject_finger")
            ax.set_title("Within-subject/finger mean probe count")
            fig.colorbar(im, ax=ax, label="Mean probe count")
            fig.tight_layout()
            out = fig_dir / "subject_finger_probe_count_heatmap.png"
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

    direction = tables.get("probing_direction_summary", pd.DataFrame())
    if not direction.empty:
        counts = direction.groupby("peak_direction", dropna=False)["n_probe_events"].sum().reindex(DIRECTION_LABELS_8 + ["unknown"]).dropna()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Probe peak direction")
        ax.set_ylabel("Number of probe events")
        ax.set_title("Direction distribution of center-to-side probes")
        fig.tight_layout()
        out = fig_dir / "probe_direction_distribution.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

    scope_manifest = save_scope_summary_plots(
        tables,
        output_root,
        namespace="probing",
        metrics=PROBING_GROUP_METRICS,
        fig_dpi=fig_dpi,
    )
    if not scope_manifest.empty:
        paths.extend(Path(p) for p in scope_manifest["figure"])
    save_csv(pd.DataFrame({"figure": [str(p) for p in paths]}), output_root, "figure_manifest.csv")
    return paths


def analysis_manifest(output_root: Path) -> pd.DataFrame:
    expected = [
        "probing_event_log.csv",
        "probing_trial_summary.csv",
        "probing_subject_finger_stiffness_summary.csv",
        "probing_stiffness_summary.csv",
        "probing_finger_stiffness_summary.csv",
        "probing_comparison_summary.csv",
        "probing_success_summary.csv",
        "probing_success_by_stiffness.csv",
        "probing_subject_finger_correlations.csv",
        "probing_direction_summary.csv",
        "probing_group_metric_summary.csv",
        "probing_group_condition_metric_summary.csv",
        "probing_within_group_condition_comparisons.csv",
        "probing_between_group_metric_comparisons.csv",
        "probing_analysis_scope_metric_summary.csv",
        "probing_analysis_scope_condition_metric_summary.csv",
        "probing_within_analysis_scope_condition_comparisons.csv",
        "probing_between_analysis_scope_metric_comparisons.csv",
        "probing_success_one_way_anova.csv",
        "probing_success_anova_factor_summary.csv",
        "probing_success_anova_pairwise.csv",
        "probing_subject_finger_stiffness_probe_metrics_summary.csv",
        "probing_group_finger_stiffness_probe_metrics_summary.csv",
        "probing_analysis_scope_finger_stiffness_probe_metrics_summary.csv",
        "probing_setup_balance.csv",
        "probing_setup_metric_summary.csv",
        "probing_setup_condition_metric_summary.csv",
        "probing_between_setup_metric_comparisons.csv",
        "probing_scope_figure_manifest.csv",
        "figure_manifest.csv",
    ]
    return pd.DataFrame(
        {
            "output": expected,
            "exists": [(output_root / x).exists() for x in expected],
            "path": [str(output_root / x) for x in expected],
        }
    )


def run_analysis(
    kinematics_results: Path,
    output_root: Path,
    *,
    center_radius_px: float = DEFAULT_CENTER_RADIUS_PX,
    side_radius_px: float = DEFAULT_SIDE_RADIUS_PX,
    min_probe_duration_s: float = DEFAULT_MIN_PROBE_DURATION_S,
    fig_dpi: int = 160,
) -> dict[str, pd.DataFrame]:
    samples, trials = load_kinematic_inputs(kinematics_results)
    tables = compute_probing_metrics(
        samples,
        trials,
        center_radius_px=center_radius_px,
        side_radius_px=side_radius_px,
        min_probe_duration_s=min_probe_duration_s,
    )
    for key, df in tables.items():
        save_csv(df, output_root, f"{key}.csv")
    save_figures(output_root, tables, fig_dpi=fig_dpi)
    manifest = analysis_manifest(output_root)
    save_csv(manifest, output_root, "analysis_manifest.csv")
    tables["analysis_manifest"] = manifest
    return tables


def _default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    return project_root / "analysis" / "Kinematics" / "results", here.parent / "results" / "probing"


def main(argv: list[str] | None = None) -> int:
    default_kinematics, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Run center-to-side probing analysis.")
    parser.add_argument(
        "--analysis-kind",
        choices=["probing", "direction-success", "direction-batch", "both"],
        default="probing",
        help=(
            "probing keeps the legacy center-to-side summaries; direction-success "
            "runs one thesis perception-action success analysis; direction-batch "
            "runs L_N_E/L_E/N_E filtered and non-filtered scopes plus subjects; "
            "both runs probing and one direction-success scope."
        ),
    )
    parser.add_argument("--kinematics-results", type=Path, default=default_kinematics)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_kinematics.parents[2] / "results",
        help="Raw subject data root used by direction-success analysis.",
    )
    parser.add_argument(
        "--selection",
        default="FILTER_ONLY",
        help="Subject/group selection, e.g. FILTER_ONLY, L_E, N_E, L_N_E, or L_E_1.",
    )
    parser.add_argument(
        "--exclude-filter-folders",
        action="store_true",
        help="Mirror kinematics/psychophysics cohort filtering for direction-success analysis.",
    )
    parser.add_argument(
        "--direction-output-root",
        type=Path,
        default=None,
        help="Optional explicit output folder for direction-success CSVs/figures.",
    )
    parser.add_argument(
        "--batch-output-root",
        type=Path,
        default=None,
        help="Optional root folder for direction-batch outputs.",
    )
    parser.add_argument("--center-radius-px", type=float, default=DEFAULT_CENTER_RADIUS_PX)
    parser.add_argument("--side-radius-px", type=float, default=DEFAULT_SIDE_RADIUS_PX)
    parser.add_argument("--min-probe-duration-s", type=float, default=DEFAULT_MIN_PROBE_DURATION_S)
    parser.add_argument("--fig-dpi", type=int, default=160)
    args = parser.parse_args(argv)

    if args.analysis_kind in {"probing", "both"}:
        tables = run_analysis(
            args.kinematics_results,
            args.output_root,
            center_radius_px=args.center_radius_px,
            side_radius_px=args.side_radius_px,
            min_probe_duration_s=args.min_probe_duration_s,
            fig_dpi=args.fig_dpi,
        )
        print("Saved probing analysis to", args.output_root)
        for name, df in tables.items():
            print(f"{name}: {df.shape}")

    if args.analysis_kind in {"direction-success", "both"}:
        direction_tables = run_full_analysis(
            args.data_root,
            output_root=args.direction_output_root,
            selection=args.selection,
            exclude_filter_folders=args.exclude_filter_folders,
        )
        direction_root = direction_tables["output_root"]
        print("Saved direction-success analysis to", direction_root)
        for name, df in direction_tables.items():
            if isinstance(df, pd.DataFrame):
                print(f"{name}: {df.shape}")
    if args.analysis_kind == "direction-batch":
        batch = run_direction_success_batch(
            args.data_root,
            batch_output_root=args.batch_output_root,
        )
        print("Saved direction-success batch to", batch["batch_output_root"])
        print(batch["batch_manifest"].to_string(index=False))
    return 0


# ===========================================================================
# Perception-vs-action direction analysis (merged from perception_action_analysis.py)
# Derives per-pair movement direction (same/different) from tracking and fits
# separate psychometric curves; tests whether different-direction exploration
# biases the PSE. perception_action_analysis.py now re-exports these names.
# ===========================================================================
# --- Constants ---------------------------------------------------------------
CENTER_X = ka.CENTER_X
CENTER_Y = ka.CENTER_Y

# A pair counts as "same direction" when the two stimulus segments' dominant
# headings differ by at most this angle, and "different" when they differ by at
# least the complementary threshold. Pairs in the grey band are labelled
# "ambiguous" and excluded from the binary contrast (but kept in the trial log).
SAME_DIRECTION_MAX_DEG = 45.0
DIFFERENT_DIRECTION_MIN_DEG = 90.0

# Minimum samples in a stimulus segment before its direction is trusted.
MIN_SEGMENT_SAMPLES = 5

RESULTS_DIRNAME = "results"


def _results_root() -> Path:
    return Path(__file__).resolve().parent / RESULTS_DIRNAME / "probing"


# --- Geometry helpers --------------------------------------------------------
def segment_dominant_direction(
    seg: pd.DataFrame,
    *,
    center_x: float = CENTER_X,
    center_y: float = CENTER_Y,
) -> dict[str, Any]:
    """Dominant outbound heading of one stimulus segment.

    The cube position (object_x, object_y) is centred, and the dominant heading
    is the angle of the net displacement from the centre to the point of maximum
    radial excursion, i.e. the outbound stroke direction the participant chose.
    Returns the heading in degrees (atan2, image-y-down flipped to maths-y-up),
    the maximum radius reached, and the sample count.
    """
    x = pd.to_numeric(seg.get("object_x"), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(seg.get("object_y"), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < MIN_SEGMENT_SAMPLES:
        return {"direction_deg": np.nan, "max_radius_px": np.nan, "n_samples": int(x.size)}
    dx = x - center_x
    dy = -(y - center_y)  # image y grows downward; flip to standard orientation
    r = np.hypot(dx, dy)
    i_max = int(np.nanargmax(r))
    direction = math.degrees(math.atan2(dy[i_max], dx[i_max]))
    return {
        "direction_deg": float(direction),
        "max_radius_px": float(r[i_max]),
        "n_samples": int(x.size),
    }


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest unsigned angle (0..180 deg) between two headings."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return float(d)


def classify_direction_pair(diff_deg: float) -> str:
    if not np.isfinite(diff_deg):
        return "unknown"
    if diff_deg <= SAME_DIRECTION_MAX_DEG:
        return "same_direction"
    if diff_deg >= DIFFERENT_DIRECTION_MIN_DEG:
        return "different_direction"
    return "ambiguous"


# --- Per-pair direction table ------------------------------------------------
def build_pair_direction_table(
    trials: pd.DataFrame,
    *,
    center_x: float = CENTER_X,
    center_y: float = CENTER_Y,
) -> pd.DataFrame:
    """One row per comparison pair, with the two stimulus headings and their gap.

    For each pair the tracking.csv is split into its two stimulus segments by the
    ``stiffness`` column (each contiguous non-zero stiffness-value block is one object), the
    dominant heading of each segment is computed, and the pair is classified
    same / different / ambiguous. The 2AFC metadata from ``trials`` is carried
    through so the result can be re-aggregated into psychometric input.
    """
    output_columns = [
        "subject_id",
        "subject_group",
        EXPERIMENT_GROUP_COLUMN,
        "trial_index_raw",
        "pair_number",
        "object_1_finger",
        "object_2_finger",
        "object_1_stiffness",
        "object_2_stiffness",
        "finger_condition",
        "comparison_value",
        "standard_value",
        "signed_stiffness_delta",
        "answer_code",
        "correct_response",
        "time_to_answer_s",
        "n_segments_detected",
        "object_1_direction_deg",
        "object_2_direction_deg",
        "object_1_direction_label",
        "object_2_direction_label",
        "standard_direction_deg",
        "standard_direction_label",
        "comparison_direction_deg",
        "comparison_direction_label",
        "direction_difference_deg",
        "direction_pair_class",
        "direction_warning",
    ]
    if trials.empty or "tracking_exists" not in trials:
        return pd.DataFrame(columns=output_columns)
    selected = trials[trials.get("tracking_exists", False).astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for _, meta in selected.iterrows():
        rec: dict[str, Any] = {
            k: meta.get(k)
            for k in (
                "subject_id",
                "subject_group",
                EXPERIMENT_GROUP_COLUMN,
                "trial_index_raw",
                "pair_number",
                "object_1_finger",
                "object_2_finger",
                "object_1_stiffness",
                "object_2_stiffness",
                "finger_condition",
                "comparison_value",
                "standard_value",
                "signed_stiffness_delta",
                "answer_code",
                "correct_response",
                "time_to_answer_s",
            )
        }
        try:
            df = pp.read_csv_flexible(Path(str(meta["tracking_file"])))
        except Exception as exc:  # noqa: BLE001
            rec["direction_warning"] = f"read_failed: {exc}"
            rows.append(rec)
            continue
        if not {"object_x", "object_y", "stiffness"}.issubset(df.columns):
            rec["direction_warning"] = "missing tracking columns"
            rows.append(rec)
            continue

        d = df.copy()
        d["stiffness_num"] = pd.to_numeric(d["stiffness"], errors="coerce").fillna(0.0)
        # Each contiguous non-zero stiffness-value block is one stimulus segment.
        # The experiment often switches directly from object 1 to object 2
        # (for example, 115 -> 85) without an intervening zero.  Splitting only
        # on non-zero active/inactive transitions therefore merges the two
        # stimuli and leaves the pair unclassified.
        active = d["stiffness_num"].to_numpy() != 0.0
        stiffness_values = d["stiffness_num"].to_numpy(dtype=float)
        starts = active & np.concatenate(
            [[True], (~active[:-1]) | (stiffness_values[1:] != stiffness_values[:-1])]
        )
        seg_id = np.cumsum(starts)
        d["_seg"] = np.where(active, seg_id, 0)
        seg_dirs: list[dict[str, Any]] = []
        for sid, seg in d[d["_seg"] > 0].groupby("_seg"):
            info = segment_dominant_direction(seg, center_x=center_x, center_y=center_y)
            info["stiffness_value"] = float(
                pd.to_numeric(seg["stiffness_num"], errors="coerce").median()
            )
            seg_dirs.append(info)

        rec["n_segments_detected"] = len(seg_dirs)
        if len(seg_dirs) < 2:
            rec["direction_warning"] = "fewer than two stimulus segments"
            rows.append(rec)
            continue

        # Use the first two detected segments as object 1 and object 2.
        s1, s2 = seg_dirs[0], seg_dirs[1]
        rec["object_1_direction_deg"] = s1["direction_deg"]
        rec["object_2_direction_deg"] = s2["direction_deg"]
        rec["object_1_direction_label"] = _direction_label(s1["direction_deg"])
        rec["object_2_direction_label"] = _direction_label(s2["direction_deg"])
        rec["object_1_max_radius_px"] = s1["max_radius_px"]
        rec["object_2_max_radius_px"] = s2["max_radius_px"]
        rec["object_max_radius_difference_px"] = (
            float(s1["max_radius_px"] - s2["max_radius_px"])
            if np.isfinite(s1["max_radius_px"]) and np.isfinite(s2["max_radius_px"])
            else np.nan
        )
        diff = angular_difference_deg(s1["direction_deg"], s2["direction_deg"])
        rec["direction_difference_deg"] = diff
        rec["direction_pair_class"] = classify_direction_pair(diff)
        rec["object_1_stimulus_role"] = _stimulus_role_for_object(pd.Series(rec), 1)
        rec["object_2_stimulus_role"] = _stimulus_role_for_object(pd.Series(rec), 2)
        for object_number in (1, 2):
            role = rec.get(f"object_{object_number}_stimulus_role")
            if role in {"standard", "comparison"}:
                rec[f"{role}_direction_deg"] = rec.get(f"object_{object_number}_direction_deg")
                rec[f"{role}_direction_label"] = rec.get(
                    f"object_{object_number}_direction_label"
                )
                rec[f"{role}_max_radius_px"] = rec.get(f"object_{object_number}_max_radius_px")
        rec["direction_warning"] = ""
        rows.append(rec)
    return pd.DataFrame(rows)


# --- Psychometric input split by direction class -----------------------------
def _response_comparison_greater(row: pd.Series) -> float:
    """1 if the participant judged the comparison object stiffer, else 0, else NaN.

    Mirrors twoafc_psychophysics: answer 0 selects object 1, answer 1 selects
    object 2; the comparison is the non-standard object of the pair.
    """
    ans = row.get("answer_code")
    s1 = row.get("object_1_stiffness")
    s2 = row.get("object_2_stiffness")
    std = row.get("standard_value")
    if not (np.isfinite(ans) and np.isfinite(s1) and np.isfinite(s2) and np.isfinite(std)):
        return np.nan
    chosen = s1 if int(ans) == 0 else s2 if int(ans) == 1 else np.nan
    if not np.isfinite(chosen):
        return np.nan
    # comparison stiffness = the object whose stiffness differs from the standard
    comp_is_obj1 = abs(s1 - std) > 1e-9
    chosen_is_comparison = (chosen == s1) == comp_is_obj1
    return float(chosen_is_comparison)


def make_direction_psychometric_input(
    pair_table: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Per-level binomial counts (n_trials, n_comparison_greater) per group.

    ``group_cols`` typically includes ``direction_pair_class`` so the same-,
    different-, and pooled curves can be fitted from one table with the same
    estimator used in Chapter 6 (twoafc_psychophysics.fit_with_scipy_logistic).
    """
    df = pair_table.copy()
    df["comparison_value"] = pd.to_numeric(df["comparison_value"], errors="coerce")
    df["response_comparison_greater"] = df.apply(_response_comparison_greater, axis=1)
    df = df.dropna(subset=["comparison_value", "response_comparison_greater"])
    grouped = (
        df.groupby(group_cols + ["comparison_value"], dropna=False)[
            "response_comparison_greater"
        ]
        .agg(n_trials="count", n_comparison_greater="sum")
        .reset_index()
    )
    grouped["n_comparison_greater"] = grouped["n_comparison_greater"].astype(float)
    grouped["proportion_comparison_greater"] = np.where(
        grouped["n_trials"] > 0,
        grouped["n_comparison_greater"] / grouped["n_trials"],
        np.nan,
    )
    return grouped


def fit_curves_by_direction(
    psychometric_input: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Fit one psychometric curve per group with the project estimator.

    Returns one row per group with PSE, PSE delta from the standard, JND, Weber
    fraction, slope at PSE, lapse, n_trials, and the fit method/warning, using
    psignifit when available and the four-parameter-logistic fallback otherwise.
    """
    psignifit_available, _ = pp.check_psignifit_available()
    rows: list[dict[str, Any]] = []
    for keys, agg in psychometric_input.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys))
        fit = None
        if psignifit_available:
            fit, _ = pp.fit_with_psignifit_if_possible(agg, psignifit_available)
        if fit is None:
            # Same estimator as Chapter 6: four-parameter logistic by MLE.
            fit = pp.fit_with_scipy_logistic(agg, n_bootstrap=0)
        pse = fit.get("pse", np.nan)
        jnd = fit.get("jnd", np.nan)
        rec.update(
            {
                "pse": pse,
                "pse_delta": fit.get(
                    "pse_delta",
                    (pse - pp.STANDARD_FALLBACK) if np.isfinite(pse) else np.nan,
                ),
                "jnd": jnd,
                "weber_fraction": fit.get(
                    "weber_fraction",
                    (jnd / pp.STANDARD_FALLBACK) if np.isfinite(jnd) else np.nan,
                ),
                "slope_at_pse": fit.get("slope_at_pse", np.nan),
                "lapse_low": fit.get("lapse_low", np.nan),
                "lapse_high": fit.get("lapse_high", np.nan),
                "n_trials": fit.get("n_trials", agg["n_trials"].sum()),
                "fit_method": fit.get("fit_method", "unknown"),
                "fit_warning": fit.get("fit_warning", ""),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _wilson_ci(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if not (np.isfinite(successes) and np.isfinite(n)) or n <= 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1.0 + (z**2 / n)
    centre = (p + (z**2 / (2.0 * n))) / denom
    half_width = z * math.sqrt((p * (1.0 - p) / n) + (z**2 / (4.0 * n**2))) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def compute_direction_success_summary(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Success-rate summary for same/ambiguous/different movement directions."""
    df = pair_table.dropna(subset=["direction_pair_class", "correct_response"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "direction_pair_class",
                "n_trials",
                "n_subjects",
                "n_correct",
                "success_rate",
                "success_rate_ci_low",
                "success_rate_ci_high",
                "mean_direction_difference_deg",
                "same_vs_different_fisher_p",
            ]
        )
    summary = (
        df.groupby("direction_pair_class")
        .agg(
            n_trials=("correct_response", "size"),
            n_subjects=("subject_id", "nunique"),
            n_correct=("correct_response", "sum"),
            success_rate=("correct_response", "mean"),
            mean_direction_difference_deg=("direction_difference_deg", "mean"),
        )
        .reset_index()
    )
    ci = summary.apply(
        lambda row: _wilson_ci(float(row["n_correct"]), float(row["n_trials"])),
        axis=1,
        result_type="expand",
    )
    summary["success_rate_ci_low"] = ci[0]
    summary["success_rate_ci_high"] = ci[1]

    fisher_p = np.nan
    binary = df[df["direction_pair_class"].isin(["same_direction", "different_direction"])]
    if scipy_stats is not None and set(binary["direction_pair_class"]) == {
        "same_direction",
        "different_direction",
    }:
        table = []
        for cls in ["same_direction", "different_direction"]:
            values = binary.loc[binary["direction_pair_class"] == cls, "correct_response"]
            table.append([int((values == 1).sum()), int((values == 0).sum())])
        try:
            fisher_p = float(scipy_stats.fisher_exact(table, alternative="two-sided").pvalue)
        except Exception:  # pragma: no cover - scipy-specific edge cases
            fisher_p = np.nan
    summary["same_vs_different_fisher_p"] = fisher_p
    return summary


def _stimulus_role_for_object(row: pd.Series, object_number: int) -> str:
    object_stiffness = pd.to_numeric(
        pd.Series([row.get(f"object_{object_number}_stiffness")]), errors="coerce"
    ).iloc[0]
    comparison_value = pd.to_numeric(
        pd.Series([row.get("comparison_value")]), errors="coerce"
    ).iloc[0]
    standard_value = pd.to_numeric(
        pd.Series([row.get("standard_value")]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(object_stiffness):
        return "unknown"
    if np.isfinite(comparison_value) and abs(float(object_stiffness) - float(comparison_value)) <= 1e-9:
        return "comparison"
    if np.isfinite(standard_value) and abs(float(object_stiffness) - float(standard_value)) <= 1e-9:
        return "standard"
    return "unknown"


def build_direction_success_trial_table(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Long-form trial success table by stimulus role and 8-way direction.

    Each comparison pair contributes up to two rows: one for the standard object
    and one for the comparison object.  The trial-level success/failure is copied
    to both rows so direction can be summarized separately for standard and
    comparison exploration directions.
    """
    rows: list[dict[str, Any]] = []
    for _, row in pair_table.iterrows():
        if pd.isna(row.get("correct_response")):
            continue
        for object_number in (1, 2):
            direction_deg = row.get(f"object_{object_number}_direction_deg")
            direction_label = _direction_label(float(direction_deg)) if pd.notna(direction_deg) else "unknown"
            if direction_label == "unknown":
                continue
            role = _stimulus_role_for_object(row, object_number)
            rows.append(
                {
                    "subject_id": row.get("subject_id"),
                    "subject_group": row.get("subject_group"),
                    EXPERIMENT_GROUP_COLUMN: row.get(EXPERIMENT_GROUP_COLUMN),
                    "trial_index_raw": row.get("trial_index_raw"),
                    "pair_number": row.get("pair_number"),
                    "finger_condition": row.get("finger_condition"),
                    "stimulus_role": role,
                    "object_number": object_number,
                    "stiffness_value": row.get(f"object_{object_number}_stiffness"),
                    "comparison_value": row.get("comparison_value"),
                    "standard_value": row.get("standard_value"),
                    "signed_stiffness_delta": row.get("signed_stiffness_delta"),
                    "direction_deg": direction_deg,
                    "direction_label": direction_label,
                    "direction_pair_class": row.get("direction_pair_class"),
                    "direction_difference_deg": row.get("direction_difference_deg"),
                    "correct_response": row.get("correct_response"),
                }
            )
    return pd.DataFrame(rows)


def _safe_one_sample_tests(
    values: pd.Series,
    *,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Return paired/participant-level one-sample tests against zero.

    The analysis hypothesis is directional (same-direction success should exceed
    different-direction success; weak directions should be lower than others),
    but the table also carries two-sided p-values so thesis text can report the
    more conservative result when needed.
    """
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    out = {
        "n_subjects": float(len(x)),
        "mean_difference": float(x.mean()) if len(x) else np.nan,
        "median_difference": float(x.median()) if len(x) else np.nan,
        "sd_difference": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "sem_difference": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan,
        "t_statistic": np.nan,
        "t_p_two_sided": np.nan,
        "t_p_directional": np.nan,
        "wilcoxon_statistic": np.nan,
        "wilcoxon_p_two_sided": np.nan,
        "wilcoxon_p_directional": np.nan,
        "sign_test_p_directional": np.nan,
    }
    if len(x) < 2 or scipy_stats is None:
        return out
    try:
        t_res = scipy_stats.ttest_1samp(x, 0.0, alternative="two-sided")
        out["t_statistic"] = float(t_res.statistic)
        out["t_p_two_sided"] = float(t_res.pvalue)
        try:
            out["t_p_directional"] = float(
                scipy_stats.ttest_1samp(x, 0.0, alternative=alternative).pvalue
            )
        except TypeError:  # older scipy without alternative argument
            p_two = out["t_p_two_sided"]
            sign_ok = (out["t_statistic"] >= 0) if alternative == "greater" else (out["t_statistic"] <= 0)
            out["t_p_directional"] = float(p_two / 2.0 if sign_ok else 1.0 - p_two / 2.0)
    except Exception:  # pragma: no cover - scipy edge cases
        pass
    nonzero = x[np.abs(x) > 1e-12]
    if len(nonzero) >= 1:
        try:
            w_two = scipy_stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
            w_dir = scipy_stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
            out["wilcoxon_statistic"] = float(w_two.statistic)
            out["wilcoxon_p_two_sided"] = float(w_two.pvalue)
            out["wilcoxon_p_directional"] = float(w_dir.pvalue)
        except Exception:  # pragma: no cover - all-zero/tie edge cases
            pass
        try:
            positives = int((nonzero > 0).sum())
            n = int(len(nonzero))
            if alternative == "greater":
                out["sign_test_p_directional"] = float(
                    scipy_stats.binomtest(positives, n, 0.5, alternative="greater").pvalue
                )
            elif alternative == "less":
                out["sign_test_p_directional"] = float(
                    scipy_stats.binomtest(positives, n, 0.5, alternative="less").pvalue
                )
        except Exception:  # pragma: no cover
            pass
    return out


def compute_direction_match_participant_tables(
    pair_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Participant-normalized same-vs-different success tables.

    Trial-weighted Fisher tests can be dominated by subjects with more usable
    tracking.  These tables first compute each participant's same/different
    success rate, then test within-participant differences.
    """
    df = pair_table[
        pair_table.get("direction_pair_class", pd.Series(dtype=object)).isin(
            ["same_direction", "different_direction", "ambiguous"]
        )
    ].copy()
    if "correct_response" not in df or df.empty:
        empty = pd.DataFrame()
        return {
            "direction_match_success_by_subject": empty,
            "direction_match_same_vs_different_participant_contrast": empty,
        }
    for col in ["subject_group", EXPERIMENT_GROUP_COLUMN]:
        if col not in df:
            df[col] = np.nan
    df["correct_response"] = pd.to_numeric(df["correct_response"], errors="coerce")
    df = df.dropna(subset=["subject_id", "direction_pair_class", "correct_response"])
    per_subject = (
        df.groupby(["subject_id", "subject_group", EXPERIMENT_GROUP_COLUMN, "direction_pair_class"], dropna=False)
        .agg(
            n_trials=("correct_response", "size"),
            n_correct=("correct_response", "sum"),
            success_rate=("correct_response", "mean"),
            mean_direction_difference_deg=("direction_difference_deg", "mean"),
        )
        .reset_index()
    )
    binary = per_subject[
        per_subject["direction_pair_class"].isin(["same_direction", "different_direction"])
    ]
    if binary.empty:
        contrast = pd.DataFrame()
    else:
        pivot = binary.pivot_table(
            index=["subject_id", "subject_group", EXPERIMENT_GROUP_COLUMN],
            columns="direction_pair_class",
            values=["success_rate", "n_trials"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{cls}" for metric, cls in pivot.columns]
        contrast = pivot.reset_index()
        for col in [
            "success_rate_same_direction",
            "success_rate_different_direction",
            "n_trials_same_direction",
            "n_trials_different_direction",
        ]:
            if col not in contrast:
                contrast[col] = np.nan
        contrast["same_minus_different_success_rate"] = (
            contrast["success_rate_same_direction"]
            - contrast["success_rate_different_direction"]
        )
        test = _safe_one_sample_tests(
            contrast["same_minus_different_success_rate"],
            alternative="greater",
        )
        for key, value in test.items():
            contrast[key] = value
    return {
        "direction_match_success_by_subject": per_subject,
        "direction_match_same_vs_different_participant_contrast": contrast,
    }


def _summarize_direction_success(
    direction_trials: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    if direction_trials.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "n_trials",
                "n_subjects",
                "n_correct",
                "success_rate",
                "success_rate_ci_low",
                "success_rate_ci_high",
            ]
        )
    summary = (
        direction_trials.groupby(group_cols, dropna=False)
        .agg(
            n_trials=("correct_response", "size"),
            n_subjects=("subject_id", "nunique"),
            n_correct=("correct_response", "sum"),
            success_rate=("correct_response", "mean"),
        )
        .reset_index()
    )
    ci = summary.apply(
        lambda row: _wilson_ci(float(row["n_correct"]), float(row["n_trials"])),
        axis=1,
        result_type="expand",
    )
    summary["success_rate_ci_low"] = ci[0]
    summary["success_rate_ci_high"] = ci[1]
    return summary


def compute_direction_success_rate_tables(pair_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Direction-only success-rate tables for participant/stiffness summaries."""
    direction_trials = build_direction_success_trial_table(pair_table)
    subject_direction = _summarize_direction_success(
        direction_trials,
        ["subject_id", "stimulus_role", "direction_label"],
    )
    participant_mean = pd.DataFrame()
    if not subject_direction.empty:
        participant_mean = (
            subject_direction.groupby(["stimulus_role", "direction_label"], dropna=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                n_subject_direction_observations=("success_rate", "size"),
                total_trials=("n_trials", "sum"),
                mean_participant_success_rate=("success_rate", "mean"),
                sd_participant_success_rate=("success_rate", "std"),
                median_participant_success_rate=("success_rate", "median"),
            )
            .reset_index()
        )
        participant_mean["sem_participant_success_rate"] = (
            participant_mean["sd_participant_success_rate"]
            / np.sqrt(participant_mean["n_subject_direction_observations"].clip(lower=1))
        )
    direction_contrasts = compute_direction_label_participant_contrasts(
        direction_trials,
        subject_direction,
    )
    return {
        "direction_success_trials": direction_trials,
        "direction_success_by_direction": _summarize_direction_success(
            direction_trials,
            ["stimulus_role", "direction_label"],
        ),
        "direction_success_by_subject_direction": subject_direction,
        "direction_success_participant_mean_by_direction": participant_mean,
        "direction_success_by_participant_stiffness": _summarize_direction_success(
            direction_trials,
            ["subject_id", "stimulus_role", "stiffness_value", "direction_label"],
        ),
        "direction_success_by_stiffness": _summarize_direction_success(
            direction_trials,
            ["stimulus_role", "stiffness_value", "direction_label"],
        ),
        "direction_success_participant_direction_contrasts": direction_contrasts,
    }


def compute_direction_label_participant_contrasts(
    direction_trials: pd.DataFrame,
    subject_direction: pd.DataFrame,
) -> pd.DataFrame:
    """Within-subject direction-vs-other contrasts for each 8-way direction.

    For every role (standard/comparison) and direction, this compares each
    participant's success in that direction with the same participant's success
    in all other directions for that role.  This is the direction-only analogue
    of a paired analysis and is more defensible than raw trial-weighted bars.
    """
    if direction_trials.empty or subject_direction.empty:
        return pd.DataFrame(
            columns=[
                "stimulus_role",
                "direction_label",
                "n_subjects",
                "mean_direction_success_rate",
                "mean_other_success_rate",
                "mean_direction_minus_other",
                "direction_rank_by_mean",
                "t_p_two_sided",
                "wilcoxon_p_two_sided",
                "wilcoxon_p_directional_less",
            ]
        )
    rows: list[dict[str, Any]] = []
    d = direction_trials.dropna(subset=["subject_id", "stimulus_role", "direction_label"]).copy()
    d["correct_response"] = pd.to_numeric(d["correct_response"], errors="coerce")
    for role in sorted(d["stimulus_role"].dropna().astype(str).unique()):
        role_trials = d[d["stimulus_role"].astype(str) == role]
        role_means = (
            subject_direction[subject_direction["stimulus_role"].astype(str) == role]
            .groupby("direction_label", dropna=False)["success_rate"]
            .mean()
            .sort_values(ascending=False)
        )
        ranks = {str(direction): int(rank + 1) for rank, direction in enumerate(role_means.index)}
        for direction in DIRECTION_LABELS_8:
            diffs: list[float] = []
            direction_rates: list[float] = []
            other_rates: list[float] = []
            for subject, subject_trials in role_trials.groupby("subject_id", dropna=False):
                in_dir = subject_trials[subject_trials["direction_label"] == direction]
                other = subject_trials[subject_trials["direction_label"] != direction]
                if in_dir.empty or other.empty:
                    continue
                dir_rate = float(in_dir["correct_response"].mean())
                other_rate = float(other["correct_response"].mean())
                direction_rates.append(dir_rate)
                other_rates.append(other_rate)
                diffs.append(dir_rate - other_rate)
            test = _safe_one_sample_tests(pd.Series(diffs, dtype=float), alternative="less")
            rows.append(
                {
                    "stimulus_role": role,
                    "direction_label": direction,
                    "n_subjects": int(len(diffs)),
                    "mean_direction_success_rate": float(np.mean(direction_rates))
                    if direction_rates
                    else np.nan,
                    "mean_other_success_rate": float(np.mean(other_rates))
                    if other_rates
                    else np.nan,
                    "mean_direction_minus_other": float(np.mean(diffs)) if diffs else np.nan,
                    "median_direction_minus_other": float(np.median(diffs)) if diffs else np.nan,
                    "direction_rank_by_mean": ranks.get(direction, np.nan),
                    "t_statistic": test["t_statistic"],
                    "t_p_two_sided": test["t_p_two_sided"],
                    "t_p_directional_less": test["t_p_directional"],
                    "wilcoxon_statistic": test["wilcoxon_statistic"],
                    "wilcoxon_p_two_sided": test["wilcoxon_p_two_sided"],
                    "wilcoxon_p_directional_less": test["wilcoxon_p_directional"],
                    "sign_test_p_directional_less": test["sign_test_p_directional"],
                }
            )
    return pd.DataFrame(rows)


def compute_optional_trial_logistic_models(
    pair_table: pd.DataFrame,
    direction_trials: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Optional trial-level logistic models with participant-clustered SEs.

    The paired participant summaries are the primary thesis evidence.  These
    models are secondary trial-level checks; they are skipped gracefully when
    statsmodels is unavailable or the selected scope is too small.
    """
    status_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    try:
        import statsmodels.api as sm  # type: ignore
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "direction_success_logistic_coefficients": pd.DataFrame(),
            "direction_success_logistic_status": pd.DataFrame(
                [
                    {
                        "model": "trial_success_direction_models",
                        "status": "skipped",
                        "reason": f"statsmodels unavailable: {exc}",
                    }
                ]
            ),
        }

    def fit_model(name: str, df: pd.DataFrame, formula: str) -> None:
        model_df = df.copy()
        model_df["correct_response"] = pd.to_numeric(
            model_df["correct_response"], errors="coerce"
        )
        model_df = model_df.dropna(subset=["correct_response", "subject_id"])
        if len(model_df) < 12 or model_df["correct_response"].nunique() < 2:
            status_rows.append(
                {
                    "model": name,
                    "status": "skipped",
                    "reason": "too few usable rows or no success/failure variation",
                    "n_rows": len(model_df),
                    "n_subjects": int(model_df["subject_id"].nunique())
                    if "subject_id" in model_df
                    else 0,
                }
            )
            return
        try:
            fit_kwargs: dict[str, Any] = {}
            if model_df["subject_id"].nunique() > 1:
                fit_kwargs = {
                    "cov_type": "cluster",
                    "cov_kwds": {"groups": model_df["subject_id"]},
                }
            fit = smf.glm(
                formula=formula,
                data=model_df,
                family=sm.families.Binomial(),
            ).fit(**fit_kwargs)
            params = fit.params
            conf = fit.conf_int()
            term_names = list(params.index) if hasattr(params, "index") else list(range(len(params)))
            for idx, term in enumerate(term_names):
                coef = params[term] if hasattr(params, "index") else params[idx]
                if hasattr(conf, "loc"):
                    ci_low = conf.loc[term, 0]
                    ci_high = conf.loc[term, 1]
                else:
                    ci_low = conf[idx, 0]
                    ci_high = conf[idx, 1]
                p_value = fit.pvalues[term] if hasattr(fit.pvalues, "index") else fit.pvalues[idx]
                coef_rows.append(
                    {
                        "model": name,
                        "term": term,
                        "coefficient_log_odds": float(coef),
                        "odds_ratio": float(math.exp(coef))
                        if np.isfinite(coef)
                        else np.nan,
                        "ci95_low_log_odds": float(ci_low),
                        "ci95_high_log_odds": float(ci_high),
                        "p_value": float(p_value),
                        "n_rows": int(len(model_df)),
                        "n_subjects": int(model_df["subject_id"].nunique()),
                    }
                )
            status_rows.append(
                {
                    "model": name,
                    "status": "ok",
                    "reason": "",
                    "n_rows": int(len(model_df)),
                    "n_subjects": int(model_df["subject_id"].nunique()),
                    "formula": formula,
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "model": name,
                    "status": "failed",
                    "reason": str(exc),
                    "n_rows": int(len(model_df)),
                    "n_subjects": int(model_df["subject_id"].nunique())
                    if "subject_id" in model_df
                    else 0,
                    "formula": formula,
                }
            )

    binary = pair_table[
        pair_table.get("direction_pair_class", pd.Series(dtype=object)).isin(
            ["same_direction", "different_direction"]
        )
    ].copy()
    if not binary.empty:
        binary["is_different_direction"] = (
            binary["direction_pair_class"] == "different_direction"
        ).astype(float)
        binary["abs_signed_stiffness_delta"] = pd.to_numeric(
            binary.get("signed_stiffness_delta", np.nan), errors="coerce"
        ).abs()
        fit_model(
            "trial_success_same_vs_different",
            binary,
            "correct_response ~ is_different_direction + abs_signed_stiffness_delta",
        )

    comparison = direction_trials[
        direction_trials.get("stimulus_role", pd.Series(dtype=object)).astype(str)
        == "comparison"
    ].copy()
    if not comparison.empty and comparison["direction_label"].nunique() >= 2:
        comparison["abs_signed_stiffness_delta"] = pd.to_numeric(
            comparison.get("signed_stiffness_delta", np.nan), errors="coerce"
        ).abs()
        fit_model(
            "trial_success_comparison_direction_8way",
            comparison,
            "correct_response ~ C(direction_label) + abs_signed_stiffness_delta",
        )

    return {
        "direction_success_logistic_coefficients": pd.DataFrame(coef_rows),
        "direction_success_logistic_status": pd.DataFrame(status_rows),
    }


def _pa_save_csv(df: pd.DataFrame, name: str, output_root: Optional[Path] = None) -> Path:
    root = Path(output_root) if output_root is not None else _results_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


# --- Figures -----------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    return plt


def save_direction_curve_figure(
    psychometric_input: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    output_root: Optional[Path] = None,
    filename: str = "psychometric_same_vs_different_direction.png",
) -> Optional[Path]:
    """Overlay the same- and different-direction psychometric curves.

    x = G_comparison - G_standard ; y = P(choose comparison > standard).
    """
    plt = _plt()
    root = (Path(output_root) if output_root is not None else _results_root()) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    colours = {"same_direction": "tab:green", "different_direction": "tab:red"}
    fig, ax = plt.subplots(figsize=(7, 5))
    std = pp.STANDARD_FALLBACK
    for cls, colour in colours.items():
        sub = psychometric_input[psychometric_input["direction_pair_class"] == cls]
        if sub.empty:
            continue
        x = sub["comparison_value"].to_numpy(dtype=float) - std
        y = sub["proportion_comparison_greater"].to_numpy(dtype=float)
        ax.scatter(x, y, s=18, color=colour, alpha=0.7, label=f"{cls} (data)")
        frow = fits[fits["direction_pair_class"] == cls]
        if not frow.empty and np.isfinite(frow.iloc[0].get("pse", np.nan)):
            ax.axvline(
                float(frow.iloc[0]["pse"]) - std,
                color=colour,
                linestyle="--",
                linewidth=1,
                label=f"{cls} PSE",
            )
    ax.axhline(0.5, color="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$G_{\mathrm{comparison}} - G_{\mathrm{standard}}$ (mm/m)")
    ax.set_ylabel(r"$P(\mathrm{choose\ comparison} > \mathrm{standard})$")
    ax.set_title("Psychometric curve by exploration-direction match")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = root / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_bias_vs_difference_figure(
    pair_table: pd.DataFrame,
    *,
    output_root: Optional[Path] = None,
    filename: str = "abs_bias_vs_direction_difference.png",
):
    """Per-subject |PSE| against mean between-object direction difference.

    A positive trend is the evidence the hypothesis predicts: participants who
    explored the two objects in more different directions show a larger bias.
    Returns (figure_path, merged_table).
    """
    plt = _plt()
    root = (Path(output_root) if output_root is not None else _results_root()) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    per_subject_input = make_direction_psychometric_input(pair_table, ["subject_id"])
    per_subject_fit = fit_curves_by_direction(per_subject_input, ["subject_id"])
    diff = (
        pair_table.groupby("subject_id")["direction_difference_deg"]
        .mean()
        .rename("mean_direction_difference_deg")
        .reset_index()
    )
    merged = per_subject_fit.merge(diff, on="subject_id", how="inner")
    merged["abs_pse_delta"] = merged["pse_delta"].abs()
    merged = merged.dropna(subset=["abs_pse_delta", "mean_direction_difference_deg"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if not merged.empty:
        ax.scatter(
            merged["mean_direction_difference_deg"],
            merged["abs_pse_delta"],
            s=30,
            alpha=0.75,
            color="tab:purple",
        )
        if len(merged) >= 3:
            from scipy import stats as _stats

            lr = _stats.linregress(
                merged["mean_direction_difference_deg"].to_numpy(dtype=float),
                merged["abs_pse_delta"].to_numpy(dtype=float),
            )
            xs = np.linspace(
                merged["mean_direction_difference_deg"].min(),
                merged["mean_direction_difference_deg"].max(),
                50,
            )
            ax.plot(
                xs,
                lr.intercept + lr.slope * xs,
                color="black",
                linewidth=1,
                label=f"slope={lr.slope:.3f}, r={lr.rvalue:.2f}, p={lr.pvalue:.3f}",
            )
            ax.legend(loc="best", fontsize=8)
            merged.attrs["trend_slope"] = float(lr.slope)
            merged.attrs["trend_r"] = float(lr.rvalue)
            merged.attrs["trend_p"] = float(lr.pvalue)
    ax.set_xlabel("Mean between-object direction difference (deg)")
    ax.set_ylabel("|PSE delta| (mm/m)")
    ax.set_title("Per-subject bias against exploration-direction mismatch")
    fig.tight_layout()
    out = root / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, merged


def save_direction_success_figure(
    success_summary: pd.DataFrame,
    *,
    output_root: Optional[Path] = None,
    filename: str = "success_rate_by_direction_match.png",
) -> Optional[Path]:
    """Bar plot of accuracy by same/ambiguous/different exploration direction."""
    if success_summary.empty:
        return None
    plt = _plt()
    root = (Path(output_root) if output_root is not None else _results_root()) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    order = ["same_direction", "ambiguous", "different_direction"]
    plot_df = (
        success_summary[success_summary["direction_pair_class"].isin(order)]
        .copy()
        .set_index("direction_pair_class")
        .reindex(order)
        .dropna(subset=["success_rate"])
        .reset_index()
    )
    if plot_df.empty:
        return None
    colours = {
        "same_direction": "tab:green",
        "ambiguous": "tab:orange",
        "different_direction": "tab:red",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    xs = np.arange(len(plot_df))
    y = plot_df["success_rate"].to_numpy(dtype=float)
    lower = y - plot_df["success_rate_ci_low"].to_numpy(dtype=float)
    upper = plot_df["success_rate_ci_high"].to_numpy(dtype=float) - y
    ax.bar(
        xs,
        y,
        yerr=np.vstack([lower, upper]),
        capsize=4,
        color=[colours.get(cls, "0.5") for cls in plot_df["direction_pair_class"]],
        alpha=0.85,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [cls.replace("_", " ") + f"\n(n={int(n)})" for cls, n in zip(plot_df["direction_pair_class"], plot_df["n_trials"])],
        fontsize=9,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate")
    ax.set_title("Lower success for different-direction exploration")
    ax.axhline(0.5, color="black", linewidth=0.7, alpha=0.5)
    fisher_values = plot_df["same_vs_different_fisher_p"].dropna()
    if not fisher_values.empty:
        ax.text(
            0.5,
            0.96,
            f"same vs different Fisher p={fisher_values.iloc[0]:.4f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    fig.tight_layout()
    out = root / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_comparison_direction_success_figure(
    direction_success_by_direction: pd.DataFrame,
    *,
    output_root: Optional[Path] = None,
    filename: str = "success_rate_by_comparison_direction.png",
) -> Optional[Path]:
    """Bar plot of success rate by the comparison object's 8-way direction."""
    if direction_success_by_direction.empty:
        return None
    plt = _plt()
    root = (Path(output_root) if output_root is not None else _results_root()) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    plot_df = (
        direction_success_by_direction[
            direction_success_by_direction["stimulus_role"] == "comparison"
        ]
        .copy()
        .set_index("direction_label")
        .reindex(DIRECTION_LABELS_8)
        .dropna(subset=["success_rate"])
        .reset_index()
    )
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xs = np.arange(len(plot_df))
    y = plot_df["success_rate"].to_numpy(dtype=float)
    lower = y - plot_df["success_rate_ci_low"].to_numpy(dtype=float)
    upper = plot_df["success_rate_ci_high"].to_numpy(dtype=float) - y
    ax.bar(xs, y, yerr=np.vstack([lower, upper]), capsize=4, color="tab:blue", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{direction}\n(n={int(n)})" for direction, n in zip(plot_df["direction_label"], plot_df["n_trials"])],
        fontsize=9,
    )
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="black", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Comparison-object movement direction")
    ax.set_title("Success rate by comparison-object direction")
    fig.tight_layout()
    out = root / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_comparison_direction_participant_mean_figure(
    participant_mean_by_direction: pd.DataFrame,
    *,
    output_root: Optional[Path] = None,
    filename: str = "participant_mean_success_rate_by_comparison_direction.png",
) -> Optional[Path]:
    """Bar plot after averaging success rates within each participant first."""
    if participant_mean_by_direction.empty:
        return None
    plt = _plt()
    root = (Path(output_root) if output_root is not None else _results_root()) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    plot_df = (
        participant_mean_by_direction[
            participant_mean_by_direction["stimulus_role"] == "comparison"
        ]
        .copy()
        .set_index("direction_label")
        .reindex(DIRECTION_LABELS_8)
        .dropna(subset=["mean_participant_success_rate"])
        .reset_index()
    )
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xs = np.arange(len(plot_df))
    y = plot_df["mean_participant_success_rate"].to_numpy(dtype=float)
    sem = plot_df["sem_participant_success_rate"].fillna(0.0).to_numpy(dtype=float)
    ax.bar(xs, y, yerr=sem, capsize=4, color="tab:cyan", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            f"{direction}\n(subj={int(n)})"
            for direction, n in zip(plot_df["direction_label"], plot_df["n_subjects"])
        ],
        fontsize=9,
    )
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="black", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("Mean participant success rate")
    ax.set_xlabel("Comparison-object movement direction")
    ax.set_title("Participant-mean success by comparison-object direction")
    fig.tight_layout()
    out = root / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --- Orchestration -----------------------------------------------------------
def run_full_analysis(
    data_root: Path,
    output_root: Optional[Path] = None,
    *,
    selection: Any = "FILTER_ONLY",
    exclude_filter_folders: bool = False,
    pilot_onboarding_trials: int = ka.PILOT_ONBOARDING_TRIALS,
) -> dict[str, Any]:
    """End-to-end: link trials, derive per-pair directions, split, fit, plot.

    Writes, under results/: per-pair directions; per-level binomial input by
    direction class; same/different/pooled PSE-JND fits; the same-finger subset
    fits; per-subject-by-direction fits; the subject bias-vs-difference table;
    participant-normalized success statistics; a figure manifest; and figures.
    Returns the in-memory tables.
    """
    if output_root is None:
        label = ka.selection_output_label(
            selection,
            exclude_filter_folders=exclude_filter_folders,
        )
        root = _results_root() / label / "perception_action"
    else:
        root = Path(output_root)
    trials = ka.discover_trials(
        Path(data_root),
        selection=selection,
        exclude_filter_folders=exclude_filter_folders,
        pilot_onboarding_trials=pilot_onboarding_trials,
    )
    pair_table = build_pair_direction_table(trials)
    pair_table["success"] = pd.to_numeric(pair_table.get("correct_response"), errors="coerce")
    pair_table["analysis_selection"] = str(selection)
    pair_table["exclude_filter_folders"] = bool(exclude_filter_folders)
    _pa_save_csv(pair_table, "perception_action_pair_directions", root)

    pair_table["any_direction"] = "pooled"
    psy_in = make_direction_psychometric_input(pair_table, ["direction_pair_class"])
    _pa_save_csv(psy_in, "perception_action_psychometric_input", root)
    by_dir = fit_curves_by_direction(psy_in, ["direction_pair_class"])
    pooled_in = make_direction_psychometric_input(pair_table, ["any_direction"])
    pooled_fit = fit_curves_by_direction(pooled_in, ["any_direction"]).rename(
        columns={"any_direction": "direction_pair_class"}
    )
    by_dir = pd.concat([by_dir, pooled_fit], ignore_index=True)
    _pa_save_csv(by_dir, "perception_action_pse_jnd_by_direction", root)

    same_finger = pair_table[pair_table["finger_condition"].notna()].copy()
    if not same_finger.empty:
        sf_in = make_direction_psychometric_input(
            same_finger, ["finger_condition", "direction_pair_class"]
        )
        sf_fit = fit_curves_by_direction(
            sf_in, ["finger_condition", "direction_pair_class"]
        )
        _pa_save_csv(sf_in, "perception_action_psychometric_input_same_finger", root)
        _pa_save_csv(sf_fit, "perception_action_pse_jnd_same_finger_by_direction", root)

    subj_in = make_direction_psychometric_input(
        pair_table, ["subject_id", "subject_group", "direction_pair_class"]
    )
    subj_fit = fit_curves_by_direction(
        subj_in, ["subject_id", "subject_group", "direction_pair_class"]
    )
    _pa_save_csv(subj_fit, "perception_action_pse_jnd_by_subject_direction", root)

    success_summary = compute_direction_success_summary(pair_table)
    _pa_save_csv(success_summary, "perception_action_success_by_direction", root)
    match_tables = compute_direction_match_participant_tables(pair_table)
    _pa_save_csv(
        match_tables["direction_match_success_by_subject"],
        "perception_action_direction_match_success_by_subject",
        root,
    )
    _pa_save_csv(
        match_tables["direction_match_same_vs_different_participant_contrast"],
        "perception_action_same_vs_different_participant_contrast",
        root,
    )
    direction_success_tables = compute_direction_success_rate_tables(pair_table)
    _pa_save_csv(
        direction_success_tables["direction_success_trials"],
        "perception_action_direction_success_trials",
        root,
    )
    _pa_save_csv(
        direction_success_tables["direction_success_by_direction"],
        "perception_action_direction_success_by_direction",
        root,
    )
    _pa_save_csv(
        direction_success_tables["direction_success_by_subject_direction"],
        "perception_action_direction_success_by_subject_direction",
        root,
    )
    _pa_save_csv(
        direction_success_tables["direction_success_participant_mean_by_direction"],
        "perception_action_direction_success_participant_mean_by_direction",
        root,
    )
    _pa_save_csv(
        direction_success_tables["direction_success_by_participant_stiffness"],
        "perception_action_direction_success_by_participant_stiffness",
        root,
    )
    # Explicit thesis-friendly alias for the requested participant x stiffness x
    # direction table.  It is identical to by_participant_stiffness but named in
    # the language used in the thesis analysis request.
    _pa_save_csv(
        direction_success_tables["direction_success_by_participant_stiffness"],
        "participant_x_stiffness_x_direction_success",
        root,
    )
    _pa_save_csv(
        direction_success_tables["direction_success_by_stiffness"],
        "perception_action_direction_success_by_stiffness",
        root,
    )
    _pa_save_csv(
        direction_success_tables["direction_success_participant_direction_contrasts"],
        "perception_action_direction_success_participant_direction_contrasts",
        root,
    )
    model_tables = compute_optional_trial_logistic_models(
        pair_table,
        direction_success_tables["direction_success_trials"],
    )
    for name, table in model_tables.items():
        _pa_save_csv(table, f"perception_action_{name}", root)

    fig1 = save_direction_curve_figure(psy_in, by_dir, output_root=root)
    fig2, merged = save_bias_vs_difference_figure(pair_table, output_root=root)
    fig3 = save_direction_success_figure(success_summary, output_root=root)
    fig4 = save_comparison_direction_success_figure(
        direction_success_tables["direction_success_by_direction"],
        output_root=root,
    )
    fig5 = save_comparison_direction_participant_mean_figure(
        direction_success_tables["direction_success_participant_mean_by_direction"],
        output_root=root,
    )
    _pa_save_csv(merged, "perception_action_subject_bias_vs_difference", root)

    manifest = pd.DataFrame(
        [
            {"figure": str(fig1), "source": "perception_action_pse_jnd_by_direction.csv"},
            {"figure": str(fig2), "source": "perception_action_subject_bias_vs_difference.csv"},
            {"figure": str(fig3), "source": "perception_action_success_by_direction.csv"},
            {"figure": str(fig4), "source": "perception_action_direction_success_by_direction.csv"},
            {"figure": str(fig5), "source": "perception_action_direction_success_participant_mean_by_direction.csv"},
        ]
    )
    _pa_save_csv(manifest, "perception_action_figure_manifest", root)

    run_manifest = pd.DataFrame(
        [
            {
                "analysis": "perception_action_direction_success",
                "selection": str(selection),
                "exclude_filter_folders": bool(exclude_filter_folders),
                "data_root": str(Path(data_root)),
                "output_root": str(root),
                "n_discovered_trial_rows": int(len(trials)),
                "n_direction_pairs": int(len(pair_table)),
                "n_subjects": int(pair_table["subject_id"].nunique())
                if "subject_id" in pair_table
                else 0,
                "n_pairs_with_direction": int(
                    pair_table["direction_pair_class"].isin(
                        ["same_direction", "different_direction", "ambiguous"]
                    ).sum()
                )
                if "direction_pair_class" in pair_table
                else 0,
            }
        ]
    )
    _pa_save_csv(run_manifest, "perception_action_run_manifest", root)

    return {
        "output_root": root,
        "trials": trials,
        "pair_directions": pair_table,
        "pse_jnd_by_direction": by_dir,
        "pse_jnd_by_subject_direction": subj_fit,
        "success_by_direction": success_summary,
        **match_tables,
        **direction_success_tables,
        **model_tables,
        "subject_bias_vs_difference": merged,
        "run_manifest": run_manifest,
    }


def _safe_batch_label(value: Any) -> str:
    if hasattr(ka, "sanitize_name"):
        return ka.sanitize_name(value, fallback="scope")
    return str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


def _subject_batch_rows(
    data_root: Path,
    *,
    groups: tuple[str, ...] = ("L_E", "N_E"),
) -> list[dict[str, Any]]:
    """Return per-subject batch rows for valid, included subject folders.

    Folder-level ignore rules are delegated to kinematics so directories marked
    ``not finish``, ``not include``, ``old``, etc. never enter the batch.
    Filter folders are represented as ``<SUBJECT>_FILTER_ONLY`` selections;
    non-filter folders use ``exclude_filter_folders=True`` to avoid accidentally
    mixing in filtered repeats of the same participant.
    """
    rows: list[dict[str, Any]] = []
    root = Path(data_root)
    if not root.exists():
        return rows
    seen: set[tuple[str, bool]] = set()
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if ka.should_ignore_analysis_path(subject_dir):
            continue
        canonical = ka.canonical_subject_id(subject_dir.name)
        group = ka.subject_group_code(canonical)
        if group not in groups:
            continue
        if not any(
            p.name.lower() == "answers.csv" and not ka.should_ignore_analysis_path(p)
            for p in subject_dir.rglob("answers.csv")
        ):
            continue
        is_filter = ka.should_filter_analysis_path(subject_dir.name)
        key = (canonical, is_filter)
        if key in seen:
            continue
        seen.add(key)
        if is_filter:
            selection = f"{canonical}_FILTER_ONLY"
            label = f"subject_{canonical}_filter_only"
            exclude_filter_folders = False
            mode = "filter_only"
        else:
            selection = canonical
            label = f"subject_{canonical}_non_filter"
            exclude_filter_folders = True
            mode = "non_filter"
        rows.append(
            {
                "batch_scope": "subject",
                "batch_mode": mode,
                "batch_label": _safe_batch_label(label),
                "selection": selection,
                "exclude_filter_folders": exclude_filter_folders,
                "source_subject_folder": subject_dir.name,
                "canonical_subject_id": canonical,
                "experiment_group": group,
            }
        )
    return rows


def build_direction_success_batch_plan(
    data_root: Path,
    *,
    group_selections: tuple[str, ...] = ("L_N_E", "L_E", "N_E"),
    include_subjects: bool = True,
) -> pd.DataFrame:
    """Plan batch runs for thesis direction-success analyses.

    For each requested group this creates:
    - ``non_filter``: the regular cohort with filter folders excluded.
    - ``filter_only``: only folders marked as filter.

    Subject rows are added separately for all valid L_E/N_E subject folders
    found under ``data_root``. Ignored/not-finished folders are never planned.
    """
    rows: list[dict[str, Any]] = []
    for selection in group_selections:
        rows.append(
            {
                "batch_scope": "group",
                "batch_mode": "non_filter",
                "batch_label": _safe_batch_label(f"{selection}_non_filter"),
                "selection": selection,
                "exclude_filter_folders": True,
                "source_subject_folder": "",
                "canonical_subject_id": "",
                "experiment_group": selection,
            }
        )
        rows.append(
            {
                "batch_scope": "group",
                "batch_mode": "filter_only",
                "batch_label": _safe_batch_label(f"{selection}_filter_only"),
                "selection": f"{selection}_FILTER_ONLY",
                "exclude_filter_folders": False,
                "source_subject_folder": "",
                "canonical_subject_id": "",
                "experiment_group": selection,
            }
        )
    if include_subjects:
        rows.extend(_subject_batch_rows(Path(data_root)))
    return pd.DataFrame(rows)


def run_direction_success_batch(
    data_root: Path,
    *,
    batch_output_root: Optional[Path] = None,
    group_selections: tuple[str, ...] = ("L_N_E", "L_E", "N_E"),
    include_subjects: bool = True,
    pilot_onboarding_trials: int = ka.PILOT_ONBOARDING_TRIALS,
) -> dict[str, Any]:
    """Run direction-success analysis for groups and each subject separately."""
    root = (
        Path(batch_output_root)
        if batch_output_root is not None
        else _results_root() / "batch_direction_success"
    )
    root.mkdir(parents=True, exist_ok=True)
    plan = build_direction_success_batch_plan(
        Path(data_root),
        group_selections=group_selections,
        include_subjects=include_subjects,
    )
    manifest_rows: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        output_root = root / str(row["batch_label"])
        rec = row.to_dict()
        rec["output_root"] = str(output_root)
        try:
            result = run_full_analysis(
                Path(data_root),
                output_root=output_root,
                selection=row["selection"],
                exclude_filter_folders=bool(row["exclude_filter_folders"]),
                pilot_onboarding_trials=pilot_onboarding_trials,
            )
            run_manifest = result.get("run_manifest", pd.DataFrame())
            if isinstance(run_manifest, pd.DataFrame) and not run_manifest.empty:
                rec.update(run_manifest.iloc[0].to_dict())
            rec["status"] = "ok"
            rec["warning"] = ""
        except Exception as exc:  # noqa: BLE001 - batch must continue other scopes
            rec["status"] = "failed"
            rec["warning"] = str(exc)
        manifest_rows.append(rec)
    manifest = pd.DataFrame(manifest_rows)
    _pa_save_csv(plan, "batch_plan", root)
    _pa_save_csv(manifest, "batch_manifest", root)
    return {
        "batch_output_root": root,
        "batch_plan": plan,
        "batch_manifest": manifest,
    }


if __name__ == "__main__":
    raise SystemExit(main())
