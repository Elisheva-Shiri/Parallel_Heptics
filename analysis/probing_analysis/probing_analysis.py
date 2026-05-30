"""Center-to-side probing analysis for tracking-derived kinematics.

This analysis counts how many times each participant moves from the center area
to a side area during each stiffness segment. It writes a dedicated set of CSVs
and figures under ``analysis/probing_analysis/results`` by default.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    )
    from analysis.scope_plots import save_scope_summary_plots
except ModuleNotFoundError:  # pragma: no cover - supports running from analysis subfolders
    import sys

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
    )
    from analysis.scope_plots import save_scope_summary_plots

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
    "radial_velocity_px_s",
    "interacting_bool",
]


def save_csv(df: pd.DataFrame, output_root: Path, name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / name
    df.to_csv(path, index=False)
    return path


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


def load_kinematic_inputs(kinematics_results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the sample-level and segment-level kinematic tables."""
    samples_path = kinematics_results / "kinematic_samples.csv"
    trials_path = kinematics_results / "trial_kinematic_summary.csv"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing sample table: {samples_path}")
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing segment summary table: {trials_path}")

    header = pd.read_csv(samples_path, nrows=0)
    usecols = [c for c in SAMPLE_COLUMNS if c in header.columns]
    samples = pd.read_csv(samples_path, usecols=usecols)
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
    active: dict[str, Any] | None = None

    for _, row in d.iterrows():
        r = float(row["r_center_px"])
        t = float(row["time_s"])
        st = float(row["stiffness_time_s"]) if pd.notna(row.get("stiffness_time_s")) else t
        angle = float(row["position_angle_deg"]) if pd.notna(row.get("position_angle_deg")) else np.nan
        speed = float(row["speed_px_s"]) if pd.notna(row.get("speed_px_s")) else np.nan

        in_center = r <= center_radius_px
        if in_center and not was_center:
            center_visits += 1
            ready_from_center = True
        was_center = in_center

        if active is None:
            if ready_from_center and r >= side_radius_px:
                active = {
                    "probe_start_time_s": t,
                    "probe_start_stiffness_time_s": st,
                    "side_cross_time_s": t,
                    "side_cross_stiffness_time_s": st,
                    "side_cross_radius_px": r,
                    "side_cross_angle_deg": angle,
                    "side_cross_direction": _direction_label(angle),
                    "peak_radius_px": r,
                    "peak_time_s": t,
                    "peak_stiffness_time_s": st,
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
            active["peak_angle_deg"] = angle
            active["peak_direction"] = _direction_label(angle)
        if np.isfinite(speed):
            previous_speed = active.get("max_speed_px_s", np.nan)
            active["max_speed_px_s"] = float(np.nanmax([previous_speed, speed]))

        if in_center:
            active["probe_end_time_s"] = t
            active["probe_end_stiffness_time_s"] = st
            duration = active["probe_end_time_s"] - active["probe_start_time_s"]
            if duration >= min_probe_duration_s:
                active["probe_duration_s"] = duration
                events.append(active)
            active = None
            ready_from_center = True

    if active is not None and not d.empty:
        last = d.iloc[-1]
        active["probe_end_time_s"] = float(last["time_s"])
        active["probe_end_stiffness_time_s"] = float(last["stiffness_time_s"]) if pd.notna(last.get("stiffness_time_s")) else float(last["time_s"])
        duration = active["probe_end_time_s"] - active["probe_start_time_s"]
        if duration >= min_probe_duration_s:
            active["probe_duration_s"] = duration
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

    return {
        "probing_subject_finger_stiffness_summary": _add_log_backtransform_columns(subject_finger_stiffness),
        "probing_stiffness_summary": _add_log_backtransform_columns(stiffness_summary),
        "probing_finger_stiffness_summary": _add_log_backtransform_columns(finger_stiffness_summary),
        "probing_comparison_summary": _add_log_backtransform_columns(comparison_summary),
        "probing_success_summary": _add_log_backtransform_columns(success_summary),
        "probing_success_by_stiffness": _add_log_backtransform_columns(success_by_stiffness),
        "probing_subject_finger_correlations": correlations,
        "probing_direction_summary": direction_summary,
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
    exact_tables = {
        f"probing_{name}": table
        for name, table in compute_group_comparison_tables(
            probing_trial_summary,
            metric_columns=PROBING_GROUP_METRICS,
            condition_cols=["finger_condition", "stiffness_value"],
        ).items()
    }
    scope_tables = {
        f"probing_{name}": table
        for name, table in compute_analysis_scope_tables(
            probing_trial_summary,
            metric_columns=PROBING_GROUP_METRICS,
            condition_cols=["finger_condition", "stiffness_value"],
        ).items()
    }
    setup_tables = {
        f"probing_{name}": table
        for name, table in compute_setup_factor_tables(
            probing_trial_summary,
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
    return project_root / "analysis" / "Kinematics" / "results", here.parent / "results"


def main(argv: list[str] | None = None) -> int:
    default_kinematics, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Run center-to-side probing analysis.")
    parser.add_argument("--kinematics-results", type=Path, default=default_kinematics)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--center-radius-px", type=float, default=DEFAULT_CENTER_RADIUS_PX)
    parser.add_argument("--side-radius-px", type=float, default=DEFAULT_SIDE_RADIUS_PX)
    parser.add_argument("--min-probe-duration-s", type=float, default=DEFAULT_MIN_PROBE_DURATION_S)
    parser.add_argument("--fig-dpi", type=int, default=160)
    args = parser.parse_args(argv)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
