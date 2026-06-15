"""Plotting / figure-generation functions for the 2AFC psychophysics pipeline.

This module was split out of ``twoafc_psychophysics`` to keep that module focused
on data loading, fitting, and statistics. The functions here are re-exported from
``twoafc_psychophysics`` (via an explicit ``from twoafc_figures import (...)`` at the
bottom of that module) so existing callers using ``pf.<plot_fn>`` continue to work
unchanged.
"""
from __future__ import annotations

import hashlib
import math
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# NOTE: the names this module needs from ``twoafc_psychophysics`` are imported at
# the BOTTOM of this file, not here. twoafc_psychophysics re-exports the plotting
# API from this module (a deliberate circular relationship); importing core at the
# bottom — after every function below is defined — means the two modules can be
# imported in either order without a partially-initialized-module ImportError.
# The constants/helpers are only referenced inside function bodies (runtime), so
# binding them at module bottom is sufficient.


def _stable_hash(*parts: Any) -> int:
    """Deterministic, process-stable integer hash for reproducible dot jitter.

    Python's built-in ``hash()`` of strings/tuples is randomized per process
    (``PYTHONHASHSEED``), which made the scatter-jitter offsets below change on
    every run. Hashing a fixed digest instead keeps the jittered dot positions
    identical across runs while preserving the same spread (the ``% 100`` /
    magnitude transforms at the call sites are unchanged).
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha1(key).hexdigest()[:8], 16)


# Seaborn's stripplot jitter and regplot bootstrap CI draw from NumPy's global
# RNG; without a fixed seed those figures change every run. Seeding immediately
# before each such call makes the rendered figure reproducible across runs.
_FIGURE_RNG_SEED = 0


def _finalize_fig(fig: Any, out_path: Path, fig_dpi: int) -> None:
    """Ensure the parent dir exists, write the figure, and close it."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=fig_dpi)
    plt.close(fig)

def set_psychometric_delta_axis(ax: Any, delta_values: Any = None) -> None:
    """Use a centred, symmetric comparison-standard delta axis (0 at the middle).

    The x-axis is G_comparison - G_standard, centred on 0 with equal negative and
    positive extent (``+/- PSYCHOMETRIC_DELTA_AXIS_LIMIT``, i.e. +/-80), widened
    only if the observed deltas genuinely extend past it so no level is clipped.
    """
    ax.set_xlabel(PSYCHOMETRIC_DELTA_AXIS_LABEL)
    limit = PSYCHOMETRIC_DELTA_AXIS_LIMIT
    if delta_values is not None:
        values = pd.to_numeric(pd.Series(delta_values), errors="coerce").dropna()
        if not values.empty:
            data_extent = float(np.max(np.abs(values.to_numpy(dtype=float))))
            if np.isfinite(data_extent):
                limit = max(limit, float(np.ceil(data_extent / 5.0) * 5.0))
    ax.set_xlim(-limit, limit)
    if delta_values is None:
        return
    observed_ticks = np.sort(values.unique().astype(float)) if not values.empty else np.array([0.0])
    if 0.0 not in observed_ticks:
        observed_ticks = np.sort(np.append(observed_ticks, 0.0))
    observed_ticks = observed_ticks[(observed_ticks >= -limit) & (observed_ticks <= limit)]
    if len(observed_ticks) > 25:
        step = int(math.ceil(len(observed_ticks) / 25))
        observed_ticks = observed_ticks[::step]
        if 0.0 not in observed_ticks:
            observed_ticks = np.sort(np.append(observed_ticks, 0.0))
    ax.set_xticks(observed_ticks)

def _finger_sort_key(value: Any) -> int:
    order = {finger: i for i, finger in enumerate(FINGER_ORDER)}
    return order.get(str(value), 99)

def _finger_order_present(values: Any) -> list[Any]:
    present = {str(v) for v in pd.Series(values).dropna().unique()}
    ordered = [finger for finger in FINGER_ORDER if finger in present]
    ordered.extend(sorted([v for v in present if v not in set(ordered)]))
    return ordered

def _finger_color(value: Any, default: str = "0.35") -> str:
    return FINGER_STYLE.get(str(value), {}).get("color", default)

def _finger_label(value: Any) -> str:
    return FINGER_STYLE.get(str(value), {}).get("label", str(value))

def _workspace_letter(value: Any) -> str | None:
    """Return 'L' or 'N' for a workspace_setup value, else None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if text.startswith("L"):
        return "L"
    if text.startswith("N"):
        return "N"
    return None

def _shade_for_workspace(base_color: Any, workspace: Any, *, dark: float = 0.45, light: float = 0.55) -> Any:
    """Shade a base colour by workspace: L -> darker, N -> lighter, else unchanged.

    Used to distinguish the L and N workspace subjects in the per-finger group
    psychometric curves (L drawn in a darker shade, N in a lighter shade).
    """
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(base_color), dtype=float)
    ws = _workspace_letter(workspace)
    if ws == "L":
        rgb = rgb * (1.0 - dark)
    elif ws == "N":
        rgb = rgb + (1.0 - rgb) * light
    return tuple(float(c) for c in np.clip(rgb, 0.0, 1.0))

def _appearance_order(values: Any) -> list[int]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(int)
    present = set(numeric.unique())
    ordered = [i for i in [1, 2, 3, 4] if i in present]
    ordered.extend(sorted([i for i in present if i not in set(ordered)]))
    return ordered

def _appearance_tick_labels(order: list[Any]) -> list[str]:
    labels = []
    for value in order:
        try:
            labels.append(FINGER_APPEARANCE_LABELS.get(int(value), str(value)))
        except Exception:
            labels.append(str(value))
    return labels

def _subject_palette(subjects: Any) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    ordered = sorted([str(s) for s in pd.Series(subjects).dropna().unique()], key=_subject_sort_key)
    cmap = plt.get_cmap("tab20", max(1, len(ordered)))
    return {subject: cmap(i % cmap.N) for i, subject in enumerate(ordered)}

def _style_color(value: Any, style_col: Optional[str], palette: Optional[dict[str, Any]] = None) -> Any:
    if style_col == "subject_id" and palette is not None:
        return palette.get(str(value), "0.35")
    return _finger_color(value)

def _select_typical_subject_for_psychometric_overlay(
    pse_jnd_by_subject_finger: pd.DataFrame,
    preferred_subject: Optional[str] = None,
) -> Optional[str]:
    if pse_jnd_by_subject_finger.empty or "subject_id" not in pse_jnd_by_subject_finger:
        return None
    if preferred_subject and preferred_subject in set(pse_jnd_by_subject_finger["subject_id"].astype(str)):
        return preferred_subject
    fits = add_fit_delta_columns(pse_jnd_by_subject_finger)
    summary = (
        fits.groupby("subject_id", dropna=False)
        .agg(
            n_fits=("pse", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
            median_abs_pse_delta=("pse_delta_comparison_minus_standard", lambda x: float(pd.to_numeric(x, errors="coerce").abs().median())),
            median_jnd=("jnd", lambda x: float(pd.to_numeric(x, errors="coerce").median())),
        )
        .reset_index()
    )
    summary = summary[summary["n_fits"] >= 3].copy()
    if summary.empty:
        return str(pse_jnd_by_subject_finger["subject_id"].iloc[0])
    group_median = summary["median_abs_pse_delta"].median()
    summary["distance_from_group_median"] = (summary["median_abs_pse_delta"] - group_median).abs()
    summary = summary.sort_values(["distance_from_group_median", "median_jnd", "subject_id"])
    return str(summary.iloc[0]["subject_id"])

def _metric_error_bars(plot_df: pd.DataFrame, metric_col: str) -> Optional[tuple[pd.Series, pd.Series]]:
    """Return ``(lower_err, upper_err)`` for ``metric_col`` if CI columns exist."""
    lower_col, upper_col = _METRIC_CI_COLUMN_HINTS.get(metric_col, (None, None))
    if not lower_col or lower_col not in plot_df.columns or upper_col not in plot_df.columns:
        return None
    metric = pd.to_numeric(plot_df[metric_col], errors="coerce")
    lower = pd.to_numeric(plot_df[lower_col], errors="coerce")
    upper = pd.to_numeric(plot_df[upper_col], errors="coerce")
    if lower.isna().all() or upper.isna().all():
        return None
    lower_err = (metric - lower).clip(lower=0)
    upper_err = (upper - metric).clip(lower=0)
    return lower_err, upper_err

def _plot_article_style_metric_lines(
    df: pd.DataFrame,
    x_col: str,
    metric_col: str,
    title: str,
    ylabel: str,
    xlabel: str,
    out_path: Path,
    x_order: Optional[list[Any]] = None,
    style_col: Optional[str] = "finger_condition",
    fig_dpi: int = 160,
) -> Optional[Path]:
    import matplotlib.pyplot as plt

    if df.empty or metric_col not in df or x_col not in df:
        return None
    plot_df = df.copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col, x_col])
    if plot_df.empty:
        return None
    if x_order is None:
        if x_col == "finger_condition":
            x_order = _finger_order_present(plot_df[x_col])
        elif x_col == "finger_appearance_order":
            x_order = _appearance_order(plot_df[x_col])
        else:
            x_order = sorted(plot_df[x_col].dropna().unique())
    x_map = {value: i for i, value in enumerate(x_order)}
    plot_df["_x_num"] = plot_df[x_col].map(x_map)
    plot_df = plot_df.dropna(subset=["_x_num"])
    if plot_df.empty:
        return None

    error_bars = _metric_error_bars(plot_df, metric_col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    style_palette = _subject_palette(plot_df["subject_id"]) if style_col == "subject_id" and "subject_id" in plot_df else None
    for subject, g in plot_df.groupby("subject_id", dropna=False):
        g = g.sort_values("_x_num")
        line_color = style_palette.get(str(subject), "0.70") if style_palette is not None else "0.70"
        ax.plot(g["_x_num"], g[metric_col], color=line_color, linewidth=1.1, alpha=0.70, zorder=1)
        if error_bars is not None:
            lower_err, upper_err = error_bars
            err = np.vstack([lower_err.loc[g.index].fillna(0).to_numpy(), upper_err.loc[g.index].fillna(0).to_numpy()])
            if np.any(err > 0):
                ax.errorbar(
                    g["_x_num"],
                    g[metric_col],
                    yerr=err,
                    fmt="none",
                    ecolor=line_color,
                    alpha=0.45,
                    elinewidth=0.8,
                    capsize=2,
                    zorder=1,
                )
        if style_col and style_col in g:
            for _, row in g.iterrows():
                style = FINGER_STYLE.get(str(row.get(style_col)), {})
                color = _style_color(row.get(style_col), style_col, style_palette)
                ax.scatter(
                    row["_x_num"],
                    row[metric_col],
                    color=color,
                    marker=style.get("marker", "o") if style_col != "subject_id" else "o",
                    edgecolor="black",
                    linewidth=0.3,
                    s=45,
                    alpha=0.9,
                    zorder=2,
                )
        else:
            ax.scatter(g["_x_num"], g[metric_col], color="0.35", s=35, alpha=0.8, zorder=2)
    group = (
        plot_df.groupby("_x_num", dropna=False)[metric_col]
        .agg(["mean", "std", "count", "sem"])
        .reset_index()
    )
    group["sem"] = group["sem"].fillna(group["std"])
    group["ci95"] = np.where(group["count"] > 1, 1.96 * group["sem"], 0.0)
    ax.errorbar(
        group["_x_num"],
        group["mean"],
        yerr=group["ci95"],
        color="black",
        linestyle=":",
        marker="o",
        linewidth=2.5,
        capsize=4,
        label="Group mean (95% CI across subjects)",
        zorder=3,
    )
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1) if "PSE" in ylabel or "slope" in ylabel.lower() else None
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(_appearance_tick_labels(x_order) if x_col == "finger_appearance_order" else [str(x) for x in x_order])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if error_bars is not None and ("PSE" in ylabel or "JND" in ylabel):
        ax.text(
            0.02,
            0.02,
            "Subject error bars: per-fit 95% CI (parametric bootstrap)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.35",
        )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def save_xy_probing_skin_stretch_figures(
    output_root: Path,
    xy_group_trajectory_bins: pd.DataFrame,
    xy_trial_summary: pd.DataFrame,
    xy_group_summary: pd.DataFrame,
    fig_dpi: int = 160,
    xy_trajectory_bins: Optional[pd.DataFrame] = None,
) -> list[Path]:
    """Save figures replacing article grip-force panels with XY/stretch panels."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    fig_root = output_root / "figures" / "xy_probing_skin_stretch"
    fig_root.mkdir(parents=True, exist_ok=True)

    if not xy_group_trajectory_bins.empty:
        # (xy_center_trajectory_by_finger figure removed — uninformative.)
        # The over-time trajectory figures now show what's behind the per-finger
        # mean: faded individual-subject trajectories plus a 25-75th percentile
        # band (replacing the old, too-summarised SEM band).
        traj = xy_trajectory_bins if xy_trajectory_bins is not None else pd.DataFrame()
        for y_col, y_label, filename, title in [
            ("mean_center_distance_px", "Distance from center (px)", "center_distance_trajectory_by_finger.png", "Probing distance from center over time"),
            (
                "mean_skin_stretch_command_proxy_px",
                "Skin-stretch command proxy (px x gain/175)",
                "skin_stretch_proxy_trajectory_by_finger.png",
                "Skin-stretch command proxy over probing time",
            ),
        ]:
            if y_col not in xy_group_trajectory_bins or xy_group_trajectory_bins[y_col].dropna().empty:
                continue
            raw_col = y_col.replace("mean_", "")
            # Per-subject mean trajectory (one curve per subject per finger).
            subj = pd.DataFrame()
            if not traj.empty and raw_col in traj.columns and {"subject_id", "finger_condition", "trajectory_time_bin"}.issubset(traj.columns):
                subj = (
                    traj.groupby(["finger_condition", "subject_id", "trajectory_time_bin"], dropna=False)
                    .agg(time_fraction=("time_fraction", "mean"), value=(raw_col, "mean"))
                    .reset_index()
                )
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for finger in sorted(xy_group_trajectory_bins["finger_condition"].dropna().unique(), key=_finger_sort_key):
                style = FINGER_STYLE.get(str(finger), {})
                color = style.get("color", None)
                g = xy_group_trajectory_bins[xy_group_trajectory_bins["finger_condition"] == finger].sort_values("time_fraction")
                if not subj.empty:
                    sg = subj[subj["finger_condition"] == finger]
                    for _sid, ss in sg.groupby("subject_id", dropna=False):
                        ss = ss.sort_values("time_fraction")
                        ax.plot(ss["time_fraction"], ss["value"], color=color, alpha=0.12, linewidth=0.7, zorder=1)
                    band = (
                        sg.groupby("trajectory_time_bin", dropna=False)
                        .agg(
                            time_fraction=("time_fraction", "mean"),
                            p25=("value", lambda s: s.quantile(0.25)),
                            p75=("value", lambda s: s.quantile(0.75)),
                        )
                        .reset_index()
                        .sort_values("time_fraction")
                    )
                    ax.fill_between(band["time_fraction"], band["p25"], band["p75"], color=color, alpha=0.18, zorder=2)
                ax.plot(g["time_fraction"], g[y_col], color=color, linewidth=2.2, label=style.get("label", str(finger)), zorder=3)
            ax.set_xlabel("Normalized time within tracked trial")
            ax.set_ylabel(y_label)
            ax.set_title(f"{title} (faded: per-subject; band: 25-75th pct)")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            out = fig_root / filename
            _finalize_fig(fig, out, fig_dpi)
            paths.append(out)

    if not xy_trial_summary.empty and "max_skin_stretch_command_proxy_px" in xy_trial_summary:
        # correct_response is binary (0/1), so a raw scatter just draws two rows
        # of dots at 0 and 1 and teaches nothing. Instead bin the stretch proxy
        # into quantiles and plot the mean success RATE per bin (Wilson 95% CI),
        # per finger, so any success-vs-stretch relationship is actually visible.
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        plot_df = xy_trial_summary[xy_trial_summary["tracking_exists"]].dropna(subset=["max_skin_stretch_command_proxy_px", "correct_response"]).copy()
        x_all = pd.to_numeric(plot_df["max_skin_stretch_command_proxy_px"], errors="coerce")
        n_bins = int(min(6, max(2, x_all.nunique())))
        try:
            plot_df["_stretch_bin"] = pd.qcut(x_all.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
        except Exception:
            plot_df["_stretch_bin"] = 0
        any_curve = False
        for finger, g in plot_df.groupby("finger_condition", dropna=False):
            style = FINGER_STYLE.get(str(finger), {})
            agg = (
                g.groupby("_stretch_bin", dropna=True)
                .agg(
                    x=("max_skin_stretch_command_proxy_px", "mean"),
                    rate=("correct_response", "mean"),
                    n=("correct_response", "size"),
                    lo=("correct_response", _wilson_ci95_lower),
                    hi=("correct_response", _wilson_ci95_upper),
                )
                .reset_index()
                .sort_values("x")
            )
            agg = agg[agg["n"] >= 3]
            if agg.empty:
                continue
            any_curve = True
            yerr = np.vstack([agg["rate"] - agg["lo"], agg["hi"] - agg["rate"]])
            ax.errorbar(
                agg["x"], agg["rate"], yerr=yerr, marker="o", capsize=3, linewidth=2,
                color=style.get("color", None), label=style.get("label", str(finger)),
            )
        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Max skin-stretch command proxy (px × gain/175), binned")
        ax.set_ylabel("Success rate (mean ± Wilson 95% CI)")
        ax.set_title("Success rate vs skin-stretch command proxy (binned)")
        if any_curve:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = fig_root / "success_vs_skin_stretch_proxy.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    if not xy_group_summary.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for finger, g in xy_group_summary.groupby("finger_condition", dropna=False):
            style = FINGER_STYLE.get(str(finger), {})
            ax.scatter(
                g["mean_max_center_distance_px"],
                g["success_rate"],
                color=style.get("color", None),
                marker=style.get("marker", "o"),
                edgecolor="black",
                linewidth=0.3,
                s=55,
                label=style.get("label", str(finger)),
            )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Mean max probing distance from center (px)")
        ax.set_ylabel("Success rate")
        ax.set_title("Participant/finger success vs probing distance")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = fig_root / "success_vs_xy_probing_distance.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    save_csv(pd.DataFrame({"figure": [str(p) for p in paths]}), output_root, "xy_probing_skin_stretch_figure_manifest.csv")
    return paths

def _stiffness_colors(values: pd.Series, plt_module: Any) -> list[Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return ["0.35"] * len(values)
    vmin, vmax = float(numeric.min()), float(numeric.max())
    span = max(vmax - vmin, 1e-12)
    cmap = plt_module.get_cmap(STIFFNESS_CMAP)
    return [cmap((float(v) - vmin) / span) if pd.notna(v) else "0.5" for v in numeric]

def _plot_finger_columns_stiffness_dots(
    df: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    title: str,
    out_path: Path,
    fig_dpi: int,
    stiffness_col: str = "comparison_value",
    subject_col: str = "subject_id",
) -> Path | None:
    """Plot finger columns with finger-colored backgrounds and stiffness-colored dots."""
    if df is None or df.empty or y_col not in df or "finger_condition" not in df:
        return None
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_col])
    if plot_df.empty:
        return None
    order = _finger_order_present(plot_df["finger_condition"])
    x_map = {finger: idx for idx, finger in enumerate(order)}
    plot_df["_x"] = plot_df["finger_condition"].astype(str).map(x_map)
    # Deterministic within-finger offsets so repeated stiffness dots are side-by-side
    # even for one subject (where subject-id based jitter would stack them).
    if stiffness_col in plot_df:
        offset_source = plot_df[stiffness_col]
    else:
        offset_source = pd.Series(np.arange(len(plot_df)), index=plot_df.index)
    jitter = pd.Series(0.0, index=plot_df.index)
    for _, idx in plot_df.groupby("finger_condition", dropna=False).groups.items():
        idx_list = list(idx)
        levels = sorted(pd.Series(offset_source.loc[idx_list]).dropna().astype(str).unique())
        if len(levels) <= 1:
            offsets = {levels[0]: 0.0} if levels else {}
        else:
            positions = np.linspace(-0.24, 0.24, len(levels))
            offsets = dict(zip(levels, positions))
        jitter.loc[idx_list] = pd.Series(offset_source.loc[idx_list]).astype(str).map(offsets).fillna(0.0).to_numpy()
    if stiffness_col in plot_df:
        dot_colors = _stiffness_colors(plot_df[stiffness_col], plt)
    else:
        dot_colors = [_finger_color(f) for f in plot_df["finger_condition"].astype(str)]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for finger, x in x_map.items():
        ax.axvspan(x - 0.5, x + 0.5, color=_finger_color(finger), alpha=0.10, zorder=0)
    ax.scatter(
        plot_df["_x"] + jitter,
        plot_df[y_col],
        c=dot_colors,
        s=58,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.88,
        zorder=2,
    )
    if stiffness_col in plot_df and pd.to_numeric(plot_df[stiffness_col], errors="coerce").notna().sum() > 1:
        numeric = pd.to_numeric(plot_df[stiffness_col], errors="coerce")
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap(STIFFNESS_CMAP),
            norm=plt.Normalize(vmin=float(numeric.min()), vmax=float(numeric.max())),
        )
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Comparison stiffness")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_finger_label(f) for f in order])
    ax.set_xlabel("Finger condition")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y_col.startswith("p_") or "rate" in y_col or "bias" in y_col:
        ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def _plot_side_bias_object_columns(
    df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    fig_dpi: int,
    stiffness_col: str = "comparison_value",
    p_object2_col: str = "p_chose_object2",
    subject_col: str = "subject_id",
) -> Path | None:
    """Side bias per finger, split into the two presented objects.

    The slot for each finger is shaded with that finger's dedicated colour
    (Index/Middle/Ring/Pinky) as a background. Inside it sit two solid,
    full-colour bars whose heights are the mean probabilities: object 1 on the
    left (orange, ``mean(1 - p_chose_object2)``) and object 2 on the right
    (blue, ``mean(p_chose_object2)``). Every dot is one comparison-stiffness
    level; its inner fill encodes the stiffness (viridis, with a colorbar) and
    its perimeter encodes subject identity.
    """
    if df is None or df.empty or p_object2_col not in df or "finger_condition" not in df:
        return None
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df[p_object2_col] = pd.to_numeric(plot_df[p_object2_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[p_object2_col])
    if plot_df.empty:
        return None
    order = _finger_order_present(plot_df["finger_condition"])
    x_map = {finger: idx for idx, finger in enumerate(order)}
    subject_colors = _subject_palette(plot_df[subject_col]) if subject_col in plot_df else {}

    numeric_stiff = pd.to_numeric(plot_df.get(stiffness_col, pd.Series(dtype=float)), errors="coerce")
    has_stiffness = numeric_stiff.notna().sum() > 0
    if has_stiffness:
        vmin, vmax = float(numeric_stiff.min()), float(numeric_stiff.max())
        span = max(vmax - vmin, 1e-12)
        cmap = plt.get_cmap(STIFFNESS_CMAP)

    obj_dx, bar_w = 0.22, 0.40
    # (object number, x-offset within the finger slot, p-transform, bar colour)
    objects = [
        (1, -obj_dx, lambda p: 1.0 - p, OBJECT1_COLOR),
        (2, +obj_dx, lambda p: p, OBJECT2_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    # Background: each finger slot shaded in its dedicated colour.
    for finger, x in x_map.items():
        ax.axvspan(x - 0.5, x + 0.5, color=_finger_color(finger), alpha=0.16, zorder=0)

    for finger, x in x_map.items():
        fdf = plot_df[plot_df["finger_condition"].astype(str) == str(finger)]
        if fdf.empty:
            continue
        # Deterministic within-sub-column offsets so repeated stiffness dots sit
        # side-by-side instead of stacking.
        offsets: dict[float, float] = {}
        if has_stiffness:
            levels = sorted(pd.to_numeric(fdf[stiffness_col], errors="coerce").dropna().unique())
            if len(levels) > 1:
                offsets = dict(zip(levels, np.linspace(-0.14, 0.14, len(levels))))
        for _obj_num, dx, ytf, color in objects:
            # Solid, full-colour bar at the mean probability for this object.
            bar_h = float(ytf(fdf[p_object2_col]).mean())
            ax.bar(x + dx, bar_h, width=bar_w, color=color, edgecolor="black",
                   linewidth=0.5, zorder=1)
            for _, row in fdf.iterrows():
                y = ytf(row[p_object2_col])
                stiff = pd.to_numeric(row.get(stiffness_col, np.nan), errors="coerce") if has_stiffness else np.nan
                jit = offsets.get(stiff, 0.0) if (has_stiffness and pd.notna(stiff)) else 0.0
                inner = cmap((float(stiff) - vmin) / span) if (has_stiffness and pd.notna(stiff)) else color
                edge = subject_colors.get(str(row.get(subject_col)), "black")
                ax.scatter(
                    x + dx + jit,
                    y,
                    c=[inner],
                    s=58,
                    edgecolor=[edge],
                    linewidth=1.1,
                    alpha=0.9,
                    zorder=2,
                )

    if has_stiffness and numeric_stiff.notna().sum() > 1:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(STIFFNESS_CMAP), norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Comparison stiffness (dot fill)")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=OBJECT1_COLOR, markeredgecolor="none", label="Object 1 (left)"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=OBJECT2_COLOR, markeredgecolor="none", label="Object 2 (right)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=8)

    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1, zorder=1)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_finger_label(f) for f in order])
    ax.set_xlabel("Finger condition")
    ax.set_ylabel("P(chose object)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def _save_subject_psychometric_curves(
    fig_dir: Path,
    psychometric_input: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    subject: str,
    fig_dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    if psychometric_input.empty or fits.empty:
        return paths
    curve_dir = fig_dir / "psychometric_curves"
    for _, fit_row in fits.sort_values("finger_condition", key=lambda s: s.map(_finger_sort_key)).iterrows():
        finger = fit_row["finger_condition"]
        agg = psychometric_input[psychometric_input["finger_condition"].astype(str) == str(finger)]
        if agg.empty:
            continue
        out = curve_dir / f"psychometric_{sanitize_name(subject)}_{sanitize_name(finger)}.png"
        path = plot_fit_curve(
            agg,
            fit_row,
            f"{subject} – {_finger_label(finger)}",
            out,
            fig_dpi=fig_dpi,
            curve_color=_finger_color(finger),
        )
        paths.append(path)
    return paths

def _save_subject_article_summary(
    fig_dir: Path,
    psychometric_input: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    subject: str,
    fig_dpi: int,
) -> Path | None:
    if psychometric_input.empty and fits.empty:
        return None
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    if not psychometric_input.empty:
        agg = add_delta_and_less_response_columns(psychometric_input)
        agg["delta"] = agg["delta_comparison_minus_standard"]
        for finger in _finger_order_present(agg["finger_condition"]):
            g = agg[agg["finger_condition"].astype(str) == str(finger)].sort_values("delta")
            ax.scatter(
                g["delta"],
                g["p_comparison_greater"],
                s=34 + 4 * pd.to_numeric(g.get("n_trials", 1), errors="coerce").fillna(1),
                color=_finger_color(finger),
                edgecolor="black",
                linewidth=0.35,
                alpha=0.85,
                label=_finger_label(finger),
            )
            frows = fits[fits["finger_condition"].astype(str) == str(finger)] if not fits.empty else pd.DataFrame()
            if not frows.empty:
                x_grid = np.linspace(float(g["delta"].min()), float(g["delta"].max()), 240)
                std = float(pd.to_numeric(g["standard_value"], errors="coerce").median())
                y_grid = _fit_row_to_delta_predictions(frows.iloc[0], std, x_grid)
                if np.isfinite(y_grid).any():
                    ax.plot(x_grid, y_grid, color=_finger_color(finger), linewidth=2.0)
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        set_psychometric_delta_axis(ax, agg["delta"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(PSYCHOMETRIC_GREATER_Y_LABEL)
    ax.set_title("4-finger psychometric curves")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    if not fits.empty:
        plot_df = add_fit_delta_columns(fits)
        order = _finger_order_present(plot_df["finger_condition"])
        x = np.arange(len(order))
        width = 0.36
        metric = "pse_delta_comparison_minus_standard" if "pse_delta_comparison_minus_standard" in plot_df else "pse"
        for offset, col, label in [(-width / 2, metric, "PSE shift"), (width / 2, "jnd", "JND")]:
            vals = [pd.to_numeric(plot_df.loc[plot_df["finger_condition"].astype(str) == str(f), col], errors="coerce").mean() for f in order]
            ax.bar(x + offset, vals, width=width, label=label, alpha=0.82)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_finger_label(f) for f in order])
        ax.set_ylabel("Stiffness units")
        ax.set_title("Fit summary")
        ax.legend(fontsize=8)
    fig.suptitle(f"Article-style subject summary: {subject}")
    fig.tight_layout()
    out = fig_dir / "psychometric_curves" / f"article_style_subject_summary_{sanitize_name(subject)}.png"
    _finalize_fig(fig, out, fig_dpi)
    return out

def _save_appearance_plot(
    df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    y_col: str = "success_rate",
    summary: str | None = None,
    fig_dpi: int,
) -> Path | None:
    if df is None or df.empty or "finger_appearance_order" not in df:
        return None
    import matplotlib.pyplot as plt

    plot_df = df.dropna(subset=["finger_appearance_order"]).copy()
    if plot_df.empty:
        return None
    if summary and y_col not in plot_df and f"{summary}_success_rate" in plot_df:
        y_col = f"{summary}_success_rate"
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_col]).sort_values("finger_appearance_order")
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if "subject_id" in plot_df and plot_df["subject_id"].nunique() > 1:
        subject_y = "success_rate" if "success_rate" in plot_df else y_col
        for _, sg in plot_df.groupby("subject_id", dropna=False):
            ax.plot(sg["finger_appearance_order"], sg[subject_y], color="0.65", linewidth=0.9, alpha=0.35, zorder=1)
    if "finger_condition" in plot_df:
        colors = [_finger_color(f) for f in plot_df["finger_condition"].astype(str)]
    else:
        colors = _stiffness_colors(plot_df["finger_appearance_order"], plt)
    ax.plot(plot_df["finger_appearance_order"], plot_df[y_col], color="black", linewidth=2.4, zorder=2)
    ax.scatter(plot_df["finger_appearance_order"], plot_df[y_col], c=colors, s=72, edgecolor="black", linewidth=0.4, zorder=3)
    order = _appearance_order(plot_df["finger_appearance_order"])
    ax.set_xticks(order)
    tick_labels = _appearance_tick_labels(order)
    if "finger_condition" in plot_df:
        order_to_finger = (
            plot_df.dropna(subset=["finger_appearance_order"])
            .drop_duplicates("finger_appearance_order")
            .set_index("finger_appearance_order")["finger_condition"]
            .to_dict()
        )
        tick_labels = [f"{label}\n{_finger_label(order_to_finger.get(pos, ''))}" for label, pos in zip(tick_labels, order)]
        for _, row in plot_df.iterrows():
            ax.annotate(
                _finger_label(row.get("finger_condition", "")),
                (row["finger_appearance_order"], row[y_col]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Finger appearance order in session")
    ax.set_ylabel("Success rate")
    ax.set_title(title)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def _save_time_fatigue_line_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    fig_dpi: int,
    summary: str | None = None,
) -> Path | None:
    if df is None or df.empty or x_col not in df:
        return None
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    if summary and y_col not in plot_df and f"{summary}_{y_col}" in plot_df:
        y_col = f"{summary}_{y_col}"
    if y_col not in plot_df:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for finger, g in plot_df.groupby("finger_condition", dropna=False):
        g = g.sort_values(x_col)
        ax.plot(g[x_col], g[y_col], marker="o", linewidth=2.2, color=_finger_color(finger), label=_finger_label(finger))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel("Success rate")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def _save_article_metric_fallback_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    metric_col: str,
    title: str,
    ylabel: str,
    xlabel: str,
    out_path: Path,
    fig_dpi: int,
) -> Path | None:
    """Write an article-style metric plot even when the metric is all-NaN."""
    if df is None or df.empty or x_col not in df:
        return None
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    if metric_col not in plot_df:
        plot_df[metric_col] = np.nan
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col])
    if plot_df.empty:
        return None
    if x_col == "finger_condition":
        x_order = _finger_order_present(plot_df[x_col])
        tick_labels = [_finger_label(x) for x in x_order]
    elif x_col == "finger_appearance_order":
        x_order = _appearance_order(plot_df[x_col])
        tick_labels = _appearance_tick_labels(x_order)
    else:
        x_order = sorted(plot_df[x_col].dropna().unique())
        tick_labels = [str(x) for x in x_order]
    x_map = {value: i for i, value in enumerate(x_order)}
    plot_df["_x_num"] = plot_df[x_col].map(x_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    finite = plot_df[metric_col].notna()
    if finite.any():
        for subject, g in plot_df[finite].groupby("subject_id", dropna=False):
            g = g.sort_values("_x_num")
            ax.plot(g["_x_num"], g[metric_col], color="0.65", linewidth=1.0, alpha=0.55, zorder=1)
            ax.scatter(g["_x_num"], g[metric_col], color="0.25", s=38, alpha=0.85, zorder=2)
        group = plot_df[finite].groupby("_x_num", dropna=False)[metric_col].mean().reset_index()
        ax.plot(group["_x_num"], group[metric_col], color="black", marker="o", linewidth=2.4, zorder=3)
    else:
        ax.text(
            0.5,
            0.5,
            f"No finite {metric_col.replace('_', ' ')} values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def _save_order_effects_plot(df: pd.DataFrame, *, out_path: Path, title: str, fig_dpi: int) -> Path | None:
    if df is None or df.empty or "finger_condition" not in df:
        return None
    return _save_time_fatigue_line_plot(
        df,
        x_col="mean_global_trial_order",
        y_col="p_comparison_greater",
        out_path=out_path,
        title=title,
        fig_dpi=fig_dpi,
    )

def _group_curve_summary(
    psychometric_input_by_subject_finger: pd.DataFrame,
    *,
    statistic: str,
) -> pd.DataFrame:
    if psychometric_input_by_subject_finger is None or psychometric_input_by_subject_finger.empty:
        return pd.DataFrame()
    points = add_delta_and_less_response_columns(psychometric_input_by_subject_finger)
    points["delta"] = points["delta_comparison_minus_standard"]
    agg_name = "median" if statistic == "median" else "mean"
    return (
        points.groupby(["finger_condition", "delta"], dropna=False)
        .agg(
            p_comparison_greater=("p_comparison_greater", agg_name),
            n_subjects=("subject_id", "nunique"),
            total_trials=("n_trials", "sum"),
            standard_value=("standard_value", "median"),
        )
        .reset_index()
        .sort_values(["finger_condition", "delta"])
    )

def _save_group_curves_plot(
    psychometric_input_by_subject_finger: pd.DataFrame,
    *,
    out_path: Path,
    statistic: str,
    title: str,
    fig_dpi: int,
) -> tuple[Path | None, pd.DataFrame]:
    summary = _group_curve_summary(psychometric_input_by_subject_finger, statistic=statistic)
    if summary.empty:
        return None, summary
    import matplotlib.pyplot as plt

    subject_points = add_delta_and_less_response_columns(psychometric_input_by_subject_finger)
    subject_points["delta"] = subject_points["delta_comparison_minus_standard"]
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for (subject, finger), g in subject_points.groupby(["subject_id", "finger_condition"], dropna=False):
        g = g.sort_values("delta")
        ax.plot(g["delta"], g["p_comparison_greater"], color=_finger_color(finger), alpha=0.14, linewidth=0.9, zorder=1)
    for finger in _finger_order_present(summary["finger_condition"]):
        g = summary[summary["finger_condition"].astype(str) == str(finger)].sort_values("delta")
        ax.plot(
            g["delta"],
            g["p_comparison_greater"],
            marker="o",
            color=_finger_color(finger),
            linewidth=3.0,
            label=f"{_finger_label(finger)} {statistic}",
            zorder=3,
        )
    ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    set_psychometric_delta_axis(ax, summary["delta"])
    ax.set_ylabel(PSYCHOMETRIC_MEAN_GREATER_Y_LABEL if statistic == "mean" else "Median P(choose comparison > standard)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path, summary

def _remove_all_only_legacy_plot(path: Path, manifest_rows: list[dict[str, Any]]) -> None:
    """Remove an all-scope legacy figure/folder and its manifest rows."""
    target = Path(path)
    if target.is_dir():
        _rmtree_windows_retry(target)
        prefix = str(target)
        manifest_rows[:] = [row for row in manifest_rows if not str(row.get("path", "")).startswith(prefix)]
    elif target.exists():
        target.unlink()
        manifest_rows[:] = [row for row in manifest_rows if str(row.get("path", "")) != str(target)]

def _append_all_figure_manifest(manifest_rows: list[dict[str, Any]], path: Path | None) -> None:
    if path:
        manifest_rows[:] = [row for row in manifest_rows if str(row.get("path", "")) != str(path)]
        manifest_rows.append({"scope": "all", "statistic": "all", "kind": "figure", "path": str(path)})

def _axis_order_and_labels(df: pd.DataFrame, x_col: str) -> tuple[list[Any], list[str]]:
    if x_col == "finger_condition":
        order = _finger_order_present(df[x_col])
        labels = [_finger_label(x) for x in order]
    elif x_col == "finger_appearance_order":
        order = _appearance_order(df[x_col])
        labels = _appearance_tick_labels(order)
    else:
        order = sorted(pd.Series(df[x_col]).dropna().unique())
        labels = [str(x) for x in order]
    return order, labels

def _save_all_subject_background_metric_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    line_color: str = "subject",
    dot_color: str = "subject",
    group_y_col: str | None = None,
    fig_dpi: int,
) -> Path | None:
    """All-scope metric plot with subject background and explicit color ownership."""
    if df is None or df.empty or x_col not in df or y_col not in df:
        return None
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        return None
    order, labels = _axis_order_and_labels(plot_df, x_col)
    x_map = {value: i for i, value in enumerate(order)}
    plot_df["_x_num"] = plot_df[x_col].map(x_map)
    plot_df = plot_df.dropna(subset=["_x_num"])
    subjects = sorted(plot_df["subject_id"].dropna().astype(str).unique(), key=_subject_sort_key) if "subject_id" in plot_df else []
    subject_colors = _subject_palette(subjects)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    if subjects:
        for subject, g in plot_df.groupby("subject_id", dropna=False):
            subject = str(subject)
            g = g.sort_values("_x_num")
            lc = subject_colors.get(subject, "0.55") if line_color == "subject" else "0.65"
            ax.plot(g["_x_num"], g[y_col], color=lc, linewidth=1.0, alpha=0.45, zorder=1)
            for _, row in g.iterrows():
                if dot_color == "finger":
                    fc = _finger_color(row.get("finger_condition"), "0.35")
                elif dot_color == "subject":
                    fc = subject_colors.get(subject, "0.35")
                else:
                    fc = "0.35"
                ax.scatter(row["_x_num"], row[y_col], s=38, facecolor=fc, edgecolor="black", linewidth=0.35, alpha=0.82, zorder=2)

    summary_col = group_y_col if group_y_col and group_y_col in plot_df else y_col
    group = plot_df.groupby("_x_num", dropna=False)[summary_col].mean().reset_index()
    ax.plot(group["_x_num"], group[summary_col], color="black", marker="o", linewidth=2.7, label="Group mean", zorder=4)
    if y_col == "success_rate" or y_col.endswith("_rate"):
        ax.set_ylim(-0.05, 1.05)
    if y_col.endswith("_slope"):
        ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def _save_all_fit_metric_by_subject_finger(
    fits: pd.DataFrame,
    *,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    fig_dpi: int,
) -> Path | None:
    """All-scope PSE/JND plot: finger-coloured columns, participant-coloured dots."""
    if fits is None or fits.empty or "finger_condition" not in fits or metric_col not in fits:
        return None
    import matplotlib.pyplot as plt

    plot_df = fits.copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["finger_condition", metric_col])
    if plot_df.empty:
        return None
    order = _finger_order_present(plot_df["finger_condition"])
    x_map = {finger: i for i, finger in enumerate(order)}
    subjects = sorted(plot_df["subject_id"].dropna().astype(str).unique(), key=_subject_sort_key) if "subject_id" in plot_df else []
    subject_colors = _subject_palette(subjects)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    for idx, finger in enumerate(order):
        ax.axvspan(idx - 0.45, idx + 0.45, color=_finger_color(finger), alpha=0.12, zorder=0)
    if subjects:
        for subject, g in plot_df.groupby("subject_id", dropna=False):
            subject = str(subject)
            g = g.sort_values("finger_condition", key=lambda s: s.astype(str).map(x_map))
            xs = np.array([x_map.get(str(f), np.nan) for f in g["finger_condition"]], dtype=float)
            valid = ~np.isnan(xs)
            if valid.sum() > 1:
                ax.plot(xs[valid], g.loc[valid, metric_col], color=subject_colors.get(subject, "0.5"), alpha=0.22, linewidth=0.8, zorder=1)
            for _, row in g.iterrows():
                finger = str(row["finger_condition"])
                x = x_map.get(finger)
                if x is None:
                    continue
                jitter = (((_stable_hash(subject, finger) % 100) / 100.0) - 0.5) * 0.22
                ax.scatter(x + jitter, row[metric_col], s=44, facecolor=subject_colors.get(subject, "0.35"), edgecolor="black", linewidth=0.35, alpha=0.9, zorder=3)
    means = plot_df.groupby("finger_condition", dropna=False)[metric_col].mean().reindex(order)
    ax.plot(range(len(order)), means.values, color="black", marker="D", linewidth=2.6, label="Group mean", zorder=4)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_finger_label(f) for f in order])
    ax.set_xlabel("Finger")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def _save_all_finger_time_plot(
    subject_bins: pd.DataFrame,
    group_bins: pd.DataFrame,
    *,
    out_path: Path,
    fig_dpi: int,
) -> Path | None:
    if subject_bins is None or subject_bins.empty or "within_finger_time_bin" not in subject_bins:
        return None
    import matplotlib.pyplot as plt

    subject_df = subject_bins.copy()
    subject_df["success_rate"] = pd.to_numeric(subject_df["success_rate"], errors="coerce")
    subject_df = subject_df.dropna(subset=["finger_condition", "within_finger_time_bin", "success_rate"])
    if subject_df.empty:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    subject_colors = _subject_palette(subject_df["subject_id"]) if "subject_id" in subject_df else {}
    for (subject, finger), g in subject_df.groupby(["subject_id", "finger_condition"], dropna=False):
        g = g.sort_values("within_finger_time_bin")
        ax.plot(g["within_finger_time_bin"], g["success_rate"], color=subject_colors.get(str(subject), "0.65"), alpha=0.22, linewidth=0.8, zorder=1)
        ax.scatter(g["within_finger_time_bin"], g["success_rate"], facecolor=_finger_color(finger), edgecolor=subject_colors.get(str(subject), "0.55"), linewidth=0.5, s=24, alpha=0.55, zorder=2)
    summary = group_bins.copy() if group_bins is not None and not group_bins.empty else (
        subject_df.groupby(["finger_condition", "within_finger_time_bin"], dropna=False)
        .agg(mean_success_rate=("success_rate", "mean"))
        .reset_index()
    )
    for finger, g in summary.groupby("finger_condition", dropna=False):
        g = g.sort_values("within_finger_time_bin")
        ax.plot(g["within_finger_time_bin"], g["mean_success_rate"], color=_finger_color(finger), marker="o", linewidth=2.8, label=_finger_label(finger), zorder=4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Trial position within finger block (1-64)")
    ax.set_ylabel("Success rate")
    ax.set_title("Within-finger learning/fatigue by finger (per trial)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def _save_all_success_order_slopes(
    slopes: pd.DataFrame,
    *,
    out_path: Path,
    fig_dpi: int,
) -> Path | None:
    if slopes is None or slopes.empty or "success_vs_order_slope" not in slopes or "finger_condition" not in slopes:
        return None
    import matplotlib.pyplot as plt

    plot_df = slopes.copy()
    plot_df["success_vs_order_slope"] = pd.to_numeric(plot_df["success_vs_order_slope"], errors="coerce")
    plot_df = plot_df.dropna(subset=["finger_condition", "success_vs_order_slope"])
    if plot_df.empty:
        return None
    order = _finger_order_present(plot_df["finger_condition"])
    x_map = {finger: i for i, finger in enumerate(order)}
    subject_colors = _subject_palette(plot_df["subject_id"]) if "subject_id" in plot_df else {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for _, row in plot_df.iterrows():
        finger = str(row["finger_condition"])
        subject = str(row.get("subject_id", ""))
        jitter = (((_stable_hash(subject, finger, "slope") % 100) / 100.0) - 0.5) * 0.28
        ax.scatter(
            x_map.get(finger, 0) + jitter,
            row["success_vs_order_slope"],
            s=58,
            facecolor=_finger_color(finger),
            edgecolor=subject_colors.get(subject, "black"),
            linewidth=1.3,
            alpha=0.92,
            zorder=3,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_finger_label(f) for f in order])
    ax.set_ylabel("Within participant success slope over session")
    ax.set_xlabel("Finger condition")
    ax.set_title("Success trend slopes: fill=finger, outline=participant")
    fig.tight_layout()
    _finalize_fig(fig, out_path, fig_dpi)
    return out_path

def plot_fit_curve(
    agg: pd.DataFrame,
    fit_row: pd.Series,
    title: str,
    out_path: Path,
    fig_dpi: int = 160,
    background_agg: Optional[pd.DataFrame] = None,
    background_fits: Optional[pd.DataFrame] = None,
    curve_color: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    finger = fit_row.get("finger_condition", None)
    curve_color = curve_color or _finger_color(finger, "black")

    if background_agg is not None and background_fits is not None and not background_agg.empty and not background_fits.empty:
        # Loop-invariant: depends only on background_agg, so compute once instead
        # of re-deriving it for every background fit (this was the super-linear
        # cost on the group/all-pooled figures). Result is identical.
        bg_all = add_delta_and_less_response_columns(background_agg)
        background_workspaces: set[str] = set()
        for _, bg_fit in background_fits.iterrows():
            bg = bg_all
            if "subject_id" in bg and "subject_id" in bg_fit:
                bg = bg[bg["subject_id"].astype(str) == str(bg_fit["subject_id"])]
            if "finger_condition" in bg and "finger_condition" in bg_fit:
                bg = bg[bg["finger_condition"].astype(str) == str(bg_fit["finger_condition"])]
            bg = bg.dropna(subset=["delta_comparison_minus_standard", "p_comparison_greater"])
            if len(bg) < 2:
                continue
            x_bg = np.linspace(float(bg["delta_comparison_minus_standard"].min()), float(bg["delta_comparison_minus_standard"].max()), 200)
            std_bg = float(pd.to_numeric(bg["standard_value"], errors="coerce").median())
            y_bg = _fit_row_to_delta_predictions(bg_fit, std_bg, x_bg)
            if np.isfinite(y_bg).any():
                base_bg = _finger_color(bg_fit.get("finger_condition", finger), "0.45")
                ws = _workspace_letter(bg_fit.get("workspace_setup"))
                # L workspace -> darker shade, N workspace -> lighter shade.
                ax.plot(x_bg, y_bg, color=_shade_for_workspace(base_bg, ws), alpha=0.22 if ws else 0.12, linewidth=0.9, zorder=1)
                if ws:
                    background_workspaces.add(ws)
        for ws in ("L", "N"):
            if ws in background_workspaces:
                ax.plot([], [], color=_shade_for_workspace(curve_color, ws), linewidth=2.0,
                        label=f"{ws} workspace ({'darker' if ws == 'L' else 'lighter'})")

    agg = add_delta_and_less_response_columns(agg).sort_values("delta_comparison_minus_standard").copy()
    if {"p_comparison_greater_ci95_lower", "p_comparison_greater_ci95_upper"}.issubset(agg.columns):
        yerr = np.vstack([
            agg["p_comparison_greater"] - agg["p_comparison_greater_ci95_lower"],
            agg["p_comparison_greater_ci95_upper"] - agg["p_comparison_greater"],
        ])
        ax.errorbar(agg["delta_comparison_minus_standard"], agg["p_comparison_greater"], yerr=yerr, fmt="none", ecolor="0.35", alpha=0.65, capsize=3, zorder=2)

    delta_values = pd.to_numeric(agg["delta_comparison_minus_standard"], errors="coerce")
    scatter = ax.scatter(
        agg["delta_comparison_minus_standard"],
        agg["p_comparison_greater"],
        c=delta_values,
        cmap=STIFFNESS_CMAP,
        s=42,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.9,
        label="Observed mean",
        zorder=3,
    )
    if delta_values.nunique(dropna=True) > 1:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(PSYCHOMETRIC_DELTA_AXIS_LABEL)

    if len(agg) >= 2:
        x_grid = np.linspace(float(agg["delta_comparison_minus_standard"].min()), float(agg["delta_comparison_minus_standard"].max()), 300)
        std = float(pd.to_numeric(agg["standard_value"], errors="coerce").median())
        y_grid = _fit_row_to_delta_predictions(fit_row, std, x_grid)
        if np.isfinite(y_grid).any():
            ax.plot(x_grid, y_grid, color=curve_color, linewidth=2.2, label="Fit", zorder=4)
    pse = pd.to_numeric(pd.Series([fit_row.get("pse_delta_comparison_minus_standard", np.nan)]), errors="coerce").iloc[0]
    if not np.isfinite(pse) and "pse" in fit_row and "standard_value" in agg and agg["standard_value"].notna().any():
        fit_pse = pd.to_numeric(pd.Series([fit_row.get("pse")]), errors="coerce").iloc[0]
        pse = fit_pse - float(pd.to_numeric(agg["standard_value"], errors="coerce").median())
    pse_ci_lower_delta = pd.to_numeric(pd.Series([fit_row.get("pse_delta_ci95_lower", np.nan)]), errors="coerce").iloc[0]
    pse_ci_upper_delta = pd.to_numeric(pd.Series([fit_row.get("pse_delta_ci95_upper", np.nan)]), errors="coerce").iloc[0]
    if (not (np.isfinite(pse_ci_lower_delta) and np.isfinite(pse_ci_upper_delta))) and "standard_value" in agg and agg["standard_value"].notna().any():
        std_for_ci = float(pd.to_numeric(agg["standard_value"], errors="coerce").median())
        pse_ci_lower_raw = pd.to_numeric(pd.Series([fit_row.get("pse_ci95_lower", np.nan)]), errors="coerce").iloc[0]
        pse_ci_upper_raw = pd.to_numeric(pd.Series([fit_row.get("pse_ci95_upper", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(pse_ci_lower_raw):
            pse_ci_lower_delta = pse_ci_lower_raw - std_for_ci
        if np.isfinite(pse_ci_upper_raw):
            pse_ci_upper_delta = pse_ci_upper_raw - std_for_ci
    standard_inside_ci = bool(np.isfinite(pse_ci_lower_delta) and np.isfinite(pse_ci_upper_delta) and pse_ci_lower_delta <= 0 <= pse_ci_upper_delta)
    if np.isfinite(pse):
        # The PSE 95% CI from degenerate fits can run far outside the tested
        # range. Clip the drawn band/error bar to the symmetric +/-axis window and
        # flag it "CI unreliable" rather than letting it blow up the figure.
        axis_limit = PSYCHOMETRIC_DELTA_AXIS_LIMIT
        have_ci = bool(np.isfinite(pse_ci_lower_delta) and np.isfinite(pse_ci_upper_delta))
        lo_clip = max(float(pse_ci_lower_delta), -axis_limit) if have_ci else np.nan
        hi_clip = min(float(pse_ci_upper_delta), axis_limit) if have_ci else np.nan
        ci_overflow = bool(have_ci and ((pse_ci_lower_delta < -axis_limit) or (pse_ci_upper_delta > axis_limit)))
        pse_out_of_band = not bool(fit_row.get("pse_in_valid_band", True))
        ci_unreliable = bool(ci_overflow or pse_out_of_band)

        pse_label = f"PSE shift={pse:.2f}"
        if have_ci:
            pse_label += f" [{lo_clip:.2f}, {hi_clip:.2f}]" + ("*" if ci_unreliable else "")
        ax.axvline(float(pse), color=curve_color, linestyle="--", linewidth=1.5, label=pse_label)
        if have_ci and hi_clip > lo_clip:
            ax.axvspan(lo_clip, hi_clip, color=curve_color, alpha=0.10, zorder=0)
            if lo_clip <= float(pse) <= hi_clip:
                ax.errorbar(
                    [float(pse)],
                    [0.5],
                    xerr=[[float(pse) - lo_clip], [hi_clip - float(pse)]],
                    fmt="o",
                    color=curve_color,
                    ecolor=curve_color,
                    elinewidth=1.6,
                    capsize=4,
                    markersize=5,
                    zorder=5,
                    label="95% CI on PSE" + (" (clipped, unreliable)" if ci_unreliable else ""),
                )
        # Info box pinned to the TOP-LEFT (empty region for an increasing curve),
        # so it stays on the left and off the plotted curve; the legend is moved
        # to the bottom-right below. The "CI unreliable" flag is folded in as a
        # red line here.
        bias_p = pd.to_numeric(pd.Series([fit_row.get("pse_bias_p_value", np.nan)]), errors="coerce").iloc[0]
        info_lines = [f"standard {'INSIDE' if standard_inside_ci else 'OUTSIDE'} 95% CI"]
        if np.isfinite(bias_p):
            info_lines.append(f"Wald p={bias_p:.3g}")
        if ci_unreliable:
            info_lines.append("CI unreliable")
        ax.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="firebrick" if ci_unreliable else "black",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="firebrick" if ci_unreliable else "0.6", linewidth=0.6),
        )
    ax.axvline(0, color="black", linestyle=":", linewidth=1, label="standard (delta=0)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    set_psychometric_delta_axis(ax, agg["delta_comparison_minus_standard"])
    ax.set_ylabel(PSYCHOMETRIC_GREATER_Y_LABEL)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=fig_dpi)
    return fig, ax

def _render_fit_curve_job(job: dict[str, Any]) -> Path:
    """Render one psychometric-curve figure to disk and return its path.

    Top-level (picklable) so it can run in a worker process. Uses the headless
    Agg backend explicitly; matplotlib's PNG rendering is deterministic, so the
    output is pixel-identical whether this runs serially or in parallel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Replicate the styling state that save_all_figures sets in the parent
    # process; worker processes do not inherit the parent's global matplotlib
    # rcParams, and the whitegrid theme changes grid/background/font rendering.
    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid")
    except Exception:  # pragma: no cover - seaborn optional
        pass

    fig, _ = plot_fit_curve(
        job["agg"],
        job["fit_row"],
        job["title"],
        job["out_path"],
        job["fig_dpi"],
        background_agg=job.get("background_agg"),
        background_fits=job.get("background_fits"),
        curve_color=job.get("curve_color"),
    )
    plt.close(fig)
    return job["out_path"]

def _render_fit_curve_jobs(jobs: list[dict[str, Any]], n_jobs: Optional[int] = None) -> list[Path]:
    """Render a list of curve jobs, in parallel across processes when worthwhile."""
    if not jobs:
        return []
    n_jobs_eff = _resolve_fit_n_jobs(len(jobs), n_jobs)
    if n_jobs_eff <= 1:
        return [_render_fit_curve_job(job) for job in jobs]
    try:
        from joblib import Parallel, delayed

        return list(Parallel(n_jobs=n_jobs_eff)(delayed(_render_fit_curve_job)(job) for job in jobs))
    except Exception:  # pragma: no cover - joblib missing/unavailable -> serial
        return [_render_fit_curve_job(job) for job in jobs]

def _group_curve_background_fits(fits: pd.DataFrame) -> pd.DataFrame:
    """Keep only the subjects that belong in the GROUP psychometric curves.

    The faded background curves on the per-finger / all-pooled group figures must
    show only the experiment subjects that enter the group analysis. This drops:
      * pilot/protocol subjects (``experiment_group`` is set only for the main
        experiment groups N_E/L_E, and is NaN for the *_P pilots); and
      * band-pass-disqualified fits whose PSE fell outside the valid range
        (``excluded_from_group_analysis``).
    Individual per-subject curves elsewhere still show every subject (marked).
    """
    if fits is None or fits.empty:
        return fits
    out = fits
    if "experiment_group" in out.columns:
        out = out[out["experiment_group"].notna()]
    if "excluded_from_group_analysis" in out.columns:
        out = out[~out["excluded_from_group_analysis"].astype(bool)]
    return out

def save_all_figures(
    output_root: Path,
    clean: pd.DataFrame,
    qc_summary: pd.DataFrame,
    psychometric_input_by_subject_finger: pd.DataFrame,
    pse_jnd_by_subject_finger: pd.DataFrame,
    psychometric_input_group_by_finger: pd.DataFrame,
    pse_jnd_group_by_finger: pd.DataFrame,
    psychometric_input_group_all_pooled: pd.DataFrame,
    pse_jnd_group_all_pooled: pd.DataFrame,
    order_effects_binned: pd.DataFrame,
    fig_dpi: int = 160,
    fig_n_jobs: Optional[int] = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore
        sns.set_theme(style="whitegrid")
    except Exception:  # pragma: no cover
        sns = None

    paths: list[Path] = []
    fig_root = output_root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    # Build all psychometric-curve figure jobs first, then render them in one
    # shared process pool. Each figure is independent and rendered deterministically
    # (Agg), so the PNGs are pixel-identical to the previous serial version; the
    # job order below preserves the original order of `paths`.
    curve_jobs: list[dict[str, Any]] = []
    if not psychometric_input_by_subject_finger.empty and not pse_jnd_by_subject_finger.empty:
        for _, fit_row in pse_jnd_by_subject_finger.iterrows():
            subject, finger = fit_row["subject_id"], fit_row["finger_condition"]
            agg = psychometric_input_by_subject_finger[
                (psychometric_input_by_subject_finger["subject_id"] == subject)
                & (psychometric_input_by_subject_finger["finger_condition"] == finger)
            ]
            out = fig_root / "subject_finger_curves" / sanitize_name(subject) / f"psychometric_{sanitize_name(subject)}_{sanitize_name(finger)}.png"
            curve_jobs.append({
                "agg": agg.copy(),
                "fit_row": fit_row,
                "title": f"Subject {subject} – finger {finger}",
                "out_path": out,
                "fig_dpi": fig_dpi,
                "curve_color": _finger_color(finger),
            })

    if not psychometric_input_group_by_finger.empty and not pse_jnd_group_by_finger.empty:
        for _, fit_row in pse_jnd_group_by_finger.iterrows():
            finger = fit_row["finger_condition"]
            agg = psychometric_input_group_by_finger[psychometric_input_group_by_finger["finger_condition"] == finger]
            bg_agg = psychometric_input_by_subject_finger[psychometric_input_by_subject_finger["finger_condition"].astype(str) == str(finger)]
            bg_fits = _group_curve_background_fits(
                pse_jnd_by_subject_finger[pse_jnd_by_subject_finger["finger_condition"].astype(str) == str(finger)]
            )
            out = fig_root / "group_curves" / f"group_finger_{sanitize_name(finger)}.png"
            curve_jobs.append({
                "agg": agg.copy(),
                "fit_row": fit_row,
                "title": f"Group pooled – finger {finger}",
                "out_path": out,
                "fig_dpi": fig_dpi,
                "background_agg": bg_agg.copy(),
                "background_fits": bg_fits.copy(),
                "curve_color": _finger_color(finger),
            })

    if not psychometric_input_group_all_pooled.empty and not pse_jnd_group_all_pooled.empty:
        curve_jobs.append({
            "agg": psychometric_input_group_all_pooled.copy(),
            "fit_row": pse_jnd_group_all_pooled.iloc[0],
            "title": "Group all-pooled",
            "out_path": fig_root / "group_curves" / "group_all_pooled.png",
            "fig_dpi": fig_dpi,
            "background_agg": psychometric_input_by_subject_finger.copy(),
            "background_fits": _group_curve_background_fits(pse_jnd_by_subject_finger).copy(),
            "curve_color": "black",
        })

    paths.extend(_render_fit_curve_jobs(curve_jobs, fig_n_jobs))

    if not pse_jnd_by_subject_finger.empty:
        for metric, ylabel, filename in [("pse", "PSE", "pse_per_subject_and_finger.png"), ("jnd", "JND", "jnd_per_subject_and_finger.png")]:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            plot_df = pse_jnd_by_subject_finger.copy()
            order = _finger_order_present(plot_df["finger_condition"])
            x_map = {finger: i for i, finger in enumerate(order)}
            plot_df["_x_num"] = plot_df["finger_condition"].map(x_map)
            for subject, g in plot_df.groupby("subject_id", dropna=False):
                g = g.sort_values("_x_num")
                ax.plot(g["_x_num"], g[metric], color="0.75", linewidth=0.8, alpha=0.45, zorder=1)
            values = pd.to_numeric(plot_df[metric], errors="coerce")
            scatter = ax.scatter(
                plot_df["_x_num"],
                values,
                c=values,
                cmap=STIFFNESS_CMAP,
                s=48,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.9,
                zorder=2,
            )
            group = plot_df.groupby("_x_num", dropna=False)[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(group["_x_num"], group["mean"], yerr=group["std"], color="black", marker="o", linewidth=2, capsize=4, label="Mean ± SD", zorder=3)
            if values.nunique(dropna=True) > 1:
                cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
                cbar.set_label(ylabel)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
            ax.legend(loc="best", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Finger condition")
            ax.set_title(f"{ylabel} per subject and finger")
            fig.tight_layout()
            out = fig_root / filename
            _finalize_fig(fig, out, fig_dpi)
            paths.append(out)

    if not order_effects_binned.empty:
        fingers = _finger_order_present(order_effects_binned["finger_condition"])
        fig, axes = plt.subplots(len(fingers), 1, figsize=(8.5, 2.4 * len(fingers)), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, finger in zip(axes, fingers):
            color = _finger_color(finger)
            g = order_effects_binned[order_effects_binned["finger_condition"].astype(str) == str(finger)].copy()
            for _, sg in g.groupby("subject_id", dropna=False):
                sg = sg.sort_values("mean_global_trial_order")
                ax.plot(sg["mean_global_trial_order"], sg["p_comparison_greater"], color=color, alpha=0.18, linewidth=0.8, zorder=1)
            group = (
                g.groupby("order_bin", dropna=False)
                .agg(
                    mean_global_trial_order=("mean_global_trial_order", "mean"),
                    p_comparison_greater=("p_comparison_greater", "mean"),
                    sem=("p_comparison_greater", _sem),
                )
                .reset_index()
                .sort_values("mean_global_trial_order")
            )
            ax.errorbar(group["mean_global_trial_order"], group["p_comparison_greater"], yerr=group["sem"], color=color, marker="o", linewidth=2, capsize=3, zorder=2)
            ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(_finger_label(finger))
        axes[-1].set_xlabel("Global trial order")
        fig.supylabel(PSYCHOMETRIC_GREATER_Y_LABEL)
        fig.suptitle("Order/fatigue trend by finger (individuals faded, mean bold)")
        fig.tight_layout()
        out = fig_root / "order_effects.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    if not clean.empty:
        side = clean.groupby(["subject_id", "finger_condition"]).agg(
            p_chose_object2=("answer_chose_object_2", "mean"),
            p_standard_on_object2=("standard_side", lambda x: float((x == "object_2").mean())),
            n_trials=("answer_chose_object_2", "size"),
        ).reset_index()
        side.to_csv(output_root / "side_bias_summary.csv", index=False)
        # Split each finger into two bars: object 1 (left, orange) showing
        # P(chose object 1) = 1 - p_chose_object2, and object 2 (right, blue)
        # showing P(chose object 2). One dot per subject, edge = subject colour.
        order = _finger_order_present(side["finger_condition"])
        x_map = {finger: idx for idx, finger in enumerate(order)}
        subject_colors = _subject_palette(side["subject_id"])
        bar_dx, bar_w = 0.21, 0.38
        objects = [
            (-bar_dx, lambda p: 1.0 - p, OBJECT1_COLOR, "Object 1 (left)"),
            (+bar_dx, lambda p: p, OBJECT2_COLOR, "Object 2 (right)"),
        ]
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for dx, ytf, color, label in objects:
            xs, means, sds = [], [], []
            for finger in order:
                vals = ytf(side.loc[side["finger_condition"].astype(str) == str(finger), "p_chose_object2"])
                xs.append(x_map[finger] + dx)
                means.append(float(vals.mean()) if len(vals) else np.nan)
                sds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
            ax.bar(
                xs, means, width=bar_w, color=color, alpha=0.55, edgecolor="black",
                linewidth=0.5, label=label, zorder=1, yerr=sds, capsize=3,
                error_kw={"zorder": 1, "elinewidth": 1},
            )
            for finger in order:
                sub = side[side["finger_condition"].astype(str) == str(finger)]
                if sub.empty:
                    continue
                x = x_map[finger] + dx
                n = len(sub)
                jit = np.linspace(-bar_w * 0.3, bar_w * 0.3, n) if n > 1 else np.array([0.0])
                yv = ytf(sub["p_chose_object2"]).to_numpy()
                edges = [subject_colors.get(str(s), "0.4") for s in sub["subject_id"]]
                ax.scatter(x + jit, yv, facecolor=color, edgecolor=edges, linewidth=1.1, s=42, alpha=0.9, zorder=3)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, zorder=2)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([_finger_label(f) for f in order])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(chose object)")
        ax.set_xlabel("Finger condition")
        ax.set_title("Side bias")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        fig.tight_layout()
        out = fig_root / "side_bias_fingers.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    if not qc_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        if sns is not None:
            sns.scatterplot(data=qc_summary, x="n_stimulus_levels", y="n_clean_trials", hue="finger_condition", hue_order=_finger_order_present(qc_summary["finger_condition"]), palette={f: _finger_color(f) for f in _finger_order_present(qc_summary["finger_condition"])}, style="qc_warnings", ax=ax)
        else:
            ax.scatter(qc_summary["n_stimulus_levels"], qc_summary["n_clean_trials"])
        ax.set_title("QC: trials and levels")
        ax.set_xlabel("Number of stimulus levels")
        ax.set_ylabel("Clean trials")
        ax = axes[1]
        plot_qc = qc_summary.copy()
        order = _finger_order_present(plot_qc["finger_condition"])
        if sns is not None:
            sns.barplot(data=plot_qc, x="finger_condition", y="apparent_error_rate", order=order, errorbar="sd", ax=ax, color="0.85")
            np.random.seed(_FIGURE_RNG_SEED)  # reproducible stripplot jitter
            sns.stripplot(
                data=plot_qc,
                x="finger_condition",
                y="apparent_error_rate",
                order=order,
                hue="finger_condition",
                hue_order=order,
                palette={f: _finger_color(f) for f in order},
                dodge=False,
                jitter=0.18,
                size=5,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.3,
                ax=ax,
            )
            if ax.legend_:
                ax.legend_.remove()
        else:
            x_map = {f: i for i, f in enumerate(order)}
            ax.scatter(plot_qc["finger_condition"].map(x_map), plot_qc["apparent_error_rate"], c=[_finger_color(f) for f in plot_qc["finger_condition"]])
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
        ax.axhline(0.35, color="red", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title("QC: apparent error rate")
        fig.tight_layout()
        out = fig_root / "qc_summary_plots.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    return paths

def save_time_fatigue_figures(
    output_root: Path,
    success_by_reaction_time_bin: pd.DataFrame,
    success_by_order_bin: pd.DataFrame,
    fatigue_first_second_summary: pd.DataFrame,
    success_summary_by_subject: pd.DataFrame,
    success_trend_slopes: pd.DataFrame,
    fig_dpi: int = 160,
) -> list[Path]:
    """Save figures for response-duration and fatigue/order success analyses."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid")
    except Exception:  # pragma: no cover
        sns = None

    paths: list[Path] = []
    fig_root = output_root / "figures" / "time_fatigue"
    fig_root.mkdir(parents=True, exist_ok=True)

    if not success_by_reaction_time_bin.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_df = success_by_reaction_time_bin.copy()
        order = _finger_order_present(plot_df["finger_condition"])
        if sns is not None:
            sns.lineplot(data=plot_df, x="reaction_time_bin", y="success_rate", hue="finger_condition", hue_order=order, palette={f: _finger_color(f) for f in order}, marker="o", errorbar="se", ax=ax)
        else:
            for finger, g in plot_df.groupby("finger_condition"):
                ax.plot(g["reaction_time_bin"], g["success_rate"], marker="o", label=str(finger))
            ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Reaction-time quartile within participant/finger")
        ax.set_ylabel("Success rate")
        ax.set_title("Effect of answer duration on success")
        fig.tight_layout()
        out = fig_root / "success_by_reaction_time_bin.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    if not success_by_order_bin.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_df = success_by_order_bin.copy()
        order = _finger_order_present(plot_df["finger_condition"])
        if sns is not None:
            sns.lineplot(data=plot_df, x="order_bin", y="success_rate", hue="finger_condition", hue_order=order, palette={f: _finger_color(f) for f in order}, marker="o", errorbar="se", ax=ax)
        else:
            for finger, g in plot_df.groupby("finger_condition"):
                ax.plot(g["order_bin"], g["success_rate"], marker="o", label=str(finger))
            ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Trial-order bin within participant/finger")
        ax.set_ylabel("Success rate")
        ax.set_title("Fatigue/learning proxy: success across session order")
        fig.tight_layout()
        out = fig_root / "success_by_trial_order_bin.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    if not fatigue_first_second_summary.empty and "success_rate_second_minus_first" in fatigue_first_second_summary:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        plot_df = fatigue_first_second_summary.copy()
        plot_df["subject_finger"] = plot_df["subject_id"].astype(str) + "-" + plot_df["finger_condition"].astype(str)
        order = _finger_order_present(plot_df["finger_condition"])
        if sns is not None:
            sns.barplot(data=plot_df, x="subject_finger", y="success_rate_second_minus_first", hue="finger_condition", hue_order=order, palette={f: _finger_color(f) for f in order}, dodge=False, ax=ax)
        else:
            ax.bar(plot_df["subject_finger"], plot_df["success_rate_second_minus_first"])
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Second-half minus first-half success rate")
        ax.set_xlabel("Participant-finger")
        ax.set_title("Within-participant fatigue/learning direction")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        out = fig_root / "fatigue_second_minus_first_by_subject_finger.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    if not success_summary_by_subject.empty:
        for x_col, xlabel, filename in [
            ("mean_reaction_time", "Mean answer duration (s)", "between_subject_success_vs_mean_reaction_time.png"),
            ("session_duration_min", "Session elapsed duration (min)", "between_subject_success_vs_session_duration.png"),
        ]:
            if x_col not in success_summary_by_subject:
                continue
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            plot_df = success_summary_by_subject.copy()
            if sns is not None:
                sns.scatterplot(data=plot_df, x=x_col, y="success_rate", hue="subject_group_label", s=80, ax=ax)
                sns.regplot(data=plot_df, x=x_col, y="success_rate", scatter=False, color="black", seed=_FIGURE_RNG_SEED, ax=ax)
            else:
                ax.scatter(plot_df[x_col], plot_df["success_rate"])
            for _, row in plot_df.iterrows():
                ax.annotate(str(row["subject_id"]), (row[x_col], row["success_rate"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Participant success rate")
            ax.set_title(xlabel + " vs success (between participants)")
            fig.tight_layout()
            out = fig_root / filename
            _finalize_fig(fig, out, fig_dpi)
            paths.append(out)

    if not success_trend_slopes.empty and "success_vs_order_slope" in success_trend_slopes:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_df = success_trend_slopes.copy()
        order = _finger_order_present(plot_df["finger_condition"])
        if sns is not None:
            np.random.seed(_FIGURE_RNG_SEED)  # reproducible stripplot jitter
            sns.stripplot(data=plot_df, x="finger_condition", y="success_vs_order_slope", order=order, hue="finger_condition", hue_order=order, palette={f: _finger_color(f) for f in order}, dodge=False, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
            elif ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            ax.scatter(plot_df["finger_condition"].astype(str), plot_df["success_vs_order_slope"])
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Within participant success slope over session")
        ax.set_xlabel("Finger condition")
        ax.set_title("Success trend slopes: positive=improves, negative=fatigue")
        fig.tight_layout()
        out = fig_root / "success_order_slopes_by_finger.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    return paths

def save_finger_time_appearance_figures(
    output_root: Path,
    finger_time_group_bins: pd.DataFrame,
    finger_appearance_order_summary: pd.DataFrame,
    finger_by_appearance_order: pd.DataFrame,
    finger_time_slope_summary: pd.DataFrame,
    fig_dpi: int = 160,
    stiffness_time_slope_summary: Optional[pd.DataFrame] = None,
    subject_finger_time_bins: Optional[pd.DataFrame] = None,
) -> list[Path]:
    """Save figures for finger-by-time and finger-appearance-order analyses.

    The within-finger success-over-time curve is saved in three resolutions:
    - ``success_by_finger_over_within_finger_time.png``: raw per-trial position
      (1..64), noisy because each x mixes different shuffled stiffnesses;
    - ``success_by_finger_over_within_finger_time_smoothed.png``: the same 64
      positions with a centred rolling mean to expose the slow trend;
    - ``success_by_finger_over_within_finger_time_8bins.png``: the coarse 8
      quantile bins (8 trials each) as in the original analysis. When
      ``subject_finger_time_bins`` is provided the 8-bin means and SEM are
      rebuilt at the subject level (identical to the original statistic);
      otherwise they are approximated by averaging the group per-trial means.
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid")
    except Exception:  # pragma: no cover
        sns = None

    paths: list[Path] = []
    fig_root = output_root / "figures" / "finger_time_appearance"
    fig_root.mkdir(parents=True, exist_ok=True)

    if not finger_time_group_bins.empty:
        fig, ax = plt.subplots(figsize=(12.0, 5.0))
        plot_df = finger_time_group_bins.copy()
        order = _finger_order_present(plot_df["finger_condition"])
        if sns is not None:
            sns.lineplot(data=plot_df, x="within_finger_time_bin", y="mean_success_rate", hue="finger_condition", hue_order=order, palette={f: _finger_color(f) for f in order}, marker="o", markersize=3, ax=ax)
        else:
            for finger, g in plot_df.groupby("finger_condition"):
                ax.plot(g["within_finger_time_bin"], g["mean_success_rate"], marker="o", color=_finger_color(finger), label=str(finger))
            ax.legend()
        for finger, g in plot_df.groupby("finger_condition"):
            if "sem_success_rate" in g:
                ax.errorbar(g["within_finger_time_bin"], g["mean_success_rate"], yerr=g["sem_success_rate"], fmt="none", color=_finger_color(finger), alpha=0.35)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Trial position within finger block (1-64)")
        ax.set_ylabel("Mean success rate across participants")
        ax.set_title("Within-block learning/fatigue by finger (per trial)")
        fig.tight_layout()
        out = fig_root / "success_by_finger_over_within_finger_time.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

        # Smoothed view: same 64 per-trial positions with a centred rolling mean
        # so the slow learning/fatigue trend is readable through the per-trial noise.
        smooth_window = 7
        fig, ax = plt.subplots(figsize=(12.0, 5.0))
        for finger in order:
            g = plot_df[plot_df["finger_condition"].astype(str) == str(finger)].sort_values("within_finger_time_bin")
            if g.empty:
                continue
            color = _finger_color(finger)
            ax.plot(g["within_finger_time_bin"], g["mean_success_rate"], color=color, alpha=0.20, linewidth=0.8, zorder=1)
            smoothed = g["mean_success_rate"].rolling(window=smooth_window, center=True, min_periods=1).mean()
            ax.plot(g["within_finger_time_bin"], smoothed, color=color, linewidth=2.6, label=_finger_label(finger), zorder=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Trial position within finger block (1-64)")
        ax.set_ylabel("Mean success rate across participants")
        ax.set_title(f"Within-block learning/fatigue by finger (per trial, rolling mean w={smooth_window})")
        ax.legend(title="finger_condition")
        fig.tight_layout()
        out = fig_root / "success_by_finger_over_within_finger_time_smoothed.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

        # Coarse 8-bin view (8 trials per bin), as in the original analysis.
        bins_df: pd.DataFrame
        if subject_finger_time_bins is not None and not subject_finger_time_bins.empty and "within_finger_time_bin" in subject_finger_time_bins:
            # Rebuild the exact original statistic: bin each subject's 64 trials
            # into 8 quantile bins, success rate per (subject, finger, bin), then
            # mean and SEM across subjects.
            sdf = subject_finger_time_bins.copy()
            sdf["within_finger_time_bin"] = pd.to_numeric(sdf["within_finger_time_bin"], errors="coerce")
            sdf["success_rate"] = pd.to_numeric(sdf["success_rate"], errors="coerce")
            sdf = sdf.dropna(subset=["finger_condition", "within_finger_time_bin", "success_rate"])
            sdf["coarse_bin"] = ((sdf["within_finger_time_bin"].astype(int) - 1) // 8) + 1
            subj_bin = (
                sdf.groupby(["subject_id", "finger_condition", "coarse_bin"], dropna=False)["success_rate"]
                .mean()
                .reset_index()
            )
            bins_df = (
                subj_bin.groupby(["finger_condition", "coarse_bin"], dropna=False)["success_rate"]
                .agg(mean_success_rate="mean", sem_success_rate=_sem)
                .reset_index()
                .rename(columns={"coarse_bin": "time_bin_8"})
            )
        else:
            # Fallback: average the group per-trial means into 8 consecutive bins
            # (means match the subject-level statistic; SEM is unavailable).
            gdf = plot_df.copy()
            gdf["time_bin_8"] = ((pd.to_numeric(gdf["within_finger_time_bin"], errors="coerce").astype(int) - 1) // 8) + 1
            bins_df = (
                gdf.groupby(["finger_condition", "time_bin_8"], dropna=False)["mean_success_rate"]
                .mean()
                .reset_index()
            )
            bins_df["sem_success_rate"] = np.nan

        if not bins_df.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            for finger in order:
                g = bins_df[bins_df["finger_condition"].astype(str) == str(finger)].sort_values("time_bin_8")
                if g.empty:
                    continue
                color = _finger_color(finger)
                ax.plot(g["time_bin_8"], g["mean_success_rate"], marker="o", color=color, linewidth=2.0, label=_finger_label(finger))
                if g["sem_success_rate"].notna().any():
                    ax.errorbar(g["time_bin_8"], g["mean_success_rate"], yerr=g["sem_success_rate"], fmt="none", color=color, alpha=0.35)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(sorted(bins_df["time_bin_8"].dropna().astype(int).unique()))
            ax.set_xlabel("Time bin within finger block (8 bins x 8 trials)")
            ax.set_ylabel("Mean success rate across participants")
            ax.set_title("Within-block learning/fatigue by finger (8 quantile bins)")
            ax.legend(title="finger_condition")
            fig.tight_layout()
            out = fig_root / "success_by_finger_over_within_finger_time_8bins.png"
            _finalize_fig(fig, out, fig_dpi)
            paths.append(out)

    if not finger_appearance_order_summary.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        plot_df = finger_appearance_order_summary.copy()
        order = _appearance_order(plot_df["finger_appearance_order"])
        ax.plot(plot_df["finger_appearance_order"], plot_df["mean_success_rate"], color="0.25", linewidth=1.5, alpha=0.7)
        ax.errorbar(
            plot_df["finger_appearance_order"],
            plot_df["mean_success_rate"],
            yerr=plot_df.get("sem_success_rate"),
            fmt="none",
            color="0.25",
            linewidth=2,
            capsize=4,
        )
        ax.scatter(
            plot_df["finger_appearance_order"],
            plot_df["mean_success_rate"],
            c=plot_df["finger_appearance_order"],
            cmap=STIFFNESS_CMAP,
            s=70,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(order)
        ax.set_xticklabels(_appearance_tick_labels(order))
        ax.set_xlabel("Finger appearance order in session")
        ax.set_ylabel("Mean success rate")
        ax.set_title("Success by protocol appearance position (pooled across finger identity)")
        fig.tight_layout()
        out = fig_root / "success_by_finger_appearance_order.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    if not finger_by_appearance_order.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        plot_df = finger_by_appearance_order.copy()
        order = _finger_order_present(plot_df["finger_condition"])
        if sns is not None:
            sns.lineplot(data=plot_df, x="finger_appearance_order", y="mean_success_rate", hue="finger_condition", hue_order=order, palette={f: _finger_color(f) for f in order}, marker="o", ax=ax)
        else:
            for finger, g in plot_df.groupby("finger_condition"):
                ax.plot(g["finger_appearance_order"], g["mean_success_rate"], marker="o", color=_finger_color(finger), label=str(finger))
            ax.legend()
        ax.set_ylim(-0.05, 1.05)
        app_order = _appearance_order(plot_df["finger_appearance_order"])
        ax.set_xticks(app_order)
        ax.set_xticklabels(_appearance_tick_labels(app_order))
        ax.set_xlabel("Finger appearance order in session")
        ax.set_ylabel("Mean success rate")
        ax.set_title("Finger identity x appearance-order success")
        fig.tight_layout()
        out = fig_root / "success_by_finger_identity_and_appearance_order.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    if not finger_time_slope_summary.empty:
        n_panels = 2 if stiffness_time_slope_summary is not None and not stiffness_time_slope_summary.empty else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(7.2 * n_panels, 4.8))
        axes = np.atleast_1d(axes)
        ax = axes[0]
        plot_df = finger_time_slope_summary.copy()
        plot_df["finger_condition"] = pd.Categorical(plot_df["finger_condition"].astype(str), categories=FINGER_ORDER, ordered=True)
        plot_df = plot_df.sort_values("finger_condition")
        ax.bar(
            plot_df["finger_condition"].astype(str),
            plot_df["mean_success_slope"],
            yerr=plot_df.get("sem_success_slope"),
            color=[_finger_color(f) for f in plot_df["finger_condition"].astype(str)],
            capsize=4,
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Finger")
        ax.set_ylabel("Mean success slope over within-finger time")
        ax.set_title("By finger")
        if n_panels == 2:
            ax2 = axes[1]
            stiff_df = stiffness_time_slope_summary.copy()
            values = pd.to_numeric(stiff_df["comparison_value"], errors="coerce")
            ax2.bar(
                stiff_df["comparison_value"].astype(str),
                stiff_df["mean_success_slope"],
                yerr=stiff_df.get("sem_success_slope"),
                color=plt.get_cmap(STIFFNESS_CMAP)((values - values.min()) / max(float(values.max() - values.min()), 1e-12)),
                capsize=4,
            )
            ax2.axhline(0, color="black", linewidth=1)
            ax2.set_xlabel("Comparison stiffness")
            ax2.set_ylabel("Mean success slope over session time")
            ax2.set_title("By stiffness")
            ax2.tick_params(axis="x", rotation=45)
        fig.suptitle("Success time trend: positive=improves, negative=declines")
        fig.tight_layout()
        out = fig_root / "success_time_slope_by_finger.png"
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

    return paths

def save_success_by_stiffness_repetition_figures(
    output_root: Path,
    trials: pd.DataFrame,
    fig_dpi: int = 160,
) -> list[Path]:
    """Save per-finger success curves over the *repetition order of each stiffness*.

    This is an alternative view to ``success_by_finger_over_within_finger_time``
    (whose x-axis is a quantile time-bin within the finger block). Here every
    stiffness level appears a fixed number of times inside its finger block
    (typically 8), and the x-axis is the **appearance order of that stiffness**
    (1 = first time the level is shown in the block, 2 = second, ...). The
    question is whether participants improve / decline across repeats of the
    same stiffness.

    Layout: one subplot per finger (e.g. Index/Middle/Ring/Pinky). Within each
    panel:
    - colour encodes the stiffness level (signed delta, ``STIFFNESS_CMAP``),
    - the thick line is the across-subject mean success at each repetition for a
      given stiffness,
    - faint markers are individual subjects (marker fill = stiffness colour,
      marker edge/perimeter = subject colour), connected by thin per-subject
      lines coloured by stiffness.

    Note: for a single subject, one (stiffness, repetition) cell is a single
    trial, so per-subject markers sit at 0 or 1; the across-subject mean (thick
    line) carries the real signal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    paths: list[Path] = []
    if trials is None or trials.empty:
        return paths
    needed = {"subject_id", "finger_condition", "signed_stiffness_delta", "correct_response", "global_trial_order"}
    if not needed.issubset(trials.columns):
        return paths

    df = trials.copy()
    df["correct_response"] = pd.to_numeric(df["correct_response"], errors="coerce")
    df["signed_stiffness_delta"] = pd.to_numeric(df["signed_stiffness_delta"], errors="coerce")
    df["global_trial_order"] = pd.to_numeric(df["global_trial_order"], errors="coerce")
    # Physically-equal trials have no defined correct answer (correct_response is NaN).
    df = df.dropna(subset=["subject_id", "finger_condition", "signed_stiffness_delta", "correct_response", "global_trial_order"])
    if df.empty:
        return paths

    # Repetition index = order of appearance of each stiffness level within the
    # subject's finger block (1 = first time that stiffness appears, ...).
    df = df.sort_values(["subject_id", "finger_condition", "signed_stiffness_delta", "global_trial_order"])
    df["stiffness_repetition"] = (
        df.groupby(["subject_id", "finger_condition", "signed_stiffness_delta"], dropna=False).cumcount() + 1
    )

    fingers = _finger_order_present(df["finger_condition"])
    if not fingers:
        return paths

    fig_root = output_root / "figures" / "finger_time_appearance"
    fig_root.mkdir(parents=True, exist_ok=True)

    # Shared stiffness colour scale across all finger panels.
    deltas = np.sort(df["signed_stiffness_delta"].unique())
    vmin, vmax = float(deltas.min()), float(deltas.max())
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(STIFFNESS_CMAP)

    def stiff_color(delta: float) -> Any:
        return cmap(norm(float(delta)))

    subject_colors = _subject_palette(df["subject_id"])

    n = len(fingers)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 4.4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = list(axes.ravel())

    for ax, finger in zip(flat_axes, fingers):
        fdf = df[df["finger_condition"].astype(str) == str(finger)]
        # Thin per-subject lines/markers: fill = stiffness colour, edge = subject.
        for (subject, delta), g in fdf.groupby(["subject_id", "signed_stiffness_delta"], dropna=False):
            g = g.sort_values("stiffness_repetition")
            color = stiff_color(delta)
            ax.plot(g["stiffness_repetition"], g["correct_response"], color=color, alpha=0.12, linewidth=0.7, zorder=1)
            ax.scatter(
                g["stiffness_repetition"],
                g["correct_response"],
                facecolor=[color],
                edgecolor=subject_colors.get(str(subject), "0.4"),
                linewidth=0.6,
                s=22,
                alpha=0.55,
                zorder=2,
            )
        # Thick across-subject mean line per stiffness level.
        mean_df = (
            fdf.groupby(["signed_stiffness_delta", "stiffness_repetition"], dropna=False)["correct_response"]
            .mean()
            .reset_index()
        )
        for delta, g in mean_df.groupby("signed_stiffness_delta", dropna=False):
            g = g.sort_values("stiffness_repetition")
            ax.plot(g["stiffness_repetition"], g["correct_response"], color=stiff_color(delta), marker="o", linewidth=2.6, zorder=4)
        ax.set_ylim(-0.05, 1.05)
        reps = np.sort(fdf["stiffness_repetition"].unique())
        if len(reps):
            ax.set_xticks(reps)
        ax.set_title(_finger_label(finger))
        ax.set_xlabel("Stiffness repetition (appearance order in finger block)")
        ax.set_ylabel("Success rate")

    # Hide any unused panels (e.g. when fewer than nrows*ncols fingers present).
    for ax in flat_axes[n:]:
        ax.axis("off")

    # Stiffness colourbar shared across panels.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=flat_axes, shrink=0.85, pad=0.02)
    cbar.set_label("Stiffness delta (comparison - standard)")

    fig.suptitle("Success across repetitions of each stiffness, per finger\n(thick = across-subject mean; marker edge colour = subject)")
    out = fig_root / "success_by_stiffness_repetition_per_finger.png"
    _finalize_fig(fig, out, fig_dpi)
    paths.append(out)
    return paths


# Imported here (bottom of module) rather than at the top so that twoafc_psychophysics
# and twoafc_figures can be imported in either order: by the time either module runs
# its cross-import, the other's definitions are already complete. These names are used
# only inside the function bodies above (runtime), never at import/def time.
from twoafc_psychophysics import (  # noqa: E402
    FINGER_APPEARANCE_LABELS,
    FINGER_ORDER,
    FINGER_STYLE,
    OBJECT1_COLOR,
    OBJECT2_COLOR,
    PSYCHOMETRIC_DELTA_AXIS_LABEL,
    PSYCHOMETRIC_DELTA_AXIS_LIMIT,
    PSYCHOMETRIC_GREATER_Y_LABEL,
    PSYCHOMETRIC_MEAN_GREATER_Y_LABEL,
    STIFFNESS_CMAP,
    _METRIC_CI_COLUMN_HINTS,
    _fit_row_to_delta_predictions,
    _resolve_fit_n_jobs,
    _rmtree_windows_retry,
    _sem,
    _subject_sort_key,
    _wilson_ci95_lower,
    _wilson_ci95_upper,
    add_delta_and_less_response_columns,
    add_fit_delta_columns,
    sanitize_name,
    save_csv,
)
