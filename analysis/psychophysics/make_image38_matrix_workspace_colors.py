"""Regenerate image38_matrix.png as a 2x2 version of the shared-legend curve.

This is a presentation-focused remake of the group pooled finger psychometric
matrix used in the ICRA paper media folder. It reads the existing analysis CSVs,
so it does not rerun the full psychophysics pipeline.

``make_figure(layout="line")`` draws the same four panels side by side (1x4)
with identical styling; ``make_image38_line_workspace_colors.py`` wraps it to
write image38_line.png.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = SCRIPT_DIR / "results" / "L_N_E" / "csv" / "all" / "shared"
MEDIA_DIR = (
    REPO_ROOT
    / "paper"
    / "icra2027_paper"
    / "icra2027_latex_transfer"
    / "media"
    / "media"
)
OUT_PATH = MEDIA_DIR / "image38_matrix.png"
BACKUP_PATH = MEDIA_DIR / "image38_matrix_before_workspace_recolor.png"
LINE_OUT_PATH = MEDIA_DIR / "image38_line.png"
LINE_BACKUP_PATH = MEDIA_DIR / "image38_line_before_workspace_recolor.png"

# Match the N/L colors used in image37_left.png.
N_COLOR = "#6A3D9A"  # darker purple
L_COLOR = "#FF7FD3"  # brighter pink
STANDARD_COLOR = "0.35"
STIFFNESS_CMAP = "viridis"

FINGER_ORDER = ["I", "M", "R", "P"]
# Text sizes (points) shared by both layouts; raised for readability at column width.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 32, 30, 24, 24
# Delta label typeset as math with subscripts (used on the x axes and colorbar).
DELTA_LABEL = r"$G_{\mathrm{comparison}} - G_{\mathrm{standard}}$"
FINGER_COLORS = {
    "I": "#1f77b4",  # blue
    "M": "#ff7f0e",  # orange
    "R": "#2ca02c",  # green
    "P": "#d62728",  # red
}


def _load_analysis_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    import twoafc_psychophysics as pf  # type: ignore

    return pf


def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _workspace_from_row(row: pd.Series) -> str | None:
    value = row.get("workspace_setup")
    if isinstance(value, str) and value.strip():
        first = value.strip().upper()[0]
        if first in {"N", "L"}:
            return first
    subject = str(row.get("subject_id", "")).strip().upper()
    if subject.startswith("N"):
        return "N"
    if subject.startswith("L"):
        return "L"
    return None


def _workspace_color(workspace: str | None) -> str:
    if workspace == "N":
        return N_COLOR
    if workspace == "L":
        return L_COLOR
    return "0.70"


def _fmt_delta(value: float) -> str:
    """Format a gain delta already expressed in mm/m (the pipeline unit)."""
    return f"{value:g}"


def _fmt_pse_value(value: float) -> str:
    """Format a PSE delta already expressed in mm/m."""
    return f"{value:.2f}"


XTICKS = np.array([-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], dtype=float)  # mm/m
X_GRID = np.linspace(-6, 6, 300)
CBAR_TICKS = [-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6]


def _draw_panel(ax, finger: str, tables: dict, pf, norm) -> None:
    """Draw one finger panel: subject fits, observed means, group fit and PSE."""
    group_agg = tables["group_agg"]
    group_fits = tables["group_fits"]
    subject_agg = tables["subject_agg"]
    subject_fits = tables["subject_fits"]
    xticks, x_grid = XTICKS, X_GRID
    if True:  # body kept at the original loop indentation
        fit_color = FINGER_COLORS.get(finger, "black")
        f_group_agg = group_agg[group_agg["finger_condition"].astype(str) == finger].copy()
        f_group_fit = group_fits[group_fits["finger_condition"].astype(str) == finger].copy()
        f_subject_agg = subject_agg[subject_agg["finger_condition"].astype(str) == finger].copy()
        f_subject_fits = subject_fits[subject_fits["finger_condition"].astype(str) == finger].copy()

        # Draw every available subject fit so the background curves are filled in
        # for all panels, with workspace encoded by the requested N/L colors.
        for _, fit_row in f_subject_fits.iterrows():
            workspace = _workspace_from_row(fit_row)
            std = pd.to_numeric(pd.Series([fit_row.get("standard_value")]), errors="coerce").iloc[0]
            if not np.isfinite(std):
                subj = str(fit_row.get("subject_id", ""))
                subj_rows = f_subject_agg[f_subject_agg["subject_id"].astype(str) == subj]
                std = pd.to_numeric(subj_rows.get("standard_value", pd.Series(dtype=float)), errors="coerce").median()
            if not np.isfinite(std):
                continue
            y_grid = pf._fit_row_to_delta_predictions(fit_row, float(std), x_grid)
            if np.isfinite(y_grid).any():
                ax.plot(
                    x_grid,
                    y_grid,
                    color=_workspace_color(workspace),
                    alpha=0.25,
                    linewidth=0.9,
                    zorder=1,
                )

        if not f_group_agg.empty:
            f_group_agg = f_group_agg.sort_values("delta_comparison_minus_standard")
            lower = pd.to_numeric(f_group_agg["p_comparison_greater_ci95_lower"], errors="coerce")
            upper = pd.to_numeric(f_group_agg["p_comparison_greater_ci95_upper"], errors="coerce")
            y = pd.to_numeric(f_group_agg["p_comparison_greater"], errors="coerce")
            x = pd.to_numeric(f_group_agg["delta_comparison_minus_standard"], errors="coerce")
            yerr = np.vstack([y - lower, upper - y])
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.35", alpha=0.65, capsize=3, zorder=2)
            scatter = ax.scatter(
                x,
                y,
                c=x,
                cmap=STIFFNESS_CMAP,
                norm=norm,
                s=120,
                edgecolor="black",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )

        if not f_group_fit.empty:
            fit_row = f_group_fit.iloc[0]
            std = pd.to_numeric(pd.Series([fit_row.get("standard_value")]), errors="coerce").iloc[0]
            if np.isfinite(std):
                y_grid = pf._fit_row_to_delta_predictions(fit_row, float(std), x_grid)
                if np.isfinite(y_grid).any():
                    ax.plot(x_grid, y_grid, color=fit_color, linewidth=2.4, zorder=4)

            pse = pd.to_numeric(pd.Series([fit_row.get("pse_delta_comparison_minus_standard")]), errors="coerce").iloc[0]
            lo = pd.to_numeric(pd.Series([fit_row.get("pse_delta_ci95_lower")]), errors="coerce").iloc[0]
            hi = pd.to_numeric(pd.Series([fit_row.get("pse_delta_ci95_upper")]), errors="coerce").iloc[0]
            if np.isfinite(pse):
                ax.axvline(float(pse), color=fit_color, linestyle="--", linewidth=1.5, zorder=5)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax.axvspan(float(lo), float(hi), color=fit_color, alpha=0.10, zorder=0)
                    if lo <= pse <= hi:
                        ax.errorbar(
                            [float(pse)],
                            [0.5],
                            xerr=[[float(pse) - float(lo)], [float(hi) - float(pse)]],
                            fmt="o",
                            color=fit_color,
                            ecolor=fit_color,
                            elinewidth=1.6,
                            capsize=4,
                            markersize=5,
                            zorder=6,
                        )

        ax.axvline(0, color=STANDARD_COLOR, linestyle=":", linewidth=1.1, zorder=2)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, zorder=1)
        ax.set_title(f"Finger {finger}", fontsize=FS_TITLE, pad=14)  # "group pooled" goes in the caption
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(xticks)
        ax.set_xticklabels([_fmt_delta(float(v)) for v in xticks])
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.grid(True, color="0.80", linewidth=0.9)


def _legend_handles(ax, Line2D):
    # The CI-on-PSE entry is a real (empty) errorbar so the legend shows the same
    # dot-with-capped-bar mark that is drawn on the plots.
    ci_handle = ax.errorbar(
        [np.nan], [np.nan], xerr=[[1.0], [1.0]],
        fmt="o", color="black", ecolor="black", elinewidth=1.6, capsize=4, markersize=5,
        label="95% CI on PSE",
    )
    return [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#542788", markeredgecolor="black", label="Observed mean"),
        Line2D([0], [0], color="black", linewidth=2.4, label="Fit"),
        Line2D([0], [0], color=STANDARD_COLOR, linestyle=":", linewidth=1.4, label="standard (delta=0)"),
        ci_handle,
        Line2D([0], [0], color=N_COLOR, linewidth=2.4, alpha=0.90, label="N workspace"),
        Line2D([0], [0], color=L_COLOR, linewidth=2.4, alpha=0.90, label="L workspace"),
    ]


def make_figure(layout: str = "matrix", out_path: Path | None = None) -> Path:
    """Render the four finger panels as a 2x2 ``matrix`` or a 1x4 ``line``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid")
    except Exception:
        pass
    plt.rcParams.update({
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        # Render mathtext in the same face as the rest of the figure.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })

    pf = _load_analysis_module()

    tables = {
        "group_agg": _read_csv("psychometric_input_group_by_finger.csv"),
        "group_fits": _read_csv("pse_jnd_group_by_finger.csv"),
        "subject_agg": _read_csv("psychometric_input_by_subject_finger.csv"),
        "subject_fits": _read_csv("pse_jnd_by_subject_finger.csv"),
    }
    norm = Normalize(vmin=-6.0, vmax=6.0)

    if layout == "matrix":
        fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.8), dpi=160, sharex=True, sharey=True)
        xlabel_axes = [axes[1, 0], axes[1, 1]]
        ylabel_x, ylabel_y = 0.024, 0.60
        subplots = dict(left=0.095, right=0.855, top=0.952, bottom=0.26, wspace=0.12, hspace=0.30)
        legend_ncol = 3
        cax_x, cax_w = 0.87, 0.015
        default_out, backup = OUT_PATH, BACKUP_PATH
    elif layout == "line":
        fig, axes = plt.subplots(1, 4, figsize=(29.0, 7.0), dpi=160, sharex=True, sharey=True)
        xlabel_axes = list(axes.ravel())
        ylabel_x, ylabel_y = 0.013, 0.60
        subplots = dict(left=0.058, right=0.91, top=0.895, bottom=0.34, wspace=0.10)
        legend_ncol = 6
        cax_x, cax_w = 0.922, 0.009
        default_out, backup = LINE_OUT_PATH, LINE_BACKUP_PATH
    else:
        raise ValueError(f"unknown layout {layout!r}")

    for ax, finger in zip(axes.ravel(), FINGER_ORDER):
        _draw_panel(ax, finger, tables, pf, norm)
        ax.set_ylabel("")
    fig.text(
        ylabel_x,
        ylabel_y,
        "P(comparison > standard)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FS_LABEL,
    )
    for ax in xlabel_axes:
        ax.set_xlabel(DELTA_LABEL)

    fig.legend(
        handles=_legend_handles(axes.ravel()[0], Line2D),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=legend_ncol,
        fontsize=FS_LEGEND,
        frameon=True,
    )
    sm = plt.cm.ScalarMappable(cmap=STIFFNESS_CMAP, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(**subplots)
    # Colorbar spans exactly the height of the plot grid (bottom of the lowest
    # panel to the top of the highest one), not the whole figure.
    boxes = [ax.get_position() for ax in axes.ravel()]
    grid_y0, grid_y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    cax = fig.add_axes([cax_x, grid_y0, cax_w, grid_y1 - grid_y0])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(DELTA_LABEL, fontsize=FS_LABEL)
    cbar.set_ticks(CBAR_TICKS)
    cbar.set_ticklabels([f"{v:g}" for v in CBAR_TICKS])
    cbar.ax.tick_params(labelsize=FS_TICK)

    out = Path(out_path) if out_path is not None else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out == default_out and out.exists() and not backup.exists():
        shutil.copy2(out, backup)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = make_figure()
    print(path)
