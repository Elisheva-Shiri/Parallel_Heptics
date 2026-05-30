"""Generate one integrated summary graph per subset with finger labels above.

This corrected export creates ONE graph for each requested subset:
- all participants
- cohorts: N_E, N_P, L_E, L_P
- protocols 1..4
- each participant

Each graph keeps the original integrated visual design:
- x-axis: continuous trial/repetition time
- y-axis: two stacked panels (2 rows x 1 column):
  - X-relative location or X velocity
  - Y-relative location or Y velocity
- background: response correctness/success
- arrows: local XY movement direction, colored by stiffness
- top annotations: finger appearance blocks, marked by letters I/M/R/P above
  the timeline with a down-facing bracket.

Outputs are written under:
- analysis/summery/results/distances (X/Y relative location)
- analysis/summery/results/velocity (X/Y velocity)

Each output folder also receives the aggregated plot data CSV used to draw its
PNG files.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def find_repo_root(start: Path | None = None) -> Path:
    start = Path(__file__).resolve() if start is None else Path(start).resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "analysis").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root")


REPO_ROOT = find_repo_root()
SUMMARY_DIR = REPO_ROOT / "analysis" / "summery"
RESULTS_DIR = SUMMARY_DIR / "results"
DISTANCE_DIR = RESULTS_DIR / "distances"
VELOCITY_DIR = RESULTS_DIR / "velocity"
for output_dir in [DISTANCE_DIR, VELOCITY_DIR]:
    output_dir.mkdir(parents=True, exist_ok=True)

PSYCHOPHYSICS_TRIALS_CSV = (
    REPO_ROOT / "analysis/psychophysics_analysis/results/combined_success_trials.csv"
)
XY_TRAJECTORY_BINS_CSV = (
    REPO_ROOT / "analysis/probing_analysis/results/xy_probing_skin_stretch_trajectory_bins.csv"
)
PROTOCOL_BY_SEQUENCE = {
    "I-M-P-R": 1,
    "M-R-I-P": 2,
    "R-P-M-I": 3,
    "P-I-R-M": 4,
}

MOTION_WINDOW_BINS = 3
MAX_ARROWS = 850
ARROW_LENGTH_X_TRIALS = 2.2
ARROW_LENGTH_Y_DISTANCE = 70.0
ARROW_LENGTH_Y_VELOCITY = 90.0

CORRECT_COLOR = "#7db7ff"
INCORRECT_COLOR = "#ff8f8f"
CORRECT_RGB = np.array([0x7D, 0xB7, 0xFF], dtype=float) / 255.0
INCORRECT_RGB = np.array([0xFF, 0x8F, 0x8F], dtype=float) / 255.0


def cohort_from_subject(subject_id: str) -> str:
    parts = str(subject_id).split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "unknown"


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )


def success_color(success_rate: float) -> tuple[float, float, float] | str:
    """Blue for success, red for failure; aggregate panels are blended."""
    if pd.isna(success_rate):
        return "0.9"
    rate = float(np.clip(success_rate, 0.0, 1.0))
    rgb = INCORRECT_RGB * (1.0 - rate) + CORRECT_RGB * rate
    return tuple(rgb)


def fixed_appearance_ranges(max_trial: int) -> list[tuple[int, int]]:
    """Return the intended appearance ranges.

    Experiment sessions have 268 trials:
    - four short first appearances in trials 1-12, 3 comparisons each
    - four long appearances of 64 comparisons each

    Protocol-only sessions have 256 trials:
    - four long appearances of 64 comparisons each
    """
    if max_trial >= 268:
        return [
            (1, 3),
            (4, 6),
            (7, 9),
            (10, 12),
            (13, 76),
            (77, 140),
            (141, 204),
            (205, 268),
        ]
    return [(1, 64), (65, 128), (129, 192), (193, 256)]


def label_for_trial_range(trials: pd.DataFrame, start: int, end: int) -> str:
    part = trials.loc[
        (trials["global_trial_order"] >= start)
        & (trials["global_trial_order"] <= end)
    ]
    if part.empty:
        return ""
    counts = (
        part[["subject_id", "global_trial_order", "finger_condition"]]
        .drop_duplicates()
        .groupby("finger_condition")
        .size()
    )
    max_n = counts.max()
    top = sorted(str(idx) for idx, n in counts.items() if n == max_n)
    return "/".join(top)


def derive_subject_protocols(trials: pd.DataFrame) -> pd.DataFrame:
    """Derive protocol from the four long 64-trial finger blocks.

    For 268-trial experiment sessions, protocol is trials 13-268:
    1=I-M-P-R, 2=M-R-I-P, 3=R-P-M-I, 4=P-I-R-M.
    The first 12 trials are still annotated separately as I/M/R/P.
    """
    rows = []
    for subject_id, subject_trials in trials.groupby("subject_id", sort=True):
        max_trial = int(subject_trials["global_trial_order"].max())
        ranges = fixed_appearance_ranges(max_trial)
        long_ranges = ranges[4:] if max_trial >= 268 else ranges
        labels = [
            label_for_trial_range(subject_trials, start, end)
            for start, end in long_ranges
        ]
        finger_sequence = "-".join(labels)
        rows.append(
            {
                "subject_id": subject_id,
                "finger_sequence": finger_sequence,
                "protocol_number": PROTOCOL_BY_SEQUENCE.get(finger_sequence),
            }
        )
    return pd.DataFrame(rows)


def load_trial_metadata() -> pd.DataFrame:
    wanted = [
        "subject_id",
        "global_trial_order",
        "finger_condition",
        "comparison_value",
        "standard_value",
        "correct_response",
        "success_label",
        "experiment_group",
        "protocol_factor",
    ]
    available = pd.read_csv(PSYCHOPHYSICS_TRIALS_CSV, nrows=0).columns.tolist()
    trials = pd.read_csv(
        PSYCHOPHYSICS_TRIALS_CSV,
        usecols=[col for col in wanted if col in available],
    )
    trials["subject_id"] = trials["subject_id"].astype(str)
    trials["cohort"] = trials["subject_id"].map(cohort_from_subject)
    trials["global_trial_order"] = pd.to_numeric(
        trials["global_trial_order"], errors="coerce"
    )
    trials["correct_response"] = pd.to_numeric(
        trials["correct_response"], errors="coerce"
    )
    return trials.merge(derive_subject_protocols(trials), on="subject_id", how="left")


def load_trajectory_with_metadata(trials: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "subject_id",
        "global_trial_order",
        "trial_index_raw",
        "trajectory_time_bin",
        "time_fraction",
        "object_dx_from_center_px",
        "object_dy_from_center_px",
        "center_distance_px",
        "reaction_time",
        "skin_stretch_gain_mm_per_m_or_condition",
        "skin_stretch_gain_mm_per_m",
        "finger_condition",
    ]
    available = pd.read_csv(XY_TRAJECTORY_BINS_CSV, nrows=0).columns.tolist()
    usecols = [col for col in wanted if col in available]
    pieces = []
    for chunk in pd.read_csv(XY_TRAJECTORY_BINS_CSV, usecols=usecols, chunksize=300_000):
        pieces.append(chunk)
    trajectory = pd.concat(pieces, ignore_index=True)
    trajectory["subject_id"] = trajectory["subject_id"].astype(str)

    for col in [
        "global_trial_order",
        "trial_index_raw",
        "trajectory_time_bin",
        "time_fraction",
        "object_dx_from_center_px",
        "object_dy_from_center_px",
        "center_distance_px",
        "reaction_time",
        "skin_stretch_gain_mm_per_m_or_condition",
        "skin_stretch_gain_mm_per_m",
    ]:
        if col in trajectory.columns:
            trajectory[col] = pd.to_numeric(trajectory[col], errors="coerce")

    metadata = trials[
        [
            "subject_id",
            "global_trial_order",
            "correct_response",
            "cohort",
            "protocol_number",
            "finger_sequence",
        ]
    ].drop_duplicates(["subject_id", "global_trial_order"])
    trajectory = trajectory.merge(
        metadata, on=["subject_id", "global_trial_order"], how="left"
    )

    trajectory["continuous_trial_time"] = (
        trajectory["global_trial_order"] - 1.0
    ) + trajectory["time_fraction"].clip(0, 1)
    trajectory["distance_from_center_px"] = trajectory["center_distance_px"].abs()
    trajectory = add_velocity_columns(trajectory)
    if "skin_stretch_gain_mm_per_m_or_condition" in trajectory.columns:
        trajectory["local_stiffness"] = trajectory[
            "skin_stretch_gain_mm_per_m_or_condition"
        ]
    else:
        trajectory["local_stiffness"] = trajectory["skin_stretch_gain_mm_per_m"]
    trajectory["local_stiffness"] = pd.to_numeric(
        trajectory["local_stiffness"], errors="coerce"
    )
    return trajectory


def add_velocity_columns(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Derive within-trial XY velocity from binned object-center positions.

    `time_fraction` is normalized within each trial. When `reaction_time` is
    present, velocity is reported in px/s; otherwise the fallback unit is
    px/trial-fraction. The current input CSV includes `reaction_time`, so the
    normal output column is in px/s.
    """
    trajectory = trajectory.sort_values(
        ["subject_id", "global_trial_order", "trajectory_time_bin"]
    ).copy()
    group = trajectory.groupby(["subject_id", "global_trial_order"], sort=False)

    if "reaction_time" in trajectory.columns:
        trajectory["time_within_trial_s"] = (
            trajectory["time_fraction"] * trajectory["reaction_time"]
        )
        time_column = "time_within_trial_s"
    else:
        trajectory["time_within_trial_fraction"] = trajectory["time_fraction"]
        time_column = "time_within_trial_fraction"

    dt = group[time_column].diff()
    dx = group["object_dx_from_center_px"].diff()
    dy = group["object_dy_from_center_px"].diff()
    radial_distance_delta = group["distance_from_center_px"].diff()

    valid_dt = dt.where(dt > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        trajectory["vx_px_s"] = dx / valid_dt
        trajectory["vy_px_s"] = dy / valid_dt
        trajectory["velocity_px_s"] = np.hypot(dx, dy) / valid_dt
        trajectory["radial_velocity_px_s"] = radial_distance_delta / valid_dt
    trajectory[
        ["vx_px_s", "vy_px_s", "velocity_px_s", "radial_velocity_px_s"]
    ] = trajectory[
        ["vx_px_s", "vy_px_s", "velocity_px_s", "radial_velocity_px_s"]
    ].replace([np.inf, -np.inf], np.nan)
    return trajectory


def aggregate_graph_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = (
        data.groupby(["global_trial_order", "trajectory_time_bin"], as_index=False)
        .agg(
            time_fraction=("time_fraction", "mean"),
            object_dx_from_center_px=("object_dx_from_center_px", "mean"),
            object_dy_from_center_px=("object_dy_from_center_px", "mean"),
            distance_from_center_px=("distance_from_center_px", "mean"),
            vx_px_s=("vx_px_s", "mean"),
            vy_px_s=("vy_px_s", "mean"),
            velocity_px_s=("velocity_px_s", "mean"),
            radial_velocity_px_s=("radial_velocity_px_s", "mean"),
            local_stiffness=("local_stiffness", "mean"),
            correct_response=("correct_response", "mean"),
            n_samples=("subject_id", "size"),
        )
        .sort_values(["global_trial_order", "trajectory_time_bin"])
    )
    panel["continuous_trial_time"] = (
        panel["global_trial_order"] - 1.0
    ) + panel["time_fraction"].clip(0, 1)

    group = panel.groupby("global_trial_order", sort=False)
    panel["motion_dx_px"] = group["object_dx_from_center_px"].shift(
        -MOTION_WINDOW_BINS
    ) - group["object_dx_from_center_px"].shift(MOTION_WINDOW_BINS)
    panel["motion_dy_px"] = group["object_dy_from_center_px"].shift(
        -MOTION_WINDOW_BINS
    ) - group["object_dy_from_center_px"].shift(MOTION_WINDOW_BINS)
    panel["motion_norm_px"] = np.hypot(panel["motion_dx_px"], panel["motion_dy_px"])
    panel["motion_angle_rad"] = np.arctan2(
        panel["motion_dy_px"], panel["motion_dx_px"]
    )

    trial_success = (
        data.groupby("global_trial_order", as_index=False)
        .agg(correct_response=("correct_response", "mean"))
        .sort_values("global_trial_order")
    )
    trial_success["x_start"] = trial_success["global_trial_order"] - 1.0
    trial_success["x_end"] = trial_success["global_trial_order"]
    return panel, trial_success


def finger_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Return fixed finger appearance segments, preserving the first 12 trials.

    This intentionally does not merge adjacent same-finger labels. Example:
    if trial 10-12 is P and trial 13-76 is also P, these remain two separate
    appearances because the first belongs to the first 12 comparisons.
    """
    max_trial = int(data["global_trial_order"].max())
    rows = []
    for start, end in fixed_appearance_ranges(max_trial):
        clipped_end = min(end, max_trial)
        if start > max_trial:
            continue
        rows.append(
            {
                "finger_label": label_for_trial_range(data, start, clipped_end),
                "start_trial": start,
                "end_trial": clipped_end,
                "n_trials": clipped_end - start + 1,
            }
        )
    segments = pd.DataFrame(rows)
    segments["x_start"] = segments["start_trial"] - 1.0
    segments["x_end"] = segments["end_trial"]
    segments["x_mid"] = (segments["x_start"] + segments["x_end"]) / 2.0
    return segments


def select_arrow_rows(
    panel: pd.DataFrame, y_column: str, arrow_length_y: float
) -> pd.DataFrame:
    arrows = panel.dropna(
        subset=[
            "continuous_trial_time",
            y_column,
            "motion_angle_rad",
            "motion_norm_px",
            "local_stiffness",
        ]
    ).copy()
    arrows = arrows.loc[arrows["motion_norm_px"] > 0]
    if arrows.empty:
        return arrows
    stride = max(1, math.ceil(len(arrows) / MAX_ARROWS))
    arrows = arrows.iloc[::stride].copy()
    arrows["arrow_u"] = np.cos(arrows["motion_angle_rad"]) * ARROW_LENGTH_X_TRIALS
    arrows["arrow_v"] = np.sin(arrows["motion_angle_rad"]) * arrow_length_y
    return arrows


def draw_finger_appearance_brackets(ax: plt.Axes, segments: pd.DataFrame) -> None:
    """Draw down-facing brackets and labels above the axes."""
    trans = ax.get_xaxis_transform()
    y_top = 1.035
    y_tick = 1.005
    for seg in segments.itertuples(index=False):
        ax.plot(
            [seg.x_start, seg.x_end],
            [y_top, y_top],
            color="0.15",
            lw=1.1,
            transform=trans,
            clip_on=False,
        )
        ax.plot(
            [seg.x_start, seg.x_start],
            [y_top, y_tick],
            color="0.15",
            lw=1.1,
            transform=trans,
            clip_on=False,
        )
        ax.plot(
            [seg.x_end, seg.x_end],
            [y_top, y_tick],
            color="0.15",
            lw=1.1,
            transform=trans,
            clip_on=False,
        )
        ax.text(
            seg.x_mid,
            y_top + 0.008,
            str(seg.finger_label),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            transform=trans,
            clip_on=False,
        )


def plot_annotated_graph(
    data: pd.DataFrame,
    subset_label: str,
    output_path: Path,
    max_trial: int,
    y_specs: list[dict[str, object]],
    metric_label: str,
) -> tuple[bool, pd.DataFrame]:
    if data.empty:
        print(f"SKIP empty graph: {subset_label}")
        return False, pd.DataFrame()

    panel, trial_success = aggregate_graph_data(data)
    segments = finger_segments(data)

    fig, axes = plt.subplots(
        nrows=len(y_specs),
        ncols=1,
        figsize=(24, 11.6),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    q_for_colorbar = None
    arrow_count_by_panel: list[int] = []

    for axis_index, (ax, spec) in enumerate(zip(axes, y_specs)):
        y_column = str(spec["column"])
        y_label = str(spec["label"])
        trend_column = str(spec["trend_column"])
        trend_label = str(spec["trend_label"])
        arrow_length_y = float(spec["arrow_length_y"])
        arrows = select_arrow_rows(
            panel, y_column=y_column, arrow_length_y=arrow_length_y
        )
        arrow_count_by_panel.append(len(arrows))

        for seg in trial_success.itertuples(index=False):
            ax.axvspan(
                seg.x_start,
                seg.x_end,
                color=success_color(seg.correct_response),
                alpha=0.24,
                lw=0,
                zorder=0,
            )

        for _, trial_panel in panel.groupby("global_trial_order", sort=False):
            ax.plot(
                trial_panel["continuous_trial_time"],
                trial_panel[y_column],
                color="black",
                lw=0.5,
                alpha=0.70,
                zorder=2,
            )

        trend = panel[["continuous_trial_time", y_column]].copy()
        trend[trend_column] = trend[y_column].rolling(
            121, center=True, min_periods=10
        ).median()
        ax.plot(
            trend["continuous_trial_time"],
            trend[trend_column],
            color="#111111",
            lw=1.5,
            alpha=0.85,
            zorder=3,
            label=trend_label,
        )
        panel[trend_column] = trend[trend_column].to_numpy()

        if not arrows.empty:
            stiffness_values = arrows["local_stiffness"].astype(float)
            vmin = float(np.nanmin(stiffness_values))
            vmax = float(np.nanmax(stiffness_values))
            if vmin == vmax:
                vmax = vmin + 1.0
            q_for_colorbar = ax.quiver(
                arrows["continuous_trial_time"],
                arrows[y_column],
                arrows["arrow_u"],
                arrows["arrow_v"],
                stiffness_values,
                cmap=mpl.cm.viridis,
                norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0014,
                headwidth=3.0,
                headlength=4.0,
                headaxislength=3.5,
                alpha=0.95,
                zorder=5,
            )

        ax.axhline(0, color="0.25", lw=0.7, alpha=0.45, zorder=1)
        ax.set_xlim(0, max_trial)
        ax.set_ylabel(y_label)
        ax.grid(True, axis="y", alpha=0.22)
        ax.legend(loc="upper right", frameon=True)
        ax.text(
            0.01,
            0.96,
            f"panel={axis_index + 1}, arrows={len(arrows)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, lw=0
            ),
        )

    if q_for_colorbar is not None:
        cbar = fig.colorbar(q_for_colorbar, ax=axes.tolist(), pad=0.01)
        cbar.set_label("Object stiffness / skin-stretch gain")

    draw_finger_appearance_brackets(axes[0], segments)

    n_subjects = data["subject_id"].nunique()
    n_trials = data[["subject_id", "global_trial_order"]].drop_duplicates().shape[0]
    axes[0].set_title(
        f"{subset_label} - integrated {metric_label} summary with finger appearances above\n"
        "Background: blue=correct/high success, red=incorrect/low success; "
        "arrows: local XY direction colored by stiffness; panels are X then Y",
        pad=38,
    )
    axes[-1].set_xlabel("Experiment time (continuous trial index)")
    ticks = list(range(0, max_trial + 1, 20))
    if max_trial not in ticks:
        ticks.append(max_trial)
    axes[-1].set_xticks(ticks)
    axes[0].text(
        0.01,
        0.86,
        (
            f"subjects={n_subjects}, trials={n_trials}, "
            f"arrows={','.join(str(n) for n in arrow_count_by_panel)}"
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, lw=0),
    )

    axes[0].legend(
        handles=[
            Patch(facecolor=CORRECT_COLOR, alpha=0.35, label="correct / high success"),
            Patch(facecolor=INCORRECT_COLOR, alpha=0.35, label="incorrect / low success"),
        ],
        loc="upper right",
        frameon=True,
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path.relative_to(REPO_ROOT)}")

    used_data = panel.copy()
    used_data.insert(0, "output_file", str(output_path.relative_to(REPO_ROOT)))
    used_data.insert(1, "subset", subset_label)
    used_data.insert(2, "metric", metric_label)
    return True, used_data


def build_subset_targets(data: pd.DataFrame) -> list[tuple[str, str, pd.Series, int]]:
    targets: list[tuple[str, str, pd.Series, int]] = [
        (
            "all_participants.png",
            "All participants",
            pd.Series(True, index=data.index),
            int(data["global_trial_order"].max()),
        )
    ]

    for cohort in ["N_E", "N_P", "L_E", "L_P"]:
        mask = data["cohort"] == cohort
        if mask.any():
            targets.append(
                (
                    f"cohort_{cohort}.png",
                    f"Cohort {cohort}",
                    mask,
                    int(data.loc[mask, "global_trial_order"].max()),
                )
            )

    for protocol in [1, 2, 3, 4]:
        mask = data["protocol_number"] == protocol
        if mask.any():
            subjects = ", ".join(sorted(data.loc[mask, "subject_id"].unique()))
            targets.append(
                (
                    f"protocol_{protocol}.png",
                    f"Protocol {protocol} ({subjects})",
                    mask,
                    int(data.loc[mask, "global_trial_order"].max()),
                )
            )

    max_trial_by_subject = data.groupby("subject_id")["global_trial_order"].max()
    for subject_id in sorted(data["subject_id"].unique()):
        mask = data["subject_id"] == subject_id
        targets.append(
            (
                f"participant_{safe_name(subject_id)}.png",
                f"Participant {subject_id}",
                mask,
                int(max_trial_by_subject.loc[subject_id]),
            )
        )
    return targets


def main() -> None:
    print("Loading trial metadata...")
    trials = load_trial_metadata()
    print(f"Loaded {len(trials):,} trial rows")

    print("Loading trajectory bins...")
    data = load_trajectory_with_metadata(trials)
    print(
        f"Loaded {len(data):,} trajectory rows for "
        f"{data['subject_id'].nunique()} participants"
    )

    metric_configs = [
        {
            "name": "distance",
            "label": "relative location",
            "output_dir": DISTANCE_DIR,
            "y_specs": [
                {
                    "column": "object_dx_from_center_px",
                    "label": "X relative to center (px)",
                    "trend_column": "x_location_trend",
                    "trend_label": "rolling median X location",
                    "arrow_length_y": ARROW_LENGTH_Y_DISTANCE,
                },
                {
                    "column": "object_dy_from_center_px",
                    "label": "Y relative to center (px)",
                    "trend_column": "y_location_trend",
                    "trend_label": "rolling median Y location",
                    "arrow_length_y": ARROW_LENGTH_Y_DISTANCE,
                },
            ],
            "plot_data_csv": "distance_plot_data.csv",
            "manifest_csv": "distance_png_manifest.csv",
        },
        {
            "name": "velocity",
            "label": "velocity",
            "output_dir": VELOCITY_DIR,
            "y_specs": [
                {
                    "column": "vx_px_s",
                    "label": "X velocity (px/s)",
                    "trend_column": "x_velocity_trend",
                    "trend_label": "rolling median X velocity",
                    "arrow_length_y": ARROW_LENGTH_Y_VELOCITY,
                },
                {
                    "column": "vy_px_s",
                    "label": "Y velocity (px/s)",
                    "trend_column": "y_velocity_trend",
                    "trend_label": "rolling median Y velocity",
                    "arrow_length_y": ARROW_LENGTH_Y_VELOCITY,
                },
            ],
            "plot_data_csv": "velocity_plot_data.csv",
            "manifest_csv": "velocity_png_manifest.csv",
        },
    ]

    targets = build_subset_targets(data)
    for config in metric_configs:
        rows = []
        used_data_frames = []
        output_dir = config["output_dir"]
        for filename, subset_label, mask, max_trial in targets:
            subset = data.loc[mask].copy()
            output_path = output_dir / filename
            saved, used_data = plot_annotated_graph(
                subset,
                subset_label=subset_label,
                output_path=output_path,
                max_trial=max_trial,
                y_specs=config["y_specs"],
                metric_label=config["label"],
            )
            if saved:
                used_data_frames.append(used_data)
                segs = finger_segments(subset)
                rows.append(
                    {
                        "file": str(output_path.relative_to(REPO_ROOT)),
                        "subset": subset_label,
                        "metric": config["label"],
                        "n_subjects": int(subset["subject_id"].nunique()),
                        "n_rows": int(len(subset)),
                        "max_trial": int(max_trial),
                        "finger_appearance_labels": " | ".join(
                            f"{r.finger_label}:{int(r.start_trial)}-{int(r.end_trial)}"
                            for r in segs.itertuples(index=False)
                        ),
                    }
                )

        manifest = pd.DataFrame(rows)
        manifest_path = output_dir / config["manifest_csv"]
        manifest.to_csv(manifest_path, index=False)
        print(f"saved {manifest_path.relative_to(REPO_ROOT)}")

        plot_data = (
            pd.concat(used_data_frames, ignore_index=True)
            if used_data_frames
            else pd.DataFrame()
        )
        plot_data_path = output_dir / config["plot_data_csv"]
        plot_data.to_csv(plot_data_path, index=False)
        print(f"saved {plot_data_path.relative_to(REPO_ROOT)}")
        print(f"Done. Saved {len(manifest)} {config['label']} PNG images.")


if __name__ == "__main__":
    main()
