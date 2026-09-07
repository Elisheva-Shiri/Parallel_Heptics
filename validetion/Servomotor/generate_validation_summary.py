"""Create manuscript-ready validation summary plots across all real runs.

This script is intentionally read-only with respect to the trusted data in
``responses/``.  It reads the three complete camera runs, reuses the corrected
angle-processing logic from ``analyze.py``, and writes combined figures/tables
to ``output/validation_summary``.

The command-reference angle is only a plotting convention:

    +/-1000 motor command ticks = +/-90 degrees

It is not an independent measured motor angle.  The measured values come from
the camera/string angle detection.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze import (  # noqa: E402  (flat module layout, see README)
    COMMAND_DEG_PER_TICK,
    add_block_relative_angle,
    add_trial_change,
    command_ticks_to_nominal_deg,
    load_log,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RESPONSES_DIR = PACKAGE_DIR / "responses"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output" / "validation_summary"

RUN_MARKERS = ["o", "s", "^", "D", "P", "X"]
RUN_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


@dataclass(frozen=True)
class RunFrames:
    """Processed frames for one validation run."""

    run_dir: Path
    run_id: str
    protocol: pd.DataFrame
    drift: pd.DataFrame


def _is_complete_camera_run(run_dir: Path) -> bool:
    """Return True for complete real-camera folders."""

    if not (run_dir / "protocol_log.csv").exists() and not (run_dir / "protocol_log.xlsx").exists():
        return False

    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return True

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return True

    camera = summary.get("camera", {})
    if camera.get("enabled") is False:
        return False
    if summary.get("n_angle_failures", 0) not in (0, None):
        return False
    return summary.get("n_rows", 1) != 0


def discover_runs(responses_dir: Path) -> list[Path]:
    """Find complete validation runs under responses/."""

    if not responses_dir.exists():
        raise FileNotFoundError(f"responses folder was not found: {responses_dir}")
    runs = [
        p
        for p in sorted(responses_dir.glob("motor_response_*"))
        if p.is_dir() and _is_complete_camera_run(p)
    ]
    if not runs:
        raise FileNotFoundError(f"No complete camera runs found in {responses_dir}")
    return runs


def _signed_target_label(target: pd.Series) -> pd.Series:
    return np.where(pd.to_numeric(target, errors="coerce") > 0, "+delta", "-delta")


def load_processed_run(run_dir: Path) -> RunFrames:
    """Load one run and construct protocol/drift samples for cross-run plots."""

    df = load_log(run_dir)
    df = add_block_relative_angle(df)
    df = add_trial_change(df)
    df["run_id"] = run_dir.name.replace("motor_response_", "")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df["nominal_command_deg"] = command_ticks_to_nominal_deg(df["target"])
    df["nominal_abs_command_deg"] = np.abs(df["nominal_command_deg"])

    protocol = df[(df["mode"] == "protocol") & (df["target"] != 0)].copy()
    protocol["measurement_type"] = "protocol"
    protocol["direction"] = _signed_target_label(protocol["target"])
    protocol["measured_deg"] = pd.to_numeric(protocol["angle_response_deg"], errors="coerce")
    protocol["measured_abs_deg"] = protocol["measured_deg"].abs()
    protocol = protocol.dropna(subset=["measured_deg", "delta"])

    drift = df[(df["mode"] == "drift") & (df["target"] != 0)].copy()
    drift["measurement_type"] = "drift"
    drift["direction"] = _signed_target_label(drift["target"])
    drift["measured_deg"] = pd.to_numeric(drift["angle_block_zeroed_display"], errors="coerce")
    drift["measured_abs_deg"] = drift["measured_deg"].abs()
    drift = drift.dropna(subset=["measured_deg", "delta"])

    return RunFrames(
        run_dir=run_dir,
        run_id=run_dir.name.replace("motor_response_", ""),
        protocol=protocol,
        drift=drift,
    )


def load_all_runs(run_dirs: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [load_processed_run(Path(p)) for p in run_dirs]
    protocol = pd.concat([f.protocol for f in frames], ignore_index=True)
    drift = pd.concat([f.drift for f in frames], ignore_index=True)
    return protocol, drift


def build_summary_tables(protocol: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create per-direction, per-run, and across-run summary tables."""

    signed = (
        protocol
        .groupby(["run_id", "delta", "direction"], observed=True)
        .agg(
            n=("measured_deg", "size"),
            mean_deg=("measured_deg", "mean"),
            std_deg=("measured_deg", "std"),
            nominal_command_deg=("nominal_command_deg", "mean"),
        )
        .reset_index()
    )

    rows: list[dict[str, float | str | int]] = []
    for (run_id, delta), sub in signed.groupby(["run_id", "delta"], observed=True):
        pos = sub[sub["direction"] == "+delta"]
        neg = sub[sub["direction"] == "-delta"]
        if pos.empty or neg.empty:
            continue
        pos_mean = float(pos["mean_deg"].iloc[0])
        neg_mean = float(neg["mean_deg"].iloc[0])
        pos_std = float(pos["std_deg"].iloc[0])
        neg_std = float(neg["std_deg"].iloc[0])
        nominal_abs = abs(float(pos["nominal_command_deg"].iloc[0]))
        mean_abs = (abs(pos_mean) + abs(neg_mean)) / 2.0
        repeatability_std = np.nanmean([pos_std, neg_std])
        rows.append({
            "run_id": run_id,
            "delta_ticks": int(delta),
            "nominal_abs_command_deg": nominal_abs,
            "mean_abs_response_deg": mean_abs,
            "gain_percent_of_nominal": (100.0 * mean_abs / nominal_abs) if nominal_abs else np.nan,
            "abs_error_deg": mean_abs - nominal_abs,
            "pos_mean_deg": pos_mean,
            "pos_std_deg": pos_std,
            "neg_mean_deg": neg_mean,
            "neg_std_deg": neg_std,
            "repeatability_std_deg": repeatability_std,
            "n_pos": int(pos["n"].iloc[0]),
            "n_neg": int(neg["n"].iloc[0]),
        })

    run_delta = pd.DataFrame(rows).sort_values(["delta_ticks", "run_id"]).reset_index(drop=True)
    across = (
        run_delta
        .groupby("delta_ticks", observed=True)
        .agg(
            n_runs=("run_id", "nunique"),
            nominal_abs_command_deg=("nominal_abs_command_deg", "mean"),
            mean_abs_response_deg=("mean_abs_response_deg", "mean"),
            across_run_sd_deg=("mean_abs_response_deg", "std"),
            gain_percent_of_nominal=("gain_percent_of_nominal", "mean"),
            abs_error_deg=("abs_error_deg", "mean"),
            pos_mean_deg=("pos_mean_deg", "mean"),
            pos_across_run_sd_deg=("pos_mean_deg", "std"),
            neg_mean_deg=("neg_mean_deg", "mean"),
            neg_across_run_sd_deg=("neg_mean_deg", "std"),
            repeatability_std_deg=("repeatability_std_deg", "mean"),
        )
        .reset_index()
    )
    return signed, run_delta, across


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    """Save a figure as high-resolution PNG and editable SVG."""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_summary_figure(run_delta: pd.DataFrame, across: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    ax_map, ax_signed, ax_repeat = axes

    # A) nominal command reference versus measured response.
    for i, (_run_id, sub) in enumerate(run_delta.groupby("run_id", observed=True)):
        marker = RUN_MARKERS[i % len(RUN_MARKERS)]
        color = RUN_COLORS[i % len(RUN_COLORS)]
        ax_map.plot(
            sub["nominal_abs_command_deg"],
            sub["mean_abs_response_deg"],
            marker=marker,
            lw=1.0,
            alpha=0.65,
            color=color,
            label=f"run {i + 1}",
        )

    ax_map.errorbar(
        across["nominal_abs_command_deg"],
        across["mean_abs_response_deg"],
        yerr=across["across_run_sd_deg"].fillna(0.0),
        fmt="o-",
        color="black",
        lw=2.0,
        capsize=3,
        label="mean +/- run SD",
    )
    lim = max(across["nominal_abs_command_deg"].max(), across["mean_abs_response_deg"].max()) * 1.05
    ax_map.plot([0, lim], [0, lim], "--", color="gray", lw=1.0, label="ideal y=x")
    ax_map.set_title("A. Command scale vs measured motion")
    ax_map.set_xlabel("nominal command angle [deg]\n(1000 ticks = 90 deg)")
    ax_map.set_ylabel("measured mean absolute angle [deg]")
    ax_map.set_xlim(left=0)
    ax_map.set_ylim(bottom=0)
    ax_map.legend(fontsize=8)

    # B) signed response: positive and negative directions should mirror.
    x = np.arange(len(across))
    labels = [str(int(v)) for v in across["delta_ticks"]]
    width = 0.36
    ax_signed.bar(
        x - width / 2,
        across["pos_mean_deg"],
        width,
        yerr=across["pos_across_run_sd_deg"].fillna(0.0),
        capsize=3,
        color="tab:blue",
        label="+delta",
    )
    ax_signed.bar(
        x + width / 2,
        across["neg_mean_deg"],
        width,
        yerr=across["neg_across_run_sd_deg"].fillna(0.0),
        capsize=3,
        color="tab:orange",
        label="-delta",
    )
    ax_signed.axhline(0, color="black", lw=0.8)
    ax_signed.set_title("B. Direction of measured response")
    ax_signed.set_xlabel("command delta [ticks]")
    ax_signed.set_ylabel("signed measured angle [deg]")
    ax_signed.set_xticks(x)
    ax_signed.set_xticklabels(labels, rotation=35, ha="right")
    ax_signed.legend()

    # C) within-run repeatability.
    for i, (_run_id, sub) in enumerate(run_delta.groupby("run_id", observed=True)):
        marker = RUN_MARKERS[i % len(RUN_MARKERS)]
        color = RUN_COLORS[i % len(RUN_COLORS)]
        ax_repeat.plot(
            sub["delta_ticks"],
            sub["repeatability_std_deg"],
            marker=marker,
            lw=1.0,
            alpha=0.65,
            color=color,
            label=f"run {i + 1}",
        )
    ax_repeat.plot(
        across["delta_ticks"],
        across["repeatability_std_deg"],
        "o-",
        color="black",
        lw=2.0,
        label="mean",
    )
    ax_repeat.set_title("C. Repeatability across trials")
    ax_repeat.set_xlabel("command delta [ticks]")
    ax_repeat.set_ylabel("within-run SD [deg]")
    ax_repeat.set_xscale("log")
    ax_repeat.set_xticks(across["delta_ticks"])
    ax_repeat.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_repeat.legend(fontsize=8)

    fig.suptitle(
        "Bowden-cable validation summary across 3 camera runs\n"
        "Measured string angle compared with nominal command scale; command scale is not a measured motor angle.",
        y=1.08,
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, out_dir / "validation_summary_figure")


def plot_protocol_vs_drift(protocol: pd.DataFrame, drift: pd.DataFrame, out_dir: Path) -> None:
    """Plot all camera samples: repeated protocol trials and drift samples."""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        protocol["nominal_command_deg"],
        protocol["measured_deg"],
        s=18,
        alpha=0.45,
        color="tab:blue",
        label="protocol trial samples",
    )
    if not drift.empty:
        ax.scatter(
            drift["nominal_command_deg"],
            drift["measured_deg"],
            s=16,
            alpha=0.35,
            marker="x",
            color="tab:gray",
            label="drift samples",
        )
    lim = max(
        np.nanmax(np.abs(protocol["nominal_command_deg"])),
        np.nanmax(np.abs(protocol["measured_deg"])),
        np.nanmax(np.abs(drift["measured_deg"])) if not drift.empty else 0.0,
    ) * 1.08
    ax.plot([-lim, lim], [-lim, lim], "--", color="black", lw=1.0, label="ideal y=x")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("All measured samples across validation runs")
    ax.set_xlabel("nominal command angle [deg]\n(1000 ticks = 90 deg; reference only)")
    ax.set_ylabel("measured camera/string angle [deg]")
    ax.legend(loc="upper left", fontsize=9)
    save_figure(fig, out_dir / "validation_protocol_vs_drift")


def plot_small_motion_zoom(protocol: pd.DataFrame, out_dir: Path) -> None:
    """Zoomed plot for the smallest command deltas."""

    small = protocol[protocol["delta"].isin([5, 10, 25])].copy()
    if small.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups = []
    values = []
    colors = []
    for delta in [5, 10, 25]:
        for direction, color in [("+delta", "tab:blue"), ("-delta", "tab:orange")]:
            vals = small[(small["delta"] == delta) & (small["direction"] == direction)]["measured_deg"].dropna()
            if vals.empty:
                continue
            groups.append(f"{direction}\n{delta} ticks")
            values.append(vals.to_numpy())
            colors.append(color)

    bp = ax.boxplot(values, tick_labels=groups, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for i, vals in enumerate(values, start=1):
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter(np.asarray(jitter) + i, vals, s=12, color="black", alpha=0.35, zorder=3)
        ax.text(i, np.nanmean(vals), f"{np.nanmean(vals):.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Small-command measured response across all runs")
    ax.set_ylabel("signed measured angle [deg]")
    ax.set_xlabel("direction and command delta")
    ax.set_ylim(-4.5, 4.5)
    fig.tight_layout()
    save_figure(fig, out_dir / "validation_small_motion_zoom")


def write_summary_text(across: pd.DataFrame, protocol: pd.DataFrame, out_dir: Path) -> None:
    """Write a short result paragraph that can be adapted for the manuscript."""

    n_runs = int(protocol["run_id"].nunique())
    n_protocol = int(len(protocol))
    deltas = ", ".join(str(int(d)) for d in sorted(protocol["delta"].dropna().unique()))
    row25 = across[across["delta_ticks"] == 25]
    row1000 = across[across["delta_ticks"] == 1000]

    small_sentence = ""
    if not row25.empty:
        small_sentence = (
            f"For 25 ticks, the nominal command scale is "
            f"{float(row25['nominal_abs_command_deg'].iloc[0]):.2f} deg and the measured "
            f"mean absolute response was {float(row25['mean_abs_response_deg'].iloc[0]):.2f} deg."
        )
    endpoint_sentence = ""
    if not row1000.empty:
        endpoint_sentence = (
            f"For the 1000-tick endpoint, the nominal command scale is "
            f"{float(row1000['nominal_abs_command_deg'].iloc[0]):.1f} deg and the measured "
            f"mean absolute response was {float(row1000['mean_abs_response_deg'].iloc[0]):.1f} deg."
        )

    text = f"""# Validation summary for manuscript insertion

These figures summarize {n_runs} complete real-camera validation runs and {n_protocol} non-zero protocol samples.
The tested command deltas were: {deltas} ticks.
For visualization, motor commands were mapped to a nominal command-angle scale using {COMMAND_DEG_PER_TICK:.3f} deg/tick, i.e. +/-1000 ticks = +/-90 deg. This command scale is a reference for comparison only; it is not a second measured angle.

{small_sentence}
{endpoint_sentence}

Suggested wording:

> The Bowden-cable transmission was validated by tracking the string/cable path from the motor side to the end of the mechanism with a camera. The measured string angle was compared with the commanded pulley displacement after mapping the motor command to a nominal angle scale (+/-1000 ticks = +/-90 deg). Across the three complete camera runs, the measured response followed the expected positive and negative command directions and showed repeatable trial-to-trial behavior. The command scale is used only as a reference; the physical measurement is the camera-derived string angle.
"""

    (out_dir / "validation_summary_text.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=DEFAULT_RESPONSES_DIR,
        help="Folder containing motor_response_* run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where combined plots and tables are written.",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        type=Path,
        action="append",
        help="Specific run folder to include. Can be repeated. Defaults to auto-discovery.",
    )
    args = parser.parse_args()

    run_dirs = args.runs if args.runs else discover_runs(args.responses_dir)
    protocol, drift = load_all_runs(run_dirs)
    signed, run_delta, across = build_summary_tables(protocol)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol.to_csv(args.output_dir / "validation_protocol_samples.csv", index=False)
    drift.to_csv(args.output_dir / "validation_drift_samples.csv", index=False)
    signed.to_csv(args.output_dir / "validation_signed_summary_by_run.csv", index=False)
    run_delta.to_csv(args.output_dir / "validation_run_delta_summary.csv", index=False)
    across.to_csv(args.output_dir / "validation_across_run_summary.csv", index=False)

    plot_summary_figure(run_delta, across, args.output_dir)
    plot_protocol_vs_drift(protocol, drift, args.output_dir)
    plot_small_motion_zoom(protocol, args.output_dir)
    write_summary_text(across, protocol, args.output_dir)

    print(f"Included {len(run_dirs)} runs:")
    for run in run_dirs:
        print(f"  - {run}")
    print(f"Wrote summary plots/tables to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
