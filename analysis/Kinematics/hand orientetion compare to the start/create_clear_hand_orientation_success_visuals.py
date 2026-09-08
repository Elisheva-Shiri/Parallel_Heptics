
"""
Create clearer hand-orientation visuals.

Design choices based on user feedback:
1. Orientation y-axis is degrees away from baseline, so it starts at 0.
2. Success/failure is shown separately from orientation degrees.
3. Relationship between orientation change and success is shown with simple bar/scatter plots.

Input is produced by compare_hand_orientation_to_start.py using:
- baseline = circular average of first 10 values
- changed = absolute change > 10 degrees
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {"L_E": "#ff69b4", "N_E": "#7b2cbf"}  # pink, purple
PLANES = ["XY", "YZ", "ZX"]
GROUPS = ["L_E", "N_E"]


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "experiment_group",
        "subject_id",
        "finger_condition",
        "run_dir",
        "trial_index_raw",
        "pair_number",
        "stiffness_order_in_trial",
        "stiffness_segment_id",
        "plane",
    ]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort").copy()
    order_cols = ["subject_id", "finger_condition", "plane"]
    df["sequence_index"] = df.groupby(order_cols, dropna=False).cumcount() + 1
    df["sequence_n"] = df.groupby(order_cols, dropna=False)["sequence_index"].transform("max")
    df["sequence_percent"] = np.where(
        df["sequence_n"] > 1,
        100 * (df["sequence_index"] - 1) / (df["sequence_n"] - 1),
        0,
    )
    df["sequence_bin"] = pd.cut(
        df["sequence_percent"],
        bins=np.arange(0, 110, 10),
        include_lowest=True,
        labels=[f"{i}-{i+10}%" for i in range(0, 100, 10)],
    )
    labels = df.get("success_label", pd.Series(index=df.index, dtype=object)).astype(str).str.lower().str.strip()
    df["success_binary"] = np.select(
        [labels.eq("success"), labels.eq("failure")],
        [1.0, 0.0],
        default=np.nan,
    )
    return df


def bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["experiment_group", "plane", "sequence_bin"], observed=True)
        .agg(
            x=("sequence_percent", "mean"),
            n=("abs_change_from_start_deg", "size"),
            mean_abs_change_deg=("abs_change_from_start_deg", "mean"),
            sem_abs_change_deg=("abs_change_from_start_deg", sem),
            median_abs_change_deg=("abs_change_from_start_deg", "median"),
            percent_changed_gt_threshold=("changed_from_start", lambda s: 100 * s.mean()),
            success_rate_percent=("success_binary", lambda s: 100 * pd.to_numeric(s, errors="coerce").mean()),
            n_success=("success_binary", lambda s: int((s == 1).sum())),
            n_failure=("success_binary", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    out["majority_success"] = out["n_success"] > out["n_failure"]
    out["majority_failure"] = out["n_failure"] > out["n_success"]
    return out


def save_orientation_degrees_plot(summary: pd.DataFrame, outdir: Path, threshold: float) -> None:
    fig, axes = plt.subplots(len(PLANES), 1, figsize=(12, 3.6 * len(PLANES)), sharex=True)
    if len(PLANES) == 1:
        axes = [axes]
    y_max = max(15, float(summary["mean_abs_change_deg"].max() + summary["sem_abs_change_deg"].fillna(0).max() + 3))
    for ax, plane in zip(axes, PLANES):
        sub = summary[summary["plane"].eq(plane)]
        for group in GROUPS:
            g = sub[sub["experiment_group"].eq(group)].dropna(subset=["x"])
            if g.empty:
                continue
            x = g["x"].to_numpy(float)
            y = g["mean_abs_change_deg"].to_numpy(float)
            e = g["sem_abs_change_deg"].fillna(0).to_numpy(float)
            ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[group], label=group)
            ax.fill_between(x, y - e, y + e, color=COLORS[group], alpha=0.18)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"change threshold = {threshold:g}?")
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Degrees from baseline\n(absolute)")
        ax.set_title(f"{plane}: hand orientation deviation from start baseline")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Trial order within participant/finger (% from start to end)")
    fig.suptitle("Hand orientation movement: y-axis starts at 0? baseline", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "CLEAR_1_orientation_degrees_from_baseline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_success_failure_plot(summary: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(len(PLANES), 1, figsize=(12, 3.5 * len(PLANES)), sharex=True)
    if len(PLANES) == 1:
        axes = [axes]
    for ax, plane in zip(axes, PLANES):
        sub = summary[summary["plane"].eq(plane)]
        for group in GROUPS:
            g = sub[sub["experiment_group"].eq(group)].dropna(subset=["x"])
            if g.empty:
                continue
            ax.plot(
                g["x"],
                g["success_rate_percent"],
                marker="o",
                linewidth=2.5,
                color=COLORS[group],
                label=f"{group} success %",
            )
            # Mark bins where group majority is failure.
            fail_bins = g[g["majority_failure"]]
            if not fail_bins.empty:
                ax.scatter(
                    fail_bins["x"],
                    fail_bins["success_rate_percent"],
                    marker="x",
                    s=90,
                    color=COLORS[group],
                    linewidths=2,
                    label=f"{group} majority failure bin",
                )
        ax.axhline(50, color="black", linestyle="--", linewidth=1.2, label="50% success/failure boundary")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Success (%)")
        ax.set_title(f"{plane}: group success/failure over trials")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Trial order within participant/finger (% from start to end)")
    fig.tight_layout()
    fig.savefig(outdir / "CLEAR_2_success_failure_percent_over_trials.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_changed_percent_plot(summary: pd.DataFrame, outdir: Path, threshold: float) -> None:
    fig, axes = plt.subplots(len(PLANES), 1, figsize=(12, 3.5 * len(PLANES)), sharex=True)
    if len(PLANES) == 1:
        axes = [axes]
    for ax, plane in zip(axes, PLANES):
        sub = summary[summary["plane"].eq(plane)]
        for group in GROUPS:
            g = sub[sub["experiment_group"].eq(group)].dropna(subset=["x"])
            if g.empty:
                continue
            ax.plot(
                g["x"],
                g["percent_changed_gt_threshold"],
                marker="o",
                linewidth=2.5,
                color=COLORS[group],
                label=f"{group}",
            )
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"% trials changed\n> {threshold:g}?")
        ax.set_title(f"{plane}: how often orientation changed more than {threshold:g}?")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Trial order within participant/finger (% from start to end)")
    fig.tight_layout()
    fig.savefig(outdir / "CLEAR_3_percent_changed_over_trials.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_success_by_changed_bar(df: pd.DataFrame, outdir: Path, threshold: float) -> None:
    valid = df.dropna(subset=["success_binary"]).copy()
    valid["orientation_status"] = np.where(
        valid["changed_from_start"], f"changed > {threshold:g} deg", f"same <= {threshold:g} deg"
    )
    status_order = [f"same <= {threshold:g} deg", f"changed > {threshold:g} deg"]
    summary = (
        valid.groupby(["experiment_group", "plane", "orientation_status"])
        .agg(
            success_rate_percent=("success_binary", lambda s: 100 * s.mean()),
            n=("success_binary", "size"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "CLEAR_success_rate_same_vs_changed.csv", index=False)

    fig, axes = plt.subplots(1, len(PLANES), figsize=(5.2 * len(PLANES), 4.6), sharey=True)
    if len(PLANES) == 1:
        axes = [axes]
    width = 0.18
    xbase = np.arange(len(status_order))
    offsets = {"L_E": -width / 1.4, "N_E": width / 1.4}
    for ax, plane in zip(axes, PLANES):
        for group in GROUPS:
            vals = []
            ns = []
            for status in status_order:
                row = summary[(summary["plane"].eq(plane)) & (summary["experiment_group"].eq(group)) & (summary["orientation_status"].eq(status))]
                vals.append(float(row["success_rate_percent"].iloc[0]) if not row.empty else np.nan)
                ns.append(int(row["n"].iloc[0]) if not row.empty else 0)
            xpos = xbase + offsets[group]
            bars = ax.bar(xpos, vals, width=width, color=COLORS[group], label=group)
            for bar, n in zip(bars, ns):
                if np.isfinite(bar.get_height()):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_title(f"{plane}: success when same vs changed")
        ax.set_xticks(xbase)
        ax.set_xticklabels(status_order)
        ax.set_ylim(0, 100)
        ax.axhline(50, color="black", linestyle="--", linewidth=1)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Success (%)")
    fig.tight_layout()
    fig.savefig(outdir / "CLEAR_4_success_rate_same_vs_changed.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def save_confusion_matrices(df: pd.DataFrame, outdir: Path, threshold: float) -> None:
    """Create confusion matrices: orientation same/changed x success/failure."""
    valid = df.dropna(subset=["success_binary"]).copy()
    valid["orientation_status"] = np.where(
        valid["changed_from_start"], f"changed > {threshold:g} deg", f"same <= {threshold:g} deg"
    )
    valid["outcome"] = np.where(valid["success_binary"].eq(1), "success", "failure")
    orientation_order = [f"same <= {threshold:g} deg", f"changed > {threshold:g} deg"]
    outcome_order = ["success", "failure"]

    rows = []
    for group in GROUPS:
        for plane in PLANES:
            g = valid[(valid["experiment_group"].eq(group)) & (valid["plane"].eq(plane))]
            if g.empty:
                continue
            table = pd.crosstab(g["orientation_status"], g["outcome"]).reindex(
                index=orientation_order, columns=outcome_order, fill_value=0
            )
            total = table.to_numpy().sum()
            for orientation_status in orientation_order:
                row_total = int(table.loc[orientation_status].sum())
                for outcome in outcome_order:
                    n = int(table.loc[orientation_status, outcome])
                    rows.append(
                        {
                            "experiment_group": group,
                            "plane": plane,
                            "orientation_status": orientation_status,
                            "outcome": outcome,
                            "n": n,
                            "percent_of_all_in_group_plane": 100 * n / total if total else np.nan,
                            "percent_within_orientation_status": 100 * n / row_total if row_total else np.nan,
                            "row_total": row_total,
                            "group_plane_total": int(total),
                        }
                    )
    matrix_long = pd.DataFrame(rows)
    matrix_long.to_csv(outdir / "CLEAR_confusion_matrix_orientation_vs_success.csv", index=False)

    # Count heatmap: one row per group/plane, cell labels are counts and row-wise outcome percentages.
    fig, axes = plt.subplots(len(PLANES), len(GROUPS), figsize=(4.6 * len(GROUPS), 3.8 * len(PLANES)))
    axes = np.atleast_2d(axes)
    for i, plane in enumerate(PLANES):
        for j, group in enumerate(GROUPS):
            ax = axes[i, j]
            g = valid[(valid["experiment_group"].eq(group)) & (valid["plane"].eq(plane))]
            table = pd.crosstab(g["orientation_status"], g["outcome"]).reindex(
                index=orientation_order, columns=outcome_order, fill_value=0
            )
            arr = table.to_numpy(float)
            im = ax.imshow(arr, cmap="Purples" if group == "N_E" else "RdPu", aspect="auto")
            ax.set_title(f"{group} {plane}")
            ax.set_xticks(range(len(outcome_order)))
            ax.set_xticklabels(outcome_order)
            ax.set_yticks(range(len(orientation_order)))
            ax.set_yticklabels(orientation_order)
            for r in range(arr.shape[0]):
                row_total = arr[r].sum()
                for c in range(arr.shape[1]):
                    pct = 100 * arr[r, c] / row_total if row_total else 0
                    ax.text(c, r, f"{int(arr[r,c])}\n{pct:.1f}%", ha="center", va="center", color="black", fontsize=9)
            ax.set_xlabel("Outcome")
            if j == 0:
                ax.set_ylabel("Orientation status")
    fig.suptitle("Confusion matrix: orientation change status vs success/failure\nCell text = count and % within orientation row", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "CLEAR_6_confusion_matrix_orientation_vs_success.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def save_orientation_success_scatter(summary: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, len(PLANES), figsize=(5.2 * len(PLANES), 4.6), sharey=True)
    if len(PLANES) == 1:
        axes = [axes]
    corr_rows = []
    for ax, plane in zip(axes, PLANES):
        sub = summary[summary["plane"].eq(plane)]
        for group in GROUPS:
            g = sub[sub["experiment_group"].eq(group)].dropna(subset=["mean_abs_change_deg", "success_rate_percent"])
            if g.empty:
                continue
            r = g["mean_abs_change_deg"].corr(g["success_rate_percent"]) if len(g) >= 3 else np.nan
            corr_rows.append({"experiment_group": group, "plane": plane, "bin_level_corr_abs_degrees_vs_success_percent": r})
            ax.scatter(
                g["mean_abs_change_deg"],
                g["success_rate_percent"],
                s=70,
                color=COLORS[group],
                alpha=0.85,
                label=f"{group} r={r:.2f}" if np.isfinite(r) else group,
            )
        ax.set_title(f"{plane}: relation between deviation and success")
        ax.set_xlabel("Mean degrees from baseline")
        ax.set_ylim(0, 100)
        ax.axhline(50, color="black", linestyle="--", linewidth=1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Success (%)")
    pd.DataFrame(corr_rows).to_csv(outdir / "CLEAR_bin_correlation_degrees_vs_success.csv", index=False)
    fig.tight_layout()
    fig.savefig(outdir / "CLEAR_5_orientation_degrees_vs_success_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create clear orientation and success/failure visuals.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=script_dir / "outputs_threshold_10deg_start_first10" / "hand_orientation_change_trials_long.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "clear_visuals_threshold_10deg_start_first10",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    threshold = float(df["threshold_deg"].dropna().iloc[0]) if "threshold_deg" in df.columns else 10.0
    df = prepare(df)
    summary = bin_summary(df)

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "CLEAR_trial_order_bin_summary.csv", index=False)

    save_orientation_degrees_plot(summary, outdir, threshold)
    save_success_failure_plot(summary, outdir)
    save_changed_percent_plot(summary, outdir, threshold)
    save_success_by_changed_bar(df, outdir, threshold)
    save_confusion_matrices(df, outdir, threshold)
    save_orientation_success_scatter(summary, outdir)

    print("Read:", args.input_csv.resolve())
    print("Wrote clearer visuals to:", outdir)
    for path in sorted(outdir.iterdir()):
        print(" -", path.name)


if __name__ == "__main__":
    main()
