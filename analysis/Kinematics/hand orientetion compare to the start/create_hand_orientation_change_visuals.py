
"""
Create visuals showing WHEN hand orientation changes happen and TO WHERE they move.

Reads the comparison CSV created by compare_hand_orientation_to_start.py.
Does not modify result_fillter.

Default input:
    hand orientetion compare to the start/outputs_threshold_10deg/hand_orientation_change_trials_long.csv

Outputs:
    hand orientetion compare to the start/visuals_threshold_10deg/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def prepare_ordered_table(df: pd.DataFrame) -> pd.DataFrame:
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
    df["change_direction"] = np.select(
        [
            df["changed_from_start"].eq(False),
            df["signed_change_from_start_deg"].gt(0),
            df["signed_change_from_start_deg"].lt(0),
        ],
        ["same (within threshold)", "positive angle", "negative angle"],
        default="zero/unknown",
    )
    return df


def plot_group_signed_change(df: pd.DataFrame, outdir: Path, threshold: float) -> None:
    planes = [p for p in ["XY", "YZ", "ZX"] if p in set(df["plane"])]
    groups = [g for g in ["L_E", "N_E"] if g in set(df["experiment_group"])]
    fig, axes = plt.subplots(len(planes), 1, figsize=(12, 3.5 * len(planes)), sharex=True)
    if len(planes) == 1:
        axes = [axes]

    colors = {"L_E": "#ff69b4", "N_E": "#7b2cbf"}
    for ax, plane in zip(axes, planes):
        sub = df[df["plane"].eq(plane)]
        for group in groups:
            g = sub[sub["experiment_group"].eq(group)]
            summary = (
                g.groupby("sequence_bin", observed=True)
                .agg(
                    mean_change=("signed_change_from_start_deg", "mean"),
                    sem_change=("signed_change_from_start_deg", sem),
                    x=("sequence_percent", "mean"),
                )
                .reset_index()
                .dropna(subset=["x"])
            )
            ax.plot(summary["x"], summary["mean_change"], marker="o", label=group, color=colors.get(group))
            ax.fill_between(
                summary["x"].to_numpy(float),
                (summary["mean_change"] - summary["sem_change"]).to_numpy(float),
                (summary["mean_change"] + summary["sem_change"]).to_numpy(float),
                color=colors.get(group),
                alpha=0.15,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.axhline(-threshold, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{plane}: where orientation moved from start")
        ax.set_ylabel("Signed change from start (deg)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("When in participant/finger sequence (% from first to last trial segment)")
    fig.suptitle(f"Hand orientation change over trial order; dashed lines = ?{threshold:g}?", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "when_and_where_signed_change_by_group_plane.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_changed_percent_bins(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby(["experiment_group", "plane", "sequence_bin"], observed=True)
        .agg(percent_changed=("changed_from_start", lambda s: 100 * s.mean()), n=("changed_from_start", "size"))
        .reset_index()
    )
    summary["x"] = summary["sequence_bin"].astype(str).str.extract(r"^(\d+)").astype(float) + 5

    planes = [p for p in ["XY", "YZ", "ZX"] if p in set(df["plane"])]
    groups = [g for g in ["L_E", "N_E"] if g in set(df["experiment_group"])]
    colors = {"L_E": "#ff69b4", "N_E": "#7b2cbf"}
    fig, axes = plt.subplots(len(planes), 1, figsize=(12, 3.3 * len(planes)), sharex=True)
    if len(planes) == 1:
        axes = [axes]
    for ax, plane in zip(axes, planes):
        sub = summary[summary["plane"].eq(plane)]
        for group in groups:
            g = sub[sub["experiment_group"].eq(group)]
            ax.plot(g["x"], g["percent_changed"], marker="o", label=group, color=colors.get(group))
        ax.set_ylim(0, 100)
        ax.set_title(f"{plane}: when orientation changed")
        ax.set_ylabel("% changed from start")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("When in participant/finger sequence (% bins)")
    fig.tight_layout()
    fig.savefig(outdir / "when_percent_changed_by_group_plane.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



def add_success_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Add success_binary: success=1, failure=0, otherwise NaN."""
    out = df.copy()
    if "success_label" not in out.columns:
        out["success_binary"] = np.nan
        return out
    labels = out["success_label"].astype(str).str.lower().str.strip()
    out["success_binary"] = np.select(
        [labels.eq("success"), labels.eq("failure")],
        [1.0, 0.0],
        default=np.nan,
    )
    return out


def pearson_or_nan(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    if x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(x[ok].corr(y[ok]))


def write_success_correlation_tables(df: pd.DataFrame, outdir: Path) -> None:
    """Save row-level and trial-order-bin correlations with success/failure."""
    df = add_success_binary(df)
    valid = df.dropna(subset=["success_binary"]).copy()
    rows = []
    for keys, g in valid.groupby(["experiment_group", "plane"], dropna=False, sort=False):
        group, plane = keys
        rows.append(
            {
                "experiment_group": group,
                "plane": plane,
                "n_trial_segments": int(len(g)),
                "success_rate_percent": 100.0 * float(g["success_binary"].mean()),
                "corr_abs_change_deg_vs_success": pearson_or_nan(g["abs_change_from_start_deg"], g["success_binary"]),
                "corr_signed_change_deg_vs_success": pearson_or_nan(g["signed_change_from_start_deg"], g["success_binary"]),
                "corr_changed_gt_threshold_vs_success": pearson_or_nan(g["changed_from_start"].astype(float), g["success_binary"]),
                "mean_abs_change_success": float(g.loc[g["success_binary"].eq(1), "abs_change_from_start_deg"].mean()),
                "mean_abs_change_failure": float(g.loc[g["success_binary"].eq(0), "abs_change_from_start_deg"].mean()),
                "percent_changed_success": 100.0 * float(g.loc[g["success_binary"].eq(1), "changed_from_start"].mean()),
                "percent_changed_failure": 100.0 * float(g.loc[g["success_binary"].eq(0), "changed_from_start"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "orientation_success_correlation_summary.csv", index=False)

    binned = (
        valid.groupby(["experiment_group", "plane", "sequence_bin"], observed=True)
        .agg(
            n=("success_binary", "size"),
            n_success=("success_binary", lambda s: int((s == 1).sum())),
            n_failure=("success_binary", lambda s: int((s == 0).sum())),
            success_rate_percent=("success_binary", lambda s: 100 * s.mean()),
            percent_changed=("changed_from_start", lambda s: 100 * s.mean()),
            mean_abs_change_deg=("abs_change_from_start_deg", "mean"),
            mean_signed_change_deg=("signed_change_from_start_deg", "mean"),
        )
        .reset_index()
    )
    binned["group_majority_success"] = binned["n_success"] > binned["n_failure"]
    binned["group_majority_failure"] = binned["n_failure"] > binned["n_success"]
    binned["group_majority_tie"] = binned["n_success"] == binned["n_failure"]
    binned["group_majority_success_binary"] = np.where(
        binned["group_majority_tie"], np.nan, binned["group_majority_success"].astype(float)
    )
    binned.to_csv(outdir / "success_and_orientation_by_trial_order_bin.csv", index=False)

    bin_corr_rows = []
    for keys, g in binned.groupby(["experiment_group", "plane"], dropna=False, sort=False):
        group, plane = keys
        bin_corr_rows.append(
            {
                "experiment_group": group,
                "plane": plane,
                "n_bins": int(len(g)),
                "corr_percent_changed_vs_success_rate": pearson_or_nan(g["percent_changed"], g["success_rate_percent"]),
                "corr_mean_abs_change_vs_success_rate": pearson_or_nan(g["mean_abs_change_deg"], g["success_rate_percent"]),
                "corr_mean_signed_change_vs_success_rate": pearson_or_nan(g["mean_signed_change_deg"], g["success_rate_percent"]),
                "corr_percent_changed_vs_group_majority_success": pearson_or_nan(g["percent_changed"], g["group_majority_success_binary"]),
                "corr_mean_abs_change_vs_group_majority_success": pearson_or_nan(g["mean_abs_change_deg"], g["group_majority_success_binary"]),
                "n_majority_success_bins": int(g["group_majority_success"].sum()),
                "n_majority_failure_bins": int(g["group_majority_failure"].sum()),
                "n_tie_bins": int(g["group_majority_tie"].sum()),
            }
        )
    pd.DataFrame(bin_corr_rows).to_csv(outdir / "orientation_success_trial_order_bin_correlations.csv", index=False)

    # Finer-grained majority rule at exact sequence index: a sequence is group success
    # only when more participant/finger trial-segments are success than failure.
    exact = (
        valid.groupby(["experiment_group", "plane", "sequence_index"], observed=True)
        .agg(
            n=("success_binary", "size"),
            n_success=("success_binary", lambda s: int((s == 1).sum())),
            n_failure=("success_binary", lambda s: int((s == 0).sum())),
            success_rate_percent=("success_binary", lambda s: 100 * s.mean()),
            percent_changed=("changed_from_start", lambda s: 100 * s.mean()),
            mean_abs_change_deg=("abs_change_from_start_deg", "mean"),
            mean_signed_change_deg=("signed_change_from_start_deg", "mean"),
        )
        .reset_index()
    )
    exact["group_majority_success"] = exact["n_success"] > exact["n_failure"]
    exact["group_majority_failure"] = exact["n_failure"] > exact["n_success"]
    exact["group_majority_tie"] = exact["n_success"] == exact["n_failure"]
    exact["group_majority_success_binary"] = np.where(
        exact["group_majority_tie"], np.nan, exact["group_majority_success"].astype(float)
    )
    exact.to_csv(outdir / "success_and_orientation_by_exact_sequence_index.csv", index=False)

    exact_corr_rows = []
    for keys, g in exact.groupby(["experiment_group", "plane"], dropna=False, sort=False):
        group, plane = keys
        exact_corr_rows.append(
            {
                "experiment_group": group,
                "plane": plane,
                "n_sequence_indices": int(len(g)),
                "corr_percent_changed_vs_success_rate": pearson_or_nan(g["percent_changed"], g["success_rate_percent"]),
                "corr_mean_abs_change_vs_success_rate": pearson_or_nan(g["mean_abs_change_deg"], g["success_rate_percent"]),
                "corr_percent_changed_vs_group_majority_success": pearson_or_nan(g["percent_changed"], g["group_majority_success_binary"]),
                "corr_mean_abs_change_vs_group_majority_success": pearson_or_nan(g["mean_abs_change_deg"], g["group_majority_success_binary"]),
                "n_majority_success_indices": int(g["group_majority_success"].sum()),
                "n_majority_failure_indices": int(g["group_majority_failure"].sum()),
                "n_tie_indices": int(g["group_majority_tie"].sum()),
            }
        )
    pd.DataFrame(exact_corr_rows).to_csv(outdir / "orientation_success_exact_sequence_correlations.csv", index=False)


def plot_change_and_success_over_trials(df: pd.DataFrame, outdir: Path) -> None:
    """Plot percent changed and success rate over normalized trial order."""
    df = add_success_binary(df)
    valid = df.dropna(subset=["success_binary"]).copy()
    if valid.empty:
        return
    summary = (
        valid.groupby(["experiment_group", "plane", "sequence_bin"], observed=True)
        .agg(
            n_success=("success_binary", lambda s: int((s == 1).sum())),
            n_failure=("success_binary", lambda s: int((s == 0).sum())),
            success_rate_percent=("success_binary", lambda s: 100 * s.mean()),
            percent_changed=("changed_from_start", lambda s: 100 * s.mean()),
            mean_abs_change_deg=("abs_change_from_start_deg", "mean"),
        )
        .reset_index()
    )
    summary["group_majority_success"] = summary["n_success"] > summary["n_failure"]
    summary["x"] = summary["sequence_bin"].astype(str).str.extract(r"^(\d+)").astype(float) + 5

    planes = [p for p in ["XY", "YZ", "ZX"] if p in set(valid["plane"])]
    groups = [g for g in ["L_E", "N_E"] if g in set(valid["experiment_group"])]
    colors = {"L_E": "#ff69b4", "N_E": "#7b2cbf"}
    fig, axes = plt.subplots(len(planes), 1, figsize=(12, 3.8 * len(planes)), sharex=True)
    if len(planes) == 1:
        axes = [axes]
    for ax, plane in zip(axes, planes):
        sub = summary[summary["plane"].eq(plane)]
        for group in groups:
            g = sub[sub["experiment_group"].eq(group)]
            color = colors.get(group)
            ax.plot(
                g["x"],
                g["percent_changed"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{group}: % changed > threshold",
            )
            ax.plot(
                g["x"],
                g["success_rate_percent"],
                marker="s",
                linestyle="--",
                color=color,
                label=f"{group}: success rate",
            )
        ax.axhline(50, color="black", linestyle=":", linewidth=1, label="50% majority line")
        ax.set_ylim(0, 100)
        ax.set_title(f"{plane}: orientation change and group success/failure over trials")
        ax.set_ylabel("Percent")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("When in participant/finger sequence (% bins)")
    fig.tight_layout()
    fig.savefig(outdir / "orientation_change_and_success_over_trials.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_direction_counts(df: pd.DataFrame, outdir: Path) -> None:
    changed = df[df["changed_from_start"].eq(True)].copy()
    changed["direction"] = np.where(changed["signed_change_from_start_deg"] > 0, "positive angle", "negative angle")
    counts = (
        changed.groupby(["experiment_group", "plane", "direction"])
        .size()
        .rename("n")
        .reset_index()
    )
    counts.to_csv(outdir / "change_direction_counts.csv", index=False)

    planes = [p for p in ["XY", "YZ", "ZX"] if p in set(df["plane"])]
    groups = [g for g in ["L_E", "N_E"] if g in set(df["experiment_group"])]
    directions = ["negative angle", "positive angle"]
    fig, axes = plt.subplots(1, len(planes), figsize=(5 * len(planes), 4), sharey=False)
    if len(planes) == 1:
        axes = [axes]
    for ax, plane in zip(axes, planes):
        labels = []
        vals = []
        colors = []
        for group in groups:
            for direction in directions:
                row = counts[(counts["plane"].eq(plane)) & (counts["experiment_group"].eq(group)) & (counts["direction"].eq(direction))]
                labels.append(f"{group}\n{direction.replace(' angle','')}")
                vals.append(int(row["n"].iloc[0]) if not row.empty else 0)
                colors.append("#9467bd" if direction.startswith("negative") else "#2ca02c")
        ax.bar(labels, vals, color=colors)
        ax.set_title(f"{plane}: to where? direction of >threshold changes")
        ax.set_ylabel("Number of changed trial segments")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "to_where_positive_vs_negative_change_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_participant_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    for plane in ["XY", "YZ", "ZX"]:
        sub = df[df["plane"].eq(plane)].copy()
        if sub.empty:
            continue
        # Use absolute change by participant across normalized sequence bins.
        heat = (
            sub.groupby(["experiment_group", "subject_id", "sequence_bin"], observed=True)["abs_change_from_start_deg"]
            .mean()
            .reset_index()
        )
        heat["participant"] = heat["experiment_group"].astype(str) + " / " + heat["subject_id"].astype(str)
        pivot = heat.pivot(index="participant", columns="sequence_bin", values="abs_change_from_start_deg")
        # Sort L_E then N_E by participant numeric suffix.
        def sort_key(label: str):
            group, sid = [x.strip() for x in label.split("/")]
            try:
                num = int(sid.split("_")[-1])
            except Exception:
                num = 9999
            return (group, num)
        pivot = pivot.loc[sorted(pivot.index, key=sort_key)]

        fig, ax = plt.subplots(figsize=(11, max(6, 0.22 * len(pivot))))
        im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(f"{plane}: participant heatmap of absolute change from start")
        ax.set_xlabel("When in sequence (% bins)")
        ax.set_ylabel("Participant")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean absolute change from start (deg)")
        fig.tight_layout()
        fig.savefig(outdir / f"participant_when_heatmap_abs_change_{plane}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create visuals for hand orientation changes from start.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=script_dir / "outputs_threshold_10deg" / "hand_orientation_change_trials_long.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "visuals_threshold_10deg",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    threshold = float(df["threshold_deg"].dropna().iloc[0]) if "threshold_deg" in df.columns else 10.0
    df = prepare_ordered_table(df)

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    plot_group_signed_change(df, outdir, threshold)
    plot_changed_percent_bins(df, outdir)
    plot_direction_counts(df, outdir)
    write_success_correlation_tables(df, outdir)
    plot_change_and_success_over_trials(df, outdir)
    plot_participant_heatmaps(df, outdir)

    print("Read:", args.input_csv.resolve())
    print("Wrote visuals to:", outdir)
    for p in sorted(outdir.glob("*.png")):
        print(" -", p.name)
    print(" - change_direction_counts.csv")
    print(" - orientation_success_correlation_summary.csv")
    print(" - success_and_orientation_by_trial_order_bin.csv")
    print(" - orientation_success_trial_order_bin_correlations.csv")
    print(" - success_and_orientation_by_exact_sequence_index.csv")
    print(" - orientation_success_exact_sequence_correlations.csv")


if __name__ == "__main__":
    main()
