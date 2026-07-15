"""Analyse a motor-response run and produce the requested plots.

Inputs: the ``protocol_log.csv`` (or .xlsx) produced by ``run_experiment.py``.
Outputs (saved next to the log file, in a ``plots/`` subfolder):

  1. ``command_vs_response_motor.png``  - per step: commanded target (motor units)
     vs ESP32 ``actual``, plus motor error (actual - target) — same style as a
     classic command/response chart.
  2. ``command_vs_response_angle.png``  - per step: linear proxy of command in
     degrees (fit ``angle ~ k·target``) vs measured ``angle_block_zeroed``,
     plus angle error (only if vision angles are present).
  3. ``timeline_full.png``           - whole-experiment timeline:
                                       commanded target vs measured angle.
  4. ``timeline_delta_<D>.png``      - same plot zoomed to a single delta block
                                       (one figure per delta).
  5. ``trial_overlay_<D>.png``       - **per delta:** each trial plots
                                       ``angle_deg − mean(angle in this delta)``
                                       versus step (0 = block mean); the numeric
                                       mean appears in the **upper right** corner.
  6. ``delta_summary.png``           - per-delta error summary:
                                       a) box-plot of |angle change - mean|
                                          per delta (repeatability),
                                       b) mean +/- std of the angle change for
                                          +delta and -delta motor commands,
                                       c) ESP32 encoder relative error per delta.
  7. ``per_delta_summary.csv``       - same data as a flat table.

The "angle change" used in the summary is computed *within each delta block*
relative to the local zero of that block (so different starting offsets per
block do not affect the comparison).

Run with::

    python -m analysis.motor_response_analizer_servo.analyze <run_dir>
    # or
    python analysis/motor_response_analizer_servo/analyze.py <run_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib styling that matches the rest of the analysis notebooks:
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


SEQUENCE_COLORS = {"A": "tab:blue", "B": "tab:orange", "drift": "tab:gray"}


# ---------------------------------------------------------------------------
# Loading + per-delta normalisation
# ---------------------------------------------------------------------------

def load_log(run_dir: Path) -> pd.DataFrame:
    csv = run_dir / "protocol_log.csv"
    xlsx = run_dir / "protocol_log.xlsx"
    if csv.exists():
        df = pd.read_csv(csv)
    elif xlsx.exists():
        df = pd.read_excel(xlsx)
    else:
        raise FileNotFoundError(f"No protocol_log.csv/xlsx in {run_dir}")
    return df


def add_block_relative_angle(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``angle_block_zeroed`` so each delta block starts at angle 0.

    Each block's first non-NaN ``angle_deg`` is treated as that block's zero.
    """
    df = df.copy()
    df["angle_block_zeroed"] = np.nan
    if "angle_deg" not in df.columns:
        return df
    for _, idxs in df.groupby("block").groups.items():
        sub = df.loc[idxs]
        valid = sub["angle_deg"].dropna()
        if valid.empty:
            continue
        zero = valid.iloc[0]
        df.loc[idxs, "angle_block_zeroed"] = sub["angle_deg"] - zero
    return df


def add_trial_change(df: pd.DataFrame) -> pd.DataFrame:
    """For each protocol trial, compute the angle change relative to that trial's
    starting angle (step_index==1 within the same trial)."""
    df = df.copy()
    df["angle_in_trial"] = np.nan
    if "angle_deg" not in df.columns:
        return df
    proto = df[df["mode"] == "protocol"]
    for (block, trial), sub in proto.groupby(["block", "trial"]):
        valid = sub["angle_deg"].dropna()
        if valid.empty:
            continue
        zero = valid.iloc[0]
        df.loc[sub.index, "angle_in_trial"] = sub["angle_deg"] - zero
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_full_timeline(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax_target = plt.subplots(figsize=(13, 4.5))
    x = np.arange(len(df))
    ax_target.step(x, df["target"], where="post", color="black", lw=0.9, label="target")
    ax_target.set_ylabel("target (motor units)", color="black")
    ax_target.set_xlabel("step index")

    ax_angle = ax_target.twinx()
    if "angle_block_zeroed" in df.columns:
        ax_angle.plot(x, df["angle_block_zeroed"], color="crimson", lw=1.2,
                      label="angle (block-zeroed)")
        ax_angle.set_ylabel("angle [deg]  (block-zeroed)", color="crimson")
        ax_angle.tick_params(axis="y", labelcolor="crimson")
        ax_angle.grid(False)

    # Vertical lines at the start of each block.
    for block, sub in df.groupby("block"):
        x0 = sub.index.min()
        ax_target.axvline(x0, color="gray", alpha=0.25, lw=0.8)
        delta = int(sub["delta"].iloc[0])
        ax_target.text(x0, ax_target.get_ylim()[1], f"  D={delta}", va="top", fontsize=9, color="gray")

    ax_target.set_title("Full timeline: target command vs measured angle")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_delta_timeline(df: pd.DataFrame, delta: int, out_path: Path) -> None:
    sub = df[df["delta"] == delta].reset_index(drop=True)
    fig, ax_t = plt.subplots(figsize=(11, 4.0))
    x = np.arange(len(sub))
    ax_t.step(x, sub["target"], where="post", color="black", lw=1.0, label="target")
    ax_t.set_ylabel("target (motor units)")
    ax_t.set_xlabel(f"step index (within delta={delta} block)")

    ax_a = ax_t.twinx()
    if "angle_block_zeroed" in sub.columns:
        ax_a.plot(x, sub["angle_block_zeroed"], color="crimson", lw=1.4, label="angle")
        ax_a.set_ylabel("angle [deg] (block-zeroed)", color="crimson")
        ax_a.tick_params(axis="y", labelcolor="crimson")
        ax_a.grid(False)

    # Shade per-trial regions
    proto = sub[sub["mode"] == "protocol"]
    for (trial,), s in proto.groupby(["trial"]):
        seq = s["sequence"].iloc[0]
        x0 = s.index.min()
        x1 = s.index.max() + 1
        ax_t.axvspan(x0, x1, color=SEQUENCE_COLORS.get(seq, "tab:gray"), alpha=0.07)
        ax_t.text((x0 + x1) / 2, ax_t.get_ylim()[1], f"T{trial}{seq}",
                  ha="center", va="top", fontsize=8, color=SEQUENCE_COLORS.get(seq, "gray"))

    drift = sub[sub["mode"] == "drift"]
    if not drift.empty:
        x0 = drift.index.min()
        x1 = drift.index.max() + 1
        ax_t.axvspan(x0, x1, color="black", alpha=0.05)
        ax_t.text((x0 + x1) / 2, ax_t.get_ylim()[1], "drift",
                  ha="center", va="top", fontsize=9, color="black")

    ax_t.set_title(f"Delta = {delta}: target vs measured angle (block-zeroed)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trial_overlay(df: pd.DataFrame, delta: int, out_path: Path) -> None:
    """Overlay protocol trials as deviation from the **block mean angle** for this delta.

    For all protocol rows in the block we compute::

        residual = angle_deg − mean(angle_deg)

    so traces are centred on **y = 0** (above / below mean). Same mean is annotated
    in the upper-right of each subplot. Sequences **A** and **B** are separate panels.
    """
    proto = df[(df["delta"] == delta) & (df["mode"] == "protocol")].copy()
    if proto.empty or "angle_deg" not in proto.columns:
        return

    ang = pd.to_numeric(proto["angle_deg"], errors="coerce").dropna()
    if ang.empty:
        return
    mu_deg = float(ang.mean())
    proto["angle_deg_clean"] = pd.to_numeric(proto["angle_deg"], errors="coerce")
    proto["residual_deg"] = proto["angle_deg_clean"] - mu_deg

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, seq in zip(axes, ["A", "B"]):
        seq_data = proto[proto["sequence"] == seq]
        seq_data = seq_data.sort_values(["trial", "step_index"])
        if seq_data.empty:
            ax.set_visible(False)
            continue
        cmap = plt.get_cmap("tab10")
        for j, (trial, s) in enumerate(seq_data.groupby("trial", sort=False)):
            s = s.sort_values("step_index")
            y = pd.to_numeric(s["residual_deg"], errors="coerce").values
            color = cmap(j % 10)
            ax.plot(
                s["step_index"].values, y,
                marker="o", lw=1.15, markersize=4, alpha=0.88,
                color=color, label=f"trial {trial}",
            )

        ax.axhline(0.0, color="black", lw=1.05, zorder=0)
        ax.set_title(f"Sequence {seq}  (delta = {delta})")
        ax.set_xlabel("step index in trial (1 .. 5)")
        if seq == "A":
            ax.set_ylabel("Angle minus block mean [deg]")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.55)
        ax.legend(fontsize=8, ncol=2, framealpha=0.9, loc="upper left")

        # Block mean angle (same for A and B subplots) — upper right corner
        ax.text(
            0.99,
            0.99,
            f"Mean angle\n= {mu_deg:.4f} deg",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.65", alpha=0.95),
            zorder=10,
        )

    fig.suptitle(f"Deviation from mean angle - delta = {delta}  (positive = above mean)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-delta summary
# ---------------------------------------------------------------------------

def per_delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-delta angle and encoder errors.

    For every delta, we look only at *non-zero* protocol commands (target = +/-D)
    and compute:
        n_pos, n_neg                    : sample counts
        angle_mean_pos / angle_std_pos  : mean & std of angle_in_trial for +D commands
        angle_mean_neg / angle_std_neg  : same for -D
        angle_repeatability_std         : std of (sample - mean_for_its_target_sign) overall
        encoder_rel_err_mean / std      : ESP32 relative_error_percent on those commands
    """
    df = df.copy()
    proto = df[(df["mode"] == "protocol") & (df["target"] != 0)].copy()
    if proto.empty:
        return pd.DataFrame()

    # For analysis we want the move *into* +/-delta which is step_index 2 or 4.
    # angle_in_trial at those steps is the actual induced rotation.
    rows: list[dict] = []
    for delta, sub in proto.groupby("delta"):
        pos = sub[sub["target"] > 0]["angle_in_trial"].dropna().to_numpy()
        neg = sub[sub["target"] < 0]["angle_in_trial"].dropna().to_numpy()

        # repeatability: deviation from each sign's mean
        rep_dev = []
        if pos.size:
            rep_dev.append(pos - pos.mean())
        if neg.size:
            rep_dev.append(neg - neg.mean())
        rep_arr = np.concatenate(rep_dev) if rep_dev else np.array([])

        enc = sub["relative_error_percent"].dropna().to_numpy()

        rows.append({
            "delta": int(delta),
            "n_pos": int(pos.size),
            "n_neg": int(neg.size),
            "angle_mean_pos_deg": float(np.mean(pos)) if pos.size else np.nan,
            "angle_std_pos_deg":  float(np.std(pos, ddof=1)) if pos.size > 1 else np.nan,
            "angle_mean_neg_deg": float(np.mean(neg)) if neg.size else np.nan,
            "angle_std_neg_deg":  float(np.std(neg, ddof=1)) if neg.size > 1 else np.nan,
            "angle_repeatability_std_deg": float(np.std(rep_arr, ddof=1)) if rep_arr.size > 1 else np.nan,
            "encoder_rel_err_mean_pct": float(np.mean(enc)) if enc.size else np.nan,
            "encoder_rel_err_std_pct":  float(np.std(enc, ddof=1)) if enc.size > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def plot_delta_summary(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    """Three stacked sub-plots summarising errors across deltas."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) repeatability box plot of (sample - mean) per delta
    proto = df[(df["mode"] == "protocol") & (df["target"] != 0)].copy()
    box_data = []
    box_labels = []
    for delta, sub in proto.groupby("delta"):
        residuals = []
        for sign, side in [(+1, "+"), (-1, "-")]:
            sel = sub[np.sign(sub["target"]) == sign]["angle_in_trial"].dropna().to_numpy()
            if sel.size > 1:
                residuals.append(sel - sel.mean())
        if residuals:
            box_data.append(np.concatenate(residuals))
            box_labels.append(int(delta))
    if box_data:
        bp = axes[0].boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                             showmeans=True, meanline=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cfe3ff")
            patch.set_edgecolor("tab:blue")
        axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_title("(a) Trial-to-trial repeatability\n(angle - mean per sign per delta)")
    axes[0].set_xlabel("delta (motor units)")
    axes[0].set_ylabel("residual angle [deg]")

    # (b) mean +/- std of angle change per sign
    if not summary.empty:
        x = np.arange(len(summary))
        w = 0.38
        axes[1].bar(x - w/2, summary["angle_mean_pos_deg"], w,
                    yerr=summary["angle_std_pos_deg"], capsize=3,
                    color="tab:blue", alpha=0.8, label="+delta")
        axes[1].bar(x + w/2, summary["angle_mean_neg_deg"], w,
                    yerr=summary["angle_std_neg_deg"], capsize=3,
                    color="tab:orange", alpha=0.8, label="-delta")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(summary["delta"].astype(int))
        axes[1].axhline(0, color="gray", lw=0.5)
        axes[1].legend(frameon=False)
    axes[1].set_title("(b) Mean angle change per delta\n(error bars = std across trials)")
    axes[1].set_xlabel("delta (motor units)")
    axes[1].set_ylabel("angle change [deg]")

    # (c) ESP32 relative-error percent per delta
    if not summary.empty:
        x = np.arange(len(summary))
        axes[2].errorbar(x, summary["encoder_rel_err_mean_pct"],
                         yerr=summary["encoder_rel_err_std_pct"],
                         marker="o", ms=6, capsize=3, color="tab:green")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(summary["delta"].astype(int))
        axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_title("(c) ESP32 encoder relative error\n(% of commanded target)")
    axes[2].set_xlabel("delta (motor units)")
    axes[2].set_ylabel("rel. error [%]")

    fig.suptitle("Per-delta error summary")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def plot_command_vs_response_motor(df: pd.DataFrame, out_path: Path) -> None:
    """Command vs encoder response each step (+ motor error subplot)."""
    steps = np.arange(len(df))
    cmd = _safe_numeric(df["target"]).to_numpy(dtype=float)
    resp = _safe_numeric(df["actual"]).to_numpy(dtype=float)
    motor_err = resp - cmd

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2.8, 1.0]},
        sharex=True,
    )
    ax0.plot(
        steps, cmd, marker="o", ls="-", lw=1.0, markersize=3.5,
        color="tab:blue", markerfacecolor="tab:blue", markeredgewidth=0.6,
        label="Command (target)",
    )
    ax0.plot(
        steps, resp, marker="x", ls="-", lw=1.0, markersize=4.5,
        color="tab:orange", markeredgewidth=1.2,
        label="Response (actual, encoder)",
    )
    ax0.set_ylabel("Motor command / response [encoder units]")
    ax0.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    ax0.set_title("Command vs Response per Step (motor units)")
    ax0.axhline(0, color="gray", lw=0.4)

    ax1.fill_between(steps, 0, motor_err, color="tab:green", alpha=0.35, step=None)
    ax1.plot(steps, motor_err, color="darkgreen", lw=1.0, marker=".", ms=2, label="Error = actual − target")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Error\n[motor units]")
    ax1.legend(loc="upper right", fontsize=9, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _linear_angle_proxy(angle_block_z: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Best-fit slope k through origin (angle ~ k·target) for nonzero targets."""
    t = np.asarray(target, dtype=float)
    a = np.asarray(angle_block_z, dtype=float)
    mask = np.isfinite(a) & np.isfinite(t) & (np.abs(t) > 1e-6)
    if mask.sum() < 2:
        return np.nan, np.zeros_like(t)
    k = np.nansum(a[mask] * t[mask]) / np.nansum(t[mask] * t[mask])
    return float(k), k * t


def plot_command_vs_response_angle(df: pd.DataFrame, out_path: Path) -> Optional[float]:
    """Vision angles: commanded proxy (linear in target) vs measured block-zeroed angle."""
    if "angle_block_zeroed" not in df.columns:
        return None
    ang = pd.to_numeric(df["angle_block_zeroed"], errors="coerce").to_numpy(dtype=float)
    tgt = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(ang)):
        return None

    k, cmd_angle = _linear_angle_proxy(ang, tgt)
    if not np.isfinite(k):
        return None

    steps = np.arange(len(df))
    meas = ang.copy()
    err = meas - cmd_angle

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2.8, 1.0]},
        sharex=True,
    )
    ax0.plot(
        steps, cmd_angle, marker="o", ls="-", lw=1.0, markersize=3.5,
        color="tab:blue",
        label=f"Command proxy (linear: {k:.5f} deg / motor unit)",
    )
    ax0.plot(
        steps, meas, marker="x", ls="-", lw=1.0, markersize=4.5,
        color="tab:orange", markeredgewidth=1.2,
        label="Response (camera angle, block-zeroed)",
    )
    ax0.set_ylabel("Angle [deg]")
    ax0.legend(loc="upper left", ncol=1, frameon=False, fontsize=9)
    ax0.set_title("Command proxy vs measured angle per step (vision)")
    ax0.axhline(0, color="gray", lw=0.4)

    ax1.plot(steps, err, color="darkgreen", lw=1.0, marker=".", ms=2, alpha=0.85)
    ax1.fill_between(steps, err, color="tab:green", alpha=0.2)
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Angle error\n(measured − k·target) [deg]")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return k


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _latest_motor_run_dir() -> Optional[Path]:
    """Pick newest ``motor_response_*`` run: prefers ``<analyze.py package>/responses/``, then ``analysis/``."""
    dirs: list[Path] = []
    pkg_resp = Path(__file__).resolve().parent / "responses"
    if pkg_resp.is_dir():
        dirs.extend(p for p in pkg_resp.glob("motor_response_*") if p.is_dir())
    cwd_analysis = Path("analysis").resolve()
    if cwd_analysis.is_dir():
        dirs.extend(p for p in cwd_analysis.glob("motor_response_*") if p.is_dir())
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def analyze(run_dir: Path) -> Path:
    df = load_log(run_dir)
    df = add_block_relative_angle(df)
    df = add_trial_change(df)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_command_vs_response_motor(df, plots_dir / "command_vs_response_motor.png")
    plot_command_vs_response_angle(df, plots_dir / "command_vs_response_angle.png")

    plot_full_timeline(df, plots_dir / "timeline_full.png")
    for delta in sorted(df["delta"].unique()):
        delta = int(delta)
        plot_delta_timeline(df, delta, plots_dir / f"timeline_delta_{delta}.png")
        plot_trial_overlay(df, delta, plots_dir / f"trial_overlay_{delta}.png")

    summary = per_delta_summary(df)
    if not summary.empty:
        summary.to_csv(run_dir / "per_delta_summary.csv", index=False)
    plot_delta_summary(df, summary, plots_dir / "delta_summary.png")

    print(f"[analyze] plots saved under: {plots_dir}")
    print(f"[analyze] per-delta summary: {run_dir / 'per_delta_summary.csv'}")
    if not summary.empty:
        print()
        print(summary.to_string(index=False))
    return plots_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Folder ``motor_response_<timestamp>``. If omitted, uses newest run "
             "under ``motor_response_analizer_servo/responses/`` then under ``analysis/``.",
    )
    args = p.parse_args()

    if args.run_dir is None:
        run_dir = _latest_motor_run_dir()
        if run_dir is None:
            raise SystemExit(
                "No motor_response_* run found. Expected under motor_response_analizer_servo/responses/ "
                "or analysis/. Pass the run folder explicitly."
            )
        print(f"[analyze] using most recent run: {run_dir}")
    else:
        run_dir = args.run_dir

    analyze(run_dir)


if __name__ == "__main__":
    main()
