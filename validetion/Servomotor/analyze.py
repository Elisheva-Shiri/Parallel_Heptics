"""Analyse a motor-response run and produce the requested plots.

Inputs: the ``protocol_log.csv`` (or .xlsx) produced by ``run_experiment.py``.
Outputs (saved next to the log file, in a ``plots/`` subfolder):

  1. ``command_vs_response_angle.png``  - per step: command ticks mapped to a
     nominal angle scale (``+/-1000 ticks = +/-90 deg``) vs measured trial-local
     camera angle. The mapped command is a reference scale, not an independent
     measured motor angle.
  3. ``timeline_full.png``           - whole-experiment timeline:
                                       commanded target vs measured angle.
  4. ``timeline_delta_<D>.png``      - same plot zoomed to a single delta block
                                       (one figure per delta).
  5. ``trial_overlay_<D>.png``       - **per delta:** each trial plots
                                       ``angle_deg − mean(angle in this delta)``
                                       versus step (0 = block mean); the numeric
                                       mean appears in the **upper right** corner.
  6. ``delta_summary.png``           - per-delta angle-response summary:
                                       a) box-plot of |angle change - mean|
                                          per delta (repeatability),
                                       b) mean +/- std of the angle change for
                                          +delta and -delta motor commands,
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
COMMAND_TICKS_AT_90_DEG = 1000.0
COMMAND_DEG_PER_TICK = 90.0 / COMMAND_TICKS_AT_90_DEG


def command_ticks_to_nominal_deg(target: pd.Series | np.ndarray) -> np.ndarray:
    """Map command ticks to a nominal degree scale for plotting only.

    The fixed display convention is +/-1000 command ticks = +/-90 degrees.
    This is a command-reference scale, not a second measured motor angle.
    """
    return pd.to_numeric(target, errors="coerce").to_numpy(dtype=float) * COMMAND_DEG_PER_TICK


def _endpoint_wrap_corrected(target: pd.Series, angle: pd.Series) -> pd.Series:
    """Correct the +/-1000 endpoint's 180-degree line-orientation ambiguity.

    The black line has an orientation ambiguity near +/-90 degrees. For the
    endpoint stress-test command only, force measured response sign to match the
    command sign while preserving the measured magnitude.
    """
    out = pd.to_numeric(angle, errors="coerce").copy()
    tgt = pd.to_numeric(target, errors="coerce")
    endpoint = tgt.abs() == COMMAND_TICKS_AT_90_DEG
    nonzero = endpoint & (tgt != 0) & out.notna()
    out.loc[nonzero] = np.sign(tgt.loc[nonzero]) * out.loc[nonzero].abs()
    return out


def _orient_to_command_reference(target: pd.Series, angle: pd.Series) -> pd.Series:
    """Display-only orientation correction for the 180-degree black-line ambiguity.

    For every non-zero command, choose between angle, angle-180, and angle+180 so
    the displayed measurement is closest to the fixed command reference
    (+/-1000 ticks = +/-90 deg). This keeps drift traces readable when an
    equivalent line orientation is recorded on the other side of the 180-degree
    circle.
    """
    out = pd.to_numeric(angle, errors="coerce").copy()
    tgt = pd.to_numeric(target, errors="coerce")
    ref = pd.Series(command_ticks_to_nominal_deg(tgt), index=out.index)
    mask = (tgt != 0) & out.notna() & ref.notna()
    for idx in out.index[mask]:
        candidates = np.array([out.loc[idx], out.loc[idx] - 180.0, out.loc[idx] + 180.0], dtype=float)
        out.loc[idx] = candidates[np.argmin(np.abs(candidates - ref.loc[idx]))]
    return out


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
    df["angle_block_zeroed_display"] = _orient_to_command_reference(df["target"], df["angle_block_zeroed"])
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
    df["angle_response_deg"] = _endpoint_wrap_corrected(df["target"], df["angle_in_trial"])
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_full_timeline(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax_target = plt.subplots(figsize=(13, 4.5))
    x = np.arange(len(df))
    ax_target.step(x, df["target"], where="post", color="black", lw=0.9,
                   label="command target [ticks]")
    ax_target.set_ylabel("command target [ticks]\n(nominal map: 1000 ticks = 90 deg)", color="black")
    ax_target.set_xlabel("step index")

    ax_angle = ax_target.twinx()
    angle_col = "angle_block_zeroed_display" if "angle_block_zeroed_display" in df.columns else "angle_block_zeroed"
    if angle_col in df.columns:
        ax_angle.plot(x, df[angle_col], color="crimson", lw=1.2,
                      label="measured angle [deg] (block-zeroed; 1000 endpoint sign-corrected)")
        ax_angle.set_ylabel("measured angle [deg]\n(block-zeroed)", color="crimson")
        ax_angle.tick_params(axis="y", labelcolor="crimson")
        ax_angle.grid(False)

    # Vertical lines at the start of each block.
    for block, sub in df.groupby("block"):
        x0 = sub.index.min()
        ax_target.axvline(x0, color="gray", alpha=0.25, lw=0.8)
        delta = int(sub["delta"].iloc[0])
        ax_target.text(x0, ax_target.get_ylim()[1], f"  D={delta}", va="top", fontsize=9, color="gray")

    lines, labels = ax_target.get_legend_handles_labels()
    lines2, labels2 = ax_angle.get_legend_handles_labels()
    ax_target.legend(
        lines + lines2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    ax_target.set_title("Full timeline: command timing and measured spool angle (different units)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def plot_delta_timeline(df: pd.DataFrame, delta: int, out_path: Path) -> None:
    sub = df[df["delta"] == delta].reset_index(drop=True)
    fig, ax_t = plt.subplots(figsize=(11, 4.0))
    x = np.arange(len(sub))
    ax_t.step(x, sub["target"], where="post", color="black", lw=1.0, label="command target [ticks]")
    ax_t.set_ylabel("command target [ticks]\n(nominal map: 1000 ticks = 90 deg)")
    ax_t.set_xlabel(f"step index (within delta={delta} block)")

    ax_a = ax_t.twinx()
    angle_col = "angle_block_zeroed_display" if "angle_block_zeroed_display" in sub.columns else "angle_block_zeroed"
    if angle_col in sub.columns:
        ax_a.plot(x, sub[angle_col], color="crimson", lw=1.4,
                  label="measured angle [deg] (block-zeroed)")
        ax_a.set_ylabel("measured angle [deg] (block-zeroed)", color="crimson")
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

    lines, labels = ax_t.get_legend_handles_labels()
    lines2, labels2 = ax_a.get_legend_handles_labels()
    ax_t.legend(
        lines + lines2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    ax_t.set_title(f"Delta = {delta}: command timing and measured angle (different units)")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def plot_trial_overlay(df: pd.DataFrame, delta: int, out_path: Path) -> None:
    """Show protocol-trial shape plus all measured values for one delta.

    Top row: the original A/B overlay, centred around the protocol block mean.
    Bottom row: all non-zero +delta protocol responses, all non-zero -delta
    protocol responses, and all drift-block measured angles. Thick horizontal
    bars mark each group's mean.
    """
    block = df[df["delta"] == delta].copy()
    proto = block[block["mode"] == "protocol"].copy()
    if proto.empty or "angle_deg" not in proto.columns:
        return

    ang = pd.to_numeric(proto["angle_deg"], errors="coerce").dropna()
    if ang.empty:
        return
    mu_deg = float(ang.mean())
    proto["angle_deg_clean"] = pd.to_numeric(proto["angle_deg"], errors="coerce")
    proto["residual_deg"] = proto["angle_deg_clean"] - mu_deg

    fig = plt.figure(figsize=(12.5, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.1, 1.35])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_values = fig.add_subplot(gs[1, :])

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
        if int(delta) in (5, 10):
            ax.set_ylim(-1.0, 1.0)
        elif int(delta) == 25:
            ax.set_ylim(-4.0, 4.0)
        ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.55)
        ax.legend(fontsize=8, ncol=2, framealpha=0.9, loc="upper left")

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

    # Bottom panel: all response values by group, including the drift block.
    response_col = "angle_response_deg" if "angle_response_deg" in block.columns else "angle_in_trial"
    drift_col = "angle_block_zeroed_display" if "angle_block_zeroed_display" in block.columns else "angle_block_zeroed"
    groups = [
        ("+delta\nprotocol", proto[pd.to_numeric(proto["target"], errors="coerce") > 0][response_col], "tab:blue"),
        ("-delta\nprotocol", proto[pd.to_numeric(proto["target"], errors="coerce") < 0][response_col], "tab:orange"),
        ("drift\nall values", block[block["mode"] == "drift"][drift_col], "tab:gray"),
    ]
    rng = np.random.default_rng(12345)
    for x0, (label, values, color) in enumerate(groups, start=1):
        vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=vals.size)
        ax_values.scatter(
            np.full(vals.size, x0) + jitter,
            vals,
            s=28,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=label.replace("\n", " "),
        )
        mean = float(np.mean(vals))
        ax_values.hlines(mean, x0 - 0.28, x0 + 0.28, colors=color, linewidth=4.0)
        ax_values.text(
            x0 + 0.31,
            mean,
            f"mean={mean:.3g} deg",
            color=color,
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax_values.axhline(0.0, color="black", lw=0.9, zorder=0)
    ax_values.set_xlim(0.45, 3.75)
    ax_values.set_xticks([1, 2, 3])
    ax_values.set_xticklabels([g[0] for g in groups])
    ax_values.set_ylabel("Measured response [deg]")
    ax_values.set_title("All measured values: protocol signs and drift block (thick line = group mean)")
    if int(delta) in (5, 10):
        ax_values.set_ylim(-1.0, 1.0)
    elif int(delta) == 25:
        ax_values.set_ylim(-4.0, 4.0)
    ax_values.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.55)

    fig.suptitle(f"Delta = {delta}: trial overlay and all measured response values")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-delta summary
# ---------------------------------------------------------------------------

def per_delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-delta camera angle responses.

    For every delta, we look only at *non-zero* protocol commands (target = +/-D)
    and compute:
        n_pos, n_neg                    : sample counts
        angle_mean_pos / angle_std_pos  : mean & std of angle_response_deg for +D commands
        angle_mean_neg / angle_std_neg  : same for -D
        angle_repeatability_std         : std of (sample - mean_for_its_target_sign) overall
    """
    df = df.copy()
    proto = df[(df["mode"] == "protocol") & (df["target"] != 0)].copy()
    if proto.empty:
        return pd.DataFrame()
    response_col = "angle_response_deg" if "angle_response_deg" in proto.columns else "angle_in_trial"

    # For analysis we want the move *into* +/-delta which is step_index 2 or 4.
    # angle_response_deg at those steps is the actual induced rotation, with only
    # the +/-1000 endpoint sign-corrected for the 180-degree line ambiguity.
    rows: list[dict] = []
    for delta, sub in proto.groupby("delta"):
        pos = sub[sub["target"] > 0][response_col].dropna().to_numpy()
        neg = sub[sub["target"] < 0][response_col].dropna().to_numpy()

        # repeatability: deviation from each sign's mean
        rep_dev = []
        if pos.size:
            rep_dev.append(pos - pos.mean())
        if neg.size:
            rep_dev.append(neg - neg.mean())
        rep_arr = np.concatenate(rep_dev) if rep_dev else np.array([])

        rows.append({
            "delta": int(delta),
            "n_pos": int(pos.size),
            "n_neg": int(neg.size),
            "angle_mean_pos_deg": float(np.mean(pos)) if pos.size else np.nan,
            "angle_std_pos_deg":  float(np.std(pos, ddof=1)) if pos.size > 1 else np.nan,
            "angle_mean_neg_deg": float(np.mean(neg)) if neg.size else np.nan,
            "angle_std_neg_deg":  float(np.std(neg, ddof=1)) if neg.size > 1 else np.nan,
            "angle_repeatability_std_deg": float(np.std(rep_arr, ddof=1)) if rep_arr.size > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def plot_delta_summary(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    """Two-panel per-delta summary of camera-angle response."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) repeatability box plot of (sample - mean) per delta
    proto = df[(df["mode"] == "protocol") & (df["target"] != 0)].copy()
    response_col = "angle_response_deg" if "angle_response_deg" in proto.columns else "angle_in_trial"
    box_data = []
    box_labels = []
    for delta, sub in proto.groupby("delta"):
        residuals = []
        for sign, side in [(+1, "+"), (-1, "-")]:
            sel = sub[np.sign(sub["target"]) == sign][response_col].dropna().to_numpy()
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
        pos_bars = axes[1].bar(x - w/2, summary["angle_mean_pos_deg"], w,
                               yerr=summary["angle_std_pos_deg"], capsize=3,
                               color="tab:blue", alpha=0.8, label="+delta")
        neg_bars = axes[1].bar(x + w/2, summary["angle_mean_neg_deg"], w,
                               yerr=summary["angle_std_neg_deg"], capsize=3,
                               color="tab:orange", alpha=0.8, label="-delta")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(summary["delta"].astype(int))
        axes[1].axhline(0, color="gray", lw=0.5)
        axes[1].legend(frameon=False)

        def _label_bars(bars) -> None:
            for bar in bars:
                height = float(bar.get_height())
                if not np.isfinite(height):
                    continue
                va = "bottom" if height >= 0 else "top"
                offset = 3 if height >= 0 else -3
                label = f"{height:.2f}" if abs(height) < 10 else f"{height:.1f}"
                axes[1].annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=7,
                    rotation=90,
                    color="black",
                )

        _label_bars(pos_bars)
        _label_bars(neg_bars)
    axes[1].set_title("(b) Mean angle change per delta\n(error bars = std across trials)")
    axes[1].set_xlabel("delta (motor units)")
    axes[1].set_ylabel("angle change [deg]")

    fig.suptitle("Per-delta camera angle-response summary")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def plot_command_vs_response_angle(df: pd.DataFrame, out_path: Path) -> None:
    """Mapped command angle vs measured block-zeroed camera angle.

    The command line is not a measured motor angle. It is the fixed display
    mapping requested for interpretation: +/-1000 command ticks = +/-90 deg.
    """
    angle_col = "angle_block_zeroed_display" if "angle_block_zeroed_display" in df.columns else "angle_block_zeroed"
    if angle_col not in df.columns:
        return None

    steps = np.arange(len(df))
    cmd_angle = command_ticks_to_nominal_deg(df["target"])
    meas = pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(meas)):
        return None
    err = meas - cmd_angle

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2.8, 1.0]},
        sharex=True,
    )
    ax0.plot(
        steps, cmd_angle, marker="o", ls="-", lw=1.0, markersize=3.5,
        color="tab:blue",
        label="Command reference (fixed map: 1000 ticks = 90 deg; not measured)",
    )
    ax0.plot(
        steps, meas, marker="x", ls="-", lw=1.0, markersize=4.5,
        color="tab:orange", markeredgewidth=1.2,
        label="Measured camera angle (block-zeroed; drift included; 1000 endpoint sign-corrected)",
    )
    ax0.set_ylabel("Angle [deg]")
    ax0.legend(loc="upper left", ncol=1, frameon=False, fontsize=9)
    ax0.set_title("Mapped command reference vs measured spool response")
    ax0.axhline(0, color="gray", lw=0.4)

    ax1.plot(steps, err, color="darkgreen", lw=1.0, marker=".", ms=2, alpha=0.85)
    ax1.fill_between(steps, err, color="tab:green", alpha=0.2)
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Angle error\n(measured - mapped command) [deg]")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return None


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
