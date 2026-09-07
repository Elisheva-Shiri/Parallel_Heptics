"""Manuscript figure and numbers for the motor-validation review response.

Pools the complete camera runs under ``responses/`` and produces one
three-panel figure plus the tables the reviewer asked for:

  A. commanded vs measured rotation, +delta and -delta branches with SD bars,
     the fitted gain over the linear region, and the assumed 0.09 deg/tick;
  B. deviation from that linear fit per command, which makes the SD bars
     readable, and exposes the small-command deadband and the 1000-tick
     endpoint roll-off;
  C. hysteresis: the return-to-zero offset by approach direction (+delta vs
     -delta), the quantity the drift periods were designed to probe.

Also prints the mm conversion for a given spool radius and the PCA9685
quantisation floor.

Run with::

    python generate_review_figure.py --spool-radius-mm 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze import add_block_relative_angle, add_trial_change, load_log
from generate_validation_summary import (
    DEFAULT_RESPONSES_DIR,
    PACKAGE_DIR,
    discover_runs,
)

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output" / "manuscript"

# Firmware: command -1000..+1000 maps onto PCA9685 ticks 130..500 (see
# Arduino/servo_pca9685_motors_controller/esp32_servo_pca9685.ino).
COMMAND_SPAN = 2000
PWM_SPAN = 500 - 130
COMMAND_UNITS_PER_PWM_STEP = COMMAND_SPAN / PWM_SPAN
NOMINAL_DEG_PER_TICK = 0.09          # the plotting convention, 1000 ticks := 90 deg
FIT_RANGE = (25, 500)                # linear region used for the calibrated gain

# Validated categorical slots (dataviz reference palette, light surface).
C_POS, C_NEG = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def load_pooled(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run in run_dirs:
        df = add_trial_change(add_block_relative_angle(load_log(run)))
        df["run"] = run.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df["response"] = pd.to_numeric(df["angle_response_deg"], errors="coerce")
    df["block_angle"] = pd.to_numeric(df["angle_block_zeroed"], errors="coerce")
    df["prev_target"] = df.groupby(["run", "block"])["target"].shift(1)
    return df


def branch_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per delta: mean and SD of the trial-local response for +delta and -delta."""
    proto = df[(df["mode"] == "protocol") & (df["target"] != 0)]
    rows = []
    for delta, sub in proto.groupby("delta"):
        pos = sub.loc[sub["target"] > 0, "response"].dropna()
        neg = sub.loc[sub["target"] < 0, "response"].dropna()
        rows.append({
            "delta": int(delta),
            "nominal_deg": delta * NOMINAL_DEG_PER_TICK,
            "pos_mean": pos.mean(), "pos_sd": pos.std(ddof=1), "n_pos": len(pos),
            "neg_mean": neg.mean(), "neg_sd": neg.std(ddof=1), "n_neg": len(neg),
        })
    t = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    t["mean_abs"] = (t["pos_mean"].abs() + t["neg_mean"].abs()) / 2
    t["asymmetry"] = t["pos_mean"].abs() - t["neg_mean"].abs()
    return t


def hysteresis_table(df: pd.DataFrame) -> pd.DataFrame:
    """Angle at commanded 0, split by whether it was reached from +delta or -delta."""
    zero = df[(df["target"] == 0) & df["prev_target"].notna() & (df["prev_target"] != 0)]
    rows = []
    for delta, sub in zero.groupby("delta"):
        a = sub.loc[sub["prev_target"] > 0, "block_angle"].dropna()
        b = sub.loc[sub["prev_target"] < 0, "block_angle"].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        rows.append({
            "delta": int(delta),
            "from_pos": a.mean(), "from_neg": b.mean(),
            "offset": a.mean() - b.mean(),
            "offset_se": np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)),
            "n": len(a) + len(b),
        })
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def fit_gain(t: pd.DataFrame) -> tuple[float, float]:
    """Through-origin slope (deg/tick) and R^2 over FIT_RANGE."""
    m = t["delta"].between(*FIT_RANGE)
    x = t.loc[m, "delta"].to_numpy(float)
    y = t.loc[m, "mean_abs"].to_numpy(float)
    k = float(x @ y / (x @ x))
    ss_res = float(((y - k * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return k, 1 - ss_res / ss_tot


def make_figure(t: pd.DataFrame, h: pd.DataFrame, k: float, r2: float, out_base: Path) -> None:
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.6))
    endpoint = t["delta"] == t["delta"].max()

    # A) commanded vs measured, both branches, magnitude scale
    x = t["delta"].to_numpy(float)
    xs = np.linspace(0, x.max(), 200)
    ax_a.plot(xs, xs * NOMINAL_DEG_PER_TICK, ls="--", lw=1.2, color=GRID, label="assumed 0.090 deg/tick")
    ax_a.plot(xs, xs * k, ls="-", lw=1.4, color=INK_2, label=f"fit {FIT_RANGE[0]}-{FIT_RANGE[1]}: {k:.4f} deg/tick")
    for col, sd, color, marker, label in [
        ("pos_mean", "pos_sd", C_POS, "o", "+delta"),
        ("neg_mean", "neg_sd", C_NEG, "s", "-delta"),
    ]:
        y = t[col].abs()
        ax_a.errorbar(x[~endpoint], y[~endpoint], yerr=t[sd][~endpoint], fmt=marker, ms=6, lw=0,
                      elinewidth=1.2, capsize=3, color=color, mec="white", mew=0.8, label=f"{label} (mean +/- SD, n=18)")
        ax_a.errorbar(x[endpoint], y[endpoint], yerr=t[sd][endpoint], fmt=marker, ms=6, lw=0,
                      elinewidth=1.2, capsize=3, color=color, mfc="white", mew=1.4)
    ax_a.annotate("1000-tick endpoint\n(excluded from fit)", xy=(x[endpoint][0], t.loc[endpoint, "mean_abs"].iloc[0]),
                  xytext=(700, 40), textcoords="data", fontsize=8, color=INK_2, ha="center",
                  arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8, "shrinkB": 6})
    ax_a.set_xlabel("command (ticks)")
    ax_a.set_ylabel("measured rotation, |angle| (deg)")
    ax_a.set_title(f"A. Command vs measured rotation  (R$^2$ = {r2:.4f})", loc="left")
    ax_a.set_xlim(0, 1050)
    ax_a.set_ylim(0, 95)
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")

    # B) deviation from the linear fit, log x so small commands are readable
    for col, sd, color, marker, label in [
        ("pos_mean", "pos_sd", C_POS, "o", "+delta"),
        ("neg_mean", "neg_sd", C_NEG, "s", "-delta"),
    ]:
        dev = t[col].abs() - x * k
        ax_b.errorbar(x, dev, yerr=t[sd], fmt=f"{marker}-", ms=6, lw=1.2, elinewidth=1.2, capsize=3,
                      color=color, mec="white", mew=0.8, label=label)
    ax_b.axhline(0, color=INK_2, lw=1.0)
    ax_b.axvline(COMMAND_UNITS_PER_PWM_STEP, color=INK_2, lw=0.9, ls=":")
    ax_b.text(COMMAND_UNITS_PER_PWM_STEP * 1.08, -13.2,
              f"1 PWM step = {COMMAND_UNITS_PER_PWM_STEP:.1f} ticks\n({COMMAND_UNITS_PER_PWM_STEP * k:.2f} deg)",
              fontsize=8, color=INK_2, va="bottom")
    ax_b.set_xscale("log")
    ax_b.set_xticks(x)
    ax_b.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_b.set_xlabel("command (ticks, log scale)")
    ax_b.set_ylabel("measured - linear fit (deg)")
    ax_b.set_title("B. Deviation from linear gain", loc="left")
    ax_b.set_ylim(-13.5, 3.5)
    ax_b.legend(frameon=False, fontsize=8, loc="upper left")

    # C) hysteresis: return-to-zero offset with SE, against the repeatability band
    rep = float(np.sqrt(((t["pos_sd"] ** 2 + t["neg_sd"] ** 2) / 2).mean()))
    hx = h["delta"].to_numpy(float)
    ax_c.axhspan(-rep, rep, color=GRID, alpha=0.5, lw=0, label=f"+/- pooled repeatability SD ({rep:.2f} deg)")
    ax_c.axhline(0, color=INK_2, lw=1.0)
    ax_c.errorbar(hx, h["offset"], yerr=h["offset_se"], fmt="D-", ms=6, lw=1.2, elinewidth=1.2, capsize=3,
                  color=INK, mec="white", mew=0.8, label="angle at 0 after +delta minus after -delta (+/- SE)")
    ax_c.set_xscale("log")
    ax_c.set_xticks(hx)
    ax_c.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_c.set_xlabel("command (ticks, log scale)")
    ax_c.set_ylabel("return-to-zero offset (deg)")
    ax_c.set_title("C. Hysteresis by approach direction", loc="left")
    ax_c.set_ylim(-1.6, 1.6)
    ax_c.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def print_report(t: pd.DataFrame, h: pd.DataFrame, k: float, r2: float, radius_mm: float, runs: list[Path]) -> None:
    mm_per_tick = np.radians(k) * radius_mm
    print(f"runs pooled: {len(runs)}")
    print(f"calibrated gain over {FIT_RANGE[0]}-{FIT_RANGE[1]} ticks: {k:.5f} deg/tick  (R^2 {r2:.5f});"
          f" assumed convention {NOMINAL_DEG_PER_TICK:.3f}")
    print()
    print("per command amplitude (protocol samples, 18 per direction):")
    print(t[["delta", "nominal_deg", "pos_mean", "pos_sd", "neg_mean", "neg_sd", "mean_abs", "asymmetry"]]
          .round(3).to_string(index=False))
    print()
    print("hysteresis (return-to-zero offset by approach direction):")
    print(h.round(3).to_string(index=False))
    print(f"  max |offset| = {h['offset'].abs().max():.3f} deg")
    print()
    print(f"cable displacement at the spool, r = {radius_mm:g} mm:")
    print(f"  1 tick                       {mm_per_tick:.5f} mm")
    print(f"  1 PWM step ({COMMAND_UNITS_PER_PWM_STEP:.2f} ticks, {COMMAND_UNITS_PER_PWM_STEP * k:.3f} deg)"
          f"   {COMMAND_UNITS_PER_PWM_STEP * mm_per_tick:.4f} mm  <- quantisation floor")
    for ticks in (10, 15, 25):
        row = t[t["delta"] == ticks]
        meas = f"   measured {np.radians(row['mean_abs'].iloc[0]) * radius_mm:.4f} mm" if not row.empty else ""
        print(f"  {ticks:2d} ticks (linear {ticks * k:.2f} deg)   {ticks * mm_per_tick:.4f} mm{meas}")
    full = t.loc[t["delta"] == t["delta"].max(), "mean_abs"].iloc[0]
    print(f"  full +/-1000 command         {np.radians(full) * radius_mm:.3f} mm each way (measured {full:.2f} deg)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--spool-radius-mm", type=float, default=4.0, help="Spool radius for the mm conversion (default 4)")
    p.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()

    runs = discover_runs(args.responses_dir)
    df = load_pooled(runs)
    t = branch_table(df)
    h = hysteresis_table(df)
    k, r2 = fit_gain(t)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.output_dir / "review_branch_table.csv", index=False)
    h.to_csv(args.output_dir / "review_hysteresis_table.csv", index=False)
    make_figure(t, h, k, r2, args.output_dir / "motor_validation_review_figure")
    print_report(t, h, k, r2, args.spool_radius_mm, runs)
    print(f"\nfigure: {args.output_dir / 'motor_validation_review_figure.png'} (+ .svg)")


if __name__ == "__main__":
    main()
