"""Motor-validation results for the manuscript: tables, figure, LaTeX.

Pools the complete camera runs under ``responses/`` and writes everything the
paper needs into ``results/<selection>/`` using the same layout as the other
analysis notebooks::

    results/<selection>/
        csv/response/branch_table.csv          per amplitude, +delta / -delta branches
        csv/hysteresis/hysteresis_table.csv    return-to-zero offset by approach direction
        csv/resolution/resolution_table.csv    tick -> deg -> mm at the spool
        csv/summary/motor_validation_table.csv one merged table, one row per amplitude
        csv/summary/motor_validation_table.tex the same table as an in-column IEEEtran float
        csv/summary/calibration.json           gain, R^2, quantisation step, runs used
        figures/review/motor_validation_review_figure.{png,svg}
        figures/review/panel_{A,B,C}_*.{png,svg}

Driven by ``motor_validation_analysis.ipynb``; also runnable directly::

    python motor_validation.py --spool-radius-mm 4
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_RESULTS_ROOT = PACKAGE_DIR / "results"
DEFAULT_SELECTION = "all_runs"
DEFAULT_SPOOL_RADIUS_MM = 4.0

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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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
    """Per amplitude: mean and SD of the trial-local response for +delta and -delta."""
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
    t["error_pct"] = (t["mean_abs"] - t["nominal_deg"]).abs() / t["nominal_deg"] * 100
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


def pooled_repeatability_sd(t: pd.DataFrame) -> float:
    return float(np.sqrt(((t["pos_sd"] ** 2 + t["neg_sd"] ** 2) / 2).mean()))


def resolution_table(t: pd.DataFrame, k: float, radius_mm: float) -> pd.DataFrame:
    """Tick -> angle -> cable displacement at the spool, linear and measured."""
    mm_per_deg = np.radians(1.0) * radius_mm
    measured = dict(zip(t["delta"], t["mean_abs"], strict=True))
    rows = [{"item": "1 command tick", "ticks": 1.0, "linear_deg": k, "measured_deg": np.nan},
            {"item": "1 PWM step (quantisation floor)", "ticks": COMMAND_UNITS_PER_PWM_STEP,
             "linear_deg": COMMAND_UNITS_PER_PWM_STEP * k, "measured_deg": np.nan}]
    for ticks in (5, 10, 15, 25):
        rows.append({"item": f"{ticks} ticks", "ticks": float(ticks), "linear_deg": ticks * k,
                     "measured_deg": measured.get(ticks, np.nan)})
    top = int(t["delta"].max())
    rows.append({"item": f"full {top}-tick command", "ticks": float(top), "linear_deg": top * k,
                 "measured_deg": measured[top]})
    r = pd.DataFrame(rows)
    r["linear_mm"] = r["linear_deg"] * mm_per_deg
    r["measured_mm"] = r["measured_deg"] * mm_per_deg
    return r


def merged_table(t: pd.DataFrame, h: pd.DataFrame, radius_mm: float) -> pd.DataFrame:
    """One row per amplitude: both branches, error, hysteresis and cable travel."""
    m = t.merge(h[["delta", "offset", "offset_se"]], on="delta", how="left")
    m["cable_mm"] = np.radians(m["mean_abs"]) * radius_mm
    cols = ["delta", "nominal_deg", "pos_mean", "pos_sd", "neg_mean", "neg_sd",
            "mean_abs", "error_pct", "offset", "offset_se", "cable_mm", "n_pos", "n_neg"]
    return m[cols].rename(columns={"offset": "hysteresis_deg", "offset_se": "hysteresis_se"})


def _tex_num(value: float, fmt: str) -> str:
    """Format a number for LaTeX with a typographic minus."""
    text = format(value, fmt)
    return "$-$" + text[1:] if text.startswith("-") else text


def merged_table_latex(m: pd.DataFrame, n_per_direction: int = 18, label: str = "tab:motor_validation") -> str:
    """The merged table as a single-column IEEEtran float (needs booktabs + array).

    The caption is a bare title by request; every column definition, including
    what the +/- values are, lives in the body text.

    Single-column on purpose: in two-column mode a ``table*`` can only sit at the
    top of a page and never on the page it is written on, so it always drifts to
    the next page. This form stays next to the paragraph that cites it.
    """
    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\fontsize{7.5}{9}\selectfont",
        r"\caption{Motor validation: measured spool rotation per command amplitude, "
        r"pooled over three runs ($n = " + str(n_per_direction) + r"$ per direction)."
        r"}"
        + r"\label{" + label + "}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}r r r@{\,$\pm$\,}l r@{\,$\pm$\,}l r r r@{\,$\pm$\,}l r@{}}",
        r"\toprule",
        r"$\Delta$ & Nom. & \multicolumn{2}{c}{$+\Delta$} & \multicolumn{2}{c}{$-\Delta$} "
        r"& $|\bar\theta|$ & Error & \multicolumn{2}{c}{Hyst.} & Cable \\",
        r"(ticks) & ($^\circ$) & \multicolumn{2}{c}{($^\circ$)} & \multicolumn{2}{c}{($^\circ$)} "
        r"& ($^\circ$) & (\%) & \multicolumn{2}{c}{($^\circ$)} & (mm) \\",
        r"\midrule",
    ]
    for r in m.itertuples(index=False):
        lines.append(
            f"{r.delta} & {r.nominal_deg:.2f} & {_tex_num(r.pos_mean, '.2f')} & {r.pos_sd:.2f} "
            f"& {_tex_num(r.neg_mean, '.2f')} & {r.neg_sd:.2f} & {r.mean_abs:.2f} & {r.error_pct:.1f} "
            f"& {_tex_num(r.hysteresis_deg, '+.2f')} & {r.hysteresis_se:.2f} & {r.cable_mm:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_BRANCHES = [("pos_mean", "pos_sd", C_POS, "o", "+delta"), ("neg_mean", "neg_sd", C_NEG, "s", "-delta")]


def draw_panel_a(ax: plt.Axes, t: pd.DataFrame, k: float, r2: float) -> None:
    """Commanded vs measured rotation, both branches, calibrated gain."""
    endpoint = (t["delta"] == t["delta"].max()).to_numpy()
    x = t["delta"].to_numpy(float)
    xs = np.linspace(0, x.max(), 200)
    ax.plot(xs, xs * NOMINAL_DEG_PER_TICK, ls="--", lw=1.2, color=GRID, label="assumed 0.090 deg/tick")
    ax.plot(xs, xs * k, ls="-", lw=1.4, color=INK_2, label=f"fit {FIT_RANGE[0]}-{FIT_RANGE[1]}: {k:.4f} deg/tick")
    for col, sd, color, marker, label in _BRANCHES:
        y = t[col].abs().to_numpy()
        e = t[sd].to_numpy()
        ax.errorbar(x[~endpoint], y[~endpoint], yerr=e[~endpoint], fmt=marker, ms=6, lw=0, elinewidth=1.2,
                    capsize=3, color=color, mec="white", mew=0.8, label=f"{label} (mean +/- SD, n=18)")
        ax.errorbar(x[endpoint], y[endpoint], yerr=e[endpoint], fmt=marker, ms=6, lw=0, elinewidth=1.2,
                    capsize=3, color=color, mfc="white", mew=1.4)
    ax.annotate("1000-tick endpoint\n(excluded from fit)", xy=(x[endpoint][0], t.loc[endpoint, "mean_abs"].iloc[0]),
                xytext=(700, 40), textcoords="data", fontsize=8, color=INK_2, ha="center",
                arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8, "shrinkB": 6})
    ax.set_xlabel("command (ticks)")
    ax.set_ylabel("measured rotation, |angle| (deg)")
    ax.set_title(f"A. Command vs measured rotation  (R$^2$ = {r2:.4f})", loc="left")
    ax.set_xlim(0, 1050)
    ax.set_ylim(0, 95)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def draw_panel_b(ax: plt.Axes, t: pd.DataFrame, k: float) -> None:
    """Deviation from the linear fit; log x so the small commands are readable."""
    x = t["delta"].to_numpy(float)
    for col, sd, color, marker, label in _BRANCHES:
        ax.errorbar(x, t[col].abs() - x * k, yerr=t[sd], fmt=f"{marker}-", ms=6, lw=1.2, elinewidth=1.2,
                    capsize=3, color=color, mec="white", mew=0.8, label=label)
    ax.axhline(0, color=INK_2, lw=1.0)
    ax.axvline(COMMAND_UNITS_PER_PWM_STEP, color=INK_2, lw=0.9, ls=":")
    ax.text(COMMAND_UNITS_PER_PWM_STEP * 1.08, -13.2,
            f"1 PWM step = {COMMAND_UNITS_PER_PWM_STEP:.1f} ticks\n({COMMAND_UNITS_PER_PWM_STEP * k:.2f} deg)",
            fontsize=8, color=INK_2, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("command (ticks, log scale)")
    ax.set_ylabel("measured - linear fit (deg)")
    ax.set_title("B. Deviation from linear gain", loc="left")
    ax.set_ylim(-13.5, 3.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def draw_panel_c(ax: plt.Axes, h: pd.DataFrame, repeatability_sd: float) -> None:
    """Return-to-zero offset by approach direction against the repeatability band."""
    hx = h["delta"].to_numpy(float)
    ax.axhspan(-repeatability_sd, repeatability_sd, color=GRID, alpha=0.5, lw=0,
               label=f"+/- pooled repeatability SD ({repeatability_sd:.2f} deg)")
    ax.axhline(0, color=INK_2, lw=1.0)
    ax.errorbar(hx, h["offset"], yerr=h["offset_se"], fmt="D-", ms=6, lw=1.2, elinewidth=1.2, capsize=3,
                color=INK, mec="white", mew=0.8, label="angle at 0 after +delta minus after -delta (+/- SE)")
    ax.set_xscale("log")
    ax.set_xticks(hx)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("command (ticks, log scale)")
    ax.set_ylabel("return-to-zero offset (deg)")
    ax.set_title("C. Hysteresis by approach direction", loc="left")
    ax.set_ylim(-1.6, 1.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _save(fig: plt.Figure, out_base: Path) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_base.with_suffix(".png"), out_base.with_suffix(".svg")]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return paths


def make_figures(t: pd.DataFrame, h: pd.DataFrame, k: float, r2: float, out_dir: Path) -> dict[str, list[Path]]:
    """The three-panel review figure plus each panel on its own, for the paper."""
    rep = pooled_repeatability_sd(t)
    panels = {
        "panel_A_command_vs_measured": lambda ax: draw_panel_a(ax, t, k, r2),
        "panel_B_deviation_from_fit": lambda ax: draw_panel_b(ax, t, k),
        "panel_C_hysteresis": lambda ax: draw_panel_c(ax, h, rep),
    }
    written: dict[str, list[Path]] = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, draw in zip(axes, panels.values(), strict=True):
        draw(ax)
    fig.tight_layout()
    written["motor_validation_review_figure"] = _save(fig, out_dir / "motor_validation_review_figure")

    for name, draw in panels.items():
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        draw(ax)
        fig.tight_layout()
        written[name] = _save(fig, out_dir / name)
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    responses_dir: Path = DEFAULT_RESPONSES_DIR,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    selection: str = DEFAULT_SELECTION,
    spool_radius_mm: float = DEFAULT_SPOOL_RADIUS_MM,
    run_dirs: list[Path] | None = None,
) -> dict:
    """Compute everything and write the ``results/<selection>/`` tree. Returns the pieces."""
    runs = run_dirs or discover_runs(responses_dir)
    df = load_pooled(runs)
    t = branch_table(df)
    h = hysteresis_table(df)
    k, r2 = fit_gain(t)
    res = resolution_table(t, k, spool_radius_mm)
    m = merged_table(t, h, spool_radius_mm)
    latex = merged_table_latex(m, int(t["n_pos"].iloc[0]))

    out = results_root / selection
    csv = out / "csv"
    for sub in ("response", "hysteresis", "resolution", "summary"):
        (csv / sub).mkdir(parents=True, exist_ok=True)
    t.to_csv(csv / "response" / "branch_table.csv", index=False)
    h.to_csv(csv / "hysteresis" / "hysteresis_table.csv", index=False)
    res.to_csv(csv / "resolution" / "resolution_table.csv", index=False)
    m.to_csv(csv / "summary" / "motor_validation_table.csv", index=False)
    (csv / "summary" / "motor_validation_table.tex").write_text(latex + "\n", encoding="utf-8")
    calibration = {
        "runs": [p.name for p in runs],
        "fit_range_ticks": list(FIT_RANGE),
        "gain_deg_per_tick": k,
        "fit_r2": r2,
        "assumed_deg_per_tick": NOMINAL_DEG_PER_TICK,
        "pooled_repeatability_sd_deg": pooled_repeatability_sd(t),
        "max_abs_hysteresis_deg": float(h["offset"].abs().max()),
        "pwm_step_ticks": COMMAND_UNITS_PER_PWM_STEP,
        "pwm_step_deg": COMMAND_UNITS_PER_PWM_STEP * k,
        "pwm_step_mm": np.radians(COMMAND_UNITS_PER_PWM_STEP * k) * spool_radius_mm,
        "spool_radius_mm": spool_radius_mm,
    }
    (csv / "summary" / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    figures = make_figures(t, h, k, r2, out / "figures" / "review")

    return {"runs": runs, "df": df, "branch": t, "hysteresis": h, "resolution": res, "merged": m,
            "latex": latex, "calibration": calibration, "figures": figures, "results_dir": out}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--spool-radius-mm", type=float, default=DEFAULT_SPOOL_RADIUS_MM)
    p.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--selection", default=DEFAULT_SELECTION, help="Name of the results/<selection>/ folder")
    args = p.parse_args()

    r = run(args.responses_dir, args.results_root, args.selection, args.spool_radius_mm)
    c = r["calibration"]
    print(f"runs pooled: {len(r['runs'])}")
    print(f"gain {c['gain_deg_per_tick']:.5f} deg/tick (R^2 {c['fit_r2']:.5f}); "
          f"repeatability SD {c['pooled_repeatability_sd_deg']:.2f} deg; "
          f"max |hysteresis| {c['max_abs_hysteresis_deg']:.2f} deg; "
          f"PWM step {c['pwm_step_ticks']:.2f} ticks = {c['pwm_step_deg']:.3f} deg = {c['pwm_step_mm']:.4f} mm")
    print()
    print(r["merged"].round(3).to_string(index=False))
    print()
    print(r["latex"])
    print(f"\nresults: {r['results_dir']}")


if __name__ == "__main__":
    main()
