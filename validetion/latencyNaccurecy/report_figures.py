"""Render the analysis as saved figures + tables (PNG/CSV) for one session.

All plotting uses the non-interactive Agg backend so it runs head-less and never
needs a display or any network access. Figures are intentionally compact
(dpi=140) to keep the Results folder small.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import session_report as SR  # noqa: E402

_FINGER_ORDER = ["index", "middle", "ring", "pinky", "thumb"]
_DPI = 140


def _order(summary: pd.DataFrame) -> pd.DataFrame:
    key = {f: i for i, f in enumerate(_FINGER_ORDER)}
    return summary.assign(_k=summary["finger"].map(lambda f: key.get(f, 99))).sort_values("_k").drop(columns="_k")


def _norm(a) -> np.ndarray:
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)


def fig_summary_table(summary: pd.DataFrame, path: Path) -> None:
    """Render the per-finger summary as a table image."""
    view = _order(summary)[SR.SUMMARY_VIEW].copy()
    num = view.select_dtypes("number").columns
    view[num] = view[num].round(2)
    fig, ax = plt.subplots(figsize=(min(22, 1.4 * len(view.columns)), 1.0 + 0.5 * len(view)))
    ax.axis("off")
    tbl = ax.table(cellText=view.astype(str).values,
                   colLabels=[c.replace("_", "\n") for c in view.columns],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.6)
    ax.set_title("Per-finger latency & accuracy summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def fig_detection_accuracy(summary: pd.DataFrame, path: Path) -> None:
    s = _order(summary)
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, s["vision_detection_rate"], 0.4, label="detection rate", color="#0072B2")
    ax.bar(x + 0.2, s["vision_accuracy_R2"], 0.4, label="accuracy R^2 (2-D affine)", color="#D55E00")
    ax.set_xticks(x); ax.set_xticklabels(s["finger"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("fraction / R^2")
    ax.set_title("Vision detection: rate & accuracy per finger")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=_DPI); plt.close(fig)


def fig_latencies(summary: pd.DataFrame, path: Path) -> None:
    s = _order(summary)
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [("hand_to_vision_latency_ms", "hand->vision", "#0072B2"),
            ("hand_to_tactor_latency_ms", "hand->tactor", "#009E73"),
            ("hand_to_motor_latency_ms", "hand->motor", "#D55E00")]
    w = 0.26
    for i, (col, lab, c) in enumerate(cols):
        vals = s[col].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * w, np.nan_to_num(vals), w, label=lab, color=c)
        for xi, v in zip(x + (i - 1) * w, vals):
            if not np.isfinite(v):
                ax.text(xi, 2, "n/a", ha="center", va="bottom", fontsize=7, rotation=90, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(s["finger"]); ax.set_ylabel("latency (ms)")
    ax.set_title("Estimated latencies per finger (gated by correlation)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=_DPI); plt.close(fig)


def fig_object_tracking(pairs: dict, summary: pd.DataFrame, path: Path) -> None:
    """One panel per finger: logged object motion vs video-detected object."""
    s = _order(summary)
    n = len(s)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.9 * n), squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], s.iterrows()):
        pr = pairs.get(row["pair"])
        if pr is None:
            continue
        sig, trk = pr["signals"], pr["tracking"]
        ax.plot(trk["t"], _norm(trk["object_x"]), lw=1, label="tracking object_x (sim)")
        ax.plot(sig["t"], _norm(sig["obj_x"]), lw=1, alpha=0.8, label="video object_x (camera)")
        ax.set_ylabel(f"{row['finger']}\n(z-x)")
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Logged vs video object position per finger", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=_DPI); plt.close(fig)


def fig_motor(log, cmd_ack: pd.DataFrame, path: Path) -> bool:
    """Command->ack histogram + commanded vs applied trace. Returns False if no log."""
    if log.df.empty or cmd_ack.empty:
        return False
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].hist(cmd_ack["latency_ms"], bins=40, color="#0072B2")
    ax[0].set_title(f"Firmware command->ack latency\n(median {cmd_ack['latency_ms'].median():.1f} ms)")
    ax[0].set_xlabel("ms")
    mcol = log.motor_columns[0] if log.motor_columns else None
    if mcol:
        cmd, ack = log.commanded(), log.acknowledged()
        ax[1].plot(cmd["timestamp"], cmd[mcol], lw=1, label=f"commanded {mcol}")
        ax[1].plot(ack["timestamp"], ack[mcol], lw=1, alpha=0.8, label=f"applied {mcol}")
        ax[1].set_title("Motor: commanded vs applied"); ax[1].legend()
        ax[1].tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=_DPI); plt.close(fig)
    return True


def render_all(rep: dict, fig_dir: Path) -> list[str]:
    """Write every figure for a session; returns the list of files written."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary, cmd_ack, log, pairs = rep["summary"], rep["cmd_ack"], rep["log"], rep["pairs"]
    written = []
    fig_summary_table(summary, fig_dir / "summary_table.png"); written.append("summary_table.png")
    fig_detection_accuracy(summary, fig_dir / "detection_accuracy.png"); written.append("detection_accuracy.png")
    fig_latencies(summary, fig_dir / "latencies.png"); written.append("latencies.png")
    fig_object_tracking(pairs, summary, fig_dir / "object_tracking.png"); written.append("object_tracking.png")
    if fig_motor(log, cmd_ack, fig_dir / "motor_command_response.png"):
        written.append("motor_command_response.png")
    return written
