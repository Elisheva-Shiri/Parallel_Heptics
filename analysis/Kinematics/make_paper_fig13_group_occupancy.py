"""Paper Fig. 13 replacement: group-level workspace occupancy per setup (single column).

Reviewer: "replace [single-subject examples] with group-level 2-D occupancy heatmaps
(or median trajectory + IQR bands) per setup".

Layout (one ICRA column, 3.5 in wide):
  top row    : occupancy heatmap, natural (N, 60x45 cm) | air-slide (L, 80x60 cm),
               equal weight per participant, shared sqrt colour scale, dashed = workspace
  bottom row : median radial distance from the workspace centre vs normalised time
               within a stiffness segment, IQR band across participants, one curve per setup

Reads the saved time-bin tables results/{N_E,L_E}/csv/trajectories/trajectory_time_bins.csv
(no tracking re-run). Writes PNG + PDF into
results/L_N_E/figures/trajectories/movement_orientation/ and, with --paper-media, copies the
PNG next to the other paper images.

    .venv/Scripts/python analysis/Kinematics/make_paper_fig13_group_occupancy.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import kinematics_analysis as ka  # noqa: E402

RESULTS_ROOT = HERE / "results"
USECOLS = [
    "subject_id", "experiment_group", "workspace_setup", "workspace_width_cm", "workspace_height_cm",
    "finger_condition", "stiffness_value", "trajectory_time_bin",
    "x_workspace_cm", "y_workspace_cm", "r_workspace_cm",
    "radial_velocity_cm_s", "tangential_velocity_cm_s",
]
SETUP_NAME = {"N": "Natural", "L": "Air-slide"}
DEFAULT_PAPER_MEDIA = (HERE.parents[1] / "paper" / "icra2027_paper" / "icra2027_latex_transfer"
                       / "media" / "media")


def load_time_bins(members=("N_E", "L_E")) -> pd.DataFrame:
    frames = []
    for m in members:
        path = RESULTS_ROOT / m / "csv" / "trajectories" / "trajectory_time_bins.csv"
        frames.append(pd.read_csv(path, usecols=USECOLS))
    d = ka._drop_lost_tracking_corner_rows(pd.concat(frames, ignore_index=True)).copy()
    for c in ["x_workspace_cm", "y_workspace_cm", "workspace_width_cm", "workspace_height_cm"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["x_workspace_cm", "y_workspace_cm"])


def make_figure(d: pd.DataFrame, out_stem: Path, *, bin_cm: float, dpi: int,
                width_in: float = 3.5, annotate: str | None = None,
                heatmaps_only: bool = False, interpolation: str = "nearest") -> list[Path]:
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42,
    })
    setups = [s for s in ["N", "L"] if s in set(d["workspace_setup"].astype(str))]
    half = ka.OCCUPANCY_HALF_EXTENT_CM
    grids = {s: ka._occupancy_grid(d[d["workspace_setup"].astype(str) == s], bin_cm, half) for s in setups}
    vmax = max(float(np.nanpercentile(H, 99.5)) for H, _, _ in grids.values()) or 1.0
    tables = ka.compute_workspace_occupancy_analysis(d, n_time_bins=50)
    rp_all = tables["radial_profile_by_setup"]

    if heatmaps_only:
        fig = plt.figure(figsize=(width_in, width_in * 0.5), layout="constrained")
        gs = fig.add_gridspec(1, len(setups))
        heat_axes = [fig.add_subplot(gs[0, j]) for j in range(len(setups))]
        ax_r = None
    else:
        fig = plt.figure(figsize=(width_in, width_in * 0.98), layout="constrained")
        gs = fig.add_gridspec(2, len(setups), height_ratios=[0.95, 0.85])
        heat_axes = [fig.add_subplot(gs[0, j]) for j in range(len(setups))]
        ax_r = fig.add_subplot(gs[1, :])

    im = None
    for ax, s in zip(heat_axes, setups):
        sub = d[d["workspace_setup"].astype(str) == s]
        H, xe, ye = grids[s]
        im = ax.imshow(H.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="magma",
                       norm=PowerNorm(gamma=0.5, vmin=0, vmax=vmax), aspect="equal",
                       interpolation=interpolation)
        w = float(sub["workspace_width_cm"].dropna().iloc[0])
        h = float(sub["workspace_height_cm"].dropna().iloc[0])
        ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, fill=False, ls="--", lw=0.8, ec="white"))
        ax.axhline(0, color="white", lw=0.3, alpha=0.5)
        ax.axvline(0, color="white", lw=0.3, alpha=0.5)
        ax.set_xlim(-44, 44)
        ax.set_ylim(-34, 34)
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-30, -15, 0, 15, 30])
        ax.set_xlabel("X (cm)")
        n = int(sub["subject_id"].nunique())
        ax.set_title(f"{SETUP_NAME.get(s, s)}\n{w:g}×{h:g} cm, n = {n}", pad=2, linespacing=1.1)
        ax.tick_params(length=2, pad=1.5)
    heat_axes[0].set_ylabel("Y (cm)")
    for ax in heat_axes[1:]:
        ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=heat_axes, shrink=0.62 if heatmaps_only else 0.55, pad=0.02, aspect=14)
    cb.set_label("% of movement time per\n2×2 cm cell (sqrt scale)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2)

    for s in ([] if ax_r is None else setups):
        rp = rp_all[rp_all["workspace_setup"].astype(str) == s]
        c = ka.GROUP_TO_COLOR.get(s, "#444444")
        ax_r.fill_between(rp["time"], rp["q25"], rp["q75"], color=c, alpha=0.22, lw=0)
        ax_r.plot(rp["time"], rp["median"], color=c, lw=1.6, label=f"{SETUP_NAME.get(s, s)} (median, IQR)")
    if ax_r is not None:
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(bottom=0)
        ax_r.set_xlabel("Normalised time within a stiffness segment")
        ax_r.set_ylabel("Radial distance\nfrom centre (cm)")
        ax_r.grid(alpha=0.25, lw=0.5)
        ax_r.tick_params(length=2, pad=1.5)
        ax_r.legend(loc="upper left", frameon=False, handlelength=1.6)
        if annotate:
            ax_r.text(0.99, 0.04, annotate, transform=ax_r.transAxes, ha="right", va="bottom", fontsize=6.3)

    out_paths = []
    for ext in ("png", "pdf"):
        out = out_stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi)
        out_paths.append(out)
    plt.close(fig)
    return out_paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--bin-cm", type=float, default=ka.OCCUPANCY_BIN_CM)
    ap.add_argument("--width-in", type=float, default=3.5)
    ap.add_argument("--heatmaps-only", action="store_true", help="drop the radial-profile panel")
    ap.add_argument("--interpolation", default="nearest", help="imshow interpolation, e.g. bilinear")
    ap.add_argument("--annotate", default=None,
                    help="text placed in the profile panel, e.g. the between-setup test")
    ap.add_argument("--paper-media", type=Path, nargs="?", const=DEFAULT_PAPER_MEDIA, default=None,
                    help="copy the PNG into this folder as fig13_group_occupancy.png")
    args = ap.parse_args(argv)

    d = load_time_bins()
    out_dir = RESULTS_ROOT / "L_N_E" / "figures" / "trajectories" / "movement_orientation"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig13_group_occupancy_heatmaps_only" if args.heatmaps_only else "fig13_group_occupancy_paper"
    paths = make_figure(d, out_dir / stem, bin_cm=args.bin_cm, dpi=args.dpi,
                        width_in=args.width_in, annotate=args.annotate,
                        heatmaps_only=args.heatmaps_only, interpolation=args.interpolation)
    for p in paths:
        print("wrote", p)
    if args.paper_media is not None:
        args.paper_media.mkdir(parents=True, exist_ok=True)
        dst = args.paper_media / (stem.replace("_paper", "") + ".png")
        shutil.copyfile(paths[0], dst)
        print("copied", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
