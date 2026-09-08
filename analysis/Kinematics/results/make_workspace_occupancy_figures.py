"""Group-level workspace occupancy heatmaps + radial profile from EXISTING results.

Reads the saved per-segment time-bin tables
``results/<G>/csv/trajectories/trajectory_time_bins.csv`` and writes, without
re-running tracking, for each member group (``N_E``, ``L_E``: one setup each) and
for the combined label (``L_N_E``: N vs L side by side):

    results/<G>/figures/trajectories/movement_orientation/
        xy_occupancy_heatmap_radial_profile_by_setup.png
        xy_occupancy_heatmap_by_setup_finger.png
    results/<G>/csv/trajectories/workspace_occupancy_*.csv

Usage::

    python analysis/Kinematics/results/make_workspace_occupancy_figures.py
    python ... --groups N_E L_E          # subset
    python ... --dpi 200 --bin-cm 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = RESULTS_ROOT.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import kinematics_analysis as ka  # noqa: E402

DEFAULT_GROUPS = ("N_E", "L_E", "L_N_E")
USECOLS = [
    "subject_id", "experiment_group", "workspace_setup", "workspace_width_cm", "workspace_height_cm",
    "finger_condition", "stiffness_value", "trajectory_time_bin",
    "x_workspace_cm", "y_workspace_cm", "r_workspace_cm",
    "radial_velocity_cm_s", "tangential_velocity_cm_s",
]


def load_member(member: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if member not in cache:
        path = RESULTS_ROOT / member / "csv" / "trajectories" / "trajectory_time_bins.csv"
        if not path.exists():
            raise FileNotFoundError(f"{member}: missing {path}")
        cache[member] = pd.read_csv(path, usecols=USECOLS)
        print(f"{member}: {len(cache[member])} time-bin rows, {cache[member]['subject_id'].nunique()} participants")
    return cache[member]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--bin-cm", type=float, default=ka.OCCUPANCY_BIN_CM)
    args = parser.parse_args(argv)

    cache: dict[str, pd.DataFrame] = {}
    for group in args.groups:
        members = ka.COMBINED_EXPERIMENT_GROUPS.get(group, (group,))
        time_bins = pd.concat([load_member(m, cache) for m in members], ignore_index=True)
        out_root = RESULTS_ROOT / group / "csv" / "trajectories"
        fig_dir = RESULTS_ROOT / group / "figures" / "trajectories" / "movement_orientation"
        paths = ka.save_workspace_occupancy_figures(
            out_root, time_bins, fig_dpi=args.dpi, bin_cm=args.bin_cm, fig_dir=fig_dir
        )
        for p in paths:
            print("  wrote", p.relative_to(RESULTS_ROOT))
        manifest = out_root / "workspace_occupancy_figure_manifest.csv"
        if manifest.exists():
            manifest.unlink()
        summary = out_root / "workspace_occupancy_extent_by_setup.csv"
        if summary.exists():
            print(pd.read_csv(summary).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
