"""Render the group-level movement-orientation XY figures from EXISTING results.

Reads each group's already-saved ``results/<G>/csv/trajectories/subject_xy_trajectory.csv``
(written by a previous notebook run) and writes, without re-running tracking:

    results/<G>/figures/trajectories/movement_orientation/
        all_xy_trajectories_with_finger_average_experiment_group_<G>.png
        all_xy_trajectories_with_stiffness_average_experiment_group_<G>.png
        median_xy_trajectory_by_finger_experiment_group_<G>.png

for ``N_E``, ``L_E`` and the combined ``L_N_E``. The combined figures are built
from the union of the two member groups so all three share one axis scale.

Usage (from anywhere)::

    python analysis/Kinematics/results/make_group_xy_trajectory_figures.py
    python ... --groups N_E L_E            # subset
    python ... --dpi 200
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


def load_group_table(group: str) -> pd.DataFrame:
    """The group's saved subject_xy_trajectory table; combined labels union their members."""
    members = ka.COMBINED_EXPERIMENT_GROUPS.get(group, (group,))
    frames = []
    for member in members:
        path = RESULTS_ROOT / member / "csv" / "trajectories" / "subject_xy_trajectory.csv"
        if not path.exists():
            raise FileNotFoundError(f"{group}: missing {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    written: list[Path] = []
    for group in args.groups:
        traj = load_group_table(group)
        fig_dir = RESULTS_ROOT / group / "figures" / "trajectories" / "movement_orientation"
        # Filter the table to this group's scope only. The helper enumerates every
        # experiment group in the table plus the combined label; for a member group
        # the table holds one group, for L_N_E it holds both, so keep only the
        # figures tagged with the requested label.
        paths = ka.save_group_xy_trajectory_figures(
            RESULTS_ROOT / group / "csv", traj, fig_dpi=args.dpi, fig_dir=fig_dir
        )
        tag = f"experiment_group_{ka.sanitize_name(group)}.png"
        for p in paths:
            if p.name.endswith(tag):
                written.append(p)
            else:
                p.unlink(missing_ok=True)
        manifest = RESULTS_ROOT / group / "csv" / "group_xy_trajectory_figure_manifest.csv"
        if manifest.exists():
            manifest.unlink()
        # Combined selections also get the N-vs-L side-by-side version (in cm).
        if group in ka.COMBINED_EXPERIMENT_GROUPS:
            for aligned in (True, False):
                written.extend(
                    ka.save_group_xy_trajectory_by_setup_figure(
                        RESULTS_ROOT / group / "csv", traj, fig_dpi=args.dpi, fig_dir=fig_dir, aligned=aligned
                    )
                )
        print(f"{group}: {traj['subject_id'].nunique()} participants, {len(traj)} rows")
    for p in written:
        print("  wrote", p.relative_to(RESULTS_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
