"""Regenerate the probing-direction rose figures from saved probing CSVs.

Reads ``probing_direction_per_setup_long_summary.csv`` and
``probing_direction_per_setup_counts_success_differences.csv`` from a probing
results folder, overlays the device reach outline (mean of the OptiTrack and
camera tactor validations) and writes the compact and extended figures plus
the supporting CSVs next to them.  Run from the repository root:

    python analysis/success_factors/make_probing_direction_rose.py
    python analysis/success_factors/make_probing_direction_rose.py --copy-to paper/.../media/media
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.success_factors import probing  # noqa: E402

# The paper's Fig. 14 numbers come from the filtered 40-subject run
# (results_fillter); fall back to the plain results folder if it is absent.
_CANDIDATE_RESULTS = [
    HERE.parent / "results_fillter" / "probing" / "L_N_E_center_to_side",
    HERE.parent / "results" / "probing" / "L_N_E_center_to_side",
]
DEFAULT_RESULTS = next((p for p in _CANDIDATE_RESULTS if p.exists()), _CANDIDATE_RESULTS[0])
TABLE_FILES = [
    "probing_direction_per_setup_long_summary.csv",
    "probing_direction_per_setup_counts_success_differences.csv",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="probing results folder with the direction CSVs")
    parser.add_argument("--camera-group", choices=["with_finger", "without_finger"], default="with_finger")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--copy-to", type=Path, default=None, help="optional folder to copy the PNG/PDF outputs into")
    args = parser.parse_args(argv)

    tables: dict[str, pd.DataFrame] = {}
    for name in TABLE_FILES:
        path = args.results / name
        if not path.exists():
            raise SystemExit(f"missing {path}")
        tables[path.stem] = pd.read_csv(path)

    outline = probing.load_device_reach_outline(camera_group=args.camera_group)
    print(outline[["direction", "optitrack_mm", "camera_with_finger_mm", "camera_without_finger_mm", "mean_mm"]].round(2).to_string(index=False))
    relation = probing.compute_direction_reach_relation(tables["probing_direction_per_setup_long_summary"], outline)
    print(relation.round(3).to_string(index=False))

    paths = probing.save_direction_rose_figures(args.results, tables, outline=outline, fig_dpi=args.dpi)
    for p in paths:
        print(p)
        if args.copy_to:
            args.copy_to.mkdir(parents=True, exist_ok=True)
            for src in [p, p.with_suffix(".pdf")]:
                if src.exists():
                    shutil.copy2(src, args.copy_to / src.name)
                    print("  ->", args.copy_to / src.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
