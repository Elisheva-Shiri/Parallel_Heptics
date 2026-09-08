"""Render the full eight-figure plot set for the accepted configs of a sweep run.

Standalone, parallel, and crash-resumable -- the same plot stage as the notebook
(`PLOT_SCOPE="accepted"`), but runnable headless/in the background so it does not
tie up a Jupyter kernel for the several hours the full render takes.

It resumes an existing run: it reads that run's frozen sweep config and its
`tables/sweep_results.csv`, selects the accepted configs, and renders any whose
`.done` marker is missing. Re-running continues where it left off.

Usage (from anywhere in the repo):
    python analysis/IK_optimizetion/render_accepted_plots.py [--run RUN_ID]
        [--scope accepted|all|subset] [--workers N] [--dpi 150] [--limit N]

Defaults: most-recent run, scope=accepted, workers=min(8, cpu_count).
Tip: pause Dropbox sync while this runs -- it writes ~8 PNGs per config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in [start, *start.parents]:
        if (path / "kinematics" / "unified_ik_starter.py").exists():
            return path
    raise FileNotFoundError("Could not find repo root containing kinematics/unified_ik_starter.py")


REPO_ROOT = find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.IK_optimizetion.ik_optimization_utils import (  # noqa: E402
    plot_config_done,
    render_and_checkpoint,
    resolve_continue_run_dir,
)


def select_plot_rows(results: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return results.copy()
    if scope == "accepted":
        return results[results["accepted"]].copy()
    raise ValueError(f"scope must be 'accepted' or 'all' for this script, got {scope!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="run_* directory name (default: most recent)")
    parser.add_argument("--scope", default="accepted", choices=["accepted", "all"])
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None, help="cap configs rendered (debug)")
    args = parser.parse_args()

    results_root = REPO_ROOT / "analysis" / "IK_optimizetion" / "results"
    run_dir = resolve_continue_run_dir(results_root, args.run)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config["plot_generation"] = {
        "plot_scope": args.scope,
        "max_full_plot_configs": args.limit,
        "plot_dpi": args.dpi,
    }

    sweep_csv = run_dir / "tables" / "sweep_results.csv"
    if not sweep_csv.exists():
        raise FileNotFoundError(f"No sweep_results.csv in {run_dir}; run the sweep first.")
    results = pd.read_csv(sweep_csv)
    plot_rows = select_plot_rows(results, args.scope)
    if args.limit is not None:
        plot_rows = plot_rows.head(int(args.limit)).copy()

    all_config_dir = run_dir / "plots" / "all_configurations"
    all_config_dir.mkdir(parents=True, exist_ok=True)

    row_dicts = [row.to_dict() for _, row in plot_rows.iterrows()]
    todo = [d for d in row_dicts if not plot_config_done(d, str(all_config_dir))]

    print(f"Run: {run_dir.name}")
    print(f"Scope '{args.scope}': {len(plot_rows):,} / {len(results):,} configs")
    print(
        f"Plots: {len(row_dicts) - len(todo):,} already complete, "
        f"{len(todo):,} to render  (workers={args.workers}, dpi={args.dpi})"
    )
    if not todo:
        print("Nothing to do -- all targeted plot sets already complete.")
        return

    t0 = time.perf_counter()
    completed = 0
    tasks = [(d, config, str(all_config_dir)) for d in todo]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(render_and_checkpoint, t) for t in tasks]
        for _ in as_completed(futures):
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else float("nan")
                remaining = (len(tasks) - completed) / rate if rate > 0 else float("nan")
                print(
                    f"  {completed:>6,}/{len(tasks):,} saved  "
                    f"elapsed={elapsed/60:,.1f}min  est_remaining={remaining/60:,.1f}min",
                    flush=True,
                )

    n_done = sum(plot_config_done(d, str(all_config_dir)) for d in row_dicts)
    print(f"Done. Completed plot-set folders: {n_done:,} / {len(row_dicts):,}")


if __name__ == "__main__":
    main()
