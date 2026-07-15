"""One command to analyse a data run (or all of them) and SAVE everything.

For each run folder under ``Data/`` it runs the full pipeline - region
separation, per-frame detection, latency/accuracy analysis, motor-log parsing -
and writes all artefacts into ``Results/<same name>/``:

    Results/<run>/
        csv/        per_finger_summary.csv, command_ack_latency.csv,
                    motor_commands_parsed.csv, sessions_meta.csv
        signals/    signals_pair_*.csv  (per-frame detections) + regions.json
        figures/    summary_table.png, detection_accuracy.png, latencies.png,
                    object_tracking.png, motor_command_response.png, regions_overlay.png
        video/      qc_overlay.mp4 + qc_frame.png  (+ region_*.mp4 with --export-regions)
        run_info.txt

Everything runs locally - **no network/downloads** - so it is safe on a slow
connection. The QC video is a short clip by default; full per-region videos are
only written with ``--export-regions`` (they are large).

Usage:
    python run_analysis.py <run-name-or-path>      # one run
    python run_analysis.py --all                   # every run under Data/
    python run_analysis.py --all --results-root /some/where --stride 2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from build_signals import build, discover_pairs
from session_report import build_summary, SUMMARY_VIEW
from report_figures import render_all
from motor_commands import find_motor_log, parse_motor_commands, empty_log
from side_camera_separator import Regions
import qc_overlay

HERE = Path(__file__).resolve().parent


def _load_colors(session_dir: Path):
    cal = session_dir / "calibration.json"
    if cal.exists():
        from calibrate import Calibration
        return Calibration.load(cal).colors
    return None


def analyse_run(session_dir: Path, results_root: Path, stride: int = 2,
                qc_seconds: float = 5.0, export_regions: bool = False,
                rebuild: bool = False) -> Path:
    """Analyse one run folder and write everything under Results/<name>/."""
    name = session_dir.name
    res = results_root / name
    sig_dir = res / "signals"
    csv_dir = res / "csv"
    fig_dir = res / "figures"
    vid_dir = res / "video"
    for d in (sig_dir, csv_dir, fig_dir, vid_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name} ===")
    # 1. region separation + per-frame signals (cached in Results/<name>/signals)
    if rebuild or not (sig_dir / "sessions_meta.csv").exists():
        build(session_dir, stride=stride, out_dir=sig_dir)
    else:
        print(f"[run] reusing cached signals in {sig_dir}")

    # 2. latency / accuracy analysis
    rep = build_summary(session_dir, out_dir=sig_dir)
    summary = rep["summary"]

    # 3. tables (CSV)
    summary[SUMMARY_VIEW].to_csv(csv_dir / "per_finger_summary.csv", index=False)
    summary.to_csv(csv_dir / "per_finger_summary_full.csv", index=False)
    rep["cmd_ack"].to_csv(csv_dir / "command_ack_latency.csv", index=False)
    rep["log"].df.to_csv(csv_dir / "motor_commands_parsed.csv", index=False)
    if (sig_dir / "sessions_meta.csv").exists():
        shutil.copy(sig_dir / "sessions_meta.csv", csv_dir / "sessions_meta.csv")

    # 4. figures
    written = render_all(rep, fig_dir)
    if (sig_dir / "regions_overlay.png").exists():
        shutil.copy(sig_dir / "regions_overlay.png", fig_dir / "regions_overlay.png")
        written.append("regions_overlay.png")

    # 5. QC video (short, low size) on the longest pair, reusing the saved layout
    pairs = discover_pairs(session_dir)
    longest = max(pairs, key=lambda p: (p / "side_camera.mp4").stat().st_size)
    regions = Regions.load(sig_dir / "regions.json") if (sig_dir / "regions.json").exists() else None
    qc = qc_overlay.render(longest / "side_camera.mp4", vid_dir, regions=regions,
                           colors=_load_colors(session_dir), seconds=qc_seconds)
    print(f"[run] QC video: {qc['video']}")

    # 6. optional full per-region videos (large)
    if export_regions and regions is not None:
        from side_camera_separator import export_region_videos
        export_region_videos(longest / "side_camera.mp4", regions, vid_dir)

    # 7. run info
    log_path = find_motor_log(session_dir)
    (res / "run_info.txt").write_text(
        f"run: {name}\n"
        f"pairs: {len(pairs)}\n"
        f"motor log: {log_path.name if log_path else 'NONE (motor latency = n/a)'}\n"
        f"fingers analysed: {', '.join(summary['finger'].tolist())}\n"
        f"figures: {', '.join(written)}\n"
        f"stride: {stride}\n",
        encoding="utf-8",
    )
    print(f"[run] saved -> {res}")
    return res


def discover_runs(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir()
                  if p.is_dir() and any(p.glob("pair_*/side_camera.mp4")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyse data run(s) and save figures/tables/videos.")
    ap.add_argument("run", nargs="?", help="run folder name (under Data/) or a full path")
    ap.add_argument("--all", action="store_true", help="analyse every run under the data root")
    ap.add_argument("--data-root", default=str(HERE / "Data"), help="folder containing the runs")
    ap.add_argument("--results-root", default=str(HERE / "Results"), help="where to write Results/<name>/")
    ap.add_argument("--stride", type=int, default=2, help="frame subsample for extraction (default 2)")
    ap.add_argument("--qc-seconds", type=float, default=5.0, help="length of the QC overlay clip")
    ap.add_argument("--export-regions", action="store_true", help="also write full per-region videos (large)")
    ap.add_argument("--rebuild", action="store_true", help="re-extract signals even if cached")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)

    if args.all:
        runs = discover_runs(data_root)
    elif args.run:
        p = Path(args.run)
        runs = [p if p.is_absolute() or p.exists() else data_root / args.run]
    else:
        ap.error("give a run name/path or --all")
        return 2

    if not runs:
        print(f"No runs found under {data_root}")
        return 1
    print(f"Analysing {len(runs)} run(s); results -> {results_root}")
    for run in runs:
        analyse_run(run, results_root, stride=args.stride, qc_seconds=args.qc_seconds,
                    export_regions=args.export_regions, rebuild=args.rebuild)
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
