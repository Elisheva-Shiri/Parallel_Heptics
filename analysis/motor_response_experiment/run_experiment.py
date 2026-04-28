"""Run the motor response characterization experiment end-to-end.

Workflow:

1.  Build the protocol (deltas + A/B trials + drift block, see ``protocol.py``).
2.  Open the ESP32 over serial (``ZM<idx>P<target>F`` -> ``OK:M<idx>P<actual>``).
3.  Open camera index 1 in the background and start recording.
4.  Pick up an initial spool ROI (auto-detect the dark spot + white spool, with
    a manual-click fallback).
5.  For every protocol step:
        * send the motor command
        * wait for the firmware ``OK`` response
        * sleep ``settle_ms`` for the motor to mechanically settle
        * grab the latest camera frame
        * measure the black-line angle on the spool
        * record everything into a row of an Excel/CSV log
6.  Stop the recording, save the angle-annotated frames, and dump a summary.

Single command does the full pipeline (**one run**):

* motor protocol → camera capture → ``protocol_log.*`` → **plots** + ``per_delta_summary.csv`` (unless ``--no-plots``).

By default all outputs go under **this package**:

    analysis/motor_response_experiment/responses/motor_response_<timestamp>/
        protocol_log.csv / .xlsx
        recording.mp4
        frames/
        plots/
        per_delta_summary.csv
        run_summary.json

Use ``--output-root`` only if you want a different parent folder.

Run with::

    python -m analysis.motor_response_experiment.run_experiment --port COM13

Or, without hardware/camera::

    python -m analysis.motor_response_experiment.run_experiment --dry-run --no-camera

The output folder layout is (default: under ``motor_response_experiment/responses/``)::

    motor_response_experiment/responses/motor_response_<timestamp>/
        protocol_log.xlsx       # full log (matches the Excel template + extra columns)
        protocol_log.csv        # same data, plain CSV
        recording.mp4           # full-length camera recording
        frames/step_0001.jpg    # settled frame for each step (annotated)
        spool_roi.png           # the auto/manual ROI overlay used
        run_summary.json        # config + counters + timing summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

# Allow running both as a script (``python file.py``) and as a module
# (``python -m analysis.motor_response_experiment.run_experiment``).
if __package__ in (None, ""):
    THIS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(THIS_DIR.parent))
    from motor_response_experiment.protocol import build_protocol, ProtocolStep, DEFAULT_DELTAS
    from motor_response_experiment.motor_io import MotorSerial, DryRunMotor
    from motor_response_experiment.vision_angle import (
        SpoolAngleDetector,
        SpoolROI,
        detect_dark_spot,
        detect_white_spool,
        find_spool_roi,
        manual_pick_spool,
    )
    from motor_response_experiment.camera_recorder import CameraConfig, CameraRecorder
else:
    from .protocol import build_protocol, ProtocolStep, DEFAULT_DELTAS
    from .motor_io import MotorSerial, DryRunMotor
    from .vision_angle import (
        SpoolAngleDetector,
        SpoolROI,
        detect_dark_spot,
        detect_white_spool,
        find_spool_roi,
        manual_pick_spool,
    )
    from .camera_recorder import CameraConfig, CameraRecorder


# ---------------------------------------------------------------------------
# Output location — default: ``<this package>/responses/``
# ---------------------------------------------------------------------------

def default_responses_parent() -> Path:
    """Directory that holds timestamped ``motor_response_<date>/`` runs (next to code)."""
    return Path(__file__).resolve().parent / "responses"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Motor protocol
    deltas: tuple[int, ...] = DEFAULT_DELTAS
    motor_index: int = 0
    trials_per_sequence: int = 3
    drift_pairs: int = 10

    # Serial
    port: str = "COM13"
    baud: int = 115200
    response_timeout_s: float = 5.0
    boot_wait_s: float = 3.0
    settle_ms: int = 1200       # wait after firmware OK before we consider the spool settled
    inter_command_ms: int = 250  # pause after vision capture before the *next* motor command
    frame_grab_timeout_s: float = 3.0  # max wait for a post-settle camera frame

    # Camera
    camera_index: int = 1
    camera_width: Optional[int] = None
    camera_height: Optional[int] = None
    camera_fps: float = 30.0

    # Vision
    roi_mode: str = "both"         # auto | manual | both
    line_threshold: Optional[int] = None
    radial_inset: float = 0.92
    background_drop: float = 0.55
    min_contrast: int = 25
    min_pixels: int = 30

    # Output (each run creates ``<output_root>/motor_response_<timestamp>/``)
    output_root: Path = field(default_factory=default_responses_parent)
    save_video: bool = True
    save_frames: bool = True
    save_annotated_frames: bool = True

    # Modes
    dry_run: bool = False
    no_camera: bool = False
    confirm_roi: bool = True
    # After saving logs, run analyze.py to write ``plots/`` and per_delta_summary.csv
    run_analysis: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out = Path(root) / f"motor_response_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    return out


def _open_motor(cfg: ExperimentConfig):
    if cfg.dry_run:
        print(f"[motor] dry-run mode: no serial connection")
        return DryRunMotor()
    return MotorSerial(
        port=cfg.port,
        baud=cfg.baud,
        boot_wait_s=cfg.boot_wait_s,
        response_timeout_s=cfg.response_timeout_s,
    )


def _open_camera(cfg: ExperimentConfig, video_path: Optional[Path]) -> Optional[CameraRecorder]:
    if cfg.no_camera:
        print("[camera] disabled (--no-camera)")
        return None
    cam_cfg = CameraConfig(
        index=cfg.camera_index,
        width=cfg.camera_width,
        height=cfg.camera_height,
        fps=cfg.camera_fps,
    )
    return CameraRecorder(cam_cfg, video_path=video_path if cfg.save_video else None)


def _pick_initial_roi(cfg: ExperimentConfig, camera: Optional[CameraRecorder]) -> Optional[SpoolROI]:
    if camera is None:
        return None
    frame = camera.wait_for_fresh(min_age_s=0.5, timeout_s=3.0)
    if frame is None:
        print("[vision] WARNING: no camera frame available for ROI selection")
        return None

    roi: Optional[SpoolROI] = None
    if cfg.roi_mode in ("auto", "both"):
        box = detect_dark_spot(frame)
        roi = detect_white_spool(frame, search_box=box)
        if roi is not None:
            print(f"[vision] auto ROI: cx={roi.cx} cy={roi.cy} r={roi.radius}")

    if roi is None and cfg.roi_mode in ("manual", "both"):
        print("[vision] auto-detection failed -> please click the spool centre and drag")
        roi = manual_pick_spool(frame)

    if roi is None and cfg.roi_mode == "auto":
        raise RuntimeError("Auto ROI detection failed and roi_mode='auto'")

    if cfg.confirm_roi and roi is not None:
        # Show overlay and let the user confirm or redo (SpoolAngleDetector is imported at module top)
        det_preview = SpoolAngleDetector(roi)
        meas = det_preview.measure(frame)
        preview = det_preview.annotate(frame, meas)
        cv2.imshow("Confirm ROI: Enter=ok, r=redo", preview)
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):
                break
            if key in (ord("r"), ord("R")):
                roi = manual_pick_spool(frame)
                det_preview = SpoolAngleDetector(roi)
                meas = det_preview.measure(frame)
                preview = det_preview.annotate(frame, meas)
                cv2.imshow("Confirm ROI: Enter=ok, r=redo", preview)
            if key == 27:
                cv2.destroyWindow("Confirm ROI: Enter=ok, r=redo")
                raise RuntimeError("ROI confirmation cancelled")
        cv2.destroyWindow("Confirm ROI: Enter=ok, r=redo")
    return roi


def _row_dict(step: ProtocolStep) -> dict:
    """Initial row dict using the same column order as the Excel template."""
    return {
        "block": step.block,
        "trial": step.trial,
        "delta": step.delta,
        "sequence": step.sequence,
        "mode": step.mode,
        "step_index": step.step_index,
        "target": step.target,
        "command": step.command,
        "actual": None,
        "response": None,
        "relative_error_percent": None,
        # extra columns we add
        "angle_deg_raw": None,
        "angle_deg": None,
        "angle_change_from_prev": None,
        "angle_change_from_zero": None,
        "angle_confidence": None,
        "angle_pixel_count": None,
        "frame_path": None,
        "send_elapsed_ms": None,
        "settle_elapsed_ms": None,
        "timestamp": None,
        "error": None,
    }


def _save_logs(rows: list[dict], out_dir: Path) -> tuple[Path, Optional[Path]]:
    df = pd.DataFrame(rows)
    csv_path = out_dir / "protocol_log.csv"
    xlsx_path = out_dir / "protocol_log.xlsx"
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False, sheet_name="protocol_log")
    except ImportError:
        print(
            "[save] WARNING: openpyxl is not installed - skipped protocol_log.xlsx. "
            "Your data is in protocol_log.csv. To enable Excel export, run:  "
            "pip install openpyxl"
        )
        return csv_path, None
    except Exception as e:
        print(f"[save] WARNING: Excel export failed ({e!s}). CSV saved at {csv_path.name}.")
        return csv_path, None
    return csv_path, xlsx_path


def _save_summary(out_dir: Path, cfg: ExperimentConfig, roi: Optional[SpoolROI],
                  rows: list[dict], camera: Optional[CameraRecorder]) -> Path:
    deltas = sorted(set(r["delta"] for r in rows))
    summary = {
        "config": {**asdict(cfg), "output_root": str(cfg.output_root)},
        "n_rows": len(rows),
        "deltas": deltas,
        "rows_per_delta": {int(d): sum(1 for r in rows if r["delta"] == d) for d in deltas},
        "camera": {
            "enabled": camera is not None,
            "frame_count": camera.frame_count if camera else 0,
            "actual_fps": camera.actual_fps if camera else 0.0,
        },
        "roi": None if roi is None else asdict(roi),
        "n_angle_failures": sum(1 for r in rows if r.get("angle_deg") is None),
        "n_motor_errors": sum(1 for r in rows if r.get("error")),
    }
    path = out_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def _run_analysis_plots(out_dir: Path) -> None:
    """Generate ``plots/*.png`` and ``per_delta_summary.csv`` (see ``analyze.py``)."""
    try:
        if __package__ in (None, ""):
            from motor_response_experiment.analyze import analyze
        else:
            from .analyze import analyze
    except ImportError as e:
        print(f"[analyze] WARNING: could not load analyze module: {e}")
        return
    try:
        plots_dir = analyze(out_dir)
        print(f"[analyze] done - open: {plots_dir}")
    except Exception as e:
        print(f"[analyze] WARNING: plot generation failed: {e!s}")


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(cfg: ExperimentConfig) -> Path:
    out_dir = _make_output_dir(cfg.output_root)
    print(f"[setup] responses folder (this run): {out_dir}")
    print("[setup] pipeline: protocol -> acquisition -> CSV/XLSX -> summary -> plots "
          "(single command; use --no-plots to skip charts)")

    proto = build_protocol(
        deltas=cfg.deltas,
        motor_index=cfg.motor_index,
        trials_per_sequence=cfg.trials_per_sequence,
        drift_pairs=cfg.drift_pairs,
    )
    print(f"[setup] protocol: {len(proto)} steps over {len(set(p.delta for p in proto))} deltas")
    print(
        f"[setup] timing: settle_ms={cfg.settle_ms}  inter_command_ms={cfg.inter_command_ms}  "
        f"frame_grab_timeout_s={cfg.frame_grab_timeout_s}"
    )

    motor = _open_motor(cfg)
    motor.open()
    print(f"[motor] connected to {motor.port_name}")

    camera = _open_camera(cfg, video_path=out_dir / "recording.mp4")
    if camera is not None:
        camera.start()

    detector: Optional[SpoolAngleDetector] = None
    roi: Optional[SpoolROI] = None
    try:
        if camera is not None:
            roi = _pick_initial_roi(cfg, camera)
            if roi is not None:
                detector = SpoolAngleDetector(
                    roi=roi,
                    line_threshold=cfg.line_threshold,
                    radial_inset=cfg.radial_inset,
                    min_pixels=cfg.min_pixels,
                    background_drop=cfg.background_drop,
                    min_contrast=cfg.min_contrast,
                )
                # Save a snapshot of the ROI overlay for documentation.
                ref_frame = camera.wait_for_fresh(min_age_s=0.5, timeout_s=2.0)
                if ref_frame is not None:
                    overlay = detector.annotate(ref_frame, detector.measure(ref_frame))
                    cv2.imwrite(str(out_dir / "spool_roi.png"), overlay)

        rows: list[dict] = []
        zero_angle: Optional[float] = None
        prev_angle: Optional[float] = None

        t_start = time.time()
        for i, step in enumerate(proto, start=1):
            row = _row_dict(step)
            row["timestamp"] = datetime.now().isoformat(timespec="milliseconds")

            # 1. Send command and wait for OK
            t_send = time.time()
            resp = motor.send_command(step.command, cfg.motor_index)
            row["send_elapsed_ms"] = int((time.time() - t_send) * 1000)
            row["response"] = resp.raw
            row["actual"] = resp.actual
            row["error"] = resp.error
            if resp.actual is not None and step.target != 0:
                row["relative_error_percent"] = (resp.actual - step.target) / step.target * 100.0
            elif resp.actual is not None and step.target == 0:
                row["relative_error_percent"] = float(resp.actual)  # absolute deviation when target=0

            # 2. Mechanical settle (after encoder reports OK)
            t_settle_start = time.time()
            time.sleep(cfg.settle_ms / 1000.0)
            t_settle_end = time.time()

            # 3. Grab one frame whose capture time is *after* settle ended (not a stale buffer)
            if camera is not None and detector is not None:
                frame = camera.wait_for_frame_after(
                    t_settle_end, timeout_s=cfg.frame_grab_timeout_s
                )
                if frame is None:
                    print(
                        f"[vision] WARNING: no frame newer than settle end at step {i}; "
                        f"falling back to latest buffer"
                    )
                    frame, _ = camera.get_latest()
                if frame is not None:
                    meas = detector.measure(frame)
                    row["angle_deg_raw"] = meas.angle_deg
                    row["angle_confidence"] = meas.confidence
                    row["angle_pixel_count"] = meas.line_pixel_count
                    if meas.angle_deg is not None:
                        if zero_angle is None:
                            zero_angle = meas.angle_deg
                        # Online unwrap relative to the previous measurement.
                        unwrapped = SpoolAngleDetector.unwrap_series(
                            [prev_angle, meas.angle_deg] if prev_angle is not None else [meas.angle_deg]
                        )
                        unwrap_now = unwrapped[-1]
                        row["angle_deg"] = unwrap_now
                        row["angle_change_from_prev"] = (
                            None if prev_angle is None else unwrap_now - prev_angle
                        )
                        row["angle_change_from_zero"] = unwrap_now - zero_angle
                        prev_angle = unwrap_now

                    if cfg.save_frames:
                        frame_path = out_dir / "frames" / f"step_{i:04d}.jpg"
                        if cfg.save_annotated_frames:
                            cv2.imwrite(str(frame_path), detector.annotate(frame, meas))
                        else:
                            cv2.imwrite(str(frame_path), frame)
                        row["frame_path"] = str(frame_path.relative_to(out_dir))
            row["settle_elapsed_ms"] = int((time.time() - t_settle_start) * 1000)

            rows.append(row)

            # Periodic progress prints
            if i == 1 or i % 25 == 0 or i == len(proto):
                pct = 100.0 * i / len(proto)
                a_str = "n/a" if row["angle_deg"] is None else f"{row['angle_deg']:+.2f} deg"
                print(
                    f"[run] {i:4d}/{len(proto)} ({pct:5.1f}%)  "
                    f"block={step.block} trial={step.trial} delta={step.delta:>3} "
                    f"target={step.target:+5d}  actual={row['actual']!s:>5}  "
                    f"angle={a_str}"
                )

            # Inter-command pacing
            if cfg.inter_command_ms > 0:
                time.sleep(cfg.inter_command_ms / 1000.0)

        elapsed = time.time() - t_start
        print(f"[run] done in {elapsed:.1f} s")

    finally:
        motor.close()
        if camera is not None:
            camera.stop()

    csv_path, xlsx_path = _save_logs(rows, out_dir)
    summary_path = _save_summary(out_dir, cfg, roi, rows, camera)
    print(f"[save] {csv_path}")
    if xlsx_path is not None:
        print(f"[save] {xlsx_path}")
    print(f"[save] {summary_path}")
    if cfg.run_analysis:
        _run_analysis_plots(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", default="COM13", help="ESP32 serial port (default: COM13)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--motor-index", type=int, default=0)
    p.add_argument(
        "--settle-ms",
        type=int,
        default=1200,
        help="Wait after firmware OK before grabbing vision (motor + spool settle; default 1200)",
    )
    p.add_argument(
        "--inter-command-ms",
        type=int,
        default=250,
        help="Extra pause after each vision capture before sending the next command (default 250)",
    )
    p.add_argument(
        "--frame-grab-timeout",
        type=float,
        default=3.0,
        metavar="SEC",
        help="Max seconds to wait for a camera frame newer than settle end (default 3.0)",
    )
    p.add_argument("--camera-index", type=int, default=1)
    p.add_argument("--camera-fps", type=float, default=30.0)
    p.add_argument("--camera-width", type=int, default=None)
    p.add_argument("--camera-height", type=int, default=None)
    p.add_argument("--roi-mode", choices=["auto", "manual", "both"], default="both")
    p.add_argument("--no-confirm-roi", action="store_true", help="Skip the ROI confirmation popup")
    p.add_argument("--no-frames", action="store_true", help="Do not save per-step frames")
    p.add_argument("--no-video", action="store_true", help="Do not save the continuous video")
    p.add_argument("--no-annotate", action="store_true", help="Save raw frames (no overlay)")
    p.add_argument("--dry-run", action="store_true", help="No serial; pretend actual = target")
    p.add_argument("--no-camera", action="store_true", help="Skip camera entirely (no angle data)")
    p.add_argument("--no-plots", action="store_true", help="Skip automatic analysis plots (plots/ folder)")
    p.add_argument(
        "--output-root",
        default=None,
        metavar="DIR",
        help="Parent folder for runs (default: motor_response_experiment/responses next to this code)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_root = Path(args.output_root) if args.output_root is not None else default_responses_parent()
    cfg = ExperimentConfig(
        port=args.port,
        baud=args.baud,
        motor_index=args.motor_index,
        settle_ms=args.settle_ms,
        inter_command_ms=args.inter_command_ms,
        frame_grab_timeout_s=args.frame_grab_timeout,
        camera_index=args.camera_index,
        camera_fps=args.camera_fps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        roi_mode=args.roi_mode,
        confirm_roi=not args.no_confirm_roi,
        save_frames=not args.no_frames,
        save_video=not args.no_video,
        save_annotated_frames=not args.no_annotate,
        dry_run=args.dry_run,
        no_camera=args.no_camera,
        run_analysis=not args.no_plots,
        output_root=out_root,
    )
    run(cfg)


if __name__ == "__main__":
    main()
