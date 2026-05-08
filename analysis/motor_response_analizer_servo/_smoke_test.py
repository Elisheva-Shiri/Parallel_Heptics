"""Smoke-test the protocol generator and the angle detector with synthetic frames.

Run from the repo root:
    python analysis/motor_response_experiment/_smoke_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from motor_response_analizer_servo.protocol import build_protocol
from motor_response_analizer_servo.vision_angle import (
    SpoolAngleDetector, find_spool_roi,
)
from motor_response_analizer_servo.motor_io import DryRunMotor


H, W = 480, 640


def synthetic(angle_deg: float, line_thickness: int = 8) -> np.ndarray:
    img = np.full((H, W, 3), 250, np.uint8)
    cv2.circle(img, (320, 240), 160, (180, 180, 180), -1)   # darker spot
    cv2.circle(img, (320, 240), 110, (240, 240, 240), -1)   # white spool
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), -math.sin(a)
    L = int(110 * 0.95)
    cv2.line(
        img,
        (int(320 - dx * L), int(240 - dy * L)),
        (int(320 + dx * L), int(240 + dy * L)),
        (10, 10, 10), line_thickness,
    )
    return img


def main() -> None:
    print("=" * 60)
    print("Protocol generator")
    print("=" * 60)
    proto = build_protocol()
    print(f"  total commands: {len(proto)}")
    deltas = sorted(set(s.delta for s in proto))
    print(f"  unique deltas:  {deltas}")
    by_delta = {d: sum(1 for s in proto if s.delta == d) for d in deltas}
    print(f"  steps / delta:  {by_delta}  (expected 50 each)")

    print()
    print("=" * 60)
    print("Dry-run motor I/O")
    print("=" * 60)
    with DryRunMotor() as m:
        for tgt in [0, 5, -5, 0, 250]:
            r = m.send(0, tgt)
            print(f"  send(target={tgt:+4d}) -> ok='{r.ok_line}' actual={r.actual}")

    print()
    print("=" * 60)
    print("Vision: auto ROI + angle accuracy")
    print("=" * 60)
    img = synthetic(0)
    roi = find_spool_roi(img, mode="auto")
    print(f"  auto-detected ROI: {roi}")
    det = SpoolAngleDetector(roi)
    errs: list[float] = []
    for true_a in [-89, -80, -60, -45, -30, -10, 0, 10, 30, 45, 60, 80, 89]:
        img = synthetic(true_a)
        meas = det.measure(img)
        meas_str = "None " if meas.angle_deg is None else f"{meas.angle_deg:+7.2f}"
        if meas.angle_deg is not None:
            errs.append(meas.angle_deg - true_a)
        print(
            f"  true={true_a:+4d}  meas={meas_str}  conf={meas.confidence:.3f}  "
            f"pix={meas.line_pixel_count}"
        )
    if errs:
        print(f"  max-abs-error: {max(abs(e) for e in errs):.3f} deg "
              f"  std-error: {float(np.std(errs)):.3f}")

    print()
    print("=" * 60)
    print("Vision: unwrap across +/-90 boundary")
    print("=" * 60)
    seq = [85.0, 89.0, -88.0, -85.0, -80.0]
    unwrapped = SpoolAngleDetector.unwrap_series(seq)
    print(f"  raw:        {seq}")
    print(f"  unwrapped:  {unwrapped}")


if __name__ == "__main__":
    main()
