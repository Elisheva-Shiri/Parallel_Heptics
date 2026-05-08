"""Generate a fake protocol_log.csv with realistic angle data, for testing analyze.py.

Each motor unit is mapped to a fictitious angle in degrees (0.05 deg/unit) plus
a small per-trial random noise and a small encoder error.  Useful to verify
that ``analyze.py`` produces sensible plots before any real hardware is run.

Run from repo root:
    python analysis/motor_response_experiment/_synthesize_log.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from motor_response_analizer_servo.protocol import build_protocol


ANGLE_PER_UNIT_DEG = 0.05
ANGLE_NOISE_DEG = 0.4
ENCODER_NOISE_PCT = 1.5

OUT_ROOT = Path("analysis")


def main() -> None:
    rng = np.random.default_rng(42)

    proto = build_protocol()
    rows = []
    block_zero = {}        # block -> the absolute reference angle for that block
    prev_angle = None

    for s in proto:
        # ESP32 actual = target with small noise on non-zero targets
        if s.target == 0:
            actual = 0
            rel_err = 0.0
        else:
            err_pct = rng.normal(0.0, ENCODER_NOISE_PCT)
            actual = int(round(s.target * (1 + err_pct / 100.0)))
            rel_err = (actual - s.target) / s.target * 100.0

        # angle is roughly proportional to (actual encoder position),
        # plus a tiny per-block offset and Gaussian noise
        if s.block not in block_zero:
            block_zero[s.block] = rng.uniform(-30, 30)
        absolute_angle = (block_zero[s.block]
                          + actual * ANGLE_PER_UNIT_DEG
                          + rng.normal(0, ANGLE_NOISE_DEG))
        # Wrap to (-90, 90] like the real detector would
        wrapped = absolute_angle
        while wrapped <= -90.0:
            wrapped += 180.0
        while wrapped > 90.0:
            wrapped -= 180.0

        rows.append({
            "block": s.block,
            "trial": s.trial,
            "delta": s.delta,
            "sequence": s.sequence,
            "mode": s.mode,
            "step_index": s.step_index,
            "target": s.target,
            "command": s.command,
            "actual": actual,
            "response": f"OK:M0P{actual}",
            "relative_error_percent": rel_err,
            "angle_deg_raw": wrapped,
            "angle_deg": absolute_angle,    # already unwrapped in our synthetic model
            "angle_change_from_prev": (None if prev_angle is None
                                       else absolute_angle - prev_angle),
            "angle_change_from_zero": absolute_angle - block_zero[s.block],
            "angle_confidence": 0.99,
            "angle_pixel_count": 1500,
            "frame_path": None,
            "send_elapsed_ms": 5,
            "settle_elapsed_ms": 600,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "error": None,
        })
        prev_angle = absolute_angle

    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out = OUT_ROOT / f"motor_response_{stamp}_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "protocol_log.csv", index=False)
    df.to_excel(out / "protocol_log.xlsx", index=False, sheet_name="protocol_log")
    print(f"Wrote synthetic log: {out}")
    print(f"  shape: {df.shape}")


if __name__ == "__main__":
    main()
