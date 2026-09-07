import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "validetion" / "Servomotor"
REPORT_PATH = PACKAGE / "generate_report.py"
RESPONSES = PACKAGE / "responses"
PRIMARY_RUN = RESPONSES / "motor_response_2026_04_28_15_02_26"
DRY_RUN = RESPONSES / "motor_response_2026_04_28_14_33_58"


def load_report_module():
    spec = importlib.util.spec_from_file_location("motor_response_generate_report", REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = load_report_module()


def test_compute_independent_summary_uses_trial_local_zero():
    raw = pd.DataFrame(
        [
            {"mode": "protocol", "block": 1, "trial": 1, "delta": 10, "target": 0, "angle_deg": 1},
            {"mode": "protocol", "block": 1, "trial": 1, "delta": 10, "target": 10, "angle_deg": 3},
            {"mode": "protocol", "block": 1, "trial": 1, "delta": 10, "target": 0, "angle_deg": 1},
            {"mode": "protocol", "block": 1, "trial": 1, "delta": 10, "target": -10, "angle_deg": -2},
            {"mode": "protocol", "block": 1, "trial": 1, "delta": 10, "target": 0, "angle_deg": 1},
            {"mode": "protocol", "block": 1, "trial": 2, "delta": 10, "target": 0, "angle_deg": 5},
            {"mode": "protocol", "block": 1, "trial": 2, "delta": 10, "target": 10, "angle_deg": 8},
            {"mode": "protocol", "block": 1, "trial": 2, "delta": 10, "target": 0, "angle_deg": 5},
            {"mode": "protocol", "block": 1, "trial": 2, "delta": 10, "target": -10, "angle_deg": 2},
            {"mode": "protocol", "block": 1, "trial": 2, "delta": 10, "target": 0, "angle_deg": 5},
            {"mode": "drift", "block": 1, "trial": 99, "delta": 10, "target": 10, "angle_deg": 99},
        ]
    )
    raw["relative_error_percent"] = 0.0

    summary = report.compute_independent_summary(raw)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["delta"] == 10
    assert row["n_pos"] == 2
    assert row["n_neg"] == 2
    assert row["angle_mean_pos_deg"] == pytest.approx(2.5)
    assert row["angle_mean_neg_deg"] == pytest.approx(-3.0)
    assert row["mean_abs_response_deg"] == pytest.approx(2.75)
    assert row["signed_asymmetry_deg"] == pytest.approx(-0.5)
    assert row["angle_repeatability_std_deg"] == pytest.approx(math.sqrt(1 / 6))


def test_primary_run_independent_summary_matches_saved_table():
    if not PRIMARY_RUN.exists():
        pytest.skip("primary motor-response result folder is not available")

    log = pd.read_csv(PRIMARY_RUN / "protocol_log.csv")
    independent = report.compute_independent_summary(log)
    diff, max_abs_diff = report.compare_saved_summary(PRIMARY_RUN, independent)

    assert diff is not None
    assert max_abs_diff is not None
    assert max_abs_diff < 1e-12


def test_real_camera_run_filter_excludes_dry_run():
    if not PRIMARY_RUN.exists() or not DRY_RUN.exists():
        pytest.skip("motor-response result folders are not available")

    assert report.is_real_camera_run(PRIMARY_RUN)
    assert not report.is_real_camera_run(DRY_RUN)
