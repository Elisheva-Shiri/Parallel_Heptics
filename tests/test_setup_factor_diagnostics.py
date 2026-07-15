from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.success_factors import setup_factors as sfd  # noqa: E402


def test_setup_factor_diagnostics_writes_summary_and_figure(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "success_rate": 0.5},
            {"subject_id": "N_E02", "finger_condition": "M", "success_rate": 0.7},
            {"subject_id": "L_E01", "finger_condition": "I", "success_rate": 0.8},
            {"subject_id": "L_P01", "finger_condition": "M", "success_rate": 0.9},
        ]
    )

    tables = sfd.analyze_table(
        df,
        category="unit",
        table_name="summary",
        metrics=["success_rate"],
        condition_cols=["finger_condition"],
        output_root=tmp_path,
        fig_dpi=72,
    )

    assert "unit_summary_setup_status" in tables
    assert tables["unit_summary_setup_status"]["setup_labels_detected"].iloc[0]
    assert "unit_summary_setup_balance" in tables
    assert (tmp_path / "unit_summary_setup_balance.csv").exists()
    manifest = tables["unit_summary_figure_manifest"]
    assert not manifest.empty
    assert Path(manifest["figure"].iloc[0]).exists()
    scope_manifest = tables["unit_summary_scope_figure_manifest"]
    assert {"all", "participant", "setup_factor"}.issubset(set(scope_manifest["summary_level"]))
    assert (tmp_path / "unit_summary_scope_figure_manifest.csv").exists()
