from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "anova_statistics"))

pytest.importorskip("pingouin")

import anova_statistics as anova  # noqa: E402


def _write_pse_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "subject_id": ["L_E_1", "N_E_1"],
            "workspace_setup": ["L", "N"],
            "finger_condition": ["I", "I"],
            "pse": [80.0, 90.0],
            "pse_delta_from_standard": [-5.0, 5.0],
            "jnd": [20.0, 25.0],
        }
    ).to_csv(path, index=False)


def _write_success_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "subject_id": ["L_E_1", "N_E_1"],
            "finger_condition": ["I", "I"],
            "success_rate": [0.8, 0.7],
            "n_trials": [40, 40],
        }
    ).to_csv(path, index=False)


def test_anova_resolves_filtered_psychophysics_results_folder(tmp_path: Path) -> None:
    results_root = tmp_path / "psychophysics" / "results"
    selected = results_root / "L_E" / "_working"
    _write_pse_summary(selected / anova.PSE_JND_SUMMARY_FILENAME)
    _write_success_summary(selected / anova.SUCCESS_SUMMARY_FILENAME)

    resolved = anova.resolve_psychophysics_summary_path(
        "L_E", results_root=str(results_root)
    )
    assert resolved.endswith(
        str(Path("L_E") / "_working" / anova.PSE_JND_SUMMARY_FILENAME)
    )

    df = anova.load_data("L_E", results_root=str(results_root))
    assert df.attrs["source_path"] == resolved
    assert set(df["Subject"].astype(str)) == {"L_E_1", "N_E_1"}
    assert anova.selected_source_label("L_E") == "L_E"

    success = anova.load_success_rates("L_E", results_root=str(results_root))
    assert set(success.columns) == {
        "Subject",
        "Finger",
        "success_rate",
        "n_success_trials",
    }
    assert success["success_rate"].tolist() == [0.8, 0.7]


def test_anova_load_data_accepts_direct_csv_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "custom_source" / anova.PSE_JND_SUMMARY_FILENAME
    _write_pse_summary(csv_path)

    df = anova.load_data(str(csv_path))
    assert df.attrs["source_path"] == str(csv_path.resolve())
    assert anova.selected_source_label(str(csv_path)) == "custom_source"


def test_missing_success_rates_return_empty_frame(tmp_path: Path) -> None:
    results_root = tmp_path / "psychophysics" / "results"

    with pytest.warns(UserWarning, match="success_lt55 subgroup filter will be skipped"):
        success = anova.load_success_rates("L_E", results_root=str(results_root))

    assert success.empty
    assert list(success.columns) == [
        "Subject",
        "Finger",
        "success_rate",
        "n_success_trials",
    ]
