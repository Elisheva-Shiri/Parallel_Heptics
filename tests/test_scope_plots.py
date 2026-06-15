from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.scope_plots import save_scope_summary_plots  # noqa: E402


def test_scope_summary_plots_write_all_group_setup_and_participant_figures(tmp_path: Path) -> None:
    tables = {
        "unit_analysis_scope_metric_summary": pd.DataFrame(
            [
                {"analysis_scope": "all", "analysis_scope_value": "all_participants", "metric": "success_rate", "mean": 0.7, "n_subjects": 3, "n_observations": 9},
                {"analysis_scope": "experiment_group", "analysis_scope_value": "N_E", "metric": "success_rate", "mean": 0.5, "n_subjects": 1, "n_observations": 3},
                {"analysis_scope": "subgroup", "analysis_scope_value": "L", "metric": "success_rate", "mean": 0.8, "n_subjects": 2, "n_observations": 6},
                {"analysis_scope": "participant", "analysis_scope_value": "N_E01", "metric": "success_rate", "mean": 0.5, "n_subjects": 1, "n_observations": 3},
            ]
        ),
        "unit_group_metric_summary": pd.DataFrame(
            [
                {"experiment_group": "N_E", "metric": "success_rate", "mean": 0.5, "n_subjects": 1, "n_observations": 3},
                {"experiment_group": "L_E", "metric": "success_rate", "mean": 0.8, "n_subjects": 1, "n_observations": 3},
            ]
        ),
        "unit_setup_metric_summary": pd.DataFrame(
            [
                {"setup_factor": "no_airsled", "metric": "success_rate", "mean": 0.5, "n_subjects": 1, "n_observations": 3},
                {"setup_factor": "airsled", "metric": "success_rate", "mean": 0.8, "n_subjects": 2, "n_observations": 6},
            ]
        ),
    }

    manifest = save_scope_summary_plots(tables, tmp_path, namespace="unit", metrics=["success_rate"], fig_dpi=72)

    assert (tmp_path / "unit_scope_figure_manifest.csv").exists()
    assert {"all", "experiment_group", "subgroup", "participant", "setup_factor"}.issubset(
        set(manifest["summary_level"])
    )
    assert all(Path(path).exists() for path in manifest["figure"])
