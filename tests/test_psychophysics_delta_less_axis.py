from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "psychophysics_analysis"))

import twoafc_psychophysics as psych  # noqa: E402


def _clean_trials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "N_E01",
                "finger_condition": "I",
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "response_comparison_greater": 0,
                "reaction_time": 1.0,
            },
            {
                "subject_id": "N_E01",
                "finger_condition": "I",
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "response_comparison_greater": 0,
                "reaction_time": 1.2,
            },
            {
                "subject_id": "N_E01",
                "finger_condition": "I",
                "comparison_value": 120.0,
                "standard_value": 85.0,
                "response_comparison_greater": 1,
                "reaction_time": 0.9,
            },
            {
                "subject_id": "N_E01",
                "finger_condition": "I",
                "comparison_value": 120.0,
                "standard_value": 85.0,
                "response_comparison_greater": 1,
                "reaction_time": 1.1,
            },
        ]
    )


def test_psychometric_input_exposes_delta_axis_and_less_probability() -> None:
    agg = psych.make_psychometric_input(_clean_trials(), ["subject_id", "finger_condition"])

    below = agg.loc[agg["comparison_value"] == 40.0].iloc[0]
    above = agg.loc[agg["comparison_value"] == 120.0].iloc[0]

    assert below["delta_comparison_minus_standard"] == -45.0
    assert above["delta_comparison_minus_standard"] == 35.0
    assert below["n_comparison_less"] == below["n_trials"] - below["n_comparison_greater"]
    assert above["n_comparison_less"] == above["n_trials"] - above["n_comparison_greater"]
    assert below["p_comparison_less"] == 1.0 - below["p_comparison_greater"]
    assert above["p_comparison_less"] == 1.0 - above["p_comparison_greater"]
    assert np.isclose(below["comparison_over_standard"], 40.0 / 85.0)
    assert np.isclose(above["signed_delta_over_standard"], 35.0 / 85.0)


def test_subject_average_keeps_requested_delta_and_less_columns() -> None:
    trials = pd.concat(
        [
            _clean_trials(),
            _clean_trials().assign(subject_id="L_E01", response_comparison_greater=[0, 1, 1, 1]),
        ],
        ignore_index=True,
    )

    averaged = psych.subject_average_psychometric(trials, ["finger_condition"])
    below = averaged.loc[averaged["comparison_value"] == 40.0].iloc[0]

    assert below["delta_comparison_minus_standard"] == -45.0
    assert "mean_p_comparison_less" in averaged.columns
    assert np.isclose(below["mean_p_comparison_less"], 0.75)


def test_plot_fit_curve_uses_delta_x_and_greater_probability_y(tmp_path: Path) -> None:
    agg = psych.make_psychometric_input(_clean_trials(), ["subject_id", "finger_condition"])
    fit_row = pd.Series(
        {
            "subject_id": "N_E01",
            "finger_condition": "I",
            "standard_value": 85.0,
            "mu": 85.0,
            "scale": 8.0,
            "lapse_low": 0.0,
            "lapse_high": 0.0,
            "pse": 85.0,
            "pse_delta_comparison_minus_standard": 0.0,
        }
    )

    fig, ax = psych.plot_fit_curve(agg, fit_row, "delta less test", tmp_path / "curve.png")
    try:
        assert ax.get_xlabel() == "G_comparison-G_standart"
        assert ax.get_ylabel() == "P(choose comparison > standard)"

        scatter_offsets = [
            collection.get_offsets()
            for collection in ax.collections
            if hasattr(collection, "get_offsets") and len(collection.get_offsets()) == 2
        ]
        offsets = np.asarray(scatter_offsets[-1], dtype=float)
        offsets = offsets[np.argsort(offsets[:, 0])]

        np.testing.assert_allclose(offsets[:, 0], [-45.0, 35.0])
        np.testing.assert_allclose(offsets[:, 1], [0.0, 1.0])
        np.testing.assert_allclose(ax.get_xticks(), [-45.0, 0.0, 35.0])

        fit_line = next(line for line in ax.lines if line.get_label() == "Fit")
        fit_x = fit_line.get_xdata()
        fit_y = fit_line.get_ydata()
        assert fit_x[0] < 0 < fit_x[-1]
        assert fit_y[0] < fit_y[-1]
        assert np.isclose(fit_y[0], psych.logistic4(np.array([40.0]), 85.0, 8.0, 0.0, 0.0)[0])
        assert np.isclose(fit_y[-1], psych.logistic4(np.array([120.0]), 85.0, 8.0, 0.0, 0.0)[0])
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
