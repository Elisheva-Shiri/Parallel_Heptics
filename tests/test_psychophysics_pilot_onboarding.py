"""Verify that ``canonicalize_trials`` drops the first
``pilot_onboarding_trials`` clean trials per subject and reroutes them into
the flagged table with a ``pilot_onboarding_excluded`` reason."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "psychophysics_analysis"))

import twoafc_psychophysics as psych  # noqa: E402


def _raw_session(subject_id: str, n_trials: int) -> pd.DataFrame:
    """Build a minimal raw dataframe with one valid 2AFC trial per row."""
    rows = []
    for i in range(n_trials):
        rows.append(
            {
                "subject_id": subject_id,
                "_subject_folder": f"data/{subject_id}",
                "_source_file": f"data/{subject_id}/answers.csv",
                "_source_file_name": "answers.csv",
                "_row_in_source": i,
                "object_1_stiffness": 85.0,
                "object_2_stiffness": 40.0 if i % 2 == 0 else 120.0,
                "object_1_finger": "index",
                "object_2_finger": "index",
                "answer": 1 if i % 2 == 0 else 0,
                "time_to_answer": 1.0,
                "pair_number": i + 1,
            }
        )
    return pd.DataFrame(rows)


def _columns() -> dict[str, str]:
    return {
        "object_1_value": "object_1_stiffness",
        "object_2_value": "object_2_stiffness",
        "object_1_finger": "object_1_finger",
        "object_2_finger": "object_2_finger",
        "answer": "answer",
        "reaction_time": "time_to_answer",
        "trial_index": "pair_number",
        "timestamp": None,
        "block": None,
    }


def test_canonicalize_trials_excludes_first_twelve_per_subject_by_default() -> None:
    raw = _raw_session("N_E01", 20)
    clean, flagged = psych.canonicalize_trials(raw, _columns(), standard_value=85.0)

    assert len(clean) == 8
    assert clean["global_trial_order"].tolist() == list(range(13, 21))

    onboarding = flagged[flagged["flag_reason"] == "pilot_onboarding_excluded"]
    assert len(onboarding) == 12
    assert sorted(onboarding["global_trial_order"].tolist()) == list(range(1, 13))


def test_canonicalize_trials_per_subject_onboarding_independent() -> None:
    raw = pd.concat(
        [_raw_session("N_E01", 15), _raw_session("L_E02", 14)],
        ignore_index=True,
    )
    clean, flagged = psych.canonicalize_trials(raw, _columns(), standard_value=85.0)

    counts = clean.groupby("subject_id").size().to_dict()
    assert counts == {"N_E01": 3, "L_E02": 2}

    onboarding = flagged[flagged["flag_reason"] == "pilot_onboarding_excluded"]
    onboarding_counts = onboarding.groupby("subject_id").size().to_dict()
    assert onboarding_counts == {"N_E01": 12, "L_E02": 12}


def test_canonicalize_trials_disabled_when_pilot_zero() -> None:
    raw = _raw_session("N_E01", 5)
    clean, flagged = psych.canonicalize_trials(
        raw, _columns(), standard_value=85.0, pilot_onboarding_trials=0
    )

    assert len(clean) == 5
    assert "pilot_onboarding_excluded" not in set(flagged.get("flag_reason", pd.Series(dtype=str)))


def test_pilot_onboarding_constant_is_twelve() -> None:
    assert psych.PILOT_ONBOARDING_TRIALS == 12
