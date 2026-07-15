from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "psychophysics"))

import twoafc_psychophysics as psych  # noqa: E402


def _write_answers(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "pair_number",
                "object_1_finger",
                "object_1_stiffness",
                "object_2_finger",
                "object_2_stiffness",
                "time_to_answer",
                "answer",
            ]
        )
        writer.writerow(
            ["2026-01-01T00:00:06", 1, "index", 40, "index", 85, 1.2, 1]
        )


def test_discover_answer_files_filter_folders_are_flagged_only_for_group_selection(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "raw"
    output_root = tmp_path / "analysis_out"
    for subject in ["L_E_1", "L_E_16 (filter)", "L_E_13 (not finish)"]:
        _write_answers(data_root / subject / "answers.csv")

    with_filtered = psych.discover_answer_files(
        data_root,
        output_root / "with_filtered",
        selection="L_E",
        exclude_filter_folders=False,
    )
    without_filtered = psych.discover_answer_files(
        data_root,
        output_root / "without_filtered",
        selection="L_E",
        exclude_filter_folders=True,
    )
    explicit_filtered_subject = psych.discover_answer_files(
        data_root,
        output_root / "explicit",
        selection="L_E_16",
        exclude_filter_folders=True,
    )
    filtered_only = psych.discover_answer_files(
        data_root,
        output_root / "filtered_only",
        selection="L_E_FILTER_ONLY",
        exclude_filter_folders=False,
    )
    all_filtered_only = psych.discover_answer_files(
        data_root,
        output_root / "all_filtered_only",
        selection="FILTER_ONLY",
        exclude_filter_folders=False,
    )

    assert with_filtered.loc[with_filtered["selected"], "subject_id"].tolist() == [
        "L_E_1",
        "L_E_16 (filter)",
    ]
    assert without_filtered.loc[without_filtered["selected"], "subject_id"].tolist() == [
        "L_E_1"
    ]
    assert explicit_filtered_subject.loc[
        explicit_filtered_subject["selected"], "subject_id"
    ].tolist() == ["L_E_16 (filter)"]
    assert filtered_only.loc[filtered_only["selected"], "subject_id"].tolist() == [
        "L_E_16 (filter)"
    ]
    assert all_filtered_only.loc[
        all_filtered_only["selected"], "subject_id"
    ].tolist() == ["L_E_16 (filter)"]
    assert psych.selection_label("L_E_FILTER") == "L_E_FILTER_ONLY"
    assert (
        psych.selection_output_label("L_E", exclude_filter_folders=False)
        == "L_E_before_filter"
    )
    assert psych.selection_output_label("L_E", exclude_filter_folders=True) == "L_E"
    assert (
        psych.selection_output_label("L_E_FILTER_ONLY", exclude_filter_folders=False)
        == "L_E_FILTER_ONLY"
    )
    assert (
        psych.selection_output_label("FILTER_ONLY", exclude_filter_folders=False)
        == "FILTER_ONLY"
    )
    assert (
        psych.selection_output_label("L_E_16", exclude_filter_folders=False)
        == "L_E_16"
    )
