from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.success_factors import probing as pa  # noqa: E402


def _segment(radii: list[float], stiffness: float = 40.0, correct: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": "S1",
            "subject_group": "S",
            "trial_index_raw": 1,
            "pair_number": 1,
            "finger_condition": "index",
            "comparison_value": stiffness,
            "standard_value": 85,
            "signed_stiffness_delta": stiffness - 85,
            "correct_response": correct,
            "answer_code": 1,
            "stiffness_value": stiffness,
            "stiffness_segment_id": 1,
            "stiffness_order_in_trial": 1,
            "time_s": np.arange(len(radii), dtype=float),
            "stiffness_time_s": np.arange(len(radii), dtype=float),
            "r_center_px": radii,
            "x_centered_px": radii,
            "y_centered_px": np.zeros(len(radii)),
            "position_angle_deg": np.zeros(len(radii)),
            "speed_px_s": np.ones(len(radii)),
        }
    )


def test_detect_probe_events_counts_center_to_side_excursions_once_per_return() -> None:
    segment = _segment([0, 10, 90, 110, 70, 20, 5, 95, 100, 15])

    events, summary = pa.detect_probe_events(segment, center_radius_px=25, side_radius_px=80)

    assert summary["probe_count"] == 2
    assert summary["center_visit_count"] == 3
    assert [event["peak_direction"] for event in events] == ["E", "E"]
    assert events[0]["peak_radius_px"] == 110
    assert np.isclose(summary["first_probe_latency_s"], 2.0)
    assert np.isclose(summary["center_dwell_fraction"], 4 / 9)
    assert summary["side_dwell_fraction"] > 0


def test_compute_probing_metrics_keeps_subject_finger_stiffness_groups() -> None:
    samples = pd.concat(
        [
            _segment([0, 90, 0], stiffness=40, correct=1),
            _segment([0, 10, 0], stiffness=85, correct=0).assign(stiffness_segment_id=2, trial_index_raw=2),
        ],
        ignore_index=True,
    )
    trial_summary = samples.groupby(["subject_id", "trial_index_raw", "stiffness_segment_id"], as_index=False).agg(
        duration_s=("stiffness_time_s", "max"),
        n_tracking_samples=("r_center_px", "count"),
        mean_r_center_px=("r_center_px", "mean"),
        max_r_center_px=("r_center_px", "max"),
        path_length_px=("r_center_px", "sum"),
        mean_speed_px_s=("speed_px_s", "mean"),
    )

    tables = pa.compute_probing_metrics(samples, trial_summary, center_radius_px=25, side_radius_px=80)
    subject_summary = tables["probing_subject_finger_stiffness_summary"]
    stiffness_summary = tables["probing_stiffness_summary"]

    assert set(subject_summary["stiffness_value"]) == {40, 85}
    assert subject_summary.loc[subject_summary["stiffness_value"] == 40, "mean_probe_count"].iloc[0] == 1
    assert subject_summary.loc[subject_summary["stiffness_value"] == 85, "mean_probe_count"].iloc[0] == 0
    assert "mean_first_probe_latency_s" in subject_summary.columns
    assert "mean_center_dwell_fraction" in subject_summary.columns
    assert set(stiffness_summary["stiffness_value"]) == {40, 85}


def test_success_one_way_anova_tables_cover_requested_scopes_and_factors() -> None:
    rows = []
    for subject_id, group, offset in [("L_E_01", "L_E", 0), ("N_E_01", "N_E", 1)]:
        for finger in ["I", "M"]:
            for stiffness in [40, 85]:
                for probe_count in [0, 2]:
                    for repeat in range(3):
                        rows.append(
                            {
                                "subject_id": subject_id,
                                "subject_group": group[0],
                                "experiment_group": group,
                                "trial_index_raw": len(rows) + 1,
                                "finger_condition": finger,
                                "stiffness_value": stiffness,
                                "probe_count": probe_count,
                                "correct_response": float((probe_count == 2) or (stiffness == 85 and repeat > offset)),
                            }
                        )
    trial_summary = pd.DataFrame(rows)

    tables = pa.compute_success_one_way_anova_tables(trial_summary)
    anova = tables["probing_success_one_way_anova"]
    factor_summary = tables["probing_success_anova_factor_summary"]
    pairwise = tables["probing_success_anova_pairwise"]

    assert {"amount_of_probing", "stiffness_value", "finger"}.issubset(set(anova["factor"]))
    assert {"all", "experiment_group", "participant"}.issubset(set(anova["analysis_scope"]))
    assert {"trial", "participant_mean"}.issubset(set(anova["observation_level"]))
    all_probe = anova[
        (anova["analysis_scope"] == "all")
        & (anova["observation_level"] == "trial")
        & (anova["factor"] == "amount_of_probing")
    ].iloc[0]
    assert all_probe["status"] == "ok"
    assert np.isfinite(all_probe["f_statistic"])
    assert np.isfinite(all_probe["p_value"])
    assert factor_summary["factor_level"].notna().any()
    assert not pairwise.empty


def test_pair_direction_table_splits_direct_stiffness_switches(tmp_path: Path) -> None:
    tracking = pd.DataFrame(
        {
            "object_x": [pa.CENTER_X + r for r in [0, 10, 20, 30, 40, 50]]
            + [pa.CENTER_X for _ in range(6)],
            "object_y": [pa.CENTER_Y for _ in range(6)]
            + [pa.CENTER_Y - r for r in [0, 10, 20, 30, 40, 50]],
            "stiffness": [115] * 6 + [85] * 6,
        }
    )
    tracking_path = tmp_path / "tracking.csv"
    tracking.to_csv(tracking_path, index=False)
    trials = pd.DataFrame(
        [
            {
                "subject_id": "S1",
                "subject_group": "S",
                "trial_index_raw": 1,
                "pair_number": 1,
                "object_1_finger": "I",
                "object_2_finger": "I",
                "object_1_stiffness": 115,
                "object_2_stiffness": 85,
                "finger_condition": "I",
                "comparison_value": 115,
                "standard_value": 85,
                "signed_stiffness_delta": 30,
                "answer_code": 0,
                "correct_response": 1,
                "time_to_answer_s": 1.0,
                "tracking_exists": True,
                "tracking_file": tracking_path,
            }
        ]
    )

    pair_table = pa.build_pair_direction_table(trials)

    row = pair_table.iloc[0]
    assert row["n_segments_detected"] == 2
    assert np.isclose(row["direction_difference_deg"], 90.0)
    assert row["direction_pair_class"] == "different_direction"
    assert row["direction_warning"] == ""


def test_direction_success_summary_reports_same_vs_different_fisher_p() -> None:
    pair_table = pd.DataFrame(
        [
            {"subject_id": "S1", "direction_pair_class": "same_direction", "correct_response": 1, "direction_difference_deg": 5},
            {"subject_id": "S1", "direction_pair_class": "same_direction", "correct_response": 1, "direction_difference_deg": 10},
            {"subject_id": "S2", "direction_pair_class": "different_direction", "correct_response": 0, "direction_difference_deg": 120},
            {"subject_id": "S2", "direction_pair_class": "different_direction", "correct_response": 1, "direction_difference_deg": 130},
        ]
    )

    summary = pa.compute_direction_success_summary(pair_table)

    same = summary[summary["direction_pair_class"] == "same_direction"].iloc[0]
    different = summary[summary["direction_pair_class"] == "different_direction"].iloc[0]
    assert same["success_rate"] == 1.0
    assert different["success_rate"] == 0.5
    assert np.isfinite(summary["same_vs_different_fisher_p"].iloc[0])


def test_direction_success_rate_tables_split_roles_stiffness_and_direction() -> None:
    pair_table = pd.DataFrame(
        [
            {
                "subject_id": "S1",
                "subject_group": "S",
                "trial_index_raw": 1,
                "pair_number": 1,
                "finger_condition": "I",
                "object_1_stiffness": 115.0,
                "object_2_stiffness": 85.0,
                "comparison_value": 115.0,
                "standard_value": 85.0,
                "signed_stiffness_delta": 30.0,
                "object_1_direction_deg": 0.0,
                "object_2_direction_deg": 90.0,
                "direction_pair_class": "different_direction",
                "direction_difference_deg": 90.0,
                "correct_response": 1.0,
            },
            {
                "subject_id": "S1",
                "subject_group": "S",
                "trial_index_raw": 2,
                "pair_number": 2,
                "finger_condition": "I",
                "object_1_stiffness": 85.0,
                "object_2_stiffness": 115.0,
                "comparison_value": 115.0,
                "standard_value": 85.0,
                "signed_stiffness_delta": 30.0,
                "object_1_direction_deg": 90.0,
                "object_2_direction_deg": 0.0,
                "direction_pair_class": "different_direction",
                "direction_difference_deg": 90.0,
                "correct_response": 0.0,
            },
        ]
    )

    tables = pa.compute_direction_success_rate_tables(pair_table)

    trial_rows = tables["direction_success_trials"]
    assert len(trial_rows) == 4
    comparison_e = tables["direction_success_by_direction"][
        (tables["direction_success_by_direction"]["stimulus_role"] == "comparison")
        & (tables["direction_success_by_direction"]["direction_label"] == "E")
    ].iloc[0]
    assert comparison_e["n_trials"] == 2
    assert comparison_e["success_rate"] == 0.5
    participant_stiffness = tables["direction_success_by_participant_stiffness"]
    assert {
        "subject_id",
        "stimulus_role",
        "stiffness_value",
        "direction_label",
        "success_rate",
    }.issubset(participant_stiffness.columns)
    participant_mean = tables["direction_success_participant_mean_by_direction"]
    comparison_e_mean = participant_mean[
        (participant_mean["stimulus_role"] == "comparison")
        & (participant_mean["direction_label"] == "E")
    ].iloc[0]
    assert comparison_e_mean["n_subjects"] == 1
    assert comparison_e_mean["mean_participant_success_rate"] == 0.5
    assert "direction_success_participant_direction_contrasts" in tables


def test_direction_match_participant_contrast_is_subject_normalized() -> None:
    pair_table = pd.DataFrame(
        [
            # S1 has many same-direction trials and one different-direction trial;
            # the participant contrast should still be one row for S1, not one
            # row per raw trial.
            *[
                {
                    "subject_id": "S1",
                    "subject_group": "S",
                    "experiment_group": "S",
                    "direction_pair_class": "same_direction",
                    "correct_response": 1.0,
                    "direction_difference_deg": 10.0,
                }
                for _ in range(5)
            ],
            {
                "subject_id": "S1",
                "subject_group": "S",
                "experiment_group": "S",
                "direction_pair_class": "different_direction",
                "correct_response": 0.0,
                "direction_difference_deg": 120.0,
            },
            {
                "subject_id": "S2",
                "subject_group": "S",
                "experiment_group": "S",
                "direction_pair_class": "same_direction",
                "correct_response": 0.0,
                "direction_difference_deg": 20.0,
            },
            {
                "subject_id": "S2",
                "subject_group": "S",
                "experiment_group": "S",
                "direction_pair_class": "different_direction",
                "correct_response": 1.0,
                "direction_difference_deg": 110.0,
            },
        ]
    )

    tables = pa.compute_direction_match_participant_tables(pair_table)

    per_subject = tables["direction_match_success_by_subject"]
    contrast = tables["direction_match_same_vs_different_participant_contrast"]
    assert set(per_subject["direction_pair_class"]) == {"same_direction", "different_direction"}
    assert len(contrast) == 2
    s1 = contrast[contrast["subject_id"] == "S1"].iloc[0]
    s2 = contrast[contrast["subject_id"] == "S2"].iloc[0]
    assert s1["same_minus_different_success_rate"] == 1.0
    assert s2["same_minus_different_success_rate"] == -1.0
    assert "t_p_directional" in contrast.columns


def test_direction_label_participant_contrasts_compare_direction_to_other_within_subject() -> None:
    direction_trials = pd.DataFrame(
        [
            {"subject_id": "S1", "stimulus_role": "comparison", "direction_label": "SW", "correct_response": 0.0},
            {"subject_id": "S1", "stimulus_role": "comparison", "direction_label": "E", "correct_response": 1.0},
            {"subject_id": "S1", "stimulus_role": "comparison", "direction_label": "N", "correct_response": 1.0},
            {"subject_id": "S2", "stimulus_role": "comparison", "direction_label": "SW", "correct_response": 0.0},
            {"subject_id": "S2", "stimulus_role": "comparison", "direction_label": "E", "correct_response": 1.0},
            {"subject_id": "S2", "stimulus_role": "comparison", "direction_label": "N", "correct_response": 0.0},
        ]
    )
    subject_direction = pa._summarize_direction_success(
        direction_trials,
        ["subject_id", "stimulus_role", "direction_label"],
    )

    contrasts = pa.compute_direction_label_participant_contrasts(
        direction_trials,
        subject_direction,
    )

    sw = contrasts[
        (contrasts["stimulus_role"] == "comparison")
        & (contrasts["direction_label"] == "SW")
    ].iloc[0]
    assert sw["n_subjects"] == 2
    assert sw["mean_direction_minus_other"] < 0
    assert "wilcoxon_p_directional_less" in contrasts.columns


def test_direction_success_batch_plan_covers_groups_subjects_and_excludes_not_finish(
    tmp_path: Path,
) -> None:
    for folder in [
        "L_E_1",
        "L_E_2 filter",
        "N_E_3",
        "N_E_4 not finish",
        "L_P_5",
    ]:
        subject_dir = tmp_path / folder
        subject_dir.mkdir()
        (subject_dir / "answers.csv").write_text("answer\n0\n", encoding="utf-8")

    plan = pa.build_direction_success_batch_plan(tmp_path)

    labels = set(plan["batch_label"])
    assert {"L_N_E_non_filter", "L_N_E_filter_only", "L_E_non_filter", "N_E_filter_only"}.issubset(labels)
    assert "subject_L_E_1_non_filter" in labels
    assert "subject_L_E_2_filter_only" in labels
    assert "subject_N_E_3_non_filter" in labels
    assert not plan["source_subject_folder"].astype(str).str.contains("not finish", case=False).any()
    assert "L_P_5" not in set(plan["source_subject_folder"])

