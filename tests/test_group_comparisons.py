from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "Kinematics"))
sys.path.insert(0, str(ROOT / "analysis" / "probing_analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "psychophysics_analysis"))

from analysis.group_comparisons import (  # noqa: E402
    add_experiment_group_columns,
    add_protocol_group_columns,
    add_setup_factor_columns,
    compute_analysis_scope_tables,
    compute_group_comparison_tables,
    compute_protocol_group_comparison_tables,
    compute_setup_balance,
    compute_setup_factor_tables,
)
import kinematics_analysis as ka  # noqa: E402
import probing_analysis as pa  # noqa: E402
import twoafc_psychophysics as psych  # noqa: E402


def test_shared_group_comparisons_infer_groups_and_compare_within_and_between() -> None:
    df = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "metric_value": 1.0},
            {"subject_id": "N_E01", "finger_condition": "M", "metric_value": 3.0},
            {"subject_id": "L_E01", "finger_condition": "I", "metric_value": 4.0},
            {"subject_id": "L_E01", "finger_condition": "M", "metric_value": 6.0},
            {"subject_id": "L_P01", "finger_condition": "I", "metric_value": 7.0},
            {"subject_id": "L_P01", "finger_condition": "M", "metric_value": 9.0},
        ]
    )

    prepared = add_experiment_group_columns(df)
    assert set(prepared["experiment_group"].dropna()) == {"N_E", "L_E"}
    protocol_prepared = add_protocol_group_columns(df)
    assert set(protocol_prepared["protocol_group"].dropna()) == {"L_P"}

    tables = compute_group_comparison_tables(df, metric_columns=["metric_value"], condition_cols=["finger_condition"])
    assert set(tables) == {
        "group_metric_summary",
        "group_condition_metric_summary",
        "within_group_condition_comparisons",
        "between_group_metric_comparisons",
    }
    assert set(tables["group_metric_summary"]["experiment_group"]) == {"N_E", "L_E"}

    protocol_tables = compute_protocol_group_comparison_tables(df, metric_columns=["metric_value"], condition_cols=["finger_condition"])
    assert set(protocol_tables["protocol_group_metric_summary"]["protocol_group"]) == {"L_P"}

    within_ne = tables["within_group_condition_comparisons"][
        (tables["within_group_condition_comparisons"]["experiment_group"] == "N_E")
        & (tables["within_group_condition_comparisons"]["comparison"] == "M - I")
    ].iloc[0]
    assert np.isclose(within_ne["mean_difference_b_minus_a"], 2.0)
    assert np.isclose(within_ne["paired_mean_difference_b_minus_a"], 2.0)

    between_all = tables["between_group_metric_comparisons"][
        (tables["between_group_metric_comparisons"]["condition_col"] == "all")
        & (tables["between_group_metric_comparisons"]["comparison"] == "L_E - N_E")
    ].iloc[0]
    assert np.isclose(between_all["mean_difference_b_minus_a"], 3.0)



def test_setup_factor_inference_balance_and_comparisons() -> None:
    df = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "metric_value": 1.0},
            {"subject_id": "N_E02", "finger_condition": "I", "metric_value": 2.0},
            {"subject_id": "L_E01", "finger_condition": "I", "metric_value": 4.0},
            {"subject_id": "L_P01", "finger_condition": "M", "metric_value": 8.0},
            {"subject_id": "E3", "finger_condition": "M", "metric_value": 3.0},
            {"subject_id": "P2", "finger_condition": "M", "metric_value": 9.0},
            {"subject_id": "explicit", "setup_factor": "airsled", "finger_condition": "I", "metric_value": 6.0},
        ]
    )

    prepared = add_setup_factor_columns(df)
    assert prepared.loc[prepared["subject_id"].str.startswith("N"), "setup_factor"].eq("no_airsled").all()
    assert prepared.loc[prepared["subject_id"].str.startswith("L"), "setup_factor"].eq("airsled").all()
    assert prepared.loc[prepared["subject_id"] == "E3", "setup_factor"].iloc[0] == "no_airsled"
    assert prepared.loc[prepared["subject_id"] == "P2", "setup_factor"].iloc[0] == "airsled"
    assert prepared.loc[prepared["subject_id"] == "explicit", "setup_factor"].iloc[0] == "airsled"

    balance = compute_setup_balance(df, recommended_total_subjects=4)
    assert balance["can_test_setup_factor_recommended_power"].all()

    tables = compute_setup_factor_tables(
        df,
        metric_columns=["metric_value"],
        condition_cols=["finger_condition"],
        recommended_total_subjects=24,
    )
    assert "setup_balance" in tables
    assert "setup_metric_summary" in tables
    assert "between_setup_metric_comparisons" in tables
    underpowered = tables["setup_balance"]
    assert underpowered["visual_check_recommended"].all()
    comparison = tables["between_setup_metric_comparisons"][
        (tables["between_setup_metric_comparisons"]["comparison"] == "airsled - no_airsled")
        & (tables["between_setup_metric_comparisons"]["metric"] == "metric_value")
    ].iloc[0]
    assert np.isclose(comparison["mean_difference_b_minus_a"], 4.75)


def test_shared_scope_tables_cover_all_subgroups_and_single_participants() -> None:
    df = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "metric_value": 1.0},
            {"subject_id": "N_E01", "finger_condition": "M", "metric_value": 3.0},
            {"subject_id": "L_E01", "finger_condition": "I", "metric_value": 4.0},
            {"subject_id": "L_E01", "finger_condition": "M", "metric_value": 6.0},
            {"subject_id": "L_P01", "finger_condition": "I", "metric_value": 7.0},
            {"subject_id": "L_P01", "finger_condition": "M", "metric_value": 9.0},
        ]
    )

    tables = compute_analysis_scope_tables(df, metric_columns=["metric_value"], condition_cols=["finger_condition"])
    summary = tables["analysis_scope_metric_summary"]

    assert ("all", "all_participants") in set(zip(summary["analysis_scope"], summary["analysis_scope_value"]))
    assert ("subgroup", "L") in set(zip(summary["analysis_scope"], summary["analysis_scope_value"]))
    assert ("subgroup", "P") in set(zip(summary["analysis_scope"], summary["analysis_scope_value"]))
    assert ("subgroup", "N") in set(zip(summary["analysis_scope"], summary["analysis_scope_value"]))
    assert ("participant", "N_E01") in set(zip(summary["analysis_scope"], summary["analysis_scope_value"]))

    all_metric = summary[
        (summary["analysis_scope"] == "all")
        & (summary["analysis_scope_value"] == "all_participants")
        & (summary["metric"] == "metric_value")
    ].iloc[0]
    assert np.isclose(all_metric["mean"], 5.0)

    participant_within = tables["within_analysis_scope_condition_comparisons"]
    ne_subject = participant_within[
        (participant_within["analysis_scope"] == "participant")
        & (participant_within["analysis_scope_value"] == "N_E01")
        & (participant_within["comparison"] == "M - I")
    ].iloc[0]
    assert np.isclose(ne_subject["mean_difference_b_minus_a"], 2.0)


def test_kinematics_group_comparison_section_returns_expected_tables() -> None:
    subject_summary = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "stiffness_value": 40, "success_rate": 0.5, "mean_speed_px_s": 10.0},
            {"subject_id": "N_E01", "finger_condition": "M", "stiffness_value": 40, "success_rate": 0.6, "mean_speed_px_s": 12.0},
            {"subject_id": "L_E01", "finger_condition": "I", "stiffness_value": 40, "success_rate": 0.7, "mean_speed_px_s": 14.0},
            {"subject_id": "L_P01", "finger_condition": "I", "stiffness_value": 40, "success_rate": 0.8, "mean_speed_px_s": 16.0},
        ]
    )

    tables = ka.compute_experiment_group_comparisons(subject_summary)

    assert "kinematic_group_metric_summary" in tables
    assert "kinematic_within_group_condition_comparisons" in tables
    assert "kinematic_between_group_metric_comparisons" in tables
    assert "kinematic_analysis_scope_metric_summary" in tables
    assert "kinematic_within_analysis_scope_condition_comparisons" in tables
    assert "kinematic_setup_balance" in tables
    assert "kinematic_between_setup_metric_comparisons" in tables
    assert set(tables["kinematic_group_metric_summary"]["experiment_group"]) == {"N_E", "L_E"}
    scope_values = set(tables["kinematic_analysis_scope_metric_summary"]["analysis_scope_value"])
    assert {"all_participants", "N", "L", "P", "N_E01"}.issubset(scope_values)
    between = tables["kinematic_between_group_metric_comparisons"]
    le_minus_ne = between[
        (between["condition_col"] == "all")
        & (between["comparison"] == "L_E - N_E")
        & (between["metric"] == "mean_speed_px_s")
    ].iloc[0]
    assert np.isclose(le_minus_ne["mean_difference_b_minus_a"], 3.0)

    trial_summary = subject_summary.assign(
        trial_index_raw=[1, 2, 1, 1],
        correct_response=subject_summary["success_rate"],
        mean_x_centered_px=[1.0, 2.0, 3.0, 4.0],
        mean_y_centered_px=[1.0, 1.0, 1.0, 1.0],
        mean_r_center_px=[4.0, 5.0, 6.0, 7.0],
        max_r_center_px=[5.0, 6.0, 7.0, 8.0],
        mean_vx_px_s=[1.0, 1.0, 1.0, 1.0],
        mean_vy_px_s=[1.0, 1.0, 1.0, 1.0],
        mean_ax_px_s2=[0.1, 0.1, 0.1, 0.1],
        mean_ay_px_s2=[0.1, 0.1, 0.1, 0.1],
        mean_acceleration_px_s2=[0.2, 0.2, 0.2, 0.2],
        path_length_px=[10.0, 12.0, 14.0, 16.0],
        straightness_index=[0.9, 0.8, 0.7, 0.6],
        dominant_movement_angle_deg=[0.0, 45.0, 90.0, 135.0],
        dominant_movement_direction=["E", "NE", "N", "NW"],
    )
    summarized = ka.summarize_kinematics(trial_summary, pd.DataFrame())
    assert "kinematic_group_metric_summary" in summarized
    assert "kinematic_between_group_metric_comparisons" in summarized


def test_probing_group_comparisons_are_returned_by_summary() -> None:
    probing_trial_summary = pd.DataFrame(
        [
            {"subject_id": "N_E01", "subject_group": "N_E", "finger_condition": "I", "comparison_value": 40, "stiffness_value": 40, "trial_index_raw": 1, "correct_response": 1, "probe_count": 1, "probe_rate_per_s": 0.5, "center_visit_count": 2, "unique_probe_directions": 1, "mean_probe_peak_radius_px": 90, "mean_probe_duration_s": 0.2, "path_length_px": 100, "mean_speed_px_s": 5},
            {"subject_id": "L_E01", "subject_group": "L_E", "finger_condition": "I", "comparison_value": 40, "stiffness_value": 40, "trial_index_raw": 1, "correct_response": 0, "probe_count": 3, "probe_rate_per_s": 1.5, "center_visit_count": 4, "unique_probe_directions": 2, "mean_probe_peak_radius_px": 95, "mean_probe_duration_s": 0.3, "path_length_px": 120, "mean_speed_px_s": 7},
            {"subject_id": "L_P01", "subject_group": "L_P", "finger_condition": "M", "comparison_value": 40, "stiffness_value": 40, "trial_index_raw": 1, "correct_response": 1, "probe_count": 5, "probe_rate_per_s": 2.5, "center_visit_count": 6, "unique_probe_directions": 3, "mean_probe_peak_radius_px": 105, "mean_probe_duration_s": 0.4, "path_length_px": 140, "mean_speed_px_s": 9},
        ]
    )
    event_log = pd.DataFrame()

    tables = pa.summarize_probing(probing_trial_summary, event_log)

    assert "probing_group_metric_summary" in tables
    assert "probing_between_group_metric_comparisons" in tables
    assert "probing_analysis_scope_metric_summary" in tables
    assert "probing_within_analysis_scope_condition_comparisons" in tables
    assert "probing_setup_balance" in tables
    assert "probing_between_setup_metric_comparisons" in tables
    assert set(tables["probing_group_metric_summary"]["experiment_group"]) == {"N_E", "L_E"}
    assert {"all_participants", "N", "L", "P", "N_E01"}.issubset(set(tables["probing_analysis_scope_metric_summary"]["analysis_scope_value"]))


def test_psychophysics_group_comparison_section_returns_trial_and_fit_tables() -> None:
    clean = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "comparison_value": 40, "standard_value": 85, "response_comparison_greater": 0, "reaction_time": 1.0, "sex": "female", "age": 21},
            {"subject_id": "N_E01", "finger_condition": "M", "comparison_value": 120, "standard_value": 85, "response_comparison_greater": 1, "reaction_time": 1.2, "sex": "female", "age": 21},
            {"subject_id": "L_E01", "finger_condition": "I", "comparison_value": 40, "standard_value": 85, "response_comparison_greater": 1, "reaction_time": 2.0, "sex": "male", "age": 28},
            {"subject_id": "L_E01", "finger_condition": "M", "comparison_value": 120, "standard_value": 85, "response_comparison_greater": 1, "reaction_time": 2.2, "sex": "male", "age": 28},
            {"subject_id": "L_P01", "finger_condition": "I", "comparison_value": 5, "standard_value": 85, "response_comparison_greater": 0, "reaction_time": 2.8, "sex": "female", "age": 31},
            {"subject_id": "L_P01", "finger_condition": "M", "comparison_value": 165, "standard_value": 85, "response_comparison_greater": 1, "reaction_time": 3.0, "sex": "female", "age": 31},
            {"subject_id": "N_P01", "finger_condition": "I", "comparison_value": 5, "standard_value": 85, "response_comparison_greater": 0, "reaction_time": 2.6, "sex": "female", "age": 24},
            {"subject_id": "N_P01", "finger_condition": "M", "comparison_value": 165, "standard_value": 85, "response_comparison_greater": 1, "reaction_time": 2.9, "sex": "female", "age": 24},
        ]
    )
    fits = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "pse": 80.0, "jnd": 10.0, "n_trials": 12},
            {"subject_id": "L_E01", "finger_condition": "I", "pse": 85.0, "jnd": 11.0, "n_trials": 12},
            {"subject_id": "L_P01", "finger_condition": "M", "pse": 90.0, "jnd": 12.0, "n_trials": 12},
            {"subject_id": "N_P01", "finger_condition": "M", "pse": 75.0, "jnd": 13.0, "n_trials": 12},
        ]
    )

    tables = psych.compute_experiment_group_comparisons(clean, pse_jnd_by_subject_finger=fits)

    assert "psychophysics_trial_group_metric_summary" in tables
    assert "psychophysics_trial_analysis_scope_metric_summary" in tables
    assert "psychophysics_trial_setup_balance" in tables
    assert "psychophysics_fit_by_subject_finger_between_group_metric_comparisons" in tables
    assert "psychophysics_trial_protocol_group_metric_summary" in tables
    assert "psychophysics_fit_by_subject_finger_protocol_group_metric_summary" in tables
    assert "psychophysics_fit_by_subject_finger_between_setup_metric_comparisons" in tables
    assert "psychophysics_fit_by_subject_finger_within_analysis_scope_condition_comparisons" in tables
    assert "psychophysics_fit_subject_pooled_between_group_metric_comparisons" in tables
    fit_summary = tables["psychophysics_fit_by_subject_finger_group_metric_summary"]
    assert "weber_fraction" in set(fit_summary["metric"])
    assert "abs_pse_delta_from_standard" in set(fit_summary["metric"])
    assert set(tables["psychophysics_trial_group_metric_summary"]["experiment_group"]) == {"N_E", "L_E"}
    assert set(tables["psychophysics_trial_protocol_group_metric_summary"]["protocol_group"]) == {"N_P", "L_P"}
    assert set(tables["psychophysics_fit_by_subject_finger_protocol_group_metric_summary"]["protocol_group"]) == {"N_P", "L_P"}
    main_scope_values = set(tables["psychophysics_trial_analysis_scope_metric_summary"]["analysis_scope_value"])
    assert {"all_participants", "N", "L", "N_E01"}.issubset(main_scope_values)
    assert "P" not in main_scope_values
    trial_success = tables["psychophysics_trial_group_metric_summary"]
    assert "correct_response" in set(trial_success["metric"])
    assert "comparison_over_standard" in set(trial_success["metric"])
    assert "raw_values_json" in trial_success.columns
    trial_conditions = tables["psychophysics_trial_group_condition_metric_summary"]
    assert {"success_label", "workspace_setup", "sex_factor", "age_group"}.intersection(trial_conditions.columns)
    factor_stats = tables["psychophysics_factor_statistics"]
    assert not factor_stats.empty
    assert {"one_way_anova", "two_way_anova"}.intersection(set(factor_stats["model_type"]))
    fit_between = tables["psychophysics_fit_by_subject_finger_between_group_metric_comparisons"]
    le_minus_ne = fit_between[
        (fit_between["condition_col"] == "all")
        & (fit_between["comparison"] == "L_E - N_E")
        & (fit_between["metric"] == "pse")
    ].iloc[0]
    assert np.isclose(le_minus_ne["mean_difference_b_minus_a"], 5.0)



def test_psychophysics_fit_delta_columns_fill_missing_standard_value() -> None:
    fits = pd.DataFrame([{"pse": 90.0, "jnd": 8.5, "standard_value": np.nan}])

    out = psych.add_fit_delta_columns(fits)

    assert np.isclose(out["standard_value"].iloc[0], psych.STANDARD_FALLBACK)
    assert np.isclose(out["pse_delta_from_standard"].iloc[0], 5.0)
    assert np.isclose(out["weber_fraction"].iloc[0], 0.1)


def test_psychophysics_answer_discovery_ignores_old_and_not_finish_paths(tmp_path: Path) -> None:
    output = tmp_path.parent / f"{tmp_path.name}_out"
    for path in [
        tmp_path / "N_E01" / "run" / "answers.csv",
        tmp_path / "N_E01_old" / "run" / "answers.csv",
        tmp_path / "L_E01" / "not finish run" / "answers.csv",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("timestamp,answer\n2026-01-01,1\n", encoding="utf-8")

    discovered = psych.discover_answer_files(tmp_path, output)
    selected = discovered[discovered["selected"]]

    assert selected["subject_id"].tolist() == ["N_E01"]
    assert not selected["source_file"].str.lower().str.contains("old|not finish").any()


def test_psychophysics_tracking_analysis_recovers_malformed_tracking_csv(tmp_path: Path) -> None:
    tracking_path = tmp_path / "tracking.csv"
    tracking_path.write_bytes(
        b"timestamp,interacting,stiffness,object_x,object_y,finger,thumb_x,thumb_y,active_finger_x,active_finger_y\n"
        b"2026-05-04T14:56:47.000000,False,85,320.0,240.0,index,421,294,469,260\n"
        b"2026-05-04T14:56:48.000000,True,85,330.0,245.0,index,422,295,470,261\n"
        b'bad,"unterminated,\xb5,not,a,row\n'
    )

    summary, trajectory = psych._analyze_tracking_file(tracking_path, n_time_bins=4)

    assert summary["tracking_warning"] == "tracking_csv_recovered_skipped_1_malformed_rows"
    assert summary["n_tracking_samples"] == 2
    assert summary["tracking_duration_seconds"] == 1.0
    assert not trajectory.empty


def test_group_comparison_save_helpers_write_expected_csvs(tmp_path: Path) -> None:
    subject_summary = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "stiffness_value": 40, "success_rate": 0.5, "mean_speed_px_s": 10.0},
            {"subject_id": "L_E01", "finger_condition": "I", "stiffness_value": 40, "success_rate": 0.7, "mean_speed_px_s": 14.0},
            {"subject_id": "L_P01", "finger_condition": "I", "stiffness_value": 40, "success_rate": 0.8, "mean_speed_px_s": 16.0},
        ]
    )
    ka.save_experiment_group_comparison_outputs(tmp_path, subject_summary)
    assert (tmp_path / "kinematic_group_metric_summary.csv").exists()
    assert (tmp_path / "kinematic_between_group_metric_comparisons.csv").exists()
    assert (tmp_path / "kinematic_analysis_scope_metric_summary.csv").exists()
    assert (tmp_path / "kinematic_scope_figure_manifest.csv").exists()

    clean = pd.DataFrame(
        [
            {"subject_id": "N_E01", "finger_condition": "I", "comparison_value": 40, "standard_value": 85, "response_comparison_greater": 0, "reaction_time": 1.0},
            {"subject_id": "L_E01", "finger_condition": "I", "comparison_value": 40, "standard_value": 85, "response_comparison_greater": 1, "reaction_time": 2.0},
            {"subject_id": "L_P01", "finger_condition": "I", "comparison_value": 40, "standard_value": 85, "response_comparison_greater": 1, "reaction_time": 3.0},
        ]
    )
    psych.save_experiment_group_comparison_outputs(tmp_path, clean)
    assert (tmp_path / "psychophysics_trial_group_metric_summary.csv").exists()
    assert (tmp_path / "psychophysics_trial_protocol_group_metric_summary.csv").exists()
    assert (tmp_path / "psychophysics_trial_between_group_metric_comparisons.csv").exists()
    assert (tmp_path / "psychophysics_trial_analysis_scope_metric_summary.csv").exists()
    assert (tmp_path / "psychophysics_factor_statistics.csv").exists()
    assert (tmp_path / "motor_control_method_references.csv").exists()
    assert (tmp_path / "psychophysics_scope_figure_manifest.csv").exists()
