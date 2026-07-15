from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "Kinematics"))
import kinematics_analysis as ka  # noqa: E402


def _write_tracking(path: Path) -> None:
    rows = [
        ("2026-01-01T00:00:00", False, 40, 320, 240, "index"),
        ("2026-01-01T00:00:01", True, 40, 321, 240, "index"),
        ("2026-01-01T00:00:02", True, 40, 323, 240, "index"),
        ("2026-01-01T00:00:03", False, 85, 330, 240, "index"),
        ("2026-01-01T00:00:04", True, 85, 331, 240, "index"),
        ("2026-01-01T00:00:05", True, 85, 333, 240, "index"),
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "interacting", "stiffness", "object_x", "object_y", "finger"])
        writer.writerows(rows)


def test_discover_trials_reads_headered_answers_without_row_shift(tmp_path: Path) -> None:
    subject = tmp_path / "E1" / "run"
    pair = subject / "pair_001"
    pair.mkdir(parents=True)
    _write_tracking(pair / "tracking.csv")
    (pair / "side_camera.mp4").write_bytes(b"not a real video")
    with open(subject / "answers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "pair_number", "object_1_finger", "object_1_stiffness", "object_2_finger", "object_2_stiffness", "time_to_answer", "answer"])
        writer.writerow(["2026-01-01T00:00:06", 1, "index", 40, "index", 85, 1.2, 1])

    manifest = ka.discover_trials(tmp_path, pilot_onboarding_trials=0)

    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["trial_index_raw"] == 1
    assert row["pair_number"] == 1
    assert row["comparison_value"] == 40
    assert row["standard_value"] == 85
    assert row["correct_response"] == 1
    assert row["object_1_finger"] == "I"
    assert row["object_2_finger"] == "I"
    assert row["finger_condition"] == "I"
    assert Path(row["tracking_file"]).name == "tracking.csv"
    assert Path(row["tracking_file"]).parent.name == "pair_001"


def test_discover_trials_ignores_old_and_not_finish_paths(tmp_path: Path) -> None:
    current = tmp_path / "N_E_1" / "run"
    old = tmp_path / "N_E_1_old" / "run"
    unfinished = tmp_path / "L_E_1" / "not finish run"
    for subject in [current, old, unfinished]:
        pair = subject / "pair_001"
        pair.mkdir(parents=True)
        _write_tracking(pair / "tracking.csv")
        with open(subject / "answers.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "pair_number", "object_1_finger", "object_1_stiffness", "object_2_finger", "object_2_stiffness", "time_to_answer", "answer"])
            writer.writerow(["2026-01-01T00:00:06", 1, "index", 40, "index", 85, 1.2, 1])

    manifest = ka.discover_trials(tmp_path, pilot_onboarding_trials=0)

    assert manifest["subject_id"].tolist() == ["N_E_1"]


def test_discover_trials_filter_folders_are_flagged_only_for_group_selection(
    tmp_path: Path,
) -> None:
    for subject in ["L_E_1", "L_E_16 (filter)", "L_E_13 (not finish)"]:
        pair = tmp_path / subject / "run" / "pair_001"
        pair.mkdir(parents=True)
        _write_tracking(pair / "tracking.csv")
        with open(pair.parent / "answers.csv", "w", newline="") as f:
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

    with_filtered = ka.discover_trials(
        tmp_path,
        selection="L_E",
        exclude_filter_folders=False,
        pilot_onboarding_trials=0,
    )
    without_filtered = ka.discover_trials(
        tmp_path,
        selection="L_E",
        exclude_filter_folders=True,
        pilot_onboarding_trials=0,
    )
    explicit_filtered_subject = ka.discover_trials(
        tmp_path,
        selection="L_E_16",
        exclude_filter_folders=True,
        pilot_onboarding_trials=0,
    )
    filtered_only = ka.discover_trials(
        tmp_path,
        selection="L_E_FILTER_ONLY",
        exclude_filter_folders=False,
        pilot_onboarding_trials=0,
    )
    all_filtered_only = ka.discover_trials(
        tmp_path,
        selection="FILTER_ONLY",
        exclude_filter_folders=False,
        pilot_onboarding_trials=0,
    )

    assert with_filtered["subject_id"].tolist() == ["L_E_1", "L_E_16 (filter)"]
    assert without_filtered["subject_id"].tolist() == ["L_E_1"]
    assert explicit_filtered_subject["subject_id"].tolist() == ["L_E_16 (filter)"]
    assert filtered_only["subject_id"].tolist() == ["L_E_16 (filter)"]
    assert all_filtered_only["subject_id"].tolist() == ["L_E_16 (filter)"]
    assert ka.selection_label("L_E_FILTER") == "L_E_FILTER_ONLY"
    assert (
        ka.selection_output_label("L_E", exclude_filter_folders=False)
        == "L_E_before_filter"
    )
    assert ka.selection_output_label("L_E", exclude_filter_folders=True) == "L_E"
    assert (
        ka.selection_output_label("L_E_FILTER_ONLY", exclude_filter_folders=False)
        == "L_E_FILTER_ONLY"
    )
    assert (
        ka.selection_output_label("FILTER_ONLY", exclude_filter_folders=False)
        == "FILTER_ONLY"
    )
    assert (
        ka.selection_output_label("L_E_16", exclude_filter_folders=False) == "L_E_16"
    )
    custom = ["L_E_1", "L_E_15", "L_E_19"]
    assert ka.selection_label(custom) == "L_E_1_L_E_15_L_E_19"
    assert (
        ka.selection_output_label(custom, exclude_filter_folders=False)
        == "L_E_1_L_E_15_L_E_19"
    )


def test_discover_trials_drops_pilot_onboarding_trials(tmp_path: Path) -> None:
    """The first ``pilot_onboarding_trials`` rows per subject are pilot/onboarding
    and must be excluded from the discovered manifest."""
    subject = tmp_path / "E1" / "run"
    n_trials = 15
    for trial_index in range(1, n_trials + 1):
        pair = subject / f"pair_{trial_index:03d}"
        pair.mkdir(parents=True)
        _write_tracking(pair / "tracking.csv")
        (pair / "side_camera.mp4").write_bytes(b"not a real video")
    with open(subject / "answers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "pair_number", "object_1_finger", "object_1_stiffness", "object_2_finger", "object_2_stiffness", "time_to_answer", "answer"])
        for trial_index in range(1, n_trials + 1):
            writer.writerow(["2026-01-01T00:00:06", trial_index, "index", 40, "index", 85, 1.2, 1])

    manifest = ka.discover_trials(tmp_path, pilot_onboarding_trials=12)
    assert manifest["trial_index_raw"].tolist() == [13, 14, 15]

    full_manifest = ka.discover_trials(tmp_path, pilot_onboarding_trials=0)
    assert full_manifest["trial_index_raw"].tolist() == list(range(1, n_trials + 1))


def test_compute_tracking_kinematics_splits_derivatives_by_stiffness(tmp_path: Path) -> None:
    pair = tmp_path / "pair_001"
    pair.mkdir()
    tracking = pair / "tracking.csv"
    _write_tracking(tracking)
    trials = pd.DataFrame(
        [
            {
                "subject_id": "N_E_1",
                "subject_group": "N_E",
                "selected": True,
                "trial_index_raw": 1,
                "pair_number": 1,
                "pair_dir": str(pair),
                "tracking_file": str(tracking),
                "side_video_file": str(pair / "side_camera.mp4"),
                "top_video_file": str(pair / "top_camera.mp4"),
                "tracking_exists": True,
                "side_video_exists": False,
                "comparison_value": 40,
                "standard_value": 85,
                "signed_stiffness_delta": -45,
                "correct_response": 1,
            }
        ]
    )

    kin = ka.compute_tracking_kinematics(trials, center_x=320, center_y=240, n_time_bins=3)
    samples = kin["samples"]
    summary = kin["trial_summary"]

    assert summary["stiffness_value"].tolist() == [40, 85]
    assert set(samples["stiffness_value"]) == {40, 85}
    first_85 = samples[samples["stiffness_value"] == 85].iloc[0]
    assert np.isnan(first_85["vx_px_s"])
    assert np.isnan(first_85["ax_px_s2"])
    assert {"mean_vx_px_s", "mean_vy_px_s", "mean_ax_px_s2", "mean_ay_px_s2"}.issubset(summary.columns)
    assert {"jx_px_s3", "jy_px_s3", "jerk_px_s3"}.issubset(samples.columns)
    assert {"curvature_1_px", "mean_curvature_1_px", "speed_curvature_power_law_slope"}.issubset(
        {*samples.columns, *summary.columns}
    )
    assert {"mean_jerk_px_s3", "max_jerk_px_s3", "normalized_jerk_cost"}.issubset(summary.columns)
    assert summary["finger_condition"].tolist() == ["I", "I"]
    assert {"workspace_setup", "workspace_label", "mean_r_workspace_normalized"}.issubset(summary.columns)
    assert {"thumb_active_span_px", "hand_orientation_xy_deg"}.issubset(samples.columns)
    assert set(summary["workspace_setup"]) == {"N"}
    assert summary["workspace_label"].str.contains("40x50 cm").all()
    assert set(summary["side_camera_side"]) == {"left"}
    assert set(summary["participant_position_context"]) == {"centered"}


def test_estimate_side_video_z_prefers_z_tracking_and_interpolates_missing_frames(
    tmp_path: Path,
) -> None:
    pair = tmp_path / "pair_001"
    pair.mkdir()
    side_video = pair / "side_camera.mp4"
    side_video.write_bytes(b"not a real video")
    pd.DataFrame(
        {
            "timestamp_s": [0.0, 1.0, 2.0, 3.0],
            "z_active_finger_lift_px": [0.0, np.nan, 20.0, 30.0],
            "z_thumb_lift_px": [1.0, np.nan, 21.0, 31.0],
            "z_hand_midpoint_lift_px": [0.5, np.nan, 20.5, 30.5],
            "z_tracking_flag": ["ok", "no_hand_detected", "ok", "ok"],
        }
    ).to_csv(pair / "z_tracking.csv", index=False)
    trial_summary = pd.DataFrame(
        [
            {
                "subject_id": "N_E_1",
                "subject_group": "N",
                "experiment_group": "N_E",
                "trial_index_raw": 1,
                "pair_number": 1,
                "finger_condition": "I",
                "stiffness_value": 40.0,
                "stiffness_segment_id": 1,
                "stiffness_order_in_trial": 1,
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "signed_stiffness_delta": -45.0,
                "correct_response": 1.0,
                "side_video_file": str(side_video),
                "side_video_exists": True,
                "stiffness_start_fraction": 0.0,
                "stiffness_end_fraction": 1.0,
            }
        ]
    )

    result = ka.estimate_side_video_z(trial_summary)

    samples = result["side_samples"].sort_values("side_time_s")
    assert samples["side_z_source"].str.contains("z_tracking.csv").all()
    assert samples["side_detected"].tolist() == [1, 0, 1, 1]
    assert np.allclose(samples["side_z_lift_px"], [0.0, 10.0, 20.0, 30.0])
    missing = samples[samples["side_detected"] == 0].iloc[0]
    assert missing["side_z_interpolated"] == 1
    assert missing["z_tracking_warning"] == "interpolated_no_hand_detection"
    trial = result["side_trial_summary"].iloc[0]
    assert np.isclose(trial["side_detection_rate"], 0.75)
    assert np.isclose(trial["mean_side_z_lift_px"], 15.0)
    assert np.isclose(trial["max_side_z_lift_px"], 30.0)


def test_estimate_side_video_z_honors_samples_per_video_limit(
    tmp_path: Path,
) -> None:
    pair = tmp_path / "pair_001"
    pair.mkdir()
    side_video = pair / "side_camera.mp4"
    side_video.write_bytes(b"not a real video")
    pd.DataFrame(
        {
            "timestamp_s": np.arange(10, dtype=float),
            "z_active_finger_lift_px": np.arange(10, dtype=float),
            "z_tracking_flag": ["ok"] * 10,
        }
    ).to_csv(pair / "z_tracking.csv", index=False)
    trial_summary = pd.DataFrame(
        [
            {
                "subject_id": "N_E_1",
                "subject_group": "N",
                "experiment_group": "N_E",
                "trial_index_raw": 1,
                "pair_number": 1,
                "finger_condition": "I",
                "stiffness_value": 40.0,
                "stiffness_segment_id": 1,
                "stiffness_order_in_trial": 1,
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "signed_stiffness_delta": -45.0,
                "correct_response": 1.0,
                "side_video_file": str(side_video),
                "side_video_exists": True,
                "stiffness_start_fraction": 0.0,
                "stiffness_end_fraction": 1.0,
            }
        ]
    )

    result = ka.estimate_side_video_z(trial_summary, samples_per_video=4)

    samples = result["side_samples"].sort_values("side_time_s")
    assert samples["side_time_s"].tolist() == [0.0, 3.0, 6.0, 9.0]
    assert samples["side_z_lift_px"].tolist() == [0.0, 3.0, 6.0, 9.0]
    trial = result["side_trial_summary"].iloc[0]
    assert trial["n_side_samples"] == 4
    assert np.isclose(trial["mean_side_z_lift_px"], 4.5)


def test_tracking_derivatives_ignore_tiny_timestamp_artifacts(tmp_path: Path) -> None:
    pair = tmp_path / "pair_001"
    pair.mkdir()
    tracking = pair / "tracking.csv"
    with open(tracking, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "interacting", "stiffness", "object_x", "object_y", "finger"]
        )
        writer.writerows(
            [
                ("2026-01-01T00:00:00.000", True, 40, 320, 240, "index"),
                # 200 px in 1 ms is a tracker timestamp artifact; derivative should
                # be NaN for this gap, not an impossible 200,000 px/s spike.
                ("2026-01-01T00:00:00.001", True, 40, 520, 240, "index"),
                ("2026-01-01T00:00:01.000", True, 40, 521, 240, "index"),
                ("2026-01-01T00:00:02.000", True, 40, 522, 240, "index"),
            ]
        )
    trial_manifest = pd.DataFrame(
        [
            {
                "subject_id": "N_E_1",
                "subject_group": "N",
                "experiment_group": "N_E",
                "selected": True,
                "trial_index_raw": 1,
                "pair_number": 1,
                "pair_dir": str(pair),
                "tracking_file": str(tracking),
                "side_video_file": str(pair / "side_camera.mp4"),
                "top_video_file": str(pair / "top_camera.mp4"),
                "tracking_exists": True,
                "side_video_exists": False,
                "comparison_value": 40,
                "standard_value": 85,
                "signed_stiffness_delta": -45,
                "correct_response": 1,
            }
        ]
    )

    result = ka.compute_tracking_kinematics(trial_manifest)

    samples = result["samples"].sort_values("time_s")
    assert np.isnan(samples.iloc[1]["vx_px_s"])
    assert np.nanmax(np.abs(samples["vx_px_s"])) < 10.0
    assert np.nanmax(np.abs(samples["radial_velocity_px_s"])) < 10.0


def test_estimate_side_video_z_does_not_fallback_when_z_tracking_missing(
    tmp_path: Path,
) -> None:
    pair = tmp_path / "pair_001"
    pair.mkdir()
    side_video = pair / "side_camera.mp4"
    side_video.write_bytes(b"not a real video")
    trial_summary = pd.DataFrame(
        [
            {
                "subject_id": "N_E_1",
                "subject_group": "N",
                "experiment_group": "N_E",
                "trial_index_raw": 1,
                "pair_number": 1,
                "finger_condition": "I",
                "stiffness_value": 40.0,
                "stiffness_segment_id": 1,
                "stiffness_order_in_trial": 1,
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "signed_stiffness_delta": -45.0,
                "correct_response": 1.0,
                "side_video_file": str(side_video),
                "side_video_exists": True,
                "stiffness_start_fraction": 0.0,
                "stiffness_end_fraction": 1.0,
            }
        ]
    )

    result = ka.estimate_side_video_z(trial_summary)

    samples = result["side_samples"]
    assert len(samples) == 1
    row = samples.iloc[0]
    assert row["side_z_source"] == "z_tracking.csv"
    assert row["z_tracking_warning"] == "missing_z_tracking_csv"
    assert row["missing_z_tracking_csv"] == 1
    assert row["side_detected"] == 0
    assert np.isnan(row["side_z_lift_px"])
    trial = result["side_trial_summary"].iloc[0]
    assert np.isclose(trial["side_detection_rate"], 0.0)
    assert trial["z_tracking_warning"] == "missing_z_tracking_csv"
    assert trial["missing_z_tracking_csv"] == 1
    assert np.isnan(trial["mean_side_z_lift_px"])


def test_kinematic_interaction_masks_split_all_standard_and_comparison() -> None:
    df = pd.DataFrame(
        [
            {"stiffness_value": 40, "comparison_value": 40, "standard_value": 85},
            {"stiffness_value": 85, "comparison_value": 40, "standard_value": 85},
            {"stiffness_value": 130, "comparison_value": 130, "standard_value": 85},
        ]
    )

    assert ka.kinematic_interaction_mask(df, "all").tolist() == [True, True, True]
    assert ka.kinematic_interaction_mask(df, "standard").tolist() == [False, True, False]
    assert ka.kinematic_interaction_mask(df, "Comparison").tolist() == [True, False, True]
    assert ka.filter_kinematic_interaction(df, "S")["stiffness_value"].tolist() == [85]
    assert ka.filter_kinematic_interaction(df, "C")["stiffness_value"].tolist() == [40, 130]


def test_save_selected_kinematic_tree_splits_existing_subject_csvs_from_group_run(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "L_E"
    run_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"subject_id": "L_E_1", "metric": "mean_speed_px_s", "value": 1.0},
            {"subject_id": "L_E_2", "metric": "mean_speed_px_s", "value": 2.0},
        ]
    ).to_csv(run_root / "subject_velocity_acceleration_metric_distribution.csv", index=False)

    manifest = ka.save_selected_kinematic_tree(run_root, tables={})

    assert (
        tmp_path
        / "results"
        / "L_E_1"
        / "csv"
        / "subject_velocity_acceleration_metric_distribution.csv"
    ).exists()
    assert (
        tmp_path
        / "results"
        / "L_E_2"
        / "csv"
        / "subject_velocity_acceleration_metric_distribution.csv"
    ).exists()
    assert set(manifest["kind"]) == {"subject"}


def test_save_selected_kinematic_tree_does_not_rescan_single_subject_run(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "L_E_1"
    run_root.mkdir(parents=True)
    source = run_root / "subject_velocity_acceleration_metric_distribution.csv"
    pd.DataFrame(
        [{"subject_id": "L_E_1", "metric": "mean_speed_px_s", "value": 1.0}]
    ).to_csv(source, index=False)

    manifest = ka.save_selected_kinematic_tree(run_root, tables={})

    assert manifest.empty
    assert not (
        tmp_path
        / "results"
        / "L_E_1"
        / "csv"
        / "subject_velocity_acceleration_metric_distribution.csv"
    ).exists()
    assert source.exists()


def test_save_interaction_filtered_outputs_creates_only_all_folder_by_default(
    tmp_path: Path,
) -> None:
    trial_summary = pd.DataFrame(
        [
            {
                "subject_id": "N_E_1",
                "subject_group": "N_E",
                "experiment_group": "N_E",
                "trial_index_raw": 1,
                "tracking_file": "a.csv",
                "stiffness_value": 40.0,
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "finger_condition": "I",
                "correct_response": 1,
                "mean_x_centered_px": 1.0,
                "mean_y_centered_px": 2.0,
                "max_r_center_px": 3.0,
                "mean_r_center_px": 2.5,
                "path_length_px": 4.0,
                "straightness_index": 0.9,
                "mean_vx_px_s": 1.0,
                "mean_vy_px_s": 1.0,
                "mean_speed_px_s": 2.0,
                "mean_ax_px_s2": 0.1,
                "mean_ay_px_s2": 0.2,
                "mean_acceleration_px_s2": 0.3,
                "mean_jerk_px_s3": 0.4,
                "normalized_jerk_cost": 0.5,
                "dominant_movement_angle_deg": 0.0,
                "dominant_movement_direction": "E",
                "movement_direction_resultant_length": 1.0,
            },
            {
                "subject_id": "N_E_1",
                "subject_group": "N_E",
                "experiment_group": "N_E",
                "trial_index_raw": 1,
                "tracking_file": "a.csv",
                "stiffness_value": 85.0,
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "finger_condition": "I",
                "correct_response": 1,
                "mean_x_centered_px": 2.0,
                "mean_y_centered_px": 3.0,
                "max_r_center_px": 4.0,
                "mean_r_center_px": 3.5,
                "path_length_px": 5.0,
                "straightness_index": 0.8,
                "mean_vx_px_s": 1.5,
                "mean_vy_px_s": 1.5,
                "mean_speed_px_s": 3.0,
                "mean_ax_px_s2": 0.2,
                "mean_ay_px_s2": 0.3,
                "mean_acceleration_px_s2": 0.4,
                "mean_jerk_px_s3": 0.5,
                "normalized_jerk_cost": 0.6,
                "dominant_movement_angle_deg": 10.0,
                "dominant_movement_direction": "E",
                "movement_direction_resultant_length": 1.0,
            },
        ]
    )
    trajectory_time_bins = pd.DataFrame(
        [
            {
                "subject_id": "N_E_1",
                "subject_group": "N_E",
                "experiment_group": "N_E",
                "tracking_file": "a.csv",
                "trial_index_raw": 1,
                "trajectory_time_bin": 1,
                "time_fraction": 0.0,
                "stiffness_time_fraction": 0.0,
                "trial_time_fraction": 0.0,
                "stiffness_value": 40.0,
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "finger_condition": "I",
                "correct_response": 1,
                "x_centered_px": 1.0,
                "y_centered_px": 2.0,
                "r_center_px": 2.2,
                "vx_px_s": 1.0,
                "vy_px_s": 1.0,
                "speed_px_s": 2.0,
                "ax_px_s2": 0.1,
                "ay_px_s2": 0.2,
                "acceleration_px_s2": 0.3,
                "jerk_px_s3": 0.4,
                "radial_velocity_px_s": 0.1,
                "tangential_velocity_px_s": 0.2,
            },
            {
                "subject_id": "N_E_1",
                "subject_group": "N_E",
                "experiment_group": "N_E",
                "tracking_file": "a.csv",
                "trial_index_raw": 1,
                "trajectory_time_bin": 1,
                "time_fraction": 0.0,
                "stiffness_time_fraction": 0.0,
                "trial_time_fraction": 0.0,
                "stiffness_value": 85.0,
                "comparison_value": 40.0,
                "standard_value": 85.0,
                "finger_condition": "I",
                "correct_response": 1,
                "x_centered_px": 2.0,
                "y_centered_px": 3.0,
                "r_center_px": 3.6,
                "vx_px_s": 1.5,
                "vy_px_s": 1.5,
                "speed_px_s": 3.0,
                "ax_px_s2": 0.2,
                "ay_px_s2": 0.3,
                "acceleration_px_s2": 0.4,
                "jerk_px_s3": 0.5,
                "radial_velocity_px_s": 0.2,
                "tangential_velocity_px_s": 0.3,
            },
        ]
    )

    index = ka.save_interaction_filtered_kinematic_outputs(
        tmp_path,
        trial_kinematic_summary=trial_summary,
        trajectory_time_bins=trajectory_time_bins,
        save_figures=False,
    )

    assert set(index["folder_name"]) == {"all"}
    assert (tmp_path / "interaction_analysis_folder_index.csv").exists()
    assert (tmp_path / "experiment_setup_context.csv").exists()
    assert (tmp_path / "all" / "subject_kinematic_summary.csv").exists()
    assert (tmp_path / "all" / "experiment_setup_context.csv").exists()
    assert not (tmp_path / "standard").exists()
    assert not (tmp_path / "Comparison").exists()
    all_trials = pd.read_csv(tmp_path / "all" / "trial_kinematic_summary.csv")
    assert all_trials["stiffness_value"].tolist() == [40.0, 85.0]


def test_workspace_normalization_and_expanded_scope_tables() -> None:
    df = pd.DataFrame(
        [
            {"subject_id": "N_E_1", "finger_condition": "I", "stiffness_value": 40, "correct_response": 1, "mean_x_centered_px": 320.0, "mean_y_centered_px": 0.0, "mean_speed_px_s": 10.0, "sex": "female", "age": 25, "participant_type": 1},
            {"subject_id": "L_P_1", "finger_condition": "M", "stiffness_value": 85, "correct_response": 0, "mean_x_centered_px": 320.0, "mean_y_centered_px": 0.0, "mean_speed_px_s": 20.0, "sex": "male", "age": 35, "participant_type": 2},
        ]
    )
    normalized = ka.add_workspace_normalization_columns(
        df.rename(columns={"mean_x_centered_px": "x_centered_px", "mean_y_centered_px": "y_centered_px"})
    )

    n_row = normalized[normalized["subject_id"] == "N_E_1"].iloc[0]
    l_row = normalized[normalized["subject_id"] == "L_P_1"].iloc[0]
    assert np.isclose(n_row["x_workspace_cm"], 20.0)
    assert np.isclose(l_row["x_workspace_cm"], 30.0)
    assert n_row["workspace_label"] == "N workspace (40x50 cm)"
    assert l_row["workspace_label"] == "L workspace (60x60 cm)"
    assert n_row["side_camera_side"] == "left"
    assert l_row["side_camera_side"] == "right"
    assert "smaller" in n_row["movement_space_context"]
    assert "slightly left" in l_row["side_camera_interpretation_note"]

    tables = ka.compute_expanded_kinematic_scope_tables(df, metric_columns=["mean_speed_px_s"])
    status = tables["expanded_scope_status"]
    assert {"all", "protocol", "sex", "age", "stiffness", "finger", "success"}.issubset(
        set(status["comparison_scope"])
    )
    summary = tables["expanded_scope_metric_summary"]
    assert ("success", "success") in set(zip(summary["comparison_scope"], summary["comparison_value"]))
    pairs = tables["expanded_scope_pairwise_mean_differences"]
    success_pair = pairs[
        (pairs["comparison_scope"] == "success") & (pairs["metric"] == "mean_speed_px_s")
    ].iloc[0]
    assert np.isclose(abs(success_pair["mean_difference_b_minus_a"]), 10.0)


def test_side_camera_angle_normalization_mirrors_left_camera_pixels() -> None:
    df = pd.DataFrame(
        [
            {
                "subject_id": "L_E_1",
                "side_camera_side": "right",
                "side_x_from_frame_center_px": 12.0,
                "side_z_lift_px": 5.0,
            },
            {
                "subject_id": "N_E_1",
                "side_camera_side": "left",
                "side_x_from_frame_center_px": 12.0,
                "side_z_lift_px": 5.0,
            },
        ]
    )

    normalized = ka.add_side_camera_angle_normalization_columns(df)

    assert normalized["side_camera_view_sign"].tolist() == [1.0, -1.0]
    assert normalized["side_x_from_center_camera_corrected_px"].tolist() == [12.0, -12.0]
    right_angle, left_angle = normalized[
        "side_lift_lateral_angle_camera_corrected_deg"
    ].tolist()
    assert np.isclose(right_angle, np.degrees(np.arctan2(5.0, 12.0)))
    assert np.isclose(left_angle, np.degrees(np.arctan2(5.0, -12.0)))


def test_motor_control_comparisons_preserve_subject_and_pair_fingers() -> None:
    subject_summary = pd.DataFrame(
        [
            {"subject_id": "S1", "finger_condition": "index", "stiffness_value": 40, "n_trials": 3, "success_rate": 0.5, "mean_speed_px_s": 10.0, "mean_path_length_px": 100.0},
            {"subject_id": "S1", "finger_condition": "index", "stiffness_value": 85, "n_trials": 3, "success_rate": 1.0, "mean_speed_px_s": 20.0, "mean_path_length_px": 130.0},
            {"subject_id": "S1", "finger_condition": "middle", "stiffness_value": 40, "n_trials": 3, "success_rate": 0.5, "mean_speed_px_s": 12.0, "mean_path_length_px": 105.0},
            {"subject_id": "S1", "finger_condition": "middle", "stiffness_value": 85, "n_trials": 3, "success_rate": 1.0, "mean_speed_px_s": 18.0, "mean_path_length_px": 125.0},
            {"subject_id": "S2", "finger_condition": "index", "stiffness_value": 40, "n_trials": 3, "success_rate": 0.0, "mean_speed_px_s": 8.0, "mean_path_length_px": 90.0},
            {"subject_id": "S2", "finger_condition": "index", "stiffness_value": 85, "n_trials": 3, "success_rate": 0.5, "mean_speed_px_s": 12.0, "mean_path_length_px": 100.0},
            {"subject_id": "S2", "finger_condition": "middle", "stiffness_value": 40, "n_trials": 3, "success_rate": 0.0, "mean_speed_px_s": 9.0, "mean_path_length_px": 92.0},
            {"subject_id": "S2", "finger_condition": "middle", "stiffness_value": 85, "n_trials": 3, "success_rate": 0.5, "mean_speed_px_s": 15.0, "mean_path_length_px": 108.0},
        ]
    )

    results = ka.compute_motor_control_comparisons(subject_summary, metric_columns=["success_rate", "mean_speed_px_s", "mean_path_length_px"])

    within_subject = results["within_subject"]
    assert set(within_subject["finger_condition"]) == {"I", "M"}
    assert "mean_speed_px_s_within_subject_centered" in within_subject.columns
    assert "mean_speed_px_s_within_subject_finger_centered" in within_subject.columns

    slopes = results["within_finger_stiffness_effects"]
    s1_index_speed = slopes[
        (slopes["subject_id"] == "S1")
        & (slopes["finger_condition"] == "I")
        & (slopes["metric"] == "mean_speed_px_s")
    ].iloc[0]
    assert np.isclose(s1_index_speed["slope_per_stiffness_unit"], (20.0 - 10.0) / (85.0 - 40.0))
    assert np.isclose(s1_index_speed["high_minus_low_stiffness_delta"], 10.0)

    paired = results["finger_comparison_paired"]
    speed_pair = paired[(paired["comparison"] == "M - I") & (paired["metric"] == "mean_speed_px_s")].iloc[0]
    assert speed_pair["n_paired_observations"] == 2
    assert np.isclose(speed_pair["mean_difference"], 1.0)
    assert np.isfinite(speed_pair["sign_flip_p"])


def test_success_kinematic_z_analysis_uses_within_subject_finger_contrasts() -> None:
    trial_summary = pd.DataFrame(
        [
            {"subject_id": "S1", "subject_group": "E", "trial_index_raw": 1, "stiffness_segment_id": 1, "stiffness_value": 40, "finger_condition": "index", "correct_response": 1, "mean_speed_px_s": 20.0, "path_length_px": 100.0},
            {"subject_id": "S1", "subject_group": "E", "trial_index_raw": 2, "stiffness_segment_id": 1, "stiffness_value": 40, "finger_condition": "index", "correct_response": 0, "mean_speed_px_s": 10.0, "path_length_px": 120.0},
            {"subject_id": "S2", "subject_group": "E", "trial_index_raw": 1, "stiffness_segment_id": 1, "stiffness_value": 40, "finger_condition": "index", "correct_response": 1, "mean_speed_px_s": 18.0, "path_length_px": 90.0},
            {"subject_id": "S2", "subject_group": "E", "trial_index_raw": 2, "stiffness_segment_id": 1, "stiffness_value": 40, "finger_condition": "index", "correct_response": 0, "mean_speed_px_s": 12.0, "path_length_px": 110.0},
        ]
    )
    side_z = pd.DataFrame(
        [
            {"subject_id": "S1", "trial_index_raw": 1, "stiffness_segment_id": 1, "stiffness_value": 40, "mean_side_z_lift_px": 5.0},
            {"subject_id": "S1", "trial_index_raw": 2, "stiffness_segment_id": 1, "stiffness_value": 40, "mean_side_z_lift_px": 1.0},
            {"subject_id": "S2", "trial_index_raw": 1, "stiffness_segment_id": 1, "stiffness_value": 40, "mean_side_z_lift_px": 6.0},
            {"subject_id": "S2", "trial_index_raw": 2, "stiffness_segment_id": 1, "stiffness_value": 40, "mean_side_z_lift_px": 2.0},
        ]
    )

    results = ka.compute_success_kinematic_z_analysis(trial_summary, side_z, metric_columns=["mean_speed_px_s", "path_length_px", "mean_side_z_lift_px"])

    by_subject = results["success_contrast_by_subject_finger"]
    s1_speed = by_subject[(by_subject["subject_id"] == "S1") & (by_subject["metric"] == "mean_speed_px_s")].iloc[0]
    assert s1_speed["finger_condition"] == "I"
    assert np.isclose(s1_speed["success_minus_failure"], 10.0)

    summary = results["success_contrast_summary"]
    z_summary = summary[summary["metric"] == "mean_side_z_lift_px"].iloc[0]
    assert z_summary["n_paired_observations"] == 2
    assert np.isclose(z_summary["mean_difference"], 4.0)


def test_trajectory_similarity_analysis_quantifies_between_finger_paths() -> None:
    rows = []
    for subject in ["S1", "S2"]:
        for finger, offset in [("index", 0.0), ("middle", 3.0)]:
            for correct in [0, 1]:
                for t in [1, 2, 3]:
                    rows.append(
                        {
                            "subject_id": subject,
                            "finger_condition": finger,
                            "stiffness_value": 40,
                            "trajectory_time_bin": t,
                            "time_fraction": (t - 1) / 2,
                            "tracking_file": f"{subject}_{finger}_{correct}",
                            "correct_response": correct,
                            "x_centered_px": float(t + offset + correct),
                            "y_centered_px": float(2 * t + offset),
                            "r_center_px": float(t),
                            "speed_px_s": float(10 + offset + correct),
                        }
                    )
    time_bins = pd.DataFrame(rows)

    results = ka.compute_trajectory_similarity_analysis(time_bins)

    distances = results["finger_trajectory_distance_paired"]
    assert set(distances["comparison"]) == {"M - I"}
    assert distances["n_matched_time_bins"].min() == 3
    assert (distances["mean_xy_trajectory_distance_px"] > 0).all()

    success_failure = results["success_failure_trajectory_distance"]
    assert not success_failure.empty
    assert np.isclose(success_failure["success_failure_mean_xy_distance_px"].iloc[0], 1.0)


def test_subject_spatial_trajectory_analysis_keeps_subject_rows() -> None:
    rows = []
    for subject in ["S1", "S2"]:
        for finger, offset in [("index", 0.0), ("middle", 10.0)]:
            for stiffness in [40, 85]:
                for t in [1, 2, 3]:
                    rows.append(
                        {
                            "subject_id": subject,
                            "subject_group": "E",
                            "finger_condition": finger,
                            "stiffness_value": stiffness,
                            "trajectory_time_bin": t,
                            "time_fraction": (t - 1) / 2,
                            "tracking_file": f"{subject}_{finger}_{stiffness}",
                            "correct_response": 1,
                            "x_centered_px": float(t + offset),
                            "y_centered_px": float(2 * t + offset),
                            "r_center_px": float(t),
                            "speed_px_s": float(10 + offset),
                        }
                    )
    time_bins = pd.DataFrame(rows)

    results = ka.compute_subject_spatial_trajectory_analysis(time_bins)

    subject_xy = results["subject_xy_trajectory"]
    assert set(subject_xy["subject_id"]) == {"S1", "S2"}
    assert set(subject_xy["finger_condition"]) == {"I", "M"}

    summary = results["subject_spatial_trajectory_summary"]
    assert len(summary) == 8  # 2 subjects x 2 fingers x 2 stiffness values
    row = summary[
        (summary["subject_id"] == "S1")
        & (summary["finger_condition"] == "I")
        & (summary["stiffness_value"] == 40)
    ].iloc[0]
    assert np.isclose(row["mean_trajectory_path_length_px"], 2 * np.sqrt(5))
    assert np.isclose(row["mean_trajectory_net_displacement_px"], np.sqrt(20))
    assert np.isclose(row["mean_trajectory_straightness"], 1.0)

    distances = results["subject_finger_spatial_distance"]
    assert set(distances["subject_id"]) == {"S1", "S2"}
    assert set(distances["comparison"]) == {"M - I"}
    assert distances["n_matched_time_bins"].min() == 3


def test_subject_xy_figures_use_median_finger_name_and_route_to_movement(
    tmp_path: Path,
) -> None:
    rows = []
    for subject in ["L_E_1"]:
        for finger, offset in [("I", 0.0), ("M", 10.0)]:
            for stiffness in [40.0, 85.0]:
                for t in [1, 2, 3]:
                    rows.append(
                        {
                            "subject_id": subject,
                            "finger_condition": finger,
                            "stiffness_value": stiffness,
                            "trajectory_time_bin": t,
                            "x_centered_px": offset + t + stiffness / 100.0,
                            "y_centered_px": offset + 2 * t,
                        }
                    )

    paths = ka.save_subject_xy_trajectory_figures(
        tmp_path / "L_E_1",
        pd.DataFrame(rows),
        fig_dpi=20,
    )
    filenames = {p.name for p in paths}

    assert "median_xy_trajectory_by_finger_subject_L_E_1.png" in filenames
    assert "subject_L_E_1_xy_trajectories.png" not in filenames
    assert "all_xy_trajectories_with_finger_average.png" in filenames
    assert "all_xy_trajectories_with_stiffness_average.png" in filenames
    assert (
        ka.figure_category_parts("median_xy_trajectory_by_finger_subject_L_E_1.png")
        == ("trajectories", "movement_orientation")
    )


def test_subject_velocity_acceleration_analysis_keeps_subject_profiles() -> None:
    rows = []
    for subject in ["S1", "S2"]:
        for finger, offset in [("index", 0.0), ("middle", 5.0)]:
            for stiffness in [40, 85]:
                for t in [1, 2, 3]:
                    rows.append(
                        {
                            "subject_id": subject,
                            "subject_group": "E",
                            "finger_condition": finger,
                            "stiffness_value": stiffness,
                            "trajectory_time_bin": t,
                            "time_fraction": (t - 1) / 2,
                            "tracking_file": f"{subject}_{finger}_{stiffness}",
                            "correct_response": 1,
                            "vx_px_s": float(t + offset),
                            "vy_px_s": float(2 * t + offset),
                            "speed_px_s": float(10 + t + offset),
                            "ax_px_s2": float(3 * t + offset),
                            "ay_px_s2": float(4 * t + offset),
                            "acceleration_px_s2": float(20 + t + offset),
                            "jx_px_s3": float(5 * t + offset),
                            "jy_px_s3": float(6 * t + offset),
                            "jerk_px_s3": float(30 + t + offset),
                        }
                    )
    time_bins = pd.DataFrame(rows)

    results = ka.compute_subject_velocity_acceleration_analysis(time_bins)

    profile = results["subject_velocity_acceleration_profile"]
    assert set(profile["subject_id"]) == {"S1", "S2"}
    assert set(profile["finger_condition"]) == {"I", "M"}
    assert "velocity_heading_deg" in profile.columns
    assert "acceleration_heading_deg" in profile.columns
    assert "jerk_px_s3" in profile.columns

    summary = results["subject_velocity_acceleration_summary"]
    assert len(summary) == 8
    s1_index = summary[
        (summary["subject_id"] == "S1")
        & (summary["finger_condition"] == "I")
        & (summary["stiffness_value"] == 40)
    ].iloc[0]
    assert np.isclose(s1_index["mean_speed_px_s"], 12.0)
    assert np.isclose(s1_index["peak_abs_speed_px_s"], 13.0)
    assert np.isclose(s1_index["late_minus_early_speed_px_s"], 2.0)
    assert np.isclose(s1_index["mean_jerk_px_s3"], 32.0)

    distance = results["subject_finger_velocity_acceleration_distance"]
    assert set(distance["comparison"]) == {"M - I"}
    assert distance["n_matched_time_bins"].min() == 3
    assert (distance["mean_velocity_vector_distance_px_s"] > 0).all()


def test_velocity_profile_carries_abs_position_and_circular_direction() -> None:
    rows = []
    for t in [1, 2, 3]:
        rows.append(
            {
                "subject_id": "S1",
                "finger_condition": "index",
                "stiffness_value": 85,
                "trajectory_time_bin": t,
                "tracking_file": f"S1_index_85_{t}",
                "vx_px_s": 1.0,
                "vy_px_s": 2.0,
                # Negative centered positions exercise the absolute-distance
                # aggregation; the circular mean of a constant angle is itself.
                "x_centered_px": -3.0,
                "y_centered_px": 4.0,
                "movement_angle_deg": 10.0,
            }
        )
    profile = ka.compute_subject_velocity_acceleration_analysis(pd.DataFrame(rows))[
        "subject_velocity_acceleration_profile"
    ]
    assert {"abs_x_centered_px", "abs_y_centered_px", "movement_angle_deg"}.issubset(
        profile.columns
    )
    assert np.allclose(profile["abs_x_centered_px"], 3.0)
    assert np.allclose(profile["abs_y_centered_px"], 4.0)
    assert np.allclose(profile["movement_angle_deg"], 10.0)


def test_velocity_acceleration_profile_builds_3d_values_from_z_tracking() -> None:
    time_bins = pd.DataFrame(
        {
            "subject_id": ["S1", "S1", "S1"],
            "trial_index_raw": [1, 1, 1],
            "stiffness_segment_id": [1, 1, 1],
            "finger_condition": ["index", "index", "index"],
            "stiffness_value": [40.0, 40.0, 40.0],
            "trajectory_time_bin": [1, 2, 3],
            "time_fraction": [0.0, 0.5, 1.0],
            "tracking_file": ["trial_1", "trial_1", "trial_1"],
            "vx_px_s": [4.0, 4.0, 4.0],
            "vy_px_s": [0.0, 0.0, 0.0],
            "speed_px_s": [4.0, 4.0, 4.0],
            "ax_px_s2": [1.0, 1.0, 1.0],
            "ay_px_s2": [2.0, 2.0, 2.0],
            "acceleration_px_s2": [np.sqrt(5.0)] * 3,
        }
    )
    side_z_samples = pd.DataFrame(
        {
            "subject_id": ["S1", "S1", "S1"],
            "trial_index_raw": [1, 1, 1],
            "stiffness_segment_id": [1, 1, 1],
            "finger_condition": ["index", "index", "index"],
            "stiffness_value": [40.0, 40.0, 40.0],
            "side_time_fraction": [0.0, 0.5, 1.0],
            "side_time_s": [0.0, 1.0, 2.0],
            # Values represent the already-interpolated z_tracking.csv lift.
            # Vz = [NaN, 3, 6], Az = [NaN, NaN, 3].
            "side_z_lift_px": [0.0, 3.0, 9.0],
            "side_z_source": ["z_tracking.csv:z_active_finger_lift_px"] * 3,
        }
    )

    result = ka.compute_subject_velocity_acceleration_analysis(
        time_bins,
        side_z_samples=side_z_samples,
        n_time_bins=3,
    )

    profile = result["subject_velocity_acceleration_profile"].sort_values(
        "trajectory_time_bin"
    )
    assert {
        "vz_3d_proxy_px_s",
        "az_3d_proxy_px_s2",
        "speed_3d_proxy_px_s",
        "acceleration_3d_proxy_px_s2",
    }.issubset(profile.columns)
    second = profile.iloc[1]
    third = profile.iloc[2]
    assert np.isclose(second["vz_3d_proxy_px_s"], 3.0)
    assert np.isclose(second["speed_3d_proxy_px_s"], 5.0)
    assert np.isclose(third["vz_3d_proxy_px_s"], 6.0)
    assert np.isclose(third["az_3d_proxy_px_s2"], 3.0)
    assert np.isclose(third["acceleration_3d_proxy_px_s2"], np.sqrt(14.0))

    summary = result["subject_velocity_acceleration_summary"].iloc[0]
    assert "mean_speed_3d_proxy_px_s" in summary.index
    assert "mean_acceleration_3d_proxy_px_s2" in summary.index
    distribution = result["subject_velocity_acceleration_metric_distribution"]
    assert "mean_speed_3d_proxy_px_s" in set(distribution["metric"])


def test_velocity_suggestion_tables_decompose_speed_radial_and_tangential() -> None:
    rows = []
    for t, x, vx, vy in [
        (1, 1.0, 2.0, 0.0),
        (2, 2.0, 2.0, 1.0),
        (3, 3.0, 2.0, 2.0),
    ]:
        rows.append(
            {
                "subject_id": "L_E_1",
                "subject_group": "L_E",
                ka.EXPERIMENT_GROUP_COLUMN: "L_E",
                "finger_condition": "index",
                "stiffness_value": 85,
                "trajectory_time_bin": t,
                "time_fraction": (t - 1) / 2,
                "tracking_file": f"trial_{t}",
                "x_centered_px": x,
                "y_centered_px": 0.0,
                "z_lift_px": 0.0,
                "vx_px_s": vx,
                "vy_px_s": vy,
                "vz_3d_proxy_px_s": 0.0,
                "speed_px_s": float(np.hypot(vx, vy)),
                "radial_velocity_px_s": vx,
                "tangential_velocity_px_s": vy,
            }
        )

    results = ka.compute_subject_velocity_acceleration_analysis(pd.DataFrame(rows))

    speed = results["velocity_magnitude_normalized_time_median"]
    components = results["velocity_xyz_components_normalized_time_median"]
    radial = results["velocity_radial_normalized_time_median"]
    tangential = results["velocity_tangential_normalized_time_median"]
    decomposition = results["velocity_decomposition_profile"]
    curviness = results["velocity_curviness_summary"]
    notes = results["velocity_interpretation_notes"]

    assert {"speed_px_s", "speed_3d_proxy_px_s"}.issubset(set(speed["metric"]))
    assert {
        "velocity_x_axis_px_s",
        "velocity_y_axis_px_s",
        "velocity_z_axis_3d_proxy_px_s",
    }.issubset(set(components["metric"]))
    assert {"radial_velocity_px_s", "radial_velocity_3d_proxy_px_s"}.issubset(
        set(radial["metric"])
    )
    assert {
        "tangential_velocity_px_s",
        "abs_tangential_velocity_px_s",
        "tangential_speed_3d_proxy_px_s",
    }.issubset(set(tangential["metric"]))
    assert "tangential_radial_velocity_ratio_3d_proxy" in decomposition.columns
    assert curviness["median_tangential_radial_ratio"].notna().any()
    assert "magnitude_vs_velocity" in set(notes["topic"])
    assert "velocity_magnitude_normalized_time_mean" not in results
    assert "velocity_xyz_components_normalized_time_mean" not in results
    assert "velocity_radial_normalized_time_mean" not in results
    assert "velocity_tangential_normalized_time_mean" not in results
    assert "mean_tangential_radial_ratio" not in curviness.columns


def test_velocity_suggestion_figures_are_nested_under_velocity_families(
    tmp_path: Path,
) -> None:
    rows = []
    for finger in ["I", "M"]:
        for stiffness in [40.0, 85.0]:
            for t in [1, 2, 3]:
                rows.append(
                    {
                        "subject_id": "L_E_1",
                        "finger_condition": finger,
                        "stiffness_value": stiffness,
                        "trajectory_time_bin": t,
                        "time_fraction": (t - 1) / 2,
                        "x_centered_px": float(t),
                        "y_centered_px": 0.0,
                        "vx_px_s": float(t),
                        "vy_px_s": float(t + 1),
                        "vz_3d_proxy_px_s": float(t + 2),
                        "speed_px_s": float(t + 2),
                        "radial_velocity_px_s": float(t),
                        "tangential_velocity_px_s": float(t + 1),
                    }
                )
    paths = ka.save_velocity_suggestion_figures(
        tmp_path,
        pd.DataFrame(rows),
        include_subject_figures=True,
        include_aggregate_figures=True,
        fig_dpi=20,
    )
    filenames = {p.name for p in paths}

    assert "velocity_magnitude_normalized_time_mean.png" not in filenames
    assert not any(name.startswith("velocity_magnitude_normalized_time_mean") for name in filenames)
    assert "velocity_x_axis_normalized_time_median_subject_L_E_1.png" in filenames
    assert "velocity_y_axis_normalized_time_median_subject_L_E_1.png" in filenames
    assert "velocity_z_axis_normalized_time_median_subject_L_E_1.png" in filenames
    assert "velocity_xyz_axes_normalized_time_median_subject_L_E_1.png" in filenames
    assert "velocity_xyz_axes_normalized_time_median_subject_L_E_1_finger_I.png" in filenames
    assert "velocity_xyz_axes_normalized_time_median.png" in filenames
    assert "velocity_magnitude_normalized_time_median_subject_L_E_1.png" in filenames
    assert "velocity_tangential_normalized_time_median_subject_L_E_1.png" in filenames
    assert "velocity_tangential_normalized_time_median_subject_L_E_1_finger_I.png" in filenames
    assert "velocity_radial_normalized_time_median_finger_M.png" in filenames
    assert "velocity_curviness_radial_vs_tangential.png" in filenames
    assert not any("_mean" in name for name in filenames)
    assert (
        ka.figure_category_parts("velocity_x_axis_normalized_time_median.png")
        == ("velocity", "components")
    )
    assert (
        ka.figure_category_parts("velocity_y_axis_normalized_time_median.png")
        == ("velocity", "components")
    )
    assert (
        ka.figure_category_parts("velocity_z_axis_normalized_time_median.png")
        == ("velocity", "components")
    )
    assert (
        ka.figure_category_parts("velocity_xyz_axes_normalized_time_median.png")
        == ("velocity", "components")
    )
    assert (
        ka.figure_category_parts("velocity_magnitude_normalized_time_median.png")
        == ("velocity", "magnitude")
    )
    assert (
        ka.figure_category_parts("velocity_radial_normalized_time_median.png")
        == ("velocity", "radial")
    )
    assert (
        ka.figure_category_parts("velocity_tangential_normalized_time_median.png")
        == ("velocity", "tangential")
    )
    assert (
        ka.figure_category_parts("velocity_curviness_radial_vs_tangential.png")
        == ("velocity", "others")
    )
    assert (
        ka.figure_category_parts("standard_vs_comparison_Magnitude_subject_L_E_1.png")
        == ("velocity", "magnitude", "s_vs_c_v")
    )
    assert (
        ka.figure_category_parts("standard_vs_comparison_RadialX_subject_L_E_1.png")
        == ("velocity", "radial", "s_vs_c_v")
    )
    assert (
        ka.figure_category_parts("standard_vs_comparison_Vz_subject_L_E_1.png")
        == ("velocity", "components", "s_vs_c_v")
    )
    assert (
        ka.figure_category_parts("standard_vs_comparison_Vxyz_subject_L_E_1.png")
        == ("velocity", "components", "s_vs_c_v")
    )
    assert (
        ka.figure_category_parts("standard_vs_comparison_Axyz_subject_L_E_1.png")
        == ("acceleration", "s_vs_c_a")
    )
    assert (
        ka.figure_category_parts("standard_vs_comparison_TangentialZ_subject_L_E_1.png")
        == ("velocity", "tangential", "s_vs_c_v")
    )
    assert (
        ka.figure_category_parts("hand_orientation_xy_vectors_subject_L_E_1.png")
        == ("trajectories", "hand_orientation")
    )
    assert (
        ka.figure_category_parts(
            "movement_cycle_xy_direction_yz_hand_orientation_subject_L_E_1.png"
        )
        == ("trajectories", "movement_orientation")
    )
    assert (
        ka.figure_category_parts("all_xy_trajectories_with_finger_average.png")
        == ("trajectories", "movement_orientation")
    )
    assert (
        ka.figure_category_parts("side_z_lift_over_time_by_finger.png")
        == ("trajectories", "z_lift")
    )



def test_subject_velocity_acceleration_figures_are_split(tmp_path: Path) -> None:
    rows = []
    for finger in ["I", "M"]:
        for stiffness in [40.0, 85.0]:
            for t in [1, 2, 3, 4]:
                rows.append(
                    {
                        "subject_id": "L_E_1",
                        "finger_condition": finger,
                        "stiffness_value": stiffness,
                        "trajectory_time_bin": t,
                        "time_fraction": (t - 1) / 3,
                        "vx_px_s": float(t),
                        "vy_px_s": float(t + 1),
                        "speed_px_s": float(t + 2),
                        "ax_px_s2": float(t + 3),
                        "ay_px_s2": float(t + 4),
                        "acceleration_px_s2": float(t + 5),
                    }
                )
    profile = pd.DataFrame(rows)

    paths = ka.save_subject_velocity_acceleration_figures(
        tmp_path,
        profile,
        include_subject_figures=True,
        include_aggregate_figures=False,
        fig_dpi=20,
    )
    filenames = {p.name for p in paths}

    assert "subject_L_E_1_velocity.png" in filenames
    assert "subject_L_E_1_velocity_x_axis.png" in filenames
    assert "subject_L_E_1_velocity_y_axis.png" in filenames
    assert "subject_L_E_1_acceleration_by_stiffness.png" in filenames
    assert "subject_L_E_1_acceleration_x_axis_by_stiffness.png" in filenames
    assert "subject_L_E_1_acceleration_y_axis_by_stiffness.png" in filenames
    assert "subject_L_E_1_acceleration_by_stiffness_finger_I.png" in filenames
    assert "subject_L_E_1_acceleration_by_stiffness_finger_M.png" in filenames
    assert "subject_L_E_1_acceleration.png" not in filenames
    assert "subject_L_E_1_velocity_acceleration.png" not in filenames
    assert (
        ka.figure_category_parts("subject_L_E_1_acceleration_by_stiffness_finger_I.png")
        == ("acceleration", "median_by_finger")
    )
    assert (
        ka.figure_category_parts("subject_L_E_1_acceleration_x_axis_by_stiffness.png")
        == ("acceleration", "components")
    )


def test_average_acceleration_vs_stiffness_combines_finger_bars(tmp_path: Path) -> None:
    rows = []
    for subject in ["L_E_1", "L_E_2"]:
        for finger in ["I", "M", "R", "P"]:
            for stiffness in [40.0, 85.0]:
                for t in [1, 2, 3]:
                    rows.append(
                        {
                            "subject_id": subject,
                            "finger_condition": finger,
                            "stiffness_value": stiffness,
                            "trajectory_time_bin": t,
                            "time_fraction": (t - 1) / 2,
                            "vx_px_s": float(t),
                            "vy_px_s": float(t + 1),
                            "speed_px_s": float(t + 2),
                            "ax_px_s2": float(t + 3),
                            "ay_px_s2": float(t + 4),
                            "acceleration_px_s2": float(t + 5),
                        }
                    )
    paths = ka.save_subject_velocity_acceleration_figures(
        tmp_path,
        pd.DataFrame(rows),
        include_subject_figures=False,
        include_aggregate_figures=True,
        fig_dpi=20,
    )
    filenames = {p.name for p in paths}

    assert "average_acceleration_vs_stiffness.png" in filenames
    assert "all_acceleration_x_axis_vs_time_by_finger_median.png" in filenames
    assert "all_acceleration_y_axis_vs_time_by_finger_median.png" in filenames
    assert any(
        name.startswith("all_acceleration_x_axis_vs_time_by_finger_median_stiffness_40")
        for name in filenames
    )
    assert "average_acceleration_vs_stiffness_by_finger_subject.png" not in filenames


def test_time_acceleration_is_saved_as_subject_acceleration_by_finger(
    tmp_path: Path,
) -> None:
    group_time = pd.DataFrame(
        [
            {
                "finger_condition": finger,
                "trajectory_time_bin": t,
                "time_fraction": (t - 1) / 2,
                "mean_r_center_px": float(t),
                "mean_speed_px_s": float(t + 1),
                "mean_acceleration_px_s2": float(t + 2),
                "mean_radial_velocity_px_s": float(t + 3),
                "mean_tangential_velocity_px_s": float(t + 4),
                "mean_x_centered_px": float(t + 0.1),
                "mean_y_centered_px": float(t + 0.2),
                "mean_ax_px_s2": float(t + 5),
                "mean_ay_px_s2": float(t + 7),
                "mean_az_px_s2": float(t + 9),
            }
            for finger in ["I", "M"]
            for t in [1, 2, 3]
        ]
    )

    paths = ka.save_kinematic_figures(
        tmp_path / "L_E_1",
        pd.DataFrame(),
        group_time,
        pd.DataFrame(),
        pd.DataFrame(),
        fig_dpi=20,
    )
    filenames = {p.name for p in paths}

    assert "subject_L_E_1_acceleration_by_finger.png" in filenames
    assert "time_acceleration.png" not in filenames
    assert "time_distance_from_center.png" not in filenames
    assert "median_xy_trajectory_by_finger.png" not in filenames


def test_time_acceleration_by_finger_uses_side_z_samples_for_az(
    tmp_path: Path,
) -> None:
    group_time = pd.DataFrame(
        [
            {
                "subject_id": "L_E_1",
                "finger_condition": "I",
                "stiffness_value": 40.0,
                "trajectory_time_bin": t,
                "time_fraction": (t - 1) / 2,
                "mean_r_center_px": float(t),
                "mean_speed_px_s": float(t + 0.5),
                "mean_acceleration_px_s2": float(t + 1.5),
                "mean_radial_velocity_px_s": float(t + 2.5),
                "mean_tangential_velocity_px_s": float(t + 3.5),
                "mean_x_centered_px": float(t + 0.1),
                "mean_y_centered_px": float(t + 0.2),
                "mean_ax_px_s2": float(t),
                "mean_ay_px_s2": float(t + 1),
            }
            for t in [1, 2, 3]
        ]
    )
    side_z_samples = pd.DataFrame(
        {
            "subject_id": ["L_E_1", "L_E_1", "L_E_1"],
            "trial_index_raw": [1, 1, 1],
            "stiffness_segment_id": [1, 1, 1],
            "finger_condition": ["I", "I", "I"],
            "stiffness_value": [40.0, 40.0, 40.0],
            "side_time_fraction": [0.0, 0.5, 1.0],
            "side_time_s": [0.0, 1.0, 2.0],
            "side_z_lift_px": [0.0, 1.0, 4.0],
        }
    )

    paths = ka.save_kinematic_figures(
        tmp_path / "L_E_1",
        pd.DataFrame(),
        group_time,
        pd.DataFrame(),
        pd.DataFrame(),
        side_z_samples=side_z_samples,
        fig_dpi=20,
    )

    assert "subject_L_E_1_acceleration_by_finger.png" in {p.name for p in paths}


def test_explicit_subject_list_aggregate_scope_does_not_expand_to_full_group(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["L_E_1", "L_E_15", "L_E_19 (filter)"],
            "stiffness_value": [40, 85, 145],
        }
    )
    fig_dir = (
        tmp_path
        / "results"
        / "L_E_1_L_E_15_L_E_19"
        / "figures"
        / "standard_vs_comparison_velocity"
    )

    frames = ka._aggregate_group_frames_for_output(fig_dir, df)

    assert [name for name, _ in frames] == ["L_E_1_L_E_15_L_E_19"]
    assert frames[0][1]["subject_id"].tolist() == [
        "L_E_1",
        "L_E_15",
        "L_E_19 (filter)",
    ]
    assert not ka._owner_allowed_experiment_groups(fig_dir)


def test_named_combined_group_keeps_only_member_and_combined_aggregate_scopes(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["L_E_1", "N_E_2", "L_P_3"],
            "stiffness_value": [40, 85, 145],
        }
    )
    fig_dir = tmp_path / "results" / "L_N_E" / "figures" / "standard_vs_comparison"

    frames = ka._aggregate_group_frames_for_output(fig_dir, df)

    assert [name for name, _ in frames] == ["L_E", "N_E", "L_N_E"]
    assert frames[-1][1]["subject_id"].tolist() == ["L_E_1", "N_E_2"]


def test_organize_kinematic_results_tree_uses_nested_requested_folders(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    run_root = results_root / "L_E_1"
    source_dir = run_root / "figures" / "subject_velocity_acceleration"
    source_dir.mkdir(parents=True)
    (source_dir / "all_acceleration_xyz_vs_time_by_finger_mean.png").touch()
    (source_dir / "all_acceleration_xyz_vs_time_by_finger_median.png").touch()
    (source_dir / "all_acceleration_xyz_vs_time_by_finger_mean_stiffness_40.png").touch()
    (source_dir / "all_acceleration_xyz_vs_time_by_finger_median_stiffness_40.png").touch()
    (source_dir / "time_acceleration.png").touch()
    (source_dir / "subject_L_E_1_acceleration.png").touch()
    hand_dir = run_root / "figures" / "hand_orientation"
    hand_dir.mkdir(parents=True)
    (hand_dir / "hand_orientation_xy_vectors_subject_L_E_1.png").touch()
    movement_dir = run_root / "figures" / "movement_orientation"
    movement_dir.mkdir(parents=True)
    (
        movement_dir
        / "movement_cycle_xy_direction_yz_hand_orientation_subject_L_E_1.png"
    ).touch()
    (run_root / "figures" / "trajectories").mkdir(parents=True, exist_ok=True)
    (
        run_root
        / "figures"
        / "trajectories"
        / "all_xy_trajectories_with_finger_average.png"
    ).touch()
    (
        run_root
        / "figures"
        / "trajectories"
        / "median_xy_trajectory_by_finger.png"
    ).touch()
    (
        run_root
        / "figures"
        / "trajectories"
        / "subject_L_E_1_xy_trajectories.png"
    ).touch()
    (
        run_root
        / "figures"
        / "trajectories"
        / "time_distance_from_center.png"
    ).touch()
    (
        run_root
        / "figures"
        / "trajectories"
        / "radiation_orientation_all.png"
    ).touch()
    z_lift_dir = run_root / "figures" / "z_lift"
    z_lift_dir.mkdir(parents=True)
    (z_lift_dir / "side_z_lift_over_time_by_finger.png").touch()
    svc_dir = run_root / "figures" / "standard_vs_comparison_velocity"
    svc_dir.mkdir(parents=True)
    (svc_dir / "standard_vs_comparison_Ax_subject_L_E_1.png").touch()
    standard_dir = run_root / "standard" / "figures"
    standard_dir.mkdir(parents=True)
    (standard_dir / "time_speed.png").touch()
    (standard_dir / "all_velocity_vs_time_by_stiffness.png").touch()
    (standard_dir / "all_velocity_vs_time_by_stiffness_finger_I.png").touch()
    (standard_dir / "velocity_vs_time_by_stiffness.png").touch()
    (standard_dir / "velocity_vs_time_by_stiffness_finger_I.png").touch()
    (standard_dir / "time_radial_velocity.png").touch()
    (standard_dir / "time_tangential_velocity.png").touch()
    (standard_dir / "average_velocity_vs_stiffness.png").touch()
    (standard_dir / "average_velocity_vs_stiffness_by_finger_subject.png").touch()
    (standard_dir / "velocity_radial_normalized_time_mean.png").touch()
    (standard_dir / "velocity_radial_normalized_time_median.png").touch()
    (standard_dir / "all_acceleration_xyz_vs_time_by_finger_mean.png").touch()
    (standard_dir / "all_acceleration_xyz_vs_time_by_finger_median.png").touch()
    (standard_dir / "all_acceleration_xyz_vs_time_by_finger_mean_stiffness_40.png").touch()
    (standard_dir / "all_acceleration_xyz_vs_time_by_finger_median_stiffness_40.png").touch()
    (standard_dir / "average_acceleration_vs_stiffness.png").touch()
    (standard_dir / "time_acceleration.png").touch()
    comparison_dir = run_root / "Comparison" / "figures"
    comparison_dir.mkdir(parents=True)
    (comparison_dir / "time_radial_velocity.png").touch()
    (comparison_dir / "time_tangential_velocity.png").touch()
    (comparison_dir / "average_velocity_vs_stiffness.png").touch()
    (comparison_dir / "all_acceleration_xyz_vs_time_by_finger_mean_stiffness_40.png").touch()
    (comparison_dir / "all_acceleration_xyz_vs_time_by_finger_median_stiffness_40.png").touch()
    (comparison_dir / "average_acceleration_vs_stiffness.png").touch()
    (comparison_dir / "time_acceleration.png").touch()
    all_dir = run_root / "all" / "figures"
    all_dir.mkdir(parents=True)
    (all_dir / "time_speed.png").touch()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "kinematic_setup_balance.csv").write_text("a\n1\n", encoding="utf-8")
    (run_root / "velocity_magnitude_normalized_time_median.csv").write_text(
        "subject_id,stiffness_value,value\nL_E_1,85,1\n",
        encoding="utf-8",
    )
    (run_root / "velocity_xyz_components_normalized_time_median.csv").write_text(
        "subject_id,stiffness_value,metric,value\nL_E_1,85,velocity_x_axis_px_s,1\n",
        encoding="utf-8",
    )
    (run_root / "velocity_curviness_summary.csv").write_text(
        "subject_id,stiffness_value,value\nL_E_1,85,1\n",
        encoding="utf-8",
    )

    ka.organize_kinematic_results_tree(results_root, run_root, "L_E_1")

    assert (
        run_root
        / "figures"
        / "trajectories"
        / "hand_orientation"
        / "hand_orientation_xy_vectors_subject_L_E_1.png"
    ).exists()
    assert (
        run_root
        / "figures"
        / "trajectories"
        / "movement_orientation"
        / "movement_cycle_xy_direction_yz_hand_orientation_subject_L_E_1.png"
    ).exists()
    assert (
        run_root
        / "figures"
        / "trajectories"
        / "movement_orientation"
        / "all_xy_trajectories_with_finger_average.png"
    ).exists()
    assert not (
        run_root / "figures" / "trajectories" / "median_xy_trajectory_by_finger.png"
    ).exists()
    assert not (
        run_root / "figures" / "trajectories" / "subject_L_E_1_xy_trajectories.png"
    ).exists()
    assert not (
        run_root / "figures" / "trajectories" / "time_distance_from_center.png"
    ).exists()
    assert not (
        run_root / "figures" / "trajectories" / "movement_orientation" / "radiation_orientation_all.png"
    ).exists()
    assert (
        run_root
        / "figures"
        / "trajectories"
        / "z_lift"
        / "side_z_lift_over_time_by_finger.png"
    ).exists()
    assert not (run_root / "figures" / "hand_orientation").exists()
    assert not (run_root / "figures" / "movement_orientation").exists()
    assert not (run_root / "figures" / "z_lift").exists()
    assert (
        run_root
        / "figures"
        / "acceleration"
        / "median_by_stiffness"
        / "all_acceleration_xyz_vs_time_by_finger_median.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "acceleration"
        / "mean_by_stiffness"
        / "all_acceleration_xyz_vs_time_by_finger_mean.png"
    ).exists()
    assert (
        run_root
        / "figures"
        / "acceleration"
        / "median_by_stiffness"
        / "all_acceleration_xyz_vs_time_by_finger_median_stiffness_40.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "acceleration"
        / "mean_by_stiffness"
        / "all_acceleration_xyz_vs_time_by_finger_mean_stiffness_40.png"
    ).exists()
    assert (
        run_root
        / "figures"
        / "acceleration"
        / "s_vs_c_a"
        / "standard_vs_comparison_Ax_subject_L_E_1.png"
    ).exists()
    assert not (
        run_root / "figures" / "velocity" / "magnitude" / "standard_time_speed.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "magnitude"
        / "standard_all_velocity_vs_time_by_stiffness.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "magnitude"
        / "standard_all_velocity_vs_time_by_stiffness_finger_I.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "magnitude"
        / "standard_velocity_vs_time_by_stiffness.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "magnitude"
        / "standard_velocity_vs_time_by_stiffness_finger_I.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "radial"
        / "standard_time_radial_velocity.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "tangential"
        / "standard_time_tangential_velocity.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "radial"
        / "Comparison_time_radial_velocity.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "tangential"
        / "Comparison_time_tangential_velocity.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "magnitude"
        / "standard_average_velocity_vs_stiffness.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "magnitude"
        / "Comparison_average_velocity_vs_stiffness.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "velocity"
        / "radial"
        / "standard_velocity_radial_normalized_time_median.png"
    ).exists()
    assert (
        run_root
        / "csv"
        / "velocity"
        / "components"
        / "velocity_xyz_components_normalized_time_median.csv"
    ).exists()
    assert (
        run_root
        / "csv"
        / "velocity"
        / "magnitude"
        / "velocity_magnitude_normalized_time_median.csv"
    ).exists()
    assert (
        run_root
        / "csv"
        / "velocity"
        / "others"
        / "velocity_curviness_summary.csv"
    ).exists()
    assert (
        run_root
        / "figures"
        / "acceleration"
        / "median_by_stiffness"
        / "standard_all_acceleration_xyz_vs_time_by_finger_median.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "acceleration"
        / "median_by_stiffness"
        / "standard_all_acceleration_xyz_vs_time_by_finger_median_stiffness_40.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "acceleration"
        / "median_by_stiffness"
        / "Comparison_all_acceleration_xyz_vs_time_by_finger_median_stiffness_40.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "acceleration"
        / "median_by_stiffness"
        / "Comparison_all_acceleration_xyz_vs_time_by_finger_median_stiffness_40.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "acceleration"
        / "standard_average_acceleration_vs_stiffness.png"
    ).exists()
    assert not (
        run_root
        / "figures"
        / "acceleration"
        / "Comparison_average_acceleration_vs_stiffness.png"
    ).exists()
    assert not (
        run_root / "figures" / "acceleration" / "standard_time_acceleration.png"
    ).exists()
    assert not (
        run_root / "figures" / "acceleration" / "Comparison_time_acceleration.png"
    ).exists()
    assert not (run_root / "figures" / "acceleration" / "time_acceleration.png").exists()
    assert not (
        run_root / "figures" / "acceleration" / "subject_L_E_1_acceleration.png"
    ).exists()
    assert not (run_root / "all").exists()
    assert not (run_root / "_scopes").exists()
    assert (run_root / "csv" / "other" / "kinematic_setup_balance.csv").exists()


def test_standard_vs_comparison_velocity_adds_radial_tangential_3d_components(
    tmp_path: Path,
) -> None:
    rows = []
    for finger_idx, finger in enumerate(["I", "M"]):
        for stiffness, offset in [(40, 1.0), (85, 2.0)]:
            for t in [1, 2, 3]:
                rows.append(
                    {
                        "subject_id": "L_E_1",
                        "finger_condition": finger,
                        "stiffness_value": stiffness,
                        "standard_value": 85,
                        "trajectory_time_bin": t,
                        "x_centered_px": offset + t,
                        "y_centered_px": offset + finger_idx + 2 * t,
                        "z_lift_px": offset + 0.5 * t,
                        "vx_px_s": offset + 3 * t,
                        "vy_px_s": offset + finger_idx + t,
                        "vz_px_s": offset + 2 * t,
                        "speed_px_s": offset + 4 * t,
                        "ax_px_s2": offset + t,
                    }
                )

    paths = ka.save_standard_vs_comparison_velocity_figures(
        tmp_path,
        pd.DataFrame(rows),
        levels=("subject",),
        fig_dpi=20,
    )
    filenames = {p.name for p in paths}

    assert "standard_vs_comparison_Magnitude_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_RadialX_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_RadialY_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_RadialZ_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_TangentialX_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_TangentialY_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_TangentialZ_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_Ax_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_Vxyz_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_RadialXYZ_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_TangentialXYZ_subject_L_E_1.png" in filenames
    assert "standard_vs_comparison_Axyz_subject_L_E_1.png" in filenames
    assert (
        tmp_path / "standard_vs_comparison_velocity_figure_manifest.csv"
    ).exists()


def test_standard_vs_comparison_position_figures_are_grouped_by_subject(
    tmp_path: Path,
) -> None:
    rows = []
    for subject in ["L_E_1", "N_E_1"]:
        for finger in ["I", "M"]:
            for stiffness, mag in [(40, 5.0), (85, 9.0)]:  # 85 is the standard
                for t in [1, 2, 3]:
                    rows.append(
                        {
                            "subject_id": subject,
                            "finger_condition": finger,
                            "stiffness_value": stiffness,
                            "trajectory_time_bin": t,
                            "standard_value": 85,
                            "x_centered_px": (-1 if stiffness == 40 else 1) * (mag + t),
                            "y_centered_px": (-1 if finger == "I" else 1) * (mag + 2 * t),
                            "z_lift_px": (-1 if subject == "L_E_1" else 1) * (mag + 3 * t),
                            "movement_angle_deg": 10.0 * t,
                        }
                    )
    profile = pd.DataFrame(rows)

    paths = ka.save_standard_vs_comparison_position_figures(
        tmp_path, profile, levels=("subject", "group")
    )
    filenames = {p.name for p in paths}

    for subject in ["L_E_1", "N_E_1"]:
        assert f"standard_vs_comparison_PosY_subject_{subject}.png" in filenames
        assert f"standard_vs_comparison_PosX_subject_{subject}.png" in filenames
        assert f"standard_vs_comparison_PosZ_subject_{subject}.png" in filenames
        assert f"standard_vs_comparison_PosXYZ_subject_{subject}.png" in filenames
        assert (
            f"standard_vs_comparison_MovementDirection_subject_{subject}.png"
            in filenames
        )
    assert any(name.endswith("_group_L_E.png") for name in filenames)
    assert (
        tmp_path / "standard_vs_comparison_position_figure_manifest.csv"
    ).exists()


def test_metric_distribution_reports_median_ci_and_log_backtransform() -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["S1", "S1", "S1", "S1"],
            "finger_condition": ["I", "I", "I", "I"],
            "positive_skew_metric": [1.0, 2.0, 10.0, 100.0],
        }
    )

    summary = ka.summarize_metric_distribution(df, ["subject_id", "finger_condition"], ["positive_skew_metric"])

    row = summary.iloc[0]
    assert row["n"] == 4
    assert np.isclose(row["median"], 6.0)
    assert np.isfinite(row["median_ci95_low"])
    assert bool(row["log_transform_valid"])
    assert bool(row["log_transform_recommended"])
    assert np.isclose(row["geometric_mean_backtransformed"], np.exp(np.log([1.0, 2.0, 10.0, 100.0]).mean()))



def test_normalized_jerk_cost_has_expected_dimensionless_value() -> None:
    movement = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "jx_px_s3": [1.0, 1.0, 1.0],
            "jy_px_s3": [0.0, 0.0, 0.0],
        }
    )

    assert np.isclose(ka._normalized_jerk_cost(movement, path_length_px=2.0), 16.0)


def test_normalized_jerk_cost_supports_numpy_without_trapezoid(monkeypatch) -> None:
    monkeypatch.delattr(np, "trapezoid", raising=False)
    movement = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "jx_px_s3": [1.0, 1.0, 1.0],
            "jy_px_s3": [0.0, 0.0, 0.0],
        }
    )

    assert np.isclose(ka._normalized_jerk_cost(movement, path_length_px=2.0), 16.0)


def test_3d_proxy_kinematics_interpolates_z_and_computes_features() -> None:
    samples = pd.DataFrame(
        {
            "subject_id": ["S1", "S1", "S1"],
            "subject_group": ["E", "E", "E"],
            "trial_index_raw": [1, 1, 1],
            "pair_number": [1, 1, 1],
            "finger_condition": ["index", "index", "index"],
            "stiffness_value": [40, 40, 40],
            "stiffness_segment_id": [2, 2, 2],
            "time_s": [0.0, 1.0, 2.0],
            "stiffness_time_fraction": [0.0, 0.5, 1.0],
            "x_centered_px": [0.0, 3.0, 6.0],
            "y_centered_px": [0.0, 4.0, 8.0],
            "correct_response": [1, 1, 1],
            "movement_direction": ["E", "E", "E"],
        }
    )
    side = pd.DataFrame(
        {
            "subject_id": ["S1", "S1"],
            "trial_index_raw": [1, 1],
            "stiffness_segment_id": [2, 2],
            "side_time_fraction": [0.0, 1.0],
            "side_z_lift_px": [0.0, 12.0],
            "side_camera_side": ["left", "left"],
            "side_x_from_frame_center_px": [10.0, 20.0],
        }
    )

    results = ka.compute_3d_proxy_kinematics(samples, side)

    sample_3d = results["kinematic_3d_proxy_samples"]
    assert np.isclose(sample_3d["z_3d_proxy_px"].iloc[1], 6.0)
    assert np.isclose(sample_3d["side_lateral_camera_corrected_px"].iloc[1], -15.0)
    trial = results["trial_3d_kinematic_summary"].iloc[0]
    assert trial["finger_condition"] == "I"
    assert np.isclose(trial["path_length_3d_proxy_px"], 2 * np.sqrt(61))
    assert np.isclose(
        trial["path_length_side_view_camera_corrected_px"],
        2 * np.sqrt(5**2 + 6**2),
    )
    assert np.isclose(trial["max_excursion_3d_from_start_px"], np.sqrt(244))
    assert np.isclose(trial["peak_velocity_3d_proxy_px_s"], np.sqrt(61))
    assert not results["subject_3d_metric_distribution"].empty


def test_hand_orientation_plane_analysis_uses_side_z_proxy() -> None:
    trial_summary = pd.DataFrame(
        {
            "subject_id": ["N_E_1"],
            "trial_index_raw": [1],
            "stiffness_segment_id": [1],
            "experiment_group": ["N_E"],
            "subject_group": ["N"],
            "finger_condition": ["I"],
            "stiffness_value": [40.0],
            "correct_response": [1],
            "mean_hand_orientation_xy_deg": [0.0],
            "mean_thumb_active_span_px": [10.0],
        }
    )
    side_summary = pd.DataFrame(
        {
            "subject_id": ["N_E_1"],
            "trial_index_raw": [1],
            "stiffness_segment_id": [1],
            "mean_side_z_lift_px": [10.0],
            "side_detection_rate": [1.0],
        }
    )

    result = ka.compute_hand_orientation_plane_analysis(trial_summary, side_summary)
    trials = result["hand_orientation_plane_trials"]
    summary = result["hand_orientation_plane_summary"]

    assert {
        "hand_orientation_xy_deg",
        "hand_orientation_yz_deg",
        "hand_orientation_zx_deg",
    }.issubset(trials.columns)
    assert np.isclose(trials.loc[0, "hand_orientation_yz_deg"], 90.0)
    assert np.isclose(trials.loc[0, "hand_orientation_zx_deg"], 45.0)
    assert {"XY", "YZ", "ZX"}.issubset(set(summary["plane"]))


def test_hand_orientation_plane_figures_make_subject_and_group_finger_matrices(
    tmp_path: Path,
) -> None:
    summary = pd.DataFrame(
        {
            "scope": ["all"],
            "group": ["all"],
            "plane": ["XY"],
            "metric": ["hand_orientation_xy_deg"],
            "n": [1],
            "circular_mean_deg": [0.0],
            "median_deg": [0.0],
            "resultant_length": [1.0],
        }
    )
    trials = pd.DataFrame(
        {
            "subject_id": ["L_E_1", "L_E_1", "L_E_1", "L_E_1", "L_P_1", "N_P_1"],
            "experiment_group": ["L_E", "L_E", "L_E", "L_E", "L_P", np.nan],
            "finger_condition": ["index", "middle", "index", "middle", "ring", "pinky"],
            "stiffness_value": [40.0, 40.0, 40.0, 40.0, 85.0, 55.0],
            "hand_orientation_xy_deg": [180.0, 180.0, 180.0, 180.0, 180.0, -90.0],
            "hand_orientation_dx_px": [10.0, 0.0, 10.0, 0.0, -10.0, 0.0],
            "hand_orientation_dy_px": [0.0, 10.0, 0.0, 10.0, 0.0, -10.0],
            "mean_x_centered_px": [999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
            "mean_y_centered_px": [-999.0, -999.0, -999.0, -999.0, -999.0, -999.0],
            "mean_r_center_px": [10.0, 20.0, 0.001, 1000.0, 30.0, 40.0],
            "mean_side_z_lift_px": [1.0, 2.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    paths = ka.save_hand_orientation_plane_figures(tmp_path, summary, trials, fig_dpi=30)
    plot_trials, vector_source = ka._prepare_hand_orientation_xy_vector_trials(trials)

    filenames = {path.name for path in paths}
    assert "hand_orientation_xy_vectors_subject_L_E_1.png" in filenames
    assert "hand_orientation_xy_vectors_subject_L_P_1.png" in filenames
    assert "hand_orientation_xy_vectors_subject_N_P_1.png" in filenames
    assert "hand_orientation_xy_vectors_group_L_E.png" in filenames
    assert "hand_orientation_xy_vectors_group_L_P.png" in filenames
    assert "hand_orientation_xy_vectors_group_N_P.png" in filenames
    # Subject figures route to the subject sibling folder under figures/.
    assert (
        tmp_path.parent
        / "L_E_1"
        / "figures"
        / "hand_orientation_xy_vectors_subject_L_E_1.png"
    ).exists()
    assert vector_source == "hand_orientation_dx_px/hand_orientation_dy_px"
    east = plot_trials[plot_trials["finger_condition"] == "I"].iloc[0]
    north = plot_trials[plot_trials["finger_condition"] == "M"].iloc[0]
    assert np.isclose(east["_vector_dx"], 10.0)
    assert np.isclose(east["_vector_dy"], 0.0)
    assert np.isclose(north["_vector_dx"], 0.0)
    assert np.isclose(north["_vector_dy"], 10.0)
    assert not np.isclose(east["_vector_dx"], east["mean_x_centered_px"])
    assert "L_P_1" in set(plot_trials["subject_id"].astype(str))
    assert plot_trials["_orientation_span_px"].min() > 0
    colors = ka._stiffness_viridis_colors([25.0, 40.0, 55.0, 145.0])
    assert list(colors) == [25.0, 40.0, 55.0, 145.0]
    assert colors[25.0] != colors[40.0] != colors[55.0] != colors[145.0]


def test_movement_cycle_hand_angle_figures_are_grouped_by_subject_and_group(
    tmp_path: Path,
) -> None:
    rows = []
    for subject, subject_group, experiment_group in [
        ("L_E_1", "L", "L_E"),
        ("L_P_1", "L", "L_P"),
        ("N_E_1", "N", "N_E"),
    ]:
        for stiffness in [40.0, 85.0]:
            for trial in [1, 2]:
                for time_bin in [1, 2, 3]:
                    rows.append(
                        {
                            "subject_id": subject,
                            "subject_group": subject_group,
                            "experiment_group": experiment_group,
                            "stiffness_value": stiffness,
                            "trajectory_time_bin": time_bin,
                            "stiffness_time_fraction": (time_bin - 1) / 2,
                            "stiffness_start_fraction": 0.0,
                            "stiffness_end_fraction": 0.5,
                            "time_to_answer_s": 4.0,
                            "tracking_file": f"{subject}_{trial}.csv",
                            "trial_index_raw": trial,
                            "stiffness_segment_id": 1,
                            "finger_condition": "index",
                            "movement_angle_deg": -170 + time_bin * 20 + trial,
                            "hand_orientation_xy_deg": 170 - time_bin * 15 - trial,
                            "hand_orientation_yz_deg": 80 - time_bin * 10 - trial,
                        }
                    )
    paths = ka.save_movement_cycle_hand_angle_figures(
        tmp_path, pd.DataFrame(rows), fig_dpi=30
    )

    filenames = {path.name for path in paths}
    assert (
        "movement_cycle_xy_direction_yz_hand_orientation_subject_L_E_1.png"
        in filenames
    )
    assert (
        "movement_cycle_xy_direction_yz_hand_orientation_subject_N_E_1.png"
        in filenames
    )
    # Coarse subject_group (L/N) figures are no longer produced; the reported
    # groups are the full experiment groups (pilots excluded -> only L_E, N_E).
    assert (
        "movement_cycle_xy_direction_yz_hand_orientation_experiment_group_L_E.png"
        in filenames
    )
    assert (
        "movement_cycle_xy_direction_yz_hand_orientation_experiment_group_N_E.png"
        in filenames
    )
    assert not any("_group_L.png" in filename for filename in filenames)
    assert not any("_group_N.png" in filename for filename in filenames)
    assert not any("_P_" in filename for filename in filenames)
    # Subject figures route to the subject sibling folder under figures/.
    assert (
        tmp_path.parent
        / "L_E_1"
        / "figures"
        / "movement_cycle_xy_direction_yz_hand_orientation_subject_L_E_1.png"
    ).exists()
    # Group-aggregate figures stage under the selection figures tree.
    assert (
        tmp_path
        / "figures"
        / "L_E"
        / "movement_cycle_xy_direction_yz_hand_orientation_experiment_group_L_E.png"
    ).exists()
    assert (tmp_path / "movement_cycle_hand_angle_figure_manifest.csv").exists()


def test_figure_routing_helpers_use_group_and_all_layout(tmp_path: Path) -> None:
    fig_dir = ka._figures_base(tmp_path)  # <tmp>/figures

    # Per-subject -> sibling subject folder under figures/.
    assert ka.subject_figure_path(fig_dir, "L_E_3", "x.png") == (
        tmp_path.parent / "L_E_3" / "figures" / "x.png"
    )
    # Group-aggregate -> staged under the selection figures tree.
    assert ka.group_figure_path(fig_dir, "N_E", "y.png") == (
        tmp_path / "figures" / "N_E" / "y.png"
    )
    # Combined groups keep the "+" and route to their own folder.
    assert ka.group_figure_path(fig_dir, "N+L_E", "y.png") == (
        tmp_path / "figures" / "N+L_E" / "y.png"
    )
    # General -> staged under figures/all/.
    assert ka.general_figure_path(fig_dir, "z.png") == tmp_path / "figures" / "all" / "z.png"

    # Scope routing for the polar/radiation families.
    assert ka._scope_output_dir(fig_dir, "all") == tmp_path / "figures" / "all"
    assert ka._scope_output_dir(fig_dir, "finger_I") == tmp_path / "figures" / "all"
    assert ka._scope_output_dir(fig_dir, "stiffness_85.0") == tmp_path / "figures" / "all"
    assert ka._scope_output_dir(fig_dir, "experiment_group_L_E") == tmp_path / "figures" / "L_E"
    assert ka._scope_output_dir(fig_dir, "experiment_group_N+L_E") == tmp_path / "figures" / "L_N_E"
    # Redundant / coarse scopes are dropped (return None -> figure not produced).
    assert ka._scope_output_dir(fig_dir, "subject_group_L") is None
    assert ka._scope_output_dir(fig_dir, "workspace_N") is None
    assert ka._scope_output_dir(fig_dir, "success_success") is None
    assert ka._scope_output_dir(fig_dir, "success_failure") is None
    assert ka._scope_output_dir(fig_dir, "group_L") is None


def _write_rich_tracking(path: Path) -> None:
    """A longer two-stiffness trajectory so velocity/acceleration figures render."""
    rows = []
    t = 0
    for stiffness, x0 in [(40, 320), (85, 330)]:
        for i in range(8):
            interacting = i != 0  # first sample of each segment is the baseline
            rows.append(
                (
                    f"2026-01-01T00:00:{t:02d}",
                    interacting,
                    stiffness,
                    x0 + i,
                    240 + (i % 3),
                    "index",
                )
            )
            t += 1
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "interacting", "stiffness", "object_x", "object_y", "finger"]
        )
        writer.writerows(rows)


def _two_subject_tracking_kin(tmp_path: Path) -> dict:
    """Run compute_tracking_kinematics for two subjects; return the full kin dict."""
    rows = []
    for subject in ["N_E_1", "N_E_2"]:
        pair = tmp_path / subject / "pair_001"
        pair.mkdir(parents=True)
        tracking = pair / "tracking.csv"
        _write_rich_tracking(tracking)
        rows.append(
            {
                "subject_id": subject,
                "subject_group": "N_E",
                "selected": True,
                "trial_index_raw": 1,
                "pair_number": 1,
                "pair_dir": str(pair),
                "tracking_file": str(tracking),
                "side_video_file": str(pair / "side_camera.mp4"),
                "top_video_file": str(pair / "top_camera.mp4"),
                "tracking_exists": True,
                "side_video_exists": False,
                "comparison_value": 40,
                "standard_value": 85,
                "signed_stiffness_delta": -45,
                "correct_response": 1,
            }
        )
    return ka.compute_tracking_kinematics(
        pd.DataFrame(rows), center_x=320, center_y=240, n_time_bins=5
    )


def test_save_individual_subject_reports_fills_each_group_member(tmp_path: Path) -> None:
    """In a group run, the AGGREGATE figures a solo run writes into results/<subject>/
    -- including the velocity_vs_time / average_velocity_vs_stiffness panels and the
    time_magnitude suite -- must be regenerated for every individual, so each member
    folder matches a standalone single-subject run."""
    kin = _two_subject_tracking_kin(tmp_path)
    results_root = tmp_path / "results"

    reports = ka.save_individual_subject_reports(
        results_root,
        trial_kinematic_summary=kin["trial_summary"],
        trajectory_time_bins=kin["time_bins"],
        kinematic_samples=kin["samples"],
        n_time_bins=5,
    )

    assert set(reports) == {"N_E_1", "N_E_2"}
    for subject in ["N_E_1", "N_E_2"]:
        assert reports[subject], f"no figures generated for {subject}"
        produced = {p.name for p in (results_root / subject / "figures").rglob("*.png")}
        # Aggregate figures that a group run otherwise only writes at the group level.
        # These are exactly the families the user reported missing per individual.
        for expected in [
            "time_magnitude.png",
            "velocity_vs_time_by_stiffness.png",
            "average_velocity_vs_stiffness.png",
        ]:
            assert expected in produced, f"{expected} missing for {subject}: {sorted(produced)}"
        # Every returned path lives inside this subject's own folder.
        for path in reports[subject]:
            assert (results_root / subject) in path.parents


def test_save_individual_subject_reports_noop_for_single_subject(tmp_path: Path) -> None:
    """A single-subject selection already wrote the suite into its own folder, so the
    per-individual filler must do nothing."""
    kin = _two_subject_tracking_kin(tmp_path)
    summary = kin["trial_summary"]
    one = summary[summary["subject_id"] == "N_E_1"].copy()
    one_bins = kin["time_bins"][kin["time_bins"]["subject_id"] == "N_E_1"].copy()
    results_root = tmp_path / "results"

    reports = ka.save_individual_subject_reports(
        results_root,
        trial_kinematic_summary=one,
        trajectory_time_bins=one_bins,
        kinematic_samples=kin["samples"][kin["samples"]["subject_id"] == "N_E_1"].copy(),
        n_time_bins=5,
    )

    assert reports == {}
    assert not results_root.exists() or not any(results_root.rglob("*.png"))
