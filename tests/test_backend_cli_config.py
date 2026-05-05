import builtins
import io
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

import backend


@pytest.fixture(autouse=True)
def _restore_debug():
    yield
    backend.set_is_debug(True)

    def test_main_passes_movement_strategy_to_backend_experiment(self):
        fake_experiment = Mock()
        config = backend.Configuration(pairs=[])

        with (
            patch.object(backend.Configuration, "read_configuration", return_value=config),
            patch.object(backend, "create_experiment_folder", return_value=Path("run")),
            patch.object(backend, "start_experiment", return_value=fake_experiment) as start_experiment,
            patch.object(backend, "sleep", side_effect=KeyboardInterrupt),
        ):
            backend.main(
                run_mode=backend.RunMode.SINGLE_FINGER,
                pair_finger=backend.FingerChoice.INDEX,
                vision_type=backend.VisionType.MEDIAPIPE,
                finger_color=None,
                side_camera_enabled=False,
                play_white_noise=False,
                mirror_hand=False,
                movement_strategy=backend.MovementStrategy.IK,
                target_cycle_count=1,
                is_debug=False,
                continue_last_experiment=False,
            )

        fake_experiment.cleanup.assert_called_once_with()
        self.assertEqual(start_experiment.call_args.args[5], backend.MovementStrategy.IK)


def test_debug_print_gate_controls_global_print():
    output = io.StringIO()

    backend.set_is_debug(False)
    builtins.print("hidden", file=output)
    assert output.getvalue() == ""

    backend.set_is_debug(True)
    builtins.print("visible", file=output)
    assert output.getvalue() == "visible\n"


def test_parse_repeated_finger_color_flags():
    assert backend._parse_finger_color_flags(["thumb=red", "index=blue"]) == {
        "thumb": "red",
        "index": "blue",
    }


def test_rejects_duplicate_finger_color_values():
    with pytest.raises(typer.BadParameter):
        backend._parse_finger_color_flags(["thumb=red", "index=red"])


def test_resolve_motor_set_index():
    assert backend._resolve_motor_set_id(3) == backend.MotorSetId.MOTORS_9_11


def test_rejects_invalid_motor_set_index():
    with pytest.raises(typer.BadParameter):
        backend._resolve_motor_set_id(5)


def test_get_protocol_files_ignores_nested_folder(tmp_path):
    protocols_folder = tmp_path / "protocols"
    nested_folder = protocols_folder / "detailed"
    nested_folder.mkdir(parents=True)
    top_level_protocol = protocols_folder / "participant_1.csv"
    top_level_protocol.write_text("1,2,3,4\n")
    (protocols_folder / "notes.txt").write_text("not a protocol")
    (nested_folder / "nested_protocol.csv").write_text("5,6,7,8\n")

    assert backend._get_protocol_files(protocols_folder) == [top_level_protocol]


def test_resolve_protocol_path_accepts_stem_name_and_number(tmp_path):
    protocols_folder = tmp_path / "protocols"
    protocols_folder.mkdir()
    first_protocol = protocols_folder / "participant_1.csv"
    second_protocol = protocols_folder / "participant_2.csv"
    first_protocol.write_text("1,2,3,4\n")
    second_protocol.write_text("5,6,7,8\n")

    assert backend._resolve_protocol_path("participant_1", protocols_folder) == first_protocol
    assert backend._resolve_protocol_path("participant_2.csv", protocols_folder) == second_protocol
    assert backend._resolve_protocol_path("2", protocols_folder) == second_protocol


def test_main_loads_selected_protocol_from_protocols_folder():
    fake_experiment = Mock()
    config = backend.Configuration(pairs=[])

    with (
        patch.object(backend.Configuration, "read_configuration", return_value=config) as read_configuration,
        patch.object(backend, "create_experiment_folder", return_value=Path("run")),
        patch.object(backend, "start_experiment", return_value=fake_experiment),
        patch.object(backend, "sleep", side_effect=KeyboardInterrupt),
    ):
        backend.main(
            run_mode=backend.RunMode.COMPARISON,
            motor_set=0,
            vision_type=backend.VisionType.MEDIAPIPE,
            protocol="participant_2",
            finger_color=None,
            side_camera_enabled=False,
            play_white_noise=False,
            mirror_hand=False,
            movement_strategy=backend.MovementStrategy.IK,
            target_cycle_count=1,
            visual_cue_mode=backend.VisualCueMode.CIRCLE_BORDER,
            is_debug=False,
            continue_last_experiment=False,
        )

    read_configuration.assert_called_once_with(
        str(backend.PROTOCOLS_FOLDER / "participant_2.csv")
    )
    fake_experiment.cleanup.assert_called_once_with()


def test_format_replay_command_includes_all_prompt_avoiding_flags_for_new_experiment():
    command = backend._format_replay_command(
        run_mode="comparison",
        motor_set_id=backend.MotorSetId.MOTORS_3_5,
        vision_type=backend.VisionType.OPTICAL_COLOR_NO_GPU,
        protocol_path=Path("participant 2.csv"),
        finger_colors={"thumb": "red", "index": "blue"},
        side_camera_off=True,
        play_white_noise=False,
        hand_orientation=backend.HandOrientation.MIRRORED,
        movement_strategy=backend.MovementStrategy.IK,
        target_cycle_count=2,
        visual_cue_mode=backend.VisualCueMode.CIRCLE_FILLED,
        is_debug=False,
        continue_last_experiment=False,
        resume_pair_number=None,
    )

    assert command == (
        'uv run python backend.py --run-mode comparison --motor-set 1 '
        '--vision-type optical_color_no_gpu --new-experiment --protocol "participant 2.csv" '
        '--finger-color index=blue --finger-color thumb=red --side-camera-disabled '
        '--no-white-noise --mirror-hand --movement-strategy ik --target-cycle-count 2 '
        '--visual-cue-mode circle_filled --no-debug'
    )


def test_format_replay_command_includes_resume_pair_for_continue():
    command = backend._format_replay_command(
        run_mode="single_finger",
        motor_set_id=backend.MotorSetId.MOTORS_0_2,
        vision_type=backend.VisionType.MEDIAPIPE,
        protocol_path=None,
        finger_colors={},
        side_camera_off=False,
        play_white_noise=True,
        hand_orientation=backend.HandOrientation.NOT_MIRRORED,
        movement_strategy=backend.MovementStrategy.FREE_FORM,
        target_cycle_count=1,
        visual_cue_mode=backend.VisualCueMode.RUBBER_BAND,
        is_debug=True,
        continue_last_experiment=True,
        resume_pair_number=7,
    )

    assert command == (
        "uv run python backend.py --run-mode single_finger --motor-set 0 "
        "--vision-type mediapipe --continue-last-experiment --from-pair 7 "
        "--side-camera-enabled --white-noise --no-mirror-hand --movement-strategy free_form "
        "--target-cycle-count 1 --visual-cue-mode rubber_band --debug"
    )


def test_main_prompts_for_omitted_relevant_initialization_flags_and_skips_irrelevant_colors():
    fake_experiment = Mock()
    protocol_path = backend.PROTOCOLS_FOLDER / "participant_2.csv"
    config = backend.Configuration(pairs=[])

    with (
        patch.object(backend, "get_is_debug", return_value=False) as get_is_debug,
        patch.object(backend, "get_continue_last_experiment", return_value=False) as get_continue_last,
        patch.object(backend, "get_protocol_path", return_value=protocol_path) as get_protocol_path,
        patch.object(backend.Configuration, "read_configuration", return_value=config),
        patch.object(backend, "get_run_mode", return_value="comparison") as get_run_mode,
        patch.object(backend, "get_motor_set_id", return_value=backend.MotorSetId.MOTORS_0_2) as get_motor_set_id,
        patch.object(backend, "get_vision_type", return_value=backend.VisionType.MEDIAPIPE) as get_vision_type,
        patch.object(backend, "get_finger_colors", side_effect=AssertionError("finger colors should be irrelevant for mediapipe")),
        patch.object(backend, "get_side_camera_off", return_value=True) as get_side_camera_off,
        patch.object(backend, "get_play_white_noise", return_value=False) as get_play_white_noise,
        patch.object(backend, "get_hand_orientation", return_value=backend.HandOrientation.NOT_MIRRORED) as get_hand_orientation,
        patch.object(backend, "get_movement_strategy", return_value=backend.MovementStrategy.IK) as get_movement_strategy,
        patch.object(backend, "get_target_cycle_count", return_value=1) as get_target_cycle_count,
        patch.object(backend, "get_visual_cue_mode", return_value=backend.VisualCueMode.CIRCLE_BORDER) as get_visual_cue_mode,
        patch.object(backend, "create_experiment_folder", return_value=Path("run")),
        patch.object(backend, "start_experiment", return_value=fake_experiment),
        patch.object(backend, "sleep", side_effect=KeyboardInterrupt),
    ):
        backend.main()

    for prompt in (
        get_is_debug,
        get_continue_last,
        get_protocol_path,
        get_run_mode,
        get_motor_set_id,
        get_vision_type,
        get_side_camera_off,
        get_play_white_noise,
        get_hand_orientation,
        get_movement_strategy,
        get_target_cycle_count,
        get_visual_cue_mode,
    ):
        prompt.assert_called_once()
    fake_experiment.cleanup.assert_called_once_with()


def test_interactive_prompt_allows_enter_to_accept_prompt_default(monkeypatch):
    responses = iter([""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    assert backend.get_run_mode() == "comparison"


def test_fingers_needed_come_from_configuration():
    config = backend.Configuration(
        pairs=[
            backend.StiffnessPair(
                first=backend.StiffnessValue(value=85, finger_id=0),
                second=backend.StiffnessValue(value=85, finger_id=2),
            )
        ]
    )

    assert backend._get_fingers_needed(config, "comparison") == ["thumb", "index", "ring"]
    assert backend._get_fingers_needed(config, "single_finger") == ["single"]
