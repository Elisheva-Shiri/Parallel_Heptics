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
