import builtins
import io

import pytest
import typer

import backend


@pytest.fixture(autouse=True)
def _restore_debug():
    yield
    backend.set_is_debug(True)


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
