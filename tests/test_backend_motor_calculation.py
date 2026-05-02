import math

import pytest

from motor_controller import HandOrientation, MotorController, MotorSetId, MovementStrategy


def _to_tuples(movements):
    return [(m.index, m.pos) for m in movements]


def _make_controller(strategy: MovementStrategy, **overrides) -> MotorController:
    kwargs = {
        "movement_strategy": strategy,
        "top_width": 1000.0,
        "top_height": 1000.0,
        "edge_threshold": 30.0,
    }
    kwargs.update(overrides)
    return MotorController(**kwargs)


def test_cardinal_strategy_uses_major_axis():
    controller = _make_controller(MovementStrategy.CARDINAL)

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=-40.0,
        obj_y=10.0,
        motors_enabled=True,
    )
    expected = controller._calculate_cardinal_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        direction="left",
        distance=40.0,
    )

    assert _to_tuples(actual) == _to_tuples(expected)


def test_cardinal_diagonal_strategy_uses_diagonal_when_threshold_met():
    controller = _make_controller(MovementStrategy.CARDINAL_DIAGONAL)
    obj_x, obj_y = 100.0, 90.0

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=obj_x,
        obj_y=obj_y,
        motors_enabled=True,
    )
    expected = controller._calculate_diagonal_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        direction="down-right",
        distance=math.hypot(obj_x, obj_y),
    )

    assert _to_tuples(actual) == _to_tuples(expected)


def test_zero_displacement_returns_no_motors():
    controller = _make_controller(MovementStrategy.CARDINAL_DIAGONAL)

    assert controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=0.0,
        obj_y=0.0,
        motors_enabled=True,
    ) == []


def test_free_form_clamps_using_screen_radius():
    controller = _make_controller(
        MovementStrategy.FREE_FORM,
        top_width=100.0,
        top_height=100.0,
        edge_threshold=10.0,
    )

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=400.0,
        obj_y=0.0,
        motors_enabled=True,
    )
    expected = controller._calculate_freeform_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=40.0,
        obj_y=0.0,
    )

    assert _to_tuples(actual) == _to_tuples(expected)


def test_non_comparison_reset_returns_selected_motor_set_zero_positions_once():
    controller = _make_controller(MovementStrategy.CARDINAL)

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=12.0,
        obj_y=8.0,
        motors_enabled=False,
        reset_to_origin=True,
    )

    assert _to_tuples(actual) == [(3, 0), (4, 0), (5, 0)]


def test_non_comparison_without_reset_returns_empty():
    controller = _make_controller(MovementStrategy.CARDINAL)

    assert controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=12.0,
        obj_y=8.0,
        motors_enabled=False,
        reset_to_origin=False,
    ) == []


@pytest.mark.parametrize(
    "motor_set_id, expected",
    [
        (MotorSetId.MOTORS_3_5, [(3, 0), (4, 0), (5, 0)]),
        (MotorSetId.MOTORS_12_14, [(12, 0), (13, 0), (14, 0)]),
    ],
)
def test_zero_motor_positions_uses_physical_motor_set(motor_set_id, expected):
    controller = _make_controller(MovementStrategy.CARDINAL)
    assert _to_tuples(controller.zero_motor_positions(motor_set_id)) == expected


def test_move_factor_scales_motor_positions():
    move_factor = 2.0
    scaled = _make_controller(MovementStrategy.CARDINAL, move_factor=move_factor)
    base = _make_controller(MovementStrategy.CARDINAL, move_factor=1.0)

    kwargs = dict(motor_set_id=MotorSetId.MOTORS_3_5, obj_x=20.0, obj_y=0.0, motors_enabled=True)
    actual = scaled.calculate_motor_movements(**kwargs)
    baseline = base.calculate_motor_movements(**kwargs)

    expected = [(m.index, m.pos * move_factor) for m in baseline]
    assert _to_tuples(actual) == expected


@pytest.mark.parametrize(
    "obj_x, obj_y, mirrored_x, mirrored_y",
    [
        (-25.0, 0.0, 25.0, 0.0),
        (0.0, -25.0, 0.0, -25.0),
    ],
    ids=["x_axis_flips", "y_axis_unchanged"],
)
def test_mirrored_orientation(obj_x, obj_y, mirrored_x, mirrored_y):
    normal = _make_controller(MovementStrategy.CARDINAL, hand_orientation=HandOrientation.NOT_MIRRORED)
    mirrored = _make_controller(MovementStrategy.CARDINAL, hand_orientation=HandOrientation.MIRRORED)

    normal_result = normal.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=obj_x,
        obj_y=obj_y,
        motors_enabled=True,
    )
    mirrored_result = mirrored.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=mirrored_x,
        obj_y=mirrored_y,
        motors_enabled=True,
    )

    assert _to_tuples(mirrored_result) == _to_tuples(normal_result)
