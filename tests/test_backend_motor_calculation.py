import math
from pathlib import Path

import pytest

from motor_controller import HandOrientation, MotorController, MotorMovement, MotorSetId, MovementStrategy
from kinematics import wire_forward_kinematics as wire_fk


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


def test_cardinal_strategy_snaps_to_major_axis_preserving_radius():
    """Cardinal keeps only the dominant axis, at the full commanded radius.

    The expected point is computed here, independently of the controller, so
    this fails if the quantiser flips a sign or drops magnitude.
    """
    controller = _make_controller(MovementStrategy.CARDINAL)
    obj_x, obj_y = -40.0, 10.0

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=obj_x,
        obj_y=obj_y,
        motors_enabled=True,
    )
    # Polarity flip is applied inside the controller, so the tactor target is
    # the opposite point; |x| > |y| there too, so X stays the dominant axis.
    expected_point = (math.hypot(obj_x, obj_y), 0.0)
    expected = controller._calculate_planar_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=expected_point[0],
        obj_y=expected_point[1],
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
    # Ratio 0.9 >= 0.5, so the target snaps to the 45-degree diagonal opposite
    # the object, keeping the full radius.
    leg = math.hypot(obj_x, obj_y) / math.sqrt(2.0)
    expected = controller._calculate_planar_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=-leg,
        obj_y=-leg,
    )

    assert _to_tuples(actual) == _to_tuples(expected)


@pytest.mark.parametrize(
    "obj_x, obj_y",
    [(0.0, 60.0), (0.0, -60.0), (60.0, 0.0), (-60.0, 0.0), (40.0, 55.0), (-55.0, -40.0)],
)
def test_all_strategies_agree_on_direction_for_the_same_input(obj_x, obj_y):
    """Every strategy must aim into the same quadrant as free-form.

    Strategies differ in how coarsely they quantise direction, never in which
    way an axis points. This is the invariant the old direction-string mapping
    violated on Y: cardinal/CD aimed opposite to free-form for the same input.

    The check is on the quantised target point, not on motor command signs -
    the anchor triangle is not axis-aligned, so zeroing an axis can legitimately
    flip an individual cable's delta.
    """
    reference_x, reference_y = _make_controller(MovementStrategy.FREE_FORM)._quantize_target(
        obj_x, obj_y
    )
    for strategy in (MovementStrategy.CARDINAL, MovementStrategy.CARDINAL_DIAGONAL):
        target_x, target_y = _make_controller(strategy)._quantize_target(obj_x, obj_y)
        # A coarser strategy may zero an axis, but must never invert one.
        assert target_x * reference_x >= 0.0, f"{strategy} inverted X"
        assert target_y * reference_y >= 0.0, f"{strategy} inverted Y"


def test_quantiser_preserves_radius_and_never_inverts_an_axis():
    """`_quantize_target` is a pure, radius-preserving direction snap."""
    for strategy in (MovementStrategy.CARDINAL, MovementStrategy.CARDINAL_DIAGONAL):
        controller = _make_controller(strategy)
        for angle_deg in range(0, 360, 7):
            radius = 137.0
            x = radius * math.cos(math.radians(angle_deg))
            y = radius * math.sin(math.radians(angle_deg))
            qx, qy = controller._quantize_target(x, y)

            assert math.isclose(math.hypot(qx, qy), radius, rel_tol=1e-9)
            # No component may point against the input component.
            assert qx * x >= 0.0
            assert qy * y >= 0.0


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
    expected = controller._calculate_planar_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=-40.0,
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


def test_build_message_preserves_motor_position_order():
    controller = _make_controller(MovementStrategy.FREE_FORM)
    motors = [
        MotorMovement(index=3, pos=120),
        MotorMovement(index=4, pos=-45),
        MotorMovement(index=5, pos=0),
    ]

    assert controller.build_message(motors) == "ZM3P120M4P-45M5P0F"


def test_ik_strategy_uses_opposite_destination_position():
    actual_controller = _make_controller(MovementStrategy.IK)
    expected_controller = _make_controller(MovementStrategy.IK)

    actual = actual_controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=160.0,
        obj_y=0.0,
        motors_enabled=True,
    )
    expected = expected_controller._calculate_ik_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=-160.0,
        obj_y=0.0,
    )

    assert _to_tuples(actual) == _to_tuples(expected)


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


def test_ik_strategy_origin_returns_zero_positions():
    controller = _make_controller(MovementStrategy.IK)

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        stiffness_value=1.0,
        obj_x=0.0,
        obj_y=0.0,
        motors_enabled=True,
    )

    assert _to_tuples(actual) == [(3, 0), (4, 0), (5, 0)]


def test_ik_strategy_returns_three_scaled_motor_commands():
    controller = _make_controller(MovementStrategy.IK)

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        stiffness_value=0.5,
        obj_x=160.0,
        obj_y=0.0,
        motors_enabled=True,
    )

    assert [movement.index for movement in actual] == [3, 4, 5]
    assert any(movement.pos != 0 for movement in actual)


def test_ik_strategy_translates_target_to_mechanism_wire_deltas():
    controller = _make_controller(MovementStrategy.IK)
    model = controller._get_ik_model()
    obj_x = 120.0
    obj_y = -60.0

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        stiffness_value=1.0,
        obj_x=obj_x,
        obj_y=obj_y,
        motors_enabled=True,
    )

    sim_to_ik_scale = controller._get_ik_base_span(model) / controller._motor_spacing
    ik_to_sim_scale = 1.0 / sim_to_ik_scale
    expected_obj_x, expected_obj_y = controller._apply_stiffness_value(obj_x, obj_y, 1.0)
    expected_obj_x, expected_obj_y = controller._apply_actuator_destination_polarity(expected_obj_x, expected_obj_y)
    p1 = (
        expected_obj_x * sim_to_ik_scale,
        expected_obj_y * sim_to_ik_scale,
        controller._get_ik_tactor_z(model),
    )
    reference_p1 = (0.0, 0.0, controller._get_ik_tactor_z(model))
    expected_deltas, _ = wire_fk.pose_to_wire_deltas(p1, model, rest_P1=reference_p1)
    expected = [
        (index, int(delta * ik_to_sim_scale))
        for index, delta in enumerate(expected_deltas, start=MotorSetId.MOTORS_3_5.base_index)
    ]

    assert _to_tuples(actual) == expected


def test_zero_motor_positions_resets_ik_branch_state_to_origin():
    controller = _make_controller(MovementStrategy.IK)

    controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=140.0,
        obj_y=70.0,
        motors_enabled=True,
    )
    moved_angles = controller._ik_previous_angles

    assert moved_angles is not None
    assert _to_tuples(controller.zero_motor_positions(MotorSetId.MOTORS_3_5)) == [(3, 0), (4, 0), (5, 0)]

    model = controller._get_ik_model()
    origin_result = controller._get_ik_module().solve_all_legs(
        (0.0, 0.0, controller._get_ik_tactor_z(model)),
        math.pi / 2.0,
        model=model,
    )
    assert controller._ik_previous_angles == controller._extract_ik_angles(origin_result)
    assert controller._ik_previous_angles != moved_angles


def test_ik_strategy_uses_repo_local_solver_module():
    controller = _make_controller(MovementStrategy.IK)
    module_path = Path(controller._get_ik_module().__file__).resolve()

    assert module_path.name == "unified_ik_starter.py"
    assert module_path.parent.name == "kinematics"
    assert module_path.parents[1] == Path(__file__).resolve().parents[1]
