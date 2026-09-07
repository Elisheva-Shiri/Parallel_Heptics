from __future__ import annotations

import math

import numpy as np

from kinematics.wire_forward_kinematics import (
    CABLE_ATTACHMENT_FRACTION,
    LEG_ORDER,
    PHI1_FIXED_DEG,
    default_model,
    platform_transform,
    pose_to_wire_deltas,
    rest_pose,
    solve_wire_fk,
)


def test_rest_wire_deltas_are_zero_and_fk_recovers_origin() -> None:
    model = default_model()
    rest = rest_pose(model)

    deltas, _ = pose_to_wire_deltas(rest, model)
    result = solve_wire_fk(deltas, model)

    assert np.allclose(deltas, np.zeros(3), atol=1e-12)
    assert result.valid
    assert result.rank == 3
    assert result.residual_rms < 1e-9
    assert np.allclose(result.P1, rest, atol=1e-8)


def test_wire_fk_round_trips_pose_with_solved_z() -> None:
    model = default_model()
    expected = np.array([1.0, -1.0, 6.2])

    deltas, _ = pose_to_wire_deltas(expected, model)
    result = solve_wire_fk(deltas, model)

    assert result.valid
    assert result.residual_rms < 1e-9
    assert np.allclose(result.P1, expected, atol=1e-7)


def test_wire_fk_round_trips_small_3d_workspace() -> None:
    model = default_model()
    for x in np.linspace(-2.0, 2.0, 3):
        for y in np.linspace(-2.0, 2.0, 3):
            for z in np.linspace(5.5, 6.5, 3):
                expected = np.array([x, y, z])
                deltas, _ = pose_to_wire_deltas(expected, model)
                result = solve_wire_fk(deltas, model)

                assert result.valid
                assert result.rank == 3
                assert result.condition_number < 10.0
                assert np.allclose(result.P1, expected, atol=1e-6)


def test_link_lengths_and_attachment_fraction_match_model() -> None:
    model = default_model()
    expected = np.array([1.0, -1.0, 6.2])

    deltas, _ = pose_to_wire_deltas(expected, model)
    result = solve_wire_fk(deltas, model)

    assert set(result.legs) == set(LEG_ORDER)
    for leg in LEG_ORDER:
        geom = result.legs[leg]
        d2 = model["lengths"]["d2"]
        d3 = model["lengths"]["d3"]
        assert np.isclose(np.linalg.norm(geom.P3 - geom.Pb), d3, atol=1e-9)
        assert np.isclose(np.linalg.norm(geom.P2 - geom.P3), d2, atol=1e-9)
        assert 0 <= geom.branch_index < geom.branch_candidate_count
        assert np.allclose(
            geom.attachment,
            geom.Pb + CABLE_ATTACHMENT_FRACTION * (geom.P3 - geom.Pb),
        )


def test_wire_fk_decodes_controller_ik_deltas_across_the_used_workspace() -> None:
    """FK must invert the deltas `MotorController` actually emits.

    The other round-trip tests encode with `pose_to_wire_deltas`, which is this
    module's own inverse, so they only prove internal self-consistency. This
    one drives the real IK path in `MotorController` and checks FK recovers the
    commanded point - the pairing the movement-strategy study depends on.

    Amplitudes cover the radius that study uses (160 controller units), which
    is outside the +/-2 model-unit box the other tests sample.
    """
    from motor_controller import HandOrientation, MotorController, MotorSetId, MovementStrategy

    controller = MotorController(
        movement_strategy=MovementStrategy.IK,
        top_width=640.0,
        top_height=480.0,
        edge_threshold=30.0,
        motor_spacing=1000.0,
        move_factor=1.0,
        hand_orientation=HandOrientation.NOT_MIRRORED,
    )
    model = controller._get_ik_model()
    sim_to_ik = controller._get_ik_base_span(model) / 1000.0
    rest = rest_pose(model)

    worst_exact = 0.0
    worst_quantised = 0.0
    for angle_deg in range(0, 360, 15):
        for radius in (40.0, 100.0, 160.0):
            x = radius * math.cos(math.radians(angle_deg))
            y = radius * math.sin(math.radians(angle_deg))

            movements = controller.calculate_motor_movements(
                motor_set_id=MotorSetId.MOTORS_0_2,
                obj_x=x,
                obj_y=y,
                motors_enabled=True,
            )
            quantised_deltas = np.array(
                [m.pos for m in sorted(movements, key=lambda m: m.index)], dtype=float
            )

            # The controller negates the target internally (actuator polarity).
            expected_p1 = np.array([-x * sim_to_ik, -y * sim_to_ik, _tactor_z(controller, model)])
            exact_deltas, _ = pose_to_wire_deltas(expected_p1, model, rest_P1=rest)

            exact = solve_wire_fk(exact_deltas, model, rest_P1=rest)
            quantised = solve_wire_fk(quantised_deltas * sim_to_ik, model, rest_P1=rest)

            assert exact.valid and quantised.valid
            worst_exact = max(worst_exact, float(np.linalg.norm(exact.P1[:2] - expected_p1[:2])))
            worst_quantised = max(
                worst_quantised, float(np.linalg.norm(quantised.P1[:2] - expected_p1[:2]))
            )

    # 1. FK genuinely inverts the mechanism the controller's IK path uses.
    assert worst_exact < 1e-6, f"FK/IK pairing broken: {worst_exact:.3e} model units"

    # 2. The residual error is command quantisation, not model mismatch.
    #    `int()` truncation costs up to ~1 count per motor, amplified by the
    #    mechanism to a few counts of tactor position. Recorded here so the
    #    movement-strategy study can quote it as its noise floor.
    assert worst_quantised < 8.0 * sim_to_ik, (
        f"quantisation floor grew to {worst_quantised / sim_to_ik:.2f} command counts"
    )
    assert worst_quantised > worst_exact


def _tactor_z(controller, model) -> float:
    return controller._get_ik_tactor_z(model)


def test_platform_transform_has_fixed_phi1_yaw_and_translation() -> None:
    P1 = np.array([1.0, 2.0, 6.0])
    T = platform_transform(P1)

    expected_yaw = math.radians(PHI1_FIXED_DEG)
    expected = np.array(
        [
            [math.cos(expected_yaw), -math.sin(expected_yaw), 0.0, 1.0],
            [math.sin(expected_yaw), math.cos(expected_yaw), 0.0, 2.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(T, expected)
