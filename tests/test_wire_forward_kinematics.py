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
