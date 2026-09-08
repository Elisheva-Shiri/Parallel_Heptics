from __future__ import annotations

import numpy as np

from kinematics.forward_kinematics import (
    FKResult,
    actuated_lengths,
    default_model,
    forward_kinematics,
    forward_kinematics_3d,
    forward_kinematics_from_deltas,
    reference_lengths,
    solve_fixed_height_fk,
    working_height,
)


def test_fixed_height_fk_round_trips_absolute_lengths() -> None:
    model = default_model()
    z = working_height(model)
    expected = np.array([2.5, -3.0, z])

    lengths = actuated_lengths(expected, model)
    result = solve_fixed_height_fk(lengths, z, model)

    assert result.converged
    assert result.residual_rms < 1e-10
    assert np.allclose(result.point, expected, atol=1e-10)


def test_forward_kinematics_keeps_original_tuple_api() -> None:
    model = default_model()
    z = working_height(model)
    expected = np.array([-1.0, 4.0, z])

    point, residual = forward_kinematics(actuated_lengths(expected, model), z, model)

    assert residual < 1e-10
    assert np.allclose(point, expected, atol=1e-10)


def test_forward_kinematics_diagnostics_api() -> None:
    model = default_model()
    z = working_height(model)
    expected = np.array([1.0, 1.5, z])

    result = forward_kinematics(
        actuated_lengths(expected, model),
        z,
        model,
        return_diagnostics=True,
    )

    assert isinstance(result, FKResult)
    assert result.converged
    assert result.rank == 2
    assert np.allclose(result.point, expected, atol=1e-10)


def test_wire_delta_fk_adds_origin_reference_lengths_before_solving() -> None:
    model = default_model()
    z = working_height(model)
    expected = np.array([3.0, 2.0, z])
    absolute_lengths = actuated_lengths(expected, model)
    deltas = absolute_lengths - reference_lengths(model, z=z)

    result = forward_kinematics_from_deltas(
        deltas,
        z,
        model,
        return_diagnostics=True,
    )

    assert result.residual_rms < 1e-10
    assert np.allclose(result.point, expected, atol=1e-10)


def test_3d_fk_selects_positive_mirror_solution_by_default() -> None:
    model = default_model()
    z = working_height(model)
    expected = np.array([-4.0, 1.0, z])

    result = forward_kinematics_3d(
        actuated_lengths(expected, model),
        model,
        return_diagnostics=True,
    )

    assert result.converged
    assert np.allclose(result.point, expected, atol=1e-10)


def test_inconsistent_lengths_keep_nonzero_residual_visible() -> None:
    model = default_model()
    z = working_height(model)
    impossible = reference_lengths(model, z=z) + np.array([4.0, -3.0, 6.0])

    result = solve_fixed_height_fk(impossible, z, model)

    assert result.residual_rms > 1e-3
    assert not np.allclose(actuated_lengths(result.point, model), impossible)
