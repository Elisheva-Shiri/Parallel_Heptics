"""
Forward kinematics for the 3-leg skin-stretch mechanism
=======================================================

This module reconstructs the tactor/platform center ``P1`` from the three wire
lengths measured from the fixed motor anchors.  It intentionally solves only the
state that is observable from those three wire lengths:

    lengths -> P1 = (x, y, z)

It does **not** recover the platform yaw ``phi1`` or the passive joint angles;
those are not uniquely encoded by three anchor-to-P1 distances.  The inverse
kinematics in ``unified_ik_starter.py`` can still be used after FK if you choose
or measure ``phi1`` separately.

Two FK entry points are provided:

* ``forward_kinematics(lengths, z, model)``
    Fixed-height FK used by the current controller.  It solves x/y while z is
    known.  The implementation is a damped Newton least-squares solve against
    the actual length residuals, with a closed-form trilateration seed.

* ``forward_kinematics_3d(lengths, model)``
    Reconstructs x/y/z from absolute lengths when the anchors are coplanar.
    Because all anchors lie in z=0, the mirror solution below the anchor plane is
    mathematically identical; the function chooses positive z by default.

For controller commands, motors usually provide length *deltas* from the origin,
not absolute lengths.  Use ``lengths_from_deltas`` or
``forward_kinematics_from_deltas`` for that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

try:  # package import when called from the repo root
    from .unified_ik_starter import default_model
except ImportError:  # script import when run from this folder
    from unified_ik_starter import default_model


LEG_ORDER = ("top", "right", "left")


@dataclass(frozen=True)
class FKResult:
    """Structured FK output with diagnostics.

    Attributes:
        point: Recovered ``P1 = [x, y, z]``.
        residual_rms: RMS length mismatch in model units.
        residuals: Per-leg modelled length minus input length, in ``LEG_ORDER``.
        iterations: Number of nonlinear iterations used.
        converged: Whether the residual/step tolerance was reached.
        rank: Numerical rank of the final Jacobian used by the solver.
        condition_number: Condition number of the final FK Jacobian block.
        message: Human-readable status.
    """

    point: np.ndarray
    residual_rms: float
    residuals: np.ndarray
    iterations: int
    converged: bool
    rank: int
    condition_number: float
    message: str


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def anchors_array(model: Dict[str, Any]) -> np.ndarray:
    """Return the 3x3 anchor array in ``LEG_ORDER``."""
    return np.array([model["anchors"][leg] for leg in LEG_ORDER], dtype=float)


def working_height(model: Dict[str, Any]) -> float:
    """Fixed tactor height used by the runtime controller."""
    return max(0.0, min(6.0, model["lengths"]["d3"] - 0.1))


def origin_point(model: Dict[str, Any], z: float | None = None) -> np.ndarray:
    """Nominal center point used as the wire-delta reference."""
    if z is None:
        z = working_height(model)
    return np.array([0.0, 0.0, float(z)], dtype=float)


def actuated_lengths(P1: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """Encode a platform point as three absolute anchor-to-P1 wire lengths."""
    P1 = np.asarray(P1, dtype=float)
    return np.linalg.norm(P1[None, :] - anchors_array(model), axis=1)


def reference_lengths(
    model: Dict[str, Any],
    z: float | None = None,
    reference_point: np.ndarray | None = None,
) -> np.ndarray:
    """Absolute wire lengths at the zero/reference platform pose."""
    if reference_point is None:
        reference_point = origin_point(model, z)
    return actuated_lengths(np.asarray(reference_point, dtype=float), model)


def lengths_from_deltas(
    deltas: np.ndarray,
    model: Dict[str, Any],
    z: float | None = None,
    reference_point: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Convert motor wire deltas into absolute FK lengths.

    The controller commands ``final_length - reference_length``.  FK cannot use
    those deltas directly; it must first add the reference/origin lengths.
    """
    deltas = _as_three_vector(deltas, "deltas")
    if reference is None:
        reference = reference_lengths(model, z=z, reference_point=reference_point)
    return np.asarray(reference, dtype=float) + deltas


# ---------------------------------------------------------------------------
# Forward kinematics solvers
# ---------------------------------------------------------------------------
def forward_kinematics(
    lengths: np.ndarray,
    z: float,
    model: Dict[str, Any],
    *,
    initial_xy: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, float] | FKResult:
    """Recover ``P1 = (x, y, z)`` from absolute wire lengths at known height.

    The default return value preserves the original API: ``(P1, residual_rms)``.
    Set ``return_diagnostics=True`` to receive an ``FKResult``.
    """
    result = solve_fixed_height_fk(
        lengths,
        z,
        model,
        initial_xy=initial_xy,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if return_diagnostics:
        return result
    return result.point, result.residual_rms


def solve_fixed_height_fk(
    lengths: np.ndarray,
    z: float,
    model: Dict[str, Any],
    *,
    initial_xy: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> FKResult:
    """Numerically solve fixed-height FK using length residuals.

    Unknowns are x/y only.  The residual equation for each leg is:

        r_i(x, y) = || [x, y, z] - Pb_i || - length_i

    A damped Gauss-Newton/Levenberg-Marquardt iteration minimizes ``sum(r_i^2)``.
    This is deliberately more transparent than returning the closed-form answer
    directly, and it behaves better for noisy or inconsistent measured lengths.
    """
    ell = _as_three_vector(lengths, "lengths")
    A = anchors_array(model)
    z = float(z)

    if initial_xy is None:
        initial_xy = _linear_fixed_height_seed(ell, z, A)
    xy = np.asarray(initial_xy, dtype=float).reshape(2)

    damping = 1e-6
    converged = False
    message = "maximum iterations reached"

    for iteration in range(max_iterations + 1):
        P = np.array([xy[0], xy[1], z], dtype=float)
        residuals, Jxy = _length_residuals_and_jacobian(P, ell, A, columns=2)
        cost = float(0.5 * residuals @ residuals)

        if _rms(residuals) <= tolerance:
            converged = True
            message = "length residual tolerance reached"
            break
        if iteration == max_iterations:
            break

        normal = Jxy.T @ Jxy
        rhs = -(Jxy.T @ residuals)

        accepted = False
        step = np.zeros(2, dtype=float)
        for _ in range(12):
            try:
                step = np.linalg.solve(normal + damping * np.eye(2), rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(normal + damping * np.eye(2), rhs, rcond=None)[0]

            trial_xy = xy + step
            trial_P = np.array([trial_xy[0], trial_xy[1], z], dtype=float)
            trial_residuals = actuated_lengths(trial_P, model) - ell
            trial_cost = float(0.5 * trial_residuals @ trial_residuals)
            if trial_cost <= cost:
                xy = trial_xy
                damping = max(damping * 0.25, 1e-12)
                accepted = True
                break
            damping *= 10.0

        if not accepted:
            message = "no improving solver step found"
            break
        if np.linalg.norm(step) <= tolerance * (1.0 + np.linalg.norm(xy)):
            converged = True
            message = "position step tolerance reached"
            break

    P = np.array([xy[0], xy[1], z], dtype=float)
    residuals, Jxy = _length_residuals_and_jacobian(P, ell, A, columns=2)
    rank, cond = _rank_and_condition(Jxy)

    # If the data were inconsistent, convergence by tiny step is not enough to
    # claim an exact geometric match.  Keep the residual visible in the result.
    if _rms(residuals) > max(1e-8, tolerance * 100.0) and converged:
        message = "best least-squares fit found; input lengths are not exact"

    return FKResult(
        point=P,
        residual_rms=_rms(residuals),
        residuals=residuals,
        iterations=iteration,
        converged=converged,
        rank=rank,
        condition_number=cond,
        message=message,
    )


def forward_kinematics_from_deltas(
    deltas: np.ndarray,
    z: float,
    model: Dict[str, Any],
    *,
    reference_point: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, float] | FKResult:
    """Recover P1 from commanded/measured wire deltas relative to the origin."""
    lengths = lengths_from_deltas(
        deltas,
        model,
        z=z,
        reference_point=reference_point,
        reference=reference,
    )
    return forward_kinematics(
        lengths,
        z,
        model,
        return_diagnostics=return_diagnostics,
    )


def forward_kinematics_3d(
    lengths: np.ndarray,
    model: Dict[str, Any],
    *,
    positive_z: bool = True,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, float] | FKResult:
    """Recover P1 from absolute lengths without a fixed-height input.

    This is valid for the current model because all three anchors are coplanar
    at the same z.  The equations have two mirror solutions: positive and
    negative z.  The physical tactor is above the anchor plane, so positive z is
    selected by default.
    """
    ell = _as_three_vector(lengths, "lengths")
    A = anchors_array(model)
    if not np.allclose(A[:, 2], A[0, 2], atol=1e-12):
        raise ValueError("forward_kinematics_3d requires coplanar anchors with equal z")

    xy = _linear_unknown_height_seed(ell, A)
    dx = xy[0] - A[0, 0]
    dy = xy[1] - A[0, 1]
    z_sq = ell[0] ** 2 - dx * dx - dy * dy
    sign = 1.0 if positive_z else -1.0

    if z_sq >= 0.0:
        z = A[0, 2] + sign * float(np.sqrt(z_sq))
        message = "closed-form 3D sphere intersection"
        converged = True
    else:
        # No real z exactly satisfies the sphere equations; report the closest
        # point on the anchor plane rather than returning NaN.
        z = A[0, 2]
        message = "no real 3D point exactly matches these lengths"
        converged = False

    P = np.array([xy[0], xy[1], z], dtype=float)
    residuals, J = _length_residuals_and_jacobian(P, ell, A, columns=3)
    rank, cond = _rank_and_condition(J)
    result = FKResult(
        point=P,
        residual_rms=_rms(residuals),
        residuals=residuals,
        iterations=0,
        converged=converged and _rms(residuals) <= 1e-8,
        rank=rank,
        condition_number=cond,
        message=message,
    )
    if return_diagnostics:
        return result
    return result.point, result.residual_rms


# ---------------------------------------------------------------------------
# Jacobian and conditioning
# ---------------------------------------------------------------------------
def jacobian(P1: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """Analytic length Jacobian ``J = d(lengths) / d(P1)``.

    Row i is the unit vector from anchor i to P1.  Therefore:

        length_rate = J @ platform_velocity
    """
    P1 = np.asarray(P1, dtype=float)
    A = anchors_array(model)
    diff = P1[None, :] - A
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    if np.any(dist <= 0.0):
        raise ValueError("P1 coincides with an anchor; Jacobian is undefined")
    return diff / dist


def planar_condition_number(P1: np.ndarray, model: Dict[str, Any]) -> float:
    """Condition number of the x/y part of the length Jacobian."""
    _, cond = _rank_and_condition(jacobian(P1, model)[:, :2])
    return cond


# ---------------------------------------------------------------------------
# Internal numeric helpers
# ---------------------------------------------------------------------------
def _as_three_vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values in {LEG_ORDER} order")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _linear_fixed_height_seed(ell: np.ndarray, z: float, A: np.ndarray) -> np.ndarray:
    """Closed-form x/y trilateration seed for the fixed-z solver."""
    dz = z - A[:, 2]
    rho2 = ell**2 - dz**2
    x0, y0 = A[0, 0], A[0, 1]
    M, b = [], []
    for i in range(1, 3):
        xi, yi = A[i, 0], A[i, 1]
        M.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
        b.append((rho2[0] - rho2[i]) - (x0**2 - xi**2) - (y0**2 - yi**2))
    return np.linalg.lstsq(np.asarray(M, dtype=float), np.asarray(b, dtype=float), rcond=None)[0]


def _linear_unknown_height_seed(ell: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Closed-form x/y seed when z is unknown but all anchors share z."""
    x0, y0 = A[0, 0], A[0, 1]
    M, b = [], []
    for i in range(1, 3):
        xi, yi = A[i, 0], A[i, 1]
        M.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
        b.append((ell[0] ** 2 - ell[i] ** 2) - (x0**2 - xi**2) - (y0**2 - yi**2))
    return np.linalg.lstsq(np.asarray(M, dtype=float), np.asarray(b, dtype=float), rcond=None)[0]


def _length_residuals_and_jacobian(
    P: np.ndarray,
    ell: np.ndarray,
    A: np.ndarray,
    *,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    diff = P[None, :] - A
    dist = np.linalg.norm(diff, axis=1)
    safe_dist = np.maximum(dist, 1e-15)
    residuals = dist - ell
    J = diff[:, :columns] / safe_dist[:, None]
    return residuals, J


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def _rank_and_condition(J: np.ndarray) -> tuple[int, float]:
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-12))
    if sv.size == 0 or sv[-1] <= 1e-15:
        return rank, float("inf")
    return rank, float(sv[0] / sv[-1])


# ---------------------------------------------------------------------------
# Validation / demo
# ---------------------------------------------------------------------------
def _validate() -> None:
    model = default_model()
    z = working_height(model)
    print(
        f"model d1,d2,d3 = {model['lengths']['d1']}, "
        f"{model['lengths']['d2']}, {model['lengths']['d3']}; "
        f"working height z = {z}"
    )
    print("FK observable output: P1=(x,y,z). phi1/passive angles need IK or sensors.")

    grid = np.linspace(-5.0, 5.0, 21)
    fixed_errs, delta_errs, full3d_errs, conds = [], [], [], []
    for x in grid:
        for y in grid:
            P1 = np.array([x, y, z], dtype=float)
            lengths = actuated_lengths(P1, model)

            fixed = solve_fixed_height_fk(lengths, z, model)
            fixed_errs.append(np.linalg.norm(fixed.point - P1))

            deltas = lengths - reference_lengths(model, z=z)
            delta = forward_kinematics_from_deltas(
                deltas,
                z,
                model,
                return_diagnostics=True,
            )
            delta_errs.append(np.linalg.norm(delta.point - P1))

            full3d = forward_kinematics_3d(lengths, model, return_diagnostics=True)
            full3d_errs.append(np.linalg.norm(full3d.point - P1))

            conds.append(planar_condition_number(P1, model))

    fixed_errs = np.asarray(fixed_errs)
    delta_errs = np.asarray(delta_errs)
    full3d_errs = np.asarray(full3d_errs)
    conds = np.asarray(conds)

    print(f"\n[1] Fixed-height FK round-trip over +/-5 mm ({fixed_errs.size} points):")
    print(f"    max position error = {fixed_errs.max():.3e} mm   mean = {fixed_errs.mean():.3e} mm")

    print("\n[2] Wire-delta FK round-trip using origin reference lengths:")
    print(f"    max position error = {delta_errs.max():.3e} mm   mean = {delta_errs.mean():.3e} mm")

    print("\n[3] 3D FK round-trip from absolute lengths, choosing positive z:")
    print(
        f"    max position error = {full3d_errs.max():.3e} mm   "
        f"mean = {full3d_errs.mean():.3e} mm"
    )

    rng = np.random.default_rng(0)
    max_j_err = 0.0
    for _ in range(200):
        P1 = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), z], dtype=float)
        Ja = jacobian(P1, model)
        Jn = np.zeros((3, 3), dtype=float)
        h = 1e-6
        for k in range(3):
            dp = np.zeros(3, dtype=float)
            dp[k] = h
            Jn[:, k] = (
                actuated_lengths(P1 + dp, model) - actuated_lengths(P1 - dp, model)
            ) / (2 * h)
        max_j_err = max(max_j_err, float(np.max(np.abs(Ja - Jn))))

    print("\n[4] Jacobian analytic vs finite difference:")
    print(f"    max abs element error = {max_j_err:.3e}")

    print("\n[5] Planar-Jacobian condition number over the workspace:")
    print(
        f"    min = {conds.min():.3f}   median = {np.median(conds):.3f}   "
        f"max = {conds.max():.3f}"
    )
    print("    (near 1 = isotropic / well-conditioned; large = near-singular)")

    ok = (
        fixed_errs.max() < 1e-6
        and delta_errs.max() < 1e-6
        and full3d_errs.max() < 1e-6
        and max_j_err < 1e-6
    )
    print(f"\nVALIDATION {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    _validate()
