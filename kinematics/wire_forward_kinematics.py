"""
Wire-driven forward kinematics for the current 3-leg parallel mechanism.

This is a simulation FK for the *same geometry* used by ``unified_ik_starter``:

    Pb_i --d3--> P3_i --d2--> P2_i --d1--> P1

but the input is motor wire release/pull deltas, not a desired pose.  The IK file
is not modified and this module does not call the IK solver; it imports only the
shared dimensions from ``default_model()``.

Model assumptions chosen from the current discussion
----------------------------------------------------
* ``phi1`` is fixed at 90 degrees.
* ``z`` is solved, not fixed.
* Each leg's first-link point ``P3`` remains in the same horizontal plane as the
  platform attachment ``P2`` (the existing IK constraint ``z3 = z2``).
* The cable attaches to the first link at ``CABLE_ATTACHMENT_FRACTION`` of the
  distance from ``Pb`` to ``P3``.  Default: halfway.
* Motors/pulleys are ignored.  Therefore the only well-defined cable input we
  can model without another fixed guide point is a **line-of-action displacement**:

      wire_delta_i = dot(A_i - A_i_rest, cable_exit_direction_i)

  where ``A_i`` is the cable attachment point on the first link.  Positive delta
  means wire released/longer by default; flip ``WIRE_RELEASE_SIGN`` if your motor
  convention is opposite.
* The cable exit direction is configurable at the top of this file.  By default
  it points from each base anchor toward the platform/rest center in the base
  plane, elevated upward by 55 degrees.  A rest-first-link azimuth option is
  kept for experiments, but it is not the default because it can make the
  three-delta FK map non-unique in this geometry.

If you later measure a real guide/pulley exit point, replace the projection model
with an exact distance ``||A_i - G_i|| - ||A_rest_i - G_i||``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

try:  # package import from repo root
    from .unified_ik_starter import (
        compute_phi2,
        compute_phi3,
        compute_phi4,
        compute_phi5,
        compute_phi6,
        compute_shared_geometry,
        construct_p2,
        construct_p3_candidates,
        default_model,
    )
except ImportError:  # script import from this folder
    from unified_ik_starter import (
        compute_phi2,
        compute_phi3,
        compute_phi4,
        compute_phi5,
        compute_phi6,
        compute_shared_geometry,
        construct_p2,
        construct_p3_candidates,
        default_model,
    )


LEG_ORDER = ("top", "right", "left")

# ---------------------------------------------------------------------------
# User-adjustable simulation variables
# ---------------------------------------------------------------------------
PHI1_FIXED_DEG = 90.0
CABLE_ATTACHMENT_FRACTION = 0.50       # 0=at Pb, 1=at P3
CABLE_EXIT_ELEVATION_DEG = 55.0        # upward angle from the base XY plane
# Options: "toward_platform_center", "rest_first_link", or CABLE_EXIT_AZIMUTH_DEG.
CABLE_EXIT_AZIMUTH_MODE = "toward_platform_center"
CABLE_EXIT_AZIMUTH_DEG: dict[str, float] | None = None
WIRE_RELEASE_SIGN = 1.0                # set -1 if positive motor command means pull/shorten

# Conservative solve bounds.  They are deliberately easy to edit for simulation.
XY_BOUND_MM = 8.0
Z_MIN_MM = 0.05
SOLVE_TOL = 1e-9
MAX_ITERATIONS = 80


@dataclass(frozen=True)
class LegWireGeometry:
    leg: str
    Pb: np.ndarray
    P2: np.ndarray
    P3: np.ndarray
    branch_index: int
    branch_candidate_count: int
    attachment: np.ndarray
    cable_exit_direction: np.ndarray
    wire_delta: float
    angles_rad: dict[str, float]


@dataclass(frozen=True)
class WireFKResult:
    valid: bool
    P1: np.ndarray
    phi1_rad: float
    transform: np.ndarray
    residuals: np.ndarray
    residual_rms: float
    iterations: int
    converged: bool
    rank: int
    condition_number: float
    message: str
    legs: dict[str, LegWireGeometry]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def rest_pose(model: dict[str, Any] | None = None) -> np.ndarray:
    """Nominal origin pose used as the zero wire-delta reference."""
    if model is None:
        model = default_model()
    z = max(0.0, min(6.0, model["lengths"]["d3"] - 0.1))
    return np.array([0.0, 0.0, z], dtype=float)


def pose_to_wire_deltas(
    P1: Iterable[float],
    model: dict[str, Any] | None = None,
    *,
    rest_P1: Iterable[float] | None = None,
) -> tuple[np.ndarray, dict[str, LegWireGeometry]]:
    """Encode a pose as three wire release deltas in ``LEG_ORDER``.

    This is the forward simulation map used inside FK residuals.  It is useful
    for validating round trips and for generating synthetic motor commands.
    """
    if model is None:
        model = default_model()
    P1 = np.asarray(P1, dtype=float)
    if rest_P1 is None:
        rest_P1 = rest_pose(model)
    rest = _solve_pose_geometry(np.asarray(rest_P1, dtype=float), model, branch_reference=None)
    current = _solve_pose_geometry(P1, model, branch_reference=rest)

    deltas = []
    legs: dict[str, LegWireGeometry] = {}
    for leg in LEG_ORDER:
        cur = current[leg]
        ref = rest[leg]
        exit_dir = _cable_exit_direction(leg, model)
        delta = WIRE_RELEASE_SIGN * float(np.dot(cur["attachment"] - ref["attachment"], exit_dir))
        geom = LegWireGeometry(
            leg=leg,
            Pb=cur["Pb"],
            P2=cur["P2"],
            P3=cur["P3"],
            branch_index=cur["branch_index"],
            branch_candidate_count=cur["branch_candidate_count"],
            attachment=cur["attachment"],
            cable_exit_direction=exit_dir,
            wire_delta=delta,
            angles_rad=cur["angles_rad"],
        )
        legs[leg] = geom
        deltas.append(delta)
    return np.asarray(deltas, dtype=float), legs


def solve_wire_fk(
    wire_deltas: Iterable[float],
    model: dict[str, Any] | None = None,
    *,
    initial_P1: Iterable[float] | None = None,
    rest_P1: Iterable[float] | None = None,
    tolerance: float = SOLVE_TOL,
    max_iterations: int = MAX_ITERATIONS,
) -> WireFKResult:
    """Solve ``P1=(x,y,z)`` from three wire deltas with ``phi1=90 deg`` fixed."""
    if model is None:
        model = default_model()
    target = _as_three_vector(wire_deltas, "wire_deltas")
    if rest_P1 is None:
        rest_P1 = rest_pose(model)
    rest_P1 = np.asarray(rest_P1, dtype=float)
    if initial_P1 is None:
        initial_P1 = rest_P1
    q = _clip_pose(np.asarray(initial_P1, dtype=float), model)

    rest_geom = _solve_pose_geometry(rest_P1, model, branch_reference=None)
    damping = 1e-5
    converged = False
    message = "maximum iterations reached"
    residuals = np.full(3, np.nan, dtype=float)

    for iteration in range(max_iterations + 1):
        residuals = _wire_residual(q, target, model, rest_geom)
        if _rms(residuals) <= tolerance:
            converged = True
            message = "wire residual tolerance reached"
            break
        if iteration == max_iterations:
            break

        J = _finite_difference_jacobian(q, target, model, rest_geom)
        normal = J.T @ J
        rhs = -(J.T @ residuals)
        cost = 0.5 * float(residuals @ residuals)

        accepted = False
        step = np.zeros(3, dtype=float)
        for _ in range(14):
            try:
                step = np.linalg.solve(normal + damping * np.eye(3), rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(normal + damping * np.eye(3), rhs, rcond=None)[0]
            trial = _clip_pose(q + step, model)
            trial_res = _wire_residual(trial, target, model, rest_geom)
            trial_cost = 0.5 * float(trial_res @ trial_res)
            if np.all(np.isfinite(trial_res)) and trial_cost <= cost:
                q = trial
                damping = max(damping * 0.25, 1e-12)
                accepted = True
                break
            damping *= 10.0

        if not accepted:
            message = "no improving solver step found"
            break
        if np.linalg.norm(step) <= tolerance * (1.0 + np.linalg.norm(q)):
            converged = True
            message = "position step tolerance reached"
            break

    residuals = _wire_residual(q, target, model, rest_geom)
    J = _finite_difference_jacobian(q, target, model, rest_geom)
    rank, cond = _rank_and_condition(J)
    _, legs = pose_to_wire_deltas(q, model, rest_P1=rest_P1)

    if _rms(residuals) > max(1e-6, tolerance * 100.0) and converged:
        message = "best least-squares pose found; wire deltas are not exact"

    return WireFKResult(
        valid=bool(converged and _rms(residuals) <= max(1e-6, tolerance * 100.0)),
        P1=q,
        phi1_rad=math.radians(PHI1_FIXED_DEG),
        transform=platform_transform(q, math.radians(PHI1_FIXED_DEG)),
        residuals=residuals,
        residual_rms=_rms(residuals),
        iterations=iteration,
        converged=converged,
        rank=rank,
        condition_number=cond,
        message=message,
        legs=legs,
    )


def platform_transform(P1: Iterable[float], phi1_rad: float | None = None) -> np.ndarray:
    """Return 4x4 platform transform.  Roll/pitch are not in this model."""
    if phi1_rad is None:
        phi1_rad = math.radians(PHI1_FIXED_DEG)
    P1 = np.asarray(P1, dtype=float)
    c = math.cos(phi1_rad)
    s = math.sin(phi1_rad)
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    T[:3, 3] = P1
    return T


# ---------------------------------------------------------------------------
# Geometry: same equations as the current IK, implemented locally for FK
# ---------------------------------------------------------------------------
def _solve_pose_geometry(
    P1: np.ndarray,
    model: dict[str, Any],
    *,
    branch_reference: dict[str, dict[str, np.ndarray]] | None,
) -> dict[str, dict[str, Any]]:
    P1 = np.asarray(P1, dtype=float)
    phi1 = math.radians(PHI1_FIXED_DEG)
    P1_tuple = _to_point_tuple(P1)
    shared = compute_shared_geometry(P1_tuple, phi1, model)
    out: dict[str, dict[str, np.ndarray | dict[str, float]]] = {}
    for leg in LEG_ORDER:
        Pb = np.asarray(model["anchors"][leg], dtype=float)
        P2 = np.asarray(construct_p2(leg, P1_tuple, shared["Pm"], model), dtype=float)
        reference = branch_reference[leg]["P3"] if branch_reference else None
        P3, branch_index, branch_candidate_count = _select_p3_candidate(P2, Pb, model, reference)
        attachment = Pb + CABLE_ATTACHMENT_FRACTION * (P3 - Pb)
        out[leg] = {
            "Pb": Pb,
            "P2": P2,
            "P3": P3,
            "branch_index": branch_index,
            "branch_candidate_count": branch_candidate_count,
            "attachment": attachment,
            "angles_rad": _angles_for_leg(P1, P2, P3, Pb, phi1),
        }
    return out


def _select_p3_candidate(
    P2: np.ndarray,
    Pb: np.ndarray,
    model: dict[str, Any],
    reference_P3: np.ndarray | None,
) -> tuple[np.ndarray, int, int]:
    d2 = float(model["lengths"]["d2"])
    d3 = float(model["lengths"]["d3"])
    candidates_raw, fail_reason = construct_p3_candidates(
        _to_point_tuple(P2),
        _to_point_tuple(Pb),
        d2,
        d3,
    )
    if fail_reason or not candidates_raw:
        raise ValueError(f"pose is unreachable: {fail_reason}")
    candidates = [np.asarray(candidate, dtype=float) for candidate in candidates_raw]
    if reference_P3 is None:
        return candidates[0], 0, len(candidates)
    reference_P3 = np.asarray(reference_P3, dtype=float)
    branch_index, P3 = min(
        enumerate(candidates),
        key=lambda item: float(np.linalg.norm(item[1] - reference_P3)),
    )
    return P3, branch_index, len(candidates)


def _angles_for_leg(
    P1: np.ndarray,
    P2: np.ndarray,
    P3: np.ndarray,
    Pb: np.ndarray,
    phi1: float,
) -> dict[str, float]:
    v_first = P3 - Pb
    horizontal = math.hypot(v_first[0], v_first[1])
    P1_tuple = _to_point_tuple(P1)
    P2_tuple = _to_point_tuple(P2)
    P3_tuple = _to_point_tuple(P3)
    Pb_tuple = _to_point_tuple(Pb)
    return {
        "phi1": phi1,
        "base_azimuth": math.atan2(v_first[1], v_first[0]),
        "base_elevation": math.atan2(v_first[2], horizontal),
        "phi2": compute_phi2(P1_tuple, P2_tuple),
        "phi3": compute_phi3(P2_tuple, P3_tuple, Pb_tuple),
        "phi4": compute_phi4(P3_tuple, Pb_tuple),
        "phi5": compute_phi5(P3_tuple, Pb_tuple),
        "phi6": compute_phi6(P3_tuple, Pb_tuple),
    }


def _cable_exit_direction(leg: str, model: dict[str, Any]) -> np.ndarray:
    elevation = math.radians(CABLE_EXIT_ELEVATION_DEG)
    if CABLE_EXIT_AZIMUTH_DEG is not None:
        azimuth = math.radians(CABLE_EXIT_AZIMUTH_DEG[leg])
        horizontal = np.array([math.cos(azimuth), math.sin(azimuth), 0.0], dtype=float)
    elif CABLE_EXIT_AZIMUTH_MODE == "rest_first_link":
        horizontal = _rest_first_link_horizontal_direction(leg, model)
    elif CABLE_EXIT_AZIMUTH_MODE in {"toward_platform_center", "toward_origin"}:
        Pb = np.asarray(model["anchors"][leg], dtype=float)
        horizontal = np.array([-Pb[0], -Pb[1], 0.0], dtype=float)
        horizontal /= np.linalg.norm(horizontal)
    else:
        raise ValueError(f"unknown CABLE_EXIT_AZIMUTH_MODE: {CABLE_EXIT_AZIMUTH_MODE}")
    vertical = np.array([0.0, 0.0, math.sin(elevation)], dtype=float)
    direction = math.cos(elevation) * horizontal + vertical
    return direction / np.linalg.norm(direction)


def _rest_first_link_horizontal_direction(leg: str, model: dict[str, Any]) -> np.ndarray:
    """Horizontal cable azimuth: from base toward that leg at the zero pose."""
    rest_geometry = _solve_pose_geometry(rest_pose(model), model, branch_reference=None)[leg]
    Pb = np.asarray(model["anchors"][leg], dtype=float)
    P3_rest = rest_geometry["P3"]
    horizontal = np.array([P3_rest[0] - Pb[0], P3_rest[1] - Pb[1], 0.0], dtype=float)
    norm = np.linalg.norm(horizontal)
    if norm <= 0.0:
        raise ValueError(f"rest first-link horizontal direction is undefined for {leg}")
    return horizontal / norm


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _wire_residual(
    P1: np.ndarray,
    target: np.ndarray,
    model: dict[str, Any],
    rest_geom: dict[str, dict[str, Any]],
) -> np.ndarray:
    try:
        current = _solve_pose_geometry(P1, model, branch_reference=rest_geom)
    except ValueError:
        return np.full(3, 1e6, dtype=float)
    predicted = []
    for leg in LEG_ORDER:
        exit_dir = _cable_exit_direction(leg, model)
        delta = WIRE_RELEASE_SIGN * float(
            np.dot(current[leg]["attachment"] - rest_geom[leg]["attachment"], exit_dir)
        )
        predicted.append(delta)
    return np.asarray(predicted, dtype=float) - target


def _finite_difference_jacobian(
    P1: np.ndarray,
    target: np.ndarray,
    model: dict[str, Any],
    rest_geom: dict[str, dict[str, Any]],
) -> np.ndarray:
    J = np.zeros((3, 3), dtype=float)
    steps = np.array([1e-5, 1e-5, 1e-5], dtype=float)
    for k, h in enumerate(steps):
        dp = np.zeros(3, dtype=float)
        dp[k] = h
        plus = _clip_pose(P1 + dp, model)
        minus = _clip_pose(P1 - dp, model)
        plus_residual = _wire_residual(plus, target, model, rest_geom)
        minus_residual = _wire_residual(minus, target, model, rest_geom)
        J[:, k] = (plus_residual - minus_residual) / (2.0 * h)
    return J


def _clip_pose(P1: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    d3 = float(model["lengths"]["d3"])
    return np.array(
        [
            float(np.clip(P1[0], -XY_BOUND_MM, XY_BOUND_MM)),
            float(np.clip(P1[1], -XY_BOUND_MM, XY_BOUND_MM)),
            float(np.clip(P1[2], Z_MIN_MM, d3 - 1e-6)),
        ],
        dtype=float,
    )


def _as_three_vector(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have exactly three values in {LEG_ORDER} order")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _to_point_tuple(point: Iterable[float]) -> tuple[float, float, float]:
    arr = _as_three_vector(point, "point")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _rank_and_condition(J: np.ndarray) -> tuple[int, float]:
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-10))
    if sv.size == 0 or sv[-1] <= 1e-14:
        return rank, float("inf")
    return rank, float(sv[0] / sv[-1])


def _demo() -> None:
    model = default_model()
    print("wire-driven FK simulation")
    print(f"phi1 fixed = {PHI1_FIXED_DEG} deg")
    print(f"attachment fraction on first link = {CABLE_ATTACHMENT_FRACTION}")
    print(
        f"cable exit mode = {CABLE_EXIT_AZIMUTH_MODE}; "
        f"elevation = {CABLE_EXIT_ELEVATION_DEG} deg"
    )
    print(f"rest P1 = {rest_pose(model)}")

    target = np.array([1.0, -1.0, 6.2], dtype=float)
    deltas, _ = pose_to_wire_deltas(target, model)
    result = solve_wire_fk(deltas, model)
    print("\nsynthetic target P1:", target)
    print("wire deltas:", dict(zip(LEG_ORDER, deltas)))
    print("recovered P1:", result.P1)
    print("residual RMS:", result.residual_rms)
    print("rank/condition:", result.rank, result.condition_number)
    print("T=\n", result.transform)
    for leg in LEG_ORDER:
        angles_deg = {k: math.degrees(v) for k, v in result.legs[leg].angles_rad.items()}
        print(f"\n{leg}: P2={result.legs[leg].P2}, P3={result.legs[leg].P3}")
        print(" angles deg:", angles_deg)


if __name__ == "__main__":
    _demo()

