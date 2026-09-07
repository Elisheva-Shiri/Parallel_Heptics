"""
Forward kinematics and Jacobian for the 3-leg skin-stretch mechanism
====================================================================

WHY THIS EXISTS
---------------
unified_ik_starter.py provides the inverse kinematics (IK): platform pose ->
per-leg geometry. The runtime controller actuates each leg by the straight-line
distance from the platform center P1 to that leg's fixed base anchor Pb_i,

    ell_i = || P1 - Pb_i ||,                                   (actuated length)

and commands wire-length *deltas* of exactly this quantity (see the paper,
"From angles to cable commands"). This module inverts that map:

  * forward_kinematics(lengths, z, model): recover the platform center P1 from
    the three actuated lengths (the actuator -> pose direction). With z known,
    this is planar trilateration against the anchor triangle -- the same anchor
    triangle used by the movement-strategy reconstruction check.

  * jacobian(P1, model): the mechanism Jacobian J = d(ell)/d(P1). Row i is the
    unit vector from anchor i to the platform,
        d ell_i / d P1 = (P1 - Pb_i) / || P1 - Pb_i ||,
    so J maps platform velocity to cable-length rate: ell_dot = J . P1_dot. Its
    conditioning tells us how evenly the three cables resolve planar motion and
    where the mechanism approaches a singularity.

VALIDATION (run as __main__):
  1. Round trip  : sample a grid of platform points in the +/-5 mm workspace,
     encode with ell_i = ||P1 - Pb_i||, decode with forward_kinematics, report
     the position error (expected ~1e-9, i.e. exact up to numerics).
  2. Jacobian    : compare the analytic J against a finite-difference J.
  3. Conditioning: report the planar-Jacobian condition number over the
     workspace (a flag for near-singular poses).

Dependencies: numpy only (plus unified_ik_starter for the shared model/anchors).
Run:  uv run python kinematics/forward_kinematics.py
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

from unified_ik_starter import default_model

LEG_ORDER = ("top", "right", "left")


def anchors_array(model: Dict[str, Any]) -> np.ndarray:
    """3x3 array of the leg base anchors, rows in LEG_ORDER."""
    return np.array([model["anchors"][leg] for leg in LEG_ORDER], dtype=float)


def working_height(model: Dict[str, Any]) -> float:
    """The fixed tactor height used by the runtime: min(6, d3 - 0.1)."""
    return min(6.0, model["lengths"]["d3"] - 0.1)


def actuated_lengths(P1: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """Encode a platform point as the three anchor-to-platform distances."""
    P1 = np.asarray(P1, float)
    return np.linalg.norm(P1[None, :] - anchors_array(model), axis=1)


def forward_kinematics(
    lengths: np.ndarray,
    z: float,
    model: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """
    Recover the platform center P1 = (x, y, z) from the three actuated cable
    lengths, with the height z known (the runtime fixes z, so the mechanism is
    driven in a plane). Planar trilateration by linear least squares.

    Returns (P1, residual) where residual is the RMS mismatch between the given
    lengths and the lengths implied by the recovered point (0 for a consistent
    triple; grows if the three lengths are inconsistent).
    """
    A = anchors_array(model)            # (3,3), anchor z == 0
    ell = np.asarray(lengths, float)    # (3,)

    # In-plane squared radius from each anchor: rho_i^2 = ell_i^2 - (z - z_anchor)^2
    dz = z - A[:, 2]
    rho2 = ell**2 - dz**2               # (3,)

    # (x - xi)^2 + (y - yi)^2 = rho_i^2. Subtract the first equation from the
    # others to linearize:  2(xi - x0) x + 2(yi - y0) y = (rho0^2 - rhoi^2)
    #                                                      - (x0^2 - xi^2) - (y0^2 - yi^2)
    x0, y0 = A[0, 0], A[0, 1]
    M, b = [], []
    for i in range(1, 3):
        xi, yi = A[i, 0], A[i, 1]
        M.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
        b.append((rho2[0] - rho2[i]) - (x0**2 - xi**2) - (y0**2 - yi**2))
    M = np.asarray(M, float)
    b = np.asarray(b, float)
    xy, *_ = np.linalg.lstsq(M, b, rcond=None)
    P1 = np.array([xy[0], xy[1], z], float)

    residual = float(np.sqrt(np.mean((actuated_lengths(P1, model) - ell) ** 2)))
    return P1, residual


def jacobian(P1: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """
    Analytic mechanism Jacobian J (3x3): row i = d ell_i / d P1 = unit(P1 - Pb_i).
    ell_dot = J . P1_dot.
    """
    P1 = np.asarray(P1, float)
    A = anchors_array(model)
    diff = P1[None, :] - A                       # (3,3)
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    return diff / dist                           # unit rows


def planar_condition_number(P1: np.ndarray, model: Dict[str, Any]) -> float:
    """Condition number of the in-plane (x,y) Jacobian: sigma_max / sigma_min."""
    Jxy = jacobian(P1, model)[:, :2]             # (3,2)
    sv = np.linalg.svd(Jxy, compute_uv=False)
    return float(sv[0] / sv[-1])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate() -> None:
    model = default_model()
    z = working_height(model)
    print(f"model d1,d2,d3 = "
          f"{model['lengths']['d1']}, {model['lengths']['d2']}, "
          f"{model['lengths']['d3']};  working height z = {z}")

    # 1) Round-trip encode/decode over the +/-5 mm nominal workspace
    grid = np.linspace(-5.0, 5.0, 21)
    errs, conds = [], []
    for x in grid:
        for y in grid:
            P1 = np.array([x, y, z])
            ell = actuated_lengths(P1, model)
            P1_rec, res = forward_kinematics(ell, z, model)
            errs.append(np.linalg.norm(P1_rec - P1))
            conds.append(planar_condition_number(P1, model))
    errs = np.asarray(errs)
    conds = np.asarray(conds)
    print("\n[1] FK round-trip over +/-5 mm workspace "
          f"({errs.size} points):")
    print(f"    max position error = {errs.max():.3e} mm   "
          f"mean = {errs.mean():.3e} mm")

    # 2) Analytic vs finite-difference Jacobian
    rng = np.random.default_rng(0)
    max_j_err = 0.0
    for _ in range(200):
        P1 = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), z])
        Ja = jacobian(P1, model)
        Jn = np.zeros((3, 3))
        h = 1e-6
        for k in range(3):
            dp = np.zeros(3); dp[k] = h
            Jn[:, k] = (actuated_lengths(P1 + dp, model)
                        - actuated_lengths(P1 - dp, model)) / (2 * h)
        max_j_err = max(max_j_err, float(np.max(np.abs(Ja - Jn))))
    print("\n[2] Jacobian analytic vs finite difference:")
    print(f"    max abs element error = {max_j_err:.3e}")

    # 3) Conditioning over the workspace
    print("\n[3] Planar-Jacobian condition number over the workspace:")
    print(f"    min = {conds.min():.3f}   median = {np.median(conds):.3f}   "
          f"max = {conds.max():.3f}")
    print("    (near 1 = isotropic / well-conditioned; large = near-singular)")

    ok = errs.max() < 1e-6 and max_j_err < 1e-6
    print(f"\nVALIDATION {'PASSED' if ok else 'FAILED'}: "
          "FK inverts the actuated-length map and the analytic Jacobian matches "
          "finite differences.")


if __name__ == "__main__":
    _validate()
