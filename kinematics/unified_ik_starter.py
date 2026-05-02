
"""
Unified IK starter for the 3-leg mechanism
=========================================

Frozen design choices from the user
-----------------------------------
1) P2 construction:
   - top:   P2_top = Pm
   - right: rotate (P1 -> Pm) by -120 deg in XY
   - left:  rotate (P1 -> Pm) by +120 deg in XY

2) phi2 definition:
   - use the angle of vector (P2 -> P1) for all legs
   - for the top leg, since P2_top = Pm, this becomes the same practical rule
     as the earlier top note:
         phi2 = atan2(y1 - yPm, x1 - xPm)

3) phi3 definition:
   - use the interior triangle angle from vectors for all legs:
         v1 = P2 - P3
         v2 = Pb - P3
         phi3 = atan2(||v1 x v2||, v1 · v2)

4) phi4, phi5, phi6 definition for all legs:
       phi4 = atan2(z3 - zp, x3 - xp)
       phi5 = atan2(y3 - yp, x3 - xp)
       phi6 = atan2(z3 - zp, y3 - yp)

5) d5 does not exist.
6) historical d4 is renamed to d2.

Design intent
-------------
- math-first implementation
- easy to translate later into C
- unified process across all legs
- only P2 construction remains leg-specific
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any


EPS = 1e-12


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return wrap_to_pi(a - b)


def vec_sub(a: Tuple[float, float, float],
            b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot3(a: Tuple[float, float, float],
         b: Tuple[float, float, float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def cross3(a: Tuple[float, float, float],
           b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def norm3(a: Tuple[float, float, float]) -> float:
    return math.sqrt(dot3(a, a))


def rotate_xy(x: float, y: float, angle_rad: float) -> Tuple[float, float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return c*x - s*y, s*x + c*y


def circle_circle_intersection_xy(
    x0: float, y0: float, r0: float,
    x1: float, y1: float, r1: float,
    eps: float = EPS
) -> List[Tuple[float, float]]:
    """
    Intersect two circles in XY.
    Returns 0, 1, or 2 points.
    """
    dx = x1 - x0
    dy = y1 - y0
    d = math.sqrt(dx*dx + dy*dy)

    if d > r0 + r1 + eps:
        return []
    if d < abs(r0 - r1) - eps:
        return []
    if d < eps and abs(r0 - r1) < eps:
        return []

    a = (r0*r0 - r1*r1 + d*d) / (2.0*d)
    h_sq = r0*r0 - a*a

    if h_sq < -eps:
        return []
    if h_sq < 0.0:
        h_sq = 0.0

    h = math.sqrt(h_sq)

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    rx = -dy * (h / d)
    ry =  dx * (h / d)

    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)

    if h <= eps:
        return [p1]

    return [p1, p2]


def default_model() -> Dict[str, Any]:
    return {
        "anchors": {
            "top":   (0.0, 15.04, 0.0),
            "right": (12.10, -10.86, 0.0),
            "left":  (-12.10, -10.86, 0.0),
        },
        "lengths": {
            "d1": 4.11,
            "d2": 11.0,   # renamed from historical d4
            "d3": 9.40,
        },
        "branch_weights": {
            "phi2": 1.0,
            "phi3": 1.0,
            "phi4": 1.0,
            "phi5": 1.0,
            "phi6": 1.0,
        },
        "leg_rotation_deg": {
            "right": -120.0,
            "left":  +120.0,
        },
    }


def compute_shared_geometry(
    P1: Tuple[float, float, float],
    phi1: float,
    model: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Shared point confirmed by the user:
        Pm = P1 + d1 * [cos(phi1), sin(phi1), 0]
    """
    x1, y1, z1 = P1
    d1 = model["lengths"]["d1"]

    Pm = (
        x1 + d1 * math.cos(phi1),
        y1 + d1 * math.sin(phi1),
        z1,
    )

    return {
        "P1": P1,
        "phi1": phi1,
        "Pm": Pm,
    }


def construct_p2(
    leg_name: str,
    P1: Tuple[float, float, float],
    Pm: Tuple[float, float, float],
    model: Dict[str, Any]
) -> Tuple[float, float, float]:
    """
    Unified leg-specific P2 construction.
    """
    if leg_name == "top":
        return Pm

    x1, y1, z1 = P1
    dx = Pm[0] - x1
    dy = Pm[1] - y1
    angle = math.radians(model["leg_rotation_deg"][leg_name])

    xr, yr = rotate_xy(dx, dy, angle)
    return (x1 + xr, y1 + yr, z1)


def construct_p3_candidates(
    P2: Tuple[float, float, float],
    Pb: Tuple[float, float, float],
    d2: float,
    d3: float
) -> Tuple[List[Tuple[float, float, float]], Optional[str]]:
    """
    Intersect:
      - circle centered at P2, radius d2, in plane z = z2
      - sphere centered at Pb, radius d3
    """
    x2, y2, z2 = P2
    xp, yp, zp = Pb

    rb_sq = d3*d3 - (z2 - zp)*(z2 - zp)
    if rb_sq < 0.0:
        return [], "sphere does not reach the circle plane"

    rb = math.sqrt(max(0.0, rb_sq))
    xy_points = circle_circle_intersection_xy(x2, y2, d2, xp, yp, rb)

    if not xy_points:
        return [], "circle-sphere intersection has no real solution"

    return [(x, y, z2) for (x, y) in xy_points], None


def compute_phi2(
    P1: Tuple[float, float, float],
    P2: Tuple[float, float, float]
) -> float:
    """
    Frozen user choice:
        phi2 = angle of vector (P2 -> P1)
    """
    return math.atan2(P1[1] - P2[1], P1[0] - P2[0])


def compute_phi3(
    P2: Tuple[float, float, float],
    P3: Tuple[float, float, float],
    Pb: Tuple[float, float, float]
) -> float:
    """
    Frozen user choice:
        phi3 = interior angle from vectors
    """
    v1 = vec_sub(P2, P3)
    v2 = vec_sub(Pb, P3)
    return math.atan2(norm3(cross3(v1, v2)), dot3(v1, v2))


def compute_phi4(
    P3: Tuple[float, float, float],
    Pb: Tuple[float, float, float]
) -> float:
    return math.atan2(P3[2] - Pb[2], P3[0] - Pb[0])


def compute_phi5(
    P3: Tuple[float, float, float],
    Pb: Tuple[float, float, float]
) -> float:
    return math.atan2(P3[1] - Pb[1], P3[0] - Pb[0])


def compute_phi6(
    P3: Tuple[float, float, float],
    Pb: Tuple[float, float, float]
) -> float:
    return math.atan2(P3[2] - Pb[2], P3[1] - Pb[1])


def extract_reference_angles(
    leg_name: str,
    reference_angles: Optional[Dict[str, Dict[str, float]]]
) -> Optional[Dict[str, float]]:
    if reference_angles is None:
        return None
    return reference_angles.get(leg_name)


def branch_cost(
    candidate_angles: Dict[str, float],
    reference_angles: Optional[Dict[str, float]],
    weights: Dict[str, float]
) -> float:
    if reference_angles is None:
        return 0.0

    total = 0.0
    for key in ("phi2", "phi3", "phi4", "phi5", "phi6"):
        da = angle_diff(candidate_angles[key], reference_angles[key])
        total += weights[key] * da * da
    return total


def solve_one_leg(
    leg_name: str,
    shared: Dict[str, Any],
    model: Dict[str, Any],
    previous_angles: Optional[Dict[str, Dict[str, float]]] = None,
    rest_angles: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:

    Pb = model["anchors"][leg_name]
    d2 = model["lengths"]["d2"]
    d3 = model["lengths"]["d3"]
    weights = model["branch_weights"]

    P1 = shared["P1"]
    Pm = shared["Pm"]
    phi1 = shared["phi1"]

    P2 = construct_p2(leg_name, P1, Pm, model)
    phi2 = compute_phi2(P1, P2)

    candidates_P3, fail_reason = construct_p3_candidates(P2, Pb, d2, d3)
    if not candidates_P3:
        return {
            "valid": False,
            "fail_reason": fail_reason,
            "P2": P2,
            "candidates_P3": [],
            "selected_P3": None,
            "phi1": phi1,
            "phi2": None,
            "phi3": None,
            "phi4": None,
            "phi5": None,
            "phi6": None,
            "selected_cost": None,
        }

    ref = extract_reference_angles(leg_name, previous_angles)
    if ref is None:
        ref = extract_reference_angles(leg_name, rest_angles)

    scored = []
    for P3 in candidates_P3:
        angles = {
            "phi2": phi2,
            "phi3": compute_phi3(P2, P3, Pb),
            "phi4": compute_phi4(P3, Pb),
            "phi5": compute_phi5(P3, Pb),
            "phi6": compute_phi6(P3, Pb),
        }
        cost = branch_cost(angles, ref, weights)
        scored.append((cost, P3, angles))

    scored.sort(key=lambda item: item[0])
    selected_cost, selected_P3, selected_angles = scored[0]

    return {
        "valid": True,
        "fail_reason": None,
        "P2": P2,
        "candidates_P3": candidates_P3,
        "selected_P3": selected_P3,
        "phi1": phi1,
        "phi2": selected_angles["phi2"],
        "phi3": selected_angles["phi3"],
        "phi4": selected_angles["phi4"],
        "phi5": selected_angles["phi5"],
        "phi6": selected_angles["phi6"],
        "selected_cost": selected_cost,
    }


def solve_all_legs(
    P1: Tuple[float, float, float],
    phi1: float,
    previous_angles: Optional[Dict[str, Dict[str, float]]] = None,
    rest_angles: Optional[Dict[str, Dict[str, float]]] = None,
    model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if model is None:
        model = default_model()

    shared = compute_shared_geometry(P1, phi1, model)

    return {
        "shared": shared,
        "top": solve_one_leg("top", shared, model, previous_angles, rest_angles),
        "right": solve_one_leg("right", shared, model, previous_angles, rest_angles),
        "left": solve_one_leg("left", shared, model, previous_angles, rest_angles),
    }


if __name__ == "__main__":
    P1 = (0.0, 0.0, 9.62)
    phi1 = math.pi / 2.0

    result = solve_all_legs(P1, phi1)

    for leg in ("top", "right", "left"):
        print(f"\n--- {leg.upper()} ---")
        print("valid      :", result[leg]["valid"])
        print("fail_reason:", result[leg]["fail_reason"])
        print("P2         :", result[leg]["P2"])
        print("P3         :", result[leg]["selected_P3"])
        print("phi1       :", result[leg]["phi1"])
        print("phi2       :", result[leg]["phi2"])
        print("phi3       :", result[leg]["phi3"])
        print("phi4       :", result[leg]["phi4"])
        print("phi5       :", result[leg]["phi5"])
        print("phi6       :", result[leg]["phi6"])
