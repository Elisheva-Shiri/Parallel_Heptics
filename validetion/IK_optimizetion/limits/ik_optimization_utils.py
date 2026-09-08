from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

# Force a non-interactive backend BEFORE importing pyplot. The sweep generates
# tens of thousands of figures; an interactive GUI backend (e.g. tkagg) leaks
# Windows GUI/GDI handles and memory across that many figures and crashes the
# kernel. Agg renders straight to PNG files with no GUI resources, and every
# worker process that imports this module inherits the same safe backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd

from kinematics.unified_ik_starter import (
    angle_diff,
    branch_cost,
    compute_phi2,
    compute_phi3,
    compute_phi4,
    compute_phi5,
    compute_phi6,
    compute_shared_geometry,
    construct_p2,
    construct_p3_candidates,
    default_model,
    solve_all_legs,
    wrap_to_pi,
)


LEGS = ("top", "right", "left")
ANGLE_KEYS = ("phi2", "phi3", "phi4", "phi5", "phi6")
LIMITED_REST_ANGLE_KEYS = ("phi4", "phi6")
PHI456_KEYS = ("phi4", "phi5", "phi6")
DEFAULT_PHI456_REST_LIMIT_DEG = 30.0
MAX_DIMENSION_SCORE_POWER = 4.0
TOTAL_ENVELOPE_TIEBREAKER_WEIGHT = 0.001
CONNECTEDNESS_SCORE_WEIGHT = 0.005
D2_GREATER_THAN_D3_SCALE_MM = 1.0
NON_ACCEPTED_SCORE_FACTOR = 0.1
FINAL_RELATIVE_PHI456_LIMITS_DEG = {
    "top": {
        "phi4": (-15.0, 15.0),
        "phi5": (-90.0, 90.0),
        "phi6": (-40.0, 40.0),
    },
    "right": {
        "phi4": (-20.0, 20.0),
        "phi5": (-90.0, 90.0),
        "phi6": (-35.0, 35.0),
    },
    "left": {
        "phi4": (-20.0, 20.0),
        "phi5": (-90.0, 90.0),
        "phi6": (-35.0, 35.0),
    },
}
FINAL_RELATIVE_PHI456_RUN_LABEL = (
    "top_phi4pm15_phi5pm90_phi6pm40__"
    "side_phi4pm20_phi5pm90_phi6pm35"
)
EPS_REL = 1e-6
_DEGEN_EPS = 1e-6


def inclusive_range(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    return np.round(np.array([start + i * step for i in range(count + 1)]), 6)


def safe_value(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def config_label(row: pd.Series) -> str:
    return (
        f"d1_{safe_value(row['d1_mm'])}__d2_{safe_value(row['d2_mm'])}"
        f"__d3_{safe_value(row['d3_mm'])}"
    )


def movement_restriction_label(cfg: dict[str, Any]) -> str:
    """Human-readable label for plot titles and run metadata."""
    restriction = cfg.get("movement_restriction", {})
    relative_limits = restriction.get("relative_limits_deg")
    if relative_limits:
        top = relative_limits["top"]
        right = relative_limits["right"]
        left = relative_limits["left"]
        side_label = "right/left" if right == left else "right,left"
        return (
            "rest-link-local relative phi limits: "
            f"top phi4 {top['phi4'][0]:g}..{top['phi4'][1]:g}, "
            f"phi5 {top['phi5'][0]:g}..{top['phi5'][1]:g}, "
            f"phi6 {top['phi6'][0]:g}..{top['phi6'][1]:g}; "
            f"{side_label} phi4 {right['phi4'][0]:g}..{right['phi4'][1]:g}, "
            f"phi5 {right['phi5'][0]:g}..{right['phi5'][1]:g}, "
            f"phi6 {right['phi6'][0]:g}..{right['phi6'][1]:g} deg"
        )
    angles = "/".join(restriction.get("angles", LIMITED_REST_ANGLE_KEYS))
    limit = restriction.get("limit_deg", DEFAULT_PHI456_REST_LIMIT_DEG)
    return f"rest-link-local {angles} +/-{limit:g} deg from rest"


def make_model(d1_mm: float, d2_mm: float, d3_mm: float) -> dict[str, Any]:
    """Build a model whose effective IK lengths are set directly.

    The three swept variables ARE the effective solver lengths:
    - d1 = platform offset
    - d2 = SR link length (historical d4)
    - d3 = RR link length

    The anchor/base geometry is copied unchanged from ``default_model()`` so the
    base remains fixed; only the three effective lengths are overridden.
    """
    base = default_model()
    model = default_model()
    model["anchors"] = dict(base["anchors"])
    model["lengths"] = dict(base["lengths"])
    model["lengths"]["d1"] = float(d1_mm)
    model["lengths"]["d2"] = float(d2_mm)
    model["lengths"]["d3"] = float(d3_mm)
    return model


def nominal_rest_pose(model: dict[str, Any]) -> tuple[float, float, float]:
    """Return the rest P1 pose used as the zero-motion reference.

    This mirrors ``kinematics.wire_forward_kinematics.rest_pose`` without
    importing FK code into the sweep workers. The rest pose is recomputed for
    every swept geometry because ``d3`` changes during optimization.
    """
    z = max(0.0, min(6.0, float(model["lengths"]["d3"]) - 0.1))
    return (0.0, 0.0, z)


def movement_restriction_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalized rest-window movement restriction settings."""
    restriction = dict(cfg.get("movement_restriction", {}))
    link_limit = restriction.get("link_elevation_limit_deg")
    relative_limits_deg = normalize_relative_limits_deg(restriction.get("relative_limits_deg"))
    return {
        "enabled": bool(restriction.get("enabled", True)),
        "angles": tuple(restriction.get("angles", LIMITED_REST_ANGLE_KEYS)),
        "limit_deg": float(restriction.get("limit_deg", DEFAULT_PHI456_REST_LIMIT_DEG)),
        "angle_frame": restriction.get("angle_frame", "rest_link_local"),
        "relative_limits_deg": relative_limits_deg,
        "relative_limits_rad": relative_limits_deg_to_rad(relative_limits_deg),
        "branch_selection": restriction.get("branch_selection", "angle_valid"),
        "link_elevation_enabled": link_limit is not None,
        "link_elevation_limit_deg": None if link_limit is None else float(link_limit),
    }


def normalize_relative_limits_deg(relative_limits: Any) -> dict[str, dict[str, tuple[float, float]]] | None:
    """Normalize per-leg relative angle windows in degrees.

    Accepted input shape:

    {
        "top": {"phi4": [-3, 9], "phi5": [-45, 20], "phi6": [-20, 35]},
        ...
    }

    Values may also be ``{"min": -3, "max": 9}``.
    """
    if not relative_limits:
        return None

    normalized: dict[str, dict[str, tuple[float, float]]] = {}
    for leg in LEGS:
        if leg not in relative_limits:
            raise ValueError(f"relative_limits_deg is missing leg {leg!r}")
        normalized[leg] = {}
        for key in PHI456_KEYS:
            if key not in relative_limits[leg]:
                raise ValueError(f"relative_limits_deg[{leg!r}] is missing angle {key!r}")
            raw = relative_limits[leg][key]
            if isinstance(raw, dict):
                lo = float(raw["min"])
                hi = float(raw["max"])
            else:
                lo = float(raw[0])
                hi = float(raw[1])
            if lo > hi:
                raise ValueError(f"relative_limits_deg[{leg!r}][{key!r}] has min > max: {lo} > {hi}")
            normalized[leg][key] = (lo, hi)
    return normalized


def relative_limits_deg_to_rad(
    relative_limits_deg: dict[str, dict[str, tuple[float, float]]] | None,
) -> dict[str, dict[str, tuple[float, float]]] | None:
    if relative_limits_deg is None:
        return None
    return {
        leg: {
            key: (math.radians(bounds[0]), math.radians(bounds[1]))
            for key, bounds in angle_limits.items()
        }
        for leg, angle_limits in relative_limits_deg.items()
    }


def compute_rest_angles(
    phi1: float,
    model: dict[str, Any],
) -> tuple[dict[str, dict[str, float]] | None, tuple[float, float, float], dict[str, Any]]:
    """Solve the rest pose and return per-leg rest angles.

    Returns ``None`` for the angle map when the rest pose itself is invalid; in
    that case no workspace point can satisfy a movement-from-rest restriction.
    """
    rest_p1 = nominal_rest_pose(model)
    rest_solution = solve_all_legs(rest_p1, phi1, model=model)
    rest_valid = all(rest_solution[leg]["valid"] for leg in LEGS)
    if not rest_valid:
        return None, rest_p1, rest_solution
    rest_angles = {
        leg: {key: float(rest_solution[leg][key]) for key in ANGLE_KEYS}
        for leg in LEGS
    }
    return rest_angles, rest_p1, rest_solution


def workspace_efficiency_score(
    metrics: dict[str, Any],
    *,
    d1_mm: float,
    d2_mm: float,
    d3_mm: float,
    largest_component_fraction: float,
    accepted: bool,
) -> dict[str, float]:
    """Rank geometries by useful XY workspace per physical link envelope.

    The design goal for this sweep is a compact mechanism in every swept
    dimension (d1, d2, and d3) while preserving an accepted connected XY
    workspace. Height is intentionally excluded.

    Score components:
    - ``workspace_area_mm2`` rewards largest-component X/Y span.
    - ``max_dimension_mm ** 4`` strongly penalizes the single largest dimension,
      so one oversized value cannot be hidden by two smaller values.
    - a tiny ``total_envelope`` tie-breaker prefers a smaller d1+d2+d3 when the
      largest dimension is the same.
    - a small connectedness bonus favors dense/continuous valid regions without
      dominating balanced compactness.
    - a smooth ``d2 > d3`` preference models the design hypothesis that a
      longer d2 than d3 improves finger-motion smoothness. This is not a hard
      limit: every geometry remains rankable, but d2 <= d3 is down-weighted.
    - non-accepted designs are still visible but strongly down-ranked.
    """
    workspace_area_mm2 = float(metrics["workspace_width_mm"]) * float(metrics["workspace_depth_mm"])
    link_envelope_mm = float(d2_mm) + float(d3_mm)
    total_envelope_mm = float(d1_mm) + link_envelope_mm
    max_dimension_mm = max(float(d1_mm), float(d2_mm), float(d3_mm))
    max_dimension_penalty = max_dimension_mm ** MAX_DIMENSION_SCORE_POWER
    if max_dimension_penalty <= 0.0:
        balanced_efficiency = 0.0
    else:
        balanced_efficiency = workspace_area_mm2 / max_dimension_penalty
    if total_envelope_mm <= 0.0:
        total_envelope_efficiency = 0.0
    else:
        total_envelope_efficiency = workspace_area_mm2 / (total_envelope_mm * total_envelope_mm)
    connectedness_bonus = CONNECTEDNESS_SCORE_WEIGHT * float(largest_component_fraction)
    compact_workspace_score = (
        balanced_efficiency
        + TOTAL_ENVELOPE_TIEBREAKER_WEIGHT * total_envelope_efficiency
        + connectedness_bonus
    )
    d2_minus_d3_mm = float(d2_mm) - float(d3_mm)
    d2_gt_d3_preference_factor = 0.5 + 0.5 * math.tanh(d2_minus_d3_mm / D2_GREATER_THAN_D3_SCALE_MM)
    raw_score = compact_workspace_score * d2_gt_d3_preference_factor
    score = raw_score if accepted else raw_score * NON_ACCEPTED_SCORE_FACTOR
    return {
        "workspace_area_mm2": float(workspace_area_mm2),
        "link_envelope_mm": float(link_envelope_mm),
        "total_envelope_mm": float(total_envelope_mm),
        "max_dimension_mm": float(max_dimension_mm),
        "max_dimension_penalty": float(max_dimension_penalty),
        "total_envelope_efficiency": float(total_envelope_efficiency),
        "workspace_efficiency": float(balanced_efficiency),
        "connectedness_bonus": float(connectedness_bonus),
        "compact_workspace_score": float(compact_workspace_score),
        "d2_minus_d3_mm": float(d2_minus_d3_mm),
        "d2_gt_d3_preference_factor": float(d2_gt_d3_preference_factor),
        "raw_score": float(raw_score),
        "score": float(score),
    }


def rest_link_frame(rest_solution: dict[str, Any], leg: str, model: dict[str, Any]) -> tuple[float, float, float, float]:
    """Local XY frame whose +x axis is the rest Pb -> P3 horizontal direction.

    This makes the movement angles local to each physical leg/link rather than
    local to the global model axes. It preserves left/right mirror symmetry for
    an isosceles triangular base because each leg is measured relative to its own
    rest link direction.
    """
    Pb = model["anchors"][leg]
    P3 = rest_solution[leg]["selected_P3"]
    hx = float(P3[0]) - float(Pb[0])
    hy = float(P3[1]) - float(Pb[1])
    norm = math.hypot(hx, hy)
    if norm <= _DEGEN_EPS:
        # Fallback to the nominal leg rotation if the rest horizontal direction
        # is degenerate. This should not occur for the current geometry.
        angle = 0.0 if leg == "top" else math.radians(model["leg_rotation_deg"][leg])
        ux, uy = math.cos(angle), math.sin(angle)
    else:
        ux, uy = hx / norm, hy / norm
    # Right-handed local XY basis in the horizontal plane.
    vx, vy = -uy, ux
    return ux, uy, vx, vy


def compute_rest_link_frames(rest_solution: dict[str, Any], model: dict[str, Any]) -> dict[str, tuple[float, float, float, float]]:
    return {leg: rest_link_frame(rest_solution, leg, model) for leg in LEGS}


def vector_pb_to_p3_local(P3, Pb, frame: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Vector Pb -> P3 expressed in a rest-link local coordinate frame."""
    ux, uy, vx, vy = frame
    dx = float(P3[0]) - float(Pb[0])
    dy = float(P3[1]) - float(Pb[1])
    dz = float(P3[2]) - float(Pb[2])
    return (dx * ux + dy * uy, dx * vx + dy * vy, dz)


def local_phi456_from_points(P3, Pb, frame: tuple[float, float, float, float]) -> dict[str, float]:
    """phi4/phi5/phi6 of Pb -> P3 measured in the leg's rest-link frame."""
    lx, ly, lz = vector_pb_to_p3_local(P3, Pb, frame)
    return {
        "phi4": math.atan2(lz, lx),
        "phi5": math.atan2(ly, lx),
        "phi6": math.atan2(lz, ly),
    }


def local_phi456_from_result(
    leg_result: dict[str, Any],
    leg: str,
    model: dict[str, Any],
    frame: tuple[float, float, float, float],
) -> dict[str, float]:
    return local_phi456_from_points(leg_result["selected_P3"], model["anchors"][leg], frame)


def compute_rest_local_phi456(
    rest_solution: dict[str, Any],
    model: dict[str, Any],
    rest_link_frames: dict[str, tuple[float, float, float, float]],
) -> dict[str, dict[str, float]]:
    return {leg: local_phi456_from_result(rest_solution[leg], leg, model, rest_link_frames[leg]) for leg in LEGS}


def phi456_within_rest_limit(
    leg_result: dict[str, Any],
    rest_local_phi456: dict[str, dict[str, float]],
    rest_link_frames: dict[str, tuple[float, float, float, float]],
    model: dict[str, Any],
    leg: str,
    *,
    limit_rad: float,
    angles: tuple[str, ...] = LIMITED_REST_ANGLE_KEYS,
    relative_limits_rad: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> bool:
    """True when rest-link-frame phi4/phi5/phi6 stay inside the configured rest window.

    ``angle_diff`` is used so wrap-around near +/-pi is handled correctly.
    """
    current = local_phi456_from_result(leg_result, leg, model, rest_link_frames[leg])
    if relative_limits_rad is not None:
        for key, (lo, hi) in relative_limits_rad[leg].items():
            delta = angle_diff(float(current[key]), float(rest_local_phi456[leg][key]))
            if delta < lo or delta > hi:
                return False
        return True

    for key in angles:
        delta = angle_diff(float(current[key]), float(rest_local_phi456[leg][key]))
        if abs(delta) > limit_rad:
            return False
    return True


def _candidate_leg_result(
    leg: str,
    shared: dict[str, Any],
    model: dict[str, Any],
    P2: tuple[float, float, float],
    P3: tuple[float, float, float],
    selected_cost: float | None,
) -> dict[str, Any]:
    Pb = model["anchors"][leg]
    phi1 = shared["phi1"]
    return {
        "valid": True,
        "fail_reason": None,
        "P2": P2,
        "candidates_P3": [P3],
        "selected_P3": P3,
        "phi1": phi1,
        "phi2": compute_phi2(shared["P1"], P2),
        "phi3": compute_phi3(P2, P3, Pb),
        "phi4": compute_phi4(P3, Pb),
        "phi5": compute_phi5(P3, Pb),
        "phi6": compute_phi6(P3, Pb),
        "selected_cost": selected_cost,
    }


def solve_one_leg_angle_valid(
    leg: str,
    shared: dict[str, Any],
    model: dict[str, Any],
    rest_angles: dict[str, dict[str, float]],
    rest_local_phi456: dict[str, dict[str, float]],
    rest_link_frames: dict[str, tuple[float, float, float, float]],
    restriction: dict[str, Any],
    *,
    limit_rad: float,
    angles: tuple[str, ...],
) -> dict[str, Any]:
    """Select the lowest-cost P3 branch that satisfies the phi456 window.

    The original IK first selected one of two possible P3 branches and only then
    applied the movement limit. For asymmetric limits this can reject a pose even
    though the other physical branch is valid. Here each candidate branch is
    tested against the requested phi4/phi5/phi6 limits first.
    """
    Pb = model["anchors"][leg]
    d2 = model["lengths"]["d2"]
    d3 = model["lengths"]["d3"]
    weights = model["branch_weights"]
    P1 = shared["P1"]
    Pm = shared["Pm"]
    P2 = construct_p2(leg, P1, Pm, model)
    candidates_P3, fail_reason = construct_p3_candidates(P2, Pb, d2, d3)
    if not candidates_P3:
        return {
            "valid": False,
            "fail_reason": fail_reason,
            "P2": P2,
            "candidates_P3": [],
            "selected_P3": None,
            "phi1": shared["phi1"],
            "phi2": None,
            "phi3": None,
            "phi4": None,
            "phi5": None,
            "phi6": None,
            "selected_cost": None,
        }

    valid_scored = []
    for P3 in candidates_P3:
        angles_now = {
            "phi2": compute_phi2(P1, P2),
            "phi3": compute_phi3(P2, P3, Pb),
            "phi4": compute_phi4(P3, Pb),
            "phi5": compute_phi5(P3, Pb),
            "phi6": compute_phi6(P3, Pb),
        }
        candidate = {
            "valid": True,
            "fail_reason": None,
            "P2": P2,
            "candidates_P3": candidates_P3,
            "selected_P3": P3,
            "phi1": shared["phi1"],
            **angles_now,
            "selected_cost": None,
        }
        if phi456_within_rest_limit(
            candidate,
            rest_local_phi456,
            rest_link_frames,
            model,
            leg,
            limit_rad=limit_rad,
            angles=angles,
            relative_limits_rad=restriction["relative_limits_rad"],
        ):
            valid_scored.append((branch_cost(angles_now, rest_angles[leg], weights), P3, angles_now))

    if not valid_scored:
        return {
            "valid": False,
            "fail_reason": "no P3 branch satisfies phi4/phi5/phi6 limits",
            "P2": P2,
            "candidates_P3": candidates_P3,
            "selected_P3": None,
            "phi1": shared["phi1"],
            "phi2": None,
            "phi3": None,
            "phi4": None,
            "phi5": None,
            "phi6": None,
            "selected_cost": None,
        }

    valid_scored.sort(key=lambda item: item[0])
    selected_cost, selected_P3, selected_angles = valid_scored[0]
    return {
        "valid": True,
        "fail_reason": None,
        "P2": P2,
        "candidates_P3": candidates_P3,
        "selected_P3": selected_P3,
        "phi1": shared["phi1"],
        "phi2": selected_angles["phi2"],
        "phi3": selected_angles["phi3"],
        "phi4": selected_angles["phi4"],
        "phi5": selected_angles["phi5"],
        "phi6": selected_angles["phi6"],
        "selected_cost": selected_cost,
    }


def solve_all_legs_angle_valid(
    P1: tuple[float, float, float],
    phi1: float,
    model: dict[str, Any],
    rest_angles: dict[str, dict[str, float]],
    rest_local_phi456: dict[str, dict[str, float]],
    rest_link_frames: dict[str, tuple[float, float, float, float]],
    restriction: dict[str, Any],
    *,
    limit_rad: float,
    angles: tuple[str, ...],
) -> dict[str, Any]:
    shared = compute_shared_geometry(P1, phi1, model)
    return {
        "shared": shared,
        **{
            leg: solve_one_leg_angle_valid(
                leg,
                shared,
                model,
                rest_angles,
                rest_local_phi456,
                rest_link_frames,
                restriction,
                limit_rad=limit_rad,
                angles=angles,
            )
            for leg in LEGS
        },
    }


def link_elevation(P3, Pb) -> float:
    """Elevation angle of the base link vector Pb -> P3."""
    dx = float(P3[0]) - float(Pb[0])
    dy = float(P3[1]) - float(Pb[1])
    dz = float(P3[2]) - float(Pb[2])
    return math.atan2(dz, math.hypot(dx, dy))


def compute_link_elevations(solution: dict[str, Any], model: dict[str, Any]) -> dict[str, float]:
    """Per-leg base-link elevation angles from a valid IK solution."""
    return {
        leg: link_elevation(solution[leg]["selected_P3"], model["anchors"][leg])
        for leg in LEGS
    }


def link_elevation_within_rest_limit(
    leg_result: dict[str, Any],
    rest_link_elevations: dict[str, float],
    model: dict[str, Any],
    leg: str,
    *,
    limit_rad: float,
) -> bool:
    """True when base-link elevation stays within +/-limit from rest."""
    current = link_elevation(leg_result["selected_P3"], model["anchors"][leg])
    return abs(angle_diff(current, rest_link_elevations[leg])) <= limit_rad


def reference_phi2(phi1: float, leg_name: str, model: dict[str, Any]) -> float:
    leg_rot = 0.0 if leg_name == "top" else math.radians(model["leg_rotation_deg"][leg_name])
    return wrap_to_pi(phi1 + leg_rot + math.pi)


def reference_phi3(P2, P3, Pb, d2: float, d3: float) -> float:
    vx = P2[0] - Pb[0]
    vy = P2[1] - Pb[1]
    vz = P2[2] - Pb[2]
    side_opposite_sq = vx * vx + vy * vy + vz * vz
    cos_phi3 = (d2 * d2 + d3 * d3 - side_opposite_sq) / (2.0 * d2 * d3)
    return math.acos(max(-1.0, min(1.0, cos_phi3)))


def _safe_tan(angle: float):
    c = math.cos(angle)
    if abs(c) < _DEGEN_EPS:
        return None
    return math.sin(angle) / c


def predict_phi4(phi5: float, phi6: float):
    t5 = _safe_tan(phi5)
    t6 = _safe_tan(phi6)
    if t5 is None or t6 is None:
        return None
    denom = 1.0 + t5 * t5 + (t5 * t6) ** 2
    if denom <= 0.0:
        return None
    sign_nx = 1.0 if math.cos(phi5) >= 0.0 else -1.0
    nx = sign_nx / math.sqrt(denom)
    ny = nx * t5
    nz = ny * t6
    return math.atan2(nz, nx)


def predict_phi5(phi4: float, phi6: float):
    t4 = _safe_tan(phi4)
    t6 = _safe_tan(phi6)
    if t4 is None or t6 is None or abs(t6) < _DEGEN_EPS:
        return None
    ratio_yx = t4 / t6
    denom = 1.0 + ratio_yx * ratio_yx + t4 * t4
    sign_nx = 1.0 if math.cos(phi4) >= 0.0 else -1.0
    nx = sign_nx / math.sqrt(denom)
    ny = nx * ratio_yx
    return math.atan2(ny, nx)


def predict_phi6(phi4: float, phi5: float):
    t4 = _safe_tan(phi4)
    t5 = _safe_tan(phi5)
    if t4 is None or t5 is None:
        return None
    denom = 1.0 + t5 * t5 + t4 * t4
    sign_nx = 1.0 if math.cos(phi5) >= 0.0 else -1.0
    nx = sign_nx / math.sqrt(denom)
    ny = nx * t5
    nz = nx * t4
    return math.atan2(nz, ny)


def rel_error(computed: float, reference: float) -> float:
    return angle_diff(computed, reference) / max(abs(reference), EPS_REL)


def largest_connected_component(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Largest 6-connected component in a boolean Z/Y/X occupancy grid."""
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int, int]] = []
    z_max, y_max, x_max = mask.shape
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for z0, y0, x0 in np.argwhere(mask):
        z0, y0, x0 = int(z0), int(y0), int(x0)
        if visited[z0, y0, x0]:
            continue
        stack = [(z0, y0, x0)]
        visited[z0, y0, x0] = True
        comp: list[tuple[int, int, int]] = []
        while stack:
            z, y, x = stack.pop()
            comp.append((z, y, x))
            for dz, dy, dx in neighbors:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < z_max and 0 <= ny < y_max and 0 <= nx < x_max:
                    if mask[nz, ny, nx] and not visited[nz, ny, nx]:
                        visited[nz, ny, nx] = True
                        stack.append((nz, ny, nx))
        if len(comp) > len(best):
            best = comp
    return best


def _component_to_xyz(component, xs, ys, zs) -> list[tuple[float, float, float]]:
    return [(float(xs[x]), float(ys[y]), float(zs[z])) for z, y, x in component]


def _summarize_xyz(points: list[tuple[float, float, float]]) -> dict[str, float]:
    if not points:
        return {
            "x_min_mm": np.nan,
            "x_max_mm": np.nan,
            "y_min_mm": np.nan,
            "y_max_mm": np.nan,
            "z_min_mm": np.nan,
            "z_max_mm": np.nan,
            "workspace_width_mm": 0.0,
            "workspace_depth_mm": 0.0,
            "height_span_mm": 0.0,
            "area_proxy_mm2": 0.0,
            "volume_proxy_mm3": 0.0,
            "center_x_mm": np.nan,
            "center_y_mm": np.nan,
            "center_z_mm": np.nan,
        }
    arr = np.asarray(points, dtype=float)
    x_min, y_min, z_min = arr.min(axis=0)
    x_max, y_max, z_max = arr.max(axis=0)
    center_x, center_y, center_z = arr.mean(axis=0)
    width = float(x_max - x_min)
    depth = float(y_max - y_min)
    height = float(z_max - z_min)
    return {
        "x_min_mm": float(x_min),
        "x_max_mm": float(x_max),
        "y_min_mm": float(y_min),
        "y_max_mm": float(y_max),
        "z_min_mm": float(z_min),
        "z_max_mm": float(z_max),
        "workspace_width_mm": width,
        "workspace_depth_mm": depth,
        "height_span_mm": height,
        "area_proxy_mm2": width * depth,
        "volume_proxy_mm3": width * depth * height,
        "center_x_mm": float(center_x),
        "center_y_mm": float(center_y),
        "center_z_mm": float(center_z),
    }


def evaluate_workspace(
    d1_mm: float,
    d2_mm: float,
    d3_mm: float,
    cfg: dict[str, Any],
    *,
    keep_detail: bool = False,
) -> dict[str, Any]:
    """Evaluate one geometry over the deterministic grid.

    The three swept variables are the effective solver lengths directly:
    d1 = platform offset, d2 = SR link length (historical d4),
    d3 = RR link length.
    """
    model = make_model(d1_mm, d2_mm, d3_mm)
    probe = cfg["workspace_probe"]
    xs = np.linspace(-probe["xy_limit_mm"], probe["xy_limit_mm"], probe["xy_points"])
    ys = np.linspace(-probe["xy_limit_mm"], probe["xy_limit_mm"], probe["xy_points"])
    zs = np.linspace(probe["z_min_mm"], probe["z_max_mm"], probe["z_points"])
    phi1 = probe["phi1_rad"]
    restriction = movement_restriction_config(cfg)
    limit_rad = math.radians(restriction["limit_deg"])
    restricted_angles = tuple(restriction["angles"])
    link_elevation_limit_rad = (
        None
        if restriction["link_elevation_limit_deg"] is None
        else math.radians(restriction["link_elevation_limit_deg"])
    )
    rest_angles, rest_p1, rest_solution = compute_rest_angles(phi1, model)
    rest_pose_valid = rest_angles is not None
    rest_link_frames = compute_rest_link_frames(rest_solution, model) if rest_pose_valid else None
    rest_local_phi456 = compute_rest_local_phi456(rest_solution, model, rest_link_frames) if rest_pose_valid else None
    rest_link_elevations = compute_link_elevations(rest_solution, model) if rest_pose_valid else None

    valid_mask = np.zeros((len(zs), len(ys), len(xs)), dtype=bool)
    per_leg_valid_xyz = {leg: [] for leg in LEGS}
    per_leg_invalid_xyz = {leg: [] for leg in LEGS}
    all_valid_xyz: list[tuple[float, float, float]] = []
    any_invalid_xyz: list[tuple[float, float, float]] = []
    angles_per_leg = {leg: {key: [] for key in ANGLE_KEYS} for leg in LEGS}
    rel_err_per_leg = {leg: {key: [] for key in ANGLE_KEYS} for leg in LEGS}

    d2 = model["lengths"]["d2"]
    d3 = model["lengths"]["d3"]
    for zi, z in enumerate(zs):
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                pt = (float(x), float(y), float(z))
                if not rest_pose_valid:
                    result = {leg: {"valid": False} for leg in LEGS}
                else:
                    if restriction["enabled"] and restriction["branch_selection"] == "angle_valid":
                        result = solve_all_legs_angle_valid(
                            pt,
                            phi1,
                            model,
                            rest_angles,
                            rest_local_phi456,
                            rest_link_frames,
                            restriction,
                            limit_rad=limit_rad,
                            angles=restricted_angles,
                        )
                    else:
                        # Use rest_angles for deterministic branch selection, then
                        # apply the requested rest-link-local angle restriction.
                        result = solve_all_legs(pt, phi1, rest_angles=rest_angles, model=model)
                all_legs_valid = True
                for leg in LEGS:
                    leg_result = result[leg]
                    leg_is_valid = bool(leg_result["valid"])
                    if leg_is_valid and restriction["enabled"]:
                        leg_is_valid = phi456_within_rest_limit(
                            leg_result,
                            rest_local_phi456,
                            rest_link_frames,
                            model,
                            leg,
                            limit_rad=limit_rad,
                            angles=restricted_angles,
                            relative_limits_rad=restriction["relative_limits_rad"],
                        )
                    if leg_is_valid and restriction["link_elevation_enabled"]:
                        leg_is_valid = link_elevation_within_rest_limit(
                            leg_result,
                            rest_link_elevations,
                            model,
                            leg,
                            limit_rad=link_elevation_limit_rad,
                        )
                    if leg_is_valid:
                        if keep_detail:
                            per_leg_valid_xyz[leg].append(pt)
                        pb = model["anchors"][leg]
                        local_phi = local_phi456_from_result(leg_result, leg, model, rest_link_frames[leg])
                        plot_angles = {
                            "phi2": float(leg_result["phi2"]),
                            "phi3": float(leg_result["phi3"]),
                            **local_phi,
                        }
                        for key in ANGLE_KEYS:
                            angles_per_leg[leg][key].append(float(plot_angles[key]))
                        refs = {
                            "phi2": reference_phi2(phi1, leg, model),
                            "phi3": reference_phi3(leg_result["P2"], leg_result["selected_P3"], pb, d2, d3),
                            "phi4": predict_phi4(local_phi["phi5"], local_phi["phi6"]),
                            "phi5": predict_phi5(local_phi["phi4"], local_phi["phi6"]),
                            "phi6": predict_phi6(local_phi["phi4"], local_phi["phi5"]),
                        }
                        for key, ref in refs.items():
                            if ref is not None:
                                rel_err_per_leg[leg][key].append(rel_error(float(plot_angles[key]), float(ref)))
                    else:
                        all_legs_valid = False
                        if keep_detail:
                            per_leg_invalid_xyz[leg].append(pt)

                if all_legs_valid:
                    valid_mask[zi, yi, xi] = True
                    if keep_detail:
                        all_valid_xyz.append(pt)
                elif keep_detail:
                    any_invalid_xyz.append(pt)

    component_xyz = _component_to_xyz(largest_connected_component(valid_mask), xs, ys, zs)
    metrics = _summarize_xyz(component_xyz)
    n_total = int(valid_mask.size)
    n_valid = int(valid_mask.sum())
    n_component = int(len(component_xyz))
    largest_component_fraction = float(n_component / n_total)
    acceptance = cfg["acceptance"]
    accepted = (
        metrics["workspace_width_mm"] >= acceptance["min_workspace_width_mm"]
        and metrics["workspace_depth_mm"] >= acceptance["min_workspace_depth_mm"]
    )
    score_parts = workspace_efficiency_score(
        metrics,
        d1_mm=d1_mm,
        d2_mm=d2_mm,
        d3_mm=d3_mm,
        largest_component_fraction=largest_component_fraction,
        accepted=accepted,
    )

    row: dict[str, Any] = {
        "d1_mm": float(d1_mm),
        "d2_mm": float(d2_mm),
        "d3_mm": float(d3_mm),
        "valid_points": n_valid,
        "largest_component_points": n_component,
        "total_points": n_total,
        "valid_fraction": float(n_valid / n_total),
        "largest_component_fraction": largest_component_fraction,
        "movement_restriction_enabled": bool(restriction["enabled"]),
        "restricted_angles": ",".join(restricted_angles),
        "phi456_rest_limit_deg": float(restriction["limit_deg"]),
        "phi456_angle_frame": str(restriction["angle_frame"]),
        "phi456_branch_selection": str(restriction["branch_selection"]),
        "phi456_relative_limits_deg": json.dumps(restriction["relative_limits_deg"]),
        "link_elevation_restriction_enabled": bool(restriction["link_elevation_enabled"]),
        "link_elevation_rest_limit_deg": restriction["link_elevation_limit_deg"],
        "rest_pose_valid": bool(rest_pose_valid),
        "rest_p1_x_mm": float(rest_p1[0]),
        "rest_p1_y_mm": float(rest_p1[1]),
        "rest_p1_z_mm": float(rest_p1[2]),
        **metrics,
        "score_method": "balanced_small_d1_d2_d3_with_soft_d2_greater_than_d3_preference",
        **score_parts,
        "accepted": bool(accepted),
    }
    if keep_detail:
        row.update(
            {
                "model": model,
                "per_leg_valid_xyz": per_leg_valid_xyz,
                "per_leg_invalid_xyz": per_leg_invalid_xyz,
                "all_valid_xyz": all_valid_xyz,
                "any_invalid_xyz": any_invalid_xyz,
                "component_xyz": component_xyz,
                "angles_per_leg": angles_per_leg,
                "rel_err_per_leg": rel_err_per_leg,
            }
        )
    return row


def _xyz_array(points: list[tuple[float, float, float]]) -> np.ndarray:
    return np.asarray(points, dtype=float) if points else np.empty((0, 3), dtype=float)


def _scatter_projection(ax, points: np.ndarray, xi: int, yi: int, *, color: str, label: str, alpha: float, size: float) -> None:
    if points.size:
        ax.scatter(points[:, xi], points[:, yi], s=size, color=color, alpha=alpha, label=label)


def _unique_xy_footprint(points: np.ndarray) -> np.ndarray:
    if not points.size:
        return np.empty((0, 2), dtype=float)
    return np.unique(np.round(points[:, :2], 6), axis=0)


def _xy_orientation_from_points(xy: np.ndarray) -> dict[str, Any]:
    if xy.size == 0:
        return {
            "xy_footprint_points": 0,
            "footprint_center_x_mm": float("nan"),
            "footprint_center_y_mm": float("nan"),
            "footprint_width_x_mm": 0.0,
            "footprint_depth_y_mm": 0.0,
            "footprint_y_over_x_extent_ratio": float("nan"),
            "footprint_major_axis_angle_from_y_deg": float("nan"),
            "footprint_elongation_ratio": float("nan"),
            "footprint_y_axis_symmetry_fraction": float("nan"),
            "footprint_y_axis_bounds_symmetry_error_mm": float("nan"),
            "footprint_shift_description": "no connected workspace",
        }

    center = xy.mean(axis=0)
    x_min = float(xy[:, 0].min())
    x_max = float(xy[:, 0].max())
    y_min = float(xy[:, 1].min())
    y_max = float(xy[:, 1].max())
    x_width = float(xy[:, 0].max() - xy[:, 0].min())
    y_depth = float(xy[:, 1].max() - xy[:, 1].min())
    y_over_x = float(y_depth / x_width) if x_width > _DEGEN_EPS else float("inf")
    bounds_symmetry_error = abs(abs(x_min) - abs(x_max))

    if len(xy) >= 2:
        centered = xy - center
        cov = np.cov(centered.T)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        major = vectors[:, 0]
        angle_from_y = math.degrees(math.atan2(float(major[0]), float(major[1])))
        # Principal axes are bidirectional. Fold to [-90, 90] deg so 0 means
        # north-south and +/-90 means east-west.
        if angle_from_y > 90.0:
            angle_from_y -= 180.0
        elif angle_from_y < -90.0:
            angle_from_y += 180.0
        elongation = (
            float(math.sqrt(max(values[0], 0.0) / max(values[1], _DEGEN_EPS)))
            if values[0] > _DEGEN_EPS
            else 1.0
        )
    else:
        angle_from_y = float("nan")
        elongation = 1.0

    rounded = {(float(x), float(y)) for x, y in np.round(xy, 6)}
    mirrored_count = sum((float(round(-x, 6)), float(y)) in rounded for x, y in rounded)
    symmetry = float(mirrored_count / len(rounded)) if rounded else float("nan")

    if center[1] < -_DEGEN_EPS:
        shift = "south of origin"
    elif center[1] > _DEGEN_EPS:
        shift = "north of origin"
    else:
        shift = "centered on north-south axis"

    return {
        "xy_footprint_points": int(len(xy)),
        "footprint_center_x_mm": float(center[0]),
        "footprint_center_y_mm": float(center[1]),
        "footprint_width_x_mm": x_width,
        "footprint_depth_y_mm": y_depth,
        "footprint_y_over_x_extent_ratio": y_over_x,
        "footprint_major_axis_angle_from_y_deg": float(angle_from_y),
        "footprint_elongation_ratio": float(elongation),
        "footprint_y_axis_symmetry_fraction": symmetry,
        "footprint_y_axis_bounds_symmetry_error_mm": float(bounds_symmetry_error),
        "footprint_shift_description": shift,
    }


def _add_xy_reference_guides(ax) -> None:
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.80, label="Y axis / symmetry")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.text(0.50, 0.98, "North (+Y)", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.text(0.50, 0.02, "South (-Y)", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    ax.text(0.98, 0.50, "East (+X)", transform=ax.transAxes, ha="right", va="center", fontsize=8)
    ax.text(0.02, 0.50, "West (-X)", transform=ax.transAxes, ha="left", va="center", fontsize=8)


def _config_output_dir(row: pd.Series, all_config_dir: Path) -> Path:
    return (
        all_config_dir
        / f"d1_{safe_value(row['d1_mm'])}"
        / f"d2_{safe_value(row['d2_mm'])}"
        / f"d3_{safe_value(row['d3_mm'])}"
    )


def save_workspace_projection_plots(
    detail: dict[str, Any],
    out_dir: Path,
    title: str,
    cfg: dict[str, Any],
    *,
    xy_detail: dict[str, Any] | None = None,
) -> None:
    xy_detail = xy_detail or detail
    all_valid = _xyz_array(xy_detail["all_valid_xyz"])
    any_invalid = _xyz_array(xy_detail["any_invalid_xyz"])

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    if any_invalid.size:
        invalid_xy = _unique_xy_footprint(any_invalid)
        ax.scatter(
            invalid_xy[:, 0],
            invalid_xy[:, 1],
            s=16,
            color="tab:red",
            alpha=0.22,
            label="invalid / not all three legs valid",
        )
    if all_valid.size:
        all_valid_xy, xy_counts = np.unique(np.round(all_valid[:, :2], 6), axis=0, return_counts=True)
        sizes = 28.0 + 9.0 * np.sqrt(xy_counts.astype(float))
        scatter = ax.scatter(
            all_valid_xy[:, 0],
            all_valid_xy[:, 1],
            s=sizes,
            c=xy_counts,
            cmap="viridis",
            alpha=0.90,
            edgecolors="black",
            linewidths=0.25,
            label="valid for all three legs",
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
        cbar.set_label("valid Z samples at this XY point")
        ax.text(
            0.02,
            0.98,
            f"{len(all_valid):,} valid 3D points\n{len(all_valid_xy):,} unique XY points",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.82),
        )
    else:
        ax.text(0.5, 0.5, "no all-legs-valid XY points", ha="center", va="center", transform=ax.transAxes)
    for leg, anchor in detail["model"]["anchors"].items():
        ax.scatter(
            [anchor[0]],
            [anchor[1]],
            marker="^",
            s=110,
            color="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        ax.annotate(
            leg,
            xy=(anchor[0], anchor[1]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
        )
    ax.set_title(f"Angle limits - ({detail['d1_mm']:.1f},{detail['d2_mm']:.1f},{detail['d3_mm']:.1f})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "workspace_projection_xy.png", dpi=cfg["plot_generation"]["plot_dpi"])
    plt.close(fig)

    projections = [
        (0, 2, "X (mm)", "Z (mm)", "workspace_projection_xz.png", "XZ projection"),
        (1, 2, "Y (mm)", "Z (mm)", "workspace_projection_yz.png", "YZ projection"),
    ]
    all_valid = _xyz_array(detail["all_valid_xyz"])
    any_invalid = _xyz_array(detail["any_invalid_xyz"])
    component = _xyz_array(detail["component_xyz"])
    for xi, yi, xlabel, ylabel, filename, projection_title in projections:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for ax, leg in zip(axes[:3], LEGS):
            valid = _xyz_array(detail["per_leg_valid_xyz"][leg])
            invalid = _xyz_array(detail["per_leg_invalid_xyz"][leg])
            _scatter_projection(ax, invalid, xi, yi, color="tab:red", label="invalid", alpha=0.12, size=4)
            _scatter_projection(ax, valid, xi, yi, color="tab:green", label="valid", alpha=0.55, size=7)
            anchor = detail["model"]["anchors"][leg]
            ax.scatter([anchor[xi]], [anchor[yi]], marker="^", s=90, color="black", label="anchor")
            ax.set_title(f"{leg} leg")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=7)
            ax.set_aspect("equal", adjustable="box")
        ax = axes[3]
        _scatter_projection(ax, any_invalid, xi, yi, color="tab:red", label="not all-valid", alpha=0.10, size=4)
        _scatter_projection(ax, all_valid, xi, yi, color="tab:green", label="all legs valid", alpha=0.45, size=7)
        _scatter_projection(ax, component, xi, yi, color="tab:blue", label="largest connected", alpha=0.75, size=10)
        ax.set_title("all legs")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        fig.suptitle(f"{title.split(' | ', 1)[0]} — {projection_title}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=cfg["plot_generation"]["plot_dpi"])
        plt.close(fig)


def save_workspace_xy_footprint_orientation_plot(detail: dict[str, Any], out_dir: Path, title: str, cfg: dict[str, Any]) -> dict[str, Any]:
    all_valid_xy = _unique_xy_footprint(_xyz_array(detail["all_valid_xyz"]))
    component_xy = _unique_xy_footprint(_xyz_array(detail["component_xyz"]))
    metrics = _xy_orientation_from_points(component_xy)

    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    if all_valid_xy.size:
        ax.scatter(
            all_valid_xy[:, 0],
            all_valid_xy[:, 1],
            s=20,
            color="tab:green",
            alpha=0.22,
            label="all valid XY samples",
        )
    if component_xy.size:
        ax.scatter(
            component_xy[:, 0],
            component_xy[:, 1],
            s=34,
            color="tab:blue",
            alpha=0.82,
            label="largest connected footprint",
        )
        x_sym = max(abs(float(component_xy[:, 0].min())), abs(float(component_xy[:, 0].max())))
        y_min = float(component_xy[:, 1].min())
        y_max = float(component_xy[:, 1].max())
        ax.plot(
            [-x_sym, x_sym, x_sym, -x_sym, -x_sym],
            [y_min, y_min, y_max, y_max, y_min],
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label="Y-axis-symmetric bounds",
        )

        center = np.array([metrics["footprint_center_x_mm"], metrics["footprint_center_y_mm"]], dtype=float)
        ax.scatter([center[0]], [center[1]], marker="*", s=230, color="gold", edgecolor="black", zorder=5, label="footprint center")
        ax.annotate(
            "center shift",
            xy=(center[0], center[1]),
            xytext=(center[0], center[1] - 1.2),
            arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.3),
            ha="center",
            va="top",
            fontsize=8,
            color="black",
        )

        if len(component_xy) >= 2 and not math.isnan(metrics["footprint_major_axis_angle_from_y_deg"]):
            centered = component_xy - center
            cov = np.cov(centered.T)
            values, vectors = np.linalg.eigh(cov)
            order = np.argsort(values)[::-1]
            vectors = vectors[:, order]
            major = vectors[:, 0]
            minor = vectors[:, 1]
            length = 0.55 * max(metrics["footprint_width_x_mm"], metrics["footprint_depth_y_mm"], 1.0)
            ax.plot(
                [center[0] - major[0] * length, center[0] + major[0] * length],
                [center[1] - major[1] * length, center[1] + major[1] * length],
                color="tab:purple",
                linewidth=2.2,
                label="major footprint axis",
            )
            ax.plot(
                [center[0] - minor[0] * length * 0.55, center[0] + minor[0] * length * 0.55],
                [center[1] - minor[1] * length * 0.55, center[1] + minor[1] * length * 0.55],
                color="tab:purple",
                linewidth=1.1,
                linestyle=":",
                label="minor footprint axis",
            )

    _add_xy_reference_guides(ax)
    ax.scatter([0.0], [0.0], marker="+", s=90, color="black", linewidths=1.4, label="origin")
    ax.set_xlabel("X (mm), east-west")
    ax.set_ylabel("Y (mm), north-south")
    ax.set_title("XY footprint orientation and Y-axis symmetry")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    if component_xy.size:
        x_pad = max(1.0, 0.16 * max(metrics["footprint_width_x_mm"], 1.0))
        y_pad = max(1.0, 0.16 * max(metrics["footprint_depth_y_mm"], 1.0))
        ax.set_xlim(float(component_xy[:, 0].min() - x_pad), float(component_xy[:, 0].max() + x_pad))
        ax.set_ylim(float(component_xy[:, 1].min() - y_pad), float(component_xy[:, 1].max() + y_pad))

    y_over_x = metrics["footprint_y_over_x_extent_ratio"]
    orientation_text = (
        f"X width: {metrics['footprint_width_x_mm']:.2f} mm\n"
        f"Y depth: {metrics['footprint_depth_y_mm']:.2f} mm\n"
        f"Y/X extent: {y_over_x:.2f}\n"
        f"Center: ({metrics['footprint_center_x_mm']:+.2f}, {metrics['footprint_center_y_mm']:+.2f}) mm\n"
        f"Shift: {metrics['footprint_shift_description']}\n"
        f"Major-axis angle from +Y: {metrics['footprint_major_axis_angle_from_y_deg']:+.1f} deg\n"
        f"Elongation ratio: {metrics['footprint_elongation_ratio']:.2f}\n"
        f"Y-axis bounds error: {metrics['footprint_y_axis_bounds_symmetry_error_mm']:.2f} mm\n"
        f"Mirror sample symmetry: {100.0 * metrics['footprint_y_axis_symmetry_fraction']:.1f}%"
    )
    ax.text(
        0.02,
        0.98,
        orientation_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.86),
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title.split(" | ", 1)[0], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "workspace_xy_footprint_orientation.png", dpi=cfg["plot_generation"]["plot_dpi"])
    plt.close(fig)
    return metrics


def save_workspace_3d_plot(detail: dict[str, Any], out_dir: Path, title: str, cfg: dict[str, Any]) -> None:
    probe = cfg["workspace_probe"]
    fig = plt.figure(figsize=(20, 5))
    for i, leg in enumerate(LEGS, start=1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        valid = _xyz_array(detail["per_leg_valid_xyz"][leg])
        invalid = _xyz_array(detail["per_leg_invalid_xyz"][leg])
        if invalid.size:
            ax.scatter(invalid[:, 0], invalid[:, 1], invalid[:, 2], s=3, color="tab:red", alpha=0.03, label="invalid")
        if valid.size:
            ax.scatter(valid[:, 0], valid[:, 1], valid[:, 2], s=5, color="tab:green", alpha=0.35, label="valid")
        anchor = detail["model"]["anchors"][leg]
        ax.scatter([anchor[0]], [anchor[1]], [anchor[2]], marker="^", s=80, color="black", label="anchor")
        _style_3d_axis(ax, f"{leg} leg", probe)
        ax.legend(loc="best", fontsize=7)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    all_valid = _xyz_array(detail["all_valid_xyz"])
    any_invalid = _xyz_array(detail["any_invalid_xyz"])
    component = _xyz_array(detail["component_xyz"])
    if any_invalid.size:
        ax.scatter(any_invalid[:, 0], any_invalid[:, 1], any_invalid[:, 2], s=3, color="tab:red", alpha=0.02, label="not all-valid")
    if all_valid.size:
        ax.scatter(all_valid[:, 0], all_valid[:, 1], all_valid[:, 2], s=5, color="tab:green", alpha=0.22, label="all legs valid")
    if component.size:
        ax.scatter(component[:, 0], component[:, 1], component[:, 2], s=8, color="tab:blue", alpha=0.70, label="largest connected")
    _style_3d_axis(ax, "all legs", probe)
    ax.legend(loc="best", fontsize=7)
    fig.suptitle(f"{title} — 3D workspace validity")
    fig.tight_layout()
    fig.savefig(out_dir / "workspace_3d_per_leg_and_all.png", dpi=cfg["plot_generation"]["plot_dpi"])
    plt.close(fig)


def _style_3d_axis(ax, title: str, probe: dict[str, Any]) -> None:
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim(-probe["xy_limit_mm"], probe["xy_limit_mm"])
    ax.set_ylim(-probe["xy_limit_mm"], probe["xy_limit_mm"])
    ax.set_zlim(probe["z_min_mm"], probe["z_max_mm"])


def save_angle_distribution_plot(detail: dict[str, Any], out_dir: Path, title: str, cfg: dict[str, Any]) -> None:
    fig, axes = plt.subplots(len(LEGS), len(ANGLE_KEYS), figsize=(3.2 * len(ANGLE_KEYS), 2.7 * len(LEGS)), squeeze=False)
    for i, leg in enumerate(LEGS):
        for j, key in enumerate(ANGLE_KEYS):
            ax = axes[i, j]
            data = np.degrees(np.asarray(detail["angles_per_leg"][leg][key], dtype=float))
            if data.size:
                ax.hist(data, bins=40, color="tab:blue", alpha=0.75)
            else:
                ax.text(0.5, 0.5, "no valid samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{leg} {key}")
            ax.set_xlabel(f"{key} (deg)")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title} — angle distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "angle_distributions.png", dpi=cfg["plot_generation"]["plot_dpi"])
    plt.close(fig)


def save_angle_deviation_plot(detail: dict[str, Any], out_dir: Path, title: str, cfg: dict[str, Any]) -> None:
    leg_colors = {"top": "tab:blue", "right": "tab:orange", "left": "tab:green"}
    fig, axes = plt.subplots(len(LEGS), len(ANGLE_KEYS), figsize=(3.2 * len(ANGLE_KEYS), 2.7 * len(LEGS)), squeeze=False)
    for i, leg in enumerate(LEGS):
        for j, key in enumerate(ANGLE_KEYS):
            ax = axes[i, j]
            arr = np.asarray(detail["angles_per_leg"][leg][key], dtype=float)
            if arr.size:
                mean_angle = float(np.mean(arr))
                dev = np.array([angle_diff(float(a), mean_angle) for a in arr], dtype=float)
                ax.hist(np.degrees(dev), bins=40, color=leg_colors[leg], alpha=0.75)
                ax.axvline(0.0, color="black", linewidth=0.8)
                ax.set_title(f"{leg} {key}; mean={math.degrees(mean_angle):+.1f}°")
            else:
                ax.text(0.5, 0.5, "no valid samples", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{leg} {key}")
            ax.set_xlabel(f"{key} - mean({key}) (deg)")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title} — deviation from sample mean")
    fig.tight_layout()
    fig.savefig(out_dir / "angle_deviation_from_mean.png", dpi=cfg["plot_generation"]["plot_dpi"])
    plt.close(fig)


def save_relative_error_plots(detail: dict[str, Any], out_dir: Path, title: str, cfg: dict[str, Any]) -> None:
    leg_colors = {"top": "tab:blue", "right": "tab:orange", "left": "tab:green"}
    positions, data, colors = [], [], []
    group_width = 0.8
    for j, key in enumerate(ANGLE_KEYS):
        for i, leg in enumerate(LEGS):
            arr = np.asarray(detail["rel_err_per_leg"][leg][key], dtype=float)
            if arr.size:
                positions.append(j + (i - (len(LEGS) - 1) / 2.0) * (group_width / len(LEGS)))
                data.append(arr)
                colors.append(leg_colors[leg])
    fig, ax = plt.subplots(figsize=(13, 4.8))
    if data:
        bp = ax.boxplot(data, positions=positions, widths=group_width / len(LEGS) * 0.9, patch_artist=True, showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.35))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(ANGLE_KEYS)))
    ax.set_xticklabels(ANGLE_KEYS)
    ax.set_xlabel("angle")
    ax.set_ylabel("signed relative error")
    ax.set_title(f"{title} — signed relative error per angle")
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=leg_colors[leg], alpha=0.6, edgecolor="black") for leg in LEGS]
    ax.legend(handles, list(LEGS), title="leg", loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "relative_error_boxplot.png", dpi=cfg["plot_generation"]["plot_dpi"])
    plt.close(fig)

    fig, (ax_mean, ax_max) = plt.subplots(1, 2, figsize=(13, 4.5))
    x_base = np.arange(len(ANGLE_KEYS))
    width = 0.8 / len(LEGS)
    for i, leg in enumerate(LEGS):
        means, stds, maxabs = [], [], []
        for key in ANGLE_KEYS:
            arr = np.asarray(detail["rel_err_per_leg"][leg][key], dtype=float)
            if arr.size:
                means.append(float(arr.mean()))
                stds.append(float(arr.std()))
                maxabs.append(float(np.max(np.abs(arr))))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                maxabs.append(np.nan)
        offset = (i - (len(LEGS) - 1) / 2.0) * width
        ax_mean.bar(x_base + offset, means, width=width, yerr=stds, capsize=3, label=leg)
        ax_max.bar(x_base + offset, maxabs, width=width, label=leg)
    ax_mean.axhline(0.0, color="black", linewidth=0.8)
    ax_mean.set_xticks(x_base)
    ax_mean.set_xticklabels(ANGLE_KEYS)
    ax_mean.set_title("Mean signed relative error ± 1 std")
    ax_mean.set_xlabel("angle")
    ax_mean.set_ylabel("signed relative error")
    ax_mean.set_yscale("symlog", linthresh=1e-16)
    ax_mean.grid(True, which="both", axis="y", alpha=0.3)
    ax_mean.legend(title="leg")
    ax_max.set_xticks(x_base)
    ax_max.set_xticklabels(ANGLE_KEYS)
    ax_max.set_title("Max |relative error|")
    ax_max.set_xlabel("angle")
    ax_max.set_ylabel("max |relative error|")
    ax_max.set_yscale("log")
    ax_max.grid(True, which="both", axis="y", alpha=0.3)
    ax_max.legend(title="leg")
    fig.suptitle(f"{title} — relative error summary")
    fig.tight_layout()
    fig.savefig(out_dir / "relative_error_mean_max.png", dpi=cfg["plot_generation"]["plot_dpi"])
    plt.close(fig)


def save_full_plot_set(row: pd.Series, cfg: dict[str, Any], all_config_dir: Path) -> Path:
    out_dir = _config_output_dir(row, all_config_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{config_label(row)} | d1/platform={row['d1_mm']:.2f} mm, "
        f"d2/SR={row['d2_mm']:.2f} mm, d3/RR={row['d3_mm']:.2f} mm | "
        f"{movement_restriction_label(cfg)}, "
        f"link elev +/-{cfg.get('movement_restriction', {}).get('link_elevation_limit_deg', 'off')} deg from rest"
    )
    detail = evaluate_workspace(row["d1_mm"], row["d2_mm"], row["d3_mm"], cfg, keep_detail=True)
    xy_detail = None
    xy_plot_points = cfg.get("plot_generation", {}).get("xy_projection_points")
    if xy_plot_points is not None:
        xy_plot_points = int(xy_plot_points)
        if xy_plot_points != int(cfg["workspace_probe"]["xy_points"]):
            xy_cfg = json.loads(json.dumps(cfg))
            xy_cfg["workspace_probe"]["xy_points"] = xy_plot_points
            xy_detail = evaluate_workspace(row["d1_mm"], row["d2_mm"], row["d3_mm"], xy_cfg, keep_detail=True)
    save_workspace_projection_plots(detail, out_dir, title, cfg, xy_detail=xy_detail)
    save_workspace_3d_plot(detail, out_dir, title, cfg)
    save_angle_distribution_plot(detail, out_dir, title, cfg)
    save_angle_deviation_plot(detail, out_dir, title, cfg)
    save_relative_error_plots(detail, out_dir, title, cfg)
    summary = {k: v for k, v in detail.items() if k not in {
        "model", "per_leg_valid_xyz", "per_leg_invalid_xyz", "all_valid_xyz",
        "any_invalid_xyz", "component_xyz", "angles_per_leg", "rel_err_per_leg",
    }}
    summary["label"] = config_label(row)
    summary["plot_files"] = [
        "workspace_projection_xy.png",
        "workspace_projection_xz.png",
        "workspace_projection_yz.png",
        "workspace_3d_per_leg_and_all.png",
        "angle_distributions.png",
        "angle_deviation_from_mean.png",
        "relative_error_boxplot.png",
        "relative_error_mean_max.png",
    ]
    (out_dir / "workspace_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def save_overview_plots(
    results: pd.DataFrame,
    current_row: pd.Series,
    suggestions: pd.DataFrame,
    cfg: dict[str, Any],
    overview_dir: Path,
) -> None:
    overview_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg["plot_generation"]["plot_dpi"]
    acceptance = cfg["acceptance"]
    best_by_link = results.groupby(["d2_mm", "d3_mm"], as_index=False)["score"].max()
    pivot = best_by_link.pivot(index="d2_mm", columns="d3_mm", values="score")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("d3 / RR link length (mm)")
    ax.set_ylabel("d2 / SR link length (mm)")
    ax.set_title("Best compact score with soft d2>d3 preference (max over d1)")
    fig.colorbar(im, ax=ax, label="compact XY score × smooth d2>d3 preference")
    fig.tight_layout()
    fig.savefig(overview_dir / "heatmap_best_score_by_d2_d3.png", dpi=dpi)
    plt.close(fig)

    accepted_results = results[results["accepted"]].copy()
    if not accepted_results.empty and "max_dimension_mm" in accepted_results.columns:
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(
            accepted_results["max_dimension_mm"],
            accepted_results["workspace_area_mm2"],
            c=accepted_results["score"],
            s=28,
            cmap="viridis",
            alpha=0.75,
        )
        ax.set_xlabel("Largest single dimension max(d1,d2,d3) (mm)")
        ax.set_ylabel("Connected XY workspace area (mm^2)")
        ax.set_title("Accepted designs: compactness vs XY workspace area")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="compact score with soft d2>d3 preference")
        fig.tight_layout()
        fig.savefig(overview_dir / "compactness_vs_workspace_area.png", dpi=dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = np.where(results["accepted"], "tab:green", "tab:red")
    ax.scatter(results["workspace_width_mm"], results["workspace_depth_mm"], c=colors, alpha=0.45, s=24)
    ax.axvline(acceptance["min_workspace_width_mm"], color="black", linestyle="--", linewidth=1)
    ax.axhline(acceptance["min_workspace_depth_mm"], color="black", linestyle="--", linewidth=1)
    # Mark the configured reference design for comparison.
    ax.scatter(
        [current_row["workspace_width_mm"]],
        [current_row["workspace_depth_mm"]],
        marker="*",
        s=420,
        facecolor="gold",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
        label=(
            f"reference design "
            f"(d1={current_row['d1_mm']:.2f}, d2={current_row['d2_mm']:.2f}, "
            f"d3={current_row['d3_mm']:.2f})"
        ),
    )
    ax.set_xlabel("Largest-component workspace width X (mm)")
    ax.set_ylabel("Largest-component workspace depth Y (mm)")
    ax.set_title("Workspace width/depth acceptance — reference design vs sweep")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(overview_dir / "workspace_width_depth_acceptance.png", dpi=dpi)
    plt.close(fig)

    compare = pd.concat(
        [
            pd.DataFrame([current_row]).assign(label="reference"),
            suggestions.head(10).copy().assign(label=lambda d: [f"suggested_{i+1}" for i in range(len(d))]),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title, threshold in zip(
        axes,
        ["workspace_width_mm", "workspace_depth_mm"],
        ["Width X", "Depth Y"],
        [
            acceptance["min_workspace_width_mm"],
            acceptance["min_workspace_depth_mm"],
        ],
    ):
        ax.bar(compare["label"], compare[metric], color=["tab:blue"] + ["tab:green"] * (len(compare) - 1))
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("mm")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Reference design vs top suggestions")
    fig.tight_layout()
    fig.savefig(overview_dir / "current_vs_suggestions_metrics.png", dpi=dpi)
    plt.close(fig)

    compact_columns = {
        "workspace_area_mm2",
        "max_dimension_mm",
        "total_envelope_mm",
        "d2_minus_d3_mm",
        "d2_gt_d3_preference_factor",
        "score",
    }
    if compact_columns.issubset(compare.columns):
        labels = compare["label"].tolist()
        x = np.arange(len(compare))
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))

        ax = axes[0, 0]
        bar_w = 0.25
        ax.bar(x - bar_w, compare["d1_mm"], width=bar_w, label="d1")
        ax.bar(x, compare["d2_mm"], width=bar_w, label="d2")
        ax.bar(x + bar_w, compare["d3_mm"], width=bar_w, label="d3")
        ax.set_title("Geometry values")
        ax.set_ylabel("mm")
        ax.legend()

        ax = axes[0, 1]
        ax.bar(x - bar_w / 2, compare["max_dimension_mm"], width=bar_w, label="max(d1,d2,d3)")
        ax.bar(x + bar_w / 2, compare["total_envelope_mm"], width=bar_w, label="d1+d2+d3")
        ax.set_title("Envelope compactness")
        ax.set_ylabel("mm")
        ax.legend()

        ax = axes[1, 0]
        ax.bar(x, compare["workspace_area_mm2"], color="tab:green")
        ax.set_title("Connected XY workspace area")
        ax.set_ylabel("mm²")

        ax = axes[1, 1]
        ax.bar(x - bar_w / 2, compare["score"], width=bar_w, label="final score")
        ax.bar(
            x + bar_w / 2,
            compare["d2_gt_d3_preference_factor"],
            width=bar_w,
            label="d2>d3 factor",
        )
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title("Score and d2>d3 smoothness preference")
        ax.legend()

        for ax in axes.ravel():
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha="right")
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle("Reference design vs top suggestions - compact score details")
        fig.tight_layout()
        fig.savefig(overview_dir / "reference_vs_suggestions_compact_score.png", dpi=dpi)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Crash-safe checkpointing + parallel execution helpers
# ---------------------------------------------------------------------------
# These are additive utilities used by the notebook to make the sweep and the
# full-plot stage (a) parallel across CPU cores and (b) resumable after a crash
# at single-configuration granularity. None of the numerical/plotting functions
# above are modified, so outputs stay identical to the original notebook.


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text so a crash can never leave a half-written file.

    Writes to a sibling ``*.tmp`` file then ``os.replace`` (atomic rename on the
    same volume). A reader therefore only ever sees the complete old file or the
    complete new file, never a truncated one.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=2))


def partial_result_label(d1_mm: float, d2_mm: float, d3_mm: float) -> str:
    """Filesystem-safe label uniquely identifying one swept configuration."""
    return (
        f"d1_{safe_value(d1_mm)}__d2_{safe_value(d2_mm)}"
        f"__d3_{safe_value(d3_mm)}"
    )


def list_run_dirs(results_root: Path) -> list[Path]:
    """All ``run_*`` directories under ``results_root``, sorted oldest-first.

    Names are timestamp-prefixed (``run_YYYYmmdd_HHMMSS_...``) so lexical sort is
    chronological; the last element is the most recent run.
    """
    results_root = Path(results_root)
    if not results_root.exists():
        return []
    return sorted(p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("run_"))


def resolve_continue_run_dir(results_root: Path, continue_run_id: str | None = None) -> Path:
    """Pick the run directory to resume.

    ``continue_run_id`` selects a specific ``run_...`` folder; ``None`` selects
    the most recent run under ``results_root``.
    """
    results_root = Path(results_root)
    if continue_run_id:
        run_dir = results_root / continue_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run to continue does not exist: {run_dir}")
        return run_dir
    runs = list_run_dirs(results_root)
    if not runs:
        raise FileNotFoundError(f"No existing run_* directories under {results_root} to continue.")
    return runs[-1]


# --- Parallel sweep worker -------------------------------------------------
# Defined at module level (not in the notebook) so it is importable by spawned
# worker processes on Windows. Each worker computes one configuration and
# atomically checkpoints it before returning, so completed work survives a crash.

def evaluate_and_checkpoint(task: tuple) -> str:
    d1_mm, d2_mm, d3_mm, cfg, partial_dir = task
    row = evaluate_workspace(d1_mm, d2_mm, d3_mm, cfg, keep_detail=False)
    out_path = Path(partial_dir) / f"{partial_result_label(d1_mm, d2_mm, d3_mm)}.json"
    atomic_write_json(out_path, row)
    return out_path.name


# --- Parallel plot worker --------------------------------------------------

def render_and_checkpoint(task: tuple) -> str:
    """Render the full plot set for one config and mark it done atomically.

    The ``.done`` marker is written only after ``save_full_plot_set`` has
    finished writing all configured PNGs plus the summary JSON, so a config folder
    is treated as complete only when it truly is. A folder interrupted
    mid-render simply has no marker and is regenerated on resume.
    """
    row_dict, cfg, all_config_dir = task
    row = pd.Series(row_dict)
    out_dir = save_full_plot_set(row, cfg, Path(all_config_dir))
    atomic_write_text(out_dir / ".done", "ok\n")
    return str(out_dir)


def plot_config_done(row: Any, all_config_dir: Path) -> bool:
    """True if this config's full plot set has already been completed."""
    out_dir = _config_output_dir(row, Path(all_config_dir))
    return (out_dir / ".done").exists()
