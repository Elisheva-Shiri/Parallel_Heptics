from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kinematics.unified_ik_starter import angle_diff, default_model, solve_all_legs, wrap_to_pi


LEGS = ("top", "right", "left")
ANGLE_KEYS = ("phi2", "phi3", "phi4", "phi5", "phi6")
EPS_REL = 1e-6
_DEGEN_EPS = 1e-6


def inclusive_range(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    return np.round(np.array([start + i * step for i in range(count + 1)]), 6)


def safe_value(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def config_label(row: pd.Series) -> str:
    return (
        f"rr_{safe_value(row['rr_mm'])}__sr_{safe_value(row['sr_mm'])}"
        f"__platform_{safe_value(row['platform_radius_mm'])}"
    )


def make_model(rr_mm: float, sr_mm: float, platform_radius_mm: float, cfg: dict[str, Any]) -> dict[str, Any]:
    """Scale the effective IK lengths around the current/chosen physical design.

    The anchor/base geometry is copied unchanged, so the base remains fixed.
    """
    model = default_model()
    base = default_model()
    current = cfg["current_values_mm"]
    model["anchors"] = dict(base["anchors"])
    model["lengths"] = dict(base["lengths"])
    model["lengths"]["d1"] = base["lengths"]["d1"] * (
        platform_radius_mm / current["platform_radius"]
    )
    model["lengths"]["d2"] = base["lengths"]["d2"] * (sr_mm / current["sr"])
    model["lengths"]["d3"] = base["lengths"]["d3"] * (rr_mm / current["rr"])
    return model


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
    rr_mm: float,
    sr_mm: float,
    platform_radius_mm: float,
    cfg: dict[str, Any],
    *,
    keep_detail: bool = False,
) -> dict[str, Any]:
    """Evaluate one geometry over the deterministic grid."""
    model = make_model(rr_mm, sr_mm, platform_radius_mm, cfg)
    probe = cfg["workspace_probe"]
    xs = np.linspace(-probe["xy_limit_mm"], probe["xy_limit_mm"], probe["xy_points"])
    ys = np.linspace(-probe["xy_limit_mm"], probe["xy_limit_mm"], probe["xy_points"])
    zs = np.linspace(probe["z_min_mm"], probe["z_max_mm"], probe["z_points"])
    phi1 = probe["phi1_rad"]

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
                result = solve_all_legs(pt, phi1, model=model)
                all_legs_valid = True
                for leg in LEGS:
                    leg_result = result[leg]
                    if leg_result["valid"]:
                        if keep_detail:
                            per_leg_valid_xyz[leg].append(pt)
                        for key in ANGLE_KEYS:
                            angles_per_leg[leg][key].append(float(leg_result[key]))
                        pb = model["anchors"][leg]
                        refs = {
                            "phi2": reference_phi2(phi1, leg, model),
                            "phi3": reference_phi3(leg_result["P2"], leg_result["selected_P3"], pb, d2, d3),
                            "phi4": predict_phi4(leg_result["phi5"], leg_result["phi6"]),
                            "phi5": predict_phi5(leg_result["phi4"], leg_result["phi6"]),
                            "phi6": predict_phi6(leg_result["phi4"], leg_result["phi5"]),
                        }
                        for key, ref in refs.items():
                            if ref is not None:
                                rel_err_per_leg[leg][key].append(rel_error(float(leg_result[key]), float(ref)))
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
    acceptance = cfg["acceptance"]
    accepted = (
        metrics["workspace_width_mm"] >= acceptance["min_workspace_width_mm"]
        and metrics["workspace_depth_mm"] >= acceptance["min_workspace_depth_mm"]
        and metrics["height_span_mm"] > acceptance["min_height_span_mm"]
    )
    score = metrics["volume_proxy_mm3"] + 0.1 * n_component - 0.01 * platform_radius_mm
    if not accepted:
        score *= 0.1

    row: dict[str, Any] = {
        "rr_mm": float(rr_mm),
        "sr_mm": float(sr_mm),
        "platform_radius_mm": float(platform_radius_mm),
        "d1_effective_platform_length": float(model["lengths"]["d1"]),
        "d2_effective_sr_length": float(model["lengths"]["d2"]),
        "d3_effective_rr_length": float(model["lengths"]["d3"]),
        "valid_points": n_valid,
        "largest_component_points": n_component,
        "total_points": n_total,
        "valid_fraction": float(n_valid / n_total),
        "largest_component_fraction": float(n_component / n_total),
        **metrics,
        "accepted": bool(accepted),
        "score": float(score),
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


def _config_output_dir(row: pd.Series, all_config_dir: Path) -> Path:
    return (
        all_config_dir
        / f"rr_{safe_value(row['rr_mm'])}"
        / f"sr_{safe_value(row['sr_mm'])}"
        / f"platform_{safe_value(row['platform_radius_mm'])}"
    )


def save_workspace_projection_plots(detail: dict[str, Any], out_dir: Path, title: str, cfg: dict[str, Any]) -> None:
    projections = [
        (0, 1, "X (mm)", "Y (mm)", "workspace_projection_xy.png", "XY projection"),
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
        fig.suptitle(f"{title} — {projection_title}")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=cfg["plot_generation"]["plot_dpi"])
        plt.close(fig)


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
        f"{config_label(row)} | RR={row['rr_mm']:.1f} mm, "
        f"SR={row['sr_mm']:.1f} mm, platform={row['platform_radius_mm']:.1f} mm"
    )
    detail = evaluate_workspace(row["rr_mm"], row["sr_mm"], row["platform_radius_mm"], cfg, keep_detail=True)
    save_workspace_projection_plots(detail, out_dir, title, cfg)
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
    best_by_rr_sr = results.groupby(["rr_mm", "sr_mm"], as_index=False)["score"].max()
    pivot = best_by_rr_sr.pivot(index="sr_mm", columns="rr_mm", values="score")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("RR height (mm)")
    ax.set_ylabel("SR height (mm)")
    ax.set_title("Best score per RR/SR pair (max over platform radius)")
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(overview_dir / "heatmap_best_score_by_rr_sr.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = np.where(results["accepted"], "tab:green", "tab:red")
    ax.scatter(results["workspace_width_mm"], results["workspace_depth_mm"], c=colors, alpha=0.45, s=24)
    ax.axvline(acceptance["min_workspace_width_mm"], color="black", linestyle="--", linewidth=1)
    ax.axhline(acceptance["min_workspace_depth_mm"], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Largest-component workspace width X (mm)")
    ax.set_ylabel("Largest-component workspace depth Y (mm)")
    ax.set_title("Workspace width/depth acceptance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(overview_dir / "workspace_width_depth_acceptance.png", dpi=dpi)
    plt.close(fig)

    compare = pd.concat(
        [
            pd.DataFrame([current_row]).assign(label="current"),
            suggestions.head(10).copy().assign(label=lambda d: [f"suggested_{i+1}" for i in range(len(d))]),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, metric, title, threshold in zip(
        axes,
        ["workspace_width_mm", "workspace_depth_mm", "height_span_mm"],
        ["Width X", "Depth Y", "Height span Z"],
        [
            acceptance["min_workspace_width_mm"],
            acceptance["min_workspace_depth_mm"],
            acceptance["min_height_span_mm"],
        ],
    ):
        ax.bar(compare["label"], compare[metric], color=["tab:blue"] + ["tab:green"] * (len(compare) - 1))
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("mm")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Current design vs top suggestions")
    fig.tight_layout()
    fig.savefig(overview_dir / "current_vs_suggestions_metrics.png", dpi=dpi)
    plt.close(fig)
