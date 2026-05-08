"""
Sweep / sanity test for the unified IK solver.

1. Samples many random targets (P1, phi1) under the user's constraints:
     - z is fixed
     - |x|, |y| are bounded by the model's own x,y extent
       (read from the leg anchors)
2. Calls solve_all_legs for every sample.
3. Classifies each sample PER LEG (valid / invalid) and also as
   "all three legs valid".
4. For every valid leg it computes a *relative error per angle*
   against an INDEPENDENT reference:
       phi2   -> closed form phi1 + leg_rotation + pi
       phi3   -> law of cosines on the triangle P2-P3-Pb
       phi4/5/6 -> recomputed directly from (P3 - Pb)
   Relative error = |computed - reference| / max(|reference|, eps).
5. Prints a text summary and draws:
       - workspace validity map per leg + global
       - angle histograms per leg
       - relative error per sample (log scatter) per angle per leg
       - mean / max relative error bar chart per angle per leg
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from unified_ik_starter import (
    solve_all_legs,
    default_model,
    wrap_to_pi,
    angle_diff,
)


# =========================================
# CONFIG
# =========================================
Z_MIN = 0.0             # z is now sampled uniformly in [Z_MIN, Z_MAX]
Z_MAX = 10.0
N_SAMPLES = 3000
RNG_SEED = 0

LEGS = ("top", "right", "left")
ANGLE_KEYS = ("phi2", "phi3", "phi4", "phi5", "phi6")

EPS_REL = 1e-6   # floor to avoid division by zero in relative error

# --- 3D rotation export (drop straight into a PowerPoint slide) ---
SAVE_3D_ANIMATION = True
ANIM_FORMAT       = "mp4"           # "mp4" (needs ffmpeg) or "gif" (Pillow)
ANIM_FILENAME     = "workspace_3d_rotation"   # extension added automatically
ANIM_N_FRAMES     = 180             # 180 frames @ 30 fps -> 6 s full turn
ANIM_FPS          = 30
ANIM_ELEV         = 20              # tilt (deg)


# =========================================
# BOUNDS FROM THE MODEL ITSELF
# =========================================
# User rule: "the x,y can't be bigger than the model x,y itself".
# Read the anchors and take the max |x|, |y| across them.
model = default_model()
anchors = model["anchors"]

x_extent = max(abs(a[0]) for a in anchors.values())
y_extent = max(abs(a[1]) for a in anchors.values())

# anchors can have x == 0 for every leg: fall back to y_extent so we
# still scan a meaningful x range.
if x_extent < 1e-9:
    x_extent = y_extent

X_LIMIT = x_extent
Y_LIMIT = y_extent


# =========================================
# INTERACTIVE OVERRIDES  (press Enter to keep the default)
# =========================================
def _ask_range(label, lo_default, hi_default):
    """Ask for 'lo hi' on one line. Blank -> keep defaults."""
    prompt = (f"  {label} range  [default {lo_default:g} {hi_default:g}] "
              f"(enter 'lo hi' or blank): ")
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return lo_default, hi_default
        if raw == "":
            return lo_default, hi_default
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("    -> need two numbers, e.g.  -5 5")
            continue
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            print("    -> could not parse numbers, try again.")
            continue
        if hi <= lo:
            print("    -> high must be greater than low, try again.")
            continue
        return lo, hi

def _ask_int(label, default):
    prompt = f"  {label}  [default {default}]: "
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return default
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("    -> need an integer, try again.")
            continue
        if val <= 0:
            print("    -> must be > 0, try again.")
            continue
        return val


print("\nSampling configuration  (press Enter to accept default):")
X_MIN, X_MAX = _ask_range("X", -X_LIMIT, X_LIMIT)
Y_MIN, Y_MAX = _ask_range("Y", -Y_LIMIT, Y_LIMIT)
Z_MIN, Z_MAX = _ask_range("Z", Z_MIN,   Z_MAX)
N_SAMPLES    = _ask_int  ("N samples", N_SAMPLES)

# keep |x|/|y| bounds (used elsewhere, e.g. 3D plot limits) in sync
# with whatever the user picked, even for asymmetric ranges.
X_LIMIT = max(abs(X_MIN), abs(X_MAX))
Y_LIMIT = max(abs(Y_MIN), abs(Y_MAX))

print(f"\nSampling box : x in [{X_MIN:g}, {X_MAX:g}], "
      f"y in [{Y_MIN:g}, {Y_MAX:g}], "
      f"z in [{Z_MIN:g}, {Z_MAX:g}]")
print(f"Samples      : {N_SAMPLES}")

# geometric feasibility band per leg: |z - z_anchor| must be <= d3.
d3 = model["lengths"]["d3"]
print("\nPer-leg reachable z band  (|z - z_anchor| <= d3):")
for leg in LEGS:
    zp = anchors[leg][2]
    z_lo = max(Z_MIN, zp - d3)
    z_hi = min(Z_MAX, zp + d3)
    reachable = z_hi > z_lo
    print(f"  {leg:5s}: z in [{zp - d3:6.3f}, {zp + d3:6.3f}],"
          f"  overlap with sample range: "
          f"{'[%.3f, %.3f]' % (z_lo, z_hi) if reachable else 'empty'}")


# =========================================
# STORAGE
# =========================================
all_points = []                             # (x, y, z) for every sample
per_leg_valid_mask = {leg: [] for leg in LEGS}   # bool list
per_leg_fail_reasons = {leg: {} for leg in LEGS} # reason -> count

# per-leg list of (x, y, z) when that leg solved / failed
per_leg_valid_xyz   = {leg: [] for leg in LEGS}
per_leg_invalid_xyz = {leg: [] for leg in LEGS}

# angle distributions per leg (only samples where leg is valid)
angles_per_leg = {leg: {k: [] for k in ANGLE_KEYS} for leg in LEGS}

# relative error per angle per leg
rel_err_per_leg = {leg: {k: [] for k in ANGLE_KEYS} for leg in LEGS}

# IK solver latency per sample  [seconds]
ik_latencies = []


# =========================================
# INDEPENDENT REFERENCE ANGLES
# =========================================
def reference_phi2(phi1: float, leg_name: str) -> float:
    """
    Closed-form expected phi2.

        P2 - P1 = d1 * (cos(phi1 + leg_rot), sin(phi1 + leg_rot))
        P1 - P2 -> angle = phi1 + leg_rot + pi
    """
    if leg_name == "top":
        leg_rot = 0.0
    else:
        leg_rot = math.radians(model["leg_rotation_deg"][leg_name])
    return wrap_to_pi(phi1 + leg_rot + math.pi)


def reference_phi3(P2, P3, Pb, d2, d3) -> float:
    """
    Law of cosines on triangle P2-P3-Pb at vertex P3.
    """
    vx = P2[0] - Pb[0]
    vy = P2[1] - Pb[1]
    vz = P2[2] - Pb[2]
    side_opposite_sq = vx * vx + vy * vy + vz * vz

    cos_phi3 = (d2 * d2 + d3 * d3 - side_opposite_sq) / (2.0 * d2 * d3)
    cos_phi3 = max(-1.0, min(1.0, cos_phi3))
    return math.acos(cos_phi3)


# ---------------------------------------------------------------
# Independent cross-predictions for phi4 / phi5 / phi6.
#
# Each angle is re-derived from the OTHER two, using the fact that
# (phi4, phi5, phi6) are projections of the same unit direction n:
#   phi4 = atan2(nz, nx)
#   phi5 = atan2(ny, nx)
#   phi6 = atan2(nz, ny)
# so any two of them determine the third. We reconstruct n from one
# pair and predict the remaining angle. This is a genuine
# self-consistency check (not a tautology like reading n from P3).
#
# Samples where cos(phi_i) is numerically ~0 for the pair being
# used are flagged as degenerate and skipped for that angle.
# ---------------------------------------------------------------
_DEGEN_EPS = 1e-6


def _safe_tan(angle: float):
    c = math.cos(angle)
    if abs(c) < _DEGEN_EPS:
        return None
    return math.sin(angle) / c


def predict_phi4(phi5: float, phi6: float):
    """
    Predict phi4 from phi5 and phi6 via reconstruction of n.

    From phi6 = atan2(nz, ny) and phi5 = atan2(ny, nx), we can solve:
        ny / nx = tan(phi5)
        nz / ny = tan(phi6)
    With nx^2 + ny^2 + nz^2 = 1. Sign of nx is sign(cos(phi5)).
    """
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
    """
    Predict phi5 from phi4 and phi6.
    """
    t4 = _safe_tan(phi4)
    t6 = _safe_tan(phi6)
    if t4 is None or t6 is None:
        return None
    # nz/nx = tan(phi4), nz/ny = tan(phi6) -> ny = nz/tan(phi6) = nx*tan(phi4)/tan(phi6)
    if abs(t6) < _DEGEN_EPS:
        return None
    ratio_yx = t4 / t6          # ny / nx
    denom = 1.0 + ratio_yx * ratio_yx + t4 * t4
    sign_nx = 1.0 if math.cos(phi4) >= 0.0 else -1.0
    nx = sign_nx / math.sqrt(denom)
    ny = nx * ratio_yx
    return math.atan2(ny, nx)


def predict_phi6(phi4: float, phi5: float):
    """
    Predict phi6 from phi4 and phi5.
    """
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
    """
    Signed relative error: (computed - reference) / |reference|.
    Uses shortest-arc signed difference so wrap-around does not fake
    large errors near +/-pi. Result can be positive or negative.
    """
    diff = angle_diff(computed, reference)   # signed, in (-pi, pi]
    denom = max(abs(reference), EPS_REL)
    return diff / denom


# =========================================
# SAMPLING LOOP
# =========================================
rng = np.random.default_rng(RNG_SEED)
d2 = model["lengths"]["d2"]

for _ in range(N_SAMPLES):
    x = float(rng.uniform(X_MIN, X_MAX))
    y = float(rng.uniform(Y_MIN, Y_MAX))
    z = float(rng.uniform(Z_MIN, Z_MAX))
    P1 = (x, y, z)
    phi1 = float(rng.uniform(-math.pi, math.pi))

    all_points.append((x, y, z))
    _t0 = time.perf_counter()
    result = solve_all_legs(P1, phi1, model=model)
    ik_latencies.append(time.perf_counter() - _t0)

    for leg in LEGS:
        r = result[leg]
        if r["valid"]:
            per_leg_valid_mask[leg].append(True)
            per_leg_valid_xyz[leg].append((x, y, z))
            Pb = anchors[leg]

            for k in ANGLE_KEYS:
                angles_per_leg[leg][k].append(r[k])

            ref_phi2 = reference_phi2(phi1, leg)
            ref_phi3 = reference_phi3(r["P2"], r["selected_P3"], Pb, d2, d3)
            # cross-predictions for phi4/5/6 (may be None near degeneracies)
            ref_phi4 = predict_phi4(r["phi5"], r["phi6"])
            ref_phi5 = predict_phi5(r["phi4"], r["phi6"])
            ref_phi6 = predict_phi6(r["phi4"], r["phi5"])

            rel_err_per_leg[leg]["phi2"].append(rel_error(r["phi2"], ref_phi2))
            rel_err_per_leg[leg]["phi3"].append(rel_error(r["phi3"], ref_phi3))
            if ref_phi4 is not None:
                rel_err_per_leg[leg]["phi4"].append(rel_error(r["phi4"], ref_phi4))
            if ref_phi5 is not None:
                rel_err_per_leg[leg]["phi5"].append(rel_error(r["phi5"], ref_phi5))
            if ref_phi6 is not None:
                rel_err_per_leg[leg]["phi6"].append(rel_error(r["phi6"], ref_phi6))
        else:
            per_leg_valid_mask[leg].append(False)
            per_leg_invalid_xyz[leg].append((x, y, z))
            reason = r["fail_reason"] or "unknown"
            per_leg_fail_reasons[leg][reason] = \
                per_leg_fail_reasons[leg].get(reason, 0) + 1


# =========================================
# SUMMARY
# =========================================
total = N_SAMPLES
n_all_three_valid = sum(
    1 for i in range(total)
    if all(per_leg_valid_mask[leg][i] for leg in LEGS)
)

print("\n===== SUMMARY =====")
print(f"Total samples : {total}")
print(f"All 3 legs valid : {n_all_three_valid}"
      f"   ({n_all_three_valid/total:.3f})")

print("\nPer-leg valid / invalid:")
print(f"  {'leg':<6s}  {'valid':>6s}  {'invalid':>8s}  {'success':>8s}")
for leg in LEGS:
    n_v = sum(per_leg_valid_mask[leg])
    n_i = total - n_v
    print(f"  {leg:<6s}  {n_v:>6d}  {n_i:>8d}  {n_v/total:>8.3f}")

lat_all = np.asarray(ik_latencies) * 1e6   # seconds -> microseconds
success_mask = np.array(
    [all(per_leg_valid_mask[leg][i] for leg in LEGS) for i in range(total)],
    dtype=bool,
)
lat_success = lat_all[success_mask]
lat_failure = lat_all[~success_mask]


def _print_latency_block(label: str, lat_us: np.ndarray) -> None:
    if lat_us.size == 0:
        print(f"  {label:<18s}: (no samples)")
        return
    total_ms = float(lat_us.sum()) / 1e3
    throughput = lat_us.size / (total_ms / 1e3) if total_ms > 0 else float("nan")
    print(f"  {label}  ({lat_us.size} samples, total {total_ms:.3f} ms, "
          f"~{throughput:.1f} solves/s)")
    print(f"     mean   : {lat_us.mean():10.3f} us")
    print(f"     std    : {lat_us.std():10.3f} us")
    print(f"     min    : {lat_us.min():10.3f} us")
    print(f"     median : {float(np.median(lat_us)):10.3f} us")
    print(f"     p90    : {float(np.percentile(lat_us, 90)):10.3f} us")
    print(f"     p99    : {float(np.percentile(lat_us, 99)):10.3f} us")
    print(f"     max    : {lat_us.max():10.3f} us")


if lat_all.size > 0:
    print("\nIK solver latency  (solve_all_legs per call, all 3 legs):")
    _print_latency_block("ALL samples        ", lat_all)
    _print_latency_block("SUCCESS (all valid)", lat_success)
    _print_latency_block("FAILURE (>=1 fail) ", lat_failure)

print("\nFailure reasons per leg:")
for leg in LEGS:
    reasons = per_leg_fail_reasons[leg]
    if not reasons:
        print(f"  {leg}: (none)")
        continue
    print(f"  {leg}:")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"     {count:6d}   {reason}")

def _fmt_row(leg, func):
    row = "  " + f"{leg:<9s}"
    for k in ANGLE_KEYS:
        arr = np.asarray(rel_err_per_leg[leg][k])
        if arr.size == 0:
            row += "  " + f"{'n/a':>11s}"
        else:
            row += "  " + f"{func(arr):+11.3e}"
    return row

header = "  leg       " + "  ".join(f"{k:>11s}" for k in ANGLE_KEYS)

print("\nMean signed relative error per angle per leg:")
print(header)
for leg in LEGS:
    print(_fmt_row(leg, np.mean))

print("\nStd  of relative error per angle per leg:")
print(header)
for leg in LEGS:
    print(_fmt_row(leg, np.std))

print("\nMax |relative error| per angle per leg:")
print(header)
for leg in LEGS:
    print(_fmt_row(leg, lambda a: np.max(np.abs(a))))


# precompute the "all 3 legs valid / not" split on the full (x,y,z) cloud
all_valid_xyz = []
any_invalid_xyz = []
for i, pt in enumerate(all_points):
    if all(per_leg_valid_mask[leg][i] for leg in LEGS):
        all_valid_xyz.append(pt)
    else:
        any_invalid_xyz.append(pt)


# =========================================
# PLOT 1 - 2D WORKSPACE VALIDITY PROJECTIONS (XY, YZ, XZ)
# =========================================
# One figure per projection. Each figure has 4 subplots:
#   top leg, right leg, left leg, and all-3-legs combined.
# Green = valid (solver succeeded), red = invalid.
# Black triangles are leg anchors in the same projection.
_AX_NAME = ("x", "y", "z")
# user-chosen range per axis index (0=x, 1=y, 2=z).
_AX_RANGE = ((X_MIN, X_MAX), (Y_MIN, Y_MAX), (Z_MIN, Z_MAX))

def plot_validity_projection(ax_idx_x: int, ax_idx_y: int, suptitle: str):
    ax_names = (f"P1.{_AX_NAME[ax_idx_x]}  [model units]",
                f"P1.{_AX_NAME[ax_idx_y]}  [model units]")
    x_lo, x_hi = _AX_RANGE[ax_idx_x]
    y_lo, y_hi = _AX_RANGE[ax_idx_y]
    fig, axes = plt.subplots(1, len(LEGS) + 1,
                             figsize=(4.2 * (len(LEGS) + 1), 4.6))

    for i, leg in enumerate(LEGS):
        ax = axes[i]
        inv = per_leg_invalid_xyz[leg]
        val = per_leg_valid_xyz[leg]
        if inv:
            coords = np.asarray(inv)
            ax.scatter(coords[:, ax_idx_x], coords[:, ax_idx_y],
                       s=5, color="tab:red", alpha=0.35,
                       label=f"invalid ({len(inv)})")
        if val:
            coords = np.asarray(val)
            ax.scatter(coords[:, ax_idx_x], coords[:, ax_idx_y],
                       s=5, color="tab:green", alpha=0.6,
                       label=f"valid ({len(val)})")
        a = anchors[leg]
        ax.scatter([a[ax_idx_x]], [a[ax_idx_y]],
                   marker="^", s=140, color="black", zorder=5)
        ax.annotate(leg, (a[ax_idx_x], a[ax_idx_y]),
                    textcoords="offset points", xytext=(6, 6))
        ax.set_title(f"{leg} leg  "
                     f"(anchor @ "
                     f"{_AX_NAME[ax_idx_x]}={a[ax_idx_x]:.2f}, "
                     f"{_AX_NAME[ax_idx_y]}={a[ax_idx_y]:.2f})")
        ax.set_xlabel(ax_names[0])
        ax.set_ylabel(ax_names[1])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[-1]
    if any_invalid_xyz:
        coords = np.asarray(any_invalid_xyz)
        ax.scatter(coords[:, ax_idx_x], coords[:, ax_idx_y],
                   s=5, color="tab:red", alpha=0.25,
                   label=f"not all-valid ({len(any_invalid_xyz)})")
    if all_valid_xyz:
        coords = np.asarray(all_valid_xyz)
        ax.scatter(coords[:, ax_idx_x], coords[:, ax_idx_y],
                   s=5, color="tab:green", alpha=0.85,
                   label=f"all 3 legs ({len(all_valid_xyz)})")
    for name, a in anchors.items():
        ax.scatter([a[ax_idx_x]], [a[ax_idx_y]],
                   marker="^", s=140, color="black", zorder=5)
        ax.annotate(name, (a[ax_idx_x], a[ax_idx_y]),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_title(f"All 3 legs valid  (z in [{Z_MIN:g}, {Z_MAX:g}])")
    ax.set_xlabel(ax_names[0])
    ax.set_ylabel(ax_names[1])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


plot_validity_projection(0, 1, "Workspace validity  (XY projection, top-down)")
plot_validity_projection(1, 2, "Workspace validity  (YZ projection, side view)")
plot_validity_projection(0, 2, "Workspace validity  (XZ projection, front view)")


# =========================================
# PLOT 1B - 3D WORKSPACE VALIDITY (full 3D scatter)
# =========================================
def _style_3d(ax, title):
    ax.set_title(title)
    ax.set_xlabel("P1.x")
    ax.set_ylabel("P1.y")
    ax.set_zlabel("P1.z")
    # use the same sampling box for all subplots so shapes are comparable
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(Z_MIN, Z_MAX)

def _plot_anchors_3d(ax, only_leg=None):
    for name, a in anchors.items():
        if only_leg is not None and name != only_leg:
            continue
        ax.scatter([a[0]], [a[1]], [a[2]],
                   marker="^", s=120, color="black", depthshade=False)
        ax.text(a[0], a[1], a[2], f"  {name}", color="black")

def _build_workspace_3d_figure():
    """
    Build the full 3D workspace-validity figure (top / right / left /
    all-3-legs). Returns (fig, axes) so callers can either display it
    as-is or animate it.
    """
    _fig = plt.figure(figsize=(4.8 * (len(LEGS) + 1), 5.2))
    _axes = []
    for i, leg in enumerate(LEGS):
        ax = _fig.add_subplot(1, len(LEGS) + 1, i + 1, projection="3d")
        inv = per_leg_invalid_xyz[leg]
        val = per_leg_valid_xyz[leg]
        if inv:
            ix, iy, iz = zip(*inv)
            ax.scatter(ix, iy, iz, s=4, color="tab:red", alpha=0.12,
                       label=f"invalid ({len(inv)})")
        if val:
            vx, vy, vz = zip(*val)
            ax.scatter(vx, vy, vz, s=6, color="tab:green", alpha=0.55,
                       label=f"valid ({len(val)})")
        _plot_anchors_3d(ax, only_leg=leg)
        _style_3d(ax, f"{leg} leg  workspace")
        ax.legend(loc="upper right", fontsize=8)
        _axes.append(ax)

    ax = _fig.add_subplot(1, len(LEGS) + 1, len(LEGS) + 1, projection="3d")
    if any_invalid_xyz:
        ix, iy, iz = zip(*any_invalid_xyz)
        ax.scatter(ix, iy, iz, s=3, color="tab:red", alpha=0.06,
                   label=f"not all-valid ({len(any_invalid_xyz)})")
    if all_valid_xyz:
        vx, vy, vz = zip(*all_valid_xyz)
        ax.scatter(vx, vy, vz, s=8, color="tab:green", alpha=0.85,
                   label=f"all 3 legs ({len(all_valid_xyz)})")
    _plot_anchors_3d(ax)
    _style_3d(ax, "All 3 legs valid  workspace")
    ax.legend(loc="upper right", fontsize=8)
    _axes.append(ax)

    _fig.suptitle("Workspace validity  (3D)")
    _fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig, _axes


# Original static 3D figure (unchanged, displayed by plt.show()).
fig_3d_static, _ = _build_workspace_3d_figure()


# =========================================
# PLOT 1B.anim - SEPARATE ROTATING 3D FIGURE, EXPORT TO VIDEO / GIF
# =========================================
# PowerPoint accepts .mp4 (Insert > Video > This Device) and .gif
# (Insert > Picture). MP4 looks crisper; GIF needs no extra deps.
# This is a SECOND figure so the static one above is not disturbed.
if SAVE_3D_ANIMATION:
    import matplotlib.animation as _anim
    import os as _os

    fig_3d_anim, axes_3d_anim = _build_workspace_3d_figure()
    fig_3d_anim.suptitle("Workspace validity  (3D, rotating)")

    def _update_rotation(frame_idx):
        azim = 360.0 * frame_idx / ANIM_N_FRAMES
        for _ax in axes_3d_anim:
            _ax.view_init(elev=ANIM_ELEV, azim=azim)
        return []

    _ani = _anim.FuncAnimation(
        fig_3d_anim, _update_rotation,
        frames=ANIM_N_FRAMES, interval=1000.0 / ANIM_FPS, blit=False,
    )

    _fmt = ANIM_FORMAT.lower()
    _out_path = f"{ANIM_FILENAME}.{_fmt}"
    try:
        if _fmt == "mp4":
            writer = _anim.FFMpegWriter(fps=ANIM_FPS, bitrate=4000)
            _ani.save(_out_path, writer=writer, dpi=140)
        elif _fmt == "gif":
            writer = _anim.PillowWriter(fps=ANIM_FPS)
            _ani.save(_out_path, writer=writer, dpi=100)
        else:
            raise ValueError(f"Unknown ANIM_FORMAT: {ANIM_FORMAT!r}")
        print(f"\n3D rotation animation saved: {_os.path.abspath(_out_path)}")
        print("  -> In PowerPoint: Insert > Video > This Device (for mp4),")
        print("     or Insert > Picture (for gif).")
    except Exception as _e:
        print(f"\n[warn] could not save 3D animation as {_fmt}: {_e}")
        if _fmt == "mp4":
            print("       mp4 needs ffmpeg on PATH. Try ANIM_FORMAT = \"gif\".")


# =========================================
# PLOT 2 - ANGLE DISTRIBUTIONS PER LEG
# =========================================
fig, axes = plt.subplots(len(LEGS), len(ANGLE_KEYS),
                         figsize=(3.0 * len(ANGLE_KEYS),
                                  2.6 * len(LEGS)),
                         squeeze=False)

for i, leg in enumerate(LEGS):
    for j, k in enumerate(ANGLE_KEYS):
        ax = axes[i, j]
        data = np.degrees(np.asarray(angles_per_leg[leg][k]))
        if data.size > 0:
            ax.hist(data, bins=40, color="tab:blue", alpha=0.75)
        else:
            ax.text(0.5, 0.5, "no valid samples",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{leg} {k}")
        ax.set_xlabel(f"{k}  [deg]")
        ax.set_ylabel("sample count")
        ax.grid(True, alpha=0.3)

fig.suptitle("Angle distributions per leg")
fig.tight_layout(rect=[0, 0, 1, 0.96])


# =========================================
# PLOT 3 - SIGNED RELATIVE ERROR BOXPLOT (compact summary)
# =========================================
# One box per (leg, angle). Shows median, IQR, whiskers, outliers
# for the signed relative error. Much more readable than the
# old per-sample scatter; same information, compressed.
fig, ax = plt.subplots(figsize=(12, 4.5))

positions = []
data = []
labels = []
leg_colors = {"top": "tab:blue", "right": "tab:orange", "left": "tab:green"}
box_colors = []

group_width = 0.8
legs_per_group = len(LEGS)
for j, k in enumerate(ANGLE_KEYS):
    for i, leg in enumerate(LEGS):
        arr = np.asarray(rel_err_per_leg[leg][k])
        if arr.size == 0:
            continue
        pos = j + (i - (legs_per_group - 1) / 2.0) * (group_width / legs_per_group)
        positions.append(pos)
        data.append(arr)
        labels.append(f"{leg}\n{k}")
        box_colors.append(leg_colors[leg])

if data:
    bp = ax.boxplot(data, positions=positions,
                    widths=group_width / legs_per_group * 0.9,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

ax.axhline(0.0, color="black", lw=0.8)
ax.set_xticks(np.arange(len(ANGLE_KEYS)))
ax.set_xticklabels(ANGLE_KEYS)
ax.set_xlabel("angle")
ax.set_ylabel("signed relative error  (computed vs. independent reference)")
ax.set_title("Signed relative error per angle  (box = IQR, whiskers = range)")
# symlog y-axis: handles both signs + wide dynamic range (10^-16 .. 10^-12)
# linthresh = 1e-16 is floating-point dust; anything above it is on log.
ax.set_yscale("symlog", linthresh=1e-16)
ax.grid(True, which="both", axis="y", alpha=0.3)

legend_handles = [plt.Rectangle((0, 0), 1, 1,
                                facecolor=leg_colors[leg], alpha=0.6,
                                edgecolor="black")
                  for leg in LEGS]
ax.legend(legend_handles, list(LEGS), title="leg", loc="best")
fig.tight_layout()


# =========================================
# PLOT 3B - DEVIATION FROM SAMPLE MEAN PER ANGLE
# =========================================
# For each (leg, angle) we compute mean_i = mean of phi_i over all
# valid samples, then plot (phi_i - mean_i). This shows the SPREAD
# of the angle itself across the workspace (not the solver error).
fig, axes = plt.subplots(len(LEGS), len(ANGLE_KEYS),
                         figsize=(3.0 * len(ANGLE_KEYS),
                                  2.6 * len(LEGS)),
                         squeeze=False)

for i, leg in enumerate(LEGS):
    for j, k in enumerate(ANGLE_KEYS):
        ax = axes[i, j]
        arr_rad = np.asarray(angles_per_leg[leg][k])
        if arr_rad.size > 0:
            mean_angle = float(np.mean(arr_rad))
            # use angle_diff so wrap-around near +/-pi is handled cleanly
            dev = np.array([angle_diff(a, mean_angle) for a in arr_rad])
            dev_deg = np.degrees(dev)
            ax.hist(dev_deg, bins=40, color=leg_colors[leg], alpha=0.75)
            ax.axvline(0.0, color="black", lw=0.8)
            ax.set_title(
                f"{leg} {k}   mean = {math.degrees(mean_angle):+.1f} deg"
            )
        else:
            ax.text(0.5, 0.5, "no valid samples",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{leg} {k}")
        ax.set_xlabel(f"{k} - mean({k})  [deg]")
        ax.set_ylabel("sample count")
        ax.grid(True, alpha=0.3)

fig.suptitle("Deviation from sample mean per angle")
fig.tight_layout(rect=[0, 0, 1, 0.96])


# =========================================
# PLOT 4 - MEAN / MAX RELATIVE ERROR BAR CHART
# =========================================
fig, (ax_mean, ax_max) = plt.subplots(1, 2, figsize=(12, 4.5))

n_legs = len(LEGS)
n_angles = len(ANGLE_KEYS)
width = 0.8 / n_legs
x_base = np.arange(n_angles)

for i, leg in enumerate(LEGS):
    means = []
    stds = []
    maxabs = []
    for k in ANGLE_KEYS:
        arr = rel_err_per_leg[leg][k]
        if arr:
            a = np.asarray(arr)
            means.append(float(a.mean()))
            stds.append(float(a.std()))
            maxabs.append(float(np.max(np.abs(a))))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            maxabs.append(np.nan)
    offset = (i - (n_legs - 1) / 2.0) * width
    # left plot: signed mean with +/- std error bars (shows positive and
    # negative direction of the error).
    ax_mean.bar(x_base + offset, means, width=width,
                yerr=stds, capsize=3, label=leg)
    # right plot: max |error|, log y (always positive).
    ax_max.bar(x_base + offset, maxabs, width=width, label=leg)

ax_mean.axhline(0.0, color="black", lw=0.8)
ax_mean.set_xticks(x_base)
ax_mean.set_xticklabels(ANGLE_KEYS)
ax_mean.set_title("Signed mean relative error  (+/- 1 std)")
ax_mean.set_xlabel("angle")
ax_mean.set_ylabel("signed relative error   [symlog]")
ax_mean.set_yscale("symlog", linthresh=1e-16)
ax_mean.grid(True, which="both", axis="y", alpha=0.3)
ax_mean.legend(title="leg")

ax_max.set_yscale("log")
ax_max.set_xticks(x_base)
ax_max.set_xticklabels(ANGLE_KEYS)
ax_max.set_title("Max |relative error|")
ax_max.set_xlabel("angle")
ax_max.set_ylabel("max |relative error|   [log scale]")
ax_max.grid(True, which="both", axis="y", alpha=0.3)
ax_max.legend(title="leg")

fig.suptitle("Relative error per angle  (summary)")
fig.tight_layout(rect=[0, 0, 1, 0.95])


plt.show()
