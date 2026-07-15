"""Generate thesis validation figures for the movement-strategy reconstruction check.

Runs the same closed-loop round trip used in `movement_strategy_simulation.ipynb`:
a canonical commanded path (line out + full circle + return) is pushed through the
real `MotorController` for each strategy; the resulting wire-length commands are
decoded back to XY by least-squares trilateration against the shared mechanism
anchor triangle (`unified_ik_starter.default_model()`), and reconstruction error /
trilateration residual are measured.

Outputs (PNG) to analysis/movement_strategy_results/figures/:
  1. reconstruction_paths_by_strategy.png  - commanded vs reconstructed, 4 strategies
  2. wire_residual_by_strategy.png         - residual RMSE per step, 4 strategies
  3. ik_triangle_mismatch_before_after.png - IK circle: mismatched vs shared triangle
Also prints the numeric summary used in the thesis text.
"""
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from consts import EDGE_THRESHOLD, TOP_HEIGHT, TOP_WIDTH
from haptic_mapping import map_object_displacement_to_tactor
from kinematics import unified_ik_starter as ik
from motor_controller import HandOrientation, MotorController, MotorSetId, MovementStrategy

OUT = os.path.join(REPO, "analysis", "movement_strategy_results", "figures")
os.makedirs(OUT, exist_ok=True)

MOTOR_SPACING = 1000.0
THRESHOLD = 1.25
N_LINE, N_CIRCLE, N_RETURN = 40, 220, 40

STRATS = [
    (MovementStrategy.CARDINAL, "cardinal"),
    (MovementStrategy.CARDINAL_DIAGONAL, "cardinal_diagonal"),
    (MovementStrategy.FREE_FORM, "free_form"),
    (MovementStrategy.IK, "ik"),
]


def shared_anchor_triangle(motor_spacing):
    """2-D mechanism anchor triangle (top, right, left), scaled to controller units."""
    a = ik.default_model()["anchors"]
    base = math.hypot(a["right"][0] - a["left"][0], a["right"][1] - a["left"][1])
    s = motor_spacing / base
    return np.array([a[leg][:2] for leg in ("top", "right", "left")], float) * s


def flat_legacy_triangle(motor_spacing):
    """The pre-fix fictional flat triangle that caused the reconstruction ellipse."""
    h = motor_spacing * math.sqrt(3) / 3
    return np.array([(0.0, 2 * h / 3), (motor_spacing / 2, -h / 3), (-motor_spacing / 2, -h / 3)], float)


def build_pattern():
    pts, seg = [], []
    hw = TOP_WIDTH / 2
    ox = hw * 0.5
    for t in np.linspace(0, 1, N_LINE + 1):
        pts.append((ox * t, 0.0)); seg.append("out")
    for a in np.linspace(0, -2 * math.pi, N_CIRCLE + 1)[1:]:
        pts.append((ox * math.cos(a), ox * math.sin(a))); seg.append("circle")
    for t in np.linspace(0, 1, N_RETURN + 1)[1:]:
        pts.append((ox * (1 - t), 0.0)); seg.append("return")
    return pts, seg, ox


def trilaterate(anchors, lengths):
    p1, r1 = anchors[0], lengths[0]
    A, b = [], []
    for i in (1, 2):
        xi, yi = anchors[i]
        A.append([2 * (xi - p1[0]), 2 * (yi - p1[1])])
        b.append((r1 ** 2 - lengths[i] ** 2) - (p1[0] ** 2 - xi ** 2) - (p1[1] ** 2 - yi ** 2))
    sol, *_ = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)
    x, y = float(sol[0]), float(sol[1])
    pred = np.linalg.norm(anchors - np.array([x, y]), axis=1)
    res = float(np.sqrt(np.mean((pred - lengths) ** 2)))
    return x, y, res


def simulate(strategy, anchors):
    """Return commanded XY, reconstructed XY, residual per step, wire-issue mask."""
    c = MotorController(movement_strategy=strategy, top_width=TOP_WIDTH, top_height=TOP_HEIGHT,
                        edge_threshold=EDGE_THRESHOLD, motor_spacing=MOTOR_SPACING, move_factor=1.0,
                        diagonal_threshold=0.5, hand_orientation=HandOrientation.NOT_MIRRORED)
    init = np.linalg.norm(anchors - np.array([0.0, 0.0]), axis=1)
    pos = {0: 0, 1: 0, 2: 0}
    cmd, rec, res, issue, lengths_log = [], [], [], [], []
    pattern, _, _ = build_pattern()
    for ox, oy in pattern:
        tx, ty = map_object_displacement_to_tactor(obj_x=ox, obj_y=oy, oppose_motion=True)
        for m in c.calculate_motor_movements(motor_set_id=MotorSetId.MOTORS_0_2, stiffness_value=1,
                                             obj_x=tx, obj_y=ty, motors_enabled=True, reset_to_origin=False):
            if m.index in pos:
                pos[m.index] = m.pos
        L = init + np.array([pos[0], pos[1], pos[2]], float)
        lengths_log.append(L.copy())
        cmd.append((tx, ty))
        if np.any(L <= 0):
            rec.append((np.nan, np.nan)); res.append(np.inf); issue.append(True); continue
        x, y, r = trilaterate(anchors, L)
        rec.append((x, y)); res.append(r); issue.append(r > THRESHOLD)
    return (np.array(cmd), np.array(rec), np.array(res), np.array(issue), np.array(lengths_log))


def circle_ratio(rec):
    seg = rec[N_LINE:N_LINE + N_CIRCLE]
    seg = seg[~np.isnan(seg[:, 0])]
    xs = seg[:, 0].max() - seg[:, 0].min()
    ys = seg[:, 1].max() - seg[:, 1].min()
    return xs, ys, (ys / xs if xs else float("nan"))


anchors = shared_anchor_triangle(MOTOR_SPACING)
results = {name: simulate(strat, anchors) for strat, name in STRATS}

# ---- numeric summary (used in thesis text) ----
print(f"{'strategy':18s} {'issues':>6s} {'max_res':>8s} {'mean_res':>8s} {'Xspan':>7s} {'Yspan':>7s} {'ratio':>6s}")
summary = {}
for _, name in STRATS:
    cmd, rec, res, issue, _ = results[name]
    finite = res[np.isfinite(res)]
    xs, ys, ratio = circle_ratio(rec)
    summary[name] = dict(issues=int(issue.sum()), maxr=float(finite.max()),
                         meanr=float(finite.mean()), xspan=xs, yspan=ys, ratio=ratio)
    print(f"{name:18s} {int(issue.sum()):6d} {finite.max():8.3f} {finite.mean():8.3f} "
          f"{xs:7.1f} {ys:7.1f} {ratio:6.3f}")

# ================= Figure 1: reconstruction paths (2x2) =================
fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for ax, (_, name) in zip(axes.ravel(), STRATS):
    cmd, rec, res, issue, _ = results[name]
    ax.plot(cmd[:, 0], cmd[:, 1], "--", color="0.5", lw=1.4, label="Commanded path", zorder=1)
    ax.plot(rec[:, 0], rec[:, 1], "-", color="C0", lw=2.0, label="Reconstructed tactor", zorder=2)
    if issue.any():
        ax.scatter(rec[issue, 0], rec[issue, 1], s=22, c="crimson", edgecolors="k",
                   linewidths=0.4, zorder=3, label="Wire-residual flag")
    s = summary[name]
    ax.set_title(f"{name}\ncircle aspect ratio = {s['ratio']:.3f}, "
                 f"max residual = {s['maxr']:.2f}, flags = {s['issues']}", fontsize=10)
    ax.set_xlabel("X (controller units)"); ax.set_ylabel("Y (controller units)")
    ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Closed-loop reconstruction of the commanded path, by movement strategy\n"
             "(shared mechanism anchor triangle; circle reconstructs round for all four)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
f1 = os.path.join(OUT, "reconstruction_paths_by_strategy.png")
fig.savefig(f1, dpi=150); plt.close(fig)

# ================= Figure 2: wire residual per step =================
fig, ax = plt.subplots(figsize=(11, 5.2))
colors = {"cardinal": "C0", "cardinal_diagonal": "C1", "free_form": "C2", "ik": "C3"}
for _, name in STRATS:
    _, _, res, _, _ = results[name]
    r = np.where(np.isfinite(res), res, np.nan)
    ax.plot(np.arange(len(r)), r, color=colors[name], lw=1.6, label=name)
ax.axhline(THRESHOLD, color="k", ls="--", lw=1.2, label=f"alert threshold = {THRESHOLD}")
ax.axvspan(0, N_LINE, color="0.92", zorder=0)
ax.axvspan(N_LINE, N_LINE + N_CIRCLE, color="0.97", zorder=0)
ax.text(N_LINE / 2, ax.get_ylim()[1] * 0.95, "line out", ha="center", fontsize=8, color="0.4")
ax.text(N_LINE + N_CIRCLE / 2, ax.get_ylim()[1] * 0.95, "circle", ha="center", fontsize=8, color="0.4")
ax.set_xlabel("Path step"); ax.set_ylabel("Trilateration residual RMSE (controller units)")
ax.set_title("Wire-length self-consistency (trilateration residual) per step, by strategy")
ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
f2 = os.path.join(OUT, "wire_residual_by_strategy.png")
fig.savefig(f2, dpi=150); plt.close(fig)

# ============ Figure 3: IK before/after triangle match ============
# Decode the SAME IK wire-length log two ways: mismatched flat triangle vs shared triangle.
_, _, _, _, ik_lengths = results["ik"]
flat = flat_legacy_triangle(MOTOR_SPACING)
# the legacy reconstruction used initial lengths from its own flat triangle
flat_init = np.linalg.norm(flat - np.array([0.0, 0.0]), axis=1)
shared_init = np.linalg.norm(anchors - np.array([0.0, 0.0]), axis=1)

# ik_lengths were accumulated on the shared-triangle baseline; rebuild the legacy
# baseline by replacing the initial term (delta is identical, only the baseline differs).
ik_deltas = ik_lengths - shared_init  # per-step wire deltas commanded by IK
rec_flat, rec_shared = [], []
for d in ik_deltas:
    Lf = flat_init + d
    Ls = shared_init + d
    if np.all(Lf > 0):
        xf, yf, _ = trilaterate(flat, Lf); rec_flat.append((xf, yf))
    else:
        rec_flat.append((np.nan, np.nan))
    if np.all(Ls > 0):
        xs2, ys2, _ = trilaterate(anchors, Ls); rec_shared.append((xs2, ys2))
    else:
        rec_shared.append((np.nan, np.nan))
rec_flat = np.array(rec_flat); rec_shared = np.array(rec_shared)
cmd_ik = results["ik"][0]

def seg_ratio(arr):
    s = arr[N_LINE:N_LINE + N_CIRCLE]; s = s[~np.isnan(s[:, 0])]
    xs = s[:, 0].max() - s[:, 0].min(); ys = s[:, 1].max() - s[:, 1].min()
    return xs, ys, ys / xs

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
for ax, arr, ttl, tri, triname in (
    (axes[0], rec_flat, "Before: mismatched triangles", flat, "legacy flat triangle"),
    (axes[1], rec_shared, "After: shared mechanism triangle", anchors, "default_model triangle"),
):
    xs, ys, ratio = seg_ratio(arr)
    ax.plot(cmd_ik[:, 0], cmd_ik[:, 1], "--", color="0.5", lw=1.4, label="Commanded path")
    ax.plot(arr[:, 0], arr[:, 1], "-", color="C3", lw=2.0, label="Reconstructed (IK)")
    ax.scatter(*tri.T, marker="^", s=90, c="k", zorder=5, label=f"Anchors ({triname})")
    ax.set_title(f"{ttl}\ncircle aspect ratio = {ratio:.2f} (Xspan {xs:.0f}, Yspan {ys:.0f})", fontsize=10)
    ax.set_xlabel("X (controller units)"); ax.set_ylabel("Y (controller units)")
    ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Reconstruction is correct only when encode and decode share the same anchor triangle\n"
             "(IK commanded circle; identical wire-length commands decoded two ways)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
f3 = os.path.join(OUT, "ik_triangle_mismatch_before_after.png")
fig.savefig(f3, dpi=150); plt.close(fig)

print("\nBefore/after IK circle aspect ratios:")
print("  mismatched flat triangle:", round(seg_ratio(rec_flat)[2], 3))
print("  shared mechanism triangle:", round(seg_ratio(rec_shared)[2], 3))
print("\nSaved:")
for f in (f1, f2, f3):
    print(" ", os.path.relpath(f, REPO))
