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
  3. cardinal_diagonal_vs_ik_circle_and_motor_commands.png
     - combined equal-height plot matrix: circular reconstruction + motor commands
Also prints the numeric summary used in the thesis text.
"""
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
current = SCRIPT_DIR
while True:
    if os.path.exists(os.path.join(current, "motor_controller.py")):
        REPO = current
        break
    parent = os.path.dirname(current)
    if parent == current:
        raise FileNotFoundError(
            f"Could not locate repository root from {SCRIPT_DIR}. "
            "Expected a parent folder containing motor_controller.py."
        )
    current = parent
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from consts import EDGE_THRESHOLD, TOP_HEIGHT, TOP_WIDTH
from haptic_mapping import map_object_displacement_to_tactor
from kinematics import unified_ik_starter as ik
from kinematics import wire_forward_kinematics as wire_fk
from motor_controller import HandOrientation, MotorController, MotorSetId, MovementStrategy

OUT = os.path.join(REPO, "analysis", "movement_strategy_results", "figures")
os.makedirs(OUT, exist_ok=True)

MOTOR_SPACING = 1000.0
THRESHOLD = 3.3
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
    model = ik.default_model()
    sim_to_ik_scale = c._get_ik_base_span(model) / MOTOR_SPACING
    ik_to_sim_scale = 1.0 / sim_to_ik_scale
    rest_p1 = wire_fk.rest_pose(model)
    previous_p1 = rest_p1.copy()
    init = np.linalg.norm(anchors - np.array([0.0, 0.0]), axis=1)
    pos = {0: 0, 1: 0, 2: 0}
    cmd, rec, res, issue, command_log = [], [], [], [], []
    pattern, _, _ = build_pattern()
    for ox, oy in pattern:
        tx, ty = map_object_displacement_to_tactor(obj_x=ox, obj_y=oy, oppose_motion=True)
        for m in c.calculate_motor_movements(motor_set_id=MotorSetId.MOTORS_0_2, stiffness_value=1,
                                             obj_x=tx, obj_y=ty, motors_enabled=True, reset_to_origin=False):
            if m.index in pos:
                pos[m.index] = m.pos
        command = np.array([pos[0], pos[1], pos[2]], float)
        command_log.append(command.copy())
        cmd.append((tx, ty))
        if strategy == MovementStrategy.IK:
            fk = wire_fk.solve_wire_fk(
                command * sim_to_ik_scale,
                model,
                initial_P1=previous_p1,
                rest_P1=rest_p1,
            )
            previous_p1 = fk.P1
            rec.append((fk.P1[0] * ik_to_sim_scale, fk.P1[1] * ik_to_sim_scale))
            r = fk.residual_rms * ik_to_sim_scale
            res.append(r)
            issue.append((not fk.valid) or r > THRESHOLD)
        else:
            L = init + command
            if np.any(L <= 0):
                rec.append((np.nan, np.nan)); res.append(np.inf); issue.append(True); continue
            x, y, r = trilaterate(anchors, L)
            rec.append((x, y)); res.append(r); issue.append(r > THRESHOLD)
    return (np.array(cmd), np.array(rec), np.array(res), np.array(issue), np.array(command_log))


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

# ========== Cardinal-diagonal vs IK: circle and motor commands ==========
# Thesis-facing comparison: one figure matrix with equal-height panels. The
# circular reconstruction keeps an equal data aspect; the motor-command panel is
# allowed to be wider so both panels can share one visual row without forcing the
# same width.
TITLE_FS = 48
LABEL_FS = 45
TICK_FS = 45
LEGEND_FS = 45
LINE_W = 2.8
GRID_ALPHA = 0.28



def style_panel(ax):
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.title.set_fontsize(TITLE_FS)
    ax.xaxis.label.set_size(LABEL_FS)
    ax.yaxis.label.set_size(LABEL_FS)
    ax.grid(True, alpha=GRID_ALPHA)


circle = slice(N_LINE, N_LINE + N_CIRCLE)
cmd_cd, rec_cd, res_cd, issue_cd, lengths_cd = results["cardinal_diagonal"]
cmd_ik, rec_ik, res_ik, issue_ik, lengths_ik = results["ik"]
deltas_cd = lengths_cd
deltas_ik = lengths_ik

ik_colors = {
    0: "#ff6b35",  # red-orange
    1: "#d7191c",  # red
    2: "#67000d",  # maroon
}
cd_colors = {
    0: "#f6c85f",  # gold
    1: "#ff9f1c",  # orange
    2: "#cc5500",  # dark orange
}

fig, (ax_circle, ax_motor) = plt.subplots(
    1, 2,
    figsize=(42, 14),
    gridspec_kw={"width_ratios": [0.90, 1.20], "wspace": 0.18},
    constrained_layout=False,
)

commanded_handle, = ax_circle.plot(
    cmd_ik[circle, 0], cmd_ik[circle, 1], "--", color="0.45", lw=LINE_W,
    label="Commanded", zorder=1,
)
ik_handle, = ax_circle.plot(
    rec_ik[circle, 0], rec_ik[circle, 1], "-", color="firebrick", lw=LINE_W + 0.6,
    alpha=0.82, label="IK", zorder=2,
)
cd_handle, = ax_circle.plot(
    rec_cd[circle, 0], rec_cd[circle, 1], "-", color="darkorange", lw=LINE_W + 0.2,
    label="CD", zorder=3,
)
ax_circle.set_title("Circular reconstruction")
ax_circle.set_xlabel("X")
ax_circle.set_ylabel("Y")
ax_circle.set_yticks(np.arange(-150, 151, 50))
ax_circle.set_aspect("equal", "box")
ax_circle.set_anchor("E")
style_panel(ax_circle)
ax_circle.legend(
    [commanded_handle, ik_handle, cd_handle],
    [h.get_label() for h in (commanded_handle, ik_handle, cd_handle)],
    loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
    fontsize=LEGEND_FS, frameon=False, handlelength=1.25, columnspacing=0.55,
)

steps = np.arange(len(deltas_ik))
ax_motor.axhline(0.0, color="0.25", lw=1.6, alpha=0.6)
for motor_idx in range(3):
    ax_motor.plot(
        steps, deltas_ik[:, motor_idx], "-", color=ik_colors[motor_idx],
        lw=LINE_W + 0.2, alpha=0.82, label=f"IK M{motor_idx}", zorder=2,
    )
for motor_idx in range(3):
    ax_motor.plot(
        steps, deltas_cd[:, motor_idx], "--", color=cd_colors[motor_idx],
        lw=LINE_W + 0.2, label=f"CD M{motor_idx}", zorder=3,
    )
ax_motor.set_title("Motor commands")
ax_motor.set_xlabel("Step")
ax_motor.set_ylabel("Command", labelpad=4)
ax_motor.set_yticks(np.arange(-150, 151, 50))
style_panel(ax_motor)
ax_motor.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6,
    fontsize=LEGEND_FS, frameon=False, handlelength=1.25, columnspacing=0.55,
)

fig.align_labels()
fig.subplots_adjust(left=0.055, right=0.99, bottom=0.25, top=0.86, wspace=0.18)
f5_combined = os.path.join(OUT, "cardinal_diagonal_vs_ik_circle_and_motor_commands.png")
f5_combined_tmp = os.path.join(OUT, "cardinal_diagonal_vs_ik_circle_and_motor_commands.tmp.png")
fig.savefig(f5_combined_tmp, dpi=150)
os.replace(f5_combined_tmp, f5_combined)
plt.close(fig)

# ================= Figure 2: wire residual per step =================
fig, ax = plt.subplots(figsize=(11, 5.2))
colors = {"cardinal": "C0", "cardinal_diagonal": "C1", "free_form": "C2", "ik": "C3"}
for _, name in STRATS:
    _, _, res, _, _ = results[name]
    r = np.where(np.isfinite(res), res, np.nan)
    ax.plot(np.arange(len(r)), r, color=colors[name], lw=1.6, label=name)
ax.axhline(THRESHOLD, color="k", ls="--", lw=1.2, label=f"strategy-fair threshold = {THRESHOLD}")
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

print("\nCardinal-diagonal vs IK comparison:")
print(f"  cardinal-diagonal circle aspect ratio: {summary['cardinal_diagonal']['ratio']:.3f}")
print(f"  IK circle aspect ratio: {summary['ik']['ratio']:.3f}")
print(f"  cardinal-diagonal max residual: {summary['cardinal_diagonal']['maxr']:.3f}")
print(f"  IK max residual: {summary['ik']['maxr']:.3f}")
print("\nSaved:")
for f in (f1, f5_combined, f2):
    print(" ", os.path.relpath(f, REPO))
