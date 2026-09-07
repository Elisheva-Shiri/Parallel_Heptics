"""
Kinematic diagrams for the 3-leg SRR-orthogonal skin-stretch mechanism
=====================================================================

WHY THIS EXISTS
---------------
The ICRA paper needs a proper mechanism figure in the style of Zhou et al.,
"Learning-based Estimation of Forward Kinematics for an Orthotic Parallel
Robotic Mechanism" (arXiv:2503.11855), Figs. 1-2: one panel showing the whole
3-chain layout, and one panel showing a single chain with every joint center,
link length and joint angle labelled.

The only leg diagram that existed in the thesis media was hand-annotated with
constants that CONTRADICT the solver (d4=15.8, d3=16.5 vs the real d1=4,
d2=11.0, d3=9.5), so it could not be used. This script draws both panels
directly from unified_ik_starter.default_model() and from a real IK solution,
so the figure can never disagree with the text.

Panel (a): base-plane layout - the three anchors Pb_top/right/left, the moving
           platform, and the three legs at a nominal pose.
Panel (b): single-leg (chain) diagram - Pb -> P3 -> P2 -> P1, with d1, d2, d3
           and the joint angles phi2, phi3 marked, plus the working height z.

OUTPUT: paper/icra2027_paper/figures/mechanism_kinematic_diagram.png (+ .pdf)

Dependencies: numpy, matplotlib (+ unified_ik_starter).
Run:  uv run python kinematics/plot_leg_geometry.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unified_ik_starter import default_model, solve_all_legs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "paper", "icra2027_paper", "figures")

LEGS = ("top", "right", "left")
LEG_COLOR = {"top": "#1f77b4", "right": "#2ca02c", "left": "#d62728"}

# nominal pose used for the drawing (centred tactor at the working height)
PHI1 = math.pi / 2.0


def solve_nominal(model):
    z = min(6.0, model["lengths"]["d3"] - 0.1)
    P1 = (0.0, 0.0, z)
    sol = solve_all_legs(P1, PHI1, model=model)
    return P1, z, sol


def panel_layout(ax, model, P1, sol):
    """Panel (a): the three chains seen in the base (XY) plane."""
    A = {leg: np.array(model["anchors"][leg], float) for leg in LEGS}
    Pm = np.array(sol["shared"]["Pm"], float)
    P1 = np.array(P1, float)

    # base anchor triangle
    tri = np.array([A["top"][:2], A["right"][:2], A["left"][:2], A["top"][:2]])
    ax.plot(tri[:, 0], tri[:, 1], "--", color="0.55", lw=1.0, zorder=1)

    for leg in LEGS:
        s = sol[leg]
        if not s["valid"]:
            continue
        Pb = A[leg]
        P3 = np.array(s["selected_P3"], float)
        P2 = np.array(s["P2"], float)
        c = LEG_COLOR[leg]
        # chain Pb -> P3 -> P2 -> P1 projected on XY
        chain = np.array([Pb[:2], P3[:2], P2[:2], P1[:2]])
        ax.plot(chain[:, 0], chain[:, 1], "-", color=c, lw=2.0, zorder=3,
                label=f"{leg} leg")
        ax.plot(*Pb[:2], "s", color=c, ms=8, mec="k", mew=0.6, zorder=4)
        ax.plot(*P3[:2], "o", color=c, ms=6, mec="k", mew=0.6, zorder=4)
        ax.plot(*P2[:2], "^", color=c, ms=6, mec="k", mew=0.6, zorder=4)
        lab_off = {"top": (6, 6), "right": (-6, 8), "left": (2, 8)}[leg]
        ax.annotate(rf"$P_b^{{\rm {leg}}}$", Pb[:2], textcoords="offset points",
                    xytext=lab_off, fontsize=8, color=c,
                    ha="right" if leg == "right" else "left")

    # platform
    ax.plot(*P1[:2], "*", color="k", ms=14, zorder=5)
    ax.annotate(r"$P_1$ (tactor)", P1[:2], textcoords="offset points",
                xytext=(8, -12), fontsize=9)
    ax.annotate(r"$P_m$", Pm[:2], textcoords="offset points",
                xytext=(6, 4), fontsize=9, color="0.3")
    ax.plot([P1[0], Pm[0]], [P1[1], Pm[1]], "-", color="0.3", lw=1.4, zorder=4)

    # d1 label on the P1->Pm offset
    mid = (P1[:2] + Pm[:2]) / 2
    ax.annotate(r"$d_1$", mid, textcoords="offset points", xytext=(-16, 0),
                fontsize=9, color="0.3")

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(a) Three-leg layout (base plane)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.margins(0.18)


def panel_chain(ax, model, P1, z, sol, leg="left"):
    """Panel (b): the IK construction for one leg, drawn IN the z=z1 plane.

    Everything here is true to scale: P1, Pm, P2 and both P3 candidates all lie
    in the plane z=z1, and the two circles whose intersection fixes P3 are drawn
    at their real radii. This is the plane where the closed-form solve happens,
    so nothing is foreshortened (a radial/elevation view would distort d2).
    """
    from unified_ik_starter import circle_circle_intersection_xy

    s = sol[leg]
    Pb = np.array(model["anchors"][leg], float)
    P2 = np.array(s["P2"], float)
    P3sel = np.array(s["selected_P3"], float)
    P1 = np.array(P1, float)
    Pm = np.array(sol["shared"]["Pm"], float)
    d = model["lengths"]

    # in-plane radius contributed by the Pb sphere at height z
    rb_sq = d["d3"] ** 2 - (z - Pb[2]) ** 2
    rb = math.sqrt(max(rb_sq, 0.0))
    cands = circle_circle_intersection_xy(P2[0], P2[1], d["d2"],
                                          Pb[0], Pb[1], rb)

    # the two circles
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(P2[0] + d["d2"] * np.cos(th), P2[1] + d["d2"] * np.sin(th),
            "-", color="#1f77b4", lw=1.2, alpha=0.75, zorder=2)
    ax.plot(Pb[0] + rb * np.cos(th), Pb[1] + rb * np.sin(th),
            "-", color="#d62728", lw=1.2, alpha=0.75, zorder=2)

    # radius call-outs
    ax.plot([P2[0], P2[0] + d["d2"]], [P2[1], P2[1]], ":", color="#1f77b4",
            lw=1.4, zorder=3)
    ax.annotate(rf"$d_2={d['d2']}$", (P2[0] + d["d2"] / 2, P2[1]),
                textcoords="offset points", xytext=(6, 9), fontsize=9,
                color="#1f77b4", fontweight="bold")
    ax.plot([Pb[0], Pb[0] + rb], [Pb[1], Pb[1]], ":", color="#d62728",
            lw=1.4, zorder=3)
    ax.annotate(rf"$R_b={rb:.2f}$", (Pb[0] + rb / 2, Pb[1]),
                textcoords="offset points", xytext=(-2, 7), fontsize=9,
                color="#d62728", fontweight="bold", ha="center")

    # the solved chain P1 -> P2 -> P3(selected), and Pb -> P3
    ax.plot([P1[0], Pm[0]], [P1[1], Pm[1]], "-", color="0.35", lw=1.8, zorder=4)
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], "-", color="0.35", lw=1.8, zorder=4)
    ax.plot([P2[0], P3sel[0]], [P2[1], P3sel[1]], "-", color="#1f77b4",
            lw=2.6, zorder=5)
    ax.plot([Pb[0], P3sel[0]], [Pb[1], P3sel[1]], "-", color="#d62728",
            lw=2.6, zorder=5)

    # candidates: selected vs rejected branch
    for c in cands:
        is_sel = math.hypot(c[0] - P3sel[0], c[1] - P3sel[1]) < 1e-6
        ax.plot(c[0], c[1], "o", ms=10 if is_sel else 8,
                color="#d62728" if is_sel else "white",
                mec="#d62728", mew=1.6, zorder=6)
        ax.annotate(r"$P_3$ (selected)" if is_sel else r"$P_3'$ (rejected"
                    "\n" r"branch)", (c[0], c[1]),
                    textcoords="offset points",
                    xytext=(10, -4) if is_sel else (8, -20),
                    fontsize=8, color="#d62728")

    # key points
    for pt, lab, mk, col, off in [
        (Pb, r"$P_b$ (S, base)", "s", "#d62728", (-8, -12)),
        (P2, r"$P_2$ (R)", "^", "#1f77b4", (-10, 4)),
        (Pm, r"$P_m$", "d", "0.35", (6, 2)),
        (P1, r"$P_1$ (tactor)", "*", "k", (10, -10)),
    ]:
        ax.plot(pt[0], pt[1], mk, color=col, ms=13 if mk == "*" else 8,
                mec="k", mew=0.6, zorder=7)
        ax.annotate(lab, (pt[0], pt[1]), textcoords="offset points",
                    xytext=off, fontsize=8)

    ax.annotate(rf"plane $z=z_1={z:.1f}$ mm" "\n"
                r"$R_b=\sqrt{d_3^2-(z_1-z_p)^2}$",
                (0.02, 0.02), xycoords="axes fraction", fontsize=8,
                color="0.35", va="bottom")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(r"(b) IK construction in the $z=z_1$ plane", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = default_model()
    P1, z, sol = solve_nominal(model)
    for leg in LEGS:
        if not sol[leg]["valid"]:
            raise RuntimeError(f"nominal pose invalid for {leg}: "
                               f"{sol[leg]['fail_reason']}")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    panel_layout(axes[0], model, P1, sol)
    panel_chain(axes[1], model, P1, z, sol, leg="left")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"mechanism_kinematic_diagram.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)

    # echo the constants so the paper table can be checked against the code
    print("\nModel constants used (must match the paper's Table):")
    print(f"  d1 (platform offset) = {model['lengths']['d1']}")
    print(f"  d2 (SR link)         = {model['lengths']['d2']}")
    print(f"  d3 (RR link)         = {model['lengths']['d3']}")
    for leg in LEGS:
        print(f"  anchor {leg:5s} = {model['anchors'][leg]}")
    print(f"  working height z     = {z}")
    base = np.linalg.norm(np.array(model['anchors']['right'])
                          - np.array(model['anchors']['left']))
    print(f"  anchor spacing right-left = {base:.2f} mm")


if __name__ == "__main__":
    main()
