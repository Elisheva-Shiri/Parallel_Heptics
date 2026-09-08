from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Flowable,
    KeepTogether,
)
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from unified_ik_starter import default_model, solve_all_legs  # noqa: E402
from forward_kinematics import (  # noqa: E402
    LEG_ORDER,
    working_height,
    actuated_lengths,
    reference_lengths,
    solve_fixed_height_fk,
    forward_kinematics_3d,
    planar_condition_number,
)
from wire_forward_kinematics import pose_to_wire_deltas, solve_wire_fk  # noqa: E402

OUT = ROOT / "output" / "pdf" / "fk_explanation.pdf"

BLUE = colors.HexColor("#1F77B4")
GREEN = colors.HexColor("#2CA02C")
RED = colors.HexColor("#D62728")
ORANGE = colors.HexColor("#FF7F0E")
PURPLE = colors.HexColor("#6A3D9A")
DARK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
LIGHT = colors.HexColor("#F3F4F6")
LIGHT_BLUE = colors.HexColor("#EAF3FF")
LIGHT_GREEN = colors.HexColor("#EAF8EF")
LEG_COLORS = {"top": BLUE, "right": GREEN, "left": RED}


def fmt_tuple(vals, nd=2):
    return "(" + ", ".join(f"{float(v):.{nd}f}" for v in vals) + ")"


class MechanismScheme(Flowable):
    def __init__(self, width=520, height=255):
        super().__init__()
        self.width = width
        self.height = height

    def _proj(self, x, y, z, cx, cy, scale=8.7, zscale=8.5):
        # Simple engineering isometric projection for a readable mechanism sketch.
        u = cx + scale * (x + 0.45 * y)
        v = cy + scale * (0.22 * y) + zscale * z
        return u, v

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        model = default_model()
        z = working_height(model)
        P1 = np.array([0.0, 0.0, z], dtype=float)
        anchors = {leg: np.array(model["anchors"][leg], dtype=float) for leg in LEG_ORDER}
        lengths = actuated_lengths(P1, model)
        cx, cy = w * 0.33, 42

        # Background panel.
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.roundRect(0, 0, w, h, 8, stroke=1, fill=1)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(16, h - 22, "Scheme: three fixed anchors measure distances to P1")

        pts = {leg: self._proj(*anchors[leg], cx, cy) for leg in LEG_ORDER}
        p1 = self._proj(*P1, cx, cy)
        pxy = self._proj(P1[0], P1[1], 0, cx, cy)

        # Base triangle and platform projection.
        base_poly = [pts["top"], pts["right"], pts["left"]]
        path = c.beginPath()
        path.moveTo(*base_poly[0])
        for p in base_poly[1:]:
            path.lineTo(*p)
        path.close()
        c.setFillColor(colors.HexColor("#F8FAFC"))
        c.setStrokeColor(colors.HexColor("#94A3B8"))
        c.setLineWidth(1)
        c.drawPath(path, stroke=1, fill=1)

        # Wires.
        c.setLineWidth(1.7)
        for idx, leg in enumerate(LEG_ORDER):
            c.setStrokeColor(LEG_COLORS[leg])
            c.line(pts[leg][0], pts[leg][1], p1[0], p1[1])
            mx = 0.55 * pts[leg][0] + 0.45 * p1[0]
            my = 0.55 * pts[leg][1] + 0.45 * p1[1]
            c.setFillColor(LEG_COLORS[leg])
            c.setFont("Helvetica", 8)
            c.drawString(mx + 4, my + 2, f"ell_{idx+1}")

        # z arrow and projection.
        c.setDash(2, 2)
        c.setStrokeColor(MUTED)
        c.line(pxy[0], pxy[1], p1[0], p1[1])
        c.setDash()
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawString(pxy[0] + 5, (pxy[1] + p1[1]) / 2, "known z")
        c.setStrokeColor(MUTED)
        c.line(pxy[0] - 4, pxy[1], pxy[0] + 4, pxy[1])

        # Anchors and P1.
        for leg in LEG_ORDER:
            x, y = pts[leg]
            c.setFillColor(LEG_COLORS[leg])
            c.setStrokeColor(colors.black)
            c.rect(x - 4, y - 4, 8, 8, stroke=1, fill=1)
            c.setFillColor(DARK)
            c.setFont("Helvetica", 8)
            offx, offy = {"top": (5, 6), "right": (5, -13), "left": (-42, -12)}[leg]
            c.drawString(x + offx, y + offy, f"Pb_{leg}")
        c.setFillColor(colors.black)
        c.circle(p1[0], p1[1], 6, stroke=0, fill=1)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(p1[0] + 8, p1[1] - 2, "P1=(x,y,z)")

        # Formula callout.
        fx, fy = w * 0.58, h - 61
        c.setFillColor(LIGHT_BLUE)
        c.setStrokeColor(BLUE)
        c.roundRect(fx, fy - 64, w * 0.36, 64, 6, stroke=1, fill=1)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(fx + 10, fy - 17, "Distance equations")
        c.setFont("Courier", 8)
        c.drawString(fx + 10, fy - 34, "ell_i = ||P1 - Pb_i||")
        c.drawString(fx + 10, fy - 48, "r_i   = model ell_i - input ell_i")

        # Single-chain scheme inset.
        ix, iy = w * 0.58, 34
        c.setFillColor(LIGHT)
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.roundRect(ix, iy, w * 0.36, 104, 6, stroke=1, fill=1)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(ix + 10, iy + 84, "Leg naming used by IK")
        pbs = (ix + 24, iy + 29)
        p3s = (ix + 90, iy + 54)
        p2s = (ix + 158, iy + 54)
        p1s = (ix + 200, iy + 76)
        c.setStrokeColor(DARK)
        c.setLineWidth(2)
        for a, b in [(pbs, p3s), (p3s, p2s), (p2s, p1s)]:
            c.line(a[0], a[1], b[0], b[1])
        for pt, lab, col in [(pbs, "Pb", RED), (p3s, "P3", ORANGE), (p2s, "P2", BLUE), (p1s, "P1", colors.black)]:
            c.setFillColor(col)
            c.circle(pt[0], pt[1], 4, stroke=0, fill=1)
            c.setFillColor(DARK)
            c.setFont("Helvetica", 8)
            c.drawCentredString(pt[0], pt[1] - 16, lab)
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawCentredString((pbs[0]+p3s[0])/2, (pbs[1]+p3s[1])/2 + 10, "d3")
        c.drawCentredString((p3s[0]+p2s[0])/2, p3s[1] + 7, "d2")
        c.drawString((p2s[0]+p1s[0])/2 - 4, (p2s[1]+p1s[1])/2 + 10, "d1")


class BlockDiagram(Flowable):
    def __init__(self, width=520, height=158, variant="length"):
        super().__init__()
        self.width = width
        self.height = height
        self.variant = variant

    def _box(self, c, x, y, bw, bh, title, subtitle="", fill=LIGHT_BLUE, stroke=BLUE):
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(1)
        c.roundRect(x, y, bw, bh, 6, stroke=1, fill=1)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(x + bw / 2, y + bh - 14, title)
        if subtitle:
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 7)
            lines = subtitle.split("\n")
            for i, line in enumerate(lines[:3]):
                c.drawCentredString(x + bw / 2, y + bh - 28 - 10 * i, line)

    def _arrow(self, c, x1, y1, x2, y2):
        c.setStrokeColor(MUTED)
        c.setFillColor(MUTED)
        c.setLineWidth(1.2)
        c.line(x1, y1, x2, y2)
        ang = math.atan2(y2 - y1, x2 - x1)
        s = 5
        p1 = (x2 - s * math.cos(ang - 0.45), y2 - s * math.sin(ang - 0.45))
        p2 = (x2 - s * math.cos(ang + 0.45), y2 - s * math.sin(ang + 0.45))
        path = c.beginPath()
        path.moveTo(x2, y2)
        path.lineTo(*p1)
        path.lineTo(*p2)
        path.close()
        c.drawPath(path, stroke=0, fill=1)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.roundRect(0, 0, w, h, 8, stroke=1, fill=1)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 10)
        title = "Block diagram: controller length-based FK" if self.variant == "length" else "Block diagram: wire-driven simulation FK"
        c.drawString(16, h - 22, title)
        if self.variant == "length":
            boxes = [
                (16, 72, 82, 48, "Motor deltas", "delta ell_i"),
                (119, 72, 88, 48, "Add reference", "ell_i = ell0_i\n+ delta ell_i"),
                (228, 72, 80, 48, "Seed", "closed-form\ntrilateration"),
                (329, 72, 82, 48, "GN / LM", "residual r\nJacobian J"),
                (432, 72, 72, 48, "Output", "P1 +\ndiagnostics"),
            ]
            fills = [LIGHT, LIGHT_GREEN, LIGHT_BLUE, colors.HexColor("#FFF7ED"), colors.HexColor("#F5F3FF")]
            strokes = [MUTED, GREEN, BLUE, ORANGE, PURPLE]
        else:
            boxes = [
                (18, 75, 85, 48, "Wire deltas", "measured or\nsynthetic"),
                (125, 75, 92, 48, "Pose guess", "q=[x,y,z]\nphi1 fixed"),
                (239, 75, 90, 48, "Leg geometry", "P2, P3, cable\nattachment"),
                (351, 75, 75, 48, "Residual", "predicted -\ntarget"),
                (448, 75, 58, 48, "Solve", "update q"),
            ]
            fills = [LIGHT, LIGHT_BLUE, LIGHT_GREEN, colors.HexColor("#FFF7ED"), colors.HexColor("#F5F3FF")]
            strokes = [MUTED, BLUE, GREEN, ORANGE, PURPLE]
        for i, b in enumerate(boxes):
            self._box(c, *b, fill=fills[i], stroke=strokes[i])
            if i < len(boxes) - 1:
                self._arrow(c, b[0] + b[2] + 5, b[1] + b[3] / 2, boxes[i + 1][0] - 5, boxes[i + 1][1] + boxes[i + 1][3] / 2)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        note = "Stop when residual RMS and step size are below tolerance; otherwise report best least-squares fit." if self.variant == "length" else "This mode uses the same d1/d2/d3 leg construction as IK, but adds cable-direction assumptions."
        c.drawString(18, 35, note)


class EquationBox(Flowable):
    def __init__(self, lines, width=520, height=None, title=""):
        super().__init__()
        self.lines = lines
        self.width = width
        self.height = height or (34 + 15 * len(lines))
        self.title = title

    def draw(self):
        c = self.canv
        c.setFillColor(colors.HexColor("#F8FAFC"))
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.roundRect(0, 0, self.width, self.height, 6, stroke=1, fill=1)
        y = self.height - 18
        if self.title:
            c.setFillColor(DARK)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(12, y, self.title)
            y -= 18
        c.setFillColor(DARK)
        c.setFont("Courier", 8.5)
        for line in self.lines:
            c.drawString(14, y, line)
            y -= 14


def make_styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="TitleCenter",
        parent=base["Title"],
        alignment=TA_CENTER,
        textColor=DARK,
        fontSize=22,
        leading=27,
        spaceAfter=7,
    ))
    base.add(ParagraphStyle(
        name="Subtitle",
        parent=base["BodyText"],
        alignment=TA_CENTER,
        textColor=MUTED,
        fontSize=10,
        leading=13,
        spaceAfter=12,
    ))
    base.add(ParagraphStyle(
        name="H1x",
        parent=base["Heading1"],
        textColor=DARK,
        fontSize=15,
        leading=18,
        spaceBefore=6,
        spaceAfter=7,
    ))
    base.add(ParagraphStyle(
        name="H2x",
        parent=base["Heading2"],
        textColor=DARK,
        fontSize=12,
        leading=15,
        spaceBefore=5,
        spaceAfter=5,
    ))
    base.add(ParagraphStyle(
        name="Bodyx",
        parent=base["BodyText"],
        textColor=DARK,
        fontSize=9.4,
        leading=12.5,
        spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="Smallx",
        parent=base["BodyText"],
        textColor=MUTED,
        fontSize=8,
        leading=10.5,
    ))
    base.add(ParagraphStyle(
        name="Callout",
        parent=base["BodyText"],
        textColor=DARK,
        backColor=colors.HexColor("#F8FAFC"),
        borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=0.7,
        borderPadding=7,
        fontSize=9.2,
        leading=12,
        spaceAfter=8,
    ))
    return base


def make_table(data, widths):
    # Use Paragraph cells so long technical text wraps instead of painting over
    # adjacent cells when the PDF is rendered.
    cell_style = ParagraphStyle(
        "TableCellWrap",
        fontName="Helvetica",
        fontSize=8,
        leading=10.2,
        textColor=DARK,
        wordWrap="CJK",
    )
    head_style = ParagraphStyle(
        "TableHeadWrap",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )
    wrapped = []
    for r, row in enumerate(data):
        style = head_style if r == 0 else cell_style
        wrapped.append([Paragraph(str(item), style) for item in row])
    t = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
    canvas.line(doc.leftMargin, height - 22, width - doc.rightMargin, height - 22)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(doc.leftMargin, height - 17, "Forward Kinematics explanation - generated from repository implementation")
    canvas.drawRightString(width - doc.rightMargin, height - 17, f"Page {doc.page}")
    canvas.line(doc.leftMargin, 24, width - doc.rightMargin, 24)
    canvas.drawString(doc.leftMargin, 13, "Sources: forward_kinematics.py, wire_forward_kinematics.py, unified_ik_starter.py")
    canvas.restoreState()


def build_pdf():
    model = default_model()
    z = working_height(model)
    rest = np.array([0.0, 0.0, z])
    ref_lengths = reference_lengths(model, z=z)
    sample = np.array([1.0, -1.0, 6.2])
    sample_lengths = actuated_lengths(sample, model)
    fixed_result = solve_fixed_height_fk(sample_lengths, sample[2], model)
    fixed_3d = forward_kinematics_3d(sample_lengths, model, return_diagnostics=True)
    deltas = sample_lengths - reference_lengths(model, z=sample[2])
    wire_deltas, _ = pose_to_wire_deltas(sample, model)
    wire_result = solve_wire_fk(wire_deltas, model)

    # Validation summary matching the repository demo, computed quickly here.
    grid = np.linspace(-5.0, 5.0, 21)
    fixed_errs, full3d_errs, conds = [], [], []
    for x in grid:
        for y in grid:
            P = np.array([x, y, z], dtype=float)
            lengths = actuated_lengths(P, model)
            res = solve_fixed_height_fk(lengths, z, model)
            fixed_errs.append(np.linalg.norm(res.point - P))
            res3 = forward_kinematics_3d(lengths, model, return_diagnostics=True)
            full3d_errs.append(np.linalg.norm(res3.point - P))
            conds.append(planar_condition_number(P, model))
    fixed_errs = np.asarray(fixed_errs)
    full3d_errs = np.asarray(full3d_errs)
    conds = np.asarray(conds)

    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=38,
        bottomMargin=34,
        title="Forward Kinematics Explanation",
        author="Codex",
    )
    story = []
    P = lambda text, style="Bodyx": Paragraph(text, styles[style])

    story.append(P("Forward Kinematics (FK) for the 3-leg skin-stretch mechanism", "TitleCenter"))
    story.append(P("One-file explanation of the repository FK model, equations, solver flow, assumptions, and validation evidence.", "Subtitle"))
    story.append(P("Target result: convert the three measured/commanded wire lengths into the observable tactor center <b>P1 = (x, y, z)</b>. In the controller-facing FK, z is supplied by the working-height model and the solver reconstructs x/y. A 3D variant can recover x/y/z from absolute lengths but has a mirror ambiguity because all anchors are coplanar.", "Callout"))
    story.append(MechanismScheme())
    story.append(Spacer(1, 8))
    const_data = [["Quantity", "Value used in code", "Meaning"]]
    for leg in LEG_ORDER:
        const_data.append([f"Pb_{leg}", fmt_tuple(model["anchors"][leg]), "Fixed motor/base anchor"])
    const_data.extend([
        ["d1", f"{model['lengths']['d1']:.2f} mm", "Platform offset from P1 to each P2 direction"],
        ["d2", f"{model['lengths']['d2']:.2f} mm", "Second link length P3 -> P2"],
        ["d3", f"{model['lengths']['d3']:.2f} mm", "First link / base reach Pb -> P3"],
        ["working z", f"{z:.2f} mm", "max(0, min(6, d3 - 0.1))"],
        ["leg order", ", ".join(LEG_ORDER), "Required input ordering for all three lengths/deltas"],
    ])
    story.append(make_table(const_data, [82, 148, 260]))

    story.append(PageBreak())
    story.append(P("1. Controller-facing FK: length trilateration", "H1x"))
    story.append(P("The file <b>forward_kinematics.py</b> treats the three inputs as absolute distances from anchors to P1. If the controller gives motor deltas, the deltas are first added to the reference/origin lengths. The result is the point P1 plus diagnostics: residuals, RMS residual, iterations, convergence flag, Jacobian rank, and condition number.", "Bodyx"))
    story.append(EquationBox([
        "Given anchors Pb_i and lengths ell_i, find P = [x, y, z].",
        "Fixed-height residual: r_i(x,y) = ||[x,y,z] - Pb_i|| - ell_i",
        "Least-squares objective: minimize 0.5 * sum_i r_i^2",
        "Jacobian row: J_i = ([x,y,z] - Pb_i) / ||[x,y,z] - Pb_i||",
        "Planar solve uses J_xy = first two columns of J because z is known.",
    ], title="Core equations"))
    story.append(Spacer(1, 8))
    story.append(BlockDiagram(variant="length"))
    story.append(Spacer(1, 8))
    steps = [
        ["Step", "What happens", "Why it matters"],
        ["1", "Validate exactly three finite lengths in order top, right, left.", "Prevents silent leg-order or shape mistakes."],
        ["2", "Build a closed-form x/y trilateration seed after subtracting the known z contribution.", "Starts Newton close to the geometric solution."],
        ["3", "Iterate damped Gauss-Newton / Levenberg-Marquardt on true length residuals.", "Handles noisy or inconsistent measured lengths better than returning only the linear seed."],
        ["4", "Return FKResult or legacy tuple (P1, residual_rms).", "Keeps controller compatibility while exposing health diagnostics."],
    ]
    story.append(make_table(steps, [36, 232, 222]))
    story.append(Spacer(1, 8))
    story.append(P("Important observability note: three anchor-to-P1 lengths do <b>not</b> uniquely encode platform yaw phi1 or passive joint angles. After FK, use the IK solver or extra sensing if phi1/passive angles are needed.", "Callout"))

    story.append(PageBreak())
    story.append(P("2. Absolute lengths, deltas, and the 3D option", "H1x"))
    story.append(P("The controller often stores wire commands as changes from an origin pose rather than as absolute geometric lengths. FK therefore has a conversion layer before solving.", "Bodyx"))
    story.append(EquationBox([
        "reference point P0 = [0, 0, z_work]",
        "ell0_i = ||P0 - Pb_i||",
        "absolute length ell_i = ell0_i + delta_i",
        "forward_kinematics_from_deltas(delta, z, model) = forward_kinematics(ell, z, model)",
    ], title="Delta-to-length conversion"))
    story.append(Spacer(1, 6))
    data = [["Reference length", "Value at P0=(0,0,z_work)"]]
    for leg, length in zip(LEG_ORDER, ref_lengths):
        data.append([f"ell0_{leg}", f"{length:.4f} mm"])
    story.append(make_table(data, [150, 190]))
    story.append(Spacer(1, 8))
    story.append(P("The 3D FK path, <b>forward_kinematics_3d(lengths, model)</b>, subtracts squared sphere equations to solve x/y, then computes z from the first sphere. Since all anchors are at z=0, the positive and negative z solutions have the same lengths; the implementation selects positive z by default.", "Bodyx"))
    story.append(EquationBox([
        "Subtract sphere equations to get linear equations in x and y.",
        "z^2 = ell_1^2 - (x - Pb_1x)^2 - (y - Pb_1y)^2",
        "choose z = +sqrt(z^2) for the physical tactor above the anchor plane.",
    ], title="3D coplanar-anchor solve"))
    story.append(Spacer(1, 8))
    sample_data = [["Sample check", "Value"]]
    sample_data.extend([
        ["Target P1", fmt_tuple(sample)],
        ["Absolute lengths", fmt_tuple(sample_lengths, 4)],
        ["Fixed-height recovered P1", fmt_tuple(fixed_result.point, 6)],
        ["Fixed-height residual RMS", f"{fixed_result.residual_rms:.3e} mm"],
        ["3D recovered P1", fmt_tuple(fixed_3d.point, 6)],
        ["3D residual RMS", f"{fixed_3d.residual_rms:.3e} mm"],
    ])
    story.append(make_table(sample_data, [165, 325]))

    story.append(PageBreak())
    story.append(P("3. Wire-driven simulation FK", "H1x"))
    story.append(P("The file <b>wire_forward_kinematics.py</b> solves a different, more assumption-heavy problem: recover P1 from three projected wire-release deltas on the first links. It fixes phi1=90 deg and uses the same d1/d2/d3 leg geometry as the IK code, then predicts cable deltas from the attachment-point motion projected along each cable exit direction.", "Bodyx"))
    story.append(BlockDiagram(variant="wire"))
    story.append(Spacer(1, 8))
    wire_table = [["Wire simulation assumption", "Current value"]]
    wire_table.extend([
        ["fixed phi1", "90 deg"],
        ["cable attachment on Pb -> P3", "0.50 of the first-link distance"],
        ["cable exit direction", "toward_platform_center with 55 deg elevation"],
        ["solve bounds", "x/y in +/-8 mm, z in [0.05, d3)"],
        ["sample target", fmt_tuple(sample)],
        ["sample wire deltas", fmt_tuple(wire_deltas, 6)],
        ["recovered P1", fmt_tuple(wire_result.P1, 6)],
        ["wire residual RMS", f"{wire_result.residual_rms:.3e}"],
    ])
    story.append(make_table(wire_table, [190, 300]))
    story.append(Spacer(1, 8))
    story.append(P("4. Validation evidence from the repository", "H1x"))
    val_table = [["Check", "Result"]]
    val_table.extend([
        ["Fixed-height FK round trip over +/-5 mm grid", f"max error {fixed_errs.max():.3e} mm, mean {fixed_errs.mean():.3e} mm"],
        ["3D FK round trip over same grid", f"max error {full3d_errs.max():.3e} mm, mean {full3d_errs.mean():.3e} mm"],
        ["Planar-Jacobian conditioning over grid", f"min {conds.min():.3f}, median {np.median(conds):.3f}, max {conds.max():.3f}"],
        ["Script validation", "python forward_kinematics.py -> VALIDATION PASSED"],
        ["Wire FK demo", "synthetic target recovered with residual RMS about 6.09e-15"],
    ])
    story.append(make_table(val_table, [210, 280]))
    story.append(Spacer(1, 8))
    story.append(P("Practical usage summary: use <b>forward_kinematics_from_deltas</b> when motor commands are deltas from origin; use <b>forward_kinematics</b> when you already have absolute anchor-to-P1 lengths; use <b>solve_wire_fk</b> only for the current wire-path simulation assumptions; use IK afterward if joint angles or phi1 are required.", "Callout"))
    story.append(P("Generated files are independent of the runtime controller and do not modify the FK source code.", "Smallx"))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build_pdf()
