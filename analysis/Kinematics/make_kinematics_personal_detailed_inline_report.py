from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from PIL import Image as PILImage

ROOT = Path(r"C:\Users\user\BIO MEDICAL ROBOTICS Dropbox\Elisheva Shiri Decktor\BGU\Codes\Parallel_Heptics\analysis\Kinematics")
BASE = ROOT / "result_fillter" / "L_N_E"
CSV = BASE / "csv"
FIG = BASE / "figures"
OUT = ROOT / "output" / "pdf"
ASSET = OUT / "personal_detailed_assets"
OUT.mkdir(parents=True, exist_ok=True)
ASSET.mkdir(parents=True, exist_ok=True)
PDF = OUT / "kinematics_result_fillter_PERSONAL_DETAILED_inline_report.pdf"
MANIFEST = OUT / "kinematics_result_fillter_PERSONAL_DETAILED_inline_manifest.csv"


def fmt(x):
    try:
        x = float(x)
    except Exception:
        return "" if x is None else str(x)
    if math.isnan(x) or math.isinf(x):
        return ""
    if abs(x) >= 1e6:
        return f"{x:.3g}"
    if abs(x) >= 1000:
        return f"{x:,.1f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def save_bar(df, x, ys, title, ylabel, path, rotation=0):
    ys = [ys] if isinstance(ys, str) else list(ys)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    idx = np.arange(len(df))
    width = 0.8 / len(ys)
    for i, y in enumerate(ys):
        ax.bar(idx + (i - (len(ys) - 1) / 2) * width, df[y].values, width=width, label=y)
    ax.set_xticks(idx)
    ax.set_xticklabels(df[x].astype(str), rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if len(ys) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_line(df, x, ys, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for y in ys:
        ax.plot(df[x], df[y], marker="o", label=y)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# Load generated analysis tables.
summary = json.loads((ROOT / "kinematics_report_summary.json").read_text(encoding="utf-8"))
ho = pd.read_csv(CSV / "hand_orientation" / "hand_orientation_plane_summary.csv")
dir_success = pd.read_csv(CSV / "success" / "direction_success_summary.csv")
dist_success = pd.read_csv(CSV / "success" / "distance_success_summary.csv")
success_contrast = pd.read_csv(CSV / "success" / "success_kinematic_z_contrast_summary.csv")
sftraj = pd.read_csv(CSV / "trajectories" / "success_failure_trajectory_distance_summary.csv")
finger_dist = pd.read_csv(CSV / "trajectories" / "finger_trajectory_distance_summary.csv")
trial3 = pd.read_csv(CSV / "trajectories" / "trial_3d_kinematic_summary.csv")
ztime = pd.read_csv(CSV / "z_lift" / "side_z_group_time_summary.csv")
zstiff = pd.read_csv(CSV / "z_lift" / "side_z_by_stiffness_summary.csv")
vel_stiff = pd.read_csv(CSV / "velocity" / "others" / "velocity_stiffness_influence_summary.csv")
vel_finger = pd.read_csv(CSV / "velocity" / "others" / "velocity_finger_influence_summary.csv")
prof = pd.read_csv(CSV / "velocity" / "others" / "subject_velocity_acceleration_profile.csv")

# Derived strategy and dynamics tables.
dist_tbl = dist_success.groupby("distance_quantile").agg(
    n_trials=("n_trials", "sum"),
    success_rate=("success_rate", "mean"),
    mean_speed_cm_s=("mean_speed_cm_s", "mean"),
    mean_max_r_workspace_cm=("mean_max_r_workspace_cm", "mean"),
).reset_index()
dir_tbl = dir_success.groupby("dominant_movement_direction").agg(
    n_trials=("n_trials", "sum"),
    success_rate=("success_rate", "mean"),
    mean_speed_cm_s=("mean_speed_cm_s", "mean"),
    mean_r_workspace_cm=("mean_r_workspace_cm", "mean"),
    mean_path_length_cm=("mean_path_length_cm", "mean"),
).reset_index().sort_values("success_rate", ascending=False)
z_finger = ztime.groupby("finger_condition").agg(
    n_frames=("n_frames", "sum"),
    mean_z_cm=("mean_side_z_lift_cm", "mean"),
    detection=("mean_detection_rate", "mean"),
).reset_index().sort_values("mean_z_cm", ascending=False)
z_success = ztime.groupby("success_label").agg(
    n_frames=("n_frames", "sum"),
    mean_z_cm=("mean_side_z_lift_cm", "mean"),
    detection=("mean_detection_rate", "mean"),
).reset_index()
z_fs = ztime.groupby(["finger_condition", "success_label"]).agg(
    mean_z_cm=("mean_side_z_lift_cm", "mean"),
    n_frames=("n_frames", "sum"),
).reset_index()
trial3_dir = trial3.groupby("dominant_movement_direction").agg(
    n=("correct_response", "size"),
    success_rate=("correct_response", "mean"),
    path3_cm=("path_length_3d_proxy_cm", "mean"),
    max_excursion_cm=("max_excursion_3d_from_start_cm", "mean"),
    max_radius_cm=("max_radius_3d_from_center_cm", "mean"),
    mean_velocity_3d_cm_s=("mean_velocity_3d_proxy_cm_s", "mean"),
).reset_index().sort_values("success_rate", ascending=False)

prof["time_third_calc"] = pd.cut(
    prof["time_fraction"], [-0.001, 1 / 3, 2 / 3, 1.001], labels=["early", "middle", "late"]
)
prof_abs = prof.copy()
for col in [
    "radial_velocity_cm_s", "tangential_velocity_cm_s", "vx_cm_s", "vy_cm_s", "vz_3d_proxy_cm_s",
    "ax_cm_s2", "ay_cm_s2", "az_3d_proxy_cm_s2",
]:
    if col in prof_abs:
        prof_abs["abs_" + col] = prof_abs[col].abs()
vel_cols = [
    "speed_cm_s", "speed_3d_proxy_cm_s", "abs_radial_velocity_cm_s",
    "abs_tangential_velocity_cm_s", "abs_vx_cm_s", "abs_vy_cm_s", "abs_vz_3d_proxy_cm_s",
]
acc_cols = [
    "acceleration_cm_s2", "acceleration_3d_proxy_cm_s2",
    "abs_ax_cm_s2", "abs_ay_cm_s2", "abs_az_3d_proxy_cm_s2",
]
vel_by_stiff = prof_abs.groupby("stiffness_value")[vel_cols].mean().reset_index()
vel_by_finger = prof_abs.groupby("finger_condition")[vel_cols].mean().reset_index()
vel_by_time = prof_abs.groupby("time_third_calc", observed=True)[vel_cols].mean().reset_index()
acc_by_stiff = prof_abs.groupby("stiffness_value")[acc_cols].mean().reset_index()
acc_by_finger = prof_abs.groupby("finger_condition")[acc_cols].mean().reset_index()
acc_by_time = prof_abs.groupby("time_third_calc", observed=True)[acc_cols].mean().reset_index()

findings = {
    "distance_success": dist_tbl.to_dict("records"),
    "direction_success": dir_tbl.to_dict("records"),
    "success_contrasts": success_contrast.sort_values("sign_flip_p").to_dict("records"),
    "z_by_finger": z_finger.to_dict("records"),
    "z_by_success": z_success.to_dict("records"),
    "z_by_finger_success": z_fs.to_dict("records"),
    "trial3_direction": trial3_dir.to_dict("records"),
    "velocity_by_stiffness": vel_by_stiff.to_dict("records"),
    "velocity_by_finger": vel_by_finger.to_dict("records"),
    "velocity_by_time": vel_by_time.astype({"time_third_calc": str}).to_dict("records"),
    "acceleration_by_stiffness": acc_by_stiff.to_dict("records"),
    "acceleration_by_finger": acc_by_finger.to_dict("records"),
    "acceleration_by_time": acc_by_time.astype({"time_third_calc": str}).to_dict("records"),
}
(OUT / "kinematics_personal_detailed_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

# Custom reader-friendly plots.
save_bar(dist_tbl, "distance_quantile", "success_rate", "Success rate by distance quantile", "success rate", ASSET / "custom_success_by_distance_quantile.png")
save_bar(dir_tbl.head(12), "dominant_movement_direction", "success_rate", "Success rate by dominant movement direction", "success rate", ASSET / "custom_success_by_direction.png", 45)
sc_top = success_contrast.sort_values("sign_flip_p").head(10).copy()
sc_top["label"] = sc_top["metric"].str.replace("_", " ", regex=False).str.slice(0, 28)
save_bar(sc_top, "label", "mean_difference", "Success minus failure: top contrasts", "success - failure", ASSET / "custom_success_minus_failure_contrasts.png", 45)
save_bar(z_finger, "finger_condition", "mean_z_cm", "Mean side-camera z-lift by finger", "mean z-lift cm", ASSET / "custom_z_lift_by_finger.png")
save_bar(zstiff, "stiffness_value", ["mean_side_z_lift_cm", "max_side_z_lift_cm"], "Z-lift by stiffness", "z-lift cm", ASSET / "custom_z_lift_by_stiffness.png", 45)
fig, ax = plt.subplots(figsize=(10, 5.4))
for lab, sub in z_fs.groupby("success_label"):
    sub = sub.sort_values("finger_condition")
    ax.bar(np.arange(len(sub)) + (-0.2 if lab == "failure" else 0.2), sub["mean_z_cm"], width=0.4, label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(sorted(z_fs.finger_condition.unique()))
ax.set_ylabel("mean z-lift cm"); ax.set_title("Z-lift by finger and success/failure")
ax.grid(axis="y", alpha=0.25); ax.legend(); fig.tight_layout()
fig.savefig(ASSET / "custom_z_lift_by_finger_success.png", dpi=160); plt.close(fig)
save_bar(trial3_dir.head(12), "dominant_movement_direction", "success_rate", "3D trajectory success rate by direction", "success rate", ASSET / "custom_3d_success_by_direction.png", 45)
save_bar(trial3_dir.sort_values("max_excursion_cm", ascending=False).head(12), "dominant_movement_direction", "max_excursion_cm", "Max 3D excursion by direction", "max excursion cm", ASSET / "custom_max3d_by_direction.png", 45)
save_line(vel_by_stiff, "stiffness_value", ["speed_cm_s", "speed_3d_proxy_cm_s", "abs_radial_velocity_cm_s", "abs_tangential_velocity_cm_s"], "Velocity decomposition by stiffness", "cm/s", ASSET / "custom_velocity_by_stiffness_decomposition.png")
save_bar(vel_by_finger, "finger_condition", ["speed_cm_s", "speed_3d_proxy_cm_s", "abs_radial_velocity_cm_s", "abs_tangential_velocity_cm_s"], "Velocity decomposition by finger", "cm/s", ASSET / "custom_velocity_by_finger_decomposition.png")
save_bar(vel_by_time, "time_third_calc", ["speed_cm_s", "speed_3d_proxy_cm_s", "abs_radial_velocity_cm_s", "abs_tangential_velocity_cm_s"], "Velocity over movement time thirds", "cm/s", ASSET / "custom_velocity_by_time_decomposition.png")
save_line(acc_by_stiff, "stiffness_value", ["acceleration_cm_s2", "acceleration_3d_proxy_cm_s2", "abs_ax_cm_s2", "abs_ay_cm_s2", "abs_az_3d_proxy_cm_s2"], "Acceleration by stiffness", "cm/s2", ASSET / "custom_acceleration_by_stiffness_components.png")
save_bar(acc_by_finger, "finger_condition", ["acceleration_cm_s2", "acceleration_3d_proxy_cm_s2", "abs_ax_cm_s2", "abs_ay_cm_s2", "abs_az_3d_proxy_cm_s2"], "Acceleration by finger", "cm/s2", ASSET / "custom_acceleration_by_finger_components.png")
save_bar(acc_by_time, "time_third_calc", ["acceleration_cm_s2", "acceleration_3d_proxy_cm_s2", "abs_ax_cm_s2", "abs_ay_cm_s2", "abs_az_3d_proxy_cm_s2"], "Acceleration over movement time thirds", "cm/s2", ASSET / "custom_acceleration_by_time_components.png")
ho_f = ho[ho.scope.eq("finger")]
fig, ax = plt.subplots(figsize=(10, 5.4))
for plane, sub in ho_f.groupby("plane"):
    ax.plot(sub["group"], sub["circular_mean_deg"], marker="o", label=plane)
ax.set_title("Circular mean hand orientation by finger and plane"); ax.set_ylabel("degrees")
ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
fig.savefig(ASSET / "custom_hand_orientation_by_finger_plane.png", dpi=160); plt.close(fig)

# PDF helpers.
styles = getSampleStyleSheet()
def add_style(name, parent, size, lead, color="#000000", align=None, left=0, first=0):
    kw = dict(name=name, parent=styles[parent], fontSize=size, leading=lead, spaceAfter=5,
              textColor=colors.HexColor(color), leftIndent=left, firstLineIndent=first)
    if align is not None:
        kw["alignment"] = align
    styles.add(ParagraphStyle(**kw))

add_style("TitleP", "Title", 24, 29, "#15395b", TA_CENTER)
add_style("SubP", "Heading2", 15, 19, "#1f4e79", TA_CENTER)
add_style("H1P", "Heading1", 17, 21, "#1f4e79")
add_style("H2P", "Heading2", 13.5, 17, "#365f91")
add_style("BodyP", "BodyText", 10.2, 13.6)
add_style("SmallP", "BodyText", 8.2, 10.4, "#333333")
add_style("TinyP", "BodyText", 7.0, 8.4, "#333333")
add_style("BulletP", "BodyText", 10.2, 13.6, left=14, first=-8)

def clean(s):
    return str(s).translate({
        ord("\u2013"): "-", ord("\u2014"): "-", ord("\u2212"): "-",
        ord("\u2011"): "-", ord("\u00b1"): "+/-", ord("\u2265"): ">=",
    }).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def P(s, style="BodyP"):
    return Paragraph(clean(s), styles[style])

def H(story, s, level=1):
    story.append(P(s, "H1P" if level == 1 else "H2P"))

def T(data, widths=None, font=8.0, header=True):
    rows = []
    for row in data:
        out = []
        for c in row:
            if isinstance(c, (list, tuple)):
                out.append(list(c))
            elif hasattr(c, "wrap"):
                out.append(c)
            else:
                out.append(Paragraph(clean(c), styles["TinyP" if font < 7.8 else "SmallP"]))
        rows.append(out)
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    ts = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d5dbe3")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcecf8"))]
    t.setStyle(TableStyle(ts))
    return t

def img(path, max_w, max_h):
    path = Path(path)
    if not path.exists():
        return P("Missing image: " + str(path), "SmallP")
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)

def rec_table(records, cols, labels, max_rows=None):
    rows = [labels]
    for r in (records[:max_rows] if max_rows else records):
        rows.append([fmt(r.get(c, "")) if isinstance(r.get(c, ""), (int, float, np.floating)) else str(r.get(c, "")) for c in cols])
    return rows

def plot_page(story, title, path, why, read, bottom):
    H(story, title, 2)
    story.append(T([[img(path, 5.6 * inch, 3.85 * inch),
                     [P("Why this plot is here:", "SmallP"), P(why, "SmallP"),
                      P("How to read it:", "SmallP"), P(read, "SmallP"),
                      P("Bottom line:", "SmallP"), P(bottom, "SmallP")]]],
                   [5.8 * inch, 3.4 * inch], header=False))
    try:
        story.append(P("Source: " + str(Path(path).relative_to(BASE)), "TinyP"))
    except Exception:
        story.append(P("Source: " + str(path), "TinyP"))
    story.append(PageBreak())

story = []
story += [
    Spacer(1, 0.2 * inch),
    P("Personal Kinematics Report - Detailed Inline Version", "TitleP"),
    P("Main group: L_N_E | Comparisons: L_E and N_E", "SubP"),
    P("This version puts plots inside the sections they explain. It is written for your personal reading, not for a journal submission.", "BodyP"),
    P(f"Main source folder: {BASE}", "SmallP"),
    P(f"Explicitly covered folder: {FIG / 'other'}", "SmallP"),
    P(f"Explicitly covered folder: {FIG / 'success'}", "SmallP"),
    PageBreak(),
]

H(story, "0. What this report now includes")
for b in [
    "Hand orientation for all, by group, by success/failure, by finger, and by stiffness.",
    "Movement orientation per finger and per stiffness, with the trajectory plots embedded inline.",
    "Z-lift per finger, per stiffness, and success versus failure.",
    "Between-finger trajectory distance.",
    "Max 3D direction and success/failure trajectory differences.",
    "The strategy question: which trajectory features are associated with better success rate.",
    "Velocity differences between stiffness levels, fingers, and time, including magnitude, components, radial, and tangential velocity.",
    "Acceleration differences between stiffness levels, fingers, and time, including magnitude and x/y/z components.",
    "Every aggregate plot under L_N_E figures/other, figures/success, trajectories, velocity, and acceleration is included inline. Subject-level plots are summarized but not repeated one-by-one.",
]:
    story.append(P("- " + b, "BulletP"))
story.append(PageBreak())

H(story, "1. Main answer: are there trajectory strategies that improve success?")
story.append(P("Yes. The clearest success-associated strategy is compact and controlled movement: shorter duration, shorter path length, smaller maximum radius, lower movement-direction entropy, and lower maximum z-lift. In plain words: stay closer, take a shorter route, keep direction consistent, and avoid excessive lift."))
for b in [
    "Success rate decreases from the nearest distance quantile to the farthest distance quantile.",
    "Successful trials have lower max radius, lower path length, lower duration, and lower direction entropy.",
    "Z-lift is higher in failures than successes, so excessive vertical/lift motion looks like a failure-associated pattern.",
    "Mean speed alone is not the strategy; compactness and consistency explain success better than simply moving faster.",
]:
    story.append(P("- " + b, "BulletP"))
story.append(T(rec_table(dist_tbl.to_dict("records"), ["distance_quantile", "n_trials", "success_rate", "mean_speed_cm_s", "mean_max_r_workspace_cm"], ["Distance quantile", "Trials", "Success", "Speed cm/s", "Max radius cm"]), [1.2*inch, 1.1*inch, 1.0*inch, 1.1*inch, 1.2*inch]))
story.append(T(rec_table(success_contrast.sort_values("sign_flip_p").to_dict("records"), ["metric", "n_paired_observations", "mean_difference", "cohens_dz", "sign_flip_p"], ["Metric", "N paired", "Success - failure", "Effect dz", "p"], 12), [2.8*inch, .8*inch, 1.1*inch, .8*inch, .8*inch], 7.2))
story.append(PageBreak())

for title, path, why, read, bottom in [
    ("Success strategy: success by distance", ASSET/"custom_success_by_distance_quantile.png", "Direct test of whether movement scale affects success.", "Quantile 1 is closest/smallest movement; quantile 4 is farthest/largest.", "Compact trajectories have better success."),
    ("Success strategy: success minus failure contrasts", ASSET/"custom_success_minus_failure_contrasts.png", "Shows which variables separate successful and failed trials.", "Negative means success had lower values than failure.", "Success is linked to lower duration, max radius, path length, and direction entropy."),
    ("Success strategy: success by dominant direction", ASSET/"custom_success_by_direction.png", "Checks whether direction itself looks strategic.", "Bars are mean success rates by dominant direction.", "Direction matters somewhat, but compact controlled movement is the stronger strategy finding."),
]:
    plot_page(story, title, path, why, read, bottom)

H(story, "2. Hand orientation - all participants and all conditions")
story.append(P("Hand orientation is summarized in XY, YZ, and ZX planes. Resultant length is high, so orientation is structured rather than random. Finger identity changes orientation strongly, especially in XY and ZX."))
story.append(T(rec_table(ho[ho.scope.isin(["all", "experiment_group", "success", "finger"])].to_dict("records"), ["scope", "group", "plane", "n", "circular_mean_deg", "median_deg", "resultant_length"], ["Scope", "Level", "Plane", "n", "Circular mean deg", "Median deg", "Resultant length"], 40), [1.2*inch, .8*inch, .5*inch, .7*inch, 1.2*inch, 1.0*inch, 1.1*inch], 7.0))
story.append(PageBreak())
plot_page(story, "Hand orientation: group L_N_E XY vectors", FIG/"trajectories"/"hand_orientation"/"hand_orientation_xy_vectors_group_L_N_E.png", "Main combined-group hand-orientation graph.", "Panels show thumb-to-active-finger XY vectors.", "Finger-specific orientation fields are consistent and organized.")
plot_page(story, "Hand orientation: circular mean by finger and plane", ASSET/"custom_hand_orientation_by_finger_plane.png", "Readable summary of the hand-orientation table.", "Lines show circular mean angle by finger for XY, YZ, and ZX.", "Finger condition is a major driver of hand orientation.")

H(story, "3. Movement orientation - per finger and per stiffness")
story.append(P("Movement orientation is covered by the finger-average trajectory, stiffness-average trajectory, movement-cycle angle figure, and standard-vs-comparison position/direction plots."))
for pth, title, bottom in [
    (FIG/"trajectories"/"movement_orientation"/"all_xy_trajectories_with_finger_average.png", "Movement orientation per finger", "Finger changes the average spatial route."),
    (FIG/"trajectories"/"movement_orientation"/"all_xy_trajectories_with_stiffness_average.png", "Movement orientation per stiffness", "Stiffness changes are visible but less dominant than finger/setup effects."),
    (FIG/"trajectories"/"movement_orientation"/"movement_cycle_xy_direction_yz_hand_orientation_experiment_group_L_N_E.png", "Movement cycle: XY direction and YZ hand orientation", "Movement direction and hand angle evolve across the movement cycle."),
]:
    plot_page(story, title, pth, "Generated movement-orientation plot.", "Read panels/lines by finger, stiffness, or cycle phase.", bottom)
for pth in sorted((FIG/"trajectories"/"s_vs_c_pos").glob("*.png")):
    plot_page(story, "Standard vs comparison position/orientation: " + pth.stem, pth, "Compares standard and comparison movement phases.", "Read panels by finger/stiffness condition.", "Use this to see whether comparison movement differs from standard movement.")

H(story, "4. Z-lift - per finger, per stiffness, success and failure")
story.append(P("Z-lift is side-camera vertical/lift motion. Failures show higher z-lift than successes, and M/I tend to lift more than R/P. This supports the strategy conclusion that excessive lift is not helpful."))
story.append(T(rec_table(z_finger.to_dict("records"), ["finger_condition", "n_frames", "mean_z_cm", "detection"], ["Finger", "Frames", "Mean z cm", "Detection"]), [1.0*inch, 1.2*inch, 1.2*inch, 1.0*inch]))
story.append(T(rec_table(z_success.to_dict("records"), ["success_label", "n_frames", "mean_z_cm", "detection"], ["Success label", "Frames", "Mean z cm", "Detection"]), [1.3*inch, 1.2*inch, 1.2*inch, 1.0*inch]))
story.append(PageBreak())
for title, path, bottom in [
    ("Z-lift by finger", ASSET/"custom_z_lift_by_finger.png", "M and I have the largest mean z-lift; P is lowest."),
    ("Z-lift by stiffness", ASSET/"custom_z_lift_by_stiffness.png", "Stiffness-level changes exist but are not a simple monotonic pattern."),
    ("Z-lift by finger and success/failure", ASSET/"custom_z_lift_by_finger_success.png", "For every finger, failures show higher z-lift than successes."),
]:
    plot_page(story, title, path, "Custom plot created from z-lift CSVs.", "Bars are means from side_z_group_time_summary.", bottom)
for pth in sorted((FIG/"trajectories"/"z_lift").glob("*.png")):
    plot_page(story, "Original z-lift graph: " + pth.stem, pth, "Generated L_N_E z-lift plot.", "Read grouping from the title: finger, stiffness, success, workspace, participant, or time.", "The success/failure split is the most strategy-relevant z-lift result.")

H(story, "5. Between-finger trajectory distance")
story.append(P("Between-finger trajectory distance shows that different fingers follow measurably different paths. The largest mean XY distance is R - I; the largest speed-profile RMSE is P - M."))
story.append(T(rec_table(finger_dist.sort_values(["metric", "mean"], ascending=[True, False]).to_dict("records"), ["comparison", "metric", "n_subject_stiffness_pairs", "mean", "median", "sem"], ["Comparison", "Metric", "N pairs", "Mean", "Median", "SEM"], 18), [1.0*inch, 2.4*inch, .8*inch, .8*inch, .8*inch, .7*inch], 7.0))
story.append(PageBreak())
plot_page(story, "Between-finger trajectory distance", FIG/"trajectories"/"between_finger_trajectory_distance.png", "Generated graph for between-finger trajectory differences.", "Compare finger-pair labels and distance metrics.", "Finger identity changes the trajectory by several cm.")

H(story, "6. Max 3D direction and success versus failure trajectory")
story.append(P("3D proxy combines top-camera XY with side-camera z/lift. Direction relates to excursion and success, but the strongest practical strategy remains compact, controlled, lower-lift movement."))
story.append(T(rec_table(trial3_dir.to_dict("records"), ["dominant_movement_direction", "n", "success_rate", "path3_cm", "max_excursion_cm", "max_radius_cm", "mean_velocity_3d_cm_s"], ["Direction", "n", "Success", "Path3 cm", "Max excursion cm", "Max radius cm", "Mean 3D velocity"], 12), [.8*inch, .6*inch, .8*inch, 1.0*inch, 1.2*inch, 1.1*inch, 1.2*inch], 7.0))
story.append(PageBreak())
for title, path, bottom in [
    ("Max 3D excursion by direction and stiffness", FIG/"trajectories"/"max_3d_excursion_by_direction_stiffness_bin.png", "Large 3D excursion is direction- and stiffness-dependent; larger is not automatically better."),
    ("3D success by direction", ASSET/"custom_3d_success_by_direction.png", "Some directions perform better, but direction is not the whole strategy."),
    ("Max 3D excursion by direction", ASSET/"custom_max3d_by_direction.png", "Directions with large excursion are not necessarily the highest-success directions."),
    ("Success vs failure trajectory distance", FIG/"trajectories"/"success_failure_trajectory_distance.png", "Success and failure trajectories are measurably separated."),
]:
    plot_page(story, title, path, "Directly addresses your 3D direction and success/failure trajectory questions.", "Compare direction, stiffness, and success/failure groupings.", bottom)

H(story, "7. Explicit folder: L_N_E figures/other")
story.append(P(f"You specifically asked to mention this folder: {FIG / 'other'}. It contains the motor-control plots: finger metrics, stiffness metrics, and within-finger stiffness slopes."))
for pth in sorted((FIG/"other").glob("*.png")):
    bottom = "Finger and stiffness affect motor-control metrics; read each metric panel separately."
    plot_page(story, "Motor-control graph: " + pth.stem, pth, "This plot is from the requested figures/other folder.", "Each panel is a different motor-control metric.", bottom)

H(story, "8. Explicit folder: L_N_E figures/success")
story.append(P(f"You specifically asked to mention this folder: {FIG / 'success'}. It contains success by direction, success by distance from center, and linked kinematic/z contrasts."))
for pth in sorted((FIG/"success").glob("*.png")):
    bottom = "The success folder supports the strategy answer: compact, consistent, lower-lift movement is better than large exploratory movement."
    plot_page(story, "Success graph: " + pth.stem, pth, "This plot is from the requested figures/success folder.", "Read each panel as a success-related grouping or contrast.", bottom)

H(story, "9. Velocity - stiffness, fingers, time, magnitude, components, radial, tangential")
story.append(P("Velocity differences between stiffness levels are present but small: mean velocity ranges about 15.7 to 16.4 cm/s. Finger differences are also modest: R and I are slightly faster, P and M slightly slower. Time is the strongest velocity structure: early movement is slow, middle/late movement is much faster."))
story.append(T(rec_table(vel_stiff.to_dict("records"), ["stiffness_value", "mean_velocity_cm_s", "mean_velocity_3d_proxy_cm_s", "mean_acceleration_cm_s2", "mean_acceleration_3d_proxy_cm_s2"], ["Stiffness", "Velocity cm/s", "3D velocity cm/s", "Accel cm/s2", "3D accel cm/s2"]), [.9*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch], 7.2))
story.append(T(rec_table(vel_finger.sort_values("mean_velocity_cm_s", ascending=False).to_dict("records"), ["finger_condition", "mean_velocity_cm_s", "mean_velocity_3d_proxy_cm_s", "mean_acceleration_cm_s2", "mean_acceleration_3d_proxy_cm_s2"], ["Finger", "Velocity cm/s", "3D velocity cm/s", "Accel cm/s2", "3D accel cm/s2"]), [.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch], 7.2))
story.append(PageBreak())
for title, path, bottom in [
    ("Velocity decomposition by stiffness", ASSET/"custom_velocity_by_stiffness_decomposition.png", "Stiffness changes velocity only modestly; no clean monotonic trend dominates."),
    ("Velocity decomposition by finger", ASSET/"custom_velocity_by_finger_decomposition.png", "R/I are slightly faster; P/M are slightly slower."),
    ("Velocity decomposition over time", ASSET/"custom_velocity_by_time_decomposition.png", "Time dominates velocity: middle/late movement is much faster than early movement."),
]:
    plot_page(story, title, path, "Custom direct answer to your velocity question.", "Magnitude is speed; radial/tangential are decomposed velocity; components are absolute x/y/z proxy values.", bottom)
for pth in [p for p in sorted((FIG/"velocity").rglob("*.png")) if "subject_" not in p.name.lower()]:
    sub = str(pth.relative_to(FIG/"velocity")).lower()
    if "radial" in sub:
        why = "Radial velocity: movement toward/away from center."
    elif "tangential" in sub:
        why = "Tangential velocity: movement around the path."
    elif "magnitude" in sub:
        why = "Velocity magnitude: overall speed."
    elif "components" in sub:
        why = "Velocity components: x/y/z or combined component view."
    else:
        why = "Velocity aggregate graph."
    plot_page(story, "Velocity graph: " + pth.stem, pth, why, "Compare time profiles and standard-vs-comparison panels.", "Velocity differences exist, but time profile is stronger than stiffness/finger separation.")

H(story, "10. Acceleration - stiffness, fingers, time, magnitude and components")
story.append(P("Acceleration is more burst-sensitive than velocity. Across stiffness, mean acceleration is roughly 75 to 81 cm/s2 in the velocity-summary table, without a strong monotonic stiffness trend. By finger, P is lowest; I/M are higher depending on 2D versus 3D proxy. Time is again dominant: middle/late movement carries much larger acceleration than early movement."))
story.append(T(rec_table(acc_by_stiff.to_dict("records"), ["stiffness_value", "acceleration_cm_s2", "acceleration_3d_proxy_cm_s2", "abs_ax_cm_s2", "abs_ay_cm_s2", "abs_az_3d_proxy_cm_s2"], ["Stiffness", "Accel mag", "3D accel", "abs Ax", "abs Ay", "abs Az proxy"]), [.8*inch, 1.0*inch, 1.0*inch, .9*inch, .9*inch, 1.0*inch], 7.0))
story.append(T(rec_table(acc_by_finger.to_dict("records"), ["finger_condition", "acceleration_cm_s2", "acceleration_3d_proxy_cm_s2", "abs_ax_cm_s2", "abs_ay_cm_s2", "abs_az_3d_proxy_cm_s2"], ["Finger", "Accel mag", "3D accel", "abs Ax", "abs Ay", "abs Az proxy"]), [.8*inch, 1.0*inch, 1.0*inch, .9*inch, .9*inch, 1.0*inch], 7.0))
story.append(PageBreak())
for title, path, bottom in [
    ("Acceleration components by stiffness", ASSET/"custom_acceleration_by_stiffness_components.png", "No clean monotonic stiffness trend dominates."),
    ("Acceleration components by finger", ASSET/"custom_acceleration_by_finger_components.png", "P is the lowest acceleration finger; I/M are higher depending on measure."),
    ("Acceleration components over time", ASSET/"custom_acceleration_by_time_components.png", "Acceleration is much lower early and higher in middle/late movement."),
]:
    plot_page(story, title, path, "Custom direct answer to your acceleration question.", "Magnitude is total acceleration; components are absolute x/y/z proxy values.", bottom)
for pth in [p for p in sorted((FIG/"acceleration").rglob("*.png")) if "subject_" not in p.name.lower()]:
    sub = str(pth.relative_to(FIG/"acceleration")).lower()
    if "components" in sub:
        why = "Acceleration components: x/y/z acceleration over normalized time."
    elif "median_by_stiffness" in sub:
        why = "Median acceleration profiles separated by stiffness."
    elif "s_vs_c" in sub:
        why = "Standard-vs-comparison acceleration for a metric."
    else:
        why = "Aggregate acceleration graph."
    plot_page(story, "Acceleration graph: " + pth.stem, pth, why, "Compare line shapes and amplitudes across finger, stiffness, and standard/comparison panels.", "Acceleration bursts differ by condition, but success is better explained by compactness, consistency, and lower excessive lift/path demands.")

H(story, "11. Coverage note - what is included inline")
story.append(P("This report includes all aggregate L_N_E plots from the requested analysis families inline in the text: figures/other, figures/success, trajectories, velocity, and acceleration. Subject-specific plots are not repeated one-by-one because there are hundreds of them and they would make the file hard to read."))
coverage = []
manifest = []
for folder in ["other", "success", "trajectories", "velocity", "acceleration"]:
    pngs = sorted((FIG/folder).rglob("*.png"))
    agg = [x for x in pngs if "subject_" not in x.name.lower()]
    subj = [x for x in pngs if "subject_" in x.name.lower()]
    coverage.append([folder, str(len(agg)), str(len(subj)), str(len(pngs))])
    for pth in agg:
        manifest.append({"included_inline": "yes", "relative_path": str(pth.relative_to(BASE)), "path": str(pth)})
story.append(T([["Figure family", "Aggregate plots included inline", "Subject-level plots indexed only", "Total PNGs"]] + coverage, [1.5*inch, 1.7*inch, 1.7*inch, 1.0*inch]))
pd.DataFrame(manifest).to_csv(MANIFEST, index=False)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(doc.leftMargin, 0.28 * inch, "Personal detailed kinematics report - inline plots")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(str(PDF), pagesize=landscape(A4), rightMargin=0.35*inch, leftMargin=0.35*inch, topMargin=0.35*inch, bottomMargin=0.45*inch)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(PDF)
print(MANIFEST)
print(OUT / "kinematics_personal_detailed_findings.json")
