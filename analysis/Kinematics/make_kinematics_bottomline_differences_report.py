from pathlib import Path
import math, json
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
OUT = ROOT / "output" / "pdf"
ASSET = OUT / "bottomline_differences_assets"
OUT.mkdir(parents=True, exist_ok=True)
ASSET.mkdir(parents=True, exist_ok=True)
PDF = OUT / "kinematics_result_fillter_BOTTOMLINE_DIFFERENCES_report.pdf"
FINDINGS = OUT / "kinematics_bottomline_differences_findings.json"


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


def save_bar(df, x, y, title, ylabel, path, color=None, rotation=0):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_grouped_bar(df, x, ys, title, ylabel, path, rotation=0):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    idx = np.arange(len(df))
    width = 0.8 / len(ys)
    for i, y in enumerate(ys):
        ax.bar(idx + (i - (len(ys)-1)/2)*width, df[y], width, label=y)
    ax.set_xticks(idx)
    ax.set_xticklabels(df[x].astype(str), rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_lines(df, x, y, hue, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for key, sub in df.groupby(hue):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", label=str(key))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_heat(df, index, columns, values, title, path, fmt_digits=2):
    piv = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    im = ax.imshow(piv.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(i) for i in piv.index])
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            ax.text(j, i, f"{val:.{fmt_digits}f}" if pd.notna(val) else "", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# Load outputs.
kg = pd.read_csv(CSV / "other" / "kinematic_group_metric_summary.csv")
kgc = pd.read_csv(CSV / "other" / "kinematic_group_condition_metric_summary.csv")
between = pd.read_csv(CSV / "other" / "kinematic_between_group_metric_comparisons.csv")
within = pd.read_csv(CSV / "other" / "kinematic_within_group_condition_comparisons.csv")
dir_success = pd.read_csv(CSV / "success" / "direction_success_summary.csv")
trial3 = pd.read_csv(CSV / "trajectories" / "trial_3d_kinematic_summary.csv")
profile = pd.read_csv(CSV / "velocity" / "others" / "subject_velocity_acceleration_profile.csv")
finger_dist = pd.read_csv(CSV / "trajectories" / "finger_trajectory_distance_summary.csv")
sftraj = pd.read_csv(CSV / "trajectories" / "success_failure_trajectory_distance_summary.csv")
zstiff = pd.read_csv(CSV / "z_lift" / "side_z_by_stiffness_summary.csv")


def ensure_experiment_group(df):
    """Normalize group labels across CSVs produced by different analysis functions."""
    if "experiment_group" in df.columns:
        return df
    if "subject_group" in df.columns:
        df = df.copy()
        df["experiment_group"] = df["subject_group"].map({"L": "L_E", "N": "N_E"}).fillna(df["subject_group"])
        return df
    if "subject_id" in df.columns:
        df = df.copy()
        s = df["subject_id"].astype(str)
        df["experiment_group"] = np.where(s.str.startswith("L_E"), "L_E",
                                           np.where(s.str.startswith("N_E"), "N_E", "unknown"))
        return df
    return df


trial3 = ensure_experiment_group(trial3)
profile = ensure_experiment_group(profile)

profile["time_third"] = pd.cut(profile["time_fraction"], [-0.001, 1/3, 2/3, 1.001], labels=["early", "middle", "late"])
for c in ["radial_velocity_cm_s", "tangential_velocity_cm_s", "vx_cm_s", "vy_cm_s", "vz_3d_proxy_cm_s",
          "ax_cm_s2", "ay_cm_s2", "az_3d_proxy_cm_s2"]:
    if c in profile:
        profile["abs_" + c] = profile[c].abs()

metric_labels = {
    "success_rate": "success",
    "mean_speed_cm_s": "speed",
    "mean_acceleration_cm_s2": "acceleration",
    "mean_path_length_cm": "path length",
    "mean_max_r_workspace_cm": "max radius",
    "mean_r_workspace_cm": "mean radius",
    "mean_curvature_1_cm": "curvature",
    "mean_jerk_cm_s3": "jerk",
}
key_metrics = list(metric_labels.keys())

# Main between-group differences.
all_between = between[between.condition_col.eq("all")].copy()
all_between["abs_d"] = all_between["cohens_d_b_minus_a"].abs()
top_between = all_between.sort_values("abs_d", ascending=False).head(16)

finger_between = between[between.condition_col.eq("finger_condition")].copy()
finger_between["abs_d"] = finger_between["cohens_d_b_minus_a"].abs()
top_finger_between = finger_between.sort_values("abs_d", ascending=False).head(18)

stiff_between = between[between.condition_col.eq("stiffness_value")].copy()
stiff_between["abs_d"] = stiff_between["cohens_d_b_minus_a"].abs()
top_stiff_between = stiff_between.sort_values("abs_d", ascending=False).head(18)

within_finger = within[within.condition_col.eq("finger_condition")].copy()
within_finger["abs_d"] = within_finger["cohens_d_b_minus_a"].abs()
within_stiff = within[within.condition_col.eq("stiffness_value")].copy()
within_stiff["abs_d"] = within_stiff["cohens_d_b_minus_a"].abs()

# Direction summaries.
dir_group = dir_success.groupby(["experiment_group", "dominant_movement_direction"]).agg(
    n_trials=("n_trials", "sum"),
    success_rate=("success_rate", "mean"),
    speed_cm_s=("mean_speed_cm_s", "mean"),
    path_cm=("mean_path_length_cm", "mean"),
    radius_cm=("mean_r_workspace_cm", "mean"),
).reset_index()
dir_piv = dir_group.pivot(index="dominant_movement_direction", columns="experiment_group", values="success_rate").reset_index()
if {"L_E", "N_E"}.issubset(dir_piv.columns):
    dir_piv["L_E_minus_N_E_success"] = dir_piv["L_E"] - dir_piv["N_E"]
dir3 = trial3.groupby(["experiment_group", "dominant_movement_direction"]).agg(
    n=("correct_response", "size"),
    success=("correct_response", "mean"),
    max3d_cm=("max_excursion_3d_from_start_cm", "mean"),
    path3d_cm=("path_length_3d_proxy_cm", "mean"),
    vel3d_cm_s=("mean_velocity_3d_proxy_cm_s", "mean"),
    acc3d_cm_s2=("mean_acceleration_3d_proxy_cm_s2", "mean"),
).reset_index()

# Trajectory / 3D by group/finger/stiffness.
traj_finger = trial3.groupby(["experiment_group", "finger_condition"]).agg(
    n=("correct_response", "size"),
    success=("correct_response", "mean"),
    path3d_cm=("path_length_3d_proxy_cm", "mean"),
    max3d_cm=("max_excursion_3d_from_start_cm", "mean"),
    radius3d_cm=("max_radius_3d_from_center_cm", "mean"),
    straightness=("straightness_3d_proxy_cm", "mean"),
).reset_index()
traj_stiff = trial3.groupby(["experiment_group", "stiffness_value"]).agg(
    n=("correct_response", "size"),
    success=("correct_response", "mean"),
    path3d_cm=("path_length_3d_proxy_cm", "mean"),
    max3d_cm=("max_excursion_3d_from_start_cm", "mean"),
    radius3d_cm=("max_radius_3d_from_center_cm", "mean"),
    straightness=("straightness_3d_proxy_cm", "mean"),
).reset_index()

# Velocity / acceleration by group/finger/stiffness/time.
vel_cols = ["speed_cm_s", "speed_3d_proxy_cm_s", "abs_radial_velocity_cm_s", "abs_tangential_velocity_cm_s", "abs_vx_cm_s", "abs_vy_cm_s", "abs_vz_3d_proxy_cm_s"]
acc_cols = ["acceleration_cm_s2", "acceleration_3d_proxy_cm_s2", "abs_ax_cm_s2", "abs_ay_cm_s2", "abs_az_3d_proxy_cm_s2"]
va_finger = profile.groupby(["experiment_group", "finger_condition"])[vel_cols + acc_cols].mean().reset_index()
va_stiff = profile.groupby(["experiment_group", "stiffness_value"])[vel_cols + acc_cols].mean().reset_index()
va_time = profile.groupby(["experiment_group", "time_third"], observed=True)[vel_cols + acc_cols].mean().reset_index()

def diff_wide(df, idx, metrics):
    out = []
    for metric in metrics:
        p = df.pivot_table(index=idx, columns="experiment_group", values=metric, aggfunc="mean").reset_index()
        if {"L_E", "N_E"}.issubset(p.columns):
            p["metric"] = metric
            p["L_E_minus_N_E"] = p["L_E"] - p["N_E"]
            out.append(p)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

traj_finger_diff = diff_wide(traj_finger, "finger_condition", ["success", "path3d_cm", "max3d_cm", "straightness"])
traj_stiff_diff = diff_wide(traj_stiff, "stiffness_value", ["success", "path3d_cm", "max3d_cm", "straightness"])
va_finger_diff = diff_wide(va_finger, "finger_condition", ["speed_cm_s", "speed_3d_proxy_cm_s", "abs_radial_velocity_cm_s", "abs_tangential_velocity_cm_s", "acceleration_cm_s2", "acceleration_3d_proxy_cm_s2"])
va_stiff_diff = diff_wide(va_stiff, "stiffness_value", ["speed_cm_s", "speed_3d_proxy_cm_s", "abs_radial_velocity_cm_s", "abs_tangential_velocity_cm_s", "acceleration_cm_s2", "acceleration_3d_proxy_cm_s2"])

# Save reader plots.
save_bar(top_between.assign(label=top_between["metric"].map(metric_labels).fillna(top_between["metric"])), "label", "cohens_d_b_minus_a", "Between groups: strongest all-condition effects (L_E - N_E)", "Cohen d", ASSET/"between_groups_top_effects.png", rotation=45)
save_heat(traj_finger, "experiment_group", "finger_condition", "max3d_cm", "Max 3D excursion by group and finger", ASSET/"traj_max3d_group_finger.png")
save_lines(traj_stiff, "stiffness_value", "max3d_cm", "experiment_group", "Max 3D excursion by stiffness and group", "cm", ASSET/"traj_max3d_group_stiffness.png")
save_heat(traj_finger, "experiment_group", "finger_condition", "path3d_cm", "3D path length by group and finger", ASSET/"traj_path_group_finger.png")
save_lines(traj_stiff, "stiffness_value", "path3d_cm", "experiment_group", "3D path length by stiffness and group", "cm", ASSET/"traj_path_group_stiffness.png")
save_bar(dir_piv.sort_values("L_E_minus_N_E_success", ascending=False), "dominant_movement_direction", "L_E_minus_N_E_success", "Direction success difference: L_E - N_E", "success difference", ASSET/"direction_success_diff.png", rotation=45)
save_heat(dir3, "experiment_group", "dominant_movement_direction", "max3d_cm", "Direction: max 3D excursion by group", ASSET/"direction_max3d_group.png")
save_heat(va_finger, "experiment_group", "finger_condition", "speed_cm_s", "Velocity magnitude by group and finger", ASSET/"velocity_group_finger.png")
save_lines(va_stiff, "stiffness_value", "speed_cm_s", "experiment_group", "Velocity magnitude by stiffness and group", "cm/s", ASSET/"velocity_group_stiffness.png")
save_heat(va_finger, "experiment_group", "finger_condition", "abs_radial_velocity_cm_s", "Radial velocity by group and finger", ASSET/"radial_velocity_group_finger.png")
save_heat(va_finger, "experiment_group", "finger_condition", "abs_tangential_velocity_cm_s", "Tangential velocity by group and finger", ASSET/"tangential_velocity_group_finger.png")
save_heat(va_finger, "experiment_group", "finger_condition", "acceleration_cm_s2", "Acceleration magnitude by group and finger", ASSET/"accel_group_finger.png")
save_lines(va_stiff, "stiffness_value", "acceleration_cm_s2", "experiment_group", "Acceleration magnitude by stiffness and group", "cm/s2", ASSET/"accel_group_stiffness.png")
save_heat(va_finger, "experiment_group", "finger_condition", "acceleration_3d_proxy_cm_s2", "3D proxy acceleration by group and finger", ASSET/"accel3d_group_finger.png")
save_grouped_bar(va_time, "time_third", ["speed_cm_s", "acceleration_cm_s2"], "Velocity and acceleration by movement time third", "mean value", ASSET/"velocity_accel_time.png")

# JSON findings for traceability.
json_out = {
    "top_between_groups_all": top_between.to_dict("records"),
    "top_between_groups_by_finger": top_finger_between.to_dict("records"),
    "top_between_groups_by_stiffness": top_stiff_between.to_dict("records"),
    "top_within_group_finger": within_finger.sort_values("abs_d", ascending=False).head(30).to_dict("records"),
    "top_within_group_stiffness": within_stiff.sort_values("abs_d", ascending=False).head(30).to_dict("records"),
    "direction_success_diff": dir_piv.to_dict("records"),
    "trajectory_by_finger_group": traj_finger.to_dict("records"),
    "trajectory_by_stiffness_group": traj_stiff.to_dict("records"),
    "velocity_accel_by_finger_group": va_finger.to_dict("records"),
    "velocity_accel_by_stiffness_group": va_stiff.to_dict("records"),
}
FINDINGS.write_text(json.dumps(json_out, indent=2), encoding="utf-8")

# PDF styles.
styles = getSampleStyleSheet()
def add_style(name, parent, size, leading, color="#000000", align=None, left=0, first=0):
    kw = dict(name=name, parent=styles[parent], fontSize=size, leading=leading, spaceAfter=5, textColor=colors.HexColor(color), leftIndent=left, firstLineIndent=first)
    if align is not None:
        kw["alignment"] = align
    styles.add(ParagraphStyle(**kw))

add_style("TitleX", "Title", 24, 29, "#15395b", TA_CENTER)
add_style("SubX", "Heading2", 15, 19, "#1f4e79", TA_CENTER)
add_style("H1X", "Heading1", 17, 21, "#1f4e79")
add_style("H2X", "Heading2", 13.5, 17, "#365f91")
add_style("BodyX", "BodyText", 10.2, 13.6)
add_style("SmallX", "BodyText", 8.1, 10.3, "#333333")
add_style("TinyX", "BodyText", 7.0, 8.4, "#333333")
add_style("BulletX", "BodyText", 10.2, 13.6, left=14, first=-8)

def clean(s):
    return str(s).translate({ord("\u2013"): "-", ord("\u2014"): "-", ord("\u2212"): "-", ord("\u2011"): "-", ord("\u00b1"): "+/-"}).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def P(s, style="BodyX"):
    return Paragraph(clean(s), styles[style])

def H(story, s, level=1):
    story.append(P(s, "H1X" if level == 1 else "H2X"))

def T(rows, widths=None, font=8.0, header=True):
    out = []
    for row in rows:
        out.append([c if hasattr(c, "wrap") else Paragraph(clean(c), styles["TinyX" if font < 7.8 else "SmallX"]) for c in row])
    t = Table(out, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    ts = [("VALIGN", (0,0), (-1,-1), "TOP"), ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#d5dbe3")), ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4), ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]
    if header:
        ts.append(("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dcecf8")))
    t.setStyle(TableStyle(ts))
    return t

def image(path, max_w=5.5*inch, max_h=3.7*inch):
    path = Path(path)
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w/w, max_h/h)
    return Image(str(path), width=w*scale, height=h*scale)

def rec_table(records, cols, labels, n=12):
    rows = [labels]
    for r in records[:n]:
        rows.append([fmt(r.get(c, "")) if isinstance(r.get(c, ""), (int, float, np.floating)) else str(r.get(c, "")) for c in cols])
    return rows

def plot_section(story, title, path, bottom):
    H(story, title, 2)
    story.append(T([[image(path, 5.7*inch, 3.9*inch), P(bottom, "SmallX")]], [5.9*inch, 3.2*inch], header=False))
    story.append(PageBreak())

story = []
story += [
    Spacer(1, .2*inch),
    P("Bottom-Line Difference Map", "TitleX"),
    P("Where differences appear by finger, stiffness, group, direction, trajectory, velocity, and acceleration", "SubX"),
    P(f"Source: {BASE}", "SmallX"),
    PageBreak(),
]

H(story, "1. Executive bottom lines")
bullets = [
    "Biggest between-group differences are spatial/trajectory scale: L_E is larger/farther/longer; N_E is smaller and more curved.",
    "Finger differences are real, but they are metric-specific. Fingers differ in trajectory geometry and hand/movement dynamics more than in simple success alone.",
    "Stiffness differences exist, but many velocity and acceleration effects are not monotonic across stiffness. Stiffness is weaker than the L_E vs N_E spatial setup difference.",
    "Direction differences exist in success and max-3D excursion. However, high excursion direction is not automatically the best success direction.",
    "Velocity: group/finger/stiffness differences are present but moderate. Time within the movement is the strongest velocity pattern.",
    "Acceleration: P tends to be lower in acceleration; acceleration varies by stiffness but without one clean increasing trend. Middle/late movement phases dominate acceleration.",
]
for b in bullets:
    story.append(P("- " + b, "BulletX"))
story.append(PageBreak())

H(story, "1A. Direct answer map: where the differences are")
story.append(P("This page is the quick bottom line. It separates differences between groups from differences between fingers and stiffness values inside the groups."))
answer_map = [
    ["Domain", "Between L_E and N_E", "Between fingers", "Between stiffness values", "Bottom line"],
    ["Direction", "Yes, but mixed by direction; some directions favor L_E and others favor N_E.", "Direction is tied to trajectory geometry, but finger-level direction effects are secondary here.", "Not the strongest stiffness story.", "Direction matters, but it is not a single success strategy."],
    ["Trajectory", "Strong. L_E has larger path, radius, max excursion, and jerk; N_E is more compact/curved.", "Clear. Fingers differ in path length and max 3D excursion.", "Present but not cleanly monotonic.", "This is the clearest difference domain."],
    ["Velocity", "Moderate. L_E is generally faster, especially radial/tangential components.", "Present but modest; I/R often higher than P/M.", "Weak-to-moderate; no clean increasing/decreasing stiffness law.", "Velocity changes exist, but they are smaller than trajectory scale."],
    ["Acceleration", "Moderate and burst-sensitive; L_E often higher.", "Present; P is often lower, I/M/R depend on component.", "Present but irregular across stiffness.", "Acceleration differences support a control/burstiness interpretation."],
    ["Success strategy", "N_E slightly higher success overall despite smaller movements.", "Success is not explained by one finger alone.", "Some stiffness values have lower success, but not through velocity alone.", "Better success aligns with compact, controlled, lower-z-lift, lower-excursion trajectories rather than simply going faster."],
]
story.append(T(answer_map, [1.0*inch, 2.2*inch, 2.0*inch, 1.8*inch, 2.2*inch], 7.2))
story.append(PageBreak())

H(story, "2. Between groups: L_E vs N_E")
story.append(P("Positive L_E - N_E means L_E is higher. Negative means N_E is higher. The strongest all-condition differences show where groups truly separate."))
story.append(T(rec_table(top_between.to_dict("records"), ["metric", "mean_a", "mean_b", "mean_difference_b_minus_a", "cohens_d_b_minus_a"], ["Metric", "N_E mean", "L_E mean", "L_E - N_E", "Cohen d"], 16), [2.3*inch, 1.0*inch, 1.0*inch, 1.1*inch, .9*inch], 7.6))
story.append(PageBreak())
plot_section(story, "Between-group effect sizes", ASSET/"between_groups_top_effects.png", "Where groups differ most: max radius, mean radius, path scale, jerk, curvature, and speed. L_E is larger/faster/jerkier; N_E is more curved and slightly more successful.")

H(story, "3. Between groups by finger")
story.append(P("This table asks: for a given finger, where do L_E and N_E differ most?"))
story.append(T(rec_table(top_finger_between.to_dict("records"), ["condition_level", "metric", "mean_a", "mean_b", "mean_difference_b_minus_a", "cohens_d_b_minus_a"], ["Finger", "Metric", "N_E mean", "L_E mean", "L_E - N_E", "d"], 18), [.75*inch, 2.1*inch, .95*inch, .95*inch, 1.0*inch, .65*inch], 7.2))
story.append(PageBreak())

H(story, "4. Between groups by stiffness")
story.append(P("This table asks: for a given stiffness value, where do L_E and N_E differ most?"))
story.append(T(rec_table(top_stiff_between.to_dict("records"), ["condition_level", "metric", "mean_a", "mean_b", "mean_difference_b_minus_a", "cohens_d_b_minus_a"], ["Stiffness", "Metric", "N_E mean", "L_E mean", "L_E - N_E", "d"], 18), [.8*inch, 2.1*inch, .95*inch, .95*inch, 1.0*inch, .65*inch], 7.2))
story.append(PageBreak())

H(story, "5. Within-group differences: fingers and stiffness")
story.append(P("These are not L_E vs N_E comparisons. They show where conditions differ inside each group. Large absolute d means a stronger within-group difference."))
story.append(T(rec_table(within_finger.sort_values("abs_d", ascending=False).to_dict("records"), ["experiment_group", "level_a", "level_b", "metric", "mean_difference_b_minus_a", "cohens_d_b_minus_a"], ["Group", "A", "B", "Metric", "B - A", "d"], 16), [.75*inch, .55*inch, .55*inch, 2.3*inch, .9*inch, .65*inch], 7.2))
story.append(Spacer(1, .1*inch))
story.append(T(rec_table(within_stiff.sort_values("abs_d", ascending=False).to_dict("records"), ["experiment_group", "level_a", "level_b", "metric", "mean_difference_b_minus_a", "cohens_d_b_minus_a"], ["Group", "A", "B", "Metric", "B - A", "d"], 16), [.75*inch, .65*inch, .65*inch, 2.2*inch, .9*inch, .65*inch], 7.2))
story.append(PageBreak())

H(story, "6. Direction differences")
story.append(P("Direction matters in success and 3D excursion, but it is not the only strategy. The table and plots below show direction differences per group and between groups."))
story.append(T(rec_table(dir_piv.sort_values("L_E_minus_N_E_success", ascending=False).to_dict("records"), ["dominant_movement_direction", "L_E", "N_E", "L_E_minus_N_E_success"], ["Direction", "L_E success", "N_E success", "L_E - N_E"], 16), [1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch], 7.4))
story.append(PageBreak())
plot_section(story, "Direction success difference", ASSET/"direction_success_diff.png", "Direction-specific success differences are mixed: some directions favor L_E and some favor N_E. Direction matters, but it does not explain the whole L_E/N_E difference.")
plot_section(story, "Direction max 3D excursion by group", ASSET/"direction_max3d_group.png", "Direction changes max 3D excursion. L_E generally has larger excursions, but high excursion does not automatically mean higher success.")

H(story, "7. Trajectory differences")
story.append(P("Trajectory differences are one of the strongest domains. L_E tends to show larger path/excursion/radius than N_E. Finger and stiffness also change trajectory geometry."))
story.append(T(rec_table(traj_finger.to_dict("records"), ["experiment_group", "finger_condition", "success", "path3d_cm", "max3d_cm", "straightness"], ["Group", "Finger", "Success", "Path3D", "Max3D", "Straightness"], 12), [.8*inch, .7*inch, .8*inch, 1.0*inch, 1.0*inch, 1.0*inch], 7.4))
story.append(T(rec_table(finger_dist.sort_values("mean", ascending=False).to_dict("records"), ["comparison", "metric", "mean", "median", "sem"], ["Finger pair", "Trajectory metric", "Mean", "Median", "SEM"], 10), [1.0*inch, 2.6*inch, .8*inch, .8*inch, .7*inch], 7.2))
story.append(PageBreak())
plot_section(story, "Trajectory max 3D by group and finger", ASSET/"traj_max3d_group_finger.png", "Finger differences are visible in max 3D excursion, but the group difference is still the larger pattern.")
plot_section(story, "Trajectory max 3D by group and stiffness", ASSET/"traj_max3d_group_stiffness.png", "Stiffness changes max 3D excursion, but the L_E vs N_E separation remains visible across stiffness.")
plot_section(story, "Trajectory path length by group and finger", ASSET/"traj_path_group_finger.png", "Finger changes path length; L_E tends to have longer 3D paths than N_E.")
plot_section(story, "Trajectory path length by group and stiffness", ASSET/"traj_path_group_stiffness.png", "Stiffness changes trajectory length, but not in a simple monotonic way.")

H(story, "8. Success/failure trajectory bottom line")
story.append(P("Success/failure trajectory distance tells us how far successful and failed movement profiles separate. Larger numbers mean success and failure trajectories are more different for that finger."))
story.append(T(rec_table(sftraj.to_dict("records"), ["finger_condition", "metric", "n_subject_stiffness_pairs", "mean", "median", "sem"], ["Finger", "Metric", "N", "Mean", "Median", "SEM"], 12), [.7*inch, 2.6*inch, .7*inch, .8*inch, .8*inch, .7*inch], 7.2))
story.append(PageBreak())

H(story, "9. Velocity differences")
story.append(P("Velocity differences between fingers/stiffness/groups exist, but they are moderate compared with trajectory scale. Radial and tangential velocity decompose how motion is directed, not just how fast it is."))
story.append(T(rec_table(va_finger.to_dict("records"), ["experiment_group", "finger_condition", "speed_cm_s", "speed_3d_proxy_cm_s", "abs_radial_velocity_cm_s", "abs_tangential_velocity_cm_s"], ["Group", "Finger", "Speed", "3D speed", "Radial abs", "Tangential abs"], 10), [.8*inch, .7*inch, .8*inch, .9*inch, 1.0*inch, 1.0*inch], 7.2))
story.append(PageBreak())
H(story, "9A. Velocity: explicit L_E - N_E differences")
story.append(P("These tables show exactly where L_E and N_E differ for velocity components. Positive values mean L_E is higher; negative values mean N_E is higher."))
story.append(T(rec_table(va_finger_diff[va_finger_diff["metric"].str.contains("speed|velocity", regex=True)].assign(absdiff=lambda d: d["L_E_minus_N_E"].abs()).sort_values("absdiff", ascending=False).to_dict("records"), ["finger_condition", "metric", "N_E", "L_E", "L_E_minus_N_E"], ["Finger", "Velocity metric", "N_E", "L_E", "L_E - N_E"], 16), [.75*inch, 2.2*inch, .9*inch, .9*inch, 1.0*inch], 7.1))
story.append(Spacer(1, .1*inch))
story.append(T(rec_table(va_stiff_diff[va_stiff_diff["metric"].str.contains("speed|velocity", regex=True)].assign(absdiff=lambda d: d["L_E_minus_N_E"].abs()).sort_values("absdiff", ascending=False).to_dict("records"), ["stiffness_value", "metric", "N_E", "L_E", "L_E_minus_N_E"], ["Stiffness", "Velocity metric", "N_E", "L_E", "L_E - N_E"], 16), [.75*inch, 2.2*inch, .9*inch, .9*inch, 1.0*inch], 7.1))
story.append(PageBreak())
plot_section(story, "Velocity by group and finger", ASSET/"velocity_group_finger.png", "Finger velocity differences are modest. R/I are often slightly higher than P/M, but the separation is not huge.")
plot_section(story, "Velocity by group and stiffness", ASSET/"velocity_group_stiffness.png", "Velocity varies across stiffness, but not as a clean increasing/decreasing stiffness trend.")
plot_section(story, "Radial velocity by group and finger", ASSET/"radial_velocity_group_finger.png", "Radial velocity differs by finger/group, showing differences in inward/outward movement strategy.")
plot_section(story, "Tangential velocity by group and finger", ASSET/"tangential_velocity_group_finger.png", "Tangential velocity differs by finger/group, showing differences in around-path movement.")

H(story, "10. Acceleration differences")
story.append(P("Acceleration is more burst-sensitive than velocity. Differences appear by finger and stiffness, but again stiffness is not a clean monotonic story. P tends to be lower in acceleration; I/M/R vary depending on 2D vs 3D proxy."))
story.append(T(rec_table(va_finger.to_dict("records"), ["experiment_group", "finger_condition", "acceleration_cm_s2", "acceleration_3d_proxy_cm_s2", "abs_ax_cm_s2", "abs_ay_cm_s2", "abs_az_3d_proxy_cm_s2"], ["Group", "Finger", "Accel", "3D accel", "abs Ax", "abs Ay", "abs Az"], 10), [.8*inch, .7*inch, .8*inch, .9*inch, .8*inch, .8*inch, .8*inch], 7.0))
story.append(PageBreak())
H(story, "10A. Acceleration: explicit L_E - N_E differences")
story.append(P("These tables show exactly where L_E and N_E differ for acceleration components. Positive values mean L_E is higher; negative values mean N_E is higher."))
story.append(T(rec_table(va_finger_diff[va_finger_diff["metric"].str.contains("acceleration", regex=True)].assign(absdiff=lambda d: d["L_E_minus_N_E"].abs()).sort_values("absdiff", ascending=False).to_dict("records"), ["finger_condition", "metric", "N_E", "L_E", "L_E_minus_N_E"], ["Finger", "Acceleration metric", "N_E", "L_E", "L_E - N_E"], 12), [.75*inch, 2.2*inch, .9*inch, .9*inch, 1.0*inch], 7.1))
story.append(Spacer(1, .1*inch))
story.append(T(rec_table(va_stiff_diff[va_stiff_diff["metric"].str.contains("acceleration", regex=True)].assign(absdiff=lambda d: d["L_E_minus_N_E"].abs()).sort_values("absdiff", ascending=False).to_dict("records"), ["stiffness_value", "metric", "N_E", "L_E", "L_E_minus_N_E"], ["Stiffness", "Acceleration metric", "N_E", "L_E", "L_E - N_E"], 16), [.75*inch, 2.2*inch, .9*inch, .9*inch, 1.0*inch], 7.1))
story.append(PageBreak())
plot_section(story, "Acceleration by group and finger", ASSET/"accel_group_finger.png", "Finger differences are visible; P is generally lower in acceleration magnitude.")
plot_section(story, "Acceleration by group and stiffness", ASSET/"accel_group_stiffness.png", "Acceleration varies across stiffness but does not show a simple monotonic stiffness law.")
plot_section(story, "3D proxy acceleration by group and finger", ASSET/"accel3d_group_finger.png", "3D proxy acceleration includes side z/lift contribution, so it highlights vertical movement dynamics too.")
plot_section(story, "Velocity and acceleration by movement time", ASSET/"velocity_accel_time.png", "The strongest dynamic pattern is time: early movement is low velocity/acceleration; middle and late phases dominate movement dynamics.")

H(story, "11. Z-lift and stiffness note")
story.append(P("Z-lift is not the central requested domain in this report, but it helps interpret trajectory/acceleration. Stiffness changes z-lift moderately; failures had higher z-lift in the previous detailed strategy report."))
story.append(T(rec_table(zstiff.to_dict("records"), ["stiffness_value", "mean_side_z_lift_cm", "max_side_z_lift_cm", "success_rate"], ["Stiffness", "Mean z lift", "Max z lift", "Success"], 9), [1.0*inch, 1.2*inch, 1.2*inch, 1.0*inch], 7.4))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(doc.leftMargin, .28*inch, "Bottom-line differences report")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, .28*inch, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(str(PDF), pagesize=landscape(A4), rightMargin=.35*inch, leftMargin=.35*inch, topMargin=.35*inch, bottomMargin=.45*inch)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(PDF)
print(FINDINGS)
