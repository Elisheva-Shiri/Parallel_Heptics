"""Inferential statistics for the kinematic section (reviewer request).

Reviewer: "Section V-2 is a wall of bare means ... no SDs, no tests, no effect sizes.
Run the t-tests/mixed models, report mean +/- SD, p, and effect size."

Unit of analysis is the participant (one value per participant per metric; the
group means printed in the paper are the means of these per-participant values,
n = 20 vs 20).

Outputs (into <results>/csv/statistics/ by default):
  between_setup_tests.csv   mean/SD/median/IQR per setup, Welch t, Mann-Whitney U,
                            Hedges g with bootstrap 95% CI, Shapiro p, Holm-adjusted p
  finger_within_subject.csv one-way repeated-measures ANOVA (+GG epsilon), Friedman,
                            paired post-hoc (pinky vs other fingers, Holm)
  finger_mixed_anova.csv    setup x finger mixed ANOVA (pingouin, when available)
  stiffness_slopes.csv      per-participant slope vs stiffness: one-sample t, Wilcoxon
  stiffness_mixed_anova.csv setup x stiffness mixed ANOVA (pingouin, when available)
  kinematic_setup_statistics_report.md  ready-to-paste sentences

Run (system python has pingouin; .venv falls back to scipy-only):
  python analysis/Kinematics/kinematic_setup_statistics.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "L_N_E" / "csv"

SETUP_LABEL = {"N_E": "natural", "L_E": "air-slide"}
FINGER_LABEL = {"I": "index", "M": "middle", "R": "ring", "P": "pinky"}

# metric column -> (paper label, unit, decimals)
BETWEEN_METRICS = {
    "mean_path_length_cm": ("path length", "cm", 2),
    "mean_r_workspace_cm": ("mean movement radius", "cm", 2),
    "mean_max_r_workspace_cm": ("maximum movement radius", "cm", 2),
    "mean_jerk_cm_s3": ("jerk", "cm/s^3", 0),
    "mean_curvature_1_cm": ("curvature", "1/cm", 3),
    "mean_speed_cm_s": ("mean speed", "cm/s", 2),
    "mean_acceleration_cm_s2": ("mean acceleration", "cm/s^2", 1),
    "mean_abs_radial_velocity_cm_s": ("mean absolute radial velocity (per-trial medians, mean)", "cm/s", 2),
    "mean_abs_tangential_velocity_cm_s": ("mean absolute tangential velocity (per-trial medians, mean)", "cm/s", 2),
    "median_abs_radial_velocity_cm_s": ("absolute radial velocity (per-participant median over time bins)", "cm/s", 2),
    "median_abs_tangential_velocity_cm_s": ("absolute tangential velocity (per-participant median over time bins)", "cm/s", 2),
    "speed_curvature_power_law_slope": ("speed-curvature power-law slope (beta)", "", 3),
    "speed_curvature_power_law_r2": ("speed-curvature power-law R^2", "", 3),
    "peak_radius_cm_p95": ("95th-percentile movement radius", "cm", 2),
    "mean_radiality_index": ("radiality index |v_r|/(|v_r|+|v_t|)", "", 3),
}
WITHIN_METRICS = ["mean_speed_cm_s", "mean_acceleration_cm_s2", "mean_path_length_cm",
                  "mean_max_r_workspace_cm", "mean_jerk_cm_s3"]


# ----------------------------------------------------------------------------- helpers
def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges g for b - a (positive = b larger), pooled SD, small-sample corrected."""
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    j = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return (b.mean() - a.mean()) / sp * j


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for k in range(n_boot):
        vals[k] = hedges_g(rng.choice(a, len(a)), rng.choice(b, len(b)))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def welch(a: np.ndarray, b: np.ndarray):
    t, p = stats.ttest_ind(b, a, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(df), float(p)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p = n/a"
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".", 1)


def fnum(x: float, d: int) -> str:
    return f"{x:,.{d}f}" if abs(x) >= 1000 else f"{x:.{d}f}"


# ----------------------------------------------------------------------------- loading
def load_per_participant(results_csv: Path) -> pd.DataFrame:
    """One row per participant: mean over stiffness x finger cells (paper convention)."""
    subj = pd.read_csv(results_csv / "other" / "subject_kinematic_summary.csv")
    metric_cols = [c for c in BETWEEN_METRICS if c in subj.columns]
    per = subj.groupby(["subject_id", "experiment_group"], as_index=False)[metric_cols].mean()

    rad_cols = ["mean_abs_radial_velocity_cm_s", "mean_abs_tangential_velocity_cm_s"]
    if not all(c in per.columns for c in rad_cols):
        # absolute radial/tangential speeds live in the trial table (median per trial)
        trial = pd.read_csv(results_csv / "other" / "trial_kinematic_summary.csv",
                            usecols=["subject_id", "experiment_group", "finger_condition",
                                     "stiffness_value"] + rad_cols)
        cell = trial.groupby(["subject_id", "experiment_group", "finger_condition",
                              "stiffness_value"], as_index=False)[rad_cols].median()
        rad = cell.groupby(["subject_id", "experiment_group"], as_index=False)[rad_cols].mean()
        per = per.merge(rad, on=["subject_id", "experiment_group"], how="left")

    tb_path = results_csv / "trajectories" / "trajectory_time_bins.csv"
    if tb_path.exists():
        # robust per-participant location of |v_r| and |v_t| (heavy-tailed within participant)
        tb = pd.read_csv(tb_path, usecols=["subject_id", "radial_velocity_cm_s", "tangential_velocity_cm_s"])
        tb["median_abs_radial_velocity_cm_s"] = tb["radial_velocity_cm_s"].abs()
        tb["median_abs_tangential_velocity_cm_s"] = tb["tangential_velocity_cm_s"].abs()
        med = tb.groupby("subject_id", as_index=False)[
            ["median_abs_radial_velocity_cm_s", "median_abs_tangential_velocity_cm_s"]].median()
        per = per.merge(med, on="subject_id", how="left")

    occ_path = results_csv / "trajectories" / "workspace_occupancy_extent_by_participant.csv"
    if occ_path.exists():
        occ = pd.read_csv(occ_path)[["subject_id", "peak_radius_cm_p95", "mean_radiality_index"]]
        per = per.merge(occ, on="subject_id", how="left")
    return per


def load_cells(results_csv: Path) -> pd.DataFrame:
    subj = pd.read_csv(results_csv / "other" / "subject_kinematic_summary.csv")
    return subj[["subject_id", "experiment_group", "finger_condition", "stiffness_value"]
                + [c for c in WITHIN_METRICS if c in subj.columns]].copy()


# ----------------------------------------------------------------------------- between setup
def between_setup_tests(per: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for col, (label, unit, dec) in BETWEEN_METRICS.items():
        if col not in per.columns:
            continue
        a = per.loc[per.experiment_group == "N_E", col].dropna().to_numpy(float)
        b = per.loc[per.experiment_group == "L_E", col].dropna().to_numpy(float)
        if len(a) < 3 or len(b) < 3:
            continue
        t, df, p = welch(a, b)
        u, p_u = stats.mannwhitneyu(b, a, alternative="two-sided")
        g = hedges_g(a, b)
        lo, hi = bootstrap_ci(a, b, n_boot=n_boot)
        rows.append({
            "metric": col, "label": label, "unit": unit, "decimals": dec,
            "n_natural": len(a), "n_airslide": len(b),
            "mean_natural": a.mean(), "sd_natural": a.std(ddof=1),
            "median_natural": np.median(a), "iqr_natural": np.subtract(*np.percentile(a, [75, 25])),
            "mean_airslide": b.mean(), "sd_airslide": b.std(ddof=1),
            "median_airslide": np.median(b), "iqr_airslide": np.subtract(*np.percentile(b, [75, 25])),
            "mean_difference_airslide_minus_natural": b.mean() - a.mean(),
            "welch_t": t, "welch_df": df, "welch_p": p,
            "mannwhitney_U": float(u), "mannwhitney_p": float(p_u),
            "rank_biserial": float(2 * u / (len(a) * len(b)) - 1),
            "hedges_g": g, "hedges_g_ci95_low": lo, "hedges_g_ci95_high": hi,
            "shapiro_p_natural": float(stats.shapiro(a).pvalue),
            "shapiro_p_airslide": float(stats.shapiro(b).pvalue),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["welch_p_holm"] = holm(out["welch_p"].to_numpy())
    out["mannwhitney_p_holm"] = holm(out["mannwhitney_p"].to_numpy())
    return out


# ----------------------------------------------------------------------------- within subject
def gg_epsilon(wide: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon from the k x k covariance of a subjects x k matrix."""
    k = wide.shape[1]
    s = np.cov(wide, rowvar=False)
    c = np.eye(k) - np.ones((k, k)) / k
    dc = c @ s @ c
    ev = np.linalg.eigvalsh(dc)
    ev = ev[ev > 1e-12]
    return float(ev.sum() ** 2 / ((k - 1) * (ev ** 2).sum()))


def rm_anova_oneway(wide: np.ndarray) -> dict:
    n, k = wide.shape
    grand = wide.mean()
    ss_cond = n * ((wide.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((wide.mean(axis=1) - grand) ** 2).sum()
    ss_tot = ((wide - grand) ** 2).sum()
    ss_err = ss_tot - ss_cond - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f = (ss_cond / df1) / (ss_err / df2)
    eps = gg_epsilon(wide)
    return {
        "n_subjects": n, "F": f, "df1": df1, "df2": df2,
        "p_uncorrected": float(stats.f.sf(f, df1, df2)),
        "gg_epsilon": eps,
        "p_gg": float(stats.f.sf(f, df1 * eps, df2 * eps)),
        "partial_eta2": ss_cond / (ss_cond + ss_err),
    }


def finger_within_subject(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fingers = ["I", "M", "R", "P"]
    per_finger = cells.groupby(["subject_id", "experiment_group", "finger_condition"],
                               as_index=False)[WITHIN_METRICS].mean()
    rows, post = [], []
    for scope in ["all", "N_E", "L_E"]:
        d = per_finger if scope == "all" else per_finger[per_finger.experiment_group == scope]
        for col in WITHIN_METRICS:
            wide = d.pivot(index="subject_id", columns="finger_condition", values=col)[fingers].dropna()
            if len(wide) < 5:
                continue
            res = rm_anova_oneway(wide.to_numpy(float))
            fr = stats.friedmanchisquare(*[wide[f].to_numpy() for f in fingers])
            kendall_w = fr.statistic / (len(wide) * (len(fingers) - 1))
            row = {"scope": scope, "metric": col, **res,
                   "friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue),
                   "kendall_w": float(kendall_w)}
            for f in fingers:
                row[f"mean_{f}"] = wide[f].mean()
                row[f"sd_{f}"] = wide[f].std(ddof=1)
            rows.append(row)
            # pinky vs each other finger, paired
            for other in ["I", "M", "R"]:
                diff = wide["P"].to_numpy() - wide[other].to_numpy()
                t, p = stats.ttest_rel(wide["P"], wide[other])
                w = stats.wilcoxon(diff)
                post.append({"scope": scope, "metric": col, "comparison": f"P - {other}",
                             "n": len(diff), "mean_difference": diff.mean(),
                             "sd_difference": diff.std(ddof=1),
                             "paired_t": float(t), "paired_p": float(p),
                             "cohens_dz": diff.mean() / diff.std(ddof=1),
                             "wilcoxon_p": float(w.pvalue)})
    post_df = pd.DataFrame(post)
    if post_df.empty:
        return pd.DataFrame(rows), post_df
    for (scope, col), grp in post_df.groupby(["scope", "metric"]):
        post_df.loc[grp.index, "paired_p_holm"] = holm(grp["paired_p"].to_numpy())
    return pd.DataFrame(rows), post_df


def stiffness_slopes(cells: pd.DataFrame) -> pd.DataFrame:
    per_stiff = cells.groupby(["subject_id", "experiment_group", "stiffness_value"],
                              as_index=False)[WITHIN_METRICS].mean()
    rows = []
    for scope in ["all", "N_E", "L_E"]:
        d = per_stiff if scope == "all" else per_stiff[per_stiff.experiment_group == scope]
        for col in WITHIN_METRICS:
            slopes = []
            for sid, g in d.groupby("subject_id"):
                g = g.dropna(subset=[col])
                if g.stiffness_value.nunique() >= 3:
                    slopes.append(np.polyfit(g.stiffness_value, g[col], 1)[0])
            s = np.asarray(slopes)
            if len(s) < 3:
                continue
            t, p = stats.ttest_1samp(s, 0.0)
            w = stats.wilcoxon(s)
            levels = np.sort(d.stiffness_value.unique())
            rows.append({"scope": scope, "metric": col, "n_subjects": len(s),
                         "stiffness_min": levels.min(), "stiffness_max": levels.max(),
                         "mean_slope_per_unit": s.mean(), "sd_slope_per_unit": s.std(ddof=1),
                         "mean_change_over_range": s.mean() * (levels.max() - levels.min()),
                         "t": float(t), "p": float(p), "cohens_dz": s.mean() / s.std(ddof=1),
                         "wilcoxon_p": float(w.pvalue),
                         "grand_mean": d[col].mean()})
    return pd.DataFrame(rows)


def mixed_anovas(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    try:
        import pingouin as pg  # type: ignore
    except Exception as exc:  # pragma: no cover
        return pd.DataFrame(), pd.DataFrame(), f"pingouin unavailable ({exc}); mixed ANOVA skipped"
    finger_rows, stiff_rows = [], []
    per_finger = cells.groupby(["subject_id", "experiment_group", "finger_condition"],
                               as_index=False)[WITHIN_METRICS].mean()
    per_stiff = cells.groupby(["subject_id", "experiment_group", "stiffness_value"],
                              as_index=False)[WITHIN_METRICS].mean()
    for col in WITHIN_METRICS:
        d = per_finger.dropna(subset=[col])
        complete = d.groupby("subject_id").finger_condition.nunique()
        d = d[d.subject_id.isin(complete[complete == 4].index)]
        a = pg.mixed_anova(data=d, dv=col, within="finger_condition", between="experiment_group",
                           subject="subject_id", correction=True)
        a.insert(0, "metric", col)
        finger_rows.append(a)
        d = per_stiff.dropna(subset=[col])
        complete = d.groupby("subject_id").stiffness_value.nunique()
        d = d[d.subject_id.isin(complete[complete == complete.max()].index)]
        a = pg.mixed_anova(data=d, dv=col, within="stiffness_value", between="experiment_group",
                           subject="subject_id", correction=True)
        a.insert(0, "metric", col)
        stiff_rows.append(a)
    return pd.concat(finger_rows, ignore_index=True), pd.concat(stiff_rows, ignore_index=True), "ok"


# ----------------------------------------------------------------------------- report
def sentence(r: pd.Series) -> str:
    d = int(r.decimals)
    unit = f" {r.unit}" if r.unit else ""
    return (f"{r.label}: air-slide {fnum(r.mean_airslide, d)} ± {fnum(r.sd_airslide, d)} vs. natural "
            f"{fnum(r.mean_natural, d)} ± {fnum(r.sd_natural, d)}{unit}; "
            f"Welch t({r.welch_df:.1f}) = {r.welch_t:.2f}, {fmt_p(r.welch_p)} "
            f"(Holm {fmt_p(r.welch_p_holm)}); Mann-Whitney {fmt_p(r.mannwhitney_p)}; "
            f"Hedges g = {r.hedges_g:.2f} [{r.hedges_g_ci95_low:.2f}, {r.hedges_g_ci95_high:.2f}]")


def write_report(out_dir: Path, between: pd.DataFrame, finger: pd.DataFrame, post: pd.DataFrame,
                 stiff: pd.DataFrame, mixed_f: pd.DataFrame, mixed_s: pd.DataFrame, note: str) -> Path:
    n_text = ("" if between.empty else
              f"n = {int(between.n_natural.iloc[0])} natural, {int(between.n_airslide.iloc[0])} air-slide. ")
    L = ["# Kinematic section: inferential statistics", "",
         "Unit of analysis = participant (mean over the 9 stiffness x 4 finger cells). " + n_text +
         "Mean ± SD; Welch two-sample t (df Welch-Satterthwaite); Mann-Whitney U as the "
         "distribution-free check; Hedges g (air-slide minus natural, pooled SD) with bootstrap "
         "95% CI (10,000 resamples); Holm correction across the between-setup family.", "",
         "## Between-setup comparisons (air-slide vs. natural)", ""]
    if between.empty:
        L.append("(skipped: both setups are needed for between-setup tests)")
    for _, r in between.iterrows():
        flag = "**" if r.welch_p_holm < 0.05 else ""
        L.append(f"- {flag}{sentence(r)}{flag}")
    L += ["", "Bold = survives Holm correction at alpha = .05.", "",
          "## Finger effect (within participant, one-way repeated measures)", ""]
    for _, r in finger[finger.scope == "all"].iterrows():
        means = ", ".join(f"{FINGER_LABEL[f]} {r[f'mean_{f}']:.2f} ± {r[f'sd_{f}']:.2f}" for f in "IMRP")
        L.append(f"- {BETWEEN_METRICS.get(r.metric, (r.metric, '', 2))[0]}: {means}; "
                 f"RM-ANOVA F({r.df1:.0f}, {r.df2:.0f}) = {r.F:.2f}, GG eps = {r.gg_epsilon:.2f}, "
                 f"{fmt_p(r.p_gg)}, partial eta^2 = {r.partial_eta2:.3f}; "
                 f"Friedman chi^2({r.df1:.0f}) = {r.friedman_chi2:.2f}, {fmt_p(r.friedman_p)}, "
                 f"Kendall W = {r.kendall_w:.3f}")
        for _, q in post[(post.scope == "all") & (post.metric == r.metric)].iterrows():
            L.append(f"    - {q.comparison}: {q.mean_difference:+.2f} ± {q.sd_difference:.2f}, "
                     f"paired t({q.n - 1}) = {q.paired_t:.2f}, {fmt_p(q.paired_p)} (Holm "
                     f"{fmt_p(q.paired_p_holm)}), dz = {q.cohens_dz:.2f}")
    L += ["", "## Stiffness effect (per-participant linear slope vs. stimulus level)", ""]
    for _, r in stiff[stiff.scope == "all"].iterrows():
        L.append(f"- {BETWEEN_METRICS.get(r.metric, (r.metric, '', 2))[0]}: grand mean {r.grand_mean:.2f}; "
                 f"slope {r.mean_slope_per_unit:+.4f} ± {r.sd_slope_per_unit:.4f} per stiffness unit "
                 f"(= {r.mean_change_over_range:+.2f} over {r.stiffness_min:.0f}-{r.stiffness_max:.0f}); "
                 f"one-sample t({r.n_subjects - 1}) = {r.t:.2f}, {fmt_p(r.p)}, dz = {r.cohens_dz:.2f}; "
                 f"Wilcoxon {fmt_p(r.wilcoxon_p)}")
    L += ["", f"## Mixed ANOVA setup x finger / setup x stiffness ({note})", ""]
    for name, tbl in [("setup x finger", mixed_f), ("setup x stiffness", mixed_s)]:
        if tbl.empty:
            continue
        L.append(f"### {name}")
        for metric, g in tbl.groupby("metric", sort=False):
            parts = []
            for _, r in g.iterrows():
                cols = set(g.columns)
                p_gg = r["p_GG_corr"] if "p_GG_corr" in cols else r.get("p-GG-corr", np.nan)
                p_unc = r["p_unc"] if "p_unc" in cols else r["p-unc"]
                p = p_gg if np.isfinite(p_gg) else p_unc
                parts.append(f"{r.Source}: F({r.DF1:.0f}, {r.DF2:.0f}) = {r.F:.2f}, {fmt_p(p)}, "
                             f"eta_p^2 = {r.np2:.3f}")
            L.append(f"- {BETWEEN_METRICS.get(metric, (metric, '', 2))[0]}: " + "; ".join(parts))
        L.append("")
    path = out_dir / "kinematic_setup_statistics_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def run(results_csv: Path = DEFAULT_RESULTS, out_dir: Path | None = None, n_boot: int = 10000) -> dict:
    results_csv = Path(results_csv)
    out_dir = Path(out_dir) if out_dir else results_csv / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    per = load_per_participant(results_csv)
    cells = load_cells(results_csv)
    between = between_setup_tests(per, n_boot)
    finger, post = finger_within_subject(cells)
    stiff = stiffness_slopes(cells)
    if cells.experiment_group.nunique() >= 2:
        mixed_f, mixed_s, note = mixed_anovas(cells)
    else:
        mixed_f, mixed_s, note = pd.DataFrame(), pd.DataFrame(), "single setup; mixed ANOVA skipped"
    per.to_csv(out_dir / "per_participant_kinematic_means.csv", index=False)
    between.to_csv(out_dir / "between_setup_tests.csv", index=False)
    finger.to_csv(out_dir / "finger_within_subject.csv", index=False)
    post.to_csv(out_dir / "finger_posthoc_pinky.csv", index=False)
    stiff.to_csv(out_dir / "stiffness_slopes.csv", index=False)
    if not mixed_f.empty:
        mixed_f.to_csv(out_dir / "finger_mixed_anova.csv", index=False)
        mixed_s.to_csv(out_dir / "stiffness_mixed_anova.csv", index=False)
    report = write_report(out_dir, between, finger, post, stiff, mixed_f, mixed_s, note)
    return {"between": between, "finger": finger, "posthoc": post, "stiffness": stiff,
            "mixed_finger": mixed_f, "mixed_stiffness": mixed_s, "report_path": report,
            "per_participant": per}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="<results>/<group>/csv folder")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    res = run(args.results, args.out, args.n_boot)
    print(res["report_path"].read_text(encoding="utf-8"))
