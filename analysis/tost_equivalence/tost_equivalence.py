"""
Equivalence analysis for the setup and finger null effects (standalone)
=======================================================================

Engine for TOST.ipynb. Every number in the paper's equivalence paragraph,
figure and companion ANOVA sentences is produced here from the per-participant
psychometric fits written by the psychophysics pipeline. No dependency on the
anova_statistics package.

Design
------
* Data: analysis/psychophysics/results/<DATA_SELECTION>/csv/all/shared/
  pse_jnd_by_subject_finger.csv (mm/m). One 4-parameter logistic fit per
  participant x finger, 64 trials each.
* Cohort: participants whose raw-data folder under Parallel_Heptics/results is
  tagged "(filter)" are excluded (currently L_E_19, N_E_17), giving N = 39
  (20 air-slide, 19 natural). The exclusion rule used in the paper is checked
  in section 1 of the notebook against the per-finger success rates.
* Unequal groups: Welch's t with Welch-Satterthwaite df (Delacre, Lakens &
  Leys, 2017; Welch, 1947), the form used by Lakens (2017, eq. 3-4).

Steps (numbering follows the supervisor plan and the notebook sections)
------------------------------------------------------------------------
0. Consistency: per-participant summaries per finger next to the pooled fits.
1. Bounds, pre-specified in perceptual units (mm/m): bias +/-1 JND = +/-1.6
   (primary) and +/-0.5 JND = +/-0.8 (secondary); JND +/-0.5 mm/m and a x1.30
   ratio on the log scale.
2. Feasibility: observed SDs and the 90% CI half-width vs the bound.
3. Tests: Welch TOST for Setup (per participant and per finger); six paired
   TOSTs for finger pairs with Holm correction (Finger factor equivalent only
   if all six pass); one-sample TOST of each finger's bias vs zero. Every
   p-value is cross-checked against pingouin.tost.
   Companion NHST: mixed-design ANOVA on the same cohort with Holm post hoc,
   plus non-parametric and log-scale robustness checks for the finger effect.
4. Bayes factors: JZS BF for the per-participant Setup t-test (prior-width
   robustness r = 0.35..1.0) and for each finger pair; JASP export for the
   omnibus inclusion BFs (not available in Python).
5. Minimal detectable effect at 80% power from the observed SDs and actual n.
6. Sensitivity: the key verdicts re-run on an alternative cohort (extra
   participants removed), to show the conclusions do not hinge on them.

Outputs: results/<cohort>/csv/*.csv, results/<cohort>/figures/*.{png,pdf},
results/<cohort>/report_sentences.md
"""

from __future__ import annotations

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pingouin as pg
except Exception:  # pragma: no cover
    pg = None

# --------------------------------------------------------------------------- #
# Configuration (overridden from the notebook's setup cell)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PSYCHO_RESULTS = os.path.join(REPO, "analysis", "psychophysics", "results")
RAW_RESULTS_DIR = os.path.join(REPO, "results")      # raw per-participant folders
OUT_ROOT = os.path.join(HERE, "results")

STANDARD = 8.5                      # mm/m
ALPHA = 0.05                        # TOST alpha; 90% CI = 1 - 2*alpha
FINGERS = ["I", "M", "R", "P"]
FINGER_NAME = {"I": "Index", "M": "Middle", "R": "Ring", "P": "Little"}
SYSTEMS = ("L", "N")
SYSTEM_NAME = {"L": "air-slide", "N": "natural"}

# Pre-specified bounds (mm/m). Constants, not derived from the data at run time.
SESOI_BIAS = {"1 JND": 1.6, "0.5 JND": 0.8}
SESOI_JND_ABS = 0.5
SESOI_JND_RATIO = 1.30              # log-scale bound = +/- ln(1.30)

# Success-rate exclusion rule used in the paper (checked, not applied, here:
# the psychophysics pipeline already drops "(filter)" folders).
SUCCESS_RULE = {"threshold": 0.60, "min_fingers": 2}

EXCLUDED_FALLBACK = ("L_E_19", "N_E_17")


def data_path(selection: str = "L_N_E") -> str:
    return os.path.join(PSYCHO_RESULTS, selection, "csv", "all", "shared", "pse_jnd_by_subject_finger.csv")


def pooled_path(selection: str = "L_N_E") -> str:
    return os.path.join(PSYCHO_RESULTS, selection, "csv", "all", "shared", "pse_jnd_group_by_finger.csv")


def success_trials_path(selection: str = "L_N_E") -> str:
    return os.path.join(PSYCHO_RESULTS, selection, "csv", "all", "shared", "success_trials.csv")


def excluded_subjects() -> tuple[str, ...]:
    """Subject ids whose raw-data folder is tagged '(filter)'."""
    ids = set(EXCLUDED_FALLBACK)
    try:
        for d in os.listdir(RAW_RESULTS_DIR):
            if "(filter)" in d:
                m = re.match(r"([LN]_[EP]_\d+)", d)
                if m:
                    ids.add(m.group(1))
    except OSError:
        pass
    return tuple(sorted(ids))


def output_dirs(root: str, cohort: str) -> dict:
    d = {"root": os.path.join(root, cohort)}
    d["csv"] = os.path.join(d["root"], "csv")
    d["figures"] = os.path.join(d["root"], "figures")
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_fits(path: str, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per-participant x finger fits in mm/m with canonical column names."""
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "Subject": raw["subject_id"].astype(str),
        "System": raw["subject_id"].astype(str).str[0],
        "Finger": raw["finger_condition"].astype(str),
        "PSE": pd.to_numeric(raw["pse"], errors="coerce"),
        "JND": pd.to_numeric(raw["jnd"], errors="coerce"),
        "fit_quality": raw.get("fit_quality", "unknown"),
        "fit_warning": raw.get("fit_warning", np.nan),
        "n_trials": raw.get("n_trials", np.nan),
    })
    df["Bias"] = df["PSE"] - STANDARD
    df["logJND"] = np.log(df["JND"])
    med = df["PSE"].median()
    if not (5.0 < med < 15.0):
        raise ValueError(f"PSE median {med:.2f} is not in mm/m (expected ~8.5); check {path}")
    if exclude:
        df = df[~df["Subject"].isin(exclude)].copy()
    return df[df["Finger"].isin(FINGERS)].reset_index(drop=True)


def cohort_summary(df: pd.DataFrame) -> dict:
    sm = subject_means(df, "Bias")
    return {"N": int(df.Subject.nunique()),
            "n_L": int((sm.System == "L").sum()), "n_N": int((sm.System == "N").sum()),
            "fits": int(len(df))}


def subject_means(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """One value per participant: mean over the four fingers (complete cases)."""
    wide = df.pivot_table(index=["Subject", "System"], columns="Finger", values=dv)
    wide = wide.dropna(subset=FINGERS)
    return wide.mean(axis=1).reset_index(name=dv)


def wide_by_finger(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    w = df.pivot_table(index="Subject", columns="Finger", values=dv)
    return w.dropna(subset=FINGERS)


# --------------------------------------------------------------------------- #
# Section 1: exclusion-rule check on per-finger success rates
# --------------------------------------------------------------------------- #
def success_by_finger(selection: str = "L_N_E", include_filtered: bool = True) -> pd.DataFrame:
    """Per-participant x finger success rate (proportion correct).

    Analysed participants come from the pipeline's success_trials.csv. The
    filtered participants are scored with the pipeline's own loader
    (twoafc_psychophysics) so the rule can be checked on everyone who finished.
    """
    t = pd.read_csv(success_trials_path(selection), low_memory=False)
    sr = t.groupby(["subject_id", "finger_condition"])["correct_response"].mean().unstack()
    if include_filtered:
        try:
            import sys, tempfile
            from pathlib import Path
            sys.path.insert(0, os.path.join(REPO, "analysis", "psychophysics"))
            import twoafc_psychophysics as pf  # type: ignore
            tmp = Path(tempfile.mkdtemp(prefix="tost_filter_only_"))
            disc = pf.discover_answer_files(Path(RAW_RESULTS_DIR), tmp, selection="FILTER_ONLY",
                                            exclude_filter_folders=False)
            raw = pf.load_selected_subject_csvs(disc)
            cols, _ = pf.detect_columns(raw, {})
            std, _ = pf.infer_standard_value(raw, cols, pf.STANDARD_RAW_FALLBACK)
            clean, _ = pf.canonicalize_trials(raw, cols, standard_value=std,
                                              standard_tolerance=pf.STANDARD_RAW_ABS_TOLERANCE)
            clean = pf.add_success_and_time_columns(clean)
            extra = clean.groupby(["subject_id", "finger_condition"])["correct_response"].mean().unstack()
            sr = pd.concat([sr[~sr.index.isin(extra.index)], extra])
        except Exception as exc:  # pragma: no cover
            print(f"[tost] filtered participants not scored ({exc}); rule checked on analysed cohort only")
    return sr[[c for c in FINGERS if c in sr.columns]]


def exclusion_rule_table(sr: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    """Which participants each candidate rule would exclude."""
    if rules is None:
        rules = {
            ">=60% on >=2 fingers (paper)": lambda r: (r >= .60).sum() >= 2,
            ">=60% on >=3 fingers": lambda r: (r >= .60).sum() >= 3,
            ">=60% on all fingers": lambda r: (r >= .60).all(),
            ">=55% on all fingers": lambda r: (r >= .55).all(),
            ">50% on all fingers": lambda r: (r > .50).all(),
        }
    rows = []
    for name, fn in rules.items():
        keep = sr.apply(fn, axis=1)
        rows.append({"rule": name, "n_excluded": int((~keep).sum()),
                     "excluded": ", ".join(sorted(sr.index[~keep]))})
    return pd.DataFrame(rows)


def apply_success_rule(sr: pd.DataFrame, threshold: float, min_fingers: int) -> list[str]:
    keep = sr.apply(lambda r: (r >= threshold).sum() >= min_fingers, axis=1)
    return sorted(sr.index[~keep])


# --------------------------------------------------------------------------- #
# Step 0: consistency with the pooled fits
# --------------------------------------------------------------------------- #
def step0_consistency(df: pd.DataFrame, pooled_csv: str) -> pd.DataFrame:
    rows = []
    pooled = pd.read_csv(pooled_csv) if os.path.exists(pooled_csv) else None
    for f in FINGERS:
        sub = df[df["Finger"] == f]
        row = {"Finger": FINGER_NAME[f], "n fits": int(len(sub)),
               "trials/fit": int(sub["n_trials"].median()) if sub["n_trials"].notna().any() else np.nan}
        for dv in ("Bias", "JND"):
            v = sub[dv].dropna()
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            row[f"{dv} mean"] = v.mean(); row[f"{dv} SD"] = v.std(ddof=1)
            row[f"{dv} median"] = med; row[f"{dv} IQR"] = f"[{q1:.2f}, {q3:.2f}]"
        if pooled is not None and "finger_condition" in pooled:
            p = pooled[pooled["finger_condition"] == f]
            if len(p):
                row["pooled bias"] = float(p["pse"].iloc[0]) - STANDARD
                row["pooled JND"] = float(p["jnd"].iloc[0])
                row["pooled trials"] = int(p["n_trials"].iloc[0]) if "n_trials" in p else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def jnd_reference(df: pd.DataFrame) -> float:
    """Median per-participant JND: the perceptual anchor for the bias bound."""
    return float(df["JND"].median())


# --------------------------------------------------------------------------- #
# TOST primitives (Lakens 2017)
# --------------------------------------------------------------------------- #
def _lakens_outcome(equivalent: bool, different: bool) -> str:
    if equivalent and not different:
        return "A: equivalent, not different"
    if not equivalent and different:
        return "B: different, not equivalent"
    if equivalent and different:
        return "C: different but equivalent (trivial)"
    return "D: undetermined"


def _finish(est, se, dof, low, high):
    t_low = (est - low) / se
    t_up = (est - high) / se
    p_low = stats.t.sf(t_low, dof)
    p_up = stats.t.cdf(t_up, dof)
    p_tost = max(p_low, p_up)
    tcrit = stats.t.ppf(1 - ALPHA, dof)
    t_nhst = est / se
    p_nhst = 2 * stats.t.sf(abs(t_nhst), dof)
    eq, diff = bool(p_tost < ALPHA), bool(p_nhst < ALPHA)
    return dict(estimate=float(est), se=float(se), df=float(dof),
                ci90_lo=float(est - tcrit * se), ci90_hi=float(est + tcrit * se),
                bound_lo=float(low), bound_hi=float(high),
                t_lower=float(t_low), t_upper=float(t_up),
                t_tost=float(t_low if p_low >= p_up else t_up),
                p_lower=float(p_low), p_upper=float(p_up), p_tost=float(p_tost),
                t_nhst=float(t_nhst), p_nhst=float(p_nhst),
                equivalent=eq, different=diff, outcome=_lakens_outcome(eq, diff))


def tost_one_sample(x, low, high, mu=0.0):
    """Lakens (2017) eq. 7; also used for paired data on the differences."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = x.size
    out = _finish(x.mean() - mu, x.std(ddof=1) / np.sqrt(n), n - 1, low, high)
    out.update(n=int(n), sd=float(x.std(ddof=1)))
    return out


def tost_welch(a, b, low, high):
    """Lakens (2017) eq. 3-4: Welch t, Welch-Satterthwaite df (unequal n is fine)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    se = np.sqrt(v1 + v2)
    dof = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    out = _finish(a.mean() - b.mean(), se, dof, low, high)
    out.update(n1=int(n1), n2=int(n2), mean1=float(a.mean()), mean2=float(b.mean()),
               sd1=float(a.std(ddof=1)), sd2=float(b.std(ddof=1)))
    return out


def pingouin_tost_p(x, y=None, bound=1.0, paired=False):
    """Cross-check with pingouin's symmetric-bound TOST (Welch for two groups)."""
    if pg is None:
        return np.nan
    try:
        if y is None:
            res = pg.tost(np.asarray(x, float), np.zeros(len(x)), bound=bound, paired=True)
        else:
            res = pg.tost(np.asarray(x, float), np.asarray(y, float), bound=bound,
                          paired=paired, correction=(not paired))
        return float(res["pval"].iloc[0])
    except Exception:
        return np.nan


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty(len(p)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def _dv_bounds():
    out = [("Bias", f"bias +/-{b} ({name})", -b, b, b) for name, b in SESOI_BIAS.items()]
    out.append(("JND", f"JND +/-{SESOI_JND_ABS} mm/m", -SESOI_JND_ABS, SESOI_JND_ABS, SESOI_JND_ABS))
    lb = np.log(SESOI_JND_RATIO)
    out.append(("logJND", f"JND ratio x{SESOI_JND_RATIO}", -lb, lb, lb))
    return out


# --------------------------------------------------------------------------- #
# Step 2: feasibility
# --------------------------------------------------------------------------- #
def step2_feasibility(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dv, bounds in (("Bias", SESOI_BIAS), ("JND", {"0.5 mm/m": SESOI_JND_ABS}),
                       ("logJND", {"x1.30": np.log(SESOI_JND_RATIO)})):
        sm = subject_means(df, dv)
        a = sm.loc[sm.System == "L", dv].to_numpy(); b = sm.loc[sm.System == "N", dv].to_numpy()
        v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1))
        hw = stats.t.ppf(1 - ALPHA, dof) * np.sqrt(v1 + v2)
        for bname, bval in bounds.items():
            rows.append(dict(DV=dv, comparison="Setup, per participant", n=f"{len(a)} vs {len(b)}",
                             SD=f"{a.std(ddof=1):.2f} / {b.std(ddof=1):.2f}", ci90_halfwidth=hw,
                             bound=bval, bound_name=bname, feasible=bool(hw < bval)))
        w = wide_by_finger(df, dv)
        sd_diff = np.median([np.std(w[x] - w[y], ddof=1) for x, y in combinations(FINGERS, 2)])
        hw = stats.t.ppf(1 - ALPHA, len(w) - 1) * sd_diff / np.sqrt(len(w))
        for bname, bval in bounds.items():
            rows.append(dict(DV=dv, comparison="Finger pairs (paired), median SD of diffs", n=str(len(w)),
                             SD=f"{sd_diff:.2f}", ci90_halfwidth=hw, bound=bval, bound_name=bname,
                             feasible=bool(hw < bval)))
        if dv == "Bias":
            for f in FINGERS:
                x = df.loc[df.Finger == f, dv].dropna().to_numpy()
                hw = stats.t.ppf(1 - ALPHA, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
                for bname, bval in bounds.items():
                    rows.append(dict(DV=dv, comparison=f"Bias vs 0, {FINGER_NAME[f]}", n=str(len(x)),
                                     SD=f"{x.std(ddof=1):.2f}", ci90_halfwidth=hw, bound=bval,
                                     bound_name=bname, feasible=bool(hw < bval)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 3: the tests
# --------------------------------------------------------------------------- #
def step3_setup(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dv, label, lo, hi, pb in _dv_bounds():
        units = {"per participant": subject_means(df, dv)}
        for f in FINGERS:
            units[FINGER_NAME[f]] = df.loc[df.Finger == f, ["System", dv]]
        for unit, g in units.items():
            a = g.loc[g.System == "L", dv].dropna().to_numpy()
            b = g.loc[g.System == "N", dv].dropna().to_numpy()
            r = tost_welch(a, b, lo, hi)
            r["p_tost_pingouin"] = pingouin_tost_p(a, b, bound=pb, paired=False)
            rows.append(dict(DV=dv, bound=label, unit=unit, **r))
    out = pd.DataFrame(rows)
    m = out.DV == "logJND"
    for c in ("estimate", "ci90_lo", "ci90_hi"):
        out.loc[m, c + "_ratio"] = np.exp(out.loc[m, c])
    return out


def step3_finger_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dv, label, lo, hi, pb in _dv_bounds():
        w = wide_by_finger(df, dv)
        block = []
        for x, y in combinations(FINGERS, 2):
            d = (w[x] - w[y]).to_numpy()
            r = tost_one_sample(d, lo, hi)
            r["p_tost_pingouin"] = pingouin_tost_p(w[x].to_numpy(), w[y].to_numpy(), bound=pb, paired=True)
            block.append(dict(DV=dv, bound=label, pair=f"{FINGER_NAME[x]}-{FINGER_NAME[y]}", **r))
        for b, p in zip(block, holm([b["p_tost"] for b in block])):
            b["p_tost_holm"] = float(p); b["equivalent_holm"] = bool(p < ALPHA)
        rows += block
    return pd.DataFrame(rows)


def step3_bias_vs_zero(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, b in SESOI_BIAS.items():
        for f in FINGERS:
            x = df.loc[df.Finger == f, "Bias"].dropna().to_numpy()
            r = tost_one_sample(x, -b, b)
            r["p_tost_pingouin"] = pingouin_tost_p(x, None, bound=b)
            rows.append(dict(DV="Bias", bound=f"bias +/-{b} ({name})", finger=FINGER_NAME[f], **r))
    return pd.DataFrame(rows)


def finger_factor_verdict(pairs: pd.DataFrame) -> pd.DataFrame:
    """Equivalence of the Finger factor: all six Holm-corrected pairs must pass."""
    return pairs.groupby(["DV", "bound"]).agg(
        n_pairs=("pair", "size"), n_equivalent_holm=("equivalent_holm", "sum"),
        all_six_pass=("equivalent_holm", "all"), max_p_holm=("p_tost_holm", "max")).reset_index()


def crosscheck_pingouin(*frames) -> float:
    """Largest |p_tost - pingouin p| over all rows (should be ~1e-15)."""
    return float(max((f.p_tost - f.p_tost_pingouin).abs().max() for f in frames))


# --------------------------------------------------------------------------- #
# Step 3b: companion NHST on the same cohort (mixed ANOVA + post hoc + robustness)
# --------------------------------------------------------------------------- #
def companion_anova(df: pd.DataFrame) -> dict:
    """Mixed-design ANOVA (Setup between, Finger within) for Bias, JND, logJND
    with Mauchly/GG, Holm post hoc for Finger, and non-parametric checks."""
    out = {"anova": [], "posthoc": [], "robust": []}
    for dv in ("Bias", "JND", "logJND"):
        a = pg.mixed_anova(data=df, dv=dv, within="Finger", between="System",
                           subject="Subject", correction=True, effsize="np2")
        sph = pg.sphericity(data=df, dv=dv, within="Finger", subject="Subject")
        eps = float(a["eps"].dropna().iloc[0]) if "eps" in a and a["eps"].notna().any() else np.nan
        for _, r in a.iterrows():
            within = r["Source"] != "System"
            p_gg = r.get("p_GG_corr", np.nan)
            use_gg = within and (not sph.spher) and np.isfinite(p_gg)
            out["anova"].append(dict(
                DV=dv, effect=r["Source"], F=float(r["F"]),
                df1=float(r["DF1"]) * (eps if use_gg else 1), df2=float(r["DF2"]) * (eps if use_gg else 1),
                p=float(p_gg if use_gg else r["p_unc"]), corrected="GG" if use_gg else "none",
                np2=float(r["np2"]), eps=eps if within else np.nan,
                mauchly_W=float(sph.W), mauchly_p=float(sph.pval)))
        ph = pg.pairwise_tests(data=df, dv=dv, within="Finger", subject="Subject", padjust="holm", effsize="cohen")
        for _, r in ph.iterrows():
            out["posthoc"].append(dict(DV=dv, pair=f"{FINGER_NAME[r['A']]}-{FINGER_NAME[r['B']]}",
                                       t=float(r["T"]), df=float(r["dof"]), p_unc=float(r["p_unc"]),
                                       p_holm=float(r["p_corr"]), d=float(r["cohen"])))
        w = wide_by_finger(df, dv)
        fr = stats.friedmanchisquare(*[w[f] for f in FINGERS])
        out["robust"].append(dict(DV=dv, test="Friedman (finger)", statistic=float(fr.statistic), p=float(fr.pvalue)))
        pw = [stats.wilcoxon(w[x], w[y]).pvalue for x, y in combinations(FINGERS, 2)]
        for (x, y), p, ph_ in zip(combinations(FINGERS, 2), pw, holm(pw)):
            out["robust"].append(dict(DV=dv, test=f"Wilcoxon {FINGER_NAME[x]}-{FINGER_NAME[y]}",
                                      statistic=float(np.median(w[x] - w[y])), p=float(p), p_holm=float(ph_)))
    return {k: pd.DataFrame(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Step 4: Bayes factors
# --------------------------------------------------------------------------- #
def step4_bf_setup_ttest(df: pd.DataFrame, radii=(0.35, 0.5, 0.707, 1.0)) -> pd.DataFrame:
    rows = []
    if pg is None:
        return pd.DataFrame(rows)
    for dv in ("Bias", "JND", "logJND"):
        sm = subject_means(df, dv)
        a = sm.loc[sm.System == "L", dv].to_numpy(); b = sm.loc[sm.System == "N", dv].to_numpy()
        t = tost_welch(a, b, -1, 1)["t_nhst"]
        row = dict(DV=dv, unit="per participant", n1=len(a), n2=len(b), t_welch=t)
        for r in radii:
            row[f"BF01 (r={r})"] = 1.0 / float(pg.bayesfactor_ttest(t, len(a), len(b), paired=False, r=r))
        rows.append(row)
    return pd.DataFrame(rows)


def _bf_label(v):
    return ("strong for null" if v > 10 else "moderate for null" if v > 3 else "anecdotal for null" if v > 1
            else "anecdotal for effect" if v > 1 / 3 else "moderate for effect" if v > 0.1 else "strong for effect")


def step4_bf_finger_pairs(df: pd.DataFrame, r: float = 0.707) -> pd.DataFrame:
    rows = []
    if pg is None:
        return pd.DataFrame(rows)
    for dv in ("Bias", "JND", "logJND"):
        w = wide_by_finger(df, dv)
        for x, y in combinations(FINGERS, 2):
            d = (w[x] - w[y]).to_numpy()
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            bf10 = float(pg.bayesfactor_ttest(t, len(d), paired=True, r=r))
            rows.append(dict(DV=dv, pair=f"{FINGER_NAME[x]}-{FINGER_NAME[y]}", n=len(d), t=float(t),
                             BF01=1.0 / bf10, BF10=bf10, evidence=_bf_label(1.0 / bf10)))
    return pd.DataFrame(rows)


def export_for_jasp(df: pd.DataFrame, path: str) -> str:
    frames = []
    for dv in ("Bias", "JND", "logJND"):
        w = df.pivot_table(index=["Subject", "System"], columns="Finger", values=dv)
        w.columns = [f"{dv}_{c}" for c in w.columns]
        frames.append(w)
    pd.concat(frames, axis=1).reset_index().to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Step 5: minimal detectable effect (80% power, two-sided alpha .05)
# --------------------------------------------------------------------------- #
def step5_mde(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dv in ("Bias", "JND", "logJND"):
        sm = subject_means(df, dv)
        a = sm.loc[sm.System == "L", dv].to_numpy(); b = sm.loc[sm.System == "N", dv].to_numpy()
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d80 = float(pg.power_ttest2n(nx=len(a), ny=len(b), power=0.8, alpha=ALPHA)) if pg else np.nan
        rows.append(dict(DV=dv, comparison="Setup, per participant", n=f"{len(a)} vs {len(b)}",
                         pooled_SD=sp, d_80=d80, MDE=d80 * sp))
        w = wide_by_finger(df, dv)
        sd_diff = float(np.median([np.std(w[x] - w[y], ddof=1) for x, y in combinations(FINGERS, 2)]))
        dz80 = float(pg.power_ttest(n=len(w), power=0.8, alpha=ALPHA, contrast="paired")) if pg else np.nan
        rows.append(dict(DV=dv, comparison="Finger pair (paired), median SD of diffs", n=str(len(w)),
                         pooled_SD=sd_diff, d_80=dz80, MDE=dz80 * sd_diff))
    out = pd.DataFrame(rows)
    m = out.DV == "logJND"
    out.loc[m, "MDE_ratio"] = np.exp(out.loc[m, "MDE"])
    return out


# --------------------------------------------------------------------------- #
# Step 6: sensitivity to extra exclusions
# --------------------------------------------------------------------------- #
def key_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """The claims the paper makes, as one row each, for cohort comparison."""
    s = step3_setup(df); p = step3_finger_pairs(df); z = step3_bias_vs_zero(df); v = finger_factor_verdict(p)
    an = companion_anova(df)
    sb = s[(s.DV == "Bias") & (s.unit == "per participant")]
    sj = s[(s.DV == "JND") & (s.unit == "per participant")].iloc[0]
    mr = p[(p.DV == "JND") & (p.pair == "Middle-Ring")].iloc[0]
    ph = an["posthoc"]; a = an["anova"]
    rows = [
        ("Bias: setup equivalent at +/-1 JND", sb[sb.bound.str.contains("1 JND")].iloc[0].equivalent,
         f"{sb.iloc[0].estimate:.2f} [{sb.iloc[0].ci90_lo:.2f}, {sb.iloc[0].ci90_hi:.2f}]"),
        ("Bias: setup equivalent at +/-0.5 JND", sb[sb.bound.str.contains("0.5 JND")].iloc[0].equivalent, ""),
        ("Bias: all six finger pairs equivalent at +/-1 JND (Holm)",
         v[(v.DV == "Bias") & v.bound.str.contains("1 JND")].iloc[0].all_six_pass, ""),
        ("Bias: every finger equivalent to 0 at +/-0.5 JND",
         z[z.bound.str.contains("0.5 JND")].equivalent.all(), f"max p {z[z.bound.str.contains('0.5 JND')].p_tost.max():.3f}"),
        ("Bias: ANOVA effects all non-significant", (a[(a.DV == "Bias")].p >= .05).all(),
         "min p %.3f" % a[a.DV == "Bias"].p.min()),
        ("JND: setup equivalent at +/-0.5 mm/m", sj.equivalent, f"{sj.estimate:.2f} [{sj.ci90_lo:.2f}, {sj.ci90_hi:.2f}]"),
        ("JND: setup ANOVA p", None, "%.3f" % a[(a.DV == "JND") & (a.effect == "System")].p.iloc[0]),
        ("JND: finger ANOVA p", None, "%.3f" % a[(a.DV == "JND") & (a.effect == "Finger")].p.iloc[0]),
        ("JND: middle > ring (Holm p)", bool(ph[(ph.DV == "JND") & (ph.pair == "Middle-Ring")].p_holm.iloc[0] < .05),
         f"{mr.estimate:.2f} [{mr.ci90_lo:.2f}, {mr.ci90_hi:.2f}], p_holm {ph[(ph.DV=='JND')&(ph.pair=='Middle-Ring')].p_holm.iloc[0]:.3f}"),
    ]
    return pd.DataFrame(rows, columns=["claim", "holds", "detail"])


def sensitivity_compare(df_main: pd.DataFrame, extra_excluded: list[str]) -> pd.DataFrame:
    alt = df_main[~df_main.Subject.isin(extra_excluded)]
    m, a = key_verdicts(df_main), key_verdicts(alt)
    cm, ca = cohort_summary(df_main), cohort_summary(alt)
    out = m.merge(a, on="claim", suffixes=(f" N={cm['N']}", f" N={ca['N']}"))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _rows_bias(setup, pairs, zero):
    s = setup[(setup.DV == "Bias") & setup.bound.str.contains("1 JND") & (setup.unit == "per participant")].iloc[0]
    out = [("Setup: air-slide - natural", s.estimate, s.ci90_lo, s.ci90_hi, bool(s.equivalent))]
    out += [(f"{r.finger} vs 0", r.estimate, r.ci90_lo, r.ci90_hi, bool(r.equivalent))
            for _, r in zero[zero.bound.str.contains("1 JND")].iterrows()]
    out += [(r.pair.replace("-", " - "), r.estimate, r.ci90_lo, r.ci90_hi, bool(r.equivalent_holm))
            for _, r in pairs[(pairs.DV == "Bias") & pairs.bound.str.contains("1 JND")].iterrows()]
    return out


def _rows_jnd(setup, pairs):
    s = setup[(setup.DV == "JND") & (setup.unit == "per participant")].iloc[0]
    out = [("Setup: air-slide - natural", s.estimate, s.ci90_lo, s.ci90_hi, bool(s.equivalent))]
    out += [(r.pair.replace("-", " - "), r.estimate, r.ci90_lo, r.ci90_hi, bool(r.equivalent_holm))
            for _, r in pairs[pairs.DV == "JND"].iterrows()]
    return out


def paper_figure(setup, pairs, zero, path: str, panels=("bias", "jnd")) -> str:
    """Single-column equivalence plot: (a) bias, (b) JND, or bias only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = []
    if "bias" in panels:
        spec.append(("(a) PSE bias (mm/m)" if len(panels) > 1 else "PSE bias (mm/m)",
                     _rows_bias(setup, pairs, zero), SESOI_BIAS["1 JND"], SESOI_BIAS["0.5 JND"]))
    if "jnd" in panels:
        spec.append(("(b) JND (mm/m)" if len(panels) > 1 else "JND (mm/m)", _rows_jnd(setup, pairs), SESOI_JND_ABS, None))
    n_rows = [len(p[1]) for p in spec]
    fig, axes = plt.subplots(len(spec), 1, figsize=(3.45, 0.19 * sum(n_rows) + 0.9 + 0.4 * len(spec)),
                             gridspec_kw={"height_ratios": n_rows}, squeeze=False)
    axes = axes[:, 0]
    for ax, (title, rows, b1, b2) in zip(axes, spec):
        y = np.arange(len(rows))[::-1]
        ax.axvspan(-b1, b1, color="#e3ecf5", zorder=0, lw=0)
        if b2 is not None:
            ax.axvspan(-b2, b2, color="#c9d9ea", zorder=0, lw=0)
        ax.axvline(0, color="0.4", lw=0.7)
        for yi, (lab, est, lo, hi, ok) in zip(y, rows):
            ax.plot([lo, hi], [yi, yi], color="#1f4e79", lw=1.4)
            ax.plot(est, yi, "o", ms=4, mfc="#1f4e79" if ok else "white", mec="#1f4e79", mew=1.2)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
        ax.tick_params(axis="x", labelsize=7); ax.set_title(title, fontsize=8, loc="left")
        ax.set_ylim(-0.7, len(rows) - 0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("Difference [90% CI]; shading = equivalence bounds", fontsize=7)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(path, dpi=300)
    if path.lower().endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Report sentences (numbers only; wording is the author's)
# --------------------------------------------------------------------------- #
def _fmt_p(p):
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def report_sentences(df, setup, pairs, verdict, zero, anova, bf_t, bf_pairs, mde, jnd_ref) -> str:
    c = cohort_summary(df)
    sb = setup[(setup.DV == "Bias") & setup.bound.str.contains("1 JND") & (setup.unit == "per participant")].iloc[0]
    sb2 = setup[(setup.DV == "Bias") & setup.bound.str.contains("0.5 JND") & (setup.unit == "per participant")].iloc[0]
    sj = setup[(setup.DV == "JND") & (setup.unit == "per participant")].iloc[0]
    vb = verdict[(verdict.DV == "Bias") & verdict.bound.str.contains("1 JND")].iloc[0]
    zb2 = zero[zero.bound.str.contains("0.5 JND")]
    a, ph = anova["anova"], anova["posthoc"]
    aj_f = a[(a.DV == "JND") & (a.effect == "Finger")].iloc[0]
    aj_s = a[(a.DV == "JND") & (a.effect == "System")].iloc[0]
    mr = ph[(ph.DV == "JND") & (ph.pair == "Middle-Ring")].iloc[0]
    mrt = pairs[(pairs.DV == "JND") & (pairs.pair == "Middle-Ring")].iloc[0]
    diffj = pairs[(pairs.DV == "JND") & pairs.different]
    bt = bf_t.set_index("DV"); pj = bf_pairs[bf_pairs.DV == "JND"].set_index("pair")
    mb = mde[(mde.DV == "Bias") & mde.comparison.str.startswith("Setup")].iloc[0]
    mj = mde[(mde.DV == "JND") & mde.comparison.str.startswith("Setup")].iloc[0]
    L = [f"## Numbers for the paper (N = {c['N']}: {c['n_L']} air-slide, {c['n_N']} natural)\n",
         "### ANOVA sentence (Results, after the ANOVA table)\n",
         f"PSE bias showed no significant effect of Setup, Finger, or their interaction (all p >= {a[a.DV=='Bias'].p.min():.3f}). "
         f"For JND, the Setup effect was not significant (F({aj_s.df1:.0f},{aj_s.df2:.0f}) = {aj_s.F:.2f}, p = {aj_s.p:.3f}, np2 = {aj_s.np2:.3f}); "
         f"the Finger effect was {'significant' if aj_f.p < .05 else 'not significant'} (F({aj_f.df1:.2f},{aj_f.df2:.1f}) = {aj_f.F:.2f}, "
         f"p{'_GG' if aj_f.corrected=='GG' else ''} = {aj_f.p:.3f}, np2 = {aj_f.np2:.3f}); Holm post hoc middle vs ring: t({mr.df:.0f}) = {mr.t:.2f}, "
         f"p = {mr.p_holm:.3f}, d = {mr.d:.2f}.\n",
         "### Equivalence paragraph\n",
         f"Bounds: +/-1 JND = +/-{SESOI_BIAS['1 JND']} mm/m (median per-participant JND {jnd_ref:.2f}), +/-0.5 JND = +/-{SESOI_BIAS['0.5 JND']}; "
         f"JND +/-{SESOI_JND_ABS} mm/m. Welch t with Welch-Satterthwaite df.\n",
         f"Setup difference in bias = {sb.estimate:.2f} mm/m, 90% CI [{sb.ci90_lo:.2f}, {sb.ci90_hi:.2f}], t({sb.df:.1f}) = {sb.t_tost:.2f}, "
         f"p_TOST {_fmt_p(sb.p_tost)} at +/-1 JND; p_TOST {_fmt_p(sb2.p_tost)} at +/-0.5 JND. "
         f"Each finger vs 0 at +/-0.5 JND: all p_TOST <= {zb2.p_tost.max():.3f} ({'all equivalent' if zb2.equivalent.all() else 'NOT all equivalent'}). "
         f"Six finger pairs at +/-1 JND: {'all equivalent' if vb.all_six_pass else 'NOT all equivalent'} (Holm-corrected p_TOST max {_fmt_p(vb.max_p_holm)}).\n",
         f"JND setup difference = {sj.estimate:.2f} mm/m, 90% CI [{sj.ci90_lo:.2f}, {sj.ci90_hi:.2f}], p_TOST {_fmt_p(sj.p_tost)}; "
         f"Welch t({sj.df:.1f}) = {sj.t_nhst:.2f}, p = {sj.p_nhst:.3f}; BF01 = {bt.loc['JND','BF01 (r=0.707)']:.2f}. "
         f"Middle - ring JND = {mrt.estimate:.2f} [{mrt.ci90_lo:.2f}, {mrt.ci90_hi:.2f}], BF10 = {pj.loc['Middle-Ring','BF10']:.0f}; "
         f"index - little BF01 = {pj.loc['Index-Little','BF01']:.1f}. Pairs that differed (90% CI excludes 0): "
         + (", ".join(diffj.pair) if len(diffj) else "none") + ".\n",
         f"MDE at 80% power: {mb.MDE:.2f} mm/m (bias), {mj.MDE:.2f} mm/m (JND).\n",
         f"Bias BF01 (setup, r = 0.707) = {bt.loc['Bias','BF01 (r=0.707)']:.2f}; finger pairs BF01 "
         f"{bf_pairs[bf_pairs.DV=='Bias'].BF01.min():.1f}-{bf_pairs[bf_pairs.DV=='Bias'].BF01.max():.1f} (optional).\n"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(selection: str = "L_N_E", cohort: str | None = None, exclude: tuple[str, ...] | None = None,
        extra_sensitivity_exclusions: tuple[str, ...] = (), out_root: str = OUT_ROOT,
        paper_panels=("bias", "jnd"), verbose: bool = True) -> dict:
    exclude = excluded_subjects() if exclude is None else tuple(exclude)
    df = load_fits(data_path(selection), exclude)
    c = cohort_summary(df)
    cohort = cohort or f"{selection}_N{c['N']}"
    P = output_dirs(out_root, cohort)
    res = {"data": df, "cohort": cohort, "dirs": P, "excluded": exclude, "summary": c,
           "jnd_ref": jnd_reference(df)}
    res["step0"] = step0_consistency(df, pooled_path(selection))
    res["step2"] = step2_feasibility(df)
    res["setup"] = step3_setup(df)
    res["pairs"] = step3_finger_pairs(df)
    res["verdict"] = finger_factor_verdict(res["pairs"])
    res["zero"] = step3_bias_vs_zero(df)
    res["crosscheck_max_abs_diff"] = crosscheck_pingouin(res["setup"], res["pairs"], res["zero"])
    an = companion_anova(df)
    res["anova"], res["posthoc"], res["robust"] = an["anova"], an["posthoc"], an["robust"]
    res["bf_ttest"] = step4_bf_setup_ttest(df)
    res["bf_pairs"] = step4_bf_finger_pairs(df)
    res["mde"] = step5_mde(df)
    if extra_sensitivity_exclusions:
        res["sensitivity"] = sensitivity_compare(df, list(extra_sensitivity_exclusions))
    for k in ("step0", "step2", "setup", "pairs", "verdict", "zero", "anova", "posthoc", "robust",
              "bf_ttest", "bf_pairs", "mde", "sensitivity"):
        if k in res:
            res[k].to_csv(os.path.join(P["csv"], f"{k}.csv"), index=False)
    res["jasp_csv"] = export_for_jasp(df, os.path.join(P["csv"], "for_jasp_wide.csv"))
    res["paper_figure"] = paper_figure(res["setup"], res["pairs"], res["zero"],
                                       os.path.join(P["figures"], "equivalence_paper.png"), panels=paper_panels)
    res["paper_figure_bias_only"] = paper_figure(res["setup"], res["pairs"], res["zero"],
                                                 os.path.join(P["figures"], "equivalence_paper_bias.png"), panels=("bias",))
    res["report"] = report_sentences(df, res["setup"], res["pairs"], res["verdict"], res["zero"], an,
                                     res["bf_ttest"], res["bf_pairs"], res["mde"], res["jnd_ref"])
    with open(os.path.join(P["root"], "report_sentences.md"), "w", encoding="utf-8") as fh:
        fh.write(res["report"])
    if verbose:
        print(f"cohort {cohort}: N = {c['N']} ({c['n_L']} L, {c['n_N']} N), {c['fits']} fits; excluded {exclude}")
        print(res["report"])
        print("Outputs under", P["root"])
    return res


if __name__ == "__main__":
    run(extra_sensitivity_exclusions=("L_E_6", "L_E_18"))
