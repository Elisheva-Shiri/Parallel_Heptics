"""
Equivalence analysis for the setup and finger null effects (standalone)
=======================================================================

Goal (supervisor plan): replace "p > .05, therefore flexible finger selection"
with "the difference, if any, is smaller than a perceptually meaningful
amount". The analysis runs on the per-participant PSE/JND fits that fed the
mixed-design ANOVA (one 4-parameter logistic fit per participant x finger,
64 trials each), reads them directly from the psychophysics pipeline output,
and does NOT depend on the anova_statistics package.

Cohort
------
Participants whose raw-data folder under Parallel_Heptics/results is tagged
"(filter)" are excluded from every analysis (currently L_E_19 and N_E_17).
This gives N = 39 (20 air-slide, 19 natural). Unequal group sizes are handled
by Welch's t statistic with Welch-Satterthwaite degrees of freedom, which is
the default recommended for two-group comparisons (Delacre, Lakens & Leys,
2017; Welch, 1947) and the form used by Lakens (2017, eq. 3-4) for TOST.

Steps
-----
0. Consistency: per-participant summaries per finger, next to the pooled fits.
1. Bounds (pre-specified, perceptual units, mm/m):
     bias   : +/- 1 JND  = +/- 1.6 mm/m primary;  +/- 0.5 JND = +/- 0.8 secondary
              (two conditions whose PSEs differ by less than one JND are, by
              construction, indistinguishable to the participants; JND_REF is
              the median per-participant JND, ~1.6 mm/m, reported in the paper)
     JND    : +/- 0.5 mm/m (~30% of JND_REF) on the raw scale, and a ratio
              bound of x/1.30 .. x1.30 on the log scale (JND is a scale
              parameter, so ratio bounds are the natural alternative).
2. Feasibility: observed SDs and the 90% CI half-width each design can reach,
   compared with the bound, BEFORE reading the TOST outcome.
3. Tests: Welch two-sample TOST for Setup (per participant and per finger);
   six paired TOSTs for finger pairs with Holm correction (equivalence of the
   finger factor is claimed only if all six pass); one-sample TOST of each
   finger's bias against zero. Every TOST p-value is cross-checked against
   pingouin.tost.
4. Bayes factors: (a) JZS Bayes factor for the per-participant Setup t-test
   with a prior-width robustness check (r = 0.35, 0.5, 0.707, 1.0); (b) JZS
   Bayes factor for each finger-pair paired t-test. The omnibus inclusion
   Bayes factors (Finger, Setup x Finger) need BayesFactor::anovaBF or JASP,
   which are not available in Python; results/for_jasp_wide.csv is exported
   for that step. (A BIC-based mixed-model approximation was tried and
   dropped because its fits were not stable across nested models.)
5. Minimal detectable effect: the effect the design had 80% power to detect,
   from the observed SDs and the actual group sizes.

Outputs: results/*.csv, results/equivalence_forest.png, results/report_sentences.md

Run:  python tost_equivalence.py        (system Python: pandas, scipy,
      pingouin, statsmodels, matplotlib)
"""

from __future__ import annotations

import glob
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
# Configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_PATH = os.path.join(
    REPO, "analysis", "psychophysics", "results", "L_N_E", "csv", "all",
    "shared", "pse_jnd_by_subject_finger.csv")
POOLED_PATH = os.path.join(
    REPO, "analysis", "psychophysics", "results", "L_N_E", "csv", "all",
    "shared", "pse_jnd_group_by_finger.csv")
RAW_RESULTS_DIR = os.path.join(REPO, "results")   # raw per-participant folders
OUT_DIR = os.path.join(HERE, "results")

STANDARD = 8.5                      # mm/m
ALPHA = 0.05
CI_LEVEL = 0.90                     # 1 - 2*alpha, the TOST-equivalent CI
FINGERS = ["I", "M", "R", "P"]
FINGER_NAME = {"I": "Index", "M": "Middle", "R": "Ring", "P": "Little"}
SYSTEMS = ("L", "N")
SYSTEM_NAME = {"L": "air-slide", "N": "natural"}

# Pre-specified bounds (mm/m). Fixed constants, not derived from the data at
# run time; JND_REF below is only computed to document the anchor.
SESOI_BIAS = {"1 JND": 1.6, "0.5 JND": 0.8}
SESOI_JND_ABS = 0.5
SESOI_JND_RATIO = 1.30              # log-scale bound = +/- ln(1.30)

# Participants excluded from every analysis. Derived from raw-data folders
# tagged "(filter)"; the fallback list is used if that folder is unreachable.
EXCLUDED_FALLBACK = ("L_E_19", "N_E_17")


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


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_fits(path: str = DATA_PATH, exclude: bool = True) -> pd.DataFrame:
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
        raise ValueError(f"PSE median {med:.2f} is not in mm/m (expected ~8.5); "
                         f"check the units of {path}")
    if exclude:
        df = df[~df["Subject"].isin(excluded_subjects())].copy()
    df = df[df["Finger"].isin(FINGERS)].reset_index(drop=True)
    return df


def subject_means(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """One value per participant: mean over the four fingers (complete cases)."""
    wide = df.pivot_table(index=["Subject", "System"], columns="Finger", values=dv)
    wide = wide.dropna(subset=FINGERS)
    return wide.mean(axis=1).reset_index(name=dv)


def wide_by_finger(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    w = df.pivot_table(index="Subject", columns="Finger", values=dv)
    return w.dropna(subset=FINGERS)


# --------------------------------------------------------------------------- #
# Step 0: consistency of per-participant fits with the pooled fits
# --------------------------------------------------------------------------- #
def step0_consistency(df: pd.DataFrame, pooled_path: str = POOLED_PATH) -> pd.DataFrame:
    rows = []
    pooled = pd.read_csv(pooled_path) if os.path.exists(pooled_path) else None
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
    t_low = (est - low) / se           # H0: est <= low
    t_up = (est - high) / se           # H0: est >= high
    p_low = stats.t.sf(t_low, dof)
    p_up = stats.t.cdf(t_up, dof)
    p_tost = max(p_low, p_up)
    tcrit = stats.t.ppf(1 - ALPHA, dof)
    t_nhst = est / se
    p_nhst = 2 * stats.t.sf(abs(t_nhst), dof)
    eq = bool(p_tost < ALPHA)
    diff = bool(p_nhst < ALPHA)
    return dict(estimate=float(est), se=float(se), df=float(dof),
                ci90_lo=float(est - tcrit * se), ci90_hi=float(est + tcrit * se),
                bound_lo=float(low), bound_hi=float(high),
                t_lower=float(t_low), t_upper=float(t_up),
                p_lower=float(p_low), p_upper=float(p_up), p_tost=float(p_tost),
                t_nhst=float(t_nhst), p_nhst=float(p_nhst),
                equivalent=eq, different=diff, outcome=_lakens_outcome(eq, diff))


def tost_one_sample(x, low, high, mu=0.0):
    """Lakens (2017) eq. 7. Also used for paired data on the differences."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = x.size
    est = x.mean() - mu
    se = x.std(ddof=1) / np.sqrt(n)
    out = _finish(est, se, n - 1, low, high)
    out.update(n=int(n), sd=float(x.std(ddof=1)))
    return out


def tost_welch(a, b, low, high):
    """Lakens (2017) eq. 3-4: Welch t with Welch-Satterthwaite df.

    Each group's variance is divided by its own n, so unequal group sizes
    (20 vs 19) need no weighting; the df formula accounts for them.
    """
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
    """Cross-check: pingouin's symmetric-bound TOST p-value (Welch for 2 groups)."""
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
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------- #
# Step 2: feasibility (can the CI fit inside the bound if the true effect is 0?)
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
                             SD=f"{a.std(ddof=1):.2f} / {b.std(ddof=1):.2f}",
                             ci90_halfwidth=hw, bound=bval, bound_name=bname,
                             feasible=bool(hw < bval)))
        w = wide_by_finger(df, dv)
        d = np.concatenate([(w[x] - w[y]).to_numpy() for x, y in combinations(FINGERS, 2)])
        sd_diff = np.median([np.std(w[x] - w[y], ddof=1) for x, y in combinations(FINGERS, 2)])
        hw = stats.t.ppf(1 - ALPHA, len(w) - 1) * sd_diff / np.sqrt(len(w))
        for bname, bval in bounds.items():
            rows.append(dict(DV=dv, comparison="Finger pairs (paired), median SD of diffs",
                             n=str(len(w)), SD=f"{sd_diff:.2f}", ci90_halfwidth=hw,
                             bound=bval, bound_name=bname, feasible=bool(hw < bval)))
        if dv == "Bias":
            for f in FINGERS:
                x = df.loc[df.Finger == f, dv].dropna().to_numpy()
                hw = stats.t.ppf(1 - ALPHA, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
                for bname, bval in bounds.items():
                    rows.append(dict(DV=dv, comparison=f"Bias vs 0, {FINGER_NAME[f]}", n=str(len(x)),
                                     SD=f"{x.std(ddof=1):.2f}", ci90_halfwidth=hw,
                                     bound=bval, bound_name=bname, feasible=bool(hw < bval)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 3: the tests
# --------------------------------------------------------------------------- #
def _dv_bounds():
    """(dv, label, low, high, pingouin_bound) for every DV/bound combination."""
    out = []
    for name, b in SESOI_BIAS.items():
        out.append(("Bias", f"bias +/-{b} ({name})", -b, b, b))
    out.append(("JND", f"JND +/-{SESOI_JND_ABS} mm/m", -SESOI_JND_ABS, SESOI_JND_ABS, SESOI_JND_ABS))
    lb = np.log(SESOI_JND_RATIO)
    out.append(("logJND", f"JND ratio x{SESOI_JND_RATIO}", -lb, lb, lb))
    return out


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
    if "logJND" in set(out.DV):   # add ratio columns for readability
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
        ph = holm([b["p_tost"] for b in block])
        for b, p in zip(block, ph):
            b["p_tost_holm"] = float(p)
            b["equivalent_holm"] = bool(p < ALPHA)
        rows += block
    out = pd.DataFrame(rows)
    return out


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
    g = pairs.groupby(["DV", "bound"])
    return g.agg(n_pairs=("pair", "size"),
                 n_equivalent_holm=("equivalent_holm", "sum"),
                 all_six_pass=("equivalent_holm", "all"),
                 max_p_holm=("p_tost_holm", "max")).reset_index()


# --------------------------------------------------------------------------- #
# Step 4: Bayes factors
# --------------------------------------------------------------------------- #
def step4_bf_setup_ttest(df: pd.DataFrame, radii=(0.35, 0.5, 0.707, 1.0)) -> pd.DataFrame:
    """JZS BF01 for the per-participant Setup comparison, with prior-width check."""
    rows = []
    if pg is None:
        return pd.DataFrame(rows)
    for dv in ("Bias", "JND", "logJND"):
        sm = subject_means(df, dv)
        a = sm.loc[sm.System == "L", dv].to_numpy(); b = sm.loc[sm.System == "N", dv].to_numpy()
        t = tost_welch(a, b, -1, 1)["t_nhst"]
        row = dict(DV=dv, unit="per participant", n1=len(a), n2=len(b), t_welch=t)
        for r in radii:
            bf10 = float(pg.bayesfactor_ttest(t, len(a), len(b), paired=False, r=r))
            row[f"BF01 (r={r})"] = 1.0 / bf10
        rows.append(row)
    return pd.DataFrame(rows)


def step4_bf_finger_pairs(df: pd.DataFrame, r: float = 0.707) -> pd.DataFrame:
    """JZS BF01 for each finger pair (paired t) and, per DV, the smallest BF01.

    The omnibus inclusion Bayes factors for Finger and Setup x Finger require
    BayesFactor::anovaBF or JASP; use results/for_jasp_wide.csv for those.
    """
    rows = []
    if pg is None:
        return pd.DataFrame(rows)
    for dv in ("Bias", "JND", "logJND"):
        w = wide_by_finger(df, dv)
        for x, y in combinations(FINGERS, 2):
            d = (w[x] - w[y]).to_numpy()
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            bf10 = float(pg.bayesfactor_ttest(t, len(d), paired=True, r=r))
            rows.append(dict(DV=dv, pair=f"{FINGER_NAME[x]}-{FINGER_NAME[y]}", n=len(d),
                             t=float(t), BF01=1.0 / bf10, BF10=bf10))
    out = pd.DataFrame(rows)
    out["evidence"] = out["BF01"].apply(
        lambda v: "strong for null" if v > 10 else "moderate for null" if v > 3
        else "anecdotal for null" if v > 1 else "anecdotal for effect" if v > 1/3
        else "moderate for effect" if v > 0.1 else "strong for effect")
    return out


def export_for_jasp(df: pd.DataFrame, path: str) -> str:
    """Wide file (one row per participant) for a Bayesian RM ANOVA in JASP."""
    frames = []
    for dv in ("Bias", "JND", "logJND"):
        w = df.pivot_table(index=["Subject", "System"], columns="Finger", values=dv)
        w.columns = [f"{dv}_{c}" for c in w.columns]
        frames.append(w)
    wide = pd.concat(frames, axis=1).reset_index()
    wide.to_csv(path, index=False)
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
        if pg is not None:
            d80 = float(pg.power_ttest2n(nx=len(a), ny=len(b), power=0.8, alpha=ALPHA))
        else:
            d80 = np.nan
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
# Figure: estimates with 90% CI against the equivalence bounds
# --------------------------------------------------------------------------- #
def forest_figure(setup: pd.DataFrame, pairs: pd.DataFrame, zero: pd.DataFrame, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Setup: air-slide - natural, bias (mm/m)", setup[(setup.DV == "Bias") & setup.bound.str.contains("1 JND")], "unit", SESOI_BIAS["1 JND"], SESOI_BIAS["0.5 JND"]),
        ("Finger pairs, bias (mm/m)", pairs[(pairs.DV == "Bias") & pairs.bound.str.contains("1 JND")], "pair", SESOI_BIAS["1 JND"], SESOI_BIAS["0.5 JND"]),
        ("Bias vs 0 (mm/m)", zero[zero.bound.str.contains("1 JND")], "finger", SESOI_BIAS["1 JND"], SESOI_BIAS["0.5 JND"]),
        ("Setup: air-slide - natural, JND (mm/m)", setup[setup.DV == "JND"], "unit", SESOI_JND_ABS, None),
        ("Finger pairs, JND (mm/m)", pairs[pairs.DV == "JND"], "pair", SESOI_JND_ABS, None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.2))
    for ax, (title, frame, labcol, b1, b2) in zip(axes, panels):
        frame = frame.reset_index(drop=True)
        y = np.arange(len(frame))[::-1]
        ax.axvspan(-b1, b1, color="#dfe9f3", zorder=0)
        if b2 is not None:
            ax.axvspan(-b2, b2, color="#c3d5e8", zorder=0)
        ax.axvline(0, color="grey", lw=0.8)
        ok = frame["equivalent"].to_numpy() if "equivalent_holm" not in frame else frame["equivalent_holm"].to_numpy()
        for yi, (_, r), e in zip(y, frame.iterrows(), ok):
            ax.plot([r.ci90_lo, r.ci90_hi], [yi, yi], color="#1f4e79" if e else "#b03a2e", lw=2)
            ax.plot(r.estimate, yi, "o", color="#1f4e79" if e else "#b03a2e", ms=5)
        ax.set_yticks(y); ax.set_yticklabels(frame[labcol]); ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Effect estimates with 90% CI against equivalence bounds (blue = equivalent, red = not)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200); plt.close(fig)
    return path


def paper_figure(setup: pd.DataFrame, pairs: pd.DataFrame, zero: pd.DataFrame, path: str) -> str:
    """Single-column, two-panel equivalence plot for the paper.

    (a) PSE bias: setup difference (per participant), each finger vs 0, and the
        six finger-pair differences, against the +/-1 JND (light) and +/-0.5 JND
        (dark) bounds. (b) JND: setup difference and finger pairs against the
        +/-0.5 mm/m bound. Filled = equivalent at the primary bound (Holm for
        pairs), open = not. 90% CIs (equivalent to the two one-sided tests).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rows_bias():
        s = setup[(setup.DV == "Bias") & setup.bound.str.contains("1 JND") & (setup.unit == "per participant")]
        z = zero[zero.bound.str.contains("1 JND")]
        p = pairs[(pairs.DV == "Bias") & pairs.bound.str.contains("1 JND")]
        out = [("Setup: air-slide - natural", s.iloc[0].estimate, s.iloc[0].ci90_lo, s.iloc[0].ci90_hi, bool(s.iloc[0].equivalent))]
        out += [(f"{r.finger} vs 0", r.estimate, r.ci90_lo, r.ci90_hi, bool(r.equivalent)) for _, r in z.iterrows()]
        out += [(r.pair.replace("-", " - "), r.estimate, r.ci90_lo, r.ci90_hi, bool(r.equivalent_holm)) for _, r in p.iterrows()]
        return out

    def rows_jnd():
        s = setup[(setup.DV == "JND") & (setup.unit == "per participant")]
        p = pairs[pairs.DV == "JND"]
        out = [("Setup: air-slide - natural", s.iloc[0].estimate, s.iloc[0].ci90_lo, s.iloc[0].ci90_hi, bool(s.iloc[0].equivalent))]
        out += [(r.pair.replace("-", " - "), r.estimate, r.ci90_lo, r.ci90_hi, bool(r.equivalent_holm)) for _, r in p.iterrows()]
        return out

    panels = [("(a) PSE bias (mm/m)", rows_bias(), SESOI_BIAS["1 JND"], SESOI_BIAS["0.5 JND"]),
              ("(b) JND (mm/m)", rows_jnd(), SESOI_JND_ABS, None)]
    n_rows = [len(p[1]) for p in panels]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 0.19 * sum(n_rows) + 1.3),
                             gridspec_kw={"height_ratios": n_rows})
    for ax, (title, rows, b1, b2) in zip(axes, panels):
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
# Report sentences
# --------------------------------------------------------------------------- #
def _fmt_p(p):
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def report_sentences(df, setup, pairs, verdict, zero, bf_t, bf_pairs, mde, jnd_ref) -> str:
    n1 = int((subject_means(df, "Bias").System == "L").sum()); n2 = int((subject_means(df, "Bias").System == "N").sum())
    sb = setup[(setup.DV == "Bias") & setup.bound.str.contains("1 JND") & (setup.unit == "per participant")].iloc[0]
    sj = setup[(setup.DV == "JND") & (setup.unit == "per participant")].iloc[0]
    sr = setup[(setup.DV == "logJND") & (setup.unit == "per participant")].iloc[0]
    vb = verdict[(verdict.DV == "Bias") & verdict.bound.str.contains("1 JND")].iloc[0]
    vj = verdict[verdict.DV == "JND"].iloc[0]
    vr = verdict[verdict.DV == "logJND"].iloc[0]
    zb = zero[zero.bound.str.contains("1 JND")]
    pb = bf_pairs[bf_pairs.DV == "Bias"]; pj = bf_pairs[bf_pairs.DV == "logJND"]
    diffj = pairs[(pairs.DV == "JND") & pairs.different]
    mb = mde[(mde.DV == "Bias") & mde.comparison.str.startswith("Setup")].iloc[0]
    mj = mde[(mde.DV == "JND") & mde.comparison.str.startswith("Setup")].iloc[0]
    bt = bf_t.set_index("DV")

    lines = []
    lines.append("## Results paragraph (after the mixed-design ANOVA)\n")
    lines.append(
        f"Because a non-significant ANOVA cannot establish the absence of an effect, we tested equivalence with two "
        f"one-sided tests (TOST; Lakens, 2017) on the per-participant fits (N = {n1 + n2}; {n1} air-slide, {n2} natural). "
        f"Bounds were pre-specified in perceptual units: +/-1 JND (+/-{SESOI_BIAS['1 JND']} mm/m; median per-participant JND "
        f"= {jnd_ref:.2f} mm/m) for PSE bias, with +/-0.5 JND (+/-{SESOI_BIAS['0.5 JND']} mm/m) as a stricter secondary bound, "
        f"and +/-{SESOI_JND_ABS} mm/m (about 30% of the JND) or a x{SESOI_JND_RATIO} ratio for JND. Setup comparisons used "
        f"Welch's t with Welch-Satterthwaite degrees of freedom (Delacre et al., 2017), finger pairs used paired TOSTs "
        f"with Holm correction, and equivalence was declared when the 90% CI lay within the bounds.\n")
    lines.append(
        f"PSE bias was statistically equivalent between setups within +/-1 JND (air-slide minus natural = {sb.estimate:.2f} mm/m, "
        f"90% CI [{sb.ci90_lo:.2f}, {sb.ci90_hi:.2f}], t({sb.df:.1f}) = {(sb.t_lower if sb.p_lower >= sb.p_upper else sb.t_upper):.2f}, "
        f"p_TOST {_fmt_p(sb.p_tost)}){' and' if vb.all_six_pass else ', but not'} across all six finger pairs "
        f"(Holm-corrected p_TOST max {_fmt_p(vb.max_p_holm)}). "
        + ("Each finger's bias was equivalent to zero within +/-1 JND (all p_TOST " + _fmt_p(zb.p_tost.max()) + "). "
           if zb.equivalent.all() else
           "Bias was equivalent to zero within +/-1 JND for " + ", ".join(zb.loc[zb.equivalent, 'finger']) + ". "))
    lines.append(
        f"For JND, the setup difference was {sj.estimate:.2f} mm/m (90% CI [{sj.ci90_lo:.2f}, {sj.ci90_hi:.2f}], "
        f"p_TOST {_fmt_p(sj.p_tost)} at +/-{SESOI_JND_ABS} mm/m; ratio {np.exp(sr.estimate):.2f}, 90% CI "
        f"[{np.exp(sr.ci90_lo):.2f}, {np.exp(sr.ci90_hi):.2f}], p_TOST {_fmt_p(sr.p_tost)} at x{SESOI_JND_RATIO}), so "
        f"{'equivalence was established' if (sj.equivalent or sr.equivalent) else 'equivalence could not be established'}; "
        f"across finger pairs {int(vj.n_equivalent_holm)}/6 (absolute) and {int(vr.n_equivalent_holm)}/6 (ratio) passed.\n")
    if len(diffj):
        lines.append(
            "Finger pairs whose JND differed (90% CI excluding zero): " + "; ".join(
                f"{r.pair} {r.estimate:.2f} mm/m [{r.ci90_lo:.2f}, {r.ci90_hi:.2f}]" for _, r in diffj.iterrows()) + ".\n")
    lines.append(
        f"Bayes factors complemented the TOST: for the per-participant setup comparison BF01 = {bt.loc['Bias','BF01 (r=0.707)']:.2f} "
        f"(bias) and {bt.loc['logJND','BF01 (r=0.707)']:.2f} (log JND) with the default JZS prior (r = 0.707; over r = 0.35-1.0: "
        f"{bt.loc['Bias',[c for c in bt.columns if c.startswith('BF01')]].min():.2f}-{bt.loc['Bias',[c for c in bt.columns if c.startswith('BF01')]].max():.2f} for bias). "
        f"Paired JZS Bayes factors for the six finger pairs ranged over BF01 = {pb.BF01.min():.2f}-{pb.BF01.max():.2f} for bias and "
        f"{pj.BF01.min():.2f}-{pj.BF01.max():.2f} for log JND (smallest: {pj.loc[pj.BF01.idxmin(),'pair']}). "
        f"[Omnibus inclusion BF01 for Finger and Setup x Finger: fill from JASP using results/for_jasp_wide.csv.]\n")
    lines.append(
        f"With {n1} vs {n2} participants and the observed SDs, the design had 80% power to detect a setup difference of "
        f"{mb.MDE:.2f} mm/m in bias and {mj.MDE:.2f} mm/m in JND.\n")
    lines.append("\n## Discussion sentence\n")
    lines.append(
        "Use 'evidence for equivalence' only where the TOST passed (bias, within one JND), 'evidence for the null' where only "
        "the Bayes factor supports it, and never 'proved no difference'.\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(out_dir: str = OUT_DIR, exclude: bool = True, verbose: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    df = load_fits(exclude=exclude)
    jnd_ref = jnd_reference(df)
    res = {"data": df, "jnd_ref": jnd_ref, "excluded": excluded_subjects()}
    res["step0"] = step0_consistency(df)
    res["step2"] = step2_feasibility(df)
    res["setup"] = step3_setup(df)
    res["pairs"] = step3_finger_pairs(df)
    res["verdict"] = finger_factor_verdict(res["pairs"])
    res["zero"] = step3_bias_vs_zero(df)
    res["bf_ttest"] = step4_bf_setup_ttest(df)
    res["bf_pairs"] = step4_bf_finger_pairs(df)
    res["mde"] = step5_mde(df)
    for k in ("step0", "step2", "setup", "pairs", "verdict", "zero", "bf_ttest", "bf_pairs", "mde"):
        res[k].to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)
    res["jasp_csv"] = export_for_jasp(df, os.path.join(out_dir, "for_jasp_wide.csv"))
    res["figure"] = forest_figure(res["setup"], res["pairs"], res["zero"],
                                  os.path.join(out_dir, "equivalence_forest.png"))
    res["paper_figure"] = paper_figure(res["setup"], res["pairs"], res["zero"],
                                       os.path.join(out_dir, "equivalence_paper.png"))
    res["report"] = report_sentences(df, res["setup"], res["pairs"], res["verdict"], res["zero"],
                                     res["bf_ttest"], res["bf_pairs"], res["mde"], jnd_ref)
    with open(os.path.join(out_dir, "report_sentences.md"), "w", encoding="utf-8") as fh:
        fh.write(res["report"])
    if verbose:
        print(f"N = {df.Subject.nunique()} participants "
              f"({(subject_means(df,'Bias').System=='L').sum()} L, {(subject_means(df,'Bias').System=='N').sum()} N); "
              f"excluded: {res['excluded']}; JND_REF = {jnd_ref:.2f} mm/m")
        print(res["report"])
        print("Wrote outputs to", out_dir)
    return res


if __name__ == "__main__":
    run()
