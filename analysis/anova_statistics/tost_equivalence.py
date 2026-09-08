"""
TOST equivalence test for finger-invariance of stiffness perception
===================================================================

WHY THIS EXISTS
---------------
The mixed-design ANOVA (see anova_statistics.py) finds *no significant* effect of
finger on PSE bias or JND. A non-significant test is "absence of evidence", not
"evidence of absence". To make the positive claim -- that the fingers are
perceptually equivalent -- we run a Two One-Sided Tests (TOST) equivalence test.

TOST logic (Lakens, 2017; Schuirmann, 1987):
  Pick a smallest effect size of interest (SESOI). Here SESOI = +/- 5 mm/m, the
  same +/-5 mm/m band the paper already uses to call a participant "unbiased"
  (about 6% of the 85 mm/m standard). Two one-sided t-tests then ask whether the
  effect is reliably *inside* (-SESOI, +SESOI). If the 90% CI of the effect lies
  entirely within the band (equivalently max(p_lower, p_upper) < alpha), we
  declare statistical equivalence.

WHAT IT TESTS
-------------
  (A) One-sample TOST per finger: is that finger's mean PSE bias within +/-5 of 0?
  (B) Paired TOST for every finger pair: is the per-subject PSE-bias difference
      within +/-5?
  (C) Welch two-sample TOST for Setup (L = air-slide vs N = natural), Lakens
      2017 eq. 3-4: is the L - N difference in bias and in JND within the
      equivalence bounds? Run on the per-subject mean over fingers (one value
      per participant, so the test unit matches the ANOVA) and within each
      finger. Reported at two pre-specified bounds: the primary +/-5 mm/m band
      and a secondary +/-10 mm/m band (a difference smaller than the 12 mm/m
      spacing between adjacent comparison stimuli). A Bayes factor (BF10,
      JZS default prior via pingouin, when installed) is given next to each
      TOST so frequentist and Bayesian evidence for the null sit side by side,
      as Lakens recommends.
All are run twice:
  - "all"   : every subject x finger fit, including degenerate ones.
  - "clean" : dropping fits flagged excluded_from_group_analysis, and PSE biases
              outside the tested comparison range (|bias| > 60 mm/m), which are
              off-scale artifacts of the psignifit fallback (see the paper's
              Statistical-analysis OPEN note). This mirrors the ANOVA sensitivity
              analysis and is the intended primary reading.

INPUT  : the same filtered psychophysics summary the ANOVA reads
         (analysis/psychophysics/results/L_N_E/_working/pse_jnd_by_subject_finger.csv,
         via anova_statistics.load_data), so the cohort is identical to the ANOVA
OUTPUT : analysis/anova_statistics/results/equivalence/
           tost_bias_one_sample.csv
           tost_bias_pairwise.csv
           tost_setup_L_vs_N.csv
           per_finger_bias_descriptives.csv
         and a printed summary.

Dependencies: numpy, pandas, scipy (same stack as anova_statistics.py);
pingouin is optional and only adds the BF10 column.
Run:  uv run python analysis/anova_statistics/tost_equivalence.py
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# --- configuration ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "equivalence")
ANOVA_DATA_SOURCE = "L_N_E"   # same psychophysics source as the ANOVA notebook

SESOI = 5.0          # smallest effect size of interest, mm/m (the +/-5 band)
SESOI_SETUP = (5.0, 10.0)  # Setup L-vs-N bounds: primary +/-5, secondary +/-10 mm/m
ALPHA = 0.05         # equivalence declared if max(p_lower, p_upper) < ALPHA
BIAS_VALID_ABS = 60.0  # |bias| beyond the tested +/-60 mm/m range = off-scale fit
FINGERS = ["I", "M", "R", "P"]
FINGER_NAME = {"I": "Index", "M": "Middle", "R": "Ring", "P": "Pinky"}
SYSTEMS = ("L", "N")   # L = air-slide (device), N = natural

try:  # optional: Bayes factor for the Setup comparison
    import pingouin as _pg
except Exception:  # pragma: no cover - pingouin is not in every environment
    _pg = None


def load_long(source: str = ANOVA_DATA_SOURCE) -> pd.DataFrame:
    """Load the SAME per-subject x finger summary the mixed-design ANOVA uses.

    The psychophysics pipeline has already applied the participant exclusions
    reported in the paper (incomplete sessions; fewer than two fingers with
    >= 55% success), so "all" here is exactly the ANOVA's main cohort.
    """
    import anova_statistics as A
    df = A.load_data(source)
    cols = ["Subject", "System", "Finger", "Bias", "JND",
            "fit_warning", "excluded_from_group_analysis"]
    df = df[[c for c in cols if c in df.columns]].copy()
    # one fit per (Subject, Finger); keep the first if duplicated
    df = df.drop_duplicates(subset=["Subject", "Finger"], keep="first")
    df["Bias"] = pd.to_numeric(df["Bias"], errors="coerce")
    df["JND"] = pd.to_numeric(df["JND"], errors="coerce")
    return df


def clean_mask(df: pd.DataFrame) -> pd.Series:
    """Rows kept for the 'clean' (sensitivity) analysis."""
    excluded = df.get("excluded_from_group_analysis", False)
    excluded = excluded.astype(str).str.lower().eq("true") if excluded is not False else False
    off_scale = df["Bias"].abs() > BIAS_VALID_ABS
    return (~excluded) & (~off_scale) & df["Bias"].notna()


def tost_one_sample(x: np.ndarray, sesoi: float):
    """One-sample TOST of mean(x) against the interval (-sesoi, +sesoi)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    df = n - 1
    # H0_lower: mean <= -sesoi  ->  reject if mean reliably > -sesoi
    t_lower = (mean - (-sesoi)) / se
    p_lower = stats.t.sf(t_lower, df)          # P(T > t_lower)
    # H0_upper: mean >= +sesoi  ->  reject if mean reliably < +sesoi
    t_upper = (mean - sesoi) / se
    p_upper = stats.t.cdf(t_upper, df)         # P(T < t_upper)
    p_tost = max(p_lower, p_upper)
    # 90% CI (equivalent to the two one-sided 95% tests)
    tcrit = stats.t.ppf(1 - ALPHA, df)
    ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se
    return dict(n=n, mean=mean, sd=sd, se=se,
                p_lower=p_lower, p_upper=p_upper, p_tost=p_tost,
                ci90_lo=ci_lo, ci90_hi=ci_hi,
                equivalent=bool(p_tost < ALPHA))


def tost_welch_two_sample(a: np.ndarray, b: np.ndarray, sesoi: float):
    """Two-sample TOST of mean(a) - mean(b) against (-sesoi, +sesoi).

    Welch's unequal-variance form (Lakens 2017, eq. 3) with Satterthwaite
    degrees of freedom (eq. 4). Lakens recommends this form by default because
    the two groups need not share a variance.
    """
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    diff = float(np.mean(a) - np.mean(b))
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    se = float(np.sqrt(v1 + v2))
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    t_lower = (diff + sesoi) / se            # H0_lower: diff <= -sesoi
    t_upper = (diff - sesoi) / se            # H0_upper: diff >= +sesoi
    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    tcrit = stats.t.ppf(1 - ALPHA, df)
    ci_lo, ci_hi = diff - tcrit * se, diff + tcrit * se
    # Companion NHST (Welch t-test) so the four Lakens outcomes can be named.
    t_nhst = diff / se
    p_nhst = 2 * stats.t.sf(abs(t_nhst), df)
    return dict(n_L=n1, n_N=n2, mean_L=float(np.mean(a)), mean_N=float(np.mean(b)),
                sd_L=float(a.std(ddof=1)), sd_N=float(b.std(ddof=1)),
                diff=diff, se=se, df_welch=float(df),
                t_lower=float(t_lower), t_upper=float(t_upper),
                p_lower=float(p_lower), p_upper=float(p_upper), p_tost=float(p_tost),
                ci90_lo=float(ci_lo), ci90_hi=float(ci_hi),
                t_nhst=float(t_nhst), p_nhst=float(p_nhst),
                equivalent=bool(p_tost < ALPHA),
                different=bool(p_nhst < ALPHA))


def bayes_factor_10(a: np.ndarray, b: np.ndarray) -> float:
    """BF10 for an independent-samples Welch t-test (JZS default prior).

    Values below 1 favour the null; 1/3 to 1 is anecdotal, 1/10 to 1/3
    moderate evidence for no difference. NaN when pingouin is unavailable.
    """
    if _pg is None:
        return float("nan")
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    try:
        res = _pg.ttest(a, b, paired=False, correction=True)
        return float(res["BF10"].iloc[0])
    except Exception:  # pragma: no cover - defensive
        return float("nan")


def lakens_outcome(row) -> str:
    """Name the outcome using Lakens (2017) Figure 1 scenarios."""
    if row["equivalent"] and not row["different"]:
        return "A: equivalent, not different"
    if not row["equivalent"] and row["different"]:
        return "B: not equivalent, different"
    if row["equivalent"] and row["different"]:
        return "C: equivalent AND different (trivially small effect)"
    return "D: undetermined (neither)"


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_long()
    datasets = {"all": df, "clean": df[clean_mask(df)]}

    # --- descriptives -----------------------------------------------------
    desc_rows = []
    for label, d in datasets.items():
        for f in FINGERS:
            b = d.loc[d["Finger"] == f, "Bias"].to_numpy(float)
            b = b[np.isfinite(b)]
            j = d.loc[d["Finger"] == f, "JND"].to_numpy(float)
            j = j[np.isfinite(j)]
            desc_rows.append(dict(
                dataset=label, Finger=f, name=FINGER_NAME[f], n=b.size,
                bias_mean=np.mean(b) if b.size else np.nan,
                bias_sd=np.std(b, ddof=1) if b.size > 1 else np.nan,
                jnd_mean=np.mean(j) if j.size else np.nan,
                jnd_median=np.median(j) if j.size else np.nan,
            ))
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(os.path.join(OUT_DIR, "per_finger_bias_descriptives.csv"), index=False)

    # --- (A) one-sample TOST per finger ----------------------------------
    one_rows = []
    for label, d in datasets.items():
        for f in FINGERS:
            b = d.loc[d["Finger"] == f, "Bias"].to_numpy(float)
            res = tost_one_sample(b, SESOI)
            one_rows.append(dict(dataset=label, Finger=f, name=FINGER_NAME[f],
                                 sesoi=SESOI, **res))
    one = pd.DataFrame(one_rows)
    one.to_csv(os.path.join(OUT_DIR, "tost_bias_one_sample.csv"), index=False)

    # --- (B) paired TOST per finger pair ---------------------------------
    pair_rows = []
    for label, d in datasets.items():
        wide = d.pivot_table(index="Subject", columns="Finger", values="Bias")
        for fa, fb in combinations(FINGERS, 2):
            if fa not in wide or fb not in wide:
                continue
            diff = (wide[fa] - wide[fb]).to_numpy(float)
            diff = diff[np.isfinite(diff)]
            if diff.size < 2:
                continue
            res = tost_one_sample(diff, SESOI)
            pair_rows.append(dict(dataset=label, pair=f"{fa}-{fb}",
                                  sesoi=SESOI, **res))
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(os.path.join(OUT_DIR, "tost_bias_pairwise.csv"), index=False)

    # --- (C) Welch two-sample TOST: Setup L vs N ---------------------------
    setup_rows = []
    for label, d in datasets.items():
        for dv in ("Bias", "JND"):
            # Unit = participant: mean over the four fingers. Only participants
            # with all four fingers are kept, matching the complete-case rule
            # of the mixed-design ANOVA (so N matches the ANOVA table).
            wide = d.pivot_table(index=["Subject", "System"], columns="Finger",
                                 values=dv)
            complete = wide.dropna(subset=[f for f in FINGERS if f in wide])
            subj_mean = complete.mean(axis=1).reset_index(name=dv)
            groups = {"subject_mean": subj_mean}
            for f in FINGERS:
                groups[f] = d.loc[d["Finger"] == f, ["System", dv]]
            for unit, g in groups.items():
                a = g.loc[g["System"] == SYSTEMS[0], dv].to_numpy(float)
                b = g.loc[g["System"] == SYSTEMS[1], dv].to_numpy(float)
                if np.isfinite(a).sum() < 2 or np.isfinite(b).sum() < 2:
                    continue
                bf10 = bayes_factor_10(a, b)
                for sesoi in SESOI_SETUP:
                    res = tost_welch_two_sample(a, b, sesoi)
                    res["outcome"] = lakens_outcome(res)
                    setup_rows.append(dict(
                        dataset=label, DV=dv, unit=unit,
                        name=FINGER_NAME.get(unit, "mean over fingers"),
                        sesoi=sesoi, BF10=bf10, **res))
    setup = pd.DataFrame(setup_rows)
    setup.to_csv(os.path.join(OUT_DIR, "tost_setup_L_vs_N.csv"), index=False)

    # --- summary ----------------------------------------------------------
    def show(title, frame, key):
        print(f"\n=== {title} ===")
        with pd.option_context("display.width", 160,
                               "display.max_columns", 20,
                               "display.float_format", lambda v: f"{v:8.3f}"):
            cols = [key, "dataset", "n", "mean", "ci90_lo", "ci90_hi",
                    "p_tost", "equivalent"]
            print(frame[cols].to_string(index=False))

    print(f"SESOI = +/-{SESOI} mm/m, alpha = {ALPHA}")
    print(f"Loaded {len(df)} subject x finger fits; "
          f"clean set keeps {int(clean_mask(df).sum())}.")
    show("(A) One-sample TOST: finger bias within +/-5 mm/m of 0",
         one, "name")
    show("(B) Paired TOST: finger-pair bias difference within +/-5 mm/m",
         pair, "pair")
    print("\n=== (C) Welch TOST: Setup L - N (Lakens 2017 eq. 3-4), "
          "with BF10 (JZS; <1 favours no difference) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:8.3f}"):
        cols = ["dataset", "DV", "unit", "sesoi", "n_L", "n_N", "diff",
                "ci90_lo", "ci90_hi", "p_tost", "equivalent", "p_nhst",
                "BF10", "outcome"]
        print(setup[cols].to_string(index=False))
    print("\nDescriptives:")
    print(desc.to_string(index=False,
                         float_format=lambda v: f"{v:8.3f}"))
    print(f"\nWrote CSVs to {OUT_DIR}")
    return {"one_sample": one, "pairwise": pair, "setup": setup,
            "descriptives": desc}


if __name__ == "__main__":
    run()
