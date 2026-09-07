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
Both are run twice:
  - "all"   : every subject x finger fit, including degenerate ones.
  - "clean" : dropping fits flagged excluded_from_group_analysis, and PSE biases
              outside the tested comparison range (|bias| > 60 mm/m), which are
              off-scale artifacts of the psignifit fallback (see the paper's
              Statistical-analysis OPEN note). This mirrors the ANOVA sensitivity
              analysis and is the intended primary reading.

INPUT  : analysis/anova_statistics/results/oneway/subjects/csv/*__pse_jnd_by_finger.csv
OUTPUT : analysis/anova_statistics/results/equivalence/
           tost_bias_one_sample.csv
           tost_bias_pairwise.csv
           per_finger_bias_descriptives.csv
         and a printed summary.

Dependencies: numpy, pandas, scipy (same stack as anova_statistics.py).
Run:  uv run python analysis/anova_statistics/tost_equivalence.py
"""

from __future__ import annotations

import glob
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# --- configuration ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECT_CSV_GLOB = os.path.join(
    HERE, "results", "oneway", "subjects", "csv", "*__pse_jnd_by_finger.csv"
)
OUT_DIR = os.path.join(HERE, "results", "equivalence")

SESOI = 5.0          # smallest effect size of interest, mm/m (the +/-5 band)
ALPHA = 0.05         # equivalence declared if max(p_lower, p_upper) < ALPHA
BIAS_VALID_ABS = 60.0  # |bias| beyond the tested +/-60 mm/m range = off-scale fit
FINGERS = ["I", "M", "R", "P"]
FINGER_NAME = {"I": "Index", "M": "Middle", "R": "Ring", "P": "Pinky"}


def load_long() -> pd.DataFrame:
    """Concatenate the per-subject PSE/JND files into one long table."""
    files = sorted(glob.glob(SUBJECT_CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"No subject CSVs matched {SUBJECT_CSV_GLOB}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
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
    print("\nDescriptives:")
    print(desc.to_string(index=False,
                         float_format=lambda v: f"{v:8.3f}"))
    print(f"\nWrote CSVs to {OUT_DIR}")


if __name__ == "__main__":
    run()
