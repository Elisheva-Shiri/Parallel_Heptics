"""
Constrained-lapse re-fit of the 2AFC psychometric curves
========================================================

WHY THIS EXISTS
---------------
psignifit was unavailable, so the pipeline fell back to a 4-parameter-logistic
MLE with the lapse asymptotes free up to 0.20 (see fit_with_scipy_logistic() in
twoafc_psychophysics.py). Many per-subject fits pinned the lapse near 0.2, which
inflates the JND/Weber values and *reduces* the power to detect (or exclude) a
finger effect. The reviewers of the ICRA paper asked us to re-fit with a
constrained lapse before making the finger-invariance claim.

WHAT THIS DOES
--------------
Re-fits every subject x finger psychometric curve straight from the raw 2AFC
trials, twice:
  - "orig"      : lapse asymptotes bounded (0, 0.20)   [reproduces the pipeline]
  - "constrained": lapse asymptotes bounded (0, 0.06)  [the requested re-fit]
Everything else -- the logistic model, the multi-start L-BFGS-B MLE, the PSE
(x at P=0.5) and JND ((x75 - x25)/2) definitions -- is copied verbatim from
twoafc_psychophysics.py so the only change is the lapse ceiling. It then:
  1. reports the mean fitted lapse before/after,
  2. writes per-subject Bias(=PSE-85)/JND tables for both variants, and
  3. re-runs the TOST equivalence test (imported from
     anova_statistics/tost_equivalence.py) on the constrained fits.

INPUT : analysis/psychophysics/results/L_N_E/csv/all/shared/clean_trials.csv
OUTPUT: analysis/psychophysics/results/constrained_lapse_refit/
          pse_jnd_by_subject_finger__orig.csv
          pse_jnd_by_subject_finger__constrained.csv
          lapse_before_after_summary.csv

Dependencies: numpy, pandas, scipy. Run:
  uv run python analysis/psychophysics/refit_constrained_lapse.py
"""

from __future__ import annotations

import math
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TRIALS = os.path.join(HERE, "results", "L_N_E", "csv", "all", "shared", "clean_trials.csv")
OUT_DIR = os.path.join(HERE, "results", "constrained_lapse_refit")

STANDARD = 85.0
MIN_TRIALS_PER_FIT = 12
MIN_LEVELS_PER_FIT = 3
BIAS_VALID_ABS = 60.0
SESOI = 5.0
FINGERS = ["I", "M", "R", "P"]

# import the reviewed TOST helper (same-repo, no heavy import)
sys.path.insert(0, os.path.join(REPO, "analysis", "anova_statistics"))
from tost_equivalence import tost_one_sample  # noqa: E402


# --- model + MLE, copied verbatim from twoafc_psychophysics.py --------------
def logistic4(x, mu, scale, lapse_low, lapse_high):
    z = (np.asarray(x, float) - mu) / max(scale, 1e-12)
    s = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    return lapse_low + (1 - lapse_low - lapse_high) * s


def x_at_probability(q, mu, scale, lapse_low, lapse_high):
    amp = 1 - lapse_low - lapse_high
    if amp <= 0:
        return np.nan
    y = (q - lapse_low) / amp
    if y <= 0 or y >= 1:
        return np.nan
    return float(mu + scale * math.log(y / (1 - y)))


def _nll(params, x, k, n, eps=1e-9):
    mu, log_scale, ll, lh = params
    scale = float(np.exp(log_scale))
    if ll + lh >= 0.45:
        return 1e9
    p = np.clip(logistic4(x, float(mu), scale, float(ll), float(lh)), eps, 1 - eps)
    return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def _fit(x, k, n, lapse_max):
    """Multi-start L-BFGS-B MLE; identical to the module but with a tunable
    lapse ceiling."""
    from scipy.optimize import minimize
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_range = max(x_max - x_min, 1.0)
    y = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    mu_guess = float(x[int(np.nanargmin(np.abs(y - 0.5)))]) if np.all(np.isfinite(y)) else float(np.median(x))
    bounds = [
        (x_min - x_range, x_max + x_range),
        (math.log(max(x_range / 200, 1e-3)), math.log(max(x_range * 10, 1.0))),
        (0.0, lapse_max),
        (0.0, lapse_max),
    ]
    l0 = min(0.02, lapse_max)
    starts = [
        [mu_guess, math.log(max(x_range / 6, 1e-3)), l0, l0],
        [float(np.median(x)), math.log(max(x_range / 4, 1e-3)), min(0.01, lapse_max), min(0.01, lapse_max)],
        [STANDARD, math.log(max(x_range / 5, 1e-3)), min(0.03, lapse_max), min(0.03, lapse_max)],
        [mu_guess, math.log(max(x_range / 10, 1e-3)), lapse_max, lapse_max],
    ]
    best = None
    for s in starts:
        r = minimize(_nll, s, args=(x, k, n), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    mu, scale = float(best.x[0]), float(np.exp(best.x[1]))
    ll, lh = float(best.x[2]), float(best.x[3])
    pse = x_at_probability(0.5, mu, scale, ll, lh)
    x25 = x_at_probability(0.25, mu, scale, ll, lh)
    x75 = x_at_probability(0.75, mu, scale, ll, lh)
    jnd = (x75 - x25) / 2 if np.isfinite(x25) and np.isfinite(x75) else np.nan
    return dict(pse=pse, jnd=jnd, lapse_rate=ll + lh, mu=mu, scale=scale)


# --- driver -----------------------------------------------------------------
def system_of(subject: str) -> str:
    return "L" if subject.upper().startswith("L") else "N"


def fit_variant(trials: pd.DataFrame, lapse_max: float) -> pd.DataFrame:
    rows = []
    for (subj, finger), g in trials.groupby(["subject_id", "comparison_finger"]):
        agg = (g.groupby("comparison_value")
                 .agg(n_trials=("response_comparison_greater", "size"),
                      k=("response_comparison_greater", "sum"))
                 .reset_index())
        if len(agg) < MIN_LEVELS_PER_FIT or agg["n_trials"].sum() < MIN_TRIALS_PER_FIT:
            continue
        f = _fit(agg["comparison_value"].to_numpy(float),
                 agg["k"].to_numpy(float),
                 agg["n_trials"].to_numpy(float),
                 lapse_max)
        bias = f["pse"] - STANDARD if np.isfinite(f["pse"]) else np.nan
        excluded = (not np.isfinite(bias)) or abs(bias) > BIAS_VALID_ABS or (not np.isfinite(f["jnd"]))
        rows.append(dict(Subject=subj, System=system_of(subj), Finger=finger,
                         Bias=bias, JND=f["jnd"], lapse_rate=f["lapse_rate"],
                         excluded_from_group_analysis=bool(excluded)))
    return pd.DataFrame(rows)


def tost_block(df: pd.DataFrame, label: str):
    clean = df[~df["excluded_from_group_analysis"] & df["Bias"].notna()]
    print(f"\n--- TOST on {label} fits (clean n/finger shown), SESOI +/-{SESOI} ---")
    for f in FINGERS:
        b = clean.loc[clean["Finger"] == f, "Bias"].to_numpy(float)
        if b.size < 2:
            continue
        r = tost_one_sample(b, SESOI)
        print(f"  {f}: n={r['n']:2d} mean={r['mean']:+6.2f} "
              f"90%CI[{r['ci90_lo']:+6.2f},{r['ci90_hi']:+6.2f}] "
              f"p_TOST={r['p_tost']:.3f} equiv={r['equivalent']}")
    wide = clean.pivot_table(index="Subject", columns="Finger", values="Bias")
    print("  pairwise:")
    for a, c in combinations(FINGERS, 2):
        if a not in wide or c not in wide:
            continue
        d = (wide[a] - wide[c]).to_numpy(float)
        d = d[np.isfinite(d)]
        if d.size < 2:
            continue
        r = tost_one_sample(d, SESOI)
        print(f"    {a}-{c}: n={r['n']:2d} mean={r['mean']:+6.2f} "
              f"90%CI[{r['ci90_lo']:+6.2f},{r['ci90_hi']:+6.2f}] equiv={r['equivalent']}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trials = pd.read_csv(TRIALS, low_memory=False)
    trials = trials[trials["excluded_from_fit"].astype(str).str.lower() != "true"]
    trials = trials.dropna(subset=["comparison_value", "comparison_finger",
                                   "response_comparison_greater"])
    print(f"Loaded {len(trials)} trials; "
          f"{trials['subject_id'].nunique()} subjects.")

    variants = {}
    for label, lapse_max in [("orig", 0.20), ("constrained", 0.06)]:
        df = fit_variant(trials, lapse_max)
        df.to_csv(os.path.join(OUT_DIR, f"pse_jnd_by_subject_finger__{label}.csv"),
                  index=False)
        variants[label] = df

    # before/after lapse + JND summary
    summ = []
    for label, df in variants.items():
        clean = df[~df["excluded_from_group_analysis"]]
        summ.append(dict(variant=label, n_fits=len(df),
                         mean_lapse=df["lapse_rate"].mean(),
                         median_lapse=df["lapse_rate"].median(),
                         mean_JND_clean=clean["JND"].mean(),
                         median_JND_clean=clean["JND"].median()))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT_DIR, "lapse_before_after_summary.csv"), index=False)
    print("\n=== lapse / JND before vs after ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    for label in ("orig", "constrained"):
        tost_block(variants[label], label)

    print(f"\nWrote CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
