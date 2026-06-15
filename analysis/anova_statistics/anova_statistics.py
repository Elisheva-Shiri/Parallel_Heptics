"""
anova_statistics.py
===================

Self-contained statistics pipeline for a 2AFC stiffness-discrimination
psychophysics study.

This module is PURELY ADDITIVE. It does NOT recompute psychometric fits and
does NOT modify any existing analysis code. It consumes a frozen per-subject x
finger psychometric summary CSV and produces:

  * Data loading + validation (transparent, never silently drops rows).
  * Exploratory one-way ANOVA on the 8 combined System x Finger groups.
  * MAIN analysis: mixed-design ANOVA  System (between) x Finger (within),
    with Subject as the repeated-measures unit, for Bias and JND.
  * Planned contrasts: System L vs N within each finger (Holm-corrected).
  * Subject-respecting bootstrap confidence intervals.
  * Publication-style figures.
  * A methods/results text report.

Domain facts (kept verbatim where reported)
-------------------------------------------
  * 2AFC stiffness task. Standard S = 85; comparison values
    25, 40, 55, 70, 100, 115, 130, 145; predictor delta = C - 85.
  * Binary response y = 1 iff the participant judged the COMPARISON stiffer
    than the standard (NOT correct/incorrect, not side, not order, not
    physical truth). This is already coded upstream.

Psychometric model (frozen upstream; NOT recomputed here)
--------------------------------------------------------
Each subject x finger psychometric function is a LAPSE-AWARE yes/no-style
function with FOUR fitted parameters (mu, scale, lapse_low, lapse_high). It is
NOT a textbook 2AFC percent-correct model with a fixed guess rate. The fit is
to P(participant judged comparison C stiffer than standard S=85), i.e. P(y=1):

    P(y=1) = lapse_low + (1 - lapse_low - lapse_high) * F(delta; mu, scale),
    delta = C - 85,  F = a monotonic sigmoid.

Fitter provenance: psignifit was used when installed (preferred); otherwise a
custom lapse-aware fallback fitter was used. The source CSV records the fitter
per row in ``fit_method`` and the psignifit availability in
``psignifit_status``; this module reports a count of fits by fitter.

Derived measures (fit is in comparison-stiffness units, so these are
equivalent to delta-space):
  * PSE  = comparison value where P(y=1) = 0.5.
  * Bias = PSE - 85 = the delta at P(y=1) = 0.5 (perceptual shift; 0 = none).
  * JND  = (x75 - x25) / 2, where x25 / x75 are the comparison values at which
           the curve reaches 0.25 / 0.75 (sensitivity; smaller = finer).
    Because of the lapse parameters, x25 / x75 are estimable ONLY if 0.25 and
    0.75 lie within the attainable range [lapse_low, 1 - lapse_high]. When they
    do not, the upstream fitter returns NaN JND together with a fit_warning
    ("jnd_quantile_outside_lapse_range" / "pse_outside_lapse_range"); a
    non-estimable JND is therefore FLAGGED, not silently invalid.

Design factors
--------------
  * System = between-subject factor (each subject is in EXACTLY one system).
  * Finger = within-subject / repeated factor (4 fingers per subject).
  * Subject = repeated-measures unit.

Why a mixed-design ANOVA and NOT an ordinary two-way ANOVA
---------------------------------------------------------
An ordinary independent two-way ANOVA assumes all observations are
independent. Here the 4 finger measurements come from the SAME subject and are
therefore correlated (repeated measures). Treating them as independent would
violate the independence assumption and mis-estimate the error term. A
mixed-design ANOVA (System between, Finger within, Subject as the repeated
unit) is the correct classical model and is the MAIN analysis.

Author: additive analysis module (no existing code touched).
"""

from __future__ import annotations

import os
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Statistics
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Plotting (configured by the notebook; safe defaults here)
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# One-way (per-system finger) analysis helpers, colocated in THIS folder as
# ``oneway_anova.py``. We reuse its numpy/scipy stat + plot helpers (NOT its
# output-tree writer) so the one-way analysis appears under this module's flat
# results/ taxonomy, via run_oneway_flat() below. Guarded so the rest of the
# pipeline still imports if the helper module is somehow missing.
# CONNECTION (2026-06-14): ``oneway_anova.py`` is a REQUIRED sibling of this file
# (consumed by run_oneway_flat). The one-way analysis was merged into THIS
# notebook/package; keep oneway_anova.py here -- do not delete it.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import oneway_anova as owa  # noqa: E402
except Exception as _owa_exc:  # pragma: no cover - defensive
    owa = None
    _OWA_IMPORT_ERROR = _owa_exc
else:
    _OWA_IMPORT_ERROR = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

STANDARD_VALUE = 85.0  # fixed standard stiffness S
RANDOM_SEED = 20240613  # fixed seed for ALL stochastic steps (bootstrap, jitter)
N_BOOTSTRAP = 5000  # bootstrap resamples
BOOTSTRAP_CI = 95  # confidence level (%)

SYSTEM_LEVELS = ["L", "N"]
FINGER_LEVELS = ["I", "M", "P", "R"]
FINGER_FULLNAMES = {"I": "Index", "M": "Middle", "P": "Pinky", "R": "Ring"}

# Source-column -> analysis-column mapping (documented for transparency).
COLUMN_MAP = {
    "subject_id": "Subject",
    "workspace_setup": "System",
    "finger_condition": "Finger",
    "pse": "PSE",
    "pse_delta_from_standard": "Bias",
    "jnd": "JND",
}

# Live psychophysics results directory. This module READS already-computed
# summaries from here; it never recomputes psychometric fits or trial scoring.
PSYCHO_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "psychophysics", "results", "L_N_E", "_working",
)

# Primary source: the live per-subject x finger PSE/JND summary produced by the
# psychophysics pipeline (NOT a frozen copy). PSE, JND and Bias are read from
# here as-is.
DEFAULT_DATA_PATH = os.path.join(
    PSYCHO_RESULTS_DIR, "pse_jnd_by_subject_finger.csv"
)
SOURCE_DATA_PATH = DEFAULT_DATA_PATH  # provenance == the live file itself

# Legacy frozen copy under data/ - kept ONLY as an offline fallback if the live
# results folder is unavailable (e.g. Dropbox not synced). load_data() warns
# when it has to fall back.
LEGACY_FROZEN_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "pse_jnd_by_subject_finger.csv",
)

# A JND larger than this (in stiffness units, same scale as the comparisons
# which span 25-145) is implausible for a discrimination threshold and almost
# always indicates a degenerate / failed psychometric fit. Used ONLY to flag
# rows for the report and the sensitivity analysis, never to silently drop.
JND_EXTREME_THRESHOLD = 100.0

# Lapse rate parameters were fitted within [0, 0.20]. A lapse value within this
# tolerance of either bound indicates the optimiser pinned it at a limit
# (a sign of a poorly constrained / shallow fit).
LAPSE_BOUND_LOW = 0.0
LAPSE_BOUND_HIGH = 0.20
LAPSE_BOUND_TOL = 1e-3


# --------------------------------------------------------------------------- #
# Validation result container
# --------------------------------------------------------------------------- #

@dataclass
class ValidationReport:
    """Collects validation findings without dropping any data."""

    n_rows: int = 0
    n_subjects: int = 0
    messages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    flagged_fit_rows: int = 0
    extreme_jnd_rows: int = 0
    excluded_rows: int = 0
    balanced: bool = True

    def info(self, msg: str) -> None:
        self.messages.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        lines = ["=" * 70, "DATA VALIDATION REPORT", "=" * 70]
        lines.append(f"Rows: {self.n_rows}   Subjects: {self.n_subjects}")
        lines.append("")
        lines.append("Checks:")
        for m in self.messages:
            lines.append(f"  [OK]   {m}")
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS (data retained, NOT dropped):")
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")
        else:
            lines.append("")
            lines.append("No warnings.")
        lines.append("=" * 70)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Load + validate
# --------------------------------------------------------------------------- #

def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the frozen per-subject x finger summary and standardise columns.

    Returns a DataFrame with canonical columns: Subject, System, Finger, PSE,
    Bias, JND, plus the quality/flag columns and a derived ``Group8`` column
    (System + "_" + Finger). No rows are dropped here.
    """
    if path is None:
        path = DEFAULT_DATA_PATH
        if not os.path.exists(path) and os.path.exists(LEGACY_FROZEN_DATA_PATH):
            warnings.warn(
                "Live psychophysics summary not found at "
                f"{path!r}; falling back to the legacy frozen copy at "
                f"{LEGACY_FROZEN_DATA_PATH!r}. Re-sync the psychophysics "
                "results to read the live file.",
                stacklevel=2,
            )
            path = LEGACY_FROZEN_DATA_PATH
    raw = pd.read_csv(path)

    missing = [c for c in COLUMN_MAP if c not in raw.columns]
    if missing:
        raise KeyError(
            f"Required source columns missing from {path}: {missing}"
        )

    df = raw.rename(columns=COLUMN_MAP).copy()

    # Carry through quality / flag / fit-detail columns if present.
    for col in [
        "fit_quality",
        "fit_warning",
        "fit_method",
        "psignifit_status",
        "excluded_from_group_analysis",
        "group_exclusion_reason",
        "pse_in_valid_band",
        "standard_value",
        "x25",
        "x75",
        "lapse_low",
        "lapse_high",
        "lapse_rate",
        "mu",
        "scale",
    ]:
        if col not in df.columns and col in raw.columns:
            df[col] = raw[col]

    # Categorical ordering for tidy tables / plots.
    df["System"] = pd.Categorical(df["System"], categories=SYSTEM_LEVELS)
    df["Finger"] = pd.Categorical(df["Finger"], categories=FINGER_LEVELS)

    # Combined 8-group label for the exploratory one-way analysis.
    df["Group8"] = (
        df["System"].astype(str) + "_" + df["Finger"].astype(str)
    )

    df.attrs["source_path"] = path
    return df


def validate_data(df: pd.DataFrame, tol: float = 1e-6) -> ValidationReport:
    """Run all data-quality checks and return a ValidationReport.

    This prints/collects clear warnings but NEVER drops rows. The caller
    decides how to use the flags (main vs sensitivity analysis).
    """
    rep = ValidationReport()
    rep.n_rows = len(df)
    rep.n_subjects = df["Subject"].nunique()

    # Required columns.
    required = ["Subject", "System", "Finger", "PSE", "Bias", "JND"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.warn(f"Missing required columns: {missing}")
    else:
        rep.info("All required columns present "
                 "(Subject, System, Finger, PSE, Bias, JND).")

    # System has exactly 2 levels.
    sys_levels = sorted(df["System"].dropna().unique().tolist())
    if sys_levels == SYSTEM_LEVELS:
        rep.info(f"System has exactly 2 levels: {sys_levels}.")
    else:
        rep.warn(f"System levels unexpected: {sys_levels} "
                 f"(expected {SYSTEM_LEVELS}).")

    # Finger has exactly 4 levels.
    fing_levels = sorted(df["Finger"].dropna().unique().tolist())
    if fing_levels == FINGER_LEVELS:
        rep.info(f"Finger has exactly 4 levels: {fing_levels}.")
    else:
        rep.warn(f"Finger levels unexpected: {fing_levels} "
                 f"(expected {FINGER_LEVELS}).")

    # Each subject in exactly one System.
    multi = df.groupby("Subject", observed=True)["System"].nunique()
    bad = multi[multi > 1]
    if len(bad) == 0:
        rep.info("Each subject belongs to exactly one System (between-subject "
                 "factor confirmed).")
    else:
        rep.warn(f"{len(bad)} subject(s) appear in >1 System: "
                 f"{bad.index.tolist()}.")

    # No duplicate Subject x Finger rows.
    dups = df.duplicated(["Subject", "Finger"]).sum()
    if dups == 0:
        rep.info("No duplicate Subject x Finger rows.")
    else:
        rep.warn(f"{dups} duplicate Subject x Finger row(s) present.")

    # Bias consistency: Bias == PSE - 85.
    bias_err = np.abs((df["PSE"] - STANDARD_VALUE) - df["Bias"])
    max_err = float(np.nanmax(bias_err)) if len(bias_err) else 0.0
    if max_err <= tol:
        rep.info(f"Bias == PSE - {STANDARD_VALUE:.0f} within tol "
                 f"(max abs diff = {max_err:.2e}).")
    else:
        rep.warn(f"Bias != PSE - {STANDARD_VALUE:.0f}: max abs diff = "
                 f"{max_err:.3g} exceeds tol {tol:g}.")

    # standard_value column, if present, should be 85.
    if "standard_value" in df.columns:
        sv = df["standard_value"].dropna().unique()
        if len(sv) == 1 and abs(float(sv[0]) - STANDARD_VALUE) <= tol:
            rep.info(f"standard_value column == {STANDARD_VALUE:.0f} for all "
                     "rows.")
        else:
            rep.warn(f"standard_value column has unexpected values: {sv}.")

    # Missing values in Bias / JND.
    n_bias_na = int(df["Bias"].isna().sum())
    n_jnd_na = int(df["JND"].isna().sum())
    if n_bias_na == 0 and n_jnd_na == 0:
        rep.info("No missing values in Bias or JND.")
    else:
        rep.warn(f"Missing values: Bias={n_bias_na}, JND={n_jnd_na}.")

    # Balance: each subject has all 4 fingers.
    counts = df.groupby("Subject", observed=True)["Finger"].nunique()
    unbalanced = counts[counts != len(FINGER_LEVELS)]
    if len(unbalanced) == 0:
        rep.info(f"Balanced design: every subject has all "
                 f"{len(FINGER_LEVELS)} fingers.")
        rep.balanced = True
    else:
        rep.warn(f"{len(unbalanced)} subject(s) do NOT have all 4 fingers: "
                 f"{unbalanced.to_dict()}.")
        rep.balanced = False

    # System cell sizes (subjects per system).
    sys_n = (df.drop_duplicates("Subject")
               .groupby("System", observed=True)["Subject"].count())
    rep.info(f"Subjects per System: {sys_n.to_dict()}.")

    # Flagged fits.
    n_flagged = 0
    if "fit_quality" in df.columns:
        n_flagged = int((df["fit_quality"] != "ok").sum())
    excl_mask = _excluded_mask(df)
    n_excl = int(excl_mask.sum())
    rep.flagged_fit_rows = n_flagged
    rep.excluded_rows = n_excl
    if n_flagged:
        rep.warn(f"{n_flagged} row(s) have fit_quality != 'ok' "
                 "(warning fits).")
    if n_excl:
        rep.warn(f"{n_excl} row(s) flagged excluded_from_group_analysis "
                 "(upstream group-analysis exclusions).")
    if n_flagged == 0 and n_excl == 0:
        rep.info("No flagged or excluded fits.")

    # Fitter provenance.
    if "fit_method" in df.columns:
        fm = df["fit_method"].value_counts().to_dict()
        rep.info(f"Fits by fitter (fit_method): {fm}.")
    if "psignifit_status" in df.columns:
        ps = df["psignifit_status"].value_counts().to_dict()
        rep.info(f"psignifit_status: {ps}.")

    # Estimability of PSE / x25 / x75 / JND (NaN => non-estimable, flagged).
    for col, label in [("PSE", "PSE"), ("x25", "x25"),
                       ("x75", "x75"), ("JND", "JND")]:
        if col in df.columns:
            n_na = int(df[col].isna().sum())
            if n_na:
                rep.warn(f"{n_na} row(s) have non-estimable {label} (NaN); "
                         "these are FLAGGED by the upstream fitter, not "
                         "silently invalid.")

    # Lapse parameters pinned at a bound (sign of poorly constrained fit).
    flags_tmp = compute_qc_flags(df)
    n_ll = int(flags_tmp["lapse_low_at_bound"].sum())
    n_lh = int(flags_tmp["lapse_high_at_bound"].sum())
    if n_ll or n_lh:
        rep.warn(f"Lapse pinned at a bound: lapse_low {n_ll} row(s), "
                 f"lapse_high {n_lh} row(s) (poorly constrained fits).")

    # Extreme / non-positive JND.
    n_nonpos = int((df["JND"] <= 0).sum())
    if n_nonpos:
        rep.warn(f"{n_nonpos} row(s) have non-positive JND.")
    n_extreme = int((df["JND"] > JND_EXTREME_THRESHOLD).sum())
    rep.extreme_jnd_rows = n_extreme
    if n_extreme:
        rng = df.loc[df["JND"] > JND_EXTREME_THRESHOLD, "JND"]
        rep.warn(
            f"{n_extreme} row(s) have JND > {JND_EXTREME_THRESHOLD:g} "
            f"(range {rng.min():.1f}-{rng.max():.1f}); these indicate "
            "degenerate fits and will inflate variance in the MAIN analysis. "
            "See the sensitivity analysis (flagged fits excluded)."
        )
    if n_nonpos == 0 and n_extreme == 0:
        rep.info("No non-positive or extreme JND values.")

    return rep


def _excluded_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows the upstream pipeline excluded from group analysis."""
    if "excluded_from_group_analysis" in df.columns:
        return df["excluded_from_group_analysis"].astype(bool).fillna(False)
    return pd.Series(False, index=df.index)


def compute_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive explicit per-Subject x Finger QC flags from EXISTING columns.

    No psychometric fits are recomputed. Returns a copy of ``df`` augmented
    with QC flag columns:

      pse_estimable, x25_estimable, x75_estimable, jnd_estimable  (value not NaN)
      lapse_low_at_bound, lapse_high_at_bound  (within tol of 0 or 0.20 bound)
      total_lapse_rate = lapse_low + lapse_high
      jnd_nonpositive  (jnd <= 0)
      jnd_extreme      (jnd > JND_EXTREME_THRESHOLD)
      fitter_used      (from fit_method)
      fit_quality, fit_warning surfaced as-is
      qc_pass          (estimable PSE & JND, positive non-extreme JND,
                        fit_quality == 'ok')
    """
    out = df.copy()

    def _estimable(col):
        return out[col].notna() if col in out.columns else pd.Series(
            True, index=out.index)

    out["pse_estimable"] = _estimable("PSE")
    out["x25_estimable"] = _estimable("x25")
    out["x75_estimable"] = _estimable("x75")
    out["jnd_estimable"] = _estimable("JND")

    if "lapse_low" in out.columns:
        ll = out["lapse_low"].astype(float)
        out["lapse_low_at_bound"] = (
            (np.abs(ll - LAPSE_BOUND_LOW) <= LAPSE_BOUND_TOL)
            | (np.abs(ll - LAPSE_BOUND_HIGH) <= LAPSE_BOUND_TOL)
        )
    else:
        out["lapse_low_at_bound"] = False
    if "lapse_high" in out.columns:
        lh = out["lapse_high"].astype(float)
        out["lapse_high_at_bound"] = (
            (np.abs(lh - LAPSE_BOUND_LOW) <= LAPSE_BOUND_TOL)
            | (np.abs(lh - LAPSE_BOUND_HIGH) <= LAPSE_BOUND_TOL)
        )
    else:
        out["lapse_high_at_bound"] = False

    if "lapse_low" in out.columns and "lapse_high" in out.columns:
        out["total_lapse_rate"] = (out["lapse_low"].astype(float)
                                   + out["lapse_high"].astype(float))
    else:
        out["total_lapse_rate"] = np.nan

    out["jnd_nonpositive"] = out["JND"] <= 0
    out["jnd_extreme"] = out["JND"] > JND_EXTREME_THRESHOLD

    out["fitter_used"] = (out["fit_method"] if "fit_method" in out.columns
                          else "unknown")
    if "fit_quality" not in out.columns:
        out["fit_quality"] = "unknown"
    if "fit_warning" not in out.columns:
        out["fit_warning"] = np.nan

    out["qc_pass"] = (
        out["pse_estimable"]
        & out["jnd_estimable"]
        & ~out["jnd_nonpositive"]
        & ~out["jnd_extreme"]
        & (out["fit_quality"] == "ok")
    )
    return out


def qc_flag_summary(df_flags: pd.DataFrame) -> pd.DataFrame:
    """Counts of each QC flag (from a frame produced by compute_qc_flags)."""
    flag_cols = [
        "pse_estimable", "x25_estimable", "x75_estimable", "jnd_estimable",
        "lapse_low_at_bound", "lapse_high_at_bound",
        "jnd_nonpositive", "jnd_extreme", "qc_pass",
    ]
    rows = []
    n = len(df_flags)
    for c in flag_cols:
        if c in df_flags.columns:
            true_n = int(df_flags[c].sum())
            rows.append({"flag": c, "n_true": true_n,
                         "n_false": n - true_n, "n_total": n})
    # Fitter and fit_quality breakdowns.
    if "fitter_used" in df_flags.columns:
        for k, v in df_flags["fitter_used"].value_counts().items():
            rows.append({"flag": f"fitter_used={k}", "n_true": int(v),
                         "n_false": n - int(v), "n_total": n})
    if "fit_quality" in df_flags.columns:
        for k, v in df_flags["fit_quality"].value_counts().items():
            rows.append({"flag": f"fit_quality={k}", "n_true": int(v),
                         "n_false": n - int(v), "n_total": n})
    if "fit_warning" in df_flags.columns:
        for k, v in df_flags["fit_warning"].dropna().value_counts().items():
            rows.append({"flag": f"fit_warning={k}", "n_true": int(v),
                         "n_false": n - int(v), "n_total": n})
    return pd.DataFrame(rows)


def split_included_excluded(df: pd.DataFrame):
    """Return (included_df, excluded_df) using the upstream exclusion flag.

    A consolidated ``exclusion_reason`` column is added to the excluded frame.
    The MAIN analysis still uses ALL valid rows; this split powers the
    transparency CSVs and the sensitivity analysis.
    """
    excl = _excluded_mask(df)
    included = df[~excl].copy()
    excluded = df[excl].copy()
    if "group_exclusion_reason" in excluded.columns:
        excluded["exclusion_reason"] = (
            excluded["group_exclusion_reason"].fillna("excluded_upstream")
        )
    else:
        excluded["exclusion_reason"] = "excluded_from_group_analysis"
    return included, excluded


def analysis_frame(df: pd.DataFrame, dv: str, drop_flagged: bool = False):
    """Return a tidy frame for ANOVA with columns Subject, System, Finger, <dv>.

    Parameters
    ----------
    dv : 'Bias' or 'JND'.
    drop_flagged : if True, remove rows flagged excluded_from_group_analysis
        (the SENSITIVITY analysis). If False (default), keep ALL valid rows
        (the MAIN analysis).
    """
    work = df.copy()
    if drop_flagged:
        work = work[~_excluded_mask(work)].copy()
    cols = ["Subject", "System", "Finger", dv]
    out = work[cols].dropna(subset=[dv]).copy()
    out["System"] = out["System"].astype(str)
    out["Finger"] = out["Finger"].astype(str)
    return out


# --------------------------------------------------------------------------- #
# Descriptive helpers
# --------------------------------------------------------------------------- #

def descriptive_table(df: pd.DataFrame, dv: str,
                      by=("System", "Finger")) -> pd.DataFrame:
    """Mean/SD/SE/N table for a DV grouped by the given factors."""
    by = list(by)
    g = df.groupby(by, observed=True)[dv]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": f"{dv}_mean",
                              "std": f"{dv}_sd",
                              "count": "n"})
    out[f"{dv}_se"] = out[f"{dv}_sd"] / np.sqrt(out["n"])
    return out


# --------------------------------------------------------------------------- #
# (2) Exploratory one-way ANOVA across 8 combined groups
# --------------------------------------------------------------------------- #

def exploratory_one_way(df: pd.DataFrame, dv: str):
    """Exploratory one-way ANOVA on the 8 combined System x Finger groups.

    NOTE: This is EXPLORATORY and does NOT model repeated measures (it treats
    all 8 cells as independent groups). Returns (anova_table, posthoc_table).
    """
    work = analysis_frame(df, dv)
    work["Group8"] = work["System"] + "_" + work["Finger"]

    aov = pg.anova(data=work, dv=dv, between="Group8", detailed=True)
    aov.insert(0, "analysis", "Exploratory one-way ANOVA "
                              "on combined System x Finger groups")
    aov.insert(1, "DV", dv)

    # Post-hoc pairwise with Holm correction across all 28 pairs.
    posthoc = pg.pairwise_tests(
        data=work, dv=dv, between="Group8",
        padjust="holm", effsize="hedges",
    )
    posthoc.insert(0, "DV", dv)

    return aov, posthoc


# --------------------------------------------------------------------------- #
# (3) MAIN: mixed-design ANOVA
# --------------------------------------------------------------------------- #

def mixed_anova(df: pd.DataFrame, dv: str, drop_flagged: bool = False):
    """Mixed-design ANOVA: System (between) x Finger (within), Subject = unit.

    Returns dict with keys: 'aov' (table), 'sphericity' (Mauchly result or
    note), 'method', 'n_subjects', 'frame'.

    Uses pingouin.mixed_anova, which reports df, F, p, partial eta-squared
    (np2), and generalized eta-squared (ges). Greenhouse-Geisser correction
    for the within factor is applied automatically by pingouin when sphericity
    is testable; with only 2 within levels at a time it is not needed, but with
    4 finger levels sphericity is assessed via Mauchly's test (reported
    separately).
    """
    work = analysis_frame(df, dv, drop_flagged=drop_flagged)

    # Only keep subjects that are complete (all 4 fingers) for the RM model;
    # report any dropped.
    counts = work.groupby("Subject")["Finger"].nunique()
    complete = counts[counts == len(FINGER_LEVELS)].index
    dropped = sorted(set(counts.index) - set(complete))
    work = work[work["Subject"].isin(complete)].copy()

    aov = pg.mixed_anova(
        data=work, dv=dv,
        within="Finger", between="System", subject="Subject",
        correction=True, effsize="np2",
    )
    aov.insert(0, "DV", dv)
    aov.insert(1, "analysis",
               "MAIN mixed-design ANOVA: System (between) x Finger (within)")

    # Mauchly's test of sphericity for the within (Finger) factor.
    try:
        sph = pg.sphericity(data=work, dv=dv, within="Finger",
                            subject="Subject")
        sph_note = (
            f"Mauchly W={sph.W:.4f}, chi2={sph.chi2:.3f}, "
            f"dof={sph.dof}, p={sph.pval:.4f}, "
            f"sphericity={'met' if sph.spher else 'VIOLATED'}"
        )
    except Exception as exc:  # pragma: no cover - defensive
        sph = None
        sph_note = f"Sphericity test unavailable: {exc}"

    return {
        "aov": aov,
        "sphericity": sph,
        "sphericity_note": sph_note,
        "method": "pingouin.mixed_anova (v%s)" % pg.__version__,
        "n_subjects": work["Subject"].nunique(),
        "dropped_subjects": dropped,
        "frame": work,
    }


# --------------------------------------------------------------------------- #
# (4) Planned contrasts: System L vs N within each finger
# --------------------------------------------------------------------------- #

def planned_contrasts(df: pd.DataFrame, dv: str, drop_flagged: bool = False):
    """System L vs N within EACH finger (between-system independent t-tests).

    Because subjects differ across systems, the L-vs-N comparison within a
    finger is a between-groups (independent) test. Holm correction is applied
    across the 4 finger comparisons. Returns a tidy table.
    """
    work = analysis_frame(df, dv, drop_flagged=drop_flagged)
    rows = []
    for finger in FINGER_LEVELS:
        sub = work[work["Finger"] == finger]
        l = sub.loc[sub["System"] == "L", dv].to_numpy()
        n = sub.loc[sub["System"] == "N", dv].to_numpy()
        if len(l) < 2 or len(n) < 2:
            continue
        # Welch's t-test (does not assume equal variance) via pingouin to get
        # CI and effect size in one call.
        tt = pg.ttest(l, n, paired=False, correction=True)
        diff = float(np.mean(l) - np.mean(n))  # L - N
        # pingouin 0.6.x uses underscore column names: p_val, CI95, cohen_d.
        ci_col = "CI95" if "CI95" in tt.columns else "CI95%"
        p_col = "p_val" if "p_val" in tt.columns else "p-val"
        d_col = "cohen_d" if "cohen_d" in tt.columns else "cohen-d"
        ci = tt[ci_col].iloc[0]
        rows.append({
            "DV": dv,
            "Finger": finger,
            "Finger_name": FINGER_FULLNAMES.get(finger, finger),
            "mean_L": float(np.mean(l)),
            "sd_L": float(np.std(l, ddof=1)),
            "n_L": int(len(l)),
            "mean_N": float(np.mean(n)),
            "sd_N": float(np.std(n, ddof=1)),
            "n_N": int(len(n)),
            "diff_L_minus_N": diff,
            "T": float(tt["T"].iloc[0]),
            "dof": float(tt["dof"].iloc[0]),
            "p_raw": float(tt[p_col].iloc[0]),
            "CI95_low": float(ci[0]),
            "CI95_high": float(ci[1]),
            "cohen_d": float(tt[d_col].iloc[0]),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
        out["sig_holm"] = out["p_holm"] < 0.05
    return out


# --------------------------------------------------------------------------- #
# (5) Subject-respecting bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_cis(df: pd.DataFrame, dv: str,
                  n_boot: int = N_BOOTSTRAP,
                  ci: float = BOOTSTRAP_CI,
                  seed: int = RANDOM_SEED,
                  drop_flagged: bool = False):
    """Subject-level bootstrap CIs respecting the repeated-measures structure.

    Subjects are resampled WITH replacement WITHIN each System, keeping each
    subject's 4 finger rows together. Returns (cell_ci, diff_ci):

      * cell_ci : mean of <dv> per System x Finger, with bootstrap CI.
      * diff_ci : L - N difference of cell means per Finger, with bootstrap CI.
    """
    work = analysis_frame(df, dv, drop_flagged=drop_flagged)
    rng = np.random.default_rng(seed)

    # Pre-index subjects per system and their per-finger values.
    subj_sys = (work.drop_duplicates("Subject")
                    .set_index("Subject")["System"].to_dict())
    subjects_by_sys = {s: [k for k, v in subj_sys.items() if v == s]
                       for s in SYSTEM_LEVELS}

    # Wide lookup: (subject, finger) -> value.
    pivot = work.pivot_table(index="Subject", columns="Finger",
                             values=dv, observed=True)

    alpha = (100 - ci) / 2.0
    cell_keys = [(s, f) for s in SYSTEM_LEVELS for f in FINGER_LEVELS]
    cell_samples = {k: np.empty(n_boot) for k in cell_keys}
    diff_samples = {f: np.empty(n_boot) for f in FINGER_LEVELS}

    for b in range(n_boot):
        # Resample subjects within each system.
        boot_means = {}  # (system, finger) -> mean
        for s in SYSTEM_LEVELS:
            pool = subjects_by_sys[s]
            picks = rng.choice(pool, size=len(pool), replace=True)
            sub_vals = pivot.loc[picks]
            for f in FINGER_LEVELS:
                boot_means[(s, f)] = np.nanmean(sub_vals[f].to_numpy())
        for k in cell_keys:
            cell_samples[k][b] = boot_means[k]
        for f in FINGER_LEVELS:
            diff_samples[f][b] = boot_means[("L", f)] - boot_means[("N", f)]

    # Observed cell means.
    obs = (work.groupby(["System", "Finger"], observed=True)[dv]
               .mean().to_dict())

    cell_rows = []
    for (s, f) in cell_keys:
        samp = cell_samples[(s, f)]
        cell_rows.append({
            "DV": dv, "System": s, "Finger": f,
            "Finger_name": FINGER_FULLNAMES.get(f, f),
            "mean": float(obs.get((s, f), np.nan)),
            "boot_mean": float(np.nanmean(samp)),
            "ci_low": float(np.nanpercentile(samp, alpha)),
            "ci_high": float(np.nanpercentile(samp, 100 - alpha)),
            "n_boot": n_boot,
        })
    cell_ci = pd.DataFrame(cell_rows)

    diff_rows = []
    for f in FINGER_LEVELS:
        samp = diff_samples[f]
        obs_diff = (obs.get(("L", f), np.nan) - obs.get(("N", f), np.nan))
        ci_low = float(np.nanpercentile(samp, alpha))
        ci_high = float(np.nanpercentile(samp, 100 - alpha))
        diff_rows.append({
            "DV": dv, "Finger": f,
            "Finger_name": FINGER_FULLNAMES.get(f, f),
            "diff_L_minus_N": float(obs_diff),
            "boot_diff": float(np.nanmean(samp)),
            "ci_low": ci_low, "ci_high": ci_high,
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            "n_boot": n_boot,
        })
    diff_ci = pd.DataFrame(diff_rows)

    return cell_ci, diff_ci


# --------------------------------------------------------------------------- #
# (6) Figures
# --------------------------------------------------------------------------- #

def _mean_ci(values: np.ndarray, ci: float = 95.0):
    """Mean and +/- half-width 95% CI (t-based) of a 1-D array."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return (float(np.mean(v)) if n else np.nan), 0.0
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n))
    tcrit = stats.t.ppf(1 - (1 - ci / 100) / 2, df=n - 1)
    return m, tcrit * se


# System colours: the marker/circle is filled by SYSTEM (pink = L, light purple = N).
_SYS_COLORS = {"L": "#E377C2", "N": "#B19CD9"}
_SYS_OFFSET = {"L": -0.14, "N": 0.14}
# Per-finger colours: every finger keeps ONE dedicated colour across all figures.
# Sourced from the colocated oneway_anova module so the one-way and two-way plots
# match; falls back to the same literal values if that module is unavailable.
_FINGER_COLORS = (getattr(owa, "_FINGER_COLORS", None) or
                  {"I": "#4C72B0", "M": "#DD8452", "R": "#55A868", "P": "#C44E52"})


def plot_dv_by_finger(df: pd.DataFrame, dv: str, hline: Optional[float],
                      ylabel: str, title: str, out_path: str,
                      seed: int = RANDOM_SEED):
    """Subject points + mean +/- 95% CI per Finger, hue = System."""
    work = analysis_frame(df, dv)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_base = {f: i for i, f in enumerate(FINGER_LEVELS)}

    for s in SYSTEM_LEVELS:
        for f in FINGER_LEVELS:
            vals = work[(work["System"] == s) &
                        (work["Finger"] == f)][dv].to_numpy()
            x0 = x_base[f] + _SYS_OFFSET[s]
            jitter = (rng.random(len(vals)) - 0.5) * 0.10
            ax.scatter(np.full(len(vals), x0) + jitter, vals,
                       color=_SYS_COLORS[s], alpha=0.35, s=22,
                       edgecolors="none", zorder=2)
            m, h = _mean_ci(vals)
            # Circle filled by SYSTEM colour; error line in this FINGER's colour.
            ax.errorbar(x0, m, yerr=h, fmt="o",
                        mfc=_SYS_COLORS[s], mec="black", markeredgewidth=0.6,
                        ecolor=_FINGER_COLORS.get(f, "#444444"),
                        markersize=10, capsize=5, elinewidth=2.2, zorder=3,
                        label=f"System {s}" if f == "I" else None)

    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1,
                   label=f"reference = {hline:g}")
    # Secondary legend mapping each finger to its dedicated error-line colour.
    finger_handles = [Line2D([0], [0], color=_FINGER_COLORS.get(f, "#444444"),
                             lw=2.2, label=f"{f} ({FINGER_FULLNAMES[f]})")
                      for f in FINGER_LEVELS]
    ax.set_xticks(list(x_base.values()))
    ax.set_xticklabels([f"{f}\n({FINGER_FULLNAMES[f]})" for f in FINGER_LEVELS])
    ax.set_xlabel("Finger")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sys_legend = ax.legend(frameon=False, loc="upper left", title="circle = system")
    ax.add_artist(sys_legend)
    ax.legend(handles=finger_handles, frameon=False, loc="upper right",
              title="line = finger", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_eight_groups(df: pd.DataFrame, dv: str, ylabel: str, title: str,
                      out_path: str, hline: Optional[float] = None,
                      seed: int = RANDOM_SEED):
    """Exploratory 8-group figure: points + mean +/- 95% CI per Group8."""
    work = analysis_frame(df, dv)
    work["Group8"] = work["System"] + "_" + work["Finger"]
    order = [f"{s}_{f}" for s in SYSTEM_LEVELS for f in FINGER_LEVELS]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, grp in enumerate(order):
        vals = work[work["Group8"] == grp][dv].to_numpy()
        s, f = grp.split("_")[0], grp.split("_")[1]
        jitter = (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=_SYS_COLORS[s], alpha=0.35, s=22,
                   edgecolors="none", zorder=2)
        m, h = _mean_ci(vals)
        # Circle filled by SYSTEM colour; error line in this FINGER's colour.
        ax.errorbar(i, m, yerr=h, fmt="o",
                    mfc=_SYS_COLORS[s], mec="black", markeredgewidth=0.6,
                    ecolor=_FINGER_COLORS.get(f, "#444444"),
                    markersize=10, capsize=5, elinewidth=2.2, zorder=3)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_xlabel("Combined System_Finger group (EXPLORATORY)")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(EXPLORATORY: ignores repeated measures)")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_contrasts(contrast_df: pd.DataFrame, dv: str, ylabel: str,
                   title: str, out_path: str):
    """Planned-contrast figure: L - N per finger with 95% CI + Holm sig."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(contrast_df))
    diffs = contrast_df["diff_L_minus_N"].to_numpy()
    lo = contrast_df["CI95_low"].to_numpy()
    hi = contrast_df["CI95_high"].to_numpy()
    # One point per finger, drawn in that finger's dedicated colour.
    for i, (_, row) in enumerate(contrast_df.iterrows()):
        col = _FINGER_COLORS.get(str(row["Finger"]), "#2c3e50")
        ax.errorbar(i, diffs[i], yerr=[[diffs[i] - lo[i]], [hi[i] - diffs[i]]],
                    fmt="o", color=col, markersize=10, capsize=6, lw=2,
                    elinewidth=2.2, markeredgecolor="black")
    ax.axhline(0, color="gray", ls="--", lw=1)
    for i, (_, row) in enumerate(contrast_df.iterrows()):
        if row.get("sig_holm", False):
            top = max(hi[i], 0) + 0.04 * (np.nanmax(hi) - np.nanmin(lo) + 1)
            ax.annotate("*", (i, top), ha="center", fontsize=18,
                        color="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r.Finger}\n({r.Finger_name})"
                        for _, r in contrast_df.iterrows()])
    ax.set_xlabel("Finger")
    ax.set_ylabel(ylabel + "  (L - N)")
    ax.set_title(title + "\n(* = Holm-corrected p < 0.05)")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

METHODS_TEXT = textwrap.dedent("""\
    METHODS (psychophysics statistics)
    ----------------------------------
    Task and response coding. Participants performed a two-alternative
    forced-choice (2AFC) stiffness-discrimination task. On each trial a
    standard stimulus (S = 85) was compared with one of eight comparison
    stimuli (25, 40, 55, 70, 100, 115, 130, 145); the predictor was the
    signed difference delta = C - 85. The binary response was coded y = 1 if
    and only if the participant judged the COMPARISON to be stiffer than the
    standard (this is not correct/incorrect, not side, not order, and not the
    physical ground truth). This coding was performed upstream.

    Psychometric model. For each subject and finger a lapse-aware yes/no-style
    psychometric function with four fitted parameters (mu, scale, lapse_low,
    lapse_high) was fitted upstream (these fits are frozen and were NOT
    recomputed here). This is not a textbook 2AFC percent-correct model with a
    fixed guess rate; it models the probability that the participant judged the
    comparison stiffer than the standard, P(y=1) = lapse_low +
    (1 - lapse_low - lapse_high) * F(delta; mu, scale), where delta = C - 85 and
    F is a monotonic sigmoid. Fits used psignifit when installed and otherwise a
    custom lapse-aware fallback fitter; the fitter used for each fit is recorded
    in the data.

    Derived measures. From each fit we used two summary measures (the fit is in
    comparison-stiffness units, so these are equivalent to delta-space): the
    Bias, defined as the point of subjective equality minus the standard
    (Bias = PSE - 85, the delta at P(y=1) = 0.5; a value of 0 indicates no
    perceptual shift), and the JND, defined as (x75 - x25)/2 where x25 and x75
    are the comparison values at which the curve reaches 0.25 and 0.75 (a
    measure of sensitivity, where smaller values indicate finer
    discrimination). Because of the lapse parameters, x25 and x75 are estimable
    only if 0.25 and 0.75 lie within the attainable range
    [lapse_low, 1 - lapse_high]; when they do not, the upstream fitter returns a
    NaN JND with a fit warning, so a non-estimable JND is flagged rather than
    silently treated as valid.

    Design. System (L vs N) was a between-subjects factor: each participant
    used exactly one system. Finger (I, M, P, R) was a within-subjects
    (repeated-measures) factor, with all four fingers measured in every
    subject. Subject served as the repeated-measures unit.

    Main analysis. Bias and JND were each analysed with a mixed-design
    analysis of variance with System as the between-subjects factor and Finger
    as the within-subjects factor, testing the main effect of System, the main
    effect of Finger, and the System x Finger interaction. Sphericity of the
    Finger factor was assessed with Mauchly's test and, where appropriate, the
    Greenhouse-Geisser correction was applied. Partial eta-squared (np2) and
    generalized eta-squared (ges) are reported as effect sizes.

    Why a mixed-design ANOVA and not an ordinary two-way ANOVA. An ordinary
    independent two-way ANOVA assumes that all observations are independent.
    Because the four finger measurements were obtained from the same subject,
    they are correlated (repeated measures); treating them as independent would
    violate the independence assumption and mis-estimate the error term. We
    therefore used the mixed-design model, which correctly partitions
    between-subject and within-subject variance.

    Exploratory analysis. As a descriptive supplement we also ran a one-way
    ANOVA across the eight combined System x Finger groups. This exploratory
    analysis treats the eight cells as independent groups and does NOT model
    the repeated-measures structure; it is reported for completeness only and
    is not the basis for inference.

    Planned contrasts. The L-vs-N system difference was tested within each
    finger using independent (between-subjects) t-tests, with Holm correction
    across the four finger comparisons.

    Robustness. Confidence intervals for cell means and for the L - N
    difference per finger were additionally estimated with a subject-level
    bootstrap (subjects resampled with replacement within each system, keeping
    each subject's four finger rows together; fixed random seed).
    """)


def get_p(row_or_df, prefer=("p_unc", "p-unc", "p_GG_corr")):
    """Return a p-value from a pingouin row/Series across version naming."""
    for c in prefer:
        if c in row_or_df:
            try:
                return float(row_or_df[c])
            except (TypeError, ValueError):
                continue
    return np.nan


def _col(df, *names, default=""):
    """First present column name from a DataFrame (version-tolerant)."""
    for n in names:
        if n in df.columns:
            return n
    return default


def _fmt_p(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return str(p)
    if np.isnan(p):
        return "nan"
    return "< .001" if p < 0.001 else f"{p:.3f}"


def build_report(validation: ValidationReport,
                 results: dict) -> str:
    """Assemble the full methods + results text report from computed results.

    ``results`` is the dict produced by run_full_pipeline (see notebook).
    """
    lines = []
    lines.append("#" * 72)
    lines.append("# 2AFC PSYCHOPHYSICS STATISTICS REPORT")
    lines.append("# (mixed-design ANOVA pipeline; additive, fits not recomputed)")
    lines.append("#" * 72)
    lines.append("")
    lines.append(f"Source data (frozen copy): {results.get('data_path','')}")
    lines.append(f"Documented provenance:     {SOURCE_DATA_PATH}")
    lines.append(f"Random seed (all stochastic steps): {RANDOM_SEED}")
    lines.append("")
    lines.append(METHODS_TEXT)
    lines.append("")
    lines.append(validation.render())
    lines.append("")

    # QC flag summary table.
    if "qc_summary" in results:
        lines.append("=" * 70)
        lines.append("PER-SUBJECT x FINGER QC FLAG SUMMARY (derived from "
                     "existing fit columns)")
        lines.append("=" * 70)
        for _, r in results["qc_summary"].iterrows():
            lines.append(f"  {r['flag']:<45} n_true={int(r['n_true'])} / "
                         f"{int(r['n_total'])}")
        lines.append("")

    for dv in ["Bias", "JND"]:
        lines.append("=" * 72)
        lines.append(f"RESULTS  --  {dv}")
        lines.append("=" * 72)

        # Mixed ANOVA (main).
        ma = results["mixed"][dv]
        aov = ma["aov"]
        lines.append("MAIN mixed-design ANOVA "
                     f"(method: {ma['method']}; "
                     f"N subjects = {ma['n_subjects']}):")
        for _, r in aov.iterrows():
            src = r["Source"]
            f = r.get("F", np.nan)
            p = get_p(r)
            np2 = r.get("np2", np.nan)
            ddof1 = r.get("DF1", r.get("ddof1", ""))
            ddof2 = r.get("DF2", r.get("ddof2", ""))
            p_gg = r.get("p_GG_corr", np.nan)
            gg_txt = ""
            if isinstance(p_gg, (int, float)) and not (
                    isinstance(p_gg, float) and np.isnan(p_gg)):
                gg_txt = f", p_GG = {_fmt_p(p_gg)}"
            lines.append(
                f"   {src:<18} F({ddof1},{ddof2}) = "
                f"{f:.3f}, p = {_fmt_p(p)}{gg_txt}, np2 = {np2:.3f}"
            )
        lines.append(f"   Sphericity (Finger): {ma['sphericity_note']}")
        if ma["dropped_subjects"]:
            lines.append(f"   (Subjects dropped for incomplete fingers: "
                         f"{ma['dropped_subjects']})")
        lines.append("")

        # Sensitivity mixed ANOVA.
        ms = results["mixed_sensitivity"][dv]
        lines.append("SENSITIVITY mixed-design ANOVA "
                     "(flagged/excluded fits removed; "
                     f"N subjects = {ms['n_subjects']}):")
        for _, r in ms["aov"].iterrows():
            src = r["Source"]
            f = r.get("F", np.nan)
            p = get_p(r)
            np2 = r.get("np2", np.nan)
            lines.append(f"   {src:<18} F = {f:.3f}, p = {_fmt_p(p)}, "
                         f"np2 = {np2:.3f}")
        lines.append("")

        # Exploratory one-way.
        ex = results["exploratory"][dv]["aov"]
        r0 = ex.iloc[0]
        lines.append("EXPLORATORY one-way ANOVA on the 8 combined "
                     "System x Finger groups (ignores repeated measures):")
        lines.append(f"   F = {r0.get('F', np.nan):.3f}, "
                     f"p = {_fmt_p(get_p(r0))}, "
                     f"np2 = {r0.get('np2', np.nan):.3f}")
        lines.append("")

        # Planned contrasts.
        ct = results["contrasts"][dv]
        lines.append("PLANNED CONTRASTS  (System L - N within each finger; "
                     "Holm-corrected):")
        for _, r in ct.iterrows():
            sig = " *" if r["sig_holm"] else ""
            lines.append(
                f"   {r['Finger']} ({r['Finger_name']}): "
                f"L={r['mean_L']:.2f}+/-{r['sd_L']:.2f}, "
                f"N={r['mean_N']:.2f}+/-{r['sd_N']:.2f}, "
                f"diff(L-N)={r['diff_L_minus_N']:.2f} "
                f"[{r['CI95_low']:.2f}, {r['CI95_high']:.2f}], "
                f"t({r['dof']:.1f})={r['T']:.2f}, "
                f"p_raw={_fmt_p(r['p_raw'])}, "
                f"p_holm={_fmt_p(r['p_holm'])}, d={r['cohen_d']:.2f}{sig}"
            )
        sig_fingers = ct.loc[ct["sig_holm"], "Finger"].tolist()
        lines.append("")

        # Interpretation (avoids overstating null results).
        sys_row = aov.loc[aov["Source"] == "System"].iloc[0]
        sys_p = get_p(sys_row)
        if sys_p >= 0.05:
            lines.append("Interpretation of the System effect: "
                         "No significant evidence for a difference between "
                         "systems was observed (this does NOT mean the systems "
                         "are identical).")
        else:
            lines.append("Interpretation of the System effect: "
                         "A statistically significant difference between "
                         "systems was observed.")
        if sig_fingers:
            lines.append(f"Finger contrasts significant after Holm: "
                         f"{sig_fingers}.")
        else:
            lines.append("No finger-level L-vs-N contrast survived Holm "
                         "correction.")
        lines.append("")

    # One-way per-system (finger) analysis (additive; pulled from
    # run_oneway_flat's return dict if present in results).
    oneway = results.get("oneway")
    if oneway and oneway.get("systems"):
        lines.append("=" * 72)
        lines.append("ONE-WAY PER-SYSTEM (FINGER) ANALYSIS "
                     "(numpy/scipy; per system, do the four fingers differ?)")
        lines.append("=" * 72)
        for system in oneway["systems"]:
            res = oneway["results"].get(system, {})
            lines.append(f"System {system}:")
            for dv in ("Bias", "JND"):
                a = res.get("anova", {}).get(dv, {})
                fr = res.get("friedman", {}).get(dv, {})
                f_val = a.get("F", np.nan)
                eta2 = a.get("eta_squared", np.nan)
                chi = fr.get("chi_square", np.nan)
                w = fr.get("kendalls_w", np.nan)
                lines.append(
                    f"   {dv:<5} one-way ANOVA: "
                    f"F({a.get('df_between','?')},{a.get('df_within','?')})="
                    f"{f_val:.3f}, p={_fmt_p(a.get('p_value', np.nan))}, "
                    f"eta2={eta2:.3f}; "
                    f"Friedman: chi2({fr.get('df','?')})={chi:.3f}, "
                    f"p={_fmt_p(fr.get('p_value', np.nan))}, "
                    f"W={w:.3f}")
            lines.append("")
    elif oneway and oneway.get("note"):
        lines.append("=" * 72)
        lines.append("ONE-WAY PER-SYSTEM (FINGER) ANALYSIS")
        lines.append("=" * 72)
        lines.append(f"   {oneway['note']}")
        lines.append("")

    lines.append("#" * 72)
    lines.append("# END OF REPORT")
    lines.append("#" * 72)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# (7) Summary tables rendered as images
# --------------------------------------------------------------------------- #
#
# These are PURELY ADDITIVE table-as-image renderers. They consume the same
# ``results`` dict assembled in the notebook (no recomputation) and write
# publication-style PNG tables alongside the existing figures. A significant
# row (p < .05) is shaded so the eye lands on it immediately.

_TABLE_HEADER_BG = "#2c3e50"
_TABLE_HEADER_FG = "white"
_TABLE_ROW_ALT = "#f4f6f8"
_TABLE_SIG_BG = "#ffe3b3"  # warm highlight for significant rows


def dataframe_to_image(df: pd.DataFrame, out_path: str,
                       title: Optional[str] = None,
                       sig_mask: Optional[np.ndarray] = None,
                       fontsize: int = 10,
                       col_scale: float = 1.0) -> str:
    """Render a (small) DataFrame as a formatted table image and save to PNG.

    Parameters
    ----------
    df : the table to render. Values are stringified as-is (format upstream).
    out_path : PNG path to write.
    title : optional title drawn above the table.
    sig_mask : optional boolean array (len == n rows). True rows are shaded
        with the "significant" highlight colour.
    """
    data = df.copy()
    for c in data.columns:
        data[c] = data[c].map(lambda v: "" if (isinstance(v, float) and
                                                np.isnan(v)) else str(v))
    n_rows, n_cols = data.shape

    # Column widths proportional to the longest string (header or cell).
    col_chars = []
    for c in data.columns:
        longest = max([len(str(c))] + [len(v) for v in data[c]])
        col_chars.append(max(longest, 4))
    total_chars = sum(col_chars)
    fig_w = max(6.0, 0.14 * total_chars * col_scale)
    fig_h = max(1.4, 0.45 * (n_rows + 1) + (0.5 if title else 0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold",
                     loc="left", pad=10)

    tbl = ax.table(cellText=data.values, colLabels=list(data.columns),
                   cellLoc="center", loc="center",
                   colWidths=[w / total_chars for w in col_chars])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.4)

    for (r, c), cell in tbl.get_cells().items() if hasattr(tbl, "get_cells") \
            else tbl.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if r == 0:  # header
            cell.set_facecolor(_TABLE_HEADER_BG)
            cell.set_text_props(color=_TABLE_HEADER_FG, fontweight="bold")
        else:
            row_idx = r - 1
            if sig_mask is not None and row_idx < len(sig_mask) \
                    and bool(sig_mask[row_idx]):
                cell.set_facecolor(_TABLE_SIG_BG)
            elif row_idx % 2 == 1:
                cell.set_facecolor(_TABLE_ROW_ALT)
            else:
                cell.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _gg_text(row) -> str:
    p_gg = row.get("p_GG_corr", np.nan)
    if isinstance(p_gg, (int, float)) and not (isinstance(p_gg, float)
                                               and np.isnan(p_gg)):
        return _fmt_p(p_gg)
    return "-"


def mixed_summary_frame(results: dict, key: str = "mixed") -> pd.DataFrame:
    """Tidy Bias+JND mixed-ANOVA effects table for rendering.

    key : 'mixed' (MAIN) or 'mixed_sensitivity'.
    """
    rows = []
    sig = []
    for dv in ["Bias", "JND"]:
        aov = results[key][dv]["aov"]
        for _, r in aov.iterrows():
            p = get_p(r)
            ddof1 = r.get("DF1", r.get("ddof1", ""))
            ddof2 = r.get("DF2", r.get("ddof2", ""))
            rows.append({
                "DV": dv,
                "Effect": r["Source"],
                "df": f"{int(ddof1)}, {int(ddof2)}"
                      if str(ddof1) != "" else "",
                "F": f"{float(r.get('F', np.nan)):.3f}",
                "p": _fmt_p(p),
                "p (GG)": _gg_text(r),
                "partial eta2": f"{float(r.get('np2', np.nan)):.3f}",
                "sig": "*" if (np.isfinite(p) and p < 0.05) else "",
            })
            sig.append(bool(np.isfinite(p) and p < 0.05))
    out = pd.DataFrame(rows)
    out.attrs["sig_mask"] = np.array(sig)
    return out


def descriptives_frame(df: pd.DataFrame) -> pd.DataFrame:
    """System x Finger mean +/- SD (n) for Bias and JND, wide by System."""
    rows = []
    for f in FINGER_LEVELS:
        row = {"Finger": f"{f} ({FINGER_FULLNAMES[f]})"}
        for dv in ["Bias", "JND"]:
            sub = df[df["Finger"] == f]
            for s in SYSTEM_LEVELS:
                v = sub.loc[sub["System"] == s, dv].dropna()
                if len(v):
                    row[f"{dv} {s} (mean+/-SD, n)"] = (
                        f"{v.mean():.1f} +/- {v.std(ddof=1):.1f} (n={len(v)})")
                else:
                    row[f"{dv} {s} (mean+/-SD, n)"] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts_summary_frame(results: dict) -> pd.DataFrame:
    """Planned L - N contrasts per finger for Bias and JND, with Holm p."""
    rows = []
    sig = []
    for dv in ["Bias", "JND"]:
        ct = results["contrasts"][dv]
        for _, r in ct.iterrows():
            rows.append({
                "DV": dv,
                "Finger": r["Finger"],
                "mean L": f"{r['mean_L']:.2f}",
                "mean N": f"{r['mean_N']:.2f}",
                "diff (L-N)": f"{r['diff_L_minus_N']:.2f}",
                "95% CI": f"[{r['CI95_low']:.1f}, {r['CI95_high']:.1f}]",
                "t": f"{r['T']:.2f}",
                "p (Holm)": _fmt_p(r["p_holm"]),
                "Cohen d": f"{r['cohen_d']:.2f}",
                "sig": "*" if bool(r.get("sig_holm", False)) else "",
            })
            sig.append(bool(r.get("sig_holm", False)))
    out = pd.DataFrame(rows)
    out.attrs["sig_mask"] = np.array(sig)
    return out


def assumption_checks_frame(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Normality (Shapiro), variance homogeneity (Levene), sphericity (Mauchly).

    A compact diagnostics table so the reader can see WHETHER the classical
    ANOVA assumptions hold for each DV. Shapiro is run on the model residuals
    (value minus its System x Finger cell mean); Levene compares the two System
    groups; Mauchly is read from the already-computed mixed-ANOVA result.
    """
    rows = []
    sig = []
    for dv in ["Bias", "JND"]:
        work = analysis_frame(df, dv)
        cell_mean = work.groupby(["System", "Finger"], observed=True)[dv] \
                        .transform("mean")
        resid = (work[dv] - cell_mean).to_numpy()
        try:
            sh_p = float(stats.shapiro(resid).pvalue)
        except Exception:
            sh_p = np.nan
        l = work.loc[work["System"] == "L", dv].dropna()
        n = work.loc[work["System"] == "N", dv].dropna()
        try:
            lev_p = float(stats.levene(l, n, center="median").pvalue)
        except Exception:
            lev_p = np.nan
        sph = results["mixed"][dv].get("sphericity_note", "")
        normal_ok = np.isfinite(sh_p) and sh_p >= 0.05
        var_ok = np.isfinite(lev_p) and lev_p >= 0.05
        rows.append({
            "DV": dv,
            "Normality (Shapiro p)": _fmt_p(sh_p),
            "Normality OK?": "yes" if normal_ok else "NO",
            "Equal var (Levene p)": _fmt_p(lev_p),
            "Equal var OK?": "yes" if var_ok else "NO",
            "Sphericity (Mauchly)": "met" if "sphericity=met" in sph
                                    or "='met'" in sph else
                                    ("VIOLATED" if "VIOLATED" in sph else "-"),
        })
        # Flag the row if any assumption fails (draws the eye to problems).
        sig.append(not (normal_ok and var_ok))
    out = pd.DataFrame(rows)
    out.attrs["sig_mask"] = np.array(sig)
    return out


def render_summary_tables(df: pd.DataFrame, results: dict,
                          out_dir: str) -> dict:
    """Render all summary tables as PNG images. Returns {name: path}.

    Produces:
      * mixed_anova_summary.png        - MAIN Bias+JND effects (sig shaded)
      * mixed_anova_sensitivity.png    - sensitivity (flagged fits removed)
      * descriptives.png               - System x Finger mean +/- SD (n)
      * planned_contrasts.png          - L - N per finger, Holm p (sig shaded)
      * assumption_checks.png          - normality / variance / sphericity
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    main_tbl = mixed_summary_frame(results, "mixed")
    paths["mixed_anova_summary"] = dataframe_to_image(
        main_tbl, os.path.join(out_dir, "mixed_anova_summary.png"),
        title="MAIN mixed-design ANOVA: System (between) x Finger (within)",
        sig_mask=main_tbl.attrs.get("sig_mask"))

    sens_tbl = mixed_summary_frame(results, "mixed_sensitivity")
    paths["mixed_anova_sensitivity"] = dataframe_to_image(
        sens_tbl, os.path.join(out_dir, "mixed_anova_sensitivity.png"),
        title="SENSITIVITY mixed-design ANOVA (flagged/degenerate fits removed)",
        sig_mask=sens_tbl.attrs.get("sig_mask"))

    desc_tbl = descriptives_frame(df)
    paths["descriptives"] = dataframe_to_image(
        desc_tbl, os.path.join(out_dir, "descriptives.png"),
        title="Descriptives: Bias and JND by System x Finger (mean +/- SD)",
        col_scale=1.15)

    ct_tbl = contrasts_summary_frame(results)
    paths["planned_contrasts"] = dataframe_to_image(
        ct_tbl, os.path.join(out_dir, "planned_contrasts.png"),
        title="Planned contrasts: System L - N within each finger (Holm)",
        sig_mask=ct_tbl.attrs.get("sig_mask"))

    asm_tbl = assumption_checks_frame(df, results)
    paths["assumption_checks"] = dataframe_to_image(
        asm_tbl, os.path.join(out_dir, "assumption_checks.png"),
        title="ANOVA assumption checks (flagged row = an assumption is violated)",
        sig_mask=asm_tbl.attrs.get("sig_mask"))

    return paths


# --------------------------------------------------------------------------- #
# (8) Subgroup engine: run the whole pipeline on raw data and on filtered
#     subgroups, mirroring the results/ folder architecture per subgroup.
# --------------------------------------------------------------------------- #
#
# Each subgroup EXCLUDES "bad" subjects at a graded strictness, so the analyses
# read as a sensitivity ladder (loosest -> strictest cleaning). Filtering is at
# the SUBJECT level: a subject's four finger rows are kept or dropped together,
# so the repeated-measures design stays balanced. A finger is "bad" if its
# metric crosses the threshold; the level says HOW MANY bad fingers a subject
# needs before the whole subject is removed.

# Per-finger "bad" thresholds.
SUCCESS_RATE_THRESHOLD = 0.70   # success rate strictly below this is "bad"
PSE_SHIFT_THRESHOLD = 100.0     # |PSE - 85| strictly above this is "bad"
JND_GT_250_THRESHOLD = 250.0    # JND strictly above this is "bad"
JND_GT_100_THRESHOLD = 100.0    # JND strictly above this is "bad"

# Per-finger success (% correct) source. This is READ from the psychophysics
# pipeline's PRECOMPUTED per-subject x finger success summary - it is NOT
# recomputed here. (The pipeline already scored every trial and aggregated
# success_rate = n_correct / n_trials per subject x finger.)
DEFAULT_SUCCESS_SUMMARY_PATH = os.path.join(
    PSYCHO_RESULTS_DIR, "success_summary_by_subject_finger.csv"
)

# (metric_key, human description). Folder name == metric_key.
SUBGROUP_PLAN = [
    ("success_lt70", "Per-finger success rate < 70%"),
    ("pse_shift_gt100", "|PSE - 85| > 100"),
    ("jnd_gt250", "JND > 250"),
    ("jnd_gt100", "JND > 100"),
]

# (level_key, min bad-finger count to EXCLUDE the subject, label).
# "all_fingers" means every one of the subject's fingers is bad.
SUBGROUP_LEVELS = [
    ("all_fingers", "all", "all fingers bad"),
    ("at_least_2_fingers", 2, ">= 2 fingers bad"),
    ("at_least_1_finger", 1, ">= 1 finger bad"),
]


def load_success_rates(path: Optional[str] = None) -> pd.DataFrame:
    """READ the precomputed per-subject x finger success rate (% correct).

    This does NOT recompute success - it reads the ``success_rate`` column that
    the psychophysics pipeline already produced in
    ``success_summary_by_subject_finger.csv``. Returns columns: Subject, Finger,
    success_rate, n_success_trials.

    The live success summary lives under the (large) psychophysics results tree.
    If that file is not available (e.g. Dropbox not synced) an EMPTY frame with
    the correct columns is returned and a warning is emitted; the ``success_lt70``
    subgroup filter is then skipped by ``run_all_subgroups`` rather than crashing
    the whole pipeline. The metric-on-the-data-frame filters (pse_shift / jnd)
    are unaffected.
    """
    path = path or DEFAULT_SUCCESS_SUMMARY_PATH
    if not os.path.exists(path):
        warnings.warn(
            "Per-finger success summary not found at "
            f"{path!r}; the success_lt70 subgroup filter will be skipped. "
            "Re-sync the psychophysics results to enable it.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["Subject", "Finger", "success_rate",
                                     "n_success_trials"])
    s = pd.read_csv(path, usecols=["subject_id", "finger_condition",
                                   "success_rate", "n_trials"])
    s = s.rename(columns={"subject_id": "Subject",
                          "finger_condition": "Finger",
                          "n_trials": "n_success_trials"})
    return s[["Subject", "Finger", "success_rate", "n_success_trials"]]


def attach_success_rate(df: pd.DataFrame,
                        success_rates: Optional[pd.DataFrame] = None
                        ) -> pd.DataFrame:
    """Left-merge per-finger success_rate onto the summary frame (no drops).

    If the success summary is empty/unavailable, a ``success_rate`` column of
    NaN is attached so downstream code can detect (and skip) the success filter.
    """
    if success_rates is None:
        success_rates = load_success_rates()
    if success_rates is None or len(success_rates) == 0:
        out = df.copy()
        out["success_rate"] = np.nan
        return out
    out = df.merge(success_rates[["Subject", "Finger", "success_rate"]],
                   on=["Subject", "Finger"], how="left")
    return out


def _finger_bad_mask(df: pd.DataFrame, metric: str) -> pd.Series:
    """Boolean per-row mask: is THIS finger 'bad' for the given metric?

    NaN metric values yield False (a non-estimable finger does not, by itself,
    trigger exclusion; such cases are surfaced by the QC report instead).
    """
    if metric == "success_lt70":
        if "success_rate" not in df.columns:
            raise KeyError("success_rate column missing; call "
                           "attach_success_rate() first.")
        return df["success_rate"] < SUCCESS_RATE_THRESHOLD
    if metric == "pse_shift_gt100":
        return df["Bias"].abs() > PSE_SHIFT_THRESHOLD
    if metric == "jnd_gt250":
        return df["JND"] > JND_GT_250_THRESHOLD
    if metric == "jnd_gt100":
        return df["JND"] > JND_GT_100_THRESHOLD
    raise ValueError(f"Unknown subgroup metric: {metric}")


def build_subgroup(df: pd.DataFrame, metric: str, level: str):
    """Return (kept_df, excluded_subjects, per_subject_counts).

    A subject is EXCLUDED when its number of bad fingers meets the level rule.
    """
    bad = _finger_bad_mask(df, metric)
    counts = (df.assign(_bad=bad)
                .groupby("Subject", observed=True)["_bad"]
                .agg(n_bad="sum", n_fing="count"))
    min_count = dict((k, v) for k, v, _ in SUBGROUP_LEVELS)[level]
    if min_count == "all":
        excl_mask = (counts["n_bad"] == counts["n_fing"]) & (counts["n_bad"] > 0)
    else:
        excl_mask = counts["n_bad"] >= int(min_count)
    excluded = set(counts.index[excl_mask])
    kept = df[~df["Subject"].isin(excluded)].copy()
    return kept, excluded, counts


def plot_bootstrap_diff(diff_ci: pd.DataFrame, dv: str, out_path: str) -> str:
    """Bootstrap L - N difference per finger with 95% CI (subgroup-safe)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    diff_ci = diff_ci.reset_index(drop=True)
    # One point per finger, drawn in that finger's dedicated colour.
    for i, row in diff_ci.iterrows():
        col = _FINGER_COLORS.get(str(row["Finger"]), "#2c3e50")
        ax.errorbar(i, row["diff_L_minus_N"],
                    yerr=[[row["diff_L_minus_N"] - row["ci_low"]],
                          [row["ci_high"] - row["diff_L_minus_N"]]],
                    fmt="o", color=col, capsize=6, lw=2, elinewidth=2.2,
                    markersize=10, markeredgecolor="black")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(list(range(len(diff_ci))))
    ax.set_xticklabels([f"{r.Finger}\n({r.Finger_name})"
                        for _, r in diff_ci.iterrows()])
    ax.set_xlabel("Finger")
    ax.set_ylabel(f"{dv}  (L - N)")
    ax.set_title(f"Bootstrap L - N difference per finger: {dv} "
                 "(subject-level, 95% CI)")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _ensure_pipeline_dirs(base_root: str, cohort: str) -> dict:
    """Per-DV layout for mixed_design/ and bootstrap/; per-cohort report/table.

    mixed_design/ and bootstrap/ split by DV (``bias``/``jnd``), each with
    ``csv/`` + ``figures/``; the cohort is encoded in the FILENAME prefix
    (``<cohort>__...``), NOT in a sub-folder.

    Returns a dict of directories:
      * ``mixed``   -> base_root/mixed_design   (then <bias|jnd>/{csv,figures}). The
                       exploratory-8group and planned-contrast outputs are FOLDED
                       IN here (context for the two-way design).
      * ``boot``    -> base_root/bootstrap      (then <bias|jnd>/{csv,figures}).
      * ``reports`` -> base_root/report/<cohort>      (kept per-cohort).
      * ``tables``  -> base_root/summary_table/<cohort> (kept per-cohort).

    The legacy ``explo`` / ``contr`` keys alias ``mixed`` so existing callers
    that still reference them route their writes into mixed_design/.
    """
    paths = {
        "mixed": os.path.join(base_root, "mixed_design"),
        "boot": os.path.join(base_root, "bootstrap"),
        "reports": os.path.join(base_root, "report", cohort),
        "tables": os.path.join(base_root, "summary_table", cohort),
    }
    # Exploratory 8-group and contrasts are folded into mixed_design.
    paths["explo"] = paths["mixed"]
    paths["contr"] = paths["mixed"]
    for base in ("mixed", "boot"):
        for sub in ("bias", "jnd"):
            os.makedirs(os.path.join(paths[base], sub, "csv"), exist_ok=True)
            os.makedirs(os.path.join(paths[base], sub, "figures"), exist_ok=True)
    os.makedirs(paths["reports"], exist_ok=True)
    os.makedirs(paths["tables"], exist_ok=True)
    return paths


def _pipeline_complete(results: dict) -> bool:
    """True iff every block needed by the report/tables is present for both DVs."""
    for block in ("exploratory", "mixed", "mixed_sensitivity", "contrasts",
                  "bootstrap"):
        for dv in ("Bias", "JND"):
            if dv not in results.get(block, {}):
                return False
    return True


def run_full_pipeline(df: pd.DataFrame, base_root: str, cohort: str, *,
                      n_bootstrap: int = N_BOOTSTRAP,
                      seed: int = RANDOM_SEED,
                      make_figures: bool = True,
                      label: str = "") -> dict:
    """Run the COMPLETE analysis on ``df`` for one ``cohort`` and write outputs
    in the FLAT layout: mixed_design/ and bootstrap/ hold ``<cohort>__``-prefixed
    files; report/<cohort> and summary_table/<cohort> stay per-cohort.
    Best-effort: any step that fails is recorded in the cohort's
    report/pipeline_notes.txt and skipped, never crashing the run.

    Returns the assembled ``results`` dict (same shape the notebook builds).
    """
    label = label or cohort
    P = _ensure_pipeline_dirs(base_root, cohort)
    notes: list = []
    validation = validate_data(df)
    df_flags = compute_qc_flags(df)
    qc_summary = qc_flag_summary(df_flags)
    df_flags.to_csv(os.path.join(P["reports"], "qc_flags_all_rows.csv"),
                    index=False)
    qc_summary.to_csv(os.path.join(P["reports"], "qc_flag_summary.csv"),
                      index=False)

    results: dict = {
        "data_path": df.attrs.get("source_path", label),
        "label": label,
        "cohort": cohort,
        "qc_summary": qc_summary,
        "exploratory": {}, "mixed": {}, "mixed_sensitivity": {},
        "contrasts": {}, "bootstrap": {}, "notes": notes,
        "n_subjects": int(df["Subject"].nunique()),
        "validation": validation,
    }

    for dv in ["Bias", "JND"]:
        sub = "bias" if dv == "Bias" else "jnd"

        # Per-DV sub-folders: mixed_design/<bias|jnd>/{csv,figures}; same for
        # bootstrap. The cohort stays a filename prefix (<cohort>__...).
        mixed_csv = os.path.join(P["mixed"], sub, "csv")
        mixed_fig = os.path.join(P["mixed"], sub, "figures")
        boot_csv = os.path.join(P["boot"], sub, "csv")
        boot_fig = os.path.join(P["boot"], sub, "figures")
        for _d in (mixed_csv, mixed_fig, boot_csv, boot_fig):
            os.makedirs(_d, exist_ok=True)

        try:
            aov, posthoc = exploratory_one_way(df, dv)
            results["exploratory"][dv] = {"aov": aov, "posthoc": posthoc}
            aov.to_csv(os.path.join(
                mixed_csv,
                f"{cohort}__exploratory_oneway_anova.csv"), index=False)
            posthoc.to_csv(os.path.join(
                mixed_csv,
                f"{cohort}__exploratory_oneway_posthoc_holm.csv"),
                index=False)
        except Exception as exc:
            notes.append(f"exploratory[{dv}] skipped: {exc}")

        try:
            results["mixed"][dv] = mixed_anova(df, dv, drop_flagged=False)
            results["mixed"][dv]["aov"].to_csv(os.path.join(
                mixed_csv, f"{cohort}__mixed_anova_main.csv"),
                index=False)
        except Exception as exc:
            notes.append(f"mixed[{dv}] skipped: {exc}")

        try:
            results["mixed_sensitivity"][dv] = mixed_anova(
                df, dv, drop_flagged=True)
            results["mixed_sensitivity"][dv]["aov"].to_csv(os.path.join(
                mixed_csv, f"{cohort}__mixed_anova_sensitivity.csv"),
                index=False)
        except Exception as exc:
            notes.append(f"mixed_sensitivity[{dv}] skipped: {exc}")

        try:
            ct = planned_contrasts(df, dv, drop_flagged=False)
            results["contrasts"][dv] = ct
            ct.to_csv(os.path.join(
                mixed_csv,
                f"{cohort}__planned_contrasts_L_vs_N.csv"), index=False)
        except Exception as exc:
            notes.append(f"contrasts[{dv}] skipped: {exc}")

        try:
            cell_ci, diff_ci = bootstrap_cis(df, dv, n_boot=n_bootstrap,
                                             seed=seed, drop_flagged=False)
            results["bootstrap"][dv] = {"cell": cell_ci, "diff": diff_ci}
            cell_ci.to_csv(os.path.join(
                boot_csv, f"{cohort}__bootstrap_cell_means.csv"),
                index=False)
            diff_ci.to_csv(os.path.join(
                boot_csv, f"{cohort}__bootstrap_LminusN_diff.csv"),
                index=False)
        except Exception as exc:
            notes.append(f"bootstrap[{dv}] skipped: {exc}")

        if make_figures:
            try:
                hline = 0.0 if dv == "Bias" else None
                plot_eight_groups(
                    df, dv, dv, f"Exploratory: {dv} by System_Finger group",
                    os.path.join(mixed_fig,
                                 f"{cohort}__exploratory_8group.png"),
                    hline=hline)
                plot_dv_by_finger(
                    df, dv, hline, dv,
                    f"{dv} by Finger and System (mean +/- 95% CI)",
                    os.path.join(mixed_fig,
                                 f"{cohort}__by_finger_system.png"))
                if dv in results["contrasts"] and len(results["contrasts"][dv]):
                    plot_contrasts(
                        results["contrasts"][dv], dv, dv,
                        f"Planned contrast: {dv} difference (L - N) per finger",
                        os.path.join(mixed_fig,
                                     f"{cohort}__planned_contrast.png"))
                if dv in results["bootstrap"]:
                    plot_bootstrap_diff(
                        results["bootstrap"][dv]["diff"], dv,
                        os.path.join(boot_fig,
                                     f"{cohort}__bootstrap_LminusN.png"))
            except Exception as exc:
                notes.append(f"figures[{dv}] skipped: {exc}")

    # Report + table images only when every required block is present.
    if _pipeline_complete(results):
        try:
            report_txt = build_report(validation, results)
            with open(os.path.join(P["reports"],
                                   "anova_statistics_report.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(report_txt)
            with open(os.path.join(P["reports"],
                                   "anova_statistics_report.md"), "w",
                      encoding="utf-8") as fh:
                fh.write("```\n" + report_txt + "\n```\n")
        except Exception as exc:
            notes.append(f"report skipped: {exc}")
        try:
            results["table_images"] = render_summary_tables(
                df, results, P["tables"])
        except Exception as exc:
            notes.append(f"summary tables skipped: {exc}")
    else:
        notes.append("report + summary tables skipped: pipeline incomplete "
                     "(a required ANOVA block failed for at least one DV).")

    with open(os.path.join(P["reports"], "pipeline_notes.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(f"label: {label}\n")
        fh.write(f"n_subjects: {results['n_subjects']}\n")
        fh.write(f"n_rows: {len(df)}\n\n")
        fh.write("\n".join(notes) if notes else "No issues; all steps ran.")
    return results


# --------------------------------------------------------------------------- #
# (9) One-way per-system (finger) analysis bridge
# --------------------------------------------------------------------------- #
#
# Reuses the numpy/scipy stat + matplotlib plot helpers from the sibling
# ``oneway_anova`` module (imported as ``owa`` at the top of this file). Only
# the SAVE layout is new: outputs are written FLAT under results/oneway_filters
# (one file per system, ``<system>__`` prefixed) and results/oneway/subjects
# (one file per subject, ``<system>_<subject>__`` prefixed). The stats are NOT
# re-derived here.

def run_oneway_flat(data_path: Optional[str] = None,
                    base_root: str = "results",
                    fig_dpi: int = 150) -> dict:
    """Run the one-way (per-system finger) analysis and save it FLAT.

    Parameters
    ----------
    data_path : path to the per-subject x finger summary. Defaults to the same
        frozen file the ANOVA uses (``LEGACY_FROZEN_DATA_PATH``).
    base_root : results root (the flat folders are created under it).
    fig_dpi : figure DPI.

    Returns
    -------
    dict with keys:
      * ``data_path`` : the file read.
      * ``systems``   : list of systems analysed.
      * ``results``   : {system: analyze_system(...) dict} (test stats reused
                        for the report + the oneway summary table image).
      * ``test_summaries`` : {system: combined test_summary DataFrame}.
      * ``note``      : present (and the rest empty) if the oneway module could
                        not be imported.
    """
    out = {"data_path": None, "systems": [], "results": {},
           "test_summaries": {}, "note": None}
    if owa is None:
        out["note"] = (f"oneway_anova module unavailable; one-way section "
                       f"skipped ({_OWA_IMPORT_ERROR}).")
        return out

    if data_path is None:
        data_path = LEGACY_FROZEN_DATA_PATH
    out["data_path"] = data_path

    df = owa.load_pse_jnd(data_path)

    of_csv = os.path.join(base_root, "oneway_filters", "csv")
    of_fig = os.path.join(base_root, "oneway_filters", "figures")
    subj_csv = os.path.join(base_root, "oneway", "subjects", "csv")
    subj_fig = os.path.join(base_root, "oneway", "subjects", "figures")
    for d in (of_csv, of_fig, subj_csv, subj_fig):
        os.makedirs(d, exist_ok=True)

    from pathlib import Path

    for system in owa.SYSTEM_LEVELS:
        sys_df = df[df["System"].astype(str) == system].copy()
        if sys_df.empty or sys_df["Subject"].nunique() < 2:
            continue
        out["systems"].append(system)
        res = owa.analyze_system(sys_df, system)
        out["results"][system] = res
        fingers = owa.fingers_present(sys_df)

        # Combined test summary (both tests x both DVs).
        summary_rows = []
        for dv in owa.DEPENDENT_VARS:
            a, fr = res["anova"][dv], res["friedman"][dv]
            summary_rows.append({
                "system": system, "dv": dv, "test": "one_way_anova",
                "statistic_name": "F", "statistic": a.get("F"),
                "df1": a.get("df_between"), "df2": a.get("df_within"),
                "p_value": a.get("p_value"),
                "effect_size_name": "eta_squared",
                "effect_size": a.get("eta_squared"), "n": a.get("n_total")})
            summary_rows.append({
                "system": system, "dv": dv, "test": "friedman",
                "statistic_name": "chi_square",
                "statistic": fr.get("chi_square"), "df1": fr.get("df"),
                "df2": np.nan, "p_value": fr.get("p_value"),
                "effect_size_name": "kendalls_w",
                "effect_size": fr.get("kendalls_w"),
                "n": fr.get("n_complete_subjects")})
        test_summary = pd.DataFrame(summary_rows)
        out["test_summaries"][system] = test_summary
        test_summary.to_csv(
            os.path.join(of_csv, f"{system}__test_summary.csv"), index=False)

        # Per-DV CSVs (descriptives / assumptions / post-hoc).
        for dv in owa.DEPENDENT_VARS:
            res["descriptives"][dv].to_csv(
                os.path.join(of_csv, f"{system}__descriptives_{dv}.csv"),
                index=False)
            res["assumptions"][dv].to_csv(
                os.path.join(of_csv, f"{system}__assumption_checks_{dv}.csv"),
                index=False)
            res["posthoc_param"][dv].to_csv(
                os.path.join(of_csv, f"{system}__posthoc_welch_{dv}.csv"),
                index=False)
            res["posthoc_nonparam"][dv].to_csv(
                os.path.join(of_csv, f"{system}__posthoc_wilcoxon_{dv}.csv"),
                index=False)
            # Figures.
            owa.plot_dv_by_finger(
                sys_df, dv, system, res["descriptives"][dv],
                res["anova"][dv], res["friedman"][dv],
                Path(of_fig) / f"{system}__{dv.lower()}_by_finger.png",
                fig_dpi)
            owa.plot_posthoc_matrix(
                res["posthoc_param"][dv], dv, system, fingers,
                Path(of_fig) / f"{system}__posthoc_welch_{dv.lower()}.png",
                fig_dpi=fig_dpi)
            owa.plot_posthoc_matrix(
                res["posthoc_nonparam"][dv], dv, system, fingers,
                Path(of_fig) / f"{system}__posthoc_wilcoxon_{dv.lower()}.png",
                fig_dpi=fig_dpi)

        # Sensitivity (band-excluded fits dropped), one combined CSV.
        sens = sys_df[~sys_df["excluded_from_group_analysis"].astype(bool)]
        if len(sens) and sens["Subject"].nunique() >= 2:
            sens_rows = []
            for dv in owa.DEPENDENT_VARS:
                a = owa.one_way_anova(sens, dv)
                fr = owa.friedman_test(sens, dv)
                sens_rows.append({
                    "system": system, "dv": dv, "test": "one_way_anova",
                    "F": a.get("F"), "p_value": a.get("p_value"),
                    "eta_squared": a.get("eta_squared"), "n": a.get("n_total")})
                sens_rows.append({
                    "system": system, "dv": dv, "test": "friedman",
                    "chi_square": fr.get("chi_square"),
                    "p_value": fr.get("p_value"),
                    "kendalls_w": fr.get("kendalls_w"),
                    "n": fr.get("n_complete_subjects")})
            pd.DataFrame(sens_rows).to_csv(
                os.path.join(of_csv, f"{system}__test_summary_sensitivity.csv"),
                index=False)

        # Per-subject FLAT outputs.
        for subject in sorted(sys_df["Subject"].astype(str).unique()):
            sub_df = sys_df[sys_df["Subject"].astype(str) == subject]
            keep_cols = [c for c in ["Subject", "System", "Finger", "Bias",
                                     "JND", "fit_warning",
                                     "excluded_from_group_analysis"]
                         if c in sub_df.columns]
            sub_df[keep_cols].to_csv(
                os.path.join(subj_csv,
                             f"{system}_{subject}__pse_jnd_by_finger.csv"),
                index=False)
            owa.plot_subject_dv(
                sub_df, subject,
                Path(subj_fig)
                / f"{system}_{subject}__bias_jnd_by_finger.png",
                fig_dpi)

    return out


def oneway_summary_frame(oneway_res: dict) -> pd.DataFrame:
    """Tidy System x DV x test summary for the oneway table image.

    Built from the per-system test_summary frames returned by run_oneway_flat.
    """
    rows = []
    sig = []
    for system in oneway_res.get("systems", []):
        ts = oneway_res["test_summaries"].get(system)
        if ts is None:
            continue
        for _, r in ts.iterrows():
            p = r.get("p_value", np.nan)
            stat = r.get("statistic", np.nan)
            es = r.get("effect_size", np.nan)
            try:
                p = float(p)
            except (TypeError, ValueError):
                p = np.nan
            rows.append({
                "System": system,
                "DV": r.get("dv", ""),
                "Test": r.get("test", ""),
                f"{r.get('statistic_name', 'stat')}": (
                    f"{float(stat):.3f}" if np.isfinite(
                        float(stat) if stat == stat else np.nan) else "-"),
                "p": _fmt_p(p),
                "effect": (f"{float(es):.3f}"
                           if (es == es and es is not None) else "-"),
                "n": ("" if r.get("n") is None or not (r.get("n") == r.get("n"))
                      else int(r.get("n"))),
            })
            sig.append(bool(np.isfinite(p) and p < 0.05))
    # statistic_name varies (F vs chi_square); normalise to one column.
    norm = []
    for r in rows:
        stat_val = r.pop("F", None)
        if stat_val is None:
            stat_val = r.pop("chi_square", None)
        # Any leftover statistic-named key.
        for k in list(r.keys()):
            if k not in ("System", "DV", "Test", "p", "effect", "n",
                         "statistic"):
                if stat_val is None:
                    stat_val = r.pop(k)
                else:
                    r.pop(k)
        r["statistic"] = stat_val if stat_val is not None else "-"
        norm.append({"System": r["System"], "DV": r["DV"], "Test": r["Test"],
                     "statistic": r["statistic"], "p": r["p"],
                     "effect": r["effect"], "n": r["n"]})
    out = pd.DataFrame(norm)
    out.attrs["sig_mask"] = np.array(sig)
    return out


def render_oneway_summary_table(oneway_res: dict, out_dir: str) -> Optional[str]:
    """Render the one-way per-system summary as a PNG table. Returns its path."""
    if not oneway_res.get("systems"):
        return None
    os.makedirs(out_dir, exist_ok=True)
    tbl = oneway_summary_frame(oneway_res)
    if tbl.empty:
        return None
    return dataframe_to_image(
        tbl, os.path.join(out_dir, "oneway_per_system_summary.png"),
        title="One-way per-system (finger) analysis: ANOVA (F) + Friedman "
              "(chi2) by System x DV",
        sig_mask=tbl.attrs.get("sig_mask"))


def subgroup_master_frame(rows: list) -> pd.DataFrame:
    """Compact cross-subgroup comparison frame for the master table image."""
    out = pd.DataFrame(rows)
    return out


def _effect_p_row(res: dict) -> dict:
    """Pull System/Finger/Interaction p (Bias & JND) from a results dict."""
    out = {}
    for dv in ("Bias", "JND"):
        aov = res.get("mixed", {}).get(dv, {}).get("aov")
        for eff in ("System", "Finger", "Interaction"):
            val = np.nan
            if aov is not None:
                sel = aov.loc[aov["Source"] == eff]
                if len(sel):
                    val = round(float(get_p(sel.iloc[0])), 4)
            out[f"{dv}_{eff}_p"] = val
    return out


def run_all_subgroups(df: pd.DataFrame, base_root: str = "results", *,
                      success_rates: Optional[pd.DataFrame] = None,
                      raw_results: Optional[dict] = None,
                      n_bootstrap: int = N_BOOTSTRAP,
                      seed: int = RANDOM_SEED,
                      min_subjects: int = 4,
                      min_per_system: int = 2) -> tuple:
    """Run the full pipeline on every (filter, level) subgroup, FLAT layout.

    Each subgroup is one cohort ``<metric>__<level>`` written across the flat
    analysis folders (``<cohort>__`` filename prefixes). A per-cohort
    exclusion record is saved to base_root/report/<cohort>/subgroup_membership.csv.
    If ``raw_results`` (the already-computed full-cohort results) is given, a
    ``raw`` row is prepended to the master so the comparison includes the
    unfiltered analysis WITHOUT recomputing it. A master summary CSV + image is
    written to base_root/. Returns (master_df, all_results).
    """
    work = df if "success_rate" in df.columns else attach_success_rate(
        df, success_rates)
    os.makedirs(base_root, exist_ok=True)

    # If success rates are unavailable (success_rate all NaN), the success_lt70
    # filter cannot be evaluated; skip it (with a note) rather than crash.
    success_available = ("success_rate" in work.columns
                         and bool(work["success_rate"].notna().any()))

    rows: list = []
    all_results: dict = {}

    # Optional raw row first (no recompute - read from the passed-in results).
    if raw_results is not None:
        raw_row = {
            "cohort": "raw", "filter": "(none)",
            "filter_desc": "Raw - all subjects", "level": "-",
            "n_excluded": 0,
            "n_subjects": int(raw_results.get("n_subjects", 0)),
            "n_L": np.nan, "n_N": np.nan, "status": "ok",
        }
        raw_row.update(_effect_p_row(raw_results))
        rows.append(raw_row)

    for metric, desc in SUBGROUP_PLAN:
        if metric == "success_lt70" and not success_available:
            # No success data: record a single skip note for this filter.
            for level, _min, level_label in SUBGROUP_LEVELS:
                cohort = f"{metric}__{level}"
                reports_dir = os.path.join(base_root, "report", cohort)
                os.makedirs(reports_dir, exist_ok=True)
                with open(os.path.join(reports_dir, "SKIPPED.txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write(f"Cohort {cohort} skipped: per-finger success "
                             "summary unavailable (psychophysics results not "
                             "synced); success_lt70 filter cannot be "
                             "evaluated.\n")
                row = {
                    "cohort": cohort, "filter": metric, "filter_desc": desc,
                    "level": level_label, "n_excluded": np.nan,
                    "n_subjects": np.nan, "n_L": np.nan, "n_N": np.nan,
                    "status": "skipped_no_success_data",
                }
                for dv in ("Bias", "JND"):
                    for eff in ("System", "Finger", "Interaction"):
                        row[f"{dv}_{eff}_p"] = np.nan
                rows.append(row)
                all_results[cohort] = {"status": "skipped_no_success_data"}
            continue
        for level, _min, level_label in SUBGROUP_LEVELS:
            kept, excluded, counts = build_subgroup(work, metric, level)
            cohort = f"{metric}__{level}"
            reports_dir = os.path.join(base_root, "report", cohort)
            os.makedirs(reports_dir, exist_ok=True)

            kept_subjects = kept.drop_duplicates("Subject")
            n_subj = int(kept["Subject"].nunique())
            per_sys = (kept_subjects.groupby("System", observed=True)["Subject"]
                       .count().reindex(SYSTEM_LEVELS).fillna(0).astype(int))

            # Persist exactly which subjects were excluded and why.
            excl_rec = counts.reset_index().rename(
                columns={"n_bad": "n_bad_fingers", "n_fing": "n_fingers"})
            excl_rec["excluded"] = excl_rec["Subject"].isin(excluded)
            excl_rec.insert(0, "level", level)
            excl_rec.insert(0, "filter", metric)
            excl_rec.to_csv(os.path.join(reports_dir,
                            "subgroup_membership.csv"), index=False)

            row = {
                "cohort": cohort, "filter": metric, "filter_desc": desc,
                "level": level_label,
                "n_excluded": len(excluded), "n_subjects": n_subj,
                "n_L": int(per_sys["L"]), "n_N": int(per_sys["N"]),
            }

            if n_subj < min_subjects or (per_sys < min_per_system).any():
                row["status"] = "skipped_too_small"
                for dv in ("Bias", "JND"):
                    for eff in ("System", "Finger", "Interaction"):
                        row[f"{dv}_{eff}_p"] = np.nan
                with open(os.path.join(reports_dir, "SKIPPED.txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write(f"Cohort {cohort} skipped: N={n_subj} "
                             f"(L={per_sys['L']}, N={per_sys['N']}); needs "
                             f">= {min_subjects} subjects and "
                             f">= {min_per_system} per system.\n")
                rows.append(row)
                all_results[cohort] = {"status": "skipped",
                                       "n_subjects": n_subj}
                continue

            kept.attrs["source_path"] = f"subgroup:{cohort}"
            res = run_full_pipeline(kept, base_root, cohort,
                                    n_bootstrap=n_bootstrap, seed=seed,
                                    label=cohort)
            all_results[cohort] = res
            row["status"] = "ok"
            row.update(_effect_p_row(res))
            rows.append(row)

    master = subgroup_master_frame(rows)
    master.to_csv(os.path.join(base_root, "subgroup_master_summary.csv"),
                  index=False)

    # Master comparison table image (System effect is the headline; Finger and
    # interaction p's included compactly).
    disp = master.copy()

    def _n_label(r):
        if pd.isna(r["n_subjects"]):
            return "-"
        if pd.isna(r["n_L"]) or pd.isna(r["n_N"]):
            return f"{int(r['n_subjects'])}"
        return f"{int(r['n_subjects'])} ({int(r['n_L'])}/{int(r['n_N'])})"

    def _excl_label(v):
        return "-" if pd.isna(v) else str(int(v))

    disp["N (L/N)"] = disp.apply(_n_label, axis=1)
    show = pd.DataFrame({
        "Filter": disp["filter_desc"],
        "Level": disp["level"],
        "excl": disp["n_excluded"].map(_excl_label),
        "N (L/N)": disp["N (L/N)"],
        "Bias: Sys p": disp["Bias_System_p"].map(_fmt_p),
        "Bias: Fing p": disp["Bias_Finger_p"].map(_fmt_p),
        "Bias: Int p": disp["Bias_Interaction_p"].map(_fmt_p),
        "JND: Sys p": disp["JND_System_p"].map(_fmt_p),
        "JND: Fing p": disp["JND_Finger_p"].map(_fmt_p),
        "JND: Int p": disp["JND_Interaction_p"].map(_fmt_p),
    })
    sig = ((master[["Bias_System_p", "JND_System_p"]] < 0.05).any(axis=1)
           ).to_numpy()
    dataframe_to_image(
        show, os.path.join(base_root, "subgroup_master_summary.png"),
        title="Subgroup sensitivity ladder: mixed-ANOVA p-values across "
              "cleaning rules (shaded = a System effect reaches p < .05)",
        sig_mask=sig, col_scale=1.25, fontsize=9)

    return master, all_results
