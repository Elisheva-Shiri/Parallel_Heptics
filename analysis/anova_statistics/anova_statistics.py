"""
anova_statistics.py
===================

Statistics/reporting pipeline for the 2AFC stiffness-discrimination study.

The module reads an existing per-subject x finger psychophysics summary CSV and
writes validation tables, exploratory one-way ANOVA, the confirmatory mixed
System x Finger ANOVA, planned contrasts, bootstrap CIs, figures, and a text
report. Psychometric fits, trial scoring, calculations, and visual defaults are
not recomputed here.

Design summary
--------------
* System is the between-subject factor (L/N).
* Finger is the within-subject repeated factor (I/M/P/R).
* Subject is the repeated-measures unit.
* Bias = PSE - 8.5 mm/m; JND = (x75 - x25) / 2 from the frozen upstream fits.

Main analysis: mixed-design ANOVA, because each subject contributes repeated
finger measurements. The one-way analyses are retained only as didactic and
exploratory checks.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Statistics
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# Plotting (configured by the notebook; safe defaults here)

from report_text import METHODS_TEXT
from anova_statistics_render import (
    RenderContext,
    dataframe_to_image,
    fatigue_order_interpretation,
    fatigue_order_summary_frame,
    plot_bootstrap_diff,
    plot_contrasts,
    plot_dv_by_finger,
    plot_eight_groups,
    plot_fatigue_order_control,
    plot_pooled_ln_by_finger,
    pooled_ln_descriptives,
    render_oneway_summary_table,
    render_summary_tables,
)

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

STANDARD_VALUE = 8.5  # fixed standard gain S, mm/m (raw device units are 0.1 mm/m)
RANDOM_SEED = 20240613  # fixed seed for ALL stochastic steps (bootstrap, jitter)
N_BOOTSTRAP = 5000  # bootstrap resamples
BOOTSTRAP_CI = 95  # confidence level (%)

SYSTEM_LEVELS = ["L", "N"]
FINGER_LEVELS = ["I", "M", "P", "R"]
FINGER_FULLNAMES = {"I": "Index", "M": "Middle", "P": "Pinky", "R": "Ring"}
DEPENDENT_VARS = ("Bias", "JND")

# Source-column -> analysis-column mapping (documented for transparency).
COLUMN_MAP = {
    "subject_id": "Subject",
    "workspace_setup": "System",
    "finger_condition": "Finger",
    "pse": "PSE",
    "pse_delta_from_standard": "Bias",
    "jnd": "JND",
}

# Live psychophysics results root. This module READS already-computed summaries
# from here; it never recomputes psychometric fits or trial scoring.
PSYCHOPHYSICS_RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "psychophysics", "results",
)
DEFAULT_RESULTS_SELECTION = "L_N_E"
PSE_JND_SUMMARY_FILENAME = "pse_jnd_by_subject_finger.csv"
SUCCESS_SUMMARY_FILENAME = "success_summary_by_subject_finger.csv"

# Backward-compatible default live directory.
PSYCHO_RESULTS_DIR = os.path.join(
    PSYCHOPHYSICS_RESULTS_ROOT, DEFAULT_RESULTS_SELECTION, "_working"
)

# Primary source: the live per-subject x finger PSE/JND summary produced by the
# psychophysics pipeline (NOT a frozen copy). PSE, JND and Bias are read from
# here as-is.
DEFAULT_DATA_PATH = os.path.join(
    PSYCHO_RESULTS_DIR, PSE_JND_SUMMARY_FILENAME
)
SOURCE_DATA_PATH = DEFAULT_DATA_PATH  # provenance == the live file itself

# Legacy frozen copy under data/ - kept ONLY as an offline fallback if the live
# results folder is unavailable (e.g. Dropbox not synced). load_data() warns
# when it has to fall back.
LEGACY_FROZEN_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", PSE_JND_SUMMARY_FILENAME,
)

# A JND larger than this (in mm/m, same scale as the comparisons which span
# 2.5-14.5) is implausible for a discrimination threshold and almost
# always indicates a degenerate / failed psychometric fit. Used ONLY to flag
# rows for the report and the sensitivity analysis, never to silently drop.
JND_EXTREME_THRESHOLD = 10.0

# Lapse rate parameters were fitted within [0, 0.20]. A lapse value within this
# tolerance of either bound indicates the optimiser pinned it at a limit
# (a sign of a poorly constrained / shallow fit).
LAPSE_BOUND_LOW = 0.0
LAPSE_BOUND_HIGH = 0.20
LAPSE_BOUND_TOL = 1e-3


def _ensure_dirs(*paths: str) -> None:
    """Create output folders used by the analysis pipeline."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: str, **kwargs) -> str:
    """Write ``df`` to ``path`` and return the path for call-site reuse."""
    df.to_csv(path, index=False, **kwargs)
    return path


def _dv_slug(dv: str) -> str:
    return dv.lower()


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

def _safe_source_label(source: Optional[str]) -> str:
    """Return a filename-safe cohort/source label for ANOVA outputs."""
    if source is None:
        source = DEFAULT_RESULTS_SELECTION
    text = os.fspath(source)
    # Direct file path -> parent folder label; direct folder path -> folder name.
    if text.lower().endswith(".csv"):
        text = os.path.basename(os.path.dirname(text)) or os.path.basename(text)
        if text == "_working":
            text = os.path.basename(os.path.dirname(os.path.dirname(os.fspath(source))))
    else:
        text = os.path.basename(os.path.normpath(text)) or text
    if text == "_working":
        text = os.path.basename(os.path.dirname(os.path.normpath(os.fspath(source))))
    text = str(text).strip() or "selected"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or "selected"


def selected_source_label(source: Optional[str] = None) -> str:
    """Public helper used by the notebook for the output cohort prefix."""
    return _safe_source_label(source)


def available_psychophysics_result_sources(
    results_root: Optional[str] = None,
    *,
    filename: str = PSE_JND_SUMMARY_FILENAME,
) -> pd.DataFrame:
    """List psychophysics result folders and whether they contain ``filename``."""
    root = os.path.abspath(results_root or PSYCHOPHYSICS_RESULTS_ROOT)
    rows = []
    if not os.path.isdir(root):
        return pd.DataFrame(columns=["selection", "folder", "summary_path", "has_summary"])
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        candidates = _summary_candidates(folder, filename)
        found = next((p for p in candidates if os.path.isfile(p)), "")
        rows.append(
            {
                "selection": name,
                "folder": folder,
                "summary_path": found,
                "has_summary": bool(found),
            }
        )
    return pd.DataFrame(rows)


def _summary_candidates(folder: str, filename: str) -> list[str]:
    """Most likely summary locations inside one psychophysics result folder."""
    return [
        os.path.join(folder, "_working", filename),
        os.path.join(folder, filename),
        os.path.join(folder, "csv", filename),
        os.path.join(folder, "csv", "psychometric_curves", filename),
        os.path.join(folder, "csv", "all", filename),
        os.path.join(folder, "csv", "all", "shared", filename),
    ]


def _find_summary_under(folder: str, filename: str) -> Optional[str]:
    for candidate in _summary_candidates(folder, filename):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    for dirpath, _, filenames in os.walk(folder):
        if filename in filenames:
            return os.path.abspath(os.path.join(dirpath, filename))
    return None


def resolve_psychophysics_summary_path(
    source: Optional[str] = None,
    *,
    filename: str = PSE_JND_SUMMARY_FILENAME,
    results_root: Optional[str] = None,
) -> str:
    """Resolve an ANOVA input source to a concrete psychophysics summary CSV.

    ``source`` can be:
      * ``None`` -> default ``L_N_E`` live summary.
      * a folder name under ``analysis/psychophysics/results`` such as
        ``L_E``, ``N_E``, or ``L_N_E``.
      * a direct results-folder path.
      * a direct CSV path.
    """
    root = os.path.abspath(results_root or PSYCHOPHYSICS_RESULTS_ROOT)
    if source is None:
        return os.path.abspath(DEFAULT_DATA_PATH)

    text = os.path.expanduser(os.fspath(source))
    if os.path.isfile(text):
        return os.path.abspath(text)
    if os.path.isdir(text):
        found = _find_summary_under(text, filename)
        if found:
            return found
        raise FileNotFoundError(
            f"Could not find {filename!r} inside results folder {text!r}."
        )

    # Treat a non-path token as a folder under the psychophysics results root.
    folder = os.path.join(root, text)
    if os.path.isdir(folder):
        found = _find_summary_under(folder, filename)
        if found:
            return found

    # Direct CSV path that does not exist: give a direct error rather than
    # silently falling back to the frozen copy.
    if text.lower().endswith(".csv") or os.path.isabs(text):
        raise FileNotFoundError(f"ANOVA source CSV/folder not found: {text!r}")

    available = available_psychophysics_result_sources(root, filename=filename)
    with_summary = (
        available.loc[available["has_summary"], "selection"].astype(str).tolist()
        if not available.empty
        else []
    )
    raise FileNotFoundError(
        f"Could not find {filename!r} for psychophysics results selection {source!r} "
        f"under {root!r}. Available selections with this CSV: {with_summary}"
    )


def load_data(
    path: Optional[str] = None,
    *,
    results_root: Optional[str] = None,
) -> pd.DataFrame:
    """Load the frozen per-subject x finger summary and standardise columns.

    Returns a DataFrame with canonical columns: Subject, System, Finger, PSE,
    Bias, JND, plus the quality/flag columns and a derived ``Group8`` column
    (System + "_" + Finger). No rows are dropped here.

    ``path`` may be a direct CSV path, a direct psychophysics-results folder, or
    a folder name under ``analysis/psychophysics/results`` (for example
    ``L_E``, ``N_E``, ``L_N_E``).
    """
    if path is None:
        path = resolve_psychophysics_summary_path(None, results_root=results_root)
        if not os.path.exists(path) and os.path.exists(LEGACY_FROZEN_DATA_PATH):
            warnings.warn(
                "Live psychophysics summary not found at "
                f"{path!r}; falling back to the legacy frozen copy at "
                f"{LEGACY_FROZEN_DATA_PATH!r}. Re-sync the psychophysics "
                "results to read the live file.",
                stacklevel=2,
            )
            path = LEGACY_FROZEN_DATA_PATH
    else:
        path = resolve_psychophysics_summary_path(path, results_root=results_root)
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
        l_values = sub.loc[sub["System"] == "L", dv].to_numpy()
        n_values = sub.loc[sub["System"] == "N", dv].to_numpy()
        if len(l_values) < 2 or len(n_values) < 2:
            continue
        # Welch's t-test (does not assume equal variance) via pingouin to get
        # CI and effect size in one call.
        tt = pg.ttest(l_values, n_values, paired=False, correction=True)
        diff = float(np.mean(l_values) - np.mean(n_values))  # L - N
        # pingouin 0.6.x uses underscore column names: p_val, CI95, cohen_d.
        ci_col = "CI95" if "CI95" in tt.columns else "CI95%"
        p_col = "p_val" if "p_val" in tt.columns else "p-val"
        d_col = "cohen_d" if "cohen_d" in tt.columns else "cohen-d"
        ci = tt[ci_col].iloc[0]
        rows.append({
            "DV": dv,
            "Finger": finger,
            "Finger_name": FINGER_FULLNAMES.get(finger, finger),
            "mean_L": float(np.mean(l_values)),
            "sd_L": float(np.std(l_values, ddof=1)),
            "n_L": int(len(l_values)),
            "mean_N": float(np.mean(n_values)),
            "sd_N": float(np.std(n_values, ddof=1)),
            "n_N": int(len(n_values)),
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



# System colours: the marker/circle is filled by SYSTEM (pink = L, light purple = N).
# Per-finger colours: every finger keeps ONE dedicated colour across all figures.
# Sourced from the colocated oneway_anova module so the one-way and two-way plots
# match; falls back to the same literal values if that module is unavailable.

# Display-only: keep JND figures readable on a linear axis by hiding QC-extreme
# JND points from the plotted points/means. Statistical analyses and CSV outputs
# still use the original data; this only affects visualization scale.



















# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #



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


def _render_context() -> RenderContext:
    """Bind rendering helpers to this module's validated analysis functions."""
    return RenderContext(
        system_levels=tuple(SYSTEM_LEVELS),
        finger_levels=tuple(FINGER_LEVELS),
        finger_fullnames=FINGER_FULLNAMES,
        random_seed=RANDOM_SEED,
        jnd_extreme_threshold=JND_EXTREME_THRESHOLD,
        analysis_frame=analysis_frame,
        get_p=get_p,
        fmt_p=_fmt_p,
        ensure_dirs=_ensure_dirs,
        owa=owa,
    )


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
    lines.append(f"Source data:               {results.get('data_path','')}")
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

    for dv in DEPENDENT_VARS:
        lines.append("=" * 72)
        lines.append(f"RESULTS  --  {dv}")
        lines.append("=" * 72)

        # Mixed ANOVA (main).
        ma = results["mixed"][dv]
        aov = ma["aov"]
        eps_gg = np.nan
        if "eps" in aov.columns and aov["eps"].notna().any():
            eps_gg = float(aov["eps"].dropna().iloc[0])
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
            # Greenhouse-Geisser epsilon corrects every effect involving the
            # within factor (Finger and the interaction), so report the
            # epsilon-corrected df next to the uncorrected ones.
            eps_txt = ""
            if src != "System" and isinstance(eps_gg, float) and np.isfinite(eps_gg):
                eps_txt = (f", eps_GG = {eps_gg:.3f}, "
                           f"df_GG = ({float(ddof1) * eps_gg:.2f}, "
                           f"{float(ddof2) * eps_gg:.2f})")
            lines.append(
                f"   {src:<18} F({ddof1},{ddof2}) = "
                f"{f:.3f}, p = {_fmt_p(p)}{gg_txt}{eps_txt}, np2 = {np2:.3f}"
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
            for dv in DEPENDENT_VARS:
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

    lines.append("=" * 72)
    lines.append("THESIS DISCUSSION NOTE: MAIN INTERPRETATION FIGURE")
    lines.append("=" * 72)
    lines.append("Main thesis figure: the L+N pooled visualization by finger.")
    lines.append("   Bias figure: mixed_design/bias/figures/"
                 "<cohort>__pooled_LplusN_by_finger.png")
    lines.append("   JND figure:  mixed_design/jnd/figures/"
                 "<cohort>__pooled_LplusN_by_finger.png")
    lines.append("")
    lines.append("Short interpretation for the thesis discussion:")
    lines.append("   - Exploratory one-way ANOVA: implemented only as a first/"
                 "didactic attempt. It is not relevant for scientific "
                 "interpretation because it does not model Finger as repeated "
                 "within subject and System as a between-subject factor.")
    lines.append("   - Mixed-design ANOVA: relevant confirmatory statistics. "
                 "For the current L+N dataset, it did not find statistically "
                 "significant evidence for a System effect, Finger effect, or "
                 "System x Finger interaction for either Bias or JND. This "
                 "should be stated as no significant evidence for a difference, "
                 "not as proof that systems or fingers are identical.")
    lines.append("   - L+N pooled visualization: main relevant thesis figure. "
                 "L subjects remain pink, N subjects remain purple, the black "
                 "point/CI shows the pooled mean +/- 95% CI, and the "
                 "finger-coloured rectangle shows the pooled median value.")
    lines.append("   - L-N planned contrasts and bootstrap L-N plots: retained "
                 "as supplementary checks, but not the main discussion focus "
                 "because the thesis presentation should emphasize L and N "
                 "together rather than the L-N difference.")
    lines.append("   - Supervisor-approved success-rate filter: relevant only as "
                 "a sensitivity check. Older JND/PSE magnitude filters are not "
                 "relevant and should not be interpreted.")
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





def _gg_text(row) -> str:
    p_gg = row.get("p_GG_corr", np.nan)
    if isinstance(p_gg, (int, float)) and not (isinstance(p_gg, float)
                                               and np.isnan(p_gg)):
        return _fmt_p(p_gg)
    return "-"












# --------------------------------------------------------------------------- #
# (8) Subgroup engine: run the whole pipeline on raw data and on filtered
#     subgroups, mirroring the results/ folder architecture per subgroup.
# --------------------------------------------------------------------------- #
#
# The supervisor-approved subgroup filter EXCLUDES subjects only when at least
# three fingers have low psychophysics success. Filtering is at the SUBJECT level:
# a subject's four finger rows are kept or dropped together, so the repeated-
# measures design stays balanced. PSE/JND magnitude filters were intentionally
# removed because they are not appropriate filtering criteria for this analysis.

# Per-finger "bad" threshold.
SUCCESS_RATE_THRESHOLD = 0.55   # success rate strictly below this is "bad"

# Per-finger success (% correct) source. This is READ from the psychophysics
# pipeline's PRECOMPUTED per-subject x finger success summary - it is NOT
# recomputed here. (The pipeline already scored every trial and aggregated
# success_rate = n_correct / n_trials per subject x finger.)
DEFAULT_SUCCESS_SUMMARY_PATH = os.path.join(
    PSYCHO_RESULTS_DIR, SUCCESS_SUMMARY_FILENAME
)

# (metric_key, human description). Folder name == metric_key.
SUBGROUP_PLAN = [
    ("success_lt55", "Per-finger success rate < 55%"),
]

# (level_key, min bad-finger count to EXCLUDE the subject, label).
SUBGROUP_LEVELS = [
    ("at_least_3_fingers", 3, ">= 3 fingers below 55% success"),
]


def load_success_rates(
    path: Optional[str] = None,
    *,
    results_root: Optional[str] = None,
) -> pd.DataFrame:
    """READ the precomputed per-subject x finger success rate (% correct).

    This does NOT recompute success - it reads the ``success_rate`` column that
    the psychophysics pipeline already produced in
    ``success_summary_by_subject_finger.csv``. Returns columns: Subject, Finger,
    success_rate, n_success_trials.

    The live success summary lives under the (large) psychophysics results tree.
    If that file is not available (e.g. Dropbox not synced) an EMPTY frame with
    the correct columns is returned and a warning is emitted; the ``success_lt55``
    subgroup filter is then skipped by ``run_all_subgroups`` rather than crashing
    the whole pipeline.
    """
    if path is None:
        path = DEFAULT_SUCCESS_SUMMARY_PATH
    else:
        try:
            path = resolve_psychophysics_summary_path(
                path, filename=SUCCESS_SUMMARY_FILENAME, results_root=results_root
            )
        except FileNotFoundError:
            # Preserve forgiving behaviour for success only: missing success data
            # skips the success_lt55 subgroup filter.
            path = os.fspath(path)
    if not os.path.exists(path):
        warnings.warn(
            "Per-finger success summary not found at "
            f"{path!r}; the success_lt55 subgroup filter will be skipped. "
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


# --------------------------------------------------------------------------- #
# Finger-order / fatigue control analysis
# --------------------------------------------------------------------------- #

FATIGUE_ORDER_FILES = {
    "finger_by_appearance_order": os.path.join(
        "finger_time_appearance", "finger_by_appearance_order.csv"),
    "finger_slope_summary": os.path.join(
        "finger_time_appearance", "finger_slope_summary.csv"),
    "finger_slope_contrasts": os.path.join(
        "finger_time_appearance", "finger_slope_contrasts.csv"),
    "group_finger_time_bins": os.path.join(
        "finger_time_appearance", "group_finger_time_bins.csv"),
    "reaction_time_bins": os.path.join(
        "time_fatigue", "reaction_time_bins.csv"),
}


def resolve_psychophysics_results_folder(
    source: Optional[str] = None,
    *,
    results_root: Optional[str] = None,
) -> str:
    """Resolve an ANOVA source to its psychophysics results folder."""
    root = os.path.abspath(results_root or PSYCHOPHYSICS_RESULTS_ROOT)
    if source is None:
        source = DEFAULT_RESULTS_SELECTION
    text = os.path.expanduser(os.fspath(source))
    if os.path.isdir(text):
        return os.path.abspath(text)

    folder = os.path.join(root, text)
    if os.path.isdir(folder):
        return os.path.abspath(folder)

    summary_path = resolve_psychophysics_summary_path(
        source, results_root=results_root)
    if os.path.abspath(summary_path).startswith(root):
        cur = os.path.abspath(os.path.dirname(summary_path))
        while True:
            parent = os.path.dirname(cur)
            if os.path.abspath(parent) == root:
                return cur
            if parent == cur:
                break
            cur = parent
    return os.path.abspath(os.path.dirname(summary_path))


def _find_fatigue_order_csv(folder: str, rel_path: str) -> Optional[str]:
    """Find one fatigue/order CSV in common group or subject layouts."""
    candidates = [
        os.path.join(folder, "csv", "all", "shared", rel_path),
        os.path.join(folder, "csv", rel_path),
        os.path.join(folder, rel_path),
    ]
    filename = os.path.basename(rel_path)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    for dirpath, _, filenames in os.walk(folder):
        if filename in filenames and os.path.basename(os.path.dirname(
                os.path.join(dirpath, filename))) == os.path.basename(
                    os.path.dirname(rel_path)):
            return os.path.abspath(os.path.join(dirpath, filename))
    return None


def load_fatigue_order_data(
    source: Optional[str] = None,
    *,
    results_root: Optional[str] = None,
) -> dict:
    """Read precomputed psychophysics finger-order/time-fatigue CSVs.

    This analysis is intentionally read-only with respect to psychophysics:
    it does not recompute trial success, reaction time, or psychometric fits.
    """
    folder = resolve_psychophysics_results_folder(
        source, results_root=results_root)
    tables = {}
    paths = {}
    missing = []
    for key, rel_path in FATIGUE_ORDER_FILES.items():
        path = _find_fatigue_order_csv(folder, rel_path)
        if path:
            paths[key] = path
            tables[key] = pd.read_csv(path)
        else:
            missing.append(key)
    return {
        "source": source if source is not None else DEFAULT_RESULTS_SELECTION,
        "source_folder": folder,
        "tables": tables,
        "paths": paths,
        "missing": missing,
    }








def run_fatigue_order_control(
    source: Optional[str] = None,
    *,
    base_root: str = "results",
    cohort: Optional[str] = None,
    results_root: Optional[str] = None,
) -> dict:
    """Run the supplementary finger-order/fatigue control section."""
    cohort = cohort or selected_source_label(source)
    out_dir = os.path.join(base_root, "fatigue_order", cohort)
    fatigue = load_fatigue_order_data(source, results_root=results_root)
    status = "ok" if "finger_slope_summary" in fatigue["tables"] else "skipped"
    figures = plot_fatigue_order_control(
        fatigue, out_dir, cohort=cohort, ctx=_render_context())
    interp = fatigue_order_interpretation(fatigue)
    _ensure_dirs(out_dir)
    with open(os.path.join(out_dir, "fatigue_order_interpretation.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(interp + "\n")
        fh.write(f"Source folder: {fatigue['source_folder']}\n")
        if fatigue["missing"]:
            fh.write(f"Missing tables: {', '.join(fatigue['missing'])}\n")
    return {
        "status": status,
        "source": fatigue["source"],
        "source_folder": fatigue["source_folder"],
        "tables": fatigue["tables"],
        "paths": fatigue["paths"],
        "missing": fatigue["missing"],
        "summary": fatigue_order_summary_frame(fatigue),
        "figures": figures,
        "output_dir": out_dir,
        "interpretation": interp,
    }


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
    if metric == "success_lt55":
        if "success_rate" not in df.columns:
            raise KeyError("success_rate column missing; call "
                           "attach_success_rate() first.")
        return df["success_rate"] < SUCCESS_RATE_THRESHOLD
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
    _ensure_dirs(
        paths["reports"],
        paths["tables"],
        *(
            os.path.join(paths[base], sub, leaf)
            for base in ("mixed", "boot")
            for sub in ("bias", "jnd")
            for leaf in ("csv", "figures")
        ),
    )
    return paths


def _pipeline_complete(results: dict) -> bool:
    """True iff every block needed by the report/tables is present for both DVs."""
    for block in ("exploratory", "mixed", "mixed_sensitivity", "contrasts",
                  "bootstrap"):
        for dv in DEPENDENT_VARS:
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
    render_ctx = _render_context()
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
        "exploratory": {}, "pooled_ln": {},
        "mixed": {}, "mixed_sensitivity": {},
        "contrasts": {}, "bootstrap": {}, "notes": notes,
        "n_subjects": int(df["Subject"].nunique()),
        "validation": validation,
    }

    # Didactic first attempt: run the per-system one-way finger analysis before
    # the mixed-design ANOVA. It is retained for documentation/reporting, but the
    # confirmatory model remains the mixed-design ANOVA below.
    try:
        results["oneway"] = run_oneway_flat(
            data_path=results["data_path"],
            base_root=base_root,
        )
    except Exception as exc:
        notes.append(f"oneway skipped: {exc}")

    for dv in DEPENDENT_VARS:
        sub = _dv_slug(dv)

        # Per-DV sub-folders: mixed_design/<bias|jnd>/{csv,figures}; same for
        # bootstrap. The cohort stays a filename prefix (<cohort>__...).
        mixed_csv = os.path.join(P["mixed"], sub, "csv")
        mixed_fig = os.path.join(P["mixed"], sub, "figures")
        boot_csv = os.path.join(P["boot"], sub, "csv")
        boot_fig = os.path.join(P["boot"], sub, "figures")
        _ensure_dirs(mixed_csv, mixed_fig, boot_csv, boot_fig)

        try:
            aov, posthoc = exploratory_one_way(df, dv)
            results["exploratory"][dv] = {"aov": aov, "posthoc": posthoc}
            _write_csv(aov, os.path.join(
                mixed_csv,
                f"{cohort}__exploratory_oneway_anova.csv"))
            _write_csv(posthoc, os.path.join(
                mixed_csv,
                f"{cohort}__exploratory_oneway_posthoc_holm.csv"))
        except Exception as exc:
            notes.append(f"exploratory[{dv}] skipped: {exc}")

        try:
            pooled_desc = pooled_ln_descriptives(df, dv, ctx=render_ctx)
            results["pooled_ln"][dv] = {"desc": pooled_desc}
            _write_csv(pooled_desc, os.path.join(
                mixed_csv, f"{cohort}__pooled_LplusN_descriptives.csv"))
        except Exception as exc:
            notes.append(f"pooled_ln[{dv}] skipped: {exc}")

        try:
            results["mixed"][dv] = mixed_anova(df, dv, drop_flagged=False)
            _write_csv(results["mixed"][dv]["aov"], os.path.join(
                mixed_csv, f"{cohort}__mixed_anova_main.csv"))
        except Exception as exc:
            notes.append(f"mixed[{dv}] skipped: {exc}")

        try:
            results["mixed_sensitivity"][dv] = mixed_anova(
                df, dv, drop_flagged=True)
            _write_csv(results["mixed_sensitivity"][dv]["aov"], os.path.join(
                mixed_csv, f"{cohort}__mixed_anova_sensitivity.csv"))
        except Exception as exc:
            notes.append(f"mixed_sensitivity[{dv}] skipped: {exc}")

        try:
            ct = planned_contrasts(df, dv, drop_flagged=False)
            results["contrasts"][dv] = ct
            _write_csv(ct, os.path.join(
                mixed_csv,
                f"{cohort}__planned_contrasts_L_vs_N.csv"))
        except Exception as exc:
            notes.append(f"contrasts[{dv}] skipped: {exc}")

        try:
            cell_ci, diff_ci = bootstrap_cis(df, dv, n_boot=n_bootstrap,
                                             seed=seed, drop_flagged=False)
            results["bootstrap"][dv] = {"cell": cell_ci, "diff": diff_ci}
            _write_csv(cell_ci, os.path.join(
                boot_csv, f"{cohort}__bootstrap_cell_means.csv"))
            _write_csv(diff_ci, os.path.join(
                boot_csv, f"{cohort}__bootstrap_LminusN_diff.csv"))
        except Exception as exc:
            notes.append(f"bootstrap[{dv}] skipped: {exc}")

        if make_figures:
            try:
                hline = 0.0 if dv == "Bias" else None
                plot_eight_groups(
                    df, dv, dv, f"Exploratory: {dv} by System_Finger group",
                    os.path.join(mixed_fig,
                                 f"{cohort}__exploratory_8group.png"),
                    ctx=render_ctx, hline=hline)
                plot_dv_by_finger(
                    df, dv, hline, dv,
                    f"{dv} by Finger and System (mean +/- 95% CI)",
                    os.path.join(mixed_fig,
                                 f"{cohort}__by_finger_system.png"),
                    ctx=render_ctx)
                plot_pooled_ln_by_finger(
                    df, dv, hline, dv,
                    f"{dv} by Finger, L+N pooled visualization",
                    os.path.join(mixed_fig,
                                 f"{cohort}__pooled_LplusN_by_finger.png"),
                    ctx=render_ctx)
                if dv in results["contrasts"] and len(results["contrasts"][dv]):
                    plot_contrasts(
                        results["contrasts"][dv], dv, dv,
                        f"Planned contrast: {dv} difference (L - N) per finger",
                        os.path.join(mixed_fig,
                                     f"{cohort}__planned_contrast.png"),
                        ctx=render_ctx)
                if dv in results["bootstrap"]:
                    plot_bootstrap_diff(
                        results["bootstrap"][dv]["diff"], dv,
                        os.path.join(boot_fig,
                                     f"{cohort}__bootstrap_LminusN.png"),
                        ctx=render_ctx)
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
                df, results, P["tables"], ctx=render_ctx)
            if "oneway" in results:
                oneway_table = render_oneway_summary_table(
                    results["oneway"], P["tables"], ctx=render_ctx)
                if oneway_table:
                    results["table_images"]["oneway_per_system_summary"] = oneway_table
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






def subgroup_master_frame(rows: list) -> pd.DataFrame:
    """Compact cross-subgroup comparison frame for the master table image."""
    out = pd.DataFrame(rows)
    return out


def _effect_p_row(res: dict) -> dict:
    """Pull System/Finger/Interaction p (Bias & JND) from a results dict."""
    out = {}
    for dv in DEPENDENT_VARS:
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
    _ensure_dirs(base_root)

    # If success rates are unavailable (success_rate all NaN), the success_lt55
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
        if metric == "success_lt55" and not success_available:
            # No success data: record a single skip note for this filter.
            for level, _min, level_label in SUBGROUP_LEVELS:
                cohort = f"{metric}__{level}"
                reports_dir = os.path.join(base_root, "report", cohort)
                _ensure_dirs(reports_dir)
                with open(os.path.join(reports_dir, "SKIPPED.txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write(f"Cohort {cohort} skipped: per-finger success "
                             "summary unavailable (psychophysics results not "
                             "synced); success_lt55 filter cannot be "
                             "evaluated.\n")
                row = {
                    "cohort": cohort, "filter": metric, "filter_desc": desc,
                    "level": level_label, "n_excluded": np.nan,
                    "n_subjects": np.nan, "n_L": np.nan, "n_N": np.nan,
                    "status": "skipped_no_success_data",
                }
                for dv in DEPENDENT_VARS:
                    for eff in ("System", "Finger", "Interaction"):
                        row[f"{dv}_{eff}_p"] = np.nan
                rows.append(row)
                all_results[cohort] = {"status": "skipped_no_success_data"}
            continue
        for level, _min, level_label in SUBGROUP_LEVELS:
            kept, excluded, counts = build_subgroup(work, metric, level)
            cohort = f"{metric}__{level}"
            reports_dir = os.path.join(base_root, "report", cohort)
            _ensure_dirs(reports_dir)

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
                for dv in DEPENDENT_VARS:
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


# --------------------------------------------------------------------------- #
