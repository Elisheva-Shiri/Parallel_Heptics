"""Utilities for 2AFC constant-stimuli psychophysics notebooks.

The functions in this module are intentionally conservative:
- raw data folders are only read;
- all generated files are written to an explicit output folder;
- raw answer codes are converted to a canonical
  ``response_comparison_greater`` after determining which object was the
  comparison stimulus.

Plotting/figure functions live in the sibling module ``twoafc_figures`` and are
re-exported here (see the bottom of this file), so ``import twoafc_psychophysics
as pf; pf.<plot_fn>`` keeps working unchanged.

Psychometric model
------------------
The task is a yes/no-style stiffness comparison: on each trial the participant
judges whether the *comparison* stimulus C is stiffer than the fixed *standard*
S = 85. The fitted binary response is

    y = 1  if the participant judged the comparison C stiffer than the standard S
    y = 0  if the participant judged the standard S stiffer than the comparison C

(stored in ``response_comparison_greater``; it is NOT percent-correct, side,
order, or physical truth -- see ``canonicalize_trials``). Because this is a
yes/no probability rather than a 2AFC percent-correct curve with a fixed guess
rate, the fitted curve is a lapse-aware sigmoid with FOUR free parameters
(mu, scale, lapse_low, lapse_high):

    P(y = 1) = lapse_low + (1 - lapse_low - lapse_high) * F(delta; mu, scale)

with ``delta = C - 85`` and ``F`` a monotonic logistic sigmoid (see
``logistic4``). Two fitters are supported and the chosen one is recorded per fit
in the ``fit_method`` / ``psignifit_status`` columns:
  1. ``psignifit`` -- preferred, used when it is installed;
  2. the custom lapse-aware maximum-likelihood fitter (``fit_with_scipy_logistic``)
     -- the fallback used when psignifit is unavailable.

Derived quantities (the fit is in comparison-stiffness units, so these are
equivalent to the delta-space definitions):
  - PSE  = comparison value where P(y=1) = 0.5  (delta_PSE = PSE - 85);
  - Bias = PSE - 85 = delta at P(y=1) = 0.5  (``pse_delta_from_standard``);
  - JND  = (x75 - x25) / 2, where x25/x75 are the values where the curve reaches
    0.25/0.75. Because of the lapse parameters, 0.25/0.75 are only attainable when
    they lie within ``[lapse_low, 1 - lapse_high]``; when they do not, the fitter
    returns NaN (PSE/JND) and a ``fit_warning`` ("pse_outside_lapse_range" /
    "jnd_quantile_outside_lapse_range") rather than a silently invalid value.
"""

from __future__ import annotations

import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from analysis.group_comparisons import (
        PILOT_ONBOARDING_TRIALS,
        add_experiment_group_columns,
        add_protocol_group_columns,
        add_setup_factor_columns,
        compute_analysis_scope_tables,
        compute_group_comparison_tables,
        compute_protocol_group_comparison_tables,
        compute_setup_factor_tables,
    )
    from analysis.scope_plots import save_scope_summary_plots
except ModuleNotFoundError:  # pragma: no cover - supports running from analysis subfolders
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from analysis.group_comparisons import (
        PILOT_ONBOARDING_TRIALS,
        add_experiment_group_columns,
        add_protocol_group_columns,
        add_setup_factor_columns,
        compute_analysis_scope_tables,
        compute_group_comparison_tables,
        compute_protocol_group_comparison_tables,
        compute_setup_factor_tables,
    )
    from analysis.scope_plots import save_scope_summary_plots


STANDARD_FALLBACK = 85.0
STANDARD_ABS_TOLERANCE = 0.75
MIN_TRIALS_PER_FIT = 12
MIN_LEVELS_PER_FIT = 3
DEFAULT_FIT_BOOTSTRAP_N = 200
MIN_BOOTSTRAP_FOR_CI = 30
DEFAULT_CENTER_X = 320.0
DEFAULT_CENTER_Y = 240.0
STIFFNESS_MAX_FOR_STRETCH_PROXY = 175.0
IGNORED_PATH_PATTERNS = (
    "old",
    "not finish",
    "not_finish",
    "not-finish",
    "unfinished",
    "notfinished",
    "not include",
    "not_include",
    "not-include",
    "notinclude",
)
SUBJECT_FOLDER_RE = re.compile(r"^(?P<setup>[LN])_(?P<protocol>[EP])_(?P<number>\d+)$", re.IGNORECASE)
GROUP_SELECTIONS: dict[str, tuple[str, ...]] = {
    "L_E": ("L_E",),
    "L_P": ("L_P",),
    "N_E": ("N_E",),
    "N_P": ("N_P",),
    # Combined L_E + N_E experiment cohort. Canonical key is ``L_N_E``;
    # ``N+L_E``/``NL_E`` remain as backward-compatible aliases.
    "L_N_E": ("L_E", "N_E"),
    "N+L_E": ("L_E", "N_E"),
    "NL_E": ("L_E", "N_E"),
}
PSYCHOMETRIC_DELTA_AXIS_LABEL = "G_comparison-G_standart"
PSYCHOMETRIC_GREATER_Y_LABEL = "P(choose comparison > standard)"
PSYCHOMETRIC_MEAN_GREATER_Y_LABEL = "Mean P(choose comparison > standard)"
# Psychometric delta x-axis is centred on 0 (comparison == standard) and drawn
# symmetric over +/- this half-range, i.e. the full tested stimulus span
# (comparison 5..165 around the standard 85 -> delta -80..+80).
PSYCHOMETRIC_DELTA_AXIS_LIMIT = 80.0
# Group-analysis PSE validity band (absolute stiffness units). Fits whose PSE
# falls outside this band are kept and flagged in the per-subject tables/curves
# but excluded from every GROUP-level aggregation (band-pass / LPF+HPF on PSE).
PSE_VALID_MIN_ABS = 25.0
PSE_VALID_MAX_ABS = 145.0
# The full table x metric x scope cross-product produced ~768 mostly-uninformative
# scope-summary figures. Only these few headline figures are generated now (as
# (source_table, summary_level, metric) triples). Edit this list to change which
# scope summaries are drawn; see save_experiment_group_comparison_outputs.
SCOPE_SUMMARY_FIGURE_WHITELIST = [
    # PSE bias (comparison - standard) by experiment group, per finger.
    ("psychophysics_fit_by_subject_finger_group_condition_metric_summary", "experiment_group", "pse_delta_from_standard"),
    # JND (discrimination threshold) by experiment group, per finger.
    ("psychophysics_fit_by_subject_finger_group_condition_metric_summary", "experiment_group", "jnd"),
    # Weber fraction (JND / standard) by experiment group, per finger.
    ("psychophysics_fit_by_subject_finger_group_condition_metric_summary", "experiment_group", "weber_fraction"),
    # Accuracy (correct response rate) by experiment group, per finger.
    ("psychophysics_trial_group_condition_metric_summary", "experiment_group", "correct_response"),
    # PSE bias by workspace setup (L vs N), per finger.
    ("psychophysics_fit_by_subject_finger_setup_condition_metric_summary", "setup_factor", "pse_delta_from_standard"),
]
WORKSPACE_SPECS_CM = {
    "N": {"width_cm": 40.0, "height_cm": 50.0, "label": "N workspace (40x50 cm)"},
    "L": {"width_cm": 60.0, "height_cm": 60.0, "label": "L workspace (60x60 cm)"},
}
MOTOR_CONTROL_METHOD_REFERENCES = [
    {
        "citation_key": "Flash_Hogan_1985_minimum_jerk",
        "authors": "Flash T.; Hogan N.",
        "year": 1985,
        "title": "The coordination of arm movements: an experimentally confirmed mathematical model",
        "venue": "The Journal of Neuroscience",
        "doi": "10.1523/JNEUROSCI.05-07-01688.1985",
        "url": "https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985",
        "analysis_use": "Minimum-jerk smoothness, bell-shaped velocity profiles, hand path in external space.",
    },
    {
        "citation_key": "Lacquaniti_Terzuolo_Viviani_1983_power_law",
        "authors": "Lacquaniti F.; Terzuolo C.; Viviani P.",
        "year": 1983,
        "title": "The law relating the kinematic and figural aspects of drawing movements",
        "venue": "Acta Psychologica",
        "doi": "10.1016/0001-6918(83)90027-6",
        "url": "https://doi.org/10.1016/0001-6918(83)90027-6",
        "analysis_use": "Speed-curvature/two-thirds power-law check added to kinematics.",
    },
    {
        "citation_key": "Viviani_Flash_1995_minimum_jerk_power_law",
        "authors": "Viviani P.; Flash T.",
        "year": 1995,
        "title": "Minimum-jerk, two-thirds power law, and isochrony: converging approaches to movement planning",
        "venue": "Journal of Experimental Psychology: Human Perception and Performance",
        "doi": "10.1037/0096-1523.21.1.32",
        "url": "https://doi.org/10.1037/0096-1523.21.1.32",
        "analysis_use": "Connects minimum-jerk and speed-curvature analyses for curved hand paths.",
    },
    {
        "citation_key": "Wichmann_Hill_2001_psychometric_I",
        "authors": "Wichmann F. A.; Hill N. J.",
        "year": 2001,
        "title": "The psychometric function: I. Fitting, sampling, and goodness of fit",
        "venue": "Perception & Psychophysics",
        "doi": "10.3758/BF03194544",
        "url": "https://doi.org/10.3758/BF03194544",
        "analysis_use": "Maximum-likelihood psychometric fits, lapse-rate caution, goodness-of-fit framing.",
    },
    {
        "citation_key": "Wichmann_Hill_2001_psychometric_II",
        "authors": "Wichmann F. A.; Hill N. J.",
        "year": 2001,
        "title": "The psychometric function: II. Bootstrap-based confidence intervals and sampling",
        "venue": "Perception & Psychophysics",
        "doi": "10.3758/BF03194545",
        "url": "https://doi.org/10.3758/BF03194545",
        "analysis_use": "Confidence intervals and sampling reliability for PSE/JND estimates.",
    },
    {
        "citation_key": "Jones_Tan_2013_haptic_psychophysics",
        "authors": "Jones L. A.; Tan H. Z.",
        "year": 2013,
        "title": "Application of psychophysical techniques to haptic research",
        "venue": "IEEE Transactions on Haptics",
        "doi": "10.1109/TOH.2012.74",
        "url": "https://doi.org/10.1109/TOH.2012.74",
        "analysis_use": "2AFC/constant-stimuli haptic threshold logic, Weber fraction, active exploration caveats.",
    },
    {
        "citation_key": "Kim_Avraham_Ivry_2021_psychology_of_reaching",
        "authors": "Kim H. E.; Avraham G.; Ivry R. B.",
        "year": 2021,
        "title": "The psychology of reaching: action selection, movement implementation, and sensorimotor learning",
        "venue": "Annual Review of Psychology",
        "doi": "10.1146/annurev-psych-010419-051053",
        "url": "https://doi.org/10.1146/annurev-psych-010419-051053",
        "analysis_use": "Upper-limb reaching framework: action selection, execution variables, and sensorimotor-learning interpretation.",
    },
    {
        "citation_key": "Prins_Kingdom_2018_model_comparison",
        "authors": "Prins N.; Kingdom F. A. A.",
        "year": 2018,
        "title": "Applying the model-comparison approach to test specific research hypotheses in psychophysical research",
        "venue": "Frontiers in Psychology",
        "doi": "10.3389/fpsyg.2018.01250",
        "url": "https://doi.org/10.3389/fpsyg.2018.01250",
        "analysis_use": "Treat ANOVA-style tables as descriptive screening; prefer planned model comparisons for final hypotheses.",
    },
    {
        "citation_key": "Pressman_Welty_Karniel_MussaIvaldi_2007_delayed_stiffness",
        "authors": "Pressman A.; Welty L. J.; Karniel A.; Mussa-Ivaldi F. A.",
        "year": 2007,
        "title": "Perception of delayed stiffness",
        "venue": "The International Journal of Robotics Research",
        "doi": "10.1177/0278364907082611",
        "url": "https://doi.org/10.1177/0278364907082611",
        "analysis_use": "Forced-choice virtual stiffness perception with psychometric curves.",
    },
    {
        "citation_key": "Nisky_Baraduc_Karniel_2010_proximodistal",
        "authors": "Nisky I.; Baraduc P.; Karniel A.",
        "year": 2010,
        "title": "Proximodistal gradient in the perception of delayed stiffness",
        "venue": "Journal of Neurophysiology",
        "doi": "10.1152/jn.00799.2009",
        "url": "https://doi.org/10.1152/jn.00799.2009",
        "analysis_use": "Finger/joint and haptic-stiffness perception context for group and finger effects.",
    },
]


def sanitize_name(value: Any, fallback: str = "unknown") -> str:
    text = str(value) if value is not None and not pd.isna(value) else fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def validate_paths(data_root: Path, output_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"DATA_ROOT is not a directory: {data_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if is_relative_to(output_root, data_root):
        raise ValueError("OUTPUT_ROOT must not be inside DATA_ROOT. Choose a separate output folder.")


def should_ignore_analysis_path(path: Path) -> bool:
    """Return True for archived or unfinished data paths that must not enter analysis."""
    normalized_parts = [str(part).lower().replace("_", " ").replace("-", " ") for part in path.parts]
    text = " ".join(normalized_parts)
    return any(pattern.replace("_", " ").replace("-", " ") in text for pattern in IGNORED_PATH_PATTERNS)


def canonical_subject_id(value: Any) -> str:
    """Return the canonical ``L_E_14``-style subject id embedded in a folder/name."""
    text = str(value).strip()
    match = re.search(r"([LN])_([EP])_(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return text
    return f"{match.group(1).upper()}_{match.group(2).upper()}_{int(match.group(3))}"


def subject_group_code(subject_id: Any) -> str | None:
    """Return ``L_E``/``L_P``/``N_E``/``N_P`` for canonical subject ids."""
    canonical = canonical_subject_id(subject_id)
    match = SUBJECT_FOLDER_RE.match(canonical)
    if not match:
        return None
    return f"{match.group('setup').upper()}_{match.group('protocol').upper()}"


def is_valid_subject_folder(path: Path) -> bool:
    """Return True for new-architecture subject folders that should enter analysis."""
    return path.is_dir() and SUBJECT_FOLDER_RE.match(canonical_subject_id(path.name)) is not None and not should_ignore_analysis_path(path)


def normalize_data_selection(selection: Any) -> str:
    """Normalize a notebook data-selection flag.

    Valid group flags are ``L_E``, ``L_P``, ``N_E``, ``N_P``, and ``L_N_E``
    (the combined L_E + N_E cohort; ``N+L_E``/``NL_E`` are accepted aliases).
    A concrete subject id such as ``L_E_14`` is also accepted.
    """
    if isinstance(selection, (list, tuple, set)):
        return "+".join(normalize_data_selection(item) for item in selection)
    text = str(selection).strip().upper().replace(" ", "")
    if text in {"L_N_E", "LN_E", "N_L_E", "N+L_E", "N_LE", "NL_E"}:
        return "L_N_E"
    group = subject_group_code(text)
    if group and SUBJECT_FOLDER_RE.match(text):
        return canonical_subject_id(text)
    return text


def selection_label(selection: Any) -> str:
    if isinstance(selection, (list, tuple, set)):
        return sanitize_name("custom_" + "_plus_".join(normalize_data_selection(item) for item in selection))
    return sanitize_name(normalize_data_selection(selection).replace("+", "_plus_"))


def subject_matches_selection(subject_id: Any, selection: Any) -> bool:
    if isinstance(selection, (list, tuple, set)):
        return any(subject_matches_selection(subject_id, item) for item in selection)
    normalized = normalize_data_selection(selection)
    subject = canonical_subject_id(subject_id)
    group = subject_group_code(subject)
    if normalized in GROUP_SELECTIONS:
        return group in GROUP_SELECTIONS[normalized]
    return subject == canonical_subject_id(normalized)


def is_single_subject_selection(selection: Any, subject_ids: list[str] | None = None) -> bool:
    """Return True when the run target is one concrete subject, not a group."""
    if isinstance(selection, (list, tuple, set)):
        return len(selection) == 1 and is_single_subject_selection(next(iter(selection)), subject_ids)
    normalized = normalize_data_selection(selection)
    is_subject = SUBJECT_FOLDER_RE.match(normalized) is not None
    if not is_subject:
        return False
    if subject_ids is None:
        return True
    return len(subject_ids) == 1 and canonical_subject_id(subject_ids[0]) == normalized


def reset_output_root(output_root: Path) -> Path:
    """Delete and recreate a run-specific output folder."""
    output_root = Path(output_root).resolve()
    if output_root == Path(output_root.anchor).resolve() or output_root == Path.home().resolve():
        raise ValueError(f"Refusing to reset unsafe output root: {output_root}")
    if output_root.name.lower() in {"results", "result", "redults"}:
        raise ValueError(
            "Refusing to delete a broad results root directly. "
            "Pass a run-specific child folder such as results/L_E."
        )
    if output_root.exists():
        _rmtree_windows_retry(output_root, attempts=20, delay_seconds=0.35)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def save_csv(df: pd.DataFrame, output_root: Path, filename: str, index: bool = False) -> Path:
    path = output_root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def _mean_ci95_bounds(values: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x.iloc[0]), float(x.iloc[0])
    half = 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    mean = x.mean()
    return float(mean - half), float(mean + half)


def _mean_ci95_lower(values: pd.Series) -> float:
    return _mean_ci95_bounds(values)[0]


def _mean_ci95_upper(values: pd.Series) -> float:
    return _mean_ci95_bounds(values)[1]


def _wilson_ci95_bounds(values: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan, np.nan
    n = float(len(x))
    phat = float(x.mean())
    z = 1.96
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _wilson_ci95_lower(values: pd.Series) -> float:
    return _wilson_ci95_bounds(values)[0]


def _wilson_ci95_upper(values: pd.Series) -> float:
    return _wilson_ci95_bounds(values)[1]


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _categorical_from_first_existing(df: pd.DataFrame, candidates: list[str], index: pd.Index) -> pd.Series:
    col = _first_existing_column(df, candidates)
    if col is None:
        return pd.Series(np.nan, index=index, dtype="object")
    values = df[col].replace("", np.nan)
    return values.astype("object")


def _infer_setup_letter(value: Any) -> str | float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    compact = re.sub(r"[^A-Z0-9]+", "", text)
    if text.startswith("L") or compact.startswith("L") or compact.startswith("P"):
        return "L"
    if text.startswith("N") or compact.startswith("N") or compact.startswith("E"):
        return "N"
    return np.nan


def _infer_workspace_setup(row: pd.Series) -> str | float:
    for col in ["workspace_setup", "experiment_group", "subject_group", "subject_id"]:
        if col in row.index:
            inferred = _infer_setup_letter(row.get(col))
            if not pd.isna(inferred):
                return inferred
    return np.nan


def motor_control_method_references() -> pd.DataFrame:
    """Return citation-ready method references used to guide analysis additions."""
    return pd.DataFrame(MOTOR_CONTROL_METHOD_REFERENCES)


def add_delta_and_less_response_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed-delta and complementary response columns.

    The fitted canonical response remains ``response_comparison_greater`` so the
    logistic fit and plotted psychometric curve are monotonic increasing on
    ``G_comparison-G_standart``. The less-than columns are retained only as
    transparent complements for tabular checks.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    if {"comparison_value", "standard_value"}.issubset(out.columns):
        comparison = pd.to_numeric(out["comparison_value"], errors="coerce")
        standard = pd.to_numeric(out["standard_value"], errors="coerce").fillna(STANDARD_FALLBACK)
        out["standard_value"] = standard
        out["delta_comparison_minus_standard"] = comparison - standard
        out["delta_standard_minus_comparison"] = -out["delta_comparison_minus_standard"]
        out["comparison_over_standard"] = np.where(standard > 0, comparison / standard, np.nan)
        out["signed_delta_over_standard"] = np.where(
            standard > 0,
            out["delta_comparison_minus_standard"] / standard,
            np.nan,
        )
        out["abs_delta_over_standard"] = out["signed_delta_over_standard"].abs()
    if "response_comparison_greater" in out.columns:
        response = pd.to_numeric(out["response_comparison_greater"], errors="coerce")
        out["response_comparison_less"] = np.where(response.notna(), 1.0 - response, np.nan)
    if {"n_trials", "n_comparison_greater"}.issubset(out.columns):
        n_trials = pd.to_numeric(out["n_trials"], errors="coerce")
        n_greater = pd.to_numeric(out["n_comparison_greater"], errors="coerce")
        out["n_comparison_less"] = n_trials - n_greater
    if "p_comparison_greater" in out.columns:
        p_greater = pd.to_numeric(out["p_comparison_greater"], errors="coerce")
        out["p_comparison_less"] = np.where(p_greater.notna(), 1.0 - p_greater, np.nan)
    if {"p_comparison_greater_ci95_lower", "p_comparison_greater_ci95_upper"}.issubset(out.columns):
        lower = pd.to_numeric(out["p_comparison_greater_ci95_lower"], errors="coerce")
        upper = pd.to_numeric(out["p_comparison_greater_ci95_upper"], errors="coerce")
        out["p_comparison_less_ci95_lower"] = 1.0 - upper
        out["p_comparison_less_ci95_upper"] = 1.0 - lower
    return out


def add_psychophysics_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent analysis context without changing raw answer values.

    Stiffness is kept in the original analog units and is additionally expressed
    relative to each trial's standard value for cross-condition comparability.
    Workspace columns document the original L/N workspaces but do not rescale
    the stiffness stimulus because it is not a spatial coordinate.
    """
    if df.empty:
        return df.copy()
    out = add_protocol_group_columns(add_setup_factor_columns(add_experiment_group_columns(df)))
    if {"comparison_value", "standard_value"}.issubset(out.columns):
        comparison = pd.to_numeric(out["comparison_value"], errors="coerce")
        standard = pd.to_numeric(out["standard_value"], errors="coerce").fillna(STANDARD_FALLBACK)
        out["standard_value"] = standard
        out["comparison_over_standard"] = np.where(standard > 0, comparison / standard, np.nan)
        if "signed_stiffness_delta" not in out:
            out["signed_stiffness_delta"] = comparison - standard
        out["delta_comparison_minus_standard"] = out["signed_stiffness_delta"]
        out["delta_standard_minus_comparison"] = -out["signed_stiffness_delta"]
        out["signed_delta_over_standard"] = np.where(standard > 0, out["signed_stiffness_delta"] / standard, np.nan)
        if "abs_stiffness_delta" not in out:
            out["abs_stiffness_delta"] = out["signed_stiffness_delta"].abs()
        out["abs_delta_over_standard"] = np.where(standard > 0, out["abs_stiffness_delta"] / standard, np.nan)
        out["standard_value_reference"] = standard.astype(str) + " analog stiffness units"
    elif "standard_value" in out:
        standard = pd.to_numeric(out["standard_value"], errors="coerce").fillna(STANDARD_FALLBACK)
        out["standard_value"] = standard

    out = add_delta_and_less_response_columns(out)

    if "correct_response" in out.columns:
        success = pd.to_numeric(out["correct_response"], errors="coerce")
        out["success_label"] = np.select(
            [success == 1, success == 0],
            ["success", "failure"],
            default="unknown_success",
        )
    elif "success_label" not in out.columns:
        out["success_label"] = "unknown_success"

    if "workspace_setup" not in out:
        out["workspace_setup"] = out.apply(_infer_workspace_setup, axis=1)
    else:
        inferred = out.apply(_infer_workspace_setup, axis=1)
        out["workspace_setup"] = out["workspace_setup"].where(out["workspace_setup"].notna(), inferred)
    out["workspace_width_cm"] = out["workspace_setup"].map(lambda v: WORKSPACE_SPECS_CM.get(str(v), {}).get("width_cm", np.nan))
    out["workspace_height_cm"] = out["workspace_setup"].map(lambda v: WORKSPACE_SPECS_CM.get(str(v), {}).get("height_cm", np.nan))
    out["workspace_label"] = out["workspace_setup"].map(lambda v: WORKSPACE_SPECS_CM.get(str(v), {}).get("label", "unknown workspace"))

    protocol = _categorical_from_first_existing(
        out,
        ["protocol", "protocol_number", "protocol_id", "protocol_factor", "experiment_protocol"],
        out.index,
    )
    if protocol.isna().all() and "experiment_group" in out:
        suffix = out["experiment_group"].astype(str).str.extract(r"_([A-Z])$", expand=False)
        mapped_suffix = suffix.map({"E": "experiment", "P": "protocol"})
        protocol = pd.Series(np.where(mapped_suffix.notna(), mapped_suffix, suffix), index=out.index, dtype="object")
    out["protocol_factor"] = protocol.fillna("unknown_protocol").astype(str)
    out["sex_factor"] = _categorical_from_first_existing(out, ["sex", "gender", "biological_sex", "sex_factor"], out.index).fillna("unknown_sex").astype(str)
    if "age_years" not in out:
        age_col = _first_existing_column(out, ["age", "participant_age", "subject_age"])
        out["age_years"] = pd.to_numeric(out[age_col], errors="coerce") if age_col else np.nan
    out["age_group"] = pd.cut(
        pd.to_numeric(out["age_years"], errors="coerce"),
        bins=[-np.inf, 18, 25, 35, 50, np.inf],
        labels=["<=18", "19-25", "26-35", "36-50", "51+"],
    ).astype("object").fillna("unknown_age")
    return out


def _add_log_reverse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Back-transform log reaction-time summaries to original seconds."""
    out = df.copy()
    for col in list(out.columns):
        if col.startswith("mean_log_reaction_time"):
            suffix = col.removeprefix("mean_log_reaction_time")
            out[f"geomean_reaction_time{suffix}_s"] = np.exp(out[col])
    if "log_reaction_time_ci95_lower" in out and "log_reaction_time_ci95_upper" in out:
        out["reaction_time_log_ci95_lower_backtransformed_s"] = np.exp(out["log_reaction_time_ci95_lower"])
        out["reaction_time_log_ci95_upper_backtransformed_s"] = np.exp(out["log_reaction_time_ci95_upper"])
    return out


def norm_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = text.replace("answares", "answers").replace("answare", "answer")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def read_csv_flexible(path: Path, *, recover_malformed: bool = False) -> pd.DataFrame:
    errors = []
    # Fast path (default reads only): the experiment answer logs are
    # comma-separated, so try the C parser with an explicit comma before the
    # ``sep=None`` calls below, which force the much slower python engine plus
    # delimiter sniffing. The fast result is accepted only when it parses into
    # more than one column (i.e. the comma really was the delimiter); otherwise
    # we fall through to the original sniffing path unchanged. The
    # malformed-recovery path (used for tracking logs) is left exactly as-is so
    # this optimisation is scoped to the psychophysics answer files, and the
    # golden-output regression check guards against any behavioural drift.
    if not recover_malformed:
        for encoding in ("utf-8-sig", "utf-8", "cp1255", "latin1"):
            try:
                fast = pd.read_csv(path, encoding=encoding, sep=",", engine="c")
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(f"{encoding} c-engine: {exc}")
                continue
            if fast.shape[1] > 1:
                fast.attrs["csv_read_recovered"] = False
                return fast
            break
    for encoding in ("utf-8-sig", "utf-8", "cp1255", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            df.attrs["csv_read_recovered"] = False
            return df
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{encoding}: {exc}")
    if recover_malformed:
        # Some tracking logs can contain a short corrupted/binary block after
        # otherwise valid comma-separated rows.  Keep the conservative default
        # for answer files, but allow tracking analysis to salvage readable rows
        # instead of aborting the notebook on one damaged CSV.
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            try:
                kwargs: dict[str, Any] = {
                    "encoding": encoding,
                    "sep": ",",
                    "engine": "python",
                    "on_bad_lines": "skip",
                }
                if encoding.startswith("utf-8"):
                    kwargs["encoding_errors"] = "replace"
                df = pd.read_csv(path, **kwargs)
                df.attrs["csv_read_recovered"] = True
                return df
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(f"{encoding} recovery: {exc}")
    raise RuntimeError(f"Could not read {path}. Tried encodings: " + " | ".join(errors))


def parse_answer_code(value: Any) -> float:
    """Return 0 for object 1 chosen, 1 for object 2 chosen, NaN otherwise."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value) if float(value) in (0.0, 1.0) else np.nan
    text = norm_name(value)
    if text in {"0", "0_0"}:
        return 0
    if text in {"1", "1_0"}:
        return 1
    if text in {"object_1", "obj_1", "object1", "obj1", "first", "stimulus_1", "stim1", "left"}:
        return 0
    if text in {"object_2", "obj_2", "object2", "obj2", "second", "stimulus_2", "stim2", "right"}:
        return 1
    return np.nan


FINGER_MAP = {
    "i": "I",
    "index": "I",
    "forefinger": "I",
    "pointer": "I",
    "m": "M",
    "middle": "M",
    "r": "R",
    "ring": "R",
    "p": "P",
    "pinky": "P",
    "little": "P",
    "little_finger": "P",
}


def normalize_finger(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = norm_name(value)
    if text in FINGER_MAP:
        return FINGER_MAP[text]
    matches = {mapped for key, mapped in FINGER_MAP.items() if re.search(rf"(^|_){re.escape(key)}($|_)", text)}
    return next(iter(matches)) if len(matches) == 1 else None


def is_answer_csv_candidate(path: Path) -> bool:
    """Return True only for files named exactly ``answers.csv``.

    No misspellings, prefixes, suffixes, case variants, or other answer-like CSV
    names are accepted.
    """
    return path.name == "answers.csv"


def score_answer_csv(path: Path) -> int:
    stem = norm_name(path.stem)
    score = 2 if path.suffix.lower() == ".csv" else 0
    if re.search(r"answer|answers|answare|answares|response|responses|choice|choices", stem):
        score += 20
    if stem in {"answers", "answer", "answares", "answare"}:
        score += 15
    if re.search(r"backup|old|copy|tmp|temp|debug|log", stem):
        score -= 5
    return score


def discover_answer_files(data_root: Path, output_root: Path, selection: Any | None = None) -> pd.DataFrame:
    validate_paths(data_root, output_root)
    rows = []
    subject_dirs = sorted(
        [p for p in data_root.rglob("*") if is_valid_subject_folder(p)],
        key=lambda p: (_subject_sort_key(canonical_subject_id(p.name)), str(p).lower()),
    )
    if selection is not None:
        subject_dirs = [p for p in subject_dirs if subject_matches_selection(p.name, selection)]
    if not subject_dirs:
        scope = f" for selection {selection!r}" if selection is not None else ""
        raise RuntimeError(f"No valid subject folders found under {data_root}{scope}")
    for subject_dir in subject_dirs:
        subject_id = canonical_subject_id(subject_dir.name)
        all_csvs = sorted(
            [p for p in subject_dir.rglob("*.csv") if not should_ignore_analysis_path(p)],
            key=lambda p: str(p).lower(),
        )
        csvs = [p for p in all_csvs if is_answer_csv_candidate(p)]
        scored = sorted([(score_answer_csv(p), p) for p in csvs], key=lambda t: (-t[0], str(t[1]).lower()))
        selected = scored[0][1] if scored else None
        for rank, (score, path) in enumerate(scored, start=1):
            rows.append(
                {
                    "subject_id": subject_dir.name,
                    "canonical_subject_id": subject_id,
                    "subject_group_code": subject_group_code(subject_id),
                    "subject_folder": str(subject_dir),
                    "candidate_rank": rank,
                    "candidate_score": score,
                    "selected": path == selected,
                    "source_file": str(path),
                    "source_file_name": path.name,
                    "n_answer_csv_candidates": len(csvs),
                    "n_total_csv_files_in_subject_folder": len(all_csvs),
                    "selection_warning": "" if score >= 10 else "low answer-file score; verify selection",
                }
            )
        if not scored:
            rows.append(
                {
                    "subject_id": subject_dir.name,
                    "canonical_subject_id": subject_id,
                    "subject_group_code": subject_group_code(subject_id),
                    "subject_folder": str(subject_dir),
                    "candidate_rank": np.nan,
                    "candidate_score": np.nan,
                    "selected": False,
                    "source_file": "",
                    "source_file_name": "",
                    "n_answer_csv_candidates": 0,
                    "n_total_csv_files_in_subject_folder": len(all_csvs),
                    "selection_warning": "no exact answers.csv file found",
                }
            )
    return pd.DataFrame(rows)


def load_selected_subject_csvs(discovery: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in discovery[discovery["selected"]].iterrows():
        path = Path(row["source_file"])
        df = read_csv_flexible(path)
        df.insert(0, "_row_in_source", np.arange(len(df), dtype=int))
        df.insert(0, "_source_file_name", path.name)
        df.insert(0, "_source_file", str(path))
        df.insert(0, "subject_id", row.get("canonical_subject_id", row["subject_id"]))
        df.insert(1, "_subject_folder", row["subject_folder"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


VALUE_TERMS = ["stiffness", "stiff", "value", "force", "intensity", "magnitude", "level", "size", "strength"]
FINGER_TERMS = ["finger", "digit", "condition", "block_finger"]
ANSWER_TERMS = ["answer", "answers", "response", "choice", "chosen", "selected", "selection"]
RT_TERMS = ["time_to_answer", "reaction_time", "response_time", "rt", "latency", "answer_time"]
TRIAL_TERMS = ["trial", "trial_index", "trial_number", "pair_number", "pair", "index"]
TIMESTAMP_TERMS = ["timestamp", "time_stamp", "datetime", "date_time", "date"]
BLOCK_TERMS = ["block", "condition", "finger_condition"]


def contains_any(norm_col: str, terms: list[str]) -> bool:
    return any(term in norm_col for term in terms)


def side_score(norm_col: str, side: int) -> int:
    terms = {
        1: ["object_1", "object1", "obj_1", "obj1", "stimulus_1", "stim1", "stim_1", "first", "left"],
        2: ["object_2", "object2", "obj_2", "obj2", "stimulus_2", "stim2", "stim_2", "second", "right"],
    }[side]
    score = sum(4 if ("object" in t or "obj" in t or "stim" in t) else 2 for t in terms if t in norm_col)
    if re.search(rf"(^|_)o?{side}($|_)", norm_col):
        score += 2
    if re.search(rf"(^|_){side}($|_)", norm_col):
        score += 1
    return score


def pick_best_column(df: pd.DataFrame, scorer, min_score: int = 1) -> tuple[Optional[str], pd.DataFrame]:
    ranked = pd.DataFrame(
        [{"column": col, "normalized": norm_name(col), "score": scorer(norm_name(col), col)} for col in df.columns]
    ).sort_values(["score", "column"], ascending=[False, True])
    best = ranked.iloc[0]
    return (best["column"] if best["score"] >= min_score else None), ranked.reset_index(drop=True)


def detect_columns(df: pd.DataFrame, manual: Optional[dict[str, str]] = None) -> tuple[dict[str, Optional[str]], pd.DataFrame]:
    manual = manual or {}
    detections: dict[str, Optional[str]] = {}
    detail = []

    def set_detected(key: str, scorer, min_score: int) -> None:
        if key in manual and manual[key] in df.columns:
            detections[key] = manual[key]
            detail.append(pd.DataFrame([{"target": key, "column": manual[key], "normalized": norm_name(manual[key]), "score": 999, "manual": True}]))
            return
        col, ranked = pick_best_column(df, scorer, min_score=min_score)
        detections[key] = col
        ranked.insert(0, "target", key)
        ranked["manual"] = False
        detail.append(ranked.head(8))

    def value_scorer(side: int):
        def scorer(n: str, _: str) -> int:
            score = side_score(n, side) * 2
            if contains_any(n, VALUE_TERMS):
                score += 5
            if "stiff" in n:
                score += 4
            if contains_any(n, FINGER_TERMS + ANSWER_TERMS + RT_TERMS + TIMESTAMP_TERMS):
                score -= 8
            return score
        return scorer

    def finger_scorer(side: int):
        def scorer(n: str, _: str) -> int:
            score = side_score(n, side) * 2
            if contains_any(n, FINGER_TERMS):
                score += 6
            if "finger" in n:
                score += 4
            if contains_any(n, VALUE_TERMS + ANSWER_TERMS + RT_TERMS + TIMESTAMP_TERMS):
                score -= 6
            return score
        return scorer

    set_detected("object_1_value", value_scorer(1), 5)
    set_detected("object_2_value", value_scorer(2), 5)
    set_detected("object_1_finger", finger_scorer(1), 5)
    set_detected("object_2_finger", finger_scorer(2), 5)
    set_detected("answer", lambda n, _: (10 if contains_any(n, ANSWER_TERMS) else 0) - (8 if contains_any(n, RT_TERMS + TIMESTAMP_TERMS + FINGER_TERMS) else 0), 5)
    set_detected("reaction_time", lambda n, _: (10 if contains_any(n, RT_TERMS) else 0) + (3 if "time_to_answer" in n else 0) - (8 if contains_any(n, ANSWER_TERMS) and "time_to_answer" not in n else 0), 5)
    set_detected("trial_index", lambda n, _: (8 if contains_any(n, TRIAL_TERMS) else 0) - (5 if contains_any(n, RT_TERMS + ANSWER_TERMS) else 0), 5)
    set_detected("timestamp", lambda n, _: (8 if contains_any(n, TIMESTAMP_TERMS) else 0) - (5 if contains_any(n, RT_TERMS + ANSWER_TERMS) else 0), 5)
    set_detected("block", lambda n, _: (7 if contains_any(n, BLOCK_TERMS) else 0) - (5 if contains_any(n, ["object_1", "object_2", "obj1", "obj2", "answer"]) else 0), 5)
    return detections, pd.concat(detail, ignore_index=True)


PROTOCOL_RE = re.compile(r"\b(stop|pause|break|finger[_\s-]*change|change[_\s-]*finger|switch[_\s-]*finger|marker|calibration|block[_\s-]*(start|end)|experiment[_\s-]*(start|end))\b", re.I)


def row_contains_protocol_marker(row: pd.Series) -> bool:
    text = " ".join(str(v) for v in row.values if not pd.isna(v))
    return bool(PROTOCOL_RE.search(text))


def infer_standard_value(df: pd.DataFrame, cols: dict[str, Optional[str]], fallback: float = STANDARD_FALLBACK) -> tuple[float, pd.DataFrame]:
    c1, c2 = cols.get("object_1_value"), cols.get("object_2_value")
    if not c1 or not c2:
        return fallback, pd.DataFrame([{"candidate_value": fallback, "reason": "fallback_missing_value_columns"}])
    valid = pd.DataFrame({"v1": pd.to_numeric(df[c1], errors="coerce"), "v2": pd.to_numeric(df[c2], errors="coerce")}).dropna()
    if valid.empty:
        return fallback, pd.DataFrame([{"candidate_value": fallback, "reason": "fallback_no_numeric_values"}])
    stacked = pd.concat([valid["v1"], valid["v2"]], ignore_index=True).round(6)
    counts = stacked.value_counts().rename_axis("candidate_value").reset_index(name="count")
    partners: dict[float, set[float]] = defaultdict(set)
    for a, b in valid[["v1", "v2"]].round(6).itertuples(index=False):
        partners[float(a)].add(float(b))
        partners[float(b)].add(float(a))
    counts["n_unique_partners"] = counts["candidate_value"].map(lambda x: len(partners[float(x)]))
    value_range = max(1.0, float(stacked.max() - stacked.min()))
    counts["distance_from_fallback"] = (counts["candidate_value"].astype(float) - fallback).abs()
    counts["standard_score"] = (
        counts["count"] / max(1, counts["count"].max())
        + 0.35 * counts["n_unique_partners"] / max(1, counts["n_unique_partners"].max())
        - 0.10 * counts["distance_from_fallback"] / value_range
    )
    counts = counts.sort_values(["standard_score", "count", "n_unique_partners"], ascending=False).reset_index(drop=True)
    return float(counts.iloc[0]["candidate_value"]), counts


def canonicalize_trials(
    df: pd.DataFrame,
    cols: dict[str, Optional[str]],
    standard_value: float,
    standard_tolerance: float = STANDARD_ABS_TOLERANCE,
    pilot_onboarding_trials: int = PILOT_ONBOARDING_TRIALS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonicalise raw answer rows into one clean dataframe per trial.

    The first ``pilot_onboarding_trials`` clean comparison trials per subject
    are an onboarding/familiarisation phase and are excluded from the returned
    ``clean`` table. They are appended to the ``flagged`` table with
    ``flag_reason='pilot_onboarding_excluded'`` so the audit trail is preserved
    while keeping them out of every downstream summary, fit, and comparison.
    Pass ``pilot_onboarding_trials=0`` to keep all clean trials (useful for
    unit tests and small synthetic fixtures).
    """
    c_v1, c_v2 = cols.get("object_1_value"), cols.get("object_2_value")
    c_f1, c_f2 = cols.get("object_1_finger"), cols.get("object_2_finger")
    c_ans, c_rt = cols.get("answer"), cols.get("reaction_time")
    c_trial, c_time, c_block = cols.get("trial_index"), cols.get("timestamp"), cols.get("block")
    clean_rows, flagged_rows = [], []
    for _, row in df.iterrows():
        flags = []
        protocol = row_contains_protocol_marker(row)
        if protocol:
            flags.append("protocol_marker")
        v1 = pd.to_numeric(pd.Series([row.get(c_v1)]), errors="coerce").iloc[0] if c_v1 else np.nan
        v2 = pd.to_numeric(pd.Series([row.get(c_v2)]), errors="coerce").iloc[0] if c_v2 else np.nan
        answer_code = parse_answer_code(row.get(c_ans)) if c_ans else np.nan
        rt = pd.to_numeric(pd.Series([row.get(c_rt)]), errors="coerce").iloc[0] if c_rt else np.nan
        f1_raw, f2_raw = (row.get(c_f1) if c_f1 else np.nan), (row.get(c_f2) if c_f2 else np.nan)
        f1, f2 = normalize_finger(f1_raw), normalize_finger(f2_raw)
        if pd.isna(v1) or pd.isna(v2):
            flags.append("missing_or_non_numeric_stimulus_value")
        if pd.isna(answer_code):
            flags.append("missing_or_invalid_answer")
        std1 = bool(np.isfinite(v1) and abs(float(v1) - standard_value) <= standard_tolerance)
        std2 = bool(np.isfinite(v2) and abs(float(v2) - standard_value) <= standard_tolerance)
        if std1 == std2:
            flags.append("unexpected_stimulus_configuration_not_exactly_one_standard")
        comparison_value, comparison_side, standard_side = np.nan, None, None
        response, comparison_finger, standard_finger = np.nan, None, None
        if not protocol and not pd.isna(answer_code) and std1 != std2:
            if std1:
                standard_side, comparison_side = "object_1", "object_2"
                comparison_value, standard_finger, comparison_finger = float(v2), f1, f2
                response = int(answer_code == 1)
            else:
                standard_side, comparison_side = "object_2", "object_1"
                comparison_value, standard_finger, comparison_finger = float(v1), f2, f1
                response = int(answer_code == 0)
        finger_warning = ""
        if f1 and f2 and f1 == f2:
            finger_condition = f1
        elif comparison_finger:
            finger_condition = comparison_finger
            if f1 and f2 and f1 != f2:
                finger_warning = "object_finger_mismatch_using_comparison_finger"
        elif f1 or f2:
            finger_condition = f1 or f2
            finger_warning = "single_detected_finger"
        else:
            finger_condition = None
            flags.append("missing_or_unrecognized_finger_condition")
        excluded = bool(flags)
        rec = {
            "subject_id": row.get("subject_id"),
            "source_file": row.get("_source_file"),
            "source_file_name": row.get("_source_file_name"),
            "row_in_source": row.get("_row_in_source"),
            "trial_index_raw": row.get(c_trial) if c_trial else np.nan,
            "timestamp": row.get(c_time) if c_time else np.nan,
            "block_raw": row.get(c_block) if c_block else np.nan,
            "object_1_value": v1,
            "object_2_value": v2,
            "object_1_finger_raw": f1_raw,
            "object_2_finger_raw": f2_raw,
            "object_1_finger": f1,
            "object_2_finger": f2,
            "standard_value": standard_value,
            "standard_side": standard_side,
            "comparison_side": comparison_side,
            "comparison_value": comparison_value,
            "standard_finger": standard_finger,
            "comparison_finger": comparison_finger,
            "finger_condition": finger_condition,
            "raw_answer": row.get(c_ans) if c_ans else np.nan,
            "answer_code": answer_code,
            "response_comparison_greater": response,
            "reaction_time": rt,
            "protocol_marker": protocol,
            "finger_warning": finger_warning,
            "excluded_from_fit": excluded,
            "flag_reason": ";".join(flags),
        }
        if excluded:
            flagged_rows.append(rec)
        else:
            clean_rows.append(rec)
            if finger_warning:
                flagged_rows.append({**rec, "excluded_from_fit": False, "flag_reason": finger_warning})
    clean, flagged = pd.DataFrame(clean_rows), pd.DataFrame(flagged_rows)
    if not clean.empty:
        clean = clean.sort_values(["subject_id", "source_file", "row_in_source"]).reset_index(drop=True)
        clean["global_trial_order"] = clean.groupby("subject_id").cumcount() + 1
        clean["answer_chose_object_2"] = clean["answer_code"].astype(float)
        if pilot_onboarding_trials and pilot_onboarding_trials > 0:
            is_onboarding = clean["global_trial_order"] <= int(pilot_onboarding_trials)
            if is_onboarding.any():
                onboarding = clean.loc[is_onboarding].copy()
                onboarding["excluded_from_fit"] = True
                onboarding["flag_reason"] = "pilot_onboarding_excluded"
                flagged = pd.concat([flagged, onboarding], ignore_index=True, sort=False)
                clean = clean.loc[~is_onboarding].reset_index(drop=True)
    return clean, flagged


def add_success_and_time_columns(clean: pd.DataFrame) -> pd.DataFrame:
    """Add psychophysics outcome columns for correctness, response time, and fatigue/order.

    ``response_comparison_greater`` is the fitted psychometric response. For
    accuracy/success analyses, a trial is correct when the participant reports
    the comparison as stiffer on trials where it is physically stiffer than the
    standard, and reports the standard as stiffer on trials where the comparison
    is physically softer than the standard.

    Fatigue is not directly measured in the raw answers file, so this function
    exposes conservative proxies: trial order, elapsed session time, block order,
    and early/middle/late session thirds.
    """
    if clean.empty:
        return clean.copy()
    out = clean.copy()
    out["signed_stiffness_delta"] = pd.to_numeric(out["comparison_value"], errors="coerce") - pd.to_numeric(out["standard_value"], errors="coerce")
    out["abs_stiffness_delta"] = out["signed_stiffness_delta"].abs()
    out["comparison_physically_greater"] = out["signed_stiffness_delta"] > 0
    out["comparison_physically_equal"] = out["signed_stiffness_delta"].abs() <= STANDARD_ABS_TOLERANCE
    response = pd.to_numeric(out["response_comparison_greater"], errors="coerce")
    correct = np.where(
        out["comparison_physically_equal"],
        np.nan,
        np.where(out["comparison_physically_greater"], response == 1, response == 0),
    )
    out["correct_response"] = pd.Series(correct, index=out.index, dtype="float").astype(float)
    out["incorrect_response"] = 1.0 - out["correct_response"]
    out["subject_group"] = out["subject_id"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    out["subject_group_label"] = out["subject_group"].map({"P": "pilot", "E": "experiment"}).fillna("other")
    out = add_experiment_group_columns(out)

    if "global_trial_order" not in out:
        sort_cols = [c for c in ["subject_id", "source_file", "row_in_source"] if c in out.columns]
        out = out.sort_values(sort_cols).copy() if sort_cols else out.copy()
        out["global_trial_order"] = out.groupby("subject_id").cumcount() + 1
    subject_n = out.groupby("subject_id")["global_trial_order"].transform("max").astype(float)
    out["trial_order_fraction"] = np.where(subject_n > 1, (out["global_trial_order"].astype(float) - 1) / (subject_n - 1), 0.0)

    if "timestamp" in out.columns:
        out["timestamp_dt"] = pd.to_datetime(out["timestamp"], errors="coerce")
    else:
        out["timestamp_dt"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    first_timestamp = out.groupby("subject_id")["timestamp_dt"].transform("min")
    out["elapsed_seconds"] = (out["timestamp_dt"] - first_timestamp).dt.total_seconds()
    out["elapsed_minutes"] = out["elapsed_seconds"] / 60.0
    max_elapsed = out.groupby("subject_id")["elapsed_seconds"].transform("max")
    out["elapsed_fraction"] = np.where(max_elapsed > 0, out["elapsed_seconds"] / max_elapsed, out["trial_order_fraction"])
    out["elapsed_fraction"] = out["elapsed_fraction"].fillna(out["trial_order_fraction"])

    if "block_number_inferred" in out:
        appearance_cols = [
            "finger_first_block_number",
            "finger_first_global_trial_order",
            "finger_first_elapsed_minutes",
            "finger_appearance_order",
        ]
        out = out.drop(columns=[c for c in appearance_cols if c in out.columns])
        max_block = out.groupby("subject_id")["block_number_inferred"].transform("max").astype(float)
        out["block_order_fraction"] = np.where(max_block > 1, (out["block_number_inferred"].astype(float) - 1) / (max_block - 1), 0.0)
        appearance = (
            out.groupby(["subject_id", "finger_condition"], dropna=False)
            .agg(
                finger_first_block_number=("block_number_inferred", "min"),
                finger_first_global_trial_order=("global_trial_order", "min"),
                finger_first_elapsed_minutes=("elapsed_minutes", "min"),
            )
            .reset_index()
            .sort_values(["subject_id", "finger_first_block_number", "finger_first_global_trial_order", "finger_condition"])
        )
        appearance["finger_appearance_order"] = appearance.groupby("subject_id", dropna=False).cumcount() + 1
        out = out.merge(appearance, on=["subject_id", "finger_condition"], how="left")
    else:
        out["block_order_fraction"] = np.nan
        out["finger_first_block_number"] = np.nan
        out["finger_first_global_trial_order"] = np.nan
        out["finger_first_elapsed_minutes"] = np.nan
        out["finger_appearance_order"] = np.nan

    out["session_half"] = np.where(out["trial_order_fraction"] < 0.5, "first_half", "second_half")
    out["fatigue_tertile"] = pd.cut(
        out["trial_order_fraction"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "middle", "late"],
    ).astype(str)
    out["reaction_time"] = pd.to_numeric(out["reaction_time"], errors="coerce")
    out["log_reaction_time"] = np.where(out["reaction_time"] > 0, np.log(out["reaction_time"]), np.nan)
    return add_psychophysics_context_columns(out)


def add_block_numbers(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clean.empty:
        return clean.copy(), pd.DataFrame()
    out = clean.sort_values(["subject_id", "global_trial_order"]).copy()
    block_numbers, block_positions, summaries = {}, {}, []
    for subject, g in out.groupby("subject_id", sort=False):
        last_finger, block_no, within = object(), 0, 0
        for ix, row in g.iterrows():
            finger = row["finger_condition"]
            if finger != last_finger:
                block_no, within = block_no + 1, 1
                summaries.append({"subject_id": subject, "block_number": block_no, "finger_condition": finger, "first_global_trial_order": row["global_trial_order"]})
                last_finger = finger
            else:
                within += 1
            block_numbers[ix], block_positions[ix] = block_no, within
    out["block_number_inferred"] = pd.Series(block_numbers)
    out["trial_in_block_inferred"] = pd.Series(block_positions)
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        counts = out.groupby(["subject_id", "block_number_inferred"]).size().rename("n_trials_in_block").reset_index()
        summary = summary.merge(counts, left_on=["subject_id", "block_number"], right_on=["subject_id", "block_number_inferred"], how="left").drop(columns=["block_number_inferred"])
    return out.reset_index(drop=True), summary


def make_farajian_style_psychometric_input(clean: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Create the 3-column count matrix used in Farajian/Nisky-style analyses.

    The original MATLAB perception scripts arrange each condition as:
    stimulus difference, number of "comparison felt stiffer" responses, and
    total repetitions. This returns the same information with explicit column
    names and both delta directions.
    """
    if clean.empty:
        return pd.DataFrame()
    enriched = add_success_and_time_columns(clean)
    out = (
        enriched.groupby(group_cols + ["signed_stiffness_delta"], dropna=False)
        .agg(
            n_comparison_greater=("response_comparison_greater", "sum"),
            n_comparison_less=("response_comparison_less", "sum"),
            n_trials=("response_comparison_greater", "size"),
            n_data_points=("response_comparison_greater", "size"),
            p_comparison_greater=("response_comparison_greater", "mean"),
            p_comparison_greater_ci95_lower=("response_comparison_greater", _wilson_ci95_lower),
            p_comparison_greater_ci95_upper=("response_comparison_greater", _wilson_ci95_upper),
            p_comparison_less=("response_comparison_less", "mean"),
            p_comparison_less_ci95_lower=("response_comparison_less", _wilson_ci95_lower),
            p_comparison_less_ci95_upper=("response_comparison_less", _wilson_ci95_upper),
            n_correct=("correct_response", "sum"),
            success_rate=("correct_response", "mean"),
            success_rate_ci95_lower=("correct_response", _wilson_ci95_lower),
            success_rate_ci95_upper=("correct_response", _wilson_ci95_upper),
            mean_reaction_time=("reaction_time", "mean"),
            median_reaction_time=("reaction_time", "median"),
            reaction_time_ci95_lower=("reaction_time", _mean_ci95_lower),
            reaction_time_ci95_upper=("reaction_time", _mean_ci95_upper),
            mean_log_reaction_time=("log_reaction_time", "mean"),
            log_reaction_time_ci95_lower=("log_reaction_time", _mean_ci95_lower),
            log_reaction_time_ci95_upper=("log_reaction_time", _mean_ci95_upper),
            standard_value=("standard_value", "median"),
            comparison_value=("comparison_value", "median"),
            comparison_over_standard=("comparison_over_standard", "median"),
            signed_delta_over_standard=("signed_delta_over_standard", "median"),
            abs_delta_over_standard=("abs_delta_over_standard", "median"),
        )
        .reset_index()
        .sort_values(group_cols + ["signed_stiffness_delta"])
    )
    out = add_delta_and_less_response_columns(_add_log_reverse_columns(out))
    out["delta_comparison_minus_standard"] = out["signed_stiffness_delta"]
    out["delta_standard_minus_comparison"] = -out["signed_stiffness_delta"]
    preferred = group_cols + [
        "delta_comparison_minus_standard",
        "delta_standard_minus_comparison",
        "n_comparison_greater",
        "n_comparison_less",
        "n_trials",
        "n_data_points",
        "p_comparison_greater",
        "p_comparison_greater_ci95_lower",
        "p_comparison_greater_ci95_upper",
        "p_comparison_less",
        "p_comparison_less_ci95_lower",
        "p_comparison_less_ci95_upper",
        "n_correct",
        "success_rate",
        "success_rate_ci95_lower",
        "success_rate_ci95_upper",
        "mean_reaction_time",
        "median_reaction_time",
        "reaction_time_ci95_lower",
        "reaction_time_ci95_upper",
        "mean_log_reaction_time",
        "geomean_reaction_time_s",
        "reaction_time_log_ci95_lower_backtransformed_s",
        "reaction_time_log_ci95_upper_backtransformed_s",
        "standard_value",
        "comparison_value",
        "comparison_over_standard",
        "signed_delta_over_standard",
        "abs_delta_over_standard",
    ]
    return out[[c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]]


def make_psychometric_input(clean: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()
    enriched = (
        add_success_and_time_columns(clean)
        if "log_reaction_time" not in clean.columns
        else add_delta_and_less_response_columns(clean.copy())
    )
    out = (
        enriched.groupby(group_cols + ["comparison_value"], dropna=False)
        .agg(
            n_trials=("response_comparison_greater", "size"),
            n_data_points=("response_comparison_greater", "size"),
            n_comparison_greater=("response_comparison_greater", "sum"),
            n_comparison_less=("response_comparison_less", "sum"),
            p_comparison_greater=("response_comparison_greater", "mean"),
            p_comparison_greater_ci95_lower=("response_comparison_greater", _wilson_ci95_lower),
            p_comparison_greater_ci95_upper=("response_comparison_greater", _wilson_ci95_upper),
            p_comparison_less=("response_comparison_less", "mean"),
            p_comparison_less_ci95_lower=("response_comparison_less", _wilson_ci95_lower),
            p_comparison_less_ci95_upper=("response_comparison_less", _wilson_ci95_upper),
            mean_rt=("reaction_time", "mean"),
            median_rt=("reaction_time", "median"),
            rt_ci95_lower=("reaction_time", _mean_ci95_lower),
            rt_ci95_upper=("reaction_time", _mean_ci95_upper),
            mean_log_reaction_time=("log_reaction_time", "mean"),
            log_reaction_time_ci95_lower=("log_reaction_time", _mean_ci95_lower),
            log_reaction_time_ci95_upper=("log_reaction_time", _mean_ci95_upper),
            standard_value=("standard_value", "median"),
            comparison_over_standard=("comparison_over_standard", "median"),
            signed_delta_over_standard=("signed_delta_over_standard", "median"),
            abs_delta_over_standard=("abs_delta_over_standard", "median"),
        )
        .reset_index()
        .sort_values(group_cols + ["comparison_value"])
    )
    out = add_delta_and_less_response_columns(_add_log_reverse_columns(out))
    return out.sort_values(group_cols + ["delta_comparison_minus_standard"]).reset_index(drop=True)


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan, np.nan
    try:
        from scipy.stats import spearmanr

        res = spearmanr(valid["x"], valid["y"])
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def _safe_linregress(x: pd.Series, y: pd.Series) -> dict[str, float]:
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2:
        return {"slope": np.nan, "intercept": np.nan, "r_value": np.nan, "p_value": np.nan, "stderr": np.nan}
    try:
        from scipy.stats import linregress

        res = linregress(valid["x"], valid["y"])
        return {
            "slope": float(res.slope),
            "intercept": float(res.intercept),
            "r_value": float(res.rvalue),
            "p_value": float(res.pvalue),
            "stderr": float(res.stderr),
        }
    except Exception:
        return {"slope": np.nan, "intercept": np.nan, "r_value": np.nan, "p_value": np.nan, "stderr": np.nan}


def _trend_label(slope: float, p_value: float, practical_threshold: float = 0.03) -> str:
    if not np.isfinite(slope):
        return "unknown"
    if abs(slope) < practical_threshold:
        return "flat"
    direction = "up" if slope > 0 else "down"
    if np.isfinite(p_value) and p_value < 0.05:
        return f"{direction}_p_lt_0.05"
    return f"{direction}_weak"


def _add_quantile_bins(g: pd.DataFrame, column: str, n_bins: int, label_col: str) -> pd.DataFrame:
    out = g.copy()
    values = pd.to_numeric(out[column], errors="coerce")
    if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
        out[label_col] = 1
        return out
    bins = min(n_bins, int(values.nunique(dropna=True)))
    try:
        out[label_col] = pd.qcut(values.rank(method="first"), q=bins, labels=False, duplicates="drop") + 1
    except Exception:
        out[label_col] = 1
    return out


def compute_success_time_fatigue(
    clean: pd.DataFrame,
    group_cols: Optional[list[str]] = None,
    n_order_bins: int = 8,
    n_rt_bins: int = 4,
) -> dict[str, pd.DataFrame]:
    """Summarize success-rate effects of reaction duration and fatigue proxies.

    Returns CSV-ready tables for both within-participant trends and
    across-participant summaries. Fatigue is represented by trial order and
    elapsed time because subjective tiredness ratings are not present in the
    answers files.
    """
    group_cols = group_cols or ["subject_id", "finger_condition"]
    trials = add_success_and_time_columns(clean)
    if trials.empty:
        return {
            "trials": trials,
            "summary": pd.DataFrame(),
            "slopes": pd.DataFrame(),
            "reaction_time_bins": pd.DataFrame(),
            "order_bins": pd.DataFrame(),
            "first_second": pd.DataFrame(),
            "between_subjects": pd.DataFrame(),
        }
    trials = trials.sort_values(group_cols + ["global_trial_order"]).copy()
    trials["_within_group_trial_order"] = trials.groupby(group_cols, dropna=False).cumcount() + 1
    within_n = trials.groupby(group_cols, dropna=False)["_within_group_trial_order"].transform("max").astype(float)
    trials["_within_group_order_fraction"] = np.where(
        within_n > 1,
        (trials["_within_group_trial_order"].astype(float) - 1) / (within_n - 1),
        0.0,
    )
    trials["_within_group_half"] = np.where(trials["_within_group_order_fraction"] < 0.5, "first_half", "second_half")

    summary = (
        trials.groupby(group_cols, dropna=False)
        .agg(
            n_trials=("correct_response", "size"),
            n_data_points=("correct_response", "size"),
            n_correct=("correct_response", "sum"),
            success_rate=("correct_response", "mean"),
            success_rate_ci95_lower=("correct_response", _wilson_ci95_lower),
            success_rate_ci95_upper=("correct_response", _wilson_ci95_upper),
            mean_reaction_time=("reaction_time", "mean"),
            median_reaction_time=("reaction_time", "median"),
            reaction_time_ci95_lower=("reaction_time", _mean_ci95_lower),
            reaction_time_ci95_upper=("reaction_time", _mean_ci95_upper),
            mean_log_reaction_time=("log_reaction_time", "mean"),
            median_log_reaction_time=("log_reaction_time", "median"),
            log_reaction_time_ci95_lower=("log_reaction_time", _mean_ci95_lower),
            log_reaction_time_ci95_upper=("log_reaction_time", _mean_ci95_upper),
            session_duration_min=("elapsed_minutes", "max"),
            first_trial_time=("timestamp_dt", "min"),
            last_trial_time=("timestamp_dt", "max"),
            mean_abs_stiffness_delta=("abs_stiffness_delta", "mean"),
            n_stimulus_levels=("comparison_value", "nunique"),
        )
        .reset_index()
    )
    summary = _add_log_reverse_columns(summary)

    slope_rows = []
    for keys, g in trials.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {key: value for key, value in zip(group_cols, keys)}
        order_fit = _safe_linregress(g["_within_group_order_fraction"], g["correct_response"])
        global_order_fit = _safe_linregress(g["trial_order_fraction"], g["correct_response"])
        elapsed_fit = _safe_linregress(g["elapsed_fraction"], g["correct_response"])
        rt_fit = _safe_linregress(g["reaction_time"], g["correct_response"])
        rho_order, p_order = _safe_spearman(g["_within_group_order_fraction"], g["correct_response"])
        rho_global_order, p_global_order = _safe_spearman(g["trial_order_fraction"], g["correct_response"])
        rho_rt, p_rt = _safe_spearman(g["reaction_time"], g["correct_response"])
        row.update(
            {
                "n_trials": int(len(g)),
                "n_data_points": int(len(g)),
                "success_rate": float(g["correct_response"].mean()),
                "success_rate_ci95_lower": _wilson_ci95_lower(g["correct_response"]),
                "success_rate_ci95_upper": _wilson_ci95_upper(g["correct_response"]),
                "median_reaction_time": float(g["reaction_time"].median()) if g["reaction_time"].notna().any() else np.nan,
                "reaction_time_ci95_lower": _mean_ci95_lower(g["reaction_time"]),
                "reaction_time_ci95_upper": _mean_ci95_upper(g["reaction_time"]),
                "mean_log_reaction_time": float(g["log_reaction_time"].mean()) if g["log_reaction_time"].notna().any() else np.nan,
                "log_reaction_time_ci95_lower": _mean_ci95_lower(g["log_reaction_time"]),
                "log_reaction_time_ci95_upper": _mean_ci95_upper(g["log_reaction_time"]),
                "success_vs_order_slope": order_fit["slope"],
                "success_vs_order_p_value": order_fit["p_value"],
                "success_vs_order_spearman_r": rho_order,
                "success_vs_order_spearman_p": p_order,
                "success_vs_order_direction": _trend_label(order_fit["slope"], order_fit["p_value"]),
                "success_vs_global_order_slope": global_order_fit["slope"],
                "success_vs_global_order_p_value": global_order_fit["p_value"],
                "success_vs_global_order_spearman_r": rho_global_order,
                "success_vs_global_order_spearman_p": p_global_order,
                "success_vs_global_order_direction": _trend_label(global_order_fit["slope"], global_order_fit["p_value"]),
                "success_vs_elapsed_slope": elapsed_fit["slope"],
                "success_vs_elapsed_p_value": elapsed_fit["p_value"],
                "success_vs_elapsed_direction": _trend_label(elapsed_fit["slope"], elapsed_fit["p_value"]),
                "success_vs_reaction_time_slope": rt_fit["slope"],
                "success_vs_reaction_time_p_value": rt_fit["p_value"],
                "success_vs_reaction_time_spearman_r": rho_rt,
                "success_vs_reaction_time_spearman_p": p_rt,
                "success_vs_reaction_time_direction": _trend_label(rt_fit["slope"], rt_fit["p_value"]),
            }
        )
        slope_rows.append(row)
    slopes = _add_log_reverse_columns(pd.DataFrame(slope_rows))

    rt_binned = []
    order_binned = []
    for keys, g in trials.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {key: value for key, value in zip(group_cols, keys)}
        gb = _add_quantile_bins(g, "reaction_time", n_rt_bins, "reaction_time_bin")
        for b, bdf in gb.groupby("reaction_time_bin", dropna=False):
            rt_binned.append(
                {
                    **base,
                    "reaction_time_bin": int(b) if pd.notna(b) else np.nan,
                    "n_trials": len(bdf),
                    "n_data_points": len(bdf),
                    "success_rate": float(bdf["correct_response"].mean()),
                    "success_rate_ci95_lower": _wilson_ci95_lower(bdf["correct_response"]),
                    "success_rate_ci95_upper": _wilson_ci95_upper(bdf["correct_response"]),
                    "mean_reaction_time": float(bdf["reaction_time"].mean()) if bdf["reaction_time"].notna().any() else np.nan,
                    "median_reaction_time": float(bdf["reaction_time"].median()) if bdf["reaction_time"].notna().any() else np.nan,
                    "reaction_time_ci95_lower": _mean_ci95_lower(bdf["reaction_time"]),
                    "reaction_time_ci95_upper": _mean_ci95_upper(bdf["reaction_time"]),
                    "mean_log_reaction_time": float(bdf["log_reaction_time"].mean()) if bdf["log_reaction_time"].notna().any() else np.nan,
                    "log_reaction_time_ci95_lower": _mean_ci95_lower(bdf["log_reaction_time"]),
                    "log_reaction_time_ci95_upper": _mean_ci95_upper(bdf["log_reaction_time"]),
                }
            )
        gb = _add_quantile_bins(g, "_within_group_trial_order", n_order_bins, "order_bin")
        for b, bdf in gb.groupby("order_bin", dropna=False):
            order_binned.append(
                {
                    **base,
                    "order_bin": int(b) if pd.notna(b) else np.nan,
                    "n_trials": len(bdf),
                    "n_data_points": len(bdf),
                    "success_rate": float(bdf["correct_response"].mean()),
                    "success_rate_ci95_lower": _wilson_ci95_lower(bdf["correct_response"]),
                    "success_rate_ci95_upper": _wilson_ci95_upper(bdf["correct_response"]),
                    "mean_trial_order_fraction": float(bdf["_within_group_order_fraction"].mean()),
                    "mean_global_trial_order_fraction": float(bdf["trial_order_fraction"].mean()),
                    "mean_elapsed_minutes": float(bdf["elapsed_minutes"].mean()) if bdf["elapsed_minutes"].notna().any() else np.nan,
                    "mean_reaction_time": float(bdf["reaction_time"].mean()) if bdf["reaction_time"].notna().any() else np.nan,
                    "median_reaction_time": float(bdf["reaction_time"].median()) if bdf["reaction_time"].notna().any() else np.nan,
                    "reaction_time_ci95_lower": _mean_ci95_lower(bdf["reaction_time"]),
                    "reaction_time_ci95_upper": _mean_ci95_upper(bdf["reaction_time"]),
                    "mean_log_reaction_time": float(bdf["log_reaction_time"].mean()) if bdf["log_reaction_time"].notna().any() else np.nan,
                    "log_reaction_time_ci95_lower": _mean_ci95_lower(bdf["log_reaction_time"]),
                    "log_reaction_time_ci95_upper": _mean_ci95_upper(bdf["log_reaction_time"]),
                }
            )

    half = (
        trials.groupby(group_cols + ["_within_group_half"], dropna=False)
        .agg(
            n_trials=("correct_response", "size"),
            n_data_points=("correct_response", "size"),
            success_rate=("correct_response", "mean"),
            success_rate_ci95_lower=("correct_response", _wilson_ci95_lower),
            success_rate_ci95_upper=("correct_response", _wilson_ci95_upper),
            mean_reaction_time=("reaction_time", "mean"),
            median_reaction_time=("reaction_time", "median"),
            reaction_time_ci95_lower=("reaction_time", _mean_ci95_lower),
            reaction_time_ci95_upper=("reaction_time", _mean_ci95_upper),
        )
        .reset_index()
    )
    first_second = half.pivot_table(index=group_cols, columns="_within_group_half", values=["success_rate", "mean_reaction_time", "n_trials"], aggfunc="first").reset_index()
    first_second.columns = ["_".join([str(x) for x in col if str(x)]) for col in first_second.columns.to_flat_index()]
    if "success_rate_second_half" in first_second and "success_rate_first_half" in first_second:
        first_second["success_rate_second_minus_first"] = first_second["success_rate_second_half"] - first_second["success_rate_first_half"]
        first_second["fatigue_direction"] = first_second["success_rate_second_minus_first"].map(lambda x: _trend_label(float(x), np.nan, practical_threshold=0.03))
    if "mean_reaction_time_second_half" in first_second and "mean_reaction_time_first_half" in first_second:
        first_second["reaction_time_second_minus_first"] = first_second["mean_reaction_time_second_half"] - first_second["mean_reaction_time_first_half"]

    subject = (
        trials.groupby(["subject_id", "subject_group_label"], dropna=False)
        .agg(
            n_trials=("correct_response", "size"),
            n_data_points=("correct_response", "size"),
            success_rate=("correct_response", "mean"),
            success_rate_ci95_lower=("correct_response", _wilson_ci95_lower),
            success_rate_ci95_upper=("correct_response", _wilson_ci95_upper),
            mean_reaction_time=("reaction_time", "mean"),
            median_reaction_time=("reaction_time", "median"),
            reaction_time_ci95_lower=("reaction_time", _mean_ci95_lower),
            reaction_time_ci95_upper=("reaction_time", _mean_ci95_upper),
            mean_log_reaction_time=("log_reaction_time", "mean"),
            log_reaction_time_ci95_lower=("log_reaction_time", _mean_ci95_lower),
            log_reaction_time_ci95_upper=("log_reaction_time", _mean_ci95_upper),
            session_duration_min=("elapsed_minutes", "max"),
            mean_abs_stiffness_delta=("abs_stiffness_delta", "mean"),
        )
        .reset_index()
    )
    subject = _add_log_reverse_columns(subject)
    between_rows = []
    for x_col in ["mean_reaction_time", "median_reaction_time", "session_duration_min", "mean_abs_stiffness_delta"]:
        fit = _safe_linregress(subject[x_col], subject["success_rate"])
        rho, p = _safe_spearman(subject[x_col], subject["success_rate"])
        between_rows.append(
            {
                "analysis": f"between_subject_success_vs_{x_col}",
                "n_subjects": int(subject[x_col].notna().sum()),
                "slope": fit["slope"],
                "linear_p_value": fit["p_value"],
                "spearman_r": rho,
                "spearman_p": p,
                "direction": _trend_label(fit["slope"], fit["p_value"]),
            }
        )
    between_subjects = pd.DataFrame(between_rows)

    return {
        "trials": trials,
        "summary": summary,
        "slopes": slopes,
        "reaction_time_bins": _add_log_reverse_columns(pd.DataFrame(rt_binned)),
        "order_bins": _add_log_reverse_columns(pd.DataFrame(order_binned)),
        "first_second": first_second,
        "between_subjects": between_subjects,
        "subject_summary": subject,
    }


def _sem(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan


def compare_fingers_over_time_and_appearance(
    clean: pd.DataFrame,
    n_time_bins: int = 8,
) -> dict[str, pd.DataFrame]:
    """Compare success/RT over time by finger and by finger appearance order.

    This analysis separates two related questions:
    1. **Between fingers across time:** do index/middle/ring/pinky have
       different success trajectories as trials progress within that finger?
    2. **Appearance of fingers:** does the first/second/third/fourth finger
       encountered in a session perform differently, regardless of which
       anatomical finger it was?
    """
    trials = add_success_and_time_columns(clean)
    if trials.empty:
        empty = pd.DataFrame()
        return {
            "trials": empty,
            "subject_finger_summary": empty,
            "group_finger_time_bins": empty,
            "subject_finger_time_bins": empty,
            "appearance_order_summary": empty,
            "finger_by_appearance_order": empty,
            "finger_order_matrix": empty,
            "finger_slope_summary": empty,
            "stiffness_slope_summary": empty,
            "finger_slope_contrasts": empty,
            "appearance_order_contrasts": empty,
        }

    trials = trials.sort_values(["subject_id", "finger_condition", "global_trial_order"]).copy()
    trials["within_finger_trial_order"] = trials.groupby(["subject_id", "finger_condition"], dropna=False).cumcount() + 1
    within_n = trials.groupby(["subject_id", "finger_condition"], dropna=False)["within_finger_trial_order"].transform("max").astype(float)
    trials["within_finger_order_fraction"] = np.where(
        within_n > 1,
        (trials["within_finger_trial_order"].astype(float) - 1) / (within_n - 1),
        0.0,
    )

    rows = []
    subject_bins = []
    for (subject, finger), g in trials.groupby(["subject_id", "finger_condition"], dropna=False):
        order_fit = _safe_linregress(g["within_finger_order_fraction"], g["correct_response"])
        rt_fit = _safe_linregress(g["within_finger_order_fraction"], g["reaction_time"])
        rho_success, p_success = _safe_spearman(g["within_finger_order_fraction"], g["correct_response"])
        rows.append(
            {
                "subject_id": subject,
                "finger_condition": finger,
                "finger_appearance_order": int(g["finger_appearance_order"].iloc[0]) if g["finger_appearance_order"].notna().any() else np.nan,
                "finger_first_block_number": float(g["finger_first_block_number"].iloc[0]) if g["finger_first_block_number"].notna().any() else np.nan,
                "finger_first_global_trial_order": float(g["finger_first_global_trial_order"].iloc[0]) if g["finger_first_global_trial_order"].notna().any() else np.nan,
                "finger_first_elapsed_minutes": float(g["finger_first_elapsed_minutes"].iloc[0]) if g["finger_first_elapsed_minutes"].notna().any() else np.nan,
                "n_trials": int(len(g)),
                "n_data_points": int(len(g)),
                "success_rate": float(g["correct_response"].mean()),
                "success_rate_ci95_lower": _wilson_ci95_lower(g["correct_response"]),
                "success_rate_ci95_upper": _wilson_ci95_upper(g["correct_response"]),
                "mean_reaction_time": float(g["reaction_time"].mean()) if g["reaction_time"].notna().any() else np.nan,
                "median_reaction_time": float(g["reaction_time"].median()) if g["reaction_time"].notna().any() else np.nan,
                "reaction_time_ci95_lower": _mean_ci95_lower(g["reaction_time"]),
                "reaction_time_ci95_upper": _mean_ci95_upper(g["reaction_time"]),
                "mean_log_reaction_time": float(g["log_reaction_time"].mean()) if g["log_reaction_time"].notna().any() else np.nan,
                "log_reaction_time_ci95_lower": _mean_ci95_lower(g["log_reaction_time"]),
                "log_reaction_time_ci95_upper": _mean_ci95_upper(g["log_reaction_time"]),
                "success_vs_within_finger_time_slope": order_fit["slope"],
                "success_vs_within_finger_time_p_value": order_fit["p_value"],
                "success_vs_within_finger_time_spearman_r": rho_success,
                "success_vs_within_finger_time_spearman_p": p_success,
                "success_vs_within_finger_time_direction": _trend_label(order_fit["slope"], order_fit["p_value"]),
                "reaction_time_vs_within_finger_time_slope": rt_fit["slope"],
                "reaction_time_vs_within_finger_time_p_value": rt_fit["p_value"],
            }
        )
        # X-axis is the actual trial position within each finger block (1..64),
        # not a coarse quantile bin. Each finger block is 8 stiffness levels x 8
        # repetitions = 64 trials, so the raw within-finger trial order keeps all
        # 64 points and aligns positions across subjects. (``n_time_bins`` is kept
        # for backward compatibility but no longer bins the within-finger axis.)
        gb = g.copy()
        gb["within_finger_time_bin"] = gb["within_finger_trial_order"].astype(int)
        for b, bdf in gb.groupby("within_finger_time_bin", dropna=False):
            subject_bins.append(
                {
                    "subject_id": subject,
                    "finger_condition": finger,
                    "finger_appearance_order": int(g["finger_appearance_order"].iloc[0]) if g["finger_appearance_order"].notna().any() else np.nan,
                    "within_finger_time_bin": int(b) if pd.notna(b) else np.nan,
                    "n_trials": int(len(bdf)),
                    "n_data_points": int(len(bdf)),
                    "success_rate": float(bdf["correct_response"].mean()),
                    "success_rate_ci95_lower": _wilson_ci95_lower(bdf["correct_response"]),
                    "success_rate_ci95_upper": _wilson_ci95_upper(bdf["correct_response"]),
                    "mean_reaction_time": float(bdf["reaction_time"].mean()) if bdf["reaction_time"].notna().any() else np.nan,
                    "median_reaction_time": float(bdf["reaction_time"].median()) if bdf["reaction_time"].notna().any() else np.nan,
                    "reaction_time_ci95_lower": _mean_ci95_lower(bdf["reaction_time"]),
                    "reaction_time_ci95_upper": _mean_ci95_upper(bdf["reaction_time"]),
                    "mean_log_reaction_time": float(bdf["log_reaction_time"].mean()) if bdf["log_reaction_time"].notna().any() else np.nan,
                    "log_reaction_time_ci95_lower": _mean_ci95_lower(bdf["log_reaction_time"]),
                    "log_reaction_time_ci95_upper": _mean_ci95_upper(bdf["log_reaction_time"]),
                    "mean_within_finger_order_fraction": float(bdf["within_finger_order_fraction"].mean()),
                    "mean_global_trial_order_fraction": float(bdf["trial_order_fraction"].mean()),
                }
            )

    subject_finger_summary = pd.DataFrame(rows)
    subject_finger_summary = _add_log_reverse_columns(subject_finger_summary)
    subject_finger_time_bins = _add_log_reverse_columns(pd.DataFrame(subject_bins))
    if not subject_finger_time_bins.empty:
        group_finger_time_bins = (
            subject_finger_time_bins.groupby(["finger_condition", "within_finger_time_bin"], dropna=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                total_trials=("n_trials", "sum"),
                n_data_points=("n_data_points", "sum"),
                mean_success_rate=("success_rate", "mean"),
                median_success_rate=("success_rate", "median"),
                sem_success_rate=("success_rate", _sem),
                success_rate_ci95_lower=("success_rate", _mean_ci95_lower),
                success_rate_ci95_upper=("success_rate", _mean_ci95_upper),
                mean_reaction_time=("mean_reaction_time", "mean"),
                median_reaction_time=("median_reaction_time", "median"),
                sem_reaction_time=("mean_reaction_time", _sem),
                reaction_time_ci95_lower=("mean_reaction_time", _mean_ci95_lower),
                reaction_time_ci95_upper=("mean_reaction_time", _mean_ci95_upper),
                mean_log_reaction_time=("mean_log_reaction_time", "mean"),
                log_reaction_time_ci95_lower=("mean_log_reaction_time", _mean_ci95_lower),
                log_reaction_time_ci95_upper=("mean_log_reaction_time", _mean_ci95_upper),
                mean_within_finger_order_fraction=("mean_within_finger_order_fraction", "mean"),
            )
            .reset_index()
        )
        group_finger_time_bins = _add_log_reverse_columns(group_finger_time_bins)
    else:
        group_finger_time_bins = pd.DataFrame()

    appearance_subject = subject_finger_summary.dropna(subset=["finger_appearance_order"]).copy()
    if not appearance_subject.empty:
        appearance_subject["finger_appearance_order"] = appearance_subject["finger_appearance_order"].astype(int)
        appearance_order_summary = (
            appearance_subject.groupby("finger_appearance_order", dropna=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                fingers_observed=("finger_condition", lambda x: ",".join(sorted(map(str, pd.Series(x).dropna().unique())))),
                mean_success_rate=("success_rate", "mean"),
                median_success_rate=("success_rate", "median"),
                sem_success_rate=("success_rate", _sem),
                success_rate_ci95_lower=("success_rate", _mean_ci95_lower),
                success_rate_ci95_upper=("success_rate", _mean_ci95_upper),
                mean_reaction_time=("mean_reaction_time", "mean"),
                median_reaction_time=("median_reaction_time", "median"),
                sem_reaction_time=("mean_reaction_time", _sem),
                reaction_time_ci95_lower=("mean_reaction_time", _mean_ci95_lower),
                reaction_time_ci95_upper=("mean_reaction_time", _mean_ci95_upper),
                mean_success_slope=("success_vs_within_finger_time_slope", "mean"),
                sem_success_slope=("success_vs_within_finger_time_slope", _sem),
            )
            .reset_index()
        )
        finger_by_appearance_order = (
            appearance_subject.groupby(["finger_appearance_order", "finger_condition"], dropna=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                mean_success_rate=("success_rate", "mean"),
                median_success_rate=("success_rate", "median"),
                sem_success_rate=("success_rate", _sem),
                success_rate_ci95_lower=("success_rate", _mean_ci95_lower),
                success_rate_ci95_upper=("success_rate", _mean_ci95_upper),
                mean_reaction_time=("mean_reaction_time", "mean"),
                median_reaction_time=("median_reaction_time", "median"),
                mean_success_slope=("success_vs_within_finger_time_slope", "mean"),
            )
            .reset_index()
        )
        finger_order_matrix = appearance_subject.pivot_table(
            index="subject_id",
            columns="finger_appearance_order",
            values="finger_condition",
            aggfunc="first",
        ).reset_index()
        finger_order_matrix.columns = ["subject_id"] + [f"appearance_{int(c)}_finger" for c in finger_order_matrix.columns[1:]]
    else:
        appearance_order_summary = pd.DataFrame()
        finger_by_appearance_order = pd.DataFrame()
        finger_order_matrix = pd.DataFrame()

    if not subject_finger_summary.empty:
        finger_slope_summary = (
            subject_finger_summary.groupby("finger_condition", dropna=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                n_data_points=("n_data_points", "sum"),
                mean_success_rate=("success_rate", "mean"),
                median_success_rate=("success_rate", "median"),
                sem_success_rate=("success_rate", _sem),
                success_rate_ci95_lower=("success_rate", _mean_ci95_lower),
                success_rate_ci95_upper=("success_rate", _mean_ci95_upper),
                mean_success_slope=("success_vs_within_finger_time_slope", "mean"),
                median_success_slope=("success_vs_within_finger_time_slope", "median"),
                sem_success_slope=("success_vs_within_finger_time_slope", _sem),
                success_slope_ci95_lower=("success_vs_within_finger_time_slope", _mean_ci95_lower),
                success_slope_ci95_upper=("success_vs_within_finger_time_slope", _mean_ci95_upper),
                mean_reaction_time_slope=("reaction_time_vs_within_finger_time_slope", "mean"),
                median_reaction_time_slope=("reaction_time_vs_within_finger_time_slope", "median"),
                sem_reaction_time_slope=("reaction_time_vs_within_finger_time_slope", _sem),
                reaction_time_slope_ci95_lower=("reaction_time_vs_within_finger_time_slope", _mean_ci95_lower),
                reaction_time_slope_ci95_upper=("reaction_time_vs_within_finger_time_slope", _mean_ci95_upper),
            )
            .reset_index()
        )
    else:
        finger_slope_summary = pd.DataFrame()

    stiffness_slope_rows = []
    if "comparison_value" in trials:
        for (subject, stiffness), g in trials.groupby(["subject_id", "comparison_value"], dropna=False):
            if len(g) < 2:
                continue
            fit = _safe_linregress(g["trial_order_fraction"], g["correct_response"])
            stiffness_slope_rows.append(
                {
                    "subject_id": subject,
                    "comparison_value": stiffness,
                    "n_trials": int(len(g)),
                    "success_rate": float(g["correct_response"].mean()),
                    "success_vs_session_time_slope": fit["slope"],
                    "success_vs_session_time_p_value": fit["p_value"],
                }
            )
    stiffness_slope_subject = pd.DataFrame(stiffness_slope_rows)
    if not stiffness_slope_subject.empty:
        stiffness_slope_summary = (
            stiffness_slope_subject.groupby("comparison_value", dropna=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                n_trials=("n_trials", "sum"),
                mean_success_rate=("success_rate", "mean"),
                mean_success_slope=("success_vs_session_time_slope", "mean"),
                median_success_slope=("success_vs_session_time_slope", "median"),
                sem_success_slope=("success_vs_session_time_slope", _sem),
                success_slope_ci95_lower=("success_vs_session_time_slope", _mean_ci95_lower),
                success_slope_ci95_upper=("success_vs_session_time_slope", _mean_ci95_upper),
            )
            .reset_index()
            .sort_values("comparison_value")
        )
    else:
        stiffness_slope_summary = pd.DataFrame()

    contrast_rows = []
    if not subject_finger_summary.empty:
        wide = subject_finger_summary.pivot(index="subject_id", columns="finger_condition", values="success_vs_within_finger_time_slope")
        fingers = [f for f in wide.columns if pd.notna(f)]
        try:
            from scipy.stats import ttest_rel
        except Exception:
            ttest_rel = None
        for i, f1 in enumerate(fingers):
            for f2 in fingers[i + 1 :]:
                pair = wide[[f1, f2]].dropna()
                diff = pair[f1] - pair[f2] if not pair.empty else pd.Series(dtype=float)
                p_value = np.nan
                t_stat = np.nan
                if ttest_rel is not None and len(pair) >= 2:
                    res = ttest_rel(pair[f1], pair[f2])
                    t_stat, p_value = float(res.statistic), float(res.pvalue)
                contrast_rows.append(
                    {
                        "contrast": f"{f1}_minus_{f2}",
                        "metric": "success_vs_within_finger_time_slope",
                        "n_paired_subjects": int(len(pair)),
                        "mean_difference": float(diff.mean()) if len(diff) else np.nan,
                        "median_difference": float(diff.median()) if len(diff) else np.nan,
                        "sem_difference": _sem(diff),
                        "difference_ci95_lower": _mean_ci95_lower(diff),
                        "difference_ci95_upper": _mean_ci95_upper(diff),
                        "t_stat": t_stat,
                        "p_value": p_value,
                    }
                )
    finger_slope_contrasts = pd.DataFrame(contrast_rows)

    appearance_contrast_rows = []
    if not appearance_subject.empty:
        wide = appearance_subject.pivot(index="subject_id", columns="finger_appearance_order", values="success_rate")
        orders = sorted([int(c) for c in wide.columns if pd.notna(c)])
        try:
            from scipy.stats import ttest_rel
        except Exception:
            ttest_rel = None
        for i, o1 in enumerate(orders):
            for o2 in orders[i + 1 :]:
                pair = wide[[o1, o2]].dropna()
                diff = pair[o1] - pair[o2] if not pair.empty else pd.Series(dtype=float)
                p_value = np.nan
                t_stat = np.nan
                if ttest_rel is not None and len(pair) >= 2:
                    res = ttest_rel(pair[o1], pair[o2])
                    t_stat, p_value = float(res.statistic), float(res.pvalue)
                appearance_contrast_rows.append(
                    {
                        "contrast": f"appearance_{o1}_minus_{o2}",
                        "metric": "success_rate",
                        "n_paired_subjects": int(len(pair)),
                        "mean_difference": float(diff.mean()) if len(diff) else np.nan,
                        "median_difference": float(diff.median()) if len(diff) else np.nan,
                        "sem_difference": _sem(diff),
                        "difference_ci95_lower": _mean_ci95_lower(diff),
                        "difference_ci95_upper": _mean_ci95_upper(diff),
                        "t_stat": t_stat,
                        "p_value": p_value,
                    }
                )
    appearance_order_contrasts = pd.DataFrame(appearance_contrast_rows)

    return {
        "trials": trials,
        "subject_finger_summary": subject_finger_summary,
        "group_finger_time_bins": group_finger_time_bins,
        "subject_finger_time_bins": subject_finger_time_bins,
        "appearance_order_summary": appearance_order_summary,
        "finger_by_appearance_order": finger_by_appearance_order,
        "finger_order_matrix": finger_order_matrix,
        "finger_slope_summary": finger_slope_summary,
        "stiffness_slope_summary": stiffness_slope_summary,
        "finger_slope_contrasts": finger_slope_contrasts,
        "appearance_order_contrasts": appearance_order_contrasts,
    }


def monotonicity_summary(g: pd.DataFrame) -> dict[str, Any]:
    agg = g.groupby("comparison_value")["response_comparison_greater"].agg(["mean", "count"]).reset_index().sort_values("comparison_value")
    if len(agg) < 3:
        return {"monotonic_violations": np.nan, "spearman_r": np.nan, "non_monotonic_flag": False}
    diffs = np.diff(agg["mean"].to_numpy())
    violations = int(np.sum(diffs < -0.20))
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(agg["comparison_value"], agg["mean"]).statistic)
    except Exception:
        rho = np.nan
    return {"monotonic_violations": violations, "spearman_r": rho, "non_monotonic_flag": bool(violations > 0 or (np.isfinite(rho) and rho < 0.3))}


def apparent_error_rate(g: pd.DataFrame) -> float:
    below = g[g["comparison_value"] < g["standard_value"]]
    above = g[g["comparison_value"] > g["standard_value"]]
    errors = []
    if not below.empty:
        errors.append(float(below["response_comparison_greater"].mean()))
    if not above.empty:
        errors.append(float(1 - above["response_comparison_greater"].mean()))
    return float(np.mean(errors)) if errors else np.nan


def rt_outlier_count(g: pd.DataFrame) -> int:
    rt = pd.to_numeric(g.get("reaction_time", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(rt) < 8:
        return 0
    q1, q3 = np.percentile(rt, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0
    return int(((rt < q1 - 3 * iqr) | (rt > q3 + 3 * iqr)).sum())


def make_qc_summary(clean: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, finger), g in clean.groupby(["subject_id", "finger_condition"], dropna=False):
        mono = monotonicity_summary(g)
        counts_per_value = g.groupby("comparison_value").size()
        err = apparent_error_rate(g)
        side_bias = float(g["answer_chose_object_2"].mean()) if "answer_chose_object_2" in g else np.nan
        warnings = []
        if len(g) < MIN_TRIALS_PER_FIT:
            warnings.append("few_trials")
        if g["comparison_value"].nunique() < MIN_LEVELS_PER_FIT:
            warnings.append("few_stimulus_levels")
        if len(counts_per_value) and counts_per_value.min() < 2:
            warnings.append("some_levels_have_lt2_trials")
        if mono["non_monotonic_flag"]:
            warnings.append("non_monotonic_curve")
        if np.isfinite(err) and err > 0.35:
            warnings.append("high_apparent_error_rate")
        if np.isfinite(side_bias) and abs(side_bias - 0.5) > 0.25:
            warnings.append("side_bias")
        rows.append(
            {
                "subject_id": subject,
                "finger_condition": finger,
                "n_clean_trials": len(g),
                "n_stimulus_levels": g["comparison_value"].nunique(),
                "min_trials_per_value": int(counts_per_value.min()) if len(counts_per_value) else 0,
                "median_trials_per_value": float(counts_per_value.median()) if len(counts_per_value) else np.nan,
                "side_bias_p_chose_object2": side_bias,
                "standard_side_p_object2": float((g["standard_side"] == "object_2").mean()),
                "apparent_error_rate": err,
                "rt_outlier_count": rt_outlier_count(g),
                **mono,
                "qc_warnings": ";".join(warnings),
            }
        )
    qc = pd.DataFrame(rows)
    if not qc.empty and not flagged.empty:
        qc = qc.merge(flagged.groupby("subject_id").size().rename("n_flagged_rows").reset_index(), on="subject_id", how="left")
    if not qc.empty:
        if "n_flagged_rows" not in qc.columns:
            qc["n_flagged_rows"] = 0
        else:
            qc["n_flagged_rows"] = qc["n_flagged_rows"].fillna(0).astype(int)
    return qc


def check_psignifit_available() -> tuple[bool, str]:
    try:
        import psignifit  # type: ignore
        return True, f"psignifit import succeeded (version={getattr(psignifit, '__version__', 'unknown')})"
    except Exception as exc:
        return False, str(exc)


def logistic4(x: np.ndarray, mu: float, scale: float, lapse_low: float, lapse_high: float) -> np.ndarray:
    """Lapse-aware yes/no psychometric function (4 parameters).

    ``P(y=1) = lapse_low + (1 - lapse_low - lapse_high) * sigmoid((x - mu)/scale)``.
    ``mu`` is the 50%-of-the-sigmoid location, ``scale`` the spread, and
    ``lapse_low``/``lapse_high`` the lower/upper asymptote lapses. This is a yes/no
    model (P of judging the comparison stiffer), not a 2AFC percent-correct curve
    with a fixed 0.5 guess rate.
    """
    z = (np.asarray(x, dtype=float) - mu) / max(scale, 1e-12)
    s = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    return lapse_low + (1 - lapse_low - lapse_high) * s


def x_at_probability(q: float, mu: float, scale: float, lapse_low: float, lapse_high: float) -> float:
    """Invert ``logistic4``: stimulus value where P(y=1) == ``q``.

    Returns NaN when ``q`` is not attainable, i.e. outside the curve's reachable
    range ``[lapse_low, 1 - lapse_high]`` (so PSE at q=0.5 and the x25/x75 used for
    JND are flagged as non-estimable rather than reported as invalid).
    """
    amp = 1 - lapse_low - lapse_high
    if amp <= 0:
        return np.nan
    y = (q - lapse_low) / amp
    if y <= 0 or y >= 1:
        return np.nan
    return float(mu + scale * math.log(y / (1 - y)))


def _psignifit_param_ci(result: Any, key_candidates: tuple[str, ...]) -> tuple[float, float]:
    """Try to read a 95% CI from a psignifit result dict for a parameter."""
    if not isinstance(result, dict):
        return np.nan, np.nan
    for key in key_candidates:
        if key in result:
            arr = np.asarray(result[key])
            if arr.size == 0:
                continue
            try:
                flat = np.ravel(arr)
                if flat.size >= 2 and np.all(np.isfinite(flat[:2])):
                    return float(flat[0]), float(flat[1])
            except Exception:
                continue
    return np.nan, np.nan


def fit_with_psignifit_if_possible(agg: pd.DataFrame, psignifit_available: bool) -> tuple[Optional[dict[str, Any]], str]:
    if not psignifit_available:
        return None, "psignifit_not_installed"
    try:
        import psignifit  # type: ignore
        data = agg[["comparison_value", "n_comparison_greater", "n_trials"]].to_numpy(dtype=float)
        if hasattr(psignifit, "psignifit"):
            result = psignifit.psignifit(data, {"sigmoidName": "logistic", "expType": "YesNo", "threshPC": 0.5})  # type: ignore[attr-defined]
            pse = np.nan
            x25 = np.nan
            x75 = np.nan
            pse_ci_lower = np.nan
            pse_ci_upper = np.nan
            x25_ci_lower = np.nan
            x25_ci_upper = np.nan
            x75_ci_lower = np.nan
            x75_ci_upper = np.nan
            if hasattr(psignifit, "getThreshold"):
                try:
                    pse_pair = psignifit.getThreshold(result, 0.5)  # type: ignore[attr-defined]
                    pse_arr = np.ravel(pse_pair)
                    pse = float(pse_arr[0])
                    if pse_arr.size >= 3:
                        pse_ci_lower = float(pse_arr[1])
                        pse_ci_upper = float(pse_arr[2])
                except Exception:
                    pass
                try:
                    x25_pair = psignifit.getThreshold(result, 0.25)  # type: ignore[attr-defined]
                    x25_arr = np.ravel(x25_pair)
                    x25 = float(x25_arr[0])
                    if x25_arr.size >= 3:
                        x25_ci_lower = float(x25_arr[1])
                        x25_ci_upper = float(x25_arr[2])
                except Exception:
                    pass
                try:
                    x75_pair = psignifit.getThreshold(result, 0.75)  # type: ignore[attr-defined]
                    x75_arr = np.ravel(x75_pair)
                    x75 = float(x75_arr[0])
                    if x75_arr.size >= 3:
                        x75_ci_lower = float(x75_arr[1])
                        x75_ci_upper = float(x75_arr[2])
                except Exception:
                    pass
            lapse = np.nan
            if isinstance(result, dict) and "Fit" in result:
                fit = np.ravel(result["Fit"])
                if np.isnan(pse) and len(fit) >= 1:
                    pse = float(fit[0])
                if len(fit) >= 3:
                    lapse = float(fit[2])
            if np.isnan(pse_ci_lower) or np.isnan(pse_ci_upper):
                pse_ci_lower, pse_ci_upper = _psignifit_param_ci(
                    result, ("conf_Intervals_threshold", "conf_Intervals", "confIntervals", "CI", "ci"),
                )
            if np.isfinite(pse):
                jnd = float((x75 - x25) / 2) if np.isfinite(x25) and np.isfinite(x75) else np.nan
                jnd_ci_lower = np.nan
                jnd_ci_upper = np.nan
                if all(np.isfinite([x25_ci_lower, x25_ci_upper, x75_ci_lower, x75_ci_upper])):
                    jnd_ci_lower = float((x75_ci_lower - x25_ci_upper) / 2)
                    jnd_ci_upper = float((x75_ci_upper - x25_ci_lower) / 2)
                pse_se = float((pse_ci_upper - pse_ci_lower) / (2 * 1.96)) if np.isfinite(pse_ci_lower) and np.isfinite(pse_ci_upper) else np.nan
                jnd_se = float((jnd_ci_upper - jnd_ci_lower) / (2 * 1.96)) if np.isfinite(jnd_ci_lower) and np.isfinite(jnd_ci_upper) else np.nan
                return {
                    "fit_method": "psignifit",
                    "pse": pse,
                    "jnd": jnd,
                    "x25": x25,
                    "x75": x75,
                    "lapse_rate": lapse,
                    "pse_ci95_lower": pse_ci_lower,
                    "pse_ci95_upper": pse_ci_upper,
                    "pse_se": pse_se,
                    "jnd_ci95_lower": jnd_ci_lower,
                    "jnd_ci95_upper": jnd_ci_upper,
                    "jnd_se": jnd_se,
                    "bootstrap_method": "psignifit_bayesian_ci",
                    "fit_warning": "psignifit_api_threshold_extraction_partial" if not (np.isfinite(x25) and np.isfinite(x75)) else "",
                    "psignifit_result_repr": repr(result)[:1000],
                }, "psignifit_success"
        return None, "psignifit_api_not_recognized"
    except Exception as exc:
        return None, f"psignifit_failed: {exc}"


def _logistic_nll(params: np.ndarray, x: np.ndarray, k: np.ndarray, n: np.ndarray, eps: float = 1e-9) -> float:
    mu, log_scale, lapse_low, lapse_high = params
    scale = float(np.exp(log_scale))
    lapse_low = float(lapse_low)
    lapse_high = float(lapse_high)
    if lapse_low + lapse_high >= 0.45:
        return 1e9
    p = np.clip(logistic4(x, float(mu), scale, lapse_low, lapse_high), eps, 1 - eps)
    return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def _fit_logistic_mle_from_starts(
    x: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    starts: list[list[float]],
    bounds: list[tuple[float, float]],
    eps: float = 1e-9,
) -> Optional[Any]:
    """Run scipy L-BFGS-B from multiple starts and return the best OptimizeResult."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return None
    best = None
    for start in starts:
        result = minimize(_logistic_nll, start, args=(x, k, n, eps), method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    return best


def _parametric_bootstrap_logistic_pse_jnd(
    x: np.ndarray,
    n: np.ndarray,
    mle_params: tuple[float, float, float, float],
    starts: list[list[float]],
    bounds: list[tuple[float, float]],
    *,
    n_bootstrap: int = DEFAULT_FIT_BOOTSTRAP_N,
    seed: int = 12345,
    eps: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Parametric binomial bootstrap of PSE / JND / lapse for a 4-parameter logistic.

    For each iteration the per-level success counts are resampled from
    ``Binomial(n_i, p_hat_i)`` where ``p_hat_i`` comes from the MLE.  The fit is
    re-run warm-started from the MLE and the resulting PSE/JND values are
    accumulated.  Returns NaN-padded arrays so callers can compute SEs and
    percentile CIs.
    """
    rng = np.random.default_rng(seed)
    mu_hat, scale_hat, lapse_low_hat, lapse_high_hat = mle_params
    p_hat = np.clip(logistic4(x, mu_hat, scale_hat, lapse_low_hat, lapse_high_hat), eps, 1 - eps)
    n_int = np.maximum(n.astype(int), 0)
    warm_start = [float(mu_hat), float(math.log(max(scale_hat, 1e-9))), float(lapse_low_hat), float(lapse_high_hat)]
    boot_starts = [warm_start] + list(starts)[:1]
    pse_samples = np.full(n_bootstrap, np.nan, dtype=float)
    jnd_samples = np.full(n_bootstrap, np.nan, dtype=float)
    lapse_samples = np.full(n_bootstrap, np.nan, dtype=float)
    if n_int.sum() <= 0 or n_bootstrap <= 0:
        return {"pse": pse_samples, "jnd": jnd_samples, "lapse_rate": lapse_samples}
    for b in range(n_bootstrap):
        k_b = rng.binomial(n_int, p_hat).astype(float)
        res = _fit_logistic_mle_from_starts(x, k_b, n.astype(float), boot_starts, bounds, eps)
        if res is None:
            continue
        mu_b = float(res.x[0])
        scale_b = float(np.exp(res.x[1]))
        lapse_low_b = float(res.x[2])
        lapse_high_b = float(res.x[3])
        pse_b = x_at_probability(0.5, mu_b, scale_b, lapse_low_b, lapse_high_b)
        x25_b = x_at_probability(0.25, mu_b, scale_b, lapse_low_b, lapse_high_b)
        x75_b = x_at_probability(0.75, mu_b, scale_b, lapse_low_b, lapse_high_b)
        pse_samples[b] = pse_b
        if np.isfinite(x25_b) and np.isfinite(x75_b):
            jnd_samples[b] = (x75_b - x25_b) / 2.0
        lapse_samples[b] = lapse_low_b + lapse_high_b
    return {"pse": pse_samples, "jnd": jnd_samples, "lapse_rate": lapse_samples}


def _bootstrap_summary(samples: np.ndarray) -> dict[str, float]:
    """Return SE and 95% percentile CI summary for a 1-D bootstrap array."""
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < MIN_BOOTSTRAP_FOR_CI:
        return {"se": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan, "n_effective": int(values.size)}
    return {
        "se": float(values.std(ddof=1)),
        "ci95_lower": float(np.percentile(values, 2.5)),
        "ci95_upper": float(np.percentile(values, 97.5)),
        "n_effective": int(values.size),
    }


def fit_with_scipy_logistic(agg: pd.DataFrame, n_bootstrap: int = DEFAULT_FIT_BOOTSTRAP_N) -> dict[str, Any]:
    x = agg["comparison_value"].to_numpy(dtype=float)
    k = agg["n_comparison_greater"].to_numpy(dtype=float)
    n = agg["n_trials"].to_numpy(dtype=float)
    out = {
        "fit_method": "scipy_logistic_fallback",
        "n_trials": int(n.sum()),
        "n_stimulus_levels": int(len(x)),
        "pse": np.nan,
        "jnd": np.nan,
        "x25": np.nan,
        "x75": np.nan,
        "slope_at_pse": np.nan,
        "lapse_low": np.nan,
        "lapse_high": np.nan,
        "lapse_rate": np.nan,
        "neg_log_likelihood": np.nan,
        "deviance": np.nan,
        "aic": np.nan,
        "pse_se": np.nan,
        "pse_ci95_lower": np.nan,
        "pse_ci95_upper": np.nan,
        "jnd_se": np.nan,
        "jnd_ci95_lower": np.nan,
        "jnd_ci95_upper": np.nan,
        "lapse_rate_se": np.nan,
        "lapse_rate_ci95_lower": np.nan,
        "lapse_rate_ci95_upper": np.nan,
        "n_bootstrap": 0,
        "n_bootstrap_effective_pse": 0,
        "n_bootstrap_effective_jnd": 0,
        "bootstrap_method": "parametric_binomial",
        "fit_quality": "not_fit",
        "fit_warning": "",
    }
    warnings = []
    if out["n_trials"] < MIN_TRIALS_PER_FIT:
        warnings.append("few_trials")
    if out["n_stimulus_levels"] < MIN_LEVELS_PER_FIT:
        warnings.append("few_stimulus_levels")
    if len(x) < 2 or n.sum() < 2:
        out["fit_warning"] = ";".join(warnings + ["insufficient_data"])
        return out
    try:
        import scipy.optimize  # noqa: F401
    except Exception as exc:
        out["fit_warning"] = ";".join(warnings + [f"scipy_unavailable: {exc}"])
        out["fit_quality"] = "failed"
        return out
    eps = 1e-9
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_range = max(float(x_max - x_min), 1.0)
    y_obs = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    mu_guess = float(x[int(np.nanargmin(np.abs(y_obs - 0.5)))]) if np.all(np.isfinite(y_obs)) else float(np.median(x))

    bounds = [
        (x_min - x_range, x_max + x_range),
        (math.log(max(x_range / 200, 1e-3)), math.log(max(x_range * 10, 1.0))),
        (0.0, 0.20),
        (0.0, 0.20),
    ]
    starts = [
        [mu_guess, math.log(max(x_range / 6, 1e-3)), 0.02, 0.02],
        [np.median(x), math.log(max(x_range / 4, 1e-3)), 0.01, 0.01],
        [STANDARD_FALLBACK, math.log(max(x_range / 5, 1e-3)), 0.03, 0.03],
        [mu_guess, math.log(max(x_range / 10, 1e-3)), 0.05, 0.05],
    ]
    best = _fit_logistic_mle_from_starts(x, k, n, starts, bounds, eps)
    if best is None or not best.success:
        warnings.append("optimizer_failed" if best is None else f"optimizer_warning:{best.message}")
    mu = float(best.x[0])
    scale = float(np.exp(best.x[1]))
    lapse_low = float(best.x[2])
    lapse_high = float(best.x[3])
    pse = x_at_probability(0.5, mu, scale, lapse_low, lapse_high)
    x25 = x_at_probability(0.25, mu, scale, lapse_low, lapse_high)
    x75 = x_at_probability(0.75, mu, scale, lapse_low, lapse_high)
    amp = 1 - lapse_low - lapse_high
    y_pse = (0.5 - lapse_low) / amp if amp > 0 else np.nan
    slope = amp * y_pse * (1 - y_pse) / scale if np.isfinite(y_pse) and scale > 0 else np.nan
    p_sat = np.clip(y_obs, eps, 1 - eps)
    saturated_nll = -float(np.sum(k * np.log(p_sat) + (n - k) * np.log(1 - p_sat)))
    nll_value = _logistic_nll(best.x, x, k, n, eps)
    if not np.isfinite(pse):
        warnings.append("pse_outside_lapse_range")
    if not np.isfinite(x25) or not np.isfinite(x75):
        warnings.append("jnd_quantile_outside_lapse_range")
    if scale > 5 * x_range:
        warnings.append("very_shallow_fit")
    if lapse_low > 0.15 or lapse_high > 0.15:
        warnings.append("high_lapse_estimate")

    bootstrap_warnings: list[str] = []
    if n_bootstrap and n_bootstrap > 0:
        try:
            samples = _parametric_bootstrap_logistic_pse_jnd(
                x, n, (mu, scale, lapse_low, lapse_high), starts, bounds,
                n_bootstrap=n_bootstrap, eps=eps,
            )
            pse_summary = _bootstrap_summary(samples["pse"])
            jnd_summary = _bootstrap_summary(samples["jnd"])
            lapse_summary = _bootstrap_summary(samples["lapse_rate"])
            out["pse_se"] = pse_summary["se"]
            out["pse_ci95_lower"] = pse_summary["ci95_lower"]
            out["pse_ci95_upper"] = pse_summary["ci95_upper"]
            out["jnd_se"] = jnd_summary["se"]
            out["jnd_ci95_lower"] = jnd_summary["ci95_lower"]
            out["jnd_ci95_upper"] = jnd_summary["ci95_upper"]
            out["lapse_rate_se"] = lapse_summary["se"]
            out["lapse_rate_ci95_lower"] = lapse_summary["ci95_lower"]
            out["lapse_rate_ci95_upper"] = lapse_summary["ci95_upper"]
            out["n_bootstrap"] = int(n_bootstrap)
            out["n_bootstrap_effective_pse"] = int(pse_summary["n_effective"])
            out["n_bootstrap_effective_jnd"] = int(jnd_summary["n_effective"])
            if pse_summary["n_effective"] < MIN_BOOTSTRAP_FOR_CI:
                bootstrap_warnings.append("bootstrap_pse_unstable")
            if jnd_summary["n_effective"] < MIN_BOOTSTRAP_FOR_CI:
                bootstrap_warnings.append("bootstrap_jnd_unstable")
        except Exception as exc:  # pragma: no cover - bootstrap is best-effort
            bootstrap_warnings.append(f"bootstrap_failed: {exc}")

    out.update(
        {
            "mu": mu,
            "scale": scale,
            "lapse_low": lapse_low,
            "lapse_high": lapse_high,
            "lapse_rate": lapse_low + lapse_high,
            "pse": pse,
            "x25": x25,
            "x75": x75,
            "jnd": float((x75 - x25) / 2) if np.isfinite(x25) and np.isfinite(x75) else np.nan,
            "slope_at_pse": slope,
            "neg_log_likelihood": nll_value,
            "deviance": max(0.0, 2 * (nll_value - saturated_nll)),
            "aic": 8 + 2 * nll_value,
            "fit_quality": "ok" if not warnings else "warning",
            "fit_warning": ";".join(warnings + bootstrap_warnings),
        }
    )
    return out


def fit_one_condition(
    agg: pd.DataFrame,
    psignifit_available: bool,
    n_bootstrap: int = DEFAULT_FIT_BOOTSTRAP_N,
) -> dict[str, Any]:
    scipy_fit = fit_with_scipy_logistic(agg, n_bootstrap=n_bootstrap)
    psig_fit, psig_status = fit_with_psignifit_if_possible(agg, psignifit_available)
    if psig_fit is not None and np.isfinite(psig_fit.get("pse", np.nan)):
        out = {**scipy_fit, **psig_fit}
        for key in ["mu", "scale", "lapse_low", "lapse_high"]:
            out[f"scipy_{key}"] = scipy_fit.get(key, np.nan)
        for ci_key in ("pse_se", "pse_ci95_lower", "pse_ci95_upper", "jnd_se", "jnd_ci95_lower", "jnd_ci95_upper"):
            scipy_value = scipy_fit.get(ci_key, np.nan)
            if np.isfinite(scipy_value):
                out[f"scipy_{ci_key}"] = scipy_value
        out["psignifit_status"] = psig_status
        return out
    scipy_fit["psignifit_status"] = psig_status
    return scipy_fit


_FIT_PREFERRED_TAIL = [
    "fit_method",
    "pse",
    "pse_se",
    "pse_ci95_lower",
    "pse_ci95_upper",
    "pse_delta_from_standard",
    "pse_delta_ci95_lower",
    "pse_delta_ci95_upper",
    "standard_inside_pse_ci95",
    "pse_bias_p_value",
    "jnd",
    "jnd_se",
    "jnd_ci95_lower",
    "jnd_ci95_upper",
    "weber_fraction",
    "jnd_over_standard",
    "x25",
    "x75",
    "slope_at_pse",
    "lapse_rate",
    "lapse_rate_ci95_lower",
    "lapse_rate_ci95_upper",
    "lapse_low",
    "lapse_high",
    "fit_quality",
    "fit_warning",
    "n_trials",
    "n_stimulus_levels",
    "n_bootstrap",
    "bootstrap_method",
    "deviance",
    "aic",
    "psignifit_status",
]


def _resolve_fit_n_jobs(n_items: int, n_jobs: Optional[int]) -> int:
    """Number of worker processes for per-condition fits.

    Bounded by the number of items and by the available cores. ``None`` (the
    default) auto-selects ``cpu_count - 1``; pass ``1`` to force serial.
    """
    if n_items <= 1:
        return 1
    if n_jobs is not None:
        return max(1, min(int(n_jobs), n_items))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu - 1, n_items))


def _fit_one_group(
    keys: Any,
    g: pd.DataFrame,
    group_cols: list[str],
    psignifit_available: bool,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Fit a single grouped condition and attach grouping/summary columns.

    Pure function of its arguments: the bootstrap RNG is seeded with a fixed
    constant inside ``_parametric_bootstrap_logistic_pse_jnd``, so the result is
    identical whether this runs serially or inside a parallel worker process.
    """
    if not isinstance(keys, tuple):
        keys = (keys,)
    fit = fit_one_condition(g, psignifit_available, n_bootstrap=n_bootstrap)
    for key, value in zip(group_cols, keys):
        fit[key] = value
    fit["n_trials"] = int(g["n_trials"].sum())
    fit["n_stimulus_levels"] = int(g["comparison_value"].nunique())
    if "standard_value" in g.columns and g["standard_value"].notna().any():
        fit["standard_value"] = float(pd.to_numeric(g["standard_value"], errors="coerce").median())
    fit["comparison_min"] = float(g["comparison_value"].min())
    fit["comparison_max"] = float(g["comparison_value"].max())
    return fit


def _fit_flat_item(item: tuple[int, Any, pd.DataFrame, list[str]], psignifit_available: bool, n_bootstrap: int):
    job_index, keys, g, group_cols = item
    return job_index, _fit_one_group(keys, g, group_cols, psignifit_available, n_bootstrap)


def _assemble_fit_table(rows: list[dict[str, Any]], group_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = add_fit_delta_columns(out)
    preferred = group_cols + _FIT_PREFERRED_TAIL
    return out[[c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]]


def fit_conditions(
    agg: pd.DataFrame,
    group_cols: list[str],
    psignifit_available: bool = False,
    *,
    n_bootstrap: int = DEFAULT_FIT_BOOTSTRAP_N,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Fit a psychometric curve per group.

    Groups are independent and each fit is deterministic (fixed-seed bootstrap),
    so the per-group fits are distributed across worker processes when more than
    one group is present. Output (values, row order, column order) is identical
    to the previous serial implementation; pass ``n_jobs=1`` to force serial.
    """
    groups = list(agg.groupby(group_cols, dropna=False))
    if not groups:
        return pd.DataFrame()
    n_jobs_eff = _resolve_fit_n_jobs(len(groups), n_jobs)
    if n_jobs_eff <= 1:
        rows = [_fit_one_group(keys, g, group_cols, psignifit_available, n_bootstrap) for keys, g in groups]
    else:
        try:
            from joblib import Parallel, delayed

            rows = Parallel(n_jobs=n_jobs_eff)(
                delayed(_fit_one_group)(keys, g, group_cols, psignifit_available, n_bootstrap)
                for keys, g in groups
            )
        except Exception:  # pragma: no cover - joblib missing/unavailable -> serial
            rows = [_fit_one_group(keys, g, group_cols, psignifit_available, n_bootstrap) for keys, g in groups]
    return _assemble_fit_table(rows, group_cols)


def fit_conditions_many(
    jobs: list[tuple[pd.DataFrame, list[str]]],
    psignifit_available: bool = False,
    *,
    n_bootstrap: int = DEFAULT_FIT_BOOTSTRAP_N,
    n_jobs: Optional[int] = None,
) -> list[pd.DataFrame]:
    """Fit several ``(agg, group_cols)`` jobs sharing one flat worker pool.

    Each returned frame is identical to ``fit_conditions(agg, group_cols, ...)``
    for the matching job, but every condition from every job is fitted in a
    single pool so cores stay busy even when individual jobs have few groups
    (e.g. the finger-pooled and all-pooled fits). Group order within and across
    jobs is preserved, so each output matches the per-call version exactly.
    """
    flat: list[tuple[int, Any, pd.DataFrame, list[str]]] = []
    for job_index, (agg, group_cols) in enumerate(jobs):
        for keys, g in agg.groupby(group_cols, dropna=False):
            flat.append((job_index, keys, g, group_cols))
    if not flat:
        return [pd.DataFrame() for _ in jobs]
    n_jobs_eff = _resolve_fit_n_jobs(len(flat), n_jobs)
    if n_jobs_eff <= 1:
        fitted = [_fit_flat_item(item, psignifit_available, n_bootstrap) for item in flat]
    else:
        try:
            from joblib import Parallel, delayed

            fitted = Parallel(n_jobs=n_jobs_eff)(
                delayed(_fit_flat_item)(item, psignifit_available, n_bootstrap) for item in flat
            )
        except Exception:  # pragma: no cover - joblib missing/unavailable -> serial
            fitted = [_fit_flat_item(item, psignifit_available, n_bootstrap) for item in flat]
    rows_by_job: list[list[dict[str, Any]]] = [[] for _ in jobs]
    for job_index, fit in fitted:
        rows_by_job[job_index].append(fit)
    return [
        _assemble_fit_table(rows_by_job[i], jobs[i][1]) if rows_by_job[i] else pd.DataFrame()
        for i in range(len(jobs))
    ]


def subject_average_psychometric(clean: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    subj = make_psychometric_input(clean, ["subject_id"] + group_cols)
    if subj.empty:
        return subj
    out = (
        subj.groupby(group_cols + ["comparison_value"], dropna=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_data_points=("n_trials", "sum"),
            mean_p_comparison_greater=("p_comparison_greater", "mean"),
            mean_p_comparison_less=("p_comparison_less", "mean"),
            median_p_comparison_greater=("p_comparison_greater", "median"),
            median_p_comparison_less=("p_comparison_less", "median"),
            sem_p_comparison_greater=("p_comparison_greater", lambda x: float(pd.Series(x).std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan),
            sem_p_comparison_less=("p_comparison_less", lambda x: float(pd.Series(x).std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan),
            p_comparison_greater_ci95_lower=("p_comparison_greater", _mean_ci95_lower),
            p_comparison_greater_ci95_upper=("p_comparison_greater", _mean_ci95_upper),
            p_comparison_less_ci95_lower=("p_comparison_less", _mean_ci95_lower),
            p_comparison_less_ci95_upper=("p_comparison_less", _mean_ci95_upper),
            total_trials=("n_trials", "sum"),
            median_rt=("median_rt", "median"),
            mean_rt=("mean_rt", "mean"),
            rt_ci95_lower=("mean_rt", _mean_ci95_lower),
            rt_ci95_upper=("mean_rt", _mean_ci95_upper),
            mean_log_reaction_time=("mean_log_reaction_time", "mean"),
            log_reaction_time_ci95_lower=("mean_log_reaction_time", _mean_ci95_lower),
            log_reaction_time_ci95_upper=("mean_log_reaction_time", _mean_ci95_upper),
            standard_value=("standard_value", "median"),
            comparison_over_standard=("comparison_over_standard", "median"),
            signed_delta_over_standard=("signed_delta_over_standard", "median"),
            abs_delta_over_standard=("abs_delta_over_standard", "median"),
        )
        .reset_index()
        .sort_values(group_cols + ["comparison_value"])
    )
    out = add_delta_and_less_response_columns(_add_log_reverse_columns(out))
    return out.sort_values(group_cols + ["delta_comparison_minus_standard"]).reset_index(drop=True)


def compute_order_effects(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "log_reaction_time" not in clean.columns:
        clean = add_success_and_time_columns(clean)
    rows, binned_rows = [], []
    for (subject, finger), g in clean.groupby(["subject_id", "finger_condition"], dropna=False):
        g = g.sort_values("global_trial_order").copy()
        mid = len(g) // 2
        first, second = g.iloc[:mid], g.iloc[mid:]
        try:
            from scipy.stats import spearmanr
            rho_resp = float(spearmanr(g["global_trial_order"], g["response_comparison_greater"]).statistic) if len(g) > 2 else np.nan
            rho_rt = float(spearmanr(g["global_trial_order"], g["reaction_time"], nan_policy="omit").statistic) if g["reaction_time"].notna().sum() > 2 else np.nan
        except Exception:
            rho_resp, rho_rt = np.nan, np.nan
        rows.append(
            {
                "subject_id": subject,
                "finger_condition": finger,
                "n_trials": len(g),
                "n_data_points": len(g),
                "first_half_p_comparison_greater": float(first["response_comparison_greater"].mean()) if len(first) else np.nan,
                "second_half_p_comparison_greater": float(second["response_comparison_greater"].mean()) if len(second) else np.nan,
                "first_half_mean_rt": float(first["reaction_time"].mean()) if first["reaction_time"].notna().any() else np.nan,
                "second_half_mean_rt": float(second["reaction_time"].mean()) if second["reaction_time"].notna().any() else np.nan,
                "median_rt": float(g["reaction_time"].median()) if g["reaction_time"].notna().any() else np.nan,
                "rt_ci95_lower": _mean_ci95_lower(g["reaction_time"]),
                "rt_ci95_upper": _mean_ci95_upper(g["reaction_time"]),
                "mean_log_reaction_time": float(g["log_reaction_time"].mean()) if g["log_reaction_time"].notna().any() else np.nan,
                "median_log_reaction_time": float(g["log_reaction_time"].median()) if g["log_reaction_time"].notna().any() else np.nan,
                "log_reaction_time_ci95_lower": _mean_ci95_lower(g["log_reaction_time"]),
                "log_reaction_time_ci95_upper": _mean_ci95_upper(g["log_reaction_time"]),
                "spearman_response_vs_order": rho_resp,
                "spearman_rt_vs_order": rho_rt,
                "first_block_number": int(g["block_number_inferred"].min()) if "block_number_inferred" in g else np.nan,
            }
        )
        bins = min(10, max(2, len(g) // 8))
        try:
            g["order_bin"] = pd.qcut(g["global_trial_order"], q=bins, labels=False, duplicates="drop") + 1
        except Exception:
            g["order_bin"] = 1
        for b, gb in g.groupby("order_bin"):
            binned_rows.append(
                {
                    "subject_id": subject,
                    "finger_condition": finger,
                    "order_bin": int(b),
                    "mean_global_trial_order": float(gb["global_trial_order"].mean()),
                    "p_comparison_greater": float(gb["response_comparison_greater"].mean()),
                    "p_comparison_greater_ci95_lower": _wilson_ci95_lower(gb["response_comparison_greater"]),
                    "p_comparison_greater_ci95_upper": _wilson_ci95_upper(gb["response_comparison_greater"]),
                    "mean_rt": float(gb["reaction_time"].mean()) if gb["reaction_time"].notna().any() else np.nan,
                    "median_rt": float(gb["reaction_time"].median()) if gb["reaction_time"].notna().any() else np.nan,
                    "rt_ci95_lower": _mean_ci95_lower(gb["reaction_time"]),
                    "rt_ci95_upper": _mean_ci95_upper(gb["reaction_time"]),
                    "mean_log_reaction_time": float(gb["log_reaction_time"].mean()) if gb["log_reaction_time"].notna().any() else np.nan,
                    "median_log_reaction_time": float(gb["log_reaction_time"].median()) if gb["log_reaction_time"].notna().any() else np.nan,
                    "log_reaction_time_ci95_lower": _mean_ci95_lower(gb["log_reaction_time"]),
                    "log_reaction_time_ci95_upper": _mean_ci95_upper(gb["log_reaction_time"]),
                    "n_trials": len(gb),
                    "n_data_points": len(gb),
                }
            )
    return _add_log_reverse_columns(pd.DataFrame(rows)), _add_log_reverse_columns(pd.DataFrame(binned_rows))


def predictions_from_fit_row(row: pd.Series, x_grid: np.ndarray) -> np.ndarray:
    mu, scale = row.get("mu", np.nan), row.get("scale", np.nan)
    lo, hi = row.get("lapse_low", np.nan), row.get("lapse_high", np.nan)
    if not np.isfinite(mu):
        mu, scale = row.get("scipy_mu", np.nan), row.get("scipy_scale", np.nan)
        lo, hi = row.get("scipy_lapse_low", np.nan), row.get("scipy_lapse_high", np.nan)
    if not all(np.isfinite(v) for v in [mu, scale, lo, hi]):
        return np.full_like(x_grid, np.nan, dtype=float)
    return logistic4(x_grid, float(mu), float(scale), float(lo), float(hi))


FINGER_ORDER = ["I", "M", "R", "P"]
FINGER_APPEARANCE_LABELS = {1: "first", 2: "second", 3: "third", 4: "fourth"}
STIFFNESS_CMAP = "viridis"
# Side-bias plots split each finger into the two presented objects: object 1 (the
# first object in the data) on the left in orange, object 2 (the second) on the
# right in blue.
OBJECT1_COLOR = "#ff7f0e"  # orange  -> first object  (left column)
OBJECT2_COLOR = "#1f77b4"  # blue    -> second object (right column)
FINGER_STYLE = {
    "I": {"label": "Index", "color": "#1f77b4", "marker": "o"},   # blue
    "M": {"label": "Middle", "color": "#ff7f0e", "marker": "s"},  # orange
    "R": {"label": "Ring", "color": "#d62728", "marker": "D"},    # red
    "P": {"label": "Pinky", "color": "#2ca02c", "marker": "^"},   # green
}


def _subject_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value)
    m = re.match(r"([A-Za-z]+)(\d+)$", text)
    if not m:
        return (9, 999, text)
    prefix, number = m.group(1).upper(), int(m.group(2))
    return (0 if prefix == "P" else 1 if prefix == "E" else 2, number, text)


def add_fit_delta_columns(fits: pd.DataFrame) -> pd.DataFrame:
    """Add PSE/JND columns expressed as comparison-standard deltas.

    Also propagates the per-fit standard error / 95% CI on the PSE to the
    delta-from-standard scale and adds a ``standard_inside_pse_ci95`` boolean
    plus a Wald-style ``pse_bias_p_value`` so callers can immediately tell
    whether each fit's PSE is statistically distinguishable from the standard.
    """
    if fits.empty:
        return fits.copy()
    out = add_psychophysics_context_columns(fits)
    if "standard_value" not in out:
        out["standard_value"] = STANDARD_FALLBACK
    standard = pd.to_numeric(out["standard_value"], errors="coerce").fillna(STANDARD_FALLBACK)
    out["standard_value"] = standard
    if "pse" in out:
        pse = pd.to_numeric(out["pse"], errors="coerce")
        out["pse_delta_comparison_minus_standard"] = pse - standard
        out["pse_delta_standard_minus_comparison"] = -out["pse_delta_comparison_minus_standard"]
        out["pse_delta_from_standard"] = out["pse_delta_comparison_minus_standard"]
        out["abs_pse_delta_from_standard"] = out["pse_delta_from_standard"].abs()
        # Band-pass (LPF+HPF) on the PSE: a fit is valid for GROUP analysis only
        # when its PSE sits inside the reliable stimulus band [25, 145]. Out-of-band
        # (or un-estimated) fits are KEPT here and drawn individually but flagged so
        # compute_experiment_group_comparisons can drop them from group aggregation.
        out["pse_in_valid_band"] = (pse >= PSE_VALID_MIN_ABS) & (pse <= PSE_VALID_MAX_ABS)
        out["excluded_from_group_analysis"] = ~out["pse_in_valid_band"]
        out["group_exclusion_reason"] = np.where(
            out["pse_in_valid_band"],
            "",
            np.where(pse.isna(), "pse_not_estimated", f"pse_outside_{PSE_VALID_MIN_ABS:g}-{PSE_VALID_MAX_ABS:g}"),
        )
        if "pse_ci95_lower" in out and "pse_ci95_upper" in out:
            pse_ci_lower = pd.to_numeric(out["pse_ci95_lower"], errors="coerce")
            pse_ci_upper = pd.to_numeric(out["pse_ci95_upper"], errors="coerce")
            out["pse_delta_ci95_lower"] = pse_ci_lower - standard
            out["pse_delta_ci95_upper"] = pse_ci_upper - standard
            inside = (pse_ci_lower <= standard) & (pse_ci_upper >= standard)
            inside_mask = pse_ci_lower.notna() & pse_ci_upper.notna()
            out["standard_inside_pse_ci95"] = pd.Series(
                np.where(inside_mask, inside, np.nan), index=out.index, dtype="object"
            )
        if "pse_se" in out:
            pse_se = pd.to_numeric(out["pse_se"], errors="coerce")
            out["pse_delta_se"] = pse_se
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(pse_se > 0, (pse - standard) / pse_se, np.nan)
            try:
                from scipy.stats import norm  # type: ignore
                p_value = 2.0 * (1.0 - norm.cdf(np.abs(z)))
            except Exception:
                p_value = np.where(np.isfinite(z), np.exp(-0.717 * np.abs(z) - 0.416 * z * z), np.nan)
            out["pse_bias_p_value"] = pd.Series(p_value, index=out.index)
    if "jnd" in out:
        jnd = pd.to_numeric(out["jnd"], errors="coerce")
        out["jnd_over_standard"] = np.where(standard > 0, jnd / standard, np.nan)
        out["weber_fraction"] = out["jnd_over_standard"]
        if "jnd_ci95_lower" in out and "jnd_ci95_upper" in out:
            jnd_ci_lower = pd.to_numeric(out["jnd_ci95_lower"], errors="coerce")
            jnd_ci_upper = pd.to_numeric(out["jnd_ci95_upper"], errors="coerce")
            out["jnd_over_standard_ci95_lower"] = np.where(standard > 0, jnd_ci_lower / standard, np.nan)
            out["jnd_over_standard_ci95_upper"] = np.where(standard > 0, jnd_ci_upper / standard, np.nan)
    return out


def _fit_row_to_delta_predictions(fit_row: pd.Series, standard_value: float, delta_grid: np.ndarray) -> np.ndarray:
    comparison_grid = standard_value + delta_grid
    return predictions_from_fit_row(fit_row, comparison_grid)


def pse_bias_summary(fits: pd.DataFrame, group_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Return a compact bias sanity-check table for PSE fits.

    Columns mirror what a reviewer expects to see next to every psychometric
    fit: PSE, SE, 95% CI on the original stimulus axis, the same on the
    comparison-minus-standard axis, a boolean flag telling whether the
    standard value sits inside the CI (no significant bias), and the Wald
    p-value for the bias.
    """
    if fits is None or fits.empty:
        return pd.DataFrame()
    enriched = add_fit_delta_columns(fits)
    keep = []
    if group_cols:
        for col in group_cols:
            if col in enriched.columns:
                keep.append(col)
    elif "subject_id" in enriched.columns:
        keep.append("subject_id")
    if "finger_condition" in enriched.columns and "finger_condition" not in keep:
        keep.append("finger_condition")
    columns = keep + [
        "n_trials",
        "n_stimulus_levels",
        "fit_method",
        "standard_value",
        "pse",
        "pse_se",
        "pse_ci95_lower",
        "pse_ci95_upper",
        "pse_delta_from_standard",
        "pse_delta_ci95_lower",
        "pse_delta_ci95_upper",
        "standard_inside_pse_ci95",
        "pse_bias_p_value",
        "jnd",
        "jnd_ci95_lower",
        "jnd_ci95_upper",
        "lapse_rate",
        "fit_warning",
    ]
    present = [c for c in columns if c in enriched.columns]
    out = enriched[present].copy()
    if "standard_inside_pse_ci95" in out.columns:
        flag = out["standard_inside_pse_ci95"]
        out["bias_verdict"] = np.where(
            flag.isna(),
            "no_ci",
            np.where(
                flag.astype(bool),
                "standard within 95% CI (no significant bias)",
                "standard OUTSIDE 95% CI (significant bias)",
            ),
        )
    return out


_METRIC_CI_COLUMN_HINTS: dict[str, tuple[str, str]] = {
    "pse": ("pse_ci95_lower", "pse_ci95_upper"),
    "pse_delta_comparison_minus_standard": ("pse_delta_ci95_lower", "pse_delta_ci95_upper"),
    "pse_delta_from_standard": ("pse_delta_ci95_lower", "pse_delta_ci95_upper"),
    "jnd": ("jnd_ci95_lower", "jnd_ci95_upper"),
    "jnd_over_standard": ("jnd_over_standard_ci95_lower", "jnd_over_standard_ci95_upper"),
    "weber_fraction": ("jnd_over_standard_ci95_lower", "jnd_over_standard_ci95_upper"),
}


def save_article_style_psychophysics_figures(
    output_root: Path,
    clean: pd.DataFrame,
    psychometric_input_by_subject_finger: pd.DataFrame,
    pse_jnd_by_subject_finger: pd.DataFrame,
    finger_time_subject_summary: Optional[pd.DataFrame] = None,
    finger_appearance_order_summary: Optional[pd.DataFrame] = None,
    finger_by_appearance_order: Optional[pd.DataFrame] = None,
    preferred_subject: Optional[str] = None,
    fig_dpi: int = 160,
    include_workspace_comparison: bool = True,
    write_csv_outputs: bool = True,
) -> tuple[list[Path], pd.DataFrame]:
    """Save Farajian/Nisky-inspired psychophysics figures adapted to this task.

    Relevant article elements applied here:
    - psychometric curves with comparison-standard stiffness difference on x;
    - subject markers/lines plus a dotted group average for PSE and JND;
    - the same visual grammar for finger appearance order and success trends.
    """
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    fig_root = output_root / "figures"

    article_destinations = {
        "group_psychometric_data_by_finger.png": "psychometric_curves",
        "group_psychometric_curves_N_vs_L_by_finger.png": "psychometric_curves",
        "psychometric_overlay_subject": "psychometric_curves",
        "pse_article_style_by_finger.png": "psychometric_curves",
        "jnd_article_style_by_finger.png": "psychometric_curves",
        "success_article_style_by_appearance_order.png": "time_fatigue",
        "time_slope_article_style_by_appearance_order.png": "time_fatigue",
        "success_article_style_by_finger.png": "finger_time_appearance",
        "success_time_slope_article_style_by_finger.png": "finger_time_appearance",
    }

    def article_path(filename: str) -> Path:
        folder = article_destinations.get(filename, "psychometric_curves")
        if filename.startswith("psychometric_overlay_subject"):
            folder = article_destinations["psychometric_overlay_subject"]
        out_path = fig_root / folder / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    fits = add_fit_delta_columns(pse_jnd_by_subject_finger)
    if write_csv_outputs and not fits.empty:
        save_csv(fits, output_root, "pse_jnd_by_subject_finger_with_deltas.csv")

    selected_subject = _select_typical_subject_for_psychometric_overlay(fits, preferred_subject) if preferred_subject else None
    if selected_subject and not psychometric_input_by_subject_finger.empty:
        sub_agg = psychometric_input_by_subject_finger[psychometric_input_by_subject_finger["subject_id"].astype(str) == selected_subject].copy()
        sub_fits = fits[fits["subject_id"].astype(str) == selected_subject].copy()
        if not sub_agg.empty and not sub_fits.empty:
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            for finger in sorted(sub_agg["finger_condition"].dropna().unique(), key=_finger_sort_key):
                style = FINGER_STYLE.get(str(finger), {})
                g = sub_agg[sub_agg["finger_condition"] == finger].copy()
                fit_rows = sub_fits[sub_fits["finger_condition"] == finger]
                if g.empty:
                    continue
                g = add_delta_and_less_response_columns(g)
                g["delta"] = g["delta_comparison_minus_standard"]
                g = g.sort_values("delta")
                ax.scatter(
                    g["delta"],
                    g["p_comparison_greater"],
                    s=30 + 6 * g["n_trials"],
                    marker=style.get("marker", "o"),
                    color=style.get("color", None),
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.9,
                    label=style.get("label", str(finger)),
                )
                if not fit_rows.empty:
                    fit_row = fit_rows.iloc[0]
                    x_grid = np.linspace(float(g["delta"].min()), float(g["delta"].max()), 300)
                    std = float(g["standard_value"].median())
                    y_grid = _fit_row_to_delta_predictions(fit_row, std, x_grid)
                    if np.isfinite(y_grid).any():
                        ax.plot(x_grid, y_grid, color=style.get("color", "black"), linewidth=2)
                    pse_delta = fit_row.get("pse_delta_comparison_minus_standard", np.nan)
                    if np.isfinite(pse_delta):
                        ax.axvline(float(pse_delta), color=style.get("color", "black"), linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
            ax.axvline(0, color="black", linestyle=":", linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            set_psychometric_delta_axis(
                ax,
                pd.to_numeric(sub_agg["comparison_value"], errors="coerce")
                - pd.to_numeric(sub_agg["standard_value"], errors="coerce"),
            )
            ax.set_ylabel(PSYCHOMETRIC_GREATER_Y_LABEL)
            ax.set_title(f"Article-style psychometric curves, participant {selected_subject}")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            out = article_path(f"psychometric_overlay_subject_{sanitize_name(selected_subject)}.png")
            _finalize_fig(fig, out, fig_dpi)
            paths.append(out)

    if not psychometric_input_by_subject_finger.empty and not fits.empty:
        subject_agg = add_delta_and_less_response_columns(psychometric_input_by_subject_finger)
        subject_agg["delta"] = subject_agg["delta_comparison_minus_standard"]
        group_agg = subject_agg.copy()
        group_agg = (
            group_agg.groupby(["finger_condition", "delta"], dropna=False)
            .agg(
                p_comparison_greater=("p_comparison_greater", "mean"),
                sem_p_comparison_greater=("p_comparison_greater", _sem),
                n_subjects=("subject_id", "nunique"),
                total_trials=("n_trials", "sum"),
                standard_value=("standard_value", "median"),
            )
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        # Individual participant/finger fits in the background: this shows the
        # spread of all participants while keeping the group mean readable.
        for _, fit_row in fits.dropna(subset=["subject_id", "finger_condition"]).iterrows():
            finger = str(fit_row["finger_condition"])
            g = subject_agg[
                (subject_agg["subject_id"].astype(str) == str(fit_row["subject_id"]))
                & (subject_agg["finger_condition"].astype(str) == finger)
            ].dropna(subset=["delta", "p_comparison_greater"])
            if len(g) < 2:
                continue
            x_grid = np.linspace(float(g["delta"].min()), float(g["delta"].max()), 200)
            std = float(pd.to_numeric(g["standard_value"], errors="coerce").median())
            y_grid = _fit_row_to_delta_predictions(fit_row, std, x_grid)
            if np.isfinite(y_grid).any():
                ax.plot(x_grid, y_grid, color=_finger_color(finger), alpha=0.14, linewidth=0.9, zorder=1)
        for finger in _finger_order_present(group_agg["finger_condition"]):
            style = FINGER_STYLE.get(str(finger), {})
            g = group_agg[group_agg["finger_condition"] == finger].sort_values("delta")
            ax.errorbar(
                g["delta"],
                g["p_comparison_greater"],
                yerr=g["sem_p_comparison_greater"],
                marker=style.get("marker", "o"),
                color=style.get("color", None),
                linewidth=1.5,
                capsize=3,
                label=style.get("label", str(finger)),
            )
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        set_psychometric_delta_axis(ax, group_agg["delta"])
        ax.set_ylabel(PSYCHOMETRIC_MEAN_GREATER_Y_LABEL)
        ax.set_title("Group psychometric data by finger")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = article_path("group_psychometric_data_by_finger.png")
        _finalize_fig(fig, out, fig_dpi)
        paths.append(out)

        if include_workspace_comparison and "subject_id" in subject_agg and not fits.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
            for ax, setup in zip(axes, ["N", "L"]):
                subj_panel = subject_agg[subject_agg["subject_id"].astype(str).str.upper().str.startswith(setup)].copy()
                fits_panel = fits[fits["subject_id"].astype(str).str.upper().str.startswith(setup)].copy()
                if subj_panel.empty or fits_panel.empty:
                    ax.set_axis_off()
                    ax.set_title(f"{setup}: no data")
                    continue
                for _, fit_row in fits_panel.dropna(subset=["subject_id", "finger_condition"]).iterrows():
                    finger = str(fit_row["finger_condition"])
                    g = subj_panel[
                        (subj_panel["subject_id"].astype(str) == str(fit_row["subject_id"]))
                        & (subj_panel["finger_condition"].astype(str) == finger)
                    ].dropna(subset=["delta", "p_comparison_greater"])
                    if len(g) < 2:
                        continue
                    x_grid = np.linspace(float(g["delta"].min()), float(g["delta"].max()), 200)
                    std = float(pd.to_numeric(g["standard_value"], errors="coerce").median())
                    y_grid = _fit_row_to_delta_predictions(fit_row, std, x_grid)
                    if np.isfinite(y_grid).any():
                        ax.plot(x_grid, y_grid, color=_finger_color(finger), alpha=0.13, linewidth=0.9, zorder=1)
                panel_group = (
                    subj_panel.groupby(["finger_condition", "delta"], dropna=False)
                    .agg(
                        p_comparison_greater=("p_comparison_greater", "mean"),
                        sem_p_comparison_greater=("p_comparison_greater", _sem),
                        standard_value=("standard_value", "median"),
                    )
                    .reset_index()
                )
                for finger in _finger_order_present(panel_group["finger_condition"]):
                    style = FINGER_STYLE.get(str(finger), {})
                    g = panel_group[panel_group["finger_condition"].astype(str) == str(finger)].sort_values("delta")
                    ax.errorbar(
                        g["delta"],
                        g["p_comparison_greater"],
                        yerr=g["sem_p_comparison_greater"],
                        marker=style.get("marker", "o"),
                        color=style.get("color", None),
                        linewidth=2.0,
                        capsize=3,
                        label=style.get("label", str(finger)),
                        zorder=3,
                    )
                ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
                ax.axvline(0, color="black", linestyle=":", linewidth=1)
                ax.set_ylim(-0.05, 1.05)
                set_psychometric_delta_axis(ax, subj_panel["delta"])
                ax.set_title(f"{setup} group")
            axes[0].set_ylabel(PSYCHOMETRIC_MEAN_GREATER_Y_LABEL)
            axes[1].legend(loc="best", fontsize=8)
            fig.suptitle("Psychometric curves by group (individual fits faded)")
            fig.tight_layout()
            out = article_path("group_psychometric_curves_N_vs_L_by_finger.png")
            _finalize_fig(fig, out, fig_dpi)
            paths.append(out)

    metric_specs = [
        ("pse_delta_comparison_minus_standard", "PSE shift (comparison - standard)", "PSE by finger", "pse_article_style_by_finger.png"),
        ("jnd", "JND", "JND by finger", "jnd_article_style_by_finger.png"),
    ]
    for metric, ylabel, title, filename in metric_specs:
        path = _plot_article_style_metric_lines(
            fits,
            "finger_condition",
            metric,
            title,
            ylabel,
            "Finger",
            article_path(filename),
            x_order=_finger_order_present(fits["finger_condition"]) if not fits.empty and "finger_condition" in fits else None,
            style_col="finger_condition",
            fig_dpi=fig_dpi,
        )
        if path:
            paths.append(path)

    if finger_time_subject_summary is not None and not finger_time_subject_summary.empty:
        for metric, ylabel, title, filename in [
            ("success_rate", "Success rate", "Success by finger", "success_article_style_by_finger.png"),
            (
                "success_vs_within_finger_time_slope",
                "Success slope over within-finger time",
                "Within-finger time trend by finger",
                "success_time_slope_article_style_by_finger.png",
            ),
        ]:
            path = _plot_article_style_metric_lines(
                finger_time_subject_summary,
                "finger_condition",
                metric,
                title,
                ylabel,
                "Finger",
                article_path(filename),
                x_order=_finger_order_present(finger_time_subject_summary["finger_condition"]),
                style_col="subject_id",
                fig_dpi=fig_dpi,
            )
            if path:
                paths.append(path)

    if finger_time_subject_summary is not None and not finger_time_subject_summary.empty and "finger_appearance_order" in finger_time_subject_summary:
        appearance_df = finger_time_subject_summary.dropna(subset=["finger_appearance_order"]).copy()
        if not appearance_df.empty:
            appearance_df["finger_appearance_order"] = appearance_df["finger_appearance_order"].astype(int)
            for metric, ylabel, title, filename in [
                ("success_rate", "Success rate", "Success by finger appearance order", "success_article_style_by_appearance_order.png"),
                (
                    "success_vs_within_finger_time_slope",
                    "Success slope over within-finger time",
                    "Time trend by finger appearance order",
                    "time_slope_article_style_by_appearance_order.png",
                ),
            ]:
                path = _plot_article_style_metric_lines(
                    appearance_df,
                    "finger_appearance_order",
                    metric,
                    title,
                    ylabel,
                    "Finger appearance order",
                    article_path(filename),
                    x_order=_appearance_order(appearance_df["finger_appearance_order"]),
                    style_col="subject_id",
                    fig_dpi=fig_dpi,
                )
                if path:
                    paths.append(path)

    manifest = pd.DataFrame(
        {
            "figure": [str(p) for p in paths],
            "selected_typical_subject": selected_subject or "",
            "style_source": "Farajian_Nisky_eLife_52653_Figure_7_adapted",
        }
    )
    if write_csv_outputs:
        save_csv(manifest, output_root, "article_style_figure_manifest.csv")
    return paths, manifest


def _tracking_path_for_trial(row: pd.Series) -> Optional[Path]:
    source_file = row.get("source_file")
    pair_number = row.get("trial_index_raw")
    if pd.isna(source_file) or pd.isna(pair_number):
        return None
    try:
        pair_idx = int(float(pair_number))
    except Exception:
        return None
    return Path(str(source_file)).parent / f"pair_{pair_idx:03d}" / "tracking.csv"


def _analyze_tracking_file(
    tracking_path: Path,
    center_x: float = DEFAULT_CENTER_X,
    center_y: float = DEFAULT_CENTER_Y,
    n_time_bins: int = 120,
) -> tuple[dict[str, Any], pd.DataFrame]:
    try:
        df = read_csv_flexible(tracking_path, recover_malformed=True)
    except RuntimeError as exc:
        return {
            "tracking_file": str(tracking_path),
            "tracking_exists": True,
            "tracking_warning": "tracking_csv_unreadable",
            "tracking_read_error": str(exc)[:500],
        }, pd.DataFrame()
    required = {"timestamp", "object_x", "object_y"}
    if not required.issubset(df.columns):
        return {"tracking_warning": "missing_required_tracking_columns"}, pd.DataFrame()
    tracking_warning = "tracking_csv_recovered_from_malformed_rows" if df.attrs.get("csv_read_recovered") else ""
    try:
        n_data_lines = max(sum(1 for _ in tracking_path.open("rb")) - 1, 0)
    except OSError:
        n_data_lines = 0
    if n_data_lines and len(df) < n_data_lines:
        tracking_warning = f"tracking_csv_recovered_skipped_{n_data_lines - len(df)}_malformed_rows"
    out = df.copy()
    out["timestamp_dt"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["elapsed_seconds"] = (out["timestamp_dt"] - out["timestamp_dt"].min()).dt.total_seconds()
    out["object_dx_from_center_px"] = pd.to_numeric(out["object_x"], errors="coerce") - center_x
    out["object_dy_from_center_px"] = pd.to_numeric(out["object_y"], errors="coerce") - center_y
    out["center_distance_px"] = np.hypot(out["object_dx_from_center_px"], out["object_dy_from_center_px"])
    interacting = out["interacting"] if "interacting" in out else pd.Series(False, index=out.index)
    out["interacting_bool"] = interacting.astype(str).str.lower().isin(["true", "1", "yes"])
    stiffness = out["stiffness"] if "stiffness" in out else pd.Series(np.nan, index=out.index)
    out["skin_stretch_gain_mm_per_m"] = pd.to_numeric(stiffness, errors="coerce")
    # Backward-compatible alias for older generated CSVs/figures.
    out["skin_stretch_gain_mm_per_m_or_condition"] = out["skin_stretch_gain_mm_per_m"]
    out["skin_stretch_gain_normalized"] = out["skin_stretch_gain_mm_per_m"] / STIFFNESS_MAX_FOR_STRETCH_PROXY
    out["skin_stretch_command_proxy_px"] = out["center_distance_px"] * out["skin_stretch_gain_normalized"]
    valid = out.dropna(subset=["elapsed_seconds", "object_dx_from_center_px", "object_dy_from_center_px"]).copy()
    n_invalid_tracking_rows = int(len(out) - len(valid))
    if n_invalid_tracking_rows:
        tracking_warning = (
            f"{tracking_warning}; tracking_csv_ignored_{n_invalid_tracking_rows}_invalid_rows"
            if tracking_warning
            else f"tracking_csv_ignored_{n_invalid_tracking_rows}_invalid_rows"
        )
    dx = valid["object_dx_from_center_px"].diff()
    dy = valid["object_dy_from_center_px"].diff()
    step = np.hypot(dx, dy)
    duration = float(valid["elapsed_seconds"].max()) if valid["elapsed_seconds"].notna().any() else np.nan
    interaction = valid[valid["interacting_bool"]]
    movement = interaction if len(interaction) >= 2 else valid
    summary = {
        "tracking_file": str(tracking_path),
        "tracking_exists": True,
        "n_tracking_samples": int(len(valid)),
        "n_raw_tracking_rows": int(len(out)),
        "n_invalid_tracking_rows_ignored": n_invalid_tracking_rows,
        "n_interaction_samples": int(valid["interacting_bool"].sum()),
        "tracking_duration_seconds": duration,
        "mean_center_distance_px": float(movement["center_distance_px"].mean()) if len(movement) else np.nan,
        "max_center_distance_px": float(movement["center_distance_px"].max()) if len(movement) else np.nan,
        "path_length_px": float(step.fillna(0).sum()),
        "mean_object_dx_from_center_px": float(movement["object_dx_from_center_px"].mean()) if len(movement) else np.nan,
        "mean_object_dy_from_center_px": float(movement["object_dy_from_center_px"].mean()) if len(movement) else np.nan,
        "max_abs_object_dx_from_center_px": float(movement["object_dx_from_center_px"].abs().max()) if len(movement) else np.nan,
        "max_abs_object_dy_from_center_px": float(movement["object_dy_from_center_px"].abs().max()) if len(movement) else np.nan,
        "median_skin_stretch_gain_mm_per_m": float(movement["skin_stretch_gain_mm_per_m"].median()) if movement["skin_stretch_gain_mm_per_m"].notna().any() else np.nan,
        "median_skin_stretch_gain_mm_per_m_or_condition": float(movement["skin_stretch_gain_mm_per_m_or_condition"].median()) if movement["skin_stretch_gain_mm_per_m_or_condition"].notna().any() else np.nan,
        "mean_skin_stretch_command_proxy_px": float(movement["skin_stretch_command_proxy_px"].mean()) if len(movement) else np.nan,
        "max_skin_stretch_command_proxy_px": float(movement["skin_stretch_command_proxy_px"].max()) if len(movement) else np.nan,
        "tracking_warning": tracking_warning,
    }
    if valid.empty:
        return summary, pd.DataFrame()
    valid["time_fraction"] = np.where(duration > 0, valid["elapsed_seconds"] / duration, 0.0)
    valid = _add_quantile_bins(valid, "time_fraction", min(n_time_bins, max(2, len(valid))), "trajectory_time_bin")
    traj = (
        valid.groupby("trajectory_time_bin", dropna=False)
        .agg(
            time_fraction=("time_fraction", "mean"),
            object_dx_from_center_px=("object_dx_from_center_px", "mean"),
            object_dy_from_center_px=("object_dy_from_center_px", "mean"),
            center_distance_px=("center_distance_px", "mean"),
            skin_stretch_gain_mm_per_m=("skin_stretch_gain_mm_per_m", "median"),
            skin_stretch_gain_mm_per_m_or_condition=("skin_stretch_gain_mm_per_m_or_condition", "median"),
            skin_stretch_command_proxy_px=("skin_stretch_command_proxy_px", "mean"),
            interacting_fraction=("interacting_bool", "mean"),
        )
        .reset_index()
    )
    return summary, traj


def compute_xy_probing_and_skin_stretch_analysis(
    clean: pd.DataFrame,
    max_trials: Optional[int] = None,
    n_time_bins: int = 120,
) -> dict[str, pd.DataFrame]:
    """Summarize article-style trajectory variables for this experiment.

    This intentionally replaces article grip-force trajectories with variables
    available in this dataset:
    - XY object/probing displacement from the center of the plane;
    - the recorded skin-stretch gain in mm/m;
    - a conservative skin-stretch command proxy: center distance x gain / 175.

    The raw tracking files are read only.
    """
    trials = add_success_and_time_columns(clean)
    if trials.empty:
        return {"trial_summary": pd.DataFrame(), "trajectory_bins": pd.DataFrame(), "group_trajectory_bins": pd.DataFrame(), "group_summary": pd.DataFrame()}
    rows, traj_rows = [], []
    iterator = trials.sort_values(["subject_id", "global_trial_order"]).iterrows()
    count = 0
    for _, row in iterator:
        if max_trials is not None and count >= max_trials:
            break
        tracking_path = _tracking_path_for_trial(row)
        base = {
            "subject_id": row.get("subject_id"),
            "subject_group_label": row.get("subject_group_label"),
            "finger_condition": row.get("finger_condition"),
            "global_trial_order": row.get("global_trial_order"),
            "trial_index_raw": row.get("trial_index_raw"),
            "comparison_value": row.get("comparison_value"),
            "standard_value": row.get("standard_value"),
            "signed_stiffness_delta": row.get("signed_stiffness_delta"),
            "correct_response": row.get("correct_response"),
            "reaction_time": row.get("reaction_time"),
            "source_file": row.get("source_file"),
            "tracking_file": str(tracking_path) if tracking_path else "",
            "tracking_exists": bool(tracking_path and tracking_path.exists()),
        }
        if not tracking_path or not tracking_path.exists():
            rows.append({**base, "tracking_warning": "tracking_file_missing"})
            continue
        summary, traj = _analyze_tracking_file(tracking_path, n_time_bins=n_time_bins)
        rows.append({**base, **summary})
        if not traj.empty:
            for col, value in base.items():
                if col not in {"tracking_file", "tracking_exists"}:
                    traj[col] = value
            traj["tracking_file"] = str(tracking_path)
            traj_rows.append(traj)
        count += 1
    trial_summary = pd.DataFrame(rows)
    trajectory_bins = pd.concat(traj_rows, ignore_index=True, sort=False) if traj_rows else pd.DataFrame()
    if not trajectory_bins.empty:
        group_trajectory_bins = (
            trajectory_bins.groupby(["finger_condition", "trajectory_time_bin"], dropna=False)
            .agg(
                n_trials=("tracking_file", "nunique"),
                time_fraction=("time_fraction", "mean"),
                mean_object_dx_from_center_px=("object_dx_from_center_px", "mean"),
                sem_object_dx_from_center_px=("object_dx_from_center_px", _sem),
                mean_object_dy_from_center_px=("object_dy_from_center_px", "mean"),
                sem_object_dy_from_center_px=("object_dy_from_center_px", _sem),
                mean_center_distance_px=("center_distance_px", "mean"),
                sem_center_distance_px=("center_distance_px", _sem),
                mean_skin_stretch_command_proxy_px=("skin_stretch_command_proxy_px", "mean"),
                sem_skin_stretch_command_proxy_px=("skin_stretch_command_proxy_px", _sem),
                median_skin_stretch_gain_mm_per_m=("skin_stretch_gain_mm_per_m", "median"),
                median_skin_stretch_gain_mm_per_m_or_condition=("skin_stretch_gain_mm_per_m_or_condition", "median"),
                mean_interacting_fraction=("interacting_fraction", "mean"),
            )
            .reset_index()
        )
    else:
        group_trajectory_bins = pd.DataFrame()
    if not trial_summary.empty:
        group_summary = (
            trial_summary[trial_summary["tracking_exists"]]
            .groupby(["subject_id", "finger_condition"], dropna=False)
            .agg(
                n_trials_with_tracking=("tracking_file", "nunique"),
                success_rate=("correct_response", "mean"),
                mean_reaction_time=("reaction_time", "mean"),
                mean_max_center_distance_px=("max_center_distance_px", "mean"),
                mean_path_length_px=("path_length_px", "mean"),
                mean_skin_stretch_gain_mm_per_m=("median_skin_stretch_gain_mm_per_m", "mean"),
                mean_skin_stretch_gain_mm_per_m_or_condition=("median_skin_stretch_gain_mm_per_m_or_condition", "mean"),
                mean_max_skin_stretch_command_proxy_px=("max_skin_stretch_command_proxy_px", "mean"),
                mean_skin_stretch_command_proxy_px=("mean_skin_stretch_command_proxy_px", "mean"),
            )
            .reset_index()
        )
    else:
        group_summary = pd.DataFrame()
    return {
        "trial_summary": trial_summary,
        "trajectory_bins": trajectory_bins,
        "group_trajectory_bins": group_trajectory_bins,
        "group_summary": group_summary,
    }


def _save_table_if_not_empty(df: pd.DataFrame | None, root: Path, filename: str) -> Path | None:
    if df is None or df.empty:
        return None
    return save_csv(df, root, filename)


def _filter_subject(df: pd.DataFrame | None, subject: Any) -> pd.DataFrame:
    if df is None or df.empty or "subject_id" not in df:
        return pd.DataFrame()
    return df[df["subject_id"].astype(str) == str(subject)].copy()


def _qc_by_stiffness(clean: pd.DataFrame) -> pd.DataFrame:
    if clean is None or clean.empty or "comparison_value" not in clean:
        return pd.DataFrame()
    trials = add_success_and_time_columns(clean)
    return (
        trials.groupby(["subject_id", "finger_condition", "comparison_value"], dropna=False)
        .agg(
            n_trials=("correct_response", "size"),
            success_rate=("correct_response", "mean"),
            p_comparison_greater=("response_comparison_greater", "mean"),
            mean_reaction_time=("reaction_time", "mean"),
        )
        .reset_index()
    )


def _side_bias_by_stiffness(clean: pd.DataFrame) -> pd.DataFrame:
    if clean is None or clean.empty or "answer_chose_object_2" not in clean:
        return pd.DataFrame()
    group_cols = ["subject_id", "finger_condition"]
    if "comparison_value" in clean:
        group_cols.append("comparison_value")
    return (
        clean.groupby(group_cols, dropna=False)
        .agg(
            p_chose_object2=("answer_chose_object_2", "mean"),
            p_standard_on_object2=("standard_side", lambda x: float((x == "object_2").mean())) if "standard_side" in clean else ("answer_chose_object_2", "size"),
            n_trials=("answer_chose_object_2", "size"),
        )
        .reset_index()
    )


def _rmtree_windows_retry(path: Path, *, attempts: int = 6, delay_seconds: float = 0.2) -> None:
    """Remove a tree, retrying transient Windows/Dropbox file-handle locks."""
    import gc
    import os
    import stat
    import time

    target = Path(path)
    if not target.exists():
        return

    def _onerror(func: Any, failed_path: str, exc_info: Any) -> None:
        try:
            os.chmod(failed_path, stat.S_IWRITE)
            func(failed_path)
        except Exception:
            raise

    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            shutil.rmtree(target, onerror=_onerror)
            return
        except (PermissionError, OSError) as exc:
            last_error = exc
            gc.collect()
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def compute_psychophysics_group_aggregates(
    clean: pd.DataFrame,
    success_trials: pd.DataFrame,
    psychometric_input_by_subject_finger: pd.DataFrame,
    pse_jnd_by_subject_finger: pd.DataFrame,
    *,
    psignifit_available: bool = False,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Compute every group-level aggregate for one cohort of subjects.

    Mirrors the notebook's group-aggregation cells so a cohort and any of its
    sub-cohorts (e.g. the L_E and N_E halves of the combined ``L_N_E`` run) all
    go through identical logic. Pooling uses only subject/finger fits that pass
    the PSE reliability band (``excluded_from_group_analysis``); the per-subject
    inputs are left untouched. Returns a dict whose keys match the corresponding
    ``save_selected_analysis_tree`` keyword arguments, plus ``group_trials``,
    ``psychometric_input_subject_pooled``, and ``pse_jnd_subject_pooled``.
    """
    valid_group_fits = pse_jnd_by_subject_finger.copy()
    if "excluded_from_group_analysis" in valid_group_fits.columns:
        valid_group_fits = valid_group_fits[~valid_group_fits["excluded_from_group_analysis"].astype(bool)]
    valid_keys = set(
        valid_group_fits["subject_id"].astype(str) + "||" + valid_group_fits["finger_condition"].astype(str)
    )
    trial_key = success_trials["subject_id"].astype(str) + "||" + success_trials["finger_condition"].astype(str)
    group_trials = success_trials[trial_key.isin(valid_keys)].copy()

    psychometric_input_subject_pooled = make_psychometric_input(group_trials, ["subject_id"])
    psychometric_input_group_by_finger = make_psychometric_input(group_trials, ["finger_condition"])
    all_pooled_trials = group_trials.copy()
    all_pooled_trials["group"] = "all_pooled"
    psychometric_input_group_all_pooled = make_psychometric_input(all_pooled_trials, ["group"])

    (
        pse_jnd_subject_pooled,
        pse_jnd_group_by_finger,
        pse_jnd_group_all_pooled,
    ) = fit_conditions_many(
        [
            (psychometric_input_subject_pooled, ["subject_id"]),
            (psychometric_input_group_by_finger, ["finger_condition"]),
            (psychometric_input_group_all_pooled, ["group"]),
        ],
        psignifit_available,
        n_jobs=n_jobs,
    )

    order_effects_summary, order_effects_binned = compute_order_effects(clean)
    success_time_fatigue = compute_success_time_fatigue(success_trials, ["subject_id", "finger_condition"])
    finger_time_appearance = compare_fingers_over_time_and_appearance(success_trials)

    return {
        "group_trials": group_trials,
        "psychometric_input_subject_pooled": psychometric_input_subject_pooled,
        "psychometric_input_group_by_finger": psychometric_input_group_by_finger,
        "psychometric_input_group_all_pooled": psychometric_input_group_all_pooled,
        "pse_jnd_subject_pooled": pse_jnd_subject_pooled,
        "pse_jnd_group_by_finger": pse_jnd_group_by_finger,
        "pse_jnd_group_all_pooled": pse_jnd_group_all_pooled,
        "order_effects_summary": order_effects_summary,
        "order_effects_binned": order_effects_binned,
        "success_time_fatigue": success_time_fatigue,
        "finger_time_appearance": finger_time_appearance,
    }


def save_selected_analysis_tree(
    output_root: Path,
    selection: Any,
    *,
    clean: pd.DataFrame,
    flagged: pd.DataFrame | None = None,
    success_trials: pd.DataFrame | None = None,
    qc_summary: pd.DataFrame | None = None,
    psychometric_input_by_subject_finger: pd.DataFrame,
    pse_jnd_by_subject_finger: pd.DataFrame,
    psychometric_input_group_by_finger: pd.DataFrame | None = None,
    pse_jnd_group_by_finger: pd.DataFrame | None = None,
    psychometric_input_group_all_pooled: pd.DataFrame | None = None,
    pse_jnd_group_all_pooled: pd.DataFrame | None = None,
    order_effects_summary: pd.DataFrame | None = None,
    order_effects_binned: pd.DataFrame | None = None,
    success_time_fatigue: dict[str, pd.DataFrame] | None = None,
    finger_time_appearance: dict[str, pd.DataFrame] | None = None,
    psychophysics_group_comparisons: dict[str, pd.DataFrame] | None = None,
    fig_dpi: int = 160,
    write_subjects: bool = True,
) -> pd.DataFrame:
    """Save the requested subject/all tree for a selected cohort.

    Tree layout:
    - group/all outputs stay under ``output_root`` (for example ``results/L_E``);
    - per-subject outputs are sibling folders of ``output_root`` (for example
      ``results/L_E_1``, ``results/L_E_2``), not nested inside the group folder.

    Set ``write_subjects=False`` to write only the group/all folder and skip the
    per-subject sibling folders. This is used by the pooled ``L_N_E`` tree, whose
    subject folders are already written identically by the L_E and N_E sub-cohort
    trees. It is ignored for single-subject selections (the subject is always
    written there).
    """
    output_root = Path(output_root)
    subject_output_base = output_root.parent

    success_trials = success_trials if success_trials is not None else add_success_and_time_columns(clean)
    flagged = flagged if flagged is not None else pd.DataFrame()
    qc_summary = qc_summary if qc_summary is not None else make_qc_summary(clean, flagged)
    success_time_fatigue = success_time_fatigue or compute_success_time_fatigue(success_trials, ["subject_id", "finger_condition"])
    finger_time_appearance = finger_time_appearance or compare_fingers_over_time_and_appearance(success_trials)
    subject_ids = sorted(clean["subject_id"].dropna().astype(str).unique(), key=_subject_sort_key)
    subject_only = is_single_subject_selection(selection, subject_ids)
    fig_all_root = output_root / "figures" / "all"
    csv_all_root = output_root / "csv" / "all"
    if not subject_only:
        for root in [fig_all_root, csv_all_root]:
            root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    def record(scope: Any, statistic: str, kind: str, path: Any) -> None:
        if path:
            manifest_rows.append({"scope": scope, "statistic": statistic, "kind": kind, "path": str(path)})

    # Shared all-group CSVs.
    if not subject_only:
        shared_all_csv = csv_all_root / "shared"
        shared_tables = {
            "clean_trials.csv": clean,
            "flagged_trials.csv": flagged,
            "success_trials.csv": success_trials,
            "qc_summary.csv": qc_summary,
            "psychometric_input_by_subject_finger.csv": psychometric_input_by_subject_finger,
            "pse_jnd_by_subject_finger.csv": pse_jnd_by_subject_finger,
            "psychometric_input_group_by_finger.csv": psychometric_input_group_by_finger,
            "pse_jnd_group_by_finger.csv": pse_jnd_group_by_finger,
            "psychometric_input_group_all_pooled.csv": psychometric_input_group_all_pooled,
            "pse_jnd_group_all_pooled.csv": pse_jnd_group_all_pooled,
            "order_effects_summary.csv": order_effects_summary,
            "order_effects_binned.csv": order_effects_binned,
        }
        for name, table in shared_tables.items():
            path = _save_table_if_not_empty(table, shared_all_csv, name)
            record("all", "shared", "csv", path)
        for prefix, tables in [("time_fatigue", success_time_fatigue), ("finger_time_appearance", finger_time_appearance), ("group_comparisons", psychophysics_group_comparisons or {})]:
            for name, table in tables.items():
                path = _save_table_if_not_empty(table, shared_all_csv / prefix, f"{sanitize_name(name)}.csv")
                record("all", "shared", "csv", path)

    for subject in (subject_ids if (write_subjects or subject_only) else []):
        subject_root = subject_output_base / sanitize_name(subject)
        sfig = subject_root / "figures"
        scsv = subject_root / "csv"
        s_clean = _filter_subject(clean, subject)
        s_success = _filter_subject(success_trials, subject)
        s_flagged = _filter_subject(flagged, subject)
        s_qc = _filter_subject(qc_summary, subject)
        s_psy = _filter_subject(psychometric_input_by_subject_finger, subject)
        s_fits = _filter_subject(pse_jnd_by_subject_finger, subject)
        s_order_summary = _filter_subject(order_effects_summary, subject)
        s_order_binned = _filter_subject(order_effects_binned, subject)
        s_tf = {k: _filter_subject(v, subject) for k, v in success_time_fatigue.items()}
        s_fa = {k: _filter_subject(v, subject) for k, v in finger_time_appearance.items()}

        for name, table in {
            "clean_trials.csv": s_clean,
            "flagged_trials.csv": s_flagged,
            "success_trials.csv": s_success,
            "qc_summary.csv": s_qc,
            "psychometric_input_by_subject_finger.csv": s_psy,
            "pse_jnd_by_subject_finger.csv": s_fits,
            "order_effects_summary.csv": s_order_summary,
            "order_effects_binned.csv": s_order_binned,
        }.items():
            path = _save_table_if_not_empty(table, scsv, name)
            record(subject, "subject", "csv", path)
        for prefix, tables in [("time_fatigue", s_tf), ("finger_time_appearance", s_fa)]:
            for name, table in tables.items():
                path = _save_table_if_not_empty(table, scsv / prefix, f"{sanitize_name(name)}.csv")
                record(subject, "subject", "csv", path)

        for path in _save_subject_psychometric_curves(sfig, s_psy, s_fits, subject=subject, fig_dpi=fig_dpi):
            manifest_rows.append({"scope": subject, "statistic": "subject", "kind": "figure", "path": str(path)})
        subject_curve_path, subject_curve_summary = _save_group_curves_plot(
            s_psy,
            out_path=sfig / "psychometric_curves" / "group_curves_mean.png",
            statistic="mean",
            title=f"Psychometric curves across four fingers: {subject}",
            fig_dpi=fig_dpi,
        )
        csv_path = _save_table_if_not_empty(subject_curve_summary, scsv / "psychometric_curves", "group_curves_mean_source.csv")
        record(subject, "subject", "csv", csv_path)
        record(subject, "subject", "figure", subject_curve_path)
        for path in [
            _save_subject_article_summary(sfig, s_psy, s_fits, subject=subject, fig_dpi=fig_dpi),
            _save_appearance_plot(
                s_fa.get("subject_finger_summary", pd.DataFrame()),
                out_path=sfig / "success_by_finger_appearance_order.png",
                title=f"Success by finger appearance order: {subject}",
                fig_dpi=fig_dpi,
            ),
            _save_time_fatigue_line_plot(
                s_tf.get("order_bins", pd.DataFrame()),
                x_col="order_bin",
                y_col="success_rate",
                out_path=sfig / "time_fatigue" / "success_by_trial_order_bin.png",
                title=f"Time/fatigue: success across order bins ({subject})",
                fig_dpi=fig_dpi,
            ),
            _save_time_fatigue_line_plot(
                s_tf.get("reaction_time_bins", pd.DataFrame()),
                x_col="reaction_time_bin",
                y_col="success_rate",
                out_path=sfig / "time_fatigue" / "success_by_reaction_time_bin.png",
                title=f"Answer duration vs success ({subject})",
                fig_dpi=fig_dpi,
            ),
            _save_order_effects_plot(
                s_order_binned,
                out_path=sfig / "order_effects.png",
                title=f"Order effects: {subject}",
                fig_dpi=fig_dpi,
            ),
        ]:
            record(subject, "subject", "figure", path)

        subject_article_paths, subject_article_manifest = save_article_style_psychophysics_figures(
            subject_root,
            s_clean,
            s_psy,
            s_fits,
            s_fa.get("subject_finger_summary", pd.DataFrame()),
            s_fa.get("appearance_order_summary", pd.DataFrame()),
            s_fa.get("finger_by_appearance_order", pd.DataFrame()),
            preferred_subject=subject,
            fig_dpi=fig_dpi,
            include_workspace_comparison=False,
            write_csv_outputs=False,
        )
        path = _save_table_if_not_empty(subject_article_manifest, scsv, "article_style_figure_manifest.csv")
        record(subject, "subject", "csv", path)
        for path in subject_article_paths:
            manifest_rows.append({"scope": subject, "statistic": "subject", "kind": "figure", "path": str(path)})

        s_qc_stiffness = _qc_by_stiffness(s_clean)
        _save_table_if_not_empty(s_qc_stiffness, scsv, "qc_by_stiffness.csv")
        path = _plot_finger_columns_stiffness_dots(
            s_qc_stiffness,
            y_col="success_rate",
            y_label="Success rate",
            title=f"QC summary by stiffness: {subject}",
            out_path=sfig / "qc_summary_plots.png",
            fig_dpi=fig_dpi,
        )
        record(subject, "subject", "figure", path)

        s_side = _side_bias_by_stiffness(s_clean)
        _save_table_if_not_empty(s_side, scsv, "side_bias_by_stiffness.csv")
        path = _plot_side_bias_object_columns(
            s_side,
            title=f"Side bias by stiffness: {subject}",
            out_path=sfig / "side_bias_stiffnesses.png",
            fig_dpi=fig_dpi,
        )
        record(subject, "subject", "figure", path)

    if subject_only:
        manifest = pd.DataFrame(manifest_rows)
        save_csv(manifest, output_root, "analysis_tree_manifest.csv")
        return manifest

    # All-group figures and CSVs. Mean/median used to duplicate the same visual
    # information, so group outputs are now written once under figures/all and
    # csv/all. Keep subject folders unchanged.
    froot = fig_all_root
    croot = csv_all_root

    legacy_tmp = output_root / "_all_legacy_tmp"
    if legacy_tmp.exists():
        _rmtree_windows_retry(legacy_tmp)
    legacy_tmp.mkdir(parents=True, exist_ok=True)
    legacy_paths: list[Path] = []
    legacy_paths.extend(
        save_all_figures(
            legacy_tmp,
            clean,
            qc_summary,
            psychometric_input_by_subject_finger,
            pse_jnd_by_subject_finger,
            psychometric_input_group_by_finger if psychometric_input_group_by_finger is not None else pd.DataFrame(),
            pse_jnd_group_by_finger if pse_jnd_group_by_finger is not None else pd.DataFrame(),
            psychometric_input_group_all_pooled if psychometric_input_group_all_pooled is not None else pd.DataFrame(),
            pse_jnd_group_all_pooled if pse_jnd_group_all_pooled is not None else pd.DataFrame(),
            order_effects_binned if order_effects_binned is not None else pd.DataFrame(),
            fig_dpi,
        )
    )
    legacy_paths.extend(
        save_time_fatigue_figures(
            legacy_tmp,
            success_time_fatigue.get("reaction_time_bins", pd.DataFrame()),
            success_time_fatigue.get("order_bins", pd.DataFrame()),
            success_time_fatigue.get("first_second", pd.DataFrame()),
            success_time_fatigue.get("subject_summary", pd.DataFrame()),
            success_time_fatigue.get("slopes", pd.DataFrame()),
            fig_dpi,
        )
    )
    legacy_paths.extend(
        save_finger_time_appearance_figures(
            legacy_tmp,
            finger_time_appearance.get("group_finger_time_bins", pd.DataFrame()),
            finger_time_appearance.get("appearance_order_summary", pd.DataFrame()),
            finger_time_appearance.get("finger_by_appearance_order", pd.DataFrame()),
            finger_time_appearance.get("finger_slope_summary", pd.DataFrame()),
            fig_dpi,
            stiffness_time_slope_summary=finger_time_appearance.get("stiffness_slope_summary", pd.DataFrame()),
            subject_finger_time_bins=finger_time_appearance.get("subject_finger_time_bins", pd.DataFrame()),
        )
    )
    legacy_paths.extend(
        save_success_by_stiffness_repetition_figures(
            legacy_tmp,
            finger_time_appearance.get("trials", pd.DataFrame()),
            fig_dpi,
        )
    )
    article_paths, article_manifest = save_article_style_psychophysics_figures(
        legacy_tmp,
        clean,
        psychometric_input_by_subject_finger,
        pse_jnd_by_subject_finger,
        finger_time_appearance.get("subject_finger_summary", pd.DataFrame()),
        finger_time_appearance.get("appearance_order_summary", pd.DataFrame()),
        finger_time_appearance.get("finger_by_appearance_order", pd.DataFrame()),
        preferred_subject=None,
        fig_dpi=fig_dpi,
    )
    legacy_paths.extend(article_paths)
    _save_table_if_not_empty(article_manifest, croot, "article_style_figure_manifest.csv")

    legacy_figures_root = legacy_tmp / "figures"
    for src in sorted(set(Path(p) for p in legacy_paths if Path(p).exists()), key=lambda p: str(p).lower()):
        try:
            rel = src.relative_to(legacy_figures_root)
        except ValueError:
            rel = Path(src.name)
        dst = froot / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        manifest_rows.append({"scope": "all", "statistic": "all", "kind": "figure", "path": str(dst)})
    if legacy_tmp.exists():
        _rmtree_windows_retry(legacy_tmp)

    group_curve_dir = froot / "group_curves"
    if group_curve_dir.exists():
        psychometric_dir = froot / "psychometric_curves"
        psychometric_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(group_curve_dir.glob("group_finger_*.png"), key=lambda p: p.name):
            dst = psychometric_dir / src.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            manifest_rows[:] = [
                ({**row, "path": str(dst)} if str(row.get("path", "")) == str(src) else row)
                for row in manifest_rows
            ]
            if not any(str(row.get("path", "")) == str(dst) for row in manifest_rows):
                manifest_rows.append({"scope": "all", "statistic": "all", "kind": "figure", "path": str(dst)})

    _remove_all_only_legacy_plot(froot / "subject_finger_curves", manifest_rows)
    _remove_all_only_legacy_plot(froot / "group_curves", manifest_rows)
    if normalize_data_selection(selection) != "L_N_E":
        _remove_all_only_legacy_plot(froot / "psychometric_curves" / "group_psychometric_curves_N_vs_L_by_finger.png", manifest_rows)

    finger_subject_summary = finger_time_appearance.get("subject_finger_summary", pd.DataFrame())
    article_dirs = {
        "success_time_slope_article_style_by_finger.png": froot / "finger_time_appearance",
        "time_slope_article_style_by_appearance_order.png": froot / "time_fatigue",
        "success_article_style_by_finger.png": froot / "finger_time_appearance",
        "success_article_style_by_appearance_order.png": froot / "time_fatigue",
        "jnd_article_style_by_finger.png": froot / "psychometric_curves",
        "pse_article_style_by_finger.png": froot / "psychometric_curves",
        "group_psychometric_data_by_finger.png": froot / "psychometric_curves",
    }

    def all_article_path(filename: str) -> Path:
        return article_dirs.get(filename, froot / "psychometric_curves") / filename

    for metric, ylabel, title, filename, x_col, xlabel, x_order in [
        (
            "success_vs_within_finger_time_slope",
            "Success slope over within-finger time",
            "Within-finger time trend by finger",
            "success_time_slope_article_style_by_finger.png",
            "finger_condition",
            "Finger",
            _finger_order_present(finger_subject_summary["finger_condition"]) if not finger_subject_summary.empty and "finger_condition" in finger_subject_summary else None,
        ),
        (
            "success_vs_within_finger_time_slope",
            "Success slope over within-finger time",
            "Time trend by finger appearance order",
            "time_slope_article_style_by_appearance_order.png",
            "finger_appearance_order",
            "Finger appearance order",
            _appearance_order(finger_subject_summary["finger_appearance_order"]) if not finger_subject_summary.empty and "finger_appearance_order" in finger_subject_summary else None,
        ),
        (
            "success_rate",
            "Success rate",
            "Success by finger",
            "success_article_style_by_finger.png",
            "finger_condition",
            "Finger",
            _finger_order_present(finger_subject_summary["finger_condition"]) if not finger_subject_summary.empty and "finger_condition" in finger_subject_summary else None,
        ),
        (
            "success_rate",
            "Success rate",
            "Success by finger appearance order",
            "success_article_style_by_appearance_order.png",
            "finger_appearance_order",
            "Finger appearance order",
            _appearance_order(finger_subject_summary["finger_appearance_order"]) if not finger_subject_summary.empty and "finger_appearance_order" in finger_subject_summary else None,
        ),
    ]:
        out = all_article_path(filename)
        if not out.exists():
            fallback = _plot_article_style_metric_lines(
                finger_subject_summary,
                x_col,
                metric,
                title,
                ylabel,
                xlabel,
                out,
                x_order=x_order,
                style_col="subject_id",
                fig_dpi=fig_dpi,
            )
            if fallback is None and metric == "success_vs_within_finger_time_slope":
                fallback = _save_article_metric_fallback_plot(
                    finger_subject_summary,
                    x_col=x_col,
                    metric_col=metric,
                    title=title,
                    ylabel=ylabel,
                    xlabel=xlabel,
                    out_path=out,
                    fig_dpi=fig_dpi,
                )
            record("all", "all", "figure", fallback)

    # All-scope-only visual corrections requested after restoring the legacy
    # plot set. These overwrite all-level files only; subject folders are left
    # untouched.
    fit_df = add_fit_delta_columns(pse_jnd_by_subject_finger) if pse_jnd_by_subject_finger is not None and not pse_jnd_by_subject_finger.empty else pd.DataFrame()
    pse_metric = "pse_delta_comparison_minus_standard" if "pse_delta_comparison_minus_standard" in fit_df else "pse"
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_fit_metric_by_subject_finger(
            fit_df,
            metric_col="jnd",
            out_path=froot / "jnd_per_subject_and_finger.png",
            title="JND per participant and finger",
            ylabel="JND",
            fig_dpi=fig_dpi,
        ),
    )
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_fit_metric_by_subject_finger(
            fit_df,
            metric_col=pse_metric,
            out_path=froot / "pse_per_subject_and_finger.png",
            title="PSE per participant and finger",
            ylabel="PSE shift (comparison - standard)" if pse_metric != "pse" else "PSE",
            fig_dpi=fig_dpi,
        ),
    )
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_subject_background_metric_plot(
            fit_df,
            x_col="finger_condition",
            y_col="jnd",
            out_path=all_article_path("jnd_article_style_by_finger.png"),
            title="JND by finger",
            xlabel="Finger",
            ylabel="JND",
            line_color="subject",
            dot_color="subject",
            fig_dpi=fig_dpi,
        ),
    )
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_subject_background_metric_plot(
            finger_subject_summary,
            x_col="finger_condition",
            y_col="success_rate",
            out_path=all_article_path("success_article_style_by_finger.png"),
            title="Success by finger",
            xlabel="Finger",
            ylabel="Success rate",
            line_color="subject",
            dot_color="subject",
            fig_dpi=fig_dpi,
        ),
    )
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_subject_background_metric_plot(
            finger_subject_summary,
            x_col="finger_appearance_order",
            y_col="success_rate",
            out_path=all_article_path("success_article_style_by_appearance_order.png"),
            title="Success by finger appearance order",
            xlabel="Finger appearance order",
            ylabel="Success rate",
            line_color="subject",
            dot_color="finger",
            fig_dpi=fig_dpi,
        ),
    )
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_subject_background_metric_plot(
            finger_subject_summary,
            x_col="finger_appearance_order",
            y_col="success_rate",
            out_path=froot / "finger_time_appearance" / "success_by_finger_appearance_order.png",
            title="Success by finger appearance order",
            xlabel="Finger appearance order",
            ylabel="Success rate",
            line_color="subject",
            dot_color="finger",
            fig_dpi=fig_dpi,
        ),
    )
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_subject_background_metric_plot(
            finger_subject_summary,
            x_col="finger_appearance_order",
            y_col="success_rate",
            out_path=froot / "finger_time_appearance" / "success_by_finger_identity_and_appearance_order.png",
            title="Finger identity x appearance-order success",
            xlabel="Finger appearance order",
            ylabel="Success rate",
            line_color="subject",
            dot_color="finger",
            fig_dpi=fig_dpi,
        ),
    )
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_finger_time_plot(
            finger_time_appearance.get("subject_finger_time_bins", pd.DataFrame()),
            finger_time_appearance.get("group_finger_time_bins", pd.DataFrame()),
            out_path=froot / "finger_time_appearance" / "success_by_finger_over_within_finger_time.png",
            fig_dpi=fig_dpi,
        ),
    )
    _append_all_figure_manifest(
        manifest_rows,
        _save_all_success_order_slopes(
            success_time_fatigue.get("slopes", pd.DataFrame()),
            out_path=froot / "time_fatigue" / "success_order_slopes_by_finger.png",
            fig_dpi=fig_dpi,
        ),
    )

    if psychophysics_group_comparisons:
        scope_tmp = froot / "_scope_tmp"
        scope_manifest = save_scope_summary_plots(
            psychophysics_group_comparisons,
            scope_tmp,
            namespace="psychophysics",
            metrics=[*PSYCHOPHYSICS_GROUP_METRICS, *PSYCHOPHYSICS_FIT_GROUP_METRICS],
            keep_only=SCOPE_SUMMARY_FIGURE_WHITELIST,
        )
        if not scope_manifest.empty:
            scope_dir = froot / "psychophysics_scope_summaries"
            scope_dir.mkdir(parents=True, exist_ok=True)
            moved_figures = []
            for figure in scope_manifest["figure"].astype(str):
                src = Path(figure)
                dst = scope_dir / src.name
                if src.exists():
                    shutil.move(str(src), str(dst))
                    moved_figures.append(str(dst))
                else:
                    moved_figures.append(str(src))
            scope_manifest = scope_manifest.copy()
            scope_manifest["figure"] = moved_figures
        if scope_tmp.exists():
            _rmtree_windows_retry(scope_tmp)
        path = _save_table_if_not_empty(scope_manifest, croot, "psychophysics_scope_figure_manifest.csv")
        record("all", "all", "csv", path)
        for _, row in scope_manifest.iterrows():
            manifest_rows.append({"scope": "all", "statistic": "all", "kind": "figure", "path": row["figure"]})

    manifest = pd.DataFrame(manifest_rows)
    save_csv(manifest, output_root, "analysis_tree_manifest.csv")
    return manifest


# ---------------------------------------------------------------------------
# Experiment-group comparison section.
#
# Main experiment-group analyses intentionally contain only *_E subjects
# (N_E/L_E). Protocol/pilot subjects (*_P, currently N_P/L_P in the data) are
# summarized separately because their comparison-stiffness values differ and
# would otherwise make psychometric axes/group summaries misleading.

PSYCHOPHYSICS_GROUP_METRICS = [
    "correct_response",
    "incorrect_response",
    "response_comparison_greater",
    "reaction_time",
    "log_reaction_time",
    "comparison_value",
    "standard_value",
    "comparison_over_standard",
    "signed_stiffness_delta",
    "abs_stiffness_delta",
    "signed_delta_over_standard",
    "abs_delta_over_standard",
    "trial_order_fraction",
    "elapsed_fraction",
    "elapsed_minutes",
]

PSYCHOPHYSICS_FIT_GROUP_METRICS = [
    "pse",
    "pse_se",
    "pse_ci95_lower",
    "pse_ci95_upper",
    "jnd",
    "jnd_se",
    "jnd_ci95_lower",
    "jnd_ci95_upper",
    "pse_delta_from_standard",
    "pse_delta_se",
    "pse_delta_ci95_lower",
    "pse_delta_ci95_upper",
    "pse_bias_p_value",
    "jnd_over_standard",
    "weber_fraction",
    "abs_pse_delta_from_standard",
    "slope_at_pse",
    "lapse_rate",
    "lapse_rate_ci95_lower",
    "lapse_rate_ci95_upper",
    "n_trials",
]


def _add_prefixed_tables(target: dict[str, pd.DataFrame], prefix: str, source: dict[str, pd.DataFrame]) -> None:
    for name, table in source.items():
        target[f"{prefix}_{name}"] = table


def _present_condition_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    cols = []
    for col in candidates:
        if col in df.columns and df[col].notna().any() and df[col].nunique(dropna=True) > 0:
            cols.append(col)
    return cols


def _psychophysics_trial_condition_columns(df: pd.DataFrame) -> list[str]:
    return _present_condition_columns(
        df,
        [
            "finger_condition",
            "comparison_value",
            "standard_value",
            "signed_stiffness_delta",
            "abs_stiffness_delta",
            "success_label",
            "workspace_setup",
            "protocol_factor",
            "sex_factor",
            "age_group",
        ],
    )


def _psychophysics_fit_condition_columns(df: pd.DataFrame, base: list[str]) -> list[str]:
    return _present_condition_columns(
        df,
        [*base, "workspace_setup", "protocol_factor", "sex_factor", "age_group"],
    )


def _experiment_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    prepared = add_experiment_group_columns(df)
    return prepared[prepared["experiment_group"].notna()].copy()


def _protocol_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    prepared = add_protocol_group_columns(df)
    return prepared[prepared["protocol_group"].notna()].copy()


def _subject_level_trial_table(trial_data: pd.DataFrame) -> pd.DataFrame:
    if trial_data.empty:
        return pd.DataFrame()
    condition_cols = _psychophysics_trial_condition_columns(trial_data)
    group_cols = [c for c in ["subject_id", *condition_cols] if c in trial_data.columns]
    metrics = [c for c in PSYCHOPHYSICS_GROUP_METRICS if c in trial_data.columns]
    if not group_cols or not metrics:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {metric: (metric, "mean") for metric in metrics if metric not in group_cols}
    aggregations["n_trials"] = ("subject_id", "size")
    return trial_data.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()


def _anova_status_row(
    *,
    source: str,
    model_type: str,
    metric: str,
    factor: str,
    status: str,
    factor_b: str | None = None,
    term: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "model_type": model_type,
        "metric": metric,
        "factor": factor,
        "factor_b": factor_b or "",
        "term": term or factor,
        "n_observations": 0,
        "n_subjects": 0,
        "df_factor": np.nan,
        "df_error": np.nan,
        "f_value": np.nan,
        "p_value": np.nan,
        "eta_squared": np.nan,
        "partial_eta_squared": np.nan,
        "status": status,
    }


def _one_way_anova_rows(df: pd.DataFrame, source: str, metrics: list[str], factors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from scipy import stats as scipy_stats
    except Exception:  # pragma: no cover - optional dependency fallback
        scipy_stats = None
    for metric in metrics:
        for factor in factors:
            if metric == factor:
                continue
            if metric not in df.columns or factor not in df.columns:
                continue
            sub = df[[metric, factor, "subject_id"] if "subject_id" in df.columns else [metric, factor]].copy()
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.dropna(subset=[metric, factor])
            groups = [g[metric].to_numpy(dtype=float) for _, g in sub.groupby(factor, dropna=True) if len(g) >= 1]
            if len(groups) < 2 or sum(len(g) for g in groups) <= len(groups):
                rows.append(_anova_status_row(source=source, model_type="one_way_anova", metric=metric, factor=factor, status="insufficient_levels_or_observations"))
                continue
            grand = sub[metric].mean()
            ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups)
            ss_within = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in groups)
            ss_total = ss_between + ss_within
            df_between = len(groups) - 1
            df_within = sum(len(g) for g in groups) - len(groups)
            group_means = np.array([float(np.mean(g)) for g in groups], dtype=float)
            all_groups_constant = all(np.allclose(g, g[0], rtol=1e-12, atol=1e-12) for g in groups if len(g) > 0)
            if df_within <= 0:
                f_value, p_value = np.nan, np.nan
                status = "insufficient_error_degrees_of_freedom"
            elif all_groups_constant:
                f_value, p_value = np.nan, np.nan
                status = (
                    "constant_input_all_groups_same_mean"
                    if np.allclose(group_means, group_means[0], rtol=1e-12, atol=1e-12)
                    else "constant_input_all_groups_different_means"
                )
            elif ss_within <= 1e-12:
                f_value, p_value = np.nan, np.nan
                status = "near_zero_within_group_variance"
            elif scipy_stats is not None:
                f_value, p_value = scipy_stats.f_oneway(*groups)
                status = "ok"
            else:
                f_value = (ss_between / df_between) / (ss_within / df_within)
                p_value = np.nan
                status = "ok_no_scipy_p_value"
            rows.append(
                {
                    "source": source,
                    "model_type": "one_way_anova",
                    "metric": metric,
                    "factor": factor,
                    "factor_b": "",
                    "term": factor,
                    "n_observations": int(len(sub)),
                    "n_subjects": int(sub["subject_id"].nunique()) if "subject_id" in sub.columns else np.nan,
                    "df_factor": float(df_between),
                    "df_error": float(df_within),
                    "f_value": float(f_value) if np.isfinite(f_value) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "eta_squared": float(ss_between / ss_total) if ss_total > 0 else np.nan,
                    "partial_eta_squared": float(ss_between / (ss_between + ss_within)) if (ss_between + ss_within) > 0 else np.nan,
                    "status": status,
                }
            )
    return rows


def _two_way_anova_rows(df: pd.DataFrame, source: str, metrics: list[str], factor_pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except Exception:  # pragma: no cover - optional dependency fallback
        smf = None
        anova_lm = None
    for metric in metrics:
        for factor_a, factor_b in factor_pairs:
            if metric in {factor_a, factor_b}:
                continue
            if metric not in df.columns or factor_a not in df.columns or factor_b not in df.columns:
                continue
            sub_cols = [metric, factor_a, factor_b, "subject_id"] if "subject_id" in df.columns else [metric, factor_a, factor_b]
            sub = df[sub_cols].copy()
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.dropna(subset=[metric, factor_a, factor_b])
            if sub.empty or sub[factor_a].nunique() < 2 or sub[factor_b].nunique() < 2:
                rows.append(_anova_status_row(source=source, model_type="two_way_anova", metric=metric, factor=factor_a, factor_b=factor_b, term=f"{factor_a}:{factor_b}", status="insufficient_levels"))
                continue
            if smf is None or anova_lm is None:
                rows.append(_anova_status_row(source=source, model_type="two_way_anova", metric=metric, factor=factor_a, factor_b=factor_b, term=f"{factor_a}:{factor_b}", status="statsmodels_unavailable"))
                continue
            model_df = pd.DataFrame(
                {
                    "_metric": sub[metric].astype(float),
                    "_factor_a": sub[factor_a].astype(str),
                    "_factor_b": sub[factor_b].astype(str),
                }
            )
            try:
                model = smf.ols("_metric ~ C(_factor_a) * C(_factor_b)", data=model_df).fit()
                anova = anova_lm(model, typ=2)
            except Exception as exc:
                rows.append(_anova_status_row(source=source, model_type="two_way_anova", metric=metric, factor=factor_a, factor_b=factor_b, term=f"{factor_a}:{factor_b}", status=f"model_failed:{type(exc).__name__}"))
                continue
            residual_ss = float(anova.loc["Residual", "sum_sq"]) if "Residual" in anova.index else np.nan
            term_map = {
                "C(_factor_a)": factor_a,
                "C(_factor_b)": factor_b,
                "C(_factor_a):C(_factor_b)": f"{factor_a}:{factor_b}",
            }
            for sm_term, readable_term in term_map.items():
                if sm_term not in anova.index:
                    continue
                ss_term = float(anova.loc[sm_term, "sum_sq"])
                partial = ss_term / (ss_term + residual_ss) if np.isfinite(residual_ss) and (ss_term + residual_ss) > 0 else np.nan
                rows.append(
                    {
                        "source": source,
                        "model_type": "two_way_anova",
                        "metric": metric,
                        "factor": factor_a,
                        "factor_b": factor_b,
                        "term": readable_term,
                        "n_observations": int(len(model_df)),
                        "n_subjects": int(sub["subject_id"].nunique()) if "subject_id" in sub.columns else np.nan,
                        "df_factor": float(anova.loc[sm_term, "df"]),
                        "df_error": float(anova.loc["Residual", "df"]) if "Residual" in anova.index else np.nan,
                        "f_value": float(anova.loc[sm_term, "F"]) if np.isfinite(anova.loc[sm_term, "F"]) else np.nan,
                        "p_value": float(anova.loc[sm_term, "PR(>F)"]) if np.isfinite(anova.loc[sm_term, "PR(>F)"]) else np.nan,
                        "eta_squared": np.nan,
                        "partial_eta_squared": float(partial) if np.isfinite(partial) else np.nan,
                        "status": "ok",
                    }
                )
    return rows


def compute_psychophysics_factor_statistics(
    clean: pd.DataFrame,
    *,
    pse_jnd_by_subject_finger: pd.DataFrame | None = None,
    pse_jnd_subject_pooled: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute descriptive one- and two-way ANOVA-style screening tables.

    The trial table is first collapsed to participant/condition means so repeated
    trials from the same participant are visible without silently dominating
    group statistics. For final inference on binary 2AFC choices, use these
    tables as screening evidence and prefer a planned mixed-effects logistic
    model or repeated-measures model in the manuscript.
    """
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    trial_data = add_success_and_time_columns(clean) if not clean.empty else pd.DataFrame()
    trial_subject = _subject_level_trial_table(trial_data)
    sources = [("trial_subject_condition_mean", trial_subject, PSYCHOPHYSICS_GROUP_METRICS)]
    for source_name, fit_df, fit_base_conditions in [
        ("fit_by_subject_finger", pse_jnd_by_subject_finger, ["finger_condition"]),
        ("fit_subject_pooled", pse_jnd_subject_pooled, []),
    ]:
        source = add_fit_delta_columns(fit_df) if fit_df is not None and not fit_df.empty else pd.DataFrame()
        sources.append((source_name, source, PSYCHOPHYSICS_FIT_GROUP_METRICS))

    factor_candidates = [
        "experiment_group",
        "protocol_group",
        "subject_group",
        "setup_factor",
        "workspace_setup",
        "protocol_factor",
        "sex_factor",
        "age_group",
        "finger_condition",
        "comparison_value",
        "standard_value",
        "signed_stiffness_delta",
        "abs_stiffness_delta",
        "success_label",
    ]
    key_pairs = [
        ("experiment_group", "finger_condition"),
        ("experiment_group", "comparison_value"),
        ("experiment_group", "success_label"),
        ("protocol_group", "finger_condition"),
        ("protocol_group", "comparison_value"),
        ("workspace_setup", "finger_condition"),
        ("workspace_setup", "comparison_value"),
        ("sex_factor", "finger_condition"),
        ("age_group", "finger_condition"),
        ("protocol_factor", "finger_condition"),
    ]
    for source_name, df, requested_metrics in sources:
        if df.empty:
            status_rows.append({"source": source_name, "status": "empty_source"})
            continue
        prepared = add_psychophysics_context_columns(df)
        metrics = [m for m in requested_metrics if m in prepared.columns and pd.to_numeric(prepared[m], errors="coerce").notna().any()]
        factors = [f for f in factor_candidates if f in prepared.columns and prepared[f].nunique(dropna=True) >= 2]
        if not metrics or not factors:
            status_rows.append({"source": source_name, "status": "no_numeric_metrics_or_factors", "n_metrics": len(metrics), "n_factors": len(factors)})
            continue
        rows.extend(_one_way_anova_rows(prepared, source_name, metrics, factors))
        pairs = [(a, b) for a, b in key_pairs if a in factors and b in factors and a != b]
        rows.extend(_two_way_anova_rows(prepared, source_name, metrics, pairs))
        status_rows.append(
            {
                "source": source_name,
                "status": "ok",
                "n_input_rows": int(len(prepared)),
                "n_subjects": int(prepared["subject_id"].nunique()) if "subject_id" in prepared.columns else np.nan,
                "metrics": ", ".join(metrics),
                "factors": ", ".join(factors),
                "note": "ANOVA-style screening; prefer mixed/repeated models for final binary repeated-measures inference.",
            }
        )
    return {
        "psychophysics_factor_statistics": pd.DataFrame(rows),
        "psychophysics_factor_statistics_status": pd.DataFrame(status_rows),
    }


def compute_experiment_group_comparisons(
    clean: pd.DataFrame,
    *,
    pse_jnd_by_subject_finger: pd.DataFrame | None = None,
    pse_jnd_subject_pooled: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    trial_data = clean.copy()
    missing_trial_metrics = [m for m in PSYCHOPHYSICS_GROUP_METRICS if m not in trial_data.columns]
    if missing_trial_metrics and {"comparison_value", "standard_value", "response_comparison_greater"}.issubset(trial_data.columns):
        trial_data = add_success_and_time_columns(trial_data)
    elif not trial_data.empty:
        trial_data = add_psychophysics_context_columns(trial_data)
    trial_experiment = _experiment_only_rows(trial_data)
    trial_protocol = _protocol_only_rows(trial_data)

    tables: dict[str, pd.DataFrame] = {}
    _add_prefixed_tables(
        tables,
        "psychophysics_trial",
        compute_group_comparison_tables(
            trial_experiment,
            metric_columns=PSYCHOPHYSICS_GROUP_METRICS,
            condition_cols=_psychophysics_trial_condition_columns(trial_experiment),
        ),
    )
    _add_prefixed_tables(
        tables,
        "psychophysics_trial",
        compute_analysis_scope_tables(
            trial_experiment,
            metric_columns=PSYCHOPHYSICS_GROUP_METRICS,
            condition_cols=_psychophysics_trial_condition_columns(trial_experiment),
        ),
    )
    _add_prefixed_tables(
        tables,
        "psychophysics_trial",
        compute_setup_factor_tables(
            trial_experiment,
            metric_columns=PSYCHOPHYSICS_GROUP_METRICS,
            condition_cols=_psychophysics_trial_condition_columns(trial_experiment),
        ),
    )
    _add_prefixed_tables(
        tables,
        "psychophysics_trial",
        compute_protocol_group_comparison_tables(
            trial_protocol,
            metric_columns=PSYCHOPHYSICS_GROUP_METRICS,
            condition_cols=_psychophysics_trial_condition_columns(trial_protocol),
        ),
    )

    fit_sources = [
        ("psychophysics_fit_by_subject_finger", pse_jnd_by_subject_finger, ["finger_condition"]),
        ("psychophysics_fit_subject_pooled", pse_jnd_subject_pooled, []),
    ]
    fit_experiment_sources: dict[str, pd.DataFrame] = {}
    fit_protocol_sources: dict[str, pd.DataFrame] = {}
    for prefix, fit_df, condition_cols in fit_sources:
        source = fit_df if fit_df is not None else pd.DataFrame()
        if not source.empty:
            source = add_fit_delta_columns(source)
            # PSE band-pass: out-of-band fits are excluded from every group-level
            # aggregation/comparison below. They are still present (and flagged)
            # in the per-subject fit tables and individual psychometric curves.
            if "excluded_from_group_analysis" in source.columns:
                source = source[~source["excluded_from_group_analysis"].astype(bool)].copy()
        source_experiment = _experiment_only_rows(source)
        source_protocol = _protocol_only_rows(source)
        fit_experiment_sources[prefix] = source_experiment
        fit_protocol_sources[prefix] = source_protocol
        _add_prefixed_tables(
            tables,
            prefix,
            compute_group_comparison_tables(
                source_experiment,
                metric_columns=PSYCHOPHYSICS_FIT_GROUP_METRICS,
                condition_cols=_psychophysics_fit_condition_columns(source_experiment, condition_cols),
            ),
        )
        _add_prefixed_tables(
            tables,
            prefix,
            compute_analysis_scope_tables(
                source_experiment,
                metric_columns=PSYCHOPHYSICS_FIT_GROUP_METRICS,
                condition_cols=_psychophysics_fit_condition_columns(source_experiment, condition_cols),
            ),
        )
        _add_prefixed_tables(
            tables,
            prefix,
            compute_setup_factor_tables(
                source_experiment,
                metric_columns=PSYCHOPHYSICS_FIT_GROUP_METRICS,
                condition_cols=_psychophysics_fit_condition_columns(source_experiment, condition_cols),
            ),
        )
        _add_prefixed_tables(
            tables,
            prefix,
            compute_protocol_group_comparison_tables(
                source_protocol,
                metric_columns=PSYCHOPHYSICS_FIT_GROUP_METRICS,
                condition_cols=_psychophysics_fit_condition_columns(source_protocol, condition_cols),
            ),
        )
    tables.update(
        compute_psychophysics_factor_statistics(
            trial_experiment,
            pse_jnd_by_subject_finger=fit_experiment_sources.get("psychophysics_fit_by_subject_finger"),
            pse_jnd_subject_pooled=fit_experiment_sources.get("psychophysics_fit_subject_pooled"),
        )
    )
    protocol_factor_tables = compute_psychophysics_factor_statistics(
        trial_protocol,
        pse_jnd_by_subject_finger=fit_protocol_sources.get("psychophysics_fit_by_subject_finger"),
        pse_jnd_subject_pooled=fit_protocol_sources.get("psychophysics_fit_subject_pooled"),
    )
    if "psychophysics_factor_statistics" in protocol_factor_tables:
        tables["psychophysics_protocol_factor_statistics"] = protocol_factor_tables["psychophysics_factor_statistics"]
    if "psychophysics_factor_statistics_status" in protocol_factor_tables:
        tables["psychophysics_protocol_factor_statistics_status"] = protocol_factor_tables["psychophysics_factor_statistics_status"]
    return tables


def save_experiment_group_comparison_outputs(
    output_root: Path,
    clean: pd.DataFrame,
    *,
    pse_jnd_by_subject_finger: pd.DataFrame | None = None,
    pse_jnd_subject_pooled: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    tables = compute_experiment_group_comparisons(
        clean,
        pse_jnd_by_subject_finger=pse_jnd_by_subject_finger,
        pse_jnd_subject_pooled=pse_jnd_subject_pooled,
    )
    for name, df in tables.items():
        save_csv(df, output_root, f"{name}.csv")
    save_csv(motor_control_method_references(), output_root, "motor_control_method_references.csv")
    figure_manifest = save_scope_summary_plots(
        tables,
        output_root,
        namespace="psychophysics",
        metrics=[*PSYCHOPHYSICS_GROUP_METRICS, *PSYCHOPHYSICS_FIT_GROUP_METRICS],
        keep_only=SCOPE_SUMMARY_FIGURE_WHITELIST,
    )
    tables["psychophysics_scope_figure_manifest"] = figure_manifest
    return tables


def analysis_manifest(output_root: Path) -> pd.DataFrame:
    required = [
        "file_discovery_summary.csv",
        "combined_raw_imported_data.csv",
        "combined_clean_trials.csv",
        "combined_success_trials.csv",
        "combined_flagged_trials.csv",
        "qc_summary.csv",
        "farajian_style_input_by_subject_finger.csv",
        "psychometric_input_by_subject_finger.csv",
        "pse_jnd_by_subject_finger.csv",
        "pse_jnd_subject_pooled.csv",
        "pse_jnd_group_by_finger.csv",
        "pse_jnd_group_all_pooled.csv",
        "pse_bias_by_subject_finger.csv",
        "pse_bias_subject_pooled.csv",
        "pse_bias_group_by_finger.csv",
        "pse_bias_group_all_pooled.csv",
        "order_effects_summary.csv",
        "success_summary_by_subject.csv",
        "success_summary_by_subject_finger.csv",
        "success_trend_slopes_by_subject_finger.csv",
        "success_by_reaction_time_bin.csv",
        "success_by_order_bin.csv",
        "fatigue_first_second_summary.csv",
        "between_subject_success_time_stats.csv",
        "finger_time_subject_summary.csv",
        "finger_time_group_bins.csv",
        "finger_time_subject_bins.csv",
        "finger_appearance_order_summary.csv",
        "finger_by_appearance_order.csv",
        "finger_order_matrix.csv",
        "finger_time_slope_summary.csv",
        "stiffness_time_slope_summary.csv",
        "finger_time_slope_contrasts.csv",
        "finger_appearance_order_contrasts.csv",
        "pse_jnd_by_subject_finger_with_deltas.csv",
        "article_style_figure_manifest.csv",
        "psychophysics_trial_group_metric_summary.csv",
        "psychophysics_trial_group_condition_metric_summary.csv",
        "psychophysics_trial_within_group_condition_comparisons.csv",
        "psychophysics_trial_between_group_metric_comparisons.csv",
        "psychophysics_trial_analysis_scope_metric_summary.csv",
        "psychophysics_trial_analysis_scope_condition_metric_summary.csv",
        "psychophysics_trial_within_analysis_scope_condition_comparisons.csv",
        "psychophysics_trial_between_analysis_scope_metric_comparisons.csv",
        "psychophysics_trial_setup_balance.csv",
        "psychophysics_trial_setup_metric_summary.csv",
        "psychophysics_trial_setup_condition_metric_summary.csv",
        "psychophysics_trial_between_setup_metric_comparisons.csv",
        "psychophysics_fit_by_subject_finger_group_metric_summary.csv",
        "psychophysics_fit_by_subject_finger_group_condition_metric_summary.csv",
        "psychophysics_fit_by_subject_finger_within_group_condition_comparisons.csv",
        "psychophysics_fit_by_subject_finger_between_group_metric_comparisons.csv",
        "psychophysics_fit_by_subject_finger_analysis_scope_metric_summary.csv",
        "psychophysics_fit_by_subject_finger_analysis_scope_condition_metric_summary.csv",
        "psychophysics_fit_by_subject_finger_within_analysis_scope_condition_comparisons.csv",
        "psychophysics_fit_by_subject_finger_between_analysis_scope_metric_comparisons.csv",
        "psychophysics_fit_by_subject_finger_setup_balance.csv",
        "psychophysics_fit_by_subject_finger_setup_metric_summary.csv",
        "psychophysics_fit_by_subject_finger_setup_condition_metric_summary.csv",
        "psychophysics_fit_by_subject_finger_between_setup_metric_comparisons.csv",
        "psychophysics_fit_subject_pooled_group_metric_summary.csv",
        "psychophysics_fit_subject_pooled_group_condition_metric_summary.csv",
        "psychophysics_fit_subject_pooled_within_group_condition_comparisons.csv",
        "psychophysics_fit_subject_pooled_between_group_metric_comparisons.csv",
        "psychophysics_fit_subject_pooled_analysis_scope_metric_summary.csv",
        "psychophysics_fit_subject_pooled_analysis_scope_condition_metric_summary.csv",
        "psychophysics_fit_subject_pooled_within_analysis_scope_condition_comparisons.csv",
        "psychophysics_fit_subject_pooled_between_analysis_scope_metric_comparisons.csv",
        "psychophysics_fit_subject_pooled_setup_balance.csv",
        "psychophysics_fit_subject_pooled_setup_metric_summary.csv",
        "psychophysics_fit_subject_pooled_setup_condition_metric_summary.csv",
        "psychophysics_fit_subject_pooled_between_setup_metric_comparisons.csv",
        "psychophysics_factor_statistics.csv",
        "psychophysics_factor_statistics_status.csv",
        "psychophysics_scope_figure_manifest.csv",
        "motor_control_method_references.csv",
    ]
    return pd.DataFrame({"output": required, "exists": [(output_root / f).exists() for f in required], "path": [str(output_root / f) for f in required]})


# Re-export the plotting / figure-generation API that now lives in twoafc_figures so
# that ``import twoafc_psychophysics as pf`` keeps exposing every plot function (the
# notebook, tests, probing_analysis.py, and summary scripts call ``pf.<name>``). This
# import sits at the very end so all core names referenced by twoafc_figures are already
# defined; an explicit name list is used because ``import *`` skips underscore names.
from twoafc_figures import (  # noqa: E402,F401  -- re-export plotting API
    _appearance_order,
    _appearance_tick_labels,
    _append_all_figure_manifest,
    _axis_order_and_labels,
    _finalize_fig,
    _finger_color,
    _finger_label,
    _finger_order_present,
    _finger_sort_key,
    _group_curve_background_fits,
    _group_curve_summary,
    _metric_error_bars,
    _plot_article_style_metric_lines,
    _plot_finger_columns_stiffness_dots,
    _plot_side_bias_object_columns,
    _remove_all_only_legacy_plot,
    _render_fit_curve_job,
    _render_fit_curve_jobs,
    _save_all_finger_time_plot,
    _save_all_fit_metric_by_subject_finger,
    _save_all_subject_background_metric_plot,
    _save_all_success_order_slopes,
    _save_appearance_plot,
    _save_article_metric_fallback_plot,
    _save_group_curves_plot,
    _save_order_effects_plot,
    _save_subject_article_summary,
    _save_subject_psychometric_curves,
    _save_time_fatigue_line_plot,
    _select_typical_subject_for_psychometric_overlay,
    _shade_for_workspace,
    _stiffness_colors,
    _style_color,
    _subject_palette,
    _workspace_letter,
    plot_fit_curve,
    save_all_figures,
    save_finger_time_appearance_figures,
    save_success_by_stiffness_repetition_figures,
    save_time_fatigue_figures,
    save_xy_probing_skin_stretch_figures,
    set_psychometric_delta_axis,
)
