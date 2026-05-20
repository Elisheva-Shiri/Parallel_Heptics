"""Shared comparison helpers for analysis modules.

The study-level groups are:
- ``N_E``: natural experiment
- ``L_E``: lab experiment
- ``L_P``: lab protocol

The helpers intentionally stay dependency-light and return CSV-friendly
``pandas.DataFrame`` tables. They summarize exact experiment groups, compare
conditions inside each group, compare groups with one another, and provide
scope-expanded tables for all participants, subgroups, and single participants.
"""
from __future__ import annotations

import math
import json
import re
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd

EXPERIMENT_GROUP_COLUMN = "experiment_group"
EXPERIMENT_GROUP_LABEL_COLUMN = "experiment_group_label"
ANALYSIS_SCOPE_COLUMN = "analysis_scope"
ANALYSIS_SCOPE_VALUE_COLUMN = "analysis_scope_value"
SETUP_FACTOR_COLUMN = "setup_factor"
SETUP_LABEL_COLUMN = "setup_label"
EXPERIMENT_GROUP_ORDER = ["N_E", "L_E", "L_P"]
SUBGROUP_ORDER = ["N", "L", "P", "E"]
SETUP_FACTOR_ORDER = ["no_airsled", "airsled"]
SETUP_FACTOR_LABELS = {
    "airsled": "with_airsled_L",
    "no_airsled": "without_airsled_N",
}
EXPERIMENT_GROUP_LABELS = {
    "N_E": "natural_experiment",
    "L_E": "lab_experiment",
    "L_P": "lab_protocol",
    "N": "natural",
    "L": "lab",
    "P": "protocol",
    "E": "experiment",
}
_GROUP_ALIASES = {
    "N": "N",
    "N_E": "N_E",
    "NE": "N_E",
    "NATURAL_EXPERIMENT": "N_E",
    "NATURAL_EXPIRIMENT": "N_E",
    "NATURAL_EXP": "N_E",
    "NATURAL": "N_E",
    "L": "L",
    "L_E": "L_E",
    "LE": "L_E",
    "LAB_EXPERIMENT": "L_E",
    "LAB_EXPIRIMENT": "L_E",
    "LAB_EXP": "L_E",
    "P": "P",
    "L_P": "L_P",
    "LP": "L_P",
    "LAB_PROTOCOL": "L_P",
    "LAB_PROTO": "L_P",
    "PROTOCOL": "L_P",
}
_GROUP_SOURCE_COLUMNS = [
    EXPERIMENT_GROUP_COLUMN,
    "study_group",
    "analysis_group",
    "protocol_group",
    "experiment_type",
    "group",
    "subject_group",
]


def _tokenize_group_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = text.replace("EXPIRIMENT", "EXPERIMENT")
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_experiment_group(value: Any) -> str | float:
    """Return a canonical experiment or broad subgroup label when recognized."""
    token = _tokenize_group_text(value)
    if not token:
        return np.nan
    if token in _GROUP_ALIASES:
        return _GROUP_ALIASES[token]
    compact = token.replace("_", "")
    if compact in _GROUP_ALIASES:
        return _GROUP_ALIASES[compact]
    # Handles subject IDs such as N_E01, L-E-03, or lab_protocol_subject_4.
    for alias, canonical in sorted(_GROUP_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        alias_compact = alias.replace("_", "")
        if token.startswith(alias + "_") or compact.startswith(alias_compact):
            return canonical
    if "NATURAL" in token and "EXPERIMENT" in token:
        return "N_E"
    if "LAB" in token and "EXPERIMENT" in token:
        return "L_E"
    if "LAB" in token and "PROTOCOL" in token:
        return "L_P"
    return np.nan


def _normalize_exact_experiment_group(value: Any) -> str | float:
    group = normalize_experiment_group(value)
    return group if group in EXPERIMENT_GROUP_ORDER else np.nan


def infer_subgroups_from_value(value: Any) -> list[str]:
    """Infer broad subgroup labels (``N``, ``L``, ``P``, ``E``) from one value.

    This supports the requested rollups beyond exact experiment groups:
    all lab participants (``L``/``_L``), all protocol participants (``P``/``_P``),
    natural participants (``N``), and the exact ``L_P``/``L_E`` groups. Values
    are stable, ordered, and de-duplicated.
    """
    token = _tokenize_group_text(value)
    if not token:
        return []
    canonical = normalize_experiment_group(value)
    subgroups: list[str] = []
    if canonical == "N_E":
        subgroups.extend(["N", "E"])
    elif canonical == "L_E":
        subgroups.extend(["L", "E"])
    elif canonical == "L_P":
        subgroups.extend(["L", "P"])
    elif canonical in set(SUBGROUP_ORDER):
        subgroups.append(str(canonical))

    parts = [p for p in token.split("_") if p]
    compact = token.replace("_", "")
    for label in SUBGROUP_ORDER:
        if label in parts or token.startswith(label + "_") or token.endswith("_" + label):
            subgroups.append(label)
    if compact.startswith("N"):
        subgroups.append("N")
    if compact.startswith("L"):
        subgroups.append("L")
    if compact.endswith("P") or "PROTOCOL" in token:
        subgroups.append("P")
    if compact.endswith("E") or "EXPERIMENT" in token:
        subgroups.append("E")
    return [g for g in SUBGROUP_ORDER if g in set(subgroups)]


def setup_factor_from_value(value: Any) -> str | float:
    """Infer the laboratory setup factor from study labels.

    Per the current protocol decision, labels beginning with ``L`` are the
    laboratory airsled setup and labels beginning with ``N`` are the no-airsled
    setup. ``L_P`` and ``L_E`` both map to ``airsled``; ``N_E`` maps to
    ``no_airsled``. Compact legacy subject IDs are also handled: ``P#`` maps to
    the lab-protocol airsled setup and ``E#`` maps to the no-airsled experiment
    setup used before the explicit L/N naming migration.
    """
    token = _tokenize_group_text(value)
    compact = token.replace("_", "")
    if token in {"AIRSLED", "WITH_AIRSLED", "WITH_AIRSLED_L"}:
        return "airsled"
    if token in {"NO_AIRSLED", "WITHOUT_AIRSLED", "WITHOUT_AIRSLED_N"}:
        return "no_airsled"
    if re.match(r"^P\d+$", compact):
        return "airsled"
    if re.match(r"^E\d+$", compact):
        return "no_airsled"
    canonical = normalize_experiment_group(value)
    if canonical in {"L", "L_E", "L_P", "P"}:
        return "airsled"
    if canonical in {"N", "N_E"}:
        return "no_airsled"
    subgroups = infer_subgroups_from_value(value)
    if "L" in subgroups or "P" in subgroups:
        return "airsled"
    if "N" in subgroups:
        return "no_airsled"
    return np.nan


def infer_setup_factor(df: pd.DataFrame, *, subject_col: str = "subject_id") -> pd.Series:
    """Infer ``airsled``/``no_airsled`` from common group columns or subject IDs."""
    if df.empty:
        return pd.Series(dtype="object", index=df.index, name=SETUP_FACTOR_COLUMN)
    out = pd.Series(np.nan, index=df.index, dtype="object", name=SETUP_FACTOR_COLUMN)
    for col in [SETUP_FACTOR_COLUMN, *_GROUP_SOURCE_COLUMNS]:
        if col not in df.columns:
            continue
        inferred = df[col].map(setup_factor_from_value)
        out = out.where(out.notna(), inferred)
    if subject_col in df.columns:
        inferred = df[subject_col].map(setup_factor_from_value)
        out = out.where(out.notna(), inferred)
    return out


def add_setup_factor_columns(df: pd.DataFrame, *, subject_col: str = "subject_id") -> pd.DataFrame:
    """Return a copy with setup factor labels for L/N airsled diagnostics."""
    out = df.copy()
    out[SETUP_FACTOR_COLUMN] = infer_setup_factor(out, subject_col=subject_col)
    out[SETUP_LABEL_COLUMN] = out[SETUP_FACTOR_COLUMN].map(SETUP_FACTOR_LABELS).fillna("unknown_setup")
    return out


def compute_setup_balance(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    recommended_total_subjects: int = 24,
) -> pd.DataFrame:
    """Report L/N setup sample-size balance and recommendation flags."""
    if df.empty:
        return pd.DataFrame()
    prepared = add_setup_factor_columns(df, subject_col=subject_col)
    prepared = prepared[prepared[SETUP_FACTOR_COLUMN].notna()].copy()
    if prepared.empty:
        return pd.DataFrame()
    min_per_setup = max(1, recommended_total_subjects // 2)
    counts = []
    for setup in SETUP_FACTOR_ORDER:
        sub = prepared[prepared[SETUP_FACTOR_COLUMN] == setup]
        n_subjects = int(sub[subject_col].nunique()) if subject_col in sub.columns else int(len(sub))
        counts.append(
            {
                SETUP_FACTOR_COLUMN: setup,
                SETUP_LABEL_COLUMN: SETUP_FACTOR_LABELS[setup],
                "n_subjects": n_subjects,
                "n_observations": int(len(sub)),
                "recommended_min_subjects": min_per_setup,
                "meets_recommended_half_sample": bool(n_subjects >= min_per_setup),
            }
        )
    out = pd.DataFrame(counts)
    present = out[out["n_subjects"] > 0]
    can_model = bool((out["n_subjects"] >= min_per_setup).all())
    visually_only = bool(len(present) == 2 and not can_model)
    if can_model:
        recommendation = "balanced_enough_to_include_setup_factor_then_collapse_if_no_effect"
    elif visually_only:
        recommendation = "underpowered_for_setup_test_use_visual_check_and_report_limitation"
    else:
        recommendation = "only_one_setup_present_analyze_as_single_setup_and_report_protocol_limitation"
    out["can_test_setup_factor_recommended_power"] = can_model
    out["visual_check_recommended"] = visually_only
    out["setup_recommendation"] = recommendation
    return out


def compute_setup_factor_tables(
    df: pd.DataFrame,
    *,
    metric_columns: Iterable[str],
    subject_col: str = "subject_id",
    condition_cols: Iterable[str] = (),
    recommended_total_subjects: int = 24,
) -> dict[str, pd.DataFrame]:
    """Compute setup-factor diagnostics for L=airsled and N=no-airsled.

    The tables are descriptive and pairwise by default. The balance table says
    whether the current data meet the professor's recommended 24-subject / half
    split before interpreting setup as a formal factor.
    """
    empty = {
        "setup_balance": pd.DataFrame(),
        "setup_metric_summary": pd.DataFrame(),
        "setup_condition_metric_summary": pd.DataFrame(),
        "between_setup_metric_comparisons": pd.DataFrame(),
    }
    if df.empty:
        return empty
    prepared = add_setup_factor_columns(df, subject_col=subject_col)
    prepared = prepared[prepared[SETUP_FACTOR_COLUMN].notna()].copy()
    if prepared.empty:
        return empty
    metrics = available_numeric_metrics(prepared, metric_columns)
    balance = compute_setup_balance(
        prepared, subject_col=subject_col, recommended_total_subjects=recommended_total_subjects
    )
    if not metrics:
        return {**empty, "setup_balance": balance}
    present_conditions = [c for c in condition_cols if c in prepared.columns]
    return {
        "setup_balance": balance,
        "setup_metric_summary": _summary_rows(prepared, [SETUP_FACTOR_COLUMN], metrics, subject_col),
        "setup_condition_metric_summary": _summary_rows(prepared, [SETUP_FACTOR_COLUMN] + present_conditions, metrics, subject_col) if present_conditions else pd.DataFrame(),
        "between_setup_metric_comparisons": _pairwise_between_categories(
            prepared,
            metrics,
            subject_col,
            present_conditions,
            category_col=SETUP_FACTOR_COLUMN,
            order=SETUP_FACTOR_ORDER,
        ),
    }

def infer_experiment_group(df: pd.DataFrame, *, subject_col: str = "subject_id") -> pd.Series:
    """Infer canonical experiment groups from common columns or subject IDs."""
    if df.empty:
        return pd.Series(dtype="object", index=df.index, name=EXPERIMENT_GROUP_COLUMN)
    out = pd.Series(np.nan, index=df.index, dtype="object", name=EXPERIMENT_GROUP_COLUMN)
    for col in _GROUP_SOURCE_COLUMNS:
        if col not in df.columns:
            continue
        inferred = df[col].map(_normalize_exact_experiment_group)
        out = out.where(out.notna(), inferred)
    if subject_col in df.columns:
        inferred = df[subject_col].map(_normalize_exact_experiment_group)
        out = out.where(out.notna(), inferred)
    return out


def infer_analysis_subgroups(df: pd.DataFrame, *, subject_col: str = "subject_id") -> pd.Series:
    """Return broad subgroup memberships as tuples for each row."""
    if df.empty:
        return pd.Series(dtype="object", index=df.index, name="analysis_subgroups")
    candidate_cols = [EXPERIMENT_GROUP_COLUMN, "subject_group", "study_group", "analysis_group", "group"]
    memberships: list[tuple[str, ...]] = []
    for _, row in df.iterrows():
        found: list[str] = []
        for col in candidate_cols:
            if col in df.columns:
                found.extend(infer_subgroups_from_value(row.get(col)))
        if subject_col in df.columns:
            found.extend(infer_subgroups_from_value(row.get(subject_col)))
        memberships.append(tuple(g for g in SUBGROUP_ORDER if g in set(found)))
    return pd.Series(memberships, index=df.index, name="analysis_subgroups")


def add_experiment_group_columns(df: pd.DataFrame, *, subject_col: str = "subject_id") -> pd.DataFrame:
    """Return a copy with canonical group and readable label columns."""
    out = df.copy()
    out[EXPERIMENT_GROUP_COLUMN] = infer_experiment_group(out, subject_col=subject_col)
    out[EXPERIMENT_GROUP_LABEL_COLUMN] = out[EXPERIMENT_GROUP_COLUMN].map(EXPERIMENT_GROUP_LABELS).fillna("other")
    return out


def expand_analysis_scopes(df: pd.DataFrame, *, subject_col: str = "subject_id") -> pd.DataFrame:
    """Duplicate rows across all, exact-group, subgroup, and participant scopes."""
    if df.empty:
        return df.copy()
    prepared = add_experiment_group_columns(df, subject_col=subject_col)
    subgroup_memberships = infer_analysis_subgroups(prepared, subject_col=subject_col)
    rows: list[pd.DataFrame] = []

    def append_scope(masked: pd.DataFrame, scope: str, value: Any) -> None:
        if masked.empty or pd.isna(value):
            return
        scoped = masked.copy()
        scoped[ANALYSIS_SCOPE_COLUMN] = scope
        scoped[ANALYSIS_SCOPE_VALUE_COLUMN] = str(value)
        rows.append(scoped)

    append_scope(prepared, "all", "all_participants")
    if EXPERIMENT_GROUP_COLUMN in prepared.columns:
        for group, sub in prepared[prepared[EXPERIMENT_GROUP_COLUMN].notna()].groupby(EXPERIMENT_GROUP_COLUMN, dropna=False):
            append_scope(sub, "experiment_group", group)
    for subgroup in SUBGROUP_ORDER:
        mask = subgroup_memberships.map(lambda memberships: subgroup in memberships)
        append_scope(prepared[mask], "subgroup", subgroup)
    if subject_col in prepared.columns:
        for subject, sub in prepared[prepared[subject_col].notna()].groupby(subject_col, dropna=False):
            append_scope(sub, "participant", subject)
    if not rows:
        return pd.DataFrame(columns=list(prepared.columns) + [ANALYSIS_SCOPE_COLUMN, ANALYSIS_SCOPE_VALUE_COLUMN])
    return pd.concat(rows, ignore_index=True, sort=False)


def available_numeric_metrics(df: pd.DataFrame, metric_columns: Iterable[str]) -> list[str]:
    metrics: list[str] = []
    for col in metric_columns:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            metrics.append(col)
    return metrics


def _sem(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def _ci95_lower(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(x.mean() - 1.96 * _sem(x)) if len(x) > 1 else float(x.mean())


def _ci95_upper(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(x.mean() + 1.96 * _sem(x)) if len(x) > 1 else float(x.mean())


def _pooled_cohens_d(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce").dropna()
    bb = pd.to_numeric(b, errors="coerce").dropna()
    if len(aa) < 2 or len(bb) < 2:
        return np.nan
    pooled_var = ((len(aa) - 1) * aa.var(ddof=1) + (len(bb) - 1) * bb.var(ddof=1)) / (len(aa) + len(bb) - 2)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return np.nan
    return float((bb.mean() - aa.mean()) / math.sqrt(pooled_var))


def _summary_rows(df: pd.DataFrame, group_cols: list[str], metrics: list[str], subject_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty or not group_cols or not metrics:
        return pd.DataFrame()
    grouped = df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            values = pd.to_numeric(g[metric], errors="coerce").dropna()
            row = {
                **base,
                "metric": metric,
                "n_observations": int(values.size),
                "mean": float(values.mean()) if values.size else np.nan,
                "median": float(values.median()) if values.size else np.nan,
                "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "sem": _sem(values),
                "ci95_lower": _ci95_lower(values),
                "ci95_upper": _ci95_upper(values),
                "raw_values_json": json.dumps([float(v) for v in values]),
            }
            row["n_subjects"] = int(g[subject_col].nunique()) if subject_col in g.columns else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _subject_level(df: pd.DataFrame, group_cols: list[str], metrics: list[str], subject_col: str) -> pd.DataFrame:
    if subject_col not in df.columns:
        return df[group_cols + metrics].copy()
    return df.groupby([subject_col] + group_cols, dropna=False).agg(**{m: (m, "mean") for m in metrics}).reset_index()


def _pairwise_between_categories(
    df: pd.DataFrame,
    metrics: list[str],
    subject_col: str,
    condition_cols: list[str],
    *,
    category_col: str,
    order: Iterable[str] = (),
    context_cols: Iterable[str] = (),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty or category_col not in df.columns:
        return pd.DataFrame()

    contexts: list[tuple[dict[str, Any], pd.DataFrame]] = [({"condition_col": "all", "condition_level": "all"}, df)]
    for condition_col in condition_cols:
        if condition_col not in df.columns:
            continue
        for level, sub in df.groupby(condition_col, dropna=False):
            contexts.append(({"condition_col": condition_col, "condition_level": level}, sub))

    context_cols = list(context_cols)
    for context, sub in contexts:
        group_cols = context_cols + [category_col]
        subject_level = _subject_level(sub, group_cols, metrics, subject_col)
        grouped = subject_level.groupby(context_cols, dropna=False) if context_cols else [((), subject_level)]
        for context_keys, context_subject_level in grouped:
            context_keys = context_keys if isinstance(context_keys, tuple) else (context_keys,)
            context_extra = dict(zip(context_cols, context_keys)) if context_cols else {}
            present = [g for g in order if g in set(context_subject_level[category_col].dropna())]
            present += sorted([g for g in context_subject_level[category_col].dropna().unique() if g not in present], key=lambda x: str(x))
            for group_a, group_b in combinations(present, 2):
                a = context_subject_level[context_subject_level[category_col] == group_a]
                b = context_subject_level[context_subject_level[category_col] == group_b]
                for metric in metrics:
                    av = pd.to_numeric(a[metric], errors="coerce").dropna()
                    bv = pd.to_numeric(b[metric], errors="coerce").dropna()
                    rows.append(
                        {
                            **context,
                            **context_extra,
                            "group_a": group_a,
                            "group_b": group_b,
                            "comparison": f"{group_b} - {group_a}",
                            "metric": metric,
                            "n_subjects_a": int(av.size),
                            "n_subjects_b": int(bv.size),
                            "mean_a": float(av.mean()) if av.size else np.nan,
                            "mean_b": float(bv.mean()) if bv.size else np.nan,
                            "mean_difference_b_minus_a": float(bv.mean() - av.mean()) if av.size and bv.size else np.nan,
                            "cohens_d_b_minus_a": _pooled_cohens_d(av, bv),
                        }
                    )
    return pd.DataFrame(rows)


def _pairwise_between_scope_values(df: pd.DataFrame, metrics: list[str], subject_col: str, condition_cols: list[str]) -> pd.DataFrame:
    rows = _pairwise_between_categories(
        df,
        metrics,
        subject_col,
        condition_cols,
        category_col=ANALYSIS_SCOPE_VALUE_COLUMN,
        order=["all_participants", *EXPERIMENT_GROUP_ORDER, *SUBGROUP_ORDER],
        context_cols=[ANALYSIS_SCOPE_COLUMN],
    )
    if rows.empty:
        return rows
    return rows[rows[ANALYSIS_SCOPE_COLUMN] != "all"].reset_index(drop=True)


def _within_category_condition_pairs(
    df: pd.DataFrame,
    metrics: list[str],
    subject_col: str,
    condition_cols: list[str],
    *,
    category_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty or not all(c in df.columns for c in category_cols):
        return pd.DataFrame()
    for keys, group_df in df.groupby(category_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if any(pd.isna(k) for k in keys):
            continue
        base = dict(zip(category_cols, keys))
        for condition_col in condition_cols:
            if condition_col not in group_df.columns:
                continue
            levels = [x for x in group_df[condition_col].dropna().unique()]
            levels = sorted(levels, key=lambda x: str(x))
            for level_a, level_b in combinations(levels, 2):
                a = group_df[group_df[condition_col] == level_a]
                b = group_df[group_df[condition_col] == level_b]
                for metric in metrics:
                    av = pd.to_numeric(a[metric], errors="coerce").dropna()
                    bv = pd.to_numeric(b[metric], errors="coerce").dropna()
                    paired_n = 0
                    paired_diff = np.nan
                    if subject_col in group_df.columns:
                        a_subject = a.groupby(subject_col, dropna=False)[metric].mean().rename("a")
                        b_subject = b.groupby(subject_col, dropna=False)[metric].mean().rename("b")
                        paired = pd.concat([a_subject, b_subject], axis=1).dropna()
                        paired_n = int(len(paired))
                        if paired_n:
                            paired_diff = float((paired["b"] - paired["a"]).mean())
                    rows.append(
                        {
                            **base,
                            "condition_col": condition_col,
                            "level_a": level_a,
                            "level_b": level_b,
                            "comparison": f"{level_b} - {level_a}",
                            "metric": metric,
                            "n_observations_a": int(av.size),
                            "n_observations_b": int(bv.size),
                            "mean_a": float(av.mean()) if av.size else np.nan,
                            "mean_b": float(bv.mean()) if bv.size else np.nan,
                            "mean_difference_b_minus_a": float(bv.mean() - av.mean()) if av.size and bv.size else np.nan,
                            "cohens_d_b_minus_a": _pooled_cohens_d(av, bv),
                            "n_paired_subjects": paired_n,
                            "paired_mean_difference_b_minus_a": paired_diff,
                        }
                    )
    return pd.DataFrame(rows)


def compute_group_comparison_tables(
    df: pd.DataFrame,
    *,
    metric_columns: Iterable[str],
    subject_col: str = "subject_id",
    condition_cols: Iterable[str] = (),
) -> dict[str, pd.DataFrame]:
    """Compute per-experiment-group, within-group, and between-group tables."""
    empty = {
        "group_metric_summary": pd.DataFrame(),
        "group_condition_metric_summary": pd.DataFrame(),
        "within_group_condition_comparisons": pd.DataFrame(),
        "between_group_metric_comparisons": pd.DataFrame(),
    }
    if df.empty:
        return empty
    prepared = add_experiment_group_columns(df, subject_col=subject_col)
    prepared = prepared[prepared[EXPERIMENT_GROUP_COLUMN].notna()].copy()
    if prepared.empty:
        return empty
    metrics = available_numeric_metrics(prepared, metric_columns)
    if not metrics:
        return empty
    present_conditions = [c for c in condition_cols if c in prepared.columns]
    return {
        "group_metric_summary": _summary_rows(prepared, [EXPERIMENT_GROUP_COLUMN], metrics, subject_col),
        "group_condition_metric_summary": _summary_rows(prepared, [EXPERIMENT_GROUP_COLUMN] + present_conditions, metrics, subject_col) if present_conditions else pd.DataFrame(),
        "within_group_condition_comparisons": _within_category_condition_pairs(
            prepared,
            metrics,
            subject_col,
            present_conditions,
            category_cols=[EXPERIMENT_GROUP_COLUMN],
        ),
        "between_group_metric_comparisons": _pairwise_between_categories(
            prepared,
            metrics,
            subject_col,
            present_conditions,
            category_col=EXPERIMENT_GROUP_COLUMN,
            order=EXPERIMENT_GROUP_ORDER,
        ),
    }


def compute_analysis_scope_tables(
    df: pd.DataFrame,
    *,
    metric_columns: Iterable[str],
    subject_col: str = "subject_id",
    condition_cols: Iterable[str] = (),
) -> dict[str, pd.DataFrame]:
    """Compute all-participant, subgroup, and single-participant analysis tables."""
    empty = {
        "analysis_scope_metric_summary": pd.DataFrame(),
        "analysis_scope_condition_metric_summary": pd.DataFrame(),
        "within_analysis_scope_condition_comparisons": pd.DataFrame(),
        "between_analysis_scope_metric_comparisons": pd.DataFrame(),
    }
    if df.empty:
        return empty
    expanded = expand_analysis_scopes(df, subject_col=subject_col)
    if expanded.empty:
        return empty
    metrics = available_numeric_metrics(expanded, metric_columns)
    if not metrics:
        return empty
    present_conditions = [c for c in condition_cols if c in expanded.columns]
    scope_cols = [ANALYSIS_SCOPE_COLUMN, ANALYSIS_SCOPE_VALUE_COLUMN]
    return {
        "analysis_scope_metric_summary": _summary_rows(expanded, scope_cols, metrics, subject_col),
        "analysis_scope_condition_metric_summary": _summary_rows(expanded, scope_cols + present_conditions, metrics, subject_col) if present_conditions else pd.DataFrame(),
        "within_analysis_scope_condition_comparisons": _within_category_condition_pairs(
            expanded,
            metrics,
            subject_col,
            present_conditions,
            category_cols=scope_cols,
        ),
        "between_analysis_scope_metric_comparisons": _pairwise_between_scope_values(expanded, metrics, subject_col, present_conditions),
    }
