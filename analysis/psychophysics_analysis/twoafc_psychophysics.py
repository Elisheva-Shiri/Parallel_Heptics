"""Utilities for 2AFC constant-stimuli psychophysics notebooks.

The functions in this module are intentionally conservative:
- raw data folders are only read;
- all generated files are written to an explicit output folder;
- raw answer codes are converted to a canonical
  ``response_comparison_greater`` after determining which object was the
  comparison stimulus.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


STANDARD_FALLBACK = 85.0
STANDARD_ABS_TOLERANCE = 0.75
MIN_TRIALS_PER_FIT = 12
MIN_LEVELS_PER_FIT = 3


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


def save_csv(df: pd.DataFrame, output_root: Path, filename: str, index: bool = False) -> Path:
    path = output_root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def norm_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = text.replace("answares", "answers").replace("answare", "answer")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1255", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{encoding}: {exc}")
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


def discover_answer_files(data_root: Path, output_root: Path) -> pd.DataFrame:
    validate_paths(data_root, output_root)
    rows = []
    subject_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if not subject_dirs:
        raise RuntimeError(f"No subject folders found under {data_root}")
    for subject_dir in subject_dirs:
        all_csvs = sorted(subject_dir.rglob("*.csv"), key=lambda p: str(p).lower())
        csvs = [p for p in all_csvs if is_answer_csv_candidate(p)]
        scored = sorted([(score_answer_csv(p), p) for p in csvs], key=lambda t: (-t[0], str(t[1]).lower()))
        selected = scored[0][1] if scored else None
        for rank, (score, path) in enumerate(scored, start=1):
            rows.append(
                {
                    "subject_id": subject_dir.name,
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
        df.insert(0, "subject_id", row["subject_id"])
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return clean, flagged


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


def make_psychometric_input(clean: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()
    return (
        clean.groupby(group_cols + ["comparison_value"], dropna=False)
        .agg(
            n_trials=("response_comparison_greater", "size"),
            n_comparison_greater=("response_comparison_greater", "sum"),
            p_comparison_greater=("response_comparison_greater", "mean"),
            mean_rt=("reaction_time", "mean"),
            standard_value=("standard_value", "median"),
        )
        .reset_index()
        .sort_values(group_cols + ["comparison_value"])
    )


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
    z = (np.asarray(x, dtype=float) - mu) / max(scale, 1e-12)
    s = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    return lapse_low + (1 - lapse_low - lapse_high) * s


def x_at_probability(q: float, mu: float, scale: float, lapse_low: float, lapse_high: float) -> float:
    amp = 1 - lapse_low - lapse_high
    if amp <= 0:
        return np.nan
    y = (q - lapse_low) / amp
    if y <= 0 or y >= 1:
        return np.nan
    return float(mu + scale * math.log(y / (1 - y)))


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
            if hasattr(psignifit, "getThreshold"):
                try:
                    pse = float(np.ravel(psignifit.getThreshold(result, 0.5))[0])  # type: ignore[attr-defined]
                    x25 = float(np.ravel(psignifit.getThreshold(result, 0.25))[0])  # type: ignore[attr-defined]
                    x75 = float(np.ravel(psignifit.getThreshold(result, 0.75))[0])  # type: ignore[attr-defined]
                except Exception:
                    pass
            lapse = np.nan
            if isinstance(result, dict) and "Fit" in result:
                fit = np.ravel(result["Fit"])
                if np.isnan(pse) and len(fit) >= 1:
                    pse = float(fit[0])
                if len(fit) >= 3:
                    lapse = float(fit[2])
            if np.isfinite(pse):
                return {
                    "fit_method": "psignifit",
                    "pse": pse,
                    "jnd": float((x75 - x25) / 2) if np.isfinite(x25) and np.isfinite(x75) else np.nan,
                    "x25": x25,
                    "x75": x75,
                    "lapse_rate": lapse,
                    "fit_warning": "psignifit_api_threshold_extraction_partial" if not (np.isfinite(x25) and np.isfinite(x75)) else "",
                    "psignifit_result_repr": repr(result)[:1000],
                }, "psignifit_success"
        return None, "psignifit_api_not_recognized"
    except Exception as exc:
        return None, f"psignifit_failed: {exc}"


def fit_with_scipy_logistic(agg: pd.DataFrame) -> dict[str, Any]:
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
        from scipy.optimize import minimize
    except Exception as exc:
        out["fit_warning"] = ";".join(warnings + [f"scipy_unavailable: {exc}"])
        out["fit_quality"] = "failed"
        return out
    eps = 1e-9
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_range = max(float(x_max - x_min), 1.0)
    y_obs = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    mu_guess = float(x[int(np.nanargmin(np.abs(y_obs - 0.5)))]) if np.all(np.isfinite(y_obs)) else float(np.median(x))

    def unpack(params):
        mu, log_scale, lapse_low, lapse_high = params
        return float(mu), float(np.exp(log_scale)), float(lapse_low), float(lapse_high)

    def nll(params):
        mu, scale, lapse_low, lapse_high = unpack(params)
        if lapse_low + lapse_high >= 0.45:
            return 1e9
        p = np.clip(logistic4(x, mu, scale, lapse_low, lapse_high), eps, 1 - eps)
        return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))

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
    best = None
    for start in starts:
        result = minimize(nll, start, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not best.success:
        warnings.append("optimizer_failed" if best is None else f"optimizer_warning:{best.message}")
    mu, scale, lapse_low, lapse_high = unpack(best.x)
    pse = x_at_probability(0.5, mu, scale, lapse_low, lapse_high)
    x25 = x_at_probability(0.25, mu, scale, lapse_low, lapse_high)
    x75 = x_at_probability(0.75, mu, scale, lapse_low, lapse_high)
    amp = 1 - lapse_low - lapse_high
    y_pse = (0.5 - lapse_low) / amp if amp > 0 else np.nan
    slope = amp * y_pse * (1 - y_pse) / scale if np.isfinite(y_pse) and scale > 0 else np.nan
    p_sat = np.clip(y_obs, eps, 1 - eps)
    saturated_nll = -float(np.sum(k * np.log(p_sat) + (n - k) * np.log(1 - p_sat)))
    nll_value = nll(best.x)
    if not np.isfinite(pse):
        warnings.append("pse_outside_lapse_range")
    if not np.isfinite(x25) or not np.isfinite(x75):
        warnings.append("jnd_quantile_outside_lapse_range")
    if scale > 5 * x_range:
        warnings.append("very_shallow_fit")
    if lapse_low > 0.15 or lapse_high > 0.15:
        warnings.append("high_lapse_estimate")
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
            "fit_warning": ";".join(warnings),
        }
    )
    return out


def fit_one_condition(agg: pd.DataFrame, psignifit_available: bool) -> dict[str, Any]:
    scipy_fit = fit_with_scipy_logistic(agg)
    psig_fit, psig_status = fit_with_psignifit_if_possible(agg, psignifit_available)
    if psig_fit is not None and np.isfinite(psig_fit.get("pse", np.nan)):
        out = {**scipy_fit, **psig_fit}
        for key in ["mu", "scale", "lapse_low", "lapse_high"]:
            out[f"scipy_{key}"] = scipy_fit.get(key, np.nan)
        out["psignifit_status"] = psig_status
        return out
    scipy_fit["psignifit_status"] = psig_status
    return scipy_fit


def fit_conditions(agg: pd.DataFrame, group_cols: list[str], psignifit_available: bool = False) -> pd.DataFrame:
    rows = []
    for keys, g in agg.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fit = fit_one_condition(g, psignifit_available)
        for key, value in zip(group_cols, keys):
            fit[key] = value
        fit["n_trials"] = int(g["n_trials"].sum())
        fit["n_stimulus_levels"] = int(g["comparison_value"].nunique())
        fit["comparison_min"] = float(g["comparison_value"].min())
        fit["comparison_max"] = float(g["comparison_value"].max())
        rows.append(fit)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    preferred = group_cols + ["fit_method", "pse", "jnd", "x25", "x75", "slope_at_pse", "lapse_rate", "lapse_low", "lapse_high", "fit_quality", "fit_warning", "n_trials", "n_stimulus_levels", "deviance", "aic", "psignifit_status"]
    return out[[c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]]


def subject_average_psychometric(clean: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    subj = make_psychometric_input(clean, ["subject_id"] + group_cols)
    if subj.empty:
        return subj
    return (
        subj.groupby(group_cols + ["comparison_value"], dropna=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_p_comparison_greater=("p_comparison_greater", "mean"),
            sem_p_comparison_greater=("p_comparison_greater", lambda x: float(pd.Series(x).std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan),
            total_trials=("n_trials", "sum"),
            standard_value=("standard_value", "median"),
        )
        .reset_index()
        .sort_values(group_cols + ["comparison_value"])
    )


def compute_order_effects(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                "first_half_p_comparison_greater": float(first["response_comparison_greater"].mean()) if len(first) else np.nan,
                "second_half_p_comparison_greater": float(second["response_comparison_greater"].mean()) if len(second) else np.nan,
                "first_half_mean_rt": float(first["reaction_time"].mean()) if first["reaction_time"].notna().any() else np.nan,
                "second_half_mean_rt": float(second["reaction_time"].mean()) if second["reaction_time"].notna().any() else np.nan,
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
                    "mean_rt": float(gb["reaction_time"].mean()) if gb["reaction_time"].notna().any() else np.nan,
                    "n_trials": len(gb),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(binned_rows)


def predictions_from_fit_row(row: pd.Series, x_grid: np.ndarray) -> np.ndarray:
    mu, scale = row.get("mu", np.nan), row.get("scale", np.nan)
    lo, hi = row.get("lapse_low", np.nan), row.get("lapse_high", np.nan)
    if not np.isfinite(mu):
        mu, scale = row.get("scipy_mu", np.nan), row.get("scipy_scale", np.nan)
        lo, hi = row.get("scipy_lapse_low", np.nan), row.get("scipy_lapse_high", np.nan)
    if not all(np.isfinite(v) for v in [mu, scale, lo, hi]):
        return np.full_like(x_grid, np.nan, dtype=float)
    return logistic4(x_grid, float(mu), float(scale), float(lo), float(hi))


def save_subject_level_outputs(output_root: Path, clean: pd.DataFrame, flagged: pd.DataFrame, subject_fits: pd.DataFrame, subject_pooled_fits: pd.DataFrame) -> None:
    subjects_dir = output_root / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    for subject, g in clean.groupby("subject_id", dropna=False):
        sdir = subjects_dir / sanitize_name(subject)
        sdir.mkdir(parents=True, exist_ok=True)
        g.to_csv(sdir / "clean_trials.csv", index=False)
        if not flagged.empty:
            flagged[flagged["subject_id"] == subject].to_csv(sdir / "flagged_trials.csv", index=False)
        if not subject_fits.empty:
            subject_fits[subject_fits["subject_id"] == subject].to_csv(sdir / "pse_jnd_by_finger.csv", index=False)
        if not subject_pooled_fits.empty:
            subject_pooled_fits[subject_pooled_fits["subject_id"] == subject].to_csv(sdir / "pse_jnd_pooled.csv", index=False)


def analysis_manifest(output_root: Path) -> pd.DataFrame:
    required = [
        "file_discovery_summary.csv",
        "combined_raw_imported_data.csv",
        "combined_clean_trials.csv",
        "combined_flagged_trials.csv",
        "qc_summary.csv",
        "psychometric_input_by_subject_finger.csv",
        "pse_jnd_by_subject_finger.csv",
        "pse_jnd_subject_pooled.csv",
        "pse_jnd_group_by_finger.csv",
        "pse_jnd_group_all_pooled.csv",
        "order_effects_summary.csv",
    ]
    return pd.DataFrame({"output": required, "exists": [(output_root / f).exists() for f in required], "path": [str(output_root / f) for f in required]})


def plot_fit_curve(agg: pd.DataFrame, fit_row: pd.Series, title: str, out_path: Path, fig_dpi: int = 160):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(
        agg["comparison_value"],
        agg["p_comparison_greater"],
        s=30 + 8 * agg["n_trials"],
        alpha=0.85,
        label="Observed",
    )
    if len(agg) >= 2:
        x_grid = np.linspace(float(agg["comparison_value"].min()), float(agg["comparison_value"].max()), 300)
        y_grid = predictions_from_fit_row(fit_row, x_grid)
        if np.isfinite(y_grid).any():
            ax.plot(x_grid, y_grid, linewidth=2, label="Fit")
    pse = fit_row.get("pse", np.nan)
    if np.isfinite(pse):
        ax.axvline(float(pse), color="tab:red", linestyle="--", linewidth=1.5, label=f"PSE={pse:.2f}")
    if "standard_value" in agg and agg["standard_value"].notna().any():
        std = float(agg["standard_value"].median())
        ax.axvline(std, color="black", linestyle=":", linewidth=1, label=f"standard={std:g}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Comparison stimulus value (physical units)")
    ax.set_ylabel("P(response comparison greater)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=fig_dpi)
    return fig, ax


def save_all_figures(
    output_root: Path,
    clean: pd.DataFrame,
    qc_summary: pd.DataFrame,
    psychometric_input_by_subject_finger: pd.DataFrame,
    pse_jnd_by_subject_finger: pd.DataFrame,
    psychometric_input_group_by_finger: pd.DataFrame,
    pse_jnd_group_by_finger: pd.DataFrame,
    psychometric_input_group_all_pooled: pd.DataFrame,
    pse_jnd_group_all_pooled: pd.DataFrame,
    order_effects_binned: pd.DataFrame,
    fig_dpi: int = 160,
) -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore
        sns.set_theme(style="whitegrid")
    except Exception:  # pragma: no cover
        sns = None

    paths: list[Path] = []
    fig_root = output_root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    if not psychometric_input_by_subject_finger.empty and not pse_jnd_by_subject_finger.empty:
        for _, fit_row in pse_jnd_by_subject_finger.iterrows():
            subject, finger = fit_row["subject_id"], fit_row["finger_condition"]
            agg = psychometric_input_by_subject_finger[
                (psychometric_input_by_subject_finger["subject_id"] == subject)
                & (psychometric_input_by_subject_finger["finger_condition"] == finger)
            ]
            out = fig_root / "subject_finger_curves" / sanitize_name(subject) / f"psychometric_{sanitize_name(subject)}_{sanitize_name(finger)}.png"
            fig, _ = plot_fit_curve(agg, fit_row, f"Subject {subject} – finger {finger}", out, fig_dpi)
            plt.close(fig)
            paths.append(out)

    if not psychometric_input_group_by_finger.empty and not pse_jnd_group_by_finger.empty:
        for _, fit_row in pse_jnd_group_by_finger.iterrows():
            finger = fit_row["finger_condition"]
            agg = psychometric_input_group_by_finger[psychometric_input_group_by_finger["finger_condition"] == finger]
            out = fig_root / "group_curves" / f"group_finger_{sanitize_name(finger)}.png"
            fig, _ = plot_fit_curve(agg, fit_row, f"Group pooled – finger {finger}", out, fig_dpi)
            plt.close(fig)
            paths.append(out)

    if not psychometric_input_group_all_pooled.empty and not pse_jnd_group_all_pooled.empty:
        out = fig_root / "group_curves" / "group_all_pooled.png"
        fig, _ = plot_fit_curve(psychometric_input_group_all_pooled, pse_jnd_group_all_pooled.iloc[0], "Group all-pooled", out, fig_dpi)
        plt.close(fig)
        paths.append(out)

    if not pse_jnd_by_subject_finger.empty:
        for metric, ylabel, filename in [("pse", "PSE", "pse_per_subject_and_finger.png"), ("jnd", "JND", "jnd_per_subject_and_finger.png")]:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            plot_df = pse_jnd_by_subject_finger.copy()
            if sns is not None:
                sns.stripplot(data=plot_df, x="finger_condition", y=metric, hue="subject_id", dodge=True, ax=ax)
                sns.pointplot(data=plot_df, x="finger_condition", y=metric, color="black", errorbar="sd", ax=ax)
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
            else:
                for subject, g in plot_df.groupby("subject_id"):
                    ax.plot(g["finger_condition"], g[metric], marker="o", alpha=0.6, label=str(subject))
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Finger condition")
            ax.set_title(f"{ylabel} per subject and finger")
            fig.tight_layout()
            out = fig_root / filename
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

    if not order_effects_binned.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        if sns is not None:
            sns.lineplot(data=order_effects_binned, x="mean_global_trial_order", y="p_comparison_greater", hue="finger_condition", marker="o", ax=ax)
        else:
            for finger, g in order_effects_binned.groupby("finger_condition"):
                ax.plot(g["mean_global_trial_order"], g["p_comparison_greater"], marker="o", label=str(finger))
            ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Global trial order")
        ax.set_ylabel("P(response comparison greater)")
        ax.set_title("Order/fatigue trend by finger")
        fig.tight_layout()
        out = fig_root / "order_effects.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

    if not clean.empty:
        side = clean.groupby(["subject_id", "finger_condition"]).agg(
            p_chose_object2=("answer_chose_object_2", "mean"),
            p_standard_on_object2=("standard_side", lambda x: float((x == "object_2").mean())),
            n_trials=("answer_chose_object_2", "size"),
        ).reset_index()
        side.to_csv(output_root / "side_bias_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        if sns is not None:
            sns.barplot(data=side, x="finger_condition", y="p_chose_object2", errorbar="sd", ax=ax)
            sns.stripplot(data=side, x="finger_condition", y="p_chose_object2", color="black", alpha=0.6, ax=ax)
        else:
            ax.bar(side["finger_condition"].astype(str), side["p_chose_object2"])
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(chose object 2)")
        ax.set_xlabel("Finger condition")
        ax.set_title("Side bias")
        fig.tight_layout()
        out = fig_root / "side_bias.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

    if not qc_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        if sns is not None:
            sns.scatterplot(data=qc_summary, x="n_stimulus_levels", y="n_clean_trials", hue="finger_condition", style="qc_warnings", ax=ax)
        else:
            ax.scatter(qc_summary["n_stimulus_levels"], qc_summary["n_clean_trials"])
        ax.set_title("QC: trials and levels")
        ax.set_xlabel("Number of stimulus levels")
        ax.set_ylabel("Clean trials")
        ax = axes[1]
        if sns is not None:
            sns.barplot(data=qc_summary, x="finger_condition", y="apparent_error_rate", errorbar="sd", ax=ax)
        else:
            ax.bar(qc_summary["finger_condition"].astype(str), qc_summary["apparent_error_rate"])
        ax.axhline(0.35, color="red", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title("QC: apparent error rate")
        fig.tight_layout()
        out = fig_root / "qc_summary_plots.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

    return paths
