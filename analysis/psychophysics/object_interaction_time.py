"""Object-interaction time per trial, measured from the per-pair ``tracking.csv``.

Why this exists
---------------
``answers.csv`` only stores ``time_to_answer`` (the QUESTION prompt -> answer
press), so it says nothing about how long the participant actually explored
the two virtual objects. The backend also writes ``pair_XXX/tracking.csv`` at
every hand-position update while the objects are presented, with columns
``timestamp``, ``interacting`` (the participant's grab/interaction state) and
``stiffness`` (the currently presented object). From that file we can measure,
for every trial and every object:

- ``presentation_time_s``  - how long the object was on screen (first -> last
  tracking sample of that stiffness segment);
- ``interaction_time_s``   - the summed time the ``interacting`` flag was ON,
  i.e. the participant was actively engaging the object;
- ``n_interaction_bouts``  - how many separate ON episodes there were;
- ``first_interaction_latency_s`` - delay from object onset to the first ON.

The per-trial totals are then joined to the cleaned psychophysics trial table
so the same 40-participant, familiarisation-excluded trial set is used, and
the trend over the session (learning / fatigue proxy) plus the relation with
success are summarised. ``trial_completion_time_s`` is the exploration span
plus ``time_to_answer``, i.e. a real "trial completion" measure.

The module is deliberately independent of ``twoafc_psychophysics.py`` (which
is regression-locked against golden outputs); it only *reads* its tables.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

TRACKING_USECOLS = ["timestamp", "interacting", "stiffness"]
# A gap between consecutive tracking samples longer than this is treated as a
# pause (moderator pause / dropped frames) and is not counted as interaction.
MAX_SAMPLE_GAP_S = 0.5

TIME_METRICS = [
    "interaction_time_s",
    "presentation_time_s",
    "trial_completion_time_s",
]
METRIC_LABELS = {
    "interaction_time_s": "Object interaction time (s)",
    "presentation_time_s": "Object presentation time (s)",
    "trial_completion_time_s": "Trial completion time (s)",
    "reaction_time": "Time to answer (s)",
    "correct_response": "Success rate",
    "interaction_fraction": "Interaction / presentation",
    "n_interaction_bouts": "Interaction bouts per trial",
}

# Okabe-Ito colour-blind-safe categorical colours (fixed order, never cycled).
_C_INTERACTION = "#0072B2"
_C_PRESENTATION = "#E69F00"
_C_COMPLETION = "#009E73"
_C_SUCCESS = "#000000"


# --------------------------------------------------------------------------- #
# Small statistical helpers (self-contained on purpose)
# --------------------------------------------------------------------------- #
def _mean_ci95(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(x.mean())
    if len(x) < 2:
        return m, np.nan, np.nan
    se = x.std(ddof=1) / math.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h


def _one_sample_tests(values: pd.Series | np.ndarray) -> dict[str, float]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    mean, lo, hi = _mean_ci95(x)
    out = {
        "n": int(len(x)),
        "mean": mean,
        "median": float(np.median(x)) if len(x) else np.nan,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "t_p_value": np.nan,
        "wilcoxon_p_value": np.nan,
        "n_negative": int((x < 0).sum()),
        "n_positive": int((x > 0).sum()),
    }
    if len(x) >= 2 and np.nanstd(x) > 0:
        out["t_p_value"] = float(stats.ttest_1samp(x, 0.0).pvalue)
        nz = x[x != 0]
        if len(nz) >= 5:
            out["wilcoxon_p_value"] = float(stats.wilcoxon(nz).pvalue)
    return out


def _linregress_slope(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    ok = xx.notna() & yy.notna()
    if ok.sum() < 3 or xx[ok].nunique() < 2:
        return np.nan, np.nan
    fit = stats.linregress(xx[ok].to_numpy(dtype=float), yy[ok].to_numpy(dtype=float))
    return float(fit.slope), float(fit.pvalue)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    ok = xx.notna() & yy.notna()
    if ok.sum() < 3 or xx[ok].nunique() < 2 or yy[ok].nunique() < 2:
        return np.nan, np.nan
    r, p = stats.spearmanr(xx[ok], yy[ok])
    return float(r), float(p)


def _direction_from_ci(lo: float, hi: float, *, faster: str, slower: str, flat: str = "no clear change") -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unknown"
    if hi < 0:
        return faster
    if lo > 0:
        return slower
    return flat


# --------------------------------------------------------------------------- #
# Reading tracking.csv
# --------------------------------------------------------------------------- #
def session_folder_map(file_discovery_summary: pd.DataFrame) -> dict[str, Path]:
    """Map ``subject_id`` -> session folder (the folder holding answers.csv)."""
    fd = file_discovery_summary
    if "selected" in fd.columns:
        fd = fd[fd["selected"].astype(bool)]
    out: dict[str, Path] = {}
    for _, row in fd.iterrows():
        sid = str(row.get("canonical_subject_id") or row.get("subject_id"))
        src = row.get("source_file")
        if isinstance(src, str) and src:
            out[sid] = Path(src).parent
    return out


def _pair_folder(session_dir: Path, pair_number: int) -> Path:
    return Path(session_dir) / f"pair_{int(pair_number):03d}"


def measure_pair_tracking(
    tracking_path: Path,
    *,
    max_gap_s: float = MAX_SAMPLE_GAP_S,
    expected_stiffness: Optional[tuple[float, ...]] = None,
) -> dict[str, Any]:
    """Measure presentation/interaction time for one ``tracking.csv``.

    Objects are the distinct ``stiffness`` values in order of appearance (the
    two stimuli of a pair are never equal in this protocol). Samples with no
    object (stiffness 0) and gaps longer than ``max_gap_s`` (pauses) are not
    counted. Returns pair-level totals plus per-object values
    (``object_1_*``, ``object_2_*``).
    """
    result: dict[str, Any] = {
        "tracking_found": False,
        "n_tracking_samples": 0,
        "n_samples_no_object": 0,
        "n_objects_detected": 0,
        "n_stiffness_segments": 0,
        "pair_span_s": np.nan,
        "presentation_time_s": np.nan,
        "interaction_time_s": np.nan,
        "n_interaction_bouts": np.nan,
        "first_interaction_latency_s": np.nan,
        "median_sample_interval_s": np.nan,
        "tracking_first_timestamp": pd.NaT,
        "tracking_last_timestamp": pd.NaT,
    }
    for k in (1, 2):
        result[f"object_{k}_stiffness"] = np.nan
        result[f"object_{k}_presentation_time_s"] = np.nan
        result[f"object_{k}_interaction_time_s"] = np.nan
        result[f"object_{k}_n_interaction_bouts"] = np.nan
        result[f"object_{k}_first_interaction_latency_s"] = np.nan

    tracking_path = Path(tracking_path)
    if not tracking_path.exists():
        return result
    try:
        df = pd.read_csv(tracking_path, usecols=TRACKING_USECOLS)
    except Exception:
        return result
    if df.empty:
        result["tracking_found"] = True
        return result

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    ok = ts.notna()
    df = df.loc[ok].copy()
    ts = ts[ok]
    if df.empty:
        result["tracking_found"] = True
        return result
    t_all = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    inter_all = df["interacting"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).to_numpy()
    stiff_all = pd.to_numeric(df["stiffness"], errors="coerce").to_numpy()

    # Time attributed to each sample = gap to the next sample, capped so that
    # moderator pauses / restarts are not counted as presentation or interaction.
    dt_all = np.diff(t_all, append=t_all[-1])
    dt_all = np.where(np.isfinite(dt_all), np.clip(dt_all, 0.0, max_gap_s), 0.0)

    # Samples with no object on screen (stiffness 0 / NaN, e.g. a resumed pair
    # before the first object is set) are excluded. If the expected pair values
    # are known, keep only samples showing one of them.
    if expected_stiffness:
        keep = np.isin(stiff_all, np.asarray(list(expected_stiffness), dtype=float))
    else:
        keep = np.isfinite(stiff_all) & (stiff_all > 0)
    result.update(
        {
            "tracking_found": True,
            "n_tracking_samples": int(len(df)),
            "n_samples_no_object": int((~keep).sum()),
            "pair_span_s": float(t_all[-1] - t_all[0]),
            "median_sample_interval_s": float(np.median(np.diff(t_all))) if len(t_all) > 1 else np.nan,
            "tracking_first_timestamp": ts.iloc[0],
            "tracking_last_timestamp": ts.iloc[-1],
        }
    )
    if not keep.any():
        result["n_objects_detected"] = 0
        return result
    t = t_all[keep]
    inter = inter_all[keep]
    stiff = stiff_all[keep]
    dt = dt_all[keep]

    # Objects = distinct stiffness values in order of first appearance. A pair
    # that was restarted (same value shown twice, separated by another value)
    # is merged back into one object per value.
    seg_change = np.concatenate([[True], stiff[1:] != stiff[:-1]])
    n_segments = int(seg_change.sum())
    values_in_order = list(dict.fromkeys(stiff.tolist()))
    n_obj = len(values_in_order)

    result.update(
        {
            "n_objects_detected": n_obj,
            "n_stiffness_segments": n_segments,
            "presentation_time_s": float(dt.sum()),
            "interaction_time_s": float(dt[inter].sum()),
            "n_interaction_bouts": int(((~np.concatenate([[False], inter[:-1]])) & inter).sum()),
            "first_interaction_latency_s": float(t[inter][0] - t[0]) if inter.any() else np.nan,
        }
    )
    for k, value in enumerate(values_in_order[:2]):
        m = stiff == value
        tk = t[m]
        ik = inter[m]
        bouts = int(((~np.concatenate([[False], ik[:-1]])) & ik).sum())
        result[f"object_{k + 1}_stiffness"] = float(value)
        result[f"object_{k + 1}_presentation_time_s"] = float(dt[m].sum())
        result[f"object_{k + 1}_interaction_time_s"] = float(dt[m][ik].sum())
        result[f"object_{k + 1}_n_interaction_bouts"] = bouts
        result[f"object_{k + 1}_first_interaction_latency_s"] = float(tk[ik][0] - tk[0]) if ik.any() else np.nan
    return result


# --------------------------------------------------------------------------- #
# Trial table
# --------------------------------------------------------------------------- #
TRIAL_KEEP_COLUMNS = [
    "subject_id",
    "trial_index_raw",
    "global_trial_order",
    "trial_order_fraction",
    "elapsed_minutes",
    "block_number_inferred",
    "finger_condition",
    "finger_appearance_order",
    "comparison_value",
    "standard_value",
    "abs_stiffness_delta",
    "object_1_value",
    "object_2_value",
    "correct_response",
    "reaction_time",
    "experiment_group",
    "workspace_setup",
    "subject_group",
]


def compute_object_interaction_trials(
    success_trials: pd.DataFrame,
    file_discovery_summary: pd.DataFrame,
    *,
    n_jobs: int = 8,
    max_gap_s: float = MAX_SAMPLE_GAP_S,
) -> pd.DataFrame:
    """Join per-trial tracking measurements onto the cleaned trial table."""
    keep = [c for c in TRIAL_KEEP_COLUMNS if c in success_trials.columns]
    trials = success_trials[keep].copy()
    trials["subject_id"] = trials["subject_id"].astype(str)
    trials["trial_index_raw"] = pd.to_numeric(trials["trial_index_raw"], errors="coerce")
    trials = trials.dropna(subset=["trial_index_raw"])
    trials["trial_index_raw"] = trials["trial_index_raw"].astype(int)

    folders = session_folder_map(file_discovery_summary)
    trials["session_folder"] = trials["subject_id"].map(lambda s: str(folders.get(s, "")))
    paths = [
        _pair_folder(Path(f), p) / "tracking.csv" if f else Path("__missing__")
        for f, p in zip(trials["session_folder"], trials["trial_index_raw"])
    ]
    if {"object_1_value", "object_2_value"}.issubset(trials.columns):
        v1 = pd.to_numeric(trials["object_1_value"], errors="coerce")
        v2 = pd.to_numeric(trials["object_2_value"], errors="coerce")
        expected = [
            tuple(v for v in (a, b) if np.isfinite(v)) or None for a, b in zip(v1, v2)
        ]
    else:
        expected = [None] * len(trials)

    def _measure(args: tuple[Path, Optional[tuple[float, ...]]]) -> dict[str, Any]:
        p, exp = args
        return measure_pair_tracking(p, max_gap_s=max_gap_s, expected_stiffness=exp)

    jobs = list(zip(paths, expected))
    if n_jobs and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            measured = list(ex.map(_measure, jobs))
    else:
        measured = [_measure(j) for j in jobs]

    meas = pd.DataFrame(measured, index=trials.index)
    out = pd.concat([trials, meas], axis=1)
    out["reaction_time"] = pd.to_numeric(out.get("reaction_time"), errors="coerce")
    out["trial_completion_time_s"] = out["presentation_time_s"] + out["reaction_time"]
    out["interaction_fraction"] = np.where(
        out["presentation_time_s"] > 0, out["interaction_time_s"] / out["presentation_time_s"], np.nan
    )
    # Which object was the comparison / standard, for a per-stimulus view.
    if {"object_1_value", "comparison_value"}.issubset(out.columns):
        comp_is_1 = pd.to_numeric(out["object_1_value"], errors="coerce") == pd.to_numeric(out["comparison_value"], errors="coerce")
        out["comparison_interaction_time_s"] = np.where(comp_is_1, out["object_1_interaction_time_s"], out["object_2_interaction_time_s"])
        out["standard_interaction_time_s"] = np.where(comp_is_1, out["object_2_interaction_time_s"], out["object_1_interaction_time_s"])
    out = out.sort_values(["subject_id", "trial_index_raw"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _add_order_bins(trials: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    out = trials.copy()
    frac = pd.to_numeric(out["trial_order_fraction"], errors="coerce").clip(0, 1)
    out["order_bin"] = np.minimum((frac * n_bins).astype(int), n_bins - 1) + 1
    return out


def summarize_object_interaction(trials: pd.DataFrame, *, n_order_bins: int = 8) -> dict[str, Any]:
    """Per-subject trends, group tests, order bins, finger table, success relation."""
    t = trials.copy()
    t = t[t["tracking_found"].astype(bool) & t["presentation_time_s"].notna()].copy()
    t["correct_response"] = pd.to_numeric(t["correct_response"], errors="coerce")
    t["_half"] = np.where(pd.to_numeric(t["trial_order_fraction"], errors="coerce") < 0.5, "first", "second")

    metrics = TIME_METRICS + ["reaction_time", "interaction_fraction", "n_interaction_bouts", "correct_response"]

    # ---- per-subject summary -------------------------------------------------
    subj_rows = []
    for sid, g in t.groupby("subject_id", sort=False):
        row: dict[str, Any] = {
            "subject_id": sid,
            "experiment_group": g["experiment_group"].iloc[0] if "experiment_group" in g else np.nan,
            "n_trials": int(len(g)),
            "n_trials_missing_tracking": int(trials.loc[trials["subject_id"] == sid, "tracking_found"].eq(False).sum()),
            "n_trials_not_two_objects": int((g["n_objects_detected"] != 2).sum()),
        }
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(v.mean())
            row[f"{m}_first_half"] = float(v[g["_half"] == "first"].mean())
            row[f"{m}_second_half"] = float(v[g["_half"] == "second"].mean())
            row[f"{m}_second_minus_first"] = row[f"{m}_second_half"] - row[f"{m}_first_half"]
            slope, p = _linregress_slope(g["trial_order_fraction"], v)
            row[f"{m}_slope_per_session"] = slope
            row[f"{m}_slope_p_value"] = p
        rho, p = _spearman(g["interaction_time_s"], g["correct_response"])
        row["success_vs_interaction_time_spearman_r"] = rho
        row["success_vs_interaction_time_spearman_p"] = p
        subj_rows.append(row)
    subject_summary = pd.DataFrame(subj_rows)

    # ---- group tests across subjects ----------------------------------------
    test_rows = []
    for m in metrics:
        first = subject_summary[f"{m}_first_half"]
        second = subject_summary[f"{m}_second_half"]
        diff = _one_sample_tests(second - first)
        slope = _one_sample_tests(subject_summary[f"{m}_slope_per_session"])
        mean_first = float(first.mean())
        mean_second = float(second.mean())
        pct = (mean_second - mean_first) / mean_first * 100.0 if mean_first not in (0.0, np.nan) and np.isfinite(mean_first) and mean_first != 0 else np.nan
        if m == "correct_response":
            direction = _direction_from_ci(diff["ci95_lower"], diff["ci95_upper"], faster="decreases over session", slower="increases over session")
        else:
            direction = _direction_from_ci(diff["ci95_lower"], diff["ci95_upper"], faster="shorter over session", slower="longer over session")
        test_rows.append(
            {
                "metric": m,
                "label": METRIC_LABELS.get(m, m),
                "n_subjects": diff["n"],
                "mean_first_half": mean_first,
                "mean_second_half": mean_second,
                "percent_change_second_vs_first": pct,
                "second_minus_first_mean": diff["mean"],
                "second_minus_first_ci95_lower": diff["ci95_lower"],
                "second_minus_first_ci95_upper": diff["ci95_upper"],
                "second_minus_first_t_p_value": diff["t_p_value"],
                "second_minus_first_wilcoxon_p_value": diff["wilcoxon_p_value"],
                "n_subjects_decreasing": diff["n_negative"],
                "n_subjects_increasing": diff["n_positive"],
                "slope_per_session_mean": slope["mean"],
                "slope_per_session_ci95_lower": slope["ci95_lower"],
                "slope_per_session_ci95_upper": slope["ci95_upper"],
                "slope_per_session_t_p_value": slope["t_p_value"],
                "slope_per_session_wilcoxon_p_value": slope["wilcoxon_p_value"],
                "n_subjects_negative_slope": slope["n_negative"],
                "n_subjects_positive_slope": slope["n_positive"],
                "interpretation": direction,
            }
        )
    group_tests = pd.DataFrame(test_rows)

    # ---- by experiment group (L vs N) ----------------------------------------
    grp_rows = []
    if "experiment_group" in subject_summary.columns:
        for grp, gs in subject_summary.groupby("experiment_group", dropna=False):
            for m in TIME_METRICS + ["correct_response"]:
                diff = _one_sample_tests(gs[f"{m}_second_half"] - gs[f"{m}_first_half"])
                grp_rows.append(
                    {
                        "experiment_group": grp,
                        "metric": m,
                        "n_subjects": diff["n"],
                        "mean_first_half": float(gs[f"{m}_first_half"].mean()),
                        "mean_second_half": float(gs[f"{m}_second_half"].mean()),
                        "second_minus_first_mean": diff["mean"],
                        "second_minus_first_ci95_lower": diff["ci95_lower"],
                        "second_minus_first_ci95_upper": diff["ci95_upper"],
                        "second_minus_first_t_p_value": diff["t_p_value"],
                        "second_minus_first_wilcoxon_p_value": diff["wilcoxon_p_value"],
                    }
                )
    group_tests_by_experiment_group = pd.DataFrame(grp_rows)

    # ---- order bins (subject-mean per bin, then across subjects) ------------
    tb = _add_order_bins(t, n_order_bins)
    per_subj_bin = tb.groupby(["subject_id", "order_bin"], sort=True)[TIME_METRICS + ["reaction_time", "correct_response"]].mean().reset_index()
    bin_rows = []
    for b, g in per_subj_bin.groupby("order_bin", sort=True):
        row = {"order_bin": int(b), "n_subjects": int(g["subject_id"].nunique()), "mean_trial_order_fraction": float(tb.loc[tb["order_bin"] == b, "trial_order_fraction"].mean())}
        for m in TIME_METRICS + ["reaction_time", "correct_response"]:
            mean, lo, hi = _mean_ci95(g[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_ci95_lower"] = lo
            row[f"{m}_ci95_upper"] = hi
        bin_rows.append(row)
    order_bins = pd.DataFrame(bin_rows)

    # ---- within-finger slopes, like finger_time_slope_summary --------------
    finger_rows = []
    if "finger_condition" in t.columns:
        tf = t.sort_values(["subject_id", "finger_condition", "global_trial_order"]).copy()
        tf["_wf_order"] = tf.groupby(["subject_id", "finger_condition"]).cumcount()
        n_wf = tf.groupby(["subject_id", "finger_condition"])["_wf_order"].transform("max").astype(float)
        tf["_wf_fraction"] = np.where(n_wf > 0, tf["_wf_order"] / n_wf, 0.0)
        sf_rows = []
        for (sid, fing), g in tf.groupby(["subject_id", "finger_condition"], sort=False):
            r: dict[str, Any] = {"subject_id": sid, "finger_condition": fing, "n_trials": int(len(g))}
            for m in TIME_METRICS + ["correct_response"]:
                r[f"{m}_mean"] = float(pd.to_numeric(g[m], errors="coerce").mean())
                r[f"{m}_within_finger_slope"], _ = _linregress_slope(g["_wf_fraction"], g[m])
            sf_rows.append(r)
        subject_finger = pd.DataFrame(sf_rows)
        for fing, g in subject_finger.groupby("finger_condition", sort=True):
            row = {"finger_condition": fing, "n_subjects": int(g["subject_id"].nunique())}
            for m in TIME_METRICS + ["correct_response"]:
                row[f"{m}_mean"] = float(g[f"{m}_mean"].mean())
                mean, lo, hi = _mean_ci95(g[f"{m}_within_finger_slope"])
                row[f"{m}_slope_mean"] = mean
                row[f"{m}_slope_ci95_lower"] = lo
                row[f"{m}_slope_ci95_upper"] = hi
                if m == "correct_response":
                    row[f"{m}_slope_interpretation"] = _direction_from_ci(lo, hi, faster="decreases over time", slower="increases over time")
                else:
                    row[f"{m}_slope_interpretation"] = _direction_from_ci(lo, hi, faster="shorter over time", slower="longer over time")
            finger_rows.append(row)
    else:
        subject_finger = pd.DataFrame()
    finger_summary = pd.DataFrame(finger_rows)

    # ---- success vs interaction time (within-subject quartiles) -------------
    q_rows = []
    for sid, g in t.groupby("subject_id", sort=False):
        v = pd.to_numeric(g["interaction_time_s"], errors="coerce")
        if v.notna().sum() < 8 or v.nunique() < 4:
            continue
        q = pd.qcut(v.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        for qq, gg in g.groupby(q):
            q_rows.append(
                {
                    "subject_id": sid,
                    "interaction_time_quartile": int(qq),
                    "mean_interaction_time_s": float(pd.to_numeric(gg["interaction_time_s"], errors="coerce").mean()),
                    "success_rate": float(gg["correct_response"].mean()),
                    "n_trials": int(len(gg)),
                }
            )
    per_subject_quartiles = pd.DataFrame(q_rows)
    sq_rows = []
    if not per_subject_quartiles.empty:
        for qq, g in per_subject_quartiles.groupby("interaction_time_quartile", sort=True):
            mean, lo, hi = _mean_ci95(g["success_rate"])
            sq_rows.append(
                {
                    "interaction_time_quartile": int(qq),
                    "n_subjects": int(g["subject_id"].nunique()),
                    "mean_interaction_time_s": float(g["mean_interaction_time_s"].mean()),
                    "success_rate_mean": mean,
                    "success_rate_ci95_lower": lo,
                    "success_rate_ci95_upper": hi,
                }
            )
    success_by_interaction_quartile = pd.DataFrame(sq_rows)
    rho_tests = _one_sample_tests(subject_summary["success_vs_interaction_time_spearman_r"])

    # ---- interaction time by stimulus difficulty ----------------------------
    diff_rows = []
    if "abs_stiffness_delta" in t.columns:
        per = t.groupby(["subject_id", "abs_stiffness_delta"])[["interaction_time_s", "presentation_time_s", "trial_completion_time_s", "correct_response"]].mean().reset_index()
        for d, g in per.groupby("abs_stiffness_delta", sort=True):
            row = {"abs_stiffness_delta": float(d), "n_subjects": int(g["subject_id"].nunique())}
            for m in ["interaction_time_s", "presentation_time_s", "trial_completion_time_s", "correct_response"]:
                mean, lo, hi = _mean_ci95(g[m])
                row[f"{m}_mean"] = mean
                row[f"{m}_ci95_lower"] = lo
                row[f"{m}_ci95_upper"] = hi
            diff_rows.append(row)
    by_stiffness_delta = pd.DataFrame(diff_rows)

    # ---- data-quality summary -----------------------------------------------
    quality = pd.DataFrame(
        [
            {
                "n_trials_in_input": int(len(trials)),
                "n_trials_with_tracking": int(trials["tracking_found"].astype(bool).sum()),
                "n_trials_without_tracking": int((~trials["tracking_found"].astype(bool)).sum()),
                "n_trials_not_two_objects": int((trials["n_objects_detected"] != 2).sum()),
                "n_subjects": int(t["subject_id"].nunique()),
                "median_sample_interval_s": float(t["median_sample_interval_s"].median()),
                "grand_mean_interaction_time_s": float(t["interaction_time_s"].mean()),
                "grand_mean_presentation_time_s": float(t["presentation_time_s"].mean()),
                "grand_mean_trial_completion_time_s": float(t["trial_completion_time_s"].mean()),
                "grand_mean_interaction_fraction": float(t["interaction_fraction"].mean()),
            }
        ]
    )

    return {
        "trials": t.drop(columns=["_half"]),
        "subject_summary": subject_summary,
        "group_tests": group_tests,
        "group_tests_by_experiment_group": group_tests_by_experiment_group,
        "order_bins": order_bins,
        "subject_finger_summary": subject_finger,
        "finger_summary": finger_summary,
        "success_by_interaction_quartile": success_by_interaction_quartile,
        "success_vs_interaction_spearman_group": pd.DataFrame([rho_tests]),
        "by_stiffness_delta": by_stiffness_delta,
        "quality": quality,
    }


# --------------------------------------------------------------------------- #
# Bottom line
# --------------------------------------------------------------------------- #
def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def bottom_line_text(summary: dict[str, Any]) -> str:
    gt = summary["group_tests"].set_index("metric")
    q = summary["quality"].iloc[0]
    lines = []
    lines.append(
        f"Object-interaction time measured from tracking.csv for {int(q['n_subjects'])} participants, "
        f"{int(q['n_trials_with_tracking'])}/{int(q['n_trials_in_input'])} trials with tracking "
        f"({int(q['n_trials_without_tracking'])} missing, {int(q['n_trials_not_two_objects'])} without exactly two objects)."
    )
    lines.append(
        f"On average a trial had {q['grand_mean_presentation_time_s']:.1f} s of object presentation, of which "
        f"{q['grand_mean_interaction_time_s']:.1f} s ({q['grand_mean_interaction_fraction'] * 100:.0f}%) was active interaction; "
        f"trial completion (exploration + answer) averaged {q['grand_mean_trial_completion_time_s']:.1f} s."
    )
    for m in ["interaction_time_s", "presentation_time_s", "trial_completion_time_s", "reaction_time"]:
        if m not in gt.index:
            continue
        r = gt.loc[m]
        lines.append(
            f"- {r['label']}: {r['mean_first_half']:.2f} s -> {r['mean_second_half']:.2f} s (first vs second half; "
            f"{r['percent_change_second_vs_first']:+.0f}%), paired diff {r['second_minus_first_mean']:+.2f} s "
            f"[95% CI {r['second_minus_first_ci95_lower']:+.2f}, {r['second_minus_first_ci95_upper']:+.2f}], "
            f"{_fmt_p(r['second_minus_first_t_p_value'])} (Wilcoxon {_fmt_p(r['second_minus_first_wilcoxon_p_value'])}); "
            f"{int(r['n_subjects_decreasing'])}/{int(r['n_subjects'])} participants shorter -> {r['interpretation']}."
        )
    if "correct_response" in gt.index:
        r = gt.loc["correct_response"]
        lines.append(
            f"- Success rate: {r['mean_first_half'] * 100:.1f}% -> {r['mean_second_half'] * 100:.1f}%, paired diff "
            f"{r['second_minus_first_mean'] * 100:+.1f} pp [95% CI {r['second_minus_first_ci95_lower'] * 100:+.1f}, "
            f"{r['second_minus_first_ci95_upper'] * 100:+.1f}], {_fmt_p(r['second_minus_first_t_p_value'])} -> {r['interpretation']}."
        )
    sp = summary["success_vs_interaction_spearman_group"].iloc[0]
    lines.append(
        f"- Within participants, success vs interaction time: mean Spearman rho={sp['mean']:+.3f} "
        f"[95% CI {sp['ci95_lower']:+.3f}, {sp['ci95_upper']:+.3f}], {_fmt_p(sp['t_p_value'])}."
    )
    fs = summary.get("finger_summary", pd.DataFrame())
    if not fs.empty:
        fingers = ", ".join(
            f"{r['finger_condition']}: {r['interaction_time_s_slope_interpretation']}" for _, r in fs.iterrows()
        )
        lines.append(f"- Within-finger interaction-time trend by finger: {fingers}.")
    # Paper sentence verdict
    it = gt.loc["interaction_time_s"] if "interaction_time_s" in gt.index else None
    tc = gt.loc["trial_completion_time_s"] if "trial_completion_time_s" in gt.index else None
    sc = gt.loc["correct_response"] if "correct_response" in gt.index else None
    if it is not None and tc is not None and sc is not None:
        time_down = (it["second_minus_first_ci95_upper"] < 0) and (tc["second_minus_first_ci95_upper"] < 0)
        success_not_down = not (sc["second_minus_first_ci95_upper"] < 0)
        verdict = "SUPPORTED" if (time_down and success_not_down) else "NOT SUPPORTED"
        lines.append(
            f"BOTTOM LINE: the statement 'trial completion time decreased over the experiment without a significant "
            f"decline in success' is {verdict} by the tracking-based measures "
            f"(interaction time {'down' if it['second_minus_first_ci95_upper'] < 0 else 'not clearly down'}, "
            f"trial completion time {'down' if tc['second_minus_first_ci95_upper'] < 0 else 'not clearly down'}, "
            f"success {'not declining' if success_not_down else 'declining'})."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures (matplotlib, deterministic)
# --------------------------------------------------------------------------- #
def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_time_over_session(order_bins: pd.DataFrame, out_path: Path, *, dpi: int = 160, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    panels = [
        ("interaction_time_s", "Object interaction time (s)", _C_INTERACTION),
        ("presentation_time_s", "Object presentation time (s)", _C_PRESENTATION),
        ("trial_completion_time_s", "Trial completion time (s)", _C_COMPLETION),
        ("correct_response", "Success rate", _C_SUCCESS),
    ]
    x = order_bins["order_bin"].to_numpy()
    for ax, (m, label, color) in zip(axes, panels):
        y = order_bins[f"{m}_mean"].to_numpy(dtype=float)
        lo = order_bins[f"{m}_ci95_lower"].to_numpy(dtype=float)
        hi = order_bins[f"{m}_ci95_upper"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Session progress (order bin)")
        ax.set_xticks(x)
        if m == "correct_response":
            ax.set_ylim(0.5, 1.0)
            ax.axhline(0.5, color="#9CA3AF", linewidth=1, linestyle="--")
        _style_axis(ax)
    axes[0].set_ylabel("Mean across participants (95% CI)")
    if title:
        fig.suptitle(title, fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_first_second_half(subject_summary: pd.DataFrame, out_path: Path, *, dpi: int = 160) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    panels = [
        ("interaction_time_s", "Object interaction time (s)", _C_INTERACTION),
        ("trial_completion_time_s", "Trial completion time (s)", _C_COMPLETION),
        ("correct_response", "Success rate", _C_SUCCESS),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for ax, (m, label, color) in zip(axes, panels):
        a = subject_summary[f"{m}_first_half"].to_numpy(dtype=float)
        b = subject_summary[f"{m}_second_half"].to_numpy(dtype=float)
        for ya, yb in zip(a, b):
            ax.plot([0, 1], [ya, yb], color="#9CA3AF", linewidth=0.8, alpha=0.7)
        ax.plot([0, 1], [np.nanmean(a), np.nanmean(b)], color=color, linewidth=2.5, marker="o", markersize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["First half", "Second half"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(label, fontsize=11)
        _style_axis(ax)
    axes[0].set_ylabel("Per-participant mean")
    fig.suptitle("Per-participant change between session halves (grey = participants, colour = group mean)", fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_success_by_interaction_quartile(quartiles: pd.DataFrame, out_path: Path, *, dpi: int = 160) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
    if not quartiles.empty:
        x = quartiles["interaction_time_quartile"].to_numpy()
        y = quartiles["success_rate_mean"].to_numpy(dtype=float)
        lo = quartiles["success_rate_ci95_lower"].to_numpy(dtype=float)
        hi = quartiles["success_rate_ci95_upper"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=_C_INTERACTION, linewidth=2, marker="o", markersize=6, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{int(v)}\n({t:.1f} s)" for v, t in zip(x, quartiles["mean_interaction_time_s"])])
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Within-participant interaction-time quartile (mean s)")
    ax.set_ylabel("Success rate (mean, 95% CI)")
    ax.set_title("Success vs object interaction time", fontsize=11)
    _style_axis(ax)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
OUTPUT_TABLES = [
    "trials",
    "subject_summary",
    "group_tests",
    "group_tests_by_experiment_group",
    "order_bins",
    "subject_finger_summary",
    "finger_summary",
    "success_by_interaction_quartile",
    "success_vs_interaction_spearman_group",
    "by_stiffness_delta",
    "quality",
]


def run_object_interaction_time_analysis(
    success_trials: pd.DataFrame,
    file_discovery_summary: pd.DataFrame,
    *,
    output_root: Path,
    run_output_root: Optional[Path] = None,
    fig_dpi: int = 160,
    n_jobs: int = 8,
    n_order_bins: int = 8,
    title: str = "",
) -> dict[str, Any]:
    """Measure, summarise, save CSVs + figures, and return everything.

    CSVs are written to ``output_root/object_interaction_time_<table>.csv`` and,
    when ``run_output_root`` is given, mirrored to
    ``run_output_root/csv/all/shared/object_interaction_time/<table>.csv``;
    figures go to ``.../figures/all/shared/object_interaction_time/``.
    """
    output_root = Path(output_root)
    trials = compute_object_interaction_trials(success_trials, file_discovery_summary, n_jobs=n_jobs)
    summary = summarize_object_interaction(trials, n_order_bins=n_order_bins)
    summary["trials_all"] = trials
    summary["bottom_line"] = bottom_line_text(summary)

    csv_targets = [output_root]
    fig_targets = [output_root / "figures" / "object_interaction_time"]
    if run_output_root is not None:
        run_output_root = Path(run_output_root)
        csv_targets.append(run_output_root / "csv" / "all" / "shared" / "object_interaction_time")
        fig_targets.append(run_output_root / "figures" / "all" / "shared" / "object_interaction_time")

    paths: dict[str, list[str]] = {}
    for name in OUTPUT_TABLES:
        table = summary.get(name)
        if table is None or table.empty:
            continue
        for i, root in enumerate(csv_targets):
            fname = f"object_interaction_time_{name}.csv" if i == 0 else f"{name}.csv"
            root.mkdir(parents=True, exist_ok=True)
            table.to_csv(root / fname, index=False)
            paths.setdefault(name, []).append(str(root / fname))
    for root in csv_targets:
        (root / ("object_interaction_time_bottom_line.txt" if root == output_root else "bottom_line.txt")).write_text(
            summary["bottom_line"], encoding="utf-8"
        )

    for root in fig_targets:
        plot_time_over_session(summary["order_bins"], root / "time_and_success_over_session.png", dpi=fig_dpi, title=title)
        plot_first_second_half(summary["subject_summary"], root / "first_vs_second_half_per_participant.png", dpi=fig_dpi)
        plot_success_by_interaction_quartile(summary["success_by_interaction_quartile"], root / "success_by_interaction_time_quartile.png", dpi=fig_dpi)
    summary["paths"] = paths
    summary["figure_roots"] = [str(r) for r in fig_targets]
    return summary
