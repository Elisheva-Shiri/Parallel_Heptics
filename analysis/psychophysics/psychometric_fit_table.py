"""Build the paper's psychometric-fit table (reviewer-proof version).

Two clearly labelled panels are produced from the pipeline CSVs, all in mm/m:

  Panel A -- POOLED fits (one lapse-aware 4-parameter logistic fitted to all
             trials of a finger, across participants): PSE, bias and JND with
             95% parametric-bootstrap CIs, Weber fraction, lapse rate lambda
             (lambda_low + lambda_high), deviance with its df and chi-square p,
             McFadden pseudo-R^2, the number of trials in the fit, and the
             test of bias = 0. The standard (8.5 mm/m) is given as a reference row.
  Panel B -- PER-PARTICIPANT fits (one fit per participant x finger; the set the
             mixed ANOVA uses): median [IQR] and mean +/- SD of bias and JND across
             participants, the number of fits, trials per fit, and how many fits
             were excluded from group analysis by the PSE band-pass.

Inputs (from ``results/<cohort>/csv/all/shared`` or a ``_working`` folder):
  pse_jnd_group_by_finger.csv, pse_jnd_group_all_pooled.csv,
  pse_jnd_by_subject_finger.csv

Outputs (next to the inputs or in ``--out``):
  psychometric_fit_table.csv   long, machine-readable (both panels)
  psychometric_fit_table.tex   booktabs LaTeX table, ready to paste
  psychometric_fit_table.md    Markdown preview

Usage:
  python psychometric_fit_table.py results/L_N_E/csv/all/shared
  python psychometric_fit_table.py results/L_N_E/_working --out results/L_N_E/csv/all/shared
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FINGER_ORDER = ["I", "M", "R", "P"]
FINGER_NAMES = {"I": "Index", "M": "Middle", "R": "Ring", "P": "Pinky"}
UNIT = "mm/m"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _num(row: pd.Series, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _fmt(v: float, nd: int = 2) -> str:
    return "" if not np.isfinite(v) else f"{v:.{nd}f}"


def _fmt_signed(v: float, nd: int = 2) -> str:
    return "" if not np.isfinite(v) else f"{v:+.{nd}f}"


def _fmt_ci(lo: float, hi: float, nd: int = 2) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _standard_from(frames: list[pd.DataFrame], fallback: float = 8.5) -> float:
    for f in frames:
        if f is not None and not f.empty and "standard_value" in f:
            v = pd.to_numeric(f["standard_value"], errors="coerce").dropna()
            if not v.empty:
                return float(v.median())
    return fallback


# --------------------------------------------------------------------------- #
# panel A: pooled fits
# --------------------------------------------------------------------------- #
def _pooled_row(label: str, row: pd.Series, standard: float) -> dict[str, Any]:
    pse = _num(row, "pse")
    bias = _num(row, "pse_delta_from_standard")
    if not np.isfinite(bias) and np.isfinite(pse):
        bias = pse - standard
    jnd = _num(row, "jnd")
    weber = _num(row, "weber_fraction")
    if not np.isfinite(weber) and np.isfinite(jnd) and standard > 0:
        weber = jnd / standard
    return {
        "panel": "A_pooled_fit",
        "finger": label,
        "pse": pse,
        "pse_ci95_lower": _num(row, "pse_ci95_lower"),
        "pse_ci95_upper": _num(row, "pse_ci95_upper"),
        "bias": bias,
        "bias_ci95_lower": _num(row, "pse_delta_ci95_lower"),
        "bias_ci95_upper": _num(row, "pse_delta_ci95_upper"),
        "jnd": jnd,
        "jnd_ci95_lower": _num(row, "jnd_ci95_lower"),
        "jnd_ci95_upper": _num(row, "jnd_ci95_upper"),
        "weber_fraction": weber,
        "weber_ci95_lower": _num(row, "jnd_over_standard_ci95_lower"),
        "weber_ci95_upper": _num(row, "jnd_over_standard_ci95_upper"),
        "lapse_rate": _num(row, "lapse_rate"),
        "lapse_rate_ci95_lower": _num(row, "lapse_rate_ci95_lower"),
        "lapse_rate_ci95_upper": _num(row, "lapse_rate_ci95_upper"),
        "deviance": _num(row, "deviance"),
        "deviance_df": _num(row, "deviance_df"),
        "deviance_p_chi2": _num(row, "deviance_p_chi2"),
        "pseudo_r2_mcfadden": _num(row, "pseudo_r2_mcfadden"),
        "n_trials": _num(row, "n_trials"),
        "n_stimulus_levels": _num(row, "n_stimulus_levels"),
        "n_bootstrap": _num(row, "n_bootstrap"),
        "p_bias_eq_0": _num(row, "pse_bias_p_value"),
        "fit_method": str(row.get("fit_method", "")),
    }


def pooled_panel(group_by_finger: pd.DataFrame, all_pooled: pd.DataFrame, standard: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [{
        "panel": "A_pooled_fit", "finger": "Standard (reference)", "pse": standard, "bias": 0.0,
        "jnd": np.nan, "weber_fraction": np.nan, "lapse_rate": np.nan, "deviance": np.nan,
        "deviance_df": np.nan, "deviance_p_chi2": np.nan, "pseudo_r2_mcfadden": np.nan,
        "n_trials": np.nan, "p_bias_eq_0": np.nan, "fit_method": "",
    }]
    if group_by_finger is not None and not group_by_finger.empty:
        g = group_by_finger.copy()
        g["_ord"] = g["finger_condition"].astype(str).map(lambda f: FINGER_ORDER.index(f) if f in FINGER_ORDER else 99)
        for _, r in g.sort_values("_ord").iterrows():
            f = str(r["finger_condition"])
            rows.append(_pooled_row(FINGER_NAMES.get(f, f), r, standard))
    if all_pooled is not None and not all_pooled.empty:
        rows.append(_pooled_row("All fingers pooled", all_pooled.iloc[0], standard))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# panel B: per-participant fits
# --------------------------------------------------------------------------- #
def _describe(values: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return {k: np.nan for k in ("n", "mean", "sd", "median", "q25", "q75")}
    return {
        "n": float(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
        "median": float(v.median()), "q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75)),
    }


def participant_panel(by_subject_finger: pd.DataFrame, standard: float) -> pd.DataFrame:
    if by_subject_finger is None or by_subject_finger.empty:
        return pd.DataFrame()
    d = by_subject_finger.copy()
    d["bias"] = pd.to_numeric(d.get("pse_delta_from_standard"), errors="coerce")
    if d["bias"].isna().all():
        d["bias"] = pd.to_numeric(d["pse"], errors="coerce") - standard
    d["jnd"] = pd.to_numeric(d["jnd"], errors="coerce")
    d["lapse_rate"] = pd.to_numeric(d.get("lapse_rate"), errors="coerce")
    excl = d["excluded_from_group_analysis"].astype(bool) if "excluded_from_group_analysis" in d else pd.Series(False, index=d.index)
    rows: list[dict[str, Any]] = []
    groups = [(FINGER_NAMES.get(f, f), d[d["finger_condition"].astype(str) == f]) for f in FINGER_ORDER
              if (d["finger_condition"].astype(str) == f).any()]
    groups.append(("All fingers", d))
    for label, sub in groups:
        sub_excl = excl.loc[sub.index]
        kept = sub[~sub_excl]
        b, j, lam = _describe(kept["bias"]), _describe(kept["jnd"]), _describe(kept["lapse_rate"])
        n_trials = pd.to_numeric(sub.get("n_trials"), errors="coerce")
        rows.append({
            "panel": "B_per_participant_fits",
            "finger": label,
            "n_fits_total": int(len(sub)),
            "n_fits_excluded_pse_band": int(sub_excl.sum()),
            "n_fits_used": int(len(kept)),
            "n_participants": int(sub["subject_id"].nunique()) if "subject_id" in sub else np.nan,
            "trials_per_fit_median": float(n_trials.median()) if n_trials.notna().any() else np.nan,
            "trials_per_fit_min": float(n_trials.min()) if n_trials.notna().any() else np.nan,
            "trials_per_fit_max": float(n_trials.max()) if n_trials.notna().any() else np.nan,
            "bias_mean": b["mean"], "bias_sd": b["sd"], "bias_median": b["median"], "bias_q25": b["q25"], "bias_q75": b["q75"],
            "jnd_mean": j["mean"], "jnd_sd": j["sd"], "jnd_median": j["median"], "jnd_q25": j["q25"], "jnd_q75": j["q75"],
            "weber_median": j["median"] / standard if np.isfinite(j["median"]) and standard > 0 else np.nan,
            "lapse_rate_mean": lam["mean"], "lapse_rate_median": lam["median"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def _tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("_", r"\_")


def to_latex(panel_a: pd.DataFrame, panel_b: pd.DataFrame, standard: float, cohort: str = "") -> str:
    L: list[str] = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{2.5pt}%")
    L.append(r"\caption{Psychometric fits by finger" + (f" ({_tex_escape(cohort)})" if cohort else "") + ".}")
    L.append(r"\label{tab:psychometric-fit}")
    # ---- panel A
    # When every finger has the same number of trials, state the counts once in
    # the panel heading and keep only p_bias in the last column (narrower table).
    fingers_a = panel_a[~panel_a["finger"].astype(str).str.startswith(("Standard", "All"))]
    n_finger = fingers_a["n_trials"].dropna().unique()
    pooled_rows = panel_a[panel_a["finger"].astype(str).str.startswith("All")]
    n_pooled = pooled_rows["n_trials"].dropna().unique()
    same_n = len(n_finger) == 1
    heading = r"\textbf{(A) Pooled fits} -- one 4-parameter logistic per finger fitted to all trials of all participants"
    if same_n:
        heading += f" ({int(n_finger[0])} trials per finger"
        heading += f", {int(n_pooled[0])} pooled)" if len(n_pooled) == 1 else ")"
    heading += r"; 95\% parametric-bootstrap CIs in brackets.\\[2pt]"
    L.append(heading)
    L.append(r"\begin{tabular}{@{}lcccccccc@{}}")
    L.append(r"\toprule")
    last_hdr = "$p_{\\mathrm{bias}=0}$" if same_n else "$N_{\\mathrm{trials}}$, $p_{\\mathrm{bias}=0}$"
    L.append("Finger & PSE [95\\% CI] & Bias [95\\% CI] & JND [95\\% CI] & Weber & $\\lambda$ & Dev.\\,(df), $p$ & ps.-$R^2$ & " + last_hdr + " \\\\")
    L.append(r"\midrule")
    for _, r in panel_a.iterrows():
        if str(r["finger"]).startswith("Standard"):
            L.append(f"Standard (reference) & {_fmt(standard)} & +0.00 & -- & -- & -- & -- & -- & -- \\\\")
            L.append(r"\midrule")
            continue
        dev = _fmt(r.get("deviance", np.nan), 1)
        df = r.get("deviance_df", np.nan)
        dev_txt = f"{dev} ({int(df)}), {_fmt_p(r.get('deviance_p_chi2', np.nan))}" if np.isfinite(df) and dev else dev
        if same_n:
            n_txt = _fmt_p(r.get("p_bias_eq_0", np.nan))
        else:
            n_txt = f"{int(r['n_trials'])}, {_fmt_p(r.get('p_bias_eq_0', np.nan))}" if np.isfinite(r.get("n_trials", np.nan)) else ""
        L.append(
            f"{_tex_escape(str(r['finger']))} & "
            f"{_fmt(r['pse'])} {_fmt_ci(r.get('pse_ci95_lower', np.nan), r.get('pse_ci95_upper', np.nan))} & "
            f"{_fmt_signed(r['bias'])} {_fmt_ci(r.get('bias_ci95_lower', np.nan), r.get('bias_ci95_upper', np.nan))} & "
            f"{_fmt(r['jnd'])} {_fmt_ci(r.get('jnd_ci95_lower', np.nan), r.get('jnd_ci95_upper', np.nan))} & "
            f"{_fmt(r['weber_fraction'], 3)} & {_fmt(r['lapse_rate'], 3)} & {dev_txt} & "
            f"{_fmt(r['pseudo_r2_mcfadden'], 3)} & {n_txt} \\\\"
        )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    # ---- panel B
    if panel_b is not None and not panel_b.empty:
        L.append(r"\\[6pt]")
        L.append(r"\textbf{(B) Per-participant fits} -- one fit per participant $\times$ finger (the estimates entering the mixed ANOVA); across-participant summaries.\\[2pt]")
        any_excluded = bool((panel_b["n_fits_excluded_pse_band"].fillna(0) > 0).any())
        L.append(r"\begin{tabular}{@{}lccccccc@{}}")
        L.append(r"\toprule")
        n_hdr = "$n$ fits (excl.)" if any_excluded else "$n$ fits"
        L.append("Finger & " + n_hdr + " & Trials/fit & Bias median [IQR] & Bias mean $\\pm$ SD & JND median [IQR] & JND mean $\\pm$ SD & $\\lambda$ median \\\\")
        L.append(r"\midrule")
        for _, r in panel_b.iterrows():
            tpf = r.get("trials_per_fit_median", np.nan)
            tmin, tmax = r.get("trials_per_fit_min", np.nan), r.get("trials_per_fit_max", np.nan)
            tpf_txt = f"{int(tpf)}" if np.isfinite(tpf) else ""
            if np.isfinite(tmin) and np.isfinite(tmax) and (tmin != tmax):
                tpf_txt += f" ({int(tmin)}--{int(tmax)})"
            n_cell = f"{int(r['n_fits_used'])} ({int(r['n_fits_excluded_pse_band'])})" if any_excluded else f"{int(r['n_fits_used'])}"
            L.append(
                f"{_tex_escape(str(r['finger']))} & {n_cell} & {tpf_txt} & "
                f"{_fmt_signed(r['bias_median'])} {_fmt_ci(r['bias_q25'], r['bias_q75'])} & "
                f"{_fmt_signed(r['bias_mean'])} $\\pm$ {_fmt(r['bias_sd'])} & "
                f"{_fmt(r['jnd_median'])} {_fmt_ci(r['jnd_q25'], r['jnd_q75'])} & "
                f"{_fmt(r['jnd_mean'])} $\\pm$ {_fmt(r['jnd_sd'])} & {_fmt(r['lapse_rate_median'], 3)} \\\\"
            )
        L.append(r"\bottomrule")
        L.append(r"\end{tabular}")
    L.append(r"\vspace{2pt}")
    L.append(
        r"\parbox{\textwidth}{\footnotesize Note: PSE, Bias ($=$ PSE $-$ " + _fmt(standard, 1) + r"), and JND are in " + UNIT
        + r"; the standard was " + _fmt(standard, 1) + " " + UNIT
        + r" and the eight comparisons spanned 2.5--14.5 " + UNIT
        + r". Weber $=$ JND / standard. $\lambda = \lambda_{l} + \lambda_{h}$ is the fitted total lapse rate (each asymptote bounded in [0, 0.20])."
        + r" Dev.\ is the deviance against the saturated model with its degrees of freedom (stimulus levels $-$ 4 parameters) and $\chi^2$ tail probability; ps.-$R^2$ is McFadden's pseudo-$R^2$."
        + r" $p_{\mathrm{bias}=0}$ tests whether the pooled bias differs from zero (bootstrap SE)."
        + (r" In (B), ``excl.'' counts participant $\times$ finger fits whose PSE fell outside the tested range and were excluded from group pooling."
           if (panel_b is not None and not panel_b.empty and bool((panel_b["n_fits_excluded_pse_band"].fillna(0) > 0).any())) else "")
        + "}"
    )
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n"


def to_markdown(panel_a: pd.DataFrame, panel_b: pd.DataFrame, standard: float) -> str:
    L: list[str] = []
    L.append(f"**(A) Pooled fits** (all values in {UNIT}; 95% bootstrap CI in brackets)\n")
    L.append("| Finger | PSE [95% CI] | Bias [95% CI] | JND [95% CI] | Weber | lambda | Dev. (df), p | pseudo-R2 | N trials | p(bias=0) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in panel_a.iterrows():
        if str(r["finger"]).startswith("Standard"):
            L.append(f"| Standard (reference) | {_fmt(standard)} | +0.00 | – | – | – | – | – | – | – |")
            continue
        df = r.get("deviance_df", np.nan)
        dev_txt = f"{_fmt(r.get('deviance', np.nan), 1)} ({int(df)}), {_fmt_p(r.get('deviance_p_chi2', np.nan))}" if np.isfinite(df) else _fmt(r.get("deviance", np.nan), 1)
        L.append(
            f"| {r['finger']} | {_fmt(r['pse'])} {_fmt_ci(r.get('pse_ci95_lower', np.nan), r.get('pse_ci95_upper', np.nan))} "
            f"| {_fmt_signed(r['bias'])} {_fmt_ci(r.get('bias_ci95_lower', np.nan), r.get('bias_ci95_upper', np.nan))} "
            f"| {_fmt(r['jnd'])} {_fmt_ci(r.get('jnd_ci95_lower', np.nan), r.get('jnd_ci95_upper', np.nan))} "
            f"| {_fmt(r['weber_fraction'], 3)} | {_fmt(r['lapse_rate'], 3)} | {dev_txt} | {_fmt(r['pseudo_r2_mcfadden'], 3)} "
            f"| {int(r['n_trials']) if np.isfinite(r.get('n_trials', np.nan)) else ''} | {_fmt_p(r.get('p_bias_eq_0', np.nan))} |"
        )
    if panel_b is not None and not panel_b.empty:
        L.append("\n**(B) Per-participant fits** (one fit per participant × finger; the ANOVA input)\n")
        L.append("| Finger | n fits (excluded) | Trials/fit | Bias median [IQR] | Bias mean ± SD | JND median [IQR] | JND mean ± SD | lambda median |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in panel_b.iterrows():
            tpf = r.get("trials_per_fit_median", np.nan)
            L.append(
                f"| {r['finger']} | {int(r['n_fits_used'])} ({int(r['n_fits_excluded_pse_band'])}) | {int(tpf) if np.isfinite(tpf) else ''} "
                f"| {_fmt_signed(r['bias_median'])} {_fmt_ci(r['bias_q25'], r['bias_q75'])} | {_fmt_signed(r['bias_mean'])} ± {_fmt(r['bias_sd'])} "
                f"| {_fmt(r['jnd_median'])} {_fmt_ci(r['jnd_q25'], r['jnd_q75'])} | {_fmt(r['jnd_mean'])} ± {_fmt(r['jnd_sd'])} | {_fmt(r['lapse_rate_median'], 3)} |"
            )
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_psychometric_fit_table(
    group_by_finger: pd.DataFrame,
    all_pooled: pd.DataFrame,
    by_subject_finger: pd.DataFrame,
    *,
    standard: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return (panel_a, panel_b, standard) from the three fit tables."""
    std = float(standard) if standard is not None else _standard_from([group_by_finger, all_pooled, by_subject_finger])
    return pooled_panel(group_by_finger, all_pooled, std), participant_panel(by_subject_finger, std), std


def write_psychometric_fit_table(
    out_dir: Path,
    group_by_finger: pd.DataFrame,
    all_pooled: pd.DataFrame,
    by_subject_finger: pd.DataFrame,
    *,
    standard: float | None = None,
    cohort: str = "",
    stem: str = "psychometric_fit_table",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b, std = build_psychometric_fit_table(group_by_finger, all_pooled, by_subject_finger, standard=standard)
    long = pd.concat([a, b], ignore_index=True, sort=False)
    long.insert(0, "units", UNIT)
    long.insert(1, "standard_mm_per_m", std)
    paths = {
        "csv": out_dir / f"{stem}.csv",
        "tex": out_dir / f"{stem}.tex",
        "md": out_dir / f"{stem}.md",
    }
    long.to_csv(paths["csv"], index=False)
    paths["tex"].write_text(to_latex(a, b, std, cohort=cohort), encoding="utf-8")
    paths["md"].write_text(to_markdown(a, b, std), encoding="utf-8")
    return paths


def _read(folder: Path, name: str) -> pd.DataFrame:
    p = folder / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder holding the pse_jnd_*.csv tables")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: same as input)")
    ap.add_argument("--cohort", default="", help="label for the LaTeX caption")
    args = ap.parse_args()
    folder = args.folder
    paths = write_psychometric_fit_table(
        args.out or folder,
        _read(folder, "pse_jnd_group_by_finger.csv"),
        _read(folder, "pse_jnd_group_all_pooled.csv"),
        _read(folder, "pse_jnd_by_subject_finger.csv"),
        cohort=args.cohort,
    )
    print(paths["md"].read_text(encoding="utf-8"))
    for k, p in paths.items():
        print(k, "->", p)


if __name__ == "__main__":
    main()
