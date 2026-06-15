"""One-way (per-system) finger analysis of psychophysics PSE/JND.

This is a deliberately LIGHTWEIGHT alternative to the mixed-design ANOVA in
``analysis/anova_statistics`` -- it depends ONLY on numpy / pandas / scipy /
matplotlib (no pingouin, no statsmodels), so it runs on any kernel including a
bare system Python.

CONNECTION / DO NOT DELETE (2026-06-14): this module is a SHARED DEPENDENCY of the
``anova_statistics`` package and now lives BESIDE it in ``analysis/anova_statistics/``.
``anova_statistics.py`` imports it as ``owa`` (see the bridge near its top) and reuses
the stat + plot helpers below via ``anova_statistics.run_oneway_flat(...)`` -- the
one-way results are written into the combined ANOVA notebook's flat ``results/`` tree
(``oneway/`` + ``oneway_filters/``). The standalone one-way notebook, its ``results/``,
and the old ``analysis/oneway_anova/`` folder were removed in favour of that single
combined notebook; this ``.py`` is the implementation. Keep public signatures stable.

Design
------
The data are per-subject x finger PSE/JND summaries with two factors:
  * System  = workspace_setup (L / N)  -- between-subject (each subject is in one).
  * Finger  = finger_condition (I / M / R / P) -- repeated (4 per subject).

For EACH system separately we ask "do the four fingers differ?" using two tests:
  1. One-way ANOVA  (scipy.stats.f_oneway) -- fingers treated as independent
     groups; the classic F-table, with eta^2 / omega^2 effect sizes.
  2. Friedman test   (scipy.stats.friedmanchisquare) -- repeated-measures
     non-parametric; respects that each subject contributes all four fingers and
     is robust to the normality violations seen in this dataset. Kendall's W is
     reported as the effect size.

Assumption checks (Shapiro normality per finger, Levene equal-variance across
fingers) and Holm-corrected pairwise post-hoc tests (Welch t for the parametric
branch, Wilcoxon signed-rank for the non-parametric branch) accompany each test.

Outputs mirror the psychophysics tree: the two systems L and N are the two main
folders; each holds group-level ``figures/all`` + ``csv/all`` plus one nested
sub-folder per subject.
"""

from __future__ import annotations

import gc
import os
import shutil
import stat
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / notebook-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SYSTEM_LEVELS = ["L", "N"]
# Anatomical index -> pinky ordering for display.
FINGER_ORDER = ["I", "M", "R", "P"]
FINGER_LABELS = {"I": "Index", "M": "Middle", "R": "Ring", "P": "Pinky"}

# Dependent variables: column in the loaded frame -> (pretty label, unit).
DEPENDENT_VARS = {
    "Bias": ("PSE bias (comparison - standard)", "stiffness units"),
    "JND": ("JND (discrimination threshold)", "stiffness units"),
}

# Source columns in the raw pse_jnd_by_subject_finger.csv.
_RAW_COLUMNS = {
    "Subject": "subject_id",
    "Finger": "finger_condition",
    "Bias": "pse_delta_from_standard",
    "JND": "jnd",
}

ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_pse_jnd(path: str | Path) -> pd.DataFrame:
    """Load the per-subject x finger summary into canonical columns.

    Returns a frame with: Subject, System (L/N), Finger (I/M/R/P), Bias, JND,
    plus the carried quality flags ``excluded_from_group_analysis`` and
    ``fit_warning``. No rows are dropped.
    """
    raw = pd.read_csv(path)
    missing = [c for c in _RAW_COLUMNS.values() if c not in raw.columns]
    if missing:
        raise KeyError(f"Source data is missing required columns: {missing}")

    # System comes from workspace_setup / setup_factor; normalise to L / N.
    system_col = "workspace_setup" if "workspace_setup" in raw.columns else "setup_factor"
    system = raw[system_col].astype(str).str.strip().str.upper().str[0]

    df = pd.DataFrame(
        {
            "Subject": raw[_RAW_COLUMNS["Subject"]].astype(str),
            "System": system,
            "Finger": raw[_RAW_COLUMNS["Finger"]].astype(str).str.strip().str.upper(),
            "Bias": pd.to_numeric(raw[_RAW_COLUMNS["Bias"]], errors="coerce"),
            "JND": pd.to_numeric(raw[_RAW_COLUMNS["JND"]], errors="coerce"),
        }
    )
    df["excluded_from_group_analysis"] = (
        raw["excluded_from_group_analysis"].astype(bool)
        if "excluded_from_group_analysis" in raw.columns
        else False
    )
    df["fit_warning"] = (
        raw["fit_warning"].astype(str) if "fit_warning" in raw.columns else ""
    )

    df = df[df["System"].isin(SYSTEM_LEVELS)].copy()
    df["System"] = pd.Categorical(df["System"], categories=SYSTEM_LEVELS, ordered=True)
    present = [f for f in FINGER_ORDER if f in set(df["Finger"])]
    extra = sorted(set(df["Finger"]) - set(FINGER_ORDER))
    df["Finger"] = pd.Categorical(df["Finger"], categories=present + extra, ordered=True)
    return df


def fingers_present(df: pd.DataFrame) -> list[str]:
    return [f for f in FINGER_ORDER if f in set(df["Finger"].astype(str))]


# --------------------------------------------------------------------------- #
# Statistics  (numpy / scipy only)
# --------------------------------------------------------------------------- #

def holm_correction(pvals: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (no statsmodels)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def _finger_groups(df: pd.DataFrame, dv: str, fingers: list[str]) -> list[np.ndarray]:
    return [
        pd.to_numeric(df.loc[df["Finger"].astype(str) == f, dv], errors="coerce")
        .dropna()
        .to_numpy()
        for f in fingers
    ]


def one_way_anova(df: pd.DataFrame, dv: str) -> dict[str, Any]:
    """One-way between-groups ANOVA across fingers (scipy.stats.f_oneway).

    Adds sum-of-squares, eta^2 and omega^2 effect sizes computed by hand.
    """
    fingers = fingers_present(df)
    groups = _finger_groups(df, dv, fingers)
    groups = [g for g in groups if g.size > 0]
    k = len(groups)
    out: dict[str, Any] = {
        "dv": dv,
        "k_groups": k,
        "n_total": int(sum(g.size for g in groups)),
        "test": "one_way_anova",
    }
    if k < 2 or any(g.size < 2 for g in groups):
        out.update({"F": np.nan, "p_value": np.nan, "df_between": np.nan,
                    "df_within": np.nan, "eta_squared": np.nan, "omega_squared": np.nan,
                    "note": "insufficient data"})
        return out

    F, p = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = float(sum(g.size * (g.mean() - grand) ** 2 for g in groups))
    ss_total = float(((allv - grand) ** 2).sum())
    ss_within = ss_total - ss_between
    n = allv.size
    df_b, df_w = k - 1, n - k
    ms_within = ss_within / df_w if df_w > 0 else np.nan
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    omega2 = (
        (ss_between - df_b * ms_within) / (ss_total + ms_within)
        if (ss_total + ms_within) > 0
        else np.nan
    )
    out.update(
        {
            "F": float(F),
            "p_value": float(p),
            "df_between": int(df_b),
            "df_within": int(df_w),
            "ss_between": ss_between,
            "ss_within": ss_within,
            "ss_total": ss_total,
            "eta_squared": float(eta2),
            "omega_squared": float(omega2),
            "significant": bool(p < ALPHA),
        }
    )
    return out


def _wide_complete(df: pd.DataFrame, dv: str, fingers: list[str]) -> pd.DataFrame:
    """Subject x finger matrix with complete cases only (for repeated tests)."""
    wide = (
        df.pivot_table(index="Subject", columns="Finger", values=dv, observed=True, aggfunc="mean")
        .reindex(columns=fingers)
    )
    return wide.dropna(axis=0, how="any")


def friedman_test(df: pd.DataFrame, dv: str) -> dict[str, Any]:
    """Repeated-measures Friedman test across fingers + Kendall's W."""
    fingers = fingers_present(df)
    wide = _wide_complete(df, dv, fingers)
    n, k = wide.shape
    out: dict[str, Any] = {
        "dv": dv,
        "k_groups": int(k),
        "n_complete_subjects": int(n),
        "test": "friedman",
    }
    if k < 3 or n < 2:
        out.update({"chi_square": np.nan, "p_value": np.nan, "df": np.nan,
                    "kendalls_w": np.nan, "note": "insufficient complete cases"})
        return out
    chi, p = stats.friedmanchisquare(*[wide[f].to_numpy() for f in fingers])
    kendalls_w = float(chi / (n * (k - 1))) if n * (k - 1) > 0 else np.nan
    out.update(
        {
            "chi_square": float(chi),
            "p_value": float(p),
            "df": int(k - 1),
            "kendalls_w": kendalls_w,
            "significant": bool(p < ALPHA),
        }
    )
    return out


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else np.nan


def posthoc_welch(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Pairwise Welch t-tests across fingers, Holm-corrected (parametric)."""
    fingers = fingers_present(df)
    rows = []
    for a, b in combinations(fingers, 2):
        x = _finger_groups(df, dv, [a])[0]
        y = _finger_groups(df, dv, [b])[0]
        if x.size < 2 or y.size < 2:
            t, p = np.nan, np.nan
        else:
            t, p = stats.ttest_ind(x, y, equal_var=False)
        rows.append(
            {
                "dv": dv, "test": "welch_t", "finger_a": a, "finger_b": b,
                "mean_a": float(np.nanmean(x)) if x.size else np.nan,
                "mean_b": float(np.nanmean(y)) if y.size else np.nan,
                "n_a": int(x.size), "n_b": int(y.size),
                "t": float(t) if t == t else np.nan,
                "cohens_d": _cohens_d(x, y),
                "p_raw": float(p) if p == p else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p_raw"].fillna(1.0).to_numpy())
    out["significant_holm"] = out["p_holm"] < ALPHA
    return out


def posthoc_wilcoxon(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank across fingers (paired), Holm-corrected."""
    fingers = fingers_present(df)
    wide = _wide_complete(df, dv, fingers)
    rows = []
    for a, b in combinations(fingers, 2):
        if a not in wide.columns or b not in wide.columns or wide.shape[0] < 1:
            stat, p = np.nan, np.nan
            n_pairs = 0
        else:
            xa, xb = wide[a].to_numpy(), wide[b].to_numpy()
            n_pairs = int(xa.size)
            diff = xa - xb
            if n_pairs < 1 or np.allclose(diff, 0):
                stat, p = np.nan, np.nan
            else:
                try:
                    stat, p = stats.wilcoxon(xa, xb)
                except ValueError:
                    stat, p = np.nan, np.nan
        rows.append(
            {
                "dv": dv, "test": "wilcoxon_signed_rank", "finger_a": a, "finger_b": b,
                "n_pairs": n_pairs,
                "statistic": float(stat) if stat == stat else np.nan,
                "p_raw": float(p) if p == p else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p_raw"].fillna(1.0).to_numpy())
    out["significant_holm"] = out["p_holm"] < ALPHA
    return out


def assumption_checks(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Shapiro normality per finger + Levene equal-variance across fingers."""
    fingers = fingers_present(df)
    rows = []
    for f in fingers:
        vals = _finger_groups(df, dv, [f])[0]
        if vals.size >= 3:
            w, p = stats.shapiro(vals)
            rows.append({"dv": dv, "check": "shapiro_normality", "scope": f,
                         "n": int(vals.size), "statistic": float(w),
                         "p_value": float(p), "ok": bool(p > ALPHA)})
        else:
            rows.append({"dv": dv, "check": "shapiro_normality", "scope": f,
                         "n": int(vals.size), "statistic": np.nan,
                         "p_value": np.nan, "ok": np.nan})
    groups = [g for g in _finger_groups(df, dv, fingers) if g.size >= 2]
    if len(groups) >= 2:
        w, p = stats.levene(*groups, center="median")
        rows.append({"dv": dv, "check": "levene_equal_variance", "scope": "across_fingers",
                     "n": int(sum(g.size for g in groups)), "statistic": float(w),
                     "p_value": float(p), "ok": bool(p > ALPHA)})
    return pd.DataFrame(rows)


def descriptives(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Per-finger mean / sd / sem / 95% CI for a DV."""
    fingers = fingers_present(df)
    rows = []
    for f in fingers:
        vals = _finger_groups(df, dv, [f])[0]
        n = vals.size
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
        half = tcrit * sem if n > 1 else np.nan
        rows.append({"dv": dv, "finger": f, "n": int(n), "mean": mean, "sd": sd,
                     "sem": sem, "ci95_lower": mean - half if half == half else np.nan,
                     "ci95_upper": mean + half if half == half else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures  (matplotlib only)
# --------------------------------------------------------------------------- #

_FINGER_COLORS = {"I": "#4C72B0", "M": "#DD8452", "R": "#55A868", "P": "#C44E52"}


def _annotation(anova: dict, friedman: dict) -> str:
    def fmt_p(p):
        return "n/a" if not (p == p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")
    a = (f"ANOVA: F({anova.get('df_between','?')},{anova.get('df_within','?')})="
         f"{anova.get('F', float('nan')):.2f}, p={fmt_p(anova.get('p_value', float('nan')))}, "
         f"η²={anova.get('eta_squared', float('nan')):.3f}")
    fr = (f"Friedman: χ²({friedman.get('df','?')})="
          f"{friedman.get('chi_square', float('nan')):.2f}, "
          f"p={fmt_p(friedman.get('p_value', float('nan')))}, "
          f"W={friedman.get('kendalls_w', float('nan')):.3f}")
    return a + "\n" + fr


def plot_dv_by_finger(df: pd.DataFrame, dv: str, system: str, desc: pd.DataFrame,
                      anova: dict, friedman: dict, out_path: Path,
                      fig_dpi: int = 150) -> Path:
    """Mean +/- 95% CI bar with per-subject points + test annotation."""
    fingers = fingers_present(df)
    label, unit = DEPENDENT_VARS[dv]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x = np.arange(len(fingers))
    means = [desc.loc[desc.finger == f, "mean"].values[0] if (desc.finger == f).any() else np.nan for f in fingers]
    lo = [desc.loc[desc.finger == f, "ci95_lower"].values[0] if (desc.finger == f).any() else np.nan for f in fingers]
    hi = [desc.loc[desc.finger == f, "ci95_upper"].values[0] if (desc.finger == f).any() else np.nan for f in fingers]
    err = np.array([[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]])
    colors = [_FINGER_COLORS.get(f, "#888888") for f in fingers]
    ax.bar(x, means, yerr=err, capsize=5, color=colors, alpha=0.55,
           edgecolor="black", linewidth=0.8, zorder=2)
    # per-subject jittered points
    rng_offsets = np.linspace(-0.18, 0.18, 7)
    for xi, f in enumerate(fingers):
        vals = _finger_groups(df, dv, [f])[0]
        if vals.size:
            jit = rng_offsets[np.arange(vals.size) % rng_offsets.size]
            ax.scatter(np.full(vals.size, xi) + jit, vals, s=20, color=colors[xi],
                       edgecolor="white", linewidth=0.5, zorder=3, alpha=0.9)
    if dv == "Bias":
        ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([FINGER_LABELS.get(f, f) for f in fingers])
    ax.set_xlabel("Finger")
    ax.set_ylabel(f"{label} [{unit}]")
    ax.set_title(f"System {system}: {dv} by finger (n={int(df['Subject'].nunique())} subjects)")
    ax.text(0.02, 0.98, _annotation(anova, friedman), transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=fig_dpi)
    plt.close(fig)
    return out_path


def plot_posthoc_matrix(posthoc: pd.DataFrame, dv: str, system: str, fingers: list[str],
                        out_path: Path, p_col: str = "p_holm", fig_dpi: int = 150) -> Path:
    """Lower-triangle heatmap of Holm-corrected pairwise p-values."""
    n = len(fingers)
    idx = {f: i for i, f in enumerate(fingers)}
    mat = np.full((n, n), np.nan)
    for _, r in posthoc.iterrows():
        if r["finger_a"] in idx and r["finger_b"] in idx:
            i, j = idx[r["finger_a"]], idx[r["finger_b"]]
            mat[max(i, j), min(i, j)] = r[p_col]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=0.1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([FINGER_LABELS.get(f, f) for f in fingers])
    ax.set_yticklabels([FINGER_LABELS.get(f, f) for f in fingers])
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                star = "*" if mat[i, j] < ALPHA else ""
                ax.text(j, i, f"{mat[i, j]:.3f}{star}", ha="center", va="center",
                        fontsize=9, color="black")
    method = posthoc["test"].iloc[0] if len(posthoc) else "pairwise"
    ax.set_title(f"System {system}: {DEPENDENT_VARS[dv][0]}\nHolm-adj p ({method}); * p<{ALPHA}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Holm-adjusted p")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=fig_dpi)
    plt.close(fig)
    return out_path


def plot_subject_dv(sub_df: pd.DataFrame, subject: str, out_path: Path,
                    fig_dpi: int = 150) -> Path:
    """Per-subject Bias & JND across fingers (one panel each)."""
    fingers = [f for f in FINGER_ORDER if f in set(sub_df["Finger"].astype(str))]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, dv in zip(axes, ["Bias", "JND"]):
        vals = [pd.to_numeric(sub_df.loc[sub_df["Finger"].astype(str) == f, dv],
                              errors="coerce").mean() for f in fingers]
        colors = [_FINGER_COLORS.get(f, "#888888") for f in fingers]
        ax.bar(range(len(fingers)), vals, color=colors, alpha=0.7, edgecolor="black")
        if dv == "Bias":
            ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(fingers)))
        ax.set_xticklabels([FINGER_LABELS.get(f, f) for f in fingers])
        ax.set_ylabel(DEPENDENT_VARS[dv][0])
        ax.set_title(dv)
    fig.suptitle(f"Subject {subject}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=fig_dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Orchestration / output tree
# --------------------------------------------------------------------------- #

def _rmtree_retry(target: Path, attempts: int = 15, delay: float = 0.35) -> None:
    """Delete a tree, retrying through transient Windows/Dropbox file locks."""
    def _onerror(func, path, _exc):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            pass

    last: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            shutil.rmtree(target, onerror=_onerror)
            if not target.exists():
                return
        except OSError as exc:
            last = exc
        gc.collect()
        time.sleep(delay)
    if target.exists() and last is not None:
        raise last


def _safe_reset(folder: Path) -> Path:
    """Reset an output folder, refusing anything outside an oneway_anova tree."""
    folder = Path(folder).resolve()
    if "oneway_anova" not in folder.parts:
        raise ValueError(f"Refusing to reset a folder outside oneway_anova: {folder}")
    if folder.exists():
        _rmtree_retry(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_csv(df: pd.DataFrame, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    df.to_csv(path, index=False)
    return path


def analyze_system(df_system: pd.DataFrame, system: str) -> dict[str, Any]:
    """Run the full per-system analysis; returns tables + test dicts (no I/O)."""
    res: dict[str, Any] = {"system": system, "anova": {}, "friedman": {},
                           "posthoc_param": {}, "posthoc_nonparam": {},
                           "assumptions": {}, "descriptives": {}}
    for dv in DEPENDENT_VARS:
        res["anova"][dv] = one_way_anova(df_system, dv)
        res["friedman"][dv] = friedman_test(df_system, dv)
        res["posthoc_param"][dv] = posthoc_welch(df_system, dv)
        res["posthoc_nonparam"][dv] = posthoc_wilcoxon(df_system, dv)
        res["assumptions"][dv] = assumption_checks(df_system, dv)
        res["descriptives"][dv] = descriptives(df_system, dv)
    return res


def save_system_tree(out_root: Path, system: str, df_system: pd.DataFrame,
                     fig_dpi: int = 150) -> pd.DataFrame:
    """Write one system's group-level outputs + per-subject sub-folders."""
    out_root = Path(out_root)
    fig_all = out_root / "figures" / "all"
    csv_all = out_root / "csv" / "all"
    fig_all.mkdir(parents=True, exist_ok=True)
    csv_all.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    def rec(scope, kind, path):
        manifest.append({"system": system, "scope": scope, "kind": kind, "path": str(path)})

    res = analyze_system(df_system, system)
    fingers = fingers_present(df_system)

    # Combined test summary table (both tests x both DVs).
    summary_rows = []
    for dv in DEPENDENT_VARS:
        a, fr = res["anova"][dv], res["friedman"][dv]
        summary_rows.append({"dv": dv, "test": "one_way_anova", "statistic_name": "F",
                             "statistic": a.get("F"), "df1": a.get("df_between"),
                             "df2": a.get("df_within"), "p_value": a.get("p_value"),
                             "effect_size_name": "eta_squared", "effect_size": a.get("eta_squared"),
                             "n": a.get("n_total")})
        summary_rows.append({"dv": dv, "test": "friedman", "statistic_name": "chi_square",
                             "statistic": fr.get("chi_square"), "df1": fr.get("df"),
                             "df2": np.nan, "p_value": fr.get("p_value"),
                             "effect_size_name": "kendalls_w", "effect_size": fr.get("kendalls_w"),
                             "n": fr.get("n_complete_subjects")})
    rec("all", "csv", _save_csv(pd.DataFrame(summary_rows), csv_all, "test_summary.csv"))

    # Per-DV CSVs + figures.
    for dv in DEPENDENT_VARS:
        rec("all", "csv", _save_csv(res["descriptives"][dv], csv_all, f"descriptives_{dv}.csv"))
        rec("all", "csv", _save_csv(res["assumptions"][dv], csv_all, f"assumption_checks_{dv}.csv"))
        rec("all", "csv", _save_csv(res["posthoc_param"][dv], csv_all, f"posthoc_welch_{dv}.csv"))
        rec("all", "csv", _save_csv(res["posthoc_nonparam"][dv], csv_all, f"posthoc_wilcoxon_{dv}.csv"))
        rec("all", "figure", plot_dv_by_finger(
            df_system, dv, system, res["descriptives"][dv], res["anova"][dv],
            res["friedman"][dv], fig_all / f"{dv.lower()}_by_finger.png", fig_dpi))
        rec("all", "figure", plot_posthoc_matrix(
            res["posthoc_param"][dv], dv, system, fingers,
            fig_all / f"posthoc_welch_{dv.lower()}.png", fig_dpi=fig_dpi))
        rec("all", "figure", plot_posthoc_matrix(
            res["posthoc_nonparam"][dv], dv, system, fingers,
            fig_all / f"posthoc_wilcoxon_{dv.lower()}.png", fig_dpi=fig_dpi))

    # Sensitivity: drop band-excluded fits, recompute the headline tests.
    sens = df_system[~df_system["excluded_from_group_analysis"].astype(bool)]
    if len(sens) and sens["Subject"].nunique() >= 2:
        sens_rows = []
        for dv in DEPENDENT_VARS:
            a, fr = one_way_anova(sens, dv), friedman_test(sens, dv)
            sens_rows.append({"dv": dv, "test": "one_way_anova", "F": a.get("F"),
                              "p_value": a.get("p_value"), "eta_squared": a.get("eta_squared"),
                              "n": a.get("n_total")})
            sens_rows.append({"dv": dv, "test": "friedman", "chi_square": fr.get("chi_square"),
                              "p_value": fr.get("p_value"), "kendalls_w": fr.get("kendalls_w"),
                              "n": fr.get("n_complete_subjects")})
        rec("all", "csv", _save_csv(pd.DataFrame(sens_rows), csv_all, "test_summary_sensitivity.csv"))

    # Per-subject sub-folders.
    for subject in sorted(df_system["Subject"].astype(str).unique()):
        sub_df = df_system[df_system["Subject"].astype(str) == subject]
        s_root = out_root / subject
        rec(subject, "csv", _save_csv(
            sub_df[["Subject", "System", "Finger", "Bias", "JND", "fit_warning",
                    "excluded_from_group_analysis"]],
            s_root / "csv", "pse_jnd_by_finger.csv"))
        rec(subject, "figure", plot_subject_dv(sub_df, subject, s_root / "figures" / "bias_jnd_by_finger.png", fig_dpi))

    manifest_df = pd.DataFrame(manifest)
    _save_csv(manifest_df, out_root, "analysis_tree_manifest.csv")
    return manifest_df


def run_oneway_anova(data_path: str | Path, out_root: str | Path,
                     fig_dpi: int = 150, reset: bool = True) -> dict[str, pd.DataFrame]:
    """End-to-end: load data, then write a tree per system under ``out_root``."""
    out_root = Path(out_root)
    df = load_pse_jnd(data_path)
    if reset:
        _safe_reset(out_root)
    else:
        out_root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for system in SYSTEM_LEVELS:
        sys_df = df[df["System"].astype(str) == system].copy()
        if sys_df.empty:
            continue
        manifests[system] = save_system_tree(out_root / system, system, sys_df, fig_dpi=fig_dpi)
    return manifests
