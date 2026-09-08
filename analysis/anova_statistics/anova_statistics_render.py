"""Rendering helpers for the ANOVA statistics pipeline.

This module contains figure and table rendering only. It receives statistical
helpers through ``RenderContext`` so the main analysis module stays the source of
truth for calculations and constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns


@dataclass(frozen=True)
class RenderContext:
    system_levels: tuple[str, ...]
    finger_levels: tuple[str, ...]
    finger_fullnames: dict[str, str]
    random_seed: int
    jnd_extreme_threshold: float
    analysis_frame: Callable[..., pd.DataFrame]
    get_p: Callable[[pd.Series], float]
    fmt_p: Callable[[float], str]
    ensure_dirs: Callable[..., None]
    owa: object | None = None


_SYS_COLORS = {"L": "#d87093", "N": "#6b4c9a"}
_SYS_OFFSET = {"L": -0.14, "N": 0.14}
PLOT_HIDE_QC_EXTREME_JND = True

_TABLE_HEADER_BG = "#2c3e50"
_TABLE_HEADER_FG = "white"
_TABLE_ROW_ALT = "#f4f6f8"
_TABLE_SIG_BG = "#ffe3b3"


def _finger_colors(ctx: RenderContext) -> dict[str, str]:
    return getattr(ctx.owa, "_FINGER_COLORS", None) or {
        "I": "#4C72B0",
        "M": "#DD8452",
        "R": "#55A868",
        "P": "#C44E52",
    }


def _mean_ci(values: np.ndarray, ci: float = 95.0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    half = float(stats.sem(values) * stats.t.ppf((1 + ci / 100.0) / 2.0, len(values) - 1))
    return mean, half


def _plot_frame(df: pd.DataFrame, dv: str, ctx: RenderContext) -> pd.DataFrame:
    work = ctx.analysis_frame(df, dv)
    if dv == "JND" and PLOT_HIDE_QC_EXTREME_JND:
        before = len(work)
        work = work[work[dv] <= ctx.jnd_extreme_threshold].copy()
        work.attrs["n_hidden_extreme_jnd"] = before - len(work)
    else:
        work.attrs["n_hidden_extreme_jnd"] = 0
    return work


def _plot_title(title: str, work: pd.DataFrame, dv: str, ctx: RenderContext) -> str:
    n_hidden = int(work.attrs.get("n_hidden_extreme_jnd", 0) or 0)
    if dv == "JND" and n_hidden:
        return (title + f"\n(display omits {n_hidden} QC-extreme "
                f"JND > {ctx.jnd_extreme_threshold:g}; stats unchanged)")
    return title


def plot_dv_by_finger(df: pd.DataFrame, dv: str, hline: Optional[float],
                      ylabel: str, title: str, out_path: str, *,
                      ctx: RenderContext, seed: Optional[int] = None,
                      show_median_bars: bool = False) -> str:
    """Subject points + mean +/- 95% CI per Finger, hue = System.

    When ``show_median_bars`` is true, add the same finger-coloured median
    rectangle convention used by the pooled JND/Bias presentation plots. The
    bar spans from the reference line (``hline``) to the pooled median for that
    finger; if no reference is provided it spans from zero.
    """
    work = _plot_frame(df, dv, ctx)
    rng = np.random.default_rng(ctx.random_seed if seed is None else seed)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_base = {f: i for i, f in enumerate(ctx.finger_levels)}
    finger_colors = _finger_colors(ctx)
    median_bar_width = 0.42

    for f in ctx.finger_levels:
        if show_median_bars:
            vals_all = work.loc[work["Finger"] == f, dv].dropna().to_numpy()
            if len(vals_all):
                median = float(np.median(vals_all))
                median_baseline = 0.0 if hline is None else float(hline)
                bar_bottom = min(median_baseline, median)
                bar_height = abs(median - median_baseline)
                ax.bar(x_base[f], bar_height, bottom=bar_bottom,
                       width=median_bar_width,
                       color=finger_colors.get(f, "#2c3e50"),
                       alpha=0.24, edgecolor="none", zorder=1,
                       align="center")
        for s in ctx.system_levels:
            vals = work.loc[(work["Finger"] == f) & (work["System"] == s), dv].to_numpy()
            jitter = (rng.random(len(vals)) - 0.5) * 0.10
            x = x_base[f] + _SYS_OFFSET[s]
            ax.scatter(np.full(len(vals), x) + jitter, vals, color=_SYS_COLORS[s],
                       alpha=0.35, s=28, edgecolors="none", zorder=2)
            if len(vals):
                m, h = _mean_ci(vals)
                ax.errorbar(x, m, yerr=h, fmt="o", color=_SYS_COLORS[s],
                            ecolor=_SYS_COLORS[s], markeredgecolor="black",
                            markersize=8, capsize=4, zorder=4)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    ax.set_xticks(list(x_base.values()))
    ax.set_xticklabels([f"{f}\n({ctx.finger_fullnames[f]})" for f in ctx.finger_levels])
    ax.set_xlabel("Finger")
    ax.set_ylabel(ylabel)
    ax.set_title(_plot_title(title, work, dv, ctx))
    sys_legend = ax.legend(frameon=False, loc="upper left", title="circle = system")
    ax.add_artist(sys_legend)
    if show_median_bars:
        finger_handles = []
        for f in ctx.finger_levels:
            vals = work.loc[work["Finger"] == f, dv].dropna().to_numpy()
            median_text = "n/a" if len(vals) == 0 else f"{np.median(vals):.2f}"
            finger_handles.append(
                Patch(facecolor=finger_colors.get(f, "#444444"), alpha=0.45,
                      label=f"{f} ({ctx.finger_fullnames[f]}): median={median_text}")
            )
        ax.legend(handles=finger_handles, frameon=False, loc="upper right",
                  title="rectangle bar = median value", fontsize=8)
    else:
        finger_handles = [Line2D([0], [0], color=finger_colors.get(f, "#444444"),
                                 lw=4, label=f"{f} = {ctx.finger_fullnames[f]}")
                          for f in ctx.finger_levels]
        ax.legend(handles=finger_handles, frameon=False, loc="upper right",
                  title="finger codes")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_eight_groups(df: pd.DataFrame, dv: str, ylabel: str, title: str,
                      out_path: str, *, ctx: RenderContext,
                      hline: Optional[float] = None,
                      seed: Optional[int] = None) -> str:
    """Exploratory 8-group figure: points + mean +/- 95% CI per Group8."""
    work = _plot_frame(df, dv, ctx)
    work["Group8"] = work["System"] + "_" + work["Finger"]
    order = [f"{s}_{f}" for s in ctx.system_levels for f in ctx.finger_levels]
    rng = np.random.default_rng(ctx.random_seed if seed is None else seed)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, g in enumerate(order):
        vals = work.loc[work["Group8"] == g, dv].to_numpy()
        jitter = (rng.random(len(vals)) - 0.5) * 0.20
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="#444444",
                   alpha=0.30, s=25, edgecolors="none", zorder=2)
        if len(vals):
            m, h = _mean_ci(vals)
            ax.errorbar(i, m, yerr=h, fmt="o", color="black", ecolor="black",
                        markersize=8, capsize=4, zorder=4)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_xlabel("Combined System_Finger group (EXPLORATORY)")
    ax.set_ylabel(ylabel)
    ax.set_title(_plot_title(title, work, dv, ctx) + "\n(EXPLORATORY: ignores repeated measures)")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def pooled_ln_descriptives(df: pd.DataFrame, dv: str, *, ctx: RenderContext) -> pd.DataFrame:
    """Finger descriptives after pooling L and N subjects together."""
    work = ctx.analysis_frame(df, dv)
    rows = []
    for f in ctx.finger_levels:
        v = work.loc[work["Finger"] == f, dv].dropna()
        rows.append({
            "DV": dv,
            "Finger": f,
            "Finger_name": ctx.finger_fullnames[f],
            "mean": float(v.mean()) if len(v) else np.nan,
            "median value": float(v.median()) if len(v) else np.nan,
            "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
            "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
            "n": int(len(v)),
        })
    return pd.DataFrame(rows)


def plot_pooled_ln_by_finger(df: pd.DataFrame, dv: str, hline: Optional[float],
                             ylabel: str, title: str, out_path: str, *,
                             ctx: RenderContext,
                             seed: Optional[int] = None) -> str:
    """L+N pooled plot: split system points, black mean/CI, median bar."""
    work = _plot_frame(df, dv, ctx)
    rng = np.random.default_rng(ctx.random_seed if seed is None else seed)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_base = {f: i for i, f in enumerate(ctx.finger_levels)}
    finger_colors = _finger_colors(ctx)

    side_offset = {"L": -0.13, "N": 0.13}
    jitter_width = 0.08
    median_bar_width = 0.42

    for f in ctx.finger_levels:
        finger_work = work[work["Finger"] == f]
        x0 = x_base[f]
        vals_all = finger_work[dv].dropna().to_numpy()
        if len(vals_all) == 0:
            continue

        mean, ci = _mean_ci(vals_all)
        median = float(np.median(vals_all))
        finger_color = finger_colors.get(f, "#2c3e50")

        bar_bottom = min(0.0, median)
        bar_height = abs(median)
        ax.bar(x0, bar_height, bottom=bar_bottom, width=median_bar_width,
               color=finger_color, alpha=0.28, edgecolor="none", zorder=1,
               align="center")

        for s in ctx.system_levels:
            vals = finger_work.loc[finger_work["System"] == s, dv].to_numpy()
            jitter = (rng.random(len(vals)) - 0.5) * jitter_width
            ax.scatter(np.full(len(vals), x0 + side_offset[s]) + jitter, vals,
                       color=_SYS_COLORS[s], alpha=0.32, s=24,
                       edgecolors="none", zorder=2)

        ax.errorbar(x0, mean, yerr=ci, fmt="o", color="black",
                    ecolor="black", markerfacecolor="black",
                    markeredgecolor="black", markersize=10, capsize=6,
                    elinewidth=2.2, zorder=4)

    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    ax.set_xticks(list(x_base.values()))
    ax.set_xticklabels([f"{f}\n({ctx.finger_fullnames[f]})" for f in ctx.finger_levels])
    ax.set_xlabel("Finger")
    ax.set_ylabel(ylabel)
    ax.set_title(_plot_title(title, work, dv, ctx))

    mark_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_SYS_COLORS["L"],
               markeredgecolor="none", alpha=0.55, markersize=8,
               label="L subjects (left)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_SYS_COLORS["N"],
               markeredgecolor="none", alpha=0.55, markersize=8,
               label="N subjects (right)"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black",
               markersize=8, label="pooled mean +/- 95% CI"),
    ]
    mark_legend = ax.legend(handles=mark_handles, frameon=False, loc="upper left",
                            title="marks", fontsize=8)
    ax.add_artist(mark_legend)

    finger_handles = []
    for f in ctx.finger_levels:
        vals = work.loc[work["Finger"] == f, dv].dropna().to_numpy()
        median_text = "n/a" if len(vals) == 0 else f"{np.median(vals):.2f}"
        finger_handles.append(
            Patch(facecolor=finger_colors.get(f, "#444444"), alpha=0.45,
                  label=f"{f} ({ctx.finger_fullnames[f]}): median={median_text}")
        )
    ax.legend(handles=finger_handles, frameon=False, loc="upper right",
              title="rectangle bar = pooled median value", fontsize=8)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_contrasts(contrasts: pd.DataFrame, dv: str, ylabel: str,
                   title: str, out_path: str, *, ctx: RenderContext) -> str:
    """Planned contrast plot: L - N difference per finger with Welch CI."""
    fig, ax = plt.subplots(figsize=(8, 5))
    finger_colors = _finger_colors(ctx)
    for i, (_, r) in enumerate(contrasts.iterrows()):
        col = finger_colors.get(r["Finger"], "#2c3e50")
        ax.errorbar(i, r["diff_L_minus_N"],
                    yerr=[[r["diff_L_minus_N"] - r["CI95_low"]],
                          [r["CI95_high"] - r["diff_L_minus_N"]]],
                    fmt="o", color=col, capsize=6, lw=2, elinewidth=2,
                    markersize=10, markeredgecolor="black")
        if bool(r.get("sig_holm", False)):
            ax.text(i, r["CI95_high"], "*", ha="center", va="bottom",
                    fontsize=16, fontweight="bold")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(list(range(len(contrasts))))
    ax.set_xticklabels([f"{r.Finger}\n({r.Finger_name})"
                        for _, r in contrasts.iterrows()])
    ax.set_xlabel("Finger")
    ax.set_ylabel(ylabel + "  (L - N)")
    ax.set_title(title + "\n* Holm-corrected p < .05")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def dataframe_to_image(df: pd.DataFrame, out_path: str,
                       title: Optional[str] = None,
                       sig_mask: Optional[np.ndarray] = None,
                       fontsize: int = 10,
                       col_scale: float = 1.0) -> str:
    """Render a small DataFrame as a formatted PNG table."""
    data = df.copy()
    for c in data.columns:
        data[c] = data[c].astype(str)

    n_rows, n_cols = data.shape
    fig_w = max(7.0, min(18.0, col_scale * (1.2 * n_cols + 1.5)))
    fig_h = max(2.0, 0.38 * (n_rows + 2) + (0.5 if title else 0.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=10)

    table = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.25)

    if sig_mask is None:
        sig_mask = np.zeros(n_rows, dtype=bool)
    sig_mask = np.asarray(sig_mask, dtype=bool)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor(_TABLE_HEADER_BG)
            cell.get_text().set_color(_TABLE_HEADER_FG)
            cell.get_text().set_fontweight("bold")
        else:
            idx = row - 1
            if idx < len(sig_mask) and sig_mask[idx]:
                cell.set_facecolor(_TABLE_SIG_BG)
            elif idx % 2:
                cell.set_facecolor(_TABLE_ROW_ALT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _finite(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and np.isfinite(v)


def _gg_epsilon(aov: pd.DataFrame) -> float:
    """Greenhouse-Geisser epsilon reported by pingouin on the within row.

    The same epsilon corrects every effect that involves the within factor
    (Finger main effect and the System x Finger interaction).
    """
    if "eps" not in aov.columns:
        return np.nan
    eps = aov["eps"].dropna()
    return float(eps.iloc[0]) if len(eps) else np.nan


def mixed_summary_frame(results: dict, key: str, *, ctx: RenderContext) -> pd.DataFrame:
    """Tidy Bias+JND mixed-ANOVA summary for image rendering.

    Effects involving the within factor (Finger, Interaction) also report the
    Greenhouse-Geisser epsilon and the epsilon-corrected degrees of freedom,
    so the table is self-contained about the sphericity correction.
    """
    rows = []
    sig = []
    for dv in ("Bias", "JND"):
        aov = results[key][dv]["aov"]
        eps = _gg_epsilon(aov)
        for _, r in aov.iterrows():
            p = ctx.get_p(r)
            df1 = r.get("DF1", r.get("ddof1", np.nan))
            df2 = r.get("DF2", r.get("ddof2", np.nan))
            within = str(r.get("Source", "")) != "System"
            has_eps = within and _finite(eps)
            p_gg = r.get("p_GG_corr", np.nan)
            rows.append({
                "DV": dv,
                "Effect": r.get("Source", ""),
                "F": f"{float(r.get('F', np.nan)):.3f}",
                "df1": df1,
                "df2": df2,
                "eps (GG)": f"{eps:.3f}" if has_eps else "-",
                "df1 (GG)": f"{float(df1) * eps:.2f}" if has_eps else "-",
                "df2 (GG)": f"{float(df2) * eps:.2f}" if has_eps else "-",
                "p": ctx.fmt_p(p),
                "p_GG": ctx.fmt_p(p_gg) if _finite(p_gg) else "-",
                "np2": f"{float(r.get('np2', np.nan)):.3f}",
                "N": results[key][dv].get("n_subjects", ""),
            })
            sig.append(bool(np.isfinite(p) and p < 0.05))
    out = pd.DataFrame(rows)
    out.attrs["sig_mask"] = np.array(sig)
    return out


def descriptives_frame(df: pd.DataFrame, *, ctx: RenderContext) -> pd.DataFrame:
    """Condition descriptives per Finger x System: mean +/- SD and median [IQR].

    The median [Q1, Q3] column is reported next to the mean because a few
    off-scale psychometric fits inflate the SD of some cells; the median shows
    the typical participant.
    """
    rows = []
    for f in ctx.finger_levels:
        row = {"Finger": f"{f} ({ctx.finger_fullnames[f]})"}
        for dv in ("Bias", "JND"):
            sub = df[df["Finger"] == f]
            for s in ctx.system_levels:
                v = sub.loc[sub["System"] == s, dv].dropna()
                if len(v) > 1:
                    q1, med, q3 = np.percentile(v, [25, 50, 75])
                    row[f"{dv} {s} mean +/- SD"] = f"{v.mean():.2f} +/- {v.std(ddof=1):.2f}"
                    row[f"{dv} {s} median [IQR]"] = f"{med:.2f} [{q1:.2f}, {q3:.2f}]"
                else:
                    row[f"{dv} {s} mean +/- SD"] = "-"
                    row[f"{dv} {s} median [IQR]"] = "-"
                row[f"n {s}"] = int(len(v))
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts_summary_frame(results: dict, *, ctx: RenderContext) -> pd.DataFrame:
    rows = []
    sig = []
    for dv in ("Bias", "JND"):
        ct = results["contrasts"].get(dv, pd.DataFrame())
        for _, r in ct.iterrows():
            rows.append({
                "DV": dv,
                "Finger": f"{r['Finger']} ({r['Finger_name']})",
                "mean L": f"{r['mean_L']:.2f}",
                "mean N": f"{r['mean_N']:.2f}",
                "L-N": f"{r['diff_L_minus_N']:.2f}",
                "95% CI": f"[{r['CI95_low']:.2f}, {r['CI95_high']:.2f}]",
                "t(df)": f"{r['T']:.2f} ({r['dof']:.1f})",
                "p Holm": ctx.fmt_p(r["p_holm"]),
                "d": f"{r['cohen_d']:.2f}",
            })
            sig.append(bool(r.get("sig_holm", False)))
    out = pd.DataFrame(rows)
    out.attrs["sig_mask"] = np.array(sig)
    return out


def assumption_checks_frame(df: pd.DataFrame, results: dict, *, ctx: RenderContext) -> pd.DataFrame:
    rows = []
    sig = []
    for dv in ("Bias", "JND"):
        work = ctx.analysis_frame(df, dv)
        cell_mean = work.groupby(["System", "Finger"], observed=True)[dv].transform("mean")
        resid = (work[dv] - cell_mean).to_numpy()
        try:
            sh_p = float(stats.shapiro(resid).pvalue)
        except Exception:
            sh_p = np.nan
        l_values = work.loc[work["System"] == "L", dv].dropna()
        n_values = work.loc[work["System"] == "N", dv].dropna()
        try:
            lev_p = float(stats.levene(l_values, n_values, center="median").pvalue)
        except Exception:
            lev_p = np.nan
        mixed_dv = results["mixed"][dv]
        sph = mixed_dv.get("sphericity_note", "")
        sph_obj = mixed_dv.get("sphericity")
        eps = _gg_epsilon(mixed_dv["aov"])
        normal_ok = np.isfinite(sh_p) and sh_p >= 0.05
        var_ok = np.isfinite(lev_p) and lev_p >= 0.05
        sph_met = ("sphericity=met" in sph) or ("='met'" in sph)
        sph_violated = "VIOLATED" in sph
        if sph_obj is not None and hasattr(sph_obj, "W"):
            mauchly_w = f"{float(sph_obj.W):.4f}"
            mauchly_chi2 = f"{float(sph_obj.chi2):.2f} ({int(sph_obj.dof)})"
            mauchly_p = ctx.fmt_p(float(sph_obj.pval))
        else:
            mauchly_w = mauchly_chi2 = mauchly_p = "-"
        rows.append({
            "DV": dv,
            "Normality (Shapiro p)": ctx.fmt_p(sh_p),
            "Normality OK?": "yes" if normal_ok else "NO",
            "Equal var (Levene p)": ctx.fmt_p(lev_p),
            "Equal var OK?": "yes" if var_ok else "NO",
            "Mauchly W": mauchly_w,
            "chi2 (dof)": mauchly_chi2,
            "Mauchly p": mauchly_p,
            "eps (GG)": f"{eps:.3f}" if _finite(eps) else "-",
            "Sphericity": "met" if sph_met else ("VIOLATED" if sph_violated else "-"),
        })
        sig.append(not (normal_ok and var_ok) or sph_violated)
    out = pd.DataFrame(rows)
    out.attrs["sig_mask"] = np.array(sig)
    return out


def render_summary_tables(df: pd.DataFrame, results: dict, out_dir: str, *,
                          ctx: RenderContext) -> dict:
    """Render all mixed-design summary tables as PNG images."""
    ctx.ensure_dirs(out_dir)
    paths = {}

    main_tbl = mixed_summary_frame(results, "mixed", ctx=ctx)
    paths["mixed_anova_summary"] = dataframe_to_image(
        main_tbl, os.path.join(out_dir, "mixed_anova_summary.png"),
        title="MAIN mixed-design ANOVA: System (between) x Finger (within)",
        sig_mask=main_tbl.attrs.get("sig_mask"))

    sens_tbl = mixed_summary_frame(results, "mixed_sensitivity", ctx=ctx)
    paths["mixed_anova_sensitivity"] = dataframe_to_image(
        sens_tbl, os.path.join(out_dir, "mixed_anova_sensitivity.png"),
        title="SENSITIVITY mixed-design ANOVA (flagged/degenerate fits removed)",
        sig_mask=sens_tbl.attrs.get("sig_mask"))

    desc_tbl = descriptives_frame(df, ctx=ctx)
    paths["descriptives"] = dataframe_to_image(
        desc_tbl, os.path.join(out_dir, "descriptives.png"),
        title="Descriptives: Bias and JND by System x Finger (mean +/- SD, median [IQR])",
        col_scale=1.15)

    ct_tbl = contrasts_summary_frame(results, ctx=ctx)
    paths["planned_contrasts"] = dataframe_to_image(
        ct_tbl, os.path.join(out_dir, "planned_contrasts.png"),
        title="Planned contrasts: System L - N within each finger (Holm)",
        sig_mask=ct_tbl.attrs.get("sig_mask"))

    asm_tbl = assumption_checks_frame(df, results, ctx=ctx)
    paths["assumption_checks"] = dataframe_to_image(
        asm_tbl, os.path.join(out_dir, "assumption_checks.png"),
        title="ANOVA assumption checks (flagged row = an assumption is violated)",
        sig_mask=asm_tbl.attrs.get("sig_mask"))

    return paths


def fatigue_order_summary_frame(fatigue: dict) -> pd.DataFrame:
    """Compact finger-level fatigue/order summary for display/reporting."""
    slopes = fatigue.get("tables", {}).get("finger_slope_summary")
    if slopes is None or slopes.empty:
        return pd.DataFrame()
    cols = [
        "finger_condition",
        "n_subjects",
        "mean_success_rate",
        "mean_success_slope",
        "success_slope_ci95_lower",
        "success_slope_ci95_upper",
        "mean_reaction_time_slope",
        "reaction_time_slope_ci95_lower",
        "reaction_time_slope_ci95_upper",
    ]
    cols = [c for c in cols if c in slopes.columns]
    out = slopes[cols].copy()
    if {"success_slope_ci95_lower", "success_slope_ci95_upper"}.issubset(out.columns):
        out["success_slope_interpretation"] = np.where(
            out["success_slope_ci95_upper"] < 0,
            "decreases over time",
            np.where(out["success_slope_ci95_lower"] > 0,
                     "increases over time", "no clear change"),
        )
    if {"reaction_time_slope_ci95_lower", "reaction_time_slope_ci95_upper"}.issubset(out.columns):
        out["reaction_time_interpretation"] = np.where(
            out["reaction_time_slope_ci95_lower"] > 0,
            "slower over time",
            np.where(out["reaction_time_slope_ci95_upper"] < 0,
                     "faster over time", "no clear change"),
        )
    return out


def fatigue_order_interpretation(fatigue: dict) -> str:
    """One-paragraph interpretation of the fatigue/order control evidence."""
    summary = fatigue_order_summary_frame(fatigue)
    if summary.empty:
        return ("Finger-order/fatigue summary is unavailable for this source; "
                "the psychophysics order CSVs were not found.")
    success_decline = (
        "success_slope_interpretation" in summary.columns
        and (summary["success_slope_interpretation"] == "decreases over time").any()
    )
    rt_slower = (
        "reaction_time_interpretation" in summary.columns
        and (summary["reaction_time_interpretation"] == "slower over time").any()
    )
    if not success_decline and not rt_slower:
        return ("Finger-order/fatigue control did not show a clear systematic "
                "decline in success rate or slowing across repeated finger "
                "appearances. This supports interpreting the main ANOVA as a "
                "finger/system analysis rather than an order artifact.")
    parts = []
    if success_decline:
        parts.append("some fingers show a success-rate decline over time")
    if rt_slower:
        parts.append("some fingers show reaction-time slowing over time")
    return ("Finger-order/fatigue control suggests that " + " and ".join(parts)
            + ". Treat the main ANOVA together with this control when discussing "
              "possible fatigue/order confounds.")


def plot_fatigue_order_control(fatigue: dict, out_dir: str, *, cohort: str,
                               ctx: RenderContext) -> dict:
    """Create fatigue/order control figures from precomputed CSV tables."""
    ctx.ensure_dirs(out_dir)
    paths = {}
    appearance = fatigue.get("tables", {}).get("finger_by_appearance_order")
    if appearance is not None and not appearance.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        for ax, metric, ylabel in [
            (axes[0], "mean_success_rate", "Success rate"),
            (axes[1], "mean_reaction_time", "Reaction time"),
        ]:
            if metric not in appearance.columns:
                ax.axis("off")
                continue
            for finger, sub in appearance.groupby("finger_condition", observed=True):
                sub = sub.sort_values("finger_appearance_order")
                ax.plot(sub["finger_appearance_order"], sub[metric],
                        marker="o", label=str(finger))
            ax.set_xlabel("Appearance order within finger")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " vs within-finger order")
            ax.legend(frameon=False, title="Finger", fontsize=8)
            sns.despine(ax=ax)
        fig.suptitle(f"{cohort}: finger-order / fatigue control")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{cohort}__finger_order_control.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["finger_order_control"] = path

    summary = fatigue_order_summary_frame(fatigue)
    if not summary.empty:
        paths["fatigue_order_summary"] = dataframe_to_image(
            summary,
            os.path.join(out_dir, f"{cohort}__fatigue_order_summary.png"),
            title="Finger-order/fatigue control summary",
            fontsize=8,
            col_scale=1.2,
        )
    return paths


def plot_bootstrap_diff(diff_ci: pd.DataFrame, dv: str, out_path: str, *,
                        ctx: RenderContext) -> str:
    """Bootstrap L - N difference per finger with 95% CI (subgroup-safe)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    diff_ci = diff_ci.reset_index(drop=True)
    finger_colors = _finger_colors(ctx)
    for i, row in diff_ci.iterrows():
        col = finger_colors.get(str(row["Finger"]), "#2c3e50")
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


def oneway_summary_frame(oneway_res: dict, *, ctx: RenderContext) -> pd.DataFrame:
    """Tidy System x DV x test summary for the one-way table image."""
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
                    f"{float(stat):.3f}" if np.isfinite(float(stat) if stat == stat else np.nan) else "-"),
                "p": ctx.fmt_p(p),
                "effect": (f"{float(es):.3f}" if (es == es and es is not None) else "-"),
                "n": ("" if r.get("n") is None or not (r.get("n") == r.get("n"))
                      else int(r.get("n"))),
            })
            sig.append(bool(np.isfinite(p) and p < 0.05))
    norm = []
    for r in rows:
        stat_val = r.pop("F", None)
        if stat_val is None:
            stat_val = r.pop("chi_square", None)
        for k in list(r.keys()):
            if k not in ("System", "DV", "Test", "p", "effect", "n", "statistic"):
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


def render_oneway_summary_table(oneway_res: dict, out_dir: str, *,
                                ctx: RenderContext) -> Optional[str]:
    """Render the one-way per-system summary as a PNG table."""
    if not oneway_res.get("systems"):
        return None
    ctx.ensure_dirs(out_dir)
    tbl = oneway_summary_frame(oneway_res, ctx=ctx)
    if tbl.empty:
        return None
    return dataframe_to_image(
        tbl, os.path.join(out_dir, "oneway_per_system_summary.png"),
        title="One-way per-system (finger) analysis: ANOVA (F) + Friedman "
              "(chi2) by System x DV",
        sig_mask=tbl.attrs.get("sig_mask"))
