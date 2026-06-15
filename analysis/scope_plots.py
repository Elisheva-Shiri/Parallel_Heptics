"""Plots for shared all/group/setup/participant summary tables."""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAT_COLUMNS = {
    "metric",
    "n_observations",
    "n_subjects",
    "mean",
    "median",
    "std",
    "sem",
    "ci95_lower",
    "ci95_upper",
    "raw_values_json",
    "raw_values_count",
    "raw_values_truncated",
}
LEVEL_SPECS = [
    ("analysis_scope", ["analysis_scope", "analysis_scope_value"]),
    ("experiment_group", ["experiment_group"]),
    ("protocol_group", ["protocol_group"]),
    ("setup_factor", ["setup_factor"]),
]
MAX_RAW_VALUES_PER_SUMMARY_POINT = 80
MAX_SUMMARY_PLOT_ROWS = 160


def sanitize_name(value: Any, fallback: str = "unknown") -> str:
    text = str(value) if value is not None and not pd.isna(value) else fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def available_summary_metrics(df: pd.DataFrame, metrics: Iterable[str] | None = None) -> list[str]:
    if "metric" not in df or "mean" not in df:
        return []
    requested = list(metrics) if metrics is not None else sorted(df["metric"].dropna().astype(str).unique())
    present = set(df["metric"].dropna().astype(str))
    return [m for m in requested if m in present]


def _summary_kind(table_name: str, df: pd.DataFrame) -> tuple[str, list[str]] | None:
    if not table_name.endswith("_metric_summary"):
        return None
    for kind, cols in LEVEL_SPECS:
        if set(cols).issubset(df.columns):
            return kind, cols
    return None


def _plot_with_ci(ax: plt.Axes, x: np.ndarray, y: pd.Series, df: pd.DataFrame) -> None:
    if {"ci95_lower", "ci95_upper"}.issubset(df.columns):
        lower = y - pd.to_numeric(df["ci95_lower"], errors="coerce")
        upper = pd.to_numeric(df["ci95_upper"], errors="coerce") - y
        if np.isfinite(lower).any() or np.isfinite(upper).any():
            ax.errorbar(x, y, yerr=np.vstack([lower.fillna(0), upper.fillna(0)]), fmt="none", color="black", capsize=3)


def _raw_values_from_summary(value: Any) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            return []
    elif not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[float] = []
    for item in parsed:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def _save_one_plot(
    plot_df: pd.DataFrame,
    *,
    fig_dir: Path,
    source_table: str,
    summary_level: str,
    metric: str,
    x_cols: list[str],
    fig_dpi: int,
) -> dict[str, Any] | None:
    cols = [
        *x_cols,
        "mean",
        *[
            c
            for c in [
                "n_subjects",
                "n_observations",
                "ci95_lower",
                "ci95_upper",
                "raw_values_json",
            ]
            if c in plot_df
        ],
    ]
    d = plot_df[cols].copy()
    d["mean"] = pd.to_numeric(d["mean"], errors="coerce")
    d = d.dropna(subset=["mean"])
    if d.empty:
        return None
    d["_plot_label"] = d[x_cols].astype(str).agg(" | ".join, axis=1)
    d = d.sort_values(x_cols)
    if len(d) > MAX_SUMMARY_PLOT_ROWS:
        return None

    width = min(18.0, max(5.5, 0.35 * len(d) + 2.5))
    fig, ax = plt.subplots(figsize=(width, 4.8))
    x = np.arange(len(d))
    if "raw_values_json" in d.columns:
        for i, raw in enumerate(d["raw_values_json"]):
            values = _raw_values_from_summary(raw)
            if not values:
                continue
            if len(values) > MAX_RAW_VALUES_PER_SUMMARY_POINT:
                sample_idx = np.linspace(0, len(values) - 1, MAX_RAW_VALUES_PER_SUMMARY_POINT, dtype=int)
                values = [values[j] for j in sample_idx]
            jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(values), x[i]) + jitter,
                values,
                s=14,
                color="0.15",
                alpha=0.16,
                linewidths=0,
                zorder=1,
            )
    ax.bar(x, d["mean"], color="#4C78A8", alpha=0.85)
    _plot_with_ci(ax, x, d["mean"], d)
    ax.set_xticks(x)
    ax.set_xticklabels(d["_plot_label"], rotation=70 if len(d) > 8 else 20, ha="right")
    ax.set_ylabel(metric)
    ax.set_xlabel(summary_level.replace("_", " "))
    ax.set_title(f"{metric} by {summary_level.replace('_', ' ')}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = fig_dir / f"{sanitize_name(source_table)}_{sanitize_name(metric)}_{sanitize_name(summary_level)}.png"
    fig.savefig(out, dpi=fig_dpi)
    plt.close(fig)
    return {
        "figure": str(out),
        "source_table": source_table,
        "summary_level": summary_level,
        "metric": metric,
        "rows_plotted": int(len(d)),
    }


def _x_columns(df: pd.DataFrame, grouping_cols: list[str]) -> list[str]:
    condition_cols = [c for c in df.columns if c not in STAT_COLUMNS and c not in grouping_cols]
    return [grouping_cols[-1], *condition_cols]


def _save_one_plot_job(job: dict[str, Any]) -> dict[str, Any] | None:
    """Render one scope-summary plot, possibly in a worker process.

    matplotlib is already forced to the Agg backend at import. The parent
    process renders these plots with seaborn's whitegrid theme active (set by
    the figure helpers that run earlier in the same notebook cell); worker
    processes do not inherit that global rcParams state, so replicate it here so
    the PNGs stay pixel-identical to the serial version.
    """
    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid")
    except Exception:  # pragma: no cover - seaborn optional
        pass
    return _save_one_plot(job["plot_df"], **job["kwargs"])


def _scope_n_jobs(n_items: int, n_jobs: int | None) -> int:
    if n_items <= 1:
        return 1
    if n_jobs is not None:
        return max(1, min(int(n_jobs), n_items))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu - 1, n_items))


def _render_scope_jobs(jobs: list[dict[str, Any]], n_jobs: int | None) -> list[dict[str, Any] | None]:
    """Render scope-plot jobs, in parallel across processes when worthwhile.

    Each plot is independent and rendered deterministically (Agg), so the PNGs
    and the returned manifest rows are identical to the serial version; job
    order is preserved so the manifest order is unchanged.
    """
    if not jobs:
        return []
    n_jobs_eff = _scope_n_jobs(len(jobs), n_jobs)
    if n_jobs_eff <= 1:
        return [_save_one_plot_job(job) for job in jobs]
    try:
        from joblib import Parallel, delayed

        return list(Parallel(n_jobs=n_jobs_eff)(delayed(_save_one_plot_job)(job) for job in jobs))
    except Exception:  # pragma: no cover - joblib missing/unavailable -> serial
        return [_save_one_plot_job(job) for job in jobs]


def save_scope_summary_plots(
    tables: dict[str, pd.DataFrame],
    output_root: Path,
    *,
    namespace: str,
    metrics: Iterable[str] | None = None,
    fig_dpi: int = 160,
    n_jobs: int | None = None,
    keep_only: Iterable[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Write summary plots and a manifest for all/group/setup/participant tables.

    When ``keep_only`` is given (an iterable of ``(source_table, summary_level,
    metric)`` triples), ONLY those figures are produced and all other
    combinations are skipped. This is how the notebook restricts the scope
    summaries to the few headline figures instead of the full cross-product.
    """
    fig_dir = output_root / "figures" / f"{sanitize_name(namespace)}_scope_summaries"
    fig_dir.mkdir(parents=True, exist_ok=True)
    keep_set = {tuple(t) for t in keep_only} if keep_only is not None else None

    # Build every plot job first (preserving the original iteration order), then
    # render them in one shared process pool.
    jobs: list[dict[str, Any]] = []
    for table_name, df in tables.items():
        if df.empty:
            continue
        kind = _summary_kind(table_name, df)
        if kind is None:
            continue
        summary_kind, grouping_cols = kind
        x_cols = _x_columns(df, grouping_cols)
        for metric in available_summary_metrics(df, metrics):
            metric_df = df[df["metric"].astype(str) == metric].copy()
            if summary_kind == "analysis_scope":
                for scope, scoped in metric_df.groupby("analysis_scope", dropna=False):
                    if keep_set is not None and (table_name, str(scope), metric) not in keep_set:
                        continue
                    jobs.append({
                        "plot_df": scoped.copy(),
                        "kwargs": {
                            "fig_dir": fig_dir,
                            "source_table": table_name,
                            "summary_level": str(scope),
                            "metric": metric,
                            "x_cols": x_cols,
                            "fig_dpi": fig_dpi,
                        },
                    })
                continue
            if keep_set is not None and (table_name, summary_kind, metric) not in keep_set:
                continue
            jobs.append({
                "plot_df": metric_df,
                "kwargs": {
                    "fig_dir": fig_dir,
                    "source_table": table_name,
                    "summary_level": summary_kind,
                    "metric": metric,
                    "x_cols": x_cols,
                    "fig_dpi": fig_dpi,
                },
            })

    rows = [row for row in _render_scope_jobs(jobs, n_jobs) if row]

    manifest = pd.DataFrame(rows, columns=["figure", "source_table", "summary_level", "metric", "rows_plotted"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_root / f"{sanitize_name(namespace)}_scope_figure_manifest.csv", index=False)
    return manifest
