"""Airsled setup-factor diagnostics for L/N protocol decisions.

`L` participants are treated as the laboratory airsled setup and `N`
participants as the no-airsled setup. The module loads existing analysis CSVs,
adds setup labels, writes balance/summary tables, and saves visual checks that
color subjects by setup when a formal setup-factor test is underpowered.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from analysis.group_comparisons import (
        SETUP_FACTOR_COLUMN,
        SETUP_FACTOR_LABELS,
        SETUP_FACTOR_ORDER,
        add_setup_factor_columns,
        compute_analysis_scope_tables,
        compute_setup_factor_tables,
    )
    from analysis.scope_plots import save_scope_summary_plots
except ModuleNotFoundError:  # pragma: no cover - supports running from this folder
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from analysis.group_comparisons import (
        SETUP_FACTOR_COLUMN,
        SETUP_FACTOR_LABELS,
        SETUP_FACTOR_ORDER,
        add_setup_factor_columns,
        compute_analysis_scope_tables,
        compute_setup_factor_tables,
    )
    from analysis.scope_plots import save_scope_summary_plots

SETUP_COLORS = {"no_airsled": "#D55E00", "airsled": "#0072B2"}
SETUP_X = {"no_airsled": 0.0, "airsled": 1.0}

DEFAULT_TABLE_SPECS: list[dict[str, Any]] = [
    {
        "category": "kinematics",
        "relative_path": Path("analysis/Kinematics/results/subject_kinematic_summary.csv"),
        "table_name": "subject_kinematic_summary",
        "metrics": [
            "success_rate",
            "mean_speed_px_s",
            "mean_path_length_px",
            "mean_jerk_px_s3",
            "mean_normalized_jerk_cost",
        ],
        "condition_cols": ["finger_condition", "stiffness_value"],
    },
    {
        "category": "kinematics",
        "relative_path": Path("analysis/Kinematics/results/subject_velocity_acceleration_summary.csv"),
        "table_name": "subject_velocity_acceleration_summary",
        "metrics": ["mean_speed_px_s", "mean_acceleration_px_s2", "mean_jerk_px_s3"],
        "condition_cols": ["finger_condition", "stiffness_value"],
    },
    {
        "category": "probing",
        "relative_path": Path("analysis/probing_analysis/results/probing_subject_finger_stiffness_summary.csv"),
        "table_name": "probing_subject_finger_stiffness_summary",
        "metrics": [
            "success_rate",
            "mean_probe_count",
            "mean_probe_rate_per_s",
            "mean_first_probe_latency_s",
            "mean_center_dwell_fraction",
            "mean_side_dwell_fraction",
        ],
        "condition_cols": ["finger_condition", "stiffness_value"],
    },
    {
        "category": "psychophysics",
        "relative_path": Path("analysis/psychophysics_analysis/results/pse_jnd_by_subject_finger.csv"),
        "table_name": "pse_jnd_by_subject_finger",
        "metrics": ["pse", "jnd", "pse_delta_from_standard", "weber_fraction"],
        "condition_cols": ["finger_condition"],
    },
    {
        "category": "psychophysics",
        "relative_path": Path("analysis/psychophysics_analysis/results/success_summary_by_subject_finger.csv"),
        "table_name": "success_summary_by_subject_finger",
        "metrics": ["success_rate", "mean_reaction_time", "median_reaction_time"],
        "condition_cols": ["finger_condition"],
    },
]


def save_csv(df: pd.DataFrame, output_root: Path, name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / name
    df.to_csv(path, index=False)
    return path


def _available_metrics(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    return [m for m in metrics if m in df.columns and pd.to_numeric(df[m], errors="coerce").notna().any()]


def _jitter(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.035, n)


def save_setup_factor_plots(
    df: pd.DataFrame,
    *,
    output_root: Path,
    category: str,
    table_name: str,
    metrics: list[str],
    max_points: int = 2000,
    fig_dpi: int = 160,
) -> list[Path]:
    plot_df = add_setup_factor_columns(df)
    plot_df = plot_df[plot_df[SETUP_FACTOR_COLUMN].notna()].copy()
    if plot_df.empty:
        return []
    paths: list[Path] = []
    fig_dir = output_root / "figures" / category
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = _available_metrics(plot_df, metrics)
    for metric in metrics:
        d = plot_df[[SETUP_FACTOR_COLUMN, metric, "subject_id"] + [c for c in ["finger_condition", "stiffness_value"] if c in plot_df.columns]].copy()
        d[metric] = pd.to_numeric(d[metric], errors="coerce")
        d = d.dropna(subset=[metric, SETUP_FACTOR_COLUMN])
        if d.empty:
            continue
        if len(d) > max_points:
            d = d.sample(max_points, random_state=20260516)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for setup in SETUP_FACTOR_ORDER:
            s = d[d[SETUP_FACTOR_COLUMN] == setup]
            if s.empty:
                continue
            x = np.full(len(s), SETUP_X[setup]) + _jitter(len(s), seed=20260516 + len(metric) + int(SETUP_X[setup]))
            ax.scatter(
                x,
                s[metric],
                s=34,
                alpha=0.70,
                color=SETUP_COLORS[setup],
                label=f"{SETUP_FACTOR_LABELS[setup]} (n={s['subject_id'].nunique() if 'subject_id' in s else len(s)})",
                edgecolor="white",
                linewidth=0.4,
            )
            subject_means = s.groupby("subject_id", dropna=False)[metric].mean() if "subject_id" in s else pd.Series(dtype=float)
            if not subject_means.empty:
                mean_x = SETUP_X[setup]
                ax.errorbar(
                    [mean_x],
                    [subject_means.mean()],
                    yerr=[subject_means.sem() if len(subject_means) > 1 else 0.0],
                    color="black",
                    marker="D",
                    markersize=6,
                    capsize=4,
                    linewidth=1.5,
                )
        ax.set_xticks([SETUP_X[s] for s in SETUP_FACTOR_ORDER])
        ax.set_xticklabels([SETUP_FACTOR_LABELS[s] for s in SETUP_FACTOR_ORDER], rotation=12, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{category}: {metric} by setup factor")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = fig_dir / f"{table_name}_{metric}_by_setup.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)
    return paths


def analyze_table(
    df: pd.DataFrame,
    *,
    category: str,
    table_name: str,
    metrics: list[str],
    condition_cols: list[str],
    output_root: Path,
    fig_dpi: int = 160,
) -> dict[str, pd.DataFrame]:
    labeled = add_setup_factor_columns(df)
    n_setup_labeled = int(labeled[SETUP_FACTOR_COLUMN].notna().sum()) if SETUP_FACTOR_COLUMN in labeled else 0
    setup_status = pd.DataFrame(
        [
            {
                "category": category,
                "table_name": table_name,
                "n_rows": int(len(df)),
                "n_setup_labeled_rows": n_setup_labeled,
                "setup_labels_detected": bool(n_setup_labeled > 0),
                "status": "ok" if n_setup_labeled > 0 else "no_L_or_N_setup_labels_detected",
            }
        ]
    )
    tables = {
        f"{category}_{table_name}_setup_status": setup_status,
        **{
            f"{category}_{table_name}_{name}": table
            for name, table in compute_setup_factor_tables(
                df,
                metric_columns=metrics,
                condition_cols=condition_cols,
            ).items()
        },
        **{
            f"{category}_{table_name}_{name}": table
            for name, table in compute_analysis_scope_tables(
                df,
                metric_columns=metrics,
                condition_cols=condition_cols,
            ).items()
        },
    }
    for name, table in tables.items():
        save_csv(table, output_root, f"{name}.csv")
    figures = save_setup_factor_plots(
        df,
        output_root=output_root,
        category=category,
        table_name=table_name,
        metrics=metrics,
        fig_dpi=fig_dpi,
    )
    tables[f"{category}_{table_name}_figure_manifest"] = pd.DataFrame(
        {"figure": [str(p) for p in figures], "category": category, "table_name": table_name}
    )
    save_csv(tables[f"{category}_{table_name}_figure_manifest"], output_root, f"{category}_{table_name}_figure_manifest.csv")
    scope_manifest = save_scope_summary_plots(
        tables,
        output_root,
        namespace=f"{category}_{table_name}",
        metrics=metrics,
        fig_dpi=fig_dpi,
    )
    tables[f"{category}_{table_name}_scope_figure_manifest"] = scope_manifest
    return tables


def run_setup_factor_diagnostics(
    repo_root: Path,
    output_root: Path,
    *,
    table_specs: list[dict[str, Any]] | None = None,
    fig_dpi: int = 160,
) -> dict[str, pd.DataFrame]:
    """Load known result tables and write setup-factor diagnostics/plots."""
    specs = table_specs or DEFAULT_TABLE_SPECS
    all_tables: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        path = repo_root / spec["relative_path"]
        exists = path.exists()
        manifest_rows.append(
            {
                "category": spec["category"],
                "table_name": spec["table_name"],
                "path": str(path),
                "exists": bool(exists),
            }
        )
        if not exists:
            continue
        df = pd.read_csv(path)
        tables = analyze_table(
            df,
            category=spec["category"],
            table_name=spec["table_name"],
            metrics=list(spec["metrics"]),
            condition_cols=list(spec["condition_cols"]),
            output_root=output_root,
            fig_dpi=fig_dpi,
        )
        all_tables.update(tables)
    manifest = pd.DataFrame(manifest_rows)
    save_csv(manifest, output_root, "setup_factor_input_manifest.csv")
    all_tables["setup_factor_input_manifest"] = manifest
    return all_tables


def _default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root, repo_root / "analysis" / "setup_factor" / "results"


def main(argv: list[str] | None = None) -> int:
    default_repo, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Run L/N airsled setup-factor diagnostics and plots.")
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--fig-dpi", type=int, default=160)
    args = parser.parse_args(argv)
    run_setup_factor_diagnostics(args.repo_root, args.output_root, fig_dpi=args.fig_dpi)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
