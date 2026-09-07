"""Figures for the movement-strategy validation.

Plotting only - every number comes from `analysis.py`. Keeping the two apart is
what stopped this study having two divergent copies of the same computation.

All axes are in controller units; see the scope note in `analysis.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import CIRCLE_SEGMENT, StudyConfig

FIGURE_DIR = Path(__file__).resolve().parent / "figures"

STRATEGY_STYLE: dict[str, dict[str, object]] = {
    "cardinal": {"color": "#d7191c", "label": "Cardinal (4-way)"},
    "cardinal_diagonal": {"color": "#fdae61", "label": "Cardinal-diagonal (8-way)"},
    "free_form": {"color": "#2c7bb6", "label": "Free-form (continuous)"},
}
COMMANDED_STYLE = {"color": "0.45", "linestyle": "--", "linewidth": 2.0}


def _save(fig: plt.Figure, name: str, output_dir: Path | None) -> Path:
    output_dir = output_dir or FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=150, bbox_inches="tight")
    temporary.replace(path)
    plt.close(fig)
    return path


def plot_reconstructed_circles(
    samples: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> Path:
    """Commanded circle vs each strategy's reconstruction, one panel each.

    The point of the panel row is that all three look round - which is exactly
    why a shape metric cannot rank them. The angular markers show where each
    strategy actually is at the same step.
    """
    strategies = [s.value for s in config.strategies]
    fig, axes = plt.subplots(1, len(strategies), figsize=(5.2 * len(strategies), 5.6))
    marker_steps = np.linspace(0, config.circle_steps - 1, 12, dtype=int)

    for axis, strategy in zip(np.atleast_1d(axes), strategies):
        part = samples.loc[
            samples["strategy"].eq(strategy) & samples["segment"].eq(CIRCLE_SEGMENT)
        ].reset_index(drop=True)
        style = STRATEGY_STYLE[strategy]

        axis.plot(part["ideal_x"], part["ideal_y"], **COMMANDED_STYLE, label="Commanded", zorder=1)
        axis.plot(
            part["reconstructed_x"],
            part["reconstructed_y"],
            color=style["color"],
            linewidth=2.4,
            label="Reconstructed",
            zorder=3,
        )
        # Same-step correspondence: commanded point -> where the tactor is.
        for step in marker_steps:
            row = part.iloc[step]
            axis.plot(
                [row["ideal_x"], row["reconstructed_x"]],
                [row["ideal_y"], row["reconstructed_y"]],
                color=style["color"],
                alpha=0.45,
                linewidth=1.0,
                zorder=2,
            )

        axis.scatter(
            part["reconstructed_x"],
            part["reconstructed_y"],
            s=9,
            color=style["color"],
            edgecolors="none",
            alpha=0.55,
            zorder=4,
        )

        rms = float(np.sqrt(np.mean(part["total_error"] ** 2)))
        distinct = len(np.unique(np.round(part[["ideal_x", "ideal_y"]].to_numpy(float), 6), axis=0))
        reachable = len(
            np.unique(np.round(part[["quantised_x", "quantised_y"]].to_numpy(float), 6), axis=0)
        )
        axis.set_title(
            f"{style['label']}\n"
            f"point-to-point RMS = {rms:.1f} units | "
            f"{reachable} of {distinct} commanded points reachable",
            fontsize=11,
        )
        axis.set_xlabel("X (controller units)")
        axis.set_ylabel("Y (controller units)")
        axis.set_aspect("equal", "box")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Same mechanism, same commanded circle - only the direction quantisation differs.\n"
        "Every reconstructed point lands on the commanded radius, so shape metrics score all "
        "three as near-perfect circles;\nthe tie lines and the reachable-point count show the "
        "along-path error those metrics cannot see.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "reconstructed_circles_by_strategy.png", output_dir)


def plot_error_decomposition(
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> Path:
    """Quantisation vs execution error: the strategy cost against the noise floor."""
    fig, (ax_trace, ax_bar) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    for strategy in (s.value for s in config.strategies):
        part = samples.loc[samples["strategy"].eq(strategy)]
        style = STRATEGY_STYLE[strategy]
        ax_trace.plot(
            part["step"],
            part["total_error"],
            color=style["color"],
            linewidth=1.6,
            label=style["label"],
        )

    ax_trace.axvspan(0, config.line_steps, color="0.93", zorder=0)
    ax_trace.axvspan(
        config.line_steps + config.circle_steps,
        config.line_steps + config.circle_steps + config.return_steps,
        color="0.93",
        zorder=0,
    )
    ax_trace.set_xlabel("Path step")
    ax_trace.set_ylabel("Total error (controller units)")
    ax_trace.set_title("Distance from the commanded point, per step")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend(fontsize=9)

    order = [s.value for s in config.strategies]
    positions = np.arange(len(order))
    width = 0.38
    ax_bar.bar(
        positions - width / 2,
        metrics.loc[order, "quantisation_rms"],
        width,
        label="Quantisation (strategy)",
        color="#d7191c",
    )
    ax_bar.bar(
        positions + width / 2,
        metrics.loc[order, "execution_rms"],
        width,
        label="Execution (model + truncation)",
        color="#2c7bb6",
    )
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([STRATEGY_STYLE[s]["label"] for s in order], fontsize=9)
    ax_bar.set_ylabel("RMS error (controller units)")
    ax_bar.set_title("Strategy cost vs shared noise floor")
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.legend(fontsize=9)

    fig.tight_layout()
    return _save(fig, "error_decomposition.png", output_dir)


def plot_motor_commands(
    samples: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> Path:
    """Motor commands per strategy - now comparable, one model for every row."""
    strategies = [s.value for s in config.strategies]
    fig, axes = plt.subplots(
        len(strategies), 1, figsize=(12, 3.1 * len(strategies)), sharex=True, sharey=True
    )

    for axis, strategy in zip(np.atleast_1d(axes), strategies):
        part = samples.loc[samples["strategy"].eq(strategy)]
        for motor_index, shade in enumerate(("#7b3294", "#c2a5cf", "#008837")):
            axis.plot(
                part["step"],
                part[f"motor_{motor_index}"],
                color=shade,
                linewidth=1.5,
                label=f"Motor {motor_index}",
            )
        axis.axhline(0.0, color="0.4", linewidth=1.0)
        axis.set_title(STRATEGY_STYLE[strategy]["label"], fontsize=11)
        axis.set_ylabel("Command")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=3, loc="upper right")

    np.atleast_1d(axes)[-1].set_xlabel("Path step")
    fig.suptitle(
        "Motor commands on a fixed kinematic model. Amplitude is near-identical across "
        "strategies;\nthe real difference is the discontinuous jump when a quantised "
        "direction snaps to a new sector.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "motor_commands_by_strategy.png", output_dir)


def save_all(
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> list[Path]:
    return [
        plot_reconstructed_circles(samples, config, output_dir),
        plot_error_decomposition(samples, metrics, config, output_dir),
        plot_motor_commands(samples, config, output_dir),
    ]


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    from analysis import run_study

    figure_config = StudyConfig()
    figure_samples, figure_metrics = run_study(figure_config)
    for saved in save_all(figure_samples, figure_metrics, figure_config):
        print("saved", saved)
