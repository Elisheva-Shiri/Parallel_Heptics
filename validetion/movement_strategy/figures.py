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
    kinematic_model: str = "ik",
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
            samples["strategy"].eq(strategy)
            & samples["segment"].eq(CIRCLE_SEGMENT)
            & samples["kinematic_model"].eq(kinematic_model)
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
    return _save(fig, f"reconstructed_circles_{kinematic_model}.png", output_dir)


MODEL_HATCH = {"planar": "", "ik": "//"}
MODEL_LABEL = {"planar": "Planar model", "ik": "IK model"}


def plot_error_decomposition(
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> Path:
    """The full strategy x model factorial, split into its two effects.

    Left: reading within a model gives the quantisation (strategy) effect;
    comparing hatched against plain gives the model effect. Right: command
    effort, where the model effect dominates - this is the difference the
    historic CD-vs-IK comparison actually measured while attributing it to
    the strategy.
    """
    order = [s.value for s in config.strategies]
    models = [m.value for m in config.kinematic_models]
    positions = np.arange(len(order))
    width = 0.8 / (len(models) * 2)

    fig, (ax_error, ax_effort) = plt.subplots(1, 2, figsize=(14.5, 5.4))

    for model_index, model in enumerate(models):
        quantisation = [metrics.loc[(model, s), "quantisation_rms"] for s in order]
        execution = [metrics.loc[(model, s), "execution_rms"] for s in order]
        offset = (model_index * 2 - len(models) + 0.5) * width

        quantisation_bars = ax_error.bar(
            positions + offset,
            quantisation,
            width,
            color="#d7191c",
            hatch=MODEL_HATCH[model],
            edgecolor="white",
            label=f"Quantisation - {MODEL_LABEL[model]}",
        )
        execution_bars = ax_error.bar(
            positions + offset + width,
            execution,
            width,
            color="#2c7bb6",
            hatch=MODEL_HATCH[model],
            edgecolor="white",
            label=f"Execution - {MODEL_LABEL[model]}",
        )
        # The execution bars are ~20x shorter than the quantisation bars, which
        # is the point - but it makes them unreadable, so label every value.
        for bars in (quantisation_bars, execution_bars):
            ax_error.bar_label(bars, fmt="%.2f", fontsize=7, padding=1, rotation=90)

        ax_effort.bar(
            positions + (model_index - len(models) / 2 + 0.5) * 0.35,
            [metrics.loc[(model, s), "rms_command"] for s in order],
            0.35,
            color="#5e3c99" if model == "planar" else "#e66101",
            label=MODEL_LABEL[model],
        )

    ax_error.set_xticks(positions)
    ax_error.set_xticklabels([STRATEGY_STYLE[s]["label"] for s in order], fontsize=9)
    ax_error.set_ylabel("RMS error (controller units)")
    ax_error.set_title(
        "Strategy cost vs execution floor\n"
        "quantisation is identical across models, as it must be"
    )
    ax_error.grid(True, alpha=0.3, axis="y")
    ax_error.legend(fontsize=8)

    ax_effort.set_xticks(positions)
    ax_effort.set_xticklabels([STRATEGY_STYLE[s]["label"] for s in order], fontsize=9)
    ax_effort.set_ylabel("RMS motor command (controller units)")
    ax_effort.set_title(
        "Command effort is a MODEL effect, not a strategy effect\n"
        "the IK mechanism transmits ~3.8x less cable travel per unit of tactor motion"
    )
    ax_effort.grid(True, alpha=0.3, axis="y")
    ax_effort.legend(fontsize=9)

    fig.tight_layout()
    return _save(fig, "error_decomposition.png", output_dir)


def plot_motor_commands(
    samples: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> Path:
    """Motor commands per strategy - now comparable, one model for every row."""
    strategies = [s.value for s in config.strategies]
    models = [m.value for m in config.kinematic_models]
    fig, axes = plt.subplots(
        len(strategies), len(models),
        figsize=(6.5 * len(models), 3.1 * len(strategies)),
        sharex=True, sharey=True, squeeze=False,
    )

    for row, strategy in enumerate(strategies):
      for column, model in enumerate(models):
        axis = axes[row][column]
        part = samples.loc[
            samples["strategy"].eq(strategy) & samples["kinematic_model"].eq(model)
        ]
        for motor_index, shade in enumerate(("#7b3294", "#c2a5cf", "#008837")):
            axis.plot(
                part["step"],
                part[f"motor_{motor_index}"],
                color=shade,
                linewidth=1.5,
                label=f"Motor {motor_index}",
            )
        axis.axhline(0.0, color="0.4", linewidth=1.0)
        axis.set_title(
            f"{STRATEGY_STYLE[strategy]['label']} - {MODEL_LABEL[model]}", fontsize=10
        )
        axis.set_ylabel("Command")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=3, loc="upper right")

    for axis in axes[-1]:
        axis.set_xlabel("Path step")
    fig.suptitle(
        "Motor commands, every strategy on both models (shared y axis).\n"
        "Amplitude is set by the MODEL, not the strategy; the strategy's own signature is "
        "the discontinuous jump when a quantised direction snaps to a new sector.",
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
        *(
            plot_reconstructed_circles(samples, config, output_dir, model.value)
            for model in config.kinematic_models
        ),
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
