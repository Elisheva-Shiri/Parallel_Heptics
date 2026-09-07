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
    "cardinal": {"color": "#d7191c", "label": "Cardinal\n4-way, planar solver"},
    "cardinal_diagonal": {"color": "#fdae61", "label": "Cardinal-diagonal\n8-way, planar solver"},
    "free_form": {"color": "#2c7bb6", "label": "Free-form\ncontinuous, planar solver"},
    "ik": {"color": "#1a9641", "label": "IK\ncontinuous, 3-D mechanism solver"},
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


def plot_reconstructed_paths(
    samples: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> Path:
    """Commanded circle vs reconstruction, one panel per movement strategy."""
    strategies = [strategy.value for strategy in config.strategies]
    fig, axes = plt.subplots(1, len(strategies), figsize=(5.0 * len(strategies), 5.8))
    marker_steps = np.linspace(0, config.circle_steps - 1, 12, dtype=int)

    for axis, strategy in zip(np.atleast_1d(axes), strategies):
        part = samples.loc[
            samples["run"].eq(strategy) & samples["segment"].eq(CIRCLE_SEGMENT)
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
        reachable = len(
            np.unique(np.round(part[["quantised_x", "quantised_y"]].to_numpy(float), 6), axis=0)
        )
        commanded = len(
            np.unique(np.round(part[["ideal_x", "ideal_y"]].to_numpy(float), 6), axis=0)
        )
        axis.set_title(
            f"{style['label']}\n"
            f"point-to-point RMS = {rms:.1f} units\n"
            f"{reachable} of {commanded} commanded points reachable",
            fontsize=10,
        )
        axis.set_xlabel("X (controller units)")
        axis.set_ylabel("Y (controller units)")
        axis.set_aspect("equal", "box")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "The four movement strategies, each with its own solver, on the same commanded circle.\n"
        "Every reconstructed point lands on the commanded radius, so shape metrics score all "
        "four as near-perfect circles;\nthe tie lines and the reachable-point count show the "
        "along-path error those metrics cannot see.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, "reconstructed_paths_by_strategy.png", output_dir)


def plot_strategy_comparison(
    metrics: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> Path:
    """Per-strategy error split, and per-strategy command effort."""
    order = [strategy.value for strategy in config.strategies]
    positions = np.arange(len(order))
    labels = [STRATEGY_STYLE[s]["label"] for s in order]

    fig, (ax_error, ax_effort) = plt.subplots(1, 2, figsize=(15.0, 5.6))

    width = 0.38
    quantisation_bars = ax_error.bar(
        positions - width / 2,
        [metrics.loc[s, "quantisation_rms"] for s in order],
        width,
        color="#d7191c",
        label="Quantisation (direction ingredient)",
    )
    execution_bars = ax_error.bar(
        positions + width / 2,
        [metrics.loc[s, "execution_rms"] for s in order],
        width,
        color="#2c7bb6",
        label="Execution (model + truncation)",
    )
    for bars in (quantisation_bars, execution_bars):
        ax_error.bar_label(bars, fmt="%.2f", fontsize=8, padding=1)

    ax_error.set_xticks(positions)
    ax_error.set_xticklabels(labels, fontsize=8)
    ax_error.set_ylabel("RMS error (controller units)")
    ax_error.set_title("Where each strategy's error comes from")
    ax_error.grid(True, alpha=0.3, axis="y")
    ax_error.legend(fontsize=9)

    effort_bars = ax_effort.bar(
        positions,
        [metrics.loc[s, "rms_command"] for s in order],
        0.6,
        color=[STRATEGY_STYLE[s]["color"] for s in order],
    )
    ax_effort.bar_label(effort_bars, fmt="%.1f", fontsize=9, padding=2)
    ax_effort.set_xticks(positions)
    ax_effort.set_xticklabels(labels, fontsize=8)
    ax_effort.set_ylabel("RMS motor command (controller units)")
    ax_effort.set_title("Command effort")
    ax_effort.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return _save(fig, "strategy_comparison.png", output_dir)


#: Publication styling for the paper-facing CD-vs-IK figure. These sizes are
#: tuned for a two-column figure at ICRA scale; do not shrink them casually.
PAPER_TITLE_FS = 48
PAPER_LABEL_FS = 45
PAPER_TICK_FS = 45
PAPER_LEGEND_FS = 45
PAPER_LINE_W = 2.8
PAPER_GRID_ALPHA = 0.28
PAPER_FIGSIZE = (42, 14)

IK_MOTOR_COLORS = {0: "#ff6b35", 1: "#d7191c", 2: "#67000d"}
CD_MOTOR_COLORS = {0: "#f6c85f", 1: "#ff9f1c", 2: "#cc5500"}


def _style_paper_panel(axis) -> None:
    axis.tick_params(axis="both", labelsize=PAPER_TICK_FS)
    axis.title.set_fontsize(PAPER_TITLE_FS)
    axis.xaxis.label.set_size(PAPER_LABEL_FS)
    axis.yaxis.label.set_size(PAPER_LABEL_FS)
    axis.grid(True, alpha=PAPER_GRID_ALPHA)


def _command_scale(part: pd.DataFrame) -> float:
    """Peak |command| across the three motors, used to normalise a method.

    Dividing by this maps a method's commands onto [-1, 1] while keeping zero
    at zero and preserving sign - which a (v - min) / (max - min) mapping would
    not, because the two methods are not symmetric about zero to the same
    degree (planar reaches +/-160, the 3-D mechanism -53 to +37). The residual
    asymmetry stays visible after normalisation, which is intended.
    """
    values = part[["motor_0", "motor_1", "motor_2"]].to_numpy(float)
    peak = float(np.abs(values).max())
    return peak if peak > 0.0 else 1.0


def plot_cd_vs_ik_paper_figure(
    samples: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
    normalise_commands: bool = False,
) -> Path:
    """Paper-facing cardinal-diagonal vs IK comparison (two equal-height panels).

    Left: the circular reconstruction, commanded path overlaid with CD and IK.
    Right: all six motor-command traces, IK solid and CD dashed.

    Note on the right panel: CD runs the planar model and IK the 3-D mechanism,
    so the two express cable deltas differently and share a y axis only for
    layout. Their amplitudes are not directly comparable - see the model note
    in `analysis.py`.
    """
    cd = samples.loc[samples["run"].eq("cardinal_diagonal")].reset_index(drop=True)
    ik = samples.loc[samples["run"].eq("ik")].reset_index(drop=True)
    on_circle = cd["segment"].eq(CIRCLE_SEGMENT)

    fig, (ax_circle, ax_motor) = plt.subplots(
        1,
        2,
        figsize=PAPER_FIGSIZE,
        gridspec_kw={"width_ratios": [0.90, 1.20], "wspace": 0.18},
        constrained_layout=False,
    )

    commanded_handle, = ax_circle.plot(
        ik.loc[on_circle, "ideal_x"], ik.loc[on_circle, "ideal_y"],
        "--", color="0.45", lw=PAPER_LINE_W, label="Commanded", zorder=1,
    )
    ik_handle, = ax_circle.plot(
        ik.loc[on_circle, "reconstructed_x"], ik.loc[on_circle, "reconstructed_y"],
        "-", color="firebrick", lw=PAPER_LINE_W + 0.6, alpha=0.82, label="IK", zorder=2,
    )
    cd_handle, = ax_circle.plot(
        cd.loc[on_circle, "reconstructed_x"], cd.loc[on_circle, "reconstructed_y"],
        "-", color="darkorange", lw=PAPER_LINE_W + 0.2, label="CD", zorder=3,
    )
    ax_circle.set_title("Circular reconstruction")
    ax_circle.set_xlabel("X")
    ax_circle.set_ylabel("Y")
    ax_circle.set_yticks(np.arange(-150, 151, 50))
    ax_circle.set_aspect("equal", "box")
    ax_circle.set_anchor("E")
    _style_paper_panel(ax_circle)
    ax_circle.legend(
        [commanded_handle, ik_handle, cd_handle],
        [handle.get_label() for handle in (commanded_handle, ik_handle, cd_handle)],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
        fontsize=PAPER_LEGEND_FS, frameon=False, handlelength=1.25, columnspacing=0.55,
    )

    ik_scale = _command_scale(ik) if normalise_commands else 1.0
    cd_scale = _command_scale(cd) if normalise_commands else 1.0

    ax_motor.axhline(0.0, color="0.25", lw=1.6, alpha=0.6)
    for motor_index in range(3):
        ax_motor.plot(
            ik["step"], ik[f"motor_{motor_index}"] / ik_scale, "-",
            color=IK_MOTOR_COLORS[motor_index], lw=PAPER_LINE_W + 0.2, alpha=0.82,
            label=f"IK M{motor_index}", zorder=2,
        )
    for motor_index in range(3):
        ax_motor.plot(
            cd["step"], cd[f"motor_{motor_index}"] / cd_scale, "--",
            color=CD_MOTOR_COLORS[motor_index], lw=PAPER_LINE_W + 0.2,
            label=f"CD M{motor_index}", zorder=3,
        )
    ax_motor.set_title("Motor commands")
    ax_motor.set_xlabel("Step")
    if normalise_commands:
        ax_motor.set_ylabel("Normalised command", labelpad=4)
        ax_motor.set_yticks(np.arange(-1.0, 1.01, 0.5))
        ax_motor.set_ylim(-1.1, 1.1)
    else:
        ax_motor.set_ylabel("Command", labelpad=4)
        ax_motor.set_yticks(np.arange(-150, 151, 50))
    _style_paper_panel(ax_motor)
    ax_motor.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6,
        fontsize=PAPER_LEGEND_FS, frameon=False, handlelength=1.25, columnspacing=0.55,
    )

    fig.align_labels()
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.25, top=0.86, wspace=0.18)
    suffix = "_normalised" if normalise_commands else ""
    return _save(
        fig, f"cardinal_diagonal_vs_ik_circle_and_motor_commands{suffix}.png", output_dir
    )


def plot_motor_commands(
    samples: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path | None = None,
) -> Path:
    """Motor commands per strategy, shared axes."""
    strategies = [strategy.value for strategy in config.strategies]
    fig, axes = plt.subplots(
        len(strategies), 1, figsize=(12, 2.9 * len(strategies)), sharex=True, sharey=True
    )

    for axis, strategy in zip(np.atleast_1d(axes), strategies):
        part = samples.loc[samples["run"].eq(strategy)]
        for motor_index, shade in enumerate(("#7b3294", "#c2a5cf", "#008837")):
            axis.plot(
                part["step"],
                part[f"motor_{motor_index}"],
                color=shade,
                linewidth=1.5,
                label=f"Motor {motor_index}",
            )
        axis.axhline(0.0, color="0.4", linewidth=1.0)
        axis.set_title(STRATEGY_STYLE[strategy]["label"].replace("\n", " "), fontsize=10)
        axis.set_ylabel("Command")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=3, loc="upper right")

    np.atleast_1d(axes)[-1].set_xlabel("Path step")
    fig.suptitle(
        "Motor commands, one panel per movement strategy, each with its own solver "
        "(shared y axis).\nIK commands are smaller because the 3-D mechanism moves less "
        "cable per unit of tactor motion than the planar approximation assumes.",
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
        plot_reconstructed_paths(samples, config, output_dir),
        plot_cd_vs_ik_paper_figure(samples, config, output_dir),
        plot_cd_vs_ik_paper_figure(samples, config, output_dir, normalise_commands=True),
        plot_strategy_comparison(metrics, config, output_dir),
        plot_motor_commands(samples, config, output_dir),
    ]


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    from analysis import run_study

    figure_config = StudyConfig()
    figure_samples, figure_metrics = run_study(figure_config)
    for saved in save_all(figure_samples, figure_metrics, figure_config):
        print("saved", saved.name)
