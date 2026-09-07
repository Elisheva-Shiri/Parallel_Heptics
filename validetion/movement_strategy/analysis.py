"""Movement-strategy validation: what does direction quantisation cost?

SCOPE - this study is standalone and concerns the control strategy only.

    One kinematic model is held fixed; the only thing that varies is how
    coarsely a `MovementStrategy` quantises the commanded direction. Every
    quantity is reported in controller units (dimensionless command counts).
    Actuator calibration - servo ticks, spool radius, dead zone, latency - is
    deliberately OUT of scope and lives in `validetion/Servomotor` and
    `validetion/Servo+thimble`. Do not mix those numbers into this study.

WHY A FORWARD MODEL IS NEEDED

    The controller's only output is three cable deltas. To ask whether the
    commanded motion is reproduced, those deltas must be decoded back into a
    tactor position - that is the forward model. Because the same mechanism
    model both encodes and decodes, this measures strategy-induced path error,
    NOT physical mechanism accuracy. FK inverting the controller's IK path is
    itself verified in `tests/test_wire_forward_kinematics.py`.

ERROR DECOMPOSITION

    ideal target ---(strategy quantisation)---> quantised target
                 ---(model + integer truncation)---> reconstructed point

    reported as `quantisation`, `execution` and `total` error respectively.
    Splitting them is what makes the comparison interpretable: quantisation is
    the strategy's intrinsic cost, execution is the shared noise floor.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "motor_controller.py").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root above {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from consts import EDGE_THRESHOLD, TOP_HEIGHT, TOP_WIDTH  # noqa: E402
from kinematics import unified_ik_starter as ik_module  # noqa: E402
from kinematics import wire_forward_kinematics as wire_fk  # noqa: E402
from motor_controller import (  # noqa: E402
    HandOrientation,
    KinematicModel,
    MotorController,
    MotorSetId,
    MovementStrategy,
)

#: Strategies under test. `IK` is excluded on purpose: it is `FREE_FORM` paired
#: with `KinematicModel.IK`, i.e. a model choice, not a strategy. Including it
#: would vary the model and the quantisation at the same time.
STRATEGIES: tuple[MovementStrategy, ...] = (
    MovementStrategy.CARDINAL,
    MovementStrategy.CARDINAL_DIAGONAL,
    MovementStrategy.FREE_FORM,
)

CIRCLE_SEGMENT = "circle"


@dataclass(frozen=True)
class StudyConfig:
    """Every tunable for this study, in one place."""

    # Mechanism / controller setup (mirrors backend usage).
    top_width: float = TOP_WIDTH
    top_height: float = TOP_HEIGHT
    edge_threshold: float = EDGE_THRESHOLD
    motor_spacing: float = 1000.0
    move_factor: float = 1.0
    diagonal_threshold: float = 0.5
    stiffness_value: float = 1.0
    hand_orientation: HandOrientation = HandOrientation.NOT_MIRRORED

    #: Held fixed across strategies. That is the point of the study.
    kinematic_model: KinematicModel = KinematicModel.IK

    # Commanded path: line out, one full circle, line back.
    outward_fraction_of_half_width: float = 0.5
    line_steps: int = 40
    circle_steps: int = 220
    return_steps: int = 40
    sweep_deg: float = -360.0

    strategies: tuple[MovementStrategy, ...] = field(default=STRATEGIES)

    @property
    def radius(self) -> float:
        """Commanded circle radius, in controller units."""
        return (self.top_width / 2.0) * self.outward_fraction_of_half_width


def build_commanded_path(config: StudyConfig) -> pd.DataFrame:
    """Return the commanded virtual-object displacement, one row per step."""
    radius = config.radius
    rows: list[dict[str, float | str]] = []

    for t in np.linspace(0.0, 1.0, config.line_steps + 1):
        rows.append({"segment": "line_out", "x": float(radius * t), "y": 0.0})

    angles = np.linspace(0.0, math.radians(config.sweep_deg), config.circle_steps + 1)[1:]
    for angle in angles:
        rows.append(
            {
                "segment": CIRCLE_SEGMENT,
                "x": float(radius * math.cos(angle)),
                "y": float(radius * math.sin(angle)),
            }
        )

    for t in np.linspace(0.0, 1.0, config.return_steps + 1)[1:]:
        rows.append({"segment": "line_back", "x": float(radius * (1.0 - t)), "y": 0.0})

    path = pd.DataFrame(rows)
    path.insert(0, "step", np.arange(len(path), dtype=int))
    return path


def _make_controller(strategy: MovementStrategy, config: StudyConfig) -> MotorController:
    return MotorController(
        movement_strategy=strategy,
        top_width=config.top_width,
        top_height=config.top_height,
        edge_threshold=config.edge_threshold,
        motor_spacing=config.motor_spacing,
        move_factor=config.move_factor,
        diagonal_threshold=config.diagonal_threshold,
        hand_orientation=config.hand_orientation,
        kinematic_model=config.kinematic_model,
    )


def simulate_strategy(
    strategy: MovementStrategy,
    path: pd.DataFrame,
    config: StudyConfig,
) -> pd.DataFrame:
    """Drive one strategy along the commanded path and decode the result.

    Returns one row per step with the ideal target, the quantised target, the
    reconstructed tactor position and the three motor commands.
    """
    controller = _make_controller(strategy, config)
    model = controller._get_ik_model()
    sim_to_ik = controller._get_ik_base_span(model) / config.motor_spacing
    ik_to_sim = 1.0 / sim_to_ik
    rest_p1 = wire_fk.rest_pose(model)
    previous_p1 = rest_p1.copy()

    positions = {0: 0, 1: 0, 2: 0}
    rows: list[dict[str, float | str | bool]] = []

    for record in path.itertuples(index=False):
        ideal, quantised = controller.resolve_target_point(
            obj_x=record.x, obj_y=record.y, stiffness_value=config.stiffness_value
        )

        for movement in controller.calculate_motor_movements(
            motor_set_id=MotorSetId.MOTORS_0_2,
            stiffness_value=config.stiffness_value,
            obj_x=record.x,
            obj_y=record.y,
            motors_enabled=True,
            reset_to_origin=False,
        ):
            if movement.index in positions:
                positions[movement.index] = movement.pos

        commands = np.array([positions[0], positions[1], positions[2]], dtype=float)
        solution = wire_fk.solve_wire_fk(
            commands * sim_to_ik, model, initial_P1=previous_p1, rest_P1=rest_p1
        )
        previous_p1 = solution.P1

        reconstructed = (
            float(solution.P1[0] * ik_to_sim),
            float(solution.P1[1] * ik_to_sim),
        )

        rows.append(
            {
                "strategy": strategy.value,
                "step": int(record.step),
                "segment": record.segment,
                "object_x": float(record.x),
                "object_y": float(record.y),
                "ideal_x": ideal[0],
                "ideal_y": ideal[1],
                "quantised_x": quantised[0],
                "quantised_y": quantised[1],
                "reconstructed_x": reconstructed[0],
                "reconstructed_y": reconstructed[1],
                "motor_0": commands[0],
                "motor_1": commands[1],
                "motor_2": commands[2],
                "fk_valid": bool(solution.valid),
                "fk_residual": float(solution.residual_rms * ik_to_sim),
                "quantisation_error": math.dist(ideal, quantised),
                "execution_error": math.dist(quantised, reconstructed),
                "total_error": math.dist(ideal, reconstructed),
            }
        )

    return pd.DataFrame(rows)


def _radial_rms(points: np.ndarray, radius: float) -> float:
    """RMS deviation from the commanded radius (not a self-fitted one)."""
    if len(points) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.linalg.norm(points, axis=1) - radius) ** 2)))


def _circle_fit_rms(points: np.ndarray) -> float:
    """RMS radial deviation about the best-fit circle through the points."""
    if len(points) < 3:
        return float("nan")
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    cx, cy, c = np.linalg.lstsq(design, x**2 + y**2, rcond=None)[0]
    fitted_radius = math.sqrt(max(c + cx**2 + cy**2, 0.0))
    deviation = np.hypot(x - cx, y - cy) - fitted_radius
    return float(np.sqrt(np.mean(deviation**2)))


def strategy_metrics(samples: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    """One row per strategy. All values in controller units."""
    rows: list[dict[str, float | str | int]] = []

    for strategy in config.strategies:
        part = samples.loc[samples["strategy"].eq(strategy.value)]
        circle = part.loc[part["segment"].eq(CIRCLE_SEGMENT)]
        reconstructed = circle[["reconstructed_x", "reconstructed_y"]].to_numpy(float)

        x_span = float(reconstructed[:, 0].max() - reconstructed[:, 0].min())
        y_span = float(reconstructed[:, 1].max() - reconstructed[:, 1].min())
        commands = part[["motor_0", "motor_1", "motor_2"]].to_numpy(float)
        steps = np.diff(commands, axis=0)
        last = part.iloc[-1]

        rows.append(
            {
                "strategy": strategy.value,
                "samples": int(len(part)),
                # --- what the strategy costs ---
                "quantisation_rms": float(np.sqrt(np.mean(part["quantisation_error"] ** 2))),
                "quantisation_max": float(part["quantisation_error"].max()),
                # --- shared noise floor (model + integer truncation) ---
                "execution_rms": float(np.sqrt(np.mean(part["execution_error"] ** 2))),
                "execution_max": float(part["execution_error"].max()),
                # --- end to end ---
                "total_rms": float(np.sqrt(np.mean(part["total_error"] ** 2))),
                "total_max": float(part["total_error"].max()),
                # --- shape of the reconstructed circle ---
                "radial_rms": _radial_rms(reconstructed, config.radius),
                "circle_fit_rms": _circle_fit_rms(reconstructed),
                "aspect_ratio": float(y_span / x_span) if x_span else float("nan"),
                # --- path closure ---
                "closure_error": float(
                    math.dist(
                        (last["reconstructed_x"], last["reconstructed_y"]),
                        (last["ideal_x"], last["ideal_y"]),
                    )
                ),
                # --- command effort (comparable: one model for all rows) ---
                "peak_abs_command": float(np.abs(commands).max()),
                "rms_command": float(np.sqrt(np.mean(commands**2))),
                "total_motor_travel": float(np.abs(steps).sum()),
                "max_step_jump": float(np.abs(steps).max()),
                "fk_invalid_steps": int((~part["fk_valid"]).sum()),
            }
        )

    return pd.DataFrame(rows).set_index("strategy")


def run_study(config: StudyConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every strategy on the fixed model. Returns (samples, metrics)."""
    config = config or StudyConfig()
    path = build_commanded_path(config)
    samples = pd.concat(
        [simulate_strategy(strategy, path, config) for strategy in config.strategies],
        ignore_index=True,
    )
    return samples, strategy_metrics(samples, config)


if __name__ == "__main__":
    study_config = StudyConfig()
    study_samples, study_metrics = run_study(study_config)

    pd.set_option("display.width", 200)
    print(
        f"Commanded path: {len(study_samples) // len(study_config.strategies)} samples, "
        f"radius {study_config.radius:.0f} controller units, "
        f"model held fixed at {study_config.kinematic_model.value}."
    )
    print("\nAll values in controller units.\n")
    print(
        study_metrics[
            [
                "quantisation_rms",
                "execution_rms",
                "total_rms",
                "total_max",
                "radial_rms",
                "aspect_ratio",
                "closure_error",
            ]
        ].round(3)
    )
    print("\nCommand effort (comparable - one kinematic model for every row):\n")
    print(
        study_metrics[
            ["peak_abs_command", "rms_command", "total_motor_travel", "max_step_jump"]
        ].round(3)
    )
