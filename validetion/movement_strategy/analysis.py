"""Movement-strategy validation: planar model vs 3-D mechanism.

SCOPE - this study is standalone and concerns the control strategy only.

    Every quantity is reported in controller units (dimensionless command
    counts). Actuator calibration - servo ticks, spool radius, dead zone,
    latency - is deliberately OUT of scope and lives in `validetion/Servomotor`
    and `validetion/Servo+thimble`. Do not mix those numbers into this study.

THE FOUR STRATEGIES

    Each runs exactly as the device defines it, with its own solver:

        cardinal            4 directions    planar model
        cardinal_diagonal   8 directions    planar model
        free_form           any direction   planar model
        ik                  any direction   3-D mechanism model

THE TWO KINEMATIC MODELS

    Both act on the SAME anchor triangle, which is isosceles, not equilateral:
    base 24.2, equal sides 28.6, apex height 25.9 (model units, mm).

    planar - each cable is a straight line from its anchor to the tactor,
        in 2-D. Cable delta is simply the change in that distance, so cable
        travel is assumed equal to tactor travel.

    3-D mechanism - solves the full leg mechanism: link lengths d1=4.0,
        d2=11.0, d3=9.5, legs rotated +/-120 degrees, tactor rest height
        z=6.0, Bowden cable attached at the midpoint of the first link. The
        cable displaces considerably less than the tactor moves, which is why
        its commands are roughly 3.8x smaller for the same commanded path.

WHY A FORWARD MODEL IS NEEDED

    The controller's only output is three cable deltas. To ask whether the
    commanded motion is reproduced, those deltas must be decoded back into a
    tactor position - that is the forward model. Each model is decoded by its
    own inverse: least-squares trilateration for the planar model, wire FK for
    the 3-D mechanism. So this measures internal consistency and command cost,
    NOT measured mechanism accuracy - deciding which model better describes the
    real device needs a measured tactor position, which is the Servo+thimble
    study. That the wire FK genuinely inverts the controller's IK path is
    verified in `tests/test_wire_forward_kinematics.py`.

ERROR DECOMPOSITION

    ideal target ---(strategy direction resolution)---> quantised target
                 ---(model + integer truncation)-----> reconstructed point

    reported as `quantisation`, `execution` and `total` error respectively.
    Direction resolution dominates path error; the model choice dominates
    command amplitude.
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
from haptic_mapping import map_object_displacement_to_tactor  # noqa: E402
from kinematics import unified_ik_starter as ik_module  # noqa: E402
from kinematics import wire_forward_kinematics as wire_fk  # noqa: E402
from motor_controller import (  # noqa: E402
    STRATEGY_DEFINITIONS,
    HandOrientation,
    KinematicModel,
    MotorController,
    MotorSetId,
    MovementStrategy,
    Quantisation,
)

#: The four movement strategies as the device runs them - the experimental
#: conditions. Each is reported as itself, with the ingredients it is built
#: from (see `motor_controller.STRATEGY_DEFINITIONS`) shown alongside, so a
#: difference between two strategies can be traced to the ingredient that
#: caused it.
STRATEGIES: tuple[MovementStrategy, ...] = (
    MovementStrategy.CARDINAL,
    MovementStrategy.CARDINAL_DIAGONAL,
    MovementStrategy.FREE_FORM,
    MovementStrategy.IK,
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

    #: Mirrors `backend.MOTOR_OPPOSES_OBJECT_MOTION`. The backend maps object
    #: displacement to a tactor target with `map_object_displacement_to_tactor`
    #: BEFORE calling the controller, and the controller then applies its own
    #: actuator polarity flip. This study reproduces that call pattern exactly
    #: so it represents the device rather than an idealised version of it.
    oppose_object_motion: bool = True

    #: Leave as None so every strategy uses the solver it is actually defined
    #: with (planar for cardinal / cardinal-diagonal / free-form, the 3-D
    #: mechanism for IK). Setting it overrides that for all four, which is only
    #: meaningful as a deliberate what-if - it is not how the device runs.
    kinematic_model: KinematicModel | None = None

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


def planar_anchors(config: StudyConfig) -> np.ndarray:
    """The 2-D anchor triangle, in controller units (order: top, right, left)."""
    anchors = ik_module.default_model()["anchors"]
    base_span = math.hypot(
        anchors["right"][0] - anchors["left"][0],
        anchors["right"][1] - anchors["left"][1],
    )
    scale = config.motor_spacing / base_span
    return np.array([anchors[leg][:2] for leg in ("top", "right", "left")], float) * scale


def _trilaterate(anchors: np.ndarray, lengths: np.ndarray) -> tuple[float, float, float]:
    """Least-squares planar trilateration: the inverse of the planar model.

    Returns (x, y, residual_rms). The residual is a self-consistency measure of
    the three distances, not a reconstruction error.
    """
    first, first_length = anchors[0], lengths[0]
    design, target = [], []
    for index in (1, 2):
        xi, yi = anchors[index]
        design.append([2.0 * (xi - first[0]), 2.0 * (yi - first[1])])
        target.append(
            (first_length**2 - lengths[index] ** 2)
            - (first[0] ** 2 - xi**2)
            - (first[1] ** 2 - yi**2)
        )
    solution, *_ = np.linalg.lstsq(np.asarray(design, float), np.asarray(target, float), rcond=None)
    x, y = float(solution[0]), float(solution[1])
    predicted = np.linalg.norm(anchors - np.array([x, y]), axis=1)
    return x, y, float(np.sqrt(np.mean((predicted - lengths) ** 2)))


def _make_controller(
    config: StudyConfig,
    strategy: MovementStrategy,
    quantisation: Quantisation | None = None,
    kinematic_model: KinematicModel | None = None,
) -> MotorController:
    return MotorController(
        movement_strategy=strategy,
        top_width=config.top_width,
        top_height=config.top_height,
        edge_threshold=config.edge_threshold,
        motor_spacing=config.motor_spacing,
        move_factor=config.move_factor,
        diagonal_threshold=config.diagonal_threshold,
        hand_orientation=config.hand_orientation,
        quantisation=quantisation,
        kinematic_model=kinematic_model,
    )


def simulate_run(
    label: str,
    controller: MotorController,
    path: pd.DataFrame,
    config: StudyConfig,
) -> pd.DataFrame:
    """Drive one configured controller along the commanded path and decode it.

    Each model is decoded by its own inverse - trilateration for `PLANAR`, wire
    FK for `IK` - because a decoder must match the encoder that produced the
    cable deltas. Both decoders return controller-unit positions, so the
    reconstructed points stay directly comparable.
    """
    model = controller._get_ik_model()
    sim_to_ik = controller._get_ik_base_span(model) / config.motor_spacing
    ik_to_sim = 1.0 / sim_to_ik
    rest_p1 = wire_fk.rest_pose(model)
    previous_p1 = rest_p1.copy()

    anchors = planar_anchors(config)
    rest_lengths = np.linalg.norm(anchors - np.array([0.0, 0.0]), axis=1)
    uses_ik = controller._kinematic_model is KinematicModel.IK

    positions = {0: 0, 1: 0, 2: 0}
    rows: list[dict[str, float | str | bool]] = []

    for record in path.itertuples(index=False):
        # Same two-step mapping the backend performs (backend.py:1123-1137).
        tactor_x, tactor_y = map_object_displacement_to_tactor(
            obj_x=record.x, obj_y=record.y, oppose_motion=config.oppose_object_motion
        )
        ideal, quantised = controller.resolve_target_point(
            obj_x=tactor_x, obj_y=tactor_y, stiffness_value=config.stiffness_value
        )

        for movement in controller.calculate_motor_movements(
            motor_set_id=MotorSetId.MOTORS_0_2,
            stiffness_value=config.stiffness_value,
            obj_x=tactor_x,
            obj_y=tactor_y,
            motors_enabled=True,
            reset_to_origin=False,
        ):
            if movement.index in positions:
                positions[movement.index] = movement.pos

        commands = np.array([positions[0], positions[1], positions[2]], dtype=float)

        if uses_ik:
            solution = wire_fk.solve_wire_fk(
                commands * sim_to_ik, model, initial_P1=previous_p1, rest_P1=rest_p1
            )
            previous_p1 = solution.P1
            reconstructed = (
                float(solution.P1[0] * ik_to_sim),
                float(solution.P1[1] * ik_to_sim),
            )
            decode_valid = bool(solution.valid)
            decode_residual = float(solution.residual_rms * ik_to_sim)
        else:
            lengths = rest_lengths + commands
            x, y, decode_residual = _trilaterate(anchors, lengths)
            reconstructed = (x, y)
            decode_valid = bool(np.all(lengths > 0.0))

        rows.append(
            {
                "run": label,
                "strategy": controller._movement_strategy.value,
                "quantisation": controller._quantisation.value,
                "kinematic_model": controller._kinematic_model.value,
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
                "decode_valid": decode_valid,
                "decode_residual": decode_residual,
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


def run_metrics(samples: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    """One row per run. All values in controller units."""
    rows: list[dict[str, float | str | int]] = []

    for label in samples["run"].unique():
        part = samples.loc[samples["run"].eq(label)]
        circle = part.loc[part["segment"].eq(CIRCLE_SEGMENT)]
        reconstructed = circle[["reconstructed_x", "reconstructed_y"]].to_numpy(float)

        x_span = float(reconstructed[:, 0].max() - reconstructed[:, 0].min())
        y_span = float(reconstructed[:, 1].max() - reconstructed[:, 1].min())
        commands = part[["motor_0", "motor_1", "motor_2"]].to_numpy(float)
        steps = np.diff(commands, axis=0)
        last = part.iloc[-1]

        rows.append(
            {
                "run": label,
                "strategy": part["strategy"].iat[0],
                "quantisation": part["quantisation"].iat[0],
                "kinematic_model": part["kinematic_model"].iat[0],
                "samples": int(len(part)),
                # --- what the quantisation ingredient costs ---
                "quantisation_rms": float(np.sqrt(np.mean(part["quantisation_error"] ** 2))),
                "quantisation_max": float(part["quantisation_error"].max()),
                # --- what the model ingredient + integer truncation cost ---
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
                # --- command effort ---
                "peak_abs_command": float(np.abs(commands).max()),
                "rms_command": float(np.sqrt(np.mean(commands**2))),
                "total_motor_travel": float(np.abs(steps).sum()),
                "max_step_jump": float(np.abs(steps).max()),
                "decode_invalid_steps": int((~part["decode_valid"]).sum()),
            }
        )

    return pd.DataFrame(rows).set_index("run")


def run_study(config: StudyConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the four movement strategies, each with the solver it is defined with."""
    config = config or StudyConfig()
    path = build_commanded_path(config)
    samples = pd.concat(
        [
            simulate_run(
                strategy.value,
                _make_controller(config, strategy, kinematic_model=config.kinematic_model),
                path,
                config,
            )
            for strategy in config.strategies
        ],
        ignore_index=True,
    )
    return samples, run_metrics(samples, config)


def comparison_table(metrics: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    """The headline strategy comparison, in controller units."""
    names = [strategy.value for strategy in config.strategies]
    columns = [
        "quantisation",
        "kinematic_model",
        "total_rms",
        "total_max",
        "quantisation_rms",
        "execution_rms",
        "rms_command",
        "max_step_jump",
        "closure_error",
    ]
    return metrics.loc[names, columns]


if __name__ == "__main__":
    study_config = StudyConfig()
    study_samples, study_metrics = run_study(study_config)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    print(
        "Four movement strategies, each with the solver it is defined with. "
        f"Radius {study_config.radius:.0f} controller units.\n"
    )
    print(comparison_table(study_metrics, study_config).round(3).to_string())
