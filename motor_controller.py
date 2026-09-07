import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kinematics import unified_ik_starter as ik
from kinematics import wire_forward_kinematics as wire_fk

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11 notebook kernels
    class StrEnum(str, Enum):
        pass


class MotorSetId(Enum):
    """Physical 3-motor clusters on the 16-channel servo driver."""

    MOTORS_0_2 = 0
    MOTORS_3_5 = 1
    MOTORS_6_8 = 2
    MOTORS_9_11 = 3
    MOTORS_12_14 = 4

    @property
    def base_index(self) -> int:
        return self.value * 3

    @property
    def label(self) -> str:
        return f"{self.base_index}-{self.base_index + 2}"


class MovementStrategy(StrEnum):
    """The movement strategies the device can run - the experimental conditions.

    Each strategy is a complete, named way of driving the tactor, and each is
    built from two independent ingredients: how the commanded direction is
    quantised (`Quantisation`) and which kinematic model converts the resulting
    target point into cable deltas (`KinematicModel`). See
    `STRATEGY_DEFINITIONS` for the exact composition of each one.

    Spelling the ingredients out matters when strategies are compared: two
    strategies that differ in both ingredients cannot have their difference
    attributed to either one.
    """

    CARDINAL = "cardinal"
    CARDINAL_DIAGONAL = "cardinal_diagonal"
    FREE_FORM = "free_form"
    IK = "ik"


class Quantisation(StrEnum):
    """How coarsely a strategy may express the commanded direction."""

    #: Any direction; the target point passes through untouched.
    NONE = "none"
    #: Snap to the nearest of 4 axis directions.
    CARDINAL_4 = "cardinal_4"
    #: Snap to the nearest of 8, subject to `diagonal_threshold`.
    CARDINAL_8 = "cardinal_8"


class KinematicModel(StrEnum):
    """Model used to turn a target point into per-motor cable deltas.

    `PLANAR` treats each cable as a straight line from its anchor to the tactor
    in 2-D. `IK` solves the full 3-D leg mechanism and measures Bowden-cable
    displacement.
    """

    PLANAR = "planar"
    IK = "ik"


#: What each strategy is made of. This is the single place the two ingredients
#: are bound together, and it is what makes the composition of a strategy
#: inspectable instead of implicit.
#:
#: Note that the four strategies populate only four of the six possible
#: (quantisation, model) cells, and that CARDINAL_DIAGONAL and IK differ in
#: BOTH ingredients - which is why comparing those two in isolation cannot
#: attribute a result to the quantisation or to the model.
STRATEGY_DEFINITIONS: dict[str, tuple["Quantisation", "KinematicModel"]] = {
    MovementStrategy.CARDINAL: (Quantisation.CARDINAL_4, KinematicModel.PLANAR),
    MovementStrategy.CARDINAL_DIAGONAL: (Quantisation.CARDINAL_8, KinematicModel.PLANAR),
    MovementStrategy.FREE_FORM: (Quantisation.NONE, KinematicModel.PLANAR),
    MovementStrategy.IK: (Quantisation.NONE, KinematicModel.IK),
}


class HandOrientation(StrEnum):
    """Hand orientation relative to the controller coordinate frame."""

    NOT_MIRRORED = "not_mirrored"
    MIRRORED = "mirrored"


@dataclass
class MotorMovement:
    """Single motor command.

    Attributes:
        pos: Target position delta for the motor.
        index: Global motor index in the controller layout.
    """

    pos: int
    index: int


class MotorController:
    """Compute and format motor movements for a triangular 3-motor cluster.

    Motor clusters are selected by physical motor set (0-2, 3-5, ...), not by
    finger identity. The active tracked finger is decided by the experiment
    configuration; this class only knows which motor indices to command.
    """

    def __init__(
        self,
        movement_strategy: MovementStrategy,
        top_width: float,
        top_height: float,
        edge_threshold: float,
        motor_spacing: float = 1000.0,
        move_factor: float = 1.0,
        diagonal_threshold: float = 0.5,
        hand_orientation: HandOrientation = HandOrientation.NOT_MIRRORED,
        kinematic_model: "KinematicModel | None" = None,
        quantisation: "Quantisation | None" = None,
    ):
        """Create a motor controller with geometry and movement parameters.

        Args:
            movement_strategy: Direction quantisation applied to the target point.
            top_width: Width of the active movement area.
            top_height: Height of the active movement area.
            edge_threshold: Margin near the boundary where free-form motion is clipped.
            motor_spacing: Distance between motors in a single physical motor cluster.
            move_factor: Scalar applied to all output motor deltas.
            diagonal_threshold: Min axis-ratio needed to classify movement as diagonal
                in `MovementStrategy.CARDINAL_DIAGONAL`.
            hand_orientation: Whether to mirror left/right motion on the X axis.
            kinematic_model: Overrides the model this strategy is defined with.
                Leave as None for normal use; set it only to hold one
                ingredient fixed while varying the other, so a difference can
                be attributed to one of them.
            quantisation: Overrides the quantisation this strategy is defined
                with. Same purpose as `kinematic_model`.
        """
        self._movement_strategy = movement_strategy
        default_quantisation, default_model = STRATEGY_DEFINITIONS[movement_strategy]
        self._quantisation = quantisation if quantisation is not None else default_quantisation
        self._kinematic_model = (
            kinematic_model if kinematic_model is not None else default_model
        )
        self._hand_orientation = hand_orientation
        self._top_width = top_width
        self._top_height = top_height
        self._edge_threshold = edge_threshold
        self._motor_spacing = motor_spacing
        self._move_factor = move_factor
        self._diagonal_threshold = diagonal_threshold
        self._ik_module: Any | None = None
        self._ik_model: dict[str, Any] | None = None
        self._ik_wire_reference: dict[str, float] | None = None
        self._ik_rest_result: dict[str, Any] | None = None
        self._ik_previous_angles: dict[str, dict[str, float]] | None = None

    def build_message(self, motors: list[MotorMovement]) -> str:
        """Build a firmware command string from motor movements.

        Output format:
            ZM{index}P{pos}M{index}P{pos}...F
        """
        message = "Z"
        for motor in motors:
            message += f"M{motor.index}P{motor.pos}"
        message += "F"
        return message

    def calculate_motor_movements(
        self,
        motor_set_id: MotorSetId,
        stiffness_value: float = 1.0,
        obj_x: float = 0.0,
        obj_y: float = 0.0,
        motors_enabled: bool = True,
        reset_to_origin: bool = False,
    ) -> list[MotorMovement]:
        """Compute motor deltas for a physical motor set from object displacement.

        Args:
            motor_set_id: Physical motor cluster to control, e.g. `MOTORS_0_2`.
            stiffness_value: Scalar applied to object displacement before kinematics.
            obj_x: X displacement from center.
            obj_y: Y displacement from center.
            motors_enabled: If `False`, movement is suppressed unless reset is requested.
            reset_to_origin: If `True` while motors are disabled, returns zero
                commands for the selected motor set.

        Returns:
            A list of per-motor deltas. Returns an empty list if no movement is needed.
        """
        if not motors_enabled:
            if not reset_to_origin:
                return []
            return self.zero_motor_positions(motor_set_id)

        obj_x, obj_y = self._apply_hand_orientation(obj_x, obj_y)
        obj_x, obj_y = self._apply_stiffness_value(obj_x, obj_y, stiffness_value)
        obj_x, obj_y = self._apply_actuator_destination_polarity(obj_x, obj_y)

        quantised = self._quantisation is not Quantisation.NONE
        if quantised and obj_x == 0 and obj_y == 0:
            # Historic behaviour: a quantised strategy emits nothing at rest,
            # while free-form/IK still emit three explicit zero commands.
            return []

        target_x, target_y = self._quantize_target(obj_x, obj_y)

        match self._kinematic_model:
            case KinematicModel.PLANAR:
                movements = self._calculate_planar_motor_movements(motor_set_id, target_x, target_y)
            case KinematicModel.IK:
                movements = self._calculate_ik_motor_movements(motor_set_id, target_x, target_y)
            case _:
                raise NotImplementedError(f"Unknown kinematic model: {self._kinematic_model}")
        return self._apply_move_factor(movements)

    def zero_motor_positions(self, motor_set_id: MotorSetId) -> list[MotorMovement]:
        """Return zero-position commands for all 3 motors in a physical motor set."""
        self._reset_ik_state_to_origin()
        base_motor_idx = motor_set_id.base_index
        return [
            MotorMovement(pos=0, index=base_motor_idx),
            MotorMovement(pos=0, index=base_motor_idx + 1),
            MotorMovement(pos=0, index=base_motor_idx + 2),
        ]

    def _apply_hand_orientation(self, obj_x: float, obj_y: float) -> tuple[float, float]:
        """Convert object displacement into controller frame based on hand orientation."""
        if self._hand_orientation == HandOrientation.MIRRORED:
            return (-obj_x, obj_y)
        return (obj_x, obj_y)

    def _apply_stiffness_value(self, obj_x: float, obj_y: float, stiffness_value: float) -> tuple[float, float]:
        """Apply stiffness value to object displacement."""
        return (obj_x * stiffness_value, obj_y * stiffness_value)

    def _apply_actuator_destination_polarity(self, obj_x: float, obj_y: float) -> tuple[float, float]:
        """Use the opposite target position because actuator motion is reversed."""
        return (-obj_x, -obj_y)

    def resolve_target_point(
        self,
        obj_x: float,
        obj_y: float,
        stiffness_value: float = 1.0,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the tactor points this controller aims at, before any cables.

        Runs the same input conditioning as `calculate_motor_movements` but
        stops before the kinematic model, so callers can separate the error a
        strategy introduces by quantising from the error introduced downstream
        by the model and integer command truncation.

        Returns:
            Tuple of (`ideal`, `quantised`) target points. `ideal` is where an
            unquantised strategy would aim; `quantised` is where this strategy
            actually aims. They are equal for `FREE_FORM` and `IK`.
        """
        x, y = self._apply_hand_orientation(obj_x, obj_y)
        x, y = self._apply_stiffness_value(x, y, stiffness_value)
        x, y = self._apply_actuator_destination_polarity(x, y)
        ideal = self._clamp_to_workspace(x, y)
        quantised = self._clamp_to_workspace(*self._quantize_target(x, y))
        return ideal, quantised

    def _quantize_target(self, obj_x: float, obj_y: float) -> tuple[float, float]:
        """Snap the target point to the directions the strategy can express.

        A pure point-to-point map driven by `self._quantisation`, which a
        strategy supplies but does not otherwise influence. Holding the model
        fixed and varying only this isolates the direction-quantisation effect.

        Quantisation is radius-preserving: the snapped point keeps
        ``hypot(obj_x, obj_y)``, so only the direction is degraded, never the
        magnitude. Diagonals are emitted only for `Quantisation.CARDINAL_8` and
        only when the axis ratio reaches `self._diagonal_threshold`.

        Returns:
            The target point the kinematic model should drive the tactor to.
        """
        if self._quantisation is Quantisation.NONE:
            return (obj_x, obj_y)

        abs_x, abs_y = abs(obj_x), abs(obj_y)
        if abs_x == 0 and abs_y == 0:
            return (0.0, 0.0)

        radius = math.hypot(obj_x, obj_y)

        if self._quantisation is Quantisation.CARDINAL_8 and abs_x > 0 and abs_y > 0:
            ratio = min(abs_x, abs_y) / max(abs_x, abs_y)
            if ratio >= self._diagonal_threshold:
                leg = radius / math.sqrt(2.0)
                return (math.copysign(leg, obj_x), math.copysign(leg, obj_y))

        if abs_x > abs_y:
            return (math.copysign(radius, obj_x), 0.0)
        return (0.0, math.copysign(radius, obj_y))

    def _calculate_planar_motor_movements(
        self,
        motor_set_id: MotorSetId,
        obj_x: float,
        obj_y: float,
    ) -> list[MotorMovement]:
        """
        Convert a target point into cable deltas with the planar model.

        Each cable is treated as a straight 2-D line from its anchor to the
        tactor. This is the model historically used by the cardinal,
        cardinal-diagonal and free-form strategies.

        Args:
            motor_set_id: Physical motor set to control.
            obj_x: X displacement from center.
            obj_y: Y displacement from center.

        Returns:
            List of per-motor deltas, clipped to the configured workspace radius.
        """
        obj_x, obj_y = self._clamp_to_workspace(obj_x, obj_y)
        return self._calculate_movements_to_point(motor_set_id, (obj_x, obj_y))

    def _calculate_ik_motor_movements(
        self,
        motor_set_id: MotorSetId,
        obj_x: float,
        obj_y: float,
    ) -> list[MotorMovement]:
        """Compute motor commands through the unified IK and wire model."""
        ik = self._get_ik_module()
        model = self._get_ik_model()
        self._ensure_ik_wire_reference()

        obj_x, obj_y = self._clamp_to_workspace(obj_x, obj_y)
        sim_to_ik_scale = self._get_ik_base_span(model) / self._motor_spacing
        ik_to_sim_scale = 1.0 / sim_to_ik_scale
        p1 = (obj_x * sim_to_ik_scale, obj_y * sim_to_ik_scale, self._get_ik_tactor_z(model))

        result = ik.solve_all_legs(
            P1=p1,
            phi1=math.pi / 2.0,
            previous_angles=self._ik_previous_angles,
            model=model,
        )
        if not all(result[leg]["valid"] for leg in ("top", "right", "left")):
            invalid = {
                leg: result[leg]["fail_reason"]
                for leg in ("top", "right", "left")
                if not result[leg]["valid"]
            }
            raise ValueError(f"IK solution is invalid for target {p1}: {invalid}")

        self._ik_previous_angles = self._extract_ik_angles(result)
        base_motor_idx = motor_set_id.base_index
        wire_deltas = self._calculate_ik_mechanism_wire_deltas(result, model)
        movements: list[MotorMovement] = []
        for i, leg in enumerate(("top", "right", "left")):
            movements.append(MotorMovement(pos=int(wire_deltas[leg] * ik_to_sim_scale), index=base_motor_idx + i))
        return movements

    def _get_ik_module(self) -> Any:
        if self._ik_module is None:
            self._ik_module = ik
        return self._ik_module

    def _get_ik_model(self) -> dict[str, Any]:
        if self._ik_model is None:
            self._ik_model = self._get_ik_module().default_model()
        return self._ik_model

    def _ensure_ik_wire_reference(self) -> None:
        if self._ik_wire_reference is not None:
            return

        ik = self._get_ik_module()
        model = self._get_ik_model()
        result = ik.solve_all_legs((0.0, 0.0, self._get_ik_tactor_z(model)), math.pi / 2.0, model=model)
        if not all(result[leg]["valid"] for leg in ("top", "right", "left")):
            raise ValueError("IK origin reference is invalid.")

        reference_p1 = result["shared"]["P1"]
        self._ik_rest_result = result
        self._ik_wire_reference = {leg: 0.0 for leg in ("top", "right", "left")}
        self._ik_previous_angles = self._extract_ik_angles(result)

    def _reset_ik_state_to_origin(self) -> None:
        """Reset IK branch continuity when the physical motors are sent home."""
        if self._movement_strategy != MovementStrategy.IK or self._ik_wire_reference is None:
            return

        ik = self._get_ik_module()
        model = self._get_ik_model()
        result = ik.solve_all_legs((0.0, 0.0, self._get_ik_tactor_z(model)), math.pi / 2.0, model=model)
        if not all(result[leg]["valid"] for leg in ("top", "right", "left")):
            raise ValueError("IK origin reference is invalid.")

        self._ik_rest_result = result
        self._ik_wire_reference = {leg: 0.0 for leg in ("top", "right", "left")}
        self._ik_previous_angles = self._extract_ik_angles(result)

    def _calculate_ik_mechanism_wire_deltas(
        self,
        result: dict[str, Any],
        model: dict[str, Any],
    ) -> dict[str, float]:
        """Return actuator cable deltas from the solved mechanism geometry.

        This is the final "Compute ΔLi" step in the IK flowchart.  The IK solver
        provides each leg's selected ``P3`` branch; the Bowden cable model then
        measures how the cable attachment on the first link moved relative to
        the calibrated rest pose.
        """
        if self._ik_rest_result is None:
            self._ensure_ik_wire_reference()
        if self._ik_rest_result is None:
            raise ValueError("IK rest reference is invalid.")

        deltas: dict[str, float] = {}
        attachment_fraction = wire_fk.CABLE_ATTACHMENT_FRACTION
        for leg in ("top", "right", "left"):
            pb = model["anchors"][leg]
            p3 = result[leg]["selected_P3"]
            p3_ref = self._ik_rest_result[leg]["selected_P3"]
            attachment = (
                pb[0] + attachment_fraction * (p3[0] - pb[0]),
                pb[1] + attachment_fraction * (p3[1] - pb[1]),
                pb[2] + attachment_fraction * (p3[2] - pb[2]),
            )
            reference_attachment = (
                pb[0] + attachment_fraction * (p3_ref[0] - pb[0]),
                pb[1] + attachment_fraction * (p3_ref[1] - pb[1]),
                pb[2] + attachment_fraction * (p3_ref[2] - pb[2]),
            )
            exit_direction = wire_fk._cable_exit_direction(leg, model)
            deltas[leg] = wire_fk.WIRE_RELEASE_SIGN * sum(
                (attachment[axis] - reference_attachment[axis]) * float(exit_direction[axis])
                for axis in range(3)
            )
        return deltas

    def _calculate_ik_tactor_wire_length(
        self,
        leg: str,
        p1: tuple[float, float, float],
        model: dict[str, Any],
    ) -> float:
        """Return the commanded wire length from a motor anchor to the tactor.

        `unified_ik_starter.py` solves the mechanism geometry and validates the
        requested tactor pose. The firmware command, however, drives the three
        motor wires, so the command delta must be measured from each fixed motor
        anchor to the requested tactor point (`P1`), not to an internal IK branch
        point such as `P3`.
        """
        pb = model["anchors"][leg]
        return math.sqrt(
            (p1[0] - pb[0]) ** 2
            + (p1[1] - pb[1]) ** 2
            + (p1[2] - pb[2]) ** 2
        )

    def _extract_ik_angles(self, result: dict[str, Any]) -> dict[str, dict[str, float]]:
        return {
            leg: {
                "phi2": result[leg]["phi2"],
                "phi3": result[leg]["phi3"],
                "phi4": result[leg]["phi4"],
                "phi5": result[leg]["phi5"],
                "phi6": result[leg]["phi6"],
            }
            for leg in ("top", "right", "left")
            if result[leg]["valid"]
        }

    def _get_ik_base_span(self, model: dict[str, Any]) -> float:
        right = model["anchors"]["right"]
        left = model["anchors"]["left"]
        return math.sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2)

    def _get_ik_tactor_z(self, model: dict[str, Any]) -> float:
        # Keep the working height inside the current IK model's reachable sphere.
        # With the default model, z=6.0 validates the clamped controller workspace.
        return max(0.0, min(6.0, model["lengths"]["d3"] - 0.1))

    def _clamp_to_workspace(self, obj_x: float, obj_y: float) -> tuple[float, float]:
        max_radius = max(0.0, min(self._top_width / 2, self._top_height / 2) - self._edge_threshold)
        distance = math.sqrt(obj_x**2 + obj_y**2)
        if distance > max_radius and distance > 0:
            scale = max_radius / distance
            return (obj_x * scale, obj_y * scale)
        return (obj_x, obj_y)

    def _get_planar_motor_anchors(self) -> list[tuple[float, float]]:
        """Return the 2-D motor anchor triangle shared by every strategy.

        There is one physical mechanism with one anchor triangle, defined by
        ``unified_ik_starter.default_model()`` (an isosceles triangle, NOT
        equilateral). Every strategy must measure cable lengths against this
        same triangle so that a commanded path reconstructs identically
        regardless of strategy. The IK anchors are given in IK units, so scale
        them to controller units by ``motor_spacing / base_span``.

        Order matches the motor layout and the IK leg order:
            0 = top, 1 = bottom-right, 2 = bottom-left.
        """
        anchors = self._get_ik_module().default_model()["anchors"]
        base_span = self._get_ik_base_span({"anchors": anchors})
        scale = self._motor_spacing / base_span
        return [
            (anchors[leg][0] * scale, anchors[leg][1] * scale)
            for leg in ("top", "right", "left")
        ]

    def _calculate_movements_to_point(
        self,
        motor_set_id: MotorSetId,
        object_end: tuple[float, float],
    ) -> list[MotorMovement]:
        """Compute cable-length deltas from origin to a target point.

        The object is assumed to start at `(0, 0)`. For each motor in the shared
        mechanism triangle (`_get_planar_motor_anchors`), this calculates:
            delta = distance(motor, object_end) - distance(motor, object_start)

        Args:
            motor_set_id: Physical motor set to control.
            object_end: Target displacement `(x, y)` from center.

        Returns:
            List of `MotorMovement` values with integer-truncated deltas and global
            motor indices for the selected motor set.
        """
        base_motor_idx = motor_set_id.base_index
        object_start = (0.0, 0.0)
        end_x, end_y = object_end

        motor_positions = self._get_planar_motor_anchors()

        movements: list[MotorMovement] = []
        for i, (motor_x, motor_y) in enumerate(motor_positions):
            initial_length = math.sqrt((motor_x - object_start[0])**2 + (motor_y - object_start[1])**2)
            final_length = math.sqrt((motor_x - end_x)**2 + (motor_y - end_y)**2)
            delta_length = final_length - initial_length
            movements.append(MotorMovement(pos=int(delta_length), index=i + base_motor_idx))
        return movements

    def _apply_move_factor(self, motors: list[MotorMovement]) -> list[MotorMovement]:
        """Scale motor deltas by `self._move_factor`.

        Args:
            motors: Unscaled motor movement commands.

        Returns:
            Original list if move factor is 1, otherwise a new list with scaled,
            integer-truncated `pos` values.
        """
        if self._move_factor == 1:
            return motors
        return [MotorMovement(pos=int(motor.pos * self._move_factor), index=motor.index) for motor in motors]
