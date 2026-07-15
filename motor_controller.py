import math
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any

from kinematics import unified_ik_starter as ik


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
    """Available movement-to-motor mapping strategies."""

    CARDINAL = "cardinal"
    CARDINAL_DIAGONAL = "cardinal_diagonal"
    FREE_FORM = "free_form"
    IK = "ik"


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
    ):
        """Create a motor controller with geometry and movement parameters.

        Args:
            movement_strategy: Strategy used to transform displacement to movement.
            top_width: Width of the active movement area.
            top_height: Height of the active movement area.
            edge_threshold: Margin near the boundary where free-form motion is clipped.
            motor_spacing: Distance between motors in a single physical motor cluster.
            move_factor: Scalar applied to all output motor deltas.
            diagonal_threshold: Min axis-ratio needed to classify movement as diagonal
                in `MovementStrategy.CARDINAL_DIAGONAL`.
            hand_orientation: Whether to mirror left/right motion on the X axis.
        """
        self._movement_strategy = movement_strategy
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

        match self._movement_strategy:
            case MovementStrategy.FREE_FORM:
                movements = self._calculate_freeform_motor_movements(motor_set_id, obj_x, obj_y)
            case MovementStrategy.IK:
                movements = self._calculate_ik_motor_movements(motor_set_id, obj_x, obj_y)
            case MovementStrategy.CARDINAL | MovementStrategy.CARDINAL_DIAGONAL:
                direction, distance = self._determine_movement_direction(obj_x, obj_y)
                if direction == "none":
                    return []

                if self._movement_strategy == MovementStrategy.CARDINAL:
                    movements = self._calculate_cardinal_motor_movements(motor_set_id, direction, distance)
                else:
                    movements = self._calculate_diagonal_motor_movements(motor_set_id, direction, distance)
            case _:
                raise NotImplementedError(f"Unknown movement strategy: {self._movement_strategy}")
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

    def _determine_movement_direction(self, obj_x: float, obj_y: float) -> tuple[str, float]:
        """Determine movement direction and distance from displacement.

        Diagonal directions are emitted only when strategy is
        `MovementStrategy.CARDINAL_DIAGONAL` and axis ratio passes
        `self._diagonal_threshold`.

        Returns:
            Tuple of (`direction`, `distance`), where direction is one of:
            `none`, `up`, `down`, `left`, `right`, `up-left`, `up-right`,
            `down-left`, or `down-right`.
        """
        if obj_x == 0 and obj_y == 0:
            return ("none", 0)

        abs_x, abs_y = abs(obj_x), abs(obj_y)

        if self._movement_strategy == MovementStrategy.CARDINAL_DIAGONAL and abs_x > 0 and abs_y > 0:
            ratio = min(abs_x, abs_y) / max(abs_x, abs_y)
            if ratio >= self._diagonal_threshold:
                distance = math.sqrt(obj_x**2 + obj_y**2)
                if obj_x < 0:
                    direction = "up-left" if obj_y < 0 else "down-left"
                else:
                    direction = "up-right" if obj_y < 0 else "down-right"
                return (direction, distance)

        if abs_x > abs_y:
            return ("left" if obj_x < 0 else "right", abs_x)
        return ("up" if obj_y < 0 else "down", abs_y)

    def _calculate_cardinal_motor_movements(
        self,
        motor_set_id: MotorSetId,
        direction: str,
        distance: float,
    ) -> list[MotorMovement]:
        """
        Calculate motor movements for cardinal directions only (up, down, left, right).

        Motor layout (top view):
            0 (top)
           / \
          2   1

        Args:
            motor_set_id: Physical motor set to control.
            direction: Movement direction ("up", "down", "left", "right")
            distance: Movement magnitude in the same units as `motor_spacing`.

        Returns:
            List of per-motor deltas.
        """
        direction_vectors = {
            "up": (0.0, distance),
            "down": (0.0, -distance),
            "left": (-distance, 0.0),
            "right": (distance, 0.0),
        }
        if direction not in direction_vectors:
            raise ValueError(f"Invalid direction: {direction}. Must be one of: up, down, left, right")
        return self._calculate_movements_to_point(motor_set_id, direction_vectors[direction])

    def _calculate_diagonal_motor_movements(
        self,
        motor_set_id: MotorSetId,
        direction: str,
        distance: float,
    ) -> list[MotorMovement]:
        """
        Calculate motor movements for all directions including diagonals.

        Motor layout (top view):
            0 (top)
           / \
          2   1

        Args:
            motor_set_id: Physical motor set to control.
            direction: Movement direction (cardinal or diagonal)
            distance: Movement magnitude in the same units as `motor_spacing`.

        Returns:
            List of per-motor deltas.
        """
        direction_vectors = {
            "up": (0.0, distance),
            "down": (0.0, -distance),
            "left": (-distance, 0.0),
            "right": (distance, 0.0),
            "up-left": (-distance / math.sqrt(2), distance / math.sqrt(2)),
            "up-right": (distance / math.sqrt(2), distance / math.sqrt(2)),
            "down-left": (-distance / math.sqrt(2), -distance / math.sqrt(2)),
            "down-right": (distance / math.sqrt(2), -distance / math.sqrt(2)),
        }
        if direction not in direction_vectors:
            allowed = ", ".join(direction_vectors.keys())
            raise ValueError(f"Invalid direction: {direction}. Must be one of: {allowed}")
        return self._calculate_movements_to_point(motor_set_id, direction_vectors[direction])

    def _calculate_freeform_motor_movements(
        self,
        motor_set_id: MotorSetId,
        obj_x: float,
        obj_y: float,
    ) -> list[MotorMovement]:
        """
        Calculate motor movements for free-form movement in any direction.

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
        movements: list[MotorMovement] = []
        for i, leg in enumerate(("top", "right", "left")):
            wire_length = self._calculate_ik_tactor_wire_length(leg, p1, model)
            wire_delta = wire_length - self._ik_wire_reference[leg]
            movements.append(MotorMovement(pos=int(wire_delta * ik_to_sim_scale), index=base_motor_idx + i))
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
        self._ik_wire_reference = self._calculate_ik_wire_reference(reference_p1, model)
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

        self._ik_wire_reference = self._calculate_ik_wire_reference(result["shared"]["P1"], model)
        self._ik_previous_angles = self._extract_ik_angles(result)

    def _calculate_ik_wire_reference(
        self,
        p1: tuple[float, float, float],
        model: dict[str, Any],
    ) -> dict[str, float]:
        return {
            leg: self._calculate_ik_tactor_wire_length(leg, p1, model)
            for leg in ("top", "right", "left")
        }

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
