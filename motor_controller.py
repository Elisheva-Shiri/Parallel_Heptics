import math
from dataclasses import dataclass
from enum import Enum, StrEnum


class FingerId(Enum):
    """Finger groups controlled by this module.

    Each value * 3 gives the base motor pin index for that finger's
    3-motor cluster on the 16-channel servo driver.
    """

    THUMB = 0   # pins 0, 1, 2
    INDEX = 1   # pins 3, 4, 5
    MIDDLE = 2  # pins 6, 7, 8
    RING = 3    # pins 9, 10, 11
    PINKY = 4   # pins 12, 13, 14


class MovementStrategy(StrEnum):
    """Available movement-to-motor mapping strategies."""

    CARDINAL = "cardinal"
    CARDINAL_DIAGONAL = "cardinal_diagonal"
    FREE_FORM = "free_form"


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
    """Compute and format motor movements for a triangular 3-motor finger cluster.

    The controller groups motors by finger, where each finger maps to 3 motors in
    a triangular layout (clockwise order: 0 top, 1 bottom-right, 2 bottom-left).
    Movement deltas are computed from object displacement and
    converted to string messages expected by the motor firmware.
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
            motor_spacing: Distance between motors in a single finger cluster.
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
        finger_id: FingerId,
        obj_x: float,
        obj_y: float,
        motors_enabled: bool,
        reset_to_origin: bool = False,
    ) -> list[MotorMovement]:
        """Compute motor deltas for a finger from object displacement.

        Args:
            finger_id: Finger group to control.
            obj_x: X displacement from center.
            obj_y: Y displacement from center.
            motors_enabled: If `False`, movement is suppressed unless reset is requested.
            reset_to_origin: If `True` and not in comparison state, returns zero
                commands for the finger's motors.

        Returns:
            A list of per-motor deltas. Returns an empty list if no movement is needed.
        """
        if not motors_enabled:
            if not reset_to_origin:
                return []
            return self._zero_motor_positions(finger_id)

        obj_x, obj_y = self._apply_hand_orientation(obj_x, obj_y)

        match self._movement_strategy:
            case MovementStrategy.FREE_FORM:
                movements = self._calculate_freeform_motor_movements(finger_id, obj_x, obj_y)
            case MovementStrategy.CARDINAL | MovementStrategy.CARDINAL_DIAGONAL:
                direction, distance = self._determine_movement_direction(obj_x, obj_y)
                if direction == "none":
                    return []

                if self._movement_strategy == MovementStrategy.CARDINAL:
                    movements = self._calculate_cardinal_motor_movements(finger_id, direction, distance)
                else:
                    movements = self._calculate_diagonal_motor_movements(finger_id, direction, distance)
            case _:
                raise NotImplementedError(f"Unknown movement strategy: {self._movement_strategy}")
        return self._apply_move_factor(movements)

    def _get_base_index(self, finger_id: FingerId) -> int:
        """Get the first global motor index for a finger."""
        return finger_id.value * 3

    def _zero_motor_positions(self, finger_id: FingerId) -> list[MotorMovement]:
        """Return zero-position commands for all 3 motors of a finger."""
        base_motor_idx = self._get_base_index(finger_id)
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
        finger_id: FingerId,
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
            finger_id: Finger group to control.
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
        return self._calculate_movements_to_point(finger_id, direction_vectors[direction])

    def _calculate_diagonal_motor_movements(
        self,
        finger_id: FingerId,
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
            finger_id: Finger group to control.
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
        return self._calculate_movements_to_point(finger_id, direction_vectors[direction])

    def _calculate_freeform_motor_movements(
        self,
        finger_id: FingerId,
        obj_x: float,
        obj_y: float,
    ) -> list[MotorMovement]:
        """
        Calculate motor movements for free-form movement in any direction.

        Args:
            finger_id: Finger group to control.
            obj_x: X displacement from center.
            obj_y: Y displacement from center.

        Returns:
            List of per-motor deltas, clipped to the configured workspace radius.
        """
        max_radius = max(0.0, min(self._top_width / 2, self._top_height / 2) - self._edge_threshold)
        distance = math.sqrt(obj_x**2 + obj_y**2)
        if distance > max_radius and distance > 0:
            scale = max_radius / distance
            obj_x *= scale
            obj_y *= scale
        return self._calculate_movements_to_point(finger_id, (obj_x, obj_y))

    def _calculate_movements_to_point(
        self,
        finger_id: FingerId,
        object_end: tuple[float, float],
    ) -> list[MotorMovement]:
        """Compute cable-length deltas from origin to a target point.

        The object is assumed to start at `(0, 0)`. For each motor in the finger's
        equilateral-triangle layout, this calculates:
            delta = distance(motor, object_end) - distance(motor, object_start)

        Args:
            finger_id: Finger group to control.
            object_end: Target displacement `(x, y)` from center.

        Returns:
            List of `MotorMovement` values with integer-truncated deltas and global
            motor indices for the selected finger.
        """
        base_motor_idx = self._get_base_index(finger_id)
        object_start = (0.0, 0.0)
        end_x, end_y = object_end

        h = self._motor_spacing * math.sqrt(3) / 3
        motor_positions = [
            (0.0, 2 * h / 3),  # Motor 0: top
            (self._motor_spacing / 2, -h / 3),  # Motor 1: bottom-right
            (-self._motor_spacing / 2, -h / 3),  # Motor 2: bottom-left
        ]

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
