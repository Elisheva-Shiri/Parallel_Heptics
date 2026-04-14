import math
import unittest

from motor_controller import FingerId, HandOrientation, MotorController, MovementStrategy


def _to_tuples(movements):
    return [(movement.index, movement.pos) for movement in movements]


class TestMotorController(unittest.TestCase):
    def test_cardinal_strategy_uses_major_axis(self):
        controller = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
        )
        actual = controller.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=-40.0,
            obj_y=10.0,
            motors_enabled=True,
        )
        expected = controller._calculate_cardinal_motor_movements(
            finger_id=FingerId.INDEX,
            direction="left",
            distance=40.0,
        )
        self.assertEqual(_to_tuples(actual), _to_tuples(expected))

    def test_cardinal_diagonal_strategy_uses_diagonal_when_threshold_met(self):
        controller = MotorController(
            movement_strategy=MovementStrategy.CARDINAL_DIAGONAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
        )
        obj_x = 100.0
        obj_y = 90.0
        distance = math.sqrt(obj_x**2 + obj_y**2)

        actual = controller.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=obj_x,
            obj_y=obj_y,
            motors_enabled=True,
        )
        expected = controller._calculate_diagonal_motor_movements(
            finger_id=FingerId.INDEX,
            direction="down-right",
            distance=distance,
        )
        self.assertEqual(_to_tuples(actual), _to_tuples(expected))

    def test_zero_displacement_returns_no_motors(self):
        controller = MotorController(
            movement_strategy=MovementStrategy.CARDINAL_DIAGONAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
        )
        actual = controller.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=0.0,
            obj_y=0.0,
            motors_enabled=True,
        )
        self.assertEqual(actual, [])

    def test_free_form_clamps_using_screen_radius(self):
        controller = MotorController(
            movement_strategy=MovementStrategy.FREE_FORM,
            top_width=100.0,
            top_height=100.0,
            edge_threshold=10.0,
        )
        actual = controller.calculate_motor_movements(
            finger_id=FingerId.OTHER,
            obj_x=400.0,
            obj_y=0.0,
            motors_enabled=True,
        )
        expected = controller._calculate_freeform_motor_movements(
            finger_id=FingerId.OTHER,
            obj_x=40.0,
            obj_y=0.0,
        )
        self.assertEqual(_to_tuples(actual), _to_tuples(expected))

    def test_non_comparison_reset_returns_zero_positions_once(self):
        controller = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
        )
        actual = controller.calculate_motor_movements(
            finger_id=FingerId.OTHER,
            obj_x=12.0,
            obj_y=8.0,
            motors_enabled=False,
            reset_to_origin=True,
        )
        self.assertEqual(_to_tuples(actual), [(3, 0), (4, 0), (5, 0)])

    def test_non_comparison_without_reset_returns_empty(self):
        controller = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
        )
        actual = controller.calculate_motor_movements(
            finger_id=FingerId.OTHER,
            obj_x=12.0,
            obj_y=8.0,
            motors_enabled=False,
            reset_to_origin=False,
        )
        self.assertEqual(actual, [])

    def test_move_factor_is_applied(self):
        controller = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
            move_factor=2.0,
        )
        actual = controller.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=20.0,
            obj_y=0.0,
            motors_enabled=True,
        )
        base_controller = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
            move_factor=1.0,
        )
        base = base_controller.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=20.0,
            obj_y=0.0,
            motors_enabled=True,
        )
        expected = [(movement.index, movement.pos * 2) for movement in base]
        self.assertEqual(_to_tuples(actual), expected)

    def test_mirrored_orientation_flips_x_axis(self):
        normal = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
            hand_orientation=HandOrientation.NOT_MIRRORED,
        )
        mirrored = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
            hand_orientation=HandOrientation.MIRRORED,
        )

        normal_as_left = normal.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=-25.0,
            obj_y=0.0,
            motors_enabled=True,
        )
        mirrored_as_right = mirrored.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=25.0,
            obj_y=0.0,
            motors_enabled=True,
        )
        self.assertEqual(_to_tuples(mirrored_as_right), _to_tuples(normal_as_left))

    def test_mirrored_orientation_keeps_y_axis(self):
        normal = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
            hand_orientation=HandOrientation.NOT_MIRRORED,
        )
        mirrored = MotorController(
            movement_strategy=MovementStrategy.CARDINAL,
            top_width=1000.0,
            top_height=1000.0,
            edge_threshold=30.0,
            hand_orientation=HandOrientation.MIRRORED,
        )

        normal_up = normal.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=0.0,
            obj_y=-25.0,
            motors_enabled=True,
        )
        mirrored_up = mirrored.calculate_motor_movements(
            finger_id=FingerId.INDEX,
            obj_x=0.0,
            obj_y=-25.0,
            motors_enabled=True,
        )
        self.assertEqual(_to_tuples(mirrored_up), _to_tuples(normal_up))


if __name__ == "__main__":
    unittest.main()
