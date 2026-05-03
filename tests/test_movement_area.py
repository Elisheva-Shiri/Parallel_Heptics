import pytest

import backend
import frontend_pygame
from motor_controller import MotorController, MotorSetId, MovementStrategy


def _to_tuples(movements):
    return [(movement.index, movement.pos) for movement in movements]


def _make_movement_experiment() -> backend.Experiment:
    experiment = object.__new__(backend.Experiment)
    experiment._top_width = backend.TOP_WIDTH
    experiment._top_height = backend.TOP_HEIGHT
    experiment.CENTER_THRESHOLD = backend.CENTER_THRESHOLD
    experiment.EDGE_THRESHOLD = backend.EDGE_THRESHOLD
    experiment.MOVEMENT_AREA_SCALE = backend.MOVEMENT_AREA_SCALE
    full_radius = max(0.0, min(experiment._top_width / 2, experiment._top_height / 2) - experiment.EDGE_THRESHOLD)
    experiment._movement_target_radius = full_radius * experiment.MOVEMENT_AREA_SCALE
    experiment.last_x = backend.TOP_WIDTH / 2
    experiment.last_y = backend.TOP_HEIGHT / 2
    experiment.reached_edge = False
    experiment.in_center = True
    experiment._require_release = False
    experiment._virtual_object = backend.VirtualObject(
        original_x=backend.TOP_WIDTH / 2,
        original_y=backend.TOP_HEIGHT / 2,
    )
    experiment._virtual_object.is_interacting = True
    return experiment


def test_side_target_radius_uses_movement_area_scale_from_consts():
    full_radius = min(backend.TOP_WIDTH / 2, backend.TOP_HEIGHT / 2) - backend.EDGE_THRESHOLD
    scaled_movement_radius = full_radius * backend.MOVEMENT_AREA_SCALE
    scaled_pygame_radius = frontend_pygame.OUTBOUND_CUE_RADIUS * frontend_pygame.MOVEMENT_AREA_SCALE

    assert scaled_movement_radius / full_radius == pytest.approx(backend.MOVEMENT_AREA_SCALE)
    assert scaled_pygame_radius / frontend_pygame.OUTBOUND_CUE_RADIUS == pytest.approx(backend.MOVEMENT_AREA_SCALE)


def test_outbound_and_return_progress_use_scaled_side_target():
    experiment = _make_movement_experiment()
    center_x = experiment._virtual_object.original_x
    center_y = experiment._virtual_object.original_y
    target_radius = experiment._movement_target_radius

    experiment._virtual_object.x = center_x + target_radius + 1
    experiment._virtual_object.y = center_y
    experiment._update_movement_progress()

    assert experiment.reached_edge is True
    assert experiment._virtual_object.progress == pytest.approx(1.0)
    assert experiment._virtual_object.return_progress == pytest.approx(0.0)

    halfway_return_distance = (target_radius + experiment.CENTER_THRESHOLD) / 2
    experiment._virtual_object.x = center_x + halfway_return_distance
    experiment._update_movement_progress()

    assert experiment._virtual_object.progress == pytest.approx(1.0)
    assert experiment._virtual_object.return_progress == pytest.approx(0.5)

    experiment._virtual_object.x = center_x
    experiment._update_movement_progress()

    assert experiment._virtual_object.cycle_counter == 1
    assert experiment._virtual_object.is_interacting is False
    assert experiment._require_release is True


def test_haptic_workspace_clamp_uses_scaled_target_from_consts():
    target_radius = (min(backend.TOP_WIDTH / 2, backend.TOP_HEIGHT / 2) - backend.EDGE_THRESHOLD) * backend.MOVEMENT_AREA_SCALE
    effective_edge_threshold = min(backend.TOP_WIDTH / 2, backend.TOP_HEIGHT / 2) - target_radius
    controller = MotorController(
        movement_strategy=MovementStrategy.FREE_FORM,
        top_width=backend.TOP_WIDTH,
        top_height=backend.TOP_HEIGHT,
        edge_threshold=effective_edge_threshold,
    )

    actual = controller.calculate_motor_movements(
        motor_set_id=MotorSetId.MOTORS_3_5,
        obj_x=400.0,
        obj_y=0.0,
        motors_enabled=True,
    )
    expected = controller._calculate_movements_to_point(
        motor_set_id=MotorSetId.MOTORS_3_5,
        object_end=(target_radius, 0.0),
    )

    assert _to_tuples(actual) == _to_tuples(expected)
