import pytest

from vision.base_vision import HandPosition
from vision.mediapipe.mp_vision import MediapipeVision


def _make_vision_stub(alpha: float = 0.25, jitter_threshold: float = 2.0) -> MediapipeVision:
    vision = object.__new__(MediapipeVision)
    vision._position_jitter_lpf_alpha = alpha
    vision._position_jitter_threshold_pixels = jitter_threshold
    vision._top_smoothed_position = None
    vision._side_smoothed_position = None
    vision._interaction_smoothed_position = None
    return vision


def test_first_position_is_not_delayed_by_smoothing():
    vision = _make_vision_stub()
    current = HandPosition(thumb_x=10.0, thumb_y=20.0, active_finger_x=30.0, active_finger_y=40.0)

    smoothed = vision._smooth_position("_top_smoothed_position", current)

    assert smoothed == current


def test_tiny_jitter_uses_low_pass_filter():
    vision = _make_vision_stub(alpha=0.25, jitter_threshold=10.0)
    previous = HandPosition(thumb_x=10.0, thumb_y=20.0, active_finger_x=30.0, active_finger_y=40.0)
    current = HandPosition(thumb_x=18.0, thumb_y=12.0, active_finger_x=22.0, active_finger_y=48.0)
    vision._smooth_position("_top_smoothed_position", previous)

    smoothed = vision._smooth_position("_top_smoothed_position", current)

    assert smoothed.thumb_x == pytest.approx(12.0)
    assert smoothed.thumb_y == pytest.approx(18.0)
    assert smoothed.active_finger_x == pytest.approx(28.0)
    assert smoothed.active_finger_y == pytest.approx(42.0)


def test_real_movement_is_not_delayed_by_smoothing():
    vision = _make_vision_stub(alpha=0.25, jitter_threshold=2.0)
    previous = HandPosition(thumb_x=10.0, thumb_y=20.0, active_finger_x=30.0, active_finger_y=40.0)
    current = HandPosition(thumb_x=18.0, thumb_y=20.0, active_finger_x=30.0, active_finger_y=40.0)
    vision._smooth_position("_top_smoothed_position", previous)

    smoothed = vision._smooth_position("_top_smoothed_position", current)

    assert smoothed == current


def test_missing_position_resets_smoothing_without_ghosting():
    vision = _make_vision_stub()
    vision._smooth_position(
        "_top_smoothed_position",
        HandPosition(thumb_x=10.0, thumb_y=20.0, active_finger_x=30.0, active_finger_y=40.0),
    )
    missing = HandPosition(thumb_x=0.0, thumb_y=0.0, active_finger_x=0.0, active_finger_y=0.0)

    smoothed = vision._smooth_position("_top_smoothed_position", missing)

    assert smoothed == missing
    assert vision._top_smoothed_position is None
