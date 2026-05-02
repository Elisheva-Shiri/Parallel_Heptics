import numpy as np

from vision.color.color_vision import ColorVision
from vision.color.detector import Detector


def test_color_vision_detect_hand_does_not_print_tracked_objects(capsys):
    vision = ColorVision(
        finger_colors={"single": "blue"},
        top_width=100,
        top_height=100,
        side_width=100,
        side_height=100,
        base_interaction_threshold=50,
        tracking_method="frame",
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:30, 20:30] = (255, 0, 0)

    position = vision.detect_hand(frame)

    assert position.active_finger_x > 0
    assert capsys.readouterr().out == ""


def test_frame_tracker_uses_configured_min_contour_area():
    detector = Detector(tracking_method="frame", min_contour_area=10)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[10:15, 10:15] = (255, 0, 0)

    tracked_objects, _ = detector.update(frame)

    assert "blue" in tracked_objects
