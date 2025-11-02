from ..base_vision import BaseVision, HandPosition
from ultralytics import YOLO  # type: ignore
from pathlib import Path

import numpy as np

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16

class YoloVision(BaseVision):
    def __init__(
        self,
        width: int,
        height: int,
        base_pinch_threshold: float,
        model_path: Path = Path(__file__).parent / "model" / "best.pt"
    ):
        self._width = width
        self._height = height
        self.model = YOLO(model_path)
        self._active_finger_tip = INDEX_TIP
        self.base_pinch_threshold = base_pinch_threshold

    def set_active_finger(self, finger: str):
        self._active_finger_tip = {
            "index": INDEX_TIP,
            "middle": MIDDLE_TIP,
            "ring": RING_TIP
        }[finger]

    def detect_hand(self, frame: np.ndarray) -> HandPosition:
        results = self.model(frame)
        result = results[0]
        if not result.keypoints or len(result.keypoints.xy) == 0:
            return HandPosition(
                thumb_x=0.0,
                thumb_y=0.0,
                active_finger_x=0.0,
                active_finger_y=0.0
            )
        
        keypoints = result.keypoints.xy[0]
        if keypoints is None or len(keypoints) <= self._active_finger_tip:
            return HandPosition(
                thumb_x=0.0,
                thumb_y=0.0,
                active_finger_x=0.0,
                active_finger_y=0.0
            )

        return HandPosition(
            thumb_x=self._width - keypoints[THUMB_TIP][0],
            thumb_y=keypoints[THUMB_TIP][1],
            active_finger_x=self._width - keypoints[self._active_finger_tip][0],
            active_finger_y=keypoints[self._active_finger_tip][1]
        )

    
    def detect_pinch(self, frame: np.ndarray) -> bool:
        results = self.model(frame)
        result = results[0]
        keypoints = result.keypoints.xy[0]
        if keypoints is None or len(keypoints) <= self._active_finger_tip:
            return False
        
        thumb_x, thumb_y = keypoints[THUMB_TIP]
        active_finger_x, active_finger_y = keypoints[self._active_finger_tip]
        
        distance = np.sqrt((thumb_x - active_finger_x)**2 + (thumb_y - active_finger_y)**2)
        return distance < self.base_pinch_threshold
