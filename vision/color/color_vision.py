import math
import numpy as np

from ..base_vision import BaseVision, HandPosition
from .detector import Detector

class ColorVision(BaseVision):
    def __init__(self,
        finger_colors: dict[str, str],
        width: int,
        height: int,
        base_pinch_threshold: float,
        tracking_method: str = "frame",
        fps: float = 30.0,
        use_gpu: bool = False,
        min_contour_area: int = 10
    ):
        self._finger_colors = finger_colors
        self._width = width
        self._height = height
        self.base_pinch_threshold = base_pinch_threshold

        self._thumb_color = self._finger_colors["thumb"]
        # Initialize active finger to index
        self._active_finger_color = self._finger_colors["index"]
        self.detector = Detector(tracking_method, fps, use_gpu, min_contour_area)
    
    def set_active_finger(self, finger: str):
        self._active_finger_color = self._finger_colors[finger]

    def detect_hand(self, frame: np.ndarray) -> HandPosition:
        tracked_objs, _ = self.detector.update(frame)  # 2nd arg is visualization frame, not used
        if tracked_objs:
            print(tracked_objs)
        
        # Get positions for all fingers
        thumb = tracked_objs.get(self._thumb_color, None)
        active_finger = tracked_objs.get(self._active_finger_color, None)
        thumb_x = thumb.position[0] if thumb else 0.0
        thumb_y = thumb.position[1] if thumb else 0.0
        active_finger_x = active_finger.position[0] if active_finger else 0.0
        active_finger_y = active_finger.position[1] if active_finger else 0.0

        return HandPosition(
            thumb_x=self._width - thumb_x,
            thumb_y=thumb_y,
            active_finger_x=self._width - active_finger_x,
            active_finger_y=active_finger_y
        )
    
    def detect_pinch(self, frame: np.ndarray) -> bool:
        """Process side camera frame to detect pinch gesture"""
        tracked_objs, _ = self.detector.update(frame)  # 2nd arg is visualization frame, not used
        
        # Get positions for all fingers
        thumb = tracked_objs.get(self._thumb_color, None)
        active_finger = tracked_objs.get(self._active_finger_color, None)
        
        if thumb and active_finger:
            distance = math.sqrt(
                (thumb.position[0] - active_finger.position[0])**2 + 
                (thumb.position[1] - active_finger.position[1])**2
            )
            return distance < self.base_pinch_threshold
        else:
            return False