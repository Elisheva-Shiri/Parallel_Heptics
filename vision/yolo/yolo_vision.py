from typing import Literal
from ..base_vision import BaseVision, HandPosition
from ultralytics import YOLO  # type: ignore
from pathlib import Path

import numpy as np

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

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
        self.top_model = YOLO(model_path)
        self.side_model = YOLO(model_path)
        self._active_finger_tip = INDEX_TIP
        self.base_pinch_threshold = base_pinch_threshold

    def set_active_finger(self, finger: str):
        self._active_finger_tip = {
            "thumb": THUMB_TIP,
            "index": INDEX_TIP,
            "middle": MIDDLE_TIP,
            "ring": RING_TIP,
            "pinky": PINKY_TIP,
        }[finger]

    def detect_hand(self, frame: np.ndarray) -> HandPosition:
        results = self.top_model(frame)
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

    def detect_side_hand(self, frame: np.ndarray) -> HandPosition:
        results = self.side_model(frame)
        result = results[0]
        if not result.keypoints or len(result.keypoints.xy) == 0:
            return HandPosition(thumb_x=0.0, thumb_y=0.0, active_finger_x=0.0, active_finger_y=0.0)

        keypoints = result.keypoints.xy[0]
        if keypoints is None or len(keypoints) <= self._active_finger_tip:
            return HandPosition(thumb_x=0.0, thumb_y=0.0, active_finger_x=0.0, active_finger_y=0.0)

        return HandPosition(
            thumb_x=self._width - keypoints[THUMB_TIP][0],
            thumb_y=keypoints[THUMB_TIP][1],
            active_finger_x=self._width - keypoints[self._active_finger_tip][0],
            active_finger_y=keypoints[self._active_finger_tip][1]
        )

    def detect_pinch(
        self, 
        frame: np.ndarray,
        top_position: HandPosition,
        camera_pos: Literal["top", "bottom", "left", "right"] = "bottom",
        min_depth: float = 0.1,
        max_depth: float = 1.0
    ) -> bool:
        """
        Process side camera frame to detect pinch gesture with depth-adjusted threshold
        
        Args:
            frame: Input frame from side camera
            top_position: Hand position from top camera (used for depth estimation)
            camera_pos: Position of the side camera relative to the play area
                    - "top": camera at top, y=0 is close, y=height is far
                    - "bottom": camera at bottom, y=height is close, y=0 is far
                    - "left": camera at left, x=0 is close, x=width is far
                    - "right": camera at right, x=width is close, x=0 is far
            min_depth: Minimum normalized depth value (hand closest to camera)
            max_depth: Maximum normalized depth value (hand furthest from camera)
        """
        results = self.side_model(frame)
        result = results[0]
        keypoints = result.keypoints.xy[0]
        
        if keypoints is None or len(keypoints) <= self._active_finger_tip:
            return False
        
        thumb_x, thumb_y = keypoints[THUMB_TIP]
        active_finger_x, active_finger_y = keypoints[self._active_finger_tip]
        
        # Calculate Euclidean distance between thumb and active finger
        distance = np.sqrt((thumb_x - active_finger_x)**2 + (thumb_y - active_finger_y)**2)
        
        # Calculate normalized depth based on camera position
        if camera_pos == "top":
            # y=0 is close (top), y=height is far (bottom)
            depth_pixel = top_position.thumb_y
            normalized_depth = depth_pixel / self._height
        elif camera_pos == "bottom":
            # y=height is close (bottom), y=0 is far (top)
            depth_pixel = self._height - top_position.thumb_y
            normalized_depth = depth_pixel / self._height
        elif camera_pos == "left":
            # x=0 is close (left), x=width is far (right)
            depth_pixel = top_position.thumb_x
            normalized_depth = depth_pixel / self._width
        else:  # camera_pos == "right"
            # x=width is close (right), x=0 is far (left)
            depth_pixel = self._width - top_position.thumb_x
            normalized_depth = depth_pixel / self._width
        
        # Clamp depth to the specified range
        normalized_depth = max(min_depth, min(normalized_depth, max_depth))
        
        # Scale the threshold based on depth
        # When close (low depth): larger threshold (fingers appear further apart)
        # When far (high depth): smaller threshold (fingers appear closer together)
        depth_scale = (max_depth - normalized_depth + min_depth) / max_depth
        adjusted_threshold = self.base_pinch_threshold * depth_scale
        
        return distance < adjusted_threshold
