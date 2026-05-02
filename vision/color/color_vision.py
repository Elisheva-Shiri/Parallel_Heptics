import math
from typing import Literal
import numpy as np

from ..base_vision import BaseVision, HandPosition
from .detector import Detector, FlowTrackedObject, TrackedObject

class ColorVision(BaseVision):
    def __init__(self,
        finger_colors: dict[str, str],
        top_width: int,
        top_height: int,
        side_width: int,
        side_height: int,
        base_interaction_threshold: float,
        tracking_method: str = "frame",
        fps: float = 30.0,
        use_gpu: bool = False,
        min_contour_area: int = 10
    ):
        self._finger_colors = finger_colors
        self._top_width = top_width
        self._top_height = top_height
        self._side_width = side_width
        self._side_height = side_height
        self.base_interaction_threshold = base_interaction_threshold

        # Single-finger mode: one color is used regardless of which finger the
        # protocol marks active. The synthetic "single" key signals this.
        self._single_color: str | None = self._finger_colors.get("single")
        if self._single_color is not None:
            self._thumb_color = ""
            self._active_finger_color = self._single_color
        else:
            self._thumb_color = self._finger_colors.get("thumb", "")
            self._active_finger_color = next(
                (c for f, c in self._finger_colors.items() if f != "thumb"), ""
            )
        self._top_detector = Detector(tracking_method, fps, use_gpu, min_contour_area)
        self._side_detector = Detector(tracking_method, fps, use_gpu, min_contour_area)

    def set_active_finger(self, finger: str):
        if self._single_color is not None:
            return
        self._active_finger_color = self._finger_colors[finger]

    def _get_hand_pos(
        self,
        thumb: TrackedObject | FlowTrackedObject | None,
        active_finger: TrackedObject | FlowTrackedObject | None,
        camera: Literal["top", "side"]
    ) -> HandPosition:
        thumb_x = thumb.position[0] if thumb else 0.0
        thumb_y = thumb.position[1] if thumb else 0.0
        active_finger_x = active_finger.position[0] if active_finger else 0.0
        active_finger_y = active_finger.position[1] if active_finger else 0.0

        match camera:
            case "top":
                width = self._top_width
            case "side":
                width = self._side_width
            case _:
                raise ValueError(f"Unknown camera: {camera}")
            
        return HandPosition(
            thumb_x=width - thumb_x,
            thumb_y=thumb_y,
            active_finger_x=width - active_finger_x,
            active_finger_y=active_finger_y
        )

    def detect_hand(self, frame: np.ndarray) -> HandPosition:
        tracked_objs, _ = self._top_detector.update(frame)  # 2nd arg is visualization frame, not used
        if tracked_objs:
            print(tracked_objs)
        
        # Get positions for all fingers
        thumb = tracked_objs.get(self._thumb_color, None)
        active_finger = tracked_objs.get(self._active_finger_color, None)
        return self._get_hand_pos(thumb, active_finger, "top")
    
    def detect_side_hand(self, frame: np.ndarray) -> HandPosition:
        tracked_objs, _ = self._side_detector.update(frame)
        thumb = tracked_objs.get(self._thumb_color, None)
        active_finger = tracked_objs.get(self._active_finger_color, None)
        return self._get_hand_pos(thumb, active_finger, "side")

    def detect_interaction(
        self, 
        frame: np.ndarray, 
        top_position: HandPosition,
        camera_pos: Literal["top", "bottom", "left", "right"] = "bottom",
        min_depth: float = 0.1,  # Minimum normalized depth (closest to camera)
        max_depth: float = 1.0   # Maximum normalized depth (furthest from camera)
    ) -> bool:
        """
        Process side camera frame to detect interaction gesture with depth-adjusted threshold
        
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
        tracked_objs, _ = self._side_detector.update(frame)
        
        # Get positions for all fingers
        thumb = tracked_objs.get(self._thumb_color, None)
        active_finger = tracked_objs.get(self._active_finger_color, None)
        side_pos = self._get_hand_pos(thumb, active_finger, "side")
        
        if thumb and active_finger:
            # Calculate Euclidean distance between thumb and active finger
            distance = math.sqrt(
                (side_pos.thumb_x - side_pos.active_finger_x)**2 + 
                (side_pos.thumb_y - side_pos.active_finger_y)**2
            )
            
            # Calculate normalized depth based on camera position
            if camera_pos == "top":
                # y=0 is close (top), y=height is far (bottom)
                depth_pixel = top_position.thumb_y
                normalized_depth = depth_pixel / self._top_height
            elif camera_pos == "bottom":
                # y=height is close (bottom), y=0 is far (top)
                depth_pixel = self._top_height - top_position.thumb_y
                normalized_depth = depth_pixel / self._top_height
            elif camera_pos == "left":
                # x=0 is close (left), x=width is far (right)
                depth_pixel = top_position.thumb_x
                normalized_depth = depth_pixel / self._top_width
            else:  # camera_pos == "right"
                # x=width is close (right), x=0 is far (left)
                depth_pixel = self._top_width - top_position.thumb_x
                normalized_depth = depth_pixel / self._top_width
            
            # Clamp depth to the specified range
            normalized_depth = max(min_depth, min(normalized_depth, max_depth))
            
            # Scale the threshold based on depth
            # When close (low depth): larger threshold (fingers appear further apart)
            # When far (high depth): smaller threshold (fingers appear closer together)
            depth_scale = (max_depth - normalized_depth + min_depth) / max_depth
            adjusted_threshold = self.base_interaction_threshold * depth_scale
            
            return distance < adjusted_threshold
        else:
            return False
