import math
from typing import Literal
import cv2
import mediapipe as mp
import numpy as np
from ..base_vision import BaseVision, HandPosition


POSITION_JITTER_THRESHOLD_PIXELS = 2.0
POSITION_JITTER_LPF_ALPHA = 0.35


class MediapipeVision(BaseVision):
    def __init__(
        self,
        top_width: int,
        top_height: int,
        side_width: int,
        side_height: int,
        static_image_mode: bool,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        base_interaction_threshold: float
    ):
        self._top_width = top_width
        self._top_height = top_height
        self._side_width = side_width
        self._side_height = side_height
        self._static_image_mode = static_image_mode
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._base_interaction_threshold = base_interaction_threshold
        self._position_jitter_threshold_pixels = POSITION_JITTER_THRESHOLD_PIXELS
        self._position_jitter_lpf_alpha = POSITION_JITTER_LPF_ALPHA
        self._top_smoothed_position: HandPosition | None = None
        self._side_smoothed_position: HandPosition | None = None
        self._interaction_smoothed_position: HandPosition | None = None
        # Initialize active finger to index
        self._active_finger_landmark = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP

        self._mp_top_hands = mp.solutions.hands.Hands(
            static_image_mode=self._static_image_mode,
            max_num_hands=self._max_num_hands,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence
        )

        self._mp_side_hands = mp.solutions.hands.Hands(
            static_image_mode=self._static_image_mode,
            max_num_hands=self._max_num_hands,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence
        )

    def set_active_finger(self, finger: str):
        self._active_finger_landmark = {
            "thumb": mp.solutions.hands.HandLandmark.THUMB_TIP,
            "index": mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
            "middle": mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
            "ring": mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
            "pinky": mp.solutions.hands.HandLandmark.PINKY_TIP,
        }[finger]
        self._reset_smoothing()

    def _reset_smoothing(self) -> None:
        self._top_smoothed_position = None
        self._side_smoothed_position = None
        self._interaction_smoothed_position = None

    @staticmethod
    def _is_missing_position(position: HandPosition) -> bool:
        return (
            position.thumb_x == 0.0
            and position.thumb_y == 0.0
            and position.active_finger_x == 0.0
            and position.active_finger_y == 0.0
        )

    @staticmethod
    def _max_position_delta(first: HandPosition, second: HandPosition) -> float:
        return max(
            abs(first.thumb_x - second.thumb_x),
            abs(first.thumb_y - second.thumb_y),
            abs(first.active_finger_x - second.active_finger_x),
            abs(first.active_finger_y - second.active_finger_y),
        )

    def _smooth_position(self, state_attr: str, current: HandPosition) -> HandPosition:
        if self._is_missing_position(current):
            setattr(self, state_attr, None)
            return current

        previous = getattr(self, state_attr)
        if previous is None:
            setattr(self, state_attr, current)
            return current

        if self._max_position_delta(previous, current) > self._position_jitter_threshold_pixels:
            setattr(self, state_attr, current)
            return current

        alpha = self._position_jitter_lpf_alpha
        smoothed = HandPosition(
            thumb_x=alpha * current.thumb_x + (1.0 - alpha) * previous.thumb_x,
            thumb_y=alpha * current.thumb_y + (1.0 - alpha) * previous.thumb_y,
            active_finger_x=alpha * current.active_finger_x + (1.0 - alpha) * previous.active_finger_x,
            active_finger_y=alpha * current.active_finger_y + (1.0 - alpha) * previous.active_finger_y,
        )
        setattr(self, state_attr, smoothed)
        return smoothed

    def detect_hand(self, frame: np.ndarray) -> HandPosition:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_top_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get positions for all fingers
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            
            # Get active finger for object control
            active_tip = hand_landmarks.landmark[self._active_finger_landmark]
            
            position = HandPosition(
                thumb_x=self._top_width - thumb_tip.x * self._top_width,
                thumb_y=thumb_tip.y * self._top_height,
                active_finger_x=self._top_width - active_tip.x * self._top_width,
                active_finger_y=active_tip.y * self._top_height
            )
            return self._smooth_position("_top_smoothed_position", position)
            
        position = HandPosition(
            thumb_x=0.0,
            thumb_y=0.0,
            active_finger_x=0.0,
            active_finger_y=0.0
        )
        return self._smooth_position("_top_smoothed_position", position)
    
    def detect_side_hand(self, frame: np.ndarray) -> HandPosition:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_side_hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            active_tip = hand_landmarks.landmark[self._active_finger_landmark]
            position = HandPosition(
                thumb_x=self._side_width - thumb_tip.x * self._side_width,
                thumb_y=thumb_tip.y * self._side_height,
                active_finger_x=self._side_width - active_tip.x * self._side_width,
                active_finger_y=active_tip.y * self._side_height
            )
            return self._smooth_position("_side_smoothed_position", position)

        position = HandPosition(thumb_x=0.0, thumb_y=0.0, active_finger_x=0.0, active_finger_y=0.0)
        return self._smooth_position("_side_smoothed_position", position)

    def detect_interaction(
        self, 
        frame: np.ndarray,
        top_position: HandPosition,
        camera_pos: Literal["top", "bottom", "left", "right"] = "bottom",
        min_depth: float = 0.1,
        max_depth: float = 1.0
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_side_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            finger_tip = hand_landmarks.landmark[self._active_finger_landmark]
            side_position = HandPosition(
                thumb_x=self._side_width - thumb_tip.x * self._side_width,
                thumb_y=thumb_tip.y * self._side_height,
                active_finger_x=self._side_width - finger_tip.x * self._side_width,
                active_finger_y=finger_tip.y * self._side_height
            )
            side_position = self._smooth_position("_interaction_smoothed_position", side_position)
            
            # Calculate Euclidean distance between thumb and active finger
            distance = math.sqrt(
                (side_position.thumb_x - side_position.active_finger_x)**2 + 
                (side_position.thumb_y - side_position.active_finger_y)**2
            )
            
            # Calculate normalized depth based on camera position
            match camera_pos:
                case "top":
                    # y=0 is close (top), y=height is far (bottom)
                    depth_pixel = top_position.thumb_y
                    normalized_depth = depth_pixel / self._top_height
                case "bottom":
                    # y=height is close (bottom), y=0 is far (top)
                    depth_pixel = self._top_height - top_position.thumb_y
                    normalized_depth = depth_pixel / self._top_height
                case "left":
                    # x=0 is close (left), x=width is far (right)
                    depth_pixel = top_position.thumb_x
                    normalized_depth = depth_pixel / self._top_width
                case "right":
                    # x=width is close (right), x=0 is far (left)
                    depth_pixel = self._top_width - top_position.thumb_x
                    normalized_depth = depth_pixel / self._top_width
                case _:
                    raise ValueError(f"Unknown camera position: {camera_pos}")
                
            # Clamp depth to the specified range
            normalized_depth = max(min_depth, min(normalized_depth, max_depth))
            
            # Scale the threshold based on depth
            # When close (low depth): larger threshold (fingers appear further apart)
            # When far (high depth): smaller threshold (fingers appear closer together)
            depth_scale = (max_depth - normalized_depth + min_depth) / max_depth
            adjusted_threshold = self._base_interaction_threshold * depth_scale

            # print all side_position and distance
            print("=====================================")
            print(f"Thumb: {side_position.thumb_x}, {side_position.thumb_y}")
            print(f"Finger: {side_position.active_finger_x}, {side_position.active_finger_y}")
            print(f"Distance: {distance}")
            print(f"Depth: {normalized_depth}")
            print(f"Threshold: {adjusted_threshold}")
            
            return distance < adjusted_threshold
        else:
            self._interaction_smoothed_position = None
            return False

        
    def cleanup(self):
        self._mp_side_hands.close()
        self._mp_top_hands.close()
