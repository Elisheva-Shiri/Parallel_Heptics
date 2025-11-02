import math
import cv2
import mediapipe as mp
import numpy as np
from ..base_vision import BaseVision, HandPosition

class MediapipeVision(BaseVision):
    def __init__(
        self,
        width: int,
        height: int,
        static_image_mode: bool,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        base_pinch_threshold: float
    ):
        self._width = width
        self._height = height
        self._static_image_mode = static_image_mode
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._base_pinch_threshold = base_pinch_threshold
        # Initialize active finger to index
        self._active_finger_landmark = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP

        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=self._static_image_mode,
            max_num_hands=self._max_num_hands,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence
        )

    def set_active_finger(self, finger: str):
        self._active_finger_landmark = {
            "index": mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
            "middle": mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
            "ring": mp.solutions.hands.HandLandmark.RING_FINGER_TIP
        }[finger]

    def detect_hand(self, frame: np.ndarray) -> HandPosition:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get positions for all fingers
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            
            # Get active finger for object control
            active_tip = hand_landmarks.landmark[self._active_finger_landmark]
            
            return HandPosition(
                thumb_x=self._width - thumb_tip.x * self._width,
                thumb_y=thumb_tip.y * self._height,
                active_finger_x=self._width - active_tip.x * self._width,
                active_finger_y=active_tip.y * self._height
            )
            
        return HandPosition(
            thumb_x=0.0,
            thumb_y=0.0,
            active_finger_x=0.0,
            active_finger_y=0.0
        )
    
    def _calculate_pinch_threshold(self, depth: float) -> float:
        """Adjust threshold based on depth (distance from side camera)"""
        return self._base_pinch_threshold * (depth)
    
    def detect_pinch(self, frame: np.ndarray) -> bool:
        """Process side camera frame to detect pinch gesture"""
        # ! detect pinch logic is not good
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            finger_tip = hand_landmarks.landmark[self._active_finger_landmark]
            
            distance = math.sqrt(
                (thumb_tip.x - finger_tip.x)**2 + 
                (thumb_tip.y - finger_tip.y)**2
            )
            
            depth = abs(finger_tip.z)
            threshold = self._calculate_pinch_threshold(depth)
            return distance < threshold
        else:
            return False
        
    def cleanup(self):
        self._mp_hands.close()
