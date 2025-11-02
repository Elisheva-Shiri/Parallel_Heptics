import cv2
import mediapipe as mp
import math
from typing import Any
import numpy as np

class SideCameraTest:
    def __init__(self):
        self._is_pinching = False
        self._active_finger = "index" # Can be changed to "middle" or "ring"
        self._active_landmark = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
        self._running = True
        self.SIDE_CAMERA = 0
        self.BASE_PINCH_THRESHOLD = 1.0  # pixels
        self._width = 640  # Default camera width
        self._height = 480  # Default camera height
        #!new
        self._threshold_multipliers = {
            "index": 1.0,
            "middle": 1.5,
            "ring": 1.7
        }
        self._finger_distances = {
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0
        }

    def _configure_camera(self, cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

    def _calculate_pinch_threshold(self, depth: float) -> float:
        """Adjust threshold based on depth (distance from side camera)"""
        #!new
        return self.BASE_PINCH_THRESHOLD * (depth) * self._threshold_multipliers[self._active_finger]

    def _detect_pinch(self, frame: np.ndarray, hands: Any):
        """Process side camera frame to detect pinch gesture"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            
            #!new
            # Calculate distances for all fingers
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            
            self._finger_distances["index"] = math.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            )
            self._finger_distances["middle"] = math.sqrt(
                (thumb_tip.x - middle_tip.x)**2 + 
                (thumb_tip.y - middle_tip.y)**2
            )
            self._finger_distances["ring"] = math.sqrt(
                (thumb_tip.x - ring_tip.x)**2 + 
                (thumb_tip.y - ring_tip.y)**2
            )
            
            # Find the finger closest to thumb
            closest_finger = min(self._finger_distances.items(), key=lambda x: x[1])[0]
            self._active_finger = closest_finger
            
            # Update active landmark based on closest finger
            if closest_finger == "index":
                self._active_landmark = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
            elif closest_finger == "middle":
                self._active_landmark = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
            else:  # ring
                self._active_landmark = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
            
            finger_tip = hand_landmarks.landmark[self._active_landmark]
            depth = abs(finger_tip.z)
            threshold = self._calculate_pinch_threshold(depth)
            
            #!new
            current_distance = self._finger_distances[self._active_finger]
            self._is_pinching = current_distance < threshold
            
            # Draw landmarks and connections for visualization
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            #!new
            # Draw pinch status with flipped text
            pinch_status = f"Pinching with {self._active_finger}" if self._is_pinching else "Not Pinching"
            flipped_frame = cv2.flip(frame.copy(), 1)  # Create a copy for text
            cv2.putText(flipped_frame, pinch_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0) if self._is_pinching else (0, 0, 255), 2)
            
            #!new
            # Draw distances for all fingers
            y_pos = 60
            for finger, distance in self._finger_distances.items():
                adjusted = distance/self._threshold_multipliers[finger]
                raw_text = f"{finger.capitalize()} Raw: {distance:.2f}"
                adj_text = f"{finger.capitalize()} Adjusted: {adjusted:.2f}"
                cv2.putText(flipped_frame, raw_text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(flipped_frame, adj_text, (10, y_pos + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_pos += 70
            
            text_region = flipped_frame[0:y_pos+30, 0:500]
            frame[0:y_pos+30, self._width-500:self._width] = cv2.flip(text_region, 1)
        else:
            self._is_pinching = False

    def start_side_camera(self):
        """Process side camera feed to detect pinch gestures"""
        cap = cv2.VideoCapture(self.SIDE_CAMERA)
        if not cap.isOpened():
            print("Side camera not available")
            return
        
        self._configure_camera(cap)

        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    continue

                self._detect_pinch(frame, hands)
                    
                cv2.imshow("Side Camera", cv2.flip(frame, 1))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

def main():
    camera = SideCameraTest()
    camera.start_side_camera()

if __name__ == "__main__":
    main()
