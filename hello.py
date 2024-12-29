import json
import math
from time import sleep
import cv2
import threading
from dataclasses import dataclass
from typing import Any, Optional, Literal
from threading import Lock
import numpy as np
import mediapipe as mp
import pygame
import socket
import serial
import csv
import time

ARDUINO_DEBUG = True
SIDE_CAMERA_DEBUG = True
@dataclass 
class HandPosition:
    thumb_x: float
    thumb_y: float
    index_x: float 
    index_y: float
    middle_x: float
    middle_y: float
    ring_x: float
    ring_y: float
    pinky_x: float
    pinky_y: float
    active_finger_x: float  # The chosen finger for object control
    active_finger_y: float

@dataclass
class VirtualObject:
    x: float
    y: float
    size: float = 40.0  # Size of cube sides
    original_x: float = 0.0  # Center of plane
    original_y: float = 0.0  # Center of plane
    is_pinched: bool = False
    prev_x: float = 0.0  # Track previous x position for movement detection
    prev_y: float = 0.0  # Track previous y position for movement detection

    movement_phase: int = 1  # Tracks the movement phase (1, 2, 3)
    progress: float = 0.0   # Tracks the movement progress (0.0 to 1.0)
    cycle_counter: int = 0  # Counts full movement cycles

FingerPair = Literal["index", "middle", "ring"]

class HandTracker:
    def __init__(self, finger_pair: FingerPair = "index", width: int = 640, height: int = 480, camera_fps: int = 30,
                 server_address: str = "localhost", server_port: int = 12345):
        self._hand_position_lock = Lock()
        self._visualization_lock = Lock()
        self._current_position: Optional[HandPosition] = None
        self._is_pinching = False
        self._width = width
        self._height = height
        self._virtual_world_fps = 30
        self._camera_fps = camera_fps
        
        # Movement tracking thresholds
        self.CENTER_THRESHOLD = 20  # Pixels from center to start movement
        self.EDGE_THRESHOLD = 30  # Pixels from edge to count as reached edge
        self.last_x = width/2  # Track last x position
        self.last_y = height/2  # Track last y position
        self.reached_edge = False  # Track if reached edge
        self.in_center = True  # Track if in center
        
        # UDP server config
        self._server_address = server_address
        self._server_port = server_port
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Create virtual object at center of screen
        self._virtual_object = VirtualObject(
            x=self._width/2,
            y=self._height/2,
            original_x=self._width/2,
            original_y=self._height/2,
            prev_x=self._width/2,
            prev_y=self._height/2
        )
        self._running = True
        
        # Set finger landmark based on selected pair
        self._finger_landmark = {
            "index": mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
            "middle": mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
            "ring": mp.solutions.hands.HandLandmark.RING_FINGER_TIP
        }[finger_pair]
        
        # Camera indices
        self.TOP_CAMERA = 0
        self.SIDE_CAMERA = 1
        
        # Threshold for pinch detection (adjust as needed)
        self.BASE_PINCH_THRESHOLD = 1.5  # pixels
        
        # Initialize pygame for visualization and keyboard input
        pygame.init()
        pygame.font.init()
        self._font = pygame.font.SysFont('Arial', 30)
        
        # Start visualization thread
        self._viz_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self._viz_thread.start()

        # Start UDP sender thread
        self._network_thread = threading.Thread(target=self._network_loop, daemon=True)
        self._network_thread.start()

        if not ARDUINO_DEBUG:
            # Arduino serial connection
            try:
                self._arduino = serial.Serial('COM3', 115200)  # Windows serial port
            except Exception as e:
                print(f"Failed to connect to Arduino - serial control disabled. Error: {e}")
                self._arduino = None
                
            # Start Arduino control thread
            self._arduino_thread = threading.Thread(target=self._arduino_control_loop, daemon=True)
            self._arduino_thread.start()

        self._top_thread = threading.Thread(target=self.start_top_camera, daemon=True)
        self._top_thread.start()

        if not SIDE_CAMERA_DEBUG:        
            self._side_thread = threading.Thread(target=self.start_side_camera, daemon=True)
            self._side_thread.start()

    def _arduino_control_loop(self):
        """Thread that controls Arduino based on hand movement"""
        MOVEMENT_THRESHOLD = 5.0  # Minimum movement to trigger signal
        steady_flag = True
        
        while self._running and self._arduino:
            if self._virtual_object.is_pinched:
                steady_flag = False
                x_movement = self._virtual_object.x - self._virtual_object.prev_x
                y_movement = self._virtual_object.y - self._virtual_object.prev_y
                
                if abs(x_movement) > MOVEMENT_THRESHOLD or abs(y_movement) > MOVEMENT_THRESHOLD:
                    if abs(x_movement) > abs(y_movement):
                        if x_movement < 0:  # Moving left
                            self._arduino.write(b'L')  # Left signal
                        else:  # Moving right
                            self._arduino.write(b'R')  # Right signal
                    else:
                        if y_movement < 0:  # Moving up
                            self._arduino.write(b'U')  # Up signal
                        else:  # Moving down
                            self._arduino.write(b'D')  # Down signal
            elif not steady_flag:
                self._arduino.write(b'S')
                steady_flag = True
                            
            sleep(1.0 / self._virtual_world_fps)

    def _network_loop(self):
        """Thread that sends hand position data over UDP"""
        while self._running:
            current_pos = self.get_hand_position()
            if current_pos:
                # Convert to list of finger positions
                finger_positions = [
                    {"x": current_pos.thumb_x / self._width, "z": current_pos.thumb_y / self._height},
                    {"x": current_pos.index_x / self._width, "z": current_pos.index_y / self._height}, 
                    {"x": current_pos.middle_x / self._width, "z": current_pos.middle_y / self._height},
                    {"x": current_pos.ring_x / self._width, "z": current_pos.ring_y / self._height},
                    {"x": current_pos.pinky_x / self._width, "z": current_pos.pinky_y / self._height}
                ]

                json_data = json.dumps({
                    "landmarks": finger_positions,
                    "trackingObject": {
                        "x": self._virtual_object.x / self._width,
                        "z": self._virtual_object.y / self._height,
                        "progress": self._virtual_object.progress,
                        "cycleCount": self._virtual_object.cycle_counter
                    }
                })
                
                try:
                    # * Send information to unity via udp socket
                    self._udp_socket.sendto(json_data.encode("utf-8"), (self._server_address, self._server_port))
                except Exception as e:
                    print(f"Failed to send UDP data: {e}")
                    
            sleep(1.0 / self._virtual_world_fps)

    def _calculate_pinch_threshold(self, depth: float) -> float:
        """Adjust threshold based on depth (distance from side camera)"""
        return self.BASE_PINCH_THRESHOLD * (depth)
    
    def _configure_camera(self, cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, self._camera_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

    def _update_movement_progress(self):
        """Update movement progress based on object position"""
        # ! BUG: Bar does not respond on the way back from edge to center
        # ! Proposed Solution: Half bar to edge in color X, Half bar to centery in color Y

        # ! BUG: If we unpinch after reaching edge, it moves object to center which increases the count by one when pinching again
        # ! Proposed Solution: Track movement back (half bar-half bar), only successful on reaching center by movement (same as solution above)
        if not self._virtual_object.is_pinched:
            self._virtual_object.progress = 0.0
            self.reached_edge = False
            self.in_center = True
            return

        center_x = self._width / 2
        center_y = self._height / 2
        distance_from_center = math.sqrt(
            (self._virtual_object.x - center_x)**2 + 
            (self._virtual_object.y - center_y)**2
        )
        
        # Check if in center region
        if distance_from_center < self.CENTER_THRESHOLD:
            if not self.in_center and self.reached_edge:
                # Completed full movement
                self._virtual_object.cycle_counter += 1
                self._virtual_object.progress = 0.0
                self.reached_edge = False
            self.in_center = True
        else:
            self.in_center = False
            
        # Check if reached edge
        max_distance = min(self._width/2, self._height/2) - self.EDGE_THRESHOLD
        if distance_from_center > max_distance:
            self.reached_edge = True
            
        # Update progress
        if not self.reached_edge:
            self._virtual_object.progress = min(distance_from_center / max_distance, 1.0)
            
            # If moving back to center before reaching edge, progress drops
            last_distance = math.sqrt(
                (self.last_x - center_x)**2 + 
                (self.last_y - center_y)**2
            )
            if distance_from_center < last_distance:
                self._virtual_object.progress = max(0.0, self._virtual_object.progress - 0.1)
                
        self.last_x = self._virtual_object.x
        self.last_y = self._virtual_object.y

    def _update_virtual_object(self):
        """Update virtual object position based on hand position and pinch state"""
        if self._current_position is None:
            return
            
        # Check if fingers are near object for pinching
        finger_midpoint_x = (self._current_position.thumb_x + self._current_position.active_finger_x) / 2
        finger_midpoint_y = (self._current_position.thumb_y + self._current_position.active_finger_y) / 2
        
        distance_to_object = math.sqrt(
            (finger_midpoint_x - self._virtual_object.x)**2 + 
            (finger_midpoint_y - self._virtual_object.y)**2
        )
        
        # Update object pinch state
        if self._is_pinching and distance_to_object < self._virtual_object.size * 0.5:
            self._virtual_object.is_pinched = True
        elif not self._is_pinching:
            self._virtual_object.is_pinched = False
            
        # Store previous positions
        self._virtual_object.prev_x = self._virtual_object.x
        self._virtual_object.prev_y = self._virtual_object.y
            
        # Move object with fingers if pinched
        if self._virtual_object.is_pinched:
            self._virtual_object.x = finger_midpoint_x
            self._virtual_object.y = finger_midpoint_y
            self._update_movement_progress()
        else:
            # Return to original position when released
            self._virtual_object.x = self._virtual_object.original_x
            self._virtual_object.y = self._virtual_object.original_y
            self._virtual_object.progress = 0.0

    def _draw_visualization(self):
        """Draw fingers and virtual object visualization"""
        # TODO - Add red rectangle signaling the middle
        with self._visualization_lock:
            for event in pygame.event.get():
                if not self._handle_pygame_events(event):
                    return

            self.screen.fill((0, 0, 0))
            
            current_position = self.get_hand_position()
            if current_position:
                # Draw all fingers as circles
                pygame.draw.circle(self.screen, (255, 0, 0),
                                 (int(current_position.thumb_x), int(current_position.thumb_y)), 5)
                pygame.draw.circle(self.screen, (0, 0, 255),
                                 (int(current_position.index_x), int(current_position.index_y)), 5)
                pygame.draw.circle(self.screen, (0, 255, 0),
                                 (int(current_position.middle_x), int(current_position.middle_y)), 5)
                pygame.draw.circle(self.screen, (255, 255, 0),
                                 (int(current_position.ring_x), int(current_position.ring_y)), 5)
                pygame.draw.circle(self.screen, (255, 0, 255),
                                 (int(current_position.pinky_x), int(current_position.pinky_y)), 5)
                
                pygame.draw.line(self.screen, (0, 255, 0),
                               (int(current_position.thumb_x), int(current_position.thumb_y)),
                               (int(current_position.active_finger_x), int(current_position.active_finger_y)), 2)
            
            # Draw virtual object
            color = (255, 165, 0) if self._virtual_object.is_pinched else (128, 128, 128)
            half_size = self._virtual_object.size / 2
            rect = pygame.Rect(
                int(self._virtual_object.x - half_size),
                int(self._virtual_object.y - half_size),
                int(self._virtual_object.size),
                int(self._virtual_object.size)
            )
            pygame.draw.rect(self.screen, color, rect)
            
            # Draw progress bar
            bar_width = 200
            bar_height = 20
            bar_x = (self._width - bar_width) // 2
            bar_y = self._height - 40
            
            # Background bar
            pygame.draw.rect(self.screen, (64, 64, 64),
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Progress fill
            fill_width = int(bar_width * self._virtual_object.progress)
            if fill_width > 0:
                pygame.draw.rect(self.screen, (0, 255, 0),
                               (bar_x, bar_y, fill_width, bar_height))
                
            # Draw movement counter
            counter_text = self._font.render(str(self._virtual_object.cycle_counter), True, (255, 255, 255))
            self.screen.blit(counter_text, (self._width - 50, self._height - 40))

            pygame.display.flip()

    def _visualization_loop(self):
        """Separate thread for visualization updates"""
        self.screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption("Hand Tracking Visualization")
        clock = pygame.time.Clock()

        while self._running:
            self._draw_visualization()
            clock.tick(self._virtual_world_fps)

    def _handle_pygame_events(self, event: pygame.event.Event) -> bool:
        """ Handle pygame events and returns if the program should continue running"""
        match event.type:
            case pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.toggle_pinch()
            case pygame.QUIT:
                self._running = False
                pygame.quit()
                return False
        return True

    def start_top_camera(self):
        """Process top camera feed to track finger positions"""
        backend = cv2.CAP_DSHOW
        cap = cv2.VideoCapture(self.TOP_CAMERA, backend) if backend else cv2.VideoCapture()
        if not cap.isOpened():
            raise RuntimeError("Failed to open top camera")

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

                detected = self._process_top_view(frame, hands)
                cv2.imshow("Top Camera", cv2.flip(frame, 1))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                with self._hand_position_lock:
                    self._current_position = detected
                    self._update_virtual_object()

    def start_side_camera(self):
        """Process side camera feed to detect pinch gestures"""
        cap = cv2.VideoCapture(self.SIDE_CAMERA)
        if not cap.isOpened():
            print("Side camera not available - using space key for pinch control")
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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def get_hand_position(self) -> Optional[HandPosition]:
        """Get the current hand position"""
        with self._hand_position_lock:
            return self._current_position

    def is_pinching(self) -> bool:
        """Get current pinch state"""
        return self._is_pinching

    def toggle_pinch(self):
        """Toggle pinch state when not using camera"""
        print(f"toggle pinch from {self._is_pinching} to {not self._is_pinching}")
        self._is_pinching = not self._is_pinching

    def _process_top_view(self, frame: np.ndarray, hands: Any) -> HandPosition:
        """Process top camera frame to detect finger positions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get positions for all fingers
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            
            # Get active finger for object control
            active_tip = hand_landmarks.landmark[self._finger_landmark]
            
            return HandPosition(
                thumb_x=self._width - thumb_tip.x * self._width,
                thumb_y=thumb_tip.y * self._height,
                index_x=self._width - index_tip.x * self._width,
                index_y=index_tip.y * self._height,
                middle_x=self._width - middle_tip.x * self._width,
                middle_y=middle_tip.y * self._height,
                ring_x=self._width - ring_tip.x * self._width,
                ring_y=ring_tip.y * self._height,
                pinky_x=self._width - pinky_tip.x * self._width,
                pinky_y=pinky_tip.y * self._height,
                active_finger_x=self._width - active_tip.x * self._width,
                active_finger_y=active_tip.y * self._height
            )
            
        return HandPosition(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _detect_pinch(self, frame: np.ndarray, hands: Any):
        """Process side camera frame to detect pinch gesture"""
        # ! BUG: Detect pinch logic is shit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            finger_tip = hand_landmarks.landmark[self._finger_landmark]
            
            distance = math.sqrt(
                (thumb_tip.x - finger_tip.x)**2 + 
                (thumb_tip.y - finger_tip.y)**2
            )
            
            depth = abs(finger_tip.z)
            threshold = self._calculate_pinch_threshold(depth)
            self._is_pinching = distance < threshold
        else:
            self._is_pinching = False
        
    def cleanup(self):
        """Clean up resources"""
        self._running = False
        cv2.destroyAllWindows()
        pygame.quit()
        self._udp_socket.close()
        if not ARDUINO_DEBUG and self._arduino:
            self._arduino.close()

def start_hand_tracking(finger_pair: FingerPair = "index"):
    """Initialize and start the hand tracking system"""
    tracker = HandTracker(finger_pair)
    
    return tracker

if __name__ == "__main__":
    config = []
    with open('configuration.csv', 'r') as file:
        csv_reader = csv.reader(file)
        config = [[int(val) for val in row] for row in csv_reader]

    tracker = start_hand_tracking()
    try:
        while True:
            sleep(0.1)
    except KeyboardInterrupt:
        tracker.cleanup()
