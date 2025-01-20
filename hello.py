import csv
from datetime import datetime
from pathlib import Path
import keyboard
import math
from time import sleep
import cv2
import threading
from typing import Any, Optional
from threading import Lock
import numpy as np
import mediapipe as mp
import socket
from pydantic import BaseModel
import serial
from structures import ExperimentControl, ExperimentState, FingerPosition, QuestionInput, StateData, TrackingObject, ExperimentPacket
from consts import BACKEND_PORT, HEIGHT, PAUSE_SLEEP_SECONDS, PYGAME_PORT, TARGET_CYCLE_COUNT, VIRTUAL_WORLD_FPS, WIDTH, FingerPair
from queue import Queue

# Create experiment folder if it doesn't exist
experiment_folder = Path("experiment")
if not experiment_folder.exists():
    experiment_folder.mkdir()

ARDUINO_DEBUG = True
SIDE_CAMERA_DEBUG = True

class StiffnessValue(BaseModel):
    value: int
    # ! Add support for different fingers
    # finger: FingerPair


class StiffnessPair(BaseModel):
    first: StiffnessValue
    second: StiffnessValue

    
class Configuration(BaseModel):
    pairs: list[StiffnessPair]

    # * Special constructor that accepts path and returns Configuration (instead of acception "pairs")
    @staticmethod
    def read_configuration(path: str):
        if not Path(path).exists():
            raise Exception("Missing configuration file")
        
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            return Configuration(pairs = [
                StiffnessPair(first=StiffnessValue(value=int(row[0])),
                                second=StiffnessValue(value=int(row[1]))) for row in csv_reader
            ])
    
    def write_configuration(self, path: str):
        """Write configuration pairs to CSV file"""
        with open(path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for pair in self.pairs:
                csv_writer.writerow([pair.first.value, pair.second.value])
        
        


class HandPosition(BaseModel):
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

class VirtualObject(BaseModel):
    x: float = 0.0
    y: float = 0.0
    size: float = 40.0  # Size of cube sides
    original_x: float = 0.0  # Center of plane
    original_y: float = 0.0  # Center of plane
    is_pinched: bool = False
    prev_x: float = 0.0  # Track previous x position for movement detection
    prev_y: float = 0.0  # Track previous y position for movement detection

    progress: float = 0.0   # Tracks the movement progress (0.0 to 1.0)
    cycle_counter: int = 0  # Counts full movement cycles

    stiffness_value: int = 0 # Object stiffness value as read from the configuration
    pair_index: int = 0 # 0 or 1


    def model_post_init(self, __context) -> None:
        self.reset()

    def reset(self):
        # Reset object location
        self.x = self.original_x
        self.y = self.original_y
        self.prev_x = self.original_x
        self.prev_y = self.original_y
        self.is_pinched = False

        # reset progress/Cycle Counter
        self.progress = 0.0
        self.cycle_counter = 0

        # Reset the stiffness value/pair index since the previous value is no longer 
        self.stiffness_value = 0
        self.pair_index = 0

class Experiment:
    def __init__(self, config: Configuration, path: Path, finger_pair: FingerPair = "index", width: int = WIDTH, height: int = HEIGHT, camera_fps: int = 30,
                 server_address: str = "localhost", frontend_port: int = PYGAME_PORT, backend_port: int = BACKEND_PORT):
        
        # SETUP
        self._hand_position_lock = Lock()
        self._visualization_lock = Lock()
        self._current_position: Optional[HandPosition] = None
        self._is_pinching = False
        self._config = config
        self._path = path
        self._width = width
        self._height = height
        self._virtual_world_fps = VIRTUAL_WORLD_FPS
        self._camera_fps = camera_fps
        # TODO - add Start screen if needed, and adjust to ExperimentState.START
        self._state = ExperimentState.COMPARISON
        self._pause_time = 0
        self._pair_counter = 0  # Counter for pair folders

        # Frame queues for recording
        self._top_frame_queue = Queue(maxsize=30)
        self._side_frame_queue = Queue(maxsize=30)
        
        # Movement tracking thresholds
        self.CENTER_THRESHOLD = 20  # Pixels from center to start movement
        self.EDGE_THRESHOLD = 30  # Pixels from edge to count as reached edge
        self.last_x = width/2  # Track last x position
        self.last_y = height/2  # Track last y position
        self.reached_edge = False  # Track if reached edge
        self.in_center = True  # Track if in center
        
        # UDP server config
        self._server_address = server_address
        self._frontend_port = frontend_port
        self._backend_port = backend_port
        self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Bind sockets
        print("Waiting for connection")
        self._listening_socket.bind((self._server_address, self._backend_port))
        self._listening_socket.listen()
        self._control_socket, _ = self._listening_socket.accept()
        print("Connection accepted")

        # Create virtual object at center of screen
        self._virtual_object = VirtualObject(
            original_x=self._width/2,
            original_y=self._height/2,
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
        
        # Video writers
        self._top_writer = None
        self._side_writer = None
        
        # Threshold for pinch detection (adjust as needed)
        self.BASE_PINCH_THRESHOLD = 1.5  # pixels

        # Start recording threads
        self._top_recording_thread = threading.Thread(target=self._record_top_frames, daemon=True)
        self._top_recording_thread.start()
        
        if not SIDE_CAMERA_DEBUG:
            self._side_recording_thread = threading.Thread(target=self._record_side_frames, daemon=True)
            self._side_recording_thread.start()

        # Start Experiment Management thread
        self._experiment_thread = threading.Thread(target=self._experiment_loop, daemon=True)
        self._experiment_thread.start()

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

    def _get_pair_path(self) -> Path:
        """Get the path for the current pair of stiffness values"""
        return self._path / f"pair_{self._pair_counter:03d}"

    def _create_pair_folder(self) -> None:
        """Create a folder for the current pair of stiffness values"""
        pair_path = self._get_pair_path()
        pair_path.mkdir()
        
    def _initialize_writers(self) -> None:
        # Initialize video writers for this pair
        pair_path = self._get_pair_path()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._top_writer = cv2.VideoWriter(str(pair_path / 'top_camera.mp4'), fourcc, self._camera_fps, (self._width, self._height))
        if not SIDE_CAMERA_DEBUG:
            self._side_writer = cv2.VideoWriter(str(pair_path / 'side_camera.mp4'), fourcc, self._camera_fps, (self._width, self._height))

    def _setup_pair(self) -> None:
        """ Setup everything needed for a new pair """
        self._create_pair_folder()
        self._initialize_writers()

    def _cleanup_writers(self):
        """Clean up video writers after pair is complete"""
        if self._top_writer:
            self._top_writer.release()
            self._top_writer = None
        if self._side_writer:
            self._side_writer.release() 
            self._side_writer = None

    def _sleep(self):
        sleep(1.0 / self._virtual_world_fps)

    def _update_virtual_object(self, stiffness_value: int, pair_index: int) -> None:
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

        # Update stiffness and index
        self._virtual_object.stiffness_value = stiffness_value
        self._virtual_object.pair_index = pair_index

    def _check_comparison_end(self) -> bool:
        return self._virtual_object.cycle_counter == TARGET_CYCLE_COUNT

    def _reset_comparison(self):
        self._is_pinching = False
        self._virtual_object.reset()

    def _experiment_loop(self):
        answers: list[QuestionInput] = []
        
        for pair in self._config.pairs:
            if not self._running:
                print("Exit via Ctrl-C")
                break

            # Read the next pair and decide action
            if pair.first.value == -1:
                if pair.second.value != -1:
                    raise Exception("Misaligned numbers, invalidated configuration/experiment - please contact the staff")
                
                print("Exit via end of experiment")
                self._state = ExperimentState.END
                break

            elif pair.first.value == 0:
                if pair.second.value != 0:
                    raise Exception("Misaligned numbers, invalidated configuration/experiment - please contact the staff")

                print("Pausing")
                self._pause_time = PAUSE_SLEEP_SECONDS
                self._state = ExperimentState.PAUSE
                for i in range(PAUSE_SLEEP_SECONDS, 0, -1):
                    # sleeping one second at a time to update frontend + exit via Ctrl-C during pause
                    sleep(1)
                    self._pause_time = i
                    if not self._running:
                        break;
            
                self._pause_time = 0

            else:
                print("Comparing")
                self._pair_counter += 1
                self._setup_pair()
                self._state = ExperimentState.COMPARISON

                # For each object in pair pass the stiffness value and pair index to the virtual
                self._reset_comparison()
                while self._running and not self._check_comparison_end():
                    self._update_virtual_object(pair.first.value, 0)

                self._reset_comparison()
                while self._running and not self._check_comparison_end():
                    self._update_virtual_object(pair.second.value, 1)

                # Clean up video writers after pair is complete
                self._cleanup_writers()

                if not self._running:
                    print("Exit via Ctrl-C")
                    break

                print("Question")
                self._state = ExperimentState.QUESTION
                # .recv is a blocking call, waiting for input from the frontend
                # TODO - add support for exit via Ctrl-C
                answer_data = self._control_socket.recv(1024)
                answer = ExperimentControl.model_validate_json(answer_data)
                answers.append(QuestionInput(answer.questionInput))


    def _arduino_control_loop(self):
        """Thread that controls Arduino based on hand movement"""
        MOVEMENT_THRESHOLD = 5.0  # Minimum movement to trigger signal
        steady_flag = True
        
        while self._running and self._arduino:
            if self._virtual_object.is_pinched:
                steady_flag = False
                x_movement = self._virtual_object.x - self._virtual_object.prev_x
                y_movement = self._virtual_object.y - self._virtual_object.prev_y
                
                stiffness_value = self._virtual_object.stiffness_value
                if abs(x_movement) > MOVEMENT_THRESHOLD or abs(y_movement) > MOVEMENT_THRESHOLD:
                    if abs(x_movement) > abs(y_movement):
                        # TODO - Add support on arduino/motor side for this communication protocol (with stiffness values)
                        if x_movement < 0:  # Moving left
                            self._arduino.write(f'L{stiffness_value}'.encode())  # Left signal
                        else:  # Moving right
                            self._arduino.write(f'R{stiffness_value}'.encode())  # Right signal
                    else:
                        if y_movement < 0:  # Moving up
                            self._arduino.write(f'U{stiffness_value}'.encode())  # Up signal
                        else:  # Moving down
                            self._arduino.write(f'D{stiffness_value}'.encode())  # Down signal
            elif not steady_flag:
                self._arduino.write(b'S')
                steady_flag = True
                            
            self._sleep()

    def _network_loop(self):
        """Thread that sends hand position data over UDP"""
        while self._running:
            current_pos = self.get_hand_position()
            if current_pos:
                # Convert to list of finger positions
                finger_positions = [
                    FingerPosition(x=current_pos.thumb_x / self._width, z=current_pos.thumb_y / self._height),
                    FingerPosition(x=current_pos.index_x / self._width, z=current_pos.index_y / self._height), 
                    FingerPosition(x=current_pos.middle_x / self._width, z=current_pos.middle_y / self._height),
                    FingerPosition(x=current_pos.ring_x / self._width, z=current_pos.ring_y / self._height),
                    FingerPosition(x=current_pos.pinky_x / self._width, z=current_pos.pinky_y / self._height)
                ]

                packet = ExperimentPacket(
                    stateData=StateData(state=self._state.value, pauseTime=self._pause_time),
                    landmarks=finger_positions,
                    trackingObject=TrackingObject(
                        x=self._virtual_object.x / self._width,
                        z=self._virtual_object.y / self._height,
                        size=self._virtual_object.size,
                        isPinched=self._virtual_object.is_pinched,
                        progress=self._virtual_object.progress,
                        cycleCount=self._virtual_object.cycle_counter,
                        pairIndex=self._virtual_object.pair_index
                    )
                )

                try:
                    # * Send information to unity via udp socket
                    self._data_socket.sendto(packet.model_dump_json().encode("utf-8"), (self._server_address, self._frontend_port))
                except Exception as e:
                    print(f"Failed to send UDP data: {e}")
                    
            self._sleep()

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

    def _record_top_frames(self):
        """Thread for recording frames from top camera"""
        while self._running:
            if self._top_writer:
                try:
                    data: tuple[np.ndarray, HandPosition] = self._top_frame_queue.get(timeout=1)
                    frame, hand_position = data
                    self._top_writer.write(frame)
                    
                    # Write hand position data to CSV
                    pair_path = self._get_pair_path()
                    tracking_file = pair_path / 'tracking.csv'

                    position_dict = hand_position.model_dump()
                    
                    # Create CSV with headers if it doesn't exist
                    if not tracking_file.exists():
                        headers = ['timestamp', 'pinching', 'stiffness', 'object_x', 'object_y', *list(position_dict.keys())]
                        with open(tracking_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)
                    
                    # Append position data
                    with open(tracking_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row_data = [
                            datetime.now().isoformat(),
                            self._is_pinching,
                            self._virtual_object.stiffness_value,
                            self._virtual_object.x,
                            self._virtual_object.y,
                            *list(position_dict.values())
                        ]
                        writer.writerow(row_data)
                        
                except Exception as e:
                    print(e)

    def _record_side_frames(self):
        """Thread for recording frames from side camera"""
        while self._running:
            if self._side_writer:
                try:
                    frame = self._side_frame_queue.get(timeout=1)
                    self._side_writer.write(frame)
                except:
                    pass

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
                
                # Add frame to recording queue if recording
                if self._top_writer:
                    try:
                        self._top_frame_queue.put_nowait((frame, detected))
                    except:
                        pass
                    
                cv2.imshow("Top Camera", cv2.flip(frame, 1))
                key = cv2.waitKey(1) & 0xFF
                if key== ord('q'):
                    break
                elif key == ord(' '):
                    self._toggle_pinch()
                
                with self._hand_position_lock:
                    self._current_position = detected

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
                
                # Add frame to recording queue if recording
                if self._side_writer:
                    try:
                        self._side_frame_queue.put_nowait(frame)
                    except:
                        pass
                    
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

    def _toggle_pinch(self):
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
            
        return HandPosition(
            thumb_x=0.0,
            thumb_y=0.0,
            index_x=0.0,
            index_y=0.0,
            middle_x=0.0,
            middle_y=0.0,
            ring_x=0.0,
            ring_y=0.0,
            pinky_x=0.0,
            pinky_y=0.0,
            active_finger_x=0.0,
            active_finger_y=0.0
        )

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
        keyboard.unhook_all()
        cv2.destroyAllWindows()
        self._data_socket.close()
        self._listening_socket.close()
        self._control_socket.close()
        self._cleanup_writers()
        if not ARDUINO_DEBUG and self._arduino:
            self._arduino.close()

def start_experiment(config: Configuration, path: Path, finger_pair: FingerPair = "index"):
    """Initialize and start the hand tracking system"""
    experiment = Experiment(config, path, finger_pair)
    
    return experiment

def create_experiment_folder(config: Configuration):
    # Check if experiment folder exists, create if not
    experiment_path = Path("experiment")
    if not experiment_path.exists():
        experiment_path.mkdir()
    
    # Get list of existing folders and find highest number
    existing_folders = [f for f in experiment_path.iterdir() if f.is_dir()]
    if not existing_folders:
        # Create first folder 001 if no folders exist
        folder_id = 1
    else:
        # Get highest numbered folder and increment
        folder_id = max(int(folder.name.split('_')[-1]) for folder in existing_folders) + 1

    # Get user input for A or B
    while True:
        user_input = input("Please enter A or B: ").upper()
        if user_input in ['A', 'B']:
            break
        print("Invalid input. Please enter either A or B.")

    # Create folder name with timestamp, ID and user input
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_path = experiment_path / f"{timestamp}_{user_input}_{folder_id:03d}"
    
    # Create the new folder
    folder_path.mkdir()

    # Write configuration to new folder
    config.write_configuration(folder_path / "configuration.csv")

    return folder_path

def main():
    config = Configuration.read_configuration('configuration.csv')
    path = create_experiment_folder(config)
    
    experiment = start_experiment(config, path)
    try:
        while True:
            sleep(0.1)
    except KeyboardInterrupt:
        experiment.cleanup()


if __name__ == "__main__":
    main()
