import csv
from datetime import datetime
from pathlib import Path
import keyboard
import math
from time import sleep
import cv2
import threading
from typing import Literal, Optional
from threading import Lock
import numpy as np
import socket
from pydantic import BaseModel
from structures import ExperimentControl, ExperimentState, FingerPosition, StateData, TrackingObject, ExperimentPacket
from consts import BACKEND_PORT, PAUSE_SLEEP_SECONDS, MOTORS_COMMUNICATION_RATE, PYGAME_PORT, TARGET_CYCLE_COUNT, HARDWARE_PORT, TOP_HEIGHT, TOP_WIDTH, SIDE_HEIGHT, SIDE_WIDTH, FRONTEND_FPS, VIRTUAL_OBJECT_FPS, PairFinger, CENTER_THRESHOLD, EDGE_THRESHOLD
import queue
from queue import Queue
from enum import StrEnum
from vision import ColorVision, HandPosition, MediapipeVision, YoloVision, FINGER_COLORS
from motor_controller import FingerId, HandOrientation, MovementStrategy, MotorController
from haptic_mapping import map_object_displacement_to_tactor

DEBUG_POSITION = 1000
DEBUG_SINGLE_MOTOR = False
DEBUG_FLIP_Y = False
DEBUG_SIDE_CAMERA_OFF = True
DEBUG_ALLOW_PRINTS = False
DEBUG_LOG_MOTOR_RANGE = True  # Log min/max motor positions on shutdown (no per-message overhead)

original_print = print
def print(*args, **kwargs):
    if DEBUG_ALLOW_PRINTS:
        original_print(*args, **kwargs)

class VisionType(StrEnum):
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"
    FRAME_COLOR = "frame_color"
    OPTICAL_COLOR_NO_GPU = "optical_color_no_gpu"
    OPTICAL_COLOR_GPU = "optical_color_gpu"

class MotorType(StrEnum):
    HARDWARE = "hardware"
    NONE = "none"

MOVE_FACTOR = 3
MOTOR_TYPE = MotorType.HARDWARE
MOVEMENT_STRATEGY = MovementStrategy.FREE_FORM
MOTOR_OPPOSES_OBJECT_MOTION = True
VISION_TYPE = VisionType.MEDIAPIPE
RECORDING_DATA = True


# Create experiment folder if it doesn't exist
REAL_EXPERIMENTS_FOLDER = Path("live_experiments")
DEBUG_EXPERIMENTS_FOLDER = Path("debug_experiments")
experiment_folder = DEBUG_EXPERIMENTS_FOLDER
if not experiment_folder.exists():
    experiment_folder.mkdir()    

class StiffnessValue(BaseModel):
    value: int
    is_index: bool

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
                StiffnessPair(first=StiffnessValue(
                    value=int(row[0]),
                    is_index=(int(row[1]) == 1)
                ),
                second=StiffnessValue(
                    value=int(row[2]),
                    is_index=(int(row[3]) == 1)
                )
                ) for row in csv_reader
            ])
    
    def write_configuration(self, path: str):
        """Write configuration pairs to CSV file"""
        with open(path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for pair in self.pairs:
                csv_writer.writerow([pair.first.value, 1 if pair.first.is_index else 0, pair.second.value, 1 if pair.second.is_index else 0])

class VirtualObject(BaseModel):
    x: float = 0.0
    y: float = 0.0
    size: float = 40.0  # Size of cube sides
    original_x: float = 0.0  # Center of plane
    original_y: float = 0.0  # Center of plane
    is_pinched: bool = False

    progress: float = 0.0   # Tracks the movement progress (0.0 to 1.0)
    return_progress: float = 0.0  # Tracks edge-to-center visual progress (0.0 to 1.0)
    cycle_counter: int = 0  # Counts full movement cycles

    stiffness_value: int = 0 # Object stiffness value as read from the configuration
    pair_index: int = 0 # 0 or 1


    def model_post_init(self, __context) -> None:
        self.reset()

    def reset(self):
        # Reset object location
        self.x = self.original_x
        self.y = self.original_y
        self.is_pinched = False

        # reset progress/Cycle Counter
        self.progress = 0.0
        self.return_progress = 0.0
        self.cycle_counter = 0

        # Reset the stiffness value/pair index since the previous value is no longer 
        self.stiffness_value = 0
        self.pair_index = 0

class Experiment:
    def __init__(self,
        config: Configuration,
        path: Path,
        pair_finger: PairFinger,
        hand_orientation: HandOrientation = HandOrientation.NOT_MIRRORED,
        target_cycle_count: int = TARGET_CYCLE_COUNT,
        top_width: int = TOP_WIDTH,
        top_height: int = TOP_HEIGHT,
        side_width: int = SIDE_WIDTH,
        side_height: int = SIDE_HEIGHT,
        camera_fps: int = 30,
        side_camera_location: Literal["top", "bottom", "left", "right"] = "bottom",
        server_address: str = "localhost",
        frontend_port: int = PYGAME_PORT,
        backend_port: int = BACKEND_PORT,
        hardware_port: int = HARDWARE_PORT
    ):
        
        # SETUP
        self._hand_position_lock = Lock()
        self._visualization_lock = Lock()
        self._current_position: Optional[HandPosition] = None
        self._is_pinching = False
        self._config = config
        self._path = path
        self._top_width = top_width
        self._top_height = top_height
        self._side_width = side_width
        self._side_height = side_height
        self._virtual_object_fps = VIRTUAL_OBJECT_FPS
        self._frontend_fps = FRONTEND_FPS
        self._motors_communication_rate = MOTORS_COMMUNICATION_RATE
        self._side_camera_location: Literal["top", "bottom", "left", "right"] = side_camera_location
        self._camera_fps = camera_fps
        # TODO - add Start screen if needed,
        # and adjust to ExperimentState.START
        self._state = ExperimentState.COMPARISON
        self._pause_time = 0
        self._pair_counter = 0  # Counter for pair folders
        self._pair_finger: PairFinger = pair_finger
        self._target_cycle_count = target_cycle_count
        self._hand_orientation = hand_orientation

        # Frame queues for recording
        self._top_frame_queue = Queue(maxsize=30)
        self._side_frame_queue = Queue(maxsize=30)
        
        # Movement tracking thresholds
        self.CENTER_THRESHOLD = CENTER_THRESHOLD  # Pixels from center to start movement
        self.EDGE_THRESHOLD = EDGE_THRESHOLD  # Pixels from edge to count as reached edge
        self.last_x = self._top_width/2  # Track last x position
        self.last_y = self._top_height/2  # Track last y position
        self.reached_edge = False  # Track if reached edge
        self.in_center = True  # Track if in center
        self._require_unpinch = False  # Require release before next cycle
        
        # Motor Controller
        self._motor_controller = MotorController(
            movement_strategy=MOVEMENT_STRATEGY,
            top_width=self._top_width,
            top_height=self._top_height,
            edge_threshold=self.EDGE_THRESHOLD,
            move_factor=MOVE_FACTOR,
            hand_orientation=self._hand_orientation,
        )

        # UDP server config
        self._server_address = server_address
        self._frontend_port = frontend_port
        self._backend_port = backend_port
        self._hardware_port = hardware_port
        self._frontend_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._hardware_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Bind sockets
        print("Waiting for connection")
        self._listening_socket.bind((self._server_address, self._backend_port))
        self._listening_socket.listen()
        self._control_socket, _ = self._listening_socket.accept()
        print("Connection accepted")

        # Create virtual object at center of screen
        self._virtual_object = VirtualObject(
            original_x=self._top_width/2,
            original_y=self._top_height/2,
        )
        self._running = True

        # Camera indices
        self.TOP_CAMERA = 0
        self.SIDE_CAMERA = 2
        
        # Video writers
        self._top_writer = None
        self._side_writer = None
        
        # Threshold for pinch detection (adjust as needed)
        self.BASE_PINCH_THRESHOLD = 50  # pixels

        match VISION_TYPE:
            case VisionType.MEDIAPIPE:
                self._vision = MediapipeVision(
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    base_pinch_threshold=self.BASE_PINCH_THRESHOLD
                )
            case VisionType.YOLO:
                self._vision = YoloVision(
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    base_pinch_threshold=self.BASE_PINCH_THRESHOLD
                )
            case VisionType.FRAME_COLOR:
                self._vision = ColorVision(
                    finger_colors=FINGER_COLORS,
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    tracking_method="frame",
                    fps=self._camera_fps,
                    base_pinch_threshold=self.BASE_PINCH_THRESHOLD
                )
            case VisionType.OPTICAL_COLOR_NO_GPU:
                self._vision = ColorVision(
                    finger_colors=FINGER_COLORS,
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    tracking_method="optical_flow",
                    fps=self._camera_fps,
                    base_pinch_threshold=self.BASE_PINCH_THRESHOLD,
                    use_gpu=False
                )
            case VisionType.OPTICAL_COLOR_GPU:
                self._vision = ColorVision(
                    finger_colors=FINGER_COLORS,
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    tracking_method="optical_flow",
                    fps=self._camera_fps,
                    base_pinch_threshold=self.BASE_PINCH_THRESHOLD,
                    use_gpu=True
                )
            case _: raise NotImplementedError("Unknown CV type")  # Should never happen

        # * Initialization only until experiment loop begins
        # * Must run after self._vision is initialized
        self._update_active_finger(False)

        # Start recording threads
        self._top_recording_thread = threading.Thread(target=self._record_top_frames, daemon=True)
        self._top_recording_thread.start()
        
        if not DEBUG_SIDE_CAMERA_OFF:
            self._side_recording_thread = threading.Thread(target=self._record_side_frames, daemon=True)
            self._side_recording_thread.start()

        # Start Experiment Management thread
        self._experiment_thread = threading.Thread(target=self._experiment_loop, daemon=True)
        self._experiment_thread.start()

        # Start UDP sender thread
        self._frontend_network_thread = threading.Thread(target=self._frontend_network_loop, daemon=True)
        self._frontend_network_thread.start()

        match MOTOR_TYPE:
            case MotorType.HARDWARE:
                self._hardware_thread = threading.Thread(target=self._hardware_control_loop, daemon=True)
                self._hardware_thread.start()
            case MotorType.NONE:
                ...  # No motor control needed
            case _: raise NotImplementedError("Unknown motor type")  # Should never happen

        self._top_thread = threading.Thread(target=self.start_top_camera, daemon=True)
        self._top_thread.start()

        if not DEBUG_SIDE_CAMERA_OFF:
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
        if RECORDING_DATA:
            pair_path = self._get_pair_path()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
            self._top_writer = cv2.VideoWriter(str(pair_path / 'top_camera.mp4'), fourcc, self._camera_fps, (self._top_width, self._top_height))
            if not DEBUG_SIDE_CAMERA_OFF:
                self._side_writer = cv2.VideoWriter(str(pair_path / 'side_camera.mp4'), fourcc, self._camera_fps, (self._top_width, self._top_height))

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

    def _sleep_virtual_object(self):
        sleep(1.0 / self._virtual_object_fps)

    def _sleep_frontend(self):
        sleep(1.0 / self._frontend_fps)

    def _sleep_motors(self):
        sleep(1.0 / self._motors_communication_rate)

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
        if self._require_unpinch:
            self._virtual_object.is_pinched = False
            if not self._is_pinching:
                self._require_unpinch = False
        elif self._is_pinching and distance_to_object < self._virtual_object.size * 0.5:
            self._virtual_object.is_pinched = True
        elif not self._is_pinching:
            self._virtual_object.is_pinched = False
            
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
            self._virtual_object.return_progress = 0.0

        # Update stiffness and index
        self._virtual_object.stiffness_value = stiffness_value
        self._virtual_object.pair_index = pair_index

    def _check_comparison_end(self) -> bool:
        return self._virtual_object.cycle_counter == self._target_cycle_count

    def _reset_comparison(self):
        self._is_pinching = False
        self._require_unpinch = False
        self._virtual_object.reset()

    def _get_finger_name(self, is_index: bool) -> Literal["index"] | PairFinger:
        return "index" if is_index else self._pair_finger

    def _update_active_finger(self, is_index: bool):
        # Set finger landmark based on selected pair
        self._active_finger = self._get_finger_name(is_index)
        self._vision.set_active_finger(self._active_finger)
        print(f"Active finger set to {self._active_finger}")

    def _experiment_loop(self):

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
                        break
            
                self._pause_time = 0

            else:
                print("Comparing")
                self._pair_counter += 1
                self._setup_pair()
                self._state = ExperimentState.COMPARISON

                # For each object in pair pass the stiffness value and pair index to the virtual
                self._reset_comparison()
                self._update_active_finger(pair.first.is_index)
                while self._running and not self._check_comparison_end():
                    self._update_virtual_object(pair.first.value, 0)
                    self._sleep_virtual_object()

                self._reset_comparison()
                self._update_active_finger(pair.second.is_index)
                while self._running and not self._check_comparison_end():
                    self._update_virtual_object(pair.second.value, 1)
                    self._sleep_virtual_object()

                # Clean up video writers after pair is complete
                self._cleanup_writers()

                if not self._running:
                    print("Exit via Ctrl-C")
                    break

                print("Question")
                question_timestamp = datetime.now()
                self._state = ExperimentState.QUESTION
                # .recv is a blocking call, waiting for input from the frontend
                # TODO - add support for exit via Ctrl-C
                answer_data = self._control_socket.recv(1024)
                answer = ExperimentControl.model_validate_json(answer_data)

                # Write to answers CSV in experiment folder
                answers_file = self._path / 'answers.csv'

                # Add headers if file does not exist
                if not answers_file.exists():
                    with open(answers_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'pair_number', 'object_1_finger', 'object_1_stiffness', 'object_2_finger', 'object_2_stiffness', 'time_to_answer', 'answer'])

                with open(answers_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        question_timestamp.isoformat(),
                        self._pair_counter,
                        self._get_finger_name(pair.first.is_index),
                        pair.first.value,
                        self._get_finger_name(pair.second.is_index),
                        pair.second.value,
                        (datetime.now() - question_timestamp).total_seconds(),
                        answer.questionInput])

    def _hardware_control_loop(self):
        """Thread that controls Hardware via UDP based on hand movement"""
        should_reset = True
        og_obj_x = 0
        og_obj_y = 0

        # Position range tracking (zero per-message overhead, summary on shutdown)
        motor_range_min: dict[int, int] = {}
        motor_range_max: dict[int, int] = {}

        while self._running:
            if self._virtual_object.is_pinched:
                # original x/y != (0,0) since they are half of screen size
                obj_x = self._virtual_object.x - self._virtual_object.original_x
                obj_y = self._virtual_object.y - self._virtual_object.original_y
                if (obj_x == og_obj_x and obj_y == og_obj_y):
                    continue

                og_obj_x = obj_x
                og_obj_y = obj_y
                tactor_x, tactor_y = map_object_displacement_to_tactor(
                    obj_x=obj_x,
                    obj_y=obj_y,
                    oppose_motion=MOTOR_OPPOSES_OBJECT_MOTION,
                )

                finger_id = FingerId.INDEX if self._active_finger == "index" else FingerId.OTHER

                motors_enabled = self._state == ExperimentState.COMPARISON
                motors = self._motor_controller.calculate_motor_movements(
                    finger_id=finger_id,
                    obj_x=tactor_x,
                    obj_y=tactor_y,
                    motors_enabled=motors_enabled,
                    reset_to_origin=should_reset and not motors_enabled,
                )
                should_reset = motors_enabled

                if motors:
                    # Build message
                    if DEBUG_SINGLE_MOTOR:
                        motors = [motors[0]]

                    # Track position ranges (dict lookup only, zero logging overhead)
                    if DEBUG_LOG_MOTOR_RANGE:
                        for m in motors:
                            if m.index not in motor_range_min:
                                motor_range_min[m.index] = m.pos
                                motor_range_max[m.index] = m.pos
                            else:
                                if m.pos < motor_range_min[m.index]:
                                    motor_range_min[m.index] = m.pos
                                if m.pos > motor_range_max[m.index]:
                                    motor_range_max[m.index] = m.pos

                    message = self._motor_controller.build_message(motors)

                    self._hardware_socket.sendto(message.encode("utf-8"), (self._server_address, self._hardware_port))
            self._sleep_motors()

        # Print position range summary on shutdown
        # Uses original_print intentionally to bypass DEBUG_ALLOW_PRINTS
        if DEBUG_LOG_MOTOR_RANGE and motor_range_min:
            original_print("=== Motor Position Range Summary ===")
            for idx in sorted(motor_range_min.keys()):
                original_print(f"  Motor {idx}: min={motor_range_min[idx]}, max={motor_range_max[idx]}")
            original_print("====================================")

    def _frontend_network_loop(self):
        """Thread that sends hand position data over UDP"""
        while self._running:
            current_pos = self.get_hand_position()
            if current_pos:
                # Convert to list of finger positions
                finger_positions = [
                    FingerPosition(x=current_pos.thumb_x / self._top_width, z=current_pos.thumb_y / self._top_height),
                    FingerPosition(x=current_pos.active_finger_x / self._top_width, z=current_pos.active_finger_y / self._top_height)
                ]

                packet = ExperimentPacket(
                    stateData=StateData(state=self._state.value, pauseTime=self._pause_time),
                    landmarks=finger_positions,
                    trackingObject=TrackingObject(
                        x=self._virtual_object.x / self._top_width,
                        z=self._virtual_object.y / self._top_height,
                        size=self._virtual_object.size,
                        isPinched=self._virtual_object.is_pinched,
                        progress=self._virtual_object.progress,
                        returnProgress=self._virtual_object.return_progress,
                        cycleCount=self._virtual_object.cycle_counter,
                        targetCycleCount=self._target_cycle_count,
                        pairIndex=self._virtual_object.pair_index
                    )
                )

                try:
                    # * Send information to unity via udp socket
                    self._frontend_socket.sendto(packet.model_dump_json().encode("utf-8"), (self._server_address, self._frontend_port))
                except queue.Empty:
                    ...  # Expected timeout, no frames available
                except Exception as e:
                    print(f"Failed to send UDP data: {e}")
                    
            self._sleep_frontend()

    def _calculate_pinch_threshold(self, depth: float) -> float:
        """Adjust threshold based on depth (distance from side camera)"""
        return self.BASE_PINCH_THRESHOLD * (depth)
    
    def _configure_camera(self, cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, self._camera_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._top_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._top_height)

    def _update_movement_progress(self):
        """Update movement progress based on object position"""
        # ! BUG: Bar does not respond on the way back from edge to center
        # ! Proposed Solution: Half bar to edge in color X, Half bar to centery in color Y

        # ! BUG: If we unpinch after reaching edge, it moves object to center which increases the count by one when pinching again
        # ! Proposed Solution: Track movement back (half bar-half bar), only successful on reaching center by movement (same as solution above)
        if not self._virtual_object.is_pinched:
            self._virtual_object.progress = 0.0
            self._virtual_object.return_progress = 0.0
            self.reached_edge = False
            self.in_center = True
            return

        center_x = self._top_width / 2
        center_y = self._top_height / 2
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
                self._virtual_object.return_progress = 0.0
                self.reached_edge = False
                self._virtual_object.is_pinched = False
                self._require_unpinch = True
            self.in_center = True
        else:
            self.in_center = False
            
        # Check if reached edge
        max_distance = min(self._top_width/2, self._top_height/2) - self.EDGE_THRESHOLD
        if distance_from_center > max_distance:
            self.reached_edge = True
            
        # Update progress
        if not self.reached_edge:
            self._virtual_object.progress = min(distance_from_center / max_distance, 1.0)
            self._virtual_object.return_progress = 0.0
            
            # If moving back to center before reaching edge, progress drops
            last_distance = math.sqrt(
                (self.last_x - center_x)**2 + 
                (self.last_y - center_y)**2
            )
            if distance_from_center < last_distance:
                self._virtual_object.progress = max(0.0, self._virtual_object.progress - 0.1)
        else:
            return_distance = max_distance - self.CENTER_THRESHOLD
            if return_distance <= 0:
                self._virtual_object.return_progress = 1.0 if distance_from_center < self.CENTER_THRESHOLD else 0.0
            else:
                self._virtual_object.return_progress = min(
                    max((max_distance - distance_from_center) / return_distance, 0.0),
                    1.0
                )
                
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
                        headers = ['timestamp', 'pinching', 'stiffness', 'object_x', 'object_y', 'finger', *list(position_dict.keys())]
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
                            self._active_finger,
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
        cap = cv2.VideoCapture(self.TOP_CAMERA, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Failed to open top camera")

        self._configure_camera(cap)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip frame in y *after* reading (to invert the y axis)
            if DEBUG_FLIP_Y:
                frame = cv2.flip(frame, 0)

            detected = self._process_top_view(frame)
            
            # Add frame to recording queue if recording
            if self._top_writer:
                try:
                    self._top_frame_queue.put_nowait((frame, detected))
                except:
                    pass
                
            cv2.imshow("Top Camera", cv2.flip(frame, 1))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self._toggle_pinch()
            
            with self._hand_position_lock:
                self._current_position = detected

    def start_side_camera(self):
        """Process side camera feed to detect pinch gestures"""
        RETRIES = 10
        
        cap = cv2.VideoCapture(self.SIDE_CAMERA, cv2.CAP_DSHOW)
        for _ in range(RETRIES-1):
            if cap.isOpened():
                break

            cap = cv2.VideoCapture(self.SIDE_CAMERA, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("Side camera not available - using space key for pinch control")
            return
        
        self._configure_camera(cap)
        # Sleeping 3 seconds to give time for the camera to warm up
        sleep(3)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            self._detect_pinch(frame)
            
            # Add frame to recording queue if recording
            if self._side_writer:
                try:
                    self._side_frame_queue.put_nowait(frame)
                except:
                    pass
                
            cv2.imshow("Side Camera", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def get_hand_position(self, with_lock = True) -> Optional[HandPosition]:
        """Get the current hand position"""
        if with_lock:
            with self._hand_position_lock:
                return self._current_position
        
        return self._current_position

    def is_pinching(self) -> bool:
        """Get current pinch state"""
        return self._is_pinching

    def _toggle_pinch(self):
        """Toggle pinch state when not using camera"""
        print(f"toggle pinch from {self._is_pinching} to {not self._is_pinching}")
        self._is_pinching = not self._is_pinching
    
    def _get_finger_color(self) -> str:
        return {
            "index": "green",
            "middle": "blue",
            "ring": "red"
        }[self._active_finger]
    
    def _process_top_view(self, frame: np.ndarray) -> HandPosition:
        return self._vision.detect_hand(frame)

    def _detect_pinch(self, frame: np.ndarray):
        current_pos = self.get_hand_position(with_lock=False)
        if current_pos is None or all([
            current_pos.thumb_x == 0,
            current_pos.thumb_y == 0,
            current_pos.active_finger_x == 0,
            current_pos.active_finger_y == 0,
        ]):
            if self._is_pinching:
                print("Can't find hand, stopping pinch")
                self._is_pinching = False
            return
        
        prev_pinch = self._is_pinching
        self._is_pinching = self._vision.detect_pinch(
            frame,
            current_pos,
            self._side_camera_location
        )
        if prev_pinch != self._is_pinching:
            print(f"Pinch changed to {self._is_pinching}")
        
    def cleanup(self):
        """Clean up resources"""
        self._running = False
        keyboard.unhook_all()
        cv2.destroyAllWindows()
        self._frontend_socket.close()
        self._listening_socket.close()
        self._control_socket.close()
        self._cleanup_writers()
        self._vision.cleanup()

        match MOTOR_TYPE:
            case MotorType.HARDWARE:
                self._hardware_socket.close()
            case MotorType.NONE:
                ...  # No cleanup needed
            case _: raise NotImplementedError("Unknown motor type")  # Should never happen

def start_experiment(
    config: Configuration,
    path: Path,
    pair_finger: PairFinger,
    hand_orientation: HandOrientation,
    target_cycle_count: int,
):
    """Initialize and start the hand tracking system"""
    experiment = Experiment(
        config,
        path,
        pair_finger,
        hand_orientation=hand_orientation,
        target_cycle_count=target_cycle_count
    )
    
    return experiment

def get_target_cycle_count() -> int:
    while True:
        user_input = input(f"How many cycles to apply? [{TARGET_CYCLE_COUNT}]: ").strip()
        if user_input == "":
            return TARGET_CYCLE_COUNT

        if user_input.isdigit() and int(user_input) > 0:
            return int(user_input)

        original_print("Invalid input. Please enter a positive whole number.")


def get_pair_finger() -> PairFinger:
    # Get user input for M or R
    while True:
        user_input = input("Please enter M or R: ").upper()
        if user_input in ['M', 'R']:
            break
        original_print("Invalid input. Please enter either M or R.")
    return "middle" if user_input == "M" else "ring"


def get_hand_orientation() -> HandOrientation:
    """Get user input for mirrored hand orientation (default: not mirrored)."""
    while True:
        user_input = input("Mirror hand orientation? [y/N]: ").strip().lower()
        if user_input in ("", "n", "no"):
            return HandOrientation.NOT_MIRRORED
        if user_input in ("y", "yes"):
            return HandOrientation.MIRRORED
        original_print("Invalid input. Please enter Y or N.")

def create_experiment_folder(config: Configuration, pair_finger: PairFinger):
    # Check if experiment folder exists, create if not
    experiment_path = Path(experiment_folder)
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

    # Create folder name with timestamp, ID and user input
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_path = experiment_path / f"{timestamp}_{pair_finger}_{folder_id:03d}"
    
    # Create the new folder
    folder_path.mkdir()

    # Write configuration to new folder
    config.write_configuration(folder_path / "configuration.csv")

    return folder_path

def main():
    config = Configuration.read_configuration('configuration.csv')
    target_cycle_count = get_target_cycle_count()
    pair_finger = get_pair_finger()
    hand_orientation = get_hand_orientation()
    path = create_experiment_folder(config, pair_finger)
    
    experiment = start_experiment(config, path, pair_finger, hand_orientation, target_cycle_count)
    try:
        while True:
            sleep(0.1)
    except KeyboardInterrupt:
        experiment.cleanup()


if __name__ == "__main__":
    main()
