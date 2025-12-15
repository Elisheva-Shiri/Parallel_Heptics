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
import serial
from structures import ExperimentControl, ExperimentState, FingerPosition, StateData, TrackingObject, ExperimentPacket
from consts import BACKEND_PORT, PAUSE_SLEEP_SECONDS, MOTORS_COMMUNICATION_RATE, PYGAME_PORT, TARGET_CYCLE_COUNT, TECHNOSOFT_PORT, TOP_HEIGHT, TOP_WIDTH, SIDE_HEIGHT, SIDE_WIDTH, FRONTEND_FPS, VIRTUAL_OBJECT_FPS, PairFinger
import queue
from queue import Queue
from enum import Enum, StrEnum
from vision import ColorVision, HandPosition, MediapipeVision, YoloVision, FINGER_COLORS

class VisionType(StrEnum):
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"
    FRAME_COLOR = "frame_color"
    OPTICAL_COLOR_NO_GPU = "optical_color_no_gpu"
    OPTICAL_COLOR_GPU = "optical_color_gpu"

class FingerId(Enum):
    INDEX = 0
    OTHER = 1

class MotorType(StrEnum):
    ARDUINO = "arduino"
    TECHNOSOFT = "technosoft"
    NONE = "none"

class MotorMovement(BaseModel):
    pos: int
    index: int

DEBUG_POSITION = 1000
DEBUG_SINGLE_MOTOR = False

MOTOR_TYPE = MotorType.TECHNOSOFT
VISION_TYPE = VisionType.MEDIAPIPE
SIDE_CAMERA_DEBUG = False
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
    cycle_counter: int = 0  # Counts full movement cycles

    stiffness_value: int = 0 # Object stiffness value as read from the configuration
    pair_index: int = 0 # 0 or 1

    def model_post_init(self, __context) -> None:
        self.reset()

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.is_pinched = False
        self.progress = 0.0
        self.cycle_counter = 0
        self.stiffness_value = 0
        self.pair_index = 0

class Experiment:
    def __init__(self,
        config: Configuration,
        path: Path,
        pair_finger: PairFinger,
        top_width: int = TOP_WIDTH,
        top_height: int = TOP_HEIGHT,
        side_width: int = SIDE_WIDTH,
        side_height: int = SIDE_HEIGHT,
        camera_fps: int = 30,
        side_camera_location: Literal["top", "bottom", "left", "right"] = "bottom",
        server_address: str = "localhost",
        frontend_port: int = PYGAME_PORT,
        backend_port: int = BACKEND_PORT,
        technosoft_port: int = TECHNOSOFT_PORT
    ):
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
        self._state = ExperimentState.COMPARISON
        self._pause_time = 0
        self._pair_counter = 0
        self._pair_finger: PairFinger = pair_finger
        self._top_frame_queue = Queue(maxsize=30)
        self._side_frame_queue = Queue(maxsize=30)
        self.CENTER_THRESHOLD = 20
        self.EDGE_THRESHOLD = 30
        self.last_x = self._top_width/2
        self.last_y = self._top_height/2
        self.reached_edge = False
        self.in_center = True
        self._server_address = server_address
        self._frontend_port = frontend_port
        self._backend_port = backend_port
        self._technosoft_port = technosoft_port
        self._frontend_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._technosoft_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print("Waiting for connection")
        self._listening_socket.bind((self._server_address, self._backend_port))
        self._listening_socket.listen()
        self._control_socket, _ = self._listening_socket.accept()
        print("Connection accepted")

        self._virtual_object = VirtualObject(
            original_x=self._top_width/2,
            original_y=self._top_height/2,
        )
        self._running = True

        self.TOP_CAMERA = 0
        self.SIDE_CAMERA = 1

        self._top_writer = None
        self._side_writer = None

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

        self._update_active_finger(False)

        self._top_recording_thread = threading.Thread(target=self._record_top_frames, daemon=True)
        self._top_recording_thread.start()
        if not SIDE_CAMERA_DEBUG:
            self._side_recording_thread = threading.Thread(target=self._record_side_frames, daemon=True)
            self._side_recording_thread.start()

        self._experiment_thread = threading.Thread(target=self._experiment_loop, daemon=True)
        self._experiment_thread.start()

        self._frontend_network_thread = threading.Thread(target=self._frontend_network_loop, daemon=True)
        self._frontend_network_thread.start()

        match MOTOR_TYPE:
            case MotorType.ARDUINO:
                try:
                    self._arduino = serial.Serial('COM3', 115200)
                except Exception as e:
                    print(f"Failed to connect to Arduino - serial control disabled. Error: {e}")
                    self._arduino = None
                    
                self._arduino_thread = threading.Thread(target=self._arduino_control_loop, daemon=True)
                self._arduino_thread.start()
            case MotorType.TECHNOSOFT:
                self._technosoft_thread = threading.Thread(target=self._technosoft_control_loop, daemon=True)
                self._technosoft_thread.start()
            case MotorType.NONE:
                ...
            case _: raise NotImplementedError("Unknown motor type")

        self._top_thread = threading.Thread(target=self.start_top_camera, daemon=True)
        self._top_thread.start()

        if not SIDE_CAMERA_DEBUG:
            self._side_thread = threading.Thread(target=self.start_side_camera, daemon=True)
            self._side_thread.start()

    def _get_pair_path(self) -> Path:
        return self._path / f"pair_{self._pair_counter:03d}"

    def _create_pair_folder(self) -> None:
        pair_path = self._get_pair_path()
        pair_path.mkdir()
        
    def _initialize_writers(self) -> None:
        if RECORDING_DATA:
            pair_path = self._get_pair_path()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
            self._top_writer = cv2.VideoWriter(str(pair_path / 'top_camera.mp4'), fourcc, self._camera_fps, (self._top_width, self._top_height))
            if not SIDE_CAMERA_DEBUG:
                self._side_writer = cv2.VideoWriter(str(pair_path / 'side_camera.mp4'), fourcc, self._camera_fps, (self._top_width, self._top_height))

    def _setup_pair(self) -> None:
        self._create_pair_folder()
        self._initialize_writers()

    def _cleanup_writers(self):
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
        if self._current_position is None:
            return

        finger_midpoint_x = (self._current_position.thumb_x + self._current_position.active_finger_x) / 2
        finger_midpoint_y = (self._current_position.thumb_y + self._current_position.active_finger_y) / 2

        distance_to_object = math.sqrt(
            (finger_midpoint_x - self._virtual_object.x)**2 + 
            (finger_midpoint_y - self._virtual_object.y)**2
        )
        
        if self._is_pinching and distance_to_object < self._virtual_object.size * 0.5:
            self._virtual_object.is_pinched = True
        elif not self._is_pinching:
            self._virtual_object.is_pinched = False

        if self._virtual_object.is_pinched:
            self._virtual_object.x = finger_midpoint_x
            self._virtual_object.y = finger_midpoint_y
            self._update_movement_progress()
        else:
            self._virtual_object.x = self._virtual_object.original_x
            self._virtual_object.y = self._virtual_object.original_y
            self._virtual_object.progress = 0.0

        self._virtual_object.stiffness_value = stiffness_value
        self._virtual_object.pair_index = pair_index

    def _check_comparison_end(self) -> bool:
        return self._virtual_object.cycle_counter == TARGET_CYCLE_COUNT

    def _reset_comparison(self):
        self._is_pinching = False
        self._virtual_object.reset()

    def _get_finger_name(self, is_index: bool) -> Literal["index"] | PairFinger:
        return "index" if is_index else self._pair_finger

    def _update_active_finger(self, is_index: bool):
        self._active_finger = self._get_finger_name(is_index)
        self._vision.set_active_finger(self._active_finger)
        print(f"Active finger set to {self._active_finger}")

    def _experiment_loop(self):
        for pair in self._config.pairs:
            if not self._running:
                print("Exit via Ctrl-C")
                break

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

                self._cleanup_writers()

                if not self._running:
                    print("Exit via Ctrl-C")
                    break

                print("Question")
                question_timestamp = datetime.now()
                self._state = ExperimentState.QUESTION
                answer_data = self._control_socket.recv(1024)
                answer = ExperimentControl.model_validate_json(answer_data)

                answers_file = self._path / 'answers.csv'

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

    def _arduino_control_loop(self):
        MOVEMENT_THRESHOLD = 5.0
        steady_flag = True
        
        while self._running and self._arduino:
            if self._virtual_object.is_pinched:
                steady_flag = False
                obj_x = self._virtual_object.x - self._virtual_object.original_x
                obj_y = self._virtual_object.y - self._virtual_object.original_y
                
                stiffness_value = self._virtual_object.stiffness_value
                finger_code = self._active_finger[0].upper()
                message = 'Z'
                if abs(obj_x) > MOVEMENT_THRESHOLD or abs(obj_y) > MOVEMENT_THRESHOLD:
                    if abs(obj_x) > abs(obj_y):
                        if obj_x < 0:
                            self._arduino.write(f'{finger_code}L{stiffness_value}'.encode())
                        else:
                            self._arduino.write(f'{finger_code}R{stiffness_value}'.encode())
                    else:
                        if obj_y < 0:
                            self._arduino.write(f'{finger_code}U{stiffness_value}'.encode())
                        else:
                            self._arduino.write(f'{finger_code}D{stiffness_value}'.encode())
            elif not steady_flag:
                self._arduino.write(b'S')
                steady_flag = True
            self._sleep_frontend()

    def _build_message(self, motors: list[MotorMovement]) -> str:
        message = "Z"
        for motor in motors:
            message += f"M{motor.index}P{motor.pos}"
        message += "F"
        return message
        
    def _get_base_index(self, finger_idx: Literal[0, 1]) -> Literal[0, 3]:
        return finger_idx * 3

    def _calculate_motor_movements(self, finger_id: FingerId, direction: str, distance: float, motor_spacing: float = 1000.0) -> list[MotorMovement]:
        direction = direction.lower()
        base_motor_idx = self._get_base_index(finger_id.value)
        h = motor_spacing * math.sqrt(3) / 3

        motor_positions = [
            (0, 2 * h / 3),
            (-motor_spacing / 2, -h / 3),
            (motor_spacing / 2, -h / 3)
        ]
        object_start = (0, 0)
        direction_vectors = {
            "up": (0, distance),
            "down": (0, -distance),
            "left": (-distance, 0),
            "right": (distance, 0)
        }
        if direction not in direction_vectors:
            raise ValueError(f"Invalid direction: {direction}. Must be one of: up, down, left, right")
        dx, dy = direction_vectors[direction]
        object_end = (object_start[0] + dx, object_start[1] + dy)
        movements = []
        for i, (mx, my) in enumerate(motor_positions):
            initial_length = math.sqrt((mx - object_start[0])**2 + (my - object_start[1])**2)
            final_length = math.sqrt((mx - object_end[0])**2 + (my - object_end[1])**2)
            delta_length = final_length - initial_length
            movements.append(MotorMovement(pos=int(delta_length), index=i+base_motor_idx))
        return movements

    def _technosoft_control_loop(self):
        is_comparison = True
        
        while self._running:
            if self._virtual_object.is_pinched:
                obj_x = self._virtual_object.x - self._virtual_object.original_x
                obj_y = self._virtual_object.y - self._virtual_object.original_y

                finger_id = FingerId.INDEX if self._active_finger == "index" else FingerId.OTHER
                stiffness_value = self._virtual_object.stiffness_value
                motors = []
                if self._state != ExperimentState.COMPARISON:
                    if is_comparison:
                        is_comparison = False
                        motors.extend([
                            MotorMovement(pos=0, index=self._get_base_index(finger_id.value)),
                            MotorMovement(pos=0, index=self._get_base_index(finger_id.value)+1),
                            MotorMovement(pos=0, index=self._get_base_index(finger_id.value)+2),
                        ])
                else:
                    is_comparison = True
                    if abs(obj_x) > abs(obj_y):
                        if obj_x < 0:
                            print(f"Moving left by {obj_x}")
                            motors.extend(self._calculate_motor_movements(
                                finger_id=finger_id,
                                direction="left",
                                distance = abs(obj_x)
                            ))
                        else:
                            print(f"Moving right by {obj_x}")    
                            motors.extend(self._calculate_motor_movements(
                                finger_id=finger_id,
                                direction="right",
                                distance = abs(obj_x)
                            ))
                    else:
                        if obj_y < 0:
                            print(f"Moving up by {obj_y}")    
                            motors.extend(self._calculate_motor_movements(
                                finger_id=finger_id,
                                direction="up",
                                distance = abs(obj_y)
                            ))
                        else:
                            print(f"Moving down by {obj_y}")    
                            motors.extend(self._calculate_motor_movements(
                                finger_id=finger_id,
                                direction="down",
                                distance = abs(obj_y)
                            ))

                if motors:
                    if DEBUG_SINGLE_MOTOR:
                        motors = [motors[0]]
                    message = self._build_message(motors)
                    self._technosoft_socket.sendto(message.encode("utf-8"), (self._server_address, self._technosoft_port))
            self._sleep_motors()

    def _frontend_network_loop(self):
        while self._running:
            current_pos = self.get_hand_position()
            if current_pos:
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
                        cycleCount=self._virtual_object.cycle_counter,
                        pairIndex=self._virtual_object.pair_index
                    )
                )
                try:
                    self._frontend_socket.sendto(packet.model_dump_json().encode("utf-8"), (self._server_address, self._frontend_port))
                except queue.Empty:
                    ...
                except Exception as e:
                    print(f"Failed to send UDP data: {e}")
            self._sleep_frontend()

    def _calculate_pinch_threshold(self, depth: float) -> float:
        return self.BASE_PINCH_THRESHOLD * (depth)
    
    def _configure_camera(self, cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, self._camera_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._top_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._top_height)

    def _update_movement_progress(self):
        if not self._virtual_object.is_pinched:
            self._virtual_object.progress = 0.0
            self.reached_edge = False
            self.in_center = True
            return

        center_x = self._top_width / 2
        center_y = self._top_height / 2
        distance_from_center = math.sqrt(
            (self._virtual_object.x - center_x)**2 + 
            (self._virtual_object.y - center_y)**2
        )
        if distance_from_center < self.CENTER_THRESHOLD:
            if not self.in_center and self.reached_edge:
                self._virtual_object.cycle_counter += 1
                self._virtual_object.progress = 0.0
                self.reached_edge = False
            self.in_center = True
        else:
            self.in_center = False
        max_distance = min(self._top_width/2, self._top_height/2) - self.EDGE_THRESHOLD
        if distance_from_center > max_distance:
            self.reached_edge = True
        if not self.reached_edge:
            self._virtual_object.progress = min(distance_from_center / max_distance, 1.0)
            last_distance = math.sqrt(
                (self.last_x - center_x)**2 + 
                (self.last_y - center_y)**2
            )
            if distance_from_center < last_distance:
                self._virtual_object.progress = max(0.0, self._virtual_object.progress - 0.1)
        self.last_x = self._virtual_object.x
        self.last_y = self._virtual_object.y

    def _record_top_frames(self):
        while self._running:
            if self._top_writer:
                try:
                    data: tuple[np.ndarray, HandPosition] = self._top_frame_queue.get(timeout=1)
                    frame, hand_position = data
                    self._top_writer.write(frame)
                    pair_path = self._get_pair_path()
                    tracking_file = pair_path / 'tracking.csv'
                    position_dict = hand_position.model_dump()
                    if not tracking_file.exists():
                        headers = ['timestamp', 'pinching', 'stiffness', 'object_x', 'object_y', 'finger', *list(position_dict.keys())]
                        with open(tracking_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)
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
            flipped_frame = cv2.flip(frame, 0)

            detected = self._process_top_view(flipped_frame)
            
            # Add frame to recording queue if recording (should save original frame, not flipped)
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
        sleep(3)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            self._detect_pinch(frame)
            
            if self._side_writer:
                try:
                    self._side_frame_queue.put_nowait(frame)
                except:
                    pass
                
            cv2.imshow("Side Camera", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def get_hand_position(self, with_lock = True) -> Optional[HandPosition]:
        if with_lock:
            with self._hand_position_lock:
                return self._current_position
        return self._current_position

    def is_pinching(self) -> bool:
        return self._is_pinching

    def _toggle_pinch(self):
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
        self._running = False
        keyboard.unhook_all()
        cv2.destroyAllWindows()
        self._frontend_socket.close()
        self._listening_socket.close()
        self._control_socket.close()
        self._cleanup_writers()
        self._vision.cleanup()

        match MOTOR_TYPE:
            case MotorType.ARDUINO:
                self._arduino.close()
            case MotorType.TECHNOSOFT:
                self._technosoft_socket.close()
            case MotorType.NONE:
                ...
            case _: raise NotImplementedError("Unknown motor type")

def start_experiment(config: Configuration, path: Path, pair_finger: PairFinger):
    experiment = Experiment(config, path, pair_finger)
    return experiment

def get_pair_finger() -> PairFinger:
    while True:
        user_input = input("Please enter M or R: ").upper()
        if user_input in ['M', 'R']:
            break
        print("Invalid input. Please enter either M or R.")
    return "middle" if user_input == "M" else "ring"

def create_experiment_folder(config: Configuration, pair_finger: PairFinger):
    experiment_path = Path(experiment_folder)
    if not experiment_path.exists():
        experiment_path.mkdir()
    existing_folders = [f for f in experiment_path.iterdir() if f.is_dir()]
    if not existing_folders:
        folder_id = 1
    else:
        folder_id = max(int(folder.name.split('_')[-1]) for folder in existing_folders) + 1
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_path = experiment_path / f"{timestamp}_{pair_finger}_{folder_id:03d}"
    folder_path.mkdir()
    config.write_configuration(folder_path / "configuration.csv")
    return folder_path

def main():
    config = Configuration.read_configuration('configuration.csv')
    pair_finger = get_pair_finger()
    path = create_experiment_folder(config, pair_finger)
    experiment = start_experiment(config, path, pair_finger)
    try:
        while True:
            sleep(0.1)
    except KeyboardInterrupt:
        experiment.cleanup()

if __name__ == "__main__":
    main()
