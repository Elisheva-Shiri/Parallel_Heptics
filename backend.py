import csv
import builtins
from datetime import datetime
import json
from pathlib import Path
import math
import signal
from time import sleep
import cv2
import threading
from typing import Literal, Optional
from threading import Lock
import numpy as np
import socket
from typing import Annotated
import typer
from pydantic import BaseModel
from structures import ControlAction, ExperimentControl, ExperimentState, FingerPosition, QuestionInput, StateData, TrackingObject, ExperimentPacket
from consts import BACKEND_PORT, PAUSE_SLEEP_SECONDS, MOTORS_COMMUNICATION_RATE, PYGAME_PORT, TARGET_CYCLE_COUNT, HARDWARE_PORT, TOP_HEIGHT, TOP_WIDTH, SIDE_HEIGHT, SIDE_WIDTH, FRONTEND_FPS, VIRTUAL_OBJECT_FPS, FINGER_NAMES, FingerName, CENTER_THRESHOLD, EDGE_THRESHOLD, TAPPING_HEIGHT_RATIO, STIFFNESS_MAX
import queue
from queue import Queue
from enum import StrEnum
from vision import ColorVision, HandPosition, MediapipeVision, YoloVision
from motor_controller import HandOrientation, MotorSetId, MovementStrategy, MotorController
from haptic_mapping import map_object_displacement_to_tactor


DEBUG_SINGLE_MOTOR = False
DEBUG_FLIP_Y = True
IS_DEBUG = True
DEBUG_LOG_MOTOR_RANGE = True  # Log min/max motor positions on shutdown (no per-message overhead)

original_print = builtins.print
PROTOCOLS_FOLDER = Path(__file__).resolve().parent / "protocols"


def set_is_debug(enabled: bool) -> None:
    global IS_DEBUG
    IS_DEBUG = enabled


def debug_print(*args, **kwargs):
    if IS_DEBUG:
        original_print(*args, **kwargs)


builtins.print = debug_print
print = debug_print

class VisionType(StrEnum):
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"
    FRAME_COLOR = "frame_color"
    OPTICAL_COLOR_NO_GPU = "optical_color_no_gpu"
    OPTICAL_COLOR_GPU = "optical_color_gpu"

class RunMode(StrEnum):
    COMPARISON = "comparison"
    SINGLE_FINGER = "single_finger"

class MotorType(StrEnum):
    HARDWARE = "hardware"
    NONE = "none"

# for free form 3
MOVE_FACTOR = 7
MOTOR_TYPE = MotorType.HARDWARE
MOVEMENT_STRATEGY = MovementStrategy.IK
MOTOR_OPPOSES_OBJECT_MOTION = True
RECORDING_DATA = True

ANSWERS_HEADER = [
    'timestamp',
    'pair_number',
    'object_1_finger',
    'object_1_stiffness',
    'object_2_finger',
    'object_2_stiffness',
    'time_to_answer',
    'answer',
]


# Create experiment folder if it doesn't exist
REAL_EXPERIMENTS_FOLDER = Path("live_experiments")
DEBUG_EXPERIMENTS_FOLDER = Path("debug_experiments")
experiment_folder = DEBUG_EXPERIMENTS_FOLDER
if not experiment_folder.exists():
    experiment_folder.mkdir()    

class StiffnessValue(BaseModel):
    value: int
    finger_id: int

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
                    finger_id=int(row[1])
                ),
                second=StiffnessValue(
                    value=int(row[2]),
                    finger_id=int(row[3])
                )
                ) for row in csv_reader
            ])
    
    def write_configuration(self, path: str):
        """Write configuration pairs to CSV file"""
        with open(path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for pair in self.pairs:
                csv_writer.writerow([pair.first.value, pair.first.finger_id, pair.second.value, pair.second.finger_id])

class VirtualObject(BaseModel):
    x: float = 0.0
    y: float = 0.0
    size: float = 40.0  # Size of cube sides
    original_x: float = 0.0  # Center of plane
    original_y: float = 0.0  # Center of plane
    is_interacting: bool = False

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
        self.is_interacting = False

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
        run_mode: Literal["comparison", "single_finger"] = "comparison",
        motor_set_id: MotorSetId = MotorSetId.MOTORS_0_2,
        hand_orientation: HandOrientation = HandOrientation.NOT_MIRRORED,
        movement_strategy: MovementStrategy = MOVEMENT_STRATEGY,
        target_cycle_count: int = TARGET_CYCLE_COUNT,
        vision_type: VisionType = VisionType.MEDIAPIPE,
        tapping_enabled: bool = False,
        finger_colors: dict[str, str] | None = None,
        side_camera_off: bool = False,
        top_width: int = TOP_WIDTH,
        top_height: int = TOP_HEIGHT,
        side_width: int = SIDE_WIDTH,
        side_height: int = SIDE_HEIGHT,
        camera_fps: int = 30,
        side_camera_location: Literal["top", "bottom", "left", "right"] = "bottom",
        server_address: str = "localhost",
        frontend_port: int = PYGAME_PORT,
        backend_port: int = BACKEND_PORT,
        hardware_port: int = HARDWARE_PORT,
        play_white_noise: bool = False,
        is_debug: bool = True,
        resume_pair_number: Optional[int] = None,
    ):

        # SETUP
        self._hand_position_lock = Lock()
        self._current_position: Optional[HandPosition] = None
        self._is_interacting = False
        self._config = config
        self._path = path
        self._run_mode = run_mode
        self._resume_pair_number = resume_pair_number
        self._top_width = top_width
        self._top_height = top_height
        self._side_width = side_width
        self._side_height = side_height
        self._virtual_object_fps = VIRTUAL_OBJECT_FPS
        self._frontend_fps = FRONTEND_FPS
        self._motors_communication_rate = MOTORS_COMMUNICATION_RATE
        self._side_camera_location: Literal["top", "bottom", "left", "right"] = side_camera_location
        self._camera_fps = camera_fps
        self._vision_type = vision_type
        self._tapping_enabled = tapping_enabled
        self._finger_colors = finger_colors or {}
        self._side_camera_off = side_camera_off
        self._manual_interaction_enabled = side_camera_off
        # TODO - add Start screen if needed,
        # and adjust to ExperimentState.START
        self._state = ExperimentState.COMPARISON
        self._pause_time = 0
        self._break_started_at: Optional[datetime] = None
        self._pair_counter = 0  # Counter for pair folders
        self._motor_set_id = motor_set_id
        self._target_cycle_count = target_cycle_count
        self._hand_orientation = hand_orientation
        self._movement_strategy = movement_strategy
        self._play_white_noise = play_white_noise
        self._is_debug = is_debug
        self._state_lock = Lock()
        self._cleanup_lock = Lock()
        self._cleaned_up = False
        self._control_queue: Queue[ExperimentControl] = Queue()
        self._client_threads: list[threading.Thread] = []
        self._moderator_pause_started_at: Optional[datetime] = None
        self._moderator_pause_state_when_started: Optional[ExperimentState] = None
        self._camera_lock = Lock()
        self._top_camera_capture: cv2.VideoCapture | None = None
        self._side_camera_capture: cv2.VideoCapture | None = None

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
        self._require_release = False  # Require release before next cycle
        
        # Motor Controller
        self._motor_controller = MotorController(
            movement_strategy=self._movement_strategy,
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

        # Create virtual object at center of screen
        self._virtual_object = VirtualObject(
            original_x=self._top_width/2,
            original_y=self._top_height/2,
        )
        
        # Bind sockets. Control clients are accepted in the background so the
        # moderator utility can connect independently from pygame/Unity.
        self._listening_socket.bind((self._server_address, self._backend_port))
        self._listening_socket.listen()
        self._listening_socket.settimeout(0.5)
        self._running = True

        # Camera indices
        self.TOP_CAMERA = 0
        self.SIDE_CAMERA = 1
        
        # Video writers
        self._top_writer = None
        self._side_writer = None
        
        # Threshold for interaction detection (adjust as needed)
        self.BASE_INTERACTION_THRESHOLD = 50  # pixels

        match self._vision_type:
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
                    base_interaction_threshold=self.BASE_INTERACTION_THRESHOLD
                )
            case VisionType.YOLO:
                self._vision = YoloVision(
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    base_interaction_threshold=self.BASE_INTERACTION_THRESHOLD
                )
            case VisionType.FRAME_COLOR:
                self._vision = ColorVision(
                    finger_colors=self._finger_colors,
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    tracking_method="frame",
                    fps=self._camera_fps,
                    base_interaction_threshold=self.BASE_INTERACTION_THRESHOLD
                )
            case VisionType.OPTICAL_COLOR_NO_GPU:
                self._vision = ColorVision(
                    finger_colors=self._finger_colors,
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    tracking_method="optical_flow",
                    fps=self._camera_fps,
                    base_interaction_threshold=self.BASE_INTERACTION_THRESHOLD,
                    use_gpu=False
                )
            case VisionType.OPTICAL_COLOR_GPU:
                self._vision = ColorVision(
                    finger_colors=self._finger_colors,
                    top_width=self._top_width,
                    top_height=self._top_height,
                    side_width=self._side_width,
                    side_height=self._side_height,
                    tracking_method="optical_flow",
                    fps=self._camera_fps,
                    base_interaction_threshold=self.BASE_INTERACTION_THRESHOLD,
                    use_gpu=True
                )
            case _:
                raise NotImplementedError("Unknown CV type")  # Should never happen

        # * Initialization only until experiment loop begins
        # * Must run after self._vision is initialized
        self._update_active_finger(self._get_first_configured_finger_id())

        # Start recording threads
        self._top_recording_thread = threading.Thread(target=self._record_top_frames, daemon=True)
        self._top_recording_thread.start()
        
        if not self._side_camera_off:
            self._side_recording_thread = threading.Thread(target=self._record_side_frames, daemon=True)
            self._side_recording_thread.start()

        # Start TCP control server for frontend question answers and moderator commands.
        self._control_thread = threading.Thread(target=self._control_server_loop, daemon=True)
        self._control_thread.start()

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
            case _:
                raise NotImplementedError("Unknown motor type")  # Should never happen

        self._top_thread = threading.Thread(target=self.start_top_camera, daemon=True)
        self._top_thread.start()

        if not self._side_camera_off:
            self._side_thread = threading.Thread(target=self.start_side_camera, daemon=True)
            self._side_thread.start()

    def _get_pair_path(self) -> Path:
        """Get the path for the current pair of stiffness values"""
        return self._path / f"pair_{self._pair_counter:03d}"

    def _create_pair_folder(self) -> None:
        """Create a folder for the current pair of stiffness values"""
        pair_path = self._get_pair_path()
        pair_path.mkdir(exist_ok=True)

    def _next_video_filenames(self) -> tuple[Path, Path]:
        """Return (top, side) video paths, picking unique names if files already exist."""
        pair_path = self._get_pair_path()
        top = pair_path / "top_camera.mp4"
        side = pair_path / "side_camera.mp4"
        if not top.exists() and not side.exists():
            return top, side

        index = 1
        while True:
            candidate_top = pair_path / f"top_camera_resume_{index:03d}.mp4"
            candidate_side = pair_path / f"side_camera_resume_{index:03d}.mp4"
            if not candidate_top.exists() and not candidate_side.exists():
                return candidate_top, candidate_side
            index += 1

    def _initialize_writers(self) -> None:
        # Initialize video writers for this pair
        if RECORDING_DATA:
            top_path, side_path = self._next_video_filenames()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._top_writer = cv2.VideoWriter(str(top_path), fourcc, self._camera_fps, (self._top_width, self._top_height))
            if not self._side_camera_off:
                self._side_writer = cv2.VideoWriter(str(side_path), fourcc, self._camera_fps, (self._top_width, self._top_height))

    def _setup_pair(self) -> None:
        """ Setup everything needed for a new pair """
        self._create_pair_folder()
        self._initialize_writers()

    @staticmethod
    def _is_end_marker(pair: StiffnessPair) -> bool:
        return pair.first.value == -1

    @staticmethod
    def _is_break_marker(pair: StiffnessPair) -> bool:
        return pair.first.value == -2

    @staticmethod
    def _is_pause_marker(pair: StiffnessPair) -> bool:
        return pair.first.value == 0

    @classmethod
    def _is_comparison_pair(cls, pair: StiffnessPair) -> bool:
        return not (cls._is_end_marker(pair) or cls._is_break_marker(pair) or cls._is_pause_marker(pair))

    def _config_index_for_pair_number(self, pair_number: int) -> int:
        """Return the config row index of the Nth comparison pair (1-based)."""
        if pair_number <= 0:
            raise ValueError(f"pair_number must be >= 1, got {pair_number}")
        seen = 0
        for i, pair in enumerate(self._config.pairs):
            if self._is_comparison_pair(pair):
                seen += 1
                if seen == pair_number:
                    return i
        raise ValueError(
            f"configuration.csv only has {seen} comparison pair(s); cannot resume at pair {pair_number}"
        )

    def _cleanup_writers(self):
        """Clean up video writers after pair is complete"""
        if self._top_writer:
            self._top_writer.release()
            self._top_writer = None
        if self._side_writer:
            self._side_writer.release() 
            self._side_writer = None

    def is_running(self) -> bool:
        """Return whether the experiment should keep its worker loops alive."""
        return self._running

    def request_shutdown(self) -> None:
        """Ask all worker loops to stop and unblock camera reads as soon as possible."""
        self._running = False
        self._release_camera_captures()

    def _register_camera_capture(self, camera_name: Literal["top", "side"], cap: cv2.VideoCapture) -> None:
        with self._camera_lock:
            if camera_name == "top":
                self._top_camera_capture = cap
            elif camera_name == "side":
                self._side_camera_capture = cap

    def _clear_camera_capture(self, camera_name: Literal["top", "side"], cap: cv2.VideoCapture) -> None:
        with self._camera_lock:
            if camera_name == "top" and self._top_camera_capture is cap:
                self._top_camera_capture = None
            elif camera_name == "side" and self._side_camera_capture is cap:
                self._side_camera_capture = None

    def _release_camera_captures(self) -> None:
        with self._camera_lock:
            captures = [self._top_camera_capture, self._side_camera_capture]
            self._top_camera_capture = None
            self._side_camera_capture = None

        for cap in captures:
            if cap is None:
                continue
            try:
                cap.release()
            except Exception as e:
                print(f"Failed to release camera capture: {e}")

    @staticmethod
    def _safe_destroy_window(window_name: str) -> None:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            # The window may already be gone, especially during Ctrl-C cleanup.
            pass

    @staticmethod
    def _safe_destroy_all_windows() -> None:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Headless OpenCV builds or already-destroyed windows can raise here.
            pass

    @staticmethod
    def _safe_close_socket(sock: socket.socket) -> None:
        try:
            sock.close()
        except OSError:
            pass

    @staticmethod
    def _join_thread(thread: threading.Thread | None, timeout: float = 1.0) -> None:
        if thread is None or thread is threading.current_thread() or not thread.is_alive():
            return
        thread.join(timeout=timeout)

    def _sleep_virtual_object(self):
        sleep(1.0 / self._virtual_object_fps)

    def _sleep_frontend(self):
        sleep(1.0 / self._frontend_fps)

    def _sleep_motors(self):
        sleep(1.0 / self._motors_communication_rate)

    def _update_virtual_object(self, stiffness_value: int, pair_index: int) -> None:
        """Update virtual object position based on hand position and interaction/tap state"""
        if self._current_position is None:
            return

        if self._tapping_enabled:
            contact_x = self._current_position.active_finger_x
            contact_y = self._current_position.active_finger_y
        else:
            contact_x = (self._current_position.thumb_x + self._current_position.active_finger_x) / 2
            contact_y = (self._current_position.thumb_y + self._current_position.active_finger_y) / 2
        
        distance_to_object = math.sqrt(
            (contact_x - self._virtual_object.x)**2 + 
            (contact_y - self._virtual_object.y)**2
        )
        
        if self._require_release:
            self._virtual_object.is_interacting = False
            if not self._is_interacting:
                self._require_release = False
        elif self._is_interacting and distance_to_object < self._virtual_object.size * 0.5:
            self._virtual_object.is_interacting = True
        elif not self._is_interacting:
            self._virtual_object.is_interacting = False
            
        if self._virtual_object.is_interacting:
            self._virtual_object.x = contact_x
            self._virtual_object.y = contact_y
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
        self._is_interacting = False
        self._require_release = False
        self._virtual_object.reset()

    def _get_finger_name(self, finger_id: int) -> FingerName:
        if finger_id not in FINGER_NAMES:
            raise ValueError(f"Invalid finger_id: {finger_id}. Must be 0-4.")
        return FINGER_NAMES[finger_id]  # type: ignore

    def _get_finger_id(self, finger_name: FingerName) -> int:
        for fid, fname in FINGER_NAMES.items():
            if fname == finger_name:
                return fid
        raise ValueError(f"Invalid finger_name: {finger_name}.")

    def _get_first_configured_finger_id(self) -> int:
        for pair in self._config.pairs:
            for configured in (pair.first, pair.second):
                if configured.value > 0:
                    return configured.finger_id
        return self._get_finger_id("index")

    def _update_active_finger(self, finger_id: int):
        self._active_finger = self._get_finger_name(finger_id)
        self._vision.set_active_finger(self._active_finger)
        print(f"Active finger set to {self._active_finger}")

    def _run_pair_object(self, stiffness_value: int, pair_index: int, finger_id: int):
        """Run one object within a pair (shared by both modes)."""
        self._reset_comparison()
        self._update_active_finger(finger_id)

        while self._running and not self._check_comparison_end():
            self._wait_while_moderator_paused()
            self._update_virtual_object(stiffness_value, pair_index)
            self._sleep_virtual_object()

    def _control_server_loop(self) -> None:
        """Accept frontend/moderator TCP control clients."""
        print(f"Listening for control commands on {self._server_address}:{self._backend_port}")
        while self._running:
            try:
                client, address = self._listening_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            print(f"Control client connected: {address}")
            thread = threading.Thread(
                target=self._control_client_loop,
                args=(client, address),
                daemon=True,
            )
            self._client_threads.append(thread)
            thread.start()

    def _control_client_loop(self, client: socket.socket, address) -> None:
        """Read ExperimentControl JSON messages from one TCP client."""
        buffer = b""
        with client:
            while self._running:
                try:
                    chunk = client.recv(4096)
                except OSError:
                    break

                if not chunk:
                    break

                buffer += chunk
                messages, buffer = self._extract_control_messages(buffer)
                for message in messages:
                    response = self._handle_control_message(message)
                    try:
                        client.sendall((json.dumps(response) + "\n").encode("utf-8"))
                    except OSError:
                        break

        print(f"Control client disconnected: {address}")

    def _extract_control_messages(self, buffer: bytes) -> tuple[list[ExperimentControl], bytes]:
        """Parse newline-delimited controls, with legacy single-JSON fallback."""
        messages: list[ExperimentControl] = []

        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            raw = raw.strip()
            if raw:
                try:
                    messages.append(ExperimentControl.model_validate_json(raw))
                except Exception as ex:
                    print(f"Ignoring malformed control message: {ex}")

        # Existing pygame/Unity clients send one JSON object without a newline.
        if buffer.strip():
            try:
                messages.append(ExperimentControl.model_validate_json(buffer.strip()))
                buffer = b""
            except Exception:
                # Keep partial data for the next recv.
                pass

        return messages, buffer

    def _handle_control_message(self, control: ExperimentControl) -> dict[str, object]:
        """Apply immediate moderator controls or enqueue state-specific answers."""
        if control.moderatorAction is not None:
            return self._handle_moderator_action(control.moderatorAction)

        valid_answers = {QuestionInput.LEFT.value, QuestionInput.RIGHT.value}
        if control.questionInput in valid_answers:
            if self._is_moderator_paused():
                return {"ok": False, "message": "Ignored question answer while moderator pause is active"}
            if self._get_state() != ExperimentState.QUESTION:
                return {"ok": False, "message": f"Ignored question answer while state is {self._get_state().name}"}

            self._control_queue.put(control)
            return {"ok": True, "message": f"Queued question answer {control.questionInput}"}

        return {"ok": False, "message": f"Ignored unknown control message: {control.model_dump()}"}

    def _handle_moderator_action(self, action: ControlAction) -> dict[str, object]:
        if action == ControlAction.TOGGLE_INTERACTION:
            if self._is_moderator_paused():
                return {"ok": False, "message": "Ignored toggle_interaction while moderator pause is active"}
            if self._get_state() != ExperimentState.COMPARISON:
                return {"ok": False, "message": f"Ignored toggle_interaction while state is {self._get_state().name}"}
            if not self._manual_interaction_enabled:
                return {"ok": False, "message": "Ignored toggle_interaction because side-camera/tap detection is active"}

            self._toggle_interaction()
            return {"ok": True, "message": f"Interaction is now {'ON' if self._is_interacting else 'OFF'}"}

        if action == ControlAction.FINISH_BREAK:
            if self._is_moderator_paused():
                return {"ok": False, "message": "Ignored finish_break while moderator pause is active; resume pause first"}
            if self._get_state() != ExperimentState.BREAK:
                return {"ok": False, "message": f"Ignored finish_break while state is {self._get_state().name}"}

            self._control_queue.put(ExperimentControl(moderatorAction=action))
            return {"ok": True, "message": "Queued break finish"}

        if action == ControlAction.TOGGLE_PAUSE:
            return self._toggle_moderator_pause()

        return {"ok": False, "message": f"Ignored unsupported moderator action: {action}"}

    def _get_state(self) -> ExperimentState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: ExperimentState) -> None:
        with self._state_lock:
            self._state = state

    def _is_moderator_paused(self) -> bool:
        with self._state_lock:
            return self._moderator_pause_started_at is not None

    def _toggle_moderator_pause(self) -> dict[str, object]:
        ended_pause: tuple[datetime, datetime, ExperimentState] | None = None
        with self._state_lock:
            now = datetime.now()
            if self._state == ExperimentState.END and self._moderator_pause_started_at is None:
                return {"ok": False, "message": "Ignored pause because experiment has ended"}

            if self._moderator_pause_started_at is None:
                self._moderator_pause_started_at = now
                self._moderator_pause_state_when_started = self._state
                self._is_interacting = False
                self._virtual_object.is_interacting = False
                return {"ok": True, "message": f"Moderator pause started from {self._state.name}"}

            started_at = self._moderator_pause_started_at
            state_when_started = self._moderator_pause_state_when_started or self._state
            self._moderator_pause_started_at = None
            self._moderator_pause_state_when_started = None
            ended_pause = (started_at, now, state_when_started)

        started_at, finished_at, state_when_started = ended_pause
        self._log_moderator_pause(started_at, finished_at, state_when_started)
        return {
            "ok": True,
            "message": (
                "Moderator pause finished after "
                f"{(finished_at - started_at).total_seconds():.3f} seconds"
            ),
        }

    def _wait_while_moderator_paused(self) -> None:
        while self._running and self._is_moderator_paused():
            self._is_interacting = False
            self._virtual_object.is_interacting = False
            sleep(0.05)

    def _log_moderator_pause(
        self,
        started_at: datetime,
        finished_at: datetime,
        state_when_started: ExperimentState,
    ) -> None:
        pauses_file = self._path / "moderator_pauses.csv"
        if not pauses_file.exists():
            with open(pauses_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_start",
                    "timestamp_finish",
                    "pause_duration_seconds",
                    "state_when_started",
                ])

        with open(pauses_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                started_at.isoformat(),
                finished_at.isoformat(),
                f"{(finished_at - started_at).total_seconds():.3f}",
                state_when_started.name,
            ])

    def _wait_for_break_continue(self) -> bool:
        """Wait until the moderator utility sends --toggle-break."""
        while self._running:
            self._wait_while_moderator_paused()
            try:
                control = self._control_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if control.moderatorAction == ControlAction.FINISH_BREAK:
                return True

            if control.questionInput is not None:
                print(f"Ignoring question answer during break: {control}")

        return False

    def _wait_for_question_answer(self) -> ExperimentControl:
        """Wait until the frontend sends a valid left/right question answer."""
        valid_answers = {QuestionInput.LEFT.value, QuestionInput.RIGHT.value}
        while self._running:
            self._wait_while_moderator_paused()
            try:
                answer = self._control_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if answer.questionInput in valid_answers:
                return answer
            print(f"Ignoring non-answer control message during question: {answer}")

        raise RuntimeError("Experiment stopped before question answer was received")

    def _experiment_loop(self):
        start_index = 0
        if self._resume_pair_number is not None:
            start_index = self._config_index_for_pair_number(self._resume_pair_number)
            self._pair_counter = self._resume_pair_number - 1
            print(f"Continuing experiment: starting from pair {self._resume_pair_number}.")

        for config_index in range(start_index, len(self._config.pairs)):
            pair = self._config.pairs[config_index]
            self._wait_while_moderator_paused()
            if not self._running:
                print("Exit via Ctrl-C")
                break

            if self._is_end_marker(pair):
                if pair.second.value != -1:
                    raise Exception("Misaligned numbers, invalidated configuration/experiment - please contact the staff")

                print("Exit via end of experiment")
                self._set_state(ExperimentState.END)
                break

            elif self._is_break_marker(pair):
                if (
                    pair.second.value != -2
                    or pair.first.finger_id != -2
                    or pair.second.finger_id != -2
                ):
                    raise Exception("Misaligned numbers, invalidated configuration/experiment - please contact the staff")

                print("Break — run moderator_control.py --toggle-break to continue (duration is logged when you continue)")
                self._set_state(ExperimentState.BREAK)
                break_started_at = datetime.now()
                self._break_started_at = break_started_at
                self._pause_time = 0
                self._wait_for_break_continue()
                break_duration = (datetime.now() - break_started_at).total_seconds()
                self._break_started_at = None
                self._pause_time = 0
                self._set_state(ExperimentState.COMPARISON)

                breaks_file = self._path / "breaks.csv"
                if not breaks_file.exists():
                    with open(breaks_file, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["timestamp_start", "break_duration_seconds"])
                with open(breaks_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([break_started_at.isoformat(), f"{break_duration:.3f}"])

            elif self._is_pause_marker(pair):
                if pair.second.value != 0:
                    raise Exception("Misaligned numbers, invalidated configuration/experiment - please contact the staff")

                print("Pausing")
                self._pause_time = PAUSE_SLEEP_SECONDS
                self._set_state(ExperimentState.PAUSE)
                for i in range(PAUSE_SLEEP_SECONDS, 0, -1):
                    self._wait_while_moderator_paused()
                    sleep(1)
                    self._pause_time = i
                    if not self._running:
                        break

                self._pause_time = 0

            else:
                print("Comparing")
                self._pair_counter += 1
                self._setup_pair()
                self._set_state(ExperimentState.COMPARISON)

                self._run_pair_object(pair.first.value, 0, pair.first.finger_id)
                self._run_pair_object(pair.second.value, 1, pair.second.finger_id)

                self._cleanup_writers()

                if not self._running:
                    print("Exit via Ctrl-C")
                    break

                answers_file = self._path / 'answers.csv'
                if _answer_already_recorded(answers_file, self._pair_counter):
                    print(f"Skipping question for pair {self._pair_counter}: answer already recorded")
                    continue

                print("Question")
                question_timestamp = datetime.now()
                self._set_state(ExperimentState.QUESTION)
                answer = self._wait_for_question_answer()

                if not answers_file.exists():
                    with open(answers_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(ANSWERS_HEADER)

                with open(answers_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        question_timestamp.isoformat(),
                        self._pair_counter,
                        self._get_finger_name(pair.first.finger_id),
                        pair.first.value,
                        self._get_finger_name(pair.second.finger_id),
                        pair.second.value,
                        (datetime.now() - question_timestamp).total_seconds(),
                        answer.questionInput])

    def _hardware_control_loop(self):
        """Thread that controls Hardware via UDP based on hand movement."""
        should_reset = True
        og_obj_x: float | None = None
        og_obj_y: float | None = None
        motors_are_displaced = False

        # Position range tracking (zero per-message overhead, summary on shutdown)
        motor_range_min: dict[int, int] = {}
        motor_range_max: dict[int, int] = {}

        def send_motors(motors: list) -> list:
            if not motors:
                return []

            if DEBUG_SINGLE_MOTOR:
                motors = [motors[0]]

            # Track position ranges (dict lookup only, zero logging overhead)
            if DEBUG_LOG_MOTOR_RANGE:
                for motor in motors:
                    if motor.index not in motor_range_min:
                        motor_range_min[motor.index] = motor.pos
                        motor_range_max[motor.index] = motor.pos
                    else:
                        if motor.pos < motor_range_min[motor.index]:
                            motor_range_min[motor.index] = motor.pos
                        if motor.pos > motor_range_max[motor.index]:
                            motor_range_max[motor.index] = motor.pos

            message = self._motor_controller.build_message(motors)
            self._hardware_socket.sendto(message.encode("utf-8"), (self._server_address, self._hardware_port))
            return motors

        while self._running:
            if not self._virtual_object.is_interacting:
                if motors_are_displaced:
                    send_motors(self._motor_controller.zero_motor_positions(self._motor_set_id))
                    motors_are_displaced = False
                    should_reset = True
                    og_obj_x = None
                    og_obj_y = None
                self._sleep_motors()
                continue

            # Collect stiffness value using protection from values greater than STIFFNESS_MAX
            stiffness_value = self._virtual_object.stiffness_value
            if stiffness_value > STIFFNESS_MAX:
                print(f"Stiffness value {stiffness_value} is greater than STIFFNESS_MAX {STIFFNESS_MAX}, setting to {STIFFNESS_MAX}")
                stiffness_value = STIFFNESS_MAX
            stiffness_value_normalized = stiffness_value / STIFFNESS_MAX

            # original x/y != (0,0) since they are half of screen size
            obj_x = self._virtual_object.x - self._virtual_object.original_x
            obj_y = self._virtual_object.y - self._virtual_object.original_y
            if obj_x == og_obj_x and obj_y == og_obj_y:
                self._sleep_motors()
                continue

            og_obj_x = obj_x
            og_obj_y = obj_y
            tactor_x, tactor_y = map_object_displacement_to_tactor(
                obj_x=obj_x,
                obj_y=obj_y,
                oppose_motion=MOTOR_OPPOSES_OBJECT_MOTION,
            )

            motors_enabled = (
                self._get_state() == ExperimentState.COMPARISON
                and not self._is_moderator_paused()
            )
            motors = self._motor_controller.calculate_motor_movements(
                motor_set_id=self._motor_set_id,
                stiffness_value=stiffness_value_normalized,
                obj_x=tactor_x,
                obj_y=tactor_y,
                motors_enabled=motors_enabled,
                reset_to_origin=should_reset and not motors_enabled,
            )
            should_reset = motors_enabled

            if not motors and motors_are_displaced and motors_enabled and obj_x == 0 and obj_y == 0:
                motors = self._motor_controller.zero_motor_positions(self._motor_set_id)

            sent_motors = send_motors(motors)
            if sent_motors:
                motors_are_displaced = motors_enabled and any(motor.pos != 0 for motor in sent_motors)

            self._sleep_motors()

        # Print position range summary on shutdown
        # Uses original_print intentionally to bypass the debug print gate.
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
                if self._tapping_enabled:
                    finger_positions = [
                        FingerPosition(x=current_pos.active_finger_x / self._top_width, z=current_pos.active_finger_y / self._top_height)
                    ]
                else:
                    finger_positions = [
                        FingerPosition(x=current_pos.thumb_x / self._top_width, z=current_pos.thumb_y / self._top_height),
                        FingerPosition(x=current_pos.active_finger_x / self._top_width, z=current_pos.active_finger_y / self._top_height)
                    ]

                state_for_packet = self._get_state()
                pause_for_packet = self._pause_time
                with self._state_lock:
                    moderator_pause_started_at = self._moderator_pause_started_at
                if moderator_pause_started_at is not None:
                    state_for_packet = ExperimentState.MODERATOR_PAUSE
                    pause_for_packet = int(
                        (datetime.now() - moderator_pause_started_at).total_seconds()
                    )
                elif state_for_packet == ExperimentState.BREAK and self._break_started_at is not None:
                    pause_for_packet = int(
                        (datetime.now() - self._break_started_at).total_seconds()
                    )

                packet = ExperimentPacket(
                    stateData=StateData(state=state_for_packet.value, pauseTime=pause_for_packet),
                    landmarks=finger_positions,
                    playWhiteNoise=self._play_white_noise,
                    isDebug=self._is_debug,
                    trackingObject=TrackingObject(
                        x=self._virtual_object.x / self._top_width,
                        z=self._virtual_object.y / self._top_height,
                        size=self._virtual_object.size,
                        isInteracting=self._virtual_object.is_interacting,
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

    def _configure_camera(self, cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, self._camera_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._top_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._top_height)

    def _update_movement_progress(self):
        """Update movement progress based on object position"""
        # ! BUG: Bar does not respond on the way back from edge to center
        # ! Proposed Solution: Half bar to edge in color X, Half bar to centery in color Y

        # ! BUG: If we release after reaching edge, it moves object to center which increases the count by one when interacting again
        # ! Proposed Solution: Track movement back (half bar-half bar), only successful on reaching center by movement (same as solution above)
        if not self._virtual_object.is_interacting:
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
                self._virtual_object.is_interacting = False
                self._require_release = True
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
            # Idle while not recording (e.g. Question/Pause/Break). Without
            # this sleep the loop becomes a CPU busy-spin that holds the GIL,
            # which starves _frontend_network_loop and stutters the frontend.
            if not self._top_writer:
                sleep(0.05)
                continue
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
                    headers = ['timestamp', 'interacting', 'stiffness', 'object_x', 'object_y', 'finger', *list(position_dict.keys())]
                    with open(tracking_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)

                # Append position data
                with open(tracking_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row_data = [
                        datetime.now().isoformat(),
                        self._is_interacting,
                        self._virtual_object.stiffness_value,
                        self._virtual_object.x,
                        self._virtual_object.y,
                        self._active_finger,
                        *list(position_dict.values())
                    ]
                    writer.writerow(row_data)

            except queue.Empty:
                continue
            except Exception as e:
                print(e)

    def _record_side_frames(self):
        """Thread for recording frames from side camera"""
        while self._running:
            # Idle while not recording — see _record_top_frames for rationale.
            if not self._side_writer:
                sleep(0.05)
                continue
            try:
                frame = self._side_frame_queue.get(timeout=1)
                self._side_writer.write(frame)
            except queue.Empty:
                continue
            except Exception:
                pass

    def start_top_camera(self):
        """Process top camera feed to track finger positions"""
        cap = cv2.VideoCapture(self.TOP_CAMERA, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Failed to open top camera")

        self._register_camera_capture("top", cap)
        try:
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
                    except Exception:
                        pass
                    
                cv2.imshow("Top Camera", cv2.flip(frame, 1))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.request_shutdown()
                    break
                
                with self._hand_position_lock:
                    self._current_position = detected
        finally:
            self._clear_camera_capture("top", cap)
            cap.release()
            self._safe_destroy_window("Top Camera")

    def start_side_camera(self):
        """Process side camera feed to detect interaction gestures"""
        RETRIES = 10
        
        cap = cv2.VideoCapture(self.SIDE_CAMERA, cv2.CAP_DSHOW)
        for _ in range(RETRIES-1):
            if cap.isOpened():
                break

            cap.release()
            cap = cv2.VideoCapture(self.SIDE_CAMERA, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            print("Side camera not available - use moderator_control.py --toggle-interaction for manual interaction control")
            self._manual_interaction_enabled = True
            return
        
        self._register_camera_capture("side", cap)
        try:
            self._configure_camera(cap)
            # Sleeping 3 seconds to give time for the camera to warm up
            sleep(3)

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    continue

                if self._tapping_enabled:
                    self._detect_tapping(frame)
                else:
                    self._detect_interaction(frame)
                
                # Add frame to recording queue if recording
                if self._side_writer:
                    try:
                        self._side_frame_queue.put_nowait(frame)
                    except Exception:
                        pass
                    
                cv2.imshow("Side Camera", cv2.flip(frame, 1))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.request_shutdown()
                    break
        finally:
            self._clear_camera_capture("side", cap)
            cap.release()
            self._safe_destroy_window("Side Camera")

    def get_hand_position(self, with_lock = True) -> Optional[HandPosition]:
        """Get the current hand position"""
        if with_lock:
            with self._hand_position_lock:
                return self._current_position
        
        return self._current_position

    def is_interacting(self) -> bool:
        """Get current interaction state"""
        return self._is_interacting

    def _toggle_interaction(self):
        """Toggle interaction state when not using camera"""
        print(f"toggle interaction from {self._is_interacting} to {not self._is_interacting}")
        self._is_interacting = not self._is_interacting
    
    def _process_top_view(self, frame: np.ndarray) -> HandPosition:
        return self._vision.detect_hand(frame)

    def _detect_interaction(self, frame: np.ndarray):
        if self._is_moderator_paused():
            self._is_interacting = False
            return

        current_pos = self.get_hand_position(with_lock=False)
        if current_pos is None or all([
            current_pos.thumb_x == 0,
            current_pos.thumb_y == 0,
            current_pos.active_finger_x == 0,
            current_pos.active_finger_y == 0,
        ]):
            if self._is_interacting:
                print("Can't find hand, stopping interaction")
                self._is_interacting = False
            return
        
        prev_interaction = self._is_interacting
        self._is_interacting = self._vision.detect_interaction(
            frame,
            current_pos,
            self._side_camera_location
        )
        if prev_interaction != self._is_interacting:
            print(f"Interaction changed to {self._is_interacting}")

    def _detect_tapping(self, frame: np.ndarray):
        """Detect tapping using side camera: finger in bottom 70% = tapping."""
        if self._is_moderator_paused():
            self._is_interacting = False
            return

        side_pos = self._vision.detect_side_hand(frame)
        finger_y = side_pos.active_finger_y
        threshold_y = self._side_height * TAPPING_HEIGHT_RATIO
        prev = self._is_interacting
        self._is_interacting = finger_y > threshold_y and finger_y > 0
        if prev != self._is_interacting:
            print(f"Tapping changed to {self._is_interacting}")
        
    def cleanup(self):
        """Clean up resources"""
        with self._cleanup_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True

        self.request_shutdown()

        with self._state_lock:
            active_pause = self._moderator_pause_started_at
            active_pause_state = self._moderator_pause_state_when_started or self._state
            self._moderator_pause_started_at = None
            self._moderator_pause_state_when_started = None
        if active_pause is not None:
            self._log_moderator_pause(active_pause, datetime.now(), active_pause_state)

        self._safe_close_socket(self._frontend_socket)
        self._safe_close_socket(self._listening_socket)
        self._join_thread(getattr(self, "_top_thread", None), timeout=2.0)
        self._join_thread(getattr(self, "_side_thread", None), timeout=2.0)
        self._join_thread(getattr(self, "_top_recording_thread", None))
        self._join_thread(getattr(self, "_side_recording_thread", None))
        self._join_thread(getattr(self, "_control_thread", None))
        self._join_thread(getattr(self, "_experiment_thread", None))
        self._join_thread(getattr(self, "_frontend_network_thread", None))
        self._join_thread(getattr(self, "_hardware_thread", None))
        self._safe_destroy_all_windows()
        self._cleanup_writers()
        self._vision.cleanup()

        match MOTOR_TYPE:
            case MotorType.HARDWARE:
                self._safe_close_socket(self._hardware_socket)
            case MotorType.NONE:
                ...  # No cleanup needed
            case _:
                raise NotImplementedError("Unknown motor type")  # Should never happen

def start_experiment(
    config: Configuration,
    path: Path,
    run_mode: Literal["comparison", "single_finger"],
    motor_set_id: MotorSetId,
    hand_orientation: HandOrientation,
    movement_strategy: MovementStrategy,
    target_cycle_count: int,
    vision_type: VisionType,
    tapping_enabled: bool,
    finger_colors: dict[str, str],
    side_camera_off: bool = False,
    play_white_noise: bool = False,
    is_debug: bool = True,
    resume_pair_number: Optional[int] = None,
):
    """Initialize and start the hand tracking system"""
    experiment = Experiment(
        config,
        path,
        run_mode=run_mode,
        motor_set_id=motor_set_id,
        hand_orientation=hand_orientation,
        movement_strategy=movement_strategy,
        target_cycle_count=target_cycle_count,
        vision_type=vision_type,
        tapping_enabled=tapping_enabled,
        finger_colors=finger_colors,
        side_camera_off=side_camera_off,
        play_white_noise=play_white_noise,
        is_debug=is_debug,
        resume_pair_number=resume_pair_number,
    )

    return experiment


PAIR_FOLDER_PATTERN = "pair_"


def _parse_pair_folder_number(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith(PAIR_FOLDER_PATTERN):
        return None
    suffix = path.name[len(PAIR_FOLDER_PATTERN):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _next_available_archive_path(preferred_archive_path: Path) -> Path:
    """Return ``preferred_archive_path`` or a numbered variant if it already exists."""
    if not preferred_archive_path.exists():
        return preferred_archive_path

    parent = preferred_archive_path.parent
    stem = preferred_archive_path.stem
    suffix = preferred_archive_path.suffix
    index = 1
    while True:
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _archive_path(path: Path, preferred_archive_path: Path) -> Path | None:
    if not path.exists():
        return None
    archive_path = _next_available_archive_path(preferred_archive_path)
    path.rename(archive_path)
    return archive_path


def _archive_resume_pair_folders(experiment_folder: Path, resume_pair_number: int) -> None:
    pair_folders: list[tuple[int, Path]] = []
    for entry in experiment_folder.iterdir():
        pair_number = _parse_pair_folder_number(entry)
        if pair_number is None or pair_number < resume_pair_number:
            continue
        pair_folders.append((pair_number, entry))

    for _, pair_folder in sorted(pair_folders, reverse=True):
        _archive_path(pair_folder, pair_folder.with_name(f"{pair_folder.name}_old"))


def _read_answers_before_pair(answers_csv: Path, resume_pair_number: int) -> tuple[list[str], list[dict[str, str]]]:
    if not answers_csv.exists():
        return ANSWERS_HEADER, []

    with open(answers_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ANSWERS_HEADER
        rows: list[dict[str, str]] = []
        for row in reader:
            try:
                pair_number = int(row.get("pair_number", ""))
            except (TypeError, ValueError):
                continue
            if pair_number < resume_pair_number:
                rows.append(row)
    return fieldnames, rows


def _archive_and_rewrite_answers(experiment_folder: Path, resume_pair_number: int) -> None:
    answers_csv = experiment_folder / "answers.csv"
    fieldnames, rows_to_keep = _read_answers_before_pair(answers_csv, resume_pair_number)

    if answers_csv.exists():
        _archive_path(answers_csv, experiment_folder / "answers_old.csv")

    with open(answers_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_to_keep)


def _prepare_resume_outputs(experiment_folder: Path, resume_pair_number: int) -> None:
    """Archive stale resumed outputs and recreate answers for rows before resume."""
    _archive_resume_pair_folders(experiment_folder, resume_pair_number)
    _archive_and_rewrite_answers(experiment_folder, resume_pair_number)


def _find_latest_experiment(root: Path) -> Path | None:
    """Return the most recently modified subfolder of ``root``, or None."""
    if not root.exists() or not root.is_dir():
        return None
    folders = [f for f in root.iterdir() if f.is_dir()]
    if not folders:
        return None
    folders.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return folders[0]


def _find_last_pair_number(experiment_folder: Path) -> int | None:
    """Return the highest ``pair_NNN`` index in ``experiment_folder``, or None."""
    if not experiment_folder.exists():
        return None
    pair_numbers: list[int] = []
    for entry in experiment_folder.iterdir():
        pair_number = _parse_pair_folder_number(entry)
        if pair_number is not None:
            pair_numbers.append(pair_number)
    if not pair_numbers:
        return None
    return max(pair_numbers)


def _answer_already_recorded(answers_csv: Path, pair_number: int) -> bool:
    """Return True if ``answers.csv`` already has a row for ``pair_number``."""
    if not answers_csv.exists():
        return False

    with open(answers_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("pair_number", "")) == pair_number:
                    return True
            except (TypeError, ValueError):
                continue
    return False

def get_target_cycle_count() -> int:
    while True:
        user_input = input(f"How many cycles to apply? [{TARGET_CYCLE_COUNT}]: ").strip()
        if user_input == "":
            return TARGET_CYCLE_COUNT

        if user_input.isdigit() and int(user_input) > 0:
            return int(user_input)

        original_print("Invalid input. Please enter a positive whole number.")


def _get_protocol_files(protocols_folder: Path = PROTOCOLS_FOLDER) -> list[Path]:
    """Return selectable top-level protocol CSVs, ignoring nested folders."""
    if not protocols_folder.exists() or not protocols_folder.is_dir():
        return []
    return sorted(
        path
        for path in protocols_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )


def _resolve_protocol_path(
    protocol: str,
    protocols_folder: Path = PROTOCOLS_FOLDER,
) -> Path:
    protocol_files = _get_protocol_files(protocols_folder)
    if not protocol_files:
        raise typer.BadParameter(
            f"No top-level CSV protocol files found in {protocols_folder}",
            param_hint="--protocol",
        )

    protocol = protocol.strip()
    if not protocol:
        raise typer.BadParameter("protocol cannot be empty", param_hint="--protocol")

    if protocol.isdigit():
        protocol_index = int(protocol)
        if 1 <= protocol_index <= len(protocol_files):
            return protocol_files[protocol_index - 1]
        raise typer.BadParameter(
            f"protocol number must be between 1 and {len(protocol_files)}",
            param_hint="--protocol",
        )

    protocol_path = Path(protocol)
    if protocol_path.is_absolute() or protocol_path.name != protocol:
        raise typer.BadParameter(
            "protocol must be a file name from the protocols folder, not a path",
            param_hint="--protocol",
        )

    requested_name = protocol if protocol_path.suffix else f"{protocol}.csv"
    for protocol_file in protocol_files:
        if protocol_file.name.lower() == requested_name.lower():
            return protocol_file

    available = ", ".join(path.name for path in protocol_files)
    raise typer.BadParameter(
        f"unknown protocol '{protocol}'. Available protocols: {available}",
        param_hint="--protocol",
    )


def get_protocol_path() -> Path:
    protocol_files = _get_protocol_files()
    if not protocol_files:
        raise typer.BadParameter(
            f"No top-level CSV protocol files found in {PROTOCOLS_FOLDER}",
            param_hint="--protocol",
        )

    original_print("Available protocols:")
    for index, protocol_file in enumerate(protocol_files, start=1):
        original_print(f"  {index}. {protocol_file.name}")

    while True:
        user_input = input("Select protocol [1]: ").strip()
        if user_input == "":
            return protocol_files[0]
        try:
            return _resolve_protocol_path(user_input)
        except typer.BadParameter as exc:
            original_print(f"Invalid input. {exc}")


def get_run_mode() -> Literal["comparison", "single_finger"]:
    while True:
        user_input = input("Run mode - [C]omparison / [S]ingle_finger [C]: ").strip().upper()
        if user_input in ("", "C"):
            return "comparison"
        if user_input == "S":
            return "single_finger"
        original_print("Invalid input. Please enter C or S.")


def get_vision_type() -> VisionType:
    options = list(VisionType)
    original_print("Available vision types:")
    for i, vt in enumerate(options):
        default_marker = " (default)" if vt == VisionType.MEDIAPIPE else ""
        original_print(f"  {i + 1}. {vt.value}{default_marker}")
    while True:
        user_input = input("Select vision type [1]: ").strip()
        if user_input == "":
            return VisionType.MEDIAPIPE
        if user_input.isdigit() and 1 <= int(user_input) <= len(options):
            return options[int(user_input) - 1]
        original_print(f"Invalid input. Please enter a number between 1 and {len(options)}.")


def get_motor_set_id() -> MotorSetId:
    options = list(MotorSetId)
    original_print("Select motor set index:")
    for option in options:
        original_print(f"  {option.value}. Motors {option.label}")

    while True:
        user_input = input("Motor set index [0]: ").strip()
        if user_input == "":
            return MotorSetId.MOTORS_0_2
        if user_input.isdigit() and 0 <= int(user_input) < len(options):
            return MotorSetId(int(user_input))
        original_print("Invalid input. Enter a motor set index from 0 to 4.")


def _resolve_motor_set_id(motor_set: int | None) -> MotorSetId:
    if motor_set is None:
        return get_motor_set_id()
    try:
        return MotorSetId(motor_set)
    except ValueError as exc:
        raise typer.BadParameter("motor set must be an index from 0 to 4", param_hint="--motor-set") from exc


def get_side_camera_off() -> bool:
    """Ask user if the side camera is available. Returns True if camera is OFF."""
    while True:
        user_input = input("Side camera enabled? [Y/n]: ").strip().lower()
        if user_input in ("", "y", "yes"):
            return False
        if user_input in ("n", "no"):
            return True
        original_print("Invalid input. Please enter Y or N.")


def get_play_white_noise() -> bool:
    """Ask whether the frontend should play white-noise masking. Default is no."""
    while True:
        user_input = input("Apply white noise? [y/N]: ").strip().lower()
        if user_input in ("", "n", "no"):
            return False
        if user_input in ("y", "yes"):
            return True
        original_print("Invalid input. Please enter Y or N.")


AVAILABLE_COLORS = ["red", "green", "blue", "yellow", "magenta"]
RunModeLiteral = Literal["comparison", "single_finger"]

def get_finger_colors(
    fingers: list[str],
    unavailable_colors: set[str] | None = None,
) -> dict[str, str]:
    """Ask user to assign a unique color to each finger from the available set."""
    finger_colors: dict[str, str] = {}
    remaining = [color for color in AVAILABLE_COLORS if color not in (unavailable_colors or set())]
    for finger in fingers:
        original_print(f"Available colors: {', '.join(c.capitalize() for c in remaining)}")
        # In single_finger mode the synthetic name "single" stands in for "the
        # one tracked finger, regardless of which finger the protocol marks active".
        prompt_label = "the single tracked finger" if finger == "single" else finger
        while True:
            user_input = input(f"  Color for {prompt_label}: ").strip().lower()
            if user_input in remaining:
                finger_colors[finger] = user_input
                remaining.remove(user_input)
                break
            original_print(f"  Invalid. Choose from: {', '.join(c.capitalize() for c in remaining)}")
    return finger_colors


def get_hand_orientation() -> HandOrientation:
    """Get user input for mirrored hand orientation (default: not mirrored)."""
    while True:
        user_input = input("Mirror hand orientation? [y/N]: ").strip().lower()
        if user_input in ("", "n", "no"):
            return HandOrientation.NOT_MIRRORED
        if user_input in ("y", "yes"):
            return HandOrientation.MIRRORED
        original_print("Invalid input. Please enter Y or N.")


def _is_color_vision(vision_type: VisionType) -> bool:
    return vision_type in (VisionType.FRAME_COLOR, VisionType.OPTICAL_COLOR_NO_GPU, VisionType.OPTICAL_COLOR_GPU)


def _parse_finger_color_flags(values: list[str] | None) -> dict[str, str]:
    colors: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise typer.BadParameter("finger colors must use FINGER=COLOR format", param_hint="--finger-color")

        finger, color = (part.strip().lower() for part in value.split("=", 1))
        # "single" is the synthetic key used by single-finger mode for the one
        # color that's tracked regardless of the protocol's active finger.
        valid_fingers = (*FINGER_NAMES.values(), "single")
        if finger not in valid_fingers:
            raise typer.BadParameter(
                f"unknown finger '{finger}'. Expected one of: {', '.join(valid_fingers)}",
                param_hint="--finger-color",
            )
        if color not in AVAILABLE_COLORS:
            raise typer.BadParameter(
                f"unknown color '{color}'. Expected one of: {', '.join(AVAILABLE_COLORS)}",
                param_hint="--finger-color",
            )
        if finger in colors:
            raise typer.BadParameter(f"duplicate color for finger '{finger}'", param_hint="--finger-color")
        if color in colors.values():
            raise typer.BadParameter(f"color '{color}' is assigned more than once", param_hint="--finger-color")
        colors[finger] = color
    return colors


def _get_fingers_needed(config: Configuration, run_mode: RunModeLiteral) -> list[str]:
    # Single-finger mode tracks one color regardless of which finger the protocol
    # marks active, so we only need to collect that one color.
    if run_mode == "single_finger":
        return ["single"]
    finger_ids_in_csv: set[int] = set()
    for pair in config.pairs:
        if pair.first.value > 0:
            finger_ids_in_csv.add(pair.first.finger_id)
            finger_ids_in_csv.add(pair.second.finger_id)
    configured_fingers = sorted(
        {FINGER_NAMES[fid] for fid in finger_ids_in_csv if fid in FINGER_NAMES and FINGER_NAMES[fid] != "thumb"},
        key=lambda n: list(FINGER_NAMES.values()).index(n)
    )
    return ["thumb"] + configured_fingers


def _resolve_finger_colors(
    config: Configuration,
    run_mode: RunModeLiteral,
    vision_type: VisionType,
    finger_color_flags: list[str] | None,
) -> dict[str, str]:
    provided_colors = _parse_finger_color_flags(finger_color_flags)
    if not _is_color_vision(vision_type):
        if provided_colors:
            raise typer.BadParameter("--finger-color can only be used with a color vision type")
        return {}

    fingers_needed = _get_fingers_needed(config, run_mode)
    unexpected = set(provided_colors) - set(fingers_needed)
    if unexpected:
        raise typer.BadParameter(
            f"color provided for unused finger(s): {', '.join(sorted(unexpected))}",
            param_hint="--finger-color",
        )

    finger_colors = dict(provided_colors)
    missing_fingers = [finger for finger in fingers_needed if finger not in finger_colors]
    if missing_fingers:
        if missing_fingers == ["single"]:
            original_print(
                "Single-finger mode: assign one color for the tracked finger "
                "(used regardless of which finger the protocol marks active)."
            )
        else:
            original_print(f"Assign colors for fingers: {', '.join(missing_fingers)}")
        selected = get_finger_colors(missing_fingers, set(finger_colors.values()))
        duplicate_colors = set(finger_colors.values()) & set(selected.values())
        if duplicate_colors:
            raise typer.BadParameter(
                f"color assigned more than once: {', '.join(sorted(duplicate_colors))}",
                param_hint="--finger-color",
            )
        finger_colors.update(selected)

    return finger_colors


def create_experiment_folder(
    config: Configuration,
    run_mode: RunModeLiteral,
    motor_set_id: MotorSetId,
):
    experiment_path = Path(experiment_folder)
    if not experiment_path.exists():
        experiment_path.mkdir()
    
    existing_folders = [f for f in experiment_path.iterdir() if f.is_dir()]
    if not existing_folders:
        folder_id = 1
    else:
        folder_id = max(int(folder.name.split('_')[-1]) for folder in existing_folders) + 1

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_path = experiment_path / f"{timestamp}_{run_mode}_config_fingers_motor_set_{motor_set_id.value}_{folder_id:03d}"
    
    folder_path.mkdir()
    config.write_configuration(folder_path / "configuration.csv")

    return folder_path


def _resolve_resume_context(
    experiment_root: Path, from_pair: int | None
) -> tuple[Path, Configuration, int]:
    """Locate the latest experiment folder and the pair_number to resume at."""
    folder = _find_latest_experiment(experiment_root)
    if folder is None:
        raise typer.BadParameter(
            f"No experiment folder found under {experiment_root}",
            param_hint="--continue-last-experiment",
        )

    config_path = folder / "configuration.csv"
    if not config_path.exists():
        raise typer.BadParameter(
            f"Resume target {folder} is missing configuration.csv",
            param_hint="--continue-last-experiment",
        )

    config = Configuration.read_configuration(str(config_path))
    last_pair_number = _find_last_pair_number(folder)

    if from_pair is not None:
        if from_pair < 1:
            raise typer.BadParameter("--from-pair must be >= 1", param_hint="--from-pair")
        if last_pair_number is None:
            if from_pair > 1:
                raise typer.BadParameter(
                    (
                        f"--from-pair {from_pair} cannot be used because no "
                        "pair_NNN folders exist yet; start at pair 1"
                    ),
                    param_hint="--from-pair",
                )
        elif from_pair > last_pair_number:
            raise typer.BadParameter(
                (
                    f"--from-pair {from_pair} is higher than the latest started "
                    f"pair ({last_pair_number}); choose pair {last_pair_number} "
                    "or earlier so the last started pair is re-done"
                ),
                param_hint="--from-pair",
            )
        resume_pair_number = from_pair
    else:
        resume_pair_number = last_pair_number if last_pair_number is not None else 1

    return folder, config, resume_pair_number


def main(
    run_mode: Annotated[RunMode | None, typer.Option("--run-mode", help="Experiment run mode. Asked interactively when omitted.")] = None,
    motor_set: Annotated[int | None, typer.Option("--motor-set", min=0, max=4, help="Physical motor set index 0-4. Index n drives motors n*3 through n*3+2. Asked interactively when omitted.")] = None,
    vision_type: Annotated[VisionType | None, typer.Option("--vision-type", help="Vision backend. Asked interactively when omitted.")] = None,
    protocol: Annotated[str | None, typer.Option("--protocol", help="Protocol CSV to load from the top level of protocols/. Accepts a file name, stem, or listed number. Asked interactively when omitted for new experiments.")] = None,
    finger_color: Annotated[list[str] | None, typer.Option("--finger-color", help="Color assignment as FINGER=COLOR. Repeat for each required color-tracked finger. In single_finger run mode use FINGER=single (e.g. --finger-color single=red): one color is tracked regardless of which finger the protocol marks active.")] = None,
    side_camera_enabled: Annotated[bool | None, typer.Option("--side-camera-enabled/--side-camera-disabled", help="Whether the side camera is enabled. Asked interactively when omitted.")] = None,
    play_white_noise: Annotated[bool | None, typer.Option("--white-noise/--no-white-noise", help="Whether frontend should play white-noise masking. Asked interactively when omitted.")] = None,
    mirror_hand: Annotated[bool | None, typer.Option("--mirror-hand/--no-mirror-hand", help="Whether to mirror hand orientation. Asked interactively when omitted.")] = None,
    movement_strategy: Annotated[MovementStrategy, typer.Option("--movement-strategy", help="Motor movement mapping strategy. Use 'ik' for the IK solver ported into this repository.")] = MOVEMENT_STRATEGY,
    target_cycle_count: Annotated[int | None, typer.Option("--target-cycle-count", min=1, help="Number of cycles to apply. Asked interactively when omitted.")] = None,
    is_debug: Annotated[bool, typer.Option("--debug/--no-debug", help="Enable debug prints/logs and send the debug flag to frontend packets.")] = True,
    continue_last_experiment: Annotated[bool, typer.Option("--continue-last-experiment", "--continue-last", help="Resume the latest experiment folder. Reuses its configuration.csv and starts at the beginning of a pair.")] = False,
    from_pair: Annotated[int | None, typer.Option("--from-pair", min=1, help="With --continue-last-experiment, resume at this pair number from configuration.csv. Must be no higher than the latest pair_NNN folder so the last started pair is re-done. Defaults to the last pair_NNN folder present.")] = None,
):
    set_is_debug(is_debug)

    if from_pair is not None and not continue_last_experiment:
        raise typer.BadParameter(
            "--from-pair requires --continue-last-experiment",
            param_hint="--from-pair",
        )

    resume_pair_number: int | None = None
    if continue_last_experiment:
        path, config, resume_pair_number = _resolve_resume_context(Path(experiment_folder), from_pair)
        print(f"Resuming experiment {path.name} at pair {resume_pair_number}")
    else:
        protocol_path = _resolve_protocol_path(protocol) if protocol is not None else get_protocol_path()
        print(f"Using protocol {protocol_path.name}")
        config = Configuration.read_configuration(str(protocol_path))
        path = None  # created after run params are resolved

    resolved_run_mode: RunModeLiteral = (run_mode.value if run_mode is not None else get_run_mode())  # type: ignore[assignment]
    resolved_motor_set_id = _resolve_motor_set_id(motor_set)
    resolved_vision_type = vision_type if vision_type is not None else get_vision_type()
    finger_colors = _resolve_finger_colors(
        config,
        resolved_run_mode,
        resolved_vision_type,
        finger_color,
    )

    side_camera_off = (not side_camera_enabled) if side_camera_enabled is not None else get_side_camera_off()
    resolved_play_white_noise = play_white_noise if play_white_noise is not None else get_play_white_noise()
    tapping_enabled = resolved_run_mode == "single_finger"
    hand_orientation = (
        HandOrientation.MIRRORED if mirror_hand else HandOrientation.NOT_MIRRORED
    ) if mirror_hand is not None else get_hand_orientation()
    resolved_target_cycle_count = target_cycle_count if target_cycle_count is not None else get_target_cycle_count()

    if path is None:
        path = create_experiment_folder(config, resolved_run_mode, resolved_motor_set_id)

    if resume_pair_number is not None:
        _prepare_resume_outputs(path, resume_pair_number)

    experiment = start_experiment(
        config, path, resolved_run_mode, resolved_motor_set_id, hand_orientation,
        movement_strategy, resolved_target_cycle_count, resolved_vision_type, tapping_enabled, finger_colors,
        side_camera_off, resolved_play_white_noise, is_debug,
        resume_pair_number=resume_pair_number,
    )

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):
        original_print("\nCtrl-C received; shutting down cameras and background workers...")
        experiment.request_shutdown()

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        while experiment.is_running():
            sleep(0.1)
    except KeyboardInterrupt:
        experiment.request_shutdown()
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        experiment.cleanup()


if __name__ == "__main__":
    typer.run(main)
