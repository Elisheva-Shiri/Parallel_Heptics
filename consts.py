from typing import Literal


VIRTUAL_OBJECT_FPS = 60
FRONTEND_FPS = 60
MOTORS_COMMUNICATION_RATE = 150
TOP_WIDTH = 640
TOP_HEIGHT = 480
SIDE_WIDTH = 640
SIDE_HEIGHT = 480
TARGET_CYCLE_COUNT = 1
PAUSE_SLEEP_SECONDS = 60
CENTER_THRESHOLD = 20
EDGE_THRESHOLD = 30
TAPPING_HEIGHT_RATIO = 0.3  # top 30% is inactive, bottom 70% is active

BACKEND_PORT = 12344
UNITY_PORT = 12345
PYGAME_PORT = 12346
HARDWARE_PORT = 12347

FINGER_NAMES: dict[int, str] = {
    0: "thumb",
    1: "index",
    2: "middle",
    3: "ring",
    4: "pinky",
}

FingerName = Literal["thumb", "index", "middle", "ring", "pinky"]
