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
MOVEMENT_AREA_SCALE = 2  # 1.0 = full size, 0.5 = half size, 2/3 = current working-area size
TAPPING_HEIGHT_RATIO = 0.3  # top 30% is inactive, bottom 70% is active
STIFFNESS_MAX = 175

BACKEND_PORT = 12344
UNITY_PORT = 12345
PYGAME_PORT = 12346
HARDWARE_PORT = 12347

# Finger IDs used by experiment configuration/protocol CSVs. The thumb is
# tracked separately for interaction geometry; configured IDs select the active
# non-thumb finger unless single-finger mode explicitly selects thumb.
FINGER_NAMES: dict[int, str] = {
    0: "index",
    1: "middle",
    2: "ring",
    3: "pinky",
    4: "thumb",
}

FingerName = Literal["thumb", "index", "middle", "ring", "pinky"]
