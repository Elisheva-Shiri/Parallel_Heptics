from pydantic import BaseModel
from enum import Enum

class ExperimentState(Enum):
    START = 0
    COMPARISON = 1
    QUESTION = 2
    PAUSE = 3
    BREAK = 4
    MODERATOR_PAUSE = 5
    END = -1

class QuestionInput(Enum):
    LEFT = 0
    RIGHT = 1

class ControlAction(str, Enum):
    TOGGLE_INTERACTION = "toggle_interaction"
    FINISH_BREAK = "finish_break"
    TOGGLE_PAUSE = "toggle_pause"

class Position(BaseModel):
    x: float
    z: float

class FingerPosition(Position):
    ...

class TrackingObject(Position):
    size: float
    isInteracting: bool
    movementAreaScale: float = 0.0
    # TODO - move progress/cycleCount to another object
    progress: float
    returnProgress: float
    cycleCount: int
    targetCycleCount: int
    pairIndex: int

class StateData(BaseModel):
    state: int
    pauseTime: int

class ExperimentPacket(BaseModel):
    stateData: StateData
    landmarks: list[FingerPosition]
    trackingObject: TrackingObject
    playWhiteNoise: bool = False
    isDebug: bool = True

class ExperimentControl(BaseModel):
    # Backward-compatible question input used by pygame/Unity answer buttons.
    questionInput: int | None = None
    # Moderator-only command path used by moderator_control.py.
    moderatorAction: ControlAction | None = None

