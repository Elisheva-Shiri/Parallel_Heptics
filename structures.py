from pydantic import BaseModel
from enum import Enum

class ExperimentState(Enum):
    START = 0
    COMPARISON = 1
    QUESTION = 2
    PAUSE = 3
    END = -1

class QuestionInput(Enum):
    LEFT = 0
    RIGHT = 1

class Position(BaseModel):
    x: float
    z: float

class FingerPosition(Position):
    ...

class TrackingObject(Position):
    size: float
    isPinched: bool
    # TODO - move progress/cycleCount to another object
    progress: float
    cycleCount: int
    pairIndex: int


class ExperimentPacket(BaseModel):
    state: int
    landmarks: list[FingerPosition]
    trackingObject: TrackingObject

class ExperimentControl(BaseModel):
    questionInput: int

