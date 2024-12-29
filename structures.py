from pydantic import BaseModel

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


class GamePacket(BaseModel):
    landmarks: list[FingerPosition]
    trackingObject: TrackingObject