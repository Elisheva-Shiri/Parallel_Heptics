from abc import ABC, abstractmethod
from pydantic import BaseModel
import numpy as np

class HandPosition(BaseModel):
    thumb_x: float
    thumb_y: float
    active_finger_x: float  # The chosen finger for object control
    active_finger_y: float

class BaseVision(ABC):
    """Base class for all vision systems"""

    @abstractmethod
    def set_active_finger(self, finger: str):
        raise NotImplementedError("Subclasses must implement set_active_finger method")
    
    @abstractmethod
    def detect_hand(self, frame: np.ndarray) -> HandPosition:
        raise NotImplementedError("Subclasses must implement update method")
    
    @abstractmethod
    def detect_pinch(self, frame: np.ndarray) -> bool:
        raise NotImplementedError("Subclasses must implement update method")
    
    def cleanup(self):
        """Clean up resources"""
        ...
