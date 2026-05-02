from abc import ABC, abstractmethod
from typing import Literal
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
        raise NotImplementedError("Subclasses must implement detect_hand method")
    
    @abstractmethod
    def detect_interaction(
        self,
        frame: np.ndarray,
        top_position: HandPosition,
        camera_pos: Literal["top", "bottom", "left", "right"] = "bottom",
        min_depth: float = 0.1,
        max_depth: float = 1.0
    ) -> bool:
        raise NotImplementedError("Subclasses must implement detect_interaction method")

    def detect_side_hand(self, frame: np.ndarray) -> HandPosition:
        """Detect hand position from side camera frame (used for tapping mode)."""
        raise NotImplementedError("Subclasses must implement detect_side_hand for tapping support")
    
    def cleanup(self):
        """Clean up resources"""
        ...
