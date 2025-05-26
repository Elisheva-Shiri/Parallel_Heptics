import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class ColorRange:
    lower: np.ndarray
    upper: np.ndarray
    name: str

class ColorTracker:
    def __init__(self):
        # Define color ranges for each finger
        self.color_ranges = {
            "yellow": ColorRange(
                lower=np.array([20, 100, 100]),  # HSV lower bounds for yellow
                upper=np.array([30, 255, 255]),  # HSV upper bounds for yellow
                name="index"
            ),
            "red": ColorRange(
                lower=np.array([0, 100, 100]),   # HSV lower bounds for red
                upper=np.array([10, 255, 255]),  # HSV upper bounds for red
                name="middle"
            ),
            "blue": ColorRange(
                lower=np.array([100, 100, 100]), # HSV lower bounds for blue
                upper=np.array([130, 255, 255]), # HSV upper bounds for blue
                name="ring"
            )
        }
        
        # Minimum contour area to consider as a valid marker
        self.min_contour_area = 100
        
        # Store last known positions for each finger
        self.last_positions: Dict[str, Optional[Tuple[float, float]]] = {
            "index": None,
            "middle": None,
            "ring": None
        }

    def find_largest_contour(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find the center of the largest contour in the mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None
            
        # Calculate the center of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
            
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        
        return (cx, cy)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Optional[Tuple[float, float]]]:
        """Process a frame and return the positions of all tracked fingers"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Dictionary to store current positions
        current_positions: Dict[str, Optional[Tuple[float, float]]] = {}
        
        # Process each color
        for color_name, color_range in self.color_ranges.items():
            # Create mask for the color
            mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find the largest contour
            position = self.find_largest_contour(mask)
            
            # Store the position
            current_positions[color_range.name] = position
            
            # Update last known position if we found a valid position
            if position is not None:
                self.last_positions[color_range.name] = position
        
        return current_positions

    def draw_debug(self, frame: np.ndarray, positions: Dict[str, Optional[Tuple[float, float]]]) -> np.ndarray:
        """Draw debug visualization on the frame"""
        debug_frame = frame.copy()
        
        # Draw circles for each detected finger
        for finger_name, position in positions.items():
            if position is not None:
                x, y = position
                color = {
                    "index": (0, 255, 255),  # Yellow
                    "middle": (0, 0, 255),   # Red
                    "ring": (255, 0, 0)      # Blue
                }[finger_name]
                
                cv2.circle(debug_frame, (int(x), int(y)), 10, color, -1)
                cv2.putText(debug_frame, finger_name, (int(x) + 15, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return debug_frame 