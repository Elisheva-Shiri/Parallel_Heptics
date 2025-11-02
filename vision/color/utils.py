import numpy as np

# Color detection HSV ranges
COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},  # Red lower range
        {'lower': np.array([160, 100, 100]), 'upper': np.array([180, 255, 255])}  # Red upper range
    ],
    'blue': [
        {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])}
    ],
    'yellow': [
        {'lower': np.array([15, 80, 80]), 'upper': np.array([35, 255, 255])}
    ],
    'green': [
        {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])}
    ]
}

FINGER_COLORS = {
    "thumb": "yellow", 
    "index": "green",
    "middle": "blue",
    "ring": "red"
}

def calculate_orientation_angle(dx: float, dy: float, prev_angle: float = -1) -> float:
    """
    Calculate orientation angle in degrees (0-360) relative to X-axis.
    Returns -1 if no motion detected.
    """
    if abs(dx) < 0.1 and abs(dy) < 0.1:  # Threshold for no motion
        return prev_angle
    
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

def check_gpu_availability() -> bool:
    """Check if GPU is available for processing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False 