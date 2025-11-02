import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .utils import COLOR_RANGES, calculate_orientation_angle

# ============================================================================
# BASE CLASSES AND DATA STRUCTURES
# ============================================================================

@dataclass
class TrackedObject:
    """Base class for tracked objects in frame-based tracking."""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    angle: float
    prev_position: Optional[Tuple[float, float]] = None

@dataclass
class FlowTrackedObject:
    """Enhanced tracked object for optical flow tracking."""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    angle: float
    flow_points: np.ndarray
    flow_vectors: np.ndarray
    confidence: float
    past_positions: List[Tuple[float, float]]
    filtered_velocity: Tuple[float, float]
    filtered_angle: float

# ============================================================================
# COLOR DETECTOR CLASS
# ============================================================================

class ColorDetector:
    """Base color detection functionality."""
    
    def __init__(self, min_contour_area: int = 100, colors: Optional[List[str]] = None):
        self.min_contour_area = min_contour_area
        self.colors = colors or list(COLOR_RANGES.keys())
    
    def detect_colors(self, frame: np.ndarray) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Detect the most prominent object of each color in the frame.
        Returns a dictionary with color names as keys and (x, y) center coordinates as values.
        Returns None for colors where no object is detected.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = {color: None for color in self.colors}
        
        for color in self.colors:
            # Create mask for the color
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for color_range in COLOR_RANGES[color]:
                color_mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                mask = cv2.bitwise_or(mask, color_mask)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > self.min_contour_area:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        results[color] = (cx, cy)
        
        return results
    
    def draw_detections(self, frame: np.ndarray, detections: Dict[str, Optional[Tuple[float, float]]]) -> np.ndarray:
        """Draw detected objects and their centers on the frame."""
        output = frame.copy()
        
        for color, center in detections.items():
            if center is not None:
                x, y = map(int, center)
                # Draw circle at center
                cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
                # Draw color label
                cv2.putText(output, color, (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output

# ============================================================================
# FRAME-BASED TRACKER SUBCLASS
# ============================================================================

class FrameTracker:
    """Frame-based tracking using color detection and velocity calculation."""
    
    def __init__(self, fps: float):
        self.detector = ColorDetector()
        self.fps = fps
        self.dt = 1.0 / fps
        self.tracked_objects: Dict[str, TrackedObject] = {}
    
    def update(self, frame: np.ndarray) -> Tuple[Dict[str, TrackedObject], np.ndarray]:
        """
        Update tracking for all colors in the frame.
        Returns tracked objects and visualization frame.
        """
        # Detect colors in current frame
        current_detections = self.detector.detect_colors(frame)
        
        # Update tracking for each color
        for color, current_pos in current_detections.items():
            if current_pos is None:
                if color in self.tracked_objects:
                    del self.tracked_objects[color]
                continue
            
            if color not in self.tracked_objects:
                # Initialize new tracked object
                self.tracked_objects[color] = TrackedObject(
                    position=current_pos,
                    velocity=(0.0, 0.0),
                    angle=-1.0,
                    prev_position=None
                )
            else:
                # Update existing tracked object
                prev_pos = self.tracked_objects[color].position
                dx = (current_pos[0] - prev_pos[0]) / self.dt
                dy = (current_pos[1] - prev_pos[1]) / self.dt
                angle = calculate_orientation_angle(dx, dy, self.tracked_objects[color].angle)
                
                self.tracked_objects[color] = TrackedObject(
                    position=current_pos,
                    velocity=(dx, dy),
                    angle=angle,
                    prev_position=prev_pos
                )
        
        # Create visualization
        vis_frame = self.detector.draw_detections(frame, current_detections)
        vis_frame = self._draw_tracking(vis_frame)
        
        return self.tracked_objects, vis_frame
    
    def _draw_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking information on the frame."""
        for color, obj in self.tracked_objects.items():
            if obj.prev_position is not None:
                # Draw motion vector
                cv2.arrowedLine(
                    frame,
                    (int(obj.prev_position[0]), int(obj.prev_position[1])),
                    (int(obj.position[0]), int(obj.position[1])),
                    (0, 255, 0),
                    2
                )
                
                # Draw velocity and angle
                info_text = f"vx: {obj.velocity[0]:.1f} vy: {obj.velocity[1]:.1f} θ: {obj.angle:.1f}°"
                cv2.putText(
                    frame,
                    info_text,
                    (int(obj.position[0]) + 10, int(obj.position[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
        
        return frame

# ============================================================================
# OPTICAL FLOW TRACKER SUBCLASS
# ============================================================================

class OpticalFlowTracker:
    """Optical flow-based tracking with enhanced motion analysis."""
    
    def __init__(self, fps: float, use_gpu: bool = False, history_length: int = 3):
        # Initialize with a lower minimum contour area for better sensitivity
        self.detector = ColorDetector(min_contour_area=50)  # Reduced from default 100
        self.fps = fps
        self.dt = 1.0 / fps
        self.use_gpu = use_gpu
        self.history_length = history_length
        
        # Initialize optical flow
        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.flow = cv2.cuda_FarnebackOpticalFlow_create()
            self.use_cuda = True
        else:
            self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            self.use_cuda = False
            
        # Background subtractor with more sensitive parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,  # Reduced history for faster adaptation
            varThreshold=8,  # Lower threshold for more sensitivity
            detectShadows=False)
        
        self.prev_frame = None
        self.tracked_objects: Dict[str, FlowTrackedObject] = {}
        
        # Parameters
        self.grid_size = 15  # Reduced grid size for denser flow points
        self.max_points = 150  # Increased max points
        self.frame_shape = None
        self.flow_reset_threshold = 3.0  # Reduced threshold for better sensitivity
        self.ema_alpha = 0.4  # Increased alpha for faster response
    
    def _create_flow_points(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create a grid of points for optical flow tracking."""
        h, w = frame_shape[:2]
        x = np.arange(self.grid_size, w - self.grid_size, self.grid_size)
        y = np.arange(self.grid_size, h - self.grid_size, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)
        return points[:self.max_points]
    
    def _clip_points_to_bounds(self, points: np.ndarray) -> np.ndarray:
        """Ensure points stay within image bounds."""
        if self.frame_shape is None:
            return points
        h, w = self.frame_shape[:2]
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        return points
    
    def update(self, frame: np.ndarray) -> Tuple[Dict[str, FlowTrackedObject], np.ndarray]:
        """Update tracking using optical flow and color detection."""
        self.frame_shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction with morphological operations
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        frame_masked = cv2.bitwise_and(frame, frame, mask=fg_mask)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.flow_points = self._create_flow_points(frame.shape)
            return self.tracked_objects, frame
        
        # Calculate optical flow
        if self.use_cuda:
            prev_gpu = cv2.cuda_GpuMat(self.prev_frame)
            curr_gpu = cv2.cuda_GpuMat(gray)
            flow_gpu = self.flow.calc(prev_gpu, curr_gpu, None)
            flow = flow_gpu.download()
        else:
            flow = self.flow.calc(self.prev_frame, gray, None)
        
        # Update flow points with bounds checking
        flow_points_int = self._clip_points_to_bounds(self.flow_points).astype(int)
        flow_vectors = flow[flow_points_int[:, 1], flow_points_int[:, 0]]
        new_points = self.flow_points + flow_vectors
        new_points = self._clip_points_to_bounds(new_points)
        
        # Check for abrupt changes
        mean_flow = np.mean(np.linalg.norm(flow_vectors, axis=1))
        if mean_flow > self.flow_reset_threshold:
            self.flow_points = self._create_flow_points(frame.shape)
            new_points = self.flow_points.copy()
        
        # Detect colors on masked frame
        current_detections = self.detector.detect_colors(frame_masked)
        
        # Update tracking for each color
        for color, current_pos in current_detections.items():
            if current_pos is None:
                if color in self.tracked_objects:
                    del self.tracked_objects[color]
                continue
            
            # Find flow vectors near the detected object with larger search radius
            distances = np.linalg.norm(new_points - current_pos, axis=1)
            nearby_indices = np.where(distances < self.grid_size * 3)[0]  # Increased search radius
            
            if len(nearby_indices) > 0:
                nearby_points = self.flow_points[nearby_indices]
                nearby_vectors = new_points[nearby_indices] - nearby_points
                
                # Calculate velocity and confidence
                avg_velocity = np.mean(nearby_vectors, axis=0) / self.dt
                vector_variance = np.var(nearby_vectors, axis=0)
                confidence = 1.0 / (1.0 + np.mean(vector_variance))
                
                # Get previous filtered values
                prev_obj = self.tracked_objects.get(color, None)
                prev_velocity = prev_obj.filtered_velocity if prev_obj else (0.0, 0.0)
                prev_angle = prev_obj.filtered_angle if prev_obj else -1.0
                
                # Apply EMA filtering
                filtered_velocity = tuple(
                    self.ema_alpha * avg_velocity[i] + (1 - self.ema_alpha) * prev_velocity[i]
                    for i in range(2)
                )
                
                angle = calculate_orientation_angle(filtered_velocity[0], filtered_velocity[1], prev_angle)
                filtered_angle = self.ema_alpha * angle + (1 - self.ema_alpha) * prev_angle
                
                # Update past positions
                past_positions = ([current_pos] + 
                    (prev_obj.past_positions if prev_obj else []))[:self.history_length]
                
                self.tracked_objects[color] = FlowTrackedObject(
                    position=current_pos,
                    velocity=tuple(avg_velocity),
                    angle=angle,
                    flow_points=nearby_points,
                    flow_vectors=nearby_vectors,
                    confidence=confidence,
                    past_positions=past_positions,
                    filtered_velocity=filtered_velocity,
                    filtered_angle=filtered_angle
                )
        
        # Update for next frame
        self.prev_frame = gray
        self.flow_points = new_points
        
        # Create visualization
        vis_frame = self.detector.draw_detections(frame, current_detections)
        vis_frame = self._draw_tracking(vis_frame)
        
        return self.tracked_objects, vis_frame
    
    def _draw_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking information and flow vectors on the frame."""
        for color, obj in self.tracked_objects.items():
            # Draw flow vectors
            for point, vector in zip(obj.flow_points, obj.flow_vectors):
                start_point = tuple(map(int, point))
                end_point = tuple(map(int, point + vector))
                cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 1)
            
            # Draw trajectory
            for i in range(len(obj.past_positions) - 1):
                p1 = tuple(map(int, obj.past_positions[i]))
                p2 = tuple(map(int, obj.past_positions[i + 1]))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)
            
            # Draw object info
            x, y = map(int, obj.position)
            info_text = (f"vx: {obj.filtered_velocity[0]:.1f} "
                        f"vy: {obj.filtered_velocity[1]:.1f} "
                        f"θ: {obj.filtered_angle:.1f}° "
                        f"conf: {obj.confidence:.2f}")
            cv2.putText(frame, info_text, (x + 10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame

# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class Detector:
    """
    Main detector class that integrates all tracking methods.
    Provides a unified interface for color detection and object tracking.
    """
    
    def __init__(self, tracking_method: str = "frame", fps: float = 30.0, 
                 use_gpu: bool = False, min_contour_area: int = 100):
        """
        Initialize the detector with specified tracking method.
        
        Args:
            tracking_method: "frame" for frame-based tracking, "optical_flow" for optical flow tracking
            fps: Frame rate for velocity calculations
            use_gpu: Whether to use GPU acceleration (for optical flow)
            min_contour_area: Minimum contour area for color detection
        """
        self.tracking_method = tracking_method
        self.fps = fps
        self.use_gpu = use_gpu
        self.min_contour_area = min_contour_area
        
        # Initialize the appropriate tracker
        if tracking_method == "frame":
            self.tracker = FrameTracker(fps)
        elif tracking_method == "optical_flow":
            self.tracker = OpticalFlowTracker(fps, use_gpu)
        else:
            raise ValueError(f"Unknown tracking method: {tracking_method}. Use 'frame' or 'optical_flow'")
        
        # Initialize color detector
        self.color_detector = ColorDetector(min_contour_area)
    
    def update(self, frame: np.ndarray):
        """
        Update tracking for the current frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (tracked_objects, processed_frame)
        """
        return self.tracker.update(frame)
    
    def detect_colors(self, frame: np.ndarray):
        """
        Detect colors in the frame without tracking.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary of color detections
        """
        return self.color_detector.detect_colors(frame)
    
    def draw_detections(self, frame: np.ndarray, detections):
        """
        Draw color detections on the frame.
        
        Args:
            frame: Input frame
            detections: Color detection results
            
        Returns:
            Frame with detections drawn
        """
        return self.color_detector.draw_detections(frame, detections)
    
    def get_tracking_method(self) -> str:
        """Get the current tracking method."""
        return self.tracking_method
    
    def get_fps(self) -> float:
        """Get the current frame rate."""
        return self.fps
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return self.use_gpu
    
    def get_min_contour_area(self) -> int:
        """Get the minimum contour area threshold."""
        return self.min_contour_area
    
    def set_min_contour_area(self, area: int):
        """Set the minimum contour area threshold."""
        self.min_contour_area = area
        self.color_detector.min_contour_area = area
        if hasattr(self.tracker, 'detector'):
            self.tracker.detector.min_contour_area = area