import cv2
import numpy as np

class BlueObjectDetector:
    def __init__(self):
        # HSV range for blue color detection
        self.lower_blue = np.array([100, 150, 150])
        self.upper_blue = np.array([130, 255, 255])
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Initialize variables for tracking
        self.prev_frame = None
        self.prev_blue_center = None
        
    def detect_blue(self, frame):
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue objects
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and show mask
        result = frame.copy()
        cv2.drawContours(result, contours, -1, (0,255,0), 2)
        
        # Show the mask and result
        cv2.imshow('Blue Mask', mask)
        cv2.imshow('Result', result)
        
        # Get largest blue object
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw center point
                cv2.circle(result, (cx, cy), 5, (0,0,255), -1)
                cv2.imshow('Result with Center', result)
                return (cx, cy)
        
        return None
        
    #! from hear I thik is the problem at Optical Flow
    def track_motion(self, frame, blue_center):
        if blue_center is None:
            self.prev_frame = None
            self.prev_blue_center = None
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None or self.prev_blue_center is None:
            self.prev_frame = gray
            self.prev_blue_center = np.array([[blue_center[0], blue_center[1]]], dtype=np.float32).reshape(-1,1,2)
            return None
            
        # Calculate optical flow for blue center point
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_blue_center, None, **self.lk_params
        )
        
        if status[0][0] == 1:
            # Draw the track
            old_point = tuple(map(int, self.prev_blue_center[0][0]))
            new_point = tuple(map(int, new_points[0][0]))
            frame = cv2.line(frame, new_point, old_point, (0,255,0), 2)
            frame = cv2.circle(frame, new_point, 5, (0,0,255), -1)
            
            cv2.imshow('Blue Object Tracking', frame)
            
            # Update for next frame
            self.prev_frame = gray
            self.prev_blue_center = new_points
            
            return new_points[0], self.prev_blue_center[0]
            
        return None

    def process_frame(self, frame):
        # Detect blue object position
        blue_pos = self.detect_blue(frame)
        
        # Track motion of blue object
        motion = self.track_motion(frame, blue_pos)
        
        # Wait for key press
        cv2.waitKey(1)
        
        return blue_pos, motion

# Example usage:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = BlueObjectDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        blue_pos, motion = detector.process_frame(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

