from pydantic import BaseModel
from typing import List
import math

class MotorMovement(BaseModel):
    pos: float  # Changed to float for precision
    index: int  # motor index

def calculate_motor_movements(direction: str, distance: float = 10.0, motor_spacing: float = 100.0) -> List[MotorMovement]:
    """
    Calculate motor movements for a triangular motor system using proper kinematics.
    
    Motor layout (top view, coordinate system):
          0 (top)
         / \
        /   \
       1     2
    (left) (right)
    
    Args:
        direction: Movement direction ("up", "down", "left", "right")
        distance: Distance to move the object (in same units as motor_spacing)
        motor_spacing: Distance between motors (for calculating geometry)
        
    Returns:
        List of MotorMovement objects for each motor
    """
    direction = direction.lower()
    
    # Define motor positions in a coordinate system (equilateral triangle)
    # Center at origin (0, 0)
    # Motor 0 (top) at (0, h) where h = motor_spacing * sqrt(3) / 3
    # Motor 1 (left) at (-motor_spacing/2, -h/2)
    # Motor 2 (right) at (motor_spacing/2, -h/2)
    
    h = motor_spacing * math.sqrt(3) / 3  # Height from center to vertex
    
    motor_positions = [
        (0, 2 * h / 3),                    # Motor 0: top
        (-motor_spacing / 2, -h / 3),      # Motor 1: left
        (motor_spacing / 2, -h / 3)        # Motor 2: right
    ]
    
    # Object starts at center (0, 0)
    object_start = (0, 0)
    
    # Calculate object's new position based on direction
    direction_vectors = {
        "up": (0, distance),
        "down": (0, -distance),
        "left": (-distance, 0),
        "right": (distance, 0)
    }
    
    if direction not in direction_vectors:
        raise ValueError(f"Invalid direction: {direction}. Must be one of: up, down, left, right")
    
    dx, dy = direction_vectors[direction]
    object_end = (object_start[0] + dx, object_start[1] + dy)
    
    # Calculate string length changes for each motor
    movements = []
    for i, (mx, my) in enumerate(motor_positions):
        # Initial string length (distance from motor to start position)
        initial_length = math.sqrt((mx - object_start[0])**2 + (my - object_start[1])**2)
        
        # Final string length (distance from motor to end position)
        final_length = math.sqrt((mx - object_end[0])**2 + (my - object_end[1])**2)
        
        # Change in string length
        # Positive = extend string (let out more), Negative = retract string (pull in)
        delta_length = final_length - initial_length
        
        movements.append(MotorMovement(pos=delta_length, index=i))
    
    return movements


# Example usage:
if __name__ == "__main__":
    # Move left by 10 units
    print("Moving LEFT by 10 units:")
    movements = calculate_motor_movements("left", distance=10, motor_spacing=100)
    for movement in movements:
        print(f"Motor {movement.index}: {movement.pos:+.3f} units ({'extend' if movement.pos > 0 else 'retract'})")
    
    print("\nMoving RIGHT by 10 units:")
    movements = calculate_motor_movements("right", distance=10, motor_spacing=100)
    for movement in movements:
        print(f"Motor {movement.index}: {movement.pos:+.3f} units ({'extend' if movement.pos > 0 else 'retract'})")
    
    print("\nMoving UP by 10 units:")
    movements = calculate_motor_movements("up", distance=10, motor_spacing=100)
    for movement in movements:
        print(f"Motor {movement.index}: {movement.pos:+.3f} units ({'extend' if movement.pos > 0 else 'retract'})")
    
    print("\nMoving DOWN by 10 units:")
    movements = calculate_motor_movements("down", distance=10, motor_spacing=100)
    for movement in movements:
        print(f"Motor {movement.index}: {movement.pos:+.3f} units ({'extend' if movement.pos > 0 else 'retract'})")