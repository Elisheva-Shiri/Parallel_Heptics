"""
Motor Control CLI Utility

Sends custom motor commands to Arduino using the triangular motor system protocol.
"""

import math
import time
from typing import Optional
import typer
import serial
from pydantic import BaseModel

app = typer.Typer(help="Motor control utility for Arduino")


class MotorMovement(BaseModel):
    pos: int
    index: int


def calculate_motor_movements(
    direction: str,
    distance: float,
    base_motor_idx: int = 0,
    motor_spacing: float = 1000.0
) -> list[MotorMovement]:
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
        base_motor_idx: Base index for motors (0, 3, 6, etc. for different fingers)
        motor_spacing: Distance between motors (for calculating geometry)

    Returns:
        List of MotorMovement objects for each motor
    """
    direction = direction.lower()

    h = motor_spacing * math.sqrt(3) / 3  # Height from center to vertex

    motor_positions = [
        (0, 2 * h / 3),                    # Motor 0: top
        (-motor_spacing / 2, -h / 3),      # Motor 1: left
        (motor_spacing / 2, -h / 3)        # Motor 2: right
    ]

    object_start = (0, 0)

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

    movements = []
    for i, (mx, my) in enumerate(motor_positions):
        initial_length = math.sqrt((mx - object_start[0])**2 + (my - object_start[1])**2)
        final_length = math.sqrt((mx - object_end[0])**2 + (my - object_end[1])**2)
        delta_length = final_length - initial_length

        movements.append(MotorMovement(pos=int(delta_length), index=i + base_motor_idx))

    return movements


def build_message(motors: list[MotorMovement]) -> str:
    """Build Arduino message from motor movements."""
    message = "Z"
    for motor in motors:
        message += f"M{motor.index}P{motor.pos}"
    message += "F"
    return message


def send_to_arduino(message: str, port: str, baud: int, dry_run: bool = False, wait_response: bool = True):
    """Send message to Arduino via serial."""
    if dry_run:
        typer.echo(f"[DRY RUN] Would send: {message}")
        return

    try:
        # Open serial with flow control disabled to prevent ESP32 reset
        arduino = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=2,
            dsrdtr=False,  # Disable DSR/DTR flow control
            rtscts=False,  # Disable RTS/CTS flow control
        )

        # Explicitly set DTR and RTS low to prevent reset
        arduino.dtr = False
        arduino.rts = False

        typer.echo(f"Connected to {port} (DTR={arduino.dtr}, RTS={arduino.rts})")

        # Wait a moment for connection to stabilize
        time.sleep(0.1)

        # Check if there's any boot message (indicates ESP32 reset)
        if arduino.in_waiting > 0:
            typer.echo("Warning: ESP32 may have reset. Boot messages:")
            while arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    typer.echo(f"  [BOOT] {line}")
            typer.echo("Waiting for ESP32 to initialize...")
            time.sleep(3)  # Wait for motor detection to complete

        # Clear any remaining buffer data
        arduino.reset_input_buffer()

        # Send the message
        bytes_written = arduino.write(message.encode())
        arduino.flush()  # Ensure data is actually sent
        typer.echo(f"Sent: {message} ({bytes_written} bytes)")

        if wait_response:
            # Wait for and display response (movement can take time)
            typer.echo("Waiting for response...")
            start_time = time.time()
            response_lines = []

            # Wait up to 5 seconds for a response
            while time.time() - start_time < 5:
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response_lines.append(line)
                        # If we got an OK or error, we can stop waiting
                        if line.startswith("OK:") or line.startswith("E:"):
                            break
                time.sleep(0.05)

            if response_lines:
                typer.echo("Response from ESP32:")
                for line in response_lines:
                    typer.echo(f"  {line}")
            else:
                typer.echo("No response received after 5 seconds")

        arduino.close()

    except serial.SerialException as e:
        typer.echo(f"Serial error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def move(
    direction: str = typer.Argument(..., help="Direction: up, down, left, right"),
    distance: float = typer.Argument(100.0, help="Distance to move"),
    port: str = typer.Option("COM3", "--port", "-p", help="Serial port"),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    base_index: int = typer.Option(0, "--base", help="Base motor index (0, 3, 6...)"),
    motor_spacing: float = typer.Option(1000.0, "--spacing", "-s", help="Motor spacing"),
    single: bool = typer.Option(False, "--single", help="Single motor mode (lowest index only)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print command without sending"),
):
    """
    Move motors using triangular kinematics calculation.

    Examples:
        motor_cli.py move up 100
        motor_cli.py move left 50 --single
        motor_cli.py move down 200 --base 3
    """
    movements = calculate_motor_movements(direction, distance, base_index, motor_spacing)

    if single:
        movements = [movements[0]]  # Keep only the lowest index motor

    message = build_message(movements)

    typer.echo(f"Direction: {direction}, Distance: {distance}")
    for m in movements:
        typer.echo(f"  Motor {m.index}: position {m.pos}")

    send_to_arduino(message, port, baud, dry_run)


@app.command()
def custom(
    port: str = typer.Option("COM3", "--port", "-p", help="Serial port"),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    m0: Optional[int] = typer.Option(None, "--m0", help="Motor 0 position"),
    m1: Optional[int] = typer.Option(None, "--m1", help="Motor 1 position"),
    m2: Optional[int] = typer.Option(None, "--m2", help="Motor 2 position"),
    m3: Optional[int] = typer.Option(None, "--m3", help="Motor 3 position"),
    m4: Optional[int] = typer.Option(None, "--m4", help="Motor 4 position"),
    m5: Optional[int] = typer.Option(None, "--m5", help="Motor 5 position"),
    m6: Optional[int] = typer.Option(None, "--m6", help="Motor 6 position"),
    m7: Optional[int] = typer.Option(None, "--m7", help="Motor 7 position"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print command without sending"),
):
    """
    Send custom positions to specific motors.

    Examples:
        motor_cli.py custom --m0 100 --m1 -50 --m2 200
        motor_cli.py custom --m3 500 --m4 500 --m5 500
        motor_cli.py custom --m0 0  # Single motor
    """
    motor_options = [m0, m1, m2, m3, m4, m5, m6, m7]

    movements = []
    for idx, pos in enumerate(motor_options):
        if pos is not None:
            movements.append(MotorMovement(pos=pos, index=idx))

    if not movements:
        typer.echo("Error: At least one motor position must be specified", err=True)
        raise typer.Exit(1)

    message = build_message(movements)

    typer.echo("Custom motor positions:")
    for m in movements:
        typer.echo(f"  Motor {m.index}: position {m.pos}")

    send_to_arduino(message, port, baud, dry_run)


@app.command()
def raw(
    message: str = typer.Argument(..., help="Raw message to send (without Z prefix and F suffix)"),
    port: str = typer.Option("COM3", "--port", "-p", help="Serial port"),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print command without sending"),
):
    """
    Send a raw motor command string.

    The Z prefix and F suffix will be added automatically.

    Examples:
        motor_cli.py raw "M0P100M1P200M2P300"
        motor_cli.py raw "M0P0M1P0M2P0"  # Reset all to zero
    """
    full_message = f"Z{message}F"
    typer.echo(f"Sending raw message: {full_message}")
    send_to_arduino(full_message, port, baud, dry_run)


@app.command()
def reset(
    port: str = typer.Option("COM3", "--port", "-p", help="Serial port"),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    num_motors: int = typer.Option(3, "--num", "-n", help="Number of motors to reset"),
    base_index: int = typer.Option(0, "--base", help="Base motor index"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Print command without sending"),
):
    """
    Reset motors to zero position.

    Examples:
        motor_cli.py reset
        motor_cli.py reset --num 6  # Reset 6 motors
        motor_cli.py reset --base 3 --num 3  # Reset motors 3, 4, 5
    """
    movements = [MotorMovement(pos=0, index=base_index + i) for i in range(num_motors)]
    message = build_message(movements)

    typer.echo(f"Resetting motors {base_index} to {base_index + num_motors - 1} to zero")
    send_to_arduino(message, port, baud, dry_run)


def send_and_wait(arduino: serial.Serial, message: str, timeout: float = 5.0):
    """Send a message and wait for response."""
    arduino.write(message.encode())
    arduino.flush()
    typer.echo(f"Sent: {message}")

    # Wait for response
    start_time = time.time()
    while time.time() - start_time < timeout:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if line:
                typer.echo(f"[ESP32] {line}")
                if line.startswith("OK:") or line.startswith("E:"):
                    break
        time.sleep(0.05)


def parse_motor_positions(parts: list[str]) -> list[MotorMovement]:
    """
    Parse motor position pairs from input parts.
    Format: m0 100 m1 -50 m2 200 (pairs of motor and position)
    """
    movements = []
    i = 0
    while i < len(parts) - 1:
        motor_part = parts[i].lower()
        # Check if this looks like a motor identifier (m0, m1, etc.)
        if motor_part.startswith('m') and motor_part[1:].isdigit():
            try:
                motor_idx = int(motor_part[1:])
                position = int(parts[i + 1])
                movements.append(MotorMovement(pos=position, index=motor_idx))
                i += 2
            except ValueError:
                i += 1
        else:
            i += 1
    return movements


def print_monitor_help():
    """Print help for monitor mode commands."""
    typer.echo("""
Monitor Mode Commands:
  Motor positions (pairs of motor and position):
    m0 100 m1 200        Direct motor positions
    custom m0 100 m1 -50 Same as above (custom prefix optional)

  Kinematics movement:
    move up 100          Move using triangular kinematics
    up 100               Shorthand (direction keywords detected)
                         Directions: up, down, left, right

  Raw motor command:
    raw M0P100M1P200     Send raw motor string (Z/F added automatically)

  Utilities:
    zero                 Reset all motors to position 0
    zero 3               Reset motors 0-2 to position 0

  ESP32 debug commands:
    STATUS               Show motor status
    RESET                Reset encoder positions
    SHOWPID              Show PID values
    HELP                 Show ESP32 help

  Monitor control:
    help, ?              Show this help
    quit, exit, q        Exit monitor mode
""")


@app.command()
def monitor(
    port: str = typer.Option("COM3", "--port", "-p", help="Serial port"),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    base_index: int = typer.Option(0, "--base", help="Base motor index for kinematics"),
    motor_spacing: float = typer.Option(1000.0, "--spacing", "-s", help="Motor spacing for kinematics"),
):
    """
    Interactive monitor mode - keeps connection open to avoid ESP32 resets.

    Supports all CLI commands: move, raw, zero, plus direct m0=100 format.

    Examples:
        motor_cli.py monitor -p COM13
    """
    try:
        arduino = serial.Serial(port=port, baudrate=baud, timeout=0.1)
        typer.echo(f"Connected to {port}")
        typer.echo("Waiting for ESP32 to initialize...")

        # Wait for and display boot messages
        time.sleep(0.5)
        while arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if line:
                typer.echo(f"  {line}")

        # Wait for motor detection to complete
        time.sleep(3)
        while arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if line:
                typer.echo(f"  {line}")

        typer.echo("\nReady! Type 'help' for commands, 'quit' to exit.\n")

        while True:
            # Check for incoming data from ESP32
            while arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    typer.echo(f"[ESP32] {line}")

            # Get user input
            try:
                user_input = input("> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            lower_input = user_input.lower()
            parts = user_input.split()

            # Exit commands
            if lower_input in ('quit', 'exit', 'q'):
                break

            # Help command
            if lower_input in ('help', '?'):
                print_monitor_help()
                continue

            # Move command with kinematics: "move up 100" or shorthand "up 100"
            directions = ('up', 'down', 'left', 'right')
            if lower_input.startswith('move ') or (parts and parts[0].lower() in directions):
                try:
                    # Handle both "move up 100" and "up 100"
                    if lower_input.startswith('move '):
                        dir_idx, dist_idx = 1, 2
                    else:
                        dir_idx, dist_idx = 0, 1

                    if len(parts) > dist_idx:
                        direction = parts[dir_idx]
                        distance = float(parts[dist_idx])
                        movements = calculate_motor_movements(direction, distance, base_index, motor_spacing)
                        message = build_message(movements)
                        typer.echo(f"Kinematics ({direction} {distance}):")
                        for m in movements:
                            typer.echo(f"  Motor {m.index}: {m.pos}")
                        send_and_wait(arduino, message)
                    else:
                        typer.echo("Usage: [move] <direction> <distance>")
                        typer.echo("  Directions: up, down, left, right")
                except ValueError as e:
                    typer.echo(f"Error: {e}")
                continue

            # Raw command: "raw M0P100M1P200"
            if lower_input.startswith('raw '):
                raw_msg = user_input[4:].strip()
                message = f"Z{raw_msg}F"
                send_and_wait(arduino, message)
                continue

            # Zero command: "zero" or "zero 3"
            if lower_input.startswith('zero'):
                num_motors = 3
                if len(parts) >= 2:
                    try:
                        num_motors = int(parts[1])
                    except ValueError:
                        pass
                movements = [MotorMovement(pos=0, index=base_index + i) for i in range(num_motors)]
                message = build_message(movements)
                typer.echo(f"Zeroing motors {base_index} to {base_index + num_motors - 1}")
                send_and_wait(arduino, message)
                continue

            # Custom/direct motor format: "m0 100 m1 200" or "custom m0 100 m1 200"
            # Strip optional "custom" prefix
            motor_parts = parts
            if lower_input.startswith('custom '):
                motor_parts = parts[1:]

            # Check if first part looks like a motor (m0, m1, etc.)
            if motor_parts and motor_parts[0].lower().startswith('m') and len(motor_parts[0]) > 1:
                movements = parse_motor_positions(motor_parts)
                if movements:
                    message = build_message(movements)
                    send_and_wait(arduino, message)
                else:
                    typer.echo("Usage: [custom] m0 100 m1 -50 m2 200")
                continue

            # Otherwise, send as raw ESP32 debug command (STATUS, RESET, etc.)
            arduino.write((user_input + '\n').encode())
            arduino.flush()
            typer.echo(f"Sent: {user_input}")

            # Wait for response
            time.sleep(0.3)
            while arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    typer.echo(f"[ESP32] {line}")

        arduino.close()
        typer.echo("Connection closed.")

    except serial.SerialException as e:
        typer.echo(f"Serial error: {e}", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
