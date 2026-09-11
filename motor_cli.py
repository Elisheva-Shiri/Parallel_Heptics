"""
Motor Control CLI Utility

Drives the ESP32 tactor cluster using the triangular motor protocol
(``ZM<idx>P<pos>...F``). Everything strategy-based goes through the real
``MotorController`` from ``motor_controller.py`` so the geometry matches the
experiment exactly.

Commands:

* ``goto X Y`` — the main entry point. Feed a normalized object *displacement*
  from the cluster center (X and Y in ``[-1, 1]``, centered at 0). It scales the
  input to tracking-frame pixels, then runs it through ``MotorController`` with the
  strategy you pick — cardinal, cardinal_diagonal, free_form, or ik — and sends
  the resulting per-motor commands.
* ``custom`` / ``raw`` — send explicit per-motor positions or a raw command
  string (no kinematics; for bring-up and debugging).
* ``reset`` — send zero (centered) to a cluster's motors.
* ``monitor`` — interactive serial session (holds the port open).

Coordinate systems (do not conflate them):

* **Object displacement (X, Y) — the input.** In tracking-frame pixel units,
  0 = centered. Bounded by the controller's workspace clamp, roughly
  ``min(width, height) / 2 - edge_threshold`` (~210 px with the 640x480
  defaults). May be negative.
* **Motor command ``pos`` — the output.** Integer values inside
  ``ZM<idx>P<pos>F``; the firmware treats them as absolute servo targets in
  ``[-1000, 1000]`` and maps them to servo ticks. This range is unrelated to the
  displacement range above.

Two transports (``--transport``):

* ``serial`` — open the ESP32's COM port directly. No bridge process needed; you
  also get the ESP32's ``OK:``/``E:`` reply back.
* ``udp`` — send the same command string to the ESP32 serial bridge over UDP
  (default port matches ``HARDWARE_PORT``). Matches how ``backend.py`` drives the
  hardware; the bridge must already be running and there is no reply over UDP.
"""

import socket
import time
from enum import Enum
from typing import Optional

import serial
import typer

from consts import EDGE_THRESHOLD, HARDWARE_PORT, STIFFNESS_MAX, TOP_HEIGHT, TOP_WIDTH
from haptic_mapping import map_object_displacement_to_tactor
from motor_controller import (
    HandOrientation,
    MotorController,
    MotorMovement,
    MotorSetId,
    MovementStrategy,
)

app = typer.Typer(help="Motor control utility for the ESP32 tactor cluster")

PORT_HELP = "Serial port, e.g. COM5 (run: uv run list_ports.py). Required for --transport serial."
MOTOR_COUNT = 15
POSITION_LIMIT = 1000


class Transport(str, Enum):
    """How to deliver the command string to the ESP32."""

    serial = "serial"
    udp = "udp"


def build_message(motors: list[MotorMovement]) -> str:
    """Build an ESP32 message from motor movements.

    The raw command intentionally remains available for firmware bring-up, but
    structured commands use ``motor_controller.MotorMovement`` so there is only
    one movement model in the application.
    """
    return "Z" + "".join(f"M{motor.index}P{motor.pos}" for motor in motors) + "F"


def validate_motor_position(index: int, position: int) -> None:
    """Reject structured commands the servo firmware cannot represent safely."""
    if not 0 <= index < MOTOR_COUNT:
        raise ValueError(f"motor index must be 0-{MOTOR_COUNT - 1}; got {index}")
    if not -POSITION_LIMIT <= position <= POSITION_LIMIT:
        raise ValueError(
            f"motor {index} position must be between {-POSITION_LIMIT} and {POSITION_LIMIT}; got {position}"
        )


def validate_motor_movements(motors: list[MotorMovement]) -> None:
    """Validate every movement before a structured command is sent."""
    for motor in motors:
        validate_motor_position(motor.index, motor.pos)


def zero_movements(base_index: int, num_motors: int) -> list[MotorMovement]:
    """Build a bounded set of zero-position commands."""
    if num_motors < 1:
        raise ValueError("number of motors must be at least 1")
    if not 0 <= base_index < MOTOR_COUNT:
        raise ValueError(f"base motor index must be 0-{MOTOR_COUNT - 1}")
    if base_index + num_motors > MOTOR_COUNT:
        raise ValueError(f"requested range exceeds motor {MOTOR_COUNT - 1}")
    return [MotorMovement(pos=0, index=base_index + i) for i in range(num_motors)]


def resolve_gain(stiffness: Optional[float], gain: Optional[float]) -> float:
    """Resolve the displacement multiplier from --stiffness / --gain.

    Two ways to say the same thing, both clamped to safe ranges to avoid
    over-driving the servos:

    * ``stiffness`` — the protocol scale ``0..STIFFNESS_MAX`` (175); 85 = the
      experiment's standard. Normalized internally: ``gain = stiffness / 175``.
    * ``gain`` — the raw multiplier the controller uses, ``0.0..1.0``.

    Give at most one. If neither is given the gain is ``1.0`` (full displacement).
    """
    if stiffness is not None and gain is not None:
        raise typer.BadParameter("Pass either --stiffness or --gain, not both.")

    if gain is not None:
        clamped = max(0.0, min(1.0, gain))
        if clamped != gain:
            typer.echo(
                f"--gain {gain} out of range [0, 1]; clamped to {clamped}.", err=True
            )
        return clamped

    if stiffness is not None:
        clamped = max(0.0, min(float(STIFFNESS_MAX), stiffness))
        if clamped != stiffness:
            typer.echo(
                f"--stiffness {stiffness} out of range [0, {STIFFNESS_MAX}]; clamped to {clamped}.",
                err=True,
            )
        return clamped / STIFFNESS_MAX

    return 1.0


def resolve_displacement(
    x: float, y: float, top_width: float, top_height: float, edge_threshold: float
) -> tuple[float, float]:
    """Map a normalized displacement in [-1, 1] to pixel displacement from center.

    ``(0, 0)`` is the center and ``±1`` is the reachable workspace edge. Inputs
    are clamped to ``[-1, 1]`` (with a warning) so a target can't be pushed past
    the workspace. Returns the pixel-space ``(obj_x, obj_y)`` the controller uses.
    """

    def clamp_unit(value: float, axis: str) -> float:
        clamped = max(-1.0, min(1.0, value))
        if clamped != value:
            typer.echo(
                f"{axis} {value} out of range [-1, 1]; clamped to {clamped}.", err=True
            )
        return clamped

    radius = max(1.0, min(top_width, top_height) / 2.0 - edge_threshold)
    return clamp_unit(x, "x") * radius, clamp_unit(y, "y") * radius


def _controller(
    strategy: MovementStrategy,
    spacing: float,
    move_factor: float,
    mirrored: bool,
    top_width: float,
    top_height: float,
    edge_threshold: float,
) -> MotorController:
    """Build a MotorController with the given strategy and geometry."""
    return MotorController(
        movement_strategy=strategy,
        top_width=top_width,
        top_height=top_height,
        edge_threshold=edge_threshold,
        motor_spacing=spacing,
        move_factor=move_factor,
        hand_orientation=HandOrientation.MIRRORED
        if mirrored
        else HandOrientation.NOT_MIRRORED,
    )


def send_to_arduino(
    message: str,
    port: Optional[str],
    baud: int,
    dry_run: bool = False,
    wait_response: bool = True,
):
    """Send message to the ESP32 via serial."""
    if dry_run:
        typer.echo(f"[DRY RUN] Would send via serial {port or 'COM<N>'}: {message}")
        return

    if not port:
        typer.echo(
            "Serial transport needs a port: pass --port COM<N> (run: uv run list_ports.py).",
            err=True,
        )
        raise typer.Exit(1)

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
                line = arduino.readline().decode("utf-8", errors="ignore").strip()
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
                    line = arduino.readline().decode("utf-8", errors="ignore").strip()
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


def send_via_udp(message: str, host: str, udp_port: int, dry_run: bool = False):
    """Send the command string to the ESP32 serial bridge over UDP.

    Mirrors how ``backend.py`` drives the hardware. The bridge forwards the
    bytes verbatim to the ESP32 over serial; there is no reply over UDP.
    """
    if dry_run:
        typer.echo(f"[DRY RUN] Would send via UDP {host}:{udp_port}: {message}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode(), (host, udp_port))
        typer.echo(f"Sent via UDP {host}:{udp_port}: {message}")
        typer.echo(
            "(No reply over UDP — check the bridge log for the ESP32's OK:/E: line.)"
        )
    except OSError as e:
        typer.echo(f"UDP error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        sock.close()


def dispatch(
    message: str,
    transport: Transport,
    port: Optional[str],
    baud: int,
    host: str,
    udp_port: int,
    dry_run: bool = False,
    wait_response: bool = True,
):
    """Send a prebuilt command string over the selected transport."""
    if transport == Transport.udp:
        send_via_udp(message, host, udp_port, dry_run)
    else:
        send_to_arduino(message, port, baud, dry_run, wait_response)


@app.command()
def goto(
    x: float = typer.Argument(
        ..., help="Normalized X displacement -1..1 (0 = center, ±1 = workspace edge)"
    ),
    y: float = typer.Argument(
        ..., help="Normalized Y displacement -1..1 (0 = center, ±1 = workspace edge)"
    ),
    strategy: MovementStrategy = typer.Option(
        MovementStrategy.IK,
        "--strategy",
        help="Movement strategy: cardinal, cardinal_diagonal, free_form, ik",
    ),
    motor_set: int = typer.Option(
        0, "--motor-set", help="Motor cluster 0-4 (channels 0-2, 3-5, 6-8, 9-11, 12-14)"
    ),
    spacing: float = typer.Option(
        1000.0, "--spacing", "-s", help="Motor spacing (controller units)"
    ),
    move_factor: float = typer.Option(
        1.0, "--move-factor", help="Global gain applied to all output deltas"
    ),
    stiffness: Optional[float] = typer.Option(
        None,
        "--stiffness",
        help=f"Protocol stiffness 0..{STIFFNESS_MAX} (85 = standard); normalized to a gain. Clamped.",
    ),
    gain: Optional[float] = typer.Option(
        None,
        "--gain",
        help="Raw displacement multiplier 0.0..1.0 (alternative to --stiffness; gain = stiffness/175). Clamped.",
    ),
    mirrored: bool = typer.Option(
        False,
        "--mirror/--no-mirror",
        help="Mirror left/right (X only), for hand orientation; off by default",
    ),
    oppose: bool = typer.Option(
        True,
        "--oppose/--no-oppose",
        help="Apply the experiment's object->tactor opposition (on by default, so tactor motion "
        "matches the experiment). Use --no-oppose for raw geometry with no opposition.",
    ),
    top_width: float = typer.Option(
        TOP_WIDTH, "--top-width", help="Frame width (sets the workspace clamp)"
    ),
    top_height: float = typer.Option(
        TOP_HEIGHT, "--top-height", help="Frame height (sets the workspace clamp)"
    ),
    edge_threshold: float = typer.Option(
        EDGE_THRESHOLD, "--edge-threshold", help="Workspace-edge clamp margin"
    ),
    transport: Transport = typer.Option(
        Transport.serial, "--transport", "-t", help="serial or udp"
    ),
    port: Optional[str] = typer.Option(None, "--port", "-p", help=PORT_HELP),
    baud: int = typer.Option(
        115200, "--baud", "-b", help="Baud rate (transport=serial)"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bridge host (transport=udp)"),
    udp_port: int = typer.Option(
        HARDWARE_PORT, "--udp-port", help="Bridge UDP port (transport=udp)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Print command without sending"
    ),
):
    """
    Move a tactor cluster to a normalized displacement using the real MotorController.

    Give a displacement in normalized units where ``(0, 0)`` is the center and
    ``±1`` is the workspace edge (values may be negative and are clamped to
    ``[-1, 1]``). It computes the per-motor commands with the chosen strategy —
    including inverse kinematics — and sends them over serial or the UDP bridge.

    By default the tactor opposes the object motion (the thimble resists);
    pass ``--no-oppose`` for raw geometry (no opposition).

    Note: negative coordinates are read as option flags, so put options first,
    then ``--``, then the coordinates.

    Examples:
        motor_cli.py goto 0.6 0 --strategy free_form --motor-set 1 -p COM<N>
        motor_cli.py goto --strategy ik --dry-run -- 0.5 -0.2
        motor_cli.py goto 0.5 0.2 --strategy ik --transport udp   # via the bridge
    """
    controller = _controller(
        strategy, spacing, move_factor, mirrored, top_width, top_height, edge_threshold
    )
    gain_value = resolve_gain(stiffness, gain)

    try:
        motor_set_id = MotorSetId(motor_set)
    except ValueError:
        raise typer.BadParameter("motor_set must be 0-4")

    px, py = resolve_displacement(x, y, top_width, top_height, edge_threshold)
    obj_x, obj_y = px, py
    if oppose:
        obj_x, obj_y = map_object_displacement_to_tactor(
            obj_x=px, obj_y=py, oppose_motion=True
        )

    try:
        movements = controller.calculate_motor_movements(
            motor_set_id=motor_set_id,
            stiffness_value=gain_value,
            obj_x=obj_x,
            obj_y=obj_y,
            motors_enabled=True,
        )
    except ValueError as e:
        # e.g. an IK target outside the reachable workspace
        typer.echo(
            f"Strategy '{strategy.value}' could not solve target ({x}, {y}): {e}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Strategy={strategy.value}  displacement=({x}, {y})  motor_set={motor_set} (channels {motor_set_id.label})"
    )
    if not movements:
        movements = controller.zero_motor_positions(motor_set_id)
        typer.echo("Displacement resolved to center; sending zero positions.")
    try:
        validate_motor_movements(movements)
    except ValueError as exc:
        typer.echo(f"Unsafe motor command: {exc}", err=True)
        raise typer.Exit(1) from exc
    for m in movements:
        typer.echo(f"  Motor {m.index}: position {m.pos}")

    message = build_message(movements)
    dispatch(message, transport, port, baud, host, udp_port, dry_run)


@app.command()
def custom(
    port: Optional[str] = typer.Option(None, "--port", "-p", help=PORT_HELP),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    m0: Optional[int] = typer.Option(None, "--m0", help="Motor 0 position"),
    m1: Optional[int] = typer.Option(None, "--m1", help="Motor 1 position"),
    m2: Optional[int] = typer.Option(None, "--m2", help="Motor 2 position"),
    m3: Optional[int] = typer.Option(None, "--m3", help="Motor 3 position"),
    m4: Optional[int] = typer.Option(None, "--m4", help="Motor 4 position"),
    m5: Optional[int] = typer.Option(None, "--m5", help="Motor 5 position"),
    m6: Optional[int] = typer.Option(None, "--m6", help="Motor 6 position"),
    m7: Optional[int] = typer.Option(None, "--m7", help="Motor 7 position"),
    m8: Optional[int] = typer.Option(None, "--m8", help="Motor 8 position"),
    m9: Optional[int] = typer.Option(None, "--m9", help="Motor 9 position"),
    m10: Optional[int] = typer.Option(None, "--m10", help="Motor 10 position"),
    m11: Optional[int] = typer.Option(None, "--m11", help="Motor 11 position"),
    m12: Optional[int] = typer.Option(None, "--m12", help="Motor 12 position"),
    m13: Optional[int] = typer.Option(None, "--m13", help="Motor 13 position"),
    m14: Optional[int] = typer.Option(None, "--m14", help="Motor 14 position"),
    transport: Transport = typer.Option(
        Transport.serial, "--transport", "-t", help="serial or udp"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bridge host (transport=udp)"),
    udp_port: int = typer.Option(
        HARDWARE_PORT, "--udp-port", help="Bridge UDP port (transport=udp)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Print command without sending"
    ),
):
    """
    Send explicit positions to specific motors (no kinematics).

    Positions are absolute servo targets in [-1000, 1000].

    Examples:
        motor_cli.py custom --m0 100 --m1 -50 --m2 200
        motor_cli.py custom --m3 500 --m4 500 --m5 500
    """
    motor_options = [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]

    movements = []
    for idx, pos in enumerate(motor_options):
        if pos is not None:
            try:
                validate_motor_position(idx, pos)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint=f"--m{idx}") from exc
            movements.append(MotorMovement(pos=pos, index=idx))

    if not movements:
        typer.echo("Error: At least one motor position must be specified", err=True)
        raise typer.Exit(1)

    message = build_message(movements)

    typer.echo("Custom motor positions:")
    for m in movements:
        typer.echo(f"  Motor {m.index}: position {m.pos}")

    dispatch(message, transport, port, baud, host, udp_port, dry_run)


@app.command()
def raw(
    message: str = typer.Argument(
        ..., help="Raw message to send (without Z prefix and F suffix)"
    ),
    port: Optional[str] = typer.Option(None, "--port", "-p", help=PORT_HELP),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    transport: Transport = typer.Option(
        Transport.serial, "--transport", "-t", help="serial or udp"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bridge host (transport=udp)"),
    udp_port: int = typer.Option(
        HARDWARE_PORT, "--udp-port", help="Bridge UDP port (transport=udp)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Print command without sending"
    ),
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
    dispatch(full_message, transport, port, baud, host, udp_port, dry_run)


@app.command()
def reset(
    port: Optional[str] = typer.Option(None, "--port", "-p", help=PORT_HELP),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    num_motors: int = typer.Option(3, "--num", help="Number of motors to reset"),
    base_index: int = typer.Option(
        0, "--base", help="Base motor index (0, 3, 6, 9, 12)"
    ),
    transport: Transport = typer.Option(
        Transport.serial, "--transport", "-t", help="serial or udp"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bridge host (transport=udp)"),
    udp_port: int = typer.Option(
        HARDWARE_PORT, "--udp-port", help="Bridge UDP port (transport=udp)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Print command without sending"
    ),
):
    """
    Reset motors to zero (centered) position.

    Examples:
        motor_cli.py reset
        motor_cli.py reset --num 6            # Reset 6 motors
        motor_cli.py reset --base 3 --num 3   # Reset motors 3, 4, 5
    """
    try:
        movements = zero_movements(base_index, num_motors)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    message = build_message(movements)

    typer.echo(
        f"Resetting motors {base_index} to {base_index + num_motors - 1} to zero"
    )
    dispatch(message, transport, port, baud, host, udp_port, dry_run)


def send_and_wait(arduino: serial.Serial, message: str, timeout: float = 5.0):
    """Send a message and wait for response."""
    arduino.write(message.encode())
    arduino.flush()
    typer.echo(f"Sent: {message}")

    # Wait for response
    start_time = time.time()
    while time.time() - start_time < timeout:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode("utf-8", errors="ignore").strip()
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
        if motor_part.startswith("m") and motor_part[1:].isdigit():
            try:
                motor_idx = int(motor_part[1:])
                position = int(parts[i + 1])
                validate_motor_position(motor_idx, position)
                movements.append(MotorMovement(pos=position, index=motor_idx))
                i += 2
            except ValueError as exc:
                typer.echo(f"Error: {exc}")
                return []
        else:
            i += 1
    return movements


def print_monitor_help(strategy: MovementStrategy, motor_set: int):
    """Print help for monitor mode commands."""
    typer.echo(f"""
Monitor Mode Commands:
  Strategy movement (uses MotorController, strategy={strategy.value}, motor_set={motor_set}):
    goto 0.5 -0.2        Move cluster to normalized displacement (X, Y), -1..1

  Direct motor positions (pairs of motor and position, no kinematics):
    m0 100 m1 200        Direct motor positions
    custom m0 100 m1 -50 Same as above (custom prefix optional)

  Raw motor command:
    raw M0P100M1P200     Send raw motor string (Z/F added automatically)

  Utilities:
    zero                 Reset the active cluster's motors to position 0
    zero 6               Reset 6 motors from the cluster base

  ESP32 debug commands:
    STATUS               Show servo status
    RESET                Center all servos
    CAL [SHOW]           Show/adjust calibration
    HELP                 Show ESP32 help

  Monitor control:
    help, ?              Show this help
    quit, exit, q        Exit monitor mode
""")


@app.command()
def monitor(
    port: str = typer.Option(
        ..., "--port", "-p", help="Serial port, e.g. COM5 (run: uv run list_ports.py)"
    ),
    baud: int = typer.Option(115200, "--baud", "-b", help="Baud rate"),
    strategy: MovementStrategy = typer.Option(
        MovementStrategy.IK, "--strategy", help="Strategy for the goto command"
    ),
    motor_set: int = typer.Option(
        0, "--motor-set", help="Motor cluster 0-4 for goto/zero"
    ),
    spacing: float = typer.Option(
        1000.0, "--spacing", "-s", help="Motor spacing (controller units)"
    ),
    move_factor: float = typer.Option(
        1.0, "--move-factor", help="Global gain on output deltas"
    ),
    stiffness: Optional[float] = typer.Option(
        None,
        "--stiffness",
        help=f"Protocol stiffness 0..{STIFFNESS_MAX} (85 = standard) for goto; normalized. Clamped.",
    ),
    gain: Optional[float] = typer.Option(
        None,
        "--gain",
        help="Raw displacement multiplier 0.0..1.0 for goto (alternative to --stiffness). Clamped.",
    ),
    mirrored: bool = typer.Option(
        False,
        "--mirror/--no-mirror",
        help="Mirror left/right (X only), for hand orientation; off by default",
    ),
    oppose: bool = typer.Option(
        True,
        "--oppose/--no-oppose",
        help="Oppose the object motion for goto (on by default; matches the experiment)",
    ),
    top_width: float = typer.Option(
        TOP_WIDTH, "--top-width", help="Frame width (workspace clamp)"
    ),
    top_height: float = typer.Option(
        TOP_HEIGHT, "--top-height", help="Frame height (workspace clamp)"
    ),
    edge_threshold: float = typer.Option(
        EDGE_THRESHOLD, "--edge-threshold", help="Workspace-edge clamp margin"
    ),
):
    """
    Interactive monitor mode - keeps the connection open to avoid ESP32 resets.

    Serial-only (it holds the COM port open). Supports strategy movement via
    ``goto X Y`` plus direct/raw/zero commands and ESP32 debug commands.

    Examples:
        motor_cli.py monitor -p COM<N> --strategy ik
    """
    try:
        motor_set_id = MotorSetId(motor_set)
    except ValueError:
        raise typer.BadParameter("motor_set must be 0-4")
    controller = _controller(
        strategy, spacing, move_factor, mirrored, top_width, top_height, edge_threshold
    )
    gain_value = resolve_gain(stiffness, gain)
    base_index = motor_set_id.base_index

    try:
        arduino = serial.Serial(port=port, baudrate=baud, timeout=0.1)
        typer.echo(f"Connected to {port}")
        typer.echo("Waiting for ESP32 to initialize...")

        # Wait for and display boot messages
        time.sleep(0.5)
        while arduino.in_waiting > 0:
            line = arduino.readline().decode("utf-8", errors="ignore").strip()
            if line:
                typer.echo(f"  {line}")

        # Wait for motor detection to complete
        time.sleep(3)
        while arduino.in_waiting > 0:
            line = arduino.readline().decode("utf-8", errors="ignore").strip()
            if line:
                typer.echo(f"  {line}")

        typer.echo(
            f"\nReady! strategy={strategy.value}, motor_set={motor_set}. Type 'help' for commands, 'quit' to exit.\n"
        )

        while True:
            # Check for incoming data from ESP32
            while arduino.in_waiting > 0:
                line = arduino.readline().decode("utf-8", errors="ignore").strip()
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
            if lower_input in ("quit", "exit", "q"):
                break

            # Help command
            if lower_input in ("help", "?"):
                print_monitor_help(strategy, motor_set)
                continue

            # Strategy movement: "goto X Y"
            if lower_input.startswith("goto"):
                if len(parts) >= 3:
                    try:
                        in_x = float(parts[1])
                        in_y = float(parts[2])
                        obj_x, obj_y = resolve_displacement(
                            in_x, in_y, top_width, top_height, edge_threshold
                        )
                        if oppose:
                            obj_x, obj_y = map_object_displacement_to_tactor(
                                obj_x=obj_x, obj_y=obj_y, oppose_motion=True
                            )
                        movements = controller.calculate_motor_movements(
                            motor_set_id=motor_set_id,
                            stiffness_value=gain_value,
                            obj_x=obj_x,
                            obj_y=obj_y,
                            motors_enabled=True,
                        )
                        if not movements:
                            movements = controller.zero_motor_positions(motor_set_id)
                            typer.echo(
                                "Displacement resolved to center; sending zero positions."
                            )
                        validate_motor_movements(movements)
                        message = build_message(movements)
                        typer.echo(f"goto ({in_x}, {in_y}) via {strategy.value}:")
                        for m in movements:
                            typer.echo(f"  Motor {m.index}: {m.pos}")
                        send_and_wait(arduino, message)
                    except ValueError as e:
                        typer.echo(f"Error: {e}")
                else:
                    typer.echo("Usage: goto <x> <y>   (normalized displacement, -1..1)")
                continue

            # Raw command: "raw M0P100M1P200"
            if lower_input.startswith("raw "):
                raw_msg = user_input[4:].strip()
                message = f"Z{raw_msg}F"
                send_and_wait(arduino, message)
                continue

            # Zero command: "zero" or "zero 6"
            if lower_input.startswith("zero"):
                num_motors = 3
                if len(parts) >= 2:
                    try:
                        num_motors = int(parts[1])
                    except ValueError:
                        pass
                try:
                    movements = zero_movements(base_index, num_motors)
                except ValueError as exc:
                    typer.echo(f"Error: {exc}")
                    continue
                message = build_message(movements)
                typer.echo(
                    f"Zeroing motors {base_index} to {base_index + num_motors - 1}"
                )
                send_and_wait(arduino, message)
                continue

            # Direct motor format: "m0 100 m1 200" or "custom m0 100 m1 200"
            motor_parts = parts
            if lower_input.startswith("custom "):
                motor_parts = parts[1:]

            if (
                motor_parts
                and motor_parts[0].lower().startswith("m")
                and len(motor_parts[0]) > 1
            ):
                movements = parse_motor_positions(motor_parts)
                if movements:
                    message = build_message(movements)
                    send_and_wait(arduino, message)
                else:
                    typer.echo("Usage: [custom] m0 100 m1 -50 m2 200")
                continue

            # Otherwise, send as raw ESP32 debug command (STATUS, RESET, etc.)
            arduino.write((user_input + "\n").encode())
            arduino.flush()
            typer.echo(f"Sent: {user_input}")

            # Wait for response
            time.sleep(0.3)
            while arduino.in_waiting > 0:
                line = arduino.readline().decode("utf-8", errors="ignore").strip()
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
