"""OptiTrack-aligned physical workspace validation runner.

The OptiTrack system is operated from another computer.  This script records a
local camera video and a timestamped motor/event log so those streams can be
aligned offline.  Motion is intentionally numeric: it sends small interpolated
steps along eight cardinal/diagonal axes using the ESP32 ``ZM...F`` protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import serial
import cv2
from serial.tools import list_ports


REPO_ROOT = Path(__file__).resolve().parents[2]
OPTITRACK_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validetion.motor_response_analizer_servo.camera_recorder import (  # noqa: E402
    CameraConfig,
    CameraRecorder,
)
from validetion.motor_response_analizer_servo.motor_io import (  # noqa: E402
    DryRunMotor,
    MotorResponse,
    MotorSerial,
)


OBSERVED_RELATIVE_BOUNDS: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
    (-672, 686),
    (-679, 693),
    (-679, 693),
)
AXIS_ORDER_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
AXIS_ORDER_16 = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)
DEFAULT_AXIS_COUNT = 8
DEFAULT_DIRECTION_ORDER = AXIS_ORDER_8
CIRCLE_CCW_ORDER = (
    "E",
    "ENE",
    "NE",
    "NNE",
    "N",
    "NNW",
    "NW",
    "WNW",
    "W",
    "WSW",
    "SW",
    "SSW",
    "S",
    "SSE",
    "SE",
    "ESE",
    "E",
)
TRAJECTORY_AXES = "axes"
TRAJECTORY_OUTER_CIRCLE = "outer-circle"
TRAJECTORY_ALIASES = {
    "axes": TRAJECTORY_AXES,
    "axis": TRAJECTORY_AXES,
    "outer-circle": TRAJECTORY_OUTER_CIRCLE,
    "outer-circuil": TRAJECTORY_OUTER_CIRCLE,
    "outer-circuit": TRAJECTORY_OUTER_CIRCLE,
    "outer_circle": TRAJECTORY_OUTER_CIRCLE,
    "circle": TRAJECTORY_OUTER_CIRCLE,
    "circuil": TRAJECTORY_OUTER_CIRCLE,
    "perimeter": TRAJECTORY_OUTER_CIRCLE,
}
DEFAULT_PERIMETER_POINTS = 301
EVENT_FIELDS = (
    "wall_time_iso",
    "monotonic_s",
    "event",
    "axis",
    "pose_index",
    "label",
    "command",
    "targets_json",
    "actuals_json",
    "serial_raw",
    "elapsed_s",
    "note",
)
OK_POSITION_PATTERN = re.compile(r"M(\d+)P(-?\d+)")


@dataclass(frozen=True)
class Pose:
    """One numeric workspace pose."""

    axis: str
    label: str
    targets: dict[int, int]


@dataclass(frozen=True)
class ValidationConfig:
    """Runtime configuration for a validation run."""

    port: str = "COM4"
    baud: int = 115200
    camera_index: int = 0
    base_index: int = 0
    output_dir: Path = OPTITRACK_DIR / "result"
    dry_run: bool = False
    no_camera: bool = False
    auto_start: bool = False
    width: int | None = None
    height: int | None = None
    fps: float = 0.0
    hold_s: float = 0.15
    step_ticks: int = 25
    max_duration_s: float = 180.0
    full_range: bool = True
    range_scale: float = 1.0
    movement_scale: float = 1.05
    axis_count: int = DEFAULT_AXIS_COUNT
    trajectory: str = TRAJECTORY_AXES
    perimeter_points: int = DEFAULT_PERIMETER_POINTS


class MotorLike(Protocol):
    """Subset shared by MotorSerial and DryRunMotor."""

    def open(self) -> list[str]: ...
    def close(self) -> None: ...
    def send_command(self, message: str, motor_index: int) -> MotorResponse: ...


class KeySource(Protocol):
    """Non-blocking source of single-character operator commands."""

    def get_key(self) -> str | None: ...
    def close(self) -> None: ...


class NullKeySource:
    """Key source used for auto-started non-interactive runs."""

    def get_key(self) -> str | None:
        return None

    def close(self) -> None:
        return None


class ConsoleKeySource:
    """Cross-platform non-blocking console key reader."""

    def __init__(self):
        self._closed = threading.Event()
        self._keys: queue.Queue[str] = queue.Queue()
        if sys.platform.startswith("win"):
            self._thread: threading.Thread | None = None
        else:
            self._thread = threading.Thread(target=self._stdin_loop, daemon=True)
            self._thread.start()

    def get_key(self) -> str | None:
        if sys.platform.startswith("win"):
            return self._get_windows_key()
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed.set()

    def _get_windows_key(self) -> str | None:
        try:
            import msvcrt
        except ImportError:
            return None
        if not msvcrt.kbhit():
            return None
        key = msvcrt.getwch()
        return key.lower() if key else None

    def _stdin_loop(self) -> None:
        while not self._closed.is_set():
            char = sys.stdin.read(1)
            if not char:
                break
            self._keys.put(char.lower())


class ControlState:
    """State machine for `s` start/stop and `p` pause/resume."""

    def __init__(self):
        self.started = False
        self.paused = False
        self.stop_requested = False

    def handle_key(self, key: str | None) -> str | None:
        if key is None:
            return None
        key = key.lower()
        if key == "s":
            if not self.started:
                self.started = True
                return "START"
            self.stop_requested = True
            self.paused = False
            return "STOP"
        if key == "p" and self.started and not self.stop_requested:
            self.paused = not self.paused
            return "PAUSE" if self.paused else "RESUME"
        return None


class EventLogger:
    """Write CSV event records and keep a small in-memory summary."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, str]] = []
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=EVENT_FIELDS)
        self._writer.writeheader()

    def close(self) -> None:
        self._file.close()

    def log(
        self,
        event: str,
        *,
        axis: str = "",
        pose_index: int | None = None,
        label: str = "",
        command: str = "",
        targets: dict[int, int] | None = None,
        actuals: dict[int, int] | None = None,
        serial_raw: str = "",
        elapsed_s: float | None = None,
        note: str = "",
    ) -> None:
        row = {
            "wall_time_iso": datetime.now(timezone.utc).isoformat(),
            "monotonic_s": f"{time.monotonic():.6f}",
            "event": event,
            "axis": axis,
            "pose_index": "" if pose_index is None else str(pose_index),
            "label": label,
            "command": command,
            "targets_json": json.dumps(targets or {}, sort_keys=True),
            "actuals_json": json.dumps(actuals or {}, sort_keys=True),
            "serial_raw": serial_raw.replace("\r", "\\r").replace("\n", "\\n"),
            "elapsed_s": "" if elapsed_s is None else f"{elapsed_s:.6f}",
            "note": note,
        }
        self._writer.writerow(row)
        self._file.flush()
        self.events.append(row)


class SerialCommandError(RuntimeError):
    """Raised when a motor command did not receive a complete OK response."""

    def __init__(
        self,
        message: str,
        *,
        raw: str,
        actuals: dict[int, int],
        elapsed_s: float,
    ):
        super().__init__(message)
        self.raw = raw
        self.actuals = actuals
        self.elapsed_s = elapsed_s


def parse_bounds(text: str, base_index: int) -> dict[int, tuple[int, int]]:
    """Parse custom bounds as `min:max,min:max,min:max` for the motor triplet."""

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("custom bounds must contain exactly three min:max pairs")
    bounds: dict[int, tuple[int, int]] = {}
    for offset, part in enumerate(parts):
        raw_min, raw_max = part.split(":", maxsplit=1)
        low = int(raw_min)
        high = int(raw_max)
        if low >= high:
            raise ValueError(f"invalid bounds {part!r}: min must be lower than max")
        bounds[base_index + offset] = (low, high)
    return bounds


def observed_bounds_for_base(base_index: int) -> dict[int, tuple[int, int]]:
    """Return observed sample bounds mapped to `base_index..base_index+2`."""

    return {
        base_index + offset: motor_bounds
        for offset, motor_bounds in enumerate(OBSERVED_RELATIVE_BOUNDS)
    }


def scale_bounds(
    bounds: dict[int, tuple[int, int]],
    *,
    range_scale: float,
    full_range: bool,
) -> dict[int, tuple[int, int]]:
    """Scale observed bounds around zero unless explicit full range is requested."""

    if full_range:
        return dict(bounds)
    if not 0 < range_scale <= 0.6:
        raise ValueError("range_scale must be > 0 and <= 0.6 unless --full-range is used")
    return {
        motor: (int(round(low * range_scale)), int(round(high * range_scale)))
        for motor, (low, high) in bounds.items()
    }


def expand_bounds(
    bounds: dict[int, tuple[int, int]],
    *,
    movement_scale: float,
) -> dict[int, tuple[int, int]]:
    """Expand configured movement bounds around zero.

    The user requested an extra 5% beyond the observed sample range.  We round
    outward so the added range is never lost to integer rounding.
    """

    if movement_scale <= 0:
        raise ValueError("movement_scale must be positive")
    return {
        motor: (math.floor(low * movement_scale), math.ceil(high * movement_scale))
        for motor, (low, high) in bounds.items()
    }


def clamp_targets(
    targets: dict[int, int],
    bounds: dict[int, tuple[int, int]],
) -> tuple[dict[int, int], bool]:
    """Clamp targets to configured bounds and report if any value changed."""

    clamped = {}
    changed = False
    for motor, target in targets.items():
        low, high = bounds[motor]
        value = min(high, max(low, target))
        clamped[motor] = value
        changed = changed or value != target
    return clamped, changed


def build_multi_motor_command(targets: dict[int, int]) -> str:
    """Format one ESP32 multi-motor command string."""

    return "Z" + "".join(f"M{motor}P{targets[motor]}" for motor in sorted(targets)) + "F"


def parse_motor_positions(text: str) -> dict[int, int]:
    """Parse all `M{idx}P{actual}` pairs from an OK/raw response string."""

    return {int(motor): int(pos) for motor, pos in OK_POSITION_PATTERN.findall(text)}


def synthesize_triple_ok(targets: dict[int, int]) -> str:
    """Build a dry-run response that mirrors firmware multi-motor OK lines."""

    return "OK:" + ",".join(f"M{motor}P{targets[motor]}" for motor in sorted(targets))


def midpoint_endpoint(first: dict[int, int], second: dict[int, int]) -> dict[int, int]:
    """Return the numeric midpoint between two neighboring axis endpoints."""

    return {
        motor: int(round((first[motor] + second[motor]) / 2))
        for motor in sorted(first)
    }


def axis_endpoints(bounds: dict[int, tuple[int, int]], base_index: int) -> dict[str, dict[int, int]]:
    """Build numeric endpoints for cardinal/diagonal and half-step axes.

    The signs follow the triangular three-motor layout used by the existing
    motor CLI: north pulls M0 negative while M1/M2 go positive; south is the
    inverse; east/west are opposing M1/M2.  Diagonals combine those bases while
    staying on the observed numeric axes.  The 16-axis mode inserts numeric
    midpoints between each neighboring 8-axis endpoint.
    """

    m0, m1, m2 = base_index, base_index + 1, base_index + 2
    n0, p0 = bounds[m0]
    n1, p1 = bounds[m1]
    n2, p2 = bounds[m2]
    endpoints = {
        "N": {m0: n0, m1: p1, m2: p2},
        "NE": {m0: n0, m1: p1, m2: 0},
        "E": {m0: 0, m1: p1, m2: n2},
        "SE": {m0: p0, m1: 0, m2: n2},
        "S": {m0: p0, m1: n1, m2: n2},
        "SW": {m0: p0, m1: n1, m2: 0},
        "W": {m0: 0, m1: n1, m2: p2},
        "NW": {m0: n0, m1: 0, m2: p2},
    }
    endpoints.update(
        {
            "NNE": midpoint_endpoint(endpoints["N"], endpoints["NE"]),
            "ENE": midpoint_endpoint(endpoints["NE"], endpoints["E"]),
            "ESE": midpoint_endpoint(endpoints["E"], endpoints["SE"]),
            "SSE": midpoint_endpoint(endpoints["SE"], endpoints["S"]),
            "SSW": midpoint_endpoint(endpoints["S"], endpoints["SW"]),
            "WSW": midpoint_endpoint(endpoints["SW"], endpoints["W"]),
            "WNW": midpoint_endpoint(endpoints["W"], endpoints["NW"]),
            "NNW": midpoint_endpoint(endpoints["NW"], endpoints["N"]),
        }
    )
    return endpoints


def direction_order(axis_count: int) -> tuple[str, ...]:
    """Return the clockwise direction order for 8- or 16-axis validation."""

    if axis_count == 8:
        return AXIS_ORDER_8
    if axis_count == 16:
        return AXIS_ORDER_16
    raise ValueError("axis_count must be 8 or 16")


def normalize_trajectory(value: str) -> str:
    """Normalize CLI trajectory names and common typos."""

    normalized = value.strip().lower()
    try:
        return TRAJECTORY_ALIASES[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(TRAJECTORY_ALIASES))
        raise ValueError(f"unknown trajectory {value!r}; choose one of: {valid}") from exc


def build_outer_circle_perimeter(
    bounds: dict[int, tuple[int, int]],
    *,
    base_index: int,
    perimeter_points: int = DEFAULT_PERIMETER_POINTS,
) -> list[Pose]:
    """Generate perimeter poses from right/E counterclockwise back to right/E.

    The 301 generated points are distributed around the 16-axis outer polygon,
    which gives a smooth outer boundary while keeping the path aligned with the
    same cardinal/diagonal command space as the axis validation modes.
    """

    if perimeter_points < 2:
        raise ValueError("perimeter_points must be at least 2")
    endpoints = axis_endpoints(bounds, base_index)
    segment_count = len(CIRCLE_CCW_ORDER) - 1
    poses: list[Pose] = []
    for point_index in range(perimeter_points):
        progress = (point_index / (perimeter_points - 1)) * segment_count
        segment_index = min(segment_count - 1, int(math.floor(progress)))
        fraction = progress - segment_index
        start_axis = CIRCLE_CCW_ORDER[segment_index]
        end_axis = CIRCLE_CCW_ORDER[segment_index + 1]
        start = endpoints[start_axis]
        end = endpoints[end_axis]
        targets = {
            motor: int(round(start[motor] + (end[motor] - start[motor]) * fraction))
            for motor in sorted(start)
        }
        poses.append(
            Pose(
                axis="OUTER_CIRCLE",
                label=f"outer_circle_ccw_{point_index:03d}_{start_axis}_to_{end_axis}",
                targets=targets,
            )
        )
    return poses


def build_center_to_endpoint_path(
    *,
    axis: str,
    endpoint: dict[int, int],
    step_ticks: int,
    outward: bool,
) -> list[Pose]:
    """Build center-to-endpoint or endpoint-to-center path with small steps."""

    if step_ticks <= 0:
        raise ValueError("step_ticks must be positive")
    max_abs = max(abs(value) for value in endpoint.values())
    steps = max(1, math.ceil(max_abs / step_ticks))
    fractions = range(steps + 1) if outward else range(steps - 1, -1, -1)
    label_mid = "to_right" if outward else "to_center"
    return [
        Pose(axis=axis, label=f"{axis.lower()}_{label_mid}_{idx:03d}", targets=_interpolate(endpoint, idx / steps))
        for idx in fractions
    ]


def build_outer_circle_sequence(
    bounds: dict[int, tuple[int, int]],
    *,
    base_index: int,
    step_ticks: int = 25,
    perimeter_points: int = DEFAULT_PERIMETER_POINTS,
) -> list[Pose]:
    """Build center -> right -> 301-point CCW perimeter -> right -> center."""

    endpoint = axis_endpoints(bounds, base_index)["E"]
    ramp_out = build_center_to_endpoint_path(
        axis="OUTER_CIRCLE",
        endpoint=endpoint,
        step_ticks=step_ticks,
        outward=True,
    )
    perimeter = build_outer_circle_perimeter(
        bounds,
        base_index=base_index,
        perimeter_points=perimeter_points,
    )
    ramp_in = build_center_to_endpoint_path(
        axis="OUTER_CIRCLE",
        endpoint=endpoint,
        step_ticks=step_ticks,
        outward=False,
    )
    return ramp_out + perimeter[1:] + ramp_in


def _interpolate(endpoint: dict[int, int], fraction: float) -> dict[int, int]:
    return {motor: int(round(value * fraction)) for motor, value in endpoint.items()}


def build_axis_path(axis: str, endpoint: dict[int, int], step_ticks: int) -> list[Pose]:
    """Create center -> endpoint -> center poses with <= step tick jumps."""

    if step_ticks <= 0:
        raise ValueError("step_ticks must be positive")
    max_abs = max(abs(value) for value in endpoint.values())
    steps = max(1, math.ceil(max_abs / step_ticks))
    poses: list[Pose] = []
    for idx in range(steps + 1):
        targets = _interpolate(endpoint, idx / steps)
        poses.append(Pose(axis=axis, label=f"{axis}_out_{idx:03d}", targets=targets))
    for idx in range(steps - 1, -1, -1):
        targets = _interpolate(endpoint, idx / steps)
        poses.append(Pose(axis=axis, label=f"{axis}_in_{steps - idx:03d}", targets=targets))
    return poses


def build_workspace_sequence(
    bounds: dict[int, tuple[int, int]],
    *,
    base_index: int,
    step_ticks: int = 25,
    directions: Iterable[str] | None = None,
    axis_count: int = DEFAULT_AXIS_COUNT,
) -> list[Pose]:
    """Build the complete validation sequence for 8 or 16 axes."""

    endpoints = axis_endpoints(bounds, base_index)
    selected_directions = tuple(directions) if directions is not None else direction_order(axis_count)
    sequence: list[Pose] = []
    previous_targets: dict[int, int] | None = None
    for direction in selected_directions:
        path = build_axis_path(direction, endpoints[direction], step_ticks)
        for pose in path:
            if previous_targets == pose.targets:
                continue
            sequence.append(pose)
            previous_targets = pose.targets
    center = {base_index: 0, base_index + 1: 0, base_index + 2: 0}
    if previous_targets != center:
        sequence.append(Pose(axis="CENTER", label="final_center", targets=center))
    return sequence


def build_validation_sequence(
    bounds: dict[int, tuple[int, int]],
    *,
    base_index: int,
    step_ticks: int,
    axis_count: int,
    trajectory: str,
    perimeter_points: int,
) -> list[Pose]:
    """Build the requested validation trajectory."""

    if trajectory == TRAJECTORY_AXES:
        return build_workspace_sequence(
            bounds,
            base_index=base_index,
            step_ticks=step_ticks,
            axis_count=axis_count,
        )
    if trajectory == TRAJECTORY_OUTER_CIRCLE:
        return build_outer_circle_sequence(
            bounds,
            base_index=base_index,
            step_ticks=step_ticks,
            perimeter_points=perimeter_points,
        )
    raise ValueError(f"unknown trajectory {trajectory!r}")


def max_step_delta(sequence: list[Pose]) -> int:
    """Return the largest absolute per-motor jump between adjacent poses."""

    largest = 0
    for previous, current in zip(sequence, sequence[1:]):
        for motor, value in current.targets.items():
            largest = max(largest, abs(value - previous.targets.get(motor, 0)))
    return largest


def estimate_duration_s(sequence: list[Pose], hold_s: float) -> float:
    return len(sequence) * hold_s


def validate_motion_config(
    *,
    sequence: list[Pose],
    hold_s: float,
    step_ticks: int,
    max_duration_s: float,
) -> None:
    """Reject unsafe or nonsensical motion settings before hardware movement."""

    if hold_s <= 0:
        raise ValueError("hold_s must be positive")
    if step_ticks <= 0:
        raise ValueError("step_ticks must be positive")
    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be positive")
    estimated = estimate_duration_s(sequence, hold_s)
    if estimated > max_duration_s:
        raise ValueError(
            f"estimated motion duration {estimated:.2f}s exceeds max_duration_s {max_duration_s:.2f}s"
        )
    largest = max_step_delta(sequence)
    if largest > step_ticks:
        raise ValueError(f"generated step jump {largest} exceeds step_ticks {step_ticks}")


def create_run_dir(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"workspace_validation_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_dir / f"workspace_validation_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def send_triple_command(
    motor: MotorLike,
    targets: dict[int, int],
    *,
    dry_run: bool,
) -> tuple[str, str, dict[int, int], float]:
    """Send/log one multi-motor command and parse/simulate all actuals."""

    command = build_multi_motor_command(targets)
    raw, actuals, elapsed_s = send_prebuilt_triple_command(motor, command, targets, dry_run=dry_run)
    return command, raw, actuals, elapsed_s


def send_prebuilt_triple_command(
    motor: MotorLike,
    command: str,
    targets: dict[int, int],
    *,
    dry_run: bool,
) -> tuple[str, dict[int, int], float]:
    """Send an already logged command and parse/simulate all actuals."""

    if dry_run:
        raw = synthesize_triple_ok(targets)
        return raw, parse_motor_positions(raw), 0.0
    first_motor = min(targets)
    response = motor.send_command(command, first_motor)
    raw = response.raw or response.ok_line or response.error or ""
    actuals = parse_motor_positions(raw)
    if response.error:
        raise SerialCommandError(
            f"firmware returned error for {command}: {response.error}",
            raw=raw,
            actuals=actuals,
            elapsed_s=response.elapsed_s,
        )
    if not response.ok_line:
        raise SerialCommandError(
            f"timeout/no OK response for {command}",
            raw=raw,
            actuals=actuals,
            elapsed_s=response.elapsed_s,
        )
    missing = sorted(set(targets) - set(actuals))
    if missing:
        raise SerialCommandError(
            f"OK response missing actuals for motors {missing}: {command}",
            raw=raw,
            actuals=actuals,
            elapsed_s=response.elapsed_s,
        )
    return raw, actuals, response.elapsed_s


def process_control_key(
    control: ControlState,
    key: str | None,
    logger: EventLogger,
) -> None:
    event = control.handle_key(key)
    if event is not None:
        logger.log(event)


def wait_until_started(
    control: ControlState,
    key_source: KeySource,
    logger: EventLogger,
    *,
    auto_start: bool,
) -> None:
    logger.log("WAITING", note="press 's' to start; press 's' again to stop; 'p' pauses")
    if auto_start:
        process_control_key(control, "s", logger)
        return
    print("Camera/serial armed. Press 's' to start, 'p' to pause/resume, 's' again to stop.")
    while not control.started and not control.stop_requested:
        process_control_key(control, key_source.get_key(), logger)
        time.sleep(0.02)


def controlled_hold(
    duration_s: float,
    control: ControlState,
    key_source: KeySource,
    logger: EventLogger,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Hold for `duration_s`, freezing remaining time while paused.

    Returns False if stop was requested during the hold.
    """

    remaining = duration_s
    last = time.monotonic()
    while remaining > 0:
        process_control_key(control, key_source.get_key(), logger)
        if control.stop_requested:
            return False
        now = time.monotonic()
        elapsed = now - last
        last = now
        if not control.paused:
            remaining -= elapsed
        sleep(min(0.02, max(0.0, remaining)))
    return True


def write_summary(
    run_dir: Path,
    *,
    config: ValidationConfig,
    sequence: list[Pose],
    bounds: dict[int, tuple[int, int]],
    video_path: Path | None,
    completed: bool,
    reset_command: str,
    recording_fps: float | None = None,
    measured_fps: float | None = None,
    driver_fps: float | None = None,
) -> None:
    summary = {
        "completed": completed,
        "port": config.port,
        "baud": config.baud,
        "camera_index": config.camera_index,
        "base_index": config.base_index,
        "dry_run": config.dry_run,
        "no_camera": config.no_camera,
        "full_range": config.full_range,
        "range_scale": config.range_scale,
        "movement_scale": config.movement_scale,
        "axis_count": config.axis_count,
        "trajectory": config.trajectory,
        "perimeter_points": config.perimeter_points,
        "requested_fps": config.fps,
        "recording_fps": recording_fps,
        "measured_fps": measured_fps,
        "driver_fps": driver_fps,
        "hold_s": config.hold_s,
        "step_ticks": config.step_ticks,
        "max_duration_s": config.max_duration_s,
        "estimated_duration_s": estimate_duration_s(sequence, config.hold_s),
        "pose_count": len(sequence),
        "max_step_delta": max_step_delta(sequence),
        "bounds": {str(k): v for k, v in sorted(bounds.items())},
        "directions": (
            list(CIRCLE_CCW_ORDER)
            if config.trajectory == TRAJECTORY_OUTER_CIRCLE
            else list(direction_order(config.axis_count))
        ),
        "video_path": "" if video_path is None else str(video_path),
        "events_path": str(run_dir / "events.csv"),
        "reset_command": reset_command,
        "optitrack_note": "OptiTrack is external; align using wall_time_iso/monotonic_s events and video pre-roll.",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def available_serial_ports() -> list[str]:
    """Return serial ports currently visible to Windows."""

    return [port.device for port in list_ports.comports()]


def serial_port_error_message(port: str, exc: serial.SerialException) -> str:
    """Build a clear operator-facing message for unavailable serial ports."""

    ports = available_serial_ports()
    visible = ", ".join(ports) if ports else "none"
    return (
        f"Could not open serial port {port!r}: {exc}\n"
        f"Visible serial ports: {visible}\n"
        "Plug in/turn on the ESP32 motor controller, check Device Manager, then retry with:\n"
        "  uv run workspace_validation.py --port COMx\n"
        "If you run from the repo root instead, use:\n"
        "  .\\.venv\\Scripts\\python.exe validetion\\optitrack\\workspace_validation.py --port COMx"
    )


def probe_camera_fps(
    *,
    camera_index: int,
    width: int | None,
    height: int | None,
    requested_fps: float,
    sample_s: float = 1.0,
) -> tuple[float, float, float]:
    """Measure camera FPS and return `(recording_fps, measured_fps, driver_fps)`.

    If `requested_fps` is positive, it is applied before measuring.  A zero
    request means "use the camera/driver default".  The returned recording FPS
    is the measured rate when possible, otherwise the driver-reported rate,
    otherwise the explicit request, and finally 30 FPS as a safe MP4 fallback.
    """

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index} for FPS probe")
    try:
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if requested_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, requested_fps)

        for _ in range(5):
            cap.read()
        driver_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        start = time.monotonic()
        deadline = start + sample_s
        frames = 0
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                frames += 1
        elapsed = max(0.001, time.monotonic() - start)
        measured_fps = frames / elapsed if frames else 0.0
        recording_fps = measured_fps or driver_fps or requested_fps or 30.0
        return recording_fps, measured_fps, driver_fps
    finally:
        cap.release()


def run_validation(
    config: ValidationConfig,
    *,
    custom_bounds: dict[int, tuple[int, int]] | None = None,
    key_source: KeySource | None = None,
    motor: MotorLike | None = None,
    camera_factory: Callable[[CameraConfig, Path], CameraRecorder] | None = None,
) -> Path:
    """Run the validation and return the output run directory."""

    raw_bounds = custom_bounds or observed_bounds_for_base(config.base_index)
    scaled_bounds = scale_bounds(raw_bounds, range_scale=config.range_scale, full_range=config.full_range)
    bounds = expand_bounds(scaled_bounds, movement_scale=config.movement_scale)
    sequence = build_validation_sequence(
        bounds,
        base_index=config.base_index,
        step_ticks=config.step_ticks,
        axis_count=config.axis_count,
        trajectory=config.trajectory,
        perimeter_points=config.perimeter_points,
    )
    validate_motion_config(
        sequence=sequence,
        hold_s=config.hold_s,
        step_ticks=config.step_ticks,
        max_duration_s=config.max_duration_s,
    )

    run_dir = create_run_dir(config.output_dir)
    events_path = run_dir / "events.csv"
    video_path = None if config.no_camera else run_dir / "recording.mp4"
    logger = EventLogger(events_path)
    key_source = key_source or (NullKeySource() if config.auto_start else ConsoleKeySource())
    motor = motor or (DryRunMotor(port="DRY", baud=0) if config.dry_run else MotorSerial(config.port, config.baud))
    camera: CameraRecorder | None = None
    reset_command = build_multi_motor_command(
        {config.base_index: 0, config.base_index + 1: 0, config.base_index + 2: 0}
    )
    completed = False
    failed = False
    serial_opened = False
    recording_fps: float | None = None
    measured_fps: float | None = None
    driver_fps: float | None = None

    try:
        logger.log(
            "CONFIG",
            note=json.dumps(
                {
                    "port": config.port,
                    "camera_index": config.camera_index,
                    "base_index": config.base_index,
                    "full_range": config.full_range,
                    "movement_scale": config.movement_scale,
                    "axis_count": config.axis_count,
                    "trajectory": config.trajectory,
                    "perimeter_points": config.perimeter_points,
                    "bounds": {str(k): v for k, v in sorted(bounds.items())},
                    "pose_count": len(sequence),
                },
                sort_keys=True,
            ),
        )
        boot_log = motor.open()
        serial_opened = True
        logger.log("SERIAL_OPEN", note=" | ".join(boot_log))

        if not config.no_camera:
            recording_fps, measured_fps, driver_fps = probe_camera_fps(
                camera_index=config.camera_index,
                width=config.width,
                height=config.height,
                requested_fps=config.fps,
            )
            logger.log(
                "CAMERA_FPS",
                note=json.dumps(
                    {
                        "requested_fps": config.fps,
                        "recording_fps": recording_fps,
                        "measured_fps": measured_fps,
                        "driver_fps": driver_fps,
                    },
                    sort_keys=True,
                ),
            )
            cfg = CameraConfig(
                index=config.camera_index,
                width=config.width,
                height=config.height,
                fps=recording_fps,
            )
            factory = camera_factory or (lambda camera_config, path: CameraRecorder(camera_config, path))
            camera = factory(cfg, video_path)  # type: ignore[arg-type]
            camera.start()
            logger.log("CAMERA_PREROLL", note=str(video_path))

        control = ControlState()
        wait_until_started(control, key_source, logger, auto_start=config.auto_start)
        if control.stop_requested:
            return run_dir

        for pose_index, pose in enumerate(sequence, start=1):
            process_control_key(control, key_source.get_key(), logger)
            if control.stop_requested:
                break
            targets, clamped = clamp_targets(pose.targets, bounds)
            command = build_multi_motor_command(targets)
            logger.log(
                "SERIAL_OUT",
                axis=pose.axis,
                pose_index=pose_index,
                label=pose.label,
                command=command,
                targets=targets,
                note="clamped" if clamped else "",
            )
            try:
                raw, actuals, elapsed_s = send_prebuilt_triple_command(
                    motor,
                    command,
                    targets,
                    dry_run=config.dry_run,
                )
            except SerialCommandError as exc:
                logger.log(
                    "SERIAL_IN",
                    axis=pose.axis,
                    pose_index=pose_index,
                    label=pose.label,
                    command=command,
                    targets=targets,
                    actuals=exc.actuals,
                    serial_raw=exc.raw,
                    elapsed_s=exc.elapsed_s,
                    note="serial command failed",
                )
                logger.log(
                    "ERROR",
                    axis=pose.axis,
                    pose_index=pose_index,
                    label=pose.label,
                    command=command,
                    note=str(exc),
                )
                failed = True
                break
            logger.log(
                "SERIAL_IN",
                axis=pose.axis,
                pose_index=pose_index,
                label=pose.label,
                command=command,
                targets=targets,
                actuals=actuals,
                serial_raw=raw,
                elapsed_s=elapsed_s,
            )
            if not controlled_hold(config.hold_s, control, key_source, logger):
                break
        completed = not control.stop_requested and not failed
        return run_dir
    except KeyboardInterrupt:
        logger.log("INTERRUPTED", note="Ctrl+C")
        return run_dir
    except Exception as exc:
        logger.log("ERROR", note=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if serial_opened:
            try:
                targets = {config.base_index: 0, config.base_index + 1: 0, config.base_index + 2: 0}
                command, raw, actuals, elapsed_s = send_triple_command(motor, targets, dry_run=config.dry_run)
                logger.log(
                    "RESET",
                    command=command,
                    targets=targets,
                    actuals=actuals,
                    serial_raw=raw,
                    elapsed_s=elapsed_s,
                )
                reset_command = command
            except Exception as exc:
                logger.log("RESET_ERROR", command=reset_command, note=f"{type(exc).__name__}: {exc}")
        try:
            motor.close()
        finally:
            if camera is not None:
                camera.stop()
            key_source.close()
            write_summary(
                run_dir,
                config=config,
                sequence=sequence,
                bounds=bounds,
                video_path=video_path,
                completed=completed,
                reset_command=reset_command,
                recording_fps=recording_fps,
                measured_fps=measured_fps,
                driver_fps=driver_fps,
            )
            logger.log("SUMMARY_WRITTEN", note=str(run_dir / "run_summary.json"))
            logger.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--base-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OPTITRACK_DIR / "result",
        help="Directory that receives one timestamped folder per experiment.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--auto-start", action="store_true", help="Start immediately; intended for dry-run tests.")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Requested camera FPS before probing; 0 means use camera default and save at measured FPS.",
    )
    parser.add_argument("--hold-s", type=float, default=0.15, help="Seconds to hold after each 25-tick command.")
    parser.add_argument("--step-ticks", type=int, default=25)
    parser.add_argument(
        "--axis-count",
        type=int,
        choices=(8, 16),
        default=DEFAULT_AXIS_COUNT,
        help="Number of main axes to validate: 8 cardinal/diagonal axes, or 16 with half-step compass axes.",
    )
    parser.add_argument(
        "--trajectory",
        default=TRAJECTORY_AXES,
        help=(
            "Trajectory mode: axes uses --axis-count; outer-circle uses a "
            "301-point counterclockwise perimeter. Also accepts aliases like "
            "outer-circuil, circle, and perimeter."
        ),
    )
    parser.add_argument(
        "--perimeter-points",
        type=int,
        default=DEFAULT_PERIMETER_POINTS,
        help="Number of generated outer-circle perimeter points; default matches the IK simulation convention.",
    )
    parser.add_argument("--max-duration-s", type=float, default=180.0)
    parser.add_argument(
        "--range-scale",
        type=float,
        default=0.6,
        help="Conservative scale used when --safe-scaled-range is passed.",
    )
    parser.add_argument(
        "--full-observed-range",
        dest="full_observed_range",
        action="store_true",
        default=True,
        help="Use full observed sample range from the 2026-08-31 motor log (default).",
    )
    parser.add_argument(
        "--safe-scaled-range",
        dest="full_observed_range",
        action="store_false",
        help="Use conservative --range-scale instead of the full observed range.",
    )
    parser.add_argument(
        "--movement-scale",
        type=float,
        default=1.05,
        help="Multiplier applied to movement bounds; default adds 5% beyond the observed range.",
    )
    parser.add_argument(
        "--custom-bounds",
        help="Optional min:max,min:max,min:max bounds for base/base+1/base+2.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> tuple[ValidationConfig, dict[int, tuple[int, int]] | None]:
    full_range = bool(args.full_observed_range or args.custom_bounds)
    trajectory = normalize_trajectory(args.trajectory)
    custom_bounds = parse_bounds(args.custom_bounds, args.base_index) if args.custom_bounds else None
    return (
        ValidationConfig(
            port=args.port,
            baud=args.baud,
            camera_index=args.camera_index,
            base_index=args.base_index,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            no_camera=args.no_camera,
            auto_start=args.auto_start,
            width=args.width,
            height=args.height,
            fps=args.fps,
            hold_s=args.hold_s,
            step_ticks=args.step_ticks,
            max_duration_s=args.max_duration_s,
            full_range=full_range,
            range_scale=1.0 if full_range else args.range_scale,
            movement_scale=args.movement_scale,
            axis_count=args.axis_count,
            trajectory=trajectory,
            perimeter_points=args.perimeter_points,
        ),
        custom_bounds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config, custom_bounds = config_from_args(args)
    try:
        run_dir = run_validation(config, custom_bounds=custom_bounds)
    except serial.SerialException as exc:
        print(serial_port_error_message(config.port, exc), file=sys.stderr)
        return 2
    print(f"Workspace validation output: {run_dir}")
    print("Final reset command emitted; inspect events.csv and run_summary.json for alignment data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
