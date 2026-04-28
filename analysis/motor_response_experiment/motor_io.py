"""Direct serial I/O for the ESP32 motor controller.

Mirrors the connection settings used by `motor_cli.py`:
    115200 baud, DTR/RTS held low to avoid resetting the ESP32 on connect.

Each command of the form ``ZM<idx>P<target>F`` is acknowledged by the ESP32
with one line of the form ``OK:M<idx>P<actual>`` (multiple motors comma-separated).
This module sends a single command and parses the actual encoder position
reported in the response.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import serial


def build_command(motor_index: int, target: int) -> str:
    """Build the ESP32 ZM<motor>P<target>F command string."""
    return f"ZM{motor_index}P{target}F"


@dataclass
class MotorResponse:
    """Parsed reply to a single motor command."""

    raw: str                       # full response text (may span multiple lines)
    ok_line: Optional[str]         # the ``OK:...`` line if found
    actual: Optional[int]          # parsed actual encoder position for our motor
    error: Optional[str]           # firmware error string, if any (``E:...``)
    elapsed_s: float               # time between send and OK/E response


_OK_PATTERN = re.compile(r"M(\d+)P(-?\d+)")


class MotorSerial:
    """Thin wrapper around `serial.Serial` for the ESP32 motor protocol."""

    def __init__(
        self,
        port: str = "COM13",
        baud: int = 115200,
        boot_wait_s: float = 3.0,
        response_timeout_s: float = 5.0,
    ):
        self._port_name = port
        self._baud = baud
        self._boot_wait_s = boot_wait_s
        self._response_timeout_s = response_timeout_s
        self._ser: Optional[serial.Serial] = None
        self._boot_log: list[str] = []

    def open(self) -> list[str]:
        """Open the serial port and wait for the ESP32 to finish booting.

        Returns the list of boot log lines emitted by the firmware (informational).
        """
        if self._ser is not None and self._ser.is_open:
            return self._boot_log

        self._ser = serial.Serial(
            port=self._port_name,
            baudrate=self._baud,
            timeout=0.1,
            dsrdtr=False,
            rtscts=False,
        )
        self._ser.dtr = False
        self._ser.rts = False

        time.sleep(0.1)
        # Drain any boot messages, give the firmware time to detect motors.
        deadline = time.time() + self._boot_wait_s
        while time.time() < deadline:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                self._boot_log.append(line)
        self._ser.reset_input_buffer()
        return self._boot_log

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def __enter__(self) -> "MotorSerial":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    @property
    def port_name(self) -> str:
        return self._port_name

    def send(self, motor_index: int, target: int) -> MotorResponse:
        """Send one ``ZM<motor>P<target>F`` and wait for the ``OK:`` reply.

        Returns a :class:`MotorResponse` with the reply text, the parsed
        actual encoder position for ``motor_index`` (or ``None`` if missing),
        and a firmware error message if the ESP32 returned ``E:...``.
        """
        if not self.is_open:
            raise RuntimeError("Serial port is not open. Call .open() first.")
        assert self._ser is not None
        msg = build_command(motor_index, target)
        return self._send_raw(msg, motor_index)

    def send_command(self, message: str, motor_index: int) -> MotorResponse:
        """Send an arbitrary already-formatted command string and parse the reply."""
        if not self.is_open:
            raise RuntimeError("Serial port is not open. Call .open() first.")
        return self._send_raw(message, motor_index)

    def _send_raw(self, message: str, motor_index: int) -> MotorResponse:
        assert self._ser is not None
        self._ser.reset_input_buffer()
        t0 = time.time()
        self._ser.write(message.encode())
        self._ser.flush()

        lines: list[str] = []
        ok_line: Optional[str] = None
        error: Optional[str] = None
        actual: Optional[int] = None

        deadline = t0 + self._response_timeout_s
        while time.time() < deadline:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            lines.append(line)
            if line.startswith("OK:"):
                ok_line = line
                # Parse "OK:M0P5,M1P-200"  ->  find the motor we asked about.
                for m_idx, val in _OK_PATTERN.findall(line):
                    if int(m_idx) == motor_index:
                        actual = int(val)
                        break
                break
            if line.startswith("E:"):
                error = line
                break

        return MotorResponse(
            raw="\n".join(lines),
            ok_line=ok_line,
            actual=actual,
            error=error,
            elapsed_s=time.time() - t0,
        )


class DryRunMotor:
    """No-op stand-in for :class:`MotorSerial` used by ``--dry-run`` / tests.

    Returns ``actual = target`` so the runner can be exercised without hardware.
    """

    def __init__(self, port: str = "DRY", baud: int = 0, **_kwargs):
        self._port_name = port
        self._open = False
        self._boot_log = ["[DRY RUN] no real serial connection"]

    def open(self) -> list[str]:
        self._open = True
        return self._boot_log

    def close(self) -> None:
        self._open = False

    def __enter__(self) -> "DryRunMotor":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def port_name(self) -> str:
        return self._port_name

    def send(self, motor_index: int, target: int) -> MotorResponse:
        msg = build_command(motor_index, target)
        return self._send_raw(msg, motor_index, target)

    def send_command(self, message: str, motor_index: int) -> MotorResponse:
        # Best-effort target parsing for dry-run accounting.
        match = re.search(rf"M{motor_index}P(-?\d+)", message)
        target = int(match.group(1)) if match else 0
        return self._send_raw(message, motor_index, target)

    def _send_raw(self, message: str, motor_index: int, target: int) -> MotorResponse:
        ok_line = f"OK:M{motor_index}P{target}"
        return MotorResponse(
            raw=ok_line, ok_line=ok_line, actual=target, error=None, elapsed_s=0.0,
        )
