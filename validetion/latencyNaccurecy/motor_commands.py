"""Parse the ESP32 serial-bridge log (``motor_commands.txt``).

The bridge logs every message that flows through it, one per line::

    [2026-06-28 15:30:48.205] [UDP IN    ] ZM0P0M1P36M2P-48F
    [2026-06-28 15:30:48.205] [SERIAL OUT] ZM0P0M1P36M2P-48F
    [2026-06-28 15:30:48.213] [SERIAL IN ] OK:M0P0,M1P36,M2P-36

Three message sources matter:

* ``UDP IN``    - the position the Python backend *commanded* (it arrived over
  UDP from backend.py).
* ``SERIAL OUT`` - the same command forwarded to the ESP32 (essentially a copy
  of ``UDP IN``; kept for completeness).
* ``SERIAL IN``  - the ESP32's acknowledgement of the position it *applied*
  (the ``OK:...`` line).  This is the closest digital proxy for "what the motor
  was told to do" after the firmware's clamping/dead-banding.

Two on-the-wire encodings are parsed:

* command form  ``ZM<idx>P<pos>M<idx>P<pos>...F``     (positions in [-1000,1000])
* ack form      ``OK:M<idx>P<pos>,M<idx>P<pos>,...``

INFO / shutdown / banner lines are ignored.

IMPORTANT - partial coverage
-----------------------------
For the current dataset the log only covers a *sub-window* of the whole session
(it was started late), so its timestamps will overlap only part of the video.
``parse_motor_commands`` therefore also reports the covered ``[t_start, t_end]``
so downstream code can mark non-overlapping trials as "no motor data".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# [2026-06-28 15:30:48.205] [UDP IN    ] <payload>
_LINE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]\s*"
    r"\[(?P<src>[^\]]+)\]\s*(?P<payload>.*)$"
)
# M<idx>P<pos> pairs, used for both command and ack payloads.
_MOTOR_RE = re.compile(r"M(\d+)P(-?\d+)")

_SOURCE_CANON = {
    "UDP IN": "UDP_IN",
    "SERIAL OUT": "SERIAL_OUT",
    "SERIAL IN": "SERIAL_IN",
}


@dataclass
class MotorCommandLog:
    """Parsed log: a tidy frame plus metadata about coverage."""

    df: pd.DataFrame                 # columns: timestamp, source, m0, m1, m2, ...
    motor_columns: list[str]         # e.g. ["m0", "m1", "m2"]
    t_start: Optional[pd.Timestamp]
    t_end: Optional[pd.Timestamp]
    n_lines: int
    n_parsed: int

    def stream(self, source: str) -> pd.DataFrame:
        """Return only one source's rows (e.g. 'SERIAL_IN' for applied positions)."""
        return self.df[self.df["source"] == source].reset_index(drop=True)

    def commanded(self) -> pd.DataFrame:
        """Backend-commanded positions (UDP_IN, falling back to SERIAL_OUT)."""
        cmd = self.stream("UDP_IN")
        if cmd.empty:
            cmd = self.stream("SERIAL_OUT")
        return cmd

    def acknowledged(self) -> pd.DataFrame:
        """ESP32-applied positions (SERIAL_IN 'OK:' lines)."""
        return self.stream("SERIAL_IN")


def _parse_payload(payload: str) -> Optional[dict[str, int]]:
    """Extract {motor_index: position} from either encoding, else None."""
    p = payload.strip()
    if not p:
        return None
    # Accept command lines (start with Z, end with F) and ack lines (start OK:).
    is_cmd = p.startswith("Z")
    is_ack = p.startswith("OK:")
    if not (is_cmd or is_ack):
        return None
    pairs = _MOTOR_RE.findall(p)
    if not pairs:
        return None
    return {f"m{int(idx)}": int(pos) for idx, pos in pairs}


# The bridge log has been saved under a few different names across sessions.
_LOG_NAMES = ("motor_commands.txt", "motor_command.txt", "motor_commands.log")


def find_motor_log(session_dir: str | Path) -> Optional[Path]:
    """Return the motor-command log path in a session dir, or None if absent.

    Handles the naming drift (``motor_commands.txt`` vs ``motor_command.txt``)
    seen across runs; returns None so callers can degrade gracefully when a
    session has no motor log at all.
    """
    session_dir = Path(session_dir)
    for name in _LOG_NAMES:
        p = session_dir / name
        if p.exists():
            return p
    return None


def empty_log() -> MotorCommandLog:
    """An empty log (used when a session has no motor-command file)."""
    return MotorCommandLog(pd.DataFrame(columns=["timestamp", "source"]), [], None, None, 0, 0)


def parse_motor_commands(path: str | Path) -> MotorCommandLog:
    """Parse the bridge log file into a :class:`MotorCommandLog`."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[dict] = []
    motor_cols: set[str] = set()
    for line in lines:
        m = _LINE_RE.match(line)
        if not m:
            continue
        src = _SOURCE_CANON.get(m.group("src").strip())
        if src is None:
            continue  # INFO, banners, etc.
        motors = _parse_payload(m.group("payload"))
        if motors is None:
            continue
        rec = {"timestamp": m.group("ts"), "source": src, **motors}
        motor_cols.update(motors.keys())
        records.append(rec)

    motor_columns = sorted(motor_cols, key=lambda c: int(c[1:]))
    if not records:
        empty = pd.DataFrame(columns=["timestamp", "source", *motor_columns])
        return MotorCommandLog(empty, motor_columns, None, None, len(lines), 0)

    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
    for c in motor_columns:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[["timestamp", "source", *motor_columns]].sort_values("timestamp").reset_index(drop=True)

    return MotorCommandLog(
        df=df,
        motor_columns=motor_columns,
        t_start=df["timestamp"].min(),
        t_end=df["timestamp"].max(),
        n_lines=len(lines),
        n_parsed=len(df),
    )


if __name__ == "__main__":
    import sys

    log = parse_motor_commands(sys.argv[1])
    print(f"lines={log.n_lines} parsed={log.n_parsed} motors={log.motor_columns}")
    print(f"coverage: {log.t_start} .. {log.t_end}")
    print("by source:\n", log.df["source"].value_counts())
    print(log.df.head(8).to_string())
