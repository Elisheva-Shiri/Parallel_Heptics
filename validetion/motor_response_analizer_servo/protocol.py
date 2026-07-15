"""Protocol generator for motor response characterization.

For each delta D in DELTAS (default spans small steps up to +/-1000 encoder units)
the protocol contains:

  * 6 trials alternating sequence "A" and "B":
        A: 0 -> +D -> 0 -> -D -> 0
        B: 0 -> -D -> 0 -> +D -> 0
        A is performed 3 times and B is performed 3 times (interleaved A,B,A,B,A,B).
  * A drift block of 20 alternating steps (+D, -D, +D, -D, ...).

The output rows mirror the Excel template `esp32_protocol_log.xlsx`:
    block, trial, delta, sequence, mode, step_index, target, command
The per-step `actual`, `response`, `relative_error_percent` and `angle_deg`
columns are filled by the experiment runner at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

# Default: coarse coverage from 5 to 1000 encoder units (positive and negative).
DEFAULT_DELTAS: tuple[int, ...] = (5, 10, 25, 75, 125, 250, 500, 1000)
TRIALS_PER_SEQUENCE: int = 3      # 3 A trials + 3 B trials = 6 trials per block
DRIFT_PAIRS: int = 10             # 10 (+D,-D) pairs = 20 drift steps


@dataclass
class ProtocolStep:
    """One row of the experimental protocol (single command)."""

    block: int
    trial: int | str          # int for protocol trials, "drift" for the drift block
    delta: int
    sequence: str             # "A", "B" or "drift"
    mode: str                 # "protocol" or "drift"
    step_index: int           # 1-based index inside the trial / drift block
    target: int               # absolute encoder target sent to the motor
    command: str              # full ESP32 string e.g. "ZM0P5F"


def build_command(motor_index: int, target: int) -> str:
    """Build the ESP32 ZM<motor>P<target>F command string."""
    return f"ZM{motor_index}P{target}F"


def _sequence_targets(seq: str, delta: int) -> list[int]:
    """Return the 5 target positions for a single trial sequence."""
    if seq == "A":
        return [0, delta, 0, -delta, 0]
    if seq == "B":
        return [0, -delta, 0, delta, 0]
    raise ValueError(f"Unknown sequence: {seq}")


def build_protocol(
    deltas: Iterable[int] = DEFAULT_DELTAS,
    motor_index: int = 0,
    trials_per_sequence: int = TRIALS_PER_SEQUENCE,
    drift_pairs: int = DRIFT_PAIRS,
) -> list[ProtocolStep]:
    """Generate the full protocol as a flat list of `ProtocolStep`s."""
    steps: list[ProtocolStep] = []
    for block, delta in enumerate(deltas, start=1):
        # Interleave A,B,A,B,... so A appears trials_per_sequence times and so does B.
        trial_idx = 0
        for repeat in range(trials_per_sequence):
            for seq in ("A", "B"):
                trial_idx += 1
                for step_index, target in enumerate(_sequence_targets(seq, delta), start=1):
                    steps.append(ProtocolStep(
                        block=block,
                        trial=trial_idx,
                        delta=delta,
                        sequence=seq,
                        mode="protocol",
                        step_index=step_index,
                        target=target,
                        command=build_command(motor_index, target),
                    ))

        # Drift block: 10 pairs of (+D, -D)  ->  20 alternating steps
        step_index = 0
        for _ in range(drift_pairs):
            for target in (delta, -delta):
                step_index += 1
                steps.append(ProtocolStep(
                    block=block,
                    trial="drift",
                    delta=delta,
                    sequence="drift",
                    mode="drift",
                    step_index=step_index,
                    target=target,
                    command=build_command(motor_index, target),
                ))
    return steps


def protocol_to_dicts(steps: list[ProtocolStep]) -> list[dict]:
    """Convert protocol steps to a list of dictionaries (for pandas/Excel)."""
    return [asdict(step) for step in steps]


if __name__ == "__main__":
    proto = build_protocol()
    print(f"Total steps: {len(proto)}")
    for s in proto[:5]:
        print(s)
    print("...")
    for s in proto[-5:]:
        print(s)
