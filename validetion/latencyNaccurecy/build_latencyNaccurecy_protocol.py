"""Build a balanced latency/accuracy validation protocol CSV.

Each CSV row is one target trial. After every trial, the participant/device should
return to CENTER before starting the next trial.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ORIENTATIONS = ["N", "NW", "NE"]
REPEATS_PER_CONDITION = 4
OUTPUT_CSV = "latencyNaccurecy_protocol.csv"


def build_protocol() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    trial = 1

    for block in range(REPEATS_PER_CONDITION):
        for step in range(len(DIRECTIONS) * len(ORIENTATIONS)):
            direction = DIRECTIONS[(step + block) % len(DIRECTIONS)]
            orientation = ORIENTATIONS[(step + block) % len(ORIENTATIONS)]
            rows.append(
                {
                    "trial": trial,
                    "direction": direction,
                    "orientation": orientation,
                }
            )
            trial += 1

    return rows


def validate_protocol(rows: list[dict[str, object]]) -> None:
    expected_trials = len(DIRECTIONS) * len(ORIENTATIONS) * REPEATS_PER_CONDITION
    if len(rows) != expected_trials:
        raise ValueError(f"Expected {expected_trials} trials, got {len(rows)}")

    counts = Counter((row["direction"], row["orientation"]) for row in rows)
    if len(counts) != len(DIRECTIONS) * len(ORIENTATIONS):
        raise ValueError("Not all direction/orientation combinations are present")

    bad_counts = {condition: count for condition, count in counts.items() if count != REPEATS_PER_CONDITION}
    if bad_counts:
        raise ValueError(f"Unbalanced condition counts: {bad_counts}")

    for previous, current in zip(rows, rows[1:]):
        if previous["direction"] == current["direction"]:
            raise ValueError(f"Adjacent repeated direction near trial {current['trial']}")
        if previous["orientation"] == current["orientation"]:
            raise ValueError(f"Adjacent repeated orientation near trial {current['trial']}")


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["trial", "direction", "orientation"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_protocol()
    validate_protocol(rows)
    output_path = Path(__file__).with_name(OUTPUT_CSV)
    write_csv(rows, output_path)
    print(f"Saved {len(rows)} trials to {output_path}")
    print("Columns: trial, direction, orientation")
    print("Instruction: return to CENTER after every trial.")


if __name__ == "__main__":
    main()
