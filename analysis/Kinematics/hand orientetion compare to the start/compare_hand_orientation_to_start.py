"""
Compare filtered hand orientation to each participant's starting orientation.

This script is READ-ONLY with respect to the filtered results: it only reads files from
`result_fillter` and writes new CSV summaries into this folder's `outputs` directory.

Default interpretation
----------------------
- Uses filtered hand-orientation trial files:
  result_fillter/<SUBJECT_ID>/csv/hand_orientation/hand_orientation_plane_trials.csv
- For each participant and active finger, the circular average of the first N valid
  angles is treated as that participant/finger's starting (rest) orientation =
  0 degrees relative change. The default N is 10.
- A trial/segment is counted as "same orientation" when it is within +/- 5 degrees
  from the starting angle. More than 5 degrees is counted as "changed".
- Angle differences are circular, so wraparound near -180/180 is handled correctly.

Outputs
-------
outputs/hand_orientation_change_trials_long.csv
    One row per subject x trial/segment x plane, with baseline and change flag.
outputs/hand_orientation_change_participant_summary.csv
    Counts and percentages changed per participant and plane.
outputs/hand_orientation_change_group_summary.csv
    Counts and percentages changed per experiment group and plane.
outputs/hand_orientation_change_participant_finger_summary.csv
    Same as participant summary but split by active finger.
outputs/hand_orientation_change_group_finger_summary.csv
    Same as group summary but split by active finger.

Run from the Kinematics folder:
    python "hand orientetion compare to the start/compare_hand_orientation_to_start.py"

Useful options:
    --threshold-deg 5
    --baseline-mode subject_finger   # default, recommended
    --baseline-mode subject          # one baseline per participant, ignoring finger
    --baseline-first-n 10            # average first 10 valid orientation values
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PLANES = {
    "XY": "hand_orientation_xy_deg",
    "YZ": "hand_orientation_yz_deg",
    "ZX": "hand_orientation_zx_deg",
}

ID_COLUMNS = [
    "subject_id",
    "subject_group",
    "experiment_group",
    "finger_condition",
    "run_dir",
    "trial_index_raw",
    "pair_number",
    "stiffness_segment_id",
    "stiffness_order_in_trial",
    "stiffness_value",
    "success_label",
]

SORT_COLUMNS = [
    "subject_id",
    "finger_condition",
    "run_dir",
    "trial_index_raw",
    "pair_number",
    "stiffness_order_in_trial",
    "stiffness_segment_id",
]


def circular_signed_diff_deg(angle: pd.Series, baseline: pd.Series) -> pd.Series:
    """Signed circular difference angle-baseline in degrees, returned in [-180, 180)."""
    return ((angle - baseline + 180.0) % 360.0) - 180.0


def circular_mean_deg(values: Iterable[float]) -> float:
    """Circular mean in degrees, ignoring NaNs."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if arr.size == 0:
        return float("nan")
    radians = np.deg2rad(arr)
    return float(math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean())))


def find_default_filtered_root(script_dir: Path) -> Path:
    """Find result_fillter next to this script folder or in the current directory."""
    candidates = [
        Path.cwd() / "result_fillter",
        script_dir.parent / "result_fillter",
        script_dir / "result_fillter",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Return the most likely path so the error message is useful.
    return (script_dir.parent / "result_fillter").resolve()


def load_filtered_hand_orientation(filtered_root: Path, groups: list[str]) -> pd.DataFrame:
    """Read filtered hand-orientation trial files without modifying them."""
    files = sorted(filtered_root.glob("*/csv/hand_orientation/hand_orientation_plane_trials.csv"))
    if not files:
        raise FileNotFoundError(
            f"No hand_orientation_plane_trials.csv files found under {filtered_root}"
        )

    frames: list[pd.DataFrame] = []
    needed = list(dict.fromkeys(ID_COLUMNS + list(PLANES.values())))

    for path in files:
        folder_subject_id = path.parts[-4]
        try:
            header = pd.read_csv(path, nrows=0)
            usecols = [c for c in needed if c in header.columns]
            if "subject_id" not in usecols or "experiment_group" not in usecols:
                continue
            df = pd.read_csv(path, usecols=usecols)
        except Exception as exc:  # keep going but make the source problem visible
            print(f"WARNING: skipped {path}: {exc}")
            continue

        # Avoid accidentally double-counting aggregate folders if such files exist later.
        if "subject_id" in df.columns:
            df = df[df["subject_id"].astype(str).eq(str(folder_subject_id))]
        if df.empty:
            continue

        df["source_filtered_file"] = str(path)
        frames.append(df)

    if not frames:
        raise RuntimeError("No subject-level filtered hand-orientation rows were loaded.")

    data = pd.concat(frames, ignore_index=True)
    if groups:
        data = data[data["experiment_group"].isin(groups)].copy()
    if data.empty:
        raise RuntimeError(f"No rows found for groups: {groups}")

    for col in SORT_COLUMNS:
        if col not in data.columns:
            data[col] = np.nan
    return data


def make_long_table(data: pd.DataFrame) -> pd.DataFrame:
    """Convert XY/YZ/ZX columns into one row per plane."""
    rows = []
    present_id_cols = [c for c in ID_COLUMNS if c in data.columns]
    for plane, angle_col in PLANES.items():
        if angle_col not in data.columns:
            continue
        part = data[present_id_cols + ["source_filtered_file", angle_col]].copy()
        part = part.rename(columns={angle_col: "orientation_deg"})
        part["plane"] = plane
        rows.append(part)
    if not rows:
        raise RuntimeError("No hand_orientation_*_deg columns were found.")
    long = pd.concat(rows, ignore_index=True)
    long["orientation_deg"] = pd.to_numeric(long["orientation_deg"], errors="coerce")
    long = long.dropna(subset=["orientation_deg", "subject_id", "experiment_group"])
    return long


def add_baseline_and_change_flags(
    long: pd.DataFrame,
    threshold_deg: float,
    baseline_mode: str,
    baseline_first_n: int,
) -> pd.DataFrame:
    """Add baseline angle, signed/absolute change from baseline, and changed flag."""
    if baseline_first_n < 1:
        raise ValueError("baseline_first_n must be >= 1")
    sort_cols = [c for c in SORT_COLUMNS if c in long.columns]
    long = long.sort_values(sort_cols + ["plane"], kind="mergesort").reset_index(drop=True)

    if baseline_mode == "subject":
        baseline_cols = ["subject_id", "plane"]
    elif baseline_mode == "subject_finger":
        baseline_cols = ["subject_id", "finger_condition", "plane"]
    else:
        raise ValueError("baseline_mode must be 'subject' or 'subject_finger'")

    # Circular average of the first N valid chronological angles is the start/rest angle
    # for this subject/finger/plane. Using a circular mean avoids errors around -180/180.
    baseline_source = long.dropna(subset=["orientation_deg"]).copy()
    baseline_source["baseline_order"] = baseline_source.groupby(
        baseline_cols, dropna=False, sort=False
    ).cumcount() + 1
    baseline_source = baseline_source[baseline_source["baseline_order"] <= baseline_first_n]
    baselines = (
        baseline_source.groupby(baseline_cols, dropna=False, sort=False)["orientation_deg"]
        .apply(circular_mean_deg)
        .rename("baseline_start_orientation_deg")
        .reset_index()
    )
    baseline_counts = (
        baseline_source.groupby(baseline_cols, dropna=False, sort=False)["orientation_deg"]
        .size()
        .rename("baseline_n_values")
        .reset_index()
    )
    baselines = baselines.merge(baseline_counts, on=baseline_cols, how="left")
    out = long.merge(baselines, on=baseline_cols, how="left")
    out["signed_change_from_start_deg"] = circular_signed_diff_deg(
        out["orientation_deg"], out["baseline_start_orientation_deg"]
    )
    out["abs_change_from_start_deg"] = out["signed_change_from_start_deg"].abs()
    out["same_as_start_within_threshold"] = out["abs_change_from_start_deg"] <= threshold_deg
    out["changed_from_start"] = out["abs_change_from_start_deg"] > threshold_deg
    out["threshold_deg"] = threshold_deg
    out["baseline_mode"] = baseline_mode
    out["baseline_first_n"] = baseline_first_n

    # Count chronological switches between same/changed states as an extra diagnostic.
    transition_cols = ["subject_id", "plane"]
    if baseline_mode == "subject_finger":
        transition_cols.insert(1, "finger_condition")
    out["changed_state_transition"] = (
        out.groupby(transition_cols, dropna=False)["changed_from_start"]
        .transform(lambda s: s.ne(s.shift()).fillna(False))
        .astype(bool)
    )
    # The first row is not a transition; reset it after transform.
    first_idx = out.groupby(transition_cols, dropna=False).head(1).index
    out.loc[first_idx, "changed_state_transition"] = False
    return out


def summarize(table: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize change counts and circular/linear change magnitudes."""
    rows: list[dict[str, object]] = []
    for keys, g in table.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        n = int(len(g))
        n_changed = int(g["changed_from_start"].sum())
        n_same = int(g["same_as_start_within_threshold"].sum())
        row.update(
            {
                "n_trial_segments": n,
                "n_changed_from_start": n_changed,
                "n_same_as_start": n_same,
                "percent_changed_from_start": (100.0 * n_changed / n) if n else np.nan,
                "mean_signed_change_deg": circular_mean_deg(g["signed_change_from_start_deg"]),
                "mean_abs_change_deg": float(g["abs_change_from_start_deg"].mean()),
                "median_abs_change_deg": float(g["abs_change_from_start_deg"].median()),
                "max_abs_change_deg": float(g["abs_change_from_start_deg"].max()),
                "n_changed_state_transitions": int(g["changed_state_transition"].sum()),
                "threshold_deg": float(g["threshold_deg"].iloc[0]),
                "baseline_mode": str(g["baseline_mode"].iloc[0]),
                "baseline_first_n": int(g["baseline_first_n"].iloc[0]),
                "min_baseline_n_values": int(g["baseline_n_values"].min()),
                "max_baseline_n_values": int(g["baseline_n_values"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare filtered hand orientation to each participant's starting orientation."
    )
    parser.add_argument(
        "--filtered-root",
        type=Path,
        default=find_default_filtered_root(script_dir),
        help="Path to result_fillter. The script only reads from this folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
        help="Folder where new summary CSVs will be written.",
    )
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=5.0,
        help="Absolute circular change <= this value is treated as the same orientation.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["subject_finger", "subject"],
        default="subject_finger",
        help=(
            "subject_finger: baseline per participant x active finger x plane. "
            "subject: baseline per participant x plane."
        ),
    )
    parser.add_argument(
        "--baseline-first-n",
        type=int,
        default=10,
        help="Use the circular average of the first N valid orientation values as the start baseline.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["L_E", "N_E"],
        help="Experiment groups to include. Default: L_E N_E.",
    )
    args = parser.parse_args()

    filtered_root = args.filtered_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_filtered_hand_orientation(filtered_root, args.groups)
    long = make_long_table(data)
    changed = add_baseline_and_change_flags(
        long,
        threshold_deg=args.threshold_deg,
        baseline_mode=args.baseline_mode,
        baseline_first_n=args.baseline_first_n,
    )

    participant_summary = summarize(changed, ["experiment_group", "subject_id", "plane"])
    group_summary = summarize(changed, ["experiment_group", "plane"])
    participant_finger_summary = summarize(
        changed, ["experiment_group", "subject_id", "finger_condition", "plane"]
    )
    group_finger_summary = summarize(
        changed, ["experiment_group", "finger_condition", "plane"]
    )

    # Stable, human-readable column order for the long table.
    preferred = [
        "experiment_group",
        "subject_id",
        "subject_group",
        "finger_condition",
        "plane",
        "run_dir",
        "trial_index_raw",
        "pair_number",
        "stiffness_segment_id",
        "stiffness_order_in_trial",
        "stiffness_value",
        "success_label",
        "orientation_deg",
        "baseline_start_orientation_deg",
        "signed_change_from_start_deg",
        "abs_change_from_start_deg",
        "same_as_start_within_threshold",
        "changed_from_start",
        "changed_state_transition",
        "threshold_deg",
        "baseline_mode",
        "baseline_first_n",
        "baseline_n_values",
        "source_filtered_file",
    ]
    long_cols = [c for c in preferred if c in changed.columns] + [
        c for c in changed.columns if c not in preferred
    ]

    changed[long_cols].to_csv(output_dir / "hand_orientation_change_trials_long.csv", index=False)
    participant_summary.to_csv(
        output_dir / "hand_orientation_change_participant_summary.csv", index=False
    )
    group_summary.to_csv(output_dir / "hand_orientation_change_group_summary.csv", index=False)
    participant_finger_summary.to_csv(
        output_dir / "hand_orientation_change_participant_finger_summary.csv", index=False
    )
    group_finger_summary.to_csv(
        output_dir / "hand_orientation_change_group_finger_summary.csv", index=False
    )

    print("READ ONLY input folder:", filtered_root)
    print("Wrote output folder:", output_dir)
    print("Rows analyzed:", len(changed))
    print("Groups:", ", ".join(map(str, sorted(changed["experiment_group"].dropna().unique()))))
    print("Subjects:", changed["subject_id"].nunique())
    print("Threshold degrees:", args.threshold_deg)
    print("Baseline mode:", args.baseline_mode)
    print("Baseline first N values:", args.baseline_first_n)
    print("\nGroup summary:")
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
