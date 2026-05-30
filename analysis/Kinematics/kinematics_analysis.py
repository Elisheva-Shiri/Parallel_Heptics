"""Standalone kinematic analysis for the 2AFC skin-stretch experiment.

Reads raw experiment outputs only and writes derived CSV/figures to a dedicated
results directory. It does not modify experiment architecture or raw data.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from analysis.group_comparisons import (
        add_experiment_group_columns,
        compute_analysis_scope_tables,
        compute_group_comparison_tables,
        compute_setup_factor_tables,
        EXPERIMENT_GROUP_COLUMN,
    )
    from analysis.scope_plots import save_scope_summary_plots
except (
    ModuleNotFoundError
):  # pragma: no cover - supports running from analysis subfolders
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from analysis.group_comparisons import (
        add_experiment_group_columns,
        compute_analysis_scope_tables,
        compute_group_comparison_tables,
        compute_setup_factor_tables,
        EXPERIMENT_GROUP_COLUMN,
    )
    from analysis.scope_plots import save_scope_summary_plots

try:
    from scipy.signal import butter, filtfilt
except Exception:  # pragma: no cover
    butter = None
    filtfilt = None

CENTER_X = 320.0
CENTER_Y = 240.0
STIFFNESS_MAX = 175.0
DIRECTION_BINS_DEG = np.arange(-180, 181, 30)
DIRECTION_LABELS = [
    "W",
    "WSW",
    "SW",
    "SSW",
    "S",
    "SSE",
    "SE",
    "ESE",
    "E",
    "ENE",
    "NE",
    "NNE",
]
FINGER_ORDER = ["I", "M", "R", "P"]
FINGER_ALIASES = {
    "i": "I",
    "idx": "I",
    "index": "I",
    "index finger": "I",
    "m": "M",
    "mid": "M",
    "middle": "M",
    "middle finger": "M",
    "r": "R",
    "ring": "R",
    "ring finger": "R",
    "p": "P",
    "pinkie": "P",
    "pinky": "P",
    "little": "P",
    "little finger": "P",
}
FINGER_LABELS = {"I": "Index", "M": "Middle", "R": "Ring", "P": "Pinky"}
FINGER_TO_COLOR = {"I": "#1f77b4", "M": "#ff7f0e", "R": "#2ca02c", "P": "#d62728"}
IGNORED_PATH_PATTERNS = (
    "old",
    "not finish",
    "not_finish",
    "not-finish",
    "notfinish",
    "unfinished",
)
WORKSPACE_SPECS_CM = {
    # User-specified physical workspaces.  These normalize derived coordinates
    # while preserving the original camera/analog pixel values in all outputs.
    "L": {"workspace_width_cm": 60.0, "workspace_height_cm": 60.0},
    "N": {"workspace_width_cm": 40.0, "workspace_height_cm": 50.0},
}
EXPERIMENT_SETUP_CONTEXT = {
    "L": {
        "side_camera_side": "right",
        "participant_position_context": "slightly_left",
        "movement_space_context": "larger 60x60 cm movement space",
        "side_camera_interpretation_note": (
            "L experiment: side camera was on the participant's right side; "
            "participant was slightly left of the workspace center."
        ),
    },
    "N": {
        "side_camera_side": "left",
        "participant_position_context": "centered",
        "movement_space_context": "smaller 40x50 cm movement space",
        "side_camera_interpretation_note": (
            "N experiment: participant sat at the center of the smaller movement "
            "space; side camera was on the participant's left side."
        ),
    },
}
STIFFNESS_CMAP = "viridis"
MOTOR_CONTROL_METRICS = [
    "success_rate",
    "mean_max_r_center_px",
    "mean_speed_px_s",
    "mean_acceleration_px_s2",
    "mean_jerk_px_s3",
    "mean_normalized_jerk_cost",
    "mean_curvature_1_px",
    "speed_curvature_power_law_slope",
    "speed_curvature_power_law_r2",
    "mean_path_length_px",
    "mean_straightness_index",
    "mean_vx_px_s",
    "mean_vy_px_s",
]
TRIAL_SUCCESS_METRICS = [
    "max_r_center_px",
    "path_length_px",
    "straightness_index",
    "mean_speed_px_s",
    "max_speed_px_s",
    "mean_acceleration_px_s2",
    "max_acceleration_px_s2",
    "mean_jerk_px_s3",
    "max_jerk_px_s3",
    "normalized_jerk_cost",
    "mean_curvature_1_px",
    "speed_curvature_power_law_slope",
    "speed_curvature_power_law_r2",
    "mean_abs_radial_velocity_px_s",
    "mean_abs_tangential_velocity_px_s",
    "movement_direction_resultant_length",
    "movement_direction_entropy_bits",
    "duration_s",
    "mean_side_z_lift_px",
    "max_side_z_lift_px",
    "side_detection_rate",
]


def save_csv(df: pd.DataFrame, output_root: Path, name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / name
    df.to_csv(path, index=False)
    return path


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1255", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def sanitize_name(value: Any, fallback: str = "unknown") -> str:
    import re

    text = str(value) if value is not None and not pd.isna(value) else fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def subject_figure_path(fig_dir: Path, subject_id: Any, filename: str) -> Path:
    """Return a figure path under a per-person folder such as ``N_E_7``."""

    subject_dir = fig_dir / sanitize_name(subject_id, fallback="unknown_subject")
    subject_dir.mkdir(parents=True, exist_ok=True)
    return subject_dir / filename


def subject_figure_path_for_scope(fig_dir: Path, scope_name: str, filename: str) -> Path:
    """Route ``subject_<id>`` scope figures into an ``<id>/`` folder."""

    subject_prefix = "subject_"
    if scope_name.startswith(subject_prefix):
        subject_id = scope_name[len(subject_prefix) :]
        return subject_figure_path(fig_dir, subject_id, filename)
    return fig_dir / filename


def normalize_finger_condition(value: Any) -> Any:
    """Return canonical finger codes (I/M/R/P) while tolerating legacy labels."""
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    canonical = FINGER_ALIASES.get(
        text.lower(), text.upper() if text.upper() in FINGER_ORDER else text
    )
    return canonical


def should_ignore_analysis_path(path: Path | str) -> bool:
    """Return True for raw-data paths excluded from the requested analysis."""

    text = str(path).replace("_", " ").replace("-", " ").lower()
    return any(
        pattern.replace("_", " ").replace("-", " ") in text
        for pattern in IGNORED_PATH_PATTERNS
    )


def experiment_group_for_subject(subject_id: Any, subject_group: Any = np.nan) -> Any:
    """Infer the canonical N_E/L_E/L_P group for one subject."""
    row = pd.DataFrame([{"subject_id": subject_id, "subject_group": subject_group}])
    return add_experiment_group_columns(row).iloc[0][EXPERIMENT_GROUP_COLUMN]


def setup_factor_for_subject(subject_id: Any, experiment_group: Any = np.nan) -> Any:
    """Infer N/L workspace/setup factor without inventing demographics."""

    for value in [experiment_group, subject_id]:
        if pd.isna(value):
            continue
        text = str(value).strip().upper()
        if text.startswith("N"):
            return "N"
        if text.startswith("L"):
            return "L"
    return np.nan


def workspace_label_for_setup(setup_factor: Any) -> str:
    setup = str(setup_factor).strip().upper()
    if setup in WORKSPACE_SPECS_CM:
        spec = WORKSPACE_SPECS_CM[setup]
        return f"{setup} workspace ({spec['workspace_width_cm']:.0f}x{spec['workspace_height_cm']:.0f} cm)"
    return "unknown workspace"


def experiment_setup_context_for_setup(setup_factor: Any, field: str) -> Any:
    setup = str(setup_factor).strip().upper()
    return EXPERIMENT_SETUP_CONTEXT.get(setup, {}).get(field, np.nan)


def side_camera_view_sign_for_side(side_camera_side: Any) -> float:
    """Return a pixel-space mirror sign for opposite side-camera placements.

    We keep raw side-camera pixels unchanged.  For derived side-view
    orientation/direction values, left-side cameras are mirrored into the same
    convention as right-side cameras so L and N can be compared without treating
    a camera-placement reversal as a movement-direction reversal.
    """

    side = str(side_camera_side).strip().lower()
    if side == "right":
        return 1.0
    if side == "left":
        return -1.0
    return np.nan


def experiment_setup_context_table() -> pd.DataFrame:
    rows = []
    for setup, spec in WORKSPACE_SPECS_CM.items():
        rows.append(
            {
                "workspace_setup": setup,
                "workspace_width_cm": spec["workspace_width_cm"],
                "workspace_height_cm": spec["workspace_height_cm"],
                **EXPERIMENT_SETUP_CONTEXT.get(setup, {}),
            }
        )
    return pd.DataFrame(rows)


def save_experiment_setup_context(output_root: Path) -> Path:
    return save_csv(
        experiment_setup_context_table(), output_root, "experiment_setup_context.csv"
    )


def _figure_colors(labels: Iterable[Any]) -> dict[Any, Any]:
    labels = list(labels)
    cmap = plt.get_cmap("tab20")
    return {label: cmap(i % cmap.N) for i, label in enumerate(labels)}


def _stiffness_viridis_colors(
    stiffness_values: Iterable[Any],
    *,
    min_stiffness: float = 25.0,
    max_stiffness: float = 145.0,
) -> dict[Any, Any]:
    """Map stiffness to viridis: 25=dark purple, 145=bright yellow."""
    cmap = plt.get_cmap("viridis")
    span = max(max_stiffness - min_stiffness, 1e-9)
    colors: dict[Any, Any] = {}
    for value in stiffness_values:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            colors[value] = "0.35"
            continue
        clipped = min(max(float(numeric), min_stiffness), max_stiffness)
        colors[value] = cmap((clipped - min_stiffness) / span)
    return colors


def _first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        col = lower.get(name.lower())
        if col is not None:
            return col
    return None


def _finite_min_max(values: pd.Series | np.ndarray) -> tuple[float, float] | None:
    arr = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce")
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return None
    return float(arr.min()), float(arr.max())


def _centered_limits_from_columns(df: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    """Return symmetric limits around zero that include every finite value."""
    values: list[np.ndarray] = []
    for col in columns:
        if col in df.columns:
            values.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
    if not values:
        return (-1.0, 1.0)
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    radius = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    radius *= 1.08
    return (-radius, radius)


def _padded_limits_from_columns(df: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    """Return one shared min/max scale for heterogeneous axes."""
    values: list[np.ndarray] = []
    for col in columns:
        if col in df.columns:
            values.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
    if not values:
        return (-1.0, 1.0)
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    if lo == hi:
        pad = max(abs(lo) * 0.08, 1.0)
    else:
        pad = (hi - lo) * 0.08
    return (lo - pad, hi + pad)


def _ordered_fingers(values: Iterable[Any]) -> list[Any]:
    seen = set(pd.Series(list(values)).dropna())
    ordered = [f for f in FINGER_ORDER if f in seen]
    ordered += sorted([f for f in seen if f not in FINGER_ORDER], key=lambda x: str(x))
    return ordered


def add_workspace_normalization_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add N/L workspace metadata and normalized physical coordinates.

    Original pixel/analog values remain unchanged.  Added centimeter columns use
    the requested workspaces: L = 60x60 cm, N = 40x50 cm.
    """

    if df.empty:
        return df.copy()
    out = df.copy()
    if EXPERIMENT_GROUP_COLUMN not in out.columns:
        out = add_experiment_group_columns(out)
    if "workspace_setup" not in out.columns:
        subjects = out.get("subject_id", pd.Series(np.nan, index=out.index))
        groups = out.get(EXPERIMENT_GROUP_COLUMN, pd.Series(np.nan, index=out.index))
        out["workspace_setup"] = [
            setup_factor_for_subject(subject, group)
            for subject, group in zip(subjects, groups)
        ]
    out["workspace_label"] = out["workspace_setup"].map(workspace_label_for_setup)
    out["workspace_width_cm"] = out["workspace_setup"].map(
        lambda s: WORKSPACE_SPECS_CM.get(str(s).strip().upper(), {}).get(
            "workspace_width_cm", np.nan
        )
    )
    out["workspace_height_cm"] = out["workspace_setup"].map(
        lambda s: WORKSPACE_SPECS_CM.get(str(s).strip().upper(), {}).get(
            "workspace_height_cm", np.nan
        )
    )
    for context_col in [
        "side_camera_side",
        "participant_position_context",
        "movement_space_context",
        "side_camera_interpretation_note",
    ]:
        if context_col not in out.columns:
            out[context_col] = out["workspace_setup"].map(
                lambda s, col=context_col: experiment_setup_context_for_setup(s, col)
            )
    if "side_camera_view_sign" not in out.columns:
        out["side_camera_view_sign"] = out["side_camera_side"].map(
            side_camera_view_sign_for_side
        )
    if "x_centered_px" in out.columns:
        out["x_workspace_cm"] = (
            pd.to_numeric(out["x_centered_px"], errors="coerce")
            / CENTER_X
            * (out["workspace_width_cm"] / 2.0)
        )
    if "y_centered_px" in out.columns:
        out["y_workspace_cm"] = (
            pd.to_numeric(out["y_centered_px"], errors="coerce")
            / CENTER_Y
            * (out["workspace_height_cm"] / 2.0)
        )
    if {"x_workspace_cm", "y_workspace_cm"}.issubset(out.columns):
        out["r_workspace_cm"] = np.hypot(out["x_workspace_cm"], out["y_workspace_cm"])
        half_diag = np.hypot(
            out["workspace_width_cm"] / 2.0, out["workspace_height_cm"] / 2.0
        )
        out["r_workspace_normalized"] = out["r_workspace_cm"] / half_diag
    return out


def add_protocol_demographic_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize protocol/sex/age factors only when evidence exists."""

    if df.empty:
        return df.copy()
    out = df.copy()
    if "protocol_factor" not in out.columns:
        protocol_col = _first_existing_column(
            out,
            [
                "protocol_factor",
                "protocol",
                "participant_type",
                "ParticipantType",
                "comparison_version",
            ],
        )
        if protocol_col is not None:
            out["protocol_factor"] = out[protocol_col].map(
                lambda v: f"protocol_{int(float(v))}"
                if pd.notna(v) and str(v).strip().replace(".", "", 1).isdigit()
                else (str(v).strip() if pd.notna(v) else np.nan)
            )
        else:
            path_cols = [
                c for c in ["source_file", "run_dir", "tracking_file"] if c in out.columns
            ]
            if path_cols:
                values = []
                for _, row in out[path_cols].iterrows():
                    text = " ".join(str(v) for v in row.dropna())
                    match = re.search(r"participant[_\s-]*type[_\s-]*(\d+)", text, re.I)
                    values.append(f"protocol_{match.group(1)}" if match else np.nan)
                out["protocol_factor"] = values
    if "sex_factor" not in out.columns:
        sex_col = _first_existing_column(
            out, ["sex", "gender", "participant_sex", "participant_gender"]
        )
        if sex_col is not None:
            out["sex_factor"] = out[sex_col].map(
                lambda v: str(v).strip().lower() if pd.notna(v) else np.nan
            )
    if "age_group" not in out.columns:
        age_col = _first_existing_column(out, ["age", "participant_age"])
        if age_col is not None:
            age = pd.to_numeric(out[age_col], errors="coerce")
            out["age_years"] = age
            out["age_group"] = pd.cut(
                age,
                bins=[0, 20, 30, 40, 50, 65, np.inf],
                labels=["<=20", "21-30", "31-40", "41-50", "51-65", "65+"],
            ).astype("object")
    return out


def add_success_label_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "correct_response" in out.columns and "success_label" not in out.columns:
        correct = pd.to_numeric(out["correct_response"], errors="coerce")
        out["success_label"] = pd.Series(np.nan, index=out.index, dtype="object")
        out.loc[correct == 1, "success_label"] = "success"
        out.loc[correct == 0, "success_label"] = "failure"
    return out


def _add_hand_actor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add thumb/active-finger vectors from tracked MediaPipe columns."""

    out = df.copy()
    for source, target in [
        ("thumb_x", "thumb_x_px"),
        ("thumb_y", "thumb_y_px"),
        ("active_finger_x", "active_finger_x_px"),
        ("active_finger_y", "active_finger_y_px"),
    ]:
        if source in out.columns and target not in out.columns:
            out[target] = pd.to_numeric(out[source], errors="coerce")
    if {"thumb_x_px", "thumb_y_px", "active_finger_x_px", "active_finger_y_px"}.issubset(
        out.columns
    ):
        out["thumb_x_centered_px"] = out["thumb_x_px"] - CENTER_X
        out["thumb_y_centered_px"] = out["thumb_y_px"] - CENTER_Y
        out["active_finger_x_centered_px"] = out["active_finger_x_px"] - CENTER_X
        out["active_finger_y_centered_px"] = out["active_finger_y_px"] - CENTER_Y
        out["thumb_active_dx_px"] = out["active_finger_x_px"] - out["thumb_x_px"]
        out["thumb_active_dy_px"] = out["active_finger_y_px"] - out["thumb_y_px"]
        out["thumb_active_span_px"] = np.hypot(
            out["thumb_active_dx_px"], out["thumb_active_dy_px"]
        )
        out["hand_orientation_xy_deg"] = np.degrees(
            np.arctan2(out["thumb_active_dy_px"], out["thumb_active_dx_px"])
        )
        out["hand_midpoint_x_centered_px"] = (
            out["thumb_x_centered_px"] + out["active_finger_x_centered_px"]
        ) / 2.0
        out["hand_midpoint_y_centered_px"] = (
            out["thumb_y_centered_px"] + out["active_finger_y_centered_px"]
        ) / 2.0
    return out


def _sem(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def _ci95_mean(s: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) <= 1:
        return (np.nan, np.nan)
    # Normal approximation keeps this dependency-light; distribution tables also
    # report medians and bootstrap median intervals for skewed data.
    half_width = 1.96 * float(x.std(ddof=1) / math.sqrt(len(x)))
    mean = float(x.mean())
    return (mean - half_width, mean + half_width)


def _ci95_median_bootstrap(
    s: pd.Series, *, n_boot: int = 2000, seed: int = 20260513
) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, x.size), replace=True)
    lo, hi = np.nanpercentile(np.nanmedian(samples, axis=1), [2.5, 97.5])
    return (float(lo), float(hi))


def _skewness(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size < 3:
        return np.nan
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / sd) ** 3))


def _log_backtransformed_summary(s: pd.Series) -> dict[str, float | bool]:
    """Return log-domain summary back-transformed to original units.

    A log summary is only meaningful for strictly positive values. The
    ``log_transform_recommended`` flag is heuristic: positive data with strong
    skew or a wide max/min ratio. The values are still reported so reviewers can
    decide whether to use them.
    """
    x = pd.to_numeric(s, errors="coerce").dropna()
    x = x[x > 0]
    if len(x) <= 1:
        return {
            "log_transform_valid": False,
            "log_transform_recommended": False,
            "geometric_mean_backtransformed": np.nan,
            "geometric_ci95_low_backtransformed": np.nan,
            "geometric_ci95_high_backtransformed": np.nan,
        }
    lx = np.log(x)
    lo, hi = _ci95_mean(pd.Series(lx))
    skew = _skewness(x)
    ratio = float(x.max() / x.min()) if x.min() > 0 else np.inf
    return {
        "log_transform_valid": True,
        "log_transform_recommended": bool(
            (np.isfinite(skew) and abs(skew) >= 1.0) or ratio >= 10.0
        ),
        "geometric_mean_backtransformed": float(np.exp(lx.mean())),
        "geometric_ci95_low_backtransformed": float(np.exp(lo))
        if np.isfinite(lo)
        else np.nan,
        "geometric_ci95_high_backtransformed": float(np.exp(hi))
        if np.isfinite(hi)
        else np.nan,
    }


def _normalized_jerk_cost(segment: pd.DataFrame, *, path_length_px: float) -> float:
    """Dimensionless movement-smoothness cost from squared jerk.

    Recent skin-stretch probing work analyzes motor-control changes across
    repeated contacts; jerk complements speed/acceleration by flagging abrupt
    corrective movements. Lower values indicate smoother probing for comparable
    duration and path length.
    """
    if not {"time_s", "jx_px_s3", "jy_px_s3"}.issubset(segment.columns):
        return np.nan
    if not np.isfinite(path_length_px) or path_length_px <= 0:
        return np.nan
    d = segment[["time_s", "jx_px_s3", "jy_px_s3"]].dropna().sort_values("time_s")
    if len(d) < 2:
        return np.nan
    duration = float(d["time_s"].iloc[-1] - d["time_s"].iloc[0])
    if duration <= 0:
        return np.nan
    squared_jerk = np.square(d["jx_px_s3"].to_numpy(dtype=float)) + np.square(
        d["jy_px_s3"].to_numpy(dtype=float)
    )
    trapezoid = getattr(np, "trapezoid", np.trapz)
    integral = float(trapezoid(squared_jerk, d["time_s"].to_numpy(dtype=float)))
    return (
        float((duration**5 / (path_length_px**2)) * integral)
        if np.isfinite(integral)
        else np.nan
    )


def _speed_curvature_power_law(segment: pd.DataFrame) -> dict[str, float]:
    """Fit log(speed) = intercept + slope*log(curvature) for the 1/3 power law.

    For planar upper-limb drawing/reaching, motor-control work often expects
    speed to decrease with curvature; the classic two-thirds/one-third form is
    roughly ``speed ∝ curvature^-1/3`` on sufficiently curved movement portions.
    """
    required = {"speed_px_s", "curvature_1_px"}
    if not required.issubset(segment.columns):
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": 0}
    d = segment[list(required)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d["speed_px_s"] > 0) & (d["curvature_1_px"] > 0)]
    if len(d) < 4:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(len(d))}
    x = np.log(d["curvature_1_px"].to_numpy(dtype=float))
    y = np.log(d["speed_px_s"].to_numpy(dtype=float))
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(len(d))}
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "n": int(len(d)),
    }


def summarize_metric_distribution(
    df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]
) -> pd.DataFrame:
    """Long-form n/mean/median/CI/log-backtransform summaries by group."""
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    present_groups = [c for c in group_cols if c in df.columns]
    present_metrics = [c for c in metric_cols if c in df.columns]
    if not present_metrics:
        return pd.DataFrame()
    grouped = (
        df.groupby(present_groups, dropna=False) if present_groups else [("all", df)]
    )
    for group_values, g in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = (
            dict(zip(present_groups, group_values))
            if present_groups
            else {"group": "all"}
        )
        for metric in present_metrics:
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            mean_low, mean_high = _ci95_mean(x)
            median_low, median_high = _ci95_median_bootstrap(x)
            row: dict[str, Any] = {
                **base,
                "metric": metric,
                "n": int(len(x)),
                "mean": float(x.mean()) if len(x) else np.nan,
                "mean_ci95_low": mean_low,
                "mean_ci95_high": mean_high,
                "median": float(x.median()) if len(x) else np.nan,
                "median_ci95_low": median_low,
                "median_ci95_high": median_high,
                "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                "sem": _sem(x),
                "skewness": _skewness(x),
            }
            row.update(_log_backtransformed_summary(x))
            rows.append(row)
    return pd.DataFrame(rows)


def _circ_mean_deg(degrees: pd.Series, weights: Optional[pd.Series] = None) -> float:
    a = np.deg2rad(pd.to_numeric(degrees, errors="coerce"))
    mask = np.isfinite(a)
    if not mask.any():
        return np.nan
    if weights is None:
        w = np.ones(mask.sum())
    else:
        w = pd.to_numeric(weights, errors="coerce").to_numpy()[mask]
        w = np.where(np.isfinite(w), w, 0.0)
        if np.sum(w) <= 0:
            w = np.ones(mask.sum())
    val = math.degrees(
        math.atan2(np.sum(np.sin(a[mask]) * w), np.sum(np.cos(a[mask]) * w))
    )
    return float(((val + 180) % 360) - 180)


def _resultant_length(degrees: pd.Series, weights: Optional[pd.Series] = None) -> float:
    a = np.deg2rad(pd.to_numeric(degrees, errors="coerce"))
    mask = np.isfinite(a)
    if not mask.any():
        return np.nan
    if weights is None:
        w = np.ones(mask.sum())
    else:
        w = pd.to_numeric(weights, errors="coerce").to_numpy()[mask]
        w = np.where(np.isfinite(w), w, 0.0)
        if np.sum(w) <= 0:
            w = np.ones(mask.sum())
    return float(
        np.hypot(np.sum(np.cos(a[mask]) * w), np.sum(np.sin(a[mask]) * w)) / np.sum(w)
    )


def _wrap_radians(values: Any) -> Any:
    """Wrap angles to [-pi, pi] while preserving pandas indexes when present."""
    wrapped = (np.asarray(values, dtype=float) + np.pi) % (2 * np.pi) - np.pi
    if isinstance(values, pd.Series):
        return pd.Series(wrapped, index=values.index)
    return wrapped


def _circ_mean_rad(radians: pd.Series) -> float:
    a = pd.to_numeric(radians, errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(math.atan2(np.sum(np.sin(a)), np.sum(np.cos(a))))


def _circ_median_rad(radians: pd.Series) -> float:
    """Circular median approximation centered on the circular mean."""
    a = pd.to_numeric(radians, errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    center = _circ_mean_rad(pd.Series(a))
    centered = _wrap_radians(a - center)
    return float(_wrap_radians(center + np.median(centered)))


def _circ_quantile_rad(radians: pd.Series, q: float) -> float:
    """Circular quantile after unwrapping around the circular mean."""
    a = pd.to_numeric(radians, errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    center = _circ_mean_rad(pd.Series(a))
    centered = _wrap_radians(a - center)
    return float(_wrap_radians(center + np.quantile(centered, q)))


def _set_pi_axis(ax: Any) -> None:
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])


def _direction_label(angle_deg: float) -> str:
    if not np.isfinite(angle_deg):
        return "unknown"
    idx = int(np.digitize([angle_deg], DIRECTION_BINS_DEG, right=False)[0] - 1)
    idx = max(0, min(idx, len(DIRECTION_LABELS) - 1))
    return DIRECTION_LABELS[idx]


def _butter_filter(
    values: pd.Series, fs_hz: float, cutoff_hz: float, btype: str
) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if (
        finite.sum() < 12
        or not np.isfinite(fs_hz)
        or fs_hz <= 0
        or cutoff_hz <= 0
        or cutoff_hz >= fs_hz / 2
    ):
        return pd.Series(out, index=values.index)
    x = pd.Series(arr).interpolate(limit_direction="both").to_numpy(dtype=float)
    if butter is None or filtfilt is None:
        window = max(3, int(round(fs_hz / max(cutoff_hz, 1e-6))))
        if window % 2 == 0:
            window += 1
        low = (
            pd.Series(x)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        y = low if btype == "lowpass" else x - low
    else:
        b, a = butter(2, cutoff_hz / (fs_hz / 2.0), btype=btype)
        try:
            y = filtfilt(b, a, x)
        except Exception:
            return pd.Series(out, index=values.index)
    out[finite] = y[finite]
    return pd.Series(out, index=values.index)


def _estimate_profile_sample_rate_hz(
    df: pd.DataFrame,
    *,
    time_col: str = "time_fraction",
    fs_col: str = "sampling_rate_hz",
) -> float:
    """Estimate a plotting sample rate for profile-level filter overlays."""
    if fs_col in df.columns:
        fs = pd.to_numeric(df[fs_col], errors="coerce").dropna()
        if not fs.empty and float(fs.median()) > 0:
            return float(fs.median())
    if time_col in df.columns:
        t = (
            pd.to_numeric(df[time_col], errors="coerce")
            .dropna()
            .sort_values()
            .drop_duplicates()
        )
        dt = t.diff().dropna()
        dt = dt[dt > 0]
        if not dt.empty:
            return float(1.0 / dt.median())
    return np.nan


def _profile_filter_pair(
    values: pd.Series,
    time_values: pd.Series,
    *,
    fs_hz: float,
    cutoff_hz: float = 10.0,
) -> tuple[pd.Series, pd.Series, float]:
    """Return LPF/HPF traces, clipping cutoff when profile bins are under-sampled."""
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        fs_hz = _estimate_profile_sample_rate_hz(
            pd.DataFrame({"time_fraction": time_values})
        )
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric, numeric * 0.0, np.nan
    effective_cutoff = min(float(cutoff_hz), fs_hz * 0.45)
    if not np.isfinite(effective_cutoff) or effective_cutoff <= 0:
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric, numeric * 0.0, np.nan
    low = _butter_filter(values, fs_hz=fs_hz, cutoff_hz=effective_cutoff, btype="lowpass")
    high = _butter_filter(values, fs_hz=fs_hz, cutoff_hz=effective_cutoff, btype="highpass")
    if low.notna().sum() == 0:
        numeric = pd.to_numeric(values, errors="coerce")
        low = numeric.rolling(5, center=True, min_periods=1).mean()
        high = numeric - low
    return low, high, effective_cutoff


def _mask_filter_plot_values(
    values: pd.Series,
    *,
    min_abs: float = 1e-1,
    max_abs: float = 1e4,
) -> pd.Series:
    """Hide near-zero speckles and extreme spikes from filter-overlay plots."""
    numeric = pd.to_numeric(values, errors="coerce")
    keep = numeric.abs().between(min_abs, max_abs, inclusive="both")
    return numeric.where(keep)


def _first_numeric(row: pd.Series, names: list[str], default: float = np.nan) -> float:
    for name in names:
        if name in row:
            val = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val)
    return default


def _first_value(row: pd.Series, names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row and pd.notna(row.get(name)):
            return row.get(name)
    return default


def _read_answers_table(answer_file: Path) -> pd.DataFrame:
    """Read current headered answers.csv and tolerate older headerless exports."""
    answers = read_csv_flexible(answer_file)
    answers.columns = [str(c).strip() for c in answers.columns]
    lower_to_original = {str(c).strip().lower(): c for c in answers.columns}
    if {"pair_number", "object_1_stiffness", "object_2_stiffness"}.issubset(
        lower_to_original
    ):
        return answers.rename(
            columns={orig: name for name, orig in lower_to_original.items()}
        )

    # Legacy fallback: some early files were headerless.  If the first row is a
    # textual header accidentally parsed as data, drop it before assigning names.
    raw = read_csv_flexible(answer_file, header=None)
    header = [
        "timestamp",
        "pair_number",
        "object_1_finger",
        "object_1_stiffness",
        "object_2_finger",
        "object_2_stiffness",
        "time_to_answer",
        "answer",
    ]
    if not raw.empty and str(raw.iloc[0, 1]).strip().lower() == "pair_number":
        raw = raw.iloc[1:].reset_index(drop=True)
    raw = raw.iloc[:, : len(header)].copy()
    raw.columns = header[: raw.shape[1]]
    return raw


def _comparison_and_standard_stiffness(
    stiffness_1: float, stiffness_2: float
) -> tuple[float, float]:
    values = [v for v in [stiffness_1, stiffness_2] if np.isfinite(v)]
    if not values:
        return np.nan, np.nan
    standard = (
        85.0
        if any(abs(v - 85.0) < 1e-9 for v in values)
        else (stiffness_2 if np.isfinite(stiffness_2) else values[-1])
    )
    non_standard = [v for v in values if abs(v - standard) > 1e-9]
    comparison = non_standard[0] if non_standard else values[0]
    return float(comparison), float(standard)


def _correct_response_from_answer(
    stiffness_1: float, stiffness_2: float, answer: float
) -> float:
    """Return whether the selected left/right object was the stiffer object.

    Frontends store answers as left=0 and right=1. Equal-stiffness comparisons do
    not have an objectively stiffer side, so they remain NaN.
    """
    if not (
        np.isfinite(stiffness_1) and np.isfinite(stiffness_2) and np.isfinite(answer)
    ):
        return np.nan
    if stiffness_1 == stiffness_2:
        return np.nan
    chosen = (
        stiffness_1 if int(answer) == 0 else stiffness_2 if int(answer) == 1 else np.nan
    )
    if not np.isfinite(chosen):
        return np.nan
    return float(chosen == max(stiffness_1, stiffness_2))


def discover_trials(data_root: Path) -> pd.DataFrame:
    """Discover answers.csv files and map answer rows to pair tracking/video files."""
    rows: list[dict[str, Any]] = []
    for subject_dir in sorted(
        p
        for p in Path(data_root).iterdir()
        if p.is_dir() and not should_ignore_analysis_path(p)
    ):
        all_answer_files = sorted(subject_dir.rglob("answers.csv"))
        answer_files = sorted(
            p for p in all_answer_files if not should_ignore_analysis_path(p)
        )
        if not answer_files:
            if all_answer_files:
                continue
            subject_group = subject_dir.name[:1].upper()
            rows.append(
                {
                    "subject_id": subject_dir.name,
                    "subject_group": subject_group,
                    EXPERIMENT_GROUP_COLUMN: experiment_group_for_subject(
                        subject_dir.name, subject_group
                    ),
                    "selected": False,
                    "warning": "no answers.csv found",
                }
            )
            continue
        answer_file = sorted(
            answer_files,
            key=lambda p: (len(list(p.parent.glob("pair_*"))), p.stat().st_mtime),
            reverse=True,
        )[0]
        try:
            answers = _read_answers_table(answer_file)
        except Exception as exc:
            subject_group = subject_dir.name[:1].upper()
            rows.append(
                {
                    "subject_id": subject_dir.name,
                    "subject_group": subject_group,
                    EXPERIMENT_GROUP_COLUMN: experiment_group_for_subject(
                        subject_dir.name, subject_group
                    ),
                    "selected": False,
                    "source_file": str(answer_file),
                    "warning": f"could not read answers: {exc}",
                }
            )
            continue
        for i, row in answers.reset_index(drop=True).iterrows():
            pair_number = _first_numeric(row, ["pair_number"], float(i + 1))
            trial_index = (
                int(pair_number)
                if np.isfinite(pair_number) and pair_number > 0
                else int(i + 1)
            )
            pair_dir = answer_file.parent / f"pair_{trial_index:03d}"
            object_1_stiffness = _first_numeric(
                row, ["object_1_stiffness", "first_stiffness", "stiffness_1"]
            )
            object_2_stiffness = _first_numeric(
                row, ["object_2_stiffness", "second_stiffness", "stiffness_2"]
            )
            object_1_finger = normalize_finger_condition(
                _first_value(row, ["object_1_finger", "first_finger"], np.nan)
            )
            object_2_finger = normalize_finger_condition(
                _first_value(row, ["object_2_finger", "second_finger"], np.nan)
            )
            finger_condition = (
                object_1_finger if object_1_finger == object_2_finger else np.nan
            )
            comparison_value, standard_value = _comparison_and_standard_stiffness(
                object_1_stiffness, object_2_stiffness
            )
            response_code = _first_numeric(
                row, ["answer", "response", "question_input"]
            )
            correct = _correct_response_from_answer(
                object_1_stiffness, object_2_stiffness, response_code
            )
            tracking_file = pair_dir / "tracking.csv"
            side_video = pair_dir / "side_camera.mp4"
            rows.append(
                {
                    "subject_id": subject_dir.name,
                    "subject_group": subject_dir.name[:1].upper(),
                    EXPERIMENT_GROUP_COLUMN: experiment_group_for_subject(
                        subject_dir.name, subject_dir.name[:1].upper()
                    ),
                    "selected": True,
                    "source_file": str(answer_file),
                    "run_dir": str(answer_file.parent),
                    "trial_index_raw": trial_index,
                    "pair_dir": str(pair_dir),
                    "tracking_file": str(tracking_file),
                    "side_video_file": str(side_video),
                    "top_video_file": str(pair_dir / "top_camera.mp4"),
                    "tracking_exists": tracking_file.exists(),
                    "side_video_exists": side_video.exists(),
                    "pair_number": trial_index,
                    "object_1_finger": object_1_finger,
                    "object_1_stiffness": object_1_stiffness,
                    "object_2_finger": object_2_finger,
                    "object_2_stiffness": object_2_stiffness,
                    "finger_condition": finger_condition,
                    "time_to_answer_s": _first_numeric(
                        row, ["time_to_answer", "response_time_s"]
                    ),
                    "answer_code": response_code,
                    "comparison_value": comparison_value,
                    "standard_value": standard_value,
                    "signed_stiffness_delta": comparison_value - standard_value
                    if np.isfinite(comparison_value) and np.isfinite(standard_value)
                    else np.nan,
                    "correct_response": correct,
                    "warning": "",
                }
            )
    return pd.DataFrame(rows)


def compute_tracking_kinematics(
    trials: pd.DataFrame,
    *,
    center_x: float = CENTER_X,
    center_y: float = CENTER_Y,
    lowpass_cutoff_hz: float = 3.0,
    highpass_cutoff_hz: float = 0.10,
    n_time_bins: int = 50,
    max_trials: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Compute 2D kinematics separately for each participant x stiffness.

    Tracking files contain one row per video sample and a ``stiffness`` column
    that changes when the experiment switches from object 1 to object 2.  The
    analysis therefore treats each contiguous non-zero stiffness block as the
    unit of analysis. Velocity is derived from 2D location within that block and
    acceleration is derived from that velocity, so derivatives never bridge two
    different stiffness values.
    """
    sample_rows: list[pd.DataFrame] = []
    segment_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    time_rows: list[pd.DataFrame] = []
    selected = (
        trials[trials.get("tracking_exists", False).astype(bool)]
        .copy()
        .sort_values(["subject_id", "trial_index_raw"])
    )
    if max_trials is not None:
        selected = selected.head(max_trials)

    for _, meta in selected.iterrows():
        path = Path(str(meta["tracking_file"]))
        try:
            df = read_csv_flexible(path)
        except Exception as exc:
            segment_rows.append(
                {**meta.to_dict(), "tracking_warning": f"read_failed: {exc}"}
            )
            continue
        required = {"timestamp", "object_x", "object_y"}
        if not required.issubset(df.columns):
            segment_rows.append(
                {**meta.to_dict(), "tracking_warning": "missing tracking columns"}
            )
            continue

        d = df.copy()
        d["timestamp_dt"] = pd.to_datetime(
            d["timestamp"], errors="coerce", format="ISO8601"
        )
        if d["timestamp_dt"].notna().any():
            d["time_s"] = (
                d["timestamp_dt"] - d["timestamp_dt"].min()
            ).dt.total_seconds()
        else:
            d["time_s"] = np.arange(len(d)) / 30.0
        d["trial_time_s"] = d["time_s"]
        trial_duration = (
            float(d["time_s"].max()) if d["time_s"].notna().any() else np.nan
        )
        d["trial_time_fraction"] = np.where(
            trial_duration > 0, d["time_s"] / trial_duration, 0.0
        )
        d["trial_time_fraction"] = (
            pd.to_numeric(d["trial_time_fraction"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        d["x_px"] = pd.to_numeric(d["object_x"], errors="coerce")
        d["y_px"] = pd.to_numeric(d["object_y"], errors="coerce")
        d["x_centered_px"] = d["x_px"] - center_x
        d["y_centered_px"] = d["y_px"] - center_y
        d["r_center_px"] = np.hypot(d["x_centered_px"], d["y_centered_px"])
        d["position_angle_deg"] = np.degrees(
            np.arctan2(d["y_centered_px"], d["x_centered_px"])
        )
        d["interacting_bool"] = (
            d.get("interacting", False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        if "finger" in d.columns:
            d["finger_condition"] = d["finger"].map(normalize_finger_condition)
        else:
            d["finger_condition"] = normalize_finger_condition(
                meta.get("finger_condition", np.nan)
            )
        d["stiffness_value"] = pd.to_numeric(
            d.get("stiffness", meta.get("comparison_value", np.nan)), errors="coerce"
        )
        d["skin_stretch_gain_mm_per_m"] = d["stiffness_value"]
        d["valid_stiffness_bool"] = d["stiffness_value"].notna() & (
            d["stiffness_value"] > 0
        )
        # Segment id increments at every stiffness transition; this keeps two
        # repeated values separate if a future protocol revisits a stiffness.
        stiffness_key = d["stiffness_value"].where(
            d["stiffness_value"].notna(), "missing"
        )
        d["stiffness_segment_id"] = (
            stiffness_key.ne(stiffness_key.shift()).cumsum().astype(int)
        )
        d["stiffness_order_in_trial"] = (
            d.groupby("stiffness_segment_id", sort=False).ngroup() + 1
        )

        whole_dt = d["time_s"].diff().replace(0, np.nan)
        median_dt = (
            float(whole_dt.dropna().median()) if whole_dt.dropna().size else 1.0 / 30.0
        )
        fs = 1.0 / median_dt if median_dt > 0 else 30.0
        d["x_lpf_px"] = _butter_filter(
            d["x_centered_px"], fs, lowpass_cutoff_hz, "lowpass"
        )
        d["y_lpf_px"] = _butter_filter(
            d["y_centered_px"], fs, lowpass_cutoff_hz, "lowpass"
        )
        d["x_hpf_px"] = _butter_filter(
            d["x_centered_px"], fs, highpass_cutoff_hz, "highpass"
        )
        d["y_hpf_px"] = _butter_filter(
            d["y_centered_px"], fs, highpass_cutoff_hz, "highpass"
        )

        x_for_deriv = d["x_lpf_px"].where(d["x_lpf_px"].notna(), d["x_centered_px"])
        y_for_deriv = d["y_lpf_px"].where(d["y_lpf_px"].notna(), d["y_centered_px"])
        segment_dt = (
            d.groupby("stiffness_segment_id", sort=False)["time_s"]
            .diff()
            .replace(0, np.nan)
        )
        d["vx_px_s"] = (
            x_for_deriv.groupby(d["stiffness_segment_id"], sort=False).diff()
            / segment_dt
        )
        d["vy_px_s"] = (
            y_for_deriv.groupby(d["stiffness_segment_id"], sort=False).diff()
            / segment_dt
        )
        d["speed_px_s"] = np.hypot(d["vx_px_s"], d["vy_px_s"])
        d["movement_angle_deg"] = np.degrees(np.arctan2(d["vy_px_s"], d["vx_px_s"]))
        d["movement_direction"] = d["movement_angle_deg"].map(_direction_label)
        d["ax_px_s2"] = (
            d.groupby("stiffness_segment_id", sort=False)["vx_px_s"].diff() / segment_dt
        )
        d["ay_px_s2"] = (
            d.groupby("stiffness_segment_id", sort=False)["vy_px_s"].diff() / segment_dt
        )
        d["acceleration_px_s2"] = np.hypot(d["ax_px_s2"], d["ay_px_s2"])
        curvature_num = (d["vx_px_s"] * d["ay_px_s2"] - d["vy_px_s"] * d["ax_px_s2"]).abs()
        curvature_den = np.power(d["speed_px_s"], 3)
        d["curvature_1_px"] = np.where(curvature_den > 0, curvature_num / curvature_den, np.nan)
        d["jx_px_s3"] = (
            d.groupby("stiffness_segment_id", sort=False)["ax_px_s2"].diff() / segment_dt
        )
        d["jy_px_s3"] = (
            d.groupby("stiffness_segment_id", sort=False)["ay_px_s2"].diff() / segment_dt
        )
        d["jerk_px_s3"] = np.hypot(d["jx_px_s3"], d["jy_px_s3"])
        d["radial_velocity_px_s"] = (
            d.groupby("stiffness_segment_id", sort=False)["r_center_px"].diff()
            / segment_dt
        )
        theta = np.deg2rad(d["position_angle_deg"])
        d["tangential_velocity_px_s"] = (
            -np.sin(theta) * d["vx_px_s"] + np.cos(theta) * d["vy_px_s"]
        )

        segment_start = d.groupby("stiffness_segment_id", sort=False)[
            "time_s"
        ].transform("min")
        segment_end = d.groupby("stiffness_segment_id", sort=False)["time_s"].transform(
            "max"
        )
        segment_duration = segment_end - segment_start
        d["stiffness_time_s"] = d["time_s"] - segment_start
        d["stiffness_duration_s"] = segment_duration
        d["stiffness_start_time_s"] = segment_start
        d["stiffness_end_time_s"] = segment_end
        d["stiffness_start_fraction"] = np.where(
            trial_duration > 0, segment_start / trial_duration, 0.0
        )
        d["stiffness_end_fraction"] = np.where(
            trial_duration > 0, segment_end / trial_duration, 1.0
        )
        d["stiffness_time_fraction"] = np.where(
            segment_duration > 0, d["stiffness_time_s"] / segment_duration, 0.0
        )
        d["stiffness_time_fraction"] = (
            pd.to_numeric(d["stiffness_time_fraction"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        # Keep the legacy column name, but make it explicit that time bins are
        # now normalized within each stiffness block, not across the whole pair.
        d["time_fraction"] = d["stiffness_time_fraction"]
        d["trajectory_time_bin"] = np.minimum(
            (d["stiffness_time_fraction"] * n_time_bins).astype(int) + 1, n_time_bins
        )

        for col, val in meta.items():
            if col not in d.columns:
                d[col] = val
        d = _add_hand_actor_columns(d)
        d = add_workspace_normalization_columns(d)
        for optional_col in [
            "thumb_x_centered_px",
            "thumb_y_centered_px",
            "active_finger_x_centered_px",
            "active_finger_y_centered_px",
            "thumb_active_dx_px",
            "thumb_active_dy_px",
            "thumb_active_span_px",
            "hand_orientation_xy_deg",
        ]:
            if optional_col not in d.columns:
                d[optional_col] = np.nan

        valid = d[d["valid_stiffness_bool"]].copy()
        pair_rows.append(
            {
                **meta.to_dict(),
                "tracking_warning": "",
                "n_tracking_samples": int(len(d)),
                "n_stiffness_segments": int(valid["stiffness_segment_id"].nunique())
                if not valid.empty
                else 0,
                "stiffness_values_present": ";".join(
                    str(float(v)).rstrip("0").rstrip(".")
                    for v in sorted(valid["stiffness_value"].dropna().unique())
                ),
                "duration_s": trial_duration,
                "sampling_rate_hz": fs,
            }
        )

        if valid.empty:
            continue

        for segment_id, seg in valid.groupby("stiffness_segment_id", sort=False):
            seg = seg.copy()
            movement = seg[seg["interacting_bool"]].copy()
            if len(movement) < 3:
                movement = seg.copy()
            path_len = float(
                np.nansum(
                    np.hypot(seg["x_centered_px"].diff(), seg["y_centered_px"].diff())
                )
            )
            net_disp = (
                float(
                    np.hypot(
                        seg["x_centered_px"].iloc[-1] - seg["x_centered_px"].iloc[0],
                        seg["y_centered_px"].iloc[-1] - seg["y_centered_px"].iloc[0],
                    )
                )
                if len(seg)
                else np.nan
            )
            dominant_angle = _circ_mean_deg(
                movement["movement_angle_deg"], movement["speed_px_s"].fillna(0)
            )
            counts = movement["movement_direction"].value_counts(normalize=True)
            entropy = (
                float(-(counts * np.log2(counts)).sum()) if len(counts) else np.nan
            )
            speed_curvature = _speed_curvature_power_law(movement)
            stiffness_value = (
                float(seg["stiffness_value"].dropna().median())
                if seg["stiffness_value"].notna().any()
                else np.nan
            )
            segment_rows.append(
                {
                    **meta.to_dict(),
                    "finger_condition": movement["finger_condition"].dropna().iloc[0]
                    if movement["finger_condition"].notna().any()
                    else meta.get("finger_condition", np.nan),
                    "tracking_warning": "",
                    "stiffness_value": stiffness_value,
                    "skin_stretch_gain_mm_per_m": stiffness_value,
                    "stiffness_segment_id": int(segment_id),
                    "stiffness_order_in_trial": int(
                        seg["stiffness_order_in_trial"].iloc[0]
                    ),
                    "stiffness_start_time_s": float(
                        seg["stiffness_start_time_s"].iloc[0]
                    ),
                    "stiffness_end_time_s": float(seg["stiffness_end_time_s"].iloc[0]),
                    "stiffness_start_fraction": float(
                        seg["stiffness_start_fraction"].iloc[0]
                    ),
                    "stiffness_end_fraction": float(
                        seg["stiffness_end_fraction"].iloc[0]
                    ),
                    "n_tracking_samples": int(len(seg)),
                    "duration_s": float(seg["stiffness_duration_s"].iloc[0]),
                    "pair_duration_s": trial_duration,
                    "sampling_rate_hz": fs,
                    "mean_x_centered_px": float(movement["x_centered_px"].mean()),
                    "mean_y_centered_px": float(movement["y_centered_px"].mean()),
                    "mean_x_workspace_cm": float(movement["x_workspace_cm"].mean())
                    if "x_workspace_cm" in movement.columns
                    else np.nan,
                    "mean_y_workspace_cm": float(movement["y_workspace_cm"].mean())
                    if "y_workspace_cm" in movement.columns
                    else np.nan,
                    "mean_r_workspace_cm": float(movement["r_workspace_cm"].mean())
                    if "r_workspace_cm" in movement.columns
                    else np.nan,
                    "mean_r_workspace_normalized": float(
                        movement["r_workspace_normalized"].mean()
                    )
                    if "r_workspace_normalized" in movement.columns
                    else np.nan,
                    "workspace_setup": movement["workspace_setup"].dropna().iloc[0]
                    if "workspace_setup" in movement.columns
                    and movement["workspace_setup"].notna().any()
                    else np.nan,
                    "workspace_label": movement["workspace_label"].dropna().iloc[0]
                    if "workspace_label" in movement.columns
                    and movement["workspace_label"].notna().any()
                    else np.nan,
                    "workspace_width_cm": float(movement["workspace_width_cm"].mean())
                    if "workspace_width_cm" in movement.columns
                    else np.nan,
                    "workspace_height_cm": float(movement["workspace_height_cm"].mean())
                    if "workspace_height_cm" in movement.columns
                    else np.nan,
                    "side_camera_side": movement["side_camera_side"].dropna().iloc[0]
                    if "side_camera_side" in movement.columns
                    and movement["side_camera_side"].notna().any()
                    else np.nan,
                    "participant_position_context": movement[
                        "participant_position_context"
                    ]
                    .dropna()
                    .iloc[0]
                    if "participant_position_context" in movement.columns
                    and movement["participant_position_context"].notna().any()
                    else np.nan,
                    "movement_space_context": movement[
                        "movement_space_context"
                    ].dropna().iloc[0]
                    if "movement_space_context" in movement.columns
                    and movement["movement_space_context"].notna().any()
                    else np.nan,
                    "side_camera_interpretation_note": movement[
                        "side_camera_interpretation_note"
                    ]
                    .dropna()
                    .iloc[0]
                    if "side_camera_interpretation_note" in movement.columns
                    and movement["side_camera_interpretation_note"].notna().any()
                    else np.nan,
                    "mean_r_center_px": float(movement["r_center_px"].mean()),
                    "max_r_center_px": float(movement["r_center_px"].max()),
                    "mean_thumb_active_span_px": float(
                        movement["thumb_active_span_px"].mean()
                    )
                    if "thumb_active_span_px" in movement.columns
                    else np.nan,
                    "mean_hand_orientation_xy_deg": _circ_mean_deg(
                        movement["hand_orientation_xy_deg"]
                    )
                    if "hand_orientation_xy_deg" in movement.columns
                    else np.nan,
                    "path_length_px": path_len,
                    "net_displacement_px": net_disp,
                    "straightness_index": net_disp / path_len
                    if path_len > 0
                    else np.nan,
                    "mean_vx_px_s": float(movement["vx_px_s"].mean()),
                    "mean_vy_px_s": float(movement["vy_px_s"].mean()),
                    "mean_speed_px_s": float(movement["speed_px_s"].mean()),
                    "max_speed_px_s": float(movement["speed_px_s"].max()),
                    "mean_ax_px_s2": float(movement["ax_px_s2"].mean()),
                    "mean_ay_px_s2": float(movement["ay_px_s2"].mean()),
                    "mean_acceleration_px_s2": float(
                        movement["acceleration_px_s2"].mean()
                    ),
                    "max_acceleration_px_s2": float(
                        movement["acceleration_px_s2"].max()
                    ),
                    "mean_jerk_px_s3": float(movement["jerk_px_s3"].mean()),
                    "max_jerk_px_s3": float(movement["jerk_px_s3"].max()),
                    "normalized_jerk_cost": _normalized_jerk_cost(movement, path_length_px=path_len),
                    "mean_curvature_1_px": float(movement["curvature_1_px"].mean()),
                    "median_curvature_1_px": float(movement["curvature_1_px"].median()),
                    "speed_curvature_power_law_slope": speed_curvature["slope"],
                    "speed_curvature_power_law_intercept": speed_curvature["intercept"],
                    "speed_curvature_power_law_r2": speed_curvature["r2"],
                    "speed_curvature_power_law_n": speed_curvature["n"],
                    "mean_radial_velocity_px_s": float(
                        movement["radial_velocity_px_s"].mean()
                    ),
                    "mean_abs_radial_velocity_px_s": float(
                        movement["radial_velocity_px_s"].abs().mean()
                    ),
                    "mean_abs_tangential_velocity_px_s": float(
                        movement["tangential_velocity_px_s"].abs().mean()
                    ),
                    "dominant_movement_angle_deg": dominant_angle,
                    "dominant_movement_direction": _direction_label(dominant_angle),
                    "movement_direction_resultant_length": _resultant_length(
                        movement["movement_angle_deg"], movement["speed_px_s"].fillna(0)
                    ),
                    "movement_direction_entropy_bits": entropy,
                    "mean_skin_stretch_gain_mm_per_m": stiffness_value,
                }
            )

        binned = (
            valid.groupby(["stiffness_segment_id", "trajectory_time_bin"], dropna=False)
            .agg(
                stiffness_value=("stiffness_value", "median"),
                stiffness_order_in_trial=("stiffness_order_in_trial", "first"),
                time_fraction=("stiffness_time_fraction", "mean"),
                stiffness_time_fraction=("stiffness_time_fraction", "mean"),
                trial_time_fraction=("trial_time_fraction", "mean"),
                stiffness_start_fraction=("stiffness_start_fraction", "first"),
                stiffness_end_fraction=("stiffness_end_fraction", "first"),
                x_centered_px=("x_centered_px", "mean"),
                y_centered_px=("y_centered_px", "mean"),
                r_center_px=("r_center_px", "mean"),
                vx_px_s=("vx_px_s", "mean"),
                vy_px_s=("vy_px_s", "mean"),
                speed_px_s=("speed_px_s", "mean"),
                ax_px_s2=("ax_px_s2", "mean"),
                ay_px_s2=("ay_px_s2", "mean"),
                acceleration_px_s2=("acceleration_px_s2", "mean"),
                curvature_1_px=("curvature_1_px", "mean"),
                jx_px_s3=("jx_px_s3", "mean"),
                jy_px_s3=("jy_px_s3", "mean"),
                jerk_px_s3=("jerk_px_s3", "mean"),
                radial_velocity_px_s=("radial_velocity_px_s", "mean"),
                tangential_velocity_px_s=("tangential_velocity_px_s", "mean"),
                movement_angle_deg=("movement_angle_deg", lambda s: _circ_mean_deg(s)),
                x_workspace_cm=("x_workspace_cm", "mean"),
                y_workspace_cm=("y_workspace_cm", "mean"),
                r_workspace_cm=("r_workspace_cm", "mean"),
                r_workspace_normalized=("r_workspace_normalized", "mean"),
                workspace_setup=("workspace_setup", "first"),
                workspace_label=("workspace_label", "first"),
                workspace_width_cm=("workspace_width_cm", "first"),
                workspace_height_cm=("workspace_height_cm", "first"),
                side_camera_side=("side_camera_side", "first"),
                participant_position_context=("participant_position_context", "first"),
                movement_space_context=("movement_space_context", "first"),
                side_camera_interpretation_note=(
                    "side_camera_interpretation_note",
                    "first",
                ),
                thumb_x_centered_px=("thumb_x_centered_px", "mean"),
                thumb_y_centered_px=("thumb_y_centered_px", "mean"),
                active_finger_x_centered_px=("active_finger_x_centered_px", "mean"),
                active_finger_y_centered_px=("active_finger_y_centered_px", "mean"),
                thumb_active_dx_px=("thumb_active_dx_px", "mean"),
                thumb_active_dy_px=("thumb_active_dy_px", "mean"),
                thumb_active_span_px=("thumb_active_span_px", "mean"),
                hand_orientation_xy_deg=(
                    "hand_orientation_xy_deg",
                    lambda s: _circ_mean_deg(s),
                ),
            )
            .reset_index()
        )
        for col, val in meta.items():
            binned[col] = val
        binned["finger_condition"] = (
            valid.groupby("stiffness_segment_id", sort=False)["finger_condition"]
            .first()
            .reindex(binned["stiffness_segment_id"])
            .to_numpy()
        )
        time_rows.append(binned)

        keep_cols = [
            "subject_id",
            "subject_group",
            EXPERIMENT_GROUP_COLUMN,
            "trial_index_raw",
            "pair_number",
            "finger_condition",
            "object_1_stiffness",
            "object_2_stiffness",
            "comparison_value",
            "standard_value",
            "signed_stiffness_delta",
            "correct_response",
            "answer_code",
            "stiffness_value",
            "skin_stretch_gain_mm_per_m",
            "stiffness_segment_id",
            "stiffness_order_in_trial",
            "time_s",
            "trial_time_fraction",
            "stiffness_time_s",
            "stiffness_time_fraction",
            "time_fraction",
            "stiffness_start_fraction",
            "stiffness_end_fraction",
            "x_centered_px",
            "y_centered_px",
            "r_center_px",
            "x_workspace_cm",
            "y_workspace_cm",
            "r_workspace_cm",
            "r_workspace_normalized",
            "workspace_setup",
            "workspace_label",
            "workspace_width_cm",
            "workspace_height_cm",
            "side_camera_side",
            "participant_position_context",
            "movement_space_context",
            "side_camera_interpretation_note",
            "position_angle_deg",
            "thumb_x_centered_px",
            "thumb_y_centered_px",
            "active_finger_x_centered_px",
            "active_finger_y_centered_px",
            "thumb_active_dx_px",
            "thumb_active_dy_px",
            "thumb_active_span_px",
            "hand_orientation_xy_deg",
            "hand_midpoint_x_centered_px",
            "hand_midpoint_y_centered_px",
            "x_lpf_px",
            "y_lpf_px",
            "x_hpf_px",
            "y_hpf_px",
            "vx_px_s",
            "vy_px_s",
            "speed_px_s",
            "movement_angle_deg",
            "movement_direction",
            "ax_px_s2",
            "ay_px_s2",
            "acceleration_px_s2",
            "curvature_1_px",
            "jx_px_s3",
            "jy_px_s3",
            "jerk_px_s3",
            "radial_velocity_px_s",
            "tangential_velocity_px_s",
            "interacting_bool",
            "tracking_file",
        ]
        sample_rows.append(d[[c for c in keep_cols if c in d.columns]])

    return {
        "samples": pd.concat(sample_rows, ignore_index=True, sort=False)
        if sample_rows
        else pd.DataFrame(),
        # Segment-level rows: one row per participant x pair x stiffness block.
        "trial_summary": pd.DataFrame(segment_rows),
        "pair_summary": pd.DataFrame(pair_rows),
        "time_bins": pd.concat(time_rows, ignore_index=True, sort=False)
        if time_rows
        else pd.DataFrame(),
    }


def _skin_mask_z_from_frame(frame: np.ndarray) -> dict[str, float]:
    _, frame_width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([25, 230, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 25, 35]), np.array([180, 230, 255]))
    mask = mask1 | mask2
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ys, xs = np.where(mask > 0)
    if len(ys) < 100:
        return {
            "side_detected": 0,
            "side_z_top_y_px": np.nan,
            "side_z_centroid_y_px": np.nan,
            "side_x_centroid_px": np.nan,
            "side_x_from_frame_center_px": np.nan,
            "side_mask_area_px": float(len(ys)),
        }
    x_centroid = float(np.mean(xs))
    return {
        "side_detected": 1,
        "side_z_top_y_px": float(np.percentile(ys, 5)),
        "side_z_centroid_y_px": float(np.mean(ys)),
        "side_x_centroid_px": x_centroid,
        "side_x_from_frame_center_px": x_centroid - (float(frame_width) / 2.0),
        "side_mask_area_px": float(len(ys)),
    }


def add_side_camera_angle_normalization_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pixel-space side-camera mirror correction for L/right vs N/left views.

    Raw columns such as ``side_x_centroid_px`` and
    ``side_x_from_frame_center_px`` stay untouched.  Corrected columns mirror
    left-camera recordings into the same right-camera convention, which makes
    side-view orientation and direction comparable in pixels without using real
    workspace centimeters.
    """

    if df.empty:
        return df.copy()
    out = add_workspace_normalization_columns(df)
    if "side_camera_view_sign" not in out.columns:
        out["side_camera_view_sign"] = out["side_camera_side"].map(
            side_camera_view_sign_for_side
        )
    if "side_x_from_frame_center_px" in out.columns:
        raw_x = pd.to_numeric(out["side_x_from_frame_center_px"], errors="coerce")
        sign = pd.to_numeric(out["side_camera_view_sign"], errors="coerce")
        out["side_x_from_center_camera_corrected_px"] = raw_x * sign
    if {"side_z_lift_px", "side_x_from_frame_center_px"}.issubset(out.columns):
        out["side_lift_lateral_angle_raw_deg"] = np.degrees(
            np.arctan2(
                pd.to_numeric(out["side_z_lift_px"], errors="coerce"),
                pd.to_numeric(out["side_x_from_frame_center_px"], errors="coerce"),
            )
        )
    if {"side_z_lift_px", "side_x_from_center_camera_corrected_px"}.issubset(
        out.columns
    ):
        out["side_lift_lateral_angle_camera_corrected_deg"] = np.degrees(
            np.arctan2(
                pd.to_numeric(out["side_z_lift_px"], errors="coerce"),
                pd.to_numeric(
                    out["side_x_from_center_camera_corrected_px"], errors="coerce"
                ),
            )
        )
    return out


def estimate_side_video_z(
    trial_summary: pd.DataFrame,
    *,
    samples_per_video: int = 6,
    max_trials: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Estimate side-view vertical motion in pixels from side_camera.mp4.

    When ``trial_summary`` is the segment-level table from
    :func:`compute_tracking_kinematics`, side frames are sampled only over that
    stiffness block's time window. This keeps side-view summaries aligned with
    participant x stiffness, matching the top-view kinematic outputs.
    """
    rows: list[dict[str, Any]] = []
    selected = (
        trial_summary[trial_summary.get("side_video_exists", False).astype(bool)]
        .copy()
        .sort_values(["subject_id", "trial_index_raw"])
    )
    selected = add_workspace_normalization_columns(selected)
    if max_trials is not None:
        selected = selected.head(max_trials)
    for _, meta in selected.iterrows():
        video = Path(str(meta.get("side_video_file", "")))
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            rows.append(
                {**meta.to_dict(), "side_video_warning": "could_not_open_side_video"}
            )
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or np.nan)
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or np.nan)
        start_fraction = pd.to_numeric(
            pd.Series([meta.get("stiffness_start_fraction", 0.0)]), errors="coerce"
        ).iloc[0]
        end_fraction = pd.to_numeric(
            pd.Series([meta.get("stiffness_end_fraction", 1.0)]), errors="coerce"
        ).iloc[0]
        if not np.isfinite(start_fraction):
            start_fraction = 0.0
        if not np.isfinite(end_fraction):
            end_fraction = 1.0
        start_fraction = float(np.clip(start_fraction, 0.0, 1.0))
        end_fraction = float(np.clip(max(end_fraction, start_fraction), 0.0, 1.0))
        start_frame = int(round(start_fraction * max(0, frame_count - 1)))
        end_frame = int(round(end_fraction * max(0, frame_count - 1)))
        positions = np.linspace(
            start_frame, max(start_frame, end_frame), max(1, samples_per_video)
        ).astype(int)
        trial_rows: list[dict[str, Any]] = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if not ok:
                continue
            trial_fraction = (
                float(pos / max(1, frame_count - 1)) if frame_count > 1 else 0.0
            )
            denom = max(end_fraction - start_fraction, 1e-12)
            side_time_fraction = float(
                np.clip((trial_fraction - start_fraction) / denom, 0.0, 1.0)
            )
            trial_rows.append(
                {
                    **meta.to_dict(),
                    "side_video_file": str(video),
                    "frame_index": int(pos),
                    "side_time_s": float(pos / fps) if fps > 0 else np.nan,
                    "side_trial_time_fraction": trial_fraction,
                    "side_time_fraction": side_time_fraction,
                    "side_fps": fps,
                    "side_frame_count": frame_count,
                    "side_frame_width": width,
                    "side_frame_height": height,
                    **_skin_mask_z_from_frame(frame),
                    "side_video_warning": "",
                }
            )
        cap.release()
        if trial_rows:
            vals = pd.DataFrame(trial_rows)
            detected = vals[vals["side_detected"] == 1]
            baseline = (
                float(detected["side_z_top_y_px"].quantile(0.95))
                if not detected.empty
                else np.nan
            )
            vals["side_z_lift_px"] = baseline - vals["side_z_top_y_px"]
            vals["side_z_lift_centroid_px"] = (
                float(vals["side_z_centroid_y_px"].quantile(0.95))
                - vals["side_z_centroid_y_px"]
                if vals["side_z_centroid_y_px"].notna().any()
                else np.nan
            )
            vals = add_side_camera_angle_normalization_columns(vals)
            vals = vals.sort_values("side_time_s").copy()
            dt = pd.to_numeric(vals["side_time_s"], errors="coerce").diff().replace(
                0, np.nan
            )
            if "side_x_from_frame_center_px" in vals.columns:
                vals["side_lateral_velocity_raw_px_s"] = (
                    pd.to_numeric(vals["side_x_from_frame_center_px"], errors="coerce")
                    .diff()
                    .divide(dt)
                )
            if "side_x_from_center_camera_corrected_px" in vals.columns:
                vals["side_lateral_velocity_camera_corrected_px_s"] = (
                    pd.to_numeric(
                        vals["side_x_from_center_camera_corrected_px"],
                        errors="coerce",
                    )
                    .diff()
                    .divide(dt)
                )
            vals["side_z_velocity_px_s"] = (
                pd.to_numeric(vals["side_z_lift_px"], errors="coerce").diff().divide(dt)
            )
            if {
                "side_lateral_velocity_raw_px_s",
                "side_z_velocity_px_s",
            }.issubset(vals.columns):
                vals["side_motion_direction_raw_deg"] = np.degrees(
                    np.arctan2(
                        vals["side_z_velocity_px_s"],
                        vals["side_lateral_velocity_raw_px_s"],
                    )
                )
            if {
                "side_lateral_velocity_camera_corrected_px_s",
                "side_z_velocity_px_s",
            }.issubset(vals.columns):
                vals["side_motion_direction_camera_corrected_deg"] = np.degrees(
                    np.arctan2(
                        vals["side_z_velocity_px_s"],
                        vals["side_lateral_velocity_camera_corrected_px_s"],
                    )
                )
            rows.extend(vals.to_dict("records"))
    side_samples = pd.DataFrame(rows)
    if side_samples.empty:
        return {
            "side_samples": side_samples,
            "side_trial_summary": pd.DataFrame(),
            "side_group_time": pd.DataFrame(),
            "side_stiffness_summary": pd.DataFrame(),
            "side_subject_stiffness_summary": pd.DataFrame(),
        }

    base_keys = ["subject_id", "trial_index_raw"]
    if "stiffness_segment_id" in side_samples.columns:
        base_keys.append("stiffness_segment_id")
    side_trial_summary = (
        side_samples.groupby(base_keys, dropna=False)
        .agg(
            subject_group=("subject_group", "first"),
            experiment_group=(EXPERIMENT_GROUP_COLUMN, "first")
            if EXPERIMENT_GROUP_COLUMN in side_samples.columns
            else ("subject_group", "first"),
            finger_condition=("finger_condition", "first"),
            stiffness_value=("stiffness_value", "first"),
            stiffness_order_in_trial=("stiffness_order_in_trial", "first"),
            comparison_value=("comparison_value", "first"),
            standard_value=("standard_value", "first"),
            signed_stiffness_delta=("signed_stiffness_delta", "first"),
            correct_response=("correct_response", "first"),
            side_video_file=("side_video_file", "first"),
            side_camera_side=("side_camera_side", "first"),
            side_camera_view_sign=("side_camera_view_sign", "first"),
            n_side_samples=("frame_index", "count"),
            side_detection_rate=("side_detected", "mean"),
            mean_side_z_lift_px=("side_z_lift_px", "mean"),
            max_side_z_lift_px=("side_z_lift_px", "max"),
            mean_side_x_from_center_raw_px=("side_x_from_frame_center_px", "mean"),
            mean_side_x_from_center_camera_corrected_px=(
                "side_x_from_center_camera_corrected_px",
                "mean",
            ),
            max_abs_side_x_from_center_camera_corrected_px=(
                "side_x_from_center_camera_corrected_px",
                lambda s: float(pd.to_numeric(s, errors="coerce").abs().max()),
            ),
            mean_side_lift_lateral_angle_raw_deg=(
                "side_lift_lateral_angle_raw_deg",
                lambda s: _circ_mean_deg(s),
            ),
            mean_side_lift_lateral_angle_camera_corrected_deg=(
                "side_lift_lateral_angle_camera_corrected_deg",
                lambda s: _circ_mean_deg(s),
            ),
            mean_side_motion_direction_camera_corrected_deg=(
                "side_motion_direction_camera_corrected_deg",
                lambda s: _circ_mean_deg(s),
            ),
            mean_side_mask_area_px=("side_mask_area_px", "mean"),
        )
        .reset_index()
    )
    side_samples = add_success_label_column(
        add_protocol_demographic_factors(add_workspace_normalization_columns(side_samples))
    )
    side_samples["side_time_bin"] = np.minimum(
        (side_samples["side_time_fraction"].fillna(0) * 20).astype(int) + 1, 20
    )
    side_group_cols = [
        "subject_id",
        "subject_group",
        "stiffness_value",
        "finger_condition",
        "side_time_bin",
    ]
    side_group_cols.extend(
        c
        for c in [
            EXPERIMENT_GROUP_COLUMN,
            "workspace_setup",
            "success_label",
            "protocol_factor",
            "sex_factor",
            "age_group",
        ]
        if c in side_samples.columns and c not in side_group_cols
    )
    side_group_time = (
        side_samples.groupby(
            side_group_cols,
            dropna=False,
        )
        .agg(
            n_frames=("frame_index", "count"),
            side_time_fraction=("side_time_fraction", "mean"),
            side_trial_time_fraction=("side_trial_time_fraction", "mean"),
            mean_side_z_lift_px=("side_z_lift_px", "mean"),
            sem_side_z_lift_px=("side_z_lift_px", _sem),
            mean_side_x_from_center_camera_corrected_px=(
                "side_x_from_center_camera_corrected_px",
                "mean",
            ),
            mean_side_lift_lateral_angle_camera_corrected_deg=(
                "side_lift_lateral_angle_camera_corrected_deg",
                lambda s: _circ_mean_deg(s),
            ),
            mean_side_motion_direction_camera_corrected_deg=(
                "side_motion_direction_camera_corrected_deg",
                lambda s: _circ_mean_deg(s),
            ),
            mean_detection_rate=("side_detected", "mean"),
        )
        .reset_index()
    )
    side_subject_stiffness_summary = (
        side_trial_summary.groupby(
            [
                "subject_id",
                "subject_group",
                "side_camera_side",
                "stiffness_value",
                "finger_condition",
            ],
            dropna=False,
        )
        .agg(
            n_trials=("trial_index_raw", "count"),
            mean_side_z_lift_px=("mean_side_z_lift_px", "mean"),
            sem_side_z_lift_px=("mean_side_z_lift_px", _sem),
            max_side_z_lift_px=("max_side_z_lift_px", "mean"),
            mean_side_x_from_center_camera_corrected_px=(
                "mean_side_x_from_center_camera_corrected_px",
                "mean",
            ),
            mean_side_lift_lateral_angle_camera_corrected_deg=(
                "mean_side_lift_lateral_angle_camera_corrected_deg",
                lambda s: _circ_mean_deg(s),
            ),
            mean_side_motion_direction_camera_corrected_deg=(
                "mean_side_motion_direction_camera_corrected_deg",
                lambda s: _circ_mean_deg(s),
            ),
            success_rate=("correct_response", "mean"),
        )
        .reset_index()
    )
    side_stiffness_summary = (
        side_trial_summary.groupby(["stiffness_value"], dropna=False)
        .agg(
            n_trials=("trial_index_raw", "count"),
            n_subjects=("subject_id", "nunique"),
            mean_side_z_lift_px=("mean_side_z_lift_px", "mean"),
            sem_side_z_lift_px=("mean_side_z_lift_px", _sem),
            max_side_z_lift_px=("max_side_z_lift_px", "mean"),
            mean_side_x_from_center_camera_corrected_px=(
                "mean_side_x_from_center_camera_corrected_px",
                "mean",
            ),
            mean_side_lift_lateral_angle_camera_corrected_deg=(
                "mean_side_lift_lateral_angle_camera_corrected_deg",
                lambda s: _circ_mean_deg(s),
            ),
            mean_side_motion_direction_camera_corrected_deg=(
                "mean_side_motion_direction_camera_corrected_deg",
                lambda s: _circ_mean_deg(s),
            ),
            success_rate=("correct_response", "mean"),
        )
        .reset_index()
    )
    return {
        "side_samples": side_samples,
        "side_trial_summary": side_trial_summary,
        "side_group_time": side_group_time,
        "side_stiffness_summary": side_stiffness_summary,
        "side_subject_stiffness_summary": side_subject_stiffness_summary,
    }


def _present_group_cols(df: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [c for c in preferred if c in df.columns]


def summarize_kinematics(
    trial_summary: pd.DataFrame, time_bins: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Summarize kinematics by participant and stiffness.

    ``trial_summary`` is expected to be one row per participant x pair x
    stiffness segment. All returned summaries retain ``stiffness_value`` where
    possible so values such as 40, 60, 85, etc. are never mixed together.
    """
    empty = {
        "direction_success": pd.DataFrame(),
        "group_time": pd.DataFrame(),
        "stiffness_time": pd.DataFrame(),
        "distance_success": pd.DataFrame(),
        "subject_summary": pd.DataFrame(),
        "participant_stiffness_summary": pd.DataFrame(),
        "stiffness_summary": pd.DataFrame(),
    }
    if trial_summary.empty:
        return empty
    ts = trial_summary.copy()
    for optional_metric in [
        "mean_curvature_1_px",
        "speed_curvature_power_law_slope",
        "speed_curvature_power_law_r2",
    ]:
        if optional_metric not in ts.columns:
            ts[optional_metric] = np.nan
    if "stiffness_value" not in ts.columns:
        ts["stiffness_value"] = pd.to_numeric(
            ts.get(
                "mean_skin_stretch_gain_mm_per_m", ts.get("comparison_value", np.nan)
            ),
            errors="coerce",
        )

    for col in ["mean_jerk_px_s3", "max_jerk_px_s3", "normalized_jerk_cost"]:
        if col not in ts.columns:
            ts[col] = np.nan

    ts = add_success_label_column(
        add_protocol_demographic_factors(add_workspace_normalization_columns(ts))
    )
    if "mean_x_workspace_cm" not in ts.columns and {
        "mean_x_centered_px",
        "workspace_width_cm",
    }.issubset(ts.columns):
        ts["mean_x_workspace_cm"] = (
            pd.to_numeric(ts["mean_x_centered_px"], errors="coerce")
            / CENTER_X
            * (ts["workspace_width_cm"] / 2.0)
        )
    if "mean_y_workspace_cm" not in ts.columns and {
        "mean_y_centered_px",
        "workspace_height_cm",
    }.issubset(ts.columns):
        ts["mean_y_workspace_cm"] = (
            pd.to_numeric(ts["mean_y_centered_px"], errors="coerce")
            / CENTER_Y
            * (ts["workspace_height_cm"] / 2.0)
        )
    if "mean_r_workspace_cm" not in ts.columns and {
        "mean_x_workspace_cm",
        "mean_y_workspace_cm",
    }.issubset(ts.columns):
        ts["mean_r_workspace_cm"] = np.hypot(
            ts["mean_x_workspace_cm"], ts["mean_y_workspace_cm"]
        )
    if "mean_r_workspace_normalized" not in ts.columns and {
        "mean_r_workspace_cm",
        "workspace_width_cm",
        "workspace_height_cm",
    }.issubset(ts.columns):
        half_diag = np.hypot(ts["workspace_width_cm"] / 2.0, ts["workspace_height_cm"] / 2.0)
        ts["mean_r_workspace_normalized"] = ts["mean_r_workspace_cm"] / half_diag
    for optional_col in ["mean_thumb_active_span_px", "mean_hand_orientation_xy_deg"]:
        if optional_col not in ts.columns:
            ts[optional_col] = np.nan
    direction_cols = _present_group_cols(
        ts,
        [
            "subject_id",
            "subject_group",
            EXPERIMENT_GROUP_COLUMN,
            "workspace_setup",
            "side_camera_side",
            "participant_position_context",
            "stiffness_value",
            "finger_condition",
            "dominant_movement_direction",
        ],
    )
    direction_success = (
        ts.groupby(direction_cols, dropna=False)
        .agg(
            n_trials=("trial_index_raw", "count"),
            success_rate=("correct_response", "mean"),
            mean_x_centered_px=("mean_x_centered_px", "mean"),
            mean_y_centered_px=("mean_y_centered_px", "mean"),
            mean_speed_px_s=("mean_speed_px_s", "mean"),
            mean_r_center_px=("mean_r_center_px", "mean"),
            mean_path_length_px=("path_length_px", "mean"),
        )
        .reset_index()
    )

    def _quartile(s: pd.Series) -> pd.Series:
        if s.notna().sum() <= 1:
            return pd.Series(np.ones(len(s)), index=s.index)
        return (
            pd.qcut(
                s.rank(method="first"),
                q=min(4, s.notna().sum()),
                labels=False,
                duplicates="drop",
            )
            + 1
        )

    quantile_groups = _present_group_cols(ts, ["subject_id", "stiffness_value"])
    ts["distance_quantile"] = (
        ts.groupby(quantile_groups)["max_r_center_px"].transform(_quartile)
        if quantile_groups
        else _quartile(ts["max_r_center_px"])
    )
    distance_cols = _present_group_cols(
        ts,
        [
            "subject_id",
            "subject_group",
            EXPERIMENT_GROUP_COLUMN,
            "workspace_setup",
            "side_camera_side",
            "participant_position_context",
            "stiffness_value",
            "finger_condition",
            "distance_quantile",
        ],
    )
    distance_success = (
        ts.groupby(distance_cols, dropna=False)
        .agg(
            n_trials=("trial_index_raw", "count"),
            mean_max_r_center_px=("max_r_center_px", "mean"),
            success_rate=("correct_response", "mean"),
            mean_speed_px_s=("mean_speed_px_s", "mean"),
        )
        .reset_index()
    )

    subject_cols = _present_group_cols(
        ts,
        [
            "subject_id",
            "subject_group",
            EXPERIMENT_GROUP_COLUMN,
            "workspace_setup",
            "side_camera_side",
            "participant_position_context",
            "stiffness_value",
            "finger_condition",
        ],
    )
    subject_summary = (
        ts.groupby(subject_cols, dropna=False)
        .agg(
            n_trials=("trial_index_raw", "count"),
            success_rate=("correct_response", "mean"),
            mean_x_centered_px=("mean_x_centered_px", "mean"),
            mean_y_centered_px=("mean_y_centered_px", "mean"),
            mean_x_workspace_cm=("mean_x_workspace_cm", "mean"),
            mean_y_workspace_cm=("mean_y_workspace_cm", "mean"),
            mean_r_workspace_cm=("mean_r_workspace_cm", "mean"),
            mean_r_workspace_normalized=("mean_r_workspace_normalized", "mean"),
            mean_max_r_center_px=("max_r_center_px", "mean"),
            mean_thumb_active_span_px=("mean_thumb_active_span_px", "mean"),
            mean_hand_orientation_xy_deg=(
                "mean_hand_orientation_xy_deg",
                lambda s: _circ_mean_deg(s),
            ),
            mean_vx_px_s=("mean_vx_px_s", "mean"),
            mean_vy_px_s=("mean_vy_px_s", "mean"),
            mean_speed_px_s=("mean_speed_px_s", "mean"),
            mean_ax_px_s2=("mean_ax_px_s2", "mean"),
            mean_ay_px_s2=("mean_ay_px_s2", "mean"),
            mean_acceleration_px_s2=("mean_acceleration_px_s2", "mean"),
            mean_jerk_px_s3=("mean_jerk_px_s3", "mean"),
            mean_normalized_jerk_cost=("normalized_jerk_cost", "mean"),
            mean_curvature_1_px=("mean_curvature_1_px", "mean"),
            speed_curvature_power_law_slope=("speed_curvature_power_law_slope", "mean"),
            speed_curvature_power_law_r2=("speed_curvature_power_law_r2", "mean"),
            mean_path_length_px=("path_length_px", "mean"),
            mean_straightness_index=("straightness_index", "mean"),
            circular_mean_direction_deg=(
                "dominant_movement_angle_deg",
                lambda s: _circ_mean_deg(s),
            ),
            movement_direction_resultant_length=(
                "dominant_movement_angle_deg",
                lambda s: _resultant_length(s),
            ),
        )
        .reset_index()
    )
    participant_stiffness_summary = subject_summary.copy()

    stiffness_summary = (
        ts.groupby(["stiffness_value"], dropna=False)
        .agg(
            n_trials=("trial_index_raw", "count"),
            n_subjects=("subject_id", "nunique"),
            success_rate=("correct_response", "mean"),
            mean_x_centered_px=("mean_x_centered_px", "mean"),
            mean_y_centered_px=("mean_y_centered_px", "mean"),
            sem_x_centered_px=("mean_x_centered_px", _sem),
            sem_y_centered_px=("mean_y_centered_px", _sem),
            mean_vx_px_s=("mean_vx_px_s", "mean"),
            mean_vy_px_s=("mean_vy_px_s", "mean"),
            sem_vx_px_s=("mean_vx_px_s", _sem),
            sem_vy_px_s=("mean_vy_px_s", _sem),
            mean_speed_px_s=("mean_speed_px_s", "mean"),
            sem_speed_px_s=("mean_speed_px_s", _sem),
            mean_ax_px_s2=("mean_ax_px_s2", "mean"),
            mean_ay_px_s2=("mean_ay_px_s2", "mean"),
            sem_ax_px_s2=("mean_ax_px_s2", _sem),
            sem_ay_px_s2=("mean_ay_px_s2", _sem),
            mean_acceleration_px_s2=("mean_acceleration_px_s2", "mean"),
            sem_acceleration_px_s2=("mean_acceleration_px_s2", _sem),
            mean_jerk_px_s3=("mean_jerk_px_s3", "mean"),
            sem_jerk_px_s3=("mean_jerk_px_s3", _sem),
            mean_normalized_jerk_cost=("normalized_jerk_cost", "mean"),
            sem_normalized_jerk_cost=("normalized_jerk_cost", _sem),
            mean_path_length_px=("path_length_px", "mean"),
            circular_mean_direction_deg=(
                "dominant_movement_angle_deg",
                lambda s: _circ_mean_deg(s),
            ),
        )
        .reset_index()
    )

    if not time_bins.empty:
        tb = add_success_label_column(
            add_protocol_demographic_factors(add_workspace_normalization_columns(time_bins))
        )
        if "stiffness_value" not in tb.columns:
            tb["stiffness_value"] = pd.to_numeric(
                tb.get("comparison_value", np.nan), errors="coerce"
            )
        group_cols = _present_group_cols(
            tb,
            [
                "subject_id",
                "subject_group",
                EXPERIMENT_GROUP_COLUMN,
                "workspace_setup",
                "side_camera_side",
                "participant_position_context",
                "success_label",
                "stiffness_value",
                "finger_condition",
                "trajectory_time_bin",
            ],
        )
        group_time = (
            tb.groupby(group_cols, dropna=False)
            .agg(
                n_trials=("tracking_file", "nunique"),
                time_fraction=("time_fraction", "mean"),
                stiffness_time_fraction=("stiffness_time_fraction", "mean"),
                trial_time_fraction=("trial_time_fraction", "mean"),
                mean_x_centered_px=("x_centered_px", "mean"),
                mean_y_centered_px=("y_centered_px", "mean"),
                mean_x_workspace_cm=("x_workspace_cm", "mean"),
                mean_y_workspace_cm=("y_workspace_cm", "mean"),
                mean_r_workspace_normalized=("r_workspace_normalized", "mean"),
                sem_x_centered_px=("x_centered_px", _sem),
                sem_y_centered_px=("y_centered_px", _sem),
                mean_r_center_px=("r_center_px", "mean"),
                sem_r_center_px=("r_center_px", _sem),
                mean_vx_px_s=("vx_px_s", "mean"),
                mean_vy_px_s=("vy_px_s", "mean"),
                sem_vx_px_s=("vx_px_s", _sem),
                sem_vy_px_s=("vy_px_s", _sem),
                mean_speed_px_s=("speed_px_s", "mean"),
                sem_speed_px_s=("speed_px_s", _sem),
                mean_ax_px_s2=("ax_px_s2", "mean"),
                mean_ay_px_s2=("ay_px_s2", "mean"),
                sem_ax_px_s2=("ax_px_s2", _sem),
                sem_ay_px_s2=("ay_px_s2", _sem),
                mean_acceleration_px_s2=("acceleration_px_s2", "mean"),
                sem_acceleration_px_s2=("acceleration_px_s2", _sem),
                mean_jerk_px_s3=("jerk_px_s3", "mean"),
                sem_jerk_px_s3=("jerk_px_s3", _sem),
                mean_radial_velocity_px_s=("radial_velocity_px_s", "mean"),
                mean_tangential_velocity_px_s=("tangential_velocity_px_s", "mean"),
            )
            .reset_index()
        )
        stiffness_time = (
            tb.groupby(["stiffness_value", "trajectory_time_bin"], dropna=False)
            .agg(
                n_trials=("tracking_file", "nunique"),
                n_subjects=("subject_id", "nunique"),
                time_fraction=("time_fraction", "mean"),
                mean_x_centered_px=("x_centered_px", "mean"),
                mean_y_centered_px=("y_centered_px", "mean"),
                mean_vx_px_s=("vx_px_s", "mean"),
                mean_vy_px_s=("vy_px_s", "mean"),
                mean_speed_px_s=("speed_px_s", "mean"),
                mean_ax_px_s2=("ax_px_s2", "mean"),
                mean_ay_px_s2=("ay_px_s2", "mean"),
                mean_acceleration_px_s2=("acceleration_px_s2", "mean"),
                mean_jerk_px_s3=("jerk_px_s3", "mean"),
            )
            .reset_index()
        )
    else:
        group_time = pd.DataFrame()
        stiffness_time = pd.DataFrame()
    group_comparisons = compute_experiment_group_comparisons(subject_summary)
    return {
        "direction_success": direction_success,
        "group_time": group_time,
        "stiffness_time": stiffness_time,
        "distance_success": distance_success,
        "subject_summary": subject_summary,
        "participant_stiffness_summary": participant_stiffness_summary,
        "stiffness_summary": stiffness_summary,
        **group_comparisons,
    }


# ---------------------------------------------------------------------------
# Experiment-group comparison section (N_E, L_E, L_P)

KINEMATIC_GROUP_METRICS = [
    "success_rate",
    "mean_x_centered_px",
    "mean_y_centered_px",
    "mean_x_workspace_cm",
    "mean_y_workspace_cm",
    "mean_r_workspace_cm",
    "mean_r_workspace_normalized",
    "mean_max_r_center_px",
    "mean_thumb_active_span_px",
    "mean_hand_orientation_xy_deg",
    "mean_vx_px_s",
    "mean_vy_px_s",
    "mean_speed_px_s",
    "mean_ax_px_s2",
    "mean_ay_px_s2",
    "mean_acceleration_px_s2",
    "mean_jerk_px_s3",
    "mean_normalized_jerk_cost",
    "mean_curvature_1_px",
    "speed_curvature_power_law_slope",
    "speed_curvature_power_law_r2",
    "mean_path_length_px",
    "mean_straightness_index",
    "movement_direction_resultant_length",
]

KINEMATIC_INTERACTION_SCOPES = [
    {
        "interaction_scope": "all",
        "interaction_code": "all",
        "folder_name": "all",
        "description": "All stiffness interactions already used by the legacy kinematic analysis.",
    },
    {
        "interaction_scope": "standard",
        "interaction_code": "S",
        "folder_name": "standard",
        "description": "Only samples/segments where the active stiffness equals the standard value.",
    },
    {
        "interaction_scope": "comparison",
        "interaction_code": "C",
        "folder_name": "Comparison",
        "description": "Only samples/segments where the active stiffness equals the comparison value.",
    },
]


def _kinematic_condition_columns(df: pd.DataFrame) -> list[str]:
    """Return optional kinematic condition columns that contain usable labels."""
    candidates = [
        "finger_condition",
        "stiffness_value",
        "success_label",
        "side_camera_side",
        "protocol_factor",
        "sex_factor",
        "age_group",
    ]
    return [c for c in candidates if c in df.columns and df[c].notna().any()]


def compute_experiment_group_comparisons(
    subject_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compare kinematic outcomes by exact groups plus all/subgroup/participant scopes."""
    subject_summary = add_success_label_column(
        add_protocol_demographic_factors(add_workspace_normalization_columns(subject_summary))
    )
    metrics = _available_metric_columns(subject_summary, KINEMATIC_GROUP_METRICS)
    condition_cols = _kinematic_condition_columns(subject_summary)
    exact_tables = {
        f"kinematic_{name}": table
        for name, table in compute_group_comparison_tables(
            subject_summary,
            metric_columns=metrics,
            condition_cols=condition_cols,
        ).items()
    }
    scope_tables = {
        f"kinematic_{name}": table
        for name, table in compute_analysis_scope_tables(
            subject_summary,
            metric_columns=metrics,
            condition_cols=condition_cols,
        ).items()
    }
    setup_tables = {
        f"kinematic_{name}": table
        for name, table in compute_setup_factor_tables(
            subject_summary,
            metric_columns=metrics,
            condition_cols=condition_cols,
        ).items()
    }
    expanded_tables = {
        f"kinematic_{name}": table
        for name, table in compute_expanded_kinematic_scope_tables(
            subject_summary,
            metric_columns=metrics,
        ).items()
    }
    return {**exact_tables, **scope_tables, **setup_tables, **expanded_tables}


def compute_expanded_kinematic_scope_tables(
    df: pd.DataFrame, metric_columns: Optional[list[str]] = None
) -> dict[str, pd.DataFrame]:
    """Summarize requested comparison scopes for any kinematic table.

    This covers all requested grouping factors while keeping absent factors
    explicit in a status table instead of manufacturing missing sex/age/protocol
    metadata.
    """

    empty = {
        "expanded_scope_metric_summary": pd.DataFrame(),
        "expanded_scope_pairwise_mean_differences": pd.DataFrame(),
        "expanded_scope_status": pd.DataFrame(),
    }
    if df.empty:
        return empty
    d = add_success_label_column(
        add_protocol_demographic_factors(add_workspace_normalization_columns(df))
    )
    metrics = _available_metric_columns(d, metric_columns or KINEMATIC_GROUP_METRICS)
    if not metrics:
        return empty
    requested_scopes = [
        ("all", None, "all_participants"),
        ("protocol", "protocol_factor", "protocol 1/2/3/4 if present/inferred"),
        ("E_vs_P", "subject_group", "E vs P"),
        ("N_vs_L_workspace", "workspace_setup", "N=40x50 cm vs L=60x60 cm"),
        ("side_camera", "side_camera_side", "right-side L vs left-side N camera"),
        ("experiment_group", EXPERIMENT_GROUP_COLUMN, "N_E/L_E/L_P exact group"),
        ("sex", "sex_factor", "male vs female if present"),
        ("age", "age_group", "age bins if present"),
        ("stiffness", "stiffness_value", "skin-stretch/stiffness value"),
        ("finger", "finger_condition", "active finger"),
        ("success", "success_label", "answers.csv-derived success vs failure"),
    ]
    summary_frames: list[pd.DataFrame] = []
    pair_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for scope, col, note in requested_scopes:
        if col is None:
            temp = d.assign(_scope_value="all_participants")
            group_col = "_scope_value"
            status_rows.append(
                {
                    "comparison_scope": scope,
                    "status": "available",
                    "column": "all",
                    "note": note,
                    "n_levels": 1,
                }
            )
        elif col not in d.columns or not d[col].notna().any():
            status_rows.append(
                {
                    "comparison_scope": scope,
                    "status": "missing",
                    "column": col,
                    "note": note,
                    "n_levels": 0,
                }
            )
            continue
        else:
            temp = d.copy()
            temp["_scope_value"] = temp[col].astype(str)
            group_col = "_scope_value"
            status_rows.append(
                {
                    "comparison_scope": scope,
                    "status": "available",
                    "column": col,
                    "note": note,
                    "n_levels": int(temp[group_col].dropna().nunique()),
                }
            )
        for metric in metrics:
            rows_for_summary: list[dict[str, Any]] = []
            for value, group in temp.groupby(group_col, dropna=False):
                vals = pd.to_numeric(group[metric], errors="coerce")
                ci_low, ci_high = _ci95_mean(vals)
                rows_for_summary.append(
                    {
                        "comparison_scope": scope,
                        "metric": metric,
                        "comparison_value": value,
                        "n_observations": int(vals.notna().sum()),
                        "n_subjects": int(group["subject_id"].nunique())
                        if "subject_id" in group.columns
                        else np.nan,
                        "mean": float(vals.mean()),
                        "median": float(vals.median()),
                        "sem": _sem(vals),
                        "ci95_mean_low": ci_low,
                        "ci95_mean_high": ci_high,
                    }
                )
            summary = pd.DataFrame(rows_for_summary)
            summary_frames.append(summary)
            means = summary[["comparison_value", "mean", "n_observations"]].dropna(
                subset=["mean"]
            )
            levels = means["comparison_value"].tolist()
            for i, level_a in enumerate(levels):
                for level_b in levels[i + 1 :]:
                    row_a = means[means["comparison_value"] == level_a].iloc[0]
                    row_b = means[means["comparison_value"] == level_b].iloc[0]
                    pair_rows.append(
                        {
                            "comparison_scope": scope,
                            "metric": metric,
                            "level_a": level_a,
                            "level_b": level_b,
                            "comparison": f"{level_b} - {level_a}",
                            "mean_a": float(row_a["mean"]),
                            "mean_b": float(row_b["mean"]),
                            "mean_difference_b_minus_a": float(row_b["mean"] - row_a["mean"]),
                            "n_a": int(row_a["n_observations"]),
                            "n_b": int(row_b["n_observations"]),
                        }
                    )
    return {
        "expanded_scope_metric_summary": pd.concat(
            summary_frames, ignore_index=True, sort=False
        )
        if summary_frames
        else pd.DataFrame(),
        "expanded_scope_pairwise_mean_differences": pd.DataFrame(pair_rows),
        "expanded_scope_status": pd.DataFrame(status_rows),
    }


def save_experiment_group_comparison_outputs(
    output_root: Path, subject_summary: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    tables = compute_experiment_group_comparisons(subject_summary)
    for name, df in tables.items():
        save_csv(df, output_root, f"{name}.csv")
    figure_manifest = save_scope_summary_plots(
        tables,
        output_root,
        namespace="kinematic",
        metrics=KINEMATIC_GROUP_METRICS,
    )
    tables["kinematic_scope_figure_manifest"] = figure_manifest
    return tables


def _interaction_value_column(df: pd.DataFrame) -> str | None:
    for candidate in [
        "stiffness_value",
        "skin_stretch_gain_mm_per_m",
        "mean_skin_stretch_gain_mm_per_m",
    ]:
        if candidate in df.columns:
            return candidate
    return None


def _numeric_close_mask(
    value: pd.Series, target: pd.Series, *, atol: float = 1e-9
) -> pd.Series:
    left = pd.to_numeric(value, errors="coerce")
    right = pd.to_numeric(target, errors="coerce")
    return (left.notna() & right.notna() & np.isclose(left, right, atol=atol)).fillna(
        False
    )


def kinematic_interaction_mask(
    df: pd.DataFrame, interaction_scope: str, *, atol: float = 1e-9
) -> pd.Series:
    """Return rows belonging to all, standard (S), or comparison (C) interactions.

    The kinematic tables are segment/sample-level after tracking is processed.
    ``standard`` keeps only rows where the active ``stiffness_value`` equals
    ``standard_value``; ``comparison`` keeps rows where active stiffness equals
    ``comparison_value``.  ``all`` preserves the previous analysis behavior.
    """

    scope = str(interaction_scope).strip().lower()
    if scope in {"all", "*", "any"}:
        return pd.Series(True, index=df.index)

    value_col = _interaction_value_column(df)
    target_col = (
        "standard_value"
        if scope in {"standard", "standart", "s"}
        else "comparison_value"
        if scope in {"comparison", "comperison", "c"}
        else None
    )
    if value_col is None or target_col is None or target_col not in df.columns:
        return pd.Series(False, index=df.index)
    return _numeric_close_mask(df[value_col], df[target_col], atol=atol)


def filter_kinematic_interaction(
    df: pd.DataFrame, interaction_scope: str, *, atol: float = 1e-9
) -> pd.DataFrame:
    """Filter any kinematic table to all/S/C interaction rows."""

    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    return df.loc[kinematic_interaction_mask(df, interaction_scope, atol=atol)].copy()


def _save_named_tables(output_root: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        save_csv(df, output_root, f"{name}.csv")


def _save_interaction_summary_block(
    output_root: Path,
    *,
    trial_kinematic_summary: pd.DataFrame,
    trajectory_time_bins: pd.DataFrame,
    side_z_trial_summary: Optional[pd.DataFrame] = None,
    fig_dpi: int = 160,
    save_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    summaries = summarize_kinematics(trial_kinematic_summary, trajectory_time_bins)
    output_tables = {
        "direction_success_summary": summaries.get("direction_success", pd.DataFrame()),
        "distance_success_summary": summaries.get("distance_success", pd.DataFrame()),
        "subject_kinematic_summary": summaries.get("subject_summary", pd.DataFrame()),
        "participant_stiffness_kinematic_summary": summaries.get(
            "participant_stiffness_summary", pd.DataFrame()
        ),
        "stiffness_kinematic_summary": summaries.get("stiffness_summary", pd.DataFrame()),
        "group_time_summary": summaries.get("group_time", pd.DataFrame()),
        "stiffness_time_summary": summaries.get("stiffness_time", pd.DataFrame()),
    }
    for name, df in summaries.items():
        if name.startswith("kinematic_"):
            output_tables[name] = df
    _save_named_tables(output_root, output_tables)

    subject_summary = summaries.get("subject_summary", pd.DataFrame())
    motor_control = compute_motor_control_comparisons(subject_summary)
    motor_tables = {
        "kinematic_within_subject": motor_control.get("within_subject", pd.DataFrame()),
        "finger_metric_summary": motor_control.get("finger_metric_summary", pd.DataFrame()),
        "within_finger_stiffness_effects": motor_control.get(
            "within_finger_stiffness_effects", pd.DataFrame()
        ),
        "within_finger_stiffness_effect_summary": motor_control.get(
            "within_finger_stiffness_effect_summary", pd.DataFrame()
        ),
        "finger_comparison_paired": motor_control.get(
            "finger_comparison_paired", pd.DataFrame()
        ),
        "finger_comparison_by_stiffness_paired": motor_control.get(
            "finger_comparison_by_stiffness_paired", pd.DataFrame()
        ),
    }
    _save_named_tables(output_root, motor_tables)

    side_trials = (
        side_z_trial_summary
        if side_z_trial_summary is not None
        else pd.DataFrame()
    )
    success_kinematic_z = compute_success_kinematic_z_analysis(
        trial_kinematic_summary, side_trials
    )
    trajectory_structure = compute_trajectory_similarity_analysis(trajectory_time_bins)
    advanced_tables = {
        "trial_success_kinematic_z_table": success_kinematic_z.get(
            "trial_success_table", pd.DataFrame()
        ),
        "success_kinematic_z_contrast_by_subject_finger": success_kinematic_z.get(
            "success_contrast_by_subject_finger", pd.DataFrame()
        ),
        "success_kinematic_z_contrast_summary": success_kinematic_z.get(
            "success_contrast_summary", pd.DataFrame()
        ),
        "success_kinematic_z_contrast_by_finger_summary": success_kinematic_z.get(
            "success_contrast_by_finger_summary", pd.DataFrame()
        ),
        "subject_finger_trajectory": trajectory_structure.get(
            "subject_finger_trajectory", pd.DataFrame()
        ),
        "trajectory_variability_summary": trajectory_structure.get(
            "trajectory_variability_summary", pd.DataFrame()
        ),
        "finger_trajectory_distance_paired": trajectory_structure.get(
            "finger_trajectory_distance_paired", pd.DataFrame()
        ),
        "finger_trajectory_distance_summary": trajectory_structure.get(
            "finger_trajectory_distance_summary", pd.DataFrame()
        ),
        "success_failure_trajectory_distance": trajectory_structure.get(
            "success_failure_trajectory_distance", pd.DataFrame()
        ),
        "success_failure_trajectory_distance_summary": trajectory_structure.get(
            "success_failure_trajectory_distance_summary", pd.DataFrame()
        ),
    }
    _save_named_tables(output_root, advanced_tables)

    subject_spatial = compute_subject_spatial_trajectory_analysis(trajectory_time_bins)
    spatial_tables = {
        "subject_xy_trajectory": subject_spatial.get("subject_xy_trajectory", pd.DataFrame()),
        "subject_spatial_trajectory_summary": subject_spatial.get(
            "subject_spatial_trajectory_summary", pd.DataFrame()
        ),
        "subject_finger_spatial_distance": subject_spatial.get(
            "subject_finger_spatial_distance", pd.DataFrame()
        ),
        "subject_spatial_metric_distribution": subject_spatial.get(
            "subject_spatial_metric_distribution", pd.DataFrame()
        ),
    }
    _save_named_tables(output_root, spatial_tables)

    subject_va = compute_subject_velocity_acceleration_analysis(trajectory_time_bins)
    va_tables = {
        "subject_velocity_acceleration_profile": subject_va.get(
            "subject_velocity_acceleration_profile", pd.DataFrame()
        ),
        "subject_velocity_acceleration_summary": subject_va.get(
            "subject_velocity_acceleration_summary", pd.DataFrame()
        ),
        "subject_finger_velocity_acceleration_distance": subject_va.get(
            "subject_finger_velocity_acceleration_distance", pd.DataFrame()
        ),
        "subject_velocity_acceleration_metric_distribution": subject_va.get(
            "subject_velocity_acceleration_metric_distribution", pd.DataFrame()
        ),
        "velocity_stiffness_influence_summary": subject_va.get(
            "velocity_stiffness_influence_summary", pd.DataFrame()
        ),
        "velocity_finger_influence_summary": subject_va.get(
            "velocity_finger_influence_summary", pd.DataFrame()
        ),
        "velocity_time_influence_summary": subject_va.get(
            "velocity_time_influence_summary", pd.DataFrame()
        ),
    }
    _save_named_tables(output_root, va_tables)

    if not side_trials.empty:
        save_csv(side_trials, output_root, "side_z_trial_summary.csv")
    hand_orientation = compute_hand_orientation_plane_analysis(
        trial_kinematic_summary, side_trials
    )
    hand_tables = {
        "hand_orientation_plane_trials": hand_orientation.get(
            "hand_orientation_plane_trials", pd.DataFrame()
        ),
        "hand_orientation_plane_summary": hand_orientation.get(
            "hand_orientation_plane_summary", pd.DataFrame()
        ),
    }
    _save_named_tables(output_root, hand_tables)

    if save_figures:
        save_motor_control_figures(output_root, motor_control, fig_dpi=fig_dpi)
        save_advanced_kinematic_figures(
            output_root, success_kinematic_z, trajectory_structure, fig_dpi=fig_dpi
        )
        save_subject_xy_trajectory_figures(
            output_root, spatial_tables["subject_xy_trajectory"], fig_dpi=fig_dpi
        )
        save_subject_velocity_acceleration_figures(
            output_root,
            va_tables["subject_velocity_acceleration_profile"],
            fig_dpi=fig_dpi,
        )
        save_hand_orientation_plane_figures(
            output_root,
            hand_tables["hand_orientation_plane_summary"],
            hand_tables["hand_orientation_plane_trials"],
            fig_dpi=fig_dpi,
        )
        save_hand_orientation_axis_matrix_figures(
            output_root, hand_tables["hand_orientation_plane_trials"], fig_dpi=fig_dpi
        )
        save_movement_cycle_hand_angle_figures(
            output_root, trajectory_time_bins, fig_dpi=fig_dpi
        )
        save_kinematic_figures(
            output_root,
            trial_kinematic_summary,
            output_tables["group_time_summary"],
            output_tables["direction_success_summary"],
            output_tables["distance_success_summary"],
            fig_dpi=fig_dpi,
        )

    return {
        **output_tables,
        **motor_tables,
        **advanced_tables,
        **spatial_tables,
        **va_tables,
        **hand_tables,
    }


def save_interaction_filtered_kinematic_outputs(
    output_root: Path,
    *,
    trial_kinematic_summary: pd.DataFrame,
    trajectory_time_bins: pd.DataFrame,
    pair_kinematic_summary: Optional[pd.DataFrame] = None,
    kinematic_samples: Optional[pd.DataFrame] = None,
    side_z_trial_summary: Optional[pd.DataFrame] = None,
    side_z_samples: Optional[pd.DataFrame] = None,
    fig_dpi: int = 160,
    save_figures: bool = True,
    save_sample_tables: bool = False,
) -> pd.DataFrame:
    """Write mirrored all/S/C kinematic analysis folders.

    The existing flat ``results`` directory remains compatible.  This helper
    adds a branch architecture under ``results``:

    - ``all``: the current/legacy analysis over every stiffness segment.
    - ``standard`` (code ``S``): only standard-value interaction segments.
    - ``Comparison`` (code ``C``): only comparison-value interaction segments.

    Each branch recalculates subject, group, scope, motor-control, trajectory,
    velocity/acceleration, hand-orientation, and success-linked tables from the
    filtered rows so group and individual outputs stay aligned with the selected
    interaction.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    save_experiment_setup_context(output_root)
    index_rows: list[dict[str, Any]] = []
    for spec in KINEMATIC_INTERACTION_SCOPES:
        scope = spec["interaction_scope"]
        folder = output_root / spec["folder_name"]
        folder.mkdir(parents=True, exist_ok=True)
        save_experiment_setup_context(folder)

        trial_subset = filter_kinematic_interaction(trial_kinematic_summary, scope)
        time_subset = filter_kinematic_interaction(trajectory_time_bins, scope)
        side_trial_subset = (
            filter_kinematic_interaction(side_z_trial_summary, scope)
            if side_z_trial_summary is not None
            else pd.DataFrame()
        )
        save_csv(trial_subset, folder, "trial_kinematic_summary.csv")
        save_csv(time_subset, folder, "trajectory_time_bins.csv")
        if pair_kinematic_summary is not None:
            pair_subset = (
                pair_kinematic_summary.copy()
                if scope == "all"
                else pair_kinematic_summary[
                    pair_kinematic_summary.get("trial_index_raw").isin(
                        trial_subset.get("trial_index_raw", pd.Series(dtype=float))
                    )
                ].copy()
                if "trial_index_raw" in pair_kinematic_summary.columns
                else pair_kinematic_summary.copy()
            )
            save_csv(pair_subset, folder, "pair_kinematic_summary.csv")
        if save_sample_tables and kinematic_samples is not None:
            save_csv(
                filter_kinematic_interaction(kinematic_samples, scope),
                folder,
                "kinematic_samples.csv",
            )
        if save_sample_tables and side_z_samples is not None:
            save_csv(
                filter_kinematic_interaction(side_z_samples, scope),
                folder,
                "side_z_samples.csv",
            )

        _save_interaction_summary_block(
            folder,
            trial_kinematic_summary=trial_subset,
            trajectory_time_bins=time_subset,
            side_z_trial_summary=side_trial_subset,
            fig_dpi=fig_dpi,
            save_figures=save_figures,
        )
        manifest = analysis_manifest(folder)
        save_csv(manifest, folder, "analysis_manifest.csv")
        index_rows.append(
            {
                **spec,
                "path": str(folder),
                "n_trial_segments": int(len(trial_subset)),
                "n_time_bins": int(len(time_subset)),
                "n_subjects": int(trial_subset["subject_id"].nunique())
                if "subject_id" in trial_subset.columns
                else 0,
                "manifest_exists": bool((folder / "analysis_manifest.csv").exists()),
            }
        )

    index = pd.DataFrame(index_rows)
    save_csv(index, output_root, "interaction_analysis_folder_index.csv")
    return index


def _available_metric_columns(
    df: pd.DataFrame, metric_columns: Optional[list[str]] = None
) -> list[str]:
    requested = metric_columns or MOTOR_CONTROL_METRICS
    return [
        c
        for c in requested
        if c in df.columns
        and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
    ]


def _paired_sign_flip_p(
    values: pd.Series,
    *,
    max_exact_n: int = 14,
    n_permutations: int = 20000,
    seed: int = 20260513,
) -> float:
    """Two-sided paired sign-flip p-value for within-subject contrasts.

    Uses exact enumeration for small N and a deterministic Monte-Carlo
    approximation for larger N. This keeps the analysis dependency-light while
    respecting the repeated-measures design.
    """
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    observed = abs(float(np.mean(x)))
    if observed == 0:
        return 1.0
    if x.size <= max_exact_n:
        n_states = 2**x.size
        count = 0
        for mask in range(n_states):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(x.size)])
            count += abs(float(np.mean(x * signs))) >= observed - 1e-12
        return float(count / n_states)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, x.size))
    null = np.abs((signs * x).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (n_permutations + 1))


def _paired_effect_summary(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(x))
    mean = float(x.mean()) if n else np.nan
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    return {
        "n_paired_observations": n,
        "mean_difference": mean,
        "median_difference": float(x.median()) if n else np.nan,
        "sem_difference": sd / math.sqrt(n) if n > 1 and np.isfinite(sd) else np.nan,
        "cohens_dz": mean / sd if n > 1 and np.isfinite(sd) and sd > 0 else np.nan,
        "sign_flip_p": _paired_sign_flip_p(x),
    }


def _ols_slope(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2 or np.unique(xx).size < 2:
        return np.nan
    return float(np.polyfit(xx, yy, 1)[0])


def compute_motor_control_comparisons(
    subject_summary: pd.DataFrame,
    *,
    metric_columns: Optional[list[str]] = None,
    subject_col: str = "subject_id",
    finger_col: str = "finger_condition",
    stiffness_col: str = "stiffness_value",
) -> dict[str, pd.DataFrame]:
    """Create repeated-measures motor-control summaries.

    Outputs are designed for subject-preserving kinematic interpretation:
    1. ``within_subject`` keeps one row per subject x finger x stiffness and
       adds subject-centered/z-scored metrics.
    2. ``within_finger_stiffness_effects`` estimates stiffness sensitivity
       separately inside each subject x finger.
    3. ``finger_comparison_paired`` compares fingers using paired
       within-subject contrasts rather than trial-pooled means.
    """
    empty = {
        "within_subject": pd.DataFrame(),
        "finger_metric_summary": pd.DataFrame(),
        "within_finger_stiffness_effects": pd.DataFrame(),
        "within_finger_stiffness_effect_summary": pd.DataFrame(),
        "finger_comparison_paired": pd.DataFrame(),
        "finger_comparison_by_stiffness_paired": pd.DataFrame(),
    }
    if subject_summary.empty or not {subject_col, finger_col}.issubset(
        subject_summary.columns
    ):
        return empty

    df = subject_summary.copy()
    df[finger_col] = df[finger_col].map(normalize_finger_condition)
    if stiffness_col in df.columns:
        df[stiffness_col] = pd.to_numeric(df[stiffness_col], errors="coerce")
    metrics = _available_metric_columns(df, metric_columns)
    if not metrics:
        return {**empty, "within_subject": df}

    keep_cols = [subject_col, finger_col] + (
        [stiffness_col] if stiffness_col in df.columns else []
    )
    if "subject_group" in df.columns:
        keep_cols.insert(1, "subject_group")
    if "n_trials" in df.columns:
        keep_cols.append("n_trials")
    keep_cols += metrics
    within_subject = df[[c for c in keep_cols if c in df.columns]].copy()
    for metric in metrics:
        values = pd.to_numeric(within_subject[metric], errors="coerce")
        subject_mean = values.groupby(within_subject[subject_col]).transform("mean")
        subject_sd = values.groupby(within_subject[subject_col]).transform(
            lambda s: s.std(ddof=1)
        )
        within_subject[f"{metric}_within_subject_centered"] = values - subject_mean
        within_subject[f"{metric}_within_subject_z"] = np.where(
            subject_sd > 0, (values - subject_mean) / subject_sd, np.nan
        )
        finger_mean = values.groupby(
            [within_subject[subject_col], within_subject[finger_col]], dropna=False
        ).transform("mean")
        within_subject[f"{metric}_within_subject_finger_centered"] = (
            values - finger_mean
        )

    finger_metric_rows: list[dict[str, Any]] = []
    subject_finger = (
        within_subject.groupby([subject_col, finger_col], dropna=False)
        .agg(
            **{metric: (metric, "mean") for metric in metrics},
            n_stiffness_levels=(stiffness_col, "nunique")
            if stiffness_col in within_subject.columns
            else (metrics[0], "count"),
        )
        .reset_index()
    )
    for finger, finger_df in subject_finger.groupby(finger_col, dropna=False):
        for metric in metrics:
            vals = pd.to_numeric(finger_df[metric], errors="coerce").dropna()
            finger_metric_rows.append(
                {
                    "finger_condition": finger,
                    "metric": metric,
                    "n_subjects": int(vals.size),
                    "mean": float(vals.mean()) if vals.size else np.nan,
                    "median": float(vals.median()) if vals.size else np.nan,
                    "sem": _sem(vals),
                }
            )
    finger_metric_summary = pd.DataFrame(finger_metric_rows)

    slope_rows: list[dict[str, Any]] = []
    if stiffness_col in within_subject.columns:
        grouped = within_subject.groupby([subject_col, finger_col], dropna=False)
        for (subject, finger), g in grouped:
            for metric in metrics:
                valid = g[[stiffness_col, metric]].dropna().sort_values(stiffness_col)
                low_high_delta = np.nan
                if valid[stiffness_col].nunique() >= 2:
                    low_x = valid[stiffness_col].min()
                    high_x = valid[stiffness_col].max()
                    low_high_delta = float(
                        valid.loc[valid[stiffness_col] == high_x, metric].mean()
                        - valid.loc[valid[stiffness_col] == low_x, metric].mean()
                    )
                slope_rows.append(
                    {
                        "subject_id": subject,
                        "finger_condition": finger,
                        "metric": metric,
                        "n_stiffness_levels": int(valid[stiffness_col].nunique())
                        if not valid.empty
                        else 0,
                        "slope_per_stiffness_unit": _ols_slope(
                            valid[stiffness_col], valid[metric]
                        ),
                        "high_minus_low_stiffness_delta": low_high_delta,
                    }
                )
    within_finger_stiffness_effects = pd.DataFrame(slope_rows)

    summary_rows: list[dict[str, Any]] = []
    if not within_finger_stiffness_effects.empty:
        for (finger, metric), g in within_finger_stiffness_effects.groupby(
            ["finger_condition", "metric"], dropna=False
        ):
            slopes = pd.to_numeric(
                g["slope_per_stiffness_unit"], errors="coerce"
            ).dropna()
            deltas = pd.to_numeric(
                g["high_minus_low_stiffness_delta"], errors="coerce"
            ).dropna()
            summary_rows.append(
                {
                    "finger_condition": finger,
                    "metric": metric,
                    "n_subjects_with_slope": int(slopes.size),
                    "mean_slope_per_stiffness_unit": float(slopes.mean())
                    if slopes.size
                    else np.nan,
                    "sem_slope_per_stiffness_unit": _sem(slopes),
                    "slope_sign_flip_p": _paired_sign_flip_p(slopes),
                    "n_subjects_with_high_low_delta": int(deltas.size),
                    "mean_high_minus_low_stiffness_delta": float(deltas.mean())
                    if deltas.size
                    else np.nan,
                    "sem_high_minus_low_stiffness_delta": _sem(deltas),
                    "high_low_delta_sign_flip_p": _paired_sign_flip_p(deltas),
                }
            )
    within_finger_stiffness_effect_summary = pd.DataFrame(summary_rows)

    paired_rows: list[dict[str, Any]] = []
    fingers = [f for f in FINGER_ORDER if f in set(subject_finger[finger_col].dropna())]
    fingers += sorted(
        [
            f
            for f in subject_finger[finger_col].dropna().unique()
            if f not in FINGER_ORDER
        ]
    )
    for i, finger_a in enumerate(fingers):
        for finger_b in fingers[i + 1 :]:
            a = subject_finger[subject_finger[finger_col] == finger_a][
                [subject_col] + metrics
            ].rename(columns={m: f"{m}_a" for m in metrics})
            b = subject_finger[subject_finger[finger_col] == finger_b][
                [subject_col] + metrics
            ].rename(columns={m: f"{m}_b" for m in metrics})
            merged = a.merge(b, on=subject_col, how="inner")
            for metric in metrics:
                diff = pd.to_numeric(
                    merged[f"{metric}_b"], errors="coerce"
                ) - pd.to_numeric(merged[f"{metric}_a"], errors="coerce")
                effect = _paired_effect_summary(diff)
                paired_rows.append(
                    {
                        "finger_a": finger_a,
                        "finger_b": finger_b,
                        "comparison": f"{finger_b} - {finger_a}",
                        "metric": metric,
                        "mean_a": float(
                            pd.to_numeric(merged[f"{metric}_a"], errors="coerce").mean()
                        )
                        if not merged.empty
                        else np.nan,
                        "mean_b": float(
                            pd.to_numeric(merged[f"{metric}_b"], errors="coerce").mean()
                        )
                        if not merged.empty
                        else np.nan,
                        **effect,
                    }
                )
    finger_comparison_paired = pd.DataFrame(paired_rows)

    by_stiffness_rows: list[dict[str, Any]] = []
    if stiffness_col in within_subject.columns:
        subject_finger_stiffness = (
            within_subject.groupby(
                [subject_col, stiffness_col, finger_col], dropna=False
            )
            .agg(**{metric: (metric, "mean") for metric in metrics})
            .reset_index()
        )
        for stiffness_value, s_df in subject_finger_stiffness.groupby(
            stiffness_col, dropna=False
        ):
            fingers_s = [f for f in fingers if f in set(s_df[finger_col].dropna())]
            for i, finger_a in enumerate(fingers_s):
                for finger_b in fingers_s[i + 1 :]:
                    a = s_df[s_df[finger_col] == finger_a][
                        [subject_col] + metrics
                    ].rename(columns={m: f"{m}_a" for m in metrics})
                    b = s_df[s_df[finger_col] == finger_b][
                        [subject_col] + metrics
                    ].rename(columns={m: f"{m}_b" for m in metrics})
                    merged = a.merge(b, on=subject_col, how="inner")
                    for metric in metrics:
                        diff = pd.to_numeric(
                            merged[f"{metric}_b"], errors="coerce"
                        ) - pd.to_numeric(merged[f"{metric}_a"], errors="coerce")
                        effect = _paired_effect_summary(diff)
                        by_stiffness_rows.append(
                            {
                                "stiffness_value": stiffness_value,
                                "finger_a": finger_a,
                                "finger_b": finger_b,
                                "comparison": f"{finger_b} - {finger_a}",
                                "metric": metric,
                                "mean_a": float(
                                    pd.to_numeric(
                                        merged[f"{metric}_a"], errors="coerce"
                                    ).mean()
                                )
                                if not merged.empty
                                else np.nan,
                                "mean_b": float(
                                    pd.to_numeric(
                                        merged[f"{metric}_b"], errors="coerce"
                                    ).mean()
                                )
                                if not merged.empty
                                else np.nan,
                                **effect,
                            }
                        )
    finger_comparison_by_stiffness_paired = pd.DataFrame(by_stiffness_rows)

    return {
        "within_subject": within_subject,
        "finger_metric_summary": finger_metric_summary,
        "within_finger_stiffness_effects": within_finger_stiffness_effects,
        "within_finger_stiffness_effect_summary": within_finger_stiffness_effect_summary,
        "finger_comparison_paired": finger_comparison_paired,
        "finger_comparison_by_stiffness_paired": finger_comparison_by_stiffness_paired,
    }


def save_motor_control_figures(
    output_root: Path,
    motor_control_results: dict[str, pd.DataFrame],
    *,
    metrics: Optional[list[str]] = None,
    fig_dpi: int = 160,
) -> list[Path]:
    """Save repeated-measures motor-control figures."""
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    selected_metrics = metrics or [
        "mean_speed_px_s",
        "mean_acceleration_px_s2",
        "mean_path_length_px",
        "mean_straightness_index",
    ]

    def _scope_subsets(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        scopes = [("All", df)]
        if "subject_group" not in df.columns:
            return scopes
        group_text = df["subject_group"].astype(str).str.upper().str[:1]
        scopes.extend([("L", df[group_text == "L"]), ("N", df[group_text == "N"])])
        return scopes

    def _plot_motor_grid(
        plot_df: pd.DataFrame,
        *,
        category_col: str,
        value_col: str,
        sem_col: str,
        filename: str,
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> None:
        if plot_df.empty:
            return
        rows = ["All", "L", "N"]
        fig, axes = plt.subplots(
            len(rows),
            len(selected_metrics),
            figsize=(4.3 * len(selected_metrics), 3.3 * len(rows)),
            squeeze=False,
            sharey="col",
        )
        for row_i, scope in enumerate(rows):
            for col_i, metric in enumerate(selected_metrics):
                ax = axes[row_i, col_i]
                sub = plot_df[
                    (plot_df["scope"] == scope) & (plot_df["metric"] == metric)
                ].copy()
                if category_col == "finger_condition":
                    order = _ordered_fingers(sub[category_col].dropna().unique())
                    colors = [FINGER_TO_COLOR.get(str(v), "#777777") for v in order]
                else:
                    order = sorted(sub[category_col].dropna().unique())
                    cmap = plt.get_cmap(STIFFNESS_CMAP)
                    colors = [cmap(i / max(1, len(order) - 1)) for i in range(len(order))]
                sub = sub.set_index(category_col).reindex(order)
                sub = sub.dropna(subset=[value_col], how="all")
                if sub.empty:
                    ax.axis("off")
                    continue
                if "slope" in value_col:
                    ax.axhline(0, color="black", linewidth=0.8)
                ax.bar(
                    [str(v) for v in sub.index],
                    sub[value_col],
                    yerr=sub[sem_col] if sem_col in sub.columns else None,
                    capsize=3,
                    color=colors[: len(sub)],
                    alpha=0.85,
                )
                if row_i == 0:
                    ax.set_title(metric)
                if col_i == 0:
                    ax.set_ylabel(f"{scope}\n{ylabel}")
                if row_i == len(rows) - 1:
                    ax.set_xlabel(xlabel)
                ax.grid(axis="y", alpha=0.25)
        fig.suptitle(title, y=1.01)
        fig.tight_layout()
        out = fig_dir / filename
        fig.savefig(out, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    within_subject = motor_control_results.get("within_subject", pd.DataFrame()).copy()
    if not within_subject.empty:
        within_subject["finger_condition"] = within_subject["finger_condition"].map(
            normalize_finger_condition
        )
        if "stiffness_value" in within_subject.columns:
            within_subject["stiffness_value"] = pd.to_numeric(
                within_subject["stiffness_value"], errors="coerce"
            )
        for metric in selected_metrics:
            if metric in within_subject.columns:
                within_subject[metric] = pd.to_numeric(
                    within_subject[metric], errors="coerce"
                )

        finger_rows: list[dict[str, Any]] = []
        if "finger_condition" in within_subject.columns:
            for scope, scope_df in _scope_subsets(within_subject):
                subject_finger = (
                    scope_df.groupby(["subject_id", "finger_condition"], dropna=False)
                    .agg(
                        **{
                            metric: (metric, "mean")
                            for metric in selected_metrics
                            if metric in scope_df.columns
                        }
                    )
                    .reset_index()
                )
                for finger, finger_df in subject_finger.groupby(
                    "finger_condition", dropna=False
                ):
                    for metric in selected_metrics:
                        if metric not in finger_df.columns:
                            continue
                        vals = pd.to_numeric(finger_df[metric], errors="coerce").dropna()
                        finger_rows.append(
                            {
                                "scope": scope,
                                "finger_condition": finger,
                                "metric": metric,
                                "mean": float(vals.mean()) if len(vals) else np.nan,
                                "sem": _sem(vals),
                            }
                        )
        _plot_motor_grid(
            pd.DataFrame(finger_rows),
            category_col="finger_condition",
            value_col="mean",
            sem_col="sem",
            filename="motor_control_finger_metric_summary.png",
            title="Motor-control metric summary by finger and group",
            xlabel="Finger",
            ylabel="Subject mean",
        )

        stiffness_rows: list[dict[str, Any]] = []
        if "stiffness_value" in within_subject.columns:
            for scope, scope_df in _scope_subsets(within_subject):
                subject_stiffness = (
                    scope_df.groupby(["subject_id", "stiffness_value"], dropna=False)
                    .agg(
                        **{
                            metric: (metric, "mean")
                            for metric in selected_metrics
                            if metric in scope_df.columns
                        }
                    )
                    .reset_index()
                )
                for stiffness, stiffness_df in subject_stiffness.groupby(
                    "stiffness_value", dropna=False
                ):
                    for metric in selected_metrics:
                        if metric not in stiffness_df.columns:
                            continue
                        vals = pd.to_numeric(
                            stiffness_df[metric], errors="coerce"
                        ).dropna()
                        stiffness_rows.append(
                            {
                                "scope": scope,
                                "stiffness_value": stiffness,
                                "metric": metric,
                                "mean": float(vals.mean()) if len(vals) else np.nan,
                                "sem": _sem(vals),
                            }
                        )
        _plot_motor_grid(
            pd.DataFrame(stiffness_rows),
            category_col="stiffness_value",
            value_col="mean",
            sem_col="sem",
            filename="motor_control_stiffness_metric_summary.png",
            title="Motor-control metric summary by stiffness value and group",
            xlabel="Stiffness value",
            ylabel="Subject mean",
        )

    slopes = motor_control_results.get("within_finger_stiffness_effects", pd.DataFrame())
    if not slopes.empty:
        slopes = slopes.copy()
        slopes["finger_condition"] = slopes["finger_condition"].map(
            normalize_finger_condition
        )
        if "subject_group" not in slopes.columns and not within_subject.empty:
            subject_groups = within_subject[["subject_id", "subject_group"]].dropna()
            subject_groups = subject_groups.drop_duplicates("subject_id")
            slopes = slopes.merge(subject_groups, on="subject_id", how="left")
        slope_rows: list[dict[str, Any]] = []
        for scope, scope_df in _scope_subsets(slopes):
            for (finger, metric), g in scope_df.groupby(
                ["finger_condition", "metric"], dropna=False
            ):
                if metric not in selected_metrics:
                    continue
                vals = pd.to_numeric(
                    g["slope_per_stiffness_unit"], errors="coerce"
                ).dropna()
                slope_rows.append(
                    {
                        "scope": scope,
                        "finger_condition": finger,
                        "metric": metric,
                        "mean_slope": float(vals.mean()) if len(vals) else np.nan,
                        "sem_slope": _sem(vals),
                    }
                )
        _plot_motor_grid(
            pd.DataFrame(slope_rows),
            category_col="finger_condition",
            value_col="mean_slope",
            sem_col="sem_slope",
            filename="motor_control_within_finger_stiffness_slopes.png",
            title="Within-finger stiffness slopes by finger and group",
            xlabel="Finger",
            ylabel="Slope per stiffness unit",
        )

    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "motor_control_figure_manifest.csv",
    )
    return paths


def compute_success_kinematic_z_analysis(
    trial_summary: pd.DataFrame,
    side_z_trial_summary: Optional[pd.DataFrame] = None,
    *,
    metric_columns: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Compare successful vs unsuccessful trials for kinematics and Z/lift.

    The main inferential table uses within-subject/finger contrasts:
    ``successful minus unsuccessful``. This avoids claiming that a trial-pooled
    difference is a participant-level effect.
    """
    empty = {
        "trial_success_table": pd.DataFrame(),
        "success_contrast_by_subject_finger": pd.DataFrame(),
        "success_contrast_summary": pd.DataFrame(),
        "success_contrast_by_finger_summary": pd.DataFrame(),
    }
    if trial_summary.empty or "correct_response" not in trial_summary.columns:
        return empty

    df = trial_summary.copy()
    df["finger_condition"] = df.get("finger_condition", np.nan).map(
        normalize_finger_condition
    )
    df["correct_response"] = pd.to_numeric(df["correct_response"], errors="coerce")
    if side_z_trial_summary is not None and not side_z_trial_summary.empty:
        side = side_z_trial_summary.copy()
        if "finger_condition" in side.columns:
            side["finger_condition"] = side["finger_condition"].map(
                normalize_finger_condition
            )
        keys = [
            c
            for c in [
                "subject_id",
                "trial_index_raw",
                "stiffness_segment_id",
                "stiffness_value",
            ]
            if c in df.columns and c in side.columns
        ]
        side_cols = keys + [
            c
            for c in [
                "n_side_samples",
                "side_detection_rate",
                "mean_side_z_lift_px",
                "max_side_z_lift_px",
                "mean_side_mask_area_px",
            ]
            if c in side.columns
        ]
        if keys and len(side_cols) > len(keys):
            df = df.merge(
                side[side_cols].drop_duplicates(keys),
                on=keys,
                how="left",
                suffixes=("", "_side"),
            )

    metrics = _available_metric_columns(df, metric_columns or TRIAL_SUCCESS_METRICS)
    if not metrics:
        return {**empty, "trial_success_table": df}

    contrast_rows: list[dict[str, Any]] = []
    group_cols = [
        c
        for c in ["subject_id", "subject_group", "finger_condition"]
        if c in df.columns
    ]
    for group_values, g in df.dropna(subset=["correct_response"]).groupby(
        group_cols, dropna=False
    ):
        group_values = (
            group_values if isinstance(group_values, tuple) else (group_values,)
        )
        base = dict(zip(group_cols, group_values))
        success = g[g["correct_response"] == 1]
        failure = g[g["correct_response"] == 0]
        for metric in metrics:
            s_vals = pd.to_numeric(success[metric], errors="coerce").dropna()
            f_vals = pd.to_numeric(failure[metric], errors="coerce").dropna()
            contrast_rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_success_trials": int(s_vals.size),
                    "n_failure_trials": int(f_vals.size),
                    "success_mean": float(s_vals.mean()) if s_vals.size else np.nan,
                    "failure_mean": float(f_vals.mean()) if f_vals.size else np.nan,
                    "success_minus_failure": float(s_vals.mean() - f_vals.mean())
                    if s_vals.size and f_vals.size
                    else np.nan,
                }
            )
    success_contrast_by_subject_finger = pd.DataFrame(contrast_rows)

    summary_rows: list[dict[str, Any]] = []
    if not success_contrast_by_subject_finger.empty:
        for metric, g in success_contrast_by_subject_finger.groupby(
            "metric", dropna=False
        ):
            effect = _paired_effect_summary(g["success_minus_failure"])
            summary_rows.append({"metric": metric, **effect})
    success_contrast_summary = pd.DataFrame(summary_rows)

    finger_rows: list[dict[str, Any]] = []
    if not success_contrast_by_subject_finger.empty:
        for (finger, metric), g in success_contrast_by_subject_finger.groupby(
            ["finger_condition", "metric"], dropna=False
        ):
            effect = _paired_effect_summary(g["success_minus_failure"])
            finger_rows.append({"finger_condition": finger, "metric": metric, **effect})
    success_contrast_by_finger_summary = pd.DataFrame(finger_rows)

    return {
        "trial_success_table": df,
        "success_contrast_by_subject_finger": success_contrast_by_subject_finger,
        "success_contrast_summary": success_contrast_summary,
        "success_contrast_by_finger_summary": success_contrast_by_finger_summary,
    }


def compute_trajectory_similarity_analysis(
    time_bins: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Analyze between-trajectory structure from normalized trajectories.

    Produces dependency-light trajectory features: subject/finger/stiffness mean
    trajectory and variability, paired distances between fingers, and distance
    between successful and unsuccessful trajectory templates.
    """
    empty = {
        "subject_finger_trajectory": pd.DataFrame(),
        "trajectory_variability_summary": pd.DataFrame(),
        "finger_trajectory_distance_paired": pd.DataFrame(),
        "finger_trajectory_distance_summary": pd.DataFrame(),
        "success_failure_trajectory_distance": pd.DataFrame(),
        "success_failure_trajectory_distance_summary": pd.DataFrame(),
    }
    required = {
        "subject_id",
        "finger_condition",
        "stiffness_value",
        "trajectory_time_bin",
        "x_centered_px",
        "y_centered_px",
    }
    if time_bins.empty or not required.issubset(time_bins.columns):
        return empty

    tb = add_success_label_column(
        add_protocol_demographic_factors(add_workspace_normalization_columns(time_bins))
    )
    tb["finger_condition"] = tb["finger_condition"].map(normalize_finger_condition)
    for col in [
        "stiffness_value",
        "trajectory_time_bin",
        "x_centered_px",
        "y_centered_px",
        "speed_px_s",
        "correct_response",
    ]:
        if col in tb.columns:
            tb[col] = pd.to_numeric(tb[col], errors="coerce")

    group_cols = [
        "subject_id",
        "finger_condition",
        "stiffness_value",
        "trajectory_time_bin",
    ]
    subject_finger_trajectory = (
        tb.groupby(group_cols, dropna=False)
        .agg(
            n_segments=("tracking_file", "nunique")
            if "tracking_file" in tb.columns
            else ("x_centered_px", "count"),
            time_fraction=("time_fraction", "mean")
            if "time_fraction" in tb.columns
            else ("trajectory_time_bin", "mean"),
            mean_x_centered_px=("x_centered_px", "mean"),
            mean_y_centered_px=("y_centered_px", "mean"),
            sd_x_centered_px=("x_centered_px", "std"),
            sd_y_centered_px=("y_centered_px", "std"),
            mean_r_center_px=("r_center_px", "mean")
            if "r_center_px" in tb.columns
            else ("x_centered_px", "mean"),
            mean_speed_px_s=("speed_px_s", "mean")
            if "speed_px_s" in tb.columns
            else ("x_centered_px", "mean"),
            sd_speed_px_s=("speed_px_s", "std")
            if "speed_px_s" in tb.columns
            else ("x_centered_px", "std"),
        )
        .reset_index()
    )
    subject_finger_trajectory["xy_variability_px"] = np.hypot(
        subject_finger_trajectory["sd_x_centered_px"],
        subject_finger_trajectory["sd_y_centered_px"],
    )

    variability = (
        subject_finger_trajectory.groupby(
            ["subject_id", "finger_condition", "stiffness_value"], dropna=False
        )
        .agg(
            n_time_bins=("trajectory_time_bin", "count"),
            mean_xy_variability_px=("xy_variability_px", "mean"),
            max_xy_variability_px=("xy_variability_px", "max"),
            mean_speed_variability_px_s=("sd_speed_px_s", "mean"),
            mean_trajectory_radius_px=("mean_r_center_px", "mean"),
        )
        .reset_index()
    )

    pair_rows: list[dict[str, Any]] = []
    fingers_seen = [
        f
        for f in FINGER_ORDER
        if f in set(subject_finger_trajectory["finger_condition"].dropna())
    ]
    fingers_seen += sorted(
        [
            f
            for f in subject_finger_trajectory["finger_condition"].dropna().unique()
            if f not in FINGER_ORDER
        ]
    )
    for (subject, stiffness), s_df in subject_finger_trajectory.groupby(
        ["subject_id", "stiffness_value"], dropna=False
    ):
        for i, finger_a in enumerate(fingers_seen):
            for finger_b in fingers_seen[i + 1 :]:
                a = s_df[s_df["finger_condition"] == finger_a][
                    [
                        "trajectory_time_bin",
                        "mean_x_centered_px",
                        "mean_y_centered_px",
                        "mean_speed_px_s",
                    ]
                ]
                b = s_df[s_df["finger_condition"] == finger_b][
                    [
                        "trajectory_time_bin",
                        "mean_x_centered_px",
                        "mean_y_centered_px",
                        "mean_speed_px_s",
                    ]
                ]
                merged = a.merge(b, on="trajectory_time_bin", suffixes=("_a", "_b"))
                if merged.empty:
                    continue
                xy_dist = np.hypot(
                    merged["mean_x_centered_px_b"] - merged["mean_x_centered_px_a"],
                    merged["mean_y_centered_px_b"] - merged["mean_y_centered_px_a"],
                )
                speed_diff = merged["mean_speed_px_s_b"] - merged["mean_speed_px_s_a"]
                pair_rows.append(
                    {
                        "subject_id": subject,
                        "stiffness_value": stiffness,
                        "finger_a": finger_a,
                        "finger_b": finger_b,
                        "comparison": f"{finger_b} - {finger_a}",
                        "n_matched_time_bins": int(len(merged)),
                        "mean_xy_trajectory_distance_px": float(np.nanmean(xy_dist)),
                        "rms_xy_trajectory_distance_px": float(
                            np.sqrt(np.nanmean(np.square(xy_dist)))
                        ),
                        "speed_profile_rmse_px_s": float(
                            np.sqrt(np.nanmean(np.square(speed_diff)))
                        ),
                    }
                )
    finger_trajectory_distance_paired = pd.DataFrame(pair_rows)

    summary_rows: list[dict[str, Any]] = []
    if not finger_trajectory_distance_paired.empty:
        for comparison, g in finger_trajectory_distance_paired.groupby(
            "comparison", dropna=False
        ):
            for metric in [
                "mean_xy_trajectory_distance_px",
                "rms_xy_trajectory_distance_px",
                "speed_profile_rmse_px_s",
            ]:
                vals = pd.to_numeric(g[metric], errors="coerce").dropna()
                summary_rows.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "n_subject_stiffness_pairs": int(vals.size),
                        "mean": float(vals.mean()) if vals.size else np.nan,
                        "median": float(vals.median()) if vals.size else np.nan,
                        "sem": _sem(vals),
                    }
                )
    finger_trajectory_distance_summary = pd.DataFrame(summary_rows)

    success_rows: list[dict[str, Any]] = []
    if "correct_response" in tb.columns:
        success_templates = (
            tb.dropna(subset=["correct_response"])
            .groupby(
                [
                    "subject_id",
                    "finger_condition",
                    "stiffness_value",
                    "correct_response",
                    "trajectory_time_bin",
                ],
                dropna=False,
            )
            .agg(
                x_centered_px=("x_centered_px", "mean"),
                y_centered_px=("y_centered_px", "mean"),
                speed_px_s=("speed_px_s", "mean")
                if "speed_px_s" in tb.columns
                else ("x_centered_px", "mean"),
            )
            .reset_index()
        )
        for (subject, finger, stiffness), g in success_templates.groupby(
            ["subject_id", "finger_condition", "stiffness_value"], dropna=False
        ):
            ok = g[g["correct_response"] == 1]
            bad = g[g["correct_response"] == 0]
            merged = ok.merge(
                bad, on="trajectory_time_bin", suffixes=("_success", "_failure")
            )
            if merged.empty:
                continue
            xy_dist = np.hypot(
                merged["x_centered_px_success"] - merged["x_centered_px_failure"],
                merged["y_centered_px_success"] - merged["y_centered_px_failure"],
            )
            speed_diff = merged["speed_px_s_success"] - merged["speed_px_s_failure"]
            success_rows.append(
                {
                    "subject_id": subject,
                    "finger_condition": finger,
                    "stiffness_value": stiffness,
                    "n_matched_time_bins": int(len(merged)),
                    "success_failure_mean_xy_distance_px": float(np.nanmean(xy_dist)),
                    "success_failure_rms_xy_distance_px": float(
                        np.sqrt(np.nanmean(np.square(xy_dist)))
                    ),
                    "success_failure_speed_rmse_px_s": float(
                        np.sqrt(np.nanmean(np.square(speed_diff)))
                    ),
                }
            )
    success_failure_trajectory_distance = pd.DataFrame(success_rows)

    sf_summary_rows: list[dict[str, Any]] = []
    if not success_failure_trajectory_distance.empty:
        for finger, g in success_failure_trajectory_distance.groupby(
            "finger_condition", dropna=False
        ):
            for metric in [
                "success_failure_mean_xy_distance_px",
                "success_failure_rms_xy_distance_px",
                "success_failure_speed_rmse_px_s",
            ]:
                vals = pd.to_numeric(g[metric], errors="coerce").dropna()
                sf_summary_rows.append(
                    {
                        "finger_condition": finger,
                        "metric": metric,
                        "n_subject_stiffness_pairs": int(vals.size),
                        "mean": float(vals.mean()) if vals.size else np.nan,
                        "median": float(vals.median()) if vals.size else np.nan,
                        "sem": _sem(vals),
                    }
                )
    success_failure_trajectory_distance_summary = pd.DataFrame(sf_summary_rows)

    return {
        "subject_finger_trajectory": subject_finger_trajectory,
        "trajectory_variability_summary": variability,
        "finger_trajectory_distance_paired": finger_trajectory_distance_paired,
        "finger_trajectory_distance_summary": finger_trajectory_distance_summary,
        "success_failure_trajectory_distance": success_failure_trajectory_distance,
        "success_failure_trajectory_distance_summary": success_failure_trajectory_distance_summary,
    }


def save_advanced_kinematic_figures(
    output_root: Path,
    success_results: dict[str, pd.DataFrame],
    trajectory_results: dict[str, pd.DataFrame],
    *,
    fig_dpi: int = 160,
) -> list[Path]:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    success_summary = success_results.get("success_contrast_summary", pd.DataFrame())
    success_metrics = [
        "mean_speed_px_s",
        "path_length_px",
        "straightness_index",
        "mean_acceleration_px_s2",
        "mean_side_z_lift_px",
        "max_side_z_lift_px",
    ]
    if not success_summary.empty:
        sub = success_summary[success_summary["metric"].isin(success_metrics)].copy()
        if not sub.empty:
            order = [m for m in success_metrics if m in set(sub["metric"])]
            sub = sub.set_index("metric").reindex(order).reset_index()
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.barh(
                sub["metric"],
                sub["mean_difference"],
                xerr=sub["sem_difference"],
                color="#4C78A8",
                alpha=0.85,
            )
            ax.set_xlabel("Successful - unsuccessful trials")
            ax.set_title(
                "Within-subject/finger success-linked kinematic and Z contrasts"
            )
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            out = fig_dir / "success_linked_kinematic_z_contrasts.png"
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

    traj_summary = trajectory_results.get(
        "finger_trajectory_distance_summary", pd.DataFrame()
    )
    if not traj_summary.empty:
        sub = traj_summary[
            traj_summary["metric"] == "mean_xy_trajectory_distance_px"
        ].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.bar(
                sub["comparison"],
                sub["mean"],
                yerr=sub["sem"],
                capsize=3,
                color="#F58518",
                alpha=0.85,
            )
            ax.set_ylabel("Mean XY trajectory distance (px)")
            ax.set_xlabel("Finger comparison")
            ax.set_title("Paired between-finger trajectory separation")
            ax.grid(axis="y", alpha=0.25)
            fig.autofmt_xdate(rotation=30, ha="right")
            fig.tight_layout()
            out = fig_dir / "between_finger_trajectory_distance.png"
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

    sf_summary = trajectory_results.get(
        "success_failure_trajectory_distance_summary", pd.DataFrame()
    )
    if not sf_summary.empty:
        sub = sf_summary[
            sf_summary["metric"] == "success_failure_mean_xy_distance_px"
        ].copy()
        if not sub.empty:
            present = set(sub["finger_condition"])
            sub = (
                sub.set_index("finger_condition")
                .reindex([f for f in FINGER_ORDER if f in present])
                .dropna(subset=["mean"], how="all")
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(6, 4.2))
            colors = [
                FINGER_TO_COLOR.get(str(f), "#777777") for f in sub["finger_condition"]
            ]
            ax.bar(
                sub["finger_condition"],
                sub["mean"],
                yerr=sub["sem"],
                capsize=3,
                color=colors,
                alpha=0.85,
            )
            ax.set_ylabel("Success vs failure trajectory distance (px)")
            ax.set_xlabel("Finger")
            ax.set_title("Trajectory templates separating correct and incorrect trials")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            out = fig_dir / "success_failure_trajectory_distance.png"
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "advanced_kinematic_figure_manifest.csv",
    )
    return paths


def compute_subject_spatial_trajectory_analysis(
    time_bins: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create subject-first XY trajectory tables in physical image space.

    Unlike group summaries, every returned row preserves ``subject_id``. The
    trajectory table contains one normalized XY path per subject x finger x
    stiffness x time-bin; the summary table collapses only within that same
    person/finger/stiffness path.
    """
    empty = {
        "subject_xy_trajectory": pd.DataFrame(),
        "subject_spatial_trajectory_summary": pd.DataFrame(),
        "subject_finger_spatial_distance": pd.DataFrame(),
        "subject_spatial_metric_distribution": pd.DataFrame(),
    }
    required = {
        "subject_id",
        "finger_condition",
        "stiffness_value",
        "trajectory_time_bin",
        "x_centered_px",
        "y_centered_px",
    }
    if time_bins.empty or not required.issubset(time_bins.columns):
        return empty

    tb = add_success_label_column(
        add_protocol_demographic_factors(add_workspace_normalization_columns(time_bins))
    )
    tb["finger_condition"] = tb["finger_condition"].map(normalize_finger_condition)
    for col in [
        "stiffness_value",
        "trajectory_time_bin",
        "time_fraction",
        "x_centered_px",
        "y_centered_px",
        "r_center_px",
        "x_workspace_cm",
        "y_workspace_cm",
        "r_workspace_cm",
        "r_workspace_normalized",
        "speed_px_s",
        "correct_response",
    ]:
        if col in tb.columns:
            tb[col] = pd.to_numeric(tb[col], errors="coerce")

    group_cols = [
        "subject_id",
        "finger_condition",
        "stiffness_value",
        "trajectory_time_bin",
    ]
    agg_spec: dict[str, tuple[str, str]] = {
        "n_segments": ("tracking_file", "nunique")
        if "tracking_file" in tb.columns
        else ("x_centered_px", "count"),
        "time_fraction": ("time_fraction", "mean")
        if "time_fraction" in tb.columns
        else ("trajectory_time_bin", "mean"),
        "x_centered_px": ("x_centered_px", "mean"),
        "y_centered_px": ("y_centered_px", "mean"),
        "sd_x_centered_px": ("x_centered_px", "std"),
        "sd_y_centered_px": ("y_centered_px", "std"),
        "r_center_px": ("r_center_px", "mean")
        if "r_center_px" in tb.columns
        else ("x_centered_px", "mean"),
        "speed_px_s": ("speed_px_s", "mean")
        if "speed_px_s" in tb.columns
        else ("x_centered_px", "mean"),
    }
    if "subject_group" in tb.columns:
        agg_spec["subject_group"] = ("subject_group", "first")
    for col in [
        "experiment_group",
        "workspace_setup",
        "workspace_label",
        "workspace_width_cm",
        "workspace_height_cm",
        "x_workspace_cm",
        "y_workspace_cm",
        "r_workspace_cm",
        "r_workspace_normalized",
        "success_label",
    ]:
        if col in tb.columns:
            agg_spec[col] = (col, "first" if col.endswith(("setup", "label", "group")) else "mean")
    if "correct_response" in tb.columns:
        agg_spec["success_rate"] = ("correct_response", "mean")
    subject_xy_trajectory = (
        tb.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    )
    subject_xy_trajectory["xy_sd_radius_px"] = np.hypot(
        subject_xy_trajectory["sd_x_centered_px"],
        subject_xy_trajectory["sd_y_centered_px"],
    )

    summary_rows: list[dict[str, Any]] = []
    for (subject, finger, stiffness), g in subject_xy_trajectory.groupby(
        ["subject_id", "finger_condition", "stiffness_value"], dropna=False
    ):
        g = g.sort_values("trajectory_time_bin")
        x = pd.to_numeric(g["x_centered_px"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["y_centered_px"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue
        dx = np.diff(x)
        dy = np.diff(y)
        path_length = float(np.nansum(np.hypot(dx, dy))) if x.size > 1 else 0.0
        net_displacement = (
            float(np.hypot(x[-1] - x[0], y[-1] - y[0])) if x.size > 1 else 0.0
        )
        signed_area = (
            float(0.5 * np.nansum(x[:-1] * y[1:] - x[1:] * y[:-1]))
            if x.size > 2
            else np.nan
        )
        row = {
            "subject_id": subject,
            "finger_condition": finger,
            "stiffness_value": stiffness,
            "subject_group": g["subject_group"].dropna().iloc[0]
            if "subject_group" in g.columns and g["subject_group"].notna().any()
            else np.nan,
            EXPERIMENT_GROUP_COLUMN: g[EXPERIMENT_GROUP_COLUMN].dropna().iloc[0]
            if EXPERIMENT_GROUP_COLUMN in g.columns
            and g[EXPERIMENT_GROUP_COLUMN].notna().any()
            else np.nan,
            "workspace_setup": g["workspace_setup"].dropna().iloc[0]
            if "workspace_setup" in g.columns and g["workspace_setup"].notna().any()
            else np.nan,
            "workspace_label": g["workspace_label"].dropna().iloc[0]
            if "workspace_label" in g.columns and g["workspace_label"].notna().any()
            else np.nan,
            "workspace_width_cm": float(
                pd.to_numeric(g.get("workspace_width_cm"), errors="coerce").mean()
            )
            if "workspace_width_cm" in g.columns
            else np.nan,
            "workspace_height_cm": float(
                pd.to_numeric(g.get("workspace_height_cm"), errors="coerce").mean()
            )
            if "workspace_height_cm" in g.columns
            else np.nan,
            "n_time_bins": int(x.size),
            "n_segments_mean": float(
                pd.to_numeric(g["n_segments"], errors="coerce").mean()
            ),
            "start_x_centered_px": float(x[0]),
            "start_y_centered_px": float(y[0]),
            "end_x_centered_px": float(x[-1]),
            "end_y_centered_px": float(y[-1]),
            "centroid_x_centered_px": float(np.nanmean(x)),
            "centroid_y_centered_px": float(np.nanmean(y)),
            "min_x_centered_px": float(np.nanmin(x)),
            "max_x_centered_px": float(np.nanmax(x)),
            "min_y_centered_px": float(np.nanmin(y)),
            "max_y_centered_px": float(np.nanmax(y)),
            "xy_width_px": float(np.nanmax(x) - np.nanmin(x)),
            "xy_height_px": float(np.nanmax(y) - np.nanmin(y)),
            "spatial_extent_area_px2": float(
                (np.nanmax(x) - np.nanmin(x)) * (np.nanmax(y) - np.nanmin(y))
            ),
            "mean_radius_px": float(np.nanmean(np.hypot(x, y))),
            "max_radius_px": float(np.nanmax(np.hypot(x, y))),
            "mean_trajectory_speed_px_s": float(
                pd.to_numeric(g["speed_px_s"], errors="coerce").mean()
            )
            if "speed_px_s" in g.columns
            else np.nan,
            "mean_xy_sd_radius_px": float(
                pd.to_numeric(g["xy_sd_radius_px"], errors="coerce").mean()
            ),
            "mean_trajectory_path_length_px": path_length,
            "mean_trajectory_net_displacement_px": net_displacement,
            "mean_trajectory_straightness": net_displacement / path_length
            if path_length > 0
            else np.nan,
            "mean_trajectory_signed_area_px2": signed_area,
        }
        if "success_rate" in g.columns:
            row["success_rate"] = float(
                pd.to_numeric(g["success_rate"], errors="coerce").mean()
            )
        summary_rows.append(row)
    subject_spatial_trajectory_summary = pd.DataFrame(summary_rows)

    distance_rows: list[dict[str, Any]] = []
    fingers_seen = [
        f
        for f in FINGER_ORDER
        if f in set(subject_xy_trajectory["finger_condition"].dropna())
    ]
    fingers_seen += sorted(
        [
            f
            for f in subject_xy_trajectory["finger_condition"].dropna().unique()
            if f not in FINGER_ORDER
        ]
    )
    for (subject, stiffness), s_df in subject_xy_trajectory.groupby(
        ["subject_id", "stiffness_value"], dropna=False
    ):
        for i, finger_a in enumerate(fingers_seen):
            for finger_b in fingers_seen[i + 1 :]:
                a = s_df[s_df["finger_condition"] == finger_a][
                    ["trajectory_time_bin", "x_centered_px", "y_centered_px"]
                ]
                b = s_df[s_df["finger_condition"] == finger_b][
                    ["trajectory_time_bin", "x_centered_px", "y_centered_px"]
                ]
                merged = a.merge(b, on="trajectory_time_bin", suffixes=("_a", "_b"))
                if merged.empty:
                    continue
                dist = np.hypot(
                    merged["x_centered_px_b"] - merged["x_centered_px_a"],
                    merged["y_centered_px_b"] - merged["y_centered_px_a"],
                )
                distance_rows.append(
                    {
                        "subject_id": subject,
                        "stiffness_value": stiffness,
                        "finger_a": finger_a,
                        "finger_b": finger_b,
                        "comparison": f"{finger_b} - {finger_a}",
                        "n_matched_time_bins": int(len(merged)),
                        "mean_xy_distance_px": float(np.nanmean(dist)),
                        "median_xy_distance_px": float(np.nanmedian(dist)),
                        "rms_xy_distance_px": float(
                            np.sqrt(np.nanmean(np.square(dist)))
                        ),
                        "max_xy_distance_px": float(np.nanmax(dist)),
                    }
                )
    subject_finger_spatial_distance = pd.DataFrame(distance_rows)
    subject_spatial_metric_distribution = summarize_metric_distribution(
        subject_spatial_trajectory_summary,
        ["subject_id", "finger_condition"],
        [
            "mean_trajectory_path_length_px",
            "mean_trajectory_net_displacement_px",
            "mean_trajectory_straightness",
            "spatial_extent_area_px2",
            "mean_radius_px",
            "max_radius_px",
            "mean_xy_sd_radius_px",
        ],
    )

    return {
        "subject_xy_trajectory": subject_xy_trajectory,
        "subject_spatial_trajectory_summary": subject_spatial_trajectory_summary,
        "subject_finger_spatial_distance": subject_finger_spatial_distance,
        "subject_spatial_metric_distribution": subject_spatial_metric_distribution,
    }


def save_subject_xy_trajectory_figures(
    output_root: Path,
    subject_xy_trajectory: pd.DataFrame,
    *,
    max_subjects: Optional[int] = None,
    fig_dpi: int = 160,
) -> list[Path]:
    """Save one spatial XY trajectory figure per subject.

    Each figure keeps the subject separate and overlays that person's finger and
    stiffness trajectories in camera-centered XY coordinates. Circle=start,
    x=end. Thin pale lines are individual stiffness trajectories; thick lines
    are the subject's finger-average trajectory.
    """
    fig_dir = output_root / "figures" / "subject_xy_trajectories"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if subject_xy_trajectory.empty:
        save_csv(
            pd.DataFrame({"figure": []}),
            output_root,
            "subject_xy_trajectory_figure_manifest.csv",
        )
        return paths

    traj = subject_xy_trajectory.copy()
    traj["finger_condition"] = traj["finger_condition"].map(normalize_finger_condition)
    for col in [
        "stiffness_value",
        "trajectory_time_bin",
        "x_centered_px",
        "y_centered_px",
    ]:
        traj[col] = pd.to_numeric(traj[col], errors="coerce")
    subjects = sorted(traj["subject_id"].dropna().unique(), key=lambda x: str(x))
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    xy_limits = _centered_limits_from_columns(
        traj.dropna(subset=["x_centered_px", "y_centered_px"]),
        ["x_centered_px", "y_centered_px"],
    )

    for subject in subjects:
        s = traj[traj["subject_id"] == subject].dropna(
            subset=["x_centered_px", "y_centered_px"]
        )
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        for finger in _ordered_fingers(s["finger_condition"].dropna().unique()):
            f_df = s[s["finger_condition"] == finger]
            color = FINGER_TO_COLOR.get(str(finger), "#777777")
            for stiffness, g in f_df.groupby("stiffness_value", dropna=False):
                g = g.sort_values("trajectory_time_bin")
                ax.plot(
                    g["x_centered_px"],
                    g["y_centered_px"],
                    color=color,
                    alpha=0.22,
                    linewidth=1.0,
                )
                if not g.empty:
                    ax.scatter(
                        g["x_centered_px"].iloc[0],
                        g["y_centered_px"].iloc[0],
                        color=color,
                        alpha=0.25,
                        s=14,
                        marker="o",
                    )
                    ax.scatter(
                        g["x_centered_px"].iloc[-1],
                        g["y_centered_px"].iloc[-1],
                        color=color,
                        alpha=0.25,
                        s=18,
                        marker="x",
                    )
            avg = f_df.groupby("trajectory_time_bin", as_index=False).agg(
                x=("x_centered_px", "mean"), y=("y_centered_px", "mean")
            )
            avg = avg.sort_values("trajectory_time_bin")
            if not avg.empty:
                ax.plot(
                    avg["x"],
                    avg["y"],
                    color=color,
                    linewidth=3.0,
                    label=FINGER_LABELS.get(str(finger), str(finger)),
                )
                ax.scatter(
                    avg["x"].iloc[0],
                    avg["y"].iloc[0],
                    color=color,
                    edgecolor="black",
                    s=45,
                    marker="o",
                    zorder=5,
                )
                ax.scatter(
                    avg["x"].iloc[-1],
                    avg["y"].iloc[-1],
                    color=color,
                    edgecolor="black",
                    s=55,
                    marker="X",
                    zorder=5,
                )
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.axvline(0, color="0.65", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xy_limits)
        ax.set_ylim(*xy_limits)
        ax.set_xlabel("X from workspace center (px)")
        ax.set_ylabel("Y from workspace center (px)")
        ax.set_title(
            f"Subject {subject}: XY trajectories in space\nthin=stiffness-specific, thick=finger average; o=start, X=end"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        out = subject_figure_path(
            fig_dir,
            subject,
            f"subject_{sanitize_name(subject)}_xy_trajectories.png",
        )
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

    plot_data = traj.dropna(subset=["x_centered_px", "y_centered_px"]).copy()
    if not plot_data.empty:
        for scope_name, group_col, filename, title in [
            (
                "finger",
                "finger_condition",
                "all_xy_trajectories_with_finger_average.png",
                "All experiment XY trajectories by finger",
            ),
            (
                "stiffness",
                "stiffness_value",
                "all_xy_trajectories_with_stiffness_average.png",
                "All experiment XY trajectories by stiffness",
            ),
        ]:
            if group_col not in plot_data.columns:
                continue
            fig, ax = plt.subplots(figsize=(7.6, 7.2))
            values = sorted(plot_data[group_col].dropna().unique(), key=lambda x: str(x))
            cmap = plt.get_cmap(STIFFNESS_CMAP)
            for i, value in enumerate(values):
                g_scope = plot_data[plot_data[group_col] == value]
                if g_scope.empty:
                    continue
                color = (
                    FINGER_TO_COLOR.get(str(value), "#777777")
                    if scope_name == "finger"
                    else cmap(i / max(1, len(values) - 1))
                )
                for _, g in g_scope.groupby(
                    ["subject_id", "stiffness_value", "finger_condition"],
                    dropna=False,
                ):
                    g = g.sort_values("trajectory_time_bin")
                    ax.plot(
                        g["x_centered_px"],
                        g["y_centered_px"],
                        color=color,
                        alpha=0.08,
                        linewidth=0.8,
                    )
                avg = (
                    g_scope.groupby("trajectory_time_bin", as_index=False)
                    .agg(x=("x_centered_px", "mean"), y=("y_centered_px", "mean"))
                    .sort_values("trajectory_time_bin")
                )
                if not avg.empty:
                    ax.plot(
                        avg["x"],
                        avg["y"],
                        color=color,
                        linewidth=3.0,
                        label=str(value),
                    )
            ax.axhline(0, color="0.65", linewidth=0.8)
            ax.axvline(0, color="0.65", linewidth=0.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*xy_limits)
            ax.set_ylim(*xy_limits)
            ax.set_xlabel("X from workspace center (px, original analog scale)")
            ax.set_ylabel("Y from workspace center (px, original analog scale)")
            ax.set_title(f"{title}\nthin=participant paths, thick=average path")
            ax.legend(fontsize=8, title=scope_name)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            out = fig_dir / filename
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "subject_xy_trajectory_figure_manifest.csv",
    )
    return paths


def compute_subject_velocity_acceleration_analysis(
    time_bins: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create subject-first velocity and acceleration profiles.

    Uses existing derivative columns computed in ``compute_tracking_kinematics``.
    Rows preserve subject x finger x stiffness x normalized-time-bin, allowing
    per-person inspection before any group-level inference.
    """
    empty = {
        "subject_velocity_acceleration_profile": pd.DataFrame(),
        "subject_velocity_acceleration_summary": pd.DataFrame(),
        "subject_finger_velocity_acceleration_distance": pd.DataFrame(),
        "subject_velocity_acceleration_metric_distribution": pd.DataFrame(),
        "velocity_stiffness_influence_summary": pd.DataFrame(),
        "velocity_finger_influence_summary": pd.DataFrame(),
        "velocity_time_influence_summary": pd.DataFrame(),
    }
    required = {
        "subject_id",
        "finger_condition",
        "stiffness_value",
        "trajectory_time_bin",
    }
    metric_required = {
        "vx_px_s",
        "vy_px_s",
        "vz_px_s",
        "vx_3d_px_s",
        "vy_3d_px_s",
        "vz_3d_proxy_px_s",
        "speed_px_s",
        "ax_px_s2",
        "ay_px_s2",
        "az_px_s2",
        "ax_3d_px_s2",
        "ay_3d_px_s2",
        "az_3d_proxy_px_s2",
        "acceleration_px_s2",
    }
    if (
        time_bins.empty
        or not required.issubset(time_bins.columns)
        or not metric_required.intersection(time_bins.columns)
    ):
        return empty

    tb = add_success_label_column(
        add_protocol_demographic_factors(add_workspace_normalization_columns(time_bins))
    )
    tb["finger_condition"] = tb["finger_condition"].map(normalize_finger_condition)
    numeric_cols = [
        "stiffness_value",
        "trajectory_time_bin",
        "time_fraction",
        "sampling_rate_hz",
        "vx_px_s",
        "vy_px_s",
        "vz_px_s",
        "vx_3d_px_s",
        "vy_3d_px_s",
        "vz_3d_proxy_px_s",
        "speed_px_s",
        "ax_px_s2",
        "ay_px_s2",
        "az_px_s2",
        "ax_3d_px_s2",
        "ay_3d_px_s2",
        "az_3d_proxy_px_s2",
        "acceleration_px_s2",
        "jx_px_s3",
        "jy_px_s3",
        "jerk_px_s3",
        "radial_velocity_px_s",
        "tangential_velocity_px_s",
        "correct_response",
    ]
    for col in numeric_cols:
        if col in tb.columns:
            tb[col] = pd.to_numeric(tb[col], errors="coerce")

    # Fill derivative magnitudes if a caller provides only components.
    if "speed_px_s" not in tb.columns and {"vx_px_s", "vy_px_s"}.issubset(tb.columns):
        tb["speed_px_s"] = np.hypot(tb["vx_px_s"], tb["vy_px_s"])
    if "acceleration_px_s2" not in tb.columns and {"ax_px_s2", "ay_px_s2"}.issubset(
        tb.columns
    ):
        tb["acceleration_px_s2"] = np.hypot(tb["ax_px_s2"], tb["ay_px_s2"])

    group_cols = [
        "subject_id",
        "finger_condition",
        "stiffness_value",
        "trajectory_time_bin",
    ]
    agg_spec: dict[str, tuple[str, str]] = {
        "n_segments": ("tracking_file", "nunique")
        if "tracking_file" in tb.columns
        else ("trajectory_time_bin", "count"),
        "time_fraction": ("time_fraction", "mean")
        if "time_fraction" in tb.columns
        else ("trajectory_time_bin", "mean"),
    }
    if "subject_group" in tb.columns:
        agg_spec["subject_group"] = ("subject_group", "first")
    for col in [
        EXPERIMENT_GROUP_COLUMN,
        "workspace_setup",
        "workspace_label",
        "workspace_width_cm",
        "workspace_height_cm",
        "success_label",
    ]:
        if col in tb.columns:
            agg_spec[col] = (col, "first")
    if "correct_response" in tb.columns:
        agg_spec["success_rate"] = ("correct_response", "mean")
    if "sampling_rate_hz" in tb.columns:
        agg_spec["sampling_rate_hz"] = ("sampling_rate_hz", "median")
    for col in [
        "vx_px_s",
        "vy_px_s",
        "speed_px_s",
        "ax_px_s2",
        "ay_px_s2",
        "acceleration_px_s2",
        "jx_px_s3",
        "jy_px_s3",
        "jerk_px_s3",
        "radial_velocity_px_s",
        "tangential_velocity_px_s",
    ]:
        if col in tb.columns:
            agg_spec[col] = (col, "mean")
            agg_spec[f"sd_{col}"] = (col, "std")

    profile = tb.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    if {"vx_px_s", "vy_px_s"}.issubset(profile.columns):
        profile["velocity_heading_deg"] = np.degrees(
            np.arctan2(profile["vy_px_s"], profile["vx_px_s"])
        )
    if {"ax_px_s2", "ay_px_s2"}.issubset(profile.columns):
        profile["acceleration_heading_deg"] = np.degrees(
            np.arctan2(profile["ay_px_s2"], profile["ax_px_s2"])
        )
    if {"speed_px_s", "acceleration_px_s2"}.issubset(profile.columns):
        profile["speed_acceleration_product"] = (
            profile["speed_px_s"] * profile["acceleration_px_s2"]
        )

    summary_rows: list[dict[str, Any]] = []
    for (subject, finger, stiffness), g in profile.groupby(
        ["subject_id", "finger_condition", "stiffness_value"], dropna=False
    ):
        g = g.sort_values("trajectory_time_bin")
        row: dict[str, Any] = {
            "subject_id": subject,
            "finger_condition": finger,
            "stiffness_value": stiffness,
            "subject_group": g["subject_group"].dropna().iloc[0]
            if "subject_group" in g.columns and g["subject_group"].notna().any()
            else np.nan,
            EXPERIMENT_GROUP_COLUMN: g[EXPERIMENT_GROUP_COLUMN].dropna().iloc[0]
            if EXPERIMENT_GROUP_COLUMN in g.columns
            and g[EXPERIMENT_GROUP_COLUMN].notna().any()
            else np.nan,
            "workspace_setup": g["workspace_setup"].dropna().iloc[0]
            if "workspace_setup" in g.columns and g["workspace_setup"].notna().any()
            else np.nan,
            "workspace_label": g["workspace_label"].dropna().iloc[0]
            if "workspace_label" in g.columns and g["workspace_label"].notna().any()
            else np.nan,
            "n_time_bins": int(len(g)),
            "n_segments_mean": float(
                pd.to_numeric(g["n_segments"], errors="coerce").mean()
            )
            if "n_segments" in g.columns
            else np.nan,
        }
        if "success_rate" in g.columns:
            row["success_rate"] = float(
                pd.to_numeric(g["success_rate"], errors="coerce").mean()
            )
        for col in [
            "vx_px_s",
            "vy_px_s",
            "vz_px_s",
            "vx_3d_px_s",
            "vy_3d_px_s",
            "vz_3d_proxy_px_s",
            "speed_px_s",
            "ax_px_s2",
            "ay_px_s2",
            "az_px_s2",
            "ax_3d_px_s2",
            "ay_3d_px_s2",
            "az_3d_proxy_px_s2",
            "acceleration_px_s2",
            "jx_px_s3",
            "jy_px_s3",
            "jerk_px_s3",
            "radial_velocity_px_s",
            "tangential_velocity_px_s",
        ]:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"mean_{col}"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"peak_abs_{col}"] = (
                float(vals.abs().max()) if vals.notna().any() else np.nan
            )
            row[f"sd_over_time_{col}"] = (
                float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
            )
            if "time_fraction" in g.columns and vals.notna().any():
                idx = vals.abs().idxmax()
                row[f"time_fraction_peak_abs_{col}"] = float(
                    g.loc[idx, "time_fraction"]
                )
        if "speed_px_s" in g.columns:
            speed = pd.to_numeric(g["speed_px_s"], errors="coerce")
            row["early_mean_speed_px_s"] = float(
                speed.iloc[: max(1, len(speed) // 3)].mean()
            )
            row["late_mean_speed_px_s"] = float(
                speed.iloc[-max(1, len(speed) // 3) :].mean()
            )
            row["late_minus_early_speed_px_s"] = (
                row["late_mean_speed_px_s"] - row["early_mean_speed_px_s"]
            )
        if "acceleration_px_s2" in g.columns:
            acc = pd.to_numeric(g["acceleration_px_s2"], errors="coerce")
            row["early_mean_acceleration_px_s2"] = float(
                acc.iloc[: max(1, len(acc) // 3)].mean()
            )
            row["late_mean_acceleration_px_s2"] = float(
                acc.iloc[-max(1, len(acc) // 3) :].mean()
            )
            row["late_minus_early_acceleration_px_s2"] = (
                row["late_mean_acceleration_px_s2"]
                - row["early_mean_acceleration_px_s2"]
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    distance_rows: list[dict[str, Any]] = []
    fingers_seen = [
        f for f in FINGER_ORDER if f in set(profile["finger_condition"].dropna())
    ]
    fingers_seen += sorted(
        [
            f
            for f in profile["finger_condition"].dropna().unique()
            if f not in FINGER_ORDER
        ]
    )
    for (subject, stiffness), s_df in profile.groupby(
        ["subject_id", "stiffness_value"], dropna=False
    ):
        for i, finger_a in enumerate(fingers_seen):
            for finger_b in fingers_seen[i + 1 :]:
                cols = ["trajectory_time_bin"] + [
                    c
                    for c in [
                        "vx_px_s",
                        "vy_px_s",
                        "speed_px_s",
                        "ax_px_s2",
                        "ay_px_s2",
                        "acceleration_px_s2",
                        "jx_px_s3",
                        "jy_px_s3",
                        "jerk_px_s3",
                    ]
                    if c in profile.columns
                ]
                a = s_df[s_df["finger_condition"] == finger_a][cols]
                b = s_df[s_df["finger_condition"] == finger_b][cols]
                merged = a.merge(b, on="trajectory_time_bin", suffixes=("_a", "_b"))
                if merged.empty:
                    continue
                row = {
                    "subject_id": subject,
                    "stiffness_value": stiffness,
                    "finger_a": finger_a,
                    "finger_b": finger_b,
                    "comparison": f"{finger_b} - {finger_a}",
                    "n_matched_time_bins": int(len(merged)),
                }
                if {"vx_px_s_a", "vy_px_s_a", "vx_px_s_b", "vy_px_s_b"}.issubset(
                    merged.columns
                ):
                    vdist = np.hypot(
                        merged["vx_px_s_b"] - merged["vx_px_s_a"],
                        merged["vy_px_s_b"] - merged["vy_px_s_a"],
                    )
                    row["mean_velocity_vector_distance_px_s"] = float(np.nanmean(vdist))
                    row["rms_velocity_vector_distance_px_s"] = float(
                        np.sqrt(np.nanmean(np.square(vdist)))
                    )
                if {"ax_px_s2_a", "ay_px_s2_a", "ax_px_s2_b", "ay_px_s2_b"}.issubset(
                    merged.columns
                ):
                    adist = np.hypot(
                        merged["ax_px_s2_b"] - merged["ax_px_s2_a"],
                        merged["ay_px_s2_b"] - merged["ay_px_s2_a"],
                    )
                    row["mean_acceleration_vector_distance_px_s2"] = float(
                        np.nanmean(adist)
                    )
                    row["rms_acceleration_vector_distance_px_s2"] = float(
                        np.sqrt(np.nanmean(np.square(adist)))
                    )
                if {"speed_px_s_a", "speed_px_s_b"}.issubset(merged.columns):
                    sdiff = merged["speed_px_s_b"] - merged["speed_px_s_a"]
                    row["speed_profile_rmse_px_s"] = float(
                        np.sqrt(np.nanmean(np.square(sdiff)))
                    )
                if {"acceleration_px_s2_a", "acceleration_px_s2_b"}.issubset(
                    merged.columns
                ):
                    accdiff = (
                        merged["acceleration_px_s2_b"] - merged["acceleration_px_s2_a"]
                    )
                    row["acceleration_profile_rmse_px_s2"] = float(
                        np.sqrt(np.nanmean(np.square(accdiff)))
                    )
                distance_rows.append(row)
    distance = pd.DataFrame(distance_rows)
    va_metric_distribution = summarize_metric_distribution(
        summary,
        ["subject_id", "finger_condition"],
        [
            "mean_speed_px_s",
            "peak_abs_speed_px_s",
            "mean_acceleration_px_s2",
            "peak_abs_acceleration_px_s2",
            "mean_jerk_px_s3",
            "peak_abs_jerk_px_s3",
            "late_minus_early_speed_px_s",
            "late_minus_early_acceleration_px_s2",
            "mean_radial_velocity_px_s",
            "mean_tangential_velocity_px_s",
        ],
    )
    velocity_stiffness_influence = (
        profile.groupby(["stiffness_value"], dropna=False)
        .agg(
            n_observations=("speed_px_s", "count"),
            n_subjects=("subject_id", "nunique"),
            mean_velocity_px_s=("speed_px_s", "mean"),
            median_velocity_px_s=("speed_px_s", "median"),
            sem_velocity_px_s=("speed_px_s", _sem),
            mean_acceleration_px_s2=("acceleration_px_s2", "mean"),
        )
        .reset_index()
        if {"speed_px_s", "acceleration_px_s2"}.issubset(profile.columns)
        else pd.DataFrame()
    )
    velocity_finger_influence = (
        profile.groupby(["finger_condition"], dropna=False)
        .agg(
            n_observations=("speed_px_s", "count"),
            n_subjects=("subject_id", "nunique"),
            mean_velocity_px_s=("speed_px_s", "mean"),
            median_velocity_px_s=("speed_px_s", "median"),
            sem_velocity_px_s=("speed_px_s", _sem),
            mean_acceleration_px_s2=("acceleration_px_s2", "mean"),
        )
        .reset_index()
        if {"speed_px_s", "acceleration_px_s2"}.issubset(profile.columns)
        else pd.DataFrame()
    )
    velocity_time_influence = pd.DataFrame()
    if {"speed_px_s", "time_fraction"}.issubset(profile.columns):
        temp = profile.copy()
        temp["time_third"] = pd.cut(
            pd.to_numeric(temp["time_fraction"], errors="coerce"),
            bins=[-0.001, 1 / 3, 2 / 3, 1.001],
            labels=["early", "middle", "late"],
        )
        velocity_time_influence = (
            temp.groupby(["time_third", "stiffness_value"], dropna=False)
            .agg(
                n_observations=("speed_px_s", "count"),
                n_subjects=("subject_id", "nunique"),
                mean_velocity_px_s=("speed_px_s", "mean"),
                median_velocity_px_s=("speed_px_s", "median"),
                mean_acceleration_px_s2=("acceleration_px_s2", "mean")
                if "acceleration_px_s2" in temp.columns
                else ("speed_px_s", "mean"),
            )
            .reset_index()
        )

    return {
        "subject_velocity_acceleration_profile": profile,
        "subject_velocity_acceleration_summary": summary,
        "subject_finger_velocity_acceleration_distance": distance,
        "subject_velocity_acceleration_metric_distribution": va_metric_distribution,
        "velocity_stiffness_influence_summary": velocity_stiffness_influence,
        "velocity_finger_influence_summary": velocity_finger_influence,
        "velocity_time_influence_summary": velocity_time_influence,
    }


def save_subject_velocity_acceleration_figures(
    output_root: Path,
    subject_velocity_acceleration_profile: pd.DataFrame,
    *,
    max_subjects: Optional[int] = None,
    fig_dpi: int = 160,
) -> list[Path]:
    """Save one velocity/acceleration profile figure per subject."""
    fig_dir = output_root / "figures" / "subject_velocity_acceleration"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if subject_velocity_acceleration_profile.empty:
        save_csv(
            pd.DataFrame({"figure": []}),
            output_root,
            "subject_velocity_acceleration_figure_manifest.csv",
        )
        return paths

    profile = subject_velocity_acceleration_profile.copy()
    profile["finger_condition"] = profile["finger_condition"].map(
        normalize_finger_condition
    )
    for col in [
        "stiffness_value",
        "trajectory_time_bin",
        "time_fraction",
        "sampling_rate_hz",
        "speed_px_s",
        "acceleration_px_s2",
        "vx_px_s",
        "vy_px_s",
        "vz_px_s",
        "vx_3d_px_s",
        "vy_3d_px_s",
        "vz_3d_proxy_px_s",
        "ax_px_s2",
        "ay_px_s2",
        "az_px_s2",
        "ax_3d_px_s2",
        "ay_3d_px_s2",
        "az_3d_proxy_px_s2",
    ]:
        if col in profile.columns:
            profile[col] = pd.to_numeric(profile[col], errors="coerce")
    velocity_components = [
        ("Vx", _first_existing_column(profile, ["vx_px_s", "vx_3d_px_s"]), "px/s"),
        ("Vy", _first_existing_column(profile, ["vy_px_s", "vy_3d_px_s"]), "px/s"),
        (
            "Vz",
            _first_existing_column(profile, ["vz_px_s", "vz_3d_proxy_px_s"]),
            "px/s",
        ),
    ]
    acceleration_components = [
        ("Ax", _first_existing_column(profile, ["ax_px_s2", "ax_3d_px_s2"]), "px/s²"),
        ("Ay", _first_existing_column(profile, ["ay_px_s2", "ay_3d_px_s2"]), "px/s²"),
        (
            "Az",
            _first_existing_column(profile, ["az_px_s2", "az_3d_proxy_px_s2"]),
            "px/s²",
        ),
    ]
    subjects = sorted(profile["subject_id"].dropna().unique(), key=lambda x: str(x))
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    for subject in subjects:
        s = profile[profile["subject_id"] == subject].copy()
        if s.empty:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        cmap = plt.get_cmap(STIFFNESS_CMAP)
        stiffness_values = sorted(
            pd.to_numeric(s["stiffness_value"], errors="coerce").dropna().unique()
        )
        stiffness_to_color = {
            stiffness: cmap(i / max(1, len(stiffness_values) - 1))
            for i, stiffness in enumerate(stiffness_values)
        }
        agg_cols = {
            "time_fraction": ("time_fraction", "mean"),
        }
        if "sampling_rate_hz" in s.columns:
            agg_cols["sampling_rate_hz"] = ("sampling_rate_hz", "median")
        for _, col, _ in velocity_components + acceleration_components:
            if col is not None:
                agg_cols[col] = (col, "mean")
        for stiffness, g_stiffness in s.groupby("stiffness_value", dropna=False):
            g_stiffness = g_stiffness.sort_values("trajectory_time_bin")
            avg = (
                g_stiffness.groupby("trajectory_time_bin", as_index=False)
                .agg(**agg_cols)
                .sort_values("trajectory_time_bin")
            )
            color = stiffness_to_color.get(stiffness, "0.35")
            label = f"{float(stiffness):g}" if pd.notna(stiffness) else "missing"
            fs_hz = _estimate_profile_sample_rate_hz(avg)
            cutoff_labels: list[str] = []
            for ax, (component, col, unit) in zip(axes[0], velocity_components):
                if col is None or col not in avg.columns:
                    ax.text(
                        0.5,
                        0.5,
                        "No Z data" if component == "Vz" else "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue
                low, high, effective_cutoff = _profile_filter_pair(
                    avg[col],
                    avg["time_fraction"],
                    fs_hz=fs_hz,
                    cutoff_hz=10.0,
                )
                if np.isfinite(effective_cutoff):
                    cutoff_labels.append(f"{effective_cutoff:g}")
                low = _mask_filter_plot_values(low)
                high = _mask_filter_plot_values(high)
                ax.plot(
                    avg["time_fraction"],
                    low,
                    color=color,
                    linewidth=2.6,
                    label=label,
                )
                ax.plot(
                    avg["time_fraction"],
                    high,
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.75,
                )
                ax.set_ylabel(f"{component} ({unit})")
            for ax, (component, col, unit) in zip(axes[1], acceleration_components):
                if col is None or col not in avg.columns:
                    ax.text(
                        0.5,
                        0.5,
                        "No Z data" if component == "Az" else "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue
                low, high, effective_cutoff = _profile_filter_pair(
                    avg[col],
                    avg["time_fraction"],
                    fs_hz=fs_hz,
                    cutoff_hz=10.0,
                )
                if np.isfinite(effective_cutoff):
                    cutoff_labels.append(f"{effective_cutoff:g}")
                low = _mask_filter_plot_values(low)
                high = _mask_filter_plot_values(high)
                ax.plot(
                    avg["time_fraction"],
                    low,
                    color=color,
                    linewidth=2.6,
                    label=label,
                )
                ax.plot(
                    avg["time_fraction"],
                    high,
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.75,
                )
                ax.set_ylabel(f"{component} ({unit})")
        for ax, (component, _, _) in zip(axes[0], velocity_components):
            ax.set_title(f"{component} vs normalized time")
        for ax, (component, _, _) in zip(axes[1], acceleration_components):
            ax.set_title(f"{component} vs normalized time")
            ax.set_xlabel("Normalized stiffness-segment time")
        for ax in axes.ravel():
            ax.axhline(0, color="0.75", linewidth=0.7)
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8, title="Stiffness")
        cutoff_note = ""
        if cutoff_labels:
            cutoff_note = f"; LPF/HPF target=10 Hz (effective <= {max(cutoff_labels, key=float)} Hz)"
        fig.suptitle(
            f"Subject {subject}: velocity and acceleration components\n"
            "solid=LPF, dashed=HPF; thick lines=subject mean by stiffness, "
            f"averaged across fingers{cutoff_note}; shown |value| range: 1e-1 to 1e4",
            y=1.02,
        )
        fig.tight_layout()
        out = subject_figure_path(
            fig_dir,
            subject,
            f"subject_{sanitize_name(subject)}_velocity_acceleration.png",
        )
        fig.savefig(out, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    all_profile = profile.dropna(subset=["time_fraction"]).copy()
    if not all_profile.empty:
        cmap = plt.get_cmap(STIFFNESS_CMAP)
        stiffness_values = sorted(
            pd.to_numeric(all_profile["stiffness_value"], errors="coerce").dropna().unique()
        )
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for finger in _ordered_fingers(all_profile["finger_condition"].dropna().unique()):
            g_finger = all_profile[all_profile["finger_condition"] == finger]
            color = FINGER_TO_COLOR.get(str(finger), "#777777")
            for _, g_sub in g_finger.groupby("subject_id", dropna=False):
                g_sub = g_sub.sort_values("trajectory_time_bin")
                ax.plot(
                    g_sub["time_fraction"],
                    g_sub["acceleration_px_s2"],
                    color=color,
                    alpha=0.08,
                    linewidth=0.8,
                )
            med = (
                g_finger.groupby("trajectory_time_bin", as_index=False)
                .agg(
                    time_fraction=("time_fraction", "median"),
                    median_acceleration_px_s2=("acceleration_px_s2", "median"),
                )
                .sort_values("trajectory_time_bin")
            )
            if not med.empty:
                ax.plot(
                    med["time_fraction"],
                    med["median_acceleration_px_s2"],
                    color=color,
                    linewidth=3.0,
                    marker="o",
                    markersize=3,
                    label=FINGER_LABELS.get(str(finger), str(finger)),
                )
        ax.set_xlabel("Normalized stiffness-segment time")
        ax.set_ylabel("Acceleration magnitude (px/s²)")
        ax.set_title(
            "Acceleration vs time for all participants (log y-axis)\n"
            "thin=participant traces, thick+markers=median by finger"
        )
        positive_acc = pd.to_numeric(
            all_profile["acceleration_px_s2"], errors="coerce"
        )
        positive_acc = positive_acc[positive_acc > 0]
        if not positive_acc.empty:
            ax.set_yscale("log")
            ax.set_ylim(bottom=float(positive_acc.min()) * 0.8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="Finger")
        fig.tight_layout()
        out = fig_dir / "all_acceleration_vs_time_by_finger_median.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)

        if stiffness_values:
            for stiffness in stiffness_values:
                stiffness_profile = all_profile[
                    pd.to_numeric(all_profile["stiffness_value"], errors="coerce")
                    == stiffness
                ]
                if stiffness_profile.empty:
                    continue
                fig, ax = plt.subplots(figsize=(9.5, 5.2))
                for finger in _ordered_fingers(
                    stiffness_profile["finger_condition"].dropna().unique()
                ):
                    g_finger = stiffness_profile[
                        stiffness_profile["finger_condition"] == finger
                    ]
                    color = FINGER_TO_COLOR.get(str(finger), "#777777")
                    for _, g_sub in g_finger.groupby("subject_id", dropna=False):
                        g_sub = g_sub.sort_values("trajectory_time_bin")
                        ax.plot(
                            g_sub["time_fraction"],
                            g_sub["acceleration_px_s2"],
                            color=color,
                            alpha=0.08,
                            linewidth=0.8,
                        )
                    med = (
                        g_finger.groupby("trajectory_time_bin", as_index=False)
                        .agg(
                            time_fraction=("time_fraction", "median"),
                            median_acceleration_px_s2=(
                                "acceleration_px_s2",
                                "median",
                            ),
                        )
                        .sort_values("trajectory_time_bin")
                    )
                    if not med.empty:
                        ax.plot(
                            med["time_fraction"],
                            med["median_acceleration_px_s2"],
                            color=color,
                            linewidth=3.0,
                            marker="o",
                            markersize=3,
                            label=FINGER_LABELS.get(str(finger), str(finger)),
                        )
                positive_acc = pd.to_numeric(
                    stiffness_profile["acceleration_px_s2"], errors="coerce"
                )
                positive_acc = positive_acc[positive_acc > 0]
                if not positive_acc.empty:
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=float(positive_acc.min()) * 0.8)
                ax.set_xlabel("Normalized stiffness-segment time")
                ax.set_ylabel("Acceleration magnitude (px/s²)")
                ax.set_title(
                    f"Acceleration vs time at stiffness {stiffness:g} (log y-axis)\n"
                    "thin=participant traces, thick+markers=median by finger"
                )
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8, title="Finger")
                fig.tight_layout()
                out = (
                    fig_dir
                    / (
                        "all_acceleration_vs_time_by_finger_median_"
                        f"stiffness_{sanitize_name(stiffness)}.png"
                    )
                )
                fig.savefig(out, dpi=fig_dpi)
                plt.close(fig)
                paths.append(out)

            fig, ax = plt.subplots(figsize=(9.5, 5.2))
            for i, stiffness in enumerate(stiffness_values):
                g = all_profile[
                    pd.to_numeric(all_profile["stiffness_value"], errors="coerce")
                    == stiffness
                ]
                color = cmap(i / max(1, len(stiffness_values) - 1))
                avg = (
                    g.groupby("trajectory_time_bin", as_index=False)
                    .agg(
                        time_fraction=("time_fraction", "mean"),
                        speed_px_s=("speed_px_s", "mean"),
                        sem_speed_px_s=("speed_px_s", _sem),
                    )
                    .sort_values("trajectory_time_bin")
                )
                ax.plot(
                    avg["time_fraction"],
                    avg["speed_px_s"],
                    color=color,
                    linewidth=2.2,
                    label=f"{stiffness:g}",
                )
            ax.set_xlabel("Normalized stiffness-segment time")
            ax.set_ylabel("Velocity / speed (px/s)")
            ax.set_title("Velocity vs time by stiffness value")
            ax.legend(fontsize=8, title="Stiffness")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            out = fig_dir / "all_velocity_vs_time_by_stiffness.png"
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

            for finger in _ordered_fingers(all_profile["finger_condition"].dropna().unique()):
                finger_profile = all_profile[all_profile["finger_condition"] == finger]
                if finger_profile.empty:
                    continue
                fig, ax = plt.subplots(figsize=(9.5, 5.2))
                for i, stiffness in enumerate(stiffness_values):
                    g = finger_profile[
                        pd.to_numeric(finger_profile["stiffness_value"], errors="coerce")
                        == stiffness
                    ]
                    if g.empty:
                        continue
                    color = cmap(i / max(1, len(stiffness_values) - 1))
                    avg = (
                        g.groupby("trajectory_time_bin", as_index=False)
                        .agg(
                            time_fraction=("time_fraction", "mean"),
                            speed_px_s=("speed_px_s", "mean"),
                        )
                        .sort_values("trajectory_time_bin")
                    )
                    ax.plot(
                        avg["time_fraction"],
                        avg["speed_px_s"],
                        color=color,
                        linewidth=2.2,
                        label=f"{stiffness:g}",
                    )
                ax.set_xlabel("Normalized stiffness-segment time")
                ax.set_ylabel("Velocity / speed (px/s)")
                ax.set_title(
                    "Velocity vs time by stiffness value\n"
                    f"Finger: {FINGER_LABELS.get(str(finger), str(finger))}"
                )
                ax.legend(fontsize=8, title="Stiffness")
                ax.grid(alpha=0.25)
                fig.tight_layout()
                out = (
                    fig_dir
                    / f"all_velocity_vs_time_by_stiffness_finger_{sanitize_name(finger)}.png"
                )
                fig.savefig(out, dpi=fig_dpi)
                plt.close(fig)
                paths.append(out)

            by_stiffness = (
                all_profile.groupby("stiffness_value", as_index=False)
                .agg(
                    mean_speed_px_s=("speed_px_s", "mean"),
                    median_speed_px_s=("speed_px_s", "median"),
                    sem_speed_px_s=("speed_px_s", _sem),
                    n_observations=("speed_px_s", "count"),
                )
                .sort_values("stiffness_value")
            )
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            colors = [
                cmap(i / max(1, len(by_stiffness) - 1)) for i in range(len(by_stiffness))
            ]
            ax.bar(
                by_stiffness["stiffness_value"].astype(str),
                by_stiffness["mean_speed_px_s"],
                yerr=by_stiffness["sem_speed_px_s"],
                color=colors,
                edgecolor=colors,
                linewidth=2,
                capsize=3,
                alpha=0.82,
            )
            for i, row in by_stiffness.reset_index(drop=True).iterrows():
                ax.text(
                    i,
                    row["mean_speed_px_s"],
                    f"{row['stiffness_value']:g}\nn={int(row['n_observations'])}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_xlabel("Stiffness value")
            ax.set_ylabel("Average velocity / speed (px/s)")
            ax.set_title("Influence of stiffness on average velocity")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            out = fig_dir / "average_velocity_vs_stiffness.png"
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

            fingers = _ordered_fingers(all_profile["finger_condition"].dropna().unique())
            if fingers:
                fig, axes = plt.subplots(
                    1,
                    len(fingers),
                    figsize=(4.6 * len(fingers), 4.8),
                    squeeze=False,
                    sharey=True,
                )
                subject_colors = _figure_colors(
                    sorted(all_profile["subject_id"].dropna().unique(), key=lambda x: str(x))
                )
                for ax, finger in zip(axes.ravel(), fingers):
                    finger_profile = all_profile[all_profile["finger_condition"] == finger]
                    subject_finger = (
                        finger_profile.groupby(
                            ["subject_id", "stiffness_value"], as_index=False, dropna=False
                        )
                        .agg(mean_speed_px_s=("speed_px_s", "mean"))
                        .sort_values("stiffness_value")
                    )
                    for subject, g_sub in subject_finger.groupby(
                        "subject_id", dropna=False
                    ):
                        ax.plot(
                            g_sub["stiffness_value"],
                            g_sub["mean_speed_px_s"],
                            color=subject_colors.get(subject, "0.45"),
                            linewidth=1.2,
                            alpha=0.45,
                            marker="o",
                            markersize=3,
                        )
                    group_mean = (
                        subject_finger.groupby("stiffness_value", as_index=False)
                        .agg(mean_speed_px_s=("mean_speed_px_s", "mean"))
                        .sort_values("stiffness_value")
                    )
                    ax.plot(
                        group_mean["stiffness_value"],
                        group_mean["mean_speed_px_s"],
                        color=FINGER_TO_COLOR.get(str(finger), "black"),
                        linewidth=3.0,
                        marker="o",
                        label="Group mean",
                    )
                    ax.set_title(FINGER_LABELS.get(str(finger), str(finger)))
                    ax.set_xlabel("Stiffness value")
                    ax.grid(alpha=0.25)
                axes[0, 0].set_ylabel("Average velocity / speed (px/s)")
                axes[0, -1].legend(fontsize=8)
                fig.suptitle(
                    "Average velocity vs stiffness per finger and per subject\n"
                    "thin=subject traces, thick=mean across subjects"
                )
                fig.tight_layout()
                out = fig_dir / "average_velocity_vs_stiffness_by_finger_subject.png"
                fig.savefig(out, dpi=fig_dpi)
                plt.close(fig)
                paths.append(out)

    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "subject_velocity_acceleration_figure_manifest.csv",
    )
    return paths


def compute_3d_proxy_kinematics(
    samples: pd.DataFrame, side_z_samples: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Merge top-camera XY with side-camera Z/lift and compute 3D proxy features.

    The current experiment does not store calibrated synchronized 3D marker
    coordinates. This function therefore builds an explicit **3D proxy**:
    top-camera centered X/Y plus interpolated side-camera Z/lift within each
    subject x trial x stiffness segment. Outputs remain per segment and per
    subject so the subject structure is not lost.
    """
    empty = {
        "kinematic_3d_proxy_samples": pd.DataFrame(),
        "trial_3d_kinematic_summary": pd.DataFrame(),
        "subject_3d_kinematic_summary": pd.DataFrame(),
        "subject_3d_metric_distribution": pd.DataFrame(),
        "stiffness_direction_3d_metric_distribution": pd.DataFrame(),
        "mixedlm_model_input": pd.DataFrame(),
    }
    required = {
        "subject_id",
        "trial_index_raw",
        "stiffness_segment_id",
        "time_s",
        "stiffness_time_fraction",
        "x_centered_px",
        "y_centered_px",
    }
    if (
        samples.empty
        or side_z_samples.empty
        or not required.issubset(samples.columns)
        or "side_z_lift_px" not in side_z_samples.columns
    ):
        return empty

    s = samples.copy()
    s["finger_condition"] = s.get("finger_condition", np.nan).map(
        normalize_finger_condition
    )
    side = side_z_samples.copy()
    if "finger_condition" in side.columns:
        side["finger_condition"] = side["finger_condition"].map(
            normalize_finger_condition
        )
    for df_, cols in [
        (
            s,
            [
                "trial_index_raw",
                "stiffness_segment_id",
                "stiffness_value",
                "time_s",
                "stiffness_time_fraction",
                "x_centered_px",
                "y_centered_px",
                "x_lpf_px",
                "y_lpf_px",
            ],
        ),
        (
            side,
            [
                "trial_index_raw",
                "stiffness_segment_id",
                "stiffness_value",
                "side_time_fraction",
                "side_z_lift_px",
                "side_x_from_center_camera_corrected_px",
            ],
        ),
    ]:
        for col in cols:
            if col in df_.columns:
                df_[col] = pd.to_numeric(df_[col], errors="coerce")

    rows: list[pd.DataFrame] = []
    keys = ["subject_id", "trial_index_raw", "stiffness_segment_id"]
    side = add_side_camera_angle_normalization_columns(side)
    side_groups = {
        k: g.sort_values("side_time_fraction")
        for k, g in side.dropna(
            subset=["side_time_fraction", "side_z_lift_px"]
        ).groupby(keys, dropna=False)
    }
    for key, g in s.groupby(keys, dropna=False):
        g = g.sort_values("time_s").copy()
        z_group = side_groups.get(key)
        if z_group is not None and not z_group.empty:
            x_old = pd.to_numeric(
                z_group["side_time_fraction"], errors="coerce"
            ).to_numpy(dtype=float)
            z_old = pd.to_numeric(z_group["side_z_lift_px"], errors="coerce").to_numpy(
                dtype=float
            )
            lateral_old = (
                pd.to_numeric(
                    z_group.get("side_x_from_center_camera_corrected_px", np.nan),
                    errors="coerce",
                ).to_numpy(dtype=float)
                if "side_x_from_center_camera_corrected_px" in z_group.columns
                else np.full_like(z_old, np.nan, dtype=float)
            )
            valid = np.isfinite(x_old) & np.isfinite(z_old)
            x_old = x_old[valid]
            z_old = z_old[valid]
            lateral_old = lateral_old[valid]
            if x_old.size == 1:
                g["z_lift_px"] = z_old[0]
                g["side_lateral_camera_corrected_px"] = lateral_old[0]
            elif x_old.size > 1:
                order = np.argsort(x_old)
                x_old = x_old[order]
                z_old = z_old[order]
                lateral_old = lateral_old[order]
                # Remove duplicated side fractions for stable interpolation.
                unique_x, unique_idx = np.unique(x_old, return_index=True)
                unique_z = z_old[unique_idx]
                unique_lateral = lateral_old[unique_idx]
                sample_fraction = (
                    pd.to_numeric(g["stiffness_time_fraction"], errors="coerce")
                    .fillna(0)
                    .clip(0, 1)
                )
                g["z_lift_px"] = np.interp(
                    sample_fraction,
                    unique_x,
                    unique_z,
                )
                g["side_lateral_camera_corrected_px"] = np.interp(
                    sample_fraction,
                    unique_x,
                    unique_lateral,
                )
            else:
                g["z_lift_px"] = np.nan
                g["side_lateral_camera_corrected_px"] = np.nan
        else:
            g["z_lift_px"] = np.nan
            g["side_lateral_camera_corrected_px"] = np.nan
        x_col = (
            "x_lpf_px"
            if "x_lpf_px" in g.columns and g["x_lpf_px"].notna().any()
            else "x_centered_px"
        )
        y_col = (
            "y_lpf_px"
            if "y_lpf_px" in g.columns and g["y_lpf_px"].notna().any()
            else "y_centered_px"
        )
        g["x_3d_px"] = pd.to_numeric(g[x_col], errors="coerce")
        g["y_3d_px"] = pd.to_numeric(g[y_col], errors="coerce")
        g["z_3d_proxy_px"] = pd.to_numeric(g["z_lift_px"], errors="coerce")
        g["side_lateral_camera_corrected_px"] = pd.to_numeric(
            g["side_lateral_camera_corrected_px"], errors="coerce"
        )
        g["side_lift_lateral_angle_camera_corrected_deg"] = np.degrees(
            np.arctan2(g["z_3d_proxy_px"], g["side_lateral_camera_corrected_px"])
        )
        if {"thumb_active_dx_px", "thumb_active_dy_px"}.issubset(g.columns):
            g["hand_orientation_xy_deg"] = np.degrees(
                np.arctan2(g["thumb_active_dy_px"], g["thumb_active_dx_px"])
            )
            g["hand_orientation_yz_deg"] = np.degrees(
                np.arctan2(g["z_3d_proxy_px"], g["thumb_active_dy_px"])
            )
            g["hand_orientation_zx_deg"] = np.degrees(
                np.arctan2(g["thumb_active_dx_px"], g["z_3d_proxy_px"])
            )
        g["r_3d_from_center_px"] = np.sqrt(
            g["x_3d_px"] ** 2 + g["y_3d_px"] ** 2 + g["z_3d_proxy_px"] ** 2
        )
        dt = pd.to_numeric(g["time_s"], errors="coerce").diff().replace(0, np.nan)
        g["vx_3d_px_s"] = g["x_3d_px"].diff() / dt
        g["vy_3d_px_s"] = g["y_3d_px"].diff() / dt
        g["vz_3d_proxy_px_s"] = g["z_3d_proxy_px"].diff() / dt
        g["speed_3d_proxy_px_s"] = np.sqrt(
            g["vx_3d_px_s"] ** 2 + g["vy_3d_px_s"] ** 2 + g["vz_3d_proxy_px_s"] ** 2
        )
        g["ax_3d_px_s2"] = g["vx_3d_px_s"].diff() / dt
        g["ay_3d_px_s2"] = g["vy_3d_px_s"].diff() / dt
        g["az_3d_proxy_px_s2"] = g["vz_3d_proxy_px_s"].diff() / dt
        g["acceleration_3d_proxy_px_s2"] = np.sqrt(
            g["ax_3d_px_s2"] ** 2 + g["ay_3d_px_s2"] ** 2 + g["az_3d_proxy_px_s2"] ** 2
        )
        rows.append(g)

    samples_3d = (
        pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    )
    summary_rows: list[dict[str, Any]] = []
    for key, g in samples_3d.groupby(keys, dropna=False):
        g = g.sort_values("time_s")
        valid = g[["x_3d_px", "y_3d_px", "z_3d_proxy_px"]].dropna()
        if valid.empty:
            continue
        dx = g["x_3d_px"].diff()
        dy = g["y_3d_px"].diff()
        dz = g["z_3d_proxy_px"].diff()
        dside = g["side_lateral_camera_corrected_px"].diff()
        step_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        side_view_step = np.sqrt(dside**2 + dz**2)
        x0 = valid["x_3d_px"].iloc[0]
        y0 = valid["y_3d_px"].iloc[0]
        z0 = valid["z_3d_proxy_px"].iloc[0]
        excursion_from_start = np.sqrt(
            (g["x_3d_px"] - x0) ** 2
            + (g["y_3d_px"] - y0) ** 2
            + (g["z_3d_proxy_px"] - z0) ** 2
        )
        row = {
            "subject_id": g["subject_id"].iloc[0],
            "subject_group": g["subject_group"].dropna().iloc[0]
            if "subject_group" in g.columns and g["subject_group"].notna().any()
            else np.nan,
            "trial_index_raw": g["trial_index_raw"].iloc[0],
            "pair_number": g["pair_number"].iloc[0]
            if "pair_number" in g.columns
            else np.nan,
            "finger_condition": g["finger_condition"].dropna().iloc[0]
            if "finger_condition" in g.columns and g["finger_condition"].notna().any()
            else np.nan,
            "stiffness_value": float(
                pd.to_numeric(g["stiffness_value"], errors="coerce").dropna().median()
            )
            if "stiffness_value" in g.columns and g["stiffness_value"].notna().any()
            else np.nan,
            "stiffness_segment_id": g["stiffness_segment_id"].iloc[0],
            "correct_response": float(
                pd.to_numeric(g["correct_response"], errors="coerce").dropna().iloc[0]
            )
            if "correct_response" in g.columns
            and pd.to_numeric(g["correct_response"], errors="coerce").notna().any()
            else np.nan,
            "dominant_movement_direction": g["movement_direction"]
            .dropna()
            .mode()
            .iloc[0]
            if "movement_direction" in g.columns
            and g["movement_direction"].notna().any()
            else np.nan,
            "n_3d_samples": int(valid.shape[0]),
            "path_length_3d_proxy_px": float(np.nansum(step_3d)),
            "max_excursion_3d_from_start_px": float(np.nanmax(excursion_from_start)),
            "max_radius_3d_from_center_px": float(
                pd.to_numeric(g["r_3d_from_center_px"], errors="coerce").max()
            ),
            "peak_velocity_3d_proxy_px_s": float(
                pd.to_numeric(g["speed_3d_proxy_px_s"], errors="coerce").max()
            ),
            "mean_velocity_3d_proxy_px_s": float(
                pd.to_numeric(g["speed_3d_proxy_px_s"], errors="coerce").mean()
            ),
            "peak_acceleration_3d_proxy_px_s2": float(
                pd.to_numeric(g["acceleration_3d_proxy_px_s2"], errors="coerce").max()
            ),
            "mean_acceleration_3d_proxy_px_s2": float(
                pd.to_numeric(g["acceleration_3d_proxy_px_s2"], errors="coerce").mean()
            ),
            "mean_z_lift_px": float(
                pd.to_numeric(g["z_3d_proxy_px"], errors="coerce").mean()
            ),
            "max_z_lift_px": float(
                pd.to_numeric(g["z_3d_proxy_px"], errors="coerce").max()
            ),
            "mean_side_lateral_camera_corrected_px": float(
                pd.to_numeric(
                    g["side_lateral_camera_corrected_px"], errors="coerce"
                ).mean()
            ),
            "path_length_side_view_camera_corrected_px": float(
                np.nansum(side_view_step)
            ),
            "mean_side_lift_lateral_angle_camera_corrected_deg": _circ_mean_deg(
                g["side_lift_lateral_angle_camera_corrected_deg"]
            ),
            "mean_hand_orientation_xy_deg": _circ_mean_deg(g["hand_orientation_xy_deg"])
            if "hand_orientation_xy_deg" in g.columns
            else np.nan,
            "mean_hand_orientation_yz_deg": _circ_mean_deg(g["hand_orientation_yz_deg"])
            if "hand_orientation_yz_deg" in g.columns
            else np.nan,
            "mean_hand_orientation_zx_deg": _circ_mean_deg(g["hand_orientation_zx_deg"])
            if "hand_orientation_zx_deg" in g.columns
            else np.nan,
        }
        row["straightness_3d_proxy"] = (
            row["max_excursion_3d_from_start_px"] / row["path_length_3d_proxy_px"]
            if row["path_length_3d_proxy_px"] > 0
            else np.nan
        )
        summary_rows.append(row)
    trial_3d = pd.DataFrame(summary_rows)

    if not trial_3d.empty:
        subject_3d = (
            trial_3d.groupby(
                ["subject_id", "subject_group", "finger_condition", "stiffness_value"],
                dropna=False,
            )
            .agg(
                n_trials=("trial_index_raw", "count"),
                success_rate=("correct_response", "mean"),
                path_length_3d_proxy_px=("path_length_3d_proxy_px", "mean"),
                max_excursion_3d_from_start_px=(
                    "max_excursion_3d_from_start_px",
                    "mean",
                ),
                max_radius_3d_from_center_px=("max_radius_3d_from_center_px", "mean"),
                peak_velocity_3d_proxy_px_s=("peak_velocity_3d_proxy_px_s", "mean"),
                peak_acceleration_3d_proxy_px_s2=(
                    "peak_acceleration_3d_proxy_px_s2",
                    "mean",
                ),
                mean_z_lift_px=("mean_z_lift_px", "mean"),
                max_z_lift_px=("max_z_lift_px", "mean"),
                mean_side_lateral_camera_corrected_px=(
                    "mean_side_lateral_camera_corrected_px",
                    "mean",
                ),
                path_length_side_view_camera_corrected_px=(
                    "path_length_side_view_camera_corrected_px",
                    "mean",
                ),
                mean_side_lift_lateral_angle_camera_corrected_deg=(
                    "mean_side_lift_lateral_angle_camera_corrected_deg",
                    lambda s: _circ_mean_deg(s),
                ),
                mean_hand_orientation_xy_deg=(
                    "mean_hand_orientation_xy_deg",
                    lambda s: _circ_mean_deg(s),
                ),
                mean_hand_orientation_yz_deg=(
                    "mean_hand_orientation_yz_deg",
                    lambda s: _circ_mean_deg(s),
                ),
                mean_hand_orientation_zx_deg=(
                    "mean_hand_orientation_zx_deg",
                    lambda s: _circ_mean_deg(s),
                ),
                straightness_3d_proxy=("straightness_3d_proxy", "mean"),
            )
            .reset_index()
        )
    else:
        subject_3d = pd.DataFrame()

    model_input = trial_3d.copy()
    if not model_input.empty:
        model_input["direction_factor"] = model_input.get(
            "target_direction", model_input.get("dominant_movement_direction", np.nan)
        )
    subject_3d_metric_distribution = summarize_metric_distribution(
        subject_3d,
        ["subject_id", "finger_condition"],
        [
            "path_length_3d_proxy_px",
            "max_excursion_3d_from_start_px",
            "max_radius_3d_from_center_px",
            "peak_velocity_3d_proxy_px_s",
            "peak_acceleration_3d_proxy_px_s2",
            "mean_z_lift_px",
            "max_z_lift_px",
            "mean_side_lateral_camera_corrected_px",
            "path_length_side_view_camera_corrected_px",
            "mean_side_lift_lateral_angle_camera_corrected_deg",
            "mean_hand_orientation_xy_deg",
            "mean_hand_orientation_yz_deg",
            "mean_hand_orientation_zx_deg",
            "straightness_3d_proxy",
        ],
    )
    stiffness_direction_3d_metric_distribution = summarize_metric_distribution(
        trial_3d,
        ["stiffness_value", "dominant_movement_direction"],
        [
            "path_length_3d_proxy_px",
            "max_excursion_3d_from_start_px",
            "max_radius_3d_from_center_px",
            "peak_velocity_3d_proxy_px_s",
            "peak_acceleration_3d_proxy_px_s2",
            "path_length_side_view_camera_corrected_px",
            "mean_side_lift_lateral_angle_camera_corrected_deg",
        ],
    )
    return {
        "kinematic_3d_proxy_samples": samples_3d,
        "trial_3d_kinematic_summary": trial_3d,
        "subject_3d_kinematic_summary": subject_3d,
        "subject_3d_metric_distribution": subject_3d_metric_distribution,
        "stiffness_direction_3d_metric_distribution": stiffness_direction_3d_metric_distribution,
        "mixedlm_model_input": model_input,
    }


def fit_optional_mixed_effects_models(
    model_input: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Fit optional LMMs when statsmodels is available.

    Formula: metric ~ stiffness_value * C(direction_factor) + (1 | subject_id).
    If statsmodels is unavailable or the data are insufficient, returns a status
    table explaining why instead of failing the notebook.
    """
    metrics = [
        "max_excursion_3d_from_start_px",
        "max_radius_3d_from_center_px",
        "path_length_3d_proxy_px",
        "peak_velocity_3d_proxy_px_s",
    ]
    status_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    if model_input.empty:
        return {
            "mixedlm_status": pd.DataFrame(
                [{"status": "skipped", "reason": "empty model input"}]
            ),
            "mixedlm_coefficients": pd.DataFrame(),
        }
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as exc:
        return {
            "mixedlm_status": pd.DataFrame(
                [{"status": "skipped", "reason": f"statsmodels unavailable: {exc}"}]
            ),
            "mixedlm_coefficients": pd.DataFrame(),
        }

    df = model_input.copy()
    df["stiffness_value"] = pd.to_numeric(df.get("stiffness_value"), errors="coerce")
    if "direction_factor" not in df.columns or df["direction_factor"].isna().all():
        df["direction_factor"] = "unknown"
    df["direction_factor"] = df["direction_factor"].astype(str).fillna("unknown")
    for metric in metrics:
        if metric not in df.columns:
            status_rows.append(
                {"metric": metric, "status": "skipped", "reason": "missing metric"}
            )
            continue
        d = df[["subject_id", "stiffness_value", "direction_factor", metric]].dropna()
        if (
            d["subject_id"].nunique() < 2
            or len(d) < 8
            or d["direction_factor"].nunique() < 2
        ):
            status_rows.append(
                {
                    "metric": metric,
                    "status": "skipped",
                    "reason": "insufficient subjects/rows/direction levels",
                    "n_rows": len(d),
                    "n_subjects": d["subject_id"].nunique(),
                    "n_directions": d["direction_factor"].nunique(),
                }
            )
            continue
        try:
            fit = smf.mixedlm(
                f"{metric} ~ stiffness_value * C(direction_factor)",
                d,
                groups=d["subject_id"],
            ).fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            status_rows.append(
                {
                    "metric": metric,
                    "status": "fit",
                    "reason": "",
                    "n_rows": len(d),
                    "n_subjects": d["subject_id"].nunique(),
                    "n_directions": d["direction_factor"].nunique(),
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                }
            )
            for term, value in fit.params.items():
                coef_rows.append(
                    {
                        "metric": metric,
                        "term": term,
                        "estimate": float(value),
                        "std_error": float(fit.bse.get(term, np.nan)),
                        "p_value": float(fit.pvalues.get(term, np.nan)),
                    }
                )
        except Exception as exc:
            status_rows.append(
                {
                    "metric": metric,
                    "status": "failed",
                    "reason": str(exc),
                    "n_rows": len(d),
                    "n_subjects": d["subject_id"].nunique(),
                    "n_directions": d["direction_factor"].nunique(),
                }
            )
    return {
        "mixedlm_status": pd.DataFrame(status_rows),
        "mixedlm_coefficients": pd.DataFrame(coef_rows),
    }


def save_3d_proxy_figures(
    output_root: Path,
    samples_3d: pd.DataFrame,
    trial_3d_summary: pd.DataFrame,
    *,
    max_subjects: Optional[int] = None,
    fig_dpi: int = 160,
) -> list[Path]:
    """Save per-subject 3D proxy trajectory figures and stiffness-bin plots."""
    fig_dir = output_root / "figures" / "subject_3d_proxy_trajectories"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not samples_3d.empty:
        subjects = sorted(
            samples_3d["subject_id"].dropna().unique(), key=lambda x: str(x)
        )
        if max_subjects is not None:
            subjects = subjects[:max_subjects]
        for subject in subjects:
            s = samples_3d[samples_3d["subject_id"] == subject].dropna(
                subset=["x_3d_px", "y_3d_px", "z_3d_proxy_px"]
            )
            if s.empty:
                continue
            stiffness_vals = pd.to_numeric(s["stiffness_value"], errors="coerce")
            finite_stiffness = stiffness_vals.dropna()
            vmin = float(finite_stiffness.min()) if not finite_stiffness.empty else 0.0
            vmax = float(finite_stiffness.max()) if not finite_stiffness.empty else 1.0
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            cmap = plt.get_cmap("viridis")
            coord_limits = _padded_limits_from_columns(
                s, ["x_3d_px", "y_3d_px", "z_3d_proxy_px"]
            )
            fig = plt.figure(figsize=(18, 10))
            grid = fig.add_gridspec(2, 3)
            ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
            ax_xy = fig.add_subplot(grid[0, 1])
            ax_z = fig.add_subplot(grid[0, 2])
            ax_xz = fig.add_subplot(grid[1, 0])
            ax_yz = fig.add_subplot(grid[1, 1])
            ax_key = fig.add_subplot(grid[1, 2])

            for (_, _, segment), g in s.groupby(
                ["finger_condition", "stiffness_value", "stiffness_segment_id"],
                dropna=False,
            ):
                g = g.sort_values("time_s")
                stiff = pd.to_numeric(g["stiffness_value"], errors="coerce").dropna()
                norm = (
                    0.5
                    if stiff.empty
                    else (float(stiff.iloc[0]) - vmin) / (vmax - vmin)
                )
                color = cmap(norm)
                ax_3d.plot(
                    g["x_3d_px"],
                    g["y_3d_px"],
                    g["z_3d_proxy_px"],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                )
                ax_xy.plot(
                    g["x_3d_px"],
                    g["y_3d_px"],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                )
                ax_xz.plot(
                    g["x_3d_px"],
                    g["z_3d_proxy_px"],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                )
                ax_yz.plot(
                    g["y_3d_px"],
                    g["z_3d_proxy_px"],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                )
                ax_z.plot(
                    g["stiffness_time_fraction"],
                    g["z_3d_proxy_px"],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                )

            ax_3d.set_xlim(*coord_limits)
            ax_3d.set_ylim(*coord_limits)
            ax_3d.set_zlim(*coord_limits)
            ax_3d.set_box_aspect((1, 1, 1))
            ax_3d.set_xlabel("X centered (px)")
            ax_3d.set_ylabel("Y centered (px)")
            ax_3d.set_zlabel("Z/lift proxy (px)")
            ax_3d.set_title("3D view")

            plane_specs = [
                (ax_xy, "XY plane", "X centered (px)", "Y centered (px)"),
                (ax_xz, "XZ plane", "X centered (px)", "Z/lift proxy (px)"),
                (ax_yz, "YZ plane", "Y centered (px)", "Z/lift proxy (px)"),
            ]
            for plane_ax, title, x_label, y_label in plane_specs:
                plane_ax.axhline(0, color="0.75", linewidth=0.8)
                plane_ax.axvline(0, color="0.75", linewidth=0.8)
                plane_ax.set_xlim(*coord_limits)
                plane_ax.set_ylim(*coord_limits)
                plane_ax.set_aspect("equal", adjustable="box")
                plane_ax.set_xlabel(x_label)
                plane_ax.set_ylabel(y_label)
                plane_ax.set_title(title)
                plane_ax.grid(alpha=0.2)
            ax_z.set_xlim(0, 1)
            ax_z.set_ylim(*coord_limits)
            ax_z.set_xlabel("Normalized stiffness-segment time")
            ax_z.set_ylabel("Z/lift proxy (px)")
            ax_z.set_title("Z/lift over time")
            ax_z.grid(alpha=0.2)

            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            fig.colorbar(sm, cax=ax_key, label="Stiffness")
            fig.suptitle(
                f"Subject {subject}: 3D proxy trajectories\n"
                "shared min/max scale for X, Y, and Z/lift; color=stiffness",
                y=0.99,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out = subject_figure_path(
                fig_dir,
                subject,
                f"subject_{sanitize_name(subject)}_3d_proxy_trajectories.png",
            )
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

    if not trial_3d_summary.empty and {
        "dominant_movement_direction",
        "stiffness_value",
        "max_excursion_3d_from_start_px",
    }.issubset(trial_3d_summary.columns):
        d = trial_3d_summary.dropna(
            subset=["stiffness_value", "max_excursion_3d_from_start_px"]
        ).copy()
        if not d.empty:
            d["stiffness_bin"] = pd.qcut(
                d["stiffness_value"].rank(method="first"),
                q=min(3, d["stiffness_value"].nunique()),
                labels=["Low", "Medium", "High"][
                    : min(3, d["stiffness_value"].nunique())
                ],
            )
            fig, ax = plt.subplots(figsize=(11, 5))
            directions = [
                x
                for x in DIRECTION_LABELS
                if x in set(d["dominant_movement_direction"].dropna())
            ]
            directions += sorted(
                [
                    x
                    for x in d["dominant_movement_direction"].dropna().unique()
                    if x not in DIRECTION_LABELS
                ]
            )
            bins = [
                b
                for b in ["Low", "Medium", "High"]
                if b in set(d["stiffness_bin"].astype(str))
            ]
            width = 0.25
            x = np.arange(len(directions))
            rng = np.random.default_rng(20260513)
            for i, b in enumerate(bins):
                sub = d[d["stiffness_bin"].astype(str) == b]
                offset = (i - (len(bins) - 1) / 2) * width
                grouped = sub.groupby("dominant_movement_direction")[
                    "max_excursion_3d_from_start_px"
                ]
                med = grouped.median().reindex(directions)
                ci_low = grouped.apply(lambda s: _ci95_median_bootstrap(s)[0]).reindex(
                    directions
                )
                ci_high = grouped.apply(lambda s: _ci95_median_bootstrap(s)[1]).reindex(
                    directions
                )
                yerr = np.vstack(
                    [(med - ci_low).clip(lower=0), (ci_high - med).clip(lower=0)]
                )
                ax.errorbar(
                    x + offset,
                    med,
                    yerr=yerr,
                    fmt="o",
                    capsize=3,
                    label=f"{b} median + 95% CI",
                )
                for j, direction in enumerate(directions):
                    vals = pd.to_numeric(
                        sub.loc[
                            sub["dominant_movement_direction"] == direction,
                            "max_excursion_3d_from_start_px",
                        ],
                        errors="coerce",
                    ).dropna()
                    if vals.empty:
                        continue
                    jitter = rng.normal(0, width * 0.12, size=len(vals))
                    ax.scatter(
                        np.full(len(vals), x[j] + offset) + jitter,
                        vals,
                        s=12,
                        alpha=0.20,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(directions, rotation=45, ha="right")
            ax.set_ylabel("Max 3D excursion from start (px)")
            ax.set_xlabel("Movement direction proxy")
            ax.set_title(
                "3D far movement by direction and stiffness bin\npoints=trials, markers=median, bars=95% bootstrap CI"
            )
            ax.legend(title="Stiffness")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            out = (
                output_root
                / "figures"
                / "max_3d_excursion_by_direction_stiffness_bin.png"
            )
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "proxy_3d_figure_manifest.csv",
    )
    return paths


HAND_ORIENTATION_PLANE_COLUMNS = [
    "hand_orientation_xy_deg",
    "hand_orientation_yz_deg",
    "hand_orientation_zx_deg",
]


def compute_hand_orientation_plane_analysis(
    trial_kinematic_summary: pd.DataFrame, side_z_trial_summary: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Estimate hand/finger orientation in XY, YZ, and ZX planes.

    XY comes from the top-camera thumb-to-active-finger vector. YZ and ZX use
    the same top-camera vector plus the side-camera Z/lift proxy. Angles remain
    in camera-pixel/proxy units; workspace-normalized XY columns are retained
    for grouping labels but not used to rescale the analog orientation values.
    """

    empty = {
        "hand_orientation_plane_trials": pd.DataFrame(),
        "hand_orientation_plane_summary": pd.DataFrame(),
    }
    if trial_kinematic_summary.empty:
        return empty

    trials = add_success_label_column(
        add_protocol_demographic_factors(
            add_workspace_normalization_columns(trial_kinematic_summary)
        )
    ).copy()
    required = {"mean_hand_orientation_xy_deg", "mean_thumb_active_span_px"}
    if not required.issubset(trials.columns):
        return empty

    if not side_z_trial_summary.empty:
        side_cols = [
            c
            for c in [
                "subject_id",
                "trial_index_raw",
                "stiffness_segment_id",
                "mean_side_z_lift_px",
                "max_side_z_lift_px",
                "side_detection_rate",
            ]
            if c in side_z_trial_summary.columns
        ]
        join_keys = [
            c
            for c in ["subject_id", "trial_index_raw", "stiffness_segment_id"]
            if c in trials.columns and c in side_z_trial_summary.columns
        ]
        if join_keys and "mean_side_z_lift_px" in side_cols:
            trials = trials.merge(
                side_z_trial_summary[side_cols].drop_duplicates(join_keys),
                on=join_keys,
                how="left",
                suffixes=("", "_side"),
            )
    if "mean_side_z_lift_px" not in trials.columns:
        trials["mean_side_z_lift_px"] = np.nan

    xy_rad = np.deg2rad(
        pd.to_numeric(trials["mean_hand_orientation_xy_deg"], errors="coerce")
    )
    span = pd.to_numeric(trials["mean_thumb_active_span_px"], errors="coerce")
    z_proxy = pd.to_numeric(trials["mean_side_z_lift_px"], errors="coerce")
    trials["hand_orientation_dx_px"] = span * np.cos(xy_rad)
    trials["hand_orientation_dy_px"] = span * np.sin(xy_rad)
    trials["hand_orientation_xy_deg"] = trials["mean_hand_orientation_xy_deg"]
    trials["hand_orientation_yz_deg"] = np.degrees(
        np.arctan2(z_proxy, trials["hand_orientation_dy_px"])
    )
    trials["hand_orientation_zx_deg"] = np.degrees(
        np.arctan2(trials["hand_orientation_dx_px"], z_proxy)
    )

    scope_columns = [
        ("all", None),
        ("experiment_group", EXPERIMENT_GROUP_COLUMN),
        ("subject_group", "subject_group"),
        ("workspace", "workspace_setup"),
        ("success", "success_label"),
        ("finger", "finger_condition"),
        ("stiffness", "stiffness_value"),
        ("protocol", "protocol_factor"),
        ("sex", "sex_factor"),
        ("age", "age_group"),
    ]
    rows: list[dict[str, Any]] = []
    for scope, col in scope_columns:
        if col is not None and (col not in trials.columns or not trials[col].notna().any()):
            continue
        grouped = [("all", trials)] if col is None else trials.groupby(col, dropna=False)
        for value, group in grouped:
            for plane_col in HAND_ORIENTATION_PLANE_COLUMNS:
                if plane_col not in group.columns:
                    continue
                values = pd.to_numeric(group[plane_col], errors="coerce").dropna()
                rows.append(
                    {
                        "scope": scope,
                        "group": str(value),
                        "plane": plane_col.replace("hand_orientation_", "")
                        .replace("_deg", "")
                        .upper(),
                        "metric": plane_col,
                        "n": int(values.size),
                        "circular_mean_deg": _circ_mean_deg(values),
                        "median_deg": float(values.median()) if not values.empty else np.nan,
                        "resultant_length": _resultant_length(values),
                    }
                )
    return {
        "hand_orientation_plane_trials": trials,
        "hand_orientation_plane_summary": pd.DataFrame(rows),
    }


def _prepare_hand_orientation_xy_vector_trials(
    hand_orientation_plane_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """Return thumb-to-active-finger XY vectors for hand-orientation plots.

    The ``hand_orientation_planes`` figures are intended to show hand posture:
    the vector from the thumb marker to the active-finger marker.  They must not
    use centered hand/workspace position columns, which describe where the hand
    moved in the workspace rather than how the hand was oriented.
    """

    if hand_orientation_plane_trials.empty:
        return pd.DataFrame(), None
    trials = add_success_label_column(
        add_protocol_demographic_factors(
            add_workspace_normalization_columns(hand_orientation_plane_trials)
        )
    ).copy()
    if "finger_condition" in trials.columns:
        trials["finger_condition"] = trials["finger_condition"].map(
            normalize_finger_condition
        )
    required = {"stiffness_value", "finger_condition"}
    if not required.issubset(trials.columns):
        return pd.DataFrame(), None

    dx_col = _first_existing_column(
        trials, ["hand_orientation_dx_px", "thumb_active_dx_px"]
    )
    dy_col = _first_existing_column(
        trials, ["hand_orientation_dy_px", "thumb_active_dy_px"]
    )
    magnitude_col = _first_existing_column(
        trials, ["mean_thumb_active_span_px", "thumb_active_span_px"]
    )
    angle_col = _first_existing_column(
        trials, ["hand_orientation_xy_deg", "mean_hand_orientation_xy_deg"]
    )

    if dx_col is not None and dy_col is not None:
        trials["_vector_dx"] = pd.to_numeric(trials[dx_col], errors="coerce")
        trials["_vector_dy"] = pd.to_numeric(trials[dy_col], errors="coerce")
        vector_source = f"{dx_col}/{dy_col}"
    elif magnitude_col is not None and angle_col is not None:
        span = pd.to_numeric(trials[magnitude_col], errors="coerce")
        angle_rad = np.deg2rad(pd.to_numeric(trials[angle_col], errors="coerce"))
        trials["_vector_dx"] = span * np.cos(angle_rad)
        trials["_vector_dy"] = span * np.sin(angle_rad)
        vector_source = f"{magnitude_col}+{angle_col}"
    else:
        return pd.DataFrame(), None

    trials["_orientation_span_px"] = np.hypot(trials["_vector_dx"], trials["_vector_dy"])
    finite_span = trials["_orientation_span_px"].replace([np.inf, -np.inf], np.nan)
    finite_span = finite_span[finite_span > 0].dropna()
    if len(finite_span) > 2:
        span_p05 = float(finite_span.quantile(0.05))
        span_p95 = float(finite_span.quantile(0.95))
        if np.isfinite(span_p05) and np.isfinite(span_p95) and span_p95 > span_p05:
            trials = trials[
                trials["_orientation_span_px"].between(
                    span_p05, span_p95, inclusive="both"
                )
            ].copy()
    trials = trials.dropna(
        subset=[
            "stiffness_value",
            "finger_condition",
            "_orientation_span_px",
            "_vector_dx",
            "_vector_dy",
        ]
    )
    trials = trials[trials["_orientation_span_px"] > 0].copy()
    return trials, vector_source


def _hand_orientation_matrix_scopes(
    trials: pd.DataFrame,
) -> list[tuple[str, str, pd.DataFrame]]:
    trials = trials.copy()
    scopes: list[tuple[str, str, pd.DataFrame]] = []
    seen: set[str] = set()

    def add_scope(name: str, label: str, subset: pd.DataFrame) -> None:
        if subset.empty or name in seen:
            return
        seen.add(name)
        scopes.append((name, label, subset.copy()))

    if "subject_id" in trials.columns:
        for subject in sorted(trials["subject_id"].dropna().unique(), key=str):
            add_scope(
                f"subject_{sanitize_name(subject)}",
                f"Subject {subject}",
                trials[trials["subject_id"].astype(str) == str(subject)],
            )
    if EXPERIMENT_GROUP_COLUMN in trials.columns or "subject_id" in trials.columns:
        group_series = (
            trials[EXPERIMENT_GROUP_COLUMN].copy()
            if EXPERIMENT_GROUP_COLUMN in trials.columns
            else pd.Series(np.nan, index=trials.index)
        )
        if "subject_id" in trials.columns:
            subject_text = trials["subject_id"].astype(str).str.upper()
            for group in ["N_E", "N_P", "L_E", "L_P"]:
                missing_group = group_series.isna() | (
                    group_series.astype(str).str.lower() == "nan"
                )
                group_series = group_series.mask(
                    missing_group & subject_text.str.startswith(group),
                    group,
                )
        trials["_hand_orientation_experiment_group"] = group_series
        group_labels = {
            "N_E": "N group (E only)",
            "N_P": "N group (P only)",
            "L_E": "L group (E only)",
            "L_P": "L group (P only)",
        }
        for group in ["N_E", "N_P", "L_E", "L_P"]:
            add_scope(
                f"group_{group}",
                group_labels[group],
                trials[
                    trials["_hand_orientation_experiment_group"].astype(str) == group
                ],
            )
    return scopes


def save_hand_orientation_plane_figures(
    output_root: Path,
    hand_orientation_plane_summary: pd.DataFrame,
    hand_orientation_plane_trials: Optional[pd.DataFrame] = None,
    *,
    fig_dpi: int = 160,
) -> list[Path]:
    """Save thumb-to-active-finger XY orientation matrices by participant/group."""

    fig_dir = output_root / "figures" / "hand_orientation_planes"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if hand_orientation_plane_trials is None or hand_orientation_plane_trials.empty:
        save_csv(
            pd.DataFrame({"figure": []}),
            output_root,
            "hand_orientation_plane_figure_manifest.csv",
        )
        return paths

    trials, vector_source = _prepare_hand_orientation_xy_vector_trials(
        hand_orientation_plane_trials
    )
    if trials.empty or vector_source is None:
        save_csv(
            pd.DataFrame({"figure": []}),
            output_root,
            "hand_orientation_plane_figure_manifest.csv",
        )
        return paths

    stiffness_values = sorted(trials["stiffness_value"].dropna().unique(), key=float)
    stiffness_colors = _stiffness_viridis_colors(stiffness_values)
    finger_values = [
        finger
        for finger in FINGER_ORDER
        if finger in set(trials["finger_condition"].dropna().astype(str))
    ]
    other_fingers = sorted(
        {str(finger) for finger in trials["finger_condition"].dropna().unique()}
        - set(finger_values),
        key=str,
    )
    finger_values.extend(other_fingers)

    for scope_name, scope_label, scope_df in _hand_orientation_matrix_scopes(trials):
        scope_values = scope_df[["_vector_dx", "_vector_dy"]].to_numpy(dtype=float)
        scope_vector_extent = (
            np.nanmax(np.abs(scope_values)) if np.isfinite(scope_values).any() else 1.0
        )
        scope_limit = max(float(scope_vector_extent) * 1.12, 1.0)
        scope_fingers = [
            finger
            for finger in finger_values
            if not scope_df[
                scope_df["finger_condition"].astype(str) == str(finger)
            ].empty
        ]
        if not scope_fingers:
            continue
        scope_stiffness_values = [
            stiffness
            for stiffness in stiffness_values
            if not scope_df[scope_df["stiffness_value"] == stiffness].empty
        ]
        ncols = min(2, len(scope_fingers))
        nrows = int(math.ceil(len(scope_fingers) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.0 * ncols, 5.4 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for ax in axes.ravel():
            ax.set_visible(False)
        for finger_idx, finger in enumerate(scope_fingers):
            ax = axes[finger_idx // ncols, finger_idx % ncols]
            ax.set_visible(True)
            finger_df = scope_df[scope_df["finger_condition"].astype(str) == str(finger)]
            for stiffness in scope_stiffness_values:
                stiff_finger_df = finger_df[finger_df["stiffness_value"] == stiffness]
                if stiff_finger_df.empty:
                    continue
                color = stiffness_colors.get(stiffness)
                dx = pd.to_numeric(stiff_finger_df["_vector_dx"], errors="coerce")
                dy = pd.to_numeric(stiff_finger_df["_vector_dy"], errors="coerce")
                draw_df = pd.DataFrame({"dx": dx, "dy": dy}).dropna()
                if len(draw_df) > 250:
                    draw_df = draw_df.sample(n=250, random_state=20260524)
                ax.quiver(
                    np.zeros(len(draw_df)),
                    np.zeros(len(draw_df)),
                    draw_df["dx"],
                    draw_df["dy"],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color,
                    alpha=0.09,
                    width=0.0022,
                )
                median_dx = float(dx.median())
                median_dy = float(dy.median())
                if np.isfinite(median_dx) and np.isfinite(median_dy):
                    ax.quiver(
                        [0],
                        [0],
                        [median_dx],
                        [median_dy],
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        color=color,
                        width=0.013,
                    )
            ax.axhline(0, color="0.55", linewidth=0.8)
            ax.axvline(0, color="0.55", linewidth=0.8)
            ax.set_xlim(-scope_limit, scope_limit)
            ax.set_ylim(-scope_limit, scope_limit)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.18)
            ax.set_title(FINGER_LABELS.get(str(finger), str(finger)))
            ax.set_xlabel("Thumb→active-finger dx (px)")
            ax.set_ylabel("Thumb→active-finger dy (px)")
        fig.suptitle(
            f"{scope_label}: thumb→active-finger XY orientation vectors by finger\n"
            "matrix panels=fingers; colors=stiffness; "
            "faded=trials; thick=median per stiffness; "
            f"orientation span trimmed to P5-P95; source={vector_source}",
            y=0.99,
        )
        if scope_stiffness_values:
            legend_handles = [
                Line2D([0], [0], color=stiffness_colors.get(stiffness), linewidth=3.0)
                for stiffness in scope_stiffness_values
            ]
            legend_labels = [f"{stiffness:g}" for stiffness in scope_stiffness_values]
            fig.legend(
                legend_handles,
                legend_labels,
                title="Stiffness\n25=purple\n145=yellow",
                loc="center right",
                bbox_to_anchor=(0.995, 0.5),
                ncol=1,
                fontsize=8,
            )
        fig.tight_layout(rect=[0, 0.02, 0.88, 0.94])
        out = subject_figure_path_for_scope(
            fig_dir,
            scope_name,
            f"hand_orientation_xy_vectors_{scope_name}.png",
        )
        fig.savefig(out, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "hand_orientation_plane_figure_manifest.csv",
    )
    return paths


def save_hand_orientation_axis_matrix_figures(
    output_root: Path,
    hand_orientation_plane_trials: pd.DataFrame,
    *,
    fig_dpi: int = 160,
) -> list[Path]:
    """Save XY/YZ/ZX orientation-vector matrices by finger and stiffness.

    Rows are projection planes (XY, YZ, ZX), columns are active fingers, faded
    arrows are trials, and thick arrows are medians per stiffness.  The XY row
    uses the corrected thumb-to-active-finger vector.  The YZ/ZX rows combine
    that top-camera vector with the side-camera Z/lift proxy.
    """

    fig_dir = output_root / "figures" / "hand_orientation_axis_matrices"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if hand_orientation_plane_trials.empty:
        save_csv(
            pd.DataFrame({"figure": []}),
            output_root,
            "hand_orientation_axis_matrix_figure_manifest.csv",
        )
        return paths

    trials, vector_source = _prepare_hand_orientation_xy_vector_trials(
        hand_orientation_plane_trials
    )
    z_col = _first_existing_column(
        trials, ["mean_side_z_lift_px", "max_side_z_lift_px", "z_3d_proxy_px"]
    )
    if trials.empty or vector_source is None or z_col is None:
        save_csv(
            pd.DataFrame({"figure": []}),
            output_root,
            "hand_orientation_axis_matrix_figure_manifest.csv",
        )
        return paths

    trials["_vector_z"] = pd.to_numeric(trials[z_col], errors="coerce")
    trials = trials.dropna(subset=["_vector_z"]).copy()
    if trials.empty:
        save_csv(
            pd.DataFrame({"figure": []}),
            output_root,
            "hand_orientation_axis_matrix_figure_manifest.csv",
        )
        return paths

    plane_specs = [
        ("XY", "_vector_dx", "_vector_dy", "dx thumb-to-active (px)", "dy thumb-to-active (px)"),
        ("YZ", "_vector_dy", "_vector_z", "dy thumb-to-active (px)", "side Z/lift proxy (px)"),
        ("ZX", "_vector_z", "_vector_dx", "side Z/lift proxy (px)", "dx thumb-to-active (px)"),
    ]
    stiffness_values = sorted(trials["stiffness_value"].dropna().unique(), key=float)
    stiffness_colors = _stiffness_viridis_colors(stiffness_values)
    finger_values = [
        finger
        for finger in FINGER_ORDER
        if finger in set(trials["finger_condition"].dropna().astype(str))
    ]
    other_fingers = sorted(
        {str(finger) for finger in trials["finger_condition"].dropna().unique()}
        - set(finger_values),
        key=str,
    )
    finger_values.extend(other_fingers)

    for scope_name, scope_label, scope_df in _hand_orientation_matrix_scopes(trials):
        scope_fingers = [
            finger
            for finger in finger_values
            if not scope_df[
                scope_df["finger_condition"].astype(str) == str(finger)
            ].empty
        ]
        if not scope_fingers:
            continue
        scope_stiffness_values = [
            stiffness
            for stiffness in stiffness_values
            if not scope_df[scope_df["stiffness_value"] == stiffness].empty
        ]

        plane_limits: dict[str, float] = {}
        for plane_name, x_col, y_col, _, _ in plane_specs:
            values = scope_df[[x_col, y_col]].to_numpy(dtype=float)
            extent = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
            plane_limits[plane_name] = max(float(extent) * 1.12, 1.0)

        fig, axes = plt.subplots(
            len(plane_specs),
            len(scope_fingers),
            figsize=(4.8 * len(scope_fingers), 4.4 * len(plane_specs)),
            squeeze=False,
            sharex=False,
            sharey=False,
        )

        for row_idx, (plane_name, x_col, y_col, x_label, y_label) in enumerate(
            plane_specs
        ):
            limit = plane_limits[plane_name]
            for col_idx, finger in enumerate(scope_fingers):
                ax = axes[row_idx, col_idx]
                finger_df = scope_df[
                    scope_df["finger_condition"].astype(str) == str(finger)
                ]
                for stiffness in scope_stiffness_values:
                    stiff_finger_df = finger_df[
                        finger_df["stiffness_value"] == stiffness
                    ]
                    if stiff_finger_df.empty:
                        continue
                    color = stiffness_colors.get(stiffness)
                    draw_df = pd.DataFrame(
                        {
                            "x": pd.to_numeric(stiff_finger_df[x_col], errors="coerce"),
                            "y": pd.to_numeric(stiff_finger_df[y_col], errors="coerce"),
                        }
                    ).dropna()
                    if len(draw_df) > 200:
                        draw_df = draw_df.sample(n=200, random_state=20260527)
                    ax.quiver(
                        np.zeros(len(draw_df)),
                        np.zeros(len(draw_df)),
                        draw_df["x"],
                        draw_df["y"],
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        color=color,
                        alpha=0.08,
                        width=0.002,
                    )
                    median_x = float(
                        pd.to_numeric(stiff_finger_df[x_col], errors="coerce").median()
                    )
                    median_y = float(
                        pd.to_numeric(stiff_finger_df[y_col], errors="coerce").median()
                    )
                    if np.isfinite(median_x) and np.isfinite(median_y):
                        ax.quiver(
                            [0],
                            [0],
                            [median_x],
                            [median_y],
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            color=color,
                            width=0.012,
                        )
                ax.axhline(0, color="0.55", linewidth=0.8)
                ax.axvline(0, color="0.55", linewidth=0.8)
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=0.18)
                if row_idx == 0:
                    ax.set_title(FINGER_LABELS.get(str(finger), str(finger)))
                if col_idx == 0:
                    ax.set_ylabel(f"{plane_name}\n{y_label}")
                else:
                    ax.set_ylabel(y_label)
                ax.set_xlabel(x_label)

        fig.suptitle(
            f"{scope_label}: hand-orientation plane matrix\n"
            "rows=axes/projection planes; columns=fingers; colors=stiffness; "
            "faded=trials; thick=median per stiffness; "
            f"XY source={vector_source}; Z source={z_col}",
            y=0.995,
        )
        if scope_stiffness_values:
            legend_handles = [
                Line2D([0], [0], color=stiffness_colors.get(stiffness), linewidth=3.0)
                for stiffness in scope_stiffness_values
            ]
            legend_labels = [f"{stiffness:g}" for stiffness in scope_stiffness_values]
            fig.legend(
                legend_handles,
                legend_labels,
                title="Stiffness\n25=purple\n145=yellow",
                loc="center right",
                bbox_to_anchor=(0.995, 0.5),
                ncol=1,
                fontsize=8,
            )
        fig.tight_layout(rect=[0, 0.02, 0.90, 0.95])
        out = subject_figure_path_for_scope(
            fig_dir,
            scope_name,
            f"hand_orientation_axis_matrix_{scope_name}.png",
        )
        fig.savefig(out, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "hand_orientation_axis_matrix_figure_manifest.csv",
    )
    return paths


def _movement_cycle_time_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    d = df.copy()
    if "stiffness_time_s" in d.columns:
        d["movement_cycle_time_s"] = pd.to_numeric(
            d["stiffness_time_s"], errors="coerce"
        )
        if d["movement_cycle_time_s"].notna().any():
            return d, "movement_cycle_time_s", "Movement-cycle time (s)"
    if {
        "stiffness_time_fraction",
        "stiffness_start_fraction",
        "stiffness_end_fraction",
        "time_to_answer_s",
    }.issubset(d.columns):
        duration = (
            pd.to_numeric(d["stiffness_end_fraction"], errors="coerce")
            - pd.to_numeric(d["stiffness_start_fraction"], errors="coerce")
        ) * pd.to_numeric(d["time_to_answer_s"], errors="coerce")
        d["movement_cycle_time_s"] = (
            pd.to_numeric(d["stiffness_time_fraction"], errors="coerce") * duration
        )
        if d["movement_cycle_time_s"].notna().any():
            return d, "movement_cycle_time_s", "Movement-cycle time (s)"
    if {"time_fraction", "time_to_answer_s"}.issubset(d.columns):
        d["movement_cycle_time_s"] = pd.to_numeric(
            d["time_fraction"], errors="coerce"
        ) * pd.to_numeric(d["time_to_answer_s"], errors="coerce")
        if d["movement_cycle_time_s"].notna().any():
            return d, "movement_cycle_time_s", "Trial time (s)"
    if "stiffness_time_fraction" in d.columns:
        return d, "stiffness_time_fraction", "Normalized movement-cycle time"
    return d, "time_fraction", "Normalized trial time"


def _movement_cycle_scope_variants(
    df: pd.DataFrame,
) -> list[tuple[str, str, pd.DataFrame]]:
    scopes: list[tuple[str, str, pd.DataFrame]] = []
    seen: set[str] = set()

    def add_scope(prefix: str, label: str, subset: pd.DataFrame) -> None:
        if subset.empty or prefix in seen:
            return
        seen.add(prefix)
        scopes.append((prefix, label, subset.copy()))

    if "subject_id" in df.columns:
        for subject in sorted(df["subject_id"].dropna().unique(), key=str):
            add_scope(
                f"subject_{sanitize_name(subject)}",
                f"Subject {subject}",
                df[df["subject_id"].astype(str) == str(subject)],
            )
    if "subject_group" in df.columns:
        for group in sorted(df["subject_group"].dropna().unique(), key=str):
            add_scope(
                f"group_{sanitize_name(group)}",
                f"Group {group}",
                df[df["subject_group"].astype(str) == str(group)],
            )
    if EXPERIMENT_GROUP_COLUMN in df.columns:
        for group in sorted(df[EXPERIMENT_GROUP_COLUMN].dropna().unique(), key=str):
            add_scope(
                f"experiment_group_{sanitize_name(group)}",
                f"Experiment group {group}",
                df[df[EXPERIMENT_GROUP_COLUMN].astype(str) == str(group)],
            )
    return scopes


def _trial_group_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "subject_id",
        "tracking_file",
        "trial_index_raw",
        "pair_number",
        "stiffness_segment_id",
    ]
    cols = [col for col in candidates if col in df.columns]
    return cols or ["trajectory_time_bin"]


def save_movement_cycle_hand_angle_figures(
    output_root: Path,
    trajectory_time_bins: pd.DataFrame,
    fig_dpi: int = 160,
) -> list[Path]:
    """Plot movement direction and hand angle over each stiffness cycle.

    Each figure overlays all trials as faint per-trial trajectories and adds
    thick circular mean/median angle trajectories per stiffness value.
    """
    fig_dir = output_root / "figures" / "movement_cycle_hand_angles"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    required = {
        "stiffness_value",
        "trajectory_time_bin",
        "movement_angle_deg",
        "hand_orientation_xy_deg",
    }
    if trajectory_time_bins.empty or not required.issubset(trajectory_time_bins.columns):
        save_csv(
            pd.DataFrame({"figure": [str(p) for p in paths]}),
            output_root,
            "movement_cycle_hand_angle_figure_manifest.csv",
        )
        return paths

    d = add_protocol_demographic_factors(
        add_workspace_normalization_columns(trajectory_time_bins.copy())
    )
    if "subject_id" in d.columns:
        subject_text = d["subject_id"].astype(str)
        d = d[~subject_text.str.contains("_P_", case=False, na=False)].copy()
    if "finger_condition" in d.columns:
        d["finger_condition"] = d["finger_condition"].map(normalize_finger_condition)
    d, time_col, _time_label = _movement_cycle_time_columns(d)
    numeric_cols = [
        "stiffness_value",
        "trajectory_time_bin",
        time_col,
        "movement_angle_deg",
        "hand_orientation_xy_deg",
    ]
    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d["movement_angle_rad"] = _wrap_radians(np.deg2rad(d["movement_angle_deg"]))
    d["hand_orientation_xy_rad"] = _wrap_radians(
        np.deg2rad(d["hand_orientation_xy_deg"])
    )
    d = d.dropna(subset=["stiffness_value", "trajectory_time_bin", time_col])
    d["duration_ms"] = pd.to_numeric(d[time_col], errors="coerce") * 1000.0
    d = d.dropna(subset=["duration_ms"])
    if d.empty:
        save_csv(
            pd.DataFrame({"figure": [str(p) for p in paths]}),
            output_root,
            "movement_cycle_hand_angle_figure_manifest.csv",
        )
        return paths

    duration_p5 = float(d["duration_ms"].quantile(0.05))
    duration_p95 = float(d["duration_ms"].quantile(0.95))
    if not np.isfinite(duration_p5):
        duration_p5 = float(d["duration_ms"].min())
    if not np.isfinite(duration_p95) or duration_p95 <= duration_p5:
        duration_p95 = float(d["duration_ms"].max())
    d = d[d["duration_ms"].between(duration_p5, duration_p95, inclusive="both")]
    if d.empty:
        save_csv(
            pd.DataFrame({"figure": [str(p) for p in paths]}),
            output_root,
            "movement_cycle_hand_angle_figure_manifest.csv",
        )
        return paths

    angle_specs = [
        ("movement_angle_rad", "Movement direction (rad)"),
        (
            "hand_orientation_xy_rad",
            "Thumb-to-active-finger XY angle (rad)",
        ),
    ]
    stiffness_values = sorted(d["stiffness_value"].dropna().unique())
    stiffness_colors = _stiffness_viridis_colors(stiffness_values)
    trial_cols = _trial_group_columns(d)

    for prefix, label, scope_df in _movement_cycle_scope_variants(d):
        scope_df = scope_df.dropna(subset=["stiffness_value", "duration_ms"])
        if scope_df.empty:
            continue
        scope_stiffness_values = [
            s
            for s in stiffness_values
            if not scope_df[scope_df["stiffness_value"] == s].empty
        ]
        if not scope_stiffness_values:
            continue
        # Keep every stiffness panel in one horizontal row so the movement-cycle
        # sequence can be scanned left-to-right without wrapping.
        ncols = max(1, len(scope_stiffness_values))
        nrows = 1
        for angle_col, angle_label in angle_specs:
            safe_angle = (
                "movement_direction"
                if angle_col == "movement_angle_rad"
                else "thumb_to_active_finger_xy_angle"
            )
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(3.4 * ncols, 2.7 * nrows),
                sharex=True,
                squeeze=False,
            )
            for ax in axes.ravel():
                ax.set_visible(False)
            for stiffness_idx, stiffness in enumerate(scope_stiffness_values):
                row = 0
                col = stiffness_idx
                ax = axes[row, col]
                ax.set_visible(True)
                ax.axhline(0, color="0.55", linewidth=0.8, alpha=0.7)
                stiff_df = scope_df[scope_df["stiffness_value"] == stiffness]
                color = stiffness_colors.get(stiffness)
                if not stiff_df.empty and stiff_df[angle_col].notna().sum() > 0:
                    for _, trial in stiff_df.groupby(trial_cols, dropna=False):
                        trial = trial.sort_values("duration_ms")
                        ax.plot(
                            trial["duration_ms"],
                            trial[angle_col],
                            color=color,
                            alpha=0.035,
                            linewidth=0.35,
                            zorder=1,
                        )
                    summary = (
                        stiff_df.groupby("trajectory_time_bin", as_index=False)
                        .agg(
                            t=("duration_ms", "mean"),
                            mean_angle=(angle_col, _circ_mean_rad),
                            median_angle=(angle_col, _circ_median_rad),
                            p05_angle=(angle_col, lambda s: _circ_quantile_rad(s, 0.05)),
                            p95_angle=(angle_col, lambda s: _circ_quantile_rad(s, 0.95)),
                        )
                        .sort_values("t")
                    )
                    ax.plot(
                        summary["t"],
                        summary["mean_angle"],
                        color=color,
                        linewidth=3.4,
                        zorder=5,
                    )
                    ax.plot(
                        summary["t"],
                        summary["median_angle"],
                        color=color,
                        linewidth=2.4,
                        linestyle="--",
                        alpha=0.95,
                        zorder=5,
                    )
                    lower = pd.to_numeric(summary["p05_angle"], errors="coerce")
                    upper = pd.to_numeric(summary["p95_angle"], errors="coerce")
                    no_wrap = lower <= upper
                    if no_wrap.any():
                        ax.fill_between(
                            summary["t"].to_numpy(dtype=float),
                            lower.to_numpy(dtype=float),
                            upper.to_numpy(dtype=float),
                            where=no_wrap.to_numpy(dtype=bool),
                            color=color,
                            alpha=0.055,
                            linewidth=0,
                            zorder=2,
                        )
                    ax.plot(
                        summary["t"],
                        lower,
                        color=color,
                        linewidth=0.8,
                        linestyle=":",
                        alpha=0.45,
                        zorder=3,
                    )
                    ax.plot(
                        summary["t"],
                        upper,
                        color=color,
                        linewidth=0.8,
                        linestyle=":",
                        alpha=0.45,
                        zorder=3,
                    )
                ax.set_title(f"Stiffness {stiffness:g}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(angle_label)
                ax.set_xlabel("Duration (ms)")
                _set_pi_axis(ax)
                ax.set_xlim(duration_p5, duration_p95)
                ax.grid(alpha=0.18)
            fig.suptitle(
                f"{label}: {angle_label} by stiffness\n"
                "thin=trial traces; solid=mean; dashed=median; shaded/dotted=P5-P95; "
                "duration axis uses the global P5-P95 millisecond window; "
                "subjects with _P_ are excluded",
                y=0.99,
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.92])
            out = subject_figure_path_for_scope(
                fig_dir,
                prefix,
                f"movement_cycle_{safe_angle}_{prefix}.png",
            )
            fig.savefig(out, dpi=fig_dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(out)

    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "movement_cycle_hand_angle_figure_manifest.csv",
    )
    return paths


def _group_variants(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    d = add_success_label_column(
        add_protocol_demographic_factors(add_workspace_normalization_columns(df))
    )
    out = [("all", d)]
    scope_columns = [
        ("experiment_group", EXPERIMENT_GROUP_COLUMN),
        ("subject_group", "subject_group"),
        ("workspace", "workspace_setup"),
        ("success", "success_label"),
        ("finger", "finger_condition"),
        ("stiffness", "stiffness_value"),
        ("protocol", "protocol_factor"),
        ("sex", "sex_factor"),
        ("age", "age_group"),
    ]
    seen = {"all"}
    for scope, col in scope_columns:
        if col not in d.columns:
            continue
        values = [v for v in d[col].dropna().unique()]
        for value in sorted(values, key=lambda x: str(x)):
            sub = d[d[col].astype(str) == str(value)]
            if sub.empty:
                continue
            name = f"{scope}_{value}"
            if name in seen:
                continue
            seen.add(name)
            out.append((name, sub))
    return out


def save_kinematic_figures(
    output_root: Path,
    trial_summary: pd.DataFrame,
    group_time: pd.DataFrame,
    direction_success: pd.DataFrame,
    distance_success: pd.DataFrame,
    side_group_time: Optional[pd.DataFrame] = None,
    side_stiffness_summary: Optional[pd.DataFrame] = None,
    fig_dpi: int = 160,
) -> list[Path]:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not group_time.empty:
        for y_col, y_label, filename in [
            (
                "mean_r_center_px",
                "Distance from center (px)",
                "time_distance_from_center.png",
            ),
            ("mean_speed_px_s", "Speed (px/s)", "time_speed.png"),
            (
                "mean_acceleration_px_s2",
                "Acceleration (px/s^2)",
                "time_acceleration.png",
            ),
            (
                "mean_radial_velocity_px_s",
                "Radial velocity (px/s)",
                "time_radial_velocity.png",
            ),
            (
                "mean_tangential_velocity_px_s",
                "Tangential velocity (px/s)",
                "time_tangential_velocity.png",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for finger in sorted(
                group_time["finger_condition"].dropna().unique(),
                key=lambda x: FINGER_ORDER.index(x) if x in FINGER_ORDER else 99,
            ):
                g = (
                    group_time[group_time["finger_condition"] == finger]
                    .groupby("trajectory_time_bin", as_index=False)
                    .agg(time_fraction=("time_fraction", "mean"), y=(y_col, "mean"))
                )
                ax.plot(
                    g["time_fraction"],
                    g["y"],
                    label=FINGER_LABELS.get(str(finger), str(finger)),
                    color=FINGER_TO_COLOR.get(str(finger)),
                )
            ax.set_xlabel("Normalized trial time")
            ax.set_ylabel(y_label)
            ax.set_title(y_label + " over time")
            ax.legend(fontsize=8)
            fig.tight_layout()
            out = fig_dir / filename
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for finger in sorted(
            group_time["finger_condition"].dropna().unique(),
            key=lambda x: FINGER_ORDER.index(x) if x in FINGER_ORDER else 99,
        ):
            g = (
                group_time[group_time["finger_condition"] == finger]
                .groupby("trajectory_time_bin", as_index=False)
                .agg(x=("mean_x_centered_px", "mean"), y=("mean_y_centered_px", "mean"))
            )
            ax.plot(
                g["x"],
                g["y"],
                label=FINGER_LABELS.get(str(finger), str(finger)),
                color=FINGER_TO_COLOR.get(str(finger)),
            )
            if not g.empty:
                ax.scatter(
                    g["x"].iloc[0],
                    g["y"].iloc[0],
                    marker="o",
                    color=FINGER_TO_COLOR.get(str(finger)),
                )
                ax.scatter(
                    g["x"].iloc[-1],
                    g["y"].iloc[-1],
                    marker="x",
                    color=FINGER_TO_COLOR.get(str(finger)),
                )
        ax.axhline(0, color="0.7")
        ax.axvline(0, color="0.7")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X from center (px)")
        ax.set_ylabel("Y from center (px)")
        ax.set_title("Mean XY trajectory by finger")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = fig_dir / "mean_xy_trajectory_by_finger.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)
    if not trial_summary.empty:
        polar_dir = fig_dir / "polar_radiation_patterns"
        polar_dir.mkdir(exist_ok=True)
        for name, sub in _group_variants(trial_summary):
            angles = pd.to_numeric(
                sub["dominant_movement_angle_deg"], errors="coerce"
            ).dropna()
            if angles.empty:
                continue
            bins = np.linspace(-180, 180, 73)
            counts, edges = np.histogram(angles, bins=bins)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="polar")
            ax.bar(
                np.deg2rad(edges[:-1] + np.diff(edges) / 2),
                counts,
                width=np.deg2rad(np.diff(edges)) * 0.92,
                bottom=0,
                alpha=0.75,
            )
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.set_title(f"Movement-orientation radiation pattern: {name}")
            out = polar_dir / f"radiation_orientation_{sanitize_name(name)}.png"
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)
        for subject, sub in trial_summary.groupby("subject_id", dropna=False):
            angles = pd.to_numeric(
                sub["dominant_movement_angle_deg"], errors="coerce"
            ).dropna()
            if angles.empty:
                continue
            bins = np.linspace(-180, 180, 73)
            counts, edges = np.histogram(angles, bins=bins)
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="polar")
            ax.bar(
                np.deg2rad(edges[:-1] + np.diff(edges) / 2),
                counts,
                width=np.deg2rad(np.diff(edges)) * 0.92,
                alpha=0.75,
            )
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.set_title(f"Orientation pattern: {subject}")
            out = subject_figure_path(
                polar_dir,
                subject,
                f"radiation_orientation_subject_{sanitize_name(subject)}.png",
            )
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)
        distance_polar_dir = fig_dir / "polar_distance_patterns"
        distance_polar_dir.mkdir(exist_ok=True)
        d_polar = add_success_label_column(
            add_workspace_normalization_columns(trial_summary)
        )
        for name, sub in _group_variants(d_polar):
            if "dominant_movement_angle_deg" not in sub.columns:
                continue
            success_values = ["success", "failure"]
            fig, axes = plt.subplots(
                1,
                2,
                subplot_kw={"projection": "polar"},
                figsize=(12.5, 5.8),
            )
            plotted_distance = False
            for ax, success_name in zip(axes, success_values):
                success_sub = sub[sub["success_label"].astype(str) == success_name].copy()
                if success_sub.empty:
                    ax.set_title(f"{success_name}: no trials")
                    ax.set_axis_off()
                    continue
                angles = pd.to_numeric(
                    success_sub["dominant_movement_angle_deg"], errors="coerce"
                )
                radius = pd.to_numeric(
                    success_sub.get(
                        "max_r_center_px",
                        success_sub.get("mean_max_r_center_px", np.nan),
                    ),
                    errors="coerce",
                )
                if "mean_r_workspace_normalized" in success_sub.columns:
                    norm_radius = pd.to_numeric(
                        success_sub["mean_r_workspace_normalized"], errors="coerce"
                    )
                else:
                    norm_radius = pd.Series(np.nan, index=success_sub.index)
                valid = angles.notna() & radius.notna()
                if not valid.any():
                    ax.set_title(f"{success_name}: no valid trials")
                    ax.set_axis_off()
                    continue
                bins = np.linspace(-180, 180, 73)
                bin_idx = pd.cut(
                    angles[valid],
                    bins=bins,
                    include_lowest=True,
                    labels=False,
                )
                plot_df = pd.DataFrame(
                    {
                        "bin": bin_idx,
                        "radius_px": radius[valid].to_numpy(),
                        "radius_norm": norm_radius[valid].to_numpy(),
                    }
                ).dropna(subset=["bin", "radius_px"])
                if plot_df.empty:
                    ax.set_title(f"{success_name}: no valid trials")
                    ax.set_axis_off()
                    continue
                grouped = plot_df.groupby("bin", as_index=False).agg(
                    radius_px=("radius_px", "median"),
                    radius_norm=("radius_norm", "median"),
                    n=("radius_px", "count"),
                )
                centers = bins[:-1] + np.diff(bins) / 2
                heights = np.zeros(len(centers))
                labels = np.zeros(len(centers))
                for _, row in grouped.iterrows():
                    idx = int(row["bin"])
                    if 0 <= idx < len(heights):
                        heights[idx] = row["radius_px"]
                        labels[idx] = row["radius_norm"]
                bars = ax.bar(
                    np.deg2rad(centers),
                    heights,
                    width=np.deg2rad(np.diff(bins)) * 0.92,
                    bottom=0,
                    alpha=0.78,
                    color="#4C78A8" if success_name == "success" else "#E45756",
                )
                for bar, norm_value in zip(bars, labels):
                    if np.isfinite(norm_value) and norm_value > 0:
                        bar.set_alpha(float(np.clip(0.25 + 0.6 * norm_value, 0.25, 0.9)))
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(-1)
                ax.set_title(f"{success_name}: median distance")
                plotted_distance = True
            if plotted_distance:
                fig.suptitle(
                    f"Direction distance pattern: {name}\n"
                    "success and failure are side-by-side; bar radius=median original px distance; "
                    "alpha=workspace-normalized distance",
                    y=1.02,
                )
                fig.tight_layout()
                out = (
                    distance_polar_dir
                    / f"polar_distance_success_failure_{sanitize_name(name)}.png"
                )
                fig.savefig(out, dpi=fig_dpi, bbox_inches="tight")
                paths.append(out)
            plt.close(fig)
            if "success_label" not in sub.columns:
                continue
            if "duration_s" in sub.columns:
                time_values = pd.to_numeric(sub["duration_s"], errors="coerce")
            else:
                time_values = pd.Series(1.0, index=sub.index)
            valid_polar = (
                pd.to_numeric(sub["dominant_movement_angle_deg"], errors="coerce").notna()
                & time_values.notna()
            )
            if not valid_polar.any():
                continue
            bins = np.linspace(-180, 180, 73)
            centers = bins[:-1] + np.diff(bins) / 2
            stiffness_labels = (
                sorted(sub["stiffness_value"].dropna().unique(), key=lambda x: float(x))
                if "stiffness_value" in sub.columns
                else ["all stiffness"]
            )
            cmap = plt.get_cmap(STIFFNESS_CMAP)
            stiffness_colors = {
                stiffness: cmap(i / max(1, len(stiffness_labels) - 1))
                for i, stiffness in enumerate(stiffness_labels)
            }
            fig, axes = plt.subplots(
                1,
                2,
                subplot_kw={"projection": "polar"},
                figsize=(12.5, 5.8),
            )
            plotted_any = False
            for ax, success_name in zip(axes, success_values):
                success_sub = sub[sub["success_label"].astype(str) == success_name].copy()
                if success_sub.empty:
                    ax.set_title(f"{success_name}: no trials")
                    ax.set_axis_off()
                    continue
                success_sub["_movement_time_s"] = pd.to_numeric(
                    success_sub.get("duration_s", 1.0), errors="coerce"
                ).fillna(1.0)
                if "stiffness_value" not in success_sub.columns:
                    success_sub["stiffness_value"] = "all stiffness"
                for stiffness, stiff_df in success_sub.groupby("stiffness_value", dropna=False):
                    angles = pd.to_numeric(
                        stiff_df["dominant_movement_angle_deg"], errors="coerce"
                    )
                    bin_idx = pd.cut(
                        angles,
                        bins=bins,
                        include_lowest=True,
                        labels=False,
                    )
                    plot_df = pd.DataFrame(
                        {
                            "bin": bin_idx,
                            "movement_time_s": stiff_df["_movement_time_s"].to_numpy(),
                        }
                    ).dropna(subset=["bin", "movement_time_s"])
                    if plot_df.empty:
                        continue
                    grouped = plot_df.groupby("bin", as_index=False).agg(
                        movement_time_s=("movement_time_s", "sum")
                    )
                    heights = np.zeros(len(centers))
                    for _, row in grouped.iterrows():
                        idx = int(row["bin"])
                        if 0 <= idx < len(heights):
                            heights[idx] = row["movement_time_s"]
                    ax.bar(
                        np.deg2rad(centers),
                        heights,
                        width=np.deg2rad(np.diff(bins)) * 0.92,
                        bottom=0,
                        alpha=0.46,
                        color=stiffness_colors.get(stiffness, "#4C78A8"),
                        edgecolor=stiffness_colors.get(stiffness, "#4C78A8"),
                        linewidth=0.5,
                        label=str(stiffness),
                    )
                    plotted_any = True
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(-1)
                ax.set_title(f"{success_name}: radius = movement time in area")
            if plotted_any:
                handles, labels = axes[-1].get_legend_handles_labels()
                if handles:
                    fig.legend(
                        handles,
                        labels,
                        loc="lower center",
                        ncol=min(6, len(labels)),
                        title="Stiffness (semi-transparent colors)",
                    )
                fig.suptitle(
                    f"Polar dwell-time pattern by stiffness: {name}\n"
                    "success and failure are side-by-side; radial distance is summed movement time",
                    y=1.02,
                )
                fig.tight_layout(rect=[0, 0.08, 1, 0.95])
                out = distance_polar_dir / f"polar_dwell_time_success_failure_{sanitize_name(name)}.png"
                fig.savefig(out, dpi=fig_dpi, bbox_inches="tight")
                paths.append(out)
            plt.close(fig)
    if not direction_success.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot = (
            direction_success.groupby("dominant_movement_direction")
            .agg(success_rate=("success_rate", "mean"), n_trials=("n_trials", "sum"))
            .reindex(DIRECTION_LABELS)
        )
        ax.bar(pivot.index.astype(str), pivot["success_rate"], color="#4C78A8")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Success rate")
        ax.set_xlabel("Dominant movement direction")
        ax.set_title("Does a particular movement direction improve selection accuracy?")
        for i, row in enumerate(pivot.itertuples()):
            n = getattr(row, "n_trials")
            y = getattr(row, "success_rate")
            if pd.notna(n):
                ax.text(
                    i,
                    min(1, (y if pd.notna(y) else 0) + 0.03),
                    f"n={int(n)}",
                    ha="center",
                    fontsize=7,
                    rotation=90,
                )
        fig.tight_layout()
        out = fig_dir / "success_by_dominant_direction.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)
    if not distance_success.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for finger, g in distance_success.groupby("finger_condition", dropna=False):
            gg = g.groupby("distance_quantile", as_index=False).agg(
                success_rate=("success_rate", "mean"),
                r=("mean_max_r_center_px", "mean"),
            )
            ax.plot(
                gg["r"],
                gg["success_rate"],
                marker="o",
                label=FINGER_LABELS.get(str(finger), str(finger)),
                color=FINGER_TO_COLOR.get(str(finger)),
            )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean max distance from center (px)")
        ax.set_ylabel("Success rate")
        ax.set_title("Selection accuracy vs distance from center")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = fig_dir / "success_vs_distance_from_center.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)
    if side_group_time is not None and not side_group_time.empty:
        side_dir = fig_dir / "side_z_by_group"
        side_dir.mkdir(exist_ok=True)
        side_plot = add_success_label_column(
            add_protocol_demographic_factors(add_workspace_normalization_columns(side_group_time))
        )
        for scope, col in [
            ("subject_group", "subject_group"),
            ("experiment_group", EXPERIMENT_GROUP_COLUMN),
            ("workspace", "workspace_setup"),
            ("finger", "finger_condition"),
            ("stiffness", "stiffness_value"),
            ("success", "success_label"),
            ("protocol", "protocol_factor"),
            ("sex", "sex_factor"),
            ("age", "age_group"),
        ]:
            if col not in side_plot.columns or not side_plot[col].notna().any():
                continue
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            for group, g in side_plot.groupby(col, dropna=False):
                gg = g.groupby("side_time_bin", as_index=False).agg(
                    t=("side_time_fraction", "mean"),
                    z=("mean_side_z_lift_px", "mean"),
                    sem=("mean_side_z_lift_px", _sem),
                )
                ax.plot(gg["t"], gg["z"], marker="o", label=str(group))
                if gg["sem"].notna().any():
                    ax.fill_between(
                        gg["t"],
                        gg["z"] - gg["sem"],
                        gg["z"] + gg["sem"],
                        alpha=0.12,
                    )
            ax.set_xlabel("Normalized side-video time")
            ax.set_ylabel("Side-view Z/lift proxy (px)")
            ax.set_title(f"Estimated Z/lift from side camera over time by {scope}")
            ax.legend(fontsize=8, title=scope)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            out = side_dir / f"side_z_lift_over_time_by_{scope}.png"
            fig.savefig(out, dpi=fig_dpi)
            if scope == "subject_group":
                legacy_out = fig_dir / "side_z_lift_over_time_by_group.png"
                fig.savefig(legacy_out, dpi=fig_dpi)
                paths.append(legacy_out)
            plt.close(fig)
            paths.append(out)
        if "subject_id" in side_plot.columns:
            participant_colors = _figure_colors(
                sorted(side_plot["subject_id"].dropna().astype(str).unique(), key=str)
            )
            fig, ax = plt.subplots(figsize=(10.5, 5.6))
            for subject, g in side_plot.groupby("subject_id", dropna=False):
                gg = g.groupby("side_time_bin", as_index=False).agg(
                    t=("side_time_fraction", "mean"),
                    z=("mean_side_z_lift_px", "mean"),
                )
                ax.plot(
                    gg["t"],
                    gg["z"],
                    marker="o",
                    linewidth=1.2,
                    alpha=0.55,
                    color=participant_colors.get(str(subject)),
                    label=str(subject),
                )
            ax.set_xlabel("Normalized side-video time")
            ax.set_ylabel("Side-view Z/lift proxy (px)")
            ax.set_title("Side-camera Z/lift over time per participant")
            ax.grid(alpha=0.25)
            if side_plot["subject_id"].nunique() <= 28:
                ax.legend(fontsize=6, ncol=3, title="Participant")
            fig.tight_layout()
            out = side_dir / "side_z_lift_over_time_per_participant.png"
            fig.savefig(out, dpi=fig_dpi)
            plt.close(fig)
            paths.append(out)

            if "stiffness_value" in side_plot.columns:
                fig, ax = plt.subplots(figsize=(10.5, 5.6))
                subject_stiffness = (
                    side_plot.groupby(["subject_id", "stiffness_value"], as_index=False)
                    .agg(z=("mean_side_z_lift_px", "mean"))
                    .sort_values(["subject_id", "stiffness_value"])
                )
                for subject, g in subject_stiffness.groupby("subject_id", dropna=False):
                    ax.plot(
                        pd.to_numeric(g["stiffness_value"], errors="coerce"),
                        pd.to_numeric(g["z"], errors="coerce"),
                        marker="o",
                        linewidth=1.2,
                        alpha=0.58,
                        color=participant_colors.get(str(subject)),
                        label=str(subject),
                    )
                ax.set_xlabel("Stiffness value (skin-stretch gain mm/m)")
                ax.set_ylabel("Mean side-view Z/lift proxy (px)")
                ax.set_title("Side-camera Z/lift over stiffness per participant")
                ax.grid(alpha=0.25)
                if side_plot["subject_id"].nunique() <= 28:
                    ax.legend(fontsize=6, ncol=3, title="Participant")
                fig.tight_layout()
                out = side_dir / "side_z_lift_over_stiffness_per_participant.png"
                fig.savefig(out, dpi=fig_dpi)
                plt.close(fig)
                paths.append(out)
    if side_stiffness_summary is not None and not side_stiffness_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_col = (
            "stiffness_value"
            if "stiffness_value" in side_stiffness_summary.columns
            else "comparison_value"
        )
        if "subject_group" in side_stiffness_summary.columns:
            groups = side_stiffness_summary.groupby("subject_group", dropna=False)
        else:
            groups = [("all participants", side_stiffness_summary)]
        for group, g in groups:
            g = g.sort_values(x_col)
            ax.errorbar(
                g[x_col],
                g["mean_side_z_lift_px"],
                yerr=g.get("sem_side_z_lift_px"),
                marker="o",
                capsize=3,
                label=str(group),
            )
        ax.set_xlabel("Skin-stretch/stiffness value")
        ax.set_ylabel("Mean side-view Z/lift proxy (px)")
        ax.set_title("Estimated Z axis across stiffness values")
        ax.legend()
        fig.tight_layout()
        out = fig_dir / "side_z_lift_by_stiffness.png"
        fig.savefig(out, dpi=fig_dpi)
        plt.close(fig)
        paths.append(out)
    save_csv(
        pd.DataFrame({"figure": [str(p) for p in paths]}),
        output_root,
        "figure_manifest.csv",
    )
    return paths


def analysis_manifest(output_root: Path) -> pd.DataFrame:
    expected = [
        "trial_file_manifest.csv",
        "experiment_setup_context.csv",
        "kinematic_samples.csv",
        "pair_kinematic_summary.csv",
        "trial_kinematic_summary.csv",
        "trajectory_time_bins.csv",
        "movement_cycle_hand_angle_figure_manifest.csv",
        "direction_success_summary.csv",
        "distance_success_summary.csv",
        "subject_kinematic_summary.csv",
        "participant_stiffness_kinematic_summary.csv",
        "stiffness_kinematic_summary.csv",
        "group_time_summary.csv",
        "stiffness_time_summary.csv",
        "kinematic_group_metric_summary.csv",
        "kinematic_group_condition_metric_summary.csv",
        "kinematic_within_group_condition_comparisons.csv",
        "kinematic_between_group_metric_comparisons.csv",
        "kinematic_analysis_scope_metric_summary.csv",
        "kinematic_analysis_scope_condition_metric_summary.csv",
        "kinematic_within_analysis_scope_condition_comparisons.csv",
        "kinematic_between_analysis_scope_metric_comparisons.csv",
        "kinematic_setup_balance.csv",
        "kinematic_setup_metric_summary.csv",
        "kinematic_setup_condition_metric_summary.csv",
        "kinematic_between_setup_metric_comparisons.csv",
        "kinematic_expanded_scope_metric_summary.csv",
        "kinematic_expanded_scope_pairwise_mean_differences.csv",
        "kinematic_expanded_scope_status.csv",
        "kinematic_scope_figure_manifest.csv",
        "kinematic_within_subject.csv",
        "finger_metric_summary.csv",
        "within_finger_stiffness_effects.csv",
        "within_finger_stiffness_effect_summary.csv",
        "finger_comparison_paired.csv",
        "finger_comparison_by_stiffness_paired.csv",
        "motor_control_figure_manifest.csv",
        "trial_success_kinematic_z_table.csv",
        "success_kinematic_z_contrast_by_subject_finger.csv",
        "success_kinematic_z_contrast_summary.csv",
        "success_kinematic_z_contrast_by_finger_summary.csv",
        "subject_finger_trajectory.csv",
        "trajectory_variability_summary.csv",
        "finger_trajectory_distance_paired.csv",
        "finger_trajectory_distance_summary.csv",
        "subject_xy_trajectory.csv",
        "subject_spatial_trajectory_summary.csv",
        "subject_finger_spatial_distance.csv",
        "subject_spatial_metric_distribution.csv",
        "subject_xy_trajectory_figure_manifest.csv",
        "subject_velocity_acceleration_profile.csv",
        "subject_velocity_acceleration_summary.csv",
        "subject_finger_velocity_acceleration_distance.csv",
        "subject_velocity_acceleration_metric_distribution.csv",
        "velocity_stiffness_influence_summary.csv",
        "velocity_finger_influence_summary.csv",
        "velocity_time_influence_summary.csv",
        "subject_velocity_acceleration_figure_manifest.csv",
        "kinematic_3d_proxy_samples.csv",
        "trial_3d_kinematic_summary.csv",
        "subject_3d_kinematic_summary.csv",
        "subject_3d_metric_distribution.csv",
        "stiffness_direction_3d_metric_distribution.csv",
        "mixedlm_model_input.csv",
        "mixedlm_status.csv",
        "mixedlm_coefficients.csv",
        "proxy_3d_figure_manifest.csv",
        "hand_orientation_plane_trials.csv",
        "hand_orientation_plane_summary.csv",
        "hand_orientation_plane_figure_manifest.csv",
        "hand_orientation_axis_matrix_figure_manifest.csv",
        "success_failure_trajectory_distance.csv",
        "success_failure_trajectory_distance_summary.csv",
        "advanced_kinematic_figure_manifest.csv",
        "side_z_samples.csv",
        "side_z_trial_summary.csv",
        "side_z_group_time_summary.csv",
        "side_z_subject_stiffness_summary.csv",
        "side_z_by_stiffness_summary.csv",
        "figure_manifest.csv",
    ]
    return pd.DataFrame(
        {
            "output": expected,
            "exists": [(output_root / x).exists() for x in expected],
            "path": [str(output_root / x) for x in expected],
        }
    )
