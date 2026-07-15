"""Pre-compute side-camera signals for every pair of one experiment session and
cache them to CSV so the analysis notebook loads instantly and reproducibly.

Regions and spool ROIs are detected ONCE (on the longest pair, where the rig is
best exposed) and reused for every pair - the camera/monitor/motor-box do not
move within a session.  Results go to ``<session>/analysis_output/``:

    regions.json                 the shared ROI layout (+ overlay png)
    signals_pair_001.csv ...     per-frame signals per pair
    sessions_meta.csv            finger<->pair map and time windows
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from side_camera_separator import annotate, detect_regions_robust, representative_frame
import cv2

from region_signals import extract_signals, init_spools, validated_spools, SpoolROI

# finger id -> name (from consts.py FINGER_NAMES)
FINGER_NAMES = {0: "index", 1: "middle", 2: "ring", 3: "pinky", 4: "thumb"}


def discover_pairs(session_dir: Path) -> list[Path]:
    return sorted(p for p in session_dir.glob("pair_*") if (p / "side_camera.mp4").exists())


def read_config_fingers(session_dir: Path) -> list[int]:
    """Finger id per pair from configuration.csv (rows like '145,0,145,0').

    Sentinel rows (negative ids such as -3/-1) are calibration/markers and are
    skipped; the remaining rows correspond 1:1 to pair_001, pair_002, ...
    """
    cfg = pd.read_csv(session_dir / "configuration.csv", header=None)
    fingers = []
    for _, row in cfg.iterrows():
        fid = int(row[1])
        if fid < 0:
            continue
        fingers.append(fid)
    return fingers


def _load_calibration(session_dir: Path):
    """Return (Calibration | None). Looks for calibration.json beside the data."""
    for cand in (session_dir / "calibration.json", session_dir / "analysis_output" / "calibration.json"):
        if cand.exists():
            from calibrate import Calibration
            print(f"[build] using calibration: {cand}")
            return Calibration.load(cand)
    return None


def build(session_dir: str | Path, stride: int = 2, out_name: str = "analysis_output",
          out_dir: str | Path | None = None) -> Path:
    session_dir = Path(session_dir)
    out = Path(out_dir) if out_dir is not None else session_dir / out_name
    out.mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(session_dir)
    if not pairs:
        raise FileNotFoundError(f"No pair_*/side_camera.mp4 under {session_dir}")

    # Prefer a human-confirmed calibration.json; otherwise auto-detect the shared
    # layout on the longest video (best exposure of the rig).
    longest = max(pairs, key=lambda p: (p / "side_camera.mp4").stat().st_size)
    ref_video = longest / "side_camera.mp4"
    cal = _load_calibration(session_dir)
    # Colour detection uses principled defaults, NOT calibration colours: the
    # on-screen object is software-rendered in known orange/blue (so the default
    # ranges generalise better than a few hand-clicks that under-sample the
    # camera's colour variation), and the tactor is tracked by colour-difference
    # channels (which ignore these ranges). Calibration's value is the clicked
    # REGIONS + SPOOL CENTRES (geometry), which is what auto-detection can't nail.
    colors = None
    if cal is not None:
        regions = cal.to_regions()
        spools = [SpoolROI(cx=c[0], cy=c[1], radius=c[2]) for c in cal.spools] or \
            init_spools(representative_frame(ref_video, 0.5), regions.motors)
    else:
        regions = detect_regions_robust(ref_video)
        # validated_spools refines centres and drops phantom discs; falls back to
        # the single-frame guess if validation finds nothing.
        spools = validated_spools(ref_video, regions.motors) or \
            init_spools(representative_frame(ref_video, 0.5), regions.motors)
        print(f"[build] auto-detected {len(spools)} spool(s). For best motor "
              f"accuracy, run calibrate.py and click the disc centres.")
    regions.save(out / "regions.json")
    cv2.imwrite(str(out / "regions_overlay.png"),
                annotate(representative_frame(ref_video, 0.5), regions))

    fingers = read_config_fingers(session_dir)
    meta_rows = []
    for i, pair in enumerate(pairs):
        fid = fingers[i] if i < len(fingers) else -1
        fname = FINGER_NAMES.get(fid, f"finger_{fid}")
        tracking = pair / "tracking.csv"
        res = extract_signals(
            pair / "side_camera.mp4", regions=regions, spools=spools,
            tracking_csv=tracking if tracking.exists() else None, stride=stride,
            colors=colors,
        )
        res.df.to_csv(out / f"signals_{pair.name}.csv", index=False)
        tdf = pd.read_csv(tracking) if tracking.exists() else pd.DataFrame()
        ts = pd.to_datetime(tdf["timestamp"]) if "timestamp" in tdf else pd.Series([], dtype="datetime64[ns]")
        meta_rows.append({
            "pair": pair.name, "finger_id": fid, "finger": fname,
            "n_frames_signals": len(res.df), "n_spools": len(spools),
            "t_start": ts.min() if len(ts) else None, "t_end": ts.max() if len(ts) else None,
        })
        print(f"{pair.name} ({fname}): {len(res.df)} frames -> signals_{pair.name}.csv")

    pd.DataFrame(meta_rows).to_csv(out / "sessions_meta.csv", index=False)
    (out / "build_info.json").write_text(json.dumps({"stride": stride, "n_spools": len(spools),
                                                      "ref_video": str(ref_video)}, indent=2))
    print(f"done -> {out}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    build(a.session_dir, stride=a.stride)
