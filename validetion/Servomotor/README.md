# Servo Motor Response Characterization

Validation pipeline for the servo/spool command path: it drives one motor
through a deterministic command protocol, films the spool, measures how far the
spool actually rotated, and reports accuracy, repeatability and drift.

The experiment does **not** tune control gains. The PCA9685 hobby-servo path has
no external PID loop, so the question it answers is: *do repeated position
commands produce consistent, correctly-signed mechanical motion over the command
range the haptic device uses?*

## Layout

```
validetion/Servomotor/
    protocol.py                     protocol generator (deltas, A/B trials, drift block)
    motor_io.py                     ESP32 serial I/O (ZM0P<n>F -> OK:M0P<actual>)
    camera_recorder.py              threaded camera capture + MP4 recording
    vision_angle.py                 black-line angle detector (PCA over dark spool pixels)

    run_experiment.py               [entry] acquisition: protocol -> logs -> plots
    analyze.py                      [entry] per-run plots and per-delta summary
    generate_report.py              [entry] verified PDF report for one/two runs
    generate_validation_summary.py  [entry] manuscript figures across all runs

    smoke_test.py                   offline self-test (no hardware, writes nothing)
    synthesize_log.py               fake run data for exercising analyze.py

    responses/                      run data (git-ignored, ~180 MB)
    output/                         generated reports and figures (git-ignored)
```

These are flat modules, not an installable package: run them from this folder,
e.g. `python analyze.py`. Every default path is resolved relative to this
folder, so the commands behave the same whatever your working directory is.

Dependencies come from the repository's `pyproject.toml` — prefix any command
with `uv run` to use the project environment.

## Protocol

For each `delta` in `[5, 10, 25, 75, 125, 250, 500, 1000]` motor ticks:

* **Sequence A**: `0 -> +D -> 0 -> -D -> 0` (5 commands), repeated 3 times.
* **Sequence B**: `0 -> -D -> 0 -> +D -> 0` (5 commands), repeated 3 times.
* Trials interleave **A, B, A, B, A, B**.
* Then a **drift block**: 10 x `(+D, -D)` = 20 alternating commands.

That is 6 x 5 + 20 = **50 commands per delta**, **400 commands** total. It yields
six positive and six negative response samples per delta.

## 1. Acquire a run

```powershell
# Real experiment (ESP32 on COM13, camera index 1):
python run_experiment.py

# Different port, longer settle and a wider gap between steps:
python run_experiment.py --port COM5 --settle-ms 1800 --inter-command-ms 400

# Always click the spool ROI by hand:
python run_experiment.py --roi-mode manual

# No hardware and no camera, just to check the pipeline runs:
python run_experiment.py --dry-run --no-camera --settle-ms 0 --inter-command-ms 0
```

| flag | default | meaning |
|---|---|---|
| `--port` | `COM13` | ESP32 serial port |
| `--baud` | `115200` | serial baud rate |
| `--motor-index` | `0` | which motor (the `M0` in the command string) |
| `--settle-ms` | `1200` | wait after the firmware `OK:` for the spool to settle |
| `--inter-command-ms` | `250` | pause after the vision capture, before the next command |
| `--frame-grab-timeout` | `3.0` | max seconds to wait for a frame newer than the settle |
| `--camera-index` | `1` | OpenCV camera index |
| `--camera-fps` | `30` | requested capture rate |
| `--camera-width` / `--camera-height` | *(driver default)* | requested capture size |
| `--deltas` | `5 10 25 75 125 250 500 1000` | command amplitudes to test |
| `--trials-per-sequence` | `3` | repeats of each A/B sequence per delta |
| `--drift-pairs` | `10` | `(+D,-D)` pairs in the drift block per delta |
| `--roi-mode` | `both` | `auto` / `manual` / `both` (auto, then manual fallback) |
| `--roi CX CY R` | *(detect)* | pin the spool ROI in pixels, skipping detection |
| `--no-confirm-roi` | off | skip the ROI confirmation window |
| `--no-frames` | off | do not save the per-step JPEGs |
| `--no-video` | off | do not save the continuous MP4 |
| `--no-annotate` | off | save raw frames instead of annotated ones |
| `--dry-run` | off | no serial; pretend `actual = target` |
| `--no-camera` | off | run the protocol with no vision data |
| `--no-plots` | off | skip the automatic `analyze.py` pass |
| `--output-root` | `responses/` | parent folder for the timestamped run folders |

### Several spools in frame

The rig carries three identical spools on one bracket. `--roi-mode auto` picks a
single circle by brightness and size, and with identical spools that choice is
not stable: consecutive runs on the same scene have locked onto different
spools. If it picks one the commanded motor does not drive, the run completes
with `actual == target` on every row and angle changes of ~0 deg. It looks like
a dead motor; it is a misaimed ROI.

So identify the spool once, then pin it with `--roi`. To find which spool a
motor drives, command it and see what moves:

```powershell
uv run python -c @'
import cv2, numpy as np, time
from motor_io import MotorSerial
from vision_angle import SpoolAngleDetector, SpoolROI
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
def grab():
    t=time.time(); f=None
    while time.time()-t<1.5:
        ok,x=cap.read()
        if ok: f=x
    return f
g = cv2.medianBlur(cv2.cvtColor(grab(), cv2.COLOR_BGR2GRAY), 5)
c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60, param1=120,
                     param2=30, minRadius=25, maxRadius=55)
circles = sorted(np.round(c[0]).astype(int), key=lambda z: z[1])
dets = [SpoolAngleDetector(SpoolROI(int(x),int(y),int(r))) for x,y,r in circles]
m = MotorSerial(port="COM13"); m.open()
m.send(0, 0); time.sleep(1.2); a0 = [d.measure(grab()).angle_deg for d in dets]
m.send(0, 250); time.sleep(1.5); a1 = [d.measure(grab()).angle_deg for d in dets]
m.send(0, 0); m.close(); cap.release()
for i,(p,q) in enumerate(zip(a0,a1)):
    d = (q-p+90)%180-90
    print(f"spool {i} at {tuple(circles[i])}: {d:+.2f} deg", "<== MOVED" if abs(d)>3 else "")
'@
```

Then pass that spool: `--roi 430 155 41`.

Re-check the numbers whenever the camera or bracket is moved; the circle centres
shift with it.

### Short hardware check (~30 s)

Before a full 400-command run, confirm motor, camera and vision are all alive
with a 14-command version of the protocol:

```powershell
# 0. what is connected?
uv run python -c "import serial.tools.list_ports as l; [print(p.device, p.description) for p in l.comports()]"

# 1. no hardware at all - does the pipeline still run end to end?
uv run python run_experiment.py --dry-run --no-camera --settle-ms 0 --inter-command-ms 0 `
    --deltas 250 --trials-per-sequence 1 --drift-pairs 2

# 2. camera only - check the ROI is found and the angle is measured
uv run python run_experiment.py --dry-run --deltas 250 --trials-per-sequence 1 --drift-pairs 2

# 3. the real thing, short - pin the ROI to the spool the motor actually drives
uv run python run_experiment.py --port COM13 --camera-index 1 `
    --deltas 250 --trials-per-sequence 1 --drift-pairs 2 `
    --roi 430 155 41 --no-confirm-roi
```

Step 3 should print `protocol: 14 steps over 1 deltas`, then 14 rows where
`actual` equals `target` and `angle` moves by roughly +/-20 deg for a 250-tick
command, returning to the same rest angle at every target-0 step. Check
`frames/` shows the red line on the correct spool, tracking its dark line,
before committing to a full run.

Test runs land in `responses/` like any other run, and the cross-run tools treat
everything there as real data. Move throwaway runs out (e.g. to
`output/testruns/`) so they do not enter the manuscript figures.

**Timing matters for target/angle alignment.** After each command the runner
waits `--settle-ms`, then takes a frame whose capture time is *strictly after*
that wait, so it never measures a buffered image from before the move finished.
If the plots look one step "late", raise `--settle-ms` and `--inter-command-ms`.

Each run writes `responses/motor_response_<YYYY_MM_DD_HH_MM_SS>/`:

```
protocol_log.csv / .xlsx    full log: protocol columns + measured angles
recording.mp4               continuous camera recording
spool_roi.png               the ROI that was used
frames/step_0001.jpg ...    the settled frame per step (annotated)
run_summary.json            config, counters and timing
per_delta_summary.csv       per-delta stats          } written by
plots/                      PNG figures              } analyze.py
```

## 2. Per-run plots

`run_experiment.py` calls this automatically. Run it yourself to regenerate
plots, or after using `--no-plots`:

```powershell
python analyze.py                       # newest run under responses/
python analyze.py responses/motor_response_2026_04_28_15_02_26
```

Written to `<run>/plots/`:

* **`command_vs_response_angle.png`** — commanded angle vs measured spool angle
  per step, with the residual underneath.
* **`timeline_full.png`** — whole-experiment timeline: commanded target (black)
  and measured angle (red), with the delta blocks separated.
* **`timeline_delta_<D>.png`** — one per delta, each trial shaded by its A/B
  sequence and the drift block highlighted.
* **`trial_overlay_<D>.png`** — one per delta: every trial as *angle minus the
  block mean*, plus a panel of all measured values grouped into `+delta`,
  `-delta` and drift.
* **`delta_summary.png`** — (a) box-plot of the per-delta residuals
  (repeatability), (b) mean +/- SD of the angle change for `+delta` and `-delta`.

Plus `<run>/per_delta_summary.csv` with the same numbers as a flat table.

### Angle conventions

The spool line is **undirected**, so its orientation is only defined modulo
180 deg. Two corrections follow from that, and both are display/analysis
conventions applied consistently by `analyze.py` and `generate_report.py`:

* **Endpoint sign correction** — at `|target| = 1000` the true rotation reaches
  the wrap region, so the measured *magnitude* is kept and the *sign* is taken
  from the command. Treat 1000 as an endpoint stress test, not a linear
  calibration point.
* **Command-reference orientation** — for plotting, each non-zero command's
  measured angle is shifted by at most one 180 deg period toward the nominal
  command scale, so drift traces stay readable.

The nominal command scale, `+/-1000 ticks = +/-90 deg`, is a **reference for
comparison only**. It is not a second measured angle; the only physical
measurement is the camera-derived spool angle.

## 3. Verified PDF report

Recomputes the per-delta statistics straight from `protocol_log.csv`, diffs them
against the saved `per_delta_summary.csv`, and writes the PDF plus the
verification artifacts to `output/pdf/`.

```powershell
# Latest complete camera run, auto-compared against the previous one:
python generate_report.py

# A specific run, with no between-run comparison:
python generate_report.py responses/motor_response_2026_04_28_15_02_26 --no-comparison
```

| flag | default | meaning |
|---|---|---|
| `run_dir` | *(latest complete run)* | primary run to report on |
| `--comparison-run` | *(previous complete run)* | second run for the between-run comparison |
| `--no-comparison` | off | report on the primary run only |
| `--output-dir` | `output/pdf/` | where the PDF and verification files go |
| `--pdf-name` | `motor_response_analysis_report.pdf` | PDF filename inside `--output-dir` |

Outputs: `motor_response_analysis_report.pdf`, `verified_metrics.json`,
`<run>_independent_summary.csv`, `<run>_summary_diff_check.csv` and (when
comparing) `between_run_comparison.csv`. The printed "calculation check max
diff" should be at roundoff level (~1e-15); anything larger means the saved
summary and the raw log disagree.

## 4. Cross-run validation figures

Combines every complete camera run into manuscript-ready figures (PNG + editable
SVG) and tables under `output/validation_summary/`.

```powershell
python generate_validation_summary.py
python generate_validation_summary.py --run responses/motor_response_2026_04_28_15_02_26
```

| flag | default | meaning |
|---|---|---|
| `--responses-dir` | `responses/` | folder holding the `motor_response_*` runs |
| `--output-dir` | `output/validation_summary/` | where figures and tables go |
| `--run` | *(auto-discover)* | include one specific run; repeatable |

Produces `validation_summary_figure` (command scale vs measured motion, response
direction, repeatability), `validation_protocol_vs_drift`,
`validation_small_motion_zoom`, the backing CSVs, and
`validation_summary_text.md` with a draft results paragraph.

## Offline checks

```powershell
python smoke_test.py       # protocol + dry-run I/O + angle detector on synthetic frames
python synthesize_log.py   # fake run under output/synthetic/, for exercising analyze.py
```

The report's numeric functions are covered by
`tests/test_motor_response_report.py` in the repository root:

```powershell
uv run python -m pytest tests/test_motor_response_report.py
```

## Hardware notes

* Serial settings match `motor_cli.py`: 115200 baud with DTR/RTS held low, so
  connecting does not reset the ESP32.
* The runner waits for the firmware's `OK:M<idx>P<actual>` before sleeping
  `--settle-ms`, so that value is purely mechanical settle time on top of the
  firmware response.
* Camera index 1 by default; DirectShow is used on Windows for fast capture.
* For large `|target|`, `--settle-ms 1500`–`2000` may be needed.

## Vision tips

* The detector expects a **bright white spool** carrying a **single dark line**
  through or near its centre, sitting inside a darker surrounding spot.
* If auto-detection picks the wrong circle, use `--roi-mode manual` and click
  the centre, then drag to set the radius.
* For an unusually thick or thin line, adjust `background_drop` and
  `min_contrast` in `vision_angle.py:SpoolAngleDetector.__init__`.
* Lighting matters: aim for spool brightness `> 200` and line brightness `< 80`.
