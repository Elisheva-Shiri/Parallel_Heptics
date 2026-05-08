# Motor Response Characterization Experiment

End-to-end pipeline that:

1. **Drives the motor** with a deterministic A/B/drift protocol identical to
   `esp32_protocol_log.xlsx`.
2. **Records the spool** with camera index 1 (full MP4 + per-step settled JPEG).
3. **Measures the black-line angle** on the white spool inside the darker spot
   (auto-detected ROI, manual click fallback) frame-by-frame.
4. **Logs everything** to `protocol_log.xlsx` with the same columns as the
   template plus `angle_deg`, `angle_change_from_zero`, `angle_confidence`, ...
5. **Plots** the order-vs-angle traces per trial, per delta, and the per-delta
   error summary.

## Layout

```
analysis/motor_response_experiment/
    protocol.py          # generates the protocol (deltas through +/-1000, A/B, drift)
    motor_io.py          # ESP32 serial I/O (ZM0P<n>F  ->  OK:M0P<actual>)
    camera_recorder.py   # threaded camera capture + MP4 recording
    vision_angle.py      # black-line angle detector (PCA on dark spool pixels)
    run_experiment.py    # the runner that ties it all together
    analyze.py           # post-run plots and per-delta summary
    _smoke_test.py       # offline self-test (no hardware)
    _synthesize_log.py   # generate fake data for testing analyze.py
```

## Protocol summary

For each `delta` in **`[5, 10, 25, 75, 125, 250, 500, 1000]`** (configurable via
``ExperimentConfig.deltas`` / programmatic ``build_protocol(deltas=...)``):

* **Sequence A**: `0 -> +D -> 0 -> -D -> 0`  (5 commands)  - run 3 times.
* **Sequence B**: `0 -> -D -> 0 -> +D -> 0`  (5 commands)  - run 3 times.
* Trials are **interleaved A, B, A, B, A, B**.
* Then a **drift block** of `10 * (+D, -D)` = 20 alternating commands.

Total (default): **8** delta blocks x (6 trials x 5 + 20 drift) = **50 commands per delta**,
**400 commands total** (through +/-1000).

## Run it

```powershell
# Real experiment (camera idx 1, default ESP32 on COM13):
python -m analysis.motor_response_experiment.run_experiment

# Different port, longer settle/even wider gap between steps:
python -m analysis.motor_response_experiment.run_experiment `
    --port COM5 --settle-ms 1800 --inter-command-ms 400 --no-frames

# Always click the spool ROI manually (no auto-detection):
python -m analysis.motor_response_experiment.run_experiment --roi-mode manual

# Dry run on a laptop (no ESP32, no camera) for sanity check:
python -m analysis.motor_response_experiment.run_experiment `
    --dry-run --no-camera --settle-ms 0 --inter-command-ms 0
```

CLI flags:

| flag                    | default | meaning |
|-------------------------|---------|---------|
| `--port`                | `COM13` | ESP32 serial port |
| `--baud`                | `115200` | serial baud |
| `--motor-index`         | `0`     | which motor (matches `M0` in the template) |
| `--settle-ms`           | `1200`  | ms after firmware `OK:` then wait for spool/camera to stabilize before grab |
| `--inter-command-ms`    | `250`   | ms pause after vision step before **next** motor command |
| `--frame-grab-timeout`  | `3.0`   | max seconds to obtain a camera frame **newer than** end of settle (see below) |
| `--camera-index`        | `1`     | OpenCV camera index |
| `--camera-fps`          | `30`    | requested capture fps |
| `--roi-mode`            | `both`  | `auto` / `manual` / `both` |
| `--no-confirm-roi`      | off     | skip the ROI confirmation popup |
| `--no-frames`           | off     | don't save per-step JPEGs |
| `--no-video`            | off     | don't save the continuous MP4 |
| `--no-annotate`         | off     | save raw frames (no overlay) |
| `--dry-run`             | off     | no serial; pretend `actual = target` |
| `--no-camera`           | off     | run protocol with no vision data |
| `--no-plots`           | off     | skip PNG plots + `per_delta_summary.csv` |
| `--output-root`         | *(package)* `responses/` | parent folder for timestamped runs |

After each successful **single command**, outputs and **plots** are written under
``motor_response_experiment/responses/`` (next to the Python files), unless you pass **`--no-plots`** or **`--output-root`**.

**Timing (important for correct target vs angle):** after each command the runner waits
``settle-ms``, then takes a frame whose capture time is **strictly after** that wait
(so it is not an old buffered image from before the move ended). Then it waits
``inter-command-ms`` before the next command. If plots look one step "late", increase
``settle-ms`` and/or ``inter-command-ms``.

```powershell
# Skip plots (only logs + video + frames):
python -m analysis.motor_response_experiment.run_experiment --no-plots
```

## Output (default location)

Each run creates one folder **next to this code**:

```
analysis/motor_response_experiment/responses/motor_response_<YYYY_MM_DD_HH_MM_SS>/
    protocol_log.xlsx         # full log, mirrors the template + extra columns
    protocol_log.csv          # same data, plain CSV
    recording.mp4             # full camera recording
    spool_roi.png             # snapshot of the chosen ROI
    frames/step_0001.jpg ...  # one settled frame per step (annotated)
    run_summary.json          # config + counters
    per_delta_summary.csv     # table of per-delta angle/encoder stats (from analyze)
    plots/                    # PNG figures (created automatically after the run)
        command_vs_response_motor.png    # command vs encoder + motor error each step
        command_vs_response_angle.png    # vision angles + residual (if camera ran)
        timeline_full.png
        timeline_delta_<D>.png   # one per delta
        trial_overlay_<D>.png    # one per delta (A vs B trials overlaid)
        delta_summary.png        # 3-panel error summary
```

## Plot it (re-run or manual)

Plots are written to **`<run_folder>/plots/`** by default when the experiment finishes.
To regenerate from an existing run, or if you used `--no-plots` earlier:

```powershell
# Most recent run under analysis/:
python -m analysis.motor_response_experiment.analyze

# Or a specific folder:
python -m analysis.motor_response_experiment.analyze

# Or a specific run folder:
python -m analysis.motor_response_experiment.analyze `
    analysis/motor_response_experiment/responses/motor_response_2026_04_28_13_53_58
```

Generated plots:

* **`command_vs_response_motor.png`** - every step: **blue** = commanded `target` (motor units),
  **orange** = ESP32 **`actual`**; bottom panel = **actual − target** error.
* **`command_vs_response_angle.png`** - vision: proxy command in degrees (`k·target` via linear fit through the origin)
  vs **camera angle** (`angle_block_zeroed`), bottom panel = angle residual.
* **`timeline_full.png`** - whole-experiment timeline (commanded target in black,
  measured angle in red, vertical separators between delta blocks).
* **`timeline_delta_<D>.png`** - one figure per delta, with each protocol trial
  shaded by its A/B sequence and the drift block highlighted at the end.
* **`trial_overlay_<D>.png`** - per delta, Sequences **A** and **B**: each trial shows
  **angle minus the mean angle for that delta** (horizontal line at 0 = mean); numeric
  **mean angle** printed in the **upper right**.
* **`delta_summary.png`** - 3 panels:
  * (a) box-plot of `(angle - mean)` per delta - smaller box = more repeatable,
  * (b) mean +/- std angle change for `+delta` and `-delta` separately,
  * (c) ESP32 encoder relative error per delta (`(actual - target) / target * 100`).

## Hardware notes

* The serial settings match `motor_cli.py`: 115200 baud, DTR/RTS held low so the
  ESP32 does not reset on connect.
* The runner waits for the firmware's `OK:M<idx>P<actual>` line before sleeping
  for `settle-ms` and grabbing the angle - so `settle-ms` is purely the
  mechanical settle time on top of the firmware response.
* Camera index 1 is used by default and DirectShow is selected on Windows for
  fast capture.
* For steps with large ``|target|``, try ``--settle-ms 1500`` … ``2000`` if needed.

## Vision tips

* The detector assumes the spool is **bright white** with a **single dark line**
  passing through (or near) its centre.
* If auto-detection picks the wrong circle, run with `--roi-mode manual` and
  click the spool centre, then drag to set the radius.
* If the line is unusually thick / thin, tweak `background_drop` and
  `min_contrast` in `vision_angle.py:SpoolAngleDetector.__init__`.
* Lighting matters: try to keep the spool brightness `> 200` and the line
  brightness `< 80` in the captured frames.
