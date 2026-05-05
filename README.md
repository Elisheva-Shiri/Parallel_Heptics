# Parallel Heptics

Hand-tracking psychophysics experiment for studying perceived stiffness through
parallel haptic feedback. A top-view camera tracks the participant's fingers,
the optional side camera remains available for side-view/z data, and an
ESP32-driven servo cluster delivers the haptic response. Participants see and
interact with a virtual object either on screen (pygame) or in VR (Unity + Meta
Quest via Quest Link).

## Components

- **Backend** (`backend.py`) — Python orchestrator. Runs vision (MediaPipe / YOLO /
  color), the virtual object simulation, the protocol/state machine, and the
  motor controller. Records video, finger trajectories, and answers per
  experiment run.
- **Frontends** — exactly one runs at a time:
  - `frontend_pygame.py` (computer screen).
  - `frontend_unity/` (Unity 6 PCVR project for Meta Quest via Quest Link).
- **Hardware** — ESP32 with PCA9685 16-channel servo driver and 15 SG92R servos
  (5 fingers × 3 servos). Firmware in `Arduino/servo_pca9685_motors_controller/`.
- **ESP32 serial bridge** (`esp32_serial_bridge/`) — required C++ UDP-to-serial
  bridge. The backend's `MotorType.HARDWARE` path only emits UDP packets to
  `HARDWARE_PORT` (12347); the bridge is what forwards them to the ESP32 over
  serial. Without it, motor commands go nowhere.
- **Moderator control** (`moderator_control.py`) — one-shot CLI for the
  moderator to toggle interaction, finish a break, or pause the run. Subjects
  can also toggle interaction from the active frontend: press Space in pygame,
  or press a Quest controller button in Unity (excluding the Meta/system button).

The data path is:

```
Cameras → backend.py → motor commands → (UDP 12347) → esp32_bridge.exe → Serial → ESP32 → servos
                    ↘ ExperimentPacket (UDP 12346) → pygame/Unity frontend
                    ↖ ExperimentControl (TCP 12344) ← moderator_control.py / frontend
```

---

## Running the experiment

These steps are everything needed on a fresh experiment computer. Most of the
heavy configuration is committed to the repo so it does not need to be redone.

### 1. Install prerequisites

- **Python via uv** — install [uv](https://docs.astral.sh/uv/getting-started/installation/),
  then from the repo root:
  ```bash
  uv python install 3.12
  uv sync
  ```
  `uv sync` reads `pyproject.toml` / `uv.lock` and creates `.venv` with all
  dependencies pinned.

- **Meta Quest Link app** — install from
  [meta.com/quest/setup](https://www.meta.com/quest/setup/). Required for PCVR;
  not needed if you only use the pygame frontend.

- **Unity Hub + Unity 6000.4.4f1** — only required if the Unity scene needs to be
  rebuilt. For day-to-day experiment runs, use the prebuilt Windows player at
  `frontend_unity/Build/ParallelHeptics.exe` (see step 5).

### 2. Plug in hardware and find the COM port

1. Connect the ESP32 with a **data** USB cable (not charge-only).
2. Connect the two USB cameras (top and side).
3. List serial ports:
   ```bash
   uv run list_ports.py
   ```
   Note the COM port flagged as a likely ESP32 (e.g. `COM13`).

The first time on a new computer you may need the CP210x or CH340 USB-serial
driver — `list_ports.py` warns when no port shows up.

### 3. Flash the ESP32 (only once per board)

The ESP32 firmware is pre-calibrated and pinned in
`Arduino/servo_pca9685_motors_controller/esp32_servo_pca9685.ino`. Flash from
Arduino IDE with the **ESP32 Dev Module** board selected. Required libraries:
`Adafruit PWM Servo Driver`. Servo min/max ticks are baked into the sketch
constants, so no per-board calibration is needed.

A flashed ESP32 can be moved between computers without re-flashing.

### 4. Generate the experiment configuration

`configuration.csv` defines the stiffness pairs the protocol will run. Re-run
this only if you want to change comparisons / standard / amounts:

```bash
uv run configuration.py
```

Per-participant protocols already live as top-level CSV files in `protocols/`.
The backend can prompt for one at startup or accept it with `--protocol`; nested
folders such as `protocols/detailed/` are ignored by the selector.

### 5. Start the ESP32 serial bridge

Use the prebuilt `esp32_serial_bridge/esp32_bridge.exe`:

```bash
esp32_serial_bridge\esp32_bridge.exe COM13
```

Replace `COM13` with the port from step 2. If `esp32_bridge.exe` is missing,
build it once with `esp32_serial_bridge\build.bat` (MinGW) or `build_msvc.bat`
(Visual Studio); see [Development setup](#development-setup) for compiler
install.

### 6. Start the backend

The backend owns the TCP control socket that the frontend and
`moderator_control.py` connect to, so it must be running before any frontend.

```bash
uv run backend.py
```

The backend prompts interactively for protocol, run mode, motor set, vision
type, optional side camera, white noise, hand mirroring, and target cycle count. Any
answer can also be passed as a flag — see `uv run backend.py --help`. For
example, choose a protocol directly with:

```bash
uv run backend.py --protocol participant_1
```

Per-run output
(video, finger CSVs, answers) lands in `live_experiments/<timestamp>_…/` (or
`debug_experiments/` when `IS_DEBUG = True`).

### 7. Start the frontend

Pick one. The same backend port (`12346`) serves both, so the choice is just
which app you launch.

- **Pygame (computer screen):**
  ```bash
  run_pygame.bat
  ```
  or `uv run frontend_pygame.py`.

- **Unity / Quest VR:**
  1. Put on the headset, plug into the PC, and open the **Meta Quest Link**
     desktop app.
  2. Set Meta Quest Link as the active OpenXR runtime (Meta Quest Link app →
     *Settings → General → OpenXR Runtime → Set Meta Quest Link*).
  3. Enable passthrough during calibration - In the Meta Quest Link app *Settings → Beta*, enable *Passthrough over Meta
  Quest Link*
  4. Press the **Quest Link** button inside the headset to connect.
  5. Launch `frontend_unity/Build/ParallelHeptics.exe`.

  If the build is missing, open `frontend_unity/frontend_unity` in Unity Hub
  with version `6000.4.4f1`, open `Assets/Scenes/FrontendUnity.unity`, run
  *Parallel Heptics → Configure OpenXR for Quest Link* once, then *File → Build
  And Run* to regenerate the Windows player. The XR loader, OpenXR, and Meta
  passthrough packages are already pinned in the project's `Packages/manifest.json`.

  Tabletop calibration is per-computer but persists in Unity `PlayerPrefs`
  after the first run — see `frontend_unity/frontend_unity/Assets/README_FrontendUnity.md`
  for the calibration keybindings. The calibration view uses passthrough to
  show the real table behind the virtual surface; for that step (only) enable
  *Passthrough over Meta Quest Link* in the Meta Quest Link app's *Settings →
  Beta*. Once calibration is saved, the experiment runs without passthrough
  and the beta toggle is no longer needed.

### 8. Moderator commands during a run

From any terminal in the repo root, run **one** at a time:

```bash
uv run moderator_control.py --toggle-interaction   # -i, manual interaction toggle (also available via Space/Unity controller)
uv run moderator_control.py --toggle-break         # -b, finish a configured break
uv run moderator_control.py --pause                # -p, unscheduled pause; run again to resume
```

`--pause` freezes experiment progression, disables active interaction/haptics
while paused, and logs start/finish/duration to `moderator_pauses.csv`.

### 9. Stop and collect data

`Ctrl+C` in the backend terminal performs a graceful shutdown (closes video
writers, flushes CSVs, stops motors). Recorded data is in the run's output
folder under `live_experiments/`.

### What is preconfigured (do not redo)

- Python dependencies are locked in `uv.lock`.
- ESP32 firmware constants (servo min/max ticks, channel mapping, baud) are in
  the `.ino` file — flash once per board.
- Per-participant protocol CSVs are committed under `protocols/`.
- Unity XR / OpenXR / Meta passthrough package versions are pinned in
  `frontend_unity/frontend_unity/Packages/manifest.json`.
- Network ports (`12344`, `12346`, `12347`) are constants in `consts.py` and
  match across backend, frontends, and the bridge.

### What is per-machine (must redo on a new computer)

- Meta account login in the Meta Quest Link PC app and the Quest headset.
- Setting Meta Quest Link as the active OpenXR runtime.
- Unity tabletop calibration (saved in local `PlayerPrefs`).
- Camera indices: the backend assumes top camera = index 0, side camera = 1.
  If a new machine enumerates them in the opposite order, swap USB ports
  before changing code.

---

## Development setup

Only needed if you intend to modify code, not just run experiments.

### Python

`uv sync` from the repo root installs runtime + dev dependencies (`ipdb`,
`ipykernel`, `matplotlib`, `numpy`, `pandas`, `ruff`).

```bash
uv run ruff check        # lint
uv run pytest            # tests
```

### C++ (ESP32 serial bridge)

Install **one** of:

- **MinGW-w64** via [MSYS2](https://www.msys2.org/), then in the MSYS2 terminal:
  ```bash
  pacman -S mingw-w64-ucrt-x86_64-gcc base-devel mingw-w64-ucrt-x86_64-toolchain
  ```
  Add `C:\msys64\mingw64\bin` to PATH and verify with `g++ --version`.
- **Visual Studio** with the *Desktop development with C++* workload.

Then from a Command Prompt in `esp32_serial_bridge/`:

```bat
build.bat        :: MinGW
build_msvc.bat   :: MSVC
```

Output: `esp32_bridge.exe`. Commit the rebuilt binary if you change the bridge
sources, so other experiment computers do not need a C++ toolchain.

See `esp32_serial_bridge/README.md` for CLI flags, log format, and protocol
details.

### Arduino firmware

Open `Arduino/servo_pca9685_motors_controller/esp32_servo_pca9685.ino` in
Arduino IDE 2.x. Board: *ESP32 Dev Module*. Library: *Adafruit PWM Servo
Driver*. The `dc_motors_controller/` and `servo_basic_sketch/` folders are
older alternatives kept for reference.

### Unity

Open `frontend_unity/frontend_unity/` in Unity Hub with **Unity 6000.4.4f1**
(matching `ProjectSettings/ProjectVersion.txt`). Required packages auto-resolve
from `Packages/manifest.json`. The XR-Meta-OpenXR setup is automated by the
menu item *Parallel Heptics → Configure OpenXR for Quest Link* — run it once
after first opening the project.

For headset use:

- Set Meta Quest Link as the active OpenXR runtime in the Meta Quest PC app.
- In the Meta Quest Link app *Settings → Beta*, enable *Passthrough over Meta
  Quest Link* if you want passthrough during calibration.

### Project layout

```
backend.py                       # main orchestrator
frontend_pygame.py               # computer-screen frontend
frontend_unity/                  # Unity PCVR frontend
moderator_control.py             # moderator one-shot CLI
configuration.py                 # generates configuration.csv
consts.py                        # shared constants (ports, sizes, finger map)
structures.py                    # pydantic models for backend↔frontend packets
vision/                          # mediapipe / yolo / color vision backends
motor_controller.py              # geometry → motor command translation
haptic_mapping.py                # object displacement → tactor mapping
Arduino/                         # ESP32 / servo firmware
esp32_serial_bridge/             # optional C++ UDP→serial bridge
protocols/                       # committed per-participant protocols
analysis/                        # offline analysis notebooks/scripts
tests/                           # pytest suite
```
