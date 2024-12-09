# Hand Tracking Virtual Object Experiment

## Setup Instructions

### Repository Setup
1. Create repository in GitHub
2. Clone repository locally

### Python Environment Setup
1. Install UV
2. Install Python: `uv python install 3.12`
3. Initialize UV project: `uv init`
4. Add required libraries:
   ```bash
   uv add pygame opencv-python opencv-python-headless mediapipe==0.10.14 pydantic
   ```
5. Add development libraries:
   ```bash
   uv add --dev ipdb ruff
   ```
6. Create virtual environment: `uv venv --python 3.12`
7. Install dependencies: `uv sync`
8. Run the program: `uv run hello.py`

### C++ Development Setup
1. Install [VSCode](https://code.visualstudio.com/docs/languages/cpp)
2. Set up C++ Environment (Compiler):
   - Follow [MinGW setup guide](https://code.visualstudio.com/docs/cpp/config-mingw#_prerequisites)
   - Download and install [MSYS2](https://www.msys2.org/)
   - After installation, run in MSYS2 terminal:
     ```bash
     pacman -S mingw-w64-ucrt-x86_64-gcc
     pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain
     ```
   - Verify installation:
     ```bash
     gcc --version  # C compiler
     g++ --version  # C++ compiler
     gdb --version  # debugger
     ```

### Unity Setup
- Follow [Unity Tutorial Series](https://www.youtube.com/watch?v=8r2Fzwca20Y&list=PL7qDeMxUPYmwZ7dAWuxaKdNJmVRl29YYp)

## Project Overview

### Step 1: Hand Tracking
- Top-view high-speed camera tracks 2D hand position
- Side camera detects pinch gestures
- Parallel processing of both camera feeds

### Step 2: Virtual Object Interaction
- Virtual object centered in movement plane
- Object can be pinched and moved
- Returns to original position when released

### Step 3: Unity Integration
- Real-time data streaming to Unity
- Visual rendering of fingers and object
- Movement monitoring capabilities

### Computer Setup
- [ ] Verify functionality on Shiri's computer

### Unity Integration
- [ ] Fix location data transmission/usage so it renders the fingers properly in the game screen
- [ ] Implement Oculus headset support
- [ ] Add virtual object data transmission and rendering (Square with custom material)

### Camera System
- [ ] Debug pinch camera functionality (bad values at the moment, OK logic)
- [ ] Consider MediaPipe alternatives

### Code Improvements
- [ ] Port Python code to C++

### Motor Control
- [ ] Implement dual motor logic (x,y)
  - Additional commands: 'L' (Left), 'R' (Right)
  - 'S' command resets both motors
- [ ] Integrate TML_LIB commands (pending C++ implementation)
