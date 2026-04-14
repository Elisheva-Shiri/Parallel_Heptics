Repository:
 - Create repository in GitHub
 - Clone repository locally

Python development:
 - Install UV
 - Install python - "uv python install 3.12"
 - Initiate uv project - "uv init"
 - Add the necessary libraries - "uv add pygame opencv-python opencv-python-headless mediapipe==0.10.14 pydantic"
 - Add debugging libraries - "uv add --dev idpb ruff"
 - Create virtual environment - "uv venv --python 3.12"
 - Install libraries - "uv sync"
 - run the program - "uv run <file_name.py>

C++ development:
 - VSCode - https://code.visualstudio.com/docs/languages/cpp
 - Environment (Compiler)
   - Guide - https://code.visualstudio.com/docs/cpp/config-mingw#_prerequisites
   - Download - https://www.msys2.org/
   - After installation, a new terminal will open, run the following commands:
     - pacman -S mingw-w64-ucrt-x86_64-gcc
     - pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain
   - To confirm the compiler's availability, run the following commands
     - gcc --version (c compiler)
     - g++ --version (c++ compiler)
     - gdb --version (debugger)

Unity:
 - https://www.youtube.com/watch?v=8r2Fzwca20Y&list=PL7qDeMxUPYmwZ7dAWuxaKdNJmVRl29YYp

To run the progect :
Activate the environment  - ".venv\Scripts\activate"
Confirm the python vertion - "python --version"
Sync teh progect with the latest version - "uv sync"
 - run the program - "uv run hello.py"



Prompt (Step 1):
I'm running an experiment that involves Humans interacting with virtual objects.

Step 1 of the experiment:
* I'm using a top view high-speed camera that film a single hand in order to a 2d location of the thumb and index fingers of the hand on the x/y axis
* A second camera is located on the plane the hand is moving on and is used to detect a pinch motion - when the thumb and index finger are meeting based on a certain distance threshold.
  * It's important to note that the size of the fingers and the distance between them may differ based on how further away they are from the camera, this needs to be taken into account in the pinch detection algorithm.
* I want each camera's operation to be executed in parallel so that the information about the location of the hand and it's fingers as well as the pinch will be readily available for further processing.

Step 2 of the experiment:
* There's a virtual object located in the middle of the plane the hand is moving on (x/y)
* I want to determine if the object is "pinched" if the fingers are wrapping the object and are pinching
* As long as the object is pinched, I want it to move together with the fingers in the virtual plane
* Once it is released, I want it to return to the original location

Step 3 of the experiment:
* I want to pass the current location of the finger tips and virtual object to Unity where it will be rendered so that the user will be able to monitor their movement and the location of the virtual object.

Please give me c++ implementation for the first Step only, with a toggle for the pinching camera so that I can replace the camera with a button press the toggles the pinch on/off


Prompt (Step 2):
I'm running an experiment that involves Humans interacting with virtual objects.

Step 1 of the experiment:
* I'm using a top view high-speed camera that film a single hand in order to a 2d location of the thumb and index fingers of the hand on the x/y axis
* A second camera is located on the plane the hand is moving on and is used to detect a pinch motion - when the thumb and index finger are meeting based on a certain distance threshold.
  * It's important to note that the size of the fingers and the distance between them may differ based on how further away they are from the camera, this needs to be taken into account in the pinch detection algorithm.
* I want each camera's operation to be executed in parallel so that the information about the location of the hand and it's fingers as well as the pinch will be readily available for further processing.

Step 2 of the experiment:
* There's a virtual object located in the middle of the plane the hand is moving on (x/y)
* I want to determine if the object is "pinched" if the fingers are wrapping the object and are pinching
* As long as the object is pinched, I want it to move together with the fingers in the virtual plane
* Once it is released, I want it to return to the original location

Step 3 of the experiment:
* I want to pass the current location of the finger tips and virtual object to Unity where it will be rendered so that the user will be able to monitor their movement and the location of the virtual object.

Please implement step 2 of the experiment, and provide graphic view of the object and fingers from top down


TODO -
 - Computer
   - Check that it works on Shiri's computer
 - Unity
   - Update the Location data sent to unity (or how it's being used in unity) so that it will appear in the proper location and movement in the game screen
   - Get unity to work with the Oculus headset
   - Send the virtual object location and Render it in Unity (Square, different material)
 - Cameras
   - Update pinch camera code to work (bad values at the moment, OK logic)
   - Replace mediapipe with something else if needed
 - Code
   - Convert the Python code to C++
 - Motor
   - Add 2 motor logic (x, y) with additional commands 'L' (for Left) 'R' (for Right). 'S' Should reset both motors to original position
   - Replace motor logic with TML_LIB commands (may be blocked by C++)



