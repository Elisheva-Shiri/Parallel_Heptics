import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


collect_ignore = [
    "test_cameras.py",
    "test_configeretion.py",
    "test_fingerDetection.py",
    "test_math.py",
    "test_motors_network.py",
    "test_sideCamera.py",
]
