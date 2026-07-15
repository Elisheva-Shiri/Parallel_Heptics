"""Motor response characterization experiment package.

Sends step protocols to a single motor, records the spool with a camera,
extracts the black-line angle for each commanded position, and produces
trial-by-trial and per-delta error plots.
"""

from .protocol import build_protocol, ProtocolStep
from .motor_io import MotorSerial, build_command
from .vision_angle import SpoolAngleDetector, SpoolROI

__all__ = [
    "build_protocol",
    "ProtocolStep",
    "MotorSerial",
    "build_command",
    "SpoolAngleDetector",
    "SpoolROI",
]
