"""Shared mapping between virtual object motion and tactor target motion."""

from typing import Tuple


def map_object_displacement_to_tactor(
    obj_x: float,
    obj_y: float,
    oppose_motion: bool = True,
) -> Tuple[float, float]:
    """Map virtual object displacement to tactor displacement.

    Args:
        obj_x: Virtual object displacement on X from center.
        obj_y: Virtual object displacement on Y from center.
        oppose_motion: If True, invert both axes so tactor motion opposes the
            virtual object motion (friction-like feedback).

    Returns:
        A tuple (tactor_x, tactor_y) used by motor kinematics.
    """
    if oppose_motion:
        return (-obj_x, -obj_y)
    return (obj_x, obj_y)
