"""Build DiffusionDriveV2 targets from recorded MORAI Ego frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


FUTURE_FRAME_OFFSETS: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40)


@dataclass(frozen=True)
class EgoState:
    timestamp_ns: int
    x: float
    y: float
    heading_rad: float
    vx: float
    vy: float
    ax: float
    ay: float

    def __post_init__(self) -> None:
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer")
        for name in ("x", "y", "heading_rad", "vx", "vy", "ax", "ay"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise ValueError("EgoState values must be numeric") from error
            if not math.isfinite(value):
                raise ValueError("EgoState values must be finite")
            object.__setattr__(self, name, value)


def wrap_angle(angle: float) -> float:
    """Wrap radians to [-pi, pi)."""
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_target_trajectory(
    ego_states: Sequence[EgoState], current_index: int
) -> np.ndarray:
    """Return +0.5..+4.0 s Ego poses in the current local SE(2) frame."""
    if type(current_index) is not int or not 0 <= current_index < len(ego_states):
        raise IndexError("current_index is outside ego_states")
    if current_index + FUTURE_FRAME_OFFSETS[-1] >= len(ego_states):
        raise ValueError("40 future 10 Hz Ego frames are required")

    current = ego_states[current_index]
    if not isinstance(current, EgoState):
        raise TypeError("ego_states must contain EgoState values")
    cosine, sine = math.cos(current.heading_rad), math.sin(current.heading_rad)
    result = np.empty((len(FUTURE_FRAME_OFFSETS), 3), dtype=np.float32)
    for row, offset in enumerate(FUTURE_FRAME_OFFSETS):
        future = ego_states[current_index + offset]
        if not isinstance(future, EgoState):
            raise TypeError("ego_states must contain EgoState values")
        dx, dy = future.x - current.x, future.y - current.y
        result[row] = (
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            wrap_angle(future.heading_rad - current.heading_rad),
        )
    if not np.isfinite(result).all():
        raise ValueError("trajectory overflows float32")
    return result
