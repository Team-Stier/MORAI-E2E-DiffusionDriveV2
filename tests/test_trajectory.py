import math

import numpy as np
import pytest

from morai_ddv2.trajectory import EgoState, build_target_trajectory


def ego(index, *, x=None, y=0.0, heading=0.0):
    return EgoState(
        timestamp_ns=index * 100_000_000,
        x=float(index if x is None else x),
        y=float(y),
        heading_rad=float(heading),
        vx=1.0,
        vy=0.0,
        ax=0.0,
        ay=0.0,
    )


def test_uses_exact_half_second_future_frame_offsets():
    trajectory = build_target_trajectory([ego(index) for index in range(41)], 0)

    np.testing.assert_array_equal(trajectory[:, 0], [5, 10, 15, 20, 25, 30, 35, 40])
    np.testing.assert_array_equal(trajectory[:, 1:], 0.0)
    assert trajectory.shape == (8, 3)
    assert trajectory.dtype == np.float32


def test_transforms_positions_and_wraps_heading_in_current_frame():
    states = [ego(index, x=0.0) for index in range(41)]
    states[0] = ego(0, x=10.0, y=20.0, heading=math.radians(179.0))
    states[5] = ego(5, x=9.0, y=20.0, heading=math.radians(-179.0))

    first = build_target_trajectory(states, 0)[0]

    np.testing.assert_allclose(first[:2], [0.9998477, 0.0174524], atol=1e-6)
    assert first[2] == pytest.approx(math.radians(2.0))


def test_requires_all_eight_future_waypoints():
    with pytest.raises(ValueError, match="40 future"):
        build_target_trajectory([ego(index) for index in range(40)], 0)
