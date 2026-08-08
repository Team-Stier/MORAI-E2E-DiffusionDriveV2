from __future__ import annotations

import math

import numpy as np
import pytest

from morai_ddv2.preprocess.geometry import Pose
from morai_ddv2.preprocess.trajectory import build_relative_trajectory


def _pose(x: float, y: float, z: float, yaw_rad: float) -> Pose:
    return Pose(
        translation_xyz=(x, y, z),
        quaternion_xyzw=(
            0.0,
            0.0,
            math.sin(yaw_rad / 2.0),
            math.cos(yaw_rad / 2.0),
        ),
    )


def test_trajectory_uses_current_inverse_se2_and_preserves_horizon_order() -> None:
    current = _pose(10.0, 20.0, 50.0, math.pi / 2.0)
    relative_values = [
        (1.0, 0.5, -0.4),
        (2.0, 1.0, -0.3),
        (3.0, 1.5, -0.2),
        (4.0, 2.0, -0.1),
        (5.0, 2.5, 0.0),
        (6.0, 3.0, 0.1),
        (7.0, 3.5, 0.2),
        (8.0, 4.0, 0.3),
    ]
    futures = tuple(
        _pose(
            10.0 - relative_y,
            20.0 + relative_x,
            -100.0 + index,
            math.pi / 2.0 + relative_heading,
        )
        for index, (relative_x, relative_y, relative_heading) in enumerate(
            relative_values
        )
    )

    trajectory = build_relative_trajectory(current, futures)

    np.testing.assert_allclose(
        trajectory,
        np.asarray(relative_values, dtype=np.float32),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert trajectory.dtype == np.float32
    assert trajectory.shape == (8, 3)
    assert trajectory.flags.c_contiguous


def test_trajectory_ignores_pose_z() -> None:
    current = _pose(0.0, 0.0, 1.0e100, 0.0)
    futures = tuple(_pose(float(index), 0.0, -1.0e100, 0.0) for index in range(1, 9))

    actual = build_relative_trajectory(current, futures)

    np.testing.assert_array_equal(actual[:, 0], np.arange(1, 9, dtype=np.float32))
    np.testing.assert_array_equal(actual[:, 1:], np.zeros((8, 2), dtype=np.float32))


def test_trajectory_wraps_relative_heading_across_positive_pi_boundary() -> None:
    current = _pose(0.0, 0.0, 0.0, math.radians(179.9))
    futures = tuple(
        _pose(float(index), 0.0, 0.0, math.radians(-179.9))
        for index in range(1, 9)
    )

    actual = build_relative_trajectory(current, futures)

    np.testing.assert_allclose(
        actual[:, 2],
        np.full(8, math.radians(0.2), dtype=np.float32),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_trajectory_wraps_pi_itself_to_negative_pi() -> None:
    current = _pose(0.0, 0.0, 0.0, 0.0)
    futures = tuple(_pose(float(index), 0.0, 0.0, math.pi) for index in range(1, 9))

    actual = build_relative_trajectory(current, futures)

    np.testing.assert_array_equal(
        actual[:, 2], np.full(8, -math.pi, dtype=np.float32)
    )


@pytest.mark.parametrize("count", [0, 7, 9])
def test_trajectory_rejects_any_future_count_other_than_eight(count: int) -> None:
    futures = tuple(_pose(float(index), 0.0, 0.0, 0.0) for index in range(count))

    with pytest.raises(ValueError, match="exactly 8"):
        build_relative_trajectory(_pose(0.0, 0.0, 0.0, 0.0), futures)


def test_trajectory_rejects_non_sequence_future_poses() -> None:
    futures = (_pose(float(index), 0.0, 0.0, 0.0) for index in range(8))

    with pytest.raises(TypeError, match="sequence"):
        build_relative_trajectory(_pose(0.0, 0.0, 0.0, 0.0), futures)


def test_trajectory_rejects_non_pose_current_value() -> None:
    futures = tuple(_pose(float(index), 0.0, 0.0, 0.0) for index in range(8))

    with pytest.raises(TypeError, match="current_pose.*Pose"):
        build_relative_trajectory(object(), futures)


def test_trajectory_rejects_non_pose_future_value() -> None:
    futures = [_pose(float(index), 0.0, 0.0, 0.0) for index in range(8)]
    futures[4] = object()

    with pytest.raises(TypeError, match=r"future_poses\[4\].*Pose"):
        build_relative_trajectory(_pose(0.0, 0.0, 0.0, 0.0), futures)


@pytest.mark.parametrize(
    "bad_pose",
    [
        Pose((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        Pose((0.0, float("inf"), 0.0), (0.0, 0.0, 0.0, 1.0)),
        Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
        Pose((0.0, 0.0, 0.0), (0.0, 0.0, float("nan"), 1.0)),
    ],
)
def test_trajectory_rejects_invalid_pose_values(bad_pose: Pose) -> None:
    futures = [_pose(float(index), 0.0, 0.0, 0.0) for index in range(1, 9)]
    futures[3] = bad_pose

    with pytest.raises(ValueError):
        build_relative_trajectory(_pose(0.0, 0.0, 0.0, 0.0), futures)


def test_trajectory_rechecks_finite_values_after_float32_cast() -> None:
    futures = tuple(_pose(3.5e38, float(index), 0.0, 0.0) for index in range(8))

    with pytest.raises(ValueError, match="float32"):
        build_relative_trajectory(_pose(0.0, 0.0, 0.0, 0.0), futures)


def test_trajectory_normalizes_python_integer_float64_overflow() -> None:
    futures = [_pose(float(index), 0.0, 0.0, 0.0) for index in range(8)]
    futures[3] = Pose((10**400, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError):
        build_relative_trajectory(_pose(0.0, 0.0, 0.0, 0.0), futures)
