from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from morai_ddv2.preprocess.geometry import (
    Pose,
    interpolate_pose,
    interpolate_vector,
    normalize_quaternion,
    quaternion_angular_distance,
    relative_se2,
    slerp_shortest,
    transform_points,
)
from morai_ddv2.preprocess.models import Bracket, StampedRecord


IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
YAW_90_QUATERNION = (
    0.0,
    0.0,
    0.7071067811865475,
    0.7071067811865476,
)


def _yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    half_angle = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))


def _bracket(before, after, weight: float):
    return Bracket(
        before=StampedRecord(0, 0, before),
        after=StampedRecord(100, 1, after),
        weight=weight,
    )


def test_pose_is_an_immutable_tuple_backed_value() -> None:
    pose = Pose((1.0, 2.0, 3.0), IDENTITY_QUATERNION)

    assert pose.translation_xyz == (1.0, 2.0, 3.0)
    assert pose.quaternion_xyzw == IDENTITY_QUATERNION
    with pytest.raises(FrozenInstanceError):
        pose.translation_xyz = (0.0, 0.0, 0.0)


def test_normalize_quaternion_returns_unit_float64_without_mutating_input() -> None:
    original = np.asarray([0.0, 0.0, 0.0, 2.0], dtype=np.float32)

    actual = normalize_quaternion(original)

    np.testing.assert_array_equal(original, np.asarray([0.0, 0.0, 0.0, 2.0]))
    np.testing.assert_array_equal(actual, np.asarray(IDENTITY_QUATERNION))
    assert actual.dtype == np.float64
    assert actual is not original


@pytest.mark.parametrize(
    "quaternion",
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [[0.0, 0.0, 0.0, 1.0]],
        [0.0, 0.0, math.nan, 1.0],
        [0.0, 0.0, math.inf, 1.0],
    ],
)
def test_normalize_quaternion_rejects_zero_wrong_shape_and_nonfinite(
    quaternion,
) -> None:
    with pytest.raises(ValueError):
        normalize_quaternion(quaternion)


def test_quaternion_angular_distance_is_sign_invariant_and_clamped() -> None:
    positive = np.asarray([0.0, 0.0, 0.0, 10.0])
    negative = -positive

    assert quaternion_angular_distance(positive, negative) == pytest.approx(0.0)
    assert quaternion_angular_distance(
        IDENTITY_QUATERNION, (0.0, 0.0, 1.0, 0.0)
    ) == pytest.approx(math.pi)


def test_slerp_flips_negative_dot_to_follow_the_shortest_arc() -> None:
    negative_yaw_90_representation = tuple(-value for value in YAW_90_QUATERNION)

    actual = slerp_shortest(
        IDENTITY_QUATERNION, negative_yaw_90_representation, 0.5
    )

    np.testing.assert_allclose(
        actual,
        np.asarray([0.0, 0.0, 0.3826834323650898, 0.9238795325112867]),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert actual.dtype == np.float64


def test_slerp_uses_normalized_linear_branch_above_dot_threshold() -> None:
    yaw_2_degrees = (
        0.0,
        0.0,
        0.01745240643728351,
        0.9998476951563913,
    )

    actual = slerp_shortest(IDENTITY_QUATERNION, yaw_2_degrees, 0.25)

    np.testing.assert_allclose(
        actual,
        np.asarray([0.0, 0.0, 0.0043632262124411684, 0.9999904810832047]),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_slerp_uses_general_branch_for_well_separated_quaternions() -> None:
    actual = slerp_shortest(IDENTITY_QUATERNION, YAW_90_QUATERNION, 0.25)

    np.testing.assert_allclose(
        actual,
        np.asarray([0.0, 0.0, 0.19509032201612825, 0.9807852804032304]),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_slerp_preserves_normalized_endpoints() -> None:
    scaled_identity = (0.0, 0.0, 0.0, 3.0)
    scaled_yaw_90 = tuple(value * 4.0 for value in YAW_90_QUATERNION)

    np.testing.assert_allclose(
        slerp_shortest(scaled_identity, scaled_yaw_90, 0.0),
        np.asarray(IDENTITY_QUATERNION),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        slerp_shortest(scaled_identity, scaled_yaw_90, 1.0),
        np.asarray(YAW_90_QUATERNION),
        rtol=0.0,
        atol=1.0e-15,
    )


@pytest.mark.parametrize("weight", [-1.0e-9, 1.000000001, math.nan, math.inf])
def test_slerp_rejects_weight_outside_finite_unit_interval(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        slerp_shortest(IDENTITY_QUATERNION, YAW_90_QUATERNION, weight)


def test_interpolate_pose_linearly_blends_translation_and_slerps_rotation() -> None:
    before = Pose((0.0, 0.0, 0.0), IDENTITY_QUATERNION)
    after = Pose((4.0, -2.0, 8.0), YAW_90_QUATERNION)

    actual = interpolate_pose(_bracket(before, after, 0.25))

    assert actual.translation_xyz == pytest.approx((1.0, -0.5, 2.0))
    assert actual.quaternion_xyzw == pytest.approx(
        (0.0, 0.0, 0.19509032201612825, 0.9807852804032304), abs=1.0e-15
    )


def test_interpolate_vector_returns_float64_linear_result() -> None:
    before = np.asarray([0.0, 2.0, -4.0], dtype=np.float32)
    after = np.asarray([4.0, 6.0, 8.0], dtype=np.float32)

    actual = interpolate_vector(_bracket(before, after, 0.25))

    np.testing.assert_array_equal(actual, np.asarray([1.0, 3.0, -1.0]))
    assert actual.dtype == np.float64


@pytest.mark.parametrize(
    "before, after",
    [
        (np.zeros(2), np.zeros(3)),
        (np.asarray([0.0, math.nan]), np.zeros(2)),
        (np.zeros(2), np.asarray([0.0, math.inf])),
    ],
)
def test_interpolate_vector_rejects_shape_mismatch_and_nonfinite(before, after) -> None:
    with pytest.raises(ValueError):
        interpolate_vector(_bracket(before, after, 0.5))


def test_transform_points_applies_rotation_then_translation_in_float64() -> None:
    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]], dtype=np.float32)
    base_from_sensor = Pose((10.0, -2.0, 1.0), YAW_90_QUATERNION)

    actual = transform_points(points, base_from_sensor)

    np.testing.assert_allclose(
        actual,
        np.asarray([[10.0, -1.0, 1.0], [8.0, -2.0, 4.0]]),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert actual.dtype == np.float64


def test_transform_points_accepts_empty_n_by_three_array() -> None:
    actual = transform_points(
        np.empty((0, 3), dtype=np.float32),
        Pose((0.0, 0.0, 0.0), IDENTITY_QUATERNION),
    )

    assert actual.shape == (0, 3)
    assert actual.dtype == np.float64


@pytest.mark.parametrize(
    "points, pose",
    [
        (np.zeros(3), Pose((0.0, 0.0, 0.0), IDENTITY_QUATERNION)),
        (np.zeros((1, 2)), Pose((0.0, 0.0, 0.0), IDENTITY_QUATERNION)),
        (
            np.asarray([[0.0, math.nan, 0.0]]),
            Pose((0.0, 0.0, 0.0), IDENTITY_QUATERNION),
        ),
        (np.zeros((1, 3)), Pose((math.inf, 0.0, 0.0), IDENTITY_QUATERNION)),
        (np.zeros((1, 3)), Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))),
    ],
)
def test_transform_points_rejects_malformed_or_nonfinite_inputs(points, pose) -> None:
    with pytest.raises(ValueError):
        transform_points(points, pose)


def test_relative_se2_rotates_world_delta_into_current_frame() -> None:
    current = Pose((1.0, 2.0, 9.0), YAW_90_QUATERNION)
    future = Pose((1.0, 4.0, -3.0), (0.0, 0.0, 1.0, 0.0))

    x_m, y_m, heading_rad = relative_se2(current, future)

    assert x_m == pytest.approx(2.0, abs=1.0e-15)
    assert y_m == pytest.approx(0.0, abs=1.0e-15)
    assert heading_rad == pytest.approx(math.pi / 2.0, abs=1.0e-15)


def test_relative_se2_wraps_heading_continuously_across_pi() -> None:
    current = Pose((0.0, 0.0, 0.0), _yaw_quaternion(179.9))
    future = Pose((0.0, 0.0, 0.0), _yaw_quaternion(-179.9))

    assert relative_se2(current, future)[2] == pytest.approx(
        0.003490658503988442, abs=1.0e-14
    )
    assert relative_se2(
        Pose((0.0, 0.0, 0.0), IDENTITY_QUATERNION),
        Pose((0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    )[2] == pytest.approx(-math.pi)
