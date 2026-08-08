from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .models import Bracket


@dataclass(frozen=True)
class Pose:
    translation_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]


def _finite_array(value: object, shape: Tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must contain numeric values".format(name)) from error
    if result.shape != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    if not np.isfinite(result).all():
        raise ValueError("{} must contain only finite values".format(name))
    return result


def normalize_quaternion(q: Sequence[float]) -> np.ndarray:
    """Return a finite unit quaternion in canonical ``(x, y, z, w)`` order."""

    quaternion = _finite_array(q, (4,), "quaternion")
    scale = float(np.max(np.abs(quaternion)))
    if scale == 0.0:
        raise ValueError("quaternion must have nonzero norm")
    scaled = quaternion / scale
    norm = float(np.linalg.norm(scaled))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must have finite nonzero norm")
    return np.asarray(scaled / norm, dtype=np.float64)


def quaternion_angular_distance(
    a: Sequence[float], b: Sequence[float]
) -> float:
    """Return the sign-invariant rotation distance in radians."""

    q0 = normalize_quaternion(a)
    q1 = normalize_quaternion(b)
    dot = float(np.dot(q0, q1))
    clamped = float(np.clip(abs(dot), 0.0, 1.0))
    return float(2.0 * math.acos(clamped))


def _validated_weight(weight: float) -> float:
    if isinstance(weight, bool):
        raise TypeError("weight must be a finite number")
    try:
        result = float(weight)
    except (TypeError, ValueError) as error:
        raise TypeError("weight must be a finite number") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("weight must be finite and in the inclusive range [0, 1]")
    return result


def slerp_shortest(
    a: Sequence[float], b: Sequence[float], weight: float
) -> np.ndarray:
    """Interpolate normalized quaternions along their shortest rotation arc."""

    interpolation_weight = _validated_weight(weight)
    q0 = normalize_quaternion(a)
    q1 = normalize_quaternion(b)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(
            q0 + interpolation_weight * (q1 - q0)
        )

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    interpolated = (
        math.sin((1.0 - interpolation_weight) * theta) / sin_theta * q0
        + math.sin(interpolation_weight * theta) / sin_theta * q1
    )
    return normalize_quaternion(interpolated)


def _validated_pose(pose: object, name: str) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(pose, Pose):
        raise TypeError("{} must be a Pose".format(name))
    translation = _finite_array(
        pose.translation_xyz, (3,), "{}.translation_xyz".format(name)
    )
    quaternion = normalize_quaternion(pose.quaternion_xyzw)
    return translation, quaternion


def interpolate_pose(bracket: Bracket[Pose]) -> Pose:
    """Linearly interpolate translation and shortest-arc interpolate rotation."""

    if not isinstance(bracket, Bracket):
        raise TypeError("bracket must be a Bracket")
    before_translation, before_quaternion = _validated_pose(
        bracket.before.value, "bracket.before.value"
    )
    after_translation, after_quaternion = _validated_pose(
        bracket.after.value, "bracket.after.value"
    )
    weight = _validated_weight(bracket.weight)
    translation = before_translation + weight * (
        after_translation - before_translation
    )
    quaternion = slerp_shortest(before_quaternion, after_quaternion, weight)
    return Pose(
        translation_xyz=tuple(float(value) for value in translation),
        quaternion_xyzw=tuple(float(value) for value in quaternion),
    )


def interpolate_vector(bracket: Bracket[np.ndarray]) -> np.ndarray:
    """Linearly interpolate finite, equal-shaped vectors in float64."""

    if not isinstance(bracket, Bracket):
        raise TypeError("bracket must be a Bracket")
    try:
        before = np.asarray(bracket.before.value, dtype=np.float64)
        after = np.asarray(bracket.after.value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("vector values must be numeric") from error
    if before.shape != after.shape:
        raise ValueError("vector values must have equal shape")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("vector values must contain only finite values")
    weight = _validated_weight(bracket.weight)
    return np.asarray(before + weight * (after - before), dtype=np.float64)


def _rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def transform_points(points_xyz: np.ndarray, base_from_sensor: Pose) -> np.ndarray:
    """Apply a rigid sensor-to-base transform to finite ``[N,3]`` points."""

    try:
        points = np.asarray(points_xyz, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("points_xyz must contain numeric values") from error
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N,3]")
    if not np.isfinite(points).all():
        raise ValueError("points_xyz must contain only finite values")
    translation, quaternion = _validated_pose(base_from_sensor, "base_from_sensor")
    rotation = _rotation_matrix(quaternion)
    return np.asarray(points @ rotation.T + translation, dtype=np.float64)


def _quaternion_yaw(quaternion_xyzw: Sequence[float]) -> float:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(float(sin_yaw), float(cos_yaw)))


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def relative_se2(current: Pose, future: Pose) -> Tuple[float, float, float]:
    """Express a future global pose in the current pose's planar frame."""

    current_translation, current_quaternion = _validated_pose(current, "current")
    future_translation, future_quaternion = _validated_pose(future, "future")
    current_yaw = _quaternion_yaw(current_quaternion)
    future_yaw = _quaternion_yaw(future_quaternion)
    delta_x = float(future_translation[0] - current_translation[0])
    delta_y = float(future_translation[1] - current_translation[1])
    cosine = math.cos(current_yaw)
    sine = math.sin(current_yaw)
    relative_x = cosine * delta_x + sine * delta_y
    relative_y = -sine * delta_x + cosine * delta_y
    relative_heading = _wrap_to_pi(future_yaw - current_yaw)
    return float(relative_x), float(relative_y), relative_heading
