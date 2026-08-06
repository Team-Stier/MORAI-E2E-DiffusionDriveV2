from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Tuple

from morai_ddv2.errors import IntegrityError, SensorPresetError

from .canonical import sha256_bytes
from .models import CameraSensor, LidarSensor, SensorPreset


_CAMERA_ROLES = {1: "front", 2: "left", 3: "right"}


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SensorPresetError("{} must be an object".format(context))
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SensorPresetError("{} must be an array".format(context))
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise SensorPresetError("{} must be numeric".format(context))
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise SensorPresetError("{} must be numeric".format(context)) from error
    if not math.isfinite(converted):
        raise SensorPresetError("{} must be finite".format(context))
    return converted


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise SensorPresetError("{} must be an integer".format(context))
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise SensorPresetError("{} must be an integer".format(context)) from error
    if isinstance(value, float) and not value.is_integer():
        raise SensorPresetError("{} must be an integer".format(context))
    return converted


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SensorPresetError("{} must be a nonempty string".format(context))
    return value


def _xyz(entry: Mapping[str, Any], key: str, context: str) -> Tuple[float, float, float]:
    vector = _mapping(entry.get(key), "{}.{}".format(context, key))
    return (
        _finite_float(vector.get("x"), "{}.{}.x".format(context, key)),
        _finite_float(vector.get("y"), "{}.{}.y".format(context, key)),
        _finite_float(vector.get("z"), "{}.{}.z".format(context, key)),
    )


def _rotation(entry: Mapping[str, Any], context: str) -> Tuple[float, float, float]:
    rotation = _mapping(entry.get("rot"), "{}.rot".format(context))
    return (
        _finite_float(rotation.get("roll"), "{}.rot.roll".format(context)),
        _finite_float(rotation.get("pitch"), "{}.rot.pitch".format(context)),
        _finite_float(rotation.get("yaw"), "{}.rot.yaw".format(context)),
    )


def _hamilton_product(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def morai_camera_quaternion(
    *, roll_deg: float, pitch_deg: float, yaw_deg: float
) -> Tuple[float, float, float, float]:
    """Convert the v1 MORAI camera convention as ``q_y * q_z``."""

    if not all(math.isfinite(value) for value in (roll_deg, pitch_deg, yaw_deg)):
        raise SensorPresetError("camera rotation must be finite")
    if abs(roll_deg) > 1.0e-12:
        raise SensorPresetError("nonzero camera roll requires a new domain version")

    wrapped_yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
    pitch_half = math.radians(pitch_deg) / 2.0
    yaw_half = math.radians(wrapped_yaw_deg) / 2.0
    q_y = (0.0, math.sin(pitch_half), 0.0, math.cos(pitch_half))
    q_z = (0.0, 0.0, math.sin(yaw_half), math.cos(yaw_half))
    product = _hamilton_product(q_y, q_z)
    norm = math.sqrt(sum(component * component for component in product))
    if norm == 0.0:
        raise SensorPresetError("camera rotation produced a zero quaternion")
    return (
        product[0] / norm,
        product[1] / norm,
        product[2] / norm,
        product[3] / norm,
    )


def _parse_camera(entry: Mapping[str, Any], sensor_id: int, role: str) -> CameraSensor:
    context = "camera {}".format(sensor_id)
    if _integer(entry.get("m_SensorType"), "{}.m_SensorType".format(context)) != 1:
        raise SensorPresetError("{} has the wrong sensor type".format(context))
    config = _mapping(entry.get("cc"), "{}.cc".format(context))
    ros = _mapping(config.get("rosConfig"), "{}.cc.rosConfig".format(context))
    sensor_size = _mapping(config.get("sensorSize"), "{}.cc.sensorSize".format(context))
    principal_point = _mapping(
        config.get("principalPoint"), "{}.cc.principalPoint".format(context)
    )
    distortion = _sequence(
        config.get("lensDistortion"), "{}.cc.lensDistortion".format(context)
    )
    if len(distortion) != 3:
        raise SensorPresetError("{}.cc.lensDistortion must contain three values".format(context))
    distortion_values = (
        _finite_float(distortion[0], "{}.cc.lensDistortion[0]".format(context)),
        _finite_float(distortion[1], "{}.cc.lensDistortion[1]".format(context)),
        _finite_float(distortion[2], "{}.cc.lensDistortion[2]".format(context)),
    )

    roll_deg, pitch_deg, yaw_deg = _rotation(entry, context)
    return CameraSensor(
        sensor_id=sensor_id,
        role=role,
        topic=_text(ros.get("Topic"), "{}.cc.rosConfig.Topic".format(context)),
        frame_id=_text(ros.get("frameID"), "{}.cc.rosConfig.frameID".format(context)),
        translation_base_from_camera_m=_xyz(entry, "pos", context),
        morai_euler_deg=(roll_deg, pitch_deg, yaw_deg),
        quaternion_base_from_camera_xyzw=morai_camera_quaternion(
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg
        ),
        source_period_s=_finite_float(
            config.get("sensorPeriod"), "{}.cc.sensorPeriod".format(context)
        ),
        sensor_size_mm=(
            _finite_float(sensor_size.get("x"), "{}.cc.sensorSize.x".format(context)),
            _finite_float(sensor_size.get("y"), "{}.cc.sensorSize.y".format(context)),
        ),
        focal_length_mm=_finite_float(
            config.get("focalLengthmm"), "{}.cc.focalLengthmm".format(context)
        ),
        focal_length_pixel=_finite_float(
            config.get("focalLengthpixel"), "{}.cc.focalLengthpixel".format(context)
        ),
        principal_point_xy=(
            _finite_float(principal_point.get("x"), "{}.cc.principalPoint.x".format(context)),
            _finite_float(principal_point.get("y"), "{}.cc.principalPoint.y".format(context)),
        ),
        resolution_wh=(
            _integer(config.get("cameraResWidth"), "{}.cc.cameraResWidth".format(context)),
            _integer(config.get("cameraResHeight"), "{}.cc.cameraResHeight".format(context)),
        ),
        compression_quality=_integer(
            config.get("compressedRatio"), "{}.cc.compressedRatio".format(context)
        ),
        horizontal_fov_deg=_finite_float(
            config.get("cameraFOV"), "{}.cc.cameraFOV".format(context)
        ),
        lens_distortion=distortion_values,
    )


def _parse_lidar(entry: Mapping[str, Any], sensor_id: int) -> LidarSensor:
    context = "LiDAR {}".format(sensor_id)
    if _integer(entry.get("m_SensorType"), "{}.m_SensorType".format(context)) != 2:
        raise SensorPresetError("{} has the wrong sensor type".format(context))
    config = _mapping(entry.get("lc"), "{}.lc".format(context))
    ros = _mapping(config.get("rosConfig"), "{}.lc.rosConfig".format(context))
    rotation = _rotation(entry, context)
    if any(abs(component) > 1.0e-12 for component in rotation):
        raise SensorPresetError("nonzero LiDAR rotation requires a new domain version")

    return LidarSensor(
        sensor_id=sensor_id,
        role="lidar",
        topic=_text(ros.get("Topic"), "{}.lc.rosConfig.Topic".format(context)),
        frame_id=_text(ros.get("frameID"), "{}.lc.rosConfig.frameID".format(context)),
        translation_base_from_lidar_m=_xyz(entry, "pos", context),
        morai_euler_deg=rotation,
        quaternion_base_from_lidar_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_period_s=_finite_float(
            config.get("sensorPeriod"), "{}.lc.sensorPeriod".format(context)
        ),
        maximum_range_m=_finite_float(
            config.get("fireDistance"), "{}.lc.fireDistance".format(context)
        ),
        noise_ratio=_finite_float(config.get("noiseRatio"), "{}.lc.noiseRatio".format(context)),
    )


def load_sensor_preset(path: Path, expected_sha256: str) -> SensorPreset:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise IntegrityError(
            "expected sensor preset SHA-256 must be 64 lowercase hexadecimal characters"
        )

    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise IntegrityError("unable to read sensor preset: {}".format(path)) from error
    actual_sha256 = sha256_bytes(raw_bytes)
    if actual_sha256 != expected_sha256:
        raise IntegrityError(
            "sensor preset SHA-256 mismatch: expected {}, got {}".format(
                expected_sha256, actual_sha256
            )
        )

    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SensorPresetError("sensor preset is not valid UTF-8 JSON") from error
    root = _mapping(raw, "sensor preset")

    camera_entries: Dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(_sequence(root.get("cameraList"), "cameraList")):
        entry = _mapping(value, "cameraList[{}]".format(index))
        sensor_id = _integer(entry.get("m_SensorUniqueID"), "camera sensor ID")
        if sensor_id in camera_entries:
            raise SensorPresetError("duplicate camera sensor ID {}".format(sensor_id))
        camera_entries[sensor_id] = entry

    missing_camera_ids = sorted(set(_CAMERA_ROLES) - set(camera_entries))
    if missing_camera_ids:
        raise SensorPresetError("missing required camera IDs: {}".format(missing_camera_ids))
    cameras = (
        _parse_camera(camera_entries[1], 1, _CAMERA_ROLES[1]),
        _parse_camera(camera_entries[2], 2, _CAMERA_ROLES[2]),
        _parse_camera(camera_entries[3], 3, _CAMERA_ROLES[3]),
    )

    lidar_entries: Dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(_sequence(root.get("Lidar3DList"), "Lidar3DList")):
        entry = _mapping(value, "Lidar3DList[{}]".format(index))
        sensor_id = _integer(entry.get("m_SensorUniqueID"), "LiDAR sensor ID")
        if sensor_id in lidar_entries:
            raise SensorPresetError("duplicate LiDAR sensor ID {}".format(sensor_id))
        lidar_entries[sensor_id] = entry
    if 4 not in lidar_entries:
        raise SensorPresetError("missing required LiDAR ID 4")

    return SensorPreset(
        path=path,
        sha256=actual_sha256,
        cameras=cameras,
        lidar=_parse_lidar(lidar_entries[4], 4),
    )
