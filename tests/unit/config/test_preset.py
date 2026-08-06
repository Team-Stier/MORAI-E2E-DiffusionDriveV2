from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from morai_ddv2.config.preset import load_sensor_preset, morai_camera_quaternion
from morai_ddv2.errors import IntegrityError, SensorPresetError


EXPECTED_PRESET_SHA256 = "b7e69c70cc0b2948c3a5b352eee022be04f8884a1df135697f89bb4bb88263fb"


def quaternion_angular_distance(left, right) -> float:
    left_norm = math.sqrt(sum(component * component for component in left))
    right_norm = math.sqrt(sum(component * component for component in right))
    dot = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return 2.0 * math.acos(min(1.0, max(0.0, abs(dot))))


def test_sensor_preset_hash_mismatch_is_rejected(repo_root) -> None:
    preset_path = repo_root / "sensor_data" / "0725demo_ros.json"

    with pytest.raises(IntegrityError, match="sensor preset SHA-256"):
        load_sensor_preset(preset_path, "0" * 64)


def test_modified_sensor_preset_copy_is_rejected(repo_root, tmp_path: Path) -> None:
    source_path = repo_root / "sensor_data" / "0725demo_ros.json"
    modified_path = tmp_path / "0725demo_ros.json"
    modified_path.write_bytes(source_path.read_bytes() + b"\n")

    with pytest.raises(IntegrityError, match="sensor preset SHA-256"):
        load_sensor_preset(modified_path, EXPECTED_PRESET_SHA256)


def test_sensor_preset_parses_required_immutable_sensors(repo_root) -> None:
    preset = load_sensor_preset(
        repo_root / "sensor_data" / "0725demo_ros.json",
        EXPECTED_PRESET_SHA256,
    )

    assert preset.sha256 == EXPECTED_PRESET_SHA256
    assert tuple(camera.sensor_id for camera in preset.cameras) == (1, 2, 3)
    assert tuple(camera.topic for camera in preset.cameras) == (
        "/image_jpeg/front/compressed",
        "/image_jpeg/left/compressed",
        "/image_jpeg/right/compressed",
    )
    assert tuple(camera.frame_id for camera in preset.cameras) == (
        "Camera-1",
        "Camera-2",
        "Camera-3",
    )
    assert preset.cameras[0].resolution_wh == (1280, 720)
    assert preset.cameras[1].focal_length_pixel == 320.0
    assert preset.lidar.sensor_id == 4
    assert preset.lidar.topic == "/lidar3D"
    assert preset.lidar.frame_id == "Lidar3D-4"

    with pytest.raises(FrozenInstanceError):
        setattr(preset.cameras[0], "topic", "/changed")


def test_nonzero_camera_roll_is_rejected() -> None:
    with pytest.raises(SensorPresetError, match="nonzero camera roll"):
        morai_camera_quaternion(roll_deg=1.0e-11, pitch_deg=2.0, yaw_deg=0.0)


@pytest.mark.parametrize(
    ("pitch", "yaw", "expected"),
    [
        (2.0, 0.0, (0.0, 0.0174524058, 0.0, 0.9998477101)),
        (10.0, 70.0, (-0.0499904864, -0.0713938102, -0.5713938475, -0.8160349727)),
        (10.0, 290.0, (-0.0499904864, 0.0713938102, -0.5713938475, 0.8160349727)),
    ],
)
def test_morai_camera_rotation_matches_bag_tf(pitch, yaw, expected) -> None:
    actual = morai_camera_quaternion(roll_deg=0.0, pitch_deg=pitch, yaw_deg=yaw)
    assert quaternion_angular_distance(actual, expected) < 1.0e-7
