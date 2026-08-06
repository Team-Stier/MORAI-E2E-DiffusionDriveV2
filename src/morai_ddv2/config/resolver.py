from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

from morai_ddv2.atomic import atomic_write_bytes
from morai_ddv2.errors import ConfigurationError, IntegrityError

from .canonical import canonical_json_bytes, sha256_bytes, sha256_file
from .models import (
    CameraSensor,
    LidarSensor,
    ResolvedDomain,
    ResolvedDomainArtifact,
    SensorPreset,
    thaw_json,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _required_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("{} must be an object".format(context))
    return value


def _camera_document(camera: CameraSensor) -> Dict[str, Any]:
    return {
        "sensor_id": camera.sensor_id,
        "role": camera.role,
        "topic": camera.topic,
        "frame_id": camera.frame_id,
        "translation_base_from_camera_m": list(camera.translation_base_from_camera_m),
        "morai_euler_deg": list(camera.morai_euler_deg),
        "quaternion_base_from_camera_xyzw": list(camera.quaternion_base_from_camera_xyzw),
        "source_period_s": camera.source_period_s,
        "sensor_size_mm": list(camera.sensor_size_mm),
        "focal_length_mm": camera.focal_length_mm,
        "focal_length_pixel_recorded": camera.focal_length_pixel,
        "principal_point_xy": list(camera.principal_point_xy),
        "resolution_wh": list(camera.resolution_wh),
        "compression_quality": camera.compression_quality,
        "horizontal_fov_deg": camera.horizontal_fov_deg,
        "lens_distortion": list(camera.lens_distortion),
    }


def _lidar_document(lidar: LidarSensor) -> Dict[str, Any]:
    return {
        "sensor_id": lidar.sensor_id,
        "role": lidar.role,
        "topic": lidar.topic,
        "frame_id": lidar.frame_id,
        "translation_base_from_lidar_m": list(lidar.translation_base_from_lidar_m),
        "morai_euler_deg": list(lidar.morai_euler_deg),
        "quaternion_base_from_lidar_xyzw": list(lidar.quaternion_base_from_lidar_xyzw),
        "source_period_s": lidar.source_period_s,
        "maximum_range_m": lidar.maximum_range_m,
        "noise_ratio": lidar.noise_ratio,
    }


def _focal_consistency(camera: CameraSensor) -> Tuple[float, bool]:
    expected_from_fov = camera.resolution_wh[0] / (
        2.0 * math.tan(math.radians(camera.horizontal_fov_deg) / 2.0)
    )
    consistent = math.isclose(
        camera.focal_length_pixel,
        expected_from_fov,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )
    return expected_from_fov, consistent


def _calibration_notes(
    cameras: Tuple[CameraSensor, CameraSensor, CameraSensor]
) -> Dict[str, Any]:
    comparisons: List[Dict[str, Any]] = []
    for camera in cameras:
        expected_from_fov, consistent = _focal_consistency(camera)
        comparisons.append(
            {
                "sensor_id": camera.sensor_id,
                "recorded_focal_length_pixel": camera.focal_length_pixel,
                "focal_length_pixel_implied_by_horizontal_fov": expected_from_fov,
                "consistent": consistent,
            }
        )
    return {
        "focal_length_pixel": {
            "recorded_from_preset": True,
            "used_for_projection": False,
            "inconsistent_with_fov": any(not item["consistent"] for item in comparisons),
            "comparisons": comparisons,
        },
        "projection_metadata": {
            "authoritative_fields": ["horizontal_fov_deg", "principal_point_xy"],
            "projection_warp_enabled": False,
        },
    }


def resolve_domain(
    *, preset: SensorPreset, raw_domain: Mapping[str, Any], source_commit: str
) -> ResolvedDomainArtifact:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ConfigurationError("source commit must be 40-character lowercase hexadecimal")

    domain = thaw_json(_required_mapping(raw_domain, "domain YAML"))
    for reserved_key in (
        "resolved_domain_sha256",
        "source_code_commit",
        "sensors",
        "calibration_notes",
    ):
        if reserved_key in domain:
            raise ConfigurationError("domain YAML contains reserved key: {}".format(reserved_key))

    domain_id = domain.get("domain_id")
    if not isinstance(domain_id, str) or not domain_id:
        raise ConfigurationError("domain_id must be a nonempty string")
    source = _required_mapping(domain.get("sensor_preset"), "sensor_preset")
    configured_sha256 = source.get("sha256")
    if configured_sha256 != preset.sha256:
        raise IntegrityError("sensor preset SHA-256 does not match the domain YAML")
    configured_path = source.get("path")
    if configured_path != "sensor_data/0725demo_ros.json":
        raise ConfigurationError(
            "sensor_preset.path must be sensor_data/0725demo_ros.json"
        )

    domain["source_code_commit"] = source_commit
    domain["sensors"] = {
        "cameras": [_camera_document(camera) for camera in preset.cameras],
        "lidar": _lidar_document(preset.lidar),
    }
    domain["calibration_notes"] = _calibration_notes(preset.cameras)

    digest = sha256_bytes(canonical_json_bytes(domain))
    return ResolvedDomainArtifact(document=domain, sha256=digest)


def publish_resolved_domain(artifact: ResolvedDomainArtifact, output_path: Path) -> None:
    atomic_write_bytes(Path(output_path), artifact.publication_bytes)


def _object_without_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError("resolved domain contains duplicate key: {}".format(key))
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ConfigurationError("resolved domain contains non-finite JSON value: {}".format(value))


def _current_preset_path(document: Mapping[str, Any]) -> Path:
    source = _required_mapping(document.get("sensor_preset"), "sensor_preset")
    configured_path = source.get("path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ConfigurationError("sensor_preset.path must be a nonempty string")
    relative_path = Path(configured_path)
    if relative_path.is_absolute():
        raise ConfigurationError("sensor_preset.path must be project-relative")
    candidate = (_PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(_PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ConfigurationError("sensor_preset.path escapes the project root") from error
    return candidate


def load_resolved_domain(path: Path) -> ResolvedDomain:
    try:
        raw_bytes = Path(path).read_bytes()
    except OSError as error:
        raise IntegrityError("unable to read resolved domain: {}".format(path)) from error

    try:
        publication = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError("resolved domain is not valid UTF-8 JSON") from error
    publication = dict(_required_mapping(publication, "resolved domain"))
    expected_sha256 = publication.pop("resolved_domain_sha256", None)
    if not isinstance(expected_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise IntegrityError("resolved domain SHA-256 is missing or malformed")

    actual_sha256 = sha256_bytes(canonical_json_bytes(publication))
    if actual_sha256 != expected_sha256:
        raise IntegrityError(
            "resolved domain SHA-256 mismatch: expected {}, got {}".format(
                expected_sha256, actual_sha256
            )
        )

    canonical_publication = dict(publication)
    canonical_publication["resolved_domain_sha256"] = expected_sha256
    if canonical_json_bytes(canonical_publication) != raw_bytes:
        raise IntegrityError("resolved domain is not canonical RFC 8785 JSON")

    source = _required_mapping(publication.get("sensor_preset"), "sensor_preset")
    preset_sha256 = source.get("sha256")
    if not isinstance(preset_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", preset_sha256
    ) is None:
        raise IntegrityError("current sensor preset SHA-256 expectation is malformed")
    preset_path = _current_preset_path(publication)
    try:
        current_sha256 = sha256_file(preset_path)
    except OSError as error:
        raise IntegrityError("unable to verify current sensor preset SHA-256") from error
    if current_sha256 != preset_sha256:
        raise IntegrityError(
            "current sensor preset SHA-256 mismatch: expected {}, got {}".format(
                preset_sha256, current_sha256
            )
        )

    return ResolvedDomain(document=publication, sha256=expected_sha256)
