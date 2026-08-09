from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
import json
from pathlib import Path
import stat
from typing import Any

from morai_ddv2.atomic import assert_atomic_output_avoids_paths, atomic_write_bytes
from morai_ddv2.config import (
    canonical_json_bytes,
    load_resolved_domain,
    sha256_file,
)
from morai_ddv2.errors import IntegrityError

from .exporter import ExportBundle
from .metadata import ExportMetadata


PROFILE_FILENAME = "profile-morai-ddv2-oracle.json"
_EXPECTED_ROLES = (
    "checkpoint",
    "checkpoint_metadata",
    "resolved_domain",
    "sensor_preset",
    "training_manifest",
    "image_backbone",
    "trajectory_anchor",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IntegrityError(f"unable to load {label}") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} must contain a JSON object")
    return payload, value


def load_required_metadata(path: Path) -> ExportMetadata:
    payload, value = _read_json_object(Path(path), label="checkpoint metadata")
    expected = tuple(field.name for field in fields(ExportMetadata))
    if set(value) != set(expected) or canonical_json_bytes(value) != payload:
        raise IntegrityError(
            "checkpoint metadata schema or canonical encoding is invalid"
        )
    values = dict(value)
    if isinstance(values.get("trainable_prefixes"), list):
        values["trainable_prefixes"] = tuple(values["trainable_prefixes"])
    try:
        return ExportMetadata(**values)
    except (TypeError, ValueError) as error:
        raise IntegrityError("checkpoint metadata values are invalid") from error


def _regular_real_file(path: Path, *, label: str) -> Path:
    candidate = Path(path).absolute()
    try:
        status = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise IntegrityError(f"{label} is unavailable") from error
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise IntegrityError(f"{label} must be a real regular file")
    return resolved


def load_required_profile_document(path: Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).absolute()
    resolved = _regular_real_file(candidate, label="generated Validator profile")
    if candidate != resolved:
        raise IntegrityError("generated Validator profile must not use symlinks")
    payload, value = _read_json_object(resolved, label="generated Validator profile")
    if canonical_json_bytes(value) != payload:
        raise IntegrityError("generated Validator profile must use canonical JSON")
    return resolved, value


def _baseline_ddv2(profile: Mapping[str, Any]) -> dict[str, Any]:
    profiles = profile.get("profiles")
    if not isinstance(profiles, list):
        raise IntegrityError("baseline profile lacks profiles")
    matches = [
        item
        for item in profiles
        if isinstance(item, dict) and item.get("id") == "diffusiondrive_v2"
    ]
    if len(matches) != 1:
        raise IntegrityError("baseline must contain one diffusiondrive_v2 profile")
    return matches[0]


def _baseline_assets(profile: Mapping[str, Any]) -> Mapping[str, dict[str, Any]]:
    assets = profile.get("runtime_assets")
    if not isinstance(assets, list):
        raise IntegrityError("baseline DDV2 runtime assets are invalid")
    by_role: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, dict) or type(asset.get("role")) is not str:
            raise IntegrityError("baseline DDV2 runtime asset is invalid")
        role = asset["role"]
        if role in by_role:
            raise IntegrityError(f"baseline DDV2 runtime role repeats: {role}")
        by_role[role] = asset
    if set(by_role) != {"checkpoint", "image_backbone", "trajectory_anchor"}:
        raise IntegrityError("baseline DDV2 runtime roles are invalid")
    return by_role


def _resolved_asset(
    declaration: Mapping[str, Any], validator_root: Path, *, role: str
) -> dict[str, Any]:
    path_value = declaration.get("path")
    expected = declaration.get("sha256")
    if type(path_value) is not str or type(expected) is not str:
        raise IntegrityError(f"baseline {role} asset declaration is invalid")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = validator_root / candidate
    path = _regular_real_file(candidate, label=f"baseline {role} asset")
    actual = sha256_file(path)
    if actual != expected:
        raise IntegrityError(f"baseline {role} asset hash is invalid")
    result = dict(declaration)
    result["path"] = str(path)
    result["sha256"] = actual
    return result


def _domain_contract(document: Mapping[str, Any]) -> Mapping[str, Any]:
    sensors = document.get("sensors")
    camera_processing = document.get("camera_processing")
    lidar_processing = document.get("lidar_processing")
    if not all(
        isinstance(value, Mapping)
        for value in (sensors, camera_processing, lidar_processing)
    ):
        raise IntegrityError("resolved domain input contract is invalid")
    camera_values = sensors.get("cameras")
    lidar = sensors.get("lidar")
    if (
        isinstance(camera_values, (str, bytes))
        or not isinstance(camera_values, Sequence)
        or not isinstance(lidar, Mapping)
    ):
        raise IntegrityError("resolved domain sensor contract is invalid")
    resolutions = {
        item.get("role"): list(item.get("resolution_wh", ()))
        for item in camera_values
        if isinstance(item, Mapping)
    }
    resize = camera_processing.get("resize_wh")
    model_camera = camera_processing.get("model_tensor")
    self_mask = lidar_processing.get("self_mask_base_link_m")
    translation = list(lidar.get("translation_base_from_lidar_m", ()))
    quaternion = list(lidar.get("quaternion_base_from_lidar_xyzw", ()))
    if (
        resolutions != {"front": [1280, 720], "left": [640, 480], "right": [640, 480]}
        or not isinstance(resize, Mapping)
        or {role: list(resize.get(role, ())) for role in ("front", "left", "right")}
        != {"front": [455, 256], "left": [341, 256], "right": [341, 256]}
        or camera_processing.get("outer_crop_x_start") != 56
        or camera_processing.get("outer_crop_x_end_exclusive") != 1080
        or not isinstance(model_camera, Mapping)
        or model_camera.get("dtype") != "float32"
        or list(model_camera.get("shape_per_sample", ())) != [3, 256, 1024]
        or list(model_camera.get("value_range", ())) != [0, 1]
        or camera_processing.get("output_color_order") != "RGB"
        or translation != [1.5, 0, 1.25]
        or quaternion != [0, 0, 0, 1]
        or not isinstance(self_mask, Mapping)
        or lidar_processing.get("normalization") != "divide_by_5"
    ):
        raise IntegrityError("resolved domain camera or LiDAR contract drifted")
    mask = {
        "x": list(self_mask.get("x", ())),
        "y": list(self_mask.get("y", ())),
        "z": list(self_mask.get("z", ())),
        "boundaries_inclusive": self_mask.get("boundaries_inclusive"),
    }
    if mask != {
        "x": [0, 2],
        "y": [-1, 1],
        "z": [0.8, 1.6],
        "boundaries_inclusive": True,
    }:
        raise IntegrityError("resolved domain LiDAR self-mask drifted")
    return {
        "camera": {
            "source_resolution_wh": resolutions,
            "resize_wh": {
                role: list(resize[role]) for role in ("front", "left", "right")
            },
            "outer_crop_x": [56, 1080],
            "output": {
                "color_order": "RGB",
                "dtype": "float32",
                "shape_chw": [3, 256, 1024],
                "value_range": [0, 1],
            },
        },
        "lidar": {
            "translation_base_from_lidar_m": translation,
            "self_mask_base_link_m": mask,
            "normalization": "count/5",
        },
    }


def build_morai_validator_profile(
    *,
    baseline_profile: Path,
    output_profile: Path,
    validator_root: Path,
    bundle: ExportBundle,
    resolved_domain: Path,
    sensor_preset: Path,
    training_manifest: Path,
) -> Path:
    if type(bundle) is not ExportBundle:
        raise TypeError("bundle must be ExportBundle")
    validator = Path(validator_root).resolve(strict=True)
    expected_baseline = validator / "src/Validator_pkg/config/profile.json"
    baseline = _regular_real_file(baseline_profile, label="baseline Validator profile")
    if baseline != expected_baseline:
        raise IntegrityError("baseline profile is outside the pinned Validator root")
    output = Path(output_profile).absolute()
    if output.name != PROFILE_FILENAME:
        raise IntegrityError(f"generated profile must be named {PROFILE_FILENAME}")
    checkpoint = _regular_real_file(bundle.checkpoint, label="export checkpoint")
    metadata_path = _regular_real_file(bundle.metadata, label="checkpoint metadata")
    resolved_path = _regular_real_file(resolved_domain, label="resolved domain")
    preset_path = _regular_real_file(sensor_preset, label="sensor preset")
    manifest_path = _regular_real_file(training_manifest, label="training manifest")
    metadata = load_required_metadata(metadata_path)
    if (
        sha256_file(checkpoint) != bundle.checkpoint_sha256
        or metadata.export_checkpoint_sha256 != bundle.checkpoint_sha256
        or metadata.export_state_dict_sha256 != bundle.state_dict_sha256
    ):
        raise IntegrityError("export bundle identity is inconsistent")

    resolved = load_resolved_domain(resolved_path)
    source = resolved.document.get("sensor_preset")
    cache = resolved.document.get("cache")
    attributes = cache.get("hdf5_attributes") if isinstance(cache, Mapping) else None
    if (
        resolved.document.get("domain_id") != metadata.domain_id
        or resolved.sha256 != metadata.resolved_domain_sha256
        or not isinstance(source, Mapping)
        or source.get("sha256") != metadata.sensor_preset_sha256
        or sha256_file(preset_path) != metadata.sensor_preset_sha256
        or sha256_file(manifest_path) != metadata.training_manifest_sha256
        or not isinstance(attributes, Mapping)
        or attributes.get("cache_schema_version") != metadata.cache_schema_version
        or attributes.get("clock_policy_version") != metadata.clock_policy_version
    ):
        raise IntegrityError(
            "resolved domain, sensor preset, or manifest identity drifted"
        )

    baseline_bytes, profile = _read_json_object(baseline, label="baseline profile")
    selected = _baseline_ddv2(profile)
    baseline_assets = _baseline_assets(selected)
    image = _resolved_asset(
        baseline_assets["image_backbone"], validator, role="image_backbone"
    )
    anchor = _resolved_asset(
        baseline_assets["trajectory_anchor"], validator, role="trajectory_anchor"
    )
    assert_atomic_output_avoids_paths(
        output,
        (
            baseline,
            checkpoint,
            metadata_path,
            resolved_path,
            preset_path,
            manifest_path,
            Path(image["path"]),
            Path(anchor["path"]),
        ),
        output_name="generated Validator profile",
    )
    selected["runtime_assets"] = [
        {
            "role": "checkpoint",
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        {
            "role": "checkpoint_metadata",
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
        {
            "role": "resolved_domain",
            "path": str(resolved_path),
            "sha256": sha256_file(resolved_path),
        },
        {
            "role": "sensor_preset",
            "path": str(preset_path),
            "sha256": sha256_file(preset_path),
        },
        {
            "role": "training_manifest",
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        image,
        anchor,
    ]
    if tuple(asset["role"] for asset in selected["runtime_assets"]) != _EXPECTED_ROLES:
        raise IntegrityError("generated profile runtime role order drifted")
    selected["export_identity"] = {
        "domain_id": metadata.domain_id,
        "resolved_domain_sha256": metadata.resolved_domain_sha256,
        "export_state_dict_sha256": metadata.export_state_dict_sha256,
        "raw_schema_sha256": metadata.raw_schema_sha256,
        "training_manifest_sha256": metadata.training_manifest_sha256,
        "cache_schema_version": metadata.cache_schema_version,
        "clock_policy_version": metadata.clock_policy_version,
    }
    inference_policy = selected.get("inference_policy")
    if not isinstance(inference_policy, dict):
        raise IntegrityError("baseline DDV2 inference policy is invalid")
    inference_policy["reference_stamp_policy"] = "new_stamp_once"
    selected["domain_contract"] = _domain_contract(resolved.document)

    common = profile.get("common_inference_policy")
    if not isinstance(common, dict):
        raise IntegrityError("baseline common inference policy is invalid")
    common["static_transforms"] = {
        "T_base_from_lidar": {
            "parent_frame_id": "base_link",
            "child_frame_id": str(resolved.document["sensors"]["lidar"]["frame_id"]),
            "translation_m": [1.5, 0, 1.25],
            "quaternion_xyzw": [0, 0, 0, 1],
        }
    }

    payload = canonical_json_bytes(profile)

    def baseline_unchanged() -> None:
        if baseline.read_bytes() != baseline_bytes:
            raise IntegrityError("baseline Validator profile changed during generation")

    atomic_write_bytes(output, payload, pre_replace_gate=baseline_unchanged)
    baseline_unchanged()
    return output.resolve(strict=True)


__all__ = (
    "PROFILE_FILENAME",
    "build_morai_validator_profile",
    "load_required_metadata",
    "load_required_profile_document",
)
