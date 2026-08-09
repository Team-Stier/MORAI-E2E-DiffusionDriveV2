from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

import torch
from torch import nn

from morai_ddv2.config import canonical_json_bytes, load_resolved_domain, sha256_file
from morai_ddv2.errors import IntegrityError
from morai_ddv2.upstream.ddv2_sel import (
    RUNTIME_SEL_KEY_COUNT,
    RawSelCheckpoint,
    SelAssets,
    SelSchemaError,
    _require_validator_module,
    _validator_source_root,
    construct_pdm_free_sel_network,
    load_raw_sel_checkpoint,
    normalize_sel_runtime_state,
)

from .metadata import ExportMetadata
from .state_hash import state_dict_sha256, state_schema_sha256
from .validator_profile import load_required_metadata, load_required_profile_document


_EXPECTED_ROLES = {
    "checkpoint",
    "checkpoint_metadata",
    "resolved_domain",
    "sensor_preset",
    "training_manifest",
    "image_backbone",
    "trajectory_anchor",
}
_DDV2_COMMIT = "1cd12a1e155c34dcc471261835444c8d5587580b"
_VALIDATOR_COMMIT = "09a3b88e19ae3354903ed676a35e02416284cb58"
_DERIVATIVE_RELATIVE_PATH = Path(
    "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py"
)
_DERIVATIVE_SHA256 = "75cc939dcf563b581cf7c85f1125304039cc05df009510c96b690144a7aa34ac"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_DOMAIN_CONTRACT = {
    "camera": {
        "source_resolution_wh": {
            "front": [1280, 720],
            "left": [640, 480],
            "right": [640, 480],
        },
        "resize_wh": {
            "front": [455, 256],
            "left": [341, 256],
            "right": [341, 256],
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
        "translation_base_from_lidar_m": [1.5, 0, 1.25],
        "self_mask_base_link_m": {
            "x": [0, 2],
            "y": [-1, 1],
            "z": [0.8, 1.6],
            "boundaries_inclusive": True,
        },
        "normalization": "count/5",
    },
}
_EXPECTED_STATIC_TRANSFORMS = {
    "T_base_from_lidar": {
        "parent_frame_id": "base_link",
        "child_frame_id": "Lidar3D-4",
        "translation_m": [1.5, 0, 1.25],
        "quaternion_xyzw": [0, 0, 0, 1],
    }
}
_TRUSTED_VALIDATOR_MODULES = {
    "src/Validator_pkg/validator_inference/src/validator_inference/profile_store.py": (
        "fa605c1069a705de318b5ceaabe44579dccaefa15d93f6b68df5ef79476ad00b"
    ),
    "src/Validator_pkg/validator_inference/src/validator_inference/assets.py": (
        "792e1320188a2477201eef4bbb2af190930ec8209256a0841512683a2c8d3565"
    ),
    "src/Validator_pkg/validator_inference/src/validator_inference/models/contracts.py": (
        "643b2b142c72a45cbf2cde7895a0f40e0fe1ec88fcef847eea7aac104ab79d68"
    ),
    "src/Validator_pkg/validator_inference/src/validator_inference/models/diffusiondrive_v2.py": (
        "07c37fd31754ad6a1fff16b3be05fc7ffb5703d58d04293e92fbc5b3fbb5a4bd"
    ),
}


@dataclass(frozen=True)
class VerifiedValidatorBundle:
    selected_profile: Any
    metadata: ExportMetadata
    raw_checkpoint: RawSelCheckpoint
    runtime_state: OrderedDict[str, torch.Tensor]
    network: nn.Module


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int]


def _git_output(repository: Path, arguments: tuple[str, ...], *, label: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise IntegrityError(f"unable to verify pinned {label}")
    return completed.stdout.strip()


def _validate_pinned_validator_root(validator_root: Path) -> Path:
    candidate = Path(validator_root).absolute()
    try:
        status = candidate.lstat()
        root = candidate.resolve(strict=True)
    except OSError as error:
        raise IntegrityError("pinned Validator root is unavailable") from error
    if (
        root != candidate
        or not stat.S_ISDIR(status.st_mode)
        or _git_output(root, ("rev-parse", "HEAD"), label="Validator commit")
        != _VALIDATOR_COMMIT
        or _git_output(
            root,
            (
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "--ignore-submodules=none",
            ),
            label="Validator tree",
        )
    ):
        raise IntegrityError("pinned Validator root or tracked tree is invalid")
    for relative, expected in _TRUSTED_VALIDATOR_MODULES.items():
        module_path = root / relative
        if sha256_file(module_path) != expected:
            raise IntegrityError(f"trusted Validator module hash mismatch: {relative}")

    checkout = (root / "models/sources/diffusiondrive_v2").resolve(strict=True)
    if _git_output(
        checkout, ("rev-parse", "HEAD"), label="DDV2 source commit"
    ) != _DDV2_COMMIT or _git_output(
        checkout,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--ignore-submodules=none",
        ),
        label="DDV2 source tree",
    ):
        raise IntegrityError("pinned DDV2 source checkout is invalid")
    derivative = checkout / _DERIVATIVE_RELATIVE_PATH
    if sha256_file(derivative) != _DERIVATIVE_SHA256:
        raise IntegrityError("pinned DDV2 derivative hash mismatch")
    return root


def _validator_module(validator_root: Path, name: str):
    _validator_source_root(validator_root)
    module = importlib.import_module(name)
    _require_validator_module(module, validator_root, name)
    return module


def _validator_load_selected_profile(
    profile_file: Path, model_id: str, validator_root: Path
):
    module = _validator_module(validator_root, "validator_inference.profile_store")
    return module.load_selected_profile(profile_file, model_id)


def _validator_verify_runtime_assets(
    profile: Mapping[str, Any], validator_root: Path
) -> Mapping[str, Any]:
    module = _validator_module(validator_root, "validator_inference.assets")
    return module.verify_runtime_assets(profile, validator_root)


def _validator_load_selector_state_dict(
    network: nn.Module,
    raw_state: dict[str, torch.Tensor],
    *,
    checkpoint_path: Path,
    validator_root: Path,
) -> None:
    module = _validator_module(
        validator_root, "validator_inference.models.diffusiondrive_v2"
    )
    module.load_diffusiondrive_selector_state_dict(
        network,
        raw_state,
        checkpoint_path=checkpoint_path,
    )


def _real_file_identity(
    path: Path, *, label: str
) -> tuple[Path, tuple[int, int, int, int, int]]:
    candidate = Path(path).absolute()
    try:
        status = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise IntegrityError(f"{label} is unavailable") from error
    if (
        candidate != resolved
        or not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
    ):
        raise IntegrityError(f"{label} must be a real single-link regular file")
    return resolved, (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _snapshot_file(path: Path, expected_sha256: str, *, label: str) -> _FileSnapshot:
    before_path, before_identity = _real_file_identity(path, label=label)
    try:
        observed_sha256 = sha256_file(before_path)
    except OSError as error:
        raise IntegrityError(f"unable to hash {label}") from error
    after_path, after_identity = _real_file_identity(before_path, label=label)
    if (
        after_path != before_path
        or after_identity != before_identity
        or observed_sha256 != expected_sha256
    ):
        raise IntegrityError(f"{label} identity or hash changed during verification")
    return _FileSnapshot(before_path, observed_sha256, before_identity)


def _reauthenticate_file(snapshot: _FileSnapshot, *, label: str) -> None:
    current_path, current_identity = _real_file_identity(snapshot.path, label=label)
    if current_path != snapshot.path or current_identity != snapshot.identity:
        raise IntegrityError(f"{label} identity changed after verification")
    try:
        observed_sha256 = sha256_file(current_path)
    except OSError as error:
        raise IntegrityError(f"unable to reauthenticate {label}") from error
    _, after_identity = _real_file_identity(current_path, label=label)
    if after_identity != snapshot.identity or observed_sha256 != snapshot.sha256:
        raise IntegrityError(f"{label} hash changed after verification")


def _reauthenticate_roles(
    snapshots: Mapping[str, _FileSnapshot], roles: tuple[str, ...]
) -> None:
    for role in roles:
        _reauthenticate_file(snapshots[role], label=f"runtime asset {role}")


def resolve_profile_asset(asset: Mapping[str, Any], validator_root: Path) -> Path:
    if not isinstance(asset, Mapping):
        raise IntegrityError("runtime asset declaration must be an object")
    role = asset.get("role")
    value = asset.get("path")
    if type(role) is not str or not role or type(value) is not str or not value:
        raise IntegrityError("runtime asset role and path must be nonempty strings")
    candidate = Path(value)
    root = Path(validator_root).resolve(strict=True)
    if not candidate.is_absolute():
        if any(part in ("", ".", "..") for part in candidate.parts):
            raise IntegrityError(f"runtime asset {role} path is unsafe")
        candidate = root / candidate
    path = candidate.absolute()
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise IntegrityError(
            f"runtime asset {role} is missing or unreadable"
        ) from error
    if not Path(value).is_absolute():
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise IntegrityError(
                f"runtime asset {role} escapes Validator root"
            ) from error
    if path != resolved or not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise IntegrityError(f"runtime asset {role} must be a regular file")
    return resolved


def sel_assets_from_verified_roles(
    profile: Mapping[str, Any],
    roles: Mapping[str, Path],
    validator_root: Path,
) -> SelAssets:
    if not isinstance(profile, Mapping) or profile.get("id") != "diffusiondrive_v2":
        raise IntegrityError("selected profile is not diffusiondrive_v2")
    if not isinstance(roles, Mapping) or set(roles) != _EXPECTED_ROLES:
        raise IntegrityError(
            "verified runtime asset roles are incomplete or duplicated"
        )
    root = Path(validator_root).resolve(strict=True)
    return SelAssets(
        v2_checkout=(root / "models/sources/diffusiondrive_v2").resolve(strict=True),
        validator_root=root,
        checkpoint=Path(roles["checkpoint"]),
        image_backbone=Path(roles["image_backbone"]),
        trajectory_anchor=Path(roles["trajectory_anchor"]),
    )


def _declared_assets(profile: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    values = profile.get("runtime_assets")
    if not isinstance(values, list):
        raise IntegrityError("selected profile runtime assets are invalid")
    seen = set()
    result = []
    for value in values:
        if not isinstance(value, Mapping) or type(value.get("role")) is not str:
            raise IntegrityError("selected profile runtime asset is invalid")
        role = value["role"]
        if (
            type(value.get("path")) is not str
            or not value["path"]
            or type(value.get("sha256")) is not str
            or _SHA256.fullmatch(value["sha256"]) is None
        ):
            raise IntegrityError(
                f"selected profile runtime asset {role} requires a lowercase SHA-256"
            )
        if role in seen:
            raise IntegrityError(f"selected profile runtime role repeats: {role}")
        seen.add(role)
        result.append(value)
    if seen != _EXPECTED_ROLES:
        raise IntegrityError("selected profile runtime asset roles are invalid")
    return tuple(result)


def _raw_selected_profile(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    profiles = profile.get("profiles")
    if not isinstance(profiles, list):
        raise IntegrityError("generated Validator profile lacks profiles")
    matches = [
        item
        for item in profiles
        if isinstance(item, Mapping) and item.get("id") == "diffusiondrive_v2"
    ]
    if len(matches) != 1:
        raise IntegrityError(
            "generated Validator profile must contain one diffusiondrive_v2 profile"
        )
    source = matches[0].get("source")
    if not isinstance(source, Mapping) or source.get("commit") != _DDV2_COMMIT:
        raise IntegrityError("generated Validator profile DDV2 source commit drifted")
    return matches[0]


def _expected_validator_selection(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = copy.deepcopy(dict(_raw_selected_profile(profile)))
    common = profile.get("common_inference_policy")
    if not isinstance(common, Mapping) or not isinstance(
        common.get("static_transforms"), Mapping
    ):
        raise IntegrityError("generated Validator profile common policy is invalid")
    selected["static_transforms"] = copy.deepcopy(common["static_transforms"])
    return selected


def _validator_document_sha256(profile: Mapping[str, Any]) -> str:
    payload = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_behavioral_contract(profile: Mapping[str, Any]) -> None:
    inference = profile.get("inference_policy")
    if (
        profile.get("domain_contract") != _EXPECTED_DOMAIN_CONTRACT
        or profile.get("static_transforms") != _EXPECTED_STATIC_TRANSFORMS
        or not isinstance(inference, Mapping)
        or inference.get("reference_stamp_policy") != "new_stamp_once"
    ):
        raise IntegrityError(
            "generated profile camera, LiDAR, transform, or stamp contract drifted"
        )


def _assert_checkpoint_hash_identity(
    declarations: Mapping[str, Mapping[str, Any]],
    metadata: ExportMetadata,
    checkpoint: Path,
) -> None:
    declared = declarations["checkpoint"].get("sha256")
    actual = sha256_file(checkpoint)
    if (
        type(declared) is not str
        or declared != metadata.export_checkpoint_sha256
        or actual != metadata.export_checkpoint_sha256
    ):
        raise IntegrityError("checkpoint file hash identity mismatch")


def _assert_export_identity(
    profile: Mapping[str, Any],
    metadata: ExportMetadata,
    roles: Mapping[str, Path],
) -> None:
    if (
        metadata.source_commits.get("validator") != _VALIDATOR_COMMIT
        or metadata.source_commits.get("diffusiondrive_v2") != _DDV2_COMMIT
        or "preprocessing" not in metadata.source_commits
        or "training" not in metadata.source_commits
    ):
        raise IntegrityError("source commits do not bind the pinned Validator and DDV2")
    identity = profile.get("export_identity")
    expected = {
        "domain_id": metadata.domain_id,
        "resolved_domain_sha256": metadata.resolved_domain_sha256,
        "export_state_dict_sha256": metadata.export_state_dict_sha256,
        "raw_schema_sha256": metadata.raw_schema_sha256,
        "training_manifest_sha256": metadata.training_manifest_sha256,
        "cache_schema_version": metadata.cache_schema_version,
        "clock_policy_version": metadata.clock_policy_version,
    }
    if identity != expected:
        for name, value in expected.items():
            if not isinstance(identity, Mapping) or identity.get(name) != value:
                raise IntegrityError(f"profile {name} identity mismatch")
        raise IntegrityError("profile export identity mismatch")

    resolved = load_resolved_domain(roles["resolved_domain"])
    source = resolved.document.get("sensor_preset")
    cache = resolved.document.get("cache")
    attributes = cache.get("hdf5_attributes") if isinstance(cache, Mapping) else None
    if (
        resolved.document.get("domain_id") != metadata.domain_id
        or resolved.sha256 != metadata.resolved_domain_sha256
    ):
        raise IntegrityError("resolved domain identity mismatch")
    if resolved.document.get("source_code_commit") != metadata.source_commits.get(
        "preprocessing"
    ):
        raise IntegrityError(
            "preprocessing source commit does not match the resolved domain"
        )
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != metadata.sensor_preset_sha256
        or sha256_file(roles["sensor_preset"]) != metadata.sensor_preset_sha256
    ):
        raise IntegrityError("sensor preset identity mismatch")
    if sha256_file(roles["training_manifest"]) != metadata.training_manifest_sha256:
        raise IntegrityError("training manifest identity mismatch")
    if (
        not isinstance(attributes, Mapping)
        or attributes.get("cache_schema_version") != metadata.cache_schema_version
        or attributes.get("clock_policy_version") != metadata.clock_policy_version
    ):
        raise IntegrityError("cache or clock identity mismatch")


def verify_validator_bundle(
    profile_file: Path, validator_root: Path
) -> VerifiedValidatorBundle:
    root = _validate_pinned_validator_root(validator_root)
    profile_path, profile_document = load_required_profile_document(profile_file)
    expected_selected = _expected_validator_selection(profile_document)
    profile_snapshot = _snapshot_file(
        profile_path,
        hashlib.sha256(canonical_json_bytes(profile_document)).hexdigest(),
        label="generated Validator profile",
    )
    _reauthenticate_file(profile_snapshot, label="generated Validator profile")
    selected = _validator_load_selected_profile(profile_path, "diffusiondrive_v2", root)
    _reauthenticate_file(profile_snapshot, label="generated Validator profile")
    if selected.model_id != "diffusiondrive_v2" or not isinstance(
        selected.data, Mapping
    ):
        raise IntegrityError("Validator selected the wrong profile")
    if canonical_json_bytes(selected.data) != canonical_json_bytes(
        expected_selected
    ) or selected.profile_sha256 != _validator_document_sha256(profile_document):
        raise IntegrityError(
            "Validator selected data does not match the strictly parsed profile"
        )
    _assert_behavioral_contract(selected.data)

    assets = _declared_assets(selected.data)
    declarations = {asset["role"]: asset for asset in assets}
    roles = {
        role: resolve_profile_asset(declaration, root)
        for role, declaration in declarations.items()
    }
    snapshots = {
        role: _snapshot_file(
            roles[role],
            declarations[role]["sha256"],
            label=f"runtime asset {role}",
        )
        for role in sorted(_EXPECTED_ROLES)
    }
    all_roles = tuple(sorted(_EXPECTED_ROLES))
    _reauthenticate_roles(snapshots, all_roles)
    _validator_verify_runtime_assets(selected.data, root)
    _reauthenticate_roles(snapshots, all_roles)

    metadata_roles = ("checkpoint_metadata",)
    _reauthenticate_roles(snapshots, metadata_roles)
    metadata = load_required_metadata(roles["checkpoint_metadata"])
    _reauthenticate_roles(snapshots, metadata_roles)

    checkpoint_roles = ("checkpoint",)
    _reauthenticate_roles(snapshots, checkpoint_roles)
    _assert_checkpoint_hash_identity(declarations, metadata, roles["checkpoint"])
    _reauthenticate_roles(snapshots, checkpoint_roles)

    _reauthenticate_roles(snapshots, checkpoint_roles)
    raw = load_raw_sel_checkpoint(roles["checkpoint"])
    _reauthenticate_roles(snapshots, checkpoint_roles)
    if raw.checkpoint_sha256 != metadata.export_checkpoint_sha256:
        raise IntegrityError("loaded checkpoint file hash identity mismatch")
    if state_dict_sha256(raw.raw_state_dict) != metadata.export_state_dict_sha256:
        raise IntegrityError("export state hash identity mismatch")
    if state_schema_sha256(raw.raw_state_dict) != metadata.raw_schema_sha256:
        raise IntegrityError("raw schema hash identity mismatch")
    identity_roles = ("resolved_domain", "sensor_preset", "training_manifest")
    _reauthenticate_roles(snapshots, identity_roles)
    _assert_export_identity(selected.data, metadata, roles)
    _reauthenticate_roles(snapshots, identity_roles)

    runtime = normalize_sel_runtime_state(raw.raw_state_dict)
    if len(runtime) != RUNTIME_SEL_KEY_COUNT:
        raise SelSchemaError(
            f"expected {RUNTIME_SEL_KEY_COUNT} runtime keys, got {len(runtime)}"
        )
    sel_assets = sel_assets_from_verified_roles(selected.data, roles, root)
    construction_roles = ("image_backbone", "trajectory_anchor")
    _reauthenticate_roles(snapshots, construction_roles)
    network = construct_pdm_free_sel_network(sel_assets, torch.device("cpu"))
    _reauthenticate_roles(snapshots, construction_roles)
    if not isinstance(network, nn.Module):
        raise SelSchemaError("PDM-free constructor did not return torch.nn.Module")
    _validator_load_selector_state_dict(
        network,
        dict(raw.raw_state_dict),
        checkpoint_path=roles["checkpoint"],
        validator_root=root,
    )
    _reauthenticate_roles(snapshots, all_roles)
    _reauthenticate_file(profile_snapshot, label="generated Validator profile")
    return VerifiedValidatorBundle(selected, metadata, raw, runtime, network)


__all__ = (
    "VerifiedValidatorBundle",
    "resolve_profile_asset",
    "sel_assets_from_verified_roles",
    "verify_validator_bundle",
)
