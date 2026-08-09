from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from morai_ddv2.config import load_resolved_domain, sha256_file
from morai_ddv2.export.exporter import ExportBundle
from morai_ddv2.export.metadata import (
    MUTABLE_BN_POLICY,
    ExportMetadata,
    write_metadata_atomic,
)
from morai_ddv2.training.contracts import TRAINABLE_PREFIXES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_ROOT = PROJECT_ROOT / "third_party/E2E-OpenLoop-ROS-Validator"
BASELINE_PROFILE = VALIDATOR_ROOT / "src/Validator_pkg/config/profile.json"
RESOLVED_DOMAIN = PROJECT_ROOT / "tests/fixtures/golden/resolved-domain.json"
SENSOR_PRESET = PROJECT_ROOT / "sensor_data/0725demo_ros.json"


def _metadata(*, checkpoint: Path, manifest: Path) -> ExportMetadata:
    resolved = load_resolved_domain(RESOLVED_DOMAIN)
    return ExportMetadata(
        base_checkpoint_sha256="1" * 64,
        base_state_dict_sha256="2" * 64,
        export_checkpoint_sha256=sha256_file(checkpoint),
        export_state_dict_sha256="3" * 64,
        raw_schema_sha256="4" * 64,
        resolved_domain_sha256=resolved.sha256,
        sensor_preset_sha256=sha256_file(SENSOR_PRESET),
        training_manifest_sha256=sha256_file(manifest),
        cache_schema_version=2,
        clock_policy_version=1,
        training_config_sha256="5" * 64,
        domain_id="morai-ddv2-v1-oracle-command",
        source_commits={
            "training": "1" * 40,
            "preprocessing": "2" * 40,
            "validator": "3" * 40,
            "diffusiondrive_v2": "4" * 40,
        },
        trainable_prefixes=TRAINABLE_PREFIXES,
        mutable_bn_policy=MUTABLE_BN_POLICY,
        raw_key_count=972,
        runtime_key_count=971,
        best_micro_ade=1.25,
        validation_run_id="morai-ddv2-v1-oracle-command-validation-v1",
        exported_at_utc="2026-08-09T00:00:00.000000Z",
    )


def _bundle(tmp_path: Path) -> tuple[ExportBundle, Path]:
    checkpoint = tmp_path / "ddv2-morai.ckpt"
    checkpoint.write_bytes(b"schema-preserving-checkpoint")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_bytes(b'{"schema_version":2}\n')
    metadata_path = checkpoint.with_suffix(".ckpt.metadata.json")
    metadata = _metadata(checkpoint=checkpoint, manifest=manifest)
    write_metadata_atomic(metadata_path, metadata)
    return (
        ExportBundle(
            checkpoint=checkpoint,
            metadata=metadata_path,
            checkpoint_sha256=metadata.export_checkpoint_sha256,
            state_dict_sha256=metadata.export_state_dict_sha256,
        ),
        manifest,
    )


def _load_selected_profile(profile: Path):
    source = VALIDATOR_ROOT / "src/Validator_pkg/validator_inference/src"
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("validator_inference.profile_store")
    return module.load_selected_profile(profile, "diffusiondrive_v2")


def test_profile_generation_preserves_baseline_and_binds_exact_domain_contract(
    tmp_path: Path,
) -> None:
    from morai_ddv2.export.validator_profile import build_morai_validator_profile

    bundle, manifest = _bundle(tmp_path)
    output = tmp_path / "configs/validator/profile-morai-ddv2-oracle.json"
    baseline_before = BASELINE_PROFILE.read_bytes()

    result = build_morai_validator_profile(
        baseline_profile=BASELINE_PROFILE,
        output_profile=output,
        validator_root=VALIDATOR_ROOT,
        bundle=bundle,
        resolved_domain=RESOLVED_DOMAIN,
        sensor_preset=SENSOR_PRESET,
        training_manifest=manifest,
    )

    assert result == output.resolve()
    assert BASELINE_PROFILE.read_bytes() == baseline_before
    assert (
        hashlib.sha256(BASELINE_PROFILE.read_bytes()).hexdigest()
        == hashlib.sha256(baseline_before).hexdigest()
    )
    profile = json.loads(output.read_text(encoding="utf-8"))
    matches = [
        item for item in profile["profiles"] if item["id"] == "diffusiondrive_v2"
    ]
    assert len(matches) == 1
    selected = matches[0]
    assert profile["selection"]["allowed_values"].count("diffusiondrive_v2") == 1
    assert "morai_diffusiondrive_v2" not in profile["selection"]["allowed_values"]

    roles = {asset["role"]: asset for asset in selected["runtime_assets"]}
    assert set(roles) == {
        "checkpoint",
        "checkpoint_metadata",
        "resolved_domain",
        "sensor_preset",
        "training_manifest",
        "image_backbone",
        "trajectory_anchor",
    }
    for asset in roles.values():
        path = Path(asset["path"])
        assert path.is_absolute()
        assert path.is_file()
        assert asset["sha256"] == sha256_file(path)

    metadata = _metadata(checkpoint=bundle.checkpoint, manifest=manifest)
    assert selected["export_identity"] == {
        "domain_id": "morai-ddv2-v1-oracle-command",
        "resolved_domain_sha256": metadata.resolved_domain_sha256,
        "export_state_dict_sha256": metadata.export_state_dict_sha256,
        "raw_schema_sha256": metadata.raw_schema_sha256,
        "training_manifest_sha256": metadata.training_manifest_sha256,
        "cache_schema_version": 2,
        "clock_policy_version": 1,
    }
    assert selected["inference_policy"]["reference_stamp_policy"] == "new_stamp_once"

    camera = selected["domain_contract"]["camera"]
    assert camera["source_resolution_wh"] == {
        "front": [1280, 720],
        "left": [640, 480],
        "right": [640, 480],
    }
    assert camera["resize_wh"] == {
        "front": [455, 256],
        "left": [341, 256],
        "right": [341, 256],
    }
    assert camera["outer_crop_x"] == [56, 1080]
    assert camera["output"] == {
        "color_order": "RGB",
        "dtype": "float32",
        "shape_chw": [3, 256, 1024],
        "value_range": [0, 1],
    }

    lidar = selected["domain_contract"]["lidar"]
    assert lidar["translation_base_from_lidar_m"] == [1.5, 0, 1.25]
    assert lidar["self_mask_base_link_m"] == {
        "x": [0, 2],
        "y": [-1, 1],
        "z": [0.8, 1.6],
        "boundaries_inclusive": True,
    }
    assert lidar["normalization"] == "count/5"

    transform = profile["common_inference_policy"]["static_transforms"][
        "T_base_from_lidar"
    ]
    assert transform["translation_m"] == [1.5, 0, 1.25]
    assert transform["quaternion_xyzw"] == [0, 0, 0, 1]
    loaded = _load_selected_profile(output)
    assert loaded.model_id == "diffusiondrive_v2"
    assert loaded.data["static_transforms"]["T_base_from_lidar"] == transform


def test_profile_generation_rejects_noncanonical_output_name_and_identity_drift(
    tmp_path: Path,
) -> None:
    from morai_ddv2.export.validator_profile import build_morai_validator_profile

    bundle, manifest = _bundle(tmp_path)
    arguments = {
        "baseline_profile": BASELINE_PROFILE,
        "output_profile": tmp_path / "wrong-profile.json",
        "validator_root": VALIDATOR_ROOT,
        "bundle": bundle,
        "resolved_domain": RESOLVED_DOMAIN,
        "sensor_preset": SENSOR_PRESET,
        "training_manifest": manifest,
    }
    with pytest.raises(Exception, match="profile-morai-ddv2-oracle"):
        build_morai_validator_profile(**arguments)

    arguments["output_profile"] = (
        tmp_path / "configs/validator/profile-morai-ddv2-oracle.json"
    )
    drifted_preset = tmp_path / "preset.json"
    drifted_preset.write_bytes(SENSOR_PRESET.read_bytes() + b"\n")
    arguments["sensor_preset"] = drifted_preset
    with pytest.raises(Exception, match="sensor preset"):
        build_morai_validator_profile(**arguments)


def test_validator_profile_ignore_rule_is_exact() -> None:
    assert (PROJECT_ROOT / "configs/validator/.gitignore").read_bytes() == (
        b"/profile-morai-ddv2-oracle.json\n"
    )


@pytest.mark.parametrize("collision", ["final", "partial"])
def test_profile_output_final_and_partial_cannot_alias_any_immutable_input(
    tmp_path: Path, collision: str
) -> None:
    from morai_ddv2.errors import IntegrityError
    from morai_ddv2.export.validator_profile import build_morai_validator_profile

    bundle, manifest = _bundle(tmp_path)
    output = tmp_path / "configs/validator/profile-morai-ddv2-oracle.json"
    protected = (
        output if collision == "final" else output.with_name(output.name + ".partial")
    )
    protected.parent.mkdir(parents=True, exist_ok=True)
    manifest.rename(protected)
    before = protected.read_bytes()

    with pytest.raises(IntegrityError, match="collides|immutable"):
        build_morai_validator_profile(
            baseline_profile=BASELINE_PROFILE,
            output_profile=output,
            validator_root=VALIDATOR_ROOT,
            bundle=bundle,
            resolved_domain=RESOLVED_DOMAIN,
            sensor_preset=SENSOR_PRESET,
            training_manifest=protected,
        )

    assert protected.read_bytes() == before
