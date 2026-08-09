from __future__ import annotations

import os
import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from morai_ddv2.cache.loader import load_tensor_sample, load_verified_manifest
from morai_ddv2.cli.export_checkpoint import _load_attestation, preflight_export
from morai_ddv2.cli.train import assert_runtime_contract
from morai_ddv2.export.state_hash import state_dict_sha256, tensor_data_sha256
from morai_ddv2.export.validator_preflight import (
    resolve_profile_asset,
    verify_validator_bundle,
)
from morai_ddv2.training.metrics import (
    selector_seed,
    validation_report_json_bytes,
)
from morai_ddv2.training.validation import validate_selector_b1
from morai_ddv2.upstream.ddv2_sel import (
    RAW_SEL_KEY_COUNT,
    RUNTIME_SEL_KEY_COUNT,
    load_raw_sel_checkpoint,
    normalize_sel_key,
    normalize_sel_runtime_state,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILE = _PROJECT_ROOT / "configs/validator/profile-morai-ddv2-oracle.json"
_RUN_DIR = _PROJECT_ROOT / "runs/morai-ddv2-v1-oracle-command"
_RUN_COMPLETE = _RUN_DIR / "run-complete.json"
_ALLOW_PENDING_ENV = "MORAI_DDV2_ALLOW_PENDING_EXPORT_TEST"
_EXPECTED_BASE_CHECKPOINT_SHA256 = (
    "452c4e0dd328c49b981f3fd5ed13eb1c6a2b0f3a9486adf43b40a5645ed543b1"
)
_EXPECTED_BASE_STATE_DICT_SHA256 = (
    "1b0885e47510102e8f0046d0aea9cffc067aab2d00f9001477c0c979d7ddf129"
)
_TRAINABLE_PREFIXES = (
    "_backbone.image_encoder.",
    "_backbone.lidar_encoder.",
    "_backbone.transformers.",
    "_backbone.lidar_channel_to_img.",
    "_backbone.img_channel_to_lidar.",
)
_BN_BUFFER_NAMES = ("running_mean", "running_var", "num_batches_tracked")
_BN_POLICY = "trainable_prefixes:running_mean,running_var,num_batches_tracked"
_FIXED_SAMPLE_IDS = (
    "1d458d3c304abb8e43666752620f56c43d090666b0c11aeed1f8a815340c5831:"
    "1785475008466000000",
    "a9e0c07e4be5b527c7791e5226f828be6bae7db42025d68f44e1c26dd5869808:"
    "1785739615495000000",
)


def _official_profile() -> Path:
    present = tuple(os.path.lexists(path) for path in (_RUN_COMPLETE, _PROFILE))
    if not any(present) and os.environ.get(_ALLOW_PENDING_ENV) == "1":
        pytest.skip("official 30-epoch run/export is explicitly pending")
    if not all(present):
        pytest.fail(
            "official run-complete and exported profile must both exist; "
            f"present={present}"
        )
    return _PROFILE


def _required_validator_root() -> Path:
    value = os.environ.get("MORAI_DDV2_VALIDATOR_ROOT")
    if not value:
        pytest.fail("MORAI_DDV2_VALIDATOR_ROOT is required for the export gate")
    root = Path(value).absolute()
    if not root.is_dir():
        pytest.fail(f"MORAI_DDV2_VALIDATOR_ROOT is not a directory: {root}")
    return root


def _runtime_asset(
    selected_profile: Mapping[str, Any], validator_root: Path, role: str
) -> Path:
    declarations = selected_profile.get("runtime_assets")
    assert isinstance(declarations, list)
    matches = [
        declaration
        for declaration in declarations
        if isinstance(declaration, Mapping) and declaration.get("role") == role
    ]
    assert len(matches) == 1
    return resolve_profile_asset(matches[0], validator_root)


def _independent_mutable_names(model: torch.nn.Module) -> tuple[set[str], set[str]]:
    trainable = {
        name
        for name, _ in model.named_parameters()
        if name.startswith(_TRAINABLE_PREFIXES)
    }
    batchnorm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    buffers = set(dict(model.named_buffers()))
    batchnorm = {
        f"{module_name}.{buffer_name}"
        for module_name, module in model.named_modules()
        if isinstance(module, batchnorm_types)
        and any(
            module_name == prefix[:-1] or module_name.startswith(prefix)
            for prefix in _TRAINABLE_PREFIXES
        )
        for buffer_name in _BN_BUFFER_NAMES
        if f"{module_name}.{buffer_name}" in buffers
    }
    return trainable, batchnorm


def _require_attested_production_run(
    *,
    profile: Path,
    validator_root: Path,
    selected_profile: Mapping[str, Any],
):
    checkpoint = _runtime_asset(selected_profile, validator_root, "checkpoint")
    resolved = _runtime_asset(selected_profile, validator_root, "resolved_domain")
    preset = _runtime_asset(selected_profile, validator_root, "sensor_preset")
    manifest = _runtime_asset(selected_profile, validator_root, "training_manifest")
    base = (
        validator_root
        / "models/diffusiondrive_v2/navsim_v2/diffusiondrivev2_sel.ckpt"
    )
    preflight = preflight_export(
        argparse.Namespace(
            training_checkpoint=_RUN_DIR / "best.pt",
            base_checkpoint=base,
            manifest=manifest,
            resolved_domain=resolved,
            sensor_preset=preset,
            baseline_profile=validator_root / "src/Validator_pkg/config/profile.json",
            output=checkpoint,
            profile_output=profile,
        )
    )
    attestation = _load_attestation(_RUN_COMPLETE)
    assert attestation.completed_epochs == 30
    assert attestation.epoch_telemetry_count == 30
    expected_epochs = [f"epoch-{index:04d}.json" for index in range(30)]
    assert sorted(
        path.name for path in (_RUN_DIR / "metrics/epochs").glob("epoch-*.json")
    ) == expected_epochs
    csv_payload = (_RUN_DIR / "metrics/epoch_metrics.csv").read_bytes()
    assert csv_payload.endswith(b"\n")
    assert len(csv_payload.splitlines()) == 31
    return preflight, attestation


def _fixed_validation_samples(manifest) -> tuple[Mapping[str, Any], ...]:
    available = {record["sample_id"] for record in manifest.sample_records}
    assert set(_FIXED_SAMPLE_IDS).issubset(available)
    result = []
    for sample_id in _FIXED_SAMPLE_IDS:
        sample = load_tensor_sample(manifest, sample_id)
        assert sample.split == "validation"
        result.append(
            {
                "features": {
                    "camera_feature": torch.from_numpy(
                        np.array(sample.camera_feature, copy=True, order="C")
                    ),
                    "lidar_feature": torch.from_numpy(
                        np.array(sample.lidar_feature, copy=True, order="C")
                    ),
                    "status_feature": torch.from_numpy(
                        np.array(sample.status_feature, copy=True, order="C")
                    ),
                },
                "trajectory": torch.from_numpy(
                    np.array(sample.trajectory, copy=True, order="C")
                ),
                "metadata": {
                    "sample_id": sample.sample_id,
                    "timestamp_ns": sample.timestamp_ns,
                    "source_bag_sha256": sample.source_bag_sha256,
                    "command_name": sample.command_name,
                },
            }
        )
    return tuple(result)


@pytest.mark.gpu
@pytest.mark.upstream
@pytest.mark.filterwarnings("ignore::DeprecationWarning:matplotlib\\..*")
def test_exported_selector_strict_loads_and_is_order_invariant_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _official_profile()
    assert_runtime_contract(require_cuda=True)
    validator_root = _required_validator_root()
    verified = verify_validator_bundle(profile, validator_root)

    assert len(verified.raw_checkpoint.raw_state_dict) == RAW_SEL_KEY_COUNT == 972
    assert len(verified.runtime_state) == RUNTIME_SEL_KEY_COUNT == 971
    assert len(verified.network.state_dict()) == RUNTIME_SEL_KEY_COUNT

    selected = verified.selected_profile.data
    preflight, attestation = _require_attested_production_run(
        profile=profile,
        validator_root=validator_root,
        selected_profile=selected,
    )
    assert (
        attestation.training_config_sha256
        == verified.metadata.training_config_sha256
    )
    assert (
        attestation.training_manifest_sha256
        == verified.metadata.training_manifest_sha256
    )
    assert (
        attestation.resolved_domain_sha256
        == verified.metadata.resolved_domain_sha256
    )
    assert attestation.best_micro_ade == verified.metadata.best_micro_ade
    assert attestation.validation_run_id == verified.metadata.validation_run_id

    base = load_raw_sel_checkpoint(
        validator_root
        / "models/diffusiondrive_v2/navsim_v2/diffusiondrivev2_sel.ckpt"
    )
    assert base.checkpoint_sha256 == _EXPECTED_BASE_CHECKPOINT_SHA256
    assert state_dict_sha256(base.raw_state_dict) == _EXPECTED_BASE_STATE_DICT_SHA256
    assert verified.metadata.base_checkpoint_sha256 == _EXPECTED_BASE_CHECKPOINT_SHA256
    assert verified.metadata.trainable_prefixes == _TRAINABLE_PREFIXES
    assert verified.metadata.mutable_bn_policy == _BN_POLICY
    assert attestation.base_checkpoint_sha256 == _EXPECTED_BASE_CHECKPOINT_SHA256
    assert set(base.raw_state_dict) == set(verified.raw_checkpoint.raw_state_dict)
    base_runtime = normalize_sel_runtime_state(base.raw_state_dict)
    assert set(base_runtime) == set(verified.runtime_state)

    trainable, batchnorm = _independent_mutable_names(verified.network)
    mutable = trainable | batchnorm
    assert set(preflight.manifest.trainable_parameters) == trainable
    assert set(preflight.manifest.mutable_state_entries) == mutable
    assert set(preflight.mutable_state) == mutable
    for name in mutable:
        assert tensor_data_sha256(preflight.mutable_state[name]) == tensor_data_sha256(
            verified.runtime_state[name]
        )

    changed = set()
    for raw_name, base_tensor in base.raw_state_dict.items():
        normalized = normalize_sel_key(raw_name)
        exported_tensor = verified.raw_checkpoint.raw_state_dict[raw_name]
        if tensor_data_sha256(base_tensor) != tensor_data_sha256(exported_tensor):
            changed.add(normalized)
        if normalized not in mutable:
            assert tensor_data_sha256(base_tensor) == tensor_data_sha256(exported_tensor)
    assert changed
    assert changed.intersection(trainable)
    assert changed.issubset(mutable)

    manifest_path = _runtime_asset(selected, validator_root, "training_manifest")
    preset_path = _runtime_asset(selected, validator_root, "sensor_preset")
    manifest = load_verified_manifest(
        manifest_path,
        preset_path,
        verified.metadata.domain_id,
        verified.metadata.resolved_domain_sha256,
        production=True,
    )
    assert manifest.manifest_sha256 == verified.metadata.training_manifest_sha256
    samples = _fixed_validation_samples(manifest)

    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    model = verified.network.to(torch.device("cuda", 0))

    forward = validate_selector_b1(model, samples)
    reverse = validate_selector_b1(model, reversed(samples))

    assert forward.micro.count == len(_FIXED_SAMPLE_IDS)
    assert validation_report_json_bytes(forward) == validation_report_json_bytes(
        reverse
    )
    assert tuple(record.sample_id for record in forward.records) == tuple(
        sorted(_FIXED_SAMPLE_IDS)
    )
    assert len({record.seed for record in forward.records}) == len(_FIXED_SAMPLE_IDS)
    for record in forward.records:
        assert record.seed == selector_seed(record.timestamp_ns)
        assert record.prediction.shape == (8, 3)
        assert np.isfinite(record.prediction).all()
