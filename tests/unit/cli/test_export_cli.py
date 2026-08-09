from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from morai_ddv2.config import canonical_json_bytes, sha256_file
from morai_ddv2.errors import IntegrityError
from morai_ddv2.export.exporter import ExportBundle
from morai_ddv2.export.metadata import ExportIdentities
from morai_ddv2.training.contracts import FreezeManifest, ProductionRunAttestation
from morai_ddv2.upstream.ddv2_sel import RawSelCheckpoint, SelAssets


HISTORICAL_TRAINING_COMMIT = "db8efccff5e9642657a1833724fa9a84709f2c9d"
DDV2_COMMIT = "1cd12a1e155c34dcc471261835444c8d5587580b"
VALIDATOR_COMMIT = "09a3b88e19ae3354903ed676a35e02416284cb58"
PREPROCESSING_COMMIT = "c2140f2177d69d33f21b8bc1045bda6670e82d16"


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _arguments(tmp_path: Path) -> argparse.Namespace:
    run = tmp_path / "run"
    validator = tmp_path / "validator"
    return argparse.Namespace(
        training_checkpoint=_write(run / "best.pt", b"best-checkpoint"),
        base_checkpoint=_write(validator / "base.ckpt", b"base-checkpoint"),
        manifest=_write(tmp_path / "manifest.jsonl", b'{"schema_version":2}\n'),
        resolved_domain=_write(tmp_path / "resolved-domain.json", b"resolved"),
        sensor_preset=_write(tmp_path / "preset.json", b"preset"),
        baseline_profile=_write(validator / "profile.json", b"profile"),
        output=tmp_path / "artifacts/ddv2-morai.ckpt",
        profile_output=tmp_path / "configs/validator/profile-morai-ddv2-oracle.json",
    )


def _complete_run(arguments: argparse.Namespace, *, epochs: int = 30):
    run = arguments.training_checkpoint.parent
    identities = {
        "domain_id": "morai-ddv2-v1-oracle-command",
        "training_manifest_sha256": sha256_file(arguments.manifest),
        "resolved_domain_sha256": "a" * 64,
        "sensor_preset_sha256": sha256_file(arguments.sensor_preset),
        "base_profile_sha256": sha256_file(arguments.baseline_profile),
        "base_checkpoint_sha256": sha256_file(arguments.base_checkpoint),
        "ddv2_source_commit": DDV2_COMMIT,
        "validator_source_commit": VALIDATOR_COMMIT,
        "preprocessing_source_commit": PREPROCESSING_COMMIT,
        "training_source_commit": HISTORICAL_TRAINING_COMMIT,
    }
    config = {
        "schema_version": 1,
        "training": {
            "epochs": 30,
            "validation_run_id": "morai-ddv2-v1-oracle-command-validation-v1",
        },
        "identities": identities,
        "smoke_bounds": {
            "max_epochs": None,
            "max_train_batches": None,
            "max_validation_samples": None,
        },
    }
    config_path = _write(run / "resolved-run-config.json", canonical_json_bytes(config))
    history = _write(run / "metrics/history.jsonl", b"history\n")
    csv = _write(run / "metrics/epoch_metrics.csv", b"epoch,loss\n")
    plot = _write(run / "plots/training-curves.png", b"\x89PNG\r\n")
    attestation = ProductionRunAttestation(
        training_config_sha256=sha256_file(config_path),
        training_manifest_sha256=sha256_file(arguments.manifest),
        resolved_domain_sha256="a" * 64,
        cache_schema_version=2,
        clock_policy_version=1,
        base_checkpoint_sha256=sha256_file(arguments.base_checkpoint),
        completed_epochs=epochs,
        best_checkpoint_sha256=sha256_file(arguments.training_checkpoint),
        best_micro_ade=1.25,
        validation_run_id="morai-ddv2-v1-oracle-command-validation-v1",
        epoch_telemetry_count=epochs,
        metrics_history_sha256=sha256_file(history),
        epoch_metrics_csv_sha256=sha256_file(csv),
        training_curves_sha256=sha256_file(plot),
    )
    _write(run / "run-complete.json", canonical_json_bytes(asdict(attestation)))
    return config, identities, attestation


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _main_preflight(module, arguments: argparse.Namespace):
    return module.ExportPreflight(
        training_checkpoint=arguments.training_checkpoint.resolve(),
        base_checkpoint=arguments.base_checkpoint.resolve(),
        base_checkpoint_sha256=sha256_file(arguments.base_checkpoint),
        base_file_identity=_file_identity(arguments.base_checkpoint),
        baseline_profile=arguments.baseline_profile.resolve(),
        validator_root=arguments.base_checkpoint.parent.resolve(),
        resolved_domain=arguments.resolved_domain.resolve(),
        sensor_preset=arguments.sensor_preset.resolve(),
        training_manifest=arguments.manifest.resolve(),
        output=arguments.output,
        profile_output=arguments.profile_output,
        mutable_state={"weight": torch.tensor([1.0])},
        manifest=FreezeManifest(("weight",), (), ("weight",)),
        identities=ExportIdentities(
            domain_id="morai-ddv2-v1-oracle-command",
            resolved_domain_sha256="a" * 64,
            sensor_preset_sha256="b" * 64,
            training_manifest_sha256="c" * 64,
            cache_schema_version=2,
            clock_policy_version=1,
            training_config_sha256="d" * 64,
            source_commits={"training": HISTORICAL_TRAINING_COMMIT},
            best_micro_ade=1.25,
            validation_run_id="morai-ddv2-v1-oracle-command-validation-v1",
        ),
        current_source_commit="9" * 40,
    )


def _patch_external_preflight(
    monkeypatch: pytest.MonkeyPatch,
    module,
    arguments: argparse.Namespace,
    config,
    identities,
):
    manifest = FreezeManifest(
        trainable_parameters=("_backbone.image_encoder.weight",),
        frozen_parameters=("_trajectory_head.weight",),
        mutable_state_entries=("_backbone.image_encoder.weight",),
    )
    mutable = {"_backbone.image_encoder.weight": torch.tensor([1.0])}
    monkeypatch.setattr(module, "assert_runtime_contract", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "probe_source_tree",
        lambda root: SimpleNamespace(
            source_code_commit="9" * 40,
            dirty=False,
            dirty_diff_sha256=None,
        ),
    )
    commits = []

    def require_commit(repository, commit, label):
        del repository, label
        commits.append(commit)
        return commit

    monkeypatch.setattr(module, "_require_commit_object", require_commit)
    monkeypatch.setattr(
        module,
        "load_resolved_domain",
        lambda path: SimpleNamespace(
            sha256="a" * 64,
            document={
                "domain_id": "morai-ddv2-v1-oracle-command",
                "source_code_commit": PREPROCESSING_COMMIT,
                "sensor_preset": {"sha256": sha256_file(arguments.sensor_preset)},
                "cache": {
                    "schema_version": 2,
                    "hdf5_attributes": {
                        "cache_schema_version": 2,
                        "clock_policy_version": 1,
                    },
                },
            },
        ),
    )
    monkeypatch.setattr(module, "_validate_production_domain", lambda resolved: None)
    verified_manifest = SimpleNamespace(manifest_sha256=sha256_file(arguments.manifest))
    monkeypatch.setattr(
        module,
        "load_verified_manifest",
        lambda *args, **kwargs: verified_manifest,
    )
    monkeypatch.setattr(
        module, "_validate_production_manifest", lambda resolved, manifest: None
    )
    assets = SelAssets(
        v2_checkout=arguments.base_checkpoint.parent,
        validator_root=arguments.base_checkpoint.parent,
        checkpoint=arguments.base_checkpoint,
        image_backbone=arguments.base_checkpoint,
        trajectory_anchor=arguments.base_checkpoint,
    )
    monkeypatch.setattr(
        module,
        "_profile_assets",
        lambda profile: (
            assets,
            sha256_file(arguments.base_checkpoint),
            sha256_file(arguments.baseline_profile),
        ),
    )
    monkeypatch.setattr(
        module,
        "_validator_root",
        lambda profile: arguments.base_checkpoint.parent.resolve(),
    )
    monkeypatch.setattr(module, "_load_payload", lambda *args, **kwargs: {"raw": True})
    monkeypatch.setattr(
        module,
        "_validated_checkpoint_payload",
        lambda *args, **kwargs: {
            "mutable_state": mutable,
            "manifest": manifest,
            "config": config,
            "best_micro_ade": 1.25,
        },
    )
    return manifest, mutable, commits


def test_export_preflight_accepts_attested_historical_training_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    config, identities, _ = _complete_run(arguments)
    manifest, mutable, commits = _patch_external_preflight(
        monkeypatch, module, arguments, config, identities
    )

    verified = module.preflight_export(arguments)

    assert verified.training_checkpoint == arguments.training_checkpoint.resolve()
    assert verified.base_checkpoint_sha256 == sha256_file(arguments.base_checkpoint)
    assert verified.base_file_identity == _file_identity(arguments.base_checkpoint)
    assert verified.mutable_state.keys() == mutable.keys()
    assert verified.manifest == manifest
    assert verified.identities.source_commits["training"] == HISTORICAL_TRAINING_COMMIT
    assert verified.current_source_commit == "9" * 40
    assert (
        verified.current_source_commit != verified.identities.source_commits["training"]
    )
    assert HISTORICAL_TRAINING_COMMIT in commits


@pytest.mark.parametrize(
    "identity_name",
    [
        "ddv2_source_commit",
        "validator_source_commit",
        "preprocessing_source_commit",
    ],
)
def test_export_preflight_rejects_nonexact_source_pins_but_not_historical_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_name: str,
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    config, identities, attestation = _complete_run(arguments)
    config["identities"][identity_name] = "8" * 40
    config_path = arguments.training_checkpoint.parent / "resolved-run-config.json"
    config_path.write_bytes(canonical_json_bytes(config))
    changed = replace(attestation, training_config_sha256=sha256_file(config_path))
    (arguments.training_checkpoint.parent / "run-complete.json").write_bytes(
        canonical_json_bytes(asdict(changed))
    )
    _patch_external_preflight(monkeypatch, module, arguments, config, identities)

    with pytest.raises(IntegrityError, match="source|commit|identity"):
        module.preflight_export(arguments)


@pytest.mark.parametrize(
    "mutation",
    ["best", "config", "manifest", "base", "epochs", "history"],
)
def test_export_preflight_rejects_every_attested_artifact_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    config, identities, attestation = _complete_run(arguments)
    _patch_external_preflight(monkeypatch, module, arguments, config, identities)
    if mutation == "best":
        arguments.training_checkpoint.write_bytes(b"changed-best")
    elif mutation == "config":
        (arguments.training_checkpoint.parent / "resolved-run-config.json").write_bytes(
            b"changed-config"
        )
    elif mutation == "manifest":
        arguments.manifest.write_bytes(b"changed-manifest")
    elif mutation == "base":
        arguments.base_checkpoint.write_bytes(b"changed-base")
    elif mutation == "history":
        (arguments.training_checkpoint.parent / "metrics/history.jsonl").write_bytes(
            b"changed-history"
        )
    else:
        changed = replace(
            attestation,
            completed_epochs=29,
            epoch_telemetry_count=29,
        )
        (arguments.training_checkpoint.parent / "run-complete.json").write_bytes(
            canonical_json_bytes(asdict(changed))
        )

    with pytest.raises(Exception):
        module.preflight_export(arguments)


def test_export_preflight_rejects_last_interrupted_and_bounded_debug_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    config, identities, _ = _complete_run(arguments)
    _patch_external_preflight(monkeypatch, module, arguments, config, identities)
    last = arguments.training_checkpoint.with_name("last.pt")
    last.write_bytes(arguments.training_checkpoint.read_bytes())
    arguments.training_checkpoint = last
    with pytest.raises(IntegrityError, match="best.pt"):
        module.preflight_export(arguments)

    arguments.training_checkpoint = _write(
        tmp_path / "bounded/debug/best.pt", b"bounded"
    )
    with pytest.raises(IntegrityError, match="run-complete"):
        module.preflight_export(arguments)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_export_preflight_rejects_run_complete_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    config, identities, _ = _complete_run(arguments)
    _patch_external_preflight(monkeypatch, module, arguments, config, identities)
    attestation = arguments.training_checkpoint.parent / "run-complete.json"
    real = attestation.with_name("run-complete-real.json")
    attestation.rename(real)
    if alias_kind == "symlink":
        attestation.symlink_to(real)
    else:
        os.link(real, attestation)

    with pytest.raises(IntegrityError, match="run-complete|regular"):
        module.preflight_export(arguments)


@pytest.mark.parametrize(
    "candidate_name",
    [
        "checkpoint_final",
        "checkpoint_partial",
        "metadata_final",
        "metadata_partial",
        "profile_final",
        "profile_partial",
    ],
)
def test_export_preflight_rejects_every_publication_path_aliasing_an_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_name: str,
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    metadata = arguments.output.with_suffix(arguments.output.suffix + ".metadata.json")
    candidates = {
        "checkpoint_final": arguments.output,
        "checkpoint_partial": arguments.output.with_name(
            arguments.output.name + ".partial"
        ),
        "metadata_final": metadata,
        "metadata_partial": metadata.with_name(metadata.name + ".partial"),
        "profile_final": arguments.profile_output,
        "profile_partial": arguments.profile_output.with_name(
            arguments.profile_output.name + ".partial"
        ),
    }
    protected = _write(candidates[candidate_name], b"resolved")
    arguments.resolved_domain = protected
    config, identities, _ = _complete_run(arguments)
    _patch_external_preflight(monkeypatch, module, arguments, config, identities)
    before = protected.read_bytes()

    with pytest.raises(IntegrityError, match="output|collid|immutable|publication"):
        module.preflight_export(arguments)

    assert protected.read_bytes() == before


def test_export_preflight_rejects_cross_output_path_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    arguments.output = arguments.profile_output
    config, identities, _ = _complete_run(arguments)
    _patch_external_preflight(monkeypatch, module, arguments, config, identities)

    with pytest.raises(IntegrityError, match="output|collid|publication"):
        module.preflight_export(arguments)


@pytest.mark.parametrize("mutation", ["missing_validation_run_id", "extra_identity"])
def test_export_preflight_requires_exact_run_config_identity_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    config, identities, attestation = _complete_run(arguments)
    _patch_external_preflight(monkeypatch, module, arguments, config, identities)
    if mutation == "missing_validation_run_id":
        del config["training"]["validation_run_id"]
    else:
        config["identities"]["unattested_extra"] = "extra"
    config_path = arguments.training_checkpoint.parent / "resolved-run-config.json"
    config_path.write_bytes(canonical_json_bytes(config))
    changed = replace(attestation, training_config_sha256=sha256_file(config_path))
    (arguments.training_checkpoint.parent / "run-complete.json").write_bytes(
        canonical_json_bytes(asdict(changed))
    )

    with pytest.raises(IntegrityError, match="config|identit"):
        module.preflight_export(arguments)


def test_export_main_publishes_checkpoint_metadata_profile_then_preflights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    output_metadata = arguments.output.with_suffix(".ckpt.metadata.json")
    preflight = _main_preflight(module, arguments)
    raw_state = OrderedDict(
        (f"key_{index}", torch.tensor([1.0])) for index in range(972)
    )
    base = RawSelCheckpoint(
        container={"state_dict": raw_state},
        raw_state_dict=raw_state,
        checkpoint_sha256=sha256_file(arguments.base_checkpoint),
    )
    bundle = ExportBundle(arguments.output, output_metadata, "f" * 64, "1" * 64)
    events = []
    monkeypatch.setattr(
        module, "_parser", lambda: SimpleNamespace(parse_args=lambda argv: arguments)
    )
    monkeypatch.setattr(module, "preflight_export", lambda current: preflight)

    def load_base(path):
        events.append("base")
        return base

    def export(*args, **kwargs):
        events.append("checkpoint_metadata")
        return bundle

    def profile(**kwargs):
        events.append("profile")
        return arguments.profile_output

    def verify(profile_file, validator_root):
        del profile_file, validator_root
        events.append("preflight")
        return object()

    monkeypatch.setattr(module, "load_raw_sel_checkpoint", load_base)
    monkeypatch.setattr(module, "export_sel_bundle", export)
    monkeypatch.setattr(module, "build_morai_validator_profile", profile)
    monkeypatch.setattr(module, "verify_validator_bundle", verify)

    assert module.main([]) == 0
    assert events == ["base", "checkpoint_metadata", "profile", "preflight"]


@pytest.mark.parametrize("mutation", ["loaded_hash", "file_identity"])
def test_export_main_reauthenticates_base_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    preflight = _main_preflight(module, arguments)
    if mutation == "file_identity":
        arguments.base_checkpoint.write_bytes(b"base-checkpoint-replaced")
    loaded = []
    published = []

    def load_base(path):
        loaded.append(path)
        return RawSelCheckpoint(
            container={},
            raw_state_dict=OrderedDict(),
            checkpoint_sha256="0" * 64,
        )

    monkeypatch.setattr(
        module, "_parser", lambda: SimpleNamespace(parse_args=lambda argv: arguments)
    )
    monkeypatch.setattr(module, "preflight_export", lambda current: preflight)
    monkeypatch.setattr(module, "load_raw_sel_checkpoint", load_base)
    monkeypatch.setattr(
        module,
        "export_sel_bundle",
        lambda *args, **kwargs: published.append(True),
    )

    with pytest.raises(IntegrityError, match="base checkpoint"):
        module.main([])

    assert published == []
    assert loaded == (
        [] if mutation == "file_identity" else [preflight.base_checkpoint]
    )


def test_export_main_rechecks_output_collisions_immediately_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.cli import export_checkpoint as module

    arguments = _arguments(tmp_path)
    protected = _write(
        arguments.output.with_name(arguments.output.name + ".partial"),
        b"resolved",
    )
    arguments.resolved_domain = protected
    preflight = replace(
        _main_preflight(module, arguments),
        protected_paths=(
            arguments.base_checkpoint.resolve(),
            arguments.resolved_domain.resolve(),
            arguments.sensor_preset.resolve(),
            arguments.manifest.resolve(),
            arguments.baseline_profile.resolve(),
            arguments.training_checkpoint.resolve(),
        ),
    )
    loaded = RawSelCheckpoint(
        container={},
        raw_state_dict=OrderedDict(),
        checkpoint_sha256=preflight.base_checkpoint_sha256,
    )
    published = []
    monkeypatch.setattr(
        module, "_parser", lambda: SimpleNamespace(parse_args=lambda argv: arguments)
    )
    monkeypatch.setattr(module, "preflight_export", lambda current: preflight)
    monkeypatch.setattr(module, "load_raw_sel_checkpoint", lambda path: loaded)
    monkeypatch.setattr(
        module,
        "export_sel_bundle",
        lambda *args, **kwargs: published.append(True),
    )

    with pytest.raises(IntegrityError, match="output|collid|immutable|publication"):
        module.main([])

    assert published == []
    assert protected.read_bytes() == b"resolved"


def test_validation_cli_propagates_preflight_failure_and_returns_zero_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.cli import validate_export as module

    profile = _write(tmp_path / "profile.json", b"{}")
    validator = tmp_path / "validator"
    validator.mkdir()
    arguments = argparse.Namespace(profile=profile, validator_root=validator)
    monkeypatch.setattr(
        module, "_parser", lambda: SimpleNamespace(parse_args=lambda argv: arguments)
    )
    monkeypatch.setattr(module, "verify_validator_bundle", lambda *args: object())
    assert module.main([]) == 0

    def fail(*args):
        del args
        raise IntegrityError("missing checkpoint metadata")

    monkeypatch.setattr(module, "verify_validator_bundle", fail)
    with pytest.raises(IntegrityError, match="metadata"):
        module.main([])
