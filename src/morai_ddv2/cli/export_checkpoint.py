from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import json
from pathlib import Path
import stat
from typing import Any, Optional

import torch

from morai_ddv2.atomic import assert_atomic_output_avoids_paths
from morai_ddv2.cache.loader import load_verified_manifest
from morai_ddv2.config import canonical_json_bytes, load_resolved_domain, sha256_file
from morai_ddv2.errors import IntegrityError
from morai_ddv2.export.exporter import ExportBundle, export_sel_bundle
from morai_ddv2.export.metadata import ExportIdentities
from morai_ddv2.export.validator_preflight import verify_validator_bundle
from morai_ddv2.export.validator_profile import build_morai_validator_profile
from morai_ddv2.training.checkpoint import (
    _load_payload,
    _validated_checkpoint_payload,
)
from morai_ddv2.training.contracts import FreezeManifest, ProductionRunAttestation
from morai_ddv2.training.metrics import DOMAIN_ID
from morai_ddv2.upstream.ddv2_sel import load_raw_sel_checkpoint

from .preprocess_bags import probe_source_tree
from .train import (
    _CHECKPOINT_IDENTITY_FIELDS,
    _DDV2_COMMIT,
    _VALIDATOR_COMMIT,
    _profile_assets,
    _require_commit_object,
    _validate_production_domain,
    _validate_production_manifest,
    _validator_root,
    assert_runtime_contract,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RUN_CONFIG_FIELDS = {"schema_version", "training", "identities", "smoke_bounds"}
_SMOKE_FIELDS = {
    "max_epochs",
    "max_train_batches",
    "max_validation_samples",
}


@dataclass(frozen=True)
class ExportPreflight:
    training_checkpoint: Path
    base_checkpoint: Path
    base_checkpoint_sha256: str
    base_file_identity: tuple[int, int, int, int, int]
    baseline_profile: Path
    validator_root: Path
    resolved_domain: Path
    sensor_preset: Path
    training_manifest: Path
    output: Path
    profile_output: Path
    mutable_state: Mapping[str, torch.Tensor]
    manifest: FreezeManifest
    identities: ExportIdentities
    current_source_commit: str
    protected_paths: tuple[Path, ...] = ()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an attested MORAI DiffusionDriveV2 checkpoint"
    )
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resolved-domain", type=Path, required=True)
    parser.add_argument("--sensor-preset", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    return parser


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key: {key}")
        result[key] = value
    return result


def _canonical_json(path: Path, *, label: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IntegrityError(f"unable to load {label}") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != payload:
        raise IntegrityError(f"{label} must be a canonical JSON object")
    return payload, value


def _regular_file(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    try:
        status = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise IntegrityError(f"{label} is not a file") from error
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise IntegrityError(f"{label} must be a regular file")
    return resolved


def _file_identity(path: Path, *, label: str) -> tuple[int, int, int, int, int]:
    resolved = _regular_file(path, label=label)
    try:
        status = resolved.stat()
    except OSError as error:
        raise IntegrityError(f"{label} identity is unavailable") from error
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _load_attestation(path: Path) -> ProductionRunAttestation:
    _, value = _canonical_json(path, label="run-complete attestation")
    if set(value) != {field.name for field in fields(ProductionRunAttestation)}:
        raise IntegrityError("run-complete attestation schema is invalid")
    try:
        return ProductionRunAttestation(**dict(value))
    except (TypeError, ValueError) as error:
        raise IntegrityError("run-complete attestation values are invalid") from error


def _require_hash(path: Path, expected: str, *, label: str) -> Path:
    resolved = _regular_file(path, label=label)
    if sha256_file(resolved) != expected:
        raise IntegrityError(f"{label} SHA-256 does not match run-complete")
    return resolved


def _attested_run(
    training_checkpoint: Path,
    manifest: Path,
    base_checkpoint: Path,
) -> tuple[ProductionRunAttestation, Mapping[str, Any], Path, Path, Path]:
    best = _regular_file(training_checkpoint, label="training checkpoint")
    if best.name != "best.pt":
        raise IntegrityError("export requires the explicitly attested best.pt")
    run_dir = best.parent
    attestation_path = run_dir / "run-complete.json"
    attestation_path = _regular_file(attestation_path, label="run-complete attestation")
    attestation = _load_attestation(attestation_path)
    _require_hash(best, attestation.best_checkpoint_sha256, label="best checkpoint")
    config_path = _require_hash(
        run_dir / "resolved-run-config.json",
        attestation.training_config_sha256,
        label="resolved run config",
    )
    _, config = _canonical_json(config_path, label="resolved run config")
    _require_hash(
        manifest,
        attestation.training_manifest_sha256,
        label="training manifest",
    )
    _require_hash(
        base_checkpoint,
        attestation.base_checkpoint_sha256,
        label="base checkpoint",
    )
    _require_hash(
        run_dir / "metrics/history.jsonl",
        attestation.metrics_history_sha256,
        label="metrics history",
    )
    _require_hash(
        run_dir / "metrics/epoch_metrics.csv",
        attestation.epoch_metrics_csv_sha256,
        label="epoch metrics CSV",
    )
    _require_hash(
        run_dir / "plots/training-curves.png",
        attestation.training_curves_sha256,
        label="training curves",
    )
    return attestation, config, best, config_path, run_dir


def _validate_run_config(
    config: Mapping[str, Any], attestation: ProductionRunAttestation
) -> Mapping[str, str]:
    if set(config) != _RUN_CONFIG_FIELDS or config.get("schema_version") != 1:
        raise IntegrityError("resolved run config schema is invalid")
    training = config.get("training")
    identities = config.get("identities")
    smoke = config.get("smoke_bounds")
    if (
        not isinstance(training, Mapping)
        or not isinstance(identities, Mapping)
        or set(identities) != set(_CHECKPOINT_IDENTITY_FIELDS)
        or not isinstance(smoke, Mapping)
        or set(smoke) != _SMOKE_FIELDS
        or any(smoke[name] is not None for name in _SMOKE_FIELDS)
    ):
        raise IntegrityError("bounded or malformed run config cannot be exported")
    if (
        type(training.get("epochs")) is not int
        or training["epochs"] != attestation.completed_epochs
        or attestation.epoch_telemetry_count != attestation.completed_epochs
        or training.get("validation_run_id") != attestation.validation_run_id
    ):
        raise IntegrityError(
            "completed epoch count or validation run identity mismatch"
        )
    result = {}
    for name, value in identities.items():
        if type(name) is not str or type(value) is not str or not name or not value:
            raise IntegrityError("resolved run identities are invalid")
        result[name] = value
    return result


def _require_identity(identities: Mapping[str, str], name: str, expected: str) -> str:
    observed = identities.get(name)
    if observed != expected:
        raise IntegrityError(f"training {name} identity mismatch")
    return observed


def _metadata_output_path(checkpoint_output: Path) -> Path:
    output = Path(checkpoint_output)
    return output.with_suffix(output.suffix + ".metadata.json")


def _partial_path(path: Path) -> Path:
    candidate = Path(path)
    return candidate.with_name(candidate.name + ".partial")


def _assert_publication_paths_avoid_inputs(
    *,
    checkpoint_output: Path,
    profile_output: Path,
    protected_paths: Sequence[Path],
) -> None:
    finals = (
        Path(checkpoint_output).absolute(),
        _metadata_output_path(checkpoint_output).absolute(),
        Path(profile_output).absolute(),
    )
    candidates = tuple(
        candidate for final in finals for candidate in (final, _partial_path(final))
    )
    try:
        resolved_candidates = tuple(
            candidate.resolve(strict=False) for candidate in candidates
        )
    except OSError as error:
        raise IntegrityError("unable to resolve export publication paths") from error
    if len(set(resolved_candidates)) != len(resolved_candidates):
        raise IntegrityError("export publication output paths collide")

    existing_identities = set()
    for candidate in candidates:
        try:
            status = candidate.stat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise IntegrityError("unable to inspect export publication path") from error
        identity = (status.st_dev, status.st_ino)
        if identity in existing_identities:
            raise IntegrityError("export publication output identities collide")
        existing_identities.add(identity)

    for final, label in zip(
        finals,
        ("export checkpoint", "export metadata", "generated Validator profile"),
    ):
        assert_atomic_output_avoids_paths(
            final,
            protected_paths,
            output_name=label,
        )


def preflight_export(arguments: argparse.Namespace) -> ExportPreflight:
    assert_runtime_contract(require_cuda=False)
    source = probe_source_tree(_PROJECT_ROOT)
    if source.dirty:
        raise IntegrityError("checkpoint export requires a clean source tree")

    attestation, run_config, best, config_path, run_dir = _attested_run(
        Path(arguments.training_checkpoint),
        Path(arguments.manifest),
        Path(arguments.base_checkpoint),
    )
    identities = _validate_run_config(run_config, attestation)
    resolved_path = _regular_file(arguments.resolved_domain, label="resolved domain")
    preset_path = _regular_file(arguments.sensor_preset, label="sensor preset")
    manifest_path = _regular_file(arguments.manifest, label="training manifest")
    baseline_path = _regular_file(
        arguments.baseline_profile, label="baseline Validator profile"
    )
    base_path = _regular_file(arguments.base_checkpoint, label="base checkpoint")

    resolved = load_resolved_domain(resolved_path)
    _validate_production_domain(resolved)
    document = resolved.document
    preset = document.get("sensor_preset")
    cache = document.get("cache")
    attributes = cache.get("hdf5_attributes") if isinstance(cache, Mapping) else None
    if (
        document.get("domain_id") != DOMAIN_ID
        or not isinstance(preset, Mapping)
        or preset.get("sha256") != sha256_file(preset_path)
        or not isinstance(attributes, Mapping)
        or attributes.get("cache_schema_version") != 2
        or attributes.get("clock_policy_version") != 1
    ):
        raise IntegrityError("resolved domain export identity is invalid")
    verified_manifest = load_verified_manifest(
        manifest_path,
        preset_path,
        DOMAIN_ID,
        resolved.sha256,
        production=True,
    )
    _validate_production_manifest(resolved, verified_manifest)
    assets, official_base_sha256, baseline_sha256 = _profile_assets(baseline_path)
    validator_root = _validator_root(baseline_path)
    if assets.checkpoint.resolve(strict=True) != base_path:
        raise IntegrityError("base checkpoint is not the exact baseline profile asset")
    protected_paths = (
        best,
        run_dir / "run-complete.json",
        config_path,
        run_dir / "metrics/history.jsonl",
        run_dir / "metrics/epoch_metrics.csv",
        run_dir / "plots/training-curves.png",
        base_path,
        baseline_path,
        resolved_path,
        preset_path,
        manifest_path,
        assets.image_backbone,
        assets.trajectory_anchor,
    )
    _assert_publication_paths_avoid_inputs(
        checkpoint_output=arguments.output,
        profile_output=arguments.profile_output,
        protected_paths=protected_paths,
    )

    _require_identity(identities, "domain_id", DOMAIN_ID)
    _require_identity(
        identities,
        "training_manifest_sha256",
        verified_manifest.manifest_sha256,
    )
    _require_identity(identities, "resolved_domain_sha256", resolved.sha256)
    _require_identity(identities, "sensor_preset_sha256", sha256_file(preset_path))
    _require_identity(identities, "base_profile_sha256", baseline_sha256)
    _require_identity(identities, "base_checkpoint_sha256", official_base_sha256)
    if (
        attestation.resolved_domain_sha256 != resolved.sha256
        or attestation.training_manifest_sha256 != verified_manifest.manifest_sha256
        or attestation.base_checkpoint_sha256 != official_base_sha256
        or attestation.cache_schema_version != 2
        or attestation.clock_policy_version != 1
    ):
        raise IntegrityError("run-complete identity does not match export inputs")

    _require_identity(identities, "ddv2_source_commit", _DDV2_COMMIT)
    _require_identity(identities, "validator_source_commit", _VALIDATOR_COMMIT)
    _require_identity(
        identities,
        "preprocessing_source_commit",
        document["source_code_commit"],
    )

    commit_roots = {
        "ddv2_source_commit": _PROJECT_ROOT / "third_party/DiffusionDriveV2",
        "validator_source_commit": validator_root,
        "preprocessing_source_commit": _PROJECT_ROOT,
        "training_source_commit": _PROJECT_ROOT,
    }
    for name, repository in commit_roots.items():
        _require_commit_object(repository, identities.get(name, ""), name)

    payload = _load_payload(
        best, expected_checkpoint_sha256=attestation.best_checkpoint_sha256
    )
    validated = _validated_checkpoint_payload(
        payload,
        expected_base_checkpoint_sha256=official_base_sha256,
        expected_checkpoint_sha256=attestation.best_checkpoint_sha256,
        expected_config=run_config,
        expected_identities=identities,
    )
    if validated["best_micro_ade"] != attestation.best_micro_ade:
        raise IntegrityError("best checkpoint metric disagrees with run-complete")
    manifest = validated["manifest"]
    mutable_state = validated["mutable_state"]
    if not isinstance(manifest, FreezeManifest) or not isinstance(
        mutable_state, Mapping
    ):
        raise IntegrityError("training checkpoint export state is invalid")

    export_identities = ExportIdentities(
        domain_id=DOMAIN_ID,
        resolved_domain_sha256=resolved.sha256,
        sensor_preset_sha256=sha256_file(preset_path),
        training_manifest_sha256=verified_manifest.manifest_sha256,
        cache_schema_version=2,
        clock_policy_version=1,
        training_config_sha256=attestation.training_config_sha256,
        source_commits={
            "diffusiondrive_v2": identities["ddv2_source_commit"],
            "validator": identities["validator_source_commit"],
            "preprocessing": identities["preprocessing_source_commit"],
            "training": identities["training_source_commit"],
        },
        best_micro_ade=attestation.best_micro_ade,
        validation_run_id=attestation.validation_run_id,
    )
    return ExportPreflight(
        training_checkpoint=best,
        base_checkpoint=base_path,
        base_checkpoint_sha256=official_base_sha256,
        base_file_identity=_file_identity(base_path, label="base checkpoint"),
        baseline_profile=baseline_path,
        validator_root=validator_root,
        resolved_domain=resolved_path,
        sensor_preset=preset_path,
        training_manifest=manifest_path,
        output=Path(arguments.output),
        profile_output=Path(arguments.profile_output),
        mutable_state=mutable_state,
        manifest=manifest,
        identities=export_identities,
        current_source_commit=source.source_code_commit,
        protected_paths=tuple(
            Path(path).resolve(strict=True) for path in protected_paths
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    preflight = preflight_export(arguments)
    if (
        _file_identity(preflight.base_checkpoint, label="base checkpoint")
        != preflight.base_file_identity
        or sha256_file(preflight.base_checkpoint) != preflight.base_checkpoint_sha256
    ):
        raise IntegrityError("base checkpoint changed after export preflight")
    base = load_raw_sel_checkpoint(preflight.base_checkpoint)
    if (
        base.checkpoint_sha256 != preflight.base_checkpoint_sha256
        or _file_identity(preflight.base_checkpoint, label="base checkpoint")
        != preflight.base_file_identity
        or sha256_file(preflight.base_checkpoint) != preflight.base_checkpoint_sha256
    ):
        raise IntegrityError("loaded base checkpoint failed reauthentication")
    _assert_publication_paths_avoid_inputs(
        checkpoint_output=preflight.output,
        profile_output=preflight.profile_output,
        protected_paths=preflight.protected_paths,
    )
    bundle: ExportBundle = export_sel_bundle(
        base,
        preflight.mutable_state,
        preflight.output,
        identities=preflight.identities,
        manifest=preflight.manifest,
    )
    profile = build_morai_validator_profile(
        baseline_profile=preflight.baseline_profile,
        output_profile=preflight.profile_output,
        validator_root=preflight.validator_root,
        bundle=bundle,
        resolved_domain=preflight.resolved_domain,
        sensor_preset=preflight.sensor_preset,
        training_manifest=preflight.training_manifest,
    )
    verify_validator_bundle(profile, preflight.validator_root)
    return 0


__all__ = ("ExportPreflight", "main", "preflight_export")
