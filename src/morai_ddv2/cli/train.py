from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from morai_ddv2.cache.loader import VerifiedManifest, load_verified_manifest
from morai_ddv2.cache.manifest import assign_bag_splits
from morai_ddv2.config import ResolvedDomain, load_resolved_domain, sha256_file
from morai_ddv2.errors import ConfigurationError, IntegrityError
from morai_ddv2.training.dataset import (
    TrainingConfig,
    build_dataloaders,
)
from morai_ddv2.training.checkpoint import inspect_training_checkpoint
from morai_ddv2.training.metrics import DOMAIN_ID
from morai_ddv2.training.trainer import fit
from morai_ddv2.upstream import SelAssets, load_pdm_free_sel_network


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_FIELDS = tuple(TrainingConfig.__dataclass_fields__)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DDV2_COMMIT = "1cd12a1e155c34dcc471261835444c8d5587580b"
_VALIDATOR_COMMIT = "09a3b88e19ae3354903ed676a35e02416284cb58"
_PREPROCESSING_SOURCE_COMMIT = "c2140f2177d69d33f21b8bc1045bda6670e82d16"
_RESOLVED_DOMAIN_SHA256 = (
    "4aeee4bf28297da2e34cb7a7028a59c95b971c6dcc4c66f90a2cf54bd891350c"
)
_DERIVATIVE_RELATIVE_PATH = Path(
    "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py"
)
_CHECKPOINT_IDENTITY_FIELDS = (
    "domain_id",
    "training_manifest_sha256",
    "resolved_domain_sha256",
    "sensor_preset_sha256",
    "base_profile_sha256",
    "base_checkpoint_sha256",
    "ddv2_source_commit",
    "validator_source_commit",
    "preprocessing_source_commit",
    "training_source_commit",
)
_ASSET_HASHES = {
    "checkpoint": "452c4e0dd328c49b981f3fd5ed13eb1c6a2b0f3a9486adf43b40a5645ed543b1",
    "image_backbone": "f6c1d7785b5d0e796a83f9643fe2cb952ac95740daa9fb2d5c8c6514fdadaeb3",
    "trajectory_anchor": "2d341e525d7882ee5e916c67d37bea571e4c793cf801f0fff3eff59d38b20d98",
}


@dataclass(frozen=True)
class TrainingPreflight:
    config: TrainingConfig
    manifest: VerifiedManifest
    assets: SelAssets
    resolved_run_config: Mapping[str, Any]
    identities: Mapping[str, str]
    base_checkpoint_sha256: str
    drop_statistics: Mapping[str, int]
    sync_statistics: Mapping[str, Union[int, float]]


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MORAI DiffusionDriveV2")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resolved-domain", type=Path, required=True)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-epochs", type=_positive_int)
    parser.add_argument("--max-train-batches", type=_positive_int)
    parser.add_argument("--max-validation-samples", type=_positive_int)
    return parser


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"JSON object repeats key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IntegrityError(f"unable to load {label}") from error
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{label} must contain a JSON object")
    return value


def load_training_config(path: Path) -> TrainingConfig:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigurationError("unable to load training YAML") from error
    if not isinstance(value, Mapping) or tuple(value) != _TRAINING_FIELDS:
        raise ConfigurationError("training YAML schema is invalid")
    try:
        return TrainingConfig(**dict(value))
    except (TypeError, ValueError) as error:
        raise ConfigurationError("training YAML values are invalid") from error


def _git_output(repository: Path, arguments: Sequence[str], label: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise IntegrityError(f"unable to verify {label}")
    return completed.stdout.strip()


def _require_commit_object(repository: Path, commit: str, label: str) -> str:
    if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
        raise IntegrityError(f"{label} is malformed")
    _git_output(repository, ("cat-file", "-e", f"{commit}^{{commit}}"), label)
    return commit


def assert_runtime_contract(*, require_cuda: bool) -> None:
    if sys.version_info[:3] != (3, 9, 25):
        raise IntegrityError("training requires CPython 3.9.25")
    if torch.__version__ != "2.0.1+cu118" or torch.version.cuda != "11.8":
        raise IntegrityError("training requires Torch 2.0.1+cu118 with CUDA 11.8")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise IntegrityError(f"{name} must be exactly 1")
    forbidden = ("/usr/lib/python3/dist-packages", "/opt/ros/", "/catkin_ws/")
    if any(
        fragment in Path(entry).resolve().as_posix()
        for entry in sys.path
        if entry
        for fragment in forbidden
    ):
        raise IntegrityError("training Python path mixes ROS/system packages")
    if require_cuda and (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 1
        or torch.cuda.get_device_properties(0).total_memory <= 0
    ):
        raise IntegrityError("CUDA is unavailable for training")


def _validator_root(profile_path: Path) -> Path:
    resolved = Path(profile_path).resolve(strict=True)
    for parent in resolved.parents:
        expected = parent / "src/Validator_pkg/config/profile.json"
        if expected.resolve(strict=False) == resolved:
            return parent
    raise IntegrityError("base profile is outside a Validator checkout")


def _profile_assets(profile_path: Path) -> Tuple[SelAssets, str, str]:
    profile_path = Path(profile_path)
    payload = _load_json(profile_path, "base Validator profile")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise IntegrityError("base profile lacks profiles")
    matches = [
        item
        for item in profiles
        if isinstance(item, Mapping) and item.get("id") == "diffusiondrive_v2"
    ]
    if len(matches) != 1:
        raise IntegrityError("base profile must contain one diffusiondrive_v2 profile")
    profile = matches[0]
    source = profile.get("source")
    if not isinstance(source, Mapping) or source.get("commit") != _DDV2_COMMIT:
        raise IntegrityError("base profile DDV2 source commit is invalid")
    declarations = profile.get("runtime_assets")
    if not isinstance(declarations, list):
        raise IntegrityError("base profile runtime assets are invalid")
    by_role = {
        item.get("role"): item for item in declarations if isinstance(item, Mapping)
    }
    if set(by_role) != set(_ASSET_HASHES):
        raise IntegrityError("base profile runtime asset roles are invalid")
    validator_root = _validator_root(profile_path)
    validator_commit = _git_output(
        validator_root, ("rev-parse", "HEAD"), "Validator checkout commit"
    )
    if validator_commit != _VALIDATOR_COMMIT:
        raise IntegrityError("Validator checkout does not match the pinned commit")
    paths: Dict[str, Path] = {}
    for role, expected_sha256 in _ASSET_HASHES.items():
        declaration = by_role[role]
        if declaration.get("sha256") != expected_sha256:
            raise IntegrityError(f"base profile {role} hash is invalid")
        relative = declaration.get("path")
        if type(relative) is not str:
            raise IntegrityError(f"base profile {role} path is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or any(
            part in ("", ".", "..") for part in relative_path.parts
        ):
            raise IntegrityError(f"base profile {role} path is unsafe")
        candidate = (validator_root / relative_path).resolve(strict=False)
        try:
            candidate.relative_to(validator_root.resolve(strict=True))
        except ValueError as error:
            raise IntegrityError(
                f"base profile {role} path escapes Validator"
            ) from error
        if not candidate.is_file() or sha256_file(candidate) != expected_sha256:
            raise IntegrityError(f"base profile {role} asset is missing or corrupt")
        paths[role] = candidate
    official_checkout = _PROJECT_ROOT / "third_party/DiffusionDriveV2"
    observed_commit = _git_output(
        official_checkout, ("rev-parse", "HEAD"), "DDV2 checkout commit"
    )
    if observed_commit != _DDV2_COMMIT:
        raise IntegrityError("DDV2 checkout does not match the pinned commit")
    v2_checkout = validator_root / "models/sources/diffusiondrive_v2"
    fetched_commit = _git_output(
        v2_checkout, ("rev-parse", "HEAD"), "fetched DDV2 source commit"
    )
    if fetched_commit != _DDV2_COMMIT:
        raise IntegrityError("fetched DDV2 source does not match the pinned commit")
    asset_manifest = _load_json(
        validator_root / "models/diffusiondrive_v2/asset-manifest.json",
        "DDV2 asset manifest",
    )
    manifest_source = asset_manifest.get("source")
    derivative_metadata = (
        manifest_source.get("local_derivative")
        if isinstance(manifest_source, Mapping)
        else None
    )
    if (
        asset_manifest.get("model_id") != "diffusiondrive_v2"
        or not isinstance(manifest_source, Mapping)
        or manifest_source.get("commit") != _DDV2_COMMIT
        or manifest_source.get("observed_commit") != _DDV2_COMMIT
        or not isinstance(derivative_metadata, Mapping)
        or derivative_metadata.get("path") != _DERIVATIVE_RELATIVE_PATH.as_posix()
    ):
        raise IntegrityError("DDV2 fetched source manifest is invalid")
    derivative = v2_checkout / _DERIVATIVE_RELATIVE_PATH
    derivative_sha256 = derivative_metadata.get("sha256")
    if (
        type(derivative_sha256) is not str
        or _SHA256.fullmatch(derivative_sha256) is None
        or not derivative.is_file()
        or sha256_file(derivative) != derivative_sha256
    ):
        raise IntegrityError("PDM-free DDV2 derivative is missing or corrupt")
    return (
        SelAssets(
            v2_checkout=v2_checkout,
            validator_root=validator_root,
            checkpoint=paths["checkpoint"],
            image_backbone=paths["image_backbone"],
            trajectory_anchor=paths["trajectory_anchor"],
        ),
        _ASSET_HASHES["checkpoint"],
        sha256_file(profile_path),
    )


def build_resolved_run_config(
    *,
    config: TrainingConfig,
    identities: Mapping[str, str],
    resume_checkpoint_sha256: Optional[str],
    max_epochs: Optional[int],
    max_train_batches: Optional[int],
    max_validation_samples: Optional[int],
) -> Mapping[str, Any]:
    if type(config) is not TrainingConfig:
        raise TypeError("config must be TrainingConfig")
    if resume_checkpoint_sha256 is not None and (
        type(resume_checkpoint_sha256) is not str
        or _SHA256.fullmatch(resume_checkpoint_sha256) is None
    ):
        raise ValueError("resume checkpoint hash must be a lowercase SHA-256")
    resolved = {
        "schema_version": 1,
        "training": asdict(config),
        "identities": dict(identities),
        "smoke_bounds": {
            "max_epochs": max_epochs,
            "max_train_batches": max_train_batches,
            "max_validation_samples": max_validation_samples,
        },
    }
    if resume_checkpoint_sha256 is not None:
        resolved["resume_checkpoint_sha256"] = resume_checkpoint_sha256
    return resolved


def _manifest_statistics(
    manifest: VerifiedManifest,
) -> tuple[Mapping[str, int], Mapping[str, Union[int, float]]]:
    drops: Dict[str, int] = {}
    for bag in manifest.bag_records:
        for name, value in bag["drop_counts"].items():
            drops[name] = drops.get(name, 0) + int(value)
    sync: Dict[str, Union[int, float]] = {
        "bag_count": len(manifest.bag_records),
        "sample_count": len(manifest.sample_records),
        "tf_ego_raw_record_count": sum(
            int(bag["tf_ego_raw_record_count"]) for bag in manifest.bag_records
        ),
        "tf_ego_epoch_local_comparable_count": sum(
            int(bag["tf_ego_epoch_local_comparable_count"])
            for bag in manifest.bag_records
        ),
        "tf_ego_no_epoch_local_bracket_count": sum(
            int(bag["tf_ego_no_epoch_local_bracket_count"])
            for bag in manifest.bag_records
        ),
        "tf_ego_ambiguous_epoch_selection_count": sum(
            int(bag["tf_ego_ambiguous_epoch_selection_count"])
            for bag in manifest.bag_records
        ),
        "tf_ego_cross_epoch_bracket_prevented_count": sum(
            int(bag["tf_ego_cross_epoch_bracket_prevented_count"])
            for bag in manifest.bag_records
        ),
    }
    return drops, sync


def _validate_production_domain(resolved: ResolvedDomain) -> None:
    if type(resolved) is not ResolvedDomain:
        raise TypeError("resolved must be a ResolvedDomain")
    if (
        resolved.sha256 != _RESOLVED_DOMAIN_SHA256
        or resolved.document.get("domain_id") != DOMAIN_ID
        or resolved.document.get("source_code_commit") != _PREPROCESSING_SOURCE_COMMIT
    ):
        raise IntegrityError("resolved domain is not the corrected Option D identity")


def _validate_production_manifest(
    resolved: ResolvedDomain, manifest: VerifiedManifest
) -> None:
    if type(manifest) is not VerifiedManifest:
        raise TypeError("manifest must be a VerifiedManifest")
    resolved_commit = resolved.document.get("source_code_commit")
    if manifest.dataset_record.get("source_code_commit") != resolved_commit:
        raise IntegrityError(
            "manifest source commit does not match resolved domain source commit"
        )
    eligible = tuple(
        bag for bag in manifest.bag_records if bag.get("disposition") == "ELIGIBLE"
    )
    if len(eligible) != 30:
        raise IntegrityError("MORAI production manifest must contain 30 eligible bags")
    expected_splits = assign_bag_splits(
        [str(bag["relative_bag_path"]) for bag in eligible]
    )
    actual_splits = {
        str(bag["relative_bag_path"]): bag.get("split") for bag in eligible
    }
    train_paths = [path for path, split in expected_splits.items() if split == "train"]
    validation_paths = [
        path for path, split in expected_splits.items() if split == "validation"
    ]
    if (
        actual_splits != dict(expected_splits)
        or len(train_paths) != 24
        or len(validation_paths) != 6
        or list(manifest.dataset_record.get("train_relative_bag_paths", ()))
        != train_paths
        or list(manifest.dataset_record.get("validation_relative_bag_paths", ()))
        != validation_paths
    ):
        raise IntegrityError("MORAI production manifest must use the exact 24/6 split")


def preflight_training(arguments: argparse.Namespace) -> TrainingPreflight:
    assert_runtime_contract(require_cuda=True)
    config = load_training_config(arguments.config)
    resolved = load_resolved_domain(arguments.resolved_domain)
    _validate_production_domain(resolved)
    document = resolved.document
    domain_id = document.get("domain_id")
    source = document.get("sensor_preset")
    cache = document.get("cache")
    if (
        type(domain_id) is not str
        or not isinstance(source, Mapping)
        or not isinstance(cache, Mapping)
    ):
        raise IntegrityError("resolved domain training identity is invalid")
    attributes = cache.get("hdf5_attributes")
    if (
        cache.get("schema_version") != 2
        or not isinstance(attributes, Mapping)
        or attributes.get("cache_schema_version") != 2
        or attributes.get("clock_policy_version") != 1
    ):
        raise IntegrityError("resolved domain cache contract is invalid")
    preset_relative = source.get("path")
    if type(preset_relative) is not str:
        raise IntegrityError("resolved domain sensor preset path is invalid")
    preset_path = _PROJECT_ROOT / preset_relative
    manifest = load_verified_manifest(
        Path(arguments.manifest),
        preset_path,
        domain_id,
        resolved.sha256,
        production=True,
    )
    _validate_production_manifest(resolved, manifest)
    preprocessing_commit = _require_commit_object(
        _PROJECT_ROOT,
        document["source_code_commit"],
        "preprocessing source commit",
    )
    training_commit = _require_commit_object(
        _PROJECT_ROOT,
        _git_output(_PROJECT_ROOT, ("rev-parse", "HEAD"), "training source commit"),
        "training source commit",
    )
    assets, base_sha256, profile_sha256 = _profile_assets(arguments.base_profile)
    identities = {
        "domain_id": domain_id,
        "training_manifest_sha256": manifest.manifest_sha256,
        "resolved_domain_sha256": resolved.sha256,
        "sensor_preset_sha256": str(source["sha256"]),
        "base_profile_sha256": profile_sha256,
        "base_checkpoint_sha256": base_sha256,
        "ddv2_source_commit": _DDV2_COMMIT,
        "validator_source_commit": _VALIDATOR_COMMIT,
        "preprocessing_source_commit": preprocessing_commit,
        "training_source_commit": training_commit,
    }
    resume_sha256 = None
    if arguments.resume is not None:
        if not arguments.resume.is_file():
            raise IntegrityError("resume checkpoint is not a file")
        resume_sha256 = sha256_file(arguments.resume)
    resolved_run_config = build_resolved_run_config(
        config=config,
        identities=identities,
        resume_checkpoint_sha256=resume_sha256,
        max_epochs=arguments.max_epochs,
        max_train_batches=arguments.max_train_batches,
        max_validation_samples=arguments.max_validation_samples,
    )
    if arguments.resume is not None:
        inspect_training_checkpoint(
            arguments.resume,
            expected_checkpoint_sha256=resume_sha256,
            expected_base_checkpoint_sha256=base_sha256,
            expected_config=resolved_run_config,
            expected_identities=identities,
        )
    drops, sync = _manifest_statistics(manifest)
    return TrainingPreflight(
        config=config,
        manifest=manifest,
        assets=assets,
        resolved_run_config=resolved_run_config,
        identities=identities,
        base_checkpoint_sha256=base_sha256,
        drop_statistics=drops,
        sync_statistics=sync,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    preflight = preflight_training(arguments)
    random.seed(preflight.config.seed)
    np.random.seed(preflight.config.seed % (1 << 32))
    torch.manual_seed(preflight.config.seed)
    torch.cuda.manual_seed_all(preflight.config.seed)
    model, raw_checkpoint = load_pdm_free_sel_network(
        preflight.assets, torch.device("cuda", 0)
    )
    if raw_checkpoint.checkpoint_sha256 != preflight.base_checkpoint_sha256:
        raise IntegrityError("loaded base checkpoint identity changed after preflight")
    train_loader, validation_loader, sampler, generator = build_dataloaders(
        preflight.manifest, preflight.config
    )
    fit(
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        sampler=sampler,
        dataloader_generator=generator,
        config=preflight.config,
        run_dir=arguments.run_dir,
        resolved_run_config=preflight.resolved_run_config,
        identities=preflight.identities,
        base_checkpoint_sha256=preflight.base_checkpoint_sha256,
        resume_path=arguments.resume,
        max_epochs=arguments.max_epochs,
        max_train_batches=arguments.max_train_batches,
        max_validation_samples=arguments.max_validation_samples,
        cache_schema_version=2,
        clock_policy_version=1,
        drop_statistics=preflight.drop_statistics,
        sync_statistics=preflight.sync_statistics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
