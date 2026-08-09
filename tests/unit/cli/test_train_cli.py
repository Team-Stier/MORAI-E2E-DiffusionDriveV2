from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import morai_ddv2.cli.train as train_cli
from morai_ddv2.cache.loader import load_verified_manifest
from morai_ddv2.cache.manifest import assign_bag_splits
from morai_ddv2.config import (
    load_resolved_domain,
    load_sensor_preset,
    publish_resolved_domain,
    resolve_domain,
)
from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.dataset import TrainingConfig
from morai_ddv2.training.sampler import ResumeContractError
from morai_ddv2.upstream import SelAssets
from tests.unit.cache.test_loader import PRESET_PATH
from tests.unit.cache.test_manifest import _publish_fixture


_REQUIRED = [
    "--manifest",
    "manifest.jsonl",
    "--resolved-domain",
    "resolved-domain.json",
    "--base-profile",
    "profile.json",
    "--config",
    "training.yaml",
    "--run-dir",
    "run",
]


def _production_resolved_domain(repo_root: Path, tmp_path: Path):
    raw_domain = yaml.safe_load(
        (repo_root / "configs/domains/morai-ddv2-v1-oracle-command.yaml").read_text(
            encoding="utf-8"
        )
    )
    preset_sha256 = raw_domain["sensor_preset"]["sha256"]
    preset = load_sensor_preset(
        repo_root / "sensor_data/0725demo_ros.json", preset_sha256
    )
    artifact = resolve_domain(
        preset=preset,
        raw_domain=raw_domain,
        source_commit=train_cli._PREPROCESSING_SOURCE_COMMIT,
    )
    assert artifact.sha256 == train_cli._RESOLVED_DOMAIN_SHA256
    path = tmp_path / "resolved-domain.json"
    publish_resolved_domain(artifact, path)
    return path, load_resolved_domain(path)


def _production_manifest(repo_root: Path, tmp_path: Path):
    resolved_path, resolved = _production_resolved_domain(repo_root, tmp_path)
    publication = _publish_fixture(tmp_path)
    manifest_path = tmp_path / publication.path
    verified = load_verified_manifest(
        manifest_path,
        PRESET_PATH,
        train_cli.DOMAIN_ID,
        manifest_path.parent.name,
        production=True,
    )
    eligible_template = next(
        bag for bag in verified.bag_records if bag["disposition"] == "ELIGIBLE"
    )
    paths = ["production/{:02d}.bag".format(index) for index in range(30)]
    splits = assign_bag_splits(paths)
    bags = tuple(
        {
            **dict(eligible_template),
            "relative_bag_path": path,
            "source_bag_sha256": "{:064x}".format(index + 1),
            "split": splits[path],
        }
        for index, path in enumerate(sorted(paths))
    )
    dataset = {
        **dict(verified.dataset_record),
        "source_code_commit": resolved.document["source_code_commit"],
        "train_relative_bag_paths": [
            path for path, split in splits.items() if split == "train"
        ],
        "validation_relative_bag_paths": [
            path for path, split in splits.items() if split == "validation"
        ],
    }
    return (
        resolved_path,
        resolved,
        replace(
            verified,
            dataset_record=dataset,
            bag_records=bags,
        ),
    )


@pytest.mark.parametrize(
    "flag", ["--max-epochs", "--max-train-batches", "--max-validation-samples"]
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_train_parser_rejects_nonpositive_smoke_bounds(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        train_cli._parser().parse_args(_REQUIRED + [flag, value])


def test_resolved_run_config_records_bounds_without_mutating_training_config() -> None:
    config = TrainingConfig()
    identities = {
        "training_manifest_sha256": "1" * 64,
        "resolved_domain_sha256": "2" * 64,
        "base_checkpoint_sha256": "3" * 64,
        "training_source_commit": "4" * 40,
        "preprocessing_source_commit": "5" * 40,
    }

    unbounded = train_cli.build_resolved_run_config(
        config=config,
        identities=identities,
        resume_checkpoint_sha256=None,
        max_epochs=None,
        max_train_batches=None,
        max_validation_samples=None,
    )
    bounded = train_cli.build_resolved_run_config(
        config=config,
        identities=identities,
        resume_checkpoint_sha256="6" * 64,
        max_epochs=2,
        max_train_batches=3,
        max_validation_samples=4,
    )

    assert unbounded["training"] == asdict(TrainingConfig())
    assert unbounded["smoke_bounds"] == {
        "max_epochs": None,
        "max_train_batches": None,
        "max_validation_samples": None,
    }
    assert bounded["training"]["epochs"] == 30
    assert bounded["smoke_bounds"] == {
        "max_epochs": 2,
        "max_train_batches": 3,
        "max_validation_samples": 4,
    }
    assert bounded["resume_checkpoint_sha256"] == "6" * 64
    assert config == TrainingConfig()


def test_profile_assets_use_fetched_derivative_and_verify_validator_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    validator_root = project_root / "third_party/E2E-OpenLoop-ROS-Validator"
    profile = validator_root / "src/Validator_pkg/config/profile.json"
    source_root = validator_root / "models/sources/diffusiondrive_v2"
    derivative_relative = Path(
        "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py"
    )
    derivative = source_root / derivative_relative
    derivative.parent.mkdir(parents=True)
    derivative.write_bytes(b"generated PDM-free derivative")
    official_derivative = (
        project_root / "third_party/DiffusionDriveV2" / derivative_relative
    )
    official_derivative.parent.mkdir(parents=True)
    official_derivative.write_bytes(b"official source placeholder")
    asset_payloads = {
        "checkpoint": b"checkpoint",
        "image_backbone": b"backbone",
        "trajectory_anchor": b"anchor",
    }
    hashes = {
        role: hashlib.sha256(payload).hexdigest()
        for role, payload in asset_payloads.items()
    }
    declarations = []
    for role, payload in asset_payloads.items():
        relative = "models/assets/{}.bin".format(role)
        target = validator_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        declarations.append({"role": role, "path": relative, "sha256": hashes[role]})
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "id": "diffusiondrive_v2",
                        "source": {"commit": train_cli._DDV2_COMMIT},
                        "runtime_assets": declarations,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = validator_root / "models/diffusiondrive_v2/asset-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "model_id": "diffusiondrive_v2",
                "source": {
                    "commit": train_cli._DDV2_COMMIT,
                    "observed_commit": train_cli._DDV2_COMMIT,
                    "local_derivative": {
                        "path": derivative_relative.as_posix(),
                        "sha256": hashlib.sha256(derivative.read_bytes()).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def recording_git_output(repository, arguments, label):
        calls.append((Path(repository), tuple(arguments), label))
        if tuple(arguments) == ("rev-parse", "HEAD"):
            if Path(repository) == validator_root:
                return train_cli._VALIDATOR_COMMIT
            return train_cli._DDV2_COMMIT
        raise AssertionError((repository, arguments, label))

    monkeypatch.setattr(train_cli, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(train_cli, "_ASSET_HASHES", hashes)
    monkeypatch.setattr(train_cli, "_git_output", recording_git_output)

    assets, _, _ = train_cli._profile_assets(profile)

    assert assets.v2_checkout == (validator_root / "models/sources/diffusiondrive_v2")
    assert (
        validator_root,
        ("rev-parse", "HEAD"),
        "Validator checkout commit",
    ) in calls


def test_production_domain_and_inventory_pin_corrected_option_d_30_bag_split(
    repo_root: Path, tmp_path: Path
) -> None:
    _resolved_path, resolved = _production_resolved_domain(repo_root, tmp_path)
    train_cli._validate_production_domain(resolved)
    old_resolved = load_resolved_domain(
        repo_root / "tests/fixtures/golden/resolved-domain.json"
    )
    with pytest.raises(IntegrityError, match="corrected Option D"):
        train_cli._validate_production_domain(old_resolved)

    publication = _publish_fixture(tmp_path)
    manifest_path = tmp_path / publication.path
    verified = load_verified_manifest(
        manifest_path,
        PRESET_PATH,
        train_cli.DOMAIN_ID,
        manifest_path.parent.name,
        production=True,
    )
    verified = replace(
        verified,
        dataset_record={
            **dict(verified.dataset_record),
            "source_code_commit": resolved.document["source_code_commit"],
        },
    )
    with pytest.raises(IntegrityError, match="30 eligible bags"):
        train_cli._validate_production_manifest(resolved, verified)

    eligible_template = next(
        bag for bag in verified.bag_records if bag["disposition"] == "ELIGIBLE"
    )
    paths = ["production/{:02d}.bag".format(index) for index in range(30)]
    splits = assign_bag_splits(paths)
    bags = tuple(
        {
            **dict(eligible_template),
            "relative_bag_path": path,
            "source_bag_sha256": "{:064x}".format(index + 1),
            "split": splits[path],
        }
        for index, path in enumerate(sorted(paths))
    )
    dataset = {
        **dict(verified.dataset_record),
        "source_code_commit": resolved.document["source_code_commit"],
        "train_relative_bag_paths": [
            path for path, split in splits.items() if split == "train"
        ],
        "validation_relative_bag_paths": [
            path for path, split in splits.items() if split == "validation"
        ],
    }
    production = replace(
        verified,
        dataset_record=dataset,
        bag_records=bags,
    )

    train_cli._validate_production_manifest(resolved, production)
    assert sum(bag["split"] == "train" for bag in production.bag_records) == 24
    assert sum(bag["split"] == "validation" for bag in production.bag_records) == 6


def test_training_main_rejects_invalid_resume_before_model_construction(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_path, _resolved, manifest = _production_manifest(
        repo_root, tmp_path / "cache"
    )
    resume = tmp_path / "invalid-resume.pt"
    resume.write_bytes(b"not a training checkpoint")
    constructed = []
    base_sha256 = "3" * 64
    assets = SelAssets(
        v2_checkout=tmp_path / "source",
        validator_root=tmp_path / "validator",
        checkpoint=tmp_path / "base.ckpt",
        image_backbone=tmp_path / "backbone.bin",
        trajectory_anchor=tmp_path / "anchor.npy",
    )
    monkeypatch.setattr(train_cli, "assert_runtime_contract", lambda **_: None)
    monkeypatch.setattr(
        train_cli, "load_verified_manifest", lambda *args, **kwargs: manifest
    )
    monkeypatch.setattr(
        train_cli,
        "_profile_assets",
        lambda _path: (assets, base_sha256, "4" * 64),
    )

    def construct(*args, **kwargs):
        constructed.append((args, kwargs))
        raise AssertionError("model construction must not run")

    monkeypatch.setattr(train_cli, "load_pdm_free_sel_network", construct)
    argv = [
        "--manifest",
        str(manifest.path),
        "--resolved-domain",
        str(resolved_path),
        "--base-profile",
        str(tmp_path / "profile.json"),
        "--config",
        str(repo_root / "configs/training/morai-ddv2-encoder-fusion-v1.yaml"),
        "--run-dir",
        str(tmp_path / "run"),
        "--resume",
        str(resume),
    ]

    with pytest.raises(ResumeContractError, match="checkpoint"):
        train_cli.main(argv)

    assert constructed == []


def test_real_preflight_rejects_manifest_resolved_source_commit_mismatch(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_path, _resolved, manifest = _production_manifest(
        repo_root, tmp_path / "cache"
    )
    mismatched = replace(
        manifest,
        dataset_record={
            **dict(manifest.dataset_record),
            "source_code_commit": "b" * 40,
        },
    )
    arguments = argparse.Namespace(
        manifest=manifest.path,
        resolved_domain=resolved_path,
        base_profile=tmp_path / "must-not-be-read.json",
        config=train_cli._PROJECT_ROOT
        / "configs/training/morai-ddv2-encoder-fusion-v1.yaml",
        run_dir=tmp_path / "run",
        resume=None,
        max_epochs=None,
        max_train_batches=None,
        max_validation_samples=None,
    )
    monkeypatch.setattr(train_cli, "assert_runtime_contract", lambda **_: None)
    monkeypatch.setattr(
        train_cli,
        "load_verified_manifest",
        lambda *args, **kwargs: mismatched,
    )

    with pytest.raises(IntegrityError, match="source commit.*resolved domain"):
        train_cli.preflight_training(arguments)


@pytest.mark.parametrize("name", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_runtime_preflight_requires_each_offline_variable_exactly_one(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv(name, "true")

    with pytest.raises(IntegrityError, match=name):
        train_cli.assert_runtime_contract(require_cuda=False)


def test_runtime_preflight_rejects_wrong_python_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_cli.sys, "version_info", (3, 9, 24))

    with pytest.raises(IntegrityError, match="3.9.25"):
        train_cli.assert_runtime_contract(require_cuda=False)


def test_training_runtime_preflight_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(train_cli.torch.cuda, "is_available", lambda: False)

    with pytest.raises(IntegrityError, match="CUDA"):
        train_cli.assert_runtime_contract(require_cuda=True)


def test_script_train_wrapper_exists(repo_root: Path) -> None:
    wrapper = (repo_root / "scripts/train.py").read_text(encoding="utf-8")
    assert "from morai_ddv2.cli.train import main" in wrapper
