from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from morai_ddv2.atomic import atomic_write_bytes
from morai_ddv2.cache.loader import VerifiedManifest, load_verified_manifest
from morai_ddv2.config import load_resolved_domain, sha256_file
from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.checkpoint import load_training_checkpoint
from morai_ddv2.training.dataset import (
    MoraiTrainingDataset,
    TrainingConfig,
    build_dataloaders,
)
from morai_ddv2.training.freeze import build_optimizer, configure_training_boundary
from morai_ddv2.training.metrics import validation_report_json_bytes
from morai_ddv2.training.validation import validate_selector_b1
from morai_ddv2.upstream import SelAssets, load_pdm_free_sel_network

from .train import (
    _PROJECT_ROOT,
    _git_output,
    _manifest_statistics,
    _profile_assets,
    _require_commit_object,
    assert_runtime_contract,
)


_DEFAULT_TRAINING_CONFIG = (
    _PROJECT_ROOT / "configs/training/morai-ddv2-encoder-fusion-v1.yaml"
)


@dataclass(frozen=True)
class EvaluationPreflight:
    manifest: VerifiedManifest
    assets: SelAssets
    identities: Mapping[str, str]
    base_checkpoint_sha256: str
    training_checkpoint: Path
    drop_statistics: Mapping[str, int]
    sync_statistics: Mapping[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MORAI DiffusionDriveV2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    offline = subparsers.add_parser(
        "offline", help="run deterministic cache evaluation"
    )
    offline.add_argument("--manifest", type=Path, required=True)
    offline.add_argument("--resolved-domain", type=Path, required=True)
    offline.add_argument("--base-profile", type=Path, required=True)
    offline.add_argument("--training-checkpoint", type=Path, required=True)
    offline.add_argument("--report", type=Path, required=True)
    offline.add_argument(
        "--sample-order",
        choices=("manifest", "reverse"),
        default="manifest",
    )
    return parser


def preflight_evaluation(arguments: argparse.Namespace) -> EvaluationPreflight:
    assert_runtime_contract(require_cuda=False)
    resolved = load_resolved_domain(arguments.resolved_domain)
    document = resolved.document
    domain_id = document.get("domain_id")
    source = document.get("sensor_preset")
    cache = document.get("cache")
    if (
        type(domain_id) is not str
        or not isinstance(source, Mapping)
        or not isinstance(cache, Mapping)
    ):
        raise IntegrityError("resolved domain evaluation identity is invalid")
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
    manifest = load_verified_manifest(
        Path(arguments.manifest),
        _PROJECT_ROOT / preset_relative,
        domain_id,
        resolved.sha256,
        production=True,
    )
    preprocessing_commit = _require_commit_object(
        _PROJECT_ROOT,
        manifest.dataset_record["source_code_commit"],
        "preprocessing source commit",
    )
    training_commit = _require_commit_object(
        _PROJECT_ROOT,
        _git_output(_PROJECT_ROOT, ("rev-parse", "HEAD"), "evaluation source commit"),
        "evaluation source commit",
    )
    assets, base_sha256, profile_sha256 = _profile_assets(arguments.base_profile)
    checkpoint = Path(arguments.training_checkpoint)
    if not checkpoint.is_file():
        raise IntegrityError("training checkpoint is not a file")
    identities = {
        "domain_id": domain_id,
        "training_manifest_sha256": manifest.manifest_sha256,
        "resolved_domain_sha256": resolved.sha256,
        "sensor_preset_sha256": str(source["sha256"]),
        "base_profile_sha256": profile_sha256,
        "base_checkpoint_sha256": base_sha256,
        "training_checkpoint_sha256": sha256_file(checkpoint),
        "preprocessing_source_commit": preprocessing_commit,
        "evaluation_source_commit": training_commit,
    }
    drops, sync = _manifest_statistics(manifest)
    return EvaluationPreflight(
        manifest=manifest,
        assets=assets,
        identities=identities,
        base_checkpoint_sha256=base_sha256,
        training_checkpoint=checkpoint,
        drop_statistics=drops,
        sync_statistics=sync,
    )


def evaluate_offline(
    *,
    model,
    samples: Sequence[Mapping[str, Any]],
    report_path: Path,
    identities: Mapping[str, str],
    run_id: str,
    sample_order: str,
    drop_statistics: Optional[Mapping[str, int]] = None,
    sync_statistics: Optional[Mapping[str, Any]] = None,
) -> Path:
    if sample_order not in ("manifest", "reverse"):
        raise ValueError("sample_order must be manifest or reverse")
    ordered = iter(samples) if sample_order == "manifest" else reversed(samples)
    report = validate_selector_b1(
        model,
        ordered,
        run_id=run_id,
        drop_statistics=drop_statistics,
        sync_statistics=sync_statistics,
        identities=identities,
    )
    path = Path(report_path)
    atomic_write_bytes(path, validation_report_json_bytes(report))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    preflight = preflight_evaluation(arguments)
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    model, raw_checkpoint = load_pdm_free_sel_network(preflight.assets, device)
    if raw_checkpoint.checkpoint_sha256 != preflight.base_checkpoint_sha256:
        raise IntegrityError("loaded base checkpoint identity changed after preflight")
    config = TrainingConfig()
    train_loader, _, sampler, _ = build_dataloaders(preflight.manifest, config)
    del train_loader
    manifest = configure_training_boundary(model)
    optimizer = build_optimizer(
        model,
        manifest,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    load_training_checkpoint(
        preflight.training_checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        sampler=sampler,
        manifest=manifest,
        expected_base_checkpoint_sha256=preflight.base_checkpoint_sha256,
    )
    samples = MoraiTrainingDataset(preflight.manifest, split="validation")
    evaluate_offline(
        model=model,
        samples=samples,
        report_path=arguments.report,
        identities=preflight.identities,
        run_id=config.validation_run_id,
        sample_order=arguments.sample_order,
        drop_statistics=preflight.drop_statistics,
        sync_statistics=preflight.sync_statistics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
