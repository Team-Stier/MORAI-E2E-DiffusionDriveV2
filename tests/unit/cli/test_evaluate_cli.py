from __future__ import annotations
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import morai_ddv2.cli.evaluate as evaluate_cli
from morai_ddv2.cli.train import build_resolved_run_config
from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.checkpoint import save_training_checkpoint
from morai_ddv2.training.freeze import extract_mutable_state
from morai_ddv2.training.metrics import PredictionRecord, compute_openloop_metrics
from morai_ddv2.training.rng import capture_rng_state
from morai_ddv2.training.sampler import ResumeContractError
from morai_ddv2.upstream import SelAssets
from tests.unit.cli.test_train_cli import _production_manifest
from tests.unit.training.test_resume import _new_training_objects, _zero_accumulator


_REQUIRED = [
    "offline",
    "--manifest",
    "manifest.jsonl",
    "--resolved-domain",
    "resolved-domain.json",
    "--base-profile",
    "profile.json",
    "--training-checkpoint",
    "best.pt",
    "--report",
    "report.json",
]


def _report(identities):
    zeros = np.zeros((8, 3), dtype=np.float64)
    return compute_openloop_metrics(
        (
            PredictionRecord(
                sample_id="A",
                bag_sha256="f" * 64,
                command_name="LEFT",
                timestamp_ns=1,
                seed=1,
                prediction=zeros,
                target=zeros,
                candidate_source="coarse_top32_post_bezier",
                candidate_count=32,
                min_ade_32=0.0,
                min_fde_32=0.0,
                argmin_ade_rank=0,
                argmin_fde_rank=0,
            ),
        ),
        identities=identities,
    )


def test_evaluate_parser_rejects_unknown_sample_order() -> None:
    with pytest.raises(SystemExit):
        evaluate_cli._parser().parse_args(_REQUIRED + ["--sample-order", "random"])


def test_reverse_changes_traversal_only_and_report_bytes_remain_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = (
        {"metadata": {"sample_id": "B"}},
        {"metadata": {"sample_id": "A"}},
    )
    traversals = []

    def validate(_model, ordered, **kwargs):
        traversals.append(tuple(item["metadata"]["sample_id"] for item in ordered))
        return _report(kwargs["identities"])

    monkeypatch.setattr(evaluate_cli, "validate_selector_b1", validate)
    manifest_report = tmp_path / "manifest.json"
    reverse_report = tmp_path / "reverse.json"

    evaluate_cli.evaluate_offline(
        model=object(),
        samples=samples,
        report_path=manifest_report,
        identities={"manifest": "1" * 64},
        run_id="validation-v1",
        sample_order="manifest",
    )
    evaluate_cli.evaluate_offline(
        model=object(),
        samples=samples,
        report_path=reverse_report,
        identities={"manifest": "1" * 64},
        run_id="validation-v1",
        sample_order="reverse",
    )

    assert traversals == [("B", "A"), ("A", "B")]
    assert manifest_report.read_bytes() == reverse_report.read_bytes()
    assert b'"sample_id":"A"' in manifest_report.read_bytes()


def test_evaluation_rejects_another_cache_checkpoint_before_model_construction(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_path, resolved, manifest = _production_manifest(
        repo_root, tmp_path / "cache"
    )
    base_sha256 = "3" * 64
    profile_sha256 = "4" * 64
    current_identities = {
        "domain_id": resolved.document["domain_id"],
        "training_manifest_sha256": manifest.manifest_sha256,
        "resolved_domain_sha256": resolved.sha256,
        "sensor_preset_sha256": resolved.document["sensor_preset"]["sha256"],
        "base_profile_sha256": profile_sha256,
        "base_checkpoint_sha256": base_sha256,
        "ddv2_source_commit": evaluate_cli._DDV2_COMMIT,
        "validator_source_commit": evaluate_cli._VALIDATOR_COMMIT,
        "preprocessing_source_commit": resolved.document["source_code_commit"],
        "training_source_commit": evaluate_cli._git_output(
            repo_root, ("rev-parse", "HEAD"), "test source commit"
        ),
    }
    stored_identities = dict(current_identities)
    stored_identities["training_manifest_sha256"] = "9" * 64
    (
        checkpoint_model,
        freeze_manifest,
        _reference,
        optimizer,
        scaler,
        sampler,
        generator,
        _loader,
    ) = _new_training_objects(num_workers=0)
    checkpoint = tmp_path / "other-cache.pt"
    checkpoint_config = build_resolved_run_config(
        config=evaluate_cli.TrainingConfig(),
        identities=stored_identities,
        resume_checkpoint_sha256=None,
        max_epochs=None,
        max_train_batches=None,
        max_validation_samples=None,
    )
    save_training_checkpoint(
        checkpoint,
        mutable_state=extract_mutable_state(checkpoint_model, freeze_manifest),
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        global_step=0,
        best_micro_ade=math.inf,
        rng=capture_rng_state(
            generator,
            dataloader_iterator_base_cpu=generator.get_state().clone(),
        ),
        sampler_state=sampler.state_dict(),
        accumulator=_zero_accumulator(),
        completed_epoch_record_sha256=None,
        base_checkpoint_sha256=base_sha256,
        manifest=freeze_manifest,
        config=checkpoint_config,
    )
    assets = SelAssets(
        v2_checkout=tmp_path / "source",
        validator_root=tmp_path / "validator",
        checkpoint=tmp_path / "base.ckpt",
        image_backbone=tmp_path / "backbone.bin",
        trajectory_anchor=tmp_path / "anchor.npy",
    )
    monkeypatch.setattr(evaluate_cli, "assert_runtime_contract", lambda **_: None)
    monkeypatch.setattr(
        evaluate_cli,
        "load_verified_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        evaluate_cli,
        "_profile_assets",
        lambda _path: (assets, base_sha256, profile_sha256),
    )
    constructed = []

    def construct(*args, **kwargs):
        constructed.append((args, kwargs))
        raise AssertionError("model construction must not run")

    monkeypatch.setattr(evaluate_cli, "load_pdm_free_sel_network", construct)
    argv = [
        "offline",
        "--manifest",
        str(manifest.path),
        "--resolved-domain",
        str(resolved_path),
        "--base-profile",
        str(tmp_path / "profile.json"),
        "--training-checkpoint",
        str(checkpoint),
        "--report",
        str(tmp_path / "report.json"),
    ]

    with pytest.raises(ResumeContractError, match="identity"):
        evaluate_cli.main(argv)

    assert constructed == []


def test_evaluation_preserves_nondefault_training_config_and_source_provenance(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_path, resolved, manifest = _production_manifest(
        repo_root, tmp_path / "cache"
    )
    base_sha256 = "3" * 64
    profile_sha256 = "4" * 64
    training_source_commit = resolved.document["source_code_commit"]
    checkpoint_identities = {
        "domain_id": resolved.document["domain_id"],
        "training_manifest_sha256": manifest.manifest_sha256,
        "resolved_domain_sha256": resolved.sha256,
        "sensor_preset_sha256": resolved.document["sensor_preset"]["sha256"],
        "base_profile_sha256": profile_sha256,
        "base_checkpoint_sha256": base_sha256,
        "ddv2_source_commit": evaluate_cli._DDV2_COMMIT,
        "validator_source_commit": evaluate_cli._VALIDATOR_COMMIT,
        "preprocessing_source_commit": resolved.document["source_code_commit"],
        "training_source_commit": training_source_commit,
    }
    training_config = replace(
        evaluate_cli.TrainingConfig(),
        epochs=2,
        batch_size=1,
        num_workers=0,
        amp=False,
    )
    (
        checkpoint_model,
        freeze_manifest,
        _reference,
        optimizer,
        scaler,
        sampler,
        generator,
        _loader,
    ) = _new_training_objects(num_workers=0)
    checkpoint = tmp_path / "nondefault.pt"
    checkpoint_config = build_resolved_run_config(
        config=training_config,
        identities=checkpoint_identities,
        resume_checkpoint_sha256="a" * 64,
        max_epochs=None,
        max_train_batches=None,
        max_validation_samples=None,
    )
    save_training_checkpoint(
        checkpoint,
        mutable_state=extract_mutable_state(checkpoint_model, freeze_manifest),
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        global_step=0,
        best_micro_ade=math.inf,
        rng=capture_rng_state(
            generator,
            dataloader_iterator_base_cpu=generator.get_state().clone(),
        ),
        sampler_state=sampler.state_dict(),
        accumulator=_zero_accumulator(),
        completed_epoch_record_sha256=None,
        base_checkpoint_sha256=base_sha256,
        manifest=freeze_manifest,
        config=checkpoint_config,
    )
    assets = SelAssets(
        v2_checkout=tmp_path / "source",
        validator_root=tmp_path / "validator",
        checkpoint=tmp_path / "base.ckpt",
        image_backbone=tmp_path / "backbone.bin",
        trajectory_anchor=tmp_path / "anchor.npy",
    )
    monkeypatch.setattr(evaluate_cli, "assert_runtime_contract", lambda **_: None)
    monkeypatch.setattr(
        evaluate_cli, "load_verified_manifest", lambda *args, **kwargs: manifest
    )
    monkeypatch.setattr(
        evaluate_cli,
        "_profile_assets",
        lambda _path: (assets, base_sha256, profile_sha256),
    )
    argv = [
        "offline",
        "--manifest",
        str(manifest.path),
        "--resolved-domain",
        str(resolved_path),
        "--base-profile",
        str(tmp_path / "profile.json"),
        "--training-checkpoint",
        str(checkpoint),
        "--report",
        str(tmp_path / "report.json"),
    ]
    arguments = evaluate_cli._parser().parse_args(argv)

    preflight = evaluate_cli.preflight_evaluation(arguments)

    assert preflight.config == training_config
    assert preflight.checkpoint_config == checkpoint_config
    assert preflight.identities["training_source_commit"] == training_source_commit
    assert preflight.identities["evaluation_source_commit"] != training_source_commit

    (
        evaluation_model,
        _evaluation_manifest,
        _evaluation_reference,
        _evaluation_optimizer,
        _evaluation_scaler,
        evaluation_sampler,
        evaluation_generator,
        _evaluation_loader,
    ) = _new_training_objects(num_workers=0)
    monkeypatch.setattr(
        evaluate_cli,
        "load_pdm_free_sel_network",
        lambda *args, **kwargs: (
            evaluation_model,
            SimpleNamespace(checkpoint_sha256=base_sha256),
        ),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "build_dataloaders",
        lambda *args, **kwargs: (
            object(),
            object(),
            evaluation_sampler,
            evaluation_generator,
        ),
    )
    monkeypatch.setattr(
        evaluate_cli, "MoraiTrainingDataset", lambda *args, **kwargs: ()
    )
    evaluated = []
    monkeypatch.setattr(
        evaluate_cli,
        "evaluate_offline",
        lambda **kwargs: evaluated.append(kwargs),
    )

    assert evaluate_cli.main(argv) == 0
    assert len(evaluated) == 1


@pytest.mark.parametrize(
    ("corruption", "expected_error", "message"),
    [
        ("config", ResumeContractError, "training config"),
        ("source", IntegrityError, "training checkpoint source commit"),
    ],
)
def test_evaluation_rejects_malformed_stored_config_or_training_source_before_model(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption,
    expected_error,
    message,
) -> None:
    resolved_path, resolved, manifest = _production_manifest(
        repo_root, tmp_path / "cache"
    )
    base_sha256 = "3" * 64
    profile_sha256 = "4" * 64
    checkpoint_identities = {
        "domain_id": resolved.document["domain_id"],
        "training_manifest_sha256": manifest.manifest_sha256,
        "resolved_domain_sha256": resolved.sha256,
        "sensor_preset_sha256": resolved.document["sensor_preset"]["sha256"],
        "base_profile_sha256": profile_sha256,
        "base_checkpoint_sha256": base_sha256,
        "ddv2_source_commit": evaluate_cli._DDV2_COMMIT,
        "validator_source_commit": evaluate_cli._VALIDATOR_COMMIT,
        "preprocessing_source_commit": resolved.document["source_code_commit"],
        "training_source_commit": resolved.document["source_code_commit"],
    }
    checkpoint_config = build_resolved_run_config(
        config=evaluate_cli.TrainingConfig(),
        identities=checkpoint_identities,
        resume_checkpoint_sha256=None,
        max_epochs=None,
        max_train_batches=None,
        max_validation_samples=None,
    )
    if corruption == "config":
        checkpoint_config["training"].pop("epochs")
    else:
        checkpoint_config["identities"]["training_source_commit"] = "f" * 40
    checkpoint = tmp_path / "malformed.pt"
    checkpoint.write_bytes(b"inspection is isolated in this test")
    assets = SelAssets(
        v2_checkout=tmp_path / "source",
        validator_root=tmp_path / "validator",
        checkpoint=tmp_path / "base.ckpt",
        image_backbone=tmp_path / "backbone.bin",
        trajectory_anchor=tmp_path / "anchor.npy",
    )
    monkeypatch.setattr(evaluate_cli, "assert_runtime_contract", lambda **_: None)
    monkeypatch.setattr(
        evaluate_cli, "load_verified_manifest", lambda *args, **kwargs: manifest
    )
    monkeypatch.setattr(
        evaluate_cli,
        "_profile_assets",
        lambda _path: (assets, base_sha256, profile_sha256),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "inspect_training_checkpoint",
        lambda *args, **kwargs: checkpoint_config,
    )
    constructed = []
    monkeypatch.setattr(
        evaluate_cli,
        "load_pdm_free_sel_network",
        lambda *args, **kwargs: constructed.append((args, kwargs)),
    )
    argv = [
        "offline",
        "--manifest",
        str(manifest.path),
        "--resolved-domain",
        str(resolved_path),
        "--base-profile",
        str(tmp_path / "profile.json"),
        "--training-checkpoint",
        str(checkpoint),
        "--report",
        str(tmp_path / "report.json"),
    ]

    with pytest.raises(expected_error, match=message):
        evaluate_cli.main(argv)

    assert constructed == []


def test_script_evaluate_wrapper_exists(repo_root: Path) -> None:
    wrapper = (repo_root / "scripts/evaluate.py").read_text(encoding="utf-8")
    assert "from morai_ddv2.cli.evaluate import main" in wrapper
