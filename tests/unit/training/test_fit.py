from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import morai_ddv2.training.trainer as trainer_module
from morai_ddv2.training.contracts import ProductionRunAttestation
from morai_ddv2.training.dataset import TrainingConfig
from morai_ddv2.training.metrics import PredictionRecord, compute_openloop_metrics
from morai_ddv2.training.sampler import StatefulPermutationSampler
from morai_ddv2.training.sampler import ResumeContractError
from morai_ddv2.training.trainer import fit
from morai_ddv2.training.validation import ValidationLossSummary


class _LossComputer:
    cls_loss_weight = 10.0
    reg_loss_weight = 8.0


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_computer = _LossComputer()


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._backbone = nn.Module()
        self._backbone.image_encoder = nn.Linear(1, 1, bias=False)
        self._trajectory_head = _Head()


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {
            "features": {"x": torch.tensor([float(index + 1)])},
            "trajectory": torch.full((8, 3), float(index + 1)),
            "metadata": {
                "sample_id": f"sample-{index}",
                "timestamp_ns": index + 10,
                "source_bag_sha256": "f" * 64,
                "command_name": "LEFT",
            },
        }


def _loaders(config: TrainingConfig):
    dataset = _TinyDataset()
    sampler = StatefulPermutationSampler(dataset, seed=config.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    train = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        generator=generator,
        num_workers=0,
    )
    validation = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        generator=generator,
        num_workers=0,
    )
    return train, validation, sampler, generator


def _validation_report(identities):
    zeros = np.zeros((8, 3), dtype=np.float64)
    prediction = zeros.copy()
    prediction[:, 0] = 1.0
    return compute_openloop_metrics(
        (
            PredictionRecord(
                sample_id="sample-0",
                bag_sha256="f" * 64,
                command_name="LEFT",
                timestamp_ns=10,
                seed=7,
                prediction=prediction,
                target=zeros,
                candidate_source="coarse_top32_post_bezier",
                candidate_count=32,
                min_ade_32=0.5,
                min_fde_32=0.75,
                argmin_ade_rank=0,
                argmin_fde_rank=1,
            ),
        ),
        identities=identities,
    )


def _assert_nested_equal(left, right) -> None:
    if is_dataclass(left) and not isinstance(left, type):
        assert type(right) is type(left)
        for field in fields(left):
            _assert_nested_equal(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(right) is type(left)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_fit_two_epochs_publishes_metrics_checkpoints_and_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        TrainingConfig(),
        epochs=2,
        batch_size=1,
        num_workers=0,
        amp=False,
    )
    model = _TinyModel()
    train, validation, sampler, generator = _loaders(config)
    identities = {
        "training_manifest_sha256": "1" * 64,
        "resolved_domain_sha256": "2" * 64,
        "preprocessing_source_commit": "3" * 40,
        "training_source_commit": "4" * 40,
    }
    boundary_calls = []
    real_configure = trainer_module.configure_training_boundary

    def configure_boundary(current):
        boundary_calls.append(current)
        return real_configure(current)

    def supervised(current, features, target):
        prediction = current._backbone.image_encoder(features["x"])
        loss = (prediction - target[:, :1, :1].reshape(-1, 1)).square().mean()
        return SimpleNamespace(trajectory_loss=loss)

    selector_batches = []
    loss_batches = []

    def selector(current, samples, **kwargs):
        del current
        batches = tuple(samples)
        selector_batches.append(batches)
        assert all(batch["trajectory"].shape[0] == 1 for batch in batches)
        return _validation_report(kwargs["identities"])

    def supervised_loss(current, samples, **kwargs):
        del current
        batches = tuple(samples)
        loss_batches.append((batches, kwargs))
        assert all(batch["trajectory"].shape[0] == 1 for batch in batches)
        return ValidationLossSummary(2, 2.0, 0.15, 0.2)

    checkpoint_paths = []
    real_save = trainer_module.save_training_checkpoint

    def save_checkpoint(path, **kwargs):
        checkpoint_paths.append(Path(path))
        return real_save(path, **kwargs)

    monkeypatch.setattr(
        trainer_module, "configure_training_boundary", configure_boundary
    )
    monkeypatch.setattr(trainer_module, "forward_trajectory_supervised", supervised)
    monkeypatch.setattr(trainer_module, "validate_selector_b1", selector)
    monkeypatch.setattr(trainer_module, "validate_supervised_loss_b1", supervised_loss)
    monkeypatch.setattr(trainer_module, "save_training_checkpoint", save_checkpoint)

    attestation = fit(
        model=model,
        train_dataloader=train,
        validation_dataloader=validation,
        sampler=sampler,
        dataloader_generator=generator,
        config=config,
        run_dir=tmp_path,
        resolved_run_config={"epochs": 2, "smoke_bounds": None},
        identities=identities,
        base_checkpoint_sha256="5" * 64,
    )

    assert isinstance(attestation, ProductionRunAttestation)
    assert len(boundary_calls) == 3
    assert len(selector_batches) == 2
    assert len(loss_batches) == 2
    assert all(call[1]["trajectory_weight"] == 12.0 for call in loss_batches)
    assert all(call[1]["preservation_weight"] == 1.0e-3 for call in loss_batches)
    assert checkpoint_paths.count(tmp_path / "best.pt") == 1
    assert checkpoint_paths.count(tmp_path / "last.pt") == 2
    assert sorted(path.name for path in (tmp_path / "metrics/epochs").iterdir()) == [
        "epoch-0000.json",
        "epoch-0001.json",
    ]
    second_epoch = json.loads(
        (tmp_path / "metrics/epochs/epoch-0001.json").read_text(encoding="utf-8")
    )
    assert second_epoch["validation"]["selected"] == {
        "count": 1,
        "ade": 1.0,
        "fde": 1.0,
        "heading_mae": 0.0,
    }
    assert second_epoch["validation"]["bag_macro_ade"] == 1.0
    assert second_epoch["validation"]["min_ade_32"] == 0.5
    assert second_epoch["validation"]["min_fde_32"] == 0.75
    assert second_epoch["is_best"] is False
    assert (tmp_path / "validation/epoch-0001.json").is_file()
    assert (tmp_path / "resolved-run-config.json").is_file()
    assert (tmp_path / "metrics/history.jsonl").is_file()
    assert (tmp_path / "metrics/epoch_metrics.csv").is_file()
    assert (tmp_path / "plots/training-curves.png").is_file()
    assert (tmp_path / "run-complete.json").is_file()
    assert attestation.completed_epochs == 2
    assert attestation.epoch_telemetry_count == 2
    assert attestation.best_micro_ade == 1.0
    assert sampler.state_dict().epoch == 2
    assert sampler.state_dict().committed_cursor == 0


def test_bounded_fit_publishes_only_debug_allowlist_and_stops_at_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        TrainingConfig(),
        epochs=2,
        batch_size=1,
        num_workers=0,
        amp=False,
    )
    model = _TinyModel()
    train, validation, sampler, generator = _loaders(config)
    identities = {
        "training_manifest_sha256": "1" * 64,
        "resolved_domain_sha256": "2" * 64,
    }

    def supervised(current, features, target):
        prediction = current._backbone.image_encoder(features["x"])
        loss = (prediction - target[:, :1, :1].reshape(-1, 1)).square().mean()
        return SimpleNamespace(trajectory_loss=loss)

    sample_counts = []

    def selector(current, samples, **kwargs):
        del current
        batches = tuple(samples)
        sample_counts.append(len(batches))
        return _validation_report(kwargs["identities"])

    def supervised_loss(current, samples, **kwargs):
        del current, kwargs
        batches = tuple(samples)
        sample_counts.append(len(batches))
        return ValidationLossSummary(len(batches), 2.0, 0.15, 0.2)

    monkeypatch.setattr(trainer_module, "forward_trajectory_supervised", supervised)
    monkeypatch.setattr(trainer_module, "validate_selector_b1", selector)
    monkeypatch.setattr(trainer_module, "validate_supervised_loss_b1", supervised_loss)
    resolved = {
        "epochs": 2,
        "smoke_bounds": {
            "max_epochs": 1,
            "max_train_batches": 1,
            "max_validation_samples": 1,
        },
    }

    result = fit(
        model=model,
        train_dataloader=train,
        validation_dataloader=validation,
        sampler=sampler,
        dataloader_generator=generator,
        config=config,
        run_dir=tmp_path,
        resolved_run_config=resolved,
        identities=identities,
        base_checkpoint_sha256="5" * 64,
        max_epochs=1,
        max_train_batches=1,
        max_validation_samples=1,
    )

    assert result is None
    assert sample_counts == [1, 1]
    files = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert files == [
        "debug/best.pt",
        "debug/last.pt",
        "debug/resolved-run-config.json",
        "debug/smoke-summary.json",
        "debug/validation.json",
    ]
    assert (
        json.loads((tmp_path / "debug/resolved-run-config.json").read_text())
        == resolved
    )
    checkpoint = torch.load(tmp_path / "debug/last.pt", map_location="cpu")
    partial_checkpoint_bytes = (tmp_path / "debug/last.pt").read_bytes()
    assert checkpoint["accumulator"].completed_batches == 1
    assert checkpoint["accumulator"].completed_samples == 1
    assert checkpoint["sampler_state"].committed_cursor == 1
    assert checkpoint["config"] == resolved
    summary = json.loads((tmp_path / "debug/smoke-summary.json").read_text())
    assert summary["completed_train_batches"] == 1
    assert summary["completed_validation_samples"] == 1
    assert not (tmp_path / "metrics").exists()
    assert not (tmp_path / "plots").exists()
    assert not (tmp_path / "run-complete.json").exists()

    resumed_model = _TinyModel()
    resumed_train, resumed_validation, resumed_sampler, resumed_generator = _loaders(
        config
    )
    fit(
        model=resumed_model,
        train_dataloader=resumed_train,
        validation_dataloader=resumed_validation,
        sampler=resumed_sampler,
        dataloader_generator=resumed_generator,
        config=config,
        run_dir=tmp_path,
        resolved_run_config=resolved,
        identities=identities,
        base_checkpoint_sha256="5" * 64,
        resume_path=tmp_path / "debug/last.pt",
        max_epochs=1,
        max_train_batches=1,
        max_validation_samples=1,
    )
    resumed_checkpoint = torch.load(tmp_path / "debug/last.pt", map_location="cpu")
    assert resumed_checkpoint["epoch"] == 1
    assert resumed_checkpoint["global_step"] == 2
    assert resumed_checkpoint["sampler_state"].epoch == 1
    assert resumed_checkpoint["sampler_state"].committed_cursor == 0
    assert resumed_checkpoint["accumulator"].epoch == 1
    assert resumed_checkpoint["accumulator"].completed_batches == 0

    partial_copy = tmp_path / "partial-copy.pt"
    partial_copy.write_bytes(partial_checkpoint_bytes)
    mismatched_dir = tmp_path / "mismatched"
    mismatch_model = _TinyModel()
    mismatch_train, mismatch_validation, mismatch_sampler, mismatch_generator = (
        _loaders(config)
    )
    with pytest.raises(ResumeContractError, match="config"):
        fit(
            model=mismatch_model,
            train_dataloader=mismatch_train,
            validation_dataloader=mismatch_validation,
            sampler=mismatch_sampler,
            dataloader_generator=mismatch_generator,
            config=config,
            run_dir=mismatched_dir,
            resolved_run_config={**resolved, "different": True},
            identities=identities,
            base_checkpoint_sha256="5" * 64,
            resume_path=partial_copy,
            max_epochs=1,
            max_train_batches=1,
            max_validation_samples=1,
        )


def test_bounded_max_epochs_never_extends_yaml_epoch_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        TrainingConfig(),
        epochs=1,
        batch_size=1,
        num_workers=0,
        amp=False,
    )
    model = _TinyModel()
    train, validation, sampler, generator = _loaders(config)
    identities = {
        "training_manifest_sha256": "1" * 64,
        "resolved_domain_sha256": "2" * 64,
    }

    def supervised(current, features, target):
        prediction = current._backbone.image_encoder(features["x"])
        loss = (prediction - target[:, :1, :1].reshape(-1, 1)).square().mean()
        return SimpleNamespace(trajectory_loss=loss)

    def selector(current, samples, **kwargs):
        del current
        tuple(samples)
        return _validation_report(kwargs["identities"])

    def supervised_loss(current, samples, **kwargs):
        del current, kwargs
        tuple(samples)
        return ValidationLossSummary(2, 2.0, 0.15, 0.2)

    epochs = []
    real_train_one_epoch = trainer_module.train_one_epoch

    def count_epochs(**kwargs):
        epochs.append(kwargs["epoch"])
        return real_train_one_epoch(**kwargs)

    monkeypatch.setattr(trainer_module, "forward_trajectory_supervised", supervised)
    monkeypatch.setattr(trainer_module, "validate_selector_b1", selector)
    monkeypatch.setattr(trainer_module, "validate_supervised_loss_b1", supervised_loss)
    monkeypatch.setattr(trainer_module, "train_one_epoch", count_epochs)

    fit(
        model=model,
        train_dataloader=train,
        validation_dataloader=validation,
        sampler=sampler,
        dataloader_generator=generator,
        config=config,
        run_dir=tmp_path,
        resolved_run_config={
            "training_epochs": 1,
            "smoke_bounds": {"max_epochs": 99},
        },
        identities=identities,
        base_checkpoint_sha256="5" * 64,
        max_epochs=99,
    )

    assert epochs == [0]
    checkpoint = torch.load(tmp_path / "debug/last.pt", map_location="cpu")
    assert checkpoint["epoch"] == 1


def test_resume_reconciles_durable_epoch_before_sampler_finished_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        TrainingConfig(),
        epochs=2,
        batch_size=1,
        num_workers=0,
        amp=False,
        checkpoint_interval_steps=1,
    )
    identities = {
        "training_manifest_sha256": "1" * 64,
        "resolved_domain_sha256": "2" * 64,
    }
    resolved = {"epochs": 2, "smoke_bounds": None}

    def supervised(current, features, target):
        prediction = current._backbone.image_encoder(features["x"])
        loss = (prediction - target[:, :1, :1].reshape(-1, 1)).square().mean()
        return SimpleNamespace(trajectory_loss=loss)

    validation_calls = []

    def selector(current, samples, **kwargs):
        del current
        tuple(samples)
        validation_calls.append("selector")
        return _validation_report(kwargs["identities"])

    def supervised_loss(current, samples, **kwargs):
        del current, kwargs
        tuple(samples)
        validation_calls.append("loss")
        return ValidationLossSummary(2, 2.0, 0.15, 0.2)

    monkeypatch.setattr(trainer_module, "forward_trajectory_supervised", supervised)
    monkeypatch.setattr(trainer_module, "validate_selector_b1", selector)
    monkeypatch.setattr(trainer_module, "validate_supervised_loss_b1", supervised_loss)
    first_model = _TinyModel()
    first_train, first_validation, first_sampler, first_generator = _loaders(config)
    real_finish = first_sampler.finish_epoch
    crashed = False

    def crash_after_epoch_json():
        nonlocal crashed
        if not crashed:
            crashed = True
            assert (tmp_path / "metrics/epochs/epoch-0000.json").is_file()
            raise RuntimeError("simulated crash before sampler-finished last")
        real_finish()

    monkeypatch.setattr(first_sampler, "finish_epoch", crash_after_epoch_json)

    with pytest.raises(RuntimeError, match="simulated crash"):
        fit(
            model=first_model,
            train_dataloader=first_train,
            validation_dataloader=first_validation,
            sampler=first_sampler,
            dataloader_generator=first_generator,
            config=config,
            run_dir=tmp_path,
            resolved_run_config=resolved,
            identities=identities,
            base_checkpoint_sha256="5" * 64,
        )

    assert validation_calls == ["selector", "loss"]
    assert not (tmp_path / "run-complete.json").exists()
    resumed_model = _TinyModel()
    resumed_train, resumed_validation, resumed_sampler, resumed_generator = _loaders(
        config
    )

    attestation = fit(
        model=resumed_model,
        train_dataloader=resumed_train,
        validation_dataloader=resumed_validation,
        sampler=resumed_sampler,
        dataloader_generator=resumed_generator,
        config=config,
        run_dir=tmp_path,
        resolved_run_config=resolved,
        identities=identities,
        base_checkpoint_sha256="5" * 64,
        resume_path=tmp_path / "last.pt",
    )

    assert isinstance(attestation, ProductionRunAttestation)
    assert validation_calls == [
        "selector",
        "loss",
        "selector",
        "loss",
        "selector",
        "loss",
    ]
    assert resumed_sampler.state_dict().epoch == 2
    assert (tmp_path / "metrics/epochs/epoch-0001.json").is_file()
    assert (tmp_path / "run-complete.json").is_file()


@pytest.mark.parametrize("corruption", ["missing", "mismatched"])
def test_next_epoch_resume_rejects_missing_or_mismatched_epoch_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    config = replace(
        TrainingConfig(),
        epochs=2,
        batch_size=1,
        num_workers=0,
        amp=False,
        checkpoint_interval_steps=1,
    )
    identities = {
        "training_manifest_sha256": "1" * 64,
        "resolved_domain_sha256": "2" * 64,
    }
    resolved = {"epochs": 2, "smoke_bounds": None}

    def supervised(current, features, target):
        prediction = current._backbone.image_encoder(features["x"])
        loss = (prediction - target[:, :1, :1].reshape(-1, 1)).square().mean()
        return SimpleNamespace(trajectory_loss=loss)

    def selector(current, samples, **kwargs):
        del current
        tuple(samples)
        return _validation_report(kwargs["identities"])

    def supervised_loss(current, samples, **kwargs):
        del current, kwargs
        tuple(samples)
        return ValidationLossSummary(2, 2.0, 0.15, 0.2)

    monkeypatch.setattr(trainer_module, "forward_trajectory_supervised", supervised)
    monkeypatch.setattr(trainer_module, "validate_selector_b1", selector)
    monkeypatch.setattr(trainer_module, "validate_supervised_loss_b1", supervised_loss)
    real_rebuild = trainer_module.rebuild_training_artifacts

    def crash_after_next_epoch_last(_run_dir):
        checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu")
        assert checkpoint["epoch"] == 1
        raise RuntimeError("simulated crash after next-epoch last")

    monkeypatch.setattr(
        trainer_module, "rebuild_training_artifacts", crash_after_next_epoch_last
    )
    model = _TinyModel()
    train, validation, sampler, generator = _loaders(config)
    with pytest.raises(RuntimeError, match="next-epoch last"):
        fit(
            model=model,
            train_dataloader=train,
            validation_dataloader=validation,
            sampler=sampler,
            dataloader_generator=generator,
            config=config,
            run_dir=tmp_path,
            resolved_run_config=resolved,
            identities=identities,
            base_checkpoint_sha256="5" * 64,
        )
    monkeypatch.setattr(trainer_module, "rebuild_training_artifacts", real_rebuild)

    epoch_path = tmp_path / "metrics/epochs/epoch-0000.json"
    if corruption == "missing":
        epoch_path.unlink()
    else:
        checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu")
        checkpoint["completed_epoch_record_sha256"] = "9" * 64
        torch.save(checkpoint, tmp_path / "last.pt")

    resumed_model = _TinyModel()
    resumed_train, resumed_validation, resumed_sampler, resumed_generator = _loaders(
        config
    )
    with pytest.raises(ResumeContractError, match="epoch|hash|telemetry"):
        fit(
            model=resumed_model,
            train_dataloader=resumed_train,
            validation_dataloader=resumed_validation,
            sampler=resumed_sampler,
            dataloader_generator=resumed_generator,
            config=config,
            run_dir=tmp_path,
            resolved_run_config=resolved,
            identities=identities,
            base_checkpoint_sha256="5" * 64,
            resume_path=tmp_path / "last.pt",
        )


def test_fit_mid_epoch_resume_continues_global_step_rng_and_accumulator_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        TrainingConfig(),
        epochs=1,
        batch_size=1,
        num_workers=0,
        amp=False,
        checkpoint_interval_steps=1,
    )
    identities = {
        "training_manifest_sha256": "1" * 64,
        "resolved_domain_sha256": "2" * 64,
    }
    resolved = {"epochs": 1, "smoke_bounds": None}
    base_model = _TinyModel()
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in base_model.state_dict().items()
    }

    def supervised(current, features, target):
        prediction = current._backbone.image_encoder(features["x"])
        loss = (prediction - target[:, :1, :1].reshape(-1, 1)).square().mean()
        return SimpleNamespace(trajectory_loss=loss)

    def selector(current, samples, **kwargs):
        del current
        tuple(samples)
        return _validation_report(kwargs["identities"])

    def supervised_loss(current, samples, **kwargs):
        del current, kwargs
        tuple(samples)
        return ValidationLossSummary(2, 2.0, 0.15, 0.2)

    monkeypatch.setattr(trainer_module, "forward_trajectory_supervised", supervised)
    monkeypatch.setattr(trainer_module, "validate_selector_b1", selector)
    monkeypatch.setattr(trainer_module, "validate_supervised_loss_b1", supervised_loss)

    baseline_dir = tmp_path / "baseline"
    baseline_model = _TinyModel()
    baseline_model.load_state_dict(base_state)
    baseline_train, baseline_validation, baseline_sampler, baseline_generator = (
        _loaders(config)
    )
    random.seed(7001)
    np.random.seed(7001)
    torch.manual_seed(7001)
    fit(
        model=baseline_model,
        train_dataloader=baseline_train,
        validation_dataloader=baseline_validation,
        sampler=baseline_sampler,
        dataloader_generator=baseline_generator,
        config=config,
        run_dir=baseline_dir,
        resolved_run_config=resolved,
        identities=identities,
        base_checkpoint_sha256="5" * 64,
    )
    baseline_last = torch.load(baseline_dir / "last.pt", map_location="cpu")

    resumed_dir = tmp_path / "resumed"
    interrupted_model = _TinyModel()
    interrupted_model.load_state_dict(base_state)
    (
        interrupted_train,
        interrupted_validation,
        interrupted_sampler,
        interrupted_generator,
    ) = _loaders(config)
    real_train_step = trainer_module.train_step
    attempts = 0

    def crash_on_second_step(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("simulated mid-epoch crash")
        return real_train_step(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "train_step", crash_on_second_step)
    random.seed(7001)
    np.random.seed(7001)
    torch.manual_seed(7001)
    with pytest.raises(RuntimeError, match="mid-epoch crash"):
        fit(
            model=interrupted_model,
            train_dataloader=interrupted_train,
            validation_dataloader=interrupted_validation,
            sampler=interrupted_sampler,
            dataloader_generator=interrupted_generator,
            config=config,
            run_dir=resumed_dir,
            resolved_run_config=resolved,
            identities=identities,
            base_checkpoint_sha256="5" * 64,
        )
    durable = torch.load(resumed_dir / "last.pt", map_location="cpu")
    assert durable["epoch"] == 0
    assert durable["global_step"] == 1
    assert durable["accumulator"].completed_batches == 1
    assert durable["sampler_state"].committed_cursor == 1
    assert not (resumed_dir / "run-complete.json").exists()

    monkeypatch.setattr(trainer_module, "train_step", real_train_step)
    real_train_one_epoch = trainer_module.train_one_epoch
    resumed_inputs = []

    def capture_resumed_inputs(**kwargs):
        resumed_inputs.append(
            (kwargs["global_step"], kwargs["accumulator"], kwargs["resume_rng"])
        )
        return real_train_one_epoch(**kwargs)

    monkeypatch.setattr(trainer_module, "train_one_epoch", capture_resumed_inputs)
    resumed_model = _TinyModel()
    resumed_model.load_state_dict(base_state)
    resumed_train, resumed_validation, resumed_sampler, resumed_generator = _loaders(
        config
    )
    fit(
        model=resumed_model,
        train_dataloader=resumed_train,
        validation_dataloader=resumed_validation,
        sampler=resumed_sampler,
        dataloader_generator=resumed_generator,
        config=config,
        run_dir=resumed_dir,
        resolved_run_config=resolved,
        identities=identities,
        base_checkpoint_sha256="5" * 64,
        resume_path=resumed_dir / "last.pt",
    )

    assert len(resumed_inputs) == 1
    incoming_step, incoming_accumulator, incoming_rng = resumed_inputs[0]
    assert incoming_step == durable["global_step"]
    assert incoming_accumulator == durable["accumulator"]
    _assert_nested_equal(incoming_rng, durable["rng"])
    resumed_last = torch.load(resumed_dir / "last.pt", map_location="cpu")
    for key in (
        "mutable_state",
        "optimizer",
        "scaler",
        "epoch",
        "global_step",
        "best_micro_ade",
        "rng",
        "sampler_state",
        "accumulator",
        "completed_epoch_record_sha256",
        "base_checkpoint_sha256",
        "manifest",
        "config",
    ):
        _assert_nested_equal(resumed_last[key], baseline_last[key])
    for relative in (
        "metrics/epochs/epoch-0000.json",
        "metrics/history.jsonl",
        "metrics/epoch_metrics.csv",
        "validation/epoch-0000.json",
    ):
        assert (resumed_dir / relative).read_bytes() == (
            baseline_dir / relative
        ).read_bytes()
