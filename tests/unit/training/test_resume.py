from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import replace
import gc
import math
import os
from pathlib import Path
import random
import socket
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from morai_ddv2.config import sha256_file
from morai_ddv2.training.checkpoint import (
    inspect_training_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from morai_ddv2.training.contracts import (
    EpochAccumulatorState,
    StepMetrics,
)
from morai_ddv2.training.freeze import (
    build_optimizer,
    configure_training_boundary,
    extract_mutable_state,
)
from morai_ddv2.training.loss import snapshot_source_parameters
from morai_ddv2.training.rng import (
    capture_rng_state,
    create_dataloader_iterator,
    restore_rng_state,
)
from morai_ddv2.training.sampler import ResumeContractError, StatefulPermutationSampler
from morai_ddv2.training.trainer import train_one_epoch, train_step
from morai_ddv2.upstream.ddv2_sel import (
    SelAssets,
    load_pdm_free_sel_network,
)


class _ResumeDataset(Dataset):
    def __init__(self, size: int = 8) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int):
        value = float(index + 1)
        return {
            "features": {
                "x": torch.tensor([value, value * value], dtype=torch.float32)
            },
            "trajectory": torch.tensor(
                [value * 0.25, -value * 0.5], dtype=torch.float32
            ),
            "metadata": {"sample_id": "sample-{:02d}".format(index)},
        }


class _RealCudaDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        scale = float(index + 1)
        x = torch.arange(1, 9, dtype=torch.float32) * (0.1 * scale)
        zeros = torch.zeros_like(x)
        return {
            "features": {
                "camera_feature": torch.full(
                    (3, 256, 1024),
                    0.01 * scale,
                    dtype=torch.float32,
                ),
                "lidar_feature": torch.full(
                    (1, 256, 256),
                    0.005 * scale,
                    dtype=torch.float32,
                ),
                "status_feature": torch.tensor(
                    [scale, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=torch.float32,
                ),
            },
            "trajectory": torch.stack((x, zeros, zeros), dim=-1),
            "metadata": {"sample_id": "real-{:02d}".format(index)},
        }


class _ResumeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._backbone = nn.Module()
        self._backbone.image_encoder = nn.Sequential(
            OrderedDict(
                (
                    ("projection", nn.Linear(2, 2, bias=False)),
                    ("norm", nn.BatchNorm1d(2)),
                )
            )
        )
        self.frozen_head = nn.Linear(2, 2, bias=False)


def _zero_accumulator() -> EpochAccumulatorState:
    return EpochAccumulatorState(
        epoch=0,
        completed_batches=0,
        completed_samples=0,
        total_loss_numerator=0.0,
        trajectory_loss_numerator=0.0,
        preservation_loss_numerator=0.0,
        grad_norm_sum=0.0,
        grad_norm_count=0,
    )


def test_model_free_checkpoint_inspection_accepts_bound_resume_identity_and_rejects_replacement(
    tmp_path: Path,
) -> None:
    (
        model,
        manifest,
        _reference,
        optimizer,
        scaler,
        sampler,
        generator,
        _loader,
    ) = _new_training_objects(num_workers=0)
    stored_config = {
        "schema_version": 1,
        "training": {"epochs": 30},
        "identities": {
            "training_manifest_sha256": "1" * 64,
            "resolved_domain_sha256": "2" * 64,
            "base_checkpoint_sha256": "3" * 64,
        },
        "smoke_bounds": {
            "max_epochs": None,
            "max_train_batches": None,
            "max_validation_samples": None,
        },
    }
    checkpoint = tmp_path / "last.pt"
    save_training_checkpoint(
        checkpoint,
        mutable_state=extract_mutable_state(model, manifest),
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
        base_checkpoint_sha256="3" * 64,
        manifest=manifest,
        config=stored_config,
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    resume_config = dict(stored_config)
    resume_config["resume_checkpoint_sha256"] = checkpoint_sha256

    inspected = inspect_training_checkpoint(
        checkpoint,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_base_checkpoint_sha256="3" * 64,
        expected_config=resume_config,
        expected_identities={"training_manifest_sha256": "1" * 64},
    )
    assert inspected == stored_config

    checkpoint.write_bytes(checkpoint.read_bytes() + b"replaced")
    with pytest.raises(ResumeContractError, match="SHA-256"):
        inspect_training_checkpoint(
            checkpoint,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_base_checkpoint_sha256="3" * 64,
            expected_config=resume_config,
        )


def test_model_free_checkpoint_inspection_rejects_cache_identity_mismatch(
    tmp_path: Path,
) -> None:
    (
        model,
        manifest,
        _reference,
        optimizer,
        scaler,
        sampler,
        generator,
        _loader,
    ) = _new_training_objects(num_workers=0)
    config = {
        "schema_version": 1,
        "training": {},
        "identities": {
            "training_manifest_sha256": "1" * 64,
            "resolved_domain_sha256": "2" * 64,
            "base_checkpoint_sha256": "3" * 64,
        },
        "smoke_bounds": {
            "max_epochs": None,
            "max_train_batches": None,
            "max_validation_samples": None,
        },
    }
    checkpoint = tmp_path / "other-cache.pt"
    save_training_checkpoint(
        checkpoint,
        mutable_state=extract_mutable_state(model, manifest),
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
        base_checkpoint_sha256="3" * 64,
        manifest=manifest,
        config=config,
    )

    with pytest.raises(ResumeContractError, match="identity"):
        inspect_training_checkpoint(
            checkpoint,
            expected_checkpoint_sha256=sha256_file(checkpoint),
            expected_base_checkpoint_sha256="3" * 64,
            expected_identities={"training_manifest_sha256": "9" * 64},
        )


def _advance_accumulator(
    state: EpochAccumulatorState,
    metrics: StepMetrics,
    batch_size: int,
) -> EpochAccumulatorState:
    assert metrics.optimizer_step_applied
    return replace(
        state,
        completed_batches=state.completed_batches + 1,
        completed_samples=state.completed_samples + batch_size,
        total_loss_numerator=(
            state.total_loss_numerator + metrics.total_loss * batch_size
        ),
        trajectory_loss_numerator=(
            state.trajectory_loss_numerator + metrics.trajectory_loss * batch_size
        ),
        preservation_loss_numerator=(
            state.preservation_loss_numerator + metrics.preservation_loss * batch_size
        ),
        grad_norm_sum=state.grad_norm_sum + metrics.grad_norm,
        grad_norm_count=state.grad_norm_count + 1,
    )


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _assert_nested_close(
    left,
    right,
    *,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-7,
) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        if left.is_floating_point() or left.is_complex():
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
        else:
            assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key], rtol=rtol, atol=atol)
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_close(left_item, right_item, rtol=rtol, atol=atol)
    else:
        assert left == right


def _assert_rng_equal(left, right) -> None:
    assert left.python == right.python
    assert left.numpy[0] == right.numpy[0]
    assert np.array_equal(left.numpy[1], right.numpy[1])
    assert left.numpy[2:] == right.numpy[2:]
    assert torch.equal(left.torch_cpu, right.torch_cpu)
    assert len(left.torch_cuda) == len(right.torch_cuda)
    assert all(
        torch.equal(left_state, right_state)
        for left_state, right_state in zip(left.torch_cuda, right.torch_cuda)
    )
    assert torch.equal(left.dataloader_cpu, right.dataloader_cpu)
    assert torch.equal(
        left.dataloader_iterator_base_cpu,
        right.dataloader_iterator_base_cpu,
    )


def _new_training_objects(
    *,
    num_workers: int,
    device: torch.device = torch.device("cpu"),
):
    torch.manual_seed(700)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(700)
    model = _ResumeModel().to(device)
    manifest = configure_training_boundary(model)
    reference = snapshot_source_parameters(model, manifest)
    optimizer = build_optimizer(model, manifest, lr=1.0e-3, weight_decay=1.0e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    dataset = _ResumeDataset()
    sampler = StatefulPermutationSampler(dataset, seed=701)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(702)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers else None,
        generator=generator,
        persistent_workers=False,
    )
    return model, manifest, reference, optimizer, scaler, sampler, generator, loader


def _install_stochastic_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    from morai_ddv2.training import trainer as trainer_module

    def fake_forward(model, features, target):
        device = next(model.parameters()).device
        output = model._backbone.image_encoder(features["x"].to(device))
        target = target.to(device)
        random_factor = random.random() + float(np.random.random_sample())
        random_factor += float(torch.rand((), device=device))
        loss = (output - target).square().mean()
        loss = loss + output.mean() * (1.0e-3 * random_factor)
        return SimpleNamespace(trajectory_loss=loss)

    monkeypatch.setattr(
        trainer_module,
        "forward_trajectory_supervised",
        fake_forward,
    )


def _run_steps(
    *,
    iterator,
    count,
    model,
    optimizer,
    scaler,
    reference,
    manifest,
    sampler,
    accumulator,
    global_step,
    amp=False,
):
    sample_ids = []
    for _ in range(count):
        batch = next(iterator)
        sample_ids.extend(batch["metadata"]["sample_id"])
        metrics = train_step(
            model,
            batch,
            optimizer,
            scaler,
            reference,
            manifest,
            amp=amp,
            grad_clip_norm=1.0,
            trajectory_weight=12.0,
            preservation_weight=1.0e-3,
        )
        assert metrics.optimizer_step_applied
        batch_size = int(batch["trajectory"].shape[0])
        sampler.mark_consumed(batch_size)
        accumulator = _advance_accumulator(accumulator, metrics, batch_size)
        global_step += 1
    return accumulator, global_step, sample_ids


def _worker_prefetch_supported() -> bool:
    probe = None
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError:
        return False
    finally:
        if probe is not None:
            probe.close()
    return True


_REAL_CHECKPOINT_SHA256 = (
    "452c4e0dd328c49b981f3fd5ed13eb1c6a2b0f3a9486adf43b40a5645ed543b1"
)
# RTX 3090 observations reached 0.2% loss drift and 4.8% grad-norm drift from
# grid_sample's atomic CUDA backward. Floors add margin; ceilings reject runaway
# variability; the measured envelope is the larger of each floor or 2x baseline.
_CUDA_VARIABILITY_FACTOR = 2.0
_CUDA_LOSS_RELATIVE_FLOOR = 5.0e-3
_CUDA_LOSS_RELATIVE_CEILING = 2.0e-2
_CUDA_GRAD_NORM_RELATIVE_FLOOR = 7.5e-2
_CUDA_GRAD_NORM_RELATIVE_CEILING = 1.5e-1
_CUDA_MODEL_UPDATE_RELATIVE_FLOOR = 1.0e-1
_CUDA_MODEL_UPDATE_RELATIVE_CEILING = 5.0e-1
_CUDA_OPTIMIZER_UPDATE_RELATIVE_FLOOR = 1.0e-1
_CUDA_OPTIMIZER_UPDATE_RELATIVE_CEILING = 5.0e-1
_CUDA_SCALAR_ATOL = 1.0e-6
_CUDA_UPDATE_NORM_ATOL = 1.0e-12


def _real_sel_assets() -> SelAssets:
    repository_root = Path(__file__).resolve().parents[3]
    validator_value = os.environ.get("MORAI_DDV2_VALIDATOR_ROOT")
    validator_root = (
        Path(validator_value).resolve()
        if validator_value
        else (repository_root / "third_party/E2E-OpenLoop-ROS-Validator").resolve()
    )
    checkpoint_value = os.environ.get("MORAI_DDV2_SEL_CHECKPOINT")
    checkpoint = (
        Path(checkpoint_value).resolve()
        if checkpoint_value
        else validator_root
        / "models/diffusiondrive_v2/navsim_v2/diffusiondrivev2_sel.ckpt"
    )
    assets = SelAssets(
        v2_checkout=validator_root / "models/sources/diffusiondrive_v2",
        validator_root=validator_root,
        checkpoint=checkpoint,
        image_backbone=(
            validator_root / "models/shared/timm/resnet34.a1_in1k/pytorch_model.bin"
        ),
        trajectory_anchor=(
            validator_root
            / "models/diffusiondrive_v2/navsim_v2/kmeans_navsim_traj_20.npy"
        ),
    )
    required = (
        assets.v2_checkout / "navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py",
        assets.checkpoint,
        assets.image_backbone,
        assets.trajectory_anchor,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip("pinned DDV2 CUDA assets are unavailable: {}".format(missing))
    return assets


def _new_real_cuda_training_objects(assets: SelAssets):
    device = torch.device("cuda", 0)
    model, raw = load_pdm_free_sel_network(assets, device)
    assert raw.checkpoint_sha256 == _REAL_CHECKPOINT_SHA256
    del raw
    assert len(model.state_dict()) == 971
    assert model.__class__.__module__ == (
        "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference"
    )
    manifest = configure_training_boundary(model)
    assert len(manifest.trainable_parameters) == 372
    assert len(manifest.mutable_state_entries) == 588
    reference = snapshot_source_parameters(model, manifest)
    optimizer = build_optimizer(model, manifest, lr=1.0e-5, weight_decay=1.0e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=1.0, enabled=True)
    dataset = _RealCudaDataset()
    sampler = StatefulPermutationSampler(dataset, seed=1201)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1202)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    return (
        model,
        manifest,
        reference,
        optimizer,
        scaler,
        sampler,
        generator,
        loader,
    )


def _seed_real_cuda_run() -> None:
    random.seed(1203)
    np.random.seed(1204)
    torch.manual_seed(1205)
    torch.cuda.manual_seed_all(1205)


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _owned_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {key: _owned_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_owned_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_owned_cpu(item) for item in value)
    return copy.deepcopy(value)


def _assert_finite_nested(value) -> None:
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_nested(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_nested(item)


def _relative_scalar_difference(left: float, right: float, *, atol: float) -> float:
    assert math.isfinite(left)
    assert math.isfinite(right)
    difference = abs(left - right)
    if difference <= atol:
        return 0.0
    scale = max(abs(left), abs(right), atol)
    return (difference - atol) / scale


def _assert_cuda_divergence_envelope(
    *,
    baseline_divergence: float,
    resumed_divergence: float,
    relative_floor: float,
    relative_ceiling: float,
    label: str,
) -> None:
    assert math.isfinite(baseline_divergence)
    assert math.isfinite(resumed_divergence)
    assert 0.0 <= baseline_divergence <= relative_ceiling, (
        f"{label} baseline divergence {baseline_divergence:.6g} exceeds "
        f"ceiling {relative_ceiling:.6g}"
    )
    assert 0.0 <= resumed_divergence <= relative_ceiling, (
        f"{label} resumed divergence {resumed_divergence:.6g} exceeds "
        f"ceiling {relative_ceiling:.6g}"
    )
    measured_envelope = max(
        relative_floor,
        _CUDA_VARIABILITY_FACTOR * baseline_divergence,
    )
    assert resumed_divergence <= measured_envelope, (
        f"{label} resumed divergence {resumed_divergence:.6g} exceeds "
        f"measured envelope {measured_envelope:.6g} "
        f"(baseline={baseline_divergence:.6g})"
    )


def _assert_cuda_scalar_continuity(
    *,
    uninterrupted: float,
    repeated_uninterrupted: float,
    resumed: float,
    relative_floor: float,
    relative_ceiling: float,
    label: str,
    atol: float = _CUDA_SCALAR_ATOL,
) -> None:
    _assert_cuda_divergence_envelope(
        baseline_divergence=_relative_scalar_difference(
            uninterrupted,
            repeated_uninterrupted,
            atol=atol,
        ),
        resumed_divergence=_relative_scalar_difference(
            repeated_uninterrupted,
            resumed,
            atol=atol,
        ),
        relative_floor=relative_floor,
        relative_ceiling=relative_ceiling,
        label=label,
    )


def _floating_update_norm(before, after, *, included_leaf_names=None) -> float:
    squared_norm = 0.0
    found = False

    def visit(left, right, leaf_name=None) -> None:
        nonlocal found, squared_norm
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor)
            assert left.shape == right.shape
            assert left.dtype == right.dtype
            if not left.is_floating_point():
                return
            if included_leaf_names is not None and leaf_name not in included_leaf_names:
                return
            update = right.detach().to(dtype=torch.float64) - left.detach().to(
                dtype=torch.float64
            )
            assert torch.isfinite(update).all()
            squared_norm += float(update.square().sum())
            found = True
        elif isinstance(left, dict):
            assert isinstance(right, dict)
            assert left.keys() == right.keys()
            for key in left:
                visit(left[key], right[key], str(key))
        elif isinstance(left, (list, tuple)):
            assert type(left) is type(right)
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                visit(left_item, right_item)
        else:
            assert type(left) is type(right)

    visit(before, after)
    assert found
    return math.sqrt(squared_norm)


def _optimizer_step_values(state) -> tuple[float, ...]:
    values = []
    for parameter_state in state["state"].values():
        step = parameter_state["step"]
        assert isinstance(step, torch.Tensor)
        assert step.numel() == 1
        value = float(step)
        assert math.isfinite(value)
        values.append(value)
    assert values
    return tuple(values)


def _run_real_cuda_committed_steps(
    *,
    iterator,
    iterator_base,
    dataloader,
    generator,
    count,
    model,
    optimizer,
    scaler,
    reference,
    manifest,
    sampler,
    accumulator,
    global_step,
):
    committed_ids = []
    committed_metrics = []
    attempts = []
    maximum_attempts = count + 24
    while len(committed_ids) < count:
        assert len(attempts) < maximum_attempts, (
            "GradScaler never reached a finite step"
        )
        batch = next(iterator)
        batch_ids = tuple(batch["metadata"]["sample_id"])
        device_batch = _to_device(batch, torch.device("cuda", 0))
        metrics = train_step(
            model,
            device_batch,
            optimizer,
            scaler,
            reference,
            manifest,
            amp=True,
            grad_clip_norm=1.0,
            trajectory_weight=12.0,
            preservation_weight=1.0e-3,
        )
        attempts.append((batch_ids, metrics.optimizer_step_applied))
        assert all(
            math.isfinite(value)
            for value in (
                metrics.total_loss,
                metrics.trajectory_loss,
                metrics.preservation_loss,
                metrics.grad_norm,
            )
        )
        if not metrics.optimizer_step_applied:
            current_generator_state = generator.get_state().clone()
            del iterator, device_batch, batch
            iterator, _ = create_dataloader_iterator(
                dataloader,
                generator,
                iterator_base_state=iterator_base,
                current_generator_state=current_generator_state,
            )
            continue
        assert metrics.grad_norm > 0.0
        batch_size = int(device_batch["trajectory"].shape[0])
        sampler.mark_consumed(batch_size)
        accumulator = _advance_accumulator(accumulator, metrics, batch_size)
        global_step += 1
        committed_ids.extend(batch_ids)
        committed_metrics.append(metrics)
    return (
        iterator,
        accumulator,
        global_step,
        committed_ids,
        attempts,
        committed_metrics,
    )


def _assert_mid_epoch_resume_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_workers: int,
    device: torch.device,
    explicit_run_ahead: bool,
) -> None:
    _install_stochastic_fake(monkeypatch)

    random.seed(703)
    np.random.seed(704)
    torch.manual_seed(705)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(705)
    (
        full_model,
        full_manifest,
        full_reference,
        full_optimizer,
        full_scaler,
        full_sampler,
        full_generator,
        full_loader,
    ) = _new_training_objects(num_workers=num_workers, device=device)
    full_iterator, full_base = create_dataloader_iterator(full_loader, full_generator)
    full_accumulator, full_global_step, full_ids = _run_steps(
        iterator=full_iterator,
        count=4,
        model=full_model,
        optimizer=full_optimizer,
        scaler=full_scaler,
        reference=full_reference,
        manifest=full_manifest,
        sampler=full_sampler,
        accumulator=_zero_accumulator(),
        global_step=0,
        amp=device.type == "cuda",
    )
    full_rng = capture_rng_state(
        full_generator,
        dataloader_iterator_base_cpu=full_base,
    )

    random.seed(703)
    np.random.seed(704)
    torch.manual_seed(705)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(705)
    (
        interrupted_model,
        interrupted_manifest,
        interrupted_reference,
        interrupted_optimizer,
        interrupted_scaler,
        interrupted_sampler,
        interrupted_generator,
        interrupted_loader,
    ) = _new_training_objects(num_workers=num_workers, device=device)
    interrupted_iterator, interrupted_base = create_dataloader_iterator(
        interrupted_loader, interrupted_generator
    )
    if explicit_run_ahead:
        prefetched_batches = [next(interrupted_iterator) for _ in range(4)]
        committed_iterator = iter(prefetched_batches)
    else:
        prefetched_batches = None
        committed_iterator = interrupted_iterator
    interrupted_accumulator, interrupted_global_step, first_ids = _run_steps(
        iterator=committed_iterator,
        count=2,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        scaler=interrupted_scaler,
        reference=interrupted_reference,
        manifest=interrupted_manifest,
        sampler=interrupted_sampler,
        accumulator=_zero_accumulator(),
        global_step=0,
        amp=device.type == "cuda",
    )
    saved_rng = capture_rng_state(
        interrupted_generator,
        dataloader_iterator_base_cpu=interrupted_base,
    )
    checkpoint_path = tmp_path / "last.pt"
    save_training_checkpoint(
        checkpoint_path,
        mutable_state=extract_mutable_state(interrupted_model, interrupted_manifest),
        optimizer=interrupted_optimizer,
        scaler=interrupted_scaler,
        epoch=0,
        global_step=interrupted_global_step,
        best_micro_ade=8.0,
        rng=saved_rng,
        sampler_state=interrupted_sampler.state_dict(),
        accumulator=interrupted_accumulator,
        completed_epoch_record_sha256=None,
        base_checkpoint_sha256="d" * 64,
        manifest=interrupted_manifest,
        config={"checkpoint_interval_steps": 2},
    )
    if prefetched_batches is not None:
        assert len(prefetched_batches[2:]) == 2
    del interrupted_iterator, committed_iterator, prefetched_batches
    gc.collect()

    (
        resumed_model,
        resumed_manifest,
        resumed_reference,
        resumed_optimizer,
        resumed_scaler,
        resumed_sampler,
        resumed_generator,
        resumed_loader,
    ) = _new_training_objects(num_workers=num_workers, device=device)
    resume = load_training_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        sampler=resumed_sampler,
        manifest=resumed_manifest,
        expected_base_checkpoint_sha256="d" * 64,
    )
    assert resume.global_step == interrupted_global_step
    assert resume.epoch == 0
    assert resume.accumulator == interrupted_accumulator
    assert resume.sampler.committed_cursor == 4
    restore_rng_state(resume.rng, resumed_generator)
    resumed_iterator, resumed_base = create_dataloader_iterator(
        resumed_loader,
        resumed_generator,
        iterator_base_state=resume.rng.dataloader_iterator_base_cpu,
        current_generator_state=resume.rng.dataloader_cpu,
    )
    assert torch.equal(resumed_base, interrupted_base)
    assert torch.equal(resumed_generator.get_state(), resume.rng.dataloader_cpu)
    resumed_accumulator, resumed_global_step, remaining_ids = _run_steps(
        iterator=resumed_iterator,
        count=2,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        reference=resumed_reference,
        manifest=resumed_manifest,
        sampler=resumed_sampler,
        accumulator=resume.accumulator,
        global_step=resume.global_step,
        amp=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    resumed_rng = capture_rng_state(
        resumed_generator,
        dataloader_iterator_base_cpu=resumed_base,
    )

    assert first_ids + remaining_ids == full_ids
    assert resumed_global_step == full_global_step == 4
    if device.type == "cuda":
        assert resumed_accumulator.epoch == full_accumulator.epoch
        assert (
            resumed_accumulator.completed_batches == full_accumulator.completed_batches
        )
        assert (
            resumed_accumulator.completed_samples == full_accumulator.completed_samples
        )
        assert resumed_accumulator.grad_norm_count == full_accumulator.grad_norm_count
        assert resumed_accumulator.total_loss_numerator == pytest.approx(
            full_accumulator.total_loss_numerator, rel=1.0e-6, abs=1.0e-7
        )
        assert resumed_accumulator.trajectory_loss_numerator == pytest.approx(
            full_accumulator.trajectory_loss_numerator, rel=1.0e-6, abs=1.0e-7
        )
        assert resumed_accumulator.preservation_loss_numerator == pytest.approx(
            full_accumulator.preservation_loss_numerator, rel=1.0e-6, abs=1.0e-7
        )
        assert resumed_accumulator.grad_norm_sum == pytest.approx(
            full_accumulator.grad_norm_sum, rel=1.0e-6, abs=1.0e-7
        )
    else:
        assert resumed_accumulator == full_accumulator
    assert resumed_sampler.state_dict().committed_cursor == len(_ResumeDataset())
    assert resumed_sampler.state_dict().epoch == full_sampler.state_dict().epoch == 0
    assert torch.equal(
        resumed_sampler.state_dict().permutation,
        full_sampler.state_dict().permutation,
    )
    for name, full_tensor in full_model.state_dict().items():
        if device.type == "cuda":
            torch.testing.assert_close(
                resumed_model.state_dict()[name],
                full_tensor,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        else:
            assert torch.equal(resumed_model.state_dict()[name], full_tensor), name
    if device.type == "cuda":
        _assert_nested_close(
            resumed_optimizer.state_dict(), full_optimizer.state_dict()
        )
    else:
        _assert_nested_equal(
            resumed_optimizer.state_dict(), full_optimizer.state_dict()
        )
    _assert_nested_equal(resumed_scaler.state_dict(), full_scaler.state_dict())
    _assert_rng_equal(resumed_rng, full_rng)


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
def test_prefetched_mid_epoch_checkpoint_resume_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_mid_epoch_resume_exact(
        tmp_path,
        monkeypatch,
        num_workers=0,
        device=torch.device("cpu"),
        explicit_run_ahead=True,
    )


@pytest.mark.skipif(
    not _worker_prefetch_supported(),
    reason="sandbox forbids DataLoader multiprocessing resource-sharing sockets",
)
def test_two_worker_prefetch_mid_epoch_resume_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_mid_epoch_resume_exact(
        tmp_path,
        monkeypatch,
        num_workers=2,
        device=torch.device("cpu"),
        explicit_run_ahead=False,
    )


def test_cuda_continuity_envelope_uses_measured_variability_and_caps_runaway():
    _assert_cuda_divergence_envelope(
        baseline_divergence=0.04,
        resumed_divergence=0.07,
        relative_floor=0.01,
        relative_ceiling=0.10,
        label="synthetic",
    )

    with pytest.raises(AssertionError, match="baseline.*ceiling"):
        _assert_cuda_divergence_envelope(
            baseline_divergence=0.11,
            resumed_divergence=0.01,
            relative_floor=0.01,
            relative_ceiling=0.10,
            label="synthetic",
        )
    with pytest.raises(AssertionError, match="resumed.*ceiling"):
        _assert_cuda_divergence_envelope(
            baseline_divergence=0.01,
            resumed_divergence=0.11,
            relative_floor=0.01,
            relative_ceiling=0.10,
            label="synthetic",
        )


def test_cuda_update_norm_detects_a_missing_continuation_step():
    before = {"weight": torch.zeros(2)}
    after = {"weight": torch.tensor([3.0, 4.0])}

    assert _floating_update_norm(before, after) == 5.0
    assert _floating_update_norm(before, before) == 0.0


@pytest.mark.gpu
@pytest.mark.upstream
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_amp_mid_epoch_resume_is_finite_and_continuous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    assets = _real_sel_assets()

    (
        full_model,
        full_manifest,
        full_reference,
        full_optimizer,
        full_scaler,
        full_sampler,
        full_generator,
        full_loader,
    ) = _new_real_cuda_training_objects(assets)
    _seed_real_cuda_run()
    full_iterator, full_base = create_dataloader_iterator(full_loader, full_generator)
    (
        full_iterator,
        full_accumulator,
        full_global_step,
        full_ids,
        full_attempts,
        full_metrics,
    ) = _run_real_cuda_committed_steps(
        iterator=full_iterator,
        iterator_base=full_base,
        dataloader=full_loader,
        generator=full_generator,
        count=2,
        model=full_model,
        optimizer=full_optimizer,
        scaler=full_scaler,
        reference=full_reference,
        manifest=full_manifest,
        sampler=full_sampler,
        accumulator=_zero_accumulator(),
        global_step=0,
    )
    torch.cuda.synchronize()
    full_rng = capture_rng_state(
        full_generator,
        dataloader_iterator_base_cpu=full_base,
    )
    full_scaler_state = _owned_cpu(full_scaler.state_dict())
    full_sampler_state = full_sampler.state_dict()
    del (
        full_iterator,
        full_loader,
        full_generator,
        full_sampler,
        full_scaler,
        full_optimizer,
        full_reference,
        full_model,
    )
    gc.collect()
    torch.cuda.empty_cache()

    (
        interrupted_model,
        interrupted_manifest,
        interrupted_reference,
        interrupted_optimizer,
        interrupted_scaler,
        interrupted_sampler,
        interrupted_generator,
        interrupted_loader,
    ) = _new_real_cuda_training_objects(assets)
    _seed_real_cuda_run()
    interrupted_iterator, interrupted_base = create_dataloader_iterator(
        interrupted_loader, interrupted_generator
    )
    (
        interrupted_iterator,
        interrupted_accumulator,
        interrupted_global_step,
        first_ids,
        first_attempts,
        interrupted_first_metrics,
    ) = _run_real_cuda_committed_steps(
        iterator=interrupted_iterator,
        iterator_base=interrupted_base,
        dataloader=interrupted_loader,
        generator=interrupted_generator,
        count=1,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        scaler=interrupted_scaler,
        reference=interrupted_reference,
        manifest=interrupted_manifest,
        sampler=interrupted_sampler,
        accumulator=_zero_accumulator(),
        global_step=0,
    )
    saved_rng = capture_rng_state(
        interrupted_generator,
        dataloader_iterator_base_cpu=interrupted_base,
    )
    checkpoint_mutable = _owned_cpu(
        extract_mutable_state(interrupted_model, interrupted_manifest)
    )
    checkpoint_optimizer_state = _owned_cpu(interrupted_optimizer.state_dict())
    checkpoint_scaler_state = _owned_cpu(interrupted_scaler.state_dict())
    checkpoint_sampler_state = interrupted_sampler.state_dict()
    checkpoint_accumulator = interrupted_accumulator
    checkpoint_global_step = interrupted_global_step
    checkpoint_optimizer_steps = _optimizer_step_values(checkpoint_optimizer_state)
    assert checkpoint_optimizer_steps
    assert all(step == 1.0 for step in checkpoint_optimizer_steps)
    assert first_ids == full_ids[:1]
    assert first_attempts == full_attempts[: len(first_attempts)]
    assert interrupted_first_metrics[0].optimizer_step_applied
    checkpoint_path = tmp_path / "real-ddv2-last.pt"
    save_training_checkpoint(
        checkpoint_path,
        mutable_state=extract_mutable_state(
            interrupted_model,
            interrupted_manifest,
        ),
        optimizer=interrupted_optimizer,
        scaler=interrupted_scaler,
        epoch=0,
        global_step=interrupted_global_step,
        best_micro_ade=math.inf,
        rng=saved_rng,
        sampler_state=interrupted_sampler.state_dict(),
        accumulator=interrupted_accumulator,
        completed_epoch_record_sha256=None,
        base_checkpoint_sha256=_REAL_CHECKPOINT_SHA256,
        manifest=interrupted_manifest,
        config={"checkpoint_interval_steps": 1},
    )
    post_save_rng = capture_rng_state(
        interrupted_generator,
        dataloader_iterator_base_cpu=interrupted_base,
    )
    _assert_rng_equal(post_save_rng, saved_rng)
    (
        interrupted_iterator,
        repeat_accumulator,
        repeat_global_step,
        repeat_second_ids,
        repeat_second_attempts,
        repeat_second_metrics,
    ) = _run_real_cuda_committed_steps(
        iterator=interrupted_iterator,
        iterator_base=interrupted_base,
        dataloader=interrupted_loader,
        generator=interrupted_generator,
        count=1,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        scaler=interrupted_scaler,
        reference=interrupted_reference,
        manifest=interrupted_manifest,
        sampler=interrupted_sampler,
        accumulator=interrupted_accumulator,
        global_step=interrupted_global_step,
    )
    torch.cuda.synchronize()
    repeat_rng = capture_rng_state(
        interrupted_generator,
        dataloader_iterator_base_cpu=interrupted_base,
    )
    repeat_mutable = _owned_cpu(
        extract_mutable_state(interrupted_model, interrupted_manifest)
    )
    repeat_optimizer_state = _owned_cpu(interrupted_optimizer.state_dict())
    _assert_finite_nested(repeat_mutable)
    _assert_finite_nested(repeat_optimizer_state)
    repeat_model_update_norm = _floating_update_norm(
        checkpoint_mutable,
        repeat_mutable,
    )
    repeat_optimizer_update_norm = _floating_update_norm(
        checkpoint_optimizer_state,
        repeat_optimizer_state,
        included_leaf_names=frozenset(("exp_avg", "exp_avg_sq")),
    )
    assert _optimizer_step_values(repeat_optimizer_state) == tuple(
        step + 1.0 for step in checkpoint_optimizer_steps
    )
    del repeat_mutable, repeat_optimizer_state
    assert first_ids + repeat_second_ids == full_ids
    assert first_attempts + repeat_second_attempts == full_attempts
    assert repeat_second_metrics[0].optimizer_step_applied
    second_grad_divergence = _relative_scalar_difference(
        full_metrics[1].grad_norm,
        repeat_second_metrics[0].grad_norm,
        atol=_CUDA_SCALAR_ATOL,
    )
    assert repeat_global_step == full_global_step == 2
    assert repeat_accumulator.epoch == full_accumulator.epoch == 0
    assert repeat_accumulator.completed_batches == 2
    assert repeat_accumulator.completed_samples == 2
    assert repeat_accumulator.grad_norm_count == 2
    assert repeat_accumulator == _advance_accumulator(
        checkpoint_accumulator,
        repeat_second_metrics[0],
        1,
    )
    _assert_rng_equal(repeat_rng, full_rng)
    del (
        interrupted_iterator,
        interrupted_loader,
        interrupted_generator,
        interrupted_sampler,
        interrupted_scaler,
        interrupted_optimizer,
        interrupted_reference,
        interrupted_model,
    )
    gc.collect()
    torch.cuda.empty_cache()

    (
        resumed_model,
        resumed_manifest,
        resumed_reference,
        resumed_optimizer,
        resumed_scaler,
        resumed_sampler,
        resumed_generator,
        resumed_loader,
    ) = _new_real_cuda_training_objects(assets)
    resume = load_training_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        sampler=resumed_sampler,
        manifest=resumed_manifest,
        expected_base_checkpoint_sha256=_REAL_CHECKPOINT_SHA256,
    )
    assert resume.epoch == 0
    assert resume.global_step == checkpoint_global_step == 1
    assert resume.accumulator == checkpoint_accumulator
    assert resume.sampler.committed_cursor == 1
    loaded_mutable = _owned_cpu(extract_mutable_state(resumed_model, resumed_manifest))
    loaded_optimizer_state = _owned_cpu(resumed_optimizer.state_dict())
    _assert_nested_equal(loaded_mutable, checkpoint_mutable)
    _assert_nested_equal(loaded_optimizer_state, checkpoint_optimizer_state)
    _assert_nested_equal(resumed_scaler.state_dict(), checkpoint_scaler_state)
    _assert_nested_equal(
        resumed_sampler.state_dict().__dict__,
        checkpoint_sampler_state.__dict__,
    )
    _assert_rng_equal(resume.rng, saved_rng)
    del (
        loaded_mutable,
        loaded_optimizer_state,
    )
    restore_rng_state(resume.rng, resumed_generator)
    resumed_iterator, resumed_base = create_dataloader_iterator(
        resumed_loader,
        resumed_generator,
        iterator_base_state=resume.rng.dataloader_iterator_base_cpu,
        current_generator_state=resume.rng.dataloader_cpu,
    )
    restored_rng = capture_rng_state(
        resumed_generator,
        dataloader_iterator_base_cpu=resumed_base,
    )
    _assert_rng_equal(restored_rng, saved_rng)
    (
        resumed_iterator,
        resumed_accumulator,
        resumed_global_step,
        remaining_ids,
        remaining_attempts,
        resumed_second_metrics,
    ) = _run_real_cuda_committed_steps(
        iterator=resumed_iterator,
        iterator_base=resumed_base,
        dataloader=resumed_loader,
        generator=resumed_generator,
        count=1,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        reference=resumed_reference,
        manifest=resumed_manifest,
        sampler=resumed_sampler,
        accumulator=resume.accumulator,
        global_step=resume.global_step,
    )
    torch.cuda.synchronize()
    resumed_rng = capture_rng_state(
        resumed_generator,
        dataloader_iterator_base_cpu=resumed_base,
    )

    assert first_ids + remaining_ids == full_ids
    assert first_attempts + remaining_attempts == full_attempts
    assert remaining_ids == repeat_second_ids
    assert remaining_attempts == repeat_second_attempts
    assert resumed_second_metrics[0].optimizer_step_applied
    for name in (
        "total_loss",
        "trajectory_loss",
        "preservation_loss",
    ):
        _assert_cuda_scalar_continuity(
            uninterrupted=getattr(full_metrics[1], name),
            repeated_uninterrupted=getattr(repeat_second_metrics[0], name),
            resumed=getattr(resumed_second_metrics[0], name),
            relative_floor=_CUDA_LOSS_RELATIVE_FLOOR,
            relative_ceiling=_CUDA_LOSS_RELATIVE_CEILING,
            label=f"second-step {name}",
        )
    _assert_cuda_scalar_continuity(
        uninterrupted=full_metrics[1].grad_norm,
        repeated_uninterrupted=repeat_second_metrics[0].grad_norm,
        resumed=resumed_second_metrics[0].grad_norm,
        relative_floor=_CUDA_GRAD_NORM_RELATIVE_FLOOR,
        relative_ceiling=_CUDA_GRAD_NORM_RELATIVE_CEILING,
        label="second-step grad_norm",
    )
    assert resumed_global_step == repeat_global_step == full_global_step == 2
    assert resumed_global_step == checkpoint_global_step + 1
    assert resumed_accumulator.epoch == full_accumulator.epoch == 0
    assert resumed_accumulator.completed_batches == 2
    assert resumed_accumulator.completed_samples == 2
    assert resumed_accumulator.grad_norm_count == 2
    assert resumed_accumulator == _advance_accumulator(
        checkpoint_accumulator,
        resumed_second_metrics[0],
        1,
    )
    resumed_mutable = _owned_cpu(extract_mutable_state(resumed_model, resumed_manifest))
    resumed_optimizer_state = _owned_cpu(resumed_optimizer.state_dict())
    _assert_finite_nested(resumed_mutable)
    _assert_finite_nested(resumed_optimizer_state)
    resumed_model_update_norm = _floating_update_norm(
        checkpoint_mutable,
        resumed_mutable,
    )
    resumed_optimizer_update_norm = _floating_update_norm(
        checkpoint_optimizer_state,
        resumed_optimizer_state,
        included_leaf_names=frozenset(("exp_avg", "exp_avg_sq")),
    )
    assert repeat_model_update_norm > 0.0
    assert resumed_model_update_norm > 0.0
    assert repeat_optimizer_update_norm > 0.0
    assert resumed_optimizer_update_norm > 0.0
    _assert_cuda_divergence_envelope(
        baseline_divergence=second_grad_divergence,
        resumed_divergence=_relative_scalar_difference(
            repeat_model_update_norm,
            resumed_model_update_norm,
            atol=_CUDA_UPDATE_NORM_ATOL,
        ),
        relative_floor=_CUDA_MODEL_UPDATE_RELATIVE_FLOOR,
        relative_ceiling=_CUDA_MODEL_UPDATE_RELATIVE_CEILING,
        label="mutable-state update",
    )
    _assert_cuda_divergence_envelope(
        baseline_divergence=second_grad_divergence,
        resumed_divergence=_relative_scalar_difference(
            repeat_optimizer_update_norm,
            resumed_optimizer_update_norm,
            atol=_CUDA_UPDATE_NORM_ATOL,
        ),
        relative_floor=_CUDA_OPTIMIZER_UPDATE_RELATIVE_FLOOR,
        relative_ceiling=_CUDA_OPTIMIZER_UPDATE_RELATIVE_CEILING,
        label="Adam moment update",
    )
    assert _optimizer_step_values(resumed_optimizer_state) == tuple(
        step + 1.0 for step in checkpoint_optimizer_steps
    )
    _assert_nested_equal(
        resumed_optimizer_state["param_groups"],
        checkpoint_optimizer_state["param_groups"],
    )
    _assert_nested_equal(resumed_scaler.state_dict(), full_scaler_state)
    resumed_sampler_state = resumed_sampler.state_dict()
    assert resumed_sampler_state.epoch == 0
    assert resumed_sampler_state.committed_cursor == 2
    _assert_nested_equal(
        resumed_sampler_state.__dict__,
        full_sampler_state.__dict__,
    )
    _assert_rng_equal(resumed_rng, full_rng)


def test_train_one_epoch_retries_overflow_without_advancing_committed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from morai_ddv2.training import trainer as trainer_module

    torch.manual_seed(801)
    model = _ResumeModel()
    manifest = configure_training_boundary(model)
    reference = snapshot_source_parameters(model, manifest)
    optimizer = build_optimizer(model, manifest, lr=1.0e-3, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    dataset = _ResumeDataset(size=4)
    sampler = StatefulPermutationSampler(dataset, seed=802)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(803)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    expected_batches = []
    probe_sampler = StatefulPermutationSampler(dataset, seed=802)
    probe_indices = list(probe_sampler)
    expected_batches.append(
        ["sample-{:02d}".format(index) for index in probe_indices[:2]]
    )
    expected_batches.append(
        ["sample-{:02d}".format(index) for index in probe_indices[2:]]
    )
    calls = []

    def fake_step(_model, batch, *_args, **_kwargs):
        calls.append(list(batch["metadata"]["sample_id"]))
        applied = len(calls) != 1
        value = float(len(calls))
        return StepMetrics(value, value + 1.0, value + 2.0, value + 3.0, applied)

    monkeypatch.setattr(trainer_module, "train_step", fake_step)
    model.eval()
    accumulator = _zero_accumulator()

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        sampler=sampler,
        optimizer=optimizer,
        scaler=scaler,
        reference=reference,
        manifest=manifest,
        amp=False,
        grad_clip_norm=1.0,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
        epoch=0,
        global_step=10,
        accumulator=accumulator,
    )

    assert calls == [expected_batches[0], expected_batches[0], expected_batches[1]]
    assert metrics.epoch == 0
    assert metrics.global_step == 12
    assert metrics.completed_batches == 2
    assert metrics.completed_samples == 4
    assert metrics.mean_total_loss == pytest.approx((2.0 * 2 + 3.0 * 2) / 4)
    assert metrics.mean_trajectory_loss == pytest.approx((3.0 * 2 + 4.0 * 2) / 4)
    assert metrics.mean_preservation_loss == pytest.approx((4.0 * 2 + 5.0 * 2) / 4)
    assert sampler.state_dict().committed_cursor == 4
    assert sampler.state_dict().epoch == 0
    assert accumulator == _zero_accumulator()
    assert not model.training
    assert model._backbone.image_encoder.training


def test_train_one_epoch_moves_cpu_batch_to_model_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from morai_ddv2.training import trainer as trainer_module

    model = _ResumeModel().to(device="meta")
    manifest = configure_training_boundary(model)
    dataset = _ResumeDataset(size=4)
    sampler = StatefulPermutationSampler(dataset, seed=821)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(822)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    observed_sample_ids = []

    def fake_step(_model, batch, *_args, **_kwargs):
        tensors = (*batch["features"].values(), batch["trajectory"])
        assert all(tensor.device.type == "meta" for tensor in tensors)
        assert all(tensor.dtype == torch.float32 for tensor in tensors)
        observed_sample_ids.extend(batch["metadata"]["sample_id"])
        return StepMetrics(1.0, 2.0, 3.0, 4.0, True)

    monkeypatch.setattr(trainer_module, "train_step", fake_step)

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        sampler=sampler,
        optimizer=SimpleNamespace(),
        scaler=SimpleNamespace(),
        reference={},
        manifest=manifest,
        amp=False,
        grad_clip_norm=1.0,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
        epoch=0,
        global_step=5,
        accumulator=_zero_accumulator(),
    )

    assert sorted(observed_sample_ids) == [
        "sample-00",
        "sample-01",
        "sample-02",
        "sample-03",
    ]
    assert metrics.global_step == 7
    assert metrics.completed_batches == 2
    assert metrics.completed_samples == 4
    assert sampler.state_dict().committed_cursor == 4


def test_train_one_epoch_requires_saved_rng_for_committed_mid_epoch_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from morai_ddv2.training import trainer as trainer_module

    model = _ResumeModel()
    manifest = configure_training_boundary(model)
    reference = snapshot_source_parameters(model, manifest)
    optimizer = build_optimizer(model, manifest, lr=1.0e-3, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    dataset = _ResumeDataset(size=4)
    sampler = StatefulPermutationSampler(dataset, seed=851)
    sampler.mark_consumed(2)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(852)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    accumulator = EpochAccumulatorState(
        epoch=0,
        completed_batches=1,
        completed_samples=2,
        total_loss_numerator=2.0,
        trajectory_loss_numerator=4.0,
        preservation_loss_numerator=6.0,
        grad_norm_sum=4.0,
        grad_norm_count=1,
    )
    monkeypatch.setattr(
        trainer_module,
        "train_step",
        lambda *_args, **_kwargs: StepMetrics(1.0, 2.0, 3.0, 4.0, True),
    )

    with pytest.raises(ResumeContractError, match="resume RNG"):
        train_one_epoch(
            model=model,
            dataloader=dataloader,
            sampler=sampler,
            optimizer=optimizer,
            scaler=scaler,
            reference=reference,
            manifest=manifest,
            amp=False,
            grad_clip_norm=1.0,
            trajectory_weight=12.0,
            preservation_weight=1.0e-3,
            epoch=0,
            global_step=1,
            accumulator=accumulator,
        )


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
def test_train_one_epoch_restores_saved_rng_before_rebuilding_resume_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from morai_ddv2.training import trainer as trainer_module

    model = _ResumeModel()
    manifest = configure_training_boundary(model)
    reference = snapshot_source_parameters(model, manifest)
    optimizer = build_optimizer(model, manifest, lr=1.0e-3, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    dataset = _ResumeDataset(size=4)
    sampler = StatefulPermutationSampler(dataset, seed=901)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(902)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    original_iterator, iterator_base = create_dataloader_iterator(dataloader, generator)
    saved_rng = capture_rng_state(
        generator,
        dataloader_iterator_base_cpu=iterator_base,
    )
    saved_current = saved_rng.dataloader_cpu.clone()
    expected_draws = [
        (random.random(), float(np.random.random_sample()), float(torch.rand(())))
        for _ in range(2)
    ]
    del original_iterator
    random.seed(911)
    np.random.seed(912)
    torch.manual_seed(913)
    generator.manual_seed(914)
    observed_draws = []

    def fake_step(_model, _batch, *_args, **_kwargs):
        observed_draws.append(
            (random.random(), float(np.random.random_sample()), float(torch.rand(())))
        )
        return StepMetrics(1.0, 2.0, 3.0, 4.0, True)

    monkeypatch.setattr(trainer_module, "train_step", fake_step)

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        sampler=sampler,
        optimizer=optimizer,
        scaler=scaler,
        reference=reference,
        manifest=manifest,
        amp=False,
        grad_clip_norm=1.0,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
        epoch=0,
        global_step=0,
        accumulator=_zero_accumulator(),
        resume_rng=saved_rng,
    )

    assert observed_draws == expected_draws
    assert torch.equal(generator.get_state(), saved_current)
    assert metrics.global_step == 2
    assert sampler.state_dict().committed_cursor == 4


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
def test_train_one_epoch_atomically_checkpoints_only_committed_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from morai_ddv2.training import trainer as trainer_module

    model = _ResumeModel()
    manifest = configure_training_boundary(model)
    reference = snapshot_source_parameters(model, manifest)
    optimizer = build_optimizer(model, manifest, lr=1.0e-3, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    dataset = _ResumeDataset(size=6)
    sampler = StatefulPermutationSampler(dataset, seed=921)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(922)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    expected_iterator_base = generator.get_state().clone()
    calls = []

    def fake_step(_model, _batch, *_args, **_kwargs):
        value = float(len(calls) + 1)
        calls.append(value)
        return StepMetrics(value, value + 1.0, value + 2.0, value + 3.0, True)

    monkeypatch.setattr(trainer_module, "train_step", fake_step)
    checkpoint_path = tmp_path / "last.pt"

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        sampler=sampler,
        optimizer=optimizer,
        scaler=scaler,
        reference=reference,
        manifest=manifest,
        amp=False,
        grad_clip_norm=1.0,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
        epoch=0,
        global_step=0,
        accumulator=_zero_accumulator(),
        checkpoint_path=checkpoint_path,
        checkpoint_interval_steps=2,
        best_micro_ade=9.0,
        completed_epoch_record_sha256=None,
        base_checkpoint_sha256="e" * 64,
        config={"checkpoint_interval_steps": 2},
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["global_step"] == 2
    assert payload["sampler_state"].committed_cursor == 4
    assert payload["accumulator"].completed_batches == 2
    assert payload["accumulator"].completed_samples == 4
    assert payload["accumulator"].total_loss_numerator == 6.0
    assert payload["accumulator"].trajectory_loss_numerator == 10.0
    assert payload["accumulator"].preservation_loss_numerator == 14.0
    assert payload["accumulator"].grad_norm_sum == 9.0
    assert payload["accumulator"].grad_norm_count == 2
    assert torch.equal(
        payload["rng"].dataloader_iterator_base_cpu,
        expected_iterator_base,
    )
    assert torch.equal(
        payload["rng"].dataloader_cpu,
        generator.get_state(),
    )
    assert not checkpoint_path.with_name("last.pt.partial").exists()
    assert metrics.global_step == 3
    assert sampler.state_dict().committed_cursor == 6
    assert sampler.state_dict().epoch == 0
