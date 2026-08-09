from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, replace
import io
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from morai_ddv2.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from morai_ddv2.training.contracts import (
    EpochAccumulatorState,
    ResumeState,
    RngState,
    TrainingBoundaryError,
)
from morai_ddv2.training.freeze import (
    build_optimizer,
    configure_training_boundary,
    extract_mutable_state,
)
from morai_ddv2.training.rng import (
    capture_rng_state,
    create_dataloader_iterator,
    restore_rng_state,
)
from morai_ddv2.training.sampler import (
    ResumeContractError,
    StatefulPermutationSampler,
)


class _CheckpointModel(nn.Module):
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
        self.frozen_head = nn.Linear(2, 1)
        self.register_buffer("frozen_buffer", torch.tensor([13.0]))


class _StatefulScaler:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def state_dict(self):
        return {"scale": self.scale, "growth_tracker": 7}

    def load_state_dict(self, state) -> None:
        self.scale = float(state["scale"])


class _FailOnceScaler(_StatefulScaler):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self._fail_next_load = True

    def load_state_dict(self, state) -> None:
        super().load_state_dict(state)
        if self._fail_next_load:
            self._fail_next_load = False
            raise RuntimeError("synthetic scaler restore failure")


def _accumulator(epoch: int = 0, samples: int = 2) -> EpochAccumulatorState:
    return EpochAccumulatorState(
        epoch=epoch,
        completed_batches=1 if samples else 0,
        completed_samples=samples,
        total_loss_numerator=3.0 if samples else 0.0,
        trajectory_loss_numerator=2.0 if samples else 0.0,
        preservation_loss_numerator=1.0 if samples else 0.0,
        grad_norm_sum=0.5 if samples else 0.0,
        grad_norm_count=1 if samples else 0,
    )


def _assert_numpy_state_equal(left, right) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


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


def test_rng_and_resume_public_contracts_are_frozen_and_exact() -> None:
    assert [field.name for field in fields(RngState)] == [
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "dataloader_cpu",
        "dataloader_iterator_base_cpu",
    ]
    assert [field.name for field in fields(EpochAccumulatorState)] == [
        "epoch",
        "completed_batches",
        "completed_samples",
        "total_loss_numerator",
        "trajectory_loss_numerator",
        "preservation_loss_numerator",
        "grad_norm_sum",
        "grad_norm_count",
    ]
    assert [field.name for field in fields(ResumeState)] == [
        "epoch",
        "global_step",
        "best_micro_ade",
        "sampler",
        "accumulator",
        "completed_epoch_record_sha256",
        "rng",
    ]


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
def test_capture_restore_owns_and_restores_every_rng_stream() -> None:
    random.seed(310)
    np.random.seed(311)
    torch.manual_seed(312)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(313)
    dataloader_generator = torch.Generator(device="cpu")
    dataloader_generator.manual_seed(314)
    iterator_base = dataloader_generator.get_state().clone()

    state = capture_rng_state(
        dataloader_generator,
        dataloader_iterator_base_cpu=iterator_base,
    )
    cuda_count = len(state.torch_cuda)
    expected_python = random.random()
    expected_numpy = np.random.random_sample()
    expected_cpu = torch.rand(4)
    expected_dataloader = torch.rand(4, generator=dataloader_generator)
    expected_cuda = tuple(
        torch.rand(4, device=torch.device("cuda", index)).cpu()
        for index in range(cuda_count)
    )

    random.seed(410)
    np.random.seed(411)
    torch.manual_seed(412)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(413)
    dataloader_generator.manual_seed(414)
    iterator_base.zero_()

    restore_rng_state(state, dataloader_generator)

    assert random.random() == expected_python
    assert np.random.random_sample() == expected_numpy
    assert torch.equal(torch.rand(4), expected_cpu)
    assert torch.equal(
        torch.rand(4, generator=dataloader_generator), expected_dataloader
    )
    observed_cuda = tuple(
        torch.rand(4, device=torch.device("cuda", index)).cpu()
        for index in range(cuda_count)
    )
    assert len(observed_cuda) == len(expected_cuda)
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(observed_cuda, expected_cuda)
    )
    assert torch.equal(
        state.dataloader_iterator_base_cpu,
        capture_rng_state(
            dataloader_generator,
            dataloader_iterator_base_cpu=state.dataloader_iterator_base_cpu,
        ).dataloader_iterator_base_cpu,
    )
    assert state.torch_cuda and torch.cuda.is_available() or not state.torch_cuda


def test_create_iterator_preserves_original_worker_seed_draw_on_rebuild() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260806)
    loader = DataLoader(
        TensorDataset(torch.arange(8)),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        generator=generator,
    )

    original_iterator, base = create_dataloader_iterator(loader, generator)
    current = generator.get_state().clone()
    original_batches = [batch[0].clone() for batch in original_iterator]
    generator.manual_seed(9)

    replacement, replacement_base = create_dataloader_iterator(
        loader,
        generator,
        iterator_base_state=base,
        current_generator_state=current,
    )

    assert torch.equal(replacement_base, base)
    assert replacement_base.data_ptr() != base.data_ptr()
    assert torch.equal(generator.get_state(), current)
    assert [batch[0].tolist() for batch in replacement] == [
        batch.tolist() for batch in original_batches
    ]
    assert torch.equal(generator.get_state(), current)


@pytest.mark.parametrize(
    ("iterator_base", "current"),
    [
        (torch.zeros(1, dtype=torch.uint8), None),
        (None, torch.zeros(1, dtype=torch.uint8)),
    ],
)
def test_create_iterator_requires_both_resume_states(iterator_base, current) -> None:
    generator = torch.Generator(device="cpu")
    loader = DataLoader(TensorDataset(torch.arange(2)), generator=generator)

    with pytest.raises(ResumeContractError, match="iterator.*current"):
        create_dataloader_iterator(
            loader,
            generator,
            iterator_base_state=iterator_base,
            current_generator_state=current,
        )


def test_training_checkpoint_is_atomic_minimal_and_exactly_restorable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(101)
    model = _CheckpointModel()
    manifest = configure_training_boundary(model)
    optimizer = build_optimizer(model, manifest, lr=1.0e-2, weight_decay=1.0e-4)
    for parameter in (
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model._backbone.image_encoder.norm.running_mean.fill_(2.5)
    scaler = _StatefulScaler(128.0)
    sampler = StatefulPermutationSampler(list(range(5)), seed=88)
    sampler.mark_consumed(2)
    dataloader_generator = torch.Generator(device="cpu")
    dataloader_generator.manual_seed(89)
    rng = capture_rng_state(
        dataloader_generator,
        dataloader_iterator_base_cpu=dataloader_generator.get_state(),
    )
    accumulator = _accumulator()
    mutable = extract_mutable_state(model, manifest)
    expected_mutable = OrderedDict(
        (name, value.clone()) for name, value in mutable.items()
    )
    expected_optimizer = optimizer.state_dict()
    checkpoint_path = tmp_path / "last.pt"
    atomic_calls = []
    from morai_ddv2.training import checkpoint as checkpoint_module

    real_atomic_write = checkpoint_module.atomic_write_bytes

    def recording_atomic_write(path, payload) -> None:
        atomic_calls.append((Path(path), bytes(payload)))
        real_atomic_write(path, payload)

    monkeypatch.setattr(checkpoint_module, "atomic_write_bytes", recording_atomic_write)

    save_training_checkpoint(
        checkpoint_path,
        mutable_state=mutable,
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        global_step=1,
        best_micro_ade=math.inf,
        rng=rng,
        sampler_state=sampler.state_dict(),
        accumulator=accumulator,
        completed_epoch_record_sha256=None,
        base_checkpoint_sha256="a" * 64,
        manifest=manifest,
        config={"seed": 20260806, "nested": {"amp": False}},
    )

    assert len(atomic_calls) == 1
    assert atomic_calls[0][0] == checkpoint_path
    assert checkpoint_path.is_file()
    assert not checkpoint_path.with_name("last.pt.partial").exists()
    payload = torch.load(io.BytesIO(atomic_calls[0][1]), map_location="cpu")
    assert set(payload) == {
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
    }
    assert tuple(payload["mutable_state"]) == manifest.mutable_state_entries
    assert not set(payload["mutable_state"]).intersection(manifest.frozen_parameters)
    assert all(
        tensor.device.type == "cpu" for tensor in payload["mutable_state"].values()
    )
    assert payload["manifest"] == manifest
    assert payload["config"] == {"seed": 20260806, "nested": {"amp": False}}

    torch.manual_seed(101)
    restored_model = _CheckpointModel()
    restored_manifest = configure_training_boundary(restored_model)
    restored_optimizer = build_optimizer(
        restored_model, restored_manifest, lr=1.0e-2, weight_decay=1.0e-4
    )
    restored_scaler = _StatefulScaler(1.0)
    restored_sampler = StatefulPermutationSampler(list(range(5)), seed=999)

    resume = load_training_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
        sampler=restored_sampler,
        manifest=restored_manifest,
        expected_base_checkpoint_sha256="a" * 64,
    )

    assert resume == ResumeState(
        epoch=0,
        global_step=1,
        best_micro_ade=math.inf,
        sampler=resume.sampler,
        accumulator=accumulator,
        completed_epoch_record_sha256=None,
        rng=resume.rng,
    )
    for name, tensor in extract_mutable_state(
        restored_model, restored_manifest
    ).items():
        assert torch.equal(tensor, expected_mutable[name])
    assert torch.equal(restored_model.frozen_buffer, torch.tensor([13.0]))
    _assert_nested_equal(restored_optimizer.state_dict(), expected_optimizer)
    assert restored_scaler.state_dict()["scale"] == 128.0
    assert restored_sampler.state_dict().committed_cursor == 2
    assert torch.equal(
        restored_sampler.state_dict().permutation, sampler.state_dict().permutation
    )
    assert torch.equal(resume.rng.torch_cpu, rng.torch_cpu)
    _assert_numpy_state_equal(resume.rng.numpy, rng.numpy)


def test_checkpoint_rejects_identity_and_progress_mismatch_before_model_mutation(
    tmp_path: Path,
) -> None:
    torch.manual_seed(505)
    model = _CheckpointModel()
    manifest = configure_training_boundary(model)
    optimizer = build_optimizer(model, manifest, lr=1.0e-2, weight_decay=0.0)
    scaler = _StatefulScaler(1.0)
    sampler = StatefulPermutationSampler(list(range(5)), seed=5)
    sampler.mark_consumed(2)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6)
    checkpoint_path = tmp_path / "last.pt"
    save_training_checkpoint(
        checkpoint_path,
        mutable_state=extract_mutable_state(model, manifest),
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        global_step=1,
        best_micro_ade=2.0,
        rng=capture_rng_state(
            generator, dataloader_iterator_base_cpu=generator.get_state()
        ),
        sampler_state=sampler.state_dict(),
        accumulator=_accumulator(),
        completed_epoch_record_sha256=None,
        base_checkpoint_sha256="b" * 64,
        manifest=manifest,
        config={},
    )

    before = extract_mutable_state(model, manifest)
    with pytest.raises(TrainingBoundaryError, match="base checkpoint"):
        load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            sampler=sampler,
            manifest=manifest,
            expected_base_checkpoint_sha256="c" * 64,
        )
    after = extract_mutable_state(model, manifest)
    assert all(torch.equal(before[name], after[name]) for name in before)

    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["accumulator"] = replace(payload["accumulator"], completed_samples=1)
    torch.save(payload, checkpoint_path)
    with pytest.raises(ResumeContractError, match="accumulator.*sampler"):
        load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            sampler=sampler,
            manifest=manifest,
            expected_base_checkpoint_sha256="b" * 64,
        )


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
def test_checkpoint_load_rolls_back_every_component_when_installation_fails(
    tmp_path: Path,
) -> None:
    torch.manual_seed(1001)
    source_model = _CheckpointModel()
    source_manifest = configure_training_boundary(source_model)
    source_optimizer = build_optimizer(
        source_model, source_manifest, lr=1.0e-2, weight_decay=0.0
    )
    for parameter in (
        parameter
        for group in source_optimizer.param_groups
        for parameter in group["params"]
    ):
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()
    source_model._backbone.image_encoder.norm.running_mean.fill_(5.0)
    source_scaler = _StatefulScaler(128.0)
    source_sampler = StatefulPermutationSampler(list(range(5)), seed=1002)
    source_sampler.mark_consumed(2)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1003)
    checkpoint_path = tmp_path / "last.pt"
    save_training_checkpoint(
        checkpoint_path,
        mutable_state=extract_mutable_state(source_model, source_manifest),
        optimizer=source_optimizer,
        scaler=source_scaler,
        epoch=0,
        global_step=1,
        best_micro_ade=3.0,
        rng=capture_rng_state(
            generator,
            dataloader_iterator_base_cpu=generator.get_state(),
        ),
        sampler_state=source_sampler.state_dict(),
        accumulator=_accumulator(),
        completed_epoch_record_sha256=None,
        base_checkpoint_sha256="f" * 64,
        manifest=source_manifest,
        config={},
    )

    torch.manual_seed(1001)
    target_model = _CheckpointModel()
    target_manifest = configure_training_boundary(target_model)
    target_optimizer = build_optimizer(
        target_model, target_manifest, lr=1.0e-2, weight_decay=0.0
    )
    target_scaler = _FailOnceScaler(4.0)
    target_sampler = StatefulPermutationSampler(list(range(5)), seed=1004)
    mutable_before = extract_mutable_state(target_model, target_manifest)
    optimizer_before = target_optimizer.state_dict()
    scaler_before = target_scaler.state_dict()
    sampler_before = target_sampler.state_dict()

    with pytest.raises(RuntimeError, match="synthetic scaler restore failure"):
        load_training_checkpoint(
            checkpoint_path,
            model=target_model,
            optimizer=target_optimizer,
            scaler=target_scaler,
            sampler=target_sampler,
            manifest=target_manifest,
            expected_base_checkpoint_sha256="f" * 64,
        )

    mutable_after = extract_mutable_state(target_model, target_manifest)
    for name, tensor in mutable_before.items():
        assert torch.equal(mutable_after[name], tensor), name
    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(target_scaler.state_dict(), scaler_before)
    assert target_sampler.state_dict().epoch == sampler_before.epoch
    assert (
        target_sampler.state_dict().committed_cursor == sampler_before.committed_cursor
    )
    assert torch.equal(
        target_sampler.state_dict().permutation,
        sampler_before.permutation,
    )
    assert torch.equal(
        target_sampler.state_dict().generator_state,
        sampler_before.generator_state,
    )
