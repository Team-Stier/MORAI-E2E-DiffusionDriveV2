from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from morai_ddv2.training.contracts import (
    EpochMetrics,
    StepMetrics,
    TrainingBoundaryError,
)
from morai_ddv2.training.freeze import configure_training_boundary
from morai_ddv2.training.loss import (
    FinetuneLoss,
    compose_finetune_loss,
    snapshot_source_parameters,
)
from morai_ddv2.training.trainer import (
    restore_batchnorm_buffers,
    snapshot_batchnorm_buffers,
    train_step,
)


class _StepModel(nn.Module):
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


class _RecordingScaler:
    def __init__(self, events, *, applied: bool) -> None:
        self._events = events
        self._applied = applied
        self._scale = 16.0
        self._get_scale_calls = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        self._events.append("scale")

        def record_backward(gradient: torch.Tensor) -> torch.Tensor:
            self._events.append("backward")
            return gradient

        loss.register_hook(record_backward)
        return loss

    def unscale_(self, _optimizer) -> None:
        self._events.append("unscale")

    def get_scale(self) -> float:
        self._get_scale_calls += 1
        self._events.append(
            "get_scale_before" if self._get_scale_calls == 1 else "get_scale_after"
        )
        return self._scale

    def step(self, optimizer) -> None:
        self._events.append("scaler_step")
        if self._applied:
            optimizer.step()

    def update(self) -> None:
        self._events.append("scaler_update")
        if not self._applied:
            self._scale /= 2.0


def _batch() -> dict:
    return {
        "features": {"x": torch.tensor([[1.0, -1.0], [3.0, 2.0]])},
        "trajectory": torch.zeros(2, 2),
    }


def _install_step_recorders(
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    events,
    optimizer,
) -> tuple[list[int], dict[str, bool]]:
    from morai_ddv2.training import trainer as trainer_module

    active = {"autocast": False}
    real_snapshot = trainer_module.snapshot_batchnorm_buffers
    real_compose = compose_finetune_loss
    real_clip = torch.nn.utils.clip_grad_norm_
    real_zero_grad = optimizer.zero_grad
    real_optimizer_step = optimizer.step
    clipped_ids = []

    def recording_zero_grad(*, set_to_none: bool) -> None:
        assert set_to_none
        events.append("zero_grad")
        real_zero_grad(set_to_none=set_to_none)

    def recording_optimizer_step(*args, **kwargs) -> None:
        events.append("optimizer_step")
        real_optimizer_step(*args, **kwargs)

    def recording_snapshot(current_model, manifest):
        events.append("snapshot_bn")
        return real_snapshot(current_model, manifest)

    @contextmanager
    def recording_autocast(*, enabled: bool):
        assert not enabled
        events.append("autocast_enter")
        active["autocast"] = True
        try:
            yield
        finally:
            active["autocast"] = False
            events.append("autocast_exit")

    def recording_forward(current_model, features, target):
        assert active["autocast"]
        events.append("forward")
        output = current_model._backbone.image_encoder(features["x"])
        return SimpleNamespace(trajectory_loss=(output - target).square().mean())

    def recording_compose(trajectory_loss, *args, **kwargs):
        assert not active["autocast"]
        assert trajectory_loss.dtype is torch.float32
        events.append("compose_fp32")
        return real_compose(trajectory_loss, *args, **kwargs)

    def recording_clip(parameters, max_norm, *args, **kwargs):
        events.append("clip")
        parameter_list = list(parameters)
        clipped_ids.extend(id(parameter) for parameter in parameter_list)
        assert max_norm == 1.0
        return real_clip(parameter_list, max_norm, *args, **kwargs)

    optimizer.zero_grad = recording_zero_grad
    optimizer.step = recording_optimizer_step
    monkeypatch.setattr(
        trainer_module, "snapshot_batchnorm_buffers", recording_snapshot
    )
    monkeypatch.setattr(trainer_module, "_autocast", recording_autocast)
    monkeypatch.setattr(
        trainer_module, "forward_trajectory_supervised", recording_forward
    )
    monkeypatch.setattr(trainer_module, "compose_finetune_loss", recording_compose)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    return clipped_ids, active


def test_step_and_epoch_metrics_public_contracts_are_exact() -> None:
    assert [field.name for field in fields(StepMetrics)] == [
        "total_loss",
        "trajectory_loss",
        "preservation_loss",
        "grad_norm",
        "optimizer_step_applied",
    ]
    assert [field.name for field in fields(EpochMetrics)] == [
        "epoch",
        "global_step",
        "completed_batches",
        "completed_samples",
        "mean_total_loss",
        "mean_trajectory_loss",
        "mean_preservation_loss",
    ]


def test_train_step_uses_exact_order_and_clips_only_the_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    model = _StepModel()
    manifest = configure_training_boundary(model)
    reference = snapshot_source_parameters(model, manifest)
    optimizer = torch.optim.SGD(
        [
            dict(model.named_parameters())[name]
            for name in manifest.trainable_parameters
        ],
        lr=0.05,
    )
    events = []
    clipped_ids, _ = _install_step_recorders(monkeypatch, model, events, optimizer)
    scaler = _RecordingScaler(events, applied=True)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in manifest.frozen_parameters
    }
    allowed_before = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name in manifest.mutable_state_entries
    }

    metrics = train_step(
        model,
        _batch(),
        optimizer,
        scaler,
        reference,
        manifest,
        amp=False,
        grad_clip_norm=1.0,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
    )

    assert events == [
        "zero_grad",
        "snapshot_bn",
        "autocast_enter",
        "forward",
        "autocast_exit",
        "compose_fp32",
        "scale",
        "backward",
        "unscale",
        "clip",
        "get_scale_before",
        "scaler_step",
        "optimizer_step",
        "scaler_update",
        "get_scale_after",
    ]
    named_parameters = dict(model.named_parameters())
    assert clipped_ids == [
        id(named_parameters[name]) for name in manifest.trainable_parameters
    ]
    assert metrics.optimizer_step_applied
    assert all(
        math.isfinite(value)
        for value in (
            metrics.total_loss,
            metrics.trajectory_loss,
            metrics.preservation_loss,
            metrics.grad_norm,
        )
    )
    for name, before in frozen_before.items():
        assert torch.equal(named_parameters[name], before), name
    named_buffers = dict(model.named_buffers())
    assert any(
        not torch.equal(named_buffers[name], before)
        for name, before in allowed_before.items()
    )


def test_overflow_restores_allowed_batchnorm_and_returns_finite_skipped_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(41)
    model = _StepModel()
    manifest = configure_training_boundary(model)
    reference = snapshot_source_parameters(model, manifest)
    optimizer = torch.optim.SGD(
        [
            dict(model.named_parameters())[name]
            for name in manifest.trainable_parameters
        ],
        lr=0.05,
    )
    events = []
    _install_step_recorders(monkeypatch, model, events, optimizer)
    scaler = _RecordingScaler(events, applied=False)
    from morai_ddv2.training import trainer as trainer_module

    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda parameters, max_norm: torch.tensor(float("inf")),
    )
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    buffers_before = snapshot_batchnorm_buffers(model, manifest)

    metrics = train_step(
        model,
        _batch(),
        optimizer,
        scaler,
        reference,
        manifest,
        amp=False,
        grad_clip_norm=1.0,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
    )

    assert not metrics.optimizer_step_applied
    assert metrics.grad_norm == 0.0
    assert all(
        math.isfinite(value)
        for value in (
            metrics.total_loss,
            metrics.trajectory_loss,
            metrics.preservation_loss,
            metrics.grad_norm,
        )
    )
    assert "optimizer_step" not in events
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters_before[name]), name
    assert snapshot_batchnorm_buffers(model, manifest).keys() == buffers_before.keys()
    for name, buffer in snapshot_batchnorm_buffers(model, manifest).items():
        assert torch.equal(buffer, buffers_before[name]), name
    assert trainer_module.restore_batchnorm_buffers is restore_batchnorm_buffers


def test_batchnorm_restore_rejects_non_allowlisted_or_inexact_snapshot() -> None:
    model = _StepModel()
    manifest = configure_training_boundary(model)
    snapshot = dict(snapshot_batchnorm_buffers(model, manifest))
    first = next(iter(snapshot))

    missing = dict(snapshot)
    del missing[first]
    with pytest.raises(TrainingBoundaryError, match="BatchNorm snapshot"):
        restore_batchnorm_buffers(model, missing, manifest)

    extra = dict(snapshot)
    extra["frozen.running_mean"] = torch.zeros(1)
    with pytest.raises(TrainingBoundaryError, match="BatchNorm snapshot"):
        restore_batchnorm_buffers(model, extra, manifest)


def test_step_metrics_reject_nonfinite_applied_grad_norm() -> None:
    loss = FinetuneLoss(
        total=torch.tensor(3.0),
        trajectory=torch.tensor(2.0),
        preservation=torch.tensor(1.0),
    )

    with pytest.raises(TrainingBoundaryError, match="grad norm.*finite"):
        StepMetrics.from_tensors(
            loss,
            torch.tensor(float("nan")),
            optimizer_step_applied=True,
        )
