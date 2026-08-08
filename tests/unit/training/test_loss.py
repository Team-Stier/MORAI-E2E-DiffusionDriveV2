from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from morai_ddv2.training.contracts import FreezeManifest, TrainingBoundaryError
from morai_ddv2.training.loss import (
    FinetuneLoss,
    assert_loss_contract,
    compose_finetune_loss,
    snapshot_source_parameters,
    source_preservation_loss,
)


class _ParameterBranch(nn.Module):
    def __init__(self, values: list[float], *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(values, dtype=dtype))


class _Backbone(nn.Module):
    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.image_encoder = _ParameterBranch([0.0], dtype=dtype)
        self.lidar_encoder = _ParameterBranch([0.0, 0.0, 0.0], dtype=dtype)


class _LossModel(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._backbone = _Backbone(dtype=dtype)
        self.frozen_head = _ParameterBranch([41.0, 42.0], dtype=dtype)


class _LossComputer(nn.Module):
    def __init__(
        self, cls_loss_weight: object = 10.0, reg_loss_weight: object = 8.0
    ) -> None:
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight


class _TrajectoryHead(nn.Module):
    def __init__(
        self, cls_loss_weight: object = 10.0, reg_loss_weight: object = 8.0
    ) -> None:
        super().__init__()
        self.loss_computer = _LossComputer(cls_loss_weight, reg_loss_weight)


def _manifest(model: nn.Module) -> FreezeManifest:
    trainable = tuple(
        name
        for name, _ in model.named_parameters()
        if name.startswith(("_backbone.image_encoder.", "_backbone.lidar_encoder."))
    )
    frozen = tuple(
        name for name, _ in model.named_parameters() if name not in trainable
    )
    return FreezeManifest(trainable, frozen, trainable)


def test_snapshot_contains_only_detached_fp32_device_clones() -> None:
    model = _LossModel(dtype=torch.float16)
    manifest = _manifest(model)
    state_before = tuple(model.state_dict())
    named_parameters = dict(model.named_parameters())

    reference = snapshot_source_parameters(model, manifest)

    assert tuple(reference) == manifest.trainable_parameters
    assert "frozen_head.weight" not in reference
    assert tuple(model.state_dict()) == state_before
    for name, baseline in reference.items():
        current = named_parameters[name]
        assert baseline.dtype is torch.float32
        assert baseline.device == current.device
        assert not baseline.requires_grad
        assert baseline.grad_fn is None
        assert baseline.data_ptr() != current.data_ptr()
        assert torch.equal(baseline, current.float())

    named_parameters[manifest.trainable_parameters[0]].data.fill_(9.0)
    assert torch.equal(reference[manifest.trainable_parameters[0]], torch.zeros(1))


def test_preservation_uses_one_global_scalar_count_for_unequal_tensors() -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = snapshot_source_parameters(model, manifest)
    named_parameters = dict(model.named_parameters())
    named_parameters[manifest.trainable_parameters[0]].data.fill_(2.0)
    named_parameters[manifest.trainable_parameters[1]].data.fill_(1.0)

    preservation = source_preservation_loss(model, reference, manifest)

    # (2**2 + 1**2 + 1**2 + 1**2) / (1 + 3) = 7/4.  Averaging
    # the two tensor means would incorrectly produce (4 + 1) / 2 = 5/2.
    assert preservation.dtype is torch.float32
    torch.testing.assert_close(preservation, torch.tensor(7.0 / 4.0), rtol=0, atol=0)
    assert preservation.item() != 5.0 / 2.0


def test_composite_is_exact_fp32_and_retains_the_backward_graph() -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = snapshot_source_parameters(model, manifest)
    named_parameters = dict(model.named_parameters())
    named_parameters[manifest.trainable_parameters[0]].data.fill_(2.0)
    named_parameters[manifest.trainable_parameters[1]].data.fill_(1.0)
    head = _TrajectoryHead()
    assert_loss_contract(head, trajectory_weight=12.0, preservation_weight=1.0e-3)
    classification = torch.tensor(0.25, dtype=torch.float16, requires_grad=True)
    regression = torch.tensor(0.5, dtype=torch.float16, requires_grad=True)
    trajectory = (
        head.loss_computer.cls_loss_weight * classification
        + head.loss_computer.reg_loss_weight * regression
    )

    loss = compose_finetune_loss(trajectory, model, reference, manifest)

    assert isinstance(loss, FinetuneLoss)
    assert loss.trajectory.dtype is torch.float32
    assert loss.preservation.dtype is torch.float32
    assert loss.total.dtype is torch.float32
    assert loss.total.requires_grad
    expected = trajectory.float() * 12.0 + torch.tensor(7.0 / 4.0) * 1.0e-3
    torch.testing.assert_close(loss.total, expected, rtol=0, atol=0)
    torch.testing.assert_close(loss.trajectory, trajectory.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        loss.preservation, torch.tensor(7.0 / 4.0), rtol=0, atol=0
    )

    loss.total.backward()

    assert classification.grad is not None
    assert regression.grad is not None
    assert named_parameters[manifest.trainable_parameters[0]].grad is not None
    assert classification.grad.item() == 120.0
    assert regression.grad.item() == 96.0


@pytest.mark.parametrize(
    ("cls_weight", "reg_weight", "trajectory_weight", "preservation_weight"),
    [
        (9.0, 8.0, 12.0, 1.0e-3),
        (10.0, 7.0, 12.0, 1.0e-3),
        (10.0, 8.0, 11.0, 1.0e-3),
        (10.0, 8.0, 12.0, 2.0e-3),
    ],
    ids=["classification", "regression", "trajectory", "preservation"],
)
def test_loss_contract_rejects_each_weight_mutation(
    cls_weight: float,
    reg_weight: float,
    trajectory_weight: float,
    preservation_weight: float,
) -> None:
    head = _TrajectoryHead(cls_weight, reg_weight)

    with pytest.raises(TrainingBoundaryError, match="loss weight"):
        assert_loss_contract(
            head,
            trajectory_weight=trajectory_weight,
            preservation_weight=preservation_weight,
        )


@pytest.mark.parametrize(
    "head",
    [nn.Identity(), nn.Module(), _TrajectoryHead(torch.tensor(10.0), 8.0)],
    ids=["missing-loss-computer", "plain-module", "tensor-weight"],
)
def test_loss_contract_rejects_missing_or_nonscalar_upstream_contract(
    head: nn.Module,
) -> None:
    with pytest.raises(TrainingBoundaryError, match="loss"):
        assert_loss_contract(head, trajectory_weight=12.0, preservation_weight=1.0e-3)


def test_compose_rejects_mutated_outer_weights_even_without_preflight() -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = snapshot_source_parameters(model, manifest)

    with pytest.raises(TrainingBoundaryError, match="trajectory loss weight"):
        compose_finetune_loss(
            torch.ones((), requires_grad=True),
            model,
            reference,
            manifest,
            trajectory_weight=11.0,
        )
    with pytest.raises(TrainingBoundaryError, match="preservation loss weight"):
        compose_finetune_loss(
            torch.ones((), requires_grad=True),
            model,
            reference,
            manifest,
            preservation_weight=2.0e-3,
        )


def test_heading_regression_is_raw_l1_across_pi_boundary() -> None:
    predicted = torch.tensor(math.pi - 0.01)
    target = torch.tensor(-math.pi + 0.01)

    raw_heading_l1 = F.l1_loss(predicted, target)

    assert raw_heading_l1.item() == pytest.approx(2.0 * math.pi - 0.02)
    assert raw_heading_l1.item() > 6.0


@pytest.mark.parametrize("operation", ["snapshot", "preservation"])
def test_empty_allowlist_is_rejected(operation: str) -> None:
    model = _LossModel()
    manifest = FreezeManifest(
        trainable_parameters=(),
        frozen_parameters=tuple(dict(model.named_parameters())),
        mutable_state_entries=(),
    )

    with pytest.raises(TrainingBoundaryError, match="no trainable parameters"):
        if operation == "snapshot":
            snapshot_source_parameters(model, manifest)
        else:
            source_preservation_loss(model, {}, manifest)


def test_unknown_manifest_parameter_is_a_boundary_error() -> None:
    model = _LossModel()
    manifest = _manifest(model)
    manifest = replace(
        manifest,
        trainable_parameters=manifest.trainable_parameters
        + ("_backbone.image_encoder.missing",),
    )

    with pytest.raises(TrainingBoundaryError, match="unknown trainable parameter"):
        snapshot_source_parameters(model, manifest)


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "attached"])
def test_preservation_rejects_invalid_reference_mapping(mutation: str) -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = dict(snapshot_source_parameters(model, manifest))
    first = manifest.trainable_parameters[0]
    if mutation == "missing":
        del reference[first]
    elif mutation == "extra":
        reference["frozen_head.weight"] = torch.zeros(2)
    elif mutation == "dtype":
        reference[first] = reference[first].double()
    else:
        reference[first] = reference[first].requires_grad_()

    with pytest.raises(TrainingBoundaryError, match="source reference"):
        source_preservation_loss(model, reference, manifest)
