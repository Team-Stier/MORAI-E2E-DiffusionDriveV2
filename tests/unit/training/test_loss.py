from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

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


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_snapshot_rejects_nonfinite_source_parameters(value: float) -> None:
    model = _LossModel()
    manifest = _manifest(model)
    parameter = dict(model.named_parameters())[manifest.trainable_parameters[0]]
    parameter.data.fill_(value)

    with pytest.raises(TrainingBoundaryError, match="source parameter.*finite"):
        snapshot_source_parameters(model, manifest)


def test_snapshot_rejects_nonfinite_fp32_clone_after_conversion() -> None:
    model = _LossModel(dtype=torch.float64)
    manifest = _manifest(model)
    parameter = dict(model.named_parameters())[manifest.trainable_parameters[0]]
    parameter.data.fill_(torch.finfo(torch.float64).max)

    with pytest.raises(TrainingBoundaryError, match="source parameter.*finite"):
        snapshot_source_parameters(model, manifest)


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


@pytest.mark.parametrize("location", ["current", "reference"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_preservation_rejects_nonfinite_inputs(location: str, value: float) -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = dict(snapshot_source_parameters(model, manifest))
    first = manifest.trainable_parameters[0]
    if location == "current":
        dict(model.named_parameters())[first].data.fill_(value)
    else:
        reference[first].fill_(value)

    with pytest.raises(TrainingBoundaryError, match=f"{location}.*finite"):
        source_preservation_loss(model, reference, manifest)


def test_preservation_rejects_nonfinite_component_from_fp32_overflow() -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = snapshot_source_parameters(model, manifest)
    first = manifest.trainable_parameters[0]
    dict(model.named_parameters())[first].data.fill_(torch.finfo(torch.float32).max)

    with pytest.raises(TrainingBoundaryError, match="preservation loss.*finite"):
        source_preservation_loss(model, reference, manifest)


def test_preservation_finite_checks_use_constant_host_synchronizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = snapshot_source_parameters(model, manifest)
    original_item = torch.Tensor.item
    item_calls = 0

    def counted_item(tensor: torch.Tensor):
        nonlocal item_calls
        item_calls += 1
        return original_item(tensor)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    preservation = source_preservation_loss(model, reference, manifest)

    assert preservation.item() == 0.0
    assert item_calls - 1 == 3


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


def test_compose_requires_zero_dimensional_trajectory_loss() -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = snapshot_source_parameters(model, manifest)

    with pytest.raises(TrainingBoundaryError, match="zero-dimensional"):
        compose_finetune_loss(torch.ones(1), model, reference, manifest)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_compose_rejects_nonfinite_trajectory_component(value: float) -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = snapshot_source_parameters(model, manifest)

    with pytest.raises(TrainingBoundaryError, match="trajectory loss.*finite"):
        compose_finetune_loss(torch.tensor(value), model, reference, manifest)


def test_compose_rejects_nonfinite_total_from_fp32_overflow() -> None:
    model = _LossModel()
    manifest = _manifest(model)
    reference = snapshot_source_parameters(model, manifest)
    trajectory = torch.tensor(torch.finfo(torch.float32).max / 2.0)

    with pytest.raises(TrainingBoundaryError, match="total loss.*finite"):
        compose_finetune_loss(trajectory, model, reference, manifest)


def test_pinned_loss_computer_uses_raw_heading_l1_across_pi_boundary(
    repo_root: Path,
) -> None:
    checkout = repo_root / "third_party/DiffusionDriveV2"
    source = checkout / "navsim/agents/diffusiondrivev2/modules/multimodal_loss.py"
    if not source.is_file():
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        common_dir = Path(result.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (repo_root / common_dir).resolve()
        checkout = common_dir.parent / "third_party/DiffusionDriveV2"
        source = checkout / "navsim/agents/diffusiondrivev2/modules/multimodal_loss.py"
    assert source.is_file()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == "1cd12a1e155c34dcc471261835444c8d5587580b"
    code = """
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from navsim.agents.diffusiondrivev2.modules import multimodal_loss

source = Path(sys.argv[1]).resolve()
assert Path(multimodal_loss.__file__).resolve() == source
config = SimpleNamespace(trajectory_cls_weight=10.0, trajectory_reg_weight=8.0)
loss_computer = multimodal_loss.LossComputer(config)
assert loss_computer.cls_loss_weight == 10.0
assert loss_computer.reg_loss_weight == 8.0
predicted = torch.tensor([[[[0.0, 0.0, math.pi - 0.01]]]])
target = torch.tensor([[[0.0, 0.0, -math.pi + 0.01]]])
logits = torch.zeros((1, 1))
plan_anchor = torch.zeros((1, 1, 1, 2))
trajectory_loss = loss_computer(
    predicted, logits, {"trajectory": target}, plan_anchor
)
classification = 10.0 * multimodal_loss.py_sigmoid_focal_loss(
    logits,
    torch.ones_like(logits),
    weight=None,
    gamma=2.0,
    alpha=0.25,
    reduction="mean",
    avg_factor=None,
)
observed_heading_l1 = (trajectory_loss - classification) * 3.0 / 8.0
print(json.dumps({
    "module_path": str(Path(multimodal_loss.__file__).resolve()),
    "observed_heading_l1": observed_heading_l1.item(),
}))
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(checkout), env.get("PYTHONPATH", "")))
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(source)],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)

    assert Path(observed["module_path"]) == source.resolve()
    assert observed["observed_heading_l1"] == pytest.approx(
        2.0 * math.pi - 0.02, abs=1.0e-6
    )
    assert observed["observed_heading_l1"] > 6.0


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
