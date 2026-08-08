from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .contracts import FreezeManifest, TrainingBoundaryError


@dataclass(frozen=True)
class FinetuneLoss:
    total: torch.Tensor
    trajectory: torch.Tensor
    preservation: torch.Tensor


def _require_finite(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise TrainingBoundaryError(f"{label} must be finite")


def snapshot_source_parameters(
    model: nn.Module, manifest: FreezeManifest
) -> Mapping[str, torch.Tensor]:
    if not manifest.trainable_parameters:
        raise TrainingBoundaryError("no trainable parameters")
    if len(set(manifest.trainable_parameters)) != len(manifest.trainable_parameters):
        raise TrainingBoundaryError("duplicate trainable parameter in freeze manifest")

    named_parameters = dict(model.named_parameters())
    unknown = tuple(
        name for name in manifest.trainable_parameters if name not in named_parameters
    )
    if unknown:
        raise TrainingBoundaryError(f"unknown trainable parameter: {unknown}")

    reference = OrderedDict()
    for name in manifest.trainable_parameters:
        baseline = named_parameters[name].detach().to(dtype=torch.float32).clone()
        _require_finite(baseline, f"source parameter {name}")
        reference[name] = baseline
    return reference


def source_preservation_loss(
    model: nn.Module,
    reference: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
) -> torch.Tensor:
    if not manifest.trainable_parameters:
        raise TrainingBoundaryError("no trainable parameters")
    if len(set(manifest.trainable_parameters)) != len(manifest.trainable_parameters):
        raise TrainingBoundaryError("duplicate trainable parameter in freeze manifest")

    named_parameters = dict(model.named_parameters())
    unknown = tuple(
        name for name in manifest.trainable_parameters if name not in named_parameters
    )
    if unknown:
        raise TrainingBoundaryError(f"unknown trainable parameter: {unknown}")
    if set(reference) != set(manifest.trainable_parameters):
        raise TrainingBoundaryError(
            "source reference keys must exactly match trainable parameters"
        )

    first = named_parameters[manifest.trainable_parameters[0]]
    numerator = torch.zeros((), dtype=torch.float32, device=first.device)
    current_finite = torch.ones((), dtype=torch.bool, device=first.device)
    reference_finite = torch.ones((), dtype=torch.bool, device=first.device)
    denominator = 0
    for name in manifest.trainable_parameters:
        current = named_parameters[name]
        baseline = reference[name]
        if not isinstance(baseline, torch.Tensor):
            raise TrainingBoundaryError(f"source reference is not a tensor: {name}")
        if baseline.dtype is not torch.float32:
            raise TrainingBoundaryError(f"source reference must be FP32: {name}")
        if baseline.device != current.device:
            raise TrainingBoundaryError(
                f"source reference must remain on the model device: {name}"
            )
        if baseline.shape != current.shape:
            raise TrainingBoundaryError(f"source reference shape mismatch: {name}")
        if baseline.requires_grad or baseline.grad_fn is not None:
            raise TrainingBoundaryError(f"source reference must be detached: {name}")
        if current.device != numerator.device:
            raise TrainingBoundaryError("trainable parameters span multiple devices")
        current_finite = current_finite & torch.isfinite(current).all()
        reference_finite = reference_finite & torch.isfinite(baseline).all()
        numerator = numerator + (current.float() - baseline).square().sum()
        denominator += current.numel()

    if not current_finite.item():
        raise TrainingBoundaryError("current trainable parameters must be finite")
    if not reference_finite.item():
        raise TrainingBoundaryError("source reference parameters must be finite")
    if denominator == 0:
        raise TrainingBoundaryError("no trainable parameter scalars")
    preservation = numerator / denominator
    _require_finite(preservation, "preservation loss")
    return preservation


def compose_finetune_loss(
    trajectory_loss: torch.Tensor,
    model: nn.Module,
    reference: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
    *,
    trajectory_weight: float = 12.0,
    preservation_weight: float = 1.0e-3,
) -> FinetuneLoss:
    if type(trajectory_weight) not in (int, float) or trajectory_weight != 12.0:
        raise TrainingBoundaryError("trajectory loss weight must be exactly 12.0")
    if type(preservation_weight) not in (int, float) or preservation_weight != 1.0e-3:
        raise TrainingBoundaryError("preservation loss weight must be exactly 1.0e-3")
    if not isinstance(trajectory_loss, torch.Tensor):
        raise TrainingBoundaryError("trajectory loss must be a tensor")
    if trajectory_loss.ndim != 0:
        raise TrainingBoundaryError("trajectory loss must be zero-dimensional")

    trajectory = trajectory_loss.float()
    _require_finite(trajectory, "trajectory loss")
    preservation = source_preservation_loss(model, reference, manifest)
    if trajectory.device != preservation.device:
        raise TrainingBoundaryError(
            "trajectory and preservation losses must share one device"
        )
    total = 12.0 * trajectory + 1.0e-3 * preservation
    _require_finite(total, "total loss")
    return FinetuneLoss(total, trajectory, preservation)


def assert_loss_contract(
    head: nn.Module,
    *,
    trajectory_weight: float,
    preservation_weight: float,
) -> None:
    if type(trajectory_weight) not in (int, float) or trajectory_weight != 12.0:
        raise TrainingBoundaryError("trajectory loss weight must be exactly 12.0")
    if type(preservation_weight) not in (int, float) or preservation_weight != 1.0e-3:
        raise TrainingBoundaryError("preservation loss weight must be exactly 1.0e-3")

    loss_computer = getattr(head, "loss_computer", None)
    if loss_computer is None:
        raise TrainingBoundaryError("trajectory head loss_computer is missing")
    cls_weight = getattr(loss_computer, "cls_loss_weight", None)
    if type(cls_weight) not in (int, float) or cls_weight != 10.0:
        raise TrainingBoundaryError("classification loss weight must be exactly 10.0")
    reg_weight = getattr(loss_computer, "reg_loss_weight", None)
    if type(reg_weight) not in (int, float) or reg_weight != 8.0:
        raise TrainingBoundaryError("regression loss weight must be exactly 8.0")
