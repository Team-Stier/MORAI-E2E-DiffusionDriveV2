from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .loss import FinetuneLoss
    from .sampler import SamplerState


TRAINABLE_PREFIXES = (
    "_backbone.image_encoder.",
    "_backbone.lidar_encoder.",
    "_backbone.transformers.",
    "_backbone.lidar_channel_to_img.",
    "_backbone.img_channel_to_lidar.",
)


class TrainingBoundaryError(RuntimeError):
    """Raised when the DDV2 trainable-state boundary has drifted."""


@dataclass(frozen=True)
class FreezeManifest:
    trainable_parameters: tuple[str, ...]
    frozen_parameters: tuple[str, ...]
    mutable_state_entries: tuple[str, ...]


@dataclass(frozen=True)
class RngState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]
    dataloader_cpu: torch.Tensor
    dataloader_iterator_base_cpu: torch.Tensor


@dataclass(frozen=True)
class EpochAccumulatorState:
    epoch: int
    completed_batches: int
    completed_samples: int
    total_loss_numerator: float
    trajectory_loss_numerator: float
    preservation_loss_numerator: float
    grad_norm_sum: float
    grad_norm_count: int


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    best_micro_ade: float
    sampler: SamplerState
    accumulator: EpochAccumulatorState
    completed_epoch_record_sha256: Optional[str]
    rng: RngState


@dataclass(frozen=True)
class StepMetrics:
    total_loss: float
    trajectory_loss: float
    preservation_loss: float
    grad_norm: float
    optimizer_step_applied: bool

    @classmethod
    def from_tensors(
        cls,
        loss: FinetuneLoss,
        grad_norm: torch.Tensor,
        *,
        optimizer_step_applied: bool,
    ) -> StepMetrics:
        if type(optimizer_step_applied) is not bool:
            raise TrainingBoundaryError("optimizer step application must be a bool")

        def finite_scalar(tensor: torch.Tensor, label: str) -> float:
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 0:
                raise TrainingBoundaryError(f"{label} must be a scalar tensor")
            value = float(tensor.detach().item())
            if not math.isfinite(value):
                raise TrainingBoundaryError(f"{label} must be finite")
            return value

        total = finite_scalar(loss.total, "total loss")
        trajectory = finite_scalar(loss.trajectory, "trajectory loss")
        preservation = finite_scalar(loss.preservation, "preservation loss")
        if not isinstance(grad_norm, torch.Tensor) or grad_norm.ndim != 0:
            raise TrainingBoundaryError("grad norm must be a scalar tensor")
        grad_norm_value = float(grad_norm.detach().item())
        if not math.isfinite(grad_norm_value):
            if optimizer_step_applied:
                raise TrainingBoundaryError("grad norm must be finite")
            grad_norm_value = 0.0
        return cls(
            total,
            trajectory,
            preservation,
            grad_norm_value,
            optimizer_step_applied,
        )


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    global_step: int
    completed_batches: int
    completed_samples: int
    mean_total_loss: float
    mean_trajectory_loss: float
    mean_preservation_loss: float
