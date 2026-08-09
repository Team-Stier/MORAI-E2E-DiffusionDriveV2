from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .loss import FinetuneLoss
    from .sampler import SamplerState
    from .validation import ValidationLossSummary


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
        import torch

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


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require_sha256(value: str, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_finite_nonnegative(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _freeze_aggregate(value: Any, *, name: str) -> Any:
    forbidden = {"records", "seed", "timestamp_ns", "generated_at", "wall_clock"}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{name} keys must be nonempty strings")
            if key in forbidden or key.endswith("_ns"):
                raise ValueError(f"{name} contains forbidden field: {key}")
            result[key] = _freeze_aggregate(item, name=f"{name}.{key}")
        return MappingProxyType(result)
    if type(value) is bool or type(value) is str:
        return value
    if type(value) is int:
        if value < 0:
            raise ValueError(f"{name} integer values must be nonnegative")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} float values must be finite")
        return value
    raise TypeError(f"{name} must contain only finite aggregate JSON values")


def _freeze_identities(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("identities must be a nonempty mapping")
    result = {}
    for key, item in value.items():
        if type(key) is not str or not key or type(item) is not str or not item:
            raise ValueError("identities must map nonempty strings to nonempty strings")
        result[key] = item
    return MappingProxyType(result)


@dataclass(frozen=True)
class TrainingArtifactHashes:
    metrics_history_sha256: str
    epoch_metrics_csv_sha256: str
    training_curves_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "metrics_history_sha256",
            "epoch_metrics_csv_sha256",
            "training_curves_sha256",
        ):
            _require_sha256(getattr(self, name), name)


@dataclass(frozen=True)
class ProductionRunAttestation:
    training_config_sha256: str
    training_manifest_sha256: str
    resolved_domain_sha256: str
    cache_schema_version: Literal[2]
    clock_policy_version: Literal[1]
    base_checkpoint_sha256: str
    completed_epochs: int
    best_checkpoint_sha256: str
    best_micro_ade: float
    validation_run_id: str
    epoch_telemetry_count: int
    metrics_history_sha256: str
    epoch_metrics_csv_sha256: str
    training_curves_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "training_config_sha256",
            "training_manifest_sha256",
            "resolved_domain_sha256",
            "base_checkpoint_sha256",
            "best_checkpoint_sha256",
            "metrics_history_sha256",
            "epoch_metrics_csv_sha256",
            "training_curves_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.cache_schema_version != 2 or type(self.cache_schema_version) is not int:
            raise ValueError("cache_schema_version must be exactly 2")
        if self.clock_policy_version != 1 or type(self.clock_policy_version) is not int:
            raise ValueError("clock_policy_version must be exactly 1")
        if type(self.completed_epochs) is not int or self.completed_epochs <= 0:
            raise ValueError("completed_epochs must be a positive integer")
        if (
            type(self.epoch_telemetry_count) is not int
            or self.epoch_telemetry_count != self.completed_epochs
        ):
            raise ValueError("epoch_telemetry_count must equal completed_epochs")
        object.__setattr__(
            self,
            "best_micro_ade",
            _require_finite_nonnegative(self.best_micro_ade, "best_micro_ade"),
        )
        if type(self.validation_run_id) is not str or not self.validation_run_id:
            raise ValueError("validation_run_id must be a nonempty string")


@dataclass(frozen=True)
class EpochTelemetry:
    schema_version: Literal[1]
    epoch: int
    global_step: int
    train: EpochMetrics
    validation_loss: ValidationLossSummary
    validation: Mapping[str, Any]
    validation_report_sha256: str
    identities: Mapping[str, str]
    best_epoch: int
    best_selected_micro_ade: float
    is_best: bool

    def __post_init__(self) -> None:
        from .validation import ValidationLossSummary

        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("schema_version must be exactly 1")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if type(self.global_step) is not int or self.global_step < 0:
            raise ValueError("global_step must be a nonnegative integer")
        if type(self.train) is not EpochMetrics:
            raise TypeError("train must be EpochMetrics")
        if self.train.epoch != self.epoch or self.train.global_step != self.global_step:
            raise ValueError("train epoch/global step must match telemetry")
        if (
            type(self.train.completed_batches) is not int
            or self.train.completed_batches <= 0
            or type(self.train.completed_samples) is not int
            or self.train.completed_samples <= 0
            or self.train.completed_batches > self.train.completed_samples
            or self.global_step < self.train.completed_batches
        ):
            raise ValueError("train counts must describe a completed nonempty epoch")
        for name in (
            "mean_total_loss",
            "mean_trajectory_loss",
            "mean_preservation_loss",
        ):
            value = getattr(self.train, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"train {name} must be finite")
        if not isinstance(self.validation_loss, ValidationLossSummary):
            raise TypeError("validation_loss must be ValidationLossSummary")
        if self.validation_loss.count <= 0:
            raise ValueError("validation loss count must be positive")
        object.__setattr__(
            self, "validation", _freeze_aggregate(self.validation, name="validation")
        )
        _require_sha256(self.validation_report_sha256, "validation_report_sha256")
        object.__setattr__(self, "identities", _freeze_identities(self.identities))
        if type(self.best_epoch) is not int or not 0 <= self.best_epoch <= self.epoch:
            raise ValueError("best_epoch must identify a completed epoch")
        object.__setattr__(
            self,
            "best_selected_micro_ade",
            _require_finite_nonnegative(
                self.best_selected_micro_ade,
                "best_selected_micro_ade",
            ),
        )
        if type(self.is_best) is not bool:
            raise TypeError("is_best must be a bool")
