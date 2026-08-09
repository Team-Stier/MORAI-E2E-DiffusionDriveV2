from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .contracts import (
    EpochAccumulatorState,
    EpochMetrics,
    FreezeManifest,
    RngState,
    StepMetrics,
    TrainingBoundaryError,
)
from .checkpoint import save_training_checkpoint
from .freeze import (
    assert_training_boundary,
    configure_training_boundary,
    extract_mutable_state,
)
from .loss import compose_finetune_loss
from .rng import (
    capture_rng_state,
    create_dataloader_iterator,
    restore_rng_state,
)
from .sampler import ResumeContractError, StatefulPermutationSampler
from .supervised_adapter import forward_trajectory_supervised


def _autocast(*, enabled: bool) -> AbstractContextManager[Any]:
    return torch.cuda.amp.autocast(enabled=enabled)


def _batchnorm_buffer_names(
    model: nn.Module,
    manifest: FreezeManifest,
) -> tuple[str, ...]:
    assert_training_boundary(model, manifest)
    named_buffers = dict(model.named_buffers())
    named_modules = dict(model.named_modules())
    names = []
    for name in manifest.mutable_state_entries:
        if name not in named_buffers:
            continue
        module_name, separator, buffer_name = name.rpartition(".")
        module = named_modules.get(module_name)
        if (
            not separator
            or not isinstance(module, nn.modules.batchnorm._BatchNorm)
            or buffer_name not in ("running_mean", "running_var", "num_batches_tracked")
        ):
            raise TrainingBoundaryError(
                f"mutable non-parameter is not an allowlisted BatchNorm buffer: {name}"
            )
        names.append(name)
    return tuple(names)


def snapshot_batchnorm_buffers(
    model: nn.Module,
    manifest: FreezeManifest,
) -> Mapping[str, torch.Tensor]:
    names = _batchnorm_buffer_names(model, manifest)
    named_buffers = dict(model.named_buffers())
    return OrderedDict((name, named_buffers[name].detach().clone()) for name in names)


def restore_batchnorm_buffers(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
) -> None:
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    names = _batchnorm_buffer_names(model, manifest)
    if tuple(snapshot) != names:
        raise TrainingBoundaryError(
            "BatchNorm snapshot keys must exactly match the freeze manifest"
        )
    named_buffers = dict(model.named_buffers())
    for name in names:
        source = snapshot[name]
        target = named_buffers[name]
        if (
            not isinstance(source, torch.Tensor)
            or source.shape != target.shape
            or source.dtype != target.dtype
            or source.device != target.device
        ):
            raise TrainingBoundaryError(f"BatchNorm snapshot tensor mismatch: {name}")
    with torch.no_grad():
        for name in names:
            named_buffers[name].copy_(snapshot[name])
    assert_training_boundary(model, manifest)


def _require_step_inputs(
    amp: bool,
    grad_clip_norm: float,
) -> None:
    if type(amp) is not bool:
        raise TypeError("amp must be a bool")
    if (
        type(grad_clip_norm) not in (int, float)
        or not math.isfinite(float(grad_clip_norm))
        or grad_clip_norm <= 0.0
    ):
        raise TrainingBoundaryError("gradient clip norm must be finite and positive")


def train_step(
    model,
    batch,
    optimizer,
    scaler,
    reference,
    manifest,
    *,
    amp: bool,
    grad_clip_norm: float,
    trajectory_weight: float,
    preservation_weight: float,
) -> StepMetrics:
    _require_step_inputs(amp, grad_clip_norm)
    optimizer.zero_grad(set_to_none=True)
    bn_before = snapshot_batchnorm_buffers(model, manifest)
    try:
        with _autocast(enabled=amp):
            result = forward_trajectory_supervised(
                model,
                batch["features"],
                batch["trajectory"],
            )
        loss = compose_finetune_loss(
            result.trajectory_loss.float(),
            model,
            reference,
            manifest,
            trajectory_weight=trajectory_weight,
            preservation_weight=preservation_weight,
        )
        scaler.scale(loss.total).backward()
        scaler.unscale_(optimizer)
        named_parameters = dict(model.named_parameters())
        try:
            parameters = [
                named_parameters[name] for name in manifest.trainable_parameters
            ]
        except KeyError as error:
            raise TrainingBoundaryError(
                f"unknown trainable parameter during clipping: {error.args[0]}"
            ) from error
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
        if not isinstance(grad_norm, torch.Tensor):
            grad_norm = torch.tensor(
                grad_norm,
                dtype=torch.float32,
                device=parameters[0].device,
            )
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if (
            not math.isfinite(scale_before)
            or not math.isfinite(scale_after)
            or scale_before <= 0.0
            or scale_after <= 0.0
        ):
            raise TrainingBoundaryError(
                "GradScaler scale must remain finite and positive"
            )
        assert_training_boundary(model, manifest)
        optimizer_step_applied = scale_after >= scale_before
        if not optimizer_step_applied:
            restore_batchnorm_buffers(model, bn_before, manifest)
        return StepMetrics.from_tensors(
            loss,
            grad_norm,
            optimizer_step_applied=optimizer_step_applied,
        )
    except Exception:
        restore_batchnorm_buffers(model, bn_before, manifest)
        raise


def _validated_accumulator(
    accumulator: EpochAccumulatorState,
    sampler: StatefulPermutationSampler,
    epoch: int,
) -> EpochAccumulatorState:
    if type(accumulator) is not EpochAccumulatorState:
        raise TypeError("accumulator must be an EpochAccumulatorState")
    sampler_state = sampler.state_dict()
    if (
        type(epoch) is not int
        or epoch < 0
        or accumulator.epoch != epoch
        or sampler_state.epoch != epoch
        or accumulator.completed_samples != sampler_state.committed_cursor
    ):
        raise ResumeContractError(
            "accumulator progress disagrees with sampler epoch/cursor"
        )
    integer_counts = (
        accumulator.completed_batches,
        accumulator.completed_samples,
        accumulator.grad_norm_count,
    )
    if any(type(value) is not int or value < 0 for value in integer_counts):
        raise ResumeContractError("accumulator counts must be nonnegative integers")
    values = (
        accumulator.total_loss_numerator,
        accumulator.trajectory_loss_numerator,
        accumulator.preservation_loss_numerator,
        accumulator.grad_norm_sum,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ResumeContractError("accumulator values must be finite")
    if accumulator.grad_norm_count != accumulator.completed_batches:
        raise ResumeContractError("accumulator grad norm count disagrees with batches")
    return accumulator


def _actual_batch_size(batch: Mapping[str, Any]) -> int:
    if not isinstance(batch, Mapping):
        raise TrainingBoundaryError("training batch must be a mapping")
    trajectory = batch.get("trajectory")
    if not isinstance(trajectory, torch.Tensor) or trajectory.ndim == 0:
        raise TrainingBoundaryError("training batch trajectory must have a batch axis")
    batch_size = int(trajectory.shape[0])
    if batch_size <= 0:
        raise TrainingBoundaryError("training batch must be nonempty")
    return batch_size


def _advance_accumulator(
    accumulator: EpochAccumulatorState,
    metrics: StepMetrics,
    batch_size: int,
) -> EpochAccumulatorState:
    if not metrics.optimizer_step_applied:
        return accumulator
    metric_values = (
        metrics.total_loss,
        metrics.trajectory_loss,
        metrics.preservation_loss,
        metrics.grad_norm,
    )
    if any(not math.isfinite(value) for value in metric_values):
        raise TrainingBoundaryError("applied step metrics must be finite")
    return replace(
        accumulator,
        completed_batches=accumulator.completed_batches + 1,
        completed_samples=accumulator.completed_samples + batch_size,
        total_loss_numerator=(
            accumulator.total_loss_numerator + metrics.total_loss * batch_size
        ),
        trajectory_loss_numerator=(
            accumulator.trajectory_loss_numerator + metrics.trajectory_loss * batch_size
        ),
        preservation_loss_numerator=(
            accumulator.preservation_loss_numerator
            + metrics.preservation_loss * batch_size
        ),
        grad_norm_sum=accumulator.grad_norm_sum + metrics.grad_norm,
        grad_norm_count=accumulator.grad_norm_count + 1,
    )


def _checkpointing_enabled(
    *,
    checkpoint_path: Optional[Path],
    checkpoint_interval_steps: Optional[int],
    base_checkpoint_sha256: Optional[str],
    config: Optional[Mapping[str, Any]],
) -> bool:
    supplied = (
        checkpoint_path is not None,
        checkpoint_interval_steps is not None,
        base_checkpoint_sha256 is not None,
        config is not None,
    )
    if any(supplied) and not all(supplied):
        raise TrainingBoundaryError(
            "step checkpoint path, interval, base identity, and config are all required"
        )
    if checkpoint_interval_steps is not None and (
        type(checkpoint_interval_steps) is not int or checkpoint_interval_steps <= 0
    ):
        raise TrainingBoundaryError(
            "step checkpoint interval must be a positive integer"
        )
    return all(supplied)


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    sampler: StatefulPermutationSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    reference: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
    amp: bool,
    grad_clip_norm: float,
    trajectory_weight: float,
    preservation_weight: float,
    epoch: int,
    global_step: int,
    accumulator: EpochAccumulatorState,
    resume_rng: Optional[RngState] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval_steps: Optional[int] = None,
    best_micro_ade: float = math.inf,
    completed_epoch_record_sha256: Optional[str] = None,
    base_checkpoint_sha256: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> EpochMetrics:
    if type(global_step) is not int or global_step < 0:
        raise ResumeContractError("global step must be a nonnegative integer")
    reapplied = configure_training_boundary(model)
    if reapplied != manifest:
        raise TrainingBoundaryError("freeze manifest changed at epoch boundary")
    current = _validated_accumulator(accumulator, sampler, epoch)
    if current.completed_samples > 0 and resume_rng is None:
        raise ResumeContractError(
            "resume RNG is required for committed mid-epoch progress"
        )
    generator = dataloader.generator
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ResumeContractError("training DataLoader must own a CPU generator")
    checkpointing = _checkpointing_enabled(
        checkpoint_path=checkpoint_path,
        checkpoint_interval_steps=checkpoint_interval_steps,
        base_checkpoint_sha256=base_checkpoint_sha256,
        config=config,
    )
    if resume_rng is None:
        iterator, iterator_base = create_dataloader_iterator(dataloader, generator)
    else:
        restore_rng_state(resume_rng, generator)
        iterator, iterator_base = create_dataloader_iterator(
            dataloader,
            generator,
            iterator_base_state=resume_rng.dataloader_iterator_base_cpu,
            current_generator_state=resume_rng.dataloader_cpu,
        )

    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        metrics = train_step(
            model,
            batch,
            optimizer,
            scaler,
            reference,
            manifest,
            amp=amp,
            grad_clip_norm=grad_clip_norm,
            trajectory_weight=trajectory_weight,
            preservation_weight=preservation_weight,
        )
        if not metrics.optimizer_step_applied:
            current_generator_state = generator.get_state().clone()
            del iterator
            iterator, _ = create_dataloader_iterator(
                dataloader,
                generator,
                iterator_base_state=iterator_base,
                current_generator_state=current_generator_state,
            )
            continue
        batch_size = _actual_batch_size(batch)
        sampler.mark_consumed(batch_size)
        current = _advance_accumulator(current, metrics, batch_size)
        global_step += 1
        if checkpointing and global_step % checkpoint_interval_steps == 0:
            rng = capture_rng_state(
                generator,
                dataloader_iterator_base_cpu=iterator_base,
            )
            save_training_checkpoint(
                checkpoint_path,
                mutable_state=extract_mutable_state(model, manifest),
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_micro_ade=best_micro_ade,
                rng=rng,
                sampler_state=sampler.state_dict(),
                accumulator=current,
                completed_epoch_record_sha256=completed_epoch_record_sha256,
                base_checkpoint_sha256=base_checkpoint_sha256,
                manifest=manifest,
                config=config,
            )

    sampler_state = sampler.state_dict()
    if current.completed_samples != sampler_state.dataset_size:
        raise ResumeContractError("training iterator ended before sampler completion")
    if current.completed_samples <= 0:
        raise ResumeContractError("training epoch completed no samples")
    return EpochMetrics(
        epoch=epoch,
        global_step=global_step,
        completed_batches=current.completed_batches,
        completed_samples=current.completed_samples,
        mean_total_loss=current.total_loss_numerator / current.completed_samples,
        mean_trajectory_loss=(
            current.trajectory_loss_numerator / current.completed_samples
        ),
        mean_preservation_loss=(
            current.preservation_loss_numerator / current.completed_samples
        ),
    )


__all__ = (
    "restore_batchnorm_buffers",
    "snapshot_batchnorm_buffers",
    "train_one_epoch",
    "train_step",
)
