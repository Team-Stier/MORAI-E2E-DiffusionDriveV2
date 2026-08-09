from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import replace
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from morai_ddv2.atomic import atomic_write_bytes
from morai_ddv2.config import canonical_json_bytes, sha256_file
from morai_ddv2.errors import IntegrityError

from .contracts import (
    EpochAccumulatorState,
    EpochMetrics,
    EpochTelemetry,
    FreezeManifest,
    ProductionRunAttestation,
    RngState,
    StepMetrics,
    TrainingBoundaryError,
)
from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .dataset import TrainingConfig
from .freeze import (
    assert_training_boundary,
    build_optimizer,
    configure_training_boundary,
    extract_mutable_state,
)
from .loss import (
    assert_loss_contract,
    compose_finetune_loss,
    snapshot_source_parameters,
)
from .metrics import ValidationReport, validation_report_json_bytes
from .rng import (
    capture_rng_state,
    create_dataloader_iterator,
    restore_rng_state,
)
from .sampler import ResumeContractError, StatefulPermutationSampler
from .supervised_adapter import forward_trajectory_supervised
from .telemetry import (
    publish_epoch_telemetry,
    publish_production_run_attestation,
    rebuild_training_artifacts,
)
from .validation import validate_selector_b1, validate_supervised_loss_b1


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
    max_batches: Optional[int] = None,
    _accumulator_output: Optional[list[EpochAccumulatorState]] = None,
) -> EpochMetrics:
    if type(global_step) is not int or global_step < 0:
        raise ResumeContractError("global step must be a nonnegative integer")
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ResumeContractError("max_batches must be a positive integer")
    if _accumulator_output is not None and (
        type(_accumulator_output) is not list or _accumulator_output
    ):
        raise TypeError("_accumulator_output must be an empty list")
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

    applied_batches = 0
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
        applied_batches += 1
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
        if max_batches is not None and applied_batches >= max_batches:
            break

    sampler_state = sampler.state_dict()
    if current.completed_samples != sampler_state.dataset_size and max_batches is None:
        raise ResumeContractError("training iterator ended before sampler completion")
    if current.completed_samples <= 0:
        raise ResumeContractError("training epoch completed no samples")
    result = EpochMetrics(
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
    if _accumulator_output is not None:
        _accumulator_output.append(current)
    return result


def _empty_accumulator(epoch: int) -> EpochAccumulatorState:
    return EpochAccumulatorState(
        epoch=epoch,
        completed_batches=0,
        completed_samples=0,
        total_loss_numerator=0.0,
        trajectory_loss_numerator=0.0,
        preservation_loss_numerator=0.0,
        grad_norm_sum=0.0,
        grad_norm_count=0,
    )


def _publish_exact(path: Path, payload: bytes, *, label: str) -> None:
    path = Path(path)
    if path.exists():
        if not path.is_file():
            raise IntegrityError(f"existing {label} is not a file")
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise IntegrityError(f"unable to read existing {label}") from error
        if existing != payload:
            raise IntegrityError(f"existing {label} conflicts with requested bytes")
        return
    atomic_write_bytes(path, payload)


def _validation_projection(report: ValidationReport) -> Dict[str, Any]:
    if not isinstance(report, ValidationReport):
        raise TypeError("selector validation must return ValidationReport")
    selected = report.micro
    values = (
        selected.ade,
        selected.fde,
        selected.heading_mae,
        report.bag_macro_ade,
        report.oracle_min_ade_32,
        report.oracle_min_fde_32,
    )
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise TrainingBoundaryError(
            "completed validation must contain finite aggregate metrics"
        )
    return {
        "selected": {
            "count": selected.count,
            "ade": float(selected.ade),
            "fde": float(selected.fde),
            "heading_mae": float(selected.heading_mae),
        },
        "bag_macro_ade": float(report.bag_macro_ade),
        "min_ade_32": float(report.oracle_min_ade_32),
        "min_fde_32": float(report.oracle_min_fde_32),
    }


def _require_identity(identities: Mapping[str, str], name: str) -> str:
    try:
        value = identities[name]
    except KeyError as error:
        raise TrainingBoundaryError(f"missing training identity: {name}") from error
    if type(value) is not str or not value:
        raise TrainingBoundaryError(f"invalid training identity: {name}")
    return value


def _read_epoch_record(
    run_dir: Path, epoch: int, *, required: bool
) -> Optional[tuple[str, Mapping[str, Any]]]:
    path = Path(run_dir) / "metrics" / "epochs" / f"epoch-{epoch:04d}.json"
    if not path.exists():
        if required:
            raise ResumeContractError(
                f"completed epoch telemetry is missing for epoch {epoch}"
            )
        return None
    if not path.is_file():
        raise ResumeContractError("completed epoch telemetry is not a file")
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ResumeContractError("completed epoch telemetry is invalid") from error
    if (
        not isinstance(value, Mapping)
        or canonical_json_bytes(dict(value)) != payload
        or value.get("schema_version") != 1
        or value.get("epoch") != epoch
    ):
        raise ResumeContractError("completed epoch telemetry is inconsistent")
    return hashlib.sha256(payload).hexdigest(), value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _resume_best_epoch(run_dir: Path, epoch: int) -> int:
    best_epoch = 0
    for candidate in range(epoch):
        record = _read_epoch_record(run_dir, candidate, required=True)
        if record is not None and record[1].get("is_best") is True:
            best_epoch = candidate
    return best_epoch


def _validate_resume_epoch_binding(
    run_dir: Path,
    *,
    epoch: int,
    completed_epoch_record_sha256: Optional[str],
) -> None:
    if epoch == 0:
        if completed_epoch_record_sha256 is not None:
            raise ResumeContractError(
                "epoch-zero checkpoint cannot bind a completed epoch record"
            )
        return
    previous = _read_epoch_record(run_dir, epoch - 1, required=True)
    if previous is None or previous[0] != completed_epoch_record_sha256:
        raise ResumeContractError(
            "checkpoint completed epoch record hash does not match telemetry"
        )


def _validate_resume_config(path: Path, expected: Mapping[str, Any]) -> None:
    try:
        payload = torch.load(str(Path(path)), map_location="cpu")
    except Exception as error:
        raise ResumeContractError(
            "unable to inspect resume checkpoint config"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("config"), Mapping)
        or dict(payload["config"]) != dict(expected)
    ):
        raise ResumeContractError("resume checkpoint config does not match invocation")


def _bounded_validation_samples(dataloader: DataLoader, maximum: Optional[int]):
    if maximum is None:
        return dataloader
    return islice(dataloader, maximum)


def _fit_bounded(
    *,
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    sampler: StatefulPermutationSampler,
    dataloader_generator: torch.Generator,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    reference: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
    config: TrainingConfig,
    run_dir: Path,
    resolved_run_config: Mapping[str, Any],
    identities: Mapping[str, str],
    base_checkpoint_sha256: str,
    resume_path: Optional[Path],
    max_epochs: Optional[int],
    max_train_batches: Optional[int],
    max_validation_samples: Optional[int],
    drop_statistics: Optional[Mapping[str, int]],
    sync_statistics: Optional[Mapping[str, Union[int, float]]],
) -> None:
    epoch_limit = (
        config.epochs if max_epochs is None else min(config.epochs, max_epochs)
    )
    remaining_batches = max_train_batches
    resume_rng = None
    if resume_path is None:
        epoch = 0
        global_step = 0
        best_micro_ade = math.inf
        accumulator = _empty_accumulator(0)
    else:
        _validate_resume_config(Path(resume_path), resolved_run_config)
        state = load_training_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            sampler=sampler,
            manifest=manifest,
            expected_base_checkpoint_sha256=base_checkpoint_sha256,
        )
        if state.completed_epoch_record_sha256 is not None:
            raise ResumeContractError(
                "bounded resume cannot consume an authoritative epoch binding"
            )
        if state.epoch > epoch_limit:
            raise ResumeContractError("bounded resume epoch exceeds the invocation cap")
        epoch = state.epoch
        global_step = state.global_step
        best_micro_ade = state.best_micro_ade
        accumulator = state.accumulator
        resume_rng = state.rng
        if accumulator.completed_samples == 0:
            restore_rng_state(resume_rng, dataloader_generator)
            resume_rng = None
    last_iterator_base = dataloader_generator.get_state().clone()
    last_metrics = None
    invocation_samples = 0
    while epoch < epoch_limit:
        if resume_rng is None:
            last_iterator_base = dataloader_generator.get_state().clone()
        else:
            last_iterator_base = resume_rng.dataloader_iterator_base_cpu.clone()
        output = []
        last_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            sampler=sampler,
            optimizer=optimizer,
            scaler=scaler,
            reference=reference,
            manifest=manifest,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            trajectory_weight=config.trajectory_weight,
            preservation_weight=config.preservation_weight,
            epoch=epoch,
            global_step=global_step,
            accumulator=accumulator,
            resume_rng=resume_rng,
            checkpoint_path=run_dir / "debug" / "last.pt",
            checkpoint_interval_steps=config.checkpoint_interval_steps,
            best_micro_ade=best_micro_ade,
            completed_epoch_record_sha256=None,
            base_checkpoint_sha256=base_checkpoint_sha256,
            config=resolved_run_config,
            max_batches=remaining_batches,
            _accumulator_output=output,
        )
        resume_rng = None
        current = output[0]
        applied = last_metrics.completed_batches - accumulator.completed_batches
        invocation_samples += current.completed_samples - accumulator.completed_samples
        global_step = last_metrics.global_step
        if remaining_batches is not None:
            remaining_batches -= applied
        sampler_state = sampler.state_dict()
        if sampler_state.committed_cursor != sampler_state.dataset_size:
            accumulator = current
            break
        sampler.finish_epoch()
        accumulator = _empty_accumulator(epoch + 1)
        epoch += 1
        if remaining_batches == 0:
            break
    if last_metrics is None:
        raise TrainingBoundaryError("bounded training completed no batches")

    report = validate_selector_b1(
        model,
        _bounded_validation_samples(validation_dataloader, max_validation_samples),
        run_id=config.validation_run_id,
        drop_statistics=drop_statistics,
        sync_statistics=sync_statistics,
        identities=identities,
    )
    validation_loss = validate_supervised_loss_b1(
        model,
        _bounded_validation_samples(validation_dataloader, max_validation_samples),
        reference=reference,
        manifest=manifest,
        trajectory_weight=config.trajectory_weight,
        preservation_weight=config.preservation_weight,
        run_id=config.validation_run_id,
    )
    report_payload = validation_report_json_bytes(report)
    atomic_write_bytes(run_dir / "debug" / "validation.json", report_payload)
    selected_micro_ade = report.micro.ade
    if selected_micro_ade is None:
        raise TrainingBoundaryError("bounded validation selected micro ADE is empty")
    checkpoint_base = (
        dataloader_generator.get_state()
        if accumulator.completed_samples == 0
        else last_iterator_base
    )
    rng = capture_rng_state(
        dataloader_generator,
        dataloader_iterator_base_cpu=checkpoint_base,
    )
    is_best = float(selected_micro_ade) < best_micro_ade
    if is_best:
        best_micro_ade = float(selected_micro_ade)
    checkpoint_values = {
        "mutable_state": extract_mutable_state(model, manifest),
        "optimizer": optimizer,
        "scaler": scaler,
        "epoch": accumulator.epoch,
        "global_step": global_step,
        "best_micro_ade": best_micro_ade,
        "rng": rng,
        "sampler_state": sampler.state_dict(),
        "accumulator": accumulator,
        "completed_epoch_record_sha256": None,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "manifest": manifest,
        "config": resolved_run_config,
    }
    if is_best:
        save_training_checkpoint(run_dir / "debug" / "best.pt", **checkpoint_values)
    save_training_checkpoint(run_dir / "debug" / "last.pt", **checkpoint_values)
    summary = {
        "schema_version": 1,
        "completed_train_batches": global_step,
        "completed_train_samples": invocation_samples,
        "completed_validation_samples": validation_loss.count,
        "epoch": accumulator.epoch,
        "global_step": global_step,
        "train": {
            "total_loss": last_metrics.mean_total_loss,
            "trajectory_loss": last_metrics.mean_trajectory_loss,
            "preservation_loss": last_metrics.mean_preservation_loss,
        },
        "validation_loss": {
            "count": validation_loss.count,
            "total": validation_loss.total,
            "trajectory": validation_loss.trajectory,
            "preservation": validation_loss.preservation,
        },
        "validation": _validation_projection(report),
        "validation_report_sha256": hashlib.sha256(report_payload).hexdigest(),
    }
    atomic_write_bytes(
        run_dir / "debug" / "smoke-summary.json",
        canonical_json_bytes(summary),
    )


def fit(
    *,
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    sampler: StatefulPermutationSampler,
    dataloader_generator: torch.Generator,
    config: TrainingConfig,
    run_dir: Path,
    resolved_run_config: Mapping[str, Any],
    identities: Mapping[str, str],
    base_checkpoint_sha256: str,
    resume_path: Optional[Path] = None,
    max_epochs: Optional[int] = None,
    max_train_batches: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    cache_schema_version: int = 2,
    clock_policy_version: int = 1,
    drop_statistics: Optional[Mapping[str, int]] = None,
    sync_statistics: Optional[Mapping[str, Union[int, float]]] = None,
) -> Optional[ProductionRunAttestation]:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if type(config) is not TrainingConfig:
        raise TypeError("config must be TrainingConfig")
    if not isinstance(resolved_run_config, Mapping):
        raise TypeError("resolved_run_config must be a mapping")
    if not isinstance(identities, Mapping):
        raise TypeError("identities must be a mapping")
    bounded = any(
        bound is not None
        for bound in (max_epochs, max_train_batches, max_validation_samples)
    )
    for name, bound in (
        ("max_epochs", max_epochs),
        ("max_train_batches", max_train_batches),
        ("max_validation_samples", max_validation_samples),
    ):
        if bound is not None and (type(bound) is not int or bound <= 0):
            raise TrainingBoundaryError(f"{name} must be a positive integer")
    if cache_schema_version != 2 or type(cache_schema_version) is not int:
        raise TrainingBoundaryError("cache schema version must be exactly 2")
    if clock_policy_version != 1 or type(clock_policy_version) is not int:
        raise TrainingBoundaryError("clock policy version must be exactly 1")

    run_dir = Path(run_dir)
    resolved_payload = canonical_json_bytes(dict(resolved_run_config))
    _publish_exact(
        run_dir
        / ("debug/resolved-run-config.json" if bounded else "resolved-run-config.json"),
        resolved_payload,
        label="resolved run config",
    )
    training_config_sha256 = hashlib.sha256(resolved_payload).hexdigest()
    training_manifest_sha256 = _require_identity(identities, "training_manifest_sha256")
    resolved_domain_sha256 = _require_identity(identities, "resolved_domain_sha256")

    manifest = configure_training_boundary(model)
    assert_loss_contract(
        model._trajectory_head,
        trajectory_weight=config.trajectory_weight,
        preservation_weight=config.preservation_weight,
    )
    reference = snapshot_source_parameters(model, manifest)
    optimizer = build_optimizer(
        model,
        manifest,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    if bounded:
        _fit_bounded(
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            sampler=sampler,
            dataloader_generator=dataloader_generator,
            optimizer=optimizer,
            scaler=scaler,
            reference=reference,
            manifest=manifest,
            config=config,
            run_dir=run_dir,
            resolved_run_config=resolved_run_config,
            identities=identities,
            base_checkpoint_sha256=base_checkpoint_sha256,
            resume_path=resume_path,
            max_epochs=max_epochs,
            max_train_batches=max_train_batches,
            max_validation_samples=max_validation_samples,
            drop_statistics=drop_statistics,
            sync_statistics=sync_statistics,
        )
        return None
    resume_rng = None
    if resume_path is None:
        epoch = 0
        accumulator = _empty_accumulator(0)
        global_step = 0
        best_micro_ade = math.inf
        best_epoch = 0
        completed_epoch_hash: Optional[str] = None
    else:
        _validate_resume_config(Path(resume_path), resolved_run_config)
        state = load_training_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            sampler=sampler,
            manifest=manifest,
            expected_base_checkpoint_sha256=base_checkpoint_sha256,
        )
        if state.epoch > config.epochs:
            raise ResumeContractError("resume epoch exceeds configured epochs")
        _validate_resume_epoch_binding(
            run_dir,
            epoch=state.epoch,
            completed_epoch_record_sha256=state.completed_epoch_record_sha256,
        )
        epoch = state.epoch
        accumulator = state.accumulator
        global_step = state.global_step
        best_micro_ade = state.best_micro_ade
        best_epoch = _resume_best_epoch(run_dir, epoch)
        completed_epoch_hash = state.completed_epoch_record_sha256
        resume_rng = state.rng
        if accumulator.completed_samples == 0:
            restore_rng_state(resume_rng, dataloader_generator)
            resume_rng = None
    artifact_hashes = None

    while epoch < config.epochs:
        if resume_rng is None:
            iterator_base = dataloader_generator.get_state().clone()
        else:
            iterator_base = resume_rng.dataloader_iterator_base_cpu.clone()
        output = []
        epoch_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            sampler=sampler,
            optimizer=optimizer,
            scaler=scaler,
            reference=reference,
            manifest=manifest,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            trajectory_weight=config.trajectory_weight,
            preservation_weight=config.preservation_weight,
            epoch=epoch,
            global_step=global_step,
            accumulator=accumulator,
            resume_rng=resume_rng,
            checkpoint_path=run_dir / "last.pt",
            checkpoint_interval_steps=config.checkpoint_interval_steps,
            best_micro_ade=best_micro_ade,
            completed_epoch_record_sha256=completed_epoch_hash,
            base_checkpoint_sha256=base_checkpoint_sha256,
            config=resolved_run_config,
            _accumulator_output=output,
        )
        resume_rng = None
        completed_accumulator = output[0]
        global_step = epoch_metrics.global_step
        report = validate_selector_b1(
            model,
            validation_dataloader,
            run_id=config.validation_run_id,
            drop_statistics=drop_statistics,
            sync_statistics=sync_statistics,
            identities=identities,
        )
        validation_loss = validate_supervised_loss_b1(
            model,
            validation_dataloader,
            reference=reference,
            manifest=manifest,
            trajectory_weight=config.trajectory_weight,
            preservation_weight=config.preservation_weight,
            run_id=config.validation_run_id,
        )
        validation_payload = validation_report_json_bytes(report)
        validation_path = run_dir / "validation" / f"epoch-{epoch:04d}.json"
        _publish_exact(
            validation_path,
            validation_payload,
            label="validation report",
        )
        validation_sha256 = hashlib.sha256(validation_payload).hexdigest()
        selected_micro_ade = report.micro.ade
        if selected_micro_ade is None:
            raise TrainingBoundaryError("validation selected micro ADE is empty")
        is_best = float(selected_micro_ade) < best_micro_ade
        checkpoint_rng = capture_rng_state(
            dataloader_generator,
            dataloader_iterator_base_cpu=iterator_base,
        )
        if is_best:
            best_micro_ade = float(selected_micro_ade)
            best_epoch = epoch
            save_training_checkpoint(
                run_dir / "best.pt",
                mutable_state=extract_mutable_state(model, manifest),
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_micro_ade=best_micro_ade,
                rng=checkpoint_rng,
                sampler_state=sampler.state_dict(),
                accumulator=completed_accumulator,
                completed_epoch_record_sha256=completed_epoch_hash,
                base_checkpoint_sha256=base_checkpoint_sha256,
                manifest=manifest,
                config=resolved_run_config,
            )
        telemetry = EpochTelemetry(
            schema_version=1,
            epoch=epoch,
            global_step=global_step,
            train=epoch_metrics,
            validation_loss=validation_loss,
            validation=_validation_projection(report),
            validation_report_sha256=validation_sha256,
            identities=identities,
            best_epoch=best_epoch,
            best_selected_micro_ade=best_micro_ade,
            is_best=is_best,
        )
        _, completed_epoch_hash = publish_epoch_telemetry(run_dir, telemetry)
        sampler.finish_epoch()
        accumulator = _empty_accumulator(epoch + 1)
        last_rng = capture_rng_state(
            dataloader_generator,
            dataloader_iterator_base_cpu=dataloader_generator.get_state(),
        )
        save_training_checkpoint(
            run_dir / "last.pt",
            mutable_state=extract_mutable_state(model, manifest),
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            best_micro_ade=best_micro_ade,
            rng=last_rng,
            sampler_state=sampler.state_dict(),
            accumulator=accumulator,
            completed_epoch_record_sha256=completed_epoch_hash,
            base_checkpoint_sha256=base_checkpoint_sha256,
            manifest=manifest,
            config=resolved_run_config,
        )
        artifact_hashes = rebuild_training_artifacts(run_dir)
        epoch += 1

    if artifact_hashes is None and epoch == config.epochs:
        artifact_hashes = rebuild_training_artifacts(run_dir)

    if artifact_hashes is None or not math.isfinite(best_micro_ade):
        raise TrainingBoundaryError("training completed no epochs")
    best_path = run_dir / "best.pt"
    attestation = ProductionRunAttestation(
        training_config_sha256=training_config_sha256,
        training_manifest_sha256=training_manifest_sha256,
        resolved_domain_sha256=resolved_domain_sha256,
        cache_schema_version=2,
        clock_policy_version=1,
        base_checkpoint_sha256=base_checkpoint_sha256,
        completed_epochs=config.epochs,
        best_checkpoint_sha256=sha256_file(best_path),
        best_micro_ade=best_micro_ade,
        validation_run_id=config.validation_run_id,
        epoch_telemetry_count=config.epochs,
        metrics_history_sha256=artifact_hashes.metrics_history_sha256,
        epoch_metrics_csv_sha256=artifact_hashes.epoch_metrics_csv_sha256,
        training_curves_sha256=artifact_hashes.training_curves_sha256,
    )
    publish_production_run_attestation(run_dir, attestation)
    return attestation


__all__ = (
    "fit",
    "restore_batchnorm_buffers",
    "snapshot_batchnorm_buffers",
    "train_one_epoch",
    "train_step",
)
