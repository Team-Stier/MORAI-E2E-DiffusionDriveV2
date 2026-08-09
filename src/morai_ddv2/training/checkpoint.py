from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import io
import math
from pathlib import Path
import re
from typing import Any, Optional

import torch
from torch import nn

from morai_ddv2.atomic import atomic_write_bytes

from .contracts import (
    EpochAccumulatorState,
    FreezeManifest,
    ResumeState,
    RngState,
    TrainingBoundaryError,
)
from .freeze import assert_training_boundary
from .rng import clone_rng_state
from .sampler import (
    ResumeContractError,
    SamplerState,
    StatefulPermutationSampler,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_KEYS = {
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


def _require_sha256(value: str, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TrainingBoundaryError(f"{label} must be a lowercase SHA-256")
    return value


def _require_optional_sha256(value: Optional[str], label: str) -> Optional[str]:
    if value is None:
        return None
    return _require_sha256(value, label)


def _owned_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _owned_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_owned_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_owned_value(item) for item in value)
    return copy.deepcopy(value)


def _owned_mutable_state(
    mutable_state: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(mutable_state, Mapping):
        raise TypeError("mutable_state must be a mapping")
    if tuple(mutable_state) != manifest.mutable_state_entries:
        raise TrainingBoundaryError(
            "mutable checkpoint keys must exactly match the freeze manifest"
        )
    owned = OrderedDict()
    for name in manifest.mutable_state_entries:
        tensor = mutable_state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TrainingBoundaryError(
                f"mutable checkpoint value is not a tensor: {name}"
            )
        owned[name] = tensor.detach().to(device="cpu").clone()
    return owned


def _owned_sampler_state(state: SamplerState) -> SamplerState:
    if type(state) is not SamplerState:
        raise TypeError("sampler_state must be a SamplerState")
    candidate = SamplerState(
        dataset_size=state.dataset_size,
        epoch=state.epoch,
        permutation=(
            state.permutation.detach().clone()
            if isinstance(state.permutation, torch.Tensor)
            else state.permutation
        ),
        committed_cursor=state.committed_cursor,
        generator_state=(
            state.generator_state.detach().clone()
            if isinstance(state.generator_state, torch.Tensor)
            else state.generator_state
        ),
    )
    if type(candidate.dataset_size) is not int or candidate.dataset_size < 0:
        raise ResumeContractError("invalid sampler dataset size")
    probe = StatefulPermutationSampler(range(candidate.dataset_size), seed=0)
    probe.load_state_dict(candidate)
    return probe.state_dict()


def _validated_accumulator(
    accumulator: EpochAccumulatorState,
    sampler_state: SamplerState,
    *,
    epoch: int,
    global_step: int,
) -> EpochAccumulatorState:
    if type(accumulator) is not EpochAccumulatorState:
        raise TypeError("accumulator must be an EpochAccumulatorState")
    integer_fields = (
        accumulator.epoch,
        accumulator.completed_batches,
        accumulator.completed_samples,
        accumulator.grad_norm_count,
    )
    if any(type(value) is not int or value < 0 for value in integer_fields):
        raise ResumeContractError("accumulator counts must be nonnegative integers")
    scalar_fields = (
        accumulator.total_loss_numerator,
        accumulator.trajectory_loss_numerator,
        accumulator.preservation_loss_numerator,
        accumulator.grad_norm_sum,
    )
    if any(
        type(value) not in (int, float) or not math.isfinite(float(value))
        for value in scalar_fields
    ):
        raise ResumeContractError("accumulator values must be finite")
    if accumulator.grad_norm_sum < 0.0:
        raise ResumeContractError("accumulator grad norm sum must be nonnegative")
    if (
        accumulator.epoch != epoch
        or sampler_state.epoch != epoch
        or accumulator.completed_samples != sampler_state.committed_cursor
    ):
        raise ResumeContractError(
            "accumulator progress disagrees with sampler epoch/cursor"
        )
    if accumulator.grad_norm_count != accumulator.completed_batches:
        raise ResumeContractError("accumulator grad norm count disagrees with batches")
    if accumulator.completed_samples == 0:
        if accumulator.completed_batches != 0:
            raise ResumeContractError("empty accumulator cannot contain batches")
    elif not 0 < accumulator.completed_batches <= accumulator.completed_samples:
        raise ResumeContractError("accumulator batch/sample counts are inconsistent")
    if global_step < accumulator.completed_batches:
        raise ResumeContractError("global step precedes completed accumulator batches")
    return EpochAccumulatorState(
        epoch=accumulator.epoch,
        completed_batches=accumulator.completed_batches,
        completed_samples=accumulator.completed_samples,
        total_loss_numerator=float(accumulator.total_loss_numerator),
        trajectory_loss_numerator=float(accumulator.trajectory_loss_numerator),
        preservation_loss_numerator=float(accumulator.preservation_loss_numerator),
        grad_norm_sum=float(accumulator.grad_norm_sum),
        grad_norm_count=accumulator.grad_norm_count,
    )


def _validated_epoch_and_step(epoch: int, global_step: int) -> tuple[int, int]:
    if type(epoch) is not int or epoch < 0:
        raise ResumeContractError("checkpoint epoch must be a nonnegative integer")
    if type(global_step) is not int or global_step < 0:
        raise ResumeContractError("global step must be a nonnegative integer")
    return epoch, global_step


def _validated_best(value: float) -> float:
    if type(value) not in (int, float):
        raise ResumeContractError("best micro ADE must be numeric")
    best = float(value)
    if math.isnan(best) or best < 0.0:
        raise ResumeContractError("best micro ADE must be nonnegative or infinity")
    return best


def _validated_manifest(manifest: FreezeManifest) -> FreezeManifest:
    if type(manifest) is not FreezeManifest:
        raise TypeError("manifest must be a FreezeManifest")
    if (
        len(set(manifest.trainable_parameters)) != len(manifest.trainable_parameters)
        or len(set(manifest.frozen_parameters)) != len(manifest.frozen_parameters)
        or len(set(manifest.mutable_state_entries))
        != len(manifest.mutable_state_entries)
    ):
        raise TrainingBoundaryError("freeze manifest contains duplicate names")
    return manifest


def save_training_checkpoint(
    path: Path,
    *,
    mutable_state,
    optimizer,
    scaler,
    epoch: int,
    global_step: int,
    best_micro_ade: float,
    rng: RngState,
    sampler_state: SamplerState,
    accumulator: EpochAccumulatorState,
    completed_epoch_record_sha256: Optional[str],
    base_checkpoint_sha256: str,
    manifest: FreezeManifest,
    config: Mapping[str, Any],
) -> None:
    manifest = _validated_manifest(manifest)
    epoch, global_step = _validated_epoch_and_step(epoch, global_step)
    sampler_owned = _owned_sampler_state(sampler_state)
    accumulator_owned = _validated_accumulator(
        accumulator,
        sampler_owned,
        epoch=epoch,
        global_step=global_step,
    )
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not hasattr(optimizer, "state_dict") or not hasattr(scaler, "state_dict"):
        raise TypeError("optimizer and scaler must expose state_dict")
    payload = {
        "mutable_state": _owned_mutable_state(mutable_state, manifest),
        "optimizer": _owned_value(optimizer.state_dict()),
        "scaler": _owned_value(scaler.state_dict()),
        "epoch": epoch,
        "global_step": global_step,
        "best_micro_ade": _validated_best(best_micro_ade),
        "rng": clone_rng_state(rng),
        "sampler_state": sampler_owned,
        "accumulator": accumulator_owned,
        "completed_epoch_record_sha256": _require_optional_sha256(
            completed_epoch_record_sha256,
            "completed epoch record hash",
        ),
        "base_checkpoint_sha256": _require_sha256(
            base_checkpoint_sha256,
            "base checkpoint hash",
        ),
        "manifest": manifest,
        "config": _owned_value(dict(config)),
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    atomic_write_bytes(Path(path), stream.getvalue())


def _load_payload(
    path: Path, *, expected_checkpoint_sha256: Optional[str] = None
) -> Mapping[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ResumeContractError(
            f"training checkpoint is not a file: {checkpoint_path}"
        )
    try:
        checkpoint_bytes = checkpoint_path.read_bytes()
    except Exception as error:
        raise ResumeContractError("unable to read training checkpoint") from error
    if expected_checkpoint_sha256 is not None:
        expected_checkpoint_sha256 = _require_sha256(
            expected_checkpoint_sha256,
            "expected training checkpoint hash",
        )
        if hashlib.sha256(checkpoint_bytes).hexdigest() != expected_checkpoint_sha256:
            raise ResumeContractError("training checkpoint SHA-256 changed")
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    except Exception as error:
        raise ResumeContractError("unable to load training checkpoint") from error
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS:
        raise ResumeContractError("training checkpoint schema is invalid")
    return payload


def _config_identity(config: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = dict(config)
    identity.pop("resume_checkpoint_sha256", None)
    return identity


def _validated_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_base_checkpoint_sha256: str,
    expected_checkpoint_sha256: Optional[str] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
    expected_identities: Optional[Mapping[str, str]] = None,
    expected_manifest: Optional[FreezeManifest] = None,
) -> Mapping[str, Any]:
    expected_base = _require_sha256(
        expected_base_checkpoint_sha256,
        "expected base checkpoint hash",
    )
    if payload["base_checkpoint_sha256"] != expected_base:
        raise TrainingBoundaryError("base checkpoint identity does not match resume")
    checkpoint_manifest = _validated_manifest(payload["manifest"])
    if expected_manifest is not None and checkpoint_manifest != expected_manifest:
        raise TrainingBoundaryError("freeze manifest identity does not match resume")
    epoch, global_step = _validated_epoch_and_step(
        payload["epoch"], payload["global_step"]
    )
    best = _validated_best(payload["best_micro_ade"])
    completed_hash = _require_optional_sha256(
        payload["completed_epoch_record_sha256"],
        "completed epoch record hash",
    )
    rng = clone_rng_state(payload["rng"])
    sampler_state = _owned_sampler_state(payload["sampler_state"])
    accumulator = _validated_accumulator(
        payload["accumulator"],
        sampler_state,
        epoch=epoch,
        global_step=global_step,
    )
    mutable_state = _owned_mutable_state(payload["mutable_state"], checkpoint_manifest)
    if not isinstance(payload["optimizer"], Mapping) or not isinstance(
        payload["scaler"], Mapping
    ):
        raise ResumeContractError("optimizer or scaler checkpoint state is invalid")
    if not isinstance(payload["config"], Mapping):
        raise ResumeContractError("training config checkpoint state is invalid")
    stored_config = dict(payload["config"])
    if expected_identities is not None:
        if not isinstance(expected_identities, Mapping):
            raise TypeError("expected_identities must be a mapping")
        stored_identities = stored_config.get("identities")
        if not isinstance(stored_identities, Mapping) or any(
            stored_identities.get(name) != value
            for name, value in expected_identities.items()
        ):
            raise ResumeContractError(
                "training checkpoint identity does not match invocation"
            )
    if expected_config is not None:
        if not isinstance(expected_config, Mapping):
            raise TypeError("expected_config must be a mapping")
        bound_resume = expected_config.get("resume_checkpoint_sha256")
        if bound_resume is not None and bound_resume != expected_checkpoint_sha256:
            raise ResumeContractError(
                "resolved run config does not bind the inspected checkpoint"
            )
        if _config_identity(stored_config) != _config_identity(expected_config):
            raise ResumeContractError(
                "resume checkpoint config does not match invocation"
            )
    stored_identities = stored_config.get("identities")
    if isinstance(stored_identities, Mapping) and (
        stored_identities.get("base_checkpoint_sha256") is not None
        and stored_identities.get("base_checkpoint_sha256") != expected_base
    ):
        raise ResumeContractError(
            "training checkpoint base identity is internally inconsistent"
        )
    return {
        "mutable_state": mutable_state,
        "optimizer": payload["optimizer"],
        "scaler": payload["scaler"],
        "epoch": epoch,
        "global_step": global_step,
        "best_micro_ade": best,
        "rng": rng,
        "sampler_state": sampler_state,
        "accumulator": accumulator,
        "completed_epoch_record_sha256": completed_hash,
        "manifest": checkpoint_manifest,
        "config": stored_config,
    }


def inspect_training_checkpoint(
    path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_base_checkpoint_sha256: str,
    expected_config: Optional[Mapping[str, Any]] = None,
    expected_identities: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    payload = _load_payload(
        Path(path), expected_checkpoint_sha256=expected_checkpoint_sha256
    )
    validated = _validated_checkpoint_payload(
        payload,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config=expected_config,
        expected_identities=expected_identities,
    )
    return _owned_value(validated["config"])


def _validated_model_mutable_state(
    model: nn.Module,
    mutable_state: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
) -> OrderedDict[str, torch.Tensor]:
    owned = _owned_mutable_state(mutable_state, manifest)
    current = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    if any(name not in current for name in manifest.mutable_state_entries):
        raise TrainingBoundaryError("model is missing a mutable checkpoint entry")
    for name, tensor in owned.items():
        target = current[name]
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise TrainingBoundaryError(f"mutable checkpoint tensor mismatch: {name}")
    return owned


def _load_mutable_state(
    model: nn.Module,
    mutable_state: Mapping[str, torch.Tensor],
) -> None:
    current = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    with torch.no_grad():
        for name, tensor in mutable_state.items():
            current[name].copy_(tensor.to(device=current[name].device))


def _snapshot_mutable_state(
    model: nn.Module,
    manifest: FreezeManifest,
) -> OrderedDict[str, torch.Tensor]:
    current = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    return OrderedDict(
        (name, current[name].detach().to(device="cpu").clone())
        for name in manifest.mutable_state_entries
    )


def load_training_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    sampler: StatefulPermutationSampler,
    manifest: FreezeManifest,
    expected_base_checkpoint_sha256: str,
    expected_checkpoint_sha256: Optional[str] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
    expected_identities: Optional[Mapping[str, str]] = None,
) -> ResumeState:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(sampler, StatefulPermutationSampler):
        raise TypeError("sampler must be a StatefulPermutationSampler")
    manifest = _validated_manifest(manifest)
    assert_training_boundary(model, manifest)
    payload = _load_payload(
        Path(path), expected_checkpoint_sha256=expected_checkpoint_sha256
    )
    validated = _validated_checkpoint_payload(
        payload,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config=expected_config,
        expected_identities=expected_identities,
        expected_manifest=manifest,
    )
    mutable_state = _validated_model_mutable_state(
        model, validated["mutable_state"], manifest
    )
    if not hasattr(optimizer, "load_state_dict") or not hasattr(
        scaler, "load_state_dict"
    ):
        raise TypeError("optimizer and scaler must expose load_state_dict")

    previous_mutable = _snapshot_mutable_state(model, manifest)
    previous_optimizer = _owned_value(optimizer.state_dict())
    previous_scaler = _owned_value(scaler.state_dict())
    previous_sampler = sampler.state_dict()
    try:
        _load_mutable_state(model, mutable_state)
        optimizer.load_state_dict(validated["optimizer"])
        scaler.load_state_dict(validated["scaler"])
        sampler.load_state_dict(validated["sampler_state"])
    except Exception:
        try:
            sampler.load_state_dict(previous_sampler)
            scaler.load_state_dict(previous_scaler)
            optimizer.load_state_dict(previous_optimizer)
            _load_mutable_state(model, previous_mutable)
            assert_training_boundary(model, manifest)
        except Exception as rollback_error:
            raise ResumeContractError(
                "training checkpoint installation failed and rollback was incomplete"
            ) from rollback_error
        raise
    assert_training_boundary(model, manifest)
    return ResumeState(
        epoch=validated["epoch"],
        global_step=validated["global_step"],
        best_micro_ade=validated["best_micro_ade"],
        sampler=sampler.state_dict(),
        accumulator=validated["accumulator"],
        completed_epoch_record_sha256=validated["completed_epoch_record_sha256"],
        rng=validated["rng"],
    )


__all__ = (
    "inspect_training_checkpoint",
    "load_training_checkpoint",
    "save_training_checkpoint",
)
