from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .loss import compose_finetune_loss
from .metrics import (
    CANDIDATE_COUNT,
    CANDIDATE_SOURCE,
    VALIDATION_RUN_ID,
    PredictionRecord,
    ValidationReport,
    compute_best_of_32_metrics,
    compute_openloop_metrics,
    reset_torch_seed,
    selector_seed,
    validation_loss_seed,
)
from .supervised_adapter import forward_trajectory_supervised


@dataclass
class CapturedShortlist:
    post_bezier_pool: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None
    candidates: Optional[torch.Tensor] = None

    @property
    def pool(self) -> Optional[torch.Tensor]:
        return self.post_bezier_pool

    @property
    def indices(self) -> Optional[torch.Tensor]:
        return self.topk_indices

    @property
    def shortlist(self) -> Optional[torch.Tensor]:
        return self.candidates


@dataclass(frozen=True)
class ValidationLossSummary:
    count: int
    total: Optional[float]
    trajectory: Optional[float]
    preservation: Optional[float]

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("count must be a nonnegative integer")
        names = ("total", "trajectory", "preservation")
        if self.count == 0:
            if any(getattr(self, name) is not None for name in names):
                raise ValueError("an empty loss summary must contain null means")
            return
        for name in names:
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise TypeError("{} must be a finite float".format(name))
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("{} must be finite".format(name))
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _PreparedSample:
    sample_id: str
    timestamp_ns: int
    features: Mapping[str, torch.Tensor]
    target: torch.Tensor
    bag_sha256: Optional[str]
    command_name: Optional[str]


def _refresh_shortlist(capture: CapturedShortlist) -> None:
    pool = capture.post_bezier_pool
    indices = capture.topk_indices
    if pool is None or indices is None:
        return
    if pool.shape != (indices.shape[0], 800, 8, 3):
        raise RuntimeError("captured post-Bezier pool must have shape [B,800,8,3]")
    if indices.ndim != 2 or indices.shape[1] != CANDIDATE_COUNT:
        raise RuntimeError("captured top-k indices must have shape [B,32]")
    if pool.device != indices.device:
        raise RuntimeError("captured pool and indices must share one device")
    if indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise RuntimeError("captured top-k indices must use an integer dtype")
    if torch.any(indices < 0).item() or torch.any(indices >= 800).item():
        raise RuntimeError("captured top-k indices must be in [0,799]")
    gather_index = indices.to(dtype=torch.int64)[:, :, None, None].expand(-1, -1, 8, 3)
    capture.candidates = torch.gather(pool, 1, gather_index).detach().clone()


@contextmanager
def capture_coarse_top32_post_bezier(
    model: nn.Module,
) -> Iterator[CapturedShortlist]:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    try:
        head = model._trajectory_head
    except AttributeError as error:
        raise RuntimeError("model has no DDV2 trajectory head") from error

    missing_instance_attribute = object()
    head_namespace = vars(head)
    original_bezier_instance_attribute = head_namespace.get(
        "bezier_xyyaw", missing_instance_attribute
    )
    original_select_instance_attribute = head_namespace.get(
        "_select_topk", missing_instance_attribute
    )
    original_bezier = head.bezier_xyyaw
    original_select = head._select_topk
    if not callable(original_bezier) or not callable(original_select):
        raise RuntimeError("DDV2 capture callables are unavailable")
    capture = CapturedShortlist()

    def wrapped_bezier(*args: Any, **kwargs: Any):
        if capture.post_bezier_pool is not None:
            raise RuntimeError("bezier_xyyaw was called more than once")
        result = original_bezier(*args, **kwargs)
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("bezier_xyyaw must return a torch.Tensor")
        if result.ndim != 4 or result.shape[1:] != (800, 8, 3):
            raise RuntimeError("bezier_xyyaw must return shape [B,800,8,3]")
        capture.post_bezier_pool = result.detach().clone()
        _refresh_shortlist(capture)
        return result

    def wrapped_select(*args: Any, **kwargs: Any):
        if capture.topk_indices is not None:
            raise RuntimeError("_select_topk was called more than once")
        result = original_select(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 4:
            raise RuntimeError("_select_topk must return the pinned tuple shape")
        indices = result[3]
        if not isinstance(indices, torch.Tensor):
            raise RuntimeError("_select_topk indices must be a torch.Tensor")
        if indices.ndim != 2 or indices.shape[1] != CANDIDATE_COUNT:
            raise RuntimeError("_select_topk indices must have shape [B,32]")
        capture.topk_indices = indices.detach().clone()
        _refresh_shortlist(capture)
        return result

    bezier_installed = False
    select_installed = False

    def restore_instance_attribute(name: str, original: Any) -> None:
        if original is missing_instance_attribute:
            try:
                delattr(head, name)
            except AttributeError:
                pass
            return
        setattr(head, name, original)

    try:
        setattr(head, "bezier_xyyaw", wrapped_bezier)
        bezier_installed = True
        setattr(head, "_select_topk", wrapped_select)
        select_installed = True
        yield capture
    finally:
        try:
            if bezier_installed:
                restore_instance_attribute(
                    "bezier_xyyaw", original_bezier_instance_attribute
                )
        finally:
            if select_installed:
                restore_instance_attribute(
                    "_select_topk", original_select_instance_attribute
                )


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return fallback


def _metadata_scalar(metadata: Mapping[str, Any], name: str) -> Any:
    try:
        value = metadata[name]
    except KeyError as error:
        raise ValueError("sample metadata is missing {}".format(name)) from error
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("sample metadata {} must contain one value".format(name))
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("sample metadata {} must contain one value".format(name))
        return value.reshape(()).item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError("sample metadata {} must contain one value".format(name))
        return value[0]
    return value


def _prepare_sample(
    model: nn.Module,
    sample: Mapping[str, Any],
    *,
    require_prediction_metadata: bool,
) -> _PreparedSample:
    if not isinstance(sample, Mapping):
        raise TypeError("validation samples must be mappings")
    try:
        features = sample["features"]
        target = sample["trajectory"]
    except KeyError as error:
        raise ValueError("validation sample lacks features or trajectory") from error
    if not isinstance(features, Mapping) or not features:
        raise ValueError("sample features must be a nonempty mapping")
    if not isinstance(target, torch.Tensor):
        raise TypeError("sample trajectory must be a torch.Tensor")
    if target.ndim == 2 and target.shape == (8, 3):
        target = target.unsqueeze(0)
    if target.ndim != 3 or target.shape != (1, 8, 3):
        raise ValueError("validation requires every trajectory sample at B=1")
    if not torch.is_floating_point(target) or not torch.isfinite(target).all().item():
        raise ValueError("sample trajectory must be finite floating point")
    device = _model_device(model, target.device)
    prepared_features: Dict[str, torch.Tensor] = {}
    expected_ranks = {
        "camera_feature": 4,
        "lidar_feature": 4,
        "status_feature": 2,
    }
    for name, value in features.items():
        if type(name) is not str or not name:
            raise ValueError("feature names must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError("feature {} must be a torch.Tensor".format(name))
        expected_rank = expected_ranks.get(name)
        if expected_rank is not None and value.ndim == expected_rank - 1:
            value = value.unsqueeze(0)
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError("validation requires every feature sample at B=1")
        if torch.is_floating_point(value) and not torch.isfinite(value).all().item():
            raise ValueError("feature {} must be finite".format(name))
        prepared_features[name] = value.to(device=device)
    target = target.to(device=device)

    metadata_value = sample.get("metadata", sample)
    if not isinstance(metadata_value, Mapping):
        raise TypeError("sample metadata must be a mapping")
    sample_id = _metadata_scalar(metadata_value, "sample_id")
    timestamp_ns = _metadata_scalar(metadata_value, "timestamp_ns")
    if type(sample_id) is not str or not sample_id:
        raise ValueError("sample_id must be a nonempty string")
    if type(timestamp_ns) is not int or timestamp_ns < 0:
        raise ValueError("timestamp_ns must be a nonnegative integer")
    bag_sha256 = None
    command_name = None
    if require_prediction_metadata:
        bag_key = (
            "source_bag_sha256"
            if "source_bag_sha256" in metadata_value
            else "bag_sha256"
        )
        bag_sha256 = _metadata_scalar(metadata_value, bag_key)
        command_name = _metadata_scalar(metadata_value, "command_name")
        if type(bag_sha256) is not str:
            raise ValueError("bag SHA-256 must be a string")
        if type(command_name) is not str:
            raise ValueError("command_name must be a string")
    return _PreparedSample(
        sample_id,
        timestamp_ns,
        prepared_features,
        target,
        bag_sha256,
        command_name,
    )


def _snapshot_modes(model: nn.Module) -> Tuple[Tuple[nn.Module, bool], ...]:
    return tuple((module, module.training) for module in model.modules())


def _restore_modes(states: Tuple[Tuple[nn.Module, bool], ...]) -> None:
    for module, training in states:
        module.training = training


def _selected_in_shortlist(selected: torch.Tensor, candidates: torch.Tensor) -> bool:
    matches = (candidates == selected[:, None]).flatten(start_dim=2).all(dim=-1)
    return bool(torch.all(torch.any(matches, dim=1)).item())


def validate_selector_b1(
    model: nn.Module,
    samples: Iterable[Mapping[str, Any]],
    *,
    run_id: str = VALIDATION_RUN_ID,
    drop_statistics: Optional[Mapping[str, int]] = None,
    sync_statistics: Optional[Mapping[str, Union[int, float]]] = None,
    identities: Optional[Mapping[str, str]] = None,
) -> ValidationReport:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be a nonempty string")
    mode_states = _snapshot_modes(model)
    records = []
    try:
        model.eval()
        with torch.no_grad():
            for sample in samples:
                prepared = _prepare_sample(
                    model, sample, require_prediction_metadata=True
                )
                seed = selector_seed(prepared.timestamp_ns, run_id)
                with capture_coarse_top32_post_bezier(model) as capture:
                    reset_torch_seed(seed)
                    output = model(prepared.features)
                if (
                    not isinstance(output, dict)
                    or tuple(output).count("trajectory") != 1
                ):
                    raise RuntimeError("selector must return a trajectory output")
                selected = output["trajectory"]
                if not isinstance(selected, torch.Tensor):
                    raise RuntimeError("selector trajectory must be a torch.Tensor")
                if selected.shape != (1, 8, 3):
                    raise RuntimeError("selector trajectory must have shape [1,8,3]")
                if not torch.isfinite(selected).all().item():
                    raise RuntimeError("selector trajectory must be finite")
                if capture.candidates is None:
                    raise RuntimeError("selector did not produce a captured shortlist")
                if capture.candidates.shape != (1, CANDIDATE_COUNT, 8, 3):
                    raise RuntimeError("captured shortlist must have shape [1,32,8,3]")
                if not _selected_in_shortlist(selected, capture.candidates):
                    raise RuntimeError(
                        "selected trajectory is not in the same-forward shortlist"
                    )
                prediction = selected[0].detach().cpu().numpy().copy()
                target = prepared.target[0].detach().cpu().numpy().copy()
                candidates = capture.candidates[0].detach().cpu().numpy().copy()
                oracle = compute_best_of_32_metrics(candidates, target)
                records.append(
                    PredictionRecord(
                        sample_id=prepared.sample_id,
                        bag_sha256=prepared.bag_sha256,
                        command_name=prepared.command_name,
                        timestamp_ns=prepared.timestamp_ns,
                        seed=seed,
                        prediction=prediction,
                        target=target,
                        candidate_source=CANDIDATE_SOURCE,
                        candidate_count=CANDIDATE_COUNT,
                        min_ade_32=oracle.min_ade_32,
                        min_fde_32=oracle.min_fde_32,
                        argmin_ade_rank=oracle.argmin_ade_rank,
                        argmin_fde_rank=oracle.argmin_fde_rank,
                    )
                )
    finally:
        _restore_modes(mode_states)
    return compute_openloop_metrics(
        records,
        run_id=run_id,
        drop_statistics=drop_statistics,
        sync_statistics=sync_statistics,
        identities=identities,
    )


def _loss_value(tensor: Any, name: str) -> float:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 0:
        raise RuntimeError("{} must be a scalar torch.Tensor".format(name))
    value = float(tensor.detach().to(dtype=torch.float64, device="cpu").item())
    if not math.isfinite(value):
        raise RuntimeError("{} must be finite".format(name))
    return value


def validate_supervised_loss_b1(
    model: nn.Module,
    samples: Iterable[Mapping[str, Any]],
    *,
    reference: Any,
    manifest: Any,
    trajectory_weight: float,
    preservation_weight: float,
    run_id: str = VALIDATION_RUN_ID,
) -> ValidationLossSummary:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be a nonempty string")
    if type(trajectory_weight) not in (int, float) or trajectory_weight != 12.0:
        raise ValueError("trajectory_weight must be exactly 12.0")
    if type(preservation_weight) not in (int, float) or preservation_weight != 1.0e-3:
        raise ValueError("preservation_weight must be exactly 1e-3")
    mode_states = _snapshot_modes(model)
    values = []
    try:
        model.eval()
        with torch.no_grad():
            for sample in samples:
                prepared = _prepare_sample(
                    model, sample, require_prediction_metadata=False
                )
                seed = validation_loss_seed(prepared.timestamp_ns, run_id)
                reset_torch_seed(seed)
                supervised = forward_trajectory_supervised(
                    model, prepared.features, prepared.target
                )
                loss = compose_finetune_loss(
                    supervised.trajectory_loss,
                    model,
                    reference,
                    manifest,
                    trajectory_weight=trajectory_weight,
                    preservation_weight=preservation_weight,
                )
                values.append(
                    (
                        prepared.sample_id,
                        _loss_value(loss.total, "total validation loss"),
                        _loss_value(loss.trajectory, "trajectory validation loss"),
                        _loss_value(loss.preservation, "preservation validation loss"),
                    )
                )
    finally:
        _restore_modes(mode_states)
    values.sort(key=lambda item: item[0])
    sample_ids = tuple(item[0] for item in values)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("validation loss samples contain duplicate sample IDs")
    count = len(values)
    if count == 0:
        return ValidationLossSummary(0, None, None, None)
    return ValidationLossSummary(
        count,
        math.fsum(item[1] for item in values) / count,
        math.fsum(item[2] for item in values) / count,
        math.fsum(item[3] for item in values) / count,
    )


__all__ = (
    "CapturedShortlist",
    "ValidationLossSummary",
    "capture_coarse_top32_post_bezier",
    "validate_selector_b1",
    "validate_supervised_loss_b1",
    "validation_loss_seed",
)
