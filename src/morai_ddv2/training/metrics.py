from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np


VALIDATION_RUN_ID = "morai-ddv2-v1-oracle-command-validation-v1"
DOMAIN_ID = "morai-ddv2-v1-oracle-command"
COMMAND_SOURCE = "oracle-command"
CANDIDATE_SOURCE = "coarse_top32_post_bezier"
CANDIDATE_COUNT = 32
COMMAND_NAMES = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")


def _nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError("{} must be a nonempty string".format(name))
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if type(value) not in (int, float, np.float32, np.float64):
        raise TypeError("{} must be numeric".format(name))
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and nonnegative".format(name))
    return result


def _trajectory_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("{} must be a NumPy array".format(name))
    if value.shape != (8, 3):
        raise ValueError("{} must have shape [8,3]".format(name))
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError("{} must be numeric".format(name))
    if not np.isfinite(value).all():
        raise ValueError("{} must be finite".format(name))
    copied = np.array(value, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _copy_drop_statistics(
    values: Optional[Mapping[str, int]],
) -> Mapping[str, int]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise TypeError("drop_statistics must be a mapping")
    copied: Dict[str, int] = {}
    names = tuple(values)
    for name in names:
        _nonempty_string(name, "drop statistic name")
    for name in sorted(names):
        value = values[name]
        if type(value) is not int or value < 0:
            raise ValueError("drop statistics must be nonnegative integers")
        copied[name] = value
    return MappingProxyType(copied)


def _copy_sync_statistics(
    values: Optional[Mapping[str, Union[int, float]]],
) -> Mapping[str, Union[int, float]]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise TypeError("sync_statistics must be a mapping")
    copied: Dict[str, Union[int, float]] = {}
    names = tuple(values)
    for name in names:
        _nonempty_string(name, "sync statistic name")
    for name in sorted(names):
        value = values[name]
        if type(value) is int:
            copied[name] = value
        elif type(value) is float and math.isfinite(value):
            copied[name] = value
        else:
            raise ValueError("sync statistics must be finite JSON numbers")
    return MappingProxyType(copied)


def _copy_identities(
    values: Optional[Mapping[str, str]],
) -> Mapping[str, str]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise TypeError("identities must be a mapping")
    copied: Dict[str, str] = {}
    names = tuple(values)
    for name in names:
        _nonempty_string(name, "identity name")
    for name in sorted(names):
        copied[name] = _nonempty_string(values[name], "identity value")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    bag_sha256: str
    command_name: str
    timestamp_ns: int
    seed: int
    prediction: np.ndarray
    target: np.ndarray
    candidate_source: Literal["coarse_top32_post_bezier"]
    candidate_count: Literal[32]
    min_ade_32: float
    min_fde_32: float
    argmin_ade_rank: int
    argmin_fde_rank: int

    def __post_init__(self) -> None:
        _nonempty_string(self.sample_id, "sample_id")
        if (
            type(self.bag_sha256) is not str
            or len(self.bag_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.bag_sha256)
        ):
            raise ValueError("bag_sha256 must be lowercase SHA-256")
        if self.command_name not in COMMAND_NAMES:
            raise ValueError("command_name is invalid")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < (1 << 64):
            raise ValueError("seed must be an unsigned 64-bit integer")
        object.__setattr__(
            self, "prediction", _trajectory_array(self.prediction, "prediction")
        )
        object.__setattr__(self, "target", _trajectory_array(self.target, "target"))
        if self.candidate_source != CANDIDATE_SOURCE:
            raise ValueError("candidate_source is invalid")
        if self.candidate_count != CANDIDATE_COUNT:
            raise ValueError("candidate_count must be exactly 32")
        object.__setattr__(
            self,
            "min_ade_32",
            _finite_nonnegative(self.min_ade_32, "min_ade_32"),
        )
        object.__setattr__(
            self,
            "min_fde_32",
            _finite_nonnegative(self.min_fde_32, "min_fde_32"),
        )
        for name in ("argmin_ade_rank", "argmin_fde_rank"):
            rank = getattr(self, name)
            if type(rank) is not int or not 0 <= rank < CANDIDATE_COUNT:
                raise ValueError("{} must be in [0,31]".format(name))


@dataclass(frozen=True)
class MetricSummary:
    count: int
    ade: Optional[float]
    fde: Optional[float]
    heading_mae: Optional[float]

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("count must be a nonnegative integer")
        names = ("ade", "fde", "heading_mae")
        if self.count == 0:
            if any(getattr(self, name) is not None for name in names):
                raise ValueError("empty metric summaries must contain null metrics")
            return
        for name in names:
            value = getattr(self, name)
            if value is None:
                raise ValueError("nonempty metric summaries require every metric")
            object.__setattr__(self, name, _finite_nonnegative(value, name))


@dataclass(frozen=True)
class BestOf32Metrics:
    min_ade_32: float
    min_fde_32: float
    argmin_ade_rank: int
    argmin_fde_rank: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_ade_32",
            _finite_nonnegative(self.min_ade_32, "min_ade_32"),
        )
        object.__setattr__(
            self,
            "min_fde_32",
            _finite_nonnegative(self.min_fde_32, "min_fde_32"),
        )
        for name in ("argmin_ade_rank", "argmin_fde_rank"):
            rank = getattr(self, name)
            if type(rank) is not int or not 0 <= rank < CANDIDATE_COUNT:
                raise ValueError("{} must be in [0,31]".format(name))


@dataclass(frozen=True)
class ValidationReport:
    domain_id: str
    command_source: str
    run_id: str
    micro: MetricSummary
    oracle_min_ade_32: Optional[float]
    oracle_min_fde_32: Optional[float]
    oracle_candidate_source: Literal["coarse_top32_post_bezier"]
    oracle_candidate_count: Literal[32]
    bag_macro_ade: Optional[float]
    per_command: Mapping[str, MetricSummary]
    records: Tuple[PredictionRecord, ...]
    drop_statistics: Mapping[str, int]
    sync_statistics: Mapping[str, Union[int, float]]
    identities: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.domain_id != DOMAIN_ID:
            raise ValueError("domain_id is invalid")
        if self.command_source != COMMAND_SOURCE:
            raise ValueError("command_source is invalid")
        _nonempty_string(self.run_id, "run_id")
        if not isinstance(self.micro, MetricSummary):
            raise TypeError("micro must be a MetricSummary")
        if self.oracle_candidate_source != CANDIDATE_SOURCE:
            raise ValueError("oracle_candidate_source is invalid")
        if self.oracle_candidate_count != CANDIDATE_COUNT:
            raise ValueError("oracle_candidate_count must be exactly 32")
        optional_metrics = (
            "oracle_min_ade_32",
            "oracle_min_fde_32",
            "bag_macro_ade",
        )
        for name in optional_metrics:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_nonnegative(value, name))
        if set(self.per_command) != set(COMMAND_NAMES):
            raise ValueError("per_command must contain exactly the four commands")
        if any(
            not isinstance(self.per_command[name], MetricSummary)
            for name in COMMAND_NAMES
        ):
            raise TypeError("per_command values must be MetricSummary instances")
        object.__setattr__(
            self,
            "per_command",
            MappingProxyType({name: self.per_command[name] for name in COMMAND_NAMES}),
        )
        records = tuple(self.records)
        if any(not isinstance(record, PredictionRecord) for record in records):
            raise TypeError("records must contain PredictionRecord instances")
        records = tuple(sorted(records, key=lambda record: record.sample_id))
        sample_ids = tuple(record.sample_id for record in records)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("records contain duplicate sample IDs")
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self, "drop_statistics", _copy_drop_statistics(self.drop_statistics)
        )
        object.__setattr__(
            self, "sync_statistics", _copy_sync_statistics(self.sync_statistics)
        )
        object.__setattr__(self, "identities", _copy_identities(self.identities))


def _validate_seed_input(timestamp_ns: int, run_id: str) -> None:
    if type(timestamp_ns) is not int or timestamp_ns < 0:
        raise ValueError("timestamp_ns must be a nonnegative integer")
    _nonempty_string(run_id, "run_id")


def selector_seed(timestamp_ns: int, run_id: str = VALIDATION_RUN_ID) -> int:
    _validate_seed_input(timestamp_ns, run_id)
    payload = "diffusiondrive_v2:{}:{}".format(timestamp_ns, run_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def validation_loss_seed(timestamp_ns: int, run_id: str = VALIDATION_RUN_ID) -> int:
    _validate_seed_input(timestamp_ns, run_id)
    payload = "diffusiondrive_v2_supervised_loss:{}:{}".format(
        timestamp_ns, run_id
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def reset_torch_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < (1 << 64):
        raise ValueError("seed must be an unsigned 64-bit integer")
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bezier_headings(selected_xy: np.ndarray) -> np.ndarray:
    if not isinstance(selected_xy, np.ndarray):
        raise TypeError("selected_xy must be a NumPy array")
    if selected_xy.shape[-2:] != (8, 2):
        raise ValueError("selected_xy must end in shape [8,2]")
    points = np.asarray(selected_xy, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("selected_xy must be finite")
    origin = np.zeros(points.shape[:-2] + (1, 2), dtype=np.float64)
    control = np.concatenate((origin, points), axis=-2)
    delta = control[..., 1:, :] - control[..., :-1, :]
    t = np.arange(1, 9, dtype=np.float64) / 8.0
    indices = np.arange(8, dtype=np.float64)
    binomial = np.asarray([math.comb(7, index) for index in range(8)])
    basis = (
        binomial[None, :]
        * t[:, None] ** indices[None, :]
        * (1.0 - t[:, None]) ** (7.0 - indices[None, :])
    )
    derivative = 8.0 * np.einsum("ki,...ij->...kj", basis, delta)
    return np.arctan2(derivative[..., 1], derivative[..., 0])


def compute_best_of_32_metrics(
    candidates: np.ndarray, target: np.ndarray
) -> BestOf32Metrics:
    if not isinstance(candidates, np.ndarray):
        raise TypeError("candidates must be a NumPy array")
    if candidates.shape != (CANDIDATE_COUNT, 8, 3):
        raise ValueError("candidates must have shape [32,8,3]")
    candidate_values = np.asarray(candidates, dtype=np.float64)
    target_values = np.asarray(_trajectory_array(target, "target"), dtype=np.float64)
    if not np.isfinite(candidate_values).all():
        raise ValueError("candidates must be finite")
    xy_error = np.linalg.norm(
        candidate_values[..., :2] - target_values[None, ..., :2], axis=-1
    )
    ade = np.mean(xy_error, axis=-1, dtype=np.float64)
    fde = xy_error[:, -1]
    argmin_ade = int(np.argmin(ade))
    argmin_fde = int(np.argmin(fde))
    return BestOf32Metrics(
        float(ade[argmin_ade]),
        float(fde[argmin_fde]),
        argmin_ade,
        argmin_fde,
    )


def _selected_metrics(record: PredictionRecord) -> Tuple[float, float, float]:
    prediction = np.asarray(record.prediction, dtype=np.float64)
    target = np.asarray(record.target, dtype=np.float64)
    xy_error = np.linalg.norm(prediction[:, :2] - target[:, :2], axis=-1)
    heading_delta = prediction[:, 2] - target[:, 2]
    circular_heading_error = np.abs(
        np.arctan2(np.sin(heading_delta), np.cos(heading_delta))
    )
    return (
        float(np.mean(xy_error, dtype=np.float64)),
        float(xy_error[-1]),
        float(np.mean(circular_heading_error, dtype=np.float64)),
    )


def _summary(
    values: Sequence[Tuple[float, float, float]],
) -> MetricSummary:
    count = len(values)
    if count == 0:
        return MetricSummary(0, None, None, None)
    return MetricSummary(
        count,
        math.fsum(value[0] for value in values) / count,
        math.fsum(value[1] for value in values) / count,
        math.fsum(value[2] for value in values) / count,
    )


def compute_openloop_metrics(
    records: Sequence[PredictionRecord],
    *,
    run_id: str = VALIDATION_RUN_ID,
    drop_statistics: Optional[Mapping[str, int]] = None,
    sync_statistics: Optional[Mapping[str, Union[int, float]]] = None,
    identities: Optional[Mapping[str, str]] = None,
) -> ValidationReport:
    _nonempty_string(run_id, "run_id")
    unordered = tuple(records)
    if any(not isinstance(record, PredictionRecord) for record in unordered):
        raise TypeError("records must contain PredictionRecord instances")
    ordered = tuple(sorted(unordered, key=lambda record: record.sample_id))
    sample_ids = tuple(record.sample_id for record in ordered)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("records contain duplicate sample IDs")
    metrics = tuple(_selected_metrics(record) for record in ordered)
    micro = _summary(metrics)

    command_metrics = {
        command: _summary(
            tuple(
                value
                for record, value in zip(ordered, metrics)
                if record.command_name == command
            )
        )
        for command in COMMAND_NAMES
    }
    bag_values: Dict[str, list] = {}
    for record, value in zip(ordered, metrics):
        bag_values.setdefault(record.bag_sha256, []).append(value[0])
    bag_means = [
        math.fsum(bag_values[bag]) / len(bag_values[bag]) for bag in sorted(bag_values)
    ]
    bag_macro = None if not bag_means else math.fsum(bag_means) / len(bag_means)
    count = len(ordered)
    oracle_min_ade = (
        None
        if count == 0
        else math.fsum(record.min_ade_32 for record in ordered) / count
    )
    oracle_min_fde = (
        None
        if count == 0
        else math.fsum(record.min_fde_32 for record in ordered) / count
    )
    return ValidationReport(
        domain_id=DOMAIN_ID,
        command_source=COMMAND_SOURCE,
        run_id=run_id,
        micro=micro,
        oracle_min_ade_32=oracle_min_ade,
        oracle_min_fde_32=oracle_min_fde,
        oracle_candidate_source=CANDIDATE_SOURCE,
        oracle_candidate_count=CANDIDATE_COUNT,
        bag_macro_ade=bag_macro,
        per_command=command_metrics,
        records=ordered,
        drop_statistics=_copy_drop_statistics(drop_statistics),
        sync_statistics=_copy_sync_statistics(sync_statistics),
        identities=_copy_identities(identities),
    )


def _metric_json(summary: MetricSummary) -> Dict[str, Any]:
    return {
        "count": summary.count,
        "ade": summary.ade,
        "fde": summary.fde,
        "heading_mae": summary.heading_mae,
    }


def _record_json(record: PredictionRecord) -> Dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "bag_sha256": record.bag_sha256,
        "command_name": record.command_name,
        "timestamp_ns": record.timestamp_ns,
        "seed": record.seed,
        "prediction": record.prediction.tolist(),
        "target": record.target.tolist(),
        "candidate_source": record.candidate_source,
        "candidate_count": record.candidate_count,
        "min_ade_32": record.min_ade_32,
        "min_fde_32": record.min_fde_32,
        "argmin_ade_rank": record.argmin_ade_rank,
        "argmin_fde_rank": record.argmin_fde_rank,
    }


def validation_report_json_bytes(report: ValidationReport) -> bytes:
    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    value = {
        "domain_id": report.domain_id,
        "command_source": report.command_source,
        "run_id": report.run_id,
        "micro": _metric_json(report.micro),
        "oracle_min_ade_32": report.oracle_min_ade_32,
        "oracle_min_fde_32": report.oracle_min_fde_32,
        "oracle_candidate_source": report.oracle_candidate_source,
        "oracle_candidate_count": report.oracle_candidate_count,
        "bag_macro_ade": report.bag_macro_ade,
        "per_command": {
            name: _metric_json(report.per_command[name]) for name in COMMAND_NAMES
        },
        "records": [_record_json(record) for record in report.records],
        "drop_statistics": dict(report.drop_statistics),
        "sync_statistics": dict(report.sync_statistics),
        "identities": dict(report.identities),
    }
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = (
    "BestOf32Metrics",
    "CANDIDATE_COUNT",
    "CANDIDATE_SOURCE",
    "COMMAND_NAMES",
    "COMMAND_SOURCE",
    "DOMAIN_ID",
    "MetricSummary",
    "PredictionRecord",
    "VALIDATION_RUN_ID",
    "ValidationReport",
    "bezier_headings",
    "compute_best_of_32_metrics",
    "compute_openloop_metrics",
    "reset_torch_seed",
    "selector_seed",
    "validation_loss_seed",
    "validation_report_json_bytes",
)
