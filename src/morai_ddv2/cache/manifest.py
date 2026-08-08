from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, TypedDict, Union

import h5py
import numpy as np

from morai_ddv2.cache.hdf5 import BagArtifactRecord, verify_bag_artifact
from morai_ddv2.cache.sidecar import (
    CommandSidecarIdentity,
    read_command_sidecar_bytes,
)
from morai_ddv2.config import canonical_json_bytes, sha256_bytes
from morai_ddv2.errors import ConfigurationError, IntegrityError
from morai_ddv2.preprocess.planner import SampleDropReason
from morai_ddv2.ros1.messages import (
    BagDisposition,
    BagRejectReason,
    EpochSafeTfEgoDiagnostic,
)

from .domain import (
    DomainSnapshotRecord,
    _atomic_publish_bytes,
    _cache_regular_identity,
    _open_cache_regular_stream,
    _read_cache_bytes,
    _verify_domain_snapshot_record,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_INT64_MAX = (1 << 63) - 1
_COMMAND_NAMES = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")


class DirtyOverrideJson(TypedDict):
    allow_dirty_development: Literal[True]
    source_code_commit: str
    dirty_diff_sha256: str


class DatasetManifestRecord(TypedDict):
    record_type: Literal["dataset"]
    schema_version: Literal[2]
    domain_id: str
    resolved_domain_path: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    source_code_commit: str
    contract_revision_context: Literal["pre_first_production_unpublished"]
    after_first_production_publication: Literal["new_domain_id_required"]
    split_seed_prefix: Literal["20260806:"]
    train_relative_bag_paths: List[str]
    validation_relative_bag_paths: List[str]
    dirty_override: Optional[DirtyOverrideJson]


class BagManifestRecord(TypedDict):
    record_type: Literal["bag"]
    relative_bag_path: str
    source_bag_sha256: str
    disposition: Literal["ELIGIBLE", "SENSOR_ONLY_EXCLUDED", "REJECTED"]
    split: Optional[Literal["train", "validation"]]
    hdf5_path: Optional[str]
    hdf5_sha256: Optional[str]
    sample_count: int
    sidecar_path: Optional[str]
    sidecar_sha256: Optional[str]
    sidecar_record_count: int
    drop_counts: Dict[str, int]
    rejection_code: Optional[str]
    tf_ego_diagnostic_report_path: Optional[str]
    tf_ego_diagnostic_report_sha256: Optional[str]
    tf_ego_raw_record_count: int
    tf_ego_epoch_local_comparable_count: int
    tf_ego_no_epoch_local_bracket_count: int
    tf_ego_ambiguous_epoch_selection_count: int
    tf_ego_cross_epoch_bracket_prevented_count: int


class SampleManifestRecord(TypedDict):
    record_type: Literal["sample"]
    sample_id: str
    relative_bag_path: str
    source_bag_sha256: str
    split: Literal["train", "validation"]
    hdf5_path: str
    hdf5_row_index: int
    timestamp_ns: int
    grid_receipt_ns: int
    sensor_source_header_stamp_ns: List[int]
    sensor_source_receipt_ns: List[int]
    sensor_source_record_index: List[int]
    side_camera_header_offset_ns: List[int]
    sensor_receipt_offset_ns: List[int]
    trajectory_anchor_tf_header_ns: int
    trajectory_anchor_tf_receipt_ns: int
    trajectory_anchor_tf_record_index: int
    trajectory_anchor_tf_clock_epoch_index: int
    ego_bracket_clock_epoch_index: int
    ego_bracket_header_stamp_ns: List[int]
    ego_bracket_receipt_ns: List[int]
    ego_bracket_record_index: List[int]
    ego_interp_weight: float
    current_pose_bracket_header_stamp_ns: List[int]
    current_pose_bracket_receipt_ns: List[int]
    current_pose_bracket_record_index: List[int]
    current_pose_interp_weight: float
    future_pose_bracket_header_stamp_ns: List[List[int]]
    future_pose_bracket_receipt_ns: List[List[int]]
    future_pose_bracket_record_index: List[List[int]]
    future_pose_interp_weight: List[float]
    sidecar_path: str
    sidecar_row_index: int
    command_index: Literal[0, 1, 2, 3]
    command_name: Literal["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]
    command_provenance: Literal["future_trajectory_oracle"]


ManifestJsonRecord = Union[
    DatasetManifestRecord,
    BagManifestRecord,
    SampleManifestRecord,
]


DATASET_RECORD_FIELDS = (
    "record_type",
    "schema_version",
    "domain_id",
    "resolved_domain_path",
    "resolved_domain_sha256",
    "sensor_preset_sha256",
    "source_code_commit",
    "contract_revision_context",
    "after_first_production_publication",
    "split_seed_prefix",
    "train_relative_bag_paths",
    "validation_relative_bag_paths",
    "dirty_override",
)
BAG_RECORD_FIELDS = (
    "record_type",
    "relative_bag_path",
    "source_bag_sha256",
    "disposition",
    "split",
    "hdf5_path",
    "hdf5_sha256",
    "sample_count",
    "sidecar_path",
    "sidecar_sha256",
    "sidecar_record_count",
    "drop_counts",
    "rejection_code",
    "tf_ego_diagnostic_report_path",
    "tf_ego_diagnostic_report_sha256",
    "tf_ego_raw_record_count",
    "tf_ego_epoch_local_comparable_count",
    "tf_ego_no_epoch_local_bracket_count",
    "tf_ego_ambiguous_epoch_selection_count",
    "tf_ego_cross_epoch_bracket_prevented_count",
)
SAMPLE_RECORD_FIELDS = (
    "record_type",
    "sample_id",
    "relative_bag_path",
    "source_bag_sha256",
    "split",
    "hdf5_path",
    "hdf5_row_index",
    "timestamp_ns",
    "grid_receipt_ns",
    "sensor_source_header_stamp_ns",
    "sensor_source_receipt_ns",
    "sensor_source_record_index",
    "side_camera_header_offset_ns",
    "sensor_receipt_offset_ns",
    "trajectory_anchor_tf_header_ns",
    "trajectory_anchor_tf_receipt_ns",
    "trajectory_anchor_tf_record_index",
    "trajectory_anchor_tf_clock_epoch_index",
    "ego_bracket_clock_epoch_index",
    "ego_bracket_header_stamp_ns",
    "ego_bracket_receipt_ns",
    "ego_bracket_record_index",
    "ego_interp_weight",
    "current_pose_bracket_header_stamp_ns",
    "current_pose_bracket_receipt_ns",
    "current_pose_bracket_record_index",
    "current_pose_interp_weight",
    "future_pose_bracket_header_stamp_ns",
    "future_pose_bracket_receipt_ns",
    "future_pose_bracket_record_index",
    "future_pose_interp_weight",
    "sidecar_path",
    "sidecar_row_index",
    "command_index",
    "command_name",
    "command_provenance",
)
DIAGNOSTIC_PROVENANCE_FIELDS = (
    "ego_header_stamp_ns",
    "ego_receipt_stamp_ns",
    "ego_stable_record_index",
    "selected_tf_clock_epoch_index",
    "selected_tf_anchor_record_index",
    "tf_bracket_header_stamp_ns",
    "tf_bracket_receipt_ns",
    "tf_bracket_record_index",
    "exclusion_reason",
)
DIAGNOSTIC_REPORT_FIELDS = (
    "record_type",
    "schema_version",
    "domain_id",
    "resolved_domain_sha256",
    "sensor_preset_sha256",
    "relative_bag_path",
    "source_bag_sha256",
    "comparable_count",
    "excluded_count",
    "position_p99_m",
    "position_max_m",
    "heading_p99_deg",
    "heading_max_deg",
    "raw_ego_record_count",
    "epoch_local_comparable_count",
    "no_epoch_local_bracket_count",
    "ambiguous_epoch_selection_count",
    "cross_epoch_bracket_prevented_count",
    "provenance",
)
PROVENANCE_DATASET_FIELDS = SAMPLE_RECORD_FIELDS[7:31]


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be 64 lowercase hexadecimal characters".format(name))
    return value


def _commit(value: Any, name: str) -> str:
    if type(value) is not str or _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be 40 lowercase hexadecimal characters".format(name))
    return value


def _relative_posix_path(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError("{} must be a string".format(name))
    if not value or value in (".", "..") or "\x00" in value or "\\" in value:
        raise ValueError("{} must be a normalized relative POSIX path".format(name))
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("{} must be valid Unicode".format(name)) from error
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("{} must be a normalized relative POSIX path".format(name))
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("{} contains an unsafe path segment".format(name))
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError("{} must be an actual Python int".format(name))
    if value < 0 or value > _INT64_MAX:
        raise ValueError("{} must be a nonnegative signed-int64 value".format(name))
    return value


@dataclass(frozen=True)
class DirtyOverride:
    allow_dirty_development: bool
    source_code_commit: str
    dirty_diff_sha256: str

    def __post_init__(self) -> None:
        if self.allow_dirty_development is not True:
            raise ValueError("allow_dirty_development must be exactly True")
        object.__setattr__(
            self, "source_code_commit", _commit(self.source_code_commit, "source_code_commit")
        )
        object.__setattr__(
            self, "dirty_diff_sha256", _sha256(self.dirty_diff_sha256, "dirty_diff_sha256")
        )

    def as_json(self) -> DirtyOverrideJson:
        return {
            "allow_dirty_development": True,
            "source_code_commit": self.source_code_commit,
            "dirty_diff_sha256": self.dirty_diff_sha256,
        }


@dataclass(frozen=True)
class BagManifestInput:
    relative_bag_path: str
    source_bag_sha256: str
    disposition: BagDisposition
    artifact: Optional[BagArtifactRecord]
    drop_counts: Mapping[str, int]
    rejection_code: Optional[BagRejectReason]
    tf_ego_diagnostic: Optional[EpochSafeTfEgoDiagnostic]

    def __post_init__(self) -> None:
        relative = _relative_posix_path(self.relative_bag_path, "relative_bag_path")
        source_sha = _sha256(self.source_bag_sha256, "source_bag_sha256")
        if not isinstance(self.disposition, BagDisposition):
            raise TypeError("disposition must be a BagDisposition")
        expected_drop_names = tuple(reason.value for reason in SampleDropReason)
        if not isinstance(self.drop_counts, Mapping):
            raise TypeError("drop_counts must be a mapping")
        if set(self.drop_counts) != set(expected_drop_names):
            raise ValueError("drop_counts must contain every sample drop reason exactly")
        drops = {
            name: _nonnegative_int(self.drop_counts[name], "drop_counts.{}".format(name))
            for name in expected_drop_names
        }
        object.__setattr__(self, "drop_counts", MappingProxyType(drops))
        if self.disposition is BagDisposition.ELIGIBLE:
            if type(self.artifact) is not BagArtifactRecord:
                raise TypeError("eligible bag requires a BagArtifactRecord")
            artifact = BagArtifactRecord(**self.artifact.__dict__)
            if (
                artifact.relative_bag_path != relative
                or artifact.source_bag_sha256 != source_sha
            ):
                raise ValueError("eligible artifact identity does not match bag input")
            object.__setattr__(self, "artifact", artifact)
            if self.rejection_code is not None:
                raise ValueError("eligible bag cannot have a rejection code")
            if not isinstance(self.tf_ego_diagnostic, EpochSafeTfEgoDiagnostic):
                raise TypeError("eligible GT-rich bag requires a TF/Ego diagnostic")
        else:
            if self.artifact is not None:
                raise ValueError("non-eligible bag cannot have an artifact")
            if self.disposition is BagDisposition.REJECTED:
                if not isinstance(self.rejection_code, BagRejectReason):
                    raise TypeError("rejected bag requires a BagRejectReason")
            elif self.rejection_code is not None:
                raise ValueError("sensor-only excluded bag cannot have a rejection code")
            if (
                self.disposition is BagDisposition.SENSOR_ONLY_EXCLUDED
                and self.tf_ego_diagnostic is not None
            ):
                raise ValueError("sensor-only excluded bag cannot have a GT diagnostic")
            if self.tf_ego_diagnostic is not None and not isinstance(
                self.tf_ego_diagnostic, EpochSafeTfEgoDiagnostic
            ):
                raise TypeError("tf_ego_diagnostic must be epoch-safe")


@dataclass(frozen=True)
class RejectionReport:
    bags: Tuple[BagManifestInput, ...]

    def __post_init__(self) -> None:
        if type(self.bags) is not tuple or not self.bags:
            raise ValueError("rejection report must contain at least one bag")
        if any(type(item) is not BagManifestInput for item in self.bags):
            raise TypeError("rejection report bags must be BagManifestInput values")
        paths = [item.relative_bag_path for item in self.bags]
        if len(paths) != len(set(paths)):
            raise ValueError("rejection report contains duplicate relative bag paths")
        eligible_hashes = [
            item.source_bag_sha256
            for item in self.bags
            if item.disposition is BagDisposition.ELIGIBLE
        ]
        if len(eligible_hashes) != len(set(eligible_hashes)):
            raise ValueError("eligible bags contain duplicate source hashes")


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    sha256: str
    bag_count: int
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_posix_path(self.path, "path"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        _nonnegative_int(self.bag_count, "bag_count")
        _nonnegative_int(self.sample_count, "sample_count")


def relative_bag_path(bag_root: Path, bag_path: Path) -> str:
    if not isinstance(bag_root, Path) or not isinstance(bag_path, Path):
        raise TypeError("bag_root and bag_path must be pathlib.Path values")
    try:
        resolved_root = bag_root.resolve(strict=True)
        resolved_bag = bag_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("bag root and bag path must exist") from error
    if not resolved_root.is_dir():
        raise ValueError("bag_root must resolve to a directory")
    if not resolved_bag.is_file():
        raise ValueError("bag_path must resolve to a regular file")
    try:
        relative = resolved_bag.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError("bag_path resolves outside bag_root") from error
    return _relative_posix_path(relative, "relative_bag_path")


def _split_key(relative_path: str) -> Tuple[bytes, bytes]:
    path_bytes = relative_path.encode("utf-8", errors="strict")
    return hashlib.sha256(b"20260806:" + path_bytes).digest(), path_bytes


def assign_bag_splits(
    relative_paths: Sequence[str],
) -> Mapping[str, Literal["train", "validation"]]:
    if isinstance(relative_paths, (str, bytes)) or not isinstance(
        relative_paths, Sequence
    ):
        raise TypeError("relative_paths must be a sequence of strings")
    validated = [
        _relative_posix_path(path, "relative_paths[{}]".format(index))
        for index, path in enumerate(relative_paths)
    ]
    if len(validated) != len(set(validated)):
        raise ValueError("relative_paths contains a duplicate path")
    ordered = sorted(validated, key=_split_key)
    train_count = math.floor(0.8 * len(ordered))
    result = {
        path: ("train" if index < train_count else "validation")
        for index, path in enumerate(ordered)
    }
    return MappingProxyType(result)


def _json_bytes(value: Mapping[str, Any], *, newline: bool) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise IntegrityError("record cannot be deterministically encoded") from error
    return payload + (b"\n" if newline else b"")


def _publish_immutable(
    cache_root: Path, relative_path: str, payload: bytes, name: str
) -> None:
    path = cache_root / relative_path
    if os.path.lexists(str(path)):
        existing, _ = _read_cache_bytes(cache_root, relative_path, name)
        if existing != payload:
            raise IntegrityError("existing {} bytes conflict".format(name))
        return
    _atomic_publish_bytes(cache_root, relative_path, payload)
    published, _ = _read_cache_bytes(cache_root, relative_path, name)
    if published != payload:
        raise IntegrityError("published {} bytes mismatch".format(name))


def _validate_diagnostic_structure(value: Mapping[str, Any]) -> None:
    for name in (
        "comparable_count",
        "excluded_count",
        "raw_ego_record_count",
        "epoch_local_comparable_count",
        "no_epoch_local_bracket_count",
        "ambiguous_epoch_selection_count",
        "cross_epoch_bracket_prevented_count",
    ):
        try:
            _nonnegative_int(value[name], "diagnostic.{}".format(name))
        except (TypeError, ValueError) as error:
            raise IntegrityError(
                "TF/Ego diagnostic counter {} is invalid".format(name)
            ) from error
    for name in (
        "position_p99_m",
        "position_max_m",
        "heading_p99_deg",
        "heading_max_deg",
    ):
        metric = value[name]
        if type(metric) is not float or not math.isfinite(metric) or metric < 0.0:
            raise IntegrityError(
                "TF/Ego diagnostic metric {} is invalid".format(name)
            )
    if value["comparable_count"] != value["epoch_local_comparable_count"]:
        raise IntegrityError("TF/Ego diagnostic comparable counters differ")
    excluded_count = (
        value["no_epoch_local_bracket_count"]
        + value["ambiguous_epoch_selection_count"]
        + value["cross_epoch_bracket_prevented_count"]
    )
    if value["excluded_count"] != excluded_count:
        raise IntegrityError("TF/Ego diagnostic excluded counters differ")
    if value["comparable_count"] + value["excluded_count"] != value[
        "raw_ego_record_count"
    ]:
        raise IntegrityError("TF/Ego diagnostic counters do not reconcile")
    if value["position_p99_m"] > value["position_max_m"]:
        raise IntegrityError("TF/Ego diagnostic position p99 exceeds its maximum")
    if value["heading_p99_deg"] > value["heading_max_deg"]:
        raise IntegrityError("TF/Ego diagnostic heading p99 exceeds its maximum")

    provenance = value["provenance"]
    if type(provenance) is not list or any(
        type(row) is not dict or tuple(row) != DIAGNOSTIC_PROVENANCE_FIELDS
        for row in provenance
    ):
        raise IntegrityError("TF/Ego diagnostic provenance schema mismatch")
    allowed_reasons = {
        "NO_EPOCH_LOCAL_BRACKET",
        "AMBIGUOUS_EPOCH_SELECTION",
        "CROSS_EPOCH_BRACKET_PREVENTED",
    }
    for row_index, row in enumerate(provenance):
        for name in (
            "ego_header_stamp_ns",
            "ego_receipt_stamp_ns",
            "ego_stable_record_index",
            "selected_tf_clock_epoch_index",
            "selected_tf_anchor_record_index",
        ):
            try:
                _nonnegative_int(
                    row[name],
                    "diagnostic.provenance[{}].{}".format(row_index, name),
                )
            except (TypeError, ValueError) as error:
                raise IntegrityError(
                    "TF/Ego diagnostic provenance integer is invalid"
                ) from error
        reason = row["exclusion_reason"]
        brackets = (
            row["tf_bracket_header_stamp_ns"],
            row["tf_bracket_receipt_ns"],
            row["tf_bracket_record_index"],
        )
        if reason is None:
            if any(bracket is None for bracket in brackets):
                raise IntegrityError(
                    "TF/Ego diagnostic comparable row lacks bracket provenance"
                )
        else:
            if reason not in allowed_reasons:
                raise IntegrityError(
                    "TF/Ego diagnostic exclusion reason is invalid"
                )
            if any(bracket is not None for bracket in brackets):
                raise IntegrityError(
                    "TF/Ego diagnostic excluded row carries a bracket"
                )
        if reason is None:
            for name, bracket in zip(
                (
                    "tf_bracket_header_stamp_ns",
                    "tf_bracket_receipt_ns",
                    "tf_bracket_record_index",
                ),
                brackets,
            ):
                if type(bracket) is not list or len(bracket) != 2:
                    raise IntegrityError(
                        "TF/Ego diagnostic {} must be a pair".format(name)
                    )
                if any(
                    type(item) is not int or item < 0 or item > _INT64_MAX
                    for item in bracket
                ):
                    raise IntegrityError(
                        "TF/Ego diagnostic {} contains an invalid integer".format(
                            name
                        )
                    )
                if bracket[0] > bracket[1]:
                    raise IntegrityError(
                        "TF/Ego diagnostic {} is not source ordered".format(name)
                    )
    indices = [row["ego_stable_record_index"] for row in provenance]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise IntegrityError("TF/Ego diagnostic provenance is not uniquely sorted")
    if len(provenance) != value["raw_ego_record_count"]:
        raise IntegrityError("TF/Ego diagnostic provenance count mismatch")
    reasons = [row["exclusion_reason"] for row in provenance]
    expected_reason_counts = (
        sum(reason is None for reason in reasons),
        reasons.count("NO_EPOCH_LOCAL_BRACKET"),
        reasons.count("AMBIGUOUS_EPOCH_SELECTION"),
        reasons.count("CROSS_EPOCH_BRACKET_PREVENTED"),
    )
    actual_reason_counts = (
        value["epoch_local_comparable_count"],
        value["no_epoch_local_bracket_count"],
        value["ambiguous_epoch_selection_count"],
        value["cross_epoch_bracket_prevented_count"],
    )
    if expected_reason_counts != actual_reason_counts:
        raise IntegrityError("TF/Ego diagnostic exclusion counters mismatch")


def _diagnostic_mapping(
    *,
    diagnostic: EpochSafeTfEgoDiagnostic,
    domain_snapshot: DomainSnapshotRecord,
    relative_path: str,
    source_sha: str,
) -> Dict[str, Any]:
    provenance = []
    ordered = sorted(
        diagnostic.provenance,
        key=lambda item: item.ego_stable_record_index,
    )
    indices = [item.ego_stable_record_index for item in ordered]
    if len(indices) != len(set(indices)):
        raise IntegrityError("TF/Ego diagnostic contains duplicate Ego stable indices")
    for item in ordered:
        row = {
            "ego_header_stamp_ns": int(item.ego_header_stamp_ns),
            "ego_receipt_stamp_ns": int(item.ego_receipt_stamp_ns),
            "ego_stable_record_index": int(item.ego_stable_record_index),
            "selected_tf_clock_epoch_index": int(
                item.selected_tf_clock_epoch_index
            ),
            "selected_tf_anchor_record_index": int(
                item.selected_tf_anchor_record_index
            ),
            "tf_bracket_header_stamp_ns": (
                None
                if item.tf_bracket_header_stamp_ns is None
                else list(item.tf_bracket_header_stamp_ns)
            ),
            "tf_bracket_receipt_ns": (
                None
                if item.tf_bracket_receipt_ns is None
                else list(item.tf_bracket_receipt_ns)
            ),
            "tf_bracket_record_index": (
                None
                if item.tf_bracket_record_index is None
                else list(item.tf_bracket_record_index)
            ),
            "exclusion_reason": item.exclusion_reason,
        }
        if tuple(row) != DIAGNOSTIC_PROVENANCE_FIELDS:
            raise AssertionError("diagnostic provenance field order drift")
        provenance.append(row)
    metric = diagnostic.metric
    values = (
        metric.position_p99_m,
        metric.position_max_m,
        metric.heading_p99_deg,
        metric.heading_max_deg,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise IntegrityError("TF/Ego diagnostic metrics must be finite and nonnegative")
    result = {
        "record_type": "tf_ego_diagnostic",
        "schema_version": 2,
        "domain_id": domain_snapshot.domain_id,
        "resolved_domain_sha256": domain_snapshot.resolved_domain_sha256,
        "sensor_preset_sha256": domain_snapshot.sensor_preset_sha256,
        "relative_bag_path": relative_path,
        "source_bag_sha256": source_sha,
        "comparable_count": int(metric.comparable_count),
        "excluded_count": int(metric.excluded_count),
        "position_p99_m": float(metric.position_p99_m),
        "position_max_m": float(metric.position_max_m),
        "heading_p99_deg": float(metric.heading_p99_deg),
        "heading_max_deg": float(metric.heading_max_deg),
        "raw_ego_record_count": diagnostic.raw_ego_record_count,
        "epoch_local_comparable_count": diagnostic.epoch_local_comparable_count,
        "no_epoch_local_bracket_count": diagnostic.no_epoch_local_bracket_count,
        "ambiguous_epoch_selection_count": diagnostic.ambiguous_epoch_selection_count,
        "cross_epoch_bracket_prevented_count": diagnostic.cross_epoch_bracket_prevented_count,
        "provenance": provenance,
    }
    if tuple(result) != DIAGNOSTIC_REPORT_FIELDS:
        raise AssertionError("diagnostic report field order drift")
    _validate_diagnostic_structure(result)
    return result


def _enforce_eligible_diagnostic_gate(
    diagnostic_mapping: Mapping[str, Any], domain: Any
) -> None:
    try:
        policy = domain.document["diagnostics"]
        minimum = policy["tf_ego_minimum_comparable_pairs"]
        position_tolerance = policy["tf_ego_position_p99_tolerance_m"]
        heading_tolerance = policy["tf_ego_heading_p99_tolerance_deg"]
    except (AttributeError, KeyError, TypeError) as error:
        raise ConfigurationError("resolved TF/Ego diagnostic gate is missing") from error
    if (
        type(minimum) is not int
        or minimum != 10
        or type(position_tolerance) is not float
        or position_tolerance != 0.051
        or type(heading_tolerance) is not float
        or heading_tolerance != 0.5
    ):
        raise ConfigurationError("resolved TF/Ego diagnostic gate contract drift")
    if diagnostic_mapping["comparable_count"] < minimum:
        raise IntegrityError("eligible TF/Ego diagnostic has fewer than 10 comparable pairs")
    if diagnostic_mapping["position_p99_m"] > position_tolerance:
        raise IntegrityError("eligible TF/Ego diagnostic position p99 exceeds 0.051 m")
    if diagnostic_mapping["heading_p99_deg"] > heading_tolerance:
        raise IntegrityError("eligible TF/Ego diagnostic heading p99 exceeds 0.5 deg")


def _publish_diagnostic(
    *,
    cache_root: Path,
    domain_snapshot: DomainSnapshotRecord,
    bag: BagManifestInput,
    domain: Any,
    enforce_gate: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    diagnostic = bag.tf_ego_diagnostic
    if diagnostic is None:
        raise IntegrityError("diagnostic publication requires an epoch-safe report")
    mapping = _diagnostic_mapping(
        diagnostic=diagnostic,
        domain_snapshot=domain_snapshot,
        relative_path=bag.relative_bag_path,
        source_sha=bag.source_bag_sha256,
    )
    if enforce_gate:
        _enforce_eligible_diagnostic_gate(mapping, domain)
    relative = "diagnostics/{}/{}/{}.json".format(
        domain_snapshot.domain_id,
        domain_snapshot.resolved_domain_sha256,
        bag.source_bag_sha256,
    )
    payload = _json_bytes(mapping, newline=False)
    _publish_immutable(
        cache_root, relative, payload, "TF/Ego diagnostic report"
    )
    published, _ = _read_cache_bytes(
        cache_root, relative, "TF/Ego diagnostic report"
    )
    if published != payload:
        raise IntegrityError("TF/Ego diagnostic report verification failed")
    return relative, sha256_bytes(published), mapping


def _command_rule_sha256(domain: Any) -> str:
    try:
        command = domain.document["command"]
        subtree = {
            "index_mapping": command["index_mapping"],
            "thresholds": command["thresholds"],
            "ordered_rules": command["ordered_rules"],
        }
        return sha256_bytes(canonical_json_bytes(subtree))
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("resolved command rule is invalid") from error


def _plain_hdf5_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        scalar = array.item()
        return float(scalar) if isinstance(scalar, float) else int(scalar)
    return array.tolist()


@contextmanager
def _open_manifest_hdf5(
    cache_root: Path, artifact: BagArtifactRecord
) -> Iterator[h5py.File]:
    try:
        with _open_cache_regular_stream(
            cache_root,
            artifact.hdf5_path,
            "HDF5 artifact",
            expected_sha256=artifact.hdf5_sha256,
        ) as stream:
            with h5py.File(stream, "r") as handle:
                yield handle
    except IntegrityError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise IntegrityError("unable to open manifest HDF5 artifact") from error


def _sample_records(
    *,
    cache_root: Path,
    artifact: BagArtifactRecord,
    split: str,
    domain: Any,
) -> List[SampleManifestRecord]:
    identity = CommandSidecarIdentity(
        schema_version=2,
        domain_id=artifact.domain_id,
        resolved_domain_sha256=artifact.resolved_domain_sha256,
        sensor_preset_sha256=artifact.sensor_preset_sha256,
        source_bag_sha256=artifact.source_bag_sha256,
        command_rule_sha256=_command_rule_sha256(domain),
        provenance="future_trajectory_oracle",
    )
    sidecar_payload, _ = _read_cache_bytes(
        cache_root, artifact.sidecar_path, "command sidecar"
    )
    if hashlib.sha256(sidecar_payload).hexdigest() != artifact.sidecar_sha256:
        raise IntegrityError("opened command sidecar bytes differ from artifact SHA-256")
    commands = tuple(read_command_sidecar_bytes(sidecar_payload, identity).values())
    result: List[SampleManifestRecord] = []
    try:
        with _open_manifest_hdf5(cache_root, artifact) as handle:
            for row_index, command in enumerate(commands):
                values = {
                    name: _plain_hdf5_value(handle[name][row_index])
                    for name in PROVENANCE_DATASET_FIELDS
                }
                record: SampleManifestRecord = {
                    "record_type": "sample",
                    "sample_id": "{}:{}".format(
                        artifact.source_bag_sha256, command.timestamp_ns
                    ),
                    "relative_bag_path": artifact.relative_bag_path,
                    "source_bag_sha256": artifact.source_bag_sha256,
                    "split": split,
                    "hdf5_path": artifact.hdf5_path,
                    "hdf5_row_index": row_index,
                    "timestamp_ns": values["timestamp_ns"],
                    "grid_receipt_ns": values["grid_receipt_ns"],
                    "sensor_source_header_stamp_ns": values[
                        "sensor_source_header_stamp_ns"
                    ],
                    "sensor_source_receipt_ns": values[
                        "sensor_source_receipt_ns"
                    ],
                    "sensor_source_record_index": values[
                        "sensor_source_record_index"
                    ],
                    "side_camera_header_offset_ns": values[
                        "side_camera_header_offset_ns"
                    ],
                    "sensor_receipt_offset_ns": values[
                        "sensor_receipt_offset_ns"
                    ],
                    "trajectory_anchor_tf_header_ns": values[
                        "trajectory_anchor_tf_header_ns"
                    ],
                    "trajectory_anchor_tf_receipt_ns": values[
                        "trajectory_anchor_tf_receipt_ns"
                    ],
                    "trajectory_anchor_tf_record_index": values[
                        "trajectory_anchor_tf_record_index"
                    ],
                    "trajectory_anchor_tf_clock_epoch_index": values[
                        "trajectory_anchor_tf_clock_epoch_index"
                    ],
                    "ego_bracket_clock_epoch_index": values[
                        "ego_bracket_clock_epoch_index"
                    ],
                    "ego_bracket_header_stamp_ns": values[
                        "ego_bracket_header_stamp_ns"
                    ],
                    "ego_bracket_receipt_ns": values["ego_bracket_receipt_ns"],
                    "ego_bracket_record_index": values["ego_bracket_record_index"],
                    "ego_interp_weight": values["ego_interp_weight"],
                    "current_pose_bracket_header_stamp_ns": values[
                        "current_pose_bracket_header_stamp_ns"
                    ],
                    "current_pose_bracket_receipt_ns": values[
                        "current_pose_bracket_receipt_ns"
                    ],
                    "current_pose_bracket_record_index": values[
                        "current_pose_bracket_record_index"
                    ],
                    "current_pose_interp_weight": values[
                        "current_pose_interp_weight"
                    ],
                    "future_pose_bracket_header_stamp_ns": values[
                        "future_pose_bracket_header_stamp_ns"
                    ],
                    "future_pose_bracket_receipt_ns": values[
                        "future_pose_bracket_receipt_ns"
                    ],
                    "future_pose_bracket_record_index": values[
                        "future_pose_bracket_record_index"
                    ],
                    "future_pose_interp_weight": values[
                        "future_pose_interp_weight"
                    ],
                    "sidecar_path": artifact.sidecar_path,
                    "sidecar_row_index": row_index,
                    "command_index": command.command_index,
                    "command_name": command.command_name,
                    "command_provenance": command.provenance,
                }
                if tuple(record) != SAMPLE_RECORD_FIELDS:
                    raise AssertionError("sample manifest field order drift")
                result.append(record)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise IntegrityError("unable to materialize manifest sample records") from error
    if len(result) != artifact.sample_count:
        raise IntegrityError("manifest sample count differs from bag artifact")
    return result


def _bag_record(
    *,
    bag: BagManifestInput,
    split: Optional[str],
    diagnostic_path: Optional[str],
    diagnostic_sha: Optional[str],
    diagnostic_mapping: Optional[Mapping[str, Any]],
) -> BagManifestRecord:
    artifact = bag.artifact
    if artifact is None:
        hdf5_path = None
        hdf5_sha = None
        sample_count = 0
        sidecar_path = None
        sidecar_sha = None
        sidecar_count = 0
    else:
        hdf5_path = artifact.hdf5_path
        hdf5_sha = artifact.hdf5_sha256
        sample_count = artifact.sample_count
        sidecar_path = artifact.sidecar_path
        sidecar_sha = artifact.sidecar_sha256
        sidecar_count = artifact.sidecar_record_count
    if diagnostic_mapping is None:
        diagnostic_counts = (0, 0, 0, 0, 0)
    else:
        diagnostic_counts = (
            diagnostic_mapping["raw_ego_record_count"],
            diagnostic_mapping["epoch_local_comparable_count"],
            diagnostic_mapping["no_epoch_local_bracket_count"],
            diagnostic_mapping["ambiguous_epoch_selection_count"],
            diagnostic_mapping["cross_epoch_bracket_prevented_count"],
        )
    result: BagManifestRecord = {
        "record_type": "bag",
        "relative_bag_path": bag.relative_bag_path,
        "source_bag_sha256": bag.source_bag_sha256,
        "disposition": bag.disposition.value,
        "split": split,
        "hdf5_path": hdf5_path,
        "hdf5_sha256": hdf5_sha,
        "sample_count": sample_count,
        "sidecar_path": sidecar_path,
        "sidecar_sha256": sidecar_sha,
        "sidecar_record_count": sidecar_count,
        "drop_counts": dict(bag.drop_counts),
        "rejection_code": (
            None if bag.rejection_code is None else bag.rejection_code.value
        ),
        "tf_ego_diagnostic_report_path": diagnostic_path,
        "tf_ego_diagnostic_report_sha256": diagnostic_sha,
        "tf_ego_raw_record_count": diagnostic_counts[0],
        "tf_ego_epoch_local_comparable_count": diagnostic_counts[1],
        "tf_ego_no_epoch_local_bracket_count": diagnostic_counts[2],
        "tf_ego_ambiguous_epoch_selection_count": diagnostic_counts[3],
        "tf_ego_cross_epoch_bracket_prevented_count": diagnostic_counts[4],
    }
    if tuple(result) != BAG_RECORD_FIELDS:
        raise AssertionError("bag manifest field order drift")
    return result


def publish_manifest(
    *,
    cache_root: Path,
    domain_snapshot: DomainSnapshotRecord,
    bags: Sequence[BagArtifactRecord],
    splits: Mapping[str, str],
    rejection_report: RejectionReport,
    dirty_override: Optional[DirtyOverride],
) -> ManifestRecord:
    if not isinstance(cache_root, Path):
        raise TypeError("cache_root must be a pathlib.Path")
    if type(domain_snapshot) is not DomainSnapshotRecord:
        raise TypeError("domain_snapshot must be a DomainSnapshotRecord")
    if isinstance(bags, (str, bytes)) or not isinstance(bags, Sequence):
        raise TypeError("bags must be a sequence of BagArtifactRecord values")
    if any(type(item) is not BagArtifactRecord for item in bags):
        raise TypeError("bags must contain only BagArtifactRecord values")
    if not isinstance(splits, Mapping):
        raise TypeError("splits must be a mapping")
    if type(rejection_report) is not RejectionReport:
        raise TypeError("rejection_report must be a RejectionReport")
    if dirty_override is not None and type(dirty_override) is not DirtyOverride:
        raise TypeError("dirty_override must be a DirtyOverride or None")

    domain = _verify_domain_snapshot_record(cache_root, domain_snapshot)
    eligible_inputs = tuple(
        item
        for item in rejection_report.bags
        if item.disposition is BagDisposition.ELIGIBLE
    )
    if not eligible_inputs:
        raise IntegrityError("manifest requires at least one eligible bag artifact")
    copied_artifacts = tuple(BagArtifactRecord(**item.__dict__) for item in bags)
    supplied = {
        (item.relative_bag_path, item.source_bag_sha256): item
        for item in copied_artifacts
    }
    if len(supplied) != len(copied_artifacts):
        raise IntegrityError("bag artifact sequence contains duplicate identities")
    expected = {
        (item.relative_bag_path, item.source_bag_sha256): item.artifact
        for item in eligible_inputs
    }
    if set(supplied) != set(expected) or any(
        supplied[key] != expected[key] for key in expected
    ):
        raise IntegrityError("eligible artifact inventory does not match report")

    eligible_paths = [item.relative_bag_path for item in eligible_inputs]
    expected_splits = assign_bag_splits(eligible_paths)
    try:
        actual_splits = {
            _relative_posix_path(path, "split path"): value
            for path, value in splits.items()
        }
    except (TypeError, ValueError) as error:
        raise IntegrityError("split mapping is invalid") from error
    if any(value not in ("train", "validation") for value in actual_splits.values()):
        raise IntegrityError("split mapping contains an invalid value")
    if actual_splits != dict(expected_splits):
        raise IntegrityError("split mapping differs from deterministic bag split")

    source_commits = {artifact.source_code_commit for artifact in copied_artifacts}
    if len(source_commits) != 1:
        raise IntegrityError("bag artifacts do not share one source code commit")
    source_commit = next(iter(source_commits))
    if dirty_override is not None and dirty_override.source_code_commit != source_commit:
        raise ValueError("dirty override source_code_commit differs from artifacts")
    for artifact in copied_artifacts:
        if (
            artifact.domain_id != domain_snapshot.domain_id
            or artifact.resolved_domain_sha256
            != domain_snapshot.resolved_domain_sha256
            or artifact.sensor_preset_sha256 != domain_snapshot.sensor_preset_sha256
        ):
            raise IntegrityError("bag artifact domain identity mismatch")
        _cache_regular_identity(cache_root, artifact.hdf5_path, "HDF5 artifact")
        _cache_regular_identity(cache_root, artifact.sidecar_path, "command sidecar")
        verify_bag_artifact(artifact, cache_root)

    ordered_inputs = sorted(
        rejection_report.bags,
        key=lambda item: item.relative_bag_path.encode("utf-8"),
    )
    bag_records: List[BagManifestRecord] = []
    samples: List[SampleManifestRecord] = []
    diagnostic_bindings = []
    for bag in ordered_inputs:
        diagnostic_path = None
        diagnostic_sha = None
        diagnostic_mapping = None
        if bag.tf_ego_diagnostic is not None:
            diagnostic_path, diagnostic_sha, diagnostic_mapping = _publish_diagnostic(
                cache_root=cache_root,
                domain_snapshot=domain_snapshot,
                bag=bag,
                domain=domain,
                enforce_gate=bag.disposition is BagDisposition.ELIGIBLE,
            )
            diagnostic_bindings.append((diagnostic_path, diagnostic_sha))
        split = actual_splits.get(bag.relative_bag_path)
        if bag.disposition is BagDisposition.ELIGIBLE:
            if split is None or bag.artifact is None:
                raise IntegrityError("eligible bag is missing split or artifact")
            samples.extend(
                _sample_records(
                    cache_root=cache_root,
                    artifact=bag.artifact,
                    split=split,
                    domain=domain,
                )
            )
        elif split is not None:
            raise IntegrityError("non-eligible bag must not have a split")
        bag_records.append(
            _bag_record(
                bag=bag,
                split=split,
                diagnostic_path=diagnostic_path,
                diagnostic_sha=diagnostic_sha,
                diagnostic_mapping=diagnostic_mapping,
            )
        )
    samples.sort(
        key=lambda item: (
            item["timestamp_ns"],
            item["relative_bag_path"].encode("utf-8"),
            item["hdf5_row_index"],
        )
    )
    sample_ids = [item["sample_id"] for item in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise IntegrityError("manifest sample IDs are not globally unique")

    ordered_split = list(expected_splits.items())
    lifecycle = domain.document["domain_lifecycle"]
    dataset: DatasetManifestRecord = {
        "record_type": "dataset",
        "schema_version": 2,
        "domain_id": domain_snapshot.domain_id,
        "resolved_domain_path": domain_snapshot.path,
        "resolved_domain_sha256": domain_snapshot.resolved_domain_sha256,
        "sensor_preset_sha256": domain_snapshot.sensor_preset_sha256,
        "source_code_commit": source_commit,
        "contract_revision_context": lifecycle["contract_revision_context"],
        "after_first_production_publication": lifecycle[
            "after_first_production_publication"
        ],
        "split_seed_prefix": "20260806:",
        "train_relative_bag_paths": [
            path for path, value in ordered_split if value == "train"
        ],
        "validation_relative_bag_paths": [
            path for path, value in ordered_split if value == "validation"
        ],
        "dirty_override": None if dirty_override is None else dirty_override.as_json(),
    }
    if tuple(dataset) != DATASET_RECORD_FIELDS:
        raise AssertionError("dataset manifest field order drift")

    for artifact in copied_artifacts:
        _cache_regular_identity(cache_root, artifact.hdf5_path, "HDF5 artifact")
        _cache_regular_identity(cache_root, artifact.sidecar_path, "command sidecar")
        verify_bag_artifact(artifact, cache_root)
    for relative, digest in diagnostic_bindings:
        published, _ = _read_cache_bytes(
            cache_root, relative, "TF/Ego diagnostic report"
        )
        if sha256_bytes(published) != digest:
            raise IntegrityError("TF/Ego diagnostic report hash changed")
    _verify_domain_snapshot_record(cache_root, domain_snapshot)

    records: List[Mapping[str, Any]] = [dataset]
    records.extend(bag_records)
    records.extend(samples)
    payload = b"".join(_json_bytes(record, newline=True) for record in records)
    relative = "manifests/{}/{}/manifest.jsonl".format(
        domain_snapshot.domain_id, domain_snapshot.resolved_domain_sha256
    )
    path = cache_root / relative
    if os.path.lexists(str(path)):
        raise FileExistsError("final manifest already exists")
    _atomic_publish_bytes(cache_root, relative, payload)
    published, _ = _read_cache_bytes(cache_root, relative, "dataset manifest")
    if published != payload:
        raise IntegrityError("published dataset manifest bytes mismatch")
    return ManifestRecord(
        path=relative,
        sha256=sha256_bytes(published),
        bag_count=len(bag_records),
        sample_count=len(samples),
    )


def _unique_json_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: {}".format(value))


def _verify_diagnostic_report(
    *,
    cache_root: Path,
    bag_record: Mapping[str, Any],
    domain_id: str,
    resolved_domain_sha256: str,
    sensor_preset_sha256: str,
    domain: Any,
    enforce_gate: bool,
) -> Mapping[str, Any]:
    relative = bag_record["tf_ego_diagnostic_report_path"]
    digest = bag_record["tf_ego_diagnostic_report_sha256"]
    if relative is None or digest is None:
        raise IntegrityError("bag diagnostic binding is missing")
    expected_relative = "diagnostics/{}/{}/{}.json".format(
        domain_id, resolved_domain_sha256, bag_record["source_bag_sha256"]
    )
    if _relative_posix_path(relative, "diagnostic path") != expected_relative:
        raise IntegrityError("bag diagnostic path identity mismatch")
    try:
        digest = _sha256(digest, "diagnostic SHA-256")
    except ValueError as error:
        raise IntegrityError("bag diagnostic SHA-256 is malformed") from error
    try:
        payload, _ = _read_cache_bytes(
            cache_root, relative, "TF/Ego diagnostic report"
        )
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IntegrityError("unable to parse TF/Ego diagnostic report") from error
    if sha256_bytes(payload) != digest:
        raise IntegrityError("TF/Ego diagnostic report SHA-256 mismatch")
    if type(value) is not dict or tuple(value) != DIAGNOSTIC_REPORT_FIELDS:
        raise IntegrityError("TF/Ego diagnostic report schema mismatch")
    if _json_bytes(value, newline=False) != payload:
        raise IntegrityError("TF/Ego diagnostic report encoding is not canonical")
    expected_identity = {
        "record_type": "tf_ego_diagnostic",
        "schema_version": 2,
        "domain_id": domain_id,
        "resolved_domain_sha256": resolved_domain_sha256,
        "sensor_preset_sha256": sensor_preset_sha256,
        "relative_bag_path": bag_record["relative_bag_path"],
        "source_bag_sha256": bag_record["source_bag_sha256"],
    }
    if any(value[name] != expected for name, expected in expected_identity.items()):
        raise IntegrityError("TF/Ego diagnostic report identity mismatch")
    if type(value["schema_version"]) is not int:
        raise IntegrityError("TF/Ego diagnostic schema version must be an integer")
    _validate_diagnostic_structure(value)
    counters = (
        "raw_ego_record_count",
        "epoch_local_comparable_count",
        "no_epoch_local_bracket_count",
        "ambiguous_epoch_selection_count",
        "cross_epoch_bracket_prevented_count",
    )
    bag_counter_names = (
        "tf_ego_raw_record_count",
        "tf_ego_epoch_local_comparable_count",
        "tf_ego_no_epoch_local_bracket_count",
        "tf_ego_ambiguous_epoch_selection_count",
        "tf_ego_cross_epoch_bracket_prevented_count",
    )
    if any(
        value[name] != bag_record[bag_name]
        for name, bag_name in zip(counters, bag_counter_names)
    ):
        raise IntegrityError("bag and diagnostic counters differ")
    if enforce_gate:
        _enforce_eligible_diagnostic_gate(value, domain)
    return MappingProxyType(value)


__all__ = (
    "BAG_RECORD_FIELDS",
    "DATASET_RECORD_FIELDS",
    "DIAGNOSTIC_PROVENANCE_FIELDS",
    "DIAGNOSTIC_REPORT_FIELDS",
    "PROVENANCE_DATASET_FIELDS",
    "SAMPLE_RECORD_FIELDS",
    "BagManifestInput",
    "BagManifestRecord",
    "DatasetManifestRecord",
    "DirtyOverride",
    "DirtyOverrideJson",
    "ManifestJsonRecord",
    "ManifestRecord",
    "RejectionReport",
    "SampleManifestRecord",
    "assign_bag_splits",
    "publish_manifest",
    "relative_bag_path",
)
