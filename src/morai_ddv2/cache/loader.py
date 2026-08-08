from __future__ import annotations

import hashlib
import json
import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Tuple

import h5py
import numpy as np

from morai_ddv2.cache.hdf5 import BagArtifactRecord, verify_bag_artifact
from morai_ddv2.cache.sidecar import (
    CommandSidecarIdentity,
    read_command_sidecar_bytes,
)
from morai_ddv2.config import ResolvedDomain
from morai_ddv2.errors import IntegrityError
from morai_ddv2.preprocess.planner import SampleDropReason
from morai_ddv2.ros1.messages import BagDisposition, BagRejectReason

from .domain import (
    _cache_regular_identity,
    _open_cache_regular_stream,
    _read_cache_bytes,
    load_domain_snapshot,
)
from .manifest import (
    BAG_RECORD_FIELDS,
    DATASET_RECORD_FIELDS,
    PROVENANCE_DATASET_FIELDS,
    SAMPLE_RECORD_FIELDS,
    _command_rule_sha256,
    _commit,
    _json_bytes,
    _plain_hdf5_value,
    _relative_posix_path,
    _sha256,
    _unique_json_object,
    _verify_diagnostic_report,
    assign_bag_splits,
)


_INT64_MAX = (1 << 63) - 1
_COMMAND_NAMES = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")


@dataclass(frozen=True)
class ModelSample:
    sample_id: str
    source_bag_sha256: str
    relative_bag_path: str
    split: Literal["train", "validation"]
    timestamp_ns: int
    command_name: Literal["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]
    command_index: Literal[0, 1, 2, 3]
    camera_feature: np.ndarray
    lidar_feature: np.ndarray
    status_feature: np.ndarray
    trajectory: np.ndarray

    def __post_init__(self) -> None:
        contracts = (
            ("camera_feature", (3, 256, 1024)),
            ("lidar_feature", (1, 256, 256)),
            ("status_feature", (8,)),
            ("trajectory", (8, 3)),
        )
        for name, shape in contracts:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError("{} must be a NumPy array".format(name))
            if value.dtype != np.dtype(np.float32) or value.shape != shape:
                raise ValueError("{} violates its float32 shape contract".format(name))
            if not np.isfinite(value).all():
                raise ValueError("{} must be finite".format(name))
            copied = np.ascontiguousarray(value, dtype=np.float32)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("sample_id must be a nonempty string")
        _sha256(self.source_bag_sha256, "source_bag_sha256")
        _relative_posix_path(self.relative_bag_path, "relative_bag_path")
        if self.split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        _nonnegative_int(self.timestamp_ns, "timestamp_ns")
        if type(self.command_index) is not int or not 0 <= self.command_index < 4:
            raise ValueError("command_index must be in [0,3]")
        if self.command_name != _COMMAND_NAMES[self.command_index]:
            raise ValueError("command_name does not match command_index")


@dataclass(frozen=True)
class VerifiedManifest:
    path: Path
    cache_root: Path
    manifest_sha256: str
    resolved_domain: ResolvedDomain
    dataset_record: Mapping[str, Any]
    bag_records: Tuple[Mapping[str, Any], ...]
    sample_records: Tuple[Mapping[str, Any], ...]
    _sample_by_id: Mapping[str, Mapping[str, Any]]
    _artifact_identities: Mapping[str, Tuple[int, ...]]
    _manifest_identity: Tuple[int, ...]


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: {}".format(value))


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0 or value > _INT64_MAX:
        raise IntegrityError("{} must be a nonnegative signed-int64 integer".format(name))
    return value


def _signed_int(value: Any, name: str) -> int:
    if type(value) is not int or value < -(_INT64_MAX + 1) or value > _INT64_MAX:
        raise IntegrityError("{} must be a signed-int64 integer".format(name))
    return value


def _finite_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise IntegrityError("{} must be a finite JSON float".format(name))
    return value


@contextmanager
def _open_verified_hdf5(
    *,
    cache_root: Path,
    relative_path: str,
    expected_identity: Tuple[int, ...],
    expected_sha256: Optional[str],
) -> Iterator[h5py.File]:
    try:
        with _open_cache_regular_stream(
            cache_root,
            relative_path,
            "HDF5 artifact",
            expected_identity=expected_identity,
            expected_sha256=expected_sha256,
        ) as stream:
            with h5py.File(stream, "r") as handle:
                yield handle
    except IntegrityError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise IntegrityError("unable to open verified HDF5 artifact") from error


def _int_list(
    value: Any,
    name: str,
    length: int,
    *,
    nonnegative: bool,
) -> List[int]:
    if type(value) is not list or len(value) != length:
        raise IntegrityError("{} must be a list of length {}".format(name, length))
    validator = _nonnegative_int if nonnegative else _signed_int
    return [validator(item, "{}[{}]".format(name, index)) for index, item in enumerate(value)]


def _matrix_int(
    value: Any, name: str, rows: int, columns: int
) -> List[List[int]]:
    if type(value) is not list or len(value) != rows:
        raise IntegrityError("{} must have {} rows".format(name, rows))
    return [
        _int_list(row, "{}[{}]".format(name, index), columns, nonnegative=True)
        for index, row in enumerate(value)
    ]


def _float_list(value: Any, name: str, length: int) -> List[float]:
    if type(value) is not list or len(value) != length:
        raise IntegrityError("{} must be a float list of length {}".format(name, length))
    result = [_finite_float(item, "{}[{}]".format(name, index)) for index, item in enumerate(value)]
    if any(item < 0.0 or item > 1.0 for item in result):
        raise IntegrityError("{} weights must be in [0,1]".format(name))
    return result


def _validate_bracket(
    *,
    header: List[int],
    receipt: List[int],
    record_index: List[int],
    weight: float,
    target_ns: int,
    name: str,
) -> None:
    if header[0] > header[1] or receipt[0] > receipt[1] or record_index[0] > record_index[1]:
        raise IntegrityError("{} provenance is not source ordered".format(name))
    if header[1] - header[0] > 100_000_000:
        raise IntegrityError("{} Header bracket exceeds 100 ms".format(name))
    if not header[0] <= target_ns <= header[1]:
        raise IntegrityError("{} does not bracket its native-Header target".format(name))
    if header[0] == header[1]:
        if receipt[0] != receipt[1] or record_index[0] != record_index[1]:
            raise IntegrityError("{} exact bracket must repeat one source".format(name))
        if weight != 0.0 or math.copysign(1.0, weight) < 0.0:
            raise IntegrityError("{} exact bracket requires positive zero weight".format(name))
        return
    expected = float(np.float32((target_ns - header[0]) / (header[1] - header[0])))
    if weight != expected:
        raise IntegrityError("{} interpolation weight does not match Header axis".format(name))


def _validate_sample_shape(record: Mapping[str, Any]) -> None:
    for name in (
        "hdf5_row_index",
        "timestamp_ns",
        "grid_receipt_ns",
        "trajectory_anchor_tf_header_ns",
        "trajectory_anchor_tf_receipt_ns",
        "trajectory_anchor_tf_record_index",
        "trajectory_anchor_tf_clock_epoch_index",
        "ego_bracket_clock_epoch_index",
        "sidecar_row_index",
        "command_index",
    ):
        _nonnegative_int(record[name], name)
    headers = _int_list(
        record["sensor_source_header_stamp_ns"],
        "sensor_source_header_stamp_ns",
        4,
        nonnegative=True,
    )
    receipts = _int_list(
        record["sensor_source_receipt_ns"],
        "sensor_source_receipt_ns",
        4,
        nonnegative=True,
    )
    indices = _int_list(
        record["sensor_source_record_index"],
        "sensor_source_record_index",
        4,
        nonnegative=True,
    )
    side_offsets = _int_list(
        record["side_camera_header_offset_ns"],
        "side_camera_header_offset_ns",
        2,
        nonnegative=False,
    )
    receipt_offsets = _int_list(
        record["sensor_receipt_offset_ns"],
        "sensor_receipt_offset_ns",
        3,
        nonnegative=False,
    )
    if record["timestamp_ns"] != headers[0]:
        raise IntegrityError("sample timestamp is not selected-front Header")
    if side_offsets != [headers[1] - headers[0], headers[2] - headers[0]]:
        raise IntegrityError("sample mixes or corrupts side-camera Header offsets")
    if receipt_offsets != [
        receipts[1] - receipts[0],
        receipts[2] - receipts[0],
        receipts[3] - receipts[0],
    ]:
        raise IntegrityError("sample mixes or corrupts sensor receipt offsets")

    anchor_header = record["trajectory_anchor_tf_header_ns"]
    anchor_receipt = record["trajectory_anchor_tf_receipt_ns"]
    anchor_index = record["trajectory_anchor_tf_record_index"]
    current_headers = _int_list(
        record["current_pose_bracket_header_stamp_ns"],
        "current_pose_bracket_header_stamp_ns",
        2,
        nonnegative=True,
    )
    current_receipts = _int_list(
        record["current_pose_bracket_receipt_ns"],
        "current_pose_bracket_receipt_ns",
        2,
        nonnegative=True,
    )
    current_indices = _int_list(
        record["current_pose_bracket_record_index"],
        "current_pose_bracket_record_index",
        2,
        nonnegative=True,
    )
    current_weight = _finite_float(
        record["current_pose_interp_weight"], "current_pose_interp_weight"
    )
    if (
        current_headers != [anchor_header, anchor_header]
        or current_receipts != [anchor_receipt, anchor_receipt]
        or current_indices != [anchor_index, anchor_index]
        or current_weight != 0.0
        or math.copysign(1.0, current_weight) < 0.0
    ):
        raise IntegrityError("current-pose provenance must repeat exact raw TF anchor")

    ego_headers = _int_list(
        record["ego_bracket_header_stamp_ns"],
        "ego_bracket_header_stamp_ns",
        2,
        nonnegative=True,
    )
    ego_receipts = _int_list(
        record["ego_bracket_receipt_ns"],
        "ego_bracket_receipt_ns",
        2,
        nonnegative=True,
    )
    ego_indices = _int_list(
        record["ego_bracket_record_index"],
        "ego_bracket_record_index",
        2,
        nonnegative=True,
    )
    ego_weight = _finite_float(record["ego_interp_weight"], "ego_interp_weight")
    if not 0.0 <= ego_weight <= 1.0:
        raise IntegrityError("ego_interp_weight must be in [0,1]")
    _validate_bracket(
        header=ego_headers,
        receipt=ego_receipts,
        record_index=ego_indices,
        weight=ego_weight,
        target_ns=anchor_header,
        name="Ego bracket",
    )

    future_headers = _matrix_int(
        record["future_pose_bracket_header_stamp_ns"],
        "future_pose_bracket_header_stamp_ns",
        8,
        2,
    )
    future_receipts = _matrix_int(
        record["future_pose_bracket_receipt_ns"],
        "future_pose_bracket_receipt_ns",
        8,
        2,
    )
    future_indices = _matrix_int(
        record["future_pose_bracket_record_index"],
        "future_pose_bracket_record_index",
        8,
        2,
    )
    future_weights = _float_list(
        record["future_pose_interp_weight"], "future_pose_interp_weight", 8
    )
    for index in range(8):
        _validate_bracket(
            header=future_headers[index],
            receipt=future_receipts[index],
            record_index=future_indices[index],
            weight=future_weights[index],
            target_ns=anchor_header + (index + 1) * 500_000_000,
            name="future bracket {}".format(index),
        )
    if indices[0] < 0:
        raise AssertionError("front index validation drift")


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _read_manifest(
    cache_root: Path, relative_path: str
) -> Tuple[bytes, Tuple[Dict[str, Any], ...], Tuple[int, ...]]:
    payload, identity = _read_cache_bytes(
        cache_root, relative_path, "dataset manifest"
    )
    if not payload or not payload.endswith(b"\n"):
        raise IntegrityError("dataset manifest must be nonempty and end with newline")
    raw_lines = payload.split(b"\n")
    if raw_lines[-1] != b"" or any(not line for line in raw_lines[:-1]):
        raise IntegrityError("dataset manifest contains a blank JSONL record")
    records = []
    for row_index, raw_line in enumerate(raw_lines[:-1]):
        try:
            value = json.loads(
                raw_line.decode("utf-8", errors="strict"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise IntegrityError(
                "dataset manifest row {} is invalid JSON".format(row_index)
            ) from error
        if type(value) is not dict:
            raise IntegrityError("dataset manifest records must be JSON objects")
        record_type = value.get("record_type")
        expected_fields = {
            "dataset": DATASET_RECORD_FIELDS,
            "bag": BAG_RECORD_FIELDS,
            "sample": SAMPLE_RECORD_FIELDS,
        }.get(record_type)
        if expected_fields is None or tuple(value) != expected_fields:
            raise IntegrityError(
                "dataset manifest row {} schema mismatch".format(row_index)
            )
        if _json_bytes(value, newline=True) != raw_line + b"\n":
            raise IntegrityError(
                "dataset manifest row {} is not deterministically encoded".format(
                    row_index
                )
            )
        records.append(value)
    return payload, tuple(records), identity


def _validate_dataset_record(
    dataset: Mapping[str, Any],
    *,
    domain: ResolvedDomain,
    expected_domain_id: str,
    expected_resolved_sha: str,
    production: bool,
) -> None:
    try:
        if dataset["schema_version"] != 2 or type(dataset["schema_version"]) is not int:
            raise ValueError("schema version")
        if dataset["domain_id"] != expected_domain_id:
            raise ValueError("domain ID")
        if dataset["resolved_domain_sha256"] != expected_resolved_sha:
            raise ValueError("resolved domain SHA-256")
        _sha256(dataset["sensor_preset_sha256"], "sensor_preset_sha256")
        _commit(dataset["source_code_commit"], "source_code_commit")
        if dataset["sensor_preset_sha256"] != domain.document["sensor_preset"]["sha256"]:
            raise ValueError("sensor preset identity")
        lifecycle = domain.document["domain_lifecycle"]
        if dataset["contract_revision_context"] != lifecycle[
            "contract_revision_context"
        ]:
            raise ValueError("contract revision context")
        if dataset["after_first_production_publication"] != lifecycle[
            "after_first_production_publication"
        ]:
            raise ValueError("post-publication policy")
        if dataset["split_seed_prefix"] != "20260806:":
            raise ValueError("split seed")
        for name in ("train_relative_bag_paths", "validation_relative_bag_paths"):
            value = dataset[name]
            if type(value) is not list:
                raise TypeError(name)
            for item in value:
                _relative_posix_path(item, name)
        dirty = dataset["dirty_override"]
        if dirty is not None:
            if type(dirty) is not dict or tuple(dirty) != (
                "allow_dirty_development",
                "source_code_commit",
                "dirty_diff_sha256",
            ):
                raise ValueError("dirty override schema")
            if dirty["allow_dirty_development"] is not True:
                raise ValueError("dirty override flag")
            if _commit(dirty["source_code_commit"], "dirty source commit") != dataset[
                "source_code_commit"
            ]:
                raise ValueError("dirty source commit")
            _sha256(dirty["dirty_diff_sha256"], "dirty diff SHA-256")
            if production:
                raise IntegrityError("dirty-override cache is forbidden in production")
    except IntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("dataset manifest identity is invalid") from error


def _validate_bag_record(record: Mapping[str, Any]) -> None:
    try:
        _relative_posix_path(record["relative_bag_path"], "relative_bag_path")
        _sha256(record["source_bag_sha256"], "source_bag_sha256")
    except (TypeError, ValueError) as error:
        raise IntegrityError("bag manifest identity is invalid") from error
    if record["disposition"] not in tuple(item.value for item in BagDisposition):
        raise IntegrityError("bag manifest disposition is invalid")
    if record["split"] not in (None, "train", "validation"):
        raise IntegrityError("bag manifest split is invalid")
    for name in ("sample_count", "sidecar_record_count"):
        _nonnegative_int(record[name], name)
    drop_counts = record["drop_counts"]
    expected_drop_names = tuple(reason.value for reason in SampleDropReason)
    if type(drop_counts) is not dict or tuple(drop_counts) != expected_drop_names:
        raise IntegrityError("bag drop_counts schema is invalid")
    for name in expected_drop_names:
        _nonnegative_int(drop_counts[name], "drop_counts.{}".format(name))
    for name in (
        "tf_ego_raw_record_count",
        "tf_ego_epoch_local_comparable_count",
        "tf_ego_no_epoch_local_bracket_count",
        "tf_ego_ambiguous_epoch_selection_count",
        "tf_ego_cross_epoch_bracket_prevented_count",
    ):
        _nonnegative_int(record[name], name)
    disposition = record["disposition"]
    artifact_fields = (
        "hdf5_path",
        "hdf5_sha256",
        "sidecar_path",
        "sidecar_sha256",
    )
    if disposition == "ELIGIBLE":
        if record["split"] is None or record["sample_count"] <= 0:
            raise IntegrityError("eligible bag lacks split or samples")
        if any(record[name] is None for name in artifact_fields):
            raise IntegrityError("eligible bag lacks artifact binding")
        if record["sample_count"] != record["sidecar_record_count"]:
            raise IntegrityError("eligible HDF5 and sidecar counts differ")
        if record["rejection_code"] is not None:
            raise IntegrityError("eligible bag has a rejection code")
        if record["tf_ego_diagnostic_report_path"] is None:
            raise IntegrityError("eligible bag lacks TF/Ego diagnostic binding")
    else:
        if record["split"] is not None or any(record[name] is not None for name in artifact_fields):
            raise IntegrityError("non-eligible bag has artifact or split binding")
        if record["sample_count"] != 0 or record["sidecar_record_count"] != 0:
            raise IntegrityError("non-eligible bag has nonzero artifact count")
        if disposition == "REJECTED":
            if record["rejection_code"] not in tuple(
                item.value for item in BagRejectReason
            ):
                raise IntegrityError("rejected bag lacks a valid rejection code")
        elif record["rejection_code"] is not None:
            raise IntegrityError("sensor-only excluded bag has rejection code")
    diagnostic_path = record["tf_ego_diagnostic_report_path"]
    diagnostic_sha = record["tf_ego_diagnostic_report_sha256"]
    if (diagnostic_path is None) != (diagnostic_sha is None):
        raise IntegrityError("bag diagnostic path/hash binding is incomplete")
    if diagnostic_path is None and any(
        record[name] != 0
        for name in (
            "tf_ego_raw_record_count",
            "tf_ego_epoch_local_comparable_count",
            "tf_ego_no_epoch_local_bracket_count",
            "tf_ego_ambiguous_epoch_selection_count",
            "tf_ego_cross_epoch_bracket_prevented_count",
        )
    ):
        raise IntegrityError("bag without diagnostic report has nonzero counters")
    if disposition == "SENSOR_ONLY_EXCLUDED" and diagnostic_path is not None:
        raise IntegrityError("sensor-only excluded bag cannot bind a TF/Ego diagnostic")


def load_verified_manifest(
    path: Path,
    sensor_preset_path: Path,
    expected_domain_id: str,
    expected_resolved_domain_sha256: str,
    production: bool = True,
) -> VerifiedManifest:
    if not isinstance(path, Path) or not isinstance(sensor_preset_path, Path):
        raise TypeError("path and sensor_preset_path must be pathlib.Path values")
    if type(production) is not bool:
        raise TypeError("production must be a bool")
    try:
        expected_resolved_domain_sha256 = _sha256(
            expected_resolved_domain_sha256, "expected_resolved_domain_sha256"
        )
        expected_domain_id.encode("utf-8", errors="strict")
    except (AttributeError, TypeError, ValueError, UnicodeEncodeError) as error:
        raise IntegrityError("expected manifest identity is invalid") from error
    path = path.absolute()
    if (
        path.name != "manifest.jsonl"
        or path.parent.name != expected_resolved_domain_sha256
        or path.parent.parent.name != expected_domain_id
        or path.parent.parent.parent.name != "manifests"
    ):
        raise IntegrityError("manifest path does not match expected identity")
    cache_root = path.parents[3]
    manifest_relative = "manifests/{}/{}/manifest.jsonl".format(
        expected_domain_id, expected_resolved_domain_sha256
    )
    payload, records, manifest_identity = _read_manifest(
        cache_root, manifest_relative
    )
    if not records or records[0]["record_type"] != "dataset":
        raise IntegrityError("manifest first record must be the dataset record")
    if any(item["record_type"] == "dataset" for item in records[1:]):
        raise IntegrityError("manifest must contain exactly one dataset record")
    seen_sample = False
    for item in records[1:]:
        if item["record_type"] == "sample":
            seen_sample = True
        elif seen_sample:
            raise IntegrityError("bag records must precede every sample record")
    dataset = records[0]
    expected_snapshot_relative = "domains/{}/{}/resolved-domain.json".format(
        expected_domain_id, expected_resolved_domain_sha256
    )
    try:
        snapshot_relative = _relative_posix_path(
            dataset["resolved_domain_path"], "resolved_domain_path"
        )
    except (TypeError, ValueError) as error:
        raise IntegrityError("resolved-domain snapshot path is invalid") from error
    if snapshot_relative != expected_snapshot_relative:
        raise IntegrityError("resolved-domain snapshot path identity mismatch")
    domain = load_domain_snapshot(
        cache_root / snapshot_relative,
        sensor_preset_path,
        expected_domain_id,
        expected_resolved_domain_sha256,
    )
    _validate_dataset_record(
        dataset,
        domain=domain,
        expected_domain_id=expected_domain_id,
        expected_resolved_sha=expected_resolved_domain_sha256,
        production=production,
    )

    bag_values = [item for item in records[1:] if item["record_type"] == "bag"]
    sample_values = [item for item in records[1:] if item["record_type"] == "sample"]
    if not bag_values:
        raise IntegrityError("manifest contains no bag inventory")
    for item in bag_values:
        _validate_bag_record(item)
    bag_paths = [item["relative_bag_path"] for item in bag_values]
    if bag_paths != sorted(bag_paths, key=lambda value: value.encode("utf-8")):
        raise IntegrityError("bag manifest records are not UTF-8 path sorted")
    if len(bag_paths) != len(set(bag_paths)):
        raise IntegrityError("bag manifest contains duplicate relative paths")
    bags_by_path = {item["relative_bag_path"]: item for item in bag_values}
    eligible = [item for item in bag_values if item["disposition"] == "ELIGIBLE"]
    if not eligible:
        raise IntegrityError("manifest contains no eligible bag")
    eligible_hashes = [item["source_bag_sha256"] for item in eligible]
    if len(eligible_hashes) != len(set(eligible_hashes)):
        raise IntegrityError("eligible bag source hashes are not unique")
    expected_splits = assign_bag_splits(
        [item["relative_bag_path"] for item in eligible]
    )
    actual_splits = {
        item["relative_bag_path"]: item["split"] for item in eligible
    }
    if actual_splits != dict(expected_splits):
        raise IntegrityError("bag records differ from deterministic split")
    expected_train = [
        bag_path for bag_path, split in expected_splits.items() if split == "train"
    ]
    expected_validation = [
        bag_path
        for bag_path, split in expected_splits.items()
        if split == "validation"
    ]
    if dataset["train_relative_bag_paths"] != expected_train or dataset[
        "validation_relative_bag_paths"
    ] != expected_validation:
        raise IntegrityError("dataset split lists differ from bag records")

    command_rule_sha = _command_rule_sha256(domain)
    artifact_records: Dict[str, BagArtifactRecord] = {}
    artifact_identities: Dict[str, Tuple[int, ...]] = {}
    sidecars: Dict[str, Tuple[Any, ...]] = {}
    for bag in bag_values:
        if bag["tf_ego_diagnostic_report_path"] is not None:
            _verify_diagnostic_report(
                cache_root=cache_root,
                bag_record=bag,
                domain_id=expected_domain_id,
                resolved_domain_sha256=expected_resolved_domain_sha256,
                sensor_preset_sha256=dataset["sensor_preset_sha256"],
                domain=domain,
                enforce_gate=bag["disposition"] == "ELIGIBLE",
            )
        if bag["disposition"] != "ELIGIBLE":
            continue
        try:
            artifact = BagArtifactRecord(
                domain_id=expected_domain_id,
                resolved_domain_sha256=expected_resolved_domain_sha256,
                sensor_preset_sha256=dataset["sensor_preset_sha256"],
                source_bag_sha256=bag["source_bag_sha256"],
                command_rule_sha256=command_rule_sha,
                source_code_commit=dataset["source_code_commit"],
                relative_bag_path=bag["relative_bag_path"],
                hdf5_path=bag["hdf5_path"],
                hdf5_sha256=bag["hdf5_sha256"],
                sample_count=bag["sample_count"],
                sidecar_path=bag["sidecar_path"],
                sidecar_sha256=bag["sidecar_sha256"],
                sidecar_record_count=bag["sidecar_record_count"],
            )
        except (TypeError, ValueError) as error:
            raise IntegrityError("eligible bag artifact record is invalid") from error
        verify_bag_artifact(artifact, cache_root)
        artifact_records[bag["relative_bag_path"]] = artifact
        artifact_identities[artifact.hdf5_path] = _cache_regular_identity(
            cache_root, artifact.hdf5_path, "HDF5 artifact"
        )
        identity = CommandSidecarIdentity(
            schema_version=2,
            domain_id=expected_domain_id,
            resolved_domain_sha256=expected_resolved_domain_sha256,
            sensor_preset_sha256=dataset["sensor_preset_sha256"],
            source_bag_sha256=artifact.source_bag_sha256,
            command_rule_sha256=command_rule_sha,
            provenance="future_trajectory_oracle",
        )
        sidecar_payload, _ = _read_cache_bytes(
            cache_root, artifact.sidecar_path, "command sidecar"
        )
        if hashlib.sha256(sidecar_payload).hexdigest() != artifact.sidecar_sha256:
            raise IntegrityError("opened command sidecar bytes differ from bound SHA-256")
        sidecars[artifact.relative_bag_path] = tuple(
            read_command_sidecar_bytes(sidecar_payload, identity).values()
        )

    sample_ids = set()
    per_bag_previous: Dict[str, Tuple[int, int]] = {}
    per_bag_front_indices: Dict[str, set] = {}
    per_bag_next_row: Dict[str, int] = {path: 0 for path in artifact_records}
    sample_by_id: Dict[str, Mapping[str, Any]] = {}
    previous_global_key = None
    hdf5_handles: Dict[str, h5py.File] = {}
    with ExitStack() as hdf5_stack:
        for artifact in artifact_records.values():
            hdf5_handles[artifact.relative_bag_path] = hdf5_stack.enter_context(
                _open_verified_hdf5(
                    cache_root=cache_root,
                    relative_path=artifact.hdf5_path,
                    expected_identity=artifact_identities[artifact.hdf5_path],
                    expected_sha256=artifact.hdf5_sha256,
                )
            )
        try:
            for sample in sample_values:
                _validate_sample_shape(sample)
                relative = sample["relative_bag_path"]
                bag = bags_by_path.get(relative)
                if bag is None or bag["disposition"] != "ELIGIBLE":
                    raise IntegrityError(
                        "sample references a non-eligible or missing bag"
                    )
                artifact = artifact_records[relative]
                if (
                    sample["source_bag_sha256"] != artifact.source_bag_sha256
                    or sample["split"] != bag["split"]
                    or sample["hdf5_path"] != artifact.hdf5_path
                    or sample["sidecar_path"] != artifact.sidecar_path
                ):
                    raise IntegrityError("sample artifact or split identity mismatch")
                expected_sample_id = "{}:{}".format(
                    artifact.source_bag_sha256, sample["timestamp_ns"]
                )
                if sample["sample_id"] != expected_sample_id:
                    raise IntegrityError(
                        "sample ID does not match bag hash and timestamp"
                    )
                if sample["sample_id"] in sample_ids:
                    raise IntegrityError("manifest contains duplicate sample IDs")
                sample_ids.add(sample["sample_id"])
                global_key = (
                    sample["timestamp_ns"],
                    relative.encode("utf-8"),
                    sample["hdf5_row_index"],
                )
                if (
                    previous_global_key is not None
                    and global_key <= previous_global_key
                ):
                    raise IntegrityError("sample records are not timestamp sorted")
                previous_global_key = global_key
                expected_row = per_bag_next_row[relative]
                if (
                    sample["hdf5_row_index"] != expected_row
                    or sample["sidecar_row_index"] != expected_row
                ):
                    raise IntegrityError(
                        "sample HDF5/sidecar row indices are not contiguous"
                    )
                per_bag_next_row[relative] += 1
                previous = per_bag_previous.get(relative)
                if previous is not None and (
                    sample["timestamp_ns"] <= previous[0]
                    or sample["grid_receipt_ns"] <= previous[1]
                ):
                    raise IntegrityError(
                        "bag sample timestamps/grids are not strictly increasing"
                    )
                per_bag_previous[relative] = (
                    sample["timestamp_ns"],
                    sample["grid_receipt_ns"],
                )
                front_index = sample["sensor_source_record_index"][0]
                indices = per_bag_front_indices.setdefault(relative, set())
                if front_index in indices:
                    raise IntegrityError(
                        "bag sample reuses a front stable record index"
                    )
                indices.add(front_index)

                handle = hdf5_handles[relative]
                row_index = sample["hdf5_row_index"]
                for name in PROVENANCE_DATASET_FIELDS:
                    actual = _plain_hdf5_value(handle[name][row_index])
                    if actual != sample[name]:
                        raise IntegrityError(
                            "manifest/HDF5 provenance mismatch for {}".format(name)
                        )
                sidecar = sidecars[relative][sample["sidecar_row_index"]]
                if (
                    sidecar.timestamp_ns != sample["timestamp_ns"]
                    or sidecar.command_index != sample["command_index"]
                    or sidecar.command_name != sample["command_name"]
                    or sidecar.provenance != sample["command_provenance"]
                ):
                    raise IntegrityError("manifest/sidecar command identity mismatch")
                if (
                    sample["command_provenance"] != "future_trajectory_oracle"
                    or type(sample["command_index"]) is not int
                    or not 0 <= sample["command_index"] < 4
                    or sample["command_name"]
                    != _COMMAND_NAMES[sample["command_index"]]
                ):
                    raise IntegrityError("sample command record is invalid")
                sample_by_id[sample["sample_id"]] = sample
        except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
            if isinstance(error, IntegrityError):
                raise
            raise IntegrityError("unable to verify manifest samples") from error
    for bag_path, artifact in artifact_records.items():
        if per_bag_next_row[bag_path] != artifact.sample_count:
            raise IntegrityError("manifest sample count differs from bag artifact")
        if artifact_identities[artifact.hdf5_path] != _cache_regular_identity(
            cache_root, artifact.hdf5_path, "HDF5 artifact"
        ):
            raise IntegrityError("HDF5 artifact changed during manifest verification")
    if len(sample_values) != sum(item.sample_count for item in artifact_records.values()):
        raise IntegrityError("manifest total sample count mismatch")
    if manifest_identity != _cache_regular_identity(
        cache_root, manifest_relative, "dataset manifest"
    ):
        raise IntegrityError("dataset manifest changed during verification")
    current_manifest, current_manifest_identity = _read_cache_bytes(
        cache_root, manifest_relative, "dataset manifest"
    )
    if current_manifest_identity != manifest_identity or payload != current_manifest:
        raise IntegrityError("dataset manifest bytes changed after verification")
    actual_manifest_sha = hashlib.sha256(current_manifest).hexdigest()

    frozen_dataset = _freeze_json(dict(dataset))
    frozen_bags = tuple(_freeze_json(dict(item)) for item in bag_values)
    frozen_samples = tuple(_freeze_json(dict(item)) for item in sample_values)
    frozen_by_id = MappingProxyType(
        {item["sample_id"]: item for item in frozen_samples}
    )
    return VerifiedManifest(
        path=path,
        cache_root=cache_root,
        manifest_sha256=actual_manifest_sha,
        resolved_domain=domain,
        dataset_record=frozen_dataset,
        bag_records=frozen_bags,
        sample_records=frozen_samples,
        _sample_by_id=frozen_by_id,
        _artifact_identities=MappingProxyType(dict(artifact_identities)),
        _manifest_identity=manifest_identity,
    )


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    result.setflags(write=False)
    return result


def load_tensor_sample(manifest: VerifiedManifest, sample_id: str) -> ModelSample:
    if type(manifest) is not VerifiedManifest:
        raise TypeError("manifest must be a VerifiedManifest")
    if type(sample_id) is not str:
        raise TypeError("sample_id must be a string")
    record = manifest._sample_by_id.get(sample_id)
    if record is None:
        raise KeyError(sample_id)
    manifest_relative = manifest.path.relative_to(manifest.cache_root).as_posix()
    if manifest._manifest_identity != _cache_regular_identity(
        manifest.cache_root, manifest_relative, "dataset manifest"
    ):
        raise IntegrityError("dataset manifest changed after verification")
    hdf5_relative = record["hdf5_path"]
    expected_identity = manifest._artifact_identities[hdf5_relative]
    if expected_identity != _cache_regular_identity(
        manifest.cache_root, hdf5_relative, "HDF5 artifact"
    ):
        raise IntegrityError("HDF5 artifact changed after manifest verification")
    row_index = record["hdf5_row_index"]
    try:
        with _open_verified_hdf5(
            cache_root=manifest.cache_root,
            relative_path=hdf5_relative,
            expected_identity=expected_identity,
            expected_sha256=None,
        ) as handle:
            camera = np.asarray(handle["camera_rgb"][row_index])
            lidar = np.asarray(handle["lidar_count"][row_index])
            status = np.asarray(handle["status"][row_index])
            trajectory = np.asarray(handle["trajectory"][row_index])
            for name in PROVENANCE_DATASET_FIELDS:
                if _freeze_json(_plain_hdf5_value(handle[name][row_index])) != record[name]:
                    raise IntegrityError(
                        "HDF5 provenance changed after manifest verification"
                    )
    except IntegrityError:
        raise
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        raise IntegrityError("unable to load tensor sample") from error
    if camera.dtype != np.dtype(np.uint8) or camera.shape != (256, 1024, 3):
        raise IntegrityError("camera cache tensor contract mismatch")
    if lidar.dtype != np.dtype(np.uint8) or lidar.shape != (256, 256):
        raise IntegrityError("LiDAR cache tensor contract mismatch")
    if int(lidar.max()) > 5:
        raise IntegrityError("LiDAR cache count exceeds clip value 5")
    if status.dtype != np.dtype(np.float32) or status.shape != (8,):
        raise IntegrityError("status cache tensor contract mismatch")
    if trajectory.dtype != np.dtype(np.float32) or trajectory.shape != (8, 3):
        raise IntegrityError("trajectory cache tensor contract mismatch")
    if not np.isfinite(status).all() or not np.isfinite(trajectory).all():
        raise IntegrityError("status and trajectory tensors must be finite")
    expected_one_hot = np.zeros(4, dtype=np.float32)
    expected_one_hot[record["command_index"]] = np.float32(1.0)
    if not np.array_equal(status[:4], expected_one_hot):
        raise IntegrityError("status one-hot differs from manifest command")
    camera_feature = _readonly_float32(
        camera.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    )
    lidar_feature = _readonly_float32(
        lidar[np.newaxis, :, :].astype(np.float32) / np.float32(5.0)
    )
    status_feature = _readonly_float32(status)
    target = _readonly_float32(trajectory)
    if (
        float(camera_feature.min()) < 0.0
        or float(camera_feature.max()) > 1.0
        or float(lidar_feature.min()) < 0.0
        or float(lidar_feature.max()) > 1.0
    ):
        raise IntegrityError("model input tensor range mismatch")
    return ModelSample(
        sample_id=record["sample_id"],
        source_bag_sha256=record["source_bag_sha256"],
        relative_bag_path=record["relative_bag_path"],
        split=record["split"],
        timestamp_ns=record["timestamp_ns"],
        command_name=record["command_name"],
        command_index=record["command_index"],
        camera_feature=camera_feature,
        lidar_feature=lidar_feature,
        status_feature=status_feature,
        trajectory=target,
    )


__all__ = (
    "ModelSample",
    "VerifiedManifest",
    "load_tensor_sample",
    "load_verified_manifest",
)
