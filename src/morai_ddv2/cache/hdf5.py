from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Iterable, Optional, Tuple

import h5py
import numpy as np

from morai_ddv2.cache import CACHE_DATASET_NAMES, CacheSample
from morai_ddv2.config import (
    ResolvedDomain,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from morai_ddv2.errors import ConfigurationError, IntegrityError
from morai_ddv2.preprocess.trajectory import Command, CommandRules

from .sidecar import (
    COMMAND_RECORD_FIELDS,
    CommandRecord,
    CommandSidecarIdentity,
    _write_command_record,
    read_command_sidecar,
)


HDF5_DATASETS = {
    "camera_rgb": (np.uint8, (256, 1024, 3)),
    "lidar_count": (np.uint8, (256, 256)),
    "status": (np.float32, (8,)),
    "trajectory": (np.float32, (8, 3)),
    "timestamp_ns": (np.int64, ()),
    "grid_receipt_ns": (np.int64, ()),
    "sensor_source_header_stamp_ns": (np.int64, (4,)),
    "sensor_source_receipt_ns": (np.int64, (4,)),
    "sensor_source_record_index": (np.int64, (4,)),
    "side_camera_header_offset_ns": (np.int64, (2,)),
    "sensor_receipt_offset_ns": (np.int64, (3,)),
    "trajectory_anchor_tf_header_ns": (np.int64, ()),
    "trajectory_anchor_tf_receipt_ns": (np.int64, ()),
    "trajectory_anchor_tf_record_index": (np.int64, ()),
    "trajectory_anchor_tf_clock_epoch_index": (np.int64, ()),
    "ego_bracket_clock_epoch_index": (np.int64, ()),
    "ego_bracket_header_stamp_ns": (np.int64, (2,)),
    "ego_bracket_receipt_ns": (np.int64, (2,)),
    "ego_bracket_record_index": (np.int64, (2,)),
    "ego_interp_weight": (np.float32, ()),
    "current_pose_bracket_header_stamp_ns": (np.int64, (2,)),
    "current_pose_bracket_receipt_ns": (np.int64, (2,)),
    "current_pose_bracket_record_index": (np.int64, (2,)),
    "current_pose_interp_weight": (np.float32, ()),
    "future_pose_bracket_header_stamp_ns": (np.int64, (8, 2)),
    "future_pose_bracket_receipt_ns": (np.int64, (8, 2)),
    "future_pose_bracket_record_index": (np.int64, (8, 2)),
    "future_pose_interp_weight": (np.float32, (8,)),
}

_HDF5_ATTRIBUTE_NAMES = {
    "cache_schema_version",
    "clock_policy_version",
    "domain_id",
    "sensor_preset_sha256",
    "resolved_domain_sha256",
    "source_bag_sha256",
    "source_code_commit",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_ARRAY_FIELDS = {
    "camera_rgb",
    "lidar_count",
    "status",
    "trajectory",
    "future_pose_bracket_header_stamp_ns",
    "future_pose_bracket_receipt_ns",
    "future_pose_bracket_record_index",
    "future_pose_interp_weight",
}
_TUPLE_FIELDS = {
    "sensor_source_header_stamp_ns",
    "sensor_source_receipt_ns",
    "sensor_source_record_index",
    "side_camera_header_offset_ns",
    "sensor_receipt_offset_ns",
    "ego_bracket_header_stamp_ns",
    "ego_bracket_receipt_ns",
    "ego_bracket_record_index",
    "current_pose_bracket_header_stamp_ns",
    "current_pose_bracket_receipt_ns",
    "current_pose_bracket_record_index",
}
_FLOAT_FIELDS = {"ego_interp_weight", "current_pose_interp_weight"}


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError("{} must be a string".format(name))
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "{} must be 64 lowercase hexadecimal characters".format(name)
        )
    return value


def _source_commit(value: Any) -> str:
    if type(value) is not str:
        raise TypeError("source_code_commit must be a string")
    if _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "source_code_commit must be 40 lowercase hexadecimal characters"
        )
    return value


def _safe_component(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError("{} must be a string".format(name))
    if (
        not value
        or "\x00" in value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("{} must be one safe nonempty path component".format(name))
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("{} must be valid Unicode".format(name)) from error
    return value


def _relative_posix_path(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError("{} must be a string".format(name))
    if not value or "\x00" in value or value in (".", "..") or "\\" in value:
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


def _positive_count(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError("{} must be an actual Python int".format(name))
    if value <= 0 or value > (1 << 63) - 1:
        raise ValueError("{} must be a positive signed-int64 value".format(name))
    return value


def _artifact_relative_paths(
    domain_id: str, resolved_sha256: str, source_sha256: str
) -> Tuple[str, str]:
    prefix = "{}/{}".format(domain_id, resolved_sha256)
    return (
        "bags/{}/{}.h5".format(prefix, source_sha256),
        "commands/{}/{}.jsonl".format(prefix, source_sha256),
    )


@dataclass(frozen=True)
class BagArtifactRecord:
    domain_id: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    source_bag_sha256: str
    command_rule_sha256: str
    source_code_commit: str
    relative_bag_path: str
    hdf5_path: str
    hdf5_sha256: str
    sample_count: int
    sidecar_path: str
    sidecar_sha256: str
    sidecar_record_count: int

    def __post_init__(self) -> None:
        domain_id = _safe_component(self.domain_id, "domain_id")
        resolved_sha = _sha256(
            self.resolved_domain_sha256, "resolved_domain_sha256"
        )
        source_sha = _sha256(self.source_bag_sha256, "source_bag_sha256")
        for name in (
            "sensor_preset_sha256",
            "command_rule_sha256",
            "hdf5_sha256",
            "sidecar_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "source_code_commit", _source_commit(self.source_code_commit))
        object.__setattr__(
            self,
            "relative_bag_path",
            _relative_posix_path(self.relative_bag_path, "relative_bag_path"),
        )
        object.__setattr__(
            self, "hdf5_path", _relative_posix_path(self.hdf5_path, "hdf5_path")
        )
        object.__setattr__(
            self,
            "sidecar_path",
            _relative_posix_path(self.sidecar_path, "sidecar_path"),
        )
        sample_count = _positive_count(self.sample_count, "sample_count")
        sidecar_count = _positive_count(
            self.sidecar_record_count, "sidecar_record_count"
        )
        if sample_count != sidecar_count:
            raise ValueError("HDF5 and sidecar counts must match")
        expected_hdf5, expected_sidecar = _artifact_relative_paths(
            domain_id, resolved_sha, source_sha
        )
        if self.hdf5_path != expected_hdf5:
            raise ValueError("hdf5_path does not match artifact identity")
        if self.sidecar_path != expected_sidecar:
            raise ValueError("sidecar_path does not match artifact identity")


def _command_rule_sha256(domain: ResolvedDomain) -> str:
    command = domain.document["command"]
    rule_document = {
        "index_mapping": command["index_mapping"],
        "thresholds": command["thresholds"],
        "ordered_rules": command["ordered_rules"],
    }
    return sha256_bytes(canonical_json_bytes(rule_document))


def _validate_domain(domain: Any) -> CommandSidecarIdentity:
    if type(domain) is not ResolvedDomain:
        raise TypeError("domain must be a ResolvedDomain")
    try:
        resolved_sha = _sha256(domain.sha256, "resolved_domain_sha256")
        if sha256_bytes(canonical_json_bytes(domain.document)) != resolved_sha:
            raise IntegrityError("resolved domain content does not match its SHA-256")
        domain_id = _safe_component(domain.document["domain_id"], "domain_id")
        sensor_sha = _sha256(
            domain.document["sensor_preset"]["sha256"],
            "sensor_preset_sha256",
        )
        CommandRules.from_resolved_domain(domain)
        cache = domain.document["cache"]
        clock_policy_version = domain.document["sampling"]["clock_policy"][
            "version"
        ]
        if type(clock_policy_version) is not int or clock_policy_version != 1:
            raise ConfigurationError("sampling.clock_policy.version must be 1")
        if type(cache["schema_version"]) is not int or cache["schema_version"] != 2:
            raise ConfigurationError("cache.schema_version must be 2")
        if (
            type(cache["command_sidecar_schema_version"]) is not int
            or cache["command_sidecar_schema_version"] != 2
        ):
            raise ConfigurationError("command sidecar schema must be 2")
        if tuple(cache["command_record_fields"]) != COMMAND_RECORD_FIELDS:
            raise ConfigurationError("command_record_fields do not match schema")
        if cache["bag_path_template"] != (
            "bags/{domain_id}/{resolved_domain_sha256}/{source_bag_sha256}.h5"
        ):
            raise ConfigurationError("bag_path_template does not match schema")
        if cache["command_path_template"] != (
            "commands/{domain_id}/{resolved_domain_sha256}/"
            "{source_bag_sha256}.jsonl"
        ):
            raise ConfigurationError("command_path_template does not match schema")
        atomic = cache["atomic_publication"]
        if set(atomic) != {
            "file_fsync",
            "parent_directory_fsync",
            "partial_suffix",
            "replace",
        }:
            raise ConfigurationError("atomic publication config does not match schema")
        if atomic["file_fsync"] is not True:
            raise ConfigurationError("atomic publication requires file fsync")
        if atomic["parent_directory_fsync"] is not True:
            raise ConfigurationError("atomic publication requires parent fsync")
        if atomic["partial_suffix"] != ".partial":
            raise ConfigurationError("atomic publication partial suffix must be .partial")
        if atomic["replace"] != "os.replace":
            raise ConfigurationError("atomic publication replace must be os.replace")
        if tuple(cache["sensor_axis_order"]) != (
            "front",
            "left",
            "right",
            "lidar",
        ):
            raise ConfigurationError("sensor_axis_order does not match schema")
        if cache["sample_id_template"] != "{source_bag_sha256}:{timestamp_ns}":
            raise ConfigurationError("sample_id_template does not match schema")
        attributes = cache["hdf5_attributes"]
        if set(attributes) != {"cache_schema_version", "clock_policy_version"}:
            raise ConfigurationError("HDF5 attribute config does not match schema")
        if (
            type(attributes["cache_schema_version"]) is not int
            or attributes["cache_schema_version"] != 2
        ):
            raise ConfigurationError("cache_schema_version must be 2")
        if (
            type(attributes["clock_policy_version"]) is not int
            or attributes["clock_policy_version"] != 1
        ):
            raise ConfigurationError("clock_policy_version must be 1")
        configured_datasets = cache["datasets"]
        if set(configured_datasets) != set(HDF5_DATASETS):
            raise ConfigurationError("configured HDF5 datasets do not match schema")
        for name, (dtype, trailing_shape) in HDF5_DATASETS.items():
            configured = configured_datasets[name]
            if set(configured) != {"dtype", "shape"}:
                raise ConfigurationError("dataset {} config is malformed".format(name))
            if configured["dtype"] != np.dtype(dtype).name:
                raise ConfigurationError("dataset {} dtype drift".format(name))
            if tuple(configured["shape"]) != ("N",) + trailing_shape:
                raise ConfigurationError("dataset {} shape drift".format(name))
        provenance = domain.document["command"].get("provenance")
        command_rule_sha = _command_rule_sha256(domain)
        return CommandSidecarIdentity(
            schema_version=2,
            domain_id=domain_id,
            resolved_domain_sha256=resolved_sha,
            sensor_preset_sha256=sensor_sha,
            source_bag_sha256="0" * 64,
            command_rule_sha256=command_rule_sha,
            provenance=provenance,
        )
    except IntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("resolved domain cache contract is invalid") from error


def _identity_for_bag(
    domain_identity: CommandSidecarIdentity, source_bag_sha256: str
) -> CommandSidecarIdentity:
    return CommandSidecarIdentity(
        schema_version=domain_identity.schema_version,
        domain_id=domain_identity.domain_id,
        resolved_domain_sha256=domain_identity.resolved_domain_sha256,
        sensor_preset_sha256=domain_identity.sensor_preset_sha256,
        source_bag_sha256=source_bag_sha256,
        command_rule_sha256=domain_identity.command_rule_sha256,
        provenance=domain_identity.provenance,
    )


def _validated_sample(sample: Any) -> CacheSample:
    if type(sample) is not CacheSample:
        raise TypeError("samples must contain only CacheSample instances")
    try:
        return CacheSample(**{field.name: getattr(sample, field.name) for field in fields(CacheSample)})
    except (TypeError, ValueError, OverflowError) as error:
        raise IntegrityError("CacheSample was mutated after validation") from error


def _create_hdf5_datasets(handle: h5py.File) -> Dict[str, h5py.Dataset]:
    result = {}
    for name, (dtype, trailing_shape) in HDF5_DATASETS.items():
        result[name] = handle.create_dataset(
            name,
            shape=(0,) + trailing_shape,
            maxshape=(None,) + trailing_shape,
            chunks=(1,) + trailing_shape,
            dtype=dtype,
        )
    return result


def _append_hdf5_sample(
    datasets: Mapping, row_index: int, sample: CacheSample
) -> None:
    for name, (_, trailing_shape) in HDF5_DATASETS.items():
        dataset = datasets[name]
        dataset.resize((row_index + 1,) + trailing_shape)
        dataset[row_index] = getattr(sample, name)


def _write_sidecar_record(stream: BinaryIO, record: CommandRecord) -> None:
    _write_command_record(stream, record)


def _open_new_partial(path: Path) -> BinaryIO:
    if _path_entry_exists(path):
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise IntegrityError("unable to inspect stale partial") from error
        if not stat.S_ISREG(mode):
            raise IntegrityError("stale partial must be a regular file")
        path.unlink()
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError as error:
        raise IntegrityError("partial path appeared during creation") from error
    try:
        return os.fdopen(descriptor, "w+b")
    except Exception:
        os.close(descriptor)
        raise


def _fsync_regular_file(path: Path) -> None:
    with path.open("rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise IntegrityError("artifact partial is not a regular file")
        os.fsync(stream.fileno())


def _replace_file(source: Path, destination: Path) -> None:
    os.replace(str(source), str(destination))


def _path_entry_exists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path.parent), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_regular_file(path: Path, name: str) -> Tuple[int, ...]:
    try:
        status = path.lstat()
    except OSError as error:
        raise IntegrityError("{} is missing or unreadable".format(name)) from error
    if not stat.S_ISREG(status.st_mode):
        raise IntegrityError("{} must be a regular file".format(name))
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _sample_from_hdf5(
    handle: h5py.File, row_index: int, command: Command
) -> CacheSample:
    values = {}
    for name in CACHE_DATASET_NAMES:
        stored = handle[name][row_index]
        if name in _ARRAY_FIELDS:
            values[name] = np.asarray(stored, dtype=HDF5_DATASETS[name][0])
        elif name in _TUPLE_FIELDS:
            values[name] = tuple(int(item) for item in np.asarray(stored).reshape(-1))
        elif name in _FLOAT_FIELDS:
            values[name] = float(stored)
        else:
            values[name] = int(stored)
    values["command"] = command
    return CacheSample(**values)


def _verify_artifact_pair(
    hdf5_path: Path,
    sidecar_path: Path,
    identity: CommandSidecarIdentity,
    source_code_commit: str,
    expected_count: Optional[int] = None,
) -> int:
    hdf5_identity = _require_regular_file(hdf5_path, "HDF5 artifact")
    sidecar_identity = _require_regular_file(sidecar_path, "command sidecar")
    commands = read_command_sidecar(sidecar_path, identity)
    try:
        with h5py.File(str(hdf5_path), "r") as handle:
            if set(handle.keys()) != set(HDF5_DATASETS):
                raise IntegrityError("HDF5 dataset set does not match schema")
            if set(handle.attrs.keys()) != _HDF5_ATTRIBUTE_NAMES:
                raise IntegrityError("HDF5 attribute set does not match schema")
            expected_attrs = {
                "cache_schema_version": 2,
                "clock_policy_version": 1,
                "domain_id": identity.domain_id,
                "sensor_preset_sha256": identity.sensor_preset_sha256,
                "resolved_domain_sha256": identity.resolved_domain_sha256,
                "source_bag_sha256": identity.source_bag_sha256,
                "source_code_commit": source_code_commit,
            }
            for name in ("cache_schema_version", "clock_policy_version"):
                actual = handle.attrs[name]
                expected = expected_attrs[name]
                if (
                    isinstance(actual, (bool, np.bool_))
                    or not isinstance(actual, (int, np.integer))
                    or int(actual) != expected
                ):
                    raise IntegrityError("HDF5 attribute {} mismatch".format(name))
            for name in _HDF5_ATTRIBUTE_NAMES - {
                "cache_schema_version",
                "clock_policy_version",
            }:
                actual = handle.attrs[name]
                if type(actual) is not str or actual != expected_attrs[name]:
                    raise IntegrityError("HDF5 attribute {} mismatch".format(name))

            row_counts = set()
            object_addresses = set()
            for name, (dtype, trailing_shape) in HDF5_DATASETS.items():
                link = handle.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise IntegrityError("HDF5 dataset {} must be a hard link".format(name))
                dataset = handle.get(name)
                if not isinstance(dataset, h5py.Dataset):
                    raise IntegrityError("HDF5 object {} must be a dataset".format(name))
                address = h5py.h5o.get_info(dataset.id).addr
                if address in object_addresses:
                    raise IntegrityError("HDF5 dataset objects must not be aliased")
                object_addresses.add(address)
                if dataset.dtype != np.dtype(dtype):
                    raise IntegrityError("HDF5 dataset {} dtype mismatch".format(name))
                if dataset.shape[1:] != trailing_shape:
                    raise IntegrityError("HDF5 dataset {} shape mismatch".format(name))
                if dataset.maxshape != (None,) + trailing_shape:
                    raise IntegrityError("HDF5 dataset {} maxshape mismatch".format(name))
                if dataset.chunks != (1,) + trailing_shape:
                    raise IntegrityError("HDF5 dataset {} chunks mismatch".format(name))
                if (
                    dataset.compression is not None
                    or dataset.shuffle
                    or dataset.fletcher32
                    or dataset.scaleoffset is not None
                ):
                    raise IntegrityError("HDF5 dataset {} filters mismatch".format(name))
                row_counts.add(dataset.shape[0])
            if len(row_counts) != 1:
                raise IntegrityError("HDF5 dataset row counts differ")
            count = row_counts.pop()
            if count <= 0 or len(commands) != count:
                raise IntegrityError("HDF5 and sidecar counts must be equal and nonzero")
            if expected_count is not None and count != expected_count:
                raise IntegrityError("artifact count does not match expected count")

            previous_timestamp = None
            previous_grid = None
            front_indices = set()
            for row_index, record in enumerate(commands.values()):
                try:
                    sample = _sample_from_hdf5(
                        handle, row_index, Command(record.command_index)
                    )
                except (TypeError, ValueError, OverflowError) as error:
                    raise IntegrityError(
                        "invalid HDF5 sample row {}".format(row_index)
                    ) from error
                front_receipt = sample.sensor_source_receipt_ns[0]
                front_index = sample.sensor_source_record_index[0]
                if (
                    record.timestamp_ns != sample.timestamp_ns
                    or record.grid_receipt_ns != sample.grid_receipt_ns
                    or record.front_receipt_ns != front_receipt
                    or record.front_record_index != front_index
                    or record.command_index != int(sample.command)
                    or record.command_name != sample.command.name
                ):
                    raise IntegrityError(
                        "HDF5 and sidecar cross-link mismatch at row {}".format(
                            row_index
                        )
                    )
                if previous_timestamp is not None and sample.timestamp_ns <= previous_timestamp:
                    raise IntegrityError("HDF5 timestamps must be strictly increasing")
                if previous_grid is not None and sample.grid_receipt_ns <= previous_grid:
                    raise IntegrityError("HDF5 grid receipts must be strictly increasing")
                if front_index in front_indices:
                    raise IntegrityError("front record indices must be unique")
                previous_timestamp = sample.timestamp_ns
                previous_grid = sample.grid_receipt_ns
                front_indices.add(front_index)
            if hdf5_identity != _require_regular_file(
                hdf5_path, "HDF5 artifact"
            ):
                raise IntegrityError("HDF5 artifact changed during pair verification")
            if sidecar_identity != _require_regular_file(
                sidecar_path, "command sidecar"
            ):
                raise IntegrityError(
                    "command sidecar changed during pair verification"
                )
            return count
    except IntegrityError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise IntegrityError("unable to verify HDF5 artifact") from error


def _write_partial_pair(
    *,
    hdf5_partial: Path,
    sidecar_partial: Path,
    iterator: Any,
    identity: CommandSidecarIdentity,
    source_code_commit: str,
) -> int:
    count = 0
    previous_timestamp = None
    previous_grid = None
    front_indices = set()
    with _open_new_partial(hdf5_partial) as hdf5_stream:
        with _open_new_partial(sidecar_partial) as sidecar_stream:
            with h5py.File(hdf5_stream, "w") as handle:
                datasets = _create_hdf5_datasets(handle)
                handle.attrs["cache_schema_version"] = 2
                handle.attrs["clock_policy_version"] = 1
                handle.attrs["domain_id"] = identity.domain_id
                handle.attrs["sensor_preset_sha256"] = identity.sensor_preset_sha256
                handle.attrs["resolved_domain_sha256"] = identity.resolved_domain_sha256
                handle.attrs["source_bag_sha256"] = identity.source_bag_sha256
                handle.attrs["source_code_commit"] = source_code_commit
                while True:
                    try:
                        raw_sample = next(iterator)
                    except StopIteration:
                        break
                    sample = _validated_sample(raw_sample)
                    front_index = sample.sensor_source_record_index[0]
                    if (
                        previous_timestamp is not None
                        and sample.timestamp_ns <= previous_timestamp
                    ):
                        raise IntegrityError(
                            "sample timestamps must be strictly increasing"
                        )
                    if previous_grid is not None and sample.grid_receipt_ns <= previous_grid:
                        raise IntegrityError(
                            "sample grid receipts must be strictly increasing"
                        )
                    if front_index in front_indices:
                        raise IntegrityError("front record indices must be unique")
                    record = CommandRecord(
                        schema_version=2,
                        domain_id=identity.domain_id,
                        resolved_domain_sha256=identity.resolved_domain_sha256,
                        sensor_preset_sha256=identity.sensor_preset_sha256,
                        source_bag_sha256=identity.source_bag_sha256,
                        command_rule_sha256=identity.command_rule_sha256,
                        timestamp_ns=sample.timestamp_ns,
                        grid_receipt_ns=sample.grid_receipt_ns,
                        front_receipt_ns=sample.sensor_source_receipt_ns[0],
                        front_record_index=front_index,
                        command_index=int(sample.command),
                        command_name=sample.command.name,
                        provenance=identity.provenance,
                    )
                    _append_hdf5_sample(datasets, count, sample)
                    _write_sidecar_record(sidecar_stream, record)
                    previous_timestamp = sample.timestamp_ns
                    previous_grid = sample.grid_receipt_ns
                    front_indices.add(front_index)
                    count += 1
                handle.flush()
    if count == 0:
        raise IntegrityError("at least one CacheSample is required")
    return count


def publish_bag_artifacts(
    *,
    cache_root: Path,
    domain: ResolvedDomain,
    source_bag_sha256: str,
    relative_bag_path: str,
    samples: Iterable[CacheSample],
    source_code_commit: str,
) -> BagArtifactRecord:
    if not isinstance(cache_root, Path):
        raise TypeError("cache_root must be a pathlib.Path")
    domain_identity = _validate_domain(domain)
    source_sha = _sha256(source_bag_sha256, "source_bag_sha256")
    identity = _identity_for_bag(domain_identity, source_sha)
    relative_bag_path = _relative_posix_path(relative_bag_path, "relative_bag_path")
    source_code_commit = _source_commit(source_code_commit)
    hdf5_relative, sidecar_relative = _artifact_relative_paths(
        identity.domain_id,
        identity.resolved_domain_sha256,
        identity.source_bag_sha256,
    )
    hdf5_path = cache_root / hdf5_relative
    sidecar_path = cache_root / sidecar_relative
    hdf5_partial = Path(str(hdf5_path) + ".partial")
    sidecar_partial = Path(str(sidecar_path) + ".partial")
    if _path_entry_exists(hdf5_path) or _path_entry_exists(sidecar_path):
        raise FileExistsError("final bag artifact already exists")
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iterator = iter(samples)
    except TypeError as error:
        raise TypeError("samples must be iterable") from error
    try:
        count = _write_partial_pair(
            hdf5_partial=hdf5_partial,
            sidecar_partial=sidecar_partial,
            iterator=iterator,
            identity=identity,
            source_code_commit=source_code_commit,
        )
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    _fsync_regular_file(hdf5_partial)
    _fsync_regular_file(sidecar_partial)
    _verify_artifact_pair(
        hdf5_partial,
        sidecar_partial,
        identity,
        source_code_commit,
        expected_count=count,
    )
    if _path_entry_exists(hdf5_path) or _path_entry_exists(sidecar_path):
        raise FileExistsError("final bag artifact appeared during publication")
    _replace_file(hdf5_partial, hdf5_path)
    _fsync_parent_directory(hdf5_path)
    if _path_entry_exists(sidecar_path):
        raise FileExistsError("final command sidecar appeared during publication")
    _replace_file(sidecar_partial, sidecar_path)
    _fsync_parent_directory(sidecar_path)

    record = BagArtifactRecord(
        domain_id=identity.domain_id,
        resolved_domain_sha256=identity.resolved_domain_sha256,
        sensor_preset_sha256=identity.sensor_preset_sha256,
        source_bag_sha256=identity.source_bag_sha256,
        command_rule_sha256=identity.command_rule_sha256,
        source_code_commit=source_code_commit,
        relative_bag_path=relative_bag_path,
        hdf5_path=hdf5_relative,
        hdf5_sha256=sha256_file(hdf5_path),
        sample_count=count,
        sidecar_path=sidecar_relative,
        sidecar_sha256=sha256_file(sidecar_path),
        sidecar_record_count=count,
    )
    verify_bag_artifact(record, cache_root)
    return record


def verify_bag_artifact(record: BagArtifactRecord, cache_root: Path) -> None:
    if type(record) is not BagArtifactRecord:
        raise TypeError("record must be a BagArtifactRecord")
    if not isinstance(cache_root, Path):
        raise TypeError("cache_root must be a pathlib.Path")
    try:
        record = BagArtifactRecord(**record.__dict__)
    except (TypeError, ValueError) as error:
        raise IntegrityError("bag artifact record is invalid") from error
    hdf5_path = cache_root / record.hdf5_path
    sidecar_path = cache_root / record.sidecar_path
    hdf5_identity = _require_regular_file(hdf5_path, "HDF5 artifact")
    sidecar_identity = _require_regular_file(sidecar_path, "command sidecar")
    try:
        actual_hdf5_sha = sha256_file(hdf5_path)
        actual_sidecar_sha = sha256_file(sidecar_path)
    except OSError as error:
        raise IntegrityError("unable to hash bag artifact") from error
    if hdf5_identity != _require_regular_file(hdf5_path, "HDF5 artifact"):
        raise IntegrityError("HDF5 artifact changed while it was being hashed")
    if sidecar_identity != _require_regular_file(sidecar_path, "command sidecar"):
        raise IntegrityError("command sidecar changed while it was being hashed")
    if actual_hdf5_sha != record.hdf5_sha256:
        raise IntegrityError("HDF5 SHA-256 mismatch")
    if actual_sidecar_sha != record.sidecar_sha256:
        raise IntegrityError("command sidecar SHA-256 mismatch")
    identity = CommandSidecarIdentity(
        schema_version=2,
        domain_id=record.domain_id,
        resolved_domain_sha256=record.resolved_domain_sha256,
        sensor_preset_sha256=record.sensor_preset_sha256,
        source_bag_sha256=record.source_bag_sha256,
        command_rule_sha256=record.command_rule_sha256,
        provenance="future_trajectory_oracle",
    )
    count = _verify_artifact_pair(
        hdf5_path,
        sidecar_path,
        identity,
        record.source_code_commit,
        expected_count=record.sample_count,
    )
    if hdf5_identity != _require_regular_file(hdf5_path, "HDF5 artifact"):
        raise IntegrityError("HDF5 artifact changed during structural verification")
    if sidecar_identity != _require_regular_file(sidecar_path, "command sidecar"):
        raise IntegrityError("command sidecar changed during structural verification")
    if count != record.sidecar_record_count:
        raise IntegrityError("sidecar record count mismatch")
