from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Dict, Mapping, Tuple

from morai_ddv2.errors import IntegrityError


_INT64_MAX = (1 << 63) - 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMAND_NAMES = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
COMMAND_RECORD_FIELDS = (
    "schema_version",
    "domain_id",
    "resolved_domain_sha256",
    "sensor_preset_sha256",
    "source_bag_sha256",
    "command_rule_sha256",
    "timestamp_ns",
    "grid_receipt_ns",
    "front_receipt_ns",
    "front_record_index",
    "command_index",
    "command_name",
    "provenance",
)


def _schema_version(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("schema_version must be an actual Python int")
    if value != 2:
        raise ValueError("schema_version must be 2")
    return value


def _identity_component(value: Any, name: str) -> str:
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


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError("{} must be a string".format(name))
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "{} must be 64 lowercase hexadecimal characters".format(name)
        )
    return value


def _provenance(value: Any) -> str:
    if type(value) is not str:
        raise TypeError("provenance must be a string")
    if value != "future_trajectory_oracle":
        raise ValueError("provenance must be future_trajectory_oracle")
    return value


def _nonnegative_int64(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError("{} must be an actual Python int".format(name))
    if value < 0 or value > _INT64_MAX:
        raise ValueError("{} must be a nonnegative signed-int64 value".format(name))
    return value


@dataclass(frozen=True)
class CommandSidecarIdentity:
    schema_version: int
    domain_id: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    source_bag_sha256: str
    command_rule_sha256: str
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "domain_id", _identity_component(self.domain_id, "domain_id"))
        for name in (
            "resolved_domain_sha256",
            "sensor_preset_sha256",
            "source_bag_sha256",
            "command_rule_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "provenance", _provenance(self.provenance))


@dataclass(frozen=True)
class CommandRecord:
    schema_version: int
    domain_id: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    source_bag_sha256: str
    command_rule_sha256: str
    timestamp_ns: int
    grid_receipt_ns: int
    front_receipt_ns: int
    front_record_index: int
    command_index: int
    command_name: str
    provenance: str

    def __post_init__(self) -> None:
        identity = CommandSidecarIdentity(
            schema_version=self.schema_version,
            domain_id=self.domain_id,
            resolved_domain_sha256=self.resolved_domain_sha256,
            sensor_preset_sha256=self.sensor_preset_sha256,
            source_bag_sha256=self.source_bag_sha256,
            command_rule_sha256=self.command_rule_sha256,
            provenance=self.provenance,
        )
        for name in (
            "schema_version",
            "domain_id",
            "resolved_domain_sha256",
            "sensor_preset_sha256",
            "source_bag_sha256",
            "command_rule_sha256",
            "provenance",
        ):
            object.__setattr__(self, name, getattr(identity, name))
        for name in (
            "timestamp_ns",
            "grid_receipt_ns",
            "front_receipt_ns",
            "front_record_index",
            "command_index",
        ):
            object.__setattr__(
                self, name, _nonnegative_int64(getattr(self, name), name)
            )
        if self.command_index >= len(_COMMAND_NAMES):
            raise ValueError("command_index must be in [0, 3]")
        if type(self.command_name) is not str:
            raise TypeError("command_name must be a string")
        if self.command_name != _COMMAND_NAMES[self.command_index]:
            raise ValueError("command_name does not match command_index")


def _record_mapping(record: CommandRecord) -> Dict[str, Any]:
    return {name: getattr(record, name) for name in COMMAND_RECORD_FIELDS}


def _command_record_bytes(record: CommandRecord) -> bytes:
    if type(record) is not CommandRecord:
        raise TypeError("record must be a CommandRecord")
    return (
        json.dumps(
            _record_mapping(record),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_command_record(stream: BinaryIO, record: CommandRecord) -> None:
    stream.write(_command_record_bytes(record))


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number: {}".format(value))


def _unique_object(pairs: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _record_identity(record: CommandRecord) -> CommandSidecarIdentity:
    return CommandSidecarIdentity(
        schema_version=record.schema_version,
        domain_id=record.domain_id,
        resolved_domain_sha256=record.resolved_domain_sha256,
        sensor_preset_sha256=record.sensor_preset_sha256,
        source_bag_sha256=record.source_bag_sha256,
        command_rule_sha256=record.command_rule_sha256,
        provenance=record.provenance,
    )


def read_command_sidecar(
    path: Path,
    expected_identity: CommandSidecarIdentity,
) -> Mapping[int, CommandRecord]:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if type(expected_identity) is not CommandSidecarIdentity:
        raise TypeError("expected_identity must be a CommandSidecarIdentity")
    try:
        expected_identity = CommandSidecarIdentity(**expected_identity.__dict__)
    except (TypeError, ValueError) as error:
        raise IntegrityError("invalid expected sidecar identity") from error
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise IntegrityError("command sidecar is not readable: {}".format(path)) from error
    if not payload or not payload.endswith(b"\n"):
        raise IntegrityError("command sidecar must be nonempty and end with newline")

    raw_lines = payload.split(b"\n")
    if raw_lines[-1] != b"" or any(not line for line in raw_lines[:-1]):
        raise IntegrityError("command sidecar contains a blank record")

    records: Dict[int, CommandRecord] = {}
    previous_timestamp = None
    for row_index, raw_line in enumerate(raw_lines[:-1]):
        try:
            text = raw_line.decode("utf-8", errors="strict")
            value = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(value, dict):
                raise ValueError("record must be an object")
            if tuple(value.keys()) != COMMAND_RECORD_FIELDS:
                raise ValueError("record keys or key order do not match schema")
            record = CommandRecord(**value)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise IntegrityError(
                "invalid command sidecar record {}".format(row_index)
            ) from error
        if _command_record_bytes(record) != raw_line + b"\n":
            raise IntegrityError(
                "command sidecar record {} is not deterministically encoded".format(
                    row_index
                )
            )
        if _record_identity(record) != expected_identity:
            raise IntegrityError(
                "command sidecar identity mismatch at row {}".format(row_index)
            )
        if previous_timestamp is not None and record.timestamp_ns <= previous_timestamp:
            raise IntegrityError("command sidecar timestamps must be strictly increasing")
        previous_timestamp = record.timestamp_ns
        records[record.timestamp_ns] = record
    return MappingProxyType(records)
