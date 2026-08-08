from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from morai_ddv2.errors import IntegrityError
from morai_ddv2.cache.sidecar import (
    CommandRecord,
    CommandSidecarIdentity,
    read_command_sidecar,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
HIGH_TIMESTAMP = (1 << 53) + 1
FIELD_NAMES = (
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


def _identity(**changes):
    values = {
        "schema_version": 2,
        "domain_id": "morai-ddv2-v1-oracle-command",
        "resolved_domain_sha256": SHA_A,
        "sensor_preset_sha256": SHA_B,
        "source_bag_sha256": SHA_C,
        "command_rule_sha256": SHA_D,
        "provenance": "future_trajectory_oracle",
    }
    values.update(changes)
    return CommandSidecarIdentity(**values)


def _record_dict(**changes):
    values = {
        "schema_version": 2,
        "domain_id": "morai-ddv2-v1-oracle-command",
        "resolved_domain_sha256": SHA_A,
        "sensor_preset_sha256": SHA_B,
        "source_bag_sha256": SHA_C,
        "command_rule_sha256": SHA_D,
        "timestamp_ns": HIGH_TIMESTAMP,
        "grid_receipt_ns": 1_700_000_000_000_000_000,
        "front_receipt_ns": 1_700_000_000_000_000_001,
        "front_record_index": 10,
        "command_index": 0,
        "command_name": "LEFT",
        "provenance": "future_trajectory_oracle",
    }
    values.update(changes)
    return values


def _line(values):
    return (
        json.dumps(
            values,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write(path: Path, *records) -> bytes:
    payload = b"".join(_line(record) for record in records)
    path.write_bytes(payload)
    return payload


def test_task10_sidecar_public_surface_is_exact() -> None:
    assert [field.name for field in fields(CommandSidecarIdentity)] == [
        "schema_version",
        "domain_id",
        "resolved_domain_sha256",
        "sensor_preset_sha256",
        "source_bag_sha256",
        "command_rule_sha256",
        "provenance",
    ]
    assert [field.name for field in fields(CommandRecord)] == [
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
    ]
    assert callable(read_command_sidecar)


def test_reader_preserves_large_integer_and_returns_immutable_mapping(tmp_path) -> None:
    path = tmp_path / "commands.jsonl"
    expected_bytes = _write(path, _record_dict())

    records = read_command_sidecar(path, _identity())

    assert path.read_bytes() == expected_bytes
    assert tuple(records) == (HIGH_TIMESTAMP,)
    assert records[HIGH_TIMESTAMP] == CommandRecord(**_record_dict())
    assert records[HIGH_TIMESTAMP].timestamp_ns > 2**53
    with pytest.raises(TypeError):
        records[HIGH_TIMESTAMP + 1] = records[HIGH_TIMESTAMP]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload[:-1],
        lambda payload: payload + b"\n",
        lambda payload: payload.replace(b'{"schema_version"', b'{ "schema_version"'),
        lambda payload: payload.replace(
            b'{"schema_version":2,',
            b'{"schema_version":2,"schema_version":2,',
        ),
        lambda payload: payload.replace(
            b'"front_record_index":10', b'"front_record_index":NaN'
        ),
        lambda payload: payload.replace(
            b'"command_name":"LEFT"', b'"extra":0,"command_name":"LEFT"'
        ),
        lambda payload: payload.replace(b'"command_name":"LEFT",', b""),
        lambda payload: payload[:-2] + b"\xff\n",
        lambda payload: payload.replace(
            b'"domain_id":"morai-ddv2-v1-oracle-command"',
            b'"domain_id":"\\ud800"',
        ),
    ],
    ids=(
        "missing-final-newline",
        "blank-record",
        "alternate-whitespace",
        "duplicate-key",
        "non-finite-number",
        "extra-key",
        "missing-key",
        "invalid-utf8",
        "unpaired-surrogate",
    ),
)
def test_reader_rejects_noncanonical_or_malformed_bytes(tmp_path, mutator) -> None:
    path = tmp_path / "commands.jsonl"
    payload = _line(_record_dict())
    path.write_bytes(mutator(payload))

    with pytest.raises(IntegrityError):
        read_command_sidecar(path, _identity())


def test_reader_rejects_reordered_keys(tmp_path) -> None:
    path = tmp_path / "commands.jsonl"
    values = _record_dict()
    reordered = {name: values[name] for name in reversed(FIELD_NAMES)}
    _write(path, reordered)

    with pytest.raises(IntegrityError):
        read_command_sidecar(path, _identity())


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 1},
        {"domain_id": "wrong-domain"},
        {"resolved_domain_sha256": "0" * 64},
        {"sensor_preset_sha256": "0" * 64},
        {"source_bag_sha256": "0" * 64},
        {"command_rule_sha256": "0" * 64},
        {"provenance": "manual"},
        {"command_index": 1},
        {"command_name": "STRAIGHT"},
        {"timestamp_ns": True},
        {"front_record_index": -1},
    ],
)
def test_reader_rejects_invalid_record_or_identity_drift(tmp_path, changes) -> None:
    path = tmp_path / "commands.jsonl"
    _write(path, _record_dict(**changes))

    with pytest.raises(IntegrityError):
        read_command_sidecar(path, _identity())


@pytest.mark.parametrize("second_timestamp", [HIGH_TIMESTAMP, HIGH_TIMESTAMP - 1])
def test_reader_rejects_duplicate_or_decreasing_timestamps(
    tmp_path, second_timestamp
) -> None:
    path = tmp_path / "commands.jsonl"
    _write(
        path,
        _record_dict(),
        _record_dict(
            timestamp_ns=second_timestamp,
            grid_receipt_ns=1_700_000_000_500_000_000,
            front_receipt_ns=1_700_000_000_500_000_001,
            front_record_index=11,
        ),
    )

    with pytest.raises(IntegrityError):
        read_command_sidecar(path, _identity())


@pytest.mark.parametrize(
    ("record_changes", "error_type"),
    [
        ({"schema_version": True}, TypeError),
        ({"timestamp_ns": object()}, TypeError),
        ({"grid_receipt_ns": -1}, ValueError),
        ({"command_index": 4}, ValueError),
        ({"command_name": "UNKNOWN"}, ValueError),
        ({"resolved_domain_sha256": "A" * 64}, ValueError),
    ],
)
def test_command_record_constructor_is_strict(record_changes, error_type) -> None:
    with pytest.raises(error_type):
        CommandRecord(**_record_dict(**record_changes))


@pytest.mark.parametrize(
    ("identity_changes", "error_type"),
    [
        ({"schema_version": True}, TypeError),
        ({"schema_version": 1}, ValueError),
        ({"domain_id": ""}, ValueError),
        ({"domain_id": "bad\x00domain"}, ValueError),
        ({"domain_id": "bad\ud800domain"}, ValueError),
        ({"source_bag_sha256": "A" * 64}, ValueError),
        ({"provenance": "manual"}, ValueError),
    ],
)
def test_sidecar_identity_constructor_is_strict(identity_changes, error_type) -> None:
    with pytest.raises(error_type):
        _identity(**identity_changes)
