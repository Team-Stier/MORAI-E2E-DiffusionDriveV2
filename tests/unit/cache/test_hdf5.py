from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from morai_ddv2.cache import CACHE_DATASET_NAMES, CacheSample
from morai_ddv2.cache.hdf5 import (
    BagArtifactRecord,
    HDF5_DATASETS,
    publish_bag_artifacts,
    verify_bag_artifact,
)
from morai_ddv2.cache.sidecar import (
    CommandSidecarIdentity,
    read_command_sidecar,
)
from morai_ddv2.config import (
    canonical_json_bytes,
    load_resolved_domain,
    sha256_bytes,
    sha256_file,
)
from morai_ddv2.config.models import ResolvedDomain, thaw_json
from morai_ddv2.errors import ConfigurationError, IntegrityError
from morai_ddv2.preprocess.trajectory import Command


ROOT = Path(__file__).resolve().parents[3]
DOMAIN_PATH = ROOT / "tests/fixtures/golden/resolved-domain.json"
SOURCE_SHA = "a" * 64
SOURCE_COMMIT = "b" * 40
RELATIVE_BAG_PATH = "session/high-clock.bag"
COMMAND_RULE_SHA = "84be65c810e6480117d9c7a31023adc3490e60b2e71f2a40c6c167ec0ae88791"
HIGH_TIMESTAMP = (1 << 53) + 1
HIGH_GRID = 1_700_000_000_000_000_000

EXPECTED_DATASETS = {
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


class SingleUseIterable:
    def __init__(self, values):
        self.values = tuple(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("sample stream was consumed more than once")
        return iter(self.values)


class CloseTrackingIterator:
    def __init__(self, values):
        self._iterator = iter(values)
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self.close_calls += 1


def _domain():
    return load_resolved_domain(DOMAIN_PATH)


def _sample(
    *,
    timestamp_ns=HIGH_TIMESTAMP,
    grid_receipt_ns=HIGH_GRID,
    front_record_index=10,
    command=Command.LEFT,
    marker=1,
):
    front_receipt = grid_receipt_ns + 1
    headers = (
        timestamp_ns,
        timestamp_ns + 10,
        timestamp_ns - 10,
        timestamp_ns + 20,
    )
    receipts = (
        front_receipt,
        front_receipt + 5,
        front_receipt - 5,
        front_receipt + 10,
    )
    anchor_header = timestamp_ns + 1_000
    anchor_receipt = front_receipt + 100
    anchor_index = 1_000 + front_record_index
    status = np.asarray(
        [
            1.0 if int(command) == index else 0.0
            for index in range(4)
        ]
        + [3.0, -1.0, 0.5, -0.5],
        dtype=np.float32,
    )
    camera = np.full((256, 1024, 3), marker, dtype=np.uint8)
    lidar = np.zeros((256, 256), dtype=np.uint8)
    lidar[marker, marker] = np.uint8(min(marker, 5))
    trajectory = np.zeros((8, 3), dtype=np.float32)
    trajectory[:, 0] = np.arange(1, 9, dtype=np.float32) + marker
    future_headers = np.asarray(
        [
            [anchor_header + 500_000_000 * index] * 2
            for index in range(1, 9)
        ],
        dtype=np.int64,
    )
    future_receipts = np.asarray(
        [
            [anchor_receipt + 500_000_000 * index] * 2
            for index in range(1, 9)
        ],
        dtype=np.int64,
    )
    future_indices = np.asarray(
        [[anchor_index + index, anchor_index + index] for index in range(1, 9)],
        dtype=np.int64,
    )
    return CacheSample(
        camera_rgb=camera,
        lidar_count=lidar,
        status=status,
        trajectory=trajectory,
        timestamp_ns=timestamp_ns,
        grid_receipt_ns=grid_receipt_ns,
        sensor_source_header_stamp_ns=headers,
        sensor_source_receipt_ns=receipts,
        sensor_source_record_index=(
            front_record_index,
            front_record_index + 100,
            front_record_index + 200,
            front_record_index + 300,
        ),
        side_camera_header_offset_ns=(10, -10),
        sensor_receipt_offset_ns=(5, -5, 10),
        trajectory_anchor_tf_header_ns=anchor_header,
        trajectory_anchor_tf_receipt_ns=anchor_receipt,
        trajectory_anchor_tf_record_index=anchor_index,
        trajectory_anchor_tf_clock_epoch_index=2,
        ego_bracket_clock_epoch_index=3,
        ego_bracket_header_stamp_ns=(anchor_header - 10, anchor_header + 10),
        ego_bracket_receipt_ns=(anchor_receipt - 10, anchor_receipt + 10),
        ego_bracket_record_index=(anchor_index - 2, anchor_index - 1),
        ego_interp_weight=0.5,
        current_pose_bracket_header_stamp_ns=(anchor_header, anchor_header),
        current_pose_bracket_receipt_ns=(anchor_receipt, anchor_receipt),
        current_pose_bracket_record_index=(anchor_index, anchor_index),
        current_pose_interp_weight=0.0,
        future_pose_bracket_header_stamp_ns=future_headers,
        future_pose_bracket_receipt_ns=future_receipts,
        future_pose_bracket_record_index=future_indices,
        future_pose_interp_weight=np.zeros(8, dtype=np.float32),
        command=command,
    )


def _samples():
    return (
        _sample(),
        _sample(
            timestamp_ns=HIGH_TIMESTAMP + 500_000_000,
            grid_receipt_ns=HIGH_GRID + 500_000_000,
            front_record_index=11,
            command=Command.RIGHT,
            marker=2,
        ),
    )


def _publish_args(tmp_path, samples=None):
    return {
        "cache_root": tmp_path,
        "domain": _domain(),
        "source_bag_sha256": SOURCE_SHA,
        "relative_bag_path": RELATIVE_BAG_PATH,
        "samples": _samples() if samples is None else samples,
        "source_code_commit": SOURCE_COMMIT,
    }


def _expected_paths(cache_root):
    domain = _domain()
    hdf5 = (
        cache_root
        / "bags"
        / domain.document["domain_id"]
        / domain.sha256
        / (SOURCE_SHA + ".h5")
    )
    sidecar = (
        cache_root
        / "commands"
        / domain.document["domain_id"]
        / domain.sha256
        / (SOURCE_SHA + ".jsonl")
    )
    return hdf5, sidecar


def _identity():
    domain = _domain()
    return CommandSidecarIdentity(
        schema_version=2,
        domain_id=domain.document["domain_id"],
        resolved_domain_sha256=domain.sha256,
        sensor_preset_sha256=domain.document["sensor_preset"]["sha256"],
        source_bag_sha256=SOURCE_SHA,
        command_rule_sha256=COMMAND_RULE_SHA,
        provenance="future_trajectory_oracle",
    )


def test_task10_hdf5_public_surface_is_exact() -> None:
    assert [field.name for field in fields(BagArtifactRecord)] == [
        "domain_id",
        "resolved_domain_sha256",
        "sensor_preset_sha256",
        "source_bag_sha256",
        "command_rule_sha256",
        "source_code_commit",
        "relative_bag_path",
        "hdf5_path",
        "hdf5_sha256",
        "sample_count",
        "sidecar_path",
        "sidecar_sha256",
        "sidecar_record_count",
    ]
    assert tuple(HDF5_DATASETS) == CACHE_DATASET_NAMES
    assert HDF5_DATASETS == EXPECTED_DATASETS
    assert callable(publish_bag_artifacts)
    assert callable(verify_bag_artifact)


def test_two_sample_one_pass_round_trip_is_exact(tmp_path) -> None:
    samples = _samples()
    single_use = SingleUseIterable(samples)

    record = publish_bag_artifacts(**_publish_args(tmp_path, single_use))

    assert single_use.iterations == 1
    hdf5_path = tmp_path / record.hdf5_path
    sidecar_path = tmp_path / record.sidecar_path
    expected_hdf5, expected_sidecar = _expected_paths(tmp_path)
    assert hdf5_path == expected_hdf5
    assert sidecar_path == expected_sidecar
    assert record.relative_bag_path == RELATIVE_BAG_PATH
    assert record.command_rule_sha256 == COMMAND_RULE_SHA
    assert record.sample_count == record.sidecar_record_count == 2
    assert record.hdf5_sha256 == sha256_file(hdf5_path)
    assert record.sidecar_sha256 == sha256_file(sidecar_path)
    assert not Path(str(hdf5_path) + ".partial").exists()
    assert not Path(str(sidecar_path) + ".partial").exists()

    with h5py.File(str(hdf5_path), "r") as handle:
        assert set(handle.keys()) == set(CACHE_DATASET_NAMES)
        assert set(handle.attrs) == {
            "cache_schema_version",
            "clock_policy_version",
            "domain_id",
            "sensor_preset_sha256",
            "resolved_domain_sha256",
            "source_bag_sha256",
            "source_code_commit",
        }
        assert handle.attrs["cache_schema_version"] == 2
        assert handle.attrs["clock_policy_version"] == 1
        assert handle.attrs["domain_id"] == _domain().document["domain_id"]
        assert handle.attrs["resolved_domain_sha256"] == _domain().sha256
        assert handle.attrs["source_bag_sha256"] == SOURCE_SHA
        assert handle.attrs["source_code_commit"] == SOURCE_COMMIT
        for name, (dtype, trailing_shape) in EXPECTED_DATASETS.items():
            dataset = handle[name]
            assert dataset.dtype == np.dtype(dtype), name
            assert dataset.shape == (2,) + trailing_shape, name
            assert dataset.maxshape == (None,) + trailing_shape, name
            assert dataset.chunks == (1,) + trailing_shape, name
            assert dataset.compression is None, name
            for row_index, sample in enumerate(samples):
                expected = getattr(sample, name)
                assert np.array_equal(dataset[row_index], expected), name

    commands = read_command_sidecar(sidecar_path, _identity())
    assert tuple(commands) == tuple(sample.timestamp_ns for sample in samples)
    assert commands[HIGH_TIMESTAMP].timestamp_ns == HIGH_TIMESTAMP
    assert HIGH_TIMESTAMP > 2**53
    assert commands[HIGH_TIMESTAMP].command_name == "LEFT"
    assert commands[HIGH_TIMESTAMP + 500_000_000].command_name == "RIGHT"
    assert [
        SOURCE_SHA + ":" + str(timestamp) for timestamp in commands
    ] == [
        SOURCE_SHA + ":" + str(HIGH_TIMESTAMP),
        SOURCE_SHA + ":" + str(HIGH_TIMESTAMP + 500_000_000),
    ]
    verify_bag_artifact(record, tmp_path)


def test_publisher_closes_sample_stream_after_success(tmp_path) -> None:
    stream = CloseTrackingIterator(_samples())

    publish_bag_artifacts(**_publish_args(tmp_path, stream))

    assert stream.close_calls == 1


@pytest.mark.parametrize(
    ("changes", "expected_error"),
    [
        ({"source_bag_sha256": "A" * 64}, ValueError),
        ({"relative_bag_path": "/absolute.bag"}, ValueError),
        ({"relative_bag_path": "."}, ValueError),
        ({"relative_bag_path": "session/../escape.bag"}, ValueError),
        ({"relative_bag_path": "session\\bag.bag"}, ValueError),
        ({"relative_bag_path": "session/\x00.bag"}, ValueError),
        ({"relative_bag_path": "session/\ud800.bag"}, ValueError),
        ({"source_code_commit": "B" * 40}, ValueError),
        ({"samples": 1}, TypeError),
        ({"cache_root": "cache"}, TypeError),
        ({"domain": object()}, TypeError),
    ],
)
def test_publisher_rejects_invalid_identity_path_commit_or_api_input(
    tmp_path, changes, expected_error
) -> None:
    arguments = _publish_args(tmp_path)
    arguments.update(changes)

    with pytest.raises(expected_error):
        publish_bag_artifacts(**arguments)


@pytest.mark.parametrize(
    "drift",
    [
        "dataset",
        "command",
        "atomic_publication",
        "sensor_axis_order",
        "sample_id_template",
        "clock_policy_version",
    ],
)
def test_publisher_rejects_resolved_domain_contract_drift(tmp_path, drift) -> None:
    document = thaw_json(_domain().document)
    if drift == "dataset":
        document["cache"]["datasets"]["status"]["dtype"] = "float64"
    elif drift == "command":
        document["command"]["thresholds"]["minimum_path_length_m"] = 2
    elif drift == "atomic_publication":
        document["cache"]["atomic_publication"]["file_fsync"] = False
    elif drift == "sensor_axis_order":
        document["cache"]["sensor_axis_order"] = [
            "left",
            "front",
            "right",
            "lidar",
        ]
    elif drift == "sample_id_template":
        document["cache"]["sample_id_template"] = "{timestamp_ns}"
    else:
        document["sampling"]["clock_policy"]["version"] = 2
    drifted = ResolvedDomain(
        document=document,
        sha256=sha256_bytes(canonical_json_bytes(document)),
    )

    with pytest.raises(ConfigurationError):
        publish_bag_artifacts(**dict(_publish_args(tmp_path), domain=drifted))


@pytest.mark.parametrize(
    "changes",
    [
        {"source_code_commit": "B" * 40},
        {"domain_id": "bad\x00domain"},
        {"domain_id": "bad\ud800domain"},
        {"relative_bag_path": "../escape.bag"},
        {"hdf5_path": "bags/wrong.h5"},
        {"sidecar_path": "commands/wrong.jsonl"},
        {"hdf5_sha256": "A" * 64},
        {"sample_count": True},
        {"sidecar_record_count": 3},
    ],
)
def test_bag_artifact_record_constructor_rejects_drift(tmp_path, changes) -> None:
    record = publish_bag_artifacts(**_publish_args(tmp_path))

    with pytest.raises((TypeError, ValueError)):
        replace(record, **changes)


@pytest.mark.parametrize("domain_id", ["bad\x00domain", "bad\ud800domain"])
def test_bag_artifact_record_rejects_non_utf8_safe_domain_component(
    tmp_path, domain_id
) -> None:
    record = publish_bag_artifacts(**_publish_args(tmp_path))
    values = dict(record.__dict__)
    values["domain_id"] = domain_id
    values["hdf5_path"] = "bags/{}/{}/{}.h5".format(
        domain_id, record.resolved_domain_sha256, record.source_bag_sha256
    )
    values["sidecar_path"] = "commands/{}/{}/{}.jsonl".format(
        domain_id, record.resolved_domain_sha256, record.source_bag_sha256
    )

    with pytest.raises(ValueError):
        BagArtifactRecord(**values)


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("empty", IntegrityError),
        ("wrong_type", TypeError),
        ("duplicate_timestamp", IntegrityError),
        ("decreasing_timestamp", IntegrityError),
        ("duplicate_grid", IntegrityError),
        ("decreasing_grid", IntegrityError),
        ("reused_front", IntegrityError),
        ("status_command_drift", IntegrityError),
    ],
)
def test_publisher_rejects_invalid_sample_stream(tmp_path, case, expected_error) -> None:
    first, second = _samples()
    if case == "empty":
        values = ()
    elif case == "wrong_type":
        values = (object(),)
    elif case == "duplicate_timestamp":
        second = _sample(
            timestamp_ns=first.timestamp_ns,
            grid_receipt_ns=second.grid_receipt_ns,
            front_record_index=11,
            command=Command.RIGHT,
            marker=2,
        )
        values = (first, second)
    elif case == "decreasing_timestamp":
        lowered = first.timestamp_ns - 1
        second = _sample(
            timestamp_ns=lowered,
            grid_receipt_ns=second.grid_receipt_ns,
            front_record_index=11,
            command=Command.RIGHT,
            marker=2,
        )
        values = (first, second)
    elif case == "duplicate_grid":
        object.__setattr__(second, "grid_receipt_ns", first.grid_receipt_ns)
        values = (first, second)
    elif case == "decreasing_grid":
        object.__setattr__(second, "grid_receipt_ns", first.grid_receipt_ns - 1)
        values = (first, second)
    elif case == "reused_front":
        indices = second.sensor_source_record_index
        object.__setattr__(
            second,
            "sensor_source_record_index",
            (first.sensor_source_record_index[0],) + indices[1:],
        )
        values = (first, second)
    else:
        changed = second.status.copy()
        changed.setflags(write=True)
        changed[:4] = (1.0, 0.0, 0.0, 0.0)
        changed.setflags(write=False)
        object.__setattr__(second, "status", changed)
        values = (first, second)

    with pytest.raises(expected_error):
        publish_bag_artifacts(**_publish_args(tmp_path, values))
    hdf5_path, sidecar_path = _expected_paths(tmp_path)
    assert not hdf5_path.exists()
    assert not sidecar_path.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_hdf5",
        "missing_sidecar",
        "hdf5_hash",
        "sidecar_hash",
        "extra_dataset",
        "extra_attr",
        "float_schema_attr",
        "dataset_count",
        "invalid_data",
        "dataset_group",
        "dataset_link",
        "cross_link",
        "sidecar_cross_link",
    ],
)
def test_verifier_rejects_missing_hash_structure_and_cross_link(
    tmp_path, corruption
) -> None:
    record = publish_bag_artifacts(**_publish_args(tmp_path))
    hdf5_path = tmp_path / record.hdf5_path
    sidecar_path = tmp_path / record.sidecar_path
    if corruption == "missing_hdf5":
        hdf5_path.unlink()
    elif corruption == "missing_sidecar":
        sidecar_path.unlink()
    elif corruption == "hdf5_hash":
        record = replace(record, hdf5_sha256="0" * 64)
    elif corruption == "sidecar_hash":
        record = replace(record, sidecar_sha256="0" * 64)
    elif corruption == "sidecar_cross_link":
        rows = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
        rows[0]["front_receipt_ns"] += 1
        sidecar_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n" for row in rows
            )
        )
        record = replace(record, sidecar_sha256=sha256_file(sidecar_path))
    else:
        with h5py.File(str(hdf5_path), "r+") as handle:
            if corruption == "extra_dataset":
                handle.create_dataset("unexpected", data=np.asarray([1]))
            elif corruption == "extra_attr":
                handle.attrs["unexpected"] = "value"
            elif corruption == "float_schema_attr":
                handle.attrs["cache_schema_version"] = 2.0
            elif corruption == "dataset_count":
                handle["timestamp_ns"].resize((1,))
            elif corruption == "invalid_data":
                handle["lidar_count"][0, 0, 0] = 6
            elif corruption == "dataset_group":
                del handle["timestamp_ns"]
                handle.create_group("timestamp_ns")
            elif corruption == "dataset_link":
                del handle["timestamp_ns"]
                handle["timestamp_ns"] = h5py.SoftLink("/grid_receipt_ns")
            else:
                handle["grid_receipt_ns"][0] += 1
        record = replace(record, hdf5_sha256=sha256_file(hdf5_path))

    with pytest.raises(IntegrityError):
        verify_bag_artifact(record, tmp_path)


def test_task10_modules_are_ros_free() -> None:
    code = r'''
import builtins, json
blocked = {"rosbag", "rospy", "sensor_msgs", "tf2_msgs", "morai_msgs"}
seen = []
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        seen.append(name)
        raise AssertionError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import morai_ddv2.cache.hdf5
import morai_ddv2.cache.sidecar
print(json.dumps(seen))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
