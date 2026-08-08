from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import cv2
import numpy as np
import pytest

from morai_ddv2.config import load_resolved_domain
from morai_ddv2.errors import IntegrityError, MoraiDDV2Error
from morai_ddv2.cache import CACHE_DATASET_NAMES, CacheSample
from morai_ddv2.preprocess.camera import CameraResolutionError
from morai_ddv2.preprocess.diagnostics import EgoDiagnosticValue
from morai_ddv2.preprocess.geometry import Pose
from morai_ddv2.preprocess.materialize import (
    MaterializationStream,
    NoValidSampleError,
    materialize_samples,
)
from morai_ddv2.preprocess.planner import (
    BagSamplePlan,
    PlannedBracket,
    PlannedSample,
    SampleDrop,
    SampleDropReason,
    build_sample_plan,
)
from morai_ddv2.preprocess.trajectory import Command, InvalidCommand
from morai_ddv2.ros1 import (
    EgoRecord,
    EgoValue,
    RecordRef,
    SensorPayload,
    SensorPayloadFailure,
    SensorPayloadKind,
    TfPoseRecord,
    index_bag,
)


ROOT = Path(__file__).resolve().parents[3]
DOMAIN = load_resolved_domain(ROOT / "tests/fixtures/golden/resolved-domain.json")
SYNTHETIC_SOURCE = ROOT / "tests/fixtures/golden/resolved-domain.json"


def _valid_cache_kwargs():
    camera = np.zeros((256, 1024, 3), dtype=np.uint8)
    camera[0, 0] = (1, 2, 3)
    lidar = np.zeros((256, 256), dtype=np.uint8)
    lidar[0, 0] = 5
    status = np.asarray([0, 1, 0, 0, 3, -1, 0.5, -0.5], dtype=np.float32)
    trajectory = np.zeros((8, 3), dtype=np.float32)
    future_header = np.asarray(
        [[2_500 + 500 * index, 2_500 + 500 * index] for index in range(8)],
        dtype=np.int64,
    )
    future_receipt = np.asarray(
        [[3_500 + 500 * index, 3_500 + 500 * index] for index in range(8)],
        dtype=np.int64,
    )
    future_index = np.asarray(
        [[20 + index, 20 + index] for index in range(8)], dtype=np.int64
    )
    return {
        "camera_rgb": camera,
        "lidar_count": lidar,
        "status": status,
        "trajectory": trajectory,
        "timestamp_ns": 100,
        "grid_receipt_ns": 1_000,
        "sensor_source_header_stamp_ns": (100, 110, 90, 777),
        "sensor_source_receipt_ns": (1_010, 1_015, 1_005, 1_020),
        "sensor_source_record_index": (1, 2, 3, 4),
        "side_camera_header_offset_ns": (10, -10),
        "sensor_receipt_offset_ns": (5, -5, 10),
        "trajectory_anchor_tf_header_ns": 2_000,
        "trajectory_anchor_tf_receipt_ns": 2_100,
        "trajectory_anchor_tf_record_index": 10,
        "trajectory_anchor_tf_clock_epoch_index": 2,
        "ego_bracket_clock_epoch_index": 3,
        "ego_bracket_header_stamp_ns": (1_990, 2_010),
        "ego_bracket_receipt_ns": (2_090, 2_110),
        "ego_bracket_record_index": (8, 9),
        "ego_interp_weight": 0.5,
        "current_pose_bracket_header_stamp_ns": (2_000, 2_000),
        "current_pose_bracket_receipt_ns": (2_100, 2_100),
        "current_pose_bracket_record_index": (10, 10),
        "current_pose_interp_weight": 0.0,
        "future_pose_bracket_header_stamp_ns": future_header,
        "future_pose_bracket_receipt_ns": future_receipt,
        "future_pose_bracket_record_index": future_index,
        "future_pose_interp_weight": np.zeros(8, dtype=np.float32),
        "command": Command.STRAIGHT,
    }


def _sample(**changes):
    values = _valid_cache_kwargs()
    values.update(changes)
    return CacheSample(**values)


def test_cache_sample_has_exact_29_python_fields_and_28_dataset_fields() -> None:
    names = [field.name for field in fields(CacheSample)]
    assert len(names) == 29
    assert names == [
        "camera_rgb",
        "lidar_count",
        "status",
        "trajectory",
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
        "command",
    ]
    assert len([name for name in names if name != "command"]) == 28
    assert callable(materialize_samples)
    assert hasattr(MaterializationStream, "__next__")


def test_task9_modules_import_without_ros_packages() -> None:
    code = r'''
import builtins, json
blocked = {"rosbag", "rospy", "sensor_msgs", "tf2_msgs", "morai_msgs"}
seen = []
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    root = name.split(".", 1)[0]
    if root in blocked:
        seen.append(name)
        raise AssertionError("eager ROS import: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import morai_ddv2.cache
import morai_ddv2.preprocess.planner
import morai_ddv2.preprocess.materialize
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


def test_cache_dataset_names_are_the_exact_schema_v2_contract() -> None:
    assert CACHE_DATASET_NAMES == (
        "camera_rgb",
        "lidar_count",
        "status",
        "trajectory",
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
    )
    assert len(CACHE_DATASET_NAMES) == len(set(CACHE_DATASET_NAMES)) == 28
    assert tuple(
        field.name for field in fields(CacheSample) if field.name != "command"
    ) == CACHE_DATASET_NAMES


def test_cache_sample_happy_path_preserves_all_28_dataset_values() -> None:
    values = _valid_cache_kwargs()
    sample = CacheSample(**values)
    for name in CACHE_DATASET_NAMES:
        actual = getattr(sample, name)
        expected = values[name]
        if isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected), name
        else:
            assert actual == expected, name


def test_cache_sample_validates_contract_and_defensively_owns_arrays() -> None:
    values = _valid_cache_kwargs()
    source_camera = values["camera_rgb"]
    source_future = values["future_pose_bracket_record_index"]

    sample = CacheSample(**values)

    assert sample.sensor_source_header_stamp_ns == (100, 110, 90, 777)
    assert sample.ego_bracket_clock_epoch_index == 3
    assert sample.command is Command.STRAIGHT
    for name in (
        "camera_rgb",
        "lidar_count",
        "status",
        "trajectory",
        "future_pose_bracket_header_stamp_ns",
        "future_pose_bracket_receipt_ns",
        "future_pose_bracket_record_index",
        "future_pose_interp_weight",
    ):
        array = getattr(sample, name)
        assert array.flags.c_contiguous
        assert not array.flags.writeable

    source_camera[0, 0, 0] = 99
    source_future[0, 0] = 999
    assert sample.camera_rgb[0, 0, 0] == 1
    assert sample.future_pose_bracket_record_index[0, 0] == 20
    with pytest.raises(FrozenInstanceError):
        sample.timestamp_ns = 101


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("camera_rgb", np.zeros((256, 1024, 3), dtype=np.float32), TypeError),
        ("camera_rgb", np.asfortranarray(np.zeros((256, 1024, 3), dtype=np.uint8)), ValueError),
        ("lidar_count", np.full((256, 256), 6, dtype=np.uint8), ValueError),
        ("status", np.full(8, np.nan, dtype=np.float32), ValueError),
        ("trajectory", np.full((8, 3), np.inf, dtype=np.float32), ValueError),
        ("future_pose_bracket_header_stamp_ns", np.zeros((8, 2), dtype=np.int32), TypeError),
        ("future_pose_interp_weight", np.zeros((8, 1), dtype=np.float32), ValueError),
        ("timestamp_ns", True, TypeError),
        ("timestamp_ns", np.int64(100), TypeError),
        ("timestamp_ns", 1 << 63, OverflowError),
        ("trajectory_anchor_tf_record_index", -1, ValueError),
        ("trajectory_anchor_tf_clock_epoch_index", -1, ValueError),
        ("ego_bracket_clock_epoch_index", -1, ValueError),
        ("sensor_source_header_stamp_ns", [100, 110, 90, 777], TypeError),
        ("sensor_source_record_index", (1, 2, 3, -1), ValueError),
        ("ego_interp_weight", 0.1, ValueError),
        ("ego_interp_weight", float("nan"), ValueError),
        ("current_pose_interp_weight", -0.0, ValueError),
        ("command", 1, TypeError),
    ],
)
def test_cache_sample_rejects_invalid_field_contract(field, value, error) -> None:
    with pytest.raises(error):
        _sample(**{field: value})


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"timestamp_ns": 101}, ValueError),
        ({"side_camera_header_offset_ns": (11, -10)}, ValueError),
        ({"sensor_receipt_offset_ns": (5, -5, 11)}, ValueError),
        ({"current_pose_bracket_header_stamp_ns": (1_999, 2_000)}, ValueError),
        ({"current_pose_bracket_receipt_ns": (2_099, 2_100)}, ValueError),
        ({"current_pose_bracket_record_index": (9, 10)}, ValueError),
        ({"ego_bracket_header_stamp_ns": (2_010, 1_990)}, ValueError),
        ({"ego_bracket_header_stamp_ns": (2_000, 2_000), "ego_interp_weight": 0.5}, ValueError),
        ({"status": np.asarray([1, 0, 0, 0, 3, -1, 0.5, -0.5], dtype=np.float32)}, ValueError),
    ],
)
def test_cache_sample_rejects_cross_field_inconsistency(changes, error) -> None:
    with pytest.raises(error):
        _sample(**changes)


@pytest.mark.parametrize(
    "case",
    [
        "ego_receipt_reversed",
        "ego_index_reversed",
        "future_header_reversed",
        "future_receipt_reversed",
        "future_index_reversed",
        "future_exact_identity_mismatch",
    ],
)
def test_cache_sample_rejects_cross_bracket_order_and_identity_drift(case) -> None:
    values = _valid_cache_kwargs()
    if case == "ego_receipt_reversed":
        values["ego_bracket_receipt_ns"] = (2_110, 2_090)
    elif case == "ego_index_reversed":
        values["ego_bracket_record_index"] = (9, 8)
    else:
        headers = values["future_pose_bracket_header_stamp_ns"].copy()
        receipts = values["future_pose_bracket_receipt_ns"].copy()
        indices = values["future_pose_bracket_record_index"].copy()
        weights = values["future_pose_interp_weight"].copy()
        if case == "future_header_reversed":
            headers[0] = (2_600, 2_500)
            receipts[0] = (3_500, 3_600)
            indices[0] = (20, 21)
            weights[0] = np.float32(0.5)
        elif case == "future_receipt_reversed":
            headers[0] = (2_500, 2_600)
            receipts[0] = (3_600, 3_500)
            indices[0] = (20, 21)
            weights[0] = np.float32(0.5)
        elif case == "future_index_reversed":
            headers[0] = (2_500, 2_600)
            receipts[0] = (3_500, 3_600)
            indices[0] = (21, 20)
            weights[0] = np.float32(0.5)
        else:
            receipts[0] = (3_500, 3_501)
            indices[0] = (20, 21)
        values["future_pose_bracket_header_stamp_ns"] = headers
        values["future_pose_bracket_receipt_ns"] = receipts
        values["future_pose_bracket_record_index"] = indices
        values["future_pose_interp_weight"] = weights

    with pytest.raises(ValueError):
        CacheSample(**values)


@pytest.mark.parametrize("weight", [-0.5, 1.5, float("inf")])
def test_cache_sample_rejects_out_of_range_or_nonfinite_ego_weight(weight) -> None:
    with pytest.raises(ValueError):
        _sample(ego_interp_weight=weight)


def test_cache_sample_rejects_nonzero_weight_for_exact_future_bracket() -> None:
    weights = np.zeros(8, dtype=np.float32)
    weights[3] = np.float32(0.5)
    with pytest.raises(ValueError):
        _sample(future_pose_interp_weight=weights)


def test_cache_sample_rejects_schema_v1_keyword() -> None:
    values = _valid_cache_kwargs()
    values["sensor_source_stamp_ns"] = values.pop("sensor_source_header_stamp_ns")
    with pytest.raises(TypeError):
        CacheSample(**values)


@pytest.mark.parametrize(
    "deprecated_name",
    [
        "sensor_source_stamp_ns",
        "sensor_sync_offset_ns",
        "pose_bracket_stamp_ns",
        "pose_bracket_record_index",
        "pose_interp_weight",
        "future_pose_bracket_stamp_ns",
    ],
)
def test_cache_sample_rejects_every_deprecated_schema_v1_keyword(
    deprecated_name,
) -> None:
    values = _valid_cache_kwargs()
    values[deprecated_name] = 0
    assert deprecated_name not in CACHE_DATASET_NAMES
    with pytest.raises(TypeError):
        CacheSample(**values)


def _same_shape_noncontiguous(array):
    expanded_shape = array.shape[:-1] + (array.shape[-1] * 2,)
    expanded = np.empty(expanded_shape, dtype=array.dtype)
    result = expanded[..., ::2]
    assert result.shape == array.shape
    assert not result.flags.c_contiguous
    return result


@pytest.mark.parametrize(
    "field",
    [
        "camera_rgb",
        "lidar_count",
        "status",
        "trajectory",
        "future_pose_bracket_header_stamp_ns",
        "future_pose_bracket_receipt_ns",
        "future_pose_bracket_record_index",
        "future_pose_interp_weight",
    ],
)
def test_every_cache_array_rejects_wrong_dtype_shape_and_contiguity(field) -> None:
    value = _valid_cache_kwargs()[field]
    wrong_dtype = np.dtype(np.float64 if value.dtype != np.float64 else np.float32)
    with pytest.raises(TypeError):
        _sample(**{field: value.astype(wrong_dtype)})
    with pytest.raises(ValueError):
        _sample(**{field: np.zeros(value.shape + (1,), dtype=value.dtype)})
    with pytest.raises(ValueError):
        _sample(**{field: _same_shape_noncontiguous(value)})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lidar_count", np.full((256, 256), 6, dtype=np.uint8)),
        ("status", np.full(8, np.nan, dtype=np.float32)),
        ("trajectory", np.full((8, 3), np.inf, dtype=np.float32)),
        (
            "future_pose_bracket_header_stamp_ns",
            np.full((8, 2), -1, dtype=np.int64),
        ),
        (
            "future_pose_bracket_receipt_ns",
            np.full((8, 2), -1, dtype=np.int64),
        ),
        (
            "future_pose_bracket_record_index",
            np.full((8, 2), -1, dtype=np.int64),
        ),
        ("future_pose_interp_weight", np.full(8, np.nan, dtype=np.float32)),
        ("future_pose_interp_weight", np.full(8, -0.01, dtype=np.float32)),
        ("future_pose_interp_weight", np.full(8, 1.01, dtype=np.float32)),
    ],
)
def test_every_cache_array_range_contract_rejects_invalid_values(field, value) -> None:
    with pytest.raises(ValueError):
        _sample(**{field: value})


def test_every_cache_tuple_rejects_container_length_and_integer_type_errors() -> None:
    values = _valid_cache_kwargs()
    tuple_fields = (
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
    )
    for field in tuple_fields:
        valid = values[field]
        with pytest.raises(TypeError):
            _sample(**{field: list(valid)})
        with pytest.raises(ValueError):
            _sample(**{field: valid[:-1]})
        with pytest.raises(TypeError):
            _sample(**{field: (True,) + valid[1:]})
        with pytest.raises(TypeError):
            _sample(**{field: (np.int64(valid[0]),) + valid[1:]})
        with pytest.raises(OverflowError):
            _sample(**{field: ((1 << 63),) + valid[1:]})


def test_cache_scalar_python_int64_boundaries_and_bool_rejection() -> None:
    maximum = (1 << 63) - 1
    sample = _sample(
        trajectory_anchor_tf_record_index=maximum,
        current_pose_bracket_record_index=(maximum, maximum),
        trajectory_anchor_tf_clock_epoch_index=maximum,
        ego_bracket_clock_epoch_index=maximum,
    )
    assert sample.trajectory_anchor_tf_record_index == maximum
    assert sample.trajectory_anchor_tf_clock_epoch_index == maximum
    assert sample.ego_bracket_clock_epoch_index == maximum

    scalar_fields = (
        "timestamp_ns",
        "grid_receipt_ns",
        "trajectory_anchor_tf_header_ns",
        "trajectory_anchor_tf_receipt_ns",
        "trajectory_anchor_tf_record_index",
        "trajectory_anchor_tf_clock_epoch_index",
        "ego_bracket_clock_epoch_index",
    )
    for field in scalar_fields:
        with pytest.raises(TypeError):
            _sample(**{field: True})
        with pytest.raises(TypeError):
            _sample(**{field: np.int64(0)})
        with pytest.raises(OverflowError):
            _sample(**{field: 1 << 63})
        with pytest.raises(ValueError):
            _sample(**{field: -1})


def test_cache_rejects_float32_scalar_overflow() -> None:
    with pytest.raises(ValueError):
        _sample(ego_interp_weight=1.0e100)


def _sensor_topics():
    cameras = DOMAIN.document["sensors"]["cameras"]
    by_role = {camera["role"]: camera["topic"] for camera in cameras}
    return by_role["front"], by_role["left"], by_role["right"], DOMAIN.document[
        "sensors"
    ]["lidar"]["topic"]


def _jpeg(height, width, bgr):
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = bgr
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    )
    assert ok
    return encoded.tobytes()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _material_plan(
    *, planner_drops=(), anchor_quaternion=(0.0, 0.0, 0.0, 1.0)
):
    front_topic, left_topic, right_topic, lidar_topic = _sensor_topics()
    front = RecordRef(front_topic, 100, 1_000, 1)
    left = RecordRef(left_topic, 110, 1_005, 2)
    right = RecordRef(right_topic, 90, 995, 3)
    lidar = RecordRef(lidar_topic, 777, 1_010, 4)
    anchor = TfPoseRecord(
        2_000,
        2_100,
        10,
        2,
        Pose((0.0, 0.0, 0.0), anchor_quaternion),
    )
    ego_before = EgoRecord(
        1_990,
        2_090,
        8,
        3,
        EgoValue(
            (2.0, -2.0),
            (0.0, -1.0),
            EgoDiagnosticValue((0.0, 0.0), 0.0),
        ),
    )
    ego_after = EgoRecord(
        2_010,
        2_110,
        9,
        3,
        EgoValue(
            (4.0, 0.0),
            (1.0, 0.0),
            EgoDiagnosticValue((0.0, 0.0), 0.0),
        ),
    )
    future = []
    for index in range(8):
        record = TfPoseRecord(
            2_000 + 500_000_000 * (index + 1),
            2_100 + 500_000_000 * (index + 1),
            20 + index,
            2,
            Pose((0.5 * (index + 1), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        )
        future.append(PlannedBracket(record, record, 0.0))
    sample = PlannedSample(
        grid_receipt_ns=1_000,
        front=front,
        left=left,
        right=right,
        lidar=lidar,
        trajectory_anchor_tf=anchor,
        ego_bracket_clock_epoch_index=3,
        ego_bracket=PlannedBracket(ego_before, ego_after, 0.5),
        current_pose_bracket=PlannedBracket(anchor, anchor, 0.0),
        future_pose_brackets=tuple(future),
    )
    return BagSamplePlan(
        bag_path=SYNTHETIC_SOURCE,
        source_bag_sha256=_sha256_file(SYNTHETIC_SOURCE),
        resolved_domain_sha256=DOMAIN.sha256,
        samples=(sample,),
        drops=tuple(planner_drops),
        selected_payload_indices={
            front_topic: frozenset({1}),
            left_topic: frozenset({2}),
            right_topic: frozenset({3}),
            lidar_topic: frozenset({4}),
        },
    )


def _success_events(plan, *, wrong_front_size=False):
    sample = plan.samples[0]
    front_jpeg = _jpeg(720, 1279 if wrong_front_size else 1280, (10, 20, 30))
    left_jpeg = _jpeg(480, 640, (40, 50, 60))
    right_jpeg = _jpeg(480, 640, (70, 80, 90))
    points = np.asarray([[2.0, 2.0, 0.0, 1.0]], dtype=np.float32)
    return [
        SensorPayload(sample.lidar, SensorPayloadKind.LIDAR_XYZI, points),
        SensorPayload(sample.right, SensorPayloadKind.CAMERA_JPEG, right_jpeg),
        SensorPayload(sample.front, SensorPayloadKind.CAMERA_JPEG, front_jpeg),
        SensorPayload(sample.left, SensorPayloadKind.CAMERA_JPEG, left_jpeg),
    ]


def _two_sample_plan():
    plan = _material_plan()
    first = plan.samples[0]
    second_front = RecordRef(first.front.topic, 101, 2_000, 5)
    second = PlannedSample(
        grid_receipt_ns=2_000,
        front=second_front,
        left=first.left,
        right=first.right,
        lidar=first.lidar,
        trajectory_anchor_tf=first.trajectory_anchor_tf,
        ego_bracket_clock_epoch_index=first.ego_bracket_clock_epoch_index,
        ego_bracket=first.ego_bracket,
        current_pose_bracket=first.current_pose_bracket,
        future_pose_brackets=first.future_pose_brackets,
    )
    selected = dict(plan.selected_payload_indices)
    selected[first.front.topic] = frozenset({1, 5})
    return BagSamplePlan(
        plan.bag_path,
        plan.source_bag_sha256,
        plan.resolved_domain_sha256,
        (first, second),
        (),
        selected,
    )


def _two_sample_success_events(plan):
    first_events = _success_events(plan)
    by_key = {
        (event.ref.topic, event.ref.stable_record_index): event
        for event in first_events
    }
    second = plan.samples[1]
    front_two = SensorPayload(
        second.front,
        SensorPayloadKind.CAMERA_JPEG,
        _jpeg(720, 1280, (11, 21, 31)),
    )
    return [
        front_two,
        by_key[(second.right.topic, second.right.stable_record_index)],
        by_key[(second.lidar.topic, second.lidar.stable_record_index)],
        by_key[(plan.samples[0].front.topic, plan.samples[0].front.stable_record_index)],
        by_key[(second.left.topic, second.left.stable_record_index)],
    ]


def test_materializer_selective_pipeline_yields_exact_schema_v2_sample(
    monkeypatch,
) -> None:
    plan = _material_plan(
        planner_drops=(SampleDrop(500, None, SampleDropReason.CAMERA_SYNC_MISS),)
    )
    calls = []

    def fake_events(path, selected):
        calls.append((path, selected))
        assert all(type(indices) is set for indices in selected.values())
        assert selected == {
            topic: set(indices)
            for topic, indices in plan.selected_payload_indices.items()
        }
        return iter(_success_events(plan))

    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads", fake_events
    )
    stream = materialize_samples(
        bag_path=plan.bag_path, plan=plan, domain=DOMAIN
    )
    samples = list(stream)

    assert len(calls) == 1
    assert len(samples) == 1
    sample = samples[0]
    assert sample.timestamp_ns == 100
    assert sample.grid_receipt_ns == 1_000
    assert sample.sensor_source_header_stamp_ns == (100, 110, 90, 777)
    assert sample.sensor_source_receipt_ns == (1_000, 1_005, 995, 1_010)
    assert sample.sensor_source_record_index == (1, 2, 3, 4)
    assert sample.side_camera_header_offset_ns == (10, -10)
    assert sample.sensor_receipt_offset_ns == (5, -5, 10)
    assert sample.trajectory_anchor_tf_header_ns == 2_000
    assert sample.trajectory_anchor_tf_receipt_ns == 2_100
    assert sample.trajectory_anchor_tf_record_index == 10
    assert sample.trajectory_anchor_tf_clock_epoch_index == 2
    assert sample.ego_bracket_clock_epoch_index == 3
    assert sample.ego_bracket_header_stamp_ns == (1_990, 2_010)
    assert sample.ego_bracket_receipt_ns == (2_090, 2_110)
    assert sample.ego_bracket_record_index == (8, 9)
    assert sample.ego_interp_weight == 0.5
    assert sample.current_pose_bracket_header_stamp_ns == (2_000, 2_000)
    assert sample.current_pose_bracket_receipt_ns == (2_100, 2_100)
    assert sample.current_pose_bracket_record_index == (10, 10)
    assert sample.current_pose_interp_weight == 0.0
    expected_future_headers = np.asarray(
        [
            [2_000 + 500_000_000 * index] * 2
            for index in range(1, 9)
        ],
        dtype=np.int64,
    )
    expected_future_receipts = np.asarray(
        [
            [2_100 + 500_000_000 * index] * 2
            for index in range(1, 9)
        ],
        dtype=np.int64,
    )
    expected_future_indices = np.asarray(
        [[index, index] for index in range(20, 28)], dtype=np.int64
    )
    assert np.array_equal(
        sample.future_pose_bracket_header_stamp_ns, expected_future_headers
    )
    assert np.array_equal(
        sample.future_pose_bracket_receipt_ns, expected_future_receipts
    )
    assert np.array_equal(
        sample.future_pose_bracket_record_index, expected_future_indices
    )
    assert np.array_equal(
        sample.future_pose_interp_weight, np.zeros(8, dtype=np.float32)
    )
    assert sample.command is Command.STRAIGHT
    assert sample.status.tolist() == [0.0, 1.0, 0.0, 0.0, 3.0, -1.0, 0.5, -0.5]
    expected_trajectory = np.zeros((8, 3), dtype=np.float32)
    expected_trajectory[:, 0] = np.arange(1, 9, dtype=np.float32) * 0.5
    assert np.array_equal(sample.trajectory, expected_trajectory)
    assert int(sample.lidar_count.sum()) == 1
    assert np.allclose(sample.camera_rgb[100, 10], (60, 50, 40), atol=2)
    assert np.allclose(sample.camera_rgb[100, 500], (30, 20, 10), atol=2)
    assert np.allclose(sample.camera_rgb[100, 900], (90, 80, 70), atol=2)
    assert stream.valid_sample_count == 1
    assert stream.drop_records == (
        SampleDrop(500, None, SampleDropReason.CAMERA_SYNC_MISS),
    )


def test_materializer_failure_drops_sample_and_raises_no_valid_after_exhaustion(
    monkeypatch,
) -> None:
    plan = _material_plan()
    events = _success_events(plan)
    events[2] = SensorPayloadFailure(
        plan.samples[0].front.topic,
        plan.samples[0].front.stable_record_index,
        "DECODE_ERROR",
        "CORRUPT_JPEG",
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(events),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    with pytest.raises(NoValidSampleError) as caught:
        list(stream)
    assert isinstance(caught.value, MoraiDDV2Error)
    assert caught.value.rejection_code.value == "NO_VALID_SAMPLE"
    assert stream.valid_sample_count == 0
    assert stream.drop_records == (
        SampleDrop(1_000, 100, SampleDropReason.DECODE_ERROR),
    )


def test_final_drop_stats_merge_planner_and_materializer_once_in_order(
    monkeypatch,
) -> None:
    planner_drop = SampleDrop(500, None, SampleDropReason.CAMERA_SYNC_MISS)
    plan = _material_plan(planner_drops=(planner_drop,))
    events = _success_events(plan)
    front = plan.samples[0].front
    events[2] = SensorPayloadFailure(
        front.topic,
        front.stable_record_index,
        "DECODE_ERROR",
        "CORRUPT_JPEG",
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(events),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)

    with pytest.raises(NoValidSampleError):
        list(stream)
    assert stream.valid_sample_count == 0
    assert stream.drop_records == (
        planner_drop,
        SampleDrop(1_000, 100, SampleDropReason.DECODE_ERROR),
    )


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "duplicate",
        "unrequested",
        "altered_ref",
        "wrong_kind",
        "wrong_failure_reason",
    ],
)
def test_materializer_rejects_structural_event_integrity(monkeypatch, mode) -> None:
    plan = _material_plan()
    events = _success_events(plan)
    if mode == "missing":
        events.pop()
    elif mode == "duplicate":
        events.append(events[0])
    else:
        if mode == "unrequested":
            events.append(
                SensorPayload(
                    RecordRef(plan.samples[0].front.topic, 100, 1_000, 999),
                    SensorPayloadKind.CAMERA_JPEG,
                    b"unused",
                )
            )
        elif mode == "altered_ref":
            original = events[2]
            events[2] = SensorPayload(
                RecordRef(
                    original.ref.topic,
                    original.ref.header_stamp_ns + 1,
                    original.ref.receipt_stamp_ns + 1,
                    original.ref.stable_record_index,
                ),
                original.kind,
                original.value,
            )
        elif mode == "wrong_kind":
            original = events[2]
            events[2] = SensorPayload(
                original.ref,
                SensorPayloadKind.LIDAR_XYZI,
                np.zeros((1, 4), dtype=np.float32),
            )
        else:
            original = plan.samples[0].front
            failure = SensorPayloadFailure(
                original.topic,
                original.stable_record_index,
                "DECODE_ERROR",
                "CORRUPT_JPEG",
            )
            object.__setattr__(failure, "reason", "OTHER")
            events[2] = failure
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(events),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    with pytest.raises(IntegrityError):
        list(stream)
    assert all(drop.reason is not SampleDropReason.DECODE_ERROR for drop in stream.drop_records)


def test_materializer_reader_exception_aborts_and_closes_without_sample_drop(
    monkeypatch,
) -> None:
    plan = _material_plan()
    closed = []

    def failing_events(path, selected):
        try:
            yield _success_events(plan)[0]
            raise OSError("synthetic bag reader failure")
        finally:
            closed.append(True)

    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        failing_events,
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)

    with pytest.raises(OSError, match="synthetic bag reader failure"):
        list(stream)
    assert closed == [True]
    assert stream.valid_sample_count == 0
    assert stream.drop_records == ()
    with pytest.raises(StopIteration):
        next(stream)


def test_materializer_resolution_mismatch_is_bag_level_and_closes_reader(
    monkeypatch,
) -> None:
    plan = _material_plan()
    closed = []

    def fake_events(path, selected):
        try:
            yield from _success_events(plan, wrong_front_size=True)
        finally:
            closed.append(True)

    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        fake_events,
    )
    with pytest.raises(CameraResolutionError):
        list(materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN))
    assert closed == [True]


def test_materializer_accepts_exact_current_pose_after_quaternion_renormalization(
    monkeypatch,
) -> None:
    plan = _material_plan(anchor_quaternion=(0.0, 0.0, 0.0, 1.0 + 1.0e-15))
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(_success_events(plan)),
    )
    samples = list(
        materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    )
    assert len(samples) == 1


def test_materializer_zero_plan_still_calls_selected_iterator_then_no_valid(
    monkeypatch,
) -> None:
    full = _material_plan()
    empty = BagSamplePlan(
        full.bag_path,
        full.source_bag_sha256,
        full.resolved_domain_sha256,
        (),
        (),
        {topic: frozenset() for topic in full.selected_payload_indices},
    )
    calls = []

    def fake_events(path, selected):
        calls.append((path, selected))
        return iter(())

    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads", fake_events
    )
    stream = materialize_samples(bag_path=empty.bag_path, plan=empty, domain=DOMAIN)
    with pytest.raises(NoValidSampleError):
        next(stream)
    assert len(calls) == 1
    with pytest.raises(StopIteration):
        next(stream)


def test_materializer_rejects_plan_binding_mismatch() -> None:
    plan = _material_plan()
    with pytest.raises(IntegrityError):
        materialize_samples(
            bag_path=Path("/tmp/other.bag"), plan=plan, domain=DOMAIN
        )


def test_materializer_rejects_source_sha_mismatch_before_pass2(monkeypatch) -> None:
    plan = _material_plan()
    opened = []
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: opened.append(True),
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.sha256_file",
        lambda path: "b" * 64,
        raising=False,
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    with pytest.raises(IntegrityError):
        next(stream)
    assert opened == []


def test_materializer_rechecks_source_sha_after_lazy_stream_construction(
    monkeypatch,
) -> None:
    plan = _material_plan()
    current_sha = [plan.source_bag_sha256]
    opened = []
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.sha256_file",
        lambda path: current_sha[0],
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: opened.append(True),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    current_sha[0] = "b" * 64
    with pytest.raises(IntegrityError):
        next(stream)
    assert opened == []


def test_materializer_rechecks_source_sha_after_full_pass2(
    monkeypatch,
) -> None:
    plan = _material_plan()
    hashes = iter((plan.source_bag_sha256, "b" * 64))
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.sha256_file",
        lambda path: next(hashes),
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(_success_events(plan)),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    with pytest.raises(IntegrityError):
        list(stream)
    assert stream.valid_sample_count == 1


def test_materializer_rejects_swapped_physical_camera_roles_before_pass2(
    monkeypatch,
) -> None:
    plan = _material_plan()
    original = plan.samples[0]
    swapped = PlannedSample(
        grid_receipt_ns=original.grid_receipt_ns,
        front=original.front,
        left=original.right,
        right=original.left,
        lidar=original.lidar,
        trajectory_anchor_tf=original.trajectory_anchor_tf,
        ego_bracket_clock_epoch_index=original.ego_bracket_clock_epoch_index,
        ego_bracket=original.ego_bracket,
        current_pose_bracket=original.current_pose_bracket,
        future_pose_brackets=original.future_pose_brackets,
    )
    malformed = BagSamplePlan(
        plan.bag_path,
        plan.source_bag_sha256,
        plan.resolved_domain_sha256,
        (swapped,),
        (),
        plan.selected_payload_indices,
    )
    opened = []
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: opened.append(True),
    )
    with pytest.raises(IntegrityError):
        materialize_samples(
            bag_path=malformed.bag_path,
            plan=malformed,
            domain=DOMAIN,
        )
    assert opened == []


def test_materialization_stream_close_releases_selected_event_iterator(
    monkeypatch,
) -> None:
    plan = _material_plan()
    closed = []

    def fake_events(path, selected):
        try:
            yield from _success_events(plan)
        finally:
            closed.append(True)

    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads", fake_events
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    assert next(stream).timestamp_ns == 100
    assert closed == []
    stream.close()
    assert closed == [True]
    with pytest.raises(StopIteration):
        next(stream)


def test_materializer_shared_payloads_decode_once_and_yield_chronologically(
    monkeypatch,
) -> None:
    plan = _two_sample_plan()
    events = _two_sample_success_events(plan)
    import morai_ddv2.preprocess.materialize as materialize_module

    original_camera = materialize_module.decode_jpeg_rgb
    original_lidar = materialize_module.build_lidar_count
    camera_calls = []
    lidar_calls = []

    def counted_camera(payload, expected_wh):
        camera_calls.append((id(payload), expected_wh))
        return original_camera(payload, expected_wh)

    def counted_lidar(points, pose, config):
        lidar_calls.append(id(points))
        return original_lidar(points, pose, config)

    monkeypatch.setattr(materialize_module, "decode_jpeg_rgb", counted_camera)
    monkeypatch.setattr(materialize_module, "build_lidar_count", counted_lidar)
    monkeypatch.setattr(
        materialize_module,
        "iter_selected_payloads",
        lambda path, selected: iter(events),
    )

    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    samples = list(stream)
    assert [sample.timestamp_ns for sample in samples] == [100, 101]
    for ref in (
        plan.samples[0].left,
        plan.samples[0].right,
        plan.samples[0].lidar,
    ):
        assert plan.selected_payload_indices[ref.topic] == frozenset(
            {ref.stable_record_index}
        )
    assert len(camera_calls) == 4
    assert len(lidar_calls) == 1
    assert stream.valid_sample_count == 2


def test_materializer_shared_inputs_allow_neighbor_to_survive_one_front_failure(
    monkeypatch,
) -> None:
    plan = _two_sample_plan()
    events = _two_sample_success_events(plan)
    first = plan.samples[0]
    events[3] = SensorPayloadFailure(
        first.front.topic,
        first.front.stable_record_index,
        "DECODE_ERROR",
        "CORRUPT_JPEG",
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(events),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    samples = list(stream)
    assert [sample.timestamp_ns for sample in samples] == [101]
    assert stream.valid_sample_count == 1
    assert stream.drop_records == (
        SampleDrop(1_000, 100, SampleDropReason.DECODE_ERROR),
    )


def test_materializer_shared_failure_drops_every_consumer_once(monkeypatch) -> None:
    plan = _two_sample_plan()
    events = _two_sample_success_events(plan)
    left = plan.samples[0].left
    events[4] = SensorPayloadFailure(
        left.topic,
        left.stable_record_index,
        "DECODE_ERROR",
        "CORRUPT_JPEG",
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(events),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    with pytest.raises(NoValidSampleError):
        list(stream)
    assert stream.drop_records == (
        SampleDrop(1_000, 100, SampleDropReason.DECODE_ERROR),
        SampleDrop(2_000, 101, SampleDropReason.DECODE_ERROR),
    )


def test_materializer_maps_corrupt_raw_jpeg_to_decode_error(monkeypatch) -> None:
    plan = _material_plan()
    events = _success_events(plan)
    events[2] = SensorPayload(
        plan.samples[0].front,
        SensorPayloadKind.CAMERA_JPEG,
        b"not-a-jpeg",
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(events),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    with pytest.raises(NoValidSampleError):
        list(stream)
    assert stream.drop_records == (
        SampleDrop(1_000, 100, SampleDropReason.DECODE_ERROR),
    )


@pytest.mark.parametrize(
    ("points", "reason"),
    [
        (np.zeros((1, 4), dtype=np.float32), SampleDropReason.EMPTY_LIDAR),
        (
            np.asarray([[np.nan, 0.0, 0.0, 0.0]], dtype=np.float32),
            SampleDropReason.NONFINITE_VALUE,
        ),
    ],
)
def test_materializer_maps_lidar_data_failures(monkeypatch, points, reason) -> None:
    plan = _material_plan()
    events = _success_events(plan)
    events[0] = SensorPayload(
        plan.samples[0].lidar,
        SensorPayloadKind.LIDAR_XYZI,
        points,
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(events),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    with pytest.raises(NoValidSampleError):
        list(stream)
    assert stream.drop_records == (SampleDrop(1_000, 100, reason),)


@pytest.mark.parametrize(
    "corruption",
    [
        "ego_velocity_nan",
        "ego_acceleration_inf",
        "anchor_pose_nan",
        "future_pose_inf",
    ],
)
def test_materializer_maps_nonfinite_ego_or_pose_data_to_one_drop(
    monkeypatch, corruption
) -> None:
    plan = _material_plan()
    sample = plan.samples[0]
    if corruption == "ego_velocity_nan":
        object.__setattr__(
            sample.ego_bracket.before.value,
            "velocity_xy",
            (float("nan"), -2.0),
        )
    elif corruption == "ego_acceleration_inf":
        object.__setattr__(
            sample.ego_bracket.after.value,
            "acceleration_xy",
            (float("inf"), 0.0),
        )
    elif corruption == "anchor_pose_nan":
        object.__setattr__(
            sample.trajectory_anchor_tf.value,
            "translation_xyz",
            (float("nan"), 0.0, 0.0),
        )
    else:
        object.__setattr__(
            sample.future_pose_brackets[0].before.value,
            "translation_xyz",
            (float("inf"), 0.0, 0.0),
        )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(_success_events(plan)),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)

    with pytest.raises(NoValidSampleError):
        list(stream)
    assert stream.valid_sample_count == 0
    assert stream.drop_records == (
        SampleDrop(1_000, 100, SampleDropReason.NONFINITE_VALUE),
    )


def test_materializer_maps_invalid_oracle_command(monkeypatch) -> None:
    plan = _material_plan()
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads",
        lambda path, selected: iter(_success_events(plan)),
    )
    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.classify_oracle_command",
        lambda trajectory, rules: (_ for _ in ()).throw(InvalidCommand("invalid")),
    )
    stream = materialize_samples(bag_path=plan.bag_path, plan=plan, domain=DOMAIN)
    with pytest.raises(NoValidSampleError):
        list(stream)
    assert stream.drop_records == (
        SampleDrop(1_000, 100, SampleDropReason.INVALID_COMMAND),
    )


def test_real_anomalous_bag_plans_and_materializes_read_only() -> None:
    raw_path = os.environ.get("MORAI_REAL_BAG")
    if raw_path is None:
        pytest.skip("MORAI_REAL_BAG is unset")
    path = Path(raw_path)
    assert path.name == (
        "s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__"
        "scenario-000__20260731T170301KST.bag"
    )
    source_before = (
        path.stat().st_size,
        path.stat().st_mtime_ns,
        _sha256_file(path),
    )

    bag = index_bag(path, DOMAIN)
    assert bag.source_bag_sha256 == source_before[2]
    plan = build_sample_plan(bag, DOMAIN)
    stream = materialize_samples(bag_path=path, plan=plan, domain=DOMAIN)
    consumed = sum(1 for _ in stream)

    assert consumed == stream.valid_sample_count
    assert consumed >= 1
    assert (
        path.stat().st_size,
        path.stat().st_mtime_ns,
        _sha256_file(path),
    ) == source_before
