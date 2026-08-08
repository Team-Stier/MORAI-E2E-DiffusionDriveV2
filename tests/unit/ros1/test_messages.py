from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import morai_ddv2.ros1 as ros1
from morai_ddv2.preprocess.diagnostics import EgoDiagnosticValue, TfEgoDiagnostic
from morai_ddv2.preprocess.geometry import Pose
from morai_ddv2.ros1.messages import (
    BagRejectReason,
    BagDisposition,
    BagIndex,
    BagInventory,
    ClockEpochView,
    EgoRecord,
    EgoValue,
    EpochSafeTfEgoDiagnostic,
    PointCloudDecodeError,
    RecordRef,
    SensorPayload,
    SensorPayloadFailure,
    SensorPayloadKind,
    TfPoseRecord,
    TfEgoGateProvenance,
    ValueRecord,
    ego_value_from_message,
    normalize_frame_id,
    pointcloud_layout_from_message,
    pointcloud_xyz_i,
    stamp_to_ns,
)


def _ns(**values):
    return SimpleNamespace(**values)


def _pose(x: float = 0.0) -> Pose:
    return Pose((x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _ego_record(index: int, epoch: int = 0) -> EgoRecord:
    value = EgoValue(
        velocity_xy=(1.0, 2.0),
        acceleration_xy=(3.0, 4.0),
        diagnostic=EgoDiagnosticValue((5.0, 6.0), 7.0),
    )
    return EgoRecord(100 + index, 900 + index, index, epoch, value)


def _tf_record(index: int, epoch: int = 0) -> TfPoseRecord:
    return TfPoseRecord(100 + index, 900 + index, index, epoch, _pose(float(index)))


def test_ros1_imports_do_not_import_ros_packages() -> None:
    project_root = Path(__file__).resolve().parents[3]
    code = r'''
import builtins, json, sys
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
import morai_ddv2.ros1
import morai_ddv2.ros1.messages
print(json.dumps(seen))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_front_timestamp_rejection_is_distinct_from_no_valid_sample() -> None:
    assert BagRejectReason.FRONT_TIMESTAMP_ORDER_VIOLATION.value == (
        "FRONT_TIMESTAMP_ORDER_VIOLATION"
    )


def test_package_root_exports_the_complete_public_task8_api() -> None:
    expected = {
        "BagDisposition",
        "BagIndex",
        "BagInventory",
        "BagReadError",
        "BagRejectReason",
        "ClockEpochView",
        "EgoRecord",
        "EgoValue",
        "EpochSafeTfEgoDiagnostic",
        "Extrinsic",
        "PayloadEvent",
        "RecordRef",
        "SensorPayload",
        "SensorPayloadFailure",
        "SensorPayloadKind",
        "TfEgoGateProvenance",
        "TfPoseRecord",
        "ValueRecord",
        "causal_tf_interpolation_view",
        "compute_epoch_safe_tf_ego_diagnostic",
        "discover_bags",
        "index_bag",
        "iter_selected_payloads",
        "stamp_to_ns",
    }

    assert set(ros1.__all__) == expected
    assert all(hasattr(ros1, name) for name in expected)
    assert BagRejectReason.FRONT_TIMESTAMP_ORDER_VIOLATION is not (
        BagRejectReason.NO_VALID_SAMPLE
    )


@pytest.mark.parametrize(
    "secs,nsecs,expected",
    [
        (0, 0, 0),
        (0, 999_999_999, 999_999_999),
        (9_007_200, 123_456_789, 9_007_200_123_456_789),
        (-1, 500_000_000, -500_000_000),
        (9_223_372_036, 854_775_807, 9_223_372_036_854_775_807),
        (-9_223_372_037, 145_224_192, -9_223_372_036_854_775_808),
    ],
)
def test_stamp_to_ns_preserves_exact_signed_int64(secs, nsecs, expected) -> None:
    stamp = _ns(secs=secs, nsecs=nsecs, to_sec=lambda: 0.0)

    assert stamp_to_ns(stamp) == expected


@pytest.mark.parametrize(
    "stamp",
    [
        _ns(nsecs=0),
        _ns(secs=0),
        _ns(secs=True, nsecs=0),
        _ns(secs=0, nsecs=False),
        _ns(secs=1.0, nsecs=0),
        _ns(secs=0, nsecs=1.0),
        _ns(secs=0, nsecs=-1),
        _ns(secs=0, nsecs=1_000_000_000),
        _ns(secs=9_223_372_037, nsecs=0),
        _ns(secs=-9_223_372_037, nsecs=145_224_191),
    ],
)
def test_stamp_to_ns_rejects_malformed_and_overflowing_stamps(stamp) -> None:
    with pytest.raises((TypeError, ValueError, OverflowError)):
        stamp_to_ns(stamp)


def test_flat_record_models_keep_both_clocks_and_are_frozen() -> None:
    ref = RecordRef("/front", 9_007_200_123_456_789, 7, 3)
    value = ValueRecord(11, 22, 4, "small")
    ego = _ego_record(5, epoch=2)
    tf = _tf_record(6, epoch=2)

    assert ref.header_stamp_ns != ref.receipt_stamp_ns
    assert not hasattr(ref, "stamp_ns")
    assert not hasattr(ego, "ref") and not hasattr(ego, "record")
    assert not hasattr(tf, "ref") and not hasattr(tf, "record")
    assert value.value == "small"
    with pytest.raises(FrozenInstanceError):
        ref.stable_record_index = 4


@pytest.mark.parametrize(
    "factory,args",
    [
        (RecordRef, ("", 1, 2, 0)),
        (RecordRef, ("/x", 1, 2, -1)),
        (RecordRef, ("/x", 1, 2, True)),
        (ValueRecord, (1, 2, -1, "x")),
        (EgoRecord, (1, 2, 0, -1, _ego_record(0).value)),
        (TfPoseRecord, (1, 2, 0, -1, _pose())),
        (TfPoseRecord, (1, 2, 0, 0, "not-a-pose")),
    ],
)
def test_flat_record_models_reject_invalid_identity_and_values(factory, args) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory(*args)


def test_clock_epoch_view_is_identity_preserving_ordered_subset() -> None:
    first = _tf_record(2, epoch=3)
    duplicate = _tf_record(4, epoch=3)
    last = _tf_record(7, epoch=3)
    view = ClockEpochView(3, (first, duplicate, last), (first, last))

    assert view.interpolation_records[0] is view.raw_records[0]
    assert view.interpolation_records[1] is view.raw_records[2]
    with pytest.raises(FrozenInstanceError):
        view.index = 4
    with pytest.raises((TypeError, ValueError)):
        ClockEpochView(3, (first, last), (last, first))
    with pytest.raises((TypeError, ValueError)):
        ClockEpochView(3, (), ())
    with pytest.raises((TypeError, ValueError)):
        ClockEpochView(2, (first,), (first,))


def test_clock_epoch_view_rejects_wrong_receipt_order_duplicate_interpolation_and_copy() -> None:
    first = TfPoseRecord(100, 10, 0, 0, _pose())
    duplicate = TfPoseRecord(100, 20, 1, 0, _pose())
    later = TfPoseRecord(200, 30, 2, 0, _pose())
    copied_first = TfPoseRecord(100, 10, 0, 0, _pose())

    with pytest.raises(ValueError, match="receipt"):
        ClockEpochView(0, (later, first), (later,))
    with pytest.raises(ValueError, match="strictly increasing"):
        ClockEpochView(0, (first, duplicate, later), (first, duplicate, later))
    with pytest.raises(ValueError, match="identity"):
        ClockEpochView(0, (first, later), (copied_first, later))


def test_bag_index_requires_flat_raw_records_to_be_epoch_view_identities() -> None:
    record = _tf_record(0)
    copied = _tf_record(0)
    view = ClockEpochView(0, (record,), (record,))
    inventory = BagInventory(
        BagDisposition.ELIGIBLE,
        None,
        (),
        (),
        (),
        None,
    )
    kwargs = dict(
        path=Path("synthetic.bag"),
        source_bag_sha256="0" * 64,
        inventory=inventory,
        front=(),
        left=(),
        right=(),
        lidar=(),
        raw_ego=(),
        ego_clock_epochs=(),
        raw_poses=(record,),
        tf_clock_epochs=(view,),
        extrinsics=(),
    )

    valid = BagIndex(**kwargs)

    assert valid.raw_poses[0] is valid.tf_clock_epochs[0].raw_records[0]
    assert not hasattr(valid, "poses")
    assert not hasattr(valid, "ego")
    with pytest.raises(ValueError, match="identity"):
        BagIndex(**dict(kwargs, raw_poses=(copied,)))
    with pytest.raises(ValueError, match="epoch views"):
        BagIndex(**dict(kwargs, tf_clock_epochs=()))


def test_epoch_safe_report_counters_must_match_each_provenance_reason() -> None:
    comparable = TfEgoGateProvenance(
        ego_header_stamp_ns=10,
        ego_receipt_stamp_ns=20,
        ego_stable_record_index=0,
        selected_tf_clock_epoch_index=0,
        selected_tf_anchor_record_index=3,
        tf_bracket_header_stamp_ns=(0, 20),
        tf_bracket_receipt_ns=(100, 120),
        tf_bracket_record_index=(2, 4),
        exclusion_reason=None,
    )
    excluded = TfEgoGateProvenance(
        ego_header_stamp_ns=30,
        ego_receipt_stamp_ns=40,
        ego_stable_record_index=1,
        selected_tf_clock_epoch_index=1,
        selected_tf_anchor_record_index=8,
        tf_bracket_header_stamp_ns=None,
        tf_bracket_receipt_ns=None,
        tf_bracket_record_index=None,
        exclusion_reason="NO_EPOCH_LOCAL_BRACKET",
    )
    metric = TfEgoDiagnostic(1, 1, 0.0, 0.0, 0.0, 0.0)

    valid = EpochSafeTfEgoDiagnostic(
        metric,
        (comparable, excluded),
        2,
        1,
        1,
        0,
        0,
    )

    assert valid.no_epoch_local_bracket_count == 1
    with pytest.raises(ValueError, match="provenance"):
        EpochSafeTfEgoDiagnostic(
            metric,
            (comparable, excluded),
            2,
            1,
            0,
            0,
            1,
        )


def test_frame_normalization_and_ego_field_extraction_are_ros_free() -> None:
    message = _ns(
        velocity=_ns(x=1.25, y=-2.5),
        acceleration=_ns(x=3.5, y=-4.5),
        position=_ns(x=10.0, y=20.0),
        heading=-179.5,
    )

    assert normalize_frame_id("///map") == "map"
    assert normalize_frame_id("/base_link") == "base_link"
    assert ego_value_from_message(message) == EgoValue(
        velocity_xy=(1.25, -2.5),
        acceleration_xy=(3.5, -4.5),
        diagnostic=EgoDiagnosticValue((10.0, 20.0), -179.5),
    )
    with pytest.raises(ValueError):
        normalize_frame_id("///")


@pytest.mark.parametrize(
    "message",
    [
        _ns(
            velocity=_ns(x=math.nan, y=0.0),
            acceleration=_ns(x=0.0, y=0.0),
            position=_ns(x=0.0, y=0.0),
            heading=0.0,
        ),
        _ns(
            velocity=_ns(x=0.0),
            acceleration=_ns(x=0.0, y=0.0),
            position=_ns(x=0.0, y=0.0),
            heading=0.0,
        ),
    ],
)
def test_ego_field_extraction_rejects_nonfinite_or_missing_values(message) -> None:
    with pytest.raises((TypeError, ValueError)):
        ego_value_from_message(message)


def test_pointcloud_layout_conversion_does_not_touch_data() -> None:
    class PoisonCloud:
        fields = (
            _ns(name="x", offset=8, datatype=7, count=1),
            _ns(name="y", offset=0, datatype=7, count=1),
            _ns(name="z", offset=12, datatype=7, count=1),
            _ns(name="intensity", offset=4, datatype=7, count=1),
        )
        point_step = 16
        is_bigendian = False

        @property
        def data(self):
            raise AssertionError("layout inspection touched PointCloud payload")

    layout = pointcloud_layout_from_message(PoisonCloud())

    assert tuple(field.name for field in layout.fields) == (
        "x",
        "y",
        "z",
        "intensity",
    )


def test_pointcloud_decode_uses_canonical_order_and_preserves_nan(monkeypatch) -> None:
    calls = []

    def read_points(message, field_names, skip_nans):
        calls.append((message, field_names, skip_nans))
        return iter(((1.0, 2.0, 3.0, 4.0), (math.nan, 6.0, 7.0, 8.0)))

    fake_module = _ns(read_points=read_points)
    monkeypatch.setitem(sys.modules, "sensor_msgs", _ns(point_cloud2=fake_module))
    monkeypatch.setitem(sys.modules, "sensor_msgs.point_cloud2", fake_module)
    message = object()

    actual = pointcloud_xyz_i(message)

    assert calls == [(message, ("x", "y", "z", "intensity"), False)]
    assert actual.dtype == np.float32
    assert actual.shape == (2, 4)
    assert actual.flags.c_contiguous
    assert math.isnan(float(actual[1, 0]))


@pytest.mark.parametrize(
    "rows,shape",
    [
        ((), (0, 4)),
        (((math.inf, 2.0, 3.0, 4.0),), (1, 4)),
    ],
)
def test_pointcloud_decode_handles_empty_and_preserves_infinity(monkeypatch, rows, shape) -> None:
    fake_module = _ns(
        read_points=lambda message, field_names, skip_nans: iter(rows)
    )
    monkeypatch.setitem(sys.modules, "sensor_msgs", _ns(point_cloud2=fake_module))
    monkeypatch.setitem(sys.modules, "sensor_msgs.point_cloud2", fake_module)

    actual = pointcloud_xyz_i(object())

    assert actual.shape == shape
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    if rows:
        assert math.isinf(float(actual[0, 0]))


def test_pointcloud_decode_wraps_ros_or_row_shape_failures(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("backend detail /tmp/private 0x123")

    fake_module = _ns(read_points=fail)
    monkeypatch.setitem(sys.modules, "sensor_msgs", _ns(point_cloud2=fake_module))
    monkeypatch.setitem(sys.modules, "sensor_msgs.point_cloud2", fake_module)
    with pytest.raises(PointCloudDecodeError, match="unable to decode"):
        pointcloud_xyz_i(object())

    fake_module.read_points = lambda *args, **kwargs: iter(((1.0, 2.0, 3.0),))
    with pytest.raises(PointCloudDecodeError, match="unable to decode"):
        pointcloud_xyz_i(object())


def test_payload_events_are_typed_frozen_and_failure_detail_is_sanitized() -> None:
    ref = RecordRef("/front", 1, 2, 3)
    camera = SensorPayload(ref, SensorPayloadKind.CAMERA_JPEG, b"jpeg")
    lidar_array = np.zeros((1, 4), dtype=np.float32)
    lidar = SensorPayload(ref, SensorPayloadKind.LIDAR_XYZI, lidar_array)
    failure = SensorPayloadFailure("/front", 3, "DECODE_ERROR", "DATA_ACCESS_FAILED")

    assert camera.value == b"jpeg"
    assert lidar.value.flags.writeable is False
    assert failure.reason == "DECODE_ERROR"
    with pytest.raises(FrozenInstanceError):
        failure.detail = "OTHER"
    for unsafe in ("ValueError('bad')", "object at 0x1234", "/tmp/input", "bad detail"):
        with pytest.raises(ValueError):
            SensorPayloadFailure("/front", 3, "DECODE_ERROR", unsafe)


@pytest.mark.parametrize(
    "kind,value",
    [
        (SensorPayloadKind.CAMERA_JPEG, bytearray(b"mutable")),
        (SensorPayloadKind.CAMERA_JPEG, np.zeros((1, 4), dtype=np.float32)),
        (SensorPayloadKind.LIDAR_XYZI, b"bytes"),
        (SensorPayloadKind.LIDAR_XYZI, np.zeros((4,), dtype=np.float32)),
        (SensorPayloadKind.LIDAR_XYZI, np.zeros((1, 4), dtype=np.float64)),
    ],
)
def test_payload_rejects_kind_value_contract_mismatches(kind, value) -> None:
    with pytest.raises((TypeError, ValueError)):
        SensorPayload(RecordRef("/sensor", 1, 2, 0), kind, value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topic": "", "stable_record_index": 0, "reason": "DECODE_ERROR", "detail": "FAILED"},
        {"topic": "/x", "stable_record_index": -1, "reason": "DECODE_ERROR", "detail": "FAILED"},
        {"topic": "/x", "stable_record_index": 0, "reason": "OTHER", "detail": "FAILED"},
        {"topic": "/x", "stable_record_index": 0, "reason": "DECODE_ERROR", "detail": ""},
    ],
)
def test_payload_failure_rejects_invalid_identity_discriminant_or_detail(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        SensorPayloadFailure(**kwargs)
