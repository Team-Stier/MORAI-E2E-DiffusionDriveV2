from __future__ import annotations

import hashlib
import json
import math
import gc
import os
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError, IntegrityError
from morai_ddv2.preprocess.diagnostics import EgoDiagnosticValue
from morai_ddv2.preprocess.geometry import Pose
from morai_ddv2.ros1 import bag_reader
from morai_ddv2.ros1.bag_reader import (
    BagReadError,
    causal_tf_interpolation_view,
    compute_epoch_safe_tf_ego_diagnostic,
    discover_bags,
    index_bag,
    iter_selected_payloads,
)
from morai_ddv2.ros1.messages import (
    BagDisposition,
    BagIndex,
    BagInventory,
    BagRejectReason,
    ClockEpochView,
    EgoRecord,
    EgoValue,
    RecordRef,
    SensorPayload,
    SensorPayloadFailure,
    SensorPayloadKind,
    TfPoseRecord,
)


GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "golden" / "resolved-domain.json"
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _ns(**values):
    return SimpleNamespace(**values)


def _stamp(value: int):
    return _ns(secs=value // 1_000_000_000, nsecs=value % 1_000_000_000)


def _header(stamp_ns: int, frame_id: str):
    return _ns(stamp=_stamp(stamp_ns), frame_id=frame_id)


def _vec(values):
    return _ns(x=values[0], y=values[1], z=values[2])


def _quaternion(values):
    return _ns(x=values[0], y=values[1], z=values[2], w=values[3])


def _transform(parent, child, header_ns, translation, quaternion=IDENTITY):
    return _ns(
        header=_header(header_ns, parent),
        child_frame_id=child,
        transform=_ns(
            translation=_vec(translation),
            rotation=_quaternion(quaternion),
        ),
    )


def _camera(header_ns: int, frame_id: str, payload: bytes):
    return _ns(
        _type="sensor_msgs/CompressedImage",
        header=_header(header_ns, frame_id),
        data=payload,
    )


def _lidar(header_ns: int, frame_id: str):
    return _ns(
        _type="sensor_msgs/PointCloud2",
        header=_header(header_ns, frame_id),
        fields=tuple(
            _ns(name=name, offset=offset, datatype=7, count=1)
            for name, offset in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))
        ),
        point_step=16,
        is_bigendian=False,
        data=b"not-retained",
    )


def _ego(header_ns: int, x: float):
    return _ns(
        _type="morai_msgs/EgoVehicleStatus",
        header=_header(header_ns, "MoraiInfo"),
        velocity=_vec((1.0, 2.0, 0.0)),
        acceleration=_vec((3.0, 4.0, 0.0)),
        position=_vec((x, 0.0, 0.0)),
        heading=0.0,
    )


class FakeBag:
    def __init__(self, records, topic_types):
        self.records = list(records)
        self.topic_types = tuple(topic_types)
        self.closed = False
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True

    def _get_connections(self):
        return tuple(_ns(topic=topic, datatype=datatype) for topic, datatype in self.topic_types)

    def read_messages(self, topics=None):
        self.read_calls += 1
        selected = None if topics is None else set(topics)
        for topic, message, receipt_ns in self.records:
            if selected is None or topic in selected:
                yield topic, message, _stamp(receipt_ns)


@pytest.fixture(scope="module")
def resolved_domain() -> ResolvedDomain:
    publication = json.loads(GOLDEN.read_text())
    digest = publication.pop("resolved_domain_sha256")
    return ResolvedDomain(publication, digest)


@pytest.fixture(scope="module")
def jpeg_by_role(resolved_domain):
    result = {}
    for camera in resolved_domain.document["sensors"]["cameras"]:
        width, height = camera["resolution_wh"]
        success, encoded = cv2.imencode(
            ".jpg", np.zeros((height, width, 3), dtype=np.uint8)
        )
        assert success
        result[camera["role"]] = bytes(encoded)
    return result


def _types(domain, *, include_tf=True, include_ego=True):
    contract = domain.document["ros_contract"]
    types = contract["message_types"]
    pairs = [
        (camera["topic"], types["camera"])
        for camera in domain.document["sensors"]["cameras"]
    ]
    pairs.append((domain.document["sensors"]["lidar"]["topic"], types["lidar"]))
    if include_tf:
        pairs.append((contract["policy_topics"]["transforms"], types["transforms"]))
    if include_ego:
        pairs.append((contract["policy_topics"]["ego_state"], types["ego_state"]))
    return pairs


def _eligible_records(domain, jpeg_by_role):
    sensors = domain.document["sensors"]
    contract = domain.document["ros_contract"]
    base_receipt = 9_000_000_000_000
    records = []
    for offset, camera in enumerate(sensors["cameras"]):
        records.append(
            (
                camera["topic"],
                _camera(1_000_000_000 + offset, camera["frame_id"], jpeg_by_role[camera["role"]]),
                base_receipt + offset,
            )
        )
    lidar = sensors["lidar"]
    records.append((lidar["topic"], _lidar(2_000_000_000, lidar["frame_id"]), base_receipt + 4))

    tf_topic = contract["policy_topics"]["transforms"]
    ego_topic = contract["policy_topics"]["ego_state"]
    for index in range(10):
        header_ns = 10_000_000_000 + index * 20_000_000
        receipt_ns = base_receipt + 100 + index * 20_000_000
        transforms = [_transform("/map", "/base_link", header_ns, (float(index), 0.0, 0.0))]
        for camera in sensors["cameras"]:
            transforms.append(
                _transform(
                    "/base_link",
                    "/" + camera["frame_id"],
                    header_ns,
                    tuple(camera["translation_base_from_camera_m"]),
                    tuple(camera["quaternion_base_from_camera_xyzw"]),
                )
            )
        transforms.append(
            _transform(
                "/base_link",
                "/" + lidar["frame_id"],
                header_ns,
                tuple(lidar["translation_base_from_lidar_m"]),
                tuple(lidar["quaternion_base_from_lidar_xyzw"]),
            )
        )
        records.append((tf_topic, _ns(_type="tf2_msgs/TFMessage", transforms=transforms), receipt_ns))
        records.append((ego_topic, _ego(header_ns, float(index)), receipt_ns + 1))
    return records


def _install_fake(monkeypatch, fake):
    monkeypatch.setattr(bag_reader, "_open_bag", lambda path: fake)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["sensors"]["cameras"][0].__setitem__("frame_id", "///"),
        lambda document: document["sensors"]["cameras"][0].__setitem__("resolution_wh", [0, 720]),
        lambda document: document["diagnostics"].__setitem__("tf_ego_minimum_comparable_pairs", 0),
        lambda document: document["ros_contract"].__setitem__(
            "ego_feature_fields", ["velocity.x", "acceleration.x"]
        ),
        lambda document: document["ros_contract"].__setitem__(
            "ego_diagnostic_fields", ["position.x", "heading_deg"]
        ),
        lambda document: document["ros_contract"]["lidar_point_fields"].__setitem__(
            "intensity", "float64"
        ),
    ],
)
def test_malformed_resolved_frame_resolution_or_minimum_is_configuration_error(
    tmp_path, mutate
) -> None:
    publication = json.loads(GOLDEN.read_text())
    digest = publication.pop("resolved_domain_sha256")
    mutate(publication)
    domain = ResolvedDomain(publication, digest)

    with pytest.raises(ConfigurationError):
        index_bag(tmp_path / "never-opened.bag", domain)


def _index_fake(tmp_path, monkeypatch, domain, records, topic_types, name="sample.bag"):
    path = tmp_path / name
    path.write_bytes(b"immutable-fake-bag")
    fake = FakeBag(records, topic_types)
    _install_fake(monkeypatch, fake)
    return index_bag(path, domain), fake


def _tf_messages(records, tf_topic):
    return [message for topic, message, receipt in records if topic == tf_topic]


def _ego_messages(records, ego_topic):
    return [message for topic, message, receipt in records if topic == ego_topic]


def test_discover_bags_is_recursive_deterministic_and_ignores_active(tmp_path) -> None:
    (tmp_path / "z").mkdir()
    (tmp_path / "a").mkdir()
    first = tmp_path / "a" / "b.bag"
    second = tmp_path / "z" / "a.bag"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    (tmp_path / "a" / "bad.bag.active").write_bytes(b"partial")
    link = tmp_path / "a" / "linked.bag"
    link.symlink_to(second)

    assert discover_bags(tmp_path) == (first, link, second)
    assert discover_bags(tmp_path / "missing") == ()
    with pytest.raises(NotADirectoryError):
        discover_bags(first)


def test_discover_empty_directory_and_active_file_never_open(tmp_path, monkeypatch, resolved_domain) -> None:
    assert discover_bags(tmp_path) == ()
    active = tmp_path / "recording.bag.active"
    active.write_bytes(b"partial")
    monkeypatch.setattr(
        bag_reader,
        "_open_bag",
        lambda path: (_ for _ in ()).throw(AssertionError("active bag was opened")),
    )

    assert discover_bags(tmp_path) == ()
    with pytest.raises(BagReadError):
        index_bag(active, resolved_domain)


@pytest.mark.parametrize(
    "include_tf,include_ego,expected,reason",
    [
        (True, True, BagDisposition.ELIGIBLE, None),
        (False, False, BagDisposition.SENSOR_ONLY_EXCLUDED, None),
        (False, True, BagDisposition.REJECTED, BagRejectReason.MISSING_REQUIRED_TOPIC),
        (True, False, BagDisposition.REJECTED, BagRejectReason.MISSING_REQUIRED_TOPIC),
    ],
)
def test_inventory_classification_matrix(
    tmp_path,
    monkeypatch,
    resolved_domain,
    jpeg_by_role,
    include_tf,
    include_ego,
    expected,
    reason,
) -> None:
    path = tmp_path / "sample.bag"
    path.write_bytes(b"immutable-fake-bag")
    records = _eligible_records(resolved_domain, jpeg_by_role)
    tf_topic = resolved_domain.document["ros_contract"]["policy_topics"]["transforms"]
    ego_topic = resolved_domain.document["ros_contract"]["policy_topics"]["ego_state"]
    records = [
        row
        for row in records
        if (include_tf or row[0] != tf_topic) and (include_ego or row[0] != ego_topic)
    ]
    fake = FakeBag(records, _types(resolved_domain, include_tf=include_tf, include_ego=include_ego))
    _install_fake(monkeypatch, fake)

    actual = index_bag(path, resolved_domain)

    assert actual.inventory.disposition is expected
    assert actual.inventory.rejection_code is reason
    assert actual.source_bag_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert fake.closed


@pytest.mark.parametrize("missing_role", ["front", "left", "right", "lidar"])
def test_each_missing_core_sensor_is_rejected(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, missing_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    if missing_role == "lidar":
        missing_topic = resolved_domain.document["sensors"]["lidar"]["topic"]
    else:
        missing_topic = next(
            camera["topic"]
            for camera in resolved_domain.document["sensors"]["cameras"]
            if camera["role"] == missing_role
        )
    records = [row for row in records if row[0] != missing_topic]
    topic_types = [row for row in _types(resolved_domain) if row[0] != missing_topic]

    actual, _ = _index_fake(tmp_path, monkeypatch, resolved_domain, records, topic_types)

    assert actual.inventory.disposition is BagDisposition.REJECTED
    assert actual.inventory.rejection_code is BagRejectReason.MISSING_REQUIRED_TOPIC


@pytest.mark.parametrize("mode", ["wrong", "mixed"])
@pytest.mark.parametrize("role", ["front", "lidar", "tf", "ego"])
def test_wrong_or_mixed_connection_types_are_rejected_and_still_counted(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, role, mode
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    policy_topics = resolved_domain.document["ros_contract"]["policy_topics"]
    if role == "front":
        topic = next(
            camera["topic"]
            for camera in resolved_domain.document["sensors"]["cameras"]
            if camera["role"] == "front"
        )
    elif role == "lidar":
        topic = resolved_domain.document["sensors"]["lidar"]["topic"]
    elif role == "tf":
        topic = policy_topics["transforms"]
    else:
        topic = policy_topics["ego_state"]
    topic_types = _types(resolved_domain)
    expected = next(datatype for candidate, datatype in topic_types if candidate == topic)
    topic_types = [row for row in topic_types if row[0] != topic]
    topic_types.append((topic, "example_msgs/Wrong"))
    if mode == "mixed":
        topic_types.append((topic, expected))
    expected_count = sum(1 for candidate, message, receipt in records if candidate == topic)

    actual, _ = _index_fake(tmp_path, monkeypatch, resolved_domain, records, topic_types)

    assert actual.inventory.rejection_code is BagRejectReason.MESSAGE_TYPE_MISMATCH
    assert dict(actual.inventory.topic_record_counts)[topic] == expected_count


@pytest.mark.parametrize("role", ["front", "left", "right", "lidar"])
def test_every_sensor_frame_mismatch_is_rejected(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    if role == "lidar":
        topic = resolved_domain.document["sensors"]["lidar"]["topic"]
    else:
        topic = next(
            camera["topic"]
            for camera in resolved_domain.document["sensors"]["cameras"]
            if camera["role"] == role
        )
    message = next(message for candidate, message, receipt in records if candidate == topic)
    message.header.frame_id = "/wrong-frame"

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.rejection_code is BagRejectReason.FRAME_ID_MISMATCH


@pytest.mark.parametrize("invalid_frame", ["", "///", None])
@pytest.mark.parametrize("sensor_kind", ["camera", "lidar"])
def test_invalid_sensor_frame_value_is_frame_mismatch_rejection(
    tmp_path,
    monkeypatch,
    resolved_domain,
    jpeg_by_role,
    sensor_kind,
    invalid_frame,
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    if sensor_kind == "camera":
        topic = next(
            camera["topic"]
            for camera in resolved_domain.document["sensors"]["cameras"]
            if camera["role"] == "front"
        )
    else:
        topic = resolved_domain.document["sensors"]["lidar"]["topic"]
    message = next(
        message for candidate, message, receipt in records if candidate == topic
    )
    message.header.frame_id = invalid_frame

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.disposition is BagDisposition.REJECTED
    assert actual.inventory.rejection_code is BagRejectReason.FRAME_ID_MISMATCH


def test_pointcloud_layout_mismatch_is_message_type_rejection(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    lidar_topic = resolved_domain.document["sensors"]["lidar"]["topic"]
    message = next(message for topic, message, receipt in records if topic == lidar_topic)
    message.fields[0].datatype = 8

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.rejection_code is BagRejectReason.MESSAGE_TYPE_MISMATCH


@pytest.mark.parametrize("role", ["front", "left", "right"])
def test_successfully_decoded_wrong_size_camera_rejects_bag(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    success, encoded = cv2.imencode(".jpg", np.zeros((16, 32, 3), dtype=np.uint8))
    assert success
    topic = next(
        camera["topic"]
        for camera in resolved_domain.document["sensors"]["cameras"]
        if camera["role"] == role
    )
    message = next(message for candidate, message, receipt in records if candidate == topic)
    message.data = bytes(encoded)

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.rejection_code is BagRejectReason.RESOLUTION_MISMATCH
    if role == "front":
        assert actual.front == ()
    else:
        assert len(getattr(actual, role)) == 1


def test_corrupt_early_front_is_excluded_but_corrupt_side_refs_remain(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    cameras = {camera["role"]: camera for camera in resolved_domain.document["sensors"]["cameras"]}
    first_receipt = min(receipt for topic, message, receipt in records)
    corrupt_rows = [
        (
            cameras[role]["topic"],
            _camera(1, cameras[role]["frame_id"], b"not-a-jpeg"),
            first_receipt - 100 + offset,
        )
        for offset, role in enumerate(("front", "left", "right"))
    ]
    records = corrupt_rows + records

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    errors = dict(actual.inventory.camera_decode_error_record_indices)
    assert errors[cameras["front"]["topic"]] == (0,)
    assert errors[cameras["left"]["topic"]] == (0,)
    assert errors[cameras["right"]["topic"]] == (0,)
    assert tuple(record.stable_record_index for record in actual.front) == (1,)
    assert tuple(record.stable_record_index for record in actual.left) == (0, 1)
    assert tuple(record.stable_record_index for record in actual.right) == (0, 1)
    assert actual.inventory.disposition is BagDisposition.ELIGIBLE


def _canonical_transform(message):
    return next(
        transform
        for transform in message.transforms
        if transform.header.frame_id.lstrip("/") == "map"
        and transform.child_frame_id.lstrip("/") == "base_link"
    )


def _extrinsic_transforms(messages, child_frame):
    return tuple(
        transform
        for message in messages
        for transform in message.transforms
        if transform.header.frame_id.lstrip("/") == "base_link"
        and transform.child_frame_id.lstrip("/") == child_frame
    )


def test_tf_and_ego_epochs_are_independent_dense_receipt_order_views(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    topics = resolved_domain.document["ros_contract"]["policy_topics"]
    tf_messages = _tf_messages(records, topics["transforms"])
    ego_messages = _ego_messages(records, topics["ego_state"])
    tf_headers = (100, 100, 101, 100, 101, 100, 101, 102, 103, 104)
    ego_headers = (10, 10, 11, 12, 11, 12, 11, 12, 13, 14)
    for index, (message, header_ns) in enumerate(zip(tf_messages, tf_headers)):
        transform = _canonical_transform(message)
        transform.header.stamp = _stamp(header_ns)
        if index == 1:
            transform.transform.translation = _vec((0.0, 0.0, 0.0))
            transform.transform.rotation = _quaternion(tuple(-value for value in IDENTITY))
    for index, (message, header_ns) in enumerate(zip(ego_messages, ego_headers)):
        message.header.stamp = _stamp(header_ns)
        if index == 1:
            message.position = _vec((0.0, 0.0, 0.0))

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert tuple(record.clock_epoch_index for record in actual.raw_poses) == (
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
    )
    assert tuple(record.clock_epoch_index for record in actual.raw_ego) == (
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        2,
        2,
    )
    assert tuple(view.index for view in actual.tf_clock_epochs) == (0, 1, 2)
    assert tuple(view.index for view in actual.ego_clock_epochs) == (0, 1, 2)
    assert actual.tf_clock_epochs[0].raw_records[:2] == actual.raw_poses[:2]
    assert actual.tf_clock_epochs[0].interpolation_records[0] is actual.raw_poses[0]
    assert actual.tf_clock_epochs[0].interpolation_records[1] is actual.raw_poses[2]
    assert actual.ego_clock_epochs[0].interpolation_records[0] is actual.raw_ego[0]
    assert actual.ego_clock_epochs[0].interpolation_records[1] is actual.raw_ego[2]
    assert tuple(record.stable_record_index for record in actual.raw_poses) == tuple(
        range(0, 50, 5)
    )
    assert tuple(record.stable_record_index for record in actual.raw_ego) == tuple(range(10))


@pytest.mark.parametrize(
    "delta,expected",
    [
        (1.0e-6, BagDisposition.ELIGIBLE),
        (1.0001e-6, BagDisposition.REJECTED),
    ],
)
def test_tf_duplicate_translation_tolerance_is_inclusive_and_q_sign_invariant(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, delta, expected
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    topics = resolved_domain.document["ros_contract"]["policy_topics"]
    tf_messages = _tf_messages(records, topics["transforms"])
    first = _canonical_transform(tf_messages[0])
    duplicate = _canonical_transform(tf_messages[1])
    duplicate.header.stamp = first.header.stamp
    duplicate.transform.translation = _vec((delta, 0.0, 0.0))
    duplicate.transform.rotation = _quaternion(tuple(-value for value in IDENTITY))
    ego_messages = _ego_messages(records, topics["ego_state"])
    ego_messages[1].header.stamp = ego_messages[0].header.stamp
    ego_messages[1].position = _vec((0.0, 0.0, 0.0))

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.disposition is expected
    assert len(actual.raw_poses) == 10
    assert len(actual.tf_clock_epochs[0].raw_records) == 10
    assert len(actual.tf_clock_epochs[0].interpolation_records) == 9
    if expected is BagDisposition.REJECTED:
        assert actual.inventory.rejection_code is BagRejectReason.DUPLICATE_STATE_CONFLICT


@pytest.mark.parametrize(
    "angle_rad,expected",
    [
        (1.0e-6, BagDisposition.ELIGIBLE),
        (1.0001e-6, BagDisposition.REJECTED),
    ],
)
def test_tf_duplicate_rotation_tolerance_is_mathematically_inclusive(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, angle_rad, expected
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    topics = resolved_domain.document["ros_contract"]["policy_topics"]
    tf_messages = _tf_messages(records, topics["transforms"])
    first = _canonical_transform(tf_messages[0])
    duplicate = _canonical_transform(tf_messages[1])
    duplicate.header.stamp = first.header.stamp
    duplicate.transform.translation = _vec((0.0, 0.0, 0.0))
    duplicate.transform.rotation = _quaternion(
        (0.0, 0.0, math.sin(angle_rad / 2.0), math.cos(angle_rad / 2.0))
    )
    ego_messages = _ego_messages(records, topics["ego_state"])
    ego_messages[1].header.stamp = ego_messages[0].header.stamp
    ego_messages[1].position = _vec((0.0, 0.0, 0.0))

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.disposition is expected
    if expected is BagDisposition.REJECTED:
        assert actual.inventory.rejection_code is BagRejectReason.DUPLICATE_STATE_CONFLICT


@pytest.mark.parametrize("conflict", [False, True])
def test_ego_duplicates_remain_raw_and_conflicting_used_fields_reject(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, conflict
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    topics = resolved_domain.document["ros_contract"]["policy_topics"]
    ego_messages = _ego_messages(records, topics["ego_state"])
    ego_messages[1].header.stamp = ego_messages[0].header.stamp
    ego_messages[1].position = _vec((0.0, 0.0, 0.0))
    if conflict:
        ego_messages[1].velocity.x += 0.25
    tf_messages = _tf_messages(records, topics["transforms"])
    second_tf = _canonical_transform(tf_messages[1])
    second_tf.header.stamp = _canonical_transform(tf_messages[0]).header.stamp
    second_tf.transform.translation = _vec((0.0, 0.0, 0.0))

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert len(actual.raw_ego) == 10
    assert len(actual.ego_clock_epochs[0].raw_records) == 10
    assert len(actual.ego_clock_epochs[0].interpolation_records) == 9
    assert actual.ego_clock_epochs[0].interpolation_records[0] is actual.raw_ego[0]
    if conflict:
        assert actual.inventory.rejection_code is BagRejectReason.DUPLICATE_STATE_CONFLICT
    else:
        assert actual.inventory.disposition is BagDisposition.ELIGIBLE


def test_equal_header_ego_diagnostic_fields_are_not_duplicate_state_fields(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    topics = resolved_domain.document["ros_contract"]["policy_topics"]
    ego_messages = _ego_messages(records, topics["ego_state"])
    ego_messages[1].header.stamp = ego_messages[0].header.stamp
    ego_messages[1].position = _vec((0.01, 0.0, 0.0))
    ego_messages[1].heading = 0.1
    tf_messages = _tf_messages(records, topics["transforms"])
    duplicate = _canonical_transform(tf_messages[1])
    duplicate.header.stamp = _canonical_transform(tf_messages[0]).header.stamp
    duplicate.transform.translation = _vec((0.0, 0.0, 0.0))

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.disposition is BagDisposition.ELIGIBLE
    assert len(actual.raw_ego) == 10
    assert len(actual.ego_clock_epochs[0].interpolation_records) == 9


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("q_sign", BagDisposition.ELIGIBLE),
        ("translation_limit", BagDisposition.ELIGIBLE),
        ("translation_over", BagDisposition.REJECTED),
        ("missing", BagDisposition.REJECTED),
    ],
)
def test_extrinsic_gate_checks_every_repeat_presence_sign_and_inclusive_tolerance(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, variant, expected
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    tf_topic = resolved_domain.document["ros_contract"]["policy_topics"]["transforms"]
    messages = _tf_messages(records, tf_topic)
    camera = next(
        row
        for row in resolved_domain.document["sensors"]["cameras"]
        if row["role"] == "left"
    )
    transforms = _extrinsic_transforms(messages, camera["frame_id"])
    if variant == "missing":
        for message in messages:
            message.transforms = [
                transform
                for transform in message.transforms
                if transform.child_frame_id.lstrip("/") != camera["frame_id"]
            ]
    elif variant == "q_sign":
        for transform in transforms:
            raw = transform.transform.rotation
            transform.transform.rotation = _quaternion((-raw.x, -raw.y, -raw.z, -raw.w))
    else:
        offset = 1.0e-4 if variant == "translation_limit" else 1.0001e-4
        for transform in transforms:
            transform.transform.translation.x += offset

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.disposition is expected
    if expected is BagDisposition.REJECTED:
        assert actual.inventory.rejection_code is BagRejectReason.EXTRINSIC_MISMATCH


@pytest.mark.parametrize(
    "angle_rad,expected",
    [
        (1.0e-4, BagDisposition.ELIGIBLE),
        (1.0001e-4, BagDisposition.REJECTED),
    ],
)
def test_extrinsic_rotation_tolerance_is_inclusive(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role, angle_rad, expected
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    tf_topic = resolved_domain.document["ros_contract"]["policy_topics"]["transforms"]
    lidar = resolved_domain.document["sensors"]["lidar"]
    for transform in _extrinsic_transforms(
        _tf_messages(records, tf_topic), lidar["frame_id"]
    ):
        transform.transform.rotation = _quaternion(
            (0.0, 0.0, math.sin(angle_rad / 2.0), math.cos(angle_rad / 2.0))
        )

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.disposition is expected
    if expected is BagDisposition.REJECTED:
        assert actual.inventory.rejection_code is BagRejectReason.EXTRINSIC_MISMATCH


def test_tf_ego_p99_failure_maps_to_declared_rejection(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    ego_topic = resolved_domain.document["ros_contract"]["policy_topics"]["ego_state"]
    for message in _ego_messages(records, ego_topic):
        message.position.x += 1.0

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.rejection_code is BagRejectReason.TF_EGO_POSE_MISMATCH
    assert actual.inventory.tf_ego_diagnostic is not None
    assert actual.inventory.tf_ego_diagnostic.metric.position_p99_m == pytest.approx(1.0)


def test_tf_ego_minimum_pair_failure_preserves_finite_nine_pair_report(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    topics = resolved_domain.document["ros_contract"]["policy_topics"]
    last_tf = _tf_messages(records, topics["transforms"])[-1]
    last_ego = _ego_messages(records, topics["ego_state"])[-1]
    records = [
        row for row in records if row[1] is not last_tf and row[1] is not last_ego
    ]

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.rejection_code is BagRejectReason.TF_EGO_POSE_MISMATCH
    assert actual.inventory.tf_ego_diagnostic is not None
    assert actual.inventory.tf_ego_diagnostic.metric.comparable_count == 9
    assert actual.inventory.tf_ego_diagnostic.epoch_local_comparable_count == 9


def test_rejection_precedence_is_frame_before_resolution_duplicate_and_diagnostic(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    sensors = resolved_domain.document["sensors"]["cameras"]
    left_topic = next(row["topic"] for row in sensors if row["role"] == "left")
    right_topic = next(row["topic"] for row in sensors if row["role"] == "right")
    next(message for topic, message, receipt in records if topic == left_topic).header.frame_id = "wrong"
    success, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert success
    next(message for topic, message, receipt in records if topic == right_topic).data = bytes(encoded)
    topics = resolved_domain.document["ros_contract"]["policy_topics"]
    tf_messages = _tf_messages(records, topics["transforms"])
    duplicate = _canonical_transform(tf_messages[1])
    duplicate.header.stamp = _canonical_transform(tf_messages[0]).header.stamp
    duplicate.transform.translation.x = 1.0
    for message in _ego_messages(records, topics["ego_state"]):
        message.position.x += 5.0

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert actual.inventory.rejection_code is BagRejectReason.FRAME_ID_MISMATCH


def test_tf_flat_indices_count_irrelevant_transforms_across_envelopes(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    records = _eligible_records(resolved_domain, jpeg_by_role)
    tf_topic = resolved_domain.document["ros_contract"]["policy_topics"]["transforms"]
    for index, message in enumerate(_tf_messages(records, tf_topic)):
        header_ns = 50_000 + index
        message.transforms.insert(0, _transform("/odom", "/other", header_ns, (0.0, 0.0, 0.0)))
        message.transforms.append(_transform("/foo", "/bar", header_ns, (0.0, 0.0, 0.0)))

    actual, _ = _index_fake(
        tmp_path, monkeypatch, resolved_domain, records, _types(resolved_domain)
    )

    assert tuple(record.stable_record_index for record in actual.raw_poses) == tuple(
        range(1, 70, 7)
    )
    assert actual.raw_poses[0] is actual.tf_clock_epochs[0].raw_records[0]


def test_index_preserves_dual_clocks_flat_indices_and_does_not_retain_payloads(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    path = tmp_path / "dual-clock.bag"
    path.write_bytes(b"dual-clock")
    records = _eligible_records(resolved_domain, jpeg_by_role)
    fake = FakeBag(records, _types(resolved_domain))
    _install_fake(monkeypatch, fake)

    actual = index_bag(path, resolved_domain)

    assert actual.inventory.disposition is BagDisposition.ELIGIBLE
    assert actual.front[0].receipt_stamp_ns - actual.front[0].header_stamp_ns > 800_000_000_000
    assert tuple(record.stable_record_index for record in actual.raw_ego) == tuple(range(10))
    assert tuple(record.stable_record_index for record in actual.raw_poses) == tuple(
        range(0, 50, 5)
    )
    assert actual.inventory.tf_ego_diagnostic.metric.comparable_count == 10

    def walk(value, seen):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        assert not isinstance(value, (bytes, bytearray, np.ndarray))
        if isinstance(value, tuple):
            for item in value:
                walk(item, seen)
        elif hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                walk(getattr(value, name), seen)

    walk(actual, set())


def _tf_record(header, receipt, index, epoch, x=0.0):
    return TfPoseRecord(header, receipt, index, epoch, Pose((x, 0.0, 0.0), IDENTITY))


def test_causal_suffix_starts_at_exact_later_duplicate_anchor() -> None:
    earlier = _tf_record(100, 10, 4, 0)
    anchor = _tf_record(100, 20, 9, 0)
    future = _tf_record(200, 30, 10, 0)
    epoch = ClockEpochView(0, (earlier, anchor, future), (earlier, future))

    actual = causal_tf_interpolation_view(epoch, anchor)

    assert actual == (anchor, future)
    assert actual[0] is anchor


def test_causal_suffix_rejects_nonidentity_wrong_epoch_and_conflicting_duplicate() -> None:
    anchor = _tf_record(100, 10, 1, 0)
    conflict = _tf_record(100, 20, 2, 0, x=1.0)
    epoch = ClockEpochView(0, (anchor, conflict), (anchor,))
    copied_anchor = _tf_record(100, 10, 1, 0)

    with pytest.raises(ValueError, match="identity"):
        causal_tf_interpolation_view(epoch, copied_anchor)
    with pytest.raises(ValueError, match="epoch"):
        causal_tf_interpolation_view(epoch, _tf_record(100, 10, 1, 1))
    with pytest.raises(ValueError, match="conflicting"):
        causal_tf_interpolation_view(epoch, anchor)


def _diagnostic_index(raw_tf, epochs, raw_ego):
    empty_inventory = BagInventory(
        disposition=BagDisposition.ELIGIBLE,
        rejection_code=None,
        topic_types=(),
        topic_record_counts=(),
        camera_decode_error_record_indices=(),
        tf_ego_diagnostic=None,
    )
    ego_interpolation = tuple(
        record
        for index, record in enumerate(raw_ego)
        if index == 0 or record.header_stamp_ns != raw_ego[index - 1].header_stamp_ns
    )
    return BagIndex(
        path=Path("synthetic.bag"),
        source_bag_sha256="0" * 64,
        inventory=empty_inventory,
        front=(),
        left=(),
        right=(),
        lidar=(),
        raw_ego=raw_ego,
        ego_clock_epochs=(ClockEpochView(0, raw_ego, ego_interpolation),),
        raw_poses=raw_tf,
        tf_clock_epochs=epochs,
        extrinsics=(),
    )


def test_epoch_safe_diagnostic_does_not_fallback_to_another_local_epoch() -> None:
    epoch0_before = _tf_record(0, 0, 0, 0)
    epoch0_after = _tf_record(100, 10, 1, 0)
    epoch1_far = _tf_record(500, 1_000, 2, 1)
    raw_tf = (epoch0_before, epoch0_after, epoch1_far)
    epochs = (
        ClockEpochView(0, (epoch0_before, epoch0_after), (epoch0_before, epoch0_after)),
        ClockEpochView(1, (epoch1_far,), (epoch1_far,)),
    )
    value = EgoValue((0.0, 0.0), (0.0, 0.0), EgoDiagnosticValue((0.0, 0.0), 0.0))
    raw_ego = tuple(EgoRecord(50, i, i, 0, value) for i in range(10)) + (
        EgoRecord(50, 1_000, 10, 0, value),
    )
    index = _diagnostic_index(raw_tf, epochs, raw_ego)

    actual = compute_epoch_safe_tf_ego_diagnostic(index, max_bracket_ns=100)

    assert actual.raw_ego_record_count == 11
    assert actual.epoch_local_comparable_count == 10
    assert actual.no_epoch_local_bracket_count == 1
    assert actual.ambiguous_epoch_selection_count == 0
    assert actual.cross_epoch_bracket_prevented_count == 0
    assert actual.provenance[0].tf_bracket_record_index == (0, 1)
    assert actual.provenance[-1].selected_tf_clock_epoch_index == 1
    assert actual.provenance[-1].selected_tf_anchor_record_index == 2
    assert actual.provenance[-1].tf_bracket_header_stamp_ns is None
    assert actual.provenance[-1].exclusion_reason == "NO_EPOCH_LOCAL_BRACKET"


def test_epoch_safe_diagnostic_reports_only_actual_global_cross_epoch_bracket() -> None:
    epoch0_before = _tf_record(0, 0, 0, 0)
    epoch0_after = _tf_record(100, 10, 1, 0)
    epoch1_before = _tf_record(0, 400, 2, 1)
    epoch1_after = _tf_record(100, 410, 3, 1)
    epoch2_anchor = _tf_record(-100, 1_000, 4, 2)
    raw_tf = (
        epoch0_before,
        epoch0_after,
        epoch1_before,
        epoch1_after,
        epoch2_anchor,
    )
    epochs = (
        ClockEpochView(0, raw_tf[:2], raw_tf[:2]),
        ClockEpochView(1, raw_tf[2:4], raw_tf[2:4]),
        ClockEpochView(2, raw_tf[4:], raw_tf[4:]),
    )
    value = EgoValue(
        (0.0, 0.0),
        (0.0, 0.0),
        EgoDiagnosticValue((0.0, 0.0), 0.0),
    )
    raw_ego = tuple(EgoRecord(50, index, index, 0, value) for index in range(10)) + (
        EgoRecord(50, 1_000, 10, 0, value),
        EgoRecord(9_999, 1_001, 11, 0, value),
    )

    actual = compute_epoch_safe_tf_ego_diagnostic(
        _diagnostic_index(raw_tf, epochs, raw_ego), max_bracket_ns=100
    )

    assert actual.metric.comparable_count == 10
    assert actual.metric.excluded_count == 2
    assert actual.ambiguous_epoch_selection_count == 0
    assert actual.no_epoch_local_bracket_count == 1
    assert actual.cross_epoch_bracket_prevented_count == 1
    assert actual.provenance[-2].exclusion_reason == "CROSS_EPOCH_BRACKET_PREVENTED"
    assert actual.provenance[-1].exclusion_reason == "NO_EPOCH_LOCAL_BRACKET"


def test_epoch_safe_receipt_selector_ties_by_receipt_then_stable_index() -> None:
    first = _tf_record(50, 0, 2, 0)
    same_receipt_later = _tf_record(50, 0, 5, 0)
    later_receipt = _tf_record(50, 20, 6, 0)
    raw_tf = (first, same_receipt_later, later_receipt)
    epochs = (ClockEpochView(0, raw_tf, (first,)),)
    value = EgoValue(
        (0.0, 0.0),
        (0.0, 0.0),
        EgoDiagnosticValue((0.0, 0.0), 0.0),
    )
    raw_ego = tuple(EgoRecord(50, 10, index, 0, value) for index in range(10))

    actual = compute_epoch_safe_tf_ego_diagnostic(
        _diagnostic_index(raw_tf, epochs, raw_ego), max_bracket_ns=0
    )

    assert all(
        item.selected_tf_anchor_record_index == 2 for item in actual.provenance
    )
    assert all(item.tf_bracket_record_index == (2, 2) for item in actual.provenance)


def test_source_stat_change_is_integrity_error(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    path = tmp_path / "mutating.bag"
    path.write_bytes(b"before")
    fake = FakeBag(_eligible_records(resolved_domain, jpeg_by_role), _types(resolved_domain))

    class MutatingContext:
        def __enter__(self):
            return fake

        def __exit__(self, exc_type, exc, traceback):
            fake.closed = True
            path.write_bytes(b"after-is-different")

    monkeypatch.setattr(bag_reader, "_open_bag", lambda ignored: MutatingContext())

    with pytest.raises(IntegrityError):
        index_bag(path, resolved_domain)


def test_same_size_same_mtime_source_byte_change_is_integrity_error(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    path = tmp_path / "hash-mutating.bag"
    path.write_bytes(b"before")
    original_mtime_ns = path.stat().st_mtime_ns
    fake = FakeBag(_eligible_records(resolved_domain, jpeg_by_role), _types(resolved_domain))

    class HashMutatingContext:
        def __enter__(self):
            return fake

        def __exit__(self, exc_type, exc, traceback):
            fake.closed = True
            path.write_bytes(b"differ")
            os.utime(path, ns=(original_mtime_ns, original_mtime_ns))

    monkeypatch.setattr(bag_reader, "_open_bag", lambda ignored: HashMutatingContext())

    with pytest.raises(IntegrityError, match="bytes|hash"):
        index_bag(path, resolved_domain)


def test_active_path_is_an_operational_error(tmp_path, resolved_domain) -> None:
    active = tmp_path / "unfinished.bag.active"
    active.write_bytes(b"partial")

    with pytest.raises(BagReadError):
        index_bag(active, resolved_domain)


def test_pass_two_touches_only_selected_records_and_continues_after_failure(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "payloads.bag"
    path.write_bytes(b"payloads")

    class Poison:
        _type = "sensor_msgs/CompressedImage"
        header = _header(1, "Camera-1")

        @property
        def data(self):
            raise AssertionError("unselected payload accessed")

    class SelectedFailure(Poison):
        @property
        def data(self):
            raise RuntimeError("environment-specific failure /tmp/x 0x123")

    records = [
        ("/camera", Poison(), 10),
        ("/camera", SelectedFailure(), 20),
        ("/camera", _camera(3, "Camera-1", b"selected"), 30),
    ]
    fake = FakeBag(records, (("/camera", "sensor_msgs/CompressedImage"),))
    _install_fake(monkeypatch, fake)

    events = tuple(iter_selected_payloads(path, {"/camera": {1, 2}}))

    assert isinstance(events[0], SensorPayloadFailure)
    assert events[0].stable_record_index == 1
    assert events[0].detail == "CAMERA_DATA_ACCESS_FAILED"
    assert events[1] == SensorPayload(
        RecordRef("/camera", 3, 30, 2), SensorPayloadKind.CAMERA_JPEG, b"selected"
    )
    assert fake.closed


def test_pass_two_does_not_ask_rosbag_to_deserialise_empty_selection_topics(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "active-and-empty-selection.bag"
    path.write_bytes(b"payloads")

    class ExactTopicsBag(FakeBag):
        def read_messages(self, topics=None):
            assert tuple(topics) == ("/camera",)
            yield from super().read_messages(topics)

    fake = ExactTopicsBag(
        [("/camera", _camera(1, "Camera-1", b"selected"), 10)],
        (
            ("/camera", "sensor_msgs/CompressedImage"),
            ("/lidar", "sensor_msgs/PointCloud2"),
        ),
    )
    _install_fake(monkeypatch, fake)

    events = tuple(
        iter_selected_payloads(path, {"/camera": {0}, "/lidar": set()})
    )

    assert len(events) == 1
    assert events[0].ref == RecordRef("/camera", 1, 10, 0)
    assert fake.read_calls == 1
    assert fake.closed


def test_pass_two_all_empty_known_selections_do_not_start_a_rosbag_stream(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "all-empty-selection.bag"
    path.write_bytes(b"payloads")

    class NoReadBag(FakeBag):
        def read_messages(self, topics=None):
            raise AssertionError("empty ROS topic filter would deserialize all records")
            yield

    fake = NoReadBag(
        [("/camera", object(), 10), ("/lidar", object(), 20)],
        (
            ("/camera", "sensor_msgs/CompressedImage"),
            ("/lidar", "sensor_msgs/PointCloud2"),
        ),
    )
    _install_fake(monkeypatch, fake)

    events = tuple(
        iter_selected_payloads(path, {"/camera": set(), "/lidar": set()})
    )

    assert events == ()
    assert fake.read_calls == 0
    assert fake.closed


def test_pass_two_rejects_unknown_or_missing_requested_index(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "payloads.bag"
    path.write_bytes(b"payloads")
    fake = FakeBag([], (("/camera", "sensor_msgs/CompressedImage"),))
    _install_fake(monkeypatch, fake)

    with pytest.raises((ValueError, IntegrityError)):
        tuple(iter_selected_payloads(path, {"/unknown": {0}}))
    with pytest.raises(ValueError):
        tuple(iter_selected_payloads(path, {"/unknown": set()}))
    with pytest.raises(IntegrityError):
        tuple(iter_selected_payloads(path, {"/camera": {9}}))


def test_pass_two_lidar_success_failure_poison_and_bag_order(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "selected-lidar.bag"
    path.write_bytes(b"payloads")

    class Cloud:
        _type = "sensor_msgs/PointCloud2"

        def __init__(self, header_ns, points):
            self.header = _header(header_ns, "Lidar3D-4")
            self.points = points

    class PoisonCloud:
        _type = "sensor_msgs/PointCloud2"
        header = _header(1, "Lidar3D-4")

        @property
        def points(self):
            raise AssertionError("unselected cloud decoded")

        @property
        def data(self):
            raise AssertionError("unselected cloud data touched")

    class BrokenCloud(Cloud):
        @property
        def points(self):
            raise RuntimeError("machine path /tmp/private object 0x123")

        @points.setter
        def points(self, value):
            del value

    point_cloud2 = _ns(
        read_points=lambda message, field_names, skip_nans: iter(message.points)
    )
    monkeypatch.setitem(sys.modules, "sensor_msgs", _ns(point_cloud2=point_cloud2))
    monkeypatch.setitem(sys.modules, "sensor_msgs.point_cloud2", point_cloud2)
    records = [
        ("/lidar", PoisonCloud(), 10),
        ("/camera", _camera(2, "Camera-1", b"first"), 20),
        ("/lidar", Cloud(3, ((1.0, 2.0, 3.0, 4.0),)), 30),
        ("/lidar", BrokenCloud(4, ()), 40),
        ("/camera", _camera(5, "Camera-1", b"last"), 50),
    ]
    fake = FakeBag(
        records,
        (
            ("/camera", "sensor_msgs/CompressedImage"),
            ("/lidar", "sensor_msgs/PointCloud2"),
        ),
    )
    _install_fake(monkeypatch, fake)

    events = tuple(
        iter_selected_payloads(
            path,
            {"/camera": {0, 1}, "/lidar": {1, 2}},
        )
    )

    assert [type(event) for event in events] == [
        SensorPayload,
        SensorPayload,
        SensorPayloadFailure,
        SensorPayload,
    ]
    assert events[0].ref == RecordRef("/camera", 2, 20, 0)
    assert events[0].value == b"first"
    assert events[1].kind is SensorPayloadKind.LIDAR_XYZI
    np.testing.assert_array_equal(
        events[1].value, np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    )
    assert events[1].value.flags.c_contiguous
    assert events[1].value.flags.writeable is False
    assert events[2] == SensorPayloadFailure(
        "/lidar", 2, "DECODE_ERROR", "POINTCLOUD_DECODE_FAILED"
    )
    assert events[3].ref == RecordRef("/camera", 5, 50, 1)
    assert events[3].value == b"last"
    assert fake.read_calls == 1
    assert fake.closed


def test_pass_two_missing_ros_backend_is_fatal_not_a_record_decode_failure(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "missing-backend.bag"
    path.write_bytes(b"payloads")
    message = _ns(
        _type="sensor_msgs/PointCloud2",
        header=_header(1, "Lidar3D-4"),
    )
    fake = FakeBag(
        [("/lidar", message, 2)],
        (("/lidar", "sensor_msgs/PointCloud2"),),
    )
    _install_fake(monkeypatch, fake)
    monkeypatch.delitem(sys.modules, "sensor_msgs", raising=False)
    monkeypatch.delitem(sys.modules, "sensor_msgs.point_cloud2", raising=False)
    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "sensor_msgs" or name.startswith("sensor_msgs."):
            raise ImportError("ROS backend absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    with pytest.raises(BagReadError):
        tuple(iter_selected_payloads(path, {"/lidar": {0}}))


def test_pass_two_backend_stream_type_error_is_wrapped_as_bag_read_error(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "stream-error.bag"
    path.write_bytes(b"payloads")

    class TypeFailingBag(FakeBag):
        def read_messages(self, topics=None):
            self.read_calls += 1
            raise TypeError("backend deserialization failed")
            yield

    fake = TypeFailingBag(
        [], (("/camera", "sensor_msgs/CompressedImage"),)
    )
    _install_fake(monkeypatch, fake)

    with pytest.raises(BagReadError):
        tuple(iter_selected_payloads(path, {"/camera": {0}}))


@pytest.mark.parametrize(
    "mapping",
    [
        {"/camera": [0]},
        {"/camera": {-1}},
        {"/camera": {True}},
        {"": {0}},
    ],
)
def test_pass_two_rejects_invalid_selection_mapping(tmp_path, monkeypatch, mapping) -> None:
    path = tmp_path / "payloads.bag"
    path.write_bytes(b"payloads")
    fake = FakeBag([], (("/camera", "sensor_msgs/CompressedImage"),))
    _install_fake(monkeypatch, fake)

    with pytest.raises((TypeError, ValueError)):
        tuple(iter_selected_payloads(path, mapping))


def test_index_opens_reads_closes_once_and_repeats_deterministically(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    path = tmp_path / "repeat.bag"
    path.write_bytes(b"repeat")
    created = []

    def factory(ignored):
        fake = FakeBag(
            _eligible_records(resolved_domain, jpeg_by_role),
            _types(resolved_domain),
        )
        created.append(fake)
        return fake

    monkeypatch.setattr(bag_reader, "_open_bag", factory)

    first = index_bag(path, resolved_domain)
    second = index_bag(path, resolved_domain)

    assert first == second
    assert len(created) == 2
    assert all(fake.read_calls == 1 and fake.closed for fake in created)


def test_index_does_not_retain_message_and_reader_can_release_it(
    tmp_path, monkeypatch, resolved_domain, jpeg_by_role
) -> None:
    path = tmp_path / "gc.bag"
    path.write_bytes(b"gc")
    records = _eligible_records(resolved_domain, jpeg_by_role)
    front_topic = next(
        row["topic"]
        for row in resolved_domain.document["sensors"]["cameras"]
        if row["role"] == "front"
    )

    class WeakCamera:
        pass

    weak_message = WeakCamera()
    original = next(message for topic, message, receipt in records if topic == front_topic)
    weak_message._type = original._type
    weak_message.header = original.header
    weak_message.data = original.data
    for index, (topic, message, receipt) in enumerate(records):
        if topic == front_topic:
            records[index] = (topic, weak_message, receipt)
            break
    reference = weakref.ref(weak_message)

    class ClearingBag(FakeBag):
        def __exit__(self, exc_type, exc, traceback):
            self.closed = True
            self.records.clear()

    fake = ClearingBag(records, _types(resolved_domain))
    _install_fake(monkeypatch, fake)

    actual = index_bag(path, resolved_domain)
    del weak_message
    del original
    del records
    gc.collect()

    assert reference() is None
    assert actual.inventory.disposition is BagDisposition.ELIGIBLE


def test_read_failure_is_bag_read_error_and_reader_closes(
    tmp_path, monkeypatch, resolved_domain
) -> None:
    path = tmp_path / "broken.bag"
    path.write_bytes(b"broken")

    class FailingBag(FakeBag):
        def read_messages(self, topics=None):
            self.read_calls += 1
            raise RuntimeError("deserialization failed at /machine/path")
            yield

    fake = FailingBag([], _types(resolved_domain))
    _install_fake(monkeypatch, fake)

    with pytest.raises(BagReadError, match="read/deserialise"):
        index_bag(path, resolved_domain)
    assert fake.closed
