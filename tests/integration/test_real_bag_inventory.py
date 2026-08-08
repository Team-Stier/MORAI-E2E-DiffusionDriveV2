from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path

import pytest

from morai_ddv2.config.resolver import load_resolved_domain
from morai_ddv2.ros1 import BagDisposition, index_bag


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOLVED_DOMAIN = PROJECT_ROOT / "tests" / "fixtures" / "golden" / "resolved-domain.json"
SENSOR_PRESET = PROJECT_ROOT / "sensor_data" / "0725demo_ros.json"

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    REALBAG = pytest.mark.realbag


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@REALBAG
@pytest.mark.filterwarnings(
    "ignore:the imp module is deprecated in favour of importlib:DeprecationWarning"
)
def test_audited_anomalous_bag_inventory_is_read_only_and_epoch_safe() -> None:
    raw_path = os.environ.get("MORAI_REAL_BAG")
    if raw_path is None:
        pytest.skip("MORAI_REAL_BAG is unset")
    path = Path(raw_path)
    assert path.name == (
        "s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__"
        "scenario-000__20260731T170301KST.bag"
    )
    source_before = (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
    preset_before = (
        SENSOR_PRESET.stat().st_size,
        SENSOR_PRESET.stat().st_mtime_ns,
        _sha256(SENSOR_PRESET),
    )
    root_entries_before = tuple(sorted(item.name for item in PROJECT_ROOT.iterdir()))
    domain = load_resolved_domain(RESOLVED_DOMAIN)

    actual = index_bag(path, domain)

    assert actual.inventory.disposition is BagDisposition.ELIGIBLE
    assert actual.inventory.rejection_code is None
    assert actual.source_bag_sha256 == source_before[2]
    assert actual.inventory.tf_ego_diagnostic is not None
    assert actual.inventory.tf_ego_diagnostic.metric.comparable_count >= 10
    assert all((actual.front, actual.left, actual.right, actual.lidar))
    assert all((actual.raw_poses, actual.raw_ego, actual.tf_clock_epochs, actual.ego_clock_epochs))
    assert actual.extrinsics
    contract = domain.document["ros_contract"]
    expected_types = {
        camera["topic"]: contract["message_types"]["camera"]
        for camera in domain.document["sensors"]["cameras"]
    }
    expected_types[domain.document["sensors"]["lidar"]["topic"]] = contract[
        "message_types"
    ]["lidar"]
    expected_types[contract["policy_topics"]["transforms"]] = contract[
        "message_types"
    ]["transforms"]
    expected_types[contract["policy_topics"]["ego_state"]] = contract[
        "message_types"
    ]["ego_state"]
    assert dict(actual.inventory.topic_types) == expected_types
    assert all(count > 0 for topic, count in actual.inventory.topic_record_counts)
    expected_extrinsic_children = {
        camera["frame_id"] for camera in domain.document["sensors"]["cameras"]
    } | {domain.document["sensors"]["lidar"]["frame_id"]}
    assert {record.value.child_frame for record in actual.extrinsics} == (
        expected_extrinsic_children
    )
    nearest_pairs = tuple(
        (
            front,
            min(
                actual.raw_poses,
                key=lambda pose: (
                    abs(pose.receipt_stamp_ns - front.receipt_stamp_ns),
                    pose.receipt_stamp_ns,
                    pose.stable_record_index,
                ),
            ),
        )
        for front in actual.front
    )
    assert any(
        abs(front.header_stamp_ns - pose.header_stamp_ns) > 800_000_000_000
        for front, pose in nearest_pairs
    )
    assert min(pose.receipt_stamp_ns for pose in actual.raw_poses) <= max(
        front.receipt_stamp_ns for front in actual.front
    )
    assert min(front.receipt_stamp_ns for front in actual.front) <= max(
        pose.receipt_stamp_ns for pose in actual.raw_poses
    )
    for ref in actual.front + actual.left + actual.right + actual.lidar:
        assert type(ref.header_stamp_ns) is int
        assert type(ref.receipt_stamp_ns) is int
    for record in actual.raw_poses + actual.raw_ego:
        assert type(record.header_stamp_ns) is int
        assert type(record.receipt_stamp_ns) is int
        assert type(record.clock_epoch_index) is int
    assert tuple(view.index for view in actual.tf_clock_epochs) == tuple(
        range(len(actual.tf_clock_epochs))
    )
    assert tuple(view.index for view in actual.ego_clock_epochs) == tuple(
        range(len(actual.ego_clock_epochs))
    )
    assert all(
        any(member is record for member in actual.tf_clock_epochs[record.clock_epoch_index].raw_records)
        for record in actual.raw_poses
    )
    assert all(
        any(member is record for member in actual.ego_clock_epochs[record.clock_epoch_index].raw_records)
        for record in actual.raw_ego
    )

    assert (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path)) == source_before
    assert (
        SENSOR_PRESET.stat().st_size,
        SENSOR_PRESET.stat().st_mtime_ns,
        _sha256(SENSOR_PRESET),
    ) == preset_before
    assert tuple(sorted(item.name for item in PROJECT_ROOT.iterdir())) == root_entries_before
