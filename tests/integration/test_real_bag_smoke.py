from __future__ import annotations

import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pytest

from morai_ddv2.cache import CACHE_DATASET_NAMES
from morai_ddv2.cache.loader import load_tensor_sample, load_verified_manifest
from morai_ddv2.cache.manifest import (
    DIAGNOSTIC_PROVENANCE_FIELDS,
    DIAGNOSTIC_REPORT_FIELDS,
    PROVENANCE_DATASET_FIELDS,
)
from morai_ddv2.cache.sidecar import COMMAND_RECORD_FIELDS
from morai_ddv2.config import canonical_json_bytes, sha256_bytes
from morai_ddv2.cli.preprocess_bags import SourceTreeState, preprocess_bags


ROOT = Path(__file__).resolve().parents[2]
DOMAIN_PATH = ROOT / "tests/fixtures/golden/resolved-domain.json"
SENSOR_PRESET_PATH = ROOT / "sensor_data/0725demo_ros.json"

EXPECTED_PROVENANCE_DATASET_FIELDS = (
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

EXPECTED_HDF5_DATASETS = {
    "camera_rgb": (np.dtype(np.uint8), (256, 1024, 3)),
    "lidar_count": (np.dtype(np.uint8), (256, 256)),
    "status": (np.dtype(np.float32), (8,)),
    "trajectory": (np.dtype(np.float32), (8, 3)),
    "timestamp_ns": (np.dtype(np.int64), ()),
    "grid_receipt_ns": (np.dtype(np.int64), ()),
    "sensor_source_header_stamp_ns": (np.dtype(np.int64), (4,)),
    "sensor_source_receipt_ns": (np.dtype(np.int64), (4,)),
    "sensor_source_record_index": (np.dtype(np.int64), (4,)),
    "side_camera_header_offset_ns": (np.dtype(np.int64), (2,)),
    "sensor_receipt_offset_ns": (np.dtype(np.int64), (3,)),
    "trajectory_anchor_tf_header_ns": (np.dtype(np.int64), ()),
    "trajectory_anchor_tf_receipt_ns": (np.dtype(np.int64), ()),
    "trajectory_anchor_tf_record_index": (np.dtype(np.int64), ()),
    "trajectory_anchor_tf_clock_epoch_index": (np.dtype(np.int64), ()),
    "ego_bracket_clock_epoch_index": (np.dtype(np.int64), ()),
    "ego_bracket_header_stamp_ns": (np.dtype(np.int64), (2,)),
    "ego_bracket_receipt_ns": (np.dtype(np.int64), (2,)),
    "ego_bracket_record_index": (np.dtype(np.int64), (2,)),
    "ego_interp_weight": (np.dtype(np.float32), ()),
    "current_pose_bracket_header_stamp_ns": (np.dtype(np.int64), (2,)),
    "current_pose_bracket_receipt_ns": (np.dtype(np.int64), (2,)),
    "current_pose_bracket_record_index": (np.dtype(np.int64), (2,)),
    "current_pose_interp_weight": (np.dtype(np.float32), ()),
    "future_pose_bracket_header_stamp_ns": (np.dtype(np.int64), (8, 2)),
    "future_pose_bracket_receipt_ns": (np.dtype(np.int64), (8, 2)),
    "future_pose_bracket_record_index": (np.dtype(np.int64), (8, 2)),
    "future_pose_interp_weight": (np.dtype(np.float32), (8,)),
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    REALBAG = pytest.mark.realbag


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_state() -> SourceTreeState:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return SourceTreeState(commit, False, None)


def _plain_hdf5_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.tolist()


def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _assert_source_bracket(
    *,
    header: Sequence[int],
    receipt: Sequence[int],
    record_index: Sequence[int],
    target_ns: int,
) -> None:
    assert len(header) == len(receipt) == len(record_index) == 2
    assert header[0] <= target_ns <= header[1]
    assert header[0] <= header[1]
    assert receipt[0] <= receipt[1]
    assert record_index[0] <= record_index[1]
    assert header[1] - header[0] <= 100_000_000
    if header[0] == header[1]:
        assert receipt[0] == receipt[1]
        assert record_index[0] == record_index[1]


def _assert_interpolation_bracket(
    *,
    header: Sequence[int],
    receipt: Sequence[int],
    record_index: Sequence[int],
    target_ns: int,
    weight: float,
) -> None:
    _assert_source_bracket(
        header=header,
        receipt=receipt,
        record_index=record_index,
        target_ns=target_ns,
    )
    assert 0.0 <= weight <= 1.0
    if header[0] == header[1]:
        assert receipt[0] == receipt[1]
        assert record_index[0] == record_index[1]
        assert weight == 0.0 and not bool(np.signbit(weight))
    else:
        expected = float(np.float32((target_ns - header[0]) / (header[1] - header[0])))
        assert weight == expected


def _command_rule_sha256(domain: Mapping[str, Any]) -> str:
    command = domain["command"]
    rule = {
        "index_mapping": command["index_mapping"],
        "thresholds": command["thresholds"],
        "ordered_rules": command["ordered_rules"],
    }
    return sha256_bytes(canonical_json_bytes(rule))


@REALBAG
@pytest.mark.filterwarnings(
    "ignore:the imp module is deprecated in favour of importlib:DeprecationWarning"
)
def test_real_bag_smoke_publishes_exact_cache_and_preserves_source(
    tmp_path: Path,
) -> None:
    raw = os.environ.get("MORAI_REAL_BAG")
    if raw is None:
        pytest.skip("MORAI_REAL_BAG is unset")
    bag_path = Path(raw)
    before = (bag_path.stat().st_size, bag_path.stat().st_mtime_ns, _sha256(bag_path))
    cache_root = tmp_path / "cache"
    domain = json.loads(DOMAIN_PATH.read_text(encoding="utf-8"))
    domain_id = domain["domain_id"]
    resolved_domain_sha256 = domain["resolved_domain_sha256"]

    result = preprocess_bags(
        bag_root=bag_path.parent,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=cache_root,
        source_probe=_source_state,
        discoverer=lambda _root: (bag_path,),
    )

    assert result.success and result.manifest is not None
    manifest_path = cache_root / result.manifest.path
    assert _sha256(manifest_path) == result.manifest.sha256
    verified = load_verified_manifest(
        path=manifest_path,
        sensor_preset_path=SENSOR_PRESET_PATH,
        expected_domain_id=domain_id,
        expected_resolved_domain_sha256=resolved_domain_sha256,
        production=True,
    )
    assert verified.manifest_sha256 == result.manifest.sha256

    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    dataset = records[0]
    samples = [item for item in records if item["record_type"] == "sample"]
    bags = [item for item in records if item["record_type"] == "bag"]
    assert samples
    eligible = next(item for item in bags if item["disposition"] == "ELIGIBLE")
    assert len([item for item in bags if item["disposition"] == "ELIGIBLE"]) == 1
    assert eligible["source_bag_sha256"] == before[2]
    assert eligible["sample_count"] == len(samples) == len(verified.sample_records)
    assert eligible["sidecar_record_count"] == len(samples)
    assert PROVENANCE_DATASET_FIELDS == EXPECTED_PROVENANCE_DATASET_FIELDS
    assert len(PROVENANCE_DATASET_FIELDS) == 24
    assert all(
        tuple(name for name in PROVENANCE_DATASET_FIELDS if name in item)
        == EXPECTED_PROVENANCE_DATASET_FIELDS
        for item in samples
    )

    front_headers = [item["sensor_source_header_stamp_ns"][0] for item in samples]
    front_receipts = [item["sensor_source_receipt_ns"][0] for item in samples]
    front_indices = [item["sensor_source_record_index"][0] for item in samples]
    grid_receipts = [item["grid_receipt_ns"] for item in samples]
    assert _strictly_increasing(front_headers)
    assert _strictly_increasing(front_receipts)
    assert _strictly_increasing(grid_receipts)
    assert len(front_indices) == len(set(front_indices))

    sidecar = cache_root / eligible["sidecar_path"]
    assert _sha256(sidecar) == eligible["sidecar_sha256"]
    sidecar_records = [
        json.loads(line)
        for line in sidecar.read_text(encoding="utf-8").splitlines()
    ]
    assert len(sidecar_records) == len(samples)
    assert all(tuple(item) == COMMAND_RECORD_FIELDS for item in sidecar_records)
    assert _strictly_increasing([item["timestamp_ns"] for item in sidecar_records])
    expected_command_rule_sha256 = _command_rule_sha256(domain)

    hdf5 = cache_root / eligible["hdf5_path"]
    assert _sha256(hdf5) == eligible["hdf5_sha256"]
    with h5py.File(str(hdf5), "r") as handle:
        assert tuple(EXPECTED_HDF5_DATASETS) == CACHE_DATASET_NAMES
        assert set(handle) == set(EXPECTED_HDF5_DATASETS)
        for name, (dtype, trailing_shape) in EXPECTED_HDF5_DATASETS.items():
            assert handle[name].dtype == dtype, name
            assert handle[name].shape == (len(samples),) + trailing_shape, name
        expected_attributes = {
            "cache_schema_version": 2,
            "clock_policy_version": 1,
            "domain_id": domain_id,
            "sensor_preset_sha256": dataset["sensor_preset_sha256"],
            "resolved_domain_sha256": resolved_domain_sha256,
            "source_bag_sha256": eligible["source_bag_sha256"],
            "source_code_commit": dataset["source_code_commit"],
        }
        assert set(handle.attrs) == set(expected_attributes)
        for name, expected in expected_attributes.items():
            assert handle.attrs[name] == expected, name

        for row_index, (sample, command_record) in enumerate(
            zip(samples, sidecar_records)
        ):
            assert sample["hdf5_row_index"] == row_index
            assert sample["sidecar_row_index"] == row_index
            for name in EXPECTED_PROVENANCE_DATASET_FIELDS:
                assert _plain_hdf5_value(handle[name][row_index]) == sample[name], name

            camera = np.asarray(handle["camera_rgb"][row_index])
            lidar = np.asarray(handle["lidar_count"][row_index])
            status = np.asarray(handle["status"][row_index])
            trajectory = np.asarray(handle["trajectory"][row_index])
            future_weights = np.asarray(handle["future_pose_interp_weight"][row_index])
            assert int(camera.min()) >= 0 and int(camera.max()) <= 255
            assert int(lidar.min()) >= 0 and int(lidar.max()) <= 5
            assert np.isfinite(status).all()
            assert np.isfinite(trajectory).all()
            assert np.isfinite(future_weights).all()
            assert bool(np.all((0.0 <= future_weights) & (future_weights <= 1.0)))
            assert bool(np.all(trajectory[:, 2] >= -np.pi))
            assert bool(np.all(trajectory[:, 2] < np.pi))

            headers = sample["sensor_source_header_stamp_ns"]
            receipts = sample["sensor_source_receipt_ns"]
            indices = sample["sensor_source_record_index"]
            assert len(headers) == len(receipts) == len(indices) == 4
            assert sample["timestamp_ns"] == headers[0]
            assert sample["side_camera_header_offset_ns"] == [
                headers[1] - headers[0],
                headers[2] - headers[0],
            ]
            assert sample["sensor_receipt_offset_ns"] == [
                receipts[1] - receipts[0],
                receipts[2] - receipts[0],
                receipts[3] - receipts[0],
            ]
            assert all(abs(value) <= 30_000_000 for value in sample[
                "side_camera_header_offset_ns"
            ])
            assert all(abs(value) <= 100_000_000 for value in sample[
                "sensor_receipt_offset_ns"
            ][:2])
            assert abs(sample["sensor_receipt_offset_ns"][2]) <= 45_000_000
            assert abs(sample["grid_receipt_ns"] - receipts[0]) <= 50_000_000

            anchor_header = sample["trajectory_anchor_tf_header_ns"]
            anchor_receipt = sample["trajectory_anchor_tf_receipt_ns"]
            anchor_index = sample["trajectory_anchor_tf_record_index"]
            assert sample["trajectory_anchor_tf_clock_epoch_index"] >= 0
            assert sample["ego_bracket_clock_epoch_index"] >= 0
            assert abs(anchor_receipt - receipts[0]) <= 50_000_000
            assert sample["current_pose_bracket_header_stamp_ns"] == [
                anchor_header,
                anchor_header,
            ]
            assert sample["current_pose_bracket_receipt_ns"] == [
                anchor_receipt,
                anchor_receipt,
            ]
            assert sample["current_pose_bracket_record_index"] == [
                anchor_index,
                anchor_index,
            ]
            assert sample["current_pose_interp_weight"] == 0.0

            _assert_interpolation_bracket(
                header=sample["ego_bracket_header_stamp_ns"],
                receipt=sample["ego_bracket_receipt_ns"],
                record_index=sample["ego_bracket_record_index"],
                target_ns=anchor_header,
                weight=sample["ego_interp_weight"],
            )
            for future_index, (
                future_header,
                future_receipt,
                future_record_index,
                future_weight,
            ) in enumerate(
                zip(
                    sample["future_pose_bracket_header_stamp_ns"],
                    sample["future_pose_bracket_receipt_ns"],
                    sample["future_pose_bracket_record_index"],
                    sample["future_pose_interp_weight"],
                )
            ):
                _assert_interpolation_bracket(
                    header=future_header,
                    receipt=future_receipt,
                    record_index=future_record_index,
                    target_ns=anchor_header + (future_index + 1) * 500_000_000,
                    weight=future_weight,
                )
                assert all(
                    (receipt_ns, record_index_value) >= (anchor_receipt, anchor_index)
                    for receipt_ns, record_index_value in zip(
                        future_receipt, future_record_index
                    )
                )

            expected_one_hot = np.zeros(4, dtype=np.float32)
            expected_one_hot[sample["command_index"]] = np.float32(1.0)
            assert np.array_equal(status[:4], expected_one_hot)
            assert np.isfinite(status[4:]).all()
            assert tuple(command_record) == COMMAND_RECORD_FIELDS
            assert command_record == {
                "schema_version": 2,
                "domain_id": domain_id,
                "resolved_domain_sha256": resolved_domain_sha256,
                "sensor_preset_sha256": dataset["sensor_preset_sha256"],
                "source_bag_sha256": eligible["source_bag_sha256"],
                "command_rule_sha256": expected_command_rule_sha256,
                "timestamp_ns": sample["timestamp_ns"],
                "grid_receipt_ns": sample["grid_receipt_ns"],
                "front_receipt_ns": receipts[0],
                "front_record_index": indices[0],
                "command_index": sample["command_index"],
                "command_name": sample["command_name"],
                "provenance": "future_trajectory_oracle",
            }

    for sample in samples:
        model_sample = load_tensor_sample(verified, sample["sample_id"])
        assert model_sample.command_name == sample["command_name"]
        assert model_sample.command_index == sample["command_index"]
        assert model_sample.camera_feature.dtype == np.dtype(np.float32)
        assert model_sample.camera_feature.shape == (3, 256, 1024)
        assert model_sample.lidar_feature.dtype == np.dtype(np.float32)
        assert model_sample.lidar_feature.shape == (1, 256, 256)
        assert model_sample.status_feature.dtype == np.dtype(np.float32)
        assert model_sample.status_feature.shape == (8,)
        assert model_sample.trajectory.dtype == np.dtype(np.float32)
        assert model_sample.trajectory.shape == (8, 3)
        for tensor in (
            model_sample.camera_feature,
            model_sample.lidar_feature,
            model_sample.status_feature,
            model_sample.trajectory,
        ):
            assert np.isfinite(tensor).all()
        assert 0.0 <= float(model_sample.camera_feature.min())
        assert float(model_sample.camera_feature.max()) <= 1.0
        assert 0.0 <= float(model_sample.lidar_feature.min())
        assert float(model_sample.lidar_feature.max()) <= 1.0

    diagnostic_path = cache_root / eligible["tf_ego_diagnostic_report_path"]
    assert _sha256(diagnostic_path) == eligible["tf_ego_diagnostic_report_sha256"]
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert tuple(diagnostic) == DIAGNOSTIC_REPORT_FIELDS
    assert diagnostic["domain_id"] == domain_id
    assert diagnostic["resolved_domain_sha256"] == resolved_domain_sha256
    assert diagnostic["sensor_preset_sha256"] == dataset["sensor_preset_sha256"]
    assert diagnostic["relative_bag_path"] == eligible["relative_bag_path"]
    assert diagnostic["source_bag_sha256"] == eligible["source_bag_sha256"]
    counter_pairs = (
        ("raw_ego_record_count", "tf_ego_raw_record_count"),
        ("epoch_local_comparable_count", "tf_ego_epoch_local_comparable_count"),
        ("no_epoch_local_bracket_count", "tf_ego_no_epoch_local_bracket_count"),
        ("ambiguous_epoch_selection_count", "tf_ego_ambiguous_epoch_selection_count"),
        (
            "cross_epoch_bracket_prevented_count",
            "tf_ego_cross_epoch_bracket_prevented_count",
        ),
    )
    assert all(
        diagnostic[diagnostic_name] == eligible[bag_name]
        for diagnostic_name, bag_name in counter_pairs
    )
    provenance = diagnostic["provenance"]
    assert len(provenance) == diagnostic["raw_ego_record_count"]
    assert diagnostic["comparable_count"] == diagnostic[
        "epoch_local_comparable_count"
    ]
    assert diagnostic["excluded_count"] == (
        diagnostic["raw_ego_record_count"] - diagnostic["comparable_count"]
    )
    assert diagnostic["raw_ego_record_count"] == sum(
        diagnostic[name]
        for name in (
            "epoch_local_comparable_count",
            "no_epoch_local_bracket_count",
            "ambiguous_epoch_selection_count",
            "cross_epoch_bracket_prevented_count",
        )
    )
    reasons = [item["exclusion_reason"] for item in provenance]
    assert reasons.count(None) == diagnostic["epoch_local_comparable_count"]
    assert reasons.count("NO_EPOCH_LOCAL_BRACKET") == diagnostic[
        "no_epoch_local_bracket_count"
    ]
    assert reasons.count("AMBIGUOUS_EPOCH_SELECTION") == diagnostic[
        "ambiguous_epoch_selection_count"
    ]
    assert reasons.count("CROSS_EPOCH_BRACKET_PREVENTED") == diagnostic[
        "cross_epoch_bracket_prevented_count"
    ]
    diagnostic_indices = [item["ego_stable_record_index"] for item in provenance]
    assert diagnostic_indices == sorted(diagnostic_indices)
    assert len(diagnostic_indices) == len(set(diagnostic_indices))
    assert diagnostic["comparable_count"] >= 10
    assert 0.0 <= diagnostic["position_p99_m"] <= 0.05
    assert 0.0 <= diagnostic["heading_p99_deg"] <= 0.5
    assert all(
        np.isfinite(diagnostic[name])
        for name in (
            "position_p99_m",
            "position_max_m",
            "heading_p99_deg",
            "heading_max_deg",
        )
    )
    for item in provenance:
        assert tuple(item) == DIAGNOSTIC_PROVENANCE_FIELDS
        assert item["selected_tf_clock_epoch_index"] >= 0
        assert item["selected_tf_anchor_record_index"] >= 0
        if item["exclusion_reason"] is None:
            _assert_source_bracket(
                header=item["tf_bracket_header_stamp_ns"],
                receipt=item["tf_bracket_receipt_ns"],
                record_index=item["tf_bracket_record_index"],
                target_ns=item["ego_header_stamp_ns"],
            )
        else:
            assert item["tf_bracket_header_stamp_ns"] is None
            assert item["tf_bracket_receipt_ns"] is None
            assert item["tf_bracket_record_index"] is None

    assert (bag_path.stat().st_size, bag_path.stat().st_mtime_ns, _sha256(bag_path)) == before
