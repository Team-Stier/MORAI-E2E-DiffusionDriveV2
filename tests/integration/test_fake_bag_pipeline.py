from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import h5py
import numpy as np

from morai_ddv2.cache import CACHE_DATASET_NAMES
from morai_ddv2.cli.preprocess_bags import SourceTreeState, preprocess_bags
from morai_ddv2.config import load_resolved_domain
from morai_ddv2.preprocess.planner import build_sample_plan
from morai_ddv2.ros1 import BagDisposition, index_bag
from morai_ddv2.ros1 import bag_reader
from tests.unit.cache.test_hdf5 import SOURCE_COMMIT
from tests.unit.preprocess.test_materialize import _success_events
from tests.unit.ros1.test_bag_reader import (
    FakeBag,
    _camera,
    _ego,
    _lidar,
    _ns,
    _transform,
    _types,
)


ROOT = Path(__file__).resolve().parents[2]
DOMAIN_PATH = ROOT / "tests/fixtures/golden/resolved-domain.json"


def _jpeg(width: int, height: int) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg", np.zeros((height, width, 3), dtype=np.uint8)
    )
    assert success
    return encoded.tobytes()


def _eligible_records(domain):
    sensors = domain.document["sensors"]
    contract = domain.document["ros_contract"]
    tf_topic = contract["policy_topics"]["transforms"]
    ego_topic = contract["policy_topics"]["ego_state"]
    header_origin_ns = 10_000_000_000
    receipt_origin_ns = 9_000_000_000_000
    records = []

    for index in range(10):
        header_ns = header_origin_ns + index * 500_000_000
        receipt_ns = receipt_origin_ns + index * 500_000_000
        transforms = [
            _transform("/map", "/base_link", header_ns, (float(index), 0.0, 0.0))
        ]
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
        lidar = sensors["lidar"]
        transforms.append(
            _transform(
                "/base_link",
                "/" + lidar["frame_id"],
                header_ns,
                tuple(lidar["translation_base_from_lidar_m"]),
                tuple(lidar["quaternion_base_from_lidar_xyzw"]),
            )
        )
        records.append(
            (
                tf_topic,
                _ns(_type="tf2_msgs/TFMessage", transforms=transforms),
                receipt_ns,
            )
        )

        if index == 0:
            for offset, camera in enumerate(sensors["cameras"]):
                width, height = camera["resolution_wh"]
                records.append(
                    (
                        camera["topic"],
                        _camera(
                            header_ns,
                            camera["frame_id"],
                            _jpeg(width, height),
                        ),
                        receipt_ns + offset,
                    )
                )
            records.append(
                (
                    lidar["topic"],
                    _lidar(header_ns, lidar["frame_id"]),
                    receipt_ns + 3,
                )
            )

        records.append((ego_topic, _ego(header_ns, float(index)), receipt_ns + 4))
    return tuple(records)


def _canonical_outputs(cache_root: Path) -> Dict[str, bytes]:
    expected_counts = {
        "domains": 1,
        "commands": 1,
        "diagnostics": 1,
        "manifests": 1,
    }
    outputs = {}
    for directory, expected_count in expected_counts.items():
        files = tuple(
            sorted(path for path in (cache_root / directory).rglob("*") if path.is_file())
        )
        assert len(files) == expected_count
        for path in files:
            outputs[path.relative_to(cache_root).as_posix()] = path.read_bytes()
    return outputs


def _clean_source() -> SourceTreeState:
    return SourceTreeState(SOURCE_COMMIT, False, None)


def test_fake_reader_runs_actual_plan_materialize_and_reproduces_cache(
    tmp_path: Path, monkeypatch
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    bag_path = bag_root / "drive.bag"
    bag_path.write_bytes(b"fake bag bytes; ROS1 backend is injected")

    domain = load_resolved_domain(DOMAIN_PATH)
    fake_bag = FakeBag(_eligible_records(domain), _types(domain))
    monkeypatch.setattr(bag_reader, "_open_bag", lambda _path: fake_bag)

    indexed = index_bag(bag_path, domain)
    assert indexed.inventory.disposition is BagDisposition.ELIGIBLE
    assert indexed.inventory.tf_ego_diagnostic is not None
    assert indexed.inventory.tf_ego_diagnostic.metric.comparable_count == 10
    expected_plan = build_sample_plan(indexed, domain)
    assert len(expected_plan.samples) == 1
    assert expected_plan.drops == ()

    def fake_payloads(path, selected):
        assert path == bag_path
        assert selected == {
            topic: set(indices)
            for topic, indices in expected_plan.selected_payload_indices.items()
        }
        return iter(_success_events(expected_plan))

    monkeypatch.setattr(
        "morai_ddv2.preprocess.materialize.iter_selected_payloads", fake_payloads
    )
    first = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache-a",
        source_probe=_clean_source,
    )
    second = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache-b",
        source_probe=_clean_source,
    )

    assert first.success and second.success
    assert first.manifest is not None and second.manifest is not None
    first_manifest = tmp_path / "cache-a" / first.manifest.path
    second_manifest = tmp_path / "cache-b" / second.manifest.path
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert _canonical_outputs(tmp_path / "cache-a") == _canonical_outputs(
        tmp_path / "cache-b"
    )
    first_hdf5 = next((tmp_path / "cache-a" / "bags").glob("*/*/*.h5"))
    second_hdf5 = next((tmp_path / "cache-b" / "bags").glob("*/*/*.h5"))
    with h5py.File(str(first_hdf5), "r") as left, h5py.File(
        str(second_hdf5), "r"
    ) as right:
        assert set(left) == set(right) == set(CACHE_DATASET_NAMES)
        assert dict(left.attrs) == dict(right.attrs)
        for name in CACHE_DATASET_NAMES:
            assert left[name].dtype == right[name].dtype
            assert left[name].shape == right[name].shape
            assert np.array_equal(left[name][...], right[name][...]), name
