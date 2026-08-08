from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from morai_ddv2.cache.loader import load_tensor_sample, load_verified_manifest
from morai_ddv2.cache.manifest import DirtyOverride
from morai_ddv2.config import sha256_file
from morai_ddv2.errors import IntegrityError
from tests.unit.cache.test_hdf5 import RELATIVE_BAG_PATH, SOURCE_SHA
from tests.unit.cache.test_manifest import _publish_fixture


PRESET_PATH = Path(__file__).resolve().parents[3] / "sensor_data/0725demo_ros.json"


def _load(cache_root: Path, *, production: bool = True):
    manifest_path = next((cache_root / "manifests").glob("*/*/manifest.jsonl"))
    return load_verified_manifest(
        path=manifest_path,
        sensor_preset_path=PRESET_PATH,
        expected_domain_id="morai-ddv2-v1-oracle-command",
        expected_resolved_domain_sha256=manifest_path.parent.name,
        production=production,
    )


def _rewrite_manifest(path: Path, mutation) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutation(records)
    payload = b"".join(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in records
    )
    path.write_bytes(payload)


def _tampered_hdf5_clone(tmp_path: Path, target: Path) -> Path:
    clone_parent = tmp_path / "same-provenance-tampered-parent"
    clone_parent.mkdir()
    clone = clone_parent / target.name
    shutil.copy2(str(target), str(clone))
    with h5py.File(str(clone), "r+") as handle:
        handle["camera_rgb"][0, 0, 0, :] = np.uint8(255)
    return clone


def _install_atomic_hdf5_open_swap(
    monkeypatch,
    *,
    target: Path,
    replacement: Path,
    skip_h5py_path_opens: int,
):
    real_h5py_file = h5py.File
    real_os_open = os.open
    target_parent = target.parent
    replacement_parent = replacement.parent
    backup_parent = target_parent.parent / "verified-original-parent-during-swap"
    state = {"h5py_path_opens": 0, "attacks": 0}

    def swap_around(open_file):
        os.replace(str(target_parent), str(backup_parent))
        os.replace(str(replacement_parent), str(target_parent))
        try:
            opened = open_file()
        finally:
            os.replace(str(target_parent), str(replacement_parent))
            os.replace(str(backup_parent), str(target_parent))
        state["attacks"] += 1
        return opened

    def swapped_h5py_file(name, *args, **kwargs):
        if isinstance(name, (str, bytes, Path)) and Path(name) == target:
            state["h5py_path_opens"] += 1
            if state["h5py_path_opens"] > skip_h5py_path_opens:
                return swap_around(lambda: real_h5py_file(name, *args, **kwargs))
        return real_h5py_file(name, *args, **kwargs)

    def swapped_os_open(name, flags, mode=0o777, *, dir_fd=None):
        raw_name = os.fspath(name)
        targets_hdf5 = raw_name == str(target) or (
            dir_fd is not None and raw_name == target.name
        )
        if (
            targets_hdf5
            and state["h5py_path_opens"] >= skip_h5py_path_opens
            and state["attacks"] == 0
        ):
            return swap_around(
                lambda: real_os_open(name, flags, mode, dir_fd=dir_fd)
            )
        return real_os_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(h5py, "File", swapped_h5py_file)
    monkeypatch.setattr(os, "open", swapped_os_open)
    return state


def test_verified_loader_converts_only_model_tensors_at_load_time(tmp_path: Path) -> None:
    record = _publish_fixture(tmp_path)

    manifest = _load(tmp_path)
    sample_id = manifest.sample_records[0]["sample_id"]
    sample = load_tensor_sample(manifest, sample_id)

    assert manifest.path == tmp_path / record.path
    assert manifest.manifest_sha256 == sha256_file(manifest.path)
    assert sample.sample_id == sample_id
    assert sample.source_bag_sha256 == SOURCE_SHA
    assert sample.relative_bag_path == RELATIVE_BAG_PATH
    assert sample.camera_feature.shape == (3, 256, 1024)
    assert sample.camera_feature.dtype == np.float32
    assert np.array_equal(
        sample.camera_feature[:, 0, 0],
        np.asarray([1.0 / 255.0] * 3, dtype=np.float32),
    )
    assert sample.lidar_feature.shape == (1, 256, 256)
    assert sample.lidar_feature.dtype == np.float32
    assert sample.lidar_feature[0, 1, 1] == np.float32(0.2)
    assert sample.status_feature.shape == (8,)
    assert sample.status_feature.dtype == np.float32
    assert sample.trajectory.shape == (8, 3)
    assert sample.trajectory.dtype == np.float32


def test_initial_verification_rejects_atomic_hdf5_open_inode_swap(
    tmp_path: Path, monkeypatch
) -> None:
    record = _publish_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in (tmp_path / record.path).read_text(encoding="utf-8").splitlines()
    ]
    bag = next(
        item for item in records if item["record_type"] == "bag" and item["split"]
    )
    target = tmp_path / bag["hdf5_path"]
    replacement = _tampered_hdf5_clone(tmp_path, target)
    state = _install_atomic_hdf5_open_swap(
        monkeypatch,
        target=target,
        replacement=replacement,
        skip_h5py_path_opens=1,
    )

    with pytest.raises(IntegrityError):
        _load(tmp_path)

    assert state["attacks"] == 1
    assert target.is_file()


def test_tensor_load_rejects_atomic_hdf5_open_inode_swap(
    tmp_path: Path, monkeypatch
) -> None:
    _publish_fixture(tmp_path)
    manifest = _load(tmp_path)
    record = manifest.sample_records[0]
    target = tmp_path / record["hdf5_path"]
    replacement = _tampered_hdf5_clone(tmp_path, target)
    state = _install_atomic_hdf5_open_swap(
        monkeypatch,
        target=target,
        replacement=replacement,
        skip_h5py_path_opens=0,
    )

    with pytest.raises(IntegrityError):
        load_tensor_sample(manifest, record["sample_id"])

    assert state["attacks"] == 1
    assert target.is_file()


@pytest.mark.parametrize(
    "ancestor", ["manifests", "domains", "diagnostics", "commands"]
)
def test_loader_rejects_cache_ancestor_symlink(
    tmp_path: Path, ancestor: str
) -> None:
    record = _publish_fixture(tmp_path)
    original = tmp_path / ancestor
    relocated = tmp_path / "relocated-{}".format(ancestor)
    original.rename(relocated)
    original.symlink_to(relocated, target_is_directory=True)
    manifest_path = tmp_path / record.path

    with pytest.raises(IntegrityError):
        load_verified_manifest(
            path=manifest_path,
            sensor_preset_path=PRESET_PATH,
            expected_domain_id="morai-ddv2-v1-oracle-command",
            expected_resolved_domain_sha256=manifest_path.parent.name,
        )


def test_orphan_final_and_partial_files_are_not_dataset_commit_markers(
    tmp_path: Path,
) -> None:
    _publish_fixture(tmp_path)
    orphan = tmp_path / "bags" / "orphan.h5"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"not-hdf5")
    partial = tmp_path / "commands" / "orphan.jsonl.partial"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"not-jsonl")

    manifest = _load(tmp_path)

    assert len(manifest.sample_records) == 2


def test_loader_rejects_sensor_only_bag_with_diagnostic_binding(
    tmp_path: Path,
) -> None:
    record = _publish_fixture(tmp_path)
    path = tmp_path / record.path

    def mutate(records):
        eligible = next(
            item
            for item in records
            if item["record_type"] == "bag" and item["disposition"] == "ELIGIBLE"
        )
        excluded = next(
            item
            for item in records
            if item["record_type"] == "bag"
            and item["disposition"] == "SENSOR_ONLY_EXCLUDED"
        )
        for name in (
            "tf_ego_diagnostic_report_path",
            "tf_ego_diagnostic_report_sha256",
            "tf_ego_raw_record_count",
            "tf_ego_epoch_local_comparable_count",
            "tf_ego_no_epoch_local_bracket_count",
            "tf_ego_ambiguous_epoch_selection_count",
            "tf_ego_cross_epoch_bracket_prevented_count",
        ):
            excluded[name] = eligible[name]

    _rewrite_manifest(path, mutate)

    with pytest.raises(IntegrityError, match="sensor-only"):
        _load(tmp_path)


@pytest.mark.parametrize("artifact", ["hdf5", "sidecar"])
def test_missing_or_hash_mismatched_referenced_artifact_fails(
    tmp_path: Path, artifact: str
) -> None:
    record = _publish_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in (tmp_path / record.path).read_text(encoding="utf-8").splitlines()
    ]
    bag = next(item for item in records if item["record_type"] == "bag" and item["split"])
    target = tmp_path / bag["{}_path".format(artifact)]
    if artifact == "hdf5":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(IntegrityError):
        _load(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda records: records[0].__setitem__("schema_version", 1),
        lambda records: records[0].__setitem__("unknown", True),
        lambda records: records[0].pop("sensor_preset_sha256"),
        lambda records: records[0].__setitem__("resolved_domain_path", "domains/wrong.json"),
        lambda records: records[0]["validation_relative_bag_paths"].clear(),
        lambda records: next(
            item for item in records if item["record_type"] == "bag" and item["split"]
        ).__setitem__("hdf5_sha256", "0" * 64),
        lambda records: next(
            item for item in records if item["record_type"] == "sample"
        ).__setitem__("side_camera_header_offset_ns", [999, 999]),
        lambda records: next(
            item for item in records if item["record_type"] == "sample"
        ).__setitem__("trajectory_anchor_tf_record_index", 999),
    ],
)
def test_loader_rejects_schema_identity_split_and_provenance_drift(
    tmp_path: Path, mutation
) -> None:
    record = _publish_fixture(tmp_path)
    path = tmp_path / record.path
    _rewrite_manifest(path, mutation)

    with pytest.raises(IntegrityError):
        _load(tmp_path)


@pytest.mark.parametrize("failure", ["timestamp", "front_index", "sample_id"])
def test_loader_rejects_nonincreasing_identity_or_duplicate_sample_id(
    tmp_path: Path, failure: str
) -> None:
    record = _publish_fixture(tmp_path)
    path = tmp_path / record.path

    def mutate(records):
        samples = [item for item in records if item["record_type"] == "sample"]
        if failure == "timestamp":
            samples[1]["timestamp_ns"] = samples[0]["timestamp_ns"]
        elif failure == "front_index":
            samples[1]["sensor_source_record_index"][0] = samples[0][
                "sensor_source_record_index"
            ][0]
        else:
            samples[1]["sample_id"] = samples[0]["sample_id"]

    _rewrite_manifest(path, mutate)

    with pytest.raises(IntegrityError):
        _load(tmp_path)


def test_dirty_override_is_rejected_in_production_but_loads_in_development(
    tmp_path: Path,
) -> None:
    _publish_fixture(
        tmp_path,
        dirty_override=DirtyOverride(
            allow_dirty_development=True,
            source_code_commit="b" * 40,
            dirty_diff_sha256="f" * 64,
        ),
    )

    with pytest.raises(IntegrityError, match="dirty"):
        _load(tmp_path, production=True)
    assert len(_load(tmp_path, production=False).sample_records) == 2


def test_loader_rechecks_tensor_contract_after_manifest_verification(
    tmp_path: Path,
) -> None:
    _publish_fixture(tmp_path)
    manifest = _load(tmp_path)
    record = manifest.sample_records[0]
    hdf5_path = tmp_path / record["hdf5_path"]
    with h5py.File(str(hdf5_path), "r+") as handle:
        handle["status"][0, 4] = np.float32(np.nan)

    with pytest.raises(IntegrityError):
        load_tensor_sample(manifest, record["sample_id"])


def test_loader_rejects_wrong_sensor_preset_hash(tmp_path: Path) -> None:
    _publish_fixture(tmp_path)
    wrong = tmp_path / "wrong-preset.json"
    wrong.write_bytes(PRESET_PATH.read_bytes() + b"\n")
    manifest_path = next((tmp_path / "manifests").glob("*/*/manifest.jsonl"))

    with pytest.raises(IntegrityError, match="preset"):
        load_verified_manifest(
            path=manifest_path,
            sensor_preset_path=wrong,
            expected_domain_id="morai-ddv2-v1-oracle-command",
            expected_resolved_domain_sha256=manifest_path.parent.name,
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_bracket",
        "metric_count",
        "below_minimum",
        "position_p99",
        "heading_p99",
    ],
)
def test_loader_rejects_rehashed_but_semantically_invalid_diagnostic_report(
    tmp_path: Path, corruption: str
) -> None:
    record = _publish_fixture(tmp_path)
    manifest_path = tmp_path / record.path
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    bag = next(
        item
        for item in records
        if item["record_type"] == "bag" and item["disposition"] == "ELIGIBLE"
    )
    diagnostic_path = tmp_path / bag["tf_ego_diagnostic_report_path"]
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if corruption == "missing_bracket":
        comparable = next(
            item for item in diagnostic["provenance"] if item["exclusion_reason"] is None
        )
        comparable["tf_bracket_record_index"] = None
    elif corruption == "metric_count":
        diagnostic["comparable_count"] += 1
    elif corruption == "below_minimum":
        comparable_rows = [
            item for item in diagnostic["provenance"] if item["exclusion_reason"] is None
        ]
        diagnostic["provenance"].remove(comparable_rows[-1])
        diagnostic["comparable_count"] -= 1
        diagnostic["epoch_local_comparable_count"] -= 1
        diagnostic["raw_ego_record_count"] -= 1
        bag["tf_ego_epoch_local_comparable_count"] -= 1
        bag["tf_ego_raw_record_count"] -= 1
    elif corruption == "position_p99":
        diagnostic["position_p99_m"] = 0.0510001
        diagnostic["position_max_m"] = 0.0510001
    else:
        diagnostic["heading_p99_deg"] = 0.500001
        diagnostic["heading_max_deg"] = 0.500001
    payload = json.dumps(
        diagnostic,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    diagnostic_path.write_bytes(payload)
    bag["tf_ego_diagnostic_report_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_payload = b"".join(
        json.dumps(
            item,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for item in records
    )
    manifest_path.write_bytes(manifest_payload)

    with pytest.raises(IntegrityError, match="diagnostic"):
        _load(tmp_path)
