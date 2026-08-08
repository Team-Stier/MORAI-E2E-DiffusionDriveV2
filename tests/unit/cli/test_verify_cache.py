from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from morai_ddv2.cli.verify_cache import verify_cache
from morai_ddv2.errors import IntegrityError
from morai_ddv2.cache.hdf5 import HDF5_DATASETS
from tests.unit.cache.test_manifest import _publish_fixture


ROOT = Path(__file__).resolve().parents[3]
PRESET = ROOT / "sensor_data/0725demo_ros.json"


def _hash_tree(root: Path):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _manifest(cache_root: Path) -> Path:
    return next((cache_root / "manifests").glob("*/*/manifest.jsonl"))


def _rewrite_hdf5_hash(manifest: Path, digest: str) -> None:
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    eligible = next(
        item
        for item in records
        if item["record_type"] == "bag" and item["disposition"] == "ELIGIBLE"
    )
    eligible["hdf5_sha256"] = digest
    manifest.write_bytes(
        b"".join(
            json.dumps(
                item,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            for item in records
        )
    )


def _repack_hdf5_with_reverse_object_order(path: Path) -> None:
    replacement = path.with_suffix(".repacked")
    with h5py.File(str(path), "r") as source, h5py.File(
        str(replacement), "w"
    ) as target:
        for name in reversed(tuple(source.attrs)):
            target.attrs[name] = source.attrs[name]
        for name in reversed(tuple(HDF5_DATASETS)):
            original = source[name]
            dataset = target.create_dataset(
                name,
                shape=original.shape,
                maxshape=original.maxshape,
                chunks=original.chunks,
                dtype=original.dtype,
            )
            dataset[...] = original[...]
    os.replace(str(replacement), str(path))


def test_verification_is_read_only_and_publishes_complete_aggregates(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    _publish_fixture(cache_root)
    manifest = _manifest(cache_root)
    before = _hash_tree(cache_root)
    report_path = tmp_path / "verification.json"

    report = verify_cache(
        manifest_path=manifest,
        sensor_preset_path=PRESET,
        report_path=report_path,
    )

    assert _hash_tree(cache_root) == before
    assert report["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    snapshot = cache_root / report["resolved_domain_path"]
    assert report["resolved_domain_file_sha256"] == hashlib.sha256(
        snapshot.read_bytes()
    ).hexdigest()
    assert report["domain_id"] == "morai-ddv2-v1-oracle-command"
    assert report["cache_schema_version"] == 2
    assert report["clock_policy_version"] == 1
    assert report["totals"]["bag_count"] == 3
    assert report["totals"]["sample_count"] == 2
    assert report["totals"]["strict_front_identity_verified_count"] == 2
    assert report["totals"]["command_counts"] == {
        "LEFT": 1,
        "STRAIGHT": 0,
        "RIGHT": 1,
        "UNKNOWN": 0,
    }
    assert report["totals"]["camera_uint8"] == {"min": 1, "max": 2}
    assert report["totals"]["lidar_count_uint8"]["max"] == 2
    assert report["totals"]["side_camera_header_offset_abs_ns"]["count"] == 4
    eligible = next(item for item in report["bags"] if item["disposition"] == "ELIGIBLE")
    assert eligible["tf_ego_full_stream"]["raw_record_count"] == 13
    assert eligible["tf_ego_full_stream"]["epoch_local_comparable_count"] == 10
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert not Path(str(report_path) + ".partial").exists()


def test_independently_verified_semantic_comparison_accepts_equal_caches(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _publish_fixture(first)
    _publish_fixture(second)

    report = verify_cache(
        manifest_path=_manifest(first),
        compare_manifest_path=_manifest(second),
        sensor_preset_path=PRESET,
        report_path=tmp_path / "compare.json",
    )

    assert report["comparison"]["semantic_equal"] is True
    assert report["comparison"]["compared_manifest_sha256"] == hashlib.sha256(
        _manifest(second).read_bytes()
    ).hexdigest()


def test_comparison_allows_only_hdf5_container_hash_drift_when_values_match(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _publish_fixture(first)
    _publish_fixture(second)
    second_hdf5 = next((second / "bags").glob("*/*/*.h5"))
    original_hash = hashlib.sha256(second_hdf5.read_bytes()).hexdigest()
    _repack_hdf5_with_reverse_object_order(second_hdf5)
    repacked_hash = hashlib.sha256(second_hdf5.read_bytes()).hexdigest()
    assert repacked_hash != original_hash
    _rewrite_hdf5_hash(_manifest(second), repacked_hash)

    report = verify_cache(
        manifest_path=_manifest(first),
        compare_manifest_path=_manifest(second),
        sensor_preset_path=PRESET,
        report_path=tmp_path / "repacked-compare.json",
    )

    assert report["comparison"]["semantic_equal"] is True
    assert report["comparison"]["hdf5_container_hash_difference_count"] == 1


def test_semantically_changed_but_independently_valid_hdf5_is_rejected(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _publish_fixture(first)
    _publish_fixture(second)
    second_hdf5 = next((second / "bags").glob("*/*/*.h5"))
    with h5py.File(str(second_hdf5), "r+") as handle:
        handle["camera_rgb"][0, 0, 0, 0] = np.uint8(99)
    _rewrite_hdf5_hash(
        _manifest(second), hashlib.sha256(second_hdf5.read_bytes()).hexdigest()
    )
    report_path = tmp_path / "semantic-difference-must-not-exist.json"

    with pytest.raises(IntegrityError, match="values differ"):
        verify_cache(
            manifest_path=_manifest(first),
            compare_manifest_path=_manifest(second),
            sensor_preset_path=PRESET,
            report_path=report_path,
        )

    assert not report_path.exists()


def test_comparison_failure_publishes_no_success_report(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _publish_fixture(first)
    _publish_fixture(second)
    sidecar = next((second / "commands").glob("*/*/*.jsonl"))
    sidecar.write_bytes(sidecar.read_bytes().replace(b'"LEFT"', b'"LEFX"', 1))
    report_path = tmp_path / "must-not-exist.json"

    with pytest.raises(IntegrityError):
        verify_cache(
            manifest_path=_manifest(first),
            compare_manifest_path=_manifest(second),
            sensor_preset_path=PRESET,
            report_path=report_path,
        )

    assert not report_path.exists()


def test_report_symlink_parent_cannot_redirect_publication_into_cache(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    _publish_fixture(cache_root)
    before = _hash_tree(cache_root)
    alias = tmp_path / "outside-alias"
    alias.symlink_to(cache_root, target_is_directory=True)

    with pytest.raises(IntegrityError, match="report"):
        verify_cache(
            manifest_path=_manifest(cache_root),
            sensor_preset_path=PRESET,
            report_path=alias / "redirected-report.json",
        )

    assert _hash_tree(cache_root) == before
    assert not (cache_root / "redirected-report.json").exists()


def test_report_missing_parent_is_created_from_a_pinned_existing_ancestor(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    _publish_fixture(cache_root)
    report_path = tmp_path / "new" / "nested" / "verification.json"

    report = verify_cache(
        manifest_path=_manifest(cache_root),
        sensor_preset_path=PRESET,
        report_path=report_path,
    )

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert not Path(str(report_path) + ".partial").exists()


def test_missing_report_parent_inside_cache_is_not_created(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    _publish_fixture(cache_root)
    before = _hash_tree(cache_root)
    report_path = cache_root / "new" / "verification.json"

    with pytest.raises(IntegrityError, match="outside"):
        verify_cache(
            manifest_path=_manifest(cache_root),
            sensor_preset_path=PRESET,
            report_path=report_path,
        )

    assert _hash_tree(cache_root) == before
    assert not report_path.parent.exists()


@pytest.mark.parametrize(
    "collision", ["final", "partial", "final_hardlink", "partial_hardlink"]
)
def test_verification_report_cannot_replace_sensor_preset(
    tmp_path: Path, collision: str
) -> None:
    cache_root = tmp_path / "cache"
    _publish_fixture(cache_root)
    report_path = tmp_path / "verification.json"
    if collision == "final":
        sensor_preset = report_path
    elif collision == "partial":
        sensor_preset = Path(str(report_path) + ".partial")
    else:
        sensor_preset = tmp_path / "sensor-preset.json"
    sensor_preset.write_bytes(PRESET.read_bytes())
    if collision == "final_hardlink":
        os.link(str(sensor_preset), str(report_path))
    elif collision == "partial_hardlink":
        os.link(str(sensor_preset), str(Path(str(report_path) + ".partial")))
    before = sensor_preset.read_bytes()

    with pytest.raises(IntegrityError, match="report|preset|input"):
        verify_cache(
            manifest_path=_manifest(cache_root),
            sensor_preset_path=sensor_preset,
            report_path=report_path,
        )

    assert sensor_preset.read_bytes() == before
    if collision in ("partial", "partial_hardlink"):
        assert not report_path.exists()


def test_verifier_hashes_orphan_cache_files_without_path_based_open(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache"
    _publish_fixture(cache_root)
    orphan = cache_root / "orphan-final.bin"
    orphan.write_bytes(b"orphan final ignored by manifest")
    real_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == orphan:
            raise AssertionError("cache file reopened by mutable path")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    report = verify_cache(
        manifest_path=_manifest(cache_root),
        sensor_preset_path=PRESET,
        report_path=tmp_path / "descriptor-hash-report.json",
    )

    assert report["verified_cache_file_count"] >= 1


def test_verifier_rereads_diagnostics_through_descriptor_bound_cache_api(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache"
    _publish_fixture(cache_root)
    manifest_records = [
        json.loads(line) for line in _manifest(cache_root).read_text().splitlines()
    ]
    diagnostic_relative = next(
        item["tf_ego_diagnostic_report_path"]
        for item in manifest_records
        if item["record_type"] == "bag"
        and item["tf_ego_diagnostic_report_path"] is not None
    )
    diagnostic = cache_root / diagnostic_relative
    real_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        if path == diagnostic:
            raise AssertionError("diagnostic reopened by mutable path")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    report = verify_cache(
        manifest_path=_manifest(cache_root),
        sensor_preset_path=PRESET,
        report_path=tmp_path / "descriptor-diagnostic-report.json",
    )

    assert report["totals"]["sample_count"] == 2


def test_cache_change_at_final_gate_leaves_no_success_report_or_partial(
    tmp_path: Path, monkeypatch
) -> None:
    import morai_ddv2.cli.verify_cache as module

    cache_root = tmp_path / "cache"
    _publish_fixture(cache_root)
    report_path = tmp_path / "must-not-survive-final-gate.json"
    real_hash_tree = module._hash_cache_tree
    calls = {"count": 0}

    def changing_hash_tree(root):
        calls["count"] += 1
        result = real_hash_tree(root)
        if calls["count"] >= 3:
            result = dict(result)
            result["concurrent-change"] = ("0" * 64, (0,))
        return result

    monkeypatch.setattr(module, "_hash_cache_tree", changing_hash_tree)

    with pytest.raises(IntegrityError, match="cache changed"):
        verify_cache(
            manifest_path=_manifest(cache_root),
            sensor_preset_path=PRESET,
            report_path=report_path,
        )

    assert not report_path.exists()
    assert not Path(str(report_path) + ".partial").exists()
