from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

from morai_ddv2.atomic import (
    assert_atomic_output_avoids_paths,
    atomic_write_bytes,
)
from morai_ddv2.cache.domain import _open_cache_regular_stream
from morai_ddv2.cache.loader import (
    VerifiedManifest,
    load_tensor_sample,
    load_verified_manifest,
)
from morai_ddv2.config import canonical_json_bytes
from morai_ddv2.errors import IntegrityError, MoraiDDV2Error
from morai_ddv2.preprocess.planner import SampleDropReason


_DOMAIN_ID = "morai-ddv2-v1-oracle-command"
_COMMAND_NAMES = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")


def _manifest_cache_root(path: Path) -> Path:
    absolute = Path(path).absolute()
    try:
        return absolute.parents[3]
    except IndexError as error:
        raise IntegrityError("manifest path is too shallow") from error


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def _hash_cache_tree(cache_root: Path) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    if not cache_root.is_dir():
        raise IntegrityError("cache root is missing")
    result: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    for path in sorted(
        cache_root.rglob("*"), key=lambda item: item.relative_to(cache_root).as_posix()
    ):
        try:
            status_before = path.lstat()
        except OSError as error:
            raise IntegrityError("cache entry disappeared during hashing") from error
        if stat.S_ISLNK(status_before.st_mode):
            raise IntegrityError("cache tree must not contain symlinks")
        if not stat.S_ISREG(status_before.st_mode):
            continue
        identity_before = (
            status_before.st_dev,
            status_before.st_ino,
            status_before.st_mode,
            status_before.st_size,
            status_before.st_mtime_ns,
            status_before.st_ctime_ns,
        )
        relative = path.relative_to(cache_root).as_posix()
        digest_state = hashlib.sha256()
        with _open_cache_regular_stream(
            cache_root, relative, "cache file"
        ) as stream:
            opened = os.fstat(stream.fileno())
            identity_after = (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            if identity_before != identity_after:
                raise IntegrityError("cache file path changed before hashing")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest_state.update(block)
        digest = digest_state.hexdigest()
        result[path.relative_to(cache_root).as_posix()] = (digest, identity_after)
    return result


def _read_cache_payload(
    cache_root: Path,
    relative_path: str,
    name: str,
    *,
    expected_sha256: Optional[str] = None,
) -> bytes:
    with _open_cache_regular_stream(
        cache_root,
        relative_path,
        name,
        expected_sha256=expected_sha256,
    ) as stream:
        return stream.read()


def _nearest_rank(values: Iterable[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return int(ordered[index])


def _offset_summary(values: Iterable[int]) -> Dict[str, int]:
    data = [abs(int(item)) for item in values]
    return {
        "count": len(data),
        "p50": _nearest_rank(data, 0.50),
        "p95": _nearest_rank(data, 0.95),
        "max": max(data) if data else 0,
    }


def _diagnostic_summary(manifest: VerifiedManifest, bag: Mapping[str, Any]) -> Any:
    relative = bag["tf_ego_diagnostic_report_path"]
    if relative is None:
        return None
    try:
        payload = _read_cache_payload(
            manifest.cache_root,
            relative,
            "TF/Ego diagnostic report",
            expected_sha256=bag["tf_ego_diagnostic_report_sha256"],
        )
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("unable to re-read TF/Ego diagnostic report") from error
    if hashlib.sha256(payload).hexdigest() != bag[
        "tf_ego_diagnostic_report_sha256"
    ]:
        raise IntegrityError("TF/Ego diagnostic report changed after manifest load")
    return {
        "position_p99_m": value["position_p99_m"],
        "position_max_m": value["position_max_m"],
        "heading_p99_deg": value["heading_p99_deg"],
        "heading_max_deg": value["heading_max_deg"],
        "raw_record_count": value["raw_ego_record_count"],
        "epoch_local_comparable_count": value["epoch_local_comparable_count"],
        "no_epoch_local_bracket_count": value[
            "no_epoch_local_bracket_count"
        ],
        "ambiguous_epoch_selection_count": value[
            "ambiguous_epoch_selection_count"
        ],
        "cross_epoch_bracket_prevented_count": value[
            "cross_epoch_bracket_prevented_count"
        ],
        "report_sha256": bag["tf_ego_diagnostic_report_sha256"],
    }


def _aggregate_manifest(
    manifest: VerifiedManifest,
    sample_loader: Callable[[VerifiedManifest, str], Any],
) -> Dict[str, Any]:
    samples_by_bag: Dict[str, List[Mapping[str, Any]]] = {}
    for sample in manifest.sample_records:
        samples_by_bag.setdefault(sample["relative_bag_path"], []).append(sample)
    total_drop_counts = {reason.value: 0 for reason in SampleDropReason}
    total_commands = {name: 0 for name in _COMMAND_NAMES}
    camera_min = 255
    camera_max = 0
    lidar_min = 5
    lidar_max = 0
    side_offsets: List[int] = []
    receipt_offsets: List[int] = []
    anchor_offsets: List[int] = []
    max_ego_gap = 0
    max_current_gap = 0
    max_future_gap = 0
    bag_rows = []
    for bag in manifest.bag_records:
        rows = samples_by_bag.get(bag["relative_bag_path"], [])
        commands = {name: 0 for name in _COMMAND_NAMES}
        tf_epochs = set()
        ego_epochs = set()
        bag_side_offsets = []
        bag_receipt_offsets = []
        bag_anchor_offsets = []
        bag_ego_gap = 0
        bag_current_gap = 0
        bag_future_gap = 0
        for sample_record in rows:
            model = sample_loader(manifest, sample_record["sample_id"])
            camera_low = int(np.rint(float(model.camera_feature.min()) * 255.0))
            camera_high = int(np.rint(float(model.camera_feature.max()) * 255.0))
            lidar_low = int(np.rint(float(model.lidar_feature.min()) * 5.0))
            lidar_high = int(np.rint(float(model.lidar_feature.max()) * 5.0))
            camera_min = min(camera_min, camera_low)
            camera_max = max(camera_max, camera_high)
            lidar_min = min(lidar_min, lidar_low)
            lidar_max = max(lidar_max, lidar_high)
            commands[sample_record["command_name"]] += 1
            total_commands[sample_record["command_name"]] += 1
            bag_side_offsets.extend(sample_record["side_camera_header_offset_ns"])
            bag_receipt_offsets.extend(sample_record["sensor_receipt_offset_ns"])
            bag_anchor_offsets.append(
                sample_record["trajectory_anchor_tf_receipt_ns"]
                - sample_record["sensor_source_receipt_ns"][0]
            )
            tf_epochs.add(sample_record["trajectory_anchor_tf_clock_epoch_index"])
            ego_epochs.add(sample_record["ego_bracket_clock_epoch_index"])
            ego_header = sample_record["ego_bracket_header_stamp_ns"]
            current_header = sample_record["current_pose_bracket_header_stamp_ns"]
            future_header = sample_record["future_pose_bracket_header_stamp_ns"]
            bag_ego_gap = max(bag_ego_gap, ego_header[1] - ego_header[0])
            bag_current_gap = max(
                bag_current_gap, current_header[1] - current_header[0]
            )
            bag_future_gap = max(
                [bag_future_gap]
                + [after - before for before, after in future_header]
            )
        side_offsets.extend(bag_side_offsets)
        receipt_offsets.extend(bag_receipt_offsets)
        anchor_offsets.extend(bag_anchor_offsets)
        max_ego_gap = max(max_ego_gap, bag_ego_gap)
        max_current_gap = max(max_current_gap, bag_current_gap)
        max_future_gap = max(max_future_gap, bag_future_gap)
        for name, count in bag["drop_counts"].items():
            total_drop_counts[name] += count
        bag_rows.append(
            {
                "relative_bag_path": bag["relative_bag_path"],
                "source_bag_sha256": bag["source_bag_sha256"],
                "disposition": bag["disposition"],
                "split": bag["split"],
                "sample_count": bag["sample_count"],
                "sidecar_record_count": bag["sidecar_record_count"],
                "drop_counts": dict(bag["drop_counts"]),
                "command_counts": commands,
                "strict_front_identity_verified_count": len(rows),
                "strict_front_header_equals_timestamp_verified_count": len(rows),
                "strict_front_record_index_unique_verified_count": len(rows),
                "side_camera_header_offset_abs_ns": _offset_summary(
                    bag_side_offsets
                ),
                "sensor_receipt_offset_abs_ns": _offset_summary(
                    bag_receipt_offsets
                ),
                "trajectory_anchor_tf_receipt_offset_abs_ns": _offset_summary(
                    bag_anchor_offsets
                ),
                "trajectory_anchor_tf_epoch_count": (
                    max(tf_epochs) + 1 if tf_epochs else 0
                ),
                "trajectory_anchor_tf_epoch_reset_count": (
                    max(tf_epochs) if tf_epochs else 0
                ),
                "ego_clock_epoch_count": max(ego_epochs) + 1 if ego_epochs else 0,
                "ego_clock_epoch_reset_count": max(ego_epochs) if ego_epochs else 0,
                "max_ego_native_header_bracket_gap_ns": bag_ego_gap,
                "max_current_native_header_bracket_gap_ns": bag_current_gap,
                "max_future_native_header_bracket_gap_ns": bag_future_gap,
                "tf_ego_full_stream": _diagnostic_summary(manifest, bag),
            }
        )
    sample_count = len(manifest.sample_records)
    dispositions = {
        disposition: sum(
            bag["disposition"] == disposition for bag in manifest.bag_records
        )
        for disposition in ("ELIGIBLE", "SENSOR_ONLY_EXCLUDED", "REJECTED")
    }
    return {
        "bag_count": len(manifest.bag_records),
        "eligible_bag_count": dispositions["ELIGIBLE"],
        "sensor_only_excluded_bag_count": dispositions[
            "SENSOR_ONLY_EXCLUDED"
        ],
        "rejected_bag_count": dispositions["REJECTED"],
        "sample_count": sample_count,
        "sidecar_record_count": sum(
            bag["sidecar_record_count"] for bag in manifest.bag_records
        ),
        "strict_front_identity_verified_count": sample_count,
        "strict_front_header_equals_timestamp_verified_count": sample_count,
        "strict_front_record_index_unique_verified_count": sample_count,
        "drop_counts": total_drop_counts,
        "command_counts": total_commands,
        "camera_uint8": {"min": camera_min, "max": camera_max},
        "lidar_count_uint8": {"min": lidar_min, "max": lidar_max},
        "side_camera_header_offset_abs_ns": _offset_summary(side_offsets),
        "sensor_receipt_offset_abs_ns": _offset_summary(receipt_offsets),
        "trajectory_anchor_tf_receipt_offset_abs_ns": _offset_summary(
            anchor_offsets
        ),
        "max_ego_native_header_bracket_gap_ns": max_ego_gap,
        "max_current_native_header_bracket_gap_ns": max_current_gap,
        "max_future_native_header_bracket_gap_ns": max_future_gap,
        "bags": bag_rows,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _attrs_equal(left: Any, right: Any) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for name in left.keys():
        a = np.asarray(left[name])
        b = np.asarray(right[name])
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            return False
    return True


def _compare_hdf5(
    *,
    left_root: Path,
    left_relative: str,
    left_sha256: str,
    right_root: Path,
    right_relative: str,
    right_sha256: str,
) -> None:
    try:
        with _open_cache_regular_stream(
            left_root,
            left_relative,
            "primary HDF5 artifact",
            expected_sha256=left_sha256,
        ) as left_stream, _open_cache_regular_stream(
            right_root,
            right_relative,
            "compared HDF5 artifact",
            expected_sha256=right_sha256,
        ) as right_stream:
            with h5py.File(left_stream, "r") as left, h5py.File(
                right_stream, "r"
            ) as right:
                names = sorted(left.keys())
                if names != sorted(right.keys()) or not _attrs_equal(
                    left.attrs, right.attrs
                ):
                    raise IntegrityError("HDF5 schema or attributes differ")
                for name in names:
                    a = left[name]
                    b = right[name]
                    if not isinstance(a, h5py.Dataset) or not isinstance(
                        b, h5py.Dataset
                    ):
                        raise IntegrityError("HDF5 member type differs")
                    if (
                        a.dtype != b.dtype
                        or a.shape != b.shape
                        or not _attrs_equal(a.attrs, b.attrs)
                    ):
                        raise IntegrityError("HDF5 dataset contract differs")
                    rows = a.shape[0] if a.shape else 1
                    for start in range(0, rows, 8):
                        if a.shape:
                            equal = np.array_equal(
                                a[start : start + 8], b[start : start + 8]
                            )
                        else:
                            equal = np.array_equal(a[()], b[()])
                        if not equal:
                            raise IntegrityError(
                                "HDF5 dataset values differ for {}".format(name)
                            )
    except IntegrityError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise IntegrityError("unable to compare HDF5 artifacts") from error


def _normalized_bags(manifest: VerifiedManifest) -> List[Dict[str, Any]]:
    result = []
    for raw in manifest.bag_records:
        bag = _plain(raw)
        if bag["hdf5_sha256"] is not None:
            bag["hdf5_sha256"] = "<verified-semantic-hdf5>"
        result.append(bag)
    return result


def _semantic_compare(
    primary: VerifiedManifest, compared: VerifiedManifest
) -> Dict[str, Any]:
    primary_snapshot_relative = primary.dataset_record["resolved_domain_path"]
    compared_snapshot_relative = compared.dataset_record["resolved_domain_path"]
    if _read_cache_payload(
        primary.cache_root,
        primary_snapshot_relative,
        "primary resolved-domain snapshot",
    ) != _read_cache_payload(
        compared.cache_root,
        compared_snapshot_relative,
        "compared resolved-domain snapshot",
    ):
        raise IntegrityError("resolved-domain snapshot bytes differ")
    if _plain(primary.dataset_record) != _plain(compared.dataset_record):
        raise IntegrityError("dataset manifest semantics differ")
    if _normalized_bags(primary) != _normalized_bags(compared):
        raise IntegrityError("bag inventory, split, drop, or artifact semantics differ")
    if _plain(primary.sample_records) != _plain(compared.sample_records):
        raise IntegrityError("sample order, provenance, or command semantics differ")
    primary_by_path = {
        bag["relative_bag_path"]: bag for bag in primary.bag_records
    }
    compared_by_path = {
        bag["relative_bag_path"]: bag for bag in compared.bag_records
    }
    differing_hdf5 = 0
    for relative, left_bag in primary_by_path.items():
        right_bag = compared_by_path[relative]
        for field in ("sidecar_path", "tf_ego_diagnostic_report_path"):
            left_relative = left_bag[field]
            right_relative = right_bag[field]
            if left_relative is None:
                continue
            digest_field = (
                "sidecar_sha256"
                if field == "sidecar_path"
                else "tf_ego_diagnostic_report_sha256"
            )
            if _read_cache_payload(
                primary.cache_root,
                left_relative,
                "primary {}".format(field),
                expected_sha256=left_bag[digest_field],
            ) != _read_cache_payload(
                compared.cache_root,
                right_relative,
                "compared {}".format(field),
                expected_sha256=right_bag[digest_field],
            ):
                raise IntegrityError("{} bytes differ".format(field))
        if left_bag["hdf5_path"] is None:
            continue
        _compare_hdf5(
            left_root=primary.cache_root,
            left_relative=left_bag["hdf5_path"],
            left_sha256=left_bag["hdf5_sha256"],
            right_root=compared.cache_root,
            right_relative=right_bag["hdf5_path"],
            right_sha256=right_bag["hdf5_sha256"],
        )
        if left_bag["hdf5_sha256"] != right_bag["hdf5_sha256"]:
            differing_hdf5 += 1
    return {
        "semantic_equal": True,
        "compared_manifest_sha256": compared.manifest_sha256,
        "hdf5_container_hash_difference_count": differing_hdf5,
    }


def verify_cache(
    *,
    manifest_path: Path,
    sensor_preset_path: Path,
    report_path: Path,
    compare_manifest_path: Optional[Path] = None,
    manifest_loader: Optional[Callable[..., VerifiedManifest]] = None,
    sample_loader: Optional[Callable[[VerifiedManifest, str], Any]] = None,
) -> Dict[str, Any]:
    """Verify every manifest row without mutating cache artifacts."""

    manifest_path = Path(manifest_path)
    sensor_preset_path = Path(sensor_preset_path)
    report_path = Path(report_path)
    compare_path = None if compare_manifest_path is None else Path(compare_manifest_path)
    load_manifest = manifest_loader or load_verified_manifest
    load_sample = sample_loader or load_tensor_sample
    primary_root = _manifest_cache_root(manifest_path)
    roots = [primary_root]
    if compare_path is not None:
        roots.append(_manifest_cache_root(compare_path))
    if any(_is_within(report_path, root) for root in roots):
        raise IntegrityError("verification report must be outside verified cache roots")
    assert_atomic_output_avoids_paths(
        report_path,
        (sensor_preset_path,),
        output_name="verification report",
    )
    before = {str(root.absolute()): _hash_cache_tree(root) for root in roots}
    primary = load_manifest(
        path=manifest_path,
        sensor_preset_path=sensor_preset_path,
        expected_domain_id=_DOMAIN_ID,
        expected_resolved_domain_sha256=manifest_path.parent.name,
        production=True,
    )
    comparison = None
    if compare_path is not None:
        compared = load_manifest(
            path=compare_path,
            sensor_preset_path=sensor_preset_path,
            expected_domain_id=_DOMAIN_ID,
            expected_resolved_domain_sha256=compare_path.parent.name,
            production=True,
        )
        comparison = _semantic_compare(primary, compared)
    aggregates = _aggregate_manifest(primary, load_sample)
    after = {str(root.absolute()): _hash_cache_tree(root) for root in roots}
    if before != after:
        raise IntegrityError("cache files changed during read-only verification")
    lifecycle = primary.resolved_domain.document["domain_lifecycle"]
    cache = primary.resolved_domain.document["cache"]
    report: Dict[str, Any] = {
        "record_type": "cache_verification",
        "schema_version": 2,
        "manifest_sha256": primary.manifest_sha256,
        "domain_id": primary.dataset_record["domain_id"],
        "resolved_domain_sha256": primary.dataset_record[
            "resolved_domain_sha256"
        ],
        "resolved_domain_path": primary.dataset_record["resolved_domain_path"],
        "resolved_domain_file_sha256": before[str(primary_root.absolute())][
            primary.dataset_record["resolved_domain_path"]
        ][0],
        "sensor_preset_sha256": primary.dataset_record["sensor_preset_sha256"],
        "source_code_commit": primary.dataset_record["source_code_commit"],
        "cache_schema_version": cache["schema_version"],
        "clock_policy_version": cache["hdf5_attributes"][
            "clock_policy_version"
        ],
        "domain_lifecycle": _plain(lifecycle),
        "verified_cache_file_count": len(before[str(primary_root.absolute())]),
        "totals": {key: value for key, value in aggregates.items() if key != "bags"},
        "bags": aggregates["bags"],
        "comparison": comparison,
    }
    def cache_gate() -> None:
        assert_atomic_output_avoids_paths(
            report_path,
            (sensor_preset_path,),
            output_name="verification report",
        )
        final_cache_state = {
            str(root.absolute()): _hash_cache_tree(root) for root in roots
        }
        if before != final_cache_state:
            raise IntegrityError("cache changed before report publication")

    atomic_write_bytes(
        report_path,
        canonical_json_bytes(report),
        pre_replace_gate=cache_gate,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MORAI DDV2 cache read-only")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sensor-preset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compare-manifest", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        verify_cache(
            manifest_path=arguments.manifest,
            compare_manifest_path=arguments.compare_manifest,
            sensor_preset_path=arguments.sensor_preset,
            report_path=arguments.report,
        )
    except (MoraiDDV2Error, OSError, TypeError, ValueError) as error:
        print("verify_cache: {}".format(error), file=sys.stderr)
        return 1
    return 0


__all__ = ("main", "verify_cache")


if __name__ == "__main__":
    raise SystemExit(main())
