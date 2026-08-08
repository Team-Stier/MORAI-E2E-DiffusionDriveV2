from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from morai_ddv2.atomic import (
    assert_atomic_output_avoids_paths,
    atomic_write_bytes,
)
from morai_ddv2.config import canonical_json_bytes, load_resolved_domain
from morai_ddv2.errors import MoraiDDV2Error
from morai_ddv2.errors import IntegrityError
from morai_ddv2.ros1 import BagDisposition, discover_bags, index_bag


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _PinnedBag:
    path: Path
    relative_bag_path: str
    identity: Tuple[int, ...]


def _file_identity(value: os.stat_result) -> Tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _pin_discovered_bag(bag_root: Path, discovered_path: Path) -> _PinnedBag:
    root_absolute = Path(os.path.abspath(str(bag_root)))
    root_resolved = bag_root.resolve(strict=True)
    if root_absolute != root_resolved or bag_root.is_symlink():
        raise IntegrityError("bag_root must not contain symlinks")
    lexical = Path(os.path.abspath(str(discovered_path)))
    try:
        lexical.relative_to(root_absolute)
    except ValueError as error:
        raise IntegrityError("discovered bag is outside bag_root") from error
    try:
        status = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise IntegrityError("discovered bag disappeared during validation") from error
    if stat.S_ISLNK(status.st_mode) or lexical != resolved:
        raise IntegrityError("discovered bag paths and ancestors must not be symlinks")
    if not stat.S_ISREG(status.st_mode):
        raise IntegrityError("discovered bag must be a regular file")
    if resolved.suffix != ".bag" or resolved.name.endswith(".bag.active"):
        raise IntegrityError("resolved finalized input must end in .bag, not .bag.active")
    try:
        relative = resolved.relative_to(root_resolved).as_posix()
    except ValueError as error:
        raise IntegrityError("resolved bag escapes bag_root") from error
    return _PinnedBag(
        path=resolved,
        relative_bag_path=relative,
        identity=_file_identity(status),
    )


def _assert_pinned_bag_unchanged(pinned: _PinnedBag) -> None:
    try:
        current = pinned.path.lstat()
    except OSError as error:
        raise IntegrityError("pinned bag disappeared") from error
    if _file_identity(current) != pinned.identity:
        raise IntegrityError("pinned bag identity changed")


def _sha256_pinned_bag(pinned: _PinnedBag) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(str(pinned.path), flags)
        if _file_identity(os.fstat(descriptor)) != pinned.identity:
            raise IntegrityError("pinned bag changed before fallback hashing")
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
        if _file_identity(os.fstat(descriptor)) != pinned.identity:
            raise IntegrityError("pinned bag changed during fallback hashing")
    except IntegrityError:
        raise
    except OSError as error:
        raise IntegrityError("unable to hash pinned bag") from error
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    _assert_pinned_bag_unchanged(pinned)
    return digest.hexdigest()


def _pin_unique_bags(bag_root: Path, paths: Sequence[Path]) -> Tuple[_PinnedBag, ...]:
    pinned = tuple(_pin_discovered_bag(bag_root, Path(path)) for path in paths)
    relative_paths = [item.relative_bag_path for item in pinned]
    identities = [(item.identity[0], item.identity[1]) for item in pinned]
    if len(relative_paths) != len(set(relative_paths)) or len(identities) != len(
        set(identities)
    ):
        raise IntegrityError("discovery contains a duplicate normalized bag identity")
    return tuple(
        sorted(pinned, key=lambda item: item.relative_bag_path.encode("utf-8"))
    )


def _assert_output_disjoint_from_bag_root(
    *,
    output_path: Path,
    bag_root: Path,
    output_name: str,
    reject_bag_root_descendant: bool,
) -> None:
    try:
        resolved_root = bag_root.resolve(strict=True)
        resolved_output = output_path.resolve(strict=False)
    except OSError as error:
        raise IntegrityError(
            "unable to validate {} against bag_root".format(output_name)
        ) from error
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        output_within_root = False
    else:
        output_within_root = True
    root_within_output = False
    if reject_bag_root_descendant:
        try:
            resolved_root.relative_to(resolved_output)
        except ValueError:
            pass
        else:
            root_within_output = True
    if output_within_root or root_within_output:
        raise IntegrityError(
            "{} must be physically disjoint from bag_root".format(output_name)
        )


def _domain_input_paths(resolved_domain_path: Path, domain: Any) -> Tuple[Path, ...]:
    sensor = domain.document["sensor_preset"]
    configured = sensor["path"]
    if not isinstance(configured, str) or not configured:
        raise IntegrityError("sensor preset path is unavailable")
    relative = Path(configured)
    if relative.is_absolute():
        raise IntegrityError("sensor preset path must be project-relative")
    paths = [Path(resolved_domain_path), _PROJECT_ROOT / relative]
    physical = sensor.get("physical_source_path")
    if isinstance(physical, str) and physical and Path(physical).exists():
        paths.append(Path(physical))
    return tuple(paths)


def _validate_inventory_report_destination(
    *,
    report_path: Path,
    bag_root: Path,
    immutable_inputs: Sequence[Path],
) -> None:
    for candidate in (
        report_path,
        report_path.with_name(report_path.name + ".partial"),
    ):
        _assert_output_disjoint_from_bag_root(
            output_path=candidate,
            bag_root=bag_root,
            output_name="inventory report",
            reject_bag_root_descendant=False,
        )
    assert_atomic_output_avoids_paths(
        report_path,
        immutable_inputs,
        output_name="inventory report",
    )


def _diagnostic_summary(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    metric = value.metric
    return {
        "raw_ego_record_count": value.raw_ego_record_count,
        "epoch_local_comparable_count": value.epoch_local_comparable_count,
        "no_epoch_local_bracket_count": value.no_epoch_local_bracket_count,
        "ambiguous_epoch_selection_count": value.ambiguous_epoch_selection_count,
        "cross_epoch_bracket_prevented_count": value.cross_epoch_bracket_prevented_count,
        "position_p99_m": metric.position_p99_m,
        "position_max_m": metric.position_max_m,
        "heading_p99_deg": metric.heading_p99_deg,
        "heading_max_deg": metric.heading_max_deg,
    }


def inspect_bags(
    *,
    bag_root: Path,
    resolved_domain_path: Path,
    report_path: Path,
    domain_loader: Optional[Callable[[Path], Any]] = None,
    discoverer: Optional[Callable[[Path], Sequence[Path]]] = None,
    indexer: Optional[Callable[[Path, Any], Any]] = None,
) -> Dict[str, Any]:
    """Index finalized bags and atomically publish a deterministic inventory."""

    bag_root = Path(bag_root)
    resolved_domain_path = Path(resolved_domain_path)
    report_path = Path(report_path)
    load_domain = domain_loader or load_resolved_domain
    discover = discoverer or discover_bags
    index = indexer or index_bag
    domain = load_domain(resolved_domain_path)
    if not bag_root.is_dir():
        raise NotADirectoryError(str(bag_root))
    immutable_inputs = _domain_input_paths(resolved_domain_path, domain)
    _validate_inventory_report_destination(
        report_path=report_path,
        bag_root=bag_root,
        immutable_inputs=immutable_inputs,
    )

    finalized = _pin_unique_bags(bag_root, tuple(discover(bag_root)))
    active = tuple(
        sorted(
            (path for path in bag_root.rglob("*.bag.active") if path.is_file()),
            key=lambda path: path.relative_to(bag_root).as_posix().encode("utf-8"),
        )
    )
    rows = []
    counts = {disposition.value: 0 for disposition in BagDisposition}
    seen_eligible_hashes = set()
    for pinned in finalized:
        indexed = index(pinned.path, domain)
        _assert_pinned_bag_unchanged(pinned)
        relative = pinned.relative_bag_path
        inventory = indexed.inventory
        if inventory.disposition is BagDisposition.ELIGIBLE:
            if indexed.source_bag_sha256 in seen_eligible_hashes:
                raise IntegrityError("inventory contains duplicate eligible source bytes")
            seen_eligible_hashes.add(indexed.source_bag_sha256)
        counts[inventory.disposition.value] += 1
        rows.append(
            {
                "relative_bag_path": relative,
                "source_bag_sha256": indexed.source_bag_sha256,
                "disposition": inventory.disposition.value,
                "rejection_code": (
                    None
                    if inventory.rejection_code is None
                    else inventory.rejection_code.value
                ),
                "topic_types": [list(item) for item in inventory.topic_types],
                "topic_record_counts": [
                    [topic, count] for topic, count in inventory.topic_record_counts
                ],
                "camera_decode_error_record_indices": [
                    [topic, list(indices)]
                    for topic, indices in inventory.camera_decode_error_record_indices
                ],
                "tf_ego_diagnostic": _diagnostic_summary(
                    inventory.tf_ego_diagnostic
                ),
            }
        )
    rows.sort(key=lambda item: item["relative_bag_path"].encode("utf-8"))
    report: Dict[str, Any] = {
        "record_type": "bag_inventory",
        "schema_version": 2,
        "domain_id": domain.document["domain_id"],
        "resolved_domain_sha256": domain.sha256,
        "bag_root": str(bag_root.resolve()),
        "finalized_bag_count": len(finalized),
        "active_bag_excluded_count": len(active),
        "active_bag_relative_paths": [
            path.relative_to(bag_root).as_posix() for path in active
        ],
        "disposition_counts": counts,
        "bags": rows,
    }
    def publication_gate() -> None:
        _validate_inventory_report_destination(
            report_path=report_path,
            bag_root=bag_root,
            immutable_inputs=immutable_inputs,
        )

    atomic_write_bytes(
        report_path,
        canonical_json_bytes(report),
        pre_replace_gate=publication_gate,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect deterministic MORAI bag inventory")
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--resolved-domain", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        inspect_bags(
            bag_root=arguments.bag_root,
            resolved_domain_path=arguments.resolved_domain,
            report_path=arguments.report,
        )
    except (MoraiDDV2Error, OSError, TypeError, ValueError) as error:
        print("inspect_bags: {}".format(error), file=sys.stderr)
        return 1
    return 0


__all__ = ("inspect_bags", "main")


if __name__ == "__main__":
    raise SystemExit(main())
