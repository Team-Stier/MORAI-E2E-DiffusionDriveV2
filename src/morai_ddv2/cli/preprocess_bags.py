from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from morai_ddv2.cache.domain import publish_domain_snapshot
from morai_ddv2.cache.hdf5 import BagArtifactRecord, publish_bag_artifacts
from morai_ddv2.cache.manifest import (
    BagManifestInput,
    DirtyOverride,
    ManifestRecord,
    RejectionReport,
    assign_bag_splits,
    publish_manifest,
)
from morai_ddv2.config import (
    ResolvedDomainArtifact,
    canonical_json_bytes,
    load_resolved_domain,
)
from morai_ddv2.errors import ConfigurationError, IntegrityError, MoraiDDV2Error
from morai_ddv2.preprocess.materialize import NoValidSampleError, materialize_samples
from morai_ddv2.preprocess.planner import (
    FrontTimestampOrderViolation,
    SampleDropReason,
    build_sample_plan,
)
from morai_ddv2.ros1 import BagDisposition, BagRejectReason, discover_bags, index_bag

from .inspect_bags import (
    _PinnedBag,
    _assert_pinned_bag_unchanged,
    _assert_output_disjoint_from_bag_root,
    _pin_discovered_bag,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class SourceTreeState:
    source_code_commit: str
    dirty: bool
    dirty_diff_sha256: Optional[str]

    def __post_init__(self) -> None:
        if type(self.source_code_commit) is not str or _COMMIT.fullmatch(
            self.source_code_commit
        ) is None:
            raise ValueError("source_code_commit must be a lowercase Git commit")
        if type(self.dirty) is not bool:
            raise TypeError("dirty must be a bool")
        if self.dirty:
            if type(self.dirty_diff_sha256) is not str or _SHA256.fullmatch(
                self.dirty_diff_sha256
            ) is None:
                raise ValueError("dirty source requires dirty_diff_sha256")
        elif self.dirty_diff_sha256 is not None:
            raise ValueError("clean source cannot carry dirty_diff_sha256")


@dataclass(frozen=True)
class BagProcessingFailure:
    relative_bag_path: str
    operation: str
    error_type: str


@dataclass(frozen=True)
class PreprocessResult:
    manifest: Optional[ManifestRecord]
    rejection_report: Optional[RejectionReport]
    failures: Tuple[BagProcessingFailure, ...]
    dirty_override: Optional[DirtyOverride]

    @property
    def success(self) -> bool:
        return self.manifest is not None and not self.failures


def _run_git(project_root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git"] + list(arguments),
        cwd=str(project_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ConfigurationError("unable to inspect source tree")
    return completed.stdout


def _frame(digest: Any, label: bytes, payload: bytes) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _stream_git_digest(project_root: Path, arguments: Sequence[str]) -> bytes:
    try:
        process = subprocess.Popen(
            ["git"] + list(arguments),
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise ConfigurationError("unable to stream source diff") from error
    if process.stdout is None:
        process.kill()
        raise ConfigurationError("source diff stream is unavailable")
    digest = hashlib.sha256()
    try:
        for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(block)
    finally:
        process.stdout.close()
    if process.wait() != 0:
        raise ConfigurationError("unable to stream source diff")
    return digest.digest()


def _git_status(project_root: Path) -> bytes:
    return _run_git(
        project_root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
    )


def _configured_submodule_paths(project_root: Path) -> Tuple[Path, ...]:
    gitmodules = project_root / ".gitmodules"
    if not gitmodules.exists():
        return ()
    try:
        status = gitmodules.lstat()
    except OSError as error:
        raise ConfigurationError("unable to inspect .gitmodules") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ConfigurationError(".gitmodules must be a regular non-symlink file")
    completed = subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(gitmodules),
            "--get-regexp",
            r"^submodule\..*\.path$",
        ],
        cwd=str(project_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode == 1:
        return ()
    if completed.returncode != 0:
        raise ConfigurationError("unable to enumerate source submodules")
    result = []
    for line in completed.stdout.splitlines():
        try:
            _, raw_path = line.split(None, 1)
            relative = raw_path.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError) as error:
            raise ConfigurationError("submodule path is malformed") from error
        candidate = project_root / relative
        absolute = Path(os.path.abspath(str(candidate)))
        resolved = candidate.resolve(strict=True)
        if absolute != resolved or candidate.is_symlink() or not resolved.is_dir():
            raise ConfigurationError("submodule worktree must be a non-symlink directory")
        result.append(resolved)
        result.extend(_configured_submodule_paths(resolved))
    return tuple(result)


def _status_entry_paths(status: bytes) -> Tuple[str, ...]:
    result = []
    for entry in status.split(b"\x00"):
        if len(entry) < 4 or entry[2:3] != b" ":
            continue
        try:
            result.append(entry[3:].decode("utf-8", errors="strict"))
        except UnicodeDecodeError as error:
            raise ConfigurationError("source status path is not UTF-8") from error
    return tuple(result)


def _ensure_submodules_clean(project_root: Path, top_status: bytes) -> None:
    submodules = _configured_submodule_paths(project_root)
    top_paths = set(_status_entry_paths(top_status))
    for worktree in submodules:
        relative = worktree.relative_to(project_root).as_posix()
        if relative in top_paths or _git_status(worktree):
            raise ConfigurationError(
                "dirty or moved submodule is not allowed in source provenance"
            )


def _source_file_identity(value: os.stat_result) -> Tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_untracked_entry(project_root: Path, raw_relative: bytes) -> bytes:
    try:
        relative = raw_relative.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ConfigurationError("untracked source path is not UTF-8") from error
    relative_path = Path(relative)
    if relative_path.is_absolute() or any(
        part in ("", ".", "..") for part in relative_path.parts
    ):
        raise ConfigurationError("untracked source path is unsafe")
    path = project_root / relative_path
    try:
        before = path.lstat()
    except OSError as error:
        raise ConfigurationError("untracked source entry disappeared") from error
    entry_digest = hashlib.sha256()
    _frame(entry_digest, b"path", raw_relative)
    _frame(entry_digest, b"mode", int(before.st_mode).to_bytes(8, "big"))
    if stat.S_ISLNK(before.st_mode):
        try:
            target = os.readlink(str(path)).encode("utf-8", errors="strict")
            after = path.lstat()
        except (OSError, UnicodeEncodeError) as error:
            raise ConfigurationError("unable to bind untracked symlink") from error
        if _source_file_identity(after) != _source_file_identity(before):
            raise ConfigurationError("untracked symlink changed during probe")
        _frame(entry_digest, b"kind", b"symlink")
        _frame(entry_digest, b"target", target)
        return entry_digest.digest()
    if not stat.S_ISREG(before.st_mode):
        raise ConfigurationError("untracked source entries must be files or symlinks")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    content = hashlib.sha256()
    try:
        descriptor = os.open(str(path), flags)
        if _source_file_identity(os.fstat(descriptor)) != _source_file_identity(before):
            raise ConfigurationError("untracked file changed before hashing")
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            content.update(block)
        if _source_file_identity(os.fstat(descriptor)) != _source_file_identity(before):
            raise ConfigurationError("untracked file changed during hashing")
    except ConfigurationError:
        raise
    except OSError as error:
        raise ConfigurationError("unable to hash untracked source file") from error
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    try:
        after = path.lstat()
    except OSError as error:
        raise ConfigurationError("untracked source file disappeared") from error
    if _source_file_identity(after) != _source_file_identity(before):
        raise ConfigurationError("untracked file path changed during hashing")
    _frame(entry_digest, b"kind", b"regular")
    _frame(entry_digest, b"content_sha256", content.digest())
    return entry_digest.digest()


def _untracked_digest(project_root: Path, status: bytes) -> bytes:
    aggregate = hashlib.sha256()
    entries = sorted(
        entry[3:]
        for entry in status.split(b"\x00")
        if entry.startswith(b"?? ")
    )
    for raw_relative in entries:
        _frame(
            aggregate,
            b"untracked_entry",
            _hash_untracked_entry(project_root, raw_relative),
        )
    return aggregate.digest()


def probe_source_tree(project_root: Path = _PROJECT_ROOT) -> SourceTreeState:
    project_root = Path(project_root)
    commit = _run_git(project_root, ("rev-parse", "HEAD")).decode("ascii").strip()
    status = _git_status(project_root)
    _ensure_submodules_clean(project_root, status)
    if not status:
        status_after = _git_status(project_root)
        _ensure_submodules_clean(project_root, status_after)
        commit_after = (
            _run_git(project_root, ("rev-parse", "HEAD"))
            .decode("ascii")
            .strip()
        )
        if status_after != status or commit_after != commit:
            raise ConfigurationError("source tree changed during provenance probe")
        return SourceTreeState(commit, False, None)
    diff_arguments = ("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    diff_before = _stream_git_digest(project_root, diff_arguments)
    untracked_before = _untracked_digest(project_root, status)
    status_after = _git_status(project_root)
    diff_after = _stream_git_digest(project_root, diff_arguments)
    untracked_after = _untracked_digest(project_root, status_after)
    _ensure_submodules_clean(project_root, status_after)
    commit_after = _run_git(project_root, ("rev-parse", "HEAD")).decode("ascii").strip()
    if (
        status_after != status
        or diff_after != diff_before
        or untracked_after != untracked_before
        or commit_after != commit
    ):
        raise ConfigurationError("source tree changed during provenance probe")
    digest = hashlib.sha256()
    _frame(digest, b"source_status", status)
    _frame(digest, b"tracked_diff_sha256", diff_before)
    _frame(digest, b"untracked_set_sha256", untracked_before)
    return SourceTreeState(commit, True, digest.hexdigest())


def _zero_drop_counts() -> dict:
    return {reason.value: 0 for reason in SampleDropReason}


def _drop_counts(drops: Sequence[Any]) -> dict:
    result = _zero_drop_counts()
    for drop in drops:
        result[drop.reason.value] += 1
    return result


def _manifest_input_from_inventory(
    *, relative: str, indexed: Any
) -> BagManifestInput:
    inventory = indexed.inventory
    return BagManifestInput(
        relative_bag_path=relative,
        source_bag_sha256=indexed.source_bag_sha256,
        disposition=inventory.disposition,
        artifact=None,
        drop_counts=_zero_drop_counts(),
        rejection_code=inventory.rejection_code,
        tf_ego_diagnostic=inventory.tf_ego_diagnostic,
    )


def _rejected_input(
    *,
    relative: str,
    source_bag_sha256: str,
    diagnostic: Any,
    rejection_code: BagRejectReason,
    drops: Sequence[Any] = (),
) -> BagManifestInput:
    return BagManifestInput(
        relative_bag_path=relative,
        source_bag_sha256=source_bag_sha256,
        disposition=BagDisposition.REJECTED,
        artifact=None,
        drop_counts=_drop_counts(drops),
        rejection_code=rejection_code,
        tf_ego_diagnostic=diagnostic,
    )


def _lexical_bag_label(bag_root: Path, bag_path: Path, index: int) -> str:
    root = Path(os.path.abspath(str(bag_root)))
    lexical = Path(os.path.abspath(str(bag_path)))
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError:
        return "invalid-outside-bag-root-{}.bag".format(index)
    if not relative or relative == ".":
        return "invalid-bag-path-{}.bag".format(index)
    try:
        relative.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return "invalid-bag-path-{}.bag".format(index)
    return relative


def _pin_preprocess_inputs(
    bag_root: Path, discovered: Sequence[Path]
) -> Tuple[Tuple[_PinnedBag, ...], Tuple[BagProcessingFailure, ...]]:
    candidates = []
    for index, raw_path in enumerate(discovered):
        try:
            path = Path(raw_path)
            label = _lexical_bag_label(bag_root, path, index)
        except (TypeError, ValueError):
            label = "invalid-bag-path-{}.bag".format(index)
            candidates.append((os.fsencode(label), index, None, label))
            continue
        candidates.append((os.fsencode(label), index, path, label))
    candidates.sort(key=lambda item: (item[0], item[1]))

    pinned = []
    failures = []
    seen_paths = set()
    seen_identities = set()
    for _, _, path, label in candidates:
        if path is None:
            failures.append(
                BagProcessingFailure(label, "validate_path", "TypeError")
            )
            continue
        try:
            item = _pin_discovered_bag(bag_root, path)
        except (IntegrityError, OSError, ValueError) as error:
            failures.append(
                BagProcessingFailure(label, "validate_path", type(error).__name__)
            )
            continue
        identity = (item.identity[0], item.identity[1])
        if item.relative_bag_path in seen_paths or identity in seen_identities:
            failures.append(
                BagProcessingFailure(
                    item.relative_bag_path, "validate_path", "IntegrityError"
                )
            )
            continue
        seen_paths.add(item.relative_bag_path)
        seen_identities.add(identity)
        pinned.append(item)
    return (
        tuple(
            sorted(
                pinned,
                key=lambda item: item.relative_bag_path.encode("utf-8"),
            )
        ),
        tuple(failures),
    )


def preprocess_bags(
    *,
    bag_root: Path,
    resolved_domain_path: Path,
    cache_root: Path,
    allow_dirty_development: bool = False,
    domain_loader: Optional[Callable[[Path], Any]] = None,
    source_probe: Optional[Callable[[], SourceTreeState]] = None,
    discoverer: Optional[Callable[[Path], Sequence[Path]]] = None,
    indexer: Optional[Callable[[Path, Any], Any]] = None,
    planner: Optional[Callable[[Any, Any], Any]] = None,
    materializer: Optional[Callable[..., Any]] = None,
    artifact_publisher: Optional[Callable[..., BagArtifactRecord]] = None,
    snapshot_publisher: Optional[Callable[..., Any]] = None,
    manifest_publisher: Optional[Callable[..., ManifestRecord]] = None,
) -> PreprocessResult:
    """Run the bounded two-pass bag pipeline and publish only a complete manifest."""

    if type(allow_dirty_development) is not bool:
        raise TypeError("allow_dirty_development must be a bool")
    bag_root = Path(bag_root)
    resolved_domain_path = Path(resolved_domain_path)
    cache_root = Path(cache_root)
    load_domain = domain_loader or load_resolved_domain
    inspect_source = source_probe or probe_source_tree
    discover = discoverer or discover_bags
    index = indexer or index_bag
    plan_samples = planner or build_sample_plan
    materialize = materializer or materialize_samples
    publish_artifact = artifact_publisher or publish_bag_artifacts
    publish_snapshot = snapshot_publisher or publish_domain_snapshot
    publish_dataset_manifest = manifest_publisher or publish_manifest

    domain = load_domain(resolved_domain_path)
    if not bag_root.is_dir():
        raise NotADirectoryError(str(bag_root))
    _assert_output_disjoint_from_bag_root(
        output_path=cache_root,
        bag_root=bag_root,
        output_name="cache_root",
        reject_bag_root_descendant=True,
    )
    source_before = inspect_source()
    if type(source_before) is not SourceTreeState:
        raise TypeError("source_probe must return SourceTreeState")
    if source_before.dirty and not allow_dirty_development:
        raise ConfigurationError(
            "source tree is dirty; use --allow-dirty-development only for development"
        )
    dirty_override = (
        DirtyOverride(
            allow_dirty_development=True,
            source_code_commit=source_before.source_code_commit,
            dirty_diff_sha256=source_before.dirty_diff_sha256,
        )
        if source_before.dirty
        else None
    )
    snapshot = publish_snapshot(
        cache_root=cache_root,
        artifact=ResolvedDomainArtifact(
            document=domain.document,
            sha256=domain.sha256,
        ),
    )

    discovered = tuple(discover(bag_root))
    if not discovered:
        raise ConfigurationError("no finalized .bag files were discovered")
    ordered, validation_failures = _pin_preprocess_inputs(bag_root, discovered)
    inputs = []
    artifacts = []
    failures = list(validation_failures)
    published_eligible_hashes = set()
    for pinned in ordered:
        relative = pinned.relative_bag_path
        bag_path = pinned.path
        try:
            indexed = index(bag_path, domain)
            _assert_pinned_bag_unchanged(pinned)
        except Exception as error:
            failures.append(
                BagProcessingFailure(relative, "index", type(error).__name__)
            )
            continue
        inventory = indexed.inventory
        if inventory.disposition is not BagDisposition.ELIGIBLE:
            inputs.append(
                _manifest_input_from_inventory(relative=relative, indexed=indexed)
            )
            continue
        if indexed.source_bag_sha256 in published_eligible_hashes:
            failures.append(
                BagProcessingFailure(relative, "validate_identity", "IntegrityError")
            )
            continue
        try:
            plan = plan_samples(indexed, domain)
        except FrontTimestampOrderViolation:
            inputs.append(
                _rejected_input(
                    relative=relative,
                    source_bag_sha256=indexed.source_bag_sha256,
                    rejection_code=BagRejectReason.FRONT_TIMESTAMP_ORDER_VIOLATION,
                    diagnostic=inventory.tf_ego_diagnostic,
                )
            )
            continue
        except Exception as error:
            failures.append(
                BagProcessingFailure(relative, "plan", type(error).__name__)
            )
            continue
        if not plan.samples:
            inputs.append(
                _rejected_input(
                    relative=relative,
                    source_bag_sha256=indexed.source_bag_sha256,
                    rejection_code=BagRejectReason.NO_VALID_SAMPLE,
                    diagnostic=inventory.tf_ego_diagnostic,
                    drops=plan.drops,
                )
            )
            continue
        stream = None
        artifact = None
        no_valid_sample = False
        material_error = None
        try:
            stream = materialize(bag_path=bag_path, plan=plan, domain=domain)
            artifact = publish_artifact(
                cache_root=cache_root,
                domain=domain,
                source_bag_sha256=indexed.source_bag_sha256,
                relative_bag_path=relative,
                samples=stream,
                source_code_commit=source_before.source_code_commit,
            )
        except NoValidSampleError:
            no_valid_sample = True
        except Exception as error:
            material_error = error
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    if material_error is None:
                        material_error = error
        drops = getattr(stream, "drop_records", plan.drops)
        if material_error is not None:
            failures.append(
                BagProcessingFailure(
                    relative, "materialize", type(material_error).__name__
                )
            )
            continue
        if no_valid_sample:
            inputs.append(
                _rejected_input(
                    relative=relative,
                    source_bag_sha256=indexed.source_bag_sha256,
                    diagnostic=inventory.tf_ego_diagnostic,
                    rejection_code=BagRejectReason.NO_VALID_SAMPLE,
                    drops=drops,
                )
            )
            continue
        if artifact is None:
            raise AssertionError("successful materialization did not publish an artifact")
        artifacts.append(artifact)
        published_eligible_hashes.add(indexed.source_bag_sha256)
        inputs.append(
            BagManifestInput(
                relative_bag_path=relative,
                source_bag_sha256=indexed.source_bag_sha256,
                disposition=BagDisposition.ELIGIBLE,
                artifact=artifact,
                drop_counts=_drop_counts(drops),
                rejection_code=None,
                tf_ego_diagnostic=inventory.tf_ego_diagnostic,
            )
        )

    source_after = inspect_source()
    if source_after != source_before:
        raise ConfigurationError("source tree changed during preprocessing")
    rejection_report = RejectionReport(tuple(inputs)) if inputs else None
    rejected = any(
        item.disposition is BagDisposition.REJECTED for item in inputs
    )
    eligible_paths = [
        item.relative_bag_path
        for item in inputs
        if item.disposition is BagDisposition.ELIGIBLE
    ]
    if failures or rejected or not eligible_paths or rejection_report is None:
        return PreprocessResult(
            manifest=None,
            rejection_report=rejection_report,
            failures=tuple(failures),
            dirty_override=dirty_override,
        )
    manifest = publish_dataset_manifest(
        cache_root=cache_root,
        domain_snapshot=snapshot,
        bags=tuple(artifacts),
        splits=assign_bag_splits(eligible_paths),
        rejection_report=rejection_report,
        dirty_override=dirty_override,
    )
    return PreprocessResult(
        manifest=manifest,
        rejection_report=rejection_report,
        failures=(),
        dirty_override=dirty_override,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess MORAI bags into DDV2 cache")
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--resolved-domain", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--allow-dirty-development", action="store_true")
    return parser


def _result_summary(result: PreprocessResult) -> dict:
    bags = []
    if result.rejection_report is not None:
        for item in result.rejection_report.bags:
            bags.append(
                {
                    "relative_bag_path": item.relative_bag_path,
                    "source_bag_sha256": item.source_bag_sha256,
                    "disposition": item.disposition.value,
                    "rejection_code": (
                        None
                        if item.rejection_code is None
                        else item.rejection_code.value
                    ),
                }
            )
    bags.sort(key=lambda item: item["relative_bag_path"].encode("utf-8"))
    failures = [
        {
            "relative_bag_path": item.relative_bag_path,
            "operation": item.operation,
            "error_type": item.error_type,
        }
        for item in result.failures
    ]
    failures.sort(
        key=lambda item: (
            item["relative_bag_path"].encode("utf-8"),
            item["operation"],
            item["error_type"],
        )
    )
    observed_paths = {
        item["relative_bag_path"] for item in bags + failures
    }
    return {
        "success": result.success,
        "manifest_path": None if result.manifest is None else result.manifest.path,
        "observed_path_count": len(observed_paths),
        "bag_count": len(bags),
        "rejected_count": sum(
            item["disposition"] == BagDisposition.REJECTED.value for item in bags
        ),
        "failure_count": len(failures),
        "bags": bags,
        "failures": failures,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = preprocess_bags(
            bag_root=arguments.bag_root,
            resolved_domain_path=arguments.resolved_domain,
            cache_root=arguments.cache_root,
            allow_dirty_development=arguments.allow_dirty_development,
        )
    except (MoraiDDV2Error, OSError, TypeError, ValueError) as error:
        print("preprocess_bags: {}".format(error), file=sys.stderr)
        return 1
    print(canonical_json_bytes(_result_summary(result)).decode("utf-8"))
    return 0 if result.success else 1


__all__ = (
    "BagProcessingFailure",
    "PreprocessResult",
    "SourceTreeState",
    "main",
    "preprocess_bags",
    "probe_source_tree",
)


if __name__ == "__main__":
    raise SystemExit(main())
