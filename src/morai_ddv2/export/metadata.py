from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Literal

from morai_ddv2.atomic import _validated_output_parent
from morai_ddv2.config import canonical_json_bytes
from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.contracts import TRAINABLE_PREFIXES, FreezeManifest
from morai_ddv2.training.metrics import DOMAIN_ID, VALIDATION_RUN_ID
from morai_ddv2.upstream.ddv2_sel import (
    RAW_SEL_KEY_COUNT,
    RUNTIME_SEL_KEY_COUNT,
    RawSelCheckpoint,
)

from .schema import _validated_mutable_names, validate_raw_schema
from .state_hash import state_dict_sha256 as _state_dict_sha256
from .state_hash import state_schema_sha256


MUTABLE_BN_POLICY = "trainable_prefixes:running_mean,running_var,num_batches_tracked"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z\Z"
)


class _FrozenCommits(dict):
    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("source_commits is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, memo: Any) -> dict:
        del memo
        return dict(self)


def _require_sha256(value: str, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: str, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")
    return value


def _require_source_commits(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source_commits must be a nonempty mapping")
    owned = {}
    for name, commit in value.items():
        _require_nonempty_string(name, "source_commits key")
        if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
            raise ValueError(
                "source_commits values must be 40-character lowercase commit IDs"
            )
        owned[name] = commit
    return _FrozenCommits(owned)


def _require_schema_versions(cache: int, clock: int) -> None:
    if type(cache) is not int or cache != 2:
        raise ValueError("cache_schema_version must be exactly 2")
    if type(clock) is not int or clock != 1:
        raise ValueError("clock_policy_version must be exactly 1")


def _require_best_micro_ade(value: float) -> float:
    if type(value) not in (int, float):
        raise TypeError("best_micro_ade must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("best_micro_ade must be finite and nonnegative")
    return result


def _require_utc_timestamp(value: str) -> str:
    if type(value) is not str or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError("exported_at_utc must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("exported_at_utc must be a valid UTC timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("exported_at_utc must be an explicit UTC timestamp")
    return value


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class ExportIdentities:
    domain_id: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    training_manifest_sha256: str
    cache_schema_version: Literal[2]
    clock_policy_version: Literal[1]
    training_config_sha256: str
    source_commits: Mapping[str, str]
    best_micro_ade: float
    validation_run_id: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.domain_id, "domain_id")
        if self.domain_id != DOMAIN_ID:
            raise ValueError("domain_id must match the MORAI training contract")
        for name in (
            "resolved_domain_sha256",
            "sensor_preset_sha256",
            "training_manifest_sha256",
            "training_config_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_schema_versions(
            self.cache_schema_version,
            self.clock_policy_version,
        )
        object.__setattr__(
            self,
            "source_commits",
            _require_source_commits(self.source_commits),
        )
        object.__setattr__(
            self,
            "best_micro_ade",
            _require_best_micro_ade(self.best_micro_ade),
        )
        _require_nonempty_string(self.validation_run_id, "validation_run_id")
        if self.validation_run_id != VALIDATION_RUN_ID:
            raise ValueError(
                "validation_run_id must match the deterministic validation contract"
            )


@dataclass(frozen=True)
class ExportMetadata:
    base_checkpoint_sha256: str
    base_state_dict_sha256: str
    export_checkpoint_sha256: str
    export_state_dict_sha256: str
    raw_schema_sha256: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    training_manifest_sha256: str
    cache_schema_version: Literal[2]
    clock_policy_version: Literal[1]
    training_config_sha256: str
    domain_id: str
    source_commits: Mapping[str, str]
    trainable_prefixes: tuple[str, ...]
    mutable_bn_policy: str
    raw_key_count: int
    runtime_key_count: int
    best_micro_ade: float
    validation_run_id: str
    exported_at_utc: str

    def __post_init__(self) -> None:
        for name in (
            "base_checkpoint_sha256",
            "base_state_dict_sha256",
            "export_checkpoint_sha256",
            "export_state_dict_sha256",
            "raw_schema_sha256",
            "resolved_domain_sha256",
            "sensor_preset_sha256",
            "training_manifest_sha256",
            "training_config_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_schema_versions(
            self.cache_schema_version,
            self.clock_policy_version,
        )
        _require_nonempty_string(self.domain_id, "domain_id")
        if self.domain_id != DOMAIN_ID:
            raise ValueError("domain_id must match the MORAI training contract")
        object.__setattr__(
            self,
            "source_commits",
            _require_source_commits(self.source_commits),
        )
        if type(self.trainable_prefixes) is not tuple or (
            self.trainable_prefixes != TRAINABLE_PREFIXES
        ):
            raise ValueError("trainable_prefixes must match the exact DDV2 allowlist")
        if self.mutable_bn_policy != MUTABLE_BN_POLICY:
            raise ValueError("mutable_bn_policy is invalid")
        if type(self.raw_key_count) is not int or (
            self.raw_key_count != RAW_SEL_KEY_COUNT
        ):
            raise ValueError(f"raw_key_count must be exactly {RAW_SEL_KEY_COUNT}")
        if type(self.runtime_key_count) is not int or (
            self.runtime_key_count != RUNTIME_SEL_KEY_COUNT
        ):
            raise ValueError(
                f"runtime_key_count must be exactly {RUNTIME_SEL_KEY_COUNT}"
            )
        object.__setattr__(
            self,
            "best_micro_ade",
            _require_best_micro_ade(self.best_micro_ade),
        )
        _require_nonempty_string(self.validation_run_id, "validation_run_id")
        if self.validation_run_id != VALIDATION_RUN_ID:
            raise ValueError(
                "validation_run_id must match the deterministic validation contract"
            )
        _require_utc_timestamp(self.exported_at_utc)


def build_export_metadata(
    *,
    base: RawSelCheckpoint,
    checkpoint_sha256: str,
    state_dict_sha256: str,
    identities: ExportIdentities,
    manifest: FreezeManifest,
) -> ExportMetadata:
    if type(base) is not RawSelCheckpoint:
        raise TypeError("base must be RawSelCheckpoint")
    if type(identities) is not ExportIdentities:
        raise TypeError("identities must be ExportIdentities")
    _require_sha256(base.checkpoint_sha256, "base_checkpoint_sha256")
    checkpoint_hash = _require_sha256(
        checkpoint_sha256,
        "export_checkpoint_sha256",
    )
    state_hash = _require_sha256(state_dict_sha256, "export_state_dict_sha256")
    _validated_mutable_names(manifest)
    report = validate_raw_schema(base.raw_state_dict, base.raw_state_dict)
    return ExportMetadata(
        base_checkpoint_sha256=base.checkpoint_sha256,
        base_state_dict_sha256=_state_dict_sha256(base.raw_state_dict),
        export_checkpoint_sha256=checkpoint_hash,
        export_state_dict_sha256=state_hash,
        raw_schema_sha256=state_schema_sha256(base.raw_state_dict),
        resolved_domain_sha256=identities.resolved_domain_sha256,
        sensor_preset_sha256=identities.sensor_preset_sha256,
        training_manifest_sha256=identities.training_manifest_sha256,
        cache_schema_version=identities.cache_schema_version,
        clock_policy_version=identities.clock_policy_version,
        training_config_sha256=identities.training_config_sha256,
        domain_id=identities.domain_id,
        source_commits=identities.source_commits,
        trainable_prefixes=TRAINABLE_PREFIXES,
        mutable_bn_policy=MUTABLE_BN_POLICY,
        raw_key_count=report.raw_key_count,
        runtime_key_count=report.runtime_key_count,
        best_micro_ade=identities.best_micro_ade,
        validation_run_id=identities.validation_run_id,
        exported_at_utc=_utc_now(),
    )


def _lstat(path: Path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise IntegrityError("unable to inspect export output path") from error


def _require_final_absent(path: Path, *, label: str) -> None:
    if _lstat(path) is not None:
        raise IntegrityError(f"{label} already exists")


def _require_publish_path_available(path: Path, *, label: str) -> None:
    path = Path(path)
    _require_final_absent(path, label=label)
    partial = path.with_name(path.name + ".partial")
    partial_status = _lstat(partial)
    if partial_status is not None and (
        not stat.S_ISREG(partial_status.st_mode) or partial_status.st_nlink != 1
    ):
        raise IntegrityError(f"{label} partial path is unsafe")


def _file_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _unlink_if_owned(
    directory_fd: int,
    leaf: str,
    identity: tuple[int, int, int],
) -> bool:
    try:
        current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    if _file_identity(current) != identity:
        return False
    try:
        os.unlink(leaf, dir_fd=directory_fd)
    except OSError:
        return False
    return True


def _cleanup_owned_publication(
    directory_fd: int,
    *,
    final: str,
    partial: str,
    identity: tuple[int, int, int],
) -> None:
    removed = _unlink_if_owned(directory_fd, final, identity)
    removed = _unlink_if_owned(directory_fd, partial, identity) or removed
    if removed:
        try:
            os.fsync(directory_fd)
        except OSError:
            pass


def _require_parent_unchanged(
    parent: Path,
    directory_fd: int,
    identity: tuple[int, int, int],
    *,
    label: str,
) -> None:
    try:
        if _file_identity(os.fstat(directory_fd)) != identity:
            raise IntegrityError(f"{label} parent identity changed")
        if _file_identity(parent.lstat()) != identity:
            raise IntegrityError(f"{label} parent path changed")
        if parent.resolve(strict=True) != parent:
            raise IntegrityError(f"{label} parent became a symlink alias")
    except IntegrityError:
        raise
    except OSError as error:
        raise IntegrityError(f"{label} parent changed") from error


def _atomic_publish_new_bytes(path: Path, payload: bytes, *, label: str) -> None:
    output = Path(path)
    _require_publish_path_available(output, label=label)
    parent = _validated_output_parent(output)
    leaf = output.name
    if leaf in ("", ".", "..") or "/" in leaf or "\\" in leaf:
        raise IntegrityError(f"{label} name is unsafe")
    partial = leaf + ".partial"
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    directory_fd = -1
    partial_fd = -1
    staged_identity = None
    try:
        parent_status = parent.lstat()
        directory_fd = os.open(str(parent), directory_flags)
        parent_identity = _file_identity(parent_status)
        if _file_identity(os.fstat(directory_fd)) != parent_identity:
            raise IntegrityError(f"{label} parent changed before staging")
        try:
            partial_status = os.stat(
                partial,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            partial_status = None
        if partial_status is not None:
            if not stat.S_ISREG(partial_status.st_mode) or partial_status.st_nlink != 1:
                raise IntegrityError(f"{label} partial path is unsafe")
            os.unlink(partial, dir_fd=directory_fd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        partial_fd = os.open(partial, flags, 0o600, dir_fd=directory_fd)
        staged_status = os.fstat(partial_fd)
        staged_identity = _file_identity(staged_status)
        with os.fdopen(partial_fd, "wb") as stream:
            partial_fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        current_partial = os.stat(
            partial,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            _file_identity(current_partial) != staged_identity
            or not stat.S_ISREG(current_partial.st_mode)
            or current_partial.st_nlink != 1
        ):
            raise IntegrityError(f"{label} partial changed before publication")
        _require_parent_unchanged(
            parent,
            directory_fd,
            parent_identity,
            label=label,
        )
        try:
            os.link(
                partial,
                leaf,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise IntegrityError(f"{label} already exists") from error
        final_status = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if _file_identity(final_status) != staged_identity:
            raise IntegrityError(f"{label} final identity changed")
        os.fsync(directory_fd)
        _require_parent_unchanged(
            parent,
            directory_fd,
            parent_identity,
            label=label,
        )
        if not _unlink_if_owned(directory_fd, partial, staged_identity):
            raise IntegrityError(f"{label} partial changed during cleanup")
    except IntegrityError:
        if directory_fd >= 0 and staged_identity is not None:
            _cleanup_owned_publication(
                directory_fd,
                final=leaf,
                partial=partial,
                identity=staged_identity,
            )
        raise
    except OSError as error:
        if directory_fd >= 0 and staged_identity is not None:
            _cleanup_owned_publication(
                directory_fd,
                final=leaf,
                partial=partial,
                identity=staged_identity,
            )
        raise IntegrityError(f"unable to atomically publish {label}") from error
    finally:
        if partial_fd >= 0:
            try:
                os.close(partial_fd)
            except OSError:
                pass
        if directory_fd >= 0:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def _metadata_document(metadata: ExportMetadata) -> Mapping[str, Any]:
    return {
        field.name: getattr(metadata, field.name) for field in fields(ExportMetadata)
    }


def write_metadata_atomic(path: Path, metadata: ExportMetadata) -> None:
    if type(metadata) is not ExportMetadata:
        raise TypeError("metadata must be ExportMetadata")
    output = Path(path)
    payload = canonical_json_bytes(_metadata_document(metadata))
    _atomic_publish_new_bytes(
        output,
        payload,
        label="export metadata",
    )


__all__ = (
    "ExportIdentities",
    "ExportMetadata",
    "build_export_metadata",
    "write_metadata_atomic",
)
