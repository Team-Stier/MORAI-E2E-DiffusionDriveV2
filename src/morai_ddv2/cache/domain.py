from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Iterator, List, Mapping, Optional, Tuple

from morai_ddv2.config import canonical_json_bytes, sha256_bytes, sha256_file
from morai_ddv2.config.models import ResolvedDomain, ResolvedDomainArtifact
from morai_ddv2.errors import ConfigurationError, IntegrityError


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be 64 lowercase hexadecimal characters".format(name))
    return value


def _commit(value: Any, name: str) -> str:
    if type(value) is not str or _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be 40 lowercase hexadecimal characters".format(name))
    return value


def _component(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError("{} must be a string".format(name))
    if (
        not value
        or "\x00" in value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("{} must be one safe path component".format(name))
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("{} must be valid Unicode".format(name)) from error
    return value


def _relative_path(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError("{} must be a string".format(name))
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("{} must be a normalized relative POSIX path".format(name))
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("{} must be valid Unicode".format(name)) from error
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("{} must be a normalized relative POSIX path".format(name))
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("{} contains an unsafe path segment".format(name))
    return value


def _regular_identity(path: Path, name: str) -> Tuple[int, ...]:
    try:
        status = path.lstat()
    except OSError as error:
        raise IntegrityError("{} is missing or unreadable".format(name)) from error
    if not stat.S_ISREG(status.st_mode):
        raise IntegrityError("{} must be a regular file".format(name))
    return _status_identity(status)


def _status_identity(status: os.stat_result) -> Tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _close_descriptors(descriptors: Tuple[Tuple[int, Tuple[int, ...]], ...]) -> None:
    for descriptor, _ in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _open_cache_parent_descriptors(
    cache_root: Path,
    relative_path: str,
    *,
    create_parents: bool,
) -> Tuple[Tuple[Tuple[int, Tuple[int, ...]], ...], str]:
    if not isinstance(cache_root, Path):
        raise TypeError("cache_root must be a pathlib.Path")
    relative = _relative_path(relative_path, "cache relative path")
    root = cache_root.absolute()
    if create_parents:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise IntegrityError("unable to create cache root") from error
    try:
        root_status = root.lstat()
    except OSError as error:
        raise IntegrityError("cache root is missing or unreadable") from error
    if not stat.S_ISDIR(root_status.st_mode):
        raise IntegrityError("cache root must be a non-symlink directory")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    opened: List[Tuple[int, Tuple[int, ...]]] = []
    try:
        root_descriptor = os.open(str(root), directory_flags)
        opened.append(
            (root_descriptor, _status_identity(os.fstat(root_descriptor)))
        )
        if not stat.S_ISDIR(opened[-1][1][2]):
            raise IntegrityError("cache root descriptor is not a directory")
        for component in PurePosixPath(relative).parts[:-1]:
            if create_parents:
                try:
                    os.mkdir(component, 0o755, dir_fd=opened[-1][0])
                except FileExistsError:
                    pass
            descriptor = os.open(
                component,
                directory_flags,
                dir_fd=opened[-1][0],
            )
            identity = _status_identity(os.fstat(descriptor))
            if not stat.S_ISDIR(identity[2]):
                os.close(descriptor)
                raise IntegrityError("cache ancestor must be a directory")
            opened.append((descriptor, identity))
    except (OSError, IntegrityError) as error:
        _close_descriptors(tuple(opened))
        if isinstance(error, IntegrityError):
            raise
        raise IntegrityError(
            "cache path contains a missing, symlinked, or non-directory ancestor"
        ) from error
    return tuple(opened), PurePosixPath(relative).name


def _open_cache_regular_fd(
    cache_root: Path, relative_path: str, name: str
) -> Tuple[int, Tuple[int, ...], Tuple[Tuple[int, Tuple[int, ...]], ...]]:
    ancestors, leaf = _open_cache_parent_descriptors(
        cache_root, relative_path, create_parents=False
    )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(leaf, flags, dir_fd=ancestors[-1][0])
        identity = _status_identity(os.fstat(descriptor))
    except OSError as error:
        _close_descriptors(ancestors)
        raise IntegrityError("{} is missing, unreadable, or a symlink".format(name)) from error
    if not stat.S_ISREG(identity[2]):
        os.close(descriptor)
        _close_descriptors(ancestors)
        raise IntegrityError("{} must be a regular file".format(name))
    return descriptor, identity, ancestors


def _verify_cache_regular_fd(
    descriptor: int,
    identity: Tuple[int, ...],
    ancestors: Tuple[Tuple[int, Tuple[int, ...]], ...],
    name: str,
) -> None:
    try:
        if _status_identity(os.fstat(descriptor)) != identity:
            raise IntegrityError("{} changed while open".format(name))
        if any(
            _status_identity(os.fstat(parent_descriptor)) != parent_identity
            for parent_descriptor, parent_identity in ancestors
        ):
            raise IntegrityError("{} cache ancestor changed while open".format(name))
    except OSError as error:
        raise IntegrityError("unable to verify open {}".format(name)) from error


def _read_cache_bytes(
    cache_root: Path, relative_path: str, name: str
) -> Tuple[bytes, Tuple[int, ...]]:
    with _open_cache_regular_stream(cache_root, relative_path, name) as stream:
        identity = _status_identity(os.fstat(stream.fileno()))
        return stream.read(), identity


def _cache_regular_identity(
    cache_root: Path, relative_path: str, name: str
) -> Tuple[int, ...]:
    descriptor = -1
    ancestors: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()
    try:
        descriptor, identity, ancestors = _open_cache_regular_fd(
            cache_root, relative_path, name
        )
        _verify_cache_regular_fd(descriptor, identity, ancestors, name)
        return identity
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _close_descriptors(ancestors)


@contextmanager
def _open_cache_regular_stream(
    cache_root: Path,
    relative_path: str,
    name: str,
    *,
    expected_identity: Optional[Tuple[int, ...]] = None,
    expected_sha256: Optional[str] = None,
) -> Iterator[BinaryIO]:
    descriptor = -1
    stream = None
    ancestors: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()
    try:
        descriptor, identity, ancestors = _open_cache_regular_fd(
            cache_root, relative_path, name
        )
        if expected_identity is not None and identity != expected_identity:
            raise IntegrityError("opened {} identity differs from verified identity".format(name))
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        if expected_sha256 is not None:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise IntegrityError("opened {} bytes differ from bound SHA-256".format(name))
            stream.seek(0)
        _verify_cache_regular_fd(stream.fileno(), identity, ancestors, name)
        yield stream
        _verify_cache_regular_fd(stream.fileno(), identity, ancestors, name)
        if _cache_regular_identity(cache_root, relative_path, name) != identity:
            raise IntegrityError("{} path identity changed while open".format(name))
    except IntegrityError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise IntegrityError("unable to open verified {}".format(name)) from error
    finally:
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _close_descriptors(ancestors)


def _atomic_publish_bytes(
    cache_root: Path, relative_path: str, payload: bytes
) -> None:
    ancestors, leaf = _open_cache_parent_descriptors(
        cache_root, relative_path, create_parents=True
    )
    parent_descriptor = ancestors[-1][0]
    partial = leaf + ".partial"
    descriptor = -1
    try:
        try:
            partial_status = os.stat(
                partial, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            partial_status = None
        if partial_status is not None:
            if not stat.S_ISREG(partial_status.st_mode):
                raise IntegrityError("stale partial must be a regular file")
            os.unlink(partial, dir_fd=parent_descriptor)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(
                partial,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as error:
            raise IntegrityError("partial path appeared during publication") from error
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            partial,
            leaf,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    except OSError as error:
        raise IntegrityError("unable to atomically publish cache file") from error
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _close_descriptors(ancestors)


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON value: {}".format(value))


def _snapshot_from_bytes(
    payload: bytes,
    *,
    sensor_preset_path: Optional[Path],
    expected_domain_id: Optional[str],
    expected_resolved_domain_sha256: Optional[str],
) -> ResolvedDomain:
    try:
        publication = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise IntegrityError("resolved-domain snapshot is not strict UTF-8 JSON") from error
    if type(publication) is not dict:
        raise IntegrityError("resolved-domain snapshot must be a JSON object")
    publication = dict(publication)
    resolved_sha = publication.pop("resolved_domain_sha256", None)
    try:
        resolved_sha = _sha256(resolved_sha, "resolved_domain_sha256")
    except (TypeError, ValueError) as error:
        raise IntegrityError("resolved-domain snapshot hash is malformed") from error
    if sha256_bytes(canonical_json_bytes(publication)) != resolved_sha:
        raise IntegrityError("resolved-domain snapshot self-excluding hash mismatch")
    canonical_publication = dict(publication)
    canonical_publication["resolved_domain_sha256"] = resolved_sha
    try:
        canonical_bytes = canonical_json_bytes(canonical_publication)
    except (TypeError, ValueError) as error:
        raise IntegrityError("resolved-domain snapshot cannot be canonicalized") from error
    if canonical_bytes != payload:
        raise IntegrityError("resolved-domain snapshot is not canonical RFC 8785 JSON")

    try:
        domain_id = _component(publication["domain_id"], "domain_id")
        sensor_sha = _sha256(
            publication["sensor_preset"]["sha256"], "sensor_preset_sha256"
        )
        _commit(publication["source_code_commit"], "source_code_commit")
        if (
            type(publication["schema_version"]) is not int
            or publication["schema_version"] != 1
        ):
            raise ValueError("resolved root schema must be 1")
        if (
            type(publication["cache"]["schema_version"]) is not int
            or publication["cache"]["schema_version"] != 2
        ):
            raise ValueError("cache schema must be 2")
        if (
            type(publication["cache"]["manifest_schema_version"]) is not int
            or publication["cache"]["manifest_schema_version"] != 2
        ):
            raise ValueError("manifest schema must be 2")
        if publication["cache"]["manifest_path_template"] != (
            "manifests/{domain_id}/{resolved_domain_sha256}/manifest.jsonl"
        ):
            raise ValueError("manifest path template mismatch")
        if (
            type(publication["sampling"]["clock_policy"]["version"]) is not int
            or publication["sampling"]["clock_policy"]["version"] != 1
        ):
            raise ValueError("clock policy must be 1")
        split = publication["split"]
        if split["seed_prefix"] != "20260806:":
            raise ValueError("split seed prefix mismatch")
        if type(split["train_fraction"]) is not float or split[
            "train_fraction"
        ] != 0.8:
            raise ValueError("split train fraction mismatch")
        if split["unit"] != "bag":
            raise ValueError("split unit mismatch")
        if type(split["ordering"]) is not list or tuple(split["ordering"]) != (
            "sha256_seed_prefix_plus_relative_path",
            "relative_path_utf8",
        ):
            raise ValueError("split ordering mismatch")
        if split["relative_path_separator"] != "/":
            raise ValueError("split path separator mismatch")
        if split["active_bag_suffix_excluded"] != ".bag.active":
            raise ValueError("active bag suffix policy mismatch")
        diagnostics = publication["diagnostics"]
        if (
            type(diagnostics["tf_ego_minimum_comparable_pairs"]) is not int
            or diagnostics["tf_ego_minimum_comparable_pairs"] != 10
        ):
            raise ValueError("TF/Ego minimum comparable-pair policy mismatch")
        if (
            type(diagnostics["tf_ego_position_p99_tolerance_m"]) is not float
            or diagnostics["tf_ego_position_p99_tolerance_m"] != 0.05
        ):
            raise ValueError("TF/Ego position p99 policy mismatch")
        if (
            type(diagnostics["tf_ego_heading_p99_tolerance_deg"]) is not float
            or diagnostics["tf_ego_heading_p99_tolerance_deg"] != 0.5
        ):
            raise ValueError("TF/Ego heading p99 policy mismatch")
        lifecycle = publication["domain_lifecycle"]
        if lifecycle["contract_revision_context"] != (
            "pre_first_production_unpublished"
        ):
            raise ValueError("contract revision context mismatch")
        if lifecycle["after_first_production_publication"] != (
            "new_domain_id_required"
        ):
            raise ValueError("post-publication policy mismatch")
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("resolved-domain snapshot contract is invalid") from error
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise IntegrityError("resolved-domain snapshot domain identity mismatch")
    if (
        expected_resolved_domain_sha256 is not None
        and resolved_sha != expected_resolved_domain_sha256
    ):
        raise IntegrityError("resolved-domain snapshot SHA-256 identity mismatch")
    if sensor_preset_path is not None:
        if not isinstance(sensor_preset_path, Path):
            raise TypeError("sensor_preset_path must be a pathlib.Path")
        identity = _regular_identity(sensor_preset_path, "sensor preset")
        try:
            actual_sensor_sha = sha256_file(sensor_preset_path)
        except OSError as error:
            raise IntegrityError("unable to hash current sensor preset") from error
        if identity != _regular_identity(sensor_preset_path, "sensor preset"):
            raise IntegrityError("sensor preset changed while being hashed")
        if actual_sensor_sha != sensor_sha:
            raise IntegrityError("current sensor preset SHA-256 mismatch")
    return ResolvedDomain(document=publication, sha256=resolved_sha)


@dataclass(frozen=True)
class DomainSnapshotRecord:
    domain_id: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    source_code_commit: str
    path: str
    file_sha256: str

    def __post_init__(self) -> None:
        domain_id = _component(self.domain_id, "domain_id")
        resolved_sha = _sha256(
            self.resolved_domain_sha256, "resolved_domain_sha256"
        )
        object.__setattr__(
            self,
            "sensor_preset_sha256",
            _sha256(self.sensor_preset_sha256, "sensor_preset_sha256"),
        )
        object.__setattr__(
            self, "source_code_commit", _commit(self.source_code_commit, "source_code_commit")
        )
        object.__setattr__(self, "file_sha256", _sha256(self.file_sha256, "file_sha256"))
        path = _relative_path(self.path, "path")
        expected = "domains/{}/{}/resolved-domain.json".format(
            domain_id, resolved_sha
        )
        if path != expected:
            raise ValueError("domain snapshot path does not match its identity")


def _validated_artifact(artifact: Any) -> ResolvedDomainArtifact:
    if type(artifact) is not ResolvedDomainArtifact:
        raise TypeError("artifact must be a ResolvedDomainArtifact")
    try:
        copied = ResolvedDomainArtifact(
            document=artifact.document,
            sha256=artifact.sha256,
        )
        parsed = _snapshot_from_bytes(
            copied.publication_bytes,
            sensor_preset_path=None,
            expected_domain_id=None,
            expected_resolved_domain_sha256=copied.sha256,
        )
    except (TypeError, ValueError) as error:
        raise IntegrityError("resolved-domain artifact is invalid") from error
    return ResolvedDomainArtifact(document=parsed.document, sha256=parsed.sha256)


def publish_domain_snapshot(
    *, cache_root: Path, artifact: ResolvedDomainArtifact
) -> DomainSnapshotRecord:
    if not isinstance(cache_root, Path):
        raise TypeError("cache_root must be a pathlib.Path")
    artifact = _validated_artifact(artifact)
    document: Mapping[str, Any] = artifact.document
    try:
        domain_id = _component(document["domain_id"], "domain_id")
        sensor_sha = _sha256(
            document["sensor_preset"]["sha256"], "sensor_preset_sha256"
        )
        source_commit = _commit(document["source_code_commit"], "source_code_commit")
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("resolved-domain identity is invalid") from error
    relative = "domains/{}/{}/resolved-domain.json".format(
        domain_id, artifact.sha256
    )
    path = cache_root / relative
    payload = artifact.publication_bytes
    if os.path.lexists(str(path)):
        existing, _ = _read_cache_bytes(
            cache_root, relative, "resolved-domain snapshot"
        )
        if existing != payload:
            raise IntegrityError("existing resolved-domain snapshot is not immutable")
    else:
        _atomic_publish_bytes(cache_root, relative, payload)
    published, _ = _read_cache_bytes(
        cache_root, relative, "resolved-domain snapshot"
    )
    if published != payload:
        raise IntegrityError("published resolved-domain snapshot bytes mismatch")
    return DomainSnapshotRecord(
        domain_id=domain_id,
        resolved_domain_sha256=artifact.sha256,
        sensor_preset_sha256=sensor_sha,
        source_code_commit=source_commit,
        path=relative,
        file_sha256=sha256_bytes(published),
    )


def load_domain_snapshot(
    path: Path,
    sensor_preset_path: Path,
    expected_domain_id: str,
    expected_resolved_domain_sha256: str,
) -> ResolvedDomain:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    expected_domain_id = _component(expected_domain_id, "expected_domain_id")
    expected_resolved_domain_sha256 = _sha256(
        expected_resolved_domain_sha256, "expected_resolved_domain_sha256"
    )
    path = path.absolute()
    expected_relative = "domains/{}/{}/resolved-domain.json".format(
        expected_domain_id, expected_resolved_domain_sha256
    )
    if (
        path.name != "resolved-domain.json"
        or path.parent.name != expected_resolved_domain_sha256
        or path.parent.parent.name != expected_domain_id
        or path.parent.parent.parent.name != "domains"
    ):
        raise IntegrityError("resolved-domain snapshot path identity mismatch")
    cache_root = path.parents[3]
    payload, _ = _read_cache_bytes(
        cache_root, expected_relative, "resolved-domain snapshot"
    )
    return _snapshot_from_bytes(
        payload,
        sensor_preset_path=sensor_preset_path,
        expected_domain_id=expected_domain_id,
        expected_resolved_domain_sha256=expected_resolved_domain_sha256,
    )


def _verify_domain_snapshot_record(
    cache_root: Path, record: DomainSnapshotRecord
) -> ResolvedDomain:
    if type(record) is not DomainSnapshotRecord:
        raise TypeError("record must be a DomainSnapshotRecord")
    try:
        record = DomainSnapshotRecord(**record.__dict__)
    except (TypeError, ValueError) as error:
        raise IntegrityError("domain snapshot record is invalid") from error
    payload, _ = _read_cache_bytes(
        cache_root, record.path, "resolved-domain snapshot"
    )
    actual_file_sha = sha256_bytes(payload)
    if actual_file_sha != record.file_sha256:
        raise IntegrityError("resolved-domain snapshot file SHA-256 mismatch")
    domain = _snapshot_from_bytes(
        payload,
        sensor_preset_path=None,
        expected_domain_id=record.domain_id,
        expected_resolved_domain_sha256=record.resolved_domain_sha256,
    )
    if domain.document["sensor_preset"]["sha256"] != record.sensor_preset_sha256:
        raise IntegrityError("resolved-domain snapshot sensor identity mismatch")
    if domain.document["source_code_commit"] != record.source_code_commit:
        raise IntegrityError("resolved-domain snapshot source commit mismatch")
    return domain


__all__ = (
    "DomainSnapshotRecord",
    "load_domain_snapshot",
    "publish_domain_snapshot",
)
