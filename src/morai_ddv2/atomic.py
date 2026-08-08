from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from morai_ddv2.errors import IntegrityError


def _directory_identity(value: os.stat_result) -> Tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode


def assert_atomic_output_avoids_paths(
    output_path: Path,
    protected_paths: Iterable[Path],
    *,
    output_name: str,
) -> None:
    output_path = Path(output_path)
    candidates = (
        output_path,
        output_path.with_name(output_path.name + ".partial"),
    )
    try:
        resolved_candidates = tuple(
            candidate.resolve(strict=False) for candidate in candidates
        )
        resolved_protected = tuple(
            Path(path).resolve(strict=True) for path in protected_paths
        )
    except OSError as error:
        raise IntegrityError(
            "unable to validate {} against immutable inputs".format(output_name)
        ) from error
    if any(
        candidate == protected
        for candidate in resolved_candidates
        for protected in resolved_protected
    ):
        raise IntegrityError(
            "{} final or partial path collides with an immutable input".format(
                output_name
            )
        )
    try:
        protected_identities = {
            (value.st_dev, value.st_ino)
            for value in (path.stat() for path in resolved_protected)
        }
        candidate_identities = set()
        for candidate in candidates:
            try:
                value = candidate.stat()
            except FileNotFoundError:
                continue
            candidate_identities.add((value.st_dev, value.st_ino))
    except OSError as error:
        raise IntegrityError(
            "unable to bind {} and immutable input identities".format(output_name)
        ) from error
    if candidate_identities & protected_identities:
        raise IntegrityError(
            "{} final or partial identity collides with an immutable input".format(
                output_name
            )
        )


def _validated_output_parent(output_path: Path) -> Path:
    absolute_parent = Path(os.path.abspath(str(output_path.parent)))
    try:
        resolved_parent = output_path.parent.resolve(strict=False)
    except OSError as error:
        raise IntegrityError("unable to resolve atomic output parent") from error
    if resolved_parent != absolute_parent:
        raise IntegrityError("atomic output parent must not contain symlinks")

    existing = absolute_parent
    missing = []
    while True:
        try:
            status = existing.lstat()
            break
        except FileNotFoundError:
            if existing == existing.parent:
                raise IntegrityError("atomic output parent has no directory anchor")
            missing.append(existing.name)
            existing = existing.parent
        except OSError as error:
            raise IntegrityError("atomic output parent is unreadable") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise IntegrityError("atomic output ancestor must be a directory")
    try:
        if existing.resolve(strict=True) != existing:
            raise IntegrityError("atomic output ancestor must not contain symlinks")
    except OSError as error:
        raise IntegrityError("unable to resolve atomic output ancestor") from error

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        anchor_identity = _directory_identity(status)
        descriptor = os.open(str(existing), flags)
        if _directory_identity(os.fstat(descriptor)) != anchor_identity:
            raise IntegrityError("atomic output ancestor changed")
        for component in reversed(missing):
            if component in ("", ".", "..") or "/" in component or "\\" in component:
                raise IntegrityError("atomic output parent component is unsafe")
            try:
                os.mkdir(component, 0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(component, flags, dir_fd=descriptor)
            child_status = os.fstat(child)
            if not stat.S_ISDIR(child_status.st_mode):
                os.close(child)
                raise IntegrityError("atomic output parent is not a directory")
            os.close(descriptor)
            descriptor = child
    except IntegrityError:
        raise
    except OSError as error:
        raise IntegrityError("unable to create atomic output parent") from error
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return absolute_parent


def atomic_write_bytes(
    output_path: Path,
    payload: bytes,
    *,
    pre_replace_gate: Optional[Callable[[], None]] = None,
) -> None:
    """Durably publish bytes through a sibling ``.partial`` file."""

    output_path = Path(output_path)
    parent = _validated_output_parent(output_path)
    leaf = output_path.name
    if leaf in ("", ".", "..") or "/" in leaf or "\\" in leaf:
        raise IntegrityError("atomic output name is unsafe")
    partial = leaf + ".partial"
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    directory_fd = -1
    partial_fd = -1
    published = False
    try:
        parent_status = parent.lstat()
        parent_identity = _directory_identity(parent_status)
        directory_fd = os.open(str(parent), directory_flags)
        if _directory_identity(os.fstat(directory_fd)) != parent_identity:
            raise IntegrityError("atomic output parent changed before staging")
        try:
            partial_status = os.stat(
                partial, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            partial_status = None
        if partial_status is not None:
            if (
                not stat.S_ISREG(partial_status.st_mode)
                or partial_status.st_nlink != 1
            ):
                raise IntegrityError(
                    "atomic output partial must be an unlinked regular file"
                )
            os.unlink(partial, dir_fd=directory_fd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        partial_fd = os.open(partial, flags, 0o600, dir_fd=directory_fd)
        with os.fdopen(partial_fd, "wb") as stream:
            partial_fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if pre_replace_gate is not None:
            pre_replace_gate()
        if _directory_identity(os.fstat(directory_fd)) != parent_identity:
            raise IntegrityError("atomic output parent changed before publication")
        try:
            if _directory_identity(parent.lstat()) != parent_identity:
                raise IntegrityError("atomic output parent path changed")
            if parent.resolve(strict=True) != parent:
                raise IntegrityError("atomic output parent became a symlink alias")
        except OSError as error:
            raise IntegrityError("atomic output parent changed") from error
        try:
            os.replace(
                partial,
                leaf,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        except TypeError:
            # Preserve compatibility with instrumented/legacy two-argument
            # ``os.replace`` implementations after the pinned-parent checks.
            os.replace(parent / partial, parent / leaf)
        published = True
        os.fsync(directory_fd)
    except IntegrityError:
        raise
    except OSError as error:
        raise IntegrityError("unable to atomically publish output") from error
    finally:
        if partial_fd >= 0:
            try:
                os.close(partial_fd)
            except OSError:
                pass
        if directory_fd >= 0:
            if not published:
                try:
                    os.unlink(partial, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            try:
                os.close(directory_fd)
            except OSError:
                pass
