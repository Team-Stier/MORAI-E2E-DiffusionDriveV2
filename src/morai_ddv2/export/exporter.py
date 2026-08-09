from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import io
from pathlib import Path
import stat

import torch

from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.contracts import FreezeManifest
from morai_ddv2.upstream.ddv2_sel import RawSelCheckpoint, load_raw_sel_checkpoint

from .metadata import (
    ExportIdentities,
    _atomic_publish_new_bytes,
    _require_publish_path_available,
    build_export_metadata,
    write_metadata_atomic,
)
from .schema import (
    assert_only_allowed_changed,
    merge_mutable_state,
    validate_raw_schema,
)
from .state_hash import state_dict_sha256


@dataclass(frozen=True)
class ExportBundle:
    checkpoint: Path
    metadata: Path
    checkpoint_sha256: str
    state_dict_sha256: str


def _metadata_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".metadata.json")


def _validate_base(base: RawSelCheckpoint) -> None:
    if type(base) is not RawSelCheckpoint:
        raise TypeError("base must be RawSelCheckpoint")
    if not isinstance(base.container, Mapping):
        raise IntegrityError("base checkpoint container must be a mapping")
    container_state = base.container.get("state_dict")
    if container_state is not base.raw_state_dict:
        raise IntegrityError(
            "base checkpoint container state_dict does not match raw_state_dict"
        )
    validate_raw_schema(base.raw_state_dict, base.raw_state_dict)


def _serialize_checkpoint(container: Mapping[str, object]) -> bytes:
    stream = io.BytesIO()
    try:
        torch.save(container, stream)
    except Exception as error:
        raise IntegrityError("unable to serialize export checkpoint") from error
    return stream.getvalue()


def _write_torch_checkpoint_atomic(path: Path, container: Mapping[str, object]) -> None:
    payload = _serialize_checkpoint(container)
    _atomic_publish_new_bytes(
        path,
        payload,
        label="export checkpoint",
    )


def _regular_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        value = path.lstat()
    except OSError as error:
        raise IntegrityError("unable to inspect finalized export checkpoint") from error
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise IntegrityError("finalized export checkpoint must be a regular file")
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def export_sel_bundle(
    base: RawSelCheckpoint,
    mutable_state: Mapping[str, torch.Tensor],
    output: Path,
    *,
    identities: ExportIdentities,
    manifest: FreezeManifest,
) -> ExportBundle:
    _validate_base(base)
    if type(identities) is not ExportIdentities:
        raise TypeError("identities must be ExportIdentities")
    output = Path(output)
    metadata_path = _metadata_path(output)
    _require_publish_path_available(output, label="export checkpoint")
    _require_publish_path_available(metadata_path, label="export metadata")

    merged = merge_mutable_state(base.raw_state_dict, mutable_state, manifest)
    validate_raw_schema(merged, base.raw_state_dict)
    assert_only_allowed_changed(base.raw_state_dict, merged, manifest)
    intended_state_hash = state_dict_sha256(merged)

    container = dict(base.container)
    container["state_dict"] = merged
    _write_torch_checkpoint_atomic(output, container)

    identity_before = _regular_file_identity(output)
    reopened = load_raw_sel_checkpoint(output)
    identity_after = _regular_file_identity(output)
    if identity_after != identity_before:
        raise IntegrityError("export checkpoint changed while being verified")
    validate_raw_schema(reopened.raw_state_dict, base.raw_state_dict)
    assert_only_allowed_changed(base.raw_state_dict, reopened.raw_state_dict, manifest)
    actual_state_hash = state_dict_sha256(reopened.raw_state_dict)
    if actual_state_hash != intended_state_hash:
        raise IntegrityError("finalized export checkpoint state hash changed")

    metadata = build_export_metadata(
        base=base,
        checkpoint_sha256=reopened.checkpoint_sha256,
        state_dict_sha256=actual_state_hash,
        identities=identities,
        manifest=manifest,
    )
    write_metadata_atomic(metadata_path, metadata)
    return ExportBundle(
        checkpoint=output,
        metadata=metadata_path,
        checkpoint_sha256=reopened.checkpoint_sha256,
        state_dict_sha256=actual_state_hash,
    )


__all__ = ("ExportBundle", "export_sel_bundle")
