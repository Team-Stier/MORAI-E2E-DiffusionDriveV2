from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path

import pytest
import torch

from morai_ddv2.errors import IntegrityError
from morai_ddv2.export.state_hash import state_dict_sha256
from morai_ddv2.training.contracts import FreezeManifest
from morai_ddv2.upstream.ddv2_sel import RawSelCheckpoint, load_raw_sel_checkpoint


PREFIX = "agent._transfuser_model."
IMAGE_WEIGHT = "_backbone.image_encoder.layer.weight"
IMAGE_RUNNING_MEAN = "_backbone.image_encoder.norm.running_mean"
FROZEN_WEIGHT = "_trajectory_head.weight"
VOCAB = "_trajectory_head.vocab"


def _raw_state() -> OrderedDict[str, torch.Tensor]:
    named = [
        (PREFIX + IMAGE_WEIGHT, torch.tensor([1.0, 2.0], dtype=torch.float32)),
        (PREFIX + IMAGE_RUNNING_MEAN, torch.tensor([0.0], dtype=torch.float32)),
        (PREFIX + FROZEN_WEIGHT, torch.tensor([[3.0]], dtype=torch.float32)),
    ]
    named.extend(
        (f"{PREFIX}layers.layer_{index:03d}.weight", torch.tensor([float(index)]))
        for index in range(968)
    )
    named.append((PREFIX + VOCAB, torch.tensor([4.0], dtype=torch.float32)))
    return OrderedDict(named)


def _base() -> RawSelCheckpoint:
    raw = _raw_state()
    return RawSelCheckpoint(
        container={"state_dict": raw, "token": b"base"},
        raw_state_dict=raw,
        checkpoint_sha256="a" * 64,
    )


def _manifest() -> FreezeManifest:
    return FreezeManifest(
        trainable_parameters=(IMAGE_WEIGHT,),
        frozen_parameters=(FROZEN_WEIGHT,),
        mutable_state_entries=(IMAGE_WEIGHT, IMAGE_RUNNING_MEAN),
    )


def _mutable_state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        [
            (IMAGE_WEIGHT, torch.tensor([8.0, 9.0], dtype=torch.float32)),
            (IMAGE_RUNNING_MEAN, torch.tensor([7.0], dtype=torch.float32)),
        ]
    )


def _identities():
    from morai_ddv2.export.metadata import ExportIdentities

    return ExportIdentities(
        domain_id="morai-ddv2-v1-oracle-command",
        resolved_domain_sha256="b" * 64,
        sensor_preset_sha256="c" * 64,
        training_manifest_sha256="d" * 64,
        cache_schema_version=2,
        clock_policy_version=1,
        training_config_sha256="e" * 64,
        source_commits={"training": "1" * 40},
        best_micro_ade=1.25,
        validation_run_id="morai-ddv2-v1-oracle-command-validation-v1",
    )


class _FailingWriteStream:
    def __init__(self, stream) -> None:
        self._stream = stream

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self._stream.close()

    def write(self, _payload) -> None:
        raise OSError("injected write failure")

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _inject_nth_write(monkeypatch, nth: int) -> None:
    from morai_ddv2 import atomic as atomic_module

    real_fdopen = atomic_module.os.fdopen
    calls = 0

    def failing_fdopen(*args, **kwargs):
        nonlocal calls
        stream = real_fdopen(*args, **kwargs)
        calls += 1
        if calls == nth:
            return _FailingWriteStream(stream)
        return stream

    monkeypatch.setattr(atomic_module.os, "fdopen", failing_fdopen)


def _inject_nth_fsync(monkeypatch, nth: int) -> None:
    from morai_ddv2 import atomic as atomic_module

    real_fsync = atomic_module.os.fsync
    calls = 0

    def failing_fsync(file_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == nth:
            raise OSError("injected fsync failure")
        real_fsync(file_descriptor)

    monkeypatch.setattr(atomic_module.os, "fsync", failing_fsync)


def _inject_nth_link(monkeypatch, nth: int) -> None:
    from morai_ddv2 import atomic as atomic_module

    real_link = atomic_module.os.link
    calls = 0

    def failing_link(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == nth:
            raise OSError("injected link failure")
        real_link(*args, **kwargs)

    monkeypatch.setattr(atomic_module.os, "link", failing_link)


def _export(output: Path):
    from morai_ddv2.export.exporter import export_sel_bundle

    return export_sel_bundle(
        _base(),
        _mutable_state(),
        output,
        identities=_identities(),
        manifest=_manifest(),
    )


@pytest.mark.parametrize("failure", ["write", "file_fsync", "link", "dir_fsync"])
def test_checkpoint_publication_failure_never_publishes_metadata(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    output = tmp_path / "ddv2.ckpt"
    metadata = output.with_suffix(".ckpt.metadata.json")
    if failure == "write":
        _inject_nth_write(monkeypatch, 1)
    elif failure == "file_fsync":
        _inject_nth_fsync(monkeypatch, 1)
    elif failure == "link":
        _inject_nth_link(monkeypatch, 1)
    else:
        _inject_nth_fsync(monkeypatch, 2)

    with pytest.raises(IntegrityError, match="publish"):
        _export(output)

    assert not metadata.exists()
    assert not metadata.with_name(metadata.name + ".partial").exists()
    assert not output.with_name(output.name + ".partial").exists()
    if output.exists():
        loaded = load_raw_sel_checkpoint(output)
        assert (
            loaded.checkpoint_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("failure", ["write", "file_fsync", "link", "dir_fsync"])
def test_metadata_publication_failure_leaves_verifiable_checkpoint_only(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    output = tmp_path / "ddv2.ckpt"
    metadata = output.with_suffix(".ckpt.metadata.json")
    if failure == "write":
        _inject_nth_write(monkeypatch, 2)
    elif failure == "file_fsync":
        _inject_nth_fsync(monkeypatch, 3)
    elif failure == "link":
        _inject_nth_link(monkeypatch, 2)
    else:
        _inject_nth_fsync(monkeypatch, 4)

    with pytest.raises(IntegrityError, match="publish"):
        _export(output)

    assert output.is_file()
    loaded = load_raw_sel_checkpoint(output)
    assert loaded.checkpoint_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert state_dict_sha256(loaded.raw_state_dict)
    assert not metadata.exists()
    assert not metadata.with_name(metadata.name + ".partial").exists()


@pytest.mark.parametrize("existing", ["checkpoint", "metadata"])
def test_export_refuses_to_replace_preexisting_final_output(
    tmp_path: Path, existing: str
) -> None:
    output = tmp_path / "ddv2.ckpt"
    metadata = output.with_suffix(".ckpt.metadata.json")
    path = output if existing == "checkpoint" else metadata
    path.write_bytes(b"preexisting final must survive")

    with pytest.raises(IntegrityError, match="already exists"):
        _export(output)

    assert path.read_bytes() == b"preexisting final must survive"
    other = metadata if existing == "checkpoint" else output
    assert not other.exists()


def test_checkpoint_no_clobber_publish_preserves_a_racing_final(
    tmp_path: Path, monkeypatch
) -> None:
    from morai_ddv2 import atomic as atomic_module

    output = tmp_path / "ddv2.ckpt"
    metadata = output.with_suffix(".ckpt.metadata.json")
    real_link = atomic_module.os.link
    events = []

    def competing_link(*args, **kwargs) -> None:
        if not events:
            output.write_bytes(b"competing final must survive")
            events.append("competing-final-created")
        real_link(*args, **kwargs)

    monkeypatch.setattr(atomic_module.os, "link", competing_link)

    with pytest.raises(IntegrityError, match="already exists|publish"):
        _export(output)

    assert events == ["competing-final-created"]
    assert output.read_bytes() == b"competing final must survive"
    assert not output.with_name(output.name + ".partial").exists()
    assert not metadata.exists()


@pytest.mark.parametrize(
    "unsafe",
    ["checkpoint_symlink", "checkpoint_partial_directory", "metadata_partial_symlink"],
)
def test_export_fails_closed_on_unsafe_output_or_partial_path(
    tmp_path: Path, unsafe: str
) -> None:
    output = tmp_path / "ddv2.ckpt"
    metadata = output.with_suffix(".ckpt.metadata.json")
    protected = tmp_path / "protected"
    protected.write_bytes(b"do not modify")
    if unsafe == "checkpoint_symlink":
        output.symlink_to(protected)
    elif unsafe == "checkpoint_partial_directory":
        output.with_name(output.name + ".partial").mkdir()
    else:
        metadata.with_name(metadata.name + ".partial").symlink_to(protected)

    with pytest.raises(IntegrityError, match="output|partial|exists"):
        _export(output)

    assert protected.read_bytes() == b"do not modify"
    assert not metadata.exists()


def test_export_never_unlinks_a_partial_replaced_during_publication(
    tmp_path: Path, monkeypatch
) -> None:
    from morai_ddv2 import atomic as atomic_module

    output = tmp_path / "ddv2.ckpt"
    partial = output.with_name(output.name + ".partial")
    metadata = output.with_suffix(".ckpt.metadata.json")
    protected = tmp_path / "protected"
    protected.write_bytes(b"competing partial must survive")
    real_fsync = atomic_module.os.fsync
    calls = 0

    def replacing_fsync(file_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            partial.unlink()
            partial.symlink_to(protected)
        real_fsync(file_descriptor)

    monkeypatch.setattr(atomic_module.os, "fsync", replacing_fsync)

    with pytest.raises(IntegrityError, match="partial.*changed"):
        _export(output)

    assert partial.is_symlink()
    assert partial.resolve() == protected
    assert protected.read_bytes() == b"competing partial must survive"
    assert not output.exists()
    assert not metadata.exists()


def test_export_safely_replaces_stale_regular_partial_files(tmp_path: Path) -> None:
    output = tmp_path / "ddv2.ckpt"
    metadata = output.with_suffix(".ckpt.metadata.json")
    output.with_name(output.name + ".partial").write_bytes(b"stale checkpoint")
    metadata.with_name(metadata.name + ".partial").write_bytes(b"stale metadata")

    bundle = _export(output)

    assert bundle.checkpoint == output
    assert bundle.metadata == metadata
    assert not output.with_name(output.name + ".partial").exists()
    assert not metadata.with_name(metadata.name + ".partial").exists()


def test_reopen_verification_failure_never_publishes_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    from morai_ddv2.export import exporter as exporter_module

    output = tmp_path / "ddv2.ckpt"
    metadata = output.with_suffix(".ckpt.metadata.json")
    real_load = exporter_module.load_raw_sel_checkpoint

    def tampering_load(path: Path):
        container = torch.load(str(path), map_location="cpu")
        container["state_dict"][PREFIX + FROZEN_WEIGHT] = torch.tensor([[99.0]])
        torch.save(container, str(path))
        return real_load(path)

    monkeypatch.setattr(exporter_module, "load_raw_sel_checkpoint", tampering_load)

    with pytest.raises(
        IntegrityError,
        match="changed while being verified|unauthorized state change",
    ):
        _export(output)

    assert output.is_file()
    assert not metadata.exists()


def test_export_rejects_inconsistent_base_container_before_writing(
    tmp_path: Path,
) -> None:
    base = _base()
    inconsistent = RawSelCheckpoint(
        container={"state_dict": _raw_state(), "token": b"base"},
        raw_state_dict=base.raw_state_dict,
        checkpoint_sha256=base.checkpoint_sha256,
    )
    output = tmp_path / "ddv2.ckpt"
    from morai_ddv2.export.exporter import export_sel_bundle

    with pytest.raises(IntegrityError, match="container.*state_dict"):
        export_sel_bundle(
            inconsistent,
            _mutable_state(),
            output,
            identities=_identities(),
            manifest=_manifest(),
        )

    assert not output.exists()
