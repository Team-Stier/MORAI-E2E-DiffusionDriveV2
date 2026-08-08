from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import pytest

import morai_ddv2.cache.hdf5 as hdf5_module
from morai_ddv2.cache import CACHE_DATASET_NAMES
from morai_ddv2.cache.hdf5 import publish_bag_artifacts, verify_bag_artifact
from morai_ddv2.cache.sidecar import read_command_sidecar
from morai_ddv2.config import sha256_file
from morai_ddv2.errors import IntegrityError
from tests.unit.cache.test_hdf5 import (
    CloseTrackingIterator,
    _expected_paths,
    _identity,
    _publish_args,
    _samples,
)


def test_task10_publication_entrypoint_is_importable() -> None:
    assert callable(publish_bag_artifacts)


def _raise_io(*args, **kwargs):
    del args, kwargs
    raise OSError("injected failure")


def _fail_on_call(monkeypatch, name, target_call):
    original = getattr(hdf5_module, name)
    calls = {"count": 0}

    def wrapper(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == target_call:
            raise OSError("injected {} failure".format(name))
        return original(*args, **kwargs)

    monkeypatch.setattr(hdf5_module, name, wrapper)


@pytest.mark.parametrize(
    ("fault", "expected_finals"),
    [
        ("hdf5_write", (False, False)),
        ("sidecar_write", (False, False)),
        ("hdf5_file_fsync", (False, False)),
        ("sidecar_file_fsync", (False, False)),
        ("partial_verify", (False, False)),
        ("hdf5_replace", (False, False)),
        ("hdf5_parent_fsync", (True, False)),
        ("sidecar_replace", (True, False)),
        ("sidecar_parent_fsync", (True, True)),
    ],
)
def test_injected_failures_never_expose_incomplete_final(
    tmp_path, monkeypatch, fault, expected_finals
) -> None:
    if fault == "hdf5_write":
        monkeypatch.setattr(hdf5_module, "_append_hdf5_sample", _raise_io)
    elif fault == "sidecar_write":
        monkeypatch.setattr(hdf5_module, "_write_sidecar_record", _raise_io)
    elif fault == "hdf5_file_fsync":
        _fail_on_call(monkeypatch, "_fsync_regular_file", 1)
    elif fault == "sidecar_file_fsync":
        _fail_on_call(monkeypatch, "_fsync_regular_file", 2)
    elif fault == "partial_verify":
        monkeypatch.setattr(hdf5_module, "_verify_artifact_pair", _raise_io)
    elif fault == "hdf5_replace":
        _fail_on_call(monkeypatch, "_replace_file", 1)
    elif fault == "hdf5_parent_fsync":
        _fail_on_call(monkeypatch, "_fsync_parent_directory", 1)
    elif fault == "sidecar_replace":
        _fail_on_call(monkeypatch, "_replace_file", 2)
    else:
        _fail_on_call(monkeypatch, "_fsync_parent_directory", 2)

    stream = CloseTrackingIterator(_samples())
    with pytest.raises(OSError):
        publish_bag_artifacts(**_publish_args(tmp_path, stream))
    assert stream.close_calls == 1

    hdf5_path, sidecar_path = _expected_paths(tmp_path)
    assert (hdf5_path.exists(), sidecar_path.exists()) == expected_finals
    if hdf5_path.exists():
        assert len(sha256_file(hdf5_path)) == 64
        with h5py.File(str(hdf5_path), "r") as handle:
            assert set(handle) == set(CACHE_DATASET_NAMES)
            assert all(handle[name].shape[0] == 2 for name in handle)
    if sidecar_path.exists():
        assert len(sha256_file(sidecar_path)) == 64
        assert len(read_command_sidecar(sidecar_path, _identity())) == 2


@pytest.mark.parametrize(
    "existing", ["hdf5", "sidecar", "both", "dangling_hdf5", "dangling_sidecar"]
)
def test_existing_final_is_never_overwritten(tmp_path, existing) -> None:
    hdf5_path, sidecar_path = _expected_paths(tmp_path)
    sentinels = {}
    if existing in ("hdf5", "both"):
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        hdf5_path.write_bytes(b"existing-hdf5")
        sentinels[hdf5_path] = hdf5_path.read_bytes()
    if existing in ("sidecar", "both"):
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_bytes(b"existing-sidecar")
        sentinels[sidecar_path] = sidecar_path.read_bytes()
    dangling = None
    if existing == "dangling_hdf5":
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        hdf5_path.symlink_to("missing-target")
        dangling = hdf5_path
    elif existing == "dangling_sidecar":
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.symlink_to("missing-target")
        dangling = sidecar_path

    with pytest.raises(FileExistsError):
        publish_bag_artifacts(**_publish_args(tmp_path))

    assert {path: path.read_bytes() for path in sentinels} == sentinels
    if dangling is not None:
        assert dangling.is_symlink()
        assert os.readlink(str(dangling)) == "missing-target"


def test_stale_partials_are_rebuilt_when_no_final_exists(tmp_path) -> None:
    hdf5_path, sidecar_path = _expected_paths(tmp_path)
    hdf5_partial = Path(str(hdf5_path) + ".partial")
    sidecar_partial = Path(str(sidecar_path) + ".partial")
    hdf5_partial.parent.mkdir(parents=True, exist_ok=True)
    sidecar_partial.parent.mkdir(parents=True, exist_ok=True)
    hdf5_partial.write_bytes(b"stale-hdf5")
    sidecar_partial.write_bytes(b"stale-sidecar")

    record = publish_bag_artifacts(**_publish_args(tmp_path))

    assert (tmp_path / record.hdf5_path).is_file()
    assert (tmp_path / record.sidecar_path).is_file()
    assert not hdf5_partial.exists()
    assert not sidecar_partial.exists()


@pytest.mark.parametrize("partial_kind", ["hdf5", "sidecar"])
def test_stale_partial_symlink_is_rejected_without_following_target(
    tmp_path, partial_kind
) -> None:
    hdf5_path, sidecar_path = _expected_paths(tmp_path)
    hdf5_partial = Path(str(hdf5_path) + ".partial")
    sidecar_partial = Path(str(sidecar_path) + ".partial")
    partial = hdf5_partial if partial_kind == "hdf5" else sidecar_partial
    target = tmp_path / "outside-target"
    target.write_bytes(b"do-not-touch")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.symlink_to(target)

    with pytest.raises(IntegrityError):
        publish_bag_artifacts(**_publish_args(tmp_path))

    assert target.read_bytes() == b"do-not-touch"
    assert partial.is_symlink()


def test_verifier_rejects_path_swap_between_hash_and_structural_open(
    tmp_path, monkeypatch
) -> None:
    record = publish_bag_artifacts(**_publish_args(tmp_path))
    hdf5_path = tmp_path / record.hdf5_path
    replacement = tmp_path / "replacement.h5"
    shutil.copyfile(str(hdf5_path), str(replacement))
    with h5py.File(str(replacement), "r+") as handle:
        handle["camera_rgb"][0, 0, 0, 0] ^= 1
    assert sha256_file(replacement) != record.hdf5_sha256
    original_sha256_file = hdf5_module.sha256_file
    swapped = {"done": False}

    def swap_after_hash(path, *args, **kwargs):
        digest = original_sha256_file(path, *args, **kwargs)
        if Path(path) == hdf5_path and not swapped["done"]:
            os.replace(str(replacement), str(hdf5_path))
            swapped["done"] = True
        return digest

    monkeypatch.setattr(hdf5_module, "sha256_file", swap_after_hash)

    with pytest.raises(IntegrityError):
        verify_bag_artifact(record, tmp_path)
