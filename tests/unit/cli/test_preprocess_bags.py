from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import morai_ddv2.cli.preprocess_bags as preprocess_module
from morai_ddv2.cli.preprocess_bags import (
    SourceTreeState,
    preprocess_bags,
    probe_source_tree,
)
from morai_ddv2.config import canonical_json_bytes
from morai_ddv2.errors import ConfigurationError, IntegrityError
from morai_ddv2.preprocess.planner import SampleDropReason
from morai_ddv2.ros1 import BagDisposition, BagInventory, BagRejectReason
from tests.unit.cache.test_hdf5 import SOURCE_COMMIT, _samples
from tests.unit.cache.test_manifest import _diagnostic


ROOT = Path(__file__).resolve().parents[3]
DOMAIN_PATH = ROOT / "tests/fixtures/golden/resolved-domain.json"
ALL_ZERO_DROPS = {reason.value: 0 for reason in SampleDropReason}


def _source(*, dirty: bool) -> SourceTreeState:
    return SourceTreeState(
        source_code_commit=SOURCE_COMMIT,
        dirty=dirty,
        dirty_diff_sha256=("c" * 64 if dirty else None),
    )


def _eligible_index(path: Path, _domain):
    return SimpleNamespace(
        path=path,
        source_bag_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        inventory=BagInventory(
            disposition=BagDisposition.ELIGIBLE,
            rejection_code=None,
            topic_types=(),
            topic_record_counts=(),
            camera_decode_error_record_indices=(),
            tf_ego_diagnostic=_diagnostic(),
        ),
    )


def _plan(index, _domain):
    return SimpleNamespace(samples=(object(),), drops=(), bag_path=index.path)


def _materialize(**_kwargs):
    return iter(_samples())


def _git(cwd: Path, *arguments: str) -> None:
    subprocess.run(
        ["git"] + list(arguments),
        cwd=str(cwd),
        check=True,
        capture_output=True,
    )


def _init_git_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Task12 Test")
    _git(path, "config", "user.email", "task12@example.invalid")
    (path / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(path, "add", "tracked.py")
    _git(path, "commit", "-qm", "initial")


def test_default_preprocessing_refuses_a_dirty_source_tree(tmp_path: Path) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    (bag_root / "drive.bag").write_bytes(b"one")

    with pytest.raises(ConfigurationError, match="dirty"):
        preprocess_bags(
            bag_root=bag_root,
            resolved_domain_path=DOMAIN_PATH,
            cache_root=tmp_path / "cache",
            source_probe=lambda: _source(dirty=True),
        )

    assert not (tmp_path / "cache" / "manifests").exists()


@pytest.mark.parametrize("collision", ["root", "bag", "alias"])
def test_preprocessing_cache_root_cannot_overlap_source_bags(
    tmp_path: Path, collision: str
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    source = bag_root / "drive.bag"
    source.write_bytes(b"source bag must remain immutable")
    before = (source.read_bytes(), source.stat().st_size, source.stat().st_mtime_ns)
    if collision == "root":
        cache_root = bag_root
    elif collision == "bag":
        cache_root = source
    else:
        cache_root = tmp_path / "cache-alias"
        cache_root.symlink_to(bag_root, target_is_directory=True)

    with pytest.raises(IntegrityError, match="cache|source|bag_root"):
        preprocess_bags(
            bag_root=bag_root,
            resolved_domain_path=DOMAIN_PATH,
            cache_root=cache_root,
            source_probe=lambda: _source(dirty=False),
        )

    assert (source.read_bytes(), source.stat().st_size, source.stat().st_mtime_ns) == before


def test_dirty_development_override_is_bound_into_the_manifest(tmp_path: Path) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    (bag_root / "drive.bag").write_bytes(b"one")
    state = _source(dirty=True)

    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        allow_dirty_development=True,
        source_probe=lambda: state,
        indexer=_eligible_index,
        planner=_plan,
        materializer=_materialize,
    )

    assert result.success
    assert result.manifest is not None
    records = [
        json.loads(line)
        for line in (tmp_path / "cache" / result.manifest.path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["dirty_override"] == {
        "allow_dirty_development": True,
        "source_code_commit": SOURCE_COMMIT,
        "dirty_diff_sha256": "c" * 64,
    }


def test_batch_completes_rejection_inventory_and_publishes_no_manifest(
    tmp_path: Path,
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    for name in ("bad.bag", "sensor-only.bag", "valid.bag"):
        (bag_root / name).write_bytes(name.encode("ascii"))

    def indexer(path: Path, domain):
        if path.name == "valid.bag":
            return _eligible_index(path, domain)
        disposition = (
            BagDisposition.REJECTED
            if path.name == "bad.bag"
            else BagDisposition.SENSOR_ONLY_EXCLUDED
        )
        reason = (
            BagRejectReason.MESSAGE_TYPE_MISMATCH
            if disposition is BagDisposition.REJECTED
            else None
        )
        return SimpleNamespace(
            path=path,
            source_bag_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            inventory=BagInventory(
                disposition=disposition,
                rejection_code=reason,
                topic_types=(),
                topic_record_counts=(),
                camera_decode_error_record_indices=(),
                tf_ego_diagnostic=None,
            ),
        )

    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        source_probe=lambda: _source(dirty=False),
        indexer=indexer,
        planner=_plan,
        materializer=_materialize,
    )

    assert not result.success
    assert result.manifest is None
    assert result.rejection_report is not None
    assert [item.relative_bag_path for item in result.rejection_report.bags] == [
        "bad.bag",
        "sensor-only.bag",
        "valid.bag",
    ]
    assert [item.disposition for item in result.rejection_report.bags] == [
        BagDisposition.REJECTED,
        BagDisposition.SENSOR_ONLY_EXCLUDED,
        BagDisposition.ELIGIBLE,
    ]
    assert not tuple((tmp_path / "cache").glob("manifests/*/*/manifest.jsonl"))
    assert result.rejection_report.bags[0].drop_counts == ALL_ZERO_DROPS


def test_preprocessing_records_invalid_bag_path_and_continues_batch(
    tmp_path: Path,
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    active = bag_root / "recording.bag.active"
    active.write_bytes(b"active")
    alias = bag_root / "alias.bag"
    alias.symlink_to(active)
    valid = bag_root / "valid.bag"
    valid.write_bytes(b"valid")
    indexed_paths = []

    def indexer(path, domain):
        indexed_paths.append(path)
        return _eligible_index(path, domain)

    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        source_probe=lambda: _source(dirty=False),
        discoverer=lambda _root: (alias, valid),
        indexer=indexer,
        planner=_plan,
        materializer=_materialize,
    )

    assert not result.success
    assert result.manifest is None
    assert len(result.failures) == 1
    assert result.failures[0].relative_bag_path == "alias.bag"
    assert result.failures[0].operation == "validate_path"
    assert result.failures[0].error_type == "IntegrityError"
    assert indexed_paths == [valid.resolve()]
    assert result.rejection_report is not None
    assert [
        item.relative_bag_path for item in result.rejection_report.bags
    ] == ["valid.bag"]


def test_bag_retargeted_during_index_is_failed_and_remaining_bags_continue(
    tmp_path: Path,
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    changed = bag_root / "a-changed.bag"
    changed.write_bytes(b"original")
    valid = bag_root / "b-valid.bag"
    valid.write_bytes(b"valid")

    def indexer(path, domain):
        indexed = _eligible_index(path, domain)
        if path.name == changed.name:
            replacement = bag_root / "replacement.tmp"
            replacement.write_bytes(b"replacement")
            os.replace(str(replacement), str(path))
        return indexed

    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        source_probe=lambda: _source(dirty=False),
        indexer=indexer,
        planner=_plan,
        materializer=_materialize,
    )

    assert not result.success
    assert result.manifest is None
    assert len(result.failures) == 1
    assert result.failures[0].relative_bag_path == "a-changed.bag"
    assert result.failures[0].operation == "index"
    assert result.failures[0].error_type == "IntegrityError"
    assert result.rejection_report is not None
    assert [
        item.relative_bag_path for item in result.rejection_report.bags
    ] == ["b-valid.bag"]


def test_cli_failure_summary_is_canonical_and_exposes_every_discovered_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    active = bag_root / "recording.bag.active"
    active.write_bytes(b"active")
    alias = bag_root / "alias.bag"
    alias.symlink_to(active)
    valid = bag_root / "valid.bag"
    valid.write_bytes(b"valid")
    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        source_probe=lambda: _source(dirty=False),
        discoverer=lambda _root: (valid, alias),
        indexer=_eligible_index,
        planner=_plan,
        materializer=_materialize,
    )
    monkeypatch.setattr(
        preprocess_module, "preprocess_bags", lambda **_kwargs: result
    )

    exit_code = preprocess_module.main(
        (
            "--bag-root",
            str(bag_root),
            "--resolved-domain",
            str(DOMAIN_PATH),
            "--cache-root",
            str(tmp_path / "unused"),
        )
    )

    output = capsys.readouterr().out.strip()
    summary = json.loads(output)
    assert exit_code == 1
    assert output.encode("utf-8") == canonical_json_bytes(summary)
    assert summary["observed_path_count"] == 2
    assert summary["bags"] == [
        {
            "relative_bag_path": "valid.bag",
            "source_bag_sha256": hashlib.sha256(b"valid").hexdigest(),
            "disposition": "ELIGIBLE",
            "rejection_code": None,
        }
    ]
    assert summary["failures"] == [
        {
            "relative_bag_path": "alias.bag",
            "operation": "validate_path",
            "error_type": "IntegrityError",
        }
    ]
    assert {
        item["relative_bag_path"]
        for item in summary["bags"] + summary["failures"]
    } == {"alias.bag", "valid.bag"}


@pytest.mark.parametrize("failing_stage", ["index", "plan", "materialize"])
def test_stage_exceptions_are_bound_into_complete_rejection_report(
    tmp_path: Path, failing_stage: str
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    (bag_root / "drive.bag").write_bytes(b"one")

    def indexer(path, domain):
        if failing_stage == "index":
            raise RuntimeError("injected index failure")
        return _eligible_index(path, domain)

    def planner(index, domain):
        if failing_stage == "plan":
            raise RuntimeError("injected plan failure")
        return _plan(index, domain)

    def materializer(**kwargs):
        if failing_stage == "materialize":
            raise RuntimeError("injected materialization failure")
        return _materialize(**kwargs)

    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        source_probe=lambda: _source(dirty=False),
        indexer=indexer,
        planner=planner,
        materializer=materializer,
    )

    assert not result.success
    assert result.manifest is None
    assert len(result.failures) == 1
    assert result.failures[0].operation == failing_stage
    assert result.rejection_report is None
    summary = preprocess_module._result_summary(result)
    assert summary["observed_path_count"] == 1
    assert summary["bags"] == []
    assert summary["failures"] == [
        {
            "relative_bag_path": "drive.bag",
            "operation": failing_stage,
            "error_type": "RuntimeError",
        }
    ]
    assert not tuple((tmp_path / "cache").glob("manifests/*/*/manifest.jsonl"))


def test_materializer_close_failure_is_recorded_without_publishing_manifest(
    tmp_path: Path,
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    (bag_root / "drive.bag").write_bytes(b"one")

    class FailingCloseStream:
        drop_records = ()

        def __iter__(self):
            return iter(_samples())

        def close(self):
            raise RuntimeError("injected close failure")

    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        source_probe=lambda: _source(dirty=False),
        indexer=_eligible_index,
        planner=_plan,
        materializer=lambda **_kwargs: FailingCloseStream(),
    )

    assert not result.success
    assert result.manifest is None
    assert len(result.failures) == 1
    assert result.failures[0].operation == "materialize"
    assert result.failures[0].error_type == "RuntimeError"
    assert result.rejection_report is None


def test_duplicate_eligible_hash_is_a_per_bag_failure_and_batch_continues(
    tmp_path: Path,
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    for name, payload in (
        ("a-first.bag", b"duplicate"),
        ("b-duplicate.bag", b"duplicate"),
        ("c-distinct.bag", b"distinct"),
    ):
        (bag_root / name).write_bytes(payload)

    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        source_probe=lambda: _source(dirty=False),
        indexer=_eligible_index,
        planner=_plan,
        materializer=_materialize,
    )

    assert not result.success and result.manifest is None
    assert len(result.failures) == 1
    assert result.failures[0].relative_bag_path == "b-duplicate.bag"
    assert result.failures[0].operation == "validate_identity"
    assert result.failures[0].error_type == "IntegrityError"
    assert result.rejection_report is not None
    assert [
        item.relative_bag_path for item in result.rejection_report.bags
    ] == ["a-first.bag", "c-distinct.bag"]
    assert all(
        item.disposition is BagDisposition.ELIGIBLE
        for item in result.rejection_report.bags
    )
    summary = preprocess_module._result_summary(result)
    assert {
        item["relative_bag_path"]
        for item in summary["bags"] + summary["failures"]
    } == {"a-first.bag", "b-duplicate.bag", "c-distinct.bag"}


@pytest.mark.parametrize(
    ("disposition", "rejection_code"),
    [
        (BagDisposition.SENSOR_ONLY_EXCLUDED, None),
        (BagDisposition.REJECTED, BagRejectReason.MESSAGE_TYPE_MISMATCH),
    ],
)
def test_equal_hash_noneligible_bags_remain_in_complete_report(
    tmp_path: Path, disposition: BagDisposition, rejection_code
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    for name in ("a.bag", "b.bag"):
        (bag_root / name).write_bytes(b"same noneligible bytes")

    def indexer(path, _domain):
        return SimpleNamespace(
            path=path,
            source_bag_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            inventory=BagInventory(
                disposition=disposition,
                rejection_code=rejection_code,
                topic_types=(),
                topic_record_counts=(),
                camera_decode_error_record_indices=(),
                tf_ego_diagnostic=None,
            ),
        )

    result = preprocess_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        cache_root=tmp_path / "cache",
        source_probe=lambda: _source(dirty=False),
        indexer=indexer,
    )

    assert not result.success and result.manifest is None
    assert result.failures == ()
    assert result.rejection_report is not None
    assert [
        item.relative_bag_path for item in result.rejection_report.bags
    ] == ["a.bag", "b.bag"]


def test_real_source_probe_streams_untracked_content_and_mode_into_identity(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _init_git_repo(repository)
    assert probe_source_tree(repository).dirty is False
    untracked = repository / "local-tool.py"
    untracked.write_text("print('one')\n", encoding="utf-8")

    first = probe_source_tree(repository)
    assert first.dirty and first.dirty_diff_sha256 is not None
    assert probe_source_tree(repository) == first
    os.chmod(str(untracked), 0o755)
    executable = probe_source_tree(repository)
    assert executable.dirty_diff_sha256 != first.dirty_diff_sha256
    untracked.write_text("print('two')\n", encoding="utf-8")
    changed = probe_source_tree(repository)
    assert changed.dirty_diff_sha256 != executable.dirty_diff_sha256


def test_real_source_probe_rechecks_a_tree_that_was_initially_clean(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repository"
    _init_git_repo(repository)
    real_status = preprocess_module._git_status
    calls = {"count": 0}

    def changing_status(project_root):
        calls["count"] += 1
        if calls["count"] == 2:
            (repository / "appeared.py").write_text("VALUE = 2\n", encoding="utf-8")
        return real_status(project_root)

    monkeypatch.setattr(preprocess_module, "_git_status", changing_status)

    with pytest.raises(ConfigurationError, match="changed"):
        probe_source_tree(repository)


def test_real_source_probe_rejects_dirty_submodule_even_for_development(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    parent = tmp_path / "parent"
    _init_git_repo(child)
    _init_git_repo(parent)
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(child),
        "third_party/child",
    )
    _git(parent, "commit", "-qam", "add submodule")
    assert probe_source_tree(parent).dirty is False
    (parent / "third_party/child/tracked.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="submodule"):
        probe_source_tree(parent)
