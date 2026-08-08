from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import morai_ddv2.cli.inspect_bags as inspect_module
from morai_ddv2.cli.inspect_bags import inspect_bags
from morai_ddv2.config import ResolvedDomainArtifact, load_resolved_domain
from morai_ddv2.config.models import thaw_json
from morai_ddv2.errors import IntegrityError
from morai_ddv2.ros1 import BagDisposition, BagInventory, BagRejectReason


ROOT = Path(__file__).resolve().parents[3]
DOMAIN_PATH = ROOT / "tests/fixtures/golden/resolved-domain.json"


def _index(path: Path, _domain):
    number = int(path.stem.split("-")[-1])
    disposition = (
        BagDisposition.ELIGIBLE
        if number < 30
        else BagDisposition.SENSOR_ONLY_EXCLUDED
    )
    inventory = BagInventory(
        disposition=disposition,
        rejection_code=None,
        topic_types=(("/camera", "sensor_msgs/CompressedImage"),),
        topic_record_counts=(("/camera", number + 1),),
        camera_decode_error_record_indices=(("/camera", ()),),
        tf_ego_diagnostic=None,
    )
    return SimpleNamespace(
        source_bag_sha256=hashlib.sha256(path.name.encode("utf-8")).hexdigest(),
        inventory=inventory,
    )


def test_inventory_reports_observed_30_10_split_and_excludes_active_bag(
    tmp_path: Path,
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    for index in range(40):
        (bag_root / "drive-{:02d}.bag".format(index)).write_bytes(b"bag")
    (bag_root / "recording.bag.active").write_bytes(b"active")
    report_path = tmp_path / "inventory.json"

    report = inspect_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        report_path=report_path,
        indexer=_index,
    )

    assert report["finalized_bag_count"] == 40
    assert report["active_bag_excluded_count"] == 1
    assert report["disposition_counts"] == {
        "ELIGIBLE": 30,
        "SENSOR_ONLY_EXCLUDED": 10,
        "REJECTED": 0,
    }
    assert len(report["bags"]) == 40
    assert "recording.bag.active" not in {
        item["relative_bag_path"] for item in report["bags"]
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert not Path(str(report_path) + ".partial").exists()


def test_inventory_requires_an_existing_resolved_snapshot(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError):
        inspect_bags(
            bag_root=tmp_path,
            resolved_domain_path=tmp_path / "missing.json",
            report_path=tmp_path / "report.json",
        )


def test_inventory_rejects_a_resolved_snapshot_with_wrong_current_preset_hash(
    tmp_path: Path,
) -> None:
    domain = load_resolved_domain(DOMAIN_PATH)
    document = thaw_json(domain.document)
    document["sensor_preset"]["sha256"] = "0" * 64
    bad = ResolvedDomainArtifact(document=document, sha256="1" * 64)
    # Recompute the self-excluding identity after changing the preset binding.
    from morai_ddv2.config import canonical_json_bytes, sha256_bytes

    bad = ResolvedDomainArtifact(
        document=document,
        sha256=sha256_bytes(canonical_json_bytes(document)),
    )
    path = tmp_path / "wrong-preset.json"
    path.write_bytes(bad.publication_bytes)

    with pytest.raises(IntegrityError, match="preset"):
        inspect_bags(
            bag_root=tmp_path,
            resolved_domain_path=path,
            report_path=tmp_path / "report.json",
        )


def test_inventory_rejects_finalized_alias_to_active_bag(tmp_path: Path) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    active = bag_root / "recording.bag.active"
    active.write_bytes(b"still recording")
    (bag_root / "alias.bag").symlink_to(active)

    with pytest.raises(IntegrityError, match="symlink|active"):
        inspect_bags(
            bag_root=bag_root,
            resolved_domain_path=DOMAIN_PATH,
            report_path=tmp_path / "report.json",
            indexer=_index,
        )


def test_inventory_rejects_duplicate_hardlink_identity(tmp_path: Path) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    original = bag_root / "drive-00.bag"
    original.write_bytes(b"same inode")
    os.link(str(original), str(bag_root / "drive-01.bag"))

    with pytest.raises(IntegrityError, match="duplicate"):
        inspect_bags(
            bag_root=bag_root,
            resolved_domain_path=DOMAIN_PATH,
            report_path=tmp_path / "report.json",
            indexer=_index,
        )


@pytest.mark.parametrize(
    ("disposition", "rejection_code"),
    [
        (BagDisposition.SENSOR_ONLY_EXCLUDED, None),
        (BagDisposition.REJECTED, BagRejectReason.MESSAGE_TYPE_MISMATCH),
    ],
)
def test_inventory_allows_equal_hash_for_distinct_noneligible_bags(
    tmp_path: Path, disposition: BagDisposition, rejection_code
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    for name in ("drive-00.bag", "drive-01.bag"):
        (bag_root / name).write_bytes(b"same noneligible bytes")

    def indexer(path, _domain):
        return SimpleNamespace(
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

    report = inspect_bags(
        bag_root=bag_root,
        resolved_domain_path=DOMAIN_PATH,
        report_path=tmp_path / "inventory.json",
        indexer=indexer,
    )

    assert len(report["bags"]) == 2
    assert all(item["disposition"] == disposition.value for item in report["bags"])


def test_inventory_rejects_equal_hash_for_distinct_eligible_bags(
    tmp_path: Path,
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    for name in ("drive-00.bag", "drive-01.bag"):
        (bag_root / name).write_bytes(b"same eligible bytes")

    def indexer(path, _domain):
        indexed = _index(path, _domain)
        return SimpleNamespace(
            source_bag_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            inventory=indexed.inventory,
        )

    with pytest.raises(IntegrityError, match="duplicate eligible"):
        inspect_bags(
            bag_root=bag_root,
            resolved_domain_path=DOMAIN_PATH,
            report_path=tmp_path / "inventory.json",
            indexer=indexer,
        )


@pytest.mark.parametrize("through_alias", [False, True])
def test_inventory_report_cannot_overwrite_a_source_bag_directly_or_via_alias(
    tmp_path: Path, through_alias: bool
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    source = bag_root / "drive-00.bag"
    source.write_bytes(b"source bag must remain immutable")
    before = source.read_bytes()
    if through_alias:
        alias = tmp_path / "bag-alias"
        alias.symlink_to(bag_root, target_is_directory=True)
        report_path = alias / source.name
    else:
        report_path = source

    with pytest.raises(IntegrityError, match="report|source|bag_root"):
        inspect_bags(
            bag_root=bag_root,
            resolved_domain_path=DOMAIN_PATH,
            report_path=report_path,
            indexer=_index,
        )

    assert source.read_bytes() == before
    assert not Path(str(report_path) + ".partial").exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_inventory_report_partial_link_cannot_modify_a_source_bag(
    tmp_path: Path, link_kind: str
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    source = bag_root / "drive-00.bag"
    source.write_bytes(b"source bag must remain immutable")
    before = (source.read_bytes(), source.stat().st_size, source.stat().st_mtime_ns)
    report_path = tmp_path / "inventory.json"
    partial = Path(str(report_path) + ".partial")
    if link_kind == "symlink":
        partial.symlink_to(source)
    else:
        os.link(str(source), str(partial))

    with pytest.raises(IntegrityError, match="partial|report|bag_root"):
        inspect_bags(
            bag_root=bag_root,
            resolved_domain_path=DOMAIN_PATH,
            report_path=report_path,
            indexer=_index,
        )

    assert (source.read_bytes(), source.stat().st_size, source.stat().st_mtime_ns) == before
    assert not report_path.exists()


@pytest.mark.parametrize(
    "collision",
    [
        "resolved_domain",
        "sensor_preset",
        "resolved_domain_hardlink",
        "sensor_preset_hardlink",
    ],
)
def test_inventory_report_cannot_replace_domain_inputs(
    tmp_path: Path, monkeypatch, collision: str
) -> None:
    bag_root = tmp_path / "bags"
    bag_root.mkdir()
    (bag_root / "drive-00.bag").write_bytes(b"bag")
    project_root = tmp_path / "project"
    project_root.mkdir()
    preset = project_root / "preset.json"
    preset.write_bytes(b"preset must remain immutable")
    resolved_path = tmp_path / "resolved-domain.json"
    resolved_path.write_bytes(b"domain must remain immutable")
    domain = load_resolved_domain(DOMAIN_PATH)
    document = thaw_json(domain.document)
    document["sensor_preset"]["path"] = "preset.json"
    loaded = SimpleNamespace(document=document, sha256=domain.sha256)
    monkeypatch.setattr(inspect_module, "_PROJECT_ROOT", project_root)
    protected = resolved_path if collision.startswith("resolved_domain") else preset
    if collision.endswith("hardlink"):
        target = tmp_path / "hardlinked-report.json"
        os.link(str(protected), str(target))
    else:
        target = protected
    before = protected.read_bytes()

    with pytest.raises(IntegrityError, match="report|input|preset|domain"):
        inspect_bags(
            bag_root=bag_root,
            resolved_domain_path=resolved_path,
            report_path=target,
            domain_loader=lambda _path: loaded,
            indexer=_index,
        )

    assert protected.read_bytes() == before
    assert not Path(str(target) + ".partial").exists()
