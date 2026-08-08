from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pytest

from morai_ddv2.cache.domain import publish_domain_snapshot
from morai_ddv2.cache.hdf5 import publish_bag_artifacts
from morai_ddv2.cache.manifest import (
    BAG_RECORD_FIELDS,
    DATASET_RECORD_FIELDS,
    DIAGNOSTIC_REPORT_FIELDS,
    SAMPLE_RECORD_FIELDS,
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
    sha256_bytes,
)
from morai_ddv2.config.models import thaw_json
from morai_ddv2.errors import ConfigurationError, IntegrityError
from morai_ddv2.preprocess.diagnostics import TfEgoDiagnostic
from morai_ddv2.preprocess.planner import SampleDropReason
from morai_ddv2.ros1 import (
    BagDisposition,
    BagRejectReason,
    EpochSafeTfEgoDiagnostic,
    TfEgoGateProvenance,
)
from tests.unit.cache.test_hdf5 import (
    RELATIVE_BAG_PATH,
    SOURCE_COMMIT,
    SOURCE_SHA,
    _domain,
    _samples,
)


ALL_ZERO_DROPS = {reason.value: 0 for reason in SampleDropReason}


def _artifact() -> ResolvedDomainArtifact:
    domain = _domain()
    return ResolvedDomainArtifact(document=domain.document, sha256=domain.sha256)


def _diagnostic(
    *,
    comparable_count: int = 10,
    position_p99_m: float = 0.01,
    heading_p99_deg: float = 0.1,
) -> EpochSafeTfEgoDiagnostic:
    comparable_rows = [
        TfEgoGateProvenance(
            ego_header_stamp_ns=100 + 10 * index,
            ego_receipt_stamp_ns=1_000 + 10 * index,
            ego_stable_record_index=index,
            selected_tf_clock_epoch_index=0,
            selected_tf_anchor_record_index=100 + index,
            tf_bracket_header_stamp_ns=(95 + 10 * index, 105 + 10 * index),
            tf_bracket_receipt_ns=(995 + 10 * index, 1_005 + 10 * index),
            tf_bracket_record_index=(200 + 2 * index, 201 + 2 * index),
            exclusion_reason=None,
        )
        for index in range(comparable_count)
    ]
    excluded_rows = [
        TfEgoGateProvenance(
            ego_header_stamp_ns=30,
            ego_receipt_stamp_ns=300,
            ego_stable_record_index=comparable_count + 2,
            selected_tf_clock_epoch_index=1,
            selected_tf_anchor_record_index=13,
            tf_bracket_header_stamp_ns=None,
            tf_bracket_receipt_ns=None,
            tf_bracket_record_index=None,
            exclusion_reason="CROSS_EPOCH_BRACKET_PREVENTED",
        ),
        TfEgoGateProvenance(
            ego_header_stamp_ns=20,
            ego_receipt_stamp_ns=200,
            ego_stable_record_index=comparable_count,
            selected_tf_clock_epoch_index=1,
            selected_tf_anchor_record_index=12,
            tf_bracket_header_stamp_ns=None,
            tf_bracket_receipt_ns=None,
            tf_bracket_record_index=None,
            exclusion_reason="NO_EPOCH_LOCAL_BRACKET",
        ),
        TfEgoGateProvenance(
            ego_header_stamp_ns=40,
            ego_receipt_stamp_ns=400,
            ego_stable_record_index=comparable_count + 1,
            selected_tf_clock_epoch_index=2,
            selected_tf_anchor_record_index=14,
            tf_bracket_header_stamp_ns=None,
            tf_bracket_receipt_ns=None,
            tf_bracket_record_index=None,
            exclusion_reason="AMBIGUOUS_EPOCH_SELECTION",
        ),
    ]
    rows = tuple(reversed(comparable_rows + excluded_rows))
    return EpochSafeTfEgoDiagnostic(
        metric=TfEgoDiagnostic(
            comparable_count,
            3,
            position_p99_m,
            max(position_p99_m, 0.02),
            heading_p99_deg,
            max(heading_p99_deg, 0.2),
        ),
        provenance=rows,
        raw_ego_record_count=comparable_count + 3,
        epoch_local_comparable_count=comparable_count,
        no_epoch_local_bracket_count=1,
        ambiguous_epoch_selection_count=1,
        cross_epoch_bracket_prevented_count=1,
    )


def _publish_fixture(
    cache_root: Path,
    *,
    dirty_override: Optional[DirtyOverride] = None,
) -> ManifestRecord:
    snapshot, artifact, report = _publication_inputs(cache_root)
    return publish_manifest(
        cache_root=cache_root,
        domain_snapshot=snapshot,
        bags=(artifact,),
        splits=assign_bag_splits([RELATIVE_BAG_PATH]),
        rejection_report=report,
        dirty_override=dirty_override,
    )


def _publication_inputs(cache_root: Path):
    snapshot = publish_domain_snapshot(cache_root=cache_root, artifact=_artifact())
    artifact = publish_bag_artifacts(
        cache_root=cache_root,
        domain=_domain(),
        source_bag_sha256=SOURCE_SHA,
        relative_bag_path=RELATIVE_BAG_PATH,
        samples=_samples(),
        source_code_commit=SOURCE_COMMIT,
    )
    excluded_path = "sensor-only/주행.bag"
    rejected_path = "rejected/bad.bag"
    report = RejectionReport(
        bags=(
            BagManifestInput(
                relative_bag_path=RELATIVE_BAG_PATH,
                source_bag_sha256=SOURCE_SHA,
                disposition=BagDisposition.ELIGIBLE,
                artifact=artifact,
                drop_counts=ALL_ZERO_DROPS,
                rejection_code=None,
                tf_ego_diagnostic=_diagnostic(),
            ),
            BagManifestInput(
                relative_bag_path=excluded_path,
                source_bag_sha256="c" * 64,
                disposition=BagDisposition.SENSOR_ONLY_EXCLUDED,
                artifact=None,
                drop_counts=ALL_ZERO_DROPS,
                rejection_code=None,
                tf_ego_diagnostic=None,
            ),
            BagManifestInput(
                relative_bag_path=rejected_path,
                source_bag_sha256="d" * 64,
                disposition=BagDisposition.REJECTED,
                artifact=None,
                drop_counts=ALL_ZERO_DROPS,
                rejection_code=BagRejectReason.MESSAGE_TYPE_MISMATCH,
                tf_ego_diagnostic=None,
            ),
        )
    )
    return snapshot, artifact, report


def _jsonl_records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_task11_public_api_is_explicitly_exported() -> None:
    import morai_ddv2.cache.domain as domain_module
    import morai_ddv2.cache.loader as loader_module
    import morai_ddv2.cache.manifest as manifest_module

    assert set(domain_module.__all__) == {
        "DomainSnapshotRecord",
        "load_domain_snapshot",
        "publish_domain_snapshot",
    }
    assert {
        "BagManifestInput",
        "DirtyOverride",
        "ManifestRecord",
        "RejectionReport",
        "assign_bag_splits",
        "publish_manifest",
        "relative_bag_path",
    }.issubset(set(manifest_module.__all__))
    assert set(loader_module.__all__) == {
        "ModelSample",
        "VerifiedManifest",
        "load_tensor_sample",
        "load_verified_manifest",
    }


def test_snapshot_manifest_and_diagnostic_are_atomically_published_with_exact_schema(
    tmp_path: Path,
) -> None:
    record = _publish_fixture(tmp_path)
    manifest_path = tmp_path / record.path

    assert isinstance(record, ManifestRecord)
    assert manifest_path.is_file()
    assert not Path(str(manifest_path) + ".partial").exists()
    records = _jsonl_records(manifest_path)
    assert tuple(records[0]) == DATASET_RECORD_FIELDS
    assert records[0]["schema_version"] == 2
    assert records[0]["train_relative_bag_paths"] == []
    assert records[0]["validation_relative_bag_paths"] == [RELATIVE_BAG_PATH]
    assert records[0]["dirty_override"] is None

    bag_records = [item for item in records if item["record_type"] == "bag"]
    sample_records = [item for item in records if item["record_type"] == "sample"]
    assert len(bag_records) == 3
    assert all(tuple(item) == BAG_RECORD_FIELDS for item in bag_records)
    assert len(sample_records) == 2
    assert all(tuple(item) == SAMPLE_RECORD_FIELDS for item in sample_records)
    assert [item["hdf5_row_index"] for item in sample_records] == [0, 1]
    assert [item["sidecar_row_index"] for item in sample_records] == [0, 1]
    assert not {"camera_rgb", "lidar_count", "status", "trajectory"}.intersection(
        sample_records[0]
    )

    eligible = next(item for item in bag_records if item["disposition"] == "ELIGIBLE")
    diagnostic_path = tmp_path / eligible["tf_ego_diagnostic_report_path"]
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert tuple(diagnostic) == DIAGNOSTIC_REPORT_FIELDS
    assert [
        item["ego_stable_record_index"] for item in diagnostic["provenance"]
    ] == list(range(13))
    assert diagnostic["no_epoch_local_bracket_count"] == 1
    assert diagnostic["ambiguous_epoch_selection_count"] == 1
    assert diagnostic["cross_epoch_bracket_prevented_count"] == 1
    assert not Path(str(diagnostic_path) + ".partial").exists()


def test_manifest_is_not_published_when_a_referenced_artifact_fails_verification(
    tmp_path: Path,
) -> None:
    snapshot = publish_domain_snapshot(cache_root=tmp_path, artifact=_artifact())
    artifact = publish_bag_artifacts(
        cache_root=tmp_path,
        domain=_domain(),
        source_bag_sha256=SOURCE_SHA,
        relative_bag_path=RELATIVE_BAG_PATH,
        samples=_samples(),
        source_code_commit=SOURCE_COMMIT,
    )
    (tmp_path / artifact.sidecar_path).write_bytes(b"corrupt\n")
    report = RejectionReport(
        bags=(
            BagManifestInput(
                relative_bag_path=RELATIVE_BAG_PATH,
                source_bag_sha256=SOURCE_SHA,
                disposition=BagDisposition.ELIGIBLE,
                artifact=artifact,
                drop_counts=ALL_ZERO_DROPS,
                rejection_code=None,
                tf_ego_diagnostic=_diagnostic(),
            ),
        )
    )
    final = (
        tmp_path
        / "manifests"
        / snapshot.domain_id
        / snapshot.resolved_domain_sha256
        / "manifest.jsonl"
    )

    with pytest.raises(IntegrityError):
        publish_manifest(
            cache_root=tmp_path,
            domain_snapshot=snapshot,
            bags=(artifact,),
            splits=assign_bag_splits([RELATIVE_BAG_PATH]),
            rejection_report=report,
            dirty_override=None,
        )

    assert not final.exists()


def test_manifest_rejects_split_or_artifact_inventory_drift(tmp_path: Path) -> None:
    snapshot = publish_domain_snapshot(cache_root=tmp_path, artifact=_artifact())
    artifact = publish_bag_artifacts(
        cache_root=tmp_path,
        domain=_domain(),
        source_bag_sha256=SOURCE_SHA,
        relative_bag_path=RELATIVE_BAG_PATH,
        samples=_samples(),
        source_code_commit=SOURCE_COMMIT,
    )
    report = RejectionReport(
        bags=(
            BagManifestInput(
                relative_bag_path=RELATIVE_BAG_PATH,
                source_bag_sha256=SOURCE_SHA,
                disposition=BagDisposition.ELIGIBLE,
                artifact=artifact,
                drop_counts=ALL_ZERO_DROPS,
                rejection_code=None,
                tf_ego_diagnostic=_diagnostic(),
            ),
        )
    )

    with pytest.raises(IntegrityError, match="split"):
        publish_manifest(
            cache_root=tmp_path,
            domain_snapshot=snapshot,
            bags=(artifact,),
            splits={RELATIVE_BAG_PATH: "train"},
            rejection_report=report,
            dirty_override=None,
        )
    with pytest.raises(IntegrityError, match="artifact"):
        publish_manifest(
            cache_root=tmp_path,
            domain_snapshot=snapshot,
            bags=(),
            splits=assign_bag_splits([RELATIVE_BAG_PATH]),
            rejection_report=report,
            dirty_override=None,
        )


def test_domain_snapshot_is_immutable_and_rejects_different_existing_bytes(
    tmp_path: Path,
) -> None:
    first = publish_domain_snapshot(cache_root=tmp_path, artifact=_artifact())
    second = publish_domain_snapshot(cache_root=tmp_path, artifact=_artifact())
    assert first == second
    path = tmp_path / first.path
    path.write_bytes(b"changed")

    with pytest.raises(IntegrityError, match="snapshot"):
        publish_domain_snapshot(cache_root=tmp_path, artifact=_artifact())


def test_dirty_override_requires_exact_artifact_source_commit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_code_commit"):
        _publish_fixture(
            tmp_path,
            dirty_override=DirtyOverride(
                allow_dirty_development=True,
                source_code_commit="e" * 40,
                dirty_diff_sha256="f" * 64,
            ),
        )


def test_snapshot_publication_refuses_stale_partial_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    domain = _domain()
    final = (
        tmp_path
        / "domains"
        / domain.document["domain_id"]
        / domain.sha256
        / "resolved-domain.json"
    )
    partial = Path(str(final) + ".partial")
    victim = tmp_path / "victim"
    victim.write_bytes(b"do-not-touch")
    partial.parent.mkdir(parents=True)
    partial.symlink_to(victim)

    with pytest.raises(IntegrityError, match="partial"):
        publish_domain_snapshot(cache_root=tmp_path, artifact=artifact)

    assert victim.read_bytes() == b"do-not-touch"
    assert partial.is_symlink()


def test_manifest_publication_refuses_stale_partial_symlink_without_commit_marker(
    tmp_path: Path,
) -> None:
    snapshot, artifact, report = _publication_inputs(tmp_path)
    final = (
        tmp_path
        / "manifests"
        / snapshot.domain_id
        / snapshot.resolved_domain_sha256
        / "manifest.jsonl"
    )
    partial = Path(str(final) + ".partial")
    victim = tmp_path / "manifest-victim"
    victim.write_bytes(b"do-not-touch")
    partial.parent.mkdir(parents=True)
    partial.symlink_to(victim)

    with pytest.raises(IntegrityError, match="partial"):
        publish_manifest(
            cache_root=tmp_path,
            domain_snapshot=snapshot,
            bags=(artifact,),
            splits=assign_bag_splits([RELATIVE_BAG_PATH]),
            rejection_report=report,
            dirty_override=None,
        )

    assert victim.read_bytes() == b"do-not-touch"
    assert partial.is_symlink()
    assert not final.exists()


@pytest.mark.parametrize("ancestor", ["domains", "diagnostics", "manifests"])
def test_publication_rejects_cache_ancestor_symlink_without_external_write(
    tmp_path: Path, ancestor: str
) -> None:
    cache_root = tmp_path / "cache"
    outside = tmp_path / "outside-{}".format(ancestor)
    cache_root.mkdir()
    outside.mkdir()

    if ancestor == "domains":
        (cache_root / ancestor).symlink_to(outside, target_is_directory=True)

        with pytest.raises(IntegrityError):
            publish_domain_snapshot(cache_root=cache_root, artifact=_artifact())
    else:
        snapshot, artifact, report = _publication_inputs(cache_root)
        (cache_root / ancestor).symlink_to(outside, target_is_directory=True)

        with pytest.raises(IntegrityError):
            publish_manifest(
                cache_root=cache_root,
                domain_snapshot=snapshot,
                bags=(artifact,),
                splits=assign_bag_splits([RELATIVE_BAG_PATH]),
                rejection_report=report,
                dirty_override=None,
            )

        final = (
            cache_root
            / "manifests"
            / snapshot.domain_id
            / snapshot.resolved_domain_sha256
            / "manifest.jsonl"
        )
        assert not final.exists()

    assert list(outside.rglob("*")) == []


@pytest.mark.parametrize("ancestor", ["bags", "commands"])
def test_manifest_rejects_symlinked_artifact_ancestor(
    tmp_path: Path, ancestor: str
) -> None:
    cache_root = tmp_path / "cache"
    snapshot, artifact, report = _publication_inputs(cache_root)
    original = cache_root / ancestor
    relocated = tmp_path / "relocated-{}".format(ancestor)
    original.rename(relocated)
    before = {
        path.relative_to(relocated): path.read_bytes()
        for path in relocated.rglob("*")
        if path.is_file()
    }
    original.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(IntegrityError):
        publish_manifest(
            cache_root=cache_root,
            domain_snapshot=snapshot,
            bags=(artifact,),
            splits=assign_bag_splits([RELATIVE_BAG_PATH]),
            rejection_report=report,
            dirty_override=None,
        )

    after = {
        path.relative_to(relocated): path.read_bytes()
        for path in relocated.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (cache_root / "manifests").exists()


def test_snapshot_rejects_boolean_schema_version_even_when_equal_to_one(
    tmp_path: Path,
) -> None:
    document = thaw_json(_domain().document)
    document["schema_version"] = True
    digest = sha256_bytes(canonical_json_bytes(document))

    with pytest.raises(ConfigurationError):
        publish_domain_snapshot(
            cache_root=tmp_path,
            artifact=ResolvedDomainArtifact(document=document, sha256=digest),
        )


def test_sensor_only_excluded_bag_cannot_claim_gt_diagnostic() -> None:
    with pytest.raises(ValueError, match="sensor-only"):
        BagManifestInput(
            relative_bag_path="sensor-only.bag",
            source_bag_sha256="e" * 64,
            disposition=BagDisposition.SENSOR_ONLY_EXCLUDED,
            artifact=None,
            drop_counts=ALL_ZERO_DROPS,
            rejection_code=None,
            tf_ego_diagnostic=_diagnostic(),
        )


@pytest.mark.parametrize(
    "diagnostic",
    [
        _diagnostic(comparable_count=9),
        _diagnostic(position_p99_m=0.0510001),
        _diagnostic(heading_p99_deg=0.500001),
    ],
)
def test_eligible_bag_cannot_publish_diagnostic_that_fails_domain_gate(
    tmp_path: Path, diagnostic: EpochSafeTfEgoDiagnostic
) -> None:
    snapshot, artifact, report = _publication_inputs(tmp_path)
    eligible = replace(report.bags[0], tf_ego_diagnostic=diagnostic)
    invalid_report = RejectionReport(bags=(eligible,) + report.bags[1:])
    final = (
        tmp_path
        / "manifests"
        / snapshot.domain_id
        / snapshot.resolved_domain_sha256
        / "manifest.jsonl"
    )

    with pytest.raises(IntegrityError, match="TF/Ego"):
        publish_manifest(
            cache_root=tmp_path,
            domain_snapshot=snapshot,
            bags=(artifact,),
            splits=assign_bag_splits([RELATIVE_BAG_PATH]),
            rejection_report=invalid_report,
            dirty_override=None,
        )

    assert not final.exists()


@pytest.mark.parametrize(
    "metric",
    [
        TfEgoDiagnostic(10, 3, 0.04, 0.03, 0.1, 0.2),
        TfEgoDiagnostic(10, 3, 0.01, 0.02, 0.2, 0.1),
    ],
)
def test_diagnostic_p99_cannot_exceed_max_before_final_manifest_publication(
    tmp_path: Path, metric: TfEgoDiagnostic
) -> None:
    snapshot, artifact, report = _publication_inputs(tmp_path)
    invalid_diagnostic = replace(_diagnostic(), metric=metric)
    eligible = replace(report.bags[0], tf_ego_diagnostic=invalid_diagnostic)
    invalid_report = RejectionReport(bags=(eligible,) + report.bags[1:])
    final = (
        tmp_path
        / "manifests"
        / snapshot.domain_id
        / snapshot.resolved_domain_sha256
        / "manifest.jsonl"
    )

    with pytest.raises(IntegrityError, match="p99"):
        publish_manifest(
            cache_root=tmp_path,
            domain_snapshot=snapshot,
            bags=(artifact,),
            splits=assign_bag_splits([RELATIVE_BAG_PATH]),
            rejection_report=invalid_report,
            dirty_override=None,
        )

    assert not final.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ego_header_stamp_ns", -1),
        ("ego_receipt_stamp_ns", -1),
        ("tf_bracket_header_stamp_ns", (-1, 105)),
        ("tf_bracket_receipt_ns", (-1, 1_005)),
        ("tf_bracket_header_stamp_ns", (105, 95)),
        ("tf_bracket_receipt_ns", (1_005, 995)),
        ("tf_bracket_record_index", (201, 200)),
    ],
)
def test_invalid_diagnostic_provenance_cannot_publish_final_manifest(
    tmp_path: Path, field: str, value
) -> None:
    snapshot, artifact, report = _publication_inputs(tmp_path)
    diagnostic = _diagnostic()
    comparable = next(
        item for item in diagnostic.provenance if item.exclusion_reason is None
    )
    invalid_row = replace(comparable, **{field: value})
    invalid_provenance = tuple(
        invalid_row if item is comparable else item for item in diagnostic.provenance
    )
    invalid_diagnostic = replace(diagnostic, provenance=invalid_provenance)
    eligible = replace(report.bags[0], tf_ego_diagnostic=invalid_diagnostic)
    invalid_report = RejectionReport(bags=(eligible,) + report.bags[1:])
    final = (
        tmp_path
        / "manifests"
        / snapshot.domain_id
        / snapshot.resolved_domain_sha256
        / "manifest.jsonl"
    )

    with pytest.raises(IntegrityError, match="TF/Ego"):
        publish_manifest(
            cache_root=tmp_path,
            domain_snapshot=snapshot,
            bags=(artifact,),
            splits=assign_bag_splits([RELATIVE_BAG_PATH]),
            rejection_report=invalid_report,
            dirty_override=None,
        )

    assert not final.exists()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("split", "train_fraction"), 0.5),
        (("split", "unit"), "frame"),
        (("split", "ordering"), ["relative_path_utf8"]),
        (("split", "relative_path_separator"), "\\"),
        (("cache", "manifest_path_template"), "manifests/{domain_id}/wrong.jsonl"),
    ],
)
def test_snapshot_rejects_rehashed_split_or_manifest_path_contract_drift(
    tmp_path: Path, path, value
) -> None:
    document = thaw_json(_domain().document)
    target = document
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    digest = sha256_bytes(canonical_json_bytes(document))

    with pytest.raises(ConfigurationError):
        publish_domain_snapshot(
            cache_root=tmp_path,
            artifact=ResolvedDomainArtifact(document=document, sha256=digest),
        )
