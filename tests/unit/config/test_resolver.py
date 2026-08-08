from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from morai_ddv2 import atomic as atomic_module
from morai_ddv2.cli.resolve_domain import main as resolve_domain_main
from morai_ddv2.config.canonical import canonical_json_bytes, sha256_bytes
from morai_ddv2.config.models import ResolvedDomainArtifact
from morai_ddv2.config.preset import load_sensor_preset
from morai_ddv2.config.resolver import (
    load_resolved_domain,
    publish_resolved_domain,
    resolve_domain,
)
from morai_ddv2.errors import ConfigurationError, IntegrityError


EXPECTED_PRESET_SHA256 = "b7e69c70cc0b2948c3a5b352eee022be04f8884a1df135697f89bb4bb88263fb"


def _resolve(
    repo_root: Path, source_commit: str = "1" * 40
) -> ResolvedDomainArtifact:
    preset = load_sensor_preset(
        repo_root / "sensor_data" / "0725demo_ros.json",
        EXPECTED_PRESET_SHA256,
    )
    raw_domain = yaml.safe_load(
        (repo_root / "configs" / "domains" / "morai-ddv2-v1-oracle-command.yaml").read_text(
            encoding="utf-8"
        )
    )
    return resolve_domain(
        preset=preset,
        raw_domain=raw_domain,
        source_commit=source_commit,
    )


def test_domain_yaml_contains_policy_but_not_preset_owned_sensor_values(repo_root) -> None:
    raw_domain = yaml.safe_load(
        (repo_root / "configs" / "domains" / "morai-ddv2-v1-oracle-command.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert "sensors" not in raw_domain
    assert raw_domain["ros_contract"]["message_types"] == {
        "camera": "sensor_msgs/CompressedImage",
        "lidar": "sensor_msgs/PointCloud2",
        "ego_state": "morai_msgs/EgoVehicleStatus",
        "transforms": "tf2_msgs/TFMessage",
    }
    sampling = raw_domain["sampling"]
    assert "timestamp_source" not in sampling
    assert sampling["sample_rate_hz"] == 2.0
    assert sampling["grid_period_ns"] == 500_000_000
    assert sampling["clock_policy"] == {
        "version": 1,
        "receipt_clock_semantics": "recorder_observation_order_not_acquisition",
        "grid": {
            "axis": "receipt_stamp_ns",
            "origin": "ceil_first_valid_front_receipt_to_period",
        },
        "front": {
            "selection_axis": "receipt_stamp_ns",
            "tolerance_ns": 50_000_000,
            "tie_break_order": [
                "absolute_receipt_offset_ns",
                "receipt_stamp_ns",
                "stable_record_index",
            ],
            "selected_header_identity": {
                "receipt_grid_order": "strictly_increasing",
                "unique": True,
                "stable_record_reuse_allowed": False,
                "violation_bag_reject_reason": "FRONT_TIMESTAMP_ORDER_VIOLATION",
            },
        },
        "side_camera": {
            "candidate_filters": [
                {
                    "axis": "header_stamp_ns",
                    "relative_to": "selected_front_header_stamp_ns",
                    "absolute_tolerance_ns": 30_000_000,
                },
                {
                    "axis": "receipt_stamp_ns",
                    "relative_to": "selected_front_receipt_stamp_ns",
                    "absolute_tolerance_ns": 100_000_000,
                },
            ],
            "selection_axis": "header_stamp_ns",
            "tie_break_order": [
                "absolute_header_offset_ns",
                "header_stamp_ns",
                "stable_record_index",
            ],
        },
        "lidar": {
            "selection_axis": "receipt_stamp_ns",
            "tolerance_ns": 45_000_000,
            "tie_break_order": [
                "absolute_receipt_offset_ns",
                "receipt_stamp_ns",
                "stable_record_index",
            ],
        },
        "trajectory_anchor_tf": {
            "source_view": "raw_map_to_base_link_records",
            "selection_axis": "receipt_stamp_ns",
            "tolerance_ns": 50_000_000,
            "tie_break_order": [
                "absolute_receipt_offset_ns",
                "receipt_stamp_ns",
                "stable_record_index",
            ],
            "current_pose": "exact_selected_raw_record",
            "preserve_original_stable_record_index": True,
            "fallback_to_second_candidate": False,
        },
        "ego_state": {
            "interpolation_axis": "header_stamp_ns",
            "target": "trajectory_anchor_tf_header_ns",
            "max_bracket_ns": 100_000_000,
            "clock_epoch": {
                "index_origin": 0,
                "source_view": "raw_ego_records",
                "ordering": ["receipt_stamp_ns", "stable_record_index"],
                "new_epoch_condition": "header_stamp_ns_strictly_decreases",
                "selector_relative_to": "trajectory_anchor_tf_receipt_ns",
                "selector_axis": "receipt_stamp_ns",
                "selector_tolerance": "none",
                "selector_tie_break_order": [
                    "absolute_receipt_offset_ns",
                    "receipt_stamp_ns",
                    "stable_record_index",
                ],
                "interpolation_view": "selected_epoch_collapsed_header_view",
                "equal_header_duplicate_policy": (
                    "collapse_equivalent_earliest_in_epoch_reject_conflict"
                ),
                "fallback_to_other_epoch": False,
            },
        },
        "future_pose": {
            "interpolation_axis": "header_stamp_ns",
            "target_origin": "trajectory_anchor_tf_header_ns",
            "max_bracket_ns": 100_000_000,
            "source_view": "selected_raw_anchor_same_epoch_causal_suffix",
            "equal_header_duplicate_policy": (
                "collapse_equivalent_earliest_at_or_after_anchor_reject_conflict"
            ),
            "extrapolation_allowed": False,
            "fallback_to_other_epoch": False,
        },
        "tf_clock_epoch": {
            "index_origin": 0,
            "source_view": "raw_map_to_base_link_records",
            "ordering": ["receipt_stamp_ns", "stable_record_index"],
            "monotonic_header_rule": "nondecreasing",
            "new_epoch_condition": "header_stamp_ns_strictly_decreases",
            "within_epoch_interpolation_view": "collapsed_header_view",
            "equal_header_duplicate_policy": (
                "collapse_equivalent_earliest_in_view_reject_conflict"
            ),
            "same_header_across_epochs": "allowed_epoch_index_disambiguates",
        },
        "tf_ego_bag_gate": {
            "axis": "native_header_stamp_ns",
            "scope": "full_stream",
            "ego_source_view": "raw_ego_records",
            "tf_epoch_selector": {
                "source_view": "raw_map_to_base_link_records",
                "relative_to": "ego_receipt_stamp_ns",
                "selection_axis": "receipt_stamp_ns",
                "tolerance": "none",
                "tie_break_order": [
                    "absolute_receipt_offset_ns",
                    "receipt_stamp_ns",
                    "stable_record_index",
                ],
            },
            "interpolation_view": "selected_tf_epoch_collapsed_header_view",
            "max_bracket_ns": 100_000_000,
            "fallback_to_other_epoch": False,
            "excluded_case_counters": [
                "no_epoch_local_bracket",
                "ambiguous_epoch_selection",
                "cross_epoch_bracket_prevented",
            ],
        },
        "tolerance_boundary_inclusive": True,
        "interpolation_exact_match_weight": 0.0,
    }
    assert raw_domain["camera_processing"]["output_wh"] == [1024, 256]
    assert raw_domain["lidar_processing"]["bins_xy"] == [256, 256]
    assert raw_domain["trajectory"]["sampling_offsets_ns"][-1] == 4_000_000_000
    assert raw_domain["command"]["ordered_rules"] == [
        "DROP_INVALID_TRAJECTORY",
        "UNKNOWN_STATIC",
        "LEFT",
        "RIGHT",
        "STRAIGHT",
        "UNKNOWN_AMBIGUOUS",
    ]
    assert raw_domain["domain_lifecycle"] == {
        "contract_revision_context": "pre_first_production_unpublished",
        "after_first_production_publication": "new_domain_id_required",
        "breaking_change_kinds": ["clock_policy", "cache_schema"],
    }
    assert raw_domain["diagnostics"]["tf_ego_epoch_safe_report"] == {
        "provenance_fields": [
            "ego_header_stamp_ns",
            "ego_receipt_stamp_ns",
            "ego_stable_record_index",
            "selected_tf_clock_epoch_index",
            "selected_tf_anchor_record_index",
            "tf_bracket_header_stamp_ns",
            "tf_bracket_receipt_ns",
            "tf_bracket_record_index",
        ],
        "counters": [
            "raw_ego_record_count",
            "epoch_local_comparable_count",
            "no_epoch_local_bracket_count",
            "ambiguous_epoch_selection_count",
            "cross_epoch_bracket_prevented_count",
        ],
    }
    cache = raw_domain["cache"]
    assert cache["schema_version"] == 2
    assert cache["command_sidecar_schema_version"] == 2
    assert cache["manifest_schema_version"] == 2
    assert cache["hdf5_attributes"] == {
        "cache_schema_version": 2,
        "clock_policy_version": 1,
    }
    assert cache["datasets"] == {
        "camera_rgb": {"dtype": "uint8", "shape": ["N", 256, 1024, 3]},
        "lidar_count": {"dtype": "uint8", "shape": ["N", 256, 256]},
        "status": {"dtype": "float32", "shape": ["N", 8]},
        "trajectory": {"dtype": "float32", "shape": ["N", 8, 3]},
        "timestamp_ns": {"dtype": "int64", "shape": ["N"]},
        "grid_receipt_ns": {"dtype": "int64", "shape": ["N"]},
        "sensor_source_header_stamp_ns": {"dtype": "int64", "shape": ["N", 4]},
        "sensor_source_receipt_ns": {"dtype": "int64", "shape": ["N", 4]},
        "sensor_source_record_index": {"dtype": "int64", "shape": ["N", 4]},
        "side_camera_header_offset_ns": {"dtype": "int64", "shape": ["N", 2]},
        "sensor_receipt_offset_ns": {"dtype": "int64", "shape": ["N", 3]},
        "trajectory_anchor_tf_header_ns": {"dtype": "int64", "shape": ["N"]},
        "trajectory_anchor_tf_receipt_ns": {"dtype": "int64", "shape": ["N"]},
        "trajectory_anchor_tf_record_index": {"dtype": "int64", "shape": ["N"]},
        "trajectory_anchor_tf_clock_epoch_index": {"dtype": "int64", "shape": ["N"]},
        "ego_bracket_clock_epoch_index": {"dtype": "int64", "shape": ["N"]},
        "ego_bracket_header_stamp_ns": {"dtype": "int64", "shape": ["N", 2]},
        "ego_bracket_receipt_ns": {"dtype": "int64", "shape": ["N", 2]},
        "ego_bracket_record_index": {"dtype": "int64", "shape": ["N", 2]},
        "ego_interp_weight": {"dtype": "float32", "shape": ["N"]},
        "current_pose_bracket_header_stamp_ns": {
            "dtype": "int64",
            "shape": ["N", 2],
        },
        "current_pose_bracket_receipt_ns": {"dtype": "int64", "shape": ["N", 2]},
        "current_pose_bracket_record_index": {"dtype": "int64", "shape": ["N", 2]},
        "current_pose_interp_weight": {"dtype": "float32", "shape": ["N"]},
        "future_pose_bracket_header_stamp_ns": {
            "dtype": "int64",
            "shape": ["N", 8, 2],
        },
        "future_pose_bracket_receipt_ns": {"dtype": "int64", "shape": ["N", 8, 2]},
        "future_pose_bracket_record_index": {
            "dtype": "int64",
            "shape": ["N", 8, 2],
        },
        "future_pose_interp_weight": {"dtype": "float32", "shape": ["N", 8]},
    }
    assert cache["command_record_fields"] == [
        "schema_version",
        "domain_id",
        "resolved_domain_sha256",
        "sensor_preset_sha256",
        "source_bag_sha256",
        "command_rule_sha256",
        "timestamp_ns",
        "grid_receipt_ns",
        "front_receipt_ns",
        "front_record_index",
        "command_index",
        "command_name",
        "provenance",
    ]
    assert len(cache["datasets"]) == 28
    assert len(cache["datasets"]) - 4 == 24
    assert len(cache["command_record_fields"]) == 13
    assert "FRONT_TIMESTAMP_ORDER_VIOLATION" in raw_domain["validity"][
        "bag_reject_reasons"
    ]
    assert raw_domain["split"]["seed_prefix"] == "20260806:"


def test_resolver_derives_physical_sensor_contract_only_from_preset(repo_root) -> None:
    artifact = _resolve(repo_root)
    document = artifact.document

    assert document["domain_id"] == "morai-ddv2-v1-oracle-command"
    assert document["source_code_commit"] == "1" * 40
    assert document["sensor_preset"]["sha256"] == EXPECTED_PRESET_SHA256
    assert document["sensors"]["cameras"][0]["topic"] == "/image_jpeg/front/compressed"
    assert document["sensors"]["cameras"][0]["resolution_wh"] == [1280, 720]
    assert document["sensors"]["lidar"]["frame_id"] == "Lidar3D-4"
    assert document["calibration_notes"]["focal_length_pixel"]["used_for_projection"] is False
    assert document["calibration_notes"]["focal_length_pixel"]["inconsistent_with_fov"] is True


def test_resolved_domain_hash_excludes_only_its_own_digest(repo_root) -> None:
    artifact = _resolve(repo_root)

    assert "resolved_domain_sha256" not in artifact.document
    assert sha256_bytes(canonical_json_bytes(artifact.document)) == artifact.sha256
    published = json.loads(artifact.publication_bytes)
    assert canonical_json_bytes(published) == artifact.publication_bytes
    assert published["resolved_domain_sha256"] == artifact.sha256
    del published["resolved_domain_sha256"]
    assert published == artifact.document


def test_resolved_domain_matches_clean_head_golden_bytes(repo_root) -> None:
    golden_path = repo_root / "tests" / "fixtures" / "golden" / "resolved-domain.json"
    golden_bytes = golden_path.read_bytes()
    golden_document = json.loads(golden_bytes)

    artifact = _resolve(repo_root, source_commit=golden_document["source_code_commit"])

    assert artifact.publication_bytes == golden_bytes


def test_resolved_domain_document_is_deeply_immutable(repo_root) -> None:
    artifact = _resolve(repo_root)

    with pytest.raises(TypeError):
        artifact.document["domain_id"] = "changed"
    with pytest.raises(TypeError):
        artifact.document["sensors"]["cameras"].append({})


def test_resolver_rejects_non_commit_source_identity(repo_root) -> None:
    preset = load_sensor_preset(
        repo_root / "sensor_data" / "0725demo_ros.json",
        EXPECTED_PRESET_SHA256,
    )
    raw_domain = yaml.safe_load(
        (repo_root / "configs" / "domains" / "morai-ddv2-v1-oracle-command.yaml").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(ConfigurationError, match="40-character lowercase hexadecimal"):
        resolve_domain(preset=preset, raw_domain=raw_domain, source_commit="dirty")


def test_publish_and_load_verify_document_and_source_preset(repo_root, tmp_path) -> None:
    artifact = _resolve(repo_root)
    output_path = tmp_path / "resolved-domain.json"

    publish_resolved_domain(artifact, output_path)
    loaded = load_resolved_domain(output_path)

    assert loaded.sha256 == artifact.sha256
    assert loaded.document == artifact.document
    assert not output_path.with_name(output_path.name + ".partial").exists()


def test_publish_fsyncs_file_then_replaces_sibling_then_fsyncs_parent(
    repo_root, tmp_path, monkeypatch
) -> None:
    artifact = _resolve(repo_root)
    output_path = tmp_path / "nested" / "resolved-domain.json"
    expected_partial = output_path.with_name(output_path.name + ".partial")
    events = []
    real_fsync = atomic_module.os.fsync
    real_replace = atomic_module.os.replace

    def recording_fsync(file_descriptor):
        events.append(("fsync", file_descriptor))
        real_fsync(file_descriptor)

    def recording_replace(source, destination):
        events.append(("replace", Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(atomic_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(atomic_module.os, "replace", recording_replace)

    publish_resolved_domain(artifact, output_path)

    assert [event[0] for event in events] == ["fsync", "replace", "fsync"]
    assert events[1] == ("replace", expected_partial, output_path)
    assert output_path.read_bytes() == artifact.publication_bytes


def test_load_rejects_tampered_resolved_document(repo_root, tmp_path) -> None:
    artifact = _resolve(repo_root)
    output_path = tmp_path / "resolved-domain.json"
    publish_resolved_domain(artifact, output_path)
    tampered = json.loads(output_path.read_bytes())
    tampered["sampling"]["sample_rate_hz"] = 4.0
    output_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(IntegrityError, match="resolved domain SHA-256"):
        load_resolved_domain(output_path)


def test_load_rejects_current_source_preset_hash_mismatch(repo_root, tmp_path) -> None:
    artifact = _resolve(repo_root)
    changed_document = dict(artifact.document)
    changed_document["sensor_preset"] = dict(changed_document["sensor_preset"])
    changed_document["sensor_preset"]["sha256"] = "0" * 64
    changed_hash = sha256_bytes(canonical_json_bytes(changed_document))
    changed_artifact = ResolvedDomainArtifact(document=changed_document, sha256=changed_hash)
    output_path = tmp_path / "resolved-domain.json"
    publish_resolved_domain(changed_artifact, output_path)

    with pytest.raises(IntegrityError, match="current sensor preset SHA-256"):
        load_resolved_domain(output_path)


def test_cli_resolves_explicit_inputs_without_writing_the_preset(repo_root, tmp_path) -> None:
    preset_path = repo_root / "sensor_data" / "0725demo_ros.json"
    preset_before = preset_path.read_bytes()
    output_path = tmp_path / "resolved-domain.json"

    exit_code = resolve_domain_main(
        [
            "--domain-yaml",
            str(repo_root / "configs" / "domains" / "morai-ddv2-v1-oracle-command.yaml"),
            "--sensor-preset",
            str(preset_path),
            "--output",
            str(output_path),
            "--source-commit",
            "2" * 40,
        ]
    )

    assert exit_code == 0
    assert load_resolved_domain(output_path).document["source_code_commit"] == "2" * 40
    assert preset_path.read_bytes() == preset_before
