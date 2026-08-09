from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path

import pytest
import rfc8785
import torch

from morai_ddv2.export.state_hash import state_dict_sha256, state_schema_sha256
from morai_ddv2.training.contracts import TRAINABLE_PREFIXES, FreezeManifest
from morai_ddv2.upstream.ddv2_sel import RawSelCheckpoint


PREFIX = "agent._transfuser_model."
IMAGE_WEIGHT = "_backbone.image_encoder.layer.weight"
IMAGE_RUNNING_MEAN = "_backbone.image_encoder.norm.running_mean"
FROZEN_WEIGHT = "_trajectory_head.weight"
VOCAB = "_trajectory_head.vocab"
MUTABLE_BN_POLICY = "trainable_prefixes:running_mean,running_var,num_batches_tracked"
METADATA_FIELDS = (
    "base_checkpoint_sha256",
    "base_state_dict_sha256",
    "export_checkpoint_sha256",
    "export_state_dict_sha256",
    "raw_schema_sha256",
    "resolved_domain_sha256",
    "sensor_preset_sha256",
    "training_manifest_sha256",
    "cache_schema_version",
    "clock_policy_version",
    "training_config_sha256",
    "domain_id",
    "source_commits",
    "trainable_prefixes",
    "mutable_bn_policy",
    "raw_key_count",
    "runtime_key_count",
    "best_micro_ade",
    "validation_run_id",
    "exported_at_utc",
)


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
        container={"state_dict": raw, "epoch": 7, "token": b"raw-container"},
        raw_state_dict=raw,
        checkpoint_sha256="a" * 64,
    )


def _manifest() -> FreezeManifest:
    return FreezeManifest(
        trainable_parameters=(IMAGE_WEIGHT,),
        frozen_parameters=(FROZEN_WEIGHT,),
        mutable_state_entries=(IMAGE_WEIGHT, IMAGE_RUNNING_MEAN),
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
        source_commits={
            "ddv2": "1" * 40,
            "validator": "2" * 40,
            "preprocessing": "3" * 40,
            "training": "4" * 40,
        },
        best_micro_ade=1.25,
        validation_run_id="morai-ddv2-v1-oracle-command-validation-v1",
    )


def _metadata():
    from morai_ddv2.export.metadata import build_export_metadata

    return build_export_metadata(
        base=_base(),
        checkpoint_sha256="f" * 64,
        state_dict_sha256="0" * 64,
        identities=_identities(),
        manifest=_manifest(),
    )


def test_build_export_metadata_has_exact_authoritative_fields() -> None:
    from morai_ddv2.export.metadata import ExportMetadata

    base = _base()
    metadata = _metadata()

    assert isinstance(metadata, ExportMetadata)
    assert tuple(field.name for field in fields(metadata)) == METADATA_FIELDS
    assert metadata.base_checkpoint_sha256 == base.checkpoint_sha256
    assert metadata.base_state_dict_sha256 == state_dict_sha256(base.raw_state_dict)
    assert metadata.export_checkpoint_sha256 == "f" * 64
    assert metadata.export_state_dict_sha256 == "0" * 64
    assert metadata.raw_schema_sha256 == state_schema_sha256(base.raw_state_dict)
    assert metadata.resolved_domain_sha256 == "b" * 64
    assert metadata.sensor_preset_sha256 == "c" * 64
    assert metadata.training_manifest_sha256 == "d" * 64
    assert metadata.cache_schema_version == 2
    assert metadata.clock_policy_version == 1
    assert metadata.training_config_sha256 == "e" * 64
    assert metadata.domain_id == "morai-ddv2-v1-oracle-command"
    assert dict(metadata.source_commits) == dict(_identities().source_commits)
    assert metadata.trainable_prefixes == TRAINABLE_PREFIXES
    assert metadata.mutable_bn_policy == MUTABLE_BN_POLICY
    assert metadata.raw_key_count == 972
    assert metadata.runtime_key_count == 971
    assert metadata.best_micro_ade == 1.25
    assert metadata.validation_run_id == ("morai-ddv2-v1-oracle-command-validation-v1")
    assert metadata.exported_at_utc.endswith("Z")
    assert "+" not in metadata.exported_at_utc


def test_write_metadata_atomic_publishes_exact_canonical_json(tmp_path: Path) -> None:
    from morai_ddv2.export.metadata import write_metadata_atomic

    metadata = _metadata()
    path = tmp_path / "model.ckpt.metadata.json"

    write_metadata_atomic(path, metadata)

    payload = path.read_bytes()
    document = json.loads(payload)
    expected = asdict(metadata)
    expected["trainable_prefixes"] = list(expected["trainable_prefixes"])
    assert payload == rfc8785.dumps(document)
    assert document == expected
    assert set(document) == set(METADATA_FIELDS)
    assert "metadata_sha256" not in document
    assert hashlib.sha256(payload).hexdigest()
    assert not path.with_name(path.name + ".partial").exists()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"domain_id": ""}, "domain_id"),
        ({"domain_id": "morai-ddv2-v1"}, "domain_id"),
        ({"resolved_domain_sha256": "A" * 64}, "resolved_domain_sha256"),
        ({"cache_schema_version": True}, "cache_schema_version"),
        ({"cache_schema_version": 1}, "cache_schema_version"),
        ({"clock_policy_version": 2}, "clock_policy_version"),
        ({"best_micro_ade": float("nan")}, "best_micro_ade"),
        ({"best_micro_ade": float("inf")}, "best_micro_ade"),
        ({"best_micro_ade": -0.1}, "best_micro_ade"),
        ({"validation_run_id": ""}, "validation_run_id"),
        ({"validation_run_id": "other-validation"}, "validation_run_id"),
        ({"source_commits": {}}, "source_commits"),
        ({"source_commits": {"training": "A" * 40}}, "source_commits"),
    ],
)
def test_export_identities_reject_invalid_trust_values(changes, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(_identities(), **changes)


@pytest.mark.parametrize(
    "timestamp",
    [
        "",
        "2026-08-09T10:11:12",
        "2026-08-09T10:11:12+09:00",
        "2026-08-09 10:11:12Z",
        "2026-02-30T10:11:12Z",
    ],
)
def test_export_metadata_requires_explicit_valid_utc_timestamp(timestamp: str) -> None:
    with pytest.raises((TypeError, ValueError), match="exported_at_utc"):
        replace(_metadata(), exported_at_utc=timestamp)


def test_export_metadata_owns_source_commit_mapping() -> None:
    source_commits = {"training": "1" * 40}
    identities = replace(_identities(), source_commits=source_commits)
    source_commits["training"] = "2" * 40

    assert identities.source_commits["training"] == "1" * 40
    with pytest.raises(TypeError):
        identities.source_commits["training"] = "3" * 40  # type: ignore[index]
