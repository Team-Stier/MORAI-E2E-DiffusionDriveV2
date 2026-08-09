from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch

from morai_ddv2.export.state_hash import state_dict_sha256, tensor_data_sha256
from morai_ddv2.training.contracts import FreezeManifest
from morai_ddv2.upstream.ddv2_sel import (
    RawSelCheckpoint,
    load_raw_sel_checkpoint,
    normalize_sel_runtime_state,
)


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
        container=OrderedDict(
            [
                ("epoch", 7),
                ("state_dict", raw),
                ("callbacks", {"best": 1.25}),
                ("token", b"raw-container"),
            ]
        ),
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


def test_export_preserves_raw_container_and_replaces_only_mutable_state(
    tmp_path: Path,
) -> None:
    from morai_ddv2.export.exporter import export_sel_bundle

    base = _base()
    base_keys = tuple(base.container)
    base_state_keys = tuple(base.raw_state_dict)
    vocab_hash = tensor_data_sha256(base.raw_state_dict[PREFIX + VOCAB])
    frozen_hashes = {
        key: tensor_data_sha256(tensor)
        for key, tensor in base.raw_state_dict.items()
        if key not in (PREFIX + IMAGE_WEIGHT, PREFIX + IMAGE_RUNNING_MEAN)
    }
    output = tmp_path / "nested" / "ddv2.ckpt"

    bundle = export_sel_bundle(
        base,
        _mutable_state(),
        output,
        identities=_identities(),
        manifest=_manifest(),
    )

    exported = torch.load(str(output), map_location="cpu")
    raw = exported["state_dict"]
    assert tuple(exported) == base_keys
    assert tuple(raw) == base_state_keys
    assert exported["epoch"] == base.container["epoch"]
    assert exported["callbacks"] == base.container["callbacks"]
    assert exported["token"] == base.container["token"]
    assert set(exported) == set(base.container)
    assert not any("metadata" in key for key in exported)
    assert len(raw) == 972
    assert len(normalize_sel_runtime_state(raw)) == 971
    assert tensor_data_sha256(raw[PREFIX + VOCAB]) == vocab_hash
    assert {key: tensor_data_sha256(raw[key]) for key in frozen_hashes} == frozen_hashes
    assert torch.equal(raw[PREFIX + IMAGE_WEIGHT], torch.tensor([8.0, 9.0]))
    assert torch.equal(raw[PREFIX + IMAGE_RUNNING_MEAN], torch.tensor([7.0]))
    assert base.container["state_dict"] is base.raw_state_dict
    assert torch.equal(
        base.raw_state_dict[PREFIX + IMAGE_WEIGHT], torch.tensor([1.0, 2.0])
    )
    assert bundle.checkpoint == output
    assert bundle.metadata == output.with_suffix(".ckpt.metadata.json")
    assert bundle.checkpoint_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert bundle.state_dict_sha256 == state_dict_sha256(raw)
    metadata = json.loads(bundle.metadata.read_bytes())
    assert metadata["export_checkpoint_sha256"] == bundle.checkpoint_sha256
    assert metadata["export_state_dict_sha256"] == bundle.state_dict_sha256


def test_export_reopens_final_checkpoint_before_metadata_publication(
    tmp_path: Path, monkeypatch
) -> None:
    from morai_ddv2.export import exporter as exporter_module

    output = tmp_path / "ddv2.ckpt"
    events = []
    real_load = exporter_module.load_raw_sel_checkpoint
    real_write_metadata = exporter_module.write_metadata_atomic

    def recording_load(path: Path):
        assert Path(path) == output
        assert output.is_file()
        events.append("reopen")
        return real_load(path)

    def recording_metadata(path, metadata):
        assert events == ["reopen"]
        assert hashlib.sha256(output.read_bytes()).hexdigest() == (
            metadata.export_checkpoint_sha256
        )
        actual = load_raw_sel_checkpoint(output)
        assert state_dict_sha256(actual.raw_state_dict) == (
            metadata.export_state_dict_sha256
        )
        events.append("metadata")
        real_write_metadata(path, metadata)

    monkeypatch.setattr(exporter_module, "load_raw_sel_checkpoint", recording_load)
    monkeypatch.setattr(exporter_module, "write_metadata_atomic", recording_metadata)

    exporter_module.export_sel_bundle(
        _base(),
        _mutable_state(),
        output,
        identities=_identities(),
        manifest=_manifest(),
    )

    assert events == ["reopen", "metadata"]


def test_export_serializes_a_shallow_copy_of_the_top_level_container(
    tmp_path: Path, monkeypatch
) -> None:
    from morai_ddv2.export import exporter as exporter_module

    base = _base()
    real_save = exporter_module.torch.save
    observed = []

    def recording_save(container, stream) -> None:
        assert container is not base.container
        assert container["callbacks"] is base.container["callbacks"]
        assert container["token"] is base.container["token"]
        assert container["state_dict"] is not base.raw_state_dict
        observed.append(True)
        real_save(container, stream)

    monkeypatch.setattr(exporter_module.torch, "save", recording_save)

    exporter_module.export_sel_bundle(
        base,
        _mutable_state(),
        tmp_path / "ddv2.ckpt",
        identities=_identities(),
        manifest=_manifest(),
    )

    assert observed == [True]
