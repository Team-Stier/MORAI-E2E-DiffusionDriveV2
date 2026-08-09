from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from morai_ddv2.config import canonical_json_bytes, load_resolved_domain, sha256_file
from morai_ddv2.errors import IntegrityError
from morai_ddv2.export.exporter import ExportBundle
from morai_ddv2.export.metadata import (
    MUTABLE_BN_POLICY,
    ExportMetadata,
    write_metadata_atomic,
)
from morai_ddv2.export.state_hash import state_dict_sha256, state_schema_sha256
from morai_ddv2.training.contracts import TRAINABLE_PREFIXES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_ROOT = PROJECT_ROOT / "third_party/E2E-OpenLoop-ROS-Validator"
BASELINE_PROFILE = VALIDATOR_ROOT / "src/Validator_pkg/config/profile.json"
RESOLVED_DOMAIN = PROJECT_ROOT / "tests/fixtures/golden/resolved-domain.json"
SENSOR_PRESET = PROJECT_ROOT / "sensor_data/0725demo_ros.json"
PREFIX = "agent._transfuser_model."
DDV2_COMMIT = "1cd12a1e155c34dcc471261835444c8d5587580b"
VALIDATOR_COMMIT = "09a3b88e19ae3354903ed676a35e02416284cb58"


class _SchemaNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {f"layer_{index:03d}": nn.Linear(1, 1, bias=False) for index in range(971)}
        )
        for layer in self.layers.values():
            nn.init.zeros_(layer.weight)


def _raw_state(*, renamed: bool = False, wrong_shape: bool = False):
    values = OrderedDict()
    for index in range(971):
        name = f"layers.layer_{index:03d}.weight"
        if renamed and index == 970:
            name = "layers.renamed.weight"
        shape = (2,) if wrong_shape and index == 970 else (1, 1)
        values[PREFIX + name] = torch.full(shape, float(index), dtype=torch.float32)
    values[PREFIX + "_trajectory_head.vocab"] = torch.tensor([1.0])
    return values


def _metadata(
    *, checkpoint: Path, manifest: Path, raw_state, changes=None
) -> ExportMetadata:
    resolved = load_resolved_domain(RESOLVED_DOMAIN)
    metadata = ExportMetadata(
        base_checkpoint_sha256="1" * 64,
        base_state_dict_sha256="2" * 64,
        export_checkpoint_sha256=sha256_file(checkpoint),
        export_state_dict_sha256=state_dict_sha256(raw_state),
        raw_schema_sha256=state_schema_sha256(raw_state),
        resolved_domain_sha256=resolved.sha256,
        sensor_preset_sha256=sha256_file(SENSOR_PRESET),
        training_manifest_sha256=sha256_file(manifest),
        cache_schema_version=2,
        clock_policy_version=1,
        training_config_sha256="5" * 64,
        domain_id="morai-ddv2-v1-oracle-command",
        source_commits={
            "diffusiondrive_v2": DDV2_COMMIT,
            "validator": VALIDATOR_COMMIT,
            "preprocessing": resolved.document["source_code_commit"],
            "training": "1" * 40,
        },
        trainable_prefixes=TRAINABLE_PREFIXES,
        mutable_bn_policy=MUTABLE_BN_POLICY,
        raw_key_count=972,
        runtime_key_count=971,
        best_micro_ade=1.25,
        validation_run_id="morai-ddv2-v1-oracle-command-validation-v1",
        exported_at_utc="2026-08-09T00:00:00.000000Z",
    )
    return replace(metadata, **(changes or {}))


def _profile_bundle(
    tmp_path: Path, *, renamed: bool = False, wrong_shape: bool = False
) -> tuple[Path, OrderedDict[str, torch.Tensor]]:
    from morai_ddv2.export.validator_profile import build_morai_validator_profile

    raw_state = _raw_state(renamed=renamed, wrong_shape=wrong_shape)
    checkpoint = tmp_path / "ddv2-morai.ckpt"
    torch.save({"epoch": 30, "state_dict": raw_state}, checkpoint)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_bytes(b'{"schema_version":2}\n')
    metadata_path = checkpoint.with_suffix(".ckpt.metadata.json")
    metadata = _metadata(checkpoint=checkpoint, manifest=manifest, raw_state=raw_state)
    write_metadata_atomic(metadata_path, metadata)
    bundle = ExportBundle(
        checkpoint=checkpoint,
        metadata=metadata_path,
        checkpoint_sha256=metadata.export_checkpoint_sha256,
        state_dict_sha256=metadata.export_state_dict_sha256,
    )
    profile = tmp_path / "configs/validator/profile-morai-ddv2-oracle.json"
    build_morai_validator_profile(
        baseline_profile=BASELINE_PROFILE,
        output_profile=profile,
        validator_root=VALIDATOR_ROOT,
        bundle=bundle,
        resolved_domain=RESOLVED_DOMAIN,
        sensor_preset=SENSOR_PRESET,
        training_manifest=manifest,
    )
    return profile, raw_state


def _rewrite_profile_asset_hash(profile: Path, role: str, sha256: str) -> None:
    value = json.loads(profile.read_text(encoding="utf-8"))
    selected = next(
        item for item in value["profiles"] if item["id"] == "diffusiondrive_v2"
    )
    asset = next(item for item in selected["runtime_assets"] if item["role"] == role)
    asset["sha256"] = sha256
    profile.write_bytes(canonical_json_bytes(value))


def _profile_document(profile: Path):
    return json.loads(profile.read_text(encoding="utf-8"))


def _selected_profile(value):
    return next(item for item in value["profiles"] if item["id"] == "diffusiondrive_v2")


def _write_profile(profile: Path, value) -> None:
    profile.write_bytes(canonical_json_bytes(value))


def test_preflight_runs_identity_checks_before_real_validator_strict_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, raw_state = _profile_bundle(tmp_path)
    events = []
    real_profile = module._validator_load_selected_profile
    real_assets = module._validator_verify_runtime_assets
    real_metadata = module.load_required_metadata
    real_raw = module.load_raw_sel_checkpoint
    real_strict = module._validator_load_selector_state_dict

    def load_profile(*args, **kwargs):
        events.append("profile")
        return real_profile(*args, **kwargs)

    def verify_assets(*args, **kwargs):
        assert events == ["profile"]
        events.append("assets")
        return real_assets(*args, **kwargs)

    def load_metadata(*args, **kwargs):
        assert events == ["profile", "assets"]
        events.append("metadata")
        return real_metadata(*args, **kwargs)

    def load_raw(*args, **kwargs):
        assert events == ["profile", "assets", "metadata"]
        events.append("raw")
        return real_raw(*args, **kwargs)

    def construct(assets, device):
        assert assets.checkpoint.is_file()
        assert device == torch.device("cpu")
        assert events == ["profile", "assets", "metadata", "raw"]
        events.append("construct")
        return _SchemaNetwork()

    def strict(*args, **kwargs):
        assert events == ["profile", "assets", "metadata", "raw", "construct"]
        events.append("strict")
        return real_strict(*args, **kwargs)

    monkeypatch.setattr(module, "_validator_load_selected_profile", load_profile)
    monkeypatch.setattr(module, "_validator_verify_runtime_assets", verify_assets)
    monkeypatch.setattr(module, "load_required_metadata", load_metadata)
    monkeypatch.setattr(module, "load_raw_sel_checkpoint", load_raw)
    monkeypatch.setattr(module, "construct_pdm_free_sel_network", construct)
    monkeypatch.setattr(module, "_validator_load_selector_state_dict", strict)

    verified = module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert events == ["profile", "assets", "metadata", "raw", "construct", "strict"]
    assert len(verified.raw_checkpoint.raw_state_dict) == 972
    assert len(verified.runtime_state) == 971
    assert tuple(verified.runtime_state) == tuple(
        key.removeprefix(PREFIX)
        for key in raw_state
        if not key.endswith("_trajectory_head.vocab")
    )
    assert isinstance(verified.network, _SchemaNetwork)
    assert torch.equal(
        verified.network.layers["layer_970"].weight,
        torch.tensor([[970.0]]),
    )


def test_preflight_rejects_unpinned_validator_root_before_first_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    fake_root = tmp_path / "fake-validator"
    fake_root.mkdir()
    imports = []

    def imported(*args, **kwargs):
        del args, kwargs
        imports.append(True)
        raise AssertionError("caller-supplied code was reached")

    monkeypatch.setattr(module, "_validator_load_selected_profile", imported)

    with pytest.raises(IntegrityError, match="pinned Validator"):
        module.verify_validator_bundle(profile, fake_root)

    assert imports == []


@pytest.mark.parametrize(
    ("target_name", "message"),
    [
        ("diffusiondrivev2_model_sel_inference.py", "derivative"),
        ("profile_store.py", "module"),
    ],
)
def test_pinned_validator_root_rejects_altered_trusted_code(
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    message: str,
) -> None:
    from morai_ddv2.export import validator_preflight as module

    real_hash = module.sha256_file

    def changed(path: Path) -> str:
        if Path(path).name == target_name:
            return "0" * 64
        return real_hash(path)

    monkeypatch.setattr(module, "sha256_file", changed)

    with pytest.raises(IntegrityError, match=message):
        module._validate_pinned_validator_root(VALIDATOR_ROOT)


@pytest.mark.parametrize("source_name", ["validator", "preprocessing"])
def test_metadata_source_pins_fail_before_network_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_name: str
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, raw_state = _profile_bundle(tmp_path)
    metadata_path = tmp_path / "ddv2-morai.ckpt.metadata.json"
    original = _metadata(
        checkpoint=tmp_path / "ddv2-morai.ckpt",
        manifest=tmp_path / "manifest.jsonl",
        raw_state=raw_state,
    )
    metadata_path.unlink()
    write_metadata_atomic(
        metadata_path,
        replace(
            original,
            source_commits={
                **dict(original.source_commits),
                source_name: "9" * 40,
            },
        ),
    )
    _rewrite_profile_asset_hash(
        profile, "checkpoint_metadata", sha256_file(metadata_path)
    )
    constructed = []
    monkeypatch.setattr(
        module,
        "construct_pdm_free_sel_network",
        lambda *args, **kwargs: constructed.append(True),
    )

    with pytest.raises(
        IntegrityError, match="source|Validator|preprocessing|resolved domain"
    ):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert constructed == []


@pytest.mark.parametrize("encoding", ["pretty", "duplicate", "nan"])
def test_profile_requires_strict_canonical_json_before_validator_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    payload = profile.read_bytes()
    if encoding == "pretty":
        payload = json.dumps(_profile_document(profile), indent=2).encode("utf-8")
    elif encoding == "duplicate":
        payload = b'{"selection":null,' + payload[1:]
    else:
        payload = b'{"nonfinite":NaN,' + payload[1:]
    profile.write_bytes(payload)
    loaded = []
    monkeypatch.setattr(
        module,
        "_validator_load_selected_profile",
        lambda *args, **kwargs: loaded.append(True),
    )

    with pytest.raises(IntegrityError, match="canonical|profile|JSON"):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert loaded == []


def test_profile_symlink_is_rejected_before_validator_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    alias = tmp_path / "profile-link.json"
    alias.symlink_to(profile)
    loaded = []
    monkeypatch.setattr(
        module,
        "_validator_load_selected_profile",
        lambda *args, **kwargs: loaded.append(True),
    )

    with pytest.raises(IntegrityError, match="profile|regular|symlink"):
        module.verify_validator_bundle(alias, VALIDATOR_ROOT)

    assert loaded == []


@pytest.mark.parametrize("invalid_hash", [None, "A" * 64])
def test_runtime_asset_requires_mandatory_lowercase_sha_before_native_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_hash: str | None,
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    value = _profile_document(profile)
    asset = next(
        item
        for item in _selected_profile(value)["runtime_assets"]
        if item["role"] == "image_backbone"
    )
    if invalid_hash is None:
        asset.pop("sha256")
    else:
        asset["sha256"] = invalid_hash
    _write_profile(profile, value)
    verified = []
    monkeypatch.setattr(
        module,
        "_validator_verify_runtime_assets",
        lambda *args, **kwargs: verified.append(True),
    )

    with pytest.raises(IntegrityError, match="SHA|hash|asset"):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert verified == []


def test_runtime_asset_symlink_is_rejected_before_native_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    value = _profile_document(profile)
    asset = next(
        item
        for item in _selected_profile(value)["runtime_assets"]
        if item["role"] == "checkpoint"
    )
    alias = tmp_path / "checkpoint-link.ckpt"
    alias.symlink_to(Path(asset["path"]))
    asset["path"] = str(alias)
    _write_profile(profile, value)
    verified = []
    monkeypatch.setattr(
        module,
        "_validator_verify_runtime_assets",
        lambda *args, **kwargs: verified.append(True),
    )

    with pytest.raises(IntegrityError, match="asset|regular|symlink"):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert verified == []


def test_profile_is_reauthenticated_across_validator_second_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    original = profile.read_bytes()
    real_load = module._validator_load_selected_profile
    constructed = []

    def swap_then_load(profile_file, model_id, validator_root):
        replacement = profile.with_name("replacement-profile.json")
        replacement.write_bytes(b'{"nonfinite":NaN,' + original[1:])
        replacement.replace(profile)
        return real_load(profile_file, model_id, validator_root)

    monkeypatch.setattr(module, "_validator_load_selected_profile", swap_then_load)
    monkeypatch.setattr(
        module,
        "construct_pdm_free_sel_network",
        lambda *args, **kwargs: constructed.append(True) or _SchemaNetwork(),
    )

    with pytest.raises(IntegrityError, match="profile.*changed|profile.*identity"):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert constructed == []


@pytest.mark.parametrize("mutation", ["extra_field", "json_type"])
def test_validator_selected_data_must_equal_strictly_parsed_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    real_load = module._validator_load_selected_profile
    verified_assets = []

    def altered_selection(profile_file, model_id, validator_root):
        selected = real_load(profile_file, model_id, validator_root)
        data = json.loads(json.dumps(selected.data))
        if mutation == "extra_field":
            data["unexpected_post_parse_field"] = True
        else:
            data["domain_contract"]["camera"]["output"]["value_range"][0] = False
        return replace(selected, data=data)

    monkeypatch.setattr(module, "_validator_load_selected_profile", altered_selection)
    monkeypatch.setattr(
        module,
        "_validator_verify_runtime_assets",
        lambda *args, **kwargs: verified_assets.append(True),
    )
    monkeypatch.setattr(
        module,
        "construct_pdm_free_sel_network",
        lambda *args, **kwargs: _SchemaNetwork(),
    )

    with pytest.raises(IntegrityError, match="selected|strict|profile"):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert verified_assets == []


def test_asset_is_reauthenticated_immediately_before_network_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    value = _profile_document(profile)
    asset = next(
        item
        for item in _selected_profile(value)["runtime_assets"]
        if item["role"] == "image_backbone"
    )
    backbone = tmp_path / "verified-backbone.bin"
    backbone.write_bytes(b"verified-backbone")
    asset["path"] = str(backbone)
    asset["sha256"] = sha256_file(backbone)
    _write_profile(profile, value)
    real_select_assets = module.sel_assets_from_verified_roles
    constructed = []

    def swap_after_asset_selection(selected, roles, validator_root):
        result = real_select_assets(selected, roles, validator_root)
        replacement = tmp_path / "replacement-backbone.bin"
        replacement.write_bytes(b"replacement-backbone")
        replacement.replace(backbone)
        return result

    monkeypatch.setattr(
        module, "sel_assets_from_verified_roles", swap_after_asset_selection
    )
    monkeypatch.setattr(
        module,
        "construct_pdm_free_sel_network",
        lambda *args, **kwargs: constructed.append(True) or _SchemaNetwork(),
    )

    with pytest.raises(
        IntegrityError, match="image_backbone|asset.*changed|asset.*hash"
    ):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert constructed == []


@pytest.mark.parametrize(
    "drift",
    ["camera_resize", "lidar_normalization", "reference_stamp", "static_transform"],
)
def test_behavioral_domain_contract_drift_fails_before_network_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path)
    value = _profile_document(profile)
    selected = _selected_profile(value)
    if drift == "camera_resize":
        selected["domain_contract"]["camera"]["resize_wh"]["front"] = [454, 256]
    elif drift == "lidar_normalization":
        selected["domain_contract"]["lidar"]["normalization"] = "count/6"
    elif drift == "reference_stamp":
        selected["inference_policy"]["reference_stamp_policy"] = "reuse"
    else:
        value["common_inference_policy"]["static_transforms"]["T_base_from_lidar"][
            "translation_m"
        ] = [1.4, 0, 1.25]
    _write_profile(profile, value)
    constructed = []
    monkeypatch.setattr(
        module,
        "construct_pdm_free_sel_network",
        lambda *args, **kwargs: constructed.append(True),
    )

    with pytest.raises(IntegrityError, match="contract|camera|LiDAR|stamp|transform"):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert constructed == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("export_checkpoint_sha256", "a" * 64, "checkpoint"),
        ("export_state_dict_sha256", "b" * 64, "state"),
        ("raw_schema_sha256", "c" * 64, "schema"),
        ("resolved_domain_sha256", "d" * 64, "resolved"),
        ("sensor_preset_sha256", "e" * 64, "sensor"),
        ("training_manifest_sha256", "f" * 64, "manifest"),
    ],
)
def test_every_metadata_or_identity_mismatch_fails_before_network_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, raw_state = _profile_bundle(tmp_path)
    metadata_path = tmp_path / "ddv2-morai.ckpt.metadata.json"
    original = _metadata(
        checkpoint=tmp_path / "ddv2-morai.ckpt",
        manifest=tmp_path / "manifest.jsonl",
        raw_state=raw_state,
    )
    metadata_path.unlink()
    write_metadata_atomic(metadata_path, replace(original, **{field: value}))
    _rewrite_profile_asset_hash(
        profile, "checkpoint_metadata", sha256_file(metadata_path)
    )
    constructed = []
    monkeypatch.setattr(
        module,
        "construct_pdm_free_sel_network",
        lambda *args, **kwargs: constructed.append(True),
    )

    with pytest.raises(Exception, match=message):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert constructed == []


@pytest.mark.parametrize(
    ("renamed", "wrong_shape"),
    [(True, False), (False, True)],
)
def test_reauthenticated_same_count_key_or_shape_drift_reaches_real_strict_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renamed: bool,
    wrong_shape: bool,
) -> None:
    from morai_ddv2.export import validator_preflight as module

    profile, _ = _profile_bundle(tmp_path, renamed=renamed, wrong_shape=wrong_shape)
    constructed = []

    def construct(*args, **kwargs):
        del args, kwargs
        constructed.append(True)
        return _SchemaNetwork()

    monkeypatch.setattr(module, "construct_pdm_free_sel_network", construct)

    with pytest.raises(Exception, match="incompatible|strict load failed"):
        module.verify_validator_bundle(profile, VALIDATOR_ROOT)

    assert constructed == [True]


def test_resolve_profile_asset_rejects_missing_duplicate_and_escaping_assets(
    tmp_path: Path,
) -> None:
    from morai_ddv2.export.validator_preflight import (
        resolve_profile_asset,
        sel_assets_from_verified_roles,
    )

    existing = tmp_path / "asset.bin"
    existing.write_bytes(b"asset")
    assert (
        resolve_profile_asset(
            {"role": "checkpoint", "path": str(existing)}, VALIDATOR_ROOT
        )
        == existing.resolve()
    )
    with pytest.raises(IntegrityError, match="missing|file"):
        resolve_profile_asset(
            {"role": "checkpoint", "path": str(tmp_path / "missing")},
            VALIDATOR_ROOT,
        )
    with pytest.raises(IntegrityError, match="role"):
        sel_assets_from_verified_roles(
            {"id": "diffusiondrive_v2", "runtime_assets": []},
            {"checkpoint": existing},
            VALIDATOR_ROOT,
        )
