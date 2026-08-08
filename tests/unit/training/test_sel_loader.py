from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib
from pathlib import Path
from types import ModuleType

import pytest
import torch

from morai_ddv2.upstream import ddv2_sel
from morai_ddv2.upstream.ddv2_sel import (
    RawSelCheckpoint,
    SelAssets,
    SelSchemaError,
    construct_pdm_free_sel_network,
    load_pdm_free_sel_network,
    load_raw_sel_checkpoint,
    normalize_sel_key,
    normalize_sel_runtime_state,
)
from tests.fakes.sel_model import (
    CapturingSelModel,
    FakeSelModel,
    FakeTransfuserConfig,
    fake_raw_state,
    write_fake_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _touch_asset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _assets(repo_root: Path, tmp_path: Path, checkpoint: Path) -> SelAssets:
    return SelAssets(
        v2_checkout=tmp_path / "v2-checkout",
        validator_root=repo_root / "third_party/E2E-OpenLoop-ROS-Validator",
        checkpoint=checkpoint,
        image_backbone=_touch_asset(tmp_path / "backbone.bin"),
        trajectory_anchor=_touch_asset(tmp_path / "anchor.npy"),
    )


@pytest.mark.parametrize(
    ("raw_key", "expected"),
    [
        (
            "module.agent._transfuser_model._backbone.weight",
            "_backbone.weight",
        ),
        ("agent._transfuser_model.head.bias", "head.bias"),
        ("module.head.bias", "head.bias"),
        ("head.bias", "head.bias"),
        (
            "agent._transfuser_model.module.head.bias",
            "module.head.bias",
        ),
    ],
)
def test_normalize_sel_key_removes_optional_prefixes_in_documented_order(
    raw_key: str, expected: str
) -> None:
    assert normalize_sel_key(raw_key) == expected


def test_normalize_runtime_state_excludes_exactly_one_normalized_vocab_key() -> None:
    kept_tensor = torch.tensor([1.0])
    similarly_named_tensor = torch.tensor([2.0])
    raw = OrderedDict(
        [
            ("module.agent._transfuser_model.encoder.weight", kept_tensor),
            (
                "agent._transfuser_model.extra._trajectory_head.vocab",
                similarly_named_tensor,
            ),
            (
                "module.agent._transfuser_model._trajectory_head.vocab",
                torch.zeros(1),
            ),
        ]
    )

    normalized = normalize_sel_runtime_state(raw)

    assert list(normalized) == [
        "encoder.weight",
        "extra._trajectory_head.vocab",
    ]
    assert normalized["encoder.weight"] is kept_tensor
    assert normalized["extra._trajectory_head.vocab"] is similarly_named_tensor


def test_normalize_runtime_state_rejects_duplicate_normalized_key() -> None:
    raw = OrderedDict(
        [
            ("agent._transfuser_model.encoder.weight", torch.zeros(1)),
            ("module.agent._transfuser_model.encoder.weight", torch.ones(1)),
            ("agent._transfuser_model._trajectory_head.vocab", torch.zeros(1)),
        ]
    )

    with pytest.raises(
        SelSchemaError, match="duplicate normalized key: encoder.weight"
    ):
        normalize_sel_runtime_state(raw)


@pytest.mark.parametrize(
    "raw",
    [
        OrderedDict([("agent._transfuser_model.encoder.weight", torch.zeros(1))]),
        OrderedDict(
            [
                ("agent._transfuser_model._trajectory_head.vocab", torch.zeros(1)),
                (
                    "module.agent._transfuser_model._trajectory_head.vocab",
                    torch.zeros(1),
                ),
            ]
        ),
    ],
)
def test_normalize_runtime_state_requires_exactly_one_vocab_key(raw) -> None:
    with pytest.raises(SelSchemaError, match="expected one vocab key"):
        normalize_sel_runtime_state(raw)


def test_load_raw_checkpoint_preserves_container_state_and_exact_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "official.ckpt"
    expected_state = write_fake_checkpoint(
        checkpoint, fake_raw_state(module_prefix=True)
    )
    real_torch_load = ddv2_sel.torch.load
    load_calls: list[tuple[str, object]] = []

    def load_on_cpu(path: str, *, map_location):
        load_calls.append((path, map_location))
        return real_torch_load(path, map_location=map_location)

    monkeypatch.setattr(ddv2_sel.torch, "load", load_on_cpu)

    loaded = load_raw_sel_checkpoint(checkpoint)

    assert isinstance(loaded, RawSelCheckpoint)
    assert loaded.container["metadata"] == {"fixture": "sel"}
    assert loaded.raw_state_dict is loaded.container["state_dict"]
    assert list(loaded.raw_state_dict) == list(expected_state)
    assert len(loaded.raw_state_dict) == 972
    assert len(normalize_sel_runtime_state(loaded.raw_state_dict)) == 971
    assert loaded.checkpoint_sha256 == _sha256(checkpoint)
    assert load_calls == [(str(checkpoint), "cpu")]


@pytest.mark.parametrize("key_count", [970, 972])
def test_load_raw_checkpoint_rejects_non_972_raw_count(
    tmp_path: Path, key_count: int
) -> None:
    checkpoint = tmp_path / f"wrong-{key_count}.ckpt"
    write_fake_checkpoint(checkpoint, fake_raw_state(key_count))

    with pytest.raises(
        SelSchemaError,
        match=rf"expected 972 raw checkpoint keys, got {key_count + 1}",
    ):
        load_raw_sel_checkpoint(checkpoint)


def test_load_raw_checkpoint_requires_state_dict_container(tmp_path: Path) -> None:
    checkpoint = tmp_path / "not-a-container.ckpt"
    torch.save(fake_raw_state(), checkpoint)

    with pytest.raises(SelSchemaError, match="top-level state_dict mapping"):
        load_raw_sel_checkpoint(checkpoint)


def test_load_raw_checkpoint_rejects_non_tensor_state_value(tmp_path: Path) -> None:
    checkpoint = tmp_path / "non-tensor.ckpt"
    raw = fake_raw_state()
    raw[next(iter(raw))] = "not a tensor"
    torch.save({"state_dict": raw}, checkpoint)

    with pytest.raises(SelSchemaError, match="must be torch.Tensor"):
        load_raw_sel_checkpoint(checkpoint)


def test_construct_network_imports_only_pdm_free_derivative_and_sets_assets(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _touch_asset(tmp_path / "official.ckpt")
    assets = _assets(repo_root, tmp_path, checkpoint)
    derivative = _touch_asset(
        assets.v2_checkout
        / "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py"
    )
    config_path = _touch_asset(
        assets.v2_checkout
        / "navsim/agents/diffusiondrivev2/diffusiondrivev2_sel_config.py"
    )
    config_module = ModuleType("fake_sel_config")
    config_module.__file__ = str(config_path)
    config_module.TransfuserConfig = FakeTransfuserConfig
    model_module = ModuleType("fake_sel_inference")
    model_module.__file__ = str(derivative)
    model_module.V2TransfuserModel = CapturingSelModel
    real_import_module = importlib.import_module
    requested: list[str] = []

    def import_module(name: str):
        requested.append(name)
        if name == "navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config":
            return config_module
        if name == (
            "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference"
        ):
            return model_module
        return real_import_module(name)

    monkeypatch.setattr(ddv2_sel.importlib, "import_module", import_module)

    network = construct_pdm_free_sel_network(assets, torch.device("cpu"))

    assert isinstance(network, CapturingSelModel)
    assert CapturingSelModel.last_config.bkb_path == str(assets.image_backbone)
    assert CapturingSelModel.last_config.plan_anchor_path == str(
        assets.trajectory_anchor
    )
    assert (
        "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference"
        in requested
    )
    assert "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel" not in requested


def test_construct_network_rejects_stale_config_from_another_checkout(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _touch_asset(tmp_path / "official.ckpt")
    assets = _assets(repo_root, tmp_path, checkpoint)
    derivative = _touch_asset(
        assets.v2_checkout
        / "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py"
    )
    stale_config_path = _touch_asset(
        tmp_path
        / "stale-checkout/navsim/agents/diffusiondrivev2/diffusiondrivev2_sel_config.py"
    )
    config_module = ModuleType("stale_sel_config")
    config_module.__file__ = str(stale_config_path)
    config_module.TransfuserConfig = FakeTransfuserConfig
    model_module = ModuleType("pinned_sel_inference")
    model_module.__file__ = str(derivative)
    model_module.V2TransfuserModel = CapturingSelModel
    real_import_module = importlib.import_module

    def import_module(name: str):
        if name == "navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config":
            return config_module
        if name == (
            "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference"
        ):
            return model_module
        return real_import_module(name)

    monkeypatch.setattr(ddv2_sel.importlib, "import_module", import_module)

    with pytest.raises(
        RuntimeError,
        match=(
            r"diffusiondrivev2_sel_config\.TransfuserConfig resolved outside "
            r"pinned checkout"
        ),
    ):
        construct_pdm_free_sel_network(assets, torch.device("cpu"))


def test_validator_loader_must_resolve_from_selected_pinned_root(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_module = ModuleType("wrong_validator_loader")
    wrong_module.__file__ = str(_touch_asset(tmp_path / "wrong/diffusiondrive_v2.py"))
    wrong_module.load_diffusiondrive_selector_state_dict = lambda *args, **kwargs: None
    real_import_module = importlib.import_module

    def import_module(name: str):
        if name == "validator_inference.models.diffusiondrive_v2":
            return wrong_module
        return real_import_module(name)

    monkeypatch.setattr(ddv2_sel.importlib, "import_module", import_module)

    with pytest.raises(SelSchemaError, match="outside pinned Validator root"):
        ddv2_sel._validator_sel_loader(
            repo_root / "third_party/E2E-OpenLoop-ROS-Validator"
        )


def test_load_network_calls_pinned_validator_strict_loader(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "official.ckpt"
    write_fake_checkpoint(checkpoint)
    assets = _assets(repo_root, tmp_path, checkpoint)
    network = FakeSelModel()
    constructed_devices: list[torch.device] = []
    validator_inputs: list[dict[str, torch.Tensor]] = []
    real_strict_loader = ddv2_sel._validator_sel_loader(assets.validator_root)

    def construct(_assets: SelAssets, device: torch.device) -> FakeSelModel:
        constructed_devices.append(device)
        return network

    def strict_loader(loaded_network, raw_state, *, checkpoint_path):
        validator_inputs.append(raw_state)
        return real_strict_loader(
            loaded_network,
            raw_state,
            checkpoint_path=checkpoint_path,
        )

    monkeypatch.setattr(ddv2_sel, "construct_pdm_free_sel_network", construct)
    monkeypatch.setattr(
        ddv2_sel,
        "_validator_sel_loader",
        lambda _validator_root: strict_loader,
    )

    loaded_network, raw = load_pdm_free_sel_network(assets, torch.device("cpu"))

    assert loaded_network is network
    assert len(raw.raw_state_dict) == 972
    assert constructed_devices == [torch.device("cpu")]
    assert network.strict_loads == [True]
    assert network.to_devices == [torch.device("cpu")]
    assert network.events == [
        ("load_state_dict", True),
        ("to", torch.device("cpu")),
    ]
    assert len(validator_inputs) == 1
    assert len(validator_inputs[0]) == 972
    assert all(
        key.startswith("agent._transfuser_model.") for key in validator_inputs[0]
    )
    assert "agent._transfuser_model._trajectory_head.vocab" in validator_inputs[0]


@pytest.mark.parametrize(
    ("network_key_count", "mutation", "message"),
    [
        (
            972,
            None,
            r"missing=\['layers.layer_971.weight'\] unexpected=\[\]",
        ),
        (
            970,
            None,
            r"missing=\[\] unexpected=\['layers.layer_970.weight'\]",
        ),
        (971, "shape", "strict load failed"),
    ],
    ids=["missing", "unexpected", "shape-mismatch"],
)
def test_load_network_rejects_every_non_vocab_strict_mismatch(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    network_key_count: int,
    mutation: str | None,
    message: str,
) -> None:
    checkpoint = tmp_path / "mismatch.ckpt"
    raw = fake_raw_state()
    if mutation == "shape":
        raw[next(iter(raw))] = torch.zeros(2, 1)
    write_fake_checkpoint(checkpoint, raw)
    assets = _assets(repo_root, tmp_path, checkpoint)
    network = FakeSelModel(network_key_count)
    monkeypatch.setattr(
        ddv2_sel,
        "construct_pdm_free_sel_network",
        lambda _assets, _device: network,
    )

    with pytest.raises(RuntimeError, match=message):
        load_pdm_free_sel_network(assets, torch.device("cpu"))
