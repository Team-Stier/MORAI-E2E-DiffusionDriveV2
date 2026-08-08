from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

from morai_ddv2.config.canonical import sha256_file
from morai_ddv2.errors import IntegrityError


RAW_SEL_KEY_COUNT = 972
RUNTIME_SEL_KEY_COUNT = 971
PDM_REMOVED_CHECKPOINT_KEY = "_trajectory_head.vocab"
SEL_CONFIG_MODULE = "navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config"
PDM_FREE_SEL_MODULE = (
    "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference"
)
PDM_FREE_SEL_SYMBOL = f"{PDM_FREE_SEL_MODULE}.V2TransfuserModel"


class SelSchemaError(IntegrityError):
    """Raised when a SEL checkpoint violates its exact deployment schema."""


@dataclass(frozen=True)
class SelAssets:
    v2_checkout: Path
    validator_root: Path
    checkpoint: Path
    image_backbone: Path
    trajectory_anchor: Path


@dataclass(frozen=True)
class RawSelCheckpoint:
    container: Mapping[str, Any]
    raw_state_dict: Mapping[str, torch.Tensor]
    checkpoint_sha256: str


def normalize_sel_key(raw_key: str) -> str:
    key = str(raw_key)
    if key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("agent._transfuser_model."):
        key = key[len("agent._transfuser_model.") :]
    return key


def normalize_sel_runtime_state(
    raw: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    removed: list[str] = []
    for raw_key, tensor in raw.items():
        key = normalize_sel_key(raw_key)
        if key == PDM_REMOVED_CHECKPOINT_KEY:
            removed.append(key)
            continue
        if key in normalized:
            raise SelSchemaError(f"duplicate normalized key: {key}")
        normalized[key] = tensor
    if removed != [PDM_REMOVED_CHECKPOINT_KEY]:
        raise SelSchemaError(f"expected one vocab key, got {removed}")
    return normalized


def load_raw_sel_checkpoint(path: Path) -> RawSelCheckpoint:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise SelSchemaError(f"SEL checkpoint is not a file: {checkpoint_path}")
    try:
        container = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as error:
        raise SelSchemaError(
            f"cannot load SEL checkpoint {checkpoint_path} on CPU: {error}"
        ) from error
    if not isinstance(container, Mapping):
        raise SelSchemaError(
            f"SEL checkpoint {checkpoint_path} must be a top-level mapping"
        )
    raw_state = container.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise SelSchemaError(
            f"SEL checkpoint {checkpoint_path} must contain a top-level state_dict mapping"
        )
    if len(raw_state) != RAW_SEL_KEY_COUNT:
        raise SelSchemaError(
            f"expected {RAW_SEL_KEY_COUNT} raw checkpoint keys, got {len(raw_state)}"
        )
    for raw_key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            raise SelSchemaError(
                f"SEL checkpoint key {raw_key} must be torch.Tensor, "
                f"got {type(value).__name__}"
            )
    runtime_state = normalize_sel_runtime_state(raw_state)
    if len(runtime_state) != RUNTIME_SEL_KEY_COUNT:
        raise SelSchemaError(
            f"expected {RUNTIME_SEL_KEY_COUNT} runtime keys, got {len(runtime_state)}"
        )
    return RawSelCheckpoint(
        container=container,
        raw_state_dict=raw_state,
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def _require_file(path: Path, role: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise SelSchemaError(f"SEL {role} is not a file: {resolved}")
    return resolved


def _validator_source_root(validator_root: Path) -> Path:
    source_root = Path(validator_root) / "src/Validator_pkg/validator_inference/src"
    package = source_root / "validator_inference/__init__.py"
    if not package.is_file():
        raise SelSchemaError(
            f"pinned Validator Python package is unavailable under {source_root}"
        )
    value = str(source_root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)
    return source_root


def _require_validator_module(module: Any, validator_root: Path, symbol: str) -> None:
    module_path = Path(getattr(module, "__file__", "")).resolve()
    root = Path(validator_root).resolve()
    if not module_path.is_file() or root not in module_path.parents:
        raise SelSchemaError(
            f"Validator symbol {symbol} resolved outside pinned Validator root "
            f"{root}: {module_path}"
        )


def _validator_contracts(validator_root: Path):
    _validator_source_root(validator_root)
    module = importlib.import_module("validator_inference.models.contracts")
    _require_validator_module(
        module,
        validator_root,
        "validator_inference.models.contracts",
    )
    return module


def _validator_sel_loader(validator_root: Path):
    _validator_source_root(validator_root)
    module = importlib.import_module("validator_inference.models.diffusiondrive_v2")
    _require_validator_module(
        module,
        validator_root,
        "validator_inference.models.diffusiondrive_v2.load_diffusiondrive_selector_state_dict",
    )
    return module.load_diffusiondrive_selector_state_dict


def construct_pdm_free_sel_network(
    assets: SelAssets, device: torch.device
) -> nn.Module:
    checkout = Path(assets.v2_checkout)
    _require_file(
        checkout
        / "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py",
        "PDM-free derivative",
    )
    backbone = _require_file(assets.image_backbone, "image backbone")
    anchor = _require_file(assets.trajectory_anchor, "trajectory anchor")

    contracts = _validator_contracts(assets.validator_root)
    contracts.prepend_source_path(checkout)
    config_module = importlib.import_module(SEL_CONFIG_MODULE)
    model_module = importlib.import_module(PDM_FREE_SEL_MODULE)
    contracts.require_module_from_checkout(model_module, checkout, PDM_FREE_SEL_SYMBOL)

    config = config_module.TransfuserConfig()
    config.bkb_path = str(backbone)
    config.plan_anchor_path = str(anchor)
    network = model_module.V2TransfuserModel(config)
    if not isinstance(network, nn.Module):
        raise SelSchemaError(f"{PDM_FREE_SEL_SYMBOL} did not construct torch.nn.Module")
    return network.to(torch.device(device))


def load_pdm_free_sel_network(
    assets: SelAssets, device: torch.device
) -> tuple[nn.Module, RawSelCheckpoint]:
    raw_checkpoint = load_raw_sel_checkpoint(assets.checkpoint)
    runtime_state = normalize_sel_runtime_state(raw_checkpoint.raw_state_dict)
    if len(runtime_state) != RUNTIME_SEL_KEY_COUNT:
        raise SelSchemaError(
            f"expected {RUNTIME_SEL_KEY_COUNT} runtime keys, got {len(runtime_state)}"
        )

    network = construct_pdm_free_sel_network(assets, torch.device("cpu"))
    strict_loader = _validator_sel_loader(assets.validator_root)
    strict_loader(
        network,
        dict(raw_checkpoint.raw_state_dict),
        checkpoint_path=assets.checkpoint,
    )
    network.to(torch.device(device))
    return network, raw_checkpoint
