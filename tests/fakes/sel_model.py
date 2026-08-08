from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


RUNTIME_KEY_COUNT = 971


class FakeSelModel(nn.Module):
    """Small scalar-parameter model with the official derivative key cardinality."""

    def __init__(self, key_count: int = RUNTIME_KEY_COUNT) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                f"layer_{index:03d}": nn.Linear(1, 1, bias=False)
                for index in range(key_count)
            }
        )
        self.strict_loads: list[bool] = []
        self.to_devices: list[torch.device] = []
        self.events: list[tuple[str, object]] = []

    def load_state_dict(self, state_dict, strict: bool = True):
        self.strict_loads.append(strict)
        self.events.append(("load_state_dict", strict))
        return super().load_state_dict(state_dict, strict=strict)

    def to(self, device, *args, **kwargs):
        resolved = torch.device(device)
        self.to_devices.append(resolved)
        self.events.append(("to", resolved))
        return super().to(device, *args, **kwargs)


class CapturingSelModel(nn.Module):
    """Constructor seam used to inspect the upstream configuration."""

    last_config = None

    def __init__(self, config) -> None:
        super().__init__()
        type(self).last_config = config


class FakeTransfuserConfig:
    def __init__(self) -> None:
        self.bkb_path = "unconfigured-backbone"
        self.plan_anchor_path = "unconfigured-anchor"


def fake_runtime_state(
    key_count: int = RUNTIME_KEY_COUNT,
) -> OrderedDict[str, torch.Tensor]:
    model = FakeSelModel(key_count)
    return OrderedDict(
        (key, tensor.clone()) for key, tensor in model.state_dict().items()
    )


def fake_raw_state(
    key_count: int = RUNTIME_KEY_COUNT,
    *,
    module_prefix: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    prefix = (
        "module.agent._transfuser_model."
        if module_prefix
        else "agent._transfuser_model."
    )
    raw = OrderedDict(
        (f"{prefix}{key}", tensor)
        for key, tensor in fake_runtime_state(key_count).items()
    )
    raw[f"{prefix}_trajectory_head.vocab"] = torch.zeros(1)
    return raw


def write_fake_checkpoint(
    path: Path,
    raw_state: Mapping[str, torch.Tensor] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict(raw_state if raw_state is not None else fake_raw_state())
    torch.save({"state_dict": state, "metadata": {"fixture": "sel"}}, path)
    return state
