from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch

from morai_ddv2.upstream.ddv2_sel import (
    SelAssets,
    load_pdm_free_sel_network,
    normalize_sel_runtime_state,
)


EXPECTED_CHECKPOINT_SHA256 = (
    "452c4e0dd328c49b981f3fd5ed13eb1c6a2b0f3a9486adf43b40a5645ed543b1"
)


def _required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is required for the pinned upstream integration test")
    path = Path(value).resolve()
    if not path.exists():
        pytest.fail(f"{variable} does not exist: {path}")
    return path


@pytest.mark.upstream
@pytest.mark.filterwarnings("ignore::DeprecationWarning:matplotlib\\..*")
def test_official_sel_checkpoint_strict_loads_into_pdm_free_derivative() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    validator_root = _required_path("MORAI_DDV2_VALIDATOR_ROOT")
    checkpoint = _required_path("MORAI_DDV2_SEL_CHECKPOINT")
    assets = SelAssets(
        v2_checkout=validator_root / "models/sources/diffusiondrive_v2",
        validator_root=validator_root,
        checkpoint=checkpoint,
        image_backbone=(
            validator_root / "models/shared/timm/resnet34.a1_in1k/pytorch_model.bin"
        ),
        trajectory_anchor=(
            validator_root
            / "models/diffusiondrive_v2/navsim_v2/kmeans_navsim_traj_20.npy"
        ),
    )

    network, raw = load_pdm_free_sel_network(assets, torch.device("cpu"))
    runtime_state = normalize_sel_runtime_state(raw.raw_state_dict)

    assert raw.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert len(raw.raw_state_dict) == 972
    assert len(runtime_state) == 971
    assert "_trajectory_head.vocab" not in runtime_state
    assert len(network.state_dict()) == 971
    assert network.__class__.__module__ == (
        "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference"
    )
    model_module = sys.modules[network.__class__.__module__]
    model_path = Path(model_module.__file__).resolve()
    assert assets.v2_checkout.resolve() in model_path.parents
    assert all(parameter.device.type == "cpu" for parameter in network.parameters())
