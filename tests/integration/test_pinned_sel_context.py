from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from morai_ddv2.training.supervised_adapter import build_shared_trajectory_context
from morai_ddv2.upstream.ddv2_sel import SelAssets, load_pdm_free_sel_network


def _required_validator_root() -> Path:
    value = os.environ.get("MORAI_DDV2_VALIDATOR_ROOT")
    if not value:
        pytest.skip("MORAI_DDV2_VALIDATOR_ROOT is required")
    root = Path(value).resolve()
    if not root.is_dir():
        pytest.fail(f"MORAI_DDV2_VALIDATOR_ROOT is not a directory: {root}")
    return root


@pytest.mark.upstream
@pytest.mark.filterwarnings("ignore::DeprecationWarning:matplotlib\\..*")
def test_pinned_sel_shared_context_matches_official_cpu_graph() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    validator_root = _required_validator_root()
    assets = SelAssets(
        v2_checkout=validator_root / "models/sources/diffusiondrive_v2",
        validator_root=validator_root,
        checkpoint=(
            validator_root
            / "models/diffusiondrive_v2/navsim_v2/diffusiondrivev2_sel.ckpt"
        ),
        image_backbone=(
            validator_root / "models/shared/timm/resnet34.a1_in1k/pytorch_model.bin"
        ),
        trajectory_anchor=(
            validator_root
            / "models/diffusiondrive_v2/navsim_v2/kmeans_navsim_traj_20.npy"
        ),
    )
    model, _ = load_pdm_free_sel_network(assets, torch.device("cpu"))
    model.eval()
    assert not hasattr(model._trajectory_head, "_pdm_pool")
    assert not hasattr(model._trajectory_head, "simulator")
    assert not hasattr(model._trajectory_head, "scorer")
    features = {
        "camera_feature": torch.zeros(1, 3, 256, 1024),
        "lidar_feature": torch.zeros(1, 1, 256, 256),
        "status_feature": torch.zeros(1, 8),
    }

    with torch.inference_mode():
        context = build_shared_trajectory_context(model, features)

    assert context.trajectory_query.shape == (1, 1, 256)
    assert context.agents_query.shape == (1, 30, 256)
    assert context.cross_bev_feature.shape == (1, 256, 64, 64)
    assert context.bev_spatial_shape == torch.Size([64, 64])
    assert context.status_encoding.shape == (1, 256)
    assert context.status_feature is features["status_feature"]
    assert context.camera_feature is features["camera_feature"]
    for tensor in (
        context.trajectory_query,
        context.agents_query,
        context.cross_bev_feature,
        context.status_encoding,
    ):
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")
        assert torch.isfinite(tensor).all()
