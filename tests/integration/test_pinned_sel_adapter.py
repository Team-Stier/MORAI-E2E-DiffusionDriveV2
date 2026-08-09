from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from morai_ddv2.training.freeze import configure_training_boundary
from morai_ddv2.training.supervised_adapter import forward_trajectory_supervised
from morai_ddv2.upstream.ddv2_sel import SelAssets, load_pdm_free_sel_network


def _required_validator_root() -> Path:
    value = os.environ.get("MORAI_DDV2_VALIDATOR_ROOT")
    if not value:
        pytest.skip("MORAI_DDV2_VALIDATOR_ROOT is required")
    root = Path(value).resolve()
    if not root.is_dir():
        pytest.fail(f"MORAI_DDV2_VALIDATOR_ROOT is not a directory: {root}")
    return root


def _assets(validator_root: Path) -> SelAssets:
    return SelAssets(
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


@pytest.mark.upstream
@pytest.mark.filterwarnings("ignore::DeprecationWarning:matplotlib\\..*")
def test_pinned_sel_supervised_adapter_backpropagates_through_frozen_head() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_pdm_free_sel_network(_assets(_required_validator_root()), device)
    manifest = configure_training_boundary(model)
    head = model._trajectory_head
    assert len(head.diff_decoder.layers) == 1
    assert head.loss_computer.cls_loss_weight == 10.0
    assert head.loss_computer.reg_loss_weight == 8.0
    torch.manual_seed(20260809)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260809)
    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024, device=device),
        "lidar_feature": torch.randn(1, 1, 256, 256, device=device),
        "status_feature": torch.randn(1, 8, device=device),
    }
    target = torch.randn(1, 8, 3, device=device)

    result = forward_trajectory_supervised(model, features, target)

    assert result.trajectory_loss.shape == torch.Size([])
    assert torch.isfinite(result.trajectory_loss)
    assert result.poses_reg.shape == (1, 20, 8, 3)
    assert result.poses_cls.shape == (1, 20)
    assert result.plan_anchor.shape == (1, 20, 8, 2)
    assert result.timesteps.shape == (1,)
    assert all(not parameter.requires_grad for parameter in head.parameters())

    result.trajectory_loss.backward()

    named_parameters = dict(model.named_parameters())
    trainable_gradients = [
        named_parameters[name].grad for name in manifest.trainable_parameters
    ]
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient).item() > 0
        for gradient in trainable_gradients
    )
    assert all(parameter.grad is None for parameter in head.parameters())
