from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from morai_ddv2.training.contracts import (
    TRAINABLE_PREFIXES,
    FreezeManifest,
    TrainingBoundaryError,
)
from morai_ddv2.training.freeze import (
    assert_training_boundary,
    build_optimizer,
    configure_training_boundary,
    extract_mutable_state,
)


class _FeatureBranch(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("projection", nn.Linear(2, 2, bias=False)),
                    ("norm", nn.BatchNorm1d(2)),
                ]
            )
        )


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = _FeatureBranch()
        self.lidar_encoder = _FeatureBranch()
        self.transformers = _FeatureBranch()
        self.lidar_channel_to_img = _FeatureBranch()
        self.img_channel_to_lidar = _FeatureBranch()
        self.image_encoder_extra = _FeatureBranch()
        self.c5_conv = nn.Linear(2, 2)
        self.up_conv5 = nn.Linear(2, 2)
        self.up_conv4 = nn.Linear(2, 2)


class _BoundaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._backbone = _Backbone()
        self._bev_downscale = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self._keyval_embedding = nn.Embedding(2, 2)
        self._query_embedding = nn.Embedding(2, 2)
        self._status_encoding = nn.Linear(2, 2)
        self._trajectory_head = nn.Linear(2, 2)
        self._agent_head = nn.Linear(2, 2)
        self._bev_semantic_head = nn.Linear(2, 2)
        self._selector = nn.Linear(2, 2)
        self.register_buffer("selector_temperature", torch.ones(()))


EXPECTED_TRAINABLE = (
    "_backbone.image_encoder.projection.weight",
    "_backbone.image_encoder.norm.weight",
    "_backbone.image_encoder.norm.bias",
    "_backbone.lidar_encoder.projection.weight",
    "_backbone.lidar_encoder.norm.weight",
    "_backbone.lidar_encoder.norm.bias",
    "_backbone.transformers.projection.weight",
    "_backbone.transformers.norm.weight",
    "_backbone.transformers.norm.bias",
    "_backbone.lidar_channel_to_img.projection.weight",
    "_backbone.lidar_channel_to_img.norm.weight",
    "_backbone.lidar_channel_to_img.norm.bias",
    "_backbone.img_channel_to_lidar.projection.weight",
    "_backbone.img_channel_to_lidar.norm.weight",
    "_backbone.img_channel_to_lidar.norm.bias",
)

EXPECTED_MUTABLE = EXPECTED_TRAINABLE + (
    "_backbone.image_encoder.norm.running_mean",
    "_backbone.image_encoder.norm.running_var",
    "_backbone.image_encoder.norm.num_batches_tracked",
    "_backbone.lidar_encoder.norm.running_mean",
    "_backbone.lidar_encoder.norm.running_var",
    "_backbone.lidar_encoder.norm.num_batches_tracked",
    "_backbone.transformers.norm.running_mean",
    "_backbone.transformers.norm.running_var",
    "_backbone.transformers.norm.num_batches_tracked",
    "_backbone.lidar_channel_to_img.norm.running_mean",
    "_backbone.lidar_channel_to_img.norm.running_var",
    "_backbone.lidar_channel_to_img.norm.num_batches_tracked",
    "_backbone.img_channel_to_lidar.norm.running_mean",
    "_backbone.img_channel_to_lidar.norm.running_var",
    "_backbone.img_channel_to_lidar.norm.num_batches_tracked",
)

EXPECTED_TRAIN_MODULES = {
    "_backbone.image_encoder",
    "_backbone.image_encoder.projection",
    "_backbone.image_encoder.norm",
    "_backbone.lidar_encoder",
    "_backbone.lidar_encoder.projection",
    "_backbone.lidar_encoder.norm",
    "_backbone.transformers",
    "_backbone.transformers.projection",
    "_backbone.transformers.norm",
    "_backbone.lidar_channel_to_img",
    "_backbone.lidar_channel_to_img.projection",
    "_backbone.lidar_channel_to_img.norm",
    "_backbone.img_channel_to_lidar",
    "_backbone.img_channel_to_lidar.projection",
    "_backbone.img_channel_to_lidar.norm",
}


def test_configure_training_boundary_freezes_everything_outside_five_paths() -> None:
    model = _BoundaryModel()

    manifest = configure_training_boundary(model)

    assert TRAINABLE_PREFIXES == (
        "_backbone.image_encoder.",
        "_backbone.lidar_encoder.",
        "_backbone.transformers.",
        "_backbone.lidar_channel_to_img.",
        "_backbone.img_channel_to_lidar.",
    )
    assert isinstance(manifest, FreezeManifest)
    assert manifest.trainable_parameters == EXPECTED_TRAINABLE
    assert manifest.mutable_state_entries == EXPECTED_MUTABLE
    assert not model.training

    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is (name in EXPECTED_TRAINABLE), name
    for name, module in model.named_modules():
        assert module.training is (name in EXPECTED_TRAIN_MODULES), name

    frozen_roots = (
        "_backbone.image_encoder_extra.",
        "_backbone.c5_conv.",
        "_backbone.up_conv5.",
        "_backbone.up_conv4.",
        "_bev_downscale.",
        "_keyval_embedding.",
        "_query_embedding.",
        "_status_encoding.",
        "_trajectory_head.",
        "_agent_head.",
        "_bev_semantic_head.",
        "_selector.",
    )
    assert all(
        any(name.startswith(root) for root in frozen_roots)
        for name in manifest.frozen_parameters
    )


def test_mutable_state_contains_only_trainable_parameters_and_allowed_bn() -> None:
    model = _BoundaryModel()
    manifest = configure_training_boundary(model)
    current = {**dict(model.named_parameters()), **dict(model.named_buffers())}

    mutable = extract_mutable_state(model, manifest)

    assert isinstance(mutable, OrderedDict)
    assert tuple(mutable) == EXPECTED_MUTABLE
    assert "selector_temperature" not in mutable
    assert "_bev_downscale.1.running_mean" not in mutable
    for name, tensor in mutable.items():
        assert torch.equal(tensor, current[name])
        assert tensor is not current[name]
        assert not tensor.requires_grad


def test_optimizer_uses_each_trainable_parameter_exactly_once() -> None:
    model = _BoundaryModel()
    manifest = configure_training_boundary(model)

    optimizer = build_optimizer(model, manifest, lr=1.0e-5, weight_decay=1.0e-4)

    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected_parameters = [
        dict(model.named_parameters())[name] for name in EXPECTED_TRAINABLE
    ]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [id(parameter) for parameter in optimizer_parameters] == [
        id(parameter) for parameter in expected_parameters
    ]
    assert len({id(parameter) for parameter in optimizer_parameters}) == len(
        optimizer_parameters
    )
    assert not {id(buffer) for buffer in model.buffers()}.intersection(
        id(parameter) for parameter in optimizer_parameters
    )


def test_assert_training_boundary_rejects_later_global_train_call() -> None:
    model = _BoundaryModel()
    manifest = configure_training_boundary(model)
    model.train()

    with pytest.raises(TrainingBoundaryError, match="model must remain globally eval"):
        assert_training_boundary(model, manifest)


@pytest.mark.parametrize("name", [EXPECTED_TRAINABLE[0], "_trajectory_head.weight"])
def test_assert_training_boundary_rejects_requires_grad_drift(name: str) -> None:
    model = _BoundaryModel()
    manifest = configure_training_boundary(model)
    parameter = dict(model.named_parameters())[name]
    parameter.requires_grad_(not parameter.requires_grad)

    with pytest.raises(TrainingBoundaryError, match="requires_grad boundary"):
        assert_training_boundary(model, manifest)
