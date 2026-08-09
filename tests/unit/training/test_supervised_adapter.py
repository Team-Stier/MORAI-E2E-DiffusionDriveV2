from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from morai_ddv2.training.freeze import configure_training_boundary
from morai_ddv2.training.supervised_adapter import (
    SupervisedTrajectoryResult,
    forward_trajectory_supervised,
)


class _ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, camera: torch.Tensor) -> torch.Tensor:
        return camera.mean(dim=(1, 2, 3)) * self.scale


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = _ImageEncoder()

    def forward(self, camera: torch.Tensor, lidar: torch.Tensor):
        value = self.image_encoder(camera) + lidar.mean(dim=(1, 2, 3))
        bev_up = value[:, None, None, None].repeat(1, 4, 2, 2)
        bev_concat = value[:, None, None, None].repeat(1, 4, 1, 1)
        return bev_up, bev_concat, None


class _StatusEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 4)

    def forward(self, status: torch.Tensor) -> torch.Tensor:
        return self.projection(status)


class _SharedDecoder(nn.Module):
    def forward(self, query: torch.Tensor, keyval: torch.Tensor) -> torch.Tensor:
        return query + keyval.mean(dim=1, keepdim=True)


class _RecordingLinear(nn.Linear):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__(input_features, output_features)
        self.calls = 0
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_input = value
        return super().forward(value)


class _TimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(50, 4)
        self.calls = 0
        self.last_input: torch.Tensor | None = None

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_input = timesteps
        return self.embedding(timesteps)


class _Scheduler:
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del noise, timesteps
        return original_samples


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(
        self,
        trajectory: torch.Tensor,
        noisy_points: torch.Tensor,
        bev: torch.Tensor,
        _spatial: torch.Size,
        agents: torch.Tensor,
        ego: torch.Tensor,
        time: torch.Tensor,
        status: torch.Tensor,
        global_img: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if status.shape[1:] != (1, 4):
            raise AssertionError("decoder status encoding must retain its query axis")
        if global_img is not None:
            raise AssertionError("supervised decoder global_img must remain None")
        shared = (
            bev.mean(dim=(2, 3))
            + agents.mean(dim=1)
            + ego.mean(dim=1)
            + time.squeeze(1)
            + status.squeeze(1)
        )
        hidden = self.projection(trajectory + shared[:, None])
        offsets = hidden[..., :2, None].transpose(-1, -2).expand(-1, -1, 8, -1)
        heading = hidden[..., 2, None, None].expand(-1, -1, 8, 1)
        poses_reg = torch.cat((noisy_points + 0.01 * offsets, heading), dim=-1)
        poses_cls = hidden[..., 3].unsqueeze(1)
        return poses_reg, poses_cls, hidden


class _DiffDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer()])
        self.calls = 0
        self.classification_without_group = False

    def forward(self, *args):
        self.calls += 1
        poses_reg, poses_cls, hidden = self.layers[0](*args)
        if self.classification_without_group:
            poses_cls = poses_cls.squeeze(1)
        return [poses_reg], [poses_cls], hidden


class _LossComputer(nn.Module):
    def __init__(self, cls_weight: float = 10.0, reg_weight: float = 8.0) -> None:
        super().__init__()
        self.cls_loss_weight = cls_weight
        self.reg_loss_weight = reg_weight
        self.calls = 0
        self.targets = None
        self.plan_anchor = None

    def forward(self, poses_reg, poses_cls, targets, plan_anchor):
        self.calls += 1
        self.targets = targets
        self.plan_anchor = plan_anchor
        return (poses_reg - targets["trajectory"][:, None]).square().mean() + (
            poses_cls.square().mean()
        )


class _TrajectoryHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.plan_anchor = nn.Parameter(torch.zeros(20, 8, 2), requires_grad=False)
        self.diffusion_scheduler = _Scheduler()
        self.plan_anchor_encoder = _RecordingLinear(512, 4)
        self.time_mlp = _TimeEmbedding()
        self.diff_decoder = _DiffDecoder()
        self.loss_computer = _LossComputer()
        self.public_forward_calls = 0

    def forward(self, *_args, **_kwargs):
        self.public_forward_calls += 1
        raise AssertionError("trajectory head public forward is forbidden")


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._query_splits = [1, 2]
        self._backbone = _Backbone()
        self._bev_downscale = nn.Identity()
        self._status_encoding = _StatusEncoding()
        self._keyval_embedding = nn.Embedding(2, 4)
        self._query_embedding = nn.Embedding(3, 4)
        self.bev_proj = nn.Linear(8, 4)
        self._tf_decoder = _SharedDecoder()
        self._trajectory_head = _TrajectoryHead()
        self.public_forward_calls = 0

    def forward(self, *_args, **_kwargs):
        self.public_forward_calls += 1
        raise AssertionError("model public forward is forbidden")


def _features(batch: int = 1) -> dict[str, torch.Tensor]:
    return {
        "camera_feature": torch.ones(batch, 3, 256, 1024),
        "lidar_feature": torch.ones(batch, 1, 256, 256),
        "status_feature": torch.ones(batch, 8),
    }


def _run(model: _Model, batch: int = 1) -> SupervisedTrajectoryResult:
    target = torch.ones(batch, 8, 3)
    result = forward_trajectory_supervised(model, _features(batch), target)
    assert model._trajectory_head.loss_computer.targets == {"trajectory": target}
    assert model._trajectory_head.loss_computer.targets["trajectory"] is target
    return result


def test_supervised_adapter_uses_one_group_one_layer_and_upstream_loss(
    monkeypatch,
) -> None:
    model = _Model()
    manifest = configure_training_boundary(model)
    torch.manual_seed(20260809)

    def _forbidden_argmax(*_args, **_kwargs):
        raise AssertionError("predicted argmax is forbidden in supervised loss")

    monkeypatch.setattr(torch, "argmax", _forbidden_argmax)
    result = _run(model)

    assert isinstance(result, SupervisedTrajectoryResult)
    assert result.trajectory_loss.shape == torch.Size([])
    assert torch.isfinite(result.trajectory_loss)
    assert result.poses_reg.shape == (1, 20, 8, 3)
    assert result.poses_cls.shape == (1, 20)
    assert result.plan_anchor.shape == (1, 20, 8, 2)
    assert result.timesteps.shape == (1,)
    assert len(model._trajectory_head.diff_decoder.layers) == 1
    assert model._trajectory_head.plan_anchor_encoder.calls == 1
    assert model._trajectory_head.time_mlp.calls == 1
    expected_sine = torch.tensor([0.0, 1.0] * 32).repeat(8)
    expected_sine = expected_sine.view(1, 1, 512).repeat(1, 20, 1)
    assert torch.equal(
        model._trajectory_head.plan_anchor_encoder.last_input, expected_sine
    )
    assert model._trajectory_head.time_mlp.last_input is result.timesteps
    assert model._trajectory_head.diff_decoder.calls == 1
    assert model._trajectory_head.loss_computer.calls == 1
    assert model._trajectory_head.loss_computer.plan_anchor is result.plan_anchor
    assert model.public_forward_calls == 0
    assert model._trajectory_head.public_forward_calls == 0

    result.trajectory_loss.backward()

    allowed = dict(model.named_parameters())[manifest.trainable_parameters[0]]
    assert allowed.grad is not None
    assert torch.isfinite(allowed.grad).all()
    assert torch.count_nonzero(allowed.grad).item() > 0
    for name, parameter in model.named_parameters():
        if name not in manifest.trainable_parameters:
            assert parameter.grad is None, name
    with pytest.raises(FrozenInstanceError):
        result.poses_cls = torch.zeros_like(result.poses_cls)


@pytest.mark.parametrize(("cls_weight", "reg_weight"), [(9.0, 8.0), (10.0, 7.0)])
def test_supervised_adapter_rejects_upstream_loss_weight_drift(
    cls_weight: float, reg_weight: float
) -> None:
    model = _Model()
    model._trajectory_head.loss_computer = _LossComputer(cls_weight, reg_weight)

    with pytest.raises(RuntimeError, match="loss weights"):
        _run(model)

    assert model._trajectory_head.loss_computer.calls == 0


def test_supervised_adapter_rejects_more_than_one_decoder_layer() -> None:
    model = _Model()
    model._trajectory_head.diff_decoder.layers.append(_DecoderLayer())

    with pytest.raises(RuntimeError, match="one decoder layer"):
        _run(model)

    assert model._trajectory_head.diff_decoder.calls == 0


def test_supervised_adapter_requires_one_group_classification_source() -> None:
    model = _Model()
    model._trajectory_head.diff_decoder.classification_without_group = True

    with pytest.raises(RuntimeError, match="classification source"):
        _run(model)

    assert model._trajectory_head.loss_computer.calls == 0


def test_supervised_adapter_requires_batched_xyz_targets() -> None:
    model = _Model()

    with pytest.raises(ValueError, match="target_trajectory"):
        forward_trajectory_supervised(model, _features(), torch.zeros(1, 8, 2))

    assert model._trajectory_head.diff_decoder.calls == 0
