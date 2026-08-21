from collections import OrderedDict

import h5py
import numpy as np
import pytest
import torch
from torch import nn

import morai_ddv2.train as train_module
from morai_ddv2.model import (
    CHECKPOINT_PREFIX,
    VOCAB_KEY,
    configure_finetuning,
    normalize_checkpoint_state,
    save_checkpoint_atomic,
    supervised_trajectory_loss,
)
from morai_ddv2.train import MoraiHDF5Dataset
from morai_ddv2.training import FinetuneLoss, TRAINABLE_PREFIXES, TrainingBoundaryError


def _raw_state():
    state = OrderedDict(
        (f"{CHECKPOINT_PREFIX}layer_{index}", torch.tensor(float(index)))
        for index in range(971)
    )
    state[f"{CHECKPOINT_PREFIX}{VOCAB_KEY}"] = torch.tensor([123.0])
    return state


def test_checkpoint_normalization_and_atomic_merge(tmp_path):
    raw = _raw_state()
    runtime = normalize_checkpoint_state(raw)
    assert len(runtime) == 971
    assert VOCAB_KEY not in runtime

    trained = OrderedDict((key, value + 1) for key, value in runtime.items())

    class StateOnlyModel:
        def state_dict(self):
            return trained

    output = save_checkpoint_atomic(
        tmp_path / "trained.ckpt",
        {"state_dict": raw, "marker": "preserved"},
        StateOnlyModel(),
    )
    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert saved["marker"] == "preserved"
    assert len(saved["state_dict"]) == 972
    assert torch.equal(
        saved["state_dict"][f"{CHECKPOINT_PREFIX}{VOCAB_KEY}"],
        raw[f"{CHECKPOINT_PREFIX}{VOCAB_KEY}"],
    )
    for key, tensor in normalize_checkpoint_state(saved["state_dict"]).items():
        assert torch.equal(tensor, trained[key])


def test_checkpoint_requires_the_exact_prefix():
    raw = _raw_state()
    key, value = raw.popitem(last=False)
    raw[f"module.{key}"] = value
    with pytest.raises(TrainingBoundaryError, match="exact"):
        normalize_checkpoint_state(raw)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(1, 1)
        self.lidar_encoder = nn.Linear(1, 1)
        self.transformers = nn.ModuleList([nn.Linear(1, 1)])
        self.lidar_channel_to_img = nn.ModuleList([nn.Linear(1, 1)])
        self.img_channel_to_lidar = nn.ModuleList([nn.Linear(1, 1)])

    def forward(self, camera, lidar):
        image = self.image_encoder(camera.mean((1, 2, 3), keepdim=False)[:, None])
        cloud = self.lidar_encoder(lidar.mean((1, 2, 3), keepdim=False)[:, None])
        value = (image + cloud).view(-1, 1, 1, 1)
        return value.repeat(1, 2, 2, 2), value.repeat(1, 4, 1, 1), None


def test_freeze_manifest_contains_only_the_five_sensor_prefixes():
    model = nn.Module()
    model._backbone = _Backbone()
    model.head = nn.Linear(1, 1)
    manifest = configure_finetuning(model)

    assert manifest.trainable_parameters
    assert all(name.startswith(TRAINABLE_PREFIXES) for name in manifest.trainable_parameters)
    assert all(
        parameter.requires_grad == name.startswith(TRAINABLE_PREFIXES)
        for name, parameter in model.named_parameters()
    )
    assert not model.training
    assert model._backbone.image_encoder.training
    assert not model.head.training


class _SharedDecoder(nn.Module):
    def forward(self, query, keyval):
        return query + keyval.mean(dim=1, keepdim=True)


class _Scheduler:
    def add_noise(self, original_samples, noise, timesteps):
        del noise, timesteps
        return original_samples


class _DiffDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(
        self,
        trajectory,
        points,
        bev,
        spatial,
        agents,
        ego,
        time,
        status,
        global_img,
    ):
        del spatial
        assert global_img is None
        shared = (
            bev.mean((2, 3))
            + agents.mean(1)
            + ego.mean(1)
            + time.squeeze(1)
            + status.squeeze(1)
        )
        hidden = trajectory + shared[:, None]
        xy = hidden[..., :2, None].transpose(-1, -2).expand(-1, -1, 8, -1)
        heading = hidden[..., 2, None, None].expand(-1, -1, 8, 1)
        regression = torch.cat((xy + points, heading), dim=-1)
        classification = hidden.mean(-1).unsqueeze(1)
        return [regression], [classification], hidden


class _LossComputer(nn.Module):
    cls_loss_weight = 10.0
    reg_loss_weight = 8.0

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, regression, classification, targets, anchor):
        del anchor
        self.calls += 1
        return (regression - targets["trajectory"][:, None]).square().mean() + classification.square().mean()


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.plan_anchor = nn.Parameter(torch.zeros(20, 8, 2), requires_grad=False)
        self.diffusion_scheduler = _Scheduler()
        self.plan_anchor_encoder = nn.Linear(512, 4)
        self.time_mlp = nn.Embedding(50, 4)
        self.diff_decoder = _DiffDecoder()
        self.loss_computer = _LossComputer()
        self.public_forward_calls = 0

    def forward(self, *args, **kwargs):
        self.public_forward_calls += 1
        raise AssertionError("public trajectory forward must not run")


class _TinySEL(nn.Module):
    def __init__(self):
        super().__init__()
        self._query_splits = [1, 2]
        self._backbone = _Backbone()
        self._bev_downscale = nn.Identity()
        self._status_encoding = nn.Linear(8, 4)
        self._keyval_embedding = nn.Embedding(2, 4)
        self._query_embedding = nn.Embedding(3, 4)
        self._tf_decoder = _SharedDecoder()
        self.bev_proj = nn.Linear(6, 4)
        self._trajectory_head = _Head()
        self.public_forward_calls = 0

    def forward(self, *args, **kwargs):
        self.public_forward_calls += 1
        raise AssertionError("public SEL forward must not run")


def test_supervised_path_bypasses_public_sel_forward():
    model = _TinySEL()
    loss = supervised_trajectory_loss(
        model,
        torch.ones(1, 3, 256, 1024),
        torch.ones(1, 1, 256, 256),
        torch.ones(1, 8),
        torch.zeros(1, 8, 3),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.public_forward_calls == 0
    assert model._trajectory_head.public_forward_calls == 0
    assert model._trajectory_head.loss_computer.calls == 1
    assert model._backbone.image_encoder.weight.grad is not None


def test_evaluation_metrics_are_deterministic_and_restore_modes(monkeypatch):
    model = nn.Sequential(nn.Linear(1, 1), nn.Dropout())
    model.eval()
    model[0].train()
    original_modes = tuple(module.training for module in model.modules())
    target = torch.zeros(2, 8, 3)
    prediction = target.clone()
    prediction[..., 0] = 2.0

    def output(*args, **kwargs):
        assert kwargs["deterministic"] is True
        return torch.tensor(3.0), prediction

    monkeypatch.setattr(train_module, "supervised_trajectory_output", output)
    monkeypatch.setattr(
        train_module,
        "compose_finetune_loss",
        lambda *args: FinetuneLoss(
            torch.tensor(36.0), torch.tensor(3.0), torch.tensor(0.0)
        ),
    )
    batch = {
        "camera_feature": torch.zeros(2, 3, 256, 1024),
        "lidar_feature": torch.zeros(2, 1, 256, 256),
        "status_feature": torch.zeros(2, 8),
        "trajectory": target,
    }
    metrics = train_module._evaluate(model, [batch], torch.device("cpu"), False, {}, None)

    assert tuple(module.training for module in model.modules()) == original_modes
    assert metrics == {
        "samples": 2,
        "total_loss": 36.0,
        "trajectory_loss": 3.0,
        "preservation_loss": 0.0,
        "ade": 2.0,
        "fde": 2.0,
    }


def test_hdf5_dataset_reconstructs_quantized_inputs(tmp_path):
    path = tmp_path / "samples.h5"
    with h5py.File(path, "w") as target:
        target.create_dataset("camera", data=np.full((1, 3, 256, 1024), 255, np.uint8))
        target.create_dataset("lidar", data=np.full((1, 1, 256, 256), 5, np.uint8))
        target.create_dataset("status", data=np.ones((1, 8), np.float32))
        target.create_dataset("trajectory", data=np.ones((1, 8, 3), np.float32))

    dataset = MoraiHDF5Dataset(path)
    sample = dataset[0]
    assert torch.equal(sample["camera_feature"], torch.ones(3, 256, 1024))
    assert torch.equal(sample["lidar_feature"], torch.ones(1, 256, 256))
    assert sample["status_feature"].dtype == torch.float32
    assert sample["trajectory"].shape == (8, 3)
    dataset.close()
