from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from morai_ddv2.training.supervised_adapter import (
    SharedTrajectoryContext,
    build_shared_trajectory_context,
)


class _EmbeddingTrap(nn.Module):
    def __init__(self, weight: torch.Tensor, calls: list[str], name: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self._calls = calls
        self._name = name

    def forward(self, *_args, **_kwargs):
        self._calls.append(self._name)
        raise AssertionError(f"{self._name} forward must not be called")


class _Backbone(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    def forward(self, camera: torch.Tensor, lidar: torch.Tensor):
        self._calls.append("backbone")
        batch = camera.shape[0]
        bev_up = torch.ones((batch, 2, 4, 4), dtype=camera.dtype, device=camera.device)
        bev_concat = torch.arange(
            batch * 3 * 2 * 2, dtype=lidar.dtype, device=lidar.device
        ).view(batch, 3, 2, 2)
        return bev_up, bev_concat, None


class _RecordIdentity(nn.Module):
    def __init__(self, calls: list[str], name: str) -> None:
        super().__init__()
        self._calls = calls
        self._name = name
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self._calls.append(self._name)
        self.last_input = value
        return value


class _StatusEncoding(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    def forward(self, status: torch.Tensor) -> torch.Tensor:
        self._calls.append("status_encoding")
        return status[:, :3]


class _BevProjection(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self._calls.append("bev_proj")
        self.last_input = value
        return value[..., :4]


class _Decoder(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls
        self.query: torch.Tensor | None = None
        self.keyval: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, keyval: torch.Tensor) -> torch.Tensor:
        self._calls.append("tf_decoder")
        self.query = query
        self.keyval = keyval
        return query + 100.0


class _CallTrap(nn.Module):
    def __init__(self, calls: list[str], name: str) -> None:
        super().__init__()
        self._calls = calls
        self._name = name

    def forward(self, *_args, **_kwargs):
        self._calls.append(self._name)
        raise AssertionError(f"forbidden call: {self._name}")


class _TrajectoryHeadTrap(_CallTrap):
    def __init__(self, calls: list[str]) -> None:
        super().__init__(calls, "trajectory_head")
        self.scorer_decoder = _CallTrap(calls, "selector_scorer")


class _FakeSelModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self._query_splits = [1, 2]
        self.keyval_sentinel = torch.tensor(
            [
                [101.0, 102.0, 103.0],
                [111.0, 112.0, 113.0],
                [121.0, 122.0, 123.0],
                [131.0, 132.0, 133.0],
                [141.0, 142.0, 143.0],
            ]
        )
        self.query_sentinel = torch.tensor(
            [[201.0, 202.0, 203.0], [211.0, 212.0, 213.0], [221.0, 222.0, 223.0]]
        )
        self._backbone = _Backbone(self.calls)
        self._bev_downscale = _RecordIdentity(self.calls, "bev_downscale")
        self._status_encoding = _StatusEncoding(self.calls)
        self.bev_proj = _BevProjection(self.calls)
        self._tf_decoder = _Decoder(self.calls)
        self._keyval_embedding = _EmbeddingTrap(
            self.keyval_sentinel.clone(), self.calls, "keyval_embedding"
        )
        self._query_embedding = _EmbeddingTrap(
            self.query_sentinel.clone(), self.calls, "query_embedding"
        )
        self._bev_semantic_head = _CallTrap(self.calls, "bev_semantic_head")
        self._agent_head = _CallTrap(self.calls, "agent_head")
        self._trajectory_head = _TrajectoryHeadTrap(self.calls)
        self._pdm_pool = _CallTrap(self.calls, "pdm_pool")

    def forward(self, *_args, **_kwargs):
        self.calls.append("public_forward")
        raise AssertionError("public model forward must not be called")


def _valid_features(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "camera_feature": torch.zeros(batch, 3, 256, 1024),
        "lidar_feature": torch.zeros(batch, 1, 256, 256),
        "status_feature": torch.arange(batch * 8, dtype=torch.float32).view(batch, 8),
    }


def test_build_context_reuses_only_the_shared_sel_graph() -> None:
    model = _FakeSelModel()
    features = _valid_features()

    context = build_shared_trajectory_context(model, features)

    assert isinstance(context, SharedTrajectoryContext)
    assert model.calls == [
        "backbone",
        "bev_downscale",
        "status_encoding",
        "bev_proj",
        "tf_decoder",
    ]
    assert model._tf_decoder.query is not None
    assert model._tf_decoder.keyval is not None
    expected_query = model.query_sentinel.unsqueeze(0).repeat(2, 1, 1)
    assert torch.equal(model._tf_decoder.query, expected_query)

    bev_tokens = torch.arange(24, dtype=torch.float32).view(2, 3, 2, 2)
    bev_tokens = bev_tokens.flatten(-2, -1).permute(0, 2, 1)
    expected_keyval = torch.cat(
        (bev_tokens, features["status_feature"][:, None, :3]), dim=1
    )
    expected_keyval = expected_keyval + model.keyval_sentinel.unsqueeze(0)
    assert torch.equal(model._tf_decoder.keyval, expected_keyval)

    assert context.trajectory_query.shape == (2, 1, 3)
    assert context.agents_query.shape == (2, 2, 3)
    assert context.cross_bev_feature.shape == (2, 4, 4, 4)
    assert context.bev_spatial_shape == torch.Size([4, 4])
    assert context.status_encoding.shape == (2, 3)
    assert context.status_feature is features["status_feature"]
    assert context.camera_feature is features["camera_feature"]
    assert torch.equal(context.trajectory_query, expected_query[:, :1] + 100.0)

    with pytest.raises(FrozenInstanceError):
        context.camera_feature = features["camera_feature"]
    assert not isinstance(context, nn.Module)
    assert not hasattr(context, "parameters")


@pytest.mark.parametrize(
    ("key", "bad_shape"),
    [
        ("camera_feature", (2, 3, 255, 1024)),
        ("lidar_feature", (2, 1, 256, 255)),
        ("status_feature", (2, 7)),
    ],
)
def test_build_context_rejects_wrong_feature_shapes(
    key: str, bad_shape: tuple[int, ...]
) -> None:
    model = _FakeSelModel()
    features = _valid_features()
    features[key] = torch.zeros(bad_shape)

    with pytest.raises(ValueError, match=key):
        build_shared_trajectory_context(model, features)

    assert model.calls == []


def test_build_context_rejects_empty_or_mismatched_batches() -> None:
    model = _FakeSelModel()
    empty = _valid_features(batch=0)
    mismatched = _valid_features()
    mismatched["lidar_feature"] = torch.zeros(1, 1, 256, 256)

    with pytest.raises(ValueError, match="batch"):
        build_shared_trajectory_context(model, empty)
    with pytest.raises(ValueError, match="batch"):
        build_shared_trajectory_context(model, mismatched)

    assert model.calls == []


def test_build_context_requires_float32_tensors_on_one_device() -> None:
    model = _FakeSelModel()
    wrong_dtype = _valid_features()
    wrong_dtype["camera_feature"] = wrong_dtype["camera_feature"].double()
    wrong_device = _valid_features()
    wrong_device["status_feature"] = torch.empty((2, 8), device="meta")

    with pytest.raises(ValueError, match="float32"):
        build_shared_trajectory_context(model, wrong_dtype)
    with pytest.raises(ValueError, match="device"):
        build_shared_trajectory_context(model, wrong_device)

    assert model.calls == []


@pytest.mark.parametrize("key", ["camera_feature", "lidar_feature", "status_feature"])
def test_build_context_rejects_nonfinite_features(key: str) -> None:
    model = _FakeSelModel()
    features = _valid_features(batch=1)
    features[key].view(-1)[0] = torch.nan

    with pytest.raises(ValueError, match=f"{key}.*finite"):
        build_shared_trajectory_context(model, features)

    assert model.calls == []


def test_build_context_rejects_missing_and_nontensor_features() -> None:
    model = _FakeSelModel()
    missing = _valid_features()
    del missing["status_feature"]
    nontensor = _valid_features()
    nontensor["status_feature"] = [0.0] * 8  # type: ignore[assignment]

    with pytest.raises(ValueError, match="missing.*status_feature"):
        build_shared_trajectory_context(model, missing)
    with pytest.raises(TypeError, match="status_feature.*Tensor"):
        build_shared_trajectory_context(model, nontensor)

    assert model.calls == []
