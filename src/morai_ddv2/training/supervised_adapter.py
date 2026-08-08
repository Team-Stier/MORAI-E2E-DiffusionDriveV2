from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SharedTrajectoryContext:
    trajectory_query: torch.Tensor
    agents_query: torch.Tensor
    cross_bev_feature: torch.Tensor
    bev_spatial_shape: torch.Size
    status_encoding: torch.Tensor
    status_feature: torch.Tensor
    camera_feature: torch.Tensor


def _validated_features(
    features: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    names_and_shapes = (
        ("camera_feature", (3, 256, 1024)),
        ("lidar_feature", (1, 256, 256)),
        ("status_feature", (8,)),
    )
    values = []
    for name, sample_shape in names_and_shapes:
        try:
            tensor = features[name]
        except KeyError as error:
            raise ValueError(f"missing required feature: {name}") from error
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != len(sample_shape) + 1 or tensor.shape[1:] != sample_shape:
            raise ValueError(
                f"{name} must have shape [B,{','.join(map(str, sample_shape))}]"
            )
        values.append(tensor)

    camera, lidar, status = values
    batch_sizes = {tensor.shape[0] for tensor in values}
    if batch_sizes == {0} or len(batch_sizes) != 1:
        raise ValueError("features must share one non-empty batch size")
    if any(tensor.dtype != torch.float32 for tensor in values):
        raise ValueError("features must use torch.float32")
    if len({tensor.device for tensor in values}) != 1:
        raise ValueError("features must share one device")
    for (name, _), tensor in zip(names_and_shapes, values):
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")
    return camera, lidar, status


def build_shared_trajectory_context(
    model: nn.Module, features: Mapping[str, torch.Tensor]
) -> SharedTrajectoryContext:
    camera, lidar, status = _validated_features(features)
    batch_size = status.shape[0]
    bev_up, bev_concat, _ = model._backbone(camera, lidar)
    spatial = bev_up.shape[2:]
    concat_spatial = bev_concat.shape[2:]
    bev_tokens = model._bev_downscale(bev_concat).flatten(-2, -1).permute(0, 2, 1)
    status_encoding = model._status_encoding(status)
    keyval = torch.cat((bev_tokens, status_encoding[:, None]), dim=1)
    keyval = keyval + model._keyval_embedding.weight[None]
    concat_cross = (
        keyval[:, :-1]
        .permute(0, 2, 1)
        .contiguous()
        .view(batch_size, -1, *concat_spatial)
    )
    concat_cross = F.interpolate(
        concat_cross, size=spatial, mode="bilinear", align_corners=False
    )
    cross = torch.cat((concat_cross, bev_up), dim=1)
    cross = model.bev_proj(cross.flatten(-2, -1).permute(0, 2, 1))
    cross = cross.permute(0, 2, 1).contiguous().view(batch_size, -1, *spatial)
    query = model._query_embedding.weight[None].repeat(batch_size, 1, 1)
    query_out = model._tf_decoder(query, keyval)
    trajectory_query, agents_query = query_out.split(model._query_splits, dim=1)
    return SharedTrajectoryContext(
        trajectory_query,
        agents_query,
        cross,
        spatial,
        status_encoding,
        status,
        camera,
    )
