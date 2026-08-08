from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass(frozen=True)
class NoisyAnchorBatch:
    plan_anchor: torch.Tensor
    decoder_input: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor


@dataclass(frozen=True)
class SupervisedTrajectoryResult:
    trajectory_loss: torch.Tensor
    poses_reg: torch.Tensor
    poses_cls: torch.Tensor
    plan_anchor: torch.Tensor
    timesteps: torch.Tensor


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


def sample_noisy_plan_anchors(
    head: nn.Module, batch_size: int, device: torch.device
) -> NoisyAnchorBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    anchor = head.plan_anchor.to(device=device, dtype=torch.float32)
    if anchor.shape != (20, 8, 2):
        raise ValueError("plan_anchor must have shape [20,8,2]")
    anchor = anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    normalized = anchor.clone()
    normalized[..., 0] /= 50.0
    normalized[..., 1] /= 20.0
    timesteps = torch.randint(0, 50, (batch_size,), device=device, dtype=torch.long)
    noise = torch.randn_like(normalized)
    noisy = head.diffusion_scheduler.add_noise(
        original_samples=normalized,
        noise=noise,
        timesteps=timesteps,
    ).float()
    decoder_input = noisy.clamp(-1.0, 1.0)
    decoder_input[..., 0] *= 50.0
    decoder_input[..., 1] *= 20.0
    return NoisyAnchorBatch(anchor, decoder_input, timesteps, noise)


def _sine_xy_embedding(points: torch.Tensor) -> torch.Tensor:
    half_hidden_dim = 32
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=points.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = points[..., 0] * (2 * math.pi)
    y_embed = points[..., 1] * (2 * math.pi)
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


def forward_trajectory_supervised(
    model: nn.Module,
    features: Mapping[str, torch.Tensor],
    target_trajectory: torch.Tensor,
) -> SupervisedTrajectoryResult:
    if not isinstance(target_trajectory, torch.Tensor):
        raise TypeError("target_trajectory must be a torch.Tensor")
    if target_trajectory.ndim != 3 or target_trajectory.shape[1:] != (8, 3):
        raise ValueError("target_trajectory must have shape [B,8,3]")
    if target_trajectory.dtype != torch.float32:
        raise ValueError("target_trajectory must use torch.float32")
    if not torch.isfinite(target_trajectory).all().item():
        raise ValueError("target_trajectory must contain only finite values")

    head = model._trajectory_head
    if len(head.diff_decoder.layers) != 1:
        raise RuntimeError("DDV2 supervised adapter requires exactly one decoder layer")
    loss_computer = head.loss_computer
    if loss_computer.cls_loss_weight != 10.0 or loss_computer.reg_loss_weight != 8.0:
        raise RuntimeError("DDV2 upstream loss weights must be cls=10.0 and reg=8.0")

    context = build_shared_trajectory_context(model, features)
    batch_size = context.trajectory_query.shape[0]
    if target_trajectory.shape[0] != batch_size:
        raise ValueError("target_trajectory batch must match features")
    if target_trajectory.device != context.trajectory_query.device:
        raise ValueError("target_trajectory must share the feature device")

    noisy = sample_noisy_plan_anchors(head, batch_size, context.trajectory_query.device)
    trajectory_embedding = _sine_xy_embedding(noisy.decoder_input).flatten(-2)
    trajectory_feature = head.plan_anchor_encoder(trajectory_embedding)
    trajectory_feature = trajectory_feature.view(batch_size, 20, -1)
    time_embedding = head.time_mlp(noisy.timesteps).view(batch_size, 1, -1)
    poses_reg_list, poses_cls_list, _ = head.diff_decoder(
        trajectory_feature,
        noisy.decoder_input,
        context.cross_bev_feature,
        context.bev_spatial_shape,
        context.agents_query,
        context.trajectory_query,
        time_embedding,
        context.status_encoding[:, None],
        None,
    )
    if len(poses_reg_list) != 1 or len(poses_cls_list) != 1:
        raise RuntimeError("DDV2 decoder must return exactly one output group")
    poses_reg = poses_reg_list[0]
    poses_cls_source = poses_cls_list[0]
    if poses_reg.shape != (batch_size, 20, 8, 3):
        raise RuntimeError("DDV2 regression source must have shape [B,20,8,3]")
    if poses_cls_source.shape != (batch_size, 1, 20):
        raise RuntimeError("DDV2 classification source must have shape [B,1,20]")
    poses_cls = poses_cls_source.squeeze(1)
    trajectory_loss = loss_computer(
        poses_reg,
        poses_cls,
        {"trajectory": target_trajectory},
        noisy.plan_anchor,
    )
    if not isinstance(trajectory_loss, torch.Tensor) or trajectory_loss.ndim != 0:
        raise RuntimeError("DDV2 trajectory loss must be a scalar tensor")
    if not torch.isfinite(trajectory_loss).item():
        raise RuntimeError("DDV2 trajectory loss must be finite")
    return SupervisedTrajectoryResult(
        trajectory_loss,
        poses_reg,
        poses_cls,
        noisy.plan_anchor,
        noisy.timesteps,
    )
