from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .training.contracts import (
    FreezeManifest,
    TRAINABLE_PREFIXES,
    TrainingBoundaryError,
)
from .training.loss import assert_loss_contract


CHECKPOINT_PREFIX = "agent._transfuser_model."
VOCAB_KEY = "_trajectory_head.vocab"
RAW_KEY_COUNT = 972
RUNTIME_KEY_COUNT = 971


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # torch<2.0 has no weights_only argument
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TrainingBoundaryError("checkpoint must be a mapping")
    if not isinstance(checkpoint.get("state_dict"), Mapping):
        raise TrainingBoundaryError("checkpoint must contain a state_dict mapping")
    return checkpoint


def normalize_checkpoint_state(
    raw_state: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    if len(raw_state) != RAW_KEY_COUNT:
        raise TrainingBoundaryError(
            f"official SEL checkpoint must have {RAW_KEY_COUNT} keys, got {len(raw_state)}"
        )

    runtime: OrderedDict[str, torch.Tensor] = OrderedDict()
    removed_vocab = 0
    for raw_key, tensor in raw_state.items():
        if not isinstance(raw_key, str) or not raw_key.startswith(CHECKPOINT_PREFIX):
            raise TrainingBoundaryError(
                f"checkpoint key lacks exact {CHECKPOINT_PREFIX!r} prefix: {raw_key!r}"
            )
        if not isinstance(tensor, torch.Tensor):
            raise TrainingBoundaryError(f"checkpoint value is not a tensor: {raw_key}")
        key = raw_key[len(CHECKPOINT_PREFIX) :]
        if key == VOCAB_KEY:
            removed_vocab += 1
            continue
        if key in runtime:
            raise TrainingBoundaryError(f"duplicate runtime checkpoint key: {key}")
        runtime[key] = tensor

    if removed_vocab != 1:
        raise TrainingBoundaryError("checkpoint must contain exactly one trajectory vocab")
    if len(runtime) != RUNTIME_KEY_COUNT:
        raise TrainingBoundaryError(
            f"runtime SEL state must have {RUNTIME_KEY_COUNT} keys, got {len(runtime)}"
        )
    return runtime


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_official_model(
    checkout: Path,
    checkpoint_path: Path,
    backbone_path: Path,
    anchor_path: Path,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, Any]]:
    checkout = checkout.expanduser().resolve()
    _require_file(
        checkout
        / "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py",
        "pinned DiffusionDriveV2 SEL model",
    )
    _require_file(
        checkout
        / "navsim/agents/diffusiondrivev2/diffusiondrivev2_sel_config.py",
        "pinned DiffusionDriveV2 SEL config",
    )
    checkpoint_path = _require_file(checkpoint_path, "official SEL checkpoint")
    backbone_path = _require_file(backbone_path, "ResNet34 backbone")
    anchor_path = _require_file(anchor_path, "trajectory anchors")

    checkpoint = _load_checkpoint(checkpoint_path)
    runtime_state = normalize_checkpoint_state(checkpoint["state_dict"])

    checkout_string = str(checkout)
    if checkout_string in sys.path:
        sys.path.remove(checkout_string)
    sys.path.insert(0, checkout_string)
    previous_directory = Path.cwd()
    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.chdir(checkout)
        config_module = importlib.import_module(
            "navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config"
        )
        model_module = importlib.import_module(
            "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel"
        )
        for module in (config_module, model_module):
            module_path = Path(module.__file__).resolve()
            if checkout not in module_path.parents:
                raise TrainingBoundaryError(
                    f"DiffusionDriveV2 import resolved outside pinned checkout: {module_path}"
                )
        config = config_module.TransfuserConfig()
        config.bkb_path = str(backbone_path)
        config.plan_anchor_path = str(anchor_path)
        model = model_module.V2TransfuserModel(config)
    finally:
        os.chdir(previous_directory)
        if previous_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_hf_offline

    pool = getattr(model._trajectory_head, "_pdm_pool", None)
    if pool is not None:
        pool.shutdown(wait=False)
        model._trajectory_head._pdm_pool = None

    model.load_state_dict(runtime_state, strict=True)
    assert_loss_contract(
        model._trajectory_head,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
    )
    model.to(device)
    return model, checkpoint


def configure_finetuning(model: nn.Module) -> FreezeManifest:
    model.eval()
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(TRAINABLE_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise TrainingBoundaryError("no sensor encoder/fusion parameters found")

    modules = dict(model.named_modules())
    for prefix in TRAINABLE_PREFIXES:
        module = modules.get(prefix[:-1])
        if module is not None:
            module.train()
    return FreezeManifest(tuple(trainable))


def _validate_batch(
    camera: torch.Tensor,
    lidar: torch.Tensor,
    status: torch.Tensor,
    trajectory: torch.Tensor,
) -> None:
    expected = (
        (camera, (3, 256, 1024), "camera"),
        (lidar, (1, 256, 256), "lidar"),
        (status, (8,), "status"),
        (trajectory, (8, 3), "trajectory"),
    )
    for tensor, sample_shape, name in expected:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.ndim != len(sample_shape) + 1 or tuple(tensor.shape[1:]) != sample_shape:
            raise ValueError(f"invalid {name} batch shape: {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} contains a non-finite value")
    if len({tensor.shape[0] for tensor, _, _ in expected}) != 1 or camera.shape[0] == 0:
        raise ValueError("batch tensors must share one nonzero batch size")
    if len({tensor.device for tensor, _, _ in expected}) != 1:
        raise ValueError("batch tensors must share one device")


def _sine_xy_embedding(points: torch.Tensor) -> torch.Tensor:
    half = 32
    frequencies = torch.arange(half, dtype=torch.float32, device=points.device)
    frequencies = 10000 ** (2 * (frequencies // 2) / half)
    x = points[..., 0, None] * (2 * math.pi) / frequencies
    y = points[..., 1, None] * (2 * math.pi) / frequencies
    x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(-2)
    y = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((y, x), dim=-1)


def supervised_trajectory_output(
    model: nn.Module,
    camera: torch.Tensor,
    lidar: torch.Tensor,
    status: torch.Tensor,
    target_trajectory: torch.Tensor,
    *,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official learned path without invoking SEL's PDM-backed forward."""

    _validate_batch(camera, lidar, status, target_trajectory)
    batch_size = camera.shape[0]

    bev_up, fused_bev, _ = model._backbone(camera, lidar)
    bev_spatial_shape = bev_up.shape[2:]
    fused_spatial_shape = fused_bev.shape[2:]
    bev_tokens = model._bev_downscale(fused_bev).flatten(-2, -1).permute(0, 2, 1)
    status_encoding = model._status_encoding(status)
    keyval = torch.cat((bev_tokens, status_encoding[:, None]), dim=1)
    keyval = keyval + model._keyval_embedding.weight[None]
    cross_bev = (
        keyval[:, :-1]
        .permute(0, 2, 1)
        .contiguous()
        .view(batch_size, -1, *fused_spatial_shape)
    )
    cross_bev = F.interpolate(
        cross_bev, size=bev_spatial_shape, mode="bilinear", align_corners=False
    )
    cross_bev = torch.cat((cross_bev, bev_up), dim=1)
    cross_bev = model.bev_proj(cross_bev.flatten(-2, -1).permute(0, 2, 1))
    cross_bev = (
        cross_bev.permute(0, 2, 1)
        .contiguous()
        .view(batch_size, -1, *bev_spatial_shape)
    )
    query = model._query_embedding.weight[None].repeat(batch_size, 1, 1)
    query = model._tf_decoder(query, keyval)
    trajectory_query, agents_query = query.split(model._query_splits, dim=1)

    head = model._trajectory_head
    if len(head.diff_decoder.layers) != 1:
        raise TrainingBoundaryError("official SEL diff_decoder must have one layer")
    assert_loss_contract(
        head, trajectory_weight=12.0, preservation_weight=1.0e-3
    )
    anchor = head.plan_anchor.to(device=camera.device, dtype=torch.float32)
    if tuple(anchor.shape) != (20, 8, 2):
        raise TrainingBoundaryError("official plan_anchor must have shape [20,8,2]")
    anchor = anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    normalized_anchor = anchor.clone()
    normalized_anchor[..., 0] /= 50.0
    normalized_anchor[..., 1] /= 20.0
    timesteps = (
        torch.zeros(batch_size, dtype=torch.long, device=camera.device)
        if deterministic
        else torch.randint(0, 50, (batch_size,), device=camera.device)
    )
    noise = (
        torch.zeros_like(normalized_anchor)
        if deterministic
        else torch.randn_like(normalized_anchor)
    )
    noisy = head.diffusion_scheduler.add_noise(
        normalized_anchor, noise, timesteps
    )
    decoder_points = noisy.float().clamp(-1.0, 1.0)
    decoder_points[..., 0] *= 50.0
    decoder_points[..., 1] *= 20.0
    trajectory_features = head.plan_anchor_encoder(
        _sine_xy_embedding(decoder_points).flatten(-2)
    ).view(batch_size, 20, -1)
    time_embedding = head.time_mlp(timesteps).view(batch_size, 1, -1)
    poses_reg, poses_cls, _ = head.diff_decoder(
        trajectory_features,
        decoder_points,
        cross_bev,
        bev_spatial_shape,
        agents_query,
        trajectory_query,
        time_embedding,
        status_encoding[:, None],
        None,
    )
    if len(poses_reg) != 1 or len(poses_cls) != 1:
        raise TrainingBoundaryError("official diff_decoder returned an unexpected layer count")
    regression = poses_reg[0]
    classification = poses_cls[0]
    if tuple(regression.shape) != (batch_size, 20, 8, 3):
        raise TrainingBoundaryError("official trajectory regression shape changed")
    if tuple(classification.shape) != (batch_size, 1, 20):
        raise TrainingBoundaryError("official trajectory classification shape changed")
    loss = head.loss_computer(
        regression,
        classification.squeeze(1),
        {"trajectory": target_trajectory},
        anchor,
    )
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss).item():
        raise TrainingBoundaryError("official trajectory loss must be a finite scalar")
    selected = classification.squeeze(1).argmax(dim=1)
    prediction = regression[torch.arange(batch_size, device=camera.device), selected]
    if tuple(prediction.shape) != (batch_size, 8, 3) or not torch.isfinite(prediction).all().item():
        raise TrainingBoundaryError("official predicted trajectory must be finite [B,8,3]")
    return loss, prediction


def supervised_trajectory_loss(
    model: nn.Module,
    camera: torch.Tensor,
    lidar: torch.Tensor,
    status: torch.Tensor,
    target_trajectory: torch.Tensor,
) -> torch.Tensor:
    return supervised_trajectory_output(
        model, camera, lidar, status, target_trajectory
    )[0]


def save_checkpoint_atomic(
    output_path: Path,
    checkpoint: Mapping[str, Any],
    model: nn.Module,
) -> Path:
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise TrainingBoundaryError("checkpoint must contain a state_dict mapping")
    source_runtime = normalize_checkpoint_state(raw_state)
    trained_runtime = model.state_dict()
    if len(trained_runtime) != RUNTIME_KEY_COUNT or set(trained_runtime) != set(source_runtime):
        raise TrainingBoundaryError("trained model state no longer matches official SEL schema")

    merged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for raw_key, original in raw_state.items():
        key = raw_key[len(CHECKPOINT_PREFIX) :]
        merged[raw_key] = (
            original
            if key == VOCAB_KEY
            else trained_runtime[key].detach().to(device="cpu").clone()
        )
    if len(merged) != RAW_KEY_COUNT:
        raise TrainingBoundaryError("export did not preserve the 972-key SEL schema")

    exported = dict(checkpoint)
    exported["state_dict"] = merged
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(exported, str(temporary))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


__all__ = (
    "CHECKPOINT_PREFIX",
    "RAW_KEY_COUNT",
    "RUNTIME_KEY_COUNT",
    "VOCAB_KEY",
    "configure_finetuning",
    "load_official_model",
    "normalize_checkpoint_state",
    "save_checkpoint_atomic",
    "supervised_trajectory_loss",
    "supervised_trajectory_output",
)
