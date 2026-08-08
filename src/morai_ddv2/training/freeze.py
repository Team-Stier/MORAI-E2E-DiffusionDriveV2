from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from .contracts import (
    TRAINABLE_PREFIXES,
    FreezeManifest,
    TrainingBoundaryError,
)


def _is_trainable_module(name: str) -> bool:
    return any(
        name == prefix[:-1] or name.startswith(prefix) for prefix in TRAINABLE_PREFIXES
    )


def _build_manifest(model: nn.Module) -> FreezeManifest:
    trainable = tuple(
        name
        for name, _ in model.named_parameters()
        if name.startswith(TRAINABLE_PREFIXES)
    )
    frozen = tuple(
        name
        for name, _ in model.named_parameters()
        if not name.startswith(TRAINABLE_PREFIXES)
    )
    named_buffers = dict(model.named_buffers())
    batchnorm_buffers = []
    for module_name, module in model.named_modules():
        if not _is_trainable_module(module_name) or not isinstance(
            module, nn.modules.batchnorm._BatchNorm
        ):
            continue
        for buffer_name in (
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ):
            name = f"{module_name}.{buffer_name}"
            if name in named_buffers:
                batchnorm_buffers.append(name)
    return FreezeManifest(trainable, frozen, trainable + tuple(batchnorm_buffers))


def configure_training_boundary(model: nn.Module) -> FreezeManifest:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    for module_name, module in model.named_modules():
        if _is_trainable_module(module_name):
            module.train()
    manifest = _build_manifest(model)
    assert_training_boundary(model, manifest)
    return manifest


def assert_training_boundary(model: nn.Module, manifest: FreezeManifest) -> None:
    expected = _build_manifest(model)
    if manifest != expected:
        raise TrainingBoundaryError("freeze manifest does not match model")
    if model.training:
        raise TrainingBoundaryError("model must remain globally eval")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != name.startswith(TRAINABLE_PREFIXES):
            raise TrainingBoundaryError(f"requires_grad boundary drifted: {name}")
    for name, module in model.named_modules():
        if module.training != _is_trainable_module(name):
            raise TrainingBoundaryError(f"module mode boundary drifted: {name}")


def extract_mutable_state(
    model: nn.Module, manifest: FreezeManifest
) -> OrderedDict[str, torch.Tensor]:
    assert_training_boundary(model, manifest)
    state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    return OrderedDict(
        (name, state[name].detach().clone()) for name in manifest.mutable_state_entries
    )


def build_optimizer(
    model: nn.Module,
    manifest: FreezeManifest,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    assert_training_boundary(model, manifest)
    named_parameters = dict(model.named_parameters())
    parameters = [named_parameters[name] for name in manifest.trainable_parameters]
    if not parameters:
        raise TrainingBoundaryError("no trainable parameters")
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
