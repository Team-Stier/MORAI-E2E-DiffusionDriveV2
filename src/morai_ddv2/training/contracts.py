from __future__ import annotations

from dataclasses import dataclass


TRAINABLE_PREFIXES = (
    "_backbone.image_encoder.",
    "_backbone.lidar_encoder.",
    "_backbone.transformers.",
    "_backbone.lidar_channel_to_img.",
    "_backbone.img_channel_to_lidar.",
)


class TrainingBoundaryError(RuntimeError):
    """Raised when the DDV2 trainable-state boundary has drifted."""


@dataclass(frozen=True)
class FreezeManifest:
    trainable_parameters: tuple[str, ...]
    frozen_parameters: tuple[str, ...]
    mutable_state_entries: tuple[str, ...]
