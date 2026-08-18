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
    """The official checkpoint or fine-tuning boundary is not what we expect."""


@dataclass(frozen=True)
class FreezeManifest:
    trainable_parameters: tuple[str, ...]
