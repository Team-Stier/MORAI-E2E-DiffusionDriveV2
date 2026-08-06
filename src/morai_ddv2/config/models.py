from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Tuple


class FrozenList(Sequence):
    """Immutable JSON-array view that compares equal to ordinary lists."""

    __slots__ = ("_values",)

    def __init__(self, values: Sequence) -> None:
        self._values = tuple(values)

    def __getitem__(self, index: Any) -> Any:
        return self._values[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FrozenList):
            return self._values == other._values
        if isinstance(other, (list, tuple)):
            return list(self._values) == list(other)
        return False

    def append(self, value: Any) -> None:
        del value
        raise TypeError("resolved domain document is immutable")

    def __repr__(self) -> str:
        return repr(list(self._values))


def freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, FrozenList):
        return value
    if isinstance(value, (list, tuple)):
        return FrozenList([freeze_json(item) for item in value])
    return value


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (FrozenList, list, tuple)):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class CameraSensor:
    sensor_id: int
    role: str
    topic: str
    frame_id: str
    translation_base_from_camera_m: Tuple[float, float, float]
    morai_euler_deg: Tuple[float, float, float]
    quaternion_base_from_camera_xyzw: Tuple[float, float, float, float]
    source_period_s: float
    sensor_size_mm: Tuple[float, float]
    focal_length_mm: float
    focal_length_pixel: float
    principal_point_xy: Tuple[float, float]
    resolution_wh: Tuple[int, int]
    compression_quality: int
    horizontal_fov_deg: float
    lens_distortion: Tuple[float, float, float]


@dataclass(frozen=True)
class LidarSensor:
    sensor_id: int
    role: str
    topic: str
    frame_id: str
    translation_base_from_lidar_m: Tuple[float, float, float]
    morai_euler_deg: Tuple[float, float, float]
    quaternion_base_from_lidar_xyzw: Tuple[float, float, float, float]
    source_period_s: float
    maximum_range_m: float
    noise_ratio: float


@dataclass(frozen=True)
class SensorPreset:
    path: Path
    sha256: str
    cameras: Tuple[CameraSensor, CameraSensor, CameraSensor]
    lidar: LidarSensor


@dataclass(frozen=True)
class ResolvedDomainArtifact:
    document: Mapping[str, Any]
    sha256: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("resolved domain SHA-256 must be 64 lowercase hexadecimal characters")
        object.__setattr__(self, "document", freeze_json(self.document))

    @property
    def publication_bytes(self) -> bytes:
        from .canonical import canonical_json_bytes

        publication = thaw_json(self.document)
        publication["resolved_domain_sha256"] = self.sha256
        return canonical_json_bytes(publication)


@dataclass(frozen=True)
class ResolvedDomain:
    document: Mapping[str, Any]
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "document", freeze_json(self.document))
