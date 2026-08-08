from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Tuple, Union

import numpy as np

from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError, MoraiDDV2Error

from .geometry import Pose, transform_points


@dataclass(frozen=True)
class PointFieldLayout:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class PointCloudLayout:
    fields: Tuple[PointFieldLayout, ...]
    point_step: int
    is_bigendian: bool


@dataclass(frozen=True)
class LidarConfig:
    ordered_steps: Tuple[str, ...]
    zero_padding_point_xyz: Tuple[float, float, float]
    self_mask_x_m: Tuple[float, float]
    self_mask_y_m: Tuple[float, float]
    self_mask_z_m: Tuple[float, float]
    self_mask_boundaries_inclusive: bool
    height_min_exclusive_m: float
    height_max_exclusive_m: float
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]
    pixels_per_meter: int
    bin_size_m: float
    bins_xy: Tuple[int, int]
    histogram_axis_order: Tuple[str, str]
    count_clip: int
    normalization: str
    rotate_or_flip: bool
    ground_plane_channel: bool
    cache_tensor_name: str
    cache_tensor_dtype: str
    cache_tensor_shape: Tuple[int, int]
    cache_tensor_value_range: Tuple[float, float]
    model_tensor_name: str
    model_tensor_dtype: str
    model_tensor_shape: Tuple[int, int, int]
    model_tensor_value_range: Tuple[float, float]


class PointCloudLayoutError(MoraiDDV2Error):
    """Raised when a PointCloud2 field layout violates the domain contract."""


class LidarPointError(MoraiDDV2Error):
    """Raised when decoded LiDAR point values are malformed or non-finite."""


class EmptyLidarError(MoraiDDV2Error):
    """Raised when no LiDAR point contributes to the canonical BEV."""


class LidarTensorContractError(MoraiDDV2Error):
    """Raised when a cached LiDAR count tensor violates its exact contract."""


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ConfigurationError("{}.{} is required".format(context, key)) from error


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("{} must be an object".format(context))
    return value


def _sequence(value: Any, length: int, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("{} must be an array".format(context))
    if len(value) != length:
        raise ConfigurationError(
            "{} must contain {} values".format(context, length)
        )
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError("{} must be a finite number".format(context))
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError("{} must be a finite number".format(context))
    return result


def _integer(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError("{} must be an integer".format(context))
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError("{} must be a boolean".format(context))
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError("{} must be a nonempty string".format(context))
    return value


def _float_tuple(value: Any, length: int, context: str) -> Tuple[float, ...]:
    sequence = _sequence(value, length, context)
    return tuple(
        _finite_number(item, "{}[{}]".format(context, index))
        for index, item in enumerate(sequence)
    )


def _integer_tuple(value: Any, length: int, context: str) -> Tuple[int, ...]:
    sequence = _sequence(value, length, context)
    return tuple(
        _integer(item, "{}[{}]".format(context, index))
        for index, item in enumerate(sequence)
    )


def _string_tuple(value: Any, length: int, context: str) -> Tuple[str, ...]:
    sequence = _sequence(value, length, context)
    return tuple(
        _string(item, "{}[{}]".format(context, index))
        for index, item in enumerate(sequence)
    )


def _is_typed_tuple(value: Any, length: int, item_type: type) -> bool:
    return (
        type(value) is tuple
        and len(value) == length
        and all(type(item) is item_type for item in value)
    )


def _validate_lidar_config(config: LidarConfig) -> None:
    if not isinstance(config, LidarConfig):
        raise ConfigurationError("config must be a LidarConfig")
    if not _is_typed_tuple(config.ordered_steps, 5, str):
        raise ConfigurationError("LiDAR ordered_steps must be five strings")
    for name, value, length in (
        ("zero_padding_point_xyz", config.zero_padding_point_xyz, 3),
        ("self_mask_x_m", config.self_mask_x_m, 2),
        ("self_mask_y_m", config.self_mask_y_m, 2),
        ("self_mask_z_m", config.self_mask_z_m, 2),
        ("x_range_m", config.x_range_m, 2),
        ("y_range_m", config.y_range_m, 2),
        ("cache_tensor_value_range", config.cache_tensor_value_range, 2),
        ("model_tensor_value_range", config.model_tensor_value_range, 2),
    ):
        if not _is_typed_tuple(value, length, float):
            raise ConfigurationError(
                "LiDAR {} must be a float tuple".format(name)
            )
    for name, value in (
        ("height_min_exclusive_m", config.height_min_exclusive_m),
        ("height_max_exclusive_m", config.height_max_exclusive_m),
        ("bin_size_m", config.bin_size_m),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise ConfigurationError("LiDAR {} must be finite float".format(name))
    for name, value in (
        ("pixels_per_meter", config.pixels_per_meter),
        ("count_clip", config.count_clip),
    ):
        if type(value) is not int:
            raise ConfigurationError("LiDAR {} must be an integer".format(name))
    if not _is_typed_tuple(config.bins_xy, 2, int):
        raise ConfigurationError("LiDAR bins_xy must be an integer tuple")
    if not _is_typed_tuple(config.cache_tensor_shape, 2, int):
        raise ConfigurationError("LiDAR cache shape must be an integer tuple")
    if not _is_typed_tuple(config.model_tensor_shape, 3, int):
        raise ConfigurationError("LiDAR model shape must be an integer tuple")
    if not _is_typed_tuple(config.histogram_axis_order, 2, str):
        raise ConfigurationError("LiDAR histogram axes must be a string tuple")
    for name, value in (
        ("self_mask_boundaries_inclusive", config.self_mask_boundaries_inclusive),
        ("rotate_or_flip", config.rotate_or_flip),
        ("ground_plane_channel", config.ground_plane_channel),
    ):
        if type(value) is not bool:
            raise ConfigurationError("LiDAR {} must be boolean".format(name))
    for name, value in (
        ("normalization", config.normalization),
        ("cache_tensor_name", config.cache_tensor_name),
        ("cache_tensor_dtype", config.cache_tensor_dtype),
        ("model_tensor_name", config.model_tensor_name),
        ("model_tensor_dtype", config.model_tensor_dtype),
    ):
        if type(value) is not str or not value:
            raise ConfigurationError("LiDAR {} must be a string".format(name))
    if config.ordered_steps != (
        "DROP_ZERO_PADDING",
        "TRANSFORM_TO_BASE_LINK",
        "APPLY_3D_SELF_MASK",
        "APPLY_HEIGHT_FILTER",
        "BUILD_XY_HISTOGRAM",
    ):
        raise ConfigurationError("LiDAR processing steps are not canonical")
    if config.zero_padding_point_xyz != (0.0, 0.0, 0.0):
        raise ConfigurationError("LiDAR zero padding marker must be [0,0,0]")
    if config.self_mask_x_m != (0.0, 2.0):
        raise ConfigurationError("LiDAR self-mask x range must be [0,2]")
    if config.self_mask_y_m != (-1.0, 1.0):
        raise ConfigurationError("LiDAR self-mask y range must be [-1,1]")
    if config.self_mask_z_m != (0.8, 1.6):
        raise ConfigurationError("LiDAR self-mask z range must be [0.8,1.6]")
    if config.self_mask_boundaries_inclusive is not True:
        raise ConfigurationError("LiDAR self-mask boundaries must be inclusive")
    if config.height_min_exclusive_m != 0.2:
        raise ConfigurationError("LiDAR minimum height must be exclusive 0.2 m")
    if config.height_max_exclusive_m != 100.0:
        raise ConfigurationError("LiDAR maximum height must be exclusive 100 m")
    if config.x_range_m != (-32.0, 32.0):
        raise ConfigurationError("LiDAR x range must be [-32,32]")
    if config.y_range_m != (-32.0, 32.0):
        raise ConfigurationError("LiDAR y range must be [-32,32]")
    if config.pixels_per_meter != 4:
        raise ConfigurationError("LiDAR pixels_per_meter must be 4")
    if config.bin_size_m != 0.25:
        raise ConfigurationError("LiDAR bin size must be 0.25 m")
    if config.bins_xy != (256, 256):
        raise ConfigurationError("LiDAR bins must be [256,256]")
    if config.histogram_axis_order != ("x_bin", "y_bin"):
        raise ConfigurationError("LiDAR histogram axes must be [x_bin,y_bin]")
    if config.count_clip != 5:
        raise ConfigurationError("LiDAR count clip must be 5")
    if config.normalization != "divide_by_5":
        raise ConfigurationError("LiDAR normalization must be divide_by_5")
    if config.rotate_or_flip is not False:
        raise ConfigurationError("LiDAR rotation and flipping must be disabled")
    if config.ground_plane_channel is not False:
        raise ConfigurationError("LiDAR ground-plane channel must be disabled")
    if (
        config.cache_tensor_name != "lidar_count"
        or config.cache_tensor_dtype != "uint8"
        or config.cache_tensor_shape != (256, 256)
        or config.cache_tensor_value_range != (0.0, 5.0)
    ):
        raise ConfigurationError("LiDAR cache tensor contract is not canonical")
    if (
        config.model_tensor_name != "lidar_feature"
        or config.model_tensor_dtype != "float32"
        or config.model_tensor_shape != (1, 256, 256)
        or config.model_tensor_value_range != (0.0, 1.0)
    ):
        raise ConfigurationError("LiDAR model tensor contract is not canonical")
    x_width = config.x_range_m[1] - config.x_range_m[0]
    y_width = config.y_range_m[1] - config.y_range_m[0]
    if x_width / config.bins_xy[0] != config.bin_size_m:
        raise ConfigurationError("LiDAR x range and bin size are inconsistent")
    if y_width / config.bins_xy[1] != config.bin_size_m:
        raise ConfigurationError("LiDAR y range and bin size are inconsistent")
    if 1.0 / config.bin_size_m != float(config.pixels_per_meter):
        raise ConfigurationError("LiDAR bin size and pixels_per_meter are inconsistent")


def lidar_config_from_resolved_domain(
    resolved_domain: Union[ResolvedDomain, Mapping[str, Any]],
) -> LidarConfig:
    """Build the complete LiDAR policy from a verified resolved-domain value."""

    if isinstance(resolved_domain, ResolvedDomain):
        document = resolved_domain.document
    elif isinstance(resolved_domain, Mapping):
        document = resolved_domain
    else:
        raise ConfigurationError(
            "resolved_domain must be a ResolvedDomain or canonical mapping"
        )
    root = _mapping(document, "resolved_domain")
    processing = _mapping(
        _required(root, "lidar_processing", "resolved_domain"),
        "resolved_domain.lidar_processing",
    )
    mask = _mapping(
        _required(processing, "self_mask_base_link_m", "lidar_processing"),
        "lidar_processing.self_mask_base_link_m",
    )
    height = _mapping(
        _required(processing, "height_filter_m", "lidar_processing"),
        "lidar_processing.height_filter_m",
    )
    cache = _mapping(
        _required(processing, "cache_tensor", "lidar_processing"),
        "lidar_processing.cache_tensor",
    )
    model = _mapping(
        _required(processing, "model_tensor", "lidar_processing"),
        "lidar_processing.model_tensor",
    )
    steps_raw = _required(processing, "ordered_steps", "lidar_processing")
    if isinstance(steps_raw, (str, bytes)) or not isinstance(steps_raw, Sequence):
        raise ConfigurationError("lidar_processing.ordered_steps must be an array")

    config = LidarConfig(
        ordered_steps=tuple(
            _string(value, "lidar_processing.ordered_steps[{}]".format(index))
            for index, value in enumerate(steps_raw)
        ),
        zero_padding_point_xyz=_float_tuple(
            _required(processing, "zero_padding_point_xyz", "lidar_processing"),
            3,
            "lidar_processing.zero_padding_point_xyz",
        ),
        self_mask_x_m=_float_tuple(
            _required(mask, "x", "self_mask_base_link_m"),
            2,
            "self_mask_base_link_m.x",
        ),
        self_mask_y_m=_float_tuple(
            _required(mask, "y", "self_mask_base_link_m"),
            2,
            "self_mask_base_link_m.y",
        ),
        self_mask_z_m=_float_tuple(
            _required(mask, "z", "self_mask_base_link_m"),
            2,
            "self_mask_base_link_m.z",
        ),
        self_mask_boundaries_inclusive=_boolean(
            _required(mask, "boundaries_inclusive", "self_mask_base_link_m"),
            "self_mask_base_link_m.boundaries_inclusive",
        ),
        height_min_exclusive_m=_finite_number(
            _required(height, "min_exclusive", "height_filter_m"),
            "height_filter_m.min_exclusive",
        ),
        height_max_exclusive_m=_finite_number(
            _required(height, "max_exclusive", "height_filter_m"),
            "height_filter_m.max_exclusive",
        ),
        x_range_m=_float_tuple(
            _required(processing, "x_range_m", "lidar_processing"),
            2,
            "lidar_processing.x_range_m",
        ),
        y_range_m=_float_tuple(
            _required(processing, "y_range_m", "lidar_processing"),
            2,
            "lidar_processing.y_range_m",
        ),
        pixels_per_meter=_integer(
            _required(processing, "pixels_per_meter", "lidar_processing"),
            "lidar_processing.pixels_per_meter",
        ),
        bin_size_m=_finite_number(
            _required(processing, "bin_size_m", "lidar_processing"),
            "lidar_processing.bin_size_m",
        ),
        bins_xy=_integer_tuple(
            _required(processing, "bins_xy", "lidar_processing"),
            2,
            "lidar_processing.bins_xy",
        ),
        histogram_axis_order=_string_tuple(
            _required(processing, "histogram_axis_order", "lidar_processing"),
            2,
            "lidar_processing.histogram_axis_order",
        ),
        count_clip=_integer(
            _required(processing, "count_clip", "lidar_processing"),
            "lidar_processing.count_clip",
        ),
        normalization=_string(
            _required(processing, "normalization", "lidar_processing"),
            "lidar_processing.normalization",
        ),
        rotate_or_flip=_boolean(
            _required(processing, "rotate_or_flip", "lidar_processing"),
            "lidar_processing.rotate_or_flip",
        ),
        ground_plane_channel=_boolean(
            _required(processing, "ground_plane_channel", "lidar_processing"),
            "lidar_processing.ground_plane_channel",
        ),
        cache_tensor_name=_string(
            _required(cache, "name", "cache_tensor"), "cache_tensor.name"
        ),
        cache_tensor_dtype=_string(
            _required(cache, "dtype", "cache_tensor"), "cache_tensor.dtype"
        ),
        cache_tensor_shape=_integer_tuple(
            _required(cache, "shape_per_sample", "cache_tensor"),
            2,
            "cache_tensor.shape_per_sample",
        ),
        cache_tensor_value_range=_float_tuple(
            _required(cache, "value_range", "cache_tensor"),
            2,
            "cache_tensor.value_range",
        ),
        model_tensor_name=_string(
            _required(model, "name", "model_tensor"), "model_tensor.name"
        ),
        model_tensor_dtype=_string(
            _required(model, "dtype", "model_tensor"), "model_tensor.dtype"
        ),
        model_tensor_shape=_integer_tuple(
            _required(model, "shape_per_sample", "model_tensor"),
            3,
            "model_tensor.shape_per_sample",
        ),
        model_tensor_value_range=_float_tuple(
            _required(model, "value_range", "model_tensor"),
            2,
            "model_tensor.value_range",
        ),
    )
    _validate_lidar_config(config)
    return config


def validate_pointcloud_layout(layout: PointCloudLayout) -> None:
    """Validate ROS-free PointCloud2 field metadata for canonical XYZI decode."""

    if not isinstance(layout, PointCloudLayout):
        raise PointCloudLayoutError("layout must be a PointCloudLayout")
    if not isinstance(layout.point_step, int) or isinstance(layout.point_step, bool):
        raise PointCloudLayoutError("point_step must be a positive integer")
    if layout.point_step <= 0:
        raise PointCloudLayoutError("point_step must be a positive integer")
    if not isinstance(layout.is_bigendian, bool):
        raise PointCloudLayoutError("is_bigendian must be a boolean")
    if layout.is_bigendian:
        raise PointCloudLayoutError("big-endian PointCloud2 layout is unsupported")
    if isinstance(layout.fields, (str, bytes)) or not isinstance(
        layout.fields, Sequence
    ):
        raise PointCloudLayoutError("fields must be a sequence")

    by_name = {}
    for field in layout.fields:
        if not isinstance(field, PointFieldLayout):
            raise PointCloudLayoutError("fields must contain PointFieldLayout values")
        if not isinstance(field.name, str) or not field.name:
            raise PointCloudLayoutError("PointCloud field name must be nonempty")
        if field.name in by_name:
            raise PointCloudLayoutError(
                "duplicate PointCloud field: {}".format(field.name)
            )
        by_name[field.name] = field

    required_names = ("x", "y", "z", "intensity")
    missing = tuple(name for name in required_names if name not in by_name)
    if missing:
        raise PointCloudLayoutError(
            "missing required PointCloud fields: {}".format(",".join(missing))
        )

    intervals = []
    for name in required_names:
        field = by_name[name]
        if not isinstance(field.datatype, int) or isinstance(field.datatype, bool):
            raise PointCloudLayoutError("{}.datatype must be ROS FLOAT32 7".format(name))
        if field.datatype != 7:
            raise PointCloudLayoutError("{}.datatype must be ROS FLOAT32 7".format(name))
        if not isinstance(field.count, int) or isinstance(field.count, bool):
            raise PointCloudLayoutError("{}.count must be 1".format(name))
        if field.count != 1:
            raise PointCloudLayoutError("{}.count must be 1".format(name))
        if not isinstance(field.offset, int) or isinstance(field.offset, bool):
            raise PointCloudLayoutError("{}.offset must be non-negative".format(name))
        if field.offset < 0:
            raise PointCloudLayoutError("{}.offset must be non-negative".format(name))
        end = field.offset + 4
        if end > layout.point_step:
            raise PointCloudLayoutError(
                "{} field extends beyond point_step".format(name)
            )
        intervals.append((field.offset, end, name))

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise PointCloudLayoutError(
                "PointCloud fields {} and {} overlap".format(previous[2], current[2])
            )


def _validated_points(points_xyzi_lidar: np.ndarray) -> np.ndarray:
    if not isinstance(points_xyzi_lidar, np.ndarray):
        raise LidarPointError("points_xyzi_lidar must be a NumPy ndarray")
    if points_xyzi_lidar.ndim != 2 or points_xyzi_lidar.shape[1] != 4:
        raise LidarPointError("points_xyzi_lidar must have shape [N,4]")
    if points_xyzi_lidar.dtype.kind not in ("i", "u", "f"):
        raise LidarPointError("points_xyzi_lidar must contain real numeric values")
    try:
        xyz = np.asarray(points_xyzi_lidar[:, :3], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise LidarPointError(
            "points_xyzi_lidar must contain real numeric values"
        ) from error
    if not np.isfinite(xyz).all():
        raise LidarPointError("LiDAR x/y/z values must be finite")
    return xyz


def build_lidar_count(
    points_xyzi_lidar: np.ndarray,
    base_from_lidar: Pose,
    config: LidarConfig,
) -> np.ndarray:
    """Build the canonical base-link XY count histogram from decoded XYZI."""

    _validate_lidar_config(config)
    xyz_lidar = _validated_points(points_xyzi_lidar)
    zero_marker = np.asarray(config.zero_padding_point_xyz, dtype=np.float64)
    xyz_lidar = xyz_lidar[~np.all(xyz_lidar == zero_marker, axis=1)]

    xyz_base = transform_points(xyz_lidar, base_from_lidar)
    if not np.isfinite(xyz_base).all():
        raise LidarPointError("base-link LiDAR x/y/z values must be finite")

    mask_x = config.self_mask_x_m
    mask_y = config.self_mask_y_m
    mask_z = config.self_mask_z_m
    inside_self = (
        (mask_x[0] <= xyz_base[:, 0])
        & (xyz_base[:, 0] <= mask_x[1])
        & (mask_y[0] <= xyz_base[:, 1])
        & (xyz_base[:, 1] <= mask_y[1])
        & (mask_z[0] <= xyz_base[:, 2])
        & (xyz_base[:, 2] <= mask_z[1])
    )
    xyz_base = xyz_base[~inside_self]
    xyz_base = xyz_base[
        (config.height_min_exclusive_m < xyz_base[:, 2])
        & (xyz_base[:, 2] < config.height_max_exclusive_m)
    ]

    x_edges = np.linspace(
        config.x_range_m[0],
        config.x_range_m[1],
        config.bins_xy[0] + 1,
        dtype=np.float64,
    )
    y_edges = np.linspace(
        config.y_range_m[0],
        config.y_range_m[1],
        config.bins_xy[1] + 1,
        dtype=np.float64,
    )
    counts, _ = np.histogramdd(xyz_base[:, :2], bins=(x_edges, y_edges))
    if not np.any(counts):
        raise EmptyLidarError("no LiDAR point contributes to the canonical BEV")
    clipped = np.clip(counts, 0, config.count_clip)
    result = np.ascontiguousarray(clipped.astype(np.uint8))
    if result.shape != config.cache_tensor_shape:
        raise LidarTensorContractError("lidar_count has an unexpected shape")
    return result


def lidar_model_tensor(lidar_count: np.ndarray) -> np.ndarray:
    """Convert exact cache counts to the one-channel DDV2 float tensor."""

    if not isinstance(lidar_count, np.ndarray):
        raise LidarTensorContractError("lidar_count must be a NumPy ndarray")
    if lidar_count.shape != (256, 256):
        raise LidarTensorContractError("lidar_count must have shape [256,256]")
    if lidar_count.dtype != np.uint8:
        raise LidarTensorContractError("lidar_count must have dtype uint8")
    if not lidar_count.flags.c_contiguous:
        raise LidarTensorContractError("lidar_count must be C-contiguous")
    if bool(np.any(lidar_count > 5)):
        raise LidarTensorContractError("lidar_count values must be in [0,5]")
    result = np.ascontiguousarray(
        lidar_count[np.newaxis, :, :].astype(np.float32) / np.float32(5.0)
    )
    if result.shape != (1, 256, 256) or result.dtype != np.float32:
        raise LidarTensorContractError("lidar_feature conversion violated its contract")
    return result
