from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Tuple, Union

import cv2
import numpy as np

from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError, MoraiDDV2Error


class CameraDecodeError(MoraiDDV2Error):
    """Raised when a compressed camera payload cannot produce a color image."""


class CameraResolutionError(MoraiDDV2Error):
    """Raised after successful decode when physical camera dimensions differ."""

    def __init__(
        self,
        *,
        actual_wh: Tuple[int, int],
        expected_wh: Tuple[int, int],
        camera_name: str = "camera",
    ) -> None:
        self.actual_wh = actual_wh
        self.expected_wh = expected_wh
        self.camera_name = camera_name
        super().__init__(
            "{} resolution mismatch: expected {}x{}, got {}x{}".format(
                camera_name,
                expected_wh[0],
                expected_wh[1],
                actual_wh[0],
                actual_wh[1],
            )
        )


class TensorContractError(MoraiDDV2Error):
    """Raised when an image or model tensor violates the v1 domain contract."""


def _validated_wh(value: object, name: str) -> Tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TensorContractError("{} must be a (width, height) tuple".format(name))
    width, height = value
    if any(
        not isinstance(dimension, int) or isinstance(dimension, bool)
        for dimension in (width, height)
    ):
        raise TensorContractError("{} dimensions must be integers".format(name))
    if width <= 0 or height <= 0:
        raise TensorContractError("{} dimensions must be positive".format(name))
    return width, height


@dataclass(frozen=True)
class CameraAdapterConfig:
    """Immutable geometry for the MORAI DDV2 v1 three-camera adapter."""

    left_source_wh: Tuple[int, int]
    front_source_wh: Tuple[int, int]
    right_source_wh: Tuple[int, int]
    left_resize_wh: Tuple[int, int]
    front_resize_wh: Tuple[int, int]
    right_resize_wh: Tuple[int, int]
    stitched_wh: Tuple[int, int]
    outer_crop_x_start: int
    outer_crop_x_end_exclusive: int
    output_wh: Tuple[int, int]
    output_boundaries_x: Tuple[int, int, int, int]
    view_order: Tuple[str, str, str]
    interpolation: int

    def __post_init__(self) -> None:
        _validated_wh(self.left_source_wh, "left_source_wh")
        _validated_wh(self.front_source_wh, "front_source_wh")
        _validated_wh(self.right_source_wh, "right_source_wh")

        left_resize_wh = _validated_wh(self.left_resize_wh, "left_resize_wh")
        front_resize_wh = _validated_wh(self.front_resize_wh, "front_resize_wh")
        right_resize_wh = _validated_wh(self.right_resize_wh, "right_resize_wh")
        stitched_wh = _validated_wh(self.stitched_wh, "stitched_wh")
        output_wh = _validated_wh(self.output_wh, "output_wh")

        if self.view_order != ("left", "front", "right"):
            raise TensorContractError("camera view order must be left, front, right")
        if self.interpolation != cv2.INTER_AREA:
            raise TensorContractError("camera interpolation must be cv2.INTER_AREA")
        if not isinstance(self.outer_crop_x_start, int) or isinstance(
            self.outer_crop_x_start, bool
        ):
            raise TensorContractError("outer_crop_x_start must be an integer")
        if not isinstance(self.outer_crop_x_end_exclusive, int) or isinstance(
            self.outer_crop_x_end_exclusive, bool
        ):
            raise TensorContractError("outer_crop_x_end_exclusive must be an integer")

        resized_heights = {
            left_resize_wh[1],
            front_resize_wh[1],
            right_resize_wh[1],
        }
        expected_stitched_wh = (
            left_resize_wh[0] + front_resize_wh[0] + right_resize_wh[0],
            left_resize_wh[1],
        )
        if len(resized_heights) != 1 or stitched_wh != expected_stitched_wh:
            raise TensorContractError("resized camera views must form stitched_wh")
        if not (
            0
            <= self.outer_crop_x_start
            < self.outer_crop_x_end_exclusive
            <= stitched_wh[0]
        ):
            raise TensorContractError("camera outer crop must lie inside stitched width")
        expected_output_wh = (
            self.outer_crop_x_end_exclusive - self.outer_crop_x_start,
            stitched_wh[1],
        )
        if output_wh != expected_output_wh:
            raise TensorContractError("camera crop must produce output_wh")

        left_end = left_resize_wh[0] - self.outer_crop_x_start
        front_end = left_end + front_resize_wh[0]
        expected_boundaries = (0, left_end, front_end, output_wh[0])
        if self.output_boundaries_x != expected_boundaries:
            raise TensorContractError("camera output boundaries do not match the crop")


def _required_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("{} must be an object".format(context))
    return value


def _required_value(
    mapping: Mapping[str, Any], key: str, context: str
) -> object:
    if key not in mapping:
        raise ConfigurationError("{}.{} is required".format(context, key))
    return mapping[key]


def _required_sequence(value: object, context: str) -> Tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ConfigurationError("{} must be an array".format(context))
    return tuple(value)


def _required_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError("{} must be an integer".format(context))
    return value


def _required_integer_tuple(
    value: object,
    *,
    length: int,
    context: str,
    positive: bool = False,
) -> Tuple[int, ...]:
    items = _required_sequence(value, context)
    if len(items) != length:
        raise ConfigurationError(
            "{} must contain {} integers".format(context, length)
        )
    result = tuple(
        _required_integer(item, "{}[{}]".format(context, index))
        for index, item in enumerate(items)
    )
    if positive and any(item <= 0 for item in result):
        raise ConfigurationError("{} values must be positive".format(context))
    return result


def _required_number_tuple(
    value: object, *, length: int, context: str
) -> Tuple[float, ...]:
    items = _required_sequence(value, context)
    if len(items) != length:
        raise ConfigurationError(
            "{} must contain {} finite numbers".format(context, length)
        )
    result = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ConfigurationError(
                "{}[{}] must be a finite number".format(context, index)
            )
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as error:
            raise ConfigurationError(
                "{}[{}] must be a finite number".format(context, index)
            ) from error
        if not math.isfinite(number):
            raise ConfigurationError(
                "{}[{}] must be finite".format(context, index)
            )
        result.append(number)
    return tuple(result)


def _expect_literal(
    mapping: Mapping[str, Any], key: str, expected: object, context: str
) -> None:
    actual = _required_value(mapping, key, context)
    if type(actual) is not type(expected) or actual != expected:
        raise ConfigurationError(
            "{}.{} must be {!r}".format(context, key, expected)
        )


def _resolved_document(
    resolved_domain: Union[ResolvedDomain, Mapping[str, Any]]
) -> Mapping[str, Any]:
    if isinstance(resolved_domain, ResolvedDomain):
        return _required_mapping(resolved_domain.document, "resolved_domain.document")
    return _required_mapping(resolved_domain, "resolved_domain")


def camera_adapter_config_from_resolved_domain(
    resolved_domain: Union[ResolvedDomain, Mapping[str, Any]]
) -> CameraAdapterConfig:
    """Build the v1 adapter from an immutable or raw resolved-domain mapping."""

    document = _resolved_document(resolved_domain)
    sensors = _required_mapping(
        _required_value(document, "sensors", "resolved_domain"),
        "resolved_domain.sensors",
    )
    camera_entries = _required_sequence(
        _required_value(sensors, "cameras", "resolved_domain.sensors"),
        "resolved_domain.sensors.cameras",
    )
    source_wh_by_role = {}
    expected_roles = {"left", "front", "right"}
    for index, entry_value in enumerate(camera_entries):
        context = "resolved_domain.sensors.cameras[{}]".format(index)
        entry = _required_mapping(entry_value, context)
        role = _required_value(entry, "role", context)
        if not isinstance(role, str) or role not in expected_roles:
            raise ConfigurationError("{}.role is not a required camera role".format(context))
        if role in source_wh_by_role:
            raise ConfigurationError("duplicate camera role: {}".format(role))
        source_wh_by_role[role] = _required_integer_tuple(
            _required_value(entry, "resolution_wh", context),
            length=2,
            context="{}.resolution_wh".format(context),
            positive=True,
        )
    if set(source_wh_by_role) != expected_roles:
        missing = sorted(expected_roles - set(source_wh_by_role))
        raise ConfigurationError("missing required camera roles: {}".format(missing))

    processing = _required_mapping(
        _required_value(document, "camera_processing", "resolved_domain"),
        "resolved_domain.camera_processing",
    )
    processing_context = "resolved_domain.camera_processing"
    for key, expected in (
        ("decoded_layout", "HWC"),
        ("decoded_dtype", "uint8"),
        ("decoded_color_order", "BGR"),
        ("output_color_order", "RGB"),
        ("exif_orientation_allowed", False),
        ("automatic_color_correction_allowed", False),
        ("interpolation", "cv2.INTER_AREA"),
    ):
        _expect_literal(processing, key, expected, processing_context)

    view_order_values = _required_sequence(
        _required_value(processing, "view_order", processing_context),
        "{}.view_order".format(processing_context),
    )
    if view_order_values != ("left", "front", "right"):
        raise ConfigurationError("camera view_order must be left, front, right")

    resize = _required_mapping(
        _required_value(processing, "resize_wh", processing_context),
        "{}.resize_wh".format(processing_context),
    )
    resize_by_role = {}
    for role in ("left", "front", "right"):
        resize_by_role[role] = _required_integer_tuple(
            _required_value(resize, role, "{}.resize_wh".format(processing_context)),
            length=2,
            context="{}.resize_wh.{}".format(processing_context, role),
            positive=True,
        )
    stitched_wh = _required_integer_tuple(
        _required_value(processing, "stitched_wh", processing_context),
        length=2,
        context="{}.stitched_wh".format(processing_context),
        positive=True,
    )
    crop_start = _required_integer(
        _required_value(processing, "outer_crop_x_start", processing_context),
        "{}.outer_crop_x_start".format(processing_context),
    )
    crop_end = _required_integer(
        _required_value(
            processing, "outer_crop_x_end_exclusive", processing_context
        ),
        "{}.outer_crop_x_end_exclusive".format(processing_context),
    )
    output_wh = _required_integer_tuple(
        _required_value(processing, "output_wh", processing_context),
        length=2,
        context="{}.output_wh".format(processing_context),
        positive=True,
    )
    output_boundaries = _required_integer_tuple(
        _required_value(processing, "output_boundaries_x", processing_context),
        length=4,
        context="{}.output_boundaries_x".format(processing_context),
    )

    cache = _required_mapping(
        _required_value(processing, "cache_tensor", processing_context),
        "{}.cache_tensor".format(processing_context),
    )
    cache_context = "{}.cache_tensor".format(processing_context)
    for key, expected in (
        ("name", "camera_rgb"),
        ("layout", "NHWC"),
        ("dtype", "uint8"),
    ):
        _expect_literal(cache, key, expected, cache_context)
    cache_shape = _required_integer_tuple(
        _required_value(cache, "shape_per_sample", cache_context),
        length=3,
        context="{}.shape_per_sample".format(cache_context),
        positive=True,
    )
    cache_range = _required_number_tuple(
        _required_value(cache, "value_range", cache_context),
        length=2,
        context="{}.value_range".format(cache_context),
    )

    model = _required_mapping(
        _required_value(processing, "model_tensor", processing_context),
        "{}.model_tensor".format(processing_context),
    )
    model_context = "{}.model_tensor".format(processing_context)
    for key, expected in (
        ("name", "camera_feature"),
        ("layout", "NCHW"),
        ("dtype", "float32"),
        ("conversion", "transpose_HWC_to_CHW_then_divide_255"),
        ("imagenet_normalization", False),
    ):
        _expect_literal(model, key, expected, model_context)
    model_shape = _required_integer_tuple(
        _required_value(model, "shape_per_sample", model_context),
        length=3,
        context="{}.shape_per_sample".format(model_context),
        positive=True,
    )
    model_range = _required_number_tuple(
        _required_value(model, "value_range", model_context),
        length=2,
        context="{}.value_range".format(model_context),
    )

    for field_name, actual, expected in (
        ("resize_wh.left", resize_by_role["left"], (341, 256)),
        ("resize_wh.front", resize_by_role["front"], (455, 256)),
        ("resize_wh.right", resize_by_role["right"], (341, 256)),
        ("stitched_wh", stitched_wh, (1137, 256)),
        ("outer_crop_x_start", crop_start, 56),
        ("outer_crop_x_end_exclusive", crop_end, 1080),
        ("output_wh", output_wh, (1024, 256)),
        ("output_boundaries_x", output_boundaries, (0, 285, 740, 1024)),
    ):
        if actual != expected:
            raise ConfigurationError(
                "{}.{} must be {!r}".format(
                    processing_context, field_name, expected
                )
            )
    if cache_shape != (output_wh[1], output_wh[0], 3) or cache_range != (0.0, 255.0):
        raise ConfigurationError("camera cache tensor semantics do not match output_wh")
    if model_shape != (3, output_wh[1], output_wh[0]) or model_range != (0.0, 1.0):
        raise ConfigurationError("camera model tensor semantics do not match output_wh")

    try:
        return CameraAdapterConfig(
            left_source_wh=source_wh_by_role["left"],
            front_source_wh=source_wh_by_role["front"],
            right_source_wh=source_wh_by_role["right"],
            left_resize_wh=resize_by_role["left"],
            front_resize_wh=resize_by_role["front"],
            right_resize_wh=resize_by_role["right"],
            stitched_wh=stitched_wh,
            outer_crop_x_start=crop_start,
            outer_crop_x_end_exclusive=crop_end,
            output_wh=output_wh,
            output_boundaries_x=output_boundaries,
            view_order=("left", "front", "right"),
            interpolation=cv2.INTER_AREA,
        )
    except TensorContractError as error:
        raise ConfigurationError("invalid camera adapter geometry") from error


def _decode_jpeg_bgr(payload: bytes) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("JPEG payload must be bytes-like")

    encoded = np.frombuffer(payload, dtype=np.uint8)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    try:
        decoded = cv2.imdecode(encoded, flags)
    except cv2.error as error:
        raise CameraDecodeError("unable to decode camera JPEG") from error

    if decoded is None:
        raise CameraDecodeError("unable to decode camera JPEG")
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.dtype != np.uint8
        or decoded.ndim != 3
        or decoded.shape[2] != 3
    ):
        raise CameraDecodeError("decoded JPEG must be HWC uint8 with three channels")
    return np.ascontiguousarray(decoded)


def inspect_jpeg(payload: bytes) -> Tuple[int, int]:
    """Decode enough to validate a JPEG and return physical ``(width, height)``."""

    decoded = _decode_jpeg_bgr(payload)
    return int(decoded.shape[1]), int(decoded.shape[0])


def decode_jpeg_rgb(
    payload: bytes, expected_wh: Tuple[int, int]
) -> np.ndarray:
    """Decode one JPEG, validate physical size, and convert OpenCV BGR to RGB."""

    expected = _validated_wh(expected_wh, "expected_wh")
    decoded_bgr = _decode_jpeg_bgr(payload)
    actual = (int(decoded_bgr.shape[1]), int(decoded_bgr.shape[0]))
    if actual != expected:
        raise CameraResolutionError(actual_wh=actual, expected_wh=expected)

    try:
        decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error as error:
        raise CameraDecodeError("unable to convert decoded camera JPEG to RGB") from error
    if decoded_rgb.shape != (expected[1], expected[0], 3):
        raise CameraDecodeError("decoded RGB JPEG has an invalid shape")
    return np.ascontiguousarray(decoded_rgb)


def _validate_source_image(
    image: np.ndarray,
    *,
    camera_name: str,
    expected_wh: Tuple[int, int],
) -> None:
    if not isinstance(image, np.ndarray):
        raise TensorContractError("{} image must be a NumPy array".format(camera_name))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise TensorContractError(
            "{} image must be HWC uint8 with three RGB channels".format(camera_name)
        )
    actual_wh = (int(image.shape[1]), int(image.shape[0]))
    if actual_wh != expected_wh:
        raise CameraResolutionError(
            actual_wh=actual_wh,
            expected_wh=expected_wh,
            camera_name=camera_name,
        )


def stitch_camera_rgb(
    left: np.ndarray,
    front: np.ndarray,
    right: np.ndarray,
    config: CameraAdapterConfig,
) -> np.ndarray:
    """Resize, stitch, and outer-crop the three physical MORAI RGB views."""

    if not isinstance(config, CameraAdapterConfig):
        raise TensorContractError("config must be a CameraAdapterConfig")
    _validate_source_image(
        left, camera_name="left", expected_wh=config.left_source_wh
    )
    _validate_source_image(
        front, camera_name="front", expected_wh=config.front_source_wh
    )
    _validate_source_image(
        right, camera_name="right", expected_wh=config.right_source_wh
    )

    resized = (
        cv2.resize(left, config.left_resize_wh, interpolation=config.interpolation),
        cv2.resize(front, config.front_resize_wh, interpolation=config.interpolation),
        cv2.resize(right, config.right_resize_wh, interpolation=config.interpolation),
    )
    stitched = np.concatenate(resized, axis=1)
    expected_stitched_shape = (config.stitched_wh[1], config.stitched_wh[0], 3)
    if stitched.dtype != np.uint8 or stitched.shape != expected_stitched_shape:
        raise TensorContractError(
            "stitched camera strip must be uint8 [{}x{}x3]".format(
                config.stitched_wh[1], config.stitched_wh[0]
            )
        )

    output = np.ascontiguousarray(
        stitched[
            :,
            config.outer_crop_x_start : config.outer_crop_x_end_exclusive,
            :,
        ]
    )
    expected_output_shape = (config.output_wh[1], config.output_wh[0], 3)
    if output.dtype != np.uint8 or output.shape != expected_output_shape:
        raise TensorContractError("camera_rgb must be uint8 [256,1024,3]")
    return output


def camera_model_tensor(camera_rgb: np.ndarray) -> np.ndarray:
    """Convert cached RGB HWC uint8 into contiguous CHW float32 in ``[0, 1]``."""

    if (
        not isinstance(camera_rgb, np.ndarray)
        or camera_rgb.dtype != np.uint8
        or camera_rgb.shape != (256, 1024, 3)
    ):
        raise TensorContractError("camera_rgb must be uint8 [256,1024,3]")

    model_tensor = np.transpose(camera_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    model_tensor = np.ascontiguousarray(model_tensor)
    if model_tensor.dtype != np.float32 or model_tensor.shape != (3, 256, 1024):
        raise TensorContractError("camera_feature must be float32 [3,256,1024]")
    return model_tensor
