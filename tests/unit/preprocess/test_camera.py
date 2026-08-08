from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import MISSING, FrozenInstanceError, fields
from pathlib import Path

import cv2
import numpy as np
import pytest

import morai_ddv2.preprocess.camera as camera_module
from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError, MoraiDDV2Error
from morai_ddv2.preprocess.camera import (
    CameraAdapterConfig,
    CameraDecodeError,
    CameraResolutionError,
    TensorContractError,
    camera_adapter_config_from_resolved_domain,
    camera_model_tensor,
    decode_jpeg_rgb,
    inspect_jpeg,
    stitch_camera_rgb,
)


@pytest.fixture
def resolved_domain_publication(repo_root: Path):
    return json.loads(
        (repo_root / "tests/fixtures/golden/resolved-domain.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def camera_config(resolved_domain_publication) -> CameraAdapterConfig:
    return camera_adapter_config_from_resolved_domain(resolved_domain_publication)


def _jpeg_bytes(shape, bgr) -> bytes:
    image = np.empty(shape, dtype=np.uint8)
    image[...] = bgr
    encoded_ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    )
    assert encoded_ok
    return encoded.tobytes()


def _coordinate_image(height: int, width: int, view_marker: int) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = view_marker
    image[..., 1] = np.arange(width, dtype=np.uint16) % 251
    image[..., 2] = (np.arange(height, dtype=np.uint16) % 241)[:, None]
    return image


def _source_images():
    return (
        np.full((480, 640, 3), 31, dtype=np.uint8),
        np.full((720, 1280, 3), 127, dtype=np.uint8),
        np.full((480, 640, 3), 223, dtype=np.uint8),
    )


def test_resolved_domain_factory_matches_independent_camera_layout_golden(
    repo_root: Path,
    resolved_domain_publication,
) -> None:
    layout = json.loads(
        (repo_root / "tests/fixtures/golden/camera-layout.json").read_text(
            encoding="utf-8"
        )
    )
    raw_before = deepcopy(resolved_domain_publication)
    config = camera_adapter_config_from_resolved_domain(resolved_domain_publication)
    loaded_document = deepcopy(resolved_domain_publication)
    loaded_sha256 = loaded_document.pop("resolved_domain_sha256")
    loaded_config = camera_adapter_config_from_resolved_domain(
        ResolvedDomain(document=loaded_document, sha256=loaded_sha256)
    )

    assert isinstance(config, CameraAdapterConfig)
    assert config == loaded_config
    assert resolved_domain_publication == raw_before
    assert layout == {
        "cache_tensor": {
            "dtype": "uint8",
            "layout": "HWC",
            "shape": [256, 1024, 3],
        },
        "interpolation": "cv2.INTER_AREA",
        "model_tensor": {
            "conversion": "transpose_HWC_to_CHW_then_divide_255",
            "dtype": "float32",
            "layout": "CHW",
            "shape": [3, 256, 1024],
        },
        "outer_crop_x": [56, 1080],
        "output_boundaries_x": [0, 285, 740, 1024],
        "output_wh": [1024, 256],
        "resize_wh": {
            "front": [455, 256],
            "left": [341, 256],
            "right": [341, 256],
        },
        "source_wh": {
            "front": [1280, 720],
            "left": [640, 480],
            "right": [640, 480],
        },
        "stitched_wh": [1137, 256],
        "view_order": ["left", "front", "right"],
    }
    assert config.left_source_wh == tuple(layout["source_wh"]["left"])
    assert config.front_source_wh == tuple(layout["source_wh"]["front"])
    assert config.right_source_wh == tuple(layout["source_wh"]["right"])
    assert config.left_resize_wh == tuple(layout["resize_wh"]["left"])
    assert config.front_resize_wh == tuple(layout["resize_wh"]["front"])
    assert config.right_resize_wh == tuple(layout["resize_wh"]["right"])
    assert config.stitched_wh == tuple(layout["stitched_wh"])
    assert (config.outer_crop_x_start, config.outer_crop_x_end_exclusive) == tuple(
        layout["outer_crop_x"]
    )
    assert config.output_wh == tuple(layout["output_wh"])
    assert config.output_boundaries_x == tuple(layout["output_boundaries_x"])
    assert config.view_order == tuple(layout["view_order"])
    assert config.interpolation == cv2.INTER_AREA

    with pytest.raises(FrozenInstanceError):
        config.output_wh = (1, 1)


def test_camera_config_has_no_default_or_module_level_fallback() -> None:
    with pytest.raises(TypeError):
        CameraAdapterConfig()
    assert all(
        field.default is MISSING and field.default_factory is MISSING
        for field in fields(CameraAdapterConfig)
    )
    assert not hasattr(camera_module, "DEFAULT_CAMERA_ADAPTER_CONFIG")


def test_factory_reads_physical_resolutions_by_role_not_list_position(
    resolved_domain_publication,
) -> None:
    reordered = deepcopy(resolved_domain_publication)
    expected_by_role = {
        "left": (642, 482),
        "front": (1282, 722),
        "right": (644, 484),
    }
    for camera in reordered["sensors"]["cameras"]:
        camera["resolution_wh"] = list(expected_by_role[camera["role"]])
    reordered["sensors"]["cameras"] = list(
        reversed(reordered["sensors"]["cameras"])
    )

    config = camera_adapter_config_from_resolved_domain(reordered)

    assert config.left_source_wh == expected_by_role["left"]
    assert config.front_source_wh == expected_by_role["front"]
    assert config.right_source_wh == expected_by_role["right"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("camera_processing", "decoded_layout"), "CHW"),
        (("camera_processing", "decoded_dtype"), "float32"),
        (("camera_processing", "decoded_color_order"), "RGB"),
        (("camera_processing", "output_color_order"), "BGR"),
        (("camera_processing", "exif_orientation_allowed"), True),
        (("camera_processing", "automatic_color_correction_allowed"), True),
        (("camera_processing", "interpolation"), "cv2.INTER_LINEAR"),
        (("camera_processing", "view_order"), ["front", "left", "right"]),
        (("camera_processing", "stitched_wh"), [1138, 256]),
        (("camera_processing", "outer_crop_x_end_exclusive"), 1079),
        (("camera_processing", "output_wh"), [1023, 256]),
        (("camera_processing", "output_boundaries_x"), [0, 286, 740, 1024]),
        (("camera_processing", "cache_tensor", "name"), "camera"),
        (("camera_processing", "cache_tensor", "layout"), "HWC"),
        (("camera_processing", "cache_tensor", "dtype"), "float32"),
        (
            ("camera_processing", "cache_tensor", "shape_per_sample"),
            [256, 1023, 3],
        ),
        (("camera_processing", "cache_tensor", "value_range"), [0, 1]),
        (("camera_processing", "model_tensor", "name"), "camera"),
        (("camera_processing", "model_tensor", "layout"), "NHWC"),
        (("camera_processing", "model_tensor", "dtype"), "float64"),
        (
            ("camera_processing", "model_tensor", "shape_per_sample"),
            [3, 256, 1023],
        ),
        (("camera_processing", "model_tensor", "value_range"), [0, 255]),
        (("camera_processing", "model_tensor", "conversion"), "other"),
        (("camera_processing", "model_tensor", "imagenet_normalization"), True),
    ],
)
def test_factory_rejects_camera_policy_drift(
    resolved_domain_publication, path, value
) -> None:
    document = deepcopy(resolved_domain_publication)
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ConfigurationError):
        camera_adapter_config_from_resolved_domain(document)


def test_factory_rejects_relationally_consistent_non_v1_camera_geometry(
    resolved_domain_publication,
) -> None:
    document = deepcopy(resolved_domain_publication)
    processing = document["camera_processing"]
    processing["resize_wh"] = {
        "left": [350, 256],
        "front": [440, 256],
        "right": [350, 256],
    }
    processing["stitched_wh"] = [1140, 256]
    processing["outer_crop_x_start"] = 58
    processing["outer_crop_x_end_exclusive"] = 1082
    processing["output_boundaries_x"] = [0, 292, 732, 1024]

    with pytest.raises(ConfigurationError):
        camera_adapter_config_from_resolved_domain(document)


@pytest.mark.parametrize(
    "tensor_name,value_range",
    [("cache_tensor", ["0", "255"]), ("model_tensor", ["0", "1"])],
)
def test_factory_rejects_numeric_string_tensor_ranges(
    resolved_domain_publication, tensor_name, value_range
) -> None:
    document = deepcopy(resolved_domain_publication)
    document["camera_processing"][tensor_name]["value_range"] = value_range

    with pytest.raises(ConfigurationError):
        camera_adapter_config_from_resolved_domain(document)


def test_factory_normalizes_giant_real_tensor_range_to_configuration_error(
    resolved_domain_publication,
) -> None:
    document = deepcopy(resolved_domain_publication)
    document["camera_processing"]["cache_tensor"]["value_range"] = [
        10**400,
        255,
    ]

    with pytest.raises(ConfigurationError):
        camera_adapter_config_from_resolved_domain(document)


@pytest.mark.parametrize(
    "path",
    [
        ("sensors",),
        ("sensors", "cameras"),
        ("sensors", "cameras", 0, "role"),
        ("sensors", "cameras", 0, "resolution_wh"),
        ("camera_processing",),
        ("camera_processing", "resize_wh"),
        ("camera_processing", "resize_wh", "left"),
        ("camera_processing", "cache_tensor"),
        ("camera_processing", "model_tensor"),
    ],
)
def test_factory_rejects_missing_required_domain_data(
    resolved_domain_publication, path
) -> None:
    document = deepcopy(resolved_domain_publication)
    target = document
    for key in path[:-1]:
        target = target[key]
    target.pop(path[-1])

    with pytest.raises(ConfigurationError):
        camera_adapter_config_from_resolved_domain(document)


@pytest.mark.parametrize("role_case", ["missing", "duplicate", "unknown"])
def test_factory_rejects_invalid_camera_roles(
    resolved_domain_publication, role_case: str
) -> None:
    document = deepcopy(resolved_domain_publication)
    cameras = document["sensors"]["cameras"]
    if role_case == "missing":
        cameras[:] = [camera for camera in cameras if camera["role"] != "left"]
    elif role_case == "duplicate":
        cameras[2]["role"] = "left"
    else:
        cameras[2]["role"] = "rear"

    with pytest.raises(ConfigurationError):
        camera_adapter_config_from_resolved_domain(document)


@pytest.mark.parametrize("resolved_domain", [None, [], "resolved"])
def test_factory_rejects_non_mapping_domain_input(resolved_domain) -> None:
    with pytest.raises(ConfigurationError):
        camera_adapter_config_from_resolved_domain(resolved_domain)


def test_coordinate_gradient_uses_area_resize_and_exact_crop_seams(
    monkeypatch, camera_config: CameraAdapterConfig
) -> None:
    left = _coordinate_image(480, 640, 31)
    front = _coordinate_image(720, 1280, 127)
    right = _coordinate_image(480, 640, 223)
    real_resize = cv2.resize
    expected = np.ascontiguousarray(
        np.concatenate(
            (
                real_resize(left, (341, 256), interpolation=cv2.INTER_AREA),
                real_resize(front, (455, 256), interpolation=cv2.INTER_AREA),
                real_resize(right, (341, 256), interpolation=cv2.INTER_AREA),
            ),
            axis=1,
        )[:, 56:1080, :]
    )
    resize_calls = []

    def recording_resize(image, dsize, interpolation):
        resize_calls.append((image.shape, dsize, interpolation))
        return real_resize(image, dsize, interpolation=interpolation)

    monkeypatch.setattr(camera_module.cv2, "resize", recording_resize)

    output = stitch_camera_rgb(left, front, right, camera_config)

    assert resize_calls == [
        ((480, 640, 3), (341, 256), cv2.INTER_AREA),
        ((720, 1280, 3), (455, 256), cv2.INTER_AREA),
        ((480, 640, 3), (341, 256), cv2.INTER_AREA),
    ]
    np.testing.assert_array_equal(output, expected)
    assert output.shape == (256, 1024, 3)
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous
    assert output[128, 284, 0] == 31
    assert output[128, 285, 0] == 127
    assert output[128, 739, 0] == 127
    assert output[128, 740, 0] == 223
    assert output[128, 0, 1] == expected[128, 0, 1]
    assert output[128, 1023, 1] == expected[128, 1023, 1]


@pytest.mark.parametrize(
    "shape,bgr,expected_wh,dominant_rgb_channel",
    [
        ((480, 640, 3), (0, 0, 255), (640, 480), 0),
        ((720, 1280, 3), (0, 255, 0), (1280, 720), 1),
        ((480, 640, 3), (255, 0, 0), (640, 480), 2),
    ],
)
def test_jpeg_decode_converts_bgr_to_contiguous_rgb(
    shape, bgr, expected_wh, dominant_rgb_channel
) -> None:
    payload = _jpeg_bytes(shape, bgr)

    assert inspect_jpeg(payload) == expected_wh
    decoded = decode_jpeg_rgb(payload, expected_wh)

    center_rgb = decoded[shape[0] // 2, shape[1] // 2]
    assert decoded.shape == shape
    assert decoded.dtype == np.uint8
    assert decoded.flags.c_contiguous
    assert int(np.argmax(center_rgb)) == dominant_rgb_channel
    assert int(center_rgb[dominant_rgb_channel]) >= 250


def test_jpeg_decode_explicitly_ignores_exif_orientation(monkeypatch) -> None:
    payload = _jpeg_bytes((480, 640, 3), (0, 0, 255))
    real_imdecode = cv2.imdecode
    observed_flags = []

    def recording_imdecode(encoded, flags):
        observed_flags.append(flags)
        return real_imdecode(encoded, flags)

    monkeypatch.setattr(camera_module.cv2, "imdecode", recording_imdecode)

    assert inspect_jpeg(payload) == (640, 480)
    assert len(observed_flags) == 1
    assert observed_flags[0] & cv2.IMREAD_IGNORE_ORIENTATION
    assert observed_flags[0] & cv2.IMREAD_COLOR


@pytest.mark.parametrize("payload", [b"", b"not-a-jpeg"])
def test_corrupt_jpeg_raises_decode_error(payload: bytes) -> None:
    with pytest.raises(CameraDecodeError):
        inspect_jpeg(payload)
    with pytest.raises(CameraDecodeError):
        decode_jpeg_rgb(payload, (640, 480))


@pytest.mark.parametrize(
    "shape,expected_wh",
    [
        ((720, 1279, 3), (1280, 720)),
        ((719, 1280, 3), (1280, 720)),
        ((480, 641, 3), (640, 480)),
        ((479, 640, 3), (640, 480)),
    ],
)
def test_decoded_wrong_size_is_not_reported_as_decode_failure(
    shape, expected_wh
) -> None:
    payload = _jpeg_bytes(shape, (0, 0, 255))

    with pytest.raises(CameraResolutionError) as caught:
        decode_jpeg_rgb(payload, expected_wh)

    assert caught.value.actual_wh == (shape[1], shape[0])
    assert caught.value.expected_wh == expected_wh
    assert not isinstance(caught.value, CameraDecodeError)


def test_camera_errors_have_distinct_types_for_planner_mapping() -> None:
    error_types = {CameraDecodeError, CameraResolutionError, TensorContractError}

    assert len(error_types) == 3
    assert all(issubclass(error_type, MoraiDDV2Error) for error_type in error_types)


def test_stitch_rejects_tensor_contract_and_physical_resolution_violations(
    camera_config: CameraAdapterConfig,
) -> None:
    left, front, right = _source_images()

    with pytest.raises(TensorContractError):
        stitch_camera_rgb(
            left.astype(np.float32),
            front,
            right,
            camera_config,
        )
    with pytest.raises(TensorContractError):
        stitch_camera_rgb(left[..., 0], front, right, camera_config)
    with pytest.raises(CameraResolutionError) as caught:
        stitch_camera_rgb(left[:, :-1, :], front, right, camera_config)

    assert caught.value.actual_wh == (639, 480)
    assert caught.value.expected_wh == (640, 480)


def test_model_tensor_is_exact_rgb_chw_float32_divide_255_and_contiguous() -> None:
    backing = np.zeros((256, 2048, 3), dtype=np.uint8)
    camera_rgb = backing[:, ::2, :]
    camera_rgb[..., 0] = 255
    camera_rgb[..., 1] = 127
    camera_rgb[..., 2] = 0
    assert camera_rgb.shape == (256, 1024, 3)
    assert not camera_rgb.flags.c_contiguous
    expected = np.transpose(camera_rgb, (2, 0, 1)).astype(np.float32) / 255.0

    feature = camera_model_tensor(camera_rgb)

    np.testing.assert_array_equal(feature, expected)
    assert feature.shape == (3, 256, 1024)
    assert feature.dtype == np.float32
    assert feature.flags.c_contiguous
    np.testing.assert_allclose(
        feature[:, 0, 0], [1.0, 127.0 / 255.0, 0.0], rtol=0.0, atol=1.0e-7
    )
    assert float(feature.min()) == 0.0
    assert float(feature.max()) == 1.0


@pytest.mark.parametrize(
    "camera_rgb",
    [
        np.zeros((256, 1024, 3), dtype=np.float32),
        np.zeros((256, 1024), dtype=np.uint8),
        np.zeros((256, 1024, 4), dtype=np.uint8),
        np.zeros((255, 1024, 3), dtype=np.uint8),
    ],
)
def test_model_tensor_rejects_wrong_cache_tensor_contract(camera_rgb) -> None:
    with pytest.raises(TensorContractError):
        camera_model_tensor(camera_rgb)
