from __future__ import annotations

import copy
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError, MoraiDDV2Error
from morai_ddv2.preprocess.geometry import Pose
from morai_ddv2.preprocess.lidar import (
    EmptyLidarError,
    LidarConfig,
    LidarPointError,
    LidarTensorContractError,
    PointCloudLayout,
    PointCloudLayoutError,
    PointFieldLayout,
    build_lidar_count,
    lidar_config_from_resolved_domain,
    lidar_model_tensor,
    validate_pointcloud_layout,
)


GOLDEN_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "golden"


def _golden_points():
    return json.loads((GOLDEN_ROOT / "lidar-points.json").read_text())


def _raw_resolved_domain():
    return json.loads((GOLDEN_ROOT / "resolved-domain.json").read_text())


def _pose(raw) -> Pose:
    return Pose(
        translation_xyz=tuple(raw["translation_xyz"]),
        quaternion_xyzw=tuple(raw["quaternion_xyzw"]),
    )


def _config() -> LidarConfig:
    return lidar_config_from_resolved_domain(_raw_resolved_domain())


def _assert_sparse_counts(actual: np.ndarray, expected_rows) -> None:
    expected = np.zeros((256, 256), dtype=np.uint8)
    for x_bin, y_bin, count in expected_rows:
        expected[x_bin, y_bin] = count
    np.testing.assert_array_equal(actual, expected)


def _real_layout() -> PointCloudLayout:
    return PointCloudLayout(
        fields=(
            PointFieldLayout("x", 0, 7, 1),
            PointFieldLayout("y", 4, 7, 1),
            PointFieldLayout("z", 8, 7, 1),
            PointFieldLayout("intensity", 12, 7, 1),
        ),
        point_step=16,
        is_bigendian=False,
    )


def test_lidar_exceptions_are_distinct_machine_mappable_types() -> None:
    errors = (
        PointCloudLayoutError,
        LidarPointError,
        EmptyLidarError,
        LidarTensorContractError,
    )

    assert all(issubclass(error, MoraiDDV2Error) for error in errors)
    assert len(set(errors)) == 4


def test_lidar_value_models_are_frozen() -> None:
    field = PointFieldLayout("x", 0, 7, 1)
    layout = _real_layout()
    config = _config()

    with pytest.raises(FrozenInstanceError):
        field.offset = 4
    with pytest.raises(FrozenInstanceError):
        layout.point_step = 32
    with pytest.raises(FrozenInstanceError):
        config.count_clip = 4


def test_factory_accepts_raw_canonical_mapping_and_loaded_immutable_document() -> None:
    raw = _raw_resolved_domain()
    publication = dict(raw)
    resolved_sha256 = publication.pop("resolved_domain_sha256")
    loaded = ResolvedDomain(document=publication, sha256=resolved_sha256)

    from_raw = lidar_config_from_resolved_domain(raw)
    from_loaded = lidar_config_from_resolved_domain(loaded)

    assert from_raw == from_loaded
    assert from_raw.x_range_m == (-32.0, 32.0)
    assert from_raw.y_range_m == (-32.0, 32.0)
    assert from_raw.bins_xy == (256, 256)
    assert from_raw.count_clip == 5


def _set_path(mapping, path, value) -> None:
    cursor = mapping
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


@pytest.mark.parametrize(
    "path, value",
    [
        (("lidar_processing", "ordered_steps"), ["BUILD_XY_HISTOGRAM"]),
        (("lidar_processing", "zero_padding_point_xyz"), [0.0, 0.0, 0.1]),
        (
            ("lidar_processing", "self_mask_base_link_m", "boundaries_inclusive"),
            False,
        ),
        (("lidar_processing", "self_mask_base_link_m", "x"), [2.0, 0.0]),
        (("lidar_processing", "height_filter_m", "min_exclusive"), 100.0),
        (("lidar_processing", "x_range_m"), [-31.0, 32.0]),
        (("lidar_processing", "pixels_per_meter"), 2),
        (("lidar_processing", "bin_size_m"), 0.5),
        (("lidar_processing", "bins_xy"), [255, 256]),
        (("lidar_processing", "histogram_axis_order"), ["y_bin", "x_bin"]),
        (("lidar_processing", "count_clip"), 4),
        (("lidar_processing", "normalization"), "divide_by_4"),
        (("lidar_processing", "rotate_or_flip"), True),
        (("lidar_processing", "ground_plane_channel"), True),
        (("lidar_processing", "cache_tensor", "dtype"), "float32"),
        (("lidar_processing", "model_tensor", "shape_per_sample"), [256, 256]),
    ],
)
def test_factory_rejects_policy_that_is_not_the_resolved_v1_contract(
    path, value
) -> None:
    raw = copy.deepcopy(_raw_resolved_domain())
    _set_path(raw, path, value)

    with pytest.raises(ConfigurationError):
        lidar_config_from_resolved_domain(raw)


def test_factory_rejects_missing_or_malformed_resolved_mapping() -> None:
    raw = _raw_resolved_domain()
    del raw["lidar_processing"]["self_mask_base_link_m"]["z"]

    with pytest.raises(ConfigurationError):
        lidar_config_from_resolved_domain(raw)
    with pytest.raises(ConfigurationError):
        lidar_config_from_resolved_domain([])


def test_validate_pointcloud_layout_accepts_observed_real_layout_and_field_order() -> None:
    validate_pointcloud_layout(_real_layout())
    shuffled = replace(_real_layout(), fields=tuple(reversed(_real_layout().fields)))
    validate_pointcloud_layout(shuffled)


@pytest.mark.parametrize("missing_name", ["x", "y", "z", "intensity"])
def test_validate_pointcloud_layout_rejects_each_missing_required_field(
    missing_name: str,
) -> None:
    layout = _real_layout()
    missing = replace(
        layout,
        fields=tuple(field for field in layout.fields if field.name != missing_name),
    )

    with pytest.raises(PointCloudLayoutError):
        validate_pointcloud_layout(missing)


def test_validate_pointcloud_layout_rejects_duplicate_required_field() -> None:
    layout = _real_layout()
    duplicate = replace(
        layout,
        fields=layout.fields + (PointFieldLayout("x", 16, 7, 1),),
        point_step=20,
    )

    with pytest.raises(PointCloudLayoutError):
        validate_pointcloud_layout(duplicate)


@pytest.mark.parametrize(
    "field",
    [
        PointFieldLayout("x", 0, 8, 1),
        PointFieldLayout("x", 0, "float32", 1),
        PointFieldLayout("x", 0, 7, 2),
        PointFieldLayout("x", 0, 7, 0),
        PointFieldLayout("x", -1, 7, 1),
        PointFieldLayout("x", True, 7, 1),
    ],
)
def test_validate_pointcloud_layout_rejects_wrong_type_count_or_offset(field) -> None:
    layout = _real_layout()
    malformed = replace(layout, fields=(field,) + layout.fields[1:])

    with pytest.raises(PointCloudLayoutError):
        validate_pointcloud_layout(malformed)


def test_validate_pointcloud_layout_rejects_overlap_and_extent_beyond_point_step() -> None:
    layout = _real_layout()
    overlapping = replace(
        layout,
        fields=(layout.fields[0], PointFieldLayout("y", 2, 7, 1)) + layout.fields[2:],
    )
    beyond = replace(
        layout,
        fields=layout.fields[:-1] + (PointFieldLayout("intensity", 13, 7, 1),),
    )

    with pytest.raises(PointCloudLayoutError):
        validate_pointcloud_layout(overlapping)
    with pytest.raises(PointCloudLayoutError):
        validate_pointcloud_layout(beyond)


@pytest.mark.parametrize("point_step", [0, -1, 16.0, True])
def test_validate_pointcloud_layout_rejects_invalid_point_step(point_step) -> None:
    with pytest.raises(PointCloudLayoutError):
        validate_pointcloud_layout(replace(_real_layout(), point_step=point_step))


def test_validate_pointcloud_layout_rejects_big_endian_and_wrong_layout_type() -> None:
    with pytest.raises(PointCloudLayoutError):
        validate_pointcloud_layout(replace(_real_layout(), is_bigendian=True))
    with pytest.raises(PointCloudLayoutError):
        validate_pointcloud_layout("not-a-layout")


def test_build_applies_canonical_identity_rotation_and_translation() -> None:
    golden = _golden_points()
    case = golden["canonical_translation"]
    points = np.asarray([case["raw_xyzi"]], dtype=np.float32)
    original = points.copy()

    actual = build_lidar_count(points, _pose(golden["canonical_pose"]), _config())

    x_bin, y_bin = case["expected_bin_xy"]
    _assert_sparse_counts(actual, [(x_bin, y_bin, 1)])
    np.testing.assert_array_equal(points, original)
    assert actual.dtype == np.uint8
    assert actual.shape == (256, 256)
    assert actual.flags.c_contiguous


def test_build_removes_zero_padding_before_transform() -> None:
    golden = _golden_points()
    zero = np.asarray([golden["canonical_zero_padding"]], dtype=np.float64)

    with pytest.raises(EmptyLidarError):
        build_lidar_count(
            zero, _pose(golden["zero_before_transform_pose"]), _config()
        )


def test_build_uses_full_3d_self_mask_and_preserves_road_below_roof() -> None:
    golden = _golden_points()
    face_points = np.asarray(
        golden["inclusive_self_mask_faces_base_xyzi"], dtype=np.float64
    )
    road = golden["canonical_road_below_mask"]

    with pytest.raises(EmptyLidarError):
        build_lidar_count(face_points, _pose(golden["identity_pose"]), _config())

    actual = build_lidar_count(
        np.asarray([road["raw_xyzi"]], dtype=np.float64),
        _pose(golden["canonical_pose"]),
        _config(),
    )
    _assert_sparse_counts(actual, [road["expected_bin_xy"] + [1]])


def test_build_self_mask_boundaries_are_inclusive_but_one_ulp_outside_survives() -> None:
    golden = _golden_points()
    points = np.asarray(
        golden["just_outside_self_mask_faces_base_xyzi"], dtype=np.float64
    )

    actual = build_lidar_count(points, _pose(golden["identity_pose"]), _config())

    _assert_sparse_counts(actual, golden["just_outside_self_mask_expected_bins"])


def test_build_applies_height_after_transform_and_uses_exclusive_boundaries() -> None:
    golden = _golden_points()
    transformed = np.asarray(
        [case["raw_xyzi"] for case in golden["canonical_height_crossing"]],
        dtype=np.float64,
    )
    expected = golden["canonical_height_crossing"][0]["expected_bin_xy"] + [1]

    actual_transformed = build_lidar_count(
        transformed, _pose(golden["canonical_pose"]), _config()
    )
    exact_boundaries = build_lidar_count(
        np.asarray(golden["exclusive_height_base_xyzi"], dtype=np.float64),
        _pose(golden["identity_pose"]),
        _config(),
    )

    _assert_sparse_counts(actual_transformed, [expected])
    _assert_sparse_counts(
        exact_boundaries, golden["exclusive_height_expected_bins"]
    )


def test_build_matches_histogramdd_edge_and_xy_axis_semantics() -> None:
    golden = _golden_points()

    actual = build_lidar_count(
        np.asarray(golden["histogram_edges_base_xyzi"], dtype=np.float64),
        _pose(golden["identity_pose"]),
        _config(),
    )

    _assert_sparse_counts(actual, golden["histogram_edges_expected_bins"])
    assert actual[0, 255] == 1
    assert actual[255, 0] == 0


@pytest.mark.parametrize("repeat_count", [6, 256])
def test_build_clips_in_wide_dtype_before_uint8_cast(repeat_count: int) -> None:
    golden = _golden_points()
    point = np.asarray(golden["clip_point_base_xyzi"], dtype=np.float64)
    points = np.repeat(point[None, :], repeat_count, axis=0)

    actual = build_lidar_count(points, _pose(golden["identity_pose"]), _config())

    x_bin, y_bin = golden["clip_expected_bin_xy"]
    _assert_sparse_counts(actual, [(x_bin, y_bin, 5)])


@pytest.mark.parametrize("coordinate", [0, 1, 2])
@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_build_rejects_nonfinite_xyz_before_filtering(
    coordinate: int, nonfinite: float
) -> None:
    points = np.asarray([[3.0, 3.0, 2.0, 1.0]], dtype=np.float64)
    points[0, coordinate] = nonfinite

    with pytest.raises(LidarPointError):
        build_lidar_count(points, Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), _config())


def test_build_ignores_nonfinite_intensity() -> None:
    points = np.asarray([[3.0, 3.0, 2.0, math.nan]], dtype=np.float64)

    actual = build_lidar_count(
        points, Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), _config()
    )

    _assert_sparse_counts(actual, [(140, 140, 1)])


@pytest.mark.parametrize(
    "points",
    [
        [],
        np.zeros(4),
        np.zeros((1, 3)),
        np.zeros((1, 5)),
        np.zeros((1, 4), dtype=np.bool_),
        np.zeros((1, 4), dtype=np.complex128),
        np.asarray([["1", "2", "3", "4"]]),
        np.asarray([[1, 2, 3, 4]], dtype=object),
    ],
)
def test_build_rejects_non_ndarray_wrong_shape_or_non_real_dtype(points) -> None:
    with pytest.raises(LidarPointError):
        build_lidar_count(
            points, Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), _config()
        )


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 4), dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        np.asarray([[33.0, 0.0, 2.0, 1.0]], dtype=np.float32),
    ],
)
def test_build_maps_empty_padding_filtered_and_outside_clouds_to_empty(points) -> None:
    with pytest.raises(EmptyLidarError):
        build_lidar_count(
            points, Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), _config()
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"count_clip": 4},
        {"count_clip": 5.0},
        {"pixels_per_meter": 4.0},
        {"bins_xy": (256.0, 256.0)},
        {"cache_tensor_shape": (256.0, 256.0)},
    ],
)
def test_build_revalidates_directly_constructed_config(changes) -> None:
    invalid = replace(_config(), **changes)

    with pytest.raises(ConfigurationError):
        build_lidar_count(
            np.asarray([[3.0, 3.0, 2.0, 1.0]]),
            Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            invalid,
        )


def test_lidar_model_tensor_maps_cache_counts_exactly() -> None:
    lidar_count = np.zeros((256, 256), dtype=np.uint8)
    lidar_count[0, 0] = 1
    lidar_count[255, 255] = 5
    original = lidar_count.copy()

    actual = lidar_model_tensor(lidar_count)

    assert actual.shape == (1, 256, 256)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert actual[0, 1, 1] == 0.0
    assert actual[0, 0, 0] == pytest.approx(0.2)
    assert actual[0, 255, 255] == 1.0
    np.testing.assert_array_equal(lidar_count, original)


@pytest.mark.parametrize(
    "lidar_count",
    [
        [[0]],
        np.zeros((256,), dtype=np.uint8),
        np.zeros((1, 256, 256), dtype=np.uint8),
        np.zeros((256, 256), dtype=np.float32),
        np.full((256, 256), 6, dtype=np.uint8),
        np.zeros((256, 512), dtype=np.uint8)[:, ::2],
    ],
)
def test_lidar_model_tensor_rejects_wrong_type_shape_dtype_range_or_contiguity(
    lidar_count,
) -> None:
    with pytest.raises(LidarTensorContractError):
        lidar_model_tensor(lidar_count)
