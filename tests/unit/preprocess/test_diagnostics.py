from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from morai_ddv2.preprocess.diagnostics import (
    EgoDiagnosticValue,
    TfEgoDiagnostic,
    TfEgoPoseMismatch,
    compute_tf_ego_diagnostic,
    nearest_rank_percentile,
)
from morai_ddv2.preprocess.geometry import Pose
from morai_ddv2.preprocess.models import StampedRecord


IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)


def _yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    half_angle = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))


def _ego_record(
    stamp_ns: int,
    index: int,
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    heading_deg: float = 0.0,
) -> StampedRecord[EgoDiagnosticValue]:
    return StampedRecord(
        stamp_ns,
        index,
        EgoDiagnosticValue(position_xy=(x_m, y_m), heading_deg=heading_deg),
    )


def _tf_record(
    stamp_ns: int,
    index: int,
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    heading_deg: float = 0.0,
) -> StampedRecord[Pose]:
    return StampedRecord(
        stamp_ns,
        index,
        Pose(
            translation_xyz=(x_m, y_m, 0.0),
            quaternion_xyzw=_yaw_quaternion(heading_deg),
        ),
    )


def _exact_records(
    count: int,
    *,
    position_errors_m=(),
    heading_errors_deg=(),
):
    position_errors = tuple(position_errors_m)
    heading_errors = tuple(heading_errors_deg)
    ego = []
    tf = []
    for index in range(count):
        position_error = position_errors[index] if index < len(position_errors) else 0.0
        heading_error = heading_errors[index] if index < len(heading_errors) else 0.0
        stamp_ns = index * 1_000
        ego.append(
            _ego_record(
                stamp_ns,
                index,
                x_m=position_error,
                heading_deg=heading_error,
            )
        )
        tf.append(_tf_record(stamp_ns, index))
    return tuple(ego), tuple(tf)


def test_diagnostic_value_and_result_are_immutable() -> None:
    value = EgoDiagnosticValue((1.0, 2.0), 3.0)
    result = TfEgoDiagnostic(10, 0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(FrozenInstanceError):
        value.heading_deg = 4.0
    with pytest.raises(FrozenInstanceError):
        result.comparable_count = 11


@pytest.mark.parametrize(
    "count, expected",
    [
        (9, 9.0),
        (10, 10.0),
        (100, 99.0),
    ],
)
def test_nearest_rank_percentile_uses_ceil_rank_minus_one(
    count: int, expected: float
) -> None:
    descending = tuple(float(value) for value in range(count, 0, -1))

    assert nearest_rank_percentile(descending, 0.99) == expected


@pytest.mark.parametrize(
    "values, percentile",
    [
        ((), 0.99),
        ((0.0, math.nan), 0.99),
        ((0.0, math.inf), 0.99),
        ((0.0,), 0.0),
        ((0.0,), -0.01),
        ((0.0,), 1.01),
        ((0.0,), math.nan),
    ],
)
def test_nearest_rank_percentile_rejects_empty_nonfinite_and_invalid_percentile(
    values, percentile
) -> None:
    with pytest.raises(ValueError):
        nearest_rank_percentile(values, percentile)


@pytest.mark.parametrize("count", [0, 9])
def test_diagnostic_rejects_fewer_than_ten_comparable_pairs(count: int) -> None:
    ego, tf = _exact_records(count)

    with pytest.raises(TfEgoPoseMismatch) as caught:
        compute_tf_ego_diagnostic(ego, tf, max_bracket_ns=0)

    assert caught.value.rejection_code == "TF_EGO_POSE_MISMATCH"


def test_diagnostic_excludes_unbracketable_edges_and_accepts_ten_pairs() -> None:
    ego = tuple(_ego_record(stamp, index) for index, stamp in enumerate(range(12)))
    tf = tuple(_tf_record(stamp, index) for index, stamp in enumerate(range(1, 11)))

    actual = compute_tf_ego_diagnostic(ego, tuple(reversed(tf)), max_bracket_ns=0)

    assert actual == TfEgoDiagnostic(
        comparable_count=10,
        excluded_count=2,
        position_p99_m=0.0,
        position_max_m=0.0,
        heading_p99_deg=0.0,
        heading_max_deg=0.0,
    )


def test_diagnostic_uses_bounded_linear_and_slerp_interpolation() -> None:
    ego = []
    tf = []
    for index in range(10):
        start_ns = index * 1_000
        ego.append(
            _ego_record(
                start_ns + 25,
                index,
                x_m=1.0,
                y_m=-0.5,
                heading_deg=22.5,
            )
        )
        tf.append(
            _tf_record(start_ns, index * 2, x_m=0.0, y_m=0.0, heading_deg=0.0)
        )
        tf.append(
            _tf_record(
                start_ns + 100,
                index * 2 + 1,
                x_m=4.0,
                y_m=-2.0,
                heading_deg=90.0,
            )
        )

    actual = compute_tf_ego_diagnostic(
        tuple(reversed(ego)), tuple(reversed(tf)), max_bracket_ns=100
    )

    assert actual.comparable_count == 10
    assert actual.excluded_count == 0
    assert actual.position_p99_m == pytest.approx(0.0, abs=1.0e-15)
    assert actual.heading_p99_deg == pytest.approx(0.0, abs=1.0e-12)


def test_diagnostic_excludes_brackets_one_nanosecond_over_limit() -> None:
    ego = tuple(_ego_record(index * 1_000 + 50, index) for index in range(10))
    tf = []
    for index in range(10):
        start_ns = index * 1_000
        tf.extend(
            (
                _tf_record(start_ns, index * 2),
                _tf_record(start_ns + 101, index * 2 + 1),
            )
        )

    with pytest.raises(TfEgoPoseMismatch, match="comparable"):
        compute_tf_ego_diagnostic(ego, tuple(tf), max_bracket_ns=100)


def test_n100_isolated_position_max_is_recorded_but_does_not_reject() -> None:
    ego, tf = _exact_records(100, position_errors_m=(0.0,) * 99 + (1.0,))

    actual = compute_tf_ego_diagnostic(ego, tf, max_bracket_ns=0)

    assert actual.position_p99_m == 0.0
    assert actual.position_max_m == 1.0


def test_n100_two_position_outliers_reject_at_p99() -> None:
    ego, tf = _exact_records(100, position_errors_m=(0.0,) * 98 + (1.0, 2.0))

    with pytest.raises(TfEgoPoseMismatch, match="position"):
        compute_tf_ego_diagnostic(ego, tf, max_bracket_ns=0)


def test_n100_isolated_heading_max_is_recorded_but_does_not_reject() -> None:
    ego, tf = _exact_records(100, heading_errors_deg=(0.0,) * 99 + (10.0,))

    actual = compute_tf_ego_diagnostic(ego, tf, max_bracket_ns=0)

    assert actual.heading_p99_deg == 0.0
    assert actual.heading_max_deg == pytest.approx(10.0)


def test_n100_two_heading_outliers_reject_at_p99() -> None:
    ego, tf = _exact_records(100, heading_errors_deg=(0.0,) * 98 + (1.0, 2.0))

    with pytest.raises(TfEgoPoseMismatch, match="heading"):
        compute_tf_ego_diagnostic(ego, tf, max_bracket_ns=0)


def test_n10_position_tolerance_is_inclusive_and_epsilon_rejects() -> None:
    at_limit_ego, at_limit_tf = _exact_records(10, position_errors_m=(0.05,) * 10)
    over_ego, over_tf = _exact_records(10, position_errors_m=(0.050001,) * 10)

    assert compute_tf_ego_diagnostic(
        at_limit_ego, at_limit_tf, max_bracket_ns=0
    ).position_p99_m == pytest.approx(0.05)
    with pytest.raises(TfEgoPoseMismatch, match="position"):
        compute_tf_ego_diagnostic(over_ego, over_tf, max_bracket_ns=0)


def test_n10_heading_tolerance_is_inclusive_and_epsilon_rejects() -> None:
    at_limit_ego, at_limit_tf = _exact_records(10, heading_errors_deg=(0.5,) * 10)
    over_ego, over_tf = _exact_records(10, heading_errors_deg=(0.500001,) * 10)

    assert compute_tf_ego_diagnostic(
        at_limit_ego, at_limit_tf, max_bracket_ns=0
    ).heading_p99_deg == pytest.approx(0.5)
    with pytest.raises(TfEgoPoseMismatch, match="heading"):
        compute_tf_ego_diagnostic(over_ego, over_tf, max_bracket_ns=0)


def test_heading_error_is_circular_across_negative_pi_boundary() -> None:
    ego = tuple(
        _ego_record(index, index, heading_deg=-179.9) for index in range(10)
    )
    tf = tuple(
        _tf_record(index, index, heading_deg=179.9) for index in range(10)
    )

    actual = compute_tf_ego_diagnostic(ego, tf, max_bracket_ns=0)

    assert actual.heading_p99_deg == pytest.approx(0.2, abs=1.0e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_bracket_ns": -1},
        {"max_bracket_ns": 0.5},
        {"max_bracket_ns": 0, "minimum_comparable_pairs": 0},
        {"max_bracket_ns": 0, "minimum_comparable_pairs": True},
        {"max_bracket_ns": 0, "position_p99_tolerance_m": -1.0},
        {"max_bracket_ns": 0, "position_p99_tolerance_m": math.nan},
        {"max_bracket_ns": 0, "heading_p99_tolerance_deg": -1.0},
        {"max_bracket_ns": 0, "heading_p99_tolerance_deg": math.inf},
    ],
)
def test_diagnostic_rejects_invalid_policy_arguments(kwargs) -> None:
    ego, tf = _exact_records(10)

    with pytest.raises((TypeError, ValueError)):
        compute_tf_ego_diagnostic(ego, tf, **kwargs)


@pytest.mark.parametrize(
    "ego_value, tf_pose",
    [
        (
            EgoDiagnosticValue((math.nan, 0.0), 0.0),
            Pose((0.0, 0.0, 0.0), IDENTITY_QUATERNION),
        ),
        (
            EgoDiagnosticValue((0.0, 0.0), math.inf),
            Pose((0.0, 0.0, 0.0), IDENTITY_QUATERNION),
        ),
        (
            EgoDiagnosticValue((0.0, 0.0), 0.0),
            Pose((math.inf, 0.0, 0.0), IDENTITY_QUATERNION),
        ),
    ],
)
def test_diagnostic_rejects_nonfinite_ego_or_tf_values(ego_value, tf_pose) -> None:
    ego = tuple(StampedRecord(index, index, ego_value) for index in range(10))
    tf = tuple(StampedRecord(index, index, tf_pose) for index in range(10))

    with pytest.raises(ValueError):
        compute_tf_ego_diagnostic(ego, tf, max_bracket_ns=0)
