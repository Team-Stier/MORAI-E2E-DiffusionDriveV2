from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from morai_ddv2.preprocess.models import (
    Bracket,
    DuplicateRecordConflict,
    StampedRecord,
)
from morai_ddv2.preprocess.timestamps import (
    ceil_div,
    collapse_duplicate_stamps,
    interpolation_bracket,
    make_grid_ns,
    nearest_record,
    stable_sort,
)
from tests.factories import stamped_record


def test_stamped_record_is_immutable_and_keeps_signed_integer_stamp() -> None:
    record = stamped_record(-10, 2, "value")

    assert record == StampedRecord(-10, 2, "value")
    with pytest.raises(FrozenInstanceError):
        record.stamp_ns = 11


@pytest.mark.parametrize("stamp_ns", [1.0, True, "1"])
def test_stamped_record_rejects_non_integer_stamp(stamp_ns) -> None:
    with pytest.raises(TypeError, match="stamp_ns"):
        StampedRecord(stamp_ns, 0, "value")


@pytest.mark.parametrize("record_index", [-1, 1.0, True, "1"])
def test_stamped_record_rejects_malformed_stable_index(record_index) -> None:
    expected = ValueError if record_index == -1 else TypeError
    with pytest.raises(expected, match="stable_record_index"):
        StampedRecord(0, record_index, "value")


def test_ceil_div_uses_integer_arithmetic_for_signed_values() -> None:
    assert ceil_div(1_000_000_001, 500_000_000) == 3
    assert ceil_div(1_000_000_000, 500_000_000) == 2
    assert ceil_div(-1, 500_000_000) == 0


@pytest.mark.parametrize("value, divisor", [(1.0, 1), (1, 1.0), (True, 1), (1, True)])
def test_ceil_div_rejects_non_integer_arguments(value, divisor) -> None:
    with pytest.raises(TypeError):
        ceil_div(value, divisor)


@pytest.mark.parametrize("divisor", [0, -1])
def test_ceil_div_requires_positive_divisor(divisor: int) -> None:
    with pytest.raises(ValueError, match="divisor"):
        ceil_div(10, divisor)


def test_make_grid_ceil_aligns_origin_and_includes_last_boundary() -> None:
    assert make_grid_ns(
        first_front_ns=1_000_000_001,
        last_front_ns=2_000_000_000,
        period_ns=500_000_000,
    ) == (1_500_000_000, 2_000_000_000)
    assert make_grid_ns(1_000_000_000, 1_000_000_000, 500_000_000) == (
        1_000_000_000,
    )


def test_make_grid_can_be_empty_when_next_origin_is_after_last_front() -> None:
    assert make_grid_ns(501, 999, 500) == ()


@pytest.mark.parametrize(
    "first_front_ns, last_front_ns, period_ns, error",
    [
        (10, 9, 1, ValueError),
        (0, 1, 0, ValueError),
        (0, 1, -1, ValueError),
        (0.0, 1, 1, TypeError),
        (0, 1.0, 1, TypeError),
        (0, 1, 1.0, TypeError),
    ],
)
def test_make_grid_rejects_invalid_arguments(
    first_front_ns, last_front_ns, period_ns, error
) -> None:
    with pytest.raises(error):
        make_grid_ns(first_front_ns, last_front_ns, period_ns)


def test_stable_sort_orders_by_stamp_then_original_record_index() -> None:
    records = (
        stamped_record(30, 0, "last"),
        stamped_record(10, 8, "same-stamp-second"),
        stamped_record(10, 3, "same-stamp-first"),
    )

    assert stable_sort(records) == (records[2], records[1], records[0])
    assert [record.stable_record_index for record in records] == [0, 8, 3]


def test_stable_sort_rejects_non_record_members() -> None:
    with pytest.raises(TypeError, match="StampedRecord"):
        stable_sort((StampedRecord(1, 0, "ok"), "not-a-record"))


def test_nearest_tie_prefers_earlier_source_then_record_index() -> None:
    records = (
        StampedRecord(1_010, 4, "later"),
        StampedRecord(990, 8, "earlier-second"),
        StampedRecord(990, 3, "earlier-first"),
    )

    assert nearest_record(records, target_ns=1_000, tolerance_ns=10) == StampedRecord(
        990, 3, "earlier-first"
    )


def test_nearest_uses_inclusive_tolerance_and_one_nanosecond_miss() -> None:
    records = (StampedRecord(1_030, 0, "at-boundary"),)

    assert nearest_record(records, target_ns=1_000, tolerance_ns=30) == records[0]
    assert nearest_record(records, target_ns=1_000, tolerance_ns=29) is None


def test_nearest_handles_empty_and_zero_tolerance() -> None:
    assert nearest_record((), target_ns=10, tolerance_ns=0) is None
    exact = StampedRecord(10, 0, "exact")
    assert nearest_record((exact,), target_ns=10, tolerance_ns=0) == exact


@pytest.mark.parametrize(
    "target_ns, tolerance_ns, error",
    [(1.0, 0, TypeError), (1, 0.0, TypeError), (1, -1, ValueError)],
)
def test_nearest_rejects_invalid_arguments(target_ns, tolerance_ns, error) -> None:
    with pytest.raises(error):
        nearest_record((), target_ns=target_ns, tolerance_ns=tolerance_ns)


def test_equivalent_duplicate_stamps_collapse_to_earliest_original_index() -> None:
    records = (
        StampedRecord(20, 9, "other"),
        StampedRecord(10, 7, {"velocity": 1.0}),
        StampedRecord(10, 2, {"velocity": 1.0}),
    )

    collapsed = collapse_duplicate_stamps(records, equivalent=lambda left, right: left == right)

    assert collapsed == (records[2], records[0])
    assert collapsed[0].stable_record_index == 2


def test_conflicting_duplicate_stamps_raise_instead_of_becoming_a_drop() -> None:
    records = (
        StampedRecord(10, 2, {"velocity": 1.0}),
        StampedRecord(10, 3, {"velocity": 2.0}),
    )

    with pytest.raises(DuplicateRecordConflict, match="stamp_ns=10"):
        collapse_duplicate_stamps(records, equivalent=lambda left, right: left == right)


def test_duplicate_collapse_handles_empty_and_rejects_non_callable_equivalence() -> None:
    assert collapse_duplicate_stamps((), equivalent=lambda left, right: left == right) == ()
    with pytest.raises(TypeError, match="equivalent"):
        collapse_duplicate_stamps((), equivalent=None)


def test_interpolation_exact_match_precedes_bracketing_and_uses_earliest_index() -> None:
    records = (
        StampedRecord(200, 0, "after"),
        StampedRecord(100, 7, "exact-second"),
        StampedRecord(100, 2, "exact-first"),
        StampedRecord(0, 0, "before"),
    )

    bracket = interpolation_bracket(records, target_ns=100, max_gap_ns=0)

    expected = StampedRecord(100, 2, "exact-first")
    assert bracket == Bracket(before=expected, after=expected, weight=0.0)


def test_interpolation_accepts_exactly_100_ms_gap() -> None:
    before = StampedRecord(1_000_000_000, 9, "before")
    after = StampedRecord(1_100_000_000, 10, "after")

    bracket = interpolation_bracket(
        (after, before),
        target_ns=1_025_000_000,
        max_gap_ns=100_000_000,
    )

    assert bracket is not None
    assert bracket.before == before
    assert bracket.after == after
    assert bracket.weight == pytest.approx(0.25)


def test_interpolation_rejects_100_ms_plus_one_nanosecond_gap() -> None:
    records = (
        StampedRecord(0, 0, "before"),
        StampedRecord(100_000_001, 1, "after"),
    )

    assert (
        interpolation_bracket(records, target_ns=50_000_000, max_gap_ns=100_000_000)
        is None
    )


def test_interpolation_never_extrapolates_and_handles_empty_or_singleton() -> None:
    records = (
        StampedRecord(100, 0, "first"),
        StampedRecord(200, 1, "last"),
    )

    assert interpolation_bracket(records, target_ns=99, max_gap_ns=100) is None
    assert interpolation_bracket(records, target_ns=201, max_gap_ns=100) is None
    assert interpolation_bracket((), target_ns=100, max_gap_ns=100) is None
    assert interpolation_bracket(records[:1], target_ns=101, max_gap_ns=100) is None


@pytest.mark.parametrize(
    "target_ns, max_gap_ns, error",
    [(1.0, 0, TypeError), (1, 0.0, TypeError), (1, -1, ValueError)],
)
def test_interpolation_rejects_invalid_arguments(target_ns, max_gap_ns, error) -> None:
    with pytest.raises(error):
        interpolation_bracket((), target_ns=target_ns, max_gap_ns=max_gap_ns)
