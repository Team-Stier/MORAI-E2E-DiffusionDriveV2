from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Sequence
from typing import Tuple, TypeVar

from .models import Bracket, DuplicateRecordConflict, StampedRecord


T = TypeVar("T")


def _require_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")


def _validated_records(
    records: Sequence[StampedRecord[T]],
) -> Tuple[StampedRecord[T], ...]:
    if not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of StampedRecord values")
    result = tuple(records)
    if any(not isinstance(record, StampedRecord) for record in result):
        raise TypeError("records must contain only StampedRecord values")
    return result


def ceil_div(value: int, divisor: int) -> int:
    """Return mathematical ceil(value / divisor) using integer arithmetic only."""

    _require_integer("value", value)
    _require_integer("divisor", divisor)
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return -(-value // divisor)


def make_grid_ns(
    first_front_ns: int,
    last_front_ns: int,
    period_ns: int,
) -> Tuple[int, ...]:
    """Build the inclusive, period-aligned grid within the front stamp range."""

    _require_integer("first_front_ns", first_front_ns)
    _require_integer("last_front_ns", last_front_ns)
    _require_integer("period_ns", period_ns)
    if period_ns <= 0:
        raise ValueError("period_ns must be positive")
    if last_front_ns < first_front_ns:
        raise ValueError("last_front_ns must be greater than or equal to first_front_ns")

    origin_ns = ceil_div(first_front_ns, period_ns) * period_ns
    return tuple(range(origin_ns, last_front_ns + 1, period_ns))


def stable_sort(
    records: Sequence[StampedRecord[T]],
) -> Tuple[StampedRecord[T], ...]:
    """Sort records without changing their original topic-local indices."""

    validated = _validated_records(records)
    return tuple(
        sorted(
            validated,
            key=lambda record: (record.stamp_ns, record.stable_record_index),
        )
    )


def nearest_record(
    records: Sequence[StampedRecord[T]],
    target_ns: int,
    tolerance_ns: int,
) -> StampedRecord[T] | None:
    """Select the deterministic nearest record inside an inclusive tolerance."""

    _require_integer("target_ns", target_ns)
    _require_integer("tolerance_ns", tolerance_ns)
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be non-negative")

    ordered = stable_sort(records)
    if not ordered:
        return None
    selected = min(
        ordered,
        key=lambda record: (
            abs(record.stamp_ns - target_ns),
            record.stamp_ns,
            record.stable_record_index,
        ),
    )
    if abs(selected.stamp_ns - target_ns) <= tolerance_ns:
        return selected
    return None


def collapse_duplicate_stamps(
    records: Sequence[StampedRecord[T]],
    equivalent: Callable[[T, T], bool],
) -> Tuple[StampedRecord[T], ...]:
    """Collapse equivalent stamp duplicates or raise on conflicting values."""

    if not callable(equivalent):
        raise TypeError("equivalent must be callable")
    ordered = stable_sort(records)
    collapsed = []

    for record in ordered:
        if not collapsed or collapsed[-1].stamp_ns != record.stamp_ns:
            collapsed.append(record)
            continue

        first = collapsed[-1]
        if not equivalent(first.value, record.value):
            raise DuplicateRecordConflict(
                stamp_ns=record.stamp_ns,
                first_record_index=first.stable_record_index,
                conflicting_record_index=record.stable_record_index,
            )

    return tuple(collapsed)


def interpolation_bracket(
    records: Sequence[StampedRecord[T]],
    target_ns: int,
    max_gap_ns: int,
) -> Bracket[T] | None:
    """Return a bounded bracket without extrapolating across missing data."""

    _require_integer("target_ns", target_ns)
    _require_integer("max_gap_ns", max_gap_ns)
    if max_gap_ns < 0:
        raise ValueError("max_gap_ns must be non-negative")

    ordered = stable_sort(records)
    if not ordered:
        return None

    stamps = tuple(record.stamp_ns for record in ordered)
    insertion_index = bisect_left(stamps, target_ns)

    if (
        insertion_index < len(ordered)
        and ordered[insertion_index].stamp_ns == target_ns
    ):
        exact = ordered[insertion_index]
        return Bracket(before=exact, after=exact, weight=0.0)

    if insertion_index == 0 or insertion_index == len(ordered):
        return None

    before = ordered[insertion_index - 1]
    after = ordered[insertion_index]
    gap_ns = after.stamp_ns - before.stamp_ns
    if gap_ns > max_gap_ns:
        return None

    weight = (target_ns - before.stamp_ns) / gap_ns
    return Bracket(before=before, after=after, weight=float(weight))
