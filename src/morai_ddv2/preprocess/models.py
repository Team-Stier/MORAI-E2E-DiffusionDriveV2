from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from morai_ddv2.errors import MoraiDDV2Error


T = TypeVar("T")


def _require_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")


@dataclass(frozen=True)
class StampedRecord(Generic[T]):
    stamp_ns: int
    stable_record_index: int
    value: T

    def __post_init__(self) -> None:
        _require_integer("stamp_ns", self.stamp_ns)
        _require_integer("stable_record_index", self.stable_record_index)
        if self.stable_record_index < 0:
            raise ValueError("stable_record_index must be non-negative")


@dataclass(frozen=True)
class Bracket(Generic[T]):
    before: StampedRecord[T]
    after: StampedRecord[T]
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.before, StampedRecord):
            raise TypeError("before must be a StampedRecord")
        if not isinstance(self.after, StampedRecord):
            raise TypeError("after must be a StampedRecord")
        if not isinstance(self.weight, float):
            raise TypeError("weight must be a float")
        if not math.isfinite(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be finite and in the inclusive range [0, 1]")


class DuplicateRecordConflict(MoraiDDV2Error):
    """Raised when records at one timestamp contain conflicting used values."""

    def __init__(
        self,
        stamp_ns: int,
        first_record_index: int,
        conflicting_record_index: int,
    ) -> None:
        self.stamp_ns = stamp_ns
        self.first_record_index = first_record_index
        self.conflicting_record_index = conflicting_record_index
        super().__init__(
            "conflicting duplicate records at "
            f"stamp_ns={stamp_ns}: stable_record_index={first_record_index} "
            f"conflicts with stable_record_index={conflicting_record_index}"
        )
