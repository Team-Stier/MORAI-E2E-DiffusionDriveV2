from __future__ import annotations

from typing import TypeVar

from morai_ddv2.preprocess.models import StampedRecord


T = TypeVar("T")


def stamped_record(
    stamp_ns: int,
    stable_record_index: int,
    value: T,
) -> StampedRecord[T]:
    """Build a timestamp fixture without changing its provenance index."""

    return StampedRecord(
        stamp_ns=stamp_ns,
        stable_record_index=stable_record_index,
        value=value,
    )
