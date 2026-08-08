from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from morai_ddv2.errors import MoraiDDV2Error

from .geometry import Pose, _quaternion_yaw, _wrap_to_pi, interpolate_pose
from .models import StampedRecord
from .timestamps import interpolation_bracket


@dataclass(frozen=True)
class EgoDiagnosticValue:
    position_xy: Tuple[float, float]
    heading_deg: float


@dataclass(frozen=True)
class TfEgoDiagnostic:
    comparable_count: int
    excluded_count: int
    position_p99_m: float
    position_max_m: float
    heading_p99_deg: float
    heading_max_deg: float


class TfEgoPoseMismatch(MoraiDDV2Error):
    rejection_code = "TF_EGO_POSE_MISMATCH"

    def __init__(
        self, message: str, diagnostic: Optional[TfEgoDiagnostic] = None
    ) -> None:
        self.diagnostic = diagnostic
        super().__init__(message)


def _finite_number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("{} must be a finite number".format(name)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    if nonnegative and result < 0.0:
        raise ValueError("{} must be non-negative".format(name))
    return result


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    """Compute the finite nearest-rank percentile using ``ceil(p*N)-1``."""

    probability = _finite_number(percentile, "percentile")
    if not 0.0 < probability <= 1.0:
        raise ValueError("percentile must be in the interval (0, 1]")
    try:
        samples = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("values must contain numeric values") from error
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("values must be a nonempty one-dimensional sequence")
    if not np.isfinite(samples).all():
        raise ValueError("values must contain only finite values")
    ordered = np.sort(samples)
    index = int(math.ceil(probability * int(ordered.size)) - 1)
    return float(ordered[index])


def _validated_records(
    records: Sequence[StampedRecord], value_type: type, name: str
) -> Tuple[StampedRecord, ...]:
    if not isinstance(records, Sequence):
        raise TypeError("{} must be a sequence of StampedRecord values".format(name))
    result = tuple(records)
    if any(not isinstance(record, StampedRecord) for record in result):
        raise TypeError("{} must contain only StampedRecord values".format(name))
    if any(not isinstance(record.value, value_type) for record in result):
        raise TypeError(
            "{} record values must be {}".format(name, value_type.__name__)
        )
    return result


def _validated_policy(
    max_bracket_ns: int,
    minimum_comparable_pairs: int,
    position_p99_tolerance_m: float,
    heading_p99_tolerance_deg: float,
) -> Tuple[int, int, float, float]:
    if not isinstance(max_bracket_ns, int) or isinstance(max_bracket_ns, bool):
        raise TypeError("max_bracket_ns must be an integer")
    if max_bracket_ns < 0:
        raise ValueError("max_bracket_ns must be non-negative")
    if not isinstance(minimum_comparable_pairs, int) or isinstance(
        minimum_comparable_pairs, bool
    ):
        raise TypeError("minimum_comparable_pairs must be an integer")
    if minimum_comparable_pairs <= 0:
        raise ValueError("minimum_comparable_pairs must be positive")
    position_tolerance = _finite_number(
        position_p99_tolerance_m,
        "position_p99_tolerance_m",
        nonnegative=True,
    )
    heading_tolerance = _finite_number(
        heading_p99_tolerance_deg,
        "heading_p99_tolerance_deg",
        nonnegative=True,
    )
    return (
        max_bracket_ns,
        minimum_comparable_pairs,
        position_tolerance,
        heading_tolerance,
    )


def _ego_values(value: EgoDiagnosticValue) -> Tuple[np.ndarray, float]:
    try:
        position = np.asarray(value.position_xy, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Ego position_xy must contain numeric values") from error
    if position.shape != (2,):
        raise ValueError("Ego position_xy must have shape (2,)")
    if not np.isfinite(position).all():
        raise ValueError("Ego position_xy must contain only finite values")
    heading_deg = _finite_number(value.heading_deg, "Ego heading_deg")
    return position, heading_deg


def aggregate_tf_ego_errors(
    position_errors_m: Sequence[float],
    heading_errors_deg: Sequence[float],
    excluded_count: int,
    *,
    minimum_comparable_pairs: int = 10,
    position_p99_tolerance_m: float = 0.051,
    heading_p99_tolerance_deg: float = 0.5,
) -> TfEgoDiagnostic:
    """Aggregate already epoch-selected TF/Ego errors under the shared gate."""

    (
        _,
        minimum_pairs,
        position_tolerance,
        heading_tolerance,
    ) = _validated_policy(
        0,
        minimum_comparable_pairs,
        position_p99_tolerance_m,
        heading_p99_tolerance_deg,
    )
    if not isinstance(excluded_count, int) or isinstance(excluded_count, bool):
        raise TypeError("excluded_count must be an integer")
    if excluded_count < 0:
        raise ValueError("excluded_count must be non-negative")
    try:
        positions = np.asarray(position_errors_m, dtype=np.float64)
        headings = np.asarray(heading_errors_deg, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("TF/Ego errors must contain numeric values") from error
    if positions.ndim != 1 or headings.ndim != 1:
        raise ValueError("TF/Ego errors must be one-dimensional sequences")
    if positions.size != headings.size:
        raise ValueError("TF/Ego position and heading error counts must match")
    if not np.isfinite(positions).all() or not np.isfinite(headings).all():
        raise ValueError("TF/Ego errors must contain only finite values")
    if np.any(positions < 0.0) or np.any(headings < 0.0):
        raise ValueError("TF/Ego errors must be non-negative")

    comparable_count = int(positions.size)
    if comparable_count < minimum_pairs:
        raise TfEgoPoseMismatch(
            "TF/Ego comparable pair count {} is below minimum {}".format(
                comparable_count, minimum_pairs
            )
        )

    diagnostic = TfEgoDiagnostic(
        comparable_count=comparable_count,
        excluded_count=excluded_count,
        position_p99_m=nearest_rank_percentile(positions, 0.99),
        position_max_m=float(np.max(positions)),
        heading_p99_deg=nearest_rank_percentile(headings, 0.99),
        heading_max_deg=float(np.max(headings)),
    )
    if diagnostic.position_p99_m > position_tolerance:
        raise TfEgoPoseMismatch(
            "TF/Ego position p99 {} m exceeds tolerance {} m".format(
                diagnostic.position_p99_m, position_tolerance
            ),
            diagnostic=diagnostic,
        )
    if diagnostic.heading_p99_deg > heading_tolerance:
        raise TfEgoPoseMismatch(
            "TF/Ego heading p99 {} deg exceeds tolerance {} deg".format(
                diagnostic.heading_p99_deg, heading_tolerance
            ),
            diagnostic=diagnostic,
        )
    return diagnostic


def compute_tf_ego_diagnostic(
    ego_records: Sequence[StampedRecord[EgoDiagnosticValue]],
    tf_records: Sequence[StampedRecord[Pose]],
    max_bracket_ns: int,
    *,
    minimum_comparable_pairs: int = 10,
    position_p99_tolerance_m: float = 0.051,
    heading_p99_tolerance_deg: float = 0.5,
) -> TfEgoDiagnostic:
    """Compare diagnostic Ego pose fields to bracketed canonical TF poses."""

    (
        bracket_limit,
        minimum_pairs,
        position_tolerance,
        heading_tolerance,
    ) = _validated_policy(
        max_bracket_ns,
        minimum_comparable_pairs,
        position_p99_tolerance_m,
        heading_p99_tolerance_deg,
    )
    ego = _validated_records(ego_records, EgoDiagnosticValue, "ego_records")
    poses = _validated_records(tf_records, Pose, "tf_records")
    position_errors = []
    heading_errors_deg = []
    excluded_count = 0

    for record in ego:
        bracket = interpolation_bracket(
            poses,
            target_ns=record.stamp_ns,
            max_gap_ns=bracket_limit,
        )
        if bracket is None:
            excluded_count += 1
            continue
        tf_pose = interpolate_pose(bracket)
        ego_position, ego_heading_deg = _ego_values(record.value)
        tf_position = np.asarray(tf_pose.translation_xyz[:2], dtype=np.float64)
        delta = ego_position - tf_position
        position_error = math.hypot(float(delta[0]), float(delta[1]))
        if not math.isfinite(position_error):
            raise ValueError("TF/Ego position error must be finite")
        position_errors.append(position_error)
        tf_heading_rad = _quaternion_yaw(tf_pose.quaternion_xyzw)
        ego_heading_rad = math.radians(ego_heading_deg)
        circular_error_rad = abs(_wrap_to_pi(ego_heading_rad - tf_heading_rad))
        heading_errors_deg.append(float(math.degrees(circular_error_rad)))

    return aggregate_tf_ego_errors(
        position_errors_m=position_errors,
        heading_errors_deg=heading_errors_deg,
        excluded_count=excluded_count,
        minimum_comparable_pairs=minimum_pairs,
        position_p99_tolerance_m=position_tolerance,
        heading_p99_tolerance_deg=heading_tolerance,
    )
