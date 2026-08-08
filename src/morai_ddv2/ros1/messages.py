from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Generic, Literal, Optional, Tuple, TypeVar, Union

import numpy as np

from morai_ddv2.errors import MoraiDDV2Error
from morai_ddv2.preprocess.diagnostics import (
    EgoDiagnosticValue,
    TfEgoDiagnostic,
)
from morai_ddv2.preprocess.geometry import Pose, normalize_quaternion
from morai_ddv2.preprocess.lidar import (
    PointCloudLayout,
    PointFieldLayout,
    validate_pointcloud_layout,
)


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_SANITIZED_DETAIL = re.compile(r"[A-Z][A-Z0-9_]*(?::[A-Z0-9_]+)*\Z")


class BagDisposition(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    SENSOR_ONLY_EXCLUDED = "SENSOR_ONLY_EXCLUDED"
    REJECTED = "REJECTED"


class BagRejectReason(str, Enum):
    MISSING_REQUIRED_TOPIC = "MISSING_REQUIRED_TOPIC"
    MESSAGE_TYPE_MISMATCH = "MESSAGE_TYPE_MISMATCH"
    FRAME_ID_MISMATCH = "FRAME_ID_MISMATCH"
    RESOLUTION_MISMATCH = "RESOLUTION_MISMATCH"
    EXTRINSIC_MISMATCH = "EXTRINSIC_MISMATCH"
    TF_EGO_POSE_MISMATCH = "TF_EGO_POSE_MISMATCH"
    DUPLICATE_STATE_CONFLICT = "DUPLICATE_STATE_CONFLICT"
    FRONT_TIMESTAMP_ORDER_VIOLATION = "FRONT_TIMESTAMP_ORDER_VIOLATION"
    NO_VALID_SAMPLE = "NO_VALID_SAMPLE"


def _integer(name: str, value: object, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be an integer".format(name))
    result = int(value)
    if result < _INT64_MIN or result > _INT64_MAX:
        raise OverflowError("{} is outside signed int64".format(name))
    if nonnegative and result < 0:
        raise ValueError("{} must be non-negative".format(name))
    return result


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a nonempty string".format(name))
    return value


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("{} must be a finite number".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _finite_tuple(name: str, value: object, length: int) -> Tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise TypeError("{} must be a tuple of length {}".format(name, length))
    return tuple(
        _finite_float("{}[{}]".format(name, index), item)
        for index, item in enumerate(value)
    )


def _validate_pose(value: object, name: str) -> Pose:
    if not isinstance(value, Pose):
        raise TypeError("{} must be a Pose".format(name))
    _finite_tuple("{}.translation_xyz".format(name), value.translation_xyz, 3)
    normalize_quaternion(value.quaternion_xyzw)
    return value


def stamp_to_ns(stamp: object) -> int:
    """Convert a ROS-like ``secs``/``nsecs`` stamp exactly to signed int64 ns."""

    try:
        secs = getattr(stamp, "secs")
        nsecs = getattr(stamp, "nsecs")
    except AttributeError as error:
        raise TypeError("stamp must expose integer secs and nsecs") from error
    seconds = _integer("stamp.secs", secs)
    nanoseconds = _integer("stamp.nsecs", nsecs)
    if not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("stamp.nsecs must be in [0, 1000000000)")
    result = seconds * 1_000_000_000 + nanoseconds
    if result < _INT64_MIN or result > _INT64_MAX:
        raise OverflowError("stamp nanoseconds are outside signed int64")
    return result


def normalize_frame_id(frame_id: object) -> str:
    """Strip ROS leading slashes and return a nonempty canonical frame name."""

    if not isinstance(frame_id, str):
        raise TypeError("frame_id must be a string")
    normalized = frame_id.lstrip("/")
    if not normalized:
        raise ValueError("frame_id must name a frame")
    return normalized


T = TypeVar("T")


@dataclass(frozen=True)
class RecordRef:
    topic: str
    header_stamp_ns: int
    receipt_stamp_ns: int
    stable_record_index: int

    def __post_init__(self) -> None:
        _nonempty_string("topic", self.topic)
        _integer("header_stamp_ns", self.header_stamp_ns)
        _integer("receipt_stamp_ns", self.receipt_stamp_ns)
        _integer("stable_record_index", self.stable_record_index, nonnegative=True)


@dataclass(frozen=True)
class ValueRecord(Generic[T]):
    header_stamp_ns: int
    receipt_stamp_ns: int
    stable_record_index: int
    value: T

    def __post_init__(self) -> None:
        _integer("header_stamp_ns", self.header_stamp_ns)
        _integer("receipt_stamp_ns", self.receipt_stamp_ns)
        _integer("stable_record_index", self.stable_record_index, nonnegative=True)
        if self.value is None:
            raise TypeError("value must not be None")


@dataclass(frozen=True)
class EgoValue:
    velocity_xy: Tuple[float, float]
    acceleration_xy: Tuple[float, float]
    diagnostic: EgoDiagnosticValue

    def __post_init__(self) -> None:
        velocity = _finite_tuple("velocity_xy", self.velocity_xy, 2)
        acceleration = _finite_tuple("acceleration_xy", self.acceleration_xy, 2)
        if not isinstance(self.diagnostic, EgoDiagnosticValue):
            raise TypeError("diagnostic must be an EgoDiagnosticValue")
        position = _finite_tuple(
            "diagnostic.position_xy", self.diagnostic.position_xy, 2
        )
        heading = _finite_float("diagnostic.heading_deg", self.diagnostic.heading_deg)
        object.__setattr__(self, "velocity_xy", velocity)
        object.__setattr__(self, "acceleration_xy", acceleration)
        object.__setattr__(
            self,
            "diagnostic",
            EgoDiagnosticValue(position_xy=position, heading_deg=heading),
        )


@dataclass(frozen=True)
class EgoRecord:
    header_stamp_ns: int
    receipt_stamp_ns: int
    stable_record_index: int
    clock_epoch_index: int
    value: EgoValue

    def __post_init__(self) -> None:
        _integer("header_stamp_ns", self.header_stamp_ns)
        _integer("receipt_stamp_ns", self.receipt_stamp_ns)
        _integer("stable_record_index", self.stable_record_index, nonnegative=True)
        _integer("clock_epoch_index", self.clock_epoch_index, nonnegative=True)
        if not isinstance(self.value, EgoValue):
            raise TypeError("value must be an EgoValue")


@dataclass(frozen=True)
class TfPoseRecord:
    header_stamp_ns: int
    receipt_stamp_ns: int
    stable_record_index: int
    clock_epoch_index: int
    value: Pose

    def __post_init__(self) -> None:
        _integer("header_stamp_ns", self.header_stamp_ns)
        _integer("receipt_stamp_ns", self.receipt_stamp_ns)
        _integer("stable_record_index", self.stable_record_index, nonnegative=True)
        _integer("clock_epoch_index", self.clock_epoch_index, nonnegative=True)
        _validate_pose(self.value, "value")


@dataclass(frozen=True)
class ClockEpochView(Generic[T]):
    index: int
    raw_records: Tuple[T, ...]
    interpolation_records: Tuple[T, ...]

    def __post_init__(self) -> None:
        _integer("index", self.index, nonnegative=True)
        if not isinstance(self.raw_records, tuple) or not self.raw_records:
            raise ValueError("raw_records must be a nonempty tuple")
        if not isinstance(self.interpolation_records, tuple) or not self.interpolation_records:
            raise ValueError("interpolation_records must be a nonempty tuple")
        for record in self.raw_records:
            if getattr(record, "clock_epoch_index", None) != self.index:
                raise ValueError("raw record clock_epoch_index must match view index")
        receipt_keys = tuple(
            (record.receipt_stamp_ns, record.stable_record_index)
            for record in self.raw_records
        )
        if any(current <= previous for previous, current in zip(receipt_keys, receipt_keys[1:])):
            raise ValueError("raw records must be in strict receipt/index order")
        raw_headers = tuple(record.header_stamp_ns for record in self.raw_records)
        if any(current < previous for previous, current in zip(raw_headers, raw_headers[1:])):
            raise ValueError("raw records in one epoch must have nondecreasing Header")
        position = 0
        for selected in self.interpolation_records:
            while position < len(self.raw_records) and self.raw_records[position] is not selected:
                position += 1
            if position == len(self.raw_records):
                raise ValueError(
                    "interpolation_records must be an ordered identity-preserving subset"
                )
            position += 1
        interpolation_headers = tuple(
            record.header_stamp_ns for record in self.interpolation_records
        )
        if any(
            current <= previous
            for previous, current in zip(
                interpolation_headers, interpolation_headers[1:]
            )
        ):
            raise ValueError(
                "interpolation record Headers must be strictly increasing"
            )


@dataclass(frozen=True)
class TfEgoGateProvenance:
    ego_header_stamp_ns: int
    ego_receipt_stamp_ns: int
    ego_stable_record_index: int
    selected_tf_clock_epoch_index: int
    selected_tf_anchor_record_index: int
    tf_bracket_header_stamp_ns: Optional[Tuple[int, int]]
    tf_bracket_receipt_ns: Optional[Tuple[int, int]]
    tf_bracket_record_index: Optional[Tuple[int, int]]
    exclusion_reason: Optional[
        Literal[
            "NO_EPOCH_LOCAL_BRACKET",
            "AMBIGUOUS_EPOCH_SELECTION",
            "CROSS_EPOCH_BRACKET_PREVENTED",
        ]
    ]

    def __post_init__(self) -> None:
        _integer("ego_header_stamp_ns", self.ego_header_stamp_ns)
        _integer("ego_receipt_stamp_ns", self.ego_receipt_stamp_ns)
        _integer("ego_stable_record_index", self.ego_stable_record_index, nonnegative=True)
        _integer(
            "selected_tf_clock_epoch_index",
            self.selected_tf_clock_epoch_index,
            nonnegative=True,
        )
        _integer(
            "selected_tf_anchor_record_index",
            self.selected_tf_anchor_record_index,
            nonnegative=True,
        )
        brackets = (
            self.tf_bracket_header_stamp_ns,
            self.tf_bracket_receipt_ns,
            self.tf_bracket_record_index,
        )
        if self.exclusion_reason is None:
            if any(value is None for value in brackets):
                raise ValueError("comparable provenance requires every TF bracket")
        else:
            if self.exclusion_reason not in (
                "NO_EPOCH_LOCAL_BRACKET",
                "AMBIGUOUS_EPOCH_SELECTION",
                "CROSS_EPOCH_BRACKET_PREVENTED",
            ):
                raise ValueError("invalid TF/Ego exclusion_reason")
            if any(value is not None for value in brackets):
                raise ValueError("excluded provenance must not carry a TF bracket")
        for name, value in zip(
            (
                "tf_bracket_header_stamp_ns",
                "tf_bracket_receipt_ns",
                "tf_bracket_record_index",
            ),
            brackets,
        ):
            if value is None:
                continue
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError("{} must be a pair".format(name))
            for item in value:
                _integer(name, item, nonnegative=name.endswith("record_index"))


@dataclass(frozen=True)
class EpochSafeTfEgoDiagnostic:
    metric: TfEgoDiagnostic
    provenance: Tuple[TfEgoGateProvenance, ...]
    raw_ego_record_count: int
    epoch_local_comparable_count: int
    no_epoch_local_bracket_count: int
    ambiguous_epoch_selection_count: int
    cross_epoch_bracket_prevented_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.metric, TfEgoDiagnostic):
            raise TypeError("metric must be a TfEgoDiagnostic")
        if not isinstance(self.provenance, tuple) or any(
            not isinstance(item, TfEgoGateProvenance) for item in self.provenance
        ):
            raise TypeError("provenance must contain TfEgoGateProvenance values")
        counts = (
            self.raw_ego_record_count,
            self.epoch_local_comparable_count,
            self.no_epoch_local_bracket_count,
            self.ambiguous_epoch_selection_count,
            self.cross_epoch_bracket_prevented_count,
        )
        for name, count in zip(
            (
                "raw_ego_record_count",
                "epoch_local_comparable_count",
                "no_epoch_local_bracket_count",
                "ambiguous_epoch_selection_count",
                "cross_epoch_bracket_prevented_count",
            ),
            counts,
        ):
            _integer(name, count, nonnegative=True)
        excluded = sum(counts[2:])
        if len(self.provenance) != self.raw_ego_record_count:
            raise ValueError("provenance count must equal raw Ego record count")
        if self.epoch_local_comparable_count + excluded != self.raw_ego_record_count:
            raise ValueError("TF/Ego counters do not reconcile")
        if self.metric.comparable_count != self.epoch_local_comparable_count:
            raise ValueError("metric comparable count does not reconcile")
        if self.metric.excluded_count != excluded:
            raise ValueError("metric excluded count does not reconcile")
        actual_comparable = sum(
            item.exclusion_reason is None for item in self.provenance
        )
        actual_no_local = sum(
            item.exclusion_reason == "NO_EPOCH_LOCAL_BRACKET"
            for item in self.provenance
        )
        actual_ambiguous = sum(
            item.exclusion_reason == "AMBIGUOUS_EPOCH_SELECTION"
            for item in self.provenance
        )
        actual_cross = sum(
            item.exclusion_reason == "CROSS_EPOCH_BRACKET_PREVENTED"
            for item in self.provenance
        )
        if (
            actual_comparable != self.epoch_local_comparable_count
            or actual_no_local != self.no_epoch_local_bracket_count
            or actual_ambiguous != self.ambiguous_epoch_selection_count
            or actual_cross != self.cross_epoch_bracket_prevented_count
        ):
            raise ValueError("TF/Ego provenance reasons do not match counters")


@dataclass(frozen=True)
class Extrinsic:
    parent_frame: str
    child_frame: str
    pose: Pose

    def __post_init__(self) -> None:
        parent = normalize_frame_id(self.parent_frame)
        child = normalize_frame_id(self.child_frame)
        if parent == child:
            raise ValueError("extrinsic parent and child frames must differ")
        _validate_pose(self.pose, "pose")
        object.__setattr__(self, "parent_frame", parent)
        object.__setattr__(self, "child_frame", child)


@dataclass(frozen=True)
class BagInventory:
    disposition: BagDisposition
    rejection_code: Optional[BagRejectReason]
    topic_types: Tuple[Tuple[str, str], ...]
    topic_record_counts: Tuple[Tuple[str, int], ...]
    camera_decode_error_record_indices: Tuple[Tuple[str, Tuple[int, ...]], ...]
    tf_ego_diagnostic: Optional[EpochSafeTfEgoDiagnostic]

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, BagDisposition):
            raise TypeError("disposition must be a BagDisposition")
        if self.disposition is BagDisposition.REJECTED:
            if not isinstance(self.rejection_code, BagRejectReason):
                raise TypeError("rejected inventory requires a BagRejectReason")
        elif self.rejection_code is not None:
            raise ValueError("non-rejected inventory cannot have a rejection_code")
        if not isinstance(self.topic_types, tuple):
            raise TypeError("topic_types must be a tuple")
        if not isinstance(self.topic_record_counts, tuple):
            raise TypeError("topic_record_counts must be a tuple")
        if not isinstance(self.camera_decode_error_record_indices, tuple):
            raise TypeError("camera decode errors must be a tuple")
        if self.tf_ego_diagnostic is not None and not isinstance(
            self.tf_ego_diagnostic, EpochSafeTfEgoDiagnostic
        ):
            raise TypeError("tf_ego_diagnostic must be epoch-safe")


@dataclass(frozen=True)
class BagIndex:
    path: Path
    source_bag_sha256: str
    inventory: BagInventory
    front: Tuple[RecordRef, ...]
    left: Tuple[RecordRef, ...]
    right: Tuple[RecordRef, ...]
    lidar: Tuple[RecordRef, ...]
    raw_ego: Tuple[EgoRecord, ...]
    ego_clock_epochs: Tuple[ClockEpochView[EgoRecord], ...]
    raw_poses: Tuple[TfPoseRecord, ...]
    tf_clock_epochs: Tuple[ClockEpochView[TfPoseRecord], ...]
    extrinsics: Tuple[ValueRecord[Extrinsic], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_bag_sha256) is None:
            raise ValueError("source_bag_sha256 must be lowercase SHA-256")
        if not isinstance(self.inventory, BagInventory):
            raise TypeError("inventory must be a BagInventory")
        for name in (
            "front",
            "left",
            "right",
            "lidar",
            "raw_ego",
            "ego_clock_epochs",
            "raw_poses",
            "tf_clock_epochs",
            "extrinsics",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError("{} must be a tuple".format(name))
        for epochs in (self.ego_clock_epochs, self.tf_clock_epochs):
            if tuple(view.index for view in epochs) != tuple(range(len(epochs))):
                raise ValueError("clock epoch views must have dense zero-based indices")
        for name in ("front", "left", "right", "lidar"):
            if any(not isinstance(record, RecordRef) for record in getattr(self, name)):
                raise TypeError("{} must contain RecordRef values".format(name))
        if any(not isinstance(record, EgoRecord) for record in self.raw_ego):
            raise TypeError("raw_ego must contain EgoRecord values")
        if any(not isinstance(record, TfPoseRecord) for record in self.raw_poses):
            raise TypeError("raw_poses must contain TfPoseRecord values")
        if any(
            not isinstance(record, ValueRecord)
            or not isinstance(record.value, Extrinsic)
            for record in self.extrinsics
        ):
            raise TypeError("extrinsics must contain ValueRecord[Extrinsic] values")
        for flat, epochs, name in (
            (self.raw_ego, self.ego_clock_epochs, "Ego"),
            (self.raw_poses, self.tf_clock_epochs, "TF"),
        ):
            if bool(flat) != bool(epochs):
                raise ValueError(
                    "{} flat raw records and epoch views must both be empty or populated".format(
                        name
                    )
                )
            flattened = tuple(
                record for view in epochs for record in view.raw_records
            )
            if len(flattened) != len(flat) or any(
                left is not right for left, right in zip(flat, flattened)
            ):
                raise ValueError(
                    "{} flat raw records must exactly identity-match epoch views".format(
                        name
                    )
                )


class SensorPayloadKind(str, Enum):
    CAMERA_JPEG = "CAMERA_JPEG"
    LIDAR_XYZI = "LIDAR_XYZI"


@dataclass(frozen=True)
class SensorPayload:
    ref: RecordRef
    kind: SensorPayloadKind
    value: Union[bytes, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.ref, RecordRef):
            raise TypeError("ref must be a RecordRef")
        if not isinstance(self.kind, SensorPayloadKind):
            raise TypeError("kind must be a SensorPayloadKind")
        if self.kind is SensorPayloadKind.CAMERA_JPEG:
            if not isinstance(self.value, bytes):
                raise TypeError("camera JPEG payload must be immutable bytes")
            return
        if not isinstance(self.value, np.ndarray):
            raise TypeError("LiDAR payload must be a NumPy ndarray")
        if self.value.dtype != np.float32 or self.value.ndim != 2 or self.value.shape[1:] != (4,):
            raise ValueError("LiDAR payload must be float32 [N,4]")
        canonical = np.array(self.value, dtype=np.float32, order="C", copy=True)
        canonical.setflags(write=False)
        object.__setattr__(self, "value", canonical)


@dataclass(frozen=True)
class SensorPayloadFailure:
    topic: str
    stable_record_index: int
    reason: Literal["DECODE_ERROR"]
    detail: str

    def __post_init__(self) -> None:
        _nonempty_string("topic", self.topic)
        _integer("stable_record_index", self.stable_record_index, nonnegative=True)
        if self.reason != "DECODE_ERROR":
            raise ValueError("reason must be DECODE_ERROR")
        if not isinstance(self.detail, str) or _SANITIZED_DETAIL.fullmatch(self.detail) is None:
            raise ValueError("detail must be a stable sanitized token")


PayloadEvent = Union[SensorPayload, SensorPayloadFailure]


class PointCloudDecodeError(MoraiDDV2Error):
    """Raised when a selected PointCloud2 cannot become canonical XYZI."""


def header_stamp_ns_from_message(message: object) -> int:
    try:
        return stamp_to_ns(getattr(getattr(message, "header"), "stamp"))
    except AttributeError as error:
        raise TypeError("message must expose header.stamp") from error


def frame_id_from_message(message: object) -> str:
    try:
        return normalize_frame_id(getattr(getattr(message, "header"), "frame_id"))
    except AttributeError as error:
        raise TypeError("message must expose header.frame_id") from error


def record_ref_from_message(
    topic: str,
    message: object,
    receipt_stamp: object,
    stable_record_index: int,
) -> RecordRef:
    return RecordRef(
        topic=topic,
        header_stamp_ns=header_stamp_ns_from_message(message),
        receipt_stamp_ns=stamp_to_ns(receipt_stamp),
        stable_record_index=stable_record_index,
    )


def ego_value_from_message(message: object) -> EgoValue:
    try:
        velocity = getattr(message, "velocity")
        acceleration = getattr(message, "acceleration")
        position = getattr(message, "position")
        return EgoValue(
            velocity_xy=(
                _finite_float("velocity.x", getattr(velocity, "x")),
                _finite_float("velocity.y", getattr(velocity, "y")),
            ),
            acceleration_xy=(
                _finite_float("acceleration.x", getattr(acceleration, "x")),
                _finite_float("acceleration.y", getattr(acceleration, "y")),
            ),
            diagnostic=EgoDiagnosticValue(
                position_xy=(
                    _finite_float("position.x", getattr(position, "x")),
                    _finite_float("position.y", getattr(position, "y")),
                ),
                heading_deg=_finite_float("heading", getattr(message, "heading")),
            ),
        )
    except AttributeError as error:
        raise TypeError("Ego message is missing a required field") from error


def pose_from_transform(transform: object) -> Pose:
    try:
        translation = getattr(getattr(transform, "transform"), "translation")
        rotation = getattr(getattr(transform, "transform"), "rotation")
        xyz = tuple(
            _finite_float("translation.{}".format(axis), getattr(translation, axis))
            for axis in ("x", "y", "z")
        )
        quaternion = normalize_quaternion(
            tuple(
                _finite_float("rotation.{}".format(axis), getattr(rotation, axis))
                for axis in ("x", "y", "z", "w")
            )
        )
    except AttributeError as error:
        raise TypeError("transform is missing translation or rotation fields") from error
    return Pose(
        translation_xyz=xyz,
        quaternion_xyzw=tuple(float(value) for value in quaternion),
    )


def transform_frames(transform: object) -> Tuple[str, str]:
    try:
        parent = getattr(getattr(transform, "header"), "frame_id")
        child = getattr(transform, "child_frame_id")
    except AttributeError as error:
        raise TypeError("transform is missing parent or child frame") from error
    return normalize_frame_id(parent), normalize_frame_id(child)


def pointcloud_layout_from_message(message: object) -> PointCloudLayout:
    try:
        fields = tuple(
            PointFieldLayout(
                name=getattr(field, "name"),
                offset=getattr(field, "offset"),
                datatype=getattr(field, "datatype"),
                count=getattr(field, "count"),
            )
            for field in getattr(message, "fields")
        )
        layout = PointCloudLayout(
            fields=fields,
            point_step=getattr(message, "point_step"),
            is_bigendian=getattr(message, "is_bigendian"),
        )
    except (AttributeError, TypeError) as error:
        raise PointCloudDecodeError("invalid PointCloud2 layout metadata") from error
    validate_pointcloud_layout(layout)
    return layout


def pointcloud_xyz_i(message: object) -> np.ndarray:
    """Decode one selected PointCloud2 in canonical XYZI order, lazily importing ROS."""

    from sensor_msgs import point_cloud2

    try:
        rows = tuple(
            point_cloud2.read_points(
                message,
                field_names=("x", "y", "z", "intensity"),
                skip_nans=False,
            )
        )
        points = np.asarray(rows, dtype=np.float32)
        if points.size == 0:
            points = np.empty((0, 4), dtype=np.float32)
        elif points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("decoded rows do not have four fields")
        return np.ascontiguousarray(points, dtype=np.float32)
    except Exception as error:
        if isinstance(error, PointCloudDecodeError):
            raise
        raise PointCloudDecodeError("unable to decode selected PointCloud2") from error
