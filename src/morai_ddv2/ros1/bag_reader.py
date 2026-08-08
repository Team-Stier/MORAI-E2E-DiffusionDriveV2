from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError, IntegrityError, MoraiDDV2Error
from morai_ddv2.preprocess.camera import CameraDecodeError, inspect_jpeg
from morai_ddv2.preprocess.diagnostics import (
    TfEgoPoseMismatch,
    aggregate_tf_ego_errors,
)
from morai_ddv2.preprocess.geometry import (
    Pose,
    _quaternion_yaw,
    _wrap_to_pi,
    interpolate_pose,
    normalize_quaternion,
)
from morai_ddv2.preprocess.lidar import PointCloudLayoutError
from morai_ddv2.preprocess.models import Bracket, StampedRecord

from .messages import (
    BagDisposition,
    BagIndex,
    BagInventory,
    BagRejectReason,
    ClockEpochView,
    EgoRecord,
    EgoValue,
    EpochSafeTfEgoDiagnostic,
    Extrinsic,
    PayloadEvent,
    PointCloudDecodeError,
    RecordRef,
    SensorPayload,
    SensorPayloadFailure,
    SensorPayloadKind,
    TfEgoGateProvenance,
    TfPoseRecord,
    ValueRecord,
    ego_value_from_message,
    frame_id_from_message,
    header_stamp_ns_from_message,
    normalize_frame_id,
    pointcloud_layout_from_message,
    pointcloud_xyz_i,
    pose_from_transform,
    record_ref_from_message,
    stamp_to_ns,
    transform_frames,
)


class BagReadError(MoraiDDV2Error):
    """Raised when a bag cannot provide a trustworthy inventory stream."""


class _EpochSafeTfEgoMismatch(TfEgoPoseMismatch):
    def __init__(
        self, message: str, report: EpochSafeTfEgoDiagnostic
    ) -> None:
        self.epoch_safe_diagnostic = report
        super().__init__(message, diagnostic=report.metric)


_REJECTION_PRECEDENCE = (
    BagRejectReason.MISSING_REQUIRED_TOPIC,
    BagRejectReason.MESSAGE_TYPE_MISMATCH,
    BagRejectReason.FRAME_ID_MISMATCH,
    BagRejectReason.RESOLUTION_MISMATCH,
    BagRejectReason.EXTRINSIC_MISMATCH,
    BagRejectReason.DUPLICATE_STATE_CONFLICT,
    BagRejectReason.TF_EGO_POSE_MISMATCH,
)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("{} must be an object".format(context))
    return value


def _sequence(value: object, context: str) -> Tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ConfigurationError("{} must be an array".format(context))
    return tuple(value)


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigurationError("{}.{} is required".format(context, key))
    return mapping[key]


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError("{} must be a nonempty string".format(context))
    return value


def _integer(
    value: object,
    context: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError("{} must be an integer".format(context))
    if nonnegative and value < 0:
        raise ConfigurationError("{} must be non-negative".format(context))
    if positive and value <= 0:
        raise ConfigurationError("{} must be positive".format(context))
    return value


def _finite(value: object, context: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ConfigurationError("{} must be a finite number".format(context))
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("{} must be a finite number".format(context)) from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ConfigurationError("{} must be a finite non-negative number".format(context))
    return result


def _float_tuple(value: object, length: int, context: str) -> Tuple[float, ...]:
    items = _sequence(value, context)
    if len(items) != length:
        raise ConfigurationError("{} must contain {} values".format(context, length))
    return tuple(_finite(item, "{}[{}]".format(context, index)) for index, item in enumerate(items))


def _configured_frame(value: object, context: str) -> str:
    try:
        return normalize_frame_id(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("{} must be a canonicalizable frame".format(context)) from error


def _domain_policy(domain: ResolvedDomain) -> Dict[str, Any]:
    if not isinstance(domain, ResolvedDomain):
        raise ConfigurationError("domain must be a verified ResolvedDomain")
    root = _mapping(domain.document, "resolved_domain")
    sensors = _mapping(_required(root, "sensors", "resolved_domain"), "sensors")
    camera_rows = _sequence(_required(sensors, "cameras", "sensors"), "sensors.cameras")
    cameras: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(camera_rows):
        row = _mapping(raw, "sensors.cameras[{}]".format(index))
        role = _string(_required(row, "role", "camera"), "camera.role")
        if role in cameras:
            raise ConfigurationError("duplicate camera role: {}".format(role))
        resolution = _sequence(_required(row, "resolution_wh", "camera"), "camera.resolution_wh")
        if len(resolution) != 2:
            raise ConfigurationError("camera.resolution_wh must contain two values")
        cameras[role] = {
            "topic": _string(_required(row, "topic", "camera"), "camera.topic"),
            "frame": _configured_frame(
                _required(row, "frame_id", "camera"), "camera.frame_id"
            ),
            "resolution": (
                _integer(resolution[0], "camera.resolution_wh[0]", positive=True),
                _integer(resolution[1], "camera.resolution_wh[1]", positive=True),
            ),
            "translation": _float_tuple(
                _required(row, "translation_base_from_camera_m", "camera"),
                3,
                "camera.translation",
            ),
            "quaternion": _float_tuple(
                _required(row, "quaternion_base_from_camera_xyzw", "camera"),
                4,
                "camera.quaternion",
            ),
        }
    if set(cameras) != {"front", "left", "right"}:
        raise ConfigurationError("resolved domain must define front, left, and right cameras")

    lidar = _mapping(_required(sensors, "lidar", "sensors"), "sensors.lidar")
    lidar_policy = {
        "topic": _string(_required(lidar, "topic", "lidar"), "lidar.topic"),
        "frame": _configured_frame(
            _required(lidar, "frame_id", "lidar"), "lidar.frame_id"
        ),
        "translation": _float_tuple(
            _required(lidar, "translation_base_from_lidar_m", "lidar"),
            3,
            "lidar.translation",
        ),
        "quaternion": _float_tuple(
            _required(lidar, "quaternion_base_from_lidar_xyzw", "lidar"),
            4,
            "lidar.quaternion",
        ),
    }

    contract = _mapping(_required(root, "ros_contract", "resolved_domain"), "ros_contract")
    ego_feature_fields = _sequence(
        _required(contract, "ego_feature_fields", "ros_contract"),
        "ros_contract.ego_feature_fields",
    )
    if ego_feature_fields != (
        "velocity.x",
        "velocity.y",
        "acceleration.x",
        "acceleration.y",
    ):
        raise ConfigurationError(
            "ros_contract.ego_feature_fields must exactly match the supported contract"
        )
    ego_diagnostic_fields = _sequence(
        _required(contract, "ego_diagnostic_fields", "ros_contract"),
        "ros_contract.ego_diagnostic_fields",
    )
    if ego_diagnostic_fields != ("position.x", "position.y", "heading_deg"):
        raise ConfigurationError(
            "ros_contract.ego_diagnostic_fields must exactly match the supported contract"
        )
    lidar_point_fields = _mapping(
        _required(contract, "lidar_point_fields", "ros_contract"),
        "ros_contract.lidar_point_fields",
    )
    if dict(lidar_point_fields) != {
        "x": "float32",
        "y": "float32",
        "z": "float32",
        "intensity": "float32",
    }:
        raise ConfigurationError(
            "ros_contract.lidar_point_fields must exactly match the supported contract"
        )
    message_types = _mapping(_required(contract, "message_types", "ros_contract"), "message_types")
    topics = _mapping(_required(contract, "policy_topics", "ros_contract"), "policy_topics")
    transform_contract = _mapping(_required(contract, "pose_transform", "ros_contract"), "pose_transform")
    diagnostics = _mapping(_required(root, "diagnostics", "resolved_domain"), "diagnostics")
    sampling = _mapping(_required(root, "sampling", "resolved_domain"), "sampling")
    clock_policy = _mapping(_required(sampling, "clock_policy", "sampling"), "sampling.clock_policy")
    tf_ego_gate = _mapping(_required(clock_policy, "tf_ego_bag_gate", "clock_policy"), "tf_ego_bag_gate")

    role_order = ("front", "left", "right")
    sensor_topics = tuple(cameras[role]["topic"] for role in role_order) + (lidar_policy["topic"],)
    tf_topic = _string(_required(topics, "transforms", "policy_topics"), "policy_topics.transforms")
    ego_topic = _string(_required(topics, "ego_state", "policy_topics"), "policy_topics.ego_state")
    expected_types = {
        cameras[role]["topic"]: _string(_required(message_types, "camera", "message_types"), "message_types.camera")
        for role in role_order
    }
    expected_types[lidar_policy["topic"]] = _string(
        _required(message_types, "lidar", "message_types"), "message_types.lidar"
    )
    expected_types[tf_topic] = _string(
        _required(message_types, "transforms", "message_types"), "message_types.transforms"
    )
    expected_types[ego_topic] = _string(
        _required(message_types, "ego_state", "message_types"), "message_types.ego_state"
    )

    return {
        "cameras": cameras,
        "lidar": lidar_policy,
        "sensor_topics": sensor_topics,
        "tf_topic": tf_topic,
        "ego_topic": ego_topic,
        "topic_order": sensor_topics + (tf_topic, ego_topic),
        "expected_types": expected_types,
        "map_frame": _configured_frame(
            _required(transform_contract, "parent_frame", "pose_transform"),
            "pose_transform.parent_frame",
        ),
        "base_frame": _configured_frame(
            _required(transform_contract, "child_frame", "pose_transform"),
            "pose_transform.child_frame",
        ),
        "duplicate_tf_translation": _finite(
            _required(diagnostics, "duplicate_tf_translation_tolerance_m", "diagnostics"),
            "diagnostics.duplicate_tf_translation_tolerance_m",
            nonnegative=True,
        ),
        "duplicate_tf_rotation": _finite(
            _required(diagnostics, "duplicate_tf_rotation_tolerance_rad", "diagnostics"),
            "diagnostics.duplicate_tf_rotation_tolerance_rad",
            nonnegative=True,
        ),
        "extrinsic_translation": _finite(
            _required(diagnostics, "extrinsic_translation_tolerance_m", "diagnostics"),
            "diagnostics.extrinsic_translation_tolerance_m",
            nonnegative=True,
        ),
        "extrinsic_rotation": _finite(
            _required(diagnostics, "extrinsic_rotation_tolerance_rad", "diagnostics"),
            "diagnostics.extrinsic_rotation_tolerance_rad",
            nonnegative=True,
        ),
        "minimum_pairs": _integer(
            _required(diagnostics, "tf_ego_minimum_comparable_pairs", "diagnostics"),
            "diagnostics.tf_ego_minimum_comparable_pairs",
            positive=True,
        ),
        "position_tolerance": _finite(
            _required(diagnostics, "tf_ego_position_p99_tolerance_m", "diagnostics"),
            "diagnostics.tf_ego_position_p99_tolerance_m",
            nonnegative=True,
        ),
        "heading_tolerance": _finite(
            _required(diagnostics, "tf_ego_heading_p99_tolerance_deg", "diagnostics"),
            "diagnostics.tf_ego_heading_p99_tolerance_deg",
            nonnegative=True,
        ),
        "diagnostic_max_bracket_ns": _integer(
            _required(tf_ego_gate, "max_bracket_ns", "tf_ego_bag_gate"),
            "tf_ego_bag_gate.max_bracket_ns",
            nonnegative=True,
        ),
    }


def discover_bags(bag_root: Path) -> Tuple[Path, ...]:
    """Return finalized ROS1 bags in deterministic relative POSIX order."""

    root = Path(bag_root)
    if not root.exists():
        return ()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    paths = tuple(path for path in root.rglob("*.bag") if path.is_file())
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise BagReadError("unable to hash bag input") from error
    return digest.hexdigest()


def _stat_identity(path: Path) -> Tuple[int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        raise BagReadError("unable to stat bag input") from error
    if not path.is_file():
        raise BagReadError("bag path must resolve to a regular file")
    return int(stat.st_size), int(stat.st_mtime_ns)


@contextmanager
def _open_bag(path: Path) -> Iterator[object]:
    """Default ROS backend; importing ROS is intentionally delayed until use."""

    try:
        import rosbag

        bag = rosbag.Bag(str(path), "r")
    except Exception as error:
        raise BagReadError("unable to open ROS1 bag") from error
    try:
        yield bag
    finally:
        try:
            bag.close()
        except Exception as error:
            raise BagReadError("unable to close ROS1 bag") from error


def _connection_types(bag: object) -> Tuple[Tuple[str, str], ...]:
    """Read every connection datatype; never trust the collapsed topic summary."""

    try:
        connection_factory = getattr(bag, "_get_connections")
        connections = tuple(connection_factory())
    except Exception as error:
        raise BagReadError("unable to inspect bag connections") from error
    pairs = []
    for connection in connections:
        try:
            topic = getattr(connection, "topic")
            datatype = getattr(connection, "datatype")
        except AttributeError as error:
            raise BagReadError("bag connection lacks topic/datatype") from error
        if not isinstance(topic, str) or not isinstance(datatype, str):
            raise BagReadError("bag connection topic/datatype must be strings")
        pairs.append((topic, datatype))
    return tuple(pairs)


def _camera_payload(message: object) -> bytes:
    try:
        value = getattr(message, "data")
    except Exception as error:
        raise CameraDecodeError("camera data is inaccessible") from error
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise CameraDecodeError("camera data is not bytes-like")
    return bytes(value)


def _pose_equivalent(a: Pose, b: Pose, translation_tolerance: float, rotation_tolerance: float) -> bool:
    delta = np.asarray(a.translation_xyz, dtype=np.float64) - np.asarray(
        b.translation_xyz, dtype=np.float64
    )
    q0 = normalize_quaternion(a.quaternion_xyzw)
    q1 = normalize_quaternion(b.quaternion_xyzw)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    rotation_distance = 4.0 * math.atan2(
        float(np.linalg.norm(q0 - q1)),
        float(np.linalg.norm(q0 + q1)),
    )
    return float(np.linalg.norm(delta)) <= translation_tolerance and (
        rotation_distance <= rotation_tolerance
    )


def _ego_equivalent(a: EgoValue, b: EgoValue) -> bool:
    return (
        a.velocity_xy == b.velocity_xy
        and a.acceleration_xy == b.acceleration_xy
    )


def _segment_ego_epochs(
    values: Sequence[ValueRecord[EgoValue]],
) -> Tuple[Tuple[EgoRecord, ...], Tuple[ClockEpochView[EgoRecord], ...], bool]:
    ordered = tuple(sorted(values, key=lambda record: (record.receipt_stamp_ns, record.stable_record_index)))
    raw: List[EgoRecord] = []
    epoch = 0
    previous_header: Optional[int] = None
    for record in ordered:
        if previous_header is not None and record.header_stamp_ns < previous_header:
            epoch += 1
        raw.append(
            EgoRecord(
                record.header_stamp_ns,
                record.receipt_stamp_ns,
                record.stable_record_index,
                epoch,
                record.value,
            )
        )
        previous_header = record.header_stamp_ns
    views = []
    conflict = False
    for index in range(epoch + 1 if raw else 0):
        members = tuple(record for record in raw if record.clock_epoch_index == index)
        collapsed = []
        for record in members:
            if collapsed and collapsed[-1].header_stamp_ns == record.header_stamp_ns:
                if not _ego_equivalent(collapsed[-1].value, record.value):
                    conflict = True
                continue
            collapsed.append(record)
        views.append(ClockEpochView(index, members, tuple(collapsed)))
    return tuple(raw), tuple(views), conflict


def _segment_tf_epochs(
    values: Sequence[ValueRecord[Pose]],
    translation_tolerance: float,
    rotation_tolerance: float,
) -> Tuple[Tuple[TfPoseRecord, ...], Tuple[ClockEpochView[TfPoseRecord], ...], bool]:
    ordered = tuple(sorted(values, key=lambda record: (record.receipt_stamp_ns, record.stable_record_index)))
    raw: List[TfPoseRecord] = []
    epoch = 0
    previous_header: Optional[int] = None
    for record in ordered:
        if previous_header is not None and record.header_stamp_ns < previous_header:
            epoch += 1
        raw.append(
            TfPoseRecord(
                record.header_stamp_ns,
                record.receipt_stamp_ns,
                record.stable_record_index,
                epoch,
                record.value,
            )
        )
        previous_header = record.header_stamp_ns
    views = []
    conflict = False
    for index in range(epoch + 1 if raw else 0):
        members = tuple(record for record in raw if record.clock_epoch_index == index)
        collapsed = []
        for record in members:
            if collapsed and collapsed[-1].header_stamp_ns == record.header_stamp_ns:
                if not _pose_equivalent(
                    collapsed[-1].value,
                    record.value,
                    translation_tolerance,
                    rotation_tolerance,
                ):
                    conflict = True
                continue
            collapsed.append(record)
        views.append(ClockEpochView(index, members, tuple(collapsed)))
    return tuple(raw), tuple(views), conflict


def causal_tf_interpolation_view(
    epoch: ClockEpochView[TfPoseRecord],
    anchor: TfPoseRecord,
) -> Tuple[TfPoseRecord, ...]:
    """Return the duplicate-collapsed same-epoch raw suffix from an exact anchor."""

    if not isinstance(epoch, ClockEpochView):
        raise TypeError("epoch must be a ClockEpochView")
    if not isinstance(anchor, TfPoseRecord):
        raise TypeError("anchor must be a TfPoseRecord")
    if anchor.clock_epoch_index != epoch.index:
        raise ValueError("anchor clock epoch does not match view")
    position = next(
        (index for index, record in enumerate(epoch.raw_records) if record is anchor),
        None,
    )
    if position is None:
        raise ValueError("anchor must be an identity member of epoch.raw_records")
    suffix = epoch.raw_records[position:]
    collapsed: List[TfPoseRecord] = []
    for record in suffix:
        if collapsed and collapsed[-1].header_stamp_ns == record.header_stamp_ns:
            if not _pose_equivalent(collapsed[-1].value, record.value, 1.0e-6, 1.0e-6):
                raise ValueError("causal TF suffix contains a conflicting duplicate")
            continue
        collapsed.append(record)
    return tuple(collapsed)


def _tf_bracket(
    records: Sequence[TfPoseRecord], target_ns: int, max_bracket_ns: int
) -> Optional[Tuple[TfPoseRecord, TfPoseRecord, float]]:
    if not records:
        return None
    stamps = tuple(record.header_stamp_ns for record in records)
    position = bisect_left(stamps, target_ns)
    if position < len(records) and records[position].header_stamp_ns == target_ns:
        exact = records[position]
        return exact, exact, 0.0
    if position == 0 or position == len(records):
        return None
    before = records[position - 1]
    after = records[position]
    gap = after.header_stamp_ns - before.header_stamp_ns
    if gap <= 0 or gap > max_bracket_ns:
        return None
    return before, after, float((target_ns - before.header_stamp_ns) / gap)


def _global_header_view(
    epochs: Sequence[ClockEpochView[TfPoseRecord]],
) -> Tuple[Tuple[TfPoseRecord, ...], Tuple[int, ...]]:
    records = tuple(sorted(
        (record for view in epochs for record in view.interpolation_records),
        key=lambda record: (record.header_stamp_ns, record.clock_epoch_index, record.stable_record_index),
    ))
    return records, tuple(record.header_stamp_ns for record in records)


def _global_cross_epoch_bracket_exists(
    records: Sequence[TfPoseRecord],
    stamps: Tuple[int, ...],
    target_ns: int,
    max_bracket_ns: int,
) -> bool:
    position = bisect_left(stamps, target_ns)
    if position < len(records) and records[position].header_stamp_ns == target_ns:
        return False
    if position == 0 or position == len(records):
        return False
    before = records[position - 1]
    after = records[position]
    return (
        before.clock_epoch_index != after.clock_epoch_index
        and after.header_stamp_ns - before.header_stamp_ns <= max_bracket_ns
    )


def _receipt_selector_view(
    records: Sequence[TfPoseRecord],
) -> Tuple[Tuple[int, ...], Tuple[TfPoseRecord, ...]]:
    unique = []
    for record in records:
        if not unique or record.receipt_stamp_ns != unique[-1].receipt_stamp_ns:
            unique.append(record)
    return (
        tuple(record.receipt_stamp_ns for record in unique),
        tuple(unique),
    )


def _nearest_receipt_tf(
    receipt_stamps: Tuple[int, ...],
    records: Tuple[TfPoseRecord, ...],
    target_ns: int,
) -> TfPoseRecord:
    position = bisect_left(receipt_stamps, target_ns)
    candidates = []
    if position < len(records):
        candidates.append(records[position])
    if position > 0:
        candidates.append(records[position - 1])
    return min(
        candidates,
        key=lambda record: (
            abs(record.receipt_stamp_ns - target_ns),
            record.receipt_stamp_ns,
            record.stable_record_index,
        ),
    )


def _epoch_safe_components(
    index: BagIndex,
    max_bracket_ns: int,
    *,
    minimum_pairs: int,
    position_tolerance: float,
    heading_tolerance: float,
) -> EpochSafeTfEgoDiagnostic:
    if not isinstance(index, BagIndex):
        raise TypeError("index must be a BagIndex")
    if not isinstance(max_bracket_ns, int) or isinstance(max_bracket_ns, bool):
        raise TypeError("max_bracket_ns must be an integer")
    if max_bracket_ns < 0:
        raise ValueError("max_bracket_ns must be non-negative")
    if not index.raw_poses:
        raise TfEgoPoseMismatch("TF/Ego comparable pair count 0 is below minimum {}".format(minimum_pairs))

    provenance: List[TfEgoGateProvenance] = []
    position_errors = []
    heading_errors = []
    no_local = 0
    ambiguous = 0
    cross_prevented = 0
    tf_receipt_stamps, tf_receipt_records = _receipt_selector_view(index.raw_poses)
    global_header_records, global_header_stamps = _global_header_view(
        index.tf_clock_epochs
    )
    for ego in index.raw_ego:
        anchor = _nearest_receipt_tf(
            tf_receipt_stamps,
            tf_receipt_records,
            ego.receipt_stamp_ns,
        )
        if anchor.clock_epoch_index >= len(index.tf_clock_epochs):
            raise IntegrityError("TF anchor clock epoch is outside the epoch views")
        epoch = index.tf_clock_epochs[anchor.clock_epoch_index]
        if epoch.index != anchor.clock_epoch_index:
            raise IntegrityError("TF clock epoch tuple position/index mismatch")
        bracket = _tf_bracket(
            epoch.interpolation_records, ego.header_stamp_ns, max_bracket_ns
        )
        if bracket is None:
            cross_epoch_bracket = _global_cross_epoch_bracket_exists(
                global_header_records,
                global_header_stamps,
                ego.header_stamp_ns,
                max_bracket_ns,
            )
            if cross_epoch_bracket:
                reason = "CROSS_EPOCH_BRACKET_PREVENTED"
                cross_prevented += 1
            else:
                reason = "NO_EPOCH_LOCAL_BRACKET"
                no_local += 1
            provenance.append(
                TfEgoGateProvenance(
                    ego.header_stamp_ns,
                    ego.receipt_stamp_ns,
                    ego.stable_record_index,
                    anchor.clock_epoch_index,
                    anchor.stable_record_index,
                    None,
                    None,
                    None,
                    reason,
                )
            )
            continue

        before, after, weight = bracket
        pose = interpolate_pose(
            Bracket(
                before=StampedRecord(before.header_stamp_ns, before.stable_record_index, before.value),
                after=StampedRecord(after.header_stamp_ns, after.stable_record_index, after.value),
                weight=weight,
            )
        )
        dx = ego.value.diagnostic.position_xy[0] - pose.translation_xyz[0]
        dy = ego.value.diagnostic.position_xy[1] - pose.translation_xyz[1]
        position_errors.append(float(math.hypot(dx, dy)))
        tf_heading = _quaternion_yaw(pose.quaternion_xyzw)
        ego_heading = math.radians(ego.value.diagnostic.heading_deg)
        heading_errors.append(float(math.degrees(abs(_wrap_to_pi(ego_heading - tf_heading)))))
        provenance.append(
            TfEgoGateProvenance(
                ego.header_stamp_ns,
                ego.receipt_stamp_ns,
                ego.stable_record_index,
                anchor.clock_epoch_index,
                anchor.stable_record_index,
                (before.header_stamp_ns, after.header_stamp_ns),
                (before.receipt_stamp_ns, after.receipt_stamp_ns),
                (before.stable_record_index, after.stable_record_index),
                None,
            )
        )

    excluded = no_local + ambiguous + cross_prevented
    comparable_count = len(position_errors)
    if comparable_count == 0:
        raise TfEgoPoseMismatch(
            "TF/Ego comparable pair count 0 is below minimum {}".format(
                minimum_pairs
            )
        )
    metric = aggregate_tf_ego_errors(
        position_errors,
        heading_errors,
        excluded,
        minimum_comparable_pairs=1,
        position_p99_tolerance_m=float(np.finfo(np.float64).max),
        heading_p99_tolerance_deg=float(np.finfo(np.float64).max),
    )
    report = EpochSafeTfEgoDiagnostic(
        metric=metric,
        provenance=tuple(provenance),
        raw_ego_record_count=len(index.raw_ego),
        epoch_local_comparable_count=comparable_count,
        no_epoch_local_bracket_count=no_local,
        ambiguous_epoch_selection_count=ambiguous,
        cross_epoch_bracket_prevented_count=cross_prevented,
    )
    if comparable_count < minimum_pairs:
        raise _EpochSafeTfEgoMismatch(
            "TF/Ego comparable pair count {} is below minimum {}".format(
                comparable_count, minimum_pairs
            ),
            report,
        )
    if metric.position_p99_m > position_tolerance:
        raise _EpochSafeTfEgoMismatch(
            "TF/Ego position p99 {} m exceeds tolerance {} m".format(
                metric.position_p99_m, position_tolerance
            ),
            report,
        )
    if metric.heading_p99_deg > heading_tolerance:
        raise _EpochSafeTfEgoMismatch(
            "TF/Ego heading p99 {} deg exceeds tolerance {} deg".format(
                metric.heading_p99_deg, heading_tolerance
            ),
            report,
        )
    return report


def compute_epoch_safe_tf_ego_diagnostic(
    index: BagIndex,
    max_bracket_ns: int,
) -> EpochSafeTfEgoDiagnostic:
    """Apply the public fixed v1 gate to receipt-selected TF epoch-local Header views."""

    return _epoch_safe_components(
        index,
        max_bracket_ns,
        minimum_pairs=10,
        position_tolerance=0.051,
        heading_tolerance=0.5,
    )


def _extrinsic_gate(
    extrinsics: Sequence[ValueRecord[Extrinsic]], policy: Mapping[str, Any]
) -> bool:
    expected = {}
    for role in ("front", "left", "right"):
        camera = policy["cameras"][role]
        expected[camera["frame"]] = (camera["translation"], camera["quaternion"])
    lidar = policy["lidar"]
    expected[lidar["frame"]] = (lidar["translation"], lidar["quaternion"])
    seen = set()
    for record in extrinsics:
        value = record.value
        if value.child_frame not in expected:
            continue
        seen.add(value.child_frame)
        translation, quaternion = expected[value.child_frame]
        expected_pose = Pose(translation, quaternion)
        if not _pose_equivalent(
            value.pose,
            expected_pose,
            policy["extrinsic_translation"],
            policy["extrinsic_rotation"],
        ):
            return False
    return seen == set(expected)


def _primary_reason(reasons: Set[BagRejectReason]) -> Optional[BagRejectReason]:
    return next((reason for reason in _REJECTION_PRECEDENCE if reason in reasons), None)


def index_bag(path: Path, domain: ResolvedDomain) -> BagIndex:
    """Index one ROS1 bag in a bounded metadata/small-state first pass."""

    bag_path = Path(path)
    if bag_path.name.endswith(".bag.active"):
        raise BagReadError("active bag paths cannot be indexed")
    if bag_path.suffix != ".bag":
        raise BagReadError("bag path must end in .bag")
    policy = _domain_policy(domain)
    before = _stat_identity(bag_path)
    source_sha256 = _sha256_file(bag_path)
    if _stat_identity(bag_path) != before:
        raise IntegrityError("bag changed while hashing")

    connection_pairs: Tuple[Tuple[str, str], ...] = ()
    counts = {topic: 0 for topic in policy["topic_order"]}
    refs: Dict[str, List[RecordRef]] = {
        topic: [] for topic in policy["sensor_topics"]
    }
    decode_errors: Dict[str, List[int]] = {
        policy["cameras"][role]["topic"]: [] for role in ("front", "left", "right")
    }
    ego_values: List[ValueRecord[EgoValue]] = []
    tf_values: List[ValueRecord[Pose]] = []
    extrinsics: List[ValueRecord[Extrinsic]] = []
    reasons: Set[BagRejectReason] = set()
    tf_flat_index = 0

    try:
        with _open_bag(bag_path) as bag:
            connection_pairs = _connection_types(bag)
            types_by_topic: Dict[str, Set[str]] = {}
            for topic, datatype in connection_pairs:
                types_by_topic.setdefault(topic, set()).add(datatype)

            core_missing = any(topic not in types_by_topic for topic in policy["sensor_topics"])
            tf_present = policy["tf_topic"] in types_by_topic
            ego_present = policy["ego_topic"] in types_by_topic
            if core_missing or tf_present != ego_present:
                reasons.add(BagRejectReason.MISSING_REQUIRED_TOPIC)
            for topic in policy["topic_order"]:
                if topic in types_by_topic and types_by_topic[topic] != {policy["expected_types"][topic]}:
                    reasons.add(BagRejectReason.MESSAGE_TYPE_MISMATCH)

            readable_topics = tuple(
                topic
                for topic in policy["topic_order"]
                if types_by_topic.get(topic) == {policy["expected_types"][topic]}
            )
            readable_topic_set = set(readable_topics)
            present_topics = tuple(
                topic for topic in policy["topic_order"] if topic in types_by_topic
            )
            for topic, message, receipt_stamp in bag.read_messages(topics=present_topics):
                if topic not in counts:
                    continue
                topic_index = counts[topic]
                counts[topic] += 1
                if topic not in readable_topic_set:
                    continue
                receipt_ns = stamp_to_ns(receipt_stamp)

                camera_role = next(
                    (
                        role
                        for role in ("front", "left", "right")
                        if policy["cameras"][role]["topic"] == topic
                    ),
                    None,
                )
                if camera_role is not None:
                    camera = policy["cameras"][camera_role]
                    ref = record_ref_from_message(topic, message, receipt_stamp, topic_index)
                    try:
                        observed_frame = frame_id_from_message(message)
                    except (TypeError, ValueError):
                        reasons.add(BagRejectReason.FRAME_ID_MISMATCH)
                    else:
                        if observed_frame != camera["frame"]:
                            reasons.add(BagRejectReason.FRAME_ID_MISMATCH)
                    valid_jpeg = False
                    try:
                        actual_wh = inspect_jpeg(_camera_payload(message))
                        valid_jpeg = True
                        if actual_wh != camera["resolution"]:
                            reasons.add(BagRejectReason.RESOLUTION_MISMATCH)
                            valid_jpeg = False
                    except (CameraDecodeError, TypeError, ValueError):
                        decode_errors[topic].append(topic_index)
                    if camera_role == "front":
                        if valid_jpeg:
                            refs[topic].append(ref)
                    else:
                        refs[topic].append(ref)
                    continue

                if topic == policy["lidar"]["topic"]:
                    ref = record_ref_from_message(topic, message, receipt_stamp, topic_index)
                    try:
                        observed_frame = frame_id_from_message(message)
                    except (TypeError, ValueError):
                        reasons.add(BagRejectReason.FRAME_ID_MISMATCH)
                    else:
                        if observed_frame != policy["lidar"]["frame"]:
                            reasons.add(BagRejectReason.FRAME_ID_MISMATCH)
                    try:
                        pointcloud_layout_from_message(message)
                    except (PointCloudLayoutError, PointCloudDecodeError, TypeError, ValueError):
                        reasons.add(BagRejectReason.MESSAGE_TYPE_MISMATCH)
                    refs[topic].append(ref)
                    continue

                if topic == policy["ego_topic"]:
                    try:
                        ego_values.append(
                            ValueRecord(
                                header_stamp_ns_from_message(message),
                                receipt_ns,
                                topic_index,
                                ego_value_from_message(message),
                            )
                        )
                    except (TypeError, ValueError, OverflowError):
                        reasons.add(BagRejectReason.MESSAGE_TYPE_MISMATCH)
                    continue

                if topic == policy["tf_topic"]:
                    try:
                        transforms = tuple(getattr(message, "transforms"))
                    except Exception as error:
                        raise BagReadError("TFMessage lacks a transform sequence") from error
                    for transform in transforms:
                        stable_index = tf_flat_index
                        tf_flat_index += 1
                        try:
                            parent, child = transform_frames(transform)
                            header_ns = stamp_to_ns(getattr(getattr(transform, "header"), "stamp"))
                        except (AttributeError, TypeError, ValueError, OverflowError) as error:
                            raise BagReadError("malformed TransformStamped identity") from error
                        if parent == policy["map_frame"] and child == policy["base_frame"]:
                            try:
                                tf_values.append(
                                    ValueRecord(
                                        header_ns,
                                        receipt_ns,
                                        stable_index,
                                        pose_from_transform(transform),
                                    )
                                )
                            except (TypeError, ValueError):
                                reasons.add(BagRejectReason.MESSAGE_TYPE_MISMATCH)
                        elif parent == policy["base_frame"]:
                            expected_children = {
                                policy["cameras"][role]["frame"]
                                for role in ("front", "left", "right")
                            } | {policy["lidar"]["frame"]}
                            if child in expected_children:
                                try:
                                    extrinsics.append(
                                        ValueRecord(
                                            header_ns,
                                            receipt_ns,
                                            stable_index,
                                            Extrinsic(parent, child, pose_from_transform(transform)),
                                        )
                                    )
                                except (TypeError, ValueError):
                                    reasons.add(BagRejectReason.EXTRINSIC_MISMATCH)
    except (ConfigurationError, IntegrityError, BagReadError):
        raise
    except Exception as error:
        raise BagReadError("unable to read/deserialise ROS1 bag") from error

    try:
        post_read_identity = _stat_identity(bag_path)
    except BagReadError as error:
        raise IntegrityError("indexed bag disappeared before verification") from error
    if post_read_identity != before:
        raise IntegrityError("bag changed while indexing")
    try:
        after_sha256 = _sha256_file(bag_path)
    except BagReadError as error:
        raise IntegrityError("unable to re-hash indexed bag") from error
    if after_sha256 != source_sha256:
        raise IntegrityError("bag source bytes changed while indexing")
    try:
        post_hash_identity = _stat_identity(bag_path)
    except BagReadError as error:
        raise IntegrityError("indexed bag disappeared during verification") from error
    if post_hash_identity != before:
        raise IntegrityError("bag changed while re-hashing after indexing")

    raw_ego, ego_epochs, ego_conflict = _segment_ego_epochs(ego_values)
    raw_tf, tf_epochs, tf_conflict = _segment_tf_epochs(
        tf_values,
        policy["duplicate_tf_translation"],
        policy["duplicate_tf_rotation"],
    )
    if ego_conflict or tf_conflict:
        reasons.add(BagRejectReason.DUPLICATE_STATE_CONFLICT)

    types_by_topic = {}
    for topic, datatype in connection_pairs:
        types_by_topic.setdefault(topic, set()).add(datatype)
    tf_present = policy["tf_topic"] in types_by_topic
    ego_present = policy["ego_topic"] in types_by_topic
    both_gt_absent = not tf_present and not ego_present
    core_present = all(topic in types_by_topic for topic in policy["sensor_topics"])
    if tf_present and ego_present and not _extrinsic_gate(extrinsics, policy):
        reasons.add(BagRejectReason.EXTRINSIC_MISMATCH)

    type_summary = tuple(
        (topic, datatype)
        for topic in policy["topic_order"]
        for datatype in sorted(types_by_topic.get(topic, ()))
    )
    count_summary = tuple((topic, counts[topic]) for topic in policy["topic_order"])
    error_summary = tuple(
        (policy["cameras"][role]["topic"], tuple(decode_errors[policy["cameras"][role]["topic"]]))
        for role in ("front", "left", "right")
    )

    provisional_reason = _primary_reason(reasons)
    if provisional_reason is not None:
        disposition = BagDisposition.REJECTED
    elif both_gt_absent and core_present:
        disposition = BagDisposition.SENSOR_ONLY_EXCLUDED
    elif tf_present and ego_present and core_present:
        disposition = BagDisposition.ELIGIBLE
    else:
        disposition = BagDisposition.REJECTED
        provisional_reason = BagRejectReason.MISSING_REQUIRED_TOPIC

    inventory = BagInventory(
        disposition=disposition,
        rejection_code=provisional_reason if disposition is BagDisposition.REJECTED else None,
        topic_types=type_summary,
        topic_record_counts=count_summary,
        camera_decode_error_record_indices=error_summary,
        tf_ego_diagnostic=None,
    )
    result = BagIndex(
        path=bag_path,
        source_bag_sha256=source_sha256,
        inventory=inventory,
        front=tuple(refs[policy["cameras"]["front"]["topic"]]),
        left=tuple(refs[policy["cameras"]["left"]["topic"]]),
        right=tuple(refs[policy["cameras"]["right"]["topic"]]),
        lidar=tuple(refs[policy["lidar"]["topic"]]),
        raw_ego=raw_ego,
        ego_clock_epochs=ego_epochs,
        raw_poses=raw_tf,
        tf_clock_epochs=tf_epochs,
        extrinsics=tuple(extrinsics),
    )

    if disposition is BagDisposition.ELIGIBLE and provisional_reason is None:
        try:
            diagnostic = _epoch_safe_components(
                result,
                policy["diagnostic_max_bracket_ns"],
                minimum_pairs=policy["minimum_pairs"],
                position_tolerance=policy["position_tolerance"],
                heading_tolerance=policy["heading_tolerance"],
            )
        except TfEgoPoseMismatch as error:
            inventory = replace(
                inventory,
                disposition=BagDisposition.REJECTED,
                rejection_code=BagRejectReason.TF_EGO_POSE_MISMATCH,
                tf_ego_diagnostic=getattr(
                    error, "epoch_safe_diagnostic", None
                ),
            )
        else:
            inventory = replace(inventory, tf_ego_diagnostic=diagnostic)
        result = replace(result, inventory=inventory)
    return result


def _validate_selected_indices(
    selected_indices: Mapping[str, Set[int]],
) -> Dict[str, Set[int]]:
    if not isinstance(selected_indices, Mapping):
        raise TypeError("selected_indices must be a mapping")
    result: Dict[str, Set[int]] = {}
    for topic, indices in selected_indices.items():
        if not isinstance(topic, str) or not topic:
            raise ValueError("selected topic names must be nonempty strings")
        if not isinstance(indices, set):
            raise TypeError("selected index values must be sets")
        normalized = set()
        for value in indices:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("selected record indices must be integers")
            if value < 0:
                raise ValueError("selected record indices must be non-negative")
            normalized.add(value)
        result[topic] = normalized
    return result


def iter_selected_payloads(
    path: Path,
    selected_indices: Mapping[str, Set[int]],
) -> Iterator[PayloadEvent]:
    """Reopen a bag and yield exactly requested sensor payload events in read order."""

    bag_path = Path(path)
    if bag_path.name.endswith(".bag.active") or bag_path.suffix != ".bag":
        raise BagReadError("selected payload source must be a finalized .bag")
    selected = _validate_selected_indices(selected_indices)
    before = _stat_identity(bag_path)
    supported_types = {
        "sensor_msgs/CompressedImage": SensorPayloadKind.CAMERA_JPEG,
        "sensor_msgs/PointCloud2": SensorPayloadKind.LIDAR_XYZI,
    }
    seen: Set[Tuple[str, int]] = set()
    counts = {topic: 0 for topic in selected}
    active_topics = tuple(topic for topic, indices in selected.items() if indices)
    try:
        bag_context = _open_bag(bag_path)
        with bag_context as bag:
            pairs = _connection_types(bag)
            types_by_topic: Dict[str, Set[str]] = {}
            for topic, datatype in pairs:
                types_by_topic.setdefault(topic, set()).add(datatype)
            for topic in selected:
                topic_types = types_by_topic.get(topic)
                if topic_types is None:
                    raise ValueError("selected topic is absent from bag")
                if len(topic_types) != 1 or next(iter(topic_types)) not in supported_types:
                    raise ValueError("selected topic is not one supported sensor type")
            try:
                stream = (
                    bag.read_messages(topics=active_topics)
                    if active_topics
                    else ()
                )
                for topic, message, receipt_stamp in stream:
                    if topic not in counts:
                        continue
                    index = counts[topic]
                    counts[topic] += 1
                    if index not in selected[topic]:
                        continue
                    identity = (topic, index)
                    if identity in seen:
                        raise IntegrityError("selected payload identity was emitted twice")
                    seen.add(identity)
                    datatype = next(iter(types_by_topic[topic]))
                    try:
                        ref = record_ref_from_message(topic, message, receipt_stamp, index)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise IntegrityError("selected payload reference is malformed") from error
                    if supported_types[datatype] is SensorPayloadKind.CAMERA_JPEG:
                        try:
                            value = _camera_payload(message)
                        except Exception:
                            yield SensorPayloadFailure(
                                topic,
                                index,
                                "DECODE_ERROR",
                                "CAMERA_DATA_ACCESS_FAILED",
                            )
                            continue
                        yield SensorPayload(ref, SensorPayloadKind.CAMERA_JPEG, value)
                    else:
                        try:
                            value = pointcloud_xyz_i(message)
                        except PointCloudDecodeError:
                            yield SensorPayloadFailure(
                                topic,
                                index,
                                "DECODE_ERROR",
                                "POINTCLOUD_DECODE_FAILED",
                            )
                            continue
                        yield SensorPayload(ref, SensorPayloadKind.LIDAR_XYZI, value)
            except (BagReadError, IntegrityError):
                raise
            except Exception as error:
                raise BagReadError("unable to stream selected payloads") from error
    except (BagReadError, IntegrityError, ValueError, TypeError):
        raise
    except Exception as error:
        raise BagReadError("unable to stream selected payloads") from error
    expected = {(topic, index) for topic, indices in selected.items() for index in indices}
    if seen != expected:
        raise IntegrityError("one or more selected payload indices were not present")
    if _stat_identity(bag_path) != before:
        raise IntegrityError("bag changed during selected payload pass")
