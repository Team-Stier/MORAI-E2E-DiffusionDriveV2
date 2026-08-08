from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np

from morai_ddv2.cache.models import CacheSample
from morai_ddv2.config import sha256_file
from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError, IntegrityError, MoraiDDV2Error
from morai_ddv2.ros1 import (
    BagRejectReason,
    RecordRef,
    SensorPayload,
    SensorPayloadFailure,
    SensorPayloadKind,
    iter_selected_payloads,
)

from .camera import (
    CameraDecodeError,
    camera_adapter_config_from_resolved_domain,
    decode_jpeg_rgb,
    stitch_camera_rgb,
)
from .geometry import (
    Pose,
    interpolate_pose,
    interpolate_vector,
    quaternion_angular_distance,
)
from .lidar import (
    EmptyLidarError,
    LidarPointError,
    build_lidar_count,
    lidar_config_from_resolved_domain,
)
from .models import Bracket, StampedRecord
from .planner import (
    BagSamplePlan,
    PlannedBracket,
    PlannedSample,
    SampleDrop,
    SampleDropReason,
    _sampling_policy,
    _sensor_topics,
)
from .trajectory import (
    CommandRules,
    InvalidCommand,
    build_relative_trajectory,
    build_status,
    classify_oracle_command,
)


PayloadKey = Tuple[str, int]
PayloadEvent = Union[SensorPayload, SensorPayloadFailure]


class NoValidSampleError(MoraiDDV2Error):
    rejection_code = BagRejectReason.NO_VALID_SAMPLE


@dataclass(frozen=True)
class _DropOutcome:
    reason: SampleDropReason


def _key(record: RecordRef) -> PayloadKey:
    return record.topic, record.stable_record_index


def _sample_refs(sample: PlannedSample) -> Tuple[RecordRef, RecordRef, RecordRef, RecordRef]:
    return sample.front, sample.left, sample.right, sample.lidar


def _verify_source_hash(bag_path: Path, expected_sha256: str) -> None:
    try:
        actual_source_sha256 = sha256_file(bag_path)
    except OSError as error:
        raise IntegrityError("materialization source bag cannot be hashed") from error
    if actual_source_sha256 != expected_sha256:
        raise IntegrityError(
            "materialization source bytes differ from the plan binding"
        )


def _domain_lidar_pose(domain: ResolvedDomain) -> Pose:
    try:
        lidar = domain.document["sensors"]["lidar"]
        translation = lidar["translation_base_from_lidar_m"]
        quaternion = lidar["quaternion_base_from_lidar_xyzw"]
    except (KeyError, TypeError) as error:
        raise ConfigurationError("resolved domain is missing the LiDAR extrinsic") from error
    if (
        isinstance(translation, (str, bytes))
        or not isinstance(translation, Sequence)
        or len(translation) != 3
        or isinstance(quaternion, (str, bytes))
        or not isinstance(quaternion, Sequence)
        or len(quaternion) != 4
    ):
        raise ConfigurationError("resolved LiDAR extrinsic has invalid axes")
    try:
        translation_tuple = tuple(float(value) for value in translation)
        quaternion_tuple = tuple(float(value) for value in quaternion)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConfigurationError("resolved LiDAR extrinsic must be numeric") from error
    if not all(math.isfinite(value) for value in translation_tuple + quaternion_tuple):
        raise ConfigurationError("resolved LiDAR extrinsic must be finite")
    return Pose(translation_tuple, quaternion_tuple)


def _expected_camera_resolutions(domain: ResolvedDomain) -> Dict[str, Tuple[int, int]]:
    try:
        cameras = domain.document["sensors"]["cameras"]
    except (KeyError, TypeError) as error:
        raise ConfigurationError("resolved domain is missing camera sensors") from error
    result = {}
    for camera in cameras:
        try:
            role = camera["role"]
            resolution = camera["resolution_wh"]
        except (KeyError, TypeError) as error:
            raise ConfigurationError("resolved camera is missing role/resolution") from error
        if role not in ("left", "front", "right") or role in result:
            raise ConfigurationError("resolved camera roles are invalid")
        if (
            isinstance(resolution, (str, bytes))
            or not isinstance(resolution, Sequence)
            or len(resolution) != 2
            or any(type(value) is not int or value <= 0 for value in resolution)
        ):
            raise ConfigurationError("resolved camera resolution is invalid")
        result[role] = (resolution[0], resolution[1])
    if set(result) != {"left", "front", "right"}:
        raise ConfigurationError("resolved domain must define left/front/right cameras")
    return result


def _expected_weight(before_ns: int, after_ns: int, target_ns: int) -> float:
    if before_ns == after_ns:
        return 0.0
    return float(np.float32((target_ns - before_ns) / float(after_ns - before_ns)))


def _validate_planned_bracket(
    bracket: PlannedBracket,
    *,
    target_ns: int,
    max_gap_ns: int,
    expected_epoch: int,
    name: str,
    require_causal_from: Optional[Tuple[int, int]] = None,
) -> None:
    before = bracket.before
    after = bracket.after
    if before.clock_epoch_index != expected_epoch or after.clock_epoch_index != expected_epoch:
        raise IntegrityError("{} crosses a clock epoch".format(name))
    if before.header_stamp_ns > target_ns or after.header_stamp_ns < target_ns:
        raise IntegrityError("{} does not contain its target Header".format(name))
    gap = after.header_stamp_ns - before.header_stamp_ns
    if gap < 0 or gap > max_gap_ns:
        raise IntegrityError("{} exceeds its Header gap contract".format(name))
    if before.header_stamp_ns == after.header_stamp_ns:
        if before is not after:
            raise IntegrityError("{} exact bracket must repeat one raw record".format(name))
    elif before.header_stamp_ns == target_ns or after.header_stamp_ns == target_ns:
        raise IntegrityError("{} must use an exact bracket for an exact target".format(name))
    expected_weight = _expected_weight(
        before.header_stamp_ns, after.header_stamp_ns, target_ns
    )
    if type(bracket.weight) is not float or bracket.weight != expected_weight:
        raise IntegrityError("{} interpolation weight is inconsistent".format(name))
    if bracket.weight == 0.0 and bool(np.signbit(bracket.weight)):
        raise IntegrityError("{} exact weight must be positive zero".format(name))
    if require_causal_from is not None:
        for endpoint in (before, after):
            if (endpoint.receipt_stamp_ns, endpoint.stable_record_index) < require_causal_from:
                raise IntegrityError("{} uses a pre-anchor TF record".format(name))


def _as_pose_bracket(bracket: PlannedBracket) -> Bracket[Pose]:
    return Bracket(
        StampedRecord(
            bracket.before.header_stamp_ns,
            bracket.before.stable_record_index,
            bracket.before.value,
        ),
        StampedRecord(
            bracket.after.header_stamp_ns,
            bracket.after.stable_record_index,
            bracket.after.value,
        ),
        bracket.weight,
    )


def _as_vector_bracket(bracket: PlannedBracket, field_name: str) -> Bracket[np.ndarray]:
    before_value = np.asarray(getattr(bracket.before.value, field_name), dtype=np.float64)
    after_value = np.asarray(getattr(bracket.after.value, field_name), dtype=np.float64)
    return Bracket(
        StampedRecord(
            bracket.before.header_stamp_ns,
            bracket.before.stable_record_index,
            before_value,
        ),
        StampedRecord(
            bracket.after.header_stamp_ns,
            bracket.after.stable_record_index,
            after_value,
        ),
        bracket.weight,
    )


class MaterializationStream(Iterator[CacheSample]):
    def __init__(
        self,
        *,
        bag_path: Path,
        plan: BagSamplePlan,
        domain: ResolvedDomain,
    ) -> None:
        self._planner_drops = tuple(plan.drops)
        self._pass2_drops = []
        self.valid_sample_count = 0
        self._terminal = False
        self._active_event_iterator: Optional[Iterator[PayloadEvent]] = None
        self._generator = self._run(bag_path, plan, domain)

    @property
    def drop_records(self) -> Tuple[SampleDrop, ...]:
        return tuple(
            sorted(
                self._planner_drops + tuple(self._pass2_drops),
                key=lambda item: item.grid_receipt_ns,
            )
        )

    def __iter__(self) -> "MaterializationStream":
        return self

    def __next__(self) -> CacheSample:
        if self._terminal:
            raise StopIteration
        try:
            return next(self._generator)
        except StopIteration:
            self._terminal = True
            raise
        except Exception:
            self._close_active_iterator()
            self._terminal = True
            raise

    def __enter__(self) -> "MaterializationStream":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        self._close_active_iterator()
        if not self._terminal:
            self._generator.close()
        self._terminal = True

    def _close_active_iterator(self) -> None:
        active = self._active_event_iterator
        close = getattr(active, "close", None)
        if callable(close):
            close()
        self._active_event_iterator = None

    def _record_drop(self, drop: SampleDrop) -> None:
        if (
            self._pass2_drops
            and drop.grid_receipt_ns <= self._pass2_drops[-1].grid_receipt_ns
        ):
            raise IntegrityError(
                "pass-2 drops must be emitted in strict receipt-grid order"
            )
        self._pass2_drops.append(drop)

    def _run(
        self,
        bag_path: Path,
        plan: BagSamplePlan,
        domain: ResolvedDomain,
    ) -> Iterator[CacheSample]:
        _verify_source_hash(bag_path, plan.source_bag_sha256)
        topics = _sensor_topics(domain)
        roles_by_topic = {
            topics[0]: "front",
            topics[1]: "left",
            topics[2]: "right",
            topics[3]: "lidar",
        }
        camera_config = camera_adapter_config_from_resolved_domain(domain)
        camera_resolutions = _expected_camera_resolutions(domain)
        lidar_config = lidar_config_from_resolved_domain(domain)
        lidar_pose = _domain_lidar_pose(domain)
        command_rules = CommandRules.from_resolved_domain(domain)
        policy = _sampling_policy(domain)

        requested_refs: Dict[PayloadKey, RecordRef] = {}
        consumers: Dict[PayloadKey, int] = {}
        sample_keys = []
        for sample in plan.samples:
            keys = []
            for record in _sample_refs(sample):
                key = _key(record)
                previous = requested_refs.get(key)
                if previous is not None and previous != record:
                    raise IntegrityError("one selected key has conflicting pass-1 provenance")
                requested_refs[key] = record
                consumers[key] = consumers.get(key, 0) + 1
                keys.append(key)
            sample_keys.append(tuple(keys))

        requested_by_topic = {topic: set() for topic in topics}
        for topic, index in requested_refs:
            if topic not in requested_by_topic:
                raise IntegrityError("plan requests a non-physical sensor topic")
            requested_by_topic[topic].add(index)
        if set(plan.selected_payload_indices) != set(topics) or any(
            set(plan.selected_payload_indices[topic]) != requested_by_topic[topic]
            for topic in topics
        ):
            raise IntegrityError("plan selected-index union does not match planned samples")

        outcomes: Dict[PayloadKey, PayloadEvent] = {}
        decoded: Dict[PayloadKey, Union[np.ndarray, _DropOutcome]] = {}
        seen = set()
        next_sample = 0

        def decode_camera(key: PayloadKey) -> Union[np.ndarray, _DropOutcome]:
            cached = decoded.get(key)
            if cached is not None:
                return cached
            event = outcomes[key]
            if isinstance(event, SensorPayloadFailure):
                result: Union[np.ndarray, _DropOutcome] = _DropOutcome(
                    SampleDropReason.DECODE_ERROR
                )
            else:
                role = roles_by_topic[key[0]]
                try:
                    result = decode_jpeg_rgb(
                        event.value, camera_resolutions[role]
                    )
                except CameraDecodeError:
                    result = _DropOutcome(SampleDropReason.DECODE_ERROR)
            decoded[key] = result
            return result

        def decode_lidar(key: PayloadKey) -> Union[np.ndarray, _DropOutcome]:
            cached = decoded.get(key)
            if cached is not None:
                return cached
            event = outcomes[key]
            if isinstance(event, SensorPayloadFailure):
                result: Union[np.ndarray, _DropOutcome] = _DropOutcome(
                    SampleDropReason.DECODE_ERROR
                )
            else:
                try:
                    result = build_lidar_count(event.value, lidar_pose, lidar_config)
                except EmptyLidarError:
                    result = _DropOutcome(SampleDropReason.EMPTY_LIDAR)
                except LidarPointError:
                    result = _DropOutcome(SampleDropReason.NONFINITE_VALUE)
            decoded[key] = result
            return result

        def resolve_sample(index: int) -> Union[CacheSample, _DropOutcome]:
            sample = plan.samples[index]
            front_key, left_key, right_key, lidar_key = sample_keys[index]
            for key in (front_key, left_key, right_key, lidar_key):
                if isinstance(outcomes[key], SensorPayloadFailure):
                    return _DropOutcome(SampleDropReason.DECODE_ERROR)

            # Physical decode order is deliberately left, front, right.
            left_rgb = decode_camera(left_key)
            if isinstance(left_rgb, _DropOutcome):
                return left_rgb
            front_rgb = decode_camera(front_key)
            if isinstance(front_rgb, _DropOutcome):
                return front_rgb
            right_rgb = decode_camera(right_key)
            if isinstance(right_rgb, _DropOutcome):
                return right_rgb
            camera_rgb = stitch_camera_rgb(
                left_rgb, front_rgb, right_rgb, camera_config
            )

            lidar_count = decode_lidar(lidar_key)
            if isinstance(lidar_count, _DropOutcome):
                return lidar_count

            anchor = sample.trajectory_anchor_tf
            if (
                sample.current_pose_bracket.before is not anchor
                or sample.current_pose_bracket.after is not anchor
                or sample.current_pose_bracket.weight != 0.0
                or bool(np.signbit(sample.current_pose_bracket.weight))
            ):
                raise IntegrityError("current pose bracket must repeat the exact raw TF anchor")
            _validate_planned_bracket(
                sample.ego_bracket,
                target_ns=anchor.header_stamp_ns,
                max_gap_ns=policy.ego_max_bracket_ns,
                expected_epoch=sample.ego_bracket_clock_epoch_index,
                name="Ego bracket",
            )
            causal_origin = (anchor.receipt_stamp_ns, anchor.stable_record_index)
            for future_index, (offset, bracket) in enumerate(
                zip(policy.future_offsets_ns, sample.future_pose_brackets)
            ):
                _validate_planned_bracket(
                    bracket,
                    target_ns=anchor.header_stamp_ns + offset,
                    max_gap_ns=policy.future_max_bracket_ns,
                    expected_epoch=anchor.clock_epoch_index,
                    name="future TF bracket {}".format(future_index),
                    require_causal_from=causal_origin,
                )

            try:
                interpolated_current = interpolate_pose(
                    _as_pose_bracket(sample.current_pose_bracket)
                )
                if not np.array_equal(
                    np.asarray(interpolated_current.translation_xyz),
                    np.asarray(anchor.value.translation_xyz),
                ) or quaternion_angular_distance(
                    interpolated_current.quaternion_xyzw,
                    anchor.value.quaternion_xyzw,
                ) > 1.0e-6:
                    raise IntegrityError("current interpolation changed the raw anchor pose")
                current_pose = anchor.value
                future_poses = tuple(
                    interpolate_pose(_as_pose_bracket(bracket))
                    for bracket in sample.future_pose_brackets
                )
                velocity = interpolate_vector(
                    _as_vector_bracket(sample.ego_bracket, "velocity_xy")
                )
                acceleration = interpolate_vector(
                    _as_vector_bracket(sample.ego_bracket, "acceleration_xy")
                )
                trajectory = build_relative_trajectory(current_pose, future_poses)
            except (ValueError, OverflowError):
                return _DropOutcome(SampleDropReason.NONFINITE_VALUE)
            try:
                command = classify_oracle_command(trajectory, command_rules)
            except InvalidCommand:
                return _DropOutcome(SampleDropReason.INVALID_COMMAND)
            try:
                status = build_status(command, velocity, acceleration)
            except (ValueError, OverflowError):
                return _DropOutcome(SampleDropReason.NONFINITE_VALUE)

            refs = _sample_refs(sample)
            future_headers = np.asarray(
                [
                    (bracket.before.header_stamp_ns, bracket.after.header_stamp_ns)
                    for bracket in sample.future_pose_brackets
                ],
                dtype=np.int64,
            )
            future_receipts = np.asarray(
                [
                    (bracket.before.receipt_stamp_ns, bracket.after.receipt_stamp_ns)
                    for bracket in sample.future_pose_brackets
                ],
                dtype=np.int64,
            )
            future_indices = np.asarray(
                [
                    (
                        bracket.before.stable_record_index,
                        bracket.after.stable_record_index,
                    )
                    for bracket in sample.future_pose_brackets
                ],
                dtype=np.int64,
            )
            future_weights = np.asarray(
                [bracket.weight for bracket in sample.future_pose_brackets],
                dtype=np.float32,
            )
            ego = sample.ego_bracket
            current = sample.current_pose_bracket
            return CacheSample(
                camera_rgb=camera_rgb,
                lidar_count=lidar_count,
                status=status,
                trajectory=trajectory,
                timestamp_ns=sample.front.header_stamp_ns,
                grid_receipt_ns=sample.grid_receipt_ns,
                sensor_source_header_stamp_ns=tuple(
                    record.header_stamp_ns for record in refs
                ),
                sensor_source_receipt_ns=tuple(
                    record.receipt_stamp_ns for record in refs
                ),
                sensor_source_record_index=tuple(
                    record.stable_record_index for record in refs
                ),
                side_camera_header_offset_ns=(
                    sample.left.header_stamp_ns - sample.front.header_stamp_ns,
                    sample.right.header_stamp_ns - sample.front.header_stamp_ns,
                ),
                sensor_receipt_offset_ns=(
                    sample.left.receipt_stamp_ns - sample.front.receipt_stamp_ns,
                    sample.right.receipt_stamp_ns - sample.front.receipt_stamp_ns,
                    sample.lidar.receipt_stamp_ns - sample.front.receipt_stamp_ns,
                ),
                trajectory_anchor_tf_header_ns=anchor.header_stamp_ns,
                trajectory_anchor_tf_receipt_ns=anchor.receipt_stamp_ns,
                trajectory_anchor_tf_record_index=anchor.stable_record_index,
                trajectory_anchor_tf_clock_epoch_index=anchor.clock_epoch_index,
                ego_bracket_clock_epoch_index=sample.ego_bracket_clock_epoch_index,
                ego_bracket_header_stamp_ns=(
                    ego.before.header_stamp_ns,
                    ego.after.header_stamp_ns,
                ),
                ego_bracket_receipt_ns=(
                    ego.before.receipt_stamp_ns,
                    ego.after.receipt_stamp_ns,
                ),
                ego_bracket_record_index=(
                    ego.before.stable_record_index,
                    ego.after.stable_record_index,
                ),
                ego_interp_weight=ego.weight,
                current_pose_bracket_header_stamp_ns=(
                    current.before.header_stamp_ns,
                    current.after.header_stamp_ns,
                ),
                current_pose_bracket_receipt_ns=(
                    current.before.receipt_stamp_ns,
                    current.after.receipt_stamp_ns,
                ),
                current_pose_bracket_record_index=(
                    current.before.stable_record_index,
                    current.after.stable_record_index,
                ),
                current_pose_interp_weight=current.weight,
                future_pose_bracket_header_stamp_ns=future_headers,
                future_pose_bracket_receipt_ns=future_receipts,
                future_pose_bracket_record_index=future_indices,
                future_pose_interp_weight=future_weights,
                command=command,
            )

        selected_argument = {
            topic: set(plan.selected_payload_indices[topic]) for topic in topics
        }
        self._active_event_iterator = iter(
            iter_selected_payloads(bag_path, selected_argument)
        )
        for event in self._active_event_iterator:
            if isinstance(event, SensorPayload):
                key = _key(event.ref)
            elif isinstance(event, SensorPayloadFailure):
                key = (event.topic, event.stable_record_index)
            else:
                raise IntegrityError("selected reader yielded an unknown event type")
            if key not in requested_refs:
                raise IntegrityError("selected reader yielded an unrequested event")
            if key in seen:
                raise IntegrityError("selected reader yielded a duplicate event")
            seen.add(key)
            if isinstance(event, SensorPayload):
                if event.ref != requested_refs[key]:
                    raise IntegrityError("pass-2 RecordRef differs from the planned ref")
                expected_kind = (
                    SensorPayloadKind.LIDAR_XYZI
                    if roles_by_topic[key[0]] == "lidar"
                    else SensorPayloadKind.CAMERA_JPEG
                )
                if event.kind is not expected_kind:
                    raise IntegrityError("pass-2 payload kind differs from the planned role")
            elif event.reason != "DECODE_ERROR":
                raise IntegrityError("selected failure reason is not DECODE_ERROR")
            outcomes[key] = event

            while next_sample < len(plan.samples) and all(
                key in outcomes for key in sample_keys[next_sample]
            ):
                sample = plan.samples[next_sample]
                result = resolve_sample(next_sample)
                for key in sample_keys[next_sample]:
                    consumers[key] -= 1
                    if consumers[key] == 0:
                        outcomes.pop(key, None)
                        decoded.pop(key, None)
                next_sample += 1
                if isinstance(result, _DropOutcome):
                    self._record_drop(
                        SampleDrop(
                            sample.grid_receipt_ns,
                            sample.front.header_stamp_ns,
                            result.reason,
                        )
                    )
                else:
                    self.valid_sample_count += 1
                    yield result

        self._active_event_iterator = None

        missing = set(requested_refs) - seen
        if missing:
            raise IntegrityError("selected reader omitted requested events")
        if next_sample != len(plan.samples):
            raise IntegrityError("not every planned sample became resolvable")
        if outcomes or decoded or any(count != 0 for count in consumers.values()):
            raise IntegrityError("selected payload reference counts did not drain")
        _verify_source_hash(bag_path, plan.source_bag_sha256)
        if self.valid_sample_count == 0:
            raise NoValidSampleError("bag has no valid materialized sample")


def materialize_samples(
    *, bag_path: Path, plan: BagSamplePlan, domain: ResolvedDomain
) -> MaterializationStream:
    if not isinstance(bag_path, Path):
        raise TypeError("bag_path must be a pathlib.Path")
    if not isinstance(plan, BagSamplePlan):
        raise TypeError("plan must be a BagSamplePlan")
    if not isinstance(domain, ResolvedDomain):
        raise TypeError("domain must be a ResolvedDomain")
    if bag_path != plan.bag_path:
        raise IntegrityError("materialization bag path differs from the plan binding")
    if domain.sha256 != plan.resolved_domain_sha256:
        raise IntegrityError("materialization domain differs from the plan binding")
    topics = _sensor_topics(domain)
    for sample in plan.samples:
        actual_topics = tuple(record.topic for record in _sample_refs(sample))
        if actual_topics != topics:
            raise IntegrityError(
                "planned sensor refs do not match front/left/right/LiDAR roles"
            )
    grids = tuple(sample.grid_receipt_ns for sample in plan.samples)
    if any(current <= previous for previous, current in zip(grids, grids[1:])):
        raise IntegrityError("planned samples must be in strict receipt-grid order")
    drop_grids = tuple(drop.grid_receipt_ns for drop in plan.drops)
    if any(current <= previous for previous, current in zip(drop_grids, drop_grids[1:])):
        raise IntegrityError("planner drops must be in strict receipt-grid order")
    if set(grids).intersection(drop_grids):
        raise IntegrityError("one receipt grid cannot be both planned and dropped")
    return MaterializationStream(bag_path=bag_path, plan=plan, domain=domain)
