from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, FrozenSet, Generic, Optional, Tuple, TypeVar

import numpy as np

from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError, IntegrityError, MoraiDDV2Error
from morai_ddv2.preprocess.timestamps import make_grid_ns
from morai_ddv2.ros1.bag_reader import causal_tf_interpolation_view
from morai_ddv2.ros1.messages import (
    BagDisposition,
    BagIndex,
    BagRejectReason,
    ClockEpochView,
    EgoRecord,
    RecordRef,
    TfPoseRecord,
)


T = TypeVar("T")
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _actual_int64(value: Any, name: str, *, nonnegative: bool = True) -> int:
    if type(value) is not int:
        raise TypeError("{} must be an actual Python int".format(name))
    if value < _INT64_MIN or value > _INT64_MAX:
        raise OverflowError("{} is outside signed int64".format(name))
    if nonnegative and value < 0:
        raise ValueError("{} must be nonnegative".format(name))
    return value


class SampleDropReason(str, Enum):
    DECODE_ERROR = "DECODE_ERROR"
    CAMERA_SYNC_MISS = "CAMERA_SYNC_MISS"
    LIDAR_SYNC_MISS = "LIDAR_SYNC_MISS"
    EGO_STATE_INTERPOLATION_MISS = "EGO_STATE_INTERPOLATION_MISS"
    POSE_INTERPOLATION_MISS = "POSE_INTERPOLATION_MISS"
    FUTURE_HORIZON_MISS = "FUTURE_HORIZON_MISS"
    NONFINITE_VALUE = "NONFINITE_VALUE"
    INVALID_COMMAND = "INVALID_COMMAND"
    EMPTY_LIDAR = "EMPTY_LIDAR"


class FrontTimestampOrderViolation(MoraiDDV2Error):
    rejection_code = BagRejectReason.FRONT_TIMESTAMP_ORDER_VIOLATION


@dataclass(frozen=True)
class SampleDrop:
    grid_receipt_ns: int
    timestamp_ns: Optional[int]
    reason: SampleDropReason

    def __post_init__(self) -> None:
        _actual_int64(self.grid_receipt_ns, "grid_receipt_ns")
        if self.timestamp_ns is not None:
            _actual_int64(self.timestamp_ns, "timestamp_ns")
        if not isinstance(self.reason, SampleDropReason):
            raise TypeError("reason must be a SampleDropReason")


@dataclass(frozen=True)
class PlannedBracket(Generic[T]):
    before: T
    after: T
    weight: float

    def __post_init__(self) -> None:
        if self.before is None or self.after is None:
            raise TypeError("planned bracket endpoints must not be None")
        if type(self.weight) is not float:
            raise TypeError("planned bracket weight must be an actual Python float")
        if not math.isfinite(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise ValueError("planned bracket weight must be finite and in [0,1]")
        if float(np.float32(self.weight)) != self.weight:
            raise ValueError("planned bracket weight must be float32-representable")
        if self.weight == 0.0 and bool(np.signbit(self.weight)):
            raise ValueError("zero planned bracket weight must be positive")


@dataclass(frozen=True)
class PlannedSample:
    grid_receipt_ns: int
    front: RecordRef
    left: RecordRef
    right: RecordRef
    lidar: RecordRef
    trajectory_anchor_tf: TfPoseRecord
    ego_bracket_clock_epoch_index: int
    ego_bracket: PlannedBracket[EgoRecord]
    current_pose_bracket: PlannedBracket[TfPoseRecord]
    future_pose_brackets: Tuple[PlannedBracket[TfPoseRecord], ...]

    def __post_init__(self) -> None:
        _actual_int64(self.grid_receipt_ns, "grid_receipt_ns")
        for name in ("front", "left", "right", "lidar"):
            if not isinstance(getattr(self, name), RecordRef):
                raise TypeError("{} must be a RecordRef".format(name))
        if not isinstance(self.trajectory_anchor_tf, TfPoseRecord):
            raise TypeError("trajectory_anchor_tf must be a TfPoseRecord")
        _actual_int64(
            self.ego_bracket_clock_epoch_index,
            "ego_bracket_clock_epoch_index",
        )
        if not isinstance(self.ego_bracket, PlannedBracket) or any(
            not isinstance(record, EgoRecord)
            for record in (self.ego_bracket.before, self.ego_bracket.after)
        ):
            raise TypeError("ego_bracket must contain EgoRecord endpoints")
        if not isinstance(self.current_pose_bracket, PlannedBracket) or any(
            not isinstance(record, TfPoseRecord)
            for record in (
                self.current_pose_bracket.before,
                self.current_pose_bracket.after,
            )
        ):
            raise TypeError("current_pose_bracket must contain TfPoseRecord endpoints")
        if type(self.future_pose_brackets) is not tuple or len(self.future_pose_brackets) != 8:
            raise ValueError("future_pose_brackets must contain exactly eight brackets")
        if any(
            not isinstance(bracket, PlannedBracket)
            or not isinstance(bracket.before, TfPoseRecord)
            or not isinstance(bracket.after, TfPoseRecord)
            for bracket in self.future_pose_brackets
        ):
            raise TypeError("future_pose_brackets must contain TF brackets")


@dataclass(frozen=True)
class BagSamplePlan:
    bag_path: Path
    source_bag_sha256: str
    resolved_domain_sha256: str
    samples: Tuple[PlannedSample, ...]
    drops: Tuple[SampleDrop, ...]
    selected_payload_indices: Mapping[str, FrozenSet[int]]

    def __post_init__(self) -> None:
        if not isinstance(self.bag_path, Path):
            raise TypeError("bag_path must be a pathlib.Path")
        for name in ("source_bag_sha256", "resolved_domain_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError("{} must be a lowercase SHA-256".format(name))
        if type(self.samples) is not tuple or any(
            not isinstance(item, PlannedSample) for item in self.samples
        ):
            raise TypeError("samples must be a tuple of PlannedSample values")
        if type(self.drops) is not tuple or any(
            not isinstance(item, SampleDrop) for item in self.drops
        ):
            raise TypeError("drops must be a tuple of SampleDrop values")
        sample_grids = tuple(sample.grid_receipt_ns for sample in self.samples)
        drop_grids = tuple(drop.grid_receipt_ns for drop in self.drops)
        if any(current <= previous for previous, current in zip(sample_grids, sample_grids[1:])):
            raise ValueError("samples must be in strict receipt-grid order")
        if any(current <= previous for previous, current in zip(drop_grids, drop_grids[1:])):
            raise ValueError("drops must be in strict receipt-grid order")
        if set(sample_grids).intersection(drop_grids):
            raise ValueError("one receipt grid cannot be both sampled and dropped")
        front_headers = tuple(sample.front.header_stamp_ns for sample in self.samples)
        if any(
            current <= previous
            for previous, current in zip(front_headers, front_headers[1:])
        ):
            raise ValueError("planned front Headers must be strictly increasing")
        front_indices = tuple(
            sample.front.stable_record_index for sample in self.samples
        )
        if len(set(front_indices)) != len(front_indices):
            raise ValueError("planned front stable record indices must not be reused")
        if not isinstance(self.selected_payload_indices, Mapping):
            raise TypeError("selected_payload_indices must be a mapping")
        selected = {}
        for topic, indices in self.selected_payload_indices.items():
            if not isinstance(topic, str) or not topic:
                raise TypeError("selected payload topics must be nonempty strings")
            if type(indices) is not frozenset or any(type(item) is not int or item < 0 for item in indices):
                raise TypeError("selected payload indices must be nonnegative Python-int frozensets")
            selected[topic] = indices
        if len(selected) != 4:
            raise ValueError("selected_payload_indices must name four physical topics")
        expected = {topic: set() for topic in selected}
        for sample in self.samples:
            refs = (sample.front, sample.left, sample.right, sample.lidar)
            if len({record.topic for record in refs}) != 4:
                raise ValueError("one planned sample must use four physical topics")
            for record in refs:
                if record.topic not in expected:
                    raise ValueError("planned ref topic is missing from selected union")
                expected[record.topic].add(record.stable_record_index)
        if any(set(selected[topic]) != expected[topic] for topic in selected):
            raise ValueError("selected payload union does not exactly match samples")
        object.__setattr__(self, "selected_payload_indices", MappingProxyType(selected))


@dataclass(frozen=True)
class _SamplingPolicy:
    grid_period_ns: int
    front_tolerance_ns: int
    side_header_tolerance_ns: int
    side_receipt_tolerance_ns: int
    lidar_tolerance_ns: int
    tf_anchor_tolerance_ns: int
    ego_max_bracket_ns: int
    future_max_bracket_ns: int
    future_offsets_ns: Tuple[int, ...]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("{} must be a mapping".format(path))
    return value


def _exact_keys(value: Mapping[str, Any], keys: set, path: str) -> None:
    if set(value) != keys:
        raise ConfigurationError("{} keys do not match the v1 contract".format(path))


def _expect(value: Any, expected: Any, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ConfigurationError("{} does not match the v1 contract".format(path))


def _expect_sequence(value: Any, expected: Tuple[Any, ...], path: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("{} must be a sequence".format(path))
    actual = tuple(value)
    if len(actual) != len(expected) or any(
        type(left) is not type(right) or left != right
        for left, right in zip(actual, expected)
    ):
        raise ConfigurationError("{} does not match the v1 contract".format(path))


def _sampling_policy(domain: ResolvedDomain) -> _SamplingPolicy:
    if not isinstance(domain, ResolvedDomain):
        raise TypeError("domain must be a ResolvedDomain")
    document = _mapping(domain.document, "resolved domain")
    sampling = _mapping(document.get("sampling"), "sampling")
    _exact_keys(
        sampling,
        {"sample_rate_hz", "grid_period_ns", "clock_policy", "drop_on_required_sensor_miss"},
        "sampling",
    )
    _expect(sampling["sample_rate_hz"], 2, "sampling.sample_rate_hz")
    _expect(sampling["grid_period_ns"], 500_000_000, "sampling.grid_period_ns")
    _expect(sampling["drop_on_required_sensor_miss"], True, "sampling.drop_on_required_sensor_miss")

    clock = _mapping(sampling["clock_policy"], "sampling.clock_policy")
    _exact_keys(
        clock,
        {
            "version", "receipt_clock_semantics", "grid", "front", "side_camera",
            "lidar", "trajectory_anchor_tf", "ego_state", "future_pose",
            "tf_clock_epoch", "tf_ego_bag_gate", "tolerance_boundary_inclusive",
            "interpolation_exact_match_weight",
        },
        "sampling.clock_policy",
    )
    _expect(clock["version"], 1, "sampling.clock_policy.version")
    _expect(clock["receipt_clock_semantics"], "recorder_observation_order_not_acquisition", "sampling.clock_policy.receipt_clock_semantics")
    _expect(clock["tolerance_boundary_inclusive"], True, "sampling.clock_policy.tolerance_boundary_inclusive")
    _expect(clock["interpolation_exact_match_weight"], 0, "sampling.clock_policy.interpolation_exact_match_weight")

    grid = _mapping(clock["grid"], "sampling.clock_policy.grid")
    _exact_keys(grid, {"axis", "origin"}, "sampling.clock_policy.grid")
    _expect(grid["axis"], "receipt_stamp_ns", "sampling.clock_policy.grid.axis")
    _expect(grid["origin"], "ceil_first_valid_front_receipt_to_period", "sampling.clock_policy.grid.origin")

    front = _mapping(clock["front"], "sampling.clock_policy.front")
    _exact_keys(front, {"selection_axis", "tolerance_ns", "tie_break_order", "selected_header_identity"}, "sampling.clock_policy.front")
    _expect(front["selection_axis"], "receipt_stamp_ns", "sampling.clock_policy.front.selection_axis")
    _expect(front["tolerance_ns"], 50_000_000, "sampling.clock_policy.front.tolerance_ns")
    _expect_sequence(front["tie_break_order"], ("absolute_receipt_offset_ns", "receipt_stamp_ns", "stable_record_index"), "sampling.clock_policy.front.tie_break_order")
    identity = _mapping(front["selected_header_identity"], "sampling.clock_policy.front.selected_header_identity")
    _exact_keys(identity, {"receipt_grid_order", "unique", "stable_record_reuse_allowed", "violation_bag_reject_reason"}, "sampling.clock_policy.front.selected_header_identity")
    _expect(identity["receipt_grid_order"], "strictly_increasing", "sampling.clock_policy.front.selected_header_identity.receipt_grid_order")
    _expect(identity["unique"], True, "sampling.clock_policy.front.selected_header_identity.unique")
    _expect(identity["stable_record_reuse_allowed"], False, "sampling.clock_policy.front.selected_header_identity.stable_record_reuse_allowed")
    _expect(identity["violation_bag_reject_reason"], "FRONT_TIMESTAMP_ORDER_VIOLATION", "sampling.clock_policy.front.selected_header_identity.violation_bag_reject_reason")

    side = _mapping(clock["side_camera"], "sampling.clock_policy.side_camera")
    _exact_keys(side, {"candidate_filters", "selection_axis", "tie_break_order"}, "sampling.clock_policy.side_camera")
    filters = side["candidate_filters"]
    if isinstance(filters, (str, bytes)) or not isinstance(filters, Sequence) or len(filters) != 2:
        raise ConfigurationError("sampling.clock_policy.side_camera.candidate_filters must contain two filters")
    header_filter = _mapping(filters[0], "sampling.clock_policy.side_camera.candidate_filters[0]")
    receipt_filter = _mapping(filters[1], "sampling.clock_policy.side_camera.candidate_filters[1]")
    filter_keys = {"axis", "relative_to", "absolute_tolerance_ns"}
    _exact_keys(header_filter, filter_keys, "sampling.clock_policy.side_camera.candidate_filters[0]")
    _exact_keys(receipt_filter, filter_keys, "sampling.clock_policy.side_camera.candidate_filters[1]")
    _expect(header_filter["axis"], "header_stamp_ns", "sampling.clock_policy.side_camera.candidate_filters[0].axis")
    _expect(header_filter["relative_to"], "selected_front_header_stamp_ns", "sampling.clock_policy.side_camera.candidate_filters[0].relative_to")
    _expect(header_filter["absolute_tolerance_ns"], 30_000_000, "sampling.clock_policy.side_camera.candidate_filters[0].absolute_tolerance_ns")
    _expect(receipt_filter["axis"], "receipt_stamp_ns", "sampling.clock_policy.side_camera.candidate_filters[1].axis")
    _expect(receipt_filter["relative_to"], "selected_front_receipt_stamp_ns", "sampling.clock_policy.side_camera.candidate_filters[1].relative_to")
    _expect(receipt_filter["absolute_tolerance_ns"], 100_000_000, "sampling.clock_policy.side_camera.candidate_filters[1].absolute_tolerance_ns")
    _expect(side["selection_axis"], "header_stamp_ns", "sampling.clock_policy.side_camera.selection_axis")
    _expect_sequence(side["tie_break_order"], ("absolute_header_offset_ns", "header_stamp_ns", "stable_record_index"), "sampling.clock_policy.side_camera.tie_break_order")

    lidar = _mapping(clock["lidar"], "sampling.clock_policy.lidar")
    _exact_keys(lidar, {"selection_axis", "tolerance_ns", "tie_break_order"}, "sampling.clock_policy.lidar")
    _expect(lidar["selection_axis"], "receipt_stamp_ns", "sampling.clock_policy.lidar.selection_axis")
    _expect(lidar["tolerance_ns"], 45_000_000, "sampling.clock_policy.lidar.tolerance_ns")
    _expect_sequence(lidar["tie_break_order"], ("absolute_receipt_offset_ns", "receipt_stamp_ns", "stable_record_index"), "sampling.clock_policy.lidar.tie_break_order")

    anchor = _mapping(clock["trajectory_anchor_tf"], "sampling.clock_policy.trajectory_anchor_tf")
    _exact_keys(anchor, {"source_view", "selection_axis", "tolerance_ns", "tie_break_order", "current_pose", "preserve_original_stable_record_index", "fallback_to_second_candidate"}, "sampling.clock_policy.trajectory_anchor_tf")
    for key, expected in (
        ("source_view", "raw_map_to_base_link_records"),
        ("selection_axis", "receipt_stamp_ns"),
        ("current_pose", "exact_selected_raw_record"),
        ("preserve_original_stable_record_index", True),
        ("fallback_to_second_candidate", False),
    ):
        _expect(anchor[key], expected, "sampling.clock_policy.trajectory_anchor_tf.{}".format(key))
    _expect(anchor["tolerance_ns"], 50_000_000, "sampling.clock_policy.trajectory_anchor_tf.tolerance_ns")
    _expect_sequence(anchor["tie_break_order"], ("absolute_receipt_offset_ns", "receipt_stamp_ns", "stable_record_index"), "sampling.clock_policy.trajectory_anchor_tf.tie_break_order")

    ego = _mapping(clock["ego_state"], "sampling.clock_policy.ego_state")
    _exact_keys(ego, {"interpolation_axis", "target", "max_bracket_ns", "clock_epoch"}, "sampling.clock_policy.ego_state")
    _expect(ego["interpolation_axis"], "header_stamp_ns", "sampling.clock_policy.ego_state.interpolation_axis")
    _expect(ego["target"], "trajectory_anchor_tf_header_ns", "sampling.clock_policy.ego_state.target")
    _expect(ego["max_bracket_ns"], 100_000_000, "sampling.clock_policy.ego_state.max_bracket_ns")
    ego_epoch = _mapping(ego["clock_epoch"], "sampling.clock_policy.ego_state.clock_epoch")
    _exact_keys(ego_epoch, {"index_origin", "source_view", "ordering", "new_epoch_condition", "selector_relative_to", "selector_axis", "selector_tolerance", "selector_tie_break_order", "interpolation_view", "equal_header_duplicate_policy", "fallback_to_other_epoch"}, "sampling.clock_policy.ego_state.clock_epoch")
    expected_ego_epoch = {
        "index_origin": 0,
        "source_view": "raw_ego_records",
        "new_epoch_condition": "header_stamp_ns_strictly_decreases",
        "selector_relative_to": "trajectory_anchor_tf_receipt_ns",
        "selector_axis": "receipt_stamp_ns",
        "selector_tolerance": "none",
        "interpolation_view": "selected_epoch_collapsed_header_view",
        "equal_header_duplicate_policy": "collapse_equivalent_earliest_in_epoch_reject_conflict",
        "fallback_to_other_epoch": False,
    }
    for key, expected in expected_ego_epoch.items():
        _expect(ego_epoch[key], expected, "sampling.clock_policy.ego_state.clock_epoch.{}".format(key))
    _expect_sequence(ego_epoch["ordering"], ("receipt_stamp_ns", "stable_record_index"), "sampling.clock_policy.ego_state.clock_epoch.ordering")
    _expect_sequence(ego_epoch["selector_tie_break_order"], ("absolute_receipt_offset_ns", "receipt_stamp_ns", "stable_record_index"), "sampling.clock_policy.ego_state.clock_epoch.selector_tie_break_order")

    future = _mapping(clock["future_pose"], "sampling.clock_policy.future_pose")
    _exact_keys(future, {"interpolation_axis", "target_origin", "max_bracket_ns", "source_view", "equal_header_duplicate_policy", "extrapolation_allowed", "fallback_to_other_epoch"}, "sampling.clock_policy.future_pose")
    expected_future = {
        "interpolation_axis": "header_stamp_ns",
        "target_origin": "trajectory_anchor_tf_header_ns",
        "max_bracket_ns": 100_000_000,
        "source_view": "selected_raw_anchor_same_epoch_causal_suffix",
        "equal_header_duplicate_policy": "collapse_equivalent_earliest_at_or_after_anchor_reject_conflict",
        "extrapolation_allowed": False,
        "fallback_to_other_epoch": False,
    }
    for key, expected in expected_future.items():
        _expect(future[key], expected, "sampling.clock_policy.future_pose.{}".format(key))

    tf_epoch = _mapping(clock["tf_clock_epoch"], "sampling.clock_policy.tf_clock_epoch")
    _exact_keys(tf_epoch, {"index_origin", "source_view", "ordering", "monotonic_header_rule", "new_epoch_condition", "within_epoch_interpolation_view", "equal_header_duplicate_policy", "same_header_across_epochs"}, "sampling.clock_policy.tf_clock_epoch")
    expected_tf_epoch = {
        "index_origin": 0,
        "source_view": "raw_map_to_base_link_records",
        "monotonic_header_rule": "nondecreasing",
        "new_epoch_condition": "header_stamp_ns_strictly_decreases",
        "within_epoch_interpolation_view": "collapsed_header_view",
        "equal_header_duplicate_policy": "collapse_equivalent_earliest_in_view_reject_conflict",
        "same_header_across_epochs": "allowed_epoch_index_disambiguates",
    }
    for key, expected in expected_tf_epoch.items():
        _expect(tf_epoch[key], expected, "sampling.clock_policy.tf_clock_epoch.{}".format(key))
    _expect_sequence(tf_epoch["ordering"], ("receipt_stamp_ns", "stable_record_index"), "sampling.clock_policy.tf_clock_epoch.ordering")

    gate = _mapping(clock["tf_ego_bag_gate"], "sampling.clock_policy.tf_ego_bag_gate")
    _exact_keys(gate, {"axis", "scope", "ego_source_view", "tf_epoch_selector", "interpolation_view", "max_bracket_ns", "fallback_to_other_epoch", "excluded_case_counters"}, "sampling.clock_policy.tf_ego_bag_gate")
    expected_gate = {
        "axis": "native_header_stamp_ns",
        "scope": "full_stream",
        "ego_source_view": "raw_ego_records",
        "interpolation_view": "selected_tf_epoch_collapsed_header_view",
        "max_bracket_ns": 100_000_000,
        "fallback_to_other_epoch": False,
    }
    for key, expected in expected_gate.items():
        _expect(gate[key], expected, "sampling.clock_policy.tf_ego_bag_gate.{}".format(key))
    _expect_sequence(gate["excluded_case_counters"], ("no_epoch_local_bracket", "ambiguous_epoch_selection", "cross_epoch_bracket_prevented"), "sampling.clock_policy.tf_ego_bag_gate.excluded_case_counters")
    selector = _mapping(gate["tf_epoch_selector"], "sampling.clock_policy.tf_ego_bag_gate.tf_epoch_selector")
    _exact_keys(selector, {"source_view", "relative_to", "selection_axis", "tolerance", "tie_break_order"}, "sampling.clock_policy.tf_ego_bag_gate.tf_epoch_selector")
    for key, expected in {
        "source_view": "raw_map_to_base_link_records",
        "relative_to": "ego_receipt_stamp_ns",
        "selection_axis": "receipt_stamp_ns",
        "tolerance": "none",
    }.items():
        _expect(selector[key], expected, "sampling.clock_policy.tf_ego_bag_gate.tf_epoch_selector.{}".format(key))
    _expect_sequence(selector["tie_break_order"], ("absolute_receipt_offset_ns", "receipt_stamp_ns", "stable_record_index"), "sampling.clock_policy.tf_ego_bag_gate.tf_epoch_selector.tie_break_order")

    trajectory = _mapping(document.get("trajectory"), "trajectory")
    offsets = trajectory.get("sampling_offsets_ns")
    expected_offsets = tuple(500_000_000 * index for index in range(1, 9))
    _expect_sequence(offsets, expected_offsets, "trajectory.sampling_offsets_ns")
    return _SamplingPolicy(
        grid_period_ns=sampling["grid_period_ns"],
        front_tolerance_ns=front["tolerance_ns"],
        side_header_tolerance_ns=header_filter["absolute_tolerance_ns"],
        side_receipt_tolerance_ns=receipt_filter["absolute_tolerance_ns"],
        lidar_tolerance_ns=lidar["tolerance_ns"],
        tf_anchor_tolerance_ns=anchor["tolerance_ns"],
        ego_max_bracket_ns=ego["max_bracket_ns"],
        future_max_bracket_ns=future["max_bracket_ns"],
        future_offsets_ns=expected_offsets,
    )


def _epoch_view(views: Tuple[ClockEpochView[T], ...], index: int, name: str) -> ClockEpochView[T]:
    if index < 0 or index >= len(views) or views[index].index != index:
        raise IntegrityError("{} epoch view is missing or non-dense".format(name))
    return views[index]


def _validate_epoch_handoff(flat: Tuple[T, ...], views: Tuple[ClockEpochView[T], ...], name: str) -> None:
    if tuple(view.index for view in views) != tuple(range(len(views))):
        raise IntegrityError("{} epoch indices must be dense and zero-based".format(name))
    flattened = tuple(record for view in views for record in view.raw_records)
    if len(flattened) != len(flat) or any(left is not right for left, right in zip(flat, flattened)):
        raise IntegrityError("{} raw records and epoch views must identity-match".format(name))
    for record in flat:
        try:
            _actual_int64(record.header_stamp_ns, "{}.header_stamp_ns".format(name))
            _actual_int64(record.receipt_stamp_ns, "{}.receipt_stamp_ns".format(name))
            _actual_int64(
                record.stable_record_index,
                "{}.stable_record_index".format(name),
            )
            _actual_int64(
                record.clock_epoch_index,
                "{}.clock_epoch_index".format(name),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise IntegrityError(
                "{} raw record integer provenance is invalid".format(name)
            ) from error
    if len({record.stable_record_index for record in flat}) != len(flat):
        raise IntegrityError("{} stable record indices must be unique".format(name))
    global_receipt_keys = tuple(
        (record.receipt_stamp_ns, record.stable_record_index) for record in flat
    )
    if any(
        current <= previous
        for previous, current in zip(
            global_receipt_keys, global_receipt_keys[1:]
        )
    ):
        raise IntegrityError("{} raw records are not globally receipt ordered".format(name))
    for previous_view, current_view in zip(views, views[1:]):
        if (
            not previous_view.raw_records
            or not current_view.raw_records
            or current_view.raw_records[0].header_stamp_ns
            >= previous_view.raw_records[-1].header_stamp_ns
        ):
            raise IntegrityError(
                "{} epoch boundaries must occur exactly on strict Header decrease".format(
                    name
                )
            )
    for view in views:
        if any(record.clock_epoch_index != view.index for record in view.raw_records):
            raise IntegrityError("{} epoch member tag mismatch".format(name))
        receipt_keys = tuple((record.receipt_stamp_ns, record.stable_record_index) for record in view.raw_records)
        if any(current <= previous for previous, current in zip(receipt_keys, receipt_keys[1:])):
            raise IntegrityError("{} raw records are not in receipt order".format(name))
        headers = tuple(record.header_stamp_ns for record in view.raw_records)
        if any(current < previous for previous, current in zip(headers, headers[1:])):
            raise IntegrityError("{} Header reset occurs inside an epoch".format(name))
        position = 0
        interpolation_headers = []
        for selected in view.interpolation_records:
            while position < len(view.raw_records) and view.raw_records[position] is not selected:
                position += 1
            if position == len(view.raw_records):
                raise IntegrityError("{} interpolation view is not an identity subsequence".format(name))
            interpolation_headers.append(selected.header_stamp_ns)
            position += 1
        if any(current <= previous for previous, current in zip(interpolation_headers, interpolation_headers[1:])):
            raise IntegrityError("{} interpolation Headers must strictly increase".format(name))
        expected_interpolation = []
        for record in view.raw_records:
            if (
                not expected_interpolation
                or expected_interpolation[-1].header_stamp_ns
                != record.header_stamp_ns
            ):
                expected_interpolation.append(record)
        if len(view.interpolation_records) != len(expected_interpolation) or any(
            actual is not expected
            for actual, expected in zip(
                view.interpolation_records, expected_interpolation
            )
        ):
            raise IntegrityError(
                "{} interpolation view must be the complete earliest-per-Header collapse".format(
                    name
                )
            )


def _bracket(records: Sequence[T], target_ns: int, max_gap_ns: int) -> Optional[PlannedBracket[T]]:
    if not records:
        return None
    stamps = tuple(record.header_stamp_ns for record in records)
    position = bisect_left(stamps, target_ns)
    if position < len(records) and stamps[position] == target_ns:
        record = records[position]
        return PlannedBracket(record, record, 0.0)
    if position == 0 or position == len(records):
        return None
    before = records[position - 1]
    after = records[position]
    gap = after.header_stamp_ns - before.header_stamp_ns
    if gap <= 0 or gap > max_gap_ns:
        return None
    weight = float(np.float32((target_ns - before.header_stamp_ns) / float(gap)))
    if not math.isfinite(weight):
        raise IntegrityError("interpolation weight is non-finite")
    return PlannedBracket(before, after, weight)


def _ordered_receipt_view(records: Sequence[T]) -> Tuple[Tuple[int, ...], Tuple[T, ...]]:
    selected = []
    for record in records:
        if not selected or selected[-1].receipt_stamp_ns != record.receipt_stamp_ns:
            selected.append(record)
    return (
        tuple(record.receipt_stamp_ns for record in selected),
        tuple(selected),
    )


def _nearest_ordered_receipt(
    receipt_stamps: Tuple[int, ...],
    records: Tuple[T, ...],
    target_ns: int,
    tolerance_ns: Optional[int],
) -> Optional[T]:
    if not records:
        return None
    position = bisect_left(receipt_stamps, target_ns)
    candidates = []
    if position < len(records):
        candidates.append(records[position])
    if position > 0:
        candidates.append(records[position - 1])
    selected = min(
        candidates,
        key=lambda record: (
            abs(record.receipt_stamp_ns - target_ns),
            record.receipt_stamp_ns,
            record.stable_record_index,
        ),
    )
    if tolerance_ns is not None and abs(selected.receipt_stamp_ns - target_ns) > tolerance_ns:
        return None
    return selected


def _validate_gate(bag: BagIndex, max_gap_ns: int) -> None:
    if bag.inventory.disposition is not BagDisposition.ELIGIBLE:
        raise IntegrityError("sample planning requires an ELIGIBLE Task 8 bag index")
    report = bag.inventory.tf_ego_diagnostic
    if report is None:
        raise IntegrityError("eligible bag is missing epoch-safe TF/Ego diagnostics")
    count_fields = (
        ("raw_ego_record_count", report.raw_ego_record_count),
        ("epoch_local_comparable_count", report.epoch_local_comparable_count),
        ("no_epoch_local_bracket_count", report.no_epoch_local_bracket_count),
        (
            "ambiguous_epoch_selection_count",
            report.ambiguous_epoch_selection_count,
        ),
        (
            "cross_epoch_bracket_prevented_count",
            report.cross_epoch_bracket_prevented_count,
        ),
        ("metric.comparable_count", report.metric.comparable_count),
        ("metric.excluded_count", report.metric.excluded_count),
    )
    try:
        for name, value in count_fields:
            _actual_int64(value, "tf_ego_diagnostic.{}".format(name))
    except (TypeError, ValueError, OverflowError) as error:
        raise IntegrityError("epoch-safe TF/Ego diagnostic counts are invalid") from error
    excluded_count = (
        report.no_epoch_local_bracket_count
        + report.ambiguous_epoch_selection_count
        + report.cross_epoch_bracket_prevented_count
    )
    actual_reason_counts = (
        sum(item.exclusion_reason is None for item in report.provenance),
        sum(
            item.exclusion_reason == "NO_EPOCH_LOCAL_BRACKET"
            for item in report.provenance
        ),
        sum(
            item.exclusion_reason == "AMBIGUOUS_EPOCH_SELECTION"
            for item in report.provenance
        ),
        sum(
            item.exclusion_reason == "CROSS_EPOCH_BRACKET_PREVENTED"
            for item in report.provenance
        ),
    )
    if (
        report.raw_ego_record_count != len(bag.raw_ego)
        or len(report.provenance) != len(bag.raw_ego)
        or report.epoch_local_comparable_count != report.metric.comparable_count
        or report.metric.excluded_count != excluded_count
        or report.epoch_local_comparable_count + excluded_count
        != report.raw_ego_record_count
        or actual_reason_counts
        != (
            report.epoch_local_comparable_count,
            report.no_epoch_local_bracket_count,
            report.ambiguous_epoch_selection_count,
            report.cross_epoch_bracket_prevented_count,
        )
        or report.epoch_local_comparable_count < 10
    ):
        raise IntegrityError("epoch-safe TF/Ego diagnostic counts are inconsistent")
    pose_receipts, pose_selector = _ordered_receipt_view(bag.raw_poses)
    for ego, provenance in zip(bag.raw_ego, report.provenance):
        if (
            provenance.ego_header_stamp_ns != ego.header_stamp_ns
            or provenance.ego_receipt_stamp_ns != ego.receipt_stamp_ns
            or provenance.ego_stable_record_index != ego.stable_record_index
        ):
            raise IntegrityError("TF/Ego diagnostic provenance does not match raw Ego identity")
        if not bag.raw_poses:
            raise IntegrityError("TF/Ego diagnostic references an empty raw TF stream")
        anchor = _nearest_ordered_receipt(
            pose_receipts, pose_selector, ego.receipt_stamp_ns, None
        )
        if anchor is None:
            raise IntegrityError("TF/Ego diagnostic cannot select a raw TF anchor")
        if (
            provenance.selected_tf_clock_epoch_index != anchor.clock_epoch_index
            or provenance.selected_tf_anchor_record_index != anchor.stable_record_index
        ):
            raise IntegrityError("TF/Ego diagnostic raw TF selection is inconsistent")
        view = _epoch_view(bag.tf_clock_epochs, anchor.clock_epoch_index, "TF")
        bracket = _bracket(view.interpolation_records, ego.header_stamp_ns, max_gap_ns)
        if provenance.exclusion_reason is None:
            if bracket is None:
                raise IntegrityError("comparable TF/Ego provenance has no epoch-local bracket")
            expected = (
                (bracket.before.header_stamp_ns, bracket.after.header_stamp_ns),
                (bracket.before.receipt_stamp_ns, bracket.after.receipt_stamp_ns),
                (bracket.before.stable_record_index, bracket.after.stable_record_index),
            )
            actual = (
                provenance.tf_bracket_header_stamp_ns,
                provenance.tf_bracket_receipt_ns,
                provenance.tf_bracket_record_index,
            )
            if actual != expected:
                raise IntegrityError("TF/Ego diagnostic bracket provenance is inconsistent")
        elif bracket is not None:
            raise IntegrityError("excluded TF/Ego provenance has a valid local bracket")


def _sensor_topics(domain: ResolvedDomain) -> Tuple[str, str, str, str]:
    sensors = _mapping(domain.document.get("sensors"), "sensors")
    cameras = sensors.get("cameras")
    if isinstance(cameras, (str, bytes)) or not isinstance(cameras, Sequence) or len(cameras) != 3:
        raise ConfigurationError("sensors.cameras must contain three cameras")
    by_role = {}
    for index, camera_value in enumerate(cameras):
        camera = _mapping(camera_value, "sensors.cameras[{}]".format(index))
        role = camera.get("role")
        topic = camera.get("topic")
        if role in by_role or role not in ("front", "left", "right") or not isinstance(topic, str) or not topic:
            raise ConfigurationError("camera roles/topics do not match the domain contract")
        by_role[role] = topic
    lidar = _mapping(sensors.get("lidar"), "sensors.lidar")
    lidar_topic = lidar.get("topic")
    if set(by_role) != {"front", "left", "right"} or not isinstance(lidar_topic, str) or not lidar_topic:
        raise ConfigurationError("sensor topics do not match the domain contract")
    topics = (by_role["front"], by_role["left"], by_role["right"], lidar_topic)
    if len(set(topics)) != 4:
        raise ConfigurationError("physical sensor topics must be unique")
    return topics


def _nearest_receipt(records: Sequence[T], target_ns: int, tolerance_ns: Optional[int]) -> Optional[T]:
    if not records:
        return None
    selected = min(
        records,
        key=lambda record: (
            abs(record.receipt_stamp_ns - target_ns),
            record.receipt_stamp_ns,
            record.stable_record_index,
        ),
    )
    if tolerance_ns is not None and abs(selected.receipt_stamp_ns - target_ns) > tolerance_ns:
        return None
    return selected


def _nearest_side(records: Sequence[RecordRef], front: RecordRef, policy: _SamplingPolicy) -> Optional[RecordRef]:
    candidates = tuple(
        record
        for record in records
        if abs(record.header_stamp_ns - front.header_stamp_ns) <= policy.side_header_tolerance_ns
        and abs(record.receipt_stamp_ns - front.receipt_stamp_ns) <= policy.side_receipt_tolerance_ns
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda record: (
            abs(record.header_stamp_ns - front.header_stamp_ns),
            record.header_stamp_ns,
            record.stable_record_index,
        ),
    )


def build_sample_plan(bag: BagIndex, domain: ResolvedDomain) -> BagSamplePlan:
    if not isinstance(bag, BagIndex):
        raise TypeError("bag must be a BagIndex")
    policy = _sampling_policy(domain)
    topics = _sensor_topics(domain)
    for name, expected_topic in zip(("front", "left", "right", "lidar"), topics):
        records = getattr(bag, name)
        if any(record.topic != expected_topic for record in records):
            raise IntegrityError("{} index contains the wrong physical topic".format(name))
        for record in records:
            try:
                _actual_int64(
                    record.header_stamp_ns, "{}.header_stamp_ns".format(name)
                )
                _actual_int64(
                    record.receipt_stamp_ns, "{}.receipt_stamp_ns".format(name)
                )
                _actual_int64(
                    record.stable_record_index,
                    "{}.stable_record_index".format(name),
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise IntegrityError(
                    "{} sensor record integer provenance is invalid".format(name)
                ) from error
        if len({record.stable_record_index for record in records}) != len(records):
            raise IntegrityError("{} stable record indices must be unique".format(name))
    _validate_epoch_handoff(bag.raw_ego, bag.ego_clock_epochs, "Ego")
    _validate_epoch_handoff(bag.raw_poses, bag.tf_clock_epochs, "TF")
    _validate_gate(bag, policy.ego_max_bracket_ns)

    selected = {topic: set() for topic in topics}
    if not bag.front:
        return BagSamplePlan(
            bag.path,
            bag.source_bag_sha256,
            domain.sha256,
            (),
            (),
            {topic: frozenset() for topic in topics},
        )
    front_receipts = tuple(record.receipt_stamp_ns for record in bag.front)
    grids = make_grid_ns(min(front_receipts), max(front_receipts), policy.grid_period_ns)
    grid_fronts = tuple(
        (
            grid,
            _nearest_receipt(bag.front, grid, policy.front_tolerance_ns),
        )
        for grid in grids
    )

    previous_header = None
    used_front_indices = set()
    for _, front in grid_fronts:
        if front is None:
            continue
        if (
            previous_header is not None
            and front.header_stamp_ns <= previous_header
        ) or front.stable_record_index in used_front_indices:
            raise FrontTimestampOrderViolation(
                "selected front Header identity is not strictly increasing and unique"
            )
        previous_header = front.header_stamp_ns
        used_front_indices.add(front.stable_record_index)

    samples = []
    drops = []
    pose_receipts, pose_selector = _ordered_receipt_view(bag.raw_poses)
    ego_receipts, ego_selector = _ordered_receipt_view(bag.raw_ego)
    for grid, front in grid_fronts:
        if front is None:
            drops.append(SampleDrop(grid, None, SampleDropReason.CAMERA_SYNC_MISS))
            continue
        timestamp = front.header_stamp_ns
        left = _nearest_side(bag.left, front, policy)
        if left is None:
            drops.append(SampleDrop(grid, timestamp, SampleDropReason.CAMERA_SYNC_MISS))
            continue
        right = _nearest_side(bag.right, front, policy)
        if right is None:
            drops.append(SampleDrop(grid, timestamp, SampleDropReason.CAMERA_SYNC_MISS))
            continue
        lidar = _nearest_receipt(bag.lidar, front.receipt_stamp_ns, policy.lidar_tolerance_ns)
        if lidar is None:
            drops.append(SampleDrop(grid, timestamp, SampleDropReason.LIDAR_SYNC_MISS))
            continue
        anchor = _nearest_ordered_receipt(
            pose_receipts,
            pose_selector,
            front.receipt_stamp_ns,
            policy.tf_anchor_tolerance_ns,
        )
        if anchor is None:
            drops.append(SampleDrop(grid, timestamp, SampleDropReason.POSE_INTERPOLATION_MISS))
            continue

        nearest_ego = _nearest_ordered_receipt(
            ego_receipts, ego_selector, anchor.receipt_stamp_ns, None
        )
        if nearest_ego is None:
            drops.append(SampleDrop(grid, timestamp, SampleDropReason.EGO_STATE_INTERPOLATION_MISS))
            continue
        ego_view = _epoch_view(bag.ego_clock_epochs, nearest_ego.clock_epoch_index, "Ego")
        if sum(record is nearest_ego for record in ego_view.raw_records) != 1:
            raise IntegrityError("selected raw Ego record is not a unique epoch member")
        ego_bracket = _bracket(ego_view.interpolation_records, anchor.header_stamp_ns, policy.ego_max_bracket_ns)
        if ego_bracket is None:
            drops.append(SampleDrop(grid, timestamp, SampleDropReason.EGO_STATE_INTERPOLATION_MISS))
            continue

        tf_view = _epoch_view(bag.tf_clock_epochs, anchor.clock_epoch_index, "TF")
        if sum(record is anchor for record in tf_view.raw_records) != 1:
            raise IntegrityError("selected raw TF anchor is not a unique epoch member")
        causal_view = causal_tf_interpolation_view(tf_view, anchor)
        future_brackets = []
        for offset in policy.future_offsets_ns:
            bracket = _bracket(causal_view, anchor.header_stamp_ns + offset, policy.future_max_bracket_ns)
            if bracket is None:
                break
            future_brackets.append(bracket)
        if len(future_brackets) != len(policy.future_offsets_ns):
            drops.append(SampleDrop(grid, timestamp, SampleDropReason.FUTURE_HORIZON_MISS))
            continue

        current = PlannedBracket(anchor, anchor, 0.0)
        sample = PlannedSample(
            grid_receipt_ns=grid,
            front=front,
            left=left,
            right=right,
            lidar=lidar,
            trajectory_anchor_tf=anchor,
            ego_bracket_clock_epoch_index=nearest_ego.clock_epoch_index,
            ego_bracket=ego_bracket,
            current_pose_bracket=current,
            future_pose_brackets=tuple(future_brackets),
        )
        samples.append(sample)
        for topic, record in zip(topics, (front, left, right, lidar)):
            selected[topic].add(record.stable_record_index)

    return BagSamplePlan(
        bag_path=bag.path,
        source_bag_sha256=bag.source_bag_sha256,
        resolved_domain_sha256=domain.sha256,
        samples=tuple(samples),
        drops=tuple(drops),
        selected_payload_indices={
            topic: frozenset(indices) for topic, indices in selected.items()
        },
    )
