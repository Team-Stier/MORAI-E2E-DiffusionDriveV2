from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from morai_ddv2.config import load_resolved_domain
from morai_ddv2.config.models import ResolvedDomain, thaw_json
from morai_ddv2.errors import ConfigurationError, IntegrityError
from morai_ddv2.preprocess.diagnostics import EgoDiagnosticValue, TfEgoDiagnostic
from morai_ddv2.preprocess.geometry import Pose
from morai_ddv2.preprocess.planner import (
    BagSamplePlan,
    FrontTimestampOrderViolation,
    PlannedBracket,
    PlannedSample,
    SampleDrop,
    SampleDropReason,
    build_sample_plan,
)
from morai_ddv2.ros1 import (
    BagDisposition,
    BagIndex,
    BagInventory,
    ClockEpochView,
    EgoRecord,
    EgoValue,
    EpochSafeTfEgoDiagnostic,
    RecordRef,
    TfEgoGateProvenance,
    TfPoseRecord,
)


ROOT = Path(__file__).resolve().parents[3]
DOMAIN = load_resolved_domain(ROOT / "tests/fixtures/golden/resolved-domain.json")
T0 = 10_000_000_000
FRONT_RECEIPT = 1_000_000_000


def _topics():
    cameras = DOMAIN.document["sensors"]["cameras"]
    by_role = {camera["role"]: camera["topic"] for camera in cameras}
    return by_role["front"], by_role["left"], by_role["right"], DOMAIN.document[
        "sensors"
    ]["lidar"]["topic"]


def _pose_record(header, receipt, index, *, x=0.0, epoch=0):
    return TfPoseRecord(
        header_stamp_ns=header,
        receipt_stamp_ns=receipt,
        stable_record_index=index,
        clock_epoch_index=epoch,
        value=Pose((float(x), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )


def _ego_record(header, receipt, index, *, epoch=0):
    return EgoRecord(
        header_stamp_ns=header,
        receipt_stamp_ns=receipt,
        stable_record_index=index,
        clock_epoch_index=epoch,
        value=EgoValue(
            velocity_xy=(3.0, -1.0),
            acceleration_xy=(0.5, -0.5),
            diagnostic=EgoDiagnosticValue((0.0, 0.0), 0.0),
        ),
    )


def _collapsed(records):
    result = []
    for record in records:
        if not result or result[-1].header_stamp_ns != record.header_stamp_ns:
            result.append(record)
    return tuple(result)


def _bracket(records, target):
    exact = [record for record in records if record.header_stamp_ns == target]
    if exact:
        return exact[0], exact[0]
    before = [record for record in records if record.header_stamp_ns < target][-1]
    after = [record for record in records if record.header_stamp_ns > target][0]
    return before, after


def _diagnostic(raw_ego, raw_poses, tf_interpolation):
    provenance = []
    for ego in raw_ego:
        anchor = min(
            raw_poses,
            key=lambda pose: (
                abs(pose.receipt_stamp_ns - ego.receipt_stamp_ns),
                pose.receipt_stamp_ns,
                pose.stable_record_index,
            ),
        )
        before, after = _bracket(tf_interpolation, ego.header_stamp_ns)
        provenance.append(
            TfEgoGateProvenance(
                ego_header_stamp_ns=ego.header_stamp_ns,
                ego_receipt_stamp_ns=ego.receipt_stamp_ns,
                ego_stable_record_index=ego.stable_record_index,
                selected_tf_clock_epoch_index=anchor.clock_epoch_index,
                selected_tf_anchor_record_index=anchor.stable_record_index,
                tf_bracket_header_stamp_ns=(
                    before.header_stamp_ns,
                    after.header_stamp_ns,
                ),
                tf_bracket_receipt_ns=(
                    before.receipt_stamp_ns,
                    after.receipt_stamp_ns,
                ),
                tf_bracket_record_index=(
                    before.stable_record_index,
                    after.stable_record_index,
                ),
                exclusion_reason=None,
            )
        )
    count = len(raw_ego)
    return EpochSafeTfEgoDiagnostic(
        metric=TfEgoDiagnostic(count, 0, 0.0, 0.0, 0.0, 0.0),
        provenance=tuple(provenance),
        raw_ego_record_count=count,
        epoch_local_comparable_count=count,
        no_epoch_local_bracket_count=0,
        ambiguous_epoch_selection_count=0,
        cross_epoch_bracket_prevented_count=0,
    )


def _bag(
    *,
    front=None,
    left=None,
    right=None,
    lidar=None,
    raw_poses=None,
    raw_ego=None,
    diagnostic=True,
):
    front_topic, left_topic, right_topic, lidar_topic = _topics()
    if front is None:
        front = (RecordRef(front_topic, 9_900_000_000, FRONT_RECEIPT, 1),)
    if left is None:
        left = (RecordRef(left_topic, 9_920_000_000, 1_090_000_000, 2),)
    if right is None:
        right = (RecordRef(right_topic, 9_880_000_000, 910_000_000, 3),)
    if lidar is None:
        lidar = (RecordRef(lidar_topic, 123, 1_045_000_000, 4),)
    if raw_poses is None:
        raw_poses = tuple(
            _pose_record(
                T0 + 500_000_000 * index,
                FRONT_RECEIPT + 500_000_000 * index,
                100 + index,
                x=index,
            )
            for index in range(9)
        )
    if raw_ego is None:
        raw_ego = tuple(
            _ego_record(T0, 900_000_000 + 10_000_000 * index, 200 + index)
            for index in range(10)
        )
    tf_interpolation = _collapsed(raw_poses)
    ego_indices = tuple(sorted({record.clock_epoch_index for record in raw_ego}))
    tf_indices = tuple(sorted({record.clock_epoch_index for record in raw_poses}))
    ego_views = tuple(
        ClockEpochView(
            epoch,
            tuple(record for record in raw_ego if record.clock_epoch_index == epoch),
            _collapsed(
                tuple(
                    record
                    for record in raw_ego
                    if record.clock_epoch_index == epoch
                )
            ),
        )
        for epoch in ego_indices
    )
    tf_views = tuple(
        ClockEpochView(
            epoch,
            tuple(record for record in raw_poses if record.clock_epoch_index == epoch),
            _collapsed(
                tuple(
                    record
                    for record in raw_poses
                    if record.clock_epoch_index == epoch
                )
            ),
        )
        for epoch in tf_indices
    )
    gate = _diagnostic(raw_ego, raw_poses, tf_interpolation) if diagnostic else None
    inventory = BagInventory(
        disposition=BagDisposition.ELIGIBLE,
        rejection_code=None,
        topic_types=(),
        topic_record_counts=(),
        camera_decode_error_record_indices=(),
        tf_ego_diagnostic=gate,
    )
    return BagIndex(
        path=Path("/tmp/synthetic.bag"),
        source_bag_sha256="a" * 64,
        inventory=inventory,
        front=tuple(front),
        left=tuple(left),
        right=tuple(right),
        lidar=tuple(lidar),
        raw_ego=tuple(raw_ego),
        ego_clock_epochs=ego_views,
        raw_poses=tuple(raw_poses),
        tf_clock_epochs=tf_views,
        extrinsics=(),
    )


def test_public_planner_models_and_drop_taxonomy_are_exact_and_immutable() -> None:
    assert [reason.value for reason in SampleDropReason] == [
        "DECODE_ERROR",
        "CAMERA_SYNC_MISS",
        "LIDAR_SYNC_MISS",
        "EGO_STATE_INTERPOLATION_MISS",
        "POSE_INTERPOLATION_MISS",
        "FUTURE_HORIZON_MISS",
        "NONFINITE_VALUE",
        "INVALID_COMMAND",
        "EMPTY_LIDAR",
    ]
    assert [field.name for field in fields(SampleDrop)] == [
        "grid_receipt_ns",
        "timestamp_ns",
        "reason",
    ]
    assert [field.name for field in fields(PlannedBracket)] == [
        "before",
        "after",
        "weight",
    ]
    assert [field.name for field in fields(PlannedSample)] == [
        "grid_receipt_ns",
        "front",
        "left",
        "right",
        "lidar",
        "trajectory_anchor_tf",
        "ego_bracket_clock_epoch_index",
        "ego_bracket",
        "current_pose_bracket",
        "future_pose_brackets",
    ]
    assert [field.name for field in fields(BagSamplePlan)] == [
        "bag_path",
        "source_bag_sha256",
        "resolved_domain_sha256",
        "samples",
        "drops",
        "selected_payload_indices",
    ]

    drop = SampleDrop(0, None, SampleDropReason.CAMERA_SYNC_MISS)
    with pytest.raises(FrozenInstanceError):
        drop.grid_receipt_ns = 1


def test_planner_builds_option_d_plan_and_exact_selected_union() -> None:
    plan = build_sample_plan(_bag(), DOMAIN)

    assert plan.bag_path == Path("/tmp/synthetic.bag")
    assert plan.resolved_domain_sha256 == DOMAIN.sha256
    assert plan.drops == ()
    assert len(plan.samples) == 1
    sample = plan.samples[0]
    assert sample.grid_receipt_ns == FRONT_RECEIPT
    assert sample.front.header_stamp_ns == 9_900_000_000
    assert sample.left.stable_record_index == 2
    assert sample.right.stable_record_index == 3
    assert sample.lidar.stable_record_index == 4
    assert sample.trajectory_anchor_tf.stable_record_index == 100
    assert sample.ego_bracket_clock_epoch_index == 0
    assert sample.current_pose_bracket == PlannedBracket(
        sample.trajectory_anchor_tf, sample.trajectory_anchor_tf, 0.0
    )
    assert len(sample.future_pose_brackets) == 8
    assert [bracket.before.header_stamp_ns for bracket in sample.future_pose_brackets] == [
        T0 + 500_000_000 * index for index in range(1, 9)
    ]
    front_topic, left_topic, right_topic, lidar_topic = _topics()
    assert plan.selected_payload_indices == {
        front_topic: frozenset({1}),
        left_topic: frozenset({2}),
        right_topic: frozenset({3}),
        lidar_topic: frozenset({4}),
    }
    with pytest.raises(TypeError):
        plan.selected_payload_indices[front_topic] = frozenset()


def test_planner_uses_raw_receipt_nearest_tf_before_duplicate_collapse() -> None:
    raw_poses = (
        _pose_record(T0, 970_000_000, 100, x=1.0),
        _pose_record(T0, 999_000_000, 101, x=1.0),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            101 + index,
            x=1.0 + index,
        )
        for index in range(1, 9)
    )

    sample = build_sample_plan(_bag(raw_poses=raw_poses), DOMAIN).samples[0]

    assert sample.trajectory_anchor_tf is raw_poses[1]
    assert sample.current_pose_bracket.before is raw_poses[1]
    assert sample.current_pose_bracket.after is raw_poses[1]
    assert (
        sample.current_pose_bracket.before.header_stamp_ns,
        sample.current_pose_bracket.after.header_stamp_ns,
    ) == (T0, T0)
    assert (
        sample.current_pose_bracket.before.receipt_stamp_ns,
        sample.current_pose_bracket.after.receipt_stamp_ns,
    ) == (999_000_000, 999_000_000)
    assert (
        sample.current_pose_bracket.before.stable_record_index,
        sample.current_pose_bracket.after.stable_record_index,
    ) == (101, 101)
    assert sample.current_pose_bracket.weight == 0.0
    assert not bool(np.signbit(sample.current_pose_bracket.weight))
    assert sample.future_pose_brackets[0].before is raw_poses[2]
    assert all(
        endpoint.clock_epoch_index == sample.trajectory_anchor_tf.clock_epoch_index
        for bracket in sample.future_pose_brackets
        for endpoint in (bracket.before, bracket.after)
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"left": ()}, SampleDropReason.CAMERA_SYNC_MISS),
        ({"lidar": ()}, SampleDropReason.LIDAR_SYNC_MISS),
        ({"tf_far": True}, SampleDropReason.POSE_INTERPOLATION_MISS),
    ],
)
def test_planner_emits_one_drop_at_first_failed_stage(change, reason) -> None:
    if change.get("tf_far"):
        raw_poses = tuple(
            _pose_record(
                T0 + 500_000_000 * index,
                2_000_000_000 + 500_000_000 * index,
                100 + index,
                x=index,
            )
            for index in range(9)
        )
        bag = _bag(raw_poses=raw_poses)
    else:
        bag = _bag(**change)
    plan = build_sample_plan(bag, DOMAIN)
    assert plan.samples == ()
    assert plan.drops == (SampleDrop(FRONT_RECEIPT, 9_900_000_000, reason),)
    assert all(not indices for indices in plan.selected_payload_indices.values())


def test_planner_side_camera_requires_header_and_receipt_filters_before_tie_key() -> None:
    _, left_topic, _, _ = _topics()
    left = (
        RecordRef(left_topic, 9_900_000_000, 1_100_000_001, 20),
        RecordRef(left_topic, 9_929_000_000, 1_080_000_000, 21),
    )
    sample = build_sample_plan(_bag(left=left), DOMAIN).samples[0]
    assert sample.left.stable_record_index == 21


def test_planner_enforces_front_header_order_before_no_valid_sample() -> None:
    front_topic, _, _, _ = _topics()
    fronts = (
        RecordRef(front_topic, 10_000, 1_000_000_000, 1),
        RecordRef(front_topic, 9_999, 1_500_000_000, 5),
    )
    with pytest.raises(FrontTimestampOrderViolation) as caught:
        build_sample_plan(_bag(front=fronts, left=(), right=(), lidar=()), DOMAIN)
    assert caught.value.rejection_code.value == "FRONT_TIMESTAMP_ORDER_VIOLATION"


def test_planner_empty_front_produces_bound_empty_plan() -> None:
    plan = build_sample_plan(_bag(front=()), DOMAIN)
    assert plan.samples == ()
    assert plan.drops == ()
    assert set(plan.selected_payload_indices) == set(_topics())
    assert all(not indices for indices in plan.selected_payload_indices.values())


def test_planner_rejects_missing_epoch_safe_gate_and_policy_drift() -> None:
    with pytest.raises(IntegrityError):
        build_sample_plan(_bag(diagnostic=False), DOMAIN)

    document = thaw_json(DOMAIN.document)
    document["sampling"]["grid_period_ns"] = 400_000_000
    drifted = ResolvedDomain(document=document, sha256="b" * 64)
    with pytest.raises(ConfigurationError):
        build_sample_plan(_bag(), drifted)


@pytest.mark.parametrize(
    "counter_name",
    [
        "no_epoch_local_bracket_count",
        "ambiguous_epoch_selection_count",
        "cross_epoch_bracket_prevented_count",
        "metric.excluded_count",
    ],
)
def test_planner_rejects_internally_inconsistent_gate_counters(
    counter_name,
) -> None:
    bag = _bag()
    report = bag.inventory.tf_ego_diagnostic
    assert report is not None
    if counter_name.startswith("metric."):
        object.__setattr__(report.metric, counter_name.split(".", 1)[1], 1)
    else:
        object.__setattr__(report, counter_name, 1)

    with pytest.raises(IntegrityError):
        build_sample_plan(bag, DOMAIN)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_tf_anchor_record_index", 999),
        ("selected_tf_clock_epoch_index", 1),
    ],
)
def test_planner_rejects_gate_provenance_that_changes_receipt_anchor(
    field, value
) -> None:
    bag = _bag()
    report = bag.inventory.tf_ego_diagnostic
    assert report is not None
    object.__setattr__(report.provenance[0], field, value)

    with pytest.raises(IntegrityError):
        build_sample_plan(bag, DOMAIN)


def test_planner_rejects_reset_stream_rewritten_by_global_header_sort() -> None:
    epoch_zero = tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            100 + index,
            x=index,
            epoch=0,
        )
        for index in range(9)
    )
    epoch_one = (
        _pose_record(T0 - 1_000_000_000, 5_500_000_000, 109, x=-1.0, epoch=1),
        _pose_record(T0, 6_000_000_000, 110, x=0.0, epoch=1),
    )
    bag = _bag(raw_poses=epoch_zero + epoch_one)
    assert len(build_sample_plan(bag, DOMAIN).samples) == 1
    globally_header_sorted = tuple(
        sorted(
            bag.raw_poses,
            key=lambda record: (
                record.header_stamp_ns,
                record.receipt_stamp_ns,
                record.stable_record_index,
            ),
        )
    )
    object.__setattr__(bag, "raw_poses", globally_header_sorted)

    with pytest.raises(IntegrityError):
        build_sample_plan(bag, DOMAIN)


def test_planner_rejects_nonmaximal_tf_epoch_split() -> None:
    bag = _bag()
    split = 4
    for record in bag.raw_poses[split:]:
        object.__setattr__(record, "clock_epoch_index", 1)
    object.__setattr__(
        bag,
        "tf_clock_epochs",
        (
            ClockEpochView(
                0,
                bag.raw_poses[:split],
                bag.raw_poses[:split],
            ),
            ClockEpochView(
                1,
                bag.raw_poses[split:],
                bag.raw_poses[split:],
            ),
        ),
    )
    with pytest.raises(IntegrityError):
        build_sample_plan(bag, DOMAIN)


def test_planner_rejects_incomplete_epoch_interpolation_view() -> None:
    bag = _bag()
    incomplete = bag.raw_poses[:3] + bag.raw_poses[4:]
    object.__setattr__(
        bag,
        "tf_clock_epochs",
        (ClockEpochView(0, bag.raw_poses, incomplete),),
    )
    with pytest.raises(IntegrityError):
        build_sample_plan(bag, DOMAIN)


def test_planner_rejects_later_equal_header_collapse_representative() -> None:
    raw_poses = (
        _pose_record(T0, 970_000_000, 100, x=1.0),
        _pose_record(T0, 999_000_000, 101, x=1.0),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            101 + index,
            x=1.0 + index,
        )
        for index in range(1, 9)
    )
    bag = _bag(raw_poses=raw_poses)
    wrong_collapse = (raw_poses[1],) + raw_poses[2:]
    object.__setattr__(
        bag,
        "tf_clock_epochs",
        (ClockEpochView(0, raw_poses, wrong_collapse),),
    )
    with pytest.raises(IntegrityError):
        build_sample_plan(bag, DOMAIN)


@pytest.mark.parametrize(
    "second_front",
    [
        RecordRef(_topics()[0], 9_899_999_999, 1_500_000_000, 5),
        RecordRef(_topics()[0], 9_900_000_001, 1_500_000_000, 1),
    ],
)
def test_bag_sample_plan_rejects_nonincreasing_or_reused_front_identity(
    second_front,
) -> None:
    plan = build_sample_plan(_bag(), DOMAIN)
    first = plan.samples[0]
    second = replace(first, grid_receipt_ns=1_500_000_000, front=second_front)
    selected = {
        topic: set(indices) for topic, indices in plan.selected_payload_indices.items()
    }
    selected[second_front.topic].add(second_front.stable_record_index)
    with pytest.raises(ValueError):
        BagSamplePlan(
            plan.bag_path,
            plan.source_bag_sha256,
            plan.resolved_domain_sha256,
            (first, second),
            (),
            {topic: frozenset(indices) for topic, indices in selected.items()},
        )


def test_front_receipt_tolerance_tie_ignores_header_and_uses_stable_index() -> None:
    front_topic, left_topic, right_topic, lidar_topic = _topics()
    fronts = (
        RecordRef(front_topic, 123, 950_000_000, 5),
        RecordRef(front_topic, 123, 950_000_000, 1),
        RecordRef(front_topic, 999_000_000_000, 1_050_000_000, 9),
    )
    bag = _bag(
        front=tuple(reversed(fronts)),
        left=(RecordRef(left_topic, 123, 1_050_000_000, 2),),
        right=(RecordRef(right_topic, 123, 850_000_000, 3),),
        lidar=(RecordRef(lidar_topic, 999_999_999_999, 995_000_000, 4),),
    )
    sample = build_sample_plan(bag, DOMAIN).samples[0]
    assert sample.grid_receipt_ns == 1_000_000_000
    assert sample.front.stable_record_index == 1
    assert sample.front.header_stamp_ns == 123


def test_front_receipt_tolerance_plus_one_drops_without_timestamp() -> None:
    front_topic, _, _, _ = _topics()
    fronts = (
        RecordRef(front_topic, 1, 949_999_999, 1),
        RecordRef(front_topic, 2, 1_050_000_001, 5),
    )
    plan = build_sample_plan(_bag(front=fronts), DOMAIN)
    assert plan.samples == ()
    assert plan.drops == (
        SampleDrop(1_000_000_000, None, SampleDropReason.CAMERA_SYNC_MISS),
    )


@pytest.mark.parametrize(
    ("header_delta", "receipt_delta", "survives"),
    [
        (30_000_000, 100_000_000, True),
        (30_000_001, 100_000_000, False),
        (30_000_000, 100_000_001, False),
    ],
)
def test_side_camera_dual_tolerance_boundaries(
    header_delta, receipt_delta, survives
) -> None:
    _, left_topic, _, _ = _topics()
    left = (
        RecordRef(
            left_topic,
            9_900_000_000 + header_delta,
            FRONT_RECEIPT + receipt_delta,
            2,
        ),
    )
    plan = build_sample_plan(_bag(left=left), DOMAIN)
    assert bool(plan.samples) is survives
    if not survives:
        assert plan.drops[0].reason is SampleDropReason.CAMERA_SYNC_MISS


@pytest.mark.parametrize(
    ("lidar_delta", "survives"),
    [(45_000_000, True), (45_000_001, False)],
)
def test_lidar_receipt_tolerance_boundary(lidar_delta, survives) -> None:
    _, _, _, lidar_topic = _topics()
    lidar = (
        RecordRef(lidar_topic, 999_000_000_000, FRONT_RECEIPT + lidar_delta, 4),
    )
    plan = build_sample_plan(_bag(lidar=lidar), DOMAIN)
    assert bool(plan.samples) is survives
    if not survives:
        assert plan.drops[0].reason is SampleDropReason.LIDAR_SYNC_MISS


@pytest.mark.parametrize(
    ("anchor_delta", "survives"),
    [(50_000_000, True), (50_000_001, False)],
)
def test_raw_tf_anchor_receipt_tolerance_boundary(anchor_delta, survives) -> None:
    raw_poses = tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + anchor_delta + 500_000_000 * index,
            100 + index,
            x=index,
        )
        for index in range(9)
    )
    plan = build_sample_plan(_bag(raw_poses=raw_poses), DOMAIN)
    assert bool(plan.samples) is survives
    if not survives:
        assert plan.drops[0].reason is SampleDropReason.POSE_INTERPOLATION_MISS


def test_ego_receipt_selected_epoch_has_no_other_epoch_fallback() -> None:
    raw_poses = (
        _pose_record(T0 - 200_000_000, 800_000_000, 90, x=-0.2),
        _pose_record(T0 - 100_000_000, 900_000_000, 91, x=-0.1),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            100 + index,
            x=index,
        )
        for index in range(9)
    )
    raw_ego = tuple(
        _ego_record(T0, 100_000_000 * (index + 1), 200 + index, epoch=0)
        for index in range(5)
    ) + tuple(
        _ego_record(
            T0 - (200_000_000 if index < 3 else 100_000_000),
            950_000_000 + 10_000_000 * index,
            205 + index,
            epoch=1,
        )
        for index in range(5)
    )
    plan = build_sample_plan(
        _bag(raw_poses=raw_poses, raw_ego=raw_ego), DOMAIN
    )
    assert plan.samples == ()
    assert plan.drops[0].reason is SampleDropReason.EGO_STATE_INTERPOLATION_MISS


def test_missing_eighth_future_target_drops_entire_grid_without_payload_refs() -> None:
    raw_poses = tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            100 + index,
            x=index,
        )
        for index in range(8)
    )
    plan = build_sample_plan(_bag(raw_poses=raw_poses), DOMAIN)
    assert plan.samples == ()
    assert plan.drops == (
        SampleDrop(
            FRONT_RECEIPT,
            9_900_000_000,
            SampleDropReason.FUTURE_HORIZON_MISS,
        ),
    )
    assert all(not indices for indices in plan.selected_payload_indices.values())


@pytest.mark.parametrize(
    ("after_delta", "survives"),
    [(50_000_000, True), (50_000_001, False)],
)
def test_ego_header_bracket_gap_boundary(after_delta, survives) -> None:
    raw_poses = (
        _pose_record(T0 - 50_000_000, 800_000_000, 90, x=-0.05),
        _pose_record(T0, FRONT_RECEIPT, 100, x=0.0),
        _pose_record(T0 + after_delta, 1_100_000_000, 101, x=0.05),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            101 + index,
            x=index,
        )
        for index in range(1, 9)
    )
    raw_ego = tuple(
        _ego_record(T0 - 50_000_000, 900_000_000 + index, 200 + index)
        for index in range(5)
    ) + tuple(
        _ego_record(T0 + after_delta, 950_000_000 + index, 205 + index)
        for index in range(5)
    )
    plan = build_sample_plan(
        _bag(raw_poses=raw_poses, raw_ego=raw_ego), DOMAIN
    )
    assert bool(plan.samples) is survives
    if survives:
        assert plan.samples[0].ego_bracket.before.header_stamp_ns == T0 - 50_000_000
        assert plan.samples[0].ego_bracket.after.header_stamp_ns == T0 + 50_000_000
        assert plan.samples[0].ego_bracket.weight == 0.5
    else:
        assert plan.drops[0].reason is SampleDropReason.EGO_STATE_INTERPOLATION_MISS


def test_nearest_raw_tf_anchor_never_falls_back_after_future_miss() -> None:
    second_t0 = T0 - 1_000_000_000
    raw_poses = (
        _pose_record(T0, FRONT_RECEIPT, 100, x=0.0, epoch=0),
        _pose_record(second_t0, 1_010_000_000, 101, x=1.0, epoch=1),
    ) + tuple(
        _pose_record(
            second_t0 + 500_000_000 * index,
            1_010_000_000 + 500_000_000 * index,
            101 + index,
            x=1.0 + index,
            epoch=1,
        )
        for index in range(1, 9)
    )
    plan = build_sample_plan(_bag(raw_poses=raw_poses), DOMAIN)
    assert plan.samples == ()
    assert plan.drops[0].reason is SampleDropReason.FUTURE_HORIZON_MISS


def test_future_targets_use_tf_t0_and_cast_interpolation_weights_once() -> None:
    raw_poses = [_pose_record(T0, FRONT_RECEIPT, 100, x=0.0)]
    for index in range(1, 9):
        target = T0 + 500_000_000 * index
        raw_poses.extend(
            (
                _pose_record(
                    target - 25_000_000,
                    FRONT_RECEIPT + 500_000_000 * index - 25_000_000,
                    100 + index * 2 - 1,
                    x=float(index) - 0.25,
                ),
                _pose_record(
                    target + 25_000_000,
                    FRONT_RECEIPT + 500_000_000 * index + 25_000_000,
                    100 + index * 2,
                    x=float(index) + 0.25,
                ),
            )
        )
    sample = build_sample_plan(_bag(raw_poses=tuple(raw_poses)), DOMAIN).samples[0]
    assert [bracket.weight for bracket in sample.future_pose_brackets] == [0.5] * 8
    assert sample.future_pose_brackets[0].before.header_stamp_ns == (
        T0 + 500_000_000 - 25_000_000
    )


def test_future_asymmetric_bracket_weight_is_cast_once_to_float32() -> None:
    first_target = T0 + 500_000_000
    raw_poses = (
        _pose_record(T0, FRONT_RECEIPT, 100, x=0.0),
        _pose_record(first_target - 10_000_000, 1_490_000_000, 101, x=0.49),
        _pose_record(first_target + 20_000_000, 1_520_000_000, 102, x=0.52),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            101 + index,
            x=index,
        )
        for index in range(2, 9)
    )

    sample = build_sample_plan(_bag(raw_poses=raw_poses), DOMAIN).samples[0]

    assert sample.future_pose_brackets[0].weight == float(np.float32(1.0 / 3.0))


def test_tf_receipt_anchor_is_independent_of_850_second_header_shift() -> None:
    shift = 850_000_000_000
    raw_poses = tuple(
        _pose_record(
            T0 + shift + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            100 + index,
            x=index,
        )
        for index in range(9)
    )
    raw_ego = tuple(
        _ego_record(
            T0 + shift,
            900_000_000 + 10_000_000 * index,
            200 + index,
        )
        for index in range(10)
    )

    sample = build_sample_plan(
        _bag(raw_poses=raw_poses, raw_ego=raw_ego), DOMAIN
    ).samples[0]

    assert sample.trajectory_anchor_tf.stable_record_index == 100
    assert sample.trajectory_anchor_tf.receipt_stamp_ns == FRONT_RECEIPT
    assert sample.trajectory_anchor_tf.header_stamp_ns == T0 + shift


@pytest.mark.parametrize(
    ("first_receipt", "last_receipt", "expected_grids", "selected_receipts"),
    [
        (
            499_999_999,
            1_000_000_000,
            (500_000_000, 1_000_000_000),
            (499_999_999, 1_000_000_000),
        ),
        (
            500_000_000,
            1_000_000_000,
            (500_000_000, 1_000_000_000),
            (500_000_000, 1_000_000_000),
        ),
        (500_000_001, 1_000_000_000, (1_000_000_000,), (1_000_000_000,)),
    ],
)
def test_receipt_grid_origin_minmax_and_header_independence(
    first_receipt, last_receipt, expected_grids, selected_receipts
) -> None:
    front_topic, _, _, _ = _topics()
    header_offset = 850_000_000_000
    fronts = (
        RecordRef(
            front_topic,
            header_offset + first_receipt,
            first_receipt,
            1,
        ),
        RecordRef(
            front_topic,
            header_offset + last_receipt,
            last_receipt,
            5,
        ),
    )

    plan = build_sample_plan(_bag(front=fronts), DOMAIN)

    outcomes = sorted(
        tuple(plan.samples) + tuple(plan.drops),
        key=lambda outcome: outcome.grid_receipt_ns,
    )
    assert tuple(outcome.grid_receipt_ns for outcome in outcomes) == expected_grids
    assert all(first_receipt <= grid <= last_receipt for grid in expected_grids)
    assert tuple(drop.timestamp_ns for drop in plan.drops) == tuple(
        header_offset + receipt for receipt in selected_receipts
    )


@pytest.mark.parametrize("axis", ["front", "side", "lidar"])
def test_sensor_candidate_permutations_preserve_complete_tie_order(axis) -> None:
    front_topic, left_topic, _, lidar_topic = _topics()
    candidates = {
        "front": (
            RecordRef(front_topic, 850_000_000_000, 950_000_000, 8),
            RecordRef(front_topic, 1, 1_050_000_000, 1),
            RecordRef(front_topic, 9_900_000_000, 950_000_000, 2),
        ),
        "side": (
            RecordRef(left_topic, 9_890_000_000, 1_099_000_000, 8),
            RecordRef(left_topic, 9_910_000_000, 901_000_000, 1),
            RecordRef(left_topic, 9_890_000_000, 901_000_000, 2),
        ),
        "lidar": (
            RecordRef(lidar_topic, 850_000_000_000, 955_000_000, 8),
            RecordRef(lidar_topic, 1, 1_045_000_000, 1),
            RecordRef(lidar_topic, 123, 955_000_000, 2),
        ),
    }[axis]

    selected = []
    for ordering in permutations(candidates):
        changes = {axis if axis != "side" else "left": ordering}
        if axis == "front":
            changes.update(
                {
                    "left": (
                        RecordRef(left_topic, 9_900_000_000, 1_000_000_000, 20),
                    ),
                    "right": (
                        RecordRef(_topics()[2], 9_900_000_000, 900_000_000, 21),
                    ),
                    "lidar": (
                        RecordRef(lidar_topic, 123, 955_000_000, 22),
                    ),
                }
            )
        sample = build_sample_plan(_bag(**changes), DOMAIN).samples[0]
        selected.append(getattr(sample, axis if axis != "side" else "left"))

    assert {record.stable_record_index for record in selected} == {2}
    if axis == "front":
        assert {record.header_stamp_ns for record in selected} == {9_900_000_000}
    elif axis == "side":
        assert {record.receipt_stamp_ns for record in selected} == {901_000_000}
    else:
        assert {record.header_stamp_ns for record in selected} == {123}


def test_tf_receipt_tie_uses_receipt_then_stable_index_not_header() -> None:
    raw_poses = (
        _pose_record(T0, 950_000_000, 100, x=0.0),
        _pose_record(T0, 950_000_000, 101, x=0.0),
        _pose_record(T0 + 10_000_000, 1_050_000_000, 102, x=0.01),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            1_500_000_000 + 500_000_000 * (index - 1),
            102 + index,
            x=index,
        )
        for index in range(1, 9)
    )

    sample = build_sample_plan(_bag(raw_poses=raw_poses), DOMAIN).samples[0]

    assert sample.trajectory_anchor_tf.stable_record_index == 100
    assert sample.trajectory_anchor_tf.header_stamp_ns == T0


def _overlapping_ego_epoch_bag():
    raw_poses = (
        _pose_record(T0 - 50_000_000, 800_000_000, 90, x=-0.05),
        _pose_record(T0, FRONT_RECEIPT, 100, x=0.0),
        _pose_record(T0 + 50_000_000, 1_100_000_000, 101, x=0.05),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            101 + index,
            x=index,
        )
        for index in range(1, 9)
    )
    overlapping_headers = (
        T0 - 50_000_000,
        T0 - 25_000_000,
        T0,
        T0 + 25_000_000,
        T0 + 50_000_000,
    )
    raw_ego = tuple(
        _ego_record(header, 100_000_000 * (index + 1), 200 + index, epoch=0)
        for index, header in enumerate(overlapping_headers)
    ) + tuple(
        _ego_record(
            header,
            950_000_000 + 10_000_000 * index,
            205 + index,
            epoch=1,
        )
        for index, header in enumerate(overlapping_headers)
    )
    return _bag(raw_poses=raw_poses, raw_ego=raw_ego)


def test_ego_exact_and_overlapping_epoch_selection_is_receipt_local() -> None:
    default = build_sample_plan(_bag(), DOMAIN).samples[0]
    assert default.ego_bracket.before is default.ego_bracket.after
    assert default.ego_bracket.weight == 0.0

    sample = build_sample_plan(_overlapping_ego_epoch_bag(), DOMAIN).samples[0]
    assert sample.ego_bracket_clock_epoch_index == 1
    assert sample.ego_bracket.before is sample.ego_bracket.after
    assert sample.ego_bracket.before.header_stamp_ns == T0
    assert sample.ego_bracket.before.stable_record_index == 207


@pytest.mark.parametrize("direction", [-1, 1])
def test_ego_bag_edge_extrapolation_is_forbidden(direction) -> None:
    ego_header = T0 + direction * 10_000_000
    support = _pose_record(
        ego_header,
        800_000_000 if direction < 0 else 1_100_000_000,
        90 if direction < 0 else 101,
        x=direction * 0.01,
    )
    anchor = _pose_record(T0, FRONT_RECEIPT, 100, x=0.0)
    future = tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            110 + index,
            x=index,
        )
        for index in range(1, 9)
    )
    raw_poses = ((support, anchor) if direction < 0 else (anchor, support)) + future
    raw_ego = tuple(
        _ego_record(ego_header, 900_000_000 + 10_000_000 * index, 200 + index)
        for index in range(10)
    )

    plan = build_sample_plan(_bag(raw_poses=raw_poses, raw_ego=raw_ego), DOMAIN)

    assert plan.samples == ()
    assert plan.drops[0].reason is SampleDropReason.EGO_STATE_INTERPOLATION_MISS


@pytest.mark.parametrize(
    ("gap_ns", "survives"),
    [(100_000_000, True), (100_000_001, False)],
)
def test_future_header_bracket_gap_boundary(gap_ns, survives) -> None:
    first_target = T0 + 500_000_000
    before_delta = 50_000_000
    raw_poses = (
        _pose_record(T0, FRONT_RECEIPT, 100, x=0.0),
        _pose_record(
            first_target - before_delta,
            1_450_000_000,
            101,
            x=0.45,
        ),
        _pose_record(
            first_target - before_delta + gap_ns,
            1_550_000_000,
            102,
            x=0.55,
        ),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            101 + index * 2,
            x=index,
        )
        for index in range(2, 9)
    )

    plan = build_sample_plan(_bag(raw_poses=raw_poses), DOMAIN)

    assert bool(plan.samples) is survives
    if survives:
        assert plan.samples[0].future_pose_brackets[0].weight == 0.5
    else:
        assert plan.drops[0].reason is SampleDropReason.FUTURE_HORIZON_MISS


def test_future_strict_header_reset_never_uses_numerically_valid_later_epoch() -> None:
    epoch_zero = (
        _pose_record(T0, FRONT_RECEIPT, 100, x=0.0, epoch=0),
    ) + tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            100 + index,
            x=index,
            epoch=0,
        )
        for index in range(1, 8)
    ) + (
        _pose_record(T0 + 5_000_000_000, 5_500_000_000, 108, x=10.0, epoch=0),
    )
    epoch_one = (
        _pose_record(T0 + 3_950_000_000, 6_000_000_000, 109, x=7.9, epoch=1),
        _pose_record(T0 + 4_050_000_000, 6_100_000_000, 110, x=8.1, epoch=1),
    )

    plan = build_sample_plan(_bag(raw_poses=epoch_zero + epoch_one), DOMAIN)

    assert plan.samples == ()
    assert plan.drops == (
        SampleDrop(
            FRONT_RECEIPT,
            9_900_000_000,
            SampleDropReason.FUTURE_HORIZON_MISS,
        ),
    )


def test_every_planner_failure_stage_has_one_deterministic_reason() -> None:
    front_topic, _, _, _ = _topics()
    front_miss = _bag(
        front=(
            RecordRef(front_topic, 1, 449_999_999, 1),
            RecordRef(front_topic, 2, 550_000_001, 5),
        ),
        left=(),
        right=(),
        lidar=(),
    )
    pose_far = tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            2_000_000_000 + 500_000_000 * index,
            100 + index,
            x=index,
        )
        for index in range(9)
    )
    future_short = tuple(
        _pose_record(
            T0 + 500_000_000 * index,
            FRONT_RECEIPT + 500_000_000 * index,
            100 + index,
            x=index,
        )
        for index in range(8)
    )
    ego_header = T0 + 10_000_000
    ego_extrapolation = _bag(
        raw_poses=(
            _pose_record(T0, FRONT_RECEIPT, 100, x=0.0),
            _pose_record(ego_header, 1_100_000_000, 150, x=0.01),
        )
        + future_short[1:],
        raw_ego=tuple(
            _ego_record(
                ego_header,
                900_000_000 + 10_000_000 * index,
                200 + index,
            )
            for index in range(10)
        ),
    )
    cases = (
        (front_miss, SampleDropReason.CAMERA_SYNC_MISS),
        (_bag(left=(), right=(), lidar=()), SampleDropReason.CAMERA_SYNC_MISS),
        (_bag(right=(), lidar=()), SampleDropReason.CAMERA_SYNC_MISS),
        (_bag(lidar=(), raw_poses=pose_far), SampleDropReason.LIDAR_SYNC_MISS),
        (_bag(raw_poses=pose_far), SampleDropReason.POSE_INTERPOLATION_MISS),
        (ego_extrapolation, SampleDropReason.EGO_STATE_INTERPOLATION_MISS),
        (_bag(raw_poses=future_short), SampleDropReason.FUTURE_HORIZON_MISS),
    )

    observed = []
    for bag, expected in cases:
        plan = build_sample_plan(bag, DOMAIN)
        assert plan.samples == ()
        assert len(plan.drops) == 1
        observed.append(plan.drops[0].reason)
        assert plan.drops[0].reason is expected
    assert observed == [
        SampleDropReason.CAMERA_SYNC_MISS,
        SampleDropReason.CAMERA_SYNC_MISS,
        SampleDropReason.CAMERA_SYNC_MISS,
        SampleDropReason.LIDAR_SYNC_MISS,
        SampleDropReason.POSE_INTERPOLATION_MISS,
        SampleDropReason.EGO_STATE_INTERPOLATION_MISS,
        SampleDropReason.FUTURE_HORIZON_MISS,
    ]


@pytest.mark.parametrize("stream_name", ["raw_poses", "raw_ego"])
def test_raw_state_receipt_order_permutation_is_rejected(stream_name) -> None:
    bag = _bag()
    source = tuple(getattr(bag, stream_name))
    permuted = (source[1], source[0]) + source[2:]
    object.__setattr__(bag, stream_name, permuted)
    with pytest.raises(IntegrityError):
        build_sample_plan(bag, DOMAIN)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("header_stamp_ns", True),
        ("receipt_stamp_ns", 1 << 63),
        ("header_stamp_ns", 1 << 100),
        ("stable_record_index", -1),
    ],
)
def test_planner_rejects_invalid_sensor_integer_handoff(field, value) -> None:
    bag = _bag()
    object.__setattr__(bag.front[0], field, value)
    with pytest.raises(IntegrityError):
        build_sample_plan(bag, DOMAIN)


def test_sampling_policy_rejects_missing_extra_bool_overflow_and_coherent_drift() -> None:
    documents = []

    missing = thaw_json(DOMAIN.document)
    missing["sampling"].pop("drop_on_required_sensor_miss")
    documents.append(missing)

    extra = thaw_json(DOMAIN.document)
    extra["sampling"]["unexpected"] = 1
    documents.append(extra)

    boolean = thaw_json(DOMAIN.document)
    boolean["sampling"]["grid_period_ns"] = True
    documents.append(boolean)

    overflow = thaw_json(DOMAIN.document)
    overflow["sampling"]["clock_policy"]["front"]["tolerance_ns"] = 1 << 63
    documents.append(overflow)

    coherent = thaw_json(DOMAIN.document)
    coherent["sampling"]["grid_period_ns"] = 250_000_000
    coherent["sampling"]["sample_rate_hz"] = 4
    documents.append(coherent)

    for index, document in enumerate(documents):
        with pytest.raises(ConfigurationError):
            build_sample_plan(
                _bag(),
                ResolvedDomain(document=document, sha256="{:064x}".format(index + 1)),
            )
