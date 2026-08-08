from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from enum import IntEnum

import numpy as np
import pytest

from morai_ddv2.config.models import ResolvedDomain
from morai_ddv2.errors import ConfigurationError
from morai_ddv2.preprocess.trajectory import (
    Command,
    CommandRules,
    InvalidCommand,
    build_status,
    classify_oracle_command,
)


EXPECTED_COMMAND_RULE_SHA256 = (
    "84be65c810e6480117d9c7a31023adc3490e60b2e71f2a40c6c167ec0ae88791"
)


@pytest.fixture
def resolved_publication(repo_root):
    return json.loads(
        (repo_root / "tests" / "fixtures" / "golden" / "resolved-domain.json").read_bytes()
    )


@pytest.fixture
def rules(resolved_publication) -> CommandRules:
    return CommandRules.from_resolved_domain(resolved_publication)


def _trajectory(final_x: float, final_y: float, final_heading: float) -> np.ndarray:
    result = np.zeros((8, 3), dtype=np.float64)
    result[-1] = (final_x, final_y, final_heading)
    return result


def test_command_enum_has_exact_indices_and_iteration_order() -> None:
    assert list(Command) == [
        Command.LEFT,
        Command.STRAIGHT,
        Command.RIGHT,
        Command.UNKNOWN,
    ]
    assert [(member.name, member.value) for member in Command] == [
        ("LEFT", 0),
        ("STRAIGHT", 1),
        ("RIGHT", 2),
        ("UNKNOWN", 3),
    ]


def test_command_rules_factory_supports_raw_full_command_and_immutable_documents(
    resolved_publication,
) -> None:
    document = dict(resolved_publication)
    digest = document.pop("resolved_domain_sha256")
    immutable = ResolvedDomain(document=document, sha256=digest)

    from_raw_document = CommandRules.from_resolved_domain(resolved_publication)
    from_raw_subtree = CommandRules.from_resolved_domain(
        resolved_publication["command"]
    )
    from_immutable_document = CommandRules.from_resolved_domain(immutable.document)
    from_resolved_object = CommandRules.from_resolved_domain(immutable)

    assert from_raw_document == from_raw_subtree
    assert from_raw_document == from_immutable_document
    assert from_raw_document == from_resolved_object
    assert from_raw_document.minimum_path_length_m == 1.0
    assert from_raw_document.left_minimum_final_heading_rad == 0.2
    assert from_raw_document.left_minimum_final_lateral_m == 0.5
    assert from_raw_document.right_maximum_final_heading_rad == -0.2
    assert from_raw_document.right_maximum_final_lateral_m == -0.5
    assert from_raw_document.straight_maximum_abs_final_heading_rad == 0.1
    assert from_raw_document.straight_maximum_abs_final_lateral_m == 0.5


def test_command_rules_are_deeply_immutable(rules: CommandRules) -> None:
    with pytest.raises(FrozenInstanceError):
        rules.minimum_path_length_m = 2.0
    with pytest.raises(TypeError):
        rules.index_mapping["LEFT"] = 9


@pytest.mark.parametrize(
    "mutation",
    [
        lambda command: command.pop("index_mapping"),
        lambda command: command["index_mapping"].update({"LEFT": 1}),
        lambda command: command["index_mapping"].update({"LEFT": False}),
        lambda command: command["index_mapping"].update({"EXTRA": 4}),
        lambda command: command["ordered_rules"].reverse(),
        lambda command: command["ordered_rules"].pop(),
        lambda command: command.update({"ordered_rules": None}),
        lambda command: command["thresholds"].pop("minimum_path_length_m"),
        lambda command: command["thresholds"].update({"minimum_path_length_m": 1.1}),
        lambda command: command["thresholds"].update({"minimum_path_length_m": True}),
        lambda command: command["thresholds"].update({"minimum_path_length_m": float("nan")}),
        lambda command: command["thresholds"].update({"nan_allowed": True}),
        lambda command: command["thresholds"].update(
            {"ambiguous_or_static_class": "STRAIGHT"}
        ),
        lambda command: command["thresholds"]["left"].update(
            {"minimum_final_heading_rad": 0.21}
        ),
        lambda command: command["thresholds"]["right"].update(
            {"maximum_final_lateral_m": float("inf")}
        ),
        lambda command: command["thresholds"]["straight"].update({"extra": 1.0}),
    ],
)
def test_command_rules_factory_rejects_missing_or_changed_contract(
    resolved_publication, mutation
) -> None:
    command = copy.deepcopy(resolved_publication["command"])
    mutation(command)

    with pytest.raises(ConfigurationError):
        CommandRules.from_resolved_domain(command)


def test_command_rules_factory_normalizes_python_integer_float_overflow(
    resolved_publication,
) -> None:
    command = copy.deepcopy(resolved_publication["command"])
    command["thresholds"]["minimum_path_length_m"] = 10**400

    with pytest.raises(ConfigurationError):
        CommandRules.from_resolved_domain(command)


@pytest.mark.parametrize("bad_value", [None, [], "command", {"other": {}}])
def test_command_rules_factory_rejects_values_without_a_command_contract(
    bad_value,
) -> None:
    with pytest.raises(ConfigurationError):
        CommandRules.from_resolved_domain(bad_value)


def test_command_rule_golden_is_exact_canonical_subtree_and_hash(
    repo_root, resolved_publication
) -> None:
    golden_bytes = (
        repo_root / "tests" / "fixtures" / "golden" / "command-rules.json"
    ).read_bytes()
    expected_bytes = (
        b'{"index_mapping":{"LEFT":0,"RIGHT":2,"STRAIGHT":1,"UNKNOWN":3},'
        b'"ordered_rules":["DROP_INVALID_TRAJECTORY","UNKNOWN_STATIC","LEFT",'
        b'"RIGHT","STRAIGHT","UNKNOWN_AMBIGUOUS"],"thresholds":{'
        b'"ambiguous_or_static_class":"UNKNOWN","left":{'
        b'"minimum_final_heading_rad":0.2,"minimum_final_lateral_m":0.5},'
        b'"minimum_path_length_m":1,"nan_allowed":false,"right":{'
        b'"maximum_final_heading_rad":-0.2,"maximum_final_lateral_m":-0.5},'
        b'"straight":{"maximum_abs_final_heading_rad":0.1,'
        b'"maximum_abs_final_lateral_m":0.5}}}'
    )

    assert golden_bytes == expected_bytes
    assert len(golden_bytes) == 497
    assert hashlib.sha256(golden_bytes).hexdigest() == EXPECTED_COMMAND_RULE_SHA256
    assert json.loads(golden_bytes) == {
        key: resolved_publication["command"][key]
        for key in ("index_mapping", "thresholds", "ordered_rules")
    }


@pytest.mark.parametrize(
    ("path_length", "expected"),
    [(0.999999, Command.UNKNOWN), (1.0, Command.STRAIGHT)],
)
def test_static_path_threshold_is_strictly_less_than(
    rules: CommandRules, path_length: float, expected: Command
) -> None:
    actual = classify_oracle_command(
        _trajectory(path_length, 0.0, 0.0), rules
    )

    assert actual is expected


@pytest.mark.parametrize(
    ("final_x", "final_y", "final_heading", "expected"),
    [
        (1.0, 0.5, 0.2, Command.LEFT),
        (1.0, -0.5, -0.2, Command.RIGHT),
        (1.0, 0.5, 0.1, Command.STRAIGHT),
        (1.0, -0.5, -0.1, Command.STRAIGHT),
        (1.0, 0.5, float(np.nextafter(0.2, -np.inf)), Command.UNKNOWN),
        (1.0, float(np.nextafter(0.5, -np.inf)), 0.2, Command.UNKNOWN),
        (1.0, -0.5, float(np.nextafter(-0.2, np.inf)), Command.UNKNOWN),
        (1.0, float(np.nextafter(-0.5, np.inf)), -0.2, Command.UNKNOWN),
        (1.0, 0.0, float(np.nextafter(0.1, np.inf)), Command.UNKNOWN),
        (1.0, float(np.nextafter(0.5, np.inf)), 0.0, Command.UNKNOWN),
        (1.0, -0.8, 0.3, Command.UNKNOWN),
        (1.0, 0.8, -0.3, Command.UNKNOWN),
        (1.0, 0.6, 0.15, Command.UNKNOWN),
    ],
)
def test_command_boundaries_and_ambiguous_regions(
    rules: CommandRules,
    final_x: float,
    final_y: float,
    final_heading: float,
    expected: Command,
) -> None:
    assert (
        classify_oracle_command(
            _trajectory(final_x, final_y, final_heading), rules
        )
        is expected
    )


def test_path_length_includes_origin_to_first_horizon_segment(
    rules: CommandRules,
) -> None:
    trajectory = np.zeros((8, 3), dtype=np.float64)
    trajectory[:, 0] = 1.0

    assert classify_oracle_command(trajectory, rules) is Command.STRAIGHT


def test_path_length_includes_all_intermediate_segments(rules: CommandRules) -> None:
    trajectory = np.zeros((8, 3), dtype=np.float64)
    trajectory[0, 0] = 0.6
    trajectory[1, 0] = 0.0
    trajectory[-1, 1] = 0.5
    trajectory[-1, 2] = 0.2

    assert classify_oracle_command(trajectory, rules) is Command.LEFT


@pytest.mark.parametrize(
    "bad_trajectory",
    [
        np.zeros((8, 2)),
        np.zeros((7, 3)),
        np.zeros((8, 3, 1)),
        np.full((8, 3), np.nan),
        np.full((8, 3), np.inf),
        np.zeros((8, 3), dtype=np.bool_),
        np.full((8, 3), True, dtype=object),
        np.zeros((8, 3), dtype=np.complex128),
        np.full((8, 3), 1.0 + 0.0j, dtype=object),
        np.full((8, 3), 10**400, dtype=object),
        np.full((8, 3), np.datetime64("2026-08-08"), dtype="datetime64[D]"),
        np.full((8, 3), np.timedelta64(1, "s"), dtype="timedelta64[s]"),
        [["0", "0", "0"] for _ in range(8)],
    ],
)
def test_invalid_trajectory_raises_instead_of_falling_back_to_unknown(
    rules: CommandRules, bad_trajectory
) -> None:
    with pytest.raises(InvalidCommand) as captured:
        classify_oracle_command(bad_trajectory, rules)

    assert captured.value.drop_reason == "INVALID_COMMAND"


def test_command_classification_requires_validated_rules() -> None:
    with pytest.raises(TypeError, match="CommandRules"):
        classify_oracle_command(np.zeros((8, 3)), {})


@pytest.mark.parametrize("command", list(Command))
def test_status_has_exact_one_hot_and_raw_body_state(command: Command) -> None:
    velocity = np.asarray([123.25, -45.5], dtype=np.float64)
    acceleration = np.asarray([-7.75, 8.125], dtype=np.float64)

    status = build_status(command, velocity, acceleration)

    expected = np.asarray(
        [
            float(command is Command.LEFT),
            float(command is Command.STRAIGHT),
            float(command is Command.RIGHT),
            float(command is Command.UNKNOWN),
            123.25,
            -45.5,
            -7.75,
            8.125,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(status, expected)
    assert status.dtype == np.float32
    assert status.shape == (8,)
    assert status.flags.c_contiguous


class _OtherCommand(IntEnum):
    LEFT = 0


@pytest.mark.parametrize("bad_command", [0, 3, True, "LEFT", _OtherCommand.LEFT])
def test_status_requires_an_actual_command_enum(bad_command) -> None:
    with pytest.raises(TypeError, match="Command"):
        build_status(bad_command, np.zeros(2), np.zeros(2))


@pytest.mark.parametrize(
    ("velocity", "acceleration"),
    [
        (np.zeros(3), np.zeros(2)),
        (np.zeros(2), np.zeros((1, 2))),
        (np.zeros(2, dtype=np.bool_), np.zeros(2)),
        (np.asarray([True, False], dtype=object), np.zeros(2)),
        (np.zeros(2), np.zeros(2, dtype=np.complex128)),
        (np.zeros(2), np.asarray([1.0 + 0.0j, 2.0], dtype=object)),
        (
            np.full(2, np.datetime64("2026-08-08"), dtype="datetime64[D]"),
            np.zeros(2),
        ),
        (
            np.zeros(2),
            np.full(2, np.timedelta64(1, "s"), dtype="timedelta64[s]"),
        ),
        (["1", "2"], np.zeros(2)),
        ([float("nan"), 0.0], np.zeros(2)),
        (np.zeros(2), [0.0, float("inf")]),
    ],
)
def test_status_rejects_malformed_or_nonfinite_state(velocity, acceleration) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_status(Command.LEFT, velocity, acceleration)


@pytest.mark.parametrize(
    ("velocity", "acceleration"),
    [([3.5e38, 0.0], [0.0, 0.0]), ([0.0, 0.0], [0.0, -3.5e38])],
)
def test_status_rechecks_finite_values_after_float32_cast(
    velocity, acceleration
) -> None:
    with pytest.raises(ValueError, match="float32"):
        build_status(Command.RIGHT, velocity, acceleration)


def test_status_normalizes_python_integer_float_overflow_to_value_error() -> None:
    velocity = np.asarray([10**400, 0], dtype=object)

    with pytest.raises(ValueError):
        build_status(Command.LEFT, velocity, np.zeros(2))
