from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from numbers import Real
from types import MappingProxyType
from typing import Any, Tuple

import numpy as np

from ..config.models import ResolvedDomain
from ..errors import ConfigurationError, MoraiDDV2Error
from .geometry import Pose, relative_se2


class Command(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    UNKNOWN = 3


class InvalidCommand(MoraiDDV2Error):
    """Raised when no command can be produced from an invalid trajectory."""

    drop_reason = "INVALID_COMMAND"


_EXPECTED_INDEX_MAPPING = {
    "LEFT": 0,
    "STRAIGHT": 1,
    "RIGHT": 2,
    "UNKNOWN": 3,
}
_EXPECTED_ORDERED_RULES = (
    "DROP_INVALID_TRAJECTORY",
    "UNKNOWN_STATIC",
    "LEFT",
    "RIGHT",
    "STRAIGHT",
    "UNKNOWN_AMBIGUOUS",
)


def _configuration_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("{} must be a mapping".format(name))
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set, name: str
) -> None:
    if set(value.keys()) != expected:
        raise ConfigurationError(
            "{} must contain exactly {}".format(name, sorted(expected))
        )


def _validated_index_mapping(value: Any) -> Mapping[str, int]:
    mapping = _configuration_mapping(value, "command.index_mapping")
    _require_exact_keys(mapping, set(_EXPECTED_INDEX_MAPPING), "command.index_mapping")
    for name, expected_index in _EXPECTED_INDEX_MAPPING.items():
        actual_index = mapping[name]
        if type(actual_index) is not int or actual_index != expected_index:
            raise ConfigurationError(
                "command.index_mapping must be LEFT=0, STRAIGHT=1, RIGHT=2, UNKNOWN=3"
            )
    return MappingProxyType(dict(mapping))


def _validated_ordered_rules(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("command.ordered_rules must be a sequence")
    ordered_rules = tuple(value)
    if ordered_rules != _EXPECTED_ORDERED_RULES:
        raise ConfigurationError(
            "command.ordered_rules does not match the versioned command contract"
        )
    return ordered_rules


def _validated_threshold(value: Any, expected: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConfigurationError("{} must be a finite number".format(name)) from error
    if not math.isfinite(result):
        raise ConfigurationError("{} must be a finite number".format(name))
    if result != expected:
        raise ConfigurationError(
            "{} does not match the versioned command contract".format(name)
        )
    return result


@dataclass(frozen=True)
class CommandRules:
    """Immutable, fully specified oracle-command classification contract."""

    index_mapping: Mapping[str, int]
    ordered_rules: Tuple[str, ...]
    minimum_path_length_m: float
    left_minimum_final_heading_rad: float
    left_minimum_final_lateral_m: float
    right_maximum_final_heading_rad: float
    right_maximum_final_lateral_m: float
    straight_maximum_abs_final_heading_rad: float
    straight_maximum_abs_final_lateral_m: float
    ambiguous_or_static_class: str
    nan_allowed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index_mapping", _validated_index_mapping(self.index_mapping)
        )
        object.__setattr__(
            self, "ordered_rules", _validated_ordered_rules(self.ordered_rules)
        )
        threshold_contract = (
            ("minimum_path_length_m", 1.0),
            ("left_minimum_final_heading_rad", 0.2),
            ("left_minimum_final_lateral_m", 0.5),
            ("right_maximum_final_heading_rad", -0.2),
            ("right_maximum_final_lateral_m", -0.5),
            ("straight_maximum_abs_final_heading_rad", 0.1),
            ("straight_maximum_abs_final_lateral_m", 0.5),
        )
        for field_name, expected in threshold_contract:
            object.__setattr__(
                self,
                field_name,
                _validated_threshold(
                    getattr(self, field_name),
                    expected,
                    "command.thresholds.{}".format(field_name),
                ),
            )
        if self.ambiguous_or_static_class != "UNKNOWN":
            raise ConfigurationError(
                "command.thresholds.ambiguous_or_static_class must be UNKNOWN"
            )
        if self.nan_allowed is not False:
            raise ConfigurationError("command.thresholds.nan_allowed must be false")

    @classmethod
    def from_resolved_domain(cls, value: Any) -> "CommandRules":
        """Build rules from a resolved document or its canonical command subtree."""

        if isinstance(value, ResolvedDomain):
            document = value.document
        else:
            document = _configuration_mapping(value, "resolved domain")

        if "command" in document:
            command = _configuration_mapping(document["command"], "command")
        else:
            command = document

        required_command_keys = {"index_mapping", "thresholds", "ordered_rules"}
        actual_command_keys = set(command.keys())
        allowed_with_provenance = required_command_keys | {"provenance"}
        if actual_command_keys not in (
            required_command_keys,
            allowed_with_provenance,
        ):
            raise ConfigurationError(
                "command must contain exactly index_mapping, thresholds, ordered_rules"
            )
        if "provenance" in command and command["provenance"] != "future_trajectory_oracle":
            raise ConfigurationError(
                "command.provenance must be future_trajectory_oracle"
            )

        thresholds = _configuration_mapping(command["thresholds"], "command.thresholds")
        _require_exact_keys(
            thresholds,
            {
                "minimum_path_length_m",
                "left",
                "right",
                "straight",
                "ambiguous_or_static_class",
                "nan_allowed",
            },
            "command.thresholds",
        )
        left = _configuration_mapping(thresholds["left"], "command.thresholds.left")
        right = _configuration_mapping(thresholds["right"], "command.thresholds.right")
        straight = _configuration_mapping(
            thresholds["straight"], "command.thresholds.straight"
        )
        _require_exact_keys(
            left,
            {"minimum_final_heading_rad", "minimum_final_lateral_m"},
            "command.thresholds.left",
        )
        _require_exact_keys(
            right,
            {"maximum_final_heading_rad", "maximum_final_lateral_m"},
            "command.thresholds.right",
        )
        _require_exact_keys(
            straight,
            {
                "maximum_abs_final_heading_rad",
                "maximum_abs_final_lateral_m",
            },
            "command.thresholds.straight",
        )

        return cls(
            index_mapping=command["index_mapping"],
            ordered_rules=_validated_ordered_rules(command["ordered_rules"]),
            minimum_path_length_m=thresholds["minimum_path_length_m"],
            left_minimum_final_heading_rad=left["minimum_final_heading_rad"],
            left_minimum_final_lateral_m=left["minimum_final_lateral_m"],
            right_maximum_final_heading_rad=right["maximum_final_heading_rad"],
            right_maximum_final_lateral_m=right["maximum_final_lateral_m"],
            straight_maximum_abs_final_heading_rad=straight[
                "maximum_abs_final_heading_rad"
            ],
            straight_maximum_abs_final_lateral_m=straight[
                "maximum_abs_final_lateral_m"
            ],
            ambiguous_or_static_class=thresholds["ambiguous_or_static_class"],
            nan_allowed=thresholds["nan_allowed"],
        )


def build_relative_trajectory(
    current_pose: Pose, future_poses: Sequence[Pose]
) -> np.ndarray:
    """Return eight future poses relative to the current base frame."""

    if not isinstance(current_pose, Pose):
        raise TypeError("current_pose must be a Pose")
    if isinstance(future_poses, (str, bytes)) or not isinstance(
        future_poses, Sequence
    ):
        raise TypeError("future_poses must be a sequence")
    if len(future_poses) != 8:
        raise ValueError("future_poses must contain exactly 8 Pose values")

    relative_values = []
    for index, future_pose in enumerate(future_poses):
        if not isinstance(future_pose, Pose):
            raise TypeError("future_poses[{}] must be a Pose".format(index))
        try:
            relative_values.append(relative_se2(current_pose, future_pose))
        except OverflowError as error:
            raise ValueError("pose values overflow finite float64") from error

    try:
        trajectory64 = np.asarray(relative_values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("relative trajectory must contain finite numbers") from error
    if trajectory64.shape != (8, 3) or not np.isfinite(trajectory64).all():
        raise ValueError("relative trajectory must be finite with shape [8,3]")
    with np.errstate(over="ignore", invalid="ignore"):
        trajectory32 = np.ascontiguousarray(trajectory64, dtype=np.float32)
    if not np.isfinite(trajectory32).all():
        raise ValueError("relative trajectory overflows finite float32")
    return trajectory32


def _numeric_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    try:
        unconverted = np.asarray(value)
    except OverflowError as error:
        raise ValueError("{} values overflow finite float64".format(name)) from error
    except (TypeError, ValueError) as error:
        raise TypeError("{} must contain numeric values".format(name)) from error
    if unconverted.dtype.kind == "O":
        for item in unconverted.flat:
            invalid_scalar_type = isinstance(
                item, (bool, np.bool_, complex, np.complexfloating)
            )
            if invalid_scalar_type or not isinstance(item, Real):
                raise TypeError("{} must contain real numeric values".format(name))
            try:
                numeric_item = float(item)
            except OverflowError as error:
                raise ValueError(
                    "{} values overflow finite float64".format(name)
                ) from error
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "{} must contain real numeric values".format(name)
                ) from error
            if not math.isfinite(numeric_item):
                raise ValueError("{} must contain only finite values".format(name))
    elif unconverted.dtype.kind not in ("i", "u", "f"):
        raise TypeError("{} must contain real numeric values".format(name))
    try:
        result = np.asarray(value, dtype=np.float64)
    except OverflowError as error:
        raise ValueError("{} values overflow finite float64".format(name)) from error
    except (TypeError, ValueError) as error:
        raise TypeError("{} must contain numeric values".format(name)) from error
    if result.shape != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    if not np.isfinite(result).all():
        raise ValueError("{} must contain only finite values".format(name))
    return result


def classify_oracle_command(
    trajectory: np.ndarray, rules: CommandRules
) -> Command:
    """Classify a validated future target using the versioned ordered rules."""

    if not isinstance(rules, CommandRules):
        raise TypeError("rules must be a CommandRules instance")
    try:
        target = _numeric_array(trajectory, (8, 3), "trajectory")
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidCommand("trajectory must be finite numeric [8,3]") from error

    xy = np.vstack((np.zeros((1, 2), dtype=np.float64), target[:, :2]))
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = np.diff(xy, axis=0)
        segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        path_length = float(np.sum(segment_lengths, dtype=np.float64))
    if not math.isfinite(path_length):
        raise InvalidCommand("trajectory path length must be finite")

    final_y = float(target[7, 1])
    final_heading = float(target[7, 2])
    if path_length < rules.minimum_path_length_m:
        return Command.UNKNOWN
    if (
        final_heading >= rules.left_minimum_final_heading_rad
        and final_y >= rules.left_minimum_final_lateral_m
    ):
        return Command.LEFT
    if (
        final_heading <= rules.right_maximum_final_heading_rad
        and final_y <= rules.right_maximum_final_lateral_m
    ):
        return Command.RIGHT
    if (
        abs(final_heading) <= rules.straight_maximum_abs_final_heading_rad
        and abs(final_y) <= rules.straight_maximum_abs_final_lateral_m
    ):
        return Command.STRAIGHT
    return Command.UNKNOWN


def build_status(
    command: Command, velocity_xy: np.ndarray, acceleration_xy: np.ndarray
) -> np.ndarray:
    """Build the raw body-frame status vector expected by DiffusionDriveV2."""

    if type(command) is not Command:
        raise TypeError("command must be an actual Command enum value")
    velocity = _numeric_array(velocity_xy, (2,), "velocity_xy")
    acceleration = _numeric_array(acceleration_xy, (2,), "acceleration_xy")

    status64 = np.zeros(8, dtype=np.float64)
    status64[int(command)] = 1.0
    status64[4:6] = velocity
    status64[6:8] = acceleration
    with np.errstate(over="ignore", invalid="ignore"):
        status32 = np.ascontiguousarray(status64, dtype=np.float32)
    if not np.isfinite(status32).all():
        raise ValueError("status values overflow finite float32")
    return status32
