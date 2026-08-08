from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from morai_ddv2.preprocess.trajectory import Command


INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

CACHE_DATASET_NAMES = (
    "camera_rgb",
    "lidar_count",
    "status",
    "trajectory",
    "timestamp_ns",
    "grid_receipt_ns",
    "sensor_source_header_stamp_ns",
    "sensor_source_receipt_ns",
    "sensor_source_record_index",
    "side_camera_header_offset_ns",
    "sensor_receipt_offset_ns",
    "trajectory_anchor_tf_header_ns",
    "trajectory_anchor_tf_receipt_ns",
    "trajectory_anchor_tf_record_index",
    "trajectory_anchor_tf_clock_epoch_index",
    "ego_bracket_clock_epoch_index",
    "ego_bracket_header_stamp_ns",
    "ego_bracket_receipt_ns",
    "ego_bracket_record_index",
    "ego_interp_weight",
    "current_pose_bracket_header_stamp_ns",
    "current_pose_bracket_receipt_ns",
    "current_pose_bracket_record_index",
    "current_pose_interp_weight",
    "future_pose_bracket_header_stamp_ns",
    "future_pose_bracket_receipt_ns",
    "future_pose_bracket_record_index",
    "future_pose_interp_weight",
)


def _python_int(value: Any, name: str, *, nonnegative: bool) -> int:
    if type(value) is not int:
        raise TypeError("{} must be an actual Python int".format(name))
    if value < INT64_MIN or value > INT64_MAX:
        raise OverflowError("{} is outside signed int64 range".format(name))
    if nonnegative and value < 0:
        raise ValueError("{} must be nonnegative".format(name))
    return value


def _int_tuple(
    value: Any, name: str, length: int, *, nonnegative: bool
) -> Tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError("{} must be a tuple".format(name))
    if len(value) != length:
        raise ValueError("{} must have length {}".format(name, length))
    return tuple(
        _python_int(item, "{}[{}]".format(name, index), nonnegative=nonnegative)
        for index, item in enumerate(value)
    )


def _owned_array(
    value: Any,
    name: str,
    *,
    dtype: np.dtype,
    shape: Tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("{} must be a NumPy array".format(name))
    if value.dtype != dtype:
        raise TypeError("{} must have dtype {}".format(name, dtype))
    if value.shape != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    if not value.flags.c_contiguous:
        raise ValueError("{} must be C-contiguous".format(name))
    result = value.copy(order="C")
    result.setflags(write=False)
    return result


def _float32_scalar(value: Any, name: str) -> float:
    if type(value) is not float:
        raise TypeError("{} must be an actual Python float".format(name))
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.float32(value)
    if not np.isfinite(converted):
        raise ValueError("{} must be finite after float32 conversion".format(name))
    result = float(converted)
    if result != value:
        raise ValueError("{} must be exactly float32-representable".format(name))
    if result < 0.0 or result > 1.0:
        raise ValueError("{} must be in [0, 1]".format(name))
    return result


def _is_positive_zero(value: float) -> bool:
    return value == 0.0 and not bool(np.signbit(value))


def _validate_bracket(
    header: Tuple[int, int],
    receipt: Tuple[int, int],
    record_index: Tuple[int, int],
    weight: float,
    name: str,
) -> None:
    if header[0] > header[1]:
        raise ValueError("{} Header stamps must be nondecreasing".format(name))
    if receipt[0] > receipt[1]:
        raise ValueError("{} receipt stamps must preserve source order".format(name))
    if record_index[0] > record_index[1]:
        raise ValueError("{} record indices must preserve source order".format(name))
    if header[0] == header[1]:
        if receipt[0] != receipt[1] or record_index[0] != record_index[1]:
            raise ValueError("{} exact bracket must repeat one record".format(name))
        if not _is_positive_zero(weight):
            raise ValueError("{} exact bracket must have positive zero weight".format(name))


@dataclass(frozen=True)
class CacheSample:
    camera_rgb: np.ndarray
    lidar_count: np.ndarray
    status: np.ndarray
    trajectory: np.ndarray
    timestamp_ns: int
    grid_receipt_ns: int
    sensor_source_header_stamp_ns: Tuple[int, int, int, int]
    sensor_source_receipt_ns: Tuple[int, int, int, int]
    sensor_source_record_index: Tuple[int, int, int, int]
    side_camera_header_offset_ns: Tuple[int, int]
    sensor_receipt_offset_ns: Tuple[int, int, int]
    trajectory_anchor_tf_header_ns: int
    trajectory_anchor_tf_receipt_ns: int
    trajectory_anchor_tf_record_index: int
    trajectory_anchor_tf_clock_epoch_index: int
    ego_bracket_clock_epoch_index: int
    ego_bracket_header_stamp_ns: Tuple[int, int]
    ego_bracket_receipt_ns: Tuple[int, int]
    ego_bracket_record_index: Tuple[int, int]
    ego_interp_weight: float
    current_pose_bracket_header_stamp_ns: Tuple[int, int]
    current_pose_bracket_receipt_ns: Tuple[int, int]
    current_pose_bracket_record_index: Tuple[int, int]
    current_pose_interp_weight: float
    future_pose_bracket_header_stamp_ns: np.ndarray
    future_pose_bracket_receipt_ns: np.ndarray
    future_pose_bracket_record_index: np.ndarray
    future_pose_interp_weight: np.ndarray
    command: Command

    def __post_init__(self) -> None:
        array_contract = (
            ("camera_rgb", np.dtype(np.uint8), (256, 1024, 3)),
            ("lidar_count", np.dtype(np.uint8), (256, 256)),
            ("status", np.dtype(np.float32), (8,)),
            ("trajectory", np.dtype(np.float32), (8, 3)),
            (
                "future_pose_bracket_header_stamp_ns",
                np.dtype(np.int64),
                (8, 2),
            ),
            ("future_pose_bracket_receipt_ns", np.dtype(np.int64), (8, 2)),
            ("future_pose_bracket_record_index", np.dtype(np.int64), (8, 2)),
            ("future_pose_interp_weight", np.dtype(np.float32), (8,)),
        )
        for name, dtype, shape in array_contract:
            object.__setattr__(
                self,
                name,
                _owned_array(getattr(self, name), name, dtype=dtype, shape=shape),
            )

        for name in (
            "timestamp_ns",
            "grid_receipt_ns",
            "trajectory_anchor_tf_header_ns",
            "trajectory_anchor_tf_receipt_ns",
            "trajectory_anchor_tf_record_index",
            "trajectory_anchor_tf_clock_epoch_index",
            "ego_bracket_clock_epoch_index",
        ):
            object.__setattr__(
                self,
                name,
                _python_int(getattr(self, name), name, nonnegative=True),
            )

        tuple_contract = (
            ("sensor_source_header_stamp_ns", 4, True),
            ("sensor_source_receipt_ns", 4, True),
            ("sensor_source_record_index", 4, True),
            ("side_camera_header_offset_ns", 2, False),
            ("sensor_receipt_offset_ns", 3, False),
            ("ego_bracket_header_stamp_ns", 2, True),
            ("ego_bracket_receipt_ns", 2, True),
            ("ego_bracket_record_index", 2, True),
            ("current_pose_bracket_header_stamp_ns", 2, True),
            ("current_pose_bracket_receipt_ns", 2, True),
            ("current_pose_bracket_record_index", 2, True),
        )
        for name, length, nonnegative in tuple_contract:
            object.__setattr__(
                self,
                name,
                _int_tuple(
                    getattr(self, name), name, length, nonnegative=nonnegative
                ),
            )

        object.__setattr__(
            self,
            "ego_interp_weight",
            _float32_scalar(self.ego_interp_weight, "ego_interp_weight"),
        )
        object.__setattr__(
            self,
            "current_pose_interp_weight",
            _float32_scalar(
                self.current_pose_interp_weight, "current_pose_interp_weight"
            ),
        )
        if type(self.command) is not Command:
            raise TypeError("command must be a Command")

        if not np.isfinite(self.status).all():
            raise ValueError("status must be finite")
        if not np.isfinite(self.trajectory).all():
            raise ValueError("trajectory must be finite")
        if int(self.lidar_count.max()) > 5:
            raise ValueError("lidar_count values must be in [0, 5]")
        for name in (
            "future_pose_bracket_header_stamp_ns",
            "future_pose_bracket_receipt_ns",
            "future_pose_bracket_record_index",
        ):
            if bool(np.any(getattr(self, name) < 0)):
                raise ValueError("{} must be nonnegative".format(name))
        if not np.isfinite(self.future_pose_interp_weight).all():
            raise ValueError("future_pose_interp_weight must be finite")
        if bool(
            np.any(self.future_pose_interp_weight < 0.0)
            or np.any(self.future_pose_interp_weight > 1.0)
        ):
            raise ValueError("future_pose_interp_weight must be in [0, 1]")

        headers = self.sensor_source_header_stamp_ns
        receipts = self.sensor_source_receipt_ns
        if self.timestamp_ns != headers[0]:
            raise ValueError("timestamp_ns must equal the front Header stamp")
        if self.side_camera_header_offset_ns != (
            headers[1] - headers[0],
            headers[2] - headers[0],
        ):
            raise ValueError("side_camera_header_offset_ns is inconsistent")
        if self.sensor_receipt_offset_ns != (
            receipts[1] - receipts[0],
            receipts[2] - receipts[0],
            receipts[3] - receipts[0],
        ):
            raise ValueError("sensor_receipt_offset_ns is inconsistent")

        anchor_header = self.trajectory_anchor_tf_header_ns
        anchor_receipt = self.trajectory_anchor_tf_receipt_ns
        anchor_index = self.trajectory_anchor_tf_record_index
        if self.current_pose_bracket_header_stamp_ns != (
            anchor_header,
            anchor_header,
        ):
            raise ValueError("current pose Header provenance must repeat raw anchor")
        if self.current_pose_bracket_receipt_ns != (
            anchor_receipt,
            anchor_receipt,
        ):
            raise ValueError("current pose receipt provenance must repeat raw anchor")
        if self.current_pose_bracket_record_index != (anchor_index, anchor_index):
            raise ValueError("current pose record provenance must repeat raw anchor")
        if not _is_positive_zero(self.current_pose_interp_weight):
            raise ValueError("current pose weight must be positive zero")

        _validate_bracket(
            self.ego_bracket_header_stamp_ns,
            self.ego_bracket_receipt_ns,
            self.ego_bracket_record_index,
            self.ego_interp_weight,
            "ego bracket",
        )
        future_headers = self.future_pose_bracket_header_stamp_ns
        future_receipts = self.future_pose_bracket_receipt_ns
        future_indices = self.future_pose_bracket_record_index
        future_weights = self.future_pose_interp_weight
        for index in range(8):
            _validate_bracket(
                (int(future_headers[index, 0]), int(future_headers[index, 1])),
                (int(future_receipts[index, 0]), int(future_receipts[index, 1])),
                (int(future_indices[index, 0]), int(future_indices[index, 1])),
                float(future_weights[index]),
                "future bracket {}".format(index),
            )

        one_hot = self.status[:4]
        expected = np.zeros(4, dtype=np.float32)
        expected[int(self.command)] = np.float32(1.0)
        if not np.array_equal(one_hot, expected):
            raise ValueError("status command one-hot does not agree with command")
