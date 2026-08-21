"""Stream a ROS1 bag into a compact DiffusionDriveV2 HDF5 dataset."""

from __future__ import annotations

import argparse
import math
import os
from bisect import bisect_right
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np

from .trajectory import EgoState, FUTURE_FRAME_OFFSETS, build_target_trajectory


CAMERA_ROLES = ("front", "left", "right")
DEFAULT_CAMERA_TOPICS = {
    "front": "/image_jpeg/front/compressed",
    "left": "/image_jpeg/left/compressed",
    "right": "/image_jpeg/right/compressed",
}
DEFAULT_LIDAR_TOPIC = "/lidar3D"
DEFAULT_EGO_TOPIC = "/Ego_topic"


def _stamp_ns(message: object, topic: str) -> int:
    try:
        stamp = message.header.stamp  # type: ignore[attr-defined]
        if hasattr(stamp, "to_nsec"):
            value = int(stamp.to_nsec())
        else:
            value = int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("{} message requires a valid Header stamp".format(topic)) from error
    if value < 0:
        raise ValueError("{} Header stamp must be non-negative".format(topic))
    return value


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be numeric".format(name)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _ego_state(message: object, stamp_ns: int) -> EgoState:
    try:
        position = message.position  # type: ignore[attr-defined]
        velocity = message.velocity  # type: ignore[attr-defined]
        acceleration = message.acceleration  # type: ignore[attr-defined]
        heading = message.heading  # type: ignore[attr-defined]
        return EgoState(
            timestamp_ns=stamp_ns,
            x=_finite(position.x, "position.x"),
            y=_finite(position.y, "position.y"),
            heading_rad=math.radians(_finite(heading, "heading")),
            vx=_finite(velocity.x, "velocity.x"),
            vy=_finite(velocity.y, "velocity.y"),
            ax=_finite(acceleration.x, "acceleration.x"),
            ay=_finite(acceleration.y, "acceleration.y"),
        )
    except AttributeError as error:
        raise ValueError("Ego message is missing a required field") from error


def _rosbag_module():
    try:
        import rosbag
    except ImportError as error:
        raise RuntimeError(
            "ROS1 rosbag is unavailable; run from the sourced ROS preprocessing environment"
        ) from error
    return rosbag


def _require_topics(bag: object, required: Sequence[str]) -> None:
    info = bag.get_type_and_topic_info()  # type: ignore[attr-defined]
    topics = info.topics if hasattr(info, "topics") else info[1]
    missing = sorted(set(required) - set(topics))
    if missing:
        raise ValueError("bag is missing required topic(s): {}".format(", ".join(missing)))


def _read_ego_states(
    bag_path: Path,
    ego_topic: str,
    required_topics: Sequence[str],
    bag_hz: float,
) -> Tuple[EgoState, ...]:
    rosbag = _rosbag_module()
    states = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        _require_topics(bag, required_topics)
        for _, message, _ in bag.read_messages(topics=[ego_topic]):
            stamp_ns = _stamp_ns(message, ego_topic)
            if states and stamp_ns == states[-1].timestamp_ns:
                continue
            if states and stamp_ns < states[-1].timestamp_ns:
                raise ValueError("Ego Header stamps must be monotonic")
            states.append(_ego_state(message, stamp_ns))

    if len(states) <= FUTURE_FRAME_OFFSETS[-1]:
        raise ValueError("bag requires at least 41 unique Ego frames")
    expected_ns = int(round(1_000_000_000.0 / bag_hz))
    tolerance_ns = max(1_000_000, expected_ns // 4)
    deltas = np.diff(np.asarray([state.timestamp_ns for state in states], dtype=np.int64))
    median_delta_ns = int(np.median(deltas))
    if abs(median_delta_ns - expected_ns) > tolerance_ns:
        raise ValueError(
            "median Ego Header cadence is not near {:.6g} Hz "
            "({:.6f} s); set --bag-hz to the recording rate".format(
                bag_hz, float(median_delta_ns) / 1_000_000_000.0
            )
        )
    return tuple(states)


def _create_datasets(output: h5py.File) -> Dict[str, h5py.Dataset]:
    shapes = {
        "camera": ((3, 256, 1024), np.uint8),
        "lidar": ((1, 256, 256), np.uint8),
        "status": ((8,), np.float32),
        "trajectory": ((8, 3), np.float32),
        "timestamp_ns": ((), np.int64),
        "future_ego_stamp_ns": ((8,), np.int64),
    }
    return {
        name: output.create_dataset(
            name,
            shape=(0,) + shape,
            maxshape=(None,) + shape,
            chunks=(1,) + shape,
            dtype=dtype,
        )
        for name, (shape, dtype) in shapes.items()
    }


def _append(datasets: Dict[str, h5py.Dataset], values: Dict[str, np.ndarray]) -> None:
    index = datasets["camera"].shape[0]
    for name, dataset in datasets.items():
        dataset.resize(index + 1, axis=0)
        dataset[index] = values[name]


def _quantize_unit(value: np.ndarray, scale: int, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("{} preprocessing returned invalid shape or values".format(name))
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("{} preprocessing returned values outside [0,1]".format(name))
    return np.rint(array * scale).astype(np.uint8)


def extract_bag(
    bag_path: Path,
    output_path: Path,
    sensor_config: Path,
    *,
    camera_topics: Optional[Dict[str, str]] = None,
    lidar_topic: str = DEFAULT_LIDAR_TOPIC,
    ego_topic: str = DEFAULT_EGO_TOPIC,
    sync_slop: float = 0.1,
    bag_hz: float = 10.0,
    max_samples: Optional[int] = None,
) -> int:
    """Extract synchronized LiDAR-anchored samples without loading the bag."""
    bag_path, output_path, sensor_config = map(Path, (bag_path, output_path, sensor_config))
    camera_topics = dict(DEFAULT_CAMERA_TOPICS if camera_topics is None else camera_topics)
    if not bag_path.is_file():
        raise FileNotFoundError("bag does not exist: {}".format(bag_path))
    if not sensor_config.is_file():
        raise FileNotFoundError("sensor config does not exist: {}".format(sensor_config))
    if not output_path.parent.is_dir():
        raise FileNotFoundError("output directory does not exist: {}".format(output_path.parent))
    if set(camera_topics) != set(CAMERA_ROLES) or any(
        not isinstance(topic, str) or not topic for topic in camera_topics.values()
    ):
        raise ValueError("camera_topics must define nonempty front, left, and right topics")
    all_topics = tuple(camera_topics[role] for role in CAMERA_ROLES) + (lidar_topic, ego_topic)
    if any(not isinstance(topic, str) or not topic for topic in all_topics) or len(set(all_topics)) != 5:
        raise ValueError("all configured topics must be nonempty and distinct")
    if not math.isfinite(sync_slop) or sync_slop < 0.0:
        raise ValueError("sync_slop must be finite and non-negative")
    if not math.isfinite(bag_hz) or bag_hz <= 0.0:
        raise ValueError("bag_hz must be finite and positive")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples must be a positive integer")

    partial_path = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial_path.exists():
        raise FileExistsError("refusing to overwrite output or partial file")

    ego_states = _read_ego_states(bag_path, ego_topic, all_topics, bag_hz)
    ego_stamps = [state.timestamp_ns for state in ego_states]
    sync_slop_ns = int(round(sync_slop * 1_000_000_000.0))

    from .validator.image_adaptor import (
        ImageAdaptor,
        prepare_diffusiondrive_v2 as prepare_camera,
    )
    from .validator.lidar_bev_processor import (
        decode_point_cloud,
        load_lidar_calibration,
        prepare_diffusiondrive_v2 as prepare_lidar,
        sensor_to_vehicle,
    )
    from .validator.opt_features import build_driving_feature_data

    calibration = load_lidar_calibration(str(sensor_config), lidar_topic)
    camera_role_by_topic = {topic: role for role, topic in camera_topics.items()}
    sensor_topics = tuple(camera_role_by_topic) + (lidar_topic,)
    buffers = {role: deque() for role in CAMERA_ROLES}
    last_stamp = {topic: None for topic in sensor_topics}
    count = 0
    created_partial = False

    try:
        output = h5py.File(str(partial_path), "x")
        created_partial = True
        with output:
            datasets = _create_datasets(output)
            output.attrs["bag_hz"] = bag_hz
            output.attrs["camera_topics"] = np.asarray(
                [camera_topics[role] for role in CAMERA_ROLES], dtype=h5py.string_dtype()
            )
            output.attrs["lidar_topic"] = lidar_topic
            output.attrs["ego_topic"] = ego_topic

            rosbag = _rosbag_module()
            with rosbag.Bag(str(bag_path), "r") as bag:
                for topic, message, _ in bag.read_messages(topics=list(sensor_topics)):
                    stamp_ns = _stamp_ns(message, topic)
                    previous = last_stamp[topic]
                    if previous is not None and stamp_ns < previous:
                        raise ValueError("{} Header stamps must be monotonic".format(topic))
                    last_stamp[topic] = stamp_ns

                    role = camera_role_by_topic.get(topic)
                    if role is not None:
                        buffers[role].append((stamp_ns, message))
                        continue

                    selected = {}
                    for camera_role in CAMERA_ROLES:
                        buffer = buffers[camera_role]
                        while len(buffer) > 1 and buffer[1][0] <= stamp_ns:
                            buffer.popleft()
                        if not buffer or buffer[0][0] > stamp_ns:
                            break
                        selected[camera_role] = buffer[0]
                    if len(selected) != 3:
                        continue
                    camera_stamps = [selected[camera_role][0] for camera_role in CAMERA_ROLES]
                    if stamp_ns - min(camera_stamps) > sync_slop_ns or (
                        max(camera_stamps) - min(camera_stamps) > sync_slop_ns
                    ):
                        continue

                    current_index = bisect_right(ego_stamps, stamp_ns) - 1
                    if current_index < 0 or (
                        current_index + FUTURE_FRAME_OFFSETS[-1] >= len(ego_states)
                    ):
                        continue
                    current = ego_states[current_index]
                    images = {
                        camera_role: ImageAdaptor.decode_compressed_image(
                            selected[camera_role][1]
                        )
                        for camera_role in CAMERA_ROLES
                    }
                    if any(image is None for image in images.values()):
                        raise ValueError("camera JPEG decode failed")
                    camera_hwc = prepare_camera(images, morai_domain=True)
                    camera = _quantize_unit(
                        np.transpose(camera_hwc, (2, 0, 1)),
                        255,
                        (3, 256, 1024),
                        "camera",
                    )

                    points = sensor_to_vehicle(decode_point_cloud(message), calibration)
                    lidar = _quantize_unit(
                        prepare_lidar(points, morai_domain=True)[None],
                        5,
                        (1, 256, 256),
                        "lidar",
                    )
                    status = np.asarray(
                        build_driving_feature_data(
                            "diffusiondrive_v2",
                            vx=current.vx,
                            vy=current.vy,
                            ax=current.ax,
                            ay=current.ay,
                            command_index=1,
                            source="rosbag",
                        ).status_feature,
                        dtype=np.float32,
                    )
                    future_stamps = np.asarray(
                        [
                            ego_states[current_index + offset].timestamp_ns
                            for offset in FUTURE_FRAME_OFFSETS
                        ],
                        dtype=np.int64,
                    )
                    _append(
                        datasets,
                        {
                            "camera": camera,
                            "lidar": lidar,
                            "status": status,
                            "trajectory": build_target_trajectory(ego_states, current_index),
                            "timestamp_ns": np.asarray(stamp_ns, dtype=np.int64),
                            "future_ego_stamp_ns": future_stamps,
                        },
                    )
                    count += 1
                    if max_samples is not None and count >= max_samples:
                        break

            if count == 0:
                raise ValueError("bag produced no synchronized samples")
            output.flush()

        with partial_path.open("rb") as partial_file:
            os.fsync(partial_file.fileno())
        if output_path.exists():
            raise FileExistsError("refusing to overwrite output created during extraction")
        os.replace(str(partial_path), str(output_path))
        created_partial = False
        return count
    finally:
        if created_partial and partial_path.exists():
            partial_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sensor-config", required=True, type=Path)
    parser.add_argument("--front-topic", default=DEFAULT_CAMERA_TOPICS["front"])
    parser.add_argument("--left-topic", default=DEFAULT_CAMERA_TOPICS["left"])
    parser.add_argument("--right-topic", default=DEFAULT_CAMERA_TOPICS["right"])
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--ego-topic", default=DEFAULT_EGO_TOPIC)
    parser.add_argument("--sync-slop", default=0.1, type=float, help="causal sensor slop in seconds")
    parser.add_argument("--bag-hz", default=10.0, type=float, help="recorded Ego frame rate")
    parser.add_argument("--max-samples", type=int, help="stop after N samples (smoke test)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    count = extract_bag(
        args.bag,
        args.output,
        args.sensor_config,
        camera_topics={
            "front": args.front_topic,
            "left": args.left_topic,
            "right": args.right_topic,
        },
        lidar_topic=args.lidar_topic,
        ego_topic=args.ego_topic,
        sync_slop=args.sync_slop,
        bag_hz=args.bag_hz,
        max_samples=args.max_samples,
    )
    print("wrote {} samples to {}".format(count, args.output))


if __name__ == "__main__":
    main()
