#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute field obstacle-avoidance metrics from ROS 2 MCAP rosbags.

Prepared for the Navigation2 agricultural truck paper.

This script computes, for each real-world run:
    - duration
    - traveled distance
    - mean and median speed
    - lateral RMSE with respect to a local straight reference line
    - heading RMSE with respect to the same local reference line
    - jerk RMS using multiple calculation variants
    - minimum footprint-to-occupied-cell clearance using synchronized samples
      from the global obstacle layer

Recommended topics:
    Odometry:
        /odometry/global

    Obstacle source:
        /global_costmap/obstacle_layer

Why multiple jerk values?
    Jerk obtained by finite differences is very sensitive to sampling jitter,
    noisy velocity, localization jumps, and differentiation method. Therefore,
    the script reports several variants:
        1. raw jerk from odometry twist.linear.x
        2. raw jerk from position-derived longitudinal speed
        3. filtered jerk from resampled and smoothed longitudinal position

The most defensible value for the paper is usually:
    jerk_rms_posproj_filtered_mps3

because it uses a uniform time grid and smoothing before differentiation.

Example command:

/usr/bin/python3 ~/NAV2_Paper_Scripts/recompute_field_metrics_from_bags.py \
  --bag "Nav2 + NavFn:$HOME/NAV2_Paper_Scripts/tarde/NAVFN_desvio_obstaculo/giro_nova_escala_ultimo_dia_NAVFN_0.mcap" \
  --bag "Human driver:$HOME/NAV2_Paper_Scripts/tarde/reta_manual/reta_manual_0.mcap" \
  --bag "Nav2 + SMAC:$HOME/NAV2_Paper_Scripts/tarde/SMAC_MPPI_desvio_de_obstaculo_correto/desvio_de_obstaculo_correto_0.mcap" \
  --odom-topic /odometry/global \
  --obstacle-topic /global_costmap/obstacle_layer \
  --nav2-params /path/to/truck_nav2_params.yaml \
  --output-csv ~/NAV2_Paper_Scripts/tarde/field_metrics_recomputed.csv \
  --output-xlsx ~/NAV2_Paper_Scripts/tarde/field_metrics_recomputed.xlsx
"""

import argparse
import ast
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

ROS_IMPORT_ERROR = None

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception as exc:
    rosbag2_py = None
    deserialize_message = None
    get_message = None
    ROS_IMPORT_ERROR = exc


# Verified from truck_nav2_params.yaml.  The footprint includes the 0.93 m
# transverse LiDAR bar: 2.548 m body width + 0.930 m bar extension = 3.478 m.
DEFAULT_NAV2_FOOTPRINT = np.array(
    [
        [-4.8025, -1.7390],
        [4.8025, -1.7390],
        [4.8025, 1.7390],
        [-4.8025, 1.7390],
    ],
    dtype=float,
)

CLEARANCE_DEFINITION = (
    "minimum Euclidean distance between the oriented Navigation2 footprint "
    "polygon and the occupied OccupancyGrid cell polygons, using the nearest "
    "time-synchronized grid/odometry samples in a common frame"
)


@dataclass
class OdomSample:
    stamp_sec: float
    frame_id: str
    child_frame_id: str
    x: float
    y: float
    yaw: float
    twist_linear_x: float
    twist_linear_y: float
    twist_angular_z: float


@dataclass
class GridSample:
    stamp_sec: float
    frame_id: str
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: np.ndarray


@dataclass
class BagData:
    label: str
    path: str
    odom_samples: List[OdomSample]
    grid_samples: List[GridSample]


@dataclass
class ClearanceResult:
    footprint_to_cell_area_m: float
    footprint_to_cell_center_m: float
    base_to_cell_center_legacy_m: float
    closest_pose_stamp_sec: float
    closest_grid_stamp_sec: float
    closest_time_delta_s: float
    closest_pose_x: float
    closest_pose_y: float
    closest_pose_yaw: float
    closest_obstacle_x: float
    closest_obstacle_y: float
    closest_grid_resolution_m: float
    grids_evaluated: int
    pose_grid_pairs_evaluated: int
    obstacle_cells_evaluated: int
    odom_frame: str
    obstacle_frame: str


def stamp_to_sec(stamp_msg) -> float:
    return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9


def normalize_frame_name(frame_name: str) -> str:
    if frame_name is None:
        return ""
    return str(frame_name).strip().lstrip("/")


def quaternion_to_yaw(q) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def infer_storage_id(path: str) -> str:
    lower = path.lower()

    if lower.endswith(".mcap"):
        return "mcap"

    if lower.endswith(".db3"):
        return "sqlite3"

    metadata_path = os.path.join(path, "metadata.yaml")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                text = f.read().lower()
            if "mcap" in text:
                return "mcap"
            if "sqlite3" in text:
                return "sqlite3"
        except Exception:
            pass

    return "mcap"


def open_bag_reader(path: str):
    if rosbag2_py is None:
        raise RuntimeError(
            "ROS 2 Python libraries were not found. Run with /usr/bin/python3 "
            "after sourcing /opt/ros/iron/setup.bash (or the ROS distribution "
            f"used on this computer). Original error: {ROS_IMPORT_ERROR}"
        )

    storage_id = infer_storage_id(path)

    storage_options = rosbag2_py.StorageOptions(
        uri=path,
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    return reader


def build_topic_type_map(reader) -> Dict[str, str]:
    topics_and_types = reader.get_all_topics_and_types()
    return {topic.name: topic.type for topic in topics_and_types}


def odom_msg_to_sample(msg) -> OdomSample:
    pose = msg.pose.pose
    twist = msg.twist.twist

    return OdomSample(
        stamp_sec=stamp_to_sec(msg.header.stamp),
        frame_id=normalize_frame_name(msg.header.frame_id),
        child_frame_id=normalize_frame_name(msg.child_frame_id),
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=quaternion_to_yaw(pose.orientation),
        twist_linear_x=float(twist.linear.x),
        twist_linear_y=float(twist.linear.y),
        twist_angular_z=float(twist.angular.z),
    )


def occupancy_grid_msg_to_sample(msg) -> GridSample:
    origin = msg.info.origin

    data = np.asarray(msg.data, dtype=np.int16).reshape(
        int(msg.info.height),
        int(msg.info.width),
    )

    return GridSample(
        stamp_sec=stamp_to_sec(msg.header.stamp),
        frame_id=normalize_frame_name(msg.header.frame_id),
        resolution=float(msg.info.resolution),
        width=int(msg.info.width),
        height=int(msg.info.height),
        origin_x=float(origin.position.x),
        origin_y=float(origin.position.y),
        origin_yaw=quaternion_to_yaw(origin.orientation),
        data=data,
    )


def parse_bag_arg(value: str) -> Tuple[str, str]:
    if ":" not in value:
        path = os.path.expanduser(os.path.expandvars(value))
        label = os.path.basename(path.rstrip("/"))
        return label, path

    label, path = value.split(":", 1)
    label = label.strip()
    path = os.path.expanduser(os.path.expandvars(path.strip()))

    if not label:
        label = os.path.basename(path.rstrip("/"))

    return label, path


def parse_roi(values: Optional[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    if values is None:
        return None

    if len(values) != 4:
        raise ValueError("ROI requires four values: xmin xmax ymin ymax")

    xmin, xmax, ymin, ymax = map(float, values)

    if xmax <= xmin:
        raise ValueError("Invalid ROI: xmax must be greater than xmin")

    if ymax <= ymin:
        raise ValueError("Invalid ROI: ymax must be greater than ymin")

    return xmin, xmax, ymin, ymax


def polygon_signed_area(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=float)
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def validate_footprint(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
        raise ValueError("Footprint must be an N x 2 array with at least three vertices")

    if not np.all(np.isfinite(vertices)):
        raise ValueError("Footprint contains non-finite values")

    if np.linalg.norm(vertices[0] - vertices[-1]) <= 1e-12:
        vertices = vertices[:-1]

    if vertices.shape[0] < 3:
        raise ValueError("Footprint must contain at least three distinct vertices")

    area = polygon_signed_area(vertices)
    if abs(area) <= 1e-9:
        raise ValueError("Footprint polygon has zero area")

    # Geometry helpers below assume counter-clockwise vertex order.
    if area < 0.0:
        vertices = vertices[::-1].copy()

    cross_values = []
    for idx in range(vertices.shape[0]):
        p0 = vertices[idx]
        p1 = vertices[(idx + 1) % vertices.shape[0]]
        p2 = vertices[(idx + 2) % vertices.shape[0]]
        edge_a = p1 - p0
        edge_b = p2 - p1
        cross_values.append(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))

    if any(value < -1e-9 for value in cross_values):
        raise ValueError("Footprint must be a convex polygon with ordered vertices")

    return vertices


def parse_footprint_value(value: object) -> np.ndarray:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse footprint string: {value}") from exc

    return validate_footprint(np.asarray(parsed, dtype=float))


def _collect_named_values(node: object, key_name: str, path: str = "") -> List[Tuple[str, object]]:
    matches: List[Tuple[str, object]] = []

    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) == key_name:
                matches.append((child_path, value))
            matches.extend(_collect_named_values(value, key_name, child_path))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            matches.extend(_collect_named_values(value, key_name, f"{path}[{idx}]"))

    return matches


def load_footprint_from_nav2_params(path: str) -> Tuple[np.ndarray, str]:
    expanded_path = os.path.expanduser(os.path.expandvars(path))

    if not os.path.isfile(expanded_path):
        raise FileNotFoundError(f"Navigation2 parameter file not found: {expanded_path}")

    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to read --nav2-params. Install python3-yaml or "
            "provide --footprint-polygon explicitly."
        ) from exc

    with open(expanded_path, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    matches = _collect_named_values(document, "footprint")
    parsed_matches: List[Tuple[str, np.ndarray]] = []

    for key_path, value in matches:
        try:
            parsed_matches.append((key_path, parse_footprint_value(value)))
        except Exception:
            # Ignore commented/auxiliary structures that are not valid polygons.
            continue

    if not parsed_matches:
        raise ValueError(f"No valid 'footprint' parameter was found in {expanded_path}")

    reference = parsed_matches[0][1]
    inconsistent = [
        key_path
        for key_path, vertices in parsed_matches[1:]
        if vertices.shape != reference.shape or not np.allclose(vertices, reference, atol=1e-9)
    ]

    if inconsistent:
        all_paths = ", ".join(key_path for key_path, _vertices in parsed_matches)
        raise ValueError(
            "Different active footprint polygons were found in the Navigation2 "
            f"parameter file ({all_paths}). Resolve the inconsistency before computing clearance."
        )

    source_paths = ",".join(key_path for key_path, _vertices in parsed_matches)
    return reference, f"nav2_params:{expanded_path}:{source_paths}"


def resolve_footprint(args) -> Tuple[np.ndarray, str]:
    if args.footprint_polygon is not None:
        return parse_footprint_value(args.footprint_polygon), "command_line"

    if args.nav2_params is not None:
        return load_footprint_from_nav2_params(args.nav2_params)

    return DEFAULT_NAV2_FOOTPRINT.copy(), "verified_default:truck_nav2_params.yaml"


def footprint_to_string(vertices: np.ndarray) -> str:
    return ";".join(f"{x:.6f},{y:.6f}" for x, y in vertices)


def read_bag(
    label: str,
    path: str,
    odom_topic: str,
    obstacle_topic: str,
    max_grids: int,
    grid_stride: int,
) -> BagData:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bag path does not exist: {path}")

    print("")
    print(f"Reading bag: {label}")
    print(f"  Path: {path}")

    reader = open_bag_reader(path)
    topic_type_map = build_topic_type_map(reader)

    if odom_topic not in topic_type_map:
        print(f"  WARNING: odom topic not found in bag: {odom_topic}")

    if obstacle_topic not in topic_type_map:
        print(f"  WARNING: obstacle topic not found in bag: {obstacle_topic}")

    message_classes = {}

    for topic_name, type_name in topic_type_map.items():
        try:
            message_classes[topic_name] = get_message(type_name)
        except Exception:
            message_classes[topic_name] = None

    odom_samples: List[OdomSample] = []
    grid_samples: List[GridSample] = []

    grid_counter = 0

    while reader.has_next():
        topic, data, _timestamp = reader.read_next()

        if topic not in {odom_topic, obstacle_topic}:
            continue

        msg_cls = message_classes.get(topic)

        if msg_cls is None:
            continue

        try:
            msg = deserialize_message(data, msg_cls)
        except Exception as exc:
            print(f"  Warning: failed to deserialize topic {topic}: {exc}")
            continue

        if topic == odom_topic:
            try:
                sample = odom_msg_to_sample(msg)
                if sample.stamp_sec <= 0.0:
                    sample.stamp_sec = float(_timestamp) * 1e-9
                odom_samples.append(sample)
            except Exception as exc:
                print(f"  Warning: failed to parse odometry message: {exc}")

        elif topic == obstacle_topic:
            grid_counter += 1

            if grid_counter % max(1, grid_stride) != 0:
                continue

            if len(grid_samples) >= max_grids:
                continue

            try:
                sample = occupancy_grid_msg_to_sample(msg)
                if sample.stamp_sec <= 0.0:
                    sample.stamp_sec = float(_timestamp) * 1e-9
                grid_samples.append(sample)
            except Exception as exc:
                print(f"  Warning: failed to parse occupancy grid message: {exc}")

    odom_samples.sort(key=lambda item: item.stamp_sec)
    grid_samples.sort(key=lambda item: item.stamp_sec)

    print(f"  Odom samples: {len(odom_samples)}")
    print(f"  Obstacle grid samples kept: {len(grid_samples)}")

    if odom_samples:
        print(f"  Odom frame: {odom_samples[0].frame_id} -> {odom_samples[0].child_frame_id}")
        print(f"  First odom: x={odom_samples[0].x:.3f}, y={odom_samples[0].y:.3f}, yaw={odom_samples[0].yaw:.3f}")
        print(f"  Last odom:  x={odom_samples[-1].x:.3f}, y={odom_samples[-1].y:.3f}, yaw={odom_samples[-1].yaw:.3f}")

    if grid_samples:
        print(f"  Obstacle grid frame: {grid_samples[0].frame_id}")
        print(f"  Grid: {grid_samples[0].width} x {grid_samples[0].height}, res={grid_samples[0].resolution:.3f}")
        print(f"  Grid origin: x={grid_samples[0].origin_x:.3f}, y={grid_samples[0].origin_y:.3f}, yaw={grid_samples[0].origin_yaw:.3f}")

    return BagData(
        label=label,
        path=path,
        odom_samples=odom_samples,
        grid_samples=grid_samples,
    )


def odom_to_arrays(odom_samples: List[OdomSample]) -> Dict[str, np.ndarray]:
    stamp = np.array([sample.stamp_sec for sample in odom_samples], dtype=float)
    t = stamp.copy()
    x = np.array([sample.x for sample in odom_samples], dtype=float)
    y = np.array([sample.y for sample in odom_samples], dtype=float)
    yaw = np.array([sample.yaw for sample in odom_samples], dtype=float)
    twist_x = np.array([sample.twist_linear_x for sample in odom_samples], dtype=float)
    twist_y = np.array([sample.twist_linear_y for sample in odom_samples], dtype=float)
    wz = np.array([sample.twist_angular_z for sample in odom_samples], dtype=float)

    # Remove duplicate or non-increasing timestamps.
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 1e-6

    stamp = stamp[keep]
    t = t[keep]
    x = x[keep]
    y = y[keep]
    yaw = yaw[keep]
    twist_x = twist_x[keep]
    twist_y = twist_y[keep]
    wz = wz[keep]

    # Normalize time to start at zero.
    if t.size > 0:
        t = t - t[0]

    return {
        "stamp": stamp,
        "t": t,
        "x": x,
        "y": y,
        "yaw": yaw,
        "twist_x": twist_x,
        "twist_y": twist_y,
        "wz": wz,
    }


def trim_arrays(arrays: Dict[str, np.ndarray], trim_start_s: float, trim_end_s: float) -> Dict[str, np.ndarray]:
    t = arrays["t"]

    if t.size == 0:
        return arrays

    t_min = trim_start_s
    t_max = t[-1] - trim_end_s

    if t_max <= t_min:
        return arrays

    valid = (t >= t_min) & (t <= t_max)

    if np.count_nonzero(valid) < 10:
        return arrays

    return {key: value[valid] for key, value in arrays.items()}


def compute_reference_line(
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
    end_fraction: float,
) -> Tuple[np.ndarray, float]:
    """
    Return a point on the line and the line heading angle.

    Modes:
        first_last:
            line from first to last trajectory point

        endpoints_mean:
            line from mean of initial segment to mean of final segment

        pca:
            principal direction of all trajectory points
    """
    points = np.column_stack([x, y])

    if points.shape[0] < 2:
        return np.array([0.0, 0.0]), 0.0

    if mode == "first_last":
        p0 = points[0]
        p1 = points[-1]
        direction = p1 - p0

        if np.linalg.norm(direction) < 1e-9:
            direction = np.array([1.0, 0.0])

        theta = math.atan2(direction[1], direction[0])
        return p0, theta

    if mode == "endpoints_mean":
        n = points.shape[0]
        k = max(3, int(round(n * end_fraction)))
        k = min(k, n // 2)

        p0 = np.mean(points[:k], axis=0)
        p1 = np.mean(points[-k:], axis=0)
        direction = p1 - p0

        if np.linalg.norm(direction) < 1e-9:
            direction = points[-1] - points[0]

        if np.linalg.norm(direction) < 1e-9:
            direction = np.array([1.0, 0.0])

        theta = math.atan2(direction[1], direction[0])
        return p0, theta

    if mode == "pca":
        center = np.mean(points, axis=0)
        centered = points - center

        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        except Exception:
            direction = points[-1] - points[0]

        if np.linalg.norm(direction) < 1e-9:
            direction = np.array([1.0, 0.0])

        # Orient direction roughly from first to last.
        if np.dot(direction, points[-1] - points[0]) < 0.0:
            direction = -direction

        theta = math.atan2(direction[1], direction[0])
        return center, theta

    raise ValueError(f"Unknown reference line mode: {mode}")


def lateral_error_to_line(
    x: np.ndarray,
    y: np.ndarray,
    line_point: np.ndarray,
    theta: float,
) -> np.ndarray:
    dx = x - line_point[0]
    dy = y - line_point[1]

    # Unit normal to line direction.
    nx = -math.sin(theta)
    ny = math.cos(theta)

    return dx * nx + dy * ny


def path_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")

    dx = np.diff(x)
    dy = np.diff(y)

    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def moving_average_edge(signal: np.ndarray, window: int) -> np.ndarray:
    if signal.size == 0:
        return signal

    if window <= 1:
        return signal.copy()

    if window % 2 == 0:
        window += 1

    if signal.size < window:
        return signal.copy()

    half = window // 2
    padded = np.pad(signal, (half, half), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")

    return smoothed


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(signal)

    if np.count_nonzero(valid) == 0:
        return float("nan")

    return float(np.sqrt(np.mean(signal[valid] ** 2)))


def p95_abs(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(signal)

    if np.count_nonzero(valid) == 0:
        return float("nan")

    return float(np.percentile(np.abs(signal[valid]), 95))


def gradient_safe(signal: np.ndarray, t: np.ndarray) -> np.ndarray:
    if signal.size < 3 or t.size < 3:
        return np.full_like(signal, np.nan, dtype=float)

    return np.gradient(signal, t)


def jerk_from_signal_raw(value: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute acceleration and jerk from a time-dependent signal using np.gradient.

    If value is velocity, acceleration = d(value)/dt and jerk = d(acceleration)/dt.
    """
    if value.size < 5:
        nan = np.full_like(value, np.nan, dtype=float)
        return nan, nan, nan

    a = gradient_safe(value, t)
    j = gradient_safe(a, t)

    return value, a, j


def resample_uniform(t: np.ndarray, signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    if t.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    t0 = float(t[0])
    t1 = float(t[-1])

    if t1 <= t0:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_uniform = np.arange(t0, t1, dt, dtype=float)

    if t_uniform.size < 5:
        return np.array([], dtype=float), np.array([], dtype=float)

    signal_uniform = np.interp(t_uniform, t, signal)

    return t_uniform, signal_uniform


def filtered_jerk_from_position_projection(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    line_point: np.ndarray,
    theta: float,
    resample_dt: float,
    smoothing_window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute filtered longitudinal jerk from position projected onto a reference line.

    Pipeline:
        1. project x,y onto longitudinal axis s(t)
        2. resample s(t) to a uniform grid
        3. smooth s(t)
        4. differentiate s -> v -> a -> j
    """
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    points = np.column_stack([x, y])
    s = (points - line_point) @ direction

    t_uniform, s_uniform = resample_uniform(t, s, resample_dt)

    if t_uniform.size < 5:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    s_smooth = moving_average_edge(s_uniform, smoothing_window)

    v = gradient_safe(s_smooth, t_uniform)
    a = gradient_safe(v, t_uniform)
    j = gradient_safe(a, t_uniform)

    return t_uniform, v, a, j


def raw_jerk_from_position_projection(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    line_point: np.ndarray,
    theta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    points = np.column_stack([x, y])
    s = (points - line_point) @ direction

    v = gradient_safe(s, t)
    a = gradient_safe(v, t)
    j = gradient_safe(a, t)

    return v, a, j


def extract_obstacle_centers_from_grid(
    grid: GridSample,
    threshold: int,
    obstacle_roi: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Return occupied-cell centers expressed in the OccupancyGrid frame."""
    occupied_rows, occupied_cols = np.where(grid.data >= threshold)

    if occupied_rows.size == 0:
        return np.empty((0, 2), dtype=float)

    x_local = (occupied_cols.astype(float) + 0.5) * grid.resolution
    y_local = (occupied_rows.astype(float) + 0.5) * grid.resolution

    c = math.cos(grid.origin_yaw)
    s = math.sin(grid.origin_yaw)

    x_global = grid.origin_x + c * x_local - s * y_local
    y_global = grid.origin_y + s * x_local + c * y_local
    centers = np.column_stack([x_global, y_global])

    if obstacle_roi is not None:
        xmin, xmax, ymin, ymax = obstacle_roi
        half_diagonal = grid.resolution / math.sqrt(2.0)
        valid = (
            (centers[:, 0] >= xmin - half_diagonal)
            & (centers[:, 0] <= xmax + half_diagonal)
            & (centers[:, 1] >= ymin - half_diagonal)
            & (centers[:, 1] <= ymax + half_diagonal)
        )
        centers = centers[valid]

    return centers


def transform_footprint_polygon(
    footprint_local: np.ndarray,
    x: float,
    y: float,
    yaw: float,
) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return footprint_local @ rotation.T + np.array([x, y], dtype=float)


def point_distances_to_convex_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Exact distance from each point to a counter-clockwise convex polygon."""
    points = np.asarray(points, dtype=float)
    polygon = validate_footprint(polygon)

    if points.size == 0:
        return np.empty((0,), dtype=float)

    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    inside = np.all(cross >= -1e-10, axis=1)

    edge_length_sq = np.sum(edges * edges, axis=1)
    projection = np.sum(relative * edges[None, :, :], axis=2) / edge_length_sq[None, :]
    projection = np.clip(projection, 0.0, 1.0)
    closest = polygon[None, :, :] + projection[:, :, None] * edges[None, :, :]
    distances = np.sqrt(np.sum((points[:, None, :] - closest) ** 2, axis=2))
    result = np.min(distances, axis=1)
    result[inside] = 0.0
    return result


def convex_polygons_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons."""
    for polygon in (first, second):
        edges = np.roll(polygon, -1, axis=0) - polygon
        axes = np.column_stack([-edges[:, 1], edges[:, 0]])

        for axis in axes:
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            axis = axis / norm
            first_projection = first @ axis
            second_projection = second @ axis
            if (
                float(np.max(first_projection)) < float(np.min(second_projection)) - 1e-10
                or float(np.max(second_projection)) < float(np.min(first_projection)) - 1e-10
            ):
                return False

    return True


def convex_polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = validate_footprint(first)
    second = validate_footprint(second)

    if convex_polygons_intersect(first, second):
        return 0.0

    first_to_second = point_distances_to_convex_polygon(first, second)
    second_to_first = point_distances_to_convex_polygon(second, first)
    return float(min(np.min(first_to_second), np.min(second_to_first)))


def occupancy_cell_polygon(
    center_x: float,
    center_y: float,
    resolution: float,
    grid_yaw: float,
) -> np.ndarray:
    half = 0.5 * float(resolution)
    local = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=float,
    )
    c = math.cos(grid_yaw)
    s = math.sin(grid_yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return local @ rotation.T + np.array([center_x, center_y], dtype=float)


def minimum_clearance_for_pose(
    footprint_world: np.ndarray,
    robot_center_xy: np.ndarray,
    obstacle_centers: np.ndarray,
    cell_resolution: float,
    grid_yaw: float,
) -> Tuple[float, float, float, int]:
    """
    Return footprint-to-cell-area, footprint-to-center, legacy base-to-center,
    and the index of the occupied cell defining the area clearance.
    """
    if obstacle_centers.size == 0:
        return float("nan"), float("nan"), float("nan"), -1

    footprint_to_centers = point_distances_to_convex_polygon(
        obstacle_centers,
        footprint_world,
    )
    footprint_to_center_min = float(np.min(footprint_to_centers))

    base_distances = np.linalg.norm(obstacle_centers - robot_center_xy[None, :], axis=1)
    base_to_center_min = float(np.min(base_distances))

    # Exact polygon-to-cell calculation with a conservative broad-phase lower
    # bound.  This avoids testing every occupied cell polygon in dense grids.
    order = np.argsort(base_distances)
    footprint_radius = float(np.max(np.linalg.norm(footprint_world - robot_center_xy, axis=1)))
    cell_radius = float(cell_resolution) / math.sqrt(2.0)
    best_distance = float("inf")
    best_index = -1

    for cell_index in order:
        lower_bound = max(
            0.0,
            float(base_distances[cell_index]) - footprint_radius - cell_radius,
        )
        if lower_bound >= best_distance - 1e-12:
            break

        center = obstacle_centers[cell_index]
        cell_polygon = occupancy_cell_polygon(
            center_x=float(center[0]),
            center_y=float(center[1]),
            resolution=cell_resolution,
            grid_yaw=grid_yaw,
        )
        distance = convex_polygon_distance(footprint_world, cell_polygon)

        if distance < best_distance:
            best_distance = distance
            best_index = int(cell_index)

        if best_distance <= 1e-12:
            best_distance = 0.0
            break

    return best_distance, footprint_to_center_min, base_to_center_min, best_index


def _single_common_frame(frame_names: List[str], description: str) -> str:
    normalized = sorted({normalize_frame_name(name) for name in frame_names if normalize_frame_name(name)})
    if len(normalized) != 1:
        raise RuntimeError(
            f"Expected one {description} frame, but found: {normalized or ['<empty>']}"
        )
    return normalized[0]


def compute_synchronized_clearance(
    arrays: Dict[str, np.ndarray],
    bag: BagData,
    footprint_local: np.ndarray,
    args,
    obstacle_roi: Optional[Tuple[float, float, float, float]],
) -> ClearanceResult:
    if not bag.grid_samples:
        return ClearanceResult(
            *([float("nan")] * 12),
            grids_evaluated=0,
            pose_grid_pairs_evaluated=0,
            obstacle_cells_evaluated=0,
            odom_frame="",
            obstacle_frame="",
        )

    odom_frame = _single_common_frame(
        [sample.frame_id for sample in bag.odom_samples],
        "odometry",
    )
    obstacle_frame = _single_common_frame(
        [sample.frame_id for sample in bag.grid_samples],
        "obstacle-grid",
    )

    if odom_frame != obstacle_frame:
        message = (
            "Clearance cannot be computed without a TF transform because the "
            f"odometry frame is '{odom_frame}' and the obstacle frame is '{obstacle_frame}'."
        )
        if args.frame_check == "strict":
            raise RuntimeError(message)
        print(f"  WARNING: {message}")

    stamps = arrays["stamp"]
    x = arrays["x"]
    y = arrays["y"]
    yaw = arrays["yaw"]

    selected_grids = [bag.grid_samples[-1]] if args.latest_grid_only else bag.grid_samples

    best_area = float("inf")
    best_center = float("inf")
    best_legacy = float("inf")
    best_details = None
    grids_evaluated = 0
    pairs_evaluated = 0
    cells_evaluated = 0

    for grid_index, grid in enumerate(selected_grids):
        if grid.stamp_sec < float(stamps[0]) or grid.stamp_sec > float(stamps[-1]):
            continue

        insertion = int(np.searchsorted(stamps, grid.stamp_sec))
        candidates = []
        if insertion < stamps.size:
            candidates.append(insertion)
        if insertion > 0:
            candidates.append(insertion - 1)
        if not candidates:
            continue

        pose_index = min(candidates, key=lambda idx: abs(float(stamps[idx]) - grid.stamp_sec))
        time_delta = abs(float(stamps[pose_index]) - grid.stamp_sec)
        if time_delta > args.clearance_max_time_delta_s:
            continue

        centers = extract_obstacle_centers_from_grid(
            grid=grid,
            threshold=args.obstacle_threshold,
            obstacle_roi=obstacle_roi,
        )
        grids_evaluated += 1

        if centers.shape[0] == 0:
            continue

        if centers.shape[0] > args.max_obstacle_cells_per_grid:
            raise RuntimeError(
                f"Grid {grid_index} contains {centers.shape[0]} occupied cells after ROI, "
                f"exceeding --max-obstacle-cells-per-grid={args.max_obstacle_cells_per_grid}. "
                "Use a physically justified --obstacle-roi; cells are never subsampled for clearance."
            )

        robot_center = np.array([x[pose_index], y[pose_index]], dtype=float)
        footprint_world = transform_footprint_polygon(
            footprint_local=footprint_local,
            x=float(x[pose_index]),
            y=float(y[pose_index]),
            yaw=float(yaw[pose_index]),
        )

        area_distance, center_distance, legacy_distance, closest_cell_index = minimum_clearance_for_pose(
            footprint_world=footprint_world,
            robot_center_xy=robot_center,
            obstacle_centers=centers,
            cell_resolution=grid.resolution,
            grid_yaw=grid.origin_yaw,
        )

        pairs_evaluated += 1
        cells_evaluated += int(centers.shape[0])
        best_center = min(best_center, center_distance)
        best_legacy = min(best_legacy, legacy_distance)

        if area_distance < best_area:
            best_area = area_distance
            closest_center = centers[closest_cell_index] if closest_cell_index >= 0 else np.array([np.nan, np.nan])
            best_details = (
                float(stamps[pose_index]),
                float(grid.stamp_sec),
                float(time_delta),
                float(x[pose_index]),
                float(y[pose_index]),
                float(yaw[pose_index]),
                float(closest_center[0]),
                float(closest_center[1]),
                float(grid.resolution),
            )

    if pairs_evaluated == 0 or best_details is None:
        print(
            "  WARNING: no occupied grid could be synchronized with trimmed odometry "
            f"within {args.clearance_max_time_delta_s:.3f} s."
        )
        nan_details = (float("nan"),) * 9
        best_area = float("nan")
        best_center = float("nan")
        best_legacy = float("nan")
        best_details = nan_details

    return ClearanceResult(
        footprint_to_cell_area_m=best_area,
        footprint_to_cell_center_m=best_center,
        base_to_cell_center_legacy_m=best_legacy,
        closest_pose_stamp_sec=best_details[0],
        closest_grid_stamp_sec=best_details[1],
        closest_time_delta_s=best_details[2],
        closest_pose_x=best_details[3],
        closest_pose_y=best_details[4],
        closest_pose_yaw=best_details[5],
        closest_obstacle_x=best_details[6],
        closest_obstacle_y=best_details[7],
        closest_grid_resolution_m=best_details[8],
        grids_evaluated=grids_evaluated,
        pose_grid_pairs_evaluated=pairs_evaluated,
        obstacle_cells_evaluated=cells_evaluated,
        odom_frame=odom_frame,
        obstacle_frame=obstacle_frame,
    )


def compute_metrics_for_bag(
    bag: BagData,
    args,
    obstacle_roi: Optional[Tuple[float, float, float, float]],
    footprint_local: np.ndarray,
    footprint_source: str,
) -> Dict[str, object]:
    arrays = odom_to_arrays(bag.odom_samples)
    arrays = trim_arrays(arrays, args.trim_start_s, args.trim_end_s)

    t = arrays["t"]
    x = arrays["x"]
    y = arrays["y"]
    yaw = arrays["yaw"]
    twist_x = arrays["twist_x"]
    twist_y = arrays["twist_y"]

    if t.size < 10:
        raise RuntimeError(f"Not enough odometry samples for {bag.label}")

    line_point, ref_theta = compute_reference_line(
        x=x,
        y=y,
        mode=args.reference_line_mode,
        end_fraction=args.reference_end_fraction,
    )

    lateral_error = lateral_error_to_line(
        x=x,
        y=y,
        line_point=line_point,
        theta=ref_theta,
    )

    yaw_unwrapped = np.unwrap(yaw)
    heading_error = wrap_to_pi(yaw_unwrapped - ref_theta)

    rmse_y = rms(lateral_error)
    rmse_psi = rms(heading_error)

    distance_m = path_distance(x, y)
    duration_s = float(t[-1] - t[0])

    speed_norm_twist = np.sqrt(twist_x * twist_x + twist_y * twist_y)

    # Jerk from odometry twist.linear.x
    _v_twist_x, a_twist_x, j_twist_x = jerk_from_signal_raw(twist_x, t)

    # Jerk from odometry twist speed norm
    _v_twist_norm, a_twist_norm, j_twist_norm = jerk_from_signal_raw(speed_norm_twist, t)

    # Raw jerk from position projection, no resampling/smoothing
    v_posproj_raw, a_posproj_raw, j_posproj_raw = raw_jerk_from_position_projection(
        t=t,
        x=x,
        y=y,
        line_point=line_point,
        theta=ref_theta,
    )

    # Filtered jerk from position projection
    t_uniform, v_filtered, a_filtered, j_filtered = filtered_jerk_from_position_projection(
        t=t,
        x=x,
        y=y,
        line_point=line_point,
        theta=ref_theta,
        resample_dt=args.resample_dt,
        smoothing_window=args.smoothing_window,
    )

    clearance = compute_synchronized_clearance(
        arrays=arrays,
        bag=bag,
        footprint_local=footprint_local,
        args=args,
        obstacle_roi=obstacle_roi,
    )

    mean_speed_twist = float(np.nanmean(speed_norm_twist))
    median_speed_twist = float(np.nanmedian(speed_norm_twist))
    mean_speed_posproj = float(np.nanmean(np.abs(v_posproj_raw)))
    median_speed_posproj = float(np.nanmedian(np.abs(v_posproj_raw)))

    result = {
        "run": bag.label,
        "bag_path": bag.path,
        "n_odom": int(t.size),
        "n_obstacle_points": int(clearance.obstacle_cells_evaluated),
        "n_obstacle_cells_evaluated": int(clearance.obstacle_cells_evaluated),
        "n_obstacle_grids_evaluated": int(clearance.grids_evaluated),
        "n_synchronized_pose_grid_pairs": int(clearance.pose_grid_pairs_evaluated),
        "duration_s": duration_s,
        "distance_m": distance_m,
        "mean_speed_twist_mps": mean_speed_twist,
        "median_speed_twist_mps": median_speed_twist,
        "mean_speed_posproj_mps": mean_speed_posproj,
        "median_speed_posproj_mps": median_speed_posproj,
        "RMSE_y_m": rmse_y,
        "RMSE_psi_rad": rmse_psi,
        "J_rms_twist_x_raw_mps3": rms(j_twist_x),
        "J_p95_twist_x_raw_mps3": p95_abs(j_twist_x),
        "J_rms_twist_norm_raw_mps3": rms(j_twist_norm),
        "J_p95_twist_norm_raw_mps3": p95_abs(j_twist_norm),
        "J_rms_posproj_raw_mps3": rms(j_posproj_raw),
        "J_p95_posproj_raw_mps3": p95_abs(j_posproj_raw),
        "J_rms_posproj_filtered_mps3": rms(j_filtered),
        "J_p95_posproj_filtered_mps3": p95_abs(j_filtered),
        # Primary clearance used in the paper after the footprint correction.
        "d_min_m": clearance.footprint_to_cell_area_m,
        "d_min_footprint_to_cell_area_m": clearance.footprint_to_cell_area_m,
        "d_min_footprint_to_cell_center_m": clearance.footprint_to_cell_center_m,
        "d_min_base_to_cell_center_legacy_m": clearance.base_to_cell_center_legacy_m,
        "clearance_definition": CLEARANCE_DEFINITION,
        "clearance_pose_stamp_sec": clearance.closest_pose_stamp_sec,
        "clearance_grid_stamp_sec": clearance.closest_grid_stamp_sec,
        "clearance_time_delta_s": clearance.closest_time_delta_s,
        "clearance_pose_x": clearance.closest_pose_x,
        "clearance_pose_y": clearance.closest_pose_y,
        "clearance_pose_yaw": clearance.closest_pose_yaw,
        "clearance_obstacle_cell_center_x": clearance.closest_obstacle_x,
        "clearance_obstacle_cell_center_y": clearance.closest_obstacle_y,
        "clearance_grid_resolution_m": clearance.closest_grid_resolution_m,
        "clearance_odom_frame": clearance.odom_frame,
        "clearance_obstacle_frame": clearance.obstacle_frame,
        "clearance_max_time_delta_s": float(args.clearance_max_time_delta_s),
        "footprint_source": footprint_source,
        "footprint_vertices_base_frame": footprint_to_string(footprint_local),
        "footprint_length_m": float(np.ptp(footprint_local[:, 0])),
        "footprint_width_m": float(np.ptp(footprint_local[:, 1])),
        "reference_line_mode": args.reference_line_mode,
        "reference_line_x0": float(line_point[0]),
        "reference_line_y0": float(line_point[1]),
        "reference_line_theta_rad": float(ref_theta),
        "resample_dt_s": float(args.resample_dt),
        "smoothing_window": int(args.smoothing_window),
        "trim_start_s": float(args.trim_start_s),
        "trim_end_s": float(args.trim_end_s),
    }

    return result


def format_float(value: object, precision: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{precision}f}"
    return str(value)


def print_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No rows to print.")
        return

    columns = [
        "run",
        "RMSE_y_m",
        "RMSE_psi_rad",
        "J_rms_twist_x_raw_mps3",
        "J_rms_posproj_raw_mps3",
        "J_rms_posproj_filtered_mps3",
        "d_min_m",
        "d_min_footprint_to_cell_center_m",
        "d_min_base_to_cell_center_legacy_m",
        "duration_s",
        "distance_m",
        "mean_speed_posproj_mps",
        "n_synchronized_pose_grid_pairs",
    ]

    table = []
    for row in rows:
        table.append([format_float(row.get(col, ""), precision=4) for col in columns])

    widths = []
    for col_idx, col in enumerate(columns):
        max_width = len(col)
        for row in table:
            max_width = max(max_width, len(row[col_idx]))
        widths.append(max_width)

    header = " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns))
    sep = "-+-".join("-" * widths[idx] for idx in range(len(columns)))

    print("")
    print(header)
    print(sep)

    for row in table:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))))

    print("")


def write_csv(rows: List[Dict[str, object]], output_csv: str) -> None:
    if not rows:
        return

    output_csv = os.path.expanduser(os.path.expandvars(output_csv))
    output_dir = os.path.dirname(output_csv)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_csv}")


def write_xlsx_if_possible(rows: List[Dict[str, object]], output_xlsx: Optional[str]) -> None:
    if not output_xlsx or not rows:
        return

    output_xlsx = os.path.expanduser(os.path.expandvars(output_xlsx))
    output_dir = os.path.dirname(output_xlsx)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        import pandas as pd
    except Exception:
        print("")
        print("WARNING: pandas is not installed. XLSX was not written.")
        print("Install if needed:")
        print("  /usr/bin/python3 -m pip install pandas openpyxl")
        return

    df = pd.DataFrame(rows)
    df.to_excel(output_xlsx, index=False)

    print(f"Saved XLSX: {output_xlsx}")


def run_geometry_self_tests() -> None:
    footprint = validate_footprint(
        np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    )
    robot_center = np.array([0.0, 0.0])

    area, center, legacy, _index = minimum_clearance_for_pose(
        footprint_world=footprint,
        robot_center_xy=robot_center,
        obstacle_centers=np.array([[4.0, 0.0]]),
        cell_resolution=2.0,
        grid_yaw=0.0,
    )
    assert math.isclose(area, 2.0, abs_tol=1e-9), area
    assert math.isclose(center, 3.0, abs_tol=1e-9), center
    assert math.isclose(legacy, 4.0, abs_tol=1e-9), legacy

    area_overlap, _center, _legacy, _index = minimum_clearance_for_pose(
        footprint_world=footprint,
        robot_center_xy=robot_center,
        obstacle_centers=np.array([[1.5, 0.0]]),
        cell_resolution=1.0,
        grid_yaw=0.0,
    )
    assert math.isclose(area_overlap, 0.0, abs_tol=1e-9), area_overlap

    rotated = transform_footprint_polygon(footprint, x=2.0, y=3.0, yaw=math.pi / 2.0)
    assert np.allclose(np.mean(rotated, axis=0), [2.0, 3.0], atol=1e-9)

    print("Footprint/cell geometry self-tests: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute field metrics from ROS 2 MCAP rosbags."
    )

    parser.add_argument(
        "--bag",
        action="append",
        required=False,
        help='Bag specification as "Label:/path/to/bag.mcap". Can be repeated.',
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run pure geometry tests without opening a ROS bag, then exit.",
    )

    parser.add_argument(
        "--nav2-params",
        default=None,
        help=(
            "Navigation2 YAML used to load and audit the footprint. If omitted, "
            "the verified 9.605 x 3.478 m truck footprint is used."
        ),
    )

    parser.add_argument(
        "--footprint-polygon",
        default=None,
        help=(
            "Explicit footprint polygon, for example "
            "'[[−4.8025,−1.739],[4.8025,−1.739],[4.8025,1.739],[−4.8025,1.739]]'. "
            "This takes precedence over --nav2-params. Use ASCII minus signs in the command."
        ),
    )

    parser.add_argument(
        "--odom-topic",
        default="/odometry/global",
        help="Odometry topic. Default: /odometry/global",
    )

    parser.add_argument(
        "--obstacle-topic",
        default="/global_costmap/obstacle_layer",
        help="OccupancyGrid obstacle topic. Default: /global_costmap/obstacle_layer",
    )

    parser.add_argument(
        "--reference-line-mode",
        choices=["first_last", "endpoints_mean", "pca"],
        default="endpoints_mean",
        help="How to define local straight reference line. Default: endpoints_mean",
    )

    parser.add_argument(
        "--reference-end-fraction",
        type=float,
        default=0.10,
        help="Fraction of trajectory used at start/end for endpoints_mean. Default: 0.10",
    )

    parser.add_argument(
        "--trim-start-s",
        type=float,
        default=0.5,
        help="Seconds removed at beginning before metrics. Default: 0.5",
    )

    parser.add_argument(
        "--trim-end-s",
        type=float,
        default=0.5,
        help="Seconds removed at end before metrics. Default: 0.5",
    )

    parser.add_argument(
        "--resample-dt",
        type=float,
        default=0.10,
        help="Uniform resampling step for filtered jerk, in seconds. Default: 0.10",
    )

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=11,
        help="Moving-average window for filtered jerk. Default: 11",
    )

    parser.add_argument(
        "--max-grids",
        type=int,
        default=200,
        help="Maximum number of OccupancyGrid messages to process per bag. Default: 200",
    )

    parser.add_argument(
        "--grid-stride",
        type=int,
        default=1,
        help="Keep one OccupancyGrid every N messages. Default: 1",
    )

    parser.add_argument(
        "--latest-grid-only",
        action="store_true",
        help=(
            "Use only the latest obstacle grid. Normally leave this disabled so each "
            "grid is paired with its nearest odometry sample."
        ),
    )

    parser.add_argument(
        "--obstacle-threshold",
        type=int,
        default=50,
        help="Occupancy threshold for obstacle cells. Default: 50",
    )

    parser.add_argument(
        "--max-obstacle-cells-per-grid",
        type=int,
        default=50000,
        help=(
            "Safety limit for occupied cells per grid after ROI. The metric never "
            "subsamples cells. Default: 50000."
        ),
    )

    parser.add_argument(
        "--max-obstacle-points",
        dest="max_obstacle_cells_per_grid",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--clearance-max-time-delta-s",
        type=float,
        default=1.0,
        help=(
            "Maximum timestamp difference between an obstacle grid and its nearest "
            "odometry pose. Default: 1.0 s."
        ),
    )

    parser.add_argument(
        "--frame-check",
        choices=["strict", "warn"],
        default="strict",
        help="Reject odometry/obstacle frame mismatch by default. Use 'warn' only for diagnostics.",
    )

    parser.add_argument(
        "--obstacle-roi",
        nargs=4,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Optional obstacle extraction ROI: xmin xmax ymin ymax.",
    )

    parser.add_argument(
        "--distance-chunk-size",
        type=int,
        default=1000,
        help="Deprecated compatibility option; no longer used by footprint clearance.",
    )

    parser.add_argument(
        "--output-csv",
        default="field_metrics_recomputed.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--output-xlsx",
        default=None,
        help="Optional output XLSX path.",
    )

    args = parser.parse_args()

    if args.self_test:
        run_geometry_self_tests()
        return

    if not args.bag:
        parser.error("at least one --bag is required unless --self-test is used")

    if args.clearance_max_time_delta_s < 0.0:
        parser.error("--clearance-max-time-delta-s must be non-negative")

    if args.max_obstacle_cells_per_grid <= 0:
        parser.error("--max-obstacle-cells-per-grid must be positive")

    obstacle_roi = parse_roi(args.obstacle_roi)
    footprint_local, footprint_source = resolve_footprint(args)

    print("")
    print("Clearance configuration:")
    print(f"  Footprint source: {footprint_source}")
    print(f"  Footprint vertices [base_footprint]: {footprint_to_string(footprint_local)}")
    print(f"  Footprint length: {np.ptp(footprint_local[:, 0]):.3f} m")
    print(f"  Footprint width:  {np.ptp(footprint_local[:, 1]):.3f} m")
    print(f"  Frame check: {args.frame_check}")
    print(f"  Maximum synchronization delta: {args.clearance_max_time_delta_s:.3f} s")

    bag_specs = [parse_bag_arg(value) for value in args.bag]

    rows: List[Dict[str, object]] = []

    for label, path in bag_specs:
        bag = read_bag(
            label=label,
            path=path,
            odom_topic=args.odom_topic,
            obstacle_topic=args.obstacle_topic,
            max_grids=args.max_grids,
            grid_stride=args.grid_stride,
        )

        row = compute_metrics_for_bag(
            bag=bag,
            args=args,
            obstacle_roi=obstacle_roi,
            footprint_local=footprint_local,
            footprint_source=footprint_source,
        )

        rows.append(row)

    print_table(rows)
    write_csv(rows, args.output_csv)
    write_xlsx_if_possible(rows, args.output_xlsx)

    print("")
    print("Recommended interpretation:")
    print("  - Raw jerk values are diagnostic only and may be inflated by timestamp jitter/noise.")
    print("  - For the paper, prefer J_rms_posproj_filtered_mps3 if the method is described.")
    print("  - d_min_m is footprint-to-occupied-cell-area clearance (the corrected primary value).")
    print("  - d_min_footprint_to_cell_center_m is a point-sampled diagnostic value.")
    print("  - d_min_base_to_cell_center_legacy_m reproduces the old, non-physical definition only for audit.")
    print("  - A zero d_min_m means footprint/cell overlap in the discretized costmap; it does not prove physical contact.")
    print("")


if __name__ == "__main__":
    main()
