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
    - minimum obstacle clearance using the global obstacle layer

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
  --output-csv ~/NAV2_Paper_Scripts/tarde/field_metrics_recomputed.csv \
  --output-xlsx ~/NAV2_Paper_Scripts/tarde/field_metrics_recomputed.xlsx
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception as exc:
    print("\nERROR: ROS 2 Python libraries were not found.")
    print("Use the system Python, not Conda, for ROS 2 bags.")
    print("")
    print("Run:")
    print("  conda deactivate")
    print("  source /opt/ros/jazzy/setup.bash")
    print("  /usr/bin/python3 ~/NAV2_Paper_Scripts/recompute_field_metrics_from_bags.py ...")
    print("")
    print(f"Original error: {exc}\n")
    sys.exit(1)


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
                odom_samples.append(odom_msg_to_sample(msg))
            except Exception as exc:
                print(f"  Warning: failed to parse odometry message: {exc}")

        elif topic == obstacle_topic:
            grid_counter += 1

            if grid_counter % max(1, grid_stride) != 0:
                continue

            if len(grid_samples) >= max_grids:
                continue

            try:
                grid_samples.append(occupancy_grid_msg_to_sample(msg))
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
    t = np.array([sample.stamp_sec for sample in odom_samples], dtype=float)
    x = np.array([sample.x for sample in odom_samples], dtype=float)
    y = np.array([sample.y for sample in odom_samples], dtype=float)
    yaw = np.array([sample.yaw for sample in odom_samples], dtype=float)
    twist_x = np.array([sample.twist_linear_x for sample in odom_samples], dtype=float)
    twist_y = np.array([sample.twist_linear_y for sample in odom_samples], dtype=float)
    wz = np.array([sample.twist_angular_z for sample in odom_samples], dtype=float)

    # Remove duplicate or non-increasing timestamps.
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 1e-6

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


def extract_obstacle_points_from_grids(
    grids: List[GridSample],
    threshold: int,
    max_points: int,
    latest_grid_only: bool,
    obstacle_roi: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    if not grids:
        return np.empty((0, 2), dtype=float)

    selected_grids = [grids[-1]] if latest_grid_only else grids

    points_list = []

    for grid in selected_grids:
        occupied_rows, occupied_cols = np.where(grid.data >= threshold)

        if occupied_rows.size == 0:
            continue

        x_local = (occupied_cols.astype(float) + 0.5) * grid.resolution
        y_local = (occupied_rows.astype(float) + 0.5) * grid.resolution

        c = math.cos(grid.origin_yaw)
        s = math.sin(grid.origin_yaw)

        x_global = grid.origin_x + c * x_local - s * y_local
        y_global = grid.origin_y + s * x_local + c * y_local

        points = np.column_stack([x_global, y_global])
        points_list.append(points)

    if not points_list:
        return np.empty((0, 2), dtype=float)

    all_points = np.vstack(points_list)

    if obstacle_roi is not None:
        xmin, xmax, ymin, ymax = obstacle_roi
        valid = (
            (all_points[:, 0] >= xmin)
            & (all_points[:, 0] <= xmax)
            & (all_points[:, 1] >= ymin)
            & (all_points[:, 1] <= ymax)
        )
        all_points = all_points[valid]

    if all_points.size == 0:
        return np.empty((0, 2), dtype=float)

    rounded = np.round(all_points, decimals=3)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    all_points = all_points[np.sort(unique_idx)]

    if all_points.shape[0] > max_points:
        idx = np.linspace(0, all_points.shape[0] - 1, max_points).astype(int)
        all_points = all_points[idx]

    return all_points


def min_distance_trajectory_to_obstacle(
    x: np.ndarray,
    y: np.ndarray,
    obstacle_xy: np.ndarray,
    chunk_size: int = 1000,
) -> float:
    if x.size == 0 or obstacle_xy.size == 0:
        return float("nan")

    traj = np.column_stack([x, y])
    min_d2 = float("inf")

    for start in range(0, traj.shape[0], chunk_size):
        end = min(start + chunk_size, traj.shape[0])
        chunk = traj[start:end]

        diff = chunk[:, None, :] - obstacle_xy[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        local_min = float(np.min(d2))

        if local_min < min_d2:
            min_d2 = local_min

    return float(math.sqrt(min_d2))


def compute_metrics_for_bag(
    bag: BagData,
    args,
    obstacle_roi: Optional[Tuple[float, float, float, float]],
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

    obstacle_xy = extract_obstacle_points_from_grids(
        grids=bag.grid_samples,
        threshold=args.obstacle_threshold,
        max_points=args.max_obstacle_points,
        latest_grid_only=args.latest_grid_only,
        obstacle_roi=obstacle_roi,
    )

    d_min = min_distance_trajectory_to_obstacle(
        x=x,
        y=y,
        obstacle_xy=obstacle_xy,
        chunk_size=args.distance_chunk_size,
    )

    mean_speed_twist = float(np.nanmean(speed_norm_twist))
    median_speed_twist = float(np.nanmedian(speed_norm_twist))
    mean_speed_posproj = float(np.nanmean(np.abs(v_posproj_raw)))
    median_speed_posproj = float(np.nanmedian(np.abs(v_posproj_raw)))

    result = {
        "run": bag.label,
        "bag_path": bag.path,
        "n_odom": int(t.size),
        "n_obstacle_points": int(obstacle_xy.shape[0]),
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
        "d_min_m": d_min,
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
        "duration_s",
        "distance_m",
        "mean_speed_posproj_mps",
        "n_obstacle_points",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute field metrics from ROS 2 MCAP rosbags."
    )

    parser.add_argument(
        "--bag",
        action="append",
        required=True,
        help='Bag specification as "Label:/path/to/bag.mcap". Can be repeated.',
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
        help="Use only the latest obstacle grid instead of accumulating selected grids.",
    )

    parser.add_argument(
        "--obstacle-threshold",
        type=int,
        default=50,
        help="Occupancy threshold for obstacle cells. Default: 50",
    )

    parser.add_argument(
        "--max-obstacle-points",
        type=int,
        default=5000,
        help="Maximum number of obstacle cells used for d_min. Default: 5000",
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
        help="Chunk size for trajectory-obstacle distance computation. Default: 1000",
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

    obstacle_roi = parse_roi(args.obstacle_roi)

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
        )

        rows.append(row)

    print_table(rows)
    write_csv(rows, args.output_csv)
    write_xlsx_if_possible(rows, args.output_xlsx)

    print("")
    print("Recommended interpretation:")
    print("  - Raw jerk values are diagnostic only and may be inflated by timestamp jitter/noise.")
    print("  - For the paper, prefer J_rms_posproj_filtered_mps3 if the method is described.")
    print("  - d_min_m is computed within each run using its own obstacle-layer projection.")
    print("")


if __name__ == "__main__":
    main()