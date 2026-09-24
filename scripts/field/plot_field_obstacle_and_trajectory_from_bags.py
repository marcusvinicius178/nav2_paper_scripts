#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot field obstacle-avoidance trajectories from ROS 2 MCAP rosbags.

Prepared for the Navigation2 agricultural truck paper.

Recommended topics:
    Trajectory:
        /odometry/global

    Obstacle source:
        /global_costmap/obstacle_layer

Why use /global_costmap/obstacle_layer?
    In the current rosbags, /scan and /cloud use frame_id='cloud'.
    The global obstacle layer uses frame_id='map', the same global frame used by
    /odometry/global. This avoids errors caused by reconstructing the obstacle
    from raw scan data and TF.

Main output:
    Blue line: executed global trajectory.
    Red squares: occupied cell areas from the global obstacle layer.
    Orange polygon (optional): the configured truck footprint at the
    time-synchronized minimum-clearance pose.

Important:
    The plot uses equal aspect ratio:
        1 m in X = 1 m in Y

New feature:
    Use --fixed-obstacle-from "LABEL" to extract the obstacle from one selected
    bag and plot the same obstacle reference in all panels. This is useful for
    paper figures comparing NavFn, Human driver, and SMAC with a common
    obstacle reference.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MatplotlibPolygon
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

try:
    from recompute_field_metrics_from_bags import (
        BagData as MetricsBagData,
        DEFAULT_NAV2_FOOTPRINT,
        compute_synchronized_clearance,
        footprint_to_string,
        load_footprint_from_nav2_params,
        occupancy_cell_polygon,
        parse_footprint_value,
        transform_footprint_polygon,
    )
except Exception as exc:
    raise RuntimeError(
        "The corrected plotting script requires recompute_field_metrics_from_bags.py "
        "in the same directory. Install both corrected files together."
    ) from exc


@dataclass
class OdomSample:
    stamp_sec: float
    frame_id: str
    child_frame_id: str
    x: float
    y: float
    yaw: float


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
class BagResult:
    label: str
    path: str
    trajectory_xy: np.ndarray
    obstacle_cells: np.ndarray
    clearance_footprint_xy: np.ndarray
    d_min_m: float
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
            "after sourcing /opt/ros/iron/setup.bash (or the installed ROS distribution). "
            f"Original error: {ROS_IMPORT_ERROR}"
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

    return OdomSample(
        stamp_sec=stamp_to_sec(msg.header.stamp),
        frame_id=normalize_frame_name(msg.header.frame_id),
        child_frame_id=normalize_frame_name(msg.child_frame_id),
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=quaternion_to_yaw(pose.orientation),
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


def apply_roi(points_xy: np.ndarray, roi: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    if roi is None or points_xy.size == 0:
        return points_xy

    xmin, xmax, ymin, ymax = roi

    valid = (
        (points_xy[:, 0] >= xmin)
        & (points_xy[:, 0] <= xmax)
        & (points_xy[:, 1] >= ymin)
        & (points_xy[:, 1] <= ymax)
    )

    return points_xy[valid]


def normalize_xy(
    points_xy: np.ndarray,
    start_x: float,
    start_y: float,
    start_yaw: float,
    rotate_to_start_heading: bool,
) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.copy()

    normalized = points_xy.copy()
    normalized[:, 0] -= start_x
    normalized[:, 1] -= start_y

    if rotate_to_start_heading:
        rotation_angle = math.pi / 2.0 - start_yaw
        c = math.cos(rotation_angle)
        s = math.sin(rotation_angle)
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        normalized = normalized @ rotation.T

    return normalized


def crop_trajectory_by_roi(
    trajectory_xy: np.ndarray,
    roi: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    if roi is None or trajectory_xy.size == 0:
        return trajectory_xy

    xmin, xmax, ymin, ymax = roi

    valid = (
        (trajectory_xy[:, 0] >= xmin)
        & (trajectory_xy[:, 0] <= xmax)
        & (trajectory_xy[:, 1] >= ymin)
        & (trajectory_xy[:, 1] <= ymax)
    )

    return trajectory_xy[valid]


def smooth_trajectory_xy(trajectory_xy: np.ndarray, window_size: int) -> np.ndarray:
    """
    Smooth trajectory only for visualization.

    This does not change metric computation.
    It only makes the plotted path easier to interpret in the paper figure.
    """
    if trajectory_xy.size == 0:
        return trajectory_xy

    if window_size <= 1:
        return trajectory_xy

    if window_size % 2 == 0:
        window_size += 1

    if trajectory_xy.shape[0] < window_size:
        return trajectory_xy

    kernel = np.ones(window_size, dtype=float) / float(window_size)

    x_smooth = np.convolve(trajectory_xy[:, 0], kernel, mode="same")
    y_smooth = np.convolve(trajectory_xy[:, 1], kernel, mode="same")

    half = window_size // 2

    x_smooth[:half] = trajectory_xy[:half, 0]
    y_smooth[:half] = trajectory_xy[:half, 1]

    x_smooth[-half:] = trajectory_xy[-half:, 0]
    y_smooth[-half:] = trajectory_xy[-half:, 1]

    return np.column_stack([x_smooth, y_smooth])


def read_bag(
    path: str,
    label: str,
    odom_topic: str,
    obstacle_topic: str,
    max_grids: int,
    grid_stride: int,
) -> Tuple[List[OdomSample], List[GridSample]]:
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

    return odom_samples, grid_samples


def extract_obstacle_points_from_grids(
    grids: List[GridSample],
    threshold: int,
    max_points: int,
    latest_grid_only: bool,
) -> np.ndarray:
    """Return rows [center_x, center_y, resolution, grid_yaw]."""
    if not grids:
        return np.empty((0, 4), dtype=float)

    selected_grids = [grids[-1]] if latest_grid_only else grids

    cells_list = []

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

        cells = np.column_stack(
            [
                x_global,
                y_global,
                np.full_like(x_global, grid.resolution, dtype=float),
                np.full_like(x_global, grid.origin_yaw, dtype=float),
            ]
        )
        cells_list.append(cells)

    if not cells_list:
        return np.empty((0, 4), dtype=float)

    all_cells = np.vstack(cells_list)

    # Remove duplicate occupied cells.
    rounded = np.round(all_cells, decimals=6)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    all_cells = all_cells[np.sort(unique_idx)]

    if all_cells.shape[0] > max_points:
        idx = np.linspace(0, all_cells.shape[0] - 1, max_points).astype(int)
        all_cells = all_cells[idx]

    return all_cells


def resolve_footprint(args) -> Tuple[np.ndarray, str]:
    if args.footprint_polygon is not None:
        return parse_footprint_value(args.footprint_polygon), "command_line"
    if args.nav2_params is not None:
        return load_footprint_from_nav2_params(args.nav2_params)
    return DEFAULT_NAV2_FOOTPRINT.copy(), "verified_default:truck_nav2_params.yaml"


def compute_bag_result(
    label: str,
    path: str,
    args,
    plot_roi: Optional[Tuple[float, float, float, float]],
    obstacle_roi: Optional[Tuple[float, float, float, float]],
    footprint_local: np.ndarray,
) -> BagResult:
    odom_samples, grid_samples = read_bag(
        path=path,
        label=label,
        odom_topic=args.odom_topic,
        obstacle_topic=args.obstacle_topic,
        max_grids=args.max_grids,
        grid_stride=args.grid_stride,
    )

    if not odom_samples:
        raise RuntimeError(f"No odometry samples were found in {label} on topic {args.odom_topic}")

    trajectory_xy = np.array([[sample.x, sample.y] for sample in odom_samples], dtype=float)

    obstacle_cells = extract_obstacle_points_from_grids(
        grids=grid_samples,
        threshold=args.obstacle_threshold,
        max_points=args.max_obstacle_points,
        latest_grid_only=args.latest_grid_only,
    )

    metric_arrays = {
        "stamp": np.array([sample.stamp_sec for sample in odom_samples], dtype=float),
        "x": np.array([sample.x for sample in odom_samples], dtype=float),
        "y": np.array([sample.y for sample in odom_samples], dtype=float),
        "yaw": np.array([sample.yaw for sample in odom_samples], dtype=float),
    }
    metric_bag = MetricsBagData(
        label=label,
        path=path,
        odom_samples=odom_samples,
        grid_samples=grid_samples,
    )
    clearance = compute_synchronized_clearance(
        arrays=metric_arrays,
        bag=metric_bag,
        footprint_local=footprint_local,
        args=args,
        obstacle_roi=obstacle_roi,
    )

    # For footprint-clearance figures, show occupied cells from the exact
    # obstacle grid synchronized with the minimum-clearance pose. Plotting
    # cells accumulated from different timestamps could create an apparent
    # footprint/obstacle intersection that was not present in the synchronized
    # clearance calculation.
    if (
        args.plot_clearance_footprint
        and math.isfinite(clearance.closest_grid_stamp_sec)
        and grid_samples
    ):
        closest_grid_for_plot = min(
            grid_samples,
            key=lambda grid: abs(
                float(grid.stamp_sec)
                - float(clearance.closest_grid_stamp_sec)
            ),
        )

        obstacle_cells = extract_obstacle_points_from_grids(
            grids=[closest_grid_for_plot],
            threshold=args.obstacle_threshold,
            max_points=args.max_obstacle_points,
            latest_grid_only=True,
        )

        print(
            "  Obstacle visualization: synchronized minimum-clearance grid "
            f"(stamp={closest_grid_for_plot.stamp_sec:.9f})"
        )

    if math.isfinite(clearance.closest_pose_x):
        clearance_footprint_xy = transform_footprint_polygon(
            footprint_local=footprint_local,
            x=clearance.closest_pose_x,
            y=clearance.closest_pose_y,
            yaw=clearance.closest_pose_yaw,
        )
    else:
        clearance_footprint_xy = np.empty((0, 2), dtype=float)

    # The obstacle ROI is always interpreted in the original common global
    # frame, matching the metric script.  Start normalization is visualization-only.
    obstacle_cells = apply_roi(obstacle_cells, obstacle_roi)

    if args.normalize_start:
        start_x = odom_samples[0].x
        start_y = odom_samples[0].y
        start_yaw = odom_samples[0].yaw

        trajectory_xy = normalize_xy(
            points_xy=trajectory_xy,
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            rotate_to_start_heading=args.rotate_to_start_heading,
        )

        obstacle_cells[:, :2] = normalize_xy(
            points_xy=obstacle_cells[:, :2],
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            rotate_to_start_heading=args.rotate_to_start_heading,
        )

        if args.rotate_to_start_heading and obstacle_cells.size > 0:
            obstacle_cells[:, 3] += math.pi / 2.0 - start_yaw

        clearance_footprint_xy = normalize_xy(
            points_xy=clearance_footprint_xy,
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            rotate_to_start_heading=args.rotate_to_start_heading,
        )

    if args.crop_trajectory_to_plot_roi:
        trajectory_xy = crop_trajectory_by_roi(trajectory_xy, plot_roi)

    if args.smooth_plot_trajectory:
        trajectory_xy = smooth_trajectory_xy(
            trajectory_xy=trajectory_xy,
            window_size=args.smoothing_window,
        )

    print(f"  Trajectory points plotted: {trajectory_xy.shape[0]}")
    print(f"  Occupied cells plotted: {obstacle_cells.shape[0]}")
    print(f"  Corrected d_min (footprint-to-cell area): {clearance.footprint_to_cell_area_m:.4f} m")

    if obstacle_cells.shape[0] > 0:
        print("  Obstacle bounds:")
        print(f"    x: [{np.min(obstacle_cells[:, 0]):.2f}, {np.max(obstacle_cells[:, 0]):.2f}] m")
        print(f"    y: [{np.min(obstacle_cells[:, 1]):.2f}, {np.max(obstacle_cells[:, 1]):.2f}] m")
        print(f"    centroid: ({np.mean(obstacle_cells[:, 0]):.2f}, {np.mean(obstacle_cells[:, 1]):.2f}) m")
    else:
        print("  WARNING: no obstacle points were extracted.")
        print("  Try lowering --obstacle-threshold or removing --obstacle-roi.")

    return BagResult(
        label=label,
        path=path,
        trajectory_xy=trajectory_xy,
        obstacle_cells=obstacle_cells,
        clearance_footprint_xy=clearance_footprint_xy,
        d_min_m=clearance.footprint_to_cell_area_m,
        odom_frame=clearance.odom_frame,
        obstacle_frame=clearance.obstacle_frame,
    )


def apply_fixed_obstacle_reference(
    results: List[BagResult],
    fixed_obstacle_from: Optional[str],
) -> None:
    """
    Replace obstacle points in all panels by the obstacle points from one selected result.

    This is useful for paper figures where the physical obstacle should be shown
    at the same reference position in all panels.
    """
    if not fixed_obstacle_from:
        return

    reference_result = None

    for result in results:
        if result.label == fixed_obstacle_from:
            reference_result = result
            break

    if reference_result is None:
        print("")
        print(f"WARNING: --fixed-obstacle-from label was not found: {fixed_obstacle_from}")
        print("Available labels are:")
        for result in results:
            print(f"  - {result.label}")
        print("Keeping each panel with its own obstacle points.")
        return

    if reference_result.obstacle_cells.size == 0:
        print("")
        print(f"WARNING: selected fixed obstacle reference is empty: {fixed_obstacle_from}")
        print("Keeping each panel with its own obstacle points.")
        return

    fixed_obstacle = reference_result.obstacle_cells.copy()

    print("")
    print(f"Using fixed obstacle reference from: {fixed_obstacle_from}")
    print(f"Fixed obstacle points: {fixed_obstacle.shape[0]}")

    for result in results:
        result.obstacle_cells = fixed_obstacle.copy()


def set_axes_from_roi(
    ax,
    roi: Optional[Tuple[float, float, float, float]],
) -> None:
    if roi is None:
        return

    xmin, xmax, ymin, ymax = roi
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def set_auto_axes(
    ax,
    trajectory_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    margin: float,
    equal_axes: bool,
) -> None:
    arrays = []

    if trajectory_xy.size > 0:
        arrays.append(trajectory_xy)

    if obstacle_xy.size > 0:
        arrays.append(obstacle_xy)

    if not arrays:
        return

    points = np.vstack(arrays)

    xmin = float(np.min(points[:, 0])) - margin
    xmax = float(np.max(points[:, 0])) + margin
    ymin = float(np.min(points[:, 1])) - margin
    ymax = float(np.max(points[:, 1])) + margin

    if equal_axes:
        width = xmax - xmin
        height = ymax - ymin

        if width > height:
            extra = 0.5 * (width - height)
            ymin -= extra
            ymax += extra
        else:
            extra = 0.5 * (height - width)
            xmin -= extra
            xmax += extra

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_occupied_cells(ax, obstacle_cells: np.ndarray, args) -> None:
    if obstacle_cells.size == 0:
        return

    patches = []
    for center_x, center_y, resolution, grid_yaw in obstacle_cells:
        polygon = occupancy_cell_polygon(
            center_x=float(center_x),
            center_y=float(center_y),
            resolution=float(resolution),
            grid_yaw=float(grid_yaw),
        )
        patches.append(MatplotlibPolygon(polygon, closed=True))

    collection = PatchCollection(
        patches,
        facecolor="red",
        edgecolor="darkred",
        linewidth=0.25,
        alpha=args.obstacle_alpha,
        label="_nolegend_",
        zorder=2,
    )
    ax.add_collection(collection)
    ax.scatter(
        [],
        [],
        s=36,
        marker="s",
        facecolor="red",
        edgecolor="darkred",
        alpha=args.obstacle_alpha,
        label=args.obstacle_label,
    )

    if args.show_cell_centers:
        ax.scatter(
            obstacle_cells[:, 0],
            obstacle_cells[:, 1],
            s=args.obstacle_marker_size,
            marker=".",
            color="black",
            alpha=0.7,
            label="Occupied-cell centers",
            zorder=3,
        )


def plot_results(
    results: List[BagResult],
    args,
    plot_roi: Optional[Tuple[float, float, float, float]],
) -> None:
    n = len(results)

    if n == 0:
        raise RuntimeError("No results to plot.")

    fig_height = max(4.0, 4.0 * n)
    fig, axes = plt.subplots(n, 1, figsize=(6.2, fig_height), squeeze=False)
    axes = [axes[i, 0] for i in range(n)]

    for idx, (ax, result) in enumerate(zip(axes, results)):
        trajectory_xy = result.trajectory_xy
        obstacle_cells = result.obstacle_cells
        obstacle_xy = obstacle_cells[:, :2]

        if trajectory_xy.size > 0:
            ax.plot(
                trajectory_xy[:, 0],
                trajectory_xy[:, 1],
                linewidth=args.trajectory_linewidth,
                color="tab:blue",
                label="Trajectory",
            )

        add_occupied_cells(ax, obstacle_cells, args)

        if args.plot_clearance_footprint and result.clearance_footprint_xy.size > 0:
            footprint_patch = MatplotlibPolygon(
                result.clearance_footprint_xy,
                closed=True,
                facecolor="tab:orange",
                edgecolor="0.25",
                linewidth=0.35,
                alpha=0.25,
                label="Footprint",
                zorder=4,
            )
            ax.add_patch(footprint_patch)

        if args.annotate_clearance and math.isfinite(result.d_min_m):
            ax.text(
                0.02,
                0.97,
                rf"$d_{{\min}}={result.d_min_m:.2f}\,\mathrm{{m}}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
                zorder=10,
            )

        if args.plot_obstacle_centroid and obstacle_xy.size > 0:
            centroid_x = float(np.mean(obstacle_xy[:, 0]))
            centroid_y = float(np.mean(obstacle_xy[:, 1]))

            ax.scatter(
                [centroid_x],
                [centroid_y],
                s=85,
                marker="x",
                color="black",
                label="Obstacle centroid",
            )

        ax.set_title(args.title, fontsize=9)
        ax.set_xlabel("X [m]", fontsize=8)
        ax.set_ylabel("Y [m]", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.35)

        # Critical for geometric correctness:
        # 1 meter in X is shown with the same visual length as 1 meter in Y.
        ax.set_aspect("equal", adjustable="box")

        ax.legend(loc=args.legend_loc, fontsize=6, framealpha=0.85, handlelength=1.4, borderpad=0.3, labelspacing=0.25)

        if plot_roi is not None:
            set_axes_from_roi(ax, plot_roi)
        else:
            set_auto_axes(
                ax=ax,
                trajectory_xy=trajectory_xy,
                obstacle_xy=obstacle_xy,
                margin=args.axis_margin,
                equal_axes=args.equal_axes,
            )

        if n > 1:
            panel_letter = chr(ord("a") + idx)
            ax.text(
                0.5,
                -0.18,
                f"({panel_letter}) {result.label}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=11,
            )

    fig.tight_layout()

    output_path = os.path.expanduser(os.path.expandvars(args.output))
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")

    print("")
    print(f"Saved figure: {output_path}")

    if args.show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot global odometry trajectory with global costmap obstacle layer from ROS 2 bags."
    )

    parser.add_argument(
        "--bag",
        action="append",
        required=True,
        help='Bag specification as "Label:/path/to/bag.mcap". Can be repeated.',
    )

    parser.add_argument(
        "--nav2-params",
        default=None,
        help=(
            "Navigation2 YAML used to load and audit the footprint. If omitted, "
            "the verified 9.605 x 3.478 m footprint is used."
        ),
    )

    parser.add_argument(
        "--footprint-polygon",
        default=None,
        help="Explicit footprint polygon; takes precedence over --nav2-params.",
    )

    parser.add_argument(
        "--odom-topic",
        default="/odometry/global",
        help="Odometry topic for the executed trajectory. Default: /odometry/global",
    )

    parser.add_argument(
        "--obstacle-topic",
        default="/global_costmap/obstacle_layer",
        help="OccupancyGrid topic used as obstacle source. Default: /global_costmap/obstacle_layer",
    )

    parser.add_argument(
        "--output",
        default="field_obstacle_costmap_layer.png",
        help="Output figure path.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI. Default: 300.",
    )

    parser.add_argument(
        "--title",
        default="Global trajectory with obstacle position",
        help="Title used in each subplot.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively.",
    )

    parser.add_argument(
        "--max-grids",
        type=int,
        default=200,
        help="Maximum number of OccupancyGrid messages to process per bag. Default: 200.",
    )

    parser.add_argument(
        "--grid-stride",
        type=int,
        default=1,
        help="Keep one OccupancyGrid every N messages. Default: 1.",
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
        help="Occupancy threshold for obstacle cells. Default: 50.",
    )

    parser.add_argument(
        "--max-obstacle-points",
        type=int,
        default=5000,
        help="Maximum number of obstacle cells plotted per bag. Default: 5000.",
    )

    parser.add_argument(
        "--max-obstacle-cells-per-grid",
        type=int,
        default=50000,
        help="Safety limit used by corrected clearance calculation. Default: 50000.",
    )

    parser.add_argument(
        "--clearance-max-time-delta-s",
        type=float,
        default=1.0,
        help="Maximum grid/odometry synchronization difference. Default: 1.0 s.",
    )

    parser.add_argument(
        "--frame-check",
        choices=["strict", "warn"],
        default="strict",
        help="Reject odometry/obstacle frame mismatch by default.",
    )

    parser.add_argument(
        "--obstacle-marker-size",
        type=float,
        default=12.0,
        help="Marker size for obstacle cells. Default: 12.0.",
    )

    parser.add_argument(
        "--obstacle-alpha",
        type=float,
        default=0.42,
        help="Opacity of occupied cell polygons. Default: 0.42.",
    )

    parser.add_argument(
        "--show-cell-centers",
        action="store_true",
        help="Also mark the centers of occupied cells.",
    )

    parser.add_argument(
        "--plot-clearance-footprint",
        action="store_true",
        help="Draw the oriented truck footprint at the synchronized minimum-clearance pose.",
    )

    parser.add_argument(
        "--annotate-clearance",
        action="store_true",
        help="Annotate each panel with the corrected footprint-to-cell-area d_min.",
    )

    parser.add_argument(
        "--obstacle-label",
        default="Obstacle layer",
        help="Legend label for the obstacle markers.",
    )

    parser.add_argument(
        "--legend-loc",
        choices=["upper right", "upper left", "lower right", "lower left", "best"],
        default="upper right",
        help="Legend position in each panel. Default: upper right.",
    )

    parser.add_argument(
        "--trajectory-linewidth",
        type=float,
        default=2.0,
        help="Trajectory line width. Default: 2.0.",
    )

    parser.add_argument(
        "--plot-obstacle-centroid",
        action="store_true",
        help="Plot obstacle centroid as a black x marker.",
    )

    parser.add_argument(
        "--fixed-obstacle-from",
        default=None,
        help=(
            "Use obstacle points from the bag with this label as a fixed obstacle reference "
            "for all panels. Example: --fixed-obstacle-from 'Nav2 with SMAC planner'"
        ),
    )

    parser.add_argument(
        "--normalize-start",
        action="store_true",
        help="Subtract the first odometry pose from trajectory and obstacle points.",
    )

    parser.add_argument(
        "--rotate-to-start-heading",
        action="store_true",
        help="When used with --normalize-start, rotate each run so the initial heading is aligned with the positive Y axis (north).",
    )

    parser.add_argument(
        "--plot-roi",
        nargs=4,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Plot axis ROI: xmin xmax ymin ymax.",
    )

    parser.add_argument(
        "--obstacle-roi",
        nargs=4,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Obstacle extraction ROI: xmin xmax ymin ymax.",
    )

    parser.add_argument(
        "--crop-trajectory-to-plot-roi",
        action="store_true",
        help="Crop trajectory points outside --plot-roi.",
    )

    parser.add_argument(
        "--smooth-plot-trajectory",
        action="store_true",
        help="Smooth the plotted trajectory for visualization only.",
    )

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=31,
        help="Moving-average window for plotted trajectory smoothing. Default: 31.",
    )

    parser.add_argument(
        "--equal-axes",
        action="store_true",
        default=True,
        help="Use equal-ish X/Y data limits when --plot-roi is not provided. Enabled by default.",
    )

    parser.add_argument(
        "--axis-margin",
        type=float,
        default=10.0,
        help="Axis margin used when --plot-roi is not provided. Default: 10.0.",
    )

    args = parser.parse_args()

    if args.clearance_max_time_delta_s < 0.0:
        parser.error("--clearance-max-time-delta-s must be non-negative")

    if args.max_obstacle_cells_per_grid <= 0:
        parser.error("--max-obstacle-cells-per-grid must be positive")

    if not 0.0 <= args.obstacle_alpha <= 1.0:
        parser.error("--obstacle-alpha must be between 0 and 1")

    if args.fixed_obstacle_from and (args.plot_clearance_footprint or args.annotate_clearance):
        parser.error(
            "--fixed-obstacle-from changes the displayed obstacle geometry and cannot be "
            "combined with --plot-clearance-footprint or --annotate-clearance"
        )

    plot_roi = parse_roi(args.plot_roi)
    obstacle_roi = parse_roi(args.obstacle_roi)
    footprint_local, footprint_source = resolve_footprint(args)

    print("")
    print("Figure geometry configuration:")
    print(f"  Footprint source: {footprint_source}")
    print(f"  Footprint vertices [base_footprint]: {footprint_to_string(footprint_local)}")

    bag_specs = [parse_bag_arg(value) for value in args.bag]

    results: List[BagResult] = []

    for label, path in bag_specs:
        result = compute_bag_result(
            label=label,
            path=path,
            args=args,
            plot_roi=plot_roi,
            obstacle_roi=obstacle_roi,
            footprint_local=footprint_local,
        )
        results.append(result)

    apply_fixed_obstacle_reference(
        results=results,
        fixed_obstacle_from=args.fixed_obstacle_from,
    )

    plot_results(
        results=results,
        args=args,
        plot_roi=plot_roi,
    )


if __name__ == "__main__":
    main()
