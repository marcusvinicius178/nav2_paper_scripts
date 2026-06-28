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
    Red points: occupied cells from the global obstacle layer.

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
    print("  /usr/bin/python3 ~/NAV2_Paper_Scripts/plot_field_obstacle_and_trajectory_from_bags.py ...")
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
    obstacle_xy: np.ndarray


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
        c = math.cos(-start_yaw)
        s = math.sin(-start_yaw)
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

    return odom_samples, grid_samples


def extract_obstacle_points_from_grids(
    grids: List[GridSample],
    threshold: int,
    max_points: int,
    latest_grid_only: bool,
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

    # Remove duplicate occupied cells.
    rounded = np.round(all_points, decimals=3)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    all_points = all_points[np.sort(unique_idx)]

    if all_points.shape[0] > max_points:
        idx = np.linspace(0, all_points.shape[0] - 1, max_points).astype(int)
        all_points = all_points[idx]

    return all_points


def compute_bag_result(
    label: str,
    path: str,
    args,
    plot_roi: Optional[Tuple[float, float, float, float]],
    obstacle_roi: Optional[Tuple[float, float, float, float]],
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

    obstacle_xy = extract_obstacle_points_from_grids(
        grids=grid_samples,
        threshold=args.obstacle_threshold,
        max_points=args.max_obstacle_points,
        latest_grid_only=args.latest_grid_only,
    )

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

        obstacle_xy = normalize_xy(
            points_xy=obstacle_xy,
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            rotate_to_start_heading=args.rotate_to_start_heading,
        )

    obstacle_xy = apply_roi(obstacle_xy, obstacle_roi)

    if args.crop_trajectory_to_plot_roi:
        trajectory_xy = crop_trajectory_by_roi(trajectory_xy, plot_roi)

    if args.smooth_plot_trajectory:
        trajectory_xy = smooth_trajectory_xy(
            trajectory_xy=trajectory_xy,
            window_size=args.smoothing_window,
        )

    print(f"  Trajectory points plotted: {trajectory_xy.shape[0]}")
    print(f"  Obstacle points plotted: {obstacle_xy.shape[0]}")

    if obstacle_xy.shape[0] > 0:
        print("  Obstacle bounds:")
        print(f"    x: [{np.min(obstacle_xy[:, 0]):.2f}, {np.max(obstacle_xy[:, 0]):.2f}] m")
        print(f"    y: [{np.min(obstacle_xy[:, 1]):.2f}, {np.max(obstacle_xy[:, 1]):.2f}] m")
        print(f"    centroid: ({np.mean(obstacle_xy[:, 0]):.2f}, {np.mean(obstacle_xy[:, 1]):.2f}) m")
    else:
        print("  WARNING: no obstacle points were extracted.")
        print("  Try lowering --obstacle-threshold or removing --obstacle-roi.")

    return BagResult(
        label=label,
        path=path,
        trajectory_xy=trajectory_xy,
        obstacle_xy=obstacle_xy,
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

    if reference_result.obstacle_xy.size == 0:
        print("")
        print(f"WARNING: selected fixed obstacle reference is empty: {fixed_obstacle_from}")
        print("Keeping each panel with its own obstacle points.")
        return

    fixed_obstacle = reference_result.obstacle_xy.copy()

    print("")
    print(f"Using fixed obstacle reference from: {fixed_obstacle_from}")
    print(f"Fixed obstacle points: {fixed_obstacle.shape[0]}")

    for result in results:
        result.obstacle_xy = fixed_obstacle.copy()


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
        obstacle_xy = result.obstacle_xy

        if trajectory_xy.size > 0:
            ax.plot(
                trajectory_xy[:, 0],
                trajectory_xy[:, 1],
                linewidth=args.trajectory_linewidth,
                color="tab:blue",
                label="Global trajectory",
            )

        if obstacle_xy.size > 0:
            ax.scatter(
                obstacle_xy[:, 0],
                obstacle_xy[:, 1],
                s=args.obstacle_marker_size,
                marker="o",
                color="red",
                alpha=0.85,
                label=args.obstacle_label,
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

        ax.set_title(args.title)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, alpha=0.35)

        # Critical for geometric correctness:
        # 1 meter in X is shown with the same visual length as 1 meter in Y.
        ax.set_aspect("equal", adjustable="box")

        ax.legend(loc="best", fontsize=8)

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
        "--obstacle-marker-size",
        type=float,
        default=12.0,
        help="Marker size for obstacle cells. Default: 12.0.",
    )

    parser.add_argument(
        "--obstacle-label",
        default="Obstacle layer",
        help="Legend label for the obstacle markers.",
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
        help="When used with --normalize-start, rotate each run so the initial heading is aligned.",
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

    plot_roi = parse_roi(args.plot_roi)
    obstacle_roi = parse_roi(args.obstacle_roi)

    bag_specs = [parse_bag_arg(value) for value in args.bag]

    results: List[BagResult] = []

    for label, path in bag_specs:
        result = compute_bag_result(
            label=label,
            path=path,
            args=args,
            plot_roi=plot_roi,
            obstacle_roi=obstacle_roi,
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