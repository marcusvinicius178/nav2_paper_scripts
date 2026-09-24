#!/usr/bin/env python3
"""Shared, ROS-aware helpers for the field rosbag audit v3.

The module is intentionally importable on computers without ROS 2. Pure
geometry, timestamp, covariance, and configuration tests therefore run in a
regular Python environment. ROS 2 imports are performed only when a bag is
opened.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - reported explicitly by load_yaml
    yaml = None


SCHEMA_VERSION = "field_audit_v3.0"
NA = "NA"

# Exact footprint used by the corrected field baseline CSV (9.625 x 3.498 m).
# A recorded /global_costmap/published_footprint always takes precedence.
BASELINE_FALLBACK_FOOTPRINT = np.array(
    [
        [-4.8125, -1.7490],
        [4.8125, -1.7490],
        [4.8125, 1.7490],
        [-4.8125, 1.7490],
    ],
    dtype=float,
)

BASELINE_CLEARANCE_M = {
    "human_driver": 1.9305638377844048,
    "navfn_mppi": 0.4053553578037854,
    "smac_mppi": 3.6877051369750995,
}

DEFAULT_TOPIC_CANDIDATES: Dict[str, List[str]] = {
    "odometry": ["/odometry/global", "/odom", "/odometry/filtered"],
    "obstacle_grid": [
        "/global_costmap/obstacle_layer",
        "/global_costmap/costmap_raw",
        "/global_costmap/costmap",
    ],
    "published_footprint": [
        "/global_costmap/published_footprint",
        "/local_costmap/published_footprint",
    ],
    "gps": ["/gps/fix", "/fix", "/navsat/fix"],
    "imu": ["/imu/data", "/imu/data_raw"],
    "tf": ["/tf"],
    "tf_static": ["/tf_static"],
    "vehicle_state": [
        "/vehicle/status/velocity_status",
        "/vehicle/status",
        "/vehicle_state",
        "/velocity_status",
    ],
    "rtk_explicit": [
        "/gps/rtk_status",
        "/gnss/rtk_status",
        "/ublox/navpvt",
        "/novatel/oem7/bestpos",
        "/fixposition/odometry_enu",
    ],
}

EXPECTED_TYPES: Dict[str, Tuple[str, ...]] = {
    "odometry": ("nav_msgs/msg/Odometry",),
    "obstacle_grid": ("nav_msgs/msg/OccupancyGrid",),
    "published_footprint": ("geometry_msgs/msg/PolygonStamped",),
    "gps": ("sensor_msgs/msg/NavSatFix",),
    "imu": ("sensor_msgs/msg/Imu",),
    "tf": ("tf2_msgs/msg/TFMessage",),
    "tf_static": ("tf2_msgs/msg/TFMessage",),
}


@dataclass(frozen=True)
class BagSpec:
    key: str
    label: str
    path: str


@dataclass
class OdomSample:
    storage_stamp_sec: float
    header_stamp_sec: Optional[float]
    frame_id: str
    child_frame_id: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    vz: float
    pose_covariance: np.ndarray
    twist_covariance: np.ndarray

    def stamp(self, domain: str) -> float:
        if domain == "header":
            if self.header_stamp_sec is None:
                raise ValueError("header timestamp unavailable for odometry sample")
            return float(self.header_stamp_sec)
        return float(self.storage_stamp_sec)


@dataclass
class GridSample:
    storage_stamp_sec: float
    header_stamp_sec: Optional[float]
    frame_id: str
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: np.ndarray

    def stamp(self, domain: str) -> float:
        if domain == "header":
            if self.header_stamp_sec is None:
                raise ValueError("header timestamp unavailable for grid sample")
            return float(self.header_stamp_sec)
        return float(self.storage_stamp_sec)


@dataclass
class FootprintSample:
    storage_stamp_sec: float
    header_stamp_sec: Optional[float]
    frame_id: str
    vertices: np.ndarray

    def stamp(self, domain: str) -> float:
        if domain == "header":
            if self.header_stamp_sec is None:
                raise ValueError("header timestamp unavailable for footprint sample")
            return float(self.header_stamp_sec)
        return float(self.storage_stamp_sec)


@dataclass(frozen=True)
class SyncPair:
    grid_index: int
    pose_index: int
    grid_stamp_sec: float
    pose_stamp_sec: float
    delta_signed_s: float
    delta_abs_s: float
    time_domain: str
    accepted: bool


@dataclass
class ClearanceEvaluation:
    clearance_area_m: float
    clearance_center_m: float
    base_to_center_m: float
    closest_cell_center_x: float
    closest_cell_center_y: float
    occupied_cells: int
    threshold: int


@dataclass
class BagMetricData:
    spec: BagSpec
    topics: Dict[str, str]
    topic_types: Dict[str, str]
    odometry: List[OdomSample] = field(default_factory=list)
    grids: List[GridSample] = field(default_factory=list)
    footprints: List[FootprintSample] = field(default_factory=list)
    observed_occupancy_values: set[int] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)


def normalize_frame(value: Any) -> str:
    return str(value or "").strip().lstrip("/")


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def stamp_to_sec(stamp: Any) -> Optional[float]:
    if stamp is None:
        return None
    try:
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def get_header(msg: Any) -> Any:
    return getattr(msg, "header", None)


def get_header_stamp(msg: Any) -> Optional[float]:
    return stamp_to_sec(getattr(get_header(msg), "stamp", None))


def get_frame_id(msg: Any) -> str:
    return normalize_frame(getattr(get_header(msg), "frame_id", ""))


def quaternion_to_yaw(quaternion: Any) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def circular_std(values: Sequence[float]) -> Optional[float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return None
    resultant = math.hypot(float(np.mean(np.cos(array))), float(np.mean(np.sin(array))))
    resultant = min(1.0, max(1e-15, resultant))
    return math.sqrt(max(0.0, -2.0 * math.log(resultant)))


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.percentile(array, quantile))


def numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "min": NA, "median": NA, "mean": NA, "p95": NA, "max": NA}
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def interval_summary(stamps: Sequence[float], gap_factor: float = 5.0) -> Dict[str, Any]:
    array = np.asarray(stamps, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "start_sec": NA,
            "end_sec": NA,
            "duration_s": NA,
            "frequency_mean_hz": NA,
            "dt_median_s": NA,
            "dt_p95_s": NA,
            "dt_max_s": NA,
            "duplicates": 0,
            "out_of_order": 0,
            "gap_threshold_s": NA,
            "gap_count": 0,
            "gap_total_s": 0.0,
        }
    deltas_in_order = np.diff(array)
    positive = deltas_in_order[deltas_in_order > 0.0]
    duplicates = int(np.sum(deltas_in_order == 0.0))
    out_of_order = int(np.sum(deltas_in_order < 0.0))
    duration = float(np.max(array) - np.min(array))
    dt_median = float(np.median(positive)) if positive.size else None
    gap_threshold = (gap_factor * dt_median) if dt_median and dt_median > 0.0 else None
    gaps = positive[positive > gap_threshold] if gap_threshold else np.empty((0,), dtype=float)
    return {
        "n": int(array.size),
        "start_sec": float(np.min(array)),
        "end_sec": float(np.max(array)),
        "duration_s": duration,
        "frequency_mean_hz": float((array.size - 1) / duration) if duration > 0.0 else NA,
        "dt_median_s": dt_median if dt_median is not None else NA,
        "dt_p95_s": float(np.percentile(positive, 95.0)) if positive.size else NA,
        "dt_max_s": float(np.max(positive)) if positive.size else NA,
        "duplicates": duplicates,
        "out_of_order": out_of_order,
        "gap_threshold_s": gap_threshold if gap_threshold is not None else NA,
        "gap_count": int(gaps.size),
        "gap_total_s": float(np.sum(gaps)) if gaps.size else 0.0,
    }


def safe_array(value: Any, expected_size: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full((expected_size,), np.nan, dtype=float)
    if array.size != expected_size:
        return np.full((expected_size,), np.nan, dtype=float)
    return array


def covariance_state(values: Sequence[float]) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return "invalid_or_nonfinite"
    if np.allclose(array, 0.0):
        return "zero_or_unknown"
    if array[0] < 0.0:
        return "unknown_sentinel"
    return "available"


def horizontal_sigma_from_covariance(covariance: Sequence[float], dimension: int) -> Optional[float]:
    array = np.asarray(covariance, dtype=float).reshape(-1)
    if dimension == 6 and array.size == 36:
        xx, yy = array[0], array[7]
    elif dimension == 3 and array.size == 9:
        xx, yy = array[0], array[4]
    else:
        return None
    if not (math.isfinite(xx) and math.isfinite(yy)) or xx < 0.0 or yy < 0.0:
        return None
    if xx == 0.0 and yy == 0.0:
        return None
    return math.sqrt(max(0.0, xx + yy))


def yaw_sigma_from_covariance(covariance: Sequence[float], dimension: int) -> Optional[float]:
    array = np.asarray(covariance, dtype=float).reshape(-1)
    index = 35 if dimension == 6 and array.size == 36 else 8 if dimension == 3 and array.size == 9 else None
    if index is None:
        return None
    variance = float(array[index])
    if not math.isfinite(variance) or variance <= 0.0:
        return None
    return math.sqrt(variance)


def pose_xy_covariance(covariance: Sequence[float]) -> Optional[np.ndarray]:
    array = np.asarray(covariance, dtype=float).reshape(-1)
    if array.size != 36 or not np.all(np.isfinite(array)):
        return None
    matrix = np.array([[array[0], array[1]], [array[6], array[7]]], dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    if np.allclose(matrix, 0.0):
        return None
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.min(eigenvalues)) < -1e-9:
        return None
    return matrix


def parse_bag_argument(value: str) -> BagSpec:
    parts = value.split(":", 2)
    if len(parts) == 3:
        key, label, path = parts
    elif len(parts) == 2:
        label, path = parts
        key = slugify(label)
    else:
        path = value
        label = Path(path).stem
        key = slugify(label)
    key = slugify(key)
    label = label.strip() or key
    path = os.path.expanduser(os.path.expandvars(path.strip()))
    if not key or not path:
        raise ValueError(f"Invalid --bag specification: {value!r}")
    return BagSpec(key=key, label=label, path=path)


def slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(value))
    return "_".join(part for part in cleaned.split("_") if part)


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install the packages listed in requirements_v3.txt.")
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return document


def resolve_path(value: str, project_root: Path) -> Path:
    expanded = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if expanded.is_absolute():
        return expanded
    return (project_root / expanded).resolve()


def bag_specs_from_config(config: Mapping[str, Any], project_root: Path) -> List[BagSpec]:
    raw_bags = config.get("bags", [])
    if not isinstance(raw_bags, list):
        raise ValueError("config key 'bags' must be a list")
    result: List[BagSpec] = []
    for raw in raw_bags:
        if not isinstance(raw, Mapping):
            raise ValueError("every bag entry must be a mapping")
        key = slugify(str(raw.get("key", "")))
        label = str(raw.get("label", key)).strip()
        raw_path = str(raw.get("path", "")).strip()
        if not key or not raw_path:
            raise ValueError(f"bag entry requires key and path: {raw}")
        result.append(BagSpec(key=key, label=label or key, path=str(resolve_path(raw_path, project_root))))
    return result


def merge_bag_specs(config_specs: Sequence[BagSpec], cli_specs: Sequence[str]) -> List[BagSpec]:
    if cli_specs:
        parsed = [parse_bag_argument(value) for value in cli_specs]
    else:
        parsed = list(config_specs)
    keys = [item.key for item in parsed]
    if len(keys) != len(set(keys)):
        raise ValueError(f"bag keys must be unique: {keys}")
    return parsed


def validate_bag_paths(specs: Sequence[BagSpec]) -> None:
    missing = [spec.path for spec in specs if not Path(spec.path).exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"Rosbag path(s) not found:\n  - {formatted}")


def infer_storage_id(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".db3"):
        return "sqlite3"
    if lower.endswith(".mcap"):
        return "mcap"
    metadata = Path(path) / "metadata.yaml"
    if metadata.is_file():
        text = metadata.read_text(encoding="utf-8", errors="replace").lower()
        if "sqlite3" in text:
            return "sqlite3"
        if "mcap" in text:
            return "mcap"
    return "mcap"


def ros_imports() -> Tuple[Any, Any, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:  # pragma: no cover - depends on ROS host
        raise RuntimeError(
            "ROS 2 Python modules are unavailable. Source the ROS 2 environment "
            "and run with its system Python, for example: "
            "source /opt/ros/iron/setup.bash. Original import error: " + repr(exc)
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def open_bag_reader(path: str) -> Any:
    rosbag2_py, _deserialize, _get_message = ros_imports()
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=infer_storage_id(path))
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def topic_type_map(reader: Any) -> Dict[str, str]:
    return {item.name: item.type for item in reader.get_all_topics_and_types()}


def message_classes(types: Iterable[str]) -> Dict[str, Any]:
    _rosbag2_py, _deserialize, get_message = ros_imports()
    result: Dict[str, Any] = {}
    for type_name in set(types):
        try:
            result[type_name] = get_message(type_name)
        except Exception:
            result[type_name] = None
    return result


def iter_bag_messages(
    path: str,
    selected_topics: Optional[set[str]] = None,
) -> Iterator[Tuple[str, str, float, Optional[Any], Optional[str]]]:
    reader = open_bag_reader(path)
    type_by_topic = topic_type_map(reader)
    selected_types = [
        type_name for topic, type_name in type_by_topic.items()
        if selected_topics is None or topic in selected_topics
    ]
    classes = message_classes(selected_types)
    _rosbag2_py, deserialize_message, _get_message = ros_imports()
    while reader.has_next():
        topic, raw, storage_ns = reader.read_next()
        if selected_topics is not None and topic not in selected_topics:
            continue
        type_name = type_by_topic.get(topic, "")
        message_class = classes.get(type_name)
        if message_class is None:
            yield topic, type_name, float(storage_ns) * 1e-9, None, "message type not installed"
            continue
        try:
            message = deserialize_message(raw, message_class)
        except Exception as exc:
            yield topic, type_name, float(storage_ns) * 1e-9, None, f"deserialize failed: {exc}"
            continue
        yield topic, type_name, float(storage_ns) * 1e-9, message, None


def resolve_topic(
    type_by_topic: Mapping[str, str],
    role: str,
    configured_candidates: Sequence[str],
) -> Optional[str]:
    candidates = [str(value) for value in configured_candidates]
    for candidate in candidates:
        if candidate in type_by_topic:
            expected = EXPECTED_TYPES.get(role)
            if expected is None or type_by_topic[candidate] in expected:
                return candidate
    expected = EXPECTED_TYPES.get(role, ())
    typed = sorted(topic for topic, type_name in type_by_topic.items() if type_name in expected)
    if len(typed) == 1:
        return typed[0]
    return None


def merged_topic_candidates(config: Mapping[str, Any]) -> Dict[str, List[str]]:
    configured = config.get("topic_candidates", {})
    result = {key: list(values) for key, values in DEFAULT_TOPIC_CANDIDATES.items()}
    if isinstance(configured, Mapping):
        for role, values in configured.items():
            if isinstance(values, str):
                values = [values]
            if isinstance(values, Sequence):
                result[str(role)] = [str(value) for value in values]
    return result


def odometry_from_message(message: Any, storage_stamp_sec: float) -> OdomSample:
    pose_with_cov = message.pose
    twist_with_cov = message.twist
    pose = pose_with_cov.pose
    twist = twist_with_cov.twist
    return OdomSample(
        storage_stamp_sec=float(storage_stamp_sec),
        header_stamp_sec=get_header_stamp(message),
        frame_id=get_frame_id(message),
        child_frame_id=normalize_frame(getattr(message, "child_frame_id", "")),
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=quaternion_to_yaw(pose.orientation),
        vx=float(twist.linear.x),
        vy=float(twist.linear.y),
        vz=float(twist.linear.z),
        pose_covariance=safe_array(getattr(pose_with_cov, "covariance", []), 36),
        twist_covariance=safe_array(getattr(twist_with_cov, "covariance", []), 36),
    )


def grid_from_message(message: Any, storage_stamp_sec: float) -> GridSample:
    info = message.info
    width = int(info.width)
    height = int(info.height)
    data = np.asarray(message.data, dtype=np.int16)
    if data.size != width * height:
        raise ValueError(f"OccupancyGrid data size {data.size} != {width} x {height}")
    origin = info.origin
    return GridSample(
        storage_stamp_sec=float(storage_stamp_sec),
        header_stamp_sec=get_header_stamp(message),
        frame_id=get_frame_id(message),
        resolution=float(info.resolution),
        width=width,
        height=height,
        origin_x=float(origin.position.x),
        origin_y=float(origin.position.y),
        origin_yaw=quaternion_to_yaw(origin.orientation),
        data=data.reshape(height, width),
    )


def footprint_from_message(message: Any, storage_stamp_sec: float) -> FootprintSample:
    points = getattr(getattr(message, "polygon", None), "points", [])
    vertices = np.array([[float(point.x), float(point.y)] for point in points], dtype=float)
    return FootprintSample(
        storage_stamp_sec=float(storage_stamp_sec),
        header_stamp_sec=get_header_stamp(message),
        frame_id=get_frame_id(message),
        vertices=validate_polygon(vertices),
    )


def load_metric_bag(
    spec: BagSpec,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> BagMetricData:
    reader = open_bag_reader(spec.path)
    type_by_topic = topic_type_map(reader)
    candidates = merged_topic_candidates(config)
    topics: Dict[str, str] = {}
    for role in ("odometry", "obstacle_grid", "published_footprint"):
        selected = resolve_topic(type_by_topic, role, candidates.get(role, []))
        if selected:
            topics[role] = selected
    result = BagMetricData(spec=spec, topics=topics, topic_types=dict(type_by_topic))
    for required in ("odometry", "obstacle_grid"):
        if required not in topics:
            result.warnings.append(
                f"{required} topic not resolved; candidates={candidates.get(required, [])}"
            )
    selected_topics = set(topics.values())
    if not selected_topics:
        return result
    logger.info("%s selected topics: %s", spec.label, topics)
    for topic, _type_name, storage_stamp, message, error in iter_bag_messages(spec.path, selected_topics):
        if error:
            result.warnings.append(f"{topic}: {error}")
            continue
        try:
            if topic == topics.get("odometry"):
                result.odometry.append(odometry_from_message(message, storage_stamp))
            elif topic == topics.get("obstacle_grid"):
                grid = grid_from_message(message, storage_stamp)
                result.grids.append(grid)
                unique = np.unique(grid.data)
                result.observed_occupancy_values.update(int(value) for value in unique.tolist())
            elif topic == topics.get("published_footprint"):
                result.footprints.append(footprint_from_message(message, storage_stamp))
        except Exception as exc:
            result.warnings.append(f"{topic}: parse failed: {exc}")
    return result


def choose_common_time_domain(
    poses: Sequence[OdomSample],
    grids: Sequence[GridSample],
    minimum_header_fraction: float = 0.95,
) -> Tuple[str, str]:
    if not poses or not grids:
        return "storage", "insufficient samples for header-domain audit"
    pose_fraction = sum(sample.header_stamp_sec is not None for sample in poses) / len(poses)
    grid_fraction = sum(sample.header_stamp_sec is not None for sample in grids) / len(grids)
    if pose_fraction < minimum_header_fraction or grid_fraction < minimum_header_fraction:
        return (
            "storage",
            f"header availability below {minimum_header_fraction:.2f} "
            f"(pose={pose_fraction:.3f}, grid={grid_fraction:.3f})",
        )
    pose_header = [sample.header_stamp_sec for sample in poses if sample.header_stamp_sec is not None]
    grid_header = [sample.header_stamp_sec for sample in grids if sample.header_stamp_sec is not None]
    overlap = min(max(pose_header), max(grid_header)) - max(min(pose_header), min(grid_header))
    if overlap <= 0.0:
        return "storage", "header timestamp ranges do not overlap"
    return "header", "both topics have valid overlapping header timestamps"


def pair_grids_to_nearest_poses(
    poses: Sequence[OdomSample],
    grids: Sequence[GridSample],
    domain: str,
    maximum_delta_s: float,
) -> List[SyncPair]:
    if maximum_delta_s < 0.0:
        raise ValueError("maximum_delta_s must be non-negative")
    pose_entries = [
        (sample.stamp(domain), index)
        for index, sample in enumerate(poses)
        if domain == "storage" or sample.header_stamp_sec is not None
    ]
    pose_entries.sort(key=lambda item: item[0])
    if not pose_entries:
        return []
    pose_stamps = np.array([item[0] for item in pose_entries], dtype=float)
    result: List[SyncPair] = []
    for grid_index, grid in enumerate(grids):
        if domain == "header" and grid.header_stamp_sec is None:
            continue
        grid_stamp = grid.stamp(domain)
        insertion = int(np.searchsorted(pose_stamps, grid_stamp))
        candidates: List[int] = []
        if insertion < pose_stamps.size:
            candidates.append(insertion)
        if insertion > 0:
            candidates.append(insertion - 1)
        if not candidates:
            continue
        local_index = min(candidates, key=lambda index: abs(float(pose_stamps[index]) - grid_stamp))
        pose_stamp, pose_index = pose_entries[local_index]
        signed = float(pose_stamp - grid_stamp)
        absolute = abs(signed)
        result.append(
            SyncPair(
                grid_index=grid_index,
                pose_index=pose_index,
                grid_stamp_sec=float(grid_stamp),
                pose_stamp_sec=float(pose_stamp),
                delta_signed_s=signed,
                delta_abs_s=absolute,
                time_domain=domain,
                accepted=absolute <= maximum_delta_s,
            )
        )
    return result


def summarize_sync_pairs(
    pairs: Sequence[SyncPair],
    tolerances: Sequence[float],
) -> Dict[str, Any]:
    absolute = [pair.delta_abs_s for pair in pairs]
    accepted = [pair.delta_abs_s for pair in pairs if pair.accepted]
    summary = numeric_summary(absolute)
    result: Dict[str, Any] = {
        "n_candidate_pairs": len(pairs),
        "n_pairs_within_max_tolerance": len(accepted),
        "delta_abs_min_s": summary["min"],
        "delta_abs_median_s": summary["median"],
        "delta_abs_p95_s": summary["p95"],
        "delta_abs_max_s": summary["max"],
    }
    for tolerance in tolerances:
        key = f"fraction_within_{tolerance:.2f}s".replace(".", "p")
        result[key] = (
            sum(value <= tolerance for value in absolute) / len(absolute)
            if absolute else NA
        )
    return result


def polygon_signed_area(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=float)
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def validate_polygon(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
        raise ValueError("polygon must be an N x 2 array with at least three vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("polygon contains non-finite coordinates")
    if np.linalg.norm(vertices[0] - vertices[-1]) <= 1e-12:
        vertices = vertices[:-1]
    area = polygon_signed_area(vertices)
    if abs(area) <= 1e-12:
        raise ValueError("polygon has zero area")
    if area < 0.0:
        vertices = vertices[::-1].copy()
    cross_values: List[float] = []
    for index in range(vertices.shape[0]):
        first = vertices[(index + 1) % vertices.shape[0]] - vertices[index]
        second = vertices[(index + 2) % vertices.shape[0]] - vertices[(index + 1) % vertices.shape[0]]
        cross_values.append(float(first[0] * second[1] - first[1] * second[0]))
    if any(value < -1e-9 for value in cross_values):
        raise ValueError("polygon must be convex and vertices must be ordered")
    return vertices


def parse_polygon(value: Any) -> np.ndarray:
    parsed = value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    return validate_polygon(np.asarray(parsed, dtype=float))


def transform_polygon(vertices: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return validate_polygon(vertices) @ rotation.T + np.array([x, y], dtype=float)


def pad_convex_polygon(vertices: np.ndarray, padding_m: float) -> np.ndarray:
    vertices = validate_polygon(vertices)
    padding_m = float(padding_m)
    if abs(padding_m) <= 1e-12:
        return vertices.copy()
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("cannot pad polygon with zero-length edge")
    # For a counter-clockwise polygon, [dy, -dx] is the outward unit normal.
    normals = np.column_stack([edges[:, 1], -edges[:, 0]]) / lengths[:, None]
    constants = np.sum(normals * vertices, axis=1) + padding_m
    padded_vertices: List[np.ndarray] = []
    for vertex_index in range(vertices.shape[0]):
        previous_edge = (vertex_index - 1) % vertices.shape[0]
        current_edge = vertex_index
        matrix = np.vstack([normals[previous_edge], normals[current_edge]])
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= 1e-12:
            raise ValueError("cannot pad polygon with parallel adjacent edges")
        vector = np.array([constants[previous_edge], constants[current_edge]], dtype=float)
        padded_vertices.append(np.linalg.solve(matrix, vector))
    return validate_polygon(np.asarray(padded_vertices, dtype=float))


def point_distances_to_convex_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    polygon = validate_polygon(polygon)
    if points.size == 0:
        return np.empty((0,), dtype=float)
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    edges = ends - starts
    relative = points[:, None, :] - starts[None, :, :]
    denominators = np.sum(edges * edges, axis=1)
    projections = np.clip(
        np.sum(relative * edges[None, :, :], axis=2) / denominators[None, :],
        0.0,
        1.0,
    )
    closest = starts[None, :, :] + projections[:, :, None] * edges[None, :, :]
    edge_distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    inside = np.all(cross >= -1e-9, axis=1)
    distances = np.min(edge_distances, axis=1)
    distances[inside] = 0.0
    return distances


def convex_polygons_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    first = validate_polygon(first)
    second = validate_polygon(second)
    for polygon in (first, second):
        ends = np.roll(polygon, -1, axis=0)
        edges = ends - polygon
        axes = np.column_stack([-edges[:, 1], edges[:, 0]])
        for axis in axes:
            first_projection = first @ axis
            second_projection = second @ axis
            if float(np.max(first_projection)) < float(np.min(second_projection)) - 1e-9:
                return False
            if float(np.max(second_projection)) < float(np.min(first_projection)) - 1e-9:
                return False
    return True


def convex_polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = validate_polygon(first)
    second = validate_polygon(second)
    if convex_polygons_intersect(first, second):
        return 0.0
    first_to_second = point_distances_to_convex_polygon(first, second)
    second_to_first = point_distances_to_convex_polygon(second, first)
    return float(min(np.min(first_to_second), np.min(second_to_first)))


def occupancy_centers(grid: GridSample, threshold: int) -> np.ndarray:
    rows, columns = np.where(grid.data >= int(threshold))
    if rows.size == 0:
        return np.empty((0, 2), dtype=float)
    local_x = (columns.astype(float) + 0.5) * grid.resolution
    local_y = (rows.astype(float) + 0.5) * grid.resolution
    c = math.cos(grid.origin_yaw)
    s = math.sin(grid.origin_yaw)
    global_x = grid.origin_x + c * local_x - s * local_y
    global_y = grid.origin_y + s * local_x + c * local_y
    return np.column_stack([global_x, global_y])


def occupancy_cell_polygon(center: np.ndarray, resolution: float, yaw: float) -> np.ndarray:
    half = float(resolution) / 2.0
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
    return transform_polygon(local, float(center[0]), float(center[1]), float(yaw))


def minimum_clearance(
    footprint_world: np.ndarray,
    robot_center_xy: np.ndarray,
    centers: np.ndarray,
    cell_resolution: float,
    grid_yaw: float,
    threshold: int,
) -> ClearanceEvaluation:
    footprint_world = validate_polygon(footprint_world)
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return ClearanceEvaluation(
            clearance_area_m=float("nan"),
            clearance_center_m=float("nan"),
            base_to_center_m=float("nan"),
            closest_cell_center_x=float("nan"),
            closest_cell_center_y=float("nan"),
            occupied_cells=0,
            threshold=int(threshold),
        )
    center_distances = point_distances_to_convex_polygon(centers, footprint_world)
    base_distances = np.linalg.norm(centers - np.asarray(robot_center_xy, dtype=float)[None, :], axis=1)
    order = np.argsort(base_distances)
    footprint_radius = float(
        np.max(np.linalg.norm(footprint_world - np.asarray(robot_center_xy)[None, :], axis=1))
    )
    cell_radius = float(cell_resolution) / math.sqrt(2.0)
    best = float("inf")
    best_index = -1
    for center_index in order:
        lower_bound = max(0.0, float(base_distances[center_index]) - footprint_radius - cell_radius)
        if lower_bound > best:
            break
        cell = occupancy_cell_polygon(centers[center_index], cell_resolution, grid_yaw)
        distance = convex_polygon_distance(footprint_world, cell)
        if distance < best:
            best = distance
            best_index = int(center_index)
            if best <= 0.0:
                break
    chosen = centers[best_index] if best_index >= 0 else np.array([np.nan, np.nan])
    return ClearanceEvaluation(
        clearance_area_m=float(best),
        clearance_center_m=float(np.min(center_distances)),
        base_to_center_m=float(np.min(base_distances)),
        closest_cell_center_x=float(chosen[0]),
        closest_cell_center_y=float(chosen[1]),
        occupied_cells=int(centers.shape[0]),
        threshold=int(threshold),
    )


def choose_occupancy_thresholds(observed_values: Iterable[int], nominal: int) -> List[int]:
    values = sorted({int(value) for value in observed_values if int(value) >= 0})
    positive = [value for value in values if value > 0]
    selected = {int(nominal)}
    if positive:
        selected.add(positive[0])
        selected.add(positive[len(positive) // 2])
        selected.add(positive[-1])
        below = [value for value in positive if value < nominal]
        above = [value for value in positive if value > nominal]
        if below:
            selected.add(below[-1])
        if above:
            selected.add(above[0])
    return sorted(value for value in selected if 0 <= value <= 100)


def nearest_footprint(
    footprints: Sequence[FootprintSample],
    target_stamp: float,
    domain: str,
    maximum_delta_s: float,
) -> Optional[FootprintSample]:
    available = [
        sample for sample in footprints
        if domain == "storage" or sample.header_stamp_sec is not None
    ]
    if not available:
        return None
    selected = min(available, key=lambda sample: abs(sample.stamp(domain) - target_stamp))
    if abs(selected.stamp(domain) - target_stamp) > maximum_delta_s:
        return None
    return selected


def _collect_named_values(node: Any, key_name: str, path: str = "") -> List[Tuple[str, Any]]:
    result: List[Tuple[str, Any]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key) == key_name:
                result.append((child, value))
            result.extend(_collect_named_values(value, key_name, child))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            result.extend(_collect_named_values(value, key_name, f"{path}[{index}]"))
    return result


def load_nav2_footprint(path: Path) -> Tuple[np.ndarray, str, Optional[float], List[str]]:
    document = load_yaml(path)
    warnings: List[str] = []
    parsed: List[Tuple[str, np.ndarray]] = []
    for key_path, value in _collect_named_values(document, "footprint"):
        try:
            parsed.append((key_path, parse_polygon(value)))
        except Exception:
            continue
    if not parsed:
        raise ValueError(f"No valid footprint found in {path}")
    first = parsed[0][1]
    differing = [
        key_path for key_path, vertices in parsed[1:]
        if vertices.shape != first.shape or not np.allclose(vertices, first, atol=1e-9)
    ]
    if differing:
        warnings.append("multiple non-identical footprint parameters found: " + ", ".join(differing))
    padding_values: List[float] = []
    for _key_path, value in _collect_named_values(document, "footprint_padding"):
        parsed_value = finite_float(value)
        if parsed_value is not None:
            padding_values.append(parsed_value)
    padding = padding_values[0] if padding_values else None
    if padding_values and any(not math.isclose(value, padding_values[0]) for value in padding_values[1:]):
        warnings.append(f"multiple footprint_padding values found: {padding_values}")
    source = "nav2_yaml:" + str(path) + ":" + ",".join(key for key, _vertices in parsed)
    return first, source, padding, warnings


def resolve_fallback_footprint(
    config: Mapping[str, Any],
    project_root: Path,
) -> Tuple[np.ndarray, str, Optional[float], List[str]]:
    clearance = config.get("clearance", {}) if isinstance(config.get("clearance", {}), Mapping) else {}
    warnings: List[str] = []
    if clearance.get("footprint_polygon") is not None:
        polygon = parse_polygon(clearance["footprint_polygon"])
        return polygon, "config:clearance.footprint_polygon", finite_float(clearance.get("footprint_padding_m")), warnings
    nav2_params = clearance.get("nav2_params") or config.get("nav2_params")
    if nav2_params:
        polygon, source, padding, yaml_warnings = load_nav2_footprint(resolve_path(str(nav2_params), project_root))
        warnings.extend(yaml_warnings)
        baseline_size = np.ptp(BASELINE_FALLBACK_FOOTPRINT, axis=0)
        yaml_size = np.ptp(polygon, axis=0)
        if not np.allclose(baseline_size, yaml_size, atol=1e-9):
            warnings.append(
                "YAML footprint size differs from corrected baseline fallback: "
                f"yaml={yaml_size[0]:.3f}x{yaml_size[1]:.3f} m, "
                f"baseline={baseline_size[0]:.3f}x{baseline_size[1]:.3f} m. "
                "Recorded published footprint remains preferred."
            )
        return polygon, source, padding, warnings
    return (
        BASELINE_FALLBACK_FOOTPRINT.copy(),
        "corrected_baseline_fallback:9.625x3.498m",
        finite_float(clearance.get("footprint_padding_m")),
        warnings,
    )


def deterministic_pose_perturbations(
    pose: OdomSample,
    sigma_multiplier: float,
) -> Tuple[List[Tuple[float, float, float, str]], str]:
    xy_covariance = pose_xy_covariance(pose.pose_covariance)
    yaw_sigma = yaw_sigma_from_covariance(pose.pose_covariance, 6)
    if xy_covariance is None and yaw_sigma is None:
        return [], "pose covariance unavailable, zero, non-finite, or invalid"
    perturbations: List[Tuple[float, float, float, str]] = [(pose.x, pose.y, pose.yaw, "nominal")]
    xy_offsets = [(0.0, 0.0, "xy_nominal")]
    if xy_covariance is not None:
        eigenvalues, eigenvectors = np.linalg.eigh(xy_covariance)
        first = eigenvectors[:, 0] * math.sqrt(max(0.0, float(eigenvalues[0]))) * float(sigma_multiplier)
        second = eigenvectors[:, 1] * math.sqrt(max(0.0, float(eigenvalues[1]))) * float(sigma_multiplier)
        xy_offsets = []
        for first_sign in (-1, 0, 1):
            for second_sign in (-1, 0, 1):
                vector = first_sign * first + second_sign * second
                xy_offsets.append(
                    (
                        float(vector[0]),
                        float(vector[1]),
                        f"xy_axis1_{first_sign:+d}_axis2_{second_sign:+d}",
                    )
                )
    yaw_offsets = [(0.0, "yaw_nominal")]
    if yaw_sigma is not None:
        value = float(yaw_sigma) * float(sigma_multiplier)
        yaw_offsets.extend([(value, "yaw_plus"), (-value, "yaw_minus")])
    perturbations = []
    for dx, dy, xy_label in xy_offsets:
        for dyaw, yaw_label in yaw_offsets:
            perturbations.append((pose.x + dx, pose.y + dy, wrap_angle(pose.yaw + dyaw), f"{xy_label}+{yaw_label}"))
    return perturbations, "deterministic covariance-axis envelope; covariance is not treated as measured error"


def setup_logger(log_path: Path, verbose: bool = False) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path.resolve()))
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: List[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        fieldnames = ordered
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_scalar(row.get(key, NA)) for key in fieldnames})


def serialize_scalar(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, float) and not math.isfinite(value):
        return NA
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        if isinstance(value, set):
            value = sorted(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")


def write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write parameters_used_v3.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(json.loads(json.dumps(document, default=json_default)), stream, sort_keys=False)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else NA
    except Exception:
        return NA


def environment_manifest() -> Dict[str, Any]:
    packages: Dict[str, str] = {}
    for name in ("numpy", "yaml", "matplotlib", "rosbag2_py", "rclpy"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "installed-version-not-exposed"))
        except Exception as exc:
            packages[name] = f"unavailable: {exc.__class__.__name__}"
    return {
        "schema_version": SCHEMA_VERSION,
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "ros_distro": os.environ.get("ROS_DISTRO", NA),
        "ros2_version": command_version(["ros2", "--help"]),
        "packages": packages,
    }


def run_pure_self_tests() -> None:
    footprint = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    centers = np.array([[4.0, 0.0]])
    evaluation = minimum_clearance(
        footprint_world=footprint,
        robot_center_xy=np.array([0.0, 0.0]),
        centers=centers,
        cell_resolution=2.0,
        grid_yaw=0.0,
        threshold=50,
    )
    assert math.isclose(evaluation.clearance_area_m, 2.0, abs_tol=1e-9)
    assert math.isclose(evaluation.clearance_center_m, 3.0, abs_tol=1e-9)
    assert math.isclose(evaluation.base_to_center_m, 4.0, abs_tol=1e-9)
    overlap = minimum_clearance(
        footprint_world=footprint,
        robot_center_xy=np.array([0.0, 0.0]),
        centers=np.array([[1.5, 0.0]]),
        cell_resolution=1.0,
        grid_yaw=0.0,
        threshold=50,
    )
    assert math.isclose(overlap.clearance_area_m, 0.0, abs_tol=1e-9)
    rotated = transform_polygon(footprint, 2.0, 3.0, math.pi / 2.0)
    assert np.allclose(np.mean(rotated, axis=0), [2.0, 3.0])
    assert np.allclose(np.ptp(BASELINE_FALLBACK_FOOTPRINT, axis=0), [9.625, 3.498])
    assert choose_occupancy_thresholds([-1, 0, 50, 100], 50) == [50, 100]
