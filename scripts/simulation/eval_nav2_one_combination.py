#!/usr/bin/env python3
"""
Evaluate one Nav2 experiment combination (one scenario + one planner/controller + one speed),
processing all run rosbags inside a directory and generating:

1) per_run_metrics.csv
2) aggregated_metrics.csv
3) primary_table_row.txt
4) secondary_table_row.txt
5) humanlikeness_table_row.txt

Methodology implemented in this version
---------------------------------------
Mission-completion metrics (main paper tables):
- success_geo
- progress_ratio (maximum path progress ratio reached during the run)
- mission_time_s

Plan-tracking metrics (main paper tables):
- rmse_y_primary_m
- p95_y_primary_m
- rmse_psi_primary_rad

Mission progress, success and the evaluation cutoff are computed against the
common benchmark route passed in --gt-csv. Lateral and heading tracking errors
are computed against the recorded Navigation2 plan active at each odometry
timestamp. Plan messages are synchronized with odometry using the MCAP storage
timestamp because some recorded Path messages have a zero header stamp.

Plan topic priority defaults to:
    /plan -> /received_global_plan -> /transformed_global_plan

If the selected plan and executed odometry use different frames, the evaluator
uses the direct time-synchronized transform recorded in /tf or /tf_static. The
executed pose is transformed into the plan frame before projection. Each pose
is then projected onto the polyline segments of the latest plan message
recorded at or before that pose. For the first and last segment, the tangent is
extended when the orthogonal projection lies outside the finite polyline. This
prevents the longitudinal gap introduced by Navigation2 plan pruning from being
misreported as lateral tracking error. This supports both static global plans
and progressively pruned /transformed_global_plan messages.

Success and validity rules
--------------------------
- Geometric success:
    success_geo = 1 if max(PR(t)) >= 0.90
- Run valid for mission-based continuous metrics:
    valid_for_mission = 1 if max(PR(t)) >= 0.70
- Run valid for plan-tracking metrics:
    valid_for_tracking = 1 if valid_for_mission = 1 and the fraction of
    evaluation samples with a same-frame active plan is at least 0.80

Mission time definition
-----------------------
- If success_geo = 1:
    mission_time = first time when PR(t) >= 0.90
- Else:
    mission_time = time when PR(t) reaches its maximum value

Continuous metrics are computed only up to that useful cutoff instant.

Human-likeness (secondary layer, raw GPS)
-----------------------------------------
- Based on raw /gps/fix from simulation bag compared to raw GT GPS mean.
- Uses a common origin from waypoint-file.
- Applies best simple variant correction (e.g., flip_y) and start alignment for shape comparison.
- Computed only for runs valid_for_mission == 1.

GT CSV in local XY
------------------
--gt-csv is the common benchmark route used only for mission progress, success
and cutoff timing. It is not used for plan-tracking RMSE. Human-likeness remains
a separate raw-GPS sensitivity/shape layer, using the common waypoint origin
and its own raw-GPS alignment procedure.

Smoothness and steering-command metrics
----------------------------------------
- Longitudinal jerk is the second time derivative of the measured longitudinal
  speed. An 11-sample centered moving average is applied to speed before the
  first derivative and to acceleration before the second derivative.
- Steering angle is reconstructed from the Navigation2 command using the
  Ackermann bicycle relation delta = atan(L_eq * omega_cmd / v_cmd). Samples
  below the configured minimum absolute command speed are excluded to avoid
  division near zero. Steering-rate RMS is computed from d(delta)/dt.
"""

import argparse
import csv
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception:
    print("ERROR: Could not import ROS 2 Python APIs.", file=sys.stderr)
    print("Run: source /opt/ros/<distro>/setup.bash", file=sys.stderr)
    raise


RUN_INDEX_RE = re.compile(r"(\d+)$")


def get_heading_window_m_for_scenario(scenario_id: int) -> float:
    return 150.0 if int(scenario_id) == 1 else 85.0


@dataclass
class RunMetrics:
    scenario_id: int
    speed_kmh: int
    planner_id: str
    controller_id: str
    run_id: int
    bag_name: str
    bag_dir: str

    primary_reference: str
    progress_reference: str
    plan_topic_used: str
    plan_frame_id: str
    executed_frame_id: str
    frame_transform_used: str
    executed_topic_used: str
    cmd_topic_used: str

    has_plan: int
    plan_message_count: int
    plan_tracking_samples: int
    plan_tracking_coverage: float
    plan_endpoint_extension_samples: int
    plan_endpoint_extension_fraction: float
    has_human_gt_gps: int

    success_geo: int
    progress_ratio: float
    valid_for_mission: int
    valid_for_tracking: int
    eval_cutoff_reason: str

    mission_time_s: float
    raw_run_duration_s: float

    final_dist_to_goal_m: float
    final_yaw_err_to_goal_rad: float

    rmse_y_primary_m: Optional[float]
    p95_y_primary_m: Optional[float]
    rmse_psi_primary_rad: Optional[float]
    rmse_y_finite_segment_diagnostic_m: Optional[float]
    p95_y_finite_segment_diagnostic_m: Optional[float]

    rmse_v_mps: Optional[float]
    rms_jx_mps3: Optional[float]
    rms_dotdelta_radps: Optional[float]
    rms_cmd_w_radps: Optional[float]

    human_gt_reference: str
    human_gt_alignment_mode: str
    human_gt_best_variant: str
    human_gt_rmse_y_m: Optional[float]
    human_gt_p95_y_m: Optional[float]
    human_gt_max_y_m: Optional[float]
    human_gt_progress: Optional[float]
    human_gt_pointwise_rmse_m: Optional[float]

    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Nav2 combination across all run rosbags.")

    parser.add_argument("--runs-dir", required=True, help="Directory containing multiple run bag folders")
    parser.add_argument("--scenario-id", type=int, required=True)
    parser.add_argument("--speed-kmh", type=int, required=True)
    parser.add_argument("--planner-id", required=True)
    parser.add_argument("--controller-id", required=True)

    parser.add_argument(
        "--gt-csv",
        required=True,
        help="Common benchmark CSV in local XY used for mission progress, success and cutoff timing.",
    )
    parser.add_argument("--out-dir", default="", help="Output directory (default: <runs-dir>/Output_metrics_assessment)")

    parser.add_argument(
        "--gt-gps-csv-glob",
        default="",
        help="Glob for GT raw GPS CSVs (e.g. .../per_bag/.../gps_fix_latlon.csv)"
    )
    parser.add_argument(
        "--waypoint-file",
        default="",
        help="Waypoint YAML/CSV used to define common geodetic origin for human-likeness metrics"
    )
    parser.add_argument("--gps-topic", default="/gps/fix", help="GPS topic used for human-likeness metrics")
    parser.add_argument("--human-resample-samples", type=int, default=300)

    parser.add_argument("--executed-topic", default="/odometry/global")
    parser.add_argument(
        "--plan-topics",
        default="/plan,/received_global_plan,/transformed_global_plan,/plan_smoothed",
        help="Comma-separated priority order for time-synchronized Navigation2 plan tracking"
    )
    parser.add_argument("--cmd-topics", default="/cmd_vel_nav,/cmd_vel")

    parser.add_argument(
        "--wheelbase-m",
        type=float,
        default=6.804,
        help="Equivalent Ackermann wheelbase used to reconstruct steering angle from cmd_vel (default: 6.804 m)",
    )
    parser.add_argument(
        "--steering-min-speed-mps",
        type=float,
        default=0.50,
        help="Minimum absolute commanded speed used for steering reconstruction (default: 0.50 m/s)",
    )

    parser.add_argument("--success-pr-threshold", type=float, default=0.90,
                        help="Run is geometric success if max PR >= this threshold")
    parser.add_argument("--tracking-pr-threshold", type=float, default=0.70,
                        help="Run is valid for continuous metrics if max PR >= this threshold")
    parser.add_argument(
        "--min-plan-coverage",
        type=float,
        default=0.80,
        help="Minimum fraction of evaluation odometry samples with an active same-frame plan (default: 0.80)",
    )

    return parser.parse_args()


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle_rad(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def cumulative_arc_length(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.array([], dtype=float)
    if xy.shape[0] == 1:
        return np.array([0.0], dtype=float)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def path_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def heading_from_path(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] < 2:
        return np.zeros((xy.shape[0],), dtype=float)
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    return np.arctan2(dy, dx)


def remove_consecutive_duplicates_xy(
    xy: np.ndarray,
    t: Optional[np.ndarray] = None,
    yaws: Optional[np.ndarray] = None,
    speeds: Optional[np.ndarray] = None,
    eps: float = 1e-9
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if xy.shape[0] <= 1:
        return xy, t, yaws, speeds

    keep = [0]
    for i in range(1, xy.shape[0]):
        if np.linalg.norm(xy[i] - xy[i - 1]) > eps:
            keep.append(i)

    k = np.asarray(keep, dtype=int)
    t_out = t[k] if t is not None else None
    y_out = yaws[k] if yaws is not None else None
    s_out = speeds[k] if speeds is not None else None
    return xy[k], t_out, y_out, s_out


def median_iqr(
    values: List[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return median, Q1 and Q3 for finite values.

    The historical function name is retained for compatibility with the rest
    of the script, but the returned interval is explicit rather than an IQR
    width. This prevents a value such as ``median [Q3-Q1]`` from being mistaken
    for ``median [Q1, Q3]`` in manuscript tables.
    """
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None, None
    arr = np.asarray(vals, dtype=float)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return med, q1, q3


def fmt_median_iqr(values: List[float], decimals: int = 2) -> str:
    med, q1, q3 = median_iqr(values)
    if med is None:
        return "--"
    return f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"


def rms(arr: np.ndarray) -> Optional[float]:
    if arr is None or arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.sqrt(np.mean(finite ** 2)))


def moving_average_1d(x: np.ndarray, window: int = 11) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    if window <= 1:
        return x.copy()
    w = min(window, x.size)
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return x.copy()
    kernel = np.ones((w,), dtype=float) / float(w)
    pad = w // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")


def moving_average_with_time(
    t_s: np.ndarray,
    x: np.ndarray,
    window: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a centered moving average with unsupported edge samples removed."""
    if t_s.size == 0 or x.size == 0 or t_s.size != x.size:
        return np.array([], dtype=float), np.array([], dtype=float)
    if window <= 1:
        return t_s.copy(), x.copy()

    w = min(int(window), int(x.size))
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return t_s.copy(), x.copy()

    kernel = np.ones((w,), dtype=float) / float(w)
    x_smooth = np.convolve(x, kernel, mode="valid")
    pad = w // 2
    t_smooth = t_s[pad:t_s.size - pad]
    return t_smooth, x_smooth


def derivative_with_time(
    t_s: np.ndarray,
    x: np.ndarray,
    min_dt_s: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    if t_s.size < 2 or x.size < 2 or t_s.size != x.size:
        return np.array([], dtype=float), np.array([], dtype=float)

    dt = np.diff(t_s)
    dx = np.diff(x)
    valid = (
        (dt > min_dt_s)
        & np.isfinite(dt)
        & np.isfinite(x[:-1])
        & np.isfinite(x[1:])
    )
    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float)

    t_mid = 0.5 * (t_s[:-1] + t_s[1:])
    return t_mid[valid], dx[valid] / dt[valid]


def derivative(t_s: np.ndarray, x: np.ndarray, min_dt_s: float = 1e-3) -> np.ndarray:
    """Return the finite-difference derivative, retaining the legacy API."""
    _, dx_dt = derivative_with_time(t_s, x, min_dt_s=min_dt_s)
    return dx_dt


def resolve_bag_uri_for_mcap(bag_dir: Path) -> Path:
    mcap_files = sorted([p for p in bag_dir.glob("*.mcap") if p.is_file()])
    if not mcap_files:
        return bag_dir
    for p in mcap_files:
        if p.name.endswith("_0.mcap"):
            return p
    return mcap_files[0]


def list_bag_dirs(runs_dir: Path) -> List[Path]:
    bag_dirs: List[Path] = []
    for meta in sorted(runs_dir.rglob("metadata.yaml")):
        bag_dirs.append(meta.parent)

    unique: List[Path] = []
    seen = set()
    for p in bag_dirs:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            unique.append(p)
    return unique


def parse_run_id_from_name(name: str) -> int:
    m = RUN_INDEX_RE.search(name)
    if m:
        return int(m.group(1))
    return 0


def extract_path_xy_from_msg(msg: Any) -> Optional[np.ndarray]:
    if not hasattr(msg, "poses"):
        return None
    pts = []
    try:
        for ps in msg.poses:
            pose = ps.pose if hasattr(ps, "pose") else ps
            p = pose.position
            pts.append((float(p.x), float(p.y)))
    except Exception:
        return None
    if len(pts) < 2:
        return None
    xy = np.asarray(pts, dtype=float)
    xy, _, _, _ = remove_consecutive_duplicates_xy(xy)
    if xy.shape[0] < 2:
        return None
    return xy


def extract_frame_id(msg: Any) -> str:
    try:
        return str(msg.header.frame_id).strip()
    except Exception:
        return ""


def extract_header_stamp_ns(msg: Any) -> int:
    try:
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except Exception:
        return 0


def extract_transform_2d(transform: Any, storage_t_ns: int, is_static: bool) -> Optional[Dict[str, Any]]:
    try:
        parent = str(transform.header.frame_id).strip()
        child = str(transform.child_frame_id).strip()
        if not parent or not child:
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = yaw_from_quaternion(
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
        return {
            "parent": parent,
            "child": child,
            "t_ns": int(storage_t_ns),
            "tx": float(translation.x),
            "ty": float(translation.y),
            "yaw": float(yaw),
            "is_static": bool(is_static),
        }
    except Exception:
        return None


def invert_transform_2d(tx: float, ty: float, yaw: float) -> Tuple[float, float, float]:
    """Invert p_parent = R(yaw) * p_child + translation."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    inv_tx = -(c * tx + s * ty)
    inv_ty = -(-s * tx + c * ty)
    return float(inv_tx), float(inv_ty), float(-yaw)


def build_tf_history(raw_transforms: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in raw_transforms:
        key = (str(item["parent"]), str(item["child"]))
        grouped.setdefault(key, []).append(item)

    history: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, items in grouped.items():
        dynamic_items = [item for item in items if not bool(item["is_static"])]
        selected = dynamic_items if dynamic_items else items
        selected.sort(key=lambda item: int(item["t_ns"]))

        history[key] = {
            "t_ns": np.asarray([int(item["t_ns"]) for item in selected], dtype=np.int64),
            "tx": np.asarray([float(item["tx"]) for item in selected], dtype=float),
            "ty": np.asarray([float(item["ty"]) for item in selected], dtype=float),
            "yaw": np.asarray([float(item["yaw"]) for item in selected], dtype=float),
            "is_static": not bool(dynamic_items),
            "count": len(selected),
        }
    return history


def direct_frames_transformable(
    source_frame: str,
    target_frame: str,
    tf_history: Dict[Tuple[str, str], Dict[str, Any]],
) -> bool:
    if source_frame == target_frame and source_frame:
        return True
    if not source_frame or not target_frame:
        return False
    return (
        (target_frame, source_frame) in tf_history
        or (source_frame, target_frame) in tf_history
    )


def lookup_direct_transform_2d(
    source_frame: str,
    target_frame: str,
    query_t_ns: int,
    tf_history: Dict[Tuple[str, str], Dict[str, Any]],
) -> Optional[Tuple[float, float, float, str]]:
    """Return T_target_source at the latest MCAP time not after query_t_ns."""
    if source_frame == target_frame and source_frame:
        return 0.0, 0.0, 0.0, "identity"

    direct_key = (target_frame, source_frame)
    reverse_key = (source_frame, target_frame)

    invert = False
    if direct_key in tf_history:
        data = tf_history[direct_key]
        parent, child = direct_key
    elif reverse_key in tf_history:
        data = tf_history[reverse_key]
        parent, child = reverse_key
        invert = True
    else:
        return None

    times = np.asarray(data["t_ns"], dtype=np.int64)
    if times.size == 0:
        return None

    if bool(data.get("is_static", False)):
        index = int(times.size - 1)
    else:
        index = int(np.searchsorted(times, int(query_t_ns), side="right") - 1)
        if index < 0:
            return None

    tx = float(data["tx"][index])
    ty = float(data["ty"][index])
    yaw = float(data["yaw"][index])

    if invert:
        tx, ty, yaw = invert_transform_2d(tx, ty, yaw)
        description = f"inverse({parent}->{child})"
    else:
        description = f"{parent}->{child}"

    return tx, ty, yaw, description


def extract_odom_sample(msg: Any) -> Optional[Tuple[float, float, float, Optional[float]]]:
    try:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        x = float(p.x)
        y = float(p.y)
        yaw = yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w))

        speed = None
        if hasattr(msg, "twist") and hasattr(msg.twist, "twist"):
            if hasattr(msg.twist.twist, "linear"):
                speed = float(msg.twist.twist.linear.x)

        return x, y, yaw, speed
    except Exception:
        return None


def extract_cmd_twist(msg: Any) -> Optional[Tuple[float, float]]:
    try:
        if hasattr(msg, "linear") and hasattr(msg, "angular"):
            return float(msg.linear.x), float(msg.angular.z)
        if hasattr(msg, "twist") and hasattr(msg.twist, "linear") and hasattr(msg.twist, "angular"):
            return float(msg.twist.linear.x), float(msg.twist.angular.z)
    except Exception:
        return None
    return None


def choose_first_valid_plan(plan_candidates: List[str], valid_plan_data: Dict[str, Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    for topic in plan_candidates:
        if topic in valid_plan_data:
            return topic, valid_plan_data[topic]
    return "", None


def read_bag_data(
    bag_dir: Path,
    executed_topic: str,
    plan_topic_candidates: List[str],
    cmd_topic_candidates: List[str]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "executed_topic_used": executed_topic,
        "executed_frame_id": "",
        "plan_topic_used": "",
        "plan_frame_id": "",
        "frame_transform_used": "",
        "cmd_topic_used": "",
        "exec": None,
        "plan": None,
        "cmd": None,
        "topic_type_map": {},
        "notes": [],
        "all_valid_plans": {},
        "tf_history": {},
    }

    reader = rosbag2_py.SequentialReader()
    bag_uri = resolve_bag_uri_for_mcap(bag_dir)

    storage_options = rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topic_type_map: Dict[str, str] = {}
    for t in reader.get_all_topics_and_types():
        topic_type_map[t.name] = t.type
    out["topic_type_map"] = topic_type_map

    if executed_topic not in topic_type_map:
        raise RuntimeError(f"Executed topic not found in bag: {executed_topic}")

    existing_plan_topics = [t for t in plan_topic_candidates if t in topic_type_map]
    existing_cmd_topics = [t for t in cmd_topic_candidates if t in topic_type_map]
    existing_tf_topics = [t for t in ["/tf", "/tf_static"] if t in topic_type_map]

    if existing_cmd_topics:
        out["cmd_topic_used"] = existing_cmd_topics[0]

    topics_needed = [executed_topic] + existing_plan_topics + existing_cmd_topics + existing_tf_topics

    try:
        if hasattr(rosbag2_py, "StorageFilter"):
            reader.set_filter(rosbag2_py.StorageFilter(topics=topics_needed))
    except Exception as e:
        out["notes"].append(f"StorageFilter not applied: {e}")

    msg_type_cache: Dict[str, Any] = {}
    for topic in topics_needed:
        msg_type_cache[topic] = get_message(topic_type_map[topic])

    exec_t_ns: List[int] = []
    exec_x: List[float] = []
    exec_y: List[float] = []
    exec_yaw: List[float] = []
    exec_v: List[float] = []
    exec_frame_ids: List[str] = []

    cmd_t_ns: List[int] = []
    cmd_vx: List[float] = []
    cmd_wz: List[float] = []

    raw_transforms: List[Dict[str, Any]] = []

    plan_history_per_topic: Dict[str, List[Dict[str, Any]]] = {
        topic: [] for topic in existing_plan_topics
    }

    chosen_cmd_topic = existing_cmd_topics[0] if existing_cmd_topics else ""

    while reader.has_next():
        topic, rawdata, t_ns = reader.read_next()
        if topic not in msg_type_cache:
            continue

        try:
            msg = deserialize_message(rawdata, msg_type_cache[topic])
        except Exception as e:
            out["notes"].append(f"Deserialize failed on {topic}: {e}")
            continue

        if topic == executed_topic:
            sample = extract_odom_sample(msg)
            if sample is None:
                continue
            x, y, yaw, speed = sample
            exec_t_ns.append(int(t_ns))
            exec_x.append(x)
            exec_y.append(y)
            exec_yaw.append(yaw)
            exec_v.append(float(speed) if speed is not None and math.isfinite(speed) else float("nan"))
            frame_id = extract_frame_id(msg)
            if frame_id:
                exec_frame_ids.append(frame_id)

        elif topic in plan_history_per_topic:
            xy = extract_path_xy_from_msg(msg)
            if xy is None:
                continue
            plan_history_per_topic[topic].append(
                {
                    "t_ns": int(t_ns),
                    "header_t_ns": extract_header_stamp_ns(msg),
                    "frame_id": extract_frame_id(msg),
                    "xy": xy,
                }
            )

        elif topic == chosen_cmd_topic:
            tw = extract_cmd_twist(msg)
            if tw is None:
                continue
            vx, wz = tw
            cmd_t_ns.append(int(t_ns))
            cmd_vx.append(vx)
            cmd_wz.append(wz)

        elif topic in existing_tf_topics:
            if not hasattr(msg, "transforms"):
                continue
            for transform in msg.transforms:
                item = extract_transform_2d(
                    transform=transform,
                    storage_t_ns=int(t_ns),
                    is_static=(topic == "/tf_static"),
                )
                if item is not None:
                    raw_transforms.append(item)

    if len(exec_t_ns) < 2:
        raise RuntimeError("Not enough executed odometry samples")

    exec_t_ns_arr = np.asarray(exec_t_ns, dtype=np.int64)
    exec_x_arr = np.asarray(exec_x, dtype=float)
    exec_y_arr = np.asarray(exec_y, dtype=float)
    exec_yaw_arr = np.asarray(exec_yaw, dtype=float)
    exec_v_arr = np.asarray(exec_v, dtype=float)

    idx = np.argsort(exec_t_ns_arr, kind="stable")
    exec_t_ns_arr = exec_t_ns_arr[idx]
    exec_x_arr = exec_x_arr[idx]
    exec_y_arr = exec_y_arr[idx]
    exec_yaw_arr = exec_yaw_arr[idx]
    exec_v_arr = exec_v_arr[idx]

    exec_xy = np.column_stack((exec_x_arr, exec_y_arr))
    exec_xy, exec_t_ns_arr, exec_yaw_arr, exec_v_arr = remove_consecutive_duplicates_xy(
        exec_xy,
        t=exec_t_ns_arr,
        yaws=exec_yaw_arr,
        speeds=exec_v_arr
    )

    executed_frame_id = Counter(exec_frame_ids).most_common(1)[0][0] if exec_frame_ids else ""
    out["executed_frame_id"] = executed_frame_id
    if not executed_frame_id:
        out["notes"].append("EXECUTED_FRAME_ID_EMPTY")

    if np.any(~np.isfinite(exec_v_arr)):
        t_s = (exec_t_ns_arr - exec_t_ns_arr[0]).astype(float) / 1e9
        ds = np.linalg.norm(np.diff(exec_xy, axis=0), axis=1) if exec_xy.shape[0] >= 2 else np.array([], dtype=float)
        dt = np.diff(t_s) if t_s.size >= 2 else np.array([], dtype=float)
        est = np.full((exec_xy.shape[0],), np.nan, dtype=float)

        if ds.size and dt.size:
            valid = dt > 1e-9
            if np.any(valid):
                v_seg = np.zeros_like(ds)
                v_seg[valid] = ds[valid] / dt[valid]
                est[1:] = v_seg
                est[0] = est[1] if est.size > 1 and math.isfinite(est[1]) else 0.0

        fill_mask = ~np.isfinite(exec_v_arr)
        exec_v_arr[fill_mask] = est[fill_mask]
        exec_v_arr[~np.isfinite(exec_v_arr)] = 0.0

    out["exec"] = {
        "t_ns": exec_t_ns_arr,
        "t_s": (exec_t_ns_arr - exec_t_ns_arr[0]).astype(float) / 1e9,
        "xy": exec_xy,
        "yaw": exec_yaw_arr,
        "v": exec_v_arr,
        "length_m": path_length(exec_xy),
        "frame_id": executed_frame_id,
    }

    tf_history = build_tf_history(raw_transforms)
    out["tf_history"] = tf_history

    valid_plan_data: Dict[str, Dict[str, Any]] = {}
    for topic, raw_events in plan_history_per_topic.items():
        if not raw_events:
            continue

        raw_events.sort(key=lambda event: int(event["t_ns"]))
        nonempty_frames = [str(event["frame_id"]) for event in raw_events if str(event["frame_id"])]
        topic_frame_id = Counter(nonempty_frames).most_common(1)[0][0] if nonempty_frames else ""

        compatible_events: List[Dict[str, Any]] = []
        incompatible_count = 0
        for event in raw_events:
            event_frame_id = str(event["frame_id"])
            if not event_frame_id or event_frame_id != topic_frame_id:
                incompatible_count += 1
                continue
            if not direct_frames_transformable(
                source_frame=executed_frame_id,
                target_frame=event_frame_id,
                tf_history=tf_history,
            ):
                incompatible_count += 1
                continue

            xy = event["xy"]
            if xy is None or xy.shape[0] < 2:
                continue

            headings = heading_from_path(xy)
            compatible_events.append(
                {
                    "t_ns": int(event["t_ns"]),
                    "header_t_ns": int(event["header_t_ns"]),
                    "frame_id": event_frame_id,
                    "xy": xy,
                    "heading": headings,
                    "s": cumulative_arc_length(xy),
                    "length_m": path_length(xy),
                    "points": int(xy.shape[0]),
                }
            )

        if incompatible_count:
            out["notes"].append(
                f"PLAN_FRAME_UNTRANSFORMABLE_DROPPED topic={topic} count={incompatible_count} "
                f"exec_frame={executed_frame_id or '<empty>'} plan_frame={topic_frame_id or '<empty>'}"
            )

        if not compatible_events:
            continue

        compatible_frames = [
            str(event["frame_id"]) for event in compatible_events if str(event["frame_id"])
        ]
        compatible_frame_id = (
            Counter(compatible_frames).most_common(1)[0][0]
            if compatible_frames else ""
        )
        plan_t_ns = np.asarray([event["t_ns"] for event in compatible_events], dtype=np.int64)
        zero_header_stamps = sum(int(event["header_t_ns"] == 0) for event in compatible_events)
        valid_plan_data[topic] = {
            "messages": compatible_events,
            "t_ns": plan_t_ns,
            "frame_id": compatible_frame_id,
            "message_count": len(compatible_events),
            "zero_header_stamp_count": zero_header_stamps,
        }

    out["all_valid_plans"] = valid_plan_data

    chosen_plan_topic, chosen_plan = choose_first_valid_plan(plan_topic_candidates, valid_plan_data)
    out["plan_topic_used"] = chosen_plan_topic
    out["plan"] = chosen_plan
    if chosen_plan is not None:
        out["plan_frame_id"] = str(chosen_plan.get("frame_id", ""))
        if executed_frame_id == out["plan_frame_id"]:
            out["frame_transform_used"] = "identity"
        elif (out["plan_frame_id"], executed_frame_id) in tf_history:
            out["frame_transform_used"] = f"{out['plan_frame_id']}->{executed_frame_id}"
        elif (executed_frame_id, out["plan_frame_id"]) in tf_history:
            out["frame_transform_used"] = f"inverse({executed_frame_id}->{out['plan_frame_id']})"
        out["notes"].append(
            f"PLAN_TIME_SOURCE=MCAP_STORAGE_TIMESTAMP; PLAN_TOPIC={chosen_plan_topic}; "
            f"FRAME_TRANSFORM={out['frame_transform_used'] or 'unavailable'}; "
            f"PLAN_MESSAGES={chosen_plan['message_count']}; "
            f"PLAN_ZERO_HEADER_STAMPS={chosen_plan['zero_header_stamp_count']}"
        )

    if len(cmd_t_ns) >= 2:
        cmd_t_ns_arr = np.asarray(cmd_t_ns, dtype=np.int64)
        cmd_vx_arr = np.asarray(cmd_vx, dtype=float)
        cmd_wz_arr = np.asarray(cmd_wz, dtype=float)

        idx = np.argsort(cmd_t_ns_arr, kind="stable")
        cmd_t_ns_arr = cmd_t_ns_arr[idx]
        cmd_vx_arr = cmd_vx_arr[idx]
        cmd_wz_arr = cmd_wz_arr[idx]

        out["cmd"] = {
            "t_ns": cmd_t_ns_arr,
            "t_s": (cmd_t_ns_arr - exec_t_ns_arr[0]).astype(float) / 1e9,
            "vx": cmd_vx_arr,
            "wz": cmd_wz_arr,
        }

    if not chosen_plan_topic:
        out["notes"].append("NO_VALID_PLAN_TOPIC_FOUND")

    return out



def load_gt_reference_csv(gt_csv: Path) -> Dict[str, Any]:
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}")

    rows = []
    with gt_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError(f"GT CSV is empty: {gt_csv}")

    required = ["x_m", "y_m", "t_ns"]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise RuntimeError(f"GT CSV missing required columns {missing}. Found: {list(rows[0].keys())}")

    x = np.asarray([float(r["x_m"]) for r in rows], dtype=float)
    y = np.asarray([float(r["y_m"]) for r in rows], dtype=float)
    t_ns = np.asarray([int(float(r["t_ns"])) for r in rows], dtype=np.int64)

    xy = np.column_stack((x, y))
    xy, t_ns, _, _ = remove_consecutive_duplicates_xy(xy, t=t_ns)

    if xy.shape[0] < 2:
        raise RuntimeError(f"GT CSV has fewer than 2 unique XY points after cleanup: {gt_csv}")

    headings = heading_from_path(xy)
    return {
        "xy": xy,
        "t_ns": t_ns,
        "heading": headings,
        "path_length_m": path_length(xy),
        "goal_xy": xy[-1].copy(),
        "goal_yaw": float(headings[-1]) if headings.size else 0.0,
        "points": int(xy.shape[0]),
        "source_csv": str(gt_csv),
        "source_name": gt_csv.name,
    }



def rotate_xy(xy: np.ndarray, theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    R = np.asarray([[c, -s], [s, c]], dtype=float)
    return xy @ R.T


def estimate_path_heading_window(xy: np.ndarray, window_m: float = 10.0, from_end: bool = False) -> float:
    if xy.shape[0] < 2:
        return 0.0
    s = cumulative_arc_length(xy)
    total = float(s[-1]) if s.size else 0.0
    if total <= 1e-9:
        return float(math.atan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0]))
    w = max(1.0, min(float(window_m), 0.25 * total))
    if from_end:
        target_s = max(0.0, total - w)
        i0 = int(np.searchsorted(s, target_s, side='left'))
        i1 = xy.shape[0] - 1
    else:
        i0 = 0
        i1 = int(np.searchsorted(s, w, side='left'))
        i1 = min(max(i1, 1), xy.shape[0] - 1)
    dx = float(xy[i1, 0] - xy[i0, 0])
    dy = float(xy[i1, 1] - xy[i0, 1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return float(math.atan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0]))
    return float(math.atan2(dy, dx))


def align_reference_to_execution_start_heading(
    ref_xy: np.ndarray,
    exec_xy: np.ndarray,
    heading_window_m: float = 10.0,
) -> Dict[str, Any]:
    if ref_xy.shape[0] < 2 or exec_xy.shape[0] < 2:
        return {
            'xy': ref_xy.copy(),
            'heading': heading_from_path(ref_xy),
            'goal_xy': ref_xy[-1].copy(),
            'goal_yaw': float(heading_from_path(ref_xy)[-1]) if ref_xy.shape[0] >= 2 else 0.0,
            'rotation_rad': 0.0,
            'translation_xy': np.zeros((2,), dtype=float),
            'mode': 'identity',
        }

    ref_h0 = estimate_path_heading_window(ref_xy, window_m=heading_window_m, from_end=False)
    exec_h0 = estimate_path_heading_window(exec_xy, window_m=heading_window_m, from_end=False)
    theta = float(wrap_angle_rad(np.asarray([exec_h0 - ref_h0], dtype=float))[0])

    ref0 = ref_xy[0].copy()
    exec0 = exec_xy[0].copy()
    rotated = rotate_xy(ref_xy - ref0, theta) + exec0
    headings = heading_from_path(rotated)
    return {
        'xy': rotated,
        'heading': headings,
        'goal_xy': rotated[-1].copy(),
        'goal_yaw': float(headings[-1]) if headings.size else 0.0,
        'rotation_rad': theta,
        'translation_xy': exec0 - rotate_xy(ref0.reshape(1,2), theta).reshape(2,),
        'mode': 'start_heading_aligned',
    }

def project_points_to_polyline(
    points_xy: np.ndarray,
    ref_xy: np.ndarray,
    extend_endpoint_tangents: bool = False,
) -> Dict[str, np.ndarray]:
    """Project points onto a reference polyline.

    The conventional finite-segment projection is used for mission progress.
    For active-plan cross-track error, ``extend_endpoint_tangents=True`` lets a
    point before the first segment or after the last segment project onto that
    endpoint segment's supporting line. This removes the along-track gap caused
    by Navigation2 plan pruning without extending interior segments.
    """
    n_pts = points_xy.shape[0]
    n_ref = ref_xy.shape[0]

    if n_pts == 0 or n_ref < 2:
        return {
            "dist": np.array([], dtype=float),
            "s_hat": np.array([], dtype=float),
            "heading_ref": np.array([], dtype=float),
            "endpoint_extrapolated": np.array([], dtype=bool),
            "finite_segment_dist": np.array([], dtype=float),
        }

    ref_seg = ref_xy[1:] - ref_xy[:-1]
    ref_seg_len2 = np.sum(ref_seg * ref_seg, axis=1)
    ref_seg_len = np.sqrt(np.maximum(ref_seg_len2, 1e-12))
    ref_heading = np.arctan2(ref_seg[:, 1], ref_seg[:, 0])
    s_ref = cumulative_arc_length(ref_xy)

    dist_out = np.zeros((n_pts,), dtype=float)
    s_out = np.zeros((n_pts,), dtype=float)
    heading_out = np.zeros((n_pts,), dtype=float)
    endpoint_extrapolated_out = np.zeros((n_pts,), dtype=bool)
    finite_segment_dist_out = np.zeros((n_pts,), dtype=float)

    for i in range(n_pts):
        p = points_xy[i]
        best_dist2 = float("inf")
        best_s = 0.0
        best_heading = 0.0
        best_endpoint_extrapolated = False

        for j in range(n_ref - 1):
            a = ref_xy[j]
            ab = ref_seg[j]
            ab_len2 = ref_seg_len2[j]

            if ab_len2 <= 1e-12:
                tau = 0.0
                q = a
            else:
                tau = float(np.dot(p - a, ab) / ab_len2)
                tau = max(0.0, min(1.0, tau))
                q = a + tau * ab

            d2 = float(np.sum((p - q) ** 2))
            if d2 < best_dist2:
                best_dist2 = d2
                best_s = float(s_ref[j] + tau * ref_seg_len[j])
                best_heading = float(ref_heading[j])
                best_endpoint_extrapolated = False

        finite_segment_dist_out[i] = math.sqrt(best_dist2)

        if extend_endpoint_tangents:
            endpoint_candidates = ((0, "before"), (n_ref - 2, "after"))
            for segment_index, side in endpoint_candidates:
                a = ref_xy[segment_index]
                ab = ref_seg[segment_index]
                ab_len2 = ref_seg_len2[segment_index]
                if ab_len2 <= 1e-12:
                    continue
                tau_raw = float(np.dot(p - a, ab) / ab_len2)
                is_valid_extension = (
                    (side == "before" and tau_raw < 0.0)
                    or (side == "after" and tau_raw > 1.0)
                )
                if not is_valid_extension:
                    continue
                q = a + tau_raw * ab
                d2 = float(np.sum((p - q) ** 2))
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_s = float(
                        s_ref[segment_index]
                        + tau_raw * ref_seg_len[segment_index]
                    )
                    best_heading = float(ref_heading[segment_index])
                    best_endpoint_extrapolated = True

        dist_out[i] = math.sqrt(best_dist2)
        s_out[i] = best_s
        heading_out[i] = best_heading
        endpoint_extrapolated_out[i] = best_endpoint_extrapolated

    return {
        "dist": dist_out,
        "s_hat": s_out,
        "heading_ref": heading_out,
        "endpoint_extrapolated": endpoint_extrapolated_out,
        "finite_segment_dist": finite_segment_dist_out,
    }


def compute_tracking_against_reference(
    exec_t_s: np.ndarray,
    exec_xy: np.ndarray,
    exec_yaw: np.ndarray,
    ref_xy: np.ndarray,
    ref_goal_xy: np.ndarray,
    ref_goal_yaw: float,
    success_pr_threshold: float,
    tracking_pr_threshold: float
) -> Dict[str, Any]:
    ref_s = cumulative_arc_length(ref_xy)
    ref_length = float(ref_s[-1]) if ref_s.size else 0.0

    proj = project_points_to_polyline(exec_xy, ref_xy)
    dist = proj["dist"]
    s_hat = proj["s_hat"]
    heading_ref = proj["heading_ref"]

    if ref_length > 1e-9 and s_hat.size:
        pr_curve = s_hat / ref_length
    else:
        pr_curve = np.zeros((exec_xy.shape[0],), dtype=float)

    if pr_curve.size:
        max_idx = int(np.argmax(pr_curve))
        max_pr = float(pr_curve[max_idx])
    else:
        max_idx = 0
        max_pr = 0.0

    success_mask = pr_curve >= success_pr_threshold
    success_geo = int(np.any(success_mask))
    valid_for_tracking = int(max_pr >= tracking_pr_threshold)

    if success_geo:
        eval_end_idx = int(np.flatnonzero(success_mask)[0])
        eval_cutoff_reason = f"first_PR_ge_{success_pr_threshold:.2f}"
    else:
        eval_end_idx = max_idx
        eval_cutoff_reason = "max_PR_valid_no_success" if valid_for_tracking else "max_PR_below_tracking_threshold"

    mission_time_s = float(exec_t_s[eval_end_idx] - exec_t_s[0]) if exec_t_s.size else 0.0
    raw_run_duration_s = float(exec_t_s[-1] - exec_t_s[0]) if exec_t_s.size else 0.0

    dist_to_goal_at_eval = float(np.linalg.norm(exec_xy[eval_end_idx] - ref_goal_xy))
    yaw_err_to_goal_at_eval = float(abs(wrap_angle_rad(np.asarray([exec_yaw[eval_end_idx] - ref_goal_yaw]))[0]))

    dist_eval = dist[:eval_end_idx + 1]
    heading_ref_eval = heading_ref[:eval_end_idx + 1]
    yaw_eval = exec_yaw[:eval_end_idx + 1]

    rmse_y = float(np.sqrt(np.mean(dist_eval ** 2))) if dist_eval.size else None
    p95_y = float(np.percentile(dist_eval, 95)) if dist_eval.size else None

    if yaw_eval.size:
        yaw_err = np.abs(wrap_angle_rad(yaw_eval - heading_ref_eval))
        rmse_psi = float(np.sqrt(np.mean(yaw_err ** 2)))
    else:
        rmse_psi = None

    return {
        "progress_ratio": max_pr,
        "success_geo": success_geo,
        "valid_for_tracking": valid_for_tracking,
        "eval_cutoff_reason": eval_cutoff_reason,
        "mission_time_s": mission_time_s,
        "raw_run_duration_s": raw_run_duration_s,
        "final_dist_to_goal_m": dist_to_goal_at_eval,
        "final_yaw_err_to_goal_rad": yaw_err_to_goal_at_eval,
        "rmse_y_m": rmse_y,
        "p95_y_m": p95_y,
        "rmse_psi_rad": rmse_psi,
        "eval_end_idx": eval_end_idx,
        "max_pr": max_pr,
    }


def compute_tracking_against_time_varying_plan(
    exec_t_ns: np.ndarray,
    exec_xy: np.ndarray,
    exec_yaw: np.ndarray,
    executed_frame_id: str,
    plan_frame_id: str,
    tf_history: Dict[Tuple[str, str], Dict[str, Any]],
    plan_history: Optional[Dict[str, Any]],
    eval_end_idx: int,
    min_plan_coverage: float,
) -> Dict[str, Any]:
    """Compute executed-to-plan error using the latest plan available in MCAP time.

    The Path header timestamp is deliberately ignored. For every executed pose,
    the latest Path message whose MCAP storage timestamp is less than or equal
    to the odometry storage timestamp is selected. Distance and reference
    heading are obtained by orthogonal projection onto the plan polyline
    segments. Endpoint tangents are extended only when a point lies before the
    first segment or after the last segment, so a pruning gap is not counted as
    lateral error.
    """
    n_eval = min(
        max(int(eval_end_idx) + 1, 0),
        int(exec_t_ns.size),
        int(exec_xy.shape[0]),
        int(exec_yaw.size),
    )

    empty = {
        "rmse_y_m": None,
        "p95_y_m": None,
        "rmse_psi_rad": None,
        "rmse_y_finite_segment_diagnostic_m": None,
        "p95_y_finite_segment_diagnostic_m": None,
        "tracking_samples": 0,
        "tracking_coverage": 0.0,
        "endpoint_extension_samples": 0,
        "endpoint_extension_fraction": 0.0,
        "plan_message_count": 0,
        "plan_age_median_s": None,
        "plan_age_max_s": None,
        "frame_transform_samples": 0,
        "frame_transform_coverage": 0.0,
        "frame_transform_used": "unavailable",
        "plan_tracking_available": 0,
    }

    if n_eval < 2 or plan_history is None:
        return empty

    messages = list(plan_history.get("messages", []))
    plan_t_ns = np.asarray(plan_history.get("t_ns", []), dtype=np.int64)
    if not messages or plan_t_ns.size == 0 or plan_t_ns.size != len(messages):
        return empty

    exec_t_eval = np.asarray(exec_t_ns[:n_eval], dtype=np.int64)
    exec_xy_input = np.asarray(exec_xy[:n_eval], dtype=float)
    exec_yaw_input = np.asarray(exec_yaw[:n_eval], dtype=float)

    exec_xy_eval = np.full_like(exec_xy_input, np.nan, dtype=float)
    exec_yaw_eval = np.full_like(exec_yaw_input, np.nan, dtype=float)
    frame_transform_valid = np.zeros((n_eval,), dtype=bool)
    frame_transform_descriptions: List[str] = []

    for sample_index in range(n_eval):
        transform = lookup_direct_transform_2d(
            source_frame=executed_frame_id,
            target_frame=plan_frame_id,
            query_t_ns=int(exec_t_eval[sample_index]),
            tf_history=tf_history,
        )
        if transform is None:
            continue

        tx, ty, theta, description = transform
        c = math.cos(theta)
        s = math.sin(theta)
        x_source = float(exec_xy_input[sample_index, 0])
        y_source = float(exec_xy_input[sample_index, 1])

        exec_xy_eval[sample_index, 0] = tx + c * x_source - s * y_source
        exec_xy_eval[sample_index, 1] = ty + s * x_source + c * y_source
        exec_yaw_eval[sample_index] = float(
            wrap_angle_rad(
                np.asarray([exec_yaw_input[sample_index] + theta], dtype=float)
            )[0]
        )
        frame_transform_valid[sample_index] = True
        frame_transform_descriptions.append(description)

    frame_transform_samples = int(np.count_nonzero(frame_transform_valid))
    frame_transform_coverage = float(frame_transform_samples / n_eval) if n_eval else 0.0
    frame_transform_used = (
        Counter(frame_transform_descriptions).most_common(1)[0][0]
        if frame_transform_descriptions else "unavailable"
    )

    active_plan_idx = np.searchsorted(plan_t_ns, exec_t_eval, side="right") - 1
    has_active_plan = (active_plan_idx >= 0) & frame_transform_valid

    dist = np.full((n_eval,), np.nan, dtype=float)
    heading_ref = np.full((n_eval,), np.nan, dtype=float)
    endpoint_extrapolated = np.zeros((n_eval,), dtype=bool)
    finite_segment_dist = np.full((n_eval,), np.nan, dtype=float)

    for plan_idx in np.unique(active_plan_idx[has_active_plan]):
        plan_idx_int = int(plan_idx)
        sample_idx = np.flatnonzero(
            (active_plan_idx == plan_idx_int) & frame_transform_valid
        )
        if sample_idx.size == 0:
            continue

        ref_xy = np.asarray(messages[plan_idx_int]["xy"], dtype=float)
        if ref_xy.shape[0] < 2:
            continue

        projected = project_points_to_polyline(
            exec_xy_eval[sample_idx],
            ref_xy,
            extend_endpoint_tangents=True,
        )
        dist[sample_idx] = projected["dist"]
        heading_ref[sample_idx] = projected["heading_ref"]
        endpoint_extrapolated[sample_idx] = projected["endpoint_extrapolated"]
        finite_segment_dist[sample_idx] = projected["finite_segment_dist"]

    finite = np.isfinite(dist) & np.isfinite(heading_ref) & np.isfinite(exec_yaw_eval)
    tracking_samples = int(np.count_nonzero(finite))
    tracking_coverage = float(tracking_samples / n_eval) if n_eval > 0 else 0.0

    if tracking_samples == 0:
        result = dict(empty)
        result["plan_message_count"] = len(messages)
        result["frame_transform_samples"] = frame_transform_samples
        result["frame_transform_coverage"] = frame_transform_coverage
        result["frame_transform_used"] = frame_transform_used
        return result

    dist_valid = dist[finite]
    finite_segment_dist_valid = finite_segment_dist[finite]
    yaw_error = wrap_angle_rad(exec_yaw_eval[finite] - heading_ref[finite])
    endpoint_extension_samples = int(
        np.count_nonzero(endpoint_extrapolated[finite])
    )
    endpoint_extension_fraction = float(
        endpoint_extension_samples / tracking_samples
    )

    matched_plan_idx = active_plan_idx[finite]
    plan_age_s = (
        exec_t_eval[finite] - plan_t_ns[matched_plan_idx]
    ).astype(float) / 1e9
    plan_age_s = plan_age_s[np.isfinite(plan_age_s) & (plan_age_s >= 0.0)]

    return {
        "rmse_y_m": float(np.sqrt(np.mean(dist_valid ** 2))),
        "p95_y_m": float(np.percentile(dist_valid, 95)),
        "rmse_psi_rad": float(np.sqrt(np.mean(yaw_error ** 2))),
        "rmse_y_finite_segment_diagnostic_m": float(
            np.sqrt(np.mean(finite_segment_dist_valid ** 2))
        ),
        "p95_y_finite_segment_diagnostic_m": float(
            np.percentile(finite_segment_dist_valid, 95)
        ),
        "tracking_samples": tracking_samples,
        "tracking_coverage": tracking_coverage,
        "endpoint_extension_samples": endpoint_extension_samples,
        "endpoint_extension_fraction": endpoint_extension_fraction,
        "plan_message_count": len(messages),
        "plan_age_median_s": float(np.median(plan_age_s)) if plan_age_s.size else None,
        "plan_age_max_s": float(np.max(plan_age_s)) if plan_age_s.size else None,
        "frame_transform_samples": frame_transform_samples,
        "frame_transform_coverage": frame_transform_coverage,
        "frame_transform_used": frame_transform_used,
        "plan_tracking_available": int(
            tracking_samples >= 2 and tracking_coverage >= float(min_plan_coverage)
        ),
    }


def compute_speed_and_smoothness(
    exec_t_s: np.ndarray,
    exec_v: np.ndarray,
    eval_end_idx: int,
    target_speed_mps: float,
    cmd: Optional[Dict[str, Any]],
    eval_end_time_s: float,
    wheelbase_m: float,
    steering_min_speed_mps: float,
) -> Dict[str, Optional[float]]:
    v_eval = exec_v[:eval_end_idx + 1]
    t_eval = exec_t_s[:eval_end_idx + 1]

    rmse_v = float(np.sqrt(np.mean((v_eval - target_speed_mps) ** 2))) if v_eval.size else None

    if v_eval.size:
        t_v_smooth, v_smooth = moving_average_with_time(t_eval, v_eval, window=11)
        t_accel, accel = derivative_with_time(t_v_smooth, v_smooth, min_dt_s=1e-2)
        t_accel_smooth, accel_smooth = moving_average_with_time(t_accel, accel, window=11)
        _, jx = derivative_with_time(t_accel_smooth, accel_smooth, min_dt_s=1e-2)
        rms_jx = rms(jx)
    else:
        rms_jx = None

    rms_dotdelta = None
    rms_cmd_w = None

    if cmd is not None:
        mask = cmd["t_s"] <= eval_end_time_s + 1e-9
        cmd_t_eval = cmd["t_s"][mask]
        vx = cmd["vx"][mask]
        wz = cmd["wz"][mask]

        if wz.size:
            rms_cmd_w = rms(wz)

        steering_mask = (
            np.isfinite(cmd_t_eval)
            & np.isfinite(vx)
            & np.isfinite(wz)
            & (np.abs(vx) >= steering_min_speed_mps)
        )
        if np.count_nonzero(steering_mask) >= 2:
            steering_t = cmd_t_eval[steering_mask]
            steering_delta = np.arctan(
                wheelbase_m * wz[steering_mask] / vx[steering_mask]
            )
            steering_t_smooth, steering_delta_smooth = moving_average_with_time(
                steering_t,
                steering_delta,
                window=11,
            )
            dotdelta = derivative(steering_t_smooth, steering_delta_smooth, min_dt_s=1e-2)
            rms_dotdelta = rms(dotdelta)

    return {
        "rmse_v_mps": rmse_v,
        "rms_jx_mps3": rms_jx,
        "rms_dotdelta_radps": rms_dotdelta,
        "rms_cmd_w_radps": rms_cmd_w,
    }


def prepare_human_gt_context(
    gt_gps_csv_glob: str,
    waypoint_file: str,
    n_samples: int,
) -> Optional[Dict[str, Any]]:
    if not gt_gps_csv_glob and not waypoint_file:
        return None
    if not gt_gps_csv_glob or not waypoint_file:
        raise RuntimeError("To enable human-likeness metrics, provide both --gt-gps-csv-glob and --waypoint-file")

    try:
        import debug_raw_gps_waypoint_alignment as human_gps_debug
    except Exception as e:
        raise RuntimeError(f"Could not import debug_raw_gps_waypoint_alignment.py: {e}")

    waypoint_path = Path(waypoint_file).expanduser().resolve()
    waypoint_ref = human_gps_debug.load_waypoint_file(waypoint_path)

    gt_csv_paths = human_gps_debug.collect_gt_csv_paths(gt_gps_csv_glob)
    gt_ref = human_gps_debug.build_gt_reference(gt_csv_paths, waypoint_ref, int(n_samples))

    return {
        "module": human_gps_debug,
        "waypoint_ref": waypoint_ref,
        "gt_ref": gt_ref,
        "n_samples": int(n_samples),
        "gt_gps_csv_glob": gt_gps_csv_glob,
        "waypoint_file": str(waypoint_path),
    }


def read_bag_gps_latlon_with_types(bag_dir: Path, gps_topic: str) -> Optional[Dict[str, Any]]:
    reader = rosbag2_py.SequentialReader()
    bag_uri = resolve_bag_uri_for_mcap(bag_dir)

    storage_options = rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topic_type_map: Dict[str, str] = {}
    for t in reader.get_all_topics_and_types():
        topic_type_map[t.name] = t.type

    if gps_topic not in topic_type_map:
        return None

    try:
        if hasattr(rosbag2_py, "StorageFilter"):
            reader.set_filter(rosbag2_py.StorageFilter(topics=[gps_topic]))
    except Exception:
        pass

    msg_type = get_message(topic_type_map[gps_topic])

    t_ns_list: List[int] = []
    lat_list: List[float] = []
    lon_list: List[float] = []

    while reader.has_next():
        topic, rawdata, t_ns = reader.read_next()
        if topic != gps_topic:
            continue
        try:
            msg = deserialize_message(rawdata, msg_type)
        except Exception:
            continue

        try:
            lat = float(msg.latitude)
            lon = float(msg.longitude)
            if not math.isfinite(lat) or not math.isfinite(lon):
                continue
            if abs(lat) > 90.0 or abs(lon) > 180.0:
                continue
        except Exception:
            continue

        t_ns_list.append(int(t_ns))
        lat_list.append(lat)
        lon_list.append(lon)

    if len(t_ns_list) < 2:
        return None

    t_ns = np.asarray(t_ns_list, dtype=np.int64)
    lat = np.asarray(lat_list, dtype=float)
    lon = np.asarray(lon_list, dtype=float)

    idx = np.argsort(t_ns, kind="stable")
    t_ns = t_ns[idx]
    lat = lat[idx]
    lon = lon[idx]

    return {"t_ns": t_ns, "lat": lat, "lon": lon}


def compute_humanlikeness_for_bag(
    bag_dir: Path,
    gps_topic: str,
    human_ctx: Optional[Dict[str, Any]],
    cutoff_t_ns: int,
    heading_window_m: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "has_human_gt_gps": 1 if human_ctx is not None else 0,
        "human_gt_reference": "not_enabled",
        "human_gt_alignment_mode": "not_enabled",
        "human_gt_best_variant": "",
        "human_gt_rmse_y_m": None,
        "human_gt_p95_y_m": None,
        "human_gt_max_y_m": None,
        "human_gt_progress": None,
        "human_gt_pointwise_rmse_m": None,
        "human_gt_note": "",
    }

    if human_ctx is None:
        return out

    mod = human_ctx["module"]
    waypoint_ref = human_ctx["waypoint_ref"]
    gt_ref = human_ctx["gt_ref"]
    n_samples = int(human_ctx["n_samples"])

    out["human_gt_reference"] = "gt_mean_raw_gps"
    out["human_gt_alignment_mode"] = "variant_heading_start_aligned"

    raw = mod.read_bag_gps_latlon(bag_dir, gps_topic)
    if raw is None or "t_ns" not in raw:
        raw2 = read_bag_gps_latlon_with_types(bag_dir, gps_topic)
        if raw2 is None:
            out["human_gt_note"] = "HUMAN_GPS_MISSING"
            return out
        raw = raw2

    mask = raw["t_ns"] <= int(cutoff_t_ns)
    if not np.any(mask):
        out["human_gt_note"] = "HUMAN_GPS_EMPTY_BEFORE_CUTOFF"
        return out

    lat = np.asarray(raw["lat"], dtype=float)[mask]
    lon = np.asarray(raw["lon"], dtype=float)[mask]
    t_ns = np.asarray(raw["t_ns"], dtype=np.int64)[mask]

    raw_cut = {"t_ns": t_ns, "lat": lat, "lon": lon}

    projected = mod.project_trace_to_common_origin(raw_cut, waypoint_ref["lat0_deg"], waypoint_ref["lon0_deg"])
    raw_xy = projected["xy"].copy()

    # Mesma correção fixa usada no script de charts
    raw_xy = np.column_stack((-raw_xy[:, 1], raw_xy[:, 0]))

    if raw_xy.shape[0] < 2:
        out["human_gt_note"] = "HUMAN_GPS_TOO_SHORT"
        return out

    best = mod.choose_best_candidate(raw_xy, waypoint_ref["xy"], gt_ref["gt_mean_xy"], n_samples)

    best_rot_xy, residual_theta_rad = mod.apply_residual_heading_alignment(
        best["xy"],
        gt_ref["gt_mean_xy"],
        window_m=heading_window_m,
    )

    aligned_xy = mod.align_trace_start_to_reference(best_rot_xy, gt_ref["gt_mean_xy"])
    metrics = mod.compute_curve_metrics_vs_reference(aligned_xy, gt_ref["gt_mean_xy"], n_samples)

    out["human_gt_best_variant"] = str(best.get("variant", ""))
    out["human_gt_note"] = f"HUMAN_GPS_VARIANT={out['human_gt_best_variant']}; RESIDUAL_THETA_DEG={math.degrees(residual_theta_rad):.3f}"
    out["human_gt_rmse_y_m"] = float(metrics["rmse_lateral_m"])
    out["human_gt_p95_y_m"] = float(metrics["p95_lateral_m"])
    out["human_gt_max_y_m"] = float(metrics["max_lateral_m"])
    out["human_gt_progress"] = float(metrics["progress"])
    out["human_gt_pointwise_rmse_m"] = float(metrics["pointwise_rmse_m"])

    if out["human_gt_best_variant"] and out["human_gt_best_variant"] != "raw":
        out["human_gt_note"] = f"HUMAN_GPS_VARIANT={out['human_gt_best_variant']}"

    return out



def to_row_dict(run: RunMetrics) -> Dict[str, Any]:
    return asdict(run)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_one_combination(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(rows)
    n_success = sum(int(r["success_geo"]) for r in rows)
    n_valid_mission = sum(int(r.get("valid_for_mission", 0)) for r in rows)
    n_valid_tracking = sum(int(r["valid_for_tracking"]) for r in rows)

    def vals_all(key: str) -> List[float]:
        outv: List[float] = []
        for r in rows:
            v = r.get(key)
            if v is None or v == "":
                continue
            try:
                fv = float(v)
                if math.isfinite(fv):
                    outv.append(fv)
            except Exception:
                pass
        return outv

    def vals_tracking_valid(key: str) -> List[float]:
        outv: List[float] = []
        for r in rows:
            if int(r["valid_for_tracking"]) != 1:
                continue
            v = r.get(key)
            if v is None or v == "":
                continue
            try:
                fv = float(v)
                if math.isfinite(fv):
                    outv.append(fv)
            except Exception:
                pass
        return outv

    def vals_mission_valid(key: str) -> List[float]:
        outv: List[float] = []
        for r in rows:
            if int(r.get("valid_for_mission", 0)) != 1:
                continue
            v = r.get(key)
            if v is None or v == "":
                continue
            try:
                fv = float(v)
                if math.isfinite(fv):
                    outv.append(fv)
            except Exception:
                pass
        return outv

    primary_ref_counts: Dict[str, int] = {}
    for r in rows:
        ref_name = str(r.get("primary_reference", "")).strip()
        if ref_name:
            primary_ref_counts[ref_name] = primary_ref_counts.get(ref_name, 0) + 1
    primary_ref_summary = max(primary_ref_counts.items(), key=lambda kv: kv[1])[0] if primary_ref_counts else "reference_missing"

    human_variant_counts: Dict[str, int] = {}
    for r in rows:
        if int(r.get("valid_for_mission", 0)) != 1:
            continue
        variant = str(r.get("human_gt_best_variant", "")).strip()
        if variant:
            human_variant_counts[variant] = human_variant_counts.get(variant, 0) + 1
    dominant_human_variant = max(human_variant_counts.items(), key=lambda kv: kv[1])[0] if human_variant_counts else "not_enabled"

    human_metric_runs = len(vals_mission_valid("human_gt_rmse_y_m"))

    agg = {
        "scenario_id": rows[0]["scenario_id"],
        "speed_kmh": rows[0]["speed_kmh"],
        "planner_id": rows[0]["planner_id"],
        "controller_id": rows[0]["controller_id"],
        "method_label": f"{rows[0]['planner_id']}, {rows[0]['controller_id']}",

        "N_total": n_total,
        "N_success_geo": n_success,
        "N_valid_for_mission": n_valid_mission,
        "N_valid_for_tracking": n_valid_tracking,
        "N_human_metric_runs": human_metric_runs,
        "SR_geo_text": f"{n_success}/{n_total}",
        "SR_geo_rate": (n_success / n_total) if n_total > 0 else None,

        "primary_tracking_reference": primary_ref_summary,
        "human_tracking_reference": "gt_mean_raw_gps_start_normalized" if human_metric_runs > 0 else "not_enabled",
        "human_best_variant_mode": dominant_human_variant,

        "PR_median_iqr": fmt_median_iqr(vals_all("progress_ratio")),
        "T_median_iqr": fmt_median_iqr(vals_mission_valid("mission_time_s")),
        "RMSE_y_primary_median_iqr": fmt_median_iqr(vals_tracking_valid("rmse_y_primary_m")),
        "P95_y_primary_median_iqr": fmt_median_iqr(vals_tracking_valid("p95_y_primary_m")),
        "RMSE_psi_primary_median_iqr": fmt_median_iqr(vals_tracking_valid("rmse_psi_primary_rad")),
        "RMSE_y_finite_segment_diagnostic_median_iqr": fmt_median_iqr(
            vals_tracking_valid("rmse_y_finite_segment_diagnostic_m")
        ),
        "P95_y_finite_segment_diagnostic_median_iqr": fmt_median_iqr(
            vals_tracking_valid("p95_y_finite_segment_diagnostic_m")
        ),
        "RMSE_v_median_iqr": fmt_median_iqr(vals_mission_valid("rmse_v_mps")),
        "RMS_jx_median_iqr": fmt_median_iqr(vals_mission_valid("rms_jx_mps3")),
        "RMS_dotdelta_median_iqr": fmt_median_iqr(vals_mission_valid("rms_dotdelta_radps")),
        "RMS_cmd_w_median_iqr": fmt_median_iqr(vals_mission_valid("rms_cmd_w_radps")),

        "Exec_Human_RMSE_y_median_iqr": fmt_median_iqr(vals_mission_valid("human_gt_rmse_y_m")),
        "Exec_Human_P95_y_median_iqr": fmt_median_iqr(vals_mission_valid("human_gt_p95_y_m")),
        "Exec_Human_MAX_y_median_iqr": fmt_median_iqr(vals_mission_valid("human_gt_max_y_m")),
        "Exec_Human_PR_median_iqr": fmt_median_iqr(vals_mission_valid("human_gt_progress")),
        "Exec_Human_Pointwise_RMSE_median_iqr": fmt_median_iqr(vals_mission_valid("human_gt_pointwise_rmse_m")),
    }
    return agg


def build_primary_table_row(agg: Dict[str, Any]) -> str:
    return (
        f"{agg['speed_kmh']} km/h & {agg['method_label']} & "
        f"{agg['SR_geo_text']} & {agg['PR_median_iqr']} & {agg['T_median_iqr']} & "
        f"{agg['RMSE_y_primary_median_iqr']} & {agg['P95_y_primary_median_iqr']} \\\\"
    )


def build_secondary_table_row(agg: Dict[str, Any]) -> str:
    return (
        f"{agg['speed_kmh']} km/h & {agg['method_label']} & "
        f"{agg['RMSE_psi_primary_median_iqr']} & {agg['RMSE_v_median_iqr']} & "
        f"{agg['RMS_jx_median_iqr']} & {agg['RMS_dotdelta_median_iqr']} \\\\"
    )


def build_humanlikeness_table_row(agg: Dict[str, Any]) -> str:
    return (
        f"{agg['speed_kmh']} km/h & {agg['method_label']} & "
        f"{agg['Exec_Human_RMSE_y_median_iqr']} & {agg['Exec_Human_P95_y_median_iqr']} & "
        f"{agg['Exec_Human_MAX_y_median_iqr']} & {agg['Exec_Human_PR_median_iqr']} \\\\"
    )


def main() -> int:
    args = parse_args()

    if args.wheelbase_m <= 0.0:
        print("ERROR: --wheelbase-m must be positive.", file=sys.stderr)
        return 7
    if args.steering_min_speed_mps < 0.0:
        print("ERROR: --steering-min-speed-mps cannot be negative.", file=sys.stderr)
        return 8
    if not 0.0 <= args.min_plan_coverage <= 1.0:
        print("ERROR: --min-plan-coverage must be in [0, 1].", file=sys.stderr)
        return 9

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = (Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (runs_dir / "Output_metrics_assessment").resolve())
    gt_csv = Path(args.gt_csv).expanduser().resolve() if args.gt_csv else None

    if not runs_dir.exists():
        print(f"ERROR: runs dir does not exist: {runs_dir}", file=sys.stderr)
        return 2

    if gt_csv is None:
        print("ERROR: --gt-csv is required.", file=sys.stderr)
        return 3

    try:
        gt_ref = load_gt_reference_csv(gt_csv)
    except Exception as e:
        print(f"ERROR: could not load common GT CSV reference: {e}", file=sys.stderr)
        return 6
    progress_ref_name = f"gt_csv_start_heading_aligned:{gt_ref['source_name']}"

    try:
        human_ctx = prepare_human_gt_context(
            gt_gps_csv_glob=args.gt_gps_csv_glob,
            waypoint_file=args.waypoint_file,
            n_samples=args.human_resample_samples,
        )
    except Exception as e:
        print(f"ERROR: could not prepare human-likeness reference: {e}", file=sys.stderr)
        return 5

    bag_dirs = list_bag_dirs(runs_dir)
    if not bag_dirs:
        print(f"ERROR: no bag folders found under: {runs_dir}", file=sys.stderr)
        return 4

    plan_topic_candidates = [x.strip() for x in args.plan_topics.split(",") if x.strip()]
    cmd_topic_candidates = [x.strip() for x in args.cmd_topics.split(",") if x.strip()]
    target_speed_mps = args.speed_kmh / 3.6
    heading_window_m = get_heading_window_m_for_scenario(args.scenario_id)

    print(f"[INFO] Found {len(bag_dirs)} run bags in {runs_dir}")
    print(f"[INFO] Metrics output dir: {out_dir}")
    print(f"[INFO] Mission-progress common benchmark CSV: {gt_ref['source_csv']}")
    print(f"[INFO] Mission-progress benchmark length [m]: {gt_ref['path_length_m']:.3f}")
    print("[INFO] Primary lateral/heading reference: time-synchronized recorded Navigation2 plan")
    print(f"[INFO] Nav2 plan topic priority: {','.join(plan_topic_candidates)}")
    print("[INFO] Plan synchronization: latest plan at or before odometry MCAP storage timestamp")
    print(f"[INFO] Minimum plan-tracking coverage = {args.min_plan_coverage:.2f}")
    if human_ctx is not None:
        print("[INFO] Human-likeness GT raw GPS enabled")
        print(f"[INFO] Human GT GPS glob: {human_ctx['gt_gps_csv_glob']}")
        print(f"[INFO] Human waypoint file: {human_ctx['waypoint_file']}")
        print(f"[INFO] Human common origin lat0_deg: {human_ctx['waypoint_ref']['lat0_deg']:.10f}")
        print(f"[INFO] Human common origin lon0_deg: {human_ctx['waypoint_ref']['lon0_deg']:.10f}")
        print("[INFO] Human metrics will be computed only for valid_for_mission runs")
    else:
        print("[INFO] Human-likeness GT raw GPS: disabled")

    print(f"[INFO] success_PR_threshold = {args.success_pr_threshold:.2f}")
    print(f"[INFO] tracking_PR_threshold = {args.tracking_pr_threshold:.2f}")
    print(f"[INFO] Residual heading alignment window_m = {heading_window_m:.1f}")
    print("[INFO] Longitudinal jerk: second derivative of 11-sample-smoothed speed")
    print(f"[INFO] Ackermann steering reconstruction wheelbase_m = {args.wheelbase_m:.3f}")
    print(f"[INFO] Steering reconstruction minimum speed_mps = {args.steering_min_speed_mps:.3f}")

    per_run_rows: List[Dict[str, Any]] = []

    for idx, bag_dir in enumerate(bag_dirs, start=1):
        bag_name = bag_dir.name
        run_id = parse_run_id_from_name(bag_name)

        print(f"[{idx}/{len(bag_dirs)}] Processing {bag_name}")

        notes: List[str] = []

        try:
            bag = read_bag_data(
                bag_dir=bag_dir,
                executed_topic=args.executed_topic,
                plan_topic_candidates=plan_topic_candidates,
                cmd_topic_candidates=cmd_topic_candidates,
            )
        except Exception as e:
            notes.append(f"BAG_READ_FAILED: {e}")
            metric = RunMetrics(
                scenario_id=args.scenario_id,
                speed_kmh=args.speed_kmh,
                planner_id=args.planner_id,
                controller_id=args.controller_id,
                run_id=run_id,
                bag_name=bag_name,
                bag_dir=str(bag_dir),

                primary_reference="nav2_plan_unavailable",
                progress_reference=progress_ref_name,
                plan_topic_used="",
                plan_frame_id="",
                executed_frame_id="",
                frame_transform_used="",
                executed_topic_used=args.executed_topic,
                cmd_topic_used="",

                has_plan=0,
                plan_message_count=0,
                plan_tracking_samples=0,
                plan_tracking_coverage=0.0,
                plan_endpoint_extension_samples=0,
                plan_endpoint_extension_fraction=0.0,
                has_human_gt_gps=int(human_ctx is not None),

                success_geo=0,
                progress_ratio=0.0,
                valid_for_mission=0,
                valid_for_tracking=0,
                eval_cutoff_reason="bag_read_failed",

                mission_time_s=float("nan"),
                raw_run_duration_s=float("nan"),

                final_dist_to_goal_m=float("nan"),
                final_yaw_err_to_goal_rad=float("nan"),

                rmse_y_primary_m=None,
                p95_y_primary_m=None,
                rmse_psi_primary_rad=None,
                rmse_y_finite_segment_diagnostic_m=None,
                p95_y_finite_segment_diagnostic_m=None,

                rmse_v_mps=None,
                rms_jx_mps3=None,
                rms_dotdelta_radps=None,
                rms_cmd_w_radps=None,

                human_gt_reference=("gt_mean_raw_gps" if human_ctx is not None else "not_enabled"),
                human_gt_alignment_mode=("variant_start_aligned" if human_ctx is not None else "not_enabled"),
                human_gt_best_variant="",
                human_gt_rmse_y_m=None,
                human_gt_p95_y_m=None,
                human_gt_max_y_m=None,
                human_gt_progress=None,
                human_gt_pointwise_rmse_m=None,

                notes="; ".join(notes),
            )
            per_run_rows.append(to_row_dict(metric))
            continue

        exec_data = bag["exec"]
        plan_data = bag["plan"]
        cmd_data = bag["cmd"]

        has_plan = int(
            plan_data is not None
            and int(plan_data.get("message_count", 0)) > 0
            and len(plan_data.get("messages", [])) > 0
        )

        progress_ref_aligned = align_reference_to_execution_start_heading(
            ref_xy=gt_ref["xy"],
            exec_xy=exec_data["xy"],
            heading_window_m=heading_window_m,
        )
        mission_metrics = compute_tracking_against_reference(
            exec_t_s=exec_data["t_s"],
            exec_xy=exec_data["xy"],
            exec_yaw=exec_data["yaw"],
            ref_xy=progress_ref_aligned["xy"],
            ref_goal_xy=progress_ref_aligned["goal_xy"],
            ref_goal_yaw=progress_ref_aligned["goal_yaw"],
            success_pr_threshold=args.success_pr_threshold,
            tracking_pr_threshold=args.tracking_pr_threshold,
        )
        notes.append(
            f"PROGRESS_REF_ALIGN_MODE={progress_ref_aligned['mode']}; "
            f"PROGRESS_REF_ALIGN_THETA_DEG={math.degrees(progress_ref_aligned['rotation_rad']):.3f}"
        )
        eval_end_idx = int(mission_metrics["eval_end_idx"])

        plan_metrics = compute_tracking_against_time_varying_plan(
            exec_t_ns=exec_data["t_ns"],
            exec_xy=exec_data["xy"],
            exec_yaw=exec_data["yaw"],
            executed_frame_id=str(bag.get("executed_frame_id", "")),
            plan_frame_id=str(bag.get("plan_frame_id", "")),
            tf_history=bag.get("tf_history", {}),
            plan_history=plan_data,
            eval_end_idx=eval_end_idx,
            min_plan_coverage=args.min_plan_coverage,
        )
        valid_for_mission = int(mission_metrics["valid_for_tracking"])
        valid_for_tracking = int(
            valid_for_mission == 1
            and int(plan_metrics["plan_tracking_available"]) == 1
        )

        plan_age_median = plan_metrics.get("plan_age_median_s")
        plan_age_max = plan_metrics.get("plan_age_max_s")
        notes.append(
            "PRIMARY_TRACKING_REF=TIME_SYNCHRONIZED_NAV2_PLAN; "
            "PLAN_MATCH=LATEST_AT_OR_BEFORE_ODOM; "
            "PLAN_CROSSTRACK_DEF=POLYLINE_SEGMENTS_WITH_ENDPOINT_TANGENT_EXTENSION; "
            f"PLAN_TRACKING_SAMPLES={plan_metrics['tracking_samples']}; "
            f"PLAN_TRACKING_COVERAGE={plan_metrics['tracking_coverage']:.6f}; "
            f"PLAN_ENDPOINT_EXTENSION_SAMPLES={plan_metrics['endpoint_extension_samples']}; "
            f"PLAN_ENDPOINT_EXTENSION_FRACTION={plan_metrics['endpoint_extension_fraction']:.6f}; "
            f"FRAME_TRANSFORM={plan_metrics['frame_transform_used']}; "
            f"FRAME_TRANSFORM_COVERAGE={plan_metrics['frame_transform_coverage']:.6f}; "
            f"PLAN_AGE_MEDIAN_S={plan_age_median if plan_age_median is not None else 'NA'}; "
            f"PLAN_AGE_MAX_S={plan_age_max if plan_age_max is not None else 'NA'}"
        )
        if valid_for_mission and not valid_for_tracking:
            notes.append(
                f"PLAN_TRACKING_INVALID_COVERAGE_LT_{args.min_plan_coverage:.2f}_OR_INSUFFICIENT_SAMPLES"
            )

        smooth = compute_speed_and_smoothness(
            exec_t_s=exec_data["t_s"],
            exec_v=exec_data["v"],
            eval_end_idx=eval_end_idx,
            target_speed_mps=target_speed_mps,
            cmd=cmd_data,
            eval_end_time_s=exec_data["t_s"][eval_end_idx],
            wheelbase_m=args.wheelbase_m,
            steering_min_speed_mps=args.steering_min_speed_mps,
        )
        notes.append(
            "JERK_DEF=d2(smoothed_speed)/dt2; "
            f"STEERING_RATE_DEF=d_atan(L*wz/vx)/dt; L_M={args.wheelbase_m:.3f}; "
            f"MIN_ABS_VX_MPS={args.steering_min_speed_mps:.3f}"
        )

        aux_human = {
            "has_human_gt_gps": int(human_ctx is not None),
            "human_gt_reference": "not_enabled",
            "human_gt_alignment_mode": "not_enabled",
            "human_gt_best_variant": "",
            "human_gt_rmse_y_m": None,
            "human_gt_p95_y_m": None,
            "human_gt_max_y_m": None,
            "human_gt_progress": None,
            "human_gt_pointwise_rmse_m": None,
            "human_gt_note": "HUMAN_GPS_SKIPPED_NOT_VALID_FOR_MISSION",
        }

        if human_ctx is not None and valid_for_mission == 1:
            cutoff_t_ns = int(exec_data["t_ns"][eval_end_idx])
            try:
                aux_human = compute_humanlikeness_for_bag(
                    bag_dir=bag_dir,
                    gps_topic=args.gps_topic,
                    human_ctx=human_ctx,
                    cutoff_t_ns=cutoff_t_ns,
                    heading_window_m=heading_window_m,
                )
            except Exception as e:
                aux_human["human_gt_note"] = f"HUMAN_GPS_COMPARE_FAILED: {e}"

        if not has_plan:
            notes.append("NO_VALID_PLAN_TOPIC_FOUND")
        if aux_human.get("human_gt_note"):
            notes.append(str(aux_human["human_gt_note"]))
        notes.extend([str(x) for x in bag.get("notes", [])])

        metric = RunMetrics(
            scenario_id=args.scenario_id,
            speed_kmh=args.speed_kmh,
            planner_id=args.planner_id,
            controller_id=args.controller_id,
            run_id=run_id,
            bag_name=bag_name,
            bag_dir=str(bag_dir),

            primary_reference=(
                f"nav2_active_plan_cross_track_v2:{bag['plan_topic_used']}"
                if has_plan else "nav2_plan_unavailable"
            ),
            progress_reference=progress_ref_name,
            plan_topic_used=bag["plan_topic_used"] if bag["plan_topic_used"] else "",
            plan_frame_id=str(bag.get("plan_frame_id", "")),
            executed_frame_id=str(bag.get("executed_frame_id", "")),
            frame_transform_used=str(plan_metrics.get("frame_transform_used", "unavailable")),
            executed_topic_used=bag["executed_topic_used"],
            cmd_topic_used=bag["cmd_topic_used"] if bag["cmd_topic_used"] else "",

            has_plan=has_plan,
            plan_message_count=int(plan_metrics["plan_message_count"]),
            plan_tracking_samples=int(plan_metrics["tracking_samples"]),
            plan_tracking_coverage=float(plan_metrics["tracking_coverage"]),
            plan_endpoint_extension_samples=int(
                plan_metrics["endpoint_extension_samples"]
            ),
            plan_endpoint_extension_fraction=float(
                plan_metrics["endpoint_extension_fraction"]
            ),
            has_human_gt_gps=int(aux_human.get("has_human_gt_gps", 0)),

            success_geo=int(mission_metrics["success_geo"]),
            progress_ratio=float(mission_metrics["progress_ratio"]),
            valid_for_mission=valid_for_mission,
            valid_for_tracking=valid_for_tracking,
            eval_cutoff_reason=str(mission_metrics["eval_cutoff_reason"]),

            mission_time_s=float(mission_metrics["mission_time_s"]),
            raw_run_duration_s=float(mission_metrics["raw_run_duration_s"]),

            final_dist_to_goal_m=float(mission_metrics["final_dist_to_goal_m"]),
            final_yaw_err_to_goal_rad=float(mission_metrics["final_yaw_err_to_goal_rad"]),

            rmse_y_primary_m=(plan_metrics["rmse_y_m"] if valid_for_tracking else None),
            p95_y_primary_m=(plan_metrics["p95_y_m"] if valid_for_tracking else None),
            rmse_psi_primary_rad=(plan_metrics["rmse_psi_rad"] if valid_for_tracking else None),
            rmse_y_finite_segment_diagnostic_m=(
                plan_metrics["rmse_y_finite_segment_diagnostic_m"]
                if valid_for_tracking else None
            ),
            p95_y_finite_segment_diagnostic_m=(
                plan_metrics["p95_y_finite_segment_diagnostic_m"]
                if valid_for_tracking else None
            ),

            rmse_v_mps=smooth["rmse_v_mps"],
            rms_jx_mps3=smooth["rms_jx_mps3"],
            rms_dotdelta_radps=smooth["rms_dotdelta_radps"],
            rms_cmd_w_radps=smooth["rms_cmd_w_radps"],

            human_gt_reference=str(aux_human.get("human_gt_reference", "not_enabled")),
            human_gt_alignment_mode=str(aux_human.get("human_gt_alignment_mode", "not_enabled")),
            human_gt_best_variant=str(aux_human.get("human_gt_best_variant", "")),
            human_gt_rmse_y_m=aux_human.get("human_gt_rmse_y_m", None),
            human_gt_p95_y_m=aux_human.get("human_gt_p95_y_m", None),
            human_gt_max_y_m=aux_human.get("human_gt_max_y_m", None),
            human_gt_progress=aux_human.get("human_gt_progress", None),
            human_gt_pointwise_rmse_m=aux_human.get("human_gt_pointwise_rmse_m", None),

            notes="; ".join([n for n in notes if n]),
        )
        per_run_rows.append(to_row_dict(metric))

    per_run_rows.sort(key=lambda r: (int(r["run_id"]), str(r["bag_name"])))

    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_csv = out_dir / "per_run_metrics.csv"
    write_csv(per_run_csv, per_run_rows)

    agg = aggregate_one_combination(per_run_rows)
    agg_csv = out_dir / "aggregated_metrics.csv"
    write_csv(agg_csv, [agg])

    primary_row_txt = out_dir / "primary_table_row.txt"
    secondary_row_txt = out_dir / "secondary_table_row.txt"
    human_row_txt = out_dir / "humanlikeness_table_row.txt"

    primary_row_txt.write_text(build_primary_table_row(agg) + "\n", encoding="utf-8")
    secondary_row_txt.write_text(build_secondary_table_row(agg) + "\n", encoding="utf-8")
    human_row_txt.write_text(build_humanlikeness_table_row(agg) + "\n", encoding="utf-8")

    print("\n[OK] Finished.")
    print(f"[OK] Per-run CSV: {per_run_csv}")
    print(f"[OK] Aggregated CSV: {agg_csv}")
    print(f"[OK] Primary table row: {primary_row_txt}")
    print(f"[OK] Secondary table row: {secondary_row_txt}")
    print(f"[OK] Human-likeness table row: {human_row_txt}")

    print("\n[SUMMARY]")
    print(f"Method: {agg['method_label']}")
    print(f"Runs: {agg['N_total']}")
    print(f"Success: {agg['SR_geo_text']}")
    print(f"Mission-valid runs: {agg['N_valid_for_mission']}/{agg['N_total']}")
    print(f"Plan-tracking-valid runs: {agg['N_valid_for_tracking']}/{agg['N_total']}")
    print(f"Primary reference: {agg['primary_tracking_reference']}")
    print(f"PR: {agg['PR_median_iqr']}")
    print(f"T: {agg['T_median_iqr']}")
    print(f"RMSE_y: {agg['RMSE_y_primary_median_iqr']}")
    print(f"P95_y: {agg['P95_y_primary_median_iqr']}")
    print(
        "Finite-segment diagnostic RMSE_y: "
        f"{agg['RMSE_y_finite_segment_diagnostic_median_iqr']}"
    )
    print(
        "Finite-segment diagnostic P95_y: "
        f"{agg['P95_y_finite_segment_diagnostic_median_iqr']}"
    )
    print(f"RMSE_psi: {agg['RMSE_psi_primary_median_iqr']}")
    print(f"RMSE_v: {agg['RMSE_v_median_iqr']}")
    print(f"RMS_jx: {agg['RMS_jx_median_iqr']}")
    print(f"RMS_dotdelta: {agg['RMS_dotdelta_median_iqr']}")
    print(f"Human-like RMSE_y: {agg['Exec_Human_RMSE_y_median_iqr']}")
    print(f"Human-like P95_y: {agg['Exec_Human_P95_y_median_iqr']}")
    print(f"Human-like Max_y: {agg['Exec_Human_MAX_y_median_iqr']}")
    print(f"Human-like PR: {agg['Exec_Human_PR_median_iqr']}")
    print(f"Human-like variant mode: {agg['human_best_variant_mode']}")
    print(f"Human metric runs (valid only): {agg['N_human_metric_runs']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
