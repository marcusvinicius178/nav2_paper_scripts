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
Primary metrics (main paper tables):
- success_geo
- progress_ratio (maximum path progress ratio reached during the run)
- mission_time_s
- rmse_y_primary_m
- p95_y_primary_m
- rmse_psi_primary_rad

Primary metrics are computed against the best valid planner path found in the bag:
priority order:
    1) /plan
    2) /plan_smoothed
    3) /transformed_global_plan

Success and validity rules
--------------------------
- Geometric success:
    success_geo = 1 if max(PR(t)) >= 0.90
- Run valid for continuous metrics:
    valid_for_tracking = 1 if max(PR(t)) >= 0.70

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
- Computed only for runs valid_for_tracking == 1 (to match primary metrics validity).

Legacy GT CSV in local XY
-------------------------
--gt-csv is accepted for backward compatibility, but ignored for metrics in this version.
"""

import argparse
import csv
import math
import re
import sys
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
    print("Run: source /opt/ros/jazzy/setup.bash", file=sys.stderr)
    raise


RUN_INDEX_RE = re.compile(r"(\d+)$")


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
    plan_topic_used: str
    executed_topic_used: str
    cmd_topic_used: str

    has_plan: int
    has_human_gt_gps: int

    success_geo: int
    progress_ratio: float
    valid_for_tracking: int
    eval_cutoff_reason: str

    mission_time_s: float
    raw_run_duration_s: float

    final_dist_to_goal_m: float
    final_yaw_err_to_goal_rad: float

    rmse_y_primary_m: Optional[float]
    p95_y_primary_m: Optional[float]
    rmse_psi_primary_rad: Optional[float]

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

    parser.add_argument("--gt-csv", required=True, help="Legacy mean GT CSV in local XY (ignored for metrics)")
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
        default="/plan,/plan_smoothed,/transformed_global_plan",
        help="Comma-separated priority order for valid plan path selection"
    )
    parser.add_argument("--cmd-topics", default="/cmd_vel_nav,/cmd_vel")

    parser.add_argument("--success-pr-threshold", type=float, default=0.90,
                        help="Run is geometric success if max PR >= this threshold")
    parser.add_argument("--tracking-pr-threshold", type=float, default=0.70,
                        help="Run is valid for continuous metrics if max PR >= this threshold")

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


def median_iqr(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return med, (q3 - q1)


def fmt_median_iqr(values: List[float], decimals: int = 2) -> str:
    med, iqr = median_iqr(values)
    if med is None:
        return "--"
    return f"{med:.{decimals}f} [{iqr:.{decimals}f}]"


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


def derivative(t_s: np.ndarray, x: np.ndarray, min_dt_s: float = 1e-3) -> np.ndarray:
    if t_s.size < 2 or x.size < 2 or t_s.size != x.size:
        return np.array([], dtype=float)
    dt = np.diff(t_s)
    dx = np.diff(x)
    valid = dt > min_dt_s
    if not np.any(valid):
        return np.array([], dtype=float)
    return dx[valid] / dt[valid]


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
        "plan_topic_used": "",
        "cmd_topic_used": "",
        "exec": None,
        "plan": None,
        "cmd": None,
        "topic_type_map": {},
        "notes": [],
        "all_valid_plans": {},
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

    if existing_cmd_topics:
        out["cmd_topic_used"] = existing_cmd_topics[0]

    topics_needed = [executed_topic] + existing_plan_topics + existing_cmd_topics

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

    cmd_t_ns: List[int] = []
    cmd_vx: List[float] = []
    cmd_wz: List[float] = []

    best_plan_per_topic: Dict[str, Dict[str, Any]] = {}
    for topic in existing_plan_topics:
        best_plan_per_topic[topic] = {"xy": None, "n_pts": -1, "t_ns": -1}

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

        elif topic in best_plan_per_topic:
            xy = extract_path_xy_from_msg(msg)
            if xy is None:
                continue
            n_pts = int(xy.shape[0])
            if n_pts > best_plan_per_topic[topic]["n_pts"] or (
                n_pts == best_plan_per_topic[topic]["n_pts"] and int(t_ns) > best_plan_per_topic[topic]["t_ns"]
            ):
                best_plan_per_topic[topic]["xy"] = xy
                best_plan_per_topic[topic]["n_pts"] = n_pts
                best_plan_per_topic[topic]["t_ns"] = int(t_ns)

        elif topic == chosen_cmd_topic:
            tw = extract_cmd_twist(msg)
            if tw is None:
                continue
            vx, wz = tw
            cmd_t_ns.append(int(t_ns))
            cmd_vx.append(vx)
            cmd_wz.append(wz)

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
    }

    valid_plan_data: Dict[str, Dict[str, Any]] = {}
    for topic, data in best_plan_per_topic.items():
        xy = data["xy"]
        if xy is None or xy.shape[0] < 2:
            continue
        headings = heading_from_path(xy)
        valid_plan_data[topic] = {
            "xy": xy,
            "heading": headings,
            "s": cumulative_arc_length(xy),
            "length_m": path_length(xy),
            "goal_xy": xy[-1].copy(),
            "goal_yaw": float(headings[-1]) if headings.size else 0.0,
            "points": int(xy.shape[0]),
        }

    out["all_valid_plans"] = valid_plan_data

    chosen_plan_topic, chosen_plan = choose_first_valid_plan(plan_topic_candidates, valid_plan_data)
    out["plan_topic_used"] = chosen_plan_topic
    out["plan"] = chosen_plan

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
            "t_s": (cmd_t_ns_arr - cmd_t_ns_arr[0]).astype(float) / 1e9,
            "vx": cmd_vx_arr,
            "wz": cmd_wz_arr,
        }

    if not chosen_plan_topic:
        out["notes"].append("NO_VALID_PLAN_TOPIC_FOUND")

    return out


def project_points_to_polyline(points_xy: np.ndarray, ref_xy: np.ndarray) -> Dict[str, np.ndarray]:
    n_pts = points_xy.shape[0]
    n_ref = ref_xy.shape[0]

    if n_pts == 0 or n_ref < 2:
        return {"dist": np.array([], dtype=float), "s_hat": np.array([], dtype=float), "heading_ref": np.array([], dtype=float)}

    ref_seg = ref_xy[1:] - ref_xy[:-1]
    ref_seg_len2 = np.sum(ref_seg * ref_seg, axis=1)
    ref_seg_len = np.sqrt(np.maximum(ref_seg_len2, 1e-12))
    ref_heading = np.arctan2(ref_seg[:, 1], ref_seg[:, 0])
    s_ref = cumulative_arc_length(ref_xy)

    dist_out = np.zeros((n_pts,), dtype=float)
    s_out = np.zeros((n_pts,), dtype=float)
    heading_out = np.zeros((n_pts,), dtype=float)

    for i in range(n_pts):
        p = points_xy[i]
        best_dist2 = float("inf")
        best_s = 0.0
        best_heading = 0.0

        for j in range(n_ref - 1):
            a = ref_xy[j]
            b = ref_xy[j + 1]
            ab = b - a
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

        dist_out[i] = math.sqrt(best_dist2)
        s_out[i] = best_s
        heading_out[i] = best_heading

    return {"dist": dist_out, "s_hat": s_out, "heading_ref": heading_out}


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


def compute_speed_and_smoothness(
    exec_t_s: np.ndarray,
    exec_v: np.ndarray,
    eval_end_idx: int,
    target_speed_mps: float,
    cmd: Optional[Dict[str, Any]],
    eval_end_time_s: float
) -> Dict[str, Optional[float]]:
    v_eval = exec_v[:eval_end_idx + 1]
    t_eval = exec_t_s[:eval_end_idx + 1]

    rmse_v = float(np.sqrt(np.mean((v_eval - target_speed_mps) ** 2))) if v_eval.size else None

    if v_eval.size:
        v_smooth = moving_average_1d(v_eval, window=11)
        jx = derivative(t_eval, v_smooth, min_dt_s=1e-2)
        rms_jx = rms(jx)
    else:
        rms_jx = None

    rms_dotdelta = None

    rms_cmd_w = None
    if cmd is not None:
        mask = cmd["t_s"] <= eval_end_time_s + 1e-9
        wz = cmd["wz"][mask]
        if wz.size:
            rms_cmd_w = rms(wz)

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
    out["human_gt_alignment_mode"] = "variant_start_aligned"

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
    raw_xy = projected["xy"]
    if raw_xy.shape[0] < 2:
        out["human_gt_note"] = "HUMAN_GPS_TOO_SHORT"
        return out

    best = mod.choose_best_candidate(raw_xy, waypoint_ref["xy"], gt_ref["gt_mean_xy"], n_samples)
    aligned_xy = mod.align_trace_start_to_reference(best["xy"], gt_ref["gt_mean_xy"])
    metrics = mod.compute_curve_metrics_vs_reference(aligned_xy, gt_ref["gt_mean_xy"], n_samples)

    out["human_gt_best_variant"] = str(best.get("variant", ""))
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
    n_valid = sum(int(r["valid_for_tracking"]) for r in rows)

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

    def vals_valid(key: str) -> List[float]:
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

    topic_counts: Dict[str, int] = {}
    for r in rows:
        topic = str(r.get("plan_topic_used", ""))
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    primary_ref_summary = max(topic_counts.items(), key=lambda kv: kv[1])[0] if topic_counts else "plan_missing"

    human_variant_counts: Dict[str, int] = {}
    for r in rows:
        if int(r.get("valid_for_tracking", 0)) != 1:
            continue
        variant = str(r.get("human_gt_best_variant", "")).strip()
        if variant:
            human_variant_counts[variant] = human_variant_counts.get(variant, 0) + 1
    dominant_human_variant = max(human_variant_counts.items(), key=lambda kv: kv[1])[0] if human_variant_counts else "not_enabled"

    human_metric_runs = len(vals_valid("human_gt_rmse_y_m"))

    agg = {
        "scenario_id": rows[0]["scenario_id"],
        "speed_kmh": rows[0]["speed_kmh"],
        "planner_id": rows[0]["planner_id"],
        "controller_id": rows[0]["controller_id"],
        "method_label": f"{rows[0]['planner_id']}, {rows[0]['controller_id']}",

        "N_total": n_total,
        "N_success_geo": n_success,
        "N_valid_for_tracking": n_valid,
        "N_human_metric_runs": human_metric_runs,
        "SR_geo_text": f"{n_success}/{n_total}",
        "SR_geo_rate": (n_success / n_total) if n_total > 0 else None,

        "primary_tracking_reference": primary_ref_summary,
        "human_tracking_reference": "gt_mean_raw_gps_start_normalized" if human_metric_runs > 0 else "not_enabled",
        "human_best_variant_mode": dominant_human_variant,

        "PR_median_iqr": fmt_median_iqr(vals_all("progress_ratio")),
        "T_median_iqr": fmt_median_iqr(vals_valid("mission_time_s")),
        "RMSE_y_primary_median_iqr": fmt_median_iqr(vals_valid("rmse_y_primary_m")),
        "P95_y_primary_median_iqr": fmt_median_iqr(vals_valid("p95_y_primary_m")),
        "RMSE_psi_primary_median_iqr": fmt_median_iqr(vals_valid("rmse_psi_primary_rad")),
        "RMSE_v_median_iqr": fmt_median_iqr(vals_valid("rmse_v_mps")),
        "RMS_jx_median_iqr": fmt_median_iqr(vals_valid("rms_jx_mps3")),
        "RMS_dotdelta_median_iqr": fmt_median_iqr(vals_valid("rms_dotdelta_radps")),
        "RMS_cmd_w_median_iqr": fmt_median_iqr(vals_valid("rms_cmd_w_radps")),

        "Exec_Human_RMSE_y_median_iqr": fmt_median_iqr(vals_valid("human_gt_rmse_y_m")),
        "Exec_Human_P95_y_median_iqr": fmt_median_iqr(vals_valid("human_gt_p95_y_m")),
        "Exec_Human_MAX_y_median_iqr": fmt_median_iqr(vals_valid("human_gt_max_y_m")),
        "Exec_Human_PR_median_iqr": fmt_median_iqr(vals_valid("human_gt_progress")),
        "Exec_Human_Pointwise_RMSE_median_iqr": fmt_median_iqr(vals_valid("human_gt_pointwise_rmse_m")),
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

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = (Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (runs_dir / "Output_metrics_assessment").resolve())
    gt_csv = Path(args.gt_csv).expanduser().resolve() if args.gt_csv else None

    if not runs_dir.exists():
        print(f"ERROR: runs dir does not exist: {runs_dir}", file=sys.stderr)
        return 2

    if gt_csv is not None:
        print(f"[INFO] Legacy --gt-csv received and ignored for metrics: {gt_csv}")

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

    print(f"[INFO] Found {len(bag_dirs)} run bags in {runs_dir}")
    print(f"[INFO] Metrics output dir: {out_dir}")
    if human_ctx is not None:
        print("[INFO] Human-likeness GT raw GPS enabled")
        print(f"[INFO] Human GT GPS glob: {human_ctx['gt_gps_csv_glob']}")
        print(f"[INFO] Human waypoint file: {human_ctx['waypoint_file']}")
        print(f"[INFO] Human common origin lat0_deg: {human_ctx['waypoint_ref']['lat0_deg']:.10f}")
        print(f"[INFO] Human common origin lon0_deg: {human_ctx['waypoint_ref']['lon0_deg']:.10f}")
        print("[INFO] Human metrics will be computed only for valid_for_tracking runs")
    else:
        print("[INFO] Human-likeness GT raw GPS: disabled")

    print(f"[INFO] success_PR_threshold = {args.success_pr_threshold:.2f}")
    print(f"[INFO] tracking_PR_threshold = {args.tracking_pr_threshold:.2f}")

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

                primary_reference="plan_missing",
                plan_topic_used="",
                executed_topic_used=args.executed_topic,
                cmd_topic_used="",

                has_plan=0,
                has_human_gt_gps=int(human_ctx is not None),

                success_geo=0,
                progress_ratio=0.0,
                valid_for_tracking=0,
                eval_cutoff_reason="bag_read_failed",

                mission_time_s=float("nan"),
                raw_run_duration_s=float("nan"),

                final_dist_to_goal_m=float("nan"),
                final_yaw_err_to_goal_rad=float("nan"),

                rmse_y_primary_m=None,
                p95_y_primary_m=None,
                rmse_psi_primary_rad=None,

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

        has_plan = int(plan_data is not None and plan_data["xy"].shape[0] >= 2)

        if has_plan:
            primary_ref_name = "plan"
            primary_ref_xy = plan_data["xy"]
            primary_goal_xy = plan_data["goal_xy"]
            primary_goal_yaw = plan_data["goal_yaw"]

            primary_metrics = compute_tracking_against_reference(
                exec_t_s=exec_data["t_s"],
                exec_xy=exec_data["xy"],
                exec_yaw=exec_data["yaw"],
                ref_xy=primary_ref_xy,
                ref_goal_xy=primary_goal_xy,
                ref_goal_yaw=primary_goal_yaw,
                success_pr_threshold=args.success_pr_threshold,
                tracking_pr_threshold=args.tracking_pr_threshold,
            )
            eval_end_idx = int(primary_metrics["eval_end_idx"])
        else:
            primary_ref_name = "plan_missing"
            notes.append("PRIMARY_METRICS_SKIPPED_NO_VALID_PLAN")
            eval_end_idx = int(exec_data["t_s"].size - 1)
            primary_metrics = {
                "progress_ratio": 0.0,
                "success_geo": 0,
                "valid_for_tracking": 0,
                "eval_cutoff_reason": "plan_missing",
                "mission_time_s": float(exec_data["t_s"][-1] - exec_data["t_s"][0]) if exec_data["t_s"].size else float("nan"),
                "raw_run_duration_s": float(exec_data["t_s"][-1] - exec_data["t_s"][0]) if exec_data["t_s"].size else float("nan"),
                "final_dist_to_goal_m": float("nan"),
                "final_yaw_err_to_goal_rad": float("nan"),
                "rmse_y_m": None,
                "p95_y_m": None,
                "rmse_psi_rad": None,
                "eval_end_idx": eval_end_idx,
            }

        smooth = compute_speed_and_smoothness(
            exec_t_s=exec_data["t_s"],
            exec_v=exec_data["v"],
            eval_end_idx=eval_end_idx,
            target_speed_mps=target_speed_mps,
            cmd=cmd_data,
            eval_end_time_s=exec_data["t_s"][eval_end_idx],
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
            "human_gt_note": "HUMAN_GPS_SKIPPED_NOT_VALID_FOR_TRACKING",
        }

        if human_ctx is not None and int(primary_metrics.get("valid_for_tracking", 0)) == 1:
            cutoff_t_ns = int(exec_data["t_ns"][eval_end_idx])
            try:
                aux_human = compute_humanlikeness_for_bag(
                    bag_dir=bag_dir,
                    gps_topic=args.gps_topic,
                    human_ctx=human_ctx,
                    cutoff_t_ns=cutoff_t_ns,
                )
            except Exception as e:
                aux_human["human_gt_note"] = f"HUMAN_GPS_COMPARE_FAILED: {e}"

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

            primary_reference=primary_ref_name,
            plan_topic_used=bag["plan_topic_used"] if bag["plan_topic_used"] else "",
            executed_topic_used=bag["executed_topic_used"],
            cmd_topic_used=bag["cmd_topic_used"] if bag["cmd_topic_used"] else "",

            has_plan=has_plan,
            has_human_gt_gps=int(aux_human.get("has_human_gt_gps", 0)),

            success_geo=int(primary_metrics["success_geo"]),
            progress_ratio=float(primary_metrics["progress_ratio"]),
            valid_for_tracking=int(primary_metrics["valid_for_tracking"]),
            eval_cutoff_reason=str(primary_metrics["eval_cutoff_reason"]),

            mission_time_s=float(primary_metrics["mission_time_s"]),
            raw_run_duration_s=float(primary_metrics["raw_run_duration_s"]),

            final_dist_to_goal_m=float(primary_metrics["final_dist_to_goal_m"]),
            final_yaw_err_to_goal_rad=float(primary_metrics["final_yaw_err_to_goal_rad"]),

            rmse_y_primary_m=primary_metrics["rmse_y_m"],
            p95_y_primary_m=primary_metrics["p95_y_m"],
            rmse_psi_primary_rad=primary_metrics["rmse_psi_rad"],

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
    print(f"Primary reference: {agg['primary_tracking_reference']}")
    print(f"PR: {agg['PR_median_iqr']}")
    print(f"T: {agg['T_median_iqr']}")
    print(f"RMSE_y: {agg['RMSE_y_primary_median_iqr']}")
    print(f"P95_y: {agg['P95_y_primary_median_iqr']}")
    print(f"RMSE_psi: {agg['RMSE_psi_primary_median_iqr']}")
    print(f"RMSE_v: {agg['RMSE_v_median_iqr']}")
    print(f"RMS_jx: {agg['RMS_jx_median_iqr']}")
    print(f"Human-like RMSE_y: {agg['Exec_Human_RMSE_y_median_iqr']}")
    print(f"Human-like P95_y: {agg['Exec_Human_P95_y_median_iqr']}")
    print(f"Human-like Max_y: {agg['Exec_Human_MAX_y_median_iqr']}")
    print(f"Human-like PR: {agg['Exec_Human_PR_median_iqr']}")
    print(f"Human-like variant mode: {agg['human_best_variant_mode']}")
    print(f"Human metric runs (valid only): {agg['N_human_metric_runs']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())