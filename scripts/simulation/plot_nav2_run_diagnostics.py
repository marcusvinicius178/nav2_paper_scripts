#!/usr/bin/env python3
"""
Improved diagnostic plotter for one ROS 2 Nav2 run bag.

What this script now generates
------------------------------
1) local_overlay_plan_vs_exec.png
   - Main visual for simulation analysis
   - Planner path(s) vs executed trajectory(ies) in local XY
   - NO GT mixed here by default

2) local_overlay_plan_vs_exec_vs_gt_rigid.png
   - Optional auxiliary visual
   - Mean GT CSV is rigidly aligned to the chosen planner path
   - Useful only as qualitative reference, not as official metric basis

3) gps_fix_latlon_bag.png
   - Raw GPS lat/lon from the simulation run bag

4) gps_fix_local_xy_bag.png
   - Raw GPS projected to local XY using the first valid fix as local origin

5) gps_fix_latlon_bag_vs_gtbag.png
   - Optional
   - Raw GPS lat/lon from simulation bag vs raw GPS lat/lon from one GT bag
   - This is the correct way to compare raw GPS visually without mixing frames

6) topic_summary.csv
7) topic_summary.txt

Important interpretation
------------------------
- Use the local XY plots (planner vs executed) for planner/controller behavior.
- Use the raw GPS plots separately for GT-related visual inspection.
- Do NOT treat the rigid-aligned GT plot as a formal metric comparison.
  It is only a visual helper.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception:
    print("ERROR: Could not import ROS 2 Python APIs.", file=sys.stderr)
    print("Run: source /opt/ros/jazzy/setup.bash", file=sys.stderr)
    raise


DEFAULT_EXEC_TOPICS = ["/odometry/global", "/odom", "/odometry/local"]
DEFAULT_PLAN_TOPICS = ["/plan", "/plan_smoothed", "/transformed_global_plan"]

EARTH_RADIUS_M = 6378137.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot diagnostics for one or multiple Nav2 run bags.")

    parser.add_argument(
        "--bag-dir",
        default="",
        help="Single bag folder (directory containing metadata.yaml). Use together with --out-dir.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for single-bag mode.",
    )
    parser.add_argument(
        "--batch-parent-dir",
        default="",
        help="Parent directory that contains multiple bag folders, for example Navfn_mppi_1 ... Navfn_mppi_10.",
    )
    parser.add_argument(
        "--batch-glob",
        default="Navfn_mppi_*",
        help="Glob used inside --batch-parent-dir to find bag folders.",
    )
    parser.add_argument(
        "--batch-out-subdir",
        default="diagnostics",
        help="Subfolder name created inside each bag directory in batch mode.",
    )

    parser.add_argument("--gt-csv", default="", help="Optional mean GT CSV in local XY")
    parser.add_argument(
        "--gt-gps-bag-dir",
        default="",
        help="Optional GT bag folder for raw GPS lat/lon visual comparison (uses /gps/fix)",
    )
    parser.add_argument(
        "--exec-topics",
        default=",".join(DEFAULT_EXEC_TOPICS),
        help="Comma-separated candidate executed odometry topics",
    )
    parser.add_argument(
        "--plan-topics",
        default=",".join(DEFAULT_PLAN_TOPICS),
        help="Comma-separated candidate plan topics",
    )
    parser.add_argument("--gps-topic", default="/gps/fix", help="GPS topic to extract raw NavSatFix")
    parser.add_argument("--rigid-align-samples", type=int, default=300)

    args = parser.parse_args()

    single_mode = bool(args.bag_dir.strip())
    batch_mode = bool(args.batch_parent_dir.strip())

    if single_mode and batch_mode:
        parser.error("Use either --bag-dir (single mode) or --batch-parent-dir (batch mode), not both.")

    if not single_mode and not batch_mode:
        parser.error("You must provide either --bag-dir for one bag or --batch-parent-dir for batch mode.")

    if single_mode and not args.out_dir.strip():
        parser.error("In single-bag mode, --out-dir is required.")

    return args


def resolve_bag_uri_for_mcap(bag_dir: Path) -> Path:
    mcap_files = sorted([p for p in bag_dir.glob("*.mcap") if p.is_file()])
    if not mcap_files:
        return bag_dir

    for p in mcap_files:
        if p.name.endswith("_0.mcap"):
            return p
    return mcap_files[0]


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def remove_consecutive_duplicates_xy(
    xy: np.ndarray,
    t_ns: Optional[np.ndarray] = None,
    yaw: Optional[np.ndarray] = None,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if xy.shape[0] <= 1:
        return xy, t_ns, yaw

    keep = [0]
    for i in range(1, xy.shape[0]):
        if np.linalg.norm(xy[i] - xy[i - 1]) > eps:
            keep.append(i)

    k = np.asarray(keep, dtype=int)
    t_out = t_ns[k] if t_ns is not None else None
    y_out = yaw[k] if yaw is not None else None
    return xy[k], t_out, y_out


def remove_consecutive_duplicates_latlon(
    lat: np.ndarray,
    lon: np.ndarray,
    t_ns: Optional[np.ndarray] = None,
    eps_deg: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if lat.size <= 1:
        return lat, lon, t_ns

    keep = [0]
    for i in range(1, lat.size):
        if abs(lat[i] - lat[i - 1]) > eps_deg or abs(lon[i] - lon[i - 1]) > eps_deg:
            keep.append(i)

    k = np.asarray(keep, dtype=int)
    t_out = t_ns[k] if t_ns is not None else None
    return lat[k], lon[k], t_out


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


def resample_by_arclength(xy: np.ndarray, n_samples: int) -> np.ndarray:
    if xy.shape[0] == 0:
        return xy.copy()
    if xy.shape[0] == 1:
        return np.repeat(xy, n_samples, axis=0)

    s = cumulative_arc_length(xy)
    total = s[-1]
    if total <= 1e-12:
        return np.repeat(xy[:1], n_samples, axis=0)

    s_new = np.linspace(0.0, total, n_samples)
    xr = np.interp(s_new, s, xy[:, 0])
    yr = np.interp(s_new, s, xy[:, 1])
    return np.column_stack((xr, yr))


def estimate_rigid_transform(ref_xy: np.ndarray, mov_xy: np.ndarray, n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    if ref_xy.shape[0] < 2 or mov_xy.shape[0] < 2:
        return np.eye(2, dtype=float), np.zeros((2,), dtype=float)

    a = resample_by_arclength(ref_xy, n_samples)
    b = resample_by_arclength(mov_xy, n_samples)

    d_direct = np.mean(np.linalg.norm(a - b, axis=1))
    d_rev = np.mean(np.linalg.norm(a - b[::-1], axis=1))
    if d_rev < d_direct:
        b = b[::-1].copy()

    a_c = a.mean(axis=0)
    b_c = b.mean(axis=0)

    X = a - a_c
    Y = b - b_c

    H = Y.T @ X
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = a_c - (b_c @ R.T)
    return R, t


def apply_rigid_transform(xy: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if xy.size == 0:
        return xy.copy()
    return (xy @ R.T) + t


def latlon_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)

    valid = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float), float("nan"), float("nan")

    idx0 = int(np.flatnonzero(valid)[0])
    lat0 = float(lat[idx0])
    lon0 = float(lon[idx0])

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    x = EARTH_RADIUS_M * dlon * math.cos(lat0_rad)
    y = EARTH_RADIUS_M * dlat
    return x, y, lat0, lon0


def extract_odom_sample(msg: Any) -> Optional[Tuple[float, float, float]]:
    try:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        x = float(p.x)
        y = float(p.y)
        yaw = yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w))
        return x, y, yaw
    except Exception:
        return None


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
    xy, _, _ = remove_consecutive_duplicates_xy(xy)
    if xy.shape[0] < 2:
        return None
    return xy


def extract_navsatfix_latlon(msg: Any) -> Optional[Tuple[float, float]]:
    try:
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        if not math.isfinite(lat) or not math.isfinite(lon):
            return None
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            return None
        return lat, lon
    except Exception:
        return None


def load_gt_csv(gt_csv: Path) -> Optional[Dict[str, Any]]:
    if not gt_csv.exists():
        return None

    rows = []
    with gt_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return None

    x = np.asarray([float(r["x_m"]) for r in rows], dtype=float)
    y = np.asarray([float(r["y_m"]) for r in rows], dtype=float)
    t_ns = np.asarray([int(float(r["t_ns"])) for r in rows], dtype=np.int64)

    xy = np.column_stack((x, y))
    xy, t_ns, _ = remove_consecutive_duplicates_xy(xy, t_ns=t_ns)

    return {
        "xy": xy,
        "t_ns": t_ns,
        "heading": heading_from_path(xy),
        "length_m": path_length(xy),
        "name": gt_csv.name,
    }


def project_points_to_polyline(points_xy: np.ndarray, ref_xy: np.ndarray) -> Dict[str, np.ndarray]:
    n_pts = points_xy.shape[0]
    n_ref = ref_xy.shape[0]

    if n_pts == 0 or n_ref < 2:
        return {
            "dist": np.array([], dtype=float),
            "s_hat": np.array([], dtype=float),
            "heading_ref": np.array([], dtype=float),
        }

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

    return {
        "dist": dist_out,
        "s_hat": s_out,
        "heading_ref": heading_out,
    }


def read_bag_topics(
    bag_dir: Path,
    exec_topics: List[str],
    plan_topics: List[str],
    gps_topic: str,
) -> Dict[str, Any]:
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

    requested_topics: List[str] = []
    for t in exec_topics + plan_topics + [gps_topic]:
        if t in topic_type_map and t not in requested_topics:
            requested_topics.append(t)

    try:
        if hasattr(rosbag2_py, "StorageFilter"):
            reader.set_filter(rosbag2_py.StorageFilter(topics=requested_topics))
    except Exception:
        pass

    msg_type_cache: Dict[str, Any] = {}
    for t in requested_topics:
        msg_type_cache[t] = get_message(topic_type_map[t])

    exec_buffers: Dict[str, Dict[str, List[float]]] = {}
    for t in exec_topics:
        if t in topic_type_map:
            exec_buffers[t] = {"t_ns": [], "x": [], "y": [], "yaw": []}

    best_plan_per_topic: Dict[str, Dict[str, Any]] = {}
    for t in plan_topics:
        if t in topic_type_map:
            best_plan_per_topic[t] = {"xy": None, "n_pts": -1, "t_ns": -1}

    gps_t_ns: List[int] = []
    gps_lat: List[float] = []
    gps_lon: List[float] = []

    while reader.has_next():
        topic, rawdata, t_ns = reader.read_next()
        if topic not in msg_type_cache:
            continue

        try:
            msg = deserialize_message(rawdata, msg_type_cache[topic])
        except Exception:
            continue

        if topic in exec_buffers:
            sample = extract_odom_sample(msg)
            if sample is None:
                continue
            x, y, yaw = sample
            exec_buffers[topic]["t_ns"].append(int(t_ns))
            exec_buffers[topic]["x"].append(x)
            exec_buffers[topic]["y"].append(y)
            exec_buffers[topic]["yaw"].append(yaw)

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

        elif topic == gps_topic:
            ll = extract_navsatfix_latlon(msg)
            if ll is None:
                continue
            lat, lon = ll
            gps_t_ns.append(int(t_ns))
            gps_lat.append(lat)
            gps_lon.append(lon)

    exec_data: Dict[str, Dict[str, Any]] = {}
    for topic, buf in exec_buffers.items():
        if len(buf["t_ns"]) < 2:
            continue

        t_ns = np.asarray(buf["t_ns"], dtype=np.int64)
        x = np.asarray(buf["x"], dtype=float)
        y = np.asarray(buf["y"], dtype=float)
        yaw = np.asarray(buf["yaw"], dtype=float)

        idx = np.argsort(t_ns, kind="stable")
        t_ns = t_ns[idx]
        x = x[idx]
        y = y[idx]
        yaw = yaw[idx]

        xy = np.column_stack((x, y))
        xy, t_ns, yaw = remove_consecutive_duplicates_xy(xy, t_ns=t_ns, yaw=yaw)

        if xy.shape[0] < 2:
            continue

        exec_data[topic] = {
            "xy": xy,
            "t_ns": t_ns,
            "yaw": yaw,
            "heading_path": heading_from_path(xy),
            "length_m": path_length(xy),
            "net_displacement_m": float(np.linalg.norm(xy[-1] - xy[0])),
            "points": int(xy.shape[0]),
        }

    plan_data: Dict[str, Dict[str, Any]] = {}
    for topic, data in best_plan_per_topic.items():
        xy = data["xy"]
        if xy is None or xy.shape[0] < 2:
            continue

        plan_data[topic] = {
            "xy": xy,
            "heading": heading_from_path(xy),
            "length_m": path_length(xy),
            "points": int(xy.shape[0]),
        }

    gps_data: Optional[Dict[str, Any]] = None
    if len(gps_t_ns) >= 2:
        t_ns = np.asarray(gps_t_ns, dtype=np.int64)
        lat = np.asarray(gps_lat, dtype=float)
        lon = np.asarray(gps_lon, dtype=float)

        idx = np.argsort(t_ns, kind="stable")
        t_ns = t_ns[idx]
        lat = lat[idx]
        lon = lon[idx]

        lat, lon, t_ns = remove_consecutive_duplicates_latlon(lat, lon, t_ns=t_ns)

        xg, yg, lat0, lon0 = latlon_to_local_xy_m(lat, lon)
        if xg.size >= 2:
            gps_xy = np.column_stack((xg, yg))
            gps_xy, _, _ = remove_consecutive_duplicates_xy(gps_xy)

            gps_data = {
                "t_ns": t_ns,
                "lat": lat,
                "lon": lon,
                "xy": gps_xy,
                "lat0": lat0,
                "lon0": lon0,
                "length_m_local_xy": path_length(gps_xy),
                "points": int(lat.size),
            }

    return {
        "topic_type_map": topic_type_map,
        "exec_data": exec_data,
        "plan_data": plan_data,
        "gps_data": gps_data,
    }


def choose_best_plan(plan_data: Dict[str, Dict[str, Any]], preferred_order: List[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
    for topic in preferred_order:
        if topic in plan_data:
            return topic, plan_data[topic]
    return "", None


def choose_best_exec(exec_data: Dict[str, Dict[str, Any]], preferred_order: List[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
    for topic in preferred_order:
        if topic in exec_data:
            return topic, exec_data[topic]
    return "", None


def compute_summary_rows(
    exec_data: Dict[str, Dict[str, Any]],
    chosen_plan_topic: str,
    chosen_plan: Optional[Dict[str, Any]],
    gt_data: Optional[Dict[str, Any]],
    chosen_exec_topic: str,
    chosen_exec: Optional[Dict[str, Any]],
    gps_data: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for topic, data in exec_data.items():
        row = {
            "kind": "executed",
            "topic": topic,
            "points": data["points"],
            "length_m": f"{data['length_m']:.6f}",
            "net_displacement_m": f"{data['net_displacement_m']:.6f}",
            "pr_raw_vs_plan": "",
            "pr_start_aligned_vs_plan": "",
            "rmse_raw_vs_plan_m": "",
            "rmse_start_aligned_vs_plan_m": "",
        }

        xy = data["xy"]

        if chosen_plan is not None:
            plan_xy = chosen_plan["xy"]
            plan_len = max(chosen_plan["length_m"], 1e-9)

            proj_raw = project_points_to_polyline(xy, plan_xy)
            pr_raw = float(np.max(proj_raw["s_hat"]) / plan_len) if proj_raw["s_hat"].size else 0.0
            rmse_raw = float(np.sqrt(np.mean(proj_raw["dist"] ** 2))) if proj_raw["dist"].size else float("nan")

            xy_aligned = xy + (plan_xy[0] - xy[0])
            proj_aligned = project_points_to_polyline(xy_aligned, plan_xy)
            pr_aligned = float(np.max(proj_aligned["s_hat"]) / plan_len) if proj_aligned["s_hat"].size else 0.0
            rmse_aligned = float(np.sqrt(np.mean(proj_aligned["dist"] ** 2))) if proj_aligned["dist"].size else float("nan")

            row["pr_raw_vs_plan"] = f"{pr_raw:.6f}"
            row["pr_start_aligned_vs_plan"] = f"{pr_aligned:.6f}"
            row["rmse_raw_vs_plan_m"] = f"{rmse_raw:.6f}"
            row["rmse_start_aligned_vs_plan_m"] = f"{rmse_aligned:.6f}"

        rows.append(row)

    if chosen_plan is not None:
        rows.append({
            "kind": "plan",
            "topic": chosen_plan_topic,
            "points": chosen_plan["points"],
            "length_m": f"{chosen_plan['length_m']:.6f}",
            "net_displacement_m": f"{float(np.linalg.norm(chosen_plan['xy'][-1] - chosen_plan['xy'][0])):.6f}",
            "pr_raw_vs_plan": "1.000000",
            "pr_start_aligned_vs_plan": "1.000000",
            "rmse_raw_vs_plan_m": "0.000000",
            "rmse_start_aligned_vs_plan_m": "0.000000",
        })

    if gt_data is not None:
        rows.append({
            "kind": "gt_xy",
            "topic": gt_data["name"],
            "points": int(gt_data["xy"].shape[0]),
            "length_m": f"{gt_data['length_m']:.6f}",
            "net_displacement_m": f"{float(np.linalg.norm(gt_data['xy'][-1] - gt_data['xy'][0])):.6f}",
            "pr_raw_vs_plan": "",
            "pr_start_aligned_vs_plan": "",
            "rmse_raw_vs_plan_m": "",
            "rmse_start_aligned_vs_plan_m": "",
        })

    if gps_data is not None:
        rows.append({
            "kind": "gps_fix",
            "topic": "/gps/fix",
            "points": gps_data["points"],
            "length_m": f"{gps_data['length_m_local_xy']:.6f}",
            "net_displacement_m": f"{float(np.linalg.norm(gps_data['xy'][-1] - gps_data['xy'][0])):.6f}",
            "pr_raw_vs_plan": "",
            "pr_start_aligned_vs_plan": "",
            "rmse_raw_vs_plan_m": "",
            "rmse_start_aligned_vs_plan_m": "",
        })

    if chosen_exec is not None:
        rows.append({
            "kind": "chosen_exec",
            "topic": chosen_exec_topic,
            "points": chosen_exec["points"],
            "length_m": f"{chosen_exec['length_m']:.6f}",
            "net_displacement_m": f"{chosen_exec['net_displacement_m']:.6f}",
            "pr_raw_vs_plan": "",
            "pr_start_aligned_vs_plan": "",
            "rmse_raw_vs_plan_m": "",
            "rmse_start_aligned_vs_plan_m": "",
        })

    return rows


def save_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_summary_txt(
    path: Path,
    bag_dir: Path,
    exec_data: Dict[str, Dict[str, Any]],
    plan_data: Dict[str, Dict[str, Any]],
    chosen_plan_topic: str,
    chosen_exec_topic: str,
    gt_data: Optional[Dict[str, Any]],
    gps_data: Optional[Dict[str, Any]],
    gt_rigid_meta: Optional[Dict[str, float]],
    rows: List[Dict[str, Any]],
    gt_gps_bag_dir: Optional[Path],
    gt_gps_data: Optional[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("Nav2 run diagnostics summary\n")
        f.write(f"Bag dir: {bag_dir}\n\n")

        f.write("Available executed topics with valid trajectories:\n")
        if exec_data:
            for topic, data in exec_data.items():
                f.write(
                    f"  - {topic}: points={data['points']}, "
                    f"length_m={data['length_m']:.6f}, "
                    f"net_displacement_m={data['net_displacement_m']:.6f}\n"
                )
        else:
            f.write("  - none\n")

        f.write("\nAvailable plan topics with valid paths:\n")
        if plan_data:
            for topic, data in plan_data.items():
                f.write(f"  - {topic}: points={data['points']}, length_m={data['length_m']:.6f}\n")
        else:
            f.write("  - none\n")

        f.write(f"\nChosen executed topic: {chosen_exec_topic if chosen_exec_topic else 'none'}\n")
        f.write(f"Chosen plan topic: {chosen_plan_topic if chosen_plan_topic else 'none'}\n")

        if gt_data is not None:
            f.write(
                f"GT mean XY CSV: {gt_data['name']}, "
                f"points={gt_data['xy'].shape[0]}, "
                f"length_m={gt_data['length_m']:.6f}\n"
            )
        else:
            f.write("GT mean XY CSV: none\n")

        if gps_data is not None:
            f.write(
                f"Bag GPS /gps/fix: points={gps_data['points']}, "
                f"local_xy_length_m={gps_data['length_m_local_xy']:.6f}, "
                f"origin_lat={gps_data['lat0']:.8f}, origin_lon={gps_data['lon0']:.8f}\n"
            )
        else:
            f.write("Bag GPS /gps/fix: none\n")

        if gt_gps_bag_dir is not None:
            f.write(f"GT GPS bag dir: {gt_gps_bag_dir}\n")
            if gt_gps_data is not None:
                f.write(
                    f"GT GPS /gps/fix: points={gt_gps_data['points']}, "
                    f"local_xy_length_m={gt_gps_data['length_m_local_xy']:.6f}\n"
                )
            else:
                f.write("GT GPS /gps/fix: none\n")

        if gt_rigid_meta is not None:
            f.write(
                "\nRigid alignment used only for qualitative GT overlay:\n"
                f"  rotation_deg={gt_rigid_meta['rotation_deg']:.6f}\n"
                f"  translation_x_m={gt_rigid_meta['tx']:.6f}\n"
                f"  translation_y_m={gt_rigid_meta['ty']:.6f}\n"
            )

        f.write("\nComputed diagnostic metrics:\n")
        for r in rows:
            if r["kind"] != "executed":
                continue
            f.write(
                f"  - {r['topic']}: "
                f"pr_raw_vs_plan={r['pr_raw_vs_plan'] or 'NA'}, "
                f"pr_start_aligned_vs_plan={r['pr_start_aligned_vs_plan'] or 'NA'}, "
                f"rmse_raw_vs_plan_m={r['rmse_raw_vs_plan_m'] or 'NA'}, "
                f"rmse_start_aligned_vs_plan_m={r['rmse_start_aligned_vs_plan_m'] or 'NA'}\n"
            )


def set_equal_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_local_overlay_plan_vs_exec(
    path: Path,
    exec_data: Dict[str, Dict[str, Any]],
    plan_data: Dict[str, Dict[str, Any]],
    chosen_plan_topic: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for topic, data in plan_data.items():
        xy = data["xy"]
        label = f"Plan {topic}"
        if topic == chosen_plan_topic:
            label += " [chosen]"
        ax.plot(xy[:, 0], xy[:, 1], linestyle="--", label=label)

    for topic, data in exec_data.items():
        xy = data["xy"]
        ax.plot(xy[:, 0], xy[:, 1], label=f"Exec {topic}")
        ax.scatter([xy[0, 0]], [xy[0, 1]], marker="o")
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], marker="x")

    ax.set_title("Local XY, planner vs executed")
    set_equal_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_local_overlay_with_gt_rigid(
    path: Path,
    exec_data: Dict[str, Dict[str, Any]],
    chosen_exec_topic: str,
    chosen_exec: Optional[Dict[str, Any]],
    chosen_plan_topic: str,
    chosen_plan: Optional[Dict[str, Any]],
    gt_data: Optional[Dict[str, Any]],
    rigid_align_samples: int,
) -> Optional[Dict[str, float]]:
    fig, ax = plt.subplots(figsize=(10, 8))

    gt_meta = None

    if chosen_plan is not None:
        xy = chosen_plan["xy"]
        ax.plot(xy[:, 0], xy[:, 1], linestyle="--", label="Plan")

    if chosen_exec is not None:
        xy = chosen_exec["xy"]
        ax.plot(xy[:, 0], xy[:, 1], label="Exec")

    if gt_data is not None and chosen_plan is not None:
        gt_xy = gt_data["xy"]
        plan_xy = chosen_plan["xy"]

        R, t = estimate_rigid_transform(plan_xy, gt_xy, n_samples=rigid_align_samples)
        gt_xy_aligned = apply_rigid_transform(gt_xy, R, t)

        rot_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        gt_meta = {
            "rotation_deg": float(rot_deg),
            "tx": float(t[0]),
            "ty": float(t[1]),
        }

        ax.plot(gt_xy_aligned[:, 0], gt_xy_aligned[:, 1], linewidth=2, label="GT mean")

    ax.set_title("Local XY, plan vs executed vs GT")
    set_equal_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return gt_meta


def plot_gps_fix_latlon(
    path: Path,
    gps_data: Optional[Dict[str, Any]],
    title: str,
    label_prefix: str,
    gt_gps_data: Optional[Dict[str, Any]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    has_any = False

    if gps_data is not None:
        ax.plot(gps_data["lon"], gps_data["lat"], label=f"{label_prefix} /gps/fix")
        ax.scatter([gps_data["lon"][0]], [gps_data["lat"][0]], marker="o")
        ax.scatter([gps_data["lon"][-1]], [gps_data["lat"][-1]], marker="x")
        has_any = True

    if gt_gps_data is not None:
        ax.plot(gt_gps_data["lon"], gt_gps_data["lat"], label="GT bag /gps/fix")
        ax.scatter([gt_gps_data["lon"][0]], [gt_gps_data["lat"][0]], marker="o")
        ax.scatter([gt_gps_data["lon"][-1]], [gt_gps_data["lat"][-1]], marker="x")
        has_any = True

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    if has_any:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gps_fix_local_xy(
    path: Path,
    gps_data: Optional[Dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    has_any = False
    if gps_data is not None and gps_data["xy"].shape[0] >= 2:
        xy = gps_data["xy"]
        ax.plot(xy[:, 0], xy[:, 1], label="Bag /gps/fix projected local XY")
        ax.scatter([xy[0, 0]], [xy[0, 1]], marker="o")
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], marker="x")
        has_any = True

    ax.set_title("Bag raw GPS projected to local XY")
    set_equal_axes(ax)
    if has_any:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def natural_sort_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.name)
    key: List[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def find_batch_bag_dirs(parent_dir: Path, bag_glob: str) -> List[Path]:
    if not parent_dir.exists() or not parent_dir.is_dir():
        return []

    candidates = [p for p in parent_dir.glob(bag_glob) if p.is_dir()]
    bag_dirs: List[Path] = []

    for p in candidates:
        has_metadata = (p / "metadata.yaml").exists()
        has_mcap = any(x.is_file() for x in p.glob("*.mcap"))
        if has_metadata or has_mcap:
            bag_dirs.append(p)

    bag_dirs.sort(key=natural_sort_key)
    return bag_dirs


def process_one_bag(
    *,
    bag_dir: Path,
    out_dir: Path,
    gt_csv: Optional[Path],
    gt_gps_bag_dir: Optional[Path],
    exec_topics: List[str],
    plan_topics: List[str],
    gps_topic: str,
    rigid_align_samples: int,
    verbose: bool = True,
) -> int:
    if not bag_dir.exists():
        print(f"ERROR: bag dir does not exist: {bag_dir}", file=sys.stderr)
        return 2

    gt_data = load_gt_csv(gt_csv) if gt_csv else None

    bag = read_bag_topics(
        bag_dir=bag_dir,
        exec_topics=exec_topics,
        plan_topics=plan_topics,
        gps_topic=gps_topic,
    )

    exec_data = bag["exec_data"]
    plan_data = bag["plan_data"]
    gps_data = bag["gps_data"]

    chosen_plan_topic, chosen_plan = choose_best_plan(plan_data, plan_topics)
    chosen_exec_topic, chosen_exec = choose_best_exec(exec_data, exec_topics)

    gt_gps_data = None
    if gt_gps_bag_dir is not None and gt_gps_bag_dir.exists():
        gt_bag = read_bag_topics(
            bag_dir=gt_gps_bag_dir,
            exec_topics=[],
            plan_topics=[],
            gps_topic=gps_topic,
        )
        gt_gps_data = gt_bag["gps_data"]

    if not exec_data:
        print(f"WARNING: No executed trajectory could be extracted for {bag_dir}.")

    if not chosen_plan:
        print(f"WARNING: No valid planner path could be extracted for {bag_dir}.")

    rows = compute_summary_rows(
        exec_data=exec_data,
        chosen_plan_topic=chosen_plan_topic,
        chosen_plan=chosen_plan,
        gt_data=gt_data,
        chosen_exec_topic=chosen_exec_topic,
        chosen_exec=chosen_exec,
        gps_data=gps_data,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    local_main_png = out_dir / "local_overlay_plan_vs_exec.png"
    local_gt_png = out_dir / "local_overlay_plan_vs_exec_vs_gt_rigid.png"
    gps_latlon_png = out_dir / "gps_fix_latlon_bag.png"
    gps_local_png = out_dir / "gps_fix_local_xy_bag.png"
    gps_latlon_vs_gt_png = out_dir / "gps_fix_latlon_bag_vs_gtbag.png"

    csv_path = out_dir / "topic_summary.csv"
    txt_path = out_dir / "topic_summary.txt"

    plot_local_overlay_plan_vs_exec(
        path=local_main_png,
        exec_data=exec_data,
        plan_data=plan_data,
        chosen_plan_topic=chosen_plan_topic,
    )

    gt_rigid_meta = None
    if gt_data is not None and chosen_plan is not None:
        gt_rigid_meta = plot_local_overlay_with_gt_rigid(
            path=local_gt_png,
            exec_data=exec_data,
            chosen_exec_topic=chosen_exec_topic,
            chosen_exec=chosen_exec,
            chosen_plan_topic=chosen_plan_topic,
            chosen_plan=chosen_plan,
            gt_data=gt_data,
            rigid_align_samples=rigid_align_samples,
        )

    plot_gps_fix_latlon(
        path=gps_latlon_png,
        gps_data=gps_data,
        title="Bag raw GPS /gps/fix in lat/lon",
        label_prefix="Bag",
    )

    plot_gps_fix_local_xy(
        path=gps_local_png,
        gps_data=gps_data,
    )

    if gt_gps_data is not None:
        plot_gps_fix_latlon(
            path=gps_latlon_vs_gt_png,
            gps_data=gps_data,
            title="Bag raw GPS vs GT bag raw GPS in lat/lon",
            label_prefix="Bag",
            gt_gps_data=gt_gps_data,
        )

    save_summary_csv(csv_path, rows)
    save_summary_txt(
        path=txt_path,
        bag_dir=bag_dir,
        exec_data=exec_data,
        plan_data=plan_data,
        chosen_plan_topic=chosen_plan_topic,
        chosen_exec_topic=chosen_exec_topic,
        gt_data=gt_data,
        gps_data=gps_data,
        gt_rigid_meta=gt_rigid_meta,
        rows=rows,
        gt_gps_bag_dir=gt_gps_bag_dir,
        gt_gps_data=gt_gps_data,
    )

    if verbose:
        print(f"[OK] Diagnostic outputs generated for: {bag_dir.name}")
        print(f"  - {local_main_png}")
        if gt_data is not None and chosen_plan is not None:
            print(f"  - {local_gt_png}")
        print(f"  - {gps_latlon_png}")
        print(f"  - {gps_local_png}")
        if gt_gps_data is not None:
            print(f"  - {gps_latlon_vs_gt_png}")
        print(f"  - {csv_path}")
        print(f"  - {txt_path}")

        print("\n[QUICK SUMMARY]")
        print(f"Chosen executed topic: {chosen_exec_topic if chosen_exec_topic else 'none'}")
        print(f"Chosen plan topic: {chosen_plan_topic if chosen_plan_topic else 'none'}")

        if exec_data:
            for topic, data in exec_data.items():
                print(
                    f"Exec {topic}: "
                    f"points={data['points']}, "
                    f"length_m={data['length_m']:.6f}, "
                    f"net_displacement_m={data['net_displacement_m']:.6f}"
                )

        if plan_data:
            for topic, data in plan_data.items():
                print(
                    f"Plan {topic}: "
                    f"points={data['points']}, "
                    f"length_m={data['length_m']:.6f}"
                )

        if gps_data is not None:
            print(
                f"GPS /gps/fix: "
                f"points={gps_data['points']}, "
                f"local_xy_length_m={gps_data['length_m_local_xy']:.6f}"
            )

        if gt_data is not None:
            print(f"GT mean XY: points={gt_data['xy'].shape[0]}, length_m={gt_data['length_m']:.6f}")

        if gt_gps_data is not None:
            print(
                f"GT bag /gps/fix: "
                f"points={gt_gps_data['points']}, "
                f"local_xy_length_m={gt_gps_data['length_m_local_xy']:.6f}"
            )

    return 0


def run_batch_mode(
    *,
    parent_dir: Path,
    bag_glob: str,
    batch_out_subdir: str,
    gt_csv: Optional[Path],
    gt_gps_bag_dir: Optional[Path],
    exec_topics: List[str],
    plan_topics: List[str],
    gps_topic: str,
    rigid_align_samples: int,
) -> int:
    bag_dirs = find_batch_bag_dirs(parent_dir, bag_glob)

    if not bag_dirs:
        print(
            f"ERROR: No bag directories matching '{bag_glob}' were found inside {parent_dir}",
            file=sys.stderr,
        )
        return 2

    print(f"[BATCH] Found {len(bag_dirs)} bag(s) in {parent_dir}")

    ok_count = 0
    failed: List[Path] = []

    for idx, bag_dir in enumerate(bag_dirs, start=1):
        out_dir = bag_dir / batch_out_subdir
        print(f"\n[BATCH {idx}/{len(bag_dirs)}] Processing: {bag_dir.name}")

        rc = process_one_bag(
            bag_dir=bag_dir,
            out_dir=out_dir,
            gt_csv=gt_csv,
            gt_gps_bag_dir=gt_gps_bag_dir,
            exec_topics=exec_topics,
            plan_topics=plan_topics,
            gps_topic=gps_topic,
            rigid_align_samples=rigid_align_samples,
            verbose=True,
        )

        if rc == 0:
            ok_count += 1
        else:
            failed.append(bag_dir)

    print("\n[BATCH SUMMARY]")
    print(f"Successful: {ok_count}/{len(bag_dirs)}")

    if failed:
        print("Failed bags:")
        for p in failed:
            print(f"  - {p}")
        return 1

    return 0


def main() -> int:
    args = parse_args()

    exec_topics = [x.strip() for x in args.exec_topics.split(",") if x.strip()]
    plan_topics = [x.strip() for x in args.plan_topics.split(",") if x.strip()]

    gt_csv = Path(args.gt_csv).expanduser().resolve() if args.gt_csv else None
    gt_gps_bag_dir = Path(args.gt_gps_bag_dir).expanduser().resolve() if args.gt_gps_bag_dir else None

    if args.batch_parent_dir:
        parent_dir = Path(args.batch_parent_dir).expanduser().resolve()
        return run_batch_mode(
            parent_dir=parent_dir,
            bag_glob=args.batch_glob,
            batch_out_subdir=args.batch_out_subdir,
            gt_csv=gt_csv,
            gt_gps_bag_dir=gt_gps_bag_dir,
            exec_topics=exec_topics,
            plan_topics=plan_topics,
            gps_topic=args.gps_topic,
            rigid_align_samples=args.rigid_align_samples,
        )

    bag_dir = Path(args.bag_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    return process_one_bag(
        bag_dir=bag_dir,
        out_dir=out_dir,
        gt_csv=gt_csv,
        gt_gps_bag_dir=gt_gps_bag_dir,
        exec_topics=exec_topics,
        plan_topics=plan_topics,
        gps_topic=args.gps_topic,
        rigid_align_samples=args.rigid_align_samples,
        verbose=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
