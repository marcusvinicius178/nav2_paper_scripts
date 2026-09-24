#!/usr/bin/env python3
"""
Extract and analyze Ground Truth trajectories directly from ROS 2 rosbags (MCAP/rosbag2).

What it does
- Scans scenario folders under a root directory (e.g., NAV2_Paper_Scripts/waypoints)
- Finds bag folders (directories containing metadata.yaml)
- Reads trajectories from rosbag topics such as:
    /odometry/global, /odometry/gps, /odometry/local, /odom, /gps/fix, /gps/filtered
- Exports per-bag CSV trajectories (XY in meters for odom topics, and projected XY for GPS topics)
- Computes per-bag summaries
- Computes GT reproducibility (GT1 vs GT2) by scenario/speed using a chosen primary topic
- Generates mean GT references from the chosen primary topic

Important
- Source ROS before running:
    source /opt/ros/jazzy/setup.bash
- If your bags were recorded in another distro, rosbag2_py can still often read them,
  but message definitions/types must exist in the sourced environment.
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
    print("ERROR: ROS 2 Python APIs not available.", file=sys.stderr)
    print("Run this script after sourcing ROS 2, e.g.: source /opt/ros/jazzy/setup.bash", file=sys.stderr)
    raise


SCENARIO_ID_RE = re.compile(r"^(\d+)-")
SPEED_RE = re.compile(r"vel_(\d+)", re.IGNORECASE)

ODOM_CANDIDATES = [
    "/odometry/global",
    "/odometry/gps",
    "/odometry/local",
    "/odom",
]
GPS_CANDIDATES = [
    "/gps/fix",
    "/gps/filtered",
]


@dataclass
class BagInfo:
    scenario_id: int
    scenario_folder: str
    bag_dir: str
    bag_name: str
    speed_kmh: Optional[int]
    collection_label: str   # primeira|segunda|unknown
    collection_index: int   # 1|2|0


@dataclass
class TopicTrajectory:
    topic: str
    kind: str  # 'odom_xy' or 'gps_ll'
    count: int
    t0_ns: Optional[int]
    t1_ns: Optional[int]
    # Odom-like XY (meters) or GPS raw LL (degrees)
    x_or_lon: np.ndarray
    y_or_lat: np.ndarray
    t_ns: np.ndarray
    # Optional speed from odom twist
    speed_mps: Optional[np.ndarray] = None


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def cumulative_arc_length(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.array([], dtype=float)
    if xy.shape[0] == 1:
        return np.array([0.0], dtype=float)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def trajectory_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


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


def heading_from_path(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] < 2:
        return np.zeros((xy.shape[0],), dtype=float)
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    return np.arctan2(dy, dx)


def remove_consecutive_duplicates(
    xy: np.ndarray,
    t_ns: Optional[np.ndarray] = None,
    eps: float = 1e-9
):
    if xy.shape[0] <= 1:
        return (xy, t_ns) if t_ns is not None else xy
    keep = [0]
    for i in range(1, xy.shape[0]):
        if np.linalg.norm(xy[i] - xy[i - 1]) > eps:
            keep.append(i)
    keep = np.asarray(keep, dtype=int)
    if t_ns is None:
        return xy[keep]
    return xy[keep], t_ns[keep]


def infer_collection_label_from_name(name: str) -> Tuple[str, int]:
    lower = name.lower()
    if "segunda" in lower:
        return "segunda", 2
    if "primeira" in lower:
        return "primeira", 1
    # If unlabeled, assume first collection (matches your naming style in one scenario)
    return "primeira", 1


def parse_scenario_id(folder_name: str) -> Optional[int]:
    m = SCENARIO_ID_RE.match(folder_name)
    return int(m.group(1)) if m else None


def parse_speed_kmh(stem: str) -> Optional[int]:
    m = SPEED_RE.search(stem)
    return int(m.group(1)) if m else None


EARTH_RADIUS_M = 6378137.0

def latlon_to_local_xy_m(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat0_deg: Optional[float] = None,
    lon0_deg: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Local tangent plane approximation (equirectangular around anchor).
    Good for short trajectories (your scenarios are short), errors are tiny.
    x = East (m), y = North (m)
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lon)

    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float), float("nan"), float("nan")

    first_idx = int(np.flatnonzero(valid)[0])
    lat0 = float(lat[first_idx] if lat0_deg is None else lat0_deg)
    lon0 = float(lon[first_idx] if lon0_deg is None else lon0_deg)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    x = EARTH_RADIUS_M * dlon * math.cos(lat0_rad)
    y = EARTH_RADIUS_M * dlat
    return x, y, lat0, lon0


def rigid_align_2d(ref_xy: np.ndarray, mov_xy: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Kabsch alignment in 2D (rotation + translation, no scale) to align mov -> ref.
    Returns aligned mov and alignment metadata.
    """
    if ref_xy.shape != mov_xy.shape:
        raise ValueError("ref_xy and mov_xy must have same shape for rigid alignment")
    if ref_xy.shape[0] < 2:
        return mov_xy.copy(), {"rotation_deg": 0.0, "tx": 0.0, "ty": 0.0}

    ref_cent = ref_xy.mean(axis=0)
    mov_cent = mov_xy.mean(axis=0)
    X = ref_xy - ref_cent
    Y = mov_xy - mov_cent

    H = Y.T @ X
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned = (Y @ R.T) + ref_cent
    t = ref_cent - (mov_cent @ R.T)
    rot_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return aligned, {"rotation_deg": float(rot_deg), "tx": float(t[0]), "ty": float(t[1])}


def compute_pair_metrics_xy(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    n_samples: int = 300,
    allow_reverse: bool = True,
    align_mode: str = "none"
) -> Dict[str, float]:
    """
    Compare two XY trajectories.
    align_mode: none | start | rigid
    """
    if a_xy.shape[0] < 2 or b_xy.shape[0] < 2:
        raise ValueError("Each trajectory must have at least 2 points")

    a_r = resample_by_arclength(a_xy, n_samples)
    b_r = resample_by_arclength(b_xy, n_samples)

    use_reversed = False
    if allow_reverse:
        d_direct = np.linalg.norm(a_r - b_r, axis=1).mean()
        d_rev = np.linalg.norm(a_r - b_r[::-1], axis=1).mean()
        if d_rev < d_direct:
            b_r = b_r[::-1].copy()
            use_reversed = True

    align_meta = {"rotation_deg": 0.0, "tx": 0.0, "ty": 0.0}
    if align_mode == "start":
        shift = a_r[0] - b_r[0]
        b_cmp = b_r + shift
        align_meta["tx"] = float(shift[0])
        align_meta["ty"] = float(shift[1])
    elif align_mode == "rigid":
        b_cmp, align_meta = rigid_align_2d(a_r, b_r)
    else:
        b_cmp = b_r

    dist = np.linalg.norm(a_r - b_cmp, axis=1)
    a_h = heading_from_path(a_r)
    b_h = heading_from_path(b_cmp)
    hdg_err = np.abs(wrap_angle_rad(a_h - b_h))

    return {
        "n_samples": int(n_samples),
        "reversed_alignment_used": int(use_reversed),
        "align_mode": align_mode,
        "align_rotation_deg": align_meta["rotation_deg"],
        "align_tx_m": align_meta["tx"],
        "align_ty_m": align_meta["ty"],
        "lateral_error_mean_m": float(np.mean(dist)),
        "lateral_error_rmse_m": float(np.sqrt(np.mean(dist ** 2))),
        "lateral_error_max_m": float(np.max(dist)),
        "lateral_error_p95_m": float(np.percentile(dist, 95)),
        "heading_error_mean_deg": float(np.degrees(np.mean(hdg_err))),
        "heading_error_p95_deg": float(np.degrees(np.percentile(hdg_err, 95))),
        "heading_error_max_deg": float(np.degrees(np.max(hdg_err))),
        "path_length_a_m": float(trajectory_length(a_xy)),
        "path_length_b_m": float(trajectory_length(b_xy)),
        "path_length_abs_diff_m": float(abs(trajectory_length(a_xy) - trajectory_length(b_xy))),
        "start_point_diff_m_raw": float(np.linalg.norm(a_xy[0] - b_xy[0])),
        "end_point_diff_m_raw": float(np.linalg.norm(a_xy[-1] - b_xy[-1])),
    }


def _extract_odom_xy_and_speed(msg: Any) -> Optional[Tuple[float, float, Optional[float]]]:
    """
    Supports nav_msgs/Odometry and pose-like messages with nested pose.position.x/y.
    """
    try:
        # nav_msgs/Odometry
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose") and hasattr(msg.pose.pose, "position"):
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            v = None
            if hasattr(msg, "twist") and hasattr(msg.twist, "twist") and hasattr(msg.twist.twist, "linear"):
                lx = float(msg.twist.twist.linear.x)
                ly = float(msg.twist.twist.linear.y)
                lz = float(getattr(msg.twist.twist.linear, "z", 0.0))
                v = math.sqrt(lx * lx + ly * ly + lz * lz)
            return x, y, v
        # geometry_msgs/PoseStamped-like
        if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
            x = float(msg.pose.position.x)
            y = float(msg.pose.position.y)
            return x, y, None
    except Exception:
        return None
    return None


def _extract_navsatfix_latlon(msg: Any) -> Optional[Tuple[float, float, float]]:
    try:
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        alt = float(getattr(msg, "altitude", float("nan")))
        return lat, lon, alt
    except Exception:
        return None


def read_bag_trajectories(
    bag_dir: Path,
    topics_of_interest: Optional[List[str]] = None
) -> Tuple[Dict[str, TopicTrajectory], Dict[str, str], List[str]]:
    """
    Returns trajectories extracted from selected topics and a topic_type map.
    """
    warnings: List[str] = []
    trajs: Dict[str, TopicTrajectory] = {}

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topic_type_map: Dict[str, str] = {}
    for t in reader.get_all_topics_and_types():
        topic_type_map[t.name] = t.type

    if topics_of_interest is None:
        topics_of_interest = ODOM_CANDIDATES + GPS_CANDIDATES

    active_topics = [t for t in topics_of_interest if t in topic_type_map]
    if not active_topics:
        warnings.append(f"No candidate topics found in bag: {bag_dir.name}")
        return trajs, topic_type_map, warnings

    msg_type_cache: Dict[str, Any] = {}
    buffers: Dict[str, Dict[str, List[float]]] = {}
    for topic in active_topics:
        kind = "gps_ll" if topic in GPS_CANDIDATES else "odom_xy"
        buffers[topic] = {
            "t_ns": [],
            "x_or_lon": [],
            "y_or_lat": [],
            "speed_mps": [] if kind == "odom_xy" else None
        }
        try:
            msg_type_cache[topic] = get_message(topic_type_map[topic])
        except Exception as e:
            warnings.append(f"Could not resolve message type for {topic} ({topic_type_map[topic]}): {e}")

    while reader.has_next():
        topic, rawdata, t_ns = reader.read_next()
        if topic not in active_topics:
            continue
        if topic not in msg_type_cache:
            continue
        try:
            msg = deserialize_message(rawdata, msg_type_cache[topic])
        except Exception as e:
            warnings.append(f"Failed to deserialize on {topic}: {e}")
            continue

        if topic in GPS_CANDIDATES:
            out = _extract_navsatfix_latlon(msg)
            if out is None:
                continue
            lat, lon, _ = out
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            if abs(lat) > 90.0 or abs(lon) > 180.0:
                continue
            buffers[topic]["t_ns"].append(int(t_ns))
            buffers[topic]["x_or_lon"].append(lon)
            buffers[topic]["y_or_lat"].append(lat)
        else:
            out = _extract_odom_xy_and_speed(msg)
            if out is None:
                continue
            x, y, v = out
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            buffers[topic]["t_ns"].append(int(t_ns))
            buffers[topic]["x_or_lon"].append(x)
            buffers[topic]["y_or_lat"].append(y)
            buffers[topic]["speed_mps"].append(float("nan") if v is None else float(v))

    for topic, buf in buffers.items():
        t_ns = np.asarray(buf["t_ns"], dtype=np.int64)
        x = np.asarray(buf["x_or_lon"], dtype=float)
        y = np.asarray(buf["y_or_lat"], dtype=float)

        if t_ns.size == 0:
            continue

        order = np.argsort(t_ns, kind="stable")
        t_ns = t_ns[order]
        x = x[order]
        y = y[order]

        kind = "gps_ll" if topic in GPS_CANDIDATES else "odom_xy"
        speed = None
        if kind == "odom_xy" and buf["speed_mps"] is not None:
            speed = np.asarray(buf["speed_mps"], dtype=float)[order]

        xy = np.column_stack((x, y))
        xy_dedup, t_ns_dedup = remove_consecutive_duplicates(xy, t_ns)
        if xy_dedup.shape[0] < 2:
            warnings.append(f"Topic {topic} in {bag_dir.name} has <2 unique trajectory points after dedup.")

        if speed is not None:
            keep_idx = [0]
            for i in range(1, xy.shape[0]):
                if np.linalg.norm(xy[i] - xy[i - 1]) > 1e-9:
                    keep_idx.append(i)
            speed = speed[np.asarray(keep_idx, dtype=int)]

        trajs[topic] = TopicTrajectory(
            topic=topic,
            kind=kind,
            count=int(xy_dedup.shape[0]),
            t0_ns=int(t_ns_dedup[0]) if t_ns_dedup.size else None,
            t1_ns=int(t_ns_dedup[-1]) if t_ns_dedup.size else None,
            x_or_lon=xy_dedup[:, 0],
            y_or_lat=xy_dedup[:, 1],
            t_ns=t_ns_dedup,
            speed_mps=speed,
        )

    return trajs, topic_type_map, warnings


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_xy_csv(
    path: Path,
    t_ns: np.ndarray,
    xy: np.ndarray,
    extra_cols: Optional[Dict[str, np.ndarray]] = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = extra_cols or {}
    headers = ["idx", "t_ns", "x_m", "y_m"] + list(extras.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(xy.shape[0]):
            row = [i, int(t_ns[i]), float(xy[i, 0]), float(xy[i, 1])]
            for k in extras.keys():
                arr = extras[k]
                row.append(float(arr[i]) if i < len(arr) and np.isfinite(arr[i]) else "")
            w.writerow(row)


def write_ll_csv(path: Path, t_ns: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_ns", "latitude_deg", "longitude_deg"])
        for i in range(lat.shape[0]):
            w.writerow([i, int(t_ns[i]), float(lat[i]), float(lon[i])])


def write_mean_reference_yaml(path: Path, mean_xy: np.ndarray, meta: Dict[str, Any]) -> None:
    try:
        import yaml
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    yaws = heading_from_path(mean_xy)
    payload = {
        "meta": meta,
        "trajectory_xy_m": [
            {"x": float(x), "y": float(y), "yaw_rad": float(yaw)}
            for (x, y), yaw in zip(mean_xy, yaws)
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def choose_primary_topic(trajs: Dict[str, TopicTrajectory], prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in trajs and trajs[prefer].count >= 2:
        return prefer

    for t in ["/odometry/global", "/odometry/gps", "/odom", "/odometry/local"]:
        if t in trajs and trajs[t].count >= 2:
            return t

    if "__gps_fix_projected_xy__" in trajs and trajs["__gps_fix_projected_xy__"].count >= 2:
        return "__gps_fix_projected_xy__"
    return None


def compare_and_mean(a_xy: np.ndarray, b_xy: np.ndarray, n_samples: int) -> Tuple[Dict[str, float], np.ndarray]:
    metrics = compute_pair_metrics_xy(a_xy, b_xy, n_samples=n_samples, allow_reverse=True, align_mode="none")

    a_r = resample_by_arclength(a_xy, n_samples)
    b_r = resample_by_arclength(b_xy, n_samples)

    d_direct = np.linalg.norm(a_r - b_r, axis=1).mean()
    d_rev = np.linalg.norm(a_r - b_r[::-1], axis=1).mean()
    if d_rev < d_direct:
        b_r = b_r[::-1].copy()

    mean_xy = 0.5 * (a_r + b_r)
    return metrics, mean_xy


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract GT trajectories directly from rosbags and analyze reproducibility.")
    parser.add_argument(
        "--bags-root",
        default=str(Path(__file__).resolve().parents[2] / "waypoints"),
        help="Root containing scenario folders with ground-truth bag directories"
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[2] / "gt_from_bag_outputs"),
        help="Output directory"
    )
    parser.add_argument("--n-samples", type=int, default=300, help="Resample points for comparisons/mean GT")
    parser.add_argument(
        "--primary-topic",
        default=None,
        help="Preferred topic for GT reproducibility (e.g., /odometry/global). If absent, auto-fallback is used."
    )
    parser.add_argument("--export-per-bag-csv", action="store_true", help="Export detailed per-bag trajectory CSVs")
    args = parser.parse_args()

    bags_root = Path(args.bags_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not bags_root.exists():
        print(f"ERROR: bags root not found: {bags_root}", file=sys.stderr)
        return 2

    scenario_dirs = sorted([p for p in bags_root.iterdir() if p.is_dir()])
    bag_infos: List[BagInfo] = []

    for scen_dir in scenario_dirs:
        scen_id = parse_scenario_id(scen_dir.name)
        if scen_id is None:
            continue
        for meta in sorted(scen_dir.rglob("metadata.yaml")):
            bag_dir = meta.parent
            bag_name = bag_dir.name
            speed = parse_speed_kmh(bag_name)
            coll_label, coll_idx = infer_collection_label_from_name(bag_name)
            bag_infos.append(BagInfo(
                scenario_id=scen_id,
                scenario_folder=scen_dir.name,
                bag_dir=str(bag_dir),
                bag_name=bag_name,
                speed_kmh=speed,
                collection_label=coll_label,
                collection_index=coll_idx,
            ))

    if not bag_infos:
        print("ERROR: No rosbag directories found (metadata.yaml not found under scenario folders).", file=sys.stderr)
        return 3

    bag_summary_rows: List[Dict[str, Any]] = []
    within_bag_compare_rows: List[Dict[str, Any]] = []
    warnings_all: List[str] = []
    primary_traj_cache: Dict[str, Dict[str, Any]] = {}

    for info in bag_infos:
        bag_dir = Path(info.bag_dir)
        trajs, topic_types, warnings = read_bag_trajectories(bag_dir)
        warnings_all.extend([f"{info.bag_name}: {w}" for w in warnings])

        # Project GPS topics to local XY and add synthetic odom-like trajectory
        for gps_topic in GPS_CANDIDATES:
            if gps_topic not in trajs:
                continue

            gps_tr = trajs[gps_topic]
            lon = gps_tr.x_or_lon
            lat = gps_tr.y_or_lat
            x_m, y_m, lat0, lon0 = latlon_to_local_xy_m(lat, lon)

            if x_m.size < 2:
                continue

            xy = np.column_stack((x_m, y_m))
            xy, t2 = remove_consecutive_duplicates(xy, gps_tr.t_ns)
            if xy.shape[0] < 2:
                continue

            synthetic_topic = "__gps_fix_projected_xy__" if gps_topic == "/gps/fix" else "__gps_filtered_projected_xy__"
            trajs[synthetic_topic] = TopicTrajectory(
                topic=synthetic_topic,
                kind="odom_xy",
                count=int(xy.shape[0]),
                t0_ns=int(t2[0]),
                t1_ns=int(t2[-1]),
                x_or_lon=xy[:, 0],
                y_or_lat=xy[:, 1],
                t_ns=t2,
                speed_mps=None,
            )
            topic_types[synthetic_topic] = (
                f"synthetic_local_xy_from_{gps_topic}_anchor_latlon({lat0:.8f},{lon0:.8f})"
            )

            if args.export_per_bag_csv:
                out_bag_dir = out_dir / "per_bag" / info.scenario_folder / info.bag_name
                write_ll_csv(
                    out_bag_dir / f"{gps_topic.strip('/').replace('/', '_')}_latlon.csv",
                    gps_tr.t_ns,
                    lat,
                    lon
                )
                write_xy_csv(
                    out_bag_dir / f"{synthetic_topic.strip('_')}.csv",
                    t2,
                    xy
                )

        if args.export_per_bag_cvs if False else False:
            pass

        if args.export_per_bag_csv:
            out_bag_dir = out_dir / "per_bag" / info.scenario_folder / info.bag_name
            for topic, tr in trajs.items():
                if tr.kind != "odom_xy":
                    continue
                xy = np.column_stack((tr.x_or_lon, tr.y_or_lat))
                extras = {}
                if tr.speed_mps is not None:
                    extras["speed_mps"] = tr.speed_mps
                fname = (
                    f"{topic.strip('/').replace('/', '_')}.csv"
                    if not topic.startswith("__")
                    else f"{topic.strip('_')}.csv"
                )
                write_xy_csv(out_bag_dir / fname, tr.t_ns, xy, extras)

        for topic, tr in trajs.items():
            row = asdict(info)
            row["topic"] = topic
            row["topic_type"] = topic_types.get(topic, "unknown")
            row["msg_count_unique_points"] = tr.count
            row["t0_ns"] = tr.t0_ns
            row["t1_ns"] = tr.t1_ns
            row["duration_s"] = (
                (tr.t1_ns - tr.t0_ns) / 1e9
                if (tr.t0_ns is not None and tr.t1_ns is not None)
                else None
            )

            if tr.kind == "odom_xy":
                xy = np.column_stack((tr.x_or_lon, tr.y_or_lat))
                row["coord_kind"] = "xy_m"
                row["path_length_m"] = trajectory_length(xy)
                row["start_x_or_lon"] = float(xy[0, 0]) if xy.shape[0] else None
                row["start_y_or_lat"] = float(xy[0, 1]) if xy.shape[0] else None
                row["end_x_or_lon"] = float(xy[-1, 0]) if xy.shape[0] else None
                row["end_y_or_lat"] = float(xy[-1, 1]) if xy.shape[0] else None

                if tr.speed_mps is not None and tr.speed_mps.size:
                    finite = tr.speed_mps[np.isfinite(tr.speed_mps)]
                    row["speed_mean_mps"] = float(np.mean(finite)) if finite.size else None
                    row["speed_mean_kmh"] = float(np.mean(finite) * 3.6) if finite.size else None
                    row["speed_p95_kmh"] = float(np.percentile(finite, 95) * 3.6) if finite.size else None
                else:
                    row["speed_mean_mps"] = None
                    row["speed_mean_kmh"] = None
                    row["speed_p95_kmh"] = None
            else:
                row["coord_kind"] = "latlon_deg"
                row["path_length_m"] = None
                row["start_x_or_lon"] = float(tr.x_or_lon[0]) if tr.count else None
                row["start_y_or_lat"] = float(tr.y_or_lat[0]) if tr.count else None
                row["end_x_or_lon"] = float(tr.x_or_lon[-1]) if tr.count else None
                row["end_y_or_lat"] = float(tr.y_or_lat[-1]) if tr.count else None
                row["speed_mean_mps"] = None
                row["speed_mean_kmh"] = None
                row["speed_p95_kmh"] = None

            bag_summary_rows.append(row)

        compare_pairs = [
            ("/odometry/global", "__gps_fix_projected_xy__"),
            ("/odometry/gps", "__gps_fix_projected_xy__"),
            ("/odom", "__gps_fix_projected_xy__"),
            ("/odometry/local", "__gps_fix_projected_xy__"),
            ("/odometry/global", "/odometry/local"),
            ("/odometry/global", "/odometry/gps"),
        ]
        for ta, tb in compare_pairs:
            if ta not in trajs or tb not in trajs:
                continue
            a = trajs[ta]
            b = trajs[tb]
            if a.kind != "odom_xy" or b.kind != "odom_xy":
                continue
            if a.count < 2 or b.count < 2:
                continue

            a_xy = np.column_stack((a.x_or_lon, a.y_or_lat))
            b_xy = np.column_stack((b.x_or_lon, b.y_or_lat))

            try:
                metrics = compute_pair_metrics_xy(
                    a_xy,
                    b_xy,
                    n_samples=args.n_samples,
                    allow_reverse=True,
                    align_mode="rigid"
                )
            except Exception as e:
                warnings_all.append(f"{info.bag_name}: failed compare {ta} vs {tb}: {e}")
                continue

            row = asdict(info)
            row["topic_a"] = ta
            row["topic_b"] = tb
            row.update(metrics)
            within_bag_compare_rows.append(row)

        primary_topic = choose_primary_topic(trajs, prefer=args.primary_topic)
        if primary_topic is None:
            warnings_all.append(f"{info.bag_name}: no valid primary topic trajectory found")
        else:
            tr = trajs[primary_topic]
            xy = np.column_stack((tr.x_or_lon, tr.y_or_lat))
            primary_traj_cache[info.bag_dir] = {
                "bag_info": info,
                "primary_topic": primary_topic,
                "xy": xy,
                "t_ns": tr.t_ns,
            }

    if bag_summary_rows:
        preferred = [
            "scenario_id", "scenario_folder", "bag_name", "bag_dir",
            "speed_kmh", "collection_label", "collection_index",
            "topic", "topic_type", "coord_kind", "msg_count_unique_points",
            "t0_ns", "t1_ns", "duration_s", "path_length_m",
            "start_x_or_lon", "start_y_or_lat", "end_x_or_lon", "end_y_or_lat",
            "speed_mean_mps", "speed_mean_kmh", "speed_p95_kmh",
        ]
        fields = [f for f in preferred if f in bag_summary_rows[0]] + [
            f for f in bag_summary_rows[0].keys() if f not in preferred
        ]
        write_csv(out_dir / "bag_topic_summary.csv", bag_summary_rows, fields)

    if within_bag_compare_rows:
        preferred = [
            "scenario_id", "scenario_folder", "bag_name", "speed_kmh",
            "collection_label", "collection_index",
            "topic_a", "topic_b",
            "align_mode", "align_rotation_deg", "align_tx_m", "align_ty_m",
            "lateral_error_mean_m", "lateral_error_rmse_m", "lateral_error_p95_m", "lateral_error_max_m",
            "heading_error_mean_deg", "heading_error_p95_deg", "heading_error_max_deg",
            "path_length_a_m", "path_length_b_m", "path_length_abs_diff_m",
        ]
        fields = [f for f in preferred if f in within_bag_compare_rows[0]] + [
            f for f in within_bag_compare_rows[0].keys() if f not in preferred
        ]
        write_csv(out_dir / "within_bag_topic_consistency.csv", within_bag_compare_rows, fields)

    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for item in primary_traj_cache.values():
        bi: BagInfo = item["bag_info"]
        if bi.speed_kmh is None:
            warnings_all.append(f"{bi.bag_name}: speed_kmh not parsed from name, excluded from GT pairing")
            continue
        grouped.setdefault((bi.scenario_id, bi.speed_kmh), []).append(item)

    gt_pair_rows: List[Dict[str, Any]] = []
    mean_refs_dir = out_dir / "gt_mean_references"

    for (scenario_id, speed_kmh), items in sorted(grouped.items()):
        items_sorted = sorted(items, key=lambda x: (x["bag_info"].collection_index, x["bag_info"].bag_name))
        gt1 = next((x for x in items_sorted if x["bag_info"].collection_index == 1), None)
        gt2 = next((x for x in items_sorted if x["bag_info"].collection_index == 2), None)

        if gt1 is None or gt2 is None:
            warnings_all.append(
                f"scenario={scenario_id}, speed={speed_kmh}: missing GT1/GT2 bag for primary-topic reproducibility"
            )
            continue

        a_xy = gt1["xy"]
        b_xy = gt2["xy"]

        try:
            metrics, mean_xy = compare_and_mean(a_xy, b_xy, n_samples=args.n_samples)
        except Exception as e:
            warnings_all.append(f"scenario={scenario_id}, speed={speed_kmh}: pair comparison failed: {e}")
            continue

        gt1_info: BagInfo = gt1["bag_info"]
        gt2_info: BagInfo = gt2["bag_info"]
        primary_topic_name = (
            gt1["primary_topic"]
            if gt1["primary_topic"] == gt2["primary_topic"]
            else f"{gt1['primary_topic']}|{gt2['primary_topic']}"
        )

        mean_name = f"scenario{scenario_id}_vel{speed_kmh}_GTmean_from_bag"
        mean_csv = mean_refs_dir / f"{mean_name}.csv"
        mean_yaml = mean_refs_dir / f"{mean_name}.yaml"

        # Synthetic time index for mean path points
        write_xy_csv(mean_csv, np.arange(mean_xy.shape[0], dtype=np.int64), mean_xy)

        write_mean_reference_yaml(mean_yaml, mean_xy, {
            "scenario_id": scenario_id,
            "speed_kmh": speed_kmh,
            "source": "rosbag_primary_topic",
            "primary_topic": primary_topic_name,
            "gt1_bag": gt1_info.bag_dir,
            "gt2_bag": gt2_info.bag_dir,
            "n_samples": args.n_samples,
        })

        row = {
            "scenario_id": scenario_id,
            "speed_kmh": speed_kmh,
            "primary_topic": primary_topic_name,
            "gt1_bag_name": gt1_info.bag_name,
            "gt2_bag_name": gt2_info.bag_name,
            "gt1_bag_dir": gt1_info.bag_dir,
            "gt2_bag_dir": gt2_info.bag_dir,
            "mean_gt_csv": str(mean_csv),
            "mean_gt_yaml": str(mean_yaml),
        }
        row.update(metrics)
        gt_pair_rows.append(row)

    if gt_pair_rows:
        preferred = [
            "scenario_id", "speed_kmh", "primary_topic",
            "gt1_bag_name", "gt2_bag_name",
            "lateral_error_mean_m", "lateral_error_rmse_m", "lateral_error_p95_m", "lateral_error_max_m",
            "heading_error_mean_deg", "heading_error_p95_deg", "heading_error_max_deg",
            "path_length_a_m", "path_length_b_m", "path_length_abs_diff_m",
            "mean_gt_csv", "mean_gt_yaml",
        ]
        fields = [f for f in preferred if f in gt_pair_rows[0]] + [
            f for f in gt_pair_rows[0].keys() if f not in preferred
        ]
        write_csv(out_dir / "gt_pair_reproducibility_from_bag.csv", gt_pair_rows, fields)

    bag_manifest_rows = [asdict(b) for b in bag_infos]
    if bag_manifest_rows:
        write_csv(out_dir / "bag_manifest.csv", bag_manifest_rows, list(bag_manifest_rows[0].keys()))

    summary_path = out_dir / "README_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Ground Truth extraction and reproducibility from rosbags\n")
        f.write(f"Bags root: {bags_root}\n")
        f.write(f"Output dir: {out_dir}\n")
        f.write(f"Discovered bag folders: {len(bag_infos)}\n")
        f.write(f"Bags with primary trajectory: {len(primary_traj_cache)}\n")
        f.write(f"GT pairs processed: {len(gt_pair_rows)}\n\n")

        if gt_pair_rows:
            f.write("GT pair reproducibility (by scenario, speed):\n")
            for r in sorted(gt_pair_rows, key=lambda z: (z["scenario_id"], z["speed_kmh"])):
                f.write(
                    f"  scenario {r['scenario_id']}, {r['speed_kmh']} km/h, topic={r['primary_topic']}, "
                    f"lat_rmse={r['lateral_error_rmse_m']:.3f} m, "
                    f"lat_p95={r['lateral_error_p95_m']:.3f} m, "
                    f"heading_mean={r['heading_error_mean_deg']:.3f} deg\n"
                )
            f.write("\n")

        if warnings_all:
            f.write("Warnings:\n")
            for w in warnings_all:
                f.write(f"  - {w}\n")

    print(f"[OK] Wrote outputs to: {out_dir}")
    print(f"[OK] Summary: {summary_path}")
    if warnings_all:
        print("\nWarnings:")
        for w in warnings_all[:40]:
            print(f"  - {w}")
        if len(warnings_all) > 40:
            print(f"  ... and {len(warnings_all)-40} more warnings")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())