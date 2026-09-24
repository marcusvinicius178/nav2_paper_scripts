#!/usr/bin/env python3
"""
GT extraction and reproducibility directly from ROS 2 rosbags (MCAP) for Marcus' dataset.

What this script does
- Scans scenario folders under --bags-root
- Finds rosbag folders by locating metadata.yaml
- Extracts trajectories from:
    - /odometry/global (preferred GT real trajectory)
    - /gps/fix (raw GPS lat/lon)
- Converts /gps/fix lat/lon to local XY (meters)
- Computes GT1 vs GT2 reproducibility by scenario/speed using the primary topic
- Computes within-bag consistency: /odometry/global vs projected /gps/fix
- Exports CSVs and a summary text

Important (updated)
- Opens the real .mcap file directly when available (ignores .mcap.zstd metadata sidecar reference)
- Applies reader topic filter when supported by your ROS 2 version
- GT pair reproducibility defaults to rigid alignment (to compensate frame offsets between collections)
- Also exports RAW (unaligned) GT pair metrics side-by-side for transparency
- Mean GT reference is generated from rigid-aligned trajectories (more consistent for planner comparison)

Usage (after sourcing ROS 2):
  python3 gt_from_rosbag_analysis.py \
    --bags-root /path/to/NAV2_Paper_Scripts/waypoints \
    --out-dir /path/to/NAV2_Paper_Scripts/gt_from_bag_outputs \
    --primary-topic /odometry/global \
    --n-samples 300 \
    --export-per-bag-csv
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
    print("ERROR: Could not import ROS 2 Python APIs (rosbag2_py/rclpy).", file=sys.stderr)
    print("Run: source /opt/ros/jazzy/setup.bash", file=sys.stderr)
    raise


SCENARIO_ID_RE = re.compile(r"^(\d+)-")
SPEED_RE = re.compile(r"vel_(\d+)", re.IGNORECASE)
EARTH_RADIUS_M = 6378137.0


@dataclass
class BagInfo:
    scenario_id: int
    scenario_folder: str
    bag_dir: str
    bag_name: str
    speed_kmh: Optional[int]
    collection_label: str   # primeira | segunda | unknown
    collection_index: int   # 1 | 2 | 0


def parse_scenario_id(folder_name: str) -> Optional[int]:
    m = SCENARIO_ID_RE.match(folder_name)
    return int(m.group(1)) if m else None


def parse_speed_kmh(name: str) -> Optional[int]:
    m = SPEED_RE.search(name)
    return int(m.group(1)) if m else None


def infer_collection_label_from_name(name: str) -> Tuple[str, int]:
    lower = name.lower()
    if "segunda" in lower:
        return "segunda", 2
    if "primeira" in lower:
        return "primeira", 1
    # In your dataset some first collections are unlabeled
    return "primeira", 1


def resolve_bag_uri_for_mcap(bag_dir: Path) -> Tuple[Path, Optional[str]]:
    """
    Prefer opening the real .mcap file directly (bypasses stale metadata pointing to .mcap.zstd).
    If no .mcap is found, fallback to the bag directory.
    Returns (uri, warning_note_or_none)
    """
    mcap_files = sorted([p for p in bag_dir.glob("*.mcap") if p.is_file()])

    if not mcap_files:
        return bag_dir, None

    # Prefer segment 0 if present (most common naming: *_0.mcap)
    for p in mcap_files:
        if p.name.endswith("_0.mcap"):
            return p, f"Opened direct MCAP file (ignoring metadata file list): {p.name}"

    return mcap_files[0], f"Opened direct MCAP file (ignoring metadata file list): {mcap_files[0].name}"


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_xy_csv(path: Path, t_ns: np.ndarray, xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_ns", "x_m", "y_m"])
        for i in range(xy.shape[0]):
            w.writerow([i, int(t_ns[i]), float(xy[i, 0]), float(xy[i, 1])])


def write_latlon_csv(path: Path, t_ns: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_ns", "latitude_deg", "longitude_deg"])
        for i in range(lat.shape[0]):
            w.writerow([i, int(t_ns[i]), float(lat[i]), float(lon[i])])


def wrap_angle_rad(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


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
    xy: np.ndarray, t_ns: np.ndarray, eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] <= 1:
        return xy, t_ns
    keep = [0]
    for i in range(1, xy.shape[0]):
        if np.linalg.norm(xy[i] - xy[i - 1]) > eps:
            keep.append(i)
    k = np.asarray(keep, dtype=int)
    return xy[k], t_ns[k]


def latlon_to_local_xy_m(
    lat_deg: np.ndarray, lon_deg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
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

    x = EARTH_RADIUS_M * dlon * math.cos(lat0_rad)  # East
    y = EARTH_RADIUS_M * dlat                       # North
    return x, y, lat0, lon0


def rigid_align_2d(ref_xy: np.ndarray, mov_xy: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    if ref_xy.shape != mov_xy.shape:
        raise ValueError("Shapes must match for rigid alignment")
    if ref_xy.shape[0] < 2:
        return mov_xy.copy(), {"rotation_deg": 0.0, "tx": 0.0, "ty": 0.0}

    ref_c = ref_xy.mean(axis=0)
    mov_c = mov_xy.mean(axis=0)

    X = ref_xy - ref_c
    Y = mov_xy - mov_c
    H = Y.T @ X
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned = (Y @ R.T) + ref_c
    t = ref_c - (mov_c @ R.T)
    rot_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return aligned, {"rotation_deg": float(rot_deg), "tx": float(t[0]), "ty": float(t[1])}


def compute_pair_metrics_xy(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    n_samples: int = 300,
    allow_reverse: bool = True,
    align_mode: str = "none"
) -> Dict[str, Any]:
    if a_xy.shape[0] < 2 or b_xy.shape[0] < 2:
        raise ValueError("Each trajectory must have at least 2 points")

    a_r = resample_by_arclength(a_xy, n_samples)
    b_r = resample_by_arclength(b_xy, n_samples)

    reversed_used = False
    if allow_reverse:
        d_direct = np.linalg.norm(a_r - b_r, axis=1).mean()
        d_rev = np.linalg.norm(a_r - b_r[::-1], axis=1).mean()
        if d_rev < d_direct:
            b_r = b_r[::-1].copy()
            reversed_used = True

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
        "reversed_alignment_used": int(reversed_used),
        "align_mode": align_mode,
        "align_rotation_deg": align_meta["rotation_deg"],
        "align_tx_m": align_meta["tx"],
        "align_ty_m": align_meta["ty"],
        "lateral_error_mean_m": float(np.mean(dist)),
        "lateral_error_rmse_m": float(np.sqrt(np.mean(dist ** 2))),
        "lateral_error_p95_m": float(np.percentile(dist, 95)),
        "lateral_error_max_m": float(np.max(dist)),
        "heading_error_mean_deg": float(np.degrees(np.mean(hdg_err))),
        "heading_error_p95_deg": float(np.degrees(np.percentile(hdg_err, 95))),
        "heading_error_max_deg": float(np.degrees(np.max(hdg_err))),
        "path_length_a_m": float(trajectory_length(a_xy)),
        "path_length_b_m": float(trajectory_length(b_xy)),
        "path_length_abs_diff_m": float(abs(trajectory_length(a_xy) - trajectory_length(b_xy))),
        "start_point_diff_m_raw": float(np.linalg.norm(a_xy[0] - b_xy[0])),
        "end_point_diff_m_raw": float(np.linalg.norm(a_xy[-1] - b_xy[-1])),
    }


def prefix_metric_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def _extract_odom_xy(msg: Any) -> Optional[Tuple[float, float]]:
    try:
        # nav_msgs/Odometry
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose") and hasattr(msg.pose.pose, "position"):
            return float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)
        # geometry_msgs/PoseStamped-like fallback
        if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
            return float(msg.pose.position.x), float(msg.pose.position.y)
    except Exception:
        return None
    return None


def _extract_navsatfix_latlon(msg: Any) -> Optional[Tuple[float, float]]:
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


def read_bag_topic_arrays(
    bag_dir: Path,
    topics_of_interest: List[str]
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, str], List[str]]:
    warnings: List[str] = []

    reader = rosbag2_py.SequentialReader()

    bag_uri, open_note = resolve_bag_uri_for_mcap(bag_dir)
    if open_note:
        warnings.append(open_note)

    storage_options = rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    topic_type_map: Dict[str, str] = {}
    for t in reader.get_all_topics_and_types():
        topic_type_map[t.name] = t.type

    available = [t for t in topics_of_interest if t in topic_type_map]
    if not available:
        warnings.append("No target topics found in bag")
        return {}, topic_type_map, warnings

    # Try to reduce IO by filtering topics in the reader (supported on many ROS 2 versions)
    try:
        if hasattr(rosbag2_py, "StorageFilter"):
            reader.set_filter(rosbag2_py.StorageFilter(topics=available))
            warnings.append(f"Applied reader topic filter: {available}")
    except Exception as e:
        warnings.append(f"Could not apply reader topic filter (continuing without filter): {e}")

    msg_type_cache: Dict[str, Any] = {}
    buffers: Dict[str, Dict[str, List[float]]] = {}
    for topic in available:
        try:
            msg_type_cache[topic] = get_message(topic_type_map[topic])
        except Exception as e:
            warnings.append(f"Cannot resolve msg type for {topic} ({topic_type_map[topic]}): {e}")
            continue
        buffers[topic] = {"t_ns": [], "a": [], "b": []}

    while reader.has_next():
        topic, rawdata, t_ns = reader.read_next()
        if topic not in buffers:
            continue

        try:
            msg = deserialize_message(rawdata, msg_type_cache[topic])
        except Exception as e:
            warnings.append(f"Deserialize failed on {topic}: {e}")
            continue

        if topic == "/odometry/global":
            xy = _extract_odom_xy(msg)
            if xy is None:
                continue
            x, y = xy
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            buffers[topic]["t_ns"].append(int(t_ns))
            buffers[topic]["a"].append(x)
            buffers[topic]["b"].append(y)

        elif topic == "/gps/fix":
            ll = _extract_navsatfix_latlon(msg)
            if ll is None:
                continue
            lat, lon = ll
            buffers[topic]["t_ns"].append(int(t_ns))
            buffers[topic]["a"].append(lat)
            buffers[topic]["b"].append(lon)

    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for topic, buf in buffers.items():
        t = np.asarray(buf["t_ns"], dtype=np.int64)
        a = np.asarray(buf["a"], dtype=float)
        b = np.asarray(buf["b"], dtype=float)
        if t.size == 0:
            continue

        idx = np.argsort(t, kind="stable")
        t = t[idx]
        a = a[idx]
        b = b[idx]

        arrays[topic] = {"t_ns": t, "a": a, "b": b}

    return arrays, topic_type_map, warnings


def choose_primary_xy_topic(extracted: Dict[str, np.ndarray], preferred: Optional[str]) -> Optional[str]:
    # extracted contains keys like "/odometry/global_xy" and "__gps_fix_projected_xy__"
    if preferred == "/odometry/global" and "/odometry/global_xy" in extracted:
        return "/odometry/global_xy"
    if preferred == "/gps/fix" and "__gps_fix_projected_xy__" in extracted:
        return "__gps_fix_projected_xy__"

    if "/odometry/global_xy" in extracted:
        return "/odometry/global_xy"
    if "__gps_fix_projected_xy__" in extracted:
        return "__gps_fix_projected_xy__"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract GT trajectories from rosbags and compute reproducibility."
    )
    parser.add_argument("--bags-root", default="/path/to/NAV2_Paper_Scripts/waypoints")
    parser.add_argument("--out-dir", default="/path/to/NAV2_Paper_Scripts/gt_from_bag_outputs")
    parser.add_argument(
        "--primary-topic",
        default="/odometry/global",
        help="Primary topic for GT pairing: /odometry/global or /gps/fix"
    )
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--export-per-bag-csv", action="store_true")

    # New defaults chosen to compensate frame offsets across repeated GT collections
    parser.add_argument(
        "--gt-pair-align-mode",
        choices=["none", "start", "rigid"],
        default="rigid",
        help="Alignment mode used for GT1 vs GT2 comparison metrics. Default: rigid"
    )
    parser.add_argument(
        "--mean-gt-align-mode",
        choices=["none", "start", "rigid"],
        default="rigid",
        help="Alignment mode applied before averaging GT1 and GT2 into mean GT. Default: rigid"
    )

    args = parser.parse_args()

    bags_root = Path(args.bags_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not bags_root.exists():
        print(f"ERROR: bags root does not exist: {bags_root}", file=sys.stderr)
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

            bag_infos.append(
                BagInfo(
                    scenario_id=scen_id,
                    scenario_folder=scen_dir.name,
                    bag_dir=str(bag_dir),
                    bag_name=bag_name,
                    speed_kmh=speed,
                    collection_label=coll_label,
                    collection_index=coll_idx,
                )
            )

    if not bag_infos:
        print(
            "ERROR: No rosbag folders found under scenario dirs (metadata.yaml not found).",
            file=sys.stderr,
        )
        return 3

    manifest_rows = [asdict(b) for b in bag_infos]
    write_csv(out_dir / "bag_manifest.csv", manifest_rows, list(manifest_rows[0].keys()))

    bag_topic_summary_rows: List[Dict[str, Any]] = []
    within_bag_rows: List[Dict[str, Any]] = []
    warnings_all: List[str] = []

    # cache primary GT trajectories for pairing GT1 vs GT2
    primary_gt_cache: Dict[str, Dict[str, Any]] = {}

    total_bags = len(bag_infos)
    for idx_bag, info in enumerate(bag_infos, start=1):
        print(
            f"[{idx_bag}/{total_bags}] Reading bag: {info.bag_name} "
            f"(scenario={info.scenario_id}, speed={info.speed_kmh})"
        )

        bag_dir = Path(info.bag_dir)

        arrays, topic_type_map, warnings = read_bag_topic_arrays(
            bag_dir, topics_of_interest=["/odometry/global", "/gps/fix"]
        )
        for w in warnings:
            warnings_all.append(f"{info.bag_name}: {w}")

        extracted_xy: Dict[str, np.ndarray] = {}
        extracted_t: Dict[str, np.ndarray] = {}

        # /odometry/global -> XY meters
        if "/odometry/global" in arrays:
            t = arrays["/odometry/global"]["t_ns"]
            x = arrays["/odometry/global"]["a"]
            y = arrays["/odometry/global"]["b"]
            xy = np.column_stack((x, y))
            xy, t = remove_consecutive_duplicates(xy, t)

            if xy.shape[0] >= 2:
                extracted_xy["/odometry/global_xy"] = xy
                extracted_t["/odometry/global_xy"] = t

                bag_topic_summary_rows.append(
                    {
                        "scenario_id": info.scenario_id,
                        "scenario_folder": info.scenario_folder,
                        "bag_name": info.bag_name,
                        "bag_dir": info.bag_dir,
                        "speed_kmh": info.speed_kmh,
                        "collection_label": info.collection_label,
                        "collection_index": info.collection_index,
                        "topic": "/odometry/global",
                        "topic_type": topic_type_map.get("/odometry/global", "unknown"),
                        "coord_kind": "xy_m",
                        "point_count": int(xy.shape[0]),
                        "duration_s": float((t[-1] - t[0]) / 1e9) if t.size >= 2 else 0.0,
                        "path_length_m": float(trajectory_length(xy)),
                        "start_x_or_lon": float(xy[0, 0]),
                        "start_y_or_lat": float(xy[0, 1]),
                        "end_x_or_lon": float(xy[-1, 0]),
                        "end_y_or_lat": float(xy[-1, 1]),
                    }
                )

                if args.export_per_bag_csv:
                    write_xy_csv(
                        out_dir / "per_bag" / info.scenario_folder / info.bag_name / "odometry_global_xy.csv",
                        t,
                        xy,
                    )
            else:
                warnings_all.append(f"{info.bag_name}: /odometry/global has <2 unique points")

        # /gps/fix -> lat/lon and projected XY
        if "/gps/fix" in arrays:
            t = arrays["/gps/fix"]["t_ns"]
            lat = arrays["/gps/fix"]["a"]
            lon = arrays["/gps/fix"]["b"]

            if lat.size >= 2:
                if args.export_per_bag_csv:
                    write_latlon_csv(
                        out_dir / "per_bag" / info.scenario_folder / info.bag_name / "gps_fix_latlon.csv",
                        t,
                        lat,
                        lon,
                    )

                xg, yg, lat0, lon0 = latlon_to_local_xy_m(lat, lon)
                _ = (lat0, lon0)  # kept for possible future reporting
                xy_gps = np.column_stack((xg, yg))
                xy_gps, t_gps = remove_consecutive_duplicates(xy_gps, t)

                if xy_gps.shape[0] >= 2:
                    extracted_xy["__gps_fix_projected_xy__"] = xy_gps
                    extracted_t["__gps_fix_projected_xy__"] = t_gps

                    bag_topic_summary_rows.append(
                        {
                            "scenario_id": info.scenario_id,
                            "scenario_folder": info.scenario_folder,
                            "bag_name": info.bag_name,
                            "bag_dir": info.bag_dir,
                            "speed_kmh": info.speed_kmh,
                            "collection_label": info.collection_label,
                            "collection_index": info.collection_index,
                            "topic": "/gps/fix(projected)",
                            "topic_type": "sensor_msgs/msg/NavSatFix -> local_xy",
                            "coord_kind": "xy_m",
                            "point_count": int(xy_gps.shape[0]),
                            "duration_s": float((t_gps[-1] - t_gps[0]) / 1e9) if t_gps.size >= 2 else 0.0,
                            "path_length_m": float(trajectory_length(xy_gps)),
                            "start_x_or_lon": float(xy_gps[0, 0]),
                            "start_y_or_lat": float(xy_gps[0, 1]),
                            "end_x_or_lon": float(xy_gps[-1, 0]),
                            "end_y_or_lat": float(xy_gps[-1, 1]),
                        }
                    )

                    if args.export_per_bag_csv:
                        write_xy_csv(
                            out_dir / "per_bag" / info.scenario_folder / info.bag_name / "gps_fix_projected_xy.csv",
                            t_gps,
                            xy_gps,
                        )

                    # Within-bag consistency: /odometry/global vs /gps/fix(projected)
                    if "/odometry/global_xy" in extracted_xy:
                        try:
                            m_within = compute_pair_metrics_xy(
                                extracted_xy["/odometry/global_xy"],
                                extracted_xy["__gps_fix_projected_xy__"],
                                n_samples=args.n_samples,
                                allow_reverse=True,
                                align_mode="rigid",  # compensates origin/rotation differences
                            )
                            within_bag_rows.append(
                                {
                                    "scenario_id": info.scenario_id,
                                    "speed_kmh": info.speed_kmh,
                                    "collection_label": info.collection_label,
                                    "collection_index": info.collection_index,
                                    "bag_name": info.bag_name,
                                    "topic_a": "/odometry/global",
                                    "topic_b": "/gps/fix(projected)",
                                    **m_within,
                                }
                            )
                        except Exception as e:
                            warnings_all.append(
                                f"{info.bag_name}: compare odom_global vs gps_fix(projected) failed: {e}"
                            )
                else:
                    warnings_all.append(f"{info.bag_name}: /gps/fix projected has <2 unique points")
            else:
                warnings_all.append(f"{info.bag_name}: /gps/fix has <2 points")

        primary_key = choose_primary_xy_topic(extracted_xy, args.primary_topic)
        if primary_key is None:
            warnings_all.append(f"{info.bag_name}: no valid primary GT trajectory found")
        else:
            primary_gt_cache[info.bag_dir] = {
                "bag_info": info,
                "primary_key": primary_key,
                "xy": extracted_xy[primary_key],
                "t_ns": extracted_t[primary_key],
            }

    if bag_topic_summary_rows:
        fields = list(bag_topic_summary_rows[0].keys())
        write_csv(out_dir / "bag_topic_summary.csv", bag_topic_summary_rows, fields)

    if within_bag_rows:
        fields = list(within_bag_rows[0].keys())
        write_csv(out_dir / "within_bag_topic_consistency.csv", within_bag_rows, fields)

    # GT1 vs GT2 grouping by scenario + speed
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for item in primary_gt_cache.values():
        bi: BagInfo = item["bag_info"]
        if bi.speed_kmh is None:
            warnings_all.append(f"{bi.bag_name}: speed not parsed from name")
            continue
        grouped.setdefault((bi.scenario_id, bi.speed_kmh), []).append(item)

    gt_pair_rows: List[Dict[str, Any]] = []
    mean_ref_dir = out_dir / "gt_mean_references"

    for (scenario_id, speed_kmh), items in sorted(grouped.items()):
        items_sorted = sorted(items, key=lambda d: (d["bag_info"].collection_index, d["bag_info"].bag_name))
        gt1 = next((d for d in items_sorted if d["bag_info"].collection_index == 1), None)
        gt2 = next((d for d in items_sorted if d["bag_info"].collection_index == 2), None)

        if gt1 is None or gt2 is None:
            warnings_all.append(f"scenario={scenario_id}, speed={speed_kmh}: missing GT1 or GT2")
            continue

        a_xy = gt1["xy"]
        b_xy = gt2["xy"]

        # Compute RAW and aligned metrics side-by-side (for transparency in article/reporting)
        try:
            metrics_raw = compute_pair_metrics_xy(
                a_xy,
                b_xy,
                n_samples=args.n_samples,
                allow_reverse=True,
                align_mode="none",
            )
        except Exception as e:
            warnings_all.append(f"scenario={scenario_id}, speed={speed_kmh}: GT pair RAW compare failed: {e}")
            continue

        try:
            metrics_aligned = compute_pair_metrics_xy(
                a_xy,
                b_xy,
                n_samples=args.n_samples,
                allow_reverse=True,
                align_mode=args.gt_pair_align_mode,  # default rigid
            )
        except Exception as e:
            warnings_all.append(
                f"scenario={scenario_id}, speed={speed_kmh}: GT pair aligned compare failed "
                f"(mode={args.gt_pair_align_mode}): {e}"
            )
            continue

        # Build mean GT trajectory (resampled average)
        # Use the configured alignment mode (default rigid) before averaging to avoid frame-offset contamination
        a_r = resample_by_arclength(a_xy, args.n_samples)
        b_r = resample_by_arclength(b_xy, args.n_samples)

        # Keep same traversal direction before alignment
        if np.linalg.norm(a_r - b_r[::-1], axis=1).mean() < np.linalg.norm(a_r - b_r, axis=1).mean():
            b_r = b_r[::-1].copy()

        mean_align_meta = {"rotation_deg": 0.0, "tx": 0.0, "ty": 0.0}
        if args.mean_gt_align_mode == "rigid":
            try:
                b_r_aligned, mean_align_meta = rigid_align_2d(a_r, b_r)
            except Exception as e:
                warnings_all.append(
                    f"scenario={scenario_id}, speed={speed_kmh}: rigid align for mean GT failed, "
                    f"using unaligned mean: {e}"
                )
                b_r_aligned = b_r
        elif args.mean_gt_align_mode == "start":
            shift = a_r[0] - b_r[0]
            b_r_aligned = b_r + shift
            mean_align_meta["tx"] = float(shift[0])
            mean_align_meta["ty"] = float(shift[1])
        else:
            b_r_aligned = b_r

        mean_xy = 0.5 * (a_r + b_r_aligned)
        mean_t = np.arange(mean_xy.shape[0], dtype=np.int64)

        mean_name = (
            f"scenario{scenario_id}_vel{speed_kmh}_GTmean_"
            f"{args.mean_gt_align_mode}Aligned_from_bag.csv"
        )
        write_xy_csv(mean_ref_dir / mean_name, mean_t, mean_xy)

        # Legacy columns keep the ALIGNED metrics (more consistent for article tables)
        # RAW metrics are exported with raw_ prefix for transparency/auditability.
        row = {
            "scenario_id": scenario_id,
            "speed_kmh": speed_kmh,
            "primary_topic_used": (
                gt1["primary_key"]
                if gt1["primary_key"] == gt2["primary_key"]
                else f"{gt1['primary_key']}|{gt2['primary_key']}"
            ),
            "gt_pair_comparison_alignment": args.gt_pair_align_mode,
            "mean_gt_generation_alignment": args.mean_gt_align_mode,
            "mean_gt_alignment_rotation_deg": mean_align_meta["rotation_deg"],
            "mean_gt_alignment_tx_m": mean_align_meta["tx"],
            "mean_gt_alignment_ty_m": mean_align_meta["ty"],
            "gt1_bag_name": gt1["bag_info"].bag_name,
            "gt2_bag_name": gt2["bag_info"].bag_name,
            "gt1_bag_dir": gt1["bag_info"].bag_dir,
            "gt2_bag_dir": gt2["bag_info"].bag_dir,
            "mean_gt_csv": str(mean_ref_dir / mean_name),
            # Selected/primary metrics (aligned)
            **metrics_aligned,
            # Raw metrics for diagnosis and publication transparency
            **prefix_metric_keys(metrics_raw, "raw_"),
        }
        gt_pair_rows.append(row)

    if gt_pair_rows:
        fields = list(gt_pair_rows[0].keys())
        write_csv(out_dir / "gt_pair_reproducibility_from_bag.csv", gt_pair_rows, fields)

    summary_path = out_dir / "README_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("GT extraction from rosbags summary\n")
        f.write(f"Bags root: {bags_root}\n")
        f.write(f"Output dir: {out_dir}\n")
        f.write(f"Discovered bag folders: {len(bag_infos)}\n")
        f.write(f"Bags with valid primary GT trajectory: {len(primary_gt_cache)}\n")
        f.write(f"GT pairs processed (scenario x speed): {len(gt_pair_rows)}\n")
        f.write(f"GT pair comparison alignment (selected metrics): {args.gt_pair_align_mode}\n")
        f.write(f"Mean GT generation alignment: {args.mean_gt_align_mode}\n\n")

        if gt_pair_rows:
            f.write("GT1 vs GT2 reproducibility (selected/aligned metrics):\n")
            for r in sorted(gt_pair_rows, key=lambda z: (z["scenario_id"], z["speed_kmh"])):
                f.write(
                    f"  scenario {r['scenario_id']}, {r['speed_kmh']} km/h, "
                    f"topic={r['primary_topic_used']}, align={r['gt_pair_comparison_alignment']}, "
                    f"lat_rmse={r['lateral_error_rmse_m']:.3f} m, "
                    f"lat_p95={r['lateral_error_p95_m']:.3f} m, "
                    f"heading_mean={r['heading_error_mean_deg']:.3f} deg"
                )
                # Include raw RMSE in summary when alignment is active (helps diagnose frame offsets)
                if r.get("gt_pair_comparison_alignment") != "none" and "raw_lateral_error_rmse_m" in r:
                    f.write(f", raw_lat_rmse={float(r['raw_lateral_error_rmse_m']):.3f} m")
                f.write("\n")
            f.write("\n")

        if warnings_all:
            f.write("Warnings:\n")
            for w in warnings_all:
                f.write(f"  - {w}\n")

    print(f"[OK] Outputs written to: {out_dir}")
    print(f"[OK] Summary: {summary_path}")

    if warnings_all:
        print("\nWarnings (first 30):")
        for w in warnings_all[:30]:
            print(f"  - {w}")
        if len(warnings_all) > 30:
            print(f"  ... and {len(warnings_all) - 30} more warnings")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())