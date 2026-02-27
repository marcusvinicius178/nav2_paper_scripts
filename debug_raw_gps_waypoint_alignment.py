#!/usr/bin/env python3
"""
Debug raw GPS alignment between configured waypoint coordinates, GT raw GPS,
and simulation raw GPS from ROS 2 bags.

Purpose
-------
This script is intentionally diagnostic.
It helps answer these exact questions:
1) Did the GT raw GPS really pass near the waypoint coordinates that were fed to the simulator?
2) Did the simulation raw GPS (/gps/fix) pass near those same waypoint coordinates?
3) Is the simulation GPS shape mirrored (very commonly a Y-sign / north-south inversion)?
4) Are the paths simply reversed in sequence, or truly reflected/misaligned?

What it does
------------
- Loads an explicit waypoint file (YAML or CSV with latitude/longitude pairs)
- Loads one or more GT raw GPS CSV files (gps_fix_latlon.csv)
- Reads /gps/fix directly from one or more simulation bags
- Projects everything to the SAME local XY plane using the FIRST configured waypoint
  as the geodetic origin (lat0/lon0)
- Builds a GT mean raw GPS curve in that common frame
- Tries multiple candidate transforms for each simulation run:
    * raw
    * raw reversed
    * flip_y
    * flip_y reversed
    * flip_x
    * flip_x reversed
    * flip_xy
    * flip_xy reversed
- Scores each candidate against BOTH:
    * configured waypoint polyline
    * GT mean raw GPS polyline
- Reports which candidate best matches
- Generates aggregate plots and per-run debug plots
- Also generates a start-aligned overlay (best variant + translated common start) so shape can be compared visually and quantitatively

Important methodological note
-----------------------------
This script debugs the RAW GPS comparison only.
It does NOT georeference the local /plan topic.
So here, "planner/control" means the EXECUTED simulated trajectory observed via /gps/fix,
not the /plan polyline itself.
"""

import argparse
import csv
import glob
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


EARTH_RADIUS_M = 6378137.0
DEFAULT_GPS_TOPIC = "/gps/fix"
DEFAULT_THRESHOLDS_M = [3.0, 5.0, 10.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug alignment between configured waypoint lat/lon, GT raw GPS, and simulation raw GPS."
    )
    parser.add_argument(
        "--sim-parent-dir",
        required=True,
        help="Parent directory containing simulation bag subfolders (e.g. .../NAVFN_MPPI_Vel_20)",
    )
    parser.add_argument(
        "--sim-glob",
        default="*",
        help="Glob used inside sim-parent-dir to find bag folders (e.g. 'Navfn_mppi_*')",
    )
    parser.add_argument(
        "--gt-csv-glob",
        required=True,
        help="Glob for GT raw GPS CSV files (gps_fix_latlon.csv)",
    )
    parser.add_argument(
        "--waypoint-file",
        required=True,
        help="Waypoint YAML or CSV file containing latitude/longitude points used to feed the simulator.",
    )
    parser.add_argument("--out-dir", default="", help="Output directory (default: <sim-parent-dir>/Output_charts_diagnostic)")
    parser.add_argument("--gps-topic", default=DEFAULT_GPS_TOPIC, help="GPS topic to read from simulation bags")
    parser.add_argument(
        "--resample-samples",
        type=int,
        default=300,
        help="Number of arclength-resampled points used in comparisons",
    )
    parser.add_argument(
        "--thresholds-m",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS_M),
        help="Comma-separated waypoint hit thresholds in meters, e.g. '3,5,10'",
    )
    parser.add_argument(
        "--save-per-run-plots",
        action="store_true",
        help="Also save one debug plot per simulation run",
    )
    return parser.parse_args()


def resolve_bag_uri_for_mcap(bag_dir: Path) -> Path:
    mcap_files = sorted([p for p in bag_dir.glob("*.mcap") if p.is_file()])
    if not mcap_files:
        return bag_dir
    for p in mcap_files:
        if p.name.endswith("_0.mcap"):
            return p
    return mcap_files[0]


def natural_sort_key(value: str) -> List[Any]:
    parts = re.split(r"(\d+)", value)
    out: List[Any] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def parse_thresholds(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    vals = [x for x in vals if x > 0.0]
    if not vals:
        raise ValueError("No positive thresholds were parsed from --thresholds-m")
    return vals


def load_waypoint_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Waypoint file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        pts = load_waypoint_yaml_like(path)
    elif suffix == ".csv":
        pts = load_waypoint_csv(path)
    else:
        raise RuntimeError(f"Unsupported waypoint file extension: {path.suffix}")

    if len(pts) < 2:
        raise RuntimeError(f"Waypoint file has fewer than 2 valid points: {path}")

    lat = np.asarray([p[0] for p in pts], dtype=float)
    lon = np.asarray([p[1] for p in pts], dtype=float)
    lat, lon, _ = remove_consecutive_duplicates_latlon(lat, lon)
    if lat.size < 2:
        raise RuntimeError(f"Waypoint file collapses to fewer than 2 unique points: {path}")

    lat0 = float(lat[0])
    lon0 = float(lon[0])
    x, y = latlon_to_local_xy_with_origin(lat, lon, lat0, lon0)
    xy = np.column_stack((x, y))

    return {
        "source": str(path),
        "lat": lat,
        "lon": lon,
        "xy": xy,
        "lat0_deg": lat0,
        "lon0_deg": lon0,
        "points": int(lat.size),
        "length_m": path_length(xy),
    }


def load_waypoint_yaml_like(path: Path) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    lat_pending: Optional[float] = None
    lon_pending: Optional[float] = None

    rx_lat = re.compile(r"^\s*-?\s*latitude\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
    rx_lon = re.compile(r"^\s*-?\s*longitude\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m_lat = rx_lat.match(line)
            if m_lat:
                lat_pending = float(m_lat.group(1))
                if lat_pending is not None and lon_pending is not None:
                    pts.append((lat_pending, lon_pending))
                    lat_pending, lon_pending = None, None
                continue

            m_lon = rx_lon.match(line)
            if m_lon:
                lon_pending = float(m_lon.group(1))
                if lat_pending is not None and lon_pending is not None:
                    pts.append((lat_pending, lon_pending))
                    lat_pending, lon_pending = None, None
                continue

    return pts


def load_waypoint_csv(path: Path) -> List[Tuple[float, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")

        lat_key = find_column(reader.fieldnames, ["latitude_deg", "latitude", "lat"])
        lon_key = find_column(reader.fieldnames, ["longitude_deg", "longitude", "lon", "lng"])
        if lat_key is None or lon_key is None:
            raise RuntimeError(
                f"Could not find latitude/longitude columns in {path}. Headers: {reader.fieldnames}"
            )

        pts: List[Tuple[float, float]] = []
        for row in reader:
            try:
                lat = float(row[lat_key])
                lon = float(row[lon_key])
            except Exception:
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            if abs(lat) > 90.0 or abs(lon) > 180.0:
                continue
            pts.append((lat, lon))
        return pts


def find_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {name.lower().strip(): name for name in fieldnames}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


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


def remove_consecutive_duplicates_xy(
    xy: np.ndarray,
    t_ns: Optional[np.ndarray] = None,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if xy.shape[0] <= 1:
        return xy, t_ns

    keep = [0]
    for i in range(1, xy.shape[0]):
        if np.linalg.norm(xy[i] - xy[i - 1]) > eps:
            keep.append(i)

    k = np.asarray(keep, dtype=int)
    t_out = t_ns[k] if t_ns is not None else None
    return xy[k], t_out


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


def path_mean_distance(a_xy: np.ndarray, b_xy: np.ndarray, n_samples: int) -> float:
    if a_xy.shape[0] < 2 or b_xy.shape[0] < 2:
        return float("inf")
    a = resample_by_arclength(a_xy, n_samples)
    b = resample_by_arclength(b_xy, n_samples)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def latlon_to_local_xy_with_origin(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = math.radians(lat0_deg)
    lon0_rad = math.radians(lon0_deg)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    x = EARTH_RADIUS_M * dlon * math.cos(lat0_rad)
    y = EARTH_RADIUS_M * dlat
    return x, y


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


def load_gps_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"GT CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")

        lat_key = find_column(reader.fieldnames, ["latitude_deg", "latitude", "lat"])
        lon_key = find_column(reader.fieldnames, ["longitude_deg", "longitude", "lon", "lng"])
        t_key = find_column(reader.fieldnames, ["t_ns", "timestamp_ns", "stamp_ns"])
        if lat_key is None or lon_key is None:
            raise RuntimeError(
                f"Could not find latitude/longitude columns in {path}. Headers: {reader.fieldnames}"
            )

        lat_vals: List[float] = []
        lon_vals: List[float] = []
        t_vals: List[int] = []

        for row in reader:
            try:
                lat = float(row[lat_key])
                lon = float(row[lon_key])
                t_ns = int(float(row[t_key])) if t_key and row.get(t_key, "") != "" else len(lat_vals)
            except Exception:
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            if abs(lat) > 90.0 or abs(lon) > 180.0:
                continue
            lat_vals.append(lat)
            lon_vals.append(lon)
            t_vals.append(t_ns)

    if len(lat_vals) < 2:
        raise RuntimeError(f"Not enough valid GPS rows in {path}")

    lat = np.asarray(lat_vals, dtype=float)
    lon = np.asarray(lon_vals, dtype=float)
    t_ns = np.asarray(t_vals, dtype=np.int64)

    idx = np.argsort(t_ns, kind="stable")
    lat = lat[idx]
    lon = lon[idx]
    t_ns = t_ns[idx]

    lat, lon, t_ns = remove_consecutive_duplicates_latlon(lat, lon, t_ns=t_ns)
    if lat.size < 2:
        raise RuntimeError(f"Not enough non-duplicate GPS rows in {path}")

    return {
        "source": str(path),
        "t_ns": t_ns,
        "lat": lat,
        "lon": lon,
    }


def read_bag_gps_latlon(bag_dir: Path, gps_topic: str) -> Dict[str, Any]:
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag folder does not exist: {bag_dir}")

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

    if gps_topic not in topic_type_map:
        raise RuntimeError(f"Topic {gps_topic} not found in {bag_dir}")

    try:
        if hasattr(rosbag2_py, "StorageFilter"):
            reader.set_filter(rosbag2_py.StorageFilter(topics=[gps_topic]))
    except Exception:
        pass

    msg_type = get_message(topic_type_map[gps_topic])

    t_list: List[int] = []
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
        ll = extract_navsatfix_latlon(msg)
        if ll is None:
            continue
        lat, lon = ll
        t_list.append(int(t_ns))
        lat_list.append(lat)
        lon_list.append(lon)

    if len(t_list) < 2:
        raise RuntimeError(f"Not enough valid {gps_topic} samples in {bag_dir}")

    t_ns = np.asarray(t_list, dtype=np.int64)
    lat = np.asarray(lat_list, dtype=float)
    lon = np.asarray(lon_list, dtype=float)

    idx = np.argsort(t_ns, kind="stable")
    t_ns = t_ns[idx]
    lat = lat[idx]
    lon = lon[idx]
    lat, lon, t_ns = remove_consecutive_duplicates_latlon(lat, lon, t_ns=t_ns)

    if lat.size < 2:
        raise RuntimeError(f"Not enough non-duplicate GPS samples in {bag_dir}")

    return {
        "source": str(bag_dir),
        "t_ns": t_ns,
        "lat": lat,
        "lon": lon,
    }


def project_trace_to_common_origin(trace: Dict[str, Any], lat0: float, lon0: float) -> Dict[str, Any]:
    x, y = latlon_to_local_xy_with_origin(trace["lat"], trace["lon"], lat0, lon0)
    xy = np.column_stack((x, y))
    xy, t_ns = remove_consecutive_duplicates_xy(xy, t_ns=trace.get("t_ns"))
    return {
        **trace,
        "t_ns": t_ns,
        "xy": xy,
        "length_m": path_length(xy),
        "points": int(xy.shape[0]),
    }


def collect_gt_csv_paths(glob_expr: str) -> List[Path]:
    paths = sorted((Path(p) for p in glob.glob(glob_expr)), key=lambda p: natural_sort_key(str(p)))
    if not paths:
        raise RuntimeError(f"No GT CSVs matched: {glob_expr}")
    return paths


def find_sim_bag_dirs(sim_parent_dir: Path, sim_glob: str) -> List[Path]:
    if not sim_parent_dir.exists():
        raise FileNotFoundError(f"Simulation parent dir does not exist: {sim_parent_dir}")

    dirs = sorted(
        [p for p in sim_parent_dir.glob(sim_glob) if p.is_dir()],
        key=lambda p: natural_sort_key(p.name),
    )
    valid_dirs: List[Path] = []
    for p in dirs:
        has_metadata = (p / "metadata.yaml").exists()
        has_mcap = any(x.is_file() for x in p.glob("*.mcap"))
        if has_metadata or has_mcap:
            valid_dirs.append(p)

    if not valid_dirs:
        raise RuntimeError(f"No bag folders matched inside {sim_parent_dir} with glob '{sim_glob}'")
    return valid_dirs


def build_gt_reference(
    gt_csv_paths: Sequence[Path],
    waypoint_ref: Dict[str, Any],
    n_samples: int,
) -> Dict[str, Any]:
    projected: List[Dict[str, Any]] = []
    waypoint_xy = waypoint_ref["xy"]

    for p in gt_csv_paths:
        raw = load_gps_csv(p)
        tr = project_trace_to_common_origin(raw, waypoint_ref["lat0_deg"], waypoint_ref["lon0_deg"])

        direct = path_mean_distance(tr["xy"], waypoint_xy, n_samples)
        rev_xy = tr["xy"][::-1].copy()
        rev = path_mean_distance(rev_xy, waypoint_xy, n_samples)
        if rev < direct:
            tr["xy"] = rev_xy
            tr["was_reversed"] = True
        else:
            tr["was_reversed"] = False

        tr["length_m"] = path_length(tr["xy"])
        tr["points"] = int(tr["xy"].shape[0])
        projected.append(tr)

    resampled = [resample_by_arclength(tr["xy"], n_samples) for tr in projected]
    mean_xy = np.mean(np.stack(resampled, axis=0), axis=0)

    return {
        "gt_traces": projected,
        "gt_mean_xy": mean_xy,
        "gt_mean_length_m": path_length(mean_xy),
    }


def project_points_to_polyline(points_xy: np.ndarray, ref_xy: np.ndarray) -> Dict[str, np.ndarray]:
    n_pts = points_xy.shape[0]
    n_ref = ref_xy.shape[0]
    if n_pts == 0 or n_ref < 2:
        return {
            "dist": np.array([], dtype=float),
            "s_hat": np.array([], dtype=float),
        }

    ref_seg = ref_xy[1:] - ref_xy[:-1]
    ref_seg_len2 = np.sum(ref_seg * ref_seg, axis=1)
    ref_seg_len = np.sqrt(np.maximum(ref_seg_len2, 1e-12))
    s_ref = cumulative_arc_length(ref_xy)

    dist_out = np.zeros((n_pts,), dtype=float)
    s_out = np.zeros((n_pts,), dtype=float)

    for i in range(n_pts):
        p = points_xy[i]
        best_dist2 = float("inf")
        best_s = 0.0

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

        dist_out[i] = math.sqrt(best_dist2)
        s_out[i] = best_s

    return {
        "dist": dist_out,
        "s_hat": s_out,
    }



def align_trace_start_to_reference(trace_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    if trace_xy.shape[0] == 0 or ref_xy.shape[0] == 0:
        return trace_xy.copy()
    delta = ref_xy[0] - trace_xy[0]
    return trace_xy + delta


def compute_curve_metrics_vs_reference(
    trace_xy: np.ndarray,
    ref_xy: np.ndarray,
    n_samples: int,
) -> Dict[str, float]:
    if trace_xy.shape[0] < 2 or ref_xy.shape[0] < 2:
        return {
            "mean_lateral_m": float("nan"),
            "rmse_lateral_m": float("nan"),
            "p95_lateral_m": float("nan"),
            "max_lateral_m": float("nan"),
            "progress": float("nan"),
            "pointwise_rmse_m": float("nan"),
            "pointwise_mean_m": float("nan"),
        }

    proj = project_points_to_polyline(trace_xy, ref_xy)
    dist = proj["dist"]
    ref_len = max(path_length(ref_xy), 1e-9)

    a = resample_by_arclength(trace_xy, n_samples)
    b = resample_by_arclength(ref_xy, n_samples)
    pointwise = np.linalg.norm(a - b, axis=1)

    return {
        "mean_lateral_m": float(np.mean(dist)) if dist.size else float("nan"),
        "rmse_lateral_m": float(np.sqrt(np.mean(dist ** 2))) if dist.size else float("nan"),
        "p95_lateral_m": float(np.percentile(dist, 95.0)) if dist.size else float("nan"),
        "max_lateral_m": float(np.max(dist)) if dist.size else float("nan"),
        "progress": float(np.max(proj["s_hat"]) / ref_len) if proj["s_hat"].size else float("nan"),
        "pointwise_rmse_m": float(np.sqrt(np.mean(pointwise ** 2))) if pointwise.size else float("nan"),
        "pointwise_mean_m": float(np.mean(pointwise)) if pointwise.size else float("nan"),
    }


def compute_mean_curve(curves: Sequence[np.ndarray], n_samples: int) -> Optional[np.ndarray]:
    valid = [xy for xy in curves if xy.shape[0] >= 2]
    if not valid:
        return None
    stack = np.stack([resample_by_arclength(xy, n_samples) for xy in valid], axis=0)
    return np.mean(stack, axis=0)


def choose_medoid_curve(curves: Sequence[np.ndarray], n_samples: int) -> Tuple[Optional[np.ndarray], Optional[int]]:
    valid_pairs = [(idx, xy) for idx, xy in enumerate(curves) if xy.shape[0] >= 2]
    if not valid_pairs:
        return None, None
    if len(valid_pairs) == 1:
        return valid_pairs[0][1], valid_pairs[0][0]

    resampled = [resample_by_arclength(xy, n_samples) for _, xy in valid_pairs]
    score = np.zeros((len(resampled),), dtype=float)

    for i in range(len(resampled)):
        for j in range(len(resampled)):
            if i == j:
                continue
            d = np.linalg.norm(resampled[i] - resampled[j], axis=1)
            score[i] += float(np.mean(d))

    best_local_idx = int(np.argmin(score))
    orig_idx, best_curve = valid_pairs[best_local_idx]
    return best_curve, orig_idx


def compute_waypoint_hit_metrics(
    trace_xy: np.ndarray,
    waypoint_xy: np.ndarray,
    thresholds_m: Sequence[float],
) -> Dict[str, float]:
    if trace_xy.shape[0] < 2 or waypoint_xy.shape[0] < 2:
        out: Dict[str, float] = {
            "wp_mean_min_dist_m": float("nan"),
            "wp_max_min_dist_m": float("nan"),
            "trace_progress_vs_wp": float("nan"),
        }
        for th in thresholds_m:
            out[f"wp_hits_within_{int(round(th))}m"] = float("nan")
        return out

    wp_to_trace = project_points_to_polyline(waypoint_xy, trace_xy)
    trace_to_wp = project_points_to_polyline(trace_xy, waypoint_xy)
    wp_len = max(path_length(waypoint_xy), 1e-9)

    out = {
        "wp_mean_min_dist_m": float(np.mean(wp_to_trace["dist"])),
        "wp_max_min_dist_m": float(np.max(wp_to_trace["dist"])),
        "trace_progress_vs_wp": float(np.max(trace_to_wp["s_hat"]) / wp_len) if trace_to_wp["s_hat"].size else float("nan"),
    }
    for th in thresholds_m:
        out[f"wp_hits_within_{int(round(th))}m"] = float(np.sum(wp_to_trace["dist"] <= th))
    return out


def make_candidate_variants(raw_xy: np.ndarray) -> List[Dict[str, Any]]:
    if raw_xy.shape[0] < 2:
        return [{"variant": "raw", "xy": raw_xy.copy()}]

    base_variants = [
        ("raw", raw_xy.copy()),
        ("flip_y", raw_xy * np.array([1.0, -1.0])),
        ("flip_x", raw_xy * np.array([-1.0, 1.0])),
        ("flip_xy", raw_xy * np.array([-1.0, -1.0])),
    ]

    out: List[Dict[str, Any]] = []
    for name, xy in base_variants:
        out.append({"variant": name, "xy": xy.copy()})
        out.append({"variant": f"{name}_reversed", "xy": xy[::-1].copy()})
    return out


def score_candidate(
    cand_xy: np.ndarray,
    waypoint_xy: np.ndarray,
    gt_mean_xy: np.ndarray,
    n_samples: int,
) -> Dict[str, float]:
    score_wp = path_mean_distance(cand_xy, waypoint_xy, n_samples)
    score_gt = path_mean_distance(cand_xy, gt_mean_xy, n_samples)
    return {
        "score_vs_waypoint_m": score_wp,
        "score_vs_gt_m": score_gt,
        "score_total_m": 0.5 * (score_wp + score_gt),
    }


def choose_best_candidate(
    raw_xy: np.ndarray,
    waypoint_xy: np.ndarray,
    gt_mean_xy: np.ndarray,
    n_samples: int,
) -> Dict[str, Any]:
    candidates = make_candidate_variants(raw_xy)
    best: Optional[Dict[str, Any]] = None
    for cand in candidates:
        scores = score_candidate(cand["xy"], waypoint_xy, gt_mean_xy, n_samples)
        merged = {**cand, **scores}
        if best is None or merged["score_total_m"] < best["score_total_m"]:
            best = merged
    assert best is not None
    return best


def process_one_run(
    bag_dir: Path,
    gps_topic: str,
    waypoint_ref: Dict[str, Any],
    gt_ref: Dict[str, Any],
    n_samples: int,
    thresholds_m: Sequence[float],
) -> Dict[str, Any]:
    raw = read_bag_gps_latlon(bag_dir, gps_topic)
    projected = project_trace_to_common_origin(raw, waypoint_ref["lat0_deg"], waypoint_ref["lon0_deg"])
    raw_xy = projected["xy"]

    raw_scores = score_candidate(raw_xy, waypoint_ref["xy"], gt_ref["gt_mean_xy"], n_samples)
    raw_hit = compute_waypoint_hit_metrics(raw_xy, waypoint_ref["xy"], thresholds_m)

    best = choose_best_candidate(raw_xy, waypoint_ref["xy"], gt_ref["gt_mean_xy"], n_samples)
    best_hit = compute_waypoint_hit_metrics(best["xy"], waypoint_ref["xy"], thresholds_m)

    aligned_xy = align_trace_start_to_reference(best["xy"], gt_ref["gt_mean_xy"])
    aligned_hit = compute_waypoint_hit_metrics(aligned_xy, waypoint_ref["xy"], thresholds_m)

    best_gt_metrics = compute_curve_metrics_vs_reference(best["xy"], gt_ref["gt_mean_xy"], n_samples)
    aligned_gt_metrics = compute_curve_metrics_vs_reference(aligned_xy, gt_ref["gt_mean_xy"], n_samples)

    return {
        "run_name": bag_dir.name,
        "bag_dir": str(bag_dir),
        "raw_xy": raw_xy,
        "best_xy": best["xy"],
        "aligned_xy": aligned_xy,
        "raw_variant": "raw",
        "best_variant": best["variant"],
        "points": int(raw_xy.shape[0]),
        "length_m": path_length(raw_xy),
        **{f"raw_{k}": v for k, v in raw_scores.items()},
        **{f"best_{k}": v for k, v in best.items() if k.startswith("score_")},
        **{f"raw_{k}": v for k, v in raw_hit.items()},
        **{f"best_{k}": v for k, v in best_hit.items()},
        **{f"best_gt_{k}": v for k, v in best_gt_metrics.items()},
        **{f"start_aligned_{k}": v for k, v in aligned_hit.items()},
        **{f"start_aligned_gt_{k}": v for k, v in aligned_gt_metrics.items()},
    }


def set_equal_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_waypoints_vs_gt(
    path: Path,
    waypoint_ref: Dict[str, Any],
    gt_ref: Dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    wp_xy = waypoint_ref["xy"]
    ax.plot(wp_xy[:, 0], wp_xy[:, 1], "k--", linewidth=2.0, label="Configured waypoints")

    for tr in gt_ref["gt_traces"]:
        xy = tr["xy"]
        ax.plot(xy[:, 0], xy[:, 1], color="0.7", linewidth=1.0, alpha=0.9)

    gt_mean_xy = gt_ref["gt_mean_xy"]
    ax.plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], color="tab:green", linewidth=2.5, label="GT mean raw GPS")

    ax.scatter([wp_xy[0, 0]], [wp_xy[0, 1]], marker="o")
    ax.scatter([wp_xy[-1, 0]], [wp_xy[-1, 1]], marker="x")

    ax.set_title("Configured waypoints vs GT raw GPS, common geodetic origin")
    set_equal_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_waypoints_vs_sim(
    path: Path,
    waypoint_ref: Dict[str, Any],
    gt_ref: Dict[str, Any],
    sim_results: Sequence[Dict[str, Any]],
    use_best_variant: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    wp_xy = waypoint_ref["xy"]
    gt_mean_xy = gt_ref["gt_mean_xy"]

    ax.plot(wp_xy[:, 0], wp_xy[:, 1], "k--", linewidth=2.0, label="Configured waypoints")
    ax.plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], color="tab:green", linewidth=2.0, label="GT mean raw GPS")

    for result in sim_results:
        xy = result["best_xy"] if use_best_variant else result["raw_xy"]
        ax.plot(xy[:, 0], xy[:, 1], color="0.7", linewidth=1.0, alpha=0.9)

    if sim_results:
        mean_inputs = [r["best_xy"] if use_best_variant else r["raw_xy"] for r in sim_results]
        sim_mean = np.mean(np.stack([resample_by_arclength(xy, 300) for xy in mean_inputs], axis=0), axis=0)
        ax.plot(sim_mean[:, 0], sim_mean[:, 1], color="tab:red", linewidth=2.2,
                label="Simulation mean best-variant" if use_best_variant else "Simulation mean raw")

    ax.scatter([wp_xy[0, 0]], [wp_xy[0, 1]], marker="o")
    ax.scatter([wp_xy[-1, 0]], [wp_xy[-1, 1]], marker="x")

    ax.set_title(
        "Configured waypoints vs simulation raw GPS, best candidate variant"
        if use_best_variant
        else "Configured waypoints vs simulation raw GPS, raw as published"
    )
    set_equal_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)



def plot_gt_vs_sim_start_aligned(
    path: Path,
    gt_ref: Dict[str, Any],
    sim_results: Sequence[Dict[str, Any]],
    n_samples: int,
    waypoint_ref: Optional[Dict[str, Any]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    if waypoint_ref is not None:
        wp_xy = waypoint_ref["xy"]
        ax.plot(wp_xy[:, 0], wp_xy[:, 1], "k--", linewidth=2.0, label="Configured waypoints")
        ax.scatter([wp_xy[0, 0]], [wp_xy[0, 1]], marker="o")
        ax.scatter([wp_xy[-1, 0]], [wp_xy[-1, 1]], marker="x")

    gt_mean_xy = gt_ref["gt_mean_xy"]
    ax.plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], color="tab:green", linewidth=2.2, label="GT mean trajectory")

    aligned_inputs: List[np.ndarray] = []
    for result in sim_results:
        xy = result["aligned_xy"]
        if xy.shape[0] < 2:
            continue
        aligned_inputs.append(xy)
        ax.plot(xy[:, 0], xy[:, 1], color="0.75", linewidth=1.0, alpha=0.9)

    sim_rep, _ = choose_medoid_curve(aligned_inputs, n_samples)
    if sim_rep is not None:
        ax.plot(sim_rep[:, 0], sim_rep[:, 1], color="tab:red", linewidth=2.6,
                label="Simulation representative trajectory")

    if waypoint_ref is not None:
        ax.set_title("Configured waypoints vs GT and representative simulation trajectory")
    else:
        ax.set_title("GT mean trajectory vs representative simulation trajectory")

    set_equal_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_run_debug(
    path: Path,
    waypoint_ref: Dict[str, Any],
    gt_ref: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    wp_xy = waypoint_ref["xy"]
    gt_mean_xy = gt_ref["gt_mean_xy"]

    axes[0].plot(wp_xy[:, 0], wp_xy[:, 1], "k--", linewidth=2.0, label="Configured waypoints")
    axes[0].plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], color="tab:green", linewidth=2.0, label="GT mean raw GPS")
    axes[0].plot(result["raw_xy"][:, 0], result["raw_xy"][:, 1], color="tab:orange", linewidth=2.0, label="Simulation raw")
    axes[0].set_title("Raw as published")
    set_equal_axes(axes[0])
    axes[0].legend(fontsize=8)

    axes[1].plot(wp_xy[:, 0], wp_xy[:, 1], "k--", linewidth=2.0, label="Configured waypoints")
    axes[1].plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], color="tab:green", linewidth=2.0, label="GT mean raw GPS")
    axes[1].plot(result["best_xy"][:, 0], result["best_xy"][:, 1], color="tab:red", linewidth=2.0,
                 label=f"Best variant: {result['best_variant']}")
    axes[1].set_title("Best candidate variant")
    set_equal_axes(axes[1])
    axes[1].legend(fontsize=8)

    axes[2].plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], color="tab:green", linewidth=2.0, label="GT mean raw GPS")
    axes[2].plot(result["aligned_xy"][:, 0], result["aligned_xy"][:, 1], color="tab:purple", linewidth=2.0,
                 label="Best variant + start aligned")
    axes[2].set_title("Shape comparison after start alignment")
    set_equal_axes(axes[2])
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"{result['run_name']} | aligned RMSE_lat={result['start_aligned_gt_rmse_lateral_m']:.2f} m, "
        f"P95={result['start_aligned_gt_p95_lateral_m']:.2f} m"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            normalized = {k: row.get(k, "") for k in fieldnames}
            w.writerow(normalized)


def summarize_gt_against_waypoints(
    waypoint_ref: Dict[str, Any],
    gt_ref: Dict[str, Any],
    thresholds_m: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    gt_mean_hit = compute_waypoint_hit_metrics(gt_ref["gt_mean_xy"], waypoint_ref["xy"], thresholds_m)
    rows.append(
        {
            "kind": "gt_mean",
            "source": "GT mean raw GPS",
            "points": int(gt_ref["gt_mean_xy"].shape[0]),
            "length_m": f"{path_length(gt_ref['gt_mean_xy']):.6f}",
            **{k: (f"{v:.6f}" if isinstance(v, float) and math.isfinite(v) else v) for k, v in gt_mean_hit.items()},
        }
    )

    for tr in gt_ref["gt_traces"]:
        hit = compute_waypoint_hit_metrics(tr["xy"], waypoint_ref["xy"], thresholds_m)
        rows.append(
            {
                "kind": "gt_trace",
                "source": tr["source"],
                "points": tr["points"],
                "length_m": f"{tr['length_m']:.6f}",
                "was_reversed": int(bool(tr.get("was_reversed", False))),
                **{k: (f"{v:.6f}" if isinstance(v, float) and math.isfinite(v) else v) for k, v in hit.items()},
            }
        )
    return rows


def build_sim_debug_rows(sim_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in sim_results:
        row: Dict[str, Any] = {
            "run_name": r["run_name"],
            "bag_dir": r["bag_dir"],
            "points": r["points"],
            "length_m": f"{r['length_m']:.6f}",
            "best_variant": r["best_variant"],
        }
        for k, v in r.items():
            if not (k.startswith("raw_") or k.startswith("best_") or k.startswith("start_aligned_")):
                continue
            if isinstance(v, float):
                row[k] = f"{v:.6f}" if math.isfinite(v) else "nan"
            else:
                row[k] = v
        rows.append(row)
    return rows


def save_summary_txt(
    path: Path,
    waypoint_ref: Dict[str, Any],
    gt_ref: Dict[str, Any],
    sim_results: Sequence[Dict[str, Any]],
    thresholds_m: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variant_counter = Counter(r["best_variant"] for r in sim_results)

    gt_hit = compute_waypoint_hit_metrics(gt_ref["gt_mean_xy"], waypoint_ref["xy"], thresholds_m)

    def finite_stats(key: str) -> Tuple[float, float]:
        vals = [float(r[key]) for r in sim_results if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]
        if not vals:
            return float("nan"), float("nan")
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr, ddof=0))

    mean_rmse, std_rmse = finite_stats("start_aligned_gt_rmse_lateral_m")
    mean_p95, std_p95 = finite_stats("start_aligned_gt_p95_lateral_m")
    mean_pw, std_pw = finite_stats("start_aligned_gt_pointwise_rmse_m")
    mean_prog, std_prog = finite_stats("start_aligned_gt_progress")

    with path.open("w", encoding="utf-8") as f:
        f.write("Raw GPS waypoint alignment debug summary\n")
        f.write(f"Waypoint source: {waypoint_ref['source']}\n")
        f.write(f"Waypoint points: {waypoint_ref['points']}\n")
        f.write(f"Common origin lat0_deg: {waypoint_ref['lat0_deg']:.10f}\n")
        f.write(f"Common origin lon0_deg: {waypoint_ref['lon0_deg']:.10f}\n")
        f.write(f"Waypoint path length [m]: {waypoint_ref['length_m']:.6f}\n\n")

        f.write("GT mean vs configured waypoints\n")
        f.write(f"  mean min waypoint distance [m]: {gt_hit['wp_mean_min_dist_m']:.6f}\n")
        f.write(f"  max min waypoint distance [m]: {gt_hit['wp_max_min_dist_m']:.6f}\n")
        f.write(f"  trace progress vs waypoint path: {gt_hit['trace_progress_vs_wp']:.6f}\n")
        for th in thresholds_m:
            key = f"wp_hits_within_{int(round(th))}m"
            f.write(f"  waypoints within {th:.1f} m: {int(gt_hit[key])}/{waypoint_ref['points']}\n")
        f.write("\n")

        f.write("Simulation best-variant counts\n")
        if variant_counter:
            for name, count in variant_counter.most_common():
                f.write(f"  {name}: {count}\n")
        else:
            f.write("  none\n")
        f.write("\n")

        f.write("Start-aligned shape metrics vs GT mean (best variant after start alignment)\n")
        f.write(f"  RMSE lateral [m]: {mean_rmse:.6f} [{std_rmse:.6f}]\n")
        f.write(f"  P95 lateral [m]: {mean_p95:.6f} [{std_p95:.6f}]\n")
        f.write(f"  Pointwise RMSE [m]: {mean_pw:.6f} [{std_pw:.6f}]\n")
        f.write(f"  Progress vs GT: {mean_prog:.6f} [{std_prog:.6f}]\n\n")

        f.write("Per-run highlights\n")
        for r in sim_results:
            f.write(
                f"  - {r['run_name']}: best_variant={r['best_variant']}, "
                f"raw_score_total_m={r['raw_score_total_m']:.3f}, "
                f"best_score_total_m={r['best_score_total_m']:.3f}, "
                f"raw_wp_mean_min_dist_m={r['raw_wp_mean_min_dist_m']:.3f}, "
                f"best_wp_mean_min_dist_m={r['best_wp_mean_min_dist_m']:.3f}, "
                f"aligned_gt_rmse_lateral_m={r['start_aligned_gt_rmse_lateral_m']:.3f}, "
                f"aligned_gt_p95_lateral_m={r['start_aligned_gt_p95_lateral_m']:.3f}\n"
            )

        if variant_counter:
            dominant = variant_counter.most_common(1)[0][0]
            f.write("\nInterpretation hint\n")
            if dominant.startswith("flip_y"):
                f.write(
                    "  The dominant best variant is flip_y. "
                    "This strongly suggests the simulation GPS path is mirrored in the north/south axis "
                    "relative to the configured geodetic waypoint route.\n"
                )
            elif dominant.startswith("flip_x"):
                f.write(
                    "  The dominant best variant is flip_x. "
                    "This suggests an east/west sign inversion in the simulated GPS path.\n"
                )
            elif dominant.startswith("flip_xy"):
                f.write(
                    "  The dominant best variant is flip_xy. "
                    "This suggests a 180-degree mirrored sign pattern relative to the configured route.\n"
                )
            elif dominant.endswith("reversed"):
                f.write(
                    "  The dominant best variant is a reversed ordering, without a dominant axis flip. "
                    "This suggests the same corridor may be used, but the sequence is reversed.\n"
                )
            else:
                f.write(
                    "  The dominant best variant is raw. "
                    "That means the main problem is probably not a simple sign flip in the GPS publisher.\n"
                )


def main() -> int:
    args = parse_args()

    sim_parent_dir = Path(args.sim_parent_dir).expanduser().resolve()
    waypoint_file = Path(args.waypoint_file).expanduser().resolve()
    out_dir = (Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (sim_parent_dir / "Output_charts_diagnostic").resolve())
    thresholds_m = parse_thresholds(args.thresholds_m)

    waypoint_ref = load_waypoint_file(waypoint_file)
    gt_csv_paths = collect_gt_csv_paths(args.gt_csv_glob)
    sim_bag_dirs = find_sim_bag_dirs(sim_parent_dir, args.sim_glob)
    gt_ref = build_gt_reference(gt_csv_paths, waypoint_ref, args.resample_samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_plot_dir = out_dir / "per_run_plots"
    if args.save_per_run_plots:
        per_run_plot_dir.mkdir(parents=True, exist_ok=True)

    sim_results: List[Dict[str, Any]] = []

    print("[INFO] Common origin taken from configured waypoint file")
    print(f"  lat0_deg = {waypoint_ref['lat0_deg']:.10f}")
    print(f"  lon0_deg = {waypoint_ref['lon0_deg']:.10f}")
    print(f"  Waypoint points = {waypoint_ref['points']}")
    print(f"  GT traces found = {len(gt_csv_paths)}")
    print(f"  Simulation bags found = {len(sim_bag_dirs)}")
    print(f"[INFO] Charts output dir: {out_dir}")

    for bag_dir in sim_bag_dirs:
        try:
            result = process_one_run(
                bag_dir=bag_dir,
                gps_topic=args.gps_topic,
                waypoint_ref=waypoint_ref,
                gt_ref=gt_ref,
                n_samples=args.resample_samples,
                thresholds_m=thresholds_m,
            )
        except Exception as exc:
            print(f"[WARN] Skipping {bag_dir.name}: {exc}")
            continue

        sim_results.append(result)
        print(
            f"[OK] {result['run_name']}: "
            f"best_variant={result['best_variant']}, "
            f"raw_score_total_m={result['raw_score_total_m']:.3f}, "
            f"best_score_total_m={result['best_score_total_m']:.3f}, "
            f"raw_wp_mean_min_dist_m={result['raw_wp_mean_min_dist_m']:.3f}, "
            f"best_wp_mean_min_dist_m={result['best_wp_mean_min_dist_m']:.3f}, "
            f"aligned_gt_rmse_lateral_m={result['start_aligned_gt_rmse_lateral_m']:.3f}"
        )

        if args.save_per_run_plots:
            plot_per_run_debug(
                path=per_run_plot_dir / f"{result['run_name']}_debug.png",
                waypoint_ref=waypoint_ref,
                gt_ref=gt_ref,
                result=result,
            )

    gt_rows = summarize_gt_against_waypoints(waypoint_ref, gt_ref, thresholds_m)
    sim_rows = build_sim_debug_rows(sim_results)

    gt_csv_out = out_dir / "gt_waypoint_debug.csv"
    sim_csv_out = out_dir / "sim_waypoint_debug.csv"
    summary_txt = out_dir / "waypoint_alignment_summary.txt"
    plot_gt = out_dir / "debug_waypoints_vs_gt_common_origin.png"
    plot_sim_raw = out_dir / "debug_waypoints_vs_sim_raw_common_origin.png"
    plot_sim_best = out_dir / "debug_waypoints_vs_sim_best_variant.png"
    plot_sim_best_aligned = out_dir / "debug_waypoints_vs_sim_best_variant_start_aligned.png"
    plot_paper_aligned = out_dir / "paper_gt_vs_sim_best_variant_start_aligned.png"

    save_csv(gt_csv_out, gt_rows)
    save_csv(sim_csv_out, sim_rows)
    save_summary_txt(summary_txt, waypoint_ref, gt_ref, sim_results, thresholds_m)
    plot_waypoints_vs_gt(plot_gt, waypoint_ref, gt_ref)
    plot_waypoints_vs_sim(plot_sim_raw, waypoint_ref, gt_ref, sim_results, use_best_variant=False)
    plot_waypoints_vs_sim(plot_sim_best, waypoint_ref, gt_ref, sim_results, use_best_variant=True)
    plot_gt_vs_sim_start_aligned(
        plot_sim_best_aligned,
        gt_ref,
        sim_results,
        args.resample_samples,
        waypoint_ref=waypoint_ref,
    )
    plot_gt_vs_sim_start_aligned(
        plot_paper_aligned,
        gt_ref,
        sim_results,
        args.resample_samples,
        waypoint_ref=None,
    )

    print("\n[OK] Outputs generated:")
    print(f"  - {plot_gt}")
    print(f"  - {plot_sim_raw}")
    print(f"  - {plot_sim_best}")
    print(f"  - {plot_sim_best_aligned}")
    print(f"  - {plot_paper_aligned}")
    print(f"  - {gt_csv_out}")
    print(f"  - {sim_csv_out}")
    print(f"  - {summary_txt}")
    if args.save_per_run_plots:
        print(f"  - {per_run_plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
