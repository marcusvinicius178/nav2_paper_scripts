#!/usr/bin/env python3
"""
Compare ground-truth raw GPS traces against simulation raw GPS traces in a single,
common geodetic frame.

Purpose
-------
This script is meant for the "human-likeness" layer of the Nav2 paper:
- primary benchmark metrics remain planner-based elsewhere
- this script adds a secondary comparison based on raw /gps/fix

What it does
------------
1) Loads 1..N GT raw GPS CSV files (typically the 2 GT bags of one scenario+speed).
2) Projects ALL GT traces to the SAME local XY frame using one shared lat0/lon0.
3) Builds a GT mean raw-GPS trajectory in that common frame.
4) Reads ALL simulation run bags inside a parent directory (e.g. Navfn_mppi_1..10).
5) Extracts /gps/fix from each run, projecting all runs to the SAME shared lat0/lon0.
6) Aligns trajectory direction automatically (reverses only if needed).
7) Computes secondary metrics against the GT mean raw-GPS reference.
8) Generates:
   - one aggregate overlay figure for the whole batch
   - optional one figure per run
   - one CSV with per-run metrics
   - one CSV with the GT mean raw-GPS trajectory in the common frame

Important methodological note
-----------------------------
Metrics are computed in the common geodetic frame (same lat0/lon0 for all traces).
No rigid transform is used for the formal metrics here.

A visual-only start-aligned plot is also generated to help qualitative inspection,
but that start alignment is NOT used as the formal metric basis.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


EARTH_RADIUS_M = 6378137.0
DEFAULT_GPS_TOPIC = "/gps/fix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GT raw GPS against simulation raw GPS in one common lat0/lon0 frame."
    )
    parser.add_argument(
        "--sim-parent-dir",
        required=True,
        help="Parent directory containing the run bag folders (e.g. .../NAVFN_MPPI_Vel_20)",
    )
    parser.add_argument(
        "--sim-glob",
        default="*",
        help="Glob used inside --sim-parent-dir to find run folders (e.g. 'Navfn_mppi_*')",
    )
    parser.add_argument(
        "--gt-csv-glob",
        required=True,
        help=(
            "Glob for GT raw GPS CSV files, usually ending with gps_fix_latlon.csv. "
            "Example: '/.../1-Scenario.../*vel_20*/gps_fix_latlon.csv'"
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for figures and CSV files",
    )
    parser.add_argument(
        "--gps-topic",
        default=DEFAULT_GPS_TOPIC,
        help="GPS topic to read from each simulation bag",
    )
    parser.add_argument(
        "--resample-samples",
        type=int,
        default=300,
        help="Number of arclength samples for GT mean and pointwise comparisons",
    )
    parser.add_argument(
        "--save-per-run-plots",
        action="store_true",
        help="Also generate per-run figures in out_dir/per_run_plots",
    )
    return parser.parse_args()


def natural_sort_key(text: str) -> List[Any]:
    parts = re.split(r"(\d+)", text)
    out: List[Any] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def resolve_bag_uri_for_mcap(bag_dir: Path) -> Path:
    mcap_files = sorted([p for p in bag_dir.glob("*.mcap") if p.is_file()])
    if not mcap_files:
        return bag_dir

    for p in mcap_files:
        if p.name.endswith("_0.mcap"):
            return p
    return mcap_files[0]


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
    lat0_rad = math.radians(float(lat0_deg))
    lon0_rad = math.radians(float(lon0_deg))

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    x = EARTH_RADIUS_M * dlon * math.cos(lat0_rad)
    y = EARTH_RADIUS_M * dlat
    return x, y


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


def maybe_reverse_to_match_reference(
    xy: np.ndarray,
    ref_xy: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, bool]:
    if xy.shape[0] < 2 or ref_xy.shape[0] < 2:
        return xy, False

    d_direct = path_mean_distance(xy, ref_xy, n_samples)
    d_rev = path_mean_distance(xy[::-1].copy(), ref_xy, n_samples)
    if d_rev < d_direct:
        return xy[::-1].copy(), True
    return xy, False


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


def load_gps_csv(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"GT CSV is empty: {csv_path}")

    lat_key = None
    lon_key = None
    for candidate in ["latitude_deg", "latitude", "lat_deg", "lat"]:
        if candidate in rows[0]:
            lat_key = candidate
            break
    for candidate in ["longitude_deg", "longitude", "lon_deg", "lon"]:
        if candidate in rows[0]:
            lon_key = candidate
            break

    if lat_key is None or lon_key is None:
        raise RuntimeError(f"Could not find latitude/longitude columns in: {csv_path}")

    t_list: List[int] = []
    lat_list: List[float] = []
    lon_list: List[float] = []

    for row in rows:
        try:
            lat = float(row[lat_key])
            lon = float(row[lon_key])
        except Exception:
            continue
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            continue

        t_value = 0
        if "t_ns" in row:
            try:
                t_value = int(float(row["t_ns"]))
            except Exception:
                t_value = len(t_list)
        else:
            t_value = len(t_list)

        t_list.append(t_value)
        lat_list.append(lat)
        lon_list.append(lon)

    if len(lat_list) < 2:
        raise RuntimeError(f"Not enough valid GPS points in: {csv_path}")

    t_ns = np.asarray(t_list, dtype=np.int64)
    lat = np.asarray(lat_list, dtype=float)
    lon = np.asarray(lon_list, dtype=float)

    idx = np.argsort(t_ns, kind="stable")
    t_ns = t_ns[idx]
    lat = lat[idx]
    lon = lon[idx]

    lat, lon, t_ns = remove_consecutive_duplicates_latlon(lat, lon, t_ns=t_ns)

    if lat.size < 2:
        raise RuntimeError(f"Not enough non-duplicate GPS points in: {csv_path}")

    return {
        "source": str(csv_path),
        "t_ns": t_ns,
        "lat": lat,
        "lon": lon,
    }


def get_rosbag_imports() -> Tuple[Any, Any, Any]:
    try:
        import rosbag2_py  # type: ignore
        from rclpy.serialization import deserialize_message  # type: ignore
        from rosidl_runtime_py.utilities import get_message  # type: ignore
        return rosbag2_py, deserialize_message, get_message
    except Exception:
        print("ERROR: Could not import ROS 2 Python APIs.", file=sys.stderr)
        print("Run: source /opt/ros/jazzy/setup.bash", file=sys.stderr)
        raise


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


def read_bag_gps_latlon(bag_dir: Path, gps_topic: str) -> Dict[str, Any]:
    rosbag2_py, deserialize_message, get_message = get_rosbag_imports()

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
        raise RuntimeError(f"GPS topic not found in bag {bag_dir}: {gps_topic}")

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

    if len(lat_list) < 2:
        raise RuntimeError(f"Not enough valid GPS points in bag: {bag_dir}")

    t_ns = np.asarray(t_list, dtype=np.int64)
    lat = np.asarray(lat_list, dtype=float)
    lon = np.asarray(lon_list, dtype=float)

    idx = np.argsort(t_ns, kind="stable")
    t_ns = t_ns[idx]
    lat = lat[idx]
    lon = lon[idx]

    lat, lon, t_ns = remove_consecutive_duplicates_latlon(lat, lon, t_ns=t_ns)

    if lat.size < 2:
        raise RuntimeError(f"Not enough non-duplicate GPS points in bag: {bag_dir}")

    return {
        "source": str(bag_dir),
        "t_ns": t_ns,
        "lat": lat,
        "lon": lon,
    }


def project_trace_to_common_origin(trace: Dict[str, Any], lat0: float, lon0: float) -> Dict[str, Any]:
    x, y = latlon_to_local_xy_with_origin(trace["lat"], trace["lon"], lat0, lon0)
    xy = np.column_stack((x, y))
    return {
        **trace,
        "xy": xy,
        "length_m": path_length(xy),
        "points": int(xy.shape[0]),
    }


def build_gt_mean_reference(
    gt_csv_paths: Sequence[Path],
    n_samples: int,
) -> Dict[str, Any]:
    if not gt_csv_paths:
        raise RuntimeError("No GT CSVs were found.")

    raw_traces = [load_gps_csv(p) for p in gt_csv_paths]
    lat0 = float(raw_traces[0]["lat"][0])
    lon0 = float(raw_traces[0]["lon"][0])

    projected: List[Dict[str, Any]] = []
    ref_xy: Optional[np.ndarray] = None
    for raw in raw_traces:
        tr = project_trace_to_common_origin(raw, lat0, lon0)
        xy = tr["xy"]
        if ref_xy is None:
            ref_xy = xy
            tr["was_reversed"] = False
        else:
            xy_fixed, was_reversed = maybe_reverse_to_match_reference(xy, ref_xy, n_samples)
            tr["xy"] = xy_fixed
            tr["length_m"] = path_length(xy_fixed)
            tr["points"] = int(xy_fixed.shape[0])
            tr["was_reversed"] = was_reversed
        projected.append(tr)

    resampled = [resample_by_arclength(tr["xy"], n_samples) for tr in projected]
    stacked = np.stack(resampled, axis=0)
    mean_xy = np.mean(stacked, axis=0)
    std_xy = np.std(stacked, axis=0)

    return {
        "lat0_deg": lat0,
        "lon0_deg": lon0,
        "gt_traces": projected,
        "gt_mean_xy": mean_xy,
        "gt_std_xy": std_xy,
        "gt_mean_length_m": path_length(mean_xy),
    }


def compute_trace_mean(
    traces_xy: Sequence[np.ndarray],
    n_samples: int,
) -> Optional[np.ndarray]:
    if not traces_xy:
        return None
    resampled = [resample_by_arclength(xy, n_samples) for xy in traces_xy if xy.shape[0] >= 2]
    if not resampled:
        return None
    return np.mean(np.stack(resampled, axis=0), axis=0)


def start_align_xy(xy: np.ndarray, ref_start_xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return xy.copy()
    return xy + (ref_start_xy - xy[0])


def compute_metrics_vs_gt(
    sim_xy: np.ndarray,
    gt_mean_xy: np.ndarray,
    n_samples: int,
) -> Dict[str, float]:
    if sim_xy.shape[0] < 2 or gt_mean_xy.shape[0] < 2:
        return {
            "gt_pr": float("nan"),
            "rmse_lateral_m": float("nan"),
            "p95_lateral_m": float("nan"),
            "rmse_pointwise_common_m": float("nan"),
            "rmse_pointwise_start_aligned_m": float("nan"),
        }

    gt_len = max(path_length(gt_mean_xy), 1e-9)
    proj = project_points_to_polyline(sim_xy, gt_mean_xy)

    rmse_lateral = float(np.sqrt(np.mean(proj["dist"] ** 2))) if proj["dist"].size else float("nan")
    p95_lateral = float(np.percentile(proj["dist"], 95.0)) if proj["dist"].size else float("nan")
    gt_pr = float(np.max(proj["s_hat"]) / gt_len) if proj["s_hat"].size else float("nan")

    gt_rs = resample_by_arclength(gt_mean_xy, n_samples)
    sim_rs = resample_by_arclength(sim_xy, n_samples)
    rmse_pointwise_common = float(np.sqrt(np.mean(np.sum((sim_rs - gt_rs) ** 2, axis=1))))

    sim_start_aligned = start_align_xy(sim_xy, gt_mean_xy[0])
    sim_start_aligned_rs = resample_by_arclength(sim_start_aligned, n_samples)
    rmse_pointwise_start_aligned = float(
        np.sqrt(np.mean(np.sum((sim_start_aligned_rs - gt_rs) ** 2, axis=1)))
    )

    return {
        "gt_pr": gt_pr,
        "rmse_lateral_m": rmse_lateral,
        "p95_lateral_m": p95_lateral,
        "rmse_pointwise_common_m": rmse_pointwise_common,
        "rmse_pointwise_start_aligned_m": rmse_pointwise_start_aligned,
    }


def set_equal_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_aggregate_common_origin(
    path: Path,
    gt_ref: Dict[str, Any],
    sim_results: Sequence[Dict[str, Any]],
    n_samples: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    for gt_tr in gt_ref["gt_traces"]:
        xy = gt_tr["xy"]
        ax.plot(xy[:, 0], xy[:, 1], linestyle=":", linewidth=1.2, alpha=0.8)

    gt_mean_xy = gt_ref["gt_mean_xy"]
    ax.plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], linewidth=2.5, label="GT mean raw GPS")

    sim_mean_inputs: List[np.ndarray] = []
    for result in sim_results:
        xy = result["sim_xy_common"]
        ax.plot(xy[:, 0], xy[:, 1], color="0.7", linewidth=1.0, alpha=0.9)
        sim_mean_inputs.append(xy)

    sim_mean_xy = compute_trace_mean(sim_mean_inputs, n_samples)
    if sim_mean_xy is not None:
        ax.plot(sim_mean_xy[:, 0], sim_mean_xy[:, 1], linewidth=2.0, label="Simulation mean raw GPS")

    ax.scatter([gt_mean_xy[0, 0]], [gt_mean_xy[0, 1]], marker="o")
    ax.scatter([gt_mean_xy[-1, 0]], [gt_mean_xy[-1, 1]], marker="x")

    ax.set_title("GT mean raw GPS vs simulation raw GPS, common geodetic origin")
    set_equal_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_aggregate_start_aligned_visual(
    path: Path,
    gt_ref: Dict[str, Any],
    sim_results: Sequence[Dict[str, Any]],
    n_samples: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    gt_mean_xy = gt_ref["gt_mean_xy"]
    ax.plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], linewidth=2.5, label="GT mean raw GPS")

    aligned_inputs: List[np.ndarray] = []
    for result in sim_results:
        xy = result["sim_xy_start_aligned"]
        ax.plot(xy[:, 0], xy[:, 1], color="0.7", linewidth=1.0, alpha=0.9)
        aligned_inputs.append(xy)

    sim_mean_xy = compute_trace_mean(aligned_inputs, n_samples)
    if sim_mean_xy is not None:
        ax.plot(sim_mean_xy[:, 0], sim_mean_xy[:, 1], linewidth=2.0, label="Simulation mean, start-aligned visual")

    ax.scatter([gt_mean_xy[0, 0]], [gt_mean_xy[0, 1]], marker="o")
    ax.scatter([gt_mean_xy[-1, 0]], [gt_mean_xy[-1, 1]], marker="x")

    ax.set_title("GT mean raw GPS vs simulation raw GPS, start-aligned visual")
    set_equal_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_run_pair(
    path: Path,
    gt_ref: Dict[str, Any],
    run_label: str,
    sim_xy_common: np.ndarray,
    sim_xy_start_aligned: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    gt_mean_xy = gt_ref["gt_mean_xy"]

    axes[0].plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], linewidth=2.5, label="GT mean raw GPS")
    axes[0].plot(sim_xy_common[:, 0], sim_xy_common[:, 1], linewidth=2.0, label=run_label)
    axes[0].set_title("Common geodetic origin")
    set_equal_axes(axes[0])
    axes[0].legend(fontsize=8)

    axes[1].plot(gt_mean_xy[:, 0], gt_mean_xy[:, 1], linewidth=2.5, label="GT mean raw GPS")
    axes[1].plot(sim_xy_start_aligned[:, 0], sim_xy_start_aligned[:, 1], linewidth=2.0, label=f"{run_label} start-aligned")
    axes[1].set_title("Start-aligned visual")
    set_equal_axes(axes[1])
    axes[1].legend(fontsize=8)

    fig.suptitle(f"GT mean raw GPS vs simulation raw GPS, {run_label}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_gt_mean_csv(path: Path, gt_ref: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_mean_xy = gt_ref["gt_mean_xy"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "x_m", "y_m", "lat0_deg", "lon0_deg"])
        for i in range(gt_mean_xy.shape[0]):
            w.writerow([
                i,
                float(gt_mean_xy[i, 0]),
                float(gt_mean_xy[i, 1]),
                float(gt_ref["lat0_deg"]),
                float(gt_ref["lon0_deg"]),
            ])


def save_metrics_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def save_summary_txt(
    path: Path,
    sim_parent_dir: Path,
    gt_csv_paths: Sequence[Path],
    gt_ref: Dict[str, Any],
    sim_results: Sequence[Dict[str, Any]],
    metrics_rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("GT raw GPS vs simulation raw GPS summary\n")
        f.write(f"Simulation parent dir: {sim_parent_dir}\n")
        f.write(f"Common origin lat0_deg: {gt_ref['lat0_deg']:.10f}\n")
        f.write(f"Common origin lon0_deg: {gt_ref['lon0_deg']:.10f}\n")
        f.write(f"GT mean length_m: {gt_ref['gt_mean_length_m']:.6f}\n\n")

        f.write("GT raw CSVs used:\n")
        for p in gt_csv_paths:
            f.write(f"  - {p}\n")

        f.write("\nGT traces after common-origin projection:\n")
        for gt_tr in gt_ref["gt_traces"]:
            f.write(
                f"  - {gt_tr['source']}: points={gt_tr['points']}, "
                f"length_m={gt_tr['length_m']:.6f}, "
                f"reversed={bool(gt_tr.get('was_reversed', False))}\n"
            )

        f.write("\nSimulation runs processed:\n")
        for result in sim_results:
            f.write(
                f"  - {result['run_name']}: points={result['points']}, "
                f"length_m={result['length_m']:.6f}, "
                f"reversed={bool(result['was_reversed'])}\n"
            )

        f.write("\nPer-run metrics against GT mean raw GPS:\n")
        for row in metrics_rows:
            f.write(
                f"  - {row['run_name']}: "
                f"gt_pr={row['gt_pr']}, "
                f"rmse_lateral_m={row['rmse_lateral_m']}, "
                f"p95_lateral_m={row['p95_lateral_m']}, "
                f"rmse_pointwise_common_m={row['rmse_pointwise_common_m']}, "
                f"rmse_pointwise_start_aligned_m={row['rmse_pointwise_start_aligned_m']}\n"
            )


def collect_gt_csv_paths(gt_csv_glob: str) -> List[Path]:
    paths = [Path(p).expanduser().resolve() for p in sorted(glob_expand(gt_csv_glob), key=natural_sort_key)]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise RuntimeError(f"No GT CSVs matched: {gt_csv_glob}")
    return paths


def glob_expand(pattern: str) -> List[str]:
    from glob import glob

    return glob(pattern)


def find_sim_bag_dirs(sim_parent_dir: Path, sim_glob: str) -> List[Path]:
    if not sim_parent_dir.exists():
        raise RuntimeError(f"Simulation parent dir does not exist: {sim_parent_dir}")

    dirs = [p for p in sim_parent_dir.glob(sim_glob) if p.is_dir()]
    dirs = sorted(dirs, key=lambda p: natural_sort_key(p.name))

    valid_dirs: List[Path] = []
    for p in dirs:
        has_metadata = (p / "metadata.yaml").exists()
        has_mcap = any(x.is_file() for x in p.glob("*.mcap"))
        if has_metadata or has_mcap:
            valid_dirs.append(p)

    if not valid_dirs:
        raise RuntimeError(f"No bag folders matched inside {sim_parent_dir} with glob '{sim_glob}'")
    return valid_dirs


def process_one_run(
    bag_dir: Path,
    gps_topic: str,
    gt_ref: Dict[str, Any],
    n_samples: int,
) -> Dict[str, Any]:
    raw = read_bag_gps_latlon(bag_dir, gps_topic)
    projected = project_trace_to_common_origin(raw, gt_ref["lat0_deg"], gt_ref["lon0_deg"])

    sim_xy = projected["xy"]
    sim_xy, was_reversed = maybe_reverse_to_match_reference(sim_xy, gt_ref["gt_mean_xy"], n_samples)
    sim_xy_start_aligned = start_align_xy(sim_xy, gt_ref["gt_mean_xy"][0])

    metrics = compute_metrics_vs_gt(sim_xy, gt_ref["gt_mean_xy"], n_samples)

    return {
        "run_name": bag_dir.name,
        "bag_dir": str(bag_dir),
        "source": projected["source"],
        "points": int(sim_xy.shape[0]),
        "length_m": path_length(sim_xy),
        "was_reversed": was_reversed,
        "sim_xy_common": sim_xy,
        "sim_xy_start_aligned": sim_xy_start_aligned,
        **metrics,
    }


def main() -> int:
    args = parse_args()

    sim_parent_dir = Path(args.sim_parent_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    gt_csv_paths = collect_gt_csv_paths(args.gt_csv_glob)
    sim_bag_dirs = find_sim_bag_dirs(sim_parent_dir, args.sim_glob)

    gt_ref = build_gt_mean_reference(gt_csv_paths, args.resample_samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_plot_dir = out_dir / "per_run_plots"
    if args.save_per_run_plots:
        per_run_plot_dir.mkdir(parents=True, exist_ok=True)

    sim_results: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []

    print("[INFO] Common geodetic origin")
    print(f"  lat0_deg = {gt_ref['lat0_deg']:.10f}")
    print(f"  lon0_deg = {gt_ref['lon0_deg']:.10f}")
    print(f"  GT traces = {len(gt_ref['gt_traces'])}")
    print(f"  Simulation runs found = {len(sim_bag_dirs)}")

    for bag_dir in sim_bag_dirs:
        try:
            result = process_one_run(
                bag_dir=bag_dir,
                gps_topic=args.gps_topic,
                gt_ref=gt_ref,
                n_samples=args.resample_samples,
            )
        except Exception as exc:
            print(f"[WARN] Skipping {bag_dir.name}: {exc}")
            continue

        sim_results.append(result)
        metrics_rows.append(
            {
                "run_name": result["run_name"],
                "bag_dir": result["bag_dir"],
                "points": result["points"],
                "length_m": f"{result['length_m']:.6f}",
                "was_reversed": int(bool(result["was_reversed"])),
                "gt_pr": f"{result['gt_pr']:.6f}",
                "rmse_lateral_m": f"{result['rmse_lateral_m']:.6f}",
                "p95_lateral_m": f"{result['p95_lateral_m']:.6f}",
                "rmse_pointwise_common_m": f"{result['rmse_pointwise_common_m']:.6f}",
                "rmse_pointwise_start_aligned_m": f"{result['rmse_pointwise_start_aligned_m']:.6f}",
            }
        )

        print(
            f"[OK] {result['run_name']}: "
            f"gt_pr={result['gt_pr']:.3f}, "
            f"rmse_lateral_m={result['rmse_lateral_m']:.3f}, "
            f"p95_lateral_m={result['p95_lateral_m']:.3f}, "
            f"rmse_pointwise_common_m={result['rmse_pointwise_common_m']:.3f}"
        )

        if args.save_per_run_plots:
            plot_per_run_pair(
                path=per_run_plot_dir / f"{result['run_name']}_gt_vs_sim_raw_gps.png",
                gt_ref=gt_ref,
                run_label=result["run_name"],
                sim_xy_common=result["sim_xy_common"],
                sim_xy_start_aligned=result["sim_xy_start_aligned"],
            )

    if not sim_results:
        print("ERROR: No simulation runs could be processed.", file=sys.stderr)
        return 2

    gt_mean_csv = out_dir / "gt_mean_raw_gps_common_origin.csv"
    metrics_csv = out_dir / "per_run_gt_vs_sim_raw_gps_metrics.csv"
    summary_txt = out_dir / "gt_vs_sim_raw_gps_summary.txt"
    aggregate_common_png = out_dir / "gt_vs_sim_raw_gps_common_origin.png"
    aggregate_visual_png = out_dir / "gt_vs_sim_raw_gps_start_aligned_visual.png"

    save_gt_mean_csv(gt_mean_csv, gt_ref)
    save_metrics_csv(metrics_csv, metrics_rows)
    save_summary_txt(
        path=summary_txt,
        sim_parent_dir=sim_parent_dir,
        gt_csv_paths=gt_csv_paths,
        gt_ref=gt_ref,
        sim_results=sim_results,
        metrics_rows=metrics_rows,
    )

    plot_aggregate_common_origin(
        path=aggregate_common_png,
        gt_ref=gt_ref,
        sim_results=sim_results,
        n_samples=args.resample_samples,
    )
    plot_aggregate_start_aligned_visual(
        path=aggregate_visual_png,
        gt_ref=gt_ref,
        sim_results=sim_results,
        n_samples=args.resample_samples,
    )

    print("\n[OK] Outputs generated:")
    print(f"  - {aggregate_common_png}")
    print(f"  - {aggregate_visual_png}")
    print(f"  - {gt_mean_csv}")
    print(f"  - {metrics_csv}")
    print(f"  - {summary_txt}")
    if args.save_per_run_plots:
        print(f"  - {per_run_plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
