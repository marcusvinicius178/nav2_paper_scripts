#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ROS 2 imports, requires sourcing the ROS environment before running
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

try:
    import rosbag2_py
except Exception as e:
    raise RuntimeError(
        "Failed to import rosbag2_py. Make sure you have ROS 2 installed and sourced.\n"
        "Example:\n"
        "  source /opt/ros/iron/setup.bash\n"
        "And packages:\n"
        "  sudo apt install -y ros-iron-rosbag2-py ros-iron-rosbag2-storage-mcap\n"
    ) from e


@dataclass
class TimeSeries:
    t: np.ndarray
    v: np.ndarray


def wrap_angle_rad(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def finite_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return np.zeros_like(y)
    return np.gradient(y, t)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def robust_percentile(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p))


def downsample_by_dt(t: np.ndarray, *arrays: np.ndarray, dt: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    if t.size == 0:
        return t, [a for a in arrays]
    keep_idx = [0]
    last_t = t[0]
    for i in range(1, t.size):
        if (t[i] - last_t) >= dt:
            keep_idx.append(i)
            last_t = t[i]
    keep_idx = np.array(keep_idx, dtype=int)
    outs = [a[keep_idx] for a in arrays]
    return t[keep_idx], outs


def polyline_cumulative_s(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] < 2:
        return np.zeros((xy.shape[0],), dtype=float)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d)))
    return s


def nearest_projection_on_polyline(
    pts: np.ndarray, poly: np.ndarray, s_poly: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each point in pts (N,2), find nearest projection onto polyline poly (M,2).
    Returns:
      s_hat: (N,) along-track coordinate on polyline
      d: (N,) lateral distance to polyline
      yaw_ref: (N,) reference heading from nearest segment
    """
    if poly.shape[0] < 2 or pts.shape[0] == 0:
        n = pts.shape[0]
        return np.full((n,), np.nan), np.full((n,), np.nan), np.full((n,), np.nan)

    a = poly[:-1]  # (M-1,2)
    b = poly[1:]   # (M-1,2)
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)  # (M-1,)

    # segment headings
    seg_yaw = np.arctan2(ab[:, 1], ab[:, 0])

    s_hat = np.zeros((pts.shape[0],), dtype=float)
    d = np.zeros((pts.shape[0],), dtype=float)
    yaw_ref = np.zeros((pts.shape[0],), dtype=float)

    # brute force, but fast enough for typical bag sizes with numpy operations per segment
    for i, p in enumerate(pts):
        ap = p - a  # (M-1,2)
        # projection factor
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.sum(ap * ab, axis=1) / ab2
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        t = np.clip(t, 0.0, 1.0)
        proj = a + (ab * t[:, None])  # (M-1,2)
        diff = p - proj
        dist2 = np.sum(diff * diff, axis=1)
        j = int(np.argmin(dist2))
        d[i] = float(np.sqrt(dist2[j]))
        s_hat[i] = float(s_poly[j] + t[j] * np.linalg.norm(ab[j]))
        yaw_ref[i] = float(seg_yaw[j])

    return s_hat, d, yaw_ref


def read_bag_messages(
    bag_uri: str,
    topic_names: List[str],
    storage_id: str = "mcap",
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[float, object]]]]:
    """
    Reads selected topics from a rosbag2 (including .mcap) and returns:
      topic_types: dict topic -> type string
      data: dict topic -> list[(t_sec, msg)]
    """
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {tt.name: tt.type for tt in topics_and_types}

    wanted = [t for t in topic_names if t in topic_types]
    data: Dict[str, List[Tuple[float, object]]] = {t: [] for t in wanted}

    msg_classes: Dict[str, object] = {}
    for t in wanted:
        msg_classes[t] = get_message(topic_types[t])

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic not in data:
            continue
        msg = deserialize_message(raw, msg_classes[topic])
        data[topic].append((t_ns * 1e-9, msg))

    return topic_types, data


def pick_first_existing(topic_types: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in topic_types:
            return c
    return None


def extract_xy_yaw_speed_from_odom_msgs(msgs: List[Tuple[float, object]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = []
    x = []
    y = []
    yaw = []
    v = []
    for ts, m in msgs:
        # nav_msgs/msg/Odometry
        px = float(m.pose.pose.position.x)
        py = float(m.pose.pose.position.y)
        q = m.pose.pose.orientation
        yawi = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        # speed from twist
        vx = float(m.twist.twist.linear.x)
        vy = float(m.twist.twist.linear.y)
        speed = float(math.sqrt(vx * vx + vy * vy))
        t.append(ts)
        x.append(px)
        y.append(py)
        yaw.append(yawi)
        v.append(speed)
    return np.array(t), np.array(x), np.array(y), np.array(yaw), np.array(v)


def extract_cmd_series(msgs: List[Tuple[float, object]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = []
    v = []
    w = []
    for ts, m in msgs:
        # geometry_msgs/msg/Twist
        t.append(ts)
        v.append(float(m.linear.x))
        w.append(float(m.angular.z))
    return np.array(t), np.array(v), np.array(w)


def extract_goal_from_paths(path_msgs: List[Tuple[float, object]]) -> Optional[Tuple[float, float, float]]:
    """
    Expects nav_msgs/Path. Returns (x,y,yaw) of the last pose of the last non-empty path.
    """
    if not path_msgs:
        return None
    # take last non-empty
    for ts, pm in reversed(path_msgs):
        if hasattr(pm, "poses") and len(pm.poses) > 0:
            last = pm.poses[-1].pose
            x = float(last.position.x)
            y = float(last.position.y)
            q = last.orientation
            yaw = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
            return (x, y, yaw)
    return None


def build_reference_from_gt_bag(
    gt_bag_uri: str,
    gt_odom_topic: str,
    storage_id: str,
    dt_ref: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      ref_xy (M,2), ref_s (M,), ref_yaw_seg (M-1,), ref_speed_s (M,)
    ref_yaw_seg is implicit via nearest projection function, so we only need ref_xy and ref_s and ref_speed.
    """
    gt_types, gt_data = read_bag_messages(gt_bag_uri, [gt_odom_topic], storage_id=storage_id)
    if gt_odom_topic not in gt_data or len(gt_data[gt_odom_topic]) == 0:
        raise RuntimeError(f"Ground truth bag does not contain odom data on topic: {gt_odom_topic}")

    t, x, y, yaw, speed = extract_xy_yaw_speed_from_odom_msgs(gt_data[gt_odom_topic])
    t, [x, y, yaw, speed] = downsample_by_dt(t, x, y, yaw, speed, dt=dt_ref)

    ref_xy = np.stack([x, y], axis=1)
    ref_s = polyline_cumulative_s(ref_xy)
    # speed profile aligned with s, for interpolation
    return ref_xy, ref_s, yaw, speed


def infer_action_success_from_bt_log(bt_msgs: List[Tuple[float, object]]) -> Optional[bool]:
    """
    Best-effort inference:
      looks for BehaviorTreeStatusChange-like events with current_status == SUCCESS (2),
      for node names containing NavigateToPose or NavigateThroughPoses.
    If message format is unknown, returns None.
    """
    if not bt_msgs:
        return None

    NAV_KEYS = ["NavigateToPose", "NavigateThroughPoses", "BtNavigator", "bt_navigator"]
    SUCCESS_CODES = {2, 4}  # common: 2=SUCCESS in BehaviorTree.CPP, sometimes 4 used in other enums

    for ts, m in reversed(bt_msgs):
        if hasattr(m, "event_log"):
            try:
                for ev in getattr(m, "event_log"):
                    node_name = ""
                    if hasattr(ev, "node_name"):
                        node_name = str(getattr(ev, "node_name"))
                    cur = None
                    if hasattr(ev, "current_status"):
                        cur = int(getattr(ev, "current_status"))
                    if cur is not None and cur in SUCCESS_CODES:
                        if any(k.lower() in node_name.lower() for k in NAV_KEYS):
                            return True
            except Exception:
                continue

        # string fallback
        for attr in ["msg", "message", "text"]:
            if hasattr(m, attr):
                s = str(getattr(m, attr)).lower()
                if ("succeeded" in s or "success" in s) and ("navigate" in s or "goal" in s):
                    return True

    # could not find success, but we also cannot claim failure
    return None


def parse_labels_from_path(bag_path: str) -> Dict[str, str]:
    """
    Heuristic parser for your folder structure:
      .../<Scenario>/<METHOD>_vel_<SPEED>/<run_folder>/<file>.mcap
    Example:
      Scenario1_reta/NAVFN_MPPI_vel_20/Navfn_mppi_1/Navfn_mppi_1_0.mcap
    """
    parts = os.path.normpath(bag_path).split(os.sep)
    labels: Dict[str, str] = {}
    # search for Scenario*
    for p in parts:
        if p.lower().startswith("scenario"):
            labels["scenario"] = p
            break

    # search for *_vel_*
    for p in parts:
        if "_vel_" in p.lower():
            labels["group"] = p
            # method is everything before _vel_
            idx = p.lower().find("_vel_")
            labels["method"] = p[:idx]
            labels["speed_kmh"] = p[idx + len("_vel_"):]
            break

    # run folder
    if len(parts) >= 2:
        labels["run_folder"] = parts[-2]

    return labels


def main():
    ap = argparse.ArgumentParser(
        description="Extract Nav2 benchmark metrics from a rosbag2 (MCAP) run, optionally compared against a GT driver bag."
    )
    ap.add_argument("--bag", required=True, help="Path to .mcap file or bag directory (uri for rosbag2).")
    ap.add_argument("--storage-id", default="mcap", help="rosbag2 storage id, default: mcap")

    ap.add_argument("--odom-topic", default="", help="Odometry topic for executed trajectory, default auto-pick")
    ap.add_argument("--cmd-topic", default="", help="cmd_vel topic for control commands, default auto-pick")
    ap.add_argument("--path-topic", default="", help="Path topic for goal extraction, default auto-pick")

    ap.add_argument("--gt-bag", default="", help="Optional GT driver bag (.mcap or directory). If provided, errors are computed vs GT.")
    ap.add_argument("--gt-odom-topic", default="", help="Odometry topic in GT bag, default same as --odom-topic (auto-picked).")

    ap.add_argument("--speed-setpoint-kmh", type=float, default=float("nan"),
                    help="If no GT speed profile is available, use this setpoint for RMSE_v. If NaN, tries to infer from folder name.")
    ap.add_argument("--d-ok", type=float, default=0.75, help="Geometric success distance tolerance, meters.")
    ap.add_argument("--yaw-ok-deg", type=float, default=10.0, help="Geometric success yaw tolerance, degrees.")

    ap.add_argument("--dt-sample", type=float, default=0.10, help="Downsample period for odom samples, seconds.")
    ap.add_argument("--dt-ref", type=float, default=0.10, help="Downsample period for GT reference, seconds.")
    ap.add_argument("--stop-speed", type=float, default=0.15, help="Speed threshold to consider stopped, m/s.")
    ap.add_argument("--stop-window", type=float, default=1.0, help="Seconds of low speed near goal to consider finished.")

    ap.add_argument("--out-json", default="", help="Write per-run metrics to JSON file.")
    ap.add_argument("--out-csv", default="", help="Append per-run metrics to CSV file (creates header if needed).")

    args = ap.parse_args()

    bag_uri = os.path.abspath(args.bag)

    # First pass, read topic list to auto-pick defaults
    reader0 = rosbag2_py.SequentialReader()
    storage_options0 = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=args.storage_id)
    converter_options0 = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader0.open(storage_options0, converter_options0)
    topics_and_types0 = reader0.get_all_topics_and_types()
    topic_types0 = {tt.name: tt.type for tt in topics_and_types0}

    odom_topic = args.odom_topic.strip()
    if not odom_topic:
        odom_topic = pick_first_existing(topic_types0, ["/odometry/local", "/odom", "/odometry/global", "/odometry/gps"])
    if not odom_topic:
        raise RuntimeError("Could not auto-pick an odometry topic. Provide --odom-topic explicitly.")

    cmd_topic = args.cmd_topic.strip()
    if not cmd_topic:
        cmd_topic = pick_first_existing(topic_types0, ["/cmd_vel_nav", "/cmd_vel"])
    if not cmd_topic:
        raise RuntimeError("Could not auto-pick a cmd_vel topic. Provide --cmd-topic explicitly.")

    path_topic = args.path_topic.strip()
    if not path_topic:
        path_topic = pick_first_existing(topic_types0, ["/plan_smoothed", "/transformed_global_plan", "/plan"])
    # path_topic can be None, then goal may be unknown, we still compute tracking if GT provided

    bt_topic = pick_first_existing(topic_types0, ["/behavior_tree_log"])

    # Read required topics from run bag
    topics_to_read = [odom_topic, cmd_topic]
    if path_topic:
        topics_to_read.append(path_topic)
    if bt_topic:
        topics_to_read.append(bt_topic)

    run_types, run_data = read_bag_messages(bag_uri, topics_to_read, storage_id=args.storage_id)

    if odom_topic not in run_data or len(run_data[odom_topic]) == 0:
        raise RuntimeError(f"No odom data found on {odom_topic} in bag: {bag_uri}")
    if cmd_topic not in run_data or len(run_data[cmd_topic]) == 0:
        raise RuntimeError(f"No cmd data found on {cmd_topic} in bag: {bag_uri}")

    t_odom, x_odom, y_odom, yaw_odom, v_odom = extract_xy_yaw_speed_from_odom_msgs(run_data[odom_topic])
    t_cmd, v_cmd, w_cmd = extract_cmd_series(run_data[cmd_topic])

    # Downsample odom for faster processing
    t_odom, [x_odom, y_odom, yaw_odom, v_odom] = downsample_by_dt(
        t_odom, x_odom, y_odom, yaw_odom, v_odom, dt=args.dt_sample
    )

    # Determine t0 as first cmd timestamp
    t0 = float(t_cmd[0])
    t_end_raw = float(t_odom[-1])

    # Goal extraction from path topic if present
    goal = None
    if path_topic and path_topic in run_data:
        goal = extract_goal_from_paths(run_data[path_topic])

    # Infer action success from BT log if possible
    sr_action = float("nan")
    action_success = None
    if bt_topic and bt_topic in run_data:
        action_success = infer_action_success_from_bt_log(run_data[bt_topic])
    if action_success is True:
        sr_action = 1.0
    elif action_success is False:
        sr_action = 0.0

    # Labels from folder naming
    labels = parse_labels_from_path(bag_uri)

    # Infer speed setpoint if not provided
    speed_setpoint_kmh = args.speed_setpoint_kmh
    if math.isnan(speed_setpoint_kmh):
        if "speed_kmh" in labels:
            try:
                speed_setpoint_kmh = float(labels["speed_kmh"])
            except Exception:
                speed_setpoint_kmh = float("nan")
    speed_setpoint_ms = float("nan")
    if not math.isnan(speed_setpoint_kmh):
        speed_setpoint_ms = speed_setpoint_kmh / 3.6

    # Build reference path
    use_gt = bool(args.gt_bag.strip())
    ref_xy = None
    ref_s = None
    ref_speed_s = None

    if use_gt:
        gt_uri = os.path.abspath(args.gt_bag)
        gt_odom_topic = args.gt_odom_topic.strip()
        if not gt_odom_topic:
            gt_odom_topic = odom_topic
        ref_xy, ref_s, gt_yaw, gt_speed = build_reference_from_gt_bag(
            gt_uri, gt_odom_topic, args.storage_id, dt_ref=args.dt_ref
        )
        # map speed by s for interpolation
        ref_speed_s = gt_speed
    else:
        if path_topic and path_topic in run_data and len(run_data[path_topic]) > 0:
            # choose last non-empty path message
            chosen = None
            for ts, pm in reversed(run_data[path_topic]):
                if hasattr(pm, "poses") and len(pm.poses) > 1:
                    chosen = pm
                    break
            if chosen is not None:
                xs = [float(ps.pose.position.x) for ps in chosen.poses]
                ys = [float(ps.pose.position.y) for ps in chosen.poses]
                ref_xy = np.stack([np.array(xs), np.array(ys)], axis=1)
                ref_s = polyline_cumulative_s(ref_xy)
                ref_speed_s = None
        if ref_xy is None:
            raise RuntimeError(
                "No GT bag provided and no usable Path topic found to build a reference polyline.\n"
                "Provide --gt-bag for driver ground truth, or ensure /plan_smoothed or /transformed_global_plan exists."
            )

    # Select interval t0..tend, with optional early finish detection near goal
    # We try to detect finish time as first time within goal tolerance AND stopped for stop_window seconds
    t_end = t_end_raw
    if goal is not None:
        gx, gy, gyaw = goal
        d_ok = float(args.d_ok)
        yaw_ok = float(args.yaw_ok_deg) * math.pi / 180.0

        # compute distance to goal
        d_goal = np.sqrt((x_odom - gx) ** 2 + (y_odom - gy) ** 2)
        # compute yaw error to goal yaw
        dyaw_goal = wrap_angle_rad(yaw_odom - gyaw)
        near_goal = (d_goal <= d_ok) & (np.abs(dyaw_goal) <= yaw_ok)
        stopped = (v_odom <= float(args.stop_speed))

        # find first index where near_goal and then stays stopped for stop_window
        if np.any(near_goal):
            win = float(args.stop_window)
            for i in range(len(t_odom)):
                if t_odom[i] < t0:
                    continue
                if not near_goal[i]:
                    continue
                # check if in [t_i, t_i+win] we are mostly stopped
                t_i = t_odom[i]
                mask = (t_odom >= t_i) & (t_odom <= (t_i + win))
                if mask.sum() < 3:
                    continue
                if np.all(stopped[mask]):
                    t_end = float(t_i + win)
                    break

    # filter odom samples within t0..t_end
    mask_odom = (t_odom >= t0) & (t_odom <= t_end)
    t_run = t_odom[mask_odom]
    x_run = x_odom[mask_odom]
    y_run = y_odom[mask_odom]
    yaw_run = yaw_odom[mask_odom]
    v_run = v_odom[mask_odom]

    if t_run.size < 5:
        raise RuntimeError("Not enough odom samples in the selected interval. Check t0 and bag content.")

    pts_run = np.stack([x_run, y_run], axis=1)

    # compute projection on reference polyline
    s_hat, e_y, yaw_ref = nearest_projection_on_polyline(pts_run, ref_xy, ref_s)

    # path progress ratio
    pr = float("nan")
    if ref_s is not None and ref_s.size > 0:
        S_gt = float(ref_s[-1]) if float(ref_s[-1]) > 1e-9 else float("nan")
        if not math.isnan(S_gt):
            pr = float(s_hat[-1] / S_gt)

    # lateral RMSE and P95
    rmse_y = float(rms(e_y))
    p95_y = float(robust_percentile(e_y, 95.0))

    # heading RMSE relative to reference heading (path tangent)
    e_psi = wrap_angle_rad(yaw_run - yaw_ref)
    rmse_psi = float(rms(e_psi))

    # speed RMSE relative to GT speed profile or setpoint
    rmse_v = float("nan")
    if use_gt and ref_speed_s is not None and ref_speed_s.size == ref_s.size:
        # interpolate GT speed by s_hat, using s_poly nodes
        v_ref = np.interp(s_hat, ref_s, ref_speed_s)
        e_v = v_run - v_ref
        rmse_v = float(rms(e_v))
    else:
        if not math.isnan(speed_setpoint_ms):
            e_v = v_run - speed_setpoint_ms
            rmse_v = float(rms(e_v))

    # jerk RMS from speed derivatives
    a_run = finite_diff(v_run, t_run)
    j_run = finite_diff(a_run, t_run)
    rms_jx = float(rms(j_run))

    # steering rate RMS from cmd angular z derivative
    mask_cmd = (t_cmd >= t0) & (t_cmd <= t_end)
    t_cmd_run = t_cmd[mask_cmd]
    w_cmd_run = w_cmd[mask_cmd]
    if t_cmd_run.size >= 3:
        w_dot = finite_diff(w_cmd_run, t_cmd_run)
        rms_wdot = float(rms(w_dot))
    else:
        rms_wdot = float("nan")

    # geometric success
    sr_geo = float("nan")
    d_final = float("nan")
    yaw_final_err = float("nan")
    if goal is not None:
        gx, gy, gyaw = goal
        d_final = float(math.sqrt((x_run[-1] - gx) ** 2 + (y_run[-1] - gy) ** 2))
        yaw_final_err = float(wrap_angle_rad(np.array([yaw_run[-1] - gyaw]))[0])
        d_ok = float(args.d_ok)
        yaw_ok = float(args.yaw_ok_deg) * math.pi / 180.0
        sr_geo = 1.0 if (d_final <= d_ok and abs(yaw_final_err) <= yaw_ok) else 0.0

    # if SR_action could not be inferred, leave NaN, you will aggregate on available values
    # mission time
    mission_time = float(t_end - t0)

    out = {
        "bag": bag_uri,
        "odom_topic": odom_topic,
        "cmd_topic": cmd_topic,
        "path_topic": path_topic if path_topic else "",
        "gt_bag": os.path.abspath(args.gt_bag) if use_gt else "",
        "gt_odom_topic": args.gt_odom_topic.strip() if use_gt else "",
        "scenario": labels.get("scenario", ""),
        "method": labels.get("method", ""),
        "speed_kmh": labels.get("speed_kmh", ""),
        "run_folder": labels.get("run_folder", ""),
        "t0": t0,
        "t_end": t_end,
        "mission_time_s": mission_time,
        "sr_action": sr_action,
        "sr_geo": sr_geo,
        "progress_ratio": pr,
        "rmse_y_m": rmse_y,
        "p95_y_m": p95_y,
        "rmse_psi_rad": rmse_psi,
        "rmse_v_ms": rmse_v,
        "rms_jerk_ms3": rms_jx,
        "rms_yawrate_dot_rads2": rms_wdot,
        "goal_x": float(goal[0]) if goal else float("nan"),
        "goal_y": float(goal[1]) if goal else float("nan"),
        "goal_yaw_rad": float(goal[2]) if goal else float("nan"),
        "final_dist_to_goal_m": d_final,
        "final_yaw_err_rad": yaw_final_err,
        "notes": "yawrate_dot uses cmd_vel.angular.z derivative as steering-equivalent proxy"
    }

    # Write JSON
    if args.out_json:
        out_json_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)

    # Append CSV
    if args.out_csv:
        out_csv_path = os.path.abspath(args.out_csv)
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        header = list(out.keys())
        write_header = not os.path.exists(out_csv_path) or os.path.getsize(out_csv_path) == 0
        with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(out)

    # Print to stdout
    print(json.dumps(out, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()