#!/usr/bin/env python3
"""Inventory field rosbags and audit localization-related recorded evidence.

This script does not claim absolute localization accuracy. It reports recorded
message timing, covariance estimates, temporal variability, frame consistency,
and explicit RTK-state observability when such a topic is present.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from field_audit_common_v3 import (
    NA,
    BagSpec,
    bag_specs_from_config,
    circular_std,
    covariance_state,
    environment_manifest,
    finite_float,
    get_frame_id,
    get_header_stamp,
    horizontal_sigma_from_covariance,
    interval_summary,
    iter_bag_messages,
    load_yaml,
    merge_bag_specs,
    merged_topic_candidates,
    numeric_summary,
    odometry_from_message,
    open_bag_reader,
    resolve_path,
    resolve_topic,
    run_pure_self_tests,
    safe_array,
    setup_logger,
    topic_type_map,
    validate_bag_paths,
    write_csv,
    write_json,
    write_yaml,
    yaw_sigma_from_covariance,
    quaternion_to_yaw,
)


@dataclass
class TopicAccumulator:
    bag_key: str
    bag_label: str
    bag_path: str
    topic: str
    type_name: str
    storage_stamps: List[float] = field(default_factory=list)
    header_stamps: List[float] = field(default_factory=list)
    frame_ids: Counter = field(default_factory=Counter)
    covariance_fields: Counter = field(default_factory=Counter)
    deserialize_failures: int = 0
    header_missing_or_invalid: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory ROS 2 field bags and audit timing, message regularity, "
            "frames, covariance estimates, and explicit RTK-state observability."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/field_audit_v3.yaml",
        help="Audit YAML. Default: configs/field_audit_v3.yaml",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="NAV2_Paper_Scripts root. Relative config paths are resolved from here.",
    )
    parser.add_argument(
        "--bag",
        action="append",
        default=[],
        help=(
            "Override YAML bags. Repeat as key:label:/path/to/bag.mcap "
            "or label:/path/to/bag.mcap."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="results/field_audit_v3",
        help="Output directory, relative to project root by default.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without opening bags.")
    parser.add_argument("--self-test", action="store_true", help="Run ROS-independent tests and exit.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging.")
    return parser.parse_args()


def message_covariance_fields(message: Any) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    if hasattr(message, "position_covariance"):
        result["position_covariance"] = safe_array(message.position_covariance, 9)
    if hasattr(message, "orientation_covariance"):
        result["orientation_covariance"] = safe_array(message.orientation_covariance, 9)
    if hasattr(message, "angular_velocity_covariance"):
        result["angular_velocity_covariance"] = safe_array(message.angular_velocity_covariance, 9)
    if hasattr(message, "linear_acceleration_covariance"):
        result["linear_acceleration_covariance"] = safe_array(message.linear_acceleration_covariance, 9)
    pose = getattr(message, "pose", None)
    if pose is not None and hasattr(pose, "covariance"):
        result["pose_covariance"] = safe_array(pose.covariance, 36)
    twist = getattr(message, "twist", None)
    if twist is not None and hasattr(twist, "covariance"):
        result["twist_covariance"] = safe_array(twist.covariance, 36)
    return result


def scalar_fields(message: Any, prefix: str = "", depth: int = 0) -> Dict[str, Any]:
    if message is None or depth > 2:
        return {}
    result: Dict[str, Any] = {}
    slots = getattr(message, "__slots__", [])
    field_names = getattr(message, "get_fields_and_field_types", lambda: {})()
    if field_names:
        names = list(field_names.keys())
    else:
        names = [name.lstrip("_") for name in slots]
    for name in names:
        try:
            value = getattr(message, name)
        except Exception:
            try:
                value = getattr(message, "_" + name)
            except Exception:
                continue
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, (str, bool, int, float)):
            result[key] = value
        elif depth < 2 and not isinstance(value, (bytes, bytearray, list, tuple, np.ndarray)):
            result.update(scalar_fields(value, key, depth + 1))
    return result


def rtk_relevant_fields(message: Any) -> Dict[str, Any]:
    tokens = ("rtk", "fix", "solution", "status", "quality", "carr", "float", "integer")
    return {
        key: value
        for key, value in scalar_fields(message).items()
        if any(token in key.lower() for token in tokens)
    }


def add_metric(
    rows: List[Dict[str, Any]],
    spec: BagSpec,
    topic: str,
    type_name: str,
    metric: str,
    values: Sequence[float],
    unit: str,
    interpretation: str,
    total_count: int,
    invalid_count: int,
    limitation: str = "",
) -> None:
    summary = numeric_summary(values)
    rows.append(
        {
            "bag_key": spec.key,
            "bag_label": spec.label,
            "bag_path": spec.path,
            "topic": topic,
            "message_type": type_name,
            "metric": metric,
            "unit": unit,
            "N_total": total_count,
            "N_valid": summary["n"],
            "N_invalid_or_unavailable": invalid_count,
            "invalid_fraction": invalid_count / total_count if total_count else NA,
            "min": summary["min"],
            "median": summary["median"],
            "mean": summary["mean"],
            "p95": summary["p95"],
            "max": summary["max"],
            "interpretation": interpretation,
            "limitation": limitation,
        }
    )


def inventory_row(accumulator: TopicAccumulator, gap_factor: float) -> Dict[str, Any]:
    storage = interval_summary(accumulator.storage_stamps, gap_factor=gap_factor)
    header = interval_summary(accumulator.header_stamps, gap_factor=gap_factor)
    count = len(accumulator.storage_stamps)
    covariance_totals: Counter = Counter()
    for combined_key, state_count in accumulator.covariance_fields.items():
        field_name, _separator, _state = combined_key.rpartition(":")
        covariance_totals[field_name] += state_count
    covariance_fractions = {
        combined_key: (
            state_count / covariance_totals[combined_key.rpartition(":")[0]]
            if covariance_totals[combined_key.rpartition(":")[0]]
            else 0.0
        )
        for combined_key, state_count in accumulator.covariance_fields.items()
    }
    return {
        "bag_key": accumulator.bag_key,
        "bag_label": accumulator.bag_label,
        "bag_path": accumulator.bag_path,
        "topic": accumulator.topic,
        "message_type": accumulator.type_name,
        "message_count": count,
        "storage_start_sec": storage["start_sec"],
        "storage_end_sec": storage["end_sec"],
        "storage_duration_s": storage["duration_s"],
        "storage_frequency_mean_hz": storage["frequency_mean_hz"],
        "storage_dt_median_s": storage["dt_median_s"],
        "storage_dt_p95_s": storage["dt_p95_s"],
        "storage_dt_max_s": storage["dt_max_s"],
        "storage_duplicates": storage["duplicates"],
        "storage_out_of_order": storage["out_of_order"],
        "storage_gap_threshold_s": storage["gap_threshold_s"],
        "storage_gap_count": storage["gap_count"],
        "storage_gap_total_s": storage["gap_total_s"],
        "header_valid_count": len(accumulator.header_stamps),
        "header_valid_fraction": len(accumulator.header_stamps) / count if count else NA,
        "header_missing_or_invalid": accumulator.header_missing_or_invalid,
        "header_start_sec": header["start_sec"],
        "header_end_sec": header["end_sec"],
        "header_duration_s": header["duration_s"],
        "header_frequency_mean_hz": header["frequency_mean_hz"],
        "header_dt_median_s": header["dt_median_s"],
        "header_dt_p95_s": header["dt_p95_s"],
        "header_dt_max_s": header["dt_max_s"],
        "header_duplicates": header["duplicates"],
        "header_out_of_order": header["out_of_order"],
        "frame_ids": sorted(accumulator.frame_ids),
        "frame_id_counts": dict(accumulator.frame_ids),
        "covariance_states": dict(accumulator.covariance_fields),
        "covariance_state_fractions": covariance_fractions,
        "deserialize_failures": accumulator.deserialize_failures,
    }


def topic_rate_row(inventory: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bag_key": inventory["bag_key"],
        "bag_label": inventory["bag_label"],
        "topic": inventory["topic"],
        "message_type": inventory["message_type"],
        "message_count": inventory["message_count"],
        "timestamp_domain": "storage",
        "frequency_mean_hz": inventory["storage_frequency_mean_hz"],
        "dt_median_s": inventory["storage_dt_median_s"],
        "dt_p95_s": inventory["storage_dt_p95_s"],
        "dt_max_s": inventory["storage_dt_max_s"],
        "duplicates": inventory["storage_duplicates"],
        "out_of_order": inventory["storage_out_of_order"],
        "gap_threshold_s": inventory["storage_gap_threshold_s"],
        "gap_count": inventory["storage_gap_count"],
        "gap_total_s": inventory["storage_gap_total_s"],
        "header_valid_fraction": inventory["header_valid_fraction"],
    }


def audit_one_bag(
    spec: BagSpec,
    config: Mapping[str, Any],
    logger: Any,
) -> Dict[str, Any]:
    logger.info("Auditing bag %s: %s", spec.label, spec.path)
    reader = open_bag_reader(spec.path)
    types = topic_type_map(reader)
    candidates = merged_topic_candidates(config)
    selected = {
        role: resolve_topic(types, role, candidates.get(role, []))
        for role in ("odometry", "gps", "imu", "tf", "tf_static", "rtk_explicit", "vehicle_state")
    }
    selected = {role: topic for role, topic in selected.items() if topic}
    logger.info("Resolved localization topics for %s: %s", spec.label, selected)

    accumulators = {
        topic: TopicAccumulator(spec.key, spec.label, spec.path, topic, type_name)
        for topic, type_name in types.items()
    }
    localization_rows: List[Dict[str, Any]] = []
    pose_horizontal_sigma: List[float] = []
    pose_yaw_sigma: List[float] = []
    twist_horizontal_sigma: List[float] = []
    odom_yaw: List[float] = []
    pose_cov_invalid = 0
    twist_cov_invalid = 0
    odom_count = 0
    gps_horizontal_sigma: List[float] = []
    gps_cov_invalid = 0
    gps_count = 0
    gps_status_counts: Counter = Counter()
    gps_covariance_type_counts: Counter = Counter()
    imu_yaw_sigma: List[float] = []
    imu_yaw: List[float] = []
    imu_cov_invalid = 0
    imu_count = 0
    explicit_rtk_values: Counter = Counter()
    explicit_rtk_topic = selected.get("rtk_explicit")
    explicit_rtk_count = 0
    tf_edges: Counter = Counter()

    for topic, type_name, storage_stamp, message, error in iter_bag_messages(spec.path):
        accumulator = accumulators[topic]
        accumulator.storage_stamps.append(storage_stamp)
        if error or message is None:
            accumulator.deserialize_failures += 1
            continue
        header_stamp = get_header_stamp(message)
        if header_stamp is None:
            accumulator.header_missing_or_invalid += 1
        else:
            accumulator.header_stamps.append(header_stamp)
        frame_id = get_frame_id(message)
        if frame_id:
            accumulator.frame_ids[frame_id] += 1
        covariance_fields = message_covariance_fields(message)
        for field_name, covariance in covariance_fields.items():
            accumulator.covariance_fields[f"{field_name}:{covariance_state(covariance)}"] += 1

        if topic == selected.get("odometry"):
            odom_count += 1
            try:
                sample = odometry_from_message(message, storage_stamp)
                odom_yaw.append(sample.yaw)
                horizontal = horizontal_sigma_from_covariance(sample.pose_covariance, 6)
                yaw_sigma = yaw_sigma_from_covariance(sample.pose_covariance, 6)
                twist_horizontal = horizontal_sigma_from_covariance(sample.twist_covariance, 6)
                if horizontal is None or yaw_sigma is None:
                    pose_cov_invalid += 1
                else:
                    pose_horizontal_sigma.append(horizontal)
                    pose_yaw_sigma.append(yaw_sigma)
                if twist_horizontal is None:
                    twist_cov_invalid += 1
                else:
                    twist_horizontal_sigma.append(twist_horizontal)
            except Exception as exc:
                logger.warning("Odometry parse failed in %s: %s", spec.label, exc)
                pose_cov_invalid += 1
                twist_cov_invalid += 1

        if type_name == "sensor_msgs/msg/NavSatFix":
            gps_count += 1
            covariance = safe_array(getattr(message, "position_covariance", []), 9)
            sigma = horizontal_sigma_from_covariance(covariance, 3)
            if sigma is None:
                gps_cov_invalid += 1
            else:
                gps_horizontal_sigma.append(sigma)
            gps_status_counts[str(getattr(getattr(message, "status", None), "status", NA))] += 1
            gps_covariance_type_counts[str(getattr(message, "position_covariance_type", NA))] += 1

        if type_name == "sensor_msgs/msg/Imu":
            imu_count += 1
            orientation_covariance = safe_array(getattr(message, "orientation_covariance", []), 9)
            sigma = yaw_sigma_from_covariance(orientation_covariance, 3)
            if sigma is None:
                imu_cov_invalid += 1
            else:
                imu_yaw_sigma.append(sigma)
            try:
                if orientation_covariance[0] >= 0.0:
                    imu_yaw.append(quaternion_to_yaw(message.orientation))
            except Exception:
                pass

        if explicit_rtk_topic and topic == explicit_rtk_topic:
            explicit_rtk_count += 1
            fields = rtk_relevant_fields(message)
            if fields:
                explicit_rtk_values[str(fields)] += 1

        if type_name == "tf2_msgs/msg/TFMessage":
            for transform in getattr(message, "transforms", []):
                parent = str(getattr(getattr(transform, "header", None), "frame_id", "")).lstrip("/")
                child = str(getattr(transform, "child_frame_id", "")).lstrip("/")
                if parent or child:
                    tf_edges[f"{parent}->{child}"] += 1

    odom_topic = selected.get("odometry", "")
    odom_type = types.get(odom_topic, "")
    add_metric(
        localization_rows, spec, odom_topic, odom_type,
        "odometry_pose_horizontal_sigma_indicated", pose_horizontal_sigma, "m",
        "Square root of recorded pose covariance xx+yy; system estimate, not absolute error.",
        odom_count, pose_cov_invalid,
        "Absolute localization accuracy is unavailable without independent surveyed ground truth.",
    )
    add_metric(
        localization_rows, spec, odom_topic, odom_type,
        "odometry_pose_yaw_sigma_indicated", [math.degrees(value) for value in pose_yaw_sigma], "deg",
        "Square root of recorded pose yaw covariance; system estimate, not measured heading error.",
        odom_count, pose_cov_invalid,
    )
    add_metric(
        localization_rows, spec, odom_topic, odom_type,
        "odometry_twist_horizontal_sigma_indicated", twist_horizontal_sigma, "m/s",
        "Square root of recorded twist covariance xx+yy; system estimate.",
        odom_count, twist_cov_invalid,
    )
    odom_circular_std = circular_std(odom_yaw)
    add_metric(
        localization_rows, spec, odom_topic, odom_type,
        "odometry_heading_temporal_circular_std", [] if odom_circular_std is None else [math.degrees(odom_circular_std)], "deg",
        "Temporal heading variability over the driven manoeuvre; not heading error.",
        odom_count, 0 if odom_circular_std is not None else odom_count,
    )

    gps_topics = [topic for topic, type_name in types.items() if type_name == "sensor_msgs/msg/NavSatFix"]
    gps_topic_label = ";".join(gps_topics)
    add_metric(
        localization_rows, spec, gps_topic_label, "sensor_msgs/msg/NavSatFix",
        "gnss_horizontal_sigma_indicated", gps_horizontal_sigma, "m",
        "Square root of recorded NavSatFix position covariance xx+yy; system estimate.",
        gps_count, gps_cov_invalid,
        "NavSatFix.status alone is not interpreted as explicit RTK fixed/float state.",
    )
    localization_rows.append(
        {
            "bag_key": spec.key,
            "bag_label": spec.label,
            "bag_path": spec.path,
            "topic": gps_topic_label,
            "message_type": "sensor_msgs/msg/NavSatFix",
            "metric": "navsatfix_status_distribution",
            "unit": "categorical",
            "N_total": gps_count,
            "N_valid": sum(gps_status_counts.values()),
            "N_invalid_or_unavailable": max(0, gps_count - sum(gps_status_counts.values())),
            "invalid_fraction": NA,
            "min": NA,
            "median": NA,
            "mean": NA,
            "p95": NA,
            "max": NA,
            "value_counts": dict(gps_status_counts),
            "covariance_type_counts": dict(gps_covariance_type_counts),
            "interpretation": "Recorded NavSatFix status and covariance-type codes.",
            "limitation": "These codes do not establish RTK fixed/float unless the receiver records that state explicitly.",
        }
    )

    imu_topics = [topic for topic, type_name in types.items() if type_name == "sensor_msgs/msg/Imu"]
    imu_topic_label = ";".join(imu_topics)
    add_metric(
        localization_rows, spec, imu_topic_label, "sensor_msgs/msg/Imu",
        "imu_yaw_sigma_indicated", [math.degrees(value) for value in imu_yaw_sigma], "deg",
        "Square root of recorded IMU yaw covariance.",
        imu_count, imu_cov_invalid,
        "No odometry-vs-IMU heading error is reported without an audited frame transform and convention.",
    )

    rtk_state_observable = bool(explicit_rtk_topic and explicit_rtk_values)
    if rtk_state_observable:
        rtk_result = "explicit topic recorded"
        rtk_limitation = "Values are preserved as recorded; receiver-specific semantics require its message documentation."
    elif explicit_rtk_topic:
        rtk_result = "RTK state not observable from the recorded fields"
        rtk_limitation = (
            "A candidate receiver topic was recorded, but no explicit fixed/float/solution field "
            "was decoded. Its message package/documentation must be supplied before interpretation."
        )
    else:
        rtk_result = "RTK state not observable from the recorded topics"
        rtk_limitation = "NavSatFix.status was not promoted to fixed/float state."
    localization_rows.append(
        {
            "bag_key": spec.key,
            "bag_label": spec.label,
            "bag_path": spec.path,
            "topic": explicit_rtk_topic or NA,
            "message_type": types.get(explicit_rtk_topic, NA) if explicit_rtk_topic else NA,
            "metric": "rtk_state_observability",
            "unit": "categorical",
            "N_total": explicit_rtk_count,
            "N_valid": sum(explicit_rtk_values.values()),
            "N_invalid_or_unavailable": (
                explicit_rtk_count - sum(explicit_rtk_values.values())
                if explicit_rtk_topic else 1
            ),
            "invalid_fraction": (
                (explicit_rtk_count - sum(explicit_rtk_values.values())) / explicit_rtk_count
                if explicit_rtk_count else NA
            ),
            "min": NA,
            "median": NA,
            "mean": NA,
            "p95": NA,
            "max": NA,
            "result": rtk_result,
            "value_counts": dict(explicit_rtk_values),
            "interpretation": "Explicit receiver RTK-state evidence only.",
            "limitation": rtk_limitation,
        }
    )
    localization_rows.append(
        {
            "bag_key": spec.key,
            "bag_label": spec.label,
            "bag_path": spec.path,
            "topic": ";".join(filter(None, [selected.get("tf"), selected.get("tf_static")])),
            "message_type": "tf2_msgs/msg/TFMessage",
            "metric": "tf_availability_and_edges",
            "unit": "categorical",
            "N_total": sum(tf_edges.values()),
            "N_valid": sum(tf_edges.values()),
            "N_invalid_or_unavailable": 0 if tf_edges else 1,
            "invalid_fraction": NA,
            "min": NA,
            "median": NA,
            "mean": NA,
            "p95": NA,
            "max": NA,
            "value_counts": dict(tf_edges),
            "result": "available" if tf_edges else "unavailable",
            "interpretation": "Recorded parent-child TF edges; this is availability, not transform-accuracy validation.",
            "limitation": "",
        }
    )

    audit = config.get("audit", {}) if isinstance(config.get("audit", {}), Mapping) else {}
    gap_factor = float(audit.get("gap_factor", 5.0))
    inventory = [inventory_row(accumulator, gap_factor) for accumulator in accumulators.values()]
    return {
        "selected_topics": selected,
        "inventory_rows": inventory,
        "topic_rate_rows": [topic_rate_row(row) for row in inventory],
        "localization_rows": localization_rows,
        "rtk_observable": rtk_state_observable,
        "tf_edges": dict(tf_edges),
    }


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_pure_self_tests()
        print("audit_field_rosbags_v3 pure self-tests: PASS")
        return 0

    project_root = Path(args.project_root).expanduser().resolve()
    config_path = resolve_path(args.config, project_root)
    config = load_yaml(config_path)
    config_specs = bag_specs_from_config(config, project_root)
    specs = merge_bag_specs(config_specs, args.bag)
    out_dir = resolve_path(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Config: {config_path}")
        print(f"Output: {out_dir}")
        for spec in specs:
            print(f"Bag {spec.key}: {spec.label} -> {spec.path}")
        print("Dry run: PASS (bags were not opened)")
        return 0

    validate_bag_paths(specs)
    logger = setup_logger(out_dir / "field_inventory_v3.log", verbose=args.verbose)
    logger.info("Starting field rosbag inventory v3")
    all_inventory: List[Dict[str, Any]] = []
    all_rates: List[Dict[str, Any]] = []
    all_localization: List[Dict[str, Any]] = []
    per_bag: Dict[str, Any] = {}
    for spec in specs:
        try:
            result = audit_one_bag(spec, config, logger)
            all_inventory.extend(result["inventory_rows"])
            all_rates.extend(result["topic_rate_rows"])
            all_localization.extend(result["localization_rows"])
            per_bag[spec.key] = {
                "status": "completed",
                "selected_topics": result["selected_topics"],
                "rtk_state_observable": result["rtk_observable"],
                "tf_edge_count": len(result["tf_edges"]),
            }
        except Exception as exc:
            logger.exception("Audit failed for %s", spec.label)
            per_bag[spec.key] = {"status": "failed", "error": str(exc)}

    write_csv(out_dir / "field_bag_inventory_v3.csv", all_inventory)
    write_csv(out_dir / "field_topic_rates_v3.csv", all_rates)
    write_csv(out_dir / "field_localization_quality_v3.csv", all_localization)
    summary = {
        "status": "completed_with_failures" if any(value["status"] == "failed" for value in per_bag.values()) else "completed",
        "methodological_scope": (
            "Recorded covariance estimates, temporal variability, frame and message regularity. "
            "No absolute localization accuracy is claimed without surveyed ground truth."
        ),
        "config": str(config_path),
        "bags": per_bag,
        "environment": environment_manifest(),
        "outputs": {
            "inventory": str(out_dir / "field_bag_inventory_v3.csv"),
            "rates": str(out_dir / "field_topic_rates_v3.csv"),
            "localization": str(out_dir / "field_localization_quality_v3.csv"),
        },
    }
    write_json(out_dir / "field_inventory_summary_v3.json", summary)
    write_yaml(
        out_dir / "audit_parameters_v3.yaml",
        {"config_path": str(config_path), "project_root": str(project_root), "bags": [spec.__dict__ for spec in specs]},
    )
    logger.info("Inventory rows: %d", len(all_inventory))
    logger.info("Localization rows: %d", len(all_localization))
    return 1 if summary["status"] == "completed_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
