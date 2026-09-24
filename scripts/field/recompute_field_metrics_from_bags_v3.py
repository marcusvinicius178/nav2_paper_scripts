#!/usr/bin/env python3
"""Recompute synchronized field clearance and uncertainty audits (v3).

Primary metric
--------------
Minimum Euclidean distance between the oriented vehicle footprint polygon and
the *area* of occupied OccupancyGrid cells. Every retained grid is paired with
its nearest odometry pose in one common timestamp domain. Occupied cells are
never subsampled for the primary metric.

The script preserves the original field scripts and writes only v3 outputs.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from field_audit_common_v3 import (
    BASELINE_CLEARANCE_M,
    NA,
    BagMetricData,
    BagSpec,
    ClearanceEvaluation,
    GridSample,
    OdomSample,
    SyncPair,
    bag_specs_from_config,
    choose_common_time_domain,
    choose_occupancy_thresholds,
    deterministic_pose_perturbations,
    environment_manifest,
    finite_float,
    load_metric_bag,
    load_yaml,
    merge_bag_specs,
    minimum_clearance,
    nearest_footprint,
    numeric_summary,
    occupancy_centers,
    pad_convex_polygon,
    pair_grids_to_nearest_poses,
    resolve_fallback_footprint,
    resolve_path,
    run_pure_self_tests,
    setup_logger,
    summarize_sync_pairs,
    transform_polygon,
    validate_bag_paths,
    validate_polygon,
    write_csv,
    write_json,
    write_yaml,
)


CLEARANCE_DEFINITION = (
    "minimum Euclidean distance between the oriented vehicle footprint polygon "
    "and occupied OccupancyGrid cell areas at nearest synchronized pose-grid pairs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute field pose-grid synchronization, critical-event speed, "
            "footprint-to-cell-area clearance, and deterministic sensitivity audits."
        )
    )
    parser.add_argument("--config", default="configs/field_audit_v3.yaml", help="Audit YAML.")
    parser.add_argument("--project-root", default=".", help="NAV2_Paper_Scripts root.")
    parser.add_argument(
        "--bag",
        action="append",
        default=[],
        help="Override YAML bags; repeat key:label:/path/to/bag.mcap.",
    )
    parser.add_argument("--out-dir", default="results/field_audit_v3", help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without opening bags.")
    parser.add_argument("--self-test", action="store_true", help="Run ROS-independent geometry/timestamp tests.")
    parser.add_argument("--verbose", action="store_true", help="Verbose console logging.")
    return parser.parse_args()


def polygon_dimensions(vertices: np.ndarray) -> Tuple[float, float]:
    vertices = validate_polygon(vertices)
    edge_lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    if edge_lengths.size == 0:
        return float("nan"), float("nan")
    return float(np.max(edge_lengths)), float(np.min(edge_lengths))


def inverse_pose_transform(vertices_world: np.ndarray, pose: OdomSample) -> np.ndarray:
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return validate_polygon((validate_polygon(vertices_world) - np.array([pose.x, pose.y])) @ rotation)


def footprint_for_pair(
    bag: BagMetricData,
    pair: SyncPair,
    fallback_local: np.ndarray,
    fallback_source: str,
    maximum_delta_s: float,
) -> Tuple[Optional[np.ndarray], str, str, Optional[np.ndarray]]:
    pose = bag.odometry[pair.pose_index]
    grid = bag.grids[pair.grid_index]
    published = nearest_footprint(
        bag.footprints,
        target_stamp=pair.grid_stamp_sec,
        domain=pair.time_domain,
        maximum_delta_s=maximum_delta_s,
    )
    if published is not None:
        if published.frame_id != grid.frame_id:
            if pose.frame_id == grid.frame_id:
                world = transform_polygon(fallback_local, pose.x, pose.y, pose.yaw)
                return (
                    world,
                    fallback_source,
                    f"recorded published footprint rejected because frame={published.frame_id}, "
                    f"grid frame={grid.frame_id}; explicit fallback used",
                    fallback_local,
                )
            return (
                None,
                "published_footprint_frame_mismatch",
                f"published footprint frame={published.frame_id}, grid frame={grid.frame_id}; "
                f"odometry frame={pose.frame_id}",
                None,
            )
        try:
            local_equivalent = inverse_pose_transform(published.vertices, pose)
        except Exception:
            local_equivalent = None
        return published.vertices, "recorded_published_footprint", "", local_equivalent
    if pose.frame_id != grid.frame_id:
        return (
            None,
            "fallback_frame_mismatch",
            f"odometry frame={pose.frame_id}, grid frame={grid.frame_id}; no audited transform applied",
            None,
        )
    world = transform_polygon(fallback_local, pose.x, pose.y, pose.yaw)
    return world, fallback_source, "published footprint unavailable at synchronized time", fallback_local


def evaluate_pair(
    bag: BagMetricData,
    pair: SyncPair,
    threshold: int,
    fallback_local: np.ndarray,
    fallback_source: str,
    maximum_footprint_delta_s: float,
    max_cells: int,
) -> Tuple[Optional[ClearanceEvaluation], Dict[str, Any]]:
    pose = bag.odometry[pair.pose_index]
    grid = bag.grids[pair.grid_index]
    footprint_world, footprint_source, limitation, local_equivalent = footprint_for_pair(
        bag,
        pair,
        fallback_local,
        fallback_source,
        maximum_footprint_delta_s,
    )
    details: Dict[str, Any] = {
        "footprint_source": footprint_source,
        "footprint_limitation": limitation,
        "footprint_world": footprint_world,
        "footprint_local_equivalent": local_equivalent,
    }
    if footprint_world is None:
        return None, details
    centers = occupancy_centers(grid, threshold)
    if centers.shape[0] > max_cells:
        raise RuntimeError(
            f"{bag.spec.label}: grid {pair.grid_index} has {centers.shape[0]} cells at "
            f"threshold {threshold}, exceeding max_occupied_cells_per_grid={max_cells}. "
            "No cells were subsampled. Constrain a physically justified ROI or increase the audited limit."
        )
    evaluation = minimum_clearance(
        footprint_world=footprint_world,
        robot_center_xy=np.array([pose.x, pose.y]),
        centers=centers,
        cell_resolution=grid.resolution,
        grid_yaw=grid.origin_yaw,
        threshold=threshold,
    )
    return evaluation, details


def unavailable_rows(spec: BagSpec, reason: str) -> Dict[str, List[Dict[str, Any]]]:
    common = {
        "bag_key": spec.key,
        "bag_label": spec.label,
        "bag_path": spec.path,
        "status": "unavailable",
        "limitation": reason,
    }
    return {
        "sync_summary": [dict(common)],
        "critical": [dict(common)],
        "nominal": [dict(common, clearance_definition=CLEARANCE_DEFINITION)],
        "sensitivity": [dict(common, variant_type="unavailable")],
    }


def sync_pair_rows(bag: BagMetricData, pairs: Sequence[SyncPair]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for pair in pairs:
        pose = bag.odometry[pair.pose_index]
        grid = bag.grids[pair.grid_index]
        result.append(
            {
                "bag_key": bag.spec.key,
                "bag_label": bag.spec.label,
                "bag_path": bag.spec.path,
                "grid_index": pair.grid_index,
                "pose_index": pair.pose_index,
                "time_domain": pair.time_domain,
                "pose_stamp_sec": pair.pose_stamp_sec,
                "grid_stamp_sec": pair.grid_stamp_sec,
                "delta_signed_pose_minus_grid_s": pair.delta_signed_s,
                "delta_abs_s": pair.delta_abs_s,
                "within_max_tolerance": int(pair.accepted),
                "pose_frame": pose.frame_id,
                "grid_frame": grid.frame_id,
                "grid_resolution_m": grid.resolution,
            }
        )
    return result


def process_bag(
    bag: BagMetricData,
    config: Mapping[str, Any],
    project_root: Path,
    logger: Any,
) -> Dict[str, Any]:
    synchronization = config.get("synchronization", {}) if isinstance(config.get("synchronization", {}), Mapping) else {}
    clearance_config = config.get("clearance", {}) if isinstance(config.get("clearance", {}), Mapping) else {}
    sensitivity_config = config.get("sensitivity", {}) if isinstance(config.get("sensitivity", {}), Mapping) else {}
    tolerances = [float(value) for value in synchronization.get("tolerances_s", [0.05, 0.10, 0.25, 0.50, 1.00])]
    maximum_delta = float(synchronization.get("maximum_delta_s", 1.0))
    if not math.isclose(maximum_delta, 1.0, rel_tol=0.0, abs_tol=1e-12):
        logger.warning(
            "%s maximum synchronization delta is %.3f s; the historical baseline limit was 1.0 s",
            bag.spec.label,
            maximum_delta,
        )
    minimum_header_fraction = float(synchronization.get("minimum_header_fraction", 0.95))
    nominal_threshold = int(clearance_config.get("nominal_occupancy_threshold", 50))
    max_cells = int(clearance_config.get("max_occupied_cells_per_grid", 250000))
    maximum_footprint_delta = float(clearance_config.get("maximum_footprint_time_delta_s", maximum_delta))

    if not bag.odometry or not bag.grids:
        reason = "; ".join(bag.warnings) or "odometry or obstacle-grid samples unavailable"
        rows = unavailable_rows(bag.spec, reason)
        rows.update({"sync_pairs": [], "summary": {"status": "unavailable", "limitation": reason}})
        return rows

    fallback_local, fallback_source, configured_padding, footprint_warnings = resolve_fallback_footprint(
        config,
        project_root,
    )
    bag.warnings.extend(footprint_warnings)
    domain, domain_reason = choose_common_time_domain(
        bag.odometry,
        bag.grids,
        minimum_header_fraction=minimum_header_fraction,
    )
    pairs = pair_grids_to_nearest_poses(
        bag.odometry,
        bag.grids,
        domain=domain,
        maximum_delta_s=maximum_delta,
    )
    accepted = [pair for pair in pairs if pair.accepted]
    sync_rows = sync_pair_rows(bag, pairs)
    sync_summary = {
        "bag_key": bag.spec.key,
        "bag_label": bag.spec.label,
        "bag_path": bag.spec.path,
        "status": "available" if accepted else "unavailable",
        "pose_topic": bag.topics.get("odometry", NA),
        "grid_topic": bag.topics.get("obstacle_grid", NA),
        "published_footprint_topic": bag.topics.get("published_footprint", NA),
        "n_poses": len(bag.odometry),
        "n_grids": len(bag.grids),
        "n_published_footprints": len(bag.footprints),
        "time_domain": domain,
        "time_domain_reason": domain_reason,
        "maximum_tolerance_s": maximum_delta,
        **summarize_sync_pairs(pairs, tolerances),
        "limitation": "; ".join(bag.warnings),
    }
    if not accepted:
        reason = "No pose-grid pair fell within the configured maximum tolerance"
        rows = unavailable_rows(bag.spec, reason)
        rows["sync_summary"] = [sync_summary]
        rows.update({"sync_pairs": sync_rows, "summary": {"status": "unavailable", "limitation": reason}})
        return rows

    nominal_evaluations: List[Tuple[SyncPair, ClearanceEvaluation, Dict[str, Any]]] = []
    for pair in accepted:
        evaluation, details = evaluate_pair(
            bag,
            pair,
            nominal_threshold,
            fallback_local,
            fallback_source,
            maximum_footprint_delta,
            max_cells,
        )
        if evaluation is not None and math.isfinite(evaluation.clearance_area_m):
            nominal_evaluations.append((pair, evaluation, details))
    if not nominal_evaluations:
        reason = "No synchronized pair produced a valid clearance; inspect frames, footprint, and occupied cells"
        rows = unavailable_rows(bag.spec, reason)
        rows["sync_summary"] = [sync_summary]
        rows.update({"sync_pairs": sync_rows, "summary": {"status": "unavailable", "limitation": reason}})
        return rows

    critical_pair, critical_evaluation, critical_details = min(
        nominal_evaluations,
        key=lambda item: item[1].clearance_area_m,
    )
    critical_pose = bag.odometry[critical_pair.pose_index]
    critical_grid = bag.grids[critical_pair.grid_index]
    speed_planar = math.hypot(critical_pose.vx, critical_pose.vy)
    footprint_world = critical_details["footprint_world"]
    footprint_length, footprint_width = polygon_dimensions(footprint_world)
    baseline = BASELINE_CLEARANCE_M.get(bag.spec.key)
    baseline_delta = critical_evaluation.clearance_area_m - baseline if baseline is not None else None

    critical_row = {
        "bag_key": bag.spec.key,
        "bag_label": bag.spec.label,
        "bag_path": bag.spec.path,
        "status": "available",
        "time_domain": domain,
        "pose_timestamp_source": f"{domain}_timestamp",
        "grid_timestamp_source": f"{domain}_timestamp",
        "pose_stamp_sec": critical_pair.pose_stamp_sec,
        "grid_stamp_sec": critical_pair.grid_stamp_sec,
        "delta_signed_pose_minus_grid_s": critical_pair.delta_signed_s,
        "delta_abs_s": critical_pair.delta_abs_s,
        "pose_x_m": critical_pose.x,
        "pose_y_m": critical_pose.y,
        "pose_yaw_rad": critical_pose.yaw,
        "pose_yaw_deg": math.degrees(critical_pose.yaw),
        "velocity_longitudinal_mps": critical_pose.vx,
        "velocity_planar_mps": speed_planar,
        "velocity_planar_kmh": speed_planar * 3.6,
        "velocity_source": f"{bag.topics.get('odometry', NA)}.twist.twist (measured/estimated odometry, not command)",
        "clearance_area_m": critical_evaluation.clearance_area_m,
        "clearance_center_diagnostic_m": critical_evaluation.clearance_center_m,
        "base_to_center_legacy_diagnostic_m": critical_evaluation.base_to_center_m,
        "occupied_cell_center_x_m": critical_evaluation.closest_cell_center_x,
        "occupied_cell_center_y_m": critical_evaluation.closest_cell_center_y,
        "grid_resolution_m": critical_grid.resolution,
        "occupancy_threshold": nominal_threshold,
        "footprint_source": critical_details["footprint_source"],
        "footprint_frame": critical_grid.frame_id,
        "footprint_length_m": footprint_length,
        "footprint_width_m": footprint_width,
        "limitation": critical_details["footprint_limitation"],
    }
    nominal_row = {
        "bag_key": bag.spec.key,
        "bag_label": bag.spec.label,
        "bag_path": bag.spec.path,
        "status": "available",
        "clearance_definition": CLEARANCE_DEFINITION,
        "nominal_clearance_m": critical_evaluation.clearance_area_m,
        "baseline_clearance_m": baseline if baseline is not None else NA,
        "v3_minus_baseline_m": baseline_delta if baseline_delta is not None else NA,
        "baseline_regression_absolute_difference_m": abs(baseline_delta) if baseline_delta is not None else NA,
        "baseline_regression_status": (
            "PASS_WITHIN_0.02M" if baseline_delta is not None and abs(baseline_delta) <= 0.02
            else "DIFFERENCE_REQUIRES_EXPLANATION" if baseline_delta is not None
            else "NO_BASELINE"
        ),
        "occupancy_threshold": nominal_threshold,
        "time_domain": domain,
        "maximum_pose_grid_delta_s": maximum_delta,
        "critical_pose_grid_delta_abs_s": critical_pair.delta_abs_s,
        "n_candidate_pose_grid_pairs": len(pairs),
        "n_accepted_pose_grid_pairs": len(accepted),
        "n_pairs_with_valid_nominal_clearance": len(nominal_evaluations),
        "footprint_source_at_critical_event": critical_details["footprint_source"],
        "fallback_footprint_source": fallback_source,
        "fallback_footprint_length_m": float(np.ptp(fallback_local[:, 0])),
        "fallback_footprint_width_m": float(np.ptp(fallback_local[:, 1])),
        "configured_footprint_padding_m": configured_padding if configured_padding is not None else NA,
        "grid_resolution_m_at_critical_event": critical_grid.resolution,
        "occupied_cells_at_critical_event": critical_evaluation.occupied_cells,
        "limitation": "; ".join(filter(None, [critical_details["footprint_limitation"], *bag.warnings])),
    }

    sensitivity_rows: List[Dict[str, Any]] = []
    for tolerance in tolerances:
        eligible = [
            item for item in nominal_evaluations
            if item[0].delta_abs_s <= tolerance
        ]
        best = min((item[1].clearance_area_m for item in eligible), default=None)
        sensitivity_rows.append(
            {
                "bag_key": bag.spec.key,
                "bag_label": bag.spec.label,
                "variant_type": "pose_grid_tolerance",
                "variant_value": tolerance,
                "variant_unit": "s",
                "occupancy_threshold": nominal_threshold,
                "clearance_min_m": best if best is not None else NA,
                "clearance_max_m": best if best is not None else NA,
                "N_evaluated": len(eligible),
                "method": "subset of nearest pose-grid pairs satisfying absolute delta",
                "status": "available" if best is not None else "unavailable",
                "limitation": "",
            }
        )

    thresholds = choose_occupancy_thresholds(bag.observed_occupancy_values, nominal_threshold)
    for threshold in thresholds:
        threshold_values: List[float] = []
        for pair in accepted:
            evaluation, _details = evaluate_pair(
                bag,
                pair,
                threshold,
                fallback_local,
                fallback_source,
                maximum_footprint_delta,
                max_cells,
            )
            if evaluation is not None and math.isfinite(evaluation.clearance_area_m):
                threshold_values.append(evaluation.clearance_area_m)
        best = min(threshold_values) if threshold_values else None
        sensitivity_rows.append(
            {
                "bag_key": bag.spec.key,
                "bag_label": bag.spec.label,
                "variant_type": "occupancy_threshold",
                "variant_value": threshold,
                "variant_unit": "OccupancyGrid value",
                "occupancy_threshold": threshold,
                "clearance_min_m": best if best is not None else NA,
                "clearance_max_m": best if best is not None else NA,
                "N_evaluated": len(threshold_values),
                "method": "all occupied cell areas at or above recorded-value threshold",
                "status": "available" if best is not None else "unavailable",
                "observed_nonnegative_grid_values": sorted(value for value in bag.observed_occupancy_values if value >= 0),
                "limitation": "Threshold alternatives are selected only from values actually recorded in the grids.",
            }
        )

    if configured_padding is not None:
        local_for_padding = critical_details.get("footprint_local_equivalent")
        if local_for_padding is None:
            local_for_padding = fallback_local
        padded = pad_convex_polygon(local_for_padding, configured_padding)
        padded_world = transform_polygon(padded, critical_pose.x, critical_pose.y, critical_pose.yaw)
        centers = occupancy_centers(critical_grid, nominal_threshold)
        padded_evaluation = minimum_clearance(
            padded_world,
            np.array([critical_pose.x, critical_pose.y]),
            centers,
            critical_grid.resolution,
            critical_grid.origin_yaw,
            nominal_threshold,
        )
        sensitivity_rows.append(
            {
                "bag_key": bag.spec.key,
                "bag_label": bag.spec.label,
                "variant_type": "configured_footprint_padding",
                "variant_value": configured_padding,
                "variant_unit": "m",
                "occupancy_threshold": nominal_threshold,
                "clearance_min_m": padded_evaluation.clearance_area_m,
                "clearance_max_m": padded_evaluation.clearance_area_m,
                "N_evaluated": 1,
                "method": "deterministic critical-grid recomputation with configured padding",
                "status": "available",
                "limitation": "Applied to a pose-local footprint representation; published footprint may already include padding.",
            }
        )
    else:
        sensitivity_rows.append(
            {
                "bag_key": bag.spec.key,
                "bag_label": bag.spec.label,
                "variant_type": "configured_footprint_padding",
                "variant_value": NA,
                "variant_unit": "m",
                "occupancy_threshold": nominal_threshold,
                "clearance_min_m": NA,
                "clearance_max_m": NA,
                "N_evaluated": 0,
                "method": "not executed",
                "status": "unavailable",
                "limitation": "No reliable footprint_padding value was identified in the supplied configuration.",
            }
        )

    local_for_covariance = critical_details.get("footprint_local_equivalent")
    if local_for_covariance is None:
        local_for_covariance = fallback_local
    centers_critical = occupancy_centers(critical_grid, nominal_threshold)
    sigma_levels = [float(value) for value in sensitivity_config.get("covariance_sigma_levels", [1.0, 2.0])]
    for sigma_level in sigma_levels:
        perturbations, covariance_note = deterministic_pose_perturbations(critical_pose, sigma_level)
        values: List[float] = []
        for x_value, y_value, yaw_value, _label in perturbations:
            perturbed_world = transform_polygon(local_for_covariance, x_value, y_value, yaw_value)
            evaluation = minimum_clearance(
                perturbed_world,
                np.array([x_value, y_value]),
                centers_critical,
                critical_grid.resolution,
                critical_grid.origin_yaw,
                nominal_threshold,
            )
            if math.isfinite(evaluation.clearance_area_m):
                values.append(evaluation.clearance_area_m)
        summary = numeric_summary(values)
        sensitivity_rows.append(
            {
                "bag_key": bag.spec.key,
                "bag_label": bag.spec.label,
                "variant_type": "pose_covariance_deterministic_envelope",
                "variant_value": sigma_level,
                "variant_unit": "sigma",
                "occupancy_threshold": nominal_threshold,
                "clearance_min_m": summary["min"],
                "clearance_median_m": summary["median"],
                "clearance_p95_m": summary["p95"],
                "clearance_max_m": summary["max"],
                "N_evaluated": summary["n"],
                "method": "deterministic principal-axis position and yaw perturbation envelope",
                "status": "available" if values else "unavailable",
                "limitation": covariance_note,
            }
        )

    return {
        "sync_pairs": sync_rows,
        "sync_summary": [sync_summary],
        "critical": [critical_row],
        "nominal": [nominal_row],
        "sensitivity": sensitivity_rows,
        "summary": {
            "status": "completed",
            "time_domain": domain,
            "time_domain_reason": domain_reason,
            "critical_clearance_m": critical_evaluation.clearance_area_m,
            "critical_pose_grid_delta_abs_s": critical_pair.delta_abs_s,
            "critical_speed_kmh": speed_planar * 3.6,
            "baseline_clearance_m": baseline,
            "v3_minus_baseline_m": baseline_delta,
            "selected_topics": bag.topics,
            "observed_occupancy_values": sorted(bag.observed_occupancy_values),
            "warnings": bag.warnings,
        },
    }


def plot_sync(sync_rows: Sequence[Mapping[str, Any]], output: Path, logger: Any) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Synchronization figure unavailable: matplotlib import failed: %s", exc)
        return
    groups: Dict[str, List[float]] = defaultdict(list)
    labels: Dict[str, str] = {}
    for row in sync_rows:
        value = finite_float(row.get("delta_abs_s"))
        if value is not None:
            groups[str(row["bag_key"])].append(value)
            labels[str(row["bag_key"])] = str(row["bag_label"])
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for key, values in groups.items():
        sorted_values = np.sort(np.asarray(values))
        fraction = np.arange(1, sorted_values.size + 1) / sorted_values.size
        axis.step(sorted_values, fraction, where="post", label=labels[key])
    for tolerance in (0.05, 0.10, 0.25, 0.50, 1.00):
        axis.axvline(tolerance, color="0.75", linewidth=0.7, linestyle="--")
    axis.set_xlabel("Absolute pose-grid timestamp delta [s]")
    axis.set_ylabel("Empirical cumulative fraction")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_sensitivity(rows: Sequence[Mapping[str, Any]], output: Path, logger: Any) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Sensitivity figure unavailable: matplotlib import failed: %s", exc)
        return
    groups: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    labels: Dict[str, str] = {}
    for row in rows:
        if row.get("variant_type") != "pose_grid_tolerance":
            continue
        x_value = finite_float(row.get("variant_value"))
        y_value = finite_float(row.get("clearance_min_m"))
        if x_value is None or y_value is None:
            continue
        key = str(row["bag_key"])
        groups[key].append((x_value, y_value))
        labels[key] = str(row["bag_label"])
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for key, values in groups.items():
        values = sorted(values)
        axis.plot([value[0] for value in values], [value[1] for value in values], marker="o", label=labels[key])
    axis.set_xlabel("Maximum pose-grid timestamp delta [s]")
    axis.set_ylabel("Minimum footprint-to-cell-area clearance [m]")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_pure_self_tests()
        print("recompute_field_metrics_from_bags_v3 pure self-tests: PASS")
        return 0

    project_root = Path(args.project_root).expanduser().resolve()
    config_path = resolve_path(args.config, project_root)
    config = load_yaml(config_path)
    specs = merge_bag_specs(bag_specs_from_config(config, project_root), args.bag)
    out_dir = resolve_path(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        fallback, source, padding, warnings = resolve_fallback_footprint(config, project_root)
        print(f"Config: {config_path}")
        print(f"Output: {out_dir}")
        print(f"Fallback footprint: {source}, size={np.ptp(fallback[:, 0]):.3f}x{np.ptp(fallback[:, 1]):.3f} m")
        print(f"Configured padding: {padding if padding is not None else 'NA'}")
        for warning in warnings:
            print(f"WARNING: {warning}")
        for spec in specs:
            print(f"Bag {spec.key}: {spec.label} -> {spec.path}")
        print("Dry run: PASS (bags were not opened)")
        return 0

    validate_bag_paths(specs)
    logger = setup_logger(out_dir / "field_metrics_v3.log", verbose=args.verbose)
    logger.info("Starting field metrics v3")
    all_sync_pairs: List[Dict[str, Any]] = []
    all_sync_summary: List[Dict[str, Any]] = []
    all_critical: List[Dict[str, Any]] = []
    all_nominal: List[Dict[str, Any]] = []
    all_sensitivity: List[Dict[str, Any]] = []
    bag_summaries: Dict[str, Any] = {}

    for spec in specs:
        try:
            bag = load_metric_bag(spec, config, logger)
            result = process_bag(bag, config, project_root, logger)
            all_sync_pairs.extend(result["sync_pairs"])
            all_sync_summary.extend(result["sync_summary"])
            all_critical.extend(result["critical"])
            all_nominal.extend(result["nominal"])
            all_sensitivity.extend(result["sensitivity"])
            bag_summaries[spec.key] = result["summary"]
            logger.info("%s processing status: %s", spec.label, result["summary"]["status"])
        except Exception as exc:
            logger.exception("Metric processing failed for %s", spec.label)
            unavailable = unavailable_rows(spec, str(exc))
            all_sync_summary.extend(unavailable["sync_summary"])
            all_critical.extend(unavailable["critical"])
            all_nominal.extend(unavailable["nominal"])
            all_sensitivity.extend(unavailable["sensitivity"])
            bag_summaries[spec.key] = {"status": "failed", "error": str(exc)}

    write_csv(out_dir / "field_pose_grid_sync_pairs_v3.csv", all_sync_pairs)
    write_csv(out_dir / "field_pose_grid_sync_summary_v3.csv", all_sync_summary)
    write_csv(out_dir / "field_critical_event_v3.csv", all_critical)
    write_csv(out_dir / "field_clearance_nominal_v3.csv", all_nominal)
    write_csv(out_dir / "field_clearance_sensitivity_v3.csv", all_sensitivity)
    plot_sync(all_sync_pairs, out_dir / "figures" / "field_pose_grid_sync_v3.png", logger)
    plot_sensitivity(all_sensitivity, out_dir / "figures" / "field_clearance_sensitivity_v3.png", logger)

    summary = {
        "status": (
            "completed_with_failures"
            if any(value.get("status") in {"failed", "unavailable"} for value in bag_summaries.values())
            else "completed"
        ),
        "clearance_definition": CLEARANCE_DEFINITION,
        "primary_metric_uses_cell_area": True,
        "occupied_cells_subsampled": False,
        "bags": bag_summaries,
        "environment": environment_manifest(),
        "outputs": {
            "sync_pairs": str(out_dir / "field_pose_grid_sync_pairs_v3.csv"),
            "sync_summary": str(out_dir / "field_pose_grid_sync_summary_v3.csv"),
            "critical_event": str(out_dir / "field_critical_event_v3.csv"),
            "nominal_clearance": str(out_dir / "field_clearance_nominal_v3.csv"),
            "sensitivity": str(out_dir / "field_clearance_sensitivity_v3.csv"),
        },
    }
    write_json(out_dir / "field_metrics_summary_v3.json", summary)
    write_yaml(
        out_dir / "metric_parameters_v3.yaml",
        {
            "config_path": str(config_path),
            "project_root": str(project_root),
            "bags": [spec.__dict__ for spec in specs],
            "synchronization": config.get("synchronization", {}),
            "clearance": config.get("clearance", {}),
            "sensitivity": config.get("sensitivity", {}),
        },
    )
    logger.info("Completed field metrics v3 with status=%s", summary["status"])
    return 1 if summary["status"] == "completed_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
