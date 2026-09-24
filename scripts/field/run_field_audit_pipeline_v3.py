#!/usr/bin/env python3
"""Orchestrate the complete v3 human-speed and field-rosbag audit."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from field_audit_common_v3 import (
    environment_manifest,
    load_yaml,
    resolve_path,
    setup_logger,
    sha256_file,
    write_csv,
    write_json,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run v3 tests, correct the human-reference speed table, inventory "
            "three field rosbags, and recompute synchronization/clearance audits."
        )
    )
    parser.add_argument("--project-root", default=".", help="NAV2_Paper_Scripts root.")
    parser.add_argument("--config", default="configs/field_audit_v3.yaml", help="Audit YAML.")
    parser.add_argument("--out-dir", default="results/field_audit_v3", help="Field output directory.")
    parser.add_argument(
        "--bag",
        action="append",
        default=[],
        help="Override YAML bags; repeat key:label:/absolute/path/to/bag.mcap.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable with ROS 2 modules.")
    parser.add_argument("--dry-run", action="store_true", help="Validate commands without opening rosbags.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit tests.")
    parser.add_argument("--skip-gt", action="store_true", help="Skip human-speed table v3 build.")
    parser.add_argument("--skip-inventory", action="store_true", help="Skip all-topic inventory/localization audit.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip synchronization/clearance processing.")
    parser.add_argument("--verbose", action="store_true", help="Verbose child scripts.")
    return parser.parse_args()


def run_command(command: Sequence[str], project_root: Path, logger: Any) -> int:
    logger.info("COMMAND: %s", " ".join(command))
    process = subprocess.Popen(
        list(command),
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        logger.info("CHILD | %s", line)
    return int(process.wait())


def read_json_if_present(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"status": "not_generated", "path": str(path)}
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as exc:
        return {"status": "invalid_json", "path": str(path), "error": str(exc)}


def traceability_rows(out_dir: Path, inventory_status: str, metrics_status: str) -> List[Dict[str, Any]]:
    return [
        {
            "advisor_request": "Temporal pose-grid synchronization",
            "analysis": "Nearest odometry pose for every obstacle grid in one timestamp domain; signed and absolute deltas",
            "rosbag": "all three field bags",
            "topic": "resolved odometry + resolved OccupancyGrid",
            "script": "scripts/field/recompute_field_metrics_from_bags_v3.py",
            "output": "field_pose_grid_sync_pairs_v3.csv; field_pose_grid_sync_summary_v3.csv",
            "result": metrics_status,
            "limitation": "Pairs outside 1.0 s are retained in the audit but rejected from clearance.",
        },
        {
            "advisor_request": "Speed at minimum-clearance event",
            "analysis": "Odometry longitudinal and planar speed at the critical synchronized pair",
            "rosbag": "all three field bags",
            "topic": "resolved odometry topic",
            "script": "scripts/field/recompute_field_metrics_from_bags_v3.py",
            "output": "field_critical_event_v3.csv",
            "result": metrics_status,
            "limitation": "Measured/estimated odometry twist; commanded velocity is not substituted.",
        },
        {
            "advisor_request": "Quantitative localization audit",
            "analysis": "Recorded covariance estimates, temporal heading variability, frames, TF and message regularity",
            "rosbag": "all three field bags",
            "topic": "odometry, GNSS, IMU, TF and discovered alternatives",
            "script": "scripts/field/audit_field_rosbags_v3.py",
            "output": "field_localization_quality_v3.csv; field_topic_rates_v3.csv",
            "result": inventory_status,
            "limitation": "Covariance is not absolute error; no accuracy claim without surveyed ground truth.",
        },
        {
            "advisor_request": "Explicit RTK status",
            "analysis": "Search explicit receiver status topics and preserve receiver-specific fields",
            "rosbag": "all three field bags",
            "topic": "discovered explicit RTK/GNSS receiver topic",
            "script": "scripts/field/audit_field_rosbags_v3.py",
            "output": "field_localization_quality_v3.csv",
            "result": inventory_status,
            "limitation": "NavSatFix.status alone is never relabelled as RTK fixed/float.",
        },
        {
            "advisor_request": "Nominal footprint clearance",
            "analysis": "Minimum polygon-to-occupied-cell-area distance using all synchronized grids",
            "rosbag": "human, NavFn-MPPI, SMAC-MPPI",
            "topic": "obstacle grid; published footprint when recorded; odometry fallback",
            "script": "scripts/field/recompute_field_metrics_from_bags_v3.py",
            "output": "field_clearance_nominal_v3.csv",
            "result": metrics_status,
            "limitation": "Recorded discretized costmap representation; N=1 per condition.",
        },
        {
            "advisor_request": "Clearance sensitivity and uncertainty",
            "analysis": "Timing tolerance, recorded occupancy thresholds, padding availability, and deterministic 1σ/2σ covariance envelopes",
            "rosbag": "all three field bags",
            "topic": "grid, footprint and odometry covariance",
            "script": "scripts/field/recompute_field_metrics_from_bags_v3.py",
            "output": "field_clearance_sensitivity_v3.csv; figures/field_clearance_sensitivity_v3.png",
            "result": metrics_status,
            "limitation": "Unavailable covariance/padding is reported as NA; no distribution is invented.",
        },
        {
            "advisor_request": "Topic inventory and rates",
            "analysis": "Counts, time spans, mean frequency, median/P95/max intervals, gaps, frames, headers and covariance availability",
            "rosbag": "all three field bags",
            "topic": "all recorded topics",
            "script": "scripts/field/audit_field_rosbags_v3.py",
            "output": "field_bag_inventory_v3.csv; field_topic_rates_v3.csv",
            "result": inventory_status,
            "limitation": "Unknown custom message packages may prevent field deserialization, but storage timing remains inventoried.",
        },
    ]


def write_release_manifest(project_root: Path, out_dir: Path) -> Path:
    candidates = [
        project_root / "scripts/human_reference/build_gt_final_table_v3.py",
        project_root / "scripts/field/field_audit_common_v3.py",
        project_root / "scripts/field/audit_field_rosbags_v3.py",
        project_root / "scripts/field/recompute_field_metrics_from_bags_v3.py",
        project_root / "scripts/field/run_field_audit_pipeline_v3.py",
        project_root / "configs/field_audit_v3.yaml",
        project_root / "requirements_v3.txt",
        project_root / "README.md",
        project_root / "docs/FIELD_AUDIT_V3_README.md",
        project_root / "docs/PUBLIC_RELEASE_CHECKLIST_v3.md",
        project_root / "docs/V3_VALIDATION_REPORT_2026-08-29.md",
        project_root / "docs/v3_acceptance_criteria.md",
        project_root / "docs/v3_script_audit.md",
        project_root / "tests/test_field_audit_v3.py",
        project_root / "tests/test_gt_speed_mapping_v3.py",
    ]
    candidates.extend(sorted(path for path in out_dir.rglob("*") if path.is_file()))
    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    manifest = out_dir / "release_manifest_v3.sha256"
    lines = [f"{sha256_file(path)}  {path.relative_to(project_root)}" for path in unique if path != manifest]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def protected_baseline_hashes(project_root: Path) -> Dict[str, str]:
    protected_paths: List[Path] = []
    for relative in (
        "scripts/simulation",
        "results/supplementary_tables_final_plan_cross_track_v2",
        "results/supplementary_tables_reanalysis_plan_cross_track_v2",
    ):
        root = project_root / relative
        if root.is_dir():
            protected_paths.extend(path for path in root.rglob("*") if path.is_file())
    return {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in sorted(protected_paths)
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    config_path = resolve_path(args.config, project_root)
    out_dir = resolve_path(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path)
    logger = setup_logger(out_dir / "field_audit_v3.log", verbose=args.verbose)
    started = datetime.now(timezone.utc).isoformat()
    commands: List[List[str]] = []
    baseline_before = protected_baseline_hashes(project_root)

    if not args.skip_tests:
        commands.append(
            [args.python, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*_v3.py", "-v"]
        )
    if not args.skip_gt:
        commands.append(
            [
                args.python,
                "scripts/human_reference/build_gt_final_table_v3.py",
                "--project-root",
                str(project_root),
                "--out-dir",
                "results/gt_final_table_v3",
            ]
        )

    common_field_args = [
        "--project-root", str(project_root),
        "--config", str(config_path),
        "--out-dir", str(out_dir),
    ]
    for bag in args.bag:
        common_field_args.extend(["--bag", bag])
    if args.dry_run:
        common_field_args.append("--dry-run")
    if args.verbose:
        common_field_args.append("--verbose")
    if not args.skip_inventory:
        commands.append([args.python, "scripts/field/audit_field_rosbags_v3.py", *common_field_args])
    if not args.skip_metrics:
        commands.append([args.python, "scripts/field/recompute_field_metrics_from_bags_v3.py", *common_field_args])

    command_results: List[Dict[str, Any]] = []
    overall_failure = False
    for command in commands:
        return_code = run_command(command, project_root, logger)
        command_results.append({"command": command, "return_code": return_code})
        if return_code != 0:
            overall_failure = True
            logger.error(
                "Command returned %d; continuing the remaining independent phases so that "
                "available metrics are still exported",
                return_code,
            )

    baseline_after = protected_baseline_hashes(project_root)
    baseline_changes = sorted(
        path
        for path in set(baseline_before) | set(baseline_after)
        if baseline_before.get(path) != baseline_after.get(path)
    )
    if baseline_changes:
        overall_failure = True
        logger.error("Protected plan_cross_track_v2/simulation files changed: %s", baseline_changes)

    inventory_summary = read_json_if_present(out_dir / "field_inventory_summary_v3.json")
    metrics_summary = read_json_if_present(out_dir / "field_metrics_summary_v3.json")
    inventory_status = "dry_run" if args.dry_run else str(inventory_summary.get("status", "not_generated"))
    metrics_status = "dry_run" if args.dry_run else str(metrics_summary.get("status", "not_generated"))
    write_csv(
        out_dir / "advisor_request_traceability_v3.csv",
        traceability_rows(out_dir, inventory_status, metrics_status),
        ["advisor_request", "analysis", "rosbag", "topic", "script", "output", "result", "limitation"],
    )
    parameters = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "project_root": str(project_root),
        "config_path": str(config_path),
        "output_directory": str(out_dir),
        "cli_bag_overrides": args.bag,
        "config": config,
        "commands": command_results,
        "environment": environment_manifest(),
        "protected_baseline_file_count": len(baseline_before),
        "protected_baseline_changes": baseline_changes,
    }
    write_yaml(out_dir / "parameters_used_v3.yaml", parameters)
    summary = {
        "status": "failed" if overall_failure else "dry_run" if args.dry_run else "completed",
        "important_interpretation": [
            "Writing scripts does not complete the advisor requests; actual bag outputs must be validated.",
            "Field N remains one run per condition and is illustrative.",
            "Recorded covariance is not absolute localization error.",
            "plan_cross_track_v2 and the 212-run population audit are not modified by this pipeline.",
        ],
        "inventory": inventory_summary,
        "field_metrics": metrics_summary,
        "commands": command_results,
        "protected_baseline_guard": {
            "status": "PASS" if not baseline_changes else "FAIL",
            "checked_file_count": len(baseline_before),
            "changed_files": baseline_changes,
        },
        "traceability": str(out_dir / "advisor_request_traceability_v3.csv"),
        "parameters": str(out_dir / "parameters_used_v3.yaml"),
    }
    write_json(out_dir / "field_audit_summary_v3.json", summary)
    logger.info("Pipeline status: %s", summary["status"])
    logger.info("Finalizing release manifest in %s", out_dir)
    # The main log is itself included in the manifest. Flush and close it
    # before hashing so that every recorded digest remains verifiable.
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    write_release_manifest(project_root, out_dir)
    return 1 if overall_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
