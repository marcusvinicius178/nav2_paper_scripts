#!/usr/bin/env python3
"""Run the complete simulation reanalysis used to build Tables S2-S4.

This wrapper encodes the experiment inventory and the simulation-to-human
reference mapping used in the study:

* simulation 20 km/h -> human reference 15 km/h
* simulation 25 km/h -> human reference 20 km/h
* diagnostic RPP 30 km/h -> human reference 15 km/h
* diagnostic RPP 35 km/h -> human reference 20 km/h

All outputs are written outside ``RUNS``. Mission completion remains referenced
to the common route, active-plan cross-track metrics use the recorded
Navigation2 plan, and human-likeness remains referenced to the human GPS data.
Existing outputs are reused only when they carry the complete v2 cross-track
method provenance and contain the same number of runs as the input combination.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


COMBO_RE = re.compile(
    r"^(NAVFN|SMAC)_(MPPI|RPP)_Vel_(\d+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: int
    directory: str
    waypoint_filename: str
    human_per_bag_directory: str


SCENARIOS: Sequence[ScenarioSpec] = (
    ScenarioSpec(
        1,
        "Scenario_1_Reta",
        "scenario1_waypoints_debug.yaml",
        "1-Scenario_reta_arco_Ground_truth",
    ),
    ScenarioSpec(
        2,
        "Scenario_2_Reta_Arco_20",
        "scenario2_waypoints_debug.yaml",
        "2-Scenario_reta_arco_20_Ground_truth",
    ),
    ScenarioSpec(
        3,
        "Scenario_3_Reta_Arco_40",
        "scenario3_waypoints_debug.yaml",
        "3-Scenario_reta_arco_40_Ground_truth",
    ),
)


@dataclass(frozen=True)
class Combination:
    scenario: ScenarioSpec
    directory: Path
    planner: str
    controller: str
    speed_kmh: int
    human_speed_kmh: int
    condition: str
    bag_count: int


def parse_args() -> argparse.Namespace:
    project_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Process all simulation bags and build Supplementary Tables S2-S4."
    )
    parser.add_argument(
        "--project-root",
        default=str(project_default),
        help="NAV2_Paper_Scripts root.",
    )
    parser.add_argument(
        "--runs-root",
        default="",
        help="RUNS root (default: <project-root>/RUNS).",
    )
    parser.add_argument(
        "--results-root",
        default="",
        help=(
            "Per-combination output root "
            "(default: <project-root>/results/supplementary_tables_reanalysis_plan_cross_track_v2)."
        ),
    )
    parser.add_argument(
        "--tables-out-dir",
        default="",
        help=(
            "Final table output directory "
            "(default: <project-root>/results/supplementary_tables_final_plan_cross_track_v2)."
        ),
    )
    parser.add_argument(
        "--figures-out-dir",
        default="",
        help=(
            "Final Figures 5-7 output directory "
            "(default: <project-root>/results/corrected_figures_5_7_plan_cross_track_v2)."
        ),
    )
    parser.add_argument(
        "--only-glob",
        default="*",
        help="Process only combination directory names matching this glob.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        type=int,
        choices=(1, 2, 3),
        help="Limit processing to a scenario; repeat for multiple scenarios.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess combinations even when a valid completed CSV exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with other combinations after a processing failure.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a partial inventory, intended only for labelled test/draft runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without processing bags.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Process combinations without invoking the S2-S4 table builder.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Build tables without regenerating Figures 5-7.",
    )
    parser.add_argument(
        "--figure-speed-kmh",
        type=int,
        default=25,
        help="Nominal speed used in Figures 5-7 (default: 25 km/h).",
    )
    parser.add_argument("--gps-topic", default="/gps/fix")
    parser.add_argument("--human-resample-samples", type=int, default=300)
    parser.add_argument(
        "--plan-topics",
        default="/plan,/received_global_plan,/transformed_global_plan,/plan_smoothed",
        help="Comma-separated priority for the recorded active Nav2 tracking plan.",
    )
    parser.add_argument(
        "--executed-topic",
        default="/odometry/global",
        help="Executed vehicle pose topic (default: /odometry/global).",
    )
    parser.add_argument(
        "--min-plan-coverage",
        type=float,
        default=0.80,
        help="Minimum active-plan coverage for plan-tracking validity (default: 0.80).",
    )
    parser.add_argument(
        "--wheelbase-m",
        type=float,
        default=6.804,
        help="Equivalent Ackermann wheelbase for steering reconstruction (default: 6.804 m).",
    )
    parser.add_argument(
        "--steering-min-speed-mps",
        type=float,
        default=0.50,
        help="Minimum absolute cmd_vel.linear.x used for steering reconstruction (default: 0.50 m/s).",
    )
    return parser.parse_args()


def normalize_planner(raw: str) -> str:
    return "NavFn" if raw.upper() == "NAVFN" else "SMAC"


def normalize_controller(raw: str) -> str:
    return raw.upper()


def human_speed_for_simulation(speed_kmh: int) -> int:
    mapping = {20: 15, 25: 20, 30: 15, 35: 20}
    if speed_kmh not in mapping:
        raise ValueError(f"No human-reference mapping defined for {speed_kmh} km/h")
    return mapping[speed_kmh]


def condition_for(speed_kmh: int, controller: str) -> str:
    if speed_kmh in {20, 25}:
        return "nominal"
    if controller == "RPP":
        return "diagnostic_RPP"
    return "diagnostic_other"


def count_bags(combination_dir: Path) -> int:
    return sum(1 for _ in combination_dir.rglob("metadata.yaml"))


def discover_combinations(
    runs_root: Path,
    only_glob: str,
    selected_scenarios: Optional[Sequence[int]],
) -> List[Combination]:
    combinations: List[Combination] = []
    selected = set(selected_scenarios or (1, 2, 3))

    for scenario in SCENARIOS:
        if scenario.scenario_id not in selected:
            continue
        scenario_dir = runs_root / scenario.directory
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        for combo_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            if not fnmatch.fnmatch(combo_dir.name, only_glob):
                continue
            match = COMBO_RE.match(combo_dir.name)
            if not match:
                print(f"[SKIP] Unrecognized combination directory: {combo_dir}")
                continue

            planner = normalize_planner(match.group(1))
            controller = normalize_controller(match.group(2))
            speed = int(match.group(3))
            combinations.append(
                Combination(
                    scenario=scenario,
                    directory=combo_dir,
                    planner=planner,
                    controller=controller,
                    speed_kmh=speed,
                    human_speed_kmh=human_speed_for_simulation(speed),
                    condition=condition_for(speed, controller),
                    bag_count=count_bags(combo_dir),
                )
            )

    return combinations


def expected_count(combo: Combination) -> Optional[int]:
    if combo.scenario.scenario_id in {1, 3}:
        if combo.speed_kmh in {20, 25}:
            return 10
        return None
    if combo.scenario.scenario_id == 2:
        if combo.controller == "MPPI" and combo.speed_kmh in {20, 25}:
            return 10
        if combo.controller == "RPP" and combo.speed_kmh in {30, 35}:
            return 3
    return None


def expected_combination_names() -> Dict[Tuple[int, str], int]:
    expected: Dict[Tuple[int, str], int] = {}
    for scenario_id in (1, 3):
        for planner in ("NAVFN", "SMAC"):
            for controller in ("MPPI", "RPP"):
                for speed in (20, 25):
                    expected[(scenario_id, f"{planner}_{controller}_Vel_{speed}")] = 10
    for planner in ("NAVFN", "SMAC"):
        for speed in (20, 25):
            expected[(2, f"{planner}_MPPI_Vel_{speed}")] = 10
        for speed in (30, 35):
            expected[(2, f"{planner}_RPP_Vel_{speed}")] = 3
    return expected


def inventory_errors(combinations: Sequence[Combination], full_selection: bool) -> List[str]:
    errors: List[str] = []
    for combo in combinations:
        wanted = expected_count(combo)
        if wanted is None:
            errors.append(
                f"Unexpected condition: S{combo.scenario.scenario_id} "
                f"{combo.directory.name} ({combo.bag_count} bags)"
            )
        elif combo.bag_count != wanted:
            errors.append(
                f"Run-count mismatch: S{combo.scenario.scenario_id} {combo.directory.name}: "
                f"found={combo.bag_count}, expected={wanted}"
            )

    if full_selection:
        expected = expected_combination_names()
        actual = {
            (combo.scenario.scenario_id, combo.directory.name): combo.bag_count
            for combo in combinations
        }
        for key in sorted(set(expected) - set(actual)):
            errors.append(f"Missing combination: S{key[0]} {key[1]}")
        for key in sorted(set(actual) - set(expected)):
            errors.append(f"Unexpected combination: S{key[0]} {key[1]}")
        total = sum(combo.bag_count for combo in combinations)
        if total != 212:
            errors.append(f"Total run count is {total}; expected 212")
    return errors


def read_per_run_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def completed_output_is_reusable(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_per_run_csv(path)
    except Exception:
        return False
    if len(rows) != expected_rows:
        return False
    required_columns = {
        "valid_for_mission",
        "valid_for_tracking",
        "plan_topic_used",
        "plan_endpoint_extension_samples",
        "plan_endpoint_extension_fraction",
    }
    if not rows or not required_columns.issubset(rows[0]):
        return False
    for row in rows:
        notes = str(row.get("notes", ""))
        primary_reference = str(row.get("primary_reference", ""))
        valid_for_tracking = int(float(row.get("valid_for_tracking", 0) or 0))
        if valid_for_tracking == 1 and not primary_reference.startswith(
            "nav2_active_plan_cross_track_v2:"
        ):
            return False
        if "PLAN_CROSSTRACK_DEF=POLYLINE_SEGMENTS_WITH_ENDPOINT_TANGENT_EXTENSION" not in notes:
            return False
        if "JERK_DEF=d2(smoothed_speed)/dt2" not in notes:
            return False
        if "STEERING_RATE_DEF=d_atan(L*wz/vx)/dt" not in notes:
            return False
    return True


def human_reference_paths(
    project_root: Path, combo: Combination
) -> Tuple[Path, str, Path]:
    human_speed = combo.human_speed_kmh
    refs_root = project_root / "gt_from_bag_outputs_aligned"
    gt_csv = (
        refs_root
        / "gt_mean_references"
        / (
            f"scenario{combo.scenario.scenario_id}_vel{human_speed}"
            "_GTmean_rigidAligned_from_bag.csv"
        )
    )
    gps_glob = str(
        refs_root
        / "per_bag"
        / combo.scenario.human_per_bag_directory
        / f"*vel_{human_speed}*"
        / "gps_fix_latlon.csv"
    )
    waypoint = (
        project_root
        / "data"
        / "waypoints"
        / combo.scenario.waypoint_filename
    )
    return gt_csv, gps_glob, waypoint


def validate_reference_files(project_root: Path, combos: Iterable[Combination]) -> List[str]:
    errors: List[str] = []
    checked: set[Tuple[str, str, str]] = set()
    for combo in combos:
        gt_csv, gps_glob, waypoint = human_reference_paths(project_root, combo)
        key = (str(gt_csv), gps_glob, str(waypoint))
        if key in checked:
            continue
        checked.add(key)
        if not gt_csv.is_file():
            errors.append(f"Missing auxiliary GT mean CSV: {gt_csv}")
        gps_matches = sorted(glob.glob(gps_glob))
        if not gps_matches:
            errors.append(f"No human GPS CSV matches: {gps_glob}")
        if not waypoint.is_file():
            errors.append(f"Missing waypoint file: {waypoint}")
    return errors


def stream_command(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command))
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_stream.write(line)
        return int(process.wait())


def print_inventory(combinations: Sequence[Combination]) -> None:
    print("\n[INVENTORY]")
    for combo in combinations:
        print(
            f"S{combo.scenario.scenario_id} | {combo.condition:14s} | "
            f"{combo.directory.name:24s} | bags={combo.bag_count:2d} | "
            f"human_ref={combo.human_speed_kmh} km/h"
        )
    print(f"Combinations: {len(combinations)}")
    print(f"Runs: {sum(combo.bag_count for combo in combinations)}")


def main() -> int:
    args = parse_args()
    if args.wheelbase_m <= 0.0:
        print("ERROR: --wheelbase-m must be positive.", file=sys.stderr)
        return 8
    if args.steering_min_speed_mps < 0.0:
        print("ERROR: --steering-min-speed-mps cannot be negative.", file=sys.stderr)
        return 9
    if not 0.0 <= args.min_plan_coverage <= 1.0:
        print("ERROR: --min-plan-coverage must be in [0, 1].", file=sys.stderr)
        return 10
    if args.figure_speed_kmh <= 0:
        print("ERROR: --figure-speed-kmh must be positive.", file=sys.stderr)
        return 11

    project_root = Path(args.project_root).expanduser().resolve()
    runs_root = (
        Path(args.runs_root).expanduser().resolve()
        if args.runs_root
        else project_root / "RUNS"
    )
    results_root = (
        Path(args.results_root).expanduser().resolve()
        if args.results_root
        else project_root / "results" / "supplementary_tables_reanalysis_plan_cross_track_v2"
    )
    tables_out_dir = (
        Path(args.tables_out_dir).expanduser().resolve()
        if args.tables_out_dir
        else project_root / "results" / "supplementary_tables_final_plan_cross_track_v2"
    )
    figures_out_dir = (
        Path(args.figures_out_dir).expanduser().resolve()
        if args.figures_out_dir
        else project_root / "results" / "corrected_figures_5_7_plan_cross_track_v2"
    )
    script_dir = Path(__file__).resolve().parent
    eval_script = script_dir / "eval_nav2_one_combination.py"
    table_builder = script_dir / "build_supplementary_tables.py"
    figure_builder = script_dir / "plot_corrected_figures_5_7.py"

    required = [project_root, runs_root, eval_script, table_builder]
    if not args.no_figures:
        required.append(figure_builder)
    missing_required = [path for path in required if not path.exists()]
    if missing_required:
        for path in missing_required:
            print(f"ERROR: required path not found: {path}", file=sys.stderr)
        return 2

    selected_scenarios = args.scenario_id or [1, 2, 3]
    try:
        combinations = discover_combinations(
            runs_root,
            only_glob=args.only_glob,
            selected_scenarios=selected_scenarios,
        )
    except Exception as exc:
        print(f"ERROR: could not discover combinations: {exc}", file=sys.stderr)
        return 3

    if not combinations:
        print("ERROR: no combinations matched the requested selection", file=sys.stderr)
        return 4

    print_inventory(combinations)
    full_selection = args.only_glob == "*" and set(selected_scenarios) == {1, 2, 3}
    errors = inventory_errors(combinations, full_selection=full_selection)
    errors.extend(validate_reference_files(project_root, combinations))
    if errors:
        print("\n[PREFLIGHT ERRORS]", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        if not args.allow_incomplete:
            print(
                "ERROR: preflight failed. Nothing was processed. "
                "Use --allow-incomplete only for an explicitly labelled test.",
                file=sys.stderr,
            )
            return 5

    if importlib.util.find_spec("rosbag2_py") is None and not args.dry_run:
        print(
            "ERROR: rosbag2_py is unavailable. Source ROS first, for example: "
            "source /opt/ros/iron/setup.bash",
            file=sys.stderr,
        )
        return 6

    failures: List[str] = []
    processed = 0
    skipped = 0
    logs_root = results_root / "_logs"

    for index, combo in enumerate(combinations, start=1):
        combo_output = results_root / combo.scenario.directory / combo.directory.name
        per_run_csv = combo_output / "per_run_metrics.csv"
        print(
            f"\n[{index}/{len(combinations)}] S{combo.scenario.scenario_id} "
            f"{combo.directory.name} ({combo.condition})"
        )

        if (
            not args.force
            and completed_output_is_reusable(per_run_csv, combo.bag_count)
        ):
            print(f"[SKIP] Reusing completed plan-cross-track-v2 output: {per_run_csv}")
            skipped += 1
            continue

        gt_csv, gps_glob, waypoint = human_reference_paths(project_root, combo)
        command = [
            sys.executable,
            str(eval_script),
            "--runs-dir",
            str(combo.directory),
            "--scenario-id",
            str(combo.scenario.scenario_id),
            "--speed-kmh",
            str(combo.speed_kmh),
            "--planner-id",
            combo.planner,
            "--controller-id",
            combo.controller,
            "--gt-csv",
            str(gt_csv),
            "--gt-gps-csv-glob",
            gps_glob,
            "--waypoint-file",
            str(waypoint),
            "--gps-topic",
            args.gps_topic,
            "--executed-topic",
            args.executed_topic,
            "--human-resample-samples",
            str(args.human_resample_samples),
            "--plan-topics",
            args.plan_topics,
            "--min-plan-coverage",
            str(args.min_plan_coverage),
            "--wheelbase-m",
            str(args.wheelbase_m),
            "--steering-min-speed-mps",
            str(args.steering_min_speed_mps),
            "--out-dir",
            str(combo_output),
        ]

        if args.dry_run:
            print("$ " + " ".join(command))
            continue

        log_path = (
            logs_root
            / f"S{combo.scenario.scenario_id}__{combo.directory.name}.log"
        )
        return_code = stream_command(command, log_path)
        if return_code != 0:
            message = (
                f"S{combo.scenario.scenario_id}/{combo.directory.name} "
                f"failed with rc={return_code}; log={log_path}"
            )
            failures.append(message)
            print(f"[ERROR] {message}", file=sys.stderr)
            if not args.continue_on_error:
                return return_code
        else:
            processed += 1

    print(
        f"\n[PROCESSING SUMMARY] processed={processed}, reused={skipped}, "
        f"failed={len(failures)}"
    )
    for failure in failures:
        print(f"- {failure}")

    if args.dry_run or args.no_build:
        return 0 if not failures else 7

    if failures and not args.allow_incomplete:
        print("ERROR: table build skipped because processing failures occurred.", file=sys.stderr)
        return 7

    build_command = [
        sys.executable,
        str(table_builder),
        "--results-root",
        str(results_root),
        "--out-dir",
        str(tables_out_dir),
        "--expected-total-runs",
        "212",
    ]
    if args.allow_incomplete or not full_selection:
        build_command.append("--allow-incomplete")

    print("\n[BUILD TABLES S2-S4]")
    build_return_code = stream_command(
        build_command,
        tables_out_dir / "build_tables.log",
    )
    if build_return_code != 0 or args.no_figures:
        return build_return_code

    figure_command = [
        sys.executable,
        str(figure_builder),
        "--project-root",
        str(project_root),
        "--tables-dir",
        str(tables_out_dir),
        "--out-dir",
        str(figures_out_dir),
        "--speed-kmh",
        str(args.figure_speed_kmh),
    ]
    print("\n[BUILD FIGURES 5-7]")
    return stream_command(
        figure_command,
        figures_out_dir / "build_figures.log",
    )


if __name__ == "__main__":
    raise SystemExit(main())
