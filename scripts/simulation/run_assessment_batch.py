#!/usr/bin/env python3
"""
Run both pipeline steps (charts + metrics) for multiple Nav2 experiment combinations.

It calls, in order, for each combination directory:
  1) debug_raw_gps_waypoint_alignment.py  -> Output_charts_diagnostic/
  2) eval_nav2_one_combination.py         -> Output_metrics_assessment/

Assumptions (matching your current project layout):
- RUNS root contains scenario directories, each contains combination directories, e.g.:
    RUNS/Scenario_1_Reta/NAVFN_MPPI_Vel_20/...
- Each combination directory contains multiple run folders with metadata.yaml, e.g.:
    Navfn_mppi_1, Navfn_mppi_2, ..., Navfn_mppi_10

- The "legacy" GT mean CSV is available at:
    <scripts-root>/gt_from_bag_outputs_aligned/gt_mean_references/
        scenario{scenario_id}_vel{speed}_GTmean_rigidAligned_from_bag.csv
  (eval script accepts it but ignores for metrics in the current methodology)

- Human-likeness reference:
    You must provide:
      --gt-gps-csv-glob (per scenario)
      --waypoint-file  (per scenario)

Usage examples
--------------
Scenario 1 only:
  source /opt/ros/jazzy/setup.bash
  python3 /home/marcus/NAV2_Paper_Scripts/run_assessment_batch.py \
    --runs-root /home/marcus/NAV2_Paper_Scripts/RUNS \
    --scenario Scenario_1_Reta:1:/home/marcus/NAV2_Paper_Scripts/scenario1_waypoints_debug.yaml:'/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/1-Scenario_reta_arco_Ground_truth/*vel_*/gps_fix_latlon.csv'

All scenarios (example placeholders for waypoint files):
  python3 run_assessment_batch.py \
    --runs-root /home/marcus/NAV2_Paper_Scripts/RUNS \
    --scenario Scenario_1_Reta:1:/home/marcus/NAV2_Paper_Scripts/scenario1_waypoints_debug.yaml:'/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/1-Scenario_reta_arco_Ground_truth/*vel_*/gps_fix_latlon.csv' \
    --scenario Scenario_2_Reta_Arco_20:2:/home/marcus/NAV2_Paper_Scripts/scenario2_waypoints_debug.yaml:'/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/2-Scenario_reta_arco_20_Ground_truth/*vel_*/gps_fix_latlon.csv' \
    --scenario Scenario_3_Reta_Arco_40:3:/home/marcus/NAV2_Paper_Scripts/scenario3_waypoints_debug.yaml:'/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/3-Scenario_reta_arco_40_Ground_truth/*vel_*/gps_fix_latlon.csv'

Notes
-----
- By default, it processes all combinations under each scenario.
- Use --only-glob to restrict combinations, e.g. 'NAVFN_MPPI_Vel_*'
- Use --continue-on-error to skip failures and keep going.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RUN_SUFFIX_RE = re.compile(r"(.*)_(\d+)$")  # e.g. Navfn_mppi_10 -> (Navfn_mppi, 10)
VEL_RE = re.compile(r"_Vel_(\d+)$", re.IGNORECASE)


@dataclass
class ScenarioCfg:
    scenario_dir_name: str
    scenario_id: int
    waypoint_file: Path
    gt_gps_csv_glob: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run charts + metrics for all combinations in RUNS.")

    p.add_argument("--runs-root", required=True, help="Path to RUNS directory")

    p.add_argument(
        "--scenario",
        action="append",
        default=[],
        help=(
            "Scenario configuration as: <scenario_dir_name>:<scenario_id>:<waypoint_file>:'<gt_gps_csv_glob>'\n"
            "Example: Scenario_1_Reta:1:/home/marcus/.../scenario1_waypoints_debug.yaml:'.../per_bag/1-Scenario.../*vel_*/gps_fix_latlon.csv'\n"
            "Repeat --scenario for multiple scenarios."
        ),
    )

    p.add_argument(
        "--only-glob",
        default="*",
        help="Glob to filter combination directories inside each scenario (default: '*')",
    )

    p.add_argument(
        "--gps-topic",
        default="/gps/fix",
        help="GPS topic to use for human-likeness metrics (default: /gps/fix)",
    )

    p.add_argument(
        "--human-resample-samples",
        type=int,
        default=300,
        help="Resample samples used in human-likeness computations (default: 300)",
    )

    p.add_argument(
        "--gt-mean-dir",
        default="",
        help=(
            "Directory containing legacy GT mean CSVs. "
            "Default: <scripts-root>/gt_from_bag_outputs_aligned/gt_mean_references"
        ),
    )

    p.add_argument(
        "--scripts-root",
        default="",
        help="Path to NAV2_Paper_Scripts root (default: directory of this script).",
    )

    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other combinations even if one fails.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )

    return p.parse_args()


def parse_scenario_cfg(s: str) -> ScenarioCfg:
    # format: dir:ID:waypoint:'glob'
    # allow glob to contain ":" by only splitting first 3 fields
    parts = s.split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --scenario '{s}'. Expected: <dir>:<id>:<waypoint_file>:'<gt_gps_csv_glob>'"
        )
    scenario_dir = parts[0].strip()
    scenario_id = int(parts[1].strip())
    waypoint_file = Path(parts[2].strip()).expanduser().resolve()
    gt_gps_glob = parts[3].strip().strip("'").strip('"')
    return ScenarioCfg(
        scenario_dir_name=scenario_dir,
        scenario_id=scenario_id,
        waypoint_file=waypoint_file,
        gt_gps_csv_glob=gt_gps_glob,
    )


def detect_sim_glob(combination_dir: Path) -> str:
    """
    Detect run directory name prefix like 'Navfn_mppi' from folders:
      Navfn_mppi_1, Navfn_mppi_2, ...
    Returns glob pattern like: 'Navfn_mppi_*'
    """
    candidates: List[str] = []
    for meta in combination_dir.rglob("metadata.yaml"):
        run_dir = meta.parent.name
        m = RUN_SUFFIX_RE.match(run_dir)
        if m:
            candidates.append(m.group(1))

    if not candidates:
        # fallback: any immediate subdir name without suffix
        subs = [p.name for p in combination_dir.iterdir() if p.is_dir()]
        if subs:
            return f"{subs[0]}*"
        return "*"

    # most common prefix
    counts: Dict[str, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return f"{best}_*"


def parse_speed_kmh_from_combo_dir(name: str) -> Optional[int]:
    m = VEL_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def parse_method_from_combo_dir(name: str) -> Tuple[str, str]:
    """
    Example dir name: NAVFN_MPPI_Vel_20 -> ('NavFn','MPPI')
                     SMAC_RPP_Vel_25  -> ('Smac','RPP')
    """
    base = name
    base = re.sub(r"_Vel_\d+$", "", base, flags=re.IGNORECASE)
    parts = base.split("_")
    if len(parts) < 2:
        return ("Unknown", "Unknown")

    planner_raw = parts[0].upper()
    ctrl_raw = parts[1].upper()

    planner_map = {
        "NAVFN": "NavFn",
        "SMAC": "Smac",
    }
    ctrl_map = {
        "MPPI": "MPPI",
        "RPP": "RPP",
        "PUREPURSUIT": "RPP",
    }

    return (planner_map.get(planner_raw, parts[0]), ctrl_map.get(ctrl_raw, parts[1]))


def run_cmd(cmd: List[str], dry_run: bool) -> int:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd)
    return int(p.returncode)


def main() -> int:
    args = parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists():
        print(f"ERROR: runs root does not exist: {runs_root}", file=sys.stderr)
        return 2

    scripts_root = Path(args.scripts_root).expanduser().resolve() if args.scripts_root else Path(__file__).resolve().parent

    debug_script = scripts_root / "debug_raw_gps_waypoint_alignment.py"
    eval_script = scripts_root / "eval_nav2_one_combination.py"

    if not debug_script.exists():
        print(f"ERROR: missing script: {debug_script}", file=sys.stderr)
        return 3
    if not eval_script.exists():
        print(f"ERROR: missing script: {eval_script}", file=sys.stderr)
        return 4

    gt_mean_dir = Path(args.gt_mean_dir).expanduser().resolve() if args.gt_mean_dir else (scripts_root / "gt_from_bag_outputs_aligned" / "gt_mean_references")
    if not gt_mean_dir.exists():
        print(f"WARNING: GT mean dir not found (will still run, but --gt-csv may fail): {gt_mean_dir}", file=sys.stderr)

    if not args.scenario:
        print("ERROR: provide at least one --scenario configuration.", file=sys.stderr)
        return 5

    scenario_cfgs = [parse_scenario_cfg(s) for s in args.scenario]

    print("[INFO] Scripts root:", scripts_root)
    print("[INFO] RUNS root:", runs_root)
    print("[INFO] Only combinations glob:", args.only_glob)
    print("[INFO] GT mean dir:", gt_mean_dir)
    print("[INFO] GPS topic:", args.gps_topic)
    print("[INFO] Human resample samples:", args.human_resample_samples)

    for scfg in scenario_cfgs:
        scenario_dir = runs_root / scfg.scenario_dir_name
        if not scenario_dir.exists():
            print(f"[WARN] Scenario dir not found, skipping: {scenario_dir}")
            continue

        print(f"\n[SCENARIO] {scfg.scenario_dir_name} (scenario_id={scfg.scenario_id})")

        combo_dirs = sorted([p for p in scenario_dir.glob(args.only_glob) if p.is_dir()])
        if not combo_dirs:
            print(f"[WARN] No combination dirs matched under: {scenario_dir}")
            continue

        for combo_dir in combo_dirs:
            speed_kmh = parse_speed_kmh_from_combo_dir(combo_dir.name)
            if speed_kmh is None:
                print(f"[SKIP] Could not parse speed from combo dir: {combo_dir.name}")
                continue

            planner_id, controller_id = parse_method_from_combo_dir(combo_dir.name)

            # Legacy gt mean csv path (required by eval script, even if ignored for metrics)
            gt_csv = gt_mean_dir / f"scenario{scfg.scenario_id}_vel{speed_kmh}_GTmean_rigidAligned_from_bag.csv"
            if not gt_csv.exists():
                print(f"[WARN] Legacy GT CSV missing: {gt_csv} (eval script may fail).")

            sim_glob = detect_sim_glob(combo_dir)

            print(f"\n[COMBO] {combo_dir}")
            print(f"  - speed_kmh: {speed_kmh}")
            print(f"  - planner_id: {planner_id}")
            print(f"  - controller_id: {controller_id}")
            print(f"  - sim_glob: {sim_glob}")

            # 1) Charts script
            cmd_charts = [
                sys.executable,
                str(debug_script),
                "--sim-parent-dir", str(combo_dir),
                "--sim-glob", sim_glob,
                "--gt-csv-glob", scfg.gt_gps_csv_glob,
                "--waypoint-file", str(scfg.waypoint_file),
                "--save-per-run-plots",
                # out-dir omitted on purpose -> defaults to Output_charts_diagnostic/ inside combo_dir
            ]
            rc = run_cmd(cmd_charts, args.dry_run)
            if rc != 0:
                msg = f"[ERROR] Charts failed for {combo_dir} (rc={rc})"
                if args.continue_on_error:
                    print(msg)
                else:
                    print(msg, file=sys.stderr)
                    return rc

            # 2) Metrics script
            cmd_metrics = [
                sys.executable,
                str(eval_script),
                "--runs-dir", str(combo_dir),
                "--scenario-id", str(scfg.scenario_id),
                "--speed-kmh", str(speed_kmh),
                "--planner-id", planner_id,
                "--controller-id", controller_id,
                "--gt-csv", str(gt_csv),
                "--gt-gps-csv-glob", scfg.gt_gps_csv_glob,
                "--waypoint-file", str(scfg.waypoint_file),
                "--gps-topic", args.gps_topic,
                "--human-resample-samples", str(args.human_resample_samples),
                # out-dir omitted on purpose -> defaults to Output_metrics_assessment/ inside combo_dir
            ]
            rc = run_cmd(cmd_metrics, args.dry_run)
            if rc != 0:
                msg = f"[ERROR] Metrics failed for {combo_dir} (rc={rc})"
                if args.continue_on_error:
                    print(msg)
                else:
                    print(msg, file=sys.stderr)
                    return rc

    print("\n[OK] Batch run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())