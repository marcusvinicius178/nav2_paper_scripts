#!/usr/bin/env python3
"""Build manuscript Supplementary Tables S2-S4 from per-run Nav2 metrics.

The script recursively discovers ``per_run_metrics.csv`` files produced by
``eval_nav2_one_combination.py``. It writes table-ready CSV files with explicit
median, Q1 and Q3 columns, compact LaTeX tables, a combined per-run CSV and an
audit report.

Statistical populations
-----------------------
* Geometric success rate: all attempted runs.
* Progress ratio: all attempted runs, so failures are not censored.
* Mission time, speed error and smoothness: mission-valid runs only.
* Active-plan lateral and heading errors: plan-tracking-valid runs only.
* Human-likeness metrics: mission-valid runs with a successfully computed human metric.

Nominal and diagnostic conditions
---------------------------------
Speeds 20 and 25 km/h are labelled ``nominal``. Non-nominal RPP speeds are
labelled ``diagnostic_RPP`` and remain separate in every output table.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PRIMARY_METRICS: Sequence[Tuple[str, str]] = (
    ("PR", "progress_ratio"),
    ("T_s", "mission_time_s"),
    ("RMSE_y_m", "rmse_y_primary_m"),
    ("P95_y_m", "p95_y_primary_m"),
)

SECONDARY_METRICS: Sequence[Tuple[str, str]] = (
    ("RMSE_psi_rad", "rmse_psi_primary_rad"),
    ("RMSE_v_mps", "rmse_v_mps"),
    ("RMS_jx_mps3", "rms_jx_mps3"),
    ("RMS_dotdelta_radps", "rms_dotdelta_radps"),
)

HUMAN_METRICS: Sequence[Tuple[str, str]] = (
    ("Human_RMSE_y_m", "human_gt_rmse_y_m"),
    ("Human_P95_y_m", "human_gt_p95_y_m"),
    ("Human_MAX_y_m", "human_gt_max_y_m"),
    ("Human_PR", "human_gt_progress"),
    ("Human_pointwise_RMSE_m", "human_gt_pointwise_rmse_m"),
)


@dataclass(frozen=True)
class Stats:
    n: int
    median: float
    q1: float
    q3: float


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build Supplementary Tables S2-S4 from per_run_metrics.csv files."
    )
    parser.add_argument(
        "--results-root",
        default=str(default_root / "results" / "supplementary_tables_reanalysis_plan_cross_track_v2"),
        help="Root containing one result directory per scenario/combination.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(default_root / "results" / "supplementary_tables_final_plan_cross_track_v2"),
        help="Destination for combined CSV, Tables S2-S4 and audit report.",
    )
    parser.add_argument(
        "--expected-total-runs",
        type=int,
        default=212,
        help="Expected total attempted simulation runs (default: 212).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write tables even when the expected inventory is incomplete.",
    )
    parser.add_argument(
        "--allow-unexpected-primary-reference",
        action="store_true",
        help=(
            "Allow rows that fail the active-plan cross-track v2 provenance audit. "
            "Intended only for explicitly labelled drafts."
        ),
    )
    return parser.parse_args()


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def to_float(value: object) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def scenario_label(scenario_id: int) -> str:
    return f"S{scenario_id}"


def normalize_planner(value: str) -> str:
    upper = value.strip().upper()
    if upper == "NAVFN":
        return "NavFn"
    if upper in {"SMAC", "SMACPLANNER", "SMAC_PLANNER"}:
        return "SMAC"
    return value.strip()


def normalize_controller(value: str) -> str:
    upper = value.strip().upper()
    if upper in {"PUREPURSUIT", "REGULATEDPUREPURSUIT"}:
        return "RPP"
    return upper


def classify_condition(speed_kmh: int, controller: str) -> str:
    if speed_kmh in {20, 25}:
        return "nominal"
    if normalize_controller(controller) == "RPP":
        return "diagnostic_RPP"
    return "diagnostic_other"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def discover_rows(results_root: Path) -> Tuple[List[Dict[str, str]], List[Path]]:
    source_files = sorted(results_root.rglob("per_run_metrics.csv"))
    source_files = [p for p in source_files if "test_one_bag" not in p.parts]
    rows: List[Dict[str, str]] = []
    seen_bags: Dict[str, Path] = {}

    for source in source_files:
        for row in read_csv_rows(source):
            bag_key = str(row.get("bag_dir", "")).strip()
            if not bag_key:
                bag_key = "|".join(
                    [
                        str(row.get("scenario_id", "")),
                        str(row.get("speed_kmh", "")),
                        str(row.get("planner_id", "")),
                        str(row.get("controller_id", "")),
                        str(row.get("run_id", "")),
                        str(row.get("bag_name", "")),
                    ]
                )
            if bag_key in seen_bags:
                raise RuntimeError(
                    f"Duplicate run detected in {source} and {seen_bags[bag_key]}: {bag_key}"
                )
            seen_bags[bag_key] = source
            enriched = dict(row)
            enriched["source_per_run_csv"] = str(source)
            rows.append(enriched)

    return rows, source_files


def group_key(row: Mapping[str, str]) -> Tuple[int, int, str, str, str]:
    scenario_id = to_int(row.get("scenario_id", 0))
    speed = to_int(row.get("speed_kmh", 0))
    planner = normalize_planner(str(row.get("planner_id", "")))
    controller = normalize_controller(str(row.get("controller_id", "")))
    condition = classify_condition(speed, controller)
    return scenario_id, speed, planner, controller, condition


def finite_values(rows: Iterable[Mapping[str, str]], field: str) -> np.ndarray:
    values = [to_float(row.get(field, "")) for row in rows]
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def stats(rows: Iterable[Mapping[str, str]], field: str) -> Stats:
    values = finite_values(rows, field)
    if values.size == 0:
        return Stats(0, float("nan"), float("nan"), float("nan"))
    return Stats(
        n=int(values.size),
        median=float(np.median(values)),
        q1=float(np.percentile(values, 25.0)),
        q3=float(np.percentile(values, 75.0)),
    )


def expected_inventory() -> Dict[Tuple[int, int, str, str, str], int]:
    expected: Dict[Tuple[int, int, str, str, str], int] = {}
    planners = ("NavFn", "SMAC")

    for scenario_id in (1, 3):
        for speed in (20, 25):
            for planner in planners:
                for controller in ("MPPI", "RPP"):
                    expected[(scenario_id, speed, planner, controller, "nominal")] = 10

    for speed in (20, 25):
        for planner in planners:
            expected[(2, speed, planner, "MPPI", "nominal")] = 10

    for speed in (30, 35):
        for planner in planners:
            expected[(2, speed, planner, "RPP", "diagnostic_RPP")] = 3

    return expected


def key_sort_value(key: Tuple[int, int, str, str, str]) -> Tuple[int, int, int, str, str]:
    scenario_id, speed, planner, controller, condition = key
    condition_rank = 0 if condition == "nominal" else 1
    return scenario_id, condition_rank, speed, planner, controller


def common_summary(
    key: Tuple[int, int, str, str, str], rows: Sequence[Mapping[str, str]]
) -> Dict[str, object]:
    scenario_id, speed, planner, controller, condition = key
    n_total = len(rows)
    n_success = sum(to_int(row.get("success_geo", 0)) == 1 for row in rows)
    mission_rows = [
        row for row in rows if to_int(row.get("valid_for_mission", 0)) == 1
    ]
    plan_rows = [
        row for row in rows if to_int(row.get("valid_for_tracking", 0)) == 1
    ]
    n_human = int(finite_values(mission_rows, "human_gt_rmse_y_m").size)
    return {
        "scenario": scenario_label(scenario_id),
        "scenario_id": scenario_id,
        "condition": condition,
        "speed_kmh": speed,
        "planner": planner,
        "controller": controller,
        "N_total": n_total,
        "N_success": n_success,
        "N_mission": len(mission_rows),
        "N_plan": len(plan_rows),
        "N_valid": len(plan_rows),
        "N_human": n_human,
        "SR_geo_rate": (n_success / n_total) if n_total else float("nan"),
    }


def append_stats_columns(target: Dict[str, object], prefix: str, value: Stats) -> None:
    target[f"{prefix}_N"] = value.n
    target[f"{prefix}_median"] = value.median
    target[f"{prefix}_Q1"] = value.q1
    target[f"{prefix}_Q3"] = value.q3


def build_table_rows(
    groups: Mapping[Tuple[int, int, str, str, str], Sequence[Mapping[str, str]]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    table_s2: List[Dict[str, object]] = []
    table_s3: List[Dict[str, object]] = []
    table_s4: List[Dict[str, object]] = []

    for key in sorted(groups, key=key_sort_value):
        rows = list(groups[key])
        mission_rows = [
            r for r in rows if to_int(r.get("valid_for_mission", 0)) == 1
        ]
        plan_rows = [
            r for r in rows if to_int(r.get("valid_for_tracking", 0)) == 1
        ]
        human_rows = [
            r
            for r in mission_rows
            if math.isfinite(to_float(r.get("human_gt_rmse_y_m", "")))
        ]

        s2 = common_summary(key, rows)
        for prefix, field in PRIMARY_METRICS:
            if field == "progress_ratio":
                population = rows
            elif field == "mission_time_s":
                population = mission_rows
            else:
                population = plan_rows
            append_stats_columns(s2, prefix, stats(population, field))
        table_s2.append(s2)

        s3 = common_summary(key, rows)
        for prefix, field in SECONDARY_METRICS:
            population = plan_rows if field == "rmse_psi_primary_rad" else mission_rows
            append_stats_columns(s3, prefix, stats(population, field))
        table_s3.append(s3)

        s4 = common_summary(key, rows)
        for prefix, field in HUMAN_METRICS:
            append_stats_columns(s4, prefix, stats(human_rows, field))
        table_s4.append(s4)

    return table_s2, table_s3, table_s4


def csv_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.9g}"
    return value


def write_dict_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_number(value: object, decimals: int = 3) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return "--"
    return f"{number:.{decimals}f}"


def stat_cell(row: Mapping[str, object], prefix: str, decimals: int = 3) -> str:
    return (
        f"{format_number(row.get(prefix + '_median'), decimals)} "
        f"[{format_number(row.get(prefix + '_Q1'), decimals)}, "
        f"{format_number(row.get(prefix + '_Q3'), decimals)}]"
    )


def condition_tex(value: object) -> str:
    mapping = {
        "nominal": "Nominal",
        "diagnostic_RPP": "RPP diagnostic",
        "diagnostic_other": "Diagnostic",
    }
    return mapping.get(str(value), latex_escape(value))


def latex_table(
    table_number: str,
    caption: str,
    label: str,
    rows: Sequence[Mapping[str, object]],
    metric_headers: Sequence[str],
    metric_prefixes: Sequence[str],
    count_columns: Sequence[Tuple[str, str]],
) -> str:
    n_columns = 5 + len(count_columns) + len(metric_headers)
    alignment = "lllll" + "r" * (n_columns - 5)
    common_headers = [
        "Scenario",
        "Condition",
        "Speed",
        "Planner",
        "Controller",
    ]
    common_headers.extend(header for header, _ in count_columns)
    headers = common_headers + list(metric_headers)

    lines = [
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    previous_scenario = None
    for row in rows:
        if previous_scenario is not None and row["scenario"] != previous_scenario:
            lines.append(r"\midrule")
        previous_scenario = row["scenario"]
        common = [
            latex_escape(row["scenario"]),
            condition_tex(row["condition"]),
            f"{to_int(row['speed_kmh'])} km/h",
            latex_escape(row["planner"]),
            latex_escape(row["controller"]),
        ]
        common.extend(str(to_int(row[key])) for _, key in count_columns)
        cells = common + [stat_cell(row, prefix) for prefix in metric_prefixes]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\vspace{{1mm}}\par\raggedright\footnotesize Note: {table_number}. Values are median [Q1, Q3].",
            r"\end{table}",
            r"\end{landscape}",
            "",
        ]
    )
    return "\n".join(lines)


def write_combined_runs(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise RuntimeError("No per-run rows available")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inventory_report(
    groups: Mapping[Tuple[int, int, str, str, str], Sequence[Mapping[str, str]]],
    expected_total_runs: int,
) -> Tuple[List[str], bool]:
    expected = expected_inventory()
    lines = ["SUPPLEMENTARY TABLE INVENTORY AUDIT", ""]
    ok = True

    actual_keys = set(groups)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys, key=key_sort_value)
    extra = sorted(actual_keys - expected_keys, key=key_sort_value)

    if missing:
        ok = False
        lines.append("Missing combinations:")
        lines.extend(f"  - {key}" for key in missing)
    if extra:
        ok = False
        lines.append("Unexpected combinations:")
        lines.extend(f"  - {key}" for key in extra)

    lines.append("")
    lines.append("Run counts by combination:")
    for key in sorted(actual_keys | expected_keys, key=key_sort_value):
        actual_count = len(groups.get(key, []))
        expected_count = expected.get(key)
        marker = "OK" if expected_count == actual_count else "MISMATCH"
        if marker != "OK":
            ok = False
        lines.append(
            f"  {marker:8s} {key}: actual={actual_count}, expected={expected_count}"
        )

    actual_total = sum(len(rows) for rows in groups.values())
    if actual_total != expected_total_runs:
        ok = False
    lines.extend(
        [
            "",
            f"Total runs: actual={actual_total}, expected={expected_total_runs}",
            f"Total combinations: actual={len(groups)}, expected={len(expected)}",
            f"Inventory status: {'PASS' if ok else 'FAIL'}",
        ]
    )
    return lines, ok


def methodology_report(
    rows: Sequence[Mapping[str, str]],
) -> Tuple[List[str], List[str]]:
    required_columns = {
        "primary_reference",
        "plan_topic_used",
        "valid_for_mission",
        "valid_for_tracking",
        "plan_tracking_samples",
        "plan_tracking_coverage",
        "plan_endpoint_extension_samples",
        "plan_endpoint_extension_fraction",
        "notes",
    }
    errors: List[str] = []
    primary_counts: Dict[str, int] = defaultdict(int)
    topic_counts: Dict[str, int] = defaultdict(int)
    total_tracking_samples = 0
    total_endpoint_extensions = 0

    for row_index, row in enumerate(rows, start=1):
        missing = sorted(required_columns - set(row))
        if missing:
            errors.append(f"row {row_index}: missing columns: {', '.join(missing)}")
            continue

        primary = str(row.get("primary_reference", "")).strip()
        topic = str(row.get("plan_topic_used", "")).strip()
        notes = str(row.get("notes", ""))
        valid_tracking = to_int(row.get("valid_for_tracking", 0)) == 1
        primary_counts[primary or "<empty>"] += 1
        topic_counts[topic or "<none>"] += 1

        if valid_tracking and not primary.startswith(
            "nav2_active_plan_cross_track_v2:"
        ):
            errors.append(
                f"row {row_index}: plan-valid row has incompatible primary reference: "
                f"{primary or '<empty>'}"
            )
        if valid_tracking and not topic:
            errors.append(f"row {row_index}: plan-valid row has no plan topic")
        if (
            "PLAN_CROSSTRACK_DEF=POLYLINE_SEGMENTS_WITH_ENDPOINT_TANGENT_EXTENSION"
            not in notes
        ):
            errors.append(f"row {row_index}: missing cross-track v2 provenance token")

        total_tracking_samples += to_int(row.get("plan_tracking_samples", 0))
        total_endpoint_extensions += to_int(
            row.get("plan_endpoint_extension_samples", 0)
        )

    extension_fraction = (
        total_endpoint_extensions / total_tracking_samples
        if total_tracking_samples > 0
        else float("nan")
    )
    lines = [
        "METHODOLOGY AUDIT",
        "",
        "Primary reference counts:",
    ]
    lines.extend(
        f"  {name}: {count}" for name, count in sorted(primary_counts.items())
    )
    lines.extend(["", "Selected plan-topic counts:"])
    lines.extend(
        f"  {name}: {count}" for name, count in sorted(topic_counts.items())
    )
    lines.extend(
        [
            "",
            f"Plan-tracking samples: {total_tracking_samples}",
            f"Endpoint-tangent extension samples: {total_endpoint_extensions}",
            (
                "Endpoint-tangent extension fraction: "
                + (f"{extension_fraction:.6f}" if math.isfinite(extension_fraction) else "NA")
            ),
            f"Methodology status: {'PASS' if not errors else 'FAIL'}",
        ]
    )
    return lines, errors


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not results_root.exists():
        print(f"ERROR: results root does not exist: {results_root}", file=sys.stderr)
        return 2

    try:
        all_rows, source_files = discover_rows(results_root)
    except Exception as exc:
        print(f"ERROR: could not collect per-run CSVs: {exc}", file=sys.stderr)
        return 3

    if not all_rows:
        print(f"ERROR: no per-run rows found under: {results_root}", file=sys.stderr)
        return 4

    method_report_lines, method_errors = methodology_report(all_rows)
    if method_errors and not args.allow_unexpected_primary_reference:
        print(
            "ERROR: active-plan cross-track v2 methodology audit failed:",
            file=sys.stderr,
        )
        for value in method_errors[:40]:
            print(f"  - {value}", file=sys.stderr)
        if len(method_errors) > 40:
            print(f"  - ... and {len(method_errors) - 40} additional errors", file=sys.stderr)
        return 5

    groups: Dict[Tuple[int, int, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        groups[group_key(row)].append(row)

    report_lines, inventory_ok = inventory_report(groups, args.expected_total_runs)
    report_lines.extend(
        [
            "",
            f"Source per_run_metrics.csv files: {len(source_files)}",
            f"Methodology errors: {len(method_errors)}",
            "",
            *method_report_lines,
        ]
    )

    if not inventory_ok and not args.allow_incomplete:
        print("\n".join(report_lines), file=sys.stderr)
        print(
            "ERROR: inventory audit failed. Use --allow-incomplete only for an explicitly labelled draft.",
            file=sys.stderr,
        )
        return 6

    table_s2, table_s3, table_s4 = build_table_rows(groups)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_combined_runs(out_dir / "all_simulation_runs_metrics.csv", all_rows)
    write_dict_csv(out_dir / "Table_S2_primary_metrics.csv", table_s2)
    write_dict_csv(out_dir / "Table_S3_secondary_metrics.csv", table_s3)
    write_dict_csv(out_dir / "Table_S4_human_likeness.csv", table_s4)

    s2_tex = latex_table(
        "Table S2",
        (
            "Primary simulation metrics for all configurations. Progress ratio uses all attempts; "
            "time uses mission-valid runs; lateral errors use plan-valid runs and the "
            "time-synchronized active Navigation2 plan."
        ),
        "tab:supp_primary_metrics",
        table_s2,
        (r"SR$_{geo}$", "PR", r"$T$ (s)", r"RMSE$_y$ (m)", r"P95$_y$ (m)"),
        ("SR_GEO_SYNTHETIC", "PR", "T_s", "RMSE_y_m", "P95_y_m"),
        (
            (r"$N_{total}$", "N_total"),
            (r"$N_{success}$", "N_success"),
            (r"$N_{mission}$", "N_mission"),
            (r"$N_{plan}$", "N_plan"),
        ),
    )
    # SRgeo is a count/rate rather than a median/IQR statistic; replace its
    # temporary synthetic cells after constructing the common table layout.
    s2_lines: List[str] = []
    data_index = 0
    for line in s2_tex.splitlines():
        if data_index < len(table_s2) and line.startswith(str(table_s2[data_index]["scenario"]) + " &"):
            row = table_s2[data_index]
            sr_text = (
                f"{to_int(row['N_success'])}/{to_int(row['N_total'])} "
                f"({100.0 * to_float(row['SR_geo_rate']):.1f}\\%)"
            )
            line = line.replace("-- [--, --]", sr_text, 1)
            data_index += 1
        s2_lines.append(line)
    s2_tex = "\n".join(s2_lines) + "\n"

    s3_tex = latex_table(
        "Table S3",
        (
            "Secondary simulation metrics. Heading error uses plan-valid runs; speed and "
            "smoothness metrics use mission-valid runs."
        ),
        "tab:supp_secondary_metrics",
        table_s3,
        (
            r"RMSE$_\psi$ (rad)",
            r"RMSE$_v$ (m/s)",
            r"RMS$_{j_x}$ (m/s$^3$)",
            r"RMS$_{\dot{\delta}}$ (rad/s)",
        ),
        ("RMSE_psi_rad", "RMSE_v_mps", "RMS_jx_mps3", "RMS_dotdelta_radps"),
        (
            (r"$N_{total}$", "N_total"),
            (r"$N_{success}$", "N_success"),
            (r"$N_{mission}$", "N_mission"),
            (r"$N_{plan}$", "N_plan"),
        ),
    )

    s4_tex = latex_table(
        "Table S4",
        (
            "Human-likeness metrics for mission-valid simulation runs with a successfully computed "
            "human-reference comparison."
        ),
        "tab:supp_human_likeness",
        table_s4,
        (
            r"RMSE$_{y,H}$ (m)",
            r"P95$_{y,H}$ (m)",
            r"Max$_{y,H}$ (m)",
            r"PR$_H$",
            r"Pointwise RMSE$_H$ (m)",
        ),
        (
            "Human_RMSE_y_m",
            "Human_P95_y_m",
            "Human_MAX_y_m",
            "Human_PR",
            "Human_pointwise_RMSE_m",
        ),
        (
            (r"$N_{total}$", "N_total"),
            (r"$N_{success}$", "N_success"),
            (r"$N_{mission}$", "N_mission"),
            (r"$N_{human}$", "N_human"),
        ),
    )

    (out_dir / "Table_S2_primary_metrics.tex").write_text(s2_tex, encoding="utf-8")
    (out_dir / "Table_S3_secondary_metrics.tex").write_text(s3_tex, encoding="utf-8")
    (out_dir / "Table_S4_human_likeness.tex").write_text(s4_tex, encoding="utf-8")
    (out_dir / "inventory_audit.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    readme = """Supplementary Tables S2-S4 outputs

CSV files contain explicit numeric median, Q1 and Q3 columns.
LaTeX cells use the compact format: median [Q1, Q3].

Populations:
- SR_geo and PR: all attempted runs.
- T, RMSE_v, RMS_jx and RMS_dotdelta: mission-valid runs only.
- Plan cross-track RMSE_y, P95_y and heading RMSE_psi: plan-tracking-valid runs only.
- Human-likeness metrics: mission-valid runs with successful human-reference metrics.

Validity counts:
- N_mission: max mission progress ratio >= 0.70.
- N_plan: mission-valid and active-plan coverage >= configured threshold.
- N_human: mission-valid and human-reference comparison available.

Plan cross-track definition:
- latest recorded plan at or before each odometry sample;
- pose transformed to the plan frame with recorded TF when required;
- orthogonal projection to polyline segments;
- first/last segment tangent extension prevents plan-pruning along-track gaps
  from being counted as lateral error.

The LaTeX fragments require: booktabs, graphicx and pdflscape.
Diagnostic RPP conditions are labelled separately from nominal conditions.
"""
    (out_dir / "README_tables.txt").write_text(readme, encoding="utf-8")

    print(f"[OK] Combined per-run CSV: {out_dir / 'all_simulation_runs_metrics.csv'}")
    print(f"[OK] Table S2 CSV/LaTeX: {out_dir / 'Table_S2_primary_metrics.csv'}")
    print(f"[OK] Table S3 CSV/LaTeX: {out_dir / 'Table_S3_secondary_metrics.csv'}")
    print(f"[OK] Table S4 CSV/LaTeX: {out_dir / 'Table_S4_human_likeness.csv'}")
    print(f"[OK] Inventory audit: {out_dir / 'inventory_audit.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
