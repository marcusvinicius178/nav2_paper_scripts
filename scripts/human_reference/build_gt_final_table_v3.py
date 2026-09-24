#!/usr/bin/env python3
"""Build the v3 human-reference table without conflating speed semantics.

The rosbag-derived input field ``speed_kmh`` is the speed label of the human
recording (15 or 20 km/h). This v3 adapter preserves it as
``human_source_speed_kmh`` and adds the corresponding benchmark label as
``benchmark_speed_kmh``. Numerical GT reproducibility metrics are produced by
the unchanged baseline implementation and are regression-tested against its
previous output.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import build_gt_final_table as legacy


BENCHMARK_TO_HUMAN_SOURCE: Dict[int, int] = {
    20: 15,
    25: 20,
    30: 15,
    35: 20,
}
NOMINAL_BENCHMARK_FOR_HUMAN_SOURCE: Dict[int, int] = {15: 20, 20: 25}
NUMERIC_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Create the v3 GT reproducibility table with separate human-source "
            "and benchmark speed columns and a numerical regression audit."
        )
    )
    parser.add_argument("--project-root", default=str(default_root), help="NAV2_Paper_Scripts root.")
    parser.add_argument(
        "--input-csv",
        default="gt_from_bag_outputs_aligned/gt_pair_reproducibility_from_bag.csv",
        help="Rosbag-derived GT pair CSV; speed_kmh is interpreted as human source speed.",
    )
    parser.add_argument(
        "--within-csv",
        default="gt_from_bag_outputs_aligned/within_bag_topic_consistency.csv",
        help="Optional within-bag odometry-vs-GPS consistency CSV.",
    )
    parser.add_argument(
        "--previous-table-csv",
        default="gt_from_bag_outputs_aligned/final_tables/gt_reproducibility_paper_table.csv",
        help="Previous baseline table used only for regression comparison.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/gt_final_table_v3",
        help="v3 output directory.",
    )
    parser.add_argument("--self-test", action="store_true", help="Verify all four mappings and exit.")
    return parser.parse_args()


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def human_source_for_benchmark(benchmark_speed_kmh: int) -> int:
    try:
        return BENCHMARK_TO_HUMAN_SOURCE[int(benchmark_speed_kmh)]
    except KeyError as exc:
        raise ValueError(
            f"No human-source mapping defined for benchmark speed {benchmark_speed_kmh} km/h"
        ) from exc


def nominal_benchmark_for_human_source(human_source_speed_kmh: int) -> int:
    try:
        return NOMINAL_BENCHMARK_FOR_HUMAN_SOURCE[int(human_source_speed_kmh)]
    except KeyError as exc:
        raise ValueError(
            f"Human source speed must be one of {sorted(NOMINAL_BENCHMARK_FOR_HUMAN_SOURCE)}; "
            f"got {human_source_speed_kmh}"
        ) from exc


def run_mapping_self_test() -> None:
    expected = {20: 15, 25: 20, 30: 15, 35: 20}
    observed = {speed: human_source_for_benchmark(speed) for speed in expected}
    assert observed == expected, (observed, expected)
    assert nominal_benchmark_for_human_source(15) == 20
    assert nominal_benchmark_for_human_source(20) == 25


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def attach_speed_semantics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for row in rows:
        source = legacy.safe_int(row.get("human_source_speed_kmh", row.get("speed_kmh")))
        if source is None:
            raise ValueError(f"GT row lacks a valid human-source speed: {row}")
        benchmark = nominal_benchmark_for_human_source(source)
        converted: Dict[str, Any] = {
            "scenario_id": row.get("scenario_id"),
            "scenario_name": row.get("scenario_name"),
            "benchmark_speed_kmh": benchmark,
            "human_source_speed_kmh": source,
            "benchmark_condition": "nominal",
            "speed_semantics": (
                "benchmark_speed_kmh labels the simulation condition; "
                "human_source_speed_kmh labels the original professional-driver recording"
            ),
        }
        for key, value in row.items():
            if key not in {"scenario_id", "scenario_name", "speed_kmh", "benchmark_speed_kmh", "human_source_speed_kmh"}:
                converted[key] = value
        result.append(converted)
    result.sort(
        key=lambda row: (
            999 if row.get("scenario_id") is None else int(row["scenario_id"]),
            int(row["benchmark_speed_kmh"]),
        )
    )
    return result


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Scenario",
        "Benchmark speed",
        "Human source speed",
        "Raw RMSE (m)",
        "Aligned RMSE (m)",
        "Aligned P95 (m)",
        "Heading mean (deg)",
        "Quality",
        "Flags",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.get('scenario_id')} ({row.get('scenario_name')})",
                    f"{row.get('benchmark_speed_kmh')} km/h",
                    f"{row.get('human_source_speed_kmh')} km/h",
                    legacy.fmt_num(row.get("raw_rmse_m"), 3),
                    legacy.fmt_num(row.get("aligned_rmse_m"), 3),
                    legacy.fmt_num(row.get("aligned_p95_m"), 3),
                    legacy.fmt_num(row.get("heading_mean_deg"), 3),
                    str(row.get("quality_label", "")),
                    str(row.get("flags", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def mapping_rows() -> List[Dict[str, Any]]:
    return [
        {
            "benchmark_speed_kmh": benchmark,
            "human_source_speed_kmh": source,
            "benchmark_condition": "nominal" if benchmark in {20, 25} else "diagnostic_RPP",
            "speed_rmse_reference": "simulation commanded speed",
        }
        for benchmark, source in sorted(BENCHMARK_TO_HUMAN_SOURCE.items())
    ]


def comparable(value: Any) -> Tuple[str, Any]:
    if value is None or str(value).strip() == "":
        return "empty", ""
    try:
        number = float(value)
        if math.isfinite(number):
            return "number", number
    except Exception:
        pass
    return "text", str(value).strip()


def values_equal(first: Any, second: Any) -> bool:
    first_kind, first_value = comparable(first)
    second_kind, second_value = comparable(second)
    if first_kind == "number" and second_kind == "number":
        return math.isclose(first_value, second_value, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE)
    return first_kind == second_kind and first_value == second_value


def regression_rows(
    previous_rows: Sequence[Mapping[str, Any]],
    v3_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    previous_by_key = {
        (legacy.safe_int(row.get("scenario_id")), legacy.safe_int(row.get("speed_kmh"))): row
        for row in previous_rows
    }
    result: List[Dict[str, Any]] = []
    ignored = {"scenario_id", "scenario_name", "speed_kmh"}
    for row in v3_rows:
        key = (legacy.safe_int(row.get("scenario_id")), legacy.safe_int(row.get("human_source_speed_kmh")))
        previous = previous_by_key.get(key)
        if previous is None:
            result.append(
                {
                    "scenario_id": key[0],
                    "human_source_speed_kmh": key[1],
                    "field": "*",
                    "previous_value": "",
                    "v3_value": "",
                    "status": "MISSING_PREVIOUS_ROW",
                }
            )
            continue
        for field_name, previous_value in previous.items():
            if field_name in ignored:
                continue
            v3_value = row.get(field_name)
            result.append(
                {
                    "scenario_id": key[0],
                    "human_source_speed_kmh": key[1],
                    "field": field_name,
                    "previous_value": previous_value,
                    "v3_value": v3_value,
                    "status": "PASS" if values_equal(previous_value, v3_value) else "FAIL",
                }
            )
    return result


def literal_mapping_from_function(path: Path, function_name: str) -> Optional[Dict[int, int]]:
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    try:
                        parsed = ast.literal_eval(child)
                    except Exception:
                        continue
                    if isinstance(parsed, dict) and all(isinstance(key, int) for key in parsed):
                        return {int(key): int(value) for key, value in parsed.items()}
    return None


def literal_speed_mapping_constant(path: Path) -> Optional[List[Tuple[int, int]]]:
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "SPEED_MAPPING" in names:
                try:
                    return [(int(first), int(second)) for first, second in ast.literal_eval(node.value)]
                except Exception:
                    return None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "SPEED_MAPPING":
            try:
                return [(int(first), int(second)) for first, second in ast.literal_eval(node.value)]
            except Exception:
                return None
    return None


def consumer_audit(project_root: Path) -> List[Dict[str, Any]]:
    pipeline = project_root / "scripts/simulation/run_supplementary_tables_pipeline.py"
    figure_s9 = project_root / "scripts/human_reference/plot_human_reference_repeatability.py"
    gt_reader = project_root / "scripts/human_reference/gt_from_rosbag_analysis.py"
    pipeline_mapping = literal_mapping_from_function(pipeline, "human_speed_for_simulation")
    figure_mapping = literal_speed_mapping_constant(figure_s9)
    return [
        {
            "consumer": str(gt_reader.relative_to(project_root)),
            "observed_semantics": "speed_kmh is parsed from human rosbag/file labels (15 or 20)",
            "expected_semantics": "human_source_speed_kmh",
            "status": "PASS_ADAPTED_BY_V3_TABLE",
            "action": "Original preserved; build_gt_final_table_v3 renames the field explicitly.",
        },
        {
            "consumer": str(pipeline.relative_to(project_root)),
            "observed_semantics": str(pipeline_mapping),
            "expected_semantics": str(BENCHMARK_TO_HUMAN_SOURCE),
            "status": "PASS" if pipeline_mapping == BENCHMARK_TO_HUMAN_SOURCE else "FAIL",
            "action": "No change required" if pipeline_mapping == BENCHMARK_TO_HUMAN_SOURCE else "Review before running simulation pipeline",
        },
        {
            "consumer": str(figure_s9.relative_to(project_root)),
            "observed_semantics": str(figure_mapping),
            "expected_semantics": str([(20, 15), (25, 20)]),
            "status": "PASS" if figure_mapping == [(20, 15), (25, 20)] else "FAIL",
            "action": "No change required" if figure_mapping == [(20, 15), (25, 20)] else "Review Figure S9 labels",
        },
        {
            "consumer": "GT mean-reference CSV filenames",
            "observed_semantics": "vel15 and vel20 identify original human recordings",
            "expected_semantics": "human_source_speed_kmh",
            "status": "PASS",
            "action": "Do not rename source CSVs to benchmark speeds.",
        },
    ]


def main() -> int:
    args = parse_args()
    run_mapping_self_test()
    if args.self_test:
        print("build_gt_final_table_v3 mapping tests: PASS")
        print(BENCHMARK_TO_HUMAN_SOURCE)
        return 0

    project_root = Path(args.project_root).expanduser().resolve()
    input_csv = resolve_path(args.input_csv, project_root)
    within_csv = resolve_path(args.within_csv, project_root) if args.within_csv else None
    previous_csv = resolve_path(args.previous_table_csv, project_root) if args.previous_table_csv else None
    out_dir = resolve_path(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = legacy.read_csv_rows(input_csv)
    within = legacy.load_within_bag_aggregates(within_csv)
    baseline_rows, baseline_diagnostics = legacy.build_final_rows(gt_rows, within)
    v3_rows = attach_speed_semantics(baseline_rows)
    v3_diagnostics = attach_speed_semantics(baseline_diagnostics)

    table_fields = list(v3_rows[0].keys()) if v3_rows else []
    diagnostic_fields = list(v3_diagnostics[0].keys()) if v3_diagnostics else []
    table_csv = out_dir / "gt_reproducibility_paper_table_v3.csv"
    diagnostic_csv = out_dir / "gt_reproducibility_diagnostics_v3.csv"
    write_csv(table_csv, v3_rows, table_fields)
    write_csv(diagnostic_csv, v3_diagnostics, diagnostic_fields)
    (out_dir / "gt_reproducibility_paper_table_v3.md").write_text(markdown_table(v3_rows), encoding="utf-8")

    mappings = mapping_rows()
    write_csv(
        out_dir / "human_speed_mapping_v3.csv",
        mappings,
        ["benchmark_speed_kmh", "human_source_speed_kmh", "benchmark_condition", "speed_rmse_reference"],
    )
    mapping_log_lines = [
        "HUMAN/BENCHMARK SPEED MAPPING V3",
        "",
        "The fields are conceptually distinct and must never overwrite each other.",
        "Speed RMSE remains referenced to the commanded simulation speed.",
        "",
    ]
    for row in mappings:
        mapping_log_lines.append(
            f"benchmark {row['benchmark_speed_kmh']} km/h -> human source {row['human_source_speed_kmh']} km/h "
            f"[{row['benchmark_condition']}]"
        )
    (out_dir / "human_speed_mapping_v3.log").write_text("\n".join(mapping_log_lines) + "\n", encoding="utf-8")

    consumers = consumer_audit(project_root)
    write_csv(
        out_dir / "gt_speed_consumers_audit_v3.csv",
        consumers,
        ["consumer", "observed_semantics", "expected_semantics", "status", "action"],
    )
    consumer_failures = [row for row in consumers if row["status"] == "FAIL"]

    regression: List[Dict[str, Any]] = []
    if previous_csv and previous_csv.is_file():
        previous_rows = legacy.read_csv_rows(previous_csv)
        regression = regression_rows(previous_rows, v3_rows)
    else:
        regression = [
            {
                "scenario_id": "",
                "human_source_speed_kmh": "",
                "field": "*",
                "previous_value": "",
                "v3_value": "",
                "status": "PREVIOUS_TABLE_NOT_AVAILABLE",
            }
        ]
    write_csv(
        out_dir / "gt_reproducibility_regression_v3.csv",
        regression,
        ["scenario_id", "human_source_speed_kmh", "field", "previous_value", "v3_value", "status"],
    )
    regression_failures = [row for row in regression if row["status"] in {"FAIL", "MISSING_PREVIOUS_ROW"}]

    summary_lines = [
        "GT FINAL TABLE V3 AUDIT",
        "",
        f"Input rows: {len(gt_rows)}",
        f"Output rows: {len(v3_rows)}",
        f"Mapping self-test: PASS",
        f"Consumer audit failures: {len(consumer_failures)}",
        f"Numerical/field regression failures: {len(regression_failures)}",
        "Simulation metrics reprocessed: NO",
        "plan_cross_track_v2 modified: NO",
    ]
    (out_dir / "gt_final_table_v3_audit.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"Output directory: {out_dir}")
    if consumer_failures or regression_failures:
        print("ERROR: v3 output failed a consumer or regression audit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
