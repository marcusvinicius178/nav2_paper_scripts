#!/usr/bin/env python3
"""Recreate manuscript Figures 5-7 from plan-cross-track-v2 tables.

Inputs are the audited Tables S2-S4 CSV files. The script uses:

    lower error = median - Q1
    upper error = Q3 - median

It therefore never treats the full IQR width as a symmetric error around the
median. The progress-ratio audit also refuses values outside [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MethodStyle:
    planner: str
    controller: str
    label: str
    color: str
    hatch: str


@dataclass(frozen=True)
class MetricSpec:
    title: str
    ylabel: str
    value_column: str
    q1_column: Optional[str]
    q3_column: Optional[str]
    n_column: Optional[str]
    fixed_ylim: Optional[Tuple[float, float]] = None


METHODS: Sequence[MethodStyle] = (
    MethodStyle("SMAC", "MPPI", "SMAC-MPPI", "#1f77b4", ""),
    MethodStyle("NavFn", "MPPI", "NavFn-MPPI", "#ff7f0e", "//"),
    MethodStyle("SMAC", "RPP", "SMAC-RPP", "#2ca02c", "\\\\"),
    MethodStyle("NavFn", "RPP", "NavFn-RPP", "#d62728", "xx"),
)

SCENARIOS: Sequence[Tuple[str, str]] = (
    ("S1", "S1\nStraight"),
    ("S2", "S2\nStraight + 20 m arc"),
    ("S3", "S3\nStraight + 40 m arc"),
)


FIGURE_5_METRICS: Sequence[MetricSpec] = (
    MetricSpec(
        "(a) Geometric success rate",
        "Success rate [-]",
        "SR_geo_rate",
        None,
        None,
        "N_total",
        (0.0, 1.05),
    ),
    MetricSpec(
        "(b) Progress ratio",
        "Progress ratio [-]",
        "PR_median",
        "PR_Q1",
        "PR_Q3",
        "PR_N",
        (0.0, 1.05),
    ),
    MetricSpec(
        "(c) Active-plan cross-track RMSE",
        "RMSE_y,plan [m]",
        "RMSE_y_m_median",
        "RMSE_y_m_Q1",
        "RMSE_y_m_Q3",
        "RMSE_y_m_N",
    ),
    MetricSpec(
        "(d) Active-plan cross-track P95",
        "P95_y,plan [m]",
        "P95_y_m_median",
        "P95_y_m_Q1",
        "P95_y_m_Q3",
        "P95_y_m_N",
    ),
)

FIGURE_6_METRICS: Sequence[MetricSpec] = (
    MetricSpec(
        "(a) Active-plan heading RMSE",
        "RMSE_psi,plan [rad]",
        "RMSE_psi_rad_median",
        "RMSE_psi_rad_Q1",
        "RMSE_psi_rad_Q3",
        "RMSE_psi_rad_N",
    ),
    MetricSpec(
        "(b) Speed RMSE",
        "RMSE_v [m/s]",
        "RMSE_v_mps_median",
        "RMSE_v_mps_Q1",
        "RMSE_v_mps_Q3",
        "RMSE_v_mps_N",
    ),
    MetricSpec(
        "(c) Longitudinal jerk RMS",
        "RMS_jx [m/s^3]",
        "RMS_jx_mps3_median",
        "RMS_jx_mps3_Q1",
        "RMS_jx_mps3_Q3",
        "RMS_jx_mps3_N",
    ),
    MetricSpec(
        "(d) Steering-rate RMS",
        "RMS_dotdelta [rad/s]",
        "RMS_dotdelta_radps_median",
        "RMS_dotdelta_radps_Q1",
        "RMS_dotdelta_radps_Q3",
        "RMS_dotdelta_radps_N",
    ),
)

FIGURE_7_METRICS: Sequence[MetricSpec] = (
    MetricSpec(
        "(a) Human-reference lateral RMSE",
        "RMSE_y,human [m]",
        "Human_RMSE_y_m_median",
        "Human_RMSE_y_m_Q1",
        "Human_RMSE_y_m_Q3",
        "Human_RMSE_y_m_N",
    ),
    MetricSpec(
        "(b) Human-reference lateral P95",
        "P95_y,human [m]",
        "Human_P95_y_m_median",
        "Human_P95_y_m_Q1",
        "Human_P95_y_m_Q3",
        "Human_P95_y_m_N",
    ),
    MetricSpec(
        "(c) Human-reference maximum deviation",
        "MAX_y,human [m]",
        "Human_MAX_y_m_median",
        "Human_MAX_y_m_Q1",
        "Human_MAX_y_m_Q3",
        "Human_MAX_y_m_N",
    ),
    MetricSpec(
        "(d) Human-reference progress ratio",
        "PR_human [-]",
        "Human_PR_median",
        "Human_PR_Q1",
        "Human_PR_Q3",
        "Human_PR_N",
        (0.0, 1.05),
    ),
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Recreate Figures 5-7 from audited S2-S4 tables."
    )
    parser.add_argument(
        "--project-root",
        default=str(default_root),
        help="NAV2_Paper_Scripts root.",
    )
    parser.add_argument(
        "--tables-dir",
        default="",
        help=(
            "Directory containing Table_S2_primary_metrics.csv, "
            "Table_S3_secondary_metrics.csv and Table_S4_human_likeness.csv "
            "(default: <project-root>/results/supplementary_tables_final_plan_cross_track_v2)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help=(
            "Output directory "
            "(default: <project-root>/results/corrected_figures_5_7_plan_cross_track_v2)."
        ),
    )
    parser.add_argument(
        "--speed-kmh",
        type=int,
        default=25,
        help="Nominal speed shown in the three figures (default: 25).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Table CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def to_float(value: object) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def to_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def select_row(
    rows: Sequence[Mapping[str, str]],
    scenario: str,
    speed_kmh: int,
    method: MethodStyle,
) -> Optional[Mapping[str, str]]:
    matches = [
        row
        for row in rows
        if str(row.get("scenario", "")) == scenario
        and to_int(row.get("speed_kmh", 0)) == speed_kmh
        and str(row.get("planner", "")) == method.planner
        and str(row.get("controller", "")) == method.controller
        and str(row.get("condition", "")) == "nominal"
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Duplicate nominal row for {scenario}, {speed_kmh}, {method.label}"
        )
    return matches[0] if matches else None


def metric_available(row: Optional[Mapping[str, str]], metric: MetricSpec) -> bool:
    if row is None:
        return False
    value = to_float(row.get(metric.value_column, ""))
    if not math.isfinite(value):
        return False
    if metric.n_column is not None and to_int(row.get(metric.n_column, 0)) <= 0:
        return False
    return True


def compute_ylim(
    rows: Sequence[Mapping[str, str]],
    metrics: Sequence[MetricSpec],
    speed_kmh: int,
) -> Dict[str, Tuple[float, float]]:
    limits: Dict[str, Tuple[float, float]] = {}
    for metric in metrics:
        if metric.fixed_ylim is not None:
            limits[metric.value_column] = metric.fixed_ylim
            continue
        values: List[float] = []
        for scenario, _ in SCENARIOS:
            for method in METHODS:
                row = select_row(rows, scenario, speed_kmh, method)
                if not metric_available(row, metric):
                    continue
                assert row is not None
                upper = (
                    to_float(row.get(metric.q3_column, ""))
                    if metric.q3_column is not None
                    else to_float(row.get(metric.value_column, ""))
                )
                if math.isfinite(upper):
                    values.append(upper)
        maximum = max(values) if values else 1.0
        limits[metric.value_column] = (0.0, max(1e-6, maximum * 1.15))
    return limits


def plot_one_figure(
    rows: Sequence[Mapping[str, str]],
    metrics: Sequence[MetricSpec],
    speed_kmh: int,
    title: str,
    output_stem: Path,
) -> List[str]:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    group_x = np.arange(len(SCENARIOS), dtype=float)
    width = 0.19
    offsets = np.asarray([-1.5, -0.5, 0.5, 1.5]) * width
    y_limits = compute_ylim(rows, metrics, speed_kmh)
    audit_lines: List[str] = []

    for axis, metric in zip(axes.flat, metrics):
        axis.set_title(metric.title)
        axis.set_ylabel(metric.ylabel)
        axis.set_xticks(group_x)
        axis.set_xticklabels([label for _, label in SCENARIOS])
        axis.grid(True, axis="y", linestyle=":", alpha=0.35)
        axis.set_ylim(*y_limits[metric.value_column])

        for method_index, method in enumerate(METHODS):
            values = np.full(len(SCENARIOS), np.nan, dtype=float)
            lower = np.zeros(len(SCENARIOS), dtype=float)
            upper = np.zeros(len(SCENARIOS), dtype=float)
            available = np.zeros(len(SCENARIOS), dtype=bool)

            for scenario_index, (scenario, _) in enumerate(SCENARIOS):
                row = select_row(rows, scenario, speed_kmh, method)
                if not metric_available(row, metric):
                    audit_lines.append(
                        f"{metric.value_column}: {scenario} {method.label}: n/a"
                    )
                    continue
                assert row is not None
                value = to_float(row[metric.value_column])
                values[scenario_index] = value
                available[scenario_index] = True

                if metric.q1_column is not None and metric.q3_column is not None:
                    q1 = to_float(row.get(metric.q1_column, ""))
                    q3 = to_float(row.get(metric.q3_column, ""))
                    if not (math.isfinite(q1) and math.isfinite(q3)):
                        raise RuntimeError(
                            f"Missing Q1/Q3 for {scenario} {method.label} {metric.value_column}"
                        )
                    if q1 > value + 1e-12 or q3 < value - 1e-12:
                        raise RuntimeError(
                            f"Invalid quartile order for {scenario} {method.label} "
                            f"{metric.value_column}: Q1={q1}, median={value}, Q3={q3}"
                        )
                    lower[scenario_index] = max(0.0, value - q1)
                    upper[scenario_index] = max(0.0, q3 - value)

                audit_lines.append(
                    f"{metric.value_column}: {scenario} {method.label}: "
                    f"median={value:.9g}, lower={lower[scenario_index]:.9g}, "
                    f"upper={upper[scenario_index]:.9g}"
                )

            x_positions = group_x + offsets[method_index]
            for scenario_index, x_position in enumerate(x_positions):
                if available[scenario_index]:
                    axis.bar(
                        x_position,
                        values[scenario_index],
                        width,
                        color=method.color,
                        edgecolor="black",
                        linewidth=0.7,
                        hatch=method.hatch,
                        label=method.label if scenario_index == 0 else None,
                        zorder=3,
                    )
                    if metric.q1_column is not None:
                        axis.errorbar(
                            x_position,
                            values[scenario_index],
                            yerr=np.asarray(
                                [
                                    [lower[scenario_index]],
                                    [upper[scenario_index]],
                                ]
                            ),
                            fmt="none",
                            ecolor="black",
                            elinewidth=1.0,
                            capsize=3,
                            capthick=1.0,
                            zorder=4,
                        )
                else:
                    axis.text(
                        x_position,
                        axis.get_ylim()[0] + 0.025 * (axis.get_ylim()[1] - axis.get_ylim()[0]),
                        "n/a",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="0.35",
                    )

    handles = []
    labels = []
    for method in METHODS:
        patch = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=method.color,
            edgecolor="black",
            hatch=method.hatch,
        )
        handles.append(patch)
        labels.append(method.label)
    figure.suptitle(title, fontsize=14, y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return audit_lines


def write_selected_values(
    path: Path,
    table_rows: Sequence[Tuple[str, Sequence[Mapping[str, str]]]],
    speed_kmh: int,
) -> None:
    selected: List[Dict[str, str]] = []
    fieldnames: List[str] = ["source_table"]
    for table_name, rows in table_rows:
        for row in rows:
            if (
                to_int(row.get("speed_kmh", 0)) == speed_kmh
                and str(row.get("condition", "")) == "nominal"
            ):
                enriched = {"source_table": table_name, **dict(row)}
                selected.append(enriched)
                for field in enriched:
                    if field not in fieldnames:
                        fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    tables_dir = (
        Path(args.tables_dir).expanduser().resolve()
        if args.tables_dir
        else project_root / "results" / "supplementary_tables_final_plan_cross_track_v2"
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else project_root / "results" / "corrected_figures_5_7_plan_cross_track_v2"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        table_s2 = read_csv(tables_dir / "Table_S2_primary_metrics.csv")
        table_s3 = read_csv(tables_dir / "Table_S3_secondary_metrics.csv")
        table_s4 = read_csv(tables_dir / "Table_S4_human_likeness.csv")
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    progress_columns = (
        (table_s2, "PR_median", "PR_Q1", "PR_Q3"),
        (table_s4, "Human_PR_median", "Human_PR_Q1", "Human_PR_Q3"),
    )
    violations: List[str] = []
    for rows, median_column, q1_column, q3_column in progress_columns:
        for row in rows:
            if to_int(row.get("speed_kmh", 0)) != args.speed_kmh:
                continue
            for column in (median_column, q1_column, q3_column):
                value = to_float(row.get(column, ""))
                if math.isfinite(value) and not (-1e-12 <= value <= 1.0 + 1e-12):
                    violations.append(
                        f"{row.get('scenario')} {row.get('planner')}-{row.get('controller')} "
                        f"{column}={value}"
                    )
    if violations:
        print("ERROR: progress-ratio values outside [0, 1]:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 3

    figure_specs = (
        (
            table_s2,
            FIGURE_5_METRICS,
            f"Mission completion and active-plan tracking at {args.speed_kmh} km/h",
            out_dir / f"Figure5_primary_{args.speed_kmh}kmh_corrected",
        ),
        (
            table_s3,
            FIGURE_6_METRICS,
            f"Plan heading, speed and smoothness at {args.speed_kmh} km/h",
            out_dir / f"Figure6_secondary_{args.speed_kmh}kmh_corrected",
        ),
        (
            table_s4,
            FIGURE_7_METRICS,
            f"Complementary human-likeness benchmark at {args.speed_kmh} km/h",
            out_dir / f"Figure7_human_likeness_{args.speed_kmh}kmh_corrected",
        ),
    )

    audit_lines: List[str] = [
        "CORRECTED FIGURES 5-7 AUDIT",
        "",
        f"Tables directory: {tables_dir}",
        f"Speed: {args.speed_kmh} km/h",
        "Error-bar definition: lower=median-Q1; upper=Q3-median",
        "Progress-ratio range audit: PASS",
        "",
    ]

    for rows, metrics, title, output_stem in figure_specs:
        try:
            lines = plot_one_figure(
                rows=rows,
                metrics=metrics,
                speed_kmh=args.speed_kmh,
                title=title,
                output_stem=output_stem,
            )
        except Exception as error:
            print(f"ERROR: could not generate {output_stem.name}: {error}", file=sys.stderr)
            return 4
        audit_lines.extend([f"[{output_stem.name}]", *lines, ""])
        print(f"[OK] {output_stem.with_suffix('.png')}")
        print(f"[OK] {output_stem.with_suffix('.pdf')}")

    selected_values = out_dir / f"figure_values_{args.speed_kmh}kmh.csv"
    write_selected_values(
        selected_values,
        (
            ("Table_S2", table_s2),
            ("Table_S3", table_s3),
            ("Table_S4", table_s4),
        ),
        args.speed_kmh,
    )

    audit_path = out_dir / "corrected_figures_audit.txt"
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"[OK] Figure values: {selected_values}")
    print(f"[OK] Audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
