#!/usr/bin/env python3
"""Generate and audit human-reference repeatability overlays.

This script uses the derived ``odometry_global_xy.csv`` files already exported
from the human-driven rosbags. It does not read the rosbags again.

For each of the six scenario/reference-speed pairs it:

1. discovers exactly two human executions;
2. resamples both trajectories by normalized arc length;
3. makes their traversal direction consistent;
4. rigidly aligns Human run 2 to Human run 1 (rotation + translation, no scale);
5. recomputes the pointwise mean trajectory;
6. compares that mean with the stored ``GTmean_rigidAligned_from_bag.csv``;
7. writes an individual two-panel diagnostic overlay;
8. writes a combined 3x2 publication overlay and a numerical audit CSV.

The benchmark-to-human-source mapping encoded here is the same mapping used by
the supplementary-table pipeline:

* benchmark 20 km/h -> human source labelled 15 km/h;
* benchmark 25 km/h -> human source labelled 20 km/h.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: int
    short_label: str
    descriptive_label: str
    per_bag_directory: str


SCENARIOS: Sequence[ScenarioSpec] = (
    ScenarioSpec(
        1,
        "S1",
        "Straight",
        "1-Scenario_reta_arco_Ground_truth",
    ),
    ScenarioSpec(
        2,
        "S2",
        "Straight + 20 m arc",
        "2-Scenario_reta_arco_20_Ground_truth",
    ),
    ScenarioSpec(
        3,
        "S3",
        "Straight + 40 m arc",
        "3-Scenario_reta_arco_40_Ground_truth",
    ),
)

SPEED_MAPPING: Sequence[Tuple[int, int]] = (
    (20, 15),
    (25, 20),
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Generate six human-run repeatability overlays and validate the "
            "stored rigid-aligned mean references."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(default_root),
        help="NAV2_Paper_Scripts root.",
    )
    parser.add_argument(
        "--per-bag-root",
        default="",
        help=(
            "Directory containing the three per-bag scenario folders "
            "(default: <project-root>/gt_from_bag_outputs_aligned/per_bag)."
        ),
    )
    parser.add_argument(
        "--mean-reference-dir",
        default="",
        help=(
            "Directory containing scenario*_GTmean_rigidAligned_from_bag.csv "
            "(default: <project-root>/gt_from_bag_outputs_aligned/gt_mean_references)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help=(
            "Output directory "
            "(default: <project-root>/results/human_reference_repeatability_overlays)."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Arc-length samples used for alignment and comparison (default: 300).",
    )
    parser.add_argument(
        "--expected-runs-per-condition",
        type=int,
        default=2,
        help="Required number of human runs per scenario/source-speed pair (default: 2).",
    )
    parser.add_argument(
        "--mean-match-tolerance-m",
        type=float,
        default=0.05,
        help=(
            "Maximum pointwise RMSE allowed between the recomputed and stored mean "
            "references before the audit is marked FAIL (default: 0.05 m)."
        ),
    )
    return parser.parse_args()


def natural_sort_key(value: str) -> List[object]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def read_xy_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"XY CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")

        x_key = find_column(
            reader.fieldnames,
            ("x_m", "x", "east_m", "easting_m", "position_x"),
        )
        y_key = find_column(
            reader.fieldnames,
            ("y_m", "y", "north_m", "northing_m", "position_y"),
        )
        if x_key is None or y_key is None:
            raise RuntimeError(
                f"Could not identify x/y columns in {path}. Headers: {reader.fieldnames}"
            )

        points: List[Tuple[float, float]] = []
        for row in reader:
            try:
                x = float(row[x_key])
                y = float(row[y_key])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))

    if len(points) < 2:
        raise RuntimeError(f"Fewer than two valid XY points in: {path}")

    xy = np.asarray(points, dtype=float)
    return remove_consecutive_duplicates(xy)


def remove_consecutive_duplicates(xy: np.ndarray, eps_m: float = 1e-9) -> np.ndarray:
    if xy.shape[0] <= 1:
        return xy.copy()
    keep = [0]
    for index in range(1, xy.shape[0]):
        if float(np.linalg.norm(xy[index] - xy[index - 1])) > eps_m:
            keep.append(index)
    return xy[np.asarray(keep, dtype=int)]


def cumulative_arc_length(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.asarray([], dtype=float)
    if xy.shape[0] == 1:
        return np.asarray([0.0], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def path_length(xy: np.ndarray) -> float:
    arc = cumulative_arc_length(xy)
    return float(arc[-1]) if arc.size else 0.0


def resample_by_arclength(xy: np.ndarray, n_samples: int) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("--n-samples must be at least 2")
    if xy.shape[0] < 2:
        raise ValueError("A trajectory must contain at least two points")

    arc = cumulative_arc_length(xy)
    if arc[-1] <= 1e-12:
        return np.repeat(xy[:1], n_samples, axis=0)
    target = np.linspace(0.0, float(arc[-1]), n_samples)
    x = np.interp(target, arc, xy[:, 0])
    y = np.interp(target, arc, xy[:, 1])
    return np.column_stack((x, y))


def orient_second_like_first(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, bool]:
    direct = float(np.mean(np.linalg.norm(first - second, axis=1)))
    reversed_second = second[::-1].copy()
    reversed_score = float(np.mean(np.linalg.norm(first - reversed_second, axis=1)))
    if reversed_score < direct:
        return reversed_second, True
    return second, False


def rigid_align_2d(reference: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    if reference.shape != moving.shape:
        raise ValueError("Rigid alignment requires trajectories with identical shapes")
    if reference.shape[0] < 2:
        raise ValueError("Rigid alignment requires at least two points")

    reference_centroid = np.mean(reference, axis=0)
    moving_centroid = np.mean(moving, axis=0)
    x = reference - reference_centroid
    y = moving - moving_centroid

    covariance = y.T @ x
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    aligned = (y @ rotation.T) + reference_centroid
    translation = reference_centroid - (moving_centroid @ rotation.T)
    angle_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return aligned, {
        "rotation_deg": float(angle_deg),
        "translation_x_m": float(translation[0]),
        "translation_y_m": float(translation[1]),
    }


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def heading_from_path(xy: np.ndarray) -> np.ndarray:
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    return np.arctan2(dy, dx)


def pair_metrics(first: np.ndarray, second_aligned: np.ndarray) -> Dict[str, float]:
    distance = np.linalg.norm(first - second_aligned, axis=1)
    heading_error = np.abs(
        wrap_angle_rad(heading_from_path(first) - heading_from_path(second_aligned))
    )
    return {
        "lateral_rmse_m": float(np.sqrt(np.mean(distance**2))),
        "lateral_p95_m": float(np.percentile(distance, 95.0)),
        "lateral_max_m": float(np.max(distance)),
        "heading_mean_deg": float(np.degrees(np.mean(heading_error))),
        "heading_p95_deg": float(np.degrees(np.percentile(heading_error, 95.0))),
        "heading_max_deg": float(np.degrees(np.max(heading_error))),
    }


def mean_reference_match(
    recomputed_mean: np.ndarray,
    stored_mean: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # The reference-generation script writes the averaged correspondence points
    # directly.  When the stored file already has n_samples rows, resampling it
    # again would perturb those correspondences and create an artificial audit
    # difference.  Resample only legacy files with another point count.
    stored_was_resampled = stored_mean.shape[0] != n_samples
    stored_resampled = (
        resample_by_arclength(stored_mean, n_samples)
        if stored_was_resampled
        else stored_mean.copy()
    )
    stored_oriented, stored_reversed = orient_second_like_first(
        recomputed_mean,
        stored_resampled,
    )
    difference = np.linalg.norm(recomputed_mean - stored_oriented, axis=1)
    return stored_oriented, {
        "stored_mean_resampled_for_comparison": int(stored_was_resampled),
        "stored_mean_reversed_for_comparison": int(stored_reversed),
        "stored_mean_pointwise_rmse_m": float(np.sqrt(np.mean(difference**2))),
        "stored_mean_pointwise_p95_m": float(np.percentile(difference, 95.0)),
        "stored_mean_pointwise_max_m": float(np.max(difference)),
    }


def set_equal_axes(axis: plt.Axes) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(True, alpha=0.25)


def mark_start_end(axis: plt.Axes, xy: np.ndarray, color: str) -> None:
    axis.scatter(xy[0, 0], xy[0, 1], color=color, marker="o", s=28, zorder=5)
    axis.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="x", s=38, zorder=5)


def plot_individual_overlay(
    output_stem: Path,
    scenario: ScenarioSpec,
    benchmark_speed: int,
    source_speed: int,
    run_names: Sequence[str],
    raw_first: np.ndarray,
    raw_second: np.ndarray,
    aligned_first: np.ndarray,
    aligned_second: np.ndarray,
    recomputed_mean: np.ndarray,
    stored_mean: np.ndarray,
    metrics: Mapping[str, float],
    mean_match: Mapping[str, float],
) -> None:
    figure = plt.figure(figsize=(16, 4.8))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.48),
        wspace=0.28,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    information_axis = figure.add_subplot(grid[0, 2])
    information_axis.axis("off")

    axes[0].plot(raw_first[:, 0], raw_first[:, 1], color="#1f77b4", linewidth=2.0, label="Human run 1")
    axes[0].plot(raw_second[:, 0], raw_second[:, 1], color="#ff7f0e", linewidth=2.0, label="Human run 2")
    mark_start_end(axes[0], raw_first, "#1f77b4")
    mark_start_end(axes[0], raw_second, "#ff7f0e")
    axes[0].set_title("Resampled executions (before rigid alignment)")
    set_equal_axes(axes[0])
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(aligned_first[:, 0], aligned_first[:, 1], color="#1f77b4", linewidth=1.7, label="Human run 1")
    axes[1].plot(aligned_second[:, 0], aligned_second[:, 1], color="#ff7f0e", linewidth=1.7, label="Human run 2, rigid-aligned")
    axes[1].plot(recomputed_mean[:, 0], recomputed_mean[:, 1], color="#2ca02c", linewidth=2.7, label="Recomputed mean")
    axes[1].plot(stored_mean[:, 0], stored_mean[:, 1], color="black", linestyle="--", linewidth=1.4, label="Stored mean reference")
    mark_start_end(axes[1], recomputed_mean, "#2ca02c")
    axes[1].set_title("Rigid-aligned executions and mean reference")
    set_equal_axes(axes[1])
    legend_handles, legend_labels = axes[1].get_legend_handles_labels()
    information_axis.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        frameon=True,
        fontsize=8,
    )

    text = (
        f"RMSE = {metrics['lateral_rmse_m']:.2f} m\n"
        f"P95 = {metrics['lateral_p95_m']:.2f} m\n"
        f"Heading mean = {metrics['heading_mean_deg']:.2f} deg\n"
        f"Heading P95 = {metrics['heading_p95_deg']:.2f} deg\n"
        f"Stored-mean match RMSE = {mean_match['stored_mean_pointwise_rmse_m']:.4f} m"
    )
    information_axis.text(
        0.02,
        0.52,
        text,
        transform=information_axis.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    figure.suptitle(
        f"{scenario.short_label}: {scenario.descriptive_label} | "
        f"benchmark {benchmark_speed} km/h | human source {source_speed} km/h\n"
        f"{run_names[0]} vs {run_names[1]}",
        fontsize=12,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.12,
        top=0.78,
        wspace=0.30,
    )
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty summary")
    fieldnames: List[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.n_samples < 2:
        print("ERROR: --n-samples must be at least 2", file=sys.stderr)
        return 2
    if args.expected_runs_per_condition < 2:
        print("ERROR: at least two human runs are required per condition", file=sys.stderr)
        return 2

    project_root = Path(args.project_root).expanduser().resolve()
    per_bag_root = (
        Path(args.per_bag_root).expanduser().resolve()
        if args.per_bag_root
        else project_root / "gt_from_bag_outputs_aligned" / "per_bag"
    )
    mean_reference_dir = (
        Path(args.mean_reference_dir).expanduser().resolve()
        if args.mean_reference_dir
        else project_root / "gt_from_bag_outputs_aligned" / "gt_mean_references"
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else project_root / "results" / "human_reference_repeatability_overlays"
    )
    individual_dir = out_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    plot_records: List[Dict[str, object]] = []
    preflight_errors: List[str] = []

    for scenario in SCENARIOS:
        scenario_root = per_bag_root / scenario.per_bag_directory
        for benchmark_speed, source_speed in SPEED_MAPPING:
            run_files = sorted(
                scenario_root.glob(
                    f"*vel_{source_speed}*/odometry_global_xy.csv"
                ),
                key=lambda path: natural_sort_key(str(path)),
            )
            if len(run_files) != args.expected_runs_per_condition:
                preflight_errors.append(
                    f"{scenario.short_label}, benchmark={benchmark_speed}, "
                    f"human_source={source_speed}: found {len(run_files)} "
                    f"odometry_global_xy.csv files; expected {args.expected_runs_per_condition}"
                )

            mean_file = (
                mean_reference_dir
                / (
                    f"scenario{scenario.scenario_id}_vel{source_speed}"
                    "_GTmean_rigidAligned_from_bag.csv"
                )
            )
            if not mean_file.is_file():
                preflight_errors.append(f"Missing stored mean reference: {mean_file}")

    if preflight_errors:
        print("[PREFLIGHT ERRORS]", file=sys.stderr)
        for error in preflight_errors:
            print(f"- {error}", file=sys.stderr)
        print("ERROR: no figures were generated.", file=sys.stderr)
        return 3

    for scenario in SCENARIOS:
        scenario_root = per_bag_root / scenario.per_bag_directory
        for benchmark_speed, source_speed in SPEED_MAPPING:
            run_files = sorted(
                scenario_root.glob(
                    f"*vel_{source_speed}*/odometry_global_xy.csv"
                ),
                key=lambda path: natural_sort_key(str(path)),
            )
            mean_file = (
                mean_reference_dir
                / (
                    f"scenario{scenario.scenario_id}_vel{source_speed}"
                    "_GTmean_rigidAligned_from_bag.csv"
                )
            )

            raw_first = read_xy_csv(run_files[0])
            raw_second = read_xy_csv(run_files[1])
            first = resample_by_arclength(raw_first, args.n_samples)
            second = resample_by_arclength(raw_second, args.n_samples)
            second_oriented, reversed_used = orient_second_like_first(first, second)
            second_aligned, alignment = rigid_align_2d(first, second_oriented)
            recomputed_mean = 0.5 * (first + second_aligned)

            stored_raw = read_xy_csv(mean_file)
            stored_mean, match = mean_reference_match(
                recomputed_mean,
                stored_raw,
                args.n_samples,
            )
            metrics = pair_metrics(first, second_aligned)
            match_pass = (
                match["stored_mean_pointwise_rmse_m"]
                <= args.mean_match_tolerance_m
            )

            run_names = [path.parent.name for path in run_files]
            output_stem = (
                individual_dir
                / (
                    f"{scenario.short_label}_benchmark{benchmark_speed}"
                    f"_human{source_speed}_repeatability"
                )
            )
            plot_individual_overlay(
                output_stem=output_stem,
                scenario=scenario,
                benchmark_speed=benchmark_speed,
                source_speed=source_speed,
                run_names=run_names,
                raw_first=first,
                raw_second=second_oriented,
                aligned_first=first,
                aligned_second=second_aligned,
                recomputed_mean=recomputed_mean,
                stored_mean=stored_mean,
                metrics=metrics,
                mean_match=match,
            )

            row: Dict[str, object] = {
                "scenario": scenario.short_label,
                "scenario_id": scenario.scenario_id,
                "scenario_description": scenario.descriptive_label,
                "benchmark_speed_kmh": benchmark_speed,
                "human_source_speed_kmh": source_speed,
                "N_human": len(run_files),
                "human_run_1": run_names[0],
                "human_run_2": run_names[1],
                "human_run_1_csv": str(run_files[0]),
                "human_run_2_csv": str(run_files[1]),
                "stored_mean_csv": str(mean_file),
                "second_run_reversed": int(reversed_used),
                "raw_start_offset_m": float(np.linalg.norm(first[0] - second_oriented[0])),
                "raw_end_offset_m": float(np.linalg.norm(first[-1] - second_oriented[-1])),
                "run_1_length_m": path_length(raw_first),
                "run_2_length_m": path_length(raw_second),
                **alignment,
                **metrics,
                **match,
                "mean_match_tolerance_m": args.mean_match_tolerance_m,
                "mean_reference_audit": "PASS" if match_pass else "FAIL",
                "individual_png": str(output_stem.with_suffix(".png")),
                "individual_pdf": str(output_stem.with_suffix(".pdf")),
            }
            summaries.append(row)
            plot_records.append(
                {
                    "scenario": scenario,
                    "benchmark_speed": benchmark_speed,
                    "source_speed": source_speed,
                    "first": first,
                    "second": second_aligned,
                    "stored_mean": stored_mean,
                    "metrics": metrics,
                }
            )

            print(
                f"[OK] {scenario.short_label}, benchmark {benchmark_speed} km/h "
                f"(human source {source_speed} km/h): "
                f"RMSE={metrics['lateral_rmse_m']:.3f} m, "
                f"P95={metrics['lateral_p95_m']:.3f} m, "
                f"stored-mean match={match['stored_mean_pointwise_rmse_m']:.6f} m, "
                f"audit={'PASS' if match_pass else 'FAIL'}"
            )

    summary_csv = out_dir / "human_reference_repeatability_audit.csv"
    write_csv(summary_csv, summaries)

    combined_figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.5))
    for axis, record in zip(axes.flat, plot_records):
        scenario = record["scenario"]
        assert isinstance(scenario, ScenarioSpec)
        first = np.asarray(record["first"])
        second = np.asarray(record["second"])
        stored_mean = np.asarray(record["stored_mean"])
        metrics = record["metrics"]
        assert isinstance(metrics, Mapping)

        axis.plot(first[:, 0], first[:, 1], color="#1f77b4", linewidth=1.5, label="Human run 1")
        axis.plot(second[:, 0], second[:, 1], color="#ff7f0e", linewidth=1.5, label="Human run 2, rigid-aligned")
        axis.plot(stored_mean[:, 0], stored_mean[:, 1], color="#2ca02c", linewidth=2.6, label="Mean reference")
        mark_start_end(axis, stored_mean, "#2ca02c")
        axis.set_title(
            f"{scenario.short_label}: {scenario.descriptive_label}\n"
            f"benchmark {record['benchmark_speed']} km/h "
            f"(human source {record['source_speed']} km/h)\n"
            f"RMSE={float(metrics['lateral_rmse_m']):.2f} m, "
            f"P95={float(metrics['lateral_p95_m']):.2f} m"
        )
        set_equal_axes(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    combined_figure.suptitle(
        "Human-reference repeatability before mean-trajectory benchmarking",
        fontsize=15,
        y=0.992,
    )
    combined_figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=3,
        frameon=False,
    )
    combined_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    combined_png = out_dir / "human_reference_repeatability_overlays_3x2.png"
    combined_pdf = out_dir / "human_reference_repeatability_overlays_3x2.pdf"
    combined_figure.savefig(combined_png, dpi=300, bbox_inches="tight")
    combined_figure.savefig(combined_pdf, bbox_inches="tight")
    plt.close(combined_figure)

    failed = [row for row in summaries if row["mean_reference_audit"] != "PASS"]
    audit_txt = out_dir / "human_reference_repeatability_audit.txt"
    with audit_txt.open("w", encoding="utf-8") as stream:
        stream.write("HUMAN REFERENCE REPEATABILITY AUDIT\n\n")
        stream.write(f"Conditions processed: {len(summaries)}\n")
        stream.write(f"Expected conditions: {len(SCENARIOS) * len(SPEED_MAPPING)}\n")
        stream.write(f"Human executions used: {sum(int(row['N_human']) for row in summaries)}\n")
        stream.write(f"Mean-reference tolerance [m]: {args.mean_match_tolerance_m:.6f}\n")
        stream.write(f"Mean-reference failures: {len(failed)}\n")
        stream.write(f"Audit status: {'PASS' if not failed else 'FAIL'}\n\n")
        for row in summaries:
            stream.write(
                f"{row['scenario']} benchmark={row['benchmark_speed_kmh']} km/h "
                f"human_source={row['human_source_speed_kmh']} km/h: "
                f"N={row['N_human']}, RMSE={float(row['lateral_rmse_m']):.3f} m, "
                f"P95={float(row['lateral_p95_m']):.3f} m, "
                f"stored_mean_match={float(row['stored_mean_pointwise_rmse_m']):.6f} m, "
                f"{row['mean_reference_audit']}\n"
            )

    print("\n[OUTPUTS]")
    print(f"- Combined PNG: {combined_png}")
    print(f"- Combined PDF: {combined_pdf}")
    print(f"- Numerical audit CSV: {summary_csv}")
    print(f"- Audit report: {audit_txt}")
    print(f"- Individual overlays: {individual_dir}")

    if failed:
        print(
            "ERROR: one or more recomputed means did not match the stored mean "
            "within tolerance. Inspect the individual diagnostics.",
            file=sys.stderr,
        )
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
