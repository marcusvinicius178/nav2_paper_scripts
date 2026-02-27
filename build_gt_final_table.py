#!/usr/bin/env python3
"""
Build final GT reproducibility table for paper/report from gt_from_rosbag_analysis outputs.

What it does
- Reads gt_pair_reproducibility_from_bag.csv
- Optionally reads within_bag_topic_consistency.csv
- Builds a compact "paper-ready" table with selected metrics
- Flags suspicious/outlier rows
- Exports:
    1) CSV table
    2) Markdown table
    3) Diagnostics CSV (full row + flags)

Usage example:
  python3 /home/marcus/NAV2_Paper_Scripts/build_gt_final_table.py \
    --input-csv /home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/gt_pair_reproducibility_from_bag.csv \
    --within-csv /home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/within_bag_topic_consistency.csv \
    --out-dir /home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/final_tables
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCENARIO_NAME_MAP = {
    1: "Straight",
    2: "Straight + Arc20m",
    3: "Straight + Arc40m",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        x = float(s)
        if math.isfinite(x):
            return x
        return None
    except Exception:
        return None


def safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def fmt_num(v: Optional[float], ndigits: int = 3) -> str:
    if v is None:
        return ""
    return f"{v:.{ndigits}f}"


def fmt_int(v: Optional[int]) -> str:
    if v is None:
        return ""
    return str(v)


def parse_scenario_name(scenario_id: Optional[int]) -> str:
    if scenario_id is None:
        return "Unknown"
    return SCENARIO_NAME_MAP.get(scenario_id, f"Scenario {scenario_id}")


def mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_within_bag_aggregates(within_csv: Optional[Path]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Aggregates within-bag odom vs gps projected metrics by (scenario_id, speed_kmh).
    Returns dict with mean and max values for context.
    """
    if within_csv is None or not within_csv.exists():
        return {}

    rows = read_csv_rows(within_csv)
    grouped: Dict[Tuple[int, int], List[Dict[str, str]]] = {}

    for r in rows:
        sid = safe_int(r.get("scenario_id"))
        spd = safe_int(r.get("speed_kmh"))
        if sid is None or spd is None:
            continue
        grouped.setdefault((sid, spd), []).append(r)

    agg: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, items in grouped.items():
        rmse_vals = []
        p95_vals = []
        hdg_vals = []
        for r in items:
            x = safe_float(r.get("lateral_error_rmse_m"))
            if x is not None:
                rmse_vals.append(x)
            x = safe_float(r.get("lateral_error_p95_m"))
            if x is not None:
                p95_vals.append(x)
            x = safe_float(r.get("heading_error_mean_deg"))
            if x is not None:
                hdg_vals.append(x)

        agg[key] = {
            "withinbag_odom_vs_gps_rmse_mean_m": mean(rmse_vals) if rmse_vals else None,
            "withinbag_odom_vs_gps_rmse_max_m": max(rmse_vals) if rmse_vals else None,
            "withinbag_odom_vs_gps_p95_mean_m": mean(p95_vals) if p95_vals else None,
            "withinbag_odom_vs_gps_heading_mean_deg": mean(hdg_vals) if hdg_vals else None,
            "withinbag_count": len(items),
        }

    return agg


def classify_row(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Build flags based on raw and aligned metrics.
    """
    raw_rmse = safe_float(row.get("raw_lateral_error_rmse_m"))
    aligned_rmse = safe_float(row.get("lateral_error_rmse_m"))
    raw_p95 = safe_float(row.get("raw_lateral_error_p95_m"))
    aligned_p95 = safe_float(row.get("lateral_error_p95_m"))
    heading_mean = safe_float(row.get("heading_error_mean_deg"))
    heading_p95 = safe_float(row.get("heading_error_p95_deg"))
    start_diff = safe_float(row.get("start_point_diff_m_raw"))
    end_diff = safe_float(row.get("end_point_diff_m_raw"))
    path_len_diff = safe_float(row.get("path_length_abs_diff_m"))
    align_rot_deg = safe_float(row.get("align_rotation_deg"))
    align_tx = safe_float(row.get("align_tx_m"))
    align_ty = safe_float(row.get("align_ty_m"))

    align_mag = None
    if align_tx is not None and align_ty is not None:
        align_mag = math.sqrt(align_tx * align_tx + align_ty * align_ty)

    ratio_raw_to_aligned = None
    if raw_rmse is not None and aligned_rmse is not None and aligned_rmse > 1e-9:
        ratio_raw_to_aligned = raw_rmse / aligned_rmse

    flags: List[str] = []

    # Frame offset suspicion, classic case when raw RMSE is huge and aligned RMSE is much smaller
    if ratio_raw_to_aligned is not None and ratio_raw_to_aligned >= 3.0:
        flags.append("FRAME_OFFSET_SUSPECTED")

    if start_diff is not None and start_diff >= 10.0:
        flags.append("LARGE_START_OFFSET")
    if end_diff is not None and end_diff >= 10.0:
        flags.append("LARGE_END_OFFSET")
    if align_mag is not None and align_mag >= 10.0:
        flags.append("LARGE_ALIGNMENT_TRANSLATION")

    # Reproducibility quality on aligned metrics, conservative thresholds
    if aligned_rmse is not None:
        if aligned_rmse > 5.0:
            flags.append("HIGH_ALIGNED_RMSE")
        elif aligned_rmse > 2.0:
            flags.append("MODERATE_ALIGNED_RMSE")

    if aligned_p95 is not None and aligned_p95 > 10.0:
        flags.append("HIGH_ALIGNED_P95")

    if heading_mean is not None:
        if heading_mean > 8.0:
            flags.append("HIGH_HEADING_MEAN")
        elif heading_mean > 3.0:
            flags.append("MODERATE_HEADING_MEAN")

    if heading_p95 is not None and heading_p95 > 15.0:
        flags.append("HIGH_HEADING_P95")

    if path_len_diff is not None and path_len_diff > 12.0:
        flags.append("PATH_LENGTH_MISMATCH")

    if not flags:
        quality = "OK"
    elif "FRAME_OFFSET_SUSPECTED" in flags and aligned_rmse is not None and aligned_rmse <= 3.0:
        quality = "OK_AFTER_ALIGNMENT"
    elif any(f in flags for f in ["HIGH_ALIGNED_RMSE", "HIGH_ALIGNED_P95", "HIGH_HEADING_MEAN", "HIGH_HEADING_P95"]):
        quality = "CHECK"
    else:
        quality = "ATTENTION"

    return {
        "flags": ";".join(flags),
        "quality_label": quality,
        "ratio_raw_to_aligned_rmse": ratio_raw_to_aligned,
        "alignment_translation_norm_m": align_mag,
        "alignment_rotation_abs_deg": abs(align_rot_deg) if align_rot_deg is not None else None,
    }


def build_final_rows(
    gt_rows: List[Dict[str, str]],
    within_agg: Dict[Tuple[int, int], Dict[str, float]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    paper_rows: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []

    for r in gt_rows:
        sid = safe_int(r.get("scenario_id"))
        spd = safe_int(r.get("speed_kmh"))
        scenario_name = parse_scenario_name(sid)

        classification = classify_row(r)

        raw_rmse = safe_float(r.get("raw_lateral_error_rmse_m"))
        raw_p95 = safe_float(r.get("raw_lateral_error_p95_m"))
        aligned_rmse = safe_float(r.get("lateral_error_rmse_m"))
        aligned_p95 = safe_float(r.get("lateral_error_p95_m"))
        aligned_max = safe_float(r.get("lateral_error_max_m"))
        heading_mean = safe_float(r.get("heading_error_mean_deg"))
        heading_p95 = safe_float(r.get("heading_error_p95_deg"))
        heading_max = safe_float(r.get("heading_error_max_deg"))
        path_len_diff = safe_float(r.get("path_length_abs_diff_m"))
        start_diff = safe_float(r.get("start_point_diff_m_raw"))
        end_diff = safe_float(r.get("end_point_diff_m_raw"))

        key = (sid, spd) if sid is not None and spd is not None else None
        within = within_agg.get(key, {}) if key is not None else {}

        paper_row = {
            "scenario_id": sid,
            "scenario_name": scenario_name,
            "speed_kmh": spd,
            "comparison_align_mode": r.get("gt_pair_comparison_alignment", r.get("align_mode", "none")),
            "mean_gt_align_mode": r.get("mean_gt_generation_alignment", ""),
            "gt1_bag_name": r.get("gt1_bag_name", ""),
            "gt2_bag_name": r.get("gt2_bag_name", ""),
            "raw_rmse_m": raw_rmse,
            "raw_p95_m": raw_p95,
            "aligned_rmse_m": aligned_rmse,
            "aligned_p95_m": aligned_p95,
            "aligned_max_m": aligned_max,
            "heading_mean_deg": heading_mean,
            "heading_p95_deg": heading_p95,
            "heading_max_deg": heading_max,
            "path_length_abs_diff_m": path_len_diff,
            "start_point_diff_raw_m": start_diff,
            "end_point_diff_raw_m": end_diff,
            "alignment_translation_norm_m": classification["alignment_translation_norm_m"],
            "alignment_rotation_abs_deg": classification["alignment_rotation_abs_deg"],
            "ratio_raw_to_aligned_rmse": classification["ratio_raw_to_aligned_rmse"],
            "withinbag_odom_vs_gps_rmse_mean_m": within.get("withinbag_odom_vs_gps_rmse_mean_m"),
            "withinbag_odom_vs_gps_p95_mean_m": within.get("withinbag_odom_vs_gps_p95_mean_m"),
            "withinbag_odom_vs_gps_heading_mean_deg": within.get("withinbag_odom_vs_gps_heading_mean_deg"),
            "quality_label": classification["quality_label"],
            "flags": classification["flags"],
            "mean_gt_csv": r.get("mean_gt_csv", ""),
        }
        paper_rows.append(paper_row)

        diag_row = dict(r)
        diag_row.update({
            "scenario_name": scenario_name,
            "quality_label": classification["quality_label"],
            "flags": classification["flags"],
            "ratio_raw_to_aligned_rmse": classification["ratio_raw_to_aligned_rmse"],
            "alignment_translation_norm_m": classification["alignment_translation_norm_m"],
            "alignment_rotation_abs_deg": classification["alignment_rotation_abs_deg"],
            "withinbag_odom_vs_gps_rmse_mean_m": within.get("withinbag_odom_vs_gps_rmse_mean_m"),
            "withinbag_odom_vs_gps_p95_mean_m": within.get("withinbag_odom_vs_gps_p95_mean_m"),
            "withinbag_odom_vs_gps_heading_mean_deg": within.get("withinbag_odom_vs_gps_heading_mean_deg"),
        })
        diagnostics_rows.append(diag_row)

    paper_rows.sort(key=lambda x: (
        999 if x["scenario_id"] is None else x["scenario_id"],
        999 if x["speed_kmh"] is None else x["speed_kmh"],
    ))

    diagnostics_rows.sort(key=lambda x: (
        0 if str(x.get("quality_label", "")) in ("CHECK", "ATTENTION") else 1,
        -(safe_float(x.get("raw_lateral_error_rmse_m")) or -1.0),
    ))

    return paper_rows, diagnostics_rows


def rows_to_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """
    Compact markdown table for article/report appendix.
    """
    headers = [
        "Scenario",
        "Speed",
        "Raw RMSE (m)",
        "Aligned RMSE (m)",
        "Aligned P95 (m)",
        "Heading mean (deg)",
        "Heading P95 (deg)",
        "Quality",
        "Flags",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        line = [
            f"{fmt_int(r.get('scenario_id'))} ({r.get('scenario_name','')})",
            f"{fmt_int(r.get('speed_kmh'))}",
            fmt_num(r.get("raw_rmse_m"), 3),
            fmt_num(r.get("aligned_rmse_m"), 3),
            fmt_num(r.get("aligned_p95_m"), 3),
            fmt_num(r.get("heading_mean_deg"), 3),
            fmt_num(r.get("heading_p95_deg"), 3),
            str(r.get("quality_label", "")),
            str(r.get("flags", "")),
        ]
        lines.append("| " + " | ".join(line) + " |")

    return "\n".join(lines) + "\n"


def build_article_notes(rows: List[Dict[str, Any]]) -> str:
    """
    Produces a small text summary to help you describe the results in the paper.
    """
    if not rows:
        return "No rows available.\n"

    aligned_rmse_vals = [r["aligned_rmse_m"] for r in rows if isinstance(r.get("aligned_rmse_m"), (int, float))]
    raw_rmse_vals = [r["raw_rmse_m"] for r in rows if isinstance(r.get("raw_rmse_m"), (int, float))]
    ratios = [r["ratio_raw_to_aligned_rmse"] for r in rows if isinstance(r.get("ratio_raw_to_aligned_rmse"), (int, float))]

    lines = []
    lines.append("Suggested article notes")
    lines.append("")
    lines.append(f"Rows (scenario x speed pairs): {len(rows)}")

    if raw_rmse_vals:
        lines.append(
            "Raw GT pair RMSE range (before alignment), "
            f"{min(raw_rmse_vals):.3f} to {max(raw_rmse_vals):.3f} m"
        )
    if aligned_rmse_vals:
        lines.append(
            "Aligned GT pair RMSE range (after compensation), "
            f"{min(aligned_rmse_vals):.3f} to {max(aligned_rmse_vals):.3f} m"
        )
    if ratios:
        lines.append(
            "Raw/aligned RMSE ratio range, "
            f"{min(ratios):.3f} to {max(ratios):.3f}"
        )

    flagged = [r for r in rows if str(r.get("flags", "")).strip()]
    if flagged:
        lines.append("")
        lines.append("Rows flagged for inspection")
        for r in flagged:
            lines.append(
                f"  scenario {r.get('scenario_id')}, speed {r.get('speed_kmh')} km/h, "
                f"quality={r.get('quality_label')}, flags={r.get('flags')}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final GT reproducibility table for paper/report.")
    parser.add_argument(
        "--input-csv",
        default="/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/gt_pair_reproducibility_from_bag.csv",
        help="Path to gt_pair_reproducibility_from_bag.csv"
    )
    parser.add_argument(
        "--within-csv",
        default="/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/within_bag_topic_consistency.csv",
        help="Optional path to within_bag_topic_consistency.csv"
    )
    parser.add_argument(
        "--out-dir",
        default="/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/final_tables",
        help="Output directory for final tables"
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    within_csv = Path(args.within_csv).expanduser().resolve() if args.within_csv else None
    out_dir = Path(args.out_dir).expanduser().resolve()

    gt_rows = read_csv_rows(input_csv)
    within_agg = load_within_bag_aggregates(within_csv)

    paper_rows, diagnostics_rows = build_final_rows(gt_rows, within_agg)

    # CSV output (numeric values preserved)
    paper_csv = out_dir / "gt_reproducibility_paper_table.csv"
    if paper_rows:
        paper_fields = list(paper_rows[0].keys())
        write_csv(paper_csv, paper_rows, paper_fields)

    # Diagnostics CSV (full details + flags)
    diag_csv = out_dir / "gt_reproducibility_diagnostics.csv"
    if diagnostics_rows:
        diag_fields = list(diagnostics_rows[0].keys())
        write_csv(diag_csv, diagnostics_rows, diag_fields)

    # Markdown table (rounded, compact)
    md_table_path = out_dir / "gt_reproducibility_paper_table.md"
    md_table = rows_to_markdown_table(paper_rows)
    md_table_path.parent.mkdir(parents=True, exist_ok=True)
    md_table_path.write_text(md_table, encoding="utf-8")

    # Notes for article text
    notes_path = out_dir / "gt_reproducibility_article_notes.txt"
    notes_path.write_text(build_article_notes(paper_rows), encoding="utf-8")

    print(f"[OK] Input GT pair CSV: {input_csv}")
    if within_csv and within_csv.exists():
        print(f"[OK] Within-bag CSV used: {within_csv}")
    else:
        print(f"[INFO] Within-bag CSV not found or not used: {within_csv}")
    print(f"[OK] Output dir: {out_dir}")
    print(f"[OK] Paper CSV: {paper_csv}")
    print(f"[OK] Diagnostics CSV: {diag_csv}")
    print(f"[OK] Markdown table: {md_table_path}")
    print(f"[OK] Article notes: {notes_path}")

    # Console preview
    print("\nPreview (Scenario, Speed, Raw RMSE, Aligned RMSE, Flags):")
    for r in paper_rows:
        print(
            f"  S{r['scenario_id']} @ {r['speed_kmh']} km/h | "
            f"raw={fmt_num(r.get('raw_rmse_m'))} m | "
            f"aligned={fmt_num(r.get('aligned_rmse_m'))} m | "
            f"{r.get('quality_label')} | {r.get('flags')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())