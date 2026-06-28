#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import the single-run extractor as a module
import subprocess
import json


@dataclass
class SummaryStats:
    median: float
    iqr: float


def median_iqr(x: np.ndarray) -> SummaryStats:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return SummaryStats(float("nan"), float("nan"))
    q1 = float(np.percentile(x, 25.0))
    q3 = float(np.percentile(x, 75.0))
    med = float(np.median(x))
    return SummaryStats(med, q3 - q1)


def find_mcap_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".mcap") and not fn.endswith(".mcap.zstd"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def parse_group_from_path(p: str) -> Dict[str, str]:
    parts = os.path.normpath(p).split(os.sep)
    labels: Dict[str, str] = {}
    for part in parts:
        if part.lower().startswith("scenario"):
            labels["scenario"] = part
            break
    for part in parts:
        if "_vel_" in part.lower():
            idx = part.lower().find("_vel_")
            labels["method"] = part[:idx]
            labels["speed_kmh"] = part[idx + len("_vel_"):]
            labels["group_folder"] = part
            break
    labels["run_folder"] = parts[-2] if len(parts) >= 2 else ""
    return labels


def run_extract_script(extract_py: str, bag_path: str, gt_bag: str, out_csv: str) -> Dict:
    cmd = ["python3", extract_py, "--bag", bag_path, "--out-csv", out_csv]
    if gt_bag:
        cmd += ["--gt-bag", gt_bag]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Extractor failed for bag:\n  {bag_path}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    # The extractor prints JSON to stdout
    return json.loads(proc.stdout)


def main():
    ap = argparse.ArgumentParser(description="Batch extract and summarize Nav2 benchmark metrics from a folder of MCAP bags.")
    ap.add_argument("--root", required=True, help="Root folder containing the bags.")
    ap.add_argument("--extract-script", default="extract_nav2_metrics.py", help="Path to extract_nav2_metrics.py")
    ap.add_argument("--gt-root", default="", help="Optional folder root with GT driver bags, matched by scenario and speed.")
    ap.add_argument("--out-dir", default="", help="Output directory for CSV summaries. Default: <root>/_metrics")
    ap.add_argument("--per-run-csv", default="", help="Path for per-run CSV (appended). Default: <out-dir>/all_runs.csv")
    ap.add_argument("--summary-csv", default="", help="Path for grouped summary CSV. Default: <out-dir>/summary_by_group.csv")

    args = ap.parse_args()

    root = os.path.abspath(args.root)
    extract_py = os.path.abspath(args.extract_script)

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(root, "_metrics")
    os.makedirs(out_dir, exist_ok=True)

    per_run_csv = os.path.abspath(args.per_run_csv) if args.per_run_csv else os.path.join(out_dir, "all_runs.csv")
    summary_csv = os.path.abspath(args.summary_csv) if args.summary_csv else os.path.join(out_dir, "summary_by_group.csv")

    bags = find_mcap_files(root)
    if not bags:
        raise RuntimeError(f"No .mcap files found under: {root}")

    # Optional GT lookup map: (scenario, speed_kmh) -> gt_bag_path
    gt_map: Dict[Tuple[str, str], str] = {}
    if args.gt_root:
        gt_root = os.path.abspath(args.gt_root)
        gt_bags = find_mcap_files(gt_root)
        for gb in gt_bags:
            lab = parse_group_from_path(gb)
            sc = lab.get("scenario", "")
            sp = lab.get("speed_kmh", "")
            if sc and sp:
                gt_map[(sc, sp)] = gb

    # Extract all runs
    rows = []
    for b in bags:
        lab = parse_group_from_path(b)
        sc = lab.get("scenario", "")
        sp = lab.get("speed_kmh", "")
        gt_bag = gt_map.get((sc, sp), "")
        try:
            res = run_extract_script(extract_py, b, gt_bag, per_run_csv)
            rows.append(res)
            print(f"OK: {b}")
        except Exception as e:
            print(f"FAIL: {b}\n{e}")

    if not rows:
        raise RuntimeError("No runs extracted successfully, check errors above.")

    df = pd.DataFrame(rows)

    # Group keys
    keys = ["scenario", "method", "speed_kmh"]
    grouped = df.groupby(keys, dropna=False)

    # Metrics to summarize
    rate_metrics = ["sr_action", "sr_geo"]
    scalar_metrics = [
        "mission_time_s",
        "progress_ratio",
        "rmse_y_m",
        "p95_y_m",
        "rmse_psi_rad",
        "rmse_v_ms",
        "rms_jerk_ms3",
        "rms_yawrate_dot_rads2",
    ]

    out_rows = []
    for gk, gdf in grouped:
        row = {"scenario": gk[0], "method": gk[1], "speed_kmh": gk[2], "N": int(len(gdf))}
        # rates as count and percent
        for rm in rate_metrics:
            vals = gdf[rm].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[f"{rm}_count"] = ""
                row[f"{rm}_pct"] = float("nan")
            else:
                count = int(np.sum(vals >= 0.5))
                row[f"{rm}_count"] = f"{count}/{len(gdf)}"
                row[f"{rm}_pct"] = float(100.0 * count / len(gdf))

        # scalars as median and IQR
        for sm in scalar_metrics:
            vals = gdf[sm].to_numpy(dtype=float)
            st = median_iqr(vals)
            row[f"{sm}_median"] = st.median
            row[f"{sm}_iqr"] = st.iqr

        out_rows.append(row)

    sdf = pd.DataFrame(out_rows)
    sdf.to_csv(summary_csv, index=False)
    print(f"\nWrote per-run CSV: {per_run_csv}")
    print(f"Wrote summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()