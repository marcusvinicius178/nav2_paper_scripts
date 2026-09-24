#!/usr/bin/env python3
"""Deterministic tests for plan cross-track v2 and table populations."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def install_ros_import_stubs_if_needed() -> None:
    """Allow pure numerical tests to run outside a sourced ROS environment."""
    if importlib.util.find_spec("rosbag2_py") is not None:
        return
    rosbag2_py = types.ModuleType("rosbag2_py")
    rclpy = types.ModuleType("rclpy")
    rclpy_serialization = types.ModuleType("rclpy.serialization")
    rosidl_runtime_py = types.ModuleType("rosidl_runtime_py")
    rosidl_utilities = types.ModuleType("rosidl_runtime_py.utilities")
    rclpy_serialization.deserialize_message = lambda *_args, **_kwargs: None
    rosidl_utilities.get_message = lambda *_args, **_kwargs: None
    sys.modules["rosbag2_py"] = rosbag2_py
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.serialization"] = rclpy_serialization
    sys.modules["rosidl_runtime_py"] = rosidl_runtime_py
    sys.modules["rosidl_runtime_py.utilities"] = rosidl_utilities


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"actual={actual}, expected={expected}")


def test_endpoint_tangent_extension(evaluator) -> None:
    reference = np.asarray([[5.0, 0.0], [10.0, 0.0]], dtype=float)

    before = np.asarray([[0.0, 1.0]], dtype=float)
    finite = evaluator.project_points_to_polyline(before, reference)
    corrected = evaluator.project_points_to_polyline(
        before,
        reference,
        extend_endpoint_tangents=True,
    )
    assert_close(float(finite["dist"][0]), math.sqrt(26.0))
    assert_close(float(corrected["dist"][0]), 1.0)
    assert_close(float(corrected["finite_segment_dist"][0]), math.sqrt(26.0))
    if not bool(corrected["endpoint_extrapolated"][0]):
        raise AssertionError("Expected first-segment tangent extension")

    after = np.asarray([[15.0, 2.0]], dtype=float)
    corrected_after = evaluator.project_points_to_polyline(
        after,
        reference,
        extend_endpoint_tangents=True,
    )
    assert_close(float(corrected_after["dist"][0]), 2.0)
    if not bool(corrected_after["endpoint_extrapolated"][0]):
        raise AssertionError("Expected last-segment tangent extension")

    interior = np.asarray([[7.0, 3.0]], dtype=float)
    corrected_interior = evaluator.project_points_to_polyline(
        interior,
        reference,
        extend_endpoint_tangents=True,
    )
    assert_close(float(corrected_interior["dist"][0]), 3.0)
    if bool(corrected_interior["endpoint_extrapolated"][0]):
        raise AssertionError("Interior projection must not use endpoint extension")


def make_row(
    bag_name: str,
    valid_mission: int,
    valid_plan: int,
    success: int,
    progress: float,
) -> dict[str, str]:
    metric_value = "1.0" if valid_plan else ""
    mission_value = "2.0" if valid_mission else ""
    human_value = "3.0" if valid_mission else ""
    return {
        "scenario_id": "1",
        "speed_kmh": "20",
        "planner_id": "NavFn",
        "controller_id": "MPPI",
        "bag_name": bag_name,
        "bag_dir": f"/synthetic/{bag_name}",
        "success_geo": str(success),
        "progress_ratio": str(progress),
        "valid_for_mission": str(valid_mission),
        "valid_for_tracking": str(valid_plan),
        "mission_time_s": mission_value,
        "rmse_y_primary_m": metric_value,
        "p95_y_primary_m": metric_value,
        "rmse_psi_primary_rad": metric_value,
        "rmse_v_mps": mission_value,
        "rms_jx_mps3": mission_value,
        "rms_dotdelta_radps": mission_value,
        "human_gt_rmse_y_m": human_value,
        "human_gt_p95_y_m": human_value,
        "human_gt_max_y_m": human_value,
        "human_gt_progress": "0.9" if valid_mission else "",
        "human_gt_pointwise_rmse_m": human_value,
    }


def test_table_populations(builder) -> None:
    key = (1, 20, "NavFn", "MPPI", "nominal")
    rows = [
        make_row("run_1", 1, 1, 1, 0.95),
        make_row("run_2", 1, 0, 0, 0.80),
        make_row("run_3", 0, 0, 0, 0.30),
    ]
    table_s2, table_s3, table_s4 = builder.build_table_rows({key: rows})
    s2 = table_s2[0]
    s3 = table_s3[0]
    s4 = table_s4[0]

    expected_counts = {
        "N_total": 3,
        "N_success": 1,
        "N_mission": 2,
        "N_plan": 1,
        "N_human": 2,
    }
    for field, expected in expected_counts.items():
        if int(s2[field]) != expected:
            raise AssertionError(f"{field}: actual={s2[field]}, expected={expected}")

    if int(s2["PR_N"]) != 3:
        raise AssertionError("Progress ratio must use all attempts")
    if int(s2["T_s_N"]) != 2:
        raise AssertionError("Mission time must use mission-valid runs")
    if int(s2["RMSE_y_m_N"]) != 1 or int(s2["P95_y_m_N"]) != 1:
        raise AssertionError("Lateral tracking metrics must use plan-valid runs")
    if int(s3["RMSE_psi_rad_N"]) != 1:
        raise AssertionError("Heading tracking must use plan-valid runs")
    for field in ("RMSE_v_mps_N", "RMS_jx_mps3_N", "RMS_dotdelta_radps_N"):
        if int(s3[field]) != 2:
            raise AssertionError(f"{field} must use mission-valid runs")
    if int(s4["Human_RMSE_y_m_N"]) != 2:
        raise AssertionError("Human-likeness must not depend on plan validity")

    latex = builder.latex_table(
        "Synthetic",
        "Synthetic population test.",
        "tab:synthetic",
        table_s2,
        ("PR", "RMSE"),
        ("PR", "RMSE_y_m"),
        (
            (r"$N_{total}$", "N_total"),
            (r"$N_{mission}$", "N_mission"),
            (r"$N_{plan}$", "N_plan"),
        ),
    )
    if r"$N_{mission}$" not in latex or "3 & 2 & 1" not in latex:
        raise AssertionError("LaTeX validity-count columns were not rendered")


def main() -> int:
    install_ros_import_stubs_if_needed()
    evaluator = load_module(
        "eval_nav2_one_combination_cross_track_v2",
        SCRIPT_DIR / "eval_nav2_one_combination.py",
    )
    builder = load_module(
        "build_supplementary_tables_cross_track_v2",
        SCRIPT_DIR / "build_supplementary_tables.py",
    )
    test_endpoint_tangent_extension(evaluator)
    test_table_populations(builder)
    print("Endpoint tangent cross-track test: PASS")
    print("Metric population separation test: PASS")
    print("Plan cross-track v2 tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
