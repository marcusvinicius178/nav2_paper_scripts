#!/usr/bin/env python3
"""Numerical regression tests for the corrected jerk and steering-rate metrics."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np


def install_ros_import_stubs() -> None:
    """Install minimal stubs so the metric module can be tested without ROS 2."""
    rosbag2_py = types.ModuleType("rosbag2_py")
    sys.modules.setdefault("rosbag2_py", rosbag2_py)

    rclpy = types.ModuleType("rclpy")
    rclpy_serialization = types.ModuleType("rclpy.serialization")
    rclpy_serialization.deserialize_message = lambda *args, **kwargs: None
    rclpy.serialization = rclpy_serialization
    sys.modules.setdefault("rclpy", rclpy)
    sys.modules.setdefault("rclpy.serialization", rclpy_serialization)

    rosidl_runtime_py = types.ModuleType("rosidl_runtime_py")
    rosidl_utilities = types.ModuleType("rosidl_runtime_py.utilities")
    rosidl_utilities.get_message = lambda *args, **kwargs: object
    rosidl_runtime_py.utilities = rosidl_utilities
    sys.modules.setdefault("rosidl_runtime_py", rosidl_runtime_py)
    sys.modules.setdefault("rosidl_runtime_py.utilities", rosidl_utilities)


def load_metric_module():
    install_ros_import_stubs()
    module_path = Path(__file__).with_name("eval_nav2_one_combination.py")
    spec = importlib.util.spec_from_file_location("eval_nav2_one_combination", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module specification: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(actual):
        raise AssertionError(f"{label}: non-finite result: {actual}")
    error = abs(actual - expected)
    if error > tolerance:
        raise AssertionError(
            f"{label}: actual={actual:.9f}, expected={expected:.9f}, "
            f"absolute_error={error:.9f}, tolerance={tolerance:.9f}"
        )


def main() -> int:
    metrics = load_metric_module()

    # Constant longitudinal jerk: v(t) = 0.5 * j * t^2.
    expected_jerk = 0.40
    t_exec = np.linspace(0.0, 100.0, 2001)
    v_exec = 0.5 * expected_jerk * t_exec**2

    # Constant Ackermann steering rate: delta(t) = delta_0 + rate * t.
    wheelbase_m = 6.804
    expected_steering_rate = 0.04
    t_cmd = np.linspace(0.0, 10.0, 1001)
    vx_cmd = np.full_like(t_cmd, 5.0)
    delta_cmd = 0.05 + expected_steering_rate * t_cmd
    wz_cmd = vx_cmd * np.tan(delta_cmd) / wheelbase_m

    result = metrics.compute_speed_and_smoothness(
        exec_t_s=t_exec,
        exec_v=v_exec,
        eval_end_idx=t_exec.size - 1,
        target_speed_mps=0.0,
        cmd={
            "t_s": t_cmd,
            "vx": vx_cmd,
            "wz": wz_cmd,
        },
        eval_end_time_s=float(t_cmd[-1]),
        wheelbase_m=wheelbase_m,
        steering_min_speed_mps=0.50,
    )

    jerk = float(result["rms_jx_mps3"])
    steering_rate = float(result["rms_dotdelta_radps"])

    # Moving-average boundary effects are intentionally included, so the
    # regression tolerance is small but non-zero.
    assert_close(jerk, expected_jerk, 0.02, "longitudinal jerk RMS")
    assert_close(steering_rate, expected_steering_rate, 0.002, "steering-rate RMS")

    print(f"Longitudinal jerk test: PASS ({jerk:.6f} m/s^3)")
    print(f"Steering-rate test: PASS ({steering_rate:.6f} rad/s)")
    print("Corrected smoothness metric tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
