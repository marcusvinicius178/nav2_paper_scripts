#!/usr/bin/env python3
"""ROS-independent tests for field_audit_common_v3."""

from __future__ import annotations

import math
import logging
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "field"))

from field_audit_common_v3 import (  # noqa: E402
    BASELINE_FALLBACK_FOOTPRINT,
    BagMetricData,
    BagSpec,
    GridSample,
    OdomSample,
    choose_common_time_domain,
    choose_occupancy_thresholds,
    deterministic_pose_perturbations,
    minimum_clearance,
    pad_convex_polygon,
    pair_grids_to_nearest_poses,
    summarize_sync_pairs,
    transform_polygon,
)
from recompute_field_metrics_from_bags_v3 import process_bag  # noqa: E402


def odom(storage: float, header: float | None, x: float = 0.0) -> OdomSample:
    covariance = np.zeros(36, dtype=float)
    covariance[0] = 0.04
    covariance[7] = 0.09
    covariance[35] = math.radians(2.0) ** 2
    return OdomSample(
        storage_stamp_sec=storage,
        header_stamp_sec=header,
        frame_id="map",
        child_frame_id="base_footprint",
        x=x,
        y=0.0,
        yaw=0.0,
        vx=1.0,
        vy=0.0,
        vz=0.0,
        pose_covariance=covariance,
        twist_covariance=covariance.copy(),
    )


def grid(storage: float, header: float | None) -> GridSample:
    return GridSample(
        storage_stamp_sec=storage,
        header_stamp_sec=header,
        frame_id="map",
        resolution=1.0,
        width=2,
        height=2,
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
        data=np.array([[0, 100], [0, -1]], dtype=np.int16),
    )


class TimestampTests(unittest.TestCase):
    def test_header_domain_when_valid_and_overlapping(self) -> None:
        poses = [odom(100.0, 10.0), odom(101.0, 11.0)]
        grids = [grid(200.0, 10.1), grid(201.0, 10.9)]
        domain, reason = choose_common_time_domain(poses, grids, minimum_header_fraction=0.95)
        self.assertEqual(domain, "header")
        self.assertIn("overlapping", reason)

    def test_storage_domain_when_header_missing(self) -> None:
        poses = [odom(100.0, None), odom(101.0, 11.0)]
        grids = [grid(100.1, 10.1), grid(100.9, 10.9)]
        domain, reason = choose_common_time_domain(poses, grids, minimum_header_fraction=0.95)
        self.assertEqual(domain, "storage")
        self.assertIn("availability", reason)

    def test_nearest_pair_signed_delta_and_tolerance(self) -> None:
        poses = [odom(10.00, None), odom(10.20, None)]
        grids = [grid(10.08, None), grid(11.00, None)]
        pairs = pair_grids_to_nearest_poses(poses, grids, "storage", maximum_delta_s=0.25)
        self.assertEqual(len(pairs), 2)
        self.assertAlmostEqual(pairs[0].delta_signed_s, -0.08)
        self.assertAlmostEqual(pairs[0].delta_abs_s, 0.08)
        self.assertTrue(pairs[0].accepted)
        self.assertFalse(pairs[1].accepted)
        summary = summarize_sync_pairs(pairs, [0.05, 0.10, 0.25, 1.00])
        self.assertEqual(summary["n_candidate_pairs"], 2)
        self.assertEqual(summary["n_pairs_within_max_tolerance"], 1)
        self.assertEqual(summary["fraction_within_0p10s"], 0.5)


class GeometryTests(unittest.TestCase):
    def test_baseline_fallback_dimensions(self) -> None:
        self.assertTrue(np.allclose(np.ptp(BASELINE_FALLBACK_FOOTPRINT, axis=0), [9.625, 3.498]))

    def test_cell_area_not_cell_center(self) -> None:
        footprint = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        result = minimum_clearance(
            footprint,
            np.array([0.0, 0.0]),
            np.array([[4.0, 0.0]]),
            cell_resolution=2.0,
            grid_yaw=0.0,
            threshold=50,
        )
        self.assertAlmostEqual(result.clearance_area_m, 2.0)
        self.assertAlmostEqual(result.clearance_center_m, 3.0)
        self.assertAlmostEqual(result.base_to_center_m, 4.0)

    def test_oriented_footprint_rotation(self) -> None:
        footprint = np.array([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]])
        rotated = transform_polygon(footprint, 3.0, 4.0, math.pi / 2.0)
        self.assertTrue(np.allclose(np.mean(rotated, axis=0), [3.0, 4.0]))
        self.assertTrue(np.allclose(np.ptp(rotated, axis=0), [2.0, 4.0]))

    def test_exact_convex_padding(self) -> None:
        rectangle = np.array([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]])
        padded = pad_convex_polygon(rectangle, 0.1)
        self.assertTrue(np.allclose(np.ptp(padded, axis=0), [4.2, 2.2]))

    def test_thresholds_come_from_recorded_values(self) -> None:
        self.assertEqual(choose_occupancy_thresholds([-1, 0, 50, 100], 50), [50, 100])
        self.assertEqual(choose_occupancy_thresholds([-1, 0, 25, 75, 100], 50), [25, 50, 75, 100])


class CovarianceTests(unittest.TestCase):
    def test_deterministic_sigma_envelope(self) -> None:
        sample = odom(1.0, 1.0)
        perturbations, note = deterministic_pose_perturbations(sample, 2.0)
        self.assertGreater(len(perturbations), 1)
        self.assertIn("deterministic", note)
        self.assertTrue(any(not math.isclose(item[0], sample.x) for item in perturbations))
        self.assertTrue(any(not math.isclose(item[2], sample.yaw) for item in perturbations))


class SyntheticPipelineTests(unittest.TestCase):
    def test_process_bag_end_to_end_without_ros(self) -> None:
        synthetic_grid = np.zeros((4, 8), dtype=np.int16)
        synthetic_grid[2, 6] = 100  # center=(4.5, 0.5), cell left edge x=4.0
        bag = BagMetricData(
            spec=BagSpec("synthetic", "Synthetic", "/tmp/synthetic.mcap"),
            topics={"odometry": "/odom", "obstacle_grid": "/grid"},
            topic_types={},
            odometry=[odom(10.0, None)],
            grids=[
                GridSample(
                    storage_stamp_sec=10.08,
                    header_stamp_sec=None,
                    frame_id="map",
                    resolution=1.0,
                    width=8,
                    height=4,
                    origin_x=-2.0,
                    origin_y=-2.0,
                    origin_yaw=0.0,
                    data=synthetic_grid,
                )
            ],
            observed_occupancy_values={0, 100},
        )
        bag.odometry[0].vx = 3.0
        config = {
            "synchronization": {
                "tolerances_s": [0.05, 0.10, 1.0],
                "maximum_delta_s": 1.0,
                "minimum_header_fraction": 0.95,
            },
            "clearance": {
                "nominal_occupancy_threshold": 50,
                "maximum_footprint_time_delta_s": 1.0,
                "max_occupied_cells_per_grid": 100,
                "footprint_polygon": [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            },
            "sensitivity": {"covariance_sigma_levels": [1.0, 2.0]},
        }
        logger = logging.getLogger("synthetic-field-audit-test")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        result = process_bag(bag, config, ROOT, logger)

        self.assertEqual(result["summary"]["status"], "completed")
        self.assertAlmostEqual(result["critical"][0]["clearance_area_m"], 3.0)
        self.assertAlmostEqual(result["critical"][0]["delta_signed_pose_minus_grid_s"], -0.08)
        self.assertAlmostEqual(result["critical"][0]["velocity_planar_kmh"], 10.8)
        self.assertEqual(result["critical"][0]["velocity_source"].split(".")[0], "/odom")
        variant_types = {row["variant_type"] for row in result["sensitivity"]}
        self.assertIn("pose_grid_tolerance", variant_types)
        self.assertIn("occupancy_threshold", variant_types)
        self.assertIn("pose_covariance_deterministic_envelope", variant_types)


if __name__ == "__main__":
    unittest.main()
