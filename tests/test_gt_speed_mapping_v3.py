#!/usr/bin/env python3
"""Regression tests for benchmark-speed versus human-source-speed mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "human_reference"))

import build_gt_final_table_v3 as module  # noqa: E402


class SpeedMappingTests(unittest.TestCase):
    def test_all_required_relations(self) -> None:
        expected = {20: 15, 25: 20, 30: 15, 35: 20}
        observed = {
            benchmark: module.human_source_for_benchmark(benchmark)
            for benchmark in expected
        }
        self.assertEqual(observed, expected)

    def test_nominal_reverse_mapping(self) -> None:
        self.assertEqual(module.nominal_benchmark_for_human_source(15), 20)
        self.assertEqual(module.nominal_benchmark_for_human_source(20), 25)

    def test_fields_remain_distinct(self) -> None:
        converted = module.attach_speed_semantics(
            [
                {
                    "scenario_id": 1,
                    "scenario_name": "Straight",
                    "speed_kmh": 15,
                    "aligned_rmse_m": 1.25,
                }
            ]
        )[0]
        self.assertEqual(converted["benchmark_speed_kmh"], 20)
        self.assertEqual(converted["human_source_speed_kmh"], 15)
        self.assertNotIn("speed_kmh", converted)
        self.assertEqual(converted["aligned_rmse_m"], 1.25)


if __name__ == "__main__":
    unittest.main()

