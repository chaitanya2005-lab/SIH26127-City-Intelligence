"""
Phase 2C Test Suite: What-If Traffic Simulator + Decision Intelligence (SIH26127)
----------------------------------------------------------------------------------
Validates TrafficSimulatorEngine against 13 test scenarios:
1. Baseline scenario (0% change)
2. Positive traffic increase (+40%)
3. Negative traffic decrease (-30%)
4. Zero change scenario
5. Multiple affected downstream cameras
6. Isolated/terminal camera
7. Invalid camera ID error handling
8. Invalid percentage bounds handling
9. Empty dataset safety
10. Deterministic repeatability
11. Confirmation of ZERO source analytics data mutation
12. Impact classification thresholds
13. Decision recommendation generation
"""

import sys
import os
import unittest
import copy

# Add backend/src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from traffic_simulator import TrafficSimulatorEngine
from analytics import TrafficAnalyticsEngine


class TestPhase2CTrafficSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = TrafficSimulatorEngine()
        self.mock_report = {
            "camera_analytics": {
                "cam_01": {"traffic_volume": 10, "unique_vehicles": 10, "congestion_level": "MODERATE"},
                "cam_02": {"traffic_volume": 5, "unique_vehicles": 5, "congestion_level": "LOW"},
                "cam_03": {"traffic_volume": 2, "unique_vehicles": 2, "congestion_level": "LOW"},
                "cam_04": {"traffic_volume": 1, "unique_vehicles": 1, "congestion_level": "LOW"},
            },
            "city_digital_twin": {
                "camera_nodes_intelligence": {
                    "cam_01": {"congestion_score": 64, "traffic_state": "CONGESTED"},
                    "cam_02": {"congestion_score": 28, "traffic_state": "FREE_FLOW"},
                    "cam_03": {"congestion_score": 16, "traffic_state": "FREE_FLOW"},
                    "cam_04": {"congestion_score": 12, "traffic_state": "FREE_FLOW"},
                }
            },
            "city_traffic_intelligence": {
                "od_matrix": {
                    "matrix": {
                        "cam_01": {"cam_01": 0, "cam_02": 3, "cam_03": 1, "cam_04": 0}
                    }
                }
            },
        }

    def test_1_baseline_scenario_zero_change(self):
        """1. Baseline scenario with 0% traffic change."""
        res = self.simulator.run_what_if_simulation("cam_01", 0, self.mock_report)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["is_simulated"])
        self.assertEqual(res["target_camera_simulation"]["volume_change"], 0)
        self.assertEqual(res["target_camera_simulation"]["delta_score"], 0)

    def test_2_positive_traffic_increase(self):
        """2. Positive traffic increase scenario (+40%)."""
        res = self.simulator.run_what_if_simulation("cam_01", 40, self.mock_report)
        self.assertEqual(res["status"], "SUCCESS")
        target = res["target_camera_simulation"]
        self.assertEqual(target["baseline_volume"], 10)
        self.assertEqual(target["simulated_volume"], 14)
        self.assertEqual(target["volume_change"], 4)
        self.assertTrue(target["delta_score"] >= 0)

    def test_3_negative_traffic_decrease(self):
        """3. Negative traffic decrease scenario (-30%)."""
        res = self.simulator.run_what_if_simulation("cam_01", -30, self.mock_report)
        self.assertEqual(res["status"], "SUCCESS")
        target = res["target_camera_simulation"]
        self.assertEqual(target["simulated_volume"], 7)
        self.assertEqual(target["volume_change"], -3)
        self.assertTrue(target["delta_score"] <= 0)

    def test_4_zero_change_scenario(self):
        """4. Explicit zero percentage change returns baseline metrics."""
        res = self.simulator.run_what_if_simulation("cam_02", 0.0, self.mock_report)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["overall_impact_level"], "LOW")

    def test_5_multiple_affected_downstream_cameras(self):
        """5. Propagation affects downstream cameras (cam_02 and cam_03)."""
        res = self.simulator.run_what_if_simulation("cam_01", 100, self.mock_report)
        downstream = res["downstream_corridors_simulation"]
        self.assertTrue(len(downstream) >= 2)
        cams = [d["camera_id"] for d in downstream]
        self.assertIn("cam_02", cams)
        self.assertIn("cam_03", cams)

    def test_6_isolated_terminal_camera(self):
        """6. Terminal camera without configured downstream nodes."""
        engine = TrafficSimulatorEngine(topology={"cam_04": []})
        res = engine.run_what_if_simulation("cam_04", 50, self.mock_report)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(len(res["downstream_corridors_simulation"]), 0)

    def test_7_invalid_camera_id_error_handling(self):
        """7. Invalid camera ID returns safe error dictionary."""
        res = self.simulator.run_what_if_simulation("cam_99", 50, self.mock_report)
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("error_message", res)

    def test_8_invalid_percentage_bounds_error_handling(self):
        """8. Out-of-bounds percentage (< -100% or > +300%) returns safe error."""
        res1 = self.simulator.run_what_if_simulation("cam_01", 500, self.mock_report)
        res2 = self.simulator.run_what_if_simulation("cam_01", -200, self.mock_report)
        self.assertEqual(res1["status"], "ERROR")
        self.assertEqual(res2["status"], "ERROR")

    def test_9_empty_dataset_safety(self):
        """9. Simulation on empty report dataset handles safely."""
        res = self.simulator.run_what_if_simulation("cam_01", 20, {})
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("target_camera_simulation", res)

    def test_10_deterministic_repeatability(self):
        """10. Two identical simulation calls yield identical outputs."""
        res1 = self.simulator.run_what_if_simulation("cam_01", 50, self.mock_report)
        res2 = self.simulator.run_what_if_simulation("cam_01", 50, self.mock_report)
        self.assertEqual(res1["target_camera_simulation"], res2["target_camera_simulation"])
        self.assertEqual(res1["overall_impact_level"], res2["overall_impact_level"])

    def test_11_zero_mutation_of_source_data(self):
        """11. Source analytics report is NOT mutated by simulation execution."""
        report_copy = copy.deepcopy(self.mock_report)
        self.simulator.run_what_if_simulation("cam_01", 100, self.mock_report)
        self.assertEqual(self.mock_report, report_copy)

    def test_12_impact_classification_thresholds(self):
        """12. Impact classification thresholds (LOW, MODERATE, HIGH, CRITICAL)."""
        self.assertEqual(self.simulator.classify_impact_level(delta_score=5, sim_score=25), "LOW")
        self.assertEqual(self.simulator.classify_impact_level(delta_score=15, sim_score=65), "MODERATE")
        self.assertEqual(self.simulator.classify_impact_level(delta_score=30, sim_score=78), "HIGH")
        self.assertEqual(self.simulator.classify_impact_level(delta_score=45, sim_score=90), "CRITICAL")

    def test_13_decision_recommendations_generation(self):
        """13. Decision recommendation generation."""
        res = self.simulator.run_what_if_simulation("cam_01", 80, self.mock_report)
        recs = res["decision_recommendations"]
        self.assertTrue(len(recs) >= 1)
        self.assertIn("title", recs[0])
        self.assertIn("description", recs[0])
        self.assertIn("action_guidance", recs[0])


if __name__ == "__main__":
    unittest.main()
