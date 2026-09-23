"""
Phase 3 Test Suite: AI City Copilot Natural-Language Intent Engine (SIH26127)
----------------------------------------------------------------------------------
Validates CityCopilotEngine against 13 core requirements:
1. Trajectory sequence query ("seen at CAM_01 and later at CAM_02")
2. Vehicle route history query ("Show route of MH12DE1408")
3. Traffic congestion status query ("Which camera has highest congestion")
4. Predictive route query ("Predicted next camera for MH12DE1408")
5. Anomaly radar query ("Which vehicles have anomalous routes")
6. Evidence chain query ("Show evidence chain for alert")
7. OD corridor query ("Highest traffic corridor")
8. What-If simulation query ("What happens if CAM_03 traffic increases by 40%")
9. Unknown vehicle query handling
10. Unknown camera / empty dataset handling
11. Insufficient data handling for unsupported concepts
12. Explicit PREDICTED flag validation
13. Malformed / empty query handling
"""

import sys
import os
import unittest

# Add backend/src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from city_copilot import CityCopilotEngine
from traffic_simulator import TrafficSimulatorEngine


class TestPhase3CityCopilot(unittest.TestCase):
    def setUp(self):
        self.simulator = TrafficSimulatorEngine()
        self.copilot = CityCopilotEngine(simulator_engine=self.simulator)

        self.mock_report = {
            "summary": {"total_detections": 10, "unique_vehicles": 4},
            "camera_analytics": {
                "cam_01": {"traffic_volume": 10, "unique_vehicles": 10, "congestion_level": "MODERATE"},
                "cam_02": {"traffic_volume": 5, "unique_vehicles": 5, "congestion_level": "LOW"},
                "cam_03": {"traffic_volume": 2, "unique_vehicles": 2, "congestion_level": "LOW"},
            },
            "reconstructed_trajectories": {
                "MH12DE1408": {
                    "fused_identity": "MH12DE1408",
                    "total_sightings": 2,
                    "camera_sequence": ["cam_01", "cam_02"],
                    "last_seen": "2026-09-15T12:00:00",
                },
                "TN09CC5544": {
                    "fused_identity": "TN09CC5544",
                    "total_sightings": 2,
                    "camera_sequence": ["cam_01", "cam_02"],
                    "last_seen": "2026-09-15T12:05:00",
                },
                "KA01AB2026": {
                    "fused_identity": "KA01AB2026",
                    "total_sightings": 1,
                    "camera_sequence": ["cam_01"],
                    "last_seen": "2026-09-15T12:10:00",
                },
            },
            "predictive_route_intelligence": {
                "predictions": {
                    "MH12DE1408": {
                        "current_camera": "cam_02",
                        "predicted_next_camera": "cam_04",
                        "prediction_probability": 0.75,
                        "prediction_confidence": 0.88,
                        "predictions": [
                            {"camera_id": "cam_04", "probability_percentage": "75.0%"},
                            {"camera_id": "cam_01", "probability_percentage": "25.0%"},
                        ],
                    }
                }
            },
            "anomalies": [
                {
                    "anomaly_id": "ANOM-001",
                    "fused_identity": "MH12DE1408",
                    "anomaly_type": "IMPOSSIBLE_TRANSITION",
                    "severity": "CRITICAL",
                    "anomaly_score": 95,
                    "camera_pair": "cam_01 ➔ cam_04",
                }
            ],
            "evidence_chain": [
                {
                    "evidence_id": "EV-ANOM-001",
                    "fused_identity": "MH12DE1408",
                    "severity": "CRITICAL",
                    "anomaly_score": 95,
                    "explanation": {
                        "primary_reason": "Impossible travel time delta (0.2s)",
                        "five_w_summary": {
                            "what": "Physically implausible camera transition",
                            "where": "cam_01 -> cam_04",
                            "when": "12:00:00",
                            "review_guidance": "Verify speed telemetry",
                        },
                        "supporting_factors": ["Speed exceeds 300 km/h"],
                    },
                }
            ],
            "city_traffic_intelligence": {
                "top_corridors": [
                    {"corridor": "cam_01 -> cam_02", "transitions": 5, "percentage_of_total": 62.5}
                ],
                "od_matrix": {"total_transitions": 8},
            },
            "city_digital_twin": {
                "city_traffic_state": "MODERATE",
                "city_average_congestion_score": 42,
                "camera_nodes_intelligence": {
                    "cam_01": {"congestion_score": 64, "traffic_state": "CONGESTED", "traffic_volume": 10},
                    "cam_02": {"congestion_score": 28, "traffic_state": "FREE_FLOW", "traffic_volume": 5},
                },
            },
        }

    def test_1_trajectory_sequence_query(self):
        """1. Query vehicles seen at CAM_01 and later at CAM_02."""
        res = self.copilot.process_query("Which vehicles were seen at CAM_01 and later at CAM_02?", self.mock_report)
        self.assertEqual(res["intent"], "TRAJECTORY_SEQUENCE")
        self.assertFalse(res["predicted"])
        self.assertIn("MH12DE1408", res["answer"])
        self.assertIn("TN09CC5544", res["answer"])
        self.assertEqual(res["confidence"], "HIGH")

    def test_2_vehicle_route_history_query(self):
        """2. Query vehicle route history for MH12DE1408."""
        res = self.copilot.process_query("Show the route of MH12DE1408.", self.mock_report)
        self.assertEqual(res["intent"], "VEHICLE_ROUTE_HISTORY")
        self.assertFalse(res["predicted"])
        self.assertIn("cam_01 ➔ cam_02", res["answer"])

    def test_3_traffic_congestion_status_query(self):
        """3. Query camera with highest congestion."""
        res = self.copilot.process_query("Which camera currently has the highest congestion?", self.mock_report)
        self.assertEqual(res["intent"], "TRAFFIC_CONGESTION_STATUS")
        self.assertFalse(res["predicted"])
        self.assertIn("CAM_01", res["answer"])
        self.assertIn("64/100", res["answer"])

    def test_4_predictive_route_query(self):
        """4. Query predicted next camera for vehicle."""
        res = self.copilot.process_query("What is the predicted next camera for MH12DE1408?", self.mock_report)
        self.assertEqual(res["intent"], "PREDICTIVE_ROUTE")
        self.assertTrue(res["predicted"])
        self.assertIn("CAM_04", res["answer"])
        self.assertIn("75.0%", res["answer"])

    def test_5_anomaly_radar_query(self):
        """5. Query anomalous vehicle routes."""
        res = self.copilot.process_query("Which vehicles have anomalous routes?", self.mock_report)
        self.assertEqual(res["intent"], "ANOMALY_DETECTION")
        self.assertFalse(res["predicted"])
        self.assertIn("MH12DE1408", res["answer"])
        self.assertIn("IMPOSSIBLE_TRANSITION", res["answer"])

    def test_6_evidence_chain_query(self):
        """6. Query evidence chain details."""
        res = self.copilot.process_query("Show the evidence chain for MH12DE1408.", self.mock_report)
        self.assertEqual(res["intent"], "EVIDENCE_CHAIN")
        self.assertFalse(res["predicted"])
        self.assertIn("EV-ANOM-001", res["answer"])
        self.assertIn("WHAT:", res["answer"])

    def test_7_od_corridor_query(self):
        """7. Query highest traffic origin-destination corridor."""
        res = self.copilot.process_query("Which origin-destination corridor has the highest traffic?", self.mock_report)
        self.assertEqual(res["intent"], "OD_CORRIDOR_INTELLIGENCE")
        self.assertFalse(res["predicted"])
        self.assertIn("cam_01 -> cam_02", res["answer"])
        self.assertIn("62.5%", res["answer"])

    def test_8_what_if_simulation_query(self):
        """8. Query What-If traffic simulation."""
        res = self.copilot.process_query("What happens if CAM_01 traffic increases by 40%?", self.mock_report)
        self.assertEqual(res["intent"], "WHAT_IF_SIMULATION")
        self.assertTrue(res["predicted"])
        self.assertIn("CAM_01", res["answer"])
        self.assertIn("PREDICTED WHAT-IF ESTIMATE", res["answer"])

    def test_9_unknown_vehicle_handling(self):
        """9. Unknown vehicle returns safe missing error message."""
        res = self.copilot.process_query("Show the route of XYZ9999.", self.mock_report)
        self.assertIn("Insufficient data available", res["answer"])
        self.assertIn("XYZ9999", res["answer"])
        self.assertEqual(res["confidence"], "0.0")

    def test_10_empty_dataset_handling(self):
        """10. Query on empty dataset handles safely without crashing."""
        res = self.copilot.process_query("Which camera has the highest congestion?", {})
        self.assertEqual(res["confidence"], "0.0")
        self.assertIn("Insufficient data available", res["answer"])

    def test_11_insufficient_data_unsupported_concept(self):
        """11. Unsupported concepts return explicit insufficient data response."""
        res = self.copilot.process_query("What is the weather forecast for tomorrow at CAM_01?", self.mock_report)
        self.assertEqual(res["intent"], "INSUFFICIENT_DATA")
        self.assertIn("Insufficient data available", res["answer"])
        self.assertEqual(res["confidence"], "0.0")

    def test_12_explicit_predicted_badge_validation(self):
        """12. Validate that predictive and simulation queries strictly set predicted=True."""
        res_pred = self.copilot.process_query("Show route predictions", self.mock_report)
        res_sim = self.copilot.process_query("Simulate a 20% increase at CAM_01", self.mock_report)
        res_traj = self.copilot.process_query("Show route of MH12DE1408", self.mock_report)

        self.assertTrue(res_pred["predicted"])
        self.assertTrue(res_sim["predicted"])
        self.assertFalse(res_traj["predicted"])

    def test_13_malformed_empty_query_handling(self):
        """13. Empty or non-string query returns safe error structure."""
        res1 = self.copilot.process_query("", self.mock_report)
        res2 = self.copilot.process_query("   ", self.mock_report)
        self.assertEqual(res1["intent"], "UNKNOWN")
        self.assertEqual(res2["intent"], "UNKNOWN")
        self.assertEqual(res1["confidence"], "0.0")


if __name__ == "__main__":
    unittest.main()
