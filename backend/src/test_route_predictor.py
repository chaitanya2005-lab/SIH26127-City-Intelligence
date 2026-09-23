"""
Phase 1C Unit Test Suite (SIH26127)
------------------------------------
Tests for Predictive Route Intelligence Engine:
1. Normal known transition
2. Multiple candidate next cameras with ranked probabilities
3. Unknown camera node handling
4. Single-camera trajectory prediction
5. Probability normalization (sum approx 1.0)
6. Empty trajectory graceful handling
7. Prediction confidence calculation
8. Existing trajectory & anomaly functionality remains unaffected
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Ensure src path is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_predictor import RoutePredictorEngine
from anomaly_radar import AnomalyRadarEngine
from analytics import TrafficAnalyticsEngine, get_mock_multi_camera_data


class TestPhase1CRoutePredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = RoutePredictorEngine()

    def test_1_normal_known_transition(self):
        """Test prediction for a vehicle with a known trajectory."""
        now = datetime.now()
        trajectory_nodes = [
            {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=5)).isoformat()},
            {"camera_id": "cam_02", "timestamp": (now - timedelta(minutes=2)).isoformat()},
        ]
        res = self.predictor.predict_next_camera("MH12DE1408", trajectory_nodes)

        self.assertEqual(res["status"], "PREDICTED")
        self.assertEqual(res["current_camera"], "cam_02")
        self.assertIsNotNone(res["predicted_next_camera"])
        self.assertGreater(len(res["predictions"]), 0)
        self.assertEqual(res["predictions"][0]["camera_id"], "cam_01" if "cam_01" in ["cam_01", "cam_04"] else res["predicted_next_camera"])

    def test_2_multiple_candidate_next_cameras(self):
        """Test ranking and candidate breakdown for cam_01 (candidates cam_02 and cam_03)."""
        now = datetime.now()
        trajectory_nodes = [
            {"camera_id": "cam_01", "timestamp": now.isoformat()},
        ]
        hist_freq = {
            "cam_01": {"cam_02": 5, "cam_03": 2}
        }
        res = self.predictor.predict_next_camera("MH12DE1408", trajectory_nodes, hist_freq)

        self.assertEqual(res["status"], "PREDICTED")
        self.assertEqual(res["current_camera"], "cam_01")
        self.assertEqual(res["predicted_next_camera"], "cam_02")
        self.assertEqual(len(res["predictions"]), 2)
        self.assertGreater(res["predictions"][0]["probability"], res["predictions"][1]["probability"])

    def test_3_unknown_camera_node(self):
        """Test handling of an unknown/unconfigured camera node."""
        now = datetime.now()
        trajectory_nodes = [
            {"camera_id": "cam_99", "timestamp": now.isoformat()},
        ]
        res = self.predictor.predict_next_camera("UNKNOWN_VEH", trajectory_nodes)

        self.assertEqual(res["status"], "TERMINAL_NODE")
        self.assertEqual(res["current_camera"], "cam_99")
        self.assertEqual(len(res["predictions"]), 0)
        self.assertEqual(res["prediction_confidence"], 0.0)

    def test_4_single_camera_trajectory(self):
        """Test prediction generation for a single sighting vehicle."""
        now = datetime.now()
        trajectory_nodes = [
            {"camera_id": "cam_01", "timestamp": now.isoformat()},
        ]
        res = self.predictor.predict_next_camera("SINGLE_CAM_VEH", trajectory_nodes)

        self.assertEqual(res["status"], "PREDICTED")
        self.assertEqual(res["current_camera"], "cam_01")
        self.assertGreater(len(res["predictions"]), 0)
        self.assertIn("reason", res["predictions"][0])

    def test_5_probability_normalization(self):
        """Test that candidate probabilities sum exactly to 1.0 (100%)."""
        now = datetime.now()
        trajectory_nodes = [
            {"camera_id": "cam_02", "timestamp": now.isoformat()},
        ]
        hist_freq = {
            "cam_02": {"cam_01": 3, "cam_04": 7}
        }
        res = self.predictor.predict_next_camera("PROB_VEH", trajectory_nodes, hist_freq)

        probs = [p["probability"] for p in res["predictions"]]
        total_prob = sum(probs)
        self.assertAlmostEqual(total_prob, 1.0, places=2, msg="Probabilities must sum to ~1.0.")

    def test_6_empty_trajectory_graceful(self):
        """Test that empty trajectory data returns clean NO_TRAJECTORY status without crashing."""
        res = self.predictor.predict_next_camera("EMPTY_VEH", [])
        self.assertEqual(res["status"], "NO_TRAJECTORY")
        self.assertIsNone(res["current_camera"])
        self.assertEqual(res["prediction_confidence"], 0.0)

    def test_7_prediction_confidence_calculation(self):
        """Test prediction confidence metric calculation."""
        now = datetime.now()
        trajectory_nodes = [
            {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=5)).isoformat()},
            {"camera_id": "cam_02", "timestamp": (now - timedelta(minutes=2)).isoformat()},
        ]
        res = self.predictor.predict_next_camera("CONF_VEH", trajectory_nodes)
        self.assertGreater(res["prediction_confidence"], 0.0)
        self.assertLessEqual(res["prediction_confidence"], 1.0)
        self.assertIn("status", res)

    def test_8_existing_functionality_unaffected(self):
        """Test that Phase 1A & Phase 1B engines run completely unaffected alongside RoutePredictorEngine."""
        engine = TrafficAnalyticsEngine()
        mock_data = get_mock_multi_camera_data()
        report = engine.generate_analytics_report(mock_data)

        self.assertIn("summary", report)
        self.assertIn("camera_analytics", report)
        self.assertIn("vehicle_classification", report)
        self.assertIn("fused_identities", report)
        self.assertIn("anomaly_radar", report)

        # Run RoutePredictorEngine on report trajectories
        predictor = RoutePredictorEngine()
        pred_report = predictor.run_route_predictions(report["reconstructed_trajectories"])
        self.assertIn("predictions", pred_report)
        self.assertEqual(len(pred_report["predictions"]), len(report["reconstructed_trajectories"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
