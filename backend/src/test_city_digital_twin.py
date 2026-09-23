"""
Phase 2B Test Suite: City Digital Twin + Congestion Intelligence (SIH26127)
--------------------------------------------------------------------------
Validates CityDigitalTwinEngine against 10 test scenarios:
1. Empty data handling
2. Single camera handling
3. Multiple cameras handling
4. Free-flow classification
5. Congestion classification
6. Score normalization
7. Camera node intelligence output structure
8. Severe congestion classification
9. Missing optional fields graceful handling
10. Backward compatibility of analytics output
"""

import sys
import os
import unittest

# Add backend/src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from city_digital_twin import CityDigitalTwinEngine
from analytics import TrafficAnalyticsEngine


class TestPhase2BCityDigitalTwin(unittest.TestCase):
    def setUp(self):
        self.engine = CityDigitalTwinEngine(max_volume_threshold=15)

    def test_1_empty_data_handling(self):
        """1. Empty data handling."""
        res = self.engine.compute_digital_twin_summary({})
        self.assertIn("summary", res)
        self.assertIn("camera_nodes_intelligence", res)
        self.assertEqual(res["summary"]["city_traffic_state"], "FREE_FLOW")
        self.assertEqual(res["summary"]["average_city_congestion_score"], 8)  # default low 0.4*20=8

    def test_2_single_camera_handling(self):
        """2. Single camera node intelligence."""
        analytics = {
            "cam_01": {"traffic_volume": 3, "unique_vehicles": 3, "congestion_level": "LOW"}
        }
        res = self.engine.compute_digital_twin_summary(camera_analytics=analytics)
        self.assertIn("cam_01", res["camera_nodes_intelligence"])
        node = res["camera_nodes_intelligence"]["cam_01"]
        self.assertEqual(node["detection_volume"], 3)
        self.assertEqual(node["traffic_state"], "FREE_FLOW")

    def test_3_multiple_cameras_handling(self):
        """3. Multiple camera nodes intelligence."""
        analytics = {
            "cam_01": {"traffic_volume": 2, "unique_vehicles": 2, "congestion_level": "LOW"},
            "cam_02": {"traffic_volume": 12, "unique_vehicles": 10, "congestion_level": "MODERATE"},
        }
        res = self.engine.compute_digital_twin_summary(camera_analytics=analytics)
        self.assertIn("cam_01", res["camera_nodes_intelligence"])
        self.assertIn("cam_02", res["camera_nodes_intelligence"])
        self.assertEqual(res["summary"]["total_monitored_nodes"], 2)

    def test_4_free_flow_classification(self):
        """4. Free flow state classification (Score < 30)."""
        score = self.engine.compute_congestion_score(traffic_volume=1, congestion_level="LOW")
        state = self.engine.classify_traffic_state(score)
        self.assertTrue(score < 30)
        self.assertEqual(state, "FREE_FLOW")

    def test_5_congestion_classification(self):
        """5. Congestion state classification (60 <= Score < 85)."""
        score = self.engine.compute_congestion_score(traffic_volume=12, congestion_level="MODERATE")
        state = self.engine.classify_traffic_state(score)
        self.assertTrue(60 <= score < 85)
        self.assertEqual(state, "CONGESTED")

    def test_6_score_normalization_range(self):
        """6. Congestion score normalization stays strictly within 0-100 range."""
        score_min = self.engine.compute_congestion_score(traffic_volume=0, congestion_level="LOW")
        score_max = self.engine.compute_congestion_score(traffic_volume=100, congestion_level="HIGH")
        self.assertTrue(0 <= score_min <= 100)
        self.assertTrue(0 <= score_max <= 100)

    def test_7_camera_node_intelligence_structure(self):
        """7. Camera node intelligence object structure."""
        analytics = {"cam_01": {"traffic_volume": 5, "unique_vehicles": 5, "congestion_level": "LOW"}}
        intel = self.engine.compute_camera_nodes_intelligence(camera_analytics=analytics)
        self.assertIn("cam_01", intel)
        node = intel["cam_01"]
        self.assertIn("camera_id", node)
        self.assertIn("traffic_state", node)
        self.assertIn("congestion_score", node)
        self.assertIn("detection_volume", node)
        self.assertIn("unique_vehicle_count", node)
        self.assertIn("incoming_flow", node)
        self.assertIn("outgoing_flow", node)

    def test_8_severe_congestion_classification(self):
        """8. Severe congestion classification (Score >= 85)."""
        score = self.engine.compute_congestion_score(traffic_volume=20, congestion_level="HIGH")
        state = self.engine.classify_traffic_state(score)
        self.assertTrue(score >= 85)
        self.assertEqual(state, "SEVERE")

    def test_9_missing_optional_fields_graceful(self):
        """9. Missing optional fields or malformed inputs handled gracefully."""
        intel = self.engine.compute_camera_nodes_intelligence(camera_analytics={"cam_01": {}})
        node = intel["cam_01"]
        self.assertEqual(node["detection_volume"], 0)
        self.assertIn("traffic_state", node)

    def test_10_backward_compatibility_analytics_report(self):
        """10. Backward compatibility of TrafficAnalyticsEngine report output."""
        analytics = TrafficAnalyticsEngine()
        mock_data = [
            {"camera_id": "cam_01", "plate_number": "MH12DE1408", "vehicle_class": "car", "timestamp": "2026-09-15T10:00:00", "confidence": 0.9},
            {"camera_id": "cam_02", "plate_number": "MH12DE1408", "vehicle_class": "car", "timestamp": "2026-09-15T10:05:00", "confidence": 0.95},
        ]
        report = analytics.generate_analytics_report(mock_data)

        # Check existing fields
        self.assertIn("timestamp", report)
        self.assertIn("summary", report)
        self.assertIn("camera_analytics", report)
        self.assertIn("vehicle_classification", report)
        self.assertIn("fused_identities", report)
        self.assertIn("anomaly_radar", report)
        self.assertIn("predictive_route_intelligence", report)
        self.assertIn("evidence_chain", report)
        self.assertIn("city_traffic_intelligence", report)

        # Check Phase 2B additive field
        self.assertIn("city_digital_twin", report)
        dt = report["city_digital_twin"]
        self.assertIn("summary", dt)
        self.assertIn("camera_nodes_intelligence", dt)
        self.assertIn("scoring_formula_documentation", dt)


if __name__ == "__main__":
    unittest.main()
