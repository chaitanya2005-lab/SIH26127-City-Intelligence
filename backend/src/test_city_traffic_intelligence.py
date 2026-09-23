"""
Phase 2A Test Suite: City-Wide Traffic Intelligence & OD Matrix (SIH26127)
-------------------------------------------------------------------------
Validates the CityTrafficIntelligenceEngine against 12 test scenarios:
1. Empty trajectory dataset
2. Single-camera trajectory
3. Normal two-camera transition
4. Multi-camera trajectory
5. Repeated detections at same camera
6. Multiple vehicles sharing same corridor
7. OD matrix dimensions
8. Flow-count correctness
9. Corridor ranking
10. Percentage normalization
11. Missing camera IDs
12. Backward compatibility of analytics output
"""

import sys
import os
import unittest
from datetime import datetime

# Add backend/src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from city_traffic_intelligence import CityTrafficIntelligenceEngine
from analytics import TrafficAnalyticsEngine


class TestPhase2ACityTrafficIntelligence(unittest.TestCase):
    def setUp(self):
        self.engine = CityTrafficIntelligenceEngine(default_camera_nodes=["cam_01", "cam_02", "cam_03", "cam_04"])

    def test_1_empty_trajectory_dataset(self):
        """1. Empty trajectory dataset handling."""
        res = self.engine.compute_city_traffic_intelligence({})
        self.assertIn("summary", res)
        self.assertEqual(res["summary"]["total_inter_camera_transitions"], 0)
        self.assertEqual(res["summary"]["unique_vehicles"], 0)
        self.assertEqual(len(res["top_corridors"]), 0)

    def test_2_single_camera_trajectory(self):
        """2. Single-camera trajectory produces 0 inter-camera OD transitions."""
        trajectories = {
            "MH12DE1408": {
                "camera_sequence": ["cam_01"],
                "total_sightings": 1,
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": "2026-09-15T10:00:00"}
                ]
            }
        }
        res = self.engine.compute_city_traffic_intelligence(trajectories)
        self.assertEqual(res["od_matrix"]["total_transitions"], 0)
        self.assertEqual(len(res["top_corridors"]), 0)

    def test_3_normal_two_camera_transition(self):
        """3. Normal two-camera transition (cam_01 -> cam_02)."""
        trajectories = {
            "MH12DE1408": {
                "camera_sequence": ["cam_01", "cam_02"],
                "total_sightings": 2,
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": "2026-09-15T10:00:00"},
                    {"camera_id": "cam_02", "timestamp": "2026-09-15T10:05:00"}
                ]
            }
        }
        res = self.engine.compute_city_traffic_intelligence(trajectories)
        self.assertEqual(res["od_matrix"]["total_transitions"], 1)
        self.assertEqual(res["od_matrix"]["matrix"]["cam_01"]["cam_02"], 1)
        self.assertEqual(len(res["top_corridors"]), 1)
        self.assertEqual(res["top_corridors"][0]["origin"], "cam_01")
        self.assertEqual(res["top_corridors"][0]["destination"], "cam_02")

    def test_4_multi_camera_trajectory(self):
        """4. Multi-camera trajectory (cam_01 -> cam_02 -> cam_03)."""
        trajectories = {
            "KA01AB2026": {
                "camera_sequence": ["cam_01", "cam_02", "cam_03"],
                "total_sightings": 3,
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": "2026-09-15T10:00:00"},
                    {"camera_id": "cam_02", "timestamp": "2026-09-15T10:05:00"},
                    {"camera_id": "cam_03", "timestamp": "2026-09-15T10:10:00"}
                ]
            }
        }
        res = self.engine.compute_city_traffic_intelligence(trajectories)
        self.assertEqual(res["od_matrix"]["total_transitions"], 2)
        self.assertEqual(res["od_matrix"]["matrix"]["cam_01"]["cam_02"], 1)
        self.assertEqual(res["od_matrix"]["matrix"]["cam_02"]["cam_03"], 1)
        self.assertEqual(len(res["top_corridors"]), 2)

    def test_5_repeated_detections_same_camera(self):
        """5. Repeated detections at same camera are ignored as self-loops."""
        trajectories = {
            "DL03XY9988": {
                "camera_sequence": ["cam_01", "cam_01", "cam_01"],
                "total_sightings": 3,
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": "2026-09-15T10:00:00"},
                    {"camera_id": "cam_01", "timestamp": "2026-09-15T10:01:00"},
                    {"camera_id": "cam_01", "timestamp": "2026-09-15T10:02:00"}
                ]
            }
        }
        res = self.engine.compute_city_traffic_intelligence(trajectories)
        self.assertEqual(res["od_matrix"]["total_transitions"], 0)
        self.assertEqual(res["od_matrix"]["matrix"]["cam_01"]["cam_01"], 0)

    def test_6_multiple_vehicles_sharing_corridor(self):
        """6. Multiple vehicles sharing same corridor accumulate count."""
        trajectories = {
            "V1": {"trajectory_nodes": [{"camera_id": "cam_01", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
            "V2": {"trajectory_nodes": [{"camera_id": "cam_01", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
            "V3": {"trajectory_nodes": [{"camera_id": "cam_01", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
        }
        res = self.engine.compute_city_traffic_intelligence(trajectories)
        self.assertEqual(res["od_matrix"]["matrix"]["cam_01"]["cam_02"], 3)
        self.assertEqual(res["top_corridors"][0]["vehicle_count"], 3)

    def test_7_od_matrix_dimensions(self):
        """7. OD matrix dimensions match active camera nodes list."""
        cams = ["cam_01", "cam_02", "cam_03", "cam_04"]
        od = self.engine.build_od_matrix({}, camera_nodes=cams)
        self.assertEqual(len(od["camera_nodes"]), 4)
        self.assertEqual(len(od["matrix"]), 4)
        for r in cams:
            self.assertEqual(len(od["matrix"][r]), 4)

    def test_8_flow_count_correctness(self):
        """8. Incoming and outgoing camera flow counts match transitions."""
        trajectories = {
            "V1": {"trajectory_nodes": [{"camera_id": "cam_01", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
            "V2": {"trajectory_nodes": [{"camera_id": "cam_03", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
        }
        flows = self.engine.compute_camera_flow_summary(trajectories)
        self.assertEqual(flows["cam_02"]["incoming_flow"], 2)
        self.assertEqual(flows["cam_01"]["outgoing_flow"], 1)
        self.assertEqual(flows["cam_03"]["outgoing_flow"], 1)

    def test_9_corridor_ranking(self):
        """9. Corridors are properly ranked by vehicle count descending."""
        trajectories = {
            "V1": {"trajectory_nodes": [{"camera_id": "cam_01", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
            "V2": {"trajectory_nodes": [{"camera_id": "cam_01", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
            "V3": {"trajectory_nodes": [{"camera_id": "cam_02", "timestamp": "T1"}, {"camera_id": "cam_03", "timestamp": "T2"}]},
        }
        res = self.engine.compute_city_traffic_intelligence(trajectories)
        top = res["top_corridors"]
        self.assertTrue(len(top) >= 2)
        self.assertEqual(top[0]["corridor_id"], "cam_01 -> cam_02")
        self.assertEqual(top[0]["vehicle_count"], 2)
        self.assertEqual(top[1]["corridor_id"], "cam_02 -> cam_03")
        self.assertEqual(top[1]["vehicle_count"], 1)

    def test_10_percentage_normalization(self):
        """10. Corridor flow percentages sum up to 100% of inter-camera flow."""
        trajectories = {
            "V1": {"trajectory_nodes": [{"camera_id": "cam_01", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
            "V2": {"trajectory_nodes": [{"camera_id": "cam_01", "timestamp": "T1"}, {"camera_id": "cam_02", "timestamp": "T2"}]},
            "V3": {"trajectory_nodes": [{"camera_id": "cam_02", "timestamp": "T1"}, {"camera_id": "cam_03", "timestamp": "T2"}]},
            "V4": {"trajectory_nodes": [{"camera_id": "cam_03", "timestamp": "T1"}, {"camera_id": "cam_04", "timestamp": "T2"}]},
        }
        res = self.engine.compute_city_traffic_intelligence(trajectories)
        top = res["top_corridors"]
        total_pct = sum(c["percentage_of_total_flow"] for c in top)
        self.assertAlmostEqual(total_pct, 100.0, delta=0.5)

    def test_11_missing_camera_ids_graceful(self):
        """11. Missing camera IDs or malformed nodes are handled gracefully."""
        trajectories = {
            "V_ERR": {
                "trajectory_nodes": [
                    {"timestamp": "T1"},  # missing camera_id
                    {"camera_id": None, "timestamp": "T2"},
                    {"camera_id": "cam_01", "timestamp": "T3"},
                ]
            }
        }
        res = self.engine.compute_city_traffic_intelligence(trajectories)
        self.assertIn("summary", res)
        self.assertEqual(res["od_matrix"]["total_transitions"], 0)

    def test_12_backward_compatibility_analytics(self):
        """12. Backward compatibility of TrafficAnalyticsEngine report output."""
        analytics = TrafficAnalyticsEngine()
        mock_data = [
            {"camera_id": "cam_01", "plate_number": "MH12DE1408", "vehicle_class": "car", "timestamp": "2026-09-15T10:00:00", "confidence": 0.9},
            {"camera_id": "cam_02", "plate_number": "MH12DE1408", "vehicle_class": "car", "timestamp": "2026-09-15T10:05:00", "confidence": 0.95},
        ]
        report = analytics.generate_analytics_report(mock_data)

        # Check existing fields remain untouched
        self.assertIn("timestamp", report)
        self.assertIn("summary", report)
        self.assertIn("camera_analytics", report)
        self.assertIn("vehicle_classification", report)
        self.assertIn("fused_identities", report)
        self.assertIn("anomaly_radar", report)
        self.assertIn("predictive_route_intelligence", report)
        self.assertIn("evidence_chain", report)

        # Check new additive field
        self.assertIn("city_traffic_intelligence", report)
        ci = report["city_traffic_intelligence"]
        self.assertIn("od_matrix", ci)
        self.assertIn("top_corridors", ci)
        self.assertIn("traffic_insights", ci)


if __name__ == "__main__":
    unittest.main()
