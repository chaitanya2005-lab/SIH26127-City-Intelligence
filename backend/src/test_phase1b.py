"""
Phase 1B Unit Test Suite (SIH26127)
------------------------------------
Tests for Anomaly Radar Engine:
1. Normal valid transition -> no anomaly
2. Unexpected transition (cam_01 -> cam_09) -> UNEXPECTED_ROUTE anomaly
3. Impossible time/speed transition (< 2s between nodes) -> IMPOSSIBLE_TRANSITION anomaly
4. Valid transition with different OCR strings but already-fused identity -> no false anomaly
5. Multiple anomalies -> each reported independently
6. No trajectory data -> graceful empty result
7. Missing optional evidence fields -> no crash
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Ensure src path is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anomaly_radar import AnomalyRadarEngine, DEFAULT_CAMERA_TOPOLOGY
from analytics import TrafficAnalyticsEngine


class TestPhase1BAnomalyRadar(unittest.TestCase):
    def setUp(self):
        self.radar = AnomalyRadarEngine()

    def test_1_normal_valid_transition(self):
        """Test that a standard valid transition along the topology graph generates no anomalies."""
        now = datetime.now()
        trajectory = {
            "MH12DE1408": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=5)).isoformat()},
                    {"camera_id": "cam_02", "timestamp": (now - timedelta(minutes=3)).isoformat()},
                    {"camera_id": "cam_04", "timestamp": (now - timedelta(minutes=1)).isoformat()},
                ]
            }
        }
        report = self.radar.run_anomaly_radar(trajectory)
        self.assertEqual(report["anomalies_detected"], 0, "Normal route along topology should have 0 anomalies.")

    def test_2_unexpected_route_anomaly(self):
        """Test that transitioning to a non-adjacent node (cam_01 -> cam_09) flags an UNEXPECTED_ROUTE anomaly."""
        now = datetime.now()
        trajectory = {
            "KA01AB2026": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=5)).isoformat()},
                    {"camera_id": "cam_09", "timestamp": (now - timedelta(minutes=3)).isoformat()},
                ]
            }
        }
        report = self.radar.run_anomaly_radar(trajectory)
        self.assertEqual(report["anomalies_detected"], 1, "Off-graph transition should flag exactly 1 anomaly.")

        anom = report["anomaly_records"][0]
        self.assertEqual(anom["anomaly_type"], "UNEXPECTED_ROUTE")
        self.assertEqual(anom["camera_from"], "cam_01")
        self.assertEqual(anom["camera_to"], "cam_09")
        self.assertGreaterEqual(anom["anomaly_score"], 80)
        self.assertEqual(anom["severity"], "HIGH")
        self.assertTrue(anom["recommended_review"])
        self.assertIn("reason", anom)
        self.assertIn("evidence", anom)

    def test_3_impossible_transition_anomaly(self):
        """Test that a 0.1s transition between distinct nodes flags an IMPOSSIBLE_TRANSITION anomaly."""
        now = datetime.now()
        trajectory = {
            "DL03XY9988": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": now.isoformat()},
                    {"camera_id": "cam_02", "timestamp": (now + timedelta(seconds=0.1)).isoformat()},
                ]
            }
        }
        report = self.radar.run_anomaly_radar(trajectory)
        self.assertEqual(report["anomalies_detected"], 1)

        anom = report["anomaly_records"][0]
        self.assertEqual(anom["anomaly_type"], "IMPOSSIBLE_TRANSITION")
        self.assertGreaterEqual(anom["anomaly_score"], 90)
        self.assertEqual(anom["severity"], "CRITICAL")
        self.assertIn("Teleportation", anom["reason"])

    def test_4_fused_identity_different_ocr_no_false_anomaly(self):
        """Test that fused identity with minor OCR variations (MH12DE1408 vs MH12DE14O8) does not generate false anomalies."""
        now = datetime.now()
        trajectory = {
            "MH12DE1408": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=5)).isoformat()},
                    {"camera_id": "cam_02", "timestamp": (now - timedelta(minutes=3)).isoformat()},
                ]
            }
        }
        fused_evidence = [
            {
                "fused_plate_number": "MH12DE1408",
                "fused_identity_confidence": 0.91,
                "is_valid_match": True,
                "evidence": {
                    "source_camera": "cam_01",
                    "destination_camera": "cam_02",
                    "timestamps": {
                        "source": (now - timedelta(minutes=5)).isoformat(),
                        "destination": (now - timedelta(minutes=3)).isoformat(),
                        "time_diff_seconds": 120.0,
                    },
                    "original_ocr_values": {
                        "source": "MH12DE1408",
                        "destination": "MH12DE14O8",
                    },
                },
            }
        ]
        report = self.radar.run_anomaly_radar(trajectory, fused_evidence)
        self.assertEqual(report["anomalies_detected"], 0, "Valid fused identity with plausible transition should not trigger false anomaly.")

    def test_5_multiple_independent_anomalies(self):
        """Test that multiple anomalous behaviors across vehicles are detected independently."""
        now = datetime.now()
        trajectories = {
            "VEH_A": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=10)).isoformat()},
                    {"camera_id": "cam_09", "timestamp": (now - timedelta(minutes=8)).isoformat()},  # UNEXPECTED_ROUTE
                ]
            },
            "VEH_B": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": now.isoformat()},
                    {"camera_id": "cam_02", "timestamp": (now + timedelta(seconds=0.2)).isoformat()},  # IMPOSSIBLE_TRANSITION
                ]
            },
        }
        report = self.radar.run_anomaly_radar(trajectories)
        self.assertEqual(report["anomalies_detected"], 2, "Multiple anomalies across distinct vehicles should be reported independently.")

        types = set(a["anomaly_type"] for a in report["anomaly_records"])
        self.assertIn("UNEXPECTED_ROUTE", types)
        self.assertIn("IMPOSSIBLE_TRANSITION", types)

    def test_6_empty_trajectory_graceful_handling(self):
        """Test that empty or missing trajectory dict returns a clean, zero-anomaly report without error."""
        report = self.radar.run_anomaly_radar({})
        self.assertEqual(report["anomalies_detected"], 0)
        self.assertEqual(report["total_trajectories_scanned"], 0)
        self.assertEqual(len(report["anomaly_records"]), 0)

    def test_7_missing_optional_fields_no_crash(self):
        """Test robust handling when node dicts are missing timestamps, camera_id, or bounding boxes."""
        malformed_trajectory = {
            "MALFORMED_01": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01"},  # missing timestamp
                    {"timestamp": "2026-09-14T21:40:00"},  # missing camera_id
                    {},  # empty node
                ]
            }
        }
        try:
            report = self.radar.run_anomaly_radar(malformed_trajectory)
            self.assertIsInstance(report, dict)
            self.assertEqual(report["anomalies_detected"], 0)
        except Exception as e:
            self.fail(f"Anomaly radar crashed on malformed node data: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
