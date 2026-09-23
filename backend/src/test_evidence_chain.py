"""
Phase 1D Unit Test Suite (SIH26127)
------------------------------------
Tests for Evidence Chain & Explainable Alert Intelligence Engine:
1. Normal trajectory -> no false alert evidence
2. IMPOSSIBLE_TRANSITION -> correct evidence structure & 5 Ws
3. UNEXPECTED_ROUTE -> correct evidence structure & 5 Ws
4. TEMPORAL_ANOMALY -> correct evidence structure & 5 Ws
5. IDENTITY_CONFLICT -> correct evidence structure & 5 Ws
6. Missing optional fields -> graceful handling (null / "Unavailable")
7. Prediction evidence marked PREDICTED with disclaimer
8. Existing anomaly output remains backward compatible
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Ensure src path is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evidence_chain import EvidenceChainEngine
from anomaly_radar import AnomalyRadarEngine
from route_predictor import RoutePredictorEngine
from analytics import TrafficAnalyticsEngine, get_mock_multi_camera_data


class TestPhase1DEvidenceChain(unittest.TestCase):
    def setUp(self):
        self.evidence_engine = EvidenceChainEngine()
        self.radar = AnomalyRadarEngine()
        self.predictor = RoutePredictorEngine()

    def test_1_normal_trajectory_no_false_evidence(self):
        """Test that a normal trajectory produces 0 anomaly evidence records."""
        now = datetime.now()
        trajectory = {
            "NORMAL_VEH": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=5)).isoformat()},
                    {"camera_id": "cam_02", "timestamp": (now - timedelta(minutes=3)).isoformat()},
                ]
            }
        }
        anom_report = self.radar.run_anomaly_radar(trajectory)
        chain = self.evidence_engine.build_evidence_chain(anom_report, trajectory)
        self.assertEqual(len(chain), 0, "Normal valid trajectory should produce 0 evidence chain alerts.")

    def test_2_impossible_transition_evidence(self):
        """Test evidence record structure for IMPOSSIBLE_TRANSITION anomaly."""
        now = datetime.now()
        trajectory = {
            "TELEPORT_VEH": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": now.isoformat()},
                    {"camera_id": "cam_02", "timestamp": (now + timedelta(seconds=0.1)).isoformat()},
                ]
            }
        }
        anom_report = self.radar.run_anomaly_radar(trajectory)
        chain = self.evidence_engine.build_evidence_chain(anom_report, trajectory)

        self.assertEqual(len(chain), 1)
        rec = chain[0]
        self.assertEqual(rec["anomaly_type"], "IMPOSSIBLE_TRANSITION")
        self.assertEqual(rec["severity"], "CRITICAL")
        self.assertIn("Teleportation", rec["explanation"]["primary_reason"])

        # Check 5 Ws
        five_ws = rec["explanation"]["five_w_summary"]
        self.assertIn("what", five_ws)
        self.assertIn("where", five_ws)
        self.assertIn("when", five_ws)
        self.assertIn("why", five_ws)
        self.assertIn("review_guidance", five_ws)

    def test_3_unexpected_route_evidence(self):
        """Test evidence record structure for UNEXPECTED_ROUTE anomaly."""
        now = datetime.now()
        trajectory = {
            "OFF_ROUTE_VEH": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=5)).isoformat()},
                    {"camera_id": "cam_09", "timestamp": (now - timedelta(minutes=3)).isoformat()},
                ]
            }
        }
        anom_report = self.radar.run_anomaly_radar(trajectory)
        chain = self.evidence_engine.build_evidence_chain(anom_report, trajectory)

        self.assertEqual(len(chain), 1)
        rec = chain[0]
        self.assertEqual(rec["anomaly_type"], "UNEXPECTED_ROUTE")
        self.assertEqual(rec["source"]["camera_id"], "cam_01")
        self.assertEqual(rec["destination"]["camera_id"], "cam_09")
        self.assertIn("configured camera transition graph", rec["explanation"]["primary_reason"])

    def test_4_temporal_anomaly_evidence(self):
        """Test evidence record structure for TEMPORAL_ANOMALY."""
        now = datetime.now()
        trajectory = {
            "DELAY_VEH": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": (now - timedelta(hours=2)).isoformat()},
                    {"camera_id": "cam_02", "timestamp": now.isoformat()},
                ]
            }
        }
        anom_report = self.radar.run_anomaly_radar(trajectory)
        chain = self.evidence_engine.build_evidence_chain(anom_report, trajectory)

        self.assertEqual(len(chain), 1)
        rec = chain[0]
        self.assertEqual(rec["anomaly_type"], "TEMPORAL_ANOMALY")
        self.assertIn("exceeds valid travel window", rec["explanation"]["primary_reason"])

    def test_5_identity_conflict_evidence(self):
        """Test evidence record structure for IDENTITY_CONFLICT."""
        now = datetime.now()
        trajectory = {
            "CLONED_VEH": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": now.isoformat()},
                ]
            }
        }
        fused_evidence = [
            {
                "fused_plate_number": "CLONED_VEH",
                "fused_identity_confidence": 0.95,
                "is_valid_match": True,
                "evidence": {
                    "source_camera": "cam_01",
                    "destination_camera": "cam_04",
                    "timestamps": {
                        "source": now.isoformat(),
                        "destination": (now + timedelta(seconds=1.0)).isoformat(),
                        "time_diff_seconds": 1.0,
                    },
                    "original_ocr_values": {
                        "source": "CLONED_VEH",
                        "destination": "CLONED_VEH",
                    },
                },
            }
        ]
        anom_report = self.radar.run_anomaly_radar(trajectory, fused_evidence)
        chain = self.evidence_engine.build_evidence_chain(anom_report, trajectory)

        self.assertEqual(len(chain), 1)
        rec = chain[0]
        self.assertEqual(rec["anomaly_type"], "IDENTITY_CONFLICT")
        self.assertEqual(rec["severity"], "CRITICAL")

    def test_6_missing_optional_fields_graceful(self):
        """Test that missing or malformed inputs produce graceful 'Unavailable' / null fields without crashing."""
        anom_item = {
            "fused_identity": None,
            "anomaly_type": None,
            "severity": None,
            "anomaly_score": None,
        }
        rec = self.evidence_engine.build_evidence_record(anom_item)

        self.assertIsNotNone(rec["evidence_id"])
        self.assertEqual(rec["source"]["camera_id"], "Unavailable")
        self.assertEqual(rec["destination"]["camera_id"], "Unavailable")
        self.assertEqual(rec["source"]["timestamp"], "Unavailable")
        self.assertIsNone(rec["trajectory_evidence"]["time_gap_seconds"])

    def test_7_prediction_evidence_marked_predicted(self):
        """Test that prediction evidence is explicitly marked as status PREDICTED with disclaimer."""
        now = datetime.now()
        trajectory = {
            "OFF_ROUTE_VEH": {
                "trajectory_nodes": [
                    {"camera_id": "cam_01", "timestamp": (now - timedelta(minutes=5)).isoformat()},
                    {"camera_id": "cam_09", "timestamp": (now - timedelta(minutes=3)).isoformat()},
                ]
            }
        }
        anom_report = self.radar.run_anomaly_radar(trajectory)
        pred_report = self.predictor.run_route_predictions(trajectory)

        chain = self.evidence_engine.build_evidence_chain(anom_report, trajectory, pred_report)
        self.assertEqual(len(chain), 1)
        rec = chain[0]

        pred_ev = rec["prediction_evidence"]
        self.assertEqual(pred_ev["status"], "PREDICTED")
        self.assertIn("probabilistic", pred_ev["disclaimer"].lower())

    def test_8_existing_anomaly_and_analytics_backward_compatible(self):
        """Test that Phase 1A, 1B, and 1C analytics output remain backward compatible when evidence chain is added."""
        engine = TrafficAnalyticsEngine()
        mock_data = get_mock_multi_camera_data()
        report = engine.generate_analytics_report(mock_data)

        self.assertIn("summary", report)
        self.assertIn("camera_analytics", report)
        self.assertIn("vehicle_classification", report)
        self.assertIn("fused_identities", report)
        self.assertIn("anomaly_radar", report)
        self.assertIn("predictive_route_intelligence", report)
        self.assertIn("evidence_chain", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
