"""
Phase 5 Test Suite: Security & Forensic Investigation Command Center (SIH26127)
--------------------------------------------------------------------------------
Validates Phase 5 Security & Forensic Command Center logic across 5 core requirements:
1. Forensic timeline assembly (linking observed camera sightings & predicted next node)
2. Privacy role isolation (masking for PUBLIC_OPERATOR vs unmasking for LAW_ENFORCEMENT)
3. Zero-fabrication audit (non-existent vehicle query handling without synthetic data)
4. Evidence chain 5 Ws completeness (what, where, when, why, review_guidance)
5. Command Center security summary aggregation
"""

import sys
import os
import unittest
from datetime import datetime

# Add backend/src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from privacy_security import PrivacySecurityEngine
from evidence_chain import EvidenceChainEngine
from route_predictor import RoutePredictorEngine
from anomaly_radar import AnomalyRadarEngine
from analytics import TrafficAnalyticsEngine, get_mock_multi_camera_data


class TestPhase5SecurityForensics(unittest.TestCase):
    def setUp(self):
        self.privacy_engine = PrivacySecurityEngine(salt="TEST_SALT_PHASE5", retention_days=30)
        self.evidence_engine = EvidenceChainEngine()
        self.predictor_engine = RoutePredictorEngine()
        self.anomaly_engine = AnomalyRadarEngine()
        self.analytics_engine = TrafficAnalyticsEngine()

        self.mock_records = get_mock_multi_camera_data()
        self.report = self.analytics_engine.generate_analytics_report(self.mock_records)

    def test_1_forensic_timeline_assembly(self):
        """1. Forensic timeline assembly (linking observed camera sightings & predicted next node)."""
        trajectories = self.report.get("reconstructed_trajectories", {})
        plate = "MH12DE1408"
        self.assertIn(plate, trajectories)

        traj_info = trajectories[plate]
        nodes = traj_info.get("trajectory_nodes", [])
        seq = traj_info.get("camera_sequence", [])

        # Get route prediction
        freq = self.predictor_engine.compute_historical_frequencies(trajectories)
        pred_res = self.predictor_engine.predict_next_camera(plate, nodes, freq)

        # Assemble unified timeline
        timeline = []
        for idx, n in enumerate(nodes, 1):
            timeline.append({
                "sighting_index": idx,
                "camera_id": n.get("camera_id"),
                "timestamp": n.get("timestamp"),
                "status": "OBSERVED",
                "confidence": n.get("confidence"),
                "ocr_confidence": n.get("ocr_confidence"),
            })

        if pred_res.get("predicted_next_camera"):
            timeline.append({
                "sighting_index": len(nodes) + 1,
                "camera_id": pred_res["predicted_next_camera"],
                "timestamp": "ESTIMATED_FUTURE",
                "status": "PREDICTED",
                "probability_percentage": pred_res["predictions"][0]["probability_percentage"],
            })

        self.assertGreaterEqual(len(timeline), 2)
        self.assertEqual(timeline[0]["status"], "OBSERVED")
        self.assertEqual(timeline[-1]["status"], "PREDICTED")
        self.assertEqual(timeline[-1]["camera_id"], pred_res["predicted_next_camera"])

    def test_2_privacy_role_isolation(self):
        """2. Privacy role isolation (masking for PUBLIC_OPERATOR vs unmasking for LAW_ENFORCEMENT)."""
        plate = "MH12DE1408"
        h_vid = self.privacy_engine.generate_hashed_vehicle_id(plate)
        masked_public = self.privacy_engine.mask_plate_number(plate, role="PUBLIC_OPERATOR")
        unmasked_le = self.privacy_engine.mask_plate_number(plate, role="LAW_ENFORCEMENT")

        self.assertTrue(h_vid.startswith("VID-"))
        self.assertEqual(masked_public, "MH12****08")
        self.assertNotIn("DE14", masked_public)
        self.assertEqual(unmasked_le, "MH12DE1408")

        # Test report wrapper privacy shielding
        wrapped_public = self.privacy_engine.apply_privacy_wrapper(self.report, role="PUBLIC_OPERATOR")
        trajs = wrapped_public.get("reconstructed_trajectories", {})
        self.assertIn("MH12****08", trajs)
        self.assertNotIn("MH12DE1408", trajs)

    def test_3_zero_fabrication_audit(self):
        """3. Zero-fabrication audit (non-existent vehicle query handling without synthetic data)."""
        non_existent_plate = "UNKNOWN9999"
        trajectories = self.report.get("reconstructed_trajectories", {})

        self.assertNotIn(non_existent_plate, trajectories)

        # Query route prediction for non-existent plate
        freq = self.predictor_engine.compute_historical_frequencies(trajectories)
        pred_res = self.predictor_engine.predict_next_camera(non_existent_plate, [], freq)

        self.assertEqual(pred_res["status"], "NO_TRAJECTORY")
        self.assertIsNone(pred_res["predicted_next_camera"])
        self.assertEqual(len(pred_res["predictions"]), 0)

    def test_4_evidence_chain_5ws_completeness(self):
        """4. Evidence chain 5 Ws completeness (what, where, when, why, review_guidance)."""
        chain = self.report.get("evidence_chain", [])

        # If no anomalies in mock, run anomaly engine with forced impossible transition
        if not chain:
            bad_trajectory = {
                "TEST_EVIL_01": {
                    "camera_sequence": ["cam_01", "cam_04"],
                    "total_sightings": 2,
                    "trajectory_nodes": [
                        {"camera_id": "cam_01", "timestamp": "2026-09-15T10:00:00.000000"},
                        {"camera_id": "cam_04", "timestamp": "2026-09-15T10:00:01.000000"},  # 1 sec gap (impossible)
                    ]
                }
            }
            anom_res = self.anomaly_engine.run_anomaly_radar(bad_trajectory)
            chain = self.evidence_engine.build_evidence_chain(anom_res, bad_trajectory)

        self.assertGreater(len(chain), 0)
        ev_item = chain[0]
        five_ws = ev_item.get("explanation", {}).get("five_w_summary", {})

        self.assertIn("what", five_ws)
        self.assertIn("where", five_ws)
        self.assertIn("when", five_ws)
        self.assertIn("why", five_ws)
        self.assertIn("review_guidance", five_ws)

        self.assertTrue(len(five_ws["what"]) > 0)
        self.assertTrue(len(five_ws["where"]) > 0)
        self.assertTrue(len(five_ws["why"]) > 0)

    def test_5_command_center_summary_aggregation(self):
        """5. Command Center security summary aggregation."""
        sec_summary = self.privacy_engine.get_security_status_summary()

        self.assertEqual(sec_summary["privacy_shield_status"], "ONLINE")
        self.assertIn("DPDP Act 2023", sec_summary["compliance_standard"])
        self.assertEqual(sec_summary["active_role"], "PUBLIC_OPERATOR")
        self.assertGreaterEqual(sec_summary["total_audit_logs_recorded"], 1)

        # Log an action and verify count increment
        self.privacy_engine.log_audit_event(
            actor="TEST_OFFICER",
            role="LAW_ENFORCEMENT",
            action="UNMASK_PLATE",
            resource_id="VID-TEST1234",
            reason="Phase 5 Forensic Audit Test",
            target_plate="MH12DE1408",
        )

        sec_summary_updated = self.privacy_engine.get_security_status_summary()
        self.assertEqual(sec_summary_updated["total_audit_logs_recorded"], sec_summary["total_audit_logs_recorded"] + 1)


if __name__ == "__main__":
    unittest.main()
