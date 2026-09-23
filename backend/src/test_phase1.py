"""
Phase 1A Unit Test Suite (SIH26127)
------------------------------------
Tests for Confidence-Aware ANPR & Vehicle Identity Fusion:
1. Exact plate match
2. OCR typo fuzzy match (O<->0, I<->1, B<->8, S<->5, Z<->2)
3. Unrelated plate rejection
4. Impossible camera transition rejection
5. Low OCR confidence classification & handling
6. Multi-camera identity fusion & evidence chain validation
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Ensure src path is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anpr import (
    fuzzy_plate_similarity,
    classify_ocr_confidence,
    canonicalize_plate,
    VehicleIdentityFusion,
)
from analytics import TrafficAnalyticsEngine, get_mock_multi_camera_data


class TestPhase1AConfidenceANPRAndFusion(unittest.TestCase):
    def setUp(self):
        self.fusion_engine = VehicleIdentityFusion()

    def test_1_exact_plate_match(self):
        """Test exact plate string similarity."""
        sim = fuzzy_plate_similarity("MH12DE1408", "MH12DE1408")
        self.assertEqual(sim, 1.0, "Exact plate match should yield 1.0 similarity score.")

    def test_2_ocr_typo_fuzzy_match(self):
        """Test OCR typo tolerance for common confusion pairs (O<->0, I<->1, B<->8, S<->5, Z<->2)."""
        pairs = [
            ("MH12DE1408", "MH12DE14O8"),  # 0 <-> O
            ("KA01AB2026", "KAOIAB2026"),  # 1 <-> I, 0 <-> O
            ("DL08XY9988", "DL0BXY9988"),  # 8 <-> B
            ("TN05CC5544", "TN0SCC5544"),  # 5 <-> S
            ("GJ02ZZ1234", "GJ02221234"),  # Z <-> 2
        ]
        for p1, p2 in pairs:
            sim = fuzzy_plate_similarity(p1, p2)
            self.assertGreaterEqual(
                sim,
                0.90,
                f"OCR typo pair ('{p1}', '{p2}') failed fuzzy similarity test (score: {sim})."
            )

    def test_3_unrelated_plate_rejection(self):
        """Test that completely distinct plates yield low similarity and are rejected."""
        sim = fuzzy_plate_similarity("MH12DE1408", "KA01AB2026")
        self.assertLess(sim, 0.40, f"Unrelated plates should have low similarity score (got {sim}).")

        obs1 = {
            "camera_id": "cam_01",
            "tracking_id": 1,
            "plate_number": "MH12DE1408",
            "ocr_confidence": 0.91,
            "timestamp": "2026-09-13T19:46:34",
        }
        obs2 = {
            "camera_id": "cam_02",
            "tracking_id": 2,
            "plate_number": "KA01AB2026",
            "ocr_confidence": 0.88,
            "timestamp": "2026-09-13T19:48:34",
        }
        res = self.fusion_engine.fuse_identities(obs1, obs2)
        self.assertFalse(res["is_valid_match"], "Unrelated plates must be rejected from identity fusion.")

    def test_4_impossible_camera_transition_rejection(self):
        """Test rejection of physically implausible camera transitions (teleportation & negative time delta)."""
        now = datetime.now()

        # Teleportation: 0.1 seconds between cam_01 and cam_02
        obs1 = {
            "camera_id": "cam_01",
            "tracking_id": 1,
            "plate_number": "MH12DE1408",
            "ocr_confidence": 0.91,
            "timestamp": now.isoformat(),
        }
        obs_teleport = {
            "camera_id": "cam_02",
            "tracking_id": 1,
            "plate_number": "MH12DE1408",
            "ocr_confidence": 0.94,
            "timestamp": (now + timedelta(seconds=0.1)).isoformat(),
        }
        res_teleport = self.fusion_engine.fuse_identities(obs1, obs_teleport)
        self.assertFalse(
            res_teleport["is_valid_match"],
            "Teleportation transition (< 2s) must be rejected."
        )
        self.assertEqual(res_teleport["fused_identity_confidence"], 0.0)

        # Reverse chronological sequence: negative time delta
        obs_reverse = {
            "camera_id": "cam_02",
            "tracking_id": 1,
            "plate_number": "MH12DE1408",
            "ocr_confidence": 0.94,
            "timestamp": (now - timedelta(minutes=5)).isoformat(),
        }
        res_reverse = self.fusion_engine.fuse_identities(obs1, obs_reverse)
        self.assertFalse(
            res_reverse["is_valid_match"],
            "Negative time delta transition must be rejected."
        )
        self.assertEqual(res_reverse["fused_identity_confidence"], 0.0)

    def test_5_low_ocr_confidence_classification(self):
        """Test classification of high, medium, and low OCR confidence values."""
        self.assertEqual(classify_ocr_confidence(0.92), "HIGH CONFIDENCE")
        self.assertEqual(classify_ocr_confidence(0.80), "HIGH CONFIDENCE")
        self.assertEqual(classify_ocr_confidence(0.75), "MEDIUM CONFIDENCE")
        self.assertEqual(classify_ocr_confidence(0.50), "MEDIUM CONFIDENCE")
        self.assertEqual(classify_ocr_confidence(0.35), "LOW CONFIDENCE")
        self.assertEqual(classify_ocr_confidence(0.0), "LOW CONFIDENCE")

    def test_6_multi_camera_identity_fusion_and_evidence(self):
        """Test identity fusion between cam_01 (MH12DE1408, conf 0.91) and cam_02 (MH12DE14O8, conf 0.76)."""
        now = datetime.now()
        obs_cam1 = {
            "camera_id": "cam_01",
            "tracking_id": 1,
            "raw_ocr_text": "MH12DE1408",
            "normalized_plate_text": "MH12DE1408",
            "plate_number": "MH12DE1408",
            "ocr_confidence": 0.91,
            "timestamp": (now - timedelta(minutes=5)).isoformat(),
        }
        obs_cam2 = {
            "camera_id": "cam_02",
            "tracking_id": 1,
            "raw_ocr_text": "MH12DE14O8",
            "normalized_plate_text": "MH12DE14O8",
            "plate_number": "MH12DE14O8",
            "ocr_confidence": 0.76,
            "timestamp": (now - timedelta(minutes=3)).isoformat(),
        }

        res = self.fusion_engine.fuse_identities(obs_cam1, obs_cam2)

        self.assertTrue(res["is_valid_match"], "Matching observations with minor OCR typo and valid transition must fuse.")
        self.assertGreaterEqual(res["fused_identity_confidence"], 0.85, "Fused confidence score should be high.")
        self.assertEqual(res["fused_plate_number"], "MH12DE1408", "Primary plate should select higher confidence observation.")

        ev = res["evidence"]
        self.assertEqual(ev["source_camera"], "cam_01")
        self.assertEqual(ev["destination_camera"], "cam_02")
        self.assertEqual(ev["original_ocr_values"]["source"], "MH12DE1408")
        self.assertEqual(ev["original_ocr_values"]["destination"], "MH12DE14O8")
        self.assertIn("time_diff_seconds", ev["timestamps"])
        self.assertGreater(ev["similarity_score"], 0.90)

    def test_7_analytics_integration(self):
        """Test integration of fused identities into TrafficAnalyticsEngine output."""
        engine = TrafficAnalyticsEngine()
        mock_data = get_mock_multi_camera_data()
        report = engine.generate_analytics_report(mock_data)

        self.assertIn("summary", report)
        self.assertIn("camera_analytics", report)
        self.assertIn("vehicle_classification", report)
        self.assertIn("reconstructed_trajectories", report)
        self.assertIn("fused_identities", report)

        fused = report["fused_identities"]
        self.assertGreaterEqual(len(fused), 1, "Analytics report should contain fused identity matches.")
        self.assertEqual(fused[0]["fused_plate_number"], "MH12DE1408")


if __name__ == "__main__":
    unittest.main(verbosity=2)
