"""
Phase 4 Test Suite: Privacy-Preserving ANPR & Security Audit Engine (SIH26127)
-------------------------------------------------------------------------------
Validates PrivacySecurityEngine against 12 security and privacy requirements:
1. Salted SHA-256 vehicle ID generation
2. Plate masking for PUBLIC_OPERATOR role
3. Plate unmasking for LAW_ENFORCEMENT role
4. Plate unmasking for SYSTEM_ADMIN role
5. Role-based access control (RBAC) elevation & rejection
6. Tamper-evident audit log creation
7. Retention policy compliance check (active record)
8. Retention policy compliance check (expired record)
9. Non-breaking privacy wrapper for analytics reports
10. Security status summary formatting
11. Non-destruction of internal plaintext plate data
12. Empty / malformed inputs handling
"""

import sys
import os
import unittest
from datetime import datetime, timedelta

# Add backend/src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from privacy_security import PrivacySecurityEngine


class TestPhase4PrivacySecurity(unittest.TestCase):
    def setUp(self):
        self.engine = PrivacySecurityEngine(salt="TEST_SALT_2026", retention_days=30)
        self.mock_report = {
            "summary": {"total_events": 5},
            "reconstructed_trajectories": {
                "MH12DE1408": {"camera_sequence": ["cam_01", "cam_02"], "total_sightings": 2},
                "TN09CC5544": {"camera_sequence": ["cam_01"], "total_sightings": 1},
            },
        }

    def test_1_sha256_hashed_vehicle_id_generation(self):
        """1. Salted SHA-256 vehicle ID generation (e.g. VID-a4f9b8c2)."""
        vid1 = self.engine.generate_hashed_vehicle_id("MH12DE1408")
        vid2 = self.engine.generate_hashed_vehicle_id("MH12DE1408")
        vid3 = self.engine.generate_hashed_vehicle_id("KA01AB2026")

        self.assertTrue(vid1.startswith("VID-"))
        self.assertEqual(len(vid1), 14)
        self.assertEqual(vid1, vid2)  # Deterministic repeatability
        self.assertNotEqual(vid1, vid3)  # Distinct hash for distinct plate

    def test_2_plate_masking_public_operator(self):
        """2. Plate masking for PUBLIC_OPERATOR role."""
        masked = self.engine.mask_plate_number("MH12DE1408", role="PUBLIC_OPERATOR")
        self.assertEqual(masked, "MH12****08")
        self.assertNotIn("DE14", masked)

    def test_3_plate_unmasking_law_enforcement(self):
        """3. Full unmasked plate for LAW_ENFORCEMENT role."""
        unmasked = self.engine.mask_plate_number("MH12DE1408", role="LAW_ENFORCEMENT")
        self.assertEqual(unmasked, "MH12DE1408")

    def test_4_plate_unmasking_system_admin(self):
        """4. Full unmasked plate for SYSTEM_ADMIN role."""
        unmasked = self.engine.mask_plate_number("MH12DE1408", role="SYSTEM_ADMIN")
        self.assertEqual(unmasked, "MH12DE1408")

    def test_5_rbac_role_elevation_and_rejection(self):
        """5. RBAC role update and rejection of invalid role strings."""
        res_ok = self.engine.set_access_role("LAW_ENFORCEMENT", actor="OFFICER_PATEL", reason="Incident audit")
        self.assertEqual(res_ok["status"], "SUCCESS")
        self.assertEqual(self.engine.current_role, "LAW_ENFORCEMENT")

        res_err = self.engine.set_access_role("INVALID_ROLE", actor="HACKER")
        self.assertEqual(res_err["status"], "ERROR")
        self.assertEqual(self.engine.current_role, "LAW_ENFORCEMENT")  # Role unchanged

    def test_6_tamper_evident_audit_log_creation(self):
        """6. Tamper-evident audit log creation."""
        entry = self.engine.log_audit_event(
            actor="OFFICER_PATEL",
            role="LAW_ENFORCEMENT",
            action="UNMASK_PLATE",
            resource_id="VID-a4f9b8c2",
            reason="Court warrant investigation",
            target_plate="MH12DE1408",
        )
        self.assertTrue(entry["audit_id"].startswith("AUD-"))
        self.assertEqual(entry["actor"], "OFFICER_PATEL")
        self.assertEqual(entry["target_plate_masked"], "MH12****08")
        self.assertIn(entry, self.engine.audit_logs)

    def test_7_retention_policy_active_record(self):
        """7. Retention policy check for recent active record."""
        now_iso = datetime.now().isoformat()
        res = self.engine.check_retention_policy(now_iso)
        self.assertTrue(res["compliant"])
        self.assertEqual(res["status"], "ACTIVE_RETENTION")

    def test_8_retention_policy_expired_record(self):
        """8. Retention policy check for expired record (> 30 days)."""
        old_iso = (datetime.now() - timedelta(days=45)).isoformat()
        res = self.engine.check_retention_policy(old_iso)
        self.assertFalse(res["compliant"])
        self.assertEqual(res["status"], "EXPIRED_RETENTION")

    def test_9_privacy_wrapper_non_breaking(self):
        """9. Non-breaking privacy wrapper for analytics report."""
        wrapped = self.engine.apply_privacy_wrapper(self.mock_report, role="PUBLIC_OPERATOR")
        self.assertIn("security_metadata", wrapped)
        trajs = wrapped["reconstructed_trajectories"]
        self.assertIn("MH12****08", trajs)
        self.assertIn("hashed_vehicle_id", trajs["MH12****08"])

    def test_10_security_status_summary_formatting(self):
        """10. Security status summary structure and DPDP compliance label."""
        summary = self.engine.get_security_status_summary()
        self.assertEqual(summary["privacy_shield_status"], "ONLINE")
        self.assertIn("DPDP Act 2023", summary["compliance_standard"])
        self.assertIn("active_role", summary)
        self.assertIn("total_audit_logs_recorded", summary)

    def test_11_non_destruction_of_internal_data(self):
        """11. Verify that internal plaintext data is preserved for Phase 1A-3 logic."""
        wrapped = self.engine.apply_privacy_wrapper(self.mock_report, role="PUBLIC_OPERATOR")
        # Ensure original self.mock_report was NOT mutated
        self.assertIn("MH12DE1408", self.mock_report["reconstructed_trajectories"])

    def test_12_empty_malformed_inputs_handling(self):
        """12. Safe handling of empty or malformed plate strings."""
        masked_empty = self.engine.mask_plate_number("", role="PUBLIC_OPERATOR")
        vid_empty = self.engine.generate_hashed_vehicle_id(None)
        self.assertEqual(masked_empty, "UNKNOWN_PLATE")
        self.assertEqual(vid_empty, "VID-00000000")


if __name__ == "__main__":
    unittest.main()
