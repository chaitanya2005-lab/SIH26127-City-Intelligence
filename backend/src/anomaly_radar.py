"""
Phase 1B — Anomaly Radar Module (SIH26127)
-------------------------------------------
Analyzes multi-camera vehicle trajectories and fused identity records against an expected
camera transition graph (topology) and spatial-temporal constraints to detect:

1. UNEXPECTED_ROUTE: Vehicle transitions to a camera node outside expected graph adjacency.
2. IMPOSSIBLE_TRANSITION: Travel speed or time delta implies physically impossible travel (teleportation).
3. TEMPORAL_ANOMALY: Time gap between camera sightings exceeds valid travel window or exhibits extreme delay.
4. IDENTITY_CONFLICT: Conflicting observations for the same fused identity at distinct locations simultaneously.

================================================================================
HOW TO RUN THIS MODULE:
================================================================================
python backend/src/anomaly_radar.py
================================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure module directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Default DEMO/SIMULATED camera topology network graph
DEFAULT_CAMERA_TOPOLOGY: Dict[str, List[str]] = {
    "cam_01": ["cam_02", "cam_03"],
    "cam_02": ["cam_01", "cam_04"],
    "cam_03": ["cam_01", "cam_04"],
    "cam_04": ["cam_02", "cam_03"],
}

# Distance matrix in kilometers between nodes (SIMULATED topology for demo)
DEFAULT_CAMERA_DISTANCES_KM: Dict[str, Dict[str, float]] = {
    "cam_01": {"cam_02": 1.5, "cam_03": 2.0},
    "cam_02": {"cam_01": 1.5, "cam_04": 3.0},
    "cam_03": {"cam_01": 2.0, "cam_04": 2.5},
    "cam_04": {"cam_02": 3.0, "cam_03": 2.5},
}


class AnomalyRadarEngine:
    def __init__(
        self,
        camera_topology: Optional[Dict[str, List[str]]] = None,
        camera_distances: Optional[Dict[str, Dict[str, float]]] = None,
        min_travel_time_sec: float = 5.0,
        max_travel_time_sec: float = 1800.0,
        max_physically_possible_speed_kmh: float = 180.0,
    ):
        """
        Initialize Anomaly Radar Engine.

        :param camera_topology: Map of allowed adjacency graph node -> List[allowed_next_nodes].
        :param camera_distances: Map of distances in km between camera nodes.
        :param min_travel_time_sec: Minimum travel time in seconds between distinct nodes.
        :param max_travel_time_sec: Maximum travel time in seconds for expected travel window.
        :param max_physically_possible_speed_kmh: Maximum physical speed threshold in km/h.
        """
        self.topology = camera_topology or DEFAULT_CAMERA_TOPOLOGY
        self.distances = camera_distances or DEFAULT_CAMERA_DISTANCES_KM
        self.min_travel_time_sec = min_travel_time_sec
        self.max_travel_time_sec = max_travel_time_sec
        self.max_speed_kmh = max_physically_possible_speed_kmh
        self.topology_source = "DEMO/SIMULATED camera topology"

    def compute_anomaly_severity(self, score: int) -> str:
        """Classify anomaly score (0-100) into severity level."""
        if score >= 90:
            return "CRITICAL"
        elif score >= 70:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def analyze_trajectory(
        self,
        plate_or_identity: str,
        trajectory_nodes: List[Dict[str, Any]],
        fused_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze a sequence of trajectory nodes for a single vehicle identity.
        """
        anomalies: List[Dict[str, Any]] = []
        # 1-3. Trajectory pairwise checks (requires at least 2 nodes)
        if trajectory_nodes and len(trajectory_nodes) >= 2:
            sorted_nodes = sorted(
                trajectory_nodes,
                key=lambda n: n.get("timestamp", "") if n.get("timestamp") else ""
            )

            for i in range(len(sorted_nodes) - 1):
                curr_node = sorted_nodes[i]
                next_node = sorted_nodes[i + 1]

                cam_from = curr_node.get("camera_id")
                cam_to = next_node.get("camera_id")

                t_from_str = curr_node.get("timestamp")
                t_to_str = next_node.get("timestamp")

                if not cam_from or not cam_to:
                    continue

                # Calculate time difference
                dt = 0.0
                if t_from_str and t_to_str:
                    try:
                        t1 = datetime.fromisoformat(t_from_str)
                        t2 = datetime.fromisoformat(t_to_str)
                        dt = (t2 - t1).total_seconds()
                    except Exception:
                        dt = 0.0

                # -----------------------------------------------------------------
                # 1. IMPOSSIBLE_TRANSITION Check (Teleportation or Speed > 180 km/h)
                # -----------------------------------------------------------------
                if cam_from != cam_to:
                    if dt < self.min_travel_time_sec:
                        score = 95
                        severity = self.compute_anomaly_severity(score)
                        anomalies.append({
                            "fused_identity": plate_or_identity,
                            "camera_from": cam_from,
                            "camera_to": cam_to,
                            "timestamp_from": t_from_str,
                            "timestamp_to": t_to_str,
                            "anomaly_score": score,
                            "severity": severity,
                            "anomaly_type": "IMPOSSIBLE_TRANSITION",
                            "reason": f"Transition from {cam_from} to {cam_to} occurred in {dt:.1f}s, violating minimum physical travel time ({self.min_travel_time_sec}s) (Teleportation / Implausible speed).",
                            "evidence": {
                                "time_diff_seconds": round(dt, 2),
                                "min_allowed_seconds": self.min_travel_time_sec,
                                "implied_behavior": "Teleportation / Duplicate plate registration",
                            },
                            "recommended_review": True,
                        })

                    elif dt > 0:
                        dist_km = self.distances.get(cam_from, {}).get(cam_to, 2.0)
                        speed_kmh = (dist_km / (dt / 3600.0)) if dt > 0 else 0.0
                        if speed_kmh > self.max_speed_kmh:
                            score = 92
                            severity = self.compute_anomaly_severity(score)
                            anomalies.append({
                                "fused_identity": plate_or_identity,
                                "camera_from": cam_from,
                                "camera_to": cam_to,
                                "timestamp_from": t_from_str,
                                "timestamp_to": t_to_str,
                                "anomaly_score": score,
                                "severity": severity,
                                "anomaly_type": "IMPOSSIBLE_TRANSITION",
                                "reason": f"Calculated travel speed ({speed_kmh:.1f} km/h) between {cam_from} and {cam_to} exceeds physical threshold ({self.max_speed_kmh} km/h).",
                                "evidence": {
                                    "distance_km": dist_km,
                                    "time_diff_seconds": round(dt, 2),
                                    "calculated_speed_kmh": round(speed_kmh, 1),
                                    "max_speed_threshold_kmh": self.max_speed_kmh,
                                },
                                "recommended_review": True,
                            })

                # -----------------------------------------------------------------
                # 2. UNEXPECTED_ROUTE Check (Transition not in topology graph)
                # -----------------------------------------------------------------
                if cam_from != cam_to:
                    allowed_next = self.topology.get(cam_from)
                    if allowed_next is not None and cam_to not in allowed_next:
                        score = 87
                        severity = self.compute_anomaly_severity(score)
                        anomalies.append({
                            "fused_identity": plate_or_identity,
                            "camera_from": cam_from,
                            "camera_to": cam_to,
                            "timestamp_from": t_from_str,
                            "timestamp_to": t_to_str,
                            "anomaly_score": score,
                            "severity": severity,
                            "anomaly_type": "UNEXPECTED_ROUTE",
                            "reason": f"Observed transition from {cam_from} to {cam_to} is outside the configured camera transition graph.",
                            "evidence": {
                                "topology_source": self.topology_source,
                                "expected_next_nodes": allowed_next,
                                "observed_next_node": cam_to,
                                "time_diff_seconds": round(dt, 2),
                            },
                            "recommended_review": True,
                        })

                # -----------------------------------------------------------------
                # 3. TEMPORAL_ANOMALY Check (Excessive travel window gap > 1800s)
                # -----------------------------------------------------------------
                if cam_from != cam_to and dt > self.max_travel_time_sec:
                    score = 65
                    severity = self.compute_anomaly_severity(score)
                    anomalies.append({
                        "fused_identity": plate_or_identity,
                        "camera_from": cam_from,
                        "camera_to": cam_to,
                        "timestamp_from": t_from_str,
                        "timestamp_to": t_to_str,
                        "anomaly_score": score,
                        "severity": severity,
                        "anomaly_type": "TEMPORAL_ANOMALY",
                        "reason": f"Time gap between {cam_from} and {cam_to} ({dt/60.0:.1f} minutes) exceeds valid travel window ({self.max_travel_time_sec/60.0:.0f} mins).",
                        "evidence": {
                            "time_diff_seconds": round(dt, 2),
                            "time_diff_minutes": round(dt / 60.0, 1),
                            "max_expected_window_seconds": self.max_travel_time_sec,
                        },
                        "recommended_review": False,
                    })

        # -----------------------------------------------------------------
        # 4. IDENTITY_CONFLICT Check (Simultaneous detections of fused identity)
        # -----------------------------------------------------------------
        if fused_evidence:
            for ev_item in fused_evidence:
                ev_data = ev_item.get("evidence", {})
                time_diff = ev_data.get("timestamps", {}).get("time_diff_seconds", 999.0)
                cam1 = ev_data.get("source_camera")
                cam2 = ev_data.get("destination_camera")

                if cam1 and cam2 and cam1 != cam2 and abs(time_diff) <= 3.0:
                    score = 98
                    severity = self.compute_anomaly_severity(score)
                    anomalies.append({
                        "fused_identity": plate_or_identity,
                        "camera_from": cam1,
                        "camera_to": cam2,
                        "timestamp_from": ev_data.get("timestamps", {}).get("source"),
                        "timestamp_to": ev_data.get("timestamps", {}).get("destination"),
                        "anomaly_score": score,
                        "severity": severity,
                        "anomaly_type": "IDENTITY_CONFLICT",
                        "reason": f"Conflicting simultaneous sightings of fused identity '{plate_or_identity}' at nodes {cam1} and {cam2} within {abs(time_diff):.1f}s.",
                        "evidence": {
                            "source_camera": cam1,
                            "destination_camera": cam2,
                            "original_ocr_source": ev_data.get("original_ocr_values", {}).get("source"),
                            "original_ocr_destination": ev_data.get("original_ocr_values", {}).get("destination"),
                            "fused_confidence": ev_item.get("fused_identity_confidence"),
                            "time_diff_seconds": abs(time_diff),
                        },
                        "recommended_review": True,
                    })

        return anomalies

    def run_anomaly_radar(
        self,
        reconstructed_trajectories: Dict[str, Any],
        fused_identities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete Anomaly Radar scan across all reconstructed vehicle trajectories and fused identities.
        """
        all_anomalies: List[Dict[str, Any]] = []
        fused_map: Dict[str, List[Dict[str, Any]]] = {}

        if fused_identities:
            for f in fused_identities:
                plate = f.get("fused_plate_number")
                if plate:
                    if plate not in fused_map:
                        fused_map[plate] = []
                    fused_map[plate].append(f)

        all_plates = list(dict.fromkeys(list(reconstructed_trajectories.keys()) + list(fused_map.keys())))

        for plate in all_plates:
            traj_info = reconstructed_trajectories.get(plate, {})
            nodes = traj_info.get("trajectory_nodes", [])
            f_ev = fused_map.get(plate, [])
            anoms = self.analyze_trajectory(plate, nodes, f_ev)
            all_anomalies.extend(anoms)

        # Compute summary metrics
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        type_counts = {
            "UNEXPECTED_ROUTE": 0,
            "IMPOSSIBLE_TRANSITION": 0,
            "TEMPORAL_ANOMALY": 0,
            "IDENTITY_CONFLICT": 0,
        }

        for a in all_anomalies:
            sev = a.get("severity", "LOW")
            atype = a.get("anomaly_type", "UNEXPECTED_ROUTE")
            if sev in severity_counts:
                severity_counts[sev] += 1
            if atype in type_counts:
                type_counts[atype] += 1

        report = {
            "scan_timestamp": datetime.now().isoformat(),
            "topology_source": self.topology_source,
            "total_trajectories_scanned": len(reconstructed_trajectories),
            "anomalies_detected": len(all_anomalies),
            "severity_breakdown": severity_counts,
            "anomaly_type_breakdown": type_counts,
            "anomaly_records": all_anomalies,
        }
        return report


def main():
    parser = argparse.ArgumentParser(
        description="SIH26127 - Phase 1B Anomaly Radar Engine."
    )
    parser.add_argument(
        "--report",
        type=str,
        default="backend/data/analytics_report.json",
        help="Path to analytics report JSON file.",
    )
    args = parser.parse_args()

    if os.path.exists(args.report):
        with open(args.report, "r") as f:
            data = json.load(f)
        trajectories = data.get("reconstructed_trajectories", {})
        fused = data.get("fused_identities", [])
    else:
        print(f"[WARNING] File not found: '{args.report}'. Running demo anomaly scan.")
        trajectories = {}
        fused = []

    engine = AnomalyRadarEngine()
    result = engine.run_anomaly_radar(trajectories, fused)

    print("\n================================================================================")
    print("                      PHASE 1B ANOMALY RADAR SCAN SUMMARY                        ")
    print("================================================================================")
    print(f"Topology Source            : {result['topology_source']}")
    print(f"Trajectories Scanned       : {result['total_trajectories_scanned']}")
    print(f"Total Anomalies Flagged    : {result['anomalies_detected']}")
    print(f"Severity Breakdown         : {result['severity_breakdown']}")
    print(f"Anomaly Type Breakdown     : {result['anomaly_type_breakdown']}")
    print("--------------------------------------------------------------------------------")
    for idx, anom in enumerate(result["anomaly_records"], 1):
        print(f"[{idx}] {anom['anomaly_type']} | Score: {anom['anomaly_score']} ({anom['severity']})")
        print(f"    Identity: {anom['fused_identity']} | Route: {anom['camera_from']} -> {anom['camera_to']}")
        print(f"    Reason  : {anom['reason']}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
