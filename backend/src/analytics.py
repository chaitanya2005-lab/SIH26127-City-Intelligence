"""
City-Wide Urban Traffic Analytics Module (SIH26127)
---------------------------------------------------
Processes multi-camera ANPR vehicle detection and trajectory records to generate:
1. Total unique vehicle count
2. Camera-wise traffic flow & volume metrics
3. Vehicle class distribution (car, motorcycle, bus, truck)
4. Vehicle trajectory & path reconstruction data across camera nodes
5. Real-time congestion index (Low, Moderate, High) per camera location

================================================================================
HOW TO RUN THIS MODULE:
================================================================================
1. Run standalone analytics engine with sample multi-camera data:
   python backend/src/analytics.py

2. Save analytics summary report to JSON:
   python backend/src/analytics.py --output backend/data/analytics_report.json
================================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class TrafficAnalyticsEngine:
    def __init__(self, congestion_thresholds: Optional[Dict[str, int]] = None):
        """
        Initialize the Urban Traffic Analytics Engine.

        :param congestion_thresholds: Dict defining vehicle count thresholds for Low, Moderate, High congestion.
        """
        self.congestion_thresholds = congestion_thresholds or {
            "LOW": 5,
            "MODERATE": 15,
        }

    def compute_vehicle_counts(self, trajectory_records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Compute total vehicle detections and unique license plate count.
        """
        total_detections = len(trajectory_records)
        unique_plates = set()
        for r in trajectory_records:
            plate = r.get("plate_number")
            if plate and plate != "UNREADABLE":
                unique_plates.add(plate)

        return {
            "total_detection_events": total_detections,
            "unique_license_plates": len(unique_plates),
        }

    def compute_camera_flow(self, trajectory_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate traffic volume, unique vehicles, and congestion status per camera node.
        """
        cam_stats: Dict[str, Dict[str, Any]] = {}

        for r in trajectory_records:
            cam_id = r.get("camera_id", "cam_01")
            plate = r.get("plate_number")
            vclass = r.get("vehicle_class", "car")

            if cam_id not in cam_stats:
                cam_stats[cam_id] = {
                    "total_vehicles": 0,
                    "unique_plates": set(),
                    "class_breakdown": {},
                }

            cam_stats[cam_id]["total_vehicles"] += 1
            if plate and plate != "UNREADABLE":
                cam_stats[cam_id]["unique_plates"].add(plate)

            cam_stats[cam_id]["class_breakdown"][vclass] = (
                cam_stats[cam_id]["class_breakdown"].get(vclass, 0) + 1
            )

        result = {}
        for cam_id, data in cam_stats.items():
            vol = data["total_vehicles"]
            if vol < self.congestion_thresholds["LOW"]:
                congestion_level = "LOW"
            elif vol < self.congestion_thresholds["MODERATE"]:
                congestion_level = "MODERATE"
            else:
                congestion_level = "HIGH"

            result[cam_id] = {
                "traffic_volume": vol,
                "unique_vehicles": len(data["unique_plates"]),
                "congestion_level": congestion_level,
                "vehicle_classes": data["class_breakdown"],
            }

        return result

    def compute_class_distribution(self, trajectory_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate vehicle class breakdown (car, motorcycle, bus, truck) across all camera nodes.
        """
        distribution: Dict[str, int] = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

        for r in trajectory_records:
            vclass = r.get("vehicle_class", "car").lower()
            if vclass in distribution:
                distribution[vclass] += 1
            else:
                distribution[vclass] = 1

        total = max(1, sum(distribution.values()))
        percentage = {
            k: round((v / total) * 100, 1) for k, v in distribution.items()
        }

        return {
            "counts": distribution,
            "percentages": percentage,
        }

    def reconstruct_trajectories(self, trajectory_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group spatial-temporal records by license plate to reconstruct complete vehicle paths.
        """
        paths: Dict[str, List[Dict[str, Any]]] = {}

        for r in trajectory_records:
            plate = r.get("plate_number")
            if not plate or plate == "UNREADABLE":
                continue

            node = {
                "camera_id": r.get("camera_id"),
                "timestamp": r.get("timestamp"),
                "vehicle_class": r.get("vehicle_class"),
                "confidence": r.get("confidence"),
                "ocr_confidence": r.get("ocr_confidence", 0.0),
                "confidence_category": r.get("confidence_category", "LOW CONFIDENCE"),
                "bbox": r.get("bbox"),
            }

            if plate not in paths:
                paths[plate] = []
            paths[plate].append(node)

        # Sort trajectory nodes chronologically per plate
        for plate in paths:
            paths[plate] = sorted(paths[plate], key=lambda x: x.get("timestamp", ""))

        return paths

    def compute_fused_identities(self, trajectory_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform confidence-aware cross-camera vehicle identity fusion and generate evidence chains.
        """
        try:
            from anpr import VehicleIdentityFusion
            fusion_engine = VehicleIdentityFusion()
        except ImportError:
            return []

        valid_records = [
            r for r in trajectory_records
            if r.get("plate_number") and r.get("plate_number") != "UNREADABLE"
        ]

        fused_matches = []
        seen_pairs = set()

        for i in range(len(valid_records)):
            for j in range(i + 1, len(valid_records)):
                r1 = valid_records[i]
                r2 = valid_records[j]
                if r1.get("camera_id") != r2.get("camera_id"):
                    pair_key = (r1.get("camera_id"), r1.get("tracking_id"), r2.get("camera_id"), r2.get("tracking_id"))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    res = fusion_engine.fuse_identities(r1, r2)
                    if res["is_valid_match"]:
                        fused_matches.append(res)

        return fused_matches

    def generate_analytics_report(self, trajectory_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate complete city-wide traffic analytics report.
        """
        vehicle_counts = self.compute_vehicle_counts(trajectory_records)
        camera_flow = self.compute_camera_flow(trajectory_records)
        class_stats = self.compute_class_distribution(trajectory_records)
        trajectories = self.reconstruct_trajectories(trajectory_records)
        fused_identities = self.compute_fused_identities(trajectory_records)

        max_vol = max([cam["traffic_volume"] for cam in camera_flow.values()], default=0)
        city_congestion = "LOW"
        if max_vol >= self.congestion_thresholds["MODERATE"]:
            city_congestion = "HIGH"
        elif max_vol >= self.congestion_thresholds["LOW"]:
            city_congestion = "MODERATE"

        # Compute Anomaly Radar Analysis
        try:
            from anomaly_radar import AnomalyRadarEngine
            radar = AnomalyRadarEngine()
            anomaly_report = radar.run_anomaly_radar(
                reconstructed_trajectories={
                    plate: {
                        "camera_sequence": list(dict.fromkeys([n["camera_id"] for n in nodes])),
                        "total_sightings": len(nodes),
                        "trajectory_nodes": nodes,
                    }
                    for plate, nodes in trajectories.items()
                },
                fused_identities=fused_identities,
            )
        except Exception:
            anomaly_report = {
                "scan_timestamp": datetime.now().isoformat(),
                "topology_source": "DEMO/SIMULATED camera topology",
                "total_trajectories_scanned": len(trajectories),
                "anomalies_detected": 0,
                "severity_breakdown": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
                "anomaly_type_breakdown": {"UNEXPECTED_ROUTE": 0, "IMPOSSIBLE_TRANSITION": 0, "TEMPORAL_ANOMALY": 0, "IDENTITY_CONFLICT": 0},
                "anomaly_records": [],
            }

        # Compute Predictive Route Intelligence Analysis
        try:
            from route_predictor import RoutePredictorEngine
            predictor = RoutePredictorEngine()
            predictive_report = predictor.run_route_predictions(
                reconstructed_trajectories={
                    plate: {
                        "camera_sequence": list(dict.fromkeys([n["camera_id"] for n in nodes])),
                        "total_sightings": len(nodes),
                        "trajectory_nodes": nodes,
                    }
                    for plate, nodes in trajectories.items()
                }
            )
        except Exception:
            predictive_report = {
                "prediction_timestamp": datetime.now().isoformat(),
                "topology_source": "DEMO/SIMULATED camera topology",
                "total_vehicles_predicted": 0,
                "predictions": {},
            }

        # Compute Phase 1D Evidence Chain Analysis
        try:
            from evidence_chain import EvidenceChainEngine
            ev_engine = EvidenceChainEngine()
            evidence_chain = ev_engine.build_evidence_chain(
                anomaly_radar_output=anomaly_report,
                reconstructed_trajectories={
                    plate: {
                        "camera_sequence": list(dict.fromkeys([n["camera_id"] for n in nodes])),
                        "total_sightings": len(nodes),
                        "trajectory_nodes": nodes,
                    }
                    for plate, nodes in trajectories.items()
                },
                predictions_output=predictive_report,
            )
        except Exception:
            evidence_chain = []

        # Compute Phase 2A City-Wide Traffic Intelligence Analysis
        try:
            from city_traffic_intelligence import CityTrafficIntelligenceEngine
            city_engine = CityTrafficIntelligenceEngine()
            city_traffic_intel = city_engine.compute_city_traffic_intelligence(
                reconstructed_trajectories={
                    plate: {
                        "camera_sequence": list(dict.fromkeys([n["camera_id"] for n in nodes])),
                        "total_sightings": len(nodes),
                        "trajectory_nodes": nodes,
                    }
                    for plate, nodes in trajectories.items()
                },
                camera_analytics=camera_flow,
            )
        except Exception:
            city_traffic_intel = {
                "summary": {
                    "total_observations": vehicle_counts["total_detection_events"],
                    "unique_vehicles": vehicle_counts["unique_license_plates"],
                    "active_cameras": len(camera_flow),
                    "total_inter_camera_transitions": 0,
                    "unique_active_corridors": 0,
                },
                "od_matrix": {"camera_nodes": list(camera_flow.keys()), "matrix": {}, "total_transitions": 0},
                "top_corridors": [],
                "camera_flow_summary": {},
                "traffic_insights": [],
            }

        # Compute Phase 2B City Digital Twin & Congestion Intelligence Analysis
        try:
            from city_digital_twin import CityDigitalTwinEngine
            dt_engine = CityDigitalTwinEngine()
            city_digital_twin = dt_engine.compute_digital_twin_summary(
                camera_analytics=camera_flow,
                reconstructed_trajectories={
                    plate: {
                        "camera_sequence": list(dict.fromkeys([n["camera_id"] for n in nodes])),
                        "total_sightings": len(nodes),
                        "trajectory_nodes": nodes,
                    }
                    for plate, nodes in trajectories.items()
                },
                city_traffic_intelligence=city_traffic_intel,
            )
        except Exception:
            city_digital_twin = {
                "summary": {
                    "city_traffic_state": "FREE_FLOW",
                    "average_city_congestion_score": 0,
                    "total_monitored_nodes": len(camera_flow),
                    "state_breakdown": {"FREE_FLOW": len(camera_flow), "MODERATE": 0, "CONGESTED": 0, "SEVERE": 0},
                    "data_source_label": "DEMO/SIMULATED Multi-Camera ANPR Analytics Pipeline",
                },
                "camera_nodes_intelligence": {},
                "scoring_formula_documentation": {},
            }

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_events": vehicle_counts["total_detection_events"],
                "unique_vehicles_tracked": vehicle_counts["unique_license_plates"],
                "active_camera_nodes": len(camera_flow),
                "city_congestion_index": city_congestion,
                "fused_identities_count": len(fused_identities),
                "anomalies_detected_count": anomaly_report.get("anomalies_detected", 0),
                "predicted_vehicles_count": predictive_report.get("total_vehicles_predicted", 0),
                "evidence_chain_records_count": len(evidence_chain),
                "active_corridors_count": city_traffic_intel.get("summary", {}).get("unique_active_corridors", 0),
                "city_traffic_state": city_digital_twin.get("summary", {}).get("city_traffic_state", "FREE_FLOW"),
                "average_city_congestion_score": city_digital_twin.get("summary", {}).get("average_city_congestion_score", 0),
            },
            "camera_analytics": camera_flow,
            "vehicle_classification": class_stats,
            "fused_identities": fused_identities,
            "anomaly_radar": anomaly_report,
            "predictive_route_intelligence": predictive_report,
            "evidence_chain": evidence_chain,
            "city_traffic_intelligence": city_traffic_intel,
            "city_digital_twin": city_digital_twin,
            "reconstructed_trajectories": {
                plate: {
                    "camera_sequence": list(dict.fromkeys([n["camera_id"] for n in nodes])),
                    "total_sightings": len(nodes),
                    "first_seen": nodes[0]["timestamp"] if nodes else None,
                    "last_seen": nodes[-1]["timestamp"] if nodes else None,
                    "trajectory_nodes": nodes,
                }
                for plate, nodes in trajectories.items()
            },
        }
        return report


def get_mock_multi_camera_data() -> List[Dict[str, Any]]:
    """
    Generates realistic multi-camera trajectory records for testing analytics & Phase 1A identity fusion.
    """
    now = datetime.now()
    records = [
        {
            "camera_id": "cam_01",
            "tracking_id": 1,
            "plate_number": "MH12DE1408",
            "raw_ocr_text": "MH12DE1408",
            "normalized_plate_text": "MH12DE1408",
            "ocr_confidence": 0.91,
            "confidence_category": "HIGH CONFIDENCE",
            "vehicle_class": "car",
            "confidence": 0.92,
            "timestamp": (now - timedelta(minutes=5)).isoformat(),
            "bbox": [100, 200, 240, 265],
        },
        {
            "camera_id": "cam_01",
            "tracking_id": 2,
            "plate_number": "KA01AB2026",
            "raw_ocr_text": "KA01AB2026",
            "normalized_plate_text": "KA01AB2026",
            "ocr_confidence": 0.88,
            "confidence_category": "HIGH CONFIDENCE",
            "vehicle_class": "truck",
            "confidence": 0.88,
            "timestamp": (now - timedelta(minutes=4)).isoformat(),
            "bbox": [150, 300, 330, 385],
        },
        {
            "camera_id": "cam_02",
            "tracking_id": 1,
            "plate_number": "MH12DE14O8",
            "raw_ocr_text": "MH12DE14O8",
            "normalized_plate_text": "MH12DE14O8",
            "ocr_confidence": 0.76,
            "confidence_category": "MEDIUM CONFIDENCE",
            "vehicle_class": "car",
            "confidence": 0.94,
            "timestamp": (now - timedelta(minutes=3)).isoformat(),
            "bbox": [120, 210, 260, 275],
        },
        {
            "camera_id": "cam_02",
            "tracking_id": 3,
            "plate_number": "DL03XY9988",
            "raw_ocr_text": "DL03XY9988",
            "normalized_plate_text": "DL03XY9988",
            "ocr_confidence": 0.91,
            "confidence_category": "HIGH CONFIDENCE",
            "vehicle_class": "car",
            "confidence": 0.91,
            "timestamp": (now - timedelta(minutes=2)).isoformat(),
            "bbox": [400, 410, 530, 470],
        },
        {
            "camera_id": "cam_01",
            "tracking_id": 4,
            "plate_number": "TN09CC5544",
            "raw_ocr_text": "TN09CC5544",
            "normalized_plate_text": "TN09CC5544",
            "ocr_confidence": 0.89,
            "confidence_category": "HIGH CONFIDENCE",
            "vehicle_class": "bus",
            "confidence": 0.89,
            "timestamp": (now - timedelta(minutes=1)).isoformat(),
            "bbox": [200, 130, 400, 220],
        },
        {
            "camera_id": "cam_02",
            "tracking_id": 4,
            "plate_number": "TN09CC5544",
            "raw_ocr_text": "TN09CC5544",
            "normalized_plate_text": "TN09CC5544",
            "ocr_confidence": 0.93,
            "confidence_category": "HIGH CONFIDENCE",
            "vehicle_class": "bus",
            "confidence": 0.93,
            "timestamp": now.isoformat(),
            "bbox": [220, 140, 420, 230],
        },
    ]
    return records


def main():
    parser = argparse.ArgumentParser(
        description="SIH26127 - City-Wide Urban Traffic Analytics Engine."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to JSON file containing trajectory records. If omitted, uses mock multi-camera data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/data/analytics_report.json",
        help="Path to output JSON report. Default: 'backend/data/analytics_report.json'.",
    )

    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        print(f"[INFO] Loading trajectory records from: '{args.data}'...")
        with open(args.data, "r") as f:
            trajectory_records = json.load(f)
    else:
        print("[INFO] Generating City-Wide Analytics from multi-camera trajectory records...")
        trajectory_records = get_mock_multi_camera_data()

    engine = TrafficAnalyticsEngine()
    report = engine.generate_analytics_report(trajectory_records)

    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # Display report summary in console
    print("\n================================================================================")
    print("                 CITY-WIDE URBAN TRAFFIC ANALYTICS SUMMARY                       ")
    print("================================================================================")
    print(f"Timestamp                 : {report['timestamp']}")
    print(f"Total Detection Events    : {report['summary']['total_events']}")
    print(f"Unique Vehicles Tracked   : {report['summary']['unique_vehicles_tracked']}")
    print(f"Active Camera Nodes       : {report['summary']['active_camera_nodes']}")
    print(f"City Congestion Index     : {report['summary']['city_congestion_index']}")
    print("--------------------------------------------------------------------------------")
    print("CAMERA-WISE TRAFFIC FLOW:")
    for cam_id, metrics in report["camera_analytics"].items():
        print(f"  - Node [{cam_id}]: Volume={metrics['traffic_volume']} | Unique={metrics['unique_vehicles']} | Congestion={metrics['congestion_level']}")
    print("--------------------------------------------------------------------------------")
    print("VEHICLE CLASSIFICATION BREAKDOWN:")
    for vclass, count in report["vehicle_classification"]["counts"].items():
        pct = report["vehicle_classification"]["percentages"][vclass]
        print(f"  - {vclass.capitalize():<12}: {count} ({pct}%)")
    print("--------------------------------------------------------------------------------")
    print("RECONSTRUCTED VEHICLE TRAJECTORIES:")
    for plate, traj in report["reconstructed_trajectories"].items():
        seq = " -> ".join(traj["camera_sequence"])
        print(f"  - Vehicle [{plate}]: Sightings={traj['total_sightings']} | Route: {seq}")
    print("================================================================================")
    print(f"[INFO] Detailed JSON Analytics Report saved to: '{args.output}'\n")


if __name__ == "__main__":
    main()
