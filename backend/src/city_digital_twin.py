"""
City Digital Twin & Congestion Intelligence Engine (SIH26127)
--------------------------------------------------------------
Derives city-wide traffic states (FREE_FLOW, MODERATE, CONGESTED, SEVERE),
computes normalized 0-100 camera node congestion scores, and generates
spatial camera node intelligence profiles from multi-camera traffic data.

Part of SIH26127 Phase 2B Enhancement.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class CityDigitalTwinEngine:
    """
    Engine for calculating normalized congestion scores, classifying digital twin
    traffic states, and building camera node spatial intelligence.
    """

    def __init__(self, max_volume_threshold: int = 15):
        """
        Initialize Digital Twin Engine with configurable volume normalization threshold.
        """
        self.max_volume_threshold = max_volume_threshold

    def compute_congestion_score(
        self,
        traffic_volume: Optional[int] = 0,
        congestion_level: Optional[str] = "LOW",
    ) -> int:
        """
        Calculate normalized 0-100 congestion score.
        Formula:
            S_vol = min(100.0, (traffic_volume / max_threshold) * 100.0)
            S_level = 20.0 (LOW), 60.0 (MODERATE), 90.0 (HIGH)
            Congestion Score = round(0.6 * S_vol + 0.4 * S_level)
        """
        vol = max(0, traffic_volume or 0)
        s_vol = min(100.0, (vol / float(self.max_volume_threshold)) * 100.0)

        level_upper = (congestion_level or "LOW").upper()
        if level_upper == "HIGH":
            s_level = 90.0
        elif level_upper == "MODERATE":
            s_level = 60.0
        else:
            s_level = 20.0

        score = int(round(0.6 * s_vol + 0.4 * s_level))
        return max(0, min(100, score))

    def classify_traffic_state(self, congestion_score: int) -> str:
        """
        Classify traffic state based on 0-100 congestion score.
        State Boundaries:
            - FREE_FLOW  : Score < 30
            - MODERATE   : 30 <= Score < 60
            - CONGESTED  : 60 <= Score < 85
            - SEVERE     : Score >= 85
        """
        score = max(0, min(100, congestion_score or 0))
        if score < 30:
            return "FREE_FLOW"
        elif score < 60:
            return "MODERATE"
        elif score < 85:
            return "CONGESTED"
        else:
            return "SEVERE"

    def compute_camera_nodes_intelligence(
        self,
        camera_analytics: Optional[Dict[str, Any]] = None,
        reconstructed_trajectories: Optional[Dict[str, Any]] = None,
        city_traffic_intelligence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build camera node spatial intelligence records.
        """
        nodes = set()

        if camera_analytics:
            nodes.update(camera_analytics.keys())

        if city_traffic_intelligence and "od_matrix" in city_traffic_intelligence:
            nodes.update(city_traffic_intelligence["od_matrix"].get("camera_nodes", []))

        if reconstructed_trajectories:
            for plate, data in reconstructed_trajectories.items():
                if isinstance(data, dict):
                    seq = data.get("camera_sequence", [])
                    nodes.update(seq)

        if not nodes:
            nodes = {"cam_01", "cam_02", "cam_03", "cam_04"}

        sorted_nodes = sorted(list(nodes))
        nodes_intel: Dict[str, Dict[str, Any]] = {}

        # Extract flows if available from city_traffic_intelligence
        flows_summary = {}
        if city_traffic_intelligence:
            flows_summary = city_traffic_intelligence.get("camera_flow_summary", {})

        for cam in sorted_nodes:
            stats = camera_analytics.get(cam, {}) if camera_analytics else {}
            vol = stats.get("traffic_volume", 0)
            unique_v = stats.get("unique_vehicles", 0)
            cong_level = stats.get("congestion_level", "LOW")

            score = self.compute_congestion_score(vol, cong_level)
            state = self.classify_traffic_state(score)

            # Flow details
            cam_flow = flows_summary.get(cam, {})
            in_flow = cam_flow.get("incoming_flow", 0)
            out_flow = cam_flow.get("outgoing_flow", 0)
            net_balance = in_flow - out_flow

            nodes_intel[cam] = {
                "camera_id": cam,
                "traffic_state": state,
                "congestion_score": score,
                "detection_volume": vol,
                "unique_vehicle_count": unique_v,
                "congestion_level_category": cong_level,
                "incoming_flow": in_flow,
                "outgoing_flow": out_flow,
                "net_flow_balance": net_balance,
                "status_indicator": {
                    "FREE_FLOW": "EMERALD (Normal Flow)",
                    "MODERATE": "CYAN (Medium Volume)",
                    "CONGESTED": "AMBER (High Density)",
                    "SEVERE": "ROSE (Severe Congestion Bottleneck)",
                }.get(state, "FREE_FLOW"),
            }

        return nodes_intel

    def compute_digital_twin_summary(
        self,
        camera_analytics: Optional[Dict[str, Any]] = None,
        reconstructed_trajectories: Optional[Dict[str, Any]] = None,
        city_traffic_intelligence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble complete City Digital Twin & Congestion Intelligence analysis.
        """
        nodes_intel = self.compute_camera_nodes_intelligence(
            camera_analytics=camera_analytics,
            reconstructed_trajectories=reconstructed_trajectories,
            city_traffic_intelligence=city_traffic_intelligence,
        )

        scores = [n["congestion_score"] for n in nodes_intel.values()]
        avg_score = int(round(sum(scores) / float(len(scores)))) if scores else 0
        city_state = self.classify_traffic_state(avg_score)

        state_counts = {"FREE_FLOW": 0, "MODERATE": 0, "CONGESTED": 0, "SEVERE": 0}
        for n in nodes_intel.values():
            st = n.get("traffic_state", "FREE_FLOW")
            if st in state_counts:
                state_counts[st] += 1
            else:
                state_counts[st] = 1

        return {
            "summary": {
                "city_traffic_state": city_state,
                "average_city_congestion_score": avg_score,
                "total_monitored_nodes": len(nodes_intel),
                "state_breakdown": state_counts,
                "data_source_label": "DEMO/SIMULATED Multi-Camera ANPR Analytics Pipeline",
            },
            "camera_nodes_intelligence": nodes_intel,
            "scoring_formula_documentation": {
                "formula": "Congestion Score C = round(0.6 * S_vol + 0.4 * S_level)",
                "volume_component": f"S_vol = min(100, (traffic_volume / {self.max_volume_threshold}) * 100)",
                "level_component": "S_level: LOW=20, MODERATE=60, HIGH=90",
                "state_thresholds": "FREE_FLOW (<30), MODERATE (30-59), CONGESTED (60-84), SEVERE (>=85)",
            },
        }
