"""
What-If Traffic Simulator & Decision Intelligence Engine (SIH26127)
--------------------------------------------------------------------
Runs isolated hypothetical traffic simulation scenarios (volume increase/decrease)
and propagates downstream congestion impacts across topology corridors without mutating
real analytics data.

Part of SIH26127 Phase 2C Enhancement.
"""

import copy
from typing import List, Dict, Any, Optional
from datetime import datetime

from city_digital_twin import CityDigitalTwinEngine


class TrafficSimulatorEngine:
    """
    Engine for running What-If traffic simulations, propagating corridor volume shifts,
    estimating simulated congestion scores, and generating decision-support guidance.
    """

    def __init__(self, topology: Optional[Dict[str, List[str]]] = None):
        """
        Initialize simulator with camera network topology graph.
        """
        self.topology = topology or {
            "cam_01": ["cam_02", "cam_03"],
            "cam_02": ["cam_01", "cam_04"],
            "cam_03": ["cam_01", "cam_04"],
            "cam_04": ["cam_02", "cam_03"],
        }
        self.dt_engine = CityDigitalTwinEngine()

    def classify_impact_level(self, delta_score: int, sim_score: int) -> str:
        """
        Classify simulation impact level based on score shift and absolute simulated score.
        Thresholds:
            - LOW      : delta == 0 or (|delta| < 10 and sim_score < 60)
            - MODERATE : |delta| < 25 and sim_score < 75
            - HIGH     : |delta| < 40 and sim_score < 85
            - CRITICAL : |delta| >= 40 or sim_score >= 85
        """
        if delta_score == 0:
            return "LOW"

        abs_delta = abs(delta_score)

        if sim_score >= 85 or abs_delta >= 40:
            return "CRITICAL"
        elif sim_score >= 75 or abs_delta >= 25:
            return "HIGH"
        elif sim_score >= 60 or abs_delta >= 10:
            return "MODERATE"
        else:
            return "LOW"

    def generate_decision_recommendations(
        self,
        target_cam: str,
        traffic_change_percent: float,
        target_sim_data: Dict[str, Any],
        downstream_sim_data: List[Dict[str, Any]],
        overall_impact: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate explainable decision-support recommendations supported strictly by simulation metrics.
        """
        recommendations = []

        change_sign = "+" if traffic_change_percent > 0 else ""
        change_str = f"{change_sign}{traffic_change_percent:.0f}%"

        # 1. Target Camera Recommendation
        t_delta = target_sim_data.get("delta_score", 0)
        t_state = target_sim_data.get("simulated_traffic_state", "FREE_FLOW")
        t_sim_vol = target_sim_data.get("simulated_volume", 0)

        if overall_impact in ["HIGH", "CRITICAL"]:
            recommendations.append({
                "type": "CONGESTION_WARNING",
                "priority": "HIGH",
                "title": f"Escalated Congestion Warning at {target_cam.upper()}",
                "description": f"Simulated {change_str} traffic shift causes congestion score at {target_cam.upper()} to increase by +{t_delta} points to state '{t_state}' ({t_sim_vol} vehicles).",
                "action_guidance": f"Prioritize monitoring at {target_cam.upper()} and prepare corridor flow diversion.",
            })
        elif overall_impact == "MODERATE":
            recommendations.append({
                "type": "CORRIDOR_MONITORING",
                "priority": "MEDIUM",
                "title": f"Monitor Primary Corridor at {target_cam.upper()}",
                "description": f"Simulated {change_str} traffic shift results in moderate score change (+{t_delta} pts) at {target_cam.upper()}.",
                "action_guidance": f"Increase video feed monitoring frequency for {target_cam.upper()}.",
            })
        else:
            recommendations.append({
                "type": "NORMAL_OPERATIONS",
                "priority": "LOW",
                "title": f"Minor Simulated Impact at {target_cam.upper()}",
                "description": f"Simulated {change_str} traffic shift produces minimal impact on baseline flow ({t_delta} pts shift).",
                "action_guidance": "No immediate tactical intervention required.",
            })

        # 2. Downstream Propagation Recommendations
        for down in downstream_sim_data:
            d_cam = down.get("camera_id", "")
            d_impact = down.get("impact_level", "LOW")
            d_delta = down.get("delta_score", 0)
            d_sim_vol = down.get("simulated_volume", 0)
            d_prop_pct = down.get("propagation_percentage", 0.0)

            if d_impact in ["MODERATE", "HIGH", "CRITICAL"] and d_delta > 0:
                recommendations.append({
                    "type": "DOWNSTREAM_BOTTLENECK",
                    "priority": "HIGH" if d_impact in ["HIGH", "CRITICAL"] else "MEDIUM",
                    "title": f"Downstream Impact at {d_cam.upper()} ({d_prop_pct}% Flow Shift)",
                    "description": f"Traffic surge from {target_cam.upper()} propagates downstream to {d_cam.upper()}, adding {d_delta} score points ({d_sim_vol} total simulated vehicles).",
                    "action_guidance": f"Consider recommending alternate routing via secondary corridors to bypass {d_cam.upper()}.",
                })

        return recommendations

    def run_what_if_simulation(
        self,
        camera_id: str,
        traffic_change_percent: float,
        analytics_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute an isolated What-If scenario simulation.
        Does NOT mutate the original analytics_report object.
        """
        # Input Validation
        if not camera_id or not isinstance(camera_id, str):
            return {
                "status": "ERROR",
                "error_message": "Invalid camera_id provided. Must be a non-empty string.",
            }

        try:
            change_pct = float(traffic_change_percent)
        except (ValueError, TypeError):
            return {
                "status": "ERROR",
                "error_message": f"Invalid traffic_change_percent '{traffic_change_percent}'. Must be a numeric value.",
            }

        if change_pct < -100.0 or change_pct > 300.0:
            return {
                "status": "ERROR",
                "error_message": f"traffic_change_percent ({change_pct}%) outside allowable bounds [-100.0%, +300.0%].",
            }

        # Deep copy to guarantee zero source data mutation
        report_copy = copy.deepcopy(analytics_report or {})

        cam_analytics = report_copy.get("camera_analytics", {})
        dt_intel = report_copy.get("city_digital_twin", {}).get("camera_nodes_intelligence", {})
        ci_intel = report_copy.get("city_traffic_intelligence", {})

        target_cam = camera_id.lower()
        if target_cam not in cam_analytics and target_cam not in dt_intel:
            # Fallback check if camera node exists in defaults
            available_cams = list(dict.fromkeys(list(cam_analytics.keys()) + list(dt_intel.keys()) + list(self.topology.keys())))
            if target_cam not in available_cams:
                return {
                    "status": "ERROR",
                    "error_message": f"Target camera node '{camera_id}' not found in analytics dataset or network topology.",
                }

        # 1. Target Camera Baseline Metrics
        target_stats = cam_analytics.get(target_cam, {})
        target_node_intel = dt_intel.get(target_cam, {})

        base_vol = target_stats.get("traffic_volume", 0)
        base_level = target_stats.get("congestion_level", "LOW")
        base_score = target_node_intel.get("congestion_score", self.dt_engine.compute_congestion_score(base_vol, base_level))

        # 2. Target Simulated Volume & Score
        delta_vol_target = int(round(base_vol * (change_pct / 100.0)))
        sim_vol_target = max(0, base_vol + delta_vol_target)

        if delta_vol_target == 0:
            sim_score_target = base_score
            delta_score_target = 0
            sim_level_target = base_level
        else:
            if sim_vol_target >= 15:
                sim_level_target = "HIGH"
            elif sim_vol_target >= 5:
                sim_level_target = "MODERATE"
            else:
                sim_level_target = "LOW"
            sim_score_target = self.dt_engine.compute_congestion_score(sim_vol_target, sim_level_target)
            delta_score_target = sim_score_target - base_score

        target_state_sim = self.dt_engine.classify_traffic_state(sim_score_target)
        target_impact = self.classify_impact_level(delta_score_target, sim_score_target)

        target_sim_data = {
            "camera_id": target_cam,
            "baseline_volume": base_vol,
            "simulated_volume": sim_vol_target,
            "volume_change": delta_vol_target,
            "baseline_score": base_score,
            "simulated_score": sim_score_target,
            "delta_score": delta_score_target,
            "simulated_traffic_state": target_state_sim,
            "impact_level": target_impact,
        }

        # 3. Downstream Nodes Propagation
        downstream_cams = self.topology.get(target_cam, [])
        od_matrix_data = ci_intel.get("od_matrix", {}).get("matrix", {})
        od_flows = od_matrix_data.get(target_cam, {})

        total_out_flow = sum(od_flows.values()) if od_flows else 0

        downstream_results = []
        max_downstream_impact_val = 0

        for down_cam in downstream_cams:
            # Calculate propagation weight
            if total_out_flow > 0 and down_cam in od_flows:
                prop_weight = od_flows[down_cam] / float(total_out_flow)
            else:
                prop_weight = 1.0 / float(len(downstream_cams)) if downstream_cams else 0.0

            delta_vol_down = int(round(delta_vol_target * prop_weight))

            down_stats = cam_analytics.get(down_cam, {})
            down_node_intel = dt_intel.get(down_cam, {})

            d_base_vol = down_stats.get("traffic_volume", 0)
            d_base_level = down_stats.get("congestion_level", "LOW")
            d_base_score = down_node_intel.get("congestion_score", self.dt_engine.compute_congestion_score(d_base_vol, d_base_level))

            d_sim_vol = max(0, d_base_vol + delta_vol_down)
            if delta_vol_down == 0:
                d_sim_score = d_base_score
                d_delta_score = 0
                d_sim_level = d_base_level
            else:
                if d_sim_vol >= 15:
                    d_sim_level = "HIGH"
                elif d_sim_vol >= 5:
                    d_sim_level = "MODERATE"
                else:
                    d_sim_level = "LOW"
                d_sim_score = self.dt_engine.compute_congestion_score(d_sim_vol, d_sim_level)
                d_delta_score = d_sim_score - d_base_score

            d_sim_state = self.dt_engine.classify_traffic_state(d_sim_score)
            d_impact = self.classify_impact_level(d_delta_score, d_sim_score)

            downstream_results.append({
                "camera_id": down_cam,
                "corridor_id": f"{target_cam} -> {down_cam}",
                "propagation_weight": round(prop_weight, 2),
                "propagation_percentage": round(prop_weight * 100.0, 1),
                "baseline_volume": d_base_vol,
                "simulated_volume": d_sim_vol,
                "volume_change": delta_vol_down,
                "baseline_score": d_base_score,
                "simulated_score": d_sim_score,
                "delta_score": d_delta_score,
                "simulated_traffic_state": d_sim_state,
                "impact_level": d_impact,
            })

        # Overall Scenario Impact
        impact_ranks = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}
        all_impacts = [target_impact] + [d["impact_level"] for d in downstream_results]
        overall_impact = max(all_impacts, key=lambda imp: impact_ranks.get(imp, 1))

        # Recommendations
        recommendations = self.generate_decision_recommendations(
            target_cam=target_cam,
            traffic_change_percent=change_pct,
            target_sim_data=target_sim_data,
            downstream_sim_data=downstream_results,
            overall_impact=overall_impact,
        )

        return {
            "status": "SUCCESS",
            "is_simulated": True,
            "status_label": "WHAT-IF SIMULATION ESTIMATE",
            "simulation_timestamp": datetime.now().isoformat(),
            "scenario": {
                "target_camera_id": target_cam,
                "traffic_change_percent": change_pct,
                "change_label": f"{change_pct:+.0f}% Volume Shift",
            },
            "overall_impact_level": overall_impact,
            "target_camera_simulation": target_sim_data,
            "downstream_corridors_simulation": downstream_results,
            "decision_recommendations": recommendations,
            "disclaimer": "DISCLAIMER: Decision-support guidance only. What-If estimates do not issue commands to traffic signal control systems.",
        }
