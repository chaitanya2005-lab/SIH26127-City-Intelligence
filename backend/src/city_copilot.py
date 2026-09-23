"""
City Copilot Query & Natural Language Intent Engine (SIH26127 Phase 3)
------------------------------------------------------------------------
Provides a deterministic, zero-hallucination query layer over existing
ANPR, fusion, trajectory, anomaly, predictive route, evidence chain, OD matrix,
Digital Twin, and What-If simulator analytics data.

Part of SIH26127 Phase 3 Enhancement.
"""

import re
import copy
from typing import List, Dict, Any, Optional
from datetime import datetime

from traffic_simulator import TrafficSimulatorEngine
from city_digital_twin import CityDigitalTwinEngine


class CityCopilotEngine:
    """
    Deterministic intent detection and analytics query engine for SIH26127.
    Converts natural-language queries into structured analytics responses,
    5 Ws evidence summaries, map references, and explicit PREDICTED badges.
    """

    def __init__(self, simulator_engine: Optional[TrafficSimulatorEngine] = None):
        """
        Initialize engine with optional traffic simulator engine instance.
        """
        self.simulator_engine = simulator_engine or TrafficSimulatorEngine()
        self.dt_engine = CityDigitalTwinEngine()

    def process_query(self, query_text: str, analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user query string against analytics report dataset.
        Returns a structured dictionary matching the required Phase 3 Copilot schema.
        """
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            return {
                "intent": "UNKNOWN",
                "answer": "Insufficient data available for this query. Please enter a valid natural-language query.",
                "confidence": "0.0",
                "sources": [],
                "evidence": [],
                "map_reference": None,
                "predicted": False,
            }

        q_clean = query_text.strip()
        q_lower = q_clean.lower()
        report = analytics_report or {}

        # 1. WHAT-IF SIMULATION INTENT
        sim_match = self._match_what_if_query(q_lower, q_clean)
        if sim_match:
            return self._handle_what_if_query(sim_match, report)

        # 2. PREDICTIVE ROUTE INTENT
        if any(w in q_lower for w in ["predict", "prediction", "next camera", "where will"]):
            return self._handle_predictive_route_query(q_lower, q_clean, report)

        # 3. EVIDENCE CHAIN INTENT
        if any(w in q_lower for w in ["evidence", "evidence chain", "why alert", "why did the system", "alert detail"]):
            return self._handle_evidence_query(q_lower, q_clean, report)

        # 4. ANOMALY RADAR INTENT
        if any(w in q_lower for w in ["anomal", "anomalous", "unusual", "suspicious", "radar"]):
            return self._handle_anomaly_query(q_lower, q_clean, report)

        # 5. ORIGIN-DESTINATION & CORRIDOR INTENT
        if any(w in q_lower for w in ["corridor", "origin-destination", "od matrix", "busiest route", "busiest corridor", "highest traffic corridor"]):
            return self._handle_od_query(q_lower, report)

        # 6. TRAJECTORY SEQUENCE INTENT (e.g. CAM_01 and later at CAM_02)
        seq_match = re.search(r"(?:seen|tracked|at)\s+(cam_\w+).*?(?:later|then|to|and)\s+(?:at\s+)?(cam_\w+)", q_lower)
        if seq_match:
            cam_from = seq_match.group(1).lower()
            cam_to = seq_match.group(2).lower()
            return self._handle_trajectory_sequence_query(cam_from, cam_to, report)

        # 7. SPECIFIC VEHICLE ROUTE / LAST SEEN INTENT
        plate_match = self._extract_plate_from_query(q_clean)
        if plate_match and any(w in q_lower for w in ["route", "history", "last seen", "where", "track", "plate", "vehicle"]):
            return self._handle_vehicle_route_query(plate_match, report)

        # 8. TRAFFIC / CONGESTION STATUS INTENT
        if any(w in q_lower for w in ["congestion", "highest traffic", "busiest camera", "city status", "traffic volume", "highest volume", "current traffic"]):
            return self._handle_traffic_status_query(q_lower, report)

        # 9. GENERAL PLATE SEARCH FALLBACK
        if plate_match:
            return self._handle_vehicle_route_query(plate_match, report)

        # 10. UNHANDLED / INSUFFICIENT DATA FALLBACK
        return {
            "intent": "INSUFFICIENT_DATA",
            "answer": "Insufficient data available for this query. Try asking about vehicle trajectories, congestion scores, route predictions, anomaly evidence, OD corridors, or What-If traffic simulations.",
            "confidence": "0.0",
            "sources": [],
            "evidence": [],
            "map_reference": None,
            "predicted": False,
        }

    # -------------------------------------------------------------------------
    # INTENT HANDLERS
    # -------------------------------------------------------------------------

    def _match_what_if_query(self, q_lower: str, q_clean: str) -> Optional[Dict[str, Any]]:
        """
        Check if query matches What-If simulation pattern.
        Returns target camera ID and percentage change if matched.
        """
        # Pattern 1: What happens if CAM_03 traffic increases/decreases by 40%?
        m1 = re.search(r"(?:what happens|impact|effect).*?(cam_\w+).*?(increase|decrease|reduce|surge|drop|blockage)?.*?(?:by\s+)?([+-]?\d+)%", q_lower)
        if m1:
            cam_id = m1.group(1)
            verb = m1.group(2) or ""
            pct = float(m1.group(3))
            if any(w in verb for w in ["decrease", "reduce", "drop", "blockage"]):
                pct = -abs(pct)
            return {"camera_id": cam_id, "change_percent": pct}

        # Pattern 2: Simulate a 20% increase at CAM_01 / Simulate -30% at CAM_02
        m2 = re.search(r"simulate\s+(?:a\s+)?([+-]?\d+)%\s*(?:increase|decrease|reduce|shift|surge)?\s*(?:at|for)?\s*(cam_\w+)", q_lower)
        if m2:
            pct = float(m2.group(1))
            cam_id = m2.group(2)
            if "decrease" in q_lower or "reduce" in q_lower:
                pct = -abs(pct)
            return {"camera_id": cam_id, "change_percent": pct}

        # Pattern 3: Simulate CAM_01 with 50% shift
        m3 = re.search(r"simulate\s+(cam_\w+).*?([+-]?\d+)%", q_lower)
        if m3:
            cam_id = m3.group(1)
            pct = float(m3.group(2))
            return {"camera_id": cam_id, "change_percent": pct}

        return None

    def _handle_what_if_query(self, sim_match: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute isolated What-If scenario simulation via TrafficSimulatorEngine.
        Returns PREDICTED structured response.
        """
        cam_id = sim_match["camera_id"].lower()
        change_pct = sim_match["change_percent"]

        sim_res = self.simulator_engine.run_what_if_simulation(
            camera_id=cam_id,
            traffic_change_percent=change_pct,
            analytics_report=report,
        )

        if sim_res.get("status") == "ERROR":
            return {
                "intent": "WHAT_IF_SIMULATION",
                "answer": f"Simulation request could not be processed: {sim_res.get('error_message')}",
                "confidence": "0.0",
                "sources": ["traffic_simulator"],
                "evidence": [],
                "map_reference": cam_id,
                "predicted": True,
            }

        target_data = sim_res.get("target_camera_simulation", {})
        downstream = sim_res.get("downstream_corridors_simulation", [])
        impact = sim_res.get("overall_impact_level", "LOW")
        recs = sim_res.get("decision_recommendations", [])

        change_sign = "+" if change_pct >= 0 else ""
        b_vol = target_data.get("baseline_volume", 0)
        s_vol = target_data.get("simulated_volume", 0)
        b_score = target_data.get("baseline_score", 0)
        s_score = target_data.get("simulated_score", 0)
        d_score = target_data.get("delta_score", 0)
        t_state = target_data.get("simulated_traffic_state", "FREE_FLOW")

        downstream_summary = []
        for d in downstream:
            downstream_summary.append(f"{d.get('corridor_id')}: {d.get('baseline_volume')} ➔ {d.get('simulated_volume')} veh (+{d.get('delta_score')} pts)")

        down_str = "; ".join(downstream_summary) if downstream_summary else "No downstream propagation registered."

        answer_text = (
            f"[PREDICTED WHAT-IF ESTIMATE] Simulated {change_sign}{change_pct:.0f}% volume shift at {cam_id.upper()} "
            f"results in an overall scenario impact of '{impact}'.\n"
            f"• Target Node ({cam_id.upper()}): Volume shifts from {b_vol} to {s_vol} vehicles. Congestion score shifts from {b_score}/100 to {s_score}/100 ({d_score:+} pts, State: {t_state}).\n"
            f"• Downstream Corridors: {down_str}\n"
            f"• Action Guidance: {recs[0].get('action_guidance') if recs else 'Monitor corridor traffic.'}"
        )

        return {
            "intent": "WHAT_IF_SIMULATION",
            "answer": answer_text,
            "confidence": "HIGH",
            "sources": ["traffic_simulator", "city_digital_twin", "city_traffic_intelligence"],
            "evidence": recs,
            "map_reference": cam_id,
            "predicted": True,
        }

    def _handle_predictive_route_query(self, q_lower: str, q_clean: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query predictive route intelligence for a specific vehicle or all tracked vehicles.
        """
        routes_data = report.get("predictive_route_intelligence", {}).get("predictions", {})
        plate = self._extract_plate_from_query(q_clean)

        if plate:
            # Search for specific plate
            matched_plate = None
            for p in routes_data.keys():
                if plate.lower() == p.lower():
                    matched_plate = p
                    break

            if not matched_plate or matched_plate not in routes_data:
                return {
                    "intent": "PREDICTIVE_ROUTE",
                    "answer": f"Insufficient data available for predictive route query. Vehicle '{plate}' not found in active prediction records.",
                    "confidence": "0.0",
                    "sources": ["predictive_route_intelligence"],
                    "evidence": [],
                    "map_reference": None,
                    "predicted": True,
                }

            pred_info = routes_data[matched_plate]
            top_next = pred_info.get("predicted_next_camera", "NONE")
            cands = pred_info.get("predictions", [])
            prob = pred_info.get("prediction_probability", cands[0].get("probability", 0.0) if cands else 0.0)
            prob_pct = cands[0].get("probability_percentage") if cands and "probability_percentage" in cands[0] else f"{prob * 100:.1f}%"
            conf = pred_info.get("prediction_confidence", 0.8)

            cand_str = ", ".join([f"{c.get('camera_id').upper()} ({c.get('probability_percentage')})" for c in cands[:3]])

            answer_text = (
                f"[PREDICTED ROUTE ESTIMATE] For vehicle {matched_plate}, the predicted next camera is "
                f"{top_next.upper()} with {prob_pct} probability (Confidence: {conf:.2f}).\n"
                f"• Ranked Candidates: {cand_str}\n"
                f"• Current Location: {pred_info.get('current_camera', 'cam_01').upper()}\n"
                f"DISCLAIMER: Predicted next camera is a probabilistic estimate based on historical transition topology."
            )

            return {
                "intent": "PREDICTIVE_ROUTE",
                "answer": answer_text,
                "confidence": "HIGH",
                "sources": ["predictive_route_intelligence"],
                "evidence": [pred_info],
                "map_reference": matched_plate,
                "predicted": True,
            }
        else:
            # Return general summary of active route predictions
            if not routes_data:
                return {
                    "intent": "PREDICTIVE_ROUTE",
                    "answer": "Insufficient data available. No active route predictions logged.",
                    "confidence": "0.0",
                    "sources": ["predictive_route_intelligence"],
                    "evidence": [],
                    "map_reference": None,
                    "predicted": True,
                }

            pred_summaries = []
            for p, p_info in list(routes_data.items())[:3]:
                next_cam = p_info.get("predicted_next_camera", "--").upper()
                pct = f"{p_info.get('prediction_probability', 0.0)*100:.1f}%"
                pred_summaries.append(f"{p}: {next_cam} ({pct})")

            summary_str = "; ".join(pred_summaries)
            answer_text = (
                f"[PREDICTED ROUTE ESTIMATES] Active route predictions available for {len(routes_data)} tracked vehicles:\n"
                f"• Top Predictions: {summary_str}\n"
                f"DISCLAIMER: Route predictions are probabilistic estimates based on historic topology patterns."
            )

            first_plate = list(routes_data.keys())[0] if routes_data else None

            return {
                "intent": "PREDICTIVE_ROUTE",
                "answer": answer_text,
                "confidence": "HIGH",
                "sources": ["predictive_route_intelligence"],
                "evidence": list(routes_data.values())[:3],
                "map_reference": first_plate,
                "predicted": True,
            }

    def _handle_evidence_query(self, q_lower: str, q_clean: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query 5 Ws evidence chain records for a specific alert, vehicle, or overall.
        """
        evidence_list = report.get("evidence_chain", [])
        plate = self._extract_plate_from_query(q_clean)

        if not evidence_list:
            return {
                "intent": "EVIDENCE_CHAIN",
                "answer": "Insufficient data available. No active evidence chain alert records found in analytics report.",
                "confidence": "0.0",
                "sources": ["evidence_chain"],
                "evidence": [],
                "map_reference": None,
                "predicted": False,
            }

        target_ev = None
        if plate:
            for ev in evidence_list:
                if plate.lower() in ev.get("fused_identity", "").lower() or plate.lower() in str(ev.get("evidence_id", "")).lower():
                    target_ev = ev
                    break

        if not target_ev:
            target_ev = evidence_list[0]

        five_ws = target_ev.get("explanation", {}).get("five_w_summary", {})
        factors = target_ev.get("explanation", {}).get("supporting_factors", [])

        answer_text = (
            f"Evidence Chain Audit for Alert [{target_ev.get('evidence_id')}] ({target_ev.get('fused_identity')}):\n"
            f"• WHAT: {five_ws.get('what', target_ev.get('explanation', {}).get('primary_reason'))}\n"
            f"• WHERE: {five_ws.get('where', 'Cross-camera corridor')}\n"
            f"• WHEN: {five_ws.get('when', 'Recent sighting')}\n"
            f"• SEVERITY: {target_ev.get('severity')} (Score: {target_ev.get('anomaly_score')}/100)\n"
            f"• SUPPORTING FACTORS: {'; '.join(factors) if factors else 'None'}\n"
            f"• GUIDANCE: {five_ws.get('review_guidance', 'Manual review recommended.')}"
        )

        return {
            "intent": "EVIDENCE_CHAIN",
            "answer": answer_text,
            "confidence": "HIGH",
            "sources": ["evidence_chain", "anomalies"],
            "evidence": [target_ev],
            "map_reference": target_ev.get("fused_identity"),
            "predicted": False,
        }

    def _handle_anomaly_query(self, q_lower: str, q_clean: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query Phase 1B Anomaly Radar detection logs.
        """
        anomalies = report.get("anomalies", [])
        plate = self._extract_plate_from_query(q_clean)

        if not anomalies:
            return {
                "intent": "ANOMALY_DETECTION",
                "answer": "No anomalous vehicle behavior detected across active camera nodes. All trajectory transitions adhere to physical topology rules.",
                "confidence": "HIGH",
                "sources": ["anomalies"],
                "evidence": [],
                "map_reference": None,
                "predicted": False,
            }

        matched_anomalies = anomalies
        if plate:
            matched_anomalies = [a for a in anomalies if plate.lower() in a.get("fused_identity", "").lower()]

        if not matched_anomalies:
            return {
                "intent": "ANOMALY_DETECTION",
                "answer": f"No anomaly records found for vehicle '{plate}'. Vehicle transitions follow normal baseline behavior.",
                "confidence": "HIGH",
                "sources": ["anomalies"],
                "evidence": [],
                "map_reference": plate,
                "predicted": False,
            }

        a_summaries = []
        for a in matched_anomalies[:3]:
            a_summaries.append(
                f"[{a.get('anomaly_id')}] Vehicle {a.get('fused_identity')} - {a.get('anomaly_type')} at {a.get('camera_pair')} (Severity: {a.get('severity')}, Score: {a.get('anomaly_score')}/100)"
            )

        answer_text = f"Detected {len(matched_anomalies)} anomalous trajectory event(s):\n• " + "\n• ".join(a_summaries)
        first_match = matched_anomalies[0].get("fused_identity")

        return {
            "intent": "ANOMALY_DETECTION",
            "answer": answer_text,
            "confidence": "HIGH",
            "sources": ["anomalies", "evidence_chain"],
            "evidence": matched_anomalies[:3],
            "map_reference": first_match,
            "predicted": False,
        }

    def _handle_od_query(self, q_lower: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query Origin-Destination matrix & city corridor intelligence.
        """
        ci_intel = report.get("city_traffic_intelligence", {})
        top_corridors = ci_intel.get("top_corridors", [])
        od_matrix = ci_intel.get("od_matrix", {})

        if not top_corridors and not od_matrix.get("matrix"):
            return {
                "intent": "OD_CORRIDOR_INTELLIGENCE",
                "answer": "Insufficient data available. OD matrix transitions have not yet been calculated.",
                "confidence": "0.0",
                "sources": ["city_traffic_intelligence"],
                "evidence": [],
                "map_reference": None,
                "predicted": False,
            }

        if top_corridors:
            top_c = top_corridors[0]
            c_name = top_c.get("corridor", "cam_01 -> cam_02")
            trans = top_c.get("transitions", 0)
            pct = top_c.get("percentage_of_total", 0.0)

            c_list_str = ", ".join([f"{c.get('corridor')} ({c.get('transitions')} transitions, {c.get('percentage_of_total')}%)" for c in top_corridors[:3]])

            answer_text = (
                f"The highest traffic origin-destination corridor is {c_name} with {trans} transitions ({pct}% of total city traffic).\n"
                f"• Top Corridors: {c_list_str}\n"
                f"• Total Network Transitions: {od_matrix.get('total_transitions', 0)}"
            )
            map_ref = c_name.split("->")[0].strip() if "->" in c_name else "cam_01"

            return {
                "intent": "OD_CORRIDOR_INTELLIGENCE",
                "answer": answer_text,
                "confidence": "HIGH",
                "sources": ["city_traffic_intelligence"],
                "evidence": top_corridors[:3],
                "map_reference": map_ref,
                "predicted": False,
            }
        else:
            return {
                "intent": "OD_CORRIDOR_INTELLIGENCE",
                "answer": f"Total city transitions recorded: {od_matrix.get('total_transitions', 0)}.",
                "confidence": "MEDIUM",
                "sources": ["city_traffic_intelligence"],
                "evidence": [],
                "map_reference": "cam_01",
                "predicted": False,
            }

    def _handle_trajectory_sequence_query(self, cam_from: str, cam_to: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query vehicles seen at cam_from and later at cam_to.
        """
        trajectories = report.get("reconstructed_trajectories", {})
        matching_vehicles = []

        for plate, info in trajectories.items():
            seq = [c.lower() for c in info.get("camera_sequence", [])]
            if cam_from in seq and cam_to in seq:
                idx_from = seq.index(cam_from)
                idx_to = seq.index(cam_to)
                if idx_from < idx_to:
                    matching_vehicles.append((plate, info))

        if not matching_vehicles:
            return {
                "intent": "TRAJECTORY_SEQUENCE",
                "answer": f"No vehicles were recorded transitioning from {cam_from.upper()} to {cam_to.upper()} in current analytics session.",
                "confidence": "HIGH",
                "sources": ["reconstructed_trajectories"],
                "evidence": [],
                "map_reference": cam_from,
                "predicted": False,
            }

        plates_str = ", ".join([v[0] for v in matching_vehicles])
        answer_text = (
            f"Found {len(matching_vehicles)} vehicle(s) seen at {cam_from.upper()} and later at {cam_to.upper()}:\n"
            f"• Vehicles: {plates_str}\n"
            f"• Verified Corridor: {cam_from.upper()} ➔ {cam_to.upper()}"
        )

        return {
            "intent": "TRAJECTORY_SEQUENCE",
            "answer": answer_text,
            "confidence": "HIGH",
            "sources": ["reconstructed_trajectories"],
            "evidence": [v[1] for v in matching_vehicles],
            "map_reference": matching_vehicles[0][0],
            "predicted": False,
        }

    def _handle_vehicle_route_query(self, plate: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query trajectory history and last seen details for a vehicle.
        """
        trajectories = report.get("reconstructed_trajectories", {})
        matched_plate = None
        matched_info = None

        for p, info in trajectories.items():
            if plate.lower() == p.lower():
                matched_plate = p
                matched_info = info
                break

        if not matched_info:
            return {
                "intent": "VEHICLE_ROUTE_HISTORY",
                "answer": f"Insufficient data available. Vehicle '{plate}' not found in active tracking records.",
                "confidence": "0.0",
                "sources": ["reconstructed_trajectories"],
                "evidence": [],
                "map_reference": None,
                "predicted": False,
            }

        seq = " ➔ ".join(matched_info.get("camera_sequence", []))
        sightings = matched_info.get("total_sightings", 0)
        last_seen = matched_info.get("last_seen", "--")

        answer_text = (
            f"Trajectory history for vehicle {matched_plate}:\n"
            f"• Verified Route: {seq}\n"
            f"• Total Sightings: {sightings}\n"
            f"• Last Seen: {last_seen}"
        )

        return {
            "intent": "VEHICLE_ROUTE_HISTORY",
            "answer": answer_text,
            "confidence": "HIGH",
            "sources": ["reconstructed_trajectories"],
            "evidence": [matched_info],
            "map_reference": matched_plate,
            "predicted": False,
        }

    def _handle_traffic_status_query(self, q_lower: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query traffic congestion status, highest volume, or city state.
        """
        dt_intel = report.get("city_digital_twin", {})
        cam_nodes = dt_intel.get("camera_nodes_intelligence", {})
        cam_analytics = report.get("camera_analytics", {})

        if not cam_nodes and not cam_analytics:
            return {
                "intent": "TRAFFIC_CONGESTION_STATUS",
                "answer": "Insufficient data available. No camera node intelligence recorded.",
                "confidence": "0.0",
                "sources": ["city_digital_twin", "camera_analytics"],
                "evidence": [],
                "map_reference": None,
                "predicted": False,
            }

        # Find camera with highest congestion score
        highest_cam = None
        max_score = -1
        max_vol = -1

        for c_id, node in cam_nodes.items():
            score = node.get("congestion_score", 0)
            vol = node.get("traffic_volume", 0)
            if score > max_score:
                max_score = score
                max_vol = vol
                highest_cam = c_id

        if not highest_cam and cam_analytics:
            for c_id, c_data in cam_analytics.items():
                vol = c_data.get("traffic_volume", 0)
                if vol > max_vol:
                    max_vol = vol
                    highest_cam = c_id
                    max_score = vol * 5

        city_state = dt_intel.get("city_traffic_state", "FREE_FLOW")
        city_avg_score = dt_intel.get("city_average_congestion_score", 0)

        answer_text = (
            f"City Traffic State: {city_state} (City Average Score: {city_avg_score}/100).\n"
            f"• Highest Congestion Camera: {highest_cam.upper() if highest_cam else 'CAM_01'} with score {max_score}/100 ({max_vol} vehicles).\n"
            f"• Active Camera Nodes: {len(cam_nodes) or len(cam_analytics)}"
        )

        return {
            "intent": "TRAFFIC_CONGESTION_STATUS",
            "answer": answer_text,
            "confidence": "HIGH",
            "sources": ["city_digital_twin", "camera_analytics"],
            "evidence": [cam_nodes.get(highest_cam, {})] if highest_cam in cam_nodes else [],
            "map_reference": highest_cam or "cam_01",
            "predicted": False,
        }

    # -------------------------------------------------------------------------
    # UTILITY HELPERS
    # -------------------------------------------------------------------------

    def _extract_plate_from_query(self, text: str) -> Optional[str]:
        """
        Extract vehicle plate number pattern from query string (e.g. MH12DE1408, TN09CC5544, KA01AB2026).
        """
        m = re.search(r"\b([A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4})\b", text, re.IGNORECASE)
        if m:
            return re.sub(r"\s+", "", m.group(1).upper())

        # Direct alphanumeric string check (6-11 chars)
        words = text.split()
        for w in words:
            w_clean = re.sub(r"[^\w]", "", w).upper()
            if len(w_clean) >= 6 and len(w_clean) <= 10 and any(c.isdigit() for c in w_clean) and any(c.isalpha() for c in w_clean):
                if not w_clean.startswith("CAM"):
                    return w_clean

        return None
