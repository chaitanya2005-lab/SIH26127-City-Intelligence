"""
Phase 1C — Predictive Route Intelligence Engine (SIH26127)
-----------------------------------------------------------
Predicts the most likely NEXT camera node for tracked vehicle trajectories using:
1. Historical camera transition frequencies across dataset
2. Recent trajectory route sequence
3. Spatial-temporal camera topology graph adjacency
4. Temporal compatibility

NOTE: This prototype uses deterministic evidence-based transition ranking.
Predictions are clearly status-labeled as "PREDICTED" (probabilistic estimates, not guarantees).

================================================================================
HOW TO RUN THIS MODULE:
================================================================================
python backend/src/route_predictor.py
================================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure src path is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anomaly_radar import DEFAULT_CAMERA_TOPOLOGY


class RoutePredictorEngine:
    def __init__(self, camera_topology: Optional[Dict[str, List[str]]] = None):
        """
        Initialize Predictive Route Intelligence Engine.

        :param camera_topology: Map of allowed next nodes per camera node.
        """
        self.topology = camera_topology or DEFAULT_CAMERA_TOPOLOGY

    def compute_historical_frequencies(
        self,
        reconstructed_trajectories: Dict[str, Any]
    ) -> Dict[str, Dict[str, int]]:
        """
        Calculate historical transition frequencies (count of A -> B transitions) across dataset.
        """
        freq_matrix: Dict[str, Dict[str, int]] = {}

        for plate, info in reconstructed_trajectories.items():
            nodes = info.get("trajectory_nodes", [])
            seq = info.get("camera_sequence", [])

            if not seq and nodes:
                seq = [n.get("camera_id") for n in nodes if n.get("camera_id")]

            for i in range(len(seq) - 1):
                src = seq[i]
                dst = seq[i + 1]
                if src and dst and src != dst:
                    if src not in freq_matrix:
                        freq_matrix[src] = {}
                    freq_matrix[src][dst] = freq_matrix[src].get(dst, 0) + 1

        return freq_matrix

    def predict_next_camera(
        self,
        plate_or_id: str,
        trajectory_nodes: List[Dict[str, Any]],
        historical_frequencies: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Predict the most likely next camera node for a given vehicle trajectory.

        :param plate_or_id: License plate / vehicle identifier.
        :param trajectory_nodes: Chronological list of trajectory node dicts.
        :param historical_frequencies: Pre-computed or global transition frequency matrix.
        :return: Dict containing vehicle_id, current_camera, predictions, prediction_confidence, status.
        """
        if not trajectory_nodes:
            return {
                "vehicle_id": plate_or_id,
                "current_camera": None,
                "predicted_next_camera": None,
                "predictions": [],
                "prediction_confidence": 0.0,
                "status": "NO_TRAJECTORY",
                "explanation": "No trajectory data available for prediction.",
            }

        # Sort nodes by timestamp if available
        sorted_nodes = sorted(
            trajectory_nodes,
            key=lambda n: n.get("timestamp", "") if n.get("timestamp") else ""
        )
        current_node = sorted_nodes[-1]
        current_cam = current_node.get("camera_id")

        if not current_cam:
            return {
                "vehicle_id": plate_or_id,
                "current_camera": None,
                "predicted_next_camera": None,
                "predictions": [],
                "prediction_confidence": 0.0,
                "status": "UNKNOWN_CAMERA",
                "explanation": "Current camera node is unknown.",
            }

        # Get candidate next cameras from camera topology graph
        candidate_cams = list(self.topology.get(current_cam, []))

        # Dynamically include candidates from historical transitions if present
        freq_for_cam = (historical_frequencies or {}).get(current_cam, {})
        for dest in freq_for_cam.keys():
            if dest not in candidate_cams and dest != current_cam:
                candidate_cams.append(dest)

        if not candidate_cams:
            return {
                "vehicle_id": plate_or_id,
                "current_camera": current_cam,
                "predicted_next_camera": None,
                "predictions": [],
                "prediction_confidence": 0.0,
                "status": "TERMINAL_NODE",
                "explanation": f"No adjacent downstream camera nodes configured for '{current_cam}'.",
            }

        # Calculate raw weights for candidates using historical frequencies & topology heuristics
        total_historical_transitions = sum(freq_for_cam.values())
        raw_scores: Dict[str, float] = {}

        for cand in candidate_cams:
            hist_count = freq_for_cam.get(cand, 0)
            if total_historical_transitions > 0:
                hist_prob = hist_count / float(total_historical_transitions)
            else:
                hist_prob = 1.0 / float(len(candidate_cams))

            # Topology adjacency bonus
            topology_bonus = 0.2 if cand in self.topology.get(current_cam, []) else 0.0

            # Combined weight formula
            weight = 0.7 * hist_prob + 0.3 * (1.0 / len(candidate_cams) + topology_bonus)
            raw_scores[cand] = weight

        # Normalize probabilities so they total approximately 1.0 (100%)
        sum_weights = sum(raw_scores.values()) or 1.0
        ranked_predictions: List[Dict[str, Any]] = []

        for cand, weight in raw_scores.items():
            prob = round(weight / sum_weights, 2)
            hist_count = freq_for_cam.get(cand, 0)

            if hist_count > 0:
                reason = f"Frequent historical transition ({hist_count} sightings) from {current_cam}"
            elif cand in self.topology.get(current_cam, []):
                reason = f"Configured topology adjacency corridor from {current_cam}"
            else:
                reason = f"Possible downstream transition from {current_cam}"

            ranked_predictions.append({
                "camera_id": cand,
                "probability": prob,
                "probability_percentage": f"{int(round(prob * 100))}%",
                "reason": reason,
            })

        # Sort predictions descending by probability
        ranked_predictions = sorted(ranked_predictions, key=lambda p: p["probability"], reverse=True)

        # Normalize exact sum to 1.0
        prob_sum = sum(p["probability"] for p in ranked_predictions)
        if prob_sum > 0 and len(ranked_predictions) > 0:
            diff = round(1.0 - prob_sum, 2)
            ranked_predictions[0]["probability"] = round(ranked_predictions[0]["probability"] + diff, 2)
            ranked_predictions[0]["probability_percentage"] = f"{int(round(ranked_predictions[0]['probability'] * 100))}%"

        # Overall prediction confidence
        top_prob = ranked_predictions[0]["probability"] if ranked_predictions else 0.0
        traj_len_factor = min(1.0, 0.7 + 0.15 * len(sorted_nodes))
        prediction_confidence = round(top_prob * traj_len_factor, 2)

        return {
            "vehicle_id": plate_or_id,
            "current_camera": current_cam,
            "predicted_next_camera": ranked_predictions[0]["camera_id"] if ranked_predictions else None,
            "predictions": ranked_predictions,
            "prediction_confidence": prediction_confidence,
            "status": "PREDICTED",
            "explanation": f"{current_cam} -> {ranked_predictions[0]['camera_id']} is the most compatible transition ({ranked_predictions[0]['probability_percentage']}).",
        }

    def run_route_predictions(
        self,
        reconstructed_trajectories: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Predictive Route Intelligence across all vehicle trajectories.
        """
        freq_matrix = self.compute_historical_frequencies(reconstructed_trajectories)
        predictions_map: Dict[str, Any] = {}

        for plate, info in reconstructed_trajectories.items():
            nodes = info.get("trajectory_nodes", [])
            pred_res = self.predict_next_camera(plate, nodes, freq_matrix)
            predictions_map[plate] = pred_res

        return {
            "prediction_timestamp": datetime.now().isoformat(),
            "topology_source": "DEMO/SIMULATED camera topology",
            "total_vehicles_predicted": len(predictions_map),
            "historical_transition_frequencies": freq_matrix,
            "predictions": predictions_map,
        }


def main():
    parser = argparse.ArgumentParser(
        description="SIH26127 - Phase 1C Predictive Route Intelligence Engine."
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
    else:
        print(f"[WARNING] File not found: '{args.report}'. Running demo prediction scan.")
        trajectories = {}

    engine = RoutePredictorEngine()
    result = engine.run_route_predictions(trajectories)

    print("\n================================================================================")
    print("                PHASE 1C PREDICTIVE ROUTE INTELLIGENCE SUMMARY                  ")
    print("================================================================================")
    print(f"Topology Source            : {result['topology_source']}")
    print(f"Total Vehicles Predicted   : {result['total_vehicles_predicted']}")
    print("--------------------------------------------------------------------------------")
    for plate, pred in result["predictions"].items():
        print(f"Vehicle: {plate} | Current: {pred['current_camera']} | Predicted Next: {pred['predicted_next_camera']} (Conf: {pred['prediction_confidence']})")
        for p in pred["predictions"]:
            print(f"   -> {p['camera_id']:<8}: {p['probability_percentage']:<5} | Reason: {p['reason']}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
