"""
City-Wide Traffic Intelligence & Origin-Destination (OD) Matrix Module (SIH26127)
----------------------------------------------------------------------------------
Derives city-level traffic flow intelligence, Origin-Destination transition matrices,
corridor rankings, and deterministic traffic insights from multi-camera vehicle trajectory records.

Part of SIH26127 Phase 2A Enhancement.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class CityTrafficIntelligenceEngine:
    """
    Engine for calculating city-wide traffic metrics, OD transition matrix,
    top traffic corridors, camera-level flows, and data-backed traffic insights.
    """

    def __init__(self, default_camera_nodes: Optional[List[str]] = None):
        """
        Initialize engine with optional default camera topology list.
        """
        self.default_camera_nodes = default_camera_nodes or ["cam_01", "cam_02", "cam_03", "cam_04"]

    def extract_camera_nodes(
        self,
        reconstructed_trajectories: Dict[str, Any],
        camera_analytics: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Extract complete list of unique camera node IDs present in trajectories,
        analytics, or defaults.
        """
        nodes = set(self.default_camera_nodes)

        if camera_analytics:
            nodes.update(camera_analytics.keys())

        if reconstructed_trajectories:
            for plate, data in reconstructed_trajectories.items():
                if isinstance(data, dict):
                    seq = data.get("camera_sequence", [])
                    nodes.update(seq)
                    tnodes = data.get("trajectory_nodes", [])
                    for n in tnodes:
                        if isinstance(n, dict) and n.get("camera_id"):
                            nodes.add(n.get("camera_id"))
                elif isinstance(data, list):
                    for n in data:
                        if isinstance(n, dict) and n.get("camera_id"):
                            nodes.add(n.get("camera_id"))

        return sorted(list(nodes))

    def build_od_matrix(
        self,
        reconstructed_trajectories: Dict[str, Any],
        camera_nodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build Origin-Destination matrix mapping origin camera nodes to destination camera nodes.
        Self-loops (consecutive detections at the exact same camera) are excluded.
        """
        nodes = camera_nodes or self.extract_camera_nodes(reconstructed_trajectories)
        matrix: Dict[str, Dict[str, int]] = {orig: {dest: 0 for dest in nodes} for orig in nodes}

        if not reconstructed_trajectories:
            return {"camera_nodes": nodes, "matrix": matrix, "total_transitions": 0}

        total_transitions = 0

        for plate, data in reconstructed_trajectories.items():
            t_nodes = []
            if isinstance(data, dict):
                t_nodes = data.get("trajectory_nodes", [])
            elif isinstance(data, list):
                t_nodes = data

            if not t_nodes or len(t_nodes) < 2:
                continue

            # Sort nodes chronologically
            sorted_nodes = sorted(
                [n for n in t_nodes if isinstance(n, dict) and n.get("camera_id")],
                key=lambda x: str(x.get("timestamp", ""))
            )

            for i in range(len(sorted_nodes) - 1):
                cam_from = sorted_nodes[i].get("camera_id")
                cam_to = sorted_nodes[i + 1].get("camera_id")

                # Exclude repeated detections at the same camera node
                if cam_from and cam_to and cam_from != cam_to:
                    if cam_from not in matrix:
                        matrix[cam_from] = {dest: 0 for dest in nodes}
                    if cam_to not in matrix[cam_from]:
                        matrix[cam_from][cam_to] = 0

                    matrix[cam_from][cam_to] += 1
                    total_transitions += 1

        return {
            "camera_nodes": nodes,
            "matrix": matrix,
            "total_transitions": total_transitions,
        }

    def compute_top_corridors(
        self,
        od_matrix_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Calculate top traffic corridors ordered by vehicle transition count descending.
        Calculates percentage of total inter-camera flow dynamically.
        """
        matrix = od_matrix_data.get("matrix", {})
        total_transitions = od_matrix_data.get("total_transitions", 0)

        corridors = []
        for orig, dest_dict in matrix.items():
            for dest, count in dest_dict.items():
                if orig != dest and count > 0:
                    pct = round((count / total_transitions * 100.0), 1) if total_transitions > 0 else 0.0
                    corridors.append({
                        "origin": orig,
                        "destination": dest,
                        "corridor_id": f"{orig} -> {dest}",
                        "vehicle_count": count,
                        "percentage_of_total_flow": pct,
                    })

        # Sort corridors by vehicle_count descending
        corridors.sort(key=lambda c: (c["vehicle_count"], c["origin"], c["destination"]), reverse=True)

        # Assign rankings
        for rank, c in enumerate(corridors, 1):
            c["rank"] = rank

        return corridors

    def compute_camera_flow_summary(
        self,
        reconstructed_trajectories: Dict[str, Any],
        camera_analytics: Optional[Dict[str, Any]] = None,
        camera_nodes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute incoming, outgoing, and total volume summary per camera location.
        """
        nodes = camera_nodes or self.extract_camera_nodes(reconstructed_trajectories, camera_analytics)
        flows: Dict[str, Dict[str, Any]] = {
            cam: {
                "camera_id": cam,
                "incoming_flow": 0,
                "outgoing_flow": 0,
                "total_volume": 0,
                "unique_vehicles": 0,
                "congestion_level": "LOW",
            }
            for cam in nodes
        }

        # Populate total volume and congestion from camera_analytics if available
        if camera_analytics:
            for cam, stats in camera_analytics.items():
                if cam in flows:
                    flows[cam]["total_volume"] = stats.get("traffic_volume", 0)
                    flows[cam]["unique_vehicles"] = stats.get("unique_vehicles", 0)
                    flows[cam]["congestion_level"] = stats.get("congestion_level", "LOW")

        # Compute inter-camera transition incoming and outgoing flows
        if reconstructed_trajectories:
            for plate, data in reconstructed_trajectories.items():
                t_nodes = []
                if isinstance(data, dict):
                    t_nodes = data.get("trajectory_nodes", [])
                elif isinstance(data, list):
                    t_nodes = data

                if not t_nodes:
                    continue

                sorted_nodes = sorted(
                    [n for n in t_nodes if isinstance(n, dict) and n.get("camera_id")],
                    key=lambda x: str(x.get("timestamp", ""))
                )

                for i in range(len(sorted_nodes) - 1):
                    cam_from = sorted_nodes[i].get("camera_id")
                    cam_to = sorted_nodes[i + 1].get("camera_id")
                    if cam_from and cam_to and cam_from != cam_to:
                        if cam_from in flows:
                            flows[cam_from]["outgoing_flow"] += 1
                        if cam_to in flows:
                            flows[cam_to]["incoming_flow"] += 1

        return flows

    def generate_deterministic_insights(
        self,
        od_matrix_data: Dict[str, Any],
        top_corridors: List[Dict[str, Any]],
        camera_flows: Dict[str, Dict[str, Any]],
        camera_analytics: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate deterministic insights supported strictly by empirical metrics.
        Insight Types: HIGHEST_FLOW_CORRIDOR, HIGHEST_ORIGIN, HIGHEST_DESTINATION,
                       LOW_FLOW_CAMERA, CONGESTION_HOTSPOT.
        """
        insights = []

        # 1. HIGHEST_FLOW_CORRIDOR
        if top_corridors:
            top = top_corridors[0]
            insights.append({
                "type": "HIGHEST_FLOW_CORRIDOR",
                "title": f"Busiest Corridor: {top['origin']} -> {top['destination']}",
                "description": f"Corridor {top['origin']} -> {top['destination']} registered the highest vehicle flow with {top['vehicle_count']} transitions ({top['percentage_of_total_flow']}% of city-wide inter-camera traffic).",
                "metric_key": "vehicle_count",
                "metric_value": top["vehicle_count"],
                "origin": top["origin"],
                "destination": top["destination"],
            })

        # 2. HIGHEST_ORIGIN
        highest_orig = None
        max_out = -1
        for cam, flow in camera_flows.items():
            if flow["outgoing_flow"] > max_out and flow["outgoing_flow"] > 0:
                max_out = flow["outgoing_flow"]
                highest_orig = cam

        if highest_orig:
            insights.append({
                "type": "HIGHEST_ORIGIN",
                "title": f"Primary Traffic Origin: {highest_orig}",
                "description": f"Camera node '{highest_orig}' generated the highest outbound vehicle flow with {max_out} departing transitions.",
                "metric_key": "outgoing_flow",
                "metric_value": max_out,
                "camera_id": highest_orig,
            })

        # 3. HIGHEST_DESTINATION
        highest_dest = None
        max_in = -1
        for cam, flow in camera_flows.items():
            if flow["incoming_flow"] > max_in and flow["incoming_flow"] > 0:
                max_in = flow["incoming_flow"]
                highest_dest = cam

        if highest_dest:
            insights.append({
                "type": "HIGHEST_DESTINATION",
                "title": f"Primary Traffic Destination: {highest_dest}",
                "description": f"Camera node '{highest_dest}' received the highest inbound vehicle flow with {max_in} arriving transitions.",
                "metric_key": "incoming_flow",
                "metric_value": max_in,
                "camera_id": highest_dest,
            })

        # 4. CONGESTION_HOTSPOT
        hotspots = []
        if camera_analytics:
            for cam, stats in camera_analytics.items():
                level = stats.get("congestion_level", "LOW")
                if level in ["MODERATE", "HIGH"]:
                    hotspots.append((cam, stats.get("traffic_volume", 0), level))

        if hotspots:
            hotspots.sort(key=lambda x: x[1], reverse=True)
            top_spot = hotspots[0]
            insights.append({
                "type": "CONGESTION_HOTSPOT",
                "title": f"Congestion Alert: {top_spot[0]} ({top_spot[2]})",
                "description": f"Camera node '{top_spot[0]}' is experiencing {top_spot[2]} congestion with {top_spot[1]} detected vehicle events.",
                "metric_key": "traffic_volume",
                "metric_value": top_spot[1],
                "camera_id": top_spot[0],
            })

        # 5. LOW_FLOW_CAMERA
        low_cam = None
        min_vol = 999999
        for cam, flow in camera_flows.items():
            vol = flow.get("total_volume", 0)
            if vol < min_vol:
                min_vol = vol
                low_cam = cam

        if low_cam and min_vol < 999999:
            insights.append({
                "type": "LOW_FLOW_CAMERA",
                "title": f"Low Volume Node: {low_cam}",
                "description": f"Camera node '{low_cam}' recorded minimum activity with {min_vol} total vehicle detection events.",
                "metric_key": "total_volume",
                "metric_value": min_vol,
                "camera_id": low_cam,
            })

        return insights

    def compute_city_traffic_intelligence(
        self,
        reconstructed_trajectories: Dict[str, Any],
        camera_analytics: Optional[Dict[str, Any]] = None,
        camera_nodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete City-Wide Traffic Intelligence analysis.
        """
        nodes = camera_nodes or self.extract_camera_nodes(reconstructed_trajectories, camera_analytics)
        od_matrix_data = self.build_od_matrix(reconstructed_trajectories, camera_nodes=nodes)
        top_corridors = self.compute_top_corridors(od_matrix_data)
        camera_flows = self.compute_camera_flow_summary(reconstructed_trajectories, camera_analytics, camera_nodes=nodes)
        insights = self.generate_deterministic_insights(od_matrix_data, top_corridors, camera_flows, camera_analytics)

        total_obs = sum(stats.get("traffic_volume", 0) for stats in camera_analytics.values()) if camera_analytics else 0

        return {
            "summary": {
                "total_observations": total_obs,
                "unique_vehicles": len(reconstructed_trajectories) if reconstructed_trajectories else 0,
                "active_cameras": len(nodes),
                "total_inter_camera_transitions": od_matrix_data["total_transitions"],
                "unique_active_corridors": len(top_corridors),
            },
            "od_matrix": od_matrix_data,
            "top_corridors": top_corridors,
            "camera_flow_summary": camera_flows,
            "traffic_insights": insights,
        }
