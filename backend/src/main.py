"""
FastAPI Backend Application Entrypoint (SIH26127)
--------------------------------------------------
Exposes REST and WebSocket endpoints for real-time video processing,
ANPR trajectory queries, urban traffic analytics, and GIS dashboard integration.
"""

import sys
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure module directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analytics import TrafficAnalyticsEngine, get_mock_multi_camera_data

app = FastAPI(
    title="City-Wide ANPR & Traffic Analytics Engine API",
    description="Backend API for Smart India Hackathon Problem Statement SIH26127 prototype.",
    version="0.1.0"
)

# Enable CORS for frontend dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analytics_engine = TrafficAnalyticsEngine()


@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "City-Wide ANPR Engine (SIH26127)",
        "version": "0.1.0"
    }


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


class TrajectoryRequest(BaseModel):
    plate_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@app.post("/api/trajectory/search")
def search_trajectory(req: TrajectoryRequest):
    """
    Query multi-camera vehicle trajectory by license plate ID.
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            report = json.load(f)
            trajectories = report.get("reconstructed_trajectories", {})
            if req.plate_id in trajectories:
                return {
                    "plate_id": req.plate_id,
                    "found": True,
                    "trajectory_data": trajectories[req.plate_id],
                }

    # Fallback search
    mock_data = get_mock_multi_camera_data()
    reconstructed = analytics_engine.reconstruct_trajectories(mock_data)
    if req.plate_id in reconstructed:
        return {
            "plate_id": req.plate_id,
            "found": True,
            "trajectory_data": {
                "camera_sequence": [n["camera_id"] for n in reconstructed[req.plate_id]],
                "total_sightings": len(reconstructed[req.plate_id]),
                "trajectory_nodes": reconstructed[req.plate_id],
            }
        }

    return {
        "plate_id": req.plate_id,
        "found": False,
        "message": f"No multi-camera trajectory records found for plate ID '{req.plate_id}'."
    }


@app.get("/api/analytics/summary")
def get_traffic_analytics():
    """
    Retrieve urban traffic metrics (vehicle counts, camera-wise flow, classification breakdown, congestion index, predictive route intelligence).
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)

    # Generate fresh analytics report if file doesn't exist
    mock_data = get_mock_multi_camera_data()
    return analytics_engine.generate_analytics_report(mock_data)


@app.get("/api/prediction/vehicle/{plate_id}")
def predict_vehicle_route(plate_id: str):
    """
    Query Predictive Route Intelligence for a specific license plate ID.
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            report = json.load(f)
            preds = report.get("predictive_route_intelligence", {}).get("predictions", {})
            if plate_id in preds:
                return preds[plate_id]

    try:
        from route_predictor import RoutePredictorEngine
        predictor = RoutePredictorEngine()
        mock_data = get_mock_multi_camera_data()
        trajectories = analytics_engine.reconstruct_trajectories(mock_data)
        if plate_id in trajectories:
            freq = predictor.compute_historical_frequencies({"data": {"trajectory_nodes": mock_data}})
            return predictor.predict_next_camera(plate_id, trajectories[plate_id], freq)
    except Exception as e:
        pass

    return {
        "vehicle_id": plate_id,
        "current_camera": None,
        "predicted_next_camera": None,
        "predictions": [],
        "prediction_confidence": 0.0,
        "status": "NOT_FOUND",
        "explanation": f"No prediction records available for vehicle '{plate_id}'.",
    }


@app.get("/api/evidence/chain")
def get_evidence_chain():
    """
    Retrieve complete Phase 1D Evidence Chain records for all flagged anomalies.
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            report = json.load(f)
            return {
                "total_records": len(report.get("evidence_chain", [])),
                "evidence_chain": report.get("evidence_chain", []),
            }
    return {"total_records": 0, "evidence_chain": []}


@app.get("/api/evidence/anomaly/{evidence_id}")
def get_anomaly_evidence_detail(evidence_id: str):
    """
    Query detailed 5 Ws evidence chain record for a specific anomaly evidence ID.
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            report = json.load(f)
            chain = report.get("evidence_chain", [])
            for item in chain:
                if item.get("evidence_id") == evidence_id:
                    return item
    return {"evidence_id": evidence_id, "found": False, "message": f"Evidence ID '{evidence_id}' not found."}


@app.get("/api/city/traffic-intelligence")
def get_city_traffic_intelligence():
    """
    Retrieve complete Phase 2A City-Wide Traffic Intelligence output (summary, OD matrix, top corridors, insights).
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            report = json.load(f)
            return report.get("city_traffic_intelligence", {})
    return {}


@app.get("/api/city/od-matrix")
def get_od_matrix():
    """
    Retrieve Phase 2A Origin-Destination (OD) matrix.
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            report = json.load(f)
            return report.get("city_traffic_intelligence", {}).get("od_matrix", {})
    return {"camera_nodes": [], "matrix": {}, "total_transitions": 0}


@app.get("/api/city/digital-twin")
def get_city_digital_twin():
    """
    Retrieve complete Phase 2B City Digital Twin state (city traffic state, average congestion score, camera node profiles).
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            report = json.load(f)
            return report.get("city_digital_twin", {})
    return {}


@app.get("/api/city/camera/{camera_id}/intelligence")
def get_camera_node_intelligence(camera_id: str):
    """
    Query spatial intelligence profile (traffic state, congestion score, volume, flow) for a specific camera node ID.
    """
    data_path = "backend/data/analytics_report.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            report = json.load(f)
            nodes_intel = report.get("city_digital_twin", {}).get("camera_nodes_intelligence", {})
            if camera_id in nodes_intel:
                return nodes_intel[camera_id]

    return {
        "camera_id": camera_id,
        "found": False,
        "message": f"Camera node '{camera_id}' not found in Digital Twin state.",
    }


class SimulationRequest(BaseModel):
    camera_id: str
    traffic_change_percent: float


@app.post("/api/traffic/simulate")
def run_traffic_simulation(req: SimulationRequest):
    """
    Execute Phase 2C What-If Traffic Simulation scenario and generate decision intelligence.
    """
    try:
        from traffic_simulator import TrafficSimulatorEngine
        simulator = TrafficSimulatorEngine()

        data_path = "backend/data/analytics_report.json"
        report = {}
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                report = json.load(f)
        else:
            mock_data = get_mock_multi_camera_data()
            report = analytics_engine.generate_analytics_report(mock_data)

        res = simulator.run_what_if_simulation(
            camera_id=req.camera_id,
            traffic_change_percent=req.traffic_change_percent,
            analytics_report=report,
        )
        if res.get("status") == "ERROR":
            raise HTTPException(status_code=400, detail=res.get("error_message"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


class CopilotQueryRequest(BaseModel):
    query: str


@app.post("/api/copilot/query")
def process_copilot_query(req: CopilotQueryRequest):
    """
    Process Phase 3 AI City Copilot natural language analytics query.
    """
    try:
        from city_copilot import CityCopilotEngine
        copilot = CityCopilotEngine()

        data_path = "backend/data/analytics_report.json"
        report = {}
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                report = json.load(f)
        else:
            mock_data = get_mock_multi_camera_data()
            report = analytics_engine.generate_analytics_report(mock_data)

        res = copilot.process_query(query_text=req.query, analytics_report=report)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Copilot query error: {str(e)}")


class RoleUpdateRequest(BaseModel):
    role: str
    actor: Optional[str] = "OPERATOR"
    reason: Optional[str] = "Dashboard access role elevation"


@app.get("/api/security/status")
def get_security_status():
    """
    Retrieve Phase 4 Security Shield status, active access role, and compliance details.
    """
    try:
        from privacy_security import PrivacySecurityEngine
        sec_engine = PrivacySecurityEngine()
        return sec_engine.get_security_status_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Security status error: {str(e)}")


@app.get("/api/security/audit-logs")
def get_security_audit_logs():
    """
    Retrieve tamper-evident security audit logs.
    """
    try:
        from privacy_security import PrivacySecurityEngine
        sec_engine = PrivacySecurityEngine()
        return {
            "total_audit_logs": len(sec_engine.audit_logs),
            "audit_logs": sec_engine.audit_logs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit log error: {str(e)}")


@app.post("/api/security/access-level")
def set_security_access_level(req: RoleUpdateRequest):
    """
    Update security access level (PUBLIC_OPERATOR, LAW_ENFORCEMENT, SYSTEM_ADMIN) with audit log.
    """
    try:
        from privacy_security import PrivacySecurityEngine
        sec_engine = PrivacySecurityEngine()
        res = sec_engine.set_access_role(role=req.role, actor=req.actor or "OPERATOR", reason=req.reason or "Role update")
        if res.get("status") == "ERROR":
            raise HTTPException(status_code=400, detail=res.get("error_message"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Role update error: {str(e)}")


@app.get("/api/security/forensic-command-center")
def get_forensic_command_center_summary():
    """
    Retrieve Phase 5 Security & Forensic Investigation Command Center aggregated state.
    Provides security status, tamper-evident audit logs, 5 Ws evidence chain, anomaly records,
    and privacy-wrapped vehicle trajectories with SHA-256 hashed vehicle IDs.
    """
    try:
        from privacy_security import PrivacySecurityEngine
        sec_engine = PrivacySecurityEngine()

        data_path = "backend/data/analytics_report.json"
        report = {}
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                report = json.load(f)
        else:
            mock_data = get_mock_multi_camera_data()
            report = analytics_engine.generate_analytics_report(mock_data)

        wrapped_report = sec_engine.apply_privacy_wrapper(report)

        return {
            "status": "SUCCESS",
            "timestamp": datetime.now().isoformat() if "datetime" in globals() else None,
            "security_status": sec_engine.get_security_status_summary(),
            "audit_logs": sec_engine.audit_logs,
            "evidence_chain": wrapped_report.get("evidence_chain", []),
            "anomalies": wrapped_report.get("anomaly_radar", {}).get("anomaly_records", []),
            "reconstructed_trajectories": wrapped_report.get("reconstructed_trajectories", {}),
            "predictive_route_intelligence": wrapped_report.get("predictive_route_intelligence", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic Command Center error: {str(e)}")
