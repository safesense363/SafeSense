"""SafeSense AI — FastAPI Application.

Provides RESTful endpoints for system health, real-time pipeline status,
event querying, event ingestion, and explainable risk configuration.
"""
from pathlib import Path
import threading
import time
import platform
from typing import Optional
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from backend.schemas import (
    HealthResponse,
    PipelineStatusResponse,
    EventPayload,
    EventListResponse,
    RiskConfigResponse,
)
from backend.event_store import EventStore
from risk.risk_analyzer import RiskAnalyzer

app = FastAPI(
    title="SafeSense AI — Incident & Risk Detection API",
    description="Backend API for real-time human activity recognition and explainable risk monitoring.",
    version="1.0.0",
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared instances
event_store = EventStore()
risk_analyzer = RiskAnalyzer()
MODEL_PATH = PROJECT_ROOT / "models" / "safesense_gru_baseline.pt"


@app.get("/", tags=["Root"])
def root():
    return {
        "project": "SafeSense AI",
        "description": "Real-time human activity recognition and explainable laboratory safety risk monitoring.",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            "/api/health",
            "/api/status",
            "/api/events",
            "/api/config/risk",
        ],
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def get_health():
    """Verify system health, model presence, and camera availability."""
    model_exists = MODEL_PATH.exists()

    # Fast camera availability check
    camera_ok = False
    try:
        import cv2
        camera_backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                           if platform.system() == "Windows" else [cv2.CAP_ANY])
        for backend in camera_backends:
            cap = cv2.VideoCapture(0, backend)
            camera_ok = cap.isOpened()
            cap.release()
            if camera_ok:
                break
    except Exception:
        camera_ok = False

    return HealthResponse(
        status="healthy" if model_exists else "degraded",
        timestamp=time.time(),
        model_loaded=model_exists,
        model_path=str(MODEL_PATH.name),
        camera_available=camera_ok,
        device="cpu",
        active_threads=threading.active_count(),
    )


@app.get("/api/status", response_model=PipelineStatusResponse, tags=["Pipeline"])
def get_status():
    """Return pipeline execution metrics."""
    return PipelineStatusResponse(
        is_running=True,
        frames_received=0,
        poses_detected=0,
        sequences_submitted=event_store.total_recorded,
        estimated_fps=15.0,
    )


@app.get("/api/events", response_model=EventListResponse, tags=["Events"])
def get_events(
    limit: int = Query(50, ge=1, le=500),
    min_risk: Optional[str] = Query(None, description="NORMAL, WARNING, HIGH, or CRITICAL"),
    only_alerts: bool = Query(False, description="Filter for triggered alerts only"),
):
    """Retrieve recent activity recognition and risk detection events."""
    raw_events = event_store.get_events(limit=limit, min_risk=min_risk, only_alerts=only_alerts)
    return EventListResponse(
        total_events=event_store.total_recorded,
        returned_count=len(raw_events),
        events=[EventPayload(**ev) for ev in raw_events],
    )


@app.post("/api/events", response_model=EventPayload, status_code=status.HTTP_201_CREATED, tags=["Events"])
def ingest_event(payload: EventPayload):
    """Ingest a new activity recognition event into memory and disk log."""
    event_dict = payload.model_dump()
    event_store.record_event(event_dict)
    return payload


@app.get("/api/config/risk", response_model=RiskConfigResponse, tags=["Configuration"])
def get_risk_config():
    """Return active explainable risk rules, persistence limits, and laboratory zones."""
    cfg = risk_analyzer.cfg
    return RiskConfigResponse(
        version=cfg.get("version", "1.0"),
        thresholds=cfg.get("thresholds", {}),
        base_risk_scores=cfg.get("base_risk_scores", {}),
        persistence_windows=cfg.get("persistence_windows", {}),
        cooldown_seconds=cfg.get("cooldown_seconds", {}),
        zones=cfg.get("zones", []),
    )
