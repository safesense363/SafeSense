"""SafeSense AI — FastAPI Application.

Provides RESTful endpoints for system health, real-time pipeline status,
event querying, event ingestion, and explainable risk configuration.
"""
from pathlib import Path
import sqlite3
import threading
import time
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
INFERENCE_FRESHNESS_SECONDS = 5.0


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
    """Report backend, model-file, and database health without opening a camera."""
    model_exists = MODEL_PATH.exists()
    try:
        database_available = event_store.check_available()
    except sqlite3.Error:
        database_available = False

    return HealthResponse(
        status="healthy" if model_exists and database_available else "degraded",
        timestamp=time.time(),
        model_available=model_exists,
        model_path=str(MODEL_PATH.name),
        camera_available=None,
        database_available=database_available,
        device=None,
        active_threads=threading.active_count(),
    )


@app.get("/api/status", response_model=PipelineStatusResponse, tags=["Pipeline"])
def get_status():
    """Report inference freshness without inventing unmeasured runtime metrics."""
    now = time.time()
    try:
        last_event_timestamp = event_store.latest_event_timestamp
        total_events = event_store.total_recorded
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event database is unavailable.",
        ) from exc
    inference_is_fresh = (
        last_event_timestamp is not None
        and 0 <= now - last_event_timestamp <= INFERENCE_FRESHNESS_SECONDS
    )
    return PipelineStatusResponse(
        is_running=inference_is_fresh,
        frames_received=None,
        poses_detected=None,
        sequences_submitted=total_events,
        estimated_fps=0.0,
        last_event_timestamp=last_event_timestamp,
    )


@app.get("/api/events", response_model=EventListResponse, tags=["Events"])
def get_events(
    limit: int = Query(50, ge=1, le=500),
    min_risk: Optional[str] = Query(None, description="NORMAL, WARNING, HIGH, or CRITICAL"),
    only_alerts: bool = Query(False, description="Filter for triggered alerts only"),
):
    """Retrieve recent activity recognition and risk detection events."""
    try:
        raw_events = event_store.get_events(
            limit=limit,
            min_risk=min_risk,
            only_alerts=only_alerts,
        )
        total_events = event_store.total_recorded
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event database is unavailable.",
        ) from exc
    return EventListResponse(
        total_events=total_events,
        returned_count=len(raw_events),
        events=[EventPayload(**ev) for ev in raw_events],
    )


@app.post("/api/events", response_model=EventPayload, status_code=status.HTTP_201_CREATED, tags=["Events"])
def ingest_event(payload: EventPayload):
    """Persist an activity recognition event before acknowledging ingestion."""
    event_dict = payload.model_dump()
    try:
        event_store.record_event(event_dict)
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event could not be saved; it was not recorded.",
        ) from exc
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
