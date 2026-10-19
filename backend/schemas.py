"""SafeSense AI — Backend API Schemas (Pydantic v2).

Defines strongly-typed request and response contracts for all API endpoints.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: float
    model_loaded: bool
    model_path: str
    camera_available: bool
    device: str
    active_threads: int


class PipelineStatusResponse(BaseModel):
    is_running: bool
    frames_received: int
    poses_detected: int
    sequences_submitted: int
    estimated_fps: float


class EventPayload(BaseModel):
    timestamp: float
    activity: str
    model: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: str
    risk_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    alert_triggered: bool = False
    incident_type: Optional[str] = None
    latency_ms: float = 0.0
    probabilities: dict[str, float] = Field(default_factory=dict)
    sequence_shape: Optional[list[int]] = None


class EventListResponse(BaseModel):
    total_events: int
    returned_count: int
    events: list[EventPayload]


class RiskConfigResponse(BaseModel):
    version: str
    thresholds: dict[str, float]
    base_risk_scores: dict[str, float]
    persistence_windows: dict[str, int]
    cooldown_seconds: dict[str, float]
    zones: list[dict[str, Any]]
