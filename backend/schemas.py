"""SafeSense AI — Backend API Schemas (Pydantic v2).

Defines strongly-typed request and response contracts for all API endpoints.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: float
    model_available: bool
    model_path: str
    camera_available: Optional[bool]
    database_available: bool
    device: Optional[str]
    active_threads: int


class PipelineStatusResponse(BaseModel):
    is_running: bool
    frames_received: Optional[int]
    poses_detected: Optional[int]
    sequences_submitted: Optional[int]
    estimated_fps: float
    last_event_timestamp: Optional[float]


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
