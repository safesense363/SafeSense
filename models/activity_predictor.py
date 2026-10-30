"""Shared result type for SafeSense activity-recognition models."""
from dataclasses import dataclass


CANONICAL_CLASSES = [
    "standing",
    "walking",
    "sitting",
    "running",
    "falling",
    "lying_down",
]


@dataclass
class ActivityPrediction:
    activity: str
    activity_id: int
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    is_confident: bool
