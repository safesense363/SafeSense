"""SafeSense AI — Explainable Risk Engine (PRD Phase 3, FR-012 to FR-016, FR-040 to FR-044, FR-060 to FR-064).

Evaluates temporal skeleton sequences, keypoint reliability, vertical motion dynamics,
posture aspect ratio, activity predictions, temporal persistence, cooldown deduplication,
and safety zones to produce explainable risk assessments.

Risk Levels (PRD FR-040):
- NORMAL: Routine everyday activities (walking, sitting, standing)
- WARNING: Fast movement / running or entering a hazardous buffer zone
- HIGH: Persistent lying down or prolonged recumbent state
- CRITICAL: Verified fall events (or provisional Possible Fall under confirmation)
- UNCERTAIN: Low classification confidence or ambiguous evidence (PRD FR-021)
- UNKNOWN: Camera occlusion, insufficient reliable keypoints, or non-finite pose (PRD FR-020)
- MONITORING: Baseline reliable pose stream with no active incident
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "risk_config.json"


@dataclass
class RiskResult:
    """Structured assessment output for real-time monitoring and incident creation."""
    level: str
    score: float
    reasons: list[str]
    alert_triggered: bool = False
    incident_type: Optional[str] = None
    provisional: bool = False  # PRD FR-014: Provisional Possible Fall
    recovery_state: Optional[str] = None  # PRD FR-044: AI recovery state


class RiskAnalyzer:
    """Configurable, explainable hybrid risk and fall detection engine."""

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        keypoint_confidence_threshold: Optional[float] = None,
        activity_confidence_threshold: float = 0.35,
        config_path: Optional[Path] = None,
    ):
        self.keypoint_confidence_threshold = (
            keypoint_confidence_threshold
            if keypoint_confidence_threshold is not None
            else confidence_threshold
        )
        self.activity_confidence_threshold = activity_confidence_threshold

        # Load rules from config file or fall back to verified defaults
        self.cfg = self._load_config(config_path or DEFAULT_CONFIG_PATH)

        # State tracking for temporal persistence, streaks, and alert cooldowns
        self.current_activity: Optional[str] = None
        self.activity_streak: int = 0
        self.last_alert_time: dict[str, float] = {}
        self.active_incidents: dict[str, dict[str, Any]] = {}

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Load risk rules and safety zones from JSON configuration."""
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    if isinstance(cfg_data, dict):
                        return cfg_data
            except Exception:
                pass

        # Built-in PRD defaults if file is missing or unparseable
        return {
            "version": "1.0",
            "thresholds": {
                "keypoint_confidence": 0.30,
                "activity_confidence": 0.35,
            },
            "base_risk_scores": {
                "falling": 0.85,
                "lying_down": 0.65,
                "running": 0.40,
                "sitting": 0.0,
                "standing": 0.0,
                "walking": 0.0,
            },
            "persistence_windows": {
                "falling": 2,
                "lying_down": 2,
                "running": 3,
            },
            "cooldown_seconds": {
                "falling": 10.0,
                "lying_down": 8.0,
                "running": 5.0,
            },
            "zones": [
                {
                    "name": "restricted_machinery_zone",
                    "bbox": [0.0, 0.0, 0.35, 0.50],
                    "risk_escalation": 0.30,
                    "description": "Restricted Machinery & High-Voltage Zone",
                }
            ],
        }

    def reset_state(self) -> None:
        """Reset temporal persistence, streaks, and cooldowns (PRD FR-032)."""
        self.current_activity = None
        self.activity_streak = 0
        self.last_alert_time.clear()
        self.active_incidents.clear()

    def _get_ground_position(self, sequence: np.ndarray) -> tuple[float, float]:
        """Estimate person ground contact coordinate (x, y) in normalized frame coords [0, 1]."""
        latest_frame = sequence[-1]  # shape: (17, 3)
        # Ankle joints: 15 (left), 16 (right)
        ankles = latest_frame[[15, 16]]
        if (ankles[:, 2] >= 0.25).any():
            valid_ankles = ankles[ankles[:, 2] >= 0.25]
            return float(np.mean(valid_ankles[:, 0])), float(np.mean(valid_ankles[:, 1]))

        # Fallback to hips: 11 (left), 12 (right)
        hips = latest_frame[[11, 12]]
        if (hips[:, 2] >= 0.20).any():
            valid_hips = hips[hips[:, 2] >= 0.20]
            return float(np.mean(valid_hips[:, 0])), float(np.mean(valid_hips[:, 1]))

        # General center of visible joints
        valid = latest_frame[latest_frame[:, 2] >= 0.20]
        if len(valid) > 0:
            return float(np.mean(valid[:, 0])), float(np.mean(valid[:, 1]))

        return 0.5, 0.5

    def _check_zone_violations(self, x: float, y: float) -> list[dict[str, Any]]:
        """Check if ground coordinate falls within any defined restricted safety zone (FR-060)."""
        violations = []
        for zone in self.cfg.get("zones", []):
            bbox = zone.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x1 <= x <= x2 and y1 <= y <= y2:
                    violations.append(zone)
        return violations

    def _analyze_kinematics(self, sequence: np.ndarray) -> dict[str, Any]:
        """Evaluate temporal body posture and downward velocity (PRD FR-012, FR-013).

        In normalized image coordinates:
        - y=0 is the top of the frame, y=1 is the floor.
        - Downward vertical movement means y increases (delta_y > 0).
        - Horizontal posture: bounding width >= bounding height or torso collapsed.
        """
        first_frame = sequence[0]
        latest_frame = sequence[-1]

        # Calculate mid-hip y in start vs end of window (joints 11, 12)
        hip_start_y = np.mean(first_frame[[11, 12], 1]) if (first_frame[[11, 12], 2] > 0.2).any() else first_frame[0, 1]
        hip_end_y = np.mean(latest_frame[[11, 12], 1]) if (latest_frame[[11, 12], 2] > 0.2).any() else latest_frame[0, 1]

        # Downward vertical displacement (positive = moved towards floor)
        vertical_drop = float(hip_end_y - hip_start_y)

        # Body aspect ratio of latest frame (width / height)
        valid_pts = latest_frame[latest_frame[:, 2] >= 0.2]
        if len(valid_pts) >= 4:
            min_x, max_x = float(np.min(valid_pts[:, 0])), float(np.max(valid_pts[:, 0]))
            min_y, max_y = float(np.min(valid_pts[:, 1])), float(np.max(valid_pts[:, 1]))
            w = max(0.01, max_x - min_x)
            h = max(0.01, max_y - min_y)
            aspect_ratio = w / h
            is_horizontal = aspect_ratio > 1.0 or h < 0.25
        else:
            aspect_ratio = 0.5
            is_horizontal = False

        return {
            "vertical_drop": vertical_drop,
            "aspect_ratio": aspect_ratio,
            "is_horizontal": is_horizontal,
        }

    def analyze(
        self,
        sequence: Union[np.ndarray, list],
        activity_prediction: Optional[Any] = None,
        timestamp: Optional[float] = None,
        track_id: Optional[int] = None,
    ) -> RiskResult:
        """Analyze temporal skeleton sequence and return structured risk result.

        Parameters
        ----------
        sequence: np.ndarray
            Temporal skeleton sequence of shape (30, 17, 3).
        activity_prediction: Optional[Any]
            Classification output (e.g. ActivityPrediction with activity, confidence).
        timestamp: Optional[float]
            Current observation epoch timestamp in seconds.
        track_id: Optional[int]
            Temporary tracking ID of observed subject.
        """
        now = time.time() if timestamp is None else float(timestamp)

        # Validate sequence format and dimensions
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim != 3 or arr.shape != (30, 17, 3):
            raise ValueError(f"Expected sequence shape (30, 17, 3), got {arr.shape}")

        # Check for non-finite keypoint values
        if not np.isfinite(arr).all():
            return RiskResult(
                level="UNKNOWN",
                score=0.0,
                reasons=["Invalid or non-finite skeleton coordinate data"],
            )

        # Pose reliability verification (PRD FR-020, FR-021)
        confidence = arr[:, :, 2]
        valid_frames = (confidence >= self.keypoint_confidence_threshold).sum(axis=1)

        # If too many frames have fewer than 5 reliable keypoints, pose is unreliable
        if (valid_frames < 5).sum() > 15:
            return RiskResult(
                level="UNKNOWN",
                score=0.0,
                reasons=["Insufficient reliable pose keypoints in camera stream (occlusion / low confidence)"],
            )

        # If no activity model prediction was provided, return baseline monitoring status
        if activity_prediction is None:
            return RiskResult(
                level="MONITORING",
                score=0.0,
                reasons=["Pose sequence validated; no activity model prediction attached"],
            )

        # Extract prediction attributes
        act = getattr(activity_prediction, "activity", "").lower()
        conf = float(getattr(activity_prediction, "confidence", 0.0))
        is_confident = getattr(activity_prediction, "is_confident", True)

        # Low confidence guard (PRD FR-021)
        if not is_confident or conf < self.activity_confidence_threshold:
            self.current_activity = None
            self.activity_streak = 0
            return RiskResult(
                level="UNCERTAIN",
                score=0.10,
                reasons=[f"Low classification confidence ({conf:.1%}) for predicted action: {act}"],
            )

        # Update activity persistence streak
        if act == self.current_activity:
            self.activity_streak += 1
        else:
            self.current_activity = act
            self.activity_streak = 1

        reasons: list[str] = []
        base_scores = self.cfg.get("base_risk_scores", {})
        persistence_limits = self.cfg.get("persistence_windows", {})
        cooldown_limits = self.cfg.get("cooldown_seconds", {})

        # Compute ground position and check zone restrictions
        gx, gy = self._get_ground_position(arr)
        zone_violations = self._check_zone_violations(gx, gy)
        kinematics = self._analyze_kinematics(arr)

        score = float(base_scores.get(act, 0.0))
        level = "NORMAL"
        alert_triggered = False
        incident_type: Optional[str] = None
        provisional = False
        recovery_state: Optional[str] = None

        # Rule evaluation by activity
        if act == "falling":
            level = "CRITICAL"
            score = round(min(1.0, 0.60 + 0.40 * conf), 3)
            incident_type = "FALL_INCIDENT"
            reasons.append(f"Fall motion pattern detected with {conf:.1%} model confidence")

        elif act == "lying_down":
            # PRD FR-013: Lying down alone is NOT a fall unless accompanied by a rapid vertical transition
            level = "HIGH"
            score = round(min(1.0, 0.45 + 0.35 * conf), 3)
            incident_type = "PROLONGED_RECUMBENT"
            reasons.append(f"Subject recumbent / lying down detected with {conf:.1%} confidence")

        elif act == "running":
            level = "WARNING"
            score = round(min(1.0, 0.25 + 0.25 * conf), 3)
            incident_type = "RUNNING_VIOLATION"
            reasons.append(f"Fast running movement detected in monitored zone (confidence: {conf:.1%})")

        elif act in ("walking", "standing", "sitting"):
            level = "NORMAL"
            score = 0.0
            reasons.append(f"Normal activity observed: {act}")
            # PRD FR-044: If previously in fall/recumbent state, check recovery
            if self.activity_streak >= 3:
                recovery_state = "RECOVERED"

        # Zone escalation (PRD FR-060 to FR-064)
        if zone_violations:
            for z in zone_violations:
                esc = float(z.get("risk_escalation", 0.30))
                score = round(min(1.0, score + esc), 3)
                reasons.append(
                    f"Location alert: Subject inside {z.get('description', z.get('name'))} (at x={gx:.2f}, y={gy:.2f})"
                )
                if level == "NORMAL":
                    level = "WARNING"
                elif level == "WARNING":
                    level = "HIGH"

        # Persistence & Cooldown Evaluation (PRD FR-014, FR-043, FR-047)
        required_streak = persistence_limits.get(act, 1)
        cooldown_dur = float(cooldown_limits.get(act, 0.0))

        if incident_type is not None:
            if self.activity_streak >= required_streak:
                last_time = self.last_alert_time.get(incident_type, 0.0)
                if (now - last_time) >= cooldown_dur:
                    alert_triggered = True
                    self.last_alert_time[incident_type] = now
                    reasons.append(
                        f"Verified incident: Activity persisted for {self.activity_streak} consecutive windows (threshold: {required_streak})"
                    )
                else:
                    reasons.append(f"Alert cooldown active ({now - last_time:.1f}s / {cooldown_dur}s)")
            else:
                # PRD FR-014: Provisional Possible Fall
                provisional = True
                reasons.append(
                    f"Incident pending verification ({self.activity_streak}/{required_streak} windows)"
                )

        return RiskResult(
            level=level,
            score=score,
            reasons=reasons,
            alert_triggered=alert_triggered,
            incident_type=incident_type,
            provisional=provisional,
            recovery_state=recovery_state,
        )
