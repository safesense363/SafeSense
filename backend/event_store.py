"""SafeSense AI — Event Store.

Provides thread-safe in-memory caching and lightweight persistent JSONL logging
for live activity recognition and risk detection events.
"""
from collections import deque
import json
from pathlib import Path
import threading
import time
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "reports" / "live_events.jsonl"

RISK_ORDER = {
    "NORMAL": 0,
    "UNCERTAIN": 1,
    "UNKNOWN": 1,
    "WARNING": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


class EventStore:
    """Thread-safe event store supporting ring-buffer memory and disk persistence."""

    def __init__(self, max_buffer_size: int = 500, log_path: Optional[Path] = None):
        self.buffer = deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()
        self.log_path = log_path or DEFAULT_LOG_PATH
        self.total_recorded = 0

        # Ensure reports directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_event(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """Record an event into memory and append to persistent log."""
        if "timestamp" not in event_data:
            event_data["timestamp"] = time.time()

        with self.lock:
            self.buffer.append(event_data)
            self.total_recorded += 1

        # Append to disk JSONL
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")
        except Exception as exc:
            print(f"Warning: Failed to persist event to disk: {exc}")

        return event_data

    def get_events(
        self,
        limit: int = 50,
        min_risk: Optional[str] = None,
        only_alerts: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve recent events matching filter criteria."""
        min_rank = RISK_ORDER.get(min_risk.upper(), 0) if min_risk else 0

        with self.lock:
            # Newest events first
            events = list(reversed(self.buffer))

        filtered = []
        for ev in events:
            if only_alerts and not ev.get("alert_triggered", False):
                continue

            lvl = ev.get("risk_level", "NORMAL")
            rank = RISK_ORDER.get(lvl, 0)
            if rank < min_rank:
                continue

            filtered.append(ev)
            if len(filtered) >= limit:
                break

        return filtered

    def clear(self):
        with self.lock:
            self.buffer.clear()
            self.total_recorded = 0
