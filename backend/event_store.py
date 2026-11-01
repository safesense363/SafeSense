"""SQLite-backed storage for activity and risk events."""
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = Path(
    os.environ.get(
        "SAFESENSE_EVENT_DB_PATH",
        str(PROJECT_ROOT / "reports" / "safesense_events.sqlite3"),
    )
)

RISK_ORDER = {
    "NORMAL": 0,
    "UNCERTAIN": 1,
    "UNKNOWN": 1,
    "WARNING": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


class EventStore:
    """Thread-safe event storage with durable SQLite persistence."""

    def __init__(
        self,
        max_buffer_size: int = 500,
        database_path: Optional[Path] = None,
    ):
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be greater than zero")

        self.max_buffer_size = max_buffer_size
        self.database_path = Path(database_path or DEFAULT_DATABASE_PATH)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(
            self.database_path,
            timeout=5.0,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    risk_rank INTEGER NOT NULL,
                    alert_triggered INTEGER NOT NULL,
                    event_json TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_risk_id "
                "ON events (risk_rank, id DESC)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_alert_id "
                "ON events (alert_triggered, id DESC)"
            )

    @property
    def total_recorded(self) -> int:
        with self.lock:
            row = self.connection.execute(
                "SELECT COUNT(*) AS event_count FROM events"
            ).fetchone()
        return int(row["event_count"])

    @property
    def latest_event_timestamp(self) -> Optional[float]:
        with self.lock:
            row = self.connection.execute(
                "SELECT MAX(timestamp) AS latest_timestamp FROM events"
            ).fetchone()
        timestamp = row["latest_timestamp"]
        return float(timestamp) if timestamp is not None else None

    def check_available(self) -> bool:
        with self.lock:
            self.connection.execute("SELECT 1").fetchone()
        return True

    def record_event(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """Commit an event before reporting it as recorded."""
        event = dict(event_data)
        event.setdefault("timestamp", time.time())
        event_json = json.dumps(event, allow_nan=False)
        risk_level = str(event.get("risk_level", "NORMAL")).upper()
        risk_rank = RISK_ORDER.get(risk_level, 0)

        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO events (timestamp, risk_rank, alert_triggered, event_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event["timestamp"],
                    risk_rank,
                    int(bool(event.get("alert_triggered", False))),
                    event_json,
                ),
            )
        return event

    def get_events(
        self,
        limit: int = 50,
        min_risk: Optional[str] = None,
        only_alerts: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve recent events matching filter criteria."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        min_rank = RISK_ORDER.get(min_risk.upper(), 0) if min_risk else 0
        bounded_limit = min(limit, self.max_buffer_size)
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT event_json
                FROM events
                WHERE risk_rank >= ?
                  AND (? = 0 OR alert_triggered = 1)
                ORDER BY id DESC
                LIMIT ?
                """,
                (min_rank, int(only_alerts), bounded_limit),
            ).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    def clear(self) -> None:
        """Remove stored events; intended for isolated test stores."""
        with self.lock, self.connection:
            self.connection.execute("DELETE FROM events")

    def close(self) -> None:
        with self.lock:
            self.connection.close()
