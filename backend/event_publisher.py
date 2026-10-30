"""SafeSense AI — Event Publisher.

Pushes completed inference events from the live pipeline (main.py) into the
running FastAPI event store via HTTP POST.  Falls back to local JSONL disk
logging if the API server is not reachable, so main.py always works standalone.

Usage (from main.py inference_function)::

    from backend.event_publisher import EventPublisher
    publisher = EventPublisher()   # once, at startup
    publisher.publish(event_dict)  # once per inference result
"""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FALLBACK_LOG = PROJECT_ROOT / "reports" / "live_events.jsonl"


class EventPublisher:
    """Non-blocking event publisher with HTTP and disk fallback.

    Args:
        api_url: Full URL to the POST /api/events endpoint.
        timeout:  HTTP socket timeout in seconds.
        enabled:  Set False to disable all publishing (dry-run / tests).
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000/api/events",
        timeout: float = 1.0,
        enabled: bool = True,
    ):
        self.api_url = api_url
        self.timeout = timeout
        self.enabled = enabled
        self._api_reachable: bool = True   # optimistic; switches on first failure
        self._consecutive_failures: int = 0
        self._retry_interval: int = 20     # re-probe after N failures

        # Ensure fallback log directory exists
        FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def publish(self, event: dict[str, Any]) -> bool:
        """Push event dict to the API or fall through to disk.

        Adds a timestamp if the caller did not provide one.
        Returns True if API POST succeeded, False otherwise.
        """
        if not self.enabled:
            return False

        if "timestamp" not in event:
            event = dict(event, timestamp=time.time())

        # Ensure required fields with defaults so Pydantic accepts the payload
        event.setdefault("confidence",      0.0)
        event.setdefault("risk_level",      "NORMAL")
        event.setdefault("risk_score",      0.0)
        event.setdefault("reasons",         [])
        event.setdefault("alert_triggered", False)
        event.setdefault("incident_type",   None)
        event.setdefault("latency_ms",      0.0)
        event.setdefault("probabilities",   {})
        event.setdefault("sequence_shape",  None)

        if self._api_reachable or (self._consecutive_failures % self._retry_interval == 0):
            ok = self._try_post(event)
            if ok:
                self._api_reachable = True
                self._consecutive_failures = 0
                return True
            else:
                self._api_reachable = False
                self._consecutive_failures += 1

        # Fallback: append to disk JSONL
        self._write_disk(event)
        return False

    def disable(self):
        """Disable all publishing (useful when running without the backend)."""
        self.enabled = False

    def enable(self):
        self.enabled = True

    # ── Internals ───────────────────────────────────────────────────────────

    def _try_post(self, event: dict[str, Any]) -> bool:
        try:
            body = json.dumps(event).encode("utf-8")
            req  = urllib.request.Request(
                self.api_url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status in (200, 201)
        except (urllib.error.URLError, OSError):
            return False
        except Exception:
            return False

    def _write_disk(self, event: dict[str, Any]) -> None:
        try:
            with FALLBACK_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            print(f"[EventPublisher] Disk fallback failed: {exc}")
