# SafeSense AI — API & Event Contract

**Contract Version:** 1.0  
**Date:** October 2026  
**Status:** Live & Implemented in Inference Worker Pipeline  

---

## 1. System Event Schema

Whenever a 30-frame skeleton window is processed by `InferenceWorker` via the live inference function, it produces a structured event dictionary following this contract:

```json
{
  "timestamp": 1729188000.123,
  "activity": "walking",
  "confidence": 0.874,
  "risk_level": "NORMAL",
  "risk_score": 0.0,
  "reasons": [
    "Normal activity observed: walking"
  ],
  "probabilities": {
    "standing": 0.032,
    "walking": 0.874,
    "sitting": 0.012,
    "running": 0.021,
    "falling": 0.008,
    "lying_down": 0.053
  },
  "latency_ms": 0.12,
  "sequence_shape": [30, 17, 3]
}
```

### Field Definitions

| Field | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `float` | Unix epoch timestamp (seconds) when inference completed. |
| `activity` | `string` | Top predicted activity name (`standing`, `walking`, `sitting`, `running`, `falling`, `lying_down`). |
| `confidence` | `float` | Softmax probability score $[0.0, 1.0]$ for the top predicted activity. |
| `risk_level` | `string` | Explainable risk category: `NORMAL`, `WARNING`, `HIGH`, `CRITICAL`, `UNCERTAIN`, or `UNKNOWN`. |
| `risk_score` | `float` | Metric score $[0.0, 1.0]$ reflecting safety severity. |
| `reasons` | `array[string]` | Human-readable explainable rationale detailing why this risk level was assigned. |
| `probabilities` | `dict[string, float]` | Full probability distribution across all 6 target classes. |
| `latency_ms` | `float` | Execution time of the temporal neural network in milliseconds. |
| `sequence_shape` | `array[int]` | Dimensions of the evaluated sliding window (always `[30, 17, 3]`). |

## 2. Persistence and failure behavior

The API stores accepted events in SQLite at
`reports/safesense_events.sqlite3` by default. Set
`SAFESENSE_EVENT_DB_PATH` to select another database file. `POST /api/events`
returns success only after the insert commits; if SQLite cannot save the event,
the API returns HTTP 503 and does not report it as recorded. Event queries and
the total event count are read from SQLite, so both survive API process
restarts. The separate JSONL file is still used by the live publisher when the
API is unavailable; existing JSONL entries are not automatically imported.

`GET /api/health` includes a `database_available` check. `GET /api/status`
reports inference as running only when the latest event timestamp is within
five seconds of the API clock. This is a recent-event freshness indicator, not
a camera or worker heartbeat. Frame and pose counts are `null` because they
are not currently instrumented; estimated FPS is `0.0` and must be treated as
unmeasured. The dashboard reads configured zone bounds from the risk-config
endpoint; it does not receive live subject-position coordinates.

---

## 3. Risk Level Semantics & Alert Policies

| Risk Level | Trigger Condition | System Action | Dashboard Indicator |
| :--- | :--- | :--- | :--- |
| **NORMAL** | Activity is `walking`, `standing`, or `sitting` with high confidence. | Log status; routine monitoring. | Green status indicator |
| **WARNING** | Activity is `running` in laboratory workshop zone. | Issue warning chime; log safety infraction. | Yellow banner |
| **HIGH** | Activity is `lying_down` on laboratory floor. | Visual alert; operator notification (check subject). | Orange banner |
| **CRITICAL** | Activity is `falling` (rapid transition detected). | Immediate audible siren, visual alarm, event snapshot. | Flashing Red banner |
| **UNCERTAIN** | Model confidence below threshold ($<0.30$). | Suppress alerts; continue multi-frame accumulation. | Gray indicator |
| **UNKNOWN** | Occlusion / insufficient valid pose keypoints ($<5$ joints). | Prompt camera line-of-sight check. | Gray indicator |

---

## 4. Backend API Endpoints

The FastAPI service currently exposes:

* `GET /api/health`: Model presence and camera availability.
* `GET /api/status`: Pipeline status summary.
* `GET /api/events`: Recent persisted events, with risk and alert filters.
* `POST /api/events`: Validate and persist a structured event.
* `GET /api/config/risk`: Active risk thresholds and zones.

WebSockets, authentication, and role-based authorization are not implemented
in this phase. The dashboard's “Dismiss Banner” action is local UI state only;
it does not acknowledge or resolve a persistent incident.
