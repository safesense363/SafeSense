# SafeSense AI — API & Event Contract

**Contract Version:** 1.0  
**Date:** October 2026  
**Status:** Live & Implemented in Inference Worker Pipeline  

---

## 1. System Event Schema

Whenever a 30-frame skeleton window is processed by [`InferenceWorker`](file:///e:/SafeSense%20-%20Copy/video_ingestion/inference_worker.py#L6) via [`main.inference_function`](file:///e:/SafeSense%20-%20Copy/main.py#L16), it produces a structured event dictionary following this contract:

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

---

## 2. Risk Level Semantics & Alert Policies

| Risk Level | Trigger Condition | System Action | Dashboard Indicator |
| :--- | :--- | :--- | :--- |
| **NORMAL** | Activity is `walking`, `standing`, or `sitting` with high confidence. | Log status; routine monitoring. | Green status indicator |
| **WARNING** | Activity is `running` in laboratory workshop zone. | Issue warning chime; log safety infraction. | Yellow banner |
| **HIGH** | Activity is `lying_down` on laboratory floor. | Visual alert; operator notification (check subject). | Orange banner |
| **CRITICAL** | Activity is `falling` (rapid transition detected). | Immediate audible siren, visual alarm, event snapshot. | Flashing Red banner |
| **UNCERTAIN** | Model confidence below threshold ($<0.30$). | Suppress alerts; continue multi-frame accumulation. | Gray indicator |
| **UNKNOWN** | Occlusion / insufficient valid pose keypoints ($<5$ joints). | Prompt camera line-of-sight check. | Gray indicator |

---

## 3. Future Backend API Endpoints (Phase 6 Preview)

When the FastAPI service is connected in Phase 6, it will expose:

* `GET /health`: Health and hardware inference status.
* `GET /status`: Latest pipeline FPS, camera state, and active worker counters.
* `GET /events/recent`: Last $N$ detected events and risk alerts.
* `POST /events`: Ingest structured events into persistent storage.
* `WS /ws/live`: Real-time WebSocket stream of live pose coordinates and alert payloads.
