import unittest
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

import backend.app as app_module
from backend.app import INFERENCE_FRESHNESS_SECONDS
from backend.event_store import EventStore


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.original_event_store = app_module.event_store
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.database_path = Path(cls.temp_directory.name) / "events.sqlite3"
        app_module.event_store = EventStore(database_path=cls.database_path)
        cls.client = TestClient(app_module.app)

    def setUp(self):
        app_module.event_store.clear()

    @classmethod
    def tearDownClass(cls):
        app_module.event_store.close()
        app_module.event_store = cls.original_event_store
        cls.temp_directory.cleanup()

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["project"], "SafeSense AI")
        self.assertIn("docs_url", data)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["status"], ("healthy", "degraded"))
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["database_available"])
        self.assertEqual(data["device"], "cpu")

    def test_status_endpoint(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["is_running"])
        self.assertIsNone(data["frames_received"])
        self.assertIsNone(data["poses_detected"])
        self.assertEqual(data["estimated_fps"], 0.0)
        self.assertIsNone(data["last_event_timestamp"])

    def test_status_reports_recent_inference_event(self):
        sample_event = {
            "timestamp": time.time(),
            "activity": "walking",
            "confidence": 0.8,
            "risk_level": "NORMAL",
            "risk_score": 0.0,
        }
        response = self.client.post("/api/events", json=sample_event)
        self.assertEqual(response.status_code, 201)

        status_response = self.client.get("/api/status")
        self.assertTrue(status_response.json()["is_running"])
        self.assertAlmostEqual(
            status_response.json()["last_event_timestamp"],
            sample_event["timestamp"],
        )

    def test_status_does_not_report_stale_inference_as_running(self):
        app_module.event_store.record_event({
            "timestamp": time.time() - INFERENCE_FRESHNESS_SECONDS - 1,
            "activity": "walking",
            "risk_level": "NORMAL",
        })

        response = self.client.get("/api/status")

        self.assertFalse(response.json()["is_running"])

    def test_event_ingestion_and_retrieval(self):
        # Initial count is 0
        resp_init = self.client.get("/api/events")
        self.assertEqual(resp_init.status_code, 200)
        self.assertEqual(resp_init.json()["returned_count"], 0)

        # Ingest a sample fall incident
        sample_event = {
            "timestamp": time.time(),
            "activity": "falling",
            "confidence": 0.88,
            "risk_level": "CRITICAL",
            "risk_score": 0.952,
            "reasons": ["Fall motion pattern detected with 88.0% model confidence"],
            "alert_triggered": True,
            "incident_type": "FALL_INCIDENT",
            "latency_ms": 0.12,
            "probabilities": {"falling": 0.88, "walking": 0.12},
            "sequence_shape": [30, 17, 3],
        }

        resp_post = self.client.post("/api/events", json=sample_event)
        self.assertEqual(resp_post.status_code, 201)
        data_post = resp_post.json()
        self.assertEqual(data_post["activity"], "falling")

        # Query events
        resp_get = self.client.get("/api/events")
        self.assertEqual(resp_get.status_code, 200)
        data_get = resp_get.json()
        self.assertEqual(data_get["returned_count"], 1)
        self.assertEqual(data_get["events"][0]["activity"], "falling")

        app_module.event_store.close()
        app_module.event_store = EventStore(database_path=self.database_path)
        resp_after_restart = self.client.get("/api/events")
        self.assertEqual(resp_after_restart.status_code, 200)
        self.assertEqual(resp_after_restart.json()["total_events"], 1)
        self.assertEqual(
            resp_after_restart.json()["events"][0]["activity"],
            "falling",
        )

        # Query only alerts
        resp_alert = self.client.get("/api/events?only_alerts=true")
        self.assertEqual(resp_alert.status_code, 200)
        self.assertEqual(resp_alert.json()["returned_count"], 1)

    def test_event_ingestion_reports_database_failure(self):
        sample_event = {
            "timestamp": time.time(),
            "activity": "walking",
            "confidence": 0.8,
            "risk_level": "NORMAL",
            "risk_score": 0.0,
        }

        with patch.object(
            app_module.event_store,
            "record_event",
            side_effect=sqlite3.OperationalError("database unavailable"),
        ):
            response = self.client.post("/api/events", json=sample_event)

        self.assertEqual(response.status_code, 503)
        self.assertIn("not recorded", response.json()["detail"])

    def test_event_history_reports_database_failure(self):
        with patch.object(
            app_module.event_store,
            "get_events",
            side_effect=sqlite3.OperationalError("database unavailable"),
        ):
            response = self.client.get("/api/events")

        self.assertEqual(response.status_code, 503)
        self.assertIn("database is unavailable", response.json()["detail"])

    def test_risk_config_endpoint(self):
        resp = self.client.get("/api/config/risk")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("base_risk_scores", data)
        self.assertIn("persistence_windows", data)
        self.assertIn("zones", data)
        self.assertGreater(len(data["zones"]), 0)


if __name__ == "__main__":
    unittest.main()
