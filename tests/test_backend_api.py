import unittest
import time
from starlette.testclient import TestClient

from backend.app import app, event_store


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        event_store.clear()

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
        self.assertEqual(data["device"], "cpu")

    def test_status_endpoint(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_running"])
        self.assertIn("estimated_fps", data)

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

        # Query only alerts
        resp_alert = self.client.get("/api/events?only_alerts=true")
        self.assertEqual(resp_alert.status_code, 200)
        self.assertEqual(resp_alert.json()["returned_count"], 1)

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
