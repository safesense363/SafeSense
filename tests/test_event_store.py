import tempfile
import unittest
from pathlib import Path

from backend.event_store import EventStore


class TestEventStore(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "events.sqlite3"
        self.store = EventStore(database_path=self.database_path)

    def tearDown(self):
        self.store.close()
        self.temp_directory.cleanup()

    def test_events_persist_across_store_restarts(self):
        event = {
            "timestamp": 100.0,
            "activity": "falling",
            "risk_level": "CRITICAL",
            "alert_triggered": True,
        }
        self.store.record_event(event)
        self.store.close()

        reopened_store = EventStore(database_path=self.database_path)
        try:
            self.assertEqual(reopened_store.total_recorded, 1)
            self.assertEqual(reopened_store.get_events()[0], event)
        finally:
            reopened_store.close()
            self.store = EventStore(database_path=self.database_path)

    def test_filters_are_applied_to_persisted_events(self):
        self.store.record_event({
            "timestamp": 1.0,
            "activity": "walking",
            "risk_level": "NORMAL",
            "alert_triggered": False,
        })
        self.store.record_event({
            "timestamp": 2.0,
            "activity": "falling",
            "risk_level": "CRITICAL",
            "alert_triggered": True,
        })

        self.assertEqual(
            [event["activity"] for event in self.store.get_events(min_risk="HIGH")],
            ["falling"],
        )
        self.assertEqual(
            [event["activity"] for event in self.store.get_events(only_alerts=True)],
            ["falling"],
        )

    def test_recording_adds_missing_timestamp_without_mutating_input(self):
        event = {"activity": "walking", "risk_level": "NORMAL"}

        recorded_event = self.store.record_event(event)

        self.assertNotIn("timestamp", event)
        self.assertIsInstance(recorded_event["timestamp"], float)
        self.assertEqual(self.store.total_recorded, 1)

    def test_latest_event_timestamp_and_database_readiness(self):
        self.assertIsNone(self.store.latest_event_timestamp)
        self.assertTrue(self.store.check_available())

        self.store.record_event({
            "timestamp": 123.5,
            "activity": "walking",
            "risk_level": "NORMAL",
        })

        self.assertEqual(self.store.latest_event_timestamp, 123.5)

    def test_invalid_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.get_events(limit=0)


if __name__ == "__main__":
    unittest.main()
