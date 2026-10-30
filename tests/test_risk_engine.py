import unittest
import numpy as np

from models.activity_predictor import ActivityPrediction
from risk.risk_analyzer import RiskAnalyzer


class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.analyzer = RiskAnalyzer()
        self.analyzer.reset_state()

        # Synthetic sequence with feet at (0.7, 0.8) [outside restricted zone]
        self.normal_seq = np.random.uniform(0.1, 0.9, (30, 17, 3)).astype(np.float32)
        self.normal_seq[:, :, 2] = 0.9
        self.normal_seq[-1, [15, 16], 0] = 0.70  # x = 0.70
        self.normal_seq[-1, [15, 16], 1] = 0.80  # y = 0.80

        # Synthetic sequence with feet at (0.2, 0.3) [inside restricted machinery zone [0, 0, 0.35, 0.5]]
        self.danger_zone_seq = np.random.uniform(0.1, 0.9, (30, 17, 3)).astype(np.float32)
        self.danger_zone_seq[:, :, 2] = 0.9
        self.danger_zone_seq[-1, [15, 16], 0] = 0.20  # x = 0.20
        self.danger_zone_seq[-1, [15, 16], 1] = 0.30  # y = 0.30

    def _make_prediction(self, activity: str, confidence: float = 0.85) -> ActivityPrediction:
        return ActivityPrediction(
            activity=activity,
            activity_id=4 if activity == "falling" else 1,
            confidence=confidence,
            probabilities={},
            latency_ms=0.1,
            is_confident=True,
        )

    def test_persistence_escalation(self):
        fall_pred = self._make_prediction("falling", 0.90)

        # Window 1: Pending verification, alert not yet triggered
        res1 = self.analyzer.analyze(self.normal_seq, fall_pred, timestamp=100.0)
        self.assertEqual(res1.level, "CRITICAL")
        self.assertFalse(res1.alert_triggered)
        self.assertTrue(any("pending" in r.lower() for r in res1.reasons))

        # Window 2: Persistence threshold met (2/2) -> alert triggers
        res2 = self.analyzer.analyze(self.normal_seq, fall_pred, timestamp=100.5)
        self.assertEqual(res2.level, "CRITICAL")
        self.assertTrue(res2.alert_triggered)
        self.assertEqual(res2.incident_type, "FALL_INCIDENT")
        self.assertTrue(any("verified incident" in r.lower() for r in res2.reasons))

    def test_alert_cooldown_deduplication(self):
        fall_pred = self._make_prediction("falling", 0.90)

        # Reach persistence threshold
        self.analyzer.analyze(self.normal_seq, fall_pred, timestamp=100.0)
        res_trigger = self.analyzer.analyze(self.normal_seq, fall_pred, timestamp=100.5)
        self.assertTrue(res_trigger.alert_triggered)

        # Window 3 (2 seconds later): within 10s cooldown -> alert suppressed
        res_suppressed = self.analyzer.analyze(self.normal_seq, fall_pred, timestamp=102.5)
        self.assertFalse(res_suppressed.alert_triggered)
        self.assertTrue(any("cooldown active" in r.lower() for r in res_suppressed.reasons))

        # Window 4 (12 seconds later): cooldown expired -> alert triggers again
        res_rearm = self.analyzer.analyze(self.normal_seq, fall_pred, timestamp=113.0)
        self.assertTrue(res_rearm.alert_triggered)

    def test_zone_violation_escalation(self):
        walk_pred = self._make_prediction("walking", 0.85)

        # Walking outside restricted zone -> NORMAL, score 0.0
        res_outside = self.analyzer.analyze(self.normal_seq, walk_pred)
        self.assertEqual(res_outside.level, "NORMAL")
        self.assertEqual(res_outside.score, 0.0)

        # Walking inside restricted machinery zone -> Escalates to WARNING, score 0.30
        res_inside = self.analyzer.analyze(self.danger_zone_seq, walk_pred)
        self.assertEqual(res_inside.level, "WARNING")
        self.assertGreaterEqual(res_inside.score, 0.30)
        self.assertTrue(any("Restricted Machinery" in r for r in res_inside.reasons))

    def test_streak_reset_on_activity_transition(self):
        fall_pred = self._make_prediction("falling", 0.85)
        walk_pred = self._make_prediction("walking", 0.85)

        # 1 window of falling
        self.analyzer.analyze(self.normal_seq, fall_pred)
        self.assertEqual(self.analyzer.activity_streak, 1)

        # Transition to walking -> resets streak
        res_walk = self.analyzer.analyze(self.normal_seq, walk_pred)
        self.assertEqual(self.analyzer.current_activity, "walking")
        self.assertEqual(self.analyzer.activity_streak, 1)
        self.assertEqual(res_walk.level, "NORMAL")


if __name__ == "__main__":
    unittest.main()
