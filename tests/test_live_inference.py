import unittest
from unittest.mock import patch

import cv2
import numpy as np

from models.activity_predictor import ActivityPredictor, ActivityPrediction, CANONICAL_CLASSES
from risk.risk_analyzer import RiskAnalyzer
from main import _make_inference_function, _render_live_frame


class FakePoseResult:
    def __init__(self):
        self.frame = np.zeros((80, 160, 3), dtype=np.uint8)

    def plot(self):
        return self.frame.copy()


class TestLiveInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = ActivityPredictor()
        cls.risk_analyzer = RiskAnalyzer()

    def test_predictor_contract_and_probabilities(self):
        # Valid test sequence
        seq = np.random.uniform(0.1, 0.9, (30, 17, 3)).astype(np.float32)
        pred = self.predictor.predict(seq)

        self.assertIsInstance(pred, ActivityPrediction)
        self.assertIn(pred.activity, CANONICAL_CLASSES)
        self.assertTrue(0 <= pred.activity_id < len(CANONICAL_CLASSES))
        self.assertTrue(0.0 <= pred.confidence <= 1.0)
        self.assertGreater(pred.latency_ms, 0.0)

        # Probabilities sum to 1
        prob_sum = sum(pred.probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_predictor_input_validation(self):
        # Wrong window size
        with self.assertRaises(ValueError):
            self.predictor.predict(np.zeros((25, 17, 3), dtype=np.float32))

        # Non-finite data
        bad_seq = np.zeros((30, 17, 3), dtype=np.float32)
        bad_seq[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.predictor.predict(bad_seq)

    def test_risk_analyzer_rules(self):
        seq = np.random.uniform(0.1, 0.9, (30, 17, 3)).astype(np.float32)
        seq[:, :, 2] = 0.9  # high confidence keypoints
        # Keep the ankle ground point outside the restricted zone so this
        # assertion measures the running rule rather than random zone overlap.
        seq[-1, [15, 16], :2] = [0.70, 0.80]

        # Fall trigger
        fall_pred = ActivityPrediction(
            activity="falling",
            activity_id=4,
            confidence=0.85,
            probabilities={},
            latency_ms=0.1,
            is_confident=True,
        )
        res_fall = self.risk_analyzer.analyze(seq, fall_pred)
        self.assertEqual(res_fall.level, "CRITICAL")
        self.assertGreaterEqual(res_fall.score, 0.6)
        self.assertTrue(any("fall" in r.lower() for r in res_fall.reasons))

        # Lying down trigger
        lying_pred = ActivityPrediction(
            activity="lying_down",
            activity_id=5,
            confidence=0.80,
            probabilities={},
            latency_ms=0.1,
            is_confident=True,
        )
        res_lying = self.risk_analyzer.analyze(seq, lying_pred)
        self.assertEqual(res_lying.level, "HIGH")
        self.assertTrue(any("lying" in r.lower() for r in res_lying.reasons))

        # Running trigger
        run_pred = ActivityPrediction(
            activity="running",
            activity_id=3,
            confidence=0.75,
            probabilities={},
            latency_ms=0.1,
            is_confident=True,
        )
        res_run = self.risk_analyzer.analyze(seq, run_pred)
        self.assertEqual(res_run.level, "WARNING")
        self.assertTrue(any("running" in r.lower() for r in res_run.reasons))

        # Normal activity (walking)
        walk_pred = ActivityPrediction(
            activity="walking",
            activity_id=1,
            confidence=0.85,
            probabilities={},
            latency_ms=0.1,
            is_confident=True,
        )
        res_walk = self.risk_analyzer.analyze(seq, walk_pred)
        self.assertEqual(res_walk.level, "NORMAL")
        self.assertEqual(res_walk.score, 0.0)

        # Low confidence -> UNCERTAIN
        low_conf_pred = ActivityPrediction(
            activity="walking",
            activity_id=1,
            confidence=0.15,
            probabilities={},
            latency_ms=0.1,
            is_confident=False,
        )
        res_low = self.risk_analyzer.analyze(seq, low_conf_pred)
        self.assertEqual(res_low.level, "UNCERTAIN")

    def test_main_inference_function_payload(self):
        seq = np.random.uniform(0.1, 0.9, (30, 17, 3)).astype(np.float32)
        seq[:, :, 2] = 0.85
        class NoopPublisher:
            def publish(self, event):
                return True

        updates = []
        infer = _make_inference_function(
            self.predictor,
            self.risk_analyzer,
            NoopPublisher(),
            quiet=True,
            on_result=updates.append,
        )
        event = infer(seq)

        self.assertIn("activity", event)
        self.assertIn("confidence", event)
        self.assertIn("risk_level", event)
        self.assertIn("risk_score", event)
        self.assertIn("reasons", event)
        self.assertIn("latency_ms", event)
        self.assertEqual(event["sequence_shape"], [30, 17, 3])
        self.assertEqual(updates, [event])

    def test_live_overlay_shows_pose_activity_confidence_and_risk(self):
        result = FakePoseResult()
        state = {
            "activity": "walking",
            "confidence": 0.82,
            "risk_level": "NORMAL",
        }

        with patch("main.cv2.putText", wraps=cv2.putText) as put_text:
            rendered = _render_live_frame(result, 1, state)

        self.assertEqual(rendered.shape, result.frame.shape)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("POSE: DETECTED (1) | ACTIVITY: walking (82.0%)", labels)
        self.assertIn("RISK: NORMAL", labels)

    def test_live_overlay_shows_missing_person_state(self):
        state = {
            "activity": "NO PERSON",
            "confidence": None,
            "risk_level": "MONITORING",
        }

        with patch("main.cv2.putText", wraps=cv2.putText) as put_text:
            _render_live_frame(FakePoseResult(), 0, state)

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("POSE: NOT DETECTED | ACTIVITY: NO PERSON (--)", labels)
        self.assertIn("RISK: MONITORING", labels)
        self.assertIn("Move into view to start pose detection", labels)


if __name__ == "__main__":
    unittest.main()
