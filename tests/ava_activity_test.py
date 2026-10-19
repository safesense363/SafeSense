import unittest
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ava_activity_predictor import AvaActivityPredictor
from models.activity_predictor import ActivityPrediction
from main import _make_inference_function
from risk.risk_analyzer import RiskAnalyzer


class FakeAvaModel:
    def __init__(self, action_id, confidence):
        self.action_id = action_id
        self.confidence = confidence

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, video, boxes):
        self.video_shape = tuple(video.shape)
        self.boxes = boxes.clone()
        scores = torch.full((1, 80), 0.01)
        scores[0, self.action_id - 1] = self.confidence
        return scores


class FakePublisher:
    def publish(self, event):
        self.event = event
        return True


class FakeClipPredictor:
    def predict_clip(self, frames, person_box):
        self.frames = frames
        self.person_box = person_box
        return ActivityPrediction(
            activity="standing",
            activity_id=0,
            confidence=0.91,
            probabilities={"standing": 0.91},
            latency_ms=12.0,
            is_confident=True,
        )


class TestAvaActivityPredictor(unittest.TestCase):

    def setUp(self):
        self.frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(8)]

    def test_maps_ava_standing_action_and_builds_model_input(self):
        model = FakeAvaModel(action_id=12, confidence=0.91)
        predictor = AvaActivityPredictor(model=model)

        prediction = predictor.predict_clip(self.frames, [40, 20, 200, 220])

        self.assertEqual(prediction.activity, "standing")
        self.assertAlmostEqual(prediction.confidence, 0.91)
        self.assertEqual(prediction.probabilities["standing"], 0.91)
        self.assertEqual(model.video_shape, (1, 3, 4, 256, 341))
        self.assertEqual(tuple(model.boxes.shape), (1, 5))
        self.assertEqual(model.boxes[0, 0].item(), 0.0)

    def test_maps_fall_and_other_ava_actions(self):
        fall = AvaActivityPredictor(
            model=FakeAvaModel(action_id=5, confidence=0.88)
        ).predict_clip(self.frames, [40, 20, 200, 220])
        other = AvaActivityPredictor(
            model=FakeAvaModel(action_id=29, confidence=0.72)
        ).predict_clip(self.frames, [40, 20, 200, 220])

        self.assertEqual(fall.activity, "falling")
        self.assertEqual(other.activity, "eating")
        self.assertIn("eating", other.probabilities)

    def test_maps_all_safety_activity_labels(self):
        expected = {
            5: "falling",
            8: "lying_down",
            10: "running",
            11: "sitting",
            12: "standing",
            14: "walking",
        }
        for action_id, activity in expected.items():
            with self.subTest(activity=activity):
                prediction = AvaActivityPredictor(
                    model=FakeAvaModel(action_id, 0.9)
                ).predict_clip(self.frames, [40, 20, 200, 220])
                self.assertEqual(prediction.activity, activity)

    def test_rejects_invalid_person_box(self):
        predictor = AvaActivityPredictor(model=FakeAvaModel(12, 0.9))
        with self.assertRaises(ValueError):
            predictor.predict_clip(self.frames, [20, 20, 20, 80])

    def test_live_inference_passes_rgb_clip_and_emits_activity_event(self):
        predictor = FakeClipPredictor()
        publisher = FakePublisher()
        infer = _make_inference_function(
            predictor,
            RiskAnalyzer(),
            publisher,
            quiet=True,
        )
        skeleton = np.zeros((30, 17, 3), dtype=np.float32)
        skeleton[:, :, 2] = 0.9
        skeleton[:, [15, 16], :2] = [0.7, 0.8]
        window = {
            "skeleton_sequence": skeleton,
            "rgb_frames": self.frames,
            "person_box": [40, 20, 200, 220],
        }

        event = infer(window)

        self.assertEqual(event["activity"], "standing")
        self.assertEqual(event["sequence_shape"], [30, 17, 3])
        self.assertEqual(publisher.event, event)
        self.assertEqual(len(predictor.frames), len(self.frames))


if __name__ == "__main__":
    unittest.main()
