import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from main import _render_live_frame, run_live
from models.activity_predictor import ActivityPrediction
from models.ava_activity_predictor import AvaActivityPredictor


class FakeAvaModel:
    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, video, boxes):
        self.video_shape = tuple(video.shape)
        self.boxes = boxes.clone()
        scores = torch.full((1, 80), 0.01)
        scores[0, 11] = 0.91
        return scores


class FakePoseResult:
    def __init__(self):
        self.boxes = SimpleNamespace(
            xyxy=torch.tensor([[20.0, 10.0, 100.0, 120.0]])
        )

    def plot(self):
        return np.zeros((120, 160, 3), dtype=np.uint8)


class FakeCamera:
    instances = []

    def __init__(self, _camera_index, _backend=None):
        self.frames_read = 0
        self.released = False
        self.instances.append(self)

    def set(self, _property, _value):
        return True

    def isOpened(self):
        return True

    def read(self):
        import time

        time.sleep(0.003)
        self.frames_read += 1
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakePoseEstimator:
    def __init__(self, **_kwargs):
        self.result = FakePoseResult()

    def process(self, _frame):
        return self.result

    def extract_keypoints(self, _result):
        points = np.linspace(10, 100, 17, dtype=np.float32)
        return [{
            "keypoints": np.column_stack((points, points)),
            "confidence": np.full(17, 0.9, dtype=np.float32),
        }]


class FakeActivityPredictor:
    calls = 0

    def __init__(self, **_kwargs):
        pass

    def predict_clip(self, frames, person_box):
        self.__class__.calls += 1
        self.__class__.last_clip_size = len(frames)
        self.__class__.last_box = person_box
        return ActivityPrediction(
            activity="walking",
            activity_id=1,
            confidence=0.91,
            probabilities={"walking": 0.91},
            latency_ms=2.0,
            is_confident=True,
        )


class WebcamLiveTests(unittest.TestCase):
    def test_ava_predictor_maps_standing_and_builds_video_box_input(self):
        model = FakeAvaModel()
        predictor = AvaActivityPredictor(model=model)
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(8)]

        prediction = predictor.predict_clip(frames, [40, 20, 200, 220])

        self.assertEqual(prediction.activity, "standing")
        self.assertAlmostEqual(prediction.confidence, 0.91)
        self.assertEqual(model.video_shape, (1, 3, 4, 256, 341))
        self.assertEqual(tuple(model.boxes.shape), (1, 5))

    def test_live_overlay_contains_pose_and_activity_labels(self):
        with patch("main.cv2.putText", wraps=cv2.putText) as put_text:
            rendered = _render_live_frame(
                FakePoseResult(),
                1,
                "walking",
                0.82,
            )

        self.assertEqual(rendered.shape, (120, 160, 3))
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn(
            "POSE: DETECTED (1) | ACTIVITY: walking (82%)",
            labels,
        )
        self.assertEqual(labels[1], "Q / Esc: quit")

    def test_live_loop_opens_camera_and_submits_activity_inference(self):
        FakeCamera.instances.clear()
        FakeActivityPredictor.calls = 0
        args = SimpleNamespace(
            device="auto",
            pose_model=None,
            camera=0,
            fps=300,
            no_display=False,
        )
        wait_calls = {"count": 0}

        def wait_key(_delay):
            wait_calls["count"] += 1
            should_quit = (
                wait_calls["count"] >= 45
                and FakeActivityPredictor.calls >= 1
            )
            return ord("q") if should_quit else -1

        with (
            patch("main.resolve_runtime_device", return_value=torch.device("cpu")),
            patch("main.pose_model_for_device", return_value="pose.pt"),
            patch("main.PoseEstimator", FakePoseEstimator),
            patch("main.AvaActivityPredictor", FakeActivityPredictor),
            patch("main.cv2.VideoCapture", FakeCamera),
            patch("main.cv2.namedWindow"),
            patch("main.cv2.imshow"),
            patch("main.cv2.waitKey", side_effect=wait_key),
            patch("main.cv2.getWindowProperty", return_value=1),
            patch("main.cv2.destroyAllWindows"),
        ):
            run_live(args)

        self.assertTrue(FakeCamera.instances[0].released)
        self.assertGreater(FakeCamera.instances[0].frames_read, 0)
        self.assertGreaterEqual(FakeActivityPredictor.calls, 1)
        self.assertEqual(FakeActivityPredictor.last_clip_size, 8)
        np.testing.assert_allclose(
            FakeActivityPredictor.last_box,
            [20.0, 10.0, 100.0, 120.0],
        )


if __name__ == "__main__":
    unittest.main()
