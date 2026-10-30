import unittest
import numpy as np

from features.skeleton import SkeletonProcessor
from features.sequence_builder import SkeletonSequenceBuilder
from risk.risk_analyzer import RiskAnalyzer


class TestPosePipeline(unittest.TestCase):

    def setUp(self):
        self.processor = SkeletonProcessor(normalize=True)
        self.builder = SkeletonSequenceBuilder(window_size=30, stride=5)
        self.risk_analyzer = RiskAnalyzer(confidence_threshold=0.3)

    def test_skeleton_normalization(self):
        person = {
            "keypoints": np.array([[320.0, 240.0]] * 17, dtype=np.float32),
            "confidence": np.full(17, 0.9, dtype=np.float32),
        }
        skel = self.processor.process(person, frame_shape=(480, 640, 3))
        self.assertEqual(skel.shape, (17, 3))
        self.assertEqual(skel.dtype, np.float32)
        # Normalized coordinates: x/640 = 0.5, y/480 = 0.5
        np.testing.assert_allclose(skel[0, :2], [0.5, 0.5])
        self.assertAlmostEqual(float(skel[0, 2]), 0.9, places=5)

    def test_missing_frame_shape_raises(self):
        person = {
            "keypoints": np.ones((17, 2), dtype=np.float32),
            "confidence": np.ones(17, dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            self.processor.process(person, frame_shape=None)

    def test_invalid_keypoint_shape_raises(self):
        person = {
            "keypoints": np.ones((15, 2), dtype=np.float32),
            "confidence": np.ones(15, dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            self.processor.process(person, frame_shape=(480, 640))

    def test_non_finite_keypoints_raises(self):
        person = {
            "keypoints": np.full((17, 2), np.nan, dtype=np.float32),
            "confidence": np.ones(17, dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            self.processor.process(person, frame_shape=(480, 640))

    def test_sequence_builder_window_and_stride(self):
        skel = np.random.uniform(0.1, 0.9, (17, 3)).astype(np.float32)

        # First 29 skeletons return None
        for _ in range(29):
            self.assertIsNone(self.builder.add_skeleton(skel))

        # 30th skeleton returns the first window
        seq1 = self.builder.add_skeleton(skel)
        self.assertIsNotNone(seq1)
        self.assertEqual(seq1.shape, (30, 17, 3))

        # Stride = 5: frames 31-34 return None
        for _ in range(4):
            self.assertIsNone(self.builder.add_skeleton(skel))

        # Frame 35 returns the next window
        seq2 = self.builder.add_skeleton(skel)
        self.assertIsNotNone(seq2)
        self.assertEqual(seq2.shape, (30, 17, 3))

    def test_sequence_builder_reset(self):
        skel = np.random.uniform(0.1, 0.9, (17, 3)).astype(np.float32)

        for _ in range(20):
            self.builder.add_skeleton(skel)

        # Person leaves view -> reset
        self.builder.reset()
        self.assertFalse(self.builder.sequence_ready)
        self.assertEqual(len(self.builder.sequence), 0)

        # Buffer must be empty: 29 additions must still yield None
        for _ in range(29):
            self.assertIsNone(self.builder.add_skeleton(skel))

        seq = self.builder.add_skeleton(skel)
        self.assertIsNotNone(seq)
        self.assertEqual(seq.shape, (30, 17, 3))

    def test_risk_analyzer_valid_and_invalid(self):
        # Valid high confidence sequence
        valid_seq = np.random.uniform(0.1, 0.9, (30, 17, 3)).astype(np.float32)
        valid_seq[:, :, 2] = 0.8  # high confidence
        res_valid = self.risk_analyzer.analyze(valid_seq)
        self.assertEqual(res_valid.level, "MONITORING")

        # Invalid low confidence sequence
        low_conf_seq = valid_seq.copy()
        low_conf_seq[:, :, 2] = 0.05  # below threshold
        res_low = self.risk_analyzer.analyze(low_conf_seq)
        self.assertEqual(res_low.level, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
