import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.runtime_device import (
    CPU_POSE_MODEL,
    CUDA_POSE_MODEL,
    pose_model_for_device,
    resolve_runtime_device,
)
from pose.pose_estimator import PoseEstimator


class RuntimeDeviceTests(unittest.TestCase):
    @patch("models.runtime_device.torch.cuda.is_available", return_value=False)
    def test_auto_uses_cpu_when_cuda_is_unavailable(self, _is_available):
        self.assertEqual(resolve_runtime_device("auto"), torch.device("cpu"))

    @patch("models.runtime_device.torch.cuda.is_available", return_value=True)
    def test_auto_uses_first_cuda_device_when_available(self, _is_available):
        self.assertEqual(resolve_runtime_device("auto"), torch.device("cuda:0"))

    @patch("models.runtime_device.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_fails_with_actionable_message(self, _is_available):
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            resolve_runtime_device("cuda")

    def test_cpu_override_and_model_choices(self):
        with patch("models.runtime_device.torch.cuda.is_available", return_value=True):
            device = resolve_runtime_device("cpu")

        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(pose_model_for_device(device), CPU_POSE_MODEL)
        self.assertEqual(
            pose_model_for_device(torch.device("cuda:0")),
            CUDA_POSE_MODEL,
        )

    @patch("pose.pose_estimator.YOLO")
    def test_pose_estimator_uses_selected_device(self, yolo_class):
        result = object()
        yolo_class.return_value.predict.return_value = [result]
        estimator = PoseEstimator(
            model_path=CUDA_POSE_MODEL,
            device="cuda:0",
        )

        self.assertIs(estimator.process(np.zeros((32, 32, 3), dtype=np.uint8)), result)
        yolo_class.assert_called_once_with(CUDA_POSE_MODEL)
        yolo_class.return_value.predict.assert_called_once_with(
            source=unittest.mock.ANY,
            conf=0.5,
            verbose=False,
            device="cuda:0",
        )


if __name__ == "__main__":
    unittest.main()
