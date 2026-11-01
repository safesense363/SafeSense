"""Root conftest.py — install lightweight stubs for optional heavy deps.

Ultralytics (YOLO), pytorchvideo, and torchvision are not required to run
the unit-test suite.  We register minimal stub modules *before* pytest
imports any test files so that ``from ultralytics import YOLO`` etc. do not
raise ModuleNotFoundError during collection.

Tests that actually exercise YOLO inference use ``unittest.mock.patch`` to
replace the stub objects anyway, so this has no effect on test correctness.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helper: register a fake module (and sub-modules) in sys.modules
# ---------------------------------------------------------------------------

def _stub(name: str, **attrs) -> types.ModuleType:
    """Create and register a stub module with the given attributes."""
    mod = types.ModuleType(name)
    mod.__spec__ = None  # type: ignore[assignment]
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# ultralytics stub  (pose_estimator.py does: from ultralytics import YOLO)
# ---------------------------------------------------------------------------
if "ultralytics" not in sys.modules:
    class _FakeYOLO:
        """Minimal YOLO stand-in; real tests patch this with MagicMock."""

        def __init__(self, model_path: str = "yolo26n-pose.pt"):
            self.model_path = model_path

        def predict(self, source, conf=0.5, verbose=False, device="cpu"):
            # Return a list with one fake result
            result = MagicMock()
            result.keypoints = None
            result.plot.return_value = source  # echo the frame back
            return [result]

    ult = _stub("ultralytics", YOLO=_FakeYOLO)
    # Some code may import ultralytics.models or sub-packages
    _stub("ultralytics.models")
    _stub("ultralytics.utils")


# ---------------------------------------------------------------------------
# pytorchvideo stub  (ava_activity_predictor.py imports it)
# ---------------------------------------------------------------------------
if "pytorchvideo" not in sys.modules:
    _stub("pytorchvideo")
    _stub("pytorchvideo.models")
    _stub("pytorchvideo.data")
    _stub("pytorchvideo.transforms")


# ---------------------------------------------------------------------------
# torchvision stub  (some models import it)
# ---------------------------------------------------------------------------
if "torchvision" not in sys.modules:
    _stub("torchvision")
    _stub("torchvision.transforms")
    _stub("torchvision.models")

