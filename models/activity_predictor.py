"""SafeSense AI — Activity Recognition Predictor.

Loads the trained SafeSenseGRU temporal baseline checkpoint, performs model
warm-up, validates incoming 30-frame skeleton sequences, and returns typed
prediction outputs with softmax probabilities and execution latencies.
"""
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "safesense_gru_baseline.pt"

CANONICAL_CLASSES = [
    "standing",
    "walking",
    "sitting",
    "running",
    "falling",
    "lying_down",
]


class SafeSenseGRU(nn.Module):
    """Compact temporal GRU for real-time CPU human activity recognition."""

    def __init__(self, input_size: int = 51, hidden_size: int = 64, num_classes: int = 6, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)


@dataclass
class ActivityPrediction:
    activity: str
    activity_id: int
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    is_confident: bool


class ActivityPredictor:
    """Production-ready inference wrapper for the SafeSense activity model."""

    model_name = "SafeSense GRU"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.30,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.classes = checkpoint.get("classes", CANONICAL_CLASSES)
        hidden_size = checkpoint.get("hidden_size", 64)
        input_size = checkpoint.get("input_size", 51)
        num_classes = checkpoint.get("num_classes", len(self.classes))

        self.model = SafeSenseGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._warmup()

    def _warmup(self):
        """Execute a dummy inference pass to eliminate first-frame latency."""
        dummy = torch.zeros((1, 30, 51), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    def predict(self, sequence: Any) -> ActivityPrediction:
        """Run temporal activity inference on a 30-frame skeleton sequence.

        Args:
            sequence: Array-like of shape (30, 17, 3) or (1, 30, 17, 3) with
                      values [x_norm, y_norm, confidence].

        Returns:
            ActivityPrediction dataclass containing predicted class, probabilities,
            confidence, and latency.
        """
        t0 = time.perf_counter()
        arr = np.asarray(sequence, dtype=np.float32)

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.shape != (30, 17, 3):
            raise ValueError(f"Expected sequence shape (30, 17, 3), got {arr.shape}")

        if not np.isfinite(arr).all():
            raise ValueError("Sequence contains non-finite (NaN or Inf) values")

        # Flatten (30, 17, 3) -> (1, 30, 51)
        flat_input = arr.reshape(1, 30, 51)
        tensor_in = torch.from_numpy(flat_input).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_conf = float(probs[pred_id])
        pred_name = self.classes[pred_id]

        prob_dict = {
            self.classes[i]: round(float(probs[i]), 4)
            for i in range(len(self.classes))
        }

        latency_ms = (time.perf_counter() - t0) * 1000

        return ActivityPrediction(
            activity=pred_name,
            activity_id=pred_id,
            confidence=round(pred_conf, 4),
            probabilities=prob_dict,
            latency_ms=round(latency_ms, 3),
            is_confident=(pred_conf >= self.confidence_threshold),
        )
