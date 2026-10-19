"""AVA-finetuned RGB action recognition for person-bounded video clips."""
import time
from typing import Sequence

import cv2
import numpy as np
import torch

from models.activity_predictor import ActivityPrediction, CANONICAL_CLASSES


AVA_ACTIONS = {
    1: "bend/bow",
    3: "crouch/kneel",
    4: "dance",
    5: "falling",
    6: "getting_up",
    7: "jumping",
    8: "lying_down",
    9: "martial_arts",
    10: "running",
    11: "sitting",
    12: "standing",
    13: "swimming",
    14: "walking",
    15: "answer_phone",
    17: "carrying_object",
    20: "climbing",
    22: "opening",
    24: "cutting",
    26: "dressing",
    27: "drinking",
    28: "driving",
    29: "eating",
    30: "entering",
    34: "hitting_object",
    36: "lifting_object",
    37: "listening_to_music",
    38: "opening_object",
    41: "playing_instrument",
    43: "pointing",
    45: "pulling_object",
    46: "pushing_object",
    47: "putting_down",
    48: "reading",
    49: "riding",
    51: "sailing",
    52: "shooting",
    54: "smoking",
    56: "taking_photo",
    57: "looking_at_phone",
    58: "throwing",
    59: "touching_object",
    60: "turning_object",
    61: "watching_tv",
    62: "working_on_computer",
    63: "writing",
    64: "fighting",
    65: "serving",
    66: "grabbing_person",
    67: "clapping",
    68: "handshake",
    69: "waving",
    70: "hugging",
    72: "kissing",
    73: "lifting_person",
    74: "listening_to_person",
    76: "pushing_person",
    77: "singing",
    78: "taking_object_from_person",
    79: "talking",
    80: "watching_person",
}

AVA_CLASS_IDS = {
    5: "falling",
    8: "lying_down",
    10: "running",
    11: "sitting",
    12: "standing",
    14: "walking",
}


class AvaActivityPredictor:
    """Run the pretrained AVA person-action detector on a short RGB clip."""

    model_name = "Slow R50 (AVA v2.2)"

    def __init__(self, model=None, device: str = "cpu"):
        self.device = torch.device(device)
        if model is None:
            try:
                from pytorchvideo.models.hub import slow_r50_detection
            except ImportError as exc:
                raise RuntimeError(
                    "AVA action recognition requires PyTorchVideo. "
                    "Install the project dependencies with "
                    "'.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
                ) from exc
            model = slow_r50_detection(pretrained=True)

        self.model = model.eval().to(self.device)

    def predict_clip(
        self,
        frames: Sequence[np.ndarray],
        person_box: Sequence[float],
    ) -> ActivityPrediction:
        """Classify a person's action from BGR frames and an xyxy pixel box."""
        started = time.perf_counter()
        video, boxes = self._prepare_input(frames, person_box)

        with torch.inference_mode():
            scores = self.model(video.unsqueeze(0).to(self.device), boxes.to(self.device))

        if not isinstance(scores, torch.Tensor) or scores.ndim != 2 or scores.shape[1] != 80:
            shape = tuple(scores.shape) if isinstance(scores, torch.Tensor) else type(scores).__name__
            raise RuntimeError(f"Expected AVA model output with shape (persons, 80), got {shape}")
        if scores.shape[0] == 0:
            raise RuntimeError("AVA model returned no action scores for the detected person")

        score_vector = scores[0].detach().float().cpu()
        if not torch.isfinite(score_vector).all():
            raise RuntimeError("AVA model returned non-finite action scores")
        if (score_vector < 0).any() or (score_vector > 1).any():
            score_vector = torch.sigmoid(score_vector)

        top_index = int(score_vector.argmax().item())
        ava_id = top_index + 1
        activity = AVA_ACTIONS.get(ava_id, f"other_activity_{ava_id}")
        confidence = float(score_vector[top_index].item())
        probabilities = {
            name: round(float(score_vector[action_id - 1].item()), 4)
            for action_id, name in AVA_CLASS_IDS.items()
        }
        if activity not in CANONICAL_CLASSES:
            probabilities[activity] = round(confidence, 4)

        return ActivityPrediction(
            activity=activity,
            activity_id=(
                CANONICAL_CLASSES.index(activity)
                if activity in CANONICAL_CLASSES
                else len(CANONICAL_CLASSES)
            ),
            confidence=round(confidence, 4),
            probabilities=probabilities,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            is_confident=confidence >= 0.35,
        )

    @staticmethod
    def _prepare_input(
        frames: Sequence[np.ndarray],
        person_box: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not frames:
            raise ValueError("At least one video frame is required")
        box = np.asarray(person_box, dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("person_box must contain finite xyxy coordinates")

        first = np.asarray(frames[0])
        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError("Video frames must be three-channel BGR images")
        height, width = first.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Video frames must have positive dimensions")
        if any(np.asarray(frame).shape != first.shape for frame in frames):
            raise ValueError("All video frames must have the same dimensions")

        indices = np.linspace(0, len(frames) - 1, num=4).round().astype(int)
        scale = 256.0 / min(height, width)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        clip_frames = []
        for index in indices:
            rgb = cv2.cvtColor(np.asarray(frames[index]), cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
            clip_frames.append(resized)

        clip = torch.from_numpy(np.stack(clip_frames)).permute(3, 0, 1, 2).float() / 255.0
        mean = torch.tensor([0.45, 0.45, 0.45], dtype=clip.dtype).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225], dtype=clip.dtype).view(3, 1, 1, 1)
        clip = (clip - mean) / std

        box[[0, 2]] *= resized_width / width
        box[[1, 3]] *= resized_height / height
        box[[0, 2]] = np.clip(box[[0, 2]], 0, resized_width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, resized_height)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("person_box must have positive area inside the video frame")

        model_boxes = torch.tensor(
            [[0.0, *(float(value) for value in box)]],
            dtype=torch.float32,
        )
        return clip, model_boxes
