# Minimal dataset validator utilities for SafeSense.

"""Validator functions used by the test suite.

The real project contains extensive validation logic; this stub provides the
behaviour required by `tests/test_dataset_validator.py`.
"""

from __future__ import annotations

import numpy as np
from typing import Mapping, Any

# Re‑use the canonical class list from activity_predictor.
from models.activity_predictor import CANONICAL_CLASSES

ALLOWED_SPLITS = {"train", "validation", "test"}


def validate_sequence(sequence: np.ndarray) -> np.ndarray:
    """Validate a (30, 17, 3) pose sequence.

    - Must have exact shape ``(30, 17, 3)``.
    - Must be ``np.float32``.
    - All values must be finite.
    - X/Y coordinates (first two channels) must be in ``[0, 1]``.
    - Confidence (third channel) must be in ``[0, 1]``.
    """
    if sequence.shape != (30, 17, 3):
        raise ValueError("wrong shape")
    if sequence.dtype != np.float32:
        raise ValueError("wrong dtype")
    if not np.isfinite(sequence).all():
        raise ValueError("non‑finite values")
    # Coordinates
    if (sequence[..., :2] < 0).any() or (sequence[..., :2] > 1).any():
        raise ValueError("out‑of‑range coordinates")
    # Confidence
    if (sequence[..., 2] < 0).any() or (sequence[..., 2] > 1).any():
        raise ValueError("invalid confidence values")
    return sequence


def validate_label(label_id: int) -> int:
    """Validate that a label index is within the canonical range."""
    if not isinstance(label_id, (int, np.integer)) or not (0 <= int(label_id) < len(CANONICAL_CLASSES)):
        raise ValueError("invalid label id")
    return int(label_id)


def validate_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate activity metadata.

    Required keys (as exercised by the tests):
    ``sequence_id, label_id, label, source, subject_id, recording_id,
    start_frame, end_frame, split``.
    """
    required = {
        "sequence_id",
        "label_id",
        "label",
        "source",
        "subject_id",
        "recording_id",
        "start_frame",
        "end_frame",
        "split",
    }
    missing = required - set(metadata.keys())
    if missing:
        raise ValueError(f"missing metadata fields: {missing}")

    # Label / id consistency
    label_id = metadata["label_id"]
    label = metadata["label"]
    if not (0 <= label_id < len(CANONICAL_CLASSES)):
        raise ValueError("invalid label_id")
    if CANONICAL_CLASSES[label_id] != label:
        raise ValueError("Label mismatch: label_id does not match label")

    # Split validation
    if metadata["split"] not in ALLOWED_SPLITS:
        raise ValueError("invalid split")

    return metadata
