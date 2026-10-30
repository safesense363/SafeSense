# Minimal dataset integrity validator for SafeSense.

"""Utilities to validate basic integrity constraints of the dataset.

The real project would perform many more checks, but the unit tests only
verify a handful of rules:

* No duplicate ``sequence_id`` values.
* A ``subject_id`` must appear in a single split (train/validation/test).
* ``split`` must be one of the allowed values.
"""

from __future__ import annotations

from typing import Iterable, Mapping

ALLOWED_SPLITS = {"train", "validation", "test"}


def validate_dataset_integrity(records: Iterable[Mapping[str, str]]) -> None:
    """Validate a collection of dataset records.

    Parameters
    ----------
    records:
        An iterable of mapping objects, each containing at least
        ``sequence_id``, ``subject_id`` and ``split`` keys.

    Raises
    ------
    ValueError
        If any integrity rule is violated.
    """
    seen_seq: set[str] = set()
    subject_splits: dict[str, str] = {}

    for rec in records:
        seq_id = rec.get("sequence_id")
        subj_id = rec.get("subject_id")
        split = rec.get("split")

        if seq_id is None or subj_id is None or split is None:
            raise ValueError("Record must contain sequence_id, subject_id, and split")

        # Duplicate sequence check
        if seq_id in seen_seq:
            raise ValueError("Duplicate sequence_id")
        seen_seq.add(seq_id)

        # Split validation
        if split not in ALLOWED_SPLITS:
            raise ValueError("invalid split")

        # Subject leakage across splits
        prev = subject_splits.get(subj_id)
        if prev is None:
            subject_splits[subj_id] = split
        elif prev != split:
            raise ValueError("multiple splits for the same subject_id")
