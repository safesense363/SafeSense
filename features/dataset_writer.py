import csv
from pathlib import Path
from typing import Any

import numpy as np

from features.dataset_config import load_dataset_schema
from features.dataset_integrity import validate_dataset_integrity
from features.dataset_validator import validate_label, validate_metadata, validate_sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def save_dataset(
    sequences: Any,
    labels: Any,
    metadata: list[dict[str, Any]],
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Validate and save aligned sequences, labels, and metadata."""
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "data" / "processed"

    sequences = np.asarray(sequences)
    labels = np.asarray(labels)

    if sequences.ndim != 4:
        raise ValueError("Sequences must have shape (N, 30, 17, 3)")

    if sequences.shape[1:] != (30, 17, 3):
        raise ValueError("Sequences must have shape (N, 30, 17, 3)")

    count = len(sequences)

    if labels.ndim != 1 or len(labels) != count or len(metadata) != count:
        raise ValueError("Sequence, label, and metadata counts must match")

    validated_sequences = np.stack(
        [validate_sequence(sequence) for sequence in sequences]
    ) if count else np.empty((0, 30, 17, 3), dtype=np.float32)

    validated_labels = np.asarray(
        [validate_label(label) for label in labels],
        dtype=np.int64,
    )

    validated_metadata = [validate_metadata(row) for row in metadata]

    for index, row in enumerate(validated_metadata):
        if row["label_id"] != int(validated_labels[index]):
            raise ValueError(
                f"Label mismatch at row {index}: labels array and metadata differ"
            )

    validate_dataset_integrity(validated_metadata)

    output_dir.mkdir(parents=True, exist_ok=True)

    sequences_path = output_dir / "sequences.npy"
    labels_path = output_dir / "labels.npy"
    metadata_path = output_dir / "metadata.csv"

    destinations = [sequences_path, labels_path, metadata_path]
    existing = [str(path) for path in destinations if path.exists()]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing dataset files: " + ", ".join(existing)
        )

    fields = load_dataset_schema()["required_fields"]

    try:
        np.save(sequences_path, validated_sequences, allow_pickle=False)
        np.save(labels_path, validated_labels, allow_pickle=False)

        with metadata_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in fields}
                for row in validated_metadata
            )
    except Exception:
        for path in destinations:
            path.unlink(missing_ok=True)
        raise

    return sequences_path, labels_path, metadata_path

