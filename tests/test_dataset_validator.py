import unittest

import numpy as np

from features.dataset_validator import (
    validate_label,
    validate_metadata,
    validate_sequence,
)


def valid_metadata():
    return {
        "sequence_id": "seq_0001",
        "label_id": 0,
        "label": "standing",
        "source": "safesense",
        "subject_id": "subject_001",
        "recording_id": "recording_001",
        "start_frame": 0,
        "end_frame": 29,
        "split": "train",
    }


class TestDatasetValidator(unittest.TestCase):

    def test_valid_sequence(self):
        sequence = np.zeros((30, 17, 3), dtype=np.float32)
        result = validate_sequence(sequence)

        self.assertEqual(result.shape, (30, 17, 3))
        self.assertEqual(result.dtype, np.float32)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            validate_sequence(np.zeros((29, 17, 3), dtype=np.float32))

    def test_non_finite_values_rejected(self):
        sequence = np.zeros((30, 17, 3), dtype=np.float32)
        sequence[0, 0, 0] = np.nan

        with self.assertRaises(ValueError):
            validate_sequence(sequence)

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):
            validate_label(6)

    def test_valid_metadata(self):
        result = validate_metadata(valid_metadata())
        self.assertEqual(result["label"], "standing")

    def test_missing_metadata_rejected(self):
        metadata = valid_metadata()
        del metadata["subject_id"]

        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_label_mismatch_rejected(self):
        metadata = valid_metadata()
        metadata["label"] = "walking"

        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_invalid_split_rejected(self):
        metadata = valid_metadata()
        metadata["split"] = "unknown"

        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_out_of_range_coordinates_rejected(self):
        sequence = np.zeros((30, 17, 3), dtype=np.float32)
        sequence[0, 0, 0] = 1.5

        with self.assertRaises(ValueError):
            validate_sequence(sequence)

    def test_invalid_confidence_rejected(self):
        sequence = np.zeros((30, 17, 3), dtype=np.float32)
        sequence[0, 0, 2] = 1.2

        with self.assertRaises(ValueError):
            validate_sequence(sequence)


if __name__ == "__main__":
    unittest.main()
