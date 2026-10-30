import tempfile
import unittest
from pathlib import Path

import numpy as np

from features.dataset_writer import save_dataset


def sample_metadata():
    return [
        {
            "sequence_id": "seq_001",
            "label_id": 0,
            "label": "standing",
            "source": "safesense",
            "subject_id": "subject_001",
            "recording_id": "recording_001",
            "start_frame": 0,
            "end_frame": 29,
            "split": "train",
        },
        {
            "sequence_id": "seq_002",
            "label_id": 4,
            "label": "falling",
            "source": "safesense",
            "subject_id": "subject_002",
            "recording_id": "recording_002",
            "start_frame": 0,
            "end_frame": 29,
            "split": "test",
        },
    ]


class TestDatasetWriter(unittest.TestCase):

    def test_saves_aligned_dataset(self):
        sequences = np.zeros((2, 30, 17, 3), dtype=np.float32)
        labels = np.array([0, 4], dtype=np.int64)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_dataset(
                sequences, labels, sample_metadata(), temp_dir
            )

            saved_sequences = np.load(paths[0], allow_pickle=False)
            saved_labels = np.load(paths[1], allow_pickle=False)

            self.assertEqual(saved_sequences.shape, (2, 30, 17, 3))
            self.assertEqual(saved_sequences.dtype, np.float32)
            np.testing.assert_array_equal(saved_labels, labels)
            self.assertTrue(paths[2].is_file())

    def test_rejects_mismatched_counts(self):
        sequences = np.zeros((2, 30, 17, 3), dtype=np.float32)
        labels = np.array([0], dtype=np.int64)

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "counts must match"):
                save_dataset(
                    sequences, labels, sample_metadata(), temp_dir
                )

    def test_refuses_to_overwrite(self):
        sequences = np.zeros((2, 30, 17, 3), dtype=np.float32)
        labels = np.array([0, 4], dtype=np.int64)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_dataset(
                sequences, labels, sample_metadata(), temp_dir
            )

            with self.assertRaises(FileExistsError):
                save_dataset(
                    sequences, labels, sample_metadata(), temp_dir
                )

    def test_rejects_metadata_label_mismatch(self):
        sequences = np.zeros((2, 30, 17, 3), dtype=np.float32)
        labels = np.array([0, 4], dtype=np.int64)
        metadata = sample_metadata()
        metadata[1]["label"] = "running"

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Label mismatch"):
                save_dataset(sequences, labels, metadata, temp_dir)


if __name__ == "__main__":
    unittest.main()
