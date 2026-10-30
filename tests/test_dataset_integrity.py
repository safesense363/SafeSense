import unittest

from features.dataset_integrity import validate_dataset_integrity


def sample_records():
    return [
        {
            "sequence_id": "seq_001",
            "subject_id": "subject_001",
            "split": "train",
        },
        {
            "sequence_id": "seq_002",
            "subject_id": "subject_002",
            "split": "test",
        },
        {
            "sequence_id": "seq_003",
            "subject_id": "subject_003",
            "split": "validation",
        },
    ]


class TestDatasetIntegrity(unittest.TestCase):

    def test_valid_subject_isolation(self):
        validate_dataset_integrity(sample_records())

    def test_duplicate_sequence_id_rejected(self):
        records = sample_records()
        records[1]["sequence_id"] = records[0]["sequence_id"]

        with self.assertRaisesRegex(ValueError, "Duplicate sequence_id"):
            validate_dataset_integrity(records)

    def test_subject_leakage_rejected(self):
        records = sample_records()
        records[1]["subject_id"] = records[0]["subject_id"]

        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_dataset_integrity(records)

    def test_invalid_split_rejected(self):
        records = sample_records()
        records[0]["split"] = "development"

        with self.assertRaisesRegex(ValueError, "invalid split"):
            validate_dataset_integrity(records)


if __name__ == "__main__":
    unittest.main()
