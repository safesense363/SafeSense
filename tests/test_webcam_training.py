import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch

from features.dataset_config import get_label_maps
from features.train_webcam_gru import (
    load_recordings,
    train,
    validate_coverage,
)
from models.activity_predictor import ActivityPredictor
from tools.collect_webcam import validate_identity


class WebcamTrainingTests(unittest.TestCase):
    def make_collection(self, root: Path, same_subject: bool = False):
        recordings_dir = root / "recordings"
        recordings_dir.mkdir(parents=True)
        name_to_id, _ = get_label_maps()
        records = []

        for split in ("train", "validation", "test"):
            for label, label_id in name_to_id.items():
                recording_id = f"{label}-{split}"
                sequence_count = 2 if split == "train" else 1
                records.append({
                    "recording_id": recording_id,
                    "subject_id": (
                        "same-person" if same_subject else f"person-{split}"
                    ),
                    "label": label,
                    "label_id": label_id,
                    "split": split,
                    "sequence_count": sequence_count,
                })
                sequences = np.zeros(
                    (sequence_count, 30, 17, 3),
                    dtype=np.float32,
                )
                sequences[..., 2] = 0.8
                sequences[..., 0] = label_id / max(1, len(name_to_id) - 1)
                np.savez_compressed(
                    recordings_dir / f"{recording_id}.npz",
                    sequences=sequences,
                    labels=np.full(sequence_count, label_id, dtype=np.int64),
                )

        (root / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return records

    def test_loads_validated_recordings_and_preserves_split_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_collection(root)

            splits, recordings, _ = load_recordings(root)

        self.assertEqual(len(recordings), 18)
        self.assertEqual(len(splits["train"]["labels"]), 12)
        self.assertEqual(len(splits["validation"]["labels"]), 6)
        self.assertEqual(len(splits["test"]["labels"]), 6)
        self.assertEqual(splits["test"]["sequences"].shape, (6, 30, 17, 3))
        self.assertEqual(len(set(splits["test"]["groups"].tolist())), 6)

    def test_same_subject_across_splits_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_collection(root, same_subject=True)

            with self.assertRaisesRegex(ValueError, "Subject leakage"):
                load_recordings(root)
            loaded, _, _ = load_recordings(root, allow_subject_overlap=True)

        self.assertEqual(len(loaded["test"]["labels"]), 6)

    def test_collector_rejects_subject_split_overlap_unless_enabled(self):
        records = [{
            "recording_id": "standing-train-01",
            "subject_id": "same-person",
            "split": "train",
        }]
        with self.assertRaisesRegex(ValueError, "already belongs to split"):
            validate_identity(
                "same-person",
                "standing-test-01",
                "test",
                records,
            )
        validate_identity(
            "same-person",
            "standing-test-01",
            "test",
            records,
            allow_subject_overlap=True,
        )

    def test_rejects_corrupt_recording_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_collection(root)
            bad_archive = root / "recordings" / "standing-train.npz"
            np.savez_compressed(
                bad_archive,
                sequences=np.zeros((2, 29, 17, 3), dtype=np.float32),
                labels=np.zeros(2, dtype=np.int64),
            )

            with self.assertRaisesRegex(ValueError, "invalid sequence shape"):
                load_recordings(root)

    def test_coverage_rejects_missing_classes(self):
        _, id_to_name = get_label_maps()
        class_names = [id_to_name[index] for index in range(len(id_to_name))]
        splits = {
            split: {"labels": np.array([], dtype=np.int64)}
            for split in ("train", "validation", "test")
        }
        splits["train"]["labels"] = np.full(29, 1, dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "Not enough labeled camera data"):
            validate_coverage(splits, class_names, 10, 3)

    def test_training_writes_loadable_personal_checkpoint_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / "collection"
            self.make_collection(collection)
            model_path = root / "webcam.pt"
            report_path = root / "metrics.json"
            train(Namespace(
                epochs=1,
                patience=1,
                batch_size=6,
                learning_rate=1e-3,
                min_train_per_class=1,
                min_eval_per_class=1,
                collection_dir=collection,
                model_path=model_path,
                report_path=report_path,
                allow_subject_overlap=False,
                overwrite=False,
            ))

            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("recording_macro_f1", report["test_metrics"])
            predictor = ActivityPredictor(model_path)
            result = predictor.predict(np.zeros((30, 17, 3), dtype=np.float32))
            self.assertIn(result.activity, report["classes"])

            checkpoint = torch.load(model_path, map_location="cpu")
            self.assertEqual(checkpoint["num_classes"], 6)

    def test_refuses_to_replace_checkpoint_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / "collection"
            self.make_collection(collection)
            model_path = root / "existing.pt"
            report_path = root / "metrics.json"
            model_path.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                train(Namespace(
                    epochs=1,
                    patience=1,
                    batch_size=6,
                    learning_rate=1e-3,
                    min_train_per_class=1,
                    min_eval_per_class=1,
                    collection_dir=collection,
                    model_path=model_path,
                    report_path=report_path,
                    allow_subject_overlap=False,
                    overwrite=False,
                ))
            self.assertEqual(model_path.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
