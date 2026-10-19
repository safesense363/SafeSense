"""Train and evaluate a camera-matched SafeSense GRU from labeled recordings."""
import argparse
import json
from pathlib import Path
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from features.dataset_config import get_label_maps
from features.dataset_validator import validate_sequence
from models.activity_predictor import SafeSenseGRU


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECTION_DIR = PROJECT_ROOT / "data" / "processed" / "safesense_collection"
MODEL_PATH = PROJECT_ROOT / "models" / "safesense_gru_webcam.pt"
REPORT_PATH = PROJECT_ROOT / "reports" / "safesense_gru_webcam_metrics.json"
SEED = 42
HIDDEN_SIZE = 64
NUM_LAYERS = 2


def load_recordings(collection_dir: Path, allow_subject_overlap: bool = False):
    """Load validated per-recording arrays and preserve recording group IDs."""
    recordings_dir = collection_dir / "recordings"
    manifest_path = collection_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Recording manifest not found: {manifest_path}")

    name_to_id, id_to_name = get_label_maps()
    records = []
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {manifest_path.name} line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Manifest line {line_number} must be a JSON object"
                )
            required = {
                "recording_id",
                "subject_id",
                "label",
                "label_id",
                "split",
            }
            missing = required - record.keys()
            if missing:
                raise ValueError(
                    f"Manifest line {line_number} is missing fields: {sorted(missing)}"
                )
            records.append(record)

    for record in records:
        if (
            not isinstance(record["recording_id"], str)
            or not record["recording_id"].strip()
        ):
            raise ValueError("recording_id values must be non-empty strings")
        if Path(record["recording_id"]).name != record["recording_id"]:
            raise ValueError(f"Invalid recording_id path: {record['recording_id']!r}")
    recording_ids = [record["recording_id"] for record in records]
    duplicates = sorted(
        recording_id
        for recording_id, count in Counter(recording_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate recording IDs in manifest: {duplicates}")

    allowed_splits = {"train", "validation", "test"}
    subject_splits = defaultdict(set)
    split_arrays = {
        split: {"sequences": [], "labels": [], "groups": [], "recording_ids": []}
        for split in allowed_splits
    }
    recording_details = []

    for group_id, record in enumerate(records):
        recording_id = record["recording_id"]
        subject_id = record["subject_id"]
        split = record["split"]
        label = record["label"]
        label_id = record["label_id"]
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError(f"Invalid subject_id for recording {recording_id}")
        if not isinstance(split, str) or split not in allowed_splits:
            raise ValueError(f"Invalid split {split!r} for recording {recording_id}")
        if (
            not isinstance(label, str)
            or not isinstance(label_id, int)
            or isinstance(label_id, bool)
            or label not in name_to_id
            or label_id != name_to_id[label]
        ):
            raise ValueError(
                f"Label mismatch for {recording_id}: {label!r} / {label_id!r}"
            )
        subject_splits[subject_id].add(split)

        archive_path = recordings_dir / f"{recording_id}.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Recording archive missing for {recording_id}: {archive_path}"
            )
        with np.load(archive_path, allow_pickle=False) as archive:
            if not {"sequences", "labels"}.issubset(archive.files):
                raise ValueError(f"{archive_path.name} must contain sequences and labels")
            sequences = archive["sequences"]
            labels = archive["labels"]

        if sequences.ndim != 4 or sequences.shape[1:] != (30, 17, 3):
            raise ValueError(
                f"{archive_path.name} has invalid sequence shape {sequences.shape}"
            )
        if labels.shape != (len(sequences),):
            raise ValueError(f"{archive_path.name} sequence/label counts do not match")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{archive_path.name} labels must use an integer dtype")
        if len(sequences) == 0:
            raise ValueError(f"{archive_path.name} contains no sequences")
        if (
            "sequence_count" in record
            and record["sequence_count"] != len(sequences)
        ):
            raise ValueError(
                f"{archive_path.name} sequence count disagrees with the manifest"
            )
        if not np.all(labels == label_id):
            raise ValueError(
                f"{archive_path.name} labels disagree with the manifest label"
            )

        validated = np.stack([validate_sequence(sequence) for sequence in sequences])
        partition = split_arrays[split]
        partition["sequences"].append(validated)
        partition["labels"].append(labels.astype(np.int64, copy=False))
        partition["groups"].append(
            np.full(len(labels), group_id, dtype=np.int64)
        )
        partition["recording_ids"].append(
            np.full(len(labels), recording_id, dtype=object)
        )
        recording_details.append({
            "recording_id": recording_id,
            "subject_id": subject_id,
            "split": split,
            "label": label,
            "sequences": len(sequences),
        })

    if not records:
        raise ValueError(f"No webcam recordings found in {manifest_path}")
    if not allow_subject_overlap:
        overlaps = {
            subject: sorted(splits)
            for subject, splits in subject_splits.items()
            if len(splits) > 1
        }
        if overlaps:
            raise ValueError(
                "Subject leakage across splits detected: "
                f"{overlaps}. Use --allow-subject-overlap only for personal "
                "same-person evaluation on completely separate recordings."
            )

    for split, partition in split_arrays.items():
        if not partition["sequences"]:
            continue
        for key in ("sequences", "labels", "groups", "recording_ids"):
            partition[key] = np.concatenate(partition[key], axis=0)

    return split_arrays, recording_details, id_to_name


def validate_coverage(
    split_arrays,
    class_names,
    min_train_per_class,
    min_eval_per_class,
):
    """Require all classes and enough non-empty windows in every split."""
    problems = []
    for split in ("train", "validation", "test"):
        partition = split_arrays[split]
        labels = partition["labels"]
        minimum = min_train_per_class if split == "train" else min_eval_per_class
        counts = (
            np.bincount(labels, minlength=len(class_names))
            if len(labels)
            else np.zeros(len(class_names), dtype=int)
        )
        missing = [
            f"{class_names[label_id]} ({counts[label_id]}/{minimum} sequences)"
            for label_id in range(len(class_names))
            if counts[label_id] < minimum
        ]
        if missing:
            problems.append(f"{split}: " + ", ".join(missing))
    if problems:
        raise ValueError(
            "Not enough labeled camera data to train/evaluate all activities. "
            "Record every activity in train, validation, and test splits; "
            f"required minimums are train={min_train_per_class}, "
            f"validation/test={min_eval_per_class} sequences per class. "
            + " | ".join(problems)
        )


def _recording_predictions(targets, probabilities, recording_ids):
    """Average overlapping-window scores within each held-out recording."""
    labels_by_recording = defaultdict(list)
    scores_by_recording = defaultdict(list)
    for target, scores, recording_id in zip(targets, probabilities, recording_ids):
        labels_by_recording[recording_id].append(int(target))
        scores_by_recording[recording_id].append(scores)

    recording_labels = []
    recording_scores = []
    for recording_id in labels_by_recording:
        labels = set(labels_by_recording[recording_id])
        if len(labels) != 1:
            raise ValueError(
                f"Recording {recording_id!r} contains multiple ground-truth labels"
            )
        recording_labels.append(labels.pop())
        recording_scores.append(np.mean(scores_by_recording[recording_id], axis=0))
    return np.asarray(recording_labels), np.asarray(recording_scores)


def _evaluate(model, loader, class_ids):
    model.eval()
    targets = []
    predictions = []
    probabilities = []
    group_ids = []
    with torch.inference_mode():
        for sequences, labels, groups in loader:
            scores = torch.softmax(model(sequences), dim=-1)
            targets.extend(labels.numpy().tolist())
            predictions.extend(scores.argmax(dim=-1).numpy().tolist())
            probabilities.extend(scores.numpy())
            group_ids.extend(groups.numpy().tolist())
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    recording_labels, recording_scores = _recording_predictions(
        targets, probabilities, group_ids
    )
    recording_predictions = recording_scores.argmax(axis=1)
    return {
        "window_accuracy": float(accuracy_score(targets, predictions)),
        "window_macro_f1": float(f1_score(
            targets,
            predictions,
            labels=class_ids,
            average="macro",
            zero_division=0,
        )),
        "window_classification_report": classification_report(
            targets,
            predictions,
            labels=class_ids,
            output_dict=True,
            zero_division=0,
        ),
        "recording_count": int(len(recording_labels)),
        "recording_accuracy": float(accuracy_score(
            recording_labels, recording_predictions
        )),
        "recording_macro_f1": float(f1_score(
            recording_labels,
            recording_predictions,
            labels=class_ids,
            average="macro",
            zero_division=0,
        )),
        "recording_classification_report": classification_report(
            recording_labels,
            recording_predictions,
            labels=class_ids,
            output_dict=True,
            zero_division=0,
        ),
        "recording_confusion_matrix": confusion_matrix(
            recording_labels, recording_predictions, labels=class_ids
        ).tolist(),
    }


def train(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive")
    if args.min_train_per_class < 1 or args.min_eval_per_class < 1:
        raise ValueError("Per-class minimum sequence counts must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    model_path = Path(args.model_path)
    report_path = Path(args.report_path)
    if model_path.resolve() == report_path.resolve():
        raise ValueError("Model and report paths must be different")
    existing = [str(path) for path in (model_path, report_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(existing)
            + ". Pass --overwrite to replace them after training succeeds."
        )

    config_path = Path(args.collection_dir)
    class_names_by_id = get_label_maps()[1]
    class_names = [class_names_by_id[index] for index in range(len(class_names_by_id))]
    split_arrays, recordings, _ = load_recordings(
        config_path,
        allow_subject_overlap=args.allow_subject_overlap,
    )
    validate_coverage(
        split_arrays,
        class_names,
        min_train_per_class=args.min_train_per_class,
        min_eval_per_class=args.min_eval_per_class,
    )

    class_ids = list(range(len(class_names)))
    loaders = {}
    class_counts = {}
    recording_counts = {}
    for split, partition in split_arrays.items():
        if not len(partition["labels"]):
            continue
        x = partition["sequences"].reshape(-1, 30, 51).astype(np.float32)
        y = partition["labels"].astype(np.int64)
        groups = partition["groups"].astype(np.int64)
        dataset = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(groups),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        class_counts[split] = {
            class_names[label_id]: int((y == label_id).sum())
            for label_id in class_ids
        }
        recording_counts[split] = len(set(partition["recording_ids"].tolist()))

    train_labels = split_arrays["train"]["labels"]
    train_counts = np.bincount(train_labels, minlength=len(class_names))
    weights = np.sqrt(
        len(train_labels) / (len(class_names) * train_counts.astype(np.float32))
    )
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )
    model = SafeSenseGRU(
        input_size=51,
        hidden_size=HIDDEN_SIZE,
        num_classes=len(class_names),
        num_layers=NUM_LAYERS,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    best_validation_macro_f1 = -1.0
    best_state = None
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for sequences, labels, _ in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(sequences), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            sample_count += len(labels)

        validation_metrics = _evaluate(
            model, loaders["validation"], class_ids
        )
        validation_score = validation_metrics["recording_macro_f1"]
        improved = validation_score > best_validation_macro_f1
        if improved:
            best_validation_macro_f1 = validation_score
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            patience_left = args.patience
        else:
            patience_left -= 1

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={loss_sum / sample_count:.4f} | "
            f"validation_recording_macro_f1={validation_score:.4f}"
            f"{' *' if improved else ''}"
        )
        if patience_left <= 0:
            print("Early stopping: validation recording macro-F1 did not improve.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint")
    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, loaders["test"], class_ids)
    validation_metrics = _evaluate(model, loaders["validation"], class_ids)

    output = {
        "model_architecture": "SafeSenseGRU",
        "model_name": "SafeSense webcam-calibrated GRU",
        "classes": class_names,
        "input_size": 51,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_classes": len(class_names),
        "model_state_dict": model.state_dict(),
        "best_validation_recording_macro_f1": best_validation_macro_f1,
        "training_recordings": recording_counts["train"],
        "validation_recordings": recording_counts["validation"],
        "test_recordings": recording_counts["test"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_name(model_path.name + ".tmp")
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    try:
        torch.save(output, temporary_model)
        report = {
            "model_name": output["model_name"],
            "classes": class_names,
            "seed": SEED,
            "epochs_requested": args.epochs,
            "early_stopping_patience": args.patience,
            "allow_subject_overlap": args.allow_subject_overlap,
            "class_sequence_counts": class_counts,
            "recording_counts": recording_counts,
            "recordings": recordings,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "notes": [
                "Window metrics include overlapping windows and are not independent samples.",
                "Recording metrics average window scores within each held-out recording.",
                (
                    "Subject overlap was explicitly allowed; results estimate "
                    "same-person performance on separate recordings only."
                    if args.allow_subject_overlap
                    else "Subjects are isolated between train, validation, and test."
                ),
                "Scores are not calibrated probabilities or a safety certification.",
            ],
        }
        temporary_report.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        temporary_model.replace(model_path)
        temporary_report.replace(report_path)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {report_path}")
    print(
        "Held-out recordings: "
        f"accuracy={test_metrics['recording_accuracy']:.4f}, "
        f"macro-F1={test_metrics['recording_macro_f1']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GRU on labeled webcam pose recordings."
    )
    parser.add_argument("--collection-dir", type=Path, default=COLLECTION_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-train-per-class", type=int, default=10)
    parser.add_argument("--min-eval-per-class", type=int, default=3)
    parser.add_argument(
        "--allow-subject-overlap",
        action="store_true",
        help="Allow same person across splits; only tests new recordings of that person.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output model and report if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
