
import argparse
import json
import time
import uuid
from pathlib import Path
import argparse
import json
import time
import uuid
import sys
from pathlib import Path

# Make the SafeSense project root importable when running from tools/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import pose
from pose.pose_estimator import PoseEstimator
from features.skeleton import SkeletonProcessor
from features.sequence_builder import SkeletonSequenceBuilder
from features.dataset_config import load_activity_config


ROOT = Path(__file__).resolve().parent.parent
COLLECTION_DIR = ROOT / "data" / "processed" / "safesense_collection"
RECORDINGS_DIR = COLLECTION_DIR / "recordings"
MANIFEST_PATH = COLLECTION_DIR / "manifest.jsonl"


def load_manifest():
    records = []

    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid manifest JSON on line {line_number}"
                    ) from exc

    return records


def validate_identity(
    subject_id,
    recording_id,
    split,
    records,
    allow_subject_overlap=False,
):
    if not subject_id.strip() or not recording_id.strip():
        raise ValueError("Subject and recording IDs cannot be empty.")

    if any(r["recording_id"] == recording_id for r in records):
        raise ValueError(
            f"Recording ID already exists: {recording_id}"
        )

    previous_splits = {
        r["split"] for r in records
        if r["subject_id"] == subject_id
    }

    if (
        previous_splits
        and split not in previous_splits
        and not allow_subject_overlap
    ):
        raise ValueError(
            f"Subject {subject_id!r} already belongs to "
            f"split {sorted(previous_splits)}. "
            "Do not place one subject in multiple splits unless using "
            "--allow-subject-overlap for personal, recording-separated evaluation."
        )


def main():
    config = load_activity_config()
    classes = {
        item["name"]: item["id"]
        for item in config["classes"]
    }

    parser = argparse.ArgumentParser(
        description="Collect one labeled SafeSense webcam recording."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--label", required=True, choices=classes.keys())
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--allow-subject-overlap",
        action="store_true",
        help=(
            "Allow the same person in train/validation/test. "
            "Evaluation then applies to new recordings of that person only."
        ),
    )
    args = parser.parse_args()

    if args.duration < 5:
        parser.error("--duration must be at least 5 seconds")

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_manifest()

    validate_identity(
        args.subject,
        args.recording,
        args.split,
        records,
        allow_subject_overlap=args.allow_subject_overlap,
    )
    if args.allow_subject_overlap:
        print(
            "Personal calibration mode: train/validation/test must use "
            "different recordings; results do not measure performance on unseen people."
        )

    output_path = RECORDINGS_DIR / f"{args.recording}.npz"
    metadata_path = RECORDINGS_DIR / f"{args.recording}.json"

    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Output already exists for {args.recording!r}; "
            "choose a new recording ID."
        )

    print(f"Activity: {args.label}")
    print(f"Subject: {args.subject} | Split: {args.split}")
    print(f"Duration: {args.duration}s")
    print("Press Q to cancel. Start the activity when collection begins.")

    pose_estimator = PoseEstimator()
    skeleton_processor = SkeletonProcessor(normalize=True)
    builder = SkeletonSequenceBuilder(window_size=30, stride=5)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. "
            "Check camera permissions or try --camera 1."
        )

    sequences = []
    start_frame = 0
    frame_index = 0
    cancelled = False
    started = time.monotonic()

    try:
        while time.monotonic() - started < args.duration:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a webcam frame.")

            frame_index += 1
            result = pose_estimator.process(frame)
            persons = pose_estimator.extract_keypoints(result)

            if not persons:
                builder.reset()
            else:
                skeleton = skeleton_processor.process(
                    persons[0],
                    frame_shape=frame.shape,
                )
                sequence = builder.add_skeleton(skeleton)

                if sequence is not None:
                    sequences.append(sequence.copy())
                    start_frame = frame_index - 29

            remaining = max(
                0, args.duration - int(time.monotonic() - started)
            )
            cv2.putText(
                frame,
                f"{args.label} | {remaining}s | "
                f"windows: {len(sequences)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imshow("SafeSense Data Collection", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                cancelled = True
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if cancelled:
        print("Collection cancelled. No dataset files saved.")
        return

    if not sequences:
        print("No complete pose sequences collected. Nothing saved.")
        return

    # Recheck before writing, in case another collector ran concurrently.
    records = load_manifest()
    validate_identity(
        args.subject,
        args.recording,
        args.split,
        records,
        allow_subject_overlap=args.allow_subject_overlap,
    )

    sequence_array = np.asarray(sequences, dtype=np.float32)
    label_id = classes[args.label]

    metadata = {
        "recording_id": args.recording,
        "subject_id": args.subject,
        "source": "safesense",
        "label": args.label,
        "label_id": label_id,
        "split": args.split,
        "sequence_count": len(sequence_array),
        "frame_count": frame_index,
        "sequence_shape": [30, 17, 3],
        "coordinate_format": "camera_relative_xy_confidence",
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    }

    # Never overwrite existing recordings.
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Recording output appeared during collection.")

    np.savez_compressed(
        output_path,
        sequences=sequence_array,
        labels=np.full(
            len(sequence_array), label_id, dtype=np.int64
        ),
    )

    try:
        with metadata_path.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        # Append the manifest only after the recording files are saved.
        with MANIFEST_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metadata) + "\n")
    except Exception:
        # Remove only files created by this recording attempt.
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    print("\nCollection saved successfully.")
    print("Sequences:", len(sequence_array))
    print("Data:", output_path.relative_to(ROOT))
    print("Metadata:", metadata_path.relative_to(ROOT))
    print("Manifest:", MANIFEST_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
