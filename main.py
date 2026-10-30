"""Live webcam pose and activity recognition."""
import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import time

import cv2
import numpy as np
import torch

from features.sequence_builder import SkeletonSequenceBuilder
from features.skeleton import SkeletonProcessor
from models.ava_activity_predictor import AvaActivityPredictor
from models.runtime_device import pose_model_for_device, resolve_runtime_device
from pose.pose_estimator import PoseEstimator


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Open a webcam and label live human pose and activity."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; auto selects CUDA when available",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        help="Optional Ultralytics pose checkpoint override",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run inference without opening an OpenCV window",
    )
    return parser.parse_args()


def _extract_person_box(result, person_index=0):
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) <= person_index:
        return None
    return boxes.xyxy[person_index].detach().cpu().numpy().astype(np.float32)


def _render_live_frame(pose_result, person_count, activity, confidence):
    """Draw the pose and current activity status over the camera frame."""
    annotated = pose_result.plot()
    confidence_label = f"{confidence:.0%}" if confidence is not None else "--"
    pose_label = (
        f"POSE: DETECTED ({person_count})"
        if person_count
        else "POSE: NOT DETECTED"
    )
    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1], 58),
        (20, 25, 35),
        thickness=-1,
    )
    cv2.putText(
        annotated,
        f"{pose_label} | ACTIVITY: {activity} ({confidence_label})",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Q / Esc: quit",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )
    if person_count == 0:
        cv2.putText(
            annotated,
            "Move into view to start pose detection",
            (12, annotated.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_live(args):
    device = resolve_runtime_device(args.device)
    pose_checkpoint = args.pose_model or pose_model_for_device(device)
    if device.type == "cuda":
        print(
            f"Using CUDA ({torch.cuda.get_device_name(device)}); "
            f"pose model: {pose_checkpoint}"
        )
    else:
        print(f"Using CPU; pose model: {pose_checkpoint}")
    print("Loading pretrained AVA activity model (first run downloads weights).")

    pose_estimator = PoseEstimator(
        model_path=pose_checkpoint,
        confidence=0.35,
        device=str(device),
    )
    activity_predictor = AvaActivityPredictor(device=str(device))
    skeleton_processor = SkeletonProcessor(normalize=True)
    sequence_builder = SkeletonSequenceBuilder(window_size=30, stride=5)
    recent_frames = deque(maxlen=8)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="activity")
    activity_future: Future | None = None
    activity = "WARMING UP"
    confidence = None
    frame_index = 0
    inference_count = 0
    started = time.monotonic()
    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        camera.release()
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(
            f"Could not open camera {args.camera}. Close other camera apps, "
            "check Windows camera permissions, or try --camera 1."
        )

    try:
        if not args.no_display:
            cv2.namedWindow("SafeSense Live Pose and Activity", cv2.WINDOW_NORMAL)

        print("Live detection started. Press Q or Esc in the camera window to stop.")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            frame_index += 1

            pose_result = pose_estimator.process(frame)
            persons = pose_estimator.extract_keypoints(pose_result)

            if not persons:
                sequence_builder.reset()
                recent_frames.clear()
                activity = "NO PERSON"
                confidence = None
            else:
                if activity == "NO PERSON":
                    activity = "WARMING UP"
                skeleton = skeleton_processor.process(persons[0])
                sequence = sequence_builder.add_skeleton(skeleton)
                recent_frames.append(frame.copy())

                if (
                    sequence is not None
                    and activity_future is None
                ):
                    person_box = _extract_person_box(pose_result)
                    if person_box is not None:
                        activity_future = executor.submit(
                            activity_predictor.predict_clip,
                            list(recent_frames),
                            person_box,
                        )

            if activity_future is not None and activity_future.done():
                prediction = activity_future.result()
                if persons:
                    activity = prediction.activity
                    confidence = prediction.confidence
                inference_count += 1
                activity_future = None
                if persons:
                    print(
                        f"Activity: {activity} ({confidence:.0%}) "
                        f"| inference: {prediction.latency_ms:.0f} ms"
                    )

            if not args.no_display:
                annotated = _render_live_frame(
                    pose_result,
                    len(persons),
                    activity,
                    confidence,
                )
                cv2.imshow("SafeSense Live Pose and Activity", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(
                    "SafeSense Live Pose and Activity",
                    cv2.WND_PROP_VISIBLE,
                ) < 1:
                    break

    except KeyboardInterrupt:
        print("\nStopping live detection.")
    finally:
        camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        executor.shutdown(wait=True, cancel_futures=True)
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"Stopped. Frames: {frame_index}, activity predictions: "
            f"{inference_count}, camera FPS: {frame_index / elapsed:.1f}"
        )


def main():
    run_live(_parse_args())


if __name__ == "__main__":
    main()
