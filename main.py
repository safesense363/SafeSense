"""SafeSense AI — Main Pipeline Entry Point.

Launches the full real-time camera → pose → activity → risk pipeline and
publishes each inference result to the FastAPI event store so the live
dashboard updates automatically.

Usage::

    # Live camera (default)
    python main.py

    # Named camera index
    python main.py --camera 1

    # Use a different trained model checkpoint
    python main.py --model models/safesense_gru_baseline.pt

    # Demo mode — streams synthetic events; no camera required
    python main.py --demo

    # Run pipeline only, no API push (standalone / offline)
    python main.py --no-api

    # Quiet (suppress per-frame console prints)
    python main.py --quiet
"""

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import time
from collections import deque

import cv2
import numpy as np
import torch

import cv2
import numpy as np
import torch

from features.sequence_builder import SkeletonSequenceBuilder
from features.skeleton import SkeletonProcessor
from models.ava_activity_predictor import AvaActivityPredictor
from models.activity_predictor import ActivityPredictor
from models.runtime_device import pose_model_for_device, resolve_runtime_device
from pose.pose_estimator import PoseEstimator


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Open a webcam and label live human pose and activity."
    )
    p.add_argument("--camera",  type=int,   default=0,
                   help="OpenCV camera device index")
    p.add_argument("--model",   type=str,   default="models/safesense_gru_baseline.pt",
                   help="Path to the GRU checkpoint (.pt)")
    p.add_argument("--activity-model", choices=("ava", "gru"), default="ava",
                   help="Activity recognizer: AVA pretrained RGB action detector or local GRU")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Inference device; auto uses CUDA when available")
    p.add_argument("--api-url", type=str,   default="http://localhost:8000/api/events",
                   help="FastAPI event ingestion endpoint")
    p.add_argument("--no-api",  action="store_true",
                   help="Disable API publishing (disk-only mode)")
    p.add_argument("--demo",    action="store_true",
                   help="Demo mode: stream synthetic events without a real camera")
    p.add_argument("--quiet",   action="store_true",
                   help="Suppress per-inference console output")
    p.add_argument("--fps",     type=int,   default=15,
                   help="Camera capture frame rate")
    p.add_argument("--no-display", action="store_true",
                   help="Run inference without opening the native camera window")
    return p.parse_args()


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

def _run_live(args: argparse.Namespace, publisher: EventPublisher) -> None:
    """Start the camera, pose estimator, activity model, and risk pipeline."""
    device = resolve_runtime_device(args.device)
    pose_model = pose_model_for_device(device)
    if device.type == "cuda":
        print(
            f"CUDA detected ({torch.cuda.get_device_name(device)}); "
            f"using {pose_model} and running activity inference on {device}."
        )
    else:
        print(f"Using CPU inference and lightweight pose model {pose_model}.")

    activity_model = getattr(args, "activity_model", "ava")
    if activity_model == "ava":
        print("Loading pretrained AVA person-action model (first run downloads its weights).")
        activity_predictor = AvaActivityPredictor(device=str(device))
    else:
        model_path = getattr(args, "model", "models/safesense_gru_baseline.pt")
        print(f"Loading GRU model: {model_path} on {device}")
        activity_predictor = ActivityPredictor(
            model_path=model_path,
            device=str(device),
        )
    risk_analyzer = RiskAnalyzer()

    display_lock = threading.Lock()
    display_state = {
        "activity": "WARMING UP",
        "confidence": None,
        "risk_level": "MONITORING",
    }
    latest_display_frame = {"frame": None}

    def update_display_state(event):
        with display_lock:
            display_state.update(
                activity=event["activity"],
                confidence=event["confidence"],
                risk_level=event["risk_level"],
            )

    quiet = getattr(args, "quiet", False)
    no_display = getattr(args, "no_display", False)
    no_api = getattr(args, "no_api", False)
    api_url = getattr(args, "api_url", "http://localhost:8000/api/events")
    camera_idx = getattr(args, "camera", 0)
    fps = getattr(args, "fps", 15)

    inference_function = _make_inference_function(
        activity_predictor,
        risk_analyzer,
        publisher,
        quiet=quiet,
        on_result=update_display_state,
    )

    pose_estimator = PoseEstimator(
        model_path=pose_model,
        confidence=0.35,
        device=str(device),
    )
    skeleton_processor = SkeletonProcessor(normalize=True)
    sequence_builder   = SkeletonSequenceBuilder(window_size=30, stride=5)
    inference_worker   = InferenceWorker(inference_function=inference_function)
    sequence_lock      = threading.Lock()
    recent_frames = deque(maxlen=8)

    counters = {"frames": 0, "poses": 0, "sequences": 0}

    def process_frame(frame):
        counters["frames"] += 1
        result  = pose_estimator.process(frame)
        persons = pose_estimator.extract_keypoints(result)

        if not persons:
            with sequence_lock:
                sequence_builder.reset()
            recent_frames.clear()
            with display_lock:
                display_state.update(
                    activity="NO PERSON",
                    confidence=None,
                    risk_level="MONITORING",
                )
        else:
            with display_lock:
                if display_state["activity"] == "NO PERSON":
                    display_state.update(
                        activity="WARMING UP",
                        confidence=None,
                        risk_level="MONITORING",
                    )
            counters["poses"] += 1
            skeleton = skeleton_processor.process(persons[0], frame_shape=frame.shape)
            recent_frames.append(frame.copy())

            with sequence_lock:
                sequence = sequence_builder.add_skeleton(skeleton)

            if sequence is not None:
                counters["sequences"] += 1
                if args.activity_model == "ava":
                    person_box = _extract_person_box(result, 0)
                    if person_box is None:
                        with display_lock:
                            display_state.update(
                                activity="NO PERSON BOX",
                                confidence=None,
                                risk_level="MONITORING",
                            )
                    else:
                        inference_worker.submit({
                            "skeleton_sequence": sequence,
                            "rgb_frames": list(recent_frames),
                            "person_box": person_box,
                        })
                else:
                    inference_worker.submit(sequence)

        with display_lock:
            state = display_state.copy()
        annotated = _render_live_frame(result, len(persons), state)
        with display_lock:
            latest_display_frame["frame"] = annotated

    pose_worker = PoseProcessingWorker(process_frame)
    camera = Camera(camera_id=args.camera, width=640, height=480, fps=args.fps)
    display_window = "SafeSense AI - Live Pose"
    display_window_created = False

    try:
        if not args.no_display:
            cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)
            display_window_created = True

        inference_worker.start()
        pose_worker.start()
        camera.start(pose_worker.submit)

        print("SafeSense AI pipeline running. Press Ctrl+C to stop.")
        if not args.no_display:
            print("Live pose window opened. Press Q or Esc in the window to stop.")
        if not args.no_api:
            print(f"Publishing events to: {args.api_url}")
        print()

        stop_event = threading.Event()
        while not stop_event.wait(timeout=0.01):
            if args.no_display:
                continue

            with display_lock:
                frame = latest_display_frame["frame"]
                if frame is not None:
                    frame = frame.copy()

            if frame is not None:
                cv2.imshow(display_window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(display_window, cv2.WND_PROP_VISIBLE) < 1:
                    break

    except KeyboardInterrupt:
        print("\nStopping SafeSense AI...")

    finally:
        camera.stop()
        pose_worker.stop()
        inference_worker.stop()
        if display_window_created:
            cv2.destroyAllWindows()

        print(f"\nSession summary:")
        print(f"  Frames captured  : {counters['frames']}")
        print(f"  Poses detected   : {counters['poses']}")
        print(f"  Sequences inferred: {counters['sequences']}")


def _make_inference_function(
    activity_predictor,
    risk_analyzer,
    publisher,
    quiet=True,
    on_result=None,
):
    """Build the shared live inference stage for a sequence of skeletons."""
    def inference_function(inference_input):
        t_event = time.time()
        if isinstance(inference_input, dict):
            sequence = inference_input["skeleton_sequence"]
            activity_pred = activity_predictor.predict_clip(
                inference_input["rgb_frames"],
                inference_input["person_box"],
            )
        else:
            sequence = inference_input
            activity_pred = activity_predictor.predict(sequence)
        risk_result    = risk_analyzer.analyze(sequence, activity_prediction=activity_pred)

        event = {
            "timestamp":       t_event,
            "model":           getattr(activity_predictor, "model_name", "Activity model"),
            "activity":        activity_pred.activity,
            "confidence":      activity_pred.confidence,
            "risk_level":      risk_result.level,
            "risk_score":      risk_result.score,
            "reasons":         risk_result.reasons,
            "alert_triggered": risk_result.alert_triggered,
            "incident_type":   risk_result.incident_type,
            "latency_ms":      activity_pred.latency_ms,
            "probabilities":   activity_pred.probabilities,
            "sequence_shape":  list(sequence.shape),
        }

        if on_result is not None:
            on_result(event)

        # Push to API / fallback to disk
        publisher.publish(event)

        if not quiet:
            if risk_result.level in ("CRITICAL", "HIGH"):
                print(f"\n[ALERT - {risk_result.level}] {activity_pred.activity} "
                      f"({activity_pred.confidence:.1%}) | score={risk_result.score} "
                      f"| {risk_result.reasons[0]}")
            elif risk_result.level == "WARNING":
                print(f"[WARN] {activity_pred.activity} ({activity_pred.confidence:.1%}) "
                      f"| score={risk_result.score}")
            else:
                print(f"[LIVE] {activity_pred.activity} ({activity_pred.confidence:.1%}) "
                      f"| risk={risk_result.level} | {activity_pred.latency_ms:.1f}ms")

        return event
    return inference_function


def _extract_person_box(pose_result, person_index):
    """Get a detected person's xyxy box in original-frame pixels."""
    boxes = getattr(pose_result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) <= person_index:
        return None
    return boxes.xyxy[person_index].detach().cpu().numpy().astype(np.float32)


def _render_live_frame(pose_result, person_count, display_state_or_activity, confidence=None):
    """Render pose landmarks and the latest temporal activity prediction."""
    if isinstance(display_state_or_activity, dict):
        activity = display_state_or_activity["activity"]
        confidence = display_state_or_activity["confidence"]
        risk_level = display_state_or_activity.get("risk_level")
        confidence_label = f"{confidence:.1%}" if confidence is not None else "--"
    else:
        activity = display_state_or_activity
        risk_level = None
        confidence_label = f"{int(round(confidence * 100))}%" if confidence is not None else "--"
    pose_label = (
        f"POSE: DETECTED ({person_count})"
        if person_count
        else "POSE: NOT DETECTED"
    )
    pose_activity_status = (
        f"{pose_label} | ACTIVITY: "
        f"{activity} ({confidence_label})"
    )

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 58), (20, 25, 35), -1)
    cv2.putText(
        annotated,
        pose_activity_status,
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
        (annotated.shape[1] - 120, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    if risk_level is not None:
        cv2.putText(
            annotated,
            f"RISK: {risk_level}",
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


# ─── Entry Point ──────────────────────────────────────────────────────────────

def run_live(args: argparse.Namespace, publisher: EventPublisher | None = None) -> None:
    """Entry point for live inference (used by tests and CLI)."""
    if publisher is None:
        publisher = EventPublisher(
            api_url=getattr(args, "api_url", "http://localhost:8000/api/events"),
            enabled=not getattr(args, "no_api", False),
        )
    return _run_live(args, publisher)


def main():
    run_live(_parse_args())


if __name__ == "__main__":
    main()
