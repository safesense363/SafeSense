import time

import cv2

from pose.pose_estimator import PoseEstimator
from features.skeleton import SkeletonProcessor
from features.sequence_builder import SkeletonSequenceBuilder


def main():

    pose_estimator = PoseEstimator()
    skeleton_processor = SkeletonProcessor()

    sequence_builder = SkeletonSequenceBuilder(
        window_size=30
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("SafeSense real-time skeleton pipeline started.")
    print("Waiting for a person...")
    print("Press Q to quit.")

    sequence_count = 0
    frame_count = 0

    start_time = time.perf_counter()

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break

        frame_count += 1

        # 1. Pose estimation
        result = pose_estimator.process(frame)

        # 2. Extract detected people
        persons = pose_estimator.extract_keypoints(
            result
        )

        if persons:

            # Stage 1: single-person pipeline
            person = persons[0]

            # 3. Convert pose to normalized skeleton
            skeleton = skeleton_processor.process(
                person
            )

            # 4. Add skeleton to temporal sequence
            sequence = sequence_builder.add_skeleton(
                skeleton
            )

            if sequence is not None:

                sequence_count += 1

                print(
                    f"Sequence {sequence_count}: "
                    f"shape={sequence.shape}"
                )

        # Display pose visualization
        annotated_frame = result.plot()

        cv2.imshow(
            "SafeSense - Real-Time Skeleton Pipeline",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.perf_counter() - start_time

    camera.release()
    cv2.destroyAllWindows()

    if elapsed > 0:

        fps = frame_count / elapsed

        print("\nPipeline stopped.")

        print(
            f"Frames processed: {frame_count}"
        )

        print(
            f"Sequences generated: {sequence_count}"
        )

        print(
            f"Average processing FPS: {fps:.2f}"
        )


if __name__ == "__main__":
    main()