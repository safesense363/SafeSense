import cv2
import numpy as np

from pose.pose_estimator import PoseEstimator
from features.skeleton import SkeletonProcessor


def main():

    pose_estimator = PoseEstimator()
    skeleton_processor = SkeletonProcessor()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Skeleton processing test started.")
    print("Press Q to quit.")

    tested = False

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break

        result = pose_estimator.process(frame)

        persons = pose_estimator.extract_keypoints(result)

        if persons and not tested:

            skeleton = skeleton_processor.process(
                persons[0]
            )

            print("\nSkeleton generated!")

            print(
                "Shape:",
                skeleton.shape
            )

            print(
                "Data type:",
                skeleton.dtype
            )

            print(
                "First joint:",
                skeleton[0]
            )

            print(
                "Coordinate range:",
                skeleton[:, :2].min(),
                "to",
                skeleton[:, :2].max()
            )

            print(
                "Confidence range:",
                skeleton[:, 2].min(),
                "to",
                skeleton[:, 2].max()
            )

            tested = True

        annotated_frame = result.plot()

        cv2.imshow(
            "SafeSense - Skeleton",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()