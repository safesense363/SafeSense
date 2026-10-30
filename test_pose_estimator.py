import cv2

from pose.pose_estimator import PoseEstimator


def main():
    estimator = PoseEstimator()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("SafeSense PoseEstimator test started.")
    print("Press Q to quit.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break

        result = estimator.process(frame)

        annotated_frame = result.plot()

        cv2.imshow(
            "SafeSense - PoseEstimator",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()