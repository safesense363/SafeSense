import cv2

from pose.pose_estimator import PoseEstimator


def main():

    estimator = PoseEstimator()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Keypoint extraction test started.")
    print("Press Q to quit.")

    printed = False

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break

        result = estimator.process(frame)

        persons = estimator.extract_keypoints(result)

        if persons and not printed:

            person = persons[0]

            print("\nPerson detected!")

            print(
                "Keypoint shape:",
                person["keypoints"].shape
            )

            print(
                "Confidence shape:",
                person["confidence"].shape
                if person["confidence"] is not None
                else None
            )

            print(
                "First keypoint:",
                person["keypoints"][0]
            )

            print(
                "First confidence:",
                person["confidence"][0]
                if person["confidence"] is not None
                else None
            )

            printed = True

        annotated_frame = result.plot()

        cv2.imshow(
            "SafeSense - Keypoints",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()