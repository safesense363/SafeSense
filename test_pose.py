import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolo26n-pose.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Pose test started. Press Q to quit.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break

        results = model.predict(
            source=frame,
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow(
            "SafeSense - Pose Test",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()