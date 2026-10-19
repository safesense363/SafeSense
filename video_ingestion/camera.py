import cv2
import time
import threading
import platform


class Camera:

    def __init__(self, camera_id=0, width=640, height=480, fps=15):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.running = False
        self.thread = None
        self.frame_callback = None

    def start(self, frame_callback):

        self.frame_callback = frame_callback

        # DirectShow is the most reliable OpenCV backend for local webcams on
        # Windows. Fall back to MSMF and then OpenCV's automatic selection.
        backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                    if platform.system() == "Windows" else [cv2.CAP_ANY])
        backend_names = {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "Media Foundation", cv2.CAP_ANY: "automatic"}
        failures = []
        for backend in backends:
            cap = cv2.VideoCapture(self.camera_id, backend)
            if cap.isOpened():
                self.cap = cap
                break
            failures.append(backend_names.get(backend, str(backend)))
            cap.release()

        if self.cap is None:
            raise RuntimeError(
                f"Could not open camera index {self.camera_id}. Tried OpenCV backends "
                f"{failures}. Check that the webcam is connected, not in use by another "
                "app, and allowed in Windows Settings > Privacy & security > Camera. "
                "Try another index with --camera 1."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

    def _capture_loop(self):

        frame_interval = 1.0 / self.fps

        while self.running:

            start_time = time.perf_counter()

            ret, frame = self.cap.read()

            if not ret:
                print("Camera opened but returned no frame; check the camera connection/permissions.")
                time.sleep(0.1)
                continue

            if self.frame_callback:
                self.frame_callback(frame)

            elapsed = time.perf_counter() - start_time

            sleep_time = max(
                0,
                frame_interval - elapsed
            )

            time.sleep(sleep_time)

    def stop(self):

        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.cap:
            self.cap.release()
