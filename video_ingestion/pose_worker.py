
import threading
import queue

from video_ingestion.frame_queue import LatestFrameQueue


class PoseProcessingWorker:
    """Processes the latest available camera frame in a separate thread."""

    def __init__(self, process_frame):
        self.process_frame = process_frame
        self.frame_queue = LatestFrameQueue()
        self.running = False
        self.thread = None

    def submit(self, frame):
        """Called by the camera thread to submit a captured frame."""
        self.frame_queue.put(frame)

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        self.thread.start()

    def _process_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.process_frame(frame)
            except Exception as exc:
                print(f"Pose processing error: {exc}")

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=3)

        self.frame_queue.clear()
