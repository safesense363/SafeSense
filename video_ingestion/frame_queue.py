
import queue
import threading


class LatestFrameQueue:
    """Thread-safe queue that retains only the newest pending frame."""

    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()

    def put(self, frame):
        """Add a frame, replacing an older pending frame if necessary."""
        with self._lock:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                # Another consumer or producer may have changed the queue.
                pass

    def get(self, timeout=None):
        """Return the next available frame."""
        return self._queue.get(timeout=timeout)

    def clear(self):
        """Remove any pending frame."""
        with self._lock:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
