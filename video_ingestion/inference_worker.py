
import threading
import queue


class InferenceWorker:

    def __init__(self, inference_function):
        self.inference_function = inference_function

        # Keep only the latest pending inference window.
        self.window_queue = queue.Queue(maxsize=1)

        self.running = False
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        if self.running:
            return

        self.stop_event.clear()
        self.running = True

        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.thread.start()

    def submit(self, window):
        # Do not accept new work after shutdown begins.
        if not self.running:
            return

        # Replace an older pending window with the latest one.
        if self.window_queue.full():
            try:
                self.window_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.window_queue.put_nowait(window)
        except queue.Full:
            pass

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                window = self.window_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self.inference_function(window)
                print(f"Inference completed: {result}")
            except Exception as exc:
                # One failed inference should not kill the worker.
                print(f"Inference error: {exc}")
            finally:
                self.window_queue.task_done()

    def stop(self):
        # Reject new submissions and signal the worker to stop.
        self.running = False
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            # Wait for any active inference call to finish.
            self.thread.join()

        # Discard pending work that was not started.
        while True:
            try:
                self.window_queue.get_nowait()
                self.window_queue.task_done()
            except queue.Empty:
                break

        self.thread = None
        print("Inference worker stopped.")
