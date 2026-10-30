
import threading
import time

from video_ingestion.inference_worker import InferenceWorker


def main():
    processed = []
    processed_event = threading.Event()

    def fake_inference(window):
        result = {
            "received": window,
            "status": "success",
        }
        processed.append(result)
        processed_event.set()
        return result

    worker = InferenceWorker(fake_inference)

    worker.start()
    worker.submit("test-skeleton-sequence")

    completed = processed_event.wait(timeout=3)

    worker.stop()

    assert completed, "Inference did not complete within 3 seconds."
    assert len(processed) == 1, f"Unexpected results: {processed}"
    assert processed[0]["received"] == "test-skeleton-sequence"

    print("\nInference worker test passed.")
    print("Processed:", processed[0])


if __name__ == "__main__":
    main()
