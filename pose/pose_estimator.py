from ultralytics import YOLO


class PoseEstimator:
    """
    SafeSense pose-estimation interface.

    Detects human poses and extracts keypoints from
    individual video frames.
    """

    def __init__(
        self,
        model_path="yolo26n-pose.pt",
        confidence=0.5,
        device="cpu",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device

        self.model = YOLO(self.model_path)

    def process(self, frame):
        """
        Run pose estimation on a single frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            Raw pose result.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )

        return results[0]

    def extract_keypoints(self, result):
        """
        Extract keypoints from a pose result.

        Returns:
            List of detected persons.

            Each person contains:
                keypoints: (N, 2)
                confidence: (N,)
        """

        if result.keypoints is None:
            return []

        if result.keypoints.xy is None:
            return []

        keypoints_xy = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            keypoints_conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )
        else:
            keypoints_conf = None

        persons = []

        for person_index, person_keypoints in enumerate(
            keypoints_xy
        ):

            person = {
                "keypoints": person_keypoints,
                "confidence": (
                    keypoints_conf[person_index]
                    if keypoints_conf is not None
                    else None
                )
            }

            persons.append(person)

        return persons