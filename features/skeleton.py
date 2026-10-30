import numpy as np


class SkeletonProcessor:
    """
    Converts pose-estimator output into camera-relative
    skeleton coordinates.

    Output shape: (17, 3)
    Features: normalized camera x, normalized camera y,
    and keypoint confidence.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def process(self, person, frame_shape=None):
        keypoints = np.asarray(
            person["keypoints"], dtype=np.float32
        )
        confidence = person.get("confidence")

        if keypoints.shape != (17, 2):
            raise ValueError(
                f"Expected keypoints (17, 2), got {keypoints.shape}"
            )

        if confidence is None:
            confidence = np.ones(17, dtype=np.float32)
        else:
            confidence = np.asarray(
                confidence, dtype=np.float32
            )

        if confidence.shape != (17,):
            raise ValueError(
                f"Expected confidence (17,), got {confidence.shape}"
            )

        if not np.isfinite(keypoints).all():
            raise ValueError("Keypoints contain invalid values")

        if not np.isfinite(confidence).all():
            raise ValueError("Confidence contains invalid values")

        if self.normalize:
            if frame_shape is None:
                raise ValueError(
                    "frame_shape=(height, width) is required "
                    "for camera-relative normalization"
                )

            height, width = frame_shape[:2]

            if height <= 0 or width <= 0:
                raise ValueError("Invalid frame dimensions")

            keypoints = keypoints.copy()
            keypoints[:, 0] /= width
            keypoints[:, 1] /= height

        confidence = np.clip(confidence, 0.0, 1.0)

        return np.concatenate(
            [keypoints, confidence[:, None]],
            axis=1,
        ).astype(np.float32)
