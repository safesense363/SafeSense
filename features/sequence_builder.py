
from collections import deque

import numpy as np


class SkeletonSequenceBuilder:

    def __init__(self, window_size=30, stride=5):
        self.window_size = window_size
        self.stride = stride

        self.sequence = deque(maxlen=window_size)
        self.new_skeletons = 0

        # CHANGE: Track whether the first complete window
        # has already been returned.
        self.sequence_ready = False

    def add_skeleton(self, skeleton):

        skeleton = np.asarray(skeleton, dtype=np.float32)

        # Validate the input skeleton shape.
        if skeleton.shape != (17, 3):
            raise ValueError(
                f"Expected skeleton shape (17, 3), "
                f"got {skeleton.shape}"
            )

        self.sequence.append(skeleton)

        # Wait until the buffer contains 30 skeletons.
        if len(self.sequence) < self.window_size:
            return None

        # CHANGE: Return the first complete window immediately
        # when skeleton number 30 arrives.
        if not self.sequence_ready:
            self.sequence_ready = True
            return np.stack(list(self.sequence), axis=0)

        # Count new skeletons after the first window.
        self.new_skeletons += 1

        # Return a new sequence every 5 additional skeletons.
        if self.new_skeletons < self.stride:
            return None

        # Reset the counter for the next sequence.
        self.new_skeletons = 0

        # Return the latest sliding window.
        return np.stack(list(self.sequence), axis=0)
