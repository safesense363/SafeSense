
import numpy as np

from features.sequence_builder import SkeletonSequenceBuilder


def main():
    builder = SkeletonSequenceBuilder(
        window_size=30,
        stride=5,
    )

    emitted_at = []

    for skeleton_number in range(1, 41):
        skeleton = np.random.rand(17, 3).astype(np.float32)
        sequence = builder.add_skeleton(skeleton)

        if sequence is not None:
            emitted_at.append(skeleton_number)

            assert sequence.shape == (30, 17, 3)
            assert sequence.dtype == np.float32

            print(
                f"Skeleton {skeleton_number}: "
                f"sequence shape={sequence.shape}, "
                f"dtype={sequence.dtype}"
            )

    assert emitted_at == [30, 35, 40], (
        f"Unexpected sequence positions: {emitted_at}"
    )

    print("\nAll sequence-builder tests passed.")
    print("Sequences emitted at:", emitted_at)

def test_reset_behavior():
    builder = SkeletonSequenceBuilder(window_size=30, stride=5)

    skeleton = np.random.rand(17, 3).astype(np.float32)

    # Add 20 skeletons — not enough for a full sequence.
    for _ in range(20):
        assert builder.add_skeleton(skeleton) is None

    # Simulate a person disappearing from the frame.
    builder.reset()

    # After reset, 29 skeletons must still be insufficient.
    for _ in range(29):
        assert builder.add_skeleton(skeleton) is None

    # The 30th fresh skeleton must produce a sequence.
    sequence = builder.add_skeleton(skeleton)

    assert sequence is not None
    assert sequence.shape == (30, 17, 3)
    assert sequence.dtype == np.float32

    print("Reset behavior test passed.")


if __name__ == "__main__":
    main()
    test_reset_behavior()
