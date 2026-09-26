import numpy as np


def normalize_images(X):
    """
    Convert pixel values from uint8 [0, 255]
    to float32 [0, 1].
    """

    X = X.astype(np.float32)
    X = X / 255.0

    return X


if __name__ == "__main__":
    # Small test example
    sample = np.array([0, 128, 255], dtype=np.uint8)

    print("Before preprocessing:")
    print(sample)
    print("Data type:", sample.dtype)

    processed = normalize_images(sample)

    print("\nAfter preprocessing:")
    print(processed)
    print("Data type:", processed.dtype)
