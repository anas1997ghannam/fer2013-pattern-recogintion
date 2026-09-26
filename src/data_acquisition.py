import numpy as np
import pandas as pd

from preprocessing import normalize_images


DATA_PATH = "data/fer2013.csv"

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def load_fer2013():
    print("Loading FER2013 dataset...")

    df = pd.read_csv(DATA_PATH)

    # Convert pixel strings into numerical arrays
    X = np.array(
        df["pixels"].apply(
            lambda pixels: np.fromstring(pixels, sep=" ")
        ).tolist(),
        dtype=np.uint8,
    )

    # Emotion labels
    y = df["emotion"].to_numpy(dtype=np.int64)

    # Original FER2013 split
    usage = df["Usage"].to_numpy()

    train_mask = usage == "Training"
    val_mask = usage == "PublicTest"
    test_mask = usage == "PrivateTest"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # Pre-processing: normalization
    X_train = normalize_images(X_train)
    X_val = normalize_images(X_val)
    X_test = normalize_images(X_test)

    print("\nDataset information:")
    print(f"Total samples: {len(X)}")
    print("Image size: 48 x 48")
    print(f"Features per image: {X.shape[1]}")
    print(f"Number of classes: {len(CLASS_NAMES)}")

    print("\nSplits:")
    print(f"Training:   {X_train.shape}, {y_train.shape}")
    print(f"Validation: {X_val.shape}, {y_val.shape}")
    print(f"Test:       {X_test.shape}, {y_test.shape}")

    print("\nPre-processing:")
    print(f"Data type: {X_train.dtype}")
    print(f"Minimum pixel value: {X_train.min():.4f}")
    print(f"Maximum pixel value: {X_train.max():.4f}")

    print("\nClass distribution:")
    for label, name in enumerate(CLASS_NAMES):
        count = np.sum(y == label)
        print(f"{label} - {name}: {count}")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    load_fer2013()
