import time

import numpy as np
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import LinearSVC

from data_acquisition import load_fer2013


CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def extract_hog_features(X):
    """
    Extract HOG features from FER2013 images.
    """

    # Convert flattened 48x48 images back to image format
    images = X.reshape(-1, 48, 48)

    features = []

    for image in images:
        feature = hog(
            image,
            orientations=9,
            pixels_per_cell=(6, 6),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )

        features.append(feature)

    return np.array(features, dtype=np.float32)


def calculate_confusion_values(cm):
    """
    Calculate TP, TN, FP, and FN for each class
    using a one-vs-rest approach.
    """

    total = np.sum(cm)

    results = []

    for i in range(len(CLASS_NAMES)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total - tp - fn - fp

        results.append({
            "class": CLASS_NAMES[i],
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        })

    return results


def main():

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_fer2013()

    print("\n" + "=" * 60)
    print("HOG + LINEAR SVM")
    print("=" * 60)

    # HOG feature extraction
    print("\nExtracting HOG features...")

    start_time = time.time()

    X_train_hog = extract_hog_features(X_train)
    X_val_hog = extract_hog_features(X_val)
    X_test_hog = extract_hog_features(X_test)

    hog_time = time.time() - start_time

    print(f"HOG extraction time: {hog_time:.2f} seconds")

    print("\nHOG feature shapes:")
    print(f"Training:   {X_train_hog.shape}")
    print(f"Validation: {X_val_hog.shape}")
    print(f"Test:       {X_test_hog.shape}")

    # Linear SVM
    model = LinearSVC(
        C=1.0,
        max_iter=3000,
        random_state=42,
    )

    print("\nTraining Linear SVM...")

    start_time = time.time()

    model.fit(X_train_hog, y_train)

    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")

    # Validation
    start_time = time.time()

    y_val_pred = model.predict(X_val_hog)

    validation_prediction_time = time.time() - start_time

    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(
        f"\nValidation prediction time: "
        f"{validation_prediction_time:.2f} seconds"
    )

    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Test
    start_time = time.time()

    y_test_pred = model.predict(X_test_hog)

    test_prediction_time = time.time() - start_time

    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTest prediction time: {test_prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # TP / TN / FP / FN
    confusion_values = calculate_confusion_values(cm)

    print("\nTP / TN / FP / FN:")
    print("-" * 70)

    for result in confusion_values:
        print(
            f"{result['class']:10s} | "
            f"TP: {result['TP']:5d} | "
            f"TN: {result['TN']:5d} | "
            f"FP: {result['FP']:5d} | "
            f"FN: {result['FN']:5d}"
        )


if __name__ == "__main__":
    main()
