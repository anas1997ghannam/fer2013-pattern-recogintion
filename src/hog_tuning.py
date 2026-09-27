import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from data_acquisition import load_fer2013
from skimage.feature import hog


def extract_hog_features(X):
    """
    Extract HOG features from FER2013 images.
    """

    X_images = X.reshape(-1, 48, 48)

    features = []

    for image in X_images:
        feature = hog(
            image,
            orientations=9,
            pixels_per_cell=(6, 6),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )

        features.append(feature)

    return np.asarray(features, dtype=np.float32)


def main():

    # =========================
    # 1. Load dataset
    # =========================

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_fer2013()

    # =========================
    # 2. HOG feature extraction
    # =========================

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

    # =========================
    # 3. Hyperparameter tuning
    # =========================

    C_VALUES = [0.01, 0.1, 1.0, 10.0]

    results = []

    print("\n==============================")
    print("HOG + LINEAR SVM TUNING")
    print("==============================")

    for C in C_VALUES:

        print(f"\nTesting C = {C}")

        start_time = time.time()

        model = LinearSVC(
            C=C,
            max_iter=3000,
            random_state=42,
        )

        model.fit(X_train_hog, y_train)

        training_time = time.time() - start_time

        # Validation prediction
        y_val_pred = model.predict(X_val_hog)

        val_accuracy = accuracy_score(
            y_val,
            y_val_pred,
        )

        val_macro_f1 = f1_score(
            y_val,
            y_val_pred,
            average="macro",
        )

        results.append(
            {
                "C": C,
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": val_macro_f1,
                "training_time": training_time,
            }
        )

        print(f"Training time:       {training_time:.2f} seconds")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro F1: {val_macro_f1:.4f}")

    # =========================
    # 4. Print comparison
    # =========================

    print("\n==============================================")
    print("HYPERPARAMETER TUNING RESULTS")
    print("==============================================")

    print(
        f"{'C':<10}"
        f"{'Val Accuracy':<18}"
        f"{'Val Macro F1':<18}"
        f"{'Training Time':<15}"
    )

    for result in results:

        print(
            f"{result['C']:<10}"
            f"{result['validation_accuracy']:<18.4f}"
            f"{result['validation_macro_f1']:<18.4f}"
            f"{result['training_time']:<15.2f}"
        )

    # =========================
    # 5. Select best C
    # =========================

    best_result = max(
        results,
        key=lambda result: result["validation_accuracy"],
    )

    best_C = best_result["C"]

    print("\n==============================================")
    print("BEST HYPERPARAMETER")
    print("==============================================")

    print(f"Best C: {best_C}")
    print(
        f"Validation accuracy: "
        f"{best_result['validation_accuracy']:.4f}"
    )

    print(
        f"Validation macro F1: "
        f"{best_result['validation_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
