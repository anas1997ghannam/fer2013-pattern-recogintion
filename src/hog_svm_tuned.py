
import time

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC

from data_acquisition import load_fer2013
from hog_tuning import extract_hog_features


CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def calculate_confusion_values(cm):
    total = cm.sum()

    values = []

    for i in range(len(CLASS_NAMES)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        values.append(
            {
                "class": CLASS_NAMES[i],
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        )

    return values


def main():

    print("Loading FER2013 dataset...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_fer2013()

    # =========================
    # HOG feature extraction
    # =========================

    print("\nExtracting HOG features...")

    start_time = time.time()

    X_train_hog = extract_hog_features(X_train)
    X_val_hog = extract_hog_features(X_val)
    X_test_hog = extract_hog_features(X_test)

    hog_time = time.time() - start_time

    print(f"HOG extraction time: {hog_time:.2f} seconds")

    print("\nFeature shapes:")
    print(f"Training:   {X_train_hog.shape}")
    print(f"Validation: {X_val_hog.shape}")
    print(f"Test:       {X_test_hog.shape}")

    # =========================
    # Train final model
    # =========================

    print("\nTraining HOG + Linear SVM...")
    print("C = 0.1")

    start_time = time.time()

    model = LinearSVC(
        C=0.1,
        max_iter=3000,
        random_state=42,
    )

    model.fit(X_train_hog, y_train)

    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")

    # =========================
    # Validation evaluation
    # =========================

    y_val_pred = model.predict(X_val_hog)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_macro_f1 = f1_score(
        y_val,
        y_val_pred,
        average="macro",
    )

    print("\nValidation:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Macro F1: {val_macro_f1:.4f}")

    # =========================
    # Test evaluation
    # =========================

    print("\nEvaluating on TEST set...")

    start_time = time.time()

    y_test_pred = model.predict(X_test_hog)

    prediction_time = time.time() - start_time

    test_accuracy = accuracy_score(y_test, y_test_pred)

    test_macro_f1 = f1_score(
        y_test,
        y_test_pred,
        average="macro",
    )

    test_weighted_f1 = f1_score(
        y_test,
        y_test_pred,
        average="weighted",
    )

    print(f"Prediction time: {prediction_time:.2f} seconds")

    print("\n===================================")
    print("FINAL TEST RESULTS")
    print("===================================")

    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Macro F1:       {test_macro_f1:.4f}")
    print(f"Weighted F1:    {test_weighted_f1:.4f}")

    # =========================
    # Classification report
    # =========================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    # =========================
    # Confusion matrix
    # =========================

    cm = confusion_matrix(
        y_test,
        y_test_pred,
    )

    print("\nConfusion Matrix:")
    print(cm)

    # =========================
    # TP / TN / FP / FN
    # =========================

    print("\nTP / TN / FP / FN:")

    values = calculate_confusion_values(cm)
    for value in values:
        print(
            f"{value['class']:<10}"
            f"TP={value['TP']:<4}"
            f"TN={value['TN']:<4}"
            f"FP={value['FP']:<4}"
            f"FN={value['FN']:<4}"
        )


if __name__ == "__main__":
    main()