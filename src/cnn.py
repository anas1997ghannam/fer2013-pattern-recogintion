import time

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

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


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(48, 48, 1)
            ),

            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dropout(0.30),

            tf.keras.layers.Dense(
                7,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


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
    # Reshape images
    # =========================

    X_train = X_train.reshape(
        -1, 48, 48, 1
    )

    X_val = X_val.reshape(
        -1, 48, 48, 1
    )

    X_test = X_test.reshape(
        -1, 48, 48, 1
    )

    print("\nCNN input shapes:")
    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    # =========================
    # Build model
    # =========================

    model = build_model()

    print("\nCNN Model:")
    model.summary()

    # =========================
    # Callbacks
    # =========================

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    # =========================
    # Training
    # =========================

    print("\nStarting CNN training...")

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=10,
        batch_size=128,
        callbacks=[
            early_stopping
        ],
        verbose=1,
    )

    training_time = time.time() - start_time

    print(
        f"\nTotal training time: "
        f"{training_time:.2f} seconds"
    )

    # =========================
    # Epoch results
    # =========================

    print("\n========================================")
    print("EPOCH RESULTS")
    print("========================================")

    print(
        f"{'Epoch':<8}"
        f"{'Train Loss':<15}"
        f"{'Train Acc':<15}"
        f"{'Val Loss':<15}"
        f"{'Val Acc':<15}"
    )

    for i in range(len(history.history["loss"])):
      print(
            f"{i + 1:<8}"
            f"{history.history['loss'][i]:<15.4f}"
            f"{history.history['accuracy'][i]:<15.4f}"
            f"{history.history['val_loss'][i]:<15.4f}"
            f"{history.history['val_accuracy'][i]:<15.4f}"
        )

    # =========================
    # Validation
    # =========================

    print("\n========================================")
    print("VALIDATION RESULTS")
    print("========================================")

    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0,
    )

    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # =========================
    # Test prediction
    # =========================

    print("\nEvaluating on TEST set...")

    start_time = time.time()

    y_probability = model.predict(
        X_test,
        verbose=0,
    )

    prediction_time = time.time() - start_time

    y_pred = np.argmax(
        y_probability,
        axis=1,
    )

    # =========================
    # Test metrics
    # =========================

    test_accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    test_macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    test_weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
    )

    print("\n========================================")
    print("FINAL TEST RESULTS")
    print("========================================")

    print(
        f"Test Accuracy:  {test_accuracy:.4f}"
    )

    print(
        f"Macro F1:       {test_macro_f1:.4f}"
    )

    print(
        f"Weighted F1:    {test_weighted_f1:.4f}"
    )

    print(
        f"Prediction time: {prediction_time:.2f} seconds"
    )

    # =========================
    # Classification report
    # =========================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    # =========================
    # Confusion matrix
    # =========================

    cm = confusion_matrix(
        y_test,
        y_pred,
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
            f"TP={value['TP']:<5}"
            f"TN={value['TN']:<5}"
            f"FP={value['FP']:<5}"
            f"FN={value['FN']:<5}"
        )


if __name__ == "__main__":
    main()