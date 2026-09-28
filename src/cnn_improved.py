
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


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),

        # Block 1
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 2
        tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 3
        tf.keras.layers.Conv2D(
            128, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(
            128, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Classification head
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.50),

        tf.keras.layers.Dense(7, activation="softmax"),
    ])

    return model


def main():
    print("Loading FER2013...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_fer2013()

    # Convert flat 48x48 vectors into image tensors.
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_val = X_val.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)

    print("\nInput shapes:")
    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel summary:")
    model.summary()

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ])

    # Augmentation is applied only to training data.
    augmented_train = (
        tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)
        )
        .shuffle(10000)
        .batch(128)
        .map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_data = (
        tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)
        )
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    print("\nStarting training...")

    start_time = time.time()

    history = model.fit(
        augmented_train,
        validation_data=validation_data,
        epochs=30,
        callbacks=[
            early_stopping,
            reduce_lr,
        ],
        verbose=1,
    )
    training_time = time.time() - start_time

    print(f"\nTraining time: {training_time:.2f} seconds")

    # Test evaluation
    print("\nEvaluating on test set...")

    start_prediction = time.time()

    y_prob = model.predict(
        X_test,
        batch_size=128,
        verbose=1,
    )

    prediction_time = time.time() - start_prediction

    y_pred = np.argmax(y_prob, axis=1)

    test_accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
    )

    print("\n" + "=" * 60)
    print("IMPROVED CNN RESULTS")
    print("=" * 60)

    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Macro F1:  {macro_f1:.4f}")
    print(f"Test Weighted F1: {weighted_f1:.4f}")
    print(f"Prediction time: {prediction_time:.2f} seconds")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nEpoch history:")

    for i in range(len(history.history["loss"])):
        print(
            f"Epoch {i + 1}: "
            f"loss={history.history['loss'][i]:.4f}, "
            f"acc={history.history['accuracy'][i]:.4f}, "
            f"val_loss={history.history['val_loss'][i]:.4f}, "
            f"val_acc={history.history['val_accuracy'][i]:.4f}"
        )


if __name__ == "__main__":
    main()