import json
from pathlib import Path

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
INFO_PATH = BASE_DIR / "model_info.json"
EPOCHS = 5
BATCH_SIZE = 128


def load_data():
    """Load and prepare the MNIST handwritten digit dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Scale pixels from 0-255 to 0-1 so the model trains more smoothly.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CNN layers expect images with a channel dimension: 28x28x1.
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Create a simple CNN for recognizing digits from 0 to 9."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    """Train the model, test it, and save the result."""
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

    print(f"Final training accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
    print(f"Testing accuracy: {test_accuracy * 100:.2f}%")
    print(f"Testing loss: {test_loss:.4f}")

    model.save(MODEL_PATH)

    model_details = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_accuracy": float(history.history["accuracy"][-1]),
        "validation_accuracy": float(history.history["val_accuracy"][-1]),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
    }

    with INFO_PATH.open("w", encoding="utf-8") as file:
        json.dump(model_details, file, indent=2)

    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Saved model details to {INFO_PATH}")


if __name__ == "__main__":
    main()
