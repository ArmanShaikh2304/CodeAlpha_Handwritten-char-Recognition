import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
INFO_PATH = BASE_DIR / "model_info.json"

app = Flask(__name__)
model = None


def load_digit_model():
    """Load the trained Keras model once and reuse it for each request."""
    global model

    if model is None:
        if not MODEL_PATH.exists():
            return None
        model = tf.keras.models.load_model(MODEL_PATH)

    return model


def get_model_info():
    """Read saved model accuracy details created by train_model.py."""
    if not INFO_PATH.exists():
        return {}

    with INFO_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def decode_canvas_image(image_data):
    """Decode a browser canvas data URL into a Pillow image."""
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA")


def preprocess_image(image):
    """
    Convert the drawing to a 28x28 MNIST-style image.

    The frontend draws white strokes on a black canvas, matching MNIST's
    white-on-black style. We crop the drawing, resize it, center it, and then
    normalize pixel values for the CNN.
    """
    black_background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    black_background.alpha_composite(image)

    gray_image = np.array(black_background.convert("L"))
    _, threshold = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(threshold)

    if points is None:
        raise ValueError("Please draw a digit before predicting.")

    x, y, width, height = cv2.boundingRect(points)
    digit = gray_image[y : y + height, x : x + width]

    scale = 20.0 / max(width, height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))

    digit = cv2.resize(
        digit,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    centered_image = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - resized_width) // 2
    y_offset = (28 - resized_height) // 2
    centered_image[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = digit

    normalized_image = centered_image.astype("float32") / 255.0
    model_input = normalized_image.reshape(1, 28, 28, 1)

    return model_input, centered_image


def image_array_to_data_url(image_array):
    """Return a larger PNG preview of the 28x28 processed image."""
    preview = cv2.resize(image_array, (140, 140), interpolation=cv2.INTER_NEAREST)
    preview_image = Image.fromarray(preview, mode="L")

    buffer = io.BytesIO()
    preview_image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"


@app.route("/")
def home():
    """Render the drawing page."""
    model_info = get_model_info()
    accuracy = model_info.get("test_accuracy")
    accuracy_text = None

    if accuracy is not None:
        accuracy_text = f"{accuracy * 100:.2f}%"

    return render_template("index.html", model_accuracy=accuracy_text)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive a drawn digit, process it, and return CNN predictions."""
    digit_model = load_digit_model()

    if digit_model is None:
        return (
            jsonify(
                {
                    "error": "model.h5 was not found. Run python train_model.py first."
                }
            ),
            503,
        )

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "No image data was received."}), 400

    try:
        image = decode_canvas_image(image_data)
        processed_image, preview_image = preprocess_image(image)
    except Exception as error:
        return jsonify({"error": str(error)}), 400

    predictions = digit_model.predict(processed_image, verbose=0)[0]
    predicted_digit = int(np.argmax(predictions))
    confidence = float(predictions[predicted_digit])

    return jsonify(
        {
            "digit": predicted_digit,
            "confidence": confidence,
            "probabilities": [float(value) for value in predictions],
            "processed_image": image_array_to_data_url(preview_image),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
