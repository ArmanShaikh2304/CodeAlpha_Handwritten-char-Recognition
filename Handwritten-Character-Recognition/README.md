# Handwritten Character Recognition

A beginner-friendly Flask web app that recognizes handwritten digits from 0 to 9 using a TensorFlow/Keras CNN trained on the MNIST dataset.

## Features

- Draw a digit in the browser.
- Predict the digit with a trained CNN model.
- Show the predicted digit and confidence score.
- Display the preprocessed 28x28 image.
- Show a probability bar chart for all digits.
- Toggle between light and dark mode.
- Display saved model test accuracy after training.

## Project Structure

```text
Handwritten-Character-Recognition/
|-- app.py
|-- train_model.py
|-- model.h5
|-- model_info.json
|-- requirements.txt
|-- README.md
|-- templates/
|   `-- index.html
|-- static/
|   |-- style.css
|   |-- script.js
|   `-- canvas.js
`-- uploads/
    `-- .gitkeep
```

`model.h5` and `model_info.json` are created when you run the training script.

## Setup

Create and activate a virtual environment if you want to keep dependencies isolated.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

Train the CNN model on MNIST.

```bash
python train_model.py
```

Start the Flask app.

```bash
python app.py
```

If your system uses `python3` instead of `python`, run `python3 train_model.py` and `python3 app.py`.

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

## How It Works

1. `train_model.py` loads the MNIST dataset from TensorFlow/Keras.
2. Pixel values are normalized from `0-255` to `0-1`.
3. A simple CNN is trained with Conv2D, MaxPooling2D, Flatten, Dense, and Softmax layers.
4. The trained model is saved as `model.h5`.
5. `app.py` loads `model.h5` and serves the web interface.
6. The browser sends the canvas image to `/predict`.
7. Flask converts the image to grayscale, resizes it to 28x28, normalizes it, and sends it to the model.
8. The app returns the predicted digit, confidence score, preprocessed image, and all digit probabilities.

## Routes

- `/` renders the homepage.
- `/predict` accepts a drawn image and returns the prediction as JSON.

## Notes

- The canvas uses white strokes on a black background to match MNIST images.
- If the app says `model.h5 was not found`, run `python train_model.py` first.
- No login, database, or upload storage is required.
