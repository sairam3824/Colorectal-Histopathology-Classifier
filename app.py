from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "colorectal_cancer_cnn.h5"
CLASS_NAMES = [
    "tumor",
    "stroma",
    "complex",
    "lympho",
    "debris",
    "mucosa",
    "adipose",
    "empty",
]
CLASS_DESCRIPTIONS = {
    "tumor": "Suspicious gland-forming tumor tissue that may represent malignancy.",
    "stroma": "Supportive connective tissue surrounding glands and other structures.",
    "complex": "Mixed or architecturally complex tissue patterns with overlapping features.",
    "lympho": "Lymphocyte-rich immune tissue with dense inflammatory cell presence.",
    "debris": "Necrotic or inflammatory tissue fragments and degraded cellular material.",
    "mucosa": "Normal colorectal mucosal lining with preserved glandular architecture.",
    "adipose": "Fat tissue with large empty-appearing adipocyte spaces.",
    "empty": "Background or low-content image region with minimal diagnostic tissue.",
}

app = Flask(__name__)


def build_model_architecture() -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=(150, 150, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )


def load_keras_model(model_path: Path) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        fallback_model = build_model_architecture()
        fallback_model(tf.zeros((1, 150, 150, 3), dtype=tf.float32))
        fallback_model.load_weights(model_path)
        return fallback_model


model = load_keras_model(MODEL_PATH)


def get_target_size() -> tuple[int, int]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height, width = input_shape[1], input_shape[2]
    if not height or not width:
        raise ValueError(f"Unable to infer model input size from shape: {input_shape}")
    return int(height), int(width)


TARGET_SIZE = get_target_size()


def find_last_conv_layer_name() -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the loaded model.")


LAST_CONV_LAYER_NAME = find_last_conv_layer_name()


def build_gradcam_models() -> tuple[tf.keras.Model, tf.keras.Model]:
    last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)
    last_conv_index = model.layers.index(last_conv_layer)

    conv_model = tf.keras.models.Model(
        inputs=model.inputs[0],
        outputs=last_conv_layer.output,
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[last_conv_index + 1 :]:
        x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    return conv_model, classifier_model


LAST_CONV_MODEL, CLASSIFIER_MODEL = build_gradcam_models()


def preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def create_heatmap(image_tensor: np.ndarray, class_index: int) -> np.ndarray:
    with tf.GradientTape() as tape:
        conv_outputs = LAST_CONV_MODEL(image_tensor, training=False)
        tape.watch(conv_outputs)
        predictions = CLASSIFIER_MODEL(conv_outputs, training=False)
        class_channel = predictions[:, class_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    if gradients is None:
        raise ValueError("Grad-CAM gradients could not be computed for this image.")
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap /= max_value
    return heatmap.numpy()


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    red = np.clip(1.5 - np.abs((4 * heatmap) - 3), 0, 1)
    green = np.clip(1.5 - np.abs((4 * heatmap) - 2), 0, 1)
    blue = np.clip(1.5 - np.abs((4 * heatmap) - 1), 0, 1)
    return np.stack([red, green, blue], axis=-1)


def build_gradcam_overlay(original_image: Image.Image, heatmap: np.ndarray) -> str:
    original_rgb = original_image.convert("RGB")
    original_array = np.asarray(original_rgb, dtype=np.float32) / 255.0

    heatmap_image = Image.fromarray(np.uint8(heatmap * 255), mode="L").resize(
        original_rgb.size,
        Image.Resampling.BILINEAR,
    )
    heatmap_resized = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    colored_heatmap = apply_colormap(heatmap_resized)

    alpha = np.clip(heatmap_resized[..., None] * 0.65 + 0.15, 0, 0.75)
    overlay = (1 - alpha) * original_array + alpha * colored_heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def serialize_original_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.get("/")
def index():
    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        target_width=TARGET_SIZE[1],
        target_height=TARGET_SIZE[0],
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_path": str(MODEL_PATH.name),
            "input_size": {"height": TARGET_SIZE[0], "width": TARGET_SIZE[1]},
            "last_conv_layer": LAST_CONV_LAYER_NAME,
        }
    )


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file was provided."}), 400

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "Please choose an image file to analyze."}), 400

    try:
        original_image = Image.open(image_file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "The uploaded file could not be read as an image."}), 400

    image_tensor = preprocess_image(original_image)
    probabilities = model.predict(image_tensor, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    heatmap = create_heatmap(image_tensor, predicted_index)
    gradcam_image = build_gradcam_overlay(original_image, heatmap)

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {
            "label": CLASS_NAMES[index],
            "confidence": round(float(probabilities[index]), 4),
        }
        for index in top_indices
    ]

    return jsonify(
        {
            "prediction": {
                "label": predicted_label,
                "confidence": round(confidence, 4),
                "description": CLASS_DESCRIPTIONS[predicted_label],
            },
            "top_predictions": top_predictions,
            "gradcam_image": gradcam_image,
            "original_image": serialize_original_image(original_image),
            "model": {
                "input_size": {"height": TARGET_SIZE[0], "width": TARGET_SIZE[1]},
                "last_conv_layer": LAST_CONV_LAYER_NAME,
            },
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    app.run(
        host="127.0.0.1",
        port=args.port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
# patch 5
