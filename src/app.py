"""Flask demo: image upload -> category -> calibrated confidence -> bin advice.

Run from the project root:
    py -3.13 src/app.py
Then open http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explain import heatmap_overlay, make_gradcam  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "experiments" / "models" / "final_model.keras"))
SUMMARY_PATH = ROOT / "experiments" / "results" / "summary.json"

model = keras.models.load_model(MODEL_PATH)
logit_predictor = keras.Model(model.input, model.get_layer("logits").output)
summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
class_names = summary.get("dataset", {}).get("classes", ["cardboard", "glass", "metal", "paper", "plastic", "trash"])
bin_mapping = summary.get("bin_mapping", {})
temperature = float(summary.get("temperature", 1.0))
threshold = float(summary.get("test", {}).get("abstention", {}).get("chosen_threshold", 0.70))
img_size = int(summary.get("settings", {}).get("img_size", 160))

STATIC = Path(app.static_folder)
SAMPLE_DIR = STATIC / "samples"


def preprocess(file_storage=None, sample_rel: str | None = None) -> np.ndarray | None:
    """Load an uploaded file or a bundled sample into a raw [0, 255] array."""
    if sample_rel:
        candidate = (STATIC / sample_rel).resolve()
        if STATIC.resolve() not in candidate.parents or not candidate.is_file():
            return None
        with Image.open(candidate) as image:
            return np.asarray(image.convert("RGB").resize((img_size, img_size)), dtype=np.float32)
    if file_storage is None:
        return None
    with Image.open(file_storage.stream) as image:
        return np.asarray(image.convert("RGB").resize((img_size, img_size)), dtype=np.float32)


@app.route("/", methods=["GET"])
def index():
    samples = [f"samples/{p.name}" for p in sorted(SAMPLE_DIR.glob("*.jpg"))] if SAMPLE_DIR.exists() else []
    return render_template("index.html", samples=samples)


@app.route("/predict", methods=["POST"])
def predict():
    sample_rel = request.form.get("sample", "").strip()
    if sample_rel:
        image = preprocess(sample_rel=sample_rel)
        if image is None:
            return render_template("index.html", error="That sample image could not be found.", samples=[])
    else:
        if "image" not in request.files or not request.files["image"].filename:
            return render_template("index.html", error="Please choose an image first.", samples=[])
        image = preprocess(file_storage=request.files["image"])
    Image.fromarray(image.astype(np.uint8)).save(STATIC / "uploaded.png")
    logits = logit_predictor.predict(np.expand_dims(image, 0), verbose=0)[0]
    z = logits / temperature
    z = z - z.max()
    probabilities = np.exp(z) / np.exp(z).sum()

    order = np.argsort(probabilities)[::-1]
    top = int(order[0])
    confidence = float(probabilities[top])
    category = class_names[top]
    mapping = bin_mapping.get(category, {"bin": "Check locally", "advice": "Ask the facility operator."})
    uncertain = confidence < threshold

    cam = make_gradcam(model, image, img_size)
    grad_path = STATIC / "gradcam.png"
    heatmap_overlay(image, cam).save(grad_path)

    return render_template(
        "index.html",
        uploaded_image="uploaded.png",
        gradcam_image="gradcam.png",
        category=category,
        confidence=round(confidence * 100, 1),
        bin_name=mapping["bin"],
        advice=mapping["advice"],
        uncertain=uncertain,
        threshold=round(threshold * 100, 1),
        top3=[(class_names[i], round(float(probabilities[i]) * 100, 1)) for i in order[:3]],
        samples=[],
    )


if __name__ == "__main__":
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(STATIC / "uploaded.png")
    app.run(debug=True, host="127.0.0.1", port=5000)
