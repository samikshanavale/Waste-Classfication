"""Cross-dataset robustness evaluation: TrashNet-trained model on RealWaste photos.

RealWaste contains realistic illumination, contamination and overlapping materials,
whereas TrashNet images are clean white-background studio photos.  Evaluating on a
fixed subset of RealWaste measures the accuracy drop caused by this distribution shift.
Only the five shared material classes plus 'Miscellaneous Trash' -> 'trash' are used.
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from data_pipeline import CLASS_NAMES

HF_API = ("https://huggingface.co/api/datasets/shahzaibvohra/realwaste/tree/main"
          "?recursive=true&expand=false&limit=1000")
HF_FILE = "https://huggingface.co/datasets/shahzaibvohra/realwaste/resolve/main/"
REALWASTE_MAP = {
    "Cardboard": "cardboard", "Glass": "glass", "Metal": "metal",
    "Paper": "paper", "Plastic": "plastic", "Miscellaneous Trash": "trash",
}


def _list_files() -> list[str]:
    url, files = HF_API, []
    while url:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        files.extend(item["path"] for item in response.json()
                     if item["path"].lower().endswith((".jpg", ".jpeg", ".png")))
        link = response.headers.get("link", "")
        url = link.split(">")[0][1:] if 'rel="next"' in link else None
    return files


def download_subset(dest: str | Path, per_class: int = 40) -> Path:
    """Download a fixed, stratified subset of RealWaste (about 200 images)."""
    dest = Path(dest)
    if all((dest / c).exists() and len(list((dest / c).glob("*"))) >= per_class for c in CLASS_NAMES):
        return dest

    files = _list_files()
    chosen = {}
    for real_name, our_name in REALWASTE_MAP.items():
        candidates = sorted(f for f in files if f.startswith(real_name + "_"))
        chosen[our_name] = candidates[:per_class]

    for our_name, filenames in chosen.items():
        folder = dest / our_name
        folder.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            target = folder / Path(filename).name
            if target.exists():
                continue
            url = HF_FILE + urllib.parse.quote(filename)
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
        print(f"  RealWaste {our_name}: {len(list(folder.glob('*')))} images")
    return dest


def evaluate_robustness(model, folder: str | Path, img_size: int, batch_size: int,
                        temperature: float, out_figure: str | Path,
                        baseline_per_class: dict | None = None) -> dict:
    folder = Path(folder)
    dataset = tf.keras.utils.image_dataset_from_directory(
        folder, label_mode="categorical", class_names=CLASS_NAMES,
        image_size=(img_size, img_size), batch_size=batch_size, shuffle=False)

    probs, labels = [], []
    for images, y in dataset:
        probs.append(model.predict(images, verbose=0))
        labels.append(y.numpy())
    probs = np.concatenate(probs)
    y_true = np.concatenate(labels).argmax(axis=1)

    z = probs / temperature
    z = z - z.max(axis=1, keepdims=True)
    calibrated = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y_pred = calibrated.argmax(axis=1)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        per_class[name] = round(float((y_pred[mask] == i).mean()) if mask.any() else 0.0, 4)

    result = {
        "dataset": "RealWaste (Hugging Face mirror of UCI dataset 908)",
        "images": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "mean_confidence": round(float(calibrated.max(axis=1).mean()), 4),
        "per_class_accuracy": per_class,
    }

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(CLASS_NAMES))
    if baseline_per_class:
        in_domain = [baseline_per_class.get(c, {}).get("recall", 0) for c in CLASS_NAMES]
        ax.bar(x - 0.2, in_domain, 0.4, label="TrashNet test (in-distribution)")
    ax.bar(x + (0.2 if baseline_per_class else 0), [per_class[c] for c in CLASS_NAMES],
           0.4, label="RealWaste subset (shifted)")
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Per-class accuracy / recall")
    ax.set_title("Cross-dataset robustness: clean studio photos vs realistic landfill photos")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_figure, dpi=150)
    plt.close(fig)
    return result
