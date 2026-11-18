"""Metrics, calibration, abstention analysis and evaluation plots."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from models import logit_model


# ----------------------------------------------------------------- predictions
def predict_arrays(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, integer labels) for a tf.data dataset."""
    probs, labels = [], []
    for images, y in dataset:
        probs.append(model.predict(images, verbose=0))
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels).argmax(axis=1)


def logits_array(model, dataset) -> np.ndarray:
    lm = logit_model(model)
    out = []
    for images, _ in dataset:
        out.append(lm.predict(images, verbose=0))
    return np.concatenate(out)


# ---------------------------------------------------------------------- metrics
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def model_complexity(model, model_path: str | Path | None = None) -> dict:
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    size_mb = round(Path(model_path).stat().st_size / 1e6, 2) if model_path and Path(model_path).exists() else None
    return {"params_total": trainable + non_trainable, "params_trainable": trainable,
            "size_mb": size_mb}


def measure_latency_ms(model, dataset, repeats: int = 8, batch_size: int = 32) -> float:
    images, _ = next(iter(dataset))
    images = images[:batch_size]
    for _ in range(2):  # warm-up
        model.predict(images, verbose=0)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(images, verbose=0)
    elapsed = time.perf_counter() - start
    return round(1000.0 * elapsed / (repeats * images.shape[0]), 3)


def per_class_report(y_true, y_pred, class_names) -> dict:
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return {name: {k: round(float(v), 4) for k, v in report[name].items()}
            for name in class_names}


# ------------------------------------------------------------------- calibration
def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Grid-search the temperature that minimises negative log-likelihood."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.50, 3.01, 0.05):
        nll = float(np.mean(-_log_softmax(logits / t)[np.arange(len(y_true)), y_true]))
        if nll < best_nll:
            best_t, best_nll = float(round(t, 2)), nll
    return best_t


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def abstention_curve(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """Accuracy / coverage as low-confidence predictions are rejected."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    rows = []
    for threshold in np.linspace(0.0, 0.95, 20):
        keep = confidence >= threshold
        rows.append({
            "threshold": round(float(threshold), 3),
            "coverage": round(float(keep.mean()), 4),
            "accuracy": round(float(correct[keep].mean()) if keep.any() else 0.0, 4),
            "risk": round(float(1 - correct[keep].mean()) if keep.any() else 1.0, 4),
        })
    # Choose the highest coverage that keeps accuracy above a target. The target is
    # relaxed adaptively so the mechanism also works for a weaker baseline model.
    chosen = None
    for target in (0.90, 0.85, 0.80, 0.75):
        valid = [r for r in rows if r["accuracy"] >= target and r["coverage"] >= 0.10]
        if valid:
            chosen = max(valid, key=lambda r: r["coverage"])
            break
    if chosen is None:
        chosen = max(rows, key=lambda r: r["coverage"] * r["accuracy"])
    return {"curve": rows, "chosen_threshold": chosen["threshold"],
            "chosen_coverage": chosen["coverage"], "chosen_accuracy": chosen["accuracy"]}


# ------------------------------------------------------------------------ plots
def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Confusion matrix — final model (test set)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in zip(axes, ["accuracy", "loss"],
                                 ["Training / validation accuracy", "Training / validation loss"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_calibration(probs_before, probs_after, y_true, out_path, bins=10):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for probs, label, marker in ((probs_before, "Before temperature scaling", "o"),
                                 (probs_after, "After temperature scaling", "s")):
        conf, pred = probs.max(axis=1), probs.argmax(axis=1)
        acc = (pred == y_true).astype(float)
        xs, ys = [], []
        for left in np.linspace(0, 1, bins + 1)[:-1]:
            mask = (conf > left) & (conf <= left + 1 / bins)
            if mask.any():
                xs.append(conf[mask].mean())
                ys.append(acc[mask].mean())
        ax.plot(xs, ys, marker, label=label)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability diagram (test set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_risk_coverage(abstention: dict, out_path):
    rows = abstention["curve"]
    coverages = [r["coverage"] for r in rows]
    risks = [r["risk"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(coverages, risks, "o-", markersize=3)
    ax.set_xlabel("Coverage (fraction of images answered)")
    ax.set_ylabel("Risk (error rate on answered images)")
    ax.set_title("Risk–coverage: abstaining on uncertain images")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
