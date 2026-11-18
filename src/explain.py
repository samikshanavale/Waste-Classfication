"""Grad-CAM explanations used for qualitative failure analysis and the demo."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from models import last_conv_layer


def _container_and_conv(model):
    """Return (container_model, conv_layer) for the last convolution.

    Transfer-learning backbones appear as a *nested* keras.Model. Keras 3 cannot
    build a functional graph directly from the outer input to an inner layer's
    output, so we locate the model that owns the conv layer and build the
    feature model from that container's input instead.
    """
    top = None
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) and len(layer.output.shape) == 4:
            top = layer
    if top is not None:
        return model, top
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            container, conv = _container_and_conv(layer)
            if conv is not None:
                return container, conv
    return model, None


def make_gradcam(model, image: np.ndarray, img_size: int = 160) -> np.ndarray:
    """Return a normalised Grad-CAM heatmap for the model's top prediction.

    ``image`` is a single float32 array of shape (H, W, 3) in [0, 255].
    Handles both the custom CNN (top-level convolutions) and nested
    transfer-learning backbones such as MobileNetV2.
    """
    container, conv_layer = _container_and_conv(model)
    if conv_layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM")
    feature_model = tf.keras.Model(
        container.input, [conv_layer.output, container.output])
    if container is model:
        pre_layers, post_layers = [], []
    else:
        index = model.layers.index(container)
        pre_layers = model.layers[:index]
        post_layers = model.layers[index + 1:]

    x = tf.constant(np.expand_dims(image.astype(np.float32), axis=0), dtype=tf.float32)
    for layer in pre_layers:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            x = layer(x, training=False)

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(x, training=False)
        h = features
        for layer in post_layers:
            h = layer(h, training=False)
        class_index = tf.argmax(h[0])
        loss = h[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    # Drop the batch axis: downstream code expects a single (H, W) heatmap.
    cam = tf.reduce_sum(conv_outputs * pooled[:, tf.newaxis, tf.newaxis, :], axis=-1)[0]
    cam = tf.maximum(cam, 0.0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., tf.newaxis], (img_size, img_size)).numpy()[..., 0]
    return cam


def heatmap_overlay(image: np.ndarray, cam: np.ndarray) -> Image.Image:
    heat = plt.cm.jet(cam)[:, :, :3]
    base = np.clip(image.astype(np.float32) / 255.0, 0, 1)
    overlay = 0.55 * base + 0.45 * heat
    return Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))


def build_explanation_grid(model, test_ds, class_names, out_path: str | Path,
                           img_size: int = 160, n_correct: int = 2, n_wrong: int = 2):
    """Save a 2x2 figure with correct and incorrect predictions + their heatmaps."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_correct, chosen_wrong = [], []

    for images, labels in test_ds:
        if len(chosen_correct) >= n_correct and len(chosen_wrong) >= n_wrong:
            break
        probs = model.predict(images, verbose=0)
        for i in range(images.shape[0]):
            y_true = int(labels[i].numpy().argmax())
            y_pred = int(probs[i].argmax())
            item = (images[i].numpy(), y_true, y_pred, float(probs[i].max()))
            if y_pred == y_true and len(chosen_correct) < n_correct:
                chosen_correct.append(item)
            elif y_pred != y_true and len(chosen_wrong) < n_wrong:
                chosen_wrong.append(item)

    examples = chosen_correct + chosen_wrong
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3.1 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, 0)
    for row, (image, y_true, y_pred, confidence) in enumerate(examples):
        cam = make_gradcam(model, image, img_size)
        axes[row, 0].imshow(image.astype(np.uint8))
        axes[row, 0].set_title(f"True: {class_names[y_true]}")
        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title("Grad-CAM heatmap")
        axes[row, 2].imshow(heatmap_overlay(image, cam))
        axes[row, 2].set_title(f"Pred: {class_names[y_pred]} ({confidence:.2f})")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Explainability: where the model looked (top rows correct, bottom rows errors)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return len(examples)
