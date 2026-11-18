"""Model definitions: a custom CNN baseline and MobileNetV2 transfer learning.

Both models end with a Dense layer named ``logits`` followed by a Softmax layer
named ``probs``.  Keeping the logits accessible is what makes temperature
scaling, calibration and uncertainty-based abstention possible.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def augmentation_layers() -> keras.Sequential:
    """Training-only augmentation: waste appears at many angles and light levels."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.15),
    ], name="augmentation")


def classification_head(x, num_classes: int, dropout: float):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    probs = layers.Softmax(name="probs")(logits)
    return probs


def build_custom_cnn(num_classes: int, img_size: int = 160, augmentation: bool = False,
                     dropout: float = 0.30) -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    if augmentation:
        x = augmentation_layers()(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    probs = classification_head(x, num_classes, dropout)
    return keras.Model(inputs, probs, name="custom_cnn_aug" if augmentation else "custom_cnn")


def build_mobilenetv2(num_classes: int, img_size: int = 160, augmentation: bool = False,
                      fine_tune: bool = False, dropout: float = 0.20,
                      weights: str = "imagenet") -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    # Equivalent to MobileNetV2 `preprocess_input` in "tf" mode: x / 127.5 - 1.
    # Using Rescaling instead of a Lambda keeps the saved model self-contained.
    x = layers.Rescaling(scale=2.0 / 255.0, offset=-1.0, name="mobilenet_preprocess")(inputs)
    if augmentation:
        x = augmentation_layers()(x)

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights)
    base.trainable = fine_tune
    if fine_tune:
        # Fine-tune only the final convolutional block with a small learning rate;
        # batch-norm layers stay frozen to keep training stable on a small dataset.
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers[-30:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    x = base(x, training=fine_tune)
    probs = classification_head(x, num_classes, dropout)
    name = "mobilenetv2_finetuned" if fine_tune else "mobilenetv2_frozen"
    if augmentation:
        name += "_aug"
    return keras.Model(inputs, probs, name=name)


def compile_model(model: keras.Model, learning_rate: float = 1e-4) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model


def logit_model(model: keras.Model) -> keras.Model:
    """Return a model that outputs raw logits (needed for calibration)."""
    return keras.Model(model.input, model.get_layer("logits").output)


def last_conv_layer(model: keras.Model):
    """Return the last convolutional layer object, including inside nested models.

    Transfer-learning backbones (MobileNetV2, EfficientNet, ...) appear as a single
    nested ``keras.Model`` at the top level, so a plain loop over ``model.layers``
    would not find their convolutional layers.
    """
    last = None
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)) and len(layer.output.shape) == 4:
            last = layer
        elif isinstance(layer, keras.Model):
            nested = last_conv_layer(layer)
            if nested is not None:
                last = nested
    return last


def last_conv_layer_name(model: keras.Model) -> str:
    """Name of the convolutional feature map used by Grad-CAM."""
    layer = last_conv_layer(model)
    if layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM")
    return layer.name
