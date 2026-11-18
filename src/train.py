"""Training loop with early stopping, LR scheduling and checkpointing."""
from __future__ import annotations

from pathlib import Path

from tensorflow import keras

from models import compile_model


def train_model(model, train_ds, val_ds, epochs: int, learning_rate: float,
                run_name: str, out_dir: str | Path = "experiments"):
    out_dir = Path(out_dir)
    log_dir = out_dir / "logs"
    model_dir = out_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    compile_model(model, learning_rate)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        keras.callbacks.CSVLogger(log_dir / f"{run_name}.csv"),
        keras.callbacks.ModelCheckpoint(
            model_dir / f"{run_name}.keras", monitor="val_accuracy",
            mode="max", save_best_only=True, verbose=0),
    ]
    print(f"\n=== Training {run_name} ({epochs} epochs, lr={learning_rate}) ===")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=1)
    return history, model
