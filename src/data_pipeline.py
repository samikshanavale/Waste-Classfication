"""Dataset acquisition, stratified splitting and tf.data pipelines.

The raw images are the resized TrashNet dataset (MIT licence):
    https://github.com/garythung/trashnet
They are arranged into data/processed/{train,val,test}/<class>/*.jpg so that a
single tf.keras.utils.image_dataset_from_directory call works both locally and
in Google Colab.
"""
from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path

import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 160
BATCH_SIZE = 32
SEED = 42
TRASHNET_URL = (
    "https://raw.githubusercontent.com/garythung/trashnet/master/"
    "data/dataset-resized.zip"
)

# category -> (recommended bin, short user-facing advice)
BIN_MAP = {
    "cardboard": ("Dry / Recyclable bin", "Flatten the carton before recycling."),
    "glass": ("Glass / Recyclable bin", "Rinse the glass; keep it separate from ceramics."),
    "metal": ("Metal / Recyclable bin", "Empty and rinse the can before recycling."),
    "paper": ("Paper / Recyclable bin", "Keep paper dry and clean for recycling."),
    "plastic": ("Plastic / Dry recyclable bin", "Empty the bottle and replace the cap."),
    "trash": ("General waste / Residual bin", "Not a recyclable stream; send to residual waste."),
}


def ensure_dataset(raw_dir: str | Path) -> Path:
    """Download and unzip TrashNet if needed; return the image root folder."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "dataset-resized.zip"
    if not zip_path.exists():
        print(f"Downloading TrashNet from {TRASHNET_URL} ...")
        response = requests.get(TRASHNET_URL, timeout=180)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    image_root = raw_dir / "dataset-resized"
    if not (image_root / "cardboard").exists():
        print("Extracting dataset ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    return image_root


def prepare_splits(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    seed: int = SEED,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Path:
    """Create stratified 70/15/15 split folders and a manifest CSV."""
    image_root = ensure_dataset(raw_dir)
    processed_dir = Path(processed_dir)

    records: list[tuple[Path, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = image_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append((image_path, class_name))

    paths = [str(p) for p, _ in records]
    labels = [c for _, c in records]

    train_p, temp_p, train_y, temp_y = train_test_split(
        paths, labels, test_size=val_fraction + test_fraction,
        stratify=labels, random_state=seed,
    )
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p, temp_y, test_size=test_fraction / (val_fraction + test_fraction),
        stratify=temp_y, random_state=seed,
    )

    manifest: list[dict] = []
    for split, split_paths, split_labels in (
        ("train", train_p, train_y), ("val", val_p, val_y), ("test", test_p, test_y)
    ):
        for i, (src, label) in enumerate(zip(split_paths, split_labels)):
            dst_dir = processed_dir / split / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{i:04d}_{Path(src).name}"
            shutil.copy2(src, dst)
            manifest.append({"split": split, "label": label, "source": src, "target": str(dst)})

    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "label", "source", "target"])
        writer.writeheader()
        writer.writerows(manifest)

    print("Split summary:")
    for split in ("train", "val", "test"):
        counts = {c: sum(1 for m in manifest if m["split"] == split and m["label"] == c)
                  for c in CLASS_NAMES}
        print(f"  {split:5s}: {sum(counts.values()):4d} images  {counts}")
    return processed_dir


def make_datasets(
    processed_dir: str | Path = "data/processed",
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    """Return (train_ds, val_ds, test_ds) yielding images in [0, 255]."""
    root = Path(processed_dir)
    common = dict(
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root / "train", shuffle=True, seed=seed, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root / "val", shuffle=False, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        root / "test", shuffle=False, **common)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds


def prepare_all(raw_dir="data/raw", processed_dir="data/processed", img_size=IMG_SIZE,
                batch_size=BATCH_SIZE):
    """One-call helper used by the notebook, run_all.py and the local driver."""
    prepare_splits(raw_dir, processed_dir)
    return make_datasets(processed_dir, img_size, batch_size)
