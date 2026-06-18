from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_model(input_shape=(64, 64, 3)):
    """
    A very small CNN for demo purposes.

    IMPORTANT:
    This model is NOT medically valid. It's only here to make the app fully runnable.
    """
    import tensorflow as tf  # type: ignore

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _make_synthetic_data(n: int = 256, input_shape=(64, 64, 3), seed: int = 42):
    """
    Creates synthetic images + labels so training runs quickly with no dataset.
    """
    rng = np.random.default_rng(seed)
    x = rng.random((n, *input_shape), dtype=np.float32)  # 0..1
    y = rng.integers(0, 2, size=(n, 1), dtype=np.int32)
    return x, y


def _get_image_datagenerators():
    import tensorflow as tf  # type: ignore

    # Keras preprocessing lives in this module.
    return tf.keras.preprocessing.image.ImageDataGenerator


def _make_generators(
    data_dir: Path,
    img_size: int = 64,
    batch_size: int = 32,
    seed: int = 42,
):
    """
    Expects directory structure:
      data_dir/
        train/
          non_autistic/
          autistic/
        valid/  (or val/)
          non_autistic/
          autistic/
        test/   (optional for training)
    """
    ImageDataGenerator = _get_image_datagenerators()

    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    if not valid_dir.exists():
        valid_dir = data_dir / "val"

    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(
            "DATA_DIR must contain 'train/' and 'valid/' (or 'val/') folders with class subfolders: "
            "'non_autistic' and 'autistic'."
        )

    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode="nearest",
    )

    datagen_valid = ImageDataGenerator(rescale=1.0 / 255.0)

    class_order = ["non_autistic", "autistic"]  # ensures autistic => label 1

    train_gen = datagen_train.flow_from_directory(
        str(train_dir),
        target_size=(img_size, img_size),
        classes=class_order,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    valid_gen = datagen_valid.flow_from_directory(
        str(valid_dir),
        target_size=(img_size, img_size),
        classes=class_order,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    return train_gen, valid_gen, class_order


def train_and_save(
    output_path: Path | str = None,
    data_dir: Path | str | None = None,
    epochs: int = 3,
    img_size: int = 64,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> Path:
    """
    Trains a small CNN and saves to `output_path` (default: model/model.h5).

    If `data_dir` is provided, trains on that dataset directory.
    Otherwise falls back to synthetic data (demo only).
    """
    output_path = Path(output_path or Path(__file__).resolve().parent / "model.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_path = output_path.parent / "threshold.json"

    model = build_model(input_shape=(img_size, img_size, 3))

    if data_dir is not None:
        train_gen, valid_gen, class_order = _make_generators(
            data_dir=Path(data_dir),
            img_size=img_size,
            batch_size=batch_size,
        )
        model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=epochs,
            verbose=1,
        )
        # Store threshold + label order so inference uses the same threshold.
        threshold_payload = {
            "threshold": float(threshold),
            "class_order": class_order,
            "img_size": int(img_size),
        }
        threshold_path.write_text(json.dumps(threshold_payload), encoding="utf-8")
    else:
        # Synthetic fallback (demo only)
        x, y = _make_synthetic_data(n=256, input_shape=(img_size, img_size, 3))
        model.fit(x, y, epochs=2, batch_size=batch_size, validation_split=0.2, verbose=1)
        threshold_payload = {
            "threshold": float(threshold),
            "class_order": ["non_autistic", "autistic"],
            "img_size": int(img_size),
        }
        threshold_path.write_text(json.dumps(threshold_payload), encoding="utf-8")

    model.save(str(output_path))
    return output_path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None, help="Path to dataset root folder.")
    p.add_argument("--output_path", type=str, default=None, help="Where to save model.h5.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path = train_and_save(
        output_path=args.output_path,
        data_dir=args.data_dir,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    print(f"Saved model to: {path}")

