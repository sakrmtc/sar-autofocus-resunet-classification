from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split

from data_preprocessing import build_classifier_dataset
from model_resunet import build_classifier
from utils import ensure_dir, plot_history, save_json


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_size = tuple(cfg["dataset"]["image_size"])

    x, y = build_classifier_dataset(
        mountain_dir=cfg["dataset"]["mountain_dir"],
        sand_dir=cfg["dataset"]["sand_dir"],
        sea_dir=cfg["dataset"]["sea_dir"],
        image_size=image_size,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg["dataset"]["test_split"], random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=cfg["dataset"]["val_split_from_train"],
        random_state=42,
        stratify=y_train,
    )

    model = build_classifier(input_shape=(image_size[0], image_size[1], 3), num_classes=3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
            monitor="val_loss",
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        shuffle=cfg["training"]["shuffle"],
        callbacks=callbacks,
        verbose=1,
    )

    model_output = cfg["paths"]["model_output"]
    ensure_dir(Path(model_output).parent)
    model.save(model_output)

    fig_dir = cfg["paths"]["figure_output_dir"]
    ensure_dir(fig_dir)
    plot_history(history, str(Path(fig_dir) / "classifier_training_history.png"), title="Classifier Training")

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "classes": ["mountain", "sand", "sea"],
    }
    save_json(metrics, cfg["paths"]["metrics_output"])
    print("Classifier training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    main(args.config)
