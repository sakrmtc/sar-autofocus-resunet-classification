from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import yaml

from data_preprocessing import load_images_from_glob, split_autofocus_data
from model_resunet import build_resunet
from utils import ensure_dir, plot_history, save_json


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_size = tuple(cfg["dataset"]["image_size"])
    inputs = load_images_from_glob(cfg["dataset"]["input_glob"], image_size=image_size)
    targets = load_images_from_glob(cfg["dataset"]["target_glob"], image_size=image_size)

    x_train, x_val, x_test, y_train, y_val, y_test = split_autofocus_data(
        inputs,
        targets,
        train_ratio=cfg["dataset"]["train_split"],
        val_ratio=cfg["dataset"]["val_split"],
        test_ratio=cfg["dataset"]["test_split"],
    )

    model = build_resunet(input_shape=(image_size[0], image_size[1], 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"]),
        loss="mse",
        metrics=["mse"],
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
    plot_history(history, str(Path(fig_dir) / "autofocus_training_history.png"), title="Autofocus Training")

    metrics = {
        "test_loss": float(model.evaluate(x_test, y_test, verbose=0)[0]),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
    }
    save_json(metrics, cfg["paths"]["metrics_output"])
    print("Autofocus training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    main(args.config)
