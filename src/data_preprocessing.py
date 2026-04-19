from __future__ import annotations

import glob
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from utils import normalize_image


def load_images_from_glob(path_pattern: str, image_size: Tuple[int, int] | None = None) -> np.ndarray:
    images = []
    files = sorted(glob.glob(path_pattern))
    for file_path in files:
        image = cv2.imread(file_path)
        if image is None:
            continue
        if image_size is not None:
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        images.append(normalize_image(image))
    if not images:
        raise FileNotFoundError(f"No images found for pattern: {path_pattern}")
    return np.asarray(images, dtype=np.float32)


def load_images_from_folder(folder: str, image_size: Tuple[int, int]) -> np.ndarray:
    path_pattern = str(Path(folder) / "*.jpg")
    return load_images_from_glob(path_pattern, image_size=image_size)


def split_autofocus_data(inputs: np.ndarray, targets: np.ndarray,
                         train_ratio: float = 0.70,
                         val_ratio: float = 0.20,
                         test_ratio: float = 0.10):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    x_train, x_temp, y_train, y_temp = train_test_split(
        inputs, targets, test_size=(1 - train_ratio), random_state=42, shuffle=True
    )
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(1 - val_fraction_of_temp), random_state=42, shuffle=True
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_classifier_dataset(mountain_dir: str, sand_dir: str, sea_dir: str,
                             image_size: Tuple[int, int]):
    mountain = load_images_from_folder(mountain_dir, image_size)
    sand = load_images_from_folder(sand_dir, image_size)
    sea = load_images_from_folder(sea_dir, image_size)

    x = np.concatenate([mountain, sand, sea], axis=0)
    y = np.concatenate([
        np.zeros(len(mountain), dtype=np.int32),
        np.ones(len(sand), dtype=np.int32),
        np.full(len(sea), 2, dtype=np.int32),
    ], axis=0)
    return x, y
