from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def residual_block(x, filters: int, kernel_size: int = 3, stride: int = 1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def encoder_block(x, filters: int):
    x = residual_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(x, skip, filters: int):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = residual_block(x, filters)
    return x


def build_resunet(input_shape=(256, 256, 3)) -> Model:
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = residual_block(p4, 512)

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = layers.Conv2D(3, (1, 1), activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="ResUNetAutofocus")
    return model


def build_classifier(input_shape=(128, 128, 3), num_classes: int = 3) -> Model:
    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),

            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="TerrainClassifier",
    )
    return model
