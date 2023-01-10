from typing import Tuple

import tensorflow as tf


def train(
    data_training: tf.data.Dataset,
    data_validation: tf.data.Dataset,
    image_size: Tuple[int, int, int],
) -> tf.keras.Model:
    image = tf.keras.layers.Input(shape=image_size, name="image")
