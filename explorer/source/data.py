from typing import Tuple

import tensorflow as tf


# pylint: disable=too-many-arguments
def read(
    path: str,
    mode: str,
    split: float = 0.2,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    image_scale: int = 225,
    random_seed: int = 42,
) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=path,
        labels=None,
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        validation_split=split,
        subset=mode,
        shuffle=True,
        seed=random_seed,
    )
    return dataset.prefetch(tf.data.AUTOTUNE).map(
        lambda image: (image / image_scale, image / image_scale)
    )
