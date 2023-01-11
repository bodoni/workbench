import logging
import os
from typing import List
from typing import Tuple

import functools
import tensorflow as tf
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments
def read(
    path: str,
    mode: str,
    split: float = 0.8,
    batch_size: int = 32,
    buffer_size: int = 256,
    random_state: int = 42,
    **options,
) -> tf.data.Dataset:
    paths = _list(path)
    logger.info("Found %d files in total.", len(paths))
    paths = train_test_split(paths, train_size=split, random_state=random_state)
    paths = paths[0 if mode == "training" else 1]
    logger.info("Using %d files for %s...", len(paths), mode)
    dataset = tf.data.Dataset.from_tensor_slices((paths,))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if mode == "training":
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=random_state)
    dataset = dataset.map(functools.partial(_decode, **options))
    dataset = dataset.batch(batch_size)
    for name, value in options.items():
        setattr(dataset, name, value)
    return dataset


def _decode(
    path: tf.Tensor,
    image_shape: Tuple[int, int, int],
    image_scale: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image.set_shape(image_shape)
    image = tf.cast(image, tf.float32)
    image = image / image_scale
    return image, image


def _list(path: str) -> List[str]:
    paths = []
    for (_, _, paths_) in os.walk(path):
        paths_ = (
            os.path.join(path, path_) for path_ in paths_ if path_.endswith(".png")
        )
        paths.extend(paths_)
    return sorted(paths)
