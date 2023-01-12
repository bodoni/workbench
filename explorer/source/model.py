import os
from typing import List

import aim
import aim.sdk.adapters.keras_mixins
import tensorflow as tf

# pylint: disable=too-few-public-methods
class Model(tf.keras.models.Model):
    def __init__(self, encoder: List[dict], decoder: List[dict]) -> "Self":
        super().__init__()
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(**options) for options in encoder],
            name="encoder",
        )
        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.Conv2DTranspose(**options) for options in decoder],
            name="decoder",
        )

    def call(self, data: tf.Tensor) -> tf.keras.models.Model:
        return self.decoder(self.encoder(data))


class AimCallback(
    aim.sdk.adapters.keras_mixins.TrackerKerasCallbackMetricsEpochEndMixin,
    tf.keras.callbacks.Callback,
):
    def __init__(self, run: aim.Run) -> "Self":
        # pylint: disable=bad-super-call
        super(tf.keras.callbacks.Callback, self).__init__()
        self._run = run


# pylint: disable=too-many-arguments
def train(
    data_training: tf.data.Dataset,
    data_validation: tf.data.Dataset,
    experiment: aim.Run,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    **options,
) -> tf.keras.Model:
    model = Model(**options)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    aim_callback = AimCallback(experiment)
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(".tensorboard", experiment.hash),
        histogram_freq=1,
    )
    data_training = data_training.batch(batch_size).map(lambda image: (image, image))
    data_validation = data_validation.batch(batch_size).map(
        lambda image: (image, image)
    )
    model.fit(
        data_training,
        validation_data=data_validation,
        epochs=epochs,
        callbacks=[
            aim_callback,
            earlystopping_callback,
            tensorboard_callback,
        ],
    )
    return model
