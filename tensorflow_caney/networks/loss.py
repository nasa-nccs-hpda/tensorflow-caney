import tensorflow as tf
from typing import Any


class TverskyLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred, beta=0.7):
        numerator = tf.reduce_sum(y_true * y_pred)
        denominator = y_true * y_pred + beta * (1 - y_true) \
            * y_pred + (1 - beta) * y_true * (1 - y_pred)
        r = 1 - (numerator + 1) / (tf.reduce_sum(denominator) + 1)
        return tf.cast(r, tf.float32)
        # y_pred = tf.convert_to_tensor_v2(y_pred)
        # y_true = tf.cast(y_true, y_pred.dtype)
        # return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)


def tversky_loss(y_true, y_pred, beta=0.7) -> Any:
    """
    Tversky index (TI) is a generalization of Dice’s coefficient.
    TI adds a weight to FP (false positives) and FN (false negatives).
    """
    def tversky_loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred)
        denominator = y_true * y_pred + beta * (1 - y_true) * \
            y_pred + (1 - beta) * y_true * (1 - y_pred)
        r = 1 - (numerator + 1) / (tf.reduce_sum(denominator) + 1)
        return tf.cast(r, tf.float32)
    return tf.numpy_function(tversky_loss, [y_true, y_pred], tf.float32)


def get_loss(loss: str) -> Any:
    # TODO: make this a dictionary switch like
    # option menu, improve menu with other losses available around
    if loss == 'tversky':
        return TverskyLoss()
    elif loss == 'categorical_crossentropy':
        return 'categorical_crossentropy'
    elif loss == 'binary_crossentropy':
        return 'binary_crossentropy'
