import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa

from typing import Any


class TverskyLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred, beta=0.7):
        numerator = tf.reduce_sum(y_true * y_pred)
        denominator = y_true * y_pred + beta * (1 - y_true) \
            * y_pred + (1 - beta) * y_true * (1 - y_pred)
        r = 1 - (numerator + 1) / (tf.reduce_sum(denominator) + 1)
        return tf.cast(r, tf.float32)


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
    """
    Get loss function from string evaluation.
    """
    try:
        loss_function = eval(loss)
    except NameError as err:
        sys.exit(f'Handling run-time error: {err}')
    return loss_function
