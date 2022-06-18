import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa
from typing import Any

__all__ = [
    "TverskyLoss", "tversky_loss", "focal_loss", "get_loss"
]


class TverskyLoss(tf.keras.losses.Loss):
    """
    TverskyLoss function.
    """
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


def focal_loss(alpha=0.25, gamma=2):
    """
    Focal Loss.
    """
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (
            tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) \
            * (weight_a + weight_b) + logits * weight_b

    def loss(y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        loss = focal_loss_with_logits(
            logits=logits, targets=y_true, alpha=alpha,
            gamma=gamma, y_pred=y_pred
        )
        return tf.reduce_mean(loss)
    return loss


def get_loss(loss: str) -> Any:
    """
    Get loss function from string evaluation.
    Args:
        loss (str): string with loss callable.
    Returns:
        Callable.
    """
    try:
        loss_function = eval(loss)
    except NameError as err:
        sys.exit(f'{err}. Accepted loss from {tf}, {sm}, {tfa}')
    return loss_function
