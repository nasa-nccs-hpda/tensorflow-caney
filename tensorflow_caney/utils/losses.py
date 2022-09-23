import sys
import tensorflow as tf
import tensorflow_addons as tfa
import segmentation_models as sm

from typing import Any

__all__ = ["get_loss"]


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
    except (NameError, AttributeError) as err:
        sys.exit(f'{err}. Accepted loss from {tf}, {sm}, {tfa}.')
    return loss_function


def dicece_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(
        y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)
