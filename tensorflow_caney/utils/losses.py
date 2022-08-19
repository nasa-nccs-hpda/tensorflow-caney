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
