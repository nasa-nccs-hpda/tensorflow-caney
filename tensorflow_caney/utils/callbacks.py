import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa
from typing import List

__all__ = ["get_callbacks"]


def get_callbacks(callbacks: List[str]) -> List:
    """
    Get callback functions from string evaluation.
    Args:
        callbacks (List[str]): string with optimizer callable.
    Returns:
        List of Callables.
    """
    callback_functions = []
    for callback in callbacks:
        try:
            callback_functions.append(eval(callback))
        except (NameError, AttributeError) as err:
            sys.exit(f'{err}. Accepted callbacks from {tf}, {sm}, {tfa}')
    return callback_functions
