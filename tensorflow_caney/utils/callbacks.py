import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa
from typing import List


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
        except NameError as err:
            sys.exit(f'{err}. Accepted models from {tf}, {sm}, {tfa}')
    return callback_functions
