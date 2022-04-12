import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa

from typing import Any, List

def get_callbacks(callbacks: List[str]) -> Any:
    """
    Get callback functions from string evaluation.
    """
    callback_functions = []
    for callback in callbacks:
        try:
            callback_functions.append(eval(callback))
        except NameError as err:
            sys.exit(f'Handling run-time error: {err}')
    return callback_functions

