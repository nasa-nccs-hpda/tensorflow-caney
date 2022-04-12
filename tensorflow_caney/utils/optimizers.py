import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa

from typing import Any, List

def get_optimizer(optimizer: str) -> Any:
    """
    Get optimizer function from string evaluation.
    """
    try:
        optimizer = eval(optimizer)
    except NameError as err:
        sys.exit(f'Handling run-time error: {err}')
    return optimizer
