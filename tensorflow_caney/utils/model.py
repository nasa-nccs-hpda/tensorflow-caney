import os
import sys
import logging
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa

import tensorflow_caney as tfc

from typing import Any
from glob import glob

__all__ = ["get_model", "load_model"]


def get_model(model: str) -> Any:
    """
    Get model function from string evaluation.
    Args:
        model (str): string with model callable.
    Returns:
        Callable.
    """
    try:
        model_function = eval(model)
    except (NameError, AttributeError) as err:
        sys.exit(f'{err}. Accepted models from {tf}, {sm}, {tfa}, {tfc}')
    return model_function


def load_model(
            model_filename: str = None,
            model_dir: str = None,
            custom_objects: dict = {'iou_score': sm.metrics.iou_score},
            model_extension: str = '*.hdf5'
        ) -> Any:
    """
    Load model from filename, take the latest model if not given.
    Args:
        model_filename (str): string with model filename.
        model_dir (str): string with model directory.
        custom_objects (dict): dictionary with callable custom objects
        model_extension (str): string with model extension
    Returns:
        Callable.
    """
    # Get the latest model from the directory if not given
    if model_filename is None:
        models_list = glob(os.path.join(model_dir, model_extension))
        model_filename = max(models_list, key=os.path.getctime)

    # Assert the existance of the model
    assert os.path.isfile(model_filename), \
        f'{model_filename} does not exist.'
    logging.info(f'Loading {model_filename}')

    # Load the model via the TensorFlow
    model = tf.keras.models.load_model(
        model_filename, custom_objects=custom_objects)
    return model
