import os
import sys
import logging
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa
import tensorflow_caney

from typing import Any
from glob import glob


def get_model(model: str) -> Any:
    """
    Get model function from string evaluation.
    """
    try:
        model_function = eval(model)
    except NameError as err:
        sys.exit(f'Handling run-time error: {err}')
    return model_function


def load_model(
        model_filename: str = None,
        model_dir: str = None,
        custom_objects: dict = {'iou_score': sm.metrics.iou_score},
        model_extension: str = '*.hdf5'
    ) -> Any:
    """
    Load model from filename, take the latest model if not given.
    """
    # Get the latest model from the directory if not given
    if model_filename is None:
        models_list = glob(os.path.join(model_dir, model_extension))
        model_filename = max(models_list, key=os.path.getctime)

    assert os.path.isfile(model_filename), \
        f'{model_filename} does not exist.'
    logging.info(f'Loading {model_filename}')

    # Load the model via the TensorFlow
    model = tf.keras.models.load_model(
        model_filename, custom_objects=custom_objects)
    return model
