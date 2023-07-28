import os
import sys
import logging
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa
import tensorflow_caney as tfc
import tensorflow_caney

from typing import Any
from glob import glob
from omegaconf import OmegaConf

from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics

__all__ = ["get_model", "load_model"]

CUSTOM_OBJECTS = {
    'iou_score': sm.metrics.iou_score,
    'focal_tversky_loss': tfc.utils.losses.focal_tversky_loss,
    'binary_tversky_loss': tfc.utils.losses.binary_tversky_loss,
}


def get_model(model: str) -> Any:
    """
    Get model function from string evaluation.
    Args:
        model (str): string with model callable.
    Returns:
        Callable
    """
    try:
        model_function = eval(model)
    except (NameError, AttributeError) as err:
        sys.exit(f'{err}. Accepted models from {tf}, {sm}, {tfa}, {tfc}')
    return model_function


def load_model(
            model_filename: str = None,
            model_dir: str = None,
            custom_objects: dict = CUSTOM_OBJECTS,
            model_extension: str = '*.hdf5',
            conf: OmegaConf = None
        ) -> Any:
    """
    Load model from filename, take the latest model if not given.
    Args:
        model_filename (str): string with model filename.
        model_dir (str): string with model directory.
        custom_objects (dict): dictionary with callable custom objects
        model_extension (str): string with model extension
    Returns:
        Callable
    """
    # Get the latest model from the directory if not given
    if model_filename is None:
        models_list = glob(os.path.join(model_dir, model_extension))
        model_filename = max(models_list, key=os.path.getctime)

    # Assert the existance of the model
    assert os.path.isfile(model_filename), \
        f'{model_filename} does not exist.'
    logging.info(f'Loading {model_filename}')

    try:
        # Load the model via the TensorFlow
        model = tf.keras.models.load_model(
            model_filename, custom_objects=custom_objects)
    except TypeError:
        # Load the model via the TensorFlow
        model = tf.keras.models.load_model(
            model_filename, custom_objects=custom_objects, compile=False)
        # Compile model
        model.compile(
            loss=get_loss(conf.loss),
            optimizer=get_optimizer(
                conf.optimizer)(conf.learning_rate),
            metrics=get_metrics(conf.metrics)
        )
    return model
