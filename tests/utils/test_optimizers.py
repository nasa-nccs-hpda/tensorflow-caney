import pytest
import logging
import tensorflow as tf
from tensorflow_caney.utils.optimizers import get_optimizer


@pytest.mark.parametrize(
    "optimizer", ["tf.keras.optimizers.Adam", "tf.keras.optimizers.Adadelta"]
)
def test_get_optimizer(optimizer):
    logging.info(f'TensorFlow version: {tf.__version__}')
    callable_optimizer = get_optimizer(optimizer)
    assert callable_optimizer == eval(optimizer)
