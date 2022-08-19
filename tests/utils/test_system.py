import pytest
import numpy as np
import tensorflow as tf
from tensorflow_caney.utils import system
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

__all__ = [
    "test_seed_everything", "test_set_gpu_strategy",
    "test_set_mixed_precision", "test_set_xla"
]


@pytest.mark.parametrize(
    "seed", [10, 1024]
)
def test_seed_everything(seed):
    system.seed_everything(seed)
    assert np.random.get_state()[1][0] == seed


@pytest.mark.parametrize(
    "gpu_devices", ["0,1,2,3", "0,1"]
)
def test_set_gpu_strategy(gpu_devices):
    if len(tf.config.list_physical_devices('GPU')) > 0:
        gpu_strategy = system.set_gpu_strategy(gpu_devices)
        assert gpu_strategy.__class__ == MirroredStrategy().__class__
    else:
        with pytest.raises(AssertionError):
            system.set_gpu_strategy(gpu_devices)


@pytest.mark.parametrize(
    "mixed_precision", [True, False]
)
def test_set_mixed_precision(mixed_precision):
    mixed_precision = system.set_mixed_precision(mixed_precision)
    assert mixed_precision is None


@pytest.mark.parametrize(
    "xla", [True, False]
)
def test_set_xla(xla):
    xla = system.set_xla(xla)
    assert xla is None
