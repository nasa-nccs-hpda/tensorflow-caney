import pytest
import numpy as np
from tensorflow_caney.utils import system


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
    xla = system.set_mixed_precision(xla)
    assert xla is None
