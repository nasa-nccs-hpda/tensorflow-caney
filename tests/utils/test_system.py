import pytest
import numpy as np
from tensorflow_caney.utils import system


@pytest.mark.parametrize(
    "seed", [10, 1024]
)
def test_seed_everything(seed):
    system.seed_everything(seed)
    assert np.random.get_state()[1][0] == seed
