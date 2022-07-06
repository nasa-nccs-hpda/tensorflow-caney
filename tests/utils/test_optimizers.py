import pytest
import tensorflow as tf
from tensorflow_caney.utils.optimizers import get_optimizer

__all__ = ["tf", "test_get_optimizer", "test_get_metrics_exception"]


@pytest.mark.parametrize(
    "optimizer", ["tf.keras.optimizers.Adam", "tf.keras.optimizers.Adadelta"]
)
def test_get_optimizer(optimizer):
    callable_optimizer = get_optimizer(optimizer)
    assert callable_optimizer == eval(optimizer)


@pytest.mark.parametrize(
    "optimizer", ["tfc.my.unrealistic.Optimizer", "tf.keras.optimizers.FakeOp"]
)
def test_get_metrics_exception(optimizer):
    with pytest.raises(SystemExit):
        get_optimizer(optimizer)
