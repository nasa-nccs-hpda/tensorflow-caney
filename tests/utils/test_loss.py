import pytest
import tensorflow as tf
from tensorflow_caney.utils.losses import get_loss

__all__ = ["tf", "test_get_loss", "test_get_loss_exception"]


@pytest.mark.parametrize(
    "loss", [
        "tf.keras.losses.CategoricalCrossentropy()",
        "sm.losses.DiceLoss(smooth=1e-08)"
    ]
)
def test_get_loss(loss):
    callable_loss = get_loss(loss)
    assert callable_loss == eval(loss)


@pytest.mark.parametrize(
    "loss", [
        "tfc.my.unrealistic.Loss",
        "tf.keras.optimizers.FakeLoss"
    ]
)
def test_get_loss_exception(loss):
    with pytest.raises(SystemExit):
        get_loss(loss)
