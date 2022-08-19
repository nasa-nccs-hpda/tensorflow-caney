import pytest
import tensorflow as tf
import segmentation_models as sm
from tensorflow_caney.utils.losses import get_loss

__all__ = ["tf", "sm", "test_get_loss", "test_get_loss_exception"]


@pytest.mark.parametrize(
    "loss", [
        "tf.keras.losses.CategoricalCrossentropy()",
        "sm.losses.DiceLoss(smooth=1e-08)"
    ]
)
def test_get_loss(loss):
    callable_loss = get_loss(loss)
    assert callable_loss.__class__ == eval(loss).__class__


@pytest.mark.parametrize(
    "loss", [
        "tfc.my.unrealistic.Loss",
        "tf.keras.optimizers.FakeLoss"
    ]
)
def test_get_loss_exception(loss):
    with pytest.raises(SystemExit):
        get_loss(loss)
