import pytest
import tensorflow as tf
from tensorflow_caney.utils.callbacks import get_callbacks

__all__ = ["tf", "test_get_callbacks", "test_get_callbacks_exception"]


@pytest.mark.parametrize(
    "callbacks",
    [
        [
            "tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss')",
            "tf.keras.callbacks.TerminateOnNaN()"
        ],
        [
            "tf.keras.callbacks.EarlyStopping(monitor='val_loss')",
            "tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss')",
            "tf.keras.callbacks.TerminateOnNaN()"
        ]
    ],
)
def test_get_callbacks(callbacks):
    evaluated_callbacks = [eval(cb).__class__ for cb in callbacks]
    callable_callbacks = get_callbacks(callbacks)
    callable_callbacks = [cb.__class__ for cb in callable_callbacks]
    assert evaluated_callbacks == callable_callbacks


@pytest.mark.parametrize(
    "callbacks",
    [["tfc.my.unrealistic.callback()"]]
)
def test_get_callbacks_exception(callbacks):
    with pytest.raises(SystemExit):
        get_callbacks(callbacks)
