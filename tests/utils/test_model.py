import pytest
import tensorflow as tf
from tensorflow_caney.utils import model as tfc_model

__all__ = ["test_get_model"]


@pytest.mark.parametrize(
    "model, expected_tile_shape, expected_channel_shape",
    [
        (
            "tfc.unet.unet_batchnorm(nclass=3, input_size=(256, 256, 9)," +
            "maps=[64, 128, 256, 512, 1024])", 256, 9
        ),
        (
            "sm.Unet('resnet34', input_shape=(512, 512, 3)," +
            "encoder_weights=None, classes=3, activation='softmax')", 512, 3
        )
    ]
)
def test_get_model(model, expected_tile_shape, expected_channel_shape):
    model = tfc_model.get_model(model)
    tile_size = model.layers[0].input_shape[0][1]
    tile_channels = model.layers[0].input_shape[0][-1]
    assert tile_size == expected_tile_shape and \
        tile_channels == expected_channel_shape


@pytest.mark.parametrize(
    "model", ["tfc.my.unrealistic.Model", "tf.keras.model.FakeOp"]
)
def test_get_model_exception(model):
    with pytest.raises(SystemExit):
        tfc_model.get_model(model)


@pytest.mark.parametrize(
    "model_filename, input_shape, expected_tile_shape, expected_channel_shape",
    [
        (
            "categorical-0.12.hdf5", (256, 256, 1), 512, 1
        )
    ]
)
def test_load_model(
            model_filename,
            input_shape,
            expected_tile_shape,
            expected_channel_shape
        ):

    # generate small model on the fly and save it
    inputs = tf.keras.layers.Input(input_shape)
    c1 = tf.keras.layers.Conv2D(
        4, (3, 3), activation='relu', padding='same')(inputs)
    model = tf.keras.models.Model(
        inputs=inputs, outputs=c1, name="UNetDropout")
    model.save(model_filename)

    # load model and test
    model = tfc_model.load_model(model_filename)
    tile_size = model.layers[0].input_shape[0][1]
    tile_channels = model.layers[0].input_shape[0][-1]
    assert tile_size == expected_tile_shape and \
        tile_channels == expected_channel_shape
