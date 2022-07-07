import pytest
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


# def load_model(
#            model_filename: str = None,
#            model_dir: str = None,
#            custom_objects: dict = {'iou_score': sm.metrics.iou_score},
#            model_extension: str = '*.hdf5'
#        ) -> Any:
