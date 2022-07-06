import pytest
import numpy as np
from tensorflow_caney.utils import augmentations as tfc_aug


@pytest.mark.parametrize(
    "image, expected_shape",
    [(
        np.random.randint(5, size=(64, 64, 8)),
        (32, 32, 8)
    )]
)
def test_center_crop(image, expected_shape):
    image = tfc_aug.center_crop(image)
    assert image.shape == expected_shape
