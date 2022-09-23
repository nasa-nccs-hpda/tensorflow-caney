import pytest
import numpy as np
from tensorflow_caney.utils.data import get_mean_std_metadata

__all__ = [
    "test_get_mean_std_metadata",
    ""
]


@pytest.mark.parametrize(
    "filename, expected_mean, expected_std",
    [(
        'tests/data/mean-std-landcover-test.csv',
        np.array([0.14177309, 0.1371255, 0.1568816, 0.26403117]),
        np.array([0.015577243, 0.02131591, 0.041413646, 0.039731026])
    )]
)
def test_get_mean_std_metadata(filename, expected_mean, expected_std):
    mean, std = get_mean_std_metadata(filename)
    assert np.array_equal(mean, expected_mean) and \
        np.array_equal(std, expected_std)


# @pytest.mark.parametrize(
#    "mask, expressions, substract_labels",
#    [(
#        np.array(), [x - 1], True
#    )]
# )
# def test_modify_label_classes(mask, expressions, substract_labels):
