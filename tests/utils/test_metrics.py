import pytest
import logging
import tensorflow as tf
import segmentation_models as sm
from tensorflow_caney.utils.metrics import get_metrics


@pytest.mark.parametrize(
    "metrics",
    [
        [
            'tf.keras.metrics.BinaryAccuracy(threshold=0.5)',
            'tf.keras.metrics.Recall()',
            'tf.keras.metrics.Precision()',
            'sm.metrics.iou_score'
        ],
        [
            'tf.keras.metrics.CategoricalAccuracy()',
            'tf.keras.metrics.Recall()',
            'tf.keras.metrics.Precision()',
            'sm.metrics.iou_score'
        ]
    ]
)
def test_get_metrics(metrics):
    logging.info(f'TensorFlow version: {tf.__version__}, {sm.__version__}')
    evaluated_metrics = [eval(cb).__class__ for cb in metrics]
    callable_metrics = get_metrics(metrics)
    callable_metrics = [cb.__class__ for cb in callable_metrics]
    assert evaluated_metrics == callable_metrics
