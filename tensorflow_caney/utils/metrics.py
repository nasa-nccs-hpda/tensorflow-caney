import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa

from typing import Any, List

__all__ = ["get_metrics"]


def get_metrics(metrics: List[str]) -> Any:
    """
    Get metric functions from string evaluation.
    Args:
        model (str): string with metric callable.
    Returns:
        Callable.
    """
    metric_functions = []
    for metric in metrics:
        try:
            metric_functions.append(eval(metric))
        except (NameError, AttributeError) as err:
            sys.exit(f'{err}. Accepted metrics from {tf}, {sm}, {tfa}')
    return metric_functions
