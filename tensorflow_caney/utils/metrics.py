import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa

from typing import Any, List


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
        except NameError as err:
            sys.exit(f'{err}. Accepted optimizers from {tf}, {sm}, {tfa}')
    return metric_functions
