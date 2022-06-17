import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa

from typing import Any, List

def get_metrics(metrics: List[str]) -> Any:
    """
    Get metric functions from string evaluation.
    """
    metric_functions = []
    for metric in metrics:
        try:
            metric_functions.append(eval(metric))#()
        except NameError as err:
            sys.exit(f'Handling run-time error: {err}')
    return metric_functions