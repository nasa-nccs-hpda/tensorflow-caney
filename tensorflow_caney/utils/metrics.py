import sys
import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa

from typing import Any, List

def get_metrics(metrics: List[str]) -> Any:
    """
    Get loss function from string evaluation.
    """
    metrics_functions = []
    for metric in metrics:
        try:
            metrics_functions.append(eval(metric))
        except NameError as err:
            sys.exit(f'Handling run-time error: {err}')
    return metrics_functions
