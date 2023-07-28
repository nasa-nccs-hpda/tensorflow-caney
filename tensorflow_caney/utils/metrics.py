import sys
import logging
import numpy as np
import tensorflow as tf
from typing import Any, List
import segmentation_models as sm
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"
__all__ = ["get_metrics"]

# ---------------------------------------------------------------------------
# module metrics
#
# General functions to compute custom metrics.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


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


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def iou_val(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def acc_val(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def prec_val(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro'), \
        precision_score(y_true, y_pred, average=None)


def recall_val(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro'), \
        recall_score(y_true, y_pred, average=None)


def compute_imf_weights(ground_truth, n_classes=None,
                        ignored_classes=[]
                        ) -> np.array:
    """
    Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.
    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    :param ground_truth: the annotations array
    :param nclasses: number of classes (defaults to max(ground_truth))
    :param ignored_classes: id of classes to ignore (optional)
    :return: numpy array with the IMF coefficients
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights


def getOccurrences(labels=[], fname='occurences.csv', nclasses=7):
    """
    Return pixel occurences per class.
    :param labels: numpy array with labels in int format
    :param fname: filename to save output
    :param nclasses: number of classes to look for
    :return: CSV file with class and occurrence per class
    """
    f = open(fname, "w+")
    f.write("class,occurence\n")
    for classes in range(nclasses):
        occ = np.count_nonzero(labels == classes)
        f.write(f'{classes},{occ}\n')
    f.close()


# -------------------------------------------------------------------------------
# module metrics Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Add unit tests here
