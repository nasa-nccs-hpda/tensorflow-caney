import sys
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import tensorflow_caney as tfc

from typing import Any
import keras.backend as K
from tensorflow.keras.losses import Loss, Reduction


__all__ = [
    "tfc", "get_loss", "dicece_loss", "cropped_loss",
    "CategoricalCrossEntropy", "CategoricalFocalLoss",
    "JaccardDistanceLoss", "TanimotoDistanceLoss"
]

epsilon = 1e-5
smooth = 1


def get_loss(loss: str) -> Any:
    """
    Get loss function from string evaluation.
    Args:
        loss (str): string with loss callable.
    Returns:
        Callable.
    """
    try:
        loss_function = eval(loss)
    except (NameError, AttributeError) as err:
        sys.exit(f'{err}. Accepted loss from {tf}, {sm}.')
    return loss_function


def dicece_loss(y_true: np.array, y_pred: np.array):
    """
    Dice Crossentropy loss function.
    Args:
        y_true (np.array): array value with truth logits
        y_pred (np.array): array value with prediction logits
    Returns:
       tf.tensor for loss value
    """
    def dice_loss(y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(
        y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)


def cropped_loss(loss_fn):
    """
    Wraps loss function. Crops the labels to match the logits size.
    """
    def _loss_fn(labels, logits):
        logits_shape = tf.shape(logits)
        labels_crop = tf.image.resize_with_crop_or_pad(
            labels, logits_shape[1], logits_shape[2])
        return loss_fn(labels_crop, logits)
    return _loss_fn


class CategoricalCrossEntropy(Loss):
    """
    Wrapper class for cross-entropy with class weights
    """
    def __init__(
                self,
                from_logits=True,
                class_weights=None,
                reduction=Reduction.AUTO,
                name='FocalLoss'
    ):
        """
        Categorical cross-entropy.
        :param from_logits: Whether predictions are logits or
        softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be applied
        to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'FocalLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.from_logits = from_logits
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.)

        # Calculate Cross Entropy
        loss = -y_true * tf.math.log(y_pred)

        # Multiply cross-entropy with class-wise weights
        if self.class_weights is not None:
            loss = tf.multiply(loss, self.class_weights)

        # Sum over classes
        loss = tf.reduce_sum(loss, axis=-1)

        return loss


class CategoricalFocalLoss(Loss):
    """
    Categorical version of focal loss.
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        Keras implementation: https://github.com/umbertogriffo/focal-loss-keras
    """

    def __init__(
                self,
                gamma=2.,
                alpha=.25,
                from_logits=True,
                class_weights=None,
                reduction=Reduction.AUTO,
                name='FocalLoss'
    ):
        """
        Categorical version of focal loss.
        :param gamma: gamma value, defaults to 2.
        :type gamma: float
        :param alpha: alpha value, defaults to .25
        :type alpha: float
        :param from_logits: Whether predictions are logits or
        softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be applied
        to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'FocalLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.class_weights = class_weights

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = self.alpha * tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

        # Multiply focal loss with class-wise weights
        if self.class_weights is not None:
            loss = tf.multiply(cross_entropy, self.class_weights)

        # Sum over classes
        loss = tf.reduce_sum(loss, axis=-1)

        return loss


class JaccardDistanceLoss(Loss):
    """
    Implementation of the Jaccard distance, or Intersection
    over Union IoU loss.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    Implementation taken from https://github.com/keras-team/keras-contrib.git
    """
    def __init__(
                self,
                smooth=1,
                from_logits=True,
                class_weights=None,
                reduction=Reduction.AUTO,
                name='JaccardLoss'
    ):
        """
        Jaccard distance loss.
        :param smooth: Smoothing factor. Default is 1.
        :type smooth: int
        :param from_logits: Whether predictions are logits
        or softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be
        applied to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'JaccardLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.smooth = smooth
        self.from_logits = from_logits

        self.class_weights = class_weights

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))

        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))

        jac = (intersection + self.smooth) / \
            (sum_ - intersection + self.smooth)

        loss = (1 - jac) * self.smooth

        if self.class_weights is not None:
            loss = tf.multiply(loss, self.class_weights)

        loss = tf.reduce_sum(loss, axis=-1)

        return loss


class TanimotoDistanceLoss(Loss):
    """
    Implementation of the Tanimoto distance, which is
    modified version of the Jaccard distance.
    Tanimoto = (|X & Y|)/ (|X|^2+ |Y|^2 - |X & Y|)
            = sum(|A*B|)/(sum(|A|^2)+sum(|B|^2)-sum(|A*B|))
    Implementation taken from
    https://github.com/feevos/resuneta
    """
    def __init__(
                self,
                smooth=1.0e-5,
                from_logits=True,
                class_weights=None,
                reduction=Reduction.AUTO,
                normalise=False,
                name='TanimotoLoss'
    ):
        """
        Tanimoto distance loss.
        :param smooth: Smoothing factor. Default is 1.0e-5.
        :type smooth: float
        :param from_logits: Whether predictions are logits
        or softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be
        applied to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: Reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param normalise: Whether to normalise loss by number
        of positive samples in class, defaults to `False`
        :type normalise: bool
        :param name: Name of the loss, defaults to 'TanimotoLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.smooth = smooth
        self.from_logits = from_logits
        self.normalise = normalise

        self.class_weights = class_weights

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        n_classes = y_true.shape[-1]

        volume = tf.reduce_mean(tf.reduce_sum(y_true, axis=(1, 2)), axis=0) \
            if self.normalise else tf.ones(n_classes, dtype=tf.float16)

        weights = tf.math.reciprocal(tf.math.square(volume))
        new_weights = tf.where(
            tf.math.is_inf(weights), tf.zeros_like(weights), weights)
        weights = tf.where(
            tf.math.is_inf(weights), tf.ones_like(weights) *
            tf.reduce_max(new_weights), weights
        )

        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))

        sum_ = tf.reduce_sum(y_true * y_true + y_pred * y_pred, axis=(1, 2))

        num_ = tf.multiply(intersection, weights) + self.smooth

        den_ = tf.multiply(sum_ - intersection, weights) + self.smooth

        tanimoto = num_ / den_

        loss = (1 - tanimoto)

        if self.class_weights is not None:
            loss = tf.multiply(loss, self.class_weights)

        loss = tf.reduce_sum(loss, axis=-1)

        return loss


@tf.function
def tversky(y_true, y_pred, alpha=0.40, beta=0.60):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """

    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    # weights
    y_weights = y_true[..., 1]
    y_weights = y_weights[..., np.newaxis]

    ones = 1
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t

    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)


def class_tversky(y_true, y_pred):
    smooth = 1

    y_true = K.permute_dimensions(y_true, (3, 1, 2, 0))
    y_pred = K.permute_dimensions(y_pred, (3, 1, 2, 0))

    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    alpha = 0.7
    return (true_pos + smooth) / \
        (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma = 0.75
    return K.sum(K.pow((1-pt_1), gamma))


def binary_tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / \
        (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def binary_tversky_loss(y_true, y_pred):
    return 1 - binary_tversky(y_true, y_pred)
