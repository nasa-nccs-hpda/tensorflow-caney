import os
import re
import csv
import glob
import warnings
import itertools
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.colors as pltc
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def confusion_matrix_func(
            y_true: list = [],
            y_pred: list = [],
            nclasses: int = 3,
            norm: bool = True,
            sample_points: bool = True,
            percent: float = 0.2,
        ):
    """
    Args:
    y_true:   2D numpy array with ground truth
    y_pred:   2D numpy array with predictions (already processed)
    nclasses: number of classes
    Returns:
    numpy array with confusion matrix
    """
    print("HELLLLOOOO")

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    label_name = [0,1,2]
    label_dict = np.unique(y_true, return_counts=True)

    # get subset of pixel
    if sample_points:

        #ind3 = np.random.choice(np.where(y_true == 3)[0], round(label_dict[1][3] * percent))
        ind2 = np.random.choice(np.where(y_true == 2)[0], round(label_dict[1][2] * percent))
        ind1 = np.random.choice(np.where(y_true == 1)[0], round(label_dict[1][1] * percent))
        ind0 = np.random.choice(np.where(y_true == 0)[0], round(label_dict[1][0] * percent))

        numpix_class0 = round(label_dict[1][0] * percent)
        numpix_class1 = round(label_dict[1][1] * percent)
        numpix_class2 = round(label_dict[1][2] * percent)
        #numpix_class3 = round(label_dict[1][3] * percent)
        #print(numpix_class0)

        y_true = np.concatenate((y_true[ind0], y_true[ind1], y_true[ind2]))
        y_pred = np.concatenate((y_pred[ind0], y_pred[ind1], y_pred[ind2]))

        # change value 3 in prediction (burned area) to value 0 (other vegetation)
        y_pred[y_pred == 3] = 0

    #print('ground truth: ', np.unique(gt))
    #print('predict: ', np.unique(pred))

    print("TRUUUUE", y_true.min(), y_true.max())
    print("PREDDD", y_pred.min(), y_pred.max())


    # get overall weighted accuracy
    try:
        accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    except:
        accuracy = np.nan
    
    try:
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
    except:
        balanced_accuracy = np.nan
    print("Accuracy: ", accuracy, "Balanced Accuracy: ", balanced_accuracy)

    # print(classification_report(y_true, y_pred))
    # if len(label_name) < 4:
    #     target_names = ['other-vegetation','tree','cropland']
    # else:
    #     target_names = ['other-vegetation','tree','cropland','burned']

    target_names = ['other-vegetation', 'tree', 'cropland']
    report = classification_report(
        y_true, y_pred, target_names=target_names,
        output_dict=True, labels=label_name)
    cfn_matrix = confusion_matrix(y_true, y_pred)

    #tree_recall = report['tree']['recall']
    #crop_recall = report['cropland']['recall']

    #tree_precision = report['tree']['precision']
    #crop_precision = report['cropland']['precision']

    ## get confusion matrix
    con_mat = tf.math.confusion_matrix(
        labels=y_true, predictions=y_pred, num_classes=nclasses
    ).numpy()

    # print(con_mat.sum(axis=1)[:, np.newaxis])
    # print(con_mat.sum(axis=1)[:, np.newaxis][0])
    # weights = [con_mat.sum(axis=1)[:, np.newaxis][0][0]/(5000*5000),con_mat.sum(axis=1)[:, np.newaxis][1][0]/(5000*5000),
    # con_mat.sum(axis=1)[:, np.newaxis][2][0]/(5000*5000),con_mat.sum(axis=1)[:, np.newaxis][3][0]/(5000*5000)]

    # print(weights)
    # get overall weighted accuracy
    # accuracy = accuracy_score(y_true, y_pred, normalize=False, sample_weight=weights)
    # print(con_mat)

    if norm:
        con_mat = np.around(
            con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis],
            decimals=2
        )

    # print(con_mat.sum(axis=1)[:, np.newaxis])
    where_are_NaNs = np.isnan(con_mat)
    con_mat[where_are_NaNs] = 0
    return report, cfn_matrix, accuracy, balanced_accuracy # tree_recall, crop_recall, tree_precision, crop_precision, cfn_matrix


def plot_confusion_matrix(cm, label_name, model, class_names=['a', 'b', 'c']):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names: list with classes for confusion matrix
    Return: confusion matrix figure.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Use white text if squares are dark; otherwise black.
    threshold = 0.55  # cm.max() / 2.
    # print(cm.shape[0], cm.shape[1]) #, threshold[0])

    print(label_name[:-27])

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.savefig(f'/home/geoint/tri/nasa_senegal/confusion_matrix/cas-tl-wcas/{model}-{label_name[:-27]}_subset_cfn_matrix_class.png')
    # plt.show()
    plt.close()
