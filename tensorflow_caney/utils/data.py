from email.mime import image
import os
import random
from typing import List
from pathlib import Path
from glob import glob

import cupy as cp
import numpy as np
import xarray as xr
import pandas as pd

import tensorflow as tf


def gen_random_tiles(
            image: cp.ndarray, label: cp.ndarray, num_classes: int,
            tile_size: int = 128, expand_dims: bool = True,
            max_patches: int = None, include: bool = False,
            augment: bool = True, output_filename: str = 'image',
            out_image_dir: str = 'image',  out_label_dir: str = 'label'
        ) -> None:

    generated_tiles = 0  # counter for generated tiles
    while generated_tiles < max_patches:

        # Generate random integers from image
        x = random.randint(0, image.shape[0] - tile_size)
        y = random.randint(0, image.shape[1] - tile_size)

        # first condition, time must have valid classes
        if label[x: (x + tile_size), y: (y + tile_size)].min() < 0 or \
                label[x: (x + tile_size), y: (y + tile_size)].max() \
                > num_classes:
            continue

        if image[x: (x + tile_size), y: (y + tile_size)].min() < -100:
            continue

        # second condition, if include, number of labels must be at least 2
        if include and cp.unique(
                label[x: (x + tile_size), y: (y + tile_size)]).shape[0] < 2:
            continue

        # Add to the tiles counter
        generated_tiles += 1

        # Generate img and mask patches
        image_tile = image[x:(x + tile_size), y:(y + tile_size)]
        label_tile = label[x:(x + tile_size), y:(y + tile_size)]

        # Apply some random transformations
        if augment:

            if cp.random.random_sample() > 0.5:
                image_tile = cp.fliplr(image_tile)
                label_tile = cp.fliplr(label_tile)
            if cp.random.random_sample() > 0.5:
                image_tile = cp.flipud(image_tile)
                label_tile = cp.flipud(label_tile)
            if cp.random.random_sample() > 0.5:
                image_tile = cp.rot90(image_tile, 1)
                label_tile = cp.rot90(label_tile, 1)
            if cp.random.random_sample() > 0.5:
                image_tile = cp.rot90(image_tile, 2)
                label_tile = cp.rot90(label_tile, 2)
            if cp.random.random_sample() > 0.5:
                image_tile = cp.rot90(image_tile, 3)
                label_tile = cp.rot90(label_tile, 3)

        if num_classes >= 2:
            label_tile = cp.eye(num_classes, dtype='uint8')[label_tile]
        else:
            if expand_dims:
                label_tile = cp.expand_dims(label_tile, axis=-1)

        filename = f'{Path(output_filename).stem}_{generated_tiles}.npy'
        cp.save(os.path.join(out_image_dir, filename), image_tile)
        cp.save(os.path.join(out_label_dir, filename), label_tile)
    return


def get_dataset_filenames(data_dir: str, ext: str = '*.npy') -> list:
    """
    Get dataset filenames for training.
    """
    data_filenames = sorted(glob(os.path.join(data_dir, ext)))
    assert len(data_filenames) > 0, f'No files under {data_dir}.'
    return data_filenames


def get_mean_std_dataset(tf_dataset, output_filename: str = None):
    """
    Get general mean and std from tensorflow dataset.
    Useful when reading from disk and not from memory.
    """
    for data, _ in tf_dataset:
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        channels_sum += tf.reduce_mean(data, [0, 1, 2])
        channels_squared_sum += tf.reduce_mean(data**2, [0, 1, 2])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    if output_filename is not None:
        mean_std = np.stack([mean.numpy(), std.numpy()], axis=0)
        pd.DataFrame(mean_std).to_csv(output_filename, header=None, index=None)
    return mean, std


def get_mean_std_metadata(output_filename):
    metadata = pd.read_csv()
    #.tolist()
    return metadata

def modify_bands(
        xraster: xr.core.dataarray.DataArray, input_bands: List[str],
        output_bands: List[str], drop_bands: List[str] = []):
    """
    Drop multiple bands to existing rasterio object
    """
    # Do not modify if image has the same number of output bands
    if xraster['band'].shape[0] == len(output_bands):
        return xraster

    # Drop any bands from input that should not be on output
    for ind_id in list(set(input_bands) - set(output_bands)):
        drop_bands.append(input_bands.index(ind_id)+1)
    return xraster.drop(dim="band", labels=drop_bands, drop=True)


def modify_label_classes(
        mask: np.ndarray, expressions: List[dict],
        substract_labels: bool = False
    ):
    """
    Change pixel label values based on expression.
    """
    if substract_labels:
        mask = mask - 1
    if expressions is not None:
        for exp in expressions:
            [(k, v)] = exp.items()
            mask[eval(k, {k.split(' ')[0]: mask})] = v
    return mask


def modify_pixel_extremity(
        img: np.ndarray, xmin: int = 0, xmax: int = 10000):
    """
    Crop ROI, from outside to inside based on pixel address
    """
    return np.clip(img, xmin, xmax)


def modify_roi(
        img: np.ndarray, mask: np.ndarray,
        ymin: int, ymax: int, xmin: int, xmax: int):
    """
    Crop ROI, from outside to inside based on pixel address.
    """
    return img[ymin:ymax, xmin:xmax], mask[ymin:ymax, xmin:xmax]


# TODO: excellent question, how do you normalize when you have many
# vegetation indices to choose from????
def normalize_image(img: np.ndarray, normalize: float):
    """
    Normalize image within parameter, simple scaling of values.
    """
    if normalize:
        img = img / normalize
    return img


def read_dataset_csv(filename: str) -> pd.core.frame.DataFrame:
    """
    Read dataset CSV from disk and load for preprocessing.
    """
    assert os.path.exists(filename), f'File {filename} not found.'
    data_df = pd.read_csv(filename)
    assert not data_df.isnull().values.any(), f'NaN found: {filename}'
    return data_df


def standardize_image(image, standardization_type, mean: list = None, std: list = None):
    """
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    if standardization_type == 'local':
        for i in range(image.shape[-1]):  # for each channel in the image
            image[:, :, i] = (image[:, :, i] - np.mean(image[:, :, i])) / \
                (np.std(image[:, :, i]) + 1e-8)
    elif standardization_type == 'global':
        print("me")
    else:
        print("muh")
        #    if np.random.random_sample() > 0.75:
        #        for i in range(x.shape[-1]):  # for each channel in the image
        #            x[:, :, i] = (x[:, :, i] - self.conf.mean[i]) / \
        #                (self.conf.std[i] + 1e-8)
        #    else:
        #        for i in range(x.shape[-1]):  # for each channel in the image
        #            x[:, :, i] = (x[:, :, i] - np.mean(x[:, :, i])) / \
        #                (np.std(x[:, :, i]) + 1e-8)
    return image


def standardize_batch(image, standardization_type, mean: list = None, std: list = None):
    """
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    if standardization_type == 'local':
        print("me")
    elif standardization_type == 'global':
        print("me")
    else:
        print("muh")
    return image