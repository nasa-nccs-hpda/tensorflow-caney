import os
import random
from typing import List
from pathlib import Path
from glob import glob

import cupy as cp
import numpy as np
import xarray as xr
import pandas as pd


def read_dataset_csv(filename: str) -> pd.core.frame.DataFrame:
    """
    Read dataset CSV from disk and load for preprocessing.
    """
    assert os.path.exists(filename), f'File {filename} not found.'
    data_df = pd.read_csv(filename)
    assert not data_df.isnull().values.any(), f'NaN found: {filename}'
    return data_df


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


def modify_label_classes(mask: np.ndarray, expressions: List[dict]):
    """
    Change pixel label values based on expression.
    """
    if expressions is not None:
        for exp in expressions:
            [(k, v)] = exp.items()
            mask[eval(k, {k.split(' ')[0]: mask})] = v
    return mask
