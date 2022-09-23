import json
import os
import time
import random
import logging
from glob import glob
from typing import List
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import rioxarray as rxr
import tensorflow as tf
from numba import jit

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def gen_random_tiles(
            image: np.ndarray,
            label: np.ndarray,
            index_id: int,
            num_classes: int,
            tile_size: int = 128,
            expand_dims: bool = True,
            max_patches: int = None,
            include: bool = False,
            augment: bool = True,
            output_filename: str = 'image',
            out_image_dir: str = 'image',
            out_label_dir: str = 'label',
            no_data: int = -10001,
            nodata_fractional: bool = False,
            nodata_fractional_tolerance: float = 0.75,
            json_tiles_dir: str = None,
            dataset_from_json: bool = False
        ) -> None:

    # verify the existance of json files to load dataset from
    if json_tiles_dir is not None and dataset_from_json:

        json_files = sorted(
            glob(os.path.join(json_tiles_dir, '*.json')),
            key=os.path.getmtime
        )
        logging.info(f'Found {len(json_files)} json dataset files.')

        if len(json_files) > 0:
            return gen_random_tiles_from_json(
                json_files,
                image,
                label,
                index_id,
                num_classes,
                tile_size,
                augment,
                expand_dims,
                out_image_dir,
                out_label_dir
            )

    generated_tiles = 0  # counter for generated tiles
    metadata = dict()
    while generated_tiles < max_patches:

        # Generate random integers from image
        y = random.randint(0, image.shape[0] - tile_size)
        x = random.randint(0, image.shape[1] - tile_size)

        # Generate img and mask patches
        image_tile = image[y:(y + tile_size), x:(x + tile_size)]
        label_tile = label[y:(y + tile_size), x:(x + tile_size)]

        # first condition, time must have valid classes
        if label_tile.min() < 0 or label_tile.max() > num_classes:
            continue

        # if image[x: (x + tile_size), y: (y + tile_size)].min() < -100:
        # second condition, if want zero nodata values, skip
        if cp.any(image_tile == no_data) and not nodata_fractional:
            continue

        # third condition, if include, number of labels must be at least 2
        if include and cp.unique(label_tile).shape[0] < 2:
            continue

        # ---
        # fourth condition, If given a tolerance for nodata,
        # check amount against tolerance
        # ---
        nodata_in_tile = cp.count_nonzero(image_tile == no_data)
        nodata_frac = (nodata_in_tile / image_tile.size)
        if nodata_frac >= nodata_fractional_tolerance and nodata_fractional:
            continue

        # Add to the tiles counter
        generated_tiles += 1
        filename = f'{Path(output_filename).stem}_{generated_tiles}.npy'
        metadata[filename] = {'x': x, 'y': y}

        # Apply some random transformations
        if augment:

            if cp.random.random_sample() > 0.5:
                metadata[filename]['fliplr'] = True
                image_tile = cp.fliplr(image_tile)
                label_tile = cp.fliplr(label_tile)

            if cp.random.random_sample() > 0.5:
                metadata[filename]['flipud'] = True
                image_tile = cp.flipud(image_tile)
                label_tile = cp.flipud(label_tile)

            if cp.random.random_sample() > 0.5:
                metadata[filename]['rot90'] = True
                image_tile = cp.rot90(image_tile, 1)
                label_tile = cp.rot90(label_tile, 1)

            if cp.random.random_sample() > 0.5:
                metadata[filename]['rot180'] = True
                image_tile = cp.rot90(image_tile, 2)
                label_tile = cp.rot90(label_tile, 2)

            if cp.random.random_sample() > 0.5:
                metadata[filename]['rot270'] = True
                image_tile = cp.rot90(image_tile, 3)
                label_tile = cp.rot90(label_tile, 3)

        if num_classes >= 2:
            label_tile = cp.eye(num_classes, dtype='uint8')[label_tile]
        else:
            if expand_dims:
                label_tile = cp.expand_dims(label_tile, axis=-1)

        # save tiles to disk
        cp.save(os.path.join(out_image_dir, filename), image_tile)
        cp.save(os.path.join(out_label_dir, filename), label_tile)

    # set json name to store values of random tiles for reproducibility
    if json_tiles_dir is not None:

        # create output directory
        os.makedirs(json_tiles_dir, exist_ok=True)

        # set output filename
        json_name = os.path.join(
            json_tiles_dir,
            f'{Path(output_filename).stem}_dataset_metadata.json')

        # store dict output into json file
        with open(json_name, 'w') as metadata_outfile:
            json.dump(metadata, metadata_outfile)

    return


def gen_random_tiles_from_json(
            json_metadata: list,
            image: np.ndarray,
            label: np.ndarray,
            index_id: int,
            num_classes: int,
            tile_size: int = 128,
            augment: bool = True,
            expand_dims: bool = True,
            out_image_dir: str = 'image',
            out_label_dir: str = 'label',
        ):
    """
    Precursor of gen_random_tiles, where a json file is accepted.
    """
    json_filename = json_metadata[index_id]
    # data_basename = Path(output_filename).stem
    # json_filename = [i for i in json_metadata if data_basename in i][0]

    # load json filename
    with open(json_filename, 'r') as j:
        tiles_metadata = json.loads(j.read())

    for tile_filename in tiles_metadata:

        # Tile indices
        x = tiles_metadata[tile_filename]['x']
        y = tiles_metadata[tile_filename]['y']

        # Generate img and mask patches
        image_tile = image[y:(y + tile_size), x:(x + tile_size)]
        label_tile = label[y:(y + tile_size), x:(x + tile_size)]

        # Apply some random transformations
        if augment:

            if 'fliplr' in tiles_metadata[tile_filename]:
                image_tile = cp.fliplr(image_tile)
                label_tile = cp.fliplr(label_tile)

            if 'flipud' in tiles_metadata[tile_filename]:
                image_tile = cp.flipud(image_tile)
                label_tile = cp.flipud(label_tile)

            if 'rot90' in tiles_metadata[tile_filename]:
                image_tile = cp.rot90(image_tile, 1)
                label_tile = cp.rot90(label_tile, 1)

            if 'rot180' in tiles_metadata[tile_filename]:
                image_tile = cp.rot90(image_tile, 2)
                label_tile = cp.rot90(label_tile, 2)

            if 'rot270' in tiles_metadata[tile_filename]:
                image_tile = cp.rot90(image_tile, 3)
                label_tile = cp.rot90(label_tile, 3)

        if num_classes >= 2:
            label_tile = cp.eye(num_classes, dtype='uint8')[label_tile]
        else:
            if expand_dims:
                label_tile = cp.expand_dims(label_tile, axis=-1)

        # save tiles to disk
        cp.save(os.path.join(out_image_dir, tile_filename), image_tile)
        cp.save(os.path.join(out_label_dir, tile_filename), label_tile)
    return


def gen_augmented_tiles(
            image: np.ndarray,
            label: np.ndarray,
            index_id: int,
            num_classes: int,
            tile_size: int = 128,
            expand_dims: bool = True,
            max_patches: int = None,
            include: bool = False,
            output_filename: str = 'image',
            out_image_dir: str = 'image',
            out_label_dir: str = 'label',
            no_data: int = -10001,
            nodata_fractional: bool = False,
            nodata_fractional_tolerance: float = 0.75,
            json_tiles_dir: str = None,
            dataset_from_json: bool = False
        ) -> None:
    """
    Function to generate all combinations of available augmentations
    without generating an exact duplicate. The need for this was to
    be able to add more specific data points to a set from a single
    """

    # verify the existance of json files to load dataset from
    if json_tiles_dir is not None and dataset_from_json:

        json_files = sorted(
            glob(os.path.join(json_tiles_dir, '*.json')),
            key=os.path.getmtime
        )
        logging.info(f'Found {len(json_files)} json dataset files.')

        if len(json_files) > 0:
            return gen_random_tiles_from_json(
                json_files,
                image,
                label,
                index_id,
                num_classes,
                tile_size,
                True,
                expand_dims,
                out_image_dir,
                out_label_dir
            )

    generated_tiles = 0  # counter for generated tiles
    metadata = dict()

    # ---
    # You could put these in integer form.
    # Keeping in bin for sanity check.
    # ---
    augmentation_dict = {
        'fliplr': 0b00001,
        'flipud': 0b00010,
        'rot090': 0b00100,
        'rot180': 0b01000,
        'rot270': 0b10000,
    }

    augmentation_list = list(range(2**len(augmentation_dict.keys())))

    while generated_tiles < max_patches:

        # Expects exact tile size
        y = 0
        x = 0

        # Generate img and mask patches
        image_tile = image[y:(y + tile_size), x:(x + tile_size)]
        label_tile = label[y:(y + tile_size), x:(x + tile_size)]

        # first condition, time must have valid classes
        if label_tile.min() < 0 or label_tile.max() > num_classes:
            continue

        # if image[x: (x + tile_size), y: (y + tile_size)].min() < -100:
        # second condition, if want zero nodata values, skip
        if cp.any(image_tile == no_data) and not nodata_fractional:
            continue

        # third condition, if include, number of labels must be at least 2
        if include and cp.unique(label_tile).shape[0] < 2:
            continue

        # ---
        # fourth condition, If given a tolerance for nodata,
        # check amount against tolerance
        # ---
        nodata_in_tile = cp.count_nonzero(image_tile == no_data)
        nodata_frac = (nodata_in_tile / image_tile.size)
        if nodata_frac >= nodata_fractional_tolerance and nodata_fractional:
            continue

        # Add to the tiles counter
        generated_tiles += 1
        filename = f'{Path(output_filename).stem}_{generated_tiles}.npy'
        metadata[filename] = {'x': x, 'y': y}

        # Pop a random bitmask that represents a combination of augmentations.
        tile_augmentation_index = random.randint(0, len(augmentation_list)-1)
        tile_augmentation = augmentation_list.pop(tile_augmentation_index)

        if tile_augmentation & augmentation_dict['fliplr'] \
                == augmentation_dict['fliplr']:
            metadata[filename]['fliplr'] = True
            image_tile = cp.fliplr(image_tile)
            label_tile = cp.fliplr(label_tile)

        if tile_augmentation & augmentation_dict['flipud'] \
                == augmentation_dict['flipud']:
            metadata[filename]['flipud'] = True
            image_tile = cp.flipud(image_tile)
            label_tile = cp.flipud(label_tile)

        if tile_augmentation & augmentation_dict['rot090'] \
                == augmentation_dict['rot090']:
            metadata[filename]['rot90'] = True
            image_tile = cp.rot90(image_tile, 1)
            label_tile = cp.rot90(label_tile, 1)

        if tile_augmentation & augmentation_dict['rot180'] \
                == augmentation_dict['rot180']:
            metadata[filename]['rot180'] = True
            image_tile = cp.rot90(image_tile, 2)
            label_tile = cp.rot90(label_tile, 2)

        if tile_augmentation & augmentation_dict['rot270'] \
                == augmentation_dict['rot270']:
            metadata[filename]['rot270'] = True
            image_tile = cp.rot90(image_tile, 3)
            label_tile = cp.rot90(label_tile, 3)

        if num_classes >= 2:
            label_tile = cp.eye(num_classes, dtype='uint8')[label_tile]
        else:
            if expand_dims:
                label_tile = cp.expand_dims(label_tile, axis=-1)

        # save tiles to disk
        cp.save(os.path.join(out_image_dir, filename), image_tile)
        cp.save(os.path.join(out_label_dir, filename), label_tile)

    # set json name to store values of random tiles for reproducibility
    if json_tiles_dir is not None:

        # create output directory
        os.makedirs(json_tiles_dir, exist_ok=True)

        # set output filename
        json_name = os.path.join(
            json_tiles_dir,
            f'{Path(output_filename).stem}_dataset_metadata.json')

        # store dict output into json file
        with open(json_name, 'w') as metadata_outfile:
            json.dump(metadata, metadata_outfile)

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


def get_mean_std_metadata(filename):
    """
    Load mean and std from disk.
    Args:
        filename (str): csv filename path to load mean and std from
    Returns:
        np.array mean, np.array std
    """
    assert os.path.isfile(filename), \
        f'{filename} does not exist.'
    metadata = pd.read_csv(filename, header=None)
    logging.info('Loading mean and std values.')
    return metadata.loc[0].values, metadata.loc[1].values


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
    mask: np.ndarray,
    expressions: List[dict],
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
    array: np.ndarray,
    xmin: int = 0,
    xmax: int = 10000
):
    """
    Clip extremity of pixels in the array to given range.
    Args:
        array (np.ndarray): numpy array with image
        xmin (int): minimum value allowed
        xmax (max): maximum value allowed
    Returns:
        np.array mean, np.array std
    """
    return np.clip(array, xmin, xmax)


def modify_roi(
    img: np.ndarray,
    mask: np.ndarray,
    ymin: int,
    ymax: int,
    xmin: int,
    xmax: int
):
    """
    Crop ROI, from outside to inside based on pixel address.
    """
    return img[ymin:ymax, xmin:xmax], mask[ymin:ymax, xmin:xmax]


def normalize_image(image: np.ndarray, normalize: float):
    """
    Normalize image within parameter, simple scaling of values.
    Args:
        image (np.ndarray): array to normalize
        normalize (float): float value to normalize with
    Returns:
        normalized np.ndarray
    """
    if normalize:
        image = image / normalize
    return image


def rescale_image(image: np.ndarray, rescale_type: str = 'per-image'):
    """
    Rescale image [0, 1] per-image or per-channel.
    Args:
        image (np.ndarray): array to rescale
        rescale_type (str): rescaling strategy
    Returns:
        rescaled np.ndarray
    """
    image = image.astype(np.float32)
    if rescale_type == 'per-image':
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif rescale_type == 'per-channel':
        for i in range(image.shape[-1]):
            image[:, :, i] = (
                image[:, :, i] - np.min(image[:, :, i])) / \
                (np.max(image[:, :, i]) - np.min(image[:, :, i]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
    return image


def read_dataset_csv(filename: str) -> pd.core.frame.DataFrame:
    """
    Read dataset CSV from disk and load for preprocessing.
    Args:
        filename (str): string dataset csv.
    Returns:
        pandas dataframe
    """
    assert os.path.exists(filename), f'File {filename} not found.'
    data_df = pd.read_csv(filename)
    assert not data_df.isnull().values.any(), f'NaN found: {filename}'
    return data_df


def standardize_image(
    image,
    standardization_type: str,
    mean: list = None,
    std: list = None
):
    """
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    image = image.astype(np.float32)
    if standardization_type == 'local':
        for i in range(image.shape[-1]):
            image[:, :, i] = (image[:, :, i] - np.mean(image[:, :, i])) / \
                (np.std(image[:, :, i]) + 1e-8)
    elif standardization_type == 'global':
        for i in range(image.shape[-1]):
            image[:, :, i] = (image[:, :, i] - mean[i]) / (std[i] + 1e-8)
    elif standardization_type == 'mixed':
        raise NotImplementedError
    return image


def standardize_batch(
    image_batch,
    standardization_type: str,
    mean: list = None,
    std: list = None
):
    """
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    for item in range(image_batch.shape[0]):
        image_batch[item, :, :, :] = standardize_image(
            image_batch[item, :, :, :], standardization_type, mean, std)
    return image_batch


@jit(nopython=True)
def standardize_image_numba(
    image, standardization_type, mean: list = None,
    std: list = None
):
    """
    TODO: FIX STABILITY
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


def standardize_batch_numba(
    image_batch, standardization_type, mean: list = None,
    std: list = None
):
    """
    TODO: FIX STABILITY
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    if standardization_type == 'local':
        for item in range(image_batch.shape[0]):
            image_batch[item] = (
                image_batch[item] - np.mean(image_batch[item], axis=(0, 1))) \
                / (np.std(image_batch[item], axis=(0, 1)) + 1e-8)
    elif standardization_type == 'global':
        print("me")
    else:
        print("muh")
    return image_batch


if __name__ == "__main__":

    experiment = 4
    x = np.random.randint(10000, size=(256, 256, 256, 4))

    if experiment == 1:
        # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXEC TIME!
        start = time.time()
        standardize_image_numba(x, 'local')
        end = time.time()
        print("Elapsed (with compilation) = %s" % (end - start))

        # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
        start = time.time()
        standardize_image_numba(x, 'local')
        end = time.time()
        print("Elapsed (after compilation) = %s" % (end - start))

    elif experiment == 2:
        # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXEC TIME!
        start = time.time()
        for item in range(x.shape[0]):
            standardize_image_numba(x, 'local')
        end = time.time()
        print("Elapsed (with compilation) = %s" % (end - start))

        # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
        start = time.time()
        for item in range(x.shape[0]):
            standardize_image_numba(x, 'local')
        end = time.time()
        print("Elapsed (after compilation) = %s" % (end - start))

    elif experiment == 3:
        # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXEC TIME!
        start = time.time()
        standardize_batch_numba(x, 'local')
        end = time.time()
        print("Elapsed (with compilation) = %s" % (end - start))

        # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
        start = time.time()
        standardize_batch_numba(x, 'local')
        end = time.time()
        print("Elapsed (after compilation) = %s" % (end - start))

    if experiment == 4:

        # chunks={'band': 'auto', 'x': 'auto', 'y': 'auto'}
        filename = '/adapt/nobackup/projects/ilab/projects/srlite/' + \
            'input/TOA_v2/Senegal/5-toas/' + \
            'WV02_20101020_M1BS_1030010007BBFA00-toa.tif'
        start = time.time()
        x = rxr.open_rasterio(filename).load()
        x = x.values
        end = time.time()
        print("Elapsed (after compilation) = %s" % (end - start))
