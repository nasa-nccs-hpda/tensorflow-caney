import os
import re
import sys
import logging
import numpy as np
import rioxarray as rxr
import tensorflow as tf
from typing import Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow_caney.utils.data import get_mean_std_metadata, \
    standardize_image, read_metadata, normalize_meanstd
from tensorflow_caney.utils.augmentations import center_crop


AUTOTUNE = tf.data.experimental.AUTOTUNE

__all__ = ["SegmentationDataLoader"]


class SegmentationDataLoader(object):

    def __init__(
                self,
                data_filenames: list,
                label_filenames: list,
                conf,
                train_step: bool = True,
            ):

        # Set configuration variables
        self.conf = conf
        self.train_step = train_step

        # Filename with mean and std metadata
        self.metadata_output_filename = os.path.join(
            self.conf.model_dir, f'mean-std-{self.conf.experiment_name}.csv')
        self.mean = None
        self.std = None

        # Set data filenames
        self.data_filenames = data_filenames
        self.label_filenames = label_filenames

        # Disable AutoShard, data lives in memory, use in memory options
        self.options = tf.data.Options()
        self.options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF

        # Total size of the dataset
        total_size = len(data_filenames)

        # If this is not a training step (e.g preprocess, predict)
        if not train_step:

            # Initialize training dataset
            self.train_dataset = self.tf_dataset(
                self.data_filenames, self.label_filenames,
                read_func=self.tf_data_loader,
                repeat=False, batch_size=conf.batch_size
            )
            self.train_dataset = self.train_dataset.with_options(self.options)

        # Else, if this is a training step (e.g. train)
        else:

            # Get total and validation size
            val_size = round(self.conf.test_size * total_size)
            logging.info(f'Train: {total_size - val_size}, Val: {val_size}')

            # Split training and validation dataset
            self.train_x, self.val_x = train_test_split(
                data_filenames, test_size=val_size,
                random_state=self.conf.seed
            )
            self.train_y, self.val_y = train_test_split(
                label_filenames, test_size=val_size,
                random_state=self.conf.seed
            )

            # Calculate training steps
            self.train_steps = len(self.train_x) // self.conf.batch_size
            self.val_steps = len(self.val_x) // self.conf.batch_size

            if len(self.train_x) % self.conf.batch_size != 0:
                self.train_steps += 1
            if len(self.val_x) % self.conf.batch_size != 0:
                self.val_steps += 1

            # Initialize training dataset
            self.train_dataset = self.tf_dataset(
                self.train_x, self.train_y,
                read_func=self.tf_data_loader,
                repeat=True, batch_size=conf.batch_size
            )
            self.train_dataset = self.train_dataset.with_options(self.options)

            # Initialize validation dataset
            self.val_dataset = self.tf_dataset(
                self.val_x, self.val_y,
                read_func=self.tf_data_loader,
                repeat=True, batch_size=conf.batch_size
            )
            self.val_dataset = self.val_dataset.with_options(self.options)

        # Load mean and std metrics, only if training and fixed standardization
        if train_step and self.conf.standardization in ['global', 'mixed']:
            self.mean, self.std = get_mean_std_metadata(
                self.metadata_output_filename)

        # Read standardization metadata
        if conf.metadata_regex is not None:
            self.metadata = read_metadata(
                conf.metadata_regex, conf.input_bands, conf.output_bands)

        """
        # sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
        self.seq = iaa.Sequential([
            # Basic aug without changing any values
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 20% of all images
            #sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops

            # Gaussian blur and gamma contrast
            sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
            # sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),

            # Sharpen
            #sometimes(iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))),
            # AddToHueAndSaturation
            # sometimes(iaa.AddToHueAndSaturation((-50, 50), per_channel=True)),

            # sometimes(iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})),
            #sometimes(iaa.Affine(rotate=(-90, 90))),

            # iaa.CoarseDropout((0.03, 0.25), size_percent=(0.02, 0.05), per_channel=True)
            # sometimes(iaa.Multiply((0.75, 1.25), per_channel=True), 0.3),
            sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
            # iaa.Add(value=(-0.5,0.5),per_channel=True),
            #sometimes(iaa.PiecewiseAffine(0.05), 0.3),
            #sometimes(iaa.PerspectiveTransform(0.01), 0.1)
        ],
            random_order=True)
        """

    def tf_dataset(
                self, x: list, y: list,
                read_func: Any,
                repeat=True,
                batch_size=64
            ) -> Any:
        """
        Fetch tensorflow dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(2048)
        dataset = dataset.map(read_func, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        if repeat:
            dataset = dataset.repeat()
        return dataset

    def tf_data_loader(self, x, y):
        """
        Initialize TensorFlow dataloader.
        """
        def _loader(x, y):
            x, y = self.load_data(x.decode(), y.decode())
            return x.astype(np.float32), y.astype(np.float32)
        x, y = tf.numpy_function(_loader, [x, y], [tf.float32, tf.float32])
        x.set_shape([
            self.conf.tile_size,
            self.conf.tile_size,
            len(self.conf.output_bands)]
        )
        y.set_shape([
            self.conf.tile_size,
            self.conf.tile_size,
            self.conf.n_classes]
        )
        return x, y

    def load_data(self, x, y):
        """
        Load data on training loop.
        """
        extension = Path(x).suffix

        if self.conf.metadata_regex is not None:
            year_match = re.search(r'(\d{4})(\d{2})(\d{2})', x)
            timestamp = str(int(year_match.group(2)))

        # Read data
        if extension == '.npy':
            # TODO: make channel dim more dynamic
            # if 0 < 1 then channel last, etc.
            x = np.load(x)
            y = np.load(y)
        elif extension == '.tif':
            x = np.moveaxis(rxr.open_rasterio(x).data, 0, -1)
            y = np.moveaxis(rxr.open_rasterio(y).data, 0, -1)
        else:
            sys.exit(f'{extension} format not supported.')

        if len(y.shape) < 3:
            y = np.expand_dims(y, axis=-1)

        # Normalize labels, default is diving by 1.0
        # Normalization should be done at preprocessing step
        # x = normalize_image(x, self.conf.normalize)

        # seq_det = self.seq.to_deterministic()
        # x = seq_det.augment_image(x)
        # y would have two channels, i.e. annotations and weights.
        # We need to augment y for operations such as crop and transform
        # y = seq_det.augment_image(y)

        # Standardize
        if self.conf.metadata_regex is not None and \
                self.conf.standardization is not None:
            x = normalize_meanstd(
                x, self.metadata[timestamp], subtract='median'
            )

        elif self.conf.standardization is not None and \
                self.conf.standardization != 'None':
            x = standardize_image(
                x, self.conf.standardization, self.mean, self.std)

        # Rescale
        if self.conf.rescale is not None and \
                self.conf.rescale != 'None':
            # print("RESCALING")
            means = x.mean(axis=(0, 1))
            stds = x.std(axis=(0, 1))
            newMin = means - (2 * stds)
            newMax = means + (2 * stds)
            x = (x - newMin) / (newMax - newMin)

        # Crop
        if self.conf.center_crop:
            x = center_crop(x, (self.conf.tile_size, self.conf.tile_size))
            y = center_crop(y, (self.conf.tile_size, self.conf.tile_size))

       
        # Augment
        if self.conf.augment:

            if np.random.random_sample() > 0.5:
                x = np.fliplr(x)
                y = np.fliplr(y)
            if np.random.random_sample() > 0.5:
                x = np.flipud(x)
                y = np.flipud(y)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 1)
                y = np.rot90(y, 1)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 2)
                y = np.rot90(y, 2)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 3)
                y = np.rot90(y, 3)

        return x, y
