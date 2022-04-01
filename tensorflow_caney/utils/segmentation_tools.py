import logging
from sys import stdout
import numpy as np
from typing import Any
import tensorflow as tf
from sklearn.model_selection import train_test_split


AUTOTUNE = tf.data.experimental.AUTOTUNE


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

        # Set data filenames
        self.data_filenames = data_filenames
        self.label_filenames = label_filenames

        # Disable AutoShard, data lives in memory, use in memory options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF

        # Get total and validation size
        total_size = len(data_filenames)
        val_size = round(self.conf.test_size * total_size)
        logging.info(f'Train: {total_size - val_size}, Val: {val_size}')

        # Split training and validation dataset
        self.train_x, self.val_x = train_test_split(
            data_filenames, test_size=val_size, random_state=self.conf.seed)
        self.train_y, self.val_y = train_test_split(
            label_filenames, test_size=val_size, random_state=self.conf.seed)

        # Calculate training steps
        self.train_steps = len(self.train_x) // self.conf.batch_size
        self.val_steps = len(self.val_x) // self.conf.batch_size

        if len(self.train_x) % self.conf.batch_size != 0:
            self.train_steps += 1
        if len(self.val_x) % self.conf.batch_size != 0:
            self.val_steps += 1

        # Read mean and std metrics
        if train_step and self.conf.standardize:
            logging.info('Loading mean and std values.')
        #    self.conf.mean = np.load(
        #        os.path.join(
        #            self.conf.data_dir,
        #            f'mean-{self.conf.experiment_name}.npy')).tolist()
        #    self.conf.std = np.load(
        #        os.path.join(
        #            self.conf.data_dir,
        #            f'std-{self.conf.experiment_name}.npy')).tolist()

        # Initialize training dataset
        self.train_dataset = self.tf_dataset(
            self.train_x, self.train_y,
            read_func=self.tf_data_loader,
            repeat=True, batch_size=conf.batch_size
        )
        self.train_dataset = self.train_dataset.with_options(options)

        # Initialize validation dataset
        self.val_dataset = self.tf_dataset(
            self.val_x, self.val_y,
            read_func=self.tf_data_loader,
            repeat=True, batch_size=conf.batch_size
        )
        self.val_dataset = self.val_dataset.with_options(options)

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
            self.conf.tile_size, self.conf.tile_size, len(self.conf.output_bands)])
        y.set_shape([
            self.conf.tile_size, self.conf.tile_size, self.conf.n_classes])
        return x, y


    def load_data(self, x, y):
        """
        Load data on training loop.
        """
        # Read data
        x = np.load(x)
        y = np.load(y)

        # Standardize
        #if self.conf.standardize:
        #    if np.random.random_sample() > 0.75:
        #        for i in range(x.shape[-1]):  # for each channel in the image
        #            x[:, :, i] = (x[:, :, i] - self.conf.mean[i]) / \
        #                (self.conf.std[i] + 1e-8)
        #    else:
        #        for i in range(x.shape[-1]):  # for each channel in the image
        #            x[:, :, i] = (x[:, :, i] - np.mean(x[:, :, i])) / \
        #                (np.std(x[:, :, i]) + 1e-8)

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

    def get_preprocessing_metadata(self):
        """
        Get preprocessing metadat, mean and std values of d
        """
        preprocess_dataset = self.tf_dataset(
            self.data_filenames, self.label_filenames,
            read_func=self.tf_data_loader,
            repeat=True, batch_size=conf.batch_size
        )
        self.train_dataset = self.train_dataset.with_options(options)

        return mean, std
