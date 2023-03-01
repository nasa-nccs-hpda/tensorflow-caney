import os
import sys
import time
import logging
import argparse
import omegaconf
import rasterio
import numpy as np
import xarray as xr
import rioxarray as rxr

from glob import glob
from pathlib import Path
from omegaconf.listconfig import ListConfig

from tensorflow_caney.model.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything, \
    set_gpu_strategy, set_mixed_precision, set_xla
from tensorflow_caney.utils.data import read_dataset_csv, \
    gen_random_tiles, modify_bands, normalize_image, rescale_image, \
    modify_label_classes, get_dataset_filenames, get_mean_std_dataset, \
    get_mean_std_metadata
from tensorflow_caney.utils import indices
from tensorflow_caney.utils.regression_tools import RegressionDataLoader
from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.model import load_model, get_model
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics
from tensorflow_caney.utils.callbacks import get_callbacks
from tensorflow_caney.inference import regression_inference

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
xp = np

__status__ = "Development"


# -----------------------------------------------------------------------------
# class CNNRegression
# -----------------------------------------------------------------------------
class CNNRegression(object):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config_filename, data_csv=None, logger=None):

        # TODO:
        # slurm filename in output dir

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # Configuration file intialization
        self.conf = self._read_config(config_filename)

        # Set Data CSV
        self.data_csv = data_csv

        # Set output directories and locations
        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.logger.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        os.makedirs(self.labels_dir, exist_ok=True)
        self.logger.info(f'Images dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # _read_config
    # -------------------------------------------------------------------------
    def _read_config(self, filename: str, config_class=Config):
        """
        Read configuration filename and initiate objects
        """
        # Configuration file initialization
        schema = omegaconf.OmegaConf.structured(config_class)
        conf = omegaconf.OmegaConf.load(filename)
        try:
            conf = omegaconf.OmegaConf.merge(schema, conf)
        except BaseException as err:
            sys.exit(f"ERROR: {err}")
        return conf

    def _set_logger(self):
        """
        Set logger configuration.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def get_filenames(self, data_regex: str) -> list:
        """
        Get filename from list of regexes
        """
        # get the paths/filenames of the regex
        filenames = []
        if isinstance(data_regex, list) or isinstance(data_regex, ListConfig):
            for regex in data_regex:
                filenames.extend(glob(regex))
        else:
            filenames = glob(data_regex)
        assert len(filenames) > 0, f'No files under {data_regex}'
        return sorted(filenames)

    # -------------------------------------------------------------------------
    # preprocess
    # -------------------------------------------------------------------------
    def preprocess(self) -> None:
        """
        Perform general preprocessing.
        """
        logging.info('Starting preprocessing stage')

        # Initialize dataframe with data details
        data_df = read_dataset_csv(self.data_csv)
        self.logger.info(data_df)

        # iterate over each file and generate dataset
        for index_id, (data_filename, label_filename, n_tiles) \
                in enumerate(data_df.values):

            logging.info(f'Processing {Path(data_filename).stem}')

            # Read imagery from disk and process both image and mask
            image = rxr.open_rasterio(data_filename, chunks=CHUNKS).load()
            label = rxr.open_rasterio(label_filename, chunks=CHUNKS).values
            logging.info(f'Image: {image.shape}, Label: {label.shape}')

            # Calculate indices and append to the original raster
            image = indices.add_indices(
                xraster=image, input_bands=self.conf.input_bands,
                output_bands=self.conf.output_bands)
            logging.info(f'Image: {image.shape}, Label: {label.shape}')

            # Lower the number of bands if required
            image = modify_bands(
                xraster=image, input_bands=self.conf.input_bands,
                output_bands=self.conf.output_bands)
            logging.info(f'Image: {image.shape}, Label: {label.shape}')

            image = image.values

            # Move from chw to hwc, squeze mask if required
            image = xp.moveaxis(image, 0, -1)
            label = xp.squeeze(label) if len(label.shape) != 2 else label
            logging.info(f'Image: {image.shape}, Label: {label.shape}')
            logging.info(f'Label classes min {label.min()}, max {label.max()}')

            # Normalize values within [0, 1] range
            image = normalize_image(image, self.conf.normalize)

            # Rescale values within [0, 1] range
            image = rescale_image(image, self.conf.rescale)

            # Modify labels, sometimes we need to merge some training classes
            label = modify_label_classes(
                label, self.conf.modify_labels, self.conf.substract_labels)
            logging.info(f'Label classes min {label.min()}, max {label.max()}')

            # generate random tiles
            gen_random_tiles(
                image=image,
                label=label,
                expand_dims=self.conf.expand_dims,
                tile_size=self.conf.tile_size,
                index_id=index_id,
                num_classes=self.conf.n_classes,
                max_patches=n_tiles,
                include=self.conf.include_classes,
                augment=self.conf.augment,
                output_filename=data_filename,
                out_image_dir=self.images_dir,
                out_label_dir=self.labels_dir,
                json_tiles_dir=self.conf.json_tiles_dir,
                dataset_from_json=self.conf.dataset_from_json,
                xp=xp
            )

        # Calculate mean and std values for training
        data_filenames = get_dataset_filenames(self.images_dir)
        label_filenames = get_dataset_filenames(self.labels_dir)
        logging.info(f'Mean and std values from {len(data_filenames)} files.')

        # Temporarily disable standardization and augmentation
        current_standardization = self.conf.standardization
        self.conf.standardization = None
        metadata_output_filename = os.path.join(
            self.conf.model_dir, f'mean-std-{self.conf.experiment_name}.csv')
        os.makedirs(self.conf.model_dir, exist_ok=True)
        logging.info(f'Standardization metrics: {metadata_output_filename}')

        # Set main data loader
        main_data_loader = RegressionDataLoader(
            data_filenames, label_filenames, self.conf, False
        )

        # Get mean and std array
        mean, std = get_mean_std_dataset(
            main_data_loader.train_dataset, metadata_output_filename)
        logging.info(f'Mean: {mean.numpy()}, Std: {std.numpy()}')

        # Re-enable standardization for next pipeline step
        self.conf.standardization = current_standardization

        return

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    def train(self) -> None:

        self.logger.info('Starting training stage')

        # Set hardware acceleration options
        gpu_strategy = set_gpu_strategy(self.conf.gpu_devices)
        set_mixed_precision(self.conf.mixed_precision)
        set_xla(self.conf.xla)

        # Get data and label filenames for training
        data_filenames = get_dataset_filenames(self.images_dir)
        label_filenames = get_dataset_filenames(self.labels_dir)
        assert len(data_filenames) == len(label_filenames), \
            'Number of data and label filenames do not match'

        logging.info(
            f'Data: {len(data_filenames)}, Label: {len(label_filenames)}')

        # Set main data loader
        main_data_loader = RegressionDataLoader(
            data_filenames, label_filenames, self.conf
        )

        # Set multi-GPU training strategy
        with gpu_strategy.scope():

            if self.conf.transfer_learning == 'fine-tuning':
                model = get_model(self.conf.model)
                model.trainable = False
                model_2 = load_model(
                    model_filename=self.conf.model_filename,
                    model_dir=self.model_dir
                )

                model.set_weights(model_2.get_weights())
                model.trainable = True
                model.compile(
                    loss=get_loss(self.conf.loss),
                    optimizer=get_optimizer(
                        self.conf.optimizer)(self.conf.learning_rate),
                    metrics=get_metrics(self.conf.metrics)
                )
            else:
                # Get and compile the model
                model = get_model(self.conf.model)
                model.compile(
                    loss=get_loss(self.conf.loss),
                    optimizer=get_optimizer(
                        self.conf.optimizer)(self.conf.learning_rate),
                    metrics=get_metrics(self.conf.metrics)
                )

        model.summary()

        # Fit the model and start training
        model.fit(
            main_data_loader.train_dataset,
            validation_data=main_data_loader.val_dataset,
            epochs=self.conf.max_epochs,
            steps_per_epoch=main_data_loader.train_steps,
            validation_steps=main_data_loader.val_steps,
            callbacks=get_callbacks(self.conf.callbacks)
        )
        logging.info(f'Done with training, models saved: {self.model_dir}')

        return

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    def predict(self) -> None:

        logging.info('Starting prediction stage')

        # Load model for inference
        model = load_model(
            model_filename=self.conf.model_filename,
            model_dir=self.model_dir
        )

        # Retrieve mean and std, there should be a more ideal place
        if self.conf.standardization in ["global", "mixed"]:
            mean, std = get_mean_std_metadata(
                os.path.join(
                    self.model_dir,
                    f'mean-std-{self.conf.experiment_name}.csv'
                )
            )
            logging.info(f'Mean: {mean}, Std: {std}')
        else:
            mean = None
            std = None

        # Gather filenames to predict
        if len(self.conf.inference_regex_list) > 0:
            data_filenames = self.get_filenames(self.conf.inference_regex_list)
        else:
            data_filenames = self.get_filenames(self.conf.inference_regex)
        logging.info(f'{len(data_filenames)} files to predict')

        # iterate files, create lock file to avoid predicting the same file
        for filename in sorted(data_filenames):

            # start timer
            start_time = time.time()

            # set output directory
            basename = os.path.basename(os.path.dirname(filename))
            if basename == 'M1BS' or basename == 'P1BS':
                basename = os.path.basename(
                    os.path.dirname(os.path.dirname(filename)))

            output_directory = os.path.join(
                self.conf.inference_save_dir, basename)
            os.makedirs(output_directory, exist_ok=True)

            # set prediction output filename
            output_filename = os.path.join(
                output_directory,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif')

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # predict only if file does not exist and no lock file
            if not os.path.isfile(output_filename) and \
                    not os.path.isfile(lock_filename):

                try:

                    logging.info(f'Starting to predict {filename}')

                    # create lock file
                    open(lock_filename, 'w').close()

                    # open filename
                    image = rxr.open_rasterio(filename)
                    logging.info(f'Prediction shape: {image.shape}')

                except rasterio.errors.RasterioIOError:
                    logging.info(f'Skipped {filename}, probably corrupted.')
                    continue

                # Calculate indices and append to the original raster
                image = indices.add_indices(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)

                # Modify the bands to match inference details
                image = modify_bands(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)
                logging.info(f'Prediction shape after modf: {image.shape}')

                # Transpose the image for channel last format
                image = image.transpose("y", "x", "band")

                # Remove no-data values to account for edge effects
                temporary_tif = xr.where(image > -100, image, 600)

                # Sliding window prediction
                prediction = regression_inference.sliding_window_tiler(
                    xraster=temporary_tif,
                    model=model,
                    n_classes=self.conf.n_classes,
                    overlap=self.conf.inference_overlap,
                    batch_size=self.conf.pred_batch_size,
                    standardization=self.conf.standardization,
                    mean=mean,
                    std=std,
                    normalize=self.conf.normalize,
                    window=self.conf.window_algorithm
                ) * self.conf.normalize_label
                prediction[prediction < 0] = 0

                # Drop image band to allow for a merge of mask
                image = image.drop(
                    dim="band",
                    labels=image.coords["band"].values[1:],
                )

                # Get metadata to save raster
                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=self.conf.experiment_type,
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs
                )

                # Add metadata to raster attributes
                prediction.attrs['long_name'] = (self.conf.experiment_type)
                prediction.attrs['model_name'] = (self.conf.model_filename)
                prediction = prediction.transpose("band", "y", "x")

                # Set nodata values on mask
                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(
                    self.conf.prediction_nodata, encoded=True, inplace=True)

                # Save output raster file to disk
                prediction.rio.to_raster(
                    output_filename,
                    BIGTIFF="IF_SAFER",
                    compress=self.conf.prediction_compress,
                    driver=self.conf.prediction_driver,
                    dtype=self.conf.prediction_dtype
                )
                del prediction

                # delete lock file
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")

            # This is the case where the prediction was already saved
            else:
                logging.info(f'{output_filename} already predicted.')
        return


# -----------------------------------------------------------------------------
# main
#
# python cnn_regression.py -c config.yaml -d config.csv -s preprocess train
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to perform CNN regression.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=False,
                        default=None,
                        dest='data_csv',
                        help='Path to the data CSV configuration file')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['preprocess', 'train', 'predict'],
                        choices=['preprocess', 'train', 'predict'])

    args = parser.parse_args()

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    timer = time.time()

    # Initialize pipeline object
    pipeline = CNNRegression(args.config_file, args.data_csv)

    # Regression CHM pipeline steps
    if "preprocess" in args.pipeline_step:
        pipeline.preprocess()
    if "train" in args.pipeline_step:
        pipeline.train()
    if "predict" in args.pipeline_step:
        pipeline.predict()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # /explore/nobackup/people/jacaraba/development/tensorflow-caney/tests/data/regression_test.yaml
    # /explore/nobackup/people/jacaraba/development/tensorflow-caney/tests/data/regression_test.csv
    sys.exit(main())
