# -*- coding: utf-8 -*-

import sys
from typing import List, Optional
from omegaconf import OmegaConf
from dataclasses import dataclass, field


@dataclass
class Config:

    # dataset_csv: str
    data_dir: str

    experiment_name: str = 'unet-cnn'
    experiment_type: str = 'landcover'

    seed: int = 24
    gpu_devices: str = '0,1,2,3'
    mixed_precision: bool = True
    xla: bool = True

    input_bands: List[str] = field(
        default_factory=lambda:
            ['Blue', 'Green', 'Red', 'NIR1', 'HOM1', 'HOM2'])
    output_bands: List[str] = field(
        default_factory=lambda: ['Blue', 'Green', 'Red', 'NIR1'])

    modify_labels: Optional[List[str]] = None

    expand_dims: bool = True
    tile_size: int = 256
    include_classes: bool = False
    augment: bool = True
    standardize: bool = True
    batch_size: int = 32
    n_classes: int = 1
    test_size: float = 0.20

    mean: List[float] = field(
        default_factory=lambda: [])
    std: List[float] = field(
        default_factory=lambda: [])

    loss: str = 'tversky'
    learning_rate: float = 0.0001
    max_epochs: int = 6000
    patience: int = 7

    model_filename: str = 'model.h5'
    inference_regex: str = '*.tif'
    inference_save_dir: str = 'results'
    window_size: int = 8120
    inference_overlap: int = 2
    inference_treshold: float = 0.5
    pred_batch_size: int = 128

    # set some strict hyperparameter attributes
    # self.seed = getattr(self, 'seed', 34)
    # self.batch_size = getattr(self, 'batch_size', 16) * \
    #    getattr(self, 'strategy.num_replicas_in_sync', 1)

    # set some data parameters manually
    # self.data_min = getattr(self, 'data_min', 0)
    # self.data_max = getattr(self, 'data_max', 10000)
    # self.tile_size = getattr(self, 'tile_size', 256)
    # self.chunks = {'band': 1, 'x': 2048, 'y': 2048}

    # set some data parameters manually
    # self.modify_labels = getattr(self, 'modify_labels', None)
    # self.test_size = getattr(self, 'test_size', 0.25)
    # self.initial_epoch = getattr(self, 'initial_epoch', 0)
    # self.max_epoch = getattr(self, 'max_epoch', 50)
    # self.shuffle_train = getattr(self, 'shuffle_train', True)

    # set model parameters
    # self.network = getattr(self, 'network', 'unet')
    # self.optimizer = getattr(self, 'optimizer', 'Adam')
    # self.loss = getattr(self, 'loss', 'categorical_crossentropy')
    # self.metrics = getattr(self, 'metrics', ['accuracy'])

    # system performance settings
    # self.cuda_devices = getattr(self, 'cuda_devices', '0,1,2,3')
    # self.mixed_precission = getattr(self, 'mixed_precission', True)
    # self.xla = getattr(self, 'xla', False)
    # self.device = torch.device(
    #    "cuda" if torch.cuda.is_available() else "cpu")

    # setup directories for input and output
    # self.dataset_dir = os.path.join(self.data_output_dir, 'dataset')

    # directories to store new image and labels tiles for training
    # self.images_dir = os.path.join(self.dataset_dir, 'images')
    # self.labels_dir = os.path.join(self.dataset_dir, 'labels')

    # logging files
    # self.logs_dir = os.path.join(self.data_output_dir, 'logs')
    # self.log_file = os.path.join(
    #    self.logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S") +
    #    f'-{self.experiment_name}.out')

    # directory to store and retrieve the model object from
    # self.model_dir = os.path.join(self.data_output_dir, 'model')

    # directory to store prediction products
    # self.predict_dir = os.path.join(self.inference_output_dir)

    # setup directory structure, create directories
    # directories_list = [
    #    self.images_dir, self.labels_dir, self.model_dir,
    #    self.predict_dir, self.logs_dir]
    # self.create_dirs(directories_list)

    # std and means filename for preprocessing and training
    # self.std_mean_filename = os.path.join(
    #    self.dataset_dir, f'{self.experiment_name}_mean_std.npz')
    # set logger
    # self.set_logger(filename=self.log_file)


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    schema = OmegaConf.structured(Config)
    conf = OmegaConf.load("../config/config_clouds/vietnam_clouds.yaml")
    try:
        OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")
    sys.exit(conf)
