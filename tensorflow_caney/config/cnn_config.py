from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    CNN data configuration class (embedded with OmegaConf).
    """

    # directory to store all data files
    data_dir: str
    model_dir: Optional[str]

    # string with model function
    model: str

    # directory to store inference output files
    inference_save_dir: str = "results"

    # experiment name to track
    experiment_name: str = "unet-cnn"

    # experiment type to track (normally embedded in the inference output)
    experiment_type: str = "landcover"

    # seed to control the randomization
    seed: Optional[int] = 24

    # gpu devices to utilize
    gpu_devices: str = "0,1,2,3"

    # bool to enable mixed_precision
    mixed_precision: Optional[bool] = True

    # bool to enable linear acceleration
    xla: Optional[bool] = False

    # input bands from the incoming dataset
    input_bands: List[str] = field(
        default_factory=lambda: [
            "Blue", "Green", "Red", "NIR1", "HOM1", "HOM2"]
    )

    # output bands that will be used to train and predict from
    output_bands: List[str] = field(
        default_factory=lambda: ["Blue", "Green", "Red", "NIR1"]
    )

    # list of strings to support the modification of labels
    modify_labels: Optional[List[str]] = None

    # subtract 1 from labels
    substract_labels: Optional[bool] = False

    expand_dims: bool = True
    tile_size: int = 256
    include_classes: bool = False
    augment: bool = True

    normalize: Optional[float] = None
    rescale: Optional[str] = None
    standardization: Optional[str] = None

    batch_size: int = 32
    n_classes: int = 1
    test_size: float = 0.20

    mean: List[float] = field(default_factory=lambda: [])
    std: List[float] = field(default_factory=lambda: [])

    # loss function expression, expects a loss function
    loss: str = "tf.keras.losses.CategoricalCrossentropy"

    # optimizer function expression, expects an optimizer function
    optimizer: str = "tf.keras.optimizers.Adam"

    # metrics function expression, expects list of metrics
    metrics: List[str] = field(
        default_factory=lambda: ["tf.keras.metrics.Recall"])

    # callbacks function expression, expects list of metrics
    callbacks: List[str] = field(
        default_factory=lambda: ["tf.keras.callbacks.ModelCheckpoint"]
    )

    learning_rate: Optional[float] = 0.0001
    max_epochs: Optional[int] = 6000
    patience: int = 7

    model_filename: Optional[str] = None
    inference_regex: str = "*.tif"
    window_size: Optional[int] = 8120
    inference_overlap: Optional[int] = 2
    inference_treshold: Optional[float] = 0.5
    pred_batch_size: Optional[int] = 128
