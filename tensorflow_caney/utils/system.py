import os
import numpy as np
import cupy as cp
import tensorflow as tf
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy


def seed_everything(seed: int = 42) -> None:
    """
    Seed libraries.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cp.random.seed(seed)
    return


def set_gpu_strategy(gpu_devices: str = "0,1,2,3") -> MirroredStrategy:
    """
    Set training strategy.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    devices = tf.config.list_physical_devices('GPU')
    assert len(devices) != 0, "No GPU devices found."
    return MirroredStrategy()


def set_mixed_precision(mixed_precision: bool = True) -> None:
    """
    Enable mixed precision.
    """
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    return


def set_xla(xla: bool = False) -> None:
    """
    Enable linear acceleration.
    """
    if xla:
        tf.config.optimizer.set_jit(True)
    return
