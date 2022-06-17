import os
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

__all__ = [
    "seed_everything", "set_gpu_strategy",
    "set_mixed_precision", "set_xla"
]


def seed_everything(seed: int = 42) -> None:
    """
    Seeds starting randomization from libraries.
    Args:
        seed (int): integer to seed libraries with.
    Returns:
        None.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if HAS_GPU:
        cp.random.seed(seed)
    return


def set_gpu_strategy(gpu_devices: str = "0,1,2,3") -> MirroredStrategy:
    """
    Set training strategy.
    Args:
        gpu_devices (str): string with gpu devices separated by commas.
    Returns:
        tf.MirroredStrategy() for multi-gpu training.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    devices = tf.config.list_physical_devices('GPU')
    assert len(devices) != 0, "No GPU devices found."
    return MirroredStrategy()


def set_mixed_precision(mixed_precision: bool = True) -> None:
    """
    Enable mixed precision.
    Args:
        mixed_precision (bool): enable mixed precision.
    Returns:
        None.
    """
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    return


def set_xla(xla: bool = False) -> None:
    """
    Enable linear algebra acceleration.
    Args:
        xla (bool): enable xla.
    Returns:
        None.
    """
    if xla:
        tf.config.optimizer.set_jit(True)
    return
