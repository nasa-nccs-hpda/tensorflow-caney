import os
import inspect
import logging
import numpy as np
import tensorflow as tf
from types import ModuleType
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

_warn_array_module_once = False

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

__all__ = [
    "seed_everything", "set_gpu_strategy",
    "set_mixed_precision", "set_xla"
]


def array_module(xp=None):
    """
    Find the array module to use, for example **numpy** or **cupy**.

    :param xp: The array module to use, for example, 'numpy'
               (normal CPU-based module) or 'cupy' (GPU-based module).
               If not given, will try to read
               from the ARRAY_MODULE environment variable. If not given and
               ARRAY_MODULE is not set,
               will use numpy. If 'cupy' is requested, will
               try to 'import cupy'. If that import fails, will
               revert to numpy.
    :type xp: optional, string or Python module
    :rtype: Python module

    >>> from pysnptools.util import array_module
    >>> xp = array_module() # will look at environment variable
    >>> print(xp.zeros((3)))
    [0. 0. 0.]
    >>> xp = array_module('cupy') # will try to import 'cupy'
    >>> print(xp.zeros((3)))
    [0. 0. 0.]

    # License: Apache 2.0
    # Carl Kadie
    # https://fastlmm.github.io/
    """
    xp = xp or os.environ.get("ARRAY_MODULE", "numpy")

    if isinstance(xp, ModuleType):
        return xp

    if xp == "cupy" or xp is None:
        try:
            import cupy as cp

            return cp
        except ModuleNotFoundError as e:
            global _warn_array_module_once
            if not _warn_array_module_once:
                logging.warning(f"Using numpy. ({e})")
                _warn_array_module_once = True
            return np

    if xp == "numpy":
        return np

    raise ValueError(f"Don't know ARRAY_MODULE '{xp}'")


def get_array_module(a):
    """
    Given an array, returns the array's
    module, for example, **numpy** or **cupy**.
    Works for numpy even when cupy is not available.

    >>> import numpy as np
    >>> zeros_np = np.zeros((3))
    >>> xp = get_array_module(zeros_np)
    >>> xp.ones((3))
    array([1., 1., 1.])
    """
    submodule = inspect.getmodule(type(a))
    module_name = submodule.__name__.split(".")[0]
    xp = array_module(module_name)
    return xp


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
    try:
        cp.random.seed(seed)
    except RuntimeError:
        return
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
        None
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
        None
    """
    if xla:
        tf.config.optimizer.set_jit(True)
    return
