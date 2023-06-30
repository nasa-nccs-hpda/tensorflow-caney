from tensorflow_caney.networks import unet
from tensorflow_caney.networks.regression import unet_regression
from tensorflow_caney.model.networks.regression import regression_unet
from tensorflow_caney.model.networks.vision import segmentation_unet
from tensorflow_caney.networks import deeplabv3_plus
from tensorflow_caney.utils import losses
from tensorflow_caney.model.losses import binary_focal_loss

from pkg_resources import get_distribution, DistributionNotFound

try:
    VERSION = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    try:
        from .version import __version__ as VERSION  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError(
            "Failed to find (autogenerated) version.py. "
            "This might be because you are installing from GitHub's tarballs, "
            "use the PyPI ones."
        )
__version__ = VERSION

__all__ = [
    "deeplabv3_plus", "unet", "unet_regression", "segmentation_unet",
    "regression_unet", "losses", "regression_unet", "binary_focal_loss",
]
