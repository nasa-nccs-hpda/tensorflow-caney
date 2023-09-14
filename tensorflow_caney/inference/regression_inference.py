import numpy as np
import tensorflow as tf
from tiler import Tiler, Merger
from ..utils.data import standardize_image


def sliding_window_tiler(
            xraster,
            model,
            n_classes: int,
            pad_style: str = 'reflect',
            overlap: float = 0.50,
            constant_value: int = 600,
            batch_size: int = 1024,
            threshold: float = 0.50,
            standardization: str = None,
            mean=None,
            std=None,
            normalize=1.0,
            window: str = 'triang'  # 'boxcar', 'overlap-tile'
        ):
    """
    Sliding window using tiler.
    """
    tile_size = model.layers[0].input_shape[0][1]
    tile_channels = model.layers[0].input_shape[0][-1]

    if tile_size is None:
        tile_size = 32

    tiler_image = Tiler(
        data_shape=xraster.shape,
        tile_shape=(tile_size, tile_size, tile_channels),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value
    )

    # Define the tiler and merger based on the output size of the prediction
    tiler_mask = Tiler(
        data_shape=(xraster.shape[0], xraster.shape[1], n_classes),
        tile_shape=(tile_size, tile_size, n_classes),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value
    )

    merger = Merger(tiler=tiler_mask, window=window)

    # Iterate over the data in batches
    for batch_id, batch_i in tiler_image(xraster, batch_size=batch_size):

        # Standardize
        batch = batch_i.copy()

        if standardization is not None:
            for item in range(batch.shape[0]):
                batch[item, :, :, :] = standardize_image(
                    batch[item, :, :, :], standardization, mean, std)

        # Predict - working on TensorRT
        # if isinstance(model, tf.python.saved_model.load._WrapperFunction):
        # batch = model(batch)['probs'].numpy()
        # else:
        batch = model.predict(batch, batch_size=batch_size, verbose=0)

        # Merge the updated data in the array
        merger.add_batch(batch_id, batch_size, batch)

    prediction = np.squeeze(merger.merge(unpad=True))
    return prediction
