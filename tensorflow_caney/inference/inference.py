import logging
import numpy as np
import tensorflow as tf
import scipy.signal.windows as w
from tqdm import tqdm
# from .tiler.tiler import Tiler, Merger
from .mosaic import from_array
from ..utils.data import normalize_image, rescale_image, \
    standardize_batch, standardize_image
from tiler import Tiler, Merger


def window2d(window_func, window_size, **kwargs):
    window = np.matrix(window_func(M=window_size, sym=False, **kwargs))
    return window.T.dot(window)


def generate_corner_windows(window_func, window_size, **kwargs):
    step = window_size >> 1
    window = window2d(window_func, window_size, **kwargs)
    window_u = np.vstack(
        [np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
    window_b = np.vstack(
        [window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
    window_l = np.hstack(
        [np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
    window_r = np.hstack(
        [window[:, :step], np.tile(window[:, step:step+1], (1, step))])
    window_ul = np.block([
        [np.ones((step, step)), window_u[:step, step:]],
        [window_l[step:, :step], window_l[step:, step:]]])
    window_ur = np.block([
        [window_u[:step, :step], np.ones((step, step))],
        [window_r[step:, :step], window_r[step:, step:]]])
    window_bl = np.block([
        [window_l[:step, :step], window_l[:step, step:]],
        [np.ones((step, step)), window_b[step:, step:]]])
    window_br = np.block([
        [window_r[:step, :step], window_r[:step, step:]],
        [window_b[step:, :step], np.ones((step, step))]])
    return np.array([
        [window_ul, window_u, window_ur],
        [window_l, window, window_r],
        [window_bl, window_b, window_br],
    ])


def generate_patch_list(
            image_width,
            image_height,
            window_func,
            window_size,
            overlapping=False
        ):
    patch_list = []
    if overlapping:
        step = window_size >> 1
        windows = generate_corner_windows(window_func, window_size)
        # max_height = int(image_height/step - 1) * step
        # max_width = int(image_width/step - 1) * step
        # print("max_height, max_width", max_height, max_width)
    else:
        step = window_size
        windows = np.ones((window_size, window_size))
        # max_height = int(image_height / step) * step
        # max_width = int(image_width / step) * step
        # print("else max_height, max_width", max_height, max_width)

    # for i in range(0, max_height, step):
    #    for j in range(0, max_width, step):
    for i in range(0, image_height-step, step):
        for j in range(0, image_width-step, step):
            if overlapping:
                # Close to border and corner cases
                # Default (1, 1) is regular center window
                border_x, border_y = 1, 1
                if i == 0:
                    border_x = 0
                if j == 0:
                    border_y = 0
                if i == image_height - step:
                    border_x = 2
                if j == image_width - step:
                    border_y = 2
                # Selecting the right window
                current_window = windows[border_x, border_y]
            else:
                current_window = windows

            # The patch is cropped when the patch size is not
            # a multiple of the image size.
            patch_height = window_size
            if i+patch_height > image_height:
                patch_height = image_height - i

            patch_width = window_size
            if j+patch_width > image_width:
                patch_width = image_width - j

            # print(f'i {i} j {j} patch_height {patch_height}
            # patch_width {patch_width}')

            # Adding the patch
            patch_list.append(
                (
                    j,
                    i,
                    patch_width,
                    patch_height,
                    current_window[:patch_width, :patch_height]
                )
            )
    return patch_list


def sliding_window(
            xraster, model, window_size, tile_size,
            inference_overlap, inference_treshold, batch_size,
            mean, std, n_classes, standardization, normalize,
            use_hanning=True
        ):

    # original_shape = xraster[:, :, 0].shape
    # xsum = int(((-xraster[:, :, 0].shape[0] % tile_size) + \
    # (tile_size * 4)) / 2)
    # ysum = int(((-xraster[:, :, 0].shape[1] % tile_size) + \
    # (tile_size * 4)) / 2)
    # print("xsum", xsum, "ysum", ysum)

    # xraster = np.pad(xraster, ((ysum, ysum), (xsum, xsum), (0, 0)),
    #   mode='symmetric')#'reflect')

    # print("RASTER SHAPE AFTER PAD", xraster.shape)

    # open rasters and get both data and coordinates
    rast_shape = xraster[:, :, 0].shape  # shape of the wider scene

    # in memory sliding window predictions
    wsy, wsx = window_size, window_size
    # wsy, wsx = rast_shape[0], rast_shape[1]

    # if the window size is bigger than the image, predict full image
    if wsy > rast_shape[0]:
        wsy = rast_shape[0]
    if wsx > rast_shape[1]:
        wsx = rast_shape[1]

    # smooth window
    # this might be problematic since there might be issues on tiles smaller
    # than actual squares
    # spline = spline_window(wsy)

    # print(rast_shape, wsy, wsx)
    prediction = np.zeros(rast_shape + (n_classes,))  # crop out the window
    logging.info(f'wsize: {wsy}x{wsx}. Prediction shape: {prediction.shape}')
    # 2022-03-09 13:47:03; INFO; wsize: 10000x10000.
    # Prediction shape: (38702, 71223, 2)

    window_func = w.hann
    patch_list = generate_patch_list(
        rast_shape[0], rast_shape[1], window_func, wsy, use_hanning)
    pp = len(patch_list)
    # logging.info(f'Patch list done {pp}')

    counter = 0
    for patch in patch_list:

        counter += 1

        logging.info(f'{counter} out of {pp}')
        patch_x, patch_y, patch_width, patch_height, window = patch

        # if patch_x + patch_width > rast_shape[1]:
        #    patch_width =

        # logging.info(f'{patch_x}, {patch_width+patch_x}, {patch_y},
        # {patch_height+patch_y}')
        # logging.info(f'{psutil.virtual_memory().percent}')

        input_path = xraster[
            patch_x:patch_x+patch_width, patch_y:patch_y+patch_height]

        # print("firts", input_path.shape)

        if np.all(input_path == input_path[0, 0, 0]):

            input_path_shape = input_path.shape
            prediction[
                patch_x:patch_x+patch_width,
                patch_y:patch_y+patch_height] = np.eye(n_classes)[
                    np.zeros(
                        (input_path_shape[0], input_path_shape[1]), dtype=int)]

        else:

            # Normalize values within [0, 1] range
            input_path = normalize_image(input_path, normalize)

            input_path = from_array(
                input_path, (tile_size, tile_size),
                overlap_factor=inference_overlap, fill_mode='reflect')

            input_path = input_path.apply(
                model.predict, progress_bar=False,
                batch_size=batch_size, mean=mean, std=std,
                standardization=standardization)

            input_path = input_path.get_fusion()

            prediction[
                patch_x:patch_x+patch_width,
                patch_y:patch_y+patch_height] += \
                input_path * np.expand_dims(window, -1)

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
        # print('Window shape bef. spline', window.shape, window_spline.shape)
        # window = window * window_spline
    else:
        prediction = np.squeeze(
            np.where(
                prediction > inference_treshold, 1, 0).astype(np.int16)
            )

    # print("SHAPR PREDICTION", prediction.shape)
    # prediction = prediction[
    # xsum:rast_shape[0] - xsum, ysum:rast_shape[1] - ysum]
    # print("SHAPR PREDICTION AFTER CROP", prediction.shape)
    return prediction

# 2022-04-18 13:13:08; INFO; Prediction shape: (51596, 9092, 8)
# 2022-04-18 13:13:55; INFO; Prediction shape after modf: (51596, 9092, 4)
# 2022-04-18 13:25:47; INFO; wsize: 10000x10000.
# Prediction shape: (52744, 10232, 1)
# SHAPR PREDICTION (52744, 10232)
# SHAPR PREDICTION AFTER CROP (51604, 9084)


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
            window: str = 'triang'  # 'overlap-tile'
        ):
    """
    Sliding window using tiler.
    """
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = \
    #    tf.data.experimental.AutoShardPolicy.OFF
    # batch = tf.data.Dataset.from_tensor_slices(
    #    np.expand_dims(batch, axis=0))
    # batch = batch.with_options(self.options)
    # batch = function(batch, batch_size=batch_size)

    tile_size = model.layers[0].input_shape[0][1]
    tile_channels = model.layers[0].input_shape[0][-1]

    tiler_image = Tiler(
        data_shape=xraster.shape,
        tile_shape=(tile_size, tile_size, tile_channels),
        channel_dimension=2,
        # overlap=overlap,
        mode=pad_style,
        constant_value=600
    )

    # Define the tiler and merger based on the output size of the prediction
    tiler_mask = Tiler(
        data_shape=(xraster.shape[0], xraster.shape[1], n_classes),
        tile_shape=(tile_size, tile_size, n_classes),
        channel_dimension=2,
        # overlap=overlap,
        mode=pad_style,
        constant_value=600
    )

    # new_shape_image, padding_image = tiler_image.calculate_padding()
    # new_shape_mask, padding_mask = tiler_mask.calculate_padding()
    # print(xraster.shape, new_shape_image, new_shape_mask)
    # tiler_image.recalculate(data_shape=new_shape_image)
    # tiler_mask.recalculate(data_shape=new_shape_mask)

    # merger = Merger(tiler=tiler_mask, window=window, logits=4)
    merger = Merger(
        tiler=tiler_mask, window=window)  # #logits=4,
    #    tile_shape_merge=(tile_size, tile_size))
    # print(merger)
    # print("WEIGHTS SHAPE", merger.weights_sum.shape)
    # print("WINDOW SHAPE", merger.window.shape)

    # xraster = xraster.pad(
    #    y=padding_image[0], x=padding_image[1],
    #    constant_values=constant_value)
    # print("After pad", xraster.shape)

    # Iterate over the data in batches
    for batch_id, batch in tiler_image(xraster, batch_size=batch_size):

        # print("AFTER SELECT", batch.shape)

        # Standardize
        batch = batch / 10000.0

        # if standardization is not None:
        #    batch = standardize_batch(batch, standardization, mean, std)

        # print("AFTER STD", batch.shape)

        # Predict
        batch = model.predict(batch, batch_size=batch_size)
        # batch = np.moveaxis(batch, -1, 1)
        # print("AFTER PREDICT", batch.shape, batch_id)

        # Merge the updated data in the array
        merger.add_batch(batch_id, batch_size, batch)

    # prediction = merger.merge(
    # extra_padding=padding_mask, unpad=True, dtype=xraster.dtype,
    # normalize_by_weights=False)
    prediction = merger.merge(unpad=True)

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
    else:
        prediction = np.squeeze(
            np.where(prediction > threshold, 1, 0).astype(np.int16)
        )

    """
    tiler1 = Tiler(
        data_shape=xraster.shape,
        tile_shape=(tile_size, tile_size, tile_channels),
        channel_dimension=2,
        overlap=0.50,
        #mode=pad_style,
        #constant_value=constant_value
    )

    tiler2 = Tiler(
        data_shape=xraster.shape,
        tile_shape=(tile_size, tile_size, n_classes),
        channel_dimension=2,
        overlap=(256, 256, 0),
        #mode=pad_style,
        #constant_value=constant_value
    )

    new_shape, padding = tiler1.calculate_padding()
    tiler1.recalculate(data_shape=new_shape)
    tiler2.recalculate(data_shape=new_shape)
    padded_image = np.pad(
        xraster.values, padding, mode=pad_style,
        constant_values=constant_value
    )

    merger = Merger(tiler=tiler2, window="overlap-tile")
    #merger = Merger(tiler=tiler2, logits=2)#, window="triang")
    #"hann")#"triang")#"barthann")#"overlap-tile")

    for batch_id, batch in tiler1(padded_image, batch_size=batch_size):
    #for batch_id, batch in tiler1(xraster, batch_size=batch_size):

        # remove no-data
        batch[batch < 0] = constant_value

        # Standardize
        if standardization is not None:
            print("STD")
            batch = standardize_batch(batch, standardization, mean, std)

        # preprocessing
        batch = model.predict(batch, batch_size=batch_size)

        batch = np.argmax(batch, axis=-1)

        # merge batch
        merger.add_batch(batch_id, batch_size, batch)

    #prediction = merger.merge(extra_padding=padding, dtype=xraster.dtype)
    # prediction = merger.merge(
    #    extra_padding=padding, dtype=xraster.dtype,
    # normalize_by_weights=False)
    #prediction = merger.merge(unpad=True)
    prediction = merger.merge(extra_padding=padding, dtype=padded_image.dtype)

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
    else:
        prediction = np.squeeze(
            np.where(prediction > threshold, 1, 0).astype(np.int16)
        )

    logging.info(f"Mask info: {prediction.shape}, {prediction.min()},
    {prediction.max()}")
    """
    return prediction


def sliding_window_tiler_multiclass(
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
            normalize: float = 1.0,
            rescale: str = None,
            window: str = 'triang'  # 'overlap-tile'
        ):
    """
    Sliding window using tiler.
    """

    tile_size = model.layers[0].input_shape[0][1]
    tile_channels = model.layers[0].input_shape[0][-1]
    # n_classes = out of the output layer, output_shape

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

    # new_shape_image, padding_image = tiler_image.calculate_padding()
    # new_shape_mask, padding_mask = tiler_mask.calculate_padding()
    # print(xraster.shape, new_shape_image, new_shape_mask)

    # tiler_image.recalculate(data_shape=new_shape_image)
    # tiler_mask.recalculate(data_shape=new_shape_mask)

    # merger = Merger(tiler=tiler_mask, window=window, logits=4)
    merger = Merger(tiler=tiler_mask, window=window)  # , #logits=4,
    #    tile_shape_merge=(tile_size, tile_size))
    # print(merger)
    # print("WEIGHTS SHAPE", merger.weights_sum.shape)
    # print("WINDOW SHAPE", merger.window.shape)

    # xraster = xraster.pad(
    #    y=padding_image[0], x=padding_image[1],
    #    constant_values=constant_value)
    # print("After pad", xraster.shape)

    xraster = normalize_image(xraster, normalize)

    if rescale is not None:
        xraster = rescale_image(xraster, rescale)

    # Iterate over the data in batches
    for batch_id, batch_i in tiler_image(xraster, batch_size=batch_size):

        # Standardize
        batch = batch_i.copy()

        if standardization is not None:
            # batch = standardize_batch(batch, standardization, mean, std)
            for item in range(batch.shape[0]):
                batch[item, :, :, :] = standardize_image(
                    batch[item, :, :, :], standardization, mean, std)

        # print("AFTER STD", batch.shape)

        # Predict
        batch = model.predict(batch, batch_size=batch_size)
        # batch = np.moveaxis(batch, -1, 1)
        # print("AFTER PREDICT", batch.shape, batch_id)

        # Merge the updated data in the array
        merger.add_batch(batch_id, batch_size, batch)

    # prediction = merger.merge(
    #    extra_padding=padding_mask, unpad=True, dtype=xraster.dtype,
    # normalize_by_weights=False)
    prediction = merger.merge(unpad=True)
    print("MIN MAX", prediction.min(), prediction.max())

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
    else:
        prediction = np.squeeze(
            np.where(prediction > threshold, 1, 0).astype(np.int16)
        )
    print("UNIQUE", np.unique(prediction))
    return prediction


def get_extract_pred_scatter(
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
            window: str = 'triang'  # 'overlap-tile'
        ):

    with tf.device('/CPU:0'):
        tile_size = model.layers[0].input_shape[0][1]
        tile_stride = tile_size
        # tile_channels = model.layers[0].input_shape[0][-1]
        PATCH_RATE = 1
        SIZES = [1, tile_size, tile_size, 1]
        STRIDES = [1, tile_stride, tile_stride, 1]
        RATES = [1, PATCH_RATE, PATCH_RATE, 1]
        PADDING = 'VALID'

        H, W, C = xraster.shape
        # patch_number
        tile_PATCH_NUMBER = (
            (H - tile_size) // tile_stride + 1) * \
            ((W - tile_size) // tile_stride + 1)
        # print(tile_PATCH_NUMBER)
        # the indices trick to reconstruct the tile
        x = tf.range(W)
        y = tf.range(H)
        x, y = tf.meshgrid(x, y)
        indices = tf.stack([y, x], axis=-1)
        # print(indices)
        # making patches, TensorShape([2, 17, 17, 786432])
        img_patches = tf.image.extract_patches(
            images=tf.expand_dims(xraster, axis=0),
            sizes=SIZES, strides=STRIDES, rates=RATES, padding=PADDING)
        ind_patches = tf.image.extract_patches(
            images=tf.expand_dims(indices, axis=0),
            sizes=SIZES, strides=STRIDES, rates=RATES, padding=PADDING)
        # print(img_patches)
        # squeezing the shape (removing dimension of size 1)
        img_patches = tf.squeeze(img_patches)
        ind_patches = tf.squeeze(ind_patches)
        # reshaping
        img_patches = tf.reshape(
            img_patches, [tile_PATCH_NUMBER, tile_size, tile_size, C])
        ind_patches = tf.reshape(
            ind_patches, [tile_PATCH_NUMBER, tile_size, tile_size, 2])

        img_patches = img_patches / 10000
        img_patches = tf.image.per_image_standardization(img_patches)
        print(type(img_patches))
    # Now predict
    # with tf.device('/GPU:0'):
    pred_patches = model.predict(img_patches, batch_size=batch_size)
    # stitch together the patch summing the overlapping patches probabilities
    pred_tile = tf.scatter_nd(
        indices=ind_patches, updates=pred_patches, shape=(H, W, n_classes))

    # pred_tile = pred_tile.numpy()
    # pred_tile = np.squeeze(
    #    np.where(pred_tile > threshold, 1, 0).astype(np.int16)
    # )
    # print("pred_tile", type(pred_tile), pred_tile.min(),
    # pred_tile.max(), pred_tile.shape)
    return pred_tile


def get_tile_tta_pred(
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
            window: str = 'triang'  # 'overlap-tile'
        ):
    """ test time augmentation prediction """
    # reading the tile content
    # if from_disk:
    #    img = imageio.imread(tile_path)
    # else:
    #    img = tile_path
    # img = tf.image.convert_image_dtype(img,tf.float32)
    H, W, C = xraster.shape
    pred_tile = tf.zeros(shape=(H, W, n_classes))
    for i in tqdm(tf.range(4)):
        rot_img = tf.image.rot90(xraster, k=i)
        # pred_tmp = get_extract_pred_scatter(rot_img, model)
        pred_tmp = get_extract_pred_scatter(
                rot_img,
                model,
                n_classes,
                pad_style,
                overlap,
                constant_value,
                batch_size,
                threshold,
                standardization,
                mean,
                std,
                window
            )
        pred_tile += tf.image.rot90(pred_tmp, k=-i)
    xraster = tf.image.flip_left_right(xraster)
    for i in tqdm(tf.range(4)):
        rot_img = tf.image.rot90(xraster, k=i)
        # pred_tmp = get_extract_pred_scatter(rot_img, model)
        pred_tmp = get_extract_pred_scatter(
                rot_img,
                model,
                n_classes,
                pad_style,
                overlap,
                constant_value,
                batch_size,
                threshold,
                standardization,
                mean,
                std,
                window
            )
        pred_tile += tf.image.flip_left_right(tf.image.rot90(pred_tmp, k=-i))
    # pred_tile    = tf.argmax(pred_tile, axis=-1, output_type=tf.int32)
    # pred_tile    = label2mask(pred_tile)
    pred_tile = pred_tile.numpy()
    pred_tile = np.squeeze(
        np.where(pred_tile > threshold, 1, 0).astype(np.int16)
    )
    return pred_tile


def sliding_window_tiler_multiclass_v2(
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
            window: str = 'triang'  # 'overlap-tile'
        ):
    """
    Sliding window using tiler.
    """

    tile_size = model.layers[0].input_shape[0][1]
    tile_channels = model.layers[0].input_shape[0][-1]

    # tiler_image = Tiler(
    #    data_shape=xraster.shape,
    #    tile_shape=(tile_size, tile_size, tile_channels),
    #    channel_dimension=2,
    #    overlap=overlap,
    #    mode=pad_style,
    #    constant_value=600
    # )
    tiler = Tiler(
        data_shape=xraster.shape,
        tile_shape=(tile_size, tile_size, tile_channels),
        channel_dimension=-1,
        overlap=0.5,
        # mode=pad_style,
        # constant_value=600
    )
    merger = Merger(
        tiler,
        ignore_channels=True,  # this allows to "turn off" channels from Tiler
        logits_n=n_classes,  # this specifies logits/segmentation classes
        logits_dim=-1,  # and in which dimension
        window=window
    )

    # Define the tiler and merger based on the output size of the prediction
    # tiler_mask = Tiler(
    #    data_shape=(xraster.shape[0], xraster.shape[1], n_classes),
    #    tile_shape=(tile_size, tile_size, n_classes),
    #    channel_dimension=2,
    #    overlap=overlap,
    #    mode=pad_style,
    #    constant_value=600
    # )

    # new_shape_image, padding_image = tiler_image.calculate_padding()
    # new_shape_mask, padding_mask = tiler_mask.calculate_padding()
    # print(xraster.shape, new_shape_image, new_shape_mask)

    # tiler_image.recalculate(data_shape=new_shape_image)
    # tiler_mask.recalculate(data_shape=new_shape_mask)

    # merger = Merger(tiler=tiler_mask, window=window, logits=4)
    #  merger = Merger(
    #    tiler=tiler_mask, window=window)#, #logits=4,
    #    tile_shape_merge=(tile_size, tile_size))
    # print(merger)
    # print("WEIGHTS SHAPE", merger.weights_sum.shape)
    # print("WINDOW SHAPE", merger.window.shape)

    # xraster = xraster.pad(
    #    y=padding_image[0], x=padding_image[1],
    #    constant_values=constant_value)
    # print("After pad", xraster.shape)

    # Iterate over the data in batches
    for batch_id, batchi in tiler(xraster, batch_size=batch_size):

        # print("AFTER SELECT", batch.shape)

        # Standardize
        batch = batchi.copy()  # / 10000.0

        if standardization is not None:
            batch = standardize_batch(batch, standardization, mean, std)

        # print("AFTER STD", batch.shape)

        # Predict
        batch = model.predict(batch, batch_size=batch_size)
        # batch = np.moveaxis(batch, -1, 1)
        # print("AFTER PREDICT", batch.shape, batch_id)

        # Merge the updated data in the array
        merger.add_batch(batch_id, batch_size, batch)

    # prediction = merger.merge(
    #    extra_padding=padding_mask, unpad=True,
    # dtype=xraster.dtype, normalize_by_weights=False)
    prediction = merger.merge(argmax=-1, unpad=True)
    print("MIN MAX", prediction.min(), prediction.max(), prediction.shape)

    # if prediction.shape[-1] > 1:
    #    prediction = np.argmax(prediction, axis=-1)
    # else:
    #    prediction = np.squeeze(
    #        np.where(prediction > threshold, 1, 0).astype(np.int16)
    #    )
    print("UNIQUE", np.unique(prediction))
    return prediction
