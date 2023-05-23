import numpy as np

__all__ = ["center_crop"]


def center_crop(image: np.array, dim: tuple) -> np.array:
    """Center crop numpy array

    :param image: image to be center cropped
    :type image: np.array
    :param dim: dimensions (width, height) to be cropped
    :type dim: tuple
    :return: center cropped image
    :rtype: np.array
    """
    width, height = image.shape[1], image.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < image.shape[1] else image.shape[1]
    crop_height = dim[1] if dim[1] < image.shape[0] else image.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = image[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img
