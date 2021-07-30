import numpy as np


def scale_image(image: np.ndarray, scale: tuple = (0, 255)) -> np.ndarray:
    """
    Scales an image in a specified range.

    Parameters
    ----------
    image
        Image to be scale.
    scale
        The scaling interval.

    Returns
    -------
    img_scaled
        Scaled image.
    """

    img_max, img_min = image.max(), image.min()
    img_std = (image - img_min) / (img_max - img_min)
    img_scaled = img_std * (scale[1] - scale[0]) + scale[0]

    return img_scaled
