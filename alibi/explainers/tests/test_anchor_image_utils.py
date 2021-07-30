import numpy as np

from alibi.explainers.anchor_image_utils import scale_image


def test_scale_image():
    image_shape = (28, 28, 1)
    scaling_offset = 260
    min_val = 0
    max_val = 255

    fake_img = np.random.random(size=image_shape) + scaling_offset
    scaled_img = scale_image(fake_img, scale=(min_val, max_val))
    assert (scaled_img <= max_val).all()
    assert (scaled_img >= min_val).all()
