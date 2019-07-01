import pytest
from alibi.datasets import imagenet


@pytest.mark.parametrize('nb_images', [3])
@pytest.mark.parametrize('category', ['Persian cat', 'volcano', 'strawberry', 'centipede', 'jellyfish'])
def test_imagenet(nb_images, category):
    data, labels = imagenet(category=category, nb_images=nb_images, target_size=(299, 299))

    assert data.shape == (nb_images, 299, 299, 3)  # 3 color channels
    assert data.max() <= 255  # RGB limits
    assert data.min() >= 0

    assert len(labels) == nb_images
