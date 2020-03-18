# flake8: noqa E731
import pytest

import numpy as np
from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR_IMG
from alibi.explainers import AnchorImage
from alibi.explainers.tests.utils import fashion_mnist_dataset


# Data preparation
data = fashion_mnist_dataset()
x_train = data['X_train']
y_train = data['y_train']


@pytest.mark.parametrize('conv_net', (data,), indirect=True)
def test_anchor_image(conv_net):

    segmentation_fn = 'slic'
    segmentation_kwargs = {'n_segments': 10, 'compactness': 10, 'sigma': .5}
    image_shape = (28, 28, 1)
    p_sample = 0.5  # probability of perturbing a superpixel
    num_samples = 10
    # img scaling settings
    scaling_offset = 260
    min_val = 0
    max_val = 255
    eps = 0.0001  # tolerance for tensor comparisons
    n_covered_ex = 3  # nb of examples where the anchor applies that are saved

    # define and train model
    clf = conv_net
    predict_fn = lambda x: clf.predict(x)

    explainer = AnchorImage(
        predict_fn,
        image_shape,
        segmentation_fn=segmentation_fn,
        segmentation_kwargs=segmentation_kwargs,
    )
    # test explainer initialization
    assert explainer.predictor(np.zeros((1,) + image_shape)).shape == (1,)
    assert explainer.custom_segmentation == False

    # test sampling and segmentation functions
    image = x_train[0]
    explainer.instance_label = predict_fn(image[np.newaxis, ...])[0]
    explainer.image = image
    explainer.n_covered_ex = n_covered_ex
    explainer.p_sample = p_sample
    segments = explainer.generate_superpixels(image)
    explainer.segments = segments
    image_preproc = explainer._preprocess_img(image)
    explainer.segment_labels = list(np.unique(segments))
    superpixels_mask = explainer._choose_superpixels(num_samples=num_samples)

    # grayscale image should be replicated across channel dim before segmentation
    assert image_preproc.shape[-1] == 3
    for channel in range(image_preproc.shape[-1]):
        assert (image.squeeze() - image_preproc[..., channel] <= eps).all() == True
    # check superpixels mask
    assert superpixels_mask.shape[0] == num_samples
    assert superpixels_mask.shape[1] == len(list(np.unique(segments)))
    assert superpixels_mask.sum(axis=1).any() <= segmentation_kwargs['n_segments']
    assert superpixels_mask.any() <= 1

    cov_true, cov_false, labels, data, coverage, _ = explainer.sampler((0, ()), num_samples)
    assert data.shape[0] == labels.shape[0]
    assert data.shape[1] == len(np.unique(segments))
    assert coverage == -1

    # test explanation
    threshold = .95
    explanation = explainer.explain(image, threshold=threshold)

    if explanation.raw['feature']:
        assert explanation.raw['examples'][-1]['covered_true'].shape[0] <= explainer.n_covered_ex
        assert explanation.raw['examples'][-1]['covered_false'].shape[0] <= explainer.n_covered_ex
    else:
        assert not explanation.raw['examples']
    assert explanation.anchor.shape == image_shape
    assert explanation.precision >= threshold
    assert len(np.unique(explanation.segments)) == len(np.unique(segments))
    assert explanation.meta.keys() == DEFAULT_META_ANCHOR.keys()
    assert explanation.data.keys() == DEFAULT_DATA_ANCHOR_IMG.keys()

    # test scaling
    fake_img = np.random.random(size=image_shape) + scaling_offset
    scaled_img = explainer._scale(fake_img, scale=(min_val, max_val))
    assert (scaled_img <= max_val).all()
    assert (scaled_img >= min_val).all()
