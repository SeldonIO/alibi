import pytest

import numpy as np
from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR_IMG
from alibi.explainers import AnchorImage
from alibi.explainers.anchor_image_sampler import AnchorImageSampler


@pytest.mark.parametrize(
    "models",
    [("mnist-cnn-tf2.2.0",), ("mnist-cnn-tf1.15.2.h5",)],
    ids="model={}".format,
    indirect=True,
)
def test_sampler(models, mnist_data):
    eps = 0.0001  # tolerance for tensor comparisons
    num_samples = 10

    x_train = mnist_data["X_train"]
    segmentation_fn = "slic"
    segmentation_kwargs = {"n_segments": 10, "compactness": 10, "sigma": 0.5}
    image_shape = (28, 28, 1)
    predict_fn = lambda x: models[0].predict(x)  # noqa: E731
    explainer = AnchorImage(
        predict_fn,
        image_shape,
        segmentation_fn=segmentation_fn,
        segmentation_kwargs=segmentation_kwargs,
    )

    image = x_train[0]
    p_sample = 0.5  # probability of perturbing a superpixel
    n_covered_ex = 3  # nb of examples where the anchor applies that are saved
    sampler = AnchorImageSampler(
        predictor=explainer.predictor,
        segmentation_fn=explainer.segmentation_fn,
        custom_segmentation=explainer.custom_segmentation,
        image=image,
        images_background=explainer.images_background,
        p_sample=p_sample,
        n_covered_ex=n_covered_ex,
    )

    image_preproc = sampler._preprocess_img(image)
    superpixels_mask = sampler._choose_superpixels(num_samples=num_samples)

    # grayscale image should be replicated across channel dim before segmentation
    assert image_preproc.shape[-1] == 3
    for channel in range(image_preproc.shape[-1]):
        assert (image.squeeze() - image_preproc[..., channel] <= eps).all()
    # check superpixels mask
    assert superpixels_mask.shape[0] == num_samples
    assert superpixels_mask.shape[1] == len(list(np.unique(sampler.segments)))
    assert superpixels_mask.sum(axis=1).any() <= segmentation_kwargs["n_segments"]
    assert superpixels_mask.any() <= 1

    cov_true, cov_false, labels, data, coverage, _ = sampler(
        (0, ()), num_samples
    )
    assert data.shape[0] == labels.shape[0]
    assert data.shape[1] == len(np.unique(sampler.segments))
    assert coverage == -1
