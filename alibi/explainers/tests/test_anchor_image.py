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
def test_anchor_image(models, mnist_data):
    x_train = mnist_data["X_train"]
    image = x_train[0]

    segmentation_fn = "slic"
    segmentation_kwargs = {"n_segments": 10, "compactness": 10, "sigma": 0.5}
    image_shape = (28, 28, 1)
    n_covered_ex = 3  # nb of examples where the anchor applies that are saved

    # define and train model
    # model = conv_net
    predict_fn = lambda x: models[0].predict(x)  # noqa: E731

    explainer = AnchorImage(
        predict_fn,
        image_shape,
        segmentation_fn=segmentation_fn,
        segmentation_kwargs=segmentation_kwargs,
    )

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

    # test explainer initialization
    assert explainer.predictor(np.zeros((1,) + image_shape)).shape == (1,)
    assert explainer.custom_segmentation is False

    # test explanation
    threshold = 0.95
    explanation = explainer.explain(image, threshold=threshold, n_covered_ex=3)

    if explanation.raw["feature"]:
        assert (
            len(explanation.raw["examples"][-1]["covered_true"]) <= sampler.n_covered_ex
        )
        assert (
            len(explanation.raw["examples"][-1]["covered_false"])
            <= sampler.n_covered_ex
        )
    else:
        assert not explanation.raw["examples"]
    assert explanation.anchor.shape == image_shape
    assert explanation.precision >= threshold
    assert len(np.unique(explanation.segments)) == len(np.unique(sampler.segments))
    assert explanation.meta.keys() == DEFAULT_META_ANCHOR.keys()
    assert explanation.data.keys() == DEFAULT_DATA_ANCHOR_IMG.keys()


@pytest.mark.parametrize(
    "models",
    [("mnist-cnn-tf2.2.0",)],
    ids="model={}".format,
    indirect=True,
)
def test_stateless_explainer(models, mnist_data):
    predict_fn = lambda x: models[0].predict(x)  # noqa: E731
    image_shape = (28, 28, 1)
    segmentation_fn = "slic"
    segmentation_kwargs = {"n_segments": 10, "compactness": 10, "sigma": 0.5}

    explainer = AnchorImage(
        predict_fn,
        image_shape,
        segmentation_fn=segmentation_fn,
        segmentation_kwargs=segmentation_kwargs,
    )

    x_train = mnist_data["X_train"]
    image = x_train[0]
    threshold = 0.95

    before_explain = explainer.__dict__
    explainer.explain(image, threshold=threshold, n_covered_ex=3)
    after_explain = explainer.__dict__

    assert before_explain == after_explain
