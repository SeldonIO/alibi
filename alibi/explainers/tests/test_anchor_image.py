import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np
import tensorflow as tf
import torch
from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR_IMG
from alibi.exceptions import AlibiPredictorCallException, AlibiPredictorReturnTypeError
from alibi.explainers.anchor_image import AnchorImage, AnchorImageSampler, scale_image


def test_scale_image():
    image_shape = (28, 28, 1)
    scaling_offset = 260
    min_val = 0
    max_val = 255

    fake_img = np.random.random(size=image_shape) + scaling_offset
    scaled_img = scale_image(fake_img, scale=(min_val, max_val))
    assert (scaled_img <= max_val).all()
    assert (scaled_img >= min_val).all()


@pytest.fixture
def predict_fn(request):
    """
    This fixture takes in a white-box model (Tensorflow or Pytorch) and returns an
    AnchorImage compatible prediction function.
    """
    if isinstance(request.param[0], tf.keras.Model):
        func = request.param[0].predict
    elif isinstance(request.param[0], torch.nn.Module):
        def func(image: np.ndarray) -> np.ndarray:
            # moveaxis is needed as torch uses 'bchw' layout instead of 'bhwc'
            # NB: torch models need dtype=torch.float32, we are not setting it here
            # to test that `dtype` argument to AnchorImage does the right thing when
            # a dummy call is made
            image = torch.as_tensor(np.moveaxis(image, -1, 1))  # type: ignore
            return request.param[0].forward(image).detach().numpy()
    else:
        raise ValueError(f'Unknown model {request.param[0]} of type {type(request.param[0])}')
    return func


@pytest.mark.parametrize('predict_fn', [lazy_fixture('models'), ], indirect=True)
@pytest.mark.parametrize('models',
                         [("mnist-cnn-tf2.2.0",), ("mnist-cnn-tf1.15.2.h5",), ("mnist-cnn-pt1.9.1.pt",)],
                         indirect=True,
                         ids='models={}'.format
                         )
def test_sampler(predict_fn, models, mnist_data):
    eps = 0.0001  # tolerance for tensor comparisons
    num_samples = 10

    x_train = mnist_data["X_train"]
    segmentation_fn = "slic"
    segmentation_kwargs = {"n_segments": 10, "compactness": 10, "sigma": 0.5}
    image_shape = (28, 28, 1)
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


@pytest.mark.parametrize('predict_fn', [lazy_fixture('models'), ], indirect=True)
@pytest.mark.parametrize('models',
                         [("mnist-cnn-tf2.2.0",), ("mnist-cnn-tf1.15.2.h5",), ("mnist-cnn-pt1.9.1.pt",)],
                         indirect=True,
                         ids='models={}'.format
                         )
@pytest.mark.parametrize('images_background', [True, False], ids='images_background={}'.format)
def test_anchor_image(predict_fn, models, mnist_data, images_background):
    x_train = mnist_data["X_train"]
    image = x_train[0]

    segmentation_fn = "slic"
    segmentation_kwargs = {"n_segments": 10, "compactness": 10, "sigma": 0.5}
    image_shape = (28, 28, 1)
    images_background = x_train[:10] if images_background else None

    explainer = AnchorImage(
        predict_fn,
        image_shape,
        segmentation_fn=segmentation_fn,
        segmentation_kwargs=segmentation_kwargs,
        images_background=images_background
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
    # dtype=np.float32 should be safe here (default behaviour when calling _transform_predictor
    assert explainer.predictor(np.zeros((1,) + image_shape, dtype=np.float32)).shape == (1,)
    assert explainer.custom_segmentation is False

    # test explanation
    threshold = 0.95

    before_explain = explainer.__dict__
    explanation = explainer.explain(image, threshold=threshold, n_covered_ex=3)
    after_explain = explainer.__dict__

    # Ensure that explainer's internal state doesn't change
    assert before_explain == after_explain
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


@pytest.mark.parametrize('predict_fn', [lazy_fixture('models'), ], indirect=True)
@pytest.mark.parametrize('models', [("mnist-cnn-pt1.9.1.pt",)], indirect=True)
def test_anchor_image_fails_init_torch_float64(predict_fn, models):
    with pytest.raises(AlibiPredictorCallException):
        explainer = AnchorImage(predict_fn, image_shape=(28, 28, 1), dtype=np.float64)  # noqa: F841


def bad_predictor(x: np.ndarray) -> list:
    """
    A dummy predictor emulating the following:
     - Expecting an array of certain dimension (4 dimensions - 1 batch, 2 spatial, 1 channel)
     - Returning an incorrect type
     This is used below to test custom exception functionality.
    """
    if x.ndim != 4:
        raise ValueError
    return list(x)


def test_anchor_image_fails_init_bad_image_shape_predictor_call():
    """
    In this test `image_shape` is misspecified leading to an exception calling the `predictor`.
    """
    with pytest.raises(AlibiPredictorCallException):
        explainer = AnchorImage(bad_predictor, image_shape=(28, 28))  # noqa: F841


def test_anchor_image_fails_bad_predictor_return_type():
    """
    In this test `image_shape` is specified correctly, but the predictor returns the wrong type.
    """
    with pytest.raises(AlibiPredictorReturnTypeError):
        explainer = AnchorImage(bad_predictor, image_shape=(28, 28, 1))  # noqa: F841
