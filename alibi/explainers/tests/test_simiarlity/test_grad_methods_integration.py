"""Integration tests for gradient methods.

Note:
    The following tests use sequential models with a single linear layer of two nodes and no bias. Similarly, the loss
    is defined as the model output. This means that the gradient w.r.t. the model parameters is just the model input.
    We can then choose the dataset in such a way to give predictable results for the similarity score for both
    `grad_cos` and `grad_dot`. For instance, we can choose the dataset to be made up of three vectors. Two that
    are close in length and angular separation and one that is very long but also further away in angular separation.
    This dataset behaves differently for each method. In the case of `grad_cos` the similarity score is larger for
    the two data points that have less angular separation. In the case of `grad_dot` the larger vectors length
    dominates.
"""

import pytest

import numpy as np
import torch.nn as nn
import torch
import tensorflow as tf
from tensorflow import keras

from alibi.explainers.similarity.grad import GradientSimilarity


def loss_torch(X, Y):
    return X


def compute_angle(a, b):
    inner = np.inner(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    cos = inner / norms
    rads = np.arccos(np.clip(cos, -1.0, 1.0))
    return np.rad2deg(rads)


@pytest.fixture(scope='module')
def ds():
    np.random.seed(seed=0)
    return np.random.normal(size=(100, 2)).astype('float32')


@pytest.fixture(scope='module')
def normed_ds():
    np.random.seed(seed=0)
    ds = np.random.normal(size=(100, 2)).astype('float32')
    return ds / np.linalg.norm(ds, axis=1, keepdims=True)


def test_correct_grad_dot_sim_result_torch(seed, normed_ds):
    """
    `grad_dot` method orders data points distributed on the unit circle by their angular separation. Test is applied to
    `torch` backend.
    """
    model = nn.Linear(2, 1, bias=False)
    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_torch,
        sim_fn='grad_dot',
        backend='pytorch'
    )
    explainer = explainer.fit(normed_ds, normed_ds)
    explanation = explainer.explain(normed_ds[0], Y=normed_ds[0])
    last = np.dot(normed_ds[0], normed_ds[0])
    for ind in explanation['ordered_indices'][0][1:]:
        current = np.dot(normed_ds[0], normed_ds[ind])
        assert current <= last
        last = current


def test_correct_grad_cos_sim_result_torch(seed, ds):
    """
    `grad_cos` method orders normally distributed data points by their angular separation. Test is applied to `torch`
    backend.
    """
    model = nn.Linear(2, 1, bias=False)
    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_torch,
        sim_fn='grad_cos',
        backend='pytorch'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], Y=ds[0])
    last = 0
    for ind in explanation['ordered_indices'][0][1:]:
        current = compute_angle(ds[0], ds[ind])
        assert current >= last
        last = current


def test_grad_cos_result_order_torch(seed):
    """
    `grad_cos` finds data points with small angular separation to be more similar independent of length. Test is
    applied to `torch` backend.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = nn.Linear(2, 1, bias=False)
    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_torch,
        sim_fn='grad_cos',
        backend='pytorch',
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], Y=ds[0])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][1]], ds[1])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][-1]], ds[-1])


def test_grad_dot_result_order_torch(seed):
    """
    Size of datapoint overrides angular closeness for `grad_dot` similarity. Test is applied to `torch` backend.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = nn.Linear(2, 1, bias=False)
    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_torch,
        sim_fn='grad_dot',
        backend='pytorch'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], Y=ds[0])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][0]], ds[-1])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][-1]], ds[1])


def loss_tf(y, x):
    return x


def test_correct_grad_dot_sim_result_tf(seed, normed_ds):
    """
    `grad_dot` method orders data points distributed on the unit circle by their angular separation. Test is applied to
    `tensorflow` backend.
    """
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])

    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 2))

    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_dot',
        backend='tensorflow'
    )
    explainer = explainer.fit(normed_ds, normed_ds)
    explanation = explainer.explain(normed_ds[0], Y=normed_ds[0])
    last = np.dot(normed_ds[0], normed_ds[0])
    for ind in explanation['ordered_indices'][0][1:]:
        current = np.dot(normed_ds[0], normed_ds[ind])
        assert current <= last
        last = current


def test_correct_grad_cos_sim_result_tf(seed, ds):
    """
    `grad_cos` method orders normally distributed data points by their angular separation. Test is applied to
    `tensorflow` backend.
    """
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 2))

    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_cos',
        backend='tensorflow'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], Y=ds[0])
    last = compute_angle(ds[0], ds[0])
    for ind in explanation['ordered_indices'][0][1:]:
        current = compute_angle(ds[0], ds[ind])
        assert current >= last
        last = current


def test_grad_dot_result_order_tf(seed):
    """
    Size of datapoint overrides angular closeness for `grad_dot` similarity. Test is applied to `torch` backend.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 2))

    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_dot',
        backend='tensorflow'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], Y=ds[0])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][0]], ds[-1])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][-1]], ds[1])


def test_grad_cos_result_order_tf(seed):
    """
    `grad_cos` finds data points with small angular separation to be more similar independent of length. Test is
    applied to `tensorflow` backend.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 2))

    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_cos',
        backend='tensorflow'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], Y=ds[0])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][1]], ds[1])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][-1]], ds[-1])


@pytest.mark.parametrize('precompute_grads', [True, False])
def test_multiple_test_instances_grad_cos(precompute_grads):
    """
    Test that multiple test instances get correct explanations for `grad_cos` similarity.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 2))

    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_cos',
        backend='tensorflow',
        precompute_grads=precompute_grads
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0:2], Y=ds[0:2])
    # Test that the first two datapoints are the most similar
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][1]], ds[1])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][1][1]], ds[0])

    # Test that the greatest difference is between the first two and the last datapoint
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][-1]], ds[-1])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][1][-1]], ds[-1])


@pytest.mark.parametrize('precompute_grads', [True, False])
def test_multiple_test_instances_grad_dot(precompute_grads):
    """
    Test that multiple test instances get correct explanations for `grad_dot` similarity.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 2))

    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_dot',
        backend='tensorflow',
        precompute_grads=precompute_grads
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0:2], Y=ds[0:2])

    # Check that the last datapoint is the most similar to the first datapoint.
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][0]], ds[-1])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][1][0]], ds[-1])


@pytest.mark.parametrize('precompute_grads', [True, False])
def test_multiple_test_instances_stored_grads_asym_dot(precompute_grads):
    """
    Test that multiple test instances get correct explanations for `grad_asym_dot`.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 2))

    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_asym_dot',
        backend='tensorflow',
        precompute_grads=precompute_grads
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0:2], Y=ds[0:2])

    # In the case of `grad_asym_dot` we manually check the similarities as the asymmetry of the metric means ordering
    # is important.
    denoms = np.array([1, (0.9**2 + 0.1**2), 2 * 50**2])
    sim_ds_0 = np.array([
        (1 * 1 + 0 * 0),
        (1 * 0.9 + 0 * 0.1),
        (1 * 50 + 0 * 50)
    ]) / (denoms + 1e-7)
    sim_ds_0.sort()

    np.testing.assert_allclose(explanation.scores[0], sim_ds_0[::-1], atol=1e-6)
    sim_ds_1 = np.array([(0.9 * 1 + 0.1 * 0), (0.9 * 0.9 + 0.1 * 0.1), (0.9 * 50 + 0.1 * 50)]) / (denoms + 1e-7)
    sim_ds_1.sort()

    np.testing.assert_allclose(explanation.scores[1], sim_ds_1[::-1], atol=1e-6)
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][0][0]], ds[1])
    np.testing.assert_array_equal(ds[explanation['ordered_indices'][1][0]], ds[1])
    explanation = explainer.explain(ds[-1], Y=ds[-1])
    scores = np.array([[50, 50, 2*50**2]], dtype=np.float32) / (denoms + 1e-7)
    scores.sort()
    np.testing.assert_allclose(explanation.scores, scores[:, ::-1], atol=1e-4)


def test_non_trainable_layer_warnings_tf():
    """Test non-trainable layer warnings `tensorflow`.

    Test that warnings are raised when user passes non-trainable layers to `GradientSimilarity` method for
    `tensorflow` models.

    Note: `Keras` batch norm layers register non-trainable weights by default and so will raise the
    warning we test for here. This is different to the `pytorch` behavour which doesn't include
    the batch norm parameters in ``model.parameters()``.
    """
    model = keras.Sequential([
        keras.layers.Dense(10),
        keras.layers.Dense(20),
        keras.layers.BatchNormalization(),
        keras.Sequential([
            keras.layers.Dense(30),
            keras.layers.Dense(40),
        ])
    ])

    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 10))

    model.layers[1].trainable = False
    model.layers[-1].layers[1].trainable = False
    num_params_non_trainable = len(model.non_trainable_weights)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    with pytest.warns(Warning) as record:
        GradientSimilarity(
            model,
            task='classification',
            loss_fn=loss_fn,
            backend='tensorflow',
        )

    assert (f"Found {num_params_non_trainable} non-trainable parameters in the model. These parameters "
            "don't have gradients and will not be included in the computation of gradient similarity."
            " This might be because your model has layers that track statistics using non-trainable "
            "parameters such as batch normalization layers. In this case, you don't need to worry. "
            "Otherwise it's because you have set some parameters to be non-trainable and alibi is "
            "letting you know.") == str(record[0].message)


def test_non_trainable_layer_warnings_pt():
    """Test non-trainable layer warnings `pytorch`.

    Test that warnings are raised when user passes non-trainbable layers to to `GradientSimilarity` method for
    `pytorch` models.
    """

    class Model(nn.Module):
        def __init__(self, input_shape, output_shape):
            super(Model, self).__init__()
            self.linear_stack = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_shape, output_shape),
                nn.Linear(output_shape, output_shape),
                nn.BatchNorm1d(output_shape),
                nn.Sequential(
                    nn.Linear(output_shape, output_shape),
                    nn.Linear(output_shape, output_shape)
                ),
                nn.Softmax()
            )

        def forward(self, x):
            x = x.type(torch.FloatTensor)
            return self.linear_stack(x)

    model = Model(10, 20)
    model.linear_stack[2].weight.requires_grad = False
    model.linear_stack[4][1].weight.requires_grad = False
    num_params_non_trainable = len([param for param in model.parameters() if not param.requires_grad])

    loss_fn = nn.CrossEntropyLoss()

    with pytest.warns(Warning) as record:
        GradientSimilarity(
            model,
            task='classification',
            loss_fn=loss_fn,
            backend='pytorch'
        )

    assert (f"Found {num_params_non_trainable} non-trainable parameters in the model. These parameters "
            "don't have gradients and will not be included in the computation of gradient similarity."
            " This might be because your model has layers that track statistics using non-trainable "
            "parameters such as batch normalization layers. In this case, you don't need to worry. "
            "Otherwise it's because you have set some parameters to be non-trainable and alibi is "
            "letting you know.") == str(record[0].message)


def test_not_trainable_model_error_tf():
    """Test non-trainable model error `tensorflow`."""

    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    # GradientSimilarity method checks weights are trainable so we need to build the model before passing it to the
    # method
    model.build((None, 2))
    model.trainable = False
    with pytest.raises(ValueError) as err:
        GradientSimilarity(
            model,
            task='regression',
            loss_fn=loss_tf,
            sim_fn='grad_dot',
            backend='tensorflow'
        )
    assert err.value.args[0] == ('The model has no trainable weights. This method requires at least '
                                 'one trainable parameter to compute the gradients for. '
                                 'Set ``trainable=True`` on the model or a model weight.')


def test_not_trainable_model_error_torch():
    """Test non-trainable model error `pytorch`."""

    model = nn.Linear(2, 1, bias=False)
    model.requires_grad_(False)

    with pytest.raises(ValueError) as err:
        GradientSimilarity(
            model,
            task='regression',
            loss_fn=loss_tf,
            sim_fn='grad_dot',
            backend='pytorch'
        )

    assert err.value.args[0] == ('The model has no trainable parameters. This method requires at least '
                                 'one trainable parameter to compute the gradients for. '
                                 "Try setting ``.requires_grad_(True)`` on the model or one of its parameters.")
