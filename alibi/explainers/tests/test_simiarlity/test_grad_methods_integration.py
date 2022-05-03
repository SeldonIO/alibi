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
    assert (ds[explanation['ordered_indices'][0][1]] == ds[1]).all()
    assert (ds[explanation['ordered_indices'][0][-1]] == ds[-1]).all()


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
    assert (ds[explanation['ordered_indices'][0][0]] == ds[-1]).all()
    assert (ds[explanation['ordered_indices'][0][-1]] == ds[1]).all()


def loss_tf(y, x):
    return x


def test_correct_grad_dot_sim_result_tf(seed, normed_ds):
    """
    `grad_dot` method orders data points distributed on the unit circle by their angular separation. Test is applied to
    `tensorflow` backend.
    """
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
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
    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_dot',
        backend='tensorflow'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], Y=ds[0])
    assert (ds[explanation['ordered_indices'][0][0]] == ds[-1]).all()
    assert (ds[explanation['ordered_indices'][0][-1]] == ds[1]).all()


def test_grad_cos_result_order_tf(seed):
    """
    `grad_cos` finds data points with small angular separation to be more similar independent of length. Test is
    applied to `tensorflow` backend.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    explainer = GradientSimilarity(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_cos',
        backend='tensorflow'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], Y=ds[0])
    assert (ds[explanation['ordered_indices'][0][1]] == ds[1]).all()
    assert (ds[explanation['ordered_indices'][0][-1]] == ds[-1]).all()


@pytest.mark.parametrize('precompute_grads', [True, False])
def test_multiple_test_instances_grad_cos(precompute_grads):
    """
    Test that multiple test instances get correct explanations for `grad_cos` similarity.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
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
    assert (ds[explanation['ordered_indices'][0][1]] == ds[1]).all()
    assert (ds[explanation['ordered_indices'][1][1]] == ds[0]).all()

    # Test that the greatest difference is between the first two and the last datapoint
    assert (ds[explanation['ordered_indices'][0][-1]] == ds[-1]).all()
    assert (ds[explanation['ordered_indices'][1][-1]] == ds[-1]).all()


@pytest.mark.parametrize('precompute_grads', [True, False])
def test_multiple_test_instances_grad_dot(precompute_grads):
    """
    Test that multiple test instances get correct explanations for `grad_dot` similarity.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
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
    assert (ds[explanation['ordered_indices'][0][0]] == ds[-1]).all()
    assert (ds[explanation['ordered_indices'][1][0]] == ds[-1]).all()


@pytest.mark.parametrize('precompute_grads', [True, False])
def test_multiple_test_instances_stored_grads_asym_dot(precompute_grads):
    """
    Test that multiple test instances get correct explanations for `grad_asym_dot`.
    """
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
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
    sim_ds_0 = np.array([(1 * 1 + 0 * 0) / 1., (1 * 0.9 + 0 * 0.1) / 1., (1 * 50 + 0 * 50) / 1.])
    sim_ds_0.sort()
    np.testing.assert_almost_equal(explanation.scores[0], sim_ds_0[::-1], decimal=3)

    d = (0.9**2 + 0.1**2)
    sim_ds_1 = np.array([(0.9 * 1 + 0.1 * 0) / d, (0.9 * 0.9 + 0.1 * 0.1) / d, (0.9 * 50 + 0.1 * 50) / d])
    sim_ds_1.sort()
    np.testing.assert_almost_equal(explanation.scores[1], sim_ds_1[::-1], decimal=3)

    assert (ds[explanation['ordered_indices'][0][0]] == ds[-1]).all()
    assert (ds[explanation['ordered_indices'][1][0]] == ds[-1]).all()
    explanation = explainer.explain(ds[-1], Y=ds[-1])
    assert (explanation.scores == np.array([[1., 0.01, 0.01]], dtype=np.float32)).all()
