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

def target_fn(X):
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
    for point in explanation['ordered_X_train'][1:]:
        current = np.dot(normed_ds[0], point)
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
    for point in explanation['ordered_X_train'][1:]:
        current = compute_angle(ds[0], point)
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
    assert (explanation['ordered_X_train'][1] == ds[1]).all()
    assert (explanation['ordered_X_train'][-1] == ds[-1]).all()


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
    assert (explanation['ordered_X_train'][0] == ds[-1]).all()
    assert (explanation['ordered_X_train'][-1] == ds[1]).all()


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
    explanation = explainer.explain(normed_ds[0][None], Y=normed_ds[0][None])
    last = np.dot(normed_ds[0], normed_ds[0])
    for point in explanation['ordered_X_train'][1:]:
        current = np.dot(normed_ds[0], point)
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
    explanation = explainer.explain(ds[0][None], Y=ds[0][None])
    last = compute_angle(ds[0], ds[0])
    for point in explanation['ordered_X_train'][1:]:
        current = compute_angle(ds[0], point)
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
    explanation = explainer.explain(ds[0, None], Y=ds[0, None])
    assert (explanation['ordered_X_train'][0] == ds[-1]).all()
    assert (explanation['ordered_X_train'][-1] == ds[1]).all()


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
    explanation = explainer.explain(ds[0, None], Y=ds[0, None])
    assert (explanation['ordered_X_train'][1] == ds[1]).all()
    assert (explanation['ordered_X_train'][-1] == ds[-1]).all()
