from alibi.explainers.similarity.grad import SimilarityExplainer
from tensorflow import keras
import torch.nn as nn
import pytest

import torch
import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
np.random.seed(0)
torch.manual_seed(0)


def loss_torch(x, y):
    return x


def target_fn(x):
    return x


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


def test_correct_grad_dot_sim_result_torch(normed_ds):
    """"""
    model = nn.Linear(2, 1, bias=False)
    explainer = SimilarityExplainer(
        model,
        task='regression',
        loss_fn=loss_torch,
        sim_fn='grad_dot',
        backend='torch'
    )
    explainer = explainer.fit(normed_ds, normed_ds)
    explanation = explainer.explain(normed_ds[0], y=target_fn)
    last = np.dot(normed_ds[0], normed_ds[0])
    for point in explanation['x_train'][1:]:
        current = np.dot(normed_ds[0], point)
        assert current <= last
        last = current


def test_correct_grad_cos_sim_result_torch(ds):
    """"""
    model = nn.Linear(2, 1, bias=False)
    explainer = SimilarityExplainer(
        model,
        task='regression',
        loss_fn=loss_torch,
        sim_fn='grad_cos',
        backend='torch'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], y=target_fn)
    last = 0
    for point in explanation['x_train'][1:]:
        current = compute_angle(ds[0], point)
        assert current >= last
        last = current


def test_grad_cos_result_order_torch():
    """"""
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = nn.Linear(2, 1, bias=False)
    explainer = SimilarityExplainer(
        model,
        task='regression',
        loss_fn=loss_torch,
        sim_fn='grad_cos',
        backend='torch',
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], y=target_fn)
    assert (explanation['x_train'][1] == ds[1]).all()
    assert (explanation['x_train'][-1] == ds[-1]).all()


def test_grad_dot_result_order_torch():
    """"""
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = nn.Linear(2, 1, bias=False)
    explainer = SimilarityExplainer(
        model,
        task='regression',
        loss_fn=loss_torch,
        sim_fn='grad_dot',
        backend='torch'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0], y=target_fn)
    assert (explanation['x_train'][0] == ds[-1]).all()
    assert (explanation['x_train'][-1] == ds[1]).all()


def loss_tf(y, x):
    return x


def test_correct_grad_dot_sim_result_tf(normed_ds):
    """"""
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    explainer = SimilarityExplainer(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_dot',
        backend='tensorflow'
    )
    explainer = explainer.fit(normed_ds, normed_ds)
    explanation = explainer.explain(normed_ds[0][None], y=target_fn)
    last = np.dot(normed_ds[0], normed_ds[0])
    for point in explanation['x_train'][1:]:
        current = np.dot(normed_ds[0], point)
        assert current <= last
        last = current


def test_correct_grad_cos_sim_result_tf(ds):
    """"""
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    explainer = SimilarityExplainer(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_cos',
        backend='tensorflow'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0][None], y=target_fn)
    last = compute_angle(ds[0], ds[0])
    for point in explanation['x_train'][1:]:
        current = compute_angle(ds[0], point)
        assert current >= last
        last = current


def test_grad_dot_result_order_tf():
    """"""
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    explainer = SimilarityExplainer(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_dot',
        backend='tensorflow'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0, None], y=target_fn)
    assert (explanation['x_train'][0] == ds[-1]).all()
    assert (explanation['x_train'][-1] == ds[1]).all()


def test_grad_cos_result_order_tf():
    """"""
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    explainer = SimilarityExplainer(
        model,
        task='regression',
        loss_fn=loss_tf,
        sim_fn='grad_cos',
        backend='tensorflow'
    )
    explainer = explainer.fit(ds, ds)
    explanation = explainer.explain(ds[0, None], y=target_fn)
    assert (explanation['x_train'][1] == ds[1]).all()
    assert (explanation['x_train'][-1] == ds[-1]).all()