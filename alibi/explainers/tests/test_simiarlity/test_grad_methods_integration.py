import numpy as np
from alibi.explainers.similarity.grad import SimilarityExplainer
import torch.nn as nn


def test_same_class_grad_dot_torch():
    """"""
    np.random.seed(seed=0)
    ds = np.random.normal(size=(100, 2)).astype('float32')
    ds = ds/np.linalg.norm(ds, axis=1)[:, None]
    model = nn.Linear(2, 1, bias=False)
    loss = lambda x, y: x
    explainer = SimilarityExplainer(model, task='regression', loss_fn=loss, sim_fn='grad_dot', backend='torch')
    explainer = explainer.fit(ds, ds)
    target_fn = lambda x: x
    explanation = explainer.explain(ds[0], y=target_fn)
    close_score = np.dot(ds[0], explanation['x_train'][1])
    np.testing.assert_almost_equal(close_score, 1, decimal=2)
    far_score = np.dot(ds[0], explanation['x_train'][-1])
    np.testing.assert_almost_equal(far_score, -1, decimal=2)


def test_same_class_grad_cos():
    """"""
    np.random.seed(seed=0)
    ds = np.array([[1, 0], [0.9, 0.1], [0.5 * 100, 0.5 * 100]]).astype('float32')
    model = nn.Linear(2, 1, bias=False)
    loss = lambda x, y: x
    explainer = SimilarityExplainer(model, task='regression', loss_fn=loss, sim_fn='grad_cos', backend='torch')
    explainer = explainer.fit(ds, ds)
    target_fn = lambda x: x
    explanation = explainer.explain(ds[0], y=target_fn)
    assert (explanation['x_train'][1] == ds[1]).all()
    assert (explanation['x_train'][-1] == ds[-1]).all()
