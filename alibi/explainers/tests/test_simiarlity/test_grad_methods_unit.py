import pytest
from alibi.explainers.similarity.grad import SimilarityExplainer
from alibi.explainers.tests.test_simiarlity.conftest import get_flattened_model_parameters

import torch
import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
np.random.seed(0)
torch.manual_seed(0)


@pytest.mark.parametrize('random_cls_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'torch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_method_explanations(linear_cls_model, random_cls_dataset):
    """"""
    backend, model, loss_fn, target_fn = linear_cls_model
    params = get_flattened_model_parameters(model)
    (x_train, y_train), (_, _) = random_cls_dataset

    y_train = y_train.astype(int)

    explainer = SimilarityExplainer(
        model=model,
        loss_fn=loss_fn,
        sim_fn='grad_dot',
        store_grads=True,
        backend=backend
    )

    # test stored gradients
    explainer.fit(x_train=x_train, y_train=y_train)
    assert explainer.grad_x_train.shape == (len(x_train), *params.shape)

    # print('x_train.shape:', x_train.shape, model(x_train))
    result = explainer.explain(x_train)
    assert result.data['scores'].shape == (100, )
    assert result.data['x_train'].shape == (100, 10)
    assert result.data['y_train'].shape == (100, )


@pytest.mark.parametrize('random_cls_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'torch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True,
                         ids=['torch-model', 'tf-model']
                         )
def test_explainer_method_preprocessing(linear_cls_model, random_cls_dataset):
    """"""
    backend, model, loss_fn, target_fn = linear_cls_model
    (x_train, y_train), (_, _) = random_cls_dataset
    y_train = y_train.astype(int)

    explainer = SimilarityExplainer(
        model=model,
        loss_fn=loss_fn,
        backend=backend,
    )

    # test stored gradients
    explainer.fit(x_train=x_train, y_train=y_train)
    x, y = explainer._preprocess_args(x_train[0:3])
    assert x.shape == (3, 10)
    assert y.shape == (3,)

    x, y = explainer._preprocess_args(x_train[0])
    assert x.shape == (1, 10)

    x, y = explainer._preprocess_args(x_train[0])
    assert x.shape == (1, 10)

    grad_x_test = explainer.backend.get_grads(model, x, y, loss_fn)
    assert grad_x_test.shape == (110, )


@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'torch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_method_sim_fn_error_messaging(linear_cls_model):
    """
    sim_fn is one of ['grad_dot', 'grad_cos', 'grad_asym_dot']
    """
    backend, model, loss_fn, target_fn = linear_cls_model

    # sim_fn is one of ['grad_dot', 'grad_cos']
    with pytest.raises(ValueError) as err:
        SimilarityExplainer(
            model=model,
            loss_fn=loss_fn,
            sim_fn='not_grad_dot',
            store_grads=False,
            backend=backend
        )

    assert 'Unknown method not_grad_dot. Consider using: `grad_dot` | `grad_cos` | `grad_asym_dot`.' in str(err.value)

    for sim_fn in ['grad_dot', 'grad_cos', 'grad_asym_dot']:
        SimilarityExplainer(
            model=model,
            loss_fn=loss_fn,
            sim_fn=sim_fn,
            store_grads=False,
            backend=backend
        )


@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'torch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_method_task_error_messaging(linear_cls_model):
    """
    task is one of ['classification', 'regression']
    """
    backend, model, loss_fn, target_fn = linear_cls_model

    # sim_fn is one of ['grad_dot', 'grad_cos']
    with pytest.raises(ValueError) as err:
        SimilarityExplainer(
            model=model,
            loss_fn=loss_fn,
            store_grads=False,
            backend=backend,
            task='not_classification'
        )

    assert 'Unknown task not_classification. Consider using: `classification` | `regression`.' in str(err.value)

    for task in ['classification', 'regression']:
        SimilarityExplainer(
            model=model,
            loss_fn=loss_fn,
            store_grads=False,
            backend=backend,
            task=task
        )


@pytest.mark.parametrize('random_cls_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'torch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_task_classification_input(random_cls_dataset, linear_cls_model):
    """
    task is one of ['classification', 'regression']
    """
    backend, model, loss_fn, target_fn = linear_cls_model
    (x_train, y_train), (_, _) = random_cls_dataset
    y_train = y_train.astype(int)

    classification_explainer = SimilarityExplainer(
        model=model,
        loss_fn=loss_fn,
        sim_fn='grad_dot',
        task='classification',
        store_grads=False,
        backend=backend
    )

    # classification similarity method y value can be none, a ndarray, or a function
    classification_explainer.fit(x_train=x_train, y_train=y_train)
    classification_explainer.explain(x_train[0:1])
    classification_explainer.explain(x_train[0:1], y_train[0:1])
    classification_explainer.explain(x_train[0:1], target_fn)


@pytest.mark.parametrize('random_reg_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_reg_model',
                         [
                             ({'framework': 'torch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True,
                         ids=['torch-model', 'tf-model']
                         )
def test_regression_task_input(linear_reg_model, random_reg_dataset):
    """
    Test method applied to regression requires y value in explain method
    """
    backend, model, loss_fn, target_fn = linear_reg_model
    (x_train, y_train), (_, _) = random_reg_dataset

    regression_explainer = SimilarityExplainer(
        model=model,
        loss_fn=loss_fn,
        sim_fn='grad_dot',
        task='regression',
        store_grads=False,
        backend=backend
    )

    # classification similarity method y value cannot be none
    regression_explainer.fit(x_train=x_train, y_train=y_train)
    with pytest.raises(ValueError) as err:
        regression_explainer.explain(x_train[0])
    assert 'Regression task requires a target value.' in str(err.value)

    regression_explainer.explain(x_train[0], y_train)
