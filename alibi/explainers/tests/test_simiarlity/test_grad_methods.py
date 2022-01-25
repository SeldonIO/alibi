import numpy as np
import pytest
from alibi.explainers.similarity.grad import SimilarityExplainer
from alibi.explainers.tests.test_simiarlity.conftest import get_flattened_model_parameters
import torch
import tensorflow as tf


@pytest.mark.parametrize('random_cls_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_grad_mat_grad_explainer_explanation(linear_cls_model, random_cls_dataset):
    """"""
    backend, linear_model, loss_fn = linear_cls_model
    params = get_flattened_model_parameters(linear_model)
    (x_train, y_train), (_, _) = random_cls_dataset

    y_train = y_train.astype(int)

    explainer = SimilarityExplainer(
        model=linear_model,
        loss_fn=loss_fn,
        sim_fn='grad_dot',
        store_grads=True,
        backend=backend
    )

    # test stored gradients
    explainer.fit(x_train=x_train, y_train=y_train)
    assert explainer.grad_x_train.shape == (len(x_train), *params.shape)

    # TODO: test correct explainer fields.
    result = explainer.explain(x_train[0:1])
    assert result.data['scores'].shape == (100, )
    assert result.data['x_train'].shape == (100, 10)
    assert result.data['y_train'].shape == (100, )


@pytest.mark.parametrize('random_cls_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True,
                         ids=['torch-model', 'tf-model']
                         )
def test_similarity_explainer_preprocessing(linear_cls_model, random_cls_dataset):
    """"""
    backend, linear_model, loss_fn = linear_cls_model
    (x_train, y_train), (_, _) = random_cls_dataset
    y_train = y_train.astype(int)

    explainer = SimilarityExplainer(
        model=linear_model,
        loss_fn=loss_fn,
        backend=backend,
    )

    # test stored gradients
    explainer.fit(x_train=x_train, y_train=y_train)
    x, y = explainer._preprocess_args(x_train[0:1])
    assert x.shape == (1, 10)
    assert y.shape == (1,)

    grad_x_test = explainer.backend.get_grads(linear_model, x, y, loss_fn)
    assert grad_x_test.shape == (115, )


@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_correct_sim_fn_error_messaging(linear_cls_model):
    """
    sim_fn is one of ['grad_dot', 'grad_cos', 'grad_asym_dot']
    """
    backend, linear_model, loss_fn = linear_cls_model

    # sim_fn is one of ['grad_dot', 'grad_cos']
    with pytest.raises(ValueError) as err:
        SimilarityExplainer(
            model=linear_model,
            loss_fn=loss_fn,
            sim_fn='not_grad_dot',
            store_grads=False,
            backend=backend
        )

    assert 'Unknown method not_grad_dot. Consider using: `grad_dot` | `grad_cos` | `grad_asym_dot`.' in str(err.value)

    for sim_fn in ['grad_dot', 'grad_cos', 'grad_asym_dot']:
        SimilarityExplainer(
            model=linear_model,
            loss_fn=loss_fn,
            sim_fn=sim_fn,
            store_grads=False,
            backend=backend
        )


@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_correct_task_error_messaging(linear_cls_model):
    """
    task is one of ['classification', 'regression']
    """
    backend, linear_model, loss_fn = linear_cls_model

    # sim_fn is one of ['grad_dot', 'grad_cos']
    with pytest.raises(ValueError) as err:
        SimilarityExplainer(
            model=linear_model,
            loss_fn=loss_fn,
            store_grads=False,
            backend=backend,
            task='not_classification'
        )

    assert 'Unknown task not_classification. Consider using: `classification` | `regression`.' in str(err.value)

    for task in ['classification', 'regression']:
        SimilarityExplainer(
            model=linear_model,
            loss_fn=loss_fn,
            store_grads=False,
            backend=backend,
            task=task
        )


# @pytest.mark.parametrize('random_reg_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
# @pytest.mark.parametrize('linear_reg_model',
#                          [
#                              ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
#                              ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
#                          ],
#                          indirect=True, ids=['torch-model', 'tf-model'])
# def test_regression_error_msgs(linear_reg_model, random_reg_dataset):
#     """
#     Test method applied to regression requires y value in explain method
#     """
#     backend, linear_model, loss_fn = linear_reg_model
#     (x_train, y_train), (_, _) = random_reg_dataset
#
#     print(x_train.dtype, y_train.dtype)
#
#     # this is incorrect task=classification case in the regression case with the wrong data set and model!
#     classification_explainer = SimilarityExplainer(
#         model=linear_model,
#         loss_fn=loss_fn,
#         sim_fn='grad_dot',
#         task='classification',
#         store_grads=False,
#         backend=backend
#     )
#
#     # classification similarity method y value can be none, a ndarray, or a function
#     classification_explainer.fit(x_train=x_train, y_train=y_train)
#     classification_explainer.explain(x_train[0:1])
#     # classification_explainer.explain(x_train[0:1], y_train[0:1])
#
#
# #     if backend == 'torch':
# #         classification_explainer.explain(x_train[0:1], lambda x: torch.argmax(linear_model(x)))
# #     else:
# #         classification_explainer.explain(x_train[0:1], lambda x: np.argmax(linear_model(x)))
# #
# #     regression_explainer = SimilarityExplainer(
# #         model=linear_model,
# #         loss_fn=loss_fn,
# #         sim_fn='grad_dot',
# #         task='regression',
# #         store_grads=False,
# #         backend=backend
# #     )
# #
# #     # classification similarity method y value cannot be none
# #     regression_explainer.fit(x_train=x_train, y_train=y_train)
# #     with pytest.raises(ValueError) as err:
# #         regression_explainer.explain(x_train[0:1])
# #     assert 'Regression task requires a target value.' in str(err.value)
# #
# #     regression_explainer.explain(x_train[0:1], y_train[0:1])
# #
# #
# # def test_same_class_grad_dot():
# #     """
# #     Test that the grad-dot similarity methods highest scoring training point is the same class as test point.
# #     """
# #
# #
# # def test_same_class_grad_cos():
# #     """
# #     Test that the grad-cos similarity methods highest scoring training point is the same class as test point.
# #     """
