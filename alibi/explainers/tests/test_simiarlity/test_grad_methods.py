import numpy as np
import pytest
from alibi.explainers.similarity.grad import SimilarityExplainer
# import torch.nn as nn
from alibi.explainers.tests.test_simiarlity.conftest import get_flattened_model_parameters


@pytest.mark.parametrize('random_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_grad_mat_grad_explainer_explanation(linear_model, random_dataset):
    """"""
    backend, linear_model, loss_fn = linear_model
    params = get_flattened_model_parameters(linear_model)
    (x_train, y_train), (_, _) = random_dataset

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
    pass

@pytest.mark.parametrize('random_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_model',
                         [({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})],
                         indirect=True)
def test_correct_error_messaging(linear_model, random_dataset):
    """
    Test SimilarityExplainer throws correct errors on incorrect arguments.

        - sim_fn is one of ['grad_dot', 'grad_cos', 'grad_asym_dot']
        - task is one of ['classification', 'regression']
        - method applied to regression requires y value in explain method
    """
    backend, linear_model, loss_fn = linear_model
    (x_train, y_train), (_, _) = random_dataset

    y_train = y_train.astype(int)

    # sim_fn is one of ['grad_dot', 'grad_cos']
    with pytest.raises(ValueError) as err:
        SimilarityExplainer(
            model=linear_model,
            loss_fn=loss_fn,
            sim_fn='not_grad_dot',
            store_grads=False,
            backend=backend
        )
    assert f'Unknown method not_grad_dot. Consider using: `grad_dot` | `grad_cos` | `grad_asym_dot`.' in str(err.value)

    for sim_fn in ['grad_dot', 'grad_cos', 'grad_asym_dot']:
        SimilarityExplainer(
            model=linear_model,
            loss_fn=loss_fn,
            sim_fn=sim_fn,
            store_grads=False,
            backend=backend
        )

    # task is one of ['classification', 'regression']
    with pytest.raises(ValueError) as err:
        SimilarityExplainer(
            model=linear_model,
            loss_fn=loss_fn,
            sim_fn='grad_dot',
            task='not_classification',
            store_grads=False,
            backend=backend
        )
    assert f'Unknown task not_classification. Consider using: `classification` | `regression`.' in str(err.value)

    classification_explainer = SimilarityExplainer(
        model=linear_model,
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
    classification_explainer.explain(x_train[0:1], lambda x: np.argmax(linear_model(x)))

    regression_explainer = SimilarityExplainer(
        model=linear_model,
        loss_fn=loss_fn,
        sim_fn='grad_dot',
        task='regression',
        store_grads=False,
        backend=backend
    )

    # classification similarity method y value cannot be none
    regression_explainer.fit(x_train=x_train, y_train=y_train)
    with pytest.raises(ValueError) as err:
        regression_explainer.explain(x_train[0:1])
    assert f'Regression task requires a target value.' in str(err.value)

    regression_explainer.explain(x_train[0:1], y_train[0:1])


def test_same_class_grad_dot():
    """
    Test that grad dot similarity method highest scoring training point is the same class as test point.
    """


def test_same_class_grad_cos():
    """
    Test that grad cos similarity method highest scoring training point is the same class as test point.
    """
