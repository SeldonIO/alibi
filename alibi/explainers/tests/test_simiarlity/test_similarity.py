import pytest
from alibi.explainers.similarity.base import GradMatrixGradExplainer
# import torch.nn as nn
from alibi.explainers.tests.test_simiarlity.conftest import get_flattened_model_parameters


@pytest.mark.parametrize('random_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_grad_mat_grad_explainer(linear_model, random_dataset):
    """
    Test that the Tensorflow and pytorch backends work as expected.
    """
    backend, linear_model, loss_fn = linear_model
    params = get_flattened_model_parameters(linear_model)
    (x_train, y_train), (_, _) = random_dataset

    y_train = y_train.astype(int)

    explainer = GradMatrixGradExplainer(
        model=linear_model,
        loss_fn=loss_fn,
        sim_fn='dot',
        store_grads=True,
        backend=backend
    )

    explainer.fit(x_train=x_train, y_train=y_train)

    assert explainer.grad_x_train.shape == (len(x_train), *params.shape)
