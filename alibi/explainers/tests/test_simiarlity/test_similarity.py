import pytest
from alibi.explainers.similarity.base import GradMatrixGradExplainer
import torch.nn as nn
from alibi.explainers.tests.test_simiarlity.conftest import get_flattened_model_parameters


@pytest.mark.parametrize('backend', ['pytorch'])
@pytest.mark.parametrize('random_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('torch_linear_model', [({'input_shape': (10,), 'output_shape': 10})], indirect=True)
def test_grad_mat_grad_explainer(backend, torch_linear_model, random_dataset):
    """
    Test that the Tensorflow and pytorch backends work as expected.
    """
    params = get_flattened_model_parameters(torch_linear_model)
    (x_train, y_train), (_, _) = random_dataset

    y_train = y_train.astype(int)

    explainer = GradMatrixGradExplainer(
        model=torch_linear_model,
        loss_fn=nn.CrossEntropyLoss(),
        sim_fn='dot',
        store_grads=True,
        backend=backend
    )

    explainer.fit(x_train=x_train, y_train=y_train)

    assert explainer.grad_x_train.shape == (len(x_train), *params.shape)
