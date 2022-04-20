import pytest

import torch
import numpy as np

from alibi.explainers.similarity.backends.tensorflow.base import _TensorFlowBackend
from alibi.explainers.similarity.backends.pytorch.base import _PytorchBackend


@pytest.mark.parametrize('random_cls_dataset', [({'shape': 10, 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_models',
                         [({'input_shape': (10,), 'output_shape': 10})],
                         indirect=True)
def test_backends(random_cls_dataset, linear_models):
    """Test that the `tensorflow` and `torch` backends work as expected.

    This test creates a `tensorflow` model and a `torch` model and computes the gradients of each through the alibi
    backend modules respectively. The test passes if the gradients are the same.
    """
    tf_model, tf_loss, torch_model, torch_loss = linear_models

    # make the models duplicates of each other
    for w1, w2 in zip(torch_model.parameters(), tf_model.trainable_weights):
        w2.assign(w1.detach().numpy().T)

    (X_train, Y_train), (_, _) = random_cls_dataset
    X = _TensorFlowBackend.to_tensor(X_train)
    Y = _TensorFlowBackend.to_tensor(Y_train)
    tf_grads = _TensorFlowBackend.get_grads(tf_model, X, Y, tf_loss)
    params = np.concatenate([w.numpy().reshape(-1)
                             for w in tf_model.trainable_weights])[None]
    assert params.shape[-1] == tf_grads.shape[-1]

    X = _PytorchBackend.to_tensor(X_train)
    Y = _PytorchBackend.to_tensor(Y_train).type(torch.LongTensor)
    torch_grads = _PytorchBackend.get_grads(torch_model, X, Y, torch_loss)
    params = np.concatenate([param.detach().numpy().reshape(-1)
                             for param in torch_model.parameters()])[None]
    assert torch_grads.shape[-1] == params.shape[-1]

    torch_grads = np.sort(torch_grads)
    tf_grads = np.sort(tf_grads)
    np.testing.assert_allclose(torch_grads, tf_grads, rtol=1e-04)
