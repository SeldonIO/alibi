import pytest
from alibi.explainers.similarity.backends.tensorflow.base import TensorFlowBackend
from alibi.explainers.similarity.backends.pytorch.base import TorchBackend

import torch
import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
np.random.seed(0)
torch.manual_seed(0)


@pytest.mark.parametrize('random_cls_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_models',
                         [({'input_shape': (10,), 'output_shape': 10})],
                         indirect=True)
def test_tf_backend(random_cls_dataset, linear_models):
    """
    Test that the Tensorflow and pytorch backends work as expected.
    """
    tf_model, tf_loss, torch_model, torch_loss = linear_models

    # # make the models duplicates of each other
    for w1, w2 in zip(torch_model.parameters(), tf_model.trainable_weights):
        w2.assign(w1.detach().numpy().T)

    (X_train, Y_train), (_, _) = random_cls_dataset

    x = TensorFlowBackend.to_tensor(X_train)
    y = TensorFlowBackend.to_tensor(Y_train)
    tf_grads = TensorFlowBackend.get_grads(tf_model, x, y, tf_loss)
    params = np.concatenate([w.numpy().reshape(-1)
                             for w in tf_model.trainable_weights])[None]
    assert params.shape[-1] == tf_grads.shape[-1]

    x = TorchBackend.to_tensor(X_train)
    y = TorchBackend.to_tensor(Y_train).type(torch.LongTensor)
    torch_grads = TorchBackend.get_grads(torch_model, x, y, torch_loss)
    params = np.concatenate([param.detach().numpy().reshape(-1)
                             for param in torch_model.parameters()])[None]
    assert torch_grads.shape[-1] == params.shape[-1]

    torch_grads = np.sort(torch_grads)
    tf_grads = np.sort(tf_grads)
    np.testing.assert_allclose(torch_grads, tf_grads, rtol=1e-05)
