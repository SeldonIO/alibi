import numpy as np
import pytest
from alibi.explainers.similarity.backends.tensorflow import base as tf_backend
from alibi.explainers.similarity.backends.pytorch import base as torch_backend
import tensorflow as tf

import torch.nn as nn
import torch


@pytest.mark.parametrize('random_dataset', [({'shape': (10,), 'size': 100})], indirect=True)
@pytest.mark.parametrize('tf_linear_model', [({'input_shape': (10,), 'output_shape': 10})], indirect=True)
@pytest.mark.parametrize('torch_linear_model', [({'input_shape': (10,), 'output_shape': 10})], indirect=True)
def test_tf_backend(torch_linear_model, tf_linear_model, random_dataset):
    """
    Test that the Tensorflow and pytorch backends work as expected.
    """
    # make the models duplicates of each other
    for w1, w2 in zip(torch_linear_model.parameters(), tf_linear_model.trainable_weights):
        w2.assign(w1.detach().numpy().T)

    (X_train, Y_train), (_, _) = random_dataset

    np.testing.assert_allclose(torch_linear_model(torch_backend.to_tensor(X_train[:1])).detach().numpy(),
                               tf_linear_model(tf_backend.to_tensor(X_train[:1])).numpy())

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    x = tf_backend.to_tensor(X_train[0:1])
    y = tf_backend.to_tensor(Y_train[0:1])
    tf_grads = tf_backend.get_grads(tf_linear_model, x, y, loss_fn)
    params = np.concatenate([w.numpy().reshape(-1) for w in tf_linear_model.trainable_weights])
    assert params.shape == tf_grads.shape

    loss_fn = nn.CrossEntropyLoss()
    x = torch_backend.to_tensor(X_train[0:1])
    y = torch_backend.to_tensor(Y_train[0:1]).type(torch.LongTensor)
    torch_grads = torch_backend.get_grads(torch_linear_model, x, y, loss_fn)
    params = np.concatenate([param.detach().numpy().reshape(-1)
                             for param in torch_linear_model.parameters()])
    assert torch_grads.shape == params.shape

    np.testing.assert_allclose(torch_grads, tf_grads)



