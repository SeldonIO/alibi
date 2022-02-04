import numpy as np
import pytest
from alibi.explainers.similarity.backends.tensorflow import base as tf_backend
from alibi.explainers.similarity.backends.pytorch import base as torch_backend
import torch


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

    # np.testing.assert_allclose(np.sort(torch_model(torch_backend.to_tensor(X_train[:1])).detach().numpy()),
    #                            np.sort(tf_model(tf_backend.to_tensor(X_train[:1])).numpy()))

    x = tf_backend.to_tensor(X_train[0:1])
    y = tf_backend.to_tensor(Y_train[0:1])
    tf_grads = tf_backend.get_grads(tf_model, x, y, tf_loss)
    params = np.concatenate([w.numpy().reshape(-1) for w in tf_model.trainable_weights])[None]
    assert params.shape[-1] == tf_grads.shape[-1]

    x = torch_backend.to_tensor(X_train[0:1])
    y = torch_backend.to_tensor(Y_train[0:1]).type(torch.LongTensor)
    torch_grads = torch_backend.get_grads(torch_model, x, y, torch_loss)
    # sort the gradients to make sure they are the same...
    params = np.concatenate([param.detach().numpy().reshape(-1)
                             for param in torch_model.parameters()])[None]
    assert torch_grads.shape[-1] == params.shape[-1]

    np.testing.assert_allclose(np.sort(torch_grads), np.sort(tf_grads))
