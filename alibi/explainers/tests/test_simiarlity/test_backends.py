import pytest

import torch
import numpy as np
import tensorflow as tf

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


@pytest.mark.parametrize('trainable_emd, grads_shape', [(True, (61, )), (False, (21, ))])
def test_tf_embedding_similarity(trainable_emd, grads_shape):
    """Test `GradientSimilarity` explainer correctly handles sparsity and non-trainable layers for `tensorflow`.

    Test that `tensorflow` embedding layers work as expected and also that layers
    marked as non-trainable are not included in the gradients.
    See https://github.com/SeldonIO/alibi/issues/828.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10, 4, input_shape=(5,), trainable=trainable_emd),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    X = tf.random.uniform(shape=(1, 5), minval=0, maxval=10, dtype=tf.float32)
    Y = tf.random.uniform(shape=(1, 1), minval=0, maxval=10, dtype=tf.float32)
    loss_fn = tf.keras.losses.MeanSquaredError()
    tf_grads = _TensorFlowBackend.get_grads(model, X, Y, loss_fn)
    assert tf_grads.shape == grads_shape  # (4 * 10) * trainable_emd + (5 * 4) + 1


@pytest.mark.parametrize('trainable_emd, grads_shape', [(True, (61, )), (False, (21, ))])
@pytest.mark.parametrize('sparse', [True, False])
def test_pytorch_embedding_similarity(trainable_emd, grads_shape, sparse):
    """Test GradientSimilarity explainer correctly handles sparsity and non-trainable layers for pytorch.

    Tests that the `pytorch` embedding layers work as expected and that layers marked as
    non-trainable are not included in the gradients.
    """

    model = torch.nn.Sequential(
        torch.nn.Embedding(10, 4, 5, sparse=sparse),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(1)
    )

    model[0].weight.requires_grad = trainable_emd

    X = torch.randint(0, 10, (1, 5))
    Y = torch.randint(0, 10, (1, 1), dtype=torch.float32)
    loss_fn = torch.nn.MSELoss()
    pt_grads = _PytorchBackend.get_grads(model, X, Y, loss_fn)
    assert pt_grads.shape == grads_shape  # (4 * 10) * trainable_emd + (5 * 4) + 1


def test_non_numpy_grads_pytorch():
    """Test that the `pytorch` backend handles gradients withtout `numpy` methods correctly.

    `_PytorchBackend` should throw an error if the gradients cannot be converted to numpy arrays.
    """
    class MockTensor():
        is_sparse = False

    with pytest.raises(TypeError) as err:
        _PytorchBackend._grad_to_numpy(MockTensor())

    assert ("Could not convert gradient to `numpy` array. To ignore these gradients in the "
            "similarity computation set ``requires_grad=False`` on the corresponding parameter.") \
        in str(err.value)

    with pytest.raises(TypeError) as err:
        _PytorchBackend._grad_to_numpy(MockTensor(), 'test')

    assert ("Could not convert gradient to `numpy` array for the named tensor: test. "
            "To ignore these gradients in the similarity computation set ``requires_grad=False``"
            " on the corresponding parameter.") in str(err.value)


def test_non_numpy_grads_tensorflow():
    """Test that the `tensorflow` backend handles gradients without `numpy` methods correctly.

    `_TensorFlowBackend` should throw an error if the gradients cannot be converted to `numpy` arrays.
    """
    class MockTensor():
        is_sparse = False

    with pytest.raises(TypeError) as err:
        _TensorFlowBackend._grad_to_numpy(MockTensor())

    assert ("Could not convert gradient to `numpy` array. To ignore these gradients "
            "in the similarity computation set ``trainable=False`` on the corresponding parameter.") \
        in str(err.value)

    with pytest.raises(TypeError) as err:
        _TensorFlowBackend._grad_to_numpy(MockTensor(), 'test')

    assert ("Could not convert gradient to `numpy` array for the named tensor: test."
            " To ignore these gradients in the similarity computation set "
            "``trainable=False`` on the corresponding parameter.") in str(err.value)
