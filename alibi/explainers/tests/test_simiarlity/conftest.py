import pytest
import random
import os

import numpy as np
from tensorflow import keras
import tensorflow as tf
import torch
import torch.nn as nn

from alibi.explainers.similarity.backends.pytorch.base import _TorchBackend
from alibi.explainers.similarity.backends.tensorflow.base import _TensorFlowBackend


def set_rdm_seed():
    tf.random.set_seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # Python std lib random seed
    random.seed(0)
    # Numpy, tensorflow
    np.random.seed(0)
    tf.random.set_seed(0)
    # Additional seeds potentially required when using a gpu
    # (see https://www.youtube.com/watch?v=TB07_mUMt0U&t=1804s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    os.environ['PYTHONHASHSEED'] = str(0)


def get_flattened_model_parameters(model):
    """
    Returns a flattened list of all `torch` or `tensorflow` model parameters.
    """
    if isinstance(model, nn.Module):
        return np.concatenate([_TorchBackend.to_numpy(p).reshape(-1) for p in model.parameters()])
    elif isinstance(model, tf.keras.Model):
        return np.concatenate([_TensorFlowBackend.to_numpy(p).reshape(-1) for p in model.trainable_weights])


@pytest.fixture(scope='module')
def random_reg_dataset(request):
    """
    Constructs a random regression dataset with 1d target.
    """
    set_rdm_seed()
    shape = request.param.get('shape', (10, ))
    size = request.param.get('size', 100)

    # define random train set
    X_train = np.random.randn(size, *shape).astype(np.float32)
    Y_train = np.random.randn(size, 1).astype(np.float32)

    # define random test set
    X_test = np.random.randn(size, *shape).astype(np.float32)
    Y_test = np.random.randn(size, 1).astype(np.float32)
    return (X_train, Y_train), (X_test, Y_test)


@pytest.fixture(scope='module')
def random_cls_dataset(request):
    """
    Constructs a random classification dataset with 10 labels.
    """
    set_rdm_seed()
    shape = request.param.get('shape', (10, ))
    size = request.param.get('size', 100)

    # define random train set
    X_train = np.random.randn(size, *shape)
    Y_train = np.random.randint(low=0, high=10, size=size).astype(np.int64)

    # define random test set
    X_test = np.random.randn(size, *shape)
    Y_test = np.random.randint(low=0, high=10, size=size).astype(np.int64)
    return (X_train, Y_train), (X_test, Y_test)


@pytest.fixture(scope='module')
def linear_cls_model(request):
    """
    Constructs a linear classification model, loss function and target function.
    """
    input_shape = request.param.get('input_shape', (10,))
    output_shape = request.param.get('output_shape', 10)
    framework = request.param.get('framework', 'tensorflow')

    model = {
        'tensorflow': lambda i_shape, o_shape: tf_linear_model(i_shape, o_shape),
        'pytorch': lambda i_shape, o_shape: torch_linear_model(i_shape, o_shape)
    }[framework](input_shape, output_shape)

    loss_fn = {
        'tensorflow': tf.keras.losses.SparseCategoricalCrossentropy,
        'pytorch': nn.CrossEntropyLoss
    }[framework]()

    target_fn = {
        'tensorflow': lambda x: np.argmax(model(x)),
        'pytorch': lambda x: torch.argmax(model(x), dim=1)
    }[framework]

    return framework, model, loss_fn, target_fn


@pytest.fixture(scope='module')
def linear_reg_model(request):
    """
    Constructs a linear regression model, loss function and target function.
    """
    input_shape = request.param.get('input_shape', (10,))
    output_shape = request.param.get('output_shape', 10)
    framework = request.param.get('framework', 'tensorflow')

    model = {
        'tensorflow': lambda i_shape, o_shape: tf_linear_model(i_shape, o_shape),
        'pytorch': lambda i_shape, o_shape: torch_linear_model(i_shape, o_shape)
    }[framework](input_shape, output_shape)

    loss_fn = {
        'tensorflow': tf.keras.losses.MeanSquaredError,
        'pytorch': nn.MSELoss
    }[framework]()

    target_fn = {
        'tensorflow': lambda x: model(x),
        'pytorch': lambda x: model(x)
    }[framework]

    return framework, model, loss_fn, target_fn


@pytest.fixture(scope='module')
def linear_models(request):
    """
    Constructs a pair of linear models and loss functions for tensorflow and torch.
    """
    set_rdm_seed()
    input_shape = request.param.get('input_shape', (10,))
    output_shape = request.param.get('output_shape', 10)
    tf_model = tf_linear_model(input_shape, output_shape)
    tf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    torch_model = torch_linear_model(input_shape, output_shape)
    torch_loss = nn.CrossEntropyLoss()
    return tf_model, tf_loss, torch_model, torch_loss


def tf_linear_model(input_shape, output_shape):
    """
    Constructs a linear model for tensorflow.
    """
    return keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(output_shape),
        keras.layers.Softmax()
    ])


def torch_linear_model(input_shape_arg, output_shape_arg):
    """
    Constructs a linear model for torch.
    """
    input_size = np.prod(input_shape_arg).item()

    class Model(nn.Module):
        def __init__(self, input_shape, output_shape):
            super(Model, self).__init__()
            self.linear_stack = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_shape, output_shape),
                nn.Softmax()
            )

        def forward(self, x):
            x = x.type(torch.FloatTensor)
            return self.linear_stack(x)

    return Model(input_size, output_shape_arg)