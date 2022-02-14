import pytest

import numpy as np
from tensorflow import keras
import tensorflow as tf
import torch
import torch.nn as nn


def get_flattened_model_parameters(model):
    """
    Returns a flattened list of all torch or tensorflow model parameters.
    """
    if isinstance(model, nn.Module):
        return np.concatenate([p.detach().numpy().reshape(-1) for p in model.parameters()])
    elif isinstance(model, tf.keras.Model):
        return np.concatenate([p.numpy().reshape(-1) for p in model.trainable_weights])


@pytest.fixture(scope='module')
def random_reg_dataset(request):
    """
    Constructs a random regression dataset with 1d target.
    """
    shape = request.param.get('shape', (10, ))
    size = request.param.get('size', 100)

    # define random train set
    x_train = np.random.randn(size, *shape).astype(np.float32)
    y_train = np.random.randn(size, 1).astype(np.float32)

    # define random test set
    x_test = np.random.randn(size, *shape).astype(np.float32)
    y_test = np.random.randn(size, 1).astype(np.float32)
    return (x_train, y_train), (x_test, y_test)


@pytest.fixture(scope='module')
def random_cls_dataset(request):
    """
    Constructs a random classification dataset with 10 labels.
    """
    shape = request.param.get('shape', (10, ))
    size = request.param.get('size', 100)

    # define random train set
    x_train = np.random.randn(size, *shape)
    y_train = np.random.randint(low=0, high=10, size=size).astype(np.int64)

    # define random test set
    x_test = np.random.randn(size, *shape)
    y_test = np.random.randint(low=0, high=10, size=size).astype(np.int64)
    return (x_train, y_train), (x_test, y_test)


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
        'torch': lambda i_shape, o_shape: torch_linear_model(i_shape, o_shape)
    }[framework](input_shape, output_shape)

    loss_fn = {
        'tensorflow': tf.keras.losses.SparseCategoricalCrossentropy,
        'torch': nn.CrossEntropyLoss
    }[framework]()

    target_fn = {
        'tensorflow': lambda x: np.argmax(model(x)),
        'torch': lambda x: torch.argmax(model(x), dim=1)
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
        'torch': lambda i_shape, o_shape: torch_linear_model(i_shape, o_shape)
    }[framework](input_shape, output_shape)

    loss_fn = {
        'tensorflow': tf.keras.losses.MeanSquaredError,
        'torch': nn.MSELoss
    }[framework]()

    target_fn = {
        'tensorflow': lambda x: model(x),
        'torch': lambda x: model(x)
    }[framework]

    return framework, model, loss_fn, target_fn


@pytest.fixture(scope='module')
def linear_models(request):
    """
    Constructs a pair of linear models and loss functions for tensorflow and torch.
    """
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
