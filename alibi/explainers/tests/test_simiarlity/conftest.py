import pytest
import numpy as np
import torch
from tensorflow import keras

import torch.nn as nn
import tensorflow as tf


def get_flattened_model_parameters(model):
    if isinstance(model, nn.Module):
        return np.concatenate([p.detach().numpy().reshape(-1) for p in model.parameters()])
    elif isinstance(model, tf.keras.Model):
        return np.concatenate([p.numpy().reshape(-1) for p in model.trainable_weights])


@pytest.fixture(scope='module')
def random_dataset(request):
    """ Constructs a random 1d dataset. """
    shape = request.param.get('shape', (10, ))
    size = request.param.get('size', 100)

    # define random train set
    x_train = np.random.randn(size, *shape)
    y_train = np.random.randint(low=0, high=10, size=size).astype(np.float64)

    # define random test set
    x_test = np.random.randn(size, *shape)
    y_test = np.random.randint(low=0, high=10, size=size).astype(np.float64)
    return (x_train, y_train), (x_test, y_test)


@pytest.fixture(scope='module')
def linear_model(request):
    input_shape = request.param.get('input_shape', (10,))
    output_shape = request.param.get('output_shape', 10)
    framework = request.param.get('framework', 'tensorflow')
    return framework, {
        'tensorflow': lambda i_shape, o_shape: tf_linear_model(i_shape, o_shape),
        'pytorch': lambda i_shape, o_shape: torch_linear_model(i_shape, o_shape)
    }[framework](input_shape, output_shape), {
        'tensorflow': lambda: tf.keras.losses.SparseCategoricalCrossentropy(),
        'pytorch': lambda: nn.CrossEntropyLoss()
    }[framework]()


@pytest.fixture(scope='module')
def linear_models(request):
    input_shape = request.param.get('input_shape', (10,))
    output_shape = request.param.get('output_shape', 10)
    tf_model = tf_linear_model(input_shape, output_shape)
    tf_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    torch_model = torch_linear_model(input_shape, output_shape)
    torch_loss = nn.CrossEntropyLoss()
    return tf_model, tf_loss, torch_model, torch_loss


def tf_linear_model(input_shape, output_shape):
    """ Constructs a linear model. """
    return keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(5),
        keras.layers.ReLU(),
        keras.layers.Dense(output_shape),
        keras.layers.Softmax()
    ])


def torch_linear_model(input_shape_arg, output_shape_arg):
    input_size = np.prod(input_shape_arg).item()

    class Model(nn.Module):
        def __init__(self, input_shape, output_shape):
            super(Model, self).__init__()
            self.linear_stack = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_size, 5),
                nn.ReLU(),
                nn.Linear(5, output_shape),
                nn.Softmax()
            )

        def forward(self, x):
            x = x.type(torch.FloatTensor)
            return self.linear_stack(x)

    return Model(input_shape_arg, output_shape_arg)
