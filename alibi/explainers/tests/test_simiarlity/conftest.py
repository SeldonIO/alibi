import pytest
import numpy as np
import torch
from tensorflow import keras

import torch.nn as nn
import torch.nn.functional as F


@pytest.fixture(scope='module')
def random_dataset(request):
    """ Constructs a random 1d dataset. """
    shape = request.param.get('shape', (10, ))
    size = request.param.get('size', 100)

    # define random train set
    X_train = np.random.randn(size, *shape)
    Y_train = np.random.randint(low=0, high=10, size=size).astype(np.float64)

    # define random test set
    X_test = np.random.randn(size, *shape)
    Y_test = np.random.randint(low=0, high=10, size=size).astype(np.float64)
    return (X_train, Y_train), (X_test, Y_test)


@pytest.fixture(scope='module')
def tf_linear_model(request):
    input_shape = request.param.get('input_shape', (10, ))
    output_shape = request.param.get('output_shape', 10)

    """ Constructs a linear model. """
    return keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(5),
        keras.layers.ReLU(),
        keras.layers.Dense(output_shape),
        keras.layers.Softmax()
    ])

@pytest.fixture(scope='module')
def torch_linear_model(request):
    input_shape_arg = request.param.get('input_shape', (10, ))
    input_size = np.prod(input_shape_arg).item()
    output_shape_arg = request.param.get('output_shape', 10)

    """ Constructs a linear model. """

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
#
#
# @pytest.fixture(scope='module')
# def tf_cnn_model(request):
#     """ Constructs a cnn model. """
#     return keras.Sequential([
#         keras.layers.InputLayer(input_shape=(16, 16, 3)),
#         keras.layers.Conv2D(filters=10, kernel_size=(2, 3), padding='SAME'),
#         keras.layers.ReLU(),
#         keras.layers.Conv2D(filters=10, kernel_size=(5, 4), padding='VALID'),
#         keras.layers.ReLU(),
#         keras.layers.Conv2D(filters=10, kernel_size=(7, 7), padding='VALID'),
#         keras.layers.Flatten(),
#         keras.layers.Dense(10)
#     ])