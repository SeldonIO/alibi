from alibi.explainers.influence.influence import InfluenceKeras
from alibi.explainers.influence.datafeeder import NumpyFeeder
import pytest
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

input_dim = 784
output_dim = nb_classes = 10

X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)


def get_model(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # logistic regression
    model = Sequential([
        Dense(output_dim,
              input_dim=input_dim,
              activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=5, batch_size=1024)

    return model


# set up model and datafeeder once
model = get_model(X_train, y_train, X_test, y_test)
feeder = NumpyFeeder(X_train, X_test, y_train, y_test)


@pytest.fixture
def influence_keras():
    exp = InfluenceKeras(model=model, workspace='tmp', datafeeder=feeder)
    return exp


def test_init(influence_keras):
    assert influence_keras.datafeeder.train_offset == 0
    X, y = influence_keras.datafeeder.test_batch([0])
    assert np.allclose(X, X_test[0])
    assert np.allclose(y, y_test[0])


def test_get_test_grad_loss(influence_keras):
    test_grad_loss = influence_keras.get_test_grad_loss(test_indices=[0])
    assert test_grad_loss[0].shape == (input_dim, output_dim)  # W
    assert test_grad_loss[1].shape == (output_dim,)  # b

    # TODO calculate test_grad_loss analytically