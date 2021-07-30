from abc import ABC

import tensorflow as tf
import tensorflow.keras as keras
from typing import List


class MNISTClassifier(keras.Model, ABC):
    """ MNIST classifier. """

    def __init__(self, output_dim: int = 10, **kwargs) -> None:
        """
        Constructor.

        Parameters
        ----------
        output_dim
            Output dimension
        """
        super().__init__(**kwargs)

        self.conv1 = keras.layers.Conv2D(64, 2, padding="same", activation="relu")
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.dropout1 = keras.layers.Dropout(0.3)
        self.conv2 = keras.layers.Conv2D(32, 2, padding="same", activation="relu")
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.dropout2 = keras.layers.Dropout(0.3)
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256, activation="relu")
        self.fc2 = keras.layers.Dense(output_dim)

    def call(self, x: tf.Tensor, training: bool = True, **kwargs) -> tf.Tensor:
        x = self.dropout1(self.maxpool1(self.conv1(x)), training=training)
        x = self.dropout2(self.maxpool2(self.conv2(x)), training=training)
        x = self.fc2(self.fc1(self.flatten(x)))
        return x


class MNISTEncoder(keras.Model, ABC):
    """ MNIST encoder. """

    def __init__(self, latent_dim: int, **kwargs) -> None:
        """
        Constructor.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        """
        super().__init__(**kwargs)

        self.conv1 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv2 = keras.layers.Conv2D(8, 3, padding="same", activation="relu")
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv3 = keras.layers.Conv2D(8, 3, padding="same", activation="relu")
        self.maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(latent_dim, activation='tanh')

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.fc1(self.flatten(x))
        return x


class MNISTDecoder(keras.Model, ABC):
    """ MNIST decoder. """

    def __init__(self, **kwargs) -> None:
        """ Constructor. """
        super().__init__(**kwargs)

        self.fc1 = keras.layers.Dense(128, activation="relu")
        self.reshape = keras.layers.Reshape((4, 4, 8))
        self.conv1 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")
        self.up1 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv2 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")
        self.up2 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv3 = keras.layers.Conv2D(8, (3, 3), padding="valid", activation="relu")
        self.up3 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv4 = keras.layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid")

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.reshape(self.fc1(x))
        x = self.up1(self.conv1(x))
        x = self.up2(self.conv2(x))
        x = self.up3(self.conv3(x))
        x = self.conv4(x)
        return x


class ADULTEncoder(keras.Model, ABC):
    """ ADULT encoder. """

    def __init__(self, hidden_dim: int, latent_dim: int, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        latent_dim
            Latent dimension.
        """
        super().__init__(**kwargs)

        self.fc1 = keras.layers.Dense(hidden_dim)
        self.fc2 = keras.layers.Dense(latent_dim)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.tanh(self.fc2(x))
        return x


class ADULTDecoder(keras.Model, ABC):
    def __init__(self, hidden_dim: int, output_dims: List[int], **kwargs):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        output_dim
            List of output dimensions.
        """
        super().__init__(**kwargs)

        self.fc1 = keras.layers.Dense(hidden_dim)
        self.fcs = [keras.layers.Dense(dim) for dim in output_dims]

    def call(self, x: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        x = tf.nn.relu(self.fc1(x))
        xs = [fc(x) for fc in self.fcs]
        return xs
