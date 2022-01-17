"""
This module contains the Tensorflow implementation of models used for the Counterfactual with Reinforcement Learning
experiments for both data modalities (image and tabular).
"""

import tensorflow as tf
import tensorflow.keras as keras
from typing import List


class MNISTClassifier(keras.Model):
    """
    MNIST classifier used in the experiments for Counterfactual with Reinforcement Learning. The model consists of two
    convolutional layers having 64 and 32 channels and a kernel size of 2 with ReLU nonlinearities, followed by
    maxpooling of size 2 and dropout of 0.3. The convolutional block is followed by a fully connected layer of 256 with
    ReLU nonlinearity, and finally a fully connected layer is used to predict the class logits (10 in MNIST case).
    """

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
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.
        training
            Training flag.
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        Classification logits.
        """
        x = self.dropout1(self.maxpool1(self.conv1(x)), training=training)
        x = self.dropout2(self.maxpool2(self.conv2(x)), training=training)
        x = self.fc2(self.fc1(self.flatten(x)))
        return x


class MNISTEncoder(keras.Model):
    """
    MNIST encoder used in the experiments for the Counterfactual with Reinforcement Learning. The model
    consists of 3 convolutional layers having 16, 8 and 8 channels and a kernel size of 3, with ReLU nonlinearities.
    Each convolutional layer is followed by a maxpooling layer of size 2. Finally, a fully connected layer
    follows the convolutional block with a tanh nonlinearity. The tanh clips the output between [-1, 1], required
    in the DDPG algorithm (e.g., [act_low, act_high]). The embedding dimension used in the paper is 32, although
    this can vary.
    """

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
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        Encoding representation having each component in the interval [-1, 1]
        """
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.fc1(self.flatten(x))
        return x


class MNISTDecoder(keras.Model):
    """
    MNIST decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of a fully
    connected layer of 128 units with ReLU activation followed by a convolutional block. The convolutional block
    consists fo 4 convolutional layers having 8, 8, 8  and 1 channels and a kernel size of 3. Each convolutional layer,
    except the last one, has ReLU nonlinearities and is followed by an up-sampling layer of size 2. The final layers
    uses a sigmoid activation to clip the output values in [0, 1].
    """

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
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        Decoded input having each component in the interval [0, 1].
        """
        x = self.reshape(self.fc1(x))
        x = self.up1(self.conv1(x))
        x = self.up2(self.conv2(x))
        x = self.up3(self.conv3(x))
        x = self.conv4(x)
        return x


class ADULTEncoder(keras.Model):
    """
    ADULT encoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
    two fully connected layers with ReLU and tanh nonlinearities. The tanh nonlinearity clips the embedding in [-1, 1]
    as required in the DDPG algorithm (e.g., [act_low, act_high]). The layers' dimensions used in the paper are
    128 and 15, although those can vary as they were selected to generalize across many datasets.
    """

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
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.
        **kwargs
            Other arguments.

        Returns
        -------
        Encoding representation having each component in the interval [-1, 1].
        """
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.tanh(self.fc2(x))
        return x


class ADULTDecoder(keras.Model):
    """
    ADULT decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
    of a fully connected layer with ReLU nonlinearity, and a multiheaded layer, one for each categorical feature and
    a single head for the rest of numerical features. The hidden dimension used in the paper is 128.
    """

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
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        List of reconstruction of the input tensor. First element corresponds to the reconstruction of all the \
        numerical features if they exist, and the rest of the elements correspond to each categorical feature.
        """
        x = tf.nn.relu(self.fc1(x))
        xs = [fc(x) for fc in self.fcs]
        return xs
