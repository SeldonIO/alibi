"""
This module contains the Tensorflow implementation of actor-critic networks used in the Counterfactual with
Reinforcement Learning for both data modalities. The models' architectures follow the standard actor-critic design and
can have broader use-cases.
"""

import tensorflow as tf
import tensorflow.keras as keras


class Actor(keras.Model):
    """
    Actor network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.
    The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
    tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.
    """

    def __init__(self, hidden_dim: int, output_dim: int, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension
        output_dim
            Output dimension
        """
        super().__init__(**kwargs)
        self.fc1 = keras.layers.Dense(hidden_dim)
        self.ln1 = keras.layers.LayerNormalization()
        self.fc2 = keras.layers.Dense(hidden_dim)
        self.ln2 = keras.layers.LayerNormalization()
        self.fc3 = keras.layers.Dense(output_dim)

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
        Continuous action.
        """
        x = tf.nn.relu(self.ln1(self.fc1(x)))
        x = tf.nn.relu(self.ln2(self.fc2(x)))
        x = tf.nn.tanh(self.fc3(x))
        return x


class Critic(keras.Model):
    """
    Critic network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.
    The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
    tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.
    """

    def __init__(self, hidden_dim: int, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        """
        super().__init__(**kwargs)
        self.fc1 = keras.layers.Dense(hidden_dim)
        self.ln1 = keras.layers.LayerNormalization()
        self.fc2 = keras.layers.Dense(hidden_dim)
        self.ln2 = keras.layers.LayerNormalization()
        self.fc3 = keras.layers.Dense(1)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Critic value.
        """
        x = tf.nn.relu(self.ln1(self.fc1(x)))
        x = tf.nn.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x
