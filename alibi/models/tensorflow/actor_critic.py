import tensorflow as tf
import tensorflow.keras as keras


class Actor(keras.Model):
    """ Actor network. """

    def __init__(self, hidden_dim: int, output_dim: int):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension
        output_dim
            Output dimension
        """
        super().__init__()
        self.fc1 = keras.layers.Dense(hidden_dim)
        self.ln1 = keras.layers.LayerNormalization()
        self.fc2 = keras.layers.Dense(hidden_dim)
        self.ln2 = keras.layers.LayerNormalization()
        self.fc3 = keras.layers.Dense(output_dim)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = tf.nn.relu(self.ln1(self.fc1(x)))
        x = tf.nn.relu(self.ln2(self.fc2(x)))
        x = tf.nn.tanh(self.fc3(x))
        return x


class Critic(keras.Model):
    """ Critic network. """

    def __init__(self, hidden_dim: int):
        """
        Constructor.

        hidden_dim
            Hidden dimension.
        """
        super().__init__()
        self.fc1 = keras.layers.Dense(hidden_dim)
        self.ln1 = keras.layers.LayerNormalization()
        self.fc2 = keras.layers.Dense(hidden_dim)
        self.ln2 = keras.layers.LayerNormalization()
        self.fc3 = keras.layers.Dense(1)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = tf.nn.relu(self.ln1(self.fc1(x)))
        x = tf.nn.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x
