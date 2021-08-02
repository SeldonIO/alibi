import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from typing import Any, List, Dict, Callable, Union, Optional, TYPE_CHECKING

from alibi.explainers.backends.cfrl_base import CounterfactualRLDataset
from alibi.models.tensorflow.actor_critic import Actor, Critic

if TYPE_CHECKING:
    from alibi.explainers.cfrl_base import NormalActionNoise


class TfCounterfactualRLDataset(CounterfactualRLDataset, keras.utils.Sequence):
    """ Tensorflow backend datasets. """

    def __init__(self,
                 X: np.ndarray,
                 preprocessor: Callable,
                 predictor: Callable,
                 conditional_func: Callable,
                 batch_size: int,
                 shuffle: bool = True) -> None:
        """
        Constructor.

        Parameters
        ----------
        X
            Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling
            the `preprocessor` function.
        preprocessor
            Preprocessor function. This function correspond to the preprocessing steps applied to the autoencoder model.
        predictor
            Prediction function. The classifier function should expect the input in the original format and preprocess
            it internally in the `predictor` if necessary.
        conditional_func
            Conditional function generator. Given an preprocesed input array, the functions generates a conditional
            array.
        batch_size
            Dimension of the batch used during training. The same batch size is used to infer the classification
            labels of the input dataset.
        shuffle
            Whether to shuffle the dataset each epoch. `True` by default.
        """
        super().__init__()

        self.X = X
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.conditional_func = conditional_func
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Infer the classification labels of the input dataset. This is performed in batches.
        self.Y_m = self.predict_batches(X=self.X,
                                        predictor=self.predictor,
                                        batch_size=self.batch_size)

        # Define number of classes for classification & minimum and maximum labels for regression
        self.num_classes: Optional[int] = None
        self.max_m: Optional[float] = None
        self.min_m: Optional[float] = None

        if self.Y_m.shape[1] > 1:
            self.num_classes = self.Y_m.shape[1]
        else:
            self.min_m = np.min(self.Y_m)
            self.max_n = np.max(self.Y_m)

        # Preprocess data.
        self.X = self.preprocessor(self.X)

        # Generate shuffled indexes.
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        This method is called every epoch and performs dataset shuffling.
        """
        self.indexes = np.arange(self.X.shape[0])

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return self.X.shape[0] // self.batch_size

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        if self.num_classes is not None:
            # Generate random targets for classification task.
            tgts = np.random.randint(low=0, high=self.num_classes, size=self.batch_size)
            Y_t = np.zeros((self.batch_size, self.num_classes))
            Y_t[np.arange(self.batch_size), tgts] = 1
        else:
            # Generate random target for regression task
            Y_t = np.random.uniform(low=self.min_m, high=self.max_m, size=(1, 1))

        # Select indices to be returned.
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        # compute conditional vector.
        C = self.conditional_func(self.X[idx * self.batch_size: (idx + 1) * self.batch_size])

        return {
            "X": self.X[indexes],
            "Y_m": self.Y_m[indexes],
            "Y_t": Y_t,
            "C": C
        }


def get_optimizer(model: Optional[keras.layers.Layer] = None, lr: float = 1e-3) -> keras.optimizers.Optimizer:
    """
    Constructs default Adam optimizer.

    Parameters
    ----------
    model
        Model to get the optimizer for. Not required for `tensorflow` backend.
    lr
        Learning rate.

    Returns
    -------
    Default optimizer.
    """
    return keras.optimizers.Adam(learning_rate=lr)


def get_actor(hidden_dim: int, output_dim: int) -> keras.layers.Layer:
    """
    Constructs the actor network.

    Parameters
    ----------
    hidden_dim
        Actor's hidden dimension
    output_dim
        Actor's output dimension.

    Returns
    -------
    Actor network.
    """
    return Actor(hidden_dim=hidden_dim, output_dim=output_dim)


def get_critic(hidden_dim: int) -> keras.layers.Layer:
    """
    Constructs the critic network.

    Parameters
    ----------
    hidden_dim:
        Critic's hidden dimension.

    Returns
    -------
    Critic network.
    """
    return Critic(hidden_dim=hidden_dim)


def sparsity_loss(X_hat_cf: tf.Tensor, X: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Default L1 sparsity loss.

    Parameters
    ----------
    X_hat_cf
        Autoencoder counterfactual reconstruction.
    X
        Input instance

    Returns
    -------
    L1 sparsity loss.
    """
    return {"sparsity_loss": tf.reduce_mean(tf.abs(X_hat_cf - X))}


def consistency_loss(Z_cf_pred: tf.Tensor, Z_cf_tgt: tf.Tensor):
    """
    Default 0 consistency loss.

    Parameters
    ----------
    Z_cf_pred
        Counterfactual embedding prediction.
    Z_cf_tgt
        Counterfactual embedding target.

    Returns
    -------
    0 consistency loss.
    """
    return {"consistency_loss": 0}


def data_generator(X: np.ndarray,
                   ae_preprocessor: Callable,
                   predictor: Callable,
                   conditional_func: Callable,
                   batch_size: int,
                   shuffle: bool = True,
                   **kwargs):
    """
    Constructs a tensorflow data generator.

    Parameters
    ----------
     X
        Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling
        the `preprocessor` function.
    ae_preprocessor
        Preprocessor function. This function correspond to the preprocessing steps applied to the autoencoder model.
    predictor
        Prediction function. The classifier function should expect the input in the original format and preprocess
        it internally in the `predictor` if necessary.
    conditional_func
        Conditional function generator. Given an preprocesed input array, the functions generates a conditional
        array.
    batch_size
        Dimension of the batch used during training. The same batch size is used to infer the classification
        labels of the input dataset.
    shuffle
        Whether to shuffle the dataset each epoch. `True` by default.
    """
    return TfCounterfactualRLDataset(X=X, preprocessor=ae_preprocessor, predictor=predictor,
                                     conditional_func=conditional_func, batch_size=batch_size, shuffle=shuffle)


def encode(X: Union[tf.Tensor, np.ndarray], ae: keras.Model, **kwargs) -> tf.Tensor:
    """
    Encodes the input tensor.

    Parameters
    ----------
    X
        Input to be encoded.
    ae
        Pre-trained autoencoder.

    Returns
    -------
    Input encoding.
    """
    return ae.encoder(X, training=False)


def decode(Z: Union[tf.Tensor, np.ndarray], ae: keras.Model, **kwargs):
    """
    Decodes an embedding tensor.

    Parameters
    ----------
    Z
        Embedding tensor to be decoded.
    ae
        Pre-trained autoencoder.

    Returns
    -------
    Embedding tensor decoding.
    """
    return ae.decoder(Z, training=False)


def generate_cf(Z: Union[np.ndarray, tf.Tensor],
                Y_m: Union[np.ndarray, tf.Tensor],
                Y_t: Union[np.ndarray, tf.Tensor],
                C: Optional[Union[np.ndarray, tf.Tensor]],
                actor: keras.Model,
                **kwargs) -> tf.Tensor:
    """
    Generates counterfactual embedding.

    Parameters
    ----------
    Z
        Input embedding tensor.
    Y_m
        Input classification label.
    Y_t
        Target counterfactual classification label.
    C
        Conditional tensor.
    actor
        Actor network. The model generates the counterfactual embedding.

    Returns
    -------
    Z_cf
        Counterfactual embedding.
    """
    # Convert labels, targets and conditiont float32
    Y_m = tf.cast(Y_m, dtype=tf.float32)
    Y_t = tf.cast(Y_t, dtype=tf.float32)
    C = tf.constant(C, dtype=tf.float32) if (C is not None) else C

    # Concatenate z_mean, y_m_ohe, y_t_ohe to create the input representation for the projection network (actor).
    state = [tf.reshape(Z, (Z.shape[0], -1)), Y_m, Y_t] + ([C] if (C is not None) else [])
    state = tf.concat(state, axis=1)

    # Pass the new input to the projection network (actor) to get the counterfactual embedding
    Z_cf = actor(state, training=False)
    return Z_cf


def add_noise(Z_cf: Union[tf.Tensor, np.ndarray],
              noise: 'NormalActionNoise',
              act_low: float,
              act_high: float,
              step: int,
              exploration_steps: int,
              **kwargs) -> tf.Tensor:
    """
    Add noise to the counterfactual embedding.

    Parameters
    ----------
    Z_cf
        Counterfactual embedding.
    noise
        Noise generator object.
    act_low
        Noise lower bound.
    act_high
        Noise upper bound.
    step
        Training step.
    exploration_steps
        Number of exploration steps. For the first `exploration_steps`, the noised counterfactul embedding
        is sampled uniformly at random.

    Returns
    -------
    Z_cf_tilde
        Noised counterfactual embedding.
    """
    # Generate noise.
    eps = noise(Z_cf.shape)

    if step > exploration_steps:
        Z_cf_tilde = Z_cf + eps
        Z_cf_tilde = tf.clip_by_value(Z_cf_tilde, clip_value_min=act_low, clip_value_max=act_high)
    else:
        # for the first exploration_steps, the action is sampled from a uniform distribution between
        # [act_low, act_high] to encourage exploration. After that, the algorithm returns to the normal exploration.
        Z_cf_tilde = tf.random.uniform(Z_cf.shape, minval=act_low, maxval=act_high)

    return Z_cf_tilde


@tf.function()
def update_actor_critic(ae: keras.Model,
                        critic: keras.Model,
                        actor: keras.Model,
                        optimizer_critic: keras.optimizers.Optimizer,
                        optimizer_actor: keras.optimizers.Optimizer,
                        sparsity_loss: Callable,
                        consistency_loss: Callable,
                        coeff_sparsity: float,
                        coeff_consistency: float,
                        X: np.ndarray,
                        X_cf: np.ndarray,
                        Z: np.ndarray,
                        Z_cf_tilde: np.ndarray,
                        Y_m: np.ndarray,
                        Y_t: np.ndarray,
                        C: Optional[np.ndarray],
                        R_tilde: np.ndarray,
                        **kwargs) -> Dict[str, Any]:
    """
    Training step. Updates actor and critic networks including additional losses.

    Parameters
    ----------
    ae
        Pre-trained autoencoder.
    critic
        Critic network.
    actor
        Actor network.
    optimizer_critic
        Critic's optimizer.
    optimizer_actor
        Actor's optimizer.
    sparsity_loss
        Sparsity loss function.
    consistency_loss
        Consistency loss function.
    coeff_sparsity
        Sparsity loss coefficient.
    coeff_consistency
        Consistency loss coefficient
    X
        Input array.
    X_cf
        Counterfactual array.
    Z
        Input embedding.
    Z_cf_tilde
        Noised counterfactual embedding.
    Y_m
        Input classification label.
    Y_t
        Target counterfactual classification label.
    C
        Conditional tensor.
    R_tilde
        Noised counterfactual reward.

    Returns
    -------
    Dictionary of losses.
    """
    # Define dictionary of losses.
    losses: Dict[str, float] = dict()

    # Transform labels and target to float32
    Y_m = tf.cast(Y_m, dtype=tf.float32)
    Y_t = tf.cast(Y_t, dtype=tf.float32)

    # Define state by concatenating the input embedding, the classification label, the target label, and optionally
    # the conditional vector if exists.
    state = [Z, Y_m, Y_t] + ([C] if C is not None else [])
    state = tf.concat(state, axis=1)

    # Define input for critic and compute q-values.
    with tf.GradientTape() as tape_critic:
        input_critic = tf.concat([state, Z_cf_tilde], axis=1)
        output_critic = tf.squeeze(critic(input_critic, training=True), axis=1)
        loss_critic = tf.reduce_mean(tf.square(output_critic - R_tilde))

    # Append critic's loss.
    losses.update({"loss_critic": loss_critic})

    # Update critic by gradient step.
    grads_critic = tape_critic.gradient(loss_critic, critic.trainable_weights)
    optimizer_critic.apply_gradients(zip(grads_critic, critic.trainable_weights))

    with tf.GradientTape() as tape_actor:
        # Compute counterfactual embedding.
        Z_cf = actor(state, training=True)

        # Compute critic's output
        input_critic = tf.concat([state, Z_cf], axis=1)
        output_critic = critic(input_critic, training=True)

        # Compute actors' loss.
        loss_actor = -tf.reduce_mean(output_critic)
        losses.update({"loss_actor": loss_actor})

        # Decode the counterfactual embedding.
        X_hat_cf = ae.decoder(Z_cf, training=False)

        # Compute sparsity losses and append sparsity loss.
        loss_sparsity = sparsity_loss(X_hat_cf, X)
        losses.update(loss_sparsity)

        # Add sparsity loss to the overall actor loss.
        for key in loss_sparsity.keys():
            loss_actor += coeff_sparsity * loss_sparsity[key]

        # Compute consistency loss and append consistency loss.
        Z_cf_tgt = ae.encoder(X_cf, training=False)
        loss_consistency = consistency_loss(Z_cf_pred=Z_cf, Z_cf_tgt=Z_cf_tgt)
        losses.update(loss_consistency)

        # Add consistency loss to the overall actor loss.
        for key in loss_consistency.keys():
            loss_actor += coeff_consistency * loss_consistency[key]

    # Update by gradient descent.
    grads_actor = tape_actor.gradient(loss_actor, actor.trainable_weights)
    optimizer_actor.apply_gradients(zip(grads_actor, actor.trainable_weights))

    # Return dictionary of losses for potential logging
    return losses


def to_numpy(X: Optional[Union[List, np.ndarray, tf.Tensor]]) -> Optional[Union[List, np.ndarray]]:
    """
    Converts given tensor to numpy array.

    Parameters
    ----------
    X
        Input tensor to be converted to numpy array.

    Returns
    -------
    Numpy representation of the input tensor.
    """
    if X is not None:
        if isinstance(X, np.ndarray):
            return X

        if isinstance(X, tf.Tensor):
            return X.numpy()

        if isinstance(X, list):
            return [to_numpy(e) for e in X]

        return np.array(X)
    return None


def to_tensor(X: Union[np.ndarray, tf.Tensor], **kwargs) -> Optional[tf.Tensor]:
    """
    Converts tensor to tf.Tensor

    Returns
    -------
    tf.Tensor conversion.
    """
    if X is not None:
        if isinstance(X, tf.Tensor):
            return X

        return tf.constant(X)

    return None


def save_model(path: Union[str, os.PathLike], model: keras.layers.Layer) -> None:
    """
    Saves a model and its optimizer.

    Parameters
    ----------
    path
        Path to the saving location.
    model
        Model to be saved.
    """
    model.save(path, save_format="tf")


def load_model(path: Union[str, os.PathLike]) -> keras.Model:
    """
    Loads a model and its optimizer.

    Parameters
    ----------
    path
        Path to the loading location.

    Returns
    -------
    Loaded model.
    """
    return keras.models.load_model(path, compile=False)


def set_seed(seed: int = 13):
    """
    Sets a seed to ensure reproducibility. Does NOT ensure reproducibility.

    Parameters
    ----------
    seed
        seed to be set
    """
    # others
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # tf random
    tf.random.set_seed(seed)
