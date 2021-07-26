import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from typing import Any, List, Dict, Callable, Union, Optional

from alibi.explainers.backends.cfrl_base import CounterfactualRLDataset, NormalActionNoise
from alibi.models.tensorflow.actor_critic import Actor, Critic


class TfCounterfactualRLDataset(CounterfactualRLDataset, keras.utils.Sequence):
    """ Tensorflow backend datasets. """

    def __init__(self,
                 x: np.ndarray,
                 preprocessor: Callable,
                 predict_func: Callable,
                 conditional_func: Callable,
                 num_classes: int,
                 batch_size: int,
                 shuffle: bool = True) -> None:
        """
        Constructor.

        Parameters
        ----------
        x
            Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling
            the `preprocessor` function.
        preprocessor
            Preprocessor function. This function correspond to the preprocessing steps applied to the autoencoder model.
        predict_func
            Prediction function. The classifier function should expect the input in the original format and preprocess
            it internally in the `predict_func` if necessary.
        conditional_func
            Conditional function generator. Given an preprocesed input array, the functions generates a conditional
            array.
        num_classes
            Number of classes in the dataset.
        batch_size
            Dimension of the batch used during training. The same batch size is used to infer the classification
            labels of the input dataset.
        shuffle
            Whether to shuffle the dataset each epoch. `True` by default.
        """
        super().__init__()

        self.x = x
        self.preprocessor = preprocessor
        self.predict_func = predict_func
        self.conditional_func = conditional_func
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = None

        # Infer the classification labels of the input dataset. This is performed in batches.
        self.y_m = TfCounterfactualRLDataset.predict_batches(x=self.x,
                                                             predict_func=self.predict_func,
                                                             batch_size=self.batch_size)

        # Preprocess data.
        self.x = self.preprocessor(self.x)

        # Generate shuffled indexes.
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        This method is called every epoch and performs dataset shuffling.
        """
        self.indexes = np.arange(self.x.shape[0])

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        self.num_classes = np.clip(self.num_classes, a_min=0, a_max=2)  # TODO: remove this

        # Select indices to be returned.
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Generate random target.
        y_t = np.random.randint(low=0, high=self.num_classes, size=self.batch_size)

        # compute conditional vector.
        c = self.conditional_func(self.x[idx * self.batch_size: (idx + 1) * self.batch_size])

        return {
            "x": self.x[indexes],
            "y_m": self.y_m[indexes],
            "y_t": y_t,
            "c": c
        }


class TfCounterfactualRLBaseBackend:
    """ Tensorflow training backend. """

    @staticmethod
    def get_optimizer(lr: float = 1e-3) -> keras.optimizers.Optimizer:
        """
        Constructs default Adam optimizer.

        Returns
        -------
        Default optimizer.
        """
        return keras.optimizers.Adam(learning_rate=lr)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def sparsity_loss(x_hat_cf: tf.Tensor, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Default L1 sparsity loss.

        Parameters
        ----------
        x_hat_cf
            Autoencoder counterfactual reconstruction.
        x
            Input instance

        Returns
        -------
        L1 sparsity loss.
        """
        return {"sparsity_loss": tf.reduce_mean(tf.abs(x_hat_cf - x))}

    @staticmethod
    def consistency_loss(z_cf_pred: tf.Tensor, z_cf_tgt: tf.Tensor):
        """
        Default 0 consistency loss.

        Parameters
        ----------
        z_cf_pred
            Counterfactual embedding prediction.
        z_cf_tgt
            Counterfactual embedding target.

        Returns
        -------
        0 consistency loss.
        """
        return {"consistency_loss": 0}

    @staticmethod
    def data_generator(x: np.ndarray,
                       ae_preprocessor: Callable,
                       predict_func: Callable,
                       conditional_func: Callable,
                       num_classes: int,
                       batch_size: int,
                       shuffle: bool = True,
                       **kwargs):
        """
        Constructs a tensorflow data generator.

        Parameters
        ----------
         x
            Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling
            the `preprocessor` function.
        ae_preprocessor
            Preprocessor function. This function correspond to the preprocessing steps applied to the autoencoder model.
        predict_func
            Prediction function. The classifier function should expect the input in the original format and preprocess
            it internally in the `predict_func` if necessary.
        conditional_func
            Conditional function generator. Given an preprocesed input array, the functions generates a conditional
            array.
        num_classes
            Number of classes in the dataset.
        batch_size
            Dimension of the batch used during training. The same batch size is used to infer the classification
            labels of the input dataset.
        shuffle
            Whether to shuffle the dataset each epoch. `True` by default.
        """
        return TfCounterfactualRLDataset(x=x, preprocessor=ae_preprocessor, predict_func=predict_func,
                                         conditional_func=conditional_func, num_classes=num_classes,
                                         batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def encode(x: Union[tf.Tensor, np.ndarray], ae: keras.Model, **kwargs):
        """
        Encodes the input tensor.

        Parameters
        ----------
        x
            Input to be encoded.
        ae
            Pre-trained autoencoder.

        Returns
        -------
        Input encoding.
        """
        return ae.encoder(x, training=False)

    @staticmethod
    def decode(z: Union[tf.Tensor, np.ndarray], ae: keras.Model, **kwargs):
        """
        Decodes an embedding tensor.

        Parameters
        ----------
        z
            Embedding tensor to be decoded.
        ae
            Pre-trained autoencoder.

        Returns
        -------
        Embedding tensor decoding.
        """
        return ae.decoder(z, training=False)

    @staticmethod
    def generate_cf(z: Union[np.ndarray, tf.Tensor],
                    y_m: Union[np.ndarray, tf.Tensor],
                    y_t: Union[np.ndarray, tf.Tensor],
                    c: Optional[Union[np.ndarray, tf.Tensor]],
                    num_classes: int,
                    actor: keras.Model,
                    **kwargs) -> tf.Tensor:
        """
        Generates counterfactual embedding.

        Parameters
        ----------
        z
            Input embedding tensor.
        y_m
            Input classification label.
        y_t
            Target counterfactual classification label.
        c
            Conditional tensor.
        num_classes
            Number of classes to be considered.
        actor
            Actor network. The model generates the counterfactual embedding.

        Returns
        -------
        z_cf
            Counterfactual embedding.
        """
        # Transform to one hot encoding model's prediction and the given target
        y_m_ohe = tf.one_hot(tf.cast(y_m, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
        y_t_ohe = tf.one_hot(tf.cast(y_t, dtype=tf.int32), depth=num_classes, dtype=tf.float32)

        # Concatenate z_mean, y_m_ohe, y_t_ohe to create the input representation for the projection network (actor).
        state = [tf.reshape(z, (z.shape[0], -1)), y_m_ohe, y_t_ohe] + \
                ([tf.constant(c, dtype=tf.float32)] if (c is not None) else [])
        state = tf.concat(state, axis=1)

        # Pass the new input to the projection network (actor) to get the counterfactual embedding
        z_cf = actor(state, training=False)
        return z_cf

    @staticmethod
    def add_noise(z_cf: Union[tf.Tensor, np.ndarray],
                  noise: NormalActionNoise,
                  act_low: float,
                  act_high: float,
                  step: int,
                  exploration_steps: int,
                  **kwargs) -> tf.Tensor:
        """
        Add noise to the counterfactual embedding.

        Parameters
        ----------
        z_cf
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
        z_cf_tilde
            Noised counterfactual embedding.
        """
        # Generate noise.
        eps = noise(z_cf.shape)

        if step > exploration_steps:
            z_cf_tilde = z_cf + eps
            z_cf_tilde = tf.clip_by_value(z_cf_tilde, clip_value_min=act_low, clip_value_max=act_high)
        else:
            # for the first exploration_steps, the action is sampled from a uniform distribution between
            # [act_low, act_high] to encourage exploration. After that, the algorithm returns to the normal exploration.
            z_cf_tilde = tf.random.uniform(z_cf.shape, minval=act_low, maxval=act_high)

        return z_cf_tilde

    @staticmethod
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
                            num_classes: int,
                            x: np.ndarray,
                            x_cf: np.ndarray,
                            z: np.ndarray,
                            z_cf_tilde: np.ndarray,
                            y_m: np.ndarray,
                            y_t: np.ndarray,
                            c: Optional[np.ndarray],
                            r_tilde: np.ndarray,
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
        num_classes
            Number of classes to be considered.
        x
            Input array.
        x_cf
            Counterfactual array.
        z
            Input embedding.
        z_cf_tilde
            Noised counterfactual embedding.
        y_m
            Input classification label.
        y_t
            Target counterfactual classification label.
        c
            Conditional tensor.
        r_tilde
            Noised counterfactual reward.

        Returns
        -------
        Dictionary of losses.
        """
        # Define dictionary of losses.
        losses: Dict[str, float] = dict()

        # Transform classification labels into one-hot encoding.
        y_m_ohe = tf.one_hot(tf.cast(y_m, tf.int32), depth=num_classes, dtype=tf.float32)
        y_t_ohe = tf.one_hot(tf.cast(y_t, tf.int32), depth=num_classes, dtype=tf.float32)

        # Define state by concatenating the input embedding, the classification label, the target label, and optionally
        # the conditional vector if exists.
        state = [z, y_m_ohe, y_t_ohe] + ([c] if c is not None else [])
        state = tf.concat(state, axis=1)

        # Define input for critic and compute q-values.
        with tf.GradientTape() as tape_critic:
            input_critic = tf.concat([state, z_cf_tilde], axis=1)
            output_critic = tf.squeeze(critic(input_critic, training=True), axis=1)
            loss_critic = tf.reduce_mean(tf.square(output_critic - r_tilde))

        # Append critic's loss.
        losses.update({"loss_critic": loss_critic})

        # Update critic by gradient step.
        grads_critic = tape_critic.gradient(loss_critic, critic.trainable_weights)
        optimizer_critic.apply_gradients(zip(grads_critic, critic.trainable_weights))

        with tf.GradientTape() as tape_actor:
            # Compute counterfactual embedding.
            z_cf = actor(state, training=True)

            # Compute critic's output
            input_critic = tf.concat([state, z_cf], axis=1)
            output_critic = critic(input_critic, training=True)

            # Compute actors' loss.
            loss_actor = -tf.reduce_mean(output_critic)
            losses.update({"loss_actor": loss_actor})

            # Decode the counterfactual embedding.
            x_hat_cf = ae.decoder(z_cf, training=False)

            # Compute sparsity losses and append sparsity loss.
            loss_sparsity = sparsity_loss(x_hat_cf, x)
            losses.update(loss_sparsity)

            # Add sparsity loss to the overall actor loss.
            for key in loss_sparsity.keys():
                loss_actor += coeff_sparsity * loss_sparsity[key]

            # Compute consistency loss and append consistency loss.
            z_cf_tgt = ae.encoder(x_cf, training=False)
            loss_consistency = consistency_loss(z_cf_pred=z_cf, z_cf_tgt=z_cf_tgt)
            losses.update(loss_consistency)

            # Add consistency loss to the overall actor loss.
            for key in loss_consistency.keys():
                loss_actor += coeff_consistency * loss_consistency[key]

        # Update by gradient descent.
        grads_actor = tape_actor.gradient(loss_actor, actor.trainable_weights)
        optimizer_actor.apply_gradients(zip(grads_actor, actor.trainable_weights))

        # Return dictionary of losses for potential logging
        return losses

    @staticmethod
    def to_numpy(x: Optional[Union[List, np.ndarray, tf.Tensor]]) -> Optional[Union[List[np.ndarray], np.ndarray]]:
        """
        Converts given tensor to numpy array.

        Parameters
        ----------
        x
            Input tensor to be converted to numpy array.

        Returns
        -------
        Numpy representation of the input tensor.
        """
        if x is not None:
            if isinstance(x, np.ndarray):
                return x

            if isinstance(x, tf.Tensor):
                return x.numpy()

            if isinstance(x, list):
                return [TfCounterfactualRLBaseBackend.to_numpy(e) for e in x]

            return np.array(x)
        return None

    @staticmethod
    def to_tensor(x: Union[np.array, tf.Tensor], **kwargs) -> Optional[tf.Tensor]:
        """
        Converts tensor to tf.Tensor

        Returns
        -------
        tf.Tensor conversion.
        """
        if x is not None:
            if isinstance(x, tf.Tensor):
                return x

            return tf.constant(x)

        return None
