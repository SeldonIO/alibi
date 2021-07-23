import os
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Any, Callable, Optional, Tuple, Dict, List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import tensorflow as tf
import tensorflow.keras as keras

from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.models.tensorflow.autoencoder import AE as TensorflowAE
from alibi.models.pytorch.autoencoder import AE as PytorchAE

logger = logging.getLogger(__name__)


class NormalActionNoise:
    """ Normal noise generator. """

    def __init__(self, mu: float, sigma: float):
        """
        Constructor.

        Parameters
        ----------
        mu
            Mean of the normal noise.
        sigma
            Standard deviation of the noise.
        """
        self.mu = mu
        self.sigma = sigma

    def __call__(self, shape: Tuple[int, ...]):
        """
        Generates normal noise with the appropriate mean and standard deviation.

        Parameters
        ----------
        shape
            Shape of the tensor to be generated

        Returns
        -------
        Normal noise with the appropriate mean, standard deviation and shape.
        """
        return self.mu + self.sigma * np.random.randn(*shape)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayBuffer(object):
    """
    Circular experience replay buffer for `CounterfactualRL`(DDPG). When the buffer is filled, then the oldest
    experience is replaced by the new one (FIFO).
    """

    def __init__(self, size=1000):
        """
        Constructor.

        Parameters
        ----------
        size
            Dimension of the buffer in batch size. This that the total memory allocated is proportional with the
            `size` * `batch_size`, where `batch_size` is inferred from the input tensors passed in the `append`
            method.
        """
        self.x, self.x_cf = None, None           # buffers for the input and the counterfactuals
        self.y_m, self.y_t = None, None          # buffers for the model's prediction and counterfactual target
        self.z, self.z_cf_tilde = None, None     # buffers for the input embedding and noised counterfactual embedding
        self.c = None                            # buffer for the conditional tensor
        self.r_tilde = None                      # buffer for the noised counterfactual reward tensor

        self.idx = 0                             # cursor for the buffer
        self.len = 0                             # current length of the buffer
        self.size = size                         # buffer's maximum capacity
        self.batch_size = None                   # batch size (inferred during `append`)

    def append(self,
               x: np.ndarray,
               x_cf: np.ndarray,
               y_m: np.ndarray,
               y_t: np.ndarray,
               z: np.ndarray,
               z_cf_tilde: np.ndarray,
               c: Optional[np.ndarray],
               r_tilde: np.ndarray,
               **kwargs) -> None:
        """
        Adds experience to the replay buffer. When the buffer is filled, then the oldest experience is replaced
        by the new one (FIFO).

        Parameters
        ----------
        x
            Input array.
        x_cf
            Counterfactual array.
        y_m
            Model's prediction class of x.
        y_t
            Counterfactual target class.
        z
            Input's embedding.
        z_cf_tilde
            Noised counterfactual embedding.
        c
            Conditional array.
        r_tilde
            Noised counterfactual reward array.
        """
        # initialize the buffers
        if self.x is None:
            self.batch_size = x.shape[0]

            # allocate memory
            self.x = np.zeros((self.size * self.batch_size, *x.shape[1:]), dtype=np.float32)
            self.x_cf = np.zeros((self.size * self.batch_size, *x_cf.shape[1:]), dtype=np.float32)
            self.y_m = np.zeros((self.size * self.batch_size, *y_m.shape[1:]), dtype=np.float32)
            self.y_t = np.zeros((self.size * self.batch_size, *y_t.shape[1:]), dtype=np.float32)
            self.z = np.zeros((self.size * self.batch_size, *z.shape[1:]), dtype=np.float32)
            self.z_cf_tilde = np.zeros((self.size * self.batch_size, *z_cf_tilde.shape[1:]), dtype=np.float32)
            self.r_tilde = np.zeros((self.size * self.batch_size, *r_tilde.shape[1:]), dtype=np.float32)

            # Conditional tensor can be `None` when no condition is included. If it is not `None`, allocate memory.
            if c is not None:
                self.c = np.zeros((self.size * self.batch_size, *c.shape[1:]), dtype=np.float32)

        # increase the length of the buffer if not full
        if self.len < self.size:
            self.len += 1

        # compute the first position where to add most recent experience
        start = self.batch_size * self.idx

        # add new data / replace old experience (note that a full batch is added at once)
        self.x[start:start + self.batch_size] = x
        self.x_cf[start:start + self.batch_size] = x_cf
        self.y_m[start:start + self.batch_size] = y_m
        self.y_t[start:start + self.batch_size] = y_t
        self.z[start:start + self.batch_size] = z
        self.z_cf_tilde[start:start + self.batch_size] = z_cf_tilde
        self.r_tilde[start:start + self.batch_size] = r_tilde

        if c is not None:
            self.c[start:start + self.batch_size] = c

        # Compute the next index. Not that if the buffer reached its maximum capacity, for the next iteration
        # we start replacing old batches.
        self.idx = (self.idx + 1) % self.size

    def sample(self) -> Dict[str, np.ndarray]:
        """
        Sample a batch of experience form the replay buffer.

        Returns
        --------
        A batch experience. For a description of the keys/values returned, see parameter descriptions in `append`
        method. The batch size returned is the same as the one passed in the `append`.
        """
        # generate random indices to be sampled
        rand_idx = torch.randint(low=0, high=self.len * self.batch_size, size=(self.batch_size, ))

        # extract data form buffers
        x = self.x[rand_idx]                                        # input array
        x_cf = self.x_cf[rand_idx]                                  # counterfactual
        y_m = self.y_m[rand_idx]                                    # model's prediction
        y_t = self.y_t[rand_idx]                                    # counterfactual target
        z = self.z[rand_idx]                                        # input embedding
        z_cf_tilde = self.z_cf_tilde[rand_idx]                      # noised counterfactual embedding
        c = self.c[rand_idx] if (self.c is not None) else None      # conditional array if exists
        r_tilde = self.r_tilde[rand_idx]                            # noised counterfactual reward

        return {
            "x": x,
            "x_cf": x_cf,
            "y_m": y_m,
            "y_t": y_t,
            "z": z,
            "z_cf_tilde": z_cf_tilde,
            "c": c,
            "r_tilde": r_tilde
        }


class CounterfactualRLDataset(ABC):
    @staticmethod
    def predict_batches(x: np.ndarray, predict_func: Callable, batch_size: int) -> np.ndarray:
        """
        Infer the classification labels of the input dataset. This is performed in batches.

        Parameters
        ----------
        x
            Input to be classified.
        predict_func
            Prediction function.
        batch_size
            Maximum batch size to be used during each inference step.

        Returns
        -------
        Classification labels.
        """
        n_minibatch = int(np.ceil(x.shape[0] / batch_size))
        y_m = np.zeros(x.shape[0])

        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, x.shape[0])
            y_m[istart:istop] = predict_func(x[istart:istop])

        return y_m

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError


class PTCounterfactualRLDataset(Dataset, CounterfactualRLDataset):
    """ Pytorch backend datasets. """

    def __init__(self,
                 x: np.ndarray,
                 preprocessor: Callable,
                 predict_func: Callable,
                 conditional_func: Callable,
                 num_classes: int,
                 batch_size: int) -> None:
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
            Conditional function generator. Given an preprocessed input array, the functions generates a conditional
            array.
        num_classes
            Number of classes in the dataset.
        batch_size
            Dimension of the batch used during training. The same batch size is used to infer the classification
            labels of the input dataset.
        """
        super().__init__()

        self.x = x
        self.preprocessor = preprocessor
        self.predict_func = predict_func
        self.conditional_func = conditional_func
        self.num_classes = num_classes
        self.batch_size = batch_size

        # Infer the classification labels of the input dataset. This is performed in batches.
        self.y_m = PTCounterfactualRLDataset.predict_batches(x=self.x,
                                                             predict_func=self.predict_func,
                                                             batch_size=self.batch_size)

        # Preprocess the input data.
        self.x = self.preprocessor(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        self.num_classes = np.clip(self.num_classes, a_min=0, a_max=2)  # TODO: remove this

        # Generate random target class.
        y_t = np.random.randint(low=0, high=self.num_classes, size=1).item()
        data = {
            "x": self.x[idx],
            "y_m": self.y_m[idx],
            "y_t": y_t,
        }

        # Construct conditional vector.
        c = self.conditional_func(self.x[idx:idx+1])
        if c is not None:
            data.update({"c": c.reshape(-1)})

        return data


class TFCounterfactualRLDataset(CounterfactualRLDataset, keras.utils.Sequence):
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
        self.y_m = TFCounterfactualRLDataset.predict_batches(x=self.x,
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


class Postprocessing(ABC):
    @abstractmethod
    def __call__(self, x_cf, x, c: np.ndarray) -> Any:
        """
        Post-processing function

        Parameters
        ----------
        x_cf
            Counterfactual instance
        x
            Input instance.
        c
            Conditional vector.

        Returns
        -------
        x_cf
            Post-processed x_cf.
        """
        raise NotImplemented





class TFCounterfactualRLBackend:
    """ Tensorflow training backend. """

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
        return TFCounterfactualRLDataset(x=x, preprocessor=ae_preprocessor, predict_func=predict_func,
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
                    **kwargs) -> Tuple[torch.Tensor, ...]:
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
    def to_numpy(x: Optional[Union[List, np.ndarray, torch.Tensor]]) -> Optional[Union[List[np.ndarray], np.ndarray]]:
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
                return [TFCounterfactualRLBackend.to_numpy(e) for e in x]

            return np.array(x)
        return None


class PTCounterfactualRLBackend:
    @staticmethod
    def data_generator(x: np.ndarray,
                       ae_preprocessor: Callable,
                       predict_func: Callable,
                       conditional_func: Callable,
                       num_classes: int,
                       batch_size: int,
                       shuffle: bool,
                       num_workers: int,
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
        num_workers
            Number of worker processes to be created.
        """
        dataset = PTCounterfactualRLDataset(x=x, preprocessor=ae_preprocessor, predict_func=predict_func,
                                            conditional_func=conditional_func, num_classes=num_classes, batch_size=batch_size)
        return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                          shuffle=shuffle, drop_last=True)

    @staticmethod
    @torch.no_grad()
    def encode(x: torch.Tensor, ae: nn.Module, device: torch.device, **kwargs):
        """
        Encodes the input tensor.

        Parameters
        ----------
        x
            Input to be encoded.
        ae
            Pre-trained autoencoder.
        device
            Device to send data to.

        Returns
        -------
        Input encoding.
        """
        ae.eval()
        return ae.encoder(x.float().to(device))

    @staticmethod
    @torch.no_grad()
    def decode(z: torch.Tensor, ae: nn.Module, device: torch.device, **kwargs):
        """
        Decodes an embedding tensor.

        Parameters
        ----------
        z
            Embedding tensor to be decoded.
        ae
            Pre-trained autoencoder.
        device
            Device to sent data to.

        Returns
        -------
        Embedding tensor decoding.
        """
        ae.eval()
        return ae.decoder(z.float().to(device))

    @staticmethod
    @torch.no_grad()
    def generate_cf(z: torch.Tensor,
                    y_m: torch.Tensor,
                    y_t: torch.Tensor,
                    c: Optional[torch.Tensor],
                    num_classes: int,
                    ae: nn.Module,
                    actor: nn.Module,
                    device: torch.device,
                    **kwargs) -> Tuple[torch.Tensor, ...]:
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
        # Set autoencoder and actor to evaluation mode.
        ae.eval()
        actor.eval()

        # Transform classification labels into one-hot encoding.
        y_m_ohe = F.one_hot(y_m.long(), num_classes=num_classes).float().to(device)
        y_t_ohe = F.one_hot(y_t.long(), num_classes=num_classes).float().to(device)

        # Concatenate z_mean, y_m_ohe, y_t_ohe to create the input representation for the projection network (actor).
        state = [z.view(z.shape[0], -1), y_m_ohe, y_t_ohe] + ([c.float().to(device)] if (c is not None) else [])
        state = torch.cat(state, dim=1)

        # Pass the new input to the projection network (actor) to get the counterfactual embedding
        z_cf = actor(state)
        return z_cf

    @staticmethod
    def add_noise(z_cf: torch.Tensor,
                  noise: NormalActionNoise,
                  act_low: float,
                  act_high: float,
                  step: int,
                  exploration_steps: int,
                  device: torch.device,
                  **kwargs) -> torch.Tensor:
        """
        Add noise to the counterfactual embedding.

        Parameters
        ----------
        z_cf
           Counterfactual embedding.
        noise
           Noise generator object.
        act_low
            Action lower bound.
        act_high
            Action upper bound.
        step
           Training step.
        exploration_steps
           Number of exploration steps. For the first `exploration_steps`, the noised counterfactul embedding
           is sampled uniformly at random.
        device
            Device to send data to.

        Returns
        -------
        z_cf_tilde
           Noised counterfactual embedding.
        """
        # Generate noise.
        eps = torch.tensor(noise(z_cf.shape)).float().to(device)

        if step > exploration_steps:
            z_cf_tilde = z_cf + eps
            z_cf_tilde = torch.clamp(z_cf_tilde, min=act_low, max=act_high)
        else:
            # for the first exploration_steps, the action is sampled from a uniform distribution between
            # [act_low, act_high] to encourage exploration. After that, the algorithm returns to the normal exploration.
            z_cf_tilde = (act_low + (act_high - act_low) * torch.rand_like(z_cf)).to(device)

        return z_cf_tilde

    @staticmethod
    def update_actor_critic(ae: nn.Module,
                            critic: nn.Module,
                            actor: nn.Module,
                            optimizer_critic: torch.optim.Optimizer,
                            optimizer_actor: torch.optim.Optimizer,
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
                            device: torch.device,
                            **kwargs):
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

        # Set autoencoder to evaluation mode.
        ae.eval()

        # Set actor and critic to training mode.
        actor.train()
        critic.train()

        # Define dictionary of losses.
        losses: Dict[str, float] = dict()

        # Transform data to tensors and it device
        x = torch.tensor(x).float().to(device)
        x_cf = torch.tensor(x_cf).float().to(device)
        z = torch.tensor(z).float().to(device)
        z_cf_tilde = torch.tensor(z_cf_tilde).float().to(device)
        y_m_ohe = F.one_hot(torch.tensor(y_m, dtype=torch.long), num_classes=num_classes).float().to(device)
        y_t_ohe = F.one_hot(torch.tensor(y_t, dtype=torch.long), num_classes=num_classes).float().to(device)
        c = torch.tensor(c).float().to(device) if (c is not None) else None
        r_tilde = torch.tensor(r_tilde).float().to(device)

        # Define state by concatenating the input embedding, the classification label, the target label, and optionally
        # the conditional vector if exists.
        state = [z, y_m_ohe, y_t_ohe] + ([c.float().to(device)] if (c is not None) else [])
        state = torch.cat(state, dim=1).to(device)

        # Define input for critic, compute q-values and append critic's loss.
        input_critic = torch.cat([state, z_cf_tilde], dim=1).float()
        output_critic = critic(input_critic).squeeze(1)
        loss_critic = F.mse_loss(output_critic, r_tilde)
        losses.update({"loss_critic": loss_critic.item()})

        # Update critic by gradient step.
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

        # Compute counterfactual embedding.
        z_cf = actor(state)

        # Compute critic's output.
        critic.eval()
        input_critic = torch.cat([state, z_cf], dim=1)
        output_critic = critic(input_critic)

        # Compute actor's loss.
        loss_actor = -torch.mean(output_critic)
        losses.update({"loss_actor": loss_actor})

        # Decode the output of the actor.
        x_hat_cf = ae.decoder(z_cf)

        # Compute sparsity losses.
        loss_sparsity = sparsity_loss(x_hat_cf, x)
        losses.update(loss_sparsity)

        # Add sparsity loss to the overall actor's loss.
        for key in loss_sparsity.keys():
            loss_actor += coeff_sparsity * loss_sparsity[key]

        # Compute consistency loss.
        z_cf_tgt = PTCounterfactualRLBackend.encode(x=x_cf, ae=ae, device=device)
        loss_consistency = consistency_loss(z_cf_pred=z_cf, z_cf_tgt=z_cf_tgt)
        losses.update(loss_consistency)

        # Add consistency loss to the overall actor loss.
        for key in loss_consistency.keys():
            loss_actor += coeff_consistency * loss_consistency[key]

        # Update by gradient descent.
        optimizer_actor.zero_grad()
        loss_actor.backward()
        optimizer_actor.step()

        # Return dictionary of losses for potential logging.
        return losses

    @staticmethod
    def to_numpy(x: Optional[Union[List, np.ndarray, torch.Tensor]]) -> Optional[Union[List[np.ndarray], np.ndarray]]:
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

            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()

            if isinstance(x, list):
                return [PTCounterfactualRLBackend.to_numpy(e) for e in x]

            return np.array(x)
        return None


DEFAULT_PARAMS = {
    "act_noise": 0.1,
    "act_low": -1.0,
    "act_high": 1.0,
    "replay_buffer_size": 1000,
    "batch_size": 128,
    "num_workers": 4,
    "shuffle": True,
    "num_classes": 2,
    "exploration_steps": 100,
    "update_every": 1,
    "update_after": 10,
    "backend_flag": "pytorch",
    "train_steps": 100000,
    "ae_preprocessor": lambda x: x,
    "ae_inv_preprocessor": lambda x: x,
    "reward_func": lambda y_pred, y_true: y_pred == y_true,
    "postprocessing_funcs": [lambda x_cf, x, c: x_cf],
    "conditional_func": lambda x: None,
    "experience_callbacks": [],
    "train_callbacks": []
}


class CounterfactualRLBase(Explainer, FitMixin):
    """ Counterfactual Reinforcement Learning Base. """

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"

    def __init__(self,
                 ae: Union[TensorflowAE, PytorchAE],
                 actor: Union[keras.Sequential, nn.Sequential],
                 critic: Union[keras.Sequential, nn.Sequential],
                 predict_func: Callable,
                 coeff_sparsity: float,
                 coeff_consistency: float,
                 num_classes: int,
                 backend: str = "tensorflow",
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        ae
            Pre-trained autoencoder.
        actor
            Actor network.
        critic
            Critic network.
        predict_func.
            Prediction function. This corresponds to the classifier.
        coeff_sparsity
            Sparsity loss coefficient.
        coeff_consistency
            Consistency loss coefficient.
        backend
            Deep learning backend: `tensorflow`|`pytorch`. Default `tensorflow`.
        """
        if backend not in [CounterfactualRLBase.PYTORCH, CounterfactualRLBase.TENSORFLOW]:
            raise ValueError(f"Backend {backend} not supported.")

        # validate arguments
        self.params, all_params = self._validate_kwargs(ae=ae,
                                                        actor=actor,
                                                        critic=critic,
                                                        predict_func=predict_func,
                                                        coeff_sparsity=coeff_sparsity,
                                                        coeff_consistency=coeff_consistency,
                                                        num_classes=num_classes,
                                                        backend=backend,
                                                        **kwargs)

        # If pytorch backend, then check if GPU is available
        if self.params["backend"] == CounterfactualRLBase.PYTORCH:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.params.update({"device": device})

            # send auto-encoder to device
            self.params["ae"].to(device)

            # sent actor and critic to device
            self.params["actor"].to(device)
            self.params["critic"].to(device)

        # Set backend according to the backend_flag.
        self.Backend = TFCounterfactualRLBackend if backend == "tensorflow" else PTCounterfactualRLBackend

        # # update metadata
        # self.meta['params'].update(**all_params)

    def _validate_kwargs(self,
                         ae: Union[TensorflowAE, PytorchAE],
                         actor: Union[keras.Model, nn.Module],
                         critic: Union[keras.Model, nn.Module],
                         predict_func: Callable,
                         coeff_sparsity: float,
                         coeff_consistency: float,
                         num_classes: int,
                         backend: str,
                         **kwargs):
        """
        Validates arguments.

        Parameters
        ----------
        ae
            Pre-trained autoencoder.
        actor
            Actor network.
        critic
            Critic network.
        predict_func.
            Prediction function. This corresponds to the classifier.
        coeff_sparsity
            Sparsity loss coefficient.
        coeff_consistency
            Consistency loss coefficient.
        num_classes
            Number of classes to consider.
        backend
            Deep learning backend: `tensorflow`|`pytorch`.
        """
        # Copy default parameters.
        params = deepcopy(DEFAULT_PARAMS)


        # Update parameters with mandatory arguments
        params.update({
            "ae": ae,
            "actor": actor,
            "critic": critic,
            "predict_func": predict_func,
            "coeff_sparsity": coeff_sparsity,
            "coeff_consistency": coeff_consistency,
            "num_classes": num_classes,
            "backend": backend
        })

        # Address backend specific params.
        # Add optimizers if not user-specified.
        optimizers = ["optimizer_actor", "optimizer_critic"]

        for optim in optimizers:
            # If the optimizer is user-specified
            if optim in kwargs:
                params.update({optim: kwargs[optim]})
                continue

            # If the optimizer is not user-specified, it need to be initialized. The initialization is backend specific.
            if backend == CounterfactualRLBase.TENSORFLOW:
                params.update({optim: keras.optimizers.Adam(learning_rate=1e-3)})
            else:
                model_name = optim.split("_")[1]      # Extract model name.
                model = params[model_name]            # Extract model.
                params.update({optim: torch.optim.Adam(model.parameters(), lr=1e-3)})

        # Add sparsity loss if not user-specified.
        if "sparsity_loss" not in kwargs:
            if backend == CounterfactualRLBase.TENSORFLOW:
                # define tensorflow backend sparsity loss
                def sparsity_loss(x_hat_cf: tf.Tensor, x: tf.Tensor) -> Dict[str, tf.Tensor]:
                    return {"sparsity_loss": tf.reduce_mean(tf.abs(x_hat_cf - x))}
            else:
                # define pytorch backend sparsity loss
                def sparsity_loss(x_hat_cf: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                    return {"sparsity_loss": F.l1_loss(x_hat_cf, x)}

            # update sparsity loss
            params["sparsity_loss"] = sparsity_loss

        # Add consistency loss if not user-specified. Although the implementation is not backend specific, it is
        # included here to be consistent with the sparsity loss.
        if "consistency_loss" not in kwargs:
            def consistency_loss(z_cf_pred: Union[torch.Tensor, tf.Tensor], z_cf_tgt: Union[torch.Tensor, tf.Tensor]):
                return {"consistency_loss": 0}

            # update consistency loss
            params["consistency_loss"] = consistency_loss

        # Define dictionary of all parameters. Shallow copy of values.
        all_params = params.copy()

        # Validate arguments.
        allowed_keys = set(params.keys())
        provided_keys = set(kwargs.keys())
        common_keys = allowed_keys & provided_keys

        # Check if some provided keys are incorrect
        if len(common_keys) < len(provided_keys):
            incorrect_keys = ", ".join(provided_keys - common_keys)
            logger.warning("The following keys are incorrect: " + incorrect_keys)

        # Update default parameters and all parameters
        params.update({key: kwargs[key] for key in common_keys})
        all_params.update(kwargs)
        return params, all_params


    @classmethod
    def load(cls, path: Union[str, os.PathLike], predictor: Any) -> "Explainer":
        return super().load(path, predictor)

    def reset_predictor(self, predictor: Any) -> None:
        pass

    def save(self, path: Union[str, os.PathLike]) -> None:
        super().save(path)

    def fit(self, x: np.ndarray, ) -> "Explainer":
        """
        Fit the model agnostic counterfactual generator.

        Parameters
        ----------
        x
            Training data array.

        Returns
        -------
        self
            The explainer itself.
        """
        # Define replay buffer (this will deal only with numpy arrays).
        replay_buff = ReplayBuffer(size=self.params["replay_buffer_size"])

        # Define noise variable.
        noise = NormalActionNoise(mu=0, sigma=self.params["act_noise"])

        # Define data generator.
        data_generator = self.Backend.data_generator(x=x, **self.params)
        data_iter = iter(data_generator)

        for step in tqdm(range(self.params["train_steps"])):
            # Sample training data.
            try:
                data = next(data_iter)
            except:
                data_iter = iter(data_generator)
                data = next(data_iter)

            # Add None condition if condition does not exist.
            if "c" not in data:
                data["c"] = None

            # Compute input embedding.
            z = self.Backend.encode(x=data["x"], **self.params)
            data.update({"z": z})

            # Compute counterfactual embedding.
            z_cf = self.Backend.generate_cf(step=step, **data, **self.params)
            data.update({"z_cf": z_cf})

            # Add noise to the counterfactual embedding.
            z_cf_tilde = self.Backend.add_noise(noise=noise, step=step, **data, **self.params)
            data.update({"z_cf_tilde": z_cf_tilde})

            # Decode counterfactual and apply postprocessing step to x_cf_tilde.
            x_cf = self.Backend.decode(z=data["z_cf"], **self.params)
            x_cf_tilde = self.Backend.decode(z=data["z_cf_tilde"], **self.params)

            for pp_func in self.params["postprocessing_funcs"]:
                # Post-process counterfactual.
                x_cf = pp_func(self.Backend.to_numpy(x_cf),
                               self.Backend.to_numpy(data["x"]),
                               self.Backend.to_numpy(data["c"]))

                # Post-process noised counterfactual.
                x_cf_tilde = pp_func(self.Backend.to_numpy(x_cf_tilde),
                                     self.Backend.to_numpy(data["x"]),
                                     self.Backend.to_numpy(data["c"]))

            data.update({
                "x_cf": x_cf,
                "x_cf_tilde": x_cf_tilde
            })

            # Compute model's prediction on the noised counterfactual
            x_cf_tilde = self.params["ae_inv_preprocessor"](self.Backend.to_numpy(data["x_cf_tilde"]))
            y_m_cf_tilde = self.params["predict_func"](x_cf_tilde)

            # Compute reward.
            r_tilde = self.params["reward_func"](self.Backend.to_numpy(y_m_cf_tilde),
                                                 self.Backend.to_numpy(data["y_t"]))
            data.update({"r_tilde": r_tilde, "y_m_cf_tilde": y_m_cf_tilde})

            # Store experience in the replay buffer.
            data = {key: self.Backend.to_numpy(data[key]) for key in data.keys()}
            replay_buff.append(**data)

            # Call all experience callbacks.
            for exp_cb in self.params["experience_callbacks"]:
                exp_cb(step=step, model=self, sample=data)

            if step % self.params['update_every'] == 0 and step > self.params["update_after"]:
                for i in range(self.params['update_every']):
                    # Sample batch of experience form the replay buffer.
                    sample = replay_buff.sample()
                    if "c" not in sample:
                        sample["c"] = None

                    # Update critic by one-step gradient descent.
                    losses = self.Backend.update_actor_critic(**sample, **self.params)

                    # Convert all losses from tensors to numpy arrays.
                    losses = {key: self.Backend.to_numpy(losses[key]).item() for key in losses.keys()}

                    # Call all train callbacks.
                    for train_cb in self.params["train_callbacks"]:
                        train_cb(step=step, update=i, model=self, sample=sample, losses=losses)

        return self

    def explain(self, x: np.ndarray, y_t: np.ndarray, c: Optional[np.ndarray] = None) -> "Explanation":
        """
        Explains an input instance

        Parameters
        ----------
        x
            Instance to be explained.
        y_t
            Counterfactual target.
        c
            Conditional vector.

        Returns
        -------
        x_cf
            Conditional counterfactual instance.
        """
        # compute models prediction
        y_m = self.params["predict_func"](x)

        # apply autoencoder preprocessing step
        x = self.params["ae_preprocessor"](x)

        # backend specific tasks
        if self.params["backend"] == CounterfactualRLBase.PYTORCH:
            x = torch.tensor(x).to(self.params["device"])
            y_m = torch.tensor(y_m).to(self.params["device"])
            y_t = torch.tensor(y_t).to(self.params["device"])

        # encode instance
        z = self.Backend.encode(x, **self.params)

        # generate counterfactual embedding
        z_cf = self.Backend.generate_cf(z, y_m, y_t, c, **self.params)

        # decode counterfactual
        x_cf = self.Backend.decode(z_cf, **self.params)
        x_cf = self.Backend.to_numpy(x_cf)

        # apply postprocessing functions
        for pp_func in self.params["postprocessing_funcs"]:
            x_cf = pp_func(x_cf, x, c)

        return self.params["ae_inv_preprocessor"](x_cf)


class ExperienceCallback(ABC):
    @abstractmethod
    def __call__(self,
                 step: int,
                 model: CounterfactualRLBase,
                 sample: Dict[str, np.ndarray]) -> None:
        """
        Experience call-back applied after gather an experience.

        Parameters
        ----------
        step
            Current experience step.
        model
            CounterfactualRLBase explainer.
        sample
            Dictionary of sample gathered in an experience. This includes dataset inputs and intermediate results
            obtained during an experience.
        """
        raise NotImplemented


class TrainingCallback(ABC):
    @abstractmethod
    def __call__(self,
                 step: int,
                 update: int,
                 model: CounterfactualRLBase,
                 sample: Dict[str, np.ndarray],
                 losses: Dict[str, float]) -> None:
        """
        Training call-back applied after every training step.

        Parameters
        -----------
        step
            Current experience step.
        update
            Current update. The ration between the number experience steps and the number of training updates is bound to 1.
        model
            CounterfactualRLBase explainer.
        sample
            Dictionary of samples used for an update. This is sampled from the replay buffer.
        losses
            Dictionary of losses.
        """
        raise NotImplementedError