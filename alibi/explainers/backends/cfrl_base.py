import numpy as np
from typing import Callable, Tuple
from abc import ABC, abstractmethod


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


class CounterfactualRLBaseBackend(ABC):
    """ Backend interface. """

    @staticmethod
    @abstractmethod
    def get_optimizer(model, lr):
        pass

    @staticmethod
    @abstractmethod
    def get_actor(hidden_dim, output_dim):
        pass

    @staticmethod
    @abstractmethod
    def get_critic(hidden_dim):
        pass

    @staticmethod
    @abstractmethod
    def sparsity_loss(x_hat_cf, x):
        pass

    @staticmethod
    @abstractmethod
    def data_generator(x, ae_preprocessor, predict_func, conditional_func, num_classes,
                       batch_size, shuffle, num_workers):
        pass

    @staticmethod
    @abstractmethod
    def encode(x, ae):
        pass

    @staticmethod
    @abstractmethod
    def decode(z, ae):
        pass

    @staticmethod
    @abstractmethod
    def generate_cf(z, y_m, y_t, c, num_classes, actor):
        pass

    @staticmethod
    @abstractmethod
    def add_noise(z_cf, noise, act_low, act_high, step, exploration_steps, device):
        pass

    @staticmethod
    @abstractmethod
    def update_actor_critic(ae, critic, actor, optimizer_critic, optimizer_actor, sparsity_loss, consistency_loss,
                            coeff_sparsity, coeff_consistency, num_classes, x, x_cf, z, z_cf_tilde, y_m, y_t,
                            c, r_tilde, device):
        pass

    @staticmethod
    @abstractmethod
    def to_numpy(x):
        pass

    @staticmethod
    @abstractmethod
    def to_tensor(x):
        pass
