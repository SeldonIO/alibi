"""
This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base` for the Pytorch backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import os
import random
import numpy as np
from typing import List, Dict, Callable, Union, Optional, TYPE_CHECKING

from alibi.explainers.backends.cfrl_base import CounterfactualRLDataset
from alibi.models.pytorch.actor_critic import Actor, Critic

if TYPE_CHECKING:
    from alibi.explainers.cfrl_base import NormalActionNoise


class PtCounterfactualRLDataset(CounterfactualRLDataset, Dataset):
    """ Pytorch backend datasets. """

    def __init__(self,
                 X: np.ndarray,
                 preprocessor: Callable,
                 predictor: Callable,
                 conditional_func: Callable,
                 batch_size: int) -> None:
        """
        Constructor.

        Parameters
        ----------
        X
            Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling
            the `preprocessor` function.
        preprocessor
            Preprocessor function. This function correspond to the preprocessing steps applied to
            the auto-encoder model.
        predictor
            Prediction function. The classifier function should expect the input in the original format and preprocess
            it internally in the `predictor` if necessary.
        conditional_func
            Conditional function generator. Given an preprocessed input array, the functions generates a conditional
            array.
        batch_size
            Dimension of the batch used during training. The same batch size is used to infer the classification
            labels of the input dataset.
        """
        super().__init__()

        self.X = X
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.conditional_func = conditional_func
        self.batch_size = batch_size

        # Infer the labels of the input dataset. This is performed in batches.
        self.Y_m = self.predict_batches(X=self.X,
                                        predictor=self.predictor,
                                        batch_size=self.batch_size)

        # Define number of classes for classification & minimum and maximum labels for regression
        if self.Y_m.shape[1] > 1:
            self.num_classes = self.Y_m.shape[1]
        else:
            self.min_m = np.min(self.Y_m)
            self.max_m = np.max(self.Y_m)

        # Preprocess the input data.
        self.X = self.preprocessor(self.X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        if hasattr(self, 'num_classes'):
            # Generate random target for classification task
            tgt = np.random.randint(low=0, high=self.num_classes, size=1)
            Y_t = np.zeros(self.num_classes)
            Y_t[tgt] = 1
        else:
            # Generate random target for regression task.
            Y_t = np.random.uniform(low=self.min_m, high=self.max_m, size=(1, 1))

        data = {
            "X": self.X[idx],
            "Y_m": self.Y_m[idx],
            "Y_t": Y_t,
        }

        # Construct conditional vector.
        C = self.conditional_func(self.X[idx:idx + 1])
        if C is not None:
            data.update({"C": C.reshape(-1)})

        return data


def get_device() -> torch.device:
    """
    Checks if `cuda` is available. If available, use `cuda` by default, else use `cpu`.

    Returns
    -------
    Device to be used.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """
    Constructs default `Adam` optimizer.

    Returns
    -------
    Default optimizer.
    """
    return torch.optim.Adam(model.parameters(), lr=lr)


def get_actor(hidden_dim: int, output_dim: int) -> nn.Module:
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


def get_critic(hidden_dim: int) -> nn.Module:
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


def sparsity_loss(X_hat_cf: torch.Tensor, X: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Default L1 sparsity loss.

    Parameters
    ----------
    X_hat_cf
        Auto-encoder counterfactual reconstruction.
    X
        Input instance

    Returns
    -------
    L1 sparsity loss.
    """
    return {"sparsity_loss": F.l1_loss(X_hat_cf, X)}


def consistency_loss(Z_cf_pred: torch.Tensor, Z_cf_tgt: torch.Tensor):
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
                   encoder_preprocessor: Callable,
                   predictor: Callable,
                   conditional_func: Callable,
                   batch_size: int,
                   shuffle: bool,
                   num_workers: int,
                   **kwargs):
    """
    Constructs a tensorflow data generator.

    Parameters
    ----------
    X
        Array of input instances. The input should NOT be preprocessed as it will be preprocessed when calling
        the `preprocessor` function.
    encoder_preprocessor
        Preprocessor function. This function correspond to the preprocessing steps applied to the
        encoder/auto-encoder model.
    predictor
        Prediction function. The classifier function should expect the input in the original format and preprocess
        it internally in the `predictor` if necessary.
    conditional_func
        Conditional function generator. Given an preprocessed input array, the functions generates a conditional
        array.
    batch_size
        Dimension of the batch used during training. The same batch size is used to infer the classification
        labels of the input dataset.
    shuffle
        Whether to shuffle the dataset each epoch. ``True`` by default.
    num_workers
        Number of worker processes to be created.
    **kwargs
        Other arguments. Not used.
    """
    dataset = PtCounterfactualRLDataset(X=X, preprocessor=encoder_preprocessor, predictor=predictor,
                                        conditional_func=conditional_func, batch_size=batch_size)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=shuffle, drop_last=True)


@torch.no_grad()
def encode(X: torch.Tensor, encoder: nn.Module, device: torch.device, **kwargs):
    """
    Encodes the input tensor.

    Parameters
    ----------
    X
        Input to be encoded.
    encoder
        Pretrained encoder network.
    device
        Device to send data to.

    Returns
    -------
        Input encoding.
    """
    encoder.eval()
    return encoder(X.float().to(device))


@torch.no_grad()
def decode(Z: torch.Tensor, decoder: nn.Module, device: torch.device, **kwargs):
    """
    Decodes an embedding tensor.

    Parameters
    ----------
    Z
        Embedding tensor to be decoded.
    decoder
        Pretrained decoder network.
    device
        Device to sent data to.

    Returns
    -------
    Embedding tensor decoding.
    """
    decoder.eval()
    return decoder(Z.float().to(device))


@torch.no_grad()
def generate_cf(Z: torch.Tensor,
                Y_m: torch.Tensor,
                Y_t: torch.Tensor,
                C: Optional[torch.Tensor],
                encoder: nn.Module,
                decoder: nn.Module,
                actor: nn.Module,
                device: torch.device,
                **kwargs) -> torch.Tensor:
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
    encoder
        Pretrained encoder network.
    decoder
        Pretrained decoder network.
    actor
        Actor network. The model generates the counterfactual embedding.
    device
        Device object to be used.

    Returns
    -------
    Z_cf
        Counterfactual embedding.
    """
    # Set autoencoder and actor to evaluation mode.
    encoder.eval()
    decoder.eval()
    actor.eval()

    # Send labels, targets and conditional vector to device.
    Y_m = Y_m.float().to(device)
    Y_t = Y_t.float().to(device)
    C = C.float().to(device) if (C is not None) else None

    # Concatenate Z_mean, Y_m_ohe, Y_t_ohe to create the input representation for the projection network (actor).
    state = [Z, Y_m, Y_t] + ([C] if (C is not None) else [])
    state = torch.cat(state, dim=1)  # type: ignore

    # Pass the new input to the projection network (actor) to get the counterfactual embedding
    Z_cf = actor(state)
    return Z_cf


def add_noise(Z_cf: torch.Tensor,
              noise: 'NormalActionNoise',
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
    Z_cf
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
       Number of exploration steps. For the first `exploration_steps`, the noised counterfactual embedding
       is sampled uniformly at random.
    device
        Device to send data to.

    Returns
    -------
    Z_cf_tilde
       Noised counterfactual embedding.
    """
    # Generate noise.
    eps = torch.tensor(noise(Z_cf.shape)).float().to(device)

    if step > exploration_steps:
        Z_cf_tilde = Z_cf + eps
        Z_cf_tilde = torch.clamp(Z_cf_tilde, min=act_low, max=act_high)
    else:
        # For the first exploration_steps, the action is sampled from a uniform distribution between
        # [act_low, act_high] to encourage exploration. After that, the algorithm returns to the normal exploration.
        Z_cf_tilde = (act_low + (act_high - act_low) * torch.rand_like(Z_cf)).to(device)

    return Z_cf_tilde


def update_actor_critic(encoder: nn.Module,
                        decoder: nn.Module,
                        critic: nn.Module,
                        actor: nn.Module,
                        optimizer_critic: torch.optim.Optimizer,
                        optimizer_actor: torch.optim.Optimizer,
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
                        device: torch.device,
                        **kwargs):
    """
    Training step. Updates actor and critic networks including additional losses.

    Parameters
    ----------
    encoder
        Pretrained encoder network.
    decoder
        Pretrained decoder network.
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
    device
        Torch device object.
    **kwargs
        Other arguments. Not used.

    Returns
    -------
    Dictionary of losses.
    """
    # Set autoencoder to evaluation mode.
    encoder.eval()
    decoder.eval()

    # Set actor and critic to training mode.
    actor.train()
    critic.train()

    # Define dictionary of losses.
    losses: Dict[str, float] = dict()

    # Transform data to tensors and it device
    X = torch.tensor(X).float().to(device)  # type: ignore
    X_cf = torch.tensor(X_cf).float().to(device)  # type: ignore
    Z = torch.tensor(Z).float().to(device)  # type: ignore
    Z_cf_tilde = torch.tensor(Z_cf_tilde).float().to(device)  # type: ignore
    Y_m = torch.tensor(Y_m).float().to(device)  # type: ignore
    Y_t = torch.tensor(Y_t).float().to(device)  # type: ignore
    C = torch.tensor(C).float().to(device) if (C is not None) else None  # type: ignore
    R_tilde = torch.tensor(R_tilde).float().to(device)  # type: ignore

    # Define state by concatenating the input embedding, the classification label, the target label, and optionally
    # the conditional vector if exists.
    state = [Z, Y_m, Y_t] + ([C] if (C is not None) else [])  # type: ignore
    state = torch.cat(state, dim=1).to(device)  # type: ignore

    # Define input for critic, compute q-values and append critic's loss.
    input_critic = torch.cat([state, Z_cf_tilde], dim=1).float()  # type: ignore
    output_critic = critic(input_critic).squeeze(1)  # type: ignore
    loss_critic = F.mse_loss(output_critic, R_tilde)  # type: ignore
    losses.update({"critic_loss": loss_critic.item()})

    # Update critic by gradient step.
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()

    # Compute counterfactual embedding.
    Z_cf = actor(state)

    # Compute critic's output.
    critic.eval()
    input_critic = torch.cat([state, Z_cf], dim=1)  # type: ignore
    output_critic = critic(input_critic)

    # Compute actor's loss.
    loss_actor = -torch.mean(output_critic)
    losses.update({"actor_loss": loss_actor.item()})

    # Decode the output of the actor.
    X_hat_cf = decoder(Z_cf)

    # Compute sparsity losses.
    loss_sparsity = sparsity_loss(X_hat_cf, X)
    losses.update(loss_sparsity)

    # Add sparsity loss to the overall actor's loss.
    for key in loss_sparsity.keys():
        loss_actor += coeff_sparsity * loss_sparsity[key]

    # Compute consistency loss.
    Z_cf_tgt = encode(X=X_cf, encoder=encoder, device=device)  # type: ignore
    loss_consistency = consistency_loss(Z_cf_pred=Z_cf, Z_cf_tgt=Z_cf_tgt)
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


def to_numpy(X: Optional[Union[List, np.ndarray, torch.Tensor]]) -> Optional[Union[List, np.ndarray]]:
    """
    Converts given tensor to `numpy` array.

    Parameters
    ----------
    X
        Input tensor to be converted to `numpy` array.

    Returns
    -------
    `Numpy` representation of the input tensor.
    """
    if X is not None:
        if isinstance(X, np.ndarray):
            return X

        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()

        if isinstance(X, list):
            return [to_numpy(e) for e in X]

        return np.array(X)

    return None


def to_tensor(X: Union[np.ndarray, torch.Tensor], device: torch.device, **kwargs) -> Optional[torch.Tensor]:
    """
    Converts tensor to `torch.Tensor`

    Returns
    -------
    `torch.Tensor` conversion.
    """
    if X is not None:
        if isinstance(X, torch.Tensor):
            return X.to(device)

        return torch.tensor(X).to(device)

    return None


def save_model(path: Union[str, os.PathLike], model: nn.Module) -> None:
    """
    Saves a model and its optimizer.

    Parameters
    ----------
    path
        Path to the saving location.
    model
        Model to be saved.
    """
    torch.save(model, path)


def load_model(path: Union[str, os.PathLike]) -> nn.Module:
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
    model = torch.load(path)
    model.eval()
    return model


def set_seed(seed: int = 13):
    """
    Sets a seed to ensure reproducibility.

    Parameters
    ----------
    seed
        Seed to be set.
    """
    # Others
    np.random.seed(seed)
    random.seed(seed)

    # Torch related
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
