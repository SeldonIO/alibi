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
from alibi.models.pytorch.autoencoder import AE

if TYPE_CHECKING:
    from alibi.explainers.cfrl_base import NormalActionNoise


class PtCounterfactualRLDataset(CounterfactualRLDataset, Dataset):
    """ Pytorch backend datasets. """

    def __init__(self,
                 x: np.ndarray,
                 preprocessor: Callable,
                 predictor: Callable,
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
        predictor
            Prediction function. The classifier function should expect the input in the original format and preprocess
            it internally in the `predictor` if necessary.
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
        self.predictor = predictor
        self.conditional_func = conditional_func
        self.num_classes = num_classes
        self.batch_size = batch_size

        # Infer the classification labels of the input dataset. This is performed in batches.
        self.y_m = PtCounterfactualRLDataset.predict_batches(x=self.x,
                                                             predictor=self.predictor,
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


def get_device() -> torch.device:
    """
    Checks if cuda is available. If available, use cuda by default, else use cpu.

    Returns
    -------
    Device to be used.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """
    Constructs default Adam optimizer.

    Returns
    -------
    Default optimizer.
    """
    return torch.optim.Adam(model.parameters(), lr=lr)


def get_actor(hidden_dim: int, output_dim: int, **kwargs) -> nn.Module:
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


def get_critic(hidden_dim: int):
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


def sparsity_loss(x_hat_cf: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
    return {"sparsity_loss": F.l1_loss(x_hat_cf, x)}


def consistency_loss(z_cf_pred: torch.Tensor, z_cf_tgt: torch.Tensor):
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


def data_generator(x: np.ndarray,
                   ae_preprocessor: Callable,
                   predictor: Callable,
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
    predictor
        Prediction function. The classifier function should expect the input in the original format and preprocess
        it internally in the `predictor` if necessary.
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
    dataset = PtCounterfactualRLDataset(x=x, preprocessor=ae_preprocessor, predictor=predictor,
                                        conditional_func=conditional_func, num_classes=num_classes,
                                        batch_size=batch_size)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=shuffle, drop_last=True)


@torch.no_grad()
def encode(x: torch.Tensor, ae: AE, device: torch.device, **kwargs):
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


@torch.no_grad()
def decode(z: torch.Tensor, ae: AE, device: torch.device, **kwargs):
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


@torch.no_grad()
def generate_cf(z: torch.Tensor,
                y_m: torch.Tensor,
                y_t: torch.Tensor,
                c: Optional[torch.Tensor],
                num_classes: int,
                ae: nn.Module,
                actor: nn.Module,
                device: torch.device,
                **kwargs) -> torch.Tensor:
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
    ae
        Pre-trained autoencoder.
    actor
        Actor network. The model generates the counterfactual embedding.
    device
        Device object to be used.

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
    state = torch.cat(state, dim=1)  # type: ignore

    # Pass the new input to the projection network (actor) to get the counterfactual embedding
    z_cf = actor(state)
    return z_cf


def add_noise(z_cf: torch.Tensor,
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


def update_actor_critic(ae: AE,
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
    device
        Torch device object.

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
    x = torch.tensor(x).float().to(device)                                                                # type: ignore
    x_cf = torch.tensor(x_cf).float().to(device)                                                          # type: ignore
    z = torch.tensor(z).float().to(device)                                                                # type: ignore
    z_cf_tilde = torch.tensor(z_cf_tilde).float().to(device)                                              # type: ignore
    y_m_ohe = F.one_hot(torch.tensor(y_m, dtype=torch.long), num_classes=num_classes).float().to(device)  # type: ignore
    y_t_ohe = F.one_hot(torch.tensor(y_t, dtype=torch.long), num_classes=num_classes).float().to(device)  # type: ignore
    c = torch.tensor(c).float().to(device) if (c is not None) else None                                   # type: ignore
    r_tilde = torch.tensor(r_tilde).float().to(device)                                                    # type: ignore

    # Define state by concatenating the input embedding, the classification label, the target label, and optionally
    # the conditional vector if exists.
    state = [z, y_m_ohe, y_t_ohe] + ([c.float().to(device)] if (c is not None) else [])  # type: ignore
    state = torch.cat(state, dim=1).to(device)                                           # type: ignore

    # Define input for critic, compute q-values and append critic's loss.
    input_critic = torch.cat([state, z_cf_tilde], dim=1).float()  # type: ignore
    output_critic = critic(input_critic).squeeze(1)               # type: ignore
    loss_critic = F.mse_loss(output_critic, r_tilde)              # type: ignore
    losses.update({"loss_critic": loss_critic.item()})

    # Update critic by gradient step.
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()

    # Compute counterfactual embedding.
    z_cf = actor(state)

    # Compute critic's output.
    critic.eval()
    input_critic = torch.cat([state, z_cf], dim=1)  # type: ignore
    output_critic = critic(input_critic)

    # Compute actor's loss.
    loss_actor = -torch.mean(output_critic)
    losses.update({"loss_actor": loss_actor.item()})

    # Decode the output of the actor.
    x_hat_cf = ae.decoder(z_cf)

    # Compute sparsity losses.
    loss_sparsity = sparsity_loss(x_hat_cf, x)
    losses.update(loss_sparsity)

    # Add sparsity loss to the overall actor's loss.
    for key in loss_sparsity.keys():
        loss_actor += coeff_sparsity * loss_sparsity[key]

    # Compute consistency loss.
    z_cf_tgt = encode(x=x_cf, ae=ae, device=device)  # type: ignore
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


def to_numpy(x: Optional[Union[List, np.ndarray, torch.Tensor]]) -> Optional[Union[List, np.ndarray]]:
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
            return [to_numpy(e) for e in x]

        return np.array(x)

    return None


def to_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device, **kwargs) -> Optional[torch.Tensor]:
    """
    Converts tensor to torch.Tensor

    Returns
    -------
    torch.Tensor conversion.
    """
    if x is not None:
        if isinstance(x, torch.Tensor):
            return x.to(device)

        return torch.tensor(x).to(device)

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


def seed(seed: int = 13):
    """
    Sets a seed to ensure reproducibility
    Parameters
    ----------
    seed
        seed to be set
    """

    # torch related
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # others
    np.random.seed(seed)
    random.seed(seed)
