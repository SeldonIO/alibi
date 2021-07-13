import os
import sys
import numpy as np

from tqdm import tqdm
from abc import abstractmethod, ABC
from typing import Union, Any, Callable, Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import tensorflow as tf
import tensorflow.keras as keras

from alibi.api.interfaces import Explainer, Explanation, FitMixin


class NormalActionNoise:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, shape: Tuple[int, ...]):
        return self.mu + self.sigma * np.random.randn(*shape)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayBuffer(object):
    def __init__(self, size=1000):
        self.x = None
        self.y_m, self.y_t = None, None
        self.z, self.z_cf_tilde = None, None
        self.c = None
        self.r = None

        self.idx = 0              # cursor through the buffer
        self.len = 0              # current length of the buffer
        self.size = size          # buffer's maximum capacity

    def append(self,
               x: np.ndarray,
               y_m: np.ndarray,
               y_t: np.ndarray,
               z: np.ndarray,
               z_cf_tilde: np.ndarray,
               c: Optional[np.ndarray],
               r: np.ndarray,
               **kwargs) -> None:
        """
        Add the state buffer and action buffer to the
        replay memory
        """
        # initialize the buffers
        if self.x is None:
            self.batch_size = x.shape[0]

            self.x = np.zeros((self.size * self.batch_size, *x.shape[1:]))
            self.y_m = np.zeros((self.size * self.batch_size, *y_m.shape[1:]))
            self.y_t = np.zeros((self.size * self.batch_size, *y_t.shape[1:]))
            self.z = np.zeros((self.size * self.batch_size, *z.shape[1:]))
            self.z_cf_tilde = np.zeros((self.size * self.batch_size, *z_cf_tilde.shape[1:]))
            self.r = np.zeros((self.size * self.batch_size, *r.shape[1:]))

            if c is not None:
                self.c = np.zeros((self.size * self.batch_size, *c.shape[1:]))

        # increase the length of the buffer if not full
        if self.len < self.size:
            self.len += 1

        start = self.batch_size * self.idx
        self.x[start:start + self.batch_size] = x
        self.y_m[start:start + self.batch_size] = y_m
        self.y_t[start:start + self.batch_size] = y_t
        self.z[start:start + self.batch_size] = z
        self.z_cf_tilde[start:start + self.batch_size] = z_cf_tilde
        self.r[start:start + self.batch_size] = r

        if c is not None:
            self.c[start:start + self.batch_size] = c

        # if the buffer reached its maximum capacity,
        # start replacing old buffers
        self.idx = (self.idx + 1) % self.size

    def sample(self) -> Dict[str, np.ndarray]:
        """
        Sample a batch of data

        Returns
        --------
        Tuple containing a batch of states, a batch of actions and a batch of rewards.
        The batch size is the same as the one input.
        """

        # generate random indices to be sampled
        rand_idx = torch.randint(low=0, high=self.len * self.batch_size, size=(self.batch_size, ))

        # extract data form buffers
        x = self.x[rand_idx]
        y_m = self.y_m[rand_idx]
        y_t = self.y_t[rand_idx]
        z = self.z[rand_idx]
        z_cf_tilde = self.z_cf_tilde[rand_idx]
        c = self.c[rand_idx] if self.c is not None else None
        r = self.r[rand_idx]

        return {
            "x": x,
            "y_m": y_m,
            "y_t": y_t,
            "z": z,
            "z_cf_tilde": z_cf_tilde,
            "c": c,
            "r": r
        }


class DatasetDDPG(Dataset):
    def __init__(self, x: np.ndarray, predict_func: Callable, num_classes: int, batch_size):
        super().__init__()

        self.x = x
        self.predict_func = predict_func
        self.num_classes = num_classes
        self.batch_size = batch_size

        # compute labels
        n_minibatch = int(np.ceil(self.x.shape[0] / self.batch_size))
        self.y_m = np.zeros(self.x.shape[0])

        for i in range(n_minibatch):
            istart, istop = i * self.batch_size, min((i + 1) * self.batch_size, self.x.shape[0])
            self.y_m[istart:istop] = predict_func(self.x[istart:istop])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # generate random target class if no target specified
        # otherwise use the predefined target
        self.num_classes = 4  # TODO: remove this
        y_t = torch.randint(0, self.num_classes, (1,)).item()

        return {
            "x": self.x[idx],
            "y_m": self.y_m[idx],
            "y_t": y_t,
        }


class CounterfactualRLBackend(ABC):
    @staticmethod
    def encode():
        pass

    @staticmethod
    def decode():
        pass

    @staticmethod
    @abstractmethod
    def generate_cf(actor, z, y_m, y_t, c):
        pass

    @staticmethod
    @abstractmethod
    def add_noise(z_cf):
        pass

    @staticmethod
    @abstractmethod
    def update_critic(critic, z, y_m, y_t, c, z_cft, R):
        pass

    @staticmethod
    @abstractmethod
    def update_actor(actor, z, y_m, y_t, c, z_cf, x_cf, postprocessing_func, enc):
        pass


class TFCounterfactualRLBackend(CounterfactualRLBackend):
    @staticmethod
    def generate_cf(actor: keras.Model,
                    z: tf.Tensor,
                    y_m: tf.Tensor,
                    y_t: tf.Tensor,
                    c: Optional[tf.Tensor]) -> tf.Tensor:
        pass

    @staticmethod
    def add_noise(z_CF: tf.Tensor) -> tf.Tensor:
        pass

    @staticmethod
    def update_critic(critic: keras.Model,
                      z: tf.Tensor,
                      y_m: tf.Tensor,
                      y_t: tf.Tensor,
                      c: tf.Tensor,
                      z_cft: tf.Tensor,
                      R: tf.Tensor):
        pass

    @staticmethod
    def update_actor(actor: keras.Model,
                     z: tf.Tensor,
                     y_m: tf.Tensor,
                     y_t: tf.Tensor,
                     c: tf.Tensor,
                     z_cf: tf.Tensor,
                     x_cf: tf.Tensor,
                     postprocessing_func: Callable,
                     enc: keras.Model):
        pass


class PTCounterfactualRLBackend(CounterfactualRLBackend):
    @staticmethod
    @torch.no_grad()
    def encode(x: torch.tensor, ae: nn.Module, device: torch.device):
        ae.eval()
        x = x.float().to(device)
        return ae.encoder(x)

    @staticmethod
    @torch.no_grad()
    def decode(z: torch.Tensor, ae: nn.Module):
        # pass counterfactual embedding through the decoder
        ae.eval()
        x_t = ae.decoder(z)
        return x_t

    @staticmethod
    @torch.no_grad()
    def generate_cf(z: torch.Tensor,
                    y_m: torch.Tensor,
                    y_t: torch.Tensor,
                    c: Optional[torch.Tensor],
                    step: int,
                    exploration_steps: int,
                    num_classes: int,
                    predict_func: Callable,
                    ae: nn.Module,
                    actor: nn.Module,
                    device: torch.device,
                    **kwargs) -> Tuple[torch.Tensor, ...]:
        ae.eval()
        actor.eval()

        # get target class
        y_m = y_m.long().to(device)
        y_t = y_t.long().to(device)

        # transform to one hot model's prediction and the given target
        y_m_ohe = F.one_hot(y_m, num_classes=num_classes).float()
        y_m_ohe = y_m_ohe.to(device)

        y_t_ohe = F.one_hot(y_t, num_classes=num_classes).float()
        y_t_ohe = y_t_ohe.to(device)

        # concatenate z_mean, y_m_ohe, y_t_ohe to create the input representation
        # for the projection network (policy)
        state = [z.view(z.shape[0], -1), y_m_ohe, y_t_ohe] + ([c] if c else [])
        state = torch.cat(state, dim=1)

        # pass new input to the policy pi to get the encoding
        # representation for the given target
        z_cf = torch.tanh(actor(state))
        return z_cf

    @staticmethod
    def add_noise(z_cf: torch.Tensor,
                  noise: NormalActionNoise,
                  step: int,
                  exploration_steps: int,
                  device: torch.device,
                  **kwargs) -> torch.Tensor:
        # construct noise
        eps = torch.tensor(noise(z_cf.shape)).float().to(device)

        if step > exploration_steps:
            z_cf_tilde = z_cf + eps
            z_cf_tilde = torch.clamp(z_cf_tilde, min=-1, max=1)
        else:
            # for the first exploration_steps, the action is sampled from a uniform distribution between
            # [-1, 1] to encourage exploration. After that, the algorithm  returns to the normal DDPG exploration.
            z_cf_tilde = (2 * (torch.rand_like(z_cf) - 0.5)).to(device)

        return z_cf_tilde

    @staticmethod
    def update_actor_critic(ae: nn.Module,
                            critic: nn.Module,
                            actor: nn.Module,
                            optimizer_critic: torch.optim.Optimizer,
                            optimizer_actor: torch.optim.Optimizer,
                            sparsity_loss: Callable,
                            coeff_sparsity: float,
                            num_classes: int,
                            x: np.ndarray,
                            z: np.ndarray,
                            z_cf_tilde: np.ndarray,
                            y_m: np.ndarray,
                            y_t: np.ndarray,
                            c: Optional[np.ndarray],
                            r: np.ndarray,
                            device: torch.device,
                            **kwargs):
        ae.eval()
        actor.train()
        critic.train()

        # define dictionary of losses
        losses: Dict[str, float] = dict()

        # transform data to tensors
        x = torch.tensor(x).float().to(device)
        z = torch.tensor(z.reshape(z.shape[0], -1)).float().to(device)                             # TODO: check the actual size
        z_cf_tilde = torch.tensor(z_cf_tilde.reshape(z_cf_tilde.shape[0], -1)).float().to(device)  # TODO: check the actual size
        y_m_ohe = F.one_hot(torch.tensor(y_m, dtype=torch.long), num_classes=num_classes).float().to(device)
        y_t_ohe = F.one_hot(torch.tensor(y_t, dtype=torch.long), num_classes=num_classes).float().to(device)
        c = torch.tensor(c).float().to(device) if c is not None else None
        r = torch.tensor(r).float().to(device)

        # define state
        state = [z, y_m_ohe, y_t_ohe] + ([c] if c is not None else [])
        state = torch.cat(state, dim=1).to(device)

        # define input for critic and compute q values
        input_critic = torch.cat([state, z_cf_tilde], dim=1).float()
        output_critic = critic(input_critic).squeeze(1)
        loss_critic = F.mse_loss(output_critic, r)
        losses.update({"loss_critic": loss_critic.item()})

        # update critic by gradient step
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

        # compute actor's output
        z_cf = torch.tanh(actor(state))
        input_critic = torch.cat([state, z_cf], dim=1)
        output_critic = critic(input_critic)

        # find the action that maximizes the q-function (critic)
        loss_actor = -torch.mean(output_critic)
        losses.update({"loss_actor": loss_actor.item()})

        # decode the output of the actor
        x_cf = ae.decoder(z_cf)

        # compute sparsity losses
        loss_sparsity = sparsity_loss(x_cf, x)
        losses.update({
            key: loss_sparsity[key].item() for key in loss_sparsity.keys()
        })

        # add sparsity loss to the overall actor loss
        for key in loss_sparsity.keys():
            loss_actor += coeff_sparsity * loss_sparsity[key]

        # update by gradient descent
        optimizer_actor.zero_grad()
        loss_actor.backward()
        optimizer_actor.step()

        # return dictionary of losses for potential logging
        return losses

    @staticmethod
    def to_numpy(x: Optional[Union[np.ndarray, torch.Tensor]]) -> Optional[np.ndarray]:
        if x is not None:
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            raise TypeError(f"Conversion from {type(x)} to {np.ndarray} is not implemented.")
        return None


CounterfactualRLDefaults = {
    "act_noise": 0.1,
    "act_low": -1.0,
    "act_high": 1.0,
    "replay_buffer_size": 1000,
    "batch_size": 128,
    "num_workers": 4,
    "num_classes": 10,
    "exploration_steps": 100,
    "update_every": 1,
    "update_after": 10,
    "backend_flag": "pytorch",
    "train_steps": 100000,
    "coeff_sparsity": 7.5,
    "noise_mu": 0,
    "noise_sigma": 0.1
}


class CounterfactualRL(Explainer, FitMixin):
    def __init__(self,
                 ae: Union[keras.Sequential, nn.Sequential],
                 actor: Union[keras.Sequential, nn.Sequential],
                 critic: Union[keras.Sequential, nn.Sequential],
                 optimizer_actor: Union[keras.optimizers.Optimizer, torch.optim.Optimizer],
                 optimizer_critic: Union[keras.optimizers.Optimizer, torch.optim.Optimizer],
                 backend: str = "pytorch",
                 **kwargs):

        # set backend variable
        self.backend = backend
        self.device = None

        # set auto-encoder
        self.ae = ae

        # set DDPG components
        self.actor = actor
        self.critic = critic
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic

        if self.backend == "pytorch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # send auto-encoder to device
            self.ae.to(self.device)

            # sent actor and critic to device
            self.actor.to(self.device)
            self.critic.to(self.device)

        # set default parameters
        self.params = CounterfactualRLDefaults
        self.params.update(kwargs)

        # Set backend according to the backend_flag.
        #  TODO: check if the packages are installed.
        self.Backend = TFCounterfactualRLBackend if self.params["backend_flag"] == "tensorflow" \
            else PTCounterfactualRLBackend

    def explain(self, X: Any) -> "Explanation":
        pass

    @classmethod
    def load(cls, path: Union[str, os.PathLike], predictor: Any) -> "Explainer":
        return super().load(path, predictor)

    def reset_predictor(self, predictor: Any) -> None:
        pass

    def save(self, path: Union[str, os.PathLike]) -> None:
        super().save(path)

    def fit(self,
            x: np.ndarray,
            predict_func: Callable,
            reward_func: Callable,
            postprocessing_func: Callable,
            sparsity_loss: Callable,
            experience_callbacks: List[Callable] = [],
            train_callbacks: List[Callable] = []) -> "Explainer":
        """
        Fit the model agnostic counterfactual generator.

        Parameters
        ----------
        x
            Training input data array.
        predict_func
            Prediction function. This corresponds to the classifier.
        reward_func
            Reward function.
        experience_callbacks
            List of callbacks that are called after each experience step.
        train_callbacks
            List of callbacks that are called after each training step.

        Returns
        -------
        self
            The explainer itself.
        """
        # Define replay buffer (this will deal only with numpy arrays)
        replay_buff = ReplayBuffer(size=self.params["replay_buffer_size"])

        # define noise variable
        noise = NormalActionNoise(mu=self.params["noise_mu"], sigma=self.params["noise_sigma"])

        # Define data generator
        dataset = DatasetDDPG(x=x, predict_func=predict_func, num_classes=self.params["num_classes"],
                              batch_size=self.params["batch_size"])
        data_generator = DataLoader(dataset=dataset, batch_size=self.params["batch_size"],
                                    num_workers=self.params["num_workers"], drop_last=True)
        data_iter = iter(data_generator)

        for step in tqdm(range(self.params["train_steps"])):
            # sample training data
            try:
                data = data_iter.next()
            except:
                data_iter = iter(data_generator)
                data = data_iter.next()

            # add None condition if condition does not exist
            if "c" not in data:
                data["c"] = None

            # Compute input embedding.
            z = self.Backend.encode(x=data["x"], ae=self.ae, device=self.device)
            data.update({"z": z})
            assert z.min() >= -1 and z.max() <= 1

            # Compute counterfactual embedding.
            z_cf = self.Backend.generate_cf(ae=self.ae, actor=self.actor, predict_func=predict_func, step=step,
                                            device=self.device, **data, **self.params)
            data.update({"z_cf": z_cf})
            assert z_cf.min() >= -1 and z_cf.max() <= 1

            # Add noise to the counterfactual embedding.
            z_cf_tilde = self.Backend.add_noise(noise=noise, step=step, device=self.device, **data, **self.params)
            data.update({"z_cf_tilde": z_cf_tilde})
            assert z_cf_tilde.min() >= -1 and z_cf_tilde.max() <= 1

            # Decode counterfactual and apply postprocessing step to x_cf_tilde.
            x_cf_tilde = self.Backend.decode(ae=self.ae, z=data["z_cf_tilde"])
            assert x_cf_tilde.min() >= 0 and x_cf_tilde.max() <= 1

            x_cf_tilde = postprocessing_func(x_cf_tilde.cpu().numpy(), data["c"])
            data.update({"x_cf_tilde": x_cf_tilde})
            assert x_cf_tilde.min() >= 0 and x_cf_tilde.max() <= 1

            # Compute reward. To compute reward, first we need to compute model's
            # prediction on the counterfactual generated.
            y_m_cf_tilde = predict_func(data["x_cf_tilde"])
            assert y_m_cf_tilde.min() >= 0 and y_m_cf_tilde.max() < self.params["num_classes"]
            r = reward_func(y_m_cf_tilde, data["y_t"].numpy())
            data.update({"r": r, "y_m_cf_tilde": y_m_cf_tilde})

            # Store experience in the replay buffer.
            data = {key: self.Backend.to_numpy(data[key]) for key in data.keys()}
            replay_buff.append(**data)

            # call all experience_callbacks
            for exp_cb in experience_callbacks:
                exp_cb(step=step, model=self, sample=data)

            if step % self.params['update_every'] == 0 and step > self.params["update_after"]:
                for i in range(self.params['update_every']):
                    # Sample batch of experience form the replay buffer.
                    sample = replay_buff.sample()

                    # Update critic by one-step gradient descent.
                    losses = self.Backend.update_actor_critic(ae=self.ae,
                                                              critic=self.critic,
                                                              actor=self.actor,
                                                              optimizer_critic=self.optimizer_critic,
                                                              optimizer_actor=self.optimizer_actor,
                                                              sparsity_loss=sparsity_loss,
                                                              device=self.device,
                                                              **sample,
                                                              **self.params)

                    # call all train_callbacks
                    for train_cb in train_callbacks:
                        train_cb(step=step, update=i, model=self, sample=sample, losses=losses)

        return self