import os
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Any, Callable, Optional, Tuple, Dict, List
from abc import ABC, abstractmethod

from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.models.tensorflow.autoencoder import AE as TensorflowAE
from alibi.models.pytorch.autoencoder import AE as PytorchAE

from alibi.utils.frameworks import has_pytorch, has_tensorflow

if has_pytorch:
    # import pytorch backend
    import alibi.explainers.backends.pytorch.cfrl_base as pytorch_base_backend

if has_tensorflow:
    # import tensorflow backend
    import alibi.explainers.backends.tflow.cfrl_base as tensorflow_base_backend

# define logger
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
        # Initialize the buffers.
        if self.x is None:
            self.batch_size = x.shape[0]

            # Allocate memory.
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

        # Increase the length of the buffer if not full.
        if self.len < self.size:
            self.len += 1

        # Compute the first position where to add most recent experience.
        start = self.batch_size * self.idx

        # Add new data / replace old experience (note that a full batch is added at once).
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

    def sample(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Sample a batch of experience form the replay buffer.

        Returns
        --------
        A batch experience. For a description of the keys/values returned, see parameter descriptions in `append`
        method. The batch size returned is the same as the one passed in the `append`.
        """
        # Generate random indices to be sampled.
        rand_idx = np.random.randint(low=0, high=self.len * self.batch_size, size=(self.batch_size,))

        # Extract data form buffers.
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


DEFAULT_BASE_PARAMS = {
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
    "train_steps": 100000,
    "backend": "tensorflow",
    "ae_preprocessor": lambda x: x,
    "ae_inv_preprocessor": lambda x: x,
    "reward_func": lambda y_pred, y_true: y_pred == y_true,
    "postprocessing_funcs": [],
    "conditional_func": lambda x: None,
    "experience_callbacks": [],
    "train_callbacks": [],
    "actor": None,
    "critic": None,
    "optimizer_actor": None,
    "optimizer_critic": None,
    "actor_hidden_dim": 256,
    "critic_hidden_dim": 256,
}
"""
Default Counterfactual with Reinforcement Learning parameters.

    - ``'act_noise'``: float, standard deviation for the normal noise added to the actor for exploration.

    - ``'act_low'``: float, minimum action value. Each action component takes values between `[act_low, act_high]`.

    - ``'act_high'``: float, maximum action value. Each action component takes values between `[act_low, act_high]`.

    - ``'replay_buffer_size'``: int, dimension of the replay buffer in `batch_size` units. The total memory
    allocated is proportional with the `size` * `batch_size`.

    - ``'batch_size'``: int, training batch size.

    - ``'num_workers'``: int, number of workers used by the data loader if `pytorch` backend is selected.

    - ``'shuffle'``: bool, whether to shuffle the datasets every epoch.

    - ``'num_classes'``: int, number of classes to be considered.

    - ``'latent_dim'``: int, autoencoder latent dimension.

    - ``'exploration_steps'``: int, number of exploration steps. For the firts `exploration_steps`, the
    counterfactual embedding coordinates are sampled uniformly at random from the interval `[act_low, act_high]`.

    - ``'update_every'``: int, number of steps that should elapse between gradient updates. Regardless of the
    waiting steps, the ratio of waiting steps to gradient steps is locked to 1.

    - ``'update_after'``: int, number of steps to wait before start updating the actor and critic. This ensures that
    the replay buffers is full enough for useful updates.

    - ``'backend'``: str, backend to be used: `tensorflow`|`pytorch`. Default `tensorflow`.

    - ``'train_steps'``: int, number of train steps (interactions).

    - ``'ae_preprocessor'``: Callable, autoencoder data preprocessors. Transforms the input data into the format
    expected by the autoencoder. By default, the identity function.

    - ``'ae_inv_preprocessor'``: Callable, autoencoder data inverse preprocessor. Transforms data from the autoencoder
    expected format to the original input format. Before calling the prediction function, the data is inverse
    preprocessed to match the original input format. By default, the identity function.

    - ``'reward_func'``: Callable, element-wise reward function. By default, checks if the counterfactual prediction
    label matches the target label. Note that this is element-wise, so a tensor is expected to be returned.

    - ``'postprocessing_funcs'``: List[Postprocessing], post-processing list of functions. The function are applied in
    the order, from low to high index. Non-differentiable postprocessing can be applied. The function expects as
    arguments `x_cf` - the counterfactual instance, `x` - the original input instance and `c` - the conditional vector,
    and returns the post-processed counterfactual instance `x_cf_pp` which is passed as `x_cf` for the following
    functions. By default, no post-processing is applied (empty list).

    - ``'conditional_func'``: Callable, generates a conditional vector given a input instance. By default, the function
    returns `None` which is equivalent to no conditioning.

    - ``'experience_callbacks'``: List[ExperienceCallback], list of callback function applied at the end of each
    experience step.

    - ``'train_callbacks'``: List[TrainingCallback], list of callback functions applied at the end of each training
    step.

    - ``'actor'``: Optional[keras.Model, torch.nn.Module], actor network.

    - ``'critic;``: Optional[keras.Model, torch.nn.Module], critic network.

    - ``'optimizer_actor'``: Optional[keras.optimizers.Optimizer, torch.optim.Optimizer], actor optimizer.

    - ``'optimizer_critic'``: Optional[keras.optimizer.Optimizer, torch.optim.Optimizer], critic optimizer.

    - ``'actor_hidden_dim'``: int, actor hidden layer dimension.

    - ``'critic_hidden_dim'``: int, critic hidden layer dimension.
"""


class CounterfactualRLBase(Explainer, FitMixin):
    """ Counterfactual Reinforcement Learning Base. """

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"

    def __init__(self,
                 predict_func: Callable,
                 ae,
                 latent_dim: int,
                 coeff_sparsity: float,
                 coeff_consistency: float,
                 num_classes: int,
                 backend: str = "tensorflow",
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        predict_func
            Prediction function. This corresponds to the classifier.
        ae
            Pre-trained autoencoder.
        latent_dim
            Autoencoder latent dimension,
        coeff_sparsity
            Sparsity loss coefficient.
        coeff_consistency
            Consistency loss coefficient.
        backend
            Deep learning backend: `tensorflow`|`pytorch`. Default `tensorflow`.
        """
        # Clean backend flag.
        backend = backend.strip().lower()

        # Check if pytorch/tensorflow backend supported.
        if (backend == CounterfactualRLBase.PYTORCH and not has_pytorch) or \
                (backend == CounterfactualRLBase.TENSORFLOW and not has_tensorflow):
            raise ImportError(f'{backend} not installed. Cannot initialize and run the CounterfactualRL'
                              f' with {backend} backend.')

        # Allow only pytorch and tensorflow.
        elif backend not in [CounterfactualRLBase.PYTORCH, CounterfactualRLBase.TENSORFLOW]:
            raise NotImplementedError(f'{backend} not implemented. Use `tensorflow` or `pytorch` instead.')

        # Select backend.
        self.backend = self._select_backend(backend, **kwargs)

        # Validate arguments.
        self.params, all_params = self._validate_kwargs(predict_func=predict_func,
                                                        ae=ae,
                                                        latent_dim=latent_dim,
                                                        coeff_sparsity=coeff_sparsity,
                                                        coeff_consistency=coeff_consistency,
                                                        num_classes=num_classes,
                                                        backend=backend,
                                                        **kwargs)

        # If pytorch backend, the if GPU available, send everything to GPU
        if self.params["backend"] == CounterfactualRLBase.PYTORCH:
            from alibi.explainers.backends.pytorch.cfrl_base import get_device
            self.params.update({"device": get_device()})

            # Send auto-encoder to device.
            self.params["ae"].to(self.params["device"])

            # Sent actor and critic to device.
            self.params["actor"].to(self.params["device"])
            self.params["critic"].to(self.params["device"])

    def _select_backend(self, backend, **kwargs):
        """
        Selects the backend according to the `backend` flag.

        Parameters
        ---------
        backend
            Deep learning backend: `tensorflow`|`pytorch`. Default `tensorflow`.
        """
        return tensorflow_base_backend if backend == "tensorflow" else pytorch_base_backend

    def _validate_kwargs(self,
                         predict_func: Callable,
                         ae: Union[TensorflowAE, PytorchAE],
                         latent_dim: float,
                         coeff_sparsity: float,
                         coeff_consistency: float,
                         num_classes: int,
                         backend: str,
                         **kwargs):
        """
        Validates arguments.

        Parameters
        ----------
        predict_func.
            Prediction function. This corresponds to the classifier.
        ae
            Pre-trained autoencoder.
        latent_dim
            Autoencoder latent dimension.
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
        params = deepcopy(DEFAULT_BASE_PARAMS)

        # Update parameters with mandatory arguments
        params.update({
            "ae": ae,
            "latent_dim": latent_dim,
            "predict_func": predict_func,
            "coeff_sparsity": coeff_sparsity,
            "coeff_consistency": coeff_consistency,
            "num_classes": num_classes,
            "backend": backend,
        })

        # Add actor if not user-specified.
        not_specified = {"actor": False, "critic": False}
        if "actor" not in kwargs:
            not_specified["actor"] = True
            params["actor"] = self.backend.get_actor(hidden_dim=params["actor_hidden_dim"],
                                                     output_dim=params["latent_dim"])

        if "critic" not in kwargs:
            not_specified["critic"] = True
            params["critic"] = self.backend.get_critic(hidden_dim=params["critic_hidden_dim"])

        # Add optimizers if not user-specified.
        optimizers = ["optimizer_actor", "optimizer_critic"]

        for optim in optimizers:
            # extract model in question
            model_name = optim.split("_")[1]
            model = params[model_name]

            # If the optimizer is user-specified, just update the params
            if optim in kwargs:
                params.update({optim: kwargs[optim]})
                if self.params["backend"] == CounterfactualRLBase.PYTORCH and not_specified[model_name]:
                    raise ValueError(f"Can not specify {optim} when {model_name} not specified for pytorch backend.")

            # If the optimizer is not user-specified, it need to be initialized. The initialization is backend specific.
            elif params['backend'] == CounterfactualRLBase.TENSORFLOW:
                params.update({optim: self.backend.get_optimizer()})
            else:
                params.update({optim: self.backend.get_optimizer(model)})

        # Add sparsity loss if not user-specified.
        params["sparsity_loss"] = self.backend.sparsity_loss if "sparsity_loss" not in kwargs \
            else kwargs["sparsity_loss"]

        # Add consistency loss if not user-specified.
        params["consistency_loss"] = self.backend.consistency_loss if "consistency_loss" not in kwargs \
            else kwargs["consistency_loss"]

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
        data_generator = self.backend.data_generator(x=x, **self.params)
        data_iter = iter(data_generator)

        for step in tqdm(range(self.params["train_steps"])):
            # Sample training data.
            try:
                data = next(data_iter)
            except Exception:
                data_iter = iter(data_generator)
                data = next(data_iter)

            # Add None condition if condition does not exist.
            if "c" not in data:
                data["c"] = None

            # Compute input embedding.
            z = self.backend.encode(x=data["x"], **self.params)
            data.update({"z": z})

            # Compute counterfactual embedding.
            z_cf = self.backend.generate_cf(step=step, **data, **self.params)
            data.update({"z_cf": z_cf})

            # Add noise to the counterfactual embedding.
            z_cf_tilde = self.backend.add_noise(noise=noise, step=step, **data, **self.params)
            data.update({"z_cf_tilde": z_cf_tilde})

            # Decode counterfactual and apply postprocessing step to x_cf_tilde.
            x_cf = self.backend.decode(z=data["z_cf"], **self.params)
            x_cf_tilde = self.backend.decode(z=data["z_cf_tilde"], **self.params)

            for pp_func in self.params["postprocessing_funcs"]:
                # Post-process counterfactual.
                x_cf = pp_func(self.backend.to_numpy(x_cf),
                               self.backend.to_numpy(data["x"]),
                               self.backend.to_numpy(data["c"]))

                # Post-process noised counterfactual.
                x_cf_tilde = pp_func(self.backend.to_numpy(x_cf_tilde),
                                     self.backend.to_numpy(data["x"]),
                                     self.backend.to_numpy(data["c"]))

            data.update({
                "x_cf": x_cf,
                "x_cf_tilde": x_cf_tilde
            })

            # Compute model's prediction on the noised counterfactual
            x_cf_tilde = self.params["ae_inv_preprocessor"](self.backend.to_numpy(data["x_cf_tilde"]))
            y_m_cf_tilde = self.params["predict_func"](x_cf_tilde)

            # Compute reward.
            r_tilde = self.params["reward_func"](self.backend.to_numpy(y_m_cf_tilde),
                                                 self.backend.to_numpy(data["y_t"]))
            data.update({"r_tilde": r_tilde, "y_m_cf_tilde": y_m_cf_tilde})

            # Store experience in the replay buffer.
            data = {key: self.backend.to_numpy(data[key]) for key in data.keys()}
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
                    losses = self.backend.update_actor_critic(**sample, **self.params)

                    # Convert all losses from tensors to numpy arrays.
                    losses = {key: self.backend.to_numpy(losses[key]).item() for key in losses.keys()}

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
        # Compute models prediction.
        y_m = self.params["predict_func"](x)

        # Apply autoencoder preprocessing step.
        x = self.params["ae_preprocessor"](x)

        # Convert to tensors.
        x = self.backend.to_tensor(x, **self.params)
        y_m = self.backend.to_tensor(y_m, **self.params)
        y_t = self.backend.to_tensor(y_t, **self.params)

        # Encode instance.
        z = self.backend.encode(x, **self.params)

        # Generate counterfactual embedding.
        z_cf = self.backend.generate_cf(z, y_m, y_t, c, **self.params)

        # Decode counterfactual.
        x_cf = self.backend.decode(z_cf, **self.params)
        x_cf = self.backend.to_numpy(x_cf)

        # Apply postprocessing functions.
        for pp_func in self.params["postprocessing_funcs"]:
            x_cf = pp_func(x_cf, x, c)

        # TODO construct explanation
        return self.params["ae_inv_preprocessor"](x_cf)


class Postprocessing(ABC):
    @abstractmethod
    def __call__(self, x_cf: List[np.ndarray], x: np.ndarray, c: np.ndarray) -> Any:
        """
        Post-processing function

        Parameters
        ----------
        x_cf
           List of counterfactual columns.
        x
           Input instance.
        c
           Conditional vector.

        Returns
        -------
        x_cf
            Post-processed x_cf.
        """
        raise NotImplementedError


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
        raise NotImplementedError


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
            Current update. The ration between the number experience steps and the number of training updates is
            bound to 1.
        model
            CounterfactualRLBase explainer.
        sample
            Dictionary of samples used for an update. This is sampled from the replay buffer.
        losses
            Dictionary of losses.
        """
        raise NotImplementedError
